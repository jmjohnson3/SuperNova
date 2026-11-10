"""Typed reader for the frozen market/rotation ledger; no rescoring or ledger writes.

The registered collector stores timestamps as JSON strings. PostgreSQL interval
arithmetic needs typed timestamps when grading the saved NCAA rows. This separate
report adapter repairs that read path without changing the registered scorer.
"""
from contextlib import closing
from zoneinfo import ZoneInfo
import argparse

import pandas as pd
import psycopg2

from edge_research import trial,feeds,reporting
from nba_props import research_data as data,research_trial as frozen
from nfl_pipeline import experiment_registry as registry,sharp_math
from nfl_pipeline.context_contract import safe_time,number
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.db import PG_DSN

NAME='market-rotation-report-v1'


def registration(create=False):
    parent=trial.active()
    config=dict(parent_experiment=trial.NAME,parent_sha256=parent['sha256'],
        repair='parse saved offer timestamps before official-result SQL interval arithmetic',
        rescoring=False,decision_mutations=False,betting_authorized=False)
    if create:
        return registry.register_experiment(NAME,config,
            dict(reader=trial.ROOT/'src/edge_research_report.py',parent_registration=trial.STORE/trial.NAME/'registration.json'),
            root=trial.ROOT,store=trial.STORE)
    doc=registry.check_experiment(NAME,root=trial.ROOT,store=trial.STORE)
    if doc['config']!=config:raise ValueError('Reporting parent/configuration drift')
    return doc


def typed_offer(saved):
    offer=dict(saved)
    for key in ('decided_at','commence_time_utc'):
        offer[key]=safe_time(saved.get(key))
        if offer[key] is None:raise ValueError('Invalid saved offer timestamp: '+key)
    return offer


def build(conn,doc,reader=None):
    """The frozen report recipe, with typed timestamps at the storage boundary."""
    saved=data.rows(conn,'SELECT id,payload FROM bets.edge_market_decisions WHERE experiment_sha256=%s ORDER BY id',(doc['sha256'],))
    index=feeds.nba_result_index(conn,data.now()) if any(r['payload']['offer']['sport']=='nba' for r in saved) else {}
    records=[];cache={}
    for r in saved:
        payload=r['payload'];o=typed_offer(payload['offer']);d=payload['decision']
        actual=feeds.outcome(conn,o,d['native'].get('forecast'),index)
        close=reporting.exact_close(conn,o,payload['schedule'],cache) if safe_time(payload['schedule']['tip'])<=data.now() else {}
        row=dict(id=r['id'],sport=o['sport'],market=o['market'],event_id=o['event_id'],player=o['player'],
            day=o['commence_time_utc'].astimezone(ZoneInfo('America/New_York')).date(),
            close_probability=close.get('probability'),close_method=close.get('method'),**actual,**d['probabilities'])
        f=d['native'].get('forecast') or {};w=d.get('workload') or {}
        mean=number(f.get('pred_margin' if o['market']=='spread' else 'pred_total')) if o['market'] in ('spread','total') else number(f.get('mu',f.get('projection')))
        row['native_error']=mean-actual['actual'] if mean is not None and actual['actual'] is not None else None
        row['rotation_error']=w['mu']-actual['actual'] if w.get('mu') is not None and actual['actual'] is not None else None
        for label,s in d['selection'].items():
            row[label+'_selected']=s.get('selected',False);row[label+'_execution']=reporting.execution(conn,o,s,close)
            cp=close.get('probability');is_a=s.get('side') in ('home','over')
            row[label+'_close_ev']=sharp_math.ev(cp if is_a else 1-cp,s['price']) if cp is not None and s.get('selected') else None
            y=actual['outcome'];profit=None
            if y is not None and s.get('selected'):
                win=bool(y) if is_a else not bool(y);profit=(s['price']/100 if s['price']>0 else 100/abs(s['price'])) if win else -1.
            row[label+'_paper_profit']=profit
        records.append(row)
    frame=pd.DataFrame(records);groups={}
    if len(frame):
        for (sport,stat),g in frame.groupby(['sport','market']):groups[sport+':'+stat]=reporting.summary(g)
    observation_counts=data.rows(conn,"""SELECT sport,count(*) observations,count(DISTINCT event_id) games,
        count(*) FILTER(WHERE payload->>'reference_status'<>'ok') missing_consensus
        FROM bets.edge_market_observations WHERE experiment_sha256=%s GROUP BY sport""",(doc['sha256'],))
    result=frozen.clean(dict(built_at=data.now(),experiment_id=trial.NAME,sha256=doc['sha256'],registered_at=doc['registered_at'],
        report_reader=NAME,report_reader_sha256=reader['sha256'] if reader else None,
        groups=groups,observations=observation_counts,records=records,betting_authorized=False,automatic_promotion=False,
        execution_note='Public-feed observations do not establish logged-in availability or accepted wagers. Missing manual checks remain unknown.',
        economics=dict(monthly_cost_usd=30,comparison_stake_usd=1,actual_profit=None,projected_monthly_net=None,
            reason='Actual wagers are reconciled separately; no net-profit claim from unknown execution.'),
        unsupported=dict(mlb='Legacy stored lines do not retain both book/market clocks; no live adapter activated.')))
    atomic_json(trial.ROOT/'reports/market_rotation_trial_latest.json',result)
    text=['# Market consensus and joint rotation trial','',f"Registered: {doc['registered_at']}. Captured pairs: {len(records)}.",
        'FanDuel excluded from reference consensus. Separate sport/market evidence; no pooled cross-sport success claim.','',
        '| Group | Pairs | Games | Settled | Exact-close coverage |','|---|---:|---:|---:|---:|']
    for key,g in groups.items():text.append(f"| {key} | {g['decisions']} | {g['games']} | {g['settled']} | {g['exact_close_coverage']:.1%} |")
    text+=['',result['execution_note'],'All variants remain research; review gates never authorize stakes.',
        f'Report reader: {NAME}; saved probabilities and selections are unchanged.']
    (trial.ROOT/'reports/market_rotation_trial_latest.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--register',action='store_true')
    args=ap.parse_args()
    reader=registration(create=args.register)
    if args.register:
        print(NAME,reader['sha256']);return
    doc=trial.active()
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        with conn.cursor() as cur:cur.execute("SET statement_timeout='90s'")
        result=build(conn,doc,reader)
    print(dict(status='ok',report_reader=NAME,pairs=len(result['records']),betting_authorized=False))


if __name__=='__main__':main()
