"""Frozen prospective count-probability challenger on identical FanDuel offers.

Reuses existing model-comparison forecasts and manual price checks. Does not
send alerts, replace an experiment, or authorize wagers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import psycopg2.extras

import model_research as parent
from model_research import ROOT, clean, rows, safe_time, number, atomic_json, registry
from nba_props.count_distribution_research import count_over
from nfl_pipeline import sharp_math
from model_research_report import exact_close, verified_result, calibration, strategy_summary, economics
from sharp_pipeline.nba_experiment import check_evidence, clustered_interval

NAME = 'nba-count-probability-v1'
STORE = ROOT/'src/model_research_experiments'
RULES = dict(markets=['points','rebounds','assists'], sport='nba', book='fanduel',
    selection='same offered side and price as parent; count probability EV > 0',
    opportunity='first parent model-comparison offer, valid prior forecast and participation',
    capture='within 300 seconds of original observation, after registration and before either tip time',
    distribution='negative binomial: variance=max(mu,a*mu+b*mu^2); Poisson at variance=mu',
    uncertainty='paired Brier and opportunity improvement, game and day clusters; all parent NBA gates retained',
    probability_review='positive paired Brier gain lower bounds against both frozen model and market',
    sends_alerts=False, automatic_betting_authorization=False)

SCHEMA = """CREATE TABLE IF NOT EXISTS bets.nba_count_trial(
    experiment_id TEXT NOT NULL, experiment_sha256 TEXT NOT NULL,
    parent_id BIGINT NOT NULL REFERENCES bets.model_research_decisions(id),
    captured_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL,
    PRIMARY KEY(experiment_id,parent_id));
DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='bets.nba_count_trial'::regclass
                AND tgname='nba_count_trial_immutable') THEN
 CREATE TRIGGER nba_count_trial_immutable BEFORE UPDATE OR DELETE ON bets.nba_count_trial
 FOR EACH ROW EXECUTE FUNCTION bets.model_research_immutable();
 END IF;
END $$;
"""


def config(doc):
    return dict(RULES,parent_experiment=doc['experiment_id'],parent_sha256=doc['sha256'])


def register():
    doc=parent.active()
    files={name:ROOT/'src/nba_props'/f'{name}.py'
           for name in ('count_distribution_trial','count_distribution_research')}
    files['parent_registration']=parent.STORE/parent.NAME/'registration.json'
    return registry.register_experiment(NAME,config(doc),files,root=ROOT,store=STORE)


def active():
    doc=registry.check_experiment(NAME,root=ROOT,store=STORE)
    if doc['config']!=config(parent.active()):raise ValueError('Count trial parent drift')
    return doc


def score(saved,doc,now):
    p=saved['payload'];offer=p['offer'];original=p['decision'];forecast=original.get('forecast') or {}
    start=safe_time(offer.get('commence_time_utc'));observed=safe_time(offer.get('decided_at'))
    captured=safe_time(saved.get('captured_at'))
    model_tip=safe_time(forecast.get('tip_utc'))
    if (saved.get('experiment_id')!=doc['config']['parent_experiment']
        or saved.get('experiment_sha256')!=doc['config']['parent_sha256']
        or offer.get('sport')!='nba' or offer.get('market') not in RULES['markets']
        or not captured or not observed or not start or captured>now
        or observed<safe_time(doc['registered_at'])
        or not 0<=(now-observed).total_seconds()<=300 or now>=start or (model_tip and now>=model_tip)
        or not parent.eligible(offer,doc['registered_at'],now)
        or original.get('reason') not in {'model_positive_ev','model_nonpositive_ev'}):return None
    if parent.assess(offer,forecast)['reason'] not in {'model_positive_ev','model_nonpositive_ev'}:return None
    over=float(count_over(float(forecast['mu']),float(offer['line']),float(forecast['disp_a']),float(forecast['disp_b'])))
    probability=over if offer['side']=='over' else 1-over
    ev=sharp_math.ev(probability,offer['price'])
    return clean(dict(offer=offer,forecast=forecast,parent_id=saved['id'],model_probability=probability,
        model_ev=ev,selected=ev>0,frozen_probability=original['model_probability'],
        frozen_selected=original['selected'],betting_authorized=False))


def capture(conn,doc):
    now=datetime.now(timezone.utc)
    saved=rows(conn,"""SELECT * FROM bets.model_research_decisions WHERE experiment_id=%s
        AND experiment_sha256=%s AND sport='nba' AND captured_at>=%s ORDER BY id""",
        (doc['config']['parent_experiment'],doc['config']['parent_sha256'],now-timedelta(seconds=300)))
    added=0
    for r in saved:
        at=datetime.now(timezone.utc);decision=score(r,doc,at)
        if decision is None:continue
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO bets.nba_count_trial
                (experiment_id,experiment_sha256,parent_id,captured_at,payload) VALUES(%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING""",(NAME,doc['sha256'],r['id'],at,psycopg2.extras.Json(decision)))
            added+=cur.rowcount
    conn.commit()
    return added


def summarize(frame,days):
    if frame.empty:return dict(captured_completed=0,status='collecting',betting_authorized=False)
    frozen=frame.copy();frozen['fair_probability']=frozen.frozen_probability
    versus_frozen=calibration(frozen);versus_market=calibration(frame)
    selected=frame[frame.selected]
    summary=strategy_summary(selected,'nba')
    paired=frame.copy()
    paired['incremental_opportunity']=(paired.selected.astype(int)-paired.frozen_selected.astype(int))*paired.opportunity_ev
    paired_intervals={label:clustered_interval(paired,'incremental_opportunity',column)
                      for label,column in (('game','event_id'),('day','day'))}
    probability_ok=all(c.get(k,{}).get('lower_95') is not None and c[k]['lower_95']>0
        for c in (versus_frozen,versus_market) for k in ('gain_by_game','gain_by_day'))
    coverage=float(frame.opportunity_ev.notna().mean())
    incremental=coverage>=.9 and all(v['lower_95'] is not None and v['lower_95']>0 for v in paired_intervals.values())
    return clean(dict(captured_completed=len(frame),selected=summary,
        probability_vs_frozen=versus_frozen,probability_vs_market=versus_market,
        selected_probability_vs_market=calibration(selected),paired_execution_coverage=coverage,
        incremental_opportunity=paired_intervals,economics=economics(frame,days),
        status='review_only' if probability_ok and incremental and summary['status']=='review_only' else 'collecting',
        betting_authorized=False))


def build_report(conn,doc):
    saved=rows(conn,"SELECT * FROM bets.nba_count_trial WHERE experiment_id=%s AND experiment_sha256=%s ORDER BY parent_id",
               (NAME,doc['sha256']))
    now=datetime.now(timezone.utc);data=[];upcoming=0
    for r in saved:
        p=r['payload'];offer=p['offer']
        if safe_time(offer['commence_time_utc'])>now:upcoming+=1;continue
        check_id=offer.get('parent_check_id')
        if check_id:
            checks=rows(conn,"""SELECT a.delivery_status,a.sent_at,c.checked_at,c.recorded_at,c.status AS check_status,
                c.displayed_line,c.displayed_price FROM bets.nba_paper_price_alerts a
                LEFT JOIN bets.nba_paper_price_checks c USING(alert_id) WHERE a.alert_id=%s""",(check_id,))
        else:
            checks=rows(conn,"""SELECT d.delivery_status,d.sent_at,c.checked_at,c.recorded_at,c.status AS check_status,
                c.displayed_line,c.displayed_price FROM bets.model_research_delivery d
                LEFT JOIN bets.model_research_checks c USING(id) WHERE d.id=%s""",(r['parent_id'],))
        close=exact_close(conn,offer)
        row=dict(offer,**(checks[0] if checks else {}),close_method=close.get('method'),
                 close_probability=close.get('probability'),close_ev=close.get('ev'),evidence_source='manual_live')
        result=verified_result(conn,offer,p['forecast'])
        from sharp_pipeline.outcomes import profit
        data.append(dict(row,**check_evidence(row),result=result,paper_profit=profit(result,offer['price']),
            selected=p['selected'],frozen_selected=p['frozen_selected'],model_probability=p['model_probability'],
            frozen_probability=p['frozen_probability'],
            day=safe_time(offer['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date()))
    days=max(1,(now-safe_time(doc['registered_at'])).total_seconds()/86400)
    result=clean(dict(experiment_id=NAME,sha256=doc['sha256'],registered_at=doc['registered_at'],
        built_at=now,captured=len(saved),upcoming=upcoming,**summarize(pd.DataFrame(data),days)))
    atomic_json(ROOT/'reports/nba_count_trial_latest.json',result)
    (ROOT/'reports/nba_count_trial_latest.md').write_text('\n'.join([
        '# NBA count-probability prospective comparison','',f'Experiment: {NAME}',
        f"Captured: {len(saved)}; upcoming: {upcoming}; status: {result['status']}.",
        'Same FanDuel offers, saved forecasts and price checks as the frozen model comparison.',
        'Independent append-only challenger; no alert changes or wagering authority.',
        'Paired probability, exact-close, execution and economic evidence is in the JSON.'])+'\n',encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['register','verify','capture','report'])
    args=ap.parse_args()
    doc=register() if args.command=='register' else active()
    result=dict(experiment_id=NAME,sha256=doc['sha256'],betting_authorized=False)
    if args.command in {'capture','report'}:
        with psycopg2.connect(parent.PG_DSN,connect_timeout=10) as conn:
            with conn.cursor() as cur:cur.execute("SET statement_timeout='30s'");cur.execute(SCHEMA)
            conn.commit()
            if args.command=='capture':result['new_captures']=capture(conn,doc)
            result['report']=build_report(conn,doc)
    print(json.dumps(clean(result),indent=2,allow_nan=False))


if __name__=='__main__':main()
