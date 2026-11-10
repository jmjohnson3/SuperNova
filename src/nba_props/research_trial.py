"""Independent NBA research: verified inputs, held-out calibration and workload forecasts.

No changes to production fits, frozen selections, alerts, manual checks or stakes.
"""
from __future__ import annotations
import argparse
from contextlib import closing
from datetime import datetime,timedelta,timezone
import hashlib
import json
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy import stats

import model_research as parent
from nba_props import research_data as data,research_models as models,research_evaluation as evaluation
from nfl_pipeline import experiment_registry as registry,sharp_math
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.context_contract import safe_time
from sharp_pipeline.nba_experiment import check_evidence,clustered_interval
from model_research_report import exact_close,strategy_summary,calibration

ROOT=data.ROOT
NAME='nba-workload-calibration-v1'
STORE=ROOT/'src/model_research_experiments'
ARTIFACT=ROOT/'src/nba_props/models/research_v1.joblib'
MANIFEST=ROOT/'reports/nba_research_training_manifest.json'
LOCK=86731432
SCHEMA="""
CREATE TABLE IF NOT EXISTS bets.nba_research_forecasts (
 id bigserial PRIMARY KEY,experiment_sha256 text NOT NULL,game_id text NOT NULL,
 player_id bigint NOT NULL,stat text NOT NULL,forecast_at timestamptz NOT NULL,tip_utc timestamptz NOT NULL,
 payload jsonb NOT NULL,UNIQUE(experiment_sha256,game_id,player_id,stat,forecast_at));
CREATE INDEX IF NOT EXISTS nba_research_forecast_lookup ON bets.nba_research_forecasts(experiment_sha256,game_id,player_id,stat,forecast_at);
CREATE TABLE IF NOT EXISTS bets.nba_research_trial (
 id bigserial PRIMARY KEY,experiment_sha256 text NOT NULL,event_id text NOT NULL,
 market text NOT NULL,player_norm text NOT NULL,captured_at timestamptz NOT NULL,payload jsonb NOT NULL,
 UNIQUE(experiment_sha256,event_id,market,player_norm));
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['nba_research_forecasts','nba_research_trial'] LOOP
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid=('bets.'||t)::regclass AND tgname='nba_research_immutable') THEN
 EXECUTE format('CREATE TRIGGER nba_research_immutable BEFORE UPDATE OR DELETE ON bets.%I FOR EACH ROW EXECUTE FUNCTION nbaprops.research_immutable()',t);
 END IF; END LOOP; END $$;
"""


def clean(value):
    # Nullable frames must not serialize NaN as an asserted numerical observation.
    return parent.clean(value)


def init(conn):
    data.init(conn)
    with conn.cursor() as cur:cur.execute(SCHEMA)
    conn.commit()


def acquire_lock(conn,wait_seconds=30):
    deadline=time.monotonic()+wait_seconds
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout='90s'")
        while True:
            cur.execute('SELECT pg_try_advisory_lock(%s)',(LOCK,))
            if cur.fetchone()[0]:return
            if time.monotonic()>=deadline:raise TimeoutError('Research task is still running; this step did not complete')
            time.sleep(.25)


def configuration():
    p=parent.active()
    return dict(models.SPEC,parents=p['config']['parents'],parent_experiment=p['experiment_id'],parent_sha256=p['sha256'],
        feature_columns=models.FEATURES,fit_parameters=models.PARAMS,fit_policy='pinned artifact; no automatic refit',
        max_forecast_age_seconds=1800,
        review='Existing NBA manual execution gates plus positive paired probability gains by game and date; human review only')


def active():
    doc=registry.check_experiment(NAME,root=ROOT,store=STORE)
    if doc['config']!=configuration():raise ValueError('Research configuration drift')
    return doc


def register():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if (manifest.get('specification')!=models.SPEC
        or manifest.get('artifact_sha256')!=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()):
        raise ValueError('Training manifest does not match the artifact and model specification')
    files={n:ROOT/'src/nba_props'/f'{n}.py' for n in ('research_trial','research_models','research_data','research_evaluation')}
    files.update(artifact=ARTIFACT,manifest=MANIFEST,parent_registration=parent.STORE/parent.NAME/'registration.json',
                 parent_report=ROOT/'src/model_research_report.py')
    return registry.register_experiment(NAME,configuration(),files,root=ROOT,store=STORE)


def train(conn):
    if registry.load_experiment(NAME,root=ROOT,store=STORE):raise ValueError('Registered artifact cannot be replaced; use a new challenger version')
    parent.active();asof=data.now();p,t=data.history(conn,asof)
    if p.empty:raise ValueError('Run research input ingestion first')
    p=models.attach_past_roles(p,conn,asof)
    ps,ts=models.states(p,t);f=models.features(p,ps,ts);g=models.game_features(t,ts)
    # Save split rules and corpus fingerprint before viewing candidate results.
    source_ids=sorted(p.source_snapshot.unique())
    manifest=dict(specification=models.SPEC,started_at=asof,corpus_sha256=data.digest(source_ids),
        input_snapshots=source_ids,verified_games=int(p.game_id.nunique()),verified_player_games=len(p),
        historical_lineups='past-game role labels only; never current-game observed features',
        missing_historical_retrieval_times=True,selection_tuned_on_evaluation=False,betting_authorized=False)
    atomic_json(MANIFEST,clean(manifest))
    predicted,gpred=models.chronological_predictions(f,g)
    variance=predicted[predicted.game_date<models.SPEC['dispersion_end']]
    artifact=dict(specification=models.SPEC,dispersion={},calibration={},game_error={},created_at=asof)
    for stat in models.STATS:artifact['dispersion'][stat]=models.fit_dispersion(variance[stat].to_numpy(),variance['pred_'+stat].to_numpy())
    err=(variance.pred_minutes-variance.minutes).dropna()
    artifact['minutes_error']=dict(q10=float(err.quantile(.1)),q90=float(err.quantile(.9)),rows=len(err))
    quotes=evaluation.prop_quotes(conn)
    calibration=predicted[(predicted.game_date>=models.SPEC['dispersion_end'])&(predicted.game_date<models.SPEC['calibration_end'])]
    for stat in models.STATS:
        artifact['calibration'][stat]=models.fit_distribution_calibration(calibration,stat,artifact['dispersion'][stat])
    for stat in ('margin','total'):
        part=gpred[gpred.game_date<models.SPEC['dispersion_end']]
        error=(part['pred_'+stat]-part[stat]).dropna()
        artifact['game_error'][stat]=dict(bias=float(error.mean()),sd=float(error.std()),rows=len(error))
    heldout=predicted[(predicted.game_date>=models.SPEC['evaluation_start'])&(predicted.game_date<models.SPEC['evaluation_end'])]
    gh=gpred[(gpred.game_date>=models.SPEC['evaluation_start'])&(gpred.game_date<models.SPEC['evaluation_end'])]
    ev=evaluation.evaluate(heldout,gh,quotes,evaluation.game_quotes(conn,gh),artifact)
    # Final weights follow the fixed recipe; future evaluation starts after registration.
    cutoff=pd.Timestamp(asof.astimezone(ZoneInfo('America/New_York')).date())-pd.Timedelta(days=2)
    artifact['fit']=models.fit(f[f.game_date<=cutoff],g[g.game_date<=cutoff])
    artifact['training_cutoff']=cutoff.isoformat();artifact['source_snapshots_sha256']=data.digest(source_ids)
    ARTIFACT.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,ARTIFACT)
    manifest.update(completed_at=data.now(),training_cutoff=cutoff,artifact_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        dispersion=artifact['dispersion'],calibration=artifact['calibration'],minutes_error=artifact['minutes_error'],game_error=artifact['game_error'])
    atomic_json(MANIFEST,clean(manifest))
    data.DATA.mkdir(parents=True,exist_ok=True)
    predicted.to_pickle(data.DATA/'development_predictions.pkl');gpred.to_pickle(data.DATA/'development_game_predictions.pkl')
    report=clean(dict(built_at=data.now(),specification=models.SPEC,evaluation=ev,
        calibration=artifact['calibration'],historical_status=models.SPEC['historical_status'],betting_authorized=False))
    atomic_json(ROOT/'reports/nba_research_development_latest.json',report)
    lines=['# NBA workload and probability research','',models.SPEC['historical_status'],
        'Fixed chronological fitting, dispersion, calibration and later evaluation periods. No production changes or betting approval.','',
        '| Forecast | Rows | MAE | Prior rolling baseline MAE | Bias |','|---|---:|---:|---:|---:|']
    for stat,r in ev['stat_forecasts'].items():
        lines.append(f"| {stat} | {r['rows']} | {r['mae']:.3f} | {r['rolling_baseline_mae']} | {r['bias']:+.3f} |")
    lines+=['','| Book / stat | Offers | Market Brier | Raw Brier | Calibrated Brier |','|---|---:|---:|---:|---:|']
    for book,parts in ev['props'].items():
        for stat,r in parts.items():
            m=r['metrics'];lines.append(f"| {book} / {stat} | {r['rows']} | {m['market']['brier']:.5f} | {m['raw']['brier']:.5f} | {m['calibrated']['brier']:.5f} |")
    lines+=['','Calibration uses fixed integer thresholds on unseen outcomes, not invented historical offers.',
        'Threes have distribution calibration and stat-error results, but no historical real-offer validation.',
        'Game forecast errors, paired uncertainty, calibration bins and chronological boundaries are in the JSON.',
        'A future prospective comparison is required, even if reconstructed historical scores improve.']
    (ROOT/'reports/nba_research_development_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def save_forecasts(conn,doc,forecasts):
    count=0
    with conn.cursor() as cur:
        for f in forecasts:
            at=safe_time(f['forecast_at']);tip=safe_time(f['tip_utc'])
            if at<safe_time(doc['registered_at']) or at>=tip:continue
            cur.execute("""INSERT INTO bets.nba_research_forecasts
                (experiment_sha256,game_id,player_id,stat,forecast_at,tip_utc,payload)
                VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (doc['sha256'],f['game_id'],f['player_id'],f['stat'],at,tip,psycopg2.extras.Json(clean(f))))
            count+=cur.rowcount
    conn.commit();return count


def project(conn,doc,artifact):
    start=data.now();day=start.astimezone(ZoneInfo('America/New_York')).date()
    roster=data.rows(conn,"""SELECT DISTINCT ON(game_id,player_id) game_id,player_id,player_name,team,opponent,game_date,tip_utc
        FROM nbaprops.projections WHERE game_date=%s AND run_at_utc<=%s AND tip_utc>%s
        ORDER BY game_id,player_id,run_at_utc DESC""",(day,start,start))
    games=data.rows(conn,"""SELECT DISTINCT ON(game_id) game_id,home_team,away_team,game_date,tip_utc,pred_margin,pred_total
        FROM nbaprops.game_projections WHERE game_date=%s AND run_at_utc<=%s AND tip_utc>%s ORDER BY game_id,run_at_utc DESC""",(day,start,start))
    if not roster and not games:return dict(status='no_upcoming_games',forecasts=0)
    p,t=data.history(conn,start);p=models.attach_past_roles(p,conn,start);ps,ts=models.states(p,t)
    schedule={g['game_id']:g for g in games}
    targets=[]
    for r in roster:
        if r['game_id'] not in schedule:continue
        r=dict(r,home=r['team']==schedule[r['game_id']]['home_team'])
        r['context']=data.context(conn,r['game_id'],r['team'],r['player_id'],start)
        r['confirmed_starter']=r['context']['starter'];r['minutes_cap']=r['context']['minutes_cap']
        targets.append(r)
    forecasts=[]
    if targets:
        f=models.features(pd.DataFrame(targets),ps,ts)
        f=f[f.games_before.ge(5)&f.fg3_pct.notna()]
        if len(f):
            predicted=models.predict(artifact['fit'],f)
            for _,r in predicted.iterrows():
                base=dict(game_id=r.game_id,game_date=r.game_date.date(),player_id=int(r.player_id),player_name=r.player_name,
                    team=r.team,opponent=r.opponent,tip_utc=r.tip_utc,input_cutoff=start,context=r.context,
                    predicted_minutes=float(r.pred_minutes),minutes_if_starter=float(r.minutes_if_starter),minutes_if_bench=float(r.minutes_if_bench),
                    role_probability=float(r.role_probability),minutes_low=max(0,float(r.pred_minutes)-artifact['minutes_error']['q90']),
                    minutes_high=min(float(r.minutes_cap) if pd.notna(r.minutes_cap) else 48,float(r.pred_minutes)-artifact['minutes_error']['q10']),
                    shooting_probability=float(r.fg3_pct),predicted_three_attempts=float(r.pred_fg3a),
                    player_history_date=r.history_date,team_history_date=r.team_history_date,
                    opponent_history_date=r.opponent_history_date,
                    features={k:None if pd.isna(r[k]) else float(r[k]) for k in models.FEATURES},
                    history_source_cutoff=start,available=r.context['availability']=='active')
                for stat in models.STATS:forecasts.append(dict(base,stat=stat,mu=float(r['pred_'+stat])))
    if games:
        team_targets=[]
        for g in games:
            for home in (True,False):team_targets.append(dict(game_id=g['game_id'],game_date=g['game_date'],
                team=g['home_team'] if home else g['away_team'],opponent=g['away_team'] if home else g['home_team'],home=home,pts=np.nan))
        gf=models.game_features(pd.DataFrame(team_targets),ts)
        for stat in ('margin','total'):
            gf['pred_'+stat]=artifact['fit']['games'][stat].predict(gf[models.GAME_FEATURES])-artifact['game_error'][stat]['bias']
        for _,r in gf.iterrows():
            for stat in ('margin','total'):
                forecasts.append(dict(game_id=r.game_id,game_date=r.game_date.date(),player_id=0,stat=stat,mu=float(r['pred_'+stat]),
                    tip_utc=schedule[r.game_id]['tip_utc'],home_team=r.home_team,away_team=r.away_team,input_cutoff=start,
                    pred_home=float((r.pred_total+r.pred_margin)/2),pred_away=float((r.pred_total-r.pred_margin)/2),
                    frozen_mu=schedule[r.game_id]['pred_margin' if stat=='margin' else 'pred_total']))
    # Timestamp completion, never the earlier start of an expensive computation.
    at=data.now()
    for f in forecasts:f['forecast_at']=at
    return dict(status='ok',forecasts=save_forecasts(conn,doc,forecasts),input_cutoff=start,completed_at=at)


def score(offer,forecast,artifact,at):
    result=dict(selected=False,raw_selected=False,reason='missing_prior_forecast',probability=None,raw_probability=None,ev=None,forecast=forecast)
    if not forecast:return result
    made=safe_time(forecast['forecast_at']);decision=safe_time(offer['decided_at']);tip=safe_time(forecast['tip_utc'])
    if not 0<=(decision-made).total_seconds()<=1800 or at>=tip:return dict(result,reason='forecast_late_stale_or_game_started')
    market=offer['market'];line=float(offer['line']);side=offer['side']
    if market in evaluation.STAT_MAP:
        stat=evaluation.STAT_MAP[market]
        raw=float(models.over(forecast['mu'],line,artifact['dispersion'][stat]))
        raw=raw if side=='over' else 1-raw
        p=models.probability(artifact,stat,forecast['mu'],line)
        p=p if side=='over' else 1-p
        eligible=forecast.get('available') is True
    else:
        stat='margin' if market=='spread' else 'total'
        # Spread rows use the offered team's line, while the mean is home margin.
        threshold=-line if side=='home' else line
        p=float(stats.norm.sf(threshold if market=='spread' else line,forecast['mu'],artifact['game_error'][stat]['sd']))
        if side in ('away','under'):p=1-p
        raw=p
        eligible=True
    ev=sharp_math.ev(p,offer['price'])
    return dict(result,selected=bool(eligible and ev>=.03),raw_selected=bool(eligible and sharp_math.ev(raw,offer['price'])>=.03),
                raw_probability=raw,reason='availability_unverified' if not eligible else 'positive_ev' if ev>=.03 else 'insufficient_ev',probability=p,ev=ev)


def capture(conn,doc,artifact):
    at=data.now();count=0
    for o in parent.candidates(conn,'nba',doc,at):
        if o['market'] in evaluation.STAT_MAP and not o.get('parent_check_id'):continue
        if not parent.eligible(o,doc['registered_at'],at):continue
        o=parent.quote_times(conn,o)
        if o is None:continue
        stat=evaluation.STAT_MAP.get(o['market'],'margin' if o['market']=='spread' else 'total')
        forecasts=data.rows(conn,"""SELECT payload FROM bets.nba_research_forecasts WHERE experiment_sha256=%s AND stat=%s
            AND forecast_at<=%s AND forecast_at>=%s-interval '30 minutes' AND tip_utc BETWEEN %s-interval '15 minutes' AND %s+interval '15 minutes'
            ORDER BY forecast_at DESC,id DESC""",(doc['sha256'],stat,o['decided_at'],o['decided_at'],o['commence_time_utc'],o['commence_time_utc']))
        from nba_props.model import match_name
        h,w=parent.nba_team(o['home_team']),parent.nba_team(o['away_team'])
        candidates=[r['payload'] for r in forecasts if
            ((stat in models.STATS and match_name(r['payload']['player_name'])==match_name(o['player'])
              and (r['payload']['team'],r['payload']['opponent']) in ((h,w),(w,h)))
             or (stat not in models.STATS and r['payload']['home_team']==h and r['payload']['away_team']==w))]
        ids={(f['game_id'],f['player_id']) for f in candidates}
        forecast=candidates[0] if len(ids)==1 else None
        frozen=parent.assess(o,parent.forecast_for(conn,o))
        captured=data.now()
        if not parent.eligible(o,doc['registered_at'],captured):continue
        decision=score(o,forecast,artifact,captured)
        payload=clean(dict(offer=o,decision=decision,frozen=frozen,captured_at=captured,betting_authorized=False))
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO bets.nba_research_trial(experiment_sha256,event_id,market,player_norm,captured_at,payload)
                VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",(doc['sha256'],o['event_id'],o['market'],o.get('player_norm') or '',captured,psycopg2.extras.Json(payload)))
            count+=cur.rowcount
    conn.commit();return dict(captured=count)


def manual_check(conn,doc,offer):
    """Reuse only the original offer's manual evidence; never restart its clock."""
    if offer.get('parent_check_id'):
        found=data.rows(conn,"""SELECT a.delivery_status,a.sent_at,c.checked_at,c.recorded_at,c.status check_status,
            c.displayed_line,c.displayed_price,c.evidence_source FROM bets.nba_paper_price_alerts a
            LEFT JOIN bets.nba_paper_price_checks c USING(alert_id) WHERE a.alert_id=%s""",(offer['parent_check_id'],))
        return found[0] if found else {}
    if offer.get('decision_id') is None:return {}
    found=data.rows(conn,"""SELECT d.payload,m.delivery_status,m.sent_at,c.checked_at,c.recorded_at,
        c.status check_status,c.displayed_line,c.displayed_price FROM bets.model_research_decisions d
        LEFT JOIN bets.model_research_delivery m USING(id) LEFT JOIN bets.model_research_checks c USING(id)
        WHERE d.experiment_sha256=%s AND d.sport='nba' AND d.event_id=%s AND d.market=%s
        AND d.payload->'offer'->>'decision_id'=%s""",
        (doc['config']['parent_sha256'],offer['event_id'],offer['market'],str(offer['decision_id'])))
    if len(found)!=1:return {}
    r=found[0];original=r.pop('payload')['offer']
    for key in ('book','event_id','market','side','player_norm'):
        if (original.get(key) or '')!=(offer.get(key) or ''):return {}
    for key in ('line','price'):
        if parent.number(original.get(key))!=parent.number(offer.get(key)):return {}
    for key in ('decided_at','commence_time_utc'):
        if safe_time(original.get(key))!=safe_time(offer.get(key)):return {}
    return dict(r,evidence_source='manual_live')


def build_report(conn,doc):
    saved=data.rows(conn,'SELECT payload FROM bets.nba_research_trial WHERE experiment_sha256=%s ORDER BY id',(doc['sha256'],))
    latest=data.rows(conn,'SELECT DISTINCT ON(game_id) payload FROM nbaprops.research_games WHERE observed_at<=%s ORDER BY game_id,observed_at DESC',(data.now(),))
    results={r['payload']['game_id']:r['payload'] for r in latest};records=[]
    for row in saved:
        p=row['payload'];o=p['offer'];d=p['decision'];f=d.get('forecast') or {};game=results.get(f.get('game_id'))
        frozen=p.get('frozen') or {};ff=frozen.get('forecast') or {}
        r=dict(o,selected=d['selected'],raw_selected=d.get('raw_selected',False),frozen_selected=frozen.get('selected',False),model_probability=d['probability'],
            raw_probability=d.get('raw_probability'),frozen_probability=frozen.get('model_probability'),
            reason=d['reason'],actual=None,result=None)
        if game and o['market'] in evaluation.STAT_MAP:
            people=[v for v in game['players'] if v['player_id']==f['player_id'] and v['team']==f['team'] and v['minutes']>0]
            if len(people)==1:
                actual=people[0][evaluation.STAT_MAP[o['market']]];minutes=people[0]['minutes']
                rate=f['mu']/max(f['predicted_minutes'],1e-9)
                r.update(actual=actual,actual_minutes=minutes,predicted_minutes=f['predicted_minutes'],forecast_error=f['mu']-actual,
                    workload_error=(f['predicted_minutes']-minutes)*rate,rate_error=minutes*rate-actual)
        elif game and o['market'] in ('spread','total'):
            home=next(t for t in game['teams'] if t['home']);away=next(t for t in game['teams'] if not t['home'])
            r['actual']=home['pts']-away['pts'] if o['market']=='spread' else home['pts']+away['pts']
            r['forecast_error']=f['mu']-r['actual']
        if r['actual'] is not None:
            old_mu=ff.get('mu') if o['market'] in evaluation.STAT_MAP else ff.get('pred_margin' if o['market']=='spread' else 'pred_total')
            if old_mu is not None:
                r['frozen_forecast_error']=float(old_mu)-r['actual']
                r['absolute_error_gain']=abs(r['frozen_forecast_error'])-abs(r['forecast_error'])
            actual=r['actual'];threshold=float(o['line'])
            delta=(actual+threshold if o['side']=='home' else -actual+threshold) if o['market']=='spread' else (actual-threshold)*(1 if o['side']=='over' else -1)
            r['result']='win' if delta>0 else 'loss' if delta<0 else 'push'
        from sharp_pipeline.outcomes import profit
        r['paper_profit']=profit(r['result'],o['price']) if r['result'] else None
        r.update(manual_check(conn,doc,o))
        close=exact_close(conn,o);r.update(close_method=close.get('method'),close_probability=close.get('probability'),close_ev=close.get('ev'))
        r.update(check_evidence(r));r['day']=safe_time(o['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date()
        records.append(r)
    groups={}
    for family in ('props','games'):
        part=[r for r in records if (r['market'] in evaluation.STAT_MAP)==(family=='props')]
        groups[family]=dict(captured=len(part),missing_forecasts=sum(r['model_probability'] is None for r in part),selected=sum(r['selected'] for r in part),settled=sum(r['result'] is not None for r in part),status='collecting')
        if part:
            frame=pd.DataFrame(part)
            groups[family]['probability_calibration']=calibration(frame)
            groups[family]['price_control']=strategy_summary(frame,'nba')
            for label,column in (('execution_review','selected'),('raw_selection','raw_selected'),('frozen_selection','frozen_selected')):
                groups[family][label]=strategy_summary(frame[frame[column]],'nba')
            groups[family]['by_market']={market:dict(probability_calibration=calibration(g),
                execution_review=strategy_summary(g[g.selected],'nba')) for market,g in frame.groupby('market')}
        settled=[r for r in part if r['result'] in ('win','loss') and r['model_probability'] is not None]
        if settled:
            frame=pd.DataFrame(settled);y=frame.result.eq('win').astype(float)
            frame['brier_gain']=(frame.fair_probability-y)**2-(frame.model_probability-y)**2
            groups[family].update(brier=float(((frame.model_probability-y)**2).mean()),market_brier=float(((frame.fair_probability-y)**2).mean()),
                paired_gain={k:clustered_interval(frame,'brier_gain',c) for k,c in (('game','event_id'),('day','day'))})
            for label in ('raw','frozen'):
                reference=pd.to_numeric(frame[label+'_probability'],errors='coerce')
                common=frame[reference.notna()].copy()
                if len(common):
                    observed=common.result.eq('win').astype(float)
                    common['gain']=(reference[reference.notna()]-observed)**2-(common.model_probability-observed)**2
                    groups[family]['vs_'+label]=dict(rows=len(common),paired_gain={k:clustered_interval(common,'gain',c) for k,c in (('game','event_id'),('day','day'))})
            if 'absolute_error_gain' in frame:
                groups[family]['stat_accuracy_vs_frozen']={k:clustered_interval(frame,'absolute_error_gain',c) for k,c in (('game','event_id'),('day','day'))}
            if family=='props':
                groups[family]['by_stat']={stat:dict(rows=len(g),brier=float(((g.model_probability-g.result.eq('win'))**2).mean()),
                    mean_abs_forecast_error=float(g.forecast_error.abs().mean()),
                    mean_abs_workload_error=float(g.workload_error.abs().mean()),mean_abs_rate_error=float(g.rate_error.abs().mean())) for stat,g in frame.groupby('market')}
            selected=pd.DataFrame([r for r in part if r['selected']])
            if len(selected):
                sy=selected[selected.result.isin(('win','loss'))]
                groups[family]['selected_probability']=dict(rows=len(sy),brier=float(((sy.model_probability-sy.result.eq('win'))**2).mean()) if len(sy) else None)
    result=clean(dict(built_at=data.now(),experiment_id=NAME,sha256=doc['sha256'],registered_at=doc['registered_at'],
        book='fanduel',reference_probability='Pinnacle exact-line no-vig benchmark; not known truth',
        groups=groups,records=records,betting_authorized=False,automatic_promotion=False,
        note='Manual checks alone count toward existing execution gates. Automated feed checks remain separate. Unknown participation stays pending.'))
    atomic_json(ROOT/'reports/nba_research_trial_latest.json',result)
    (ROOT/'reports/nba_research_trial_latest.md').write_text('\n'.join(['# NBA workload/calibration prospective trial','',
        f"Registered: {doc['registered_at']}. Captured: {len(records)}.",
        'Frozen research only. Existing price, model and count experiments remain unchanged.',
        'Forecasts must precede the original offer. Missing forecasts are retained as misses; no backfill.',
        'Minutes, role, attempts, shooting efficiency, stat means and game predictions are saved separately.',
        'The JSON reports paired probability errors, workload/rate errors, exact closes and manual execution evidence.',
        'No automatic promotion or wagering authority.'])+'\n',encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['ingest','audit','train','register','verify','project','capture','report','cap'])
    ap.add_argument('--years',default=None);ap.add_argument('--game');ap.add_argument('--player',type=int);ap.add_argument('--team')
    ap.add_argument('--minutes',type=float);ap.add_argument('--observed-at');ap.add_argument('--source');ap.add_argument('--note')
    args=ap.parse_args()
    if args.command in ('register','verify'):
        doc=register() if args.command=='register' else active();print(json.dumps(dict(experiment_id=NAME,sha256=doc['sha256'])));return
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        acquire_lock(conn)
        init(conn)
        if args.command=='ingest':
            year=data.now().year if data.now().month>=9 else data.now().year-1
            years=[year] if args.years is None else list(range(int(args.years.split('-')[0]),int(args.years.split('-')[-1])+1))
            result=data.ingest(conn,years)
        elif args.command=='audit':result=data.audit(conn)
        elif args.command=='train':result=train(conn)
        elif args.command=='cap':
            try:
                parsed=datetime.fromisoformat((args.observed_at or '').replace('Z','+00:00'))
                at=safe_time(parsed) if parsed.tzinfo is not None else None
            except ValueError:at=None
            if not all([args.game,args.player,args.team,args.source,args.note,at,args.minutes]) or not 0<args.minutes<=48 or at>data.now():ap.error('Supply game, player, team, verified minutes cap, observed time with offset, source and note')
            with conn.cursor() as cur:
                cur.execute('INSERT INTO nbaprops.research_context(game_id,player_id,team,observed_at,minutes_cap,source,note) VALUES(%s,%s,%s,%s,%s,%s,%s)',
                    (args.game,args.player,args.team,at,args.minutes,args.source,args.note))
            conn.commit();result=dict(status='recorded_context_only')
        else:
            doc=active()
            if args.command=='report':result=build_report(conn,doc)
            else:
                artifact=joblib.load(ARTIFACT)
                result=project(conn,doc,artifact) if args.command=='project' else capture(conn,doc,artifact)
        print(json.dumps(clean(result if args.command not in ('train','report') else dict(status='ok',report=args.command)),indent=2))
        if args.command=='ingest' and any(r.get('error') for r in result):raise SystemExit(1)


if __name__=='__main__':main()
