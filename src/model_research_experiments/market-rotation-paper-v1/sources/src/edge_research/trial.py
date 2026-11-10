"""Fixed independent-market and NBA joint-rotation paper challengers."""
from __future__ import annotations
from collections import Counter
from contextlib import closing
from datetime import datetime,timedelta,timezone
import hashlib
import json
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nba_props import research_data as data,research_models as base,research_trial as frozen
from nfl_pipeline import experiment_registry as registry
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json
from edge_research import market,rotation,rotation_study,feeds

ROOT=data.ROOT

NAME='market-rotation-paper-v1'
STORE=ROOT/'src/model_research_experiments'
WEIGHTS=ROOT/'reports/market_rotation_weights.json'
LOCK=86731443
PROTOCOL=dict(sports=list(feeds.SUPPORTED),book='fanduel',fit_before='2026-10-28',calibrate_before='2026-11-01',
    historical_evaluation_before='2026-11-04',prospective='after immutable registration only',
    primary='blend_production vs consensus alone',secondary='blend_rotation vs consensus alone',
    diagnostics=['production','rotation_mean','rotation_mixture'],
    fitting='separate sport and market; earlier data only; never refit registered weights',
    unit='one paired primary line per event/player/market; correlations clustered by event and date',
    missing_references='save observation; decision can qualify at a later observation',
    missing_model='save first eligible consensus decision with unavailable model; never backfill',
    forecast_age_seconds=1800,capture_delay_seconds=300,
    execution='manual, public feed, and actual wagers remain separate',
    review=dict(min_days=21,min_games=20,min_checked_closes=50,exact_close_coverage=.9,
        manual_check_coverage=.9,availability=.8,positive_game_and_day_95_bounds=True),
    betting_authorized=False,automatic_promotion=False)
SCHEMA="""
CREATE TABLE IF NOT EXISTS bets.edge_rotation_forecasts(
 id bigserial PRIMARY KEY,experiment_sha256 text NOT NULL,game_id text NOT NULL,player_id bigint NOT NULL,
 stat text NOT NULL,forecast_at timestamptz NOT NULL,tip_utc timestamptz NOT NULL,payload jsonb NOT NULL,
 UNIQUE(experiment_sha256,game_id,player_id,stat,forecast_at));
CREATE INDEX IF NOT EXISTS edge_rotation_lookup ON bets.edge_rotation_forecasts(experiment_sha256,game_id,stat,forecast_at);
CREATE TABLE IF NOT EXISTS bets.edge_market_observations(
 observation_key text PRIMARY KEY,experiment_sha256 text NOT NULL,sport text NOT NULL,event_id text NOT NULL,
 observed_at timestamptz NOT NULL,captured_at timestamptz NOT NULL,payload jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS bets.edge_market_decisions(
 id bigserial PRIMARY KEY,experiment_sha256 text NOT NULL,sport text NOT NULL,event_id text NOT NULL,
 market text NOT NULL,player_norm text NOT NULL,captured_at timestamptz NOT NULL,payload jsonb NOT NULL,
 UNIQUE(experiment_sha256,sport,event_id,market,player_norm));
DO $$ DECLARE t text; BEGIN
 FOREACH t IN ARRAY ARRAY['edge_rotation_forecasts','edge_market_observations','edge_market_decisions'] LOOP
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid=('bets.'||t)::regclass AND tgname='edge_research_immutable') THEN
 EXECUTE format('CREATE TRIGGER edge_research_immutable BEFORE UPDATE OR DELETE ON bets.%I FOR EACH ROW EXECUTE FUNCTION nbaprops.research_immutable()',t);
 END IF; END LOOP; END $$;
"""


def init(conn):
    data.init(conn)
    with conn.cursor() as cur:cur.execute(SCHEMA)
    conn.commit()


def configuration():
    parent=frozen.active()
    from nba_props import fanduel_live
    observer=fanduel_live.active()
    return dict(protocol=PROTOCOL,market=market.RULES,rotation=rotation.SPEC,
        workload_parent=parent['sha256'],feed_observer=observer['sha256'],minimum_live_roster_players=8)


def active():
    doc=registry.check_experiment(NAME,root=ROOT,store=STORE)
    if doc['config']!=configuration():raise ValueError('Market/rotation experiment drift')
    return doc


def register():
    weights=json.loads(WEIGHTS.read_text(encoding='utf-8'))
    if weights['protocol']!=PROTOCOL:raise ValueError('Weight protocol mismatch')
    manifest=json.loads(rotation_study.MANIFEST.read_text(encoding='utf-8'))
    if (manifest['specification']!=rotation.SPEC or manifest['artifact_sha256']!=hashlib.sha256(rotation_study.ARTIFACT.read_bytes()).hexdigest()
        or manifest['parent_artifact_sha256']!=hashlib.sha256(frozen.ARTIFACT.read_bytes()).hexdigest()):
        raise ValueError('Rotation artifact/manifest mismatch')
    files={p.stem:p for p in (ROOT/'src/edge_research').glob('*.py') if not p.name.startswith('test_')}
    files.update(rotation_artifact=rotation_study.ARTIFACT,rotation_manifest=rotation_study.MANIFEST,weights=WEIGHTS,
        parent_registration=frozen.STORE/frozen.NAME/'registration.json',parent_weights=frozen.ARTIFACT,
        ncaa_projection=ROOT/'src/ncaaf_pipeline/projections.py',ncaa_ratings=ROOT/'src/ncaaf_pipeline/ratings.py')
    return registry.register_experiment(NAME,configuration(),files,root=ROOT,store=STORE)


def artifact():
    return dict(rotation=joblib.load(rotation_study.ARTIFACT),parent=joblib.load(frozen.ARTIFACT),weights=json.loads(WEIGHTS.read_text(encoding='utf-8')))


def project(conn,doc,art,save=True):
    start=data.now();day=start.astimezone(ZoneInfo('America/New_York')).date()
    games=data.rows(conn,"""SELECT DISTINCT ON(game_id) game_id,home_team,away_team,game_date,tip_utc FROM nbaprops.game_projections
        WHERE game_date=%s AND run_at_utc<=%s AND tip_utc>%s ORDER BY game_id,run_at_utc DESC""",(day,start,start))
    if not games:return dict(status='no_upcoming_games',forecasts=0)
    p,t=data.history(conn,start);p=base.attach_past_roles(p,conn,start)
    ps,ts,budgets=rotation_study.states(p,t);targets=[];missing=[]
    for g in games:
        for team,opponent,home in ((g['home_team'],g['away_team'],True),(g['away_team'],g['home_team'],False)):
            roster=data.rows(conn,"""SELECT * FROM nbaprops.lineup_snapshots WHERE game_id=%s AND team=%s
                AND captured_at_utc=(SELECT max(captured_at_utc) FROM nbaprops.lineup_snapshots WHERE game_id=%s AND team=%s
                    AND captured_at_utc<=%s AND captured_at_utc>=%s-interval '6 hours')""",(g['game_id'],team,g['game_id'],team,start,start))
            # The early feed can contain only five expected starters. That is not a rotation.
            if len(roster)<8:missing.append(dict(game_id=g['game_id'],team=team,reason='incomplete_roster_wait_for_bench'));continue
            if any(safe_time(r.get('source_timestamp')) and safe_time(r['source_timestamp'])>start for r in roster):
                missing.append(dict(game_id=g['game_id'],team=team,reason='future_roster_publication'));continue
            for r in roster:
                ctx=data.context(conn,g['game_id'],team,r['player_id'],start)
                # Confirmed absences receive no minutes/opportunities. Their missing playing
                # history must not invalidate the active roster or get imputed as participation.
                if ctx['availability']=='out':continue
                targets.append(dict(game_id=g['game_id'],game_date=g['game_date'],tip_utc=g['tip_utc'],player_id=r['player_id'],
                    player_name=r['player_name'],team=team,opponent=opponent,home=home,availability=ctx['availability'],
                    confirmed_starter=ctx['starter'],minutes_cap=ctx['minutes_cap'],context=ctx))
    if not targets:return dict(status='no_verified_rosters',forecasts=0,excluded=missing)
    f=rotation_study.features(pd.DataFrame(targets),ps,ts,budgets)
    predicted,bad=rotation_study.predict_groups(art['parent']['fit'],f,art['rotation']['ot_probabilities'])
    completed=data.now();count=0
    for r in predicted.to_dict('records'):
        if not safe_time(doc['registered_at'])<=completed<safe_time(r['tip_utc']):continue
        for stat in base.STATS:
            payload=dict(game_id=r['game_id'],player_id=int(r['player_id']),player_name=r['player_name'],team=r['team'],opponent=r['opponent'],
                stat=stat,forecast_at=completed,tip_utc=r['tip_utc'],input_cutoff=start,context=r['context'],
                team_context_verified=r['team_context_verified'],available=r['availability']=='active',
                mu=r['pred_'+stat],means=r['scenario_'+stat].tolist(),weights=r['scenario_weights'].tolist(),
                predicted_minutes=r['pred_minutes'],minutes=r['scenario_minutes'].tolist(),
                shots=r['scenario_fga'].tolist(),three_attempts=r['scenario_fg3a'].tolist(),assists=r['scenario_ast'].tolist(),
                features={k:r[k] for k in base.FEATURES},player_history_date=r['history_date'],team_history_date=r['team_history_date'])
            if not save:count+=1;continue
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO bets.edge_rotation_forecasts(experiment_sha256,game_id,player_id,stat,forecast_at,tip_utc,payload)
                    VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (doc['sha256'],r['game_id'],r['player_id'],stat,completed,r['tip_utc'],psycopg2.extras.Json(frozen.clean(payload))))
                count+=cur.rowcount
    conn.commit()
    result=dict(status='ok' if save else 'preview_only',forecasts=count if save else 0,preview_forecasts=count if not save else None,
        excluded=dict(history_or_capacity=bad,rosters=missing),input_cutoff=start,completed_at=completed)
    atomic_json(ROOT/'reports/nba_rotation_projection_latest.json',frozen.clean(result));return result


def verified_schedule(conn,o):
    at=safe_time(o['decided_at']);tip=safe_time(o['commence_time_utc'])
    if o['sport']=='nba':
        found=data.rows(conn,"""SELECT game_id,tip_utc FROM nbaprops.game_projections WHERE home_team=%s AND away_team=%s
            AND run_at_utc<=%s AND tip_utc BETWEEN %s-interval '15 minutes' AND %s+interval '15 minutes'""",
            (feeds.parent.nba_team(o['home_team']),feeds.parent.nba_team(o['away_team']),at,tip,tip))
    elif o['sport']=='nfl':
        found=data.rows(conn,'SELECT game_id,start_ts_utc tip_utc FROM raw.nfl_games WHERE game_id=%s AND start_ts_utc=%s',(o['game_id'],tip))
    else:
        rows=data.rows(conn,"SELECT game_id,home_team,away_team,start_date tip_utc FROM ncaaf.games WHERE start_date BETWEEN %s-interval '15 minutes' AND %s+interval '15 minutes'",(tip,tip))
        home=feeds.match_school(o['home_team'],[r['home_team'] for r in rows]);away=feeds.match_school(o['away_team'],[r['away_team'] for r in rows])
        found=[r for r in rows if r['home_team']==home and r['away_team']==away]
    if not found or len({str(r['game_id']) for r in found})!=1:return None
    return dict(game_id=str(found[0]['game_id']),tip=min([tip,*[safe_time(r['tip_utc']) for r in found]]))


def rotation_forecast(conn,doc,o,schedule):
    if o['sport']!='nba' or o['market'] not in feeds.evaluation.STAT_MAP:return None
    from nba_props.model import match_name
    found=data.rows(conn,"""SELECT payload FROM bets.edge_rotation_forecasts WHERE experiment_sha256=%s AND game_id=%s AND stat=%s
        AND forecast_at<=%s AND forecast_at>=%s-interval '30 minutes' ORDER BY forecast_at DESC,id DESC""",
        (doc['sha256'],schedule['game_id'],feeds.evaluation.STAT_MAP[o['market']],o['decided_at'],o['decided_at']))
    matches=[r['payload'] for r in found if match_name(r['payload']['player_name'])==match_name(o['player'])]
    if matches and len({r['player_id'] for r in matches})==1:return matches[0]
    return None


def score(o,native,workload,art):
    ref=o['consensus']['probability'];probabilities={'consensus':ref,'production':native.get('probability')}
    participation=o.get('participation_verified',o['market'] in ('spread','total'))
    availability={'consensus':participation,'production':participation and native.get('context_verified',False)}
    if workload:
        stat=feeds.evaluation.STAT_MAP[o['market']];model=art['rotation']
        raw=float(rotation.mixture_over(workload['means'],workload['weights'],o['line'],model['dispersion'][stat]))
        probabilities['rotation_mixture']=float(base.calibrate(raw,model['calibration'][stat]))
        probabilities['rotation_mean']=float(base.calibrate(base.over(workload['mu'],o['line'],model['mean_dispersion'][stat]),model['mean_calibration'][stat]))
        for key in ('rotation_mean','rotation_mixture'):availability[key]=bool(workload['available'] and workload['team_context_verified'])
    else:
        probabilities.update(rotation_mean=None,rotation_mixture=None)
        availability.update(rotation_mean=False,rotation_mixture=False)
    used={}
    for label,source in (('blend_production','production'),('blend_rotation','rotation_mixture')):
        spec=art['weights']['weights'].get(':'.join((o['sport'],o['market'],source)),dict(weight=0.,calibration={'slope':1.,'intercept':0.},status='no_earlier_model_evidence'))
        weight=spec['weight'] if availability[source] and probabilities[source] is not None else 0.
        p=market.blend(ref,probabilities[source],weight)
        probabilities[label]=float(base.calibrate(p,spec['calibration'])) if weight>0 else ref
        availability[label]=participation;used[label]=dict(weight=weight,registered_weight=spec['weight'],status=spec['status'])
    return dict(probabilities=probabilities,selection={k:market.selection(o,p,availability[k]) for k,p in probabilities.items()},
        weights=used,native=native,workload=workload,betting_authorized=False)


def capture(conn,doc,art,sports):
    start=data.now();counts=Counter();after=max(safe_time(doc['registered_at']),start-timedelta(seconds=300))
    for sport in sports:
        for packet in feeds.snapshots(conn,sport,after,start):
            for o in market.offers(packet['event'],sport,packet['at']):
                at=data.now()
                if at>=safe_time(o['commence_time_utc']) or (at-packet['at']).total_seconds()>300:counts['late']+=1;continue
                key=data.digest([doc['sha256'],packet['id'],o['market'],o['player_norm'],o['line']])
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO bets.edge_market_observations VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (key,doc['sha256'],sport,o['event_id'],packet['at'],at,psycopg2.extras.Json(frozen.clean(o))))
                    counts['observations']+=cur.rowcount
                if o['consensus'] is None:counts['missing_consensus']+=1;continue
                existing=data.rows(conn,"""SELECT id FROM bets.edge_market_decisions WHERE experiment_sha256=%s AND sport=%s AND event_id=%s AND market=%s AND player_norm=%s""",
                    (doc['sha256'],sport,o['event_id'],o['market'],o['player_norm']))
                if existing:continue
                schedule=verified_schedule(conn,o)
                if not schedule or at>=schedule['tip']:counts['unverified_or_started']+=1;continue
                native=feeds.native_forecast(conn,o,art['weights']);workload=rotation_forecast(conn,doc,o,schedule)
                o['participation_verified']=participation_verified(conn,o,schedule,native)
                decision=score(o,native,workload,art);captured=data.now()
                if captured>=schedule['tip'] or (captured-packet['at']).total_seconds()>300:counts['late']+=1;continue
                payload=frozen.clean(dict(offer=o,decision=decision,schedule=schedule,source_id=packet['id'],captured_at=captured))
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO bets.edge_market_decisions(experiment_sha256,sport,event_id,market,player_norm,captured_at,payload)
                        VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (doc['sha256'],sport,o['event_id'],o['market'],o['player_norm'],captured,psycopg2.extras.Json(payload)))
                    counts['decisions']+=cur.rowcount
    conn.commit();return dict(counts)


def participation_verified(conn,o,schedule,native):
    if o['market'] in ('spread','total'):return True
    if o['sport']!='nba':return native.get('context_verified') is True
    from nba_props.model import match_name
    records=data.rows(conn,"""SELECT DISTINCT ON(player_id) player_id,player_name,team FROM nbaprops.lineup_snapshots
        WHERE game_id=%s AND captured_at_utc<=%s AND captured_at_utc>=%s-interval '6 hours'
        ORDER BY player_id,captured_at_utc DESC""",(schedule['game_id'],o['decided_at'],o['decided_at']))
    found=[r for r in records if match_name(r['player_name'])==match_name(o['player'])]
    if len(found)!=1:return False
    r=found[0]
    return data.context(conn,schedule['game_id'],r['team'],r['player_id'],safe_time(o['decided_at']))['availability']=='active'
