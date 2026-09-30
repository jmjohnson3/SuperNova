"""Prospective tests for independently accepted outputs; never emit bet rows."""
import argparse
from collections import Counter
from datetime import date,datetime,timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine,text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT,active_release,atomic_json
from nfl_pipeline.context_contract import safe_time,validate_evidence
from nfl_pipeline.modeling.challenger_models import player_features,curve_summary
from nfl_pipeline.modeling.component_validation import calibrate_curve
from nfl_pipeline.modeling.scoring_capture import replay
from nfl_pipeline.modeling.team_workload import PregameTeamHistory,COUNTS,YARDS
from nfl_pipeline.modeling.train_accuracy_challengers import mixture_curve


def load_artifact():
    pointer=MODEL_ROOT/'accuracy_components'/'latest.json'
    if not pointer.exists(): return None,None
    manifest=json.loads(pointer.read_text(encoding='utf-8'))
    path=MODEL_ROOT/'accuracy_components'/manifest['run_id']/'models.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']: raise RuntimeError('Component checksum mismatch')
    artifact=joblib.load(path)
    if artifact['production_release']!=active_release()['release_id']: raise RuntimeError('Component production cohort mismatch')
    return artifact,manifest


def capture_team_context(day):
    artifact,_=load_artifact()
    if not artifact or not any(k.startswith('team_') and m['enabled_outputs'] for k,m in artifact['models'].items()):
        return {'status':'no_accepted_team_component'}
    started=datetime.now(timezone.utc)
    with create_engine(PG_DSN).connect() as conn:
        conn.execute(text("SET statement_timeout='90s'"))
        raw=pd.read_sql(text("""SELECT * FROM (
            SELECT p.game_id,p.player_id,p.team_abbr,p.opponent_abbr,p.position,g.game_date_et,
                p.pass_attempts,p.carries,p.targets,p.passing_yards,p.rushing_yards,p.receiving_yards,
                dense_rank() OVER(PARTITION BY p.team_abbr ORDER BY g.game_date_et DESC,p.game_id DESC) AS history_rank
            FROM raw.nfl_player_gamelogs p JOIN raw.nfl_games g USING(game_id)
            WHERE g.status='final' AND g.game_date_et<:day AND p.updated_at_utc<=:cutoff
            ) h WHERE history_rank<=10"""),conn,params={'day':day,'cutoff':started})
        games=pd.read_sql(text("""SELECT game_id,season,week,home_team_abbr,away_team_abbr
            FROM raw.nfl_games WHERE game_date_et=:day AND start_ts_utc>:cutoff"""),conn,params={'day':day,'cutoff':started})
    from nfl_pipeline.game_scope import filter_frame
    games = filter_frame(games)
    from nfl_pipeline.modeling.predict_player_props import _load_player_context
    with psycopg2.connect(PG_DSN) as conn:
        context=_load_player_context(conn,day,started)
    contexts={str(r['player_id']):r for r in context.to_dict('records')}
    state=PregameTeamHistory()
    raw=raw.drop_duplicates(['game_id','team_abbr','player_id']).dropna(subset=list(COUNTS)+list(YARDS))
    for (past_day,game,team),group in raw.sort_values('game_date_et').groupby(['game_date_et','game_id','team_abbr'],sort=True):
        state.update(team,past_day,group.to_dict('records'))
    teams=[]; rosters=[]
    for g in games.itertuples():
        for team,opponent in ((g.home_team_abbr,g.away_team_abbr),(g.away_team_abbr,g.home_team_abbr)):
            t,pool=state.context(g.game_id,team,opponent,day,g.season,g.week)
            teams.append(t)
            for player in pool:
                ctx=contexts.get(str(player['player_id']),{})
                when=safe_time(ctx.get('injury_observed_at'))
                status=ctx.get('report_status')
                if when and when<=started and status is not None:
                    player['injury_out']=float(str(status).lower() in {'out','injured reserve'})
                when=safe_time(ctx.get('depth_observed_at'))
                if when and when<=started: player['depth_rank']=ctx.get('pos_rank')
            rosters.extend(pool)
    at=datetime.now(timezone.utc)
    record=clean({'source_cutoff':started.isoformat(),'captured_at':at.isoformat(),'date':str(day),
                  'teams':teams,'roster':rosters,'contract':'nfl-team-context-v1'})
    path=MODEL_ROOT/'accuracy_components'/'team_context'/str(day)/(at.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_json(path,record)
    return {'status':'captured','teams':len(teams),'roster_rows':len(rosters)}


def snapshot_before(paths,cutoff):
    eligible=[s for s in paths if s.get('contract')=='nfl-team-context-v1'
              and safe_time(s.get('captured_at')) and safe_time(s['captured_at'])<=cutoff
              and safe_time(s.get('source_cutoff')) and safe_time(s['source_cutoff'])<=safe_time(s['captured_at'])]
    return max(eligible,key=lambda s:safe_time(s['captured_at'])) if eligible else None


def empirical_residuals(values,weights,center):
    if np.allclose(weights,1/len(weights)):
        return (np.asarray(values)-center).tolist()
    order=np.argsort(values); v=np.asarray(values)[order]; w=np.asarray(weights)[order]
    cdf=np.cumsum(w/w.sum())
    # Deterministic quadrature; CDF error is at most 1/2001 for weighted mixtures.
    indices=np.searchsorted(cdf,(np.arange(2001)+.5)/2001).clip(0,len(v)-1)
    return (v[indices]-center).tolist()


def full_path(payload,values,weights,projection):
    captured=payload.get('scoring_replay')
    if not captured: raise ValueError('missing_lock_time_scoring_inputs')
    baseline=replay(captured)
    if baseline['side']!=payload['side'] or abs(baseline['probability']-payload['probability'])>1e-10:
        raise ValueError('production_replay_mismatch')
    revised=dict(captured,projection=float(projection),distribution={
        'kind':'empirical_oof_residual','residual_quantiles':empirical_residuals(values,weights,projection)})
    candidate=replay(revised)
    return {'raw_over_probability':candidate['raw_over_probability'],
        'calibrated_over_probability':candidate['probability'] if candidate['side']=='over' else 1-candidate['probability'],
        'candidate_side':candidate['side'],'candidate_probability':candidate['probability'],
        'probability_trace':candidate['probability_trace'],'scoring_stage':'complete_live_path',
        'locked_line':payload['line'],'locked_book':payload['book'],'locked_price':payload['price'],
        'live_p10':candidate['projection_p10'],'live_p90':candidate['projection_p90']}


def score_records(records,artifact,team_snapshots=()):
    results={}; excluded=Counter()
    for name,model in artifact['models'].items():
        outputs=model['enabled_outputs']
        if not outputs: continue
        stat=model['stat']; selected=[]; seen=set()
        for forecast_id,p in records:
            key=(p['game_id'],p['player_id'],p['stat'])
            if p['stat']!=stat or key in seen: continue
            features=p.get('forecast_features')
            cutoff=safe_time(p.get('prediction_context_cutoff_utc'))
            if (not features or p['model_version']!=artifact['production_release'] or
                not cutoff or validate_evidence(p.get('context_evidence'),cutoff) is None):
                excluded['missing_or_invalid_lock_inputs']+=1; continue
            seen.add(key); selected.append((forecast_id,p,dict(features,context_evidence=p['context_evidence'])))
        if not selected: continue
        frame=pd.DataFrame([r[2] for r in selected]); point=np.array([r[1]['projection'] for r in selected])
        valid=np.ones(len(frame),dtype=bool)
        if name=='receiving_conditional':
            values,weights=model['uncertainty'].mixture(player_features(frame,stat),point,np.ones((len(frame),1)))
        elif name=='qb_tail':
            b=model['bundle']; values,weights,comp=mixture_curve(b['model'],b['uncertainty'],frame,stat)
            op=(comp['weights']*comp['opportunities']).sum(axis=1)
            values,weights=b['tail'].transform(values,weights,op)
            values,weights=calibrate_curve(b['calibrator'],frame,stat,values,weights,
                books=[p.get('book') or 'unpriced_proxy' for _,p,_ in selected],
                extra_lines=[[p.get('line',np.nan)] for _,p,_ in selected])
        else:
            b=model['bundle']; center=b['reference'].predict(frame)
            cached={}
            for i,(_,p,_) in enumerate(selected):
                snapshot=snapshot_before(team_snapshots,safe_time(p['prediction_context_cutoff_utc']))
                if not snapshot:
                    valid[i]=False; excluded['missing_pregame_team_snapshot']+=1; continue
                stamp=snapshot['captured_at']
                if stamp not in cached:
                    teams=pd.DataFrame(snapshot['teams']); roster=pd.DataFrame(snapshot['roster'])
                    teams=teams.loc[teams.history_games.ge(3)]
                    roster=roster.merge(teams[['game_id','team_abbr']],on=['game_id','team_abbr'],validate='many_to_one')
                    out=b['team'].predict(teams,roster)
                    cached[stamp]=out.set_index(['game_id','team_abbr','player_id']).projection.to_dict()
                value=cached[stamp].get((p['game_id'],p['team_abbr'],str(p['player_id'])))
                if value is None or not np.isfinite(value):
                    valid[i]=False; excluded['unknown_pregame_roster_member']+=1
                else: center[i]=value
            values,weights=b['uncertainty'].mixture(player_features(frame,stat),center,np.ones((len(frame),1)))
        summary=curve_summary(values,weights); scored=[]
        for i,(forecast_id,p,_) in enumerate(selected):
            if not valid[i]: continue
            changed_mean='expected_mean' in outputs
            row={'forecast_id':forecast_id,'game_id':p['game_id'],'player_id':p['player_id'],'stat':stat,
                'production_release':p['model_version'],'production_projection':p['projection'],
                'challenger_run':artifact['run_id']+'-'+name,'component':name,
                'source_context_cutoff':p['prediction_context_cutoff_utc'],
                'projection_mean':float(summary['mean'][i]) if changed_mean else p['projection'],
                'distribution_mean':float(summary['mean'][i]),'projection_median':float(summary['median'][i]),
                'p10':float(summary['p10'][i]),'p90':float(summary['p90'][i]),
                'enabled_outputs':outputs,'point_forecast_preserved':not changed_mean,'betting_eligible':False}
            if 'probability' in outputs and p.get('line') is not None:
                try:
                    row.update(full_path(p,values[i],weights[i],row['projection_mean']))
                    row['p10']=row.pop('live_p10'); row['p90']=row.pop('live_p90')
                except ValueError as exc:
                    excluded[str(exc)]+=1; continue
            scored.append(row)
        results[name]=scored
    return results,dict(excluded)


def run(day):
    artifact,manifest=load_artifact()
    if artifact is not None and date.fromisoformat(artifact['training_end'])>=day:
        raise RuntimeError('Component trained on forecast date')
    with psycopg2.connect(PG_DSN) as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT p.id,p.forecast_payload FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g USING(game_id) WHERE p.is_current AND p.game_date_et=%s
            AND g.start_ts_utc>NOW() AND p.created_at_utc<g.start_ts_utc AND p.integrity_version='nfl-asof-v2'
            ORDER BY p.created_at_utc,p.id""",(day,))
        records=cur.fetchall()
    from nfl_pipeline.modeling.target_point_capture import capture_records
    point_capture = capture_records(records)
    if artifact is None:
        return {'status':'waiting_for_distribution_components','point_component':point_capture,'production_changed':False}
    snapshots=[json.loads(p.read_text()) for p in (MODEL_ROOT/'accuracy_components'/'team_context'/str(day)).glob('*.json')]
    results,excluded=score_records(records,artifact,snapshots)
    at=datetime.now(timezone.utc); stamp=at.strftime('%Y%m%dT%H%M%S%fZ')
    for component,rows in results.items():
        cohort=artifact['run_id']+'-'+component
        payload=clean({'status':'component_prospective_only','rows':rows,'scored_at':at.isoformat(),
            'training_end':artifact['training_end'],'source_manifest':manifest,'exclusions':excluded})
        atomic_json(MODEL_ROOT/'challengers'/cohort/'prospective'/str(day)/(stamp+'.json'),payload)
    return {'status':'prospective_only','rows':{k:len(v) for k,v in results.items()},'excluded':excluded,
            'point_component':point_capture,'production_changed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--date',required=True)
    parser.add_argument('--capture-team-context',action='store_true'); args=parser.parse_args()
    print(json.dumps((capture_team_context if args.capture_team_context else run)(date.fromisoformat(args.date)),indent=2))
