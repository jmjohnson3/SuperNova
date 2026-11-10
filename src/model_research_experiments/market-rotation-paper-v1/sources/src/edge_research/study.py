"""Fit model weights on earlier observed offers; never manufacture consensus history."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import json

import numpy as np
import pandas as pd

from edge_research import feeds,market,trial
from nba_props import research_data as data,research_models as base,research_trial as frozen
from nfl_pipeline import experiment_registry as registry
from nfl_pipeline.integrity import atomic_json
from sharp_pipeline.nba_experiment import clustered_interval


def game_errors(conn):
    """Independent normal-error fits from forecasts recorded before October 1."""
    sql={
      'nba':"""SELECT DISTINCT ON(p.game_id) p.game_id,p.pred_margin,p.pred_total,
          g.payload FROM nbaprops.game_projections p JOIN nbaprops.research_games g USING(game_id)
          WHERE p.run_at_utc<p.tip_utc AND p.tip_utc<'2026-10-01' ORDER BY p.game_id,p.run_at_utc DESC,g.observed_at DESC""",
      'nfl':"""SELECT DISTINCT ON(p.game_id) p.game_id,p.predicted_home_margin pred_margin,p.predicted_total_points pred_total,
          g.home_score,g.away_score FROM bets.nfl_game_predictions p JOIN raw.nfl_games g USING(game_id)
          WHERE g.status='final' AND p.created_at_utc<g.start_ts_utc AND g.start_ts_utc<'2026-10-01'
          ORDER BY p.game_id,p.created_at_utc DESC""",
      'ncaaf':"""SELECT DISTINCT ON(p.game_id) p.game_id,p.pred_margin,p.pred_total,
          g.home_points home_score,g.away_points away_score FROM ncaaf.projections p JOIN ncaaf.games g USING(game_id)
          WHERE g.completed AND p.generated_at_utc<g.start_date AND g.start_date<'2026-10-01'
          ORDER BY p.game_id,p.generated_at_utc DESC"""}
    errors={};coverage={}
    for sport,query in sql.items():
        found=data.rows(conn,query);residuals={'margin':[],'total':[]}
        for r in found:
            if sport=='nba':
                home=next(t['pts'] for t in r['payload']['teams'] if t['home']);away=next(t['pts'] for t in r['payload']['teams'] if not t['home'])
            else:home,away=r['home_score'],r['away_score']
            if home is None or away is None:continue
            for stat,actual in (('margin',float(home)-float(away)),('total',float(home)+float(away))):
                mu=feeds.number(r['pred_'+stat])
                if mu is not None:residuals[stat].append(mu-actual)
        errors[sport]={};coverage[sport]={}
        for stat,values in residuals.items():
            coverage[sport][stat]=dict(games=len(values),minimum=100,fit_before='2026-10-01')
            if len(values)>=100 and np.std(values,ddof=1)>0:
                errors[sport][stat]=dict(bias=float(np.mean(values)),sd=float(np.std(values,ddof=1)),rows=len(values),fit_before='2026-10-01')
    return errors,coverage


def fit_calibration(frame,weight):
    identity=dict(slope=1.,intercept=0.,status='identity_no_supported_model_adjustment')
    if not weight or frame.empty:return identity
    f=frame.dropna(subset=['reference','model','outcome'])
    days=(pd.to_datetime(f.day).max()-pd.to_datetime(f.day).min()).days+1 if len(f) else 0
    if f.event_id.nunique()<market.RULES['minimum_calibration_games'] or days<market.RULES['minimum_calibration_days']:
        return dict(identity,status='insufficient_separate_calibration_evidence')
    p=np.array([market.blend(r.reference,r.model,weight) for r in f.itertuples()])
    weights=1/f.groupby('event_id').event_id.transform('size')
    return base.fit_calibration(p,f.outcome.to_numpy(float),weights)


def train(conn):
    if registry.load_experiment(trial.NAME,root=trial.ROOT,store=trial.STORE):raise ValueError('Registered weights cannot be replaced')
    frozen.active();errors,error_coverage=game_errors(conn)
    artifact=dict(protocol=trial.PROTOCOL,game_error=errors,game_error_coverage=error_coverage,weights={},
        historical_status='development only: prior timestamped offers, no presumed execution',betting_authorized=False)
    # Publish intended dates/recipe before examining weight scores.
    atomic_json(trial.ROOT/'reports/market_rotation_fit_protocol.json',dict(protocol=trial.PROTOCOL,market=market.RULES,started_at=data.now()))
    start=datetime(2026,10,1,tzinfo=timezone.utc);end=datetime(2026,11,4,tzinfo=timezone.utc)
    nba_results=feeds.nba_result_index(conn,data.now());records=[];counts=Counter();seen=set()
    for sport in feeds.SUPPORTED:
        packets=feeds.snapshots(conn,sport,start,end)
        print('Consensus development',sport,'snapshots',len(packets),flush=True)
        for packet in packets:
            for o in market.offers(packet['event'],sport,packet['at']):
                counts[sport+':observed_pairs']+=1
                if o['consensus'] is None:counts[sport+':missing_consensus']+=1;continue
                key=(sport,o['event_id'],o['market'],o['player_norm'])
                if key in seen:continue
                seen.add(key)
                native=feeds.native_forecast(conn,o,artifact)
                actual=feeds.outcome(conn,o,native.get('forecast'),nba_results)
                day=packet['at'].date().isoformat()
                records.append(dict(sport=sport,event_id=o['event_id'],market=o['market'],player_norm=o['player_norm'],day=day,
                    reference=o['consensus']['probability'],model=native['probability'] if native.get('context_verified') else None,
                    outcome=actual['outcome'],offer=frozen.clean(o),forecast=frozen.clean(native.get('forecast')),
                    forecast_status=native.get('reason')))
    frame=pd.DataFrame(records);summaries={}
    for sport in feeds.SUPPORTED:
        markets=sorted(frame[frame.sport==sport].market.unique()) if len(frame) else []
        markets=sorted(set(markets)|({'points','rebounds','assists','threes','spread','total'} if sport=='nba' else {'spread','total'}))
        for stat in markets:
            f=frame[(frame.sport==sport)&(frame.market==stat)] if len(frame) else pd.DataFrame()
            earlier=f[f.day<trial.PROTOCOL['fit_before']] if len(f) else f
            fit=market.fit_weight(earlier)
            calibration=f[(f.day>=trial.PROTOCOL['fit_before'])&(f.day<trial.PROTOCOL['calibrate_before'])] if len(f) else f
            cal=fit_calibration(calibration,fit['weight'])
            # A supported blend also needs a separate calibration period; otherwise use zero.
            if fit['weight']>0 and cal.get('status')=='insufficient_separate_calibration_evidence':
                fit=dict(fit,proposed_weight=fit['weight'],weight=0.,status='insufficient_separate_calibration_evidence')
            artifact['weights'][sport+':'+stat+':production']=dict(fit,calibration=cal)
            artifact['weights'][sport+':'+stat+':rotation_mixture']=dict(weight=0.,status='no_prior_prospective_rotation_forecasts',rows=0,games=0,days=0,
                calibration={'slope':1.,'intercept':0.})
            evaluation=f[f.day>=trial.PROTOCOL['calibrate_before']].dropna(subset=['outcome']) if len(f) else f
            summary=dict(rows=len(f),fit=fit,calibration_rows=len(calibration),evaluation_rows=len(evaluation))
            if len(evaluation):
                e=evaluation.copy()
                e['blended']=[base.calibrate(market.blend(r.reference,r.model,fit['weight']),cal) if fit['weight']>0 else r.reference for r in e.itertuples()]
                e['gain']=(e.reference-e.outcome)**2-(e.blended-e.outcome)**2
                summary.update(consensus_brier=float(((e.reference-e.outcome)**2).mean()),blend_brier=float(((e.blended-e.outcome)**2).mean()),
                    paired_gain={k:clustered_interval(e,'gain',c) for k,c in (('game','event_id'),('day','day'))})
            summaries[sport+':'+stat]=summary
    artifact.update(built_at=data.now(),development_records=len(records),source_counts=dict(counts),source_records_sha256=data.digest(records))
    atomic_json(trial.WEIGHTS,frozen.clean(artifact))
    atomic_json(trial.ROOT/'reports/market_consensus_development_latest.json',frozen.clean(dict(artifact=artifact,groups=summaries)))
    data.DATA.mkdir(parents=True,exist_ok=True);frame.to_pickle(data.DATA/'consensus_development.pkl')
    return dict(rows=len(records),groups=summaries,weights=artifact['weights'])
