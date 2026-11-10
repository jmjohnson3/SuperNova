"""Reconstructed development study; prospective rows are never backfilled."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_props import research_data as data,research_models as base,research_trial as frozen,research_evaluation as evaluate
from nfl_pipeline.integrity import atomic_json
from edge_research import rotation

ROOT=data.ROOT
ARTIFACT=ROOT/'src/nba_props/models/rotation_v1.joblib'
MANIFEST=ROOT/'reports/nba_rotation_training_manifest.json'
CACHE=data.DATA/'rotation_development.pkl'


def states(players,teams):
    ps,ts=base.states(players,teams)
    p=players[players.minutes.gt(0)].copy();p['game_date']=pd.to_datetime(p.game_date)
    p=p.sort_values(['player_id','game_date','game_id']).reset_index(drop=True)
    for label,values in (('two_m',p.fgm-p.fg3m),('two_a',p.fga-p.fg3a),('free_m',p.ftm),('free_a',p.fta)):
        p[label]=values.groupby(p.player_id).transform(lambda x:x.rolling(40,min_periods=1).sum())
    ps['two_pct']=(p.two_m+53)/(p.two_a+100);ps['ft_pct']=(p.free_m+39)/(p.free_a+50)
    t=teams.copy();t['game_date']=pd.to_datetime(t.game_date)
    t=t.sort_values(['team','game_date','game_id']).reset_index(drop=True)
    budgets=t[['team','game_date']].rename(columns={'game_date':'team_history_date'}).copy()
    def rolling_mean(x):
        return x.rolling(20,min_periods=5).apply(lambda a:np.average(a,weights=.5**(np.arange(len(a)-1,-1,-1)/8)),raw=True)
    for s in ('fga','fg3a','fta','ast','reb'):
        normalized=t[s]*240/t.minutes
        budgets['budget_'+s]=normalized.groupby(t.team).transform(rolling_mean)
    return ps,ts,budgets


def features(targets,ps,ts,budgets):
    f=base.features(targets,ps,ts)
    return f.merge(budgets,on=['team','team_history_date'],how='left',validate='many_to_one')


def historical_pool(players,teams):
    """Construct roster candidates from prior games only, including eventual nonplayers."""
    p=players.copy();p['game_date']=pd.to_datetime(p.game_date)
    t=teams.copy();t['game_date']=pd.to_datetime(t.game_date)
    parts=[]
    for day,games in t[t.game_date.between('2024-10-01','2026-06-30')].groupby('game_date'):
        cutoff=day-pd.Timedelta(days=2)
        latest=p[p.game_date.between(cutoff-pd.Timedelta(days=rotation.SPEC['history_days']),cutoff)]
        latest=latest.sort_values(['game_date','game_id']).drop_duplicates('player_id',keep='last')
        for g in games.itertuples():
            roster=latest[latest.team==g.team][['player_id','player_name']].copy()
            roster=roster.assign(game_id=g.game_id,game_date=day,team=g.team,opponent=g.opponent,home=g.home,
                availability='unknown',confirmed_starter=np.nan,minutes_cap=np.nan)
            parts.append(roster)
    return pd.concat(parts,ignore_index=True)


def predict_groups(fit,frame,ot):
    records=[];excluded={}
    for (gid,team),g in frame.groupby(['game_id','team'],sort=True):
        budget={s:float(g['budget_'+s].iloc[0]) for s in ('fga','fg3a','fta','ast','reb')}
        # Unknown rookies/roster gaps cannot silently donate all minutes to known players.
        if g.games_before.lt(5).any() or g[['two_pct','ft_pct',*base.FEATURES]].isna().any().any() or not all(np.isfinite(list(budget.values()))):
            excluded['incomplete_history']=excluded.get('incomplete_history',0)+1;continue
        try:result=rotation.scenarios(fit,g,budget,ot,gid+':'+team)
        except ValueError as exc:
            excluded[str(exc)]=excluded.get(str(exc),0)+1;continue
        identities=g.set_index('player_id')
        for i,pid in enumerate(result['player_ids']):
            row=identities.loc[pid].to_dict();row['player_id']=pid
            for s,values in result['values'].items():
                row['scenario_'+s]=values[i];row['pred_'+s]=float(values[i]@result['weights'])
            row['scenario_weights']=result['weights'];row['team_context_verified']=result['selection_eligible']
            records.append(row)
    return pd.DataFrame(records),excluded


def train(conn):
    from edge_research.trial import NAME,STORE
    from nfl_pipeline import experiment_registry as registry
    if registry.load_experiment(NAME,root=ROOT,store=STORE):raise ValueError('Registered study cannot be replaced')
    frozen.active();asof=data.now();p,t=data.history(conn,asof);p=base.attach_past_roles(p,conn,asof)
    ps,ts,budgets=states(p,t)
    training=base.features(p,ps,ts);games=base.game_features(t,ts)
    pool=features(historical_pool(p,t),ps,ts,budgets)
    ot=rotation.overtime_probabilities(t,'2024-09-29')
    manifest=dict(started_at=asof,specification=rotation.SPEC,corpus_sha256=data.digest(sorted(p.source_snapshot.unique())),
        parent_artifact_sha256=hashlib.sha256(frozen.ARTIFACT.read_bytes()).hexdigest(),historical_only=True,
        current_game_participation_used_as_feature=False,betting_authorized=False)
    atomic_json(MANIFEST,frozen.clean(manifest))
    predicted=[];excluded={}
    bounds=['2024-10-01','2025-02-01','2025-07-01','2026-07-01']
    for start,end in zip(bounds,bounds[1:]):
        cutoff=pd.Timestamp(start)-pd.Timedelta(days=2)
        fitted=base.fit(training[training.game_date<cutoff],games[games.game_date<cutoff])
        part=pool[(pool.game_date>=start)&(pool.game_date<end)]
        print('Rotation fold',start,'targets',len(part),flush=True)
        forecast,bad=predict_groups(fitted,part,ot)
        forecast['trained_before']=start;predicted.append(forecast);excluded[start]=bad
    f=pd.concat(predicted,ignore_index=True)
    labels=p[['game_id','player_id','minutes',*base.STATS,'fg3a']]
    f=f.merge(labels,on=['game_id','player_id'],how='left',validate='one_to_one')
    artifact=dict(specification=rotation.SPEC,ot_probabilities=ot,dispersion={},calibration={},parent_artifact_sha256=manifest['parent_artifact_sha256'])
    for stat in base.STATS:
        a=f[(f.game_date<rotation.SPEC['dispersion_period'][1])&f[stat].notna()&f.minutes.gt(0)]
        means=np.stack(a['scenario_'+stat]);w=np.stack(a.scenario_weights)
        print('Fit conditional dispersion',stat,len(a),flush=True)
        dispersion=rotation.fit_dispersion(means,w,a[stat].to_numpy())
        c=f[(f.game_date>=rotation.SPEC['calibration_period'][0])&(f.game_date<rotation.SPEC['calibration_period'][1])&f[stat].notna()&f.minutes.gt(0)]
        cal=rotation.fit_calibration(np.stack(c['scenario_'+stat]),np.stack(c.scenario_weights),c[stat].to_numpy(),stat,dispersion)
        artifact['dispersion'][stat]=dispersion;artifact['calibration'][stat]=cal
    ARTIFACT.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,ARTIFACT)
    f.to_pickle(CACHE)
    manifest.update(completed_at=data.now(),excluded_teams=excluded,forecast_rows=len(f),
        artifact_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),dispersion=artifact['dispersion'],calibration=artifact['calibration'])
    atomic_json(MANIFEST,frozen.clean(manifest))
    return finish_mean_ablation(conn)


def finish_mean_ablation(conn):
    """Give the collapsed-mean comparator its own disjoint dispersion/calibration fits."""
    from edge_research.trial import NAME,STORE
    from nfl_pipeline import experiment_registry as registry
    if registry.load_experiment(NAME,root=ROOT,store=STORE):raise ValueError('Registered artifact cannot be replaced')
    f=pd.read_pickle(CACHE);artifact=joblib.load(ARTIFACT)
    artifact['mean_dispersion']={};artifact['mean_calibration']={}
    for stat in base.STATS:
        a=f[(f.game_date<rotation.SPEC['dispersion_period'][1])&f[stat].notna()&f.minutes.gt(0)]
        d=base.fit_dispersion(a[stat].to_numpy(),a['pred_'+stat].to_numpy())
        c=f[(f.game_date>=rotation.SPEC['calibration_period'][0])&(f.game_date<rotation.SPEC['calibration_period'][1])&f[stat].notna()&f.minutes.gt(0)]
        artifact['mean_dispersion'][stat]=d
        artifact['mean_calibration'][stat]=base.fit_distribution_calibration(c,stat,d)
    joblib.dump(artifact,ARTIFACT)
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    manifest.update(artifact_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),mean_dispersion=artifact['mean_dispersion'],mean_calibration=artifact['mean_calibration'])
    atomic_json(MANIFEST,frozen.clean(manifest))
    return report(conn,f,artifact)


def report(conn,f,artifact):
    h=f[(f.game_date>=rotation.SPEC['evaluation_period'][0])&(f.game_date<rotation.SPEC['evaluation_period'][1])&f.minutes.gt(0)].copy()
    old=pd.read_pickle(data.DATA/'development_predictions.pkl')
    out=dict(built_at=data.now(),historical_status=rotation.SPEC['historical_status'],stat_forecasts={},props={},betting_authorized=False)
    common=h.merge(old[['game_id','player_id',*['pred_'+s for s in ('minutes',*base.STATS,'fg3a')]]],on=['game_id','player_id'],suffixes=('','_frozen'))
    for stat in ('minutes',*base.STATS,'fg3a'):
        out['stat_forecasts'][stat]=dict(rows=len(common),mae=float((common['pred_'+stat]-common[stat]).abs().mean()),
            frozen_mae=float((common['pred_'+stat+'_frozen']-common[stat]).abs().mean()))
    matched=evaluate.match_props(h,evaluate.prop_quotes(conn))
    if not matched.empty:
        from nba_props.model import match_name
        scenario=h.assign(name_key=h.player_name.map(match_name))
        matched=matched.merge(scenario[['game_id','name_key','scenario_weights',*['scenario_'+s for s in base.STATS]]],on=['game_id','name_key'],validate='many_to_one')
        for (book,stat),g in matched.groupby(['book','stat']):
            g=g.copy();d=artifact['dispersion'][stat]
            g['p_raw']=base.over(g.mu,g.line,d) # same means/dispersion, collapse workload for ablation
            raw=rotation.mixture_over(np.stack(g['scenario_'+stat]),np.stack(g.scenario_weights),g.line.to_numpy(),d)
            g['p_calibrated']=base.calibrate(raw,artifact['calibration'][stat])
            result=evaluate.probability_metrics(g)
            result['raw_mixture_brier']=float(np.mean((raw-g.won_over)**2))
            mean_p=base.calibrate(base.over(g.mu,g.line,artifact['mean_dispersion'][stat]),artifact['mean_calibration'][stat])
            result['separately_calibrated_mean_brier']=float(np.mean((mean_p-g.won_over)**2))
            result['raw_label']='collapsed mean, same conditional dispersion'
            out['props'].setdefault(book,{})[stat]=result
    atomic_json(ROOT/'reports/nba_rotation_development_latest.json',frozen.clean(out))
    lines=['# Joint rotation development comparison','',rotation.SPEC['historical_status'],
        'New hypotheses tested on previously examined historical periods; prospective evidence starts after registration.','',
        '| Stat | Common player-games | Joint rotation MAE | Frozen workload MAE |','|---|---:|---:|---:|']
    for s,r in out['stat_forecasts'].items():lines.append(f"| {s} | {r['rows']} | {r['mae']:.4f} | {r['frozen_mae']:.4f} |")
    lines+=['','Historical rosters come from earlier games. No current-game absences or final lineups are inserted as known news.',
        'Missing participation remains unknown. No historical checks establish execution. See JSON for probability ablations by sportsbook.']
    (ROOT/'reports/nba_rotation_development_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return out
