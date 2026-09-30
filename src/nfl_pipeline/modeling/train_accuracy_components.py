"""Evaluate narrowly scoped accuracy changes without publishing production."""
import argparse
import hashlib
import json
import logging
from datetime import datetime,timezone

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.integrity import MODEL_ROOT,atomic_json,atomic_joblib,freeze_production
from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual,player_features,numeric,curve_summary,OPPORTUNITY,FLOORS,POSITIONS)
from nfl_pipeline.modeling.component_validation import WorkloadTailCalibration,output_gates,calibrate_curve
from nfl_pipeline.modeling.evaluation import expanding_week_folds,inner_partitions,probability_metrics
from nfl_pipeline.modeling.team_workload import load_history,prepare_history,TeamAllocationModel
from nfl_pipeline.modeling.train_accuracy_challengers import (
    ROOT,load_training,fit_bundle,mixture_curve,line_frame,ReferenceRecipe)

log=logging.getLogger(__name__)


def proxy_lines(frame,stat):
    prior=numeric(frame,f'{stat}_avg_5').fillna(0).to_numpy()
    return np.maximum(.5,np.floor(prior[:,None]*np.array([.75,1.,1.25]))+.5)


def fit_qb_tail(train):
    bundle=fit_bundle(train,'passing_yards')
    _,_,_,fit,gate=inner_partitions(train)
    values,weights,comp=mixture_curve(bundle['calibration_model'],bundle['uncertainty'],fit,'passing_yards')
    opportunity=(comp['weights']*comp['opportunities']).sum(axis=1)
    tail=WorkloadTailCalibration().fit(numeric(fit,'passing_yards').to_numpy(),curve_summary(values,weights),opportunity)
    values,weights,comp=mixture_curve(bundle['calibration_model'],bundle['uncertainty'],gate,'passing_yards')
    opportunity=(comp['weights']*comp['opportunities']).sum(axis=1)
    repaired,masses=tail.transform(values,weights,opportunity,force=True)
    values,weights=calibrate_curve(bundle['calibrator'],gate,'passing_yards',values,weights,extra_lines=proxy_lines(gate,'passing_yards'))
    repaired,masses=calibrate_curve(bundle['calibrator'],gate,'passing_yards',repaired,masses,extra_lines=proxy_lines(gate,'passing_yards'))
    before=line_frame(gate,'passing_yards',values,weights); after=line_frame(gate,'passing_yards',repaired,masses)
    brier=lambda f:probability_metrics(f.probability,f.outcome,f.weight)['brier']
    tail.validate(numeric(gate,'passing_yards').to_numpy(),curve_summary(values,weights),
                  curve_summary(repaired,masses),brier(before),brier(after))
    bundle['tail']=tail
    return bundle


def aligned_reference(lines,source):
    keys=['player_game','line']
    return lines.merge(source[keys+['reference_probability']],on=keys,how='inner',validate='one_to_one')


def evaluate_qb(data,source,seasons):
    data=data.loc[data.position.eq('QB')].copy(); rows=[]; lines=[]; validations=[]
    lookup=source['player_games'].set_index(['game_id','player_id'])
    for name,train,test in expanding_week_folds(data,seasons):
        bundle=fit_qb_tail(train)
        values,weights,comp=mixture_curve(bundle['model'],bundle['uncertainty'],test,'passing_yards')
        op=(comp['weights']*comp['opportunities']).sum(axis=1)
        repaired,masses=bundle['tail'].transform(values,weights,op)
        repaired,masses=calibrate_curve(bundle['calibrator'],test,'passing_yards',repaired,masses,extra_lines=proxy_lines(test,'passing_yards'))
        summary=curve_summary(repaired,masses)
        r=test[['game_id','player_id','season','week']].copy()
        r['reference']=[lookup.loc[(a,b),'reference'] for a,b in zip(r.game_id,r.player_id)]
        r=r.assign(actual=numeric(test,'passing_yards').to_numpy(),challenger=summary['mean'],
            median=summary['median'],p10=summary['p10'],p90=summary['p90'],fold=name)
        l=line_frame(test,'passing_yards',repaired,masses)
        l['candidate_probability']=l.probability
        l=aligned_reference(l,source['proxy_lines']); rows.append(r); lines.append(l)
        validations.append({'fold':name,**bundle['tail'].validation})
        log.info('QB tail %s enabled=%s',name,bundle['tail'].enabled)
    rows=pd.concat(rows,ignore_index=True); lines=pd.concat(lines,ignore_index=True)
    gates=output_gates(rows,lines,'candidate_probability')
    gates['inner_folds']=validations
    gates['tail_enabled_folds']=sum(v['enabled'] for v in validations)
    return gates,rows,lines,fit_qb_tail(data)


def fit_team_model(train,teams,rosters,stat):
    end=train.game_date_et.max()
    t=teams.loc[teams.game_date_et<=end]; r=rosters.loc[rosters.game_date_et<=end]
    return {'team':TeamAllocationModel().fit(t,r,stat),'reference':ReferenceRecipe().fit(train,stat)}


def team_predictions(bundle,frame,teams,roster,stat):
    ids=set(frame.game_id)
    team=teams.loc[teams.game_id.isin(ids) & teams.history_games.ge(3)]
    pool=roster.loc[roster.game_id.isin(team.game_id)]
    # The pool is built from prior games, not the frame's eventual participants.
    output=bundle['team'].predict(team,pool) if len(pool) else pd.DataFrame()
    point=bundle['reference'].predict(frame); covered=np.zeros(len(frame),dtype=bool)
    if len(output):
        lookup=output.set_index(['game_id','team_abbr','player_id']).projection.to_dict()
        for i,row in enumerate(frame.itertuples()):
            value=lookup.get((row.game_id,row.team_abbr,str(row.player_id)))
            if value is not None and np.isfinite(value): point[i]=value; covered[i]=True
    return point,covered


def fit_team_bundle(train,teams,rosters,stat):
    early,scale,residual,_,_=inner_partitions(train)
    provisional=fit_team_model(early,teams,rosters,stat)
    pred,_=team_predictions(provisional,scale,teams,rosters,stat)
    u=ConditionalResidual().fit_scale(player_features(scale,stat),numeric(scale,stat).to_numpy()-pred,pred,FLOORS[stat])
    pred,_=team_predictions(provisional,residual,teams,rosters,stat)
    u.fit_residuals(player_features(residual,stat),numeric(residual,stat).to_numpy()-pred)
    final=fit_team_model(train,teams,rosters,stat); final['uncertainty']=u
    return final


def evaluate_team(data,teams,rosters,stat,source,seasons):
    data=data.loc[data.position.isin(POSITIONS[stat])].copy(); rows=[]; lines=[]
    lookup=source['player_games'].set_index(['game_id','player_id'])
    for name,train,test in expanding_week_folds(data,seasons):
        bundle=fit_team_bundle(train,teams,rosters,stat)
        center,covered=team_predictions(bundle,test,teams,rosters,stat)
        values,weights=bundle['uncertainty'].mixture(player_features(test,stat),center,np.ones((len(test),1)))
        summary=curve_summary(values,weights)
        r=test[['game_id','player_id','season','week']].copy()
        r['reference']=[lookup.loc[(a,b),'reference'] for a,b in zip(r.game_id,r.player_id)]
        r=r.assign(actual=numeric(test,stat).to_numpy(),challenger=summary['mean'],median=summary['median'],
            p10=summary['p10'],p90=summary['p90'],allocation_available=covered,fold=name)
        l=aligned_reference(line_frame(test,stat,values,weights),source['proxy_lines'])
        rows.append(r); lines.append(l)
        log.info('Team allocation %s %s covered=%s/%s',stat,name,int(covered.sum()),len(test))
    rows=pd.concat(rows,ignore_index=True); lines=pd.concat(lines,ignore_index=True)
    gates=output_gates(rows,lines,'probability'); gates['allocation_coverage']=float(rows.allocation_available.mean())
    return gates,rows,lines,fit_team_bundle(data,teams,rosters,stat)


def prospective_outputs(name,gates,model):
    enabled=[kind for kind in ('expected_mean','median','probability') if gates[kind]['pass']]
    if name=='qb_tail':
        tail=model['bundle']['tail']
        gates['latest_tail_validation']=tail.validation
        if not tail.enabled:
            # A failed tail adjustment must not veto separately proven point forecasts.
            enabled=[kind for kind in enabled if kind!='probability']
            gates['prospective_blocker']='tail_probability_disabled_latest_calibration'
            return enabled
    gates['prospective_blocker']=None if enabled else 'no_output_improved_held_out_accuracy'
    return enabled


def publish_artifact(artifact,report):
    root=MODEL_ROOT/'accuracy_components'/artifact['run_id']
    atomic_joblib(root/'models.joblib',artifact)
    atomic_json(root/'report.json',report)
    atomic_json(MODEL_ROOT/'accuracy_components'/'latest.json',{
        'run_id':artifact['run_id'],'production_release':artifact['production_release'],
        'sha256':hashlib.sha256((root/'models.joblib').read_bytes()).hexdigest()})
    write_latest_report(report)


def refresh_approvals():
    """Reevaluate output contracts without refitting models or mutating prior runs."""
    from nfl_pipeline.modeling.score_accuracy_components import load_artifact
    artifact,manifest=load_artifact()
    if artifact is None: raise RuntimeError('No component evaluation to refresh')
    report=json.loads((MODEL_ROOT/'accuracy_components'/manifest['run_id']/'report.json').read_text())
    original=artifact['run_id']
    artifact['run_id']=report['run_id']=datetime.now(timezone.utc).strftime('components-%Y%m%dT%H%M%S%fZ')
    artifact['approval_source_run']=report['approval_source_run']=original
    artifact['oof_source_run']=artifact.get('oof_source_run',original)
    for name,gates in report['components'].items():
        enabled=prospective_outputs(name,gates,artifact['models'][name])
        gates['enabled_prospective_outputs']=artifact['models'][name]['enabled_outputs']=enabled
    publish_artifact(artifact,report)
    return {name:m['enabled_outputs'] for name,m in artifact['models'].items()}


def write_latest_report(report):
    atomic_json(ROOT/'reports'/'nfl_accuracy_components_latest.json',report)
    lines=['# NFL Accuracy Components','',f"Frozen production: {report['production_release']}",'',
        'Mean, median, and probability approvals are independent. These are historical screens, not betting approvals.','',
        '| Component | Mean RMSE (candidate / reference) | Brier (candidate / reference) | 80% coverage | Prospective outputs |','|---|---:|---:|---:|---|']
    for name,g in report['components'].items():
        lines.append(f"| {name} | {g['expected_mean']['metrics']['rmse']:.3f} / {g['expected_mean']['reference']['rmse']:.3f} | {g['probability']['metrics']['brier']:.4f} / {g['probability']['reference']['brier']:.4f} | {g['probability']['coverage_80']:.1%} | {', '.join(g['enabled_prospective_outputs']) or 'none'} |")
    lines+=['','## Latest Calibration Block','']
    for name,g in report['components'].items():
        v=g.get('latest_tail_validation')
        if v:
            lines.append(f"- {name}: {'enabled' if v['enabled'] else 'disabled'}; coverage {v['coverage_before']:.1%} -> {v['coverage_after']:.1%}; interval score {v['interval_score_before']:.3f} -> {v['interval_score_after']:.3f}; Brier {v['brier_before']:.4f} -> {v['brier_after']:.4f} ({v['rows']} rows).")
            if v['rows']<40:
                lines.append(f"  Later-block evidence: {v['rows']} rows; at least 40 required. This is separate from the outer-fold coverage requirement.")
    lines+=['',*['- '+s for s in report['limitations']]]
    (ROOT/'reports'/'nfl_accuracy_components_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    freeze=freeze_production('Component-level accuracy tests; production remains frozen')
    production=(MODEL_ROOT/'active_release.json').read_bytes()
    source_manifest=json.loads((MODEL_ROOT/'challengers'/'latest.json').read_text())
    source_root=MODEL_ROOT/'challengers'/source_manifest['run_id']
    model_path=source_root/'models.joblib'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=source_manifest['sha256']:
        raise RuntimeError('Source challenger checksum mismatch')
    source_models=joblib.load(model_path)
    if source_models['production_release']!=freeze['release_id']: raise RuntimeError('Source release mismatch')
    log.info('Evaluating components from %s against frozen %s',source_manifest['run_id'],freeze['release_id'])
    sources={s:joblib.load(source_root/f'{s}_oof.joblib') for s in OPPORTUNITY}
    players,_=load_training()
    players=players.loc[players.game_date_et<=pd.Timestamp(source_models['training_end']).date()]
    seasons=tuple(sorted(set(sources['receiving_yards']['player_games'].season)))
    raw=load_history(); raw=raw.loc[raw.game_date_et<=players.game_date_et.max()]
    teams,roster=prepare_history(raw)
    log.info('Loaded %s player-games, %s team-games, %s prior-roster rows',len(players),len(teams),len(roster))
    run_id=datetime.now(timezone.utc).strftime('components-%Y%m%dT%H%M%S%fZ')
    root=MODEL_ROOT/'accuracy_components'/run_id
    report={'run_id':run_id,'production_release':freeze['release_id'],'source_run':source_manifest['run_id'],
        'components':{},'existing_output_contracts':{},'production_changed':False,
        'limitations':['Historical tests use proxy lines, not executable sportsbook prices.',
            'Previously inspected seasons are research evidence, not an untouched final test.',
            'Team pools use earlier team games; unknown newcomers keep unallocated volume and use the reference forecast.',
            'No true routes, first-read data, or unobserved historical injuries are fabricated.',
            'Historical component approval permits prospective scoring only, never bankroll promotion.']}
    models={}
    for stat,source in sources.items():
        report['existing_output_contracts'][stat]=output_gates(source['player_games'],source['proxy_lines'],'calibrated_probability')
    source=sources['receiving_yards']; r=source['player_games'].copy()
    r['challenger']=r.reference; r['median']=r.reference
    r['p10']=r.conditional_p10; r['p90']=r.conditional_p90
    report['components']['receiving_conditional']=output_gates(r,source['proxy_lines'],'conditional_reference_probability',point_preserved=True)
    models['receiving_conditional']={'stat':'receiving_yards','uncertainty':source_models['models']['receiving_yards']['reference_uncertainty']}
    report['components']['qb_tail'],rows,lines,bundle=evaluate_qb(players,sources['passing_yards'],seasons)
    models['qb_tail']={'stat':'passing_yards','bundle':bundle}
    atomic_joblib(root/'qb_tail_oof.joblib',{'player_games':rows,'proxy_lines':lines})
    for stat in OPPORTUNITY:
        name='team_'+stat
        report['components'][name],rows,lines,bundle=evaluate_team(players,teams,roster,stat,sources[stat],seasons)
        models[name]={'stat':stat,'bundle':bundle}
        atomic_joblib(root/f'{name}_oof.joblib',{'player_games':rows,'proxy_lines':lines})
    for name,gates in report['components'].items():
        enabled=prospective_outputs(name,gates,models[name])
        gates['enabled_prospective_outputs']=enabled
        models[name]['enabled_outputs']=enabled
    artifact={'run_id':run_id,'production_release':freeze['release_id'],'source_manifest':source_manifest,
        'training_end':str(players.game_date_et.max()),'models':models,'status':'prospective_only'}
    publish_artifact(artifact,report)
    if production!=(MODEL_ROOT/'active_release.json').read_bytes(): raise RuntimeError('Production release changed')
    return {name:g['enabled_prospective_outputs'] for name,g in report['components'].items()}


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-approvals',action='store_true')
    args=parser.parse_args()
    print(json.dumps(refresh_approvals() if args.refresh_approvals else run(),indent=2))
