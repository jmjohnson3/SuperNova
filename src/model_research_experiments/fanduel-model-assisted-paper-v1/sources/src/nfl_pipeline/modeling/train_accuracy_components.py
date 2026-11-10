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
    ConditionalResidual,OpportunityRateModel,HierarchicalCalibration,player_features,numeric,
    curve_summary,workload_state,OPPORTUNITY,FLOORS,POSITIONS)
from nfl_pipeline.modeling.component_validation import WorkloadTailCalibration,output_gates,calibrate_curve
from nfl_pipeline.modeling.evaluation import expanding_week_folds,inner_partitions,probability_metrics
from nfl_pipeline.modeling.team_workload import (
    load_history,prepare_history,season_recency_weights,TeamAllocationModel)
from nfl_pipeline.modeling.forecast_repair import StableEfficiency
from nfl_pipeline.modeling.train_accuracy_challengers import (
    ROOT,load_training,fit_bundle,mixture_curve,line_frame,ReferenceRecipe)

log=logging.getLogger(__name__)

TEAM_TAIL_CUTOFFS={
    'receiving_yards':(4.,10.),
    'rushing_yards':(6.,14.),
    'passing_yards':(20.,35.),
}

RECEIVING_OPPORTUNITY_BLEND_GRID=(0.,.15,.25,.35,.50,.75,1.)


def _stable_role_opportunity(frame,stat):
    """Use the longest available pregame role sample as the stable anchor."""
    op=OPPORTUNITY[stat]
    recent=numeric(frame,op+'_avg_5').fillna(0).clip(lower=0).to_numpy()
    if stat!='receiving_yards':
        return recent
    long_role=numeric(frame,op+'_avg_10')
    return long_role.where(long_role.notna() & long_role.ge(0),recent).to_numpy()


def _blend_opportunity(frame,team_opportunity,covered,stat,blend):
    """Shrink volatile team allocations toward a stable player-role history."""
    blend=float(blend)
    if not 0<=blend<=1:
        raise ValueError('Opportunity blend must be in [0,1]')
    stable=_stable_role_opportunity(frame,stat)
    team=np.asarray(team_opportunity,dtype=float)
    covered=np.asarray(covered,dtype=bool) & np.isfinite(team)
    return np.where(covered,(1-blend)*stable+blend*team,stable)


def _preserve_curve_mean(values,weights,expected):
    """Shape the tails without silently changing the expected-yard forecast."""
    values=np.asarray(values,dtype=float)
    weights=np.asarray(weights,dtype=float)
    expected=np.asarray(expected,dtype=float)
    mass=weights.sum(axis=1)
    current=np.divide((values*weights).sum(axis=1),mass,
        out=np.zeros(len(values),dtype=float),where=mass>0)
    return values+(expected-current)[:,None],weights


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
    team=TeamAllocationModel().fit(t,r,stat)
    recency_strength=float(getattr(team,'recency_strength',0.))
    return {'team':team,'reference':ReferenceRecipe().fit(train,stat),
            'rate':StableEfficiency().fit(train,stat),
            'state':OpportunityRateModel().fit(
                train,stat,season_recency_weights(train,recency_strength)),
            'season_recency_strength':recency_strength,
            'season_recency_validation':getattr(team,'recency_validation',None)}


def team_opportunities(bundle,frame,teams,roster,stat,*,snapshot=False):
    if snapshot:
        output=bundle['team'].predict_snapshot(frame,bundle.get('volume_strength'),bundle.get('share_strength'))
    else:
        ids=set(frame.game_id)
        team=teams.loc[teams.game_id.isin(ids) & teams.history_games.ge(3)]
        pool=roster.loc[roster.game_id.isin(team.game_id)]
        # The pool is built from prior games, not the frame's eventual participants.
        output=(bundle['team'].predict(team,pool,bundle.get('volume_strength'),bundle.get('share_strength'))
                if len(pool) else pd.DataFrame())
    opportunity=np.full(len(frame),np.nan)
    if len(output):
        lookup=output.set_index(['game_id','team_abbr','player_id']).expected_opportunity.to_dict()
        for i,row in enumerate(frame.itertuples()):
            value=lookup.get((row.game_id,row.team_abbr,str(row.player_id)))
            if value is not None and np.isfinite(value): opportunity[i]=value
    return opportunity,np.isfinite(opportunity)


def team_components(bundle,frame,teams,roster,stat,*,base=None,snapshot=False,early_exit=None):
    reference=bundle['reference'].predict(frame) if base is None else np.asarray(base,dtype=float)
    opportunity,covered=team_opportunities(bundle,frame,teams,roster,stat,snapshot=snapshot)
    state=bundle['state'].components(frame)
    weights=np.asarray(state['weights'],dtype=float).copy()
    raw_op=np.asarray(state['opportunities'],dtype=float)
    raw_mean=(weights*raw_op).sum(axis=1)
    fallback=_stable_role_opportunity(frame,stat)
    opportunity_blend=float(bundle.get('opportunity_blend',1.))
    desired=_blend_opportunity(frame,opportunity,covered,stat,opportunity_blend)
    scaled=raw_op*np.divide(desired,raw_mean,out=np.ones_like(desired),where=raw_mean>1e-8)[:,None]
    rate=np.maximum(0,bundle['rate'].predict(frame))
    centers=scaled*rate[:,None]
    team_point=(centers*weights).sum(axis=1)
    blend=float(bundle.get('point_blend',0.))
    point=(1-blend)*reference+blend*team_point
    p=float(bundle.get('early_exit_probability',0.) if early_exit is None else early_exit)
    if p:
        established=fallback>=({'receiving_yards':4.,'rushing_yards':8.,'passing_yards':24.}[stat])
        add=np.where(established,p,0.)
        weights*=1-add[:,None]
        weights[:,0]+=add
    # Preserve the expected point while allowing low/normal/spike paths to
    # change downside and upside mass.
    centers=np.maximum(0,centers+(point-(centers*weights).sum(axis=1))[:,None])
    centers[~covered]=reference[~covered,None]
    return dict(centers=centers,weights=weights,point=point,opportunity=desired,
                rate=rate,covered=covered,state_probabilities=weights)


def team_curve(bundle,frame,teams,roster,stat,*,base=None,snapshot=False,early_exit=None,
               books=None,extra_lines=None,calibrate=True):
    comp=team_components(bundle,frame,teams,roster,stat,base=base,snapshot=snapshot,early_exit=early_exit)
    values,weights=bundle['uncertainty'].mixture(player_features(frame,stat),comp['centers'],comp['weights'])
    tail=bundle.get('tail')
    if tail is not None:
        values,weights=tail.transform(values,weights,comp['opportunity'])
    if calibrate and bundle.get('calibrator') is not None:
        values,weights=calibrate_curve(bundle['calibrator'],frame,stat,values,weights,
            books=books,extra_lines=extra_lines)
    if bundle.get('mean_preserving_distribution',False):
        values,weights=_preserve_curve_mean(values,weights,comp['point'])
    return values,weights,comp


def team_predictions(bundle,frame,teams,roster,stat):
    comp=team_components(bundle,frame,teams,roster,stat)
    return comp['point'],comp['covered']


def _tune_team_structure(bundle,frame,teams,roster,stat):
    actual=numeric(frame,stat).to_numpy()
    actual_opportunity=numeric(frame,OPPORTUNITY[stat]).to_numpy()
    rate=np.maximum(0,bundle['rate'].predict(frame))
    reference=bundle['reference'].predict(frame)
    opportunity_grid=(RECEIVING_OPPORTUNITY_BLEND_GRID if stat=='receiving_yards' else (1.,))
    rate_scale=max(1.,float(np.nanmedian(rate))**2)
    best=None
    for volume in (0.,.25,.5):
        for share in (0.,.25,.5):
            bundle['volume_strength']=volume; bundle['share_strength']=share
            opportunity,covered=team_opportunities(bundle,frame,teams,roster,stat)
            for opportunity_blend in opportunity_grid:
                desired=_blend_opportunity(frame,opportunity,covered,stat,opportunity_blend)
                team_point=desired*rate
                target_error=desired-actual_opportunity
                for blend in (0.,.25,.5,.75):
                    point=np.where(covered,(1-blend)*reference+blend*team_point,reference)
                    error=point-actual
                    # Yardage remains primary. A smaller target penalty prevents
                    # a compensating efficiency error from selecting unrealistic volume.
                    score=float(np.mean(error**2)+.1*np.mean(error)**2
                                +.25*rate_scale*np.mean(target_error**2))
                    choice=(score,blend,volume,share,opportunity_blend)
                    if best is None or choice<best: best=choice
    _,blend,volume,share,opportunity_blend=best
    bundle.update(point_blend=blend,volume_strength=volume,share_strength=share,
                  opportunity_blend=opportunity_blend,
                  structure_tuning=dict(score=best[0],point_blend=blend,
                    volume_strength=volume,share_strength=share,
                    opportunity_blend=opportunity_blend,rows=len(frame)))
    bundle['team'].volume_strength=volume; bundle['team'].share_strength=share
    return bundle


def _tune_early_exit(bundle,fit,gate,teams,roster,stat):
    def score(frame,p):
        values,weights,_=team_curve(bundle,frame,teams,roster,stat,early_exit=p,calibrate=False)
        summary=curve_summary(values,weights)
        lines=line_frame(frame,stat,values,weights)
        brier=probability_metrics(lines.probability,lines.outcome,lines.weight)['brier']
        coverage=float(np.mean((numeric(frame,stat)>=summary['p10']) & (numeric(frame,stat)<=summary['p90'])))
        interval=float(np.mean((summary['p90']-summary['p10'])+
            10*np.maximum(summary['p10']-numeric(frame,stat),0)+
            10*np.maximum(numeric(frame,stat)-summary['p90'],0)))
        return dict(brier=brier,coverage=coverage,interval_score=interval)
    candidates={p:score(fit,p) for p in (0.,.01,.02,.03)}
    selected=min(candidates,key=lambda p:(candidates[p]['brier']+abs(candidates[p]['coverage']-.8)*.02,p))
    before,after=score(gate,0.),score(gate,selected)
    enabled=bool(selected>0 and after['brier']<before['brier'] and
        abs(after['coverage']-.8)<=abs(before['coverage']-.8) and
        after['interval_score']<=before['interval_score'])
    bundle['early_exit_probability']=selected if enabled else 0.
    bundle['early_exit_validation']=dict(selected=selected,enabled=enabled,fit=candidates,
        gate_without=before,gate_with=after,interpretation='generic early-exit mass; not injury prediction')
    return bundle


def _tune_team_tail(bundle,fit,gate,teams,roster,stat):
    """Repair stat-specific distribution width without changing workload centers."""
    values,weights,comp=team_curve(bundle,fit,teams,roster,stat,calibrate=False)
    tail=WorkloadTailCalibration(*TEAM_TAIL_CUTOFFS[stat]).fit(
        numeric(fit,stat).to_numpy(),curve_summary(values,weights),comp['opportunity'])
    values,weights,comp=team_curve(bundle,gate,teams,roster,stat,calibrate=False)
    repaired,masses=tail.transform(values,weights,comp['opportunity'],force=True)
    before=line_frame(gate,stat,values,weights)
    after=line_frame(gate,stat,repaired,masses)
    brier=lambda frame:probability_metrics(frame.probability,frame.outcome,frame.weight)['brier']
    tail.validate(numeric(gate,stat).to_numpy(),curve_summary(values,weights),
                  curve_summary(repaired,masses),brier(before),brier(after))
    bundle['tail']=tail
    return bundle


def fit_team_bundle(train,teams,rosters,stat):
    early,scale,residual,cal_fit,cal_gate=inner_partitions(train)
    provisional=fit_team_model(early,teams,rosters,stat)
    provisional['rate'].tune(scale)
    _tune_team_structure(provisional,scale,teams,rosters,stat)
    provisional['mean_preserving_distribution']=stat=='receiving_yards'
    comp=team_components(provisional,scale,teams,rosters,stat)
    states=workload_state(scale,stat)
    chosen=comp['centers'][np.arange(len(scale)),states]
    u=ConditionalResidual().fit_scale(player_features(scale,stat),numeric(scale,stat).to_numpy()-chosen,
        comp['point'],FLOORS[stat],confidence_errors=numeric(scale,stat).to_numpy()-comp['point'])
    comp=team_components(provisional,residual,teams,rosters,stat)
    states=workload_state(residual,stat)
    u.fit_residuals(player_features(residual,stat),
        numeric(residual,stat).to_numpy()-comp['centers'][np.arange(len(residual)),states],states)
    provisional['uncertainty']=u
    _tune_early_exit(provisional,cal_fit,cal_gate,teams,rosters,stat)
    _tune_team_tail(provisional,cal_fit,cal_gate,teams,rosters,stat)
    values,weights,_=team_curve(provisional,cal_fit,teams,rosters,stat,calibrate=False)
    provisional['calibrator']=HierarchicalCalibration().fit(line_frame(cal_fit,stat,values,weights))
    values,weights,_=team_curve(provisional,cal_gate,teams,rosters,stat,calibrate=False)
    provisional['calibrator'].validate(line_frame(cal_gate,stat,values,weights))
    final=fit_team_model(train,teams,rosters,stat)
    final['rate'].strength=provisional['rate'].strength
    for key in ('volume_strength','share_strength','point_blend','opportunity_blend',
                'mean_preserving_distribution','structure_tuning',
                'early_exit_probability','early_exit_validation','tail'):
        final[key]=provisional[key]
    final['team'].volume_strength=final['volume_strength']; final['team'].share_strength=final['share_strength']
    final['uncertainty']=u; final['calibrator']=provisional['calibrator']
    return final


def evaluate_team(data,teams,rosters,stat,source,seasons):
    data=data.loc[data.position.isin(POSITIONS[stat])].copy()
    lookup=source['player_games'].set_index(['game_id','player_id'])
    comparable=pd.MultiIndex.from_frame(data[['game_id','player_id']]).isin(lookup.index)
    missing_reference_rows=int((~comparable).sum())
    data=data.loc[comparable].copy()
    opportunity=numeric(data,OPPORTUNITY[stat]); yards=numeric(data,stat)
    valid=np.isfinite(opportunity) & np.isfinite(yards) & opportunity.ge(0) & ~(opportunity.eq(0) & yards.ne(0))
    invalid_rows=int((~valid).sum())
    data=data.loc[valid].copy(); rows=[]; lines=[]; skipped_folds=[]; tail_validations=[]
    for name,train,test in expanding_week_folds(data,seasons):
        try:
            bundle=fit_team_bundle(train,teams,rosters,stat)
        except ValueError as exc:
            if 'Insufficient chronological training weeks' not in str(exc):
                raise
            skipped_folds.append({'fold':name,'reason':str(exc),'train_rows':len(train)})
            log.info('Skipping team allocation %s %s: %s',stat,name,exc)
            continue
        values,weights,comp=team_curve(bundle,test,teams,rosters,stat)
        center,covered=comp['point'],comp['covered']
        summary=curve_summary(values,weights)
        r=test[['game_id','player_id','season','week']].copy()
        r['reference']=[lookup.loc[(a,b),'reference'] for a,b in zip(r.game_id,r.player_id)]
        r=r.assign(actual=numeric(test,stat).to_numpy(),challenger=summary['mean'],median=summary['median'],
            p10=summary['p10'],p90=summary['p90'],allocation_available=covered,
            predicted_opportunity=comp['opportunity'],actual_opportunity=numeric(test,OPPORTUNITY[stat]).to_numpy(),
            limited_probability=comp['state_probabilities'][:,0],spike_probability=comp['state_probabilities'][:,2],fold=name)
        l=aligned_reference(line_frame(test,stat,values,weights),source['proxy_lines'])
        rows.append(r); lines.append(l)
        tail_validations.append({'fold':name,**bundle['tail'].validation})
        log.info('Team allocation %s %s covered=%s/%s',stat,name,int(covered.sum()),len(test))
    if not rows:
        raise ValueError('No eligible team-allocation evaluation folds')
    rows=pd.concat(rows,ignore_index=True); lines=pd.concat(lines,ignore_index=True)
    gates=output_gates(rows,lines,'probability'); gates['allocation_coverage']=float(rows.allocation_available.mean())
    gates['tail_validations']=tail_validations
    gates['tail_enabled_folds']=sum(bool(v.get('enabled')) for v in tail_validations)
    gates['invalid_opportunity_rows_excluded']=invalid_rows
    gates['missing_frozen_reference_rows_excluded']=missing_reference_rows
    gates['skipped_folds']=skipped_folds
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


def publish_stat_candidates(artifact,report,*,replace_components=None):
    """Register per-stat shadow scorers without changing production or cash status."""
    from nfl_pipeline.modeling.stat_runtime import (
        ConditionalUncertaintyRuntime,QBAttemptsEfficiencyRuntime,TeamOpportunityRuntime)
    from nfl_pipeline.modeling.stat_model_registry import load_registry,register_candidate
    root=MODEL_ROOT/'accuracy_components'/artifact['run_id']/'stat_candidates'
    replace_components=set(replace_components or ())
    registered={}
    pinned=(load_registry().get('candidates') or {})
    choices=(
        ('receiving_yards','team_receiving_yards',TeamOpportunityRuntime),
        ('receiving_yards','receiving_conditional',ConditionalUncertaintyRuntime),
        ('rushing_yards','team_rushing_yards',TeamOpportunityRuntime),
        ('passing_yards','qb_tail',QBAttemptsEfficiencyRuntime),
    )
    for stat,name,runtime_cls in choices:
        component=artifact['models'].get(name)
        gates=report['components'].get(name)
        if not component or not gates:
            continue
        historical=list(gates.get('enabled_prospective_outputs') or ())
        if not historical:
            continue
        existing=next((entry for entry in (pinned.get(stat) or {}).values()
            if set(entry.get('outputs') or ()).intersection(historical)),None)
        replace_existing=name in replace_components
        if existing and not replace_existing:
            registered.setdefault(stat,{})[name]=existing
            gates['candidate_registration_status']='preserved_existing_prospective_trial'
            gates['registered_component_id']=existing['component_id']
            continue
        runtime=(runtime_cls(component['uncertainty'],stat)
            if runtime_cls is ConditionalUncertaintyRuntime
            else runtime_cls(component['bundle'],stat))
        payload={'contract':'nfl-team-opportunity-stat-v2','stat':stat,
            'runtime':runtime,
            'production_release':artifact['production_release'],'run_id':artifact['run_id']+'-'+name,
            'training_end':artifact['training_end'],'component':name,'betting_approved':False}
        # Candidate files are immutable.  Reusing model.joblib can overwrite the
        # bytes referenced by the current registry before its checksum changes.
        model_path=root/stat/f'{name}.joblib'
        atomic_joblib(model_path,payload)
        sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
        review={'contract':'nfl-stat-candidate-evidence-v1','stat':stat,'component':name,
            'artifact_sha256':sha,'historical_screen_outputs':historical,
            'approved_outputs':[],'deployment_approved':False,'betting_approved':False,
            'blockers':['prospective_identical_fanduel_line_validation_required'],
            'built_at':datetime.now(timezone.utc).isoformat()}
        evidence=root/stat/f'{name}.evidence.json'; atomic_json(evidence,review)
        entry=register_candidate(
            stat,model_path,historical,evidence=evidence,
            replace_existing=replace_existing,
        )
        registered.setdefault(stat,{})[name]=entry
        gates['candidate_registration_status']=(
            'registered_replacement_prospective_trial'
            if replace_existing and entry['component_id']==payload['run_id']
            else 'registered_new_prospective_trial'
            if entry['component_id']==payload['run_id']
            else 'preserved_existing_prospective_trial')
        gates['registered_component_id']=entry['component_id']
    return registered


def refresh_candidate_registry(*,replace_components=None):
    """Publish accepted outputs from the latest immutable training artifact."""
    manifest=json.loads((MODEL_ROOT/'accuracy_components'/'latest.json').read_text(encoding='utf-8'))
    root=MODEL_ROOT/'accuracy_components'/manifest['run_id']
    model_path=root/'models.joblib'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=manifest['sha256']:
        raise RuntimeError('Accuracy component checksum mismatch')
    artifact=joblib.load(model_path)
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    registered=publish_stat_candidates(
        artifact,report,replace_components=replace_components,
    )
    report['stat_registry_candidates']={
        stat:{name:entry['component_id'] for name,entry in components.items()}
        for stat,components in registered.items()
    }
    atomic_json(root/'report.json',report)
    write_latest_report(report)
    return report['stat_registry_candidates']


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
    history_cache=MODEL_ROOT/'accuracy_components'/'team_history_cache.joblib'
    history_contract='team-history-v3-lagged-usage'
    through=str(players.game_date_et.max())
    cached=joblib.load(history_cache) if history_cache.exists() else None
    if (isinstance(cached,dict) and cached.get('through')==through
            and cached.get('contract')==history_contract):
        teams,roster=cached['teams'],cached['roster']
        log.info('Using cached leakage-safe team history through %s',through)
    else:
        raw=load_history(); raw=raw.loc[raw.game_date_et<=players.game_date_et.max()]
        teams,roster=prepare_history(raw)
        atomic_joblib(history_cache,{'contract':history_contract,'through':through,'teams':teams,'roster':roster,
            'built_at':datetime.now(timezone.utc).isoformat(),'source_rows':len(raw)})
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
    report['stat_registry_candidates']={
        stat:{name:entry['component_id'] for name,entry in components.items()}
        for stat,components in publish_stat_candidates(artifact,report).items()
    }
    atomic_json(root/'report.json',report)
    write_latest_report(report)
    if production!=(MODEL_ROOT/'active_release.json').read_bytes(): raise RuntimeError('Production release changed')
    return {name:g['enabled_prospective_outputs'] for name,g in report['components'].items()}


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-approvals',action='store_true')
    parser.add_argument('--refresh-candidates',action='store_true')
    parser.add_argument(
        '--replace-candidate-component',action='append',default=[],
        choices=('team_receiving_yards','receiving_conditional','team_rushing_yards','qb_tail'),
        help='Start a new prospective cohort for this component output; requires --refresh-candidates.',
    )
    args=parser.parse_args()
    if args.refresh_approvals and args.refresh_candidates:
        parser.error('choose only one refresh mode')
    if args.replace_candidate_component and not args.refresh_candidates:
        parser.error('--replace-candidate-component requires --refresh-candidates')
    action=(refresh_approvals if args.refresh_approvals else
            (lambda: refresh_candidate_registry(
                replace_components=args.replace_candidate_component,
            )) if args.refresh_candidates else run)
    print(json.dumps(action(),indent=2))
