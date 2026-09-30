"""Offline asymmetric receiving tails, learned before each later-week test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json, atomic_joblib
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.modeling.challenger_models import curve_summary, numeric, POSITIONS
from nfl_pipeline.modeling.model_benchmark import fit_uncertainty, raw_curve, curve_metrics, summarize
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, line_frame, ROOT
from nfl_pipeline.modeling.sample_aware_splits import sample_aware_partitions, describe_blocks
from nfl_pipeline.modeling.evaluation import expanding_week_folds

STAT='receiving_yards'
LOG=logging.getLogger(__name__)


class ReceivingTailRepair:
    """Monotone support transform; no independent line-probability clipping."""
    @staticmethod
    def groups(frame):
        targets=numeric(frame,'targets_avg_5')
        bucket=np.where(targets.isna(),'unknown',np.where(targets<4,'low',np.where(targets<8,'normal','high')))
        return frame.position.fillna('unknown').astype(str).to_numpy()+'|'+bucket

    def fit(self,frame,values,weights):
        if frame.duplicated(['game_id','player_id']).any():
            raise ValueError('Tail repair requires unique player-games')
        s=curve_summary(values,weights); y=numeric(frame,STAT).to_numpy()
        low=(y-s['median'])/np.maximum(5.,s['median']-s['p10'])
        high=(y-s['median'])/np.maximum(5.,s['p90']-s['median'])
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError('Tail training requires finite outcomes')
        def factors(mask):
            return np.array([-np.quantile(low[mask],.1),np.quantile(high[mask],.9)]).clip(.75,2.5)
        self.global_factors=factors(np.ones(len(frame),dtype=bool))
        self.by_group={}; groups=self.groups(frame)
        for key in np.unique(groups):
            mask=groups==key; n=mask.sum()
            if n>=80:
                self.by_group[key]=(n*factors(mask)+200*self.global_factors)/(n+200)
        self.strength=0.; self.enabled=False
        return self

    def transform(self,frame,values,weights,strength=None):
        if strength is None:
            strength=self.strength if self.enabled else 0.
        if not 0<=strength<=1:
            raise ValueError('Tail strength must be in [0,1]')
        s=curve_summary(values,weights)
        factors=np.array([self.by_group.get(g,self.global_factors) for g in self.groups(frame)])
        delta=values-s['median'][:,None]
        scales=np.where(delta<0,factors[:,0,None],factors[:,1,None])
        return s['median'][:,None]+delta*(1+strength*(scales-1)),weights.copy()


def improvement(before,after):
    b=before['expected']['coverage_80']; a=after['expected']['coverage_80']
    return bool(after['probability']['brier']<before['probability']['brier']
        and after['probability']['calibration_error']<=before['probability']['calibration_error']
        and abs(a-.8)<abs(b-.8) and .75<=a<=.85
        and after['interval_score']<=before['interval_score'])


def fit_bundle(train):
    early,scale,residual,fit,tune,gate=sample_aware_partitions(train,STAT)
    reference=ReferenceRecipe().fit(early,STAT)
    base=dict(kind='conditional',model=reference,uncertainty=fit_uncertainty(reference,scale,residual,STAT))
    repair=ReceivingTailRepair().fit(fit,*raw_curve(base,fit,STAT))
    v,w=raw_curve(base,tune,STAT)
    scores={a:curve_metrics(tune,STAT,*repair.transform(tune,v,w,a)) for a in (0.,.25,.5,.75,1.)}
    repair.strength=min(scores,key=lambda a:scores[a]['probability']['brier'])
    v,w=raw_curve(base,gate,STAT)
    before=curve_metrics(gate,STAT,v,w)
    after=curve_metrics(gate,STAT,*repair.transform(gate,v,w,repair.strength))
    repair.enabled=repair.strength>0 and improvement(before,after)
    repair.validation=dict(enabled=repair.enabled,strength=repair.strength,before=before,after=after)
    errors=numeric(residual,STAT).to_numpy()-reference.predict(residual)
    pooled=dict(kind='pooled',model=reference,errors=np.quantile(errors,np.linspace(.005,.995,101)))
    return dict(base=base,pooled=pooled,repair=repair,blocks=describe_blocks((early,scale,residual,fit,tune,gate)),
        model_fit_end=str(early.game_date_et.max()),training_end=str(train.game_date_et.max()))


def evaluate_fold(bundle,frame,fold):
    v,w=raw_curve(bundle['base'],frame,STAT)
    curves={'reference':raw_curve(bundle['pooled'],frame,STAT),'conditional':(v,w),
        'tail_candidate':bundle['repair'].transform(frame,v,w,bundle['repair'].strength),
        'gated_tail':bundle['repair'].transform(frame,v,w)}
    rows=[];lines=[]
    for name,(v,w) in curves.items():
        rows.append(frame[['game_id','player_id','season','week','game_date_et']].reset_index(drop=True).assign(
            family=name,fold=fold,actual=numeric(frame,STAT).to_numpy(),**curve_summary(v,w)))
        lines.append(line_frame(frame,STAT,v,w).assign(family=name,fold=fold))
    return pd.concat(rows,ignore_index=True),pd.concat(lines,ignore_index=True)


def run(cache,seasons):
    frozen=active_release(); started=datetime.now(timezone.utc)
    run_id='receiving-tails-'+started.strftime('%Y%m%dT%H%M%S%fZ')
    path=MODEL_ROOT/'receiving_uncertainty_repair'/run_id
    frame=joblib.load(cache)
    frame=frame.loc[frame.position.isin(POSITIONS[STAT]) & numeric(frame,STAT).notna()].copy()
    if frame.duplicated(['game_id','player_id']).any():
        raise ValueError('Input must contain unique player-games')
    rows=[];lines=[];folds=[]
    for fold,train,test in expanding_week_folds(frame,seasons):
        LOG.info('Receiving tails fold %s: %s train / %s test',fold,len(train),len(test))
        try:
            bundle=fit_bundle(train)
        except ValueError as exc:
            folds.append(dict(fold=fold,status='insufficient_history',reason=str(exc)));continue
        r,l=evaluate_fold(bundle,test,fold);rows.append(r);lines.append(l)
        folds.append(dict(fold=fold,status='evaluated',train_end=str(train.game_date_et.max()),
            test_start=str(test.game_date_et.min()),gate=bundle['repair'].validation))
    if not rows:
        raise ValueError('No chronological folds could be evaluated')
    r=pd.concat(rows,ignore_index=True);l=pd.concat(lines,ignore_index=True)
    summary=summarize(r,l)
    # Compare against conditional uncertainty as well as the pooled reference.
    base_r=r.loc[r.family.eq('conditional')].assign(family='reference')
    base_l=l.loc[l.family.eq('conditional')].assign(family='reference')
    conditional=summarize(pd.concat([base_r,r.loc[r.family.eq('gated_tail')]]),
                          pd.concat([base_l,l.loc[l.family.eq('gated_tail')]]))['gated_tail']
    new=summary['gated_tail']; baseline=summary['reference']
    gain=conditional['brier_gain']
    ready=bool(gain.get('lower_95') is not None and gain['lower_95']>0
        and new['probability']['brier']<baseline['probability']['brier']
        and new['probability']['calibration_error']<=baseline['probability']['calibration_error']
        and new['probability']['calibration_error']<=summary['conditional']['probability']['calibration_error']
        and .75<=new['expected']['coverage_80']<=.85
        and new['interval_score']<=summary['conditional']['interval_score'])
    report=dict(run_id=run_id,status='ready_for_separate_prospective_trial' if ready else 'not_accepted',
        built_at=datetime.now(timezone.utc).isoformat(),production_release=frozen['release_id'],
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),folds=folds,
        summary=summary,versus_conditional=conditional,production_changed=False,betting_approved=False,
        limitations=['Historical refit uses lagged final player-games, not archived live lock-time features.',
            'Proxy lines are not true offered-line or betting-edge proof.',
            'Pinned receiving trial and production artifacts are untouched.',
            'Raw tail_candidate uses tuning strength even if the separate selection gate rejected it.'])
    atomic_joblib(path/'oof.joblib',dict(rows=r,lines=l))
    # Save a diagnostic model without publishing any pointer consumed by production.
    final=fit_bundle(frame)
    atomic_joblib(path/'model.joblib',dict(bundle=final,run_id=run_id,created_at=report['built_at'],
        production_release=frozen['release_id'],training_end=str(frame.game_date_et.max())))
    atomic_json(path/'report.json',clean(report))
    atomic_json(ROOT/'reports/nfl_receiving_uncertainty_repair_latest.json',clean(report))
    text=['# Receiving Uncertainty Repair','',f'Status: {report["status"]}',
        'Offline only. Production and the pinned prospective trial were not changed.','',
        '| Variant | Brier | Calibration | 80% coverage | Interval score | Mean RMSE | Median MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        text.append(f"| {name} | {s['probability']['brier']:.4f} | {s['probability']['calibration_error']:.2%} | {s['expected']['coverage_80']:.1%} | {s['interval_score']:.2f} | {s['expected']['rmse']:.2f} | {s['typical']['mae']:.2f} |")
    text+=['','Gated tail vs conditional, week-grouped Brier gain: '+json.dumps(gain),'',*report['limitations']]
    (ROOT/'reports/nfl_receiving_uncertainty_repair_latest.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if active_release()!=frozen:
        raise RuntimeError('Production release changed during offline training')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--seasons',type=int,nargs='+',default=[2024,2025,2026]);a=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    from nfl_pipeline.modeling.receiving_uncertainty_repair import run as train
    r=train(a.cache,a.seasons);print(json.dumps(dict(run_id=r['run_id'],status=r['status'])))
