"""Isolated rate repair with chronological tuning and independent output decisions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_joblib, atomic_json
from nfl_pipeline.modeling.challenger_models import numeric, curve_summary
from nfl_pipeline.modeling.component_validation import interval_score
from nfl_pipeline.modeling.evaluation import inner_partitions, expanding_week_folds, projection_metrics, probability_metrics, clustered_gain
from nfl_pipeline.modeling.receiving_efficiency import CONTRACT, ReceivingEfficiency, forecast, frozen_opportunity
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, line_frame, ROOT


def fit_bundle(frame):
    early, residual, tuning, _, _ = inner_partitions(frame)
    reference = ReferenceRecipe().fit(early, 'receiving_yards')
    errors = numeric(residual, 'receiving_yards').to_numpy()-reference.predict(residual)
    bundle = dict(reference=reference, reference_errors=np.quantile(errors, np.linspace(.005,.995,101)))
    targets = frozen_opportunity(tuning)
    actual = numeric(tuning, 'receiving_yards').to_numpy()
    reference_mean = reference.predict(tuning)+bundle['reference_errors'].mean()
    trials = []; winner = None
    for strength in (20., 60., 120.):
        model = ReceivingEfficiency(strength).fit(early)
        for residual_weight in (0., .5, 1.):
            rate = model.predict(tuning, residual_weight)
            for alpha in (0., .25, .5, .75, 1.):
                prediction = reference_mean+alpha*(targets*rate-reference_mean)
                mse = float(np.mean((prediction-actual)**2))
                record = dict(prior_targets=strength, residual_weight=residual_weight, alpha=alpha, tuning_mse=mse)
                trials.append(record)
                if winner is None or mse < winner['tuning_mse']:
                    winner = record
                    bundle.update(model=model, residual_weight=residual_weight, alpha=alpha)
    bundle.update(tuning=winner, tuning_candidates=trials, training_end=str(frame.game_date_et.max()),
        model_fit_end=str(early.game_date_et.max()), residual_end=str(residual.game_date_et.max()),
        tuning_end=str(tuning.game_date_et.max()))
    return bundle


def metrics(rows, lines):
    result = {}
    for variant, group in rows.groupby('variant'):
        l = lines.loc[lines.variant.eq(variant)]
        result[variant] = dict(expected=projection_metrics(group.actual, group['mean']),
            typical=projection_metrics(group.actual, group['median']),
            distribution=projection_metrics(group.actual, group['mean'], group.p10, group.p90),
            probability=probability_metrics(l.probability,l.outcome,l.weight),
            interval_score=float(np.mean(interval_score(group.actual,group.p10,group.p90))))
    return result


def decisions(rows, lines):
    scores = metrics(rows,lines)
    gains = {}
    for name in ('reference','rolling_baseline'):
        ref = rows.loc[rows.variant.eq(name)]
        new = rows.loc[rows.variant.eq('rate_repair')]
        joined = new.merge(ref,on=['game_id','player_id','season','week'],suffixes=('', '_ref'),validate='one_to_one')
        if len(joined) != len(ref) or len(joined) != len(new) or not np.array_equal(joined.actual,joined.actual_ref):
            raise ValueError('Point comparisons require identical player-games and outcomes')
        l = lines.loc[lines.variant.eq('rate_repair')].merge(lines.loc[lines.variant.eq(name)],
            on=['player_game','line','season','week'],suffixes=('', '_ref'),validate='one_to_one')
        if len(l) != len(lines.loc[lines.variant.eq(name)]) or not np.array_equal(l.outcome,l.outcome_ref):
            raise ValueError('Probability comparisons require identical lines and outcomes')
        gains[name] = dict(
            expected_mean=clustered_gain(joined.assign(reference_error=(joined['mean_ref']-joined.actual)**2,
                                                       challenger_error=(joined['mean']-joined.actual)**2)),
            median=clustered_gain(joined.assign(reference_error=abs(joined.median_ref-joined.actual),
                                                challenger_error=abs(joined['median']-joined.actual))),
            probability=clustered_gain(l.assign(reference_error=(l.probability_ref-l.outcome)**2,
                challenger_error=(l.probability-l.outcome)**2).groupby(['season','week','player_game'])[
                    ['reference_error','challenger_error']].mean().reset_index()))
    new = scores['rate_repair']; ref = scores['reference']
    result = {}
    for output in ('expected_mean','median','probability'):
        passes = all(g[output].get('lower_95') is not None and g[output]['lower_95'] > 0 for g in gains.values())
        if output == 'expected_mean':
            passes = passes and abs(new['expected']['bias']) <= abs(ref['expected']['bias'])
        if output == 'probability':
            passes = passes and new['probability']['calibration_error'] <= ref['probability']['calibration_error']
            passes = passes and abs(new['distribution']['coverage_80']-.8) <= abs(ref['distribution']['coverage_80']-.8)
            passes = passes and new['interval_score'] <= ref['interval_score']
        result[output] = dict(historical_screen_passed=bool(passes), deployment_approved=False,
            betting_approved=False, status='historical_screen_passed' if passes else 'rejected')
    return scores, gains, result


def run(cache, through, seasons):
    before = active_release()
    frame = joblib.load(cache)
    frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through)) & frame.position.isin(('WR','TE','RB'))].copy()
    if frame.duplicated(['game_id','player_id']).any():
        raise ValueError('Expected unique player-games, not offer rows')
    valid = numeric(frame,'targets').ge(0) & numeric(frame,'receiving_yards').notna() & numeric(frame,'targets_avg_5').ge(0)
    excluded = int((~valid).sum()); frame = frame.loc[valid].copy()
    parts=[]; price_parts=[]; folds=[]
    for name, train, test in expanding_week_folds(frame,seasons):
        # Previously examined 2026 Week 3+ results cannot be an untouched acceptance set.
        test = test.loc[~((test.season==2026)&(test.week>=3))]
        if test.empty:
            continue
        logging.info('%s: %d training / %d later player-games',name,len(train),len(test))
        b=fit_bundle(train); c=forecast(b,test)
        _, residual, _, _, _=inner_partitions(train)
        base_errors=np.quantile(numeric(residual,'receiving_yards').to_numpy()-b['reference'].base(residual),np.linspace(.005,.995,101))
        for variant, values in (('reference',c['reference']), ('rate_repair',c['candidate']),
                                ('rolling_baseline',b['reference'].base(test)[:,None]+base_errors)):
            parts.append(test[['game_id','player_id','season','week','position','game_date_et']].reset_index(drop=True).assign(
                variant=variant,fold=name,actual=numeric(test,'receiving_yards').to_numpy(),targets=c['targets'],
                predicted_rate=c['rate'],**curve_summary(values,c['weights'])))
            price_parts.append(line_frame(test,'receiving_yards',values,c['weights']).assign(variant=variant,fold=name))
        folds.append(dict(fold=name,model_fit_end=b['model_fit_end'],residual_end=b['residual_end'],
            tuning_end=b['tuning_end'],test_start=str(test.game_date_et.min()),tuning=b['tuning']))
    if not parts:
        raise ValueError('No chronological evaluation folds')
    rows=pd.concat(parts,ignore_index=True); lines=pd.concat(price_parts,ignore_index=True)
    scores,gains,outputs=decisions(rows,lines)
    run_id='receiving-efficiency-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=MODEL_ROOT/'receiving_efficiency'/run_id
    artifact=dict(contract=CONTRACT,run_id=run_id,production_release=before['release_id'],
        training_end=str(frame.game_date_et.max()),bundle=fit_bundle(frame),output_decisions=outputs,
        deployment_approved=False,betting_approved=False)
    atomic_joblib(path/'model.joblib',artifact)
    atomic_joblib(path/'oof.joblib',dict(rows=rows,lines=lines))
    report={k:v for k,v in artifact.items() if k!='bundle'}
    report.update(metrics=scores,comparisons=gains,folds=folds,excluded_missing_targets_or_outcomes=excluded,
        feature_support=artifact['bundle']['model'].feature_support,
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256((path/'model.joblib').read_bytes()).hexdigest(),
        production_changed=False,limitations=[
            'Historical refit of the conservative architecture, not the actual locked production forecasts.',
            'Proxy-line Brier is not evidence of profit or offered-line pricing accuracy.',
            'Targets and uncertainty shape are fixed; this experiment changes efficiency only.',
            'Current-game results, odds, injuries and route proxies are not rate-model features.',
            'Previously examined historical folds are a development screen, not prospective acceptance.',
            'No automatic model deployment or change to cash approval.'])
    atomic_json(path/'report.json',clean(report))
    atomic_json(ROOT/'reports/nfl_receiving_efficiency_latest.json',clean(report))
    text=['# Receiving Efficiency Repair','','| Variant | Mean RMSE | Bias | Median MAE | Brier | Calibration | 80% coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in scores.items():
        text.append(f"| {name} | {m['expected']['rmse']:.3f} | {m['expected']['bias']:.3f} | {m['typical']['mae']:.3f} | {m['probability']['brier']:.4f} | {m['probability']['calibration_error']:.2%} | {m['distribution']['coverage_80']:.1%} |")
    text += ['',*['- '+k+': '+v['status'] for k,v in outputs.items()],'',*report['limitations']]
    (ROOT/'reports/nfl_receiving_efficiency_latest.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if active_release()!=before:
        raise RuntimeError('Production changed during offline training')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--through',required=True)
    p.add_argument('--seasons',nargs='+',type=int,default=[2025,2026])
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    r=run(a.cache,a.through,a.seasons)
    print(json.dumps(dict(run_id=r['run_id'],outputs=r['output_decisions'],metrics=r['metrics'])))
