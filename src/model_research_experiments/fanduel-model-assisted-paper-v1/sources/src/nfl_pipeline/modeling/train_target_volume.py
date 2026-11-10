"""Train one isolated target-volume challenger; never install it in production."""
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
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json, atomic_joblib
from nfl_pipeline.modeling.challenger_models import numeric, curve_summary
from nfl_pipeline.modeling.component_validation import interval_score
from nfl_pipeline.modeling.evaluation import inner_partitions, expanding_week_folds, projection_metrics, probability_metrics, clustered_gain
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, line_frame, ROOT
from nfl_pipeline.modeling.role_context_training import load_sources, join_role_context
from nfl_pipeline.modeling.target_volume import TargetVolume, target_states, CONTRACT

VARIANTS = ('control', 'long_role', 'recent_share', 'challenger')


def fit_bundle(frame):
    early, residual, tuning, _, _ = inner_partitions(frame)
    reference = ReferenceRecipe().fit(early, 'receiving_yards')
    model = TargetVolume().fit(early).fit_residuals(residual, reference).tune(tuning)
    errors = numeric(residual, 'receiving_yards').to_numpy()-reference.predict(residual)
    return dict(model=model, reference=reference, reference_errors=np.quantile(errors, np.linspace(.005, .995, 101)),
        training_end=str(frame.game_date_et.max()), model_fit_end=str(early.game_date_et.max()),
        residual_end=str(residual.game_date_et.max()), tuning_end=str(tuning.game_date_et.max()))


def summarize(rows, lines):
    scores = {}
    for variant, group in rows.groupby('variant'):
        l = lines.loc[lines.variant.eq(variant)]
        scores[variant] = dict(target=projection_metrics(group.actual_targets, group.predicted_targets),
            yardage=projection_metrics(group.actual, group['mean'], group.p10, group.p90),
            probability=probability_metrics(l.probability, l.outcome, l.weight),
            interval_score=float(np.mean(interval_score(group.actual, group.p10, group.p90))),
            below_80=float(np.mean(group.actual < group.p10)), above_80=float(np.mean(group.actual > group.p90)))
    return scores


def comparison(rows, lines, reference='control'):
    keys = ['game_id', 'player_id', 'season', 'week']
    a = rows.loc[rows.variant.eq(reference)]
    b = rows.loc[rows.variant.eq('challenger')]
    d = a.merge(b, on=keys, validate='one_to_one', suffixes=('_ref', '_new'))
    gain = {}
    for name, pred, actual in (('targets', 'predicted_targets', 'actual_targets'), ('yards', 'mean', 'actual')):
        gain[name] = clustered_gain(d.assign(reference_error=(d[pred+'_ref']-d[actual+'_ref'])**2,
                                             challenger_error=(d[pred+'_new']-d[actual+'_new'])**2))
    la = lines.loc[lines.variant.eq(reference)]; lb = lines.loc[lines.variant.eq('challenger')]
    p = la.merge(lb, on=['season', 'week', 'player_game', 'line'], validate='one_to_one', suffixes=('_ref', '_new'))
    if len(p) != len(la) or len(p) != len(lb) or not np.array_equal(p.outcome_ref, p.outcome_new):
        raise ValueError('Probability comparison requires identical outcomes and lines')
    p = p.assign(reference_error=(p.probability_ref-p.outcome_ref)**2, challenger_error=(p.probability_new-p.outcome_new)**2)
    gain['brier'] = clustered_gain(p.groupby(['season', 'week', 'player_game'])[['reference_error','challenger_error']].mean().reset_index())
    return gain


def accepted(scores, gain):
    candidate = scores['challenger']; control = scores['control']; reference = scores['reference']
    return bool(all(gain[k].get('lower_95') is not None and gain[k]['lower_95'] > 0 for k in ('targets', 'yards', 'brier'))
        and candidate['probability']['brier'] < reference['probability']['brier']
        and candidate['probability']['calibration_error'] <= min(control['probability']['calibration_error'], reference['probability']['calibration_error'])
        and abs(candidate['yardage']['coverage_80']-.8) <= abs(control['yardage']['coverage_80']-.8)
        and candidate['interval_score'] <= control['interval_score'])


def run(cache, seasons, through):
    production = active_release()
    run_id = 'target-volume-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT/'target_volume'/run_id
    frame = joblib.load(cache)
    frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through)) & frame.position.isin(('WR','TE','RB'))
        & numeric(frame, 'targets').notna() & numeric(frame, 'receiving_yards').notna()].copy()
    locks, observations = load_sources(frame.game_id.unique(), datetime.now(timezone.utc))
    frame, context = join_role_context(frame, locks, observations)
    atomic_joblib(path/'training.joblib', frame)
    atomic_json(path/'context.json', clean(context))
    logging.info('Role context: %s', context['overall'])
    rows = []; lines = []; folds = []
    for name, train, test in expanding_week_folds(frame, seasons):
        # Week 3 was used to formulate the repair, never an untouched acceptance set.
        test = test.loc[~((test.season == 2026) & (test.week == 3))]
        if test.empty:
            continue
        logging.info('Training %s (%d / %d player-games)', name, len(train), len(test))
        bundle = fit_bundle(train); model = bundle['model']; ref = bundle['reference']
        actual = numeric(test, 'receiving_yards').to_numpy(); targets = numeric(test, 'targets').to_numpy()
        for variant in (*VARIANTS, 'reference'):
            if variant == 'reference':
                v = ref.predict(test)[:,None]+bundle['reference_errors'][None,:]
                w = np.full_like(v, 1/v.shape[1]); predicted_targets = model.baseline(test)
            else:
                v, w, components = model.yardage_curve(test, ref, variant)
                predicted_targets = components['targets']
            summary = curve_summary(v,w)
            rows.append(test[['game_id','player_id','season','week','game_date_et']].reset_index(drop=True).assign(
                variant=variant, fold=name, actual=actual, actual_targets=targets,
                predicted_targets=predicted_targets, **summary))
            lines.append(line_frame(test, 'receiving_yards', v, w).assign(variant=variant, fold=name))
        folds.append(dict(fold=name, alpha=model.alpha, tuning=model.tuning,
            model_fit_end=bundle['model_fit_end'], residual_end=bundle['residual_end'], tuning_end=bundle['tuning_end'],
            used_role_features=[c for c in model.columns if c.startswith('role_')]))
    if not rows:
        raise ValueError('No chronological test folds')
    r, l = pd.concat(rows, ignore_index=True), pd.concat(lines, ignore_index=True)
    scores = summarize(r,l); gain = comparison(r,l)
    passed = accepted(scores,gain)
    final = fit_bundle(frame)
    report = dict(run_id=run_id, contract=CONTRACT, status='historical_screen_passed' if passed else 'rejected',
        built_at=datetime.now(timezone.utc).isoformat(), production_release=production['release_id'],
        training_end=str(frame.game_date_et.max()), context=context, metrics=scores, comparison=gain, folds=folds,
        historical_screen_passed=passed, forecast_deployment_approved=False, betting_approved=False,
        prospective_acceptance_start='2026-10-01', production_changed=False,
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        limitations=['Efficiency recipe, efficiency residuals, and ranking are not tuned between target variants.',
            'The control and target variants share one conditional yardage mixture; the reference is tested separately.',
            'Historical proxy lines are not real offered-line or profitable-bet evidence.',
            'No missing historical lock is reconstructed. Unlearnable role fields are excluded and reported.',
            'Week 3 is diagnostic only; post-development locked evidence must pass before deployment.'])
    artifact = dict(bundle=final, run_id=run_id, contract=CONTRACT, created_at=report['built_at'],
        production_release=production['release_id'], training_end=report['training_end'],
        historical_screen_passed=passed, prospective_acceptance_start=report['prospective_acceptance_start'])
    atomic_joblib(path/'model.joblib', artifact)
    atomic_joblib(path/'oof.joblib', dict(rows=r, lines=l))
    report['artifact_sha256'] = hashlib.sha256((path/'model.joblib').read_bytes()).hexdigest()
    atomic_json(path/'report.json', clean(report))
    atomic_json(ROOT/'reports/nfl_target_volume_latest.json', clean(report))
    text = ['# Target-Volume-Only Challenger', '', report['status'], '',
        '| Variant | Target RMSE | Yard RMSE | Brier | Calibration | 80% coverage |', '|---|---:|---:|---:|---:|---:|']
    for variant,m in scores.items():
        text.append(f"| {variant} | {m['target']['rmse']:.3f} | {m['yardage']['rmse']:.3f} | {m['probability']['brier']:.4f} | {m['probability']['calibration_error']:.2%} | {m['yardage']['coverage_80']:.1%} |")
    text += ['', '## Context', json.dumps(clean(context['overall'])), '', *report['limitations'],
             '', 'No production deployment or cash approval.']
    (ROOT/'reports/nfl_target_volume_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    if active_release() != production:
        raise RuntimeError('Production changed during offline experiment')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--through', required=True)
    p.add_argument('--seasons', nargs='+', type=int, default=[2025,2026])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from nfl_pipeline.modeling.train_target_volume import run as train
    result = train(a.cache,a.seasons,a.through)
    print(json.dumps({k:result[k] for k in ('run_id','status','production_changed')}))
