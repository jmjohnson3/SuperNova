"""Factorial target/uncertainty experiment with independent output decisions."""
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
from nfl_pipeline.modeling.component_validation import output_gates
from nfl_pipeline.modeling.evaluation import inner_partitions, expanding_week_folds, clustered_gain
from nfl_pipeline.modeling.train_accuracy_challengers import line_frame, ROOT
from nfl_pipeline.modeling.train_target_volume import fit_bundle, summarize

CONTRACT = 'nfl-target-components-v1'
VARIANTS = ('reference', 'targets_only', 'uncertainty_only', 'combined')


def target_mean(bundle, frame, base=None):
    model = bundle['model']
    base = model.baseline(frame) if base is None else np.asarray(base, dtype=float)
    raw = model.count_head.predict(model.features(frame)).clip(0)
    return (1-bundle['point_alpha'])*base + bundle['point_alpha']*raw


def fit_components(frame):
    bundle = fit_bundle(frame)
    _, _, tuning, _, _ = inner_partitions(frame)
    actual = numeric(tuning, 'targets').to_numpy()
    scores = {}
    for alpha in (0., .25, .5):
        bundle['point_alpha'] = alpha
        scores[alpha] = float(np.mean((target_mean(bundle, tuning)-actual)**2))
    bundle['point_alpha'] = min(scores, key=scores.get)
    bundle['point_tuning'] = scores
    return bundle


def curves(bundle, frame, *, projection=None, baseline_targets=None, residuals=None):
    """Mean-only shifts preserve uncertainty; shape-only preserves the mean."""
    model = bundle['model']
    base = model.baseline(frame) if baseline_targets is None else np.asarray(baseline_targets, dtype=float)
    point = bundle['reference'].predict(frame) if projection is None else np.asarray(projection, dtype=float)
    if not np.isfinite(point).all() or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Missing positive target baseline or frozen projection')
    errors = bundle['reference_errors'] if residuals is None else np.asarray(residuals, dtype=float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError('Missing frozen uncertainty residuals')
    reference = point[:, None]+errors[None, :]
    equal = np.full_like(reference, 1/reference.shape[1])
    center = reference.mean(axis=1)
    predicted_targets = target_mean(bundle, frame, base)
    # The same efficiency estimate is used in all four cells of this ablation.
    rate = point/np.maximum(.5, base)
    shift = (predicted_targets-base)*rate
    mixed, mass, _ = model.yardage_curve(frame, None, 'challenger', point, base)
    centered = mixed-(mixed*mass).sum(axis=1)[:, None]+center[:, None]
    return {
        'reference': (reference, equal, base),
        'targets_only': (reference+shift[:, None], equal, predicted_targets),
        'uncertainty_only': (centered, mass, base),
        'combined': (centered+shift[:, None], mass, predicted_targets),
    }


def output_decisions(rows, lines):
    keys = ['game_id', 'player_id', 'season', 'week']
    reference = rows.loc[rows.variant.eq('reference')]
    ref_lines = lines.loc[lines.variant.eq('reference')]
    decisions = {}
    for name in VARIANTS[1:]:
        candidate = rows.loc[rows.variant.eq(name)]
        joined = candidate.merge(reference, on=keys, suffixes=('', '_ref'), validate='one_to_one')
        paired = lines.loc[lines.variant.eq(name)].merge(ref_lines,
            on=['season', 'week', 'player_game', 'line'], suffixes=('', '_ref'), validate='one_to_one')
        if len(joined) != len(reference) or len(paired) != len(ref_lines):
            raise ValueError('Component comparisons require identical player-games and lines')
        if not np.array_equal(joined.actual, joined.actual_ref) or not np.array_equal(paired.outcome, paired.outcome_ref):
            raise ValueError('Component comparison outcomes differ')
        gates = output_gates(joined.assign(reference=joined['mean_ref']),
            paired.assign(reference_probability=paired.probability_ref), 'probability', mean='mean',
            point_preserved=name == 'uncertainty_only')
        # Typical outcome is compared against the reference median, not its mean.
        median_gain = clustered_gain(joined.assign(reference_error=abs(joined.median_ref-joined.actual),
            challenger_error=abs(joined['median']-joined.actual)))
        gates['median']['clustered_absolute_error_gain'] = median_gain
        gates['median']['pass'] = bool(median_gain.get('lower_95') is not None and median_gain['lower_95'] > 0)
        target_comparisons = {}
        for baseline in ('control_targets', 'long_targets', 'share_targets'):
            target_comparisons[baseline] = clustered_gain(joined.assign(
                reference_error=(joined[baseline]-joined.actual_targets)**2,
                challenger_error=(joined.predicted_targets-joined.actual_targets)**2))
        target_pass = bool(name != 'uncertainty_only' and all(
            g.get('lower_95') is not None and g['lower_95'] > 0 for g in target_comparisons.values()))
        gates['target_volume'] = dict(pass_=target_pass, comparisons=target_comparisons)
        gates['target_volume']['pass'] = gates['target_volume'].pop('pass_')
        gates['probability']['pass'] = bool(gates['probability']['pass'] and
            gates['probability']['interval_score_80'] <= _interval_score(reference) and
            abs(gates['probability']['coverage_80']-.8) <= abs(_coverage(reference)-.8))
        for output in ('target_volume', 'expected_mean', 'median', 'probability'):
            gates[output].update(historical_screen_passed=gates[output]['pass'], deployment_approved=False,
                status='historical_screen_passed' if gates[output]['pass'] else 'rejected',
                requires='untouched_prospective_evidence', betting_approved=False)
        # Projection gates deliberately do not depend on the probability gate or cash policy.
        decisions[name] = gates
    return decisions


def _interval_score(rows):
    from nfl_pipeline.modeling.component_validation import interval_score
    return float(np.mean(interval_score(rows.actual, rows.p10, rows.p90)))


def _coverage(rows):
    return float(np.mean((rows.actual >= rows.p10) & (rows.actual <= rows.p90)))


def run(cache, through, seasons):
    production = active_release()
    frame = joblib.load(cache)
    frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through)) &
        frame.position.isin(('WR', 'TE', 'RB')) & numeric(frame, 'targets').notna() &
        numeric(frame, 'receiving_yards').notna()].copy()
    if 'target_role_evidence' not in frame or frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Use the verified unique player-game role-context cache')
    run_id = 'target-components-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT/'target_components'/run_id
    parts = []; price_parts = []; folds = []
    for name, train, test in expanding_week_folds(frame, seasons):
        test = test.loc[~((test.season == 2026) & (test.week == 3))]
        if test.empty:
            continue
        logging.info('Component fold %s: %d training / %d test', name, len(train), len(test))
        bundle = fit_components(train)
        # Zero historical targets are not silently turned into a valid efficiency denominator.
        test = test.loc[bundle['model'].baseline(test) > 0]
        for variant, (values, weights, targets) in curves(bundle, test).items():
            parts.append(test[['game_id', 'player_id', 'season', 'week', 'game_date_et']].reset_index(drop=True).assign(
                variant=variant, fold=name, actual=numeric(test, 'receiving_yards').to_numpy(),
                actual_targets=numeric(test, 'targets').to_numpy(), predicted_targets=targets,
                control_targets=bundle['model'].baseline(test),
                long_targets=bundle['model'].baseline(test, 'long_role'),
                share_targets=bundle['model'].baseline(test, 'recent_share'), **curve_summary(values, weights)))
            price_parts.append(line_frame(test, 'receiving_yards', values, weights).assign(variant=variant, fold=name))
        folds.append(dict(fold=name, point_alpha=bundle['point_alpha'], point_tuning=bundle['point_tuning'],
            mixture_alpha=bundle['model'].alpha, train_end=bundle['training_end'],
            test_start=str(test.game_date_et.min()), model_fit_end=bundle['model_fit_end'],
            residual_end=bundle['residual_end'], tuning_end=bundle['tuning_end']))
    if not parts:
        raise ValueError('No chronological component folds')
    rows, lines = pd.concat(parts, ignore_index=True), pd.concat(price_parts, ignore_index=True)
    decisions = output_decisions(rows, lines)
    artifact = dict(contract=CONTRACT, run_id=run_id, bundle=fit_components(frame),
        created_at=datetime.now(timezone.utc).isoformat(), training_end=str(frame.game_date_et.max()),
        production_release=production['release_id'], output_decisions=decisions,
        prospective_acceptance_start='2026-10-01', deployment_approved=False, betting_approved=False)
    atomic_joblib(path/'model.joblib', artifact)
    atomic_joblib(path/'oof.joblib', dict(rows=rows, lines=lines))
    report = {k: v for k, v in artifact.items() if k != 'bundle'}
    report.update(metrics=summarize(rows, lines), folds=folds,
        artifact_sha256=hashlib.sha256((path/'model.joblib').read_bytes()).hexdigest(),
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(), production_changed=False,
        decision_contract='Independent output screens. No probability or cash requirement for point forecasts.',
        limitations=['Historical proxy-line screen, not offered-line or prospective acceptance.',
            'Week 3 is development evidence only.', 'Efficiency is shared; no new efficiency head was trained.',
            'No component is installed automatically. Failed probabilities cannot veto a separate point screen.'])
    atomic_json(path/'report.json', clean(report))
    atomic_json(ROOT/'reports/nfl_target_components_latest.json', clean(report))
    text = ['# Independent Target Components', '',
        '| Variant | Target RMSE | Yard RMSE | Brier | Calibration | 80% coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for variant, m in report['metrics'].items():
        text.append(f"| {variant} | {m['target']['rmse']:.3f} | {m['yardage']['rmse']:.3f} | {m['probability']['brier']:.4f} | {m['probability']['calibration_error']:.2%} | {m['yardage']['coverage_80']:.1%} |")
    text += ['', '| Variant | Targets | Expected yards | Typical yards | Probability |', '|---|---|---|---|---|']
    for name, decision in decisions.items():
        text.append('| '+name+' | '+' | '.join(decision[k]['status'] for k in ('target_volume','expected_mean','median','probability'))+' |')
    text += ['', report['decision_contract'], '', *report['limitations']]
    (ROOT/'reports/nfl_target_components_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    if active_release() != production:
        raise RuntimeError('Production release changed during offline evaluation')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--through', required=True)
    p.add_argument('--seasons', nargs='+', type=int, default=[2025, 2026])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    result = run(args.cache, args.through, args.seasons)
    print(json.dumps(dict(run_id=result['run_id'], decisions={k: {o: d[o]['status'] for o in
        ('target_volume','expected_mean','median','probability')} for k,d in result['output_decisions'].items()})))
