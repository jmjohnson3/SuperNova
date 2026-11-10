"""Chronological quantile forecast/calibration experiment; no cash authorization."""
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
from nfl_pipeline.modeling.evaluation import expanding_week_folds
from nfl_pipeline.modeling.model_benchmark import fit_uncertainty, raw_curve, summarize
from nfl_pipeline.modeling.receiving_quantiles import CONTRACT, ReceivingQuantiles, blend
from nfl_pipeline.modeling.sample_aware_splits import sample_aware_partitions, describe_blocks
from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, line_frame, ROOT

STAT = 'receiving_yards'
LOG = logging.getLogger(__name__)


def fit_bundle(frame):
    parts = sample_aware_partitions(frame, STAT)
    early, scale, residual, cal_fit, cal_tune, gate = parts
    reference = ReferenceRecipe().fit(early, STAT)
    conditional = dict(kind='conditional', model=reference,
        uncertainty=fit_uncertainty(reference, scale, residual, STAT))
    errors = numeric(residual, STAT).to_numpy()-reference.predict(residual)
    pooled = dict(kind='pooled', model=reference, errors=np.quantile(errors, np.linspace(.005, .995, 101)))
    quantiles = ReceivingQuantiles().fit(early).correct(scale)
    bundle = dict(reference=pooled, conditional=conditional, quantiles=quantiles)
    frames = [line_frame(f, STAT, *blend_curve(bundle, f)) for f in (cal_fit, cal_tune, gate)]
    bundle['calibrator'] = TailPreservingCalibration().fit(*frames)
    bundle['blocks'] = describe_blocks(parts)
    bundle['training_end'] = str(frame.game_date_et.max())
    return bundle


def blend_curve(bundle, frame):
    return blend(raw_curve(bundle['conditional'], frame, STAT), bundle['quantiles'].curve(frame))


def curves(bundle, frame):
    mixed = blend_curve(bundle, frame)
    return dict(reference=raw_curve(bundle['reference'], frame, STAT),
        conditional=raw_curve(bundle['conditional'], frame, STAT),
        quantile=bundle['quantiles'].curve(frame), blend=mixed,
        calibrated_blend=bundle['calibrator'].transform(*mixed))


def compare(rows, lines):
    summary = summarize(rows, lines)
    # The stronger conditional-uncertainty comparator also has to be beaten.
    conditional_rows = rows.loc[rows.family.eq('conditional')].assign(family='reference')
    conditional_lines = lines.loc[lines.family.eq('conditional')].assign(family='reference')
    versus = summarize(pd.concat([conditional_rows, rows.loc[~rows.family.isin(('reference', 'conditional'))]]),
        pd.concat([conditional_lines, lines.loc[~lines.family.isin(('reference', 'conditional'))]]))
    decisions = {}
    for name in ('quantile', 'blend', 'calibrated_blend'):
        decisions[name] = {}
        for output, gain_key in (('expected_yards', 'expected_squared_error_gain'),
                                 ('typical_yards', 'median_absolute_error_gain'), ('probability', 'brier_gain')):
            gains = [summary[name][gain_key], versus[name][gain_key]]
            passed = all(g.get('lower_95') is not None and g['lower_95'] > 0 for g in gains)
            for ref in ('reference', 'conditional'):
                a, b = summary[name], summary[ref]
                if output == 'expected_yards':
                    passed = passed and abs(a['expected']['bias']) <= abs(b['expected']['bias'])
                if output == 'probability':
                    passed = passed and a['probability']['calibration_error'] <= b['probability']['calibration_error']
                    passed = passed and abs(a['expected']['coverage_80']-.8) <= abs(b['expected']['coverage_80']-.8)
                    passed = passed and a['interval_score'] <= b['interval_score']
            decisions[name][output] = dict(historical_screen_passed=bool(passed),
                status='historical_screen_passed' if passed else 'not_accepted',
                deployment_approved=False, betting_approved=False)
    return summary, versus, decisions


def run(cache, through, seasons):
    production = active_release()
    frame = joblib.load(cache)
    frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through)) &
        frame.position.isin(('WR', 'TE', 'RB'))].copy()
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Use unique player-game training rows, not duplicated betting offers')
    valid = numeric(frame, STAT).notna() & numeric(frame, 'targets').ge(0)
    excluded = int((~valid).sum())
    frame = frame.loc[valid].copy()
    parts, line_parts, folds = [], [], []
    for fold, train, test in expanding_week_folds(frame, seasons):
        # These weeks already informed repairs; they cannot grant acceptance.
        test = test.loc[~((test.season == 2026) & (test.week >= 3))]
        if test.empty:
            continue
        LOG.info('%s: fitting %d / evaluating %d later player-games', fold, len(train), len(test))
        bundle = fit_bundle(train)
        for name, (v, w) in curves(bundle, test).items():
            parts.append(test[['game_id', 'player_id', 'season', 'week', 'position', 'game_date_et']]
                .reset_index(drop=True).assign(family=name, fold=fold,
                    actual=numeric(test, STAT).to_numpy(), **curve_summary(v, w)))
            line_parts.append(line_frame(test, STAT, v, w).assign(family=name, fold=fold))
        folds.append(dict(fold=fold, blocks=bundle['blocks'], test_start=str(test.game_date_et.min()),
            test_end=str(test.game_date_et.max()), calibration=bundle['calibrator'].validation))
    if not parts:
        raise ValueError('No chronological evaluation folds')
    rows, lines = pd.concat(parts, ignore_index=True), pd.concat(line_parts, ignore_index=True)
    summary, comparisons, decisions = compare(rows, lines)
    run_id = 'receiving-quantiles-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT/'receiving_quantiles'/run_id
    artifact = dict(contract=CONTRACT, run_id=run_id, bundle=fit_bundle(frame),
        production_release=production['release_id'], training_end=str(frame.game_date_et.max()),
        output_decisions=decisions, deployment_approved=False, betting_approved=False)
    atomic_joblib(path/'model.joblib', artifact)
    atomic_joblib(path/'oof.joblib', dict(rows=rows, lines=lines))
    report = {k: v for k, v in artifact.items() if k != 'bundle'}
    report.update(summary=summary, versus_conditional=comparisons, folds=folds,
        excluded_missing_outcomes=excluded, production_changed=False,
        feature_support=artifact['bundle']['quantiles'].feature_support,
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256((path/'model.joblib').read_bytes()).hexdigest(),
        limitations=['Chronological refit reference, not immutable historical production forecasts.',
            'Proxy-line Brier is not actual offered-line or selected-pick accuracy.',
            'Historical development screen, not an untouched prospective acceptance test.',
            'Calibration transforms one distribution, not each offered line independently.',
            'No production pointer, betting limits, rankings or fixed trial selections changed.'])
    atomic_json(path/'report.json', clean(report))
    atomic_json(ROOT/'reports/nfl_receiving_quantiles_latest.json', clean(report))
    text = ['# Receiving Quantile Forecasts', '', '| Variant | Mean RMSE | Bias | Median MAE | Brier | Calibration | 80% coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in summary.items():
        text.append(f"| {name} | {m['expected']['rmse']:.3f} | {m['expected']['bias']:+.3f} | {m['typical']['mae']:.3f} | "
            f"{m['probability']['brier']:.4f} | {m['probability']['calibration_error']:.2%} | {m['expected']['coverage_80']:.1%} |")
    text += ['', '| Variant | Expected yards | Typical yards | Probability |', '|---|---|---|---|']
    for name, decision in decisions.items():
        text.append('| '+name+' | '+' | '.join(d['status'] for d in decision.values())+' |')
    text += ['', *report['limitations']]
    (ROOT/'reports/nfl_receiving_quantiles_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    if active_release() != production:
        raise RuntimeError('Production release changed during training')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--through', required=True)
    p.add_argument('--seasons', type=int, nargs='+', default=[2025, 2026])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    r = run(a.cache, a.through, a.seasons)
    print(json.dumps(dict(run_id=r['run_id'], outputs=r['output_decisions'], summary=r['summary'])))
