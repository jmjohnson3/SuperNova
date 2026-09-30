"""Train workload repairs with same-fold ablations; never publish production."""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json, atomic_joblib
from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual, FLOORS, numeric, player_features)
from nfl_pipeline.modeling.evaluation import inner_partitions, clustered_gain
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, ROOT
from nfl_pipeline.modeling.train_workload_depth import evaluate
from nfl_pipeline.modeling.workload_depth_model import WorkloadDepthModel, STATS
from nfl_pipeline.modeling.workload_repair import (
    WorkloadRepairModel, AsymmetricWorkloadResidual, CONTRACT)


def reference_uncertainty(frame, stat):
    early, scale, residual, _, _ = inner_partitions(frame)
    model = ReferenceRecipe().fit(early, stat)
    prediction = model.predict(scale)
    uncertainty = AsymmetricWorkloadResidual().fit_scale(player_features(scale, stat),
        numeric(scale, stat).to_numpy() - prediction, prediction, FLOORS[stat])
    uncertainty.fit_residuals(player_features(residual, stat),
        numeric(residual, stat).to_numpy() - model.predict(residual))
    return uncertainty


def paired_gain(old, new):
    keys = ['season', 'week', 'player_game', 'line']
    a = old[keys + ['outcome', 'calibrated_probability']]
    b = new[keys + ['outcome', 'calibrated_probability']]
    rows = a.merge(b, on=keys, validate='one_to_one', suffixes=('_old', '_new'))
    if len(rows) != len(old) or len(rows) != len(new) or not np.array_equal(rows.outcome_old, rows.outcome_new):
        raise ValueError('Ablations must evaluate identical player-games and lines')
    rows['reference_error'] = (rows.calibrated_probability_old - rows.outcome_old) ** 2
    rows['challenger_error'] = (rows.calibrated_probability_new - rows.outcome_new) ** 2
    grouped = rows.groupby(['season', 'week', 'player_game'])[['reference_error', 'challenger_error']].mean().reset_index()
    return clustered_gain(grouped)


def run(cache, seasons, through, stats):
    before = active_release()
    run_id = 'workload-repair-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT / 'workload_repair' / run_id
    frame = joblib.load(cache)
    import pandas as pd
    frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through))].copy()
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Expected one row per player-game')
    report = dict(run_id=run_id, contract=CONTRACT, production_release=before['release_id'],
        training_end=str(frame.game_date_et.max()), created_at=datetime.now(timezone.utc).isoformat(),
        cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(), stats={},
        production_changed=False, betting_approved=False,
        limitations=['Historical player-game refits use date-lagged finalized data, not recreated historical locks.',
            'Proxy lines screen forecast quality, not betting profitability.',
            'Actual future lock scoring and independent-week evidence remain necessary.',
            'No refit after residual/calibration blocks: uncertainty and calibration match the tested model.',
            'Missing routes/injuries remain unknown; no new data provider is fabricated.'])
    models = {}
    for stat in stats:
        comparisons = {}
        outputs = {}
        for name, model, uncertainty in (
            ('existing_workload', WorkloadDepthModel, ConditionalResidual),
            ('conditional_efficiency', WorkloadRepairModel, ConditionalResidual),
            ('asymmetric_repair', WorkloadRepairModel, AsymmetricWorkloadResidual)):
            logging.info('Training %s %s', stat, name)
            result, rows, lines, bundle = evaluate(frame, stat, seasons, model, uncertainty, refit=False)
            comparisons[name] = result
            outputs[name] = lines
            atomic_joblib(path / f'{stat}_{name}_oof.joblib', dict(rows=rows, lines=lines))
            if name == 'asymmetric_repair':
                eligible = frame.loc[frame.position.isin(('RB', 'WR', 'TE') if stat == 'receiving_yards' else ('QB', 'RB'))
                    & numeric(frame, stat).notna() & numeric(frame, STATS[stat]).notna()]
                bundle['reference_uncertainty'] = reference_uncertainty(eligible, stat)
                models[stat] = bundle
        gain = paired_gain(outputs['existing_workload'], outputs['asymmetric_repair'])
        comparisons['versus_existing_workload'] = gain
        comparisons['uncertainty_increment'] = paired_gain(outputs['conditional_efficiency'], outputs['asymmetric_repair'])
        comparisons['ready_for_prospective_test'] = bool(gain.get('lower_95') is not None and gain['lower_95'] > 0
            and comparisons['asymmetric_repair']['output_gates']['probability']['pass'])
        report['stats'][stat] = comparisons
        atomic_json(path / 'report.json', clean(report))
    artifact = dict(run_id=run_id, contract=CONTRACT, models=models, training_end=report['training_end'],
                    created_at=report['created_at'], production_release=before['release_id'])
    atomic_joblib(path / 'models.joblib', artifact)
    if active_release() != before:
        raise RuntimeError('Production changed during offline training')
    atomic_json(MODEL_ROOT / 'workload_repair' / 'latest.json', dict(run_id=run_id,
        sha256=hashlib.sha256((path / 'models.joblib').read_bytes()).hexdigest()))
    atomic_json(ROOT / 'reports/nfl_workload_repair_latest.json', clean(report))
    lines = ['# Workload and Uncertainty Repair', '', 'Offline experiments only; no production or cash change.', '',
        '| Stat / variant | MAE | Baseline MAE | Opportunity MAE | Brier | Calibration | 80% coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stat, variants in report['stats'].items():
        for name in ('existing_workload', 'conditional_efficiency', 'asymmetric_repair'):
            r = variants[name]; p = r['probabilities']['calibrated_probability']
            lines.append(f"| {stat} / {name} | {r['projection']['mae']:.3f} | {r['baseline']['mae']:.3f} | "
                f"{r['opportunity']['mae']:.3f} | {p['brier']:.4f} | {p['calibration_error']:.2%} | {r['projection']['coverage_80']:.1%} |")
        lines += ['', f"{stat} paired week-grouped Brier gain: {json.dumps(variants['versus_existing_workload'])}",
                  f"Ready for separate prospective test: {variants['ready_for_prospective_test']}"]
    lines += ['', *report['limitations']]
    (ROOT / 'reports/nfl_workload_repair_latest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--through', required=True)
    parser.add_argument('--seasons', type=int, nargs='+', default=[2025, 2026])
    parser.add_argument('--stats', nargs='+', choices=tuple(STATS), default=list(STATS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from nfl_pipeline.modeling.train_workload_repair import run as train
    result = train(args.cache, args.seasons, args.through, args.stats)
    print(json.dumps(dict(run_id=result['run_id'], production_changed=False)))
