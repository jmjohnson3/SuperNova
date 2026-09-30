"""Chronological workload/depth experiment with isolated artifacts and no promotion."""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_joblib, atomic_json
from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual, HierarchicalCalibration, player_features, numeric,
    curve_summary, POSITIONS, FLOORS)
from nfl_pipeline.modeling.component_validation import output_gates, calibrate_curve, interval_score
from nfl_pipeline.modeling.evaluation import (
    inner_partitions, expanding_week_folds, projection_metrics, probability_metrics)
from nfl_pipeline.modeling.receiver_role_model import load_role_history
from nfl_pipeline.modeling.train_accuracy_challengers import load_training, ReferenceRecipe, line_frame, ROOT
from nfl_pipeline.modeling.workload_depth_model import WorkloadDepthModel, historical_features, states, STATS, CONTRACT

log = logging.getLogger(__name__)


def prepare(players, raw):
    raw = raw.drop_duplicates(['game_id', 'player_id'])
    fields = ['targets', 'carries', 'receptions', 'receiving_air_yards',
              'receiving_yards_after_catch', 'receiving_yards', 'rushing_yards', 'start_ts_utc']
    actuals = raw[['game_id', 'player_id'] + fields]
    frame = players.drop(columns=[c for c in fields if c in players]).merge(
        actuals, on=['game_id', 'player_id'], validate='one_to_one')
    totals = raw.groupby(['game_id', 'team_abbr'])[['pass_attempts', 'carries']].sum(min_count=1).rename(
        columns={'pass_attempts': 'team_actual_pass_attempts', 'carries': 'team_actual_carries'}).reset_index()
    frame = frame.merge(totals, on=['game_id', 'team_abbr'], validate='many_to_one')
    frame['completed_air_yards'] = numeric(frame, 'receiving_yards') - numeric(frame, 'receiving_yards_after_catch')
    return pd.concat([frame, historical_features(raw, frame)], axis=1)


def curve(bundle, frame, stat):
    comp = bundle['model'].components(frame)
    values, weights = bundle['uncertainty'].mixture(player_features(frame, stat), comp['centers'], comp['weights'])
    values[numeric(frame, 'wd_injury_out').eq(1).to_numpy()] = 0.
    return values, weights, comp


def calibration_curve_passes(before, after):
    """Brier alone cannot justify distorting the predictive interval."""
    return bool(after['brier'] < before['brier']
        and after['calibration_error'] <= before['calibration_error']
        and abs(after['coverage_80']-.8) <= abs(before['coverage_80']-.8)
        and after['interval_score'] <= before['interval_score'])


def guard_calibration(calibrator, frame, stat, values, weights):
    if not calibrator.enabled:
        return {'enabled': False, 'reason': 'line_calibration_not_accepted'}
    lines = line_frame(frame, stat, values, weights)
    requests = [lines.loc[lines.player_game.eq(str(r.game_id)+'|'+str(r.player_id)), 'line'].to_numpy()
                for r in frame.itertuples()]
    cv, cw = calibrate_curve(calibrator, frame, stat, values, weights, extra_lines=requests)
    actual = numeric(frame, stat).to_numpy()
    def measure(v, w):
        summary = curve_summary(v, w)
        scored = line_frame(frame, stat, v, w)
        return dict(probability_metrics(scored.probability, scored.outcome, scored.weight),
            coverage_80=float(np.mean((actual >= summary['p10']) & (actual <= summary['p90']))),
            interval_score=float(np.mean(interval_score(actual, summary['p10'], summary['p90']))))
    before, after = measure(values, weights), measure(cv, cw)
    calibrator.enabled = calibration_curve_passes(before, after)
    audit = dict(enabled=calibrator.enabled, before=before, after=after,
                 reason='accepted' if calibrator.enabled else 'full_curve_calibration_gate_failed')
    calibrator.validation['full_curve_guard'] = audit
    return audit


def fit_bundle(train, stat, model_factory=WorkloadDepthModel, uncertainty_factory=ConditionalResidual, refit=True):
    early, scale, residual, cal_fit, cal_gate = inner_partitions(train)
    model = model_factory().fit(early, stat)
    trials = {}
    for w in (0., .25, .5, 1.):
        model.rate_weight = w
        trials[w] = projection_metrics(numeric(scale, stat), model.predict(scale))['rmse']
    model.rate_weight = min(trials, key=trials.get)
    c = model.components(scale)
    idx = states(scale, stat)
    center = (c['centers'] * c['weights']).sum(axis=1)
    uncertainty = uncertainty_factory().fit_scale(player_features(scale, stat),
        numeric(scale, stat).to_numpy() - c['centers'][np.arange(len(scale)), idx], center, FLOORS[stat],
        confidence_errors=numeric(scale, stat).to_numpy() - center)
    c = model.components(residual)
    idx = states(residual, stat)
    uncertainty.fit_residuals(player_features(residual, stat),
        numeric(residual, stat).to_numpy() - c['centers'][np.arange(len(residual)), idx], idx)
    provisional = dict(model=model, uncertainty=uncertainty)
    v, w, _ = curve(provisional, cal_fit, stat)
    calibrator = HierarchicalCalibration().fit(line_frame(cal_fit, stat, v, w))
    v, w, _ = curve(provisional, cal_gate, stat)
    calibrator.validate(line_frame(cal_gate, stat, v, w))
    if not refit:
        guard_calibration(calibrator, cal_gate, stat, v, w)
    ref = ReferenceRecipe().fit(early, stat)
    errors = numeric(residual, stat).to_numpy() - ref.predict(residual)
    base_errors = numeric(residual, stat).to_numpy() - ref.base(residual)
    return dict(model=model_factory(model.rate_weight).fit(train, stat) if refit else model, uncertainty=uncertainty,
        calibrator=calibrator, reference=ReferenceRecipe().fit(train, stat) if refit else ref,
        reference_errors=np.quantile(errors, np.linspace(.005, .995, 101)),
        baseline_errors=np.quantile(base_errors, np.linspace(.005, .995, 101)),
        training_end=str(train.game_date_et.max()), model_fit_end=str((train if refit else early).game_date_et.max()),
        rate_weight=model.rate_weight,
        inner_rate_weight_rmse=trials)


def component_metrics(rows, frame, stat):
    lag = frame[['game_id', 'player_id', f'{stat}_avg_5', f'{STATS[stat]}_avg_5',
                 'wd_depth_sum_5', 'wd_depth_exposure_5']].drop_duplicates(['game_id', 'player_id'])
    d = rows.merge(lag, on=['game_id', 'player_id'], validate='one_to_one')
    actual = d.actual / d.actual_opportunity.replace(0, np.nan)
    baseline = d[f'{stat}_avg_5'] / d[f'{STATS[stat]}_avg_5'].replace(0, np.nan)
    valid = np.isfinite(actual) & np.isfinite(d.predicted_rate) & np.isfinite(baseline)
    report = {'efficiency': dict(model=projection_metrics(actual[valid], d.loc[valid, 'predicted_rate']),
        baseline=projection_metrics(actual[valid], baseline[valid]), excluded_unpaired_rows=int((~valid).sum()),
        interpretation='Paired rows, conditional on positive realized opportunity; not a feature or selection filter.')}
    if 'actual_depth' in d:
        baseline = d.wd_depth_sum_5 / d.wd_depth_exposure_5.replace(0, np.nan)
        valid = np.isfinite(d.actual_depth) & np.isfinite(d.predicted_depth) & np.isfinite(baseline)
        report['target_depth'] = projection_metrics(d.loc[valid, 'actual_depth'], d.loc[valid, 'predicted_depth'])
        report['target_depth_baseline'] = projection_metrics(d.loc[valid, 'actual_depth'], baseline[valid])
        report['target_depth_excluded_unpaired_rows'] = int((~valid).sum())
    return report


def evaluate(frame, stat, seasons, model_factory=WorkloadDepthModel, uncertainty_factory=ConditionalResidual, refit=True):
    frame = frame.loc[frame.position.isin(POSITIONS[stat]) & numeric(frame, stat).notna()
                      & numeric(frame, STATS[stat]).notna()].copy()
    row_parts = []; line_parts = []; folds = []
    for name, train, test in expanding_week_folds(frame, seasons):
        b = fit_bundle(train, stat, model_factory, uncertainty_factory, refit)
        values, weights, comp = curve(b, test, stat)
        summary = curve_summary(values, weights)
        ref = b['reference'].predict(test); base = b['reference'].base(test)
        actual = numeric(test, stat).to_numpy()
        rows = test[['game_id', 'player_id', 'season', 'week', 'position', 'game_date_et']].reset_index(drop=True)
        rows = rows.assign(stat=stat, fold=name, actual=actual, reference=ref, baseline=base,
            challenger=summary['mean'], median=summary['median'], p10=summary['p10'], p90=summary['p90'],
            predicted_opportunity=(comp['opportunities'] * comp['weights']).sum(axis=1),
            actual_opportunity=numeric(test, STATS[stat]).to_numpy(),
            baseline_opportunity=numeric(test, f'{STATS[stat]}_avg_5').to_numpy(),
            low_probability=comp['weights'][:, 0], normal_probability=comp['weights'][:, 1],
            high_probability=comp['weights'][:, 2], actual_state=states(test, stat), predicted_rate=comp['rate'])
        if stat == 'receiving_yards':
            rows['predicted_depth'] = comp['depth']
            rows['actual_depth'] = (numeric(test, 'receiving_air_yards') / numeric(test, 'targets').replace(0, np.nan)).to_numpy()
        lines = line_frame(test, stat, values, weights)
        requested = [lines.loc[lines.player_game.eq(str(r.game_id)+'|'+str(r.player_id)), 'line'].to_numpy()
                     for r in test.itertuples()]
        cv, cw = calibrate_curve(b['calibrator'], test, stat, values, weights, extra_lines=requested)
        calibrated_summary = curve_summary(cv, cw)
        rows['raw_p10'] = rows.p10; rows['raw_p90'] = rows.p90
        rows['p10'] = calibrated_summary['p10']; rows['p90'] = calibrated_summary['p90']
        equal = np.ones((len(test), 101)) / 101
        lines['reference_probability'] = line_frame(test, stat, ref[:, None] + b['reference_errors'], equal).probability.to_numpy()
        lines['baseline_probability'] = line_frame(test, stat, base[:, None] + b['baseline_errors'], equal).probability.to_numpy()
        lines['calibrated_probability'] = line_frame(test, stat, cv, cw).probability.to_numpy()
        lines['fold'] = name; lines['stat'] = stat
        row_parts.append(rows); line_parts.append(lines)
        folds.append(dict(fold=name, train_end=b['training_end'], test_start=str(test.game_date_et.min()),
            train_rows=len(train), test_rows=len(test), rate_weight=b['rate_weight'], calibration=b['calibrator'].validation))
        log.info('%s %s RMSE %.3f reference %.3f', stat, name,
                 projection_metrics(actual, summary['mean'])['rmse'], projection_metrics(actual, ref)['rmse'])
    rows = pd.concat(row_parts, ignore_index=True); lines = pd.concat(line_parts, ignore_index=True)
    report = dict(projection=projection_metrics(rows.actual, rows.challenger, rows.p10, rows.p90),
        reference=projection_metrics(rows.actual, rows.reference), baseline=projection_metrics(rows.actual, rows.baseline),
        median=projection_metrics(rows.actual, rows['median']),
        opportunity=projection_metrics(rows.actual_opportunity, rows.predicted_opportunity),
        baseline_opportunity=projection_metrics(rows.actual_opportunity, rows.baseline_opportunity),
        workload_states={name: probability_metrics(rows[col], rows.actual_state.eq(i))
                         for i, (name, col) in enumerate((('low', 'low_probability'), ('normal', 'normal_probability'), ('high', 'high_probability')))},
        probabilities={col: probability_metrics(lines[col], lines.outcome, lines.weight)
                       for col in ('probability', 'calibrated_probability', 'reference_probability', 'baseline_probability')},
        output_gates=output_gates(rows, lines, 'calibrated_probability'), folds=folds,
        by_workload={str(k): projection_metrics(g.actual, g.challenger) for k, g in rows.groupby('actual_state')},
        context_coverage={c: float(frame[c].notna().mean()) for c in ('wd_depth', 'wd_starter', 'wd_injury_out', 'wd_teammate_absences')},
        automatic_promotion=False, bankroll_approval=False)
    report['interval_contract'] = 'p10/p90 follow the calibrated probability curve; mean and median are tested as separate raw-distribution outputs.'
    report['allocation_contract'] = ('Player opportunity is bounded by predicted team volume, independently of other scored players. '
        'This is not a complete-roster allocation model; absent verified pregame rosters, no participant-based normalization is allowed.')
    report.update(component_metrics(rows, frame, stat))
    return report, rows, lines, fit_bundle(frame, stat, model_factory, uncertainty_factory, refit)


def run(seasons=(2024, 2025, 2026), through=None, cache=None):
    before = active_release()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = 'workload-depth-' + stamp
    path = MODEL_ROOT / 'workload_depth' / run_id
    path.mkdir(parents=True, exist_ok=False)
    if cache:
        frame = joblib.load(cache)
        if not {'wd_history', 'team_actual_pass_attempts', 'team_actual_carries'}.issubset(frame):
            raise ValueError('Not a workload-depth player-game cache')
    else:
        players, _ = load_training(); raw = load_role_history()
        if through:
            players = players.loc[pd.to_datetime(players.game_date_et).le(pd.Timestamp(through))]
        frame = prepare(players, raw)
    if through:
        frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through))].copy()
    atomic_joblib(path / 'training.joblib', frame)
    report = dict(run_id=run_id, contract=CONTRACT, production_release=before['release_id'], training_end=str(frame.game_date_et.max()),
                  status='offline_challenger', historical_source='corrected final player-games, date-lagged; not a lock-time replay', stats={})
    models = {}
    for stat in STATS:
        result, rows, lines, bundle = evaluate(frame, stat, seasons)
        report['stats'][stat] = result; models[stat] = bundle
        atomic_joblib(path / (stat + '_oof.joblib'), dict(rows=rows, lines=lines))
        atomic_json(path / 'report.json', clean(report))
    artifact = dict(run_id=run_id, contract=CONTRACT, production_release=before['release_id'], training_end=report['training_end'], models=models)
    atomic_joblib(path / 'models.joblib', artifact)
    if active_release() != before:
        raise RuntimeError('Production release changed during experiment; do not publish this comparison')
    atomic_json(MODEL_ROOT / 'workload_depth' / 'latest.json', dict(run_id=run_id, sha256=hashlib.sha256((path / 'models.joblib').read_bytes()).hexdigest()))
    atomic_json(ROOT / 'reports' / 'nfl_workload_depth_latest.json', clean(report))
    write_markdown(report)
    return report


def write_markdown(report):
    text = ['# NFL Workload and Target Depth', '', 'Offline only. Production and betting gates unchanged.',
            'Historical proxy lines are not real offered-line betting proof.',
            'Reference: the conservative architecture refitted before each fold, not hindsight scoring with the current live artifact.',
            '', '## Point Forecasts', '',
            '| Stat | Mean RMSE | Reference RMSE | Median MAE | Reference MAE |', '|---|---:|---:|---:|---:|']
    for stat, r in report['stats'].items():
        text.append(f"| {stat} | {r['projection']['rmse']:.3f} | {r['reference']['rmse']:.3f} | {r['median']['mae']:.3f} | {r['reference']['mae']:.3f} |")
    text += ['', '## Probability Curves', '',
             '| Stat | Brier | Reference Brier | Nominal 80% Coverage | Probability Gate |', '|---|---:|---:|---:|---|']
    for stat, r in report['stats'].items():
        text.append(f"| {stat} | {r['probabilities']['calibrated_probability']['brier']:.4f} | {r['probabilities']['reference_probability']['brier']:.4f} | {r['projection']['coverage_80']:.1%} | {r['output_gates']['probability']['pass']} |")
    text += ['', '## Component Checks']
    for stat, r in report['stats'].items():
        text += ['', f"{stat} output gates: " + json.dumps({k: v['pass'] for k, v in r['output_gates'].items() if isinstance(v, dict) and 'pass' in v}),
                 f"{stat} verified historical context coverage: " + json.dumps(r['context_coverage'])]
        e = r.get('efficiency', {})
        if e.get('model', {}).get('rows'):
            text += [f"Paired efficiency MAE: {e['model']['mae']:.3f} versus {e['baseline']['mae']:.3f}; {e['model']['rows']} player-games."]
        if r.get('target_depth', {}).get('rows'):
            text += [f"Paired target-depth MAE: {r['target_depth']['mae']:.3f} versus {r['target_depth_baseline']['mae']:.3f}."]
    text += ['', 'Missing availability stays unknown. No normalization over offered players or actual participants.',
             'Historical point-forecast approval is separate from probability approval and does not enable real-money betting.']
    (ROOT / 'reports' / 'nfl_workload_depth_latest.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def refresh_diagnostics():
    """Recompute paired component metrics from saved OOF rows, never refit models."""
    pointer = json.loads((MODEL_ROOT / 'workload_depth' / 'latest.json').read_text())
    path = MODEL_ROOT / 'workload_depth' / pointer['run_id']
    report = json.loads((path / 'report.json').read_text())
    if report.get('contract') != CONTRACT:
        raise ValueError('Workload contract mismatch; retraining is required')
    frame = joblib.load(path / 'training.joblib')
    for stat, r in report['stats'].items():
        rows = joblib.load(path / (stat + '_oof.joblib'))['rows']
        r.update(component_metrics(rows, frame, stat))
    atomic_json(path / 'report.json', clean(report))
    atomic_json(ROOT / 'reports' / 'nfl_workload_depth_latest.json', clean(report))
    write_markdown(report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seasons', default='2024,2025,2026'); p.add_argument('--through'); p.add_argument('--cache')
    p.add_argument('--refresh-diagnostics', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    r = refresh_diagnostics() if args.refresh_diagnostics else run(tuple(map(int, args.seasons.split(','))), args.through, args.cache)
    print(json.dumps({'run_id': r['run_id'], 'production_changed': False}))
