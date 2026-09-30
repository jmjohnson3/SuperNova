"""Chronological receiving recency repair; publishes challenger artifacts only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import warnings

import numpy as np
import pandas as pd

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_joblib, atomic_json, freeze_production
from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual, HierarchicalCalibration, curve_summary, numeric)
from nfl_pipeline.modeling.component_validation import calibrate_curve, output_gates
from nfl_pipeline.modeling.evaluation import expanding_week_folds, inner_partitions, projection_metrics, probability_metrics
from nfl_pipeline.modeling.receiver_role_model import (
    CONTRACT, ROLES, RoleConfig, RoleAwareReceiver, code_fingerprint, inputs, load_role_history, role_features)
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, ROOT, line_frame, load_training

log = logging.getLogger(__name__)


def variant(frame, variants, weight):
    return variants[weight].loc[frame.index].copy()


def probability_frame(frame):
    # Keep offered/proxy lines fixed, but update workload calibration context too.
    return frame.assign(targets_avg_5=numeric(frame, 'rr_targets_5'))


def tune_config(early, variants):
    weeks = early.groupby(['season', 'week']).game_date_et.min().sort_values()
    cutoff = weeks.iloc[-4]
    fit = early.loc[early.game_date_et < cutoff]
    check = early.loc[early.game_date_et >= cutoff]
    if len(fit) < 100 or check.empty:
        raise ValueError('Insufficient inner weeks for role-weight selection')
    trials = []
    for weight in (1., .25):
        for strength in (30., 90.):
            for recent in (.25, .75):
                config = RoleConfig(weight, strength, recent)
                model = RoleAwareReceiver(config).fit(variant(fit, variants, weight))
                held = variant(check, variants, weight)
                comp = model.components(held)
                score = projection_metrics(held.receiving_yards, comp['center'])
                op = projection_metrics(held.targets, comp['targets'])
                # Fixed loss, selected on inner weeks only. No outer results choose weights.
                loss = score['mae'] + .25 * score['rmse'] + 3 * op['mae']
                trials.append({'config': asdict(config), 'loss': loss, 'yards': score, 'targets': op})
    best = min(trials, key=lambda r: r['loss'])
    return RoleConfig(**best['config']), {'fit_end': str(fit.game_date_et.max()),
        'selection_start': str(check.game_date_et.min()), 'trials': trials, 'selected': best['config']}


def curve(bundle, frame, *, calibrated=True, books=None, lines=None):
    center = bundle['model'].predict(frame)
    values, masses = bundle['uncertainty'].mixture(inputs(frame, 'all'), center, np.ones((len(frame), 1)))
    if calibrated:
        if lines is None:
            prior = numeric(frame, 'receiving_yards_avg_5').fillna(0).to_numpy()
            lines = np.maximum(.5, np.floor(prior[:, None] * np.array([.75, 1., 1.25])) + .5)
        values, masses = calibrate_curve(bundle['calibrator'], probability_frame(frame), 'receiving_yards', values, masses,
            books=books, extra_lines=lines)
    return values, masses


def fit_bundle(train, variants):
    early, scale, residual, cal_fit, cal_gate = inner_partitions(train)
    config, tuning = tune_config(early, variants)
    early, scale, residual, cal_fit, cal_gate = [variant(f, variants, config.latest_weight)
        for f in (early, scale, residual, cal_fit, cal_gate)]
    model = RoleAwareReceiver(config).fit(early)
    center = model.predict(scale)
    uncertainty = ConditionalResidual().fit_scale(inputs(scale, 'all'), scale.receiving_yards.to_numpy() - center, center, 5.)
    uncertainty.fit_residuals(inputs(residual, 'all'), residual.receiving_yards.to_numpy() - model.predict(residual))
    provisional = {'model': model, 'uncertainty': uncertainty}
    v, w = curve(provisional, cal_fit, calibrated=False)
    calibrator = HierarchicalCalibration().fit(line_frame(probability_frame(cal_fit), 'receiving_yards', v, w))
    v, w = curve(provisional, cal_gate, calibrated=False)
    calibrator.validate(line_frame(probability_frame(cal_gate), 'receiving_yards', v, w))
    # The historical comparator is refit on earlier rows, not today's artifact.
    ref = ReferenceRecipe().fit(early, 'receiving_yards')
    ref_error = residual.receiving_yards.to_numpy() - ref.predict(residual)
    base_error = residual.receiving_yards.to_numpy() - ref.base(residual)
    final = variant(train, variants, config.latest_weight)
    return {'model': RoleAwareReceiver(config).fit(final), 'uncertainty': uncertainty,
        'calibrator': calibrator, 'reference': ReferenceRecipe().fit(train, 'receiving_yards'),
        'reference_errors': np.quantile(ref_error, np.linspace(.005, .995, 101)),
        'baseline_errors': np.quantile(base_error, np.linspace(.005, .995, 101)),
        'tuning': tuning, 'config': asdict(config), 'training_end': str(train.game_date_et.max())}


def evaluate(data, variants, seasons):
    rows = []; lines = []; folds = []
    for name, train, test in expanding_week_folds(data, seasons):
        log.info('Receiving role %s: %s train, %s test', name, len(train), len(test))
        bundle = fit_bundle(train, variants)
        test = variant(test, variants, bundle['config']['latest_weight'])
        v, w = curve(bundle, test)
        summary = curve_summary(v, w)
        comp = bundle['model'].components(test)
        reference = bundle['reference'].predict(test)
        baseline = bundle['reference'].base(test)
        out = test[['game_id', 'player_id', 'game_date_et', 'season', 'week', 'position']].copy()
        out = out.assign(actual=test.receiving_yards, reference=reference, baseline=baseline,
            challenger=summary['mean'], median=summary['median'], p10=summary['p10'], p90=summary['p90'],
            targets=comp['targets'], actual_targets=test.targets,
            ypt=comp['yards_per_target'], fold=name)
        probabilities = line_frame(probability_frame(test), 'receiving_yards', v, w)
        equal = np.full((len(test), 101), 1 / 101)
        probabilities['reference_probability'] = line_frame(test, 'receiving_yards',
            reference[:, None] + bundle['reference_errors'], equal).probability.to_numpy()
        probabilities['baseline_probability'] = line_frame(test, 'receiving_yards',
            baseline[:, None] + bundle['baseline_errors'], equal).probability.to_numpy()
        probabilities['fold'] = name
        gates = output_gates(out, probabilities, 'probability')
        folds.append({'fold': name, 'train_end': bundle['training_end'],
            'test_start': str(test.game_date_et.min()), 'rows': len(test), 'config': bundle['config'],
            'selection': bundle['tuning'], 'calibration': bundle['calibrator'].validation,
            'metrics': gates})
        rows.append(out); lines.append(probabilities)
        log.info('%s MAE %.3f vs %.3f; Brier %.4f vs %.4f', name,
            gates['median']['metrics']['mae'], gates['expected_mean']['reference']['mae'],
            gates['probability']['metrics']['brier'], gates['probability']['reference']['brier'])
    if not rows:
        raise ValueError('No chronological evaluation folds')
    out = pd.concat(rows, ignore_index=True); probabilities = pd.concat(lines, ignore_index=True)
    gates = output_gates(out, probabilities, 'probability')
    baseline_rows = out.assign(reference=out.baseline)
    baseline_lines = probabilities.assign(reference_probability=probabilities.baseline_probability)
    baseline_gates = output_gates(baseline_rows, baseline_lines, 'probability')
    seasons_report = {}
    for season, group in out.groupby('season'):
        p = probabilities.loc[probabilities.season == season]
        seasons_report[str(season)] = {'median': projection_metrics(group.actual, group['median']),
            'reference': projection_metrics(group.actual, group.reference),
            'probability': probability_metrics(p.probability, p.outcome, p.weight),
            'reference_probability': probability_metrics(p.reference_probability, p.outcome, p.weight)}
    return {'vs_reference': gates, 'vs_baseline': baseline_gates, 'seasons': seasons_report,
        'opportunity': projection_metrics(out.actual_targets, out.targets), 'folds': folds,
        'historical_pass': all(g[k]['pass'] for g in (gates, baseline_gates) for k in ('expected_mean', 'median', 'probability'))}, out, probabilities


def write_report(report):
    root = ROOT / 'reports'
    atomic_json(root / 'nfl_receiver_role_latest.json', clean(report))
    g = report['evaluation']['vs_reference']
    lines = ['# NFL Role-Aware Receiving Challenger', '',
        f"Frozen production: {report['production_release']}",
        f"Challenger: {report['run_id']}", '',
        '**Production unchanged. Historical proxy-line results are not betting proof.**', '',
        '| Output | Challenger | Chronological reference | Passed |', '|---|---:|---:|---|',
        f"| Mean RMSE | {g['expected_mean']['metrics']['rmse']:.3f} | {g['expected_mean']['reference']['rmse']:.3f} | {g['expected_mean']['pass']} |",
        f"| Median MAE | {g['median']['metrics']['mae']:.3f} | {g['expected_mean']['reference']['mae']:.3f} | {g['median']['pass']} |",
        f"| Line Brier | {g['probability']['metrics']['brier']:.4f} | {g['probability']['reference']['brier']:.4f} | {g['probability']['pass']} |",
        f"| Calibration error | {g['probability']['metrics']['calibration_error']:.2%} | {g['probability']['reference']['calibration_error']:.2%} | |", '',
        f"80% interval coverage: {g['probability']['coverage_80']:.1%}",
        f"All historical gates versus reference AND baseline: {report['evaluation']['historical_pass']}",
        f"Latest inner-selected weights: {report['final_config']}", '', '## Interpretation', '',
        *['- ' + value for value in report['limitations']]]
    (root / 'nfl_receiver_role_latest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(seasons=(2024, 2025, 2026), through=None):
    frozen = freeze_production('Role-aware receiving validation; do not publish')
    active_bytes = (MODEL_ROOT / 'active_release.json').read_bytes()
    log.info('Loading final player-games and exposure history')
    players, _ = load_training()
    raw = load_role_history()
    if through:
        players = players.loc[players.game_date_et <= through]
        raw = raw.loc[raw.game_date_et <= through]
    players = players.loc[players.position.isin(ROLES) & numeric(players, 'targets').notna()
        & numeric(players, 'receiving_yards').notna()].copy().reset_index(drop=True)
    starts = raw[['game_id', 'start_ts_utc']].drop_duplicates('game_id')
    players = players.drop(columns=['start_ts_utc'], errors='ignore').merge(starts, on='game_id', validate='many_to_one')
    team = raw.groupby(['game_id', 'team_abbr']).pass_attempts.sum(min_count=1).rename('team_actual_pass_attempts').reset_index()
    players = players.merge(team, on=['game_id', 'team_abbr'], validate='many_to_one')
    variants = {}
    for weight in (1., .25):
        log.info('Building pregame role history, last-game weight=%s (%s rows)', weight, len(players))
        variants[weight] = pd.concat([players, role_features(raw, players, weight)], axis=1)
    data = variants[1.]
    evaluation, rows, lines = evaluate(data, variants, seasons)
    log.info('Fitting final isolated challenger')
    bundle = fit_bundle(data, variants)
    run_id = datetime.now(timezone.utc).strftime('receiver-role-%Y%m%dT%H%M%S%fZ')
    report = {'contract': CONTRACT, 'run_id': run_id, 'code_fingerprint': code_fingerprint(),
        'production_release': frozen['release_id'],
        'training_end': str(data.game_date_et.max()), 'evaluation': evaluation,
        'final_config': bundle['config'], 'production_changed': False,
        'context_coverage': {c: float(data[c].notna().mean()) for c in ('rr_depth', 'rr_injury_out', 'rr_teammate_absences')},
        'limitations': [
            'Corrected historical results are used; historical revisions are not claimed to be immutable lock-time snapshots.',
            'Reference architecture is refit chronologically; today\'s frozen model is not used on its own past training outcomes.',
            'Historical tests use unpriced proxy lines, not executable prices or ROI.',
            'Complete live scoring replay is a separate diagnostic on captured offers; historical proxy Brier does not validate uncaptured live overlays.',
            'Latest-game weight, recent-role blend, and prior strength are selected on inner weeks only.',
            'Low snap counts are limited-appearance proxies, not verified injury diagnoses.',
            'Targets weight efficiency evidence; zero-target games affect workload but supply no YPT observation.',
            'Unknown injury, depth and teammate reports stay missing. Sparse historical context cannot establish the value of news features.',
            'Team pass budget constrains each player; a full current-roster joint allocation remains a separate challenger.',
            'Previously examined seasons remain research evidence. Prospective locked comparisons are required before publication.',
            'No bankroll/micro gates, forecasts, Discord picks, or active artifacts were changed.']}
    root = MODEL_ROOT / 'receiver_role' / run_id
    artifact = {'contract': CONTRACT, 'run_id': run_id, 'code_fingerprint': code_fingerprint(),
        'production_release': frozen['release_id'],
        'training_end': report['training_end'], 'bundle': bundle, 'historical_pass': evaluation['historical_pass'],
        'status': 'challenger_only', 'betting_eligible': False}
    atomic_joblib(root / 'models.joblib', artifact)
    atomic_joblib(root / 'oof.joblib', {'player_games': rows, 'proxy_lines': lines})
    atomic_json(root / 'report.json', clean(report))
    atomic_json(MODEL_ROOT / 'receiver_role' / 'latest.json', {'run_id': run_id,
        'production_release': frozen['release_id'], 'sha256': hashlib.sha256((root / 'models.joblib').read_bytes()).hexdigest()})
    write_report(report)
    if active_bytes != (MODEL_ROOT / 'active_release.json').read_bytes():
        raise RuntimeError('Production changed during validation')
    return {'run_id': run_id, 'historical_pass': evaluation['historical_pass'], 'production_changed': False}


if __name__ == '__main__':
    from datetime import date
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons', default='2024,2025,2026')
    parser.add_argument('--through', type=date.fromisoformat)
    args = parser.parse_args()
    from nfl_pipeline.modeling.train_receiver_role import run as train
    print(json.dumps(train(tuple(map(int, args.seasons.split(','))), args.through), indent=2))
