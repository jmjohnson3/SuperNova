"""Nested, player-game yardage tournament. Never replaces production artifacts."""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_joblib, atomic_json
from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual, FittedHead, OpportunityRateModel, FLOORS, OPPORTUNITY, POSITIONS,
    curve_summary, numeric, player_features, workload_state)
from nfl_pipeline.modeling.evaluation import (
    clustered_gain, expanding_week_folds, probability_metrics, projection_metrics)
from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, ROOT, line_frame
from nfl_pipeline.modeling.workload_depth_model import WorkloadDepthModel, states
from nfl_pipeline.modeling.sample_aware_splits import sample_aware_partitions, describe_blocks
from nfl_pipeline.modeling.qb_tail_distribution import QBTailDistribution

CONTRACT = 'nfl-shared-yardage-benchmark-v2'
SUPPORTED_CONTRACTS = {CONTRACT, 'nfl-shared-yardage-benchmark-v1'}
LOG = logging.getLogger(__name__)


class SimplePoint:
    def __init__(self, stat, kind):
        self.stat, self.kind = stat, kind

    def fit(self, frame):
        op = OPPORTUNITY[self.stat]
        totals = frame.groupby('position')[[self.stat, op]].sum()
        self.rates = (totals[self.stat] / totals[op].replace(0, np.nan)).to_dict()
        self.global_rate = float(frame[self.stat].sum() / max(1., frame[op].sum()))
        return self

    def predict(self, frame):
        if self.kind == 'rolling':
            return numeric(frame, self.stat+'_avg_5').fillna(0).to_numpy().clip(0)
        op = OPPORTUNITY[self.stat]
        prior = frame.position.map(self.rates).fillna(self.global_rate)
        n = numeric(frame, 'n_games_prev_5').fillna(0).clip(0, 5)
        exposure = numeric(frame, op+'_avg_5').fillna(0).clip(0)*n
        yards = numeric(frame, self.stat+'_avg_5').fillna(0)*n
        rate = (yards + 40*prior)/(exposure+40)
        return (numeric(frame, op+'_avg_5').fillna(0).clip(0)*rate).to_numpy()


class BoostedMean:
    def fit(self, frame, stat):
        self.stat = stat
        self.head = FittedHead().fit(player_features(frame, stat), numeric(frame, stat).to_numpy())
        return self

    def predict(self, frame):
        return self.head.predict(player_features(frame, self.stat))


def fit_uncertainty(model, scale, residual, stat, mixture=False, depth=False):
    def centers(frame):
        if not mixture:
            point = model.predict(frame)
            return point, point, None
        comp = model.components(frame)
        labels = states(frame, stat) if depth else workload_state(frame, stat)
        return (comp['centers'][np.arange(len(frame)), labels],
                (comp['centers']*comp['weights']).sum(axis=1), labels)
    conditional, expected, _ = centers(scale)
    u = ConditionalResidual().fit_scale(player_features(scale, stat), numeric(scale, stat)-conditional,
        expected, FLOORS[stat], confidence_errors=numeric(scale, stat)-expected)
    conditional, _, labels = centers(residual)
    return u.fit_residuals(player_features(residual, stat), numeric(residual, stat)-conditional, labels)


def raw_curve(candidate, frame, stat):
    if candidate['kind'] == 'qb_tail':
        v, w = raw_curve(candidate['base'], frame, stat)
        return candidate['tail'].transform(frame, v, w)
    if candidate['kind'] == 'ensemble':
        curves = [raw_curve(m, frame, stat) for m in candidate['members']]
        return np.concatenate([c[0] for c in curves], axis=1), np.concatenate([c[1]/len(curves) for c in curves], axis=1)
    model = candidate['model']
    if candidate['kind'] == 'pooled':
        v = model.predict(frame)[:, None] + candidate['errors'][None, :]
        return v, np.full(v.shape, 1/v.shape[1])
    if candidate['kind'] == 'conditional':
        center = model.predict(frame)
        return candidate['uncertainty'].mixture(player_features(frame, stat), center, np.ones((len(frame), 1)))
    comp = model.components(frame)
    v, w = candidate['uncertainty'].mixture(player_features(frame, stat), comp['centers'], comp['weights'])
    if candidate.get('needs_workload_history'):
        v[numeric(frame, 'wd_injury_out').eq(1).to_numpy()] = 0.
    return v, w


def candidate_curve(candidate, frame, stat, calibrated=False):
    v, w = raw_curve(candidate, frame, stat)
    return candidate['calibrator'].transform(v, w) if calibrated else (v, w)


def curve_metrics(frame, stat, v, w):
    s = curve_summary(v, w)
    lines = line_frame(frame, stat, v, w)
    y = numeric(frame, stat).to_numpy()
    interval_score = s['p90']-s['p10'] + 10*np.maximum(0, s['p10']-y) + 10*np.maximum(0, y-s['p90'])
    return dict(expected=projection_metrics(y, s['mean'], s['p10'], s['p90']),
                typical=projection_metrics(y, s['median']),
                probability=probability_metrics(lines.probability, lines.outcome, lines.weight),
                interval_score=float(np.mean(interval_score)))


def select_outputs(scores):
    """All scores must be from the inner, pre-test selection block."""
    mean = min(scores, key=lambda k: (scores[k]['expected']['rmse'], abs(scores[k]['expected']['bias']), k))
    median = min(scores, key=lambda k: (scores[k]['typical']['mae'], k))
    eligible = {k: s for k, s in scores.items()
                if .75 <= s['expected']['coverage_80'] <= .85}
    probability = min(eligible, key=lambda k: (eligible[k]['probability']['brier'],
                      eligible[k]['probability']['calibration_error'], k)) if eligible else None
    return dict(expected=mean, typical=median, probability=probability,
                probability_fallback='reference' if probability is None else None)


def fit_points(frame, stat):
    return {
        'rolling': SimplePoint(stat, 'rolling').fit(frame),
        'shrunk_rate': SimplePoint(stat, 'rate').fit(frame),
        'reference': ReferenceRecipe().fit(frame, stat),
        'boosted_mean': BoostedMean().fit(frame, stat),
    }


def refit_core(candidates, train, stat):
    """Refit before the outer test; calibration-transfer risk is part of OOF loss."""
    for name, model in fit_points(train, stat).items():
        candidates[name]['model'] = model
    candidates['reference_conditional']['model'] = candidates['reference']['model']
    candidates['opportunity_rate']['model'] = OpportunityRateModel().fit(train, stat)
    if stat != 'passing_yards':
        candidates['workload_depth']['model'] = WorkloadDepthModel(rate_weight=.25).fit(train, stat)
    if 'qb_tail' in candidates:
        old = candidates['qb_tail']['tail']
        tail = QBTailDistribution().fit(train)
        tail.strength, tail.validation = old.strength, old.validation
        candidates['qb_tail']['tail'] = tail


def fit_suite(train, stat):
    parts = sample_aware_partitions(train, stat)
    early, scale, residual, cal_fit, cal_tune, gate = parts
    models = fit_points(early, stat)
    candidates = {}
    for name, model in models.items():
        errors = numeric(residual, stat).to_numpy()-model.predict(residual)
        candidates[name] = dict(kind='pooled', model=model, errors=np.quantile(errors, np.linspace(.005, .995, 101)))
    candidates['reference_conditional'] = dict(kind='conditional', model=models['reference'],
        uncertainty=fit_uncertainty(models['reference'], scale, residual, stat))
    model = OpportunityRateModel().fit(early, stat)
    candidates['opportunity_rate'] = dict(kind='mixture', model=model,
        uncertainty=fit_uncertainty(model, scale, residual, stat, mixture=True))
    if stat != 'passing_yards':
        model = WorkloadDepthModel(rate_weight=.25).fit(early, stat)
        candidates['workload_depth'] = dict(kind='mixture', model=model, needs_workload_history=True,
            uncertainty=fit_uncertainty(model, scale, residual, stat, mixture=True, depth=True))
    members = [candidates[k] for k in ('reference_conditional', 'boosted_mean', 'opportunity_rate')]
    candidates['ensemble'] = dict(kind='ensemble', members=members)
    if stat == 'passing_yards':
        tail = QBTailDistribution().fit(early)
        v, w = raw_curve(candidates['ensemble'], cal_fit, stat)
        tail.choose_strength(cal_fit, v, w)
        candidates['qb_tail'] = dict(kind='qb_tail', base=candidates['ensemble'], tail=tail)
    scores = {}
    for name, candidate in candidates.items():
        frames = [line_frame(f, stat, *raw_curve(candidate, f, stat)) for f in (cal_fit, cal_tune, gate)]
        candidate['calibrator'] = TailPreservingCalibration().fit(*frames)
        scores[name] = curve_metrics(gate, stat, *raw_curve(candidate, gate, stat))
        scores[name+'+tail_cal'] = curve_metrics(gate, stat, *candidate_curve(candidate, gate, stat, True))
    refit_core(candidates, train, stat)
    return dict(candidates=candidates, choices=select_outputs(scores), selection_scores=scores,
                model_fit_end=str(train.game_date_et.max()), inner_model_fit_end=str(early.game_date_et.max()),
                training_end=str(train.game_date_et.max()), blocks=describe_blocks(parts),
                fit_policy='refit_core_after_inner_selection_test_complete_transferred_curve',
                prospective_variants=['stable_ensemble', 'stable_ensemble_calibrated'] +
                                     (['qb_tail', 'qb_tail_calibrated'] if stat == 'passing_yards' else []))


def named_curve(suite, frame, stat, name):
    family = name.removesuffix('+tail_cal')
    return candidate_curve(suite['candidates'][family], frame, stat, name.endswith('+tail_cal'))


def score_suite(suite, frame, stat, fold):
    rows = []; lines = []
    for name in suite['selection_scores']:
        v, w = named_curve(suite, frame, stat, name)
        summary = curve_summary(v, w)
        r = frame[['game_id', 'player_id', 'season', 'week', 'game_date_et']].reset_index(drop=True)
        r = r.assign(fold=fold, stat=stat, family=name, actual=numeric(frame, stat).to_numpy(),
                     **summary)
        l = line_frame(frame, stat, v, w).assign(fold=fold, stat=stat, family=name)
        rows.append(r); lines.append(l)
    r = pd.concat(rows, ignore_index=True); l = pd.concat(lines, ignore_index=True)
    # The policy family is selected entirely before this fold's outcomes are observed.
    for output in ('expected', 'typical', 'probability'):
        name = suite['choices'][output]
        selected = name or 'reference'
        rows.append(r.loc[r.family.eq(selected)].assign(family='selected_'+output, chosen_family=selected,
                                                        accepted_for_test=name is not None))
        if output == 'probability':
            lines.append(l.loc[l.family.eq(selected)].assign(family='selected_probability', chosen_family=selected,
                                                             accepted_for_test=name is not None))
    return pd.concat(rows, ignore_index=True), pd.concat(lines, ignore_index=True)


def summarize(rows, lines):
    report = {}
    reference = rows.loc[rows.family.eq('reference')].set_index(['game_id', 'player_id'])
    ref_lines = lines.loc[lines.family.eq('reference')].set_index(['player_game', 'line'])
    for name, r in rows.groupby('family'):
        result = dict(player_games=len(r), expected=projection_metrics(r.actual, r['mean'], r.p10, r.p90),
                      typical=projection_metrics(r.actual, r['median']))
        result['tail_rates'] = dict(below_p10=float((r.actual < r.p10).mean()),
                                   above_p90=float((r.actual > r.p90).mean()))
        result['interval_score'] = float(np.mean(r.p90-r.p10 + 10*np.maximum(0, r.p10-r.actual)
                                                + 10*np.maximum(0, r.actual-r.p90)))
        paired = r.set_index(['game_id', 'player_id']).join(reference[['mean', 'median']], rsuffix='_reference')
        for metric, pred, ref, squared in (
            ('expected_squared_error_gain', 'mean', 'mean_reference', True),
            ('median_absolute_error_gain', 'median', 'median_reference', False)):
            a = abs(paired[pred]-paired.actual); b = abs(paired[ref]-paired.actual)
            result[metric] = clustered_gain(paired.assign(challenger_error=a**2 if squared else a,
                                                          reference_error=b**2 if squared else b))
        lp = lines.loc[lines.family.eq(name)]
        if len(lp):
            result['probability'] = probability_metrics(lp.probability, lp.outcome, lp.weight)
            p = lp.set_index(['player_game', 'line']).join(ref_lines[['probability']], rsuffix='_reference')
            p['challenger_error'] = (p.probability-p.outcome)**2
            p['reference_error'] = (p.probability_reference-p.outcome)**2
            grouped = p.reset_index().groupby(['season', 'week', 'player_game'])[['challenger_error', 'reference_error']].mean().reset_index()
            result['brier_gain'] = clustered_gain(grouped)
        report[name] = result
    return report


def stable_comparison(rows, lines):
    comparisons = {}
    for reference, challenger in (('selected_probability', 'ensemble'),
                                  ('selected_probability', 'ensemble+tail_cal'),
                                  ('ensemble', 'qb_tail'), ('ensemble+tail_cal', 'qb_tail+tail_cal')):
        ref = lines.loc[lines.family.eq(reference)]
        candidate = lines.loc[lines.family.eq(challenger)]
        if candidate.empty:
            continue
        paired = candidate.merge(ref[['player_game', 'line', 'probability']], on=['player_game', 'line'],
                                 suffixes=('', '_reference'), validate='one_to_one')
        paired['challenger_error'] = (paired.probability-paired.outcome)**2
        paired['reference_error'] = (paired.probability_reference-paired.outcome)**2
        grouped = paired.groupby(['season', 'week', 'player_game'])[['challenger_error', 'reference_error']].mean().reset_index()
        by_season = {str(s): probability_metrics(g.probability, g.outcome, g.weight)
                     for s, g in paired.groupby('season')}
        comparisons[challenger+'_vs_'+reference] = dict(brier_gain=clustered_gain(grouped),
            later_fold_probabilities=by_season, comparison_rows=len(paired))
    return comparisons


def complete_calibration_controls(rows, lines):
    """A rejected calibrator is identity, not a missing fold in its leaderboard."""
    frames = []
    for frame in (rows, lines):
        additions = []
        present = set(zip(frame.fold, frame.family))
        for (fold, family), group in frame.groupby(['fold', 'family']):
            if family.startswith('selected_') or family.endswith('+tail_cal'):
                continue
            if (fold, family+'+tail_cal') not in present:
                additions.append(group.assign(family=family+'+tail_cal'))
        frames.append(pd.concat([frame, *additions], ignore_index=True))
    return tuple(frames)


def refresh_saved_report(run_id):
    """Report-only compatibility for identity calibrators; no model or pick changes."""
    path = MODEL_ROOT/'model_benchmark'/run_id
    report = json.loads((path/'report.json').read_text())
    for stat, result in report['stats'].items():
        stored = joblib.load(path/(stat+'_oof.joblib'))
        rows, lines = complete_calibration_controls(stored['rows'], stored['lines'])
        result['families'] = summarize(rows, lines)
        atomic_joblib(path/(stat+'_oof.joblib'), dict(rows=rows, lines=lines))
    atomic_json(path/'report.json', clean(report))
    write_report(report)
    return report


def frame_identity(frame):
    keys = frame[['game_id', 'player_id']].astype(str).sort_values(['game_id', 'player_id'])
    return hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()


def evaluate(frame, stat, seasons):
    frame = frame.loc[frame.position.isin(POSITIONS[stat]) & numeric(frame, stat).notna()
                      & numeric(frame, OPPORTUNITY[stat]).notna()].copy()
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Duplicate player-game training rows')
    rows = []; lines = []; folds = []
    for fold, train, test in expanding_week_folds(frame, seasons):
        suite = fit_suite(train, stat)
        r, l = score_suite(suite, test, stat, fold)
        rows.append(r); lines.append(l)
        folds.append(dict(fold=fold, train_rows=len(train), test_rows=len(test),
                          test_start=str(test.game_date_et.min()), test_end=str(test.game_date_et.max()),
                          train_identity=frame_identity(train), test_identity=frame_identity(test),
                          blocks=suite['blocks'], choices=suite['choices'],
                          fit_policy=suite['fit_policy'], model_fit_end=suite['model_fit_end'],
                          qb_tail=suite['candidates']['qb_tail']['tail'].validation if stat == 'passing_yards' else None,
                          calibration={k: v['calibrator'].validation for k, v in suite['candidates'].items()}))
        LOG.info('%s %s choices=%s', stat, fold, suite['choices'])
    if not rows:
        raise ValueError('No chronological outer folds')
    rows = pd.concat(rows, ignore_index=True); lines = pd.concat(lines, ignore_index=True)
    return dict(folds=folds, families=summarize(rows, lines), stable_comparison=stable_comparison(rows, lines)), rows, lines, fit_suite(frame, stat)


def write_report(report):
    lines = ['# NFL Shared Model Benchmark', '', 'Offline comparison. Production unchanged; no betting approval.',
             'Same outer rows and nested dates for every family. Inner-week winners are evaluated on untouched later weeks.',
             'Reference is a chronological refit of the conservative recipe, not the actual historical deployed artifact.',
             'Probability scores below use proxy lines. Actual offered lines and original micro IDs are evaluated separately.',
             '', '| Stat / family | Mean RMSE | Mean bias | Median MAE | Brier | 80% coverage |',
             '|---|---:|---:|---:|---:|---:|']
    for stat, r in report['stats'].items():
        for name, m in r['families'].items():
            brier = m.get('probability', {}).get('brier')
            lines.append(f"| {stat} / {name} | {m['expected']['rmse']:.3f} | {m['expected']['bias']:+.3f} | "
                         f"{m['typical']['mae']:.3f} | {f'{brier:.4f}' if brier is not None else '-'} | {m['expected']['coverage_80']:.1%} |")
    lines += ['', '## Next Prospective Choices', '']
    for stat, r in report['stats'].items():
        lines += [stat + ': ' + json.dumps(r['prospective_choices']), '']
    lines += ['## Matched Fixed-Ensemble Tests', '',
              '| Stat / comparison | Brier gain | Week-grouped 95% interval |', '|---|---:|---|']
    for stat, r in report['stats'].items():
        for name, comparison in r.get('stable_comparison', {}).items():
            gain = comparison['brier_gain']
            lines.append(f"| {stat} / {name} | {gain.get('mean_gain', 0):+.5f} | "
                         f"{gain.get('lower_95')} to {gain.get('upper_95')} |")
    lines += ['', '## Final Training Blocks', '',
              '| Stat / block | Player-games | Weeks | Regular-season rows | Through |',
              '|---|---:|---:|---:|---|']
    for stat, r in report['stats'].items():
        for block in r.get('final_blocks', []):
            lines.append(f"| {stat} / {block['name']} | {block['rows']} | {block['weeks']} | "
                         f"{block['regular_rows']} | {block['last']} |")
    lines += ['']
    for stat, r in report['stats'].items():
        if r.get('model_fit_end'):
            lines += [stat + ' final core refit through ' + r['model_fit_end'] + '.', '']
    lines += ['Core models are refitted after inner selection, before each outer test. Reported OOF losses include calibration-transfer risk.',
              'Calibration blocks expand by whole weeks until both player-game and regular-season sample requirements pass.',
              'Top rows in a descriptive leaderboard are not unbiased winners. Use selected_* policy results and week-grouped intervals.',
              'Historical context coverage is incomplete. Missing information remains unknown.',
              'Central calibration leaves tail probabilities and the 10th/90th percentiles unchanged. It cannot fix bad raw tails.']
    (ROOT/'reports'/'nfl_model_benchmark_latest.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    atomic_json(ROOT/'reports'/'nfl_model_benchmark_latest.json', clean(report))


def run(cache, seasons=(2024, 2025, 2026), stats=tuple(OPPORTUNITY)):
    before = active_release()
    started = datetime.now(timezone.utc)
    run_id = 'benchmark-'+started.strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT/'model_benchmark'/run_id
    path.mkdir(parents=True, exist_ok=False)
    frame = joblib.load(cache)
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Input cache must have unique player-games')
    report = dict(run_id=run_id, contract=CONTRACT, production_release=before['release_id'],
                  training_end=str(frame.game_date_et.max()), stats={}, automatic_promotion=False)
    models = {}
    for stat in stats:
        result, rows, lines, suite = evaluate(frame, stat, seasons)
        result['prospective_choices'] = suite['choices']
        result['final_blocks'] = suite['blocks']
        result['model_fit_end'] = suite['model_fit_end']
        report['stats'][stat] = result; models[stat] = suite
        atomic_joblib(path/(stat+'_oof.joblib'), dict(rows=rows, lines=lines))
        atomic_json(path/'report.json', clean(report))
    from nfl_pipeline.modeling.receiver_role_model import load_role_history
    history = load_role_history(cutoff=started)
    history = history.loc[pd.to_datetime(history.game_date_et).le(pd.Timestamp(report['training_end']))].copy()
    history['completed_air_yards'] = numeric(history, 'receiving_yards')-numeric(history, 'receiving_yards_after_catch')
    artifact = dict(run_id=run_id, contract=CONTRACT, production_release=before['release_id'],
                    training_end=report['training_end'], created_at=datetime.now(timezone.utc).isoformat(),
                    models=models, history=history)
    atomic_joblib(path/'models.joblib', artifact)
    if active_release() != before:
        raise RuntimeError('Frozen release changed during benchmark')
    atomic_json(MODEL_ROOT/'model_benchmark'/'latest.json', dict(run_id=run_id,
                sha256=hashlib.sha256((path/'models.joblib').read_bytes()).hexdigest()))
    write_report(report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--seasons', nargs='+', type=int, default=[2024, 2025, 2026])
    p.add_argument('--stats', nargs='+', choices=list(OPPORTUNITY), default=list(OPPORTUNITY))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Persist importable class names, never __main__.SimplePoint in joblib files.
    from nfl_pipeline.modeling.model_benchmark import run as benchmark_run
    result = benchmark_run(args.cache, tuple(args.seasons), tuple(args.stats))
    print(json.dumps(dict(run_id=result['run_id'], production_changed=False)))
