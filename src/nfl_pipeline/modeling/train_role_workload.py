"""Isolate workload/tail changes with grouped, expanding chronological tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlb_pipeline.atomic_io import atomic_write_text
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_joblib, atomic_json
from nfl_pipeline.modeling.challenger_models import numeric, curve_summary
from nfl_pipeline.modeling.component_validation import interval_score
from nfl_pipeline.modeling.evaluation import expanding_week_folds, projection_metrics, probability_metrics, clustered_gain
from nfl_pipeline.modeling.role_workload import CONTRACT, STATS, VARIANTS, RoleWorkload, roles, validate
from nfl_pipeline.modeling.sample_aware_splits import sample_aware_partitions, describe_blocks, REQUIREMENTS
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, line_frame, ROOT

LOG = logging.getLogger(__name__)


def fit_bundle(frame, stat):
    parts = sample_aware_partitions(frame, stat, requirements=REQUIREMENTS[stat][:3])
    early, count_tail, efficiency_tail, tune = parts
    reference = ReferenceRecipe().fit(early, stat)
    model = (RoleWorkload().fit(early, stat).fit_count_tails(count_tail)
             .fit_efficiency_tails(efficiency_tail, reference).tune(tune))
    errors = numeric(efficiency_tail, stat).to_numpy() - reference.predict(efficiency_tail)
    bundle = dict(model=model, reference=reference,
        reference_errors=np.quantile(errors, (np.arange(101)+.5)/101),
        blocks=describe_blocks(parts), training_end=str(frame.game_date_et.max()),
        final_fit_end=model.last_end)
    for block, name in zip(bundle['blocks'], ('model_fit', 'count_tails', 'efficiency_tails', 'count_blend_tuning')):
        block['name'] = name
    return bundle


def curves(bundle, frame, projection=None):
    model, reference = bundle['model'], bundle['reference']
    point = reference.predict(frame) if projection is None else np.asarray(projection)
    values = point[:, None]+bundle['reference_errors'][None, :]
    yield 'reference', values, np.full_like(values, 1/values.shape[1]), dict(opportunity=model.baseline(frame))
    for variant in VARIANTS:
        v, w, components = model.yardage_curve(frame, reference, variant, projection=projection)
        yield variant, v, w, components


def summarize(rows, lines):
    scores = {}
    for variant, group in rows.groupby('variant'):
        line = lines.loc[lines.variant.eq(variant)]
        scores[variant] = dict(
            opportunity=projection_metrics(group.actual_opportunity, group.opportunity),
            expected=projection_metrics(group.actual, group['mean'], group.p10, group.p90),
            typical=projection_metrics(group.actual, group['median']),
            probability=probability_metrics(line.probability, line.outcome, line.weight),
            interval_score=float(np.mean(interval_score(group.actual, group.p10, group.p90))),
            below_80=float((group.actual < group.p10).mean()),
            above_80=float((group.actual > group.p90).mean()))
    return scores


def decisions(rows, lines, scores):
    keys = ['game_id', 'player_id', 'season', 'week']
    result = {}
    for variant in VARIANTS:
        result[variant] = {}
        new = rows.loc[rows.variant.eq(variant)]
        gains = {}
        for comparator in ('reference', 'control'):
            pair = rows.loc[rows.variant.eq(comparator)].merge(new, on=keys, validate='one_to_one', suffixes=('_old', '_new'))
            if len(pair) != len(new):
                raise ValueError('Workload comparisons need identical player-games')
            outputs = {}
            for name, column, actual, power in (('opportunity', 'opportunity', 'actual_opportunity', 2),
                    ('expected_yards', 'mean', 'actual', 2), ('typical_yards', 'median', 'actual', 1)):
                outputs[name] = clustered_gain(pair.assign(
                    reference_error=abs(pair[column+'_old']-pair[actual+'_old'])**power,
                    challenger_error=abs(pair[column+'_new']-pair[actual+'_new'])**power))
            l = lines.loc[lines.variant.eq(comparator)].merge(lines.loc[lines.variant.eq(variant)],
                on=['season', 'week', 'player_game', 'line'], validate='one_to_one', suffixes=('_old', '_new'))
            l = l.assign(reference_error=(l.probability_old-l.outcome_old)**2,
                         challenger_error=(l.probability_new-l.outcome_new)**2)
            outputs['probability'] = clustered_gain(l.groupby(['season', 'week', 'player_game'])[
                ['reference_error', 'challenger_error']].mean().reset_index())
            gains[comparator] = outputs
        for output in ('opportunity', 'expected_yards', 'typical_yards', 'probability'):
            passed = variant != 'control' and all(g[output].get('lower_95') is not None and g[output]['lower_95'] > 0
                for g in gains.values())
            a = scores[variant]
            if output in ('expected_yards', 'probability'):
                for reference in ('reference', 'control'):
                    b = scores[reference]
                    if output == 'expected_yards':
                        passed &= abs(a['expected']['bias']) <= abs(b['expected']['bias'])
                    else:
                        passed &= (a['probability']['calibration_error'] <= b['probability']['calibration_error']
                            and abs(a['expected']['coverage_80']-.8) <= abs(b['expected']['coverage_80']-.8)
                            and a['interval_score'] <= b['interval_score'])
            result[variant][output] = dict(historical_screen_passed=bool(passed),
                versus={k: v[output] for k, v in gains.items()}, deployment_approved=False, betting_approved=False)
    return result


def render(report):
    lines = ['# Role-Specific Workload Distribution', '', 'Stat: '+report['stat'], '',
        '| Variant | Opportunity RMSE | Yard RMSE | Bias | Median MAE | Brier | Calibration | 80% coverage |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, m in report['scores'].items():
        lines.append(f"| {name} | {m['opportunity']['rmse']:.3f} | {m['expected']['rmse']:.3f} | "
            f"{m['expected']['bias']:+.3f} | {m['typical']['mae']:.3f} | {m['probability']['brier']:.4f} | "
            f"{m['probability']['calibration_error']:.2%} | {m['expected']['coverage_80']:.1%} |")
    lines += ['', '## Prior-Volume Roles', '', '| Role / variant | Rows | Below | Above | Coverage |', '|---|---:|---:|---:|---:|']
    for role, scores in report['by_role'].items():
        for name, m in scores.items():
            lines.append(f"| {role} / {name} | {m['expected']['rows']} | {m['below_80']:.1%} | {m['above_80']:.1%} | {m['expected']['coverage_80']:.1%} |")
    lines += ['', '## Output Decisions', '']
    for name, outputs in report['output_decisions'].items():
        lines.append(name+': '+', '.join(k+'='+('historical_screen_passed' if v['historical_screen_passed'] else 'not_accepted')
            for k, v in outputs.items()))
    lines += ['', *report['limitations']]
    return '\n'.join(lines)+'\n'


def run(cache, through, seasons, stat='receiving_yards'):
    production = active_release()
    frame = joblib.load(cache)
    positions = {'receiving_yards': ('WR', 'TE', 'RB'), 'rushing_yards': ('QB', 'RB'),
                 'passing_yards': ('QB',)}[stat]
    frame = frame.loc[pd.to_datetime(frame.game_date_et).le(pd.Timestamp(through)) & frame.position.isin(positions)].copy()
    op = STATS[stat]
    invalid = (~np.isfinite(numeric(frame, stat)) | ~np.isfinite(numeric(frame, op)) | numeric(frame, op).lt(0)
        | (numeric(frame, op).eq(0) & numeric(frame, stat).ne(0)))
    excluded = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    validate(frame, stat)
    rows, lines, folds = [], [], []
    for fold, train, test in expanding_week_folds(frame, seasons):
        # The repair was designed after examining weeks 3-4; do not relabel them as acceptance.
        test = test.loc[~((test.season == 2026) & (test.week >= 3))]
        if test.empty:
            continue
        LOG.info('%s: fitting %d / testing %d unique player-games', fold, len(train), len(test))
        bundle = fit_bundle(train, stat)
        for variant, v, w, component in curves(bundle, test):
            rows.append(test[['game_id', 'player_id', 'season', 'week', 'position', 'game_date_et']].reset_index(drop=True).assign(
                variant=variant, fold=fold, role=roles(test, stat).to_numpy(),
                actual=numeric(test, stat).to_numpy(), actual_opportunity=numeric(test, op).to_numpy(),
                opportunity=component['opportunity'], **curve_summary(v, w)))
            lines.append(line_frame(test, stat, v, w).assign(variant=variant, fold=fold,
                role=lambda x: x.player_game.map(dict(zip(test.game_id.astype(str)+'|'+test.player_id.astype(str), roles(test, stat))))))
        folds.append(dict(fold=fold, blocks=bundle['blocks'], alpha=bundle['model'].alpha,
            count_crps_tuning=bundle['model'].tuning, test_start=str(test.game_date_et.min()),
            test_end=str(test.game_date_et.max()), used_features=bundle['model'].columns))
    if not rows:
        raise ValueError('No chronological evaluation folds')
    rows, lines = pd.concat(rows, ignore_index=True), pd.concat(lines, ignore_index=True)
    scores = summarize(rows, lines)
    output_decisions = decisions(rows, lines, scores)
    LOG.info('Fitting final isolated artifact through %s', through)
    bundle = fit_bundle(frame, stat)
    run_id = 'role-workload-'+stat+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = MODEL_ROOT/'role_workload'/run_id
    artifact = dict(contract=CONTRACT, run_id=run_id, stat=stat, bundle=bundle,
        training_end=str(frame.game_date_et.max()), production_release=production['release_id'],
        output_decisions=output_decisions, deployment_approved=False, betting_approved=False)
    atomic_joblib(path/'model.joblib', artifact)
    atomic_joblib(path/'oof.joblib', dict(rows=rows, lines=lines))
    report = {k: v for k, v in artifact.items() if k != 'bundle'}
    report.update(built_at=datetime.now(timezone.utc).isoformat(), artifact=str(path/'model.joblib'),
        artifact_sha256=hashlib.sha256((path/'model.joblib').read_bytes()).hexdigest(),
        source_cache_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        scores=scores, by_role={role: summarize(g, lines.loc[lines.role.eq(role)]) for role, g in rows.groupby('role')},
        folds=folds, feature_support=bundle['model'].support, alpha=bundle['model'].alpha,
        excluded_invalid_outcomes=excluded, production_changed=False,
        limitations=['Historical development screen, not untouched prospective acceptance.',
            'One row per player-game; expanding week tests and week-clustered uncertainty.',
            'Efficiency point is identical between workload variants; role tails have zero mean.',
            'Workload states are derived from one count curve, not contradictory independent classifiers.',
            'Role means prior volume, not verified starter status. Unknown injury context stays unknown.',
            'No normalization over offered players or hindsight participant pools.',
            'Proxy-line scores do not prove actual offered-line or selected-pick improvement.',
            'No production release, frozen trial, rankings, locks or betting approval changed.'])
    atomic_json(path/'report.json', clean(report))
    atomic_json(ROOT/f'reports/nfl_role_workload_{stat}_latest.json', clean(report))
    atomic_write_text(ROOT/f'reports/nfl_role_workload_{stat}_latest.md', render(report))
    if active_release() != production:
        raise RuntimeError('Production release changed during offline training')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--through', required=True)
    p.add_argument('--seasons', nargs='+', type=int, default=[2025, 2026])
    p.add_argument('--stat', choices=list(STATS), default='receiving_yards')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # Import under the stable module name so joblib can reload this artifact.
    from nfl_pipeline.modeling.train_role_workload import run as train
    result = train(a.cache, a.through, a.seasons, a.stat)
    print(json.dumps(dict(artifact=result['artifact'], scores=result['scores'], decisions=result['output_decisions'])))
