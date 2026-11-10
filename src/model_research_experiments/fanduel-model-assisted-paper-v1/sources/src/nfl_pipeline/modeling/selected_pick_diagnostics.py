"""Descriptive diagnostics from immutable locks, never reconstructed old forecasts."""
from collections import Counter
import math

import numpy as np
import pandas as pd

from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.evaluation import probability_metrics
from nfl_pipeline.modeling.live_scoring_replay import validate_lock, ROOT


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def locked_workload(payload, stat):
    features = payload.get('forecast_features') or (payload.get('scoring_replay') or {}).get('row') or {}
    keys = {
        'receiving_yards': ('receiver_projected_targets_v3', 'receiver_projected_targets_v2'),
        'rushing_yards': ('rb_projected_carries_v3', 'rb_projected_carries_v2'),
        'passing_yards': ('_pred_qb_pass_attempts',),
    }.get(stat, ())
    value = None; source = None
    for key in keys:
        number = finite(features.get(key))
        if number is not None and number >= 0:
            value = number; source = key; break
    thresholds = {'receiving_yards': (4, 8), 'rushing_yards': (8, 18), 'passing_yards': (25, 40)}
    bucket = 'unknown'
    if value is not None:
        low, high = thresholds[stat]
        bucket = 'low' if value < low else 'elevated' if value >= high else 'normal'
    role_key, role_threshold = {'receiving_yards': ('targets', 4), 'rushing_yards': ('carries', 8),
                               'passing_yards': ('pass_attempts', 25)}.get(stat, ('targets', 4))
    prior = finite(features.get(role_key+'_avg_5'))
    role = 'unknown' if prior is None else f"prior_{role_key}_{'ge' if prior >= role_threshold else 'lt'}_{role_threshold}"
    injury = features.get('injury_report_status') or payload.get('injury_report_status')
    return dict(workload_bucket=bucket, projected_opportunity=value, opportunity_source=source,
                role_proxy=role, injury_status=injury or 'unknown')


def population_scores(frame):
    if frame.empty:
        return dict(rows=0)
    # Each release/player-game/stat contributes one unit, regardless of offer multiplicity.
    weights = 1 / frame.groupby(['model_version', 'player_game', 'stat']).probability.transform('size')
    result = dict(probability_metrics(frame.probability, frame.outcome, weights),
        mean_probability=float(np.average(frame.probability, weights=weights)),
        win_rate=float(np.average(frame.outcome, weights=weights)),
        unique_player_games=int(frame.player_game.nunique()),
        weeks=len(frame[['season', 'week']].drop_duplicates()),
        replay_status=frame.replay_status.value_counts().to_dict())
    paired = frame.dropna(subset=['market']) if 'market' in frame else frame.iloc[:0]
    if len(paired):
        w = 1 / paired.groupby(['model_version', 'player_game', 'stat']).probability.transform('size')
        result['paired_market_comparison'] = dict(
            model=probability_metrics(paired.probability, paired.outcome, w),
            market=probability_metrics(paired.market, paired.outcome, w))
    matched = frame.loc[frame.replay_status.eq('matched')].dropna(subset=['raw', 'heuristic', 'post_exact'])
    w = (1 / matched.groupby(['model_version', 'player_game', 'stat']).probability.transform('size')) if len(matched) else None
    result['matched_stages'] = {col: probability_metrics(matched[col], matched.outcome, w)
                               for col in ('raw', 'heuristic', 'post_exact', 'probability')}
    if 'context_blend' in matched:
        detailed = matched.dropna(subset=['context_blend'])
        dw = 1 / detailed.groupby(['model_version', 'player_game', 'stat']).probability.transform('size') if len(detailed) else None
        result['detailed_matched_stages'] = {col: probability_metrics(detailed[col], detailed.outcome, dw)
            for col in ('raw', 'context_blend', 'heuristic', 'post_exact', 'probability')}
    return result


def selected_report(offers, micro):
    populations = {'all_eligible': offers.loc[offers.eligible].copy(), 'exact_locked_micro': micro.copy()}
    cohorts = set(zip(micro.model_version, micro.day)) if len(micro) else set()
    eligible = populations['all_eligible']
    populations['eligible_on_micro_release_dates'] = eligible.loc[
        [(v, d) in cohorts for v, d in zip(eligible.model_version, eligible.day)]].copy()
    output = {}
    for name, frame in populations.items():
        frame = frame.copy()
        frame['confidence_bucket'] = pd.cut(frame.probability, [0, .55, .6, .7, .8, 1],
            labels=['<=55%', '55-60%', '60-70%', '70-80%', '>80%'], include_lowest=True).astype(str)
        market = pd.to_numeric(frame.get('market', pd.Series(np.nan, index=frame.index)), errors='coerce')
        frame['disagreement_bucket'] = pd.cut(frame.probability - market, [-np.inf, 0, .05, .10, .20, np.inf],
            labels=['<=0pp', '0-5pp', '5-10pp', '10-20pp', '>20pp']).astype(str).replace('nan', 'unknown')
        groups = {}
        for columns in [('stat', 'side'), ('line_bucket',), ('workload_bucket',), ('confidence_bucket',),
                        ('disagreement_bucket',), ('model_version',), ('position',), ('model_version', 'scoring_version')]:
            if any(column not in frame for column in columns):
                continue
            groups['|'.join(columns)] = {'|'.join(map(str, key if isinstance(key, tuple) else (key,))): population_scores(g)
                for key, g in frame.groupby(list(columns), dropna=False)}
        output[name] = dict(summary=population_scores(frame), breakdowns=groups)
    return clean(dict(populations=output, limitations=[
        'Actual selections use validated exact ledger IDs, not a reranked historical top five.',
        'Eligibility is the stored pre-cap screen; no current rules are applied retrospectively.',
        'Release/date matching is not an exact publication-batch replay.',
        'Repeated offers are weighted by player-game/stat; weeks and player-games, not offer counts, measure independence.',
        'Small groups are descriptive, not evidence to deploy a calibrator or claims about causes.',
        'Pushes are excluded from binary probability scores; new probabilities are conditional on no push.']))


def workload_report(records):
    rows = []; excluded = Counter(); seen = set()
    for record in sorted(records, key=lambda r: (str(r['created_at_utc']), int(r['id']))):
        if record.get('stat') not in ('receiving_yards', 'rushing_yards', 'passing_yards'):
            continue
        reason = validate_lock(record)
        if reason:
            excluded[reason] += 1; continue
        if record.get('actual') is None:
            excluded['not_settled_or_no_verified_participation'] += 1; continue
        p = record.get('forecast_payload') or {}
        key = (p.get('model_version'), record['game_id'], record['player_id'], record['stat'])
        if key in seen:
            continue
        seen.add(key)
        workload = locked_workload(p, record['stat'])
        projection = finite(p.get('projection')); actual = finite(record['actual'])
        lower = finite(p.get('projection_p10')); upper = finite(p.get('projection_p90'))
        actual_key = {'receiving_yards': 'actual_targets', 'rushing_yards': 'actual_carries',
                      'passing_yards': 'actual_pass_attempts'}[record['stat']]
        actual_opportunity = finite(record.get(actual_key))
        volume = workload['projected_opportunity']
        interval = 'unknown' if lower is None or upper is None or lower > upper else (
            'below' if actual < lower else 'above' if actual > upper else 'inside')
        # An accounting identity, not a causal model or a reconstructed rate-head forecast.
        rate = projection / volume if projection is not None and volume is not None and volume > 0 else None
        volume_error = (volume-actual_opportunity)*rate if rate is not None and actual_opportunity is not None else None
        rate_error = actual_opportunity*rate-actual if rate is not None and actual_opportunity is not None else None
        dominant = 'unknown'
        if volume_error is not None:
            dominant = 'opportunity' if abs(volume_error) > abs(rate_error) else 'implied_efficiency_residual'
        rows.append(dict(prediction_id=record['id'], model_version=p.get('model_version'),
            game_id=record['game_id'], player_id=record['player_id'], player=p.get('player_name'),
            stat=record['stat'], season=record['season'], week=record['week'], position=p.get('position') or 'unknown',
            projection=projection, actual=actual, p10=lower, p90=upper, interval=interval,
            actual_opportunity=actual_opportunity, implied_yards_per_opportunity=rate,
            opportunity_error_yards=volume_error, efficiency_residual_yards=rate_error, dominant_component=dominant,
            **workload))
    frame = pd.DataFrame(rows); groups = {}
    if not frame.empty:
        for columns in [('model_version', 'stat'), ('stat', 'role_proxy'), ('stat', 'workload_bucket'),
                        ('stat', 'position'), ('stat', 'interval')]:
            scores = {}
            for key, group in frame.groupby(list(columns), dropna=False):
                valid = group.loc[group.interval.ne('unknown')]
                scores['|'.join(map(str, key))] = dict(rows=len(group), weeks=len(group[['season', 'week']].drop_duplicates()),
                    interval_rows=len(valid), below_rate=float(valid.interval.eq('below').mean()) if len(valid) else None,
                    above_rate=float(valid.interval.eq('above').mean()) if len(valid) else None,
                    coverage_80=float(valid.interval.eq('inside').mean()) if len(valid) else None,
                    bias=float((group.projection-group.actual).mean()), mae=float((group.projection-group.actual).abs().mean()),
                    opportunity_known=int(group.projected_opportunity.notna().sum()),
                    decomposition_known=int(group.opportunity_error_yards.notna().sum()),
                    dominant_components=group.dominant_component.value_counts().to_dict(),
                    injury_unknown=int(group.injury_status.eq('unknown').sum()))
            groups['|'.join(columns)] = scores
    return clean(dict(rows=rows, breakdowns=groups, exclusions=dict(excluded), limitations=[
        'One earliest valid settled lock per release/player-game/stat; later revisions are not substituted.',
        'Opportunity is a saved lock-time feature with its exact source, not necessarily the driver of the yardage head.',
        'Implied efficiency is projection divided by saved opportunity, not a separately learned rate forecast.',
        'Opportunity error plus efficiency residual equals projection minus actual; this is descriptive, not causal attribution.',
        'Role proxies use lagged targets (4), carries (8), or attempts (25), not verified starter status. Missing injury data is unknown.',
        'No intervals, targets, or forecasts are reconstructed when absent from the original lock.']))


def monotonicity_violations(records):
    """Compare unconditional tails only within the same immutable scoring context."""
    groups = {}; uncheckable = 0
    for r in records:
        p = r.get('forecast_payload') or {}
        if validate_lock(r) or r.get('line') is None:
            continue
        line = float(r['line'])
        if line.is_integer() and p.get('probability_basis') != 'win_given_no_push':
            uncheckable += 1; continue
        over = finite(p.get('over_probability')); under = finite(p.get('under_probability'))
        if over is None or under is None:
            uncheckable += 1; continue
        key = (p.get('model_version'), r['game_id'], r['player_id'], r['stat'], r['book'],
               p.get('prediction_context_cutoff_utc'))
        groups.setdefault(key, []).append((line, over, under, int(r['id'])))
    violations = []; pairs = 0
    for group in groups.values():
        ordered = sorted(set(group))
        for i, low in enumerate(ordered):
            for high in ordered[i+1:]:
                if high[0] <= low[0]:
                    continue
                pairs += 1
                if high[1] > low[1] + 1e-10 or high[2] < low[2] - 1e-10:
                    violations.append(dict(lower_id=low[3], higher_id=high[3], lower_line=low[0], higher_line=high[0],
                        lower_over=low[1], higher_over=high[1], lower_under=low[2], higher_under=high[2]))
    return dict(status='violations_found' if violations else 'checked' if pairs else 'no_comparable_pairs',
                comparable_pairs=pairs, violations=violations, uncheckable_rows=uncheckable,
                action='diagnostic_only_no_historical_probability_rewrite')


def write_diagnostics(selected, workload, consistency):
    for name, report in [('selected_pick_diagnostic', selected), ('workload_uncertainty_diagnostic', workload),
                         ('probability_consistency', consistency)]:
        root = ROOT / 'reports'
        atomic_json(root / f'nfl_{name}_latest.json', report)
        lines = ['# NFL ' + name.replace('_', ' ').title(), '', 'Diagnostic only. No model deployment or betting approval.', '']
        if name == 'selected_pick_diagnostic':
            lines += ['| Population | Rows | Player-games | Weeks | Mean P | Win rate | Brier | Calibration |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for key, value in report['populations'].items():
                s = value['summary']
                if s.get('rows'):
                    lines.append(f"| {key} | {s['rows']} | {s['unique_player_games']} | {s['weeks']} | {s['mean_probability']:.1%} | {s['win_rate']:.1%} | {s['brier']:.4f} | {s['calibration_error']:.1%} |")
        elif name == 'workload_uncertainty_diagnostic':
            lines += ['| Stat / prior-volume role | Rows | Intervals | Below | Above | Coverage | Known opportunity |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for key, s in report['breakdowns'].get('stat|role_proxy', {}).items():
                fmt = lambda v: 'unknown' if v is None else f'{v:.1%}'
                lines.append(f"| {key.replace('|', ' / ')} | {s['rows']} | {s['interval_rows']} | {fmt(s['below_rate'])} | {fmt(s['above_rate'])} | {fmt(s['coverage_80'])} | {s['opportunity_known']} |")
        else:
            lines += [f"Status: {report['status']}. Comparable pairs: {report['comparable_pairs']}; violations: {len(report['violations'])}; uncheckable rows: {report['uncheckable_rows']}.",
                      'No comparable pairs is missing evidence, not a passed monotonicity audit. Independent final calibrations can still reverse a curve; this report flags them without rewriting historical locks.']
        lines += ['', *['- '+x for x in report.get('limitations', [])], '', 'Full row details and breakdowns are in the adjacent JSON file.']
        (root / f'nfl_{name}_latest.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
