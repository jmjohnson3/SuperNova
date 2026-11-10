"""Audit original yardage locks and test uncertainty through archived live scoring.

Read-only retrospective experiment. Never rewrites locks, changes selections,
publishes Discord, enables an exact-line model, or authorizes cash.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.context_contract import safe_time, validate_evidence, number
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling.challenger_models import player_features, curve_summary
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics
from nfl_pipeline.modeling.prospective_market import locked_market
from nfl_pipeline.modeling.receiving_selection_experiment import ranked
from nfl_pipeline.modeling.scoring_capture import replay
from nfl_pipeline.modeling.train_accuracy_challengers import ROOT

STATS = ('passing_yards', 'receiving_yards', 'rushing_yards')
OP_KEYS = {'receiving_yards': ('_pred_receiver_targets', 'receiver_projected_targets_v3', 'receiver_projected_targets_v2'),
           'rushing_yards': ('_pred_rb_carries', 'rb_projected_carries_v3', 'rb_projected_carries_v2'),
           'passing_yards': ('_pred_qb_pass_attempts',)}


def workload_decomposition(record):
    p = record['forecast_payload']; features = (p.get('scoring_replay') or {}).get('row') or {}
    op = next((number(features[k]) for k in OP_KEYS[record['stat']]
               if number(features.get(k)) is not None), None)
    actual_op = record.get('actual_opportunity'); actual = record.get('actual')
    if op is None or op <= 0 or actual_op is None or actual is None:
        return dict(status='missing_decomposition_inputs', projected_opportunity=op)
    rate = float(p['projection']) / op
    workload = (float(actual_op) - op) * rate
    efficiency = float(actual) - float(actual_op) * rate
    return dict(status='descriptive_identity', projected_opportunity=op, actual_opportunity=float(actual_op),
        implied_rate=rate, workload_yards=workload, efficiency_yards=efficiency,
        total_error=float(actual)-float(p['projection']),
        dominant='workload' if abs(workload) >= abs(efficiency) else 'efficiency',
        caveat='Implied projection/opportunity ratio, not a claim that the live model multiplied these heads.')


def revised_score(payload, uncertainty):
    """Preserve the point forecast and every archived live adjustment."""
    captured = payload.get('scoring_replay')
    original = replay(captured)
    if original['side'] != payload['side'] or abs(original['probability'] - payload['probability']) > 1e-10:
        raise ValueError('original_probability_replay_mismatch')
    if float(payload['line']).is_integer():
        raise ValueError('integer_push_curve_requires_separate_validation')
    features = pd.DataFrame([dict(captured['row'], context_evidence=payload.get('context_evidence'))])
    X = player_features(features, payload['stat'])
    center = np.array([float(payload['projection'])])
    values, weights = uncertainty.mixture(X, center, np.ones((1, 1)))
    confidence = float(uncertainty.confidence(X)[0])
    revised = deepcopy(captured)
    revised['distribution'] = dict(kind='empirical_oof_residual',
        residual_quantiles=(values[0] - center[0]).tolist(), projection_confidence=confidence)
    candidate = replay(revised)
    over = candidate['probability'] if candidate['side'] == 'over' else 1 - candidate['probability']
    summary = curve_summary(values, weights)
    return dict(probability=float(over if payload['side'] == 'over' else 1-over),
        raw_probability=float(candidate['raw_over_probability'] if payload['side'] == 'over' else 1-candidate['raw_over_probability']),
        candidate_side=candidate['side'], confidence=confidence,
        p10=float(summary['p10'][0]), p90=float(summary['p90'][0]),
        interval_contract='raw conditional curve; final line adjustments are not a certified full CDF')


def frame_records(records, artifact=None):
    output = []; excluded = Counter()
    for r in records:
        p = r['forecast_payload']; captured = p.get('scoring_replay')
        if not captured:
            excluded['missing_immutable_scoring_inputs'] += 1; continue
        price = number(p.get('price')); probability = number(p.get('probability'))
        if (price is None or abs(price) < 100 or probability is None or not 0 <= probability <= 1
                or p.get('side') not in ('over', 'under') or number(p.get('line')) is None):
            excluded['invalid_locked_offer'] += 1; continue
        lock, start = safe_time(r['created_at_utc']), safe_time(r['start_ts_utc'])
        if not lock or not start or lock >= start:
            excluded['not_pregame'] += 1; continue
        context = validate_evidence(p.get('context_evidence'), lock)
        market = locked_market(r)
        trace = p.get('probability_trace') or {}
        side = p['side']; flip = lambda value: float(value) if side == 'over' else 1-float(value)
        payout = price/100 if price > 0 else 100/-price
        result = r.get('result'); actual = r.get('actual')
        quote_time = safe_time(p.get('quote_fetched_at_utc'))
        fresh = quote_time is not None and 0 <= (lock - quote_time).total_seconds() <= 1800
        row = dict(prediction_id=int(r['id']), stat=r['stat'], player_name=r['player_name'],
            game_id=r['game_id'], player_id=r['player_id'], season=r['season'], week=r['week'],
            day=str(r['game_date_et']), batch=str(p.get('prediction_context_cutoff_utc')),
            model_version=p['model_version'], scoring_version=captured['scoring_fingerprint'],
            side=side, line=float(p['line']), price=price, payout=payout,
            probability=float(p['probability']), market=market['market_probability'],
            raw=flip(trace['raw_over']) if trace.get('raw_over') is not None else np.nan,
            heuristic=flip(trace['heuristic_over']) if trace.get('heuristic_over') is not None else np.nan,
            outcome=1. if result == 'win' else 0. if result == 'loss' else np.nan,
            result=result, actual=float(actual) if actual is not None else np.nan,
            projection=float(p['projection']), p10=p.get('projection_p10'), p90=p.get('projection_p90'),
            push_probability=float(p.get('push_probability') or 0),
            eligible=bool(fresh and p.get('link') and p.get('drift_guard_pass') and market['market_probability'] is not None),
            uncertainty_proxy=1., confidence=float(p.get('projection_confidence') or 0), edge=float(p.get('edge') or 0),
            context_valid=context is not None, injury_unknown=not context or bool(context.get('injury_missing'))
                or (context.get('injury_status') is None and context.get('practice_status') is None),
            routes_source=(context or {}).get('routes_source', 'invalid_or_missing'),
            first_read_unknown=(context or {}).get('first_read_history') is None,
            true_pair_reason=market['market_evidence'], exact_line_model=p.get('exact_line_model_version'),
            decomposition=workload_decomposition(r))
        if row['p10'] is not None and row['p90'] is not None:
            row['uncertainty_proxy'] = min(1., max(0., (float(row['p90'])-float(row['p10']))/(2*max(10., abs(row['projection'])))))
        if artifact and r['stat'] in artifact['models']:
            cutoff = safe_time(p.get('prediction_context_cutoff_utc'))
            if context is None or cutoff is None:
                excluded['challenger_context_invalid'] += 1
            elif artifact['training_end'] >= cutoff.date().isoformat():
                excluded['challenger_training_not_before_lock'] += 1
            elif artifact['production_release'] != p['model_version']:
                excluded['challenger_release_mismatch'] += 1
            else:
                try:
                    result = revised_score(p, artifact['models'][r['stat']]['reference_uncertainty'])
                    row.update({'challenger_' + k: v for k, v in result.items()})
                except (ValueError, KeyError) as exc:
                    excluded['challenger_' + str(exc)] += 1
        output.append(row)
    return pd.DataFrame(output), dict(excluded)


def population(frame):
    if frame.empty:
        return dict(rows=0)
    settled = frame.loc[frame.outcome.notna()]
    stages = {}
    for name in ('raw', 'heuristic', 'probability', 'market', 'challenger_probability'):
        if name in settled:
            stages[name] = probability_metrics(settled[name], settled.outcome)
    comparable = settled.loc[settled.get('challenger_probability', pd.Series(np.nan, index=settled.index)).notna() & settled.market.notna()]
    return dict(rows=len(frame), binary_rows=len(settled),
        record=dict(Counter(frame.result.fillna('unresolved_or_pending'))), stages=stages,
        paired_comparison={c: probability_metrics(comparable[c], comparable.outcome)
            for c in ('probability', 'market', 'challenger_probability') if c in comparable},
        original_intervals=projection_metrics(settled.actual, settled.projection, settled.p10, settled.p90),
        challenger_raw_intervals=projection_metrics(comparable.actual, comparable.projection,
            comparable.challenger_p10, comparable.challenger_p90) if len(comparable) else None,
        ids=frame.prediction_id.astype(int).tolist())


def write_markdown(report, path):
    def metric(stages, key):
        row = stages.get(key, {})
        return f"{row['brier']:.4f}" if row.get('rows') and row.get('brier') is not None else '-'

    text = ['# Locked Yardage Repair Validation', '', *report['limitations'], '',
        '## Archived Production', '',
        '| Stat | Settled rows | Raw Brier | Final Brier | Market Brier |',
        '|---|---:|---:|---:|---:|']
    for stat, result in report['stats'].items():
        p = result['all_displayed']; s = p['stages']
        text.append(f"| {stat} | {p['binary_rows']} | {metric(s, 'raw')} | {metric(s, 'probability')} | {metric(s, 'market')} |")
    text += ['', '## Identical-Offer Replay', '',
        'Only offers with both a challenger replay and valid market evidence appear below.', '',
        '| Population | Paired rows | Production Brier | Challenger Brier | Market Brier |',
        '|---|---:|---:|---:|---:|']
    populations = {stat: r['all_displayed'] for stat, r in report['stats'].items()}
    populations['Fixed research selections'] = report['fixed_research']
    for label, p in populations.items():
        s = p.get('paired_comparison', {})
        n = s.get('probability', {}).get('rows', 0)
        text.append(f"| {label} | {n} | {metric(s, 'probability')} | {metric(s, 'challenger_probability')} | {metric(s, 'market')} |")
    text += ['', '## Context and Error Components', '',
        '| Stat | Rows | Injury unknown | True routes | Workload-dominated | Efficiency-dominated | Missing components |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stat, result in report['stats'].items():
        c = result['context']; d = result['decomposition']
        text.append(f"| {stat} | {c['rows']} | {c['unknown_injury']} | {c['routes'].get('observed_prior_games', 0)} | "
            f"{d.get('workload', 0)} | {d.get('efficiency', 0)} | {d.get('missing_decomposition_inputs', 0)} |")
    text += ['', 'Error components are descriptive accounting, not causal attribution.', '',
        '## Fixed Research Record', '',
        f"Original selections: {json.dumps(report['fixed_research'].get('record', {}))}. These are not confirmed cash wagers.", '',
        '## Counterfactual Ranking', '',
        'Reselected from the archived card pool, not actual placed bets or independent proof.', '',
        '| Ranking | Picks | Wins | Losses | Unsettled / other |', '|---|---:|---:|---:|---:|']
    for label, p in report['ranking'].items():
        record = p.get('record', {}); wins = record.get('win', 0); losses = record.get('loss', 0)
        text.append(f"| {label} | {p['rows']} | {wins} | {losses} | {p['rows']-wins-losses} |")
    text += ['', 'Full stage metrics, intervals, exclusions, and prediction IDs are in the companion JSON.', '',
        'No production deployment or betting approval.']
    path.write_text('\n'.join(text) + '\n', encoding='utf-8')


def run(season, week, artifact_path=None):
    from nfl_pipeline.modeling import receiving_research_trial as trial
    research_ids = {int(r['forecast_id']) for doc in trial.captures(trial.load_registration())
                    for r in doc.get('selected', []) if r.get('season') == season and r.get('week') == week}
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='90s'")
        cur.execute('SELECT DISTINCT game_date_et FROM raw.nfl_games WHERE season=%s AND week=%s', (season, week))
        ids = set()
        for d in cur.fetchall():
            for path in (ROOT / 'reports').glob(f'nfl_discord_matchups_{d["game_date_et"]}*.json'):
                for card in json.loads(path.read_text()).get('cards', []):
                    ids.update(int(r['forecast_id']) for r in card.get('forecast_manifest', [])
                        if r.get('kind') == 'prop' and r.get('book') == 'fanduel' and r.get('line') is not None)
        cur.execute("""SELECT p.id,p.game_id,p.player_id,p.player_name,p.stat,p.season,p.week,p.game_date_et,
            p.created_at_utc,g.start_ts_utc,r.result,r.actual_stat AS actual,
            CASE p.stat WHEN 'receiving_yards' THEN a.targets WHEN 'rushing_yards' THEN a.carries
                WHEN 'passing_yards' THEN a.pass_attempts END AS actual_opportunity,
            p.forecast_payload - 'forecast_features' - 'model_inputs' - 'forecast_distribution' AS forecast_payload
            FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
            LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
            LEFT JOIN raw.nfl_player_gamelogs a ON a.game_id=p.game_id AND a.player_id=p.player_id AND a.team_abbr=p.team_abbr
            WHERE p.id=ANY(%s) AND p.stat=ANY(%s) AND p.season=%s AND p.week=%s
            ORDER BY p.created_at_utc,p.id""", (list(ids | research_ids), list(STATS), season, week))
        records = cur.fetchall()
    artifact = joblib.load(artifact_path) if artifact_path else None
    frame, exclusions = frame_records(records, artifact)
    if frame.empty:
        raise ValueError('No archived priced yardage manifests')
    displayed = frame.loc[frame.prediction_id.isin(ids)].drop_duplicates(['game_id', 'player_id', 'stat'], keep='last')
    fixed = frame.loc[frame.prediction_id.isin(research_ids)]
    report = dict(season=season, week=week, built_at=datetime.now(timezone.utc).isoformat(),
        artifact=artifact.get('run_id') if artifact else None, exclusions=exclusions, stats={},
        production_changed=False, deployment_approved=False, cash_approved=False,
        limitations=['Latest archived Discord forecast per player-game/stat, not all generated offers.',
            'Ranking tests use the archived card pool chronologically, not an assertion of actual selected wagers.',
            'Original fixed research IDs are scored separately and never replaced.',
            'Challenger artifact was trained retrospectively on pre-lock history; this is NOT a prospective capture.',
            'Raw interval coverage is not final-CDF coverage. No deployment without coherent full-path validation.',
            'Context counts describe observed/missing evidence, not player health.',
            'Week 3 informed the repair hypothesis; it is diagnostic, not an untouched acceptance set.'])
    for stat, group in displayed.groupby('stat'):
        report['stats'][stat] = dict(all_displayed=population(group),
            context=dict(rows=len(group), settled_binary_rows=int(group.outcome.notna().sum()),
                unknown_injury=int(group.injury_unknown.sum()),
                routes=dict(Counter(group.routes_source)), unknown_first_read=int(group.first_read_unknown.sum())),
            decomposition=dict(Counter(r.get('dominant', r['status']) for r in group.decomposition)),
            by_cohort={str(k): population(g) for k, g in group.groupby(['model_version', 'scoring_version'])})
    report['fixed_research'] = population(fixed)
    original_pool = frame.loc[frame.prediction_id.isin(ids)].copy()
    report['ranking'] = dict(ev=population(ranked(original_pool)),
        conservative=population(ranked(original_pool, penalty=.5, uncertainty_aware=True)))
    if 'challenger_probability' in original_pool:
        usable = original_pool.loc[original_pool.challenger_probability.notna()].copy()
        report['ranking']['paired_current'] = population(ranked(usable))
        usable['uncertainty_proxy'] = ((usable.challenger_p90-usable.challenger_p10)
            / (2*usable.projection.abs().clip(lower=10))).clip(0, 1)
        usable['confidence'] = usable.challenger_confidence
        report['ranking']['challenger_conservative'] = population(ranked(usable, penalty=.5,
            uncertainty_aware=True, probability='challenger_probability'))
    report['rows'] = frame.to_dict('records')
    name = f'nfl_yardage_lock_validation_{season}_week{week}'
    atomic_json(ROOT / 'reports' / (name + '.json'), clean(report))
    write_markdown(report, ROOT / 'reports' / (name + '.md'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, required=True); p.add_argument('--week', type=int, required=True)
    p.add_argument('--artifact', type=Path)
    a = p.parse_args(); result = run(a.season, a.week, a.artifact)
    print(json.dumps(dict(stats=list(result['stats']), exclusions=result['exclusions'], production_changed=False)))
