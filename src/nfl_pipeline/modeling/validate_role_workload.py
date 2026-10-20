"""Compare workload curves through original scoring and unchanged micro ranking.

Read-only historical replay. This does not publish forecasts, rewrite locks,
replace a pinned trial, or authorize bets.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mlb_pipeline.atomic_io import atomic_write_text
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import active_release, atomic_json
from nfl_pipeline.modeling import distribution_policy as dp
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics, clustered_gain
from nfl_pipeline.modeling.final_probability_validation import load_ledger
from nfl_pipeline.modeling.fit_market_calibration import prepare_captured
from nfl_pipeline.modeling.live_scoring_replay import load_rows, ROOT
from nfl_pipeline.modeling.prospective_market import locked_market
from nfl_pipeline.modeling.role_workload import CONTRACT, STATS, VARIANTS, roles
from nfl_pipeline.modeling.scoring_capture import replay
from nfl_pipeline.modeling.scoring_versions import archived_candidate
from nfl_pipeline.modeling.score_accuracy_components import empirical_residuals
from nfl_pipeline.modeling import receiving_research_trial as trial

LOG = logging.getLogger(__name__)


def candidate_for_curve(captured, values, weights, locked_at=None):
    """Replace only the raw forecast/distribution; preserve every later adjustment."""
    s = curve_summary(values[None, :], weights[None, :])
    revised = deepcopy(captured)
    point = float(s['median'][0])
    revised['projection'] = point
    revised['distribution'] = dict(captured.get('distribution') or {}, kind='empirical_oof_residual',
        residual_quantiles=empirical_residuals(values, weights, point))
    # Legacy scorers did not all handle integer pushes. Never claim that path was tested.
    if float(captured['offer']['line']).is_integer():
        raise ValueError('integer_line_not_validated_in_archived_replay')
    if locked_at is not None:
        from nfl_pipeline.modeling.validate_distribution_policy import candidate_at_lock
        return candidate_at_lock(dict(captured=revised, locked_at=locked_at), revised['offer'])
    return replay(revised)


def eligibility(row, artifact):
    cutoff = safe_time(row['payload'].get('prediction_context_cutoff_utc'))
    if row['model_version'] != artifact['production_release']:
        return 'production_release_mismatch'
    if not cutoff or artifact['training_end'] >= cutoff.date().isoformat():
        return 'training_not_before_lock'
    cap = row['captured']
    model = artifact.get('bundle', {}).get('model')
    if model is not None and cap['row'].get('position') not in model.position_prior:
        return 'position_without_training_support'
    for key in (STATS[artifact['stat']]+'_avg_5', artifact['stat']+'_avg_5'):
        try:
            valid = np.isfinite(float(cap['row'][key]))
        except (KeyError, ValueError, TypeError):
            valid = False
        if not valid:
            return 'missing_original_player_history'
    if float(row['line']).is_integer():
        return 'integer_line_not_validated_in_archived_replay'
    return None


def selected_ids(rows, candidates):
    """Use the archived cap, including unchanged non-target markets in each batch."""
    batches, selected, excluded = {}, [], Counter()
    for row in rows:
        key = (row['day'], row['batch'], row['source_scoring_version'])
        batches.setdefault(key, []).append(row)
    for (_, _, fingerprint), group in batches.items():
        if any(row['id'] not in candidates for row in group):
            excluded['incomplete_replay_batch'] += 1
            continue
        fn = archived_candidate(fingerprint).__globals__.get('_apply_daily_micro_cap')
        if fn is None:
            excluded['archived_ranking_unavailable'] += 1
            continue
        ordered = sorted(group, key=lambda r: r['id'])
        choices = [deepcopy(candidates[r['id']]) for r in ordered]
        fn(choices, 5)
        selected.extend(r['id'] for r, c in zip(ordered, choices) if c.get('tier') == 'micro_projection')
    return set(selected), dict(excluded)


def probability_score(frame, column, outcome='outcome'):
    if frame.empty:
        return dict(rows=0, player_games=0, weeks=0)
    w = 1/frame.groupby(['game_id', 'player_id']).id.transform('size')
    result = probability_metrics(frame[column], frame[outcome], w)
    settled = frame.loc[frame[outcome].notna()]
    result.update(player_games=len(settled[['game_id', 'player_id']].drop_duplicates()),
        weeks=len(settled[['season', 'week']].drop_duplicates()))
    return result


def evaluate(frame):
    if frame.empty:
        return dict(status='no_replayable_offers', cohorts={})
    cohorts = {}
    for key, group in frame.groupby(['model_version', 'scoring_version']):
        scores = {}
        for pool, data in [('all_offers', group), ('exact_micro', group.loc[group.exact_micro]),
                           ('fixed_research', group.loc[group.fixed_research]),
                           ('production_ranked', group.loc[group.production_ranked])]:
            scores[pool] = {name: probability_score(data, name) for name in ('production', 'market',
                *[v+'_raw' for v in VARIANTS], *[v+'_final' for v in VARIANTS])}
        for variant in VARIANTS:
            selected = group.loc[group[variant+'_ranked']]
            scores[variant+'_ranked'] = dict(
                candidate=probability_score(selected, variant+'_selected_probability', variant+'_selected_outcome'),
                market=probability_score(selected, variant+'_selected_market', variant+'_selected_outcome'),
                selected_ids=selected.id.astype(int).tolist())
        unique = group.sort_values(['batch', 'id']).drop_duplicates(['game_id', 'player_id'])
        scores['projections'] = dict(production_central=projection_metrics(unique.actual, unique.production_point),
            production_raw=projection_metrics(unique.actual, unique.production_mean, unique.production_p10, unique.production_p90))
        scores['roles'] = {}
        for variant in VARIANTS:
            scores['projections'][variant] = dict(
                expected=projection_metrics(unique.actual, unique[variant+'_mean'], unique[variant+'_p10'], unique[variant+'_p90']),
                typical=projection_metrics(unique.actual, unique[variant+'_median']),
                opportunity=projection_metrics(unique.actual_opportunity, unique[variant+'_opportunity']))
            settled = group.loc[group.outcome.notna()]
            errors = settled.assign(reference_error=(settled.production-settled.outcome)**2,
                challenger_error=(settled[variant+'_final']-settled.outcome)**2)
            scores[variant+'_final_brier_gain'] = clustered_gain(errors.groupby(['season', 'week', 'game_id', 'player_id'])[
                ['reference_error', 'challenger_error']].mean().reset_index())
        for role, g in unique.groupby('role'):
            settled = g.loc[g.actual.notna()]
            scores['roles'][role] = {}
            for variant in ('production', *VARIANTS):
                low, high = settled[variant+'_p10'], settled[variant+'_p90']
                scores['roles'][role][variant] = dict(rows=len(settled),
                    below=float((settled.actual < low).mean()) if len(settled) else None,
                    above=float((settled.actual > high).mean()) if len(settled) else None,
                    coverage=float(((settled.actual >= low) & (settled.actual <= high)).mean()) if len(settled) else None)
        scores['masked_by_scoring'] = {v: int(np.isclose(group[v+'_final'], group.production, rtol=0, atol=1e-12).sum()) for v in VARIANTS}
        cohorts['|'.join(key)] = scores
    return dict(status='retrospective_development_only', cohorts=cohorts)


def score_records(records, ledger, fixed_ids, artifact):
    if artifact.get('contract') != CONTRACT:
        raise ValueError('Wrong workload artifact contract')
    micro_ids = {int(r['prediction_id']) for r in ledger}
    frame, excluded = prepare_captured(records, micro_ids | set(fixed_ids), require_market_layer=False)
    LOG.info('Original scoring replay: %d eligible offers / %d inputs', len(frame), len(records))
    excluded = Counter(excluded)
    if frame.empty:
        return pd.DataFrame(), dict(excluded), {}
    lookup = {int(r['id']): r for r in records}
    universe, target = [], []
    for row in frame.to_dict('records'):
        p = lookup[row['id']]['forecast_payload']
        if p['book'] != 'fanduel':
            excluded['not_fanduel'] += 1
            continue
        row.update(payload=p, day=str(p['game_date_et']), batch=p['prediction_context_cutoff_utc'])
        universe.append(row)
        if row['stat'] != artifact['stat']:
            continue
        error = eligibility(row, artifact)
        if error:
            excluded[error] += 1
        else:
            target.append(row)
    from nfl_pipeline.modeling.validate_distribution_policy import candidate_at_lock
    original_candidates = {r['id']: candidate_at_lock(r, r['captured']['offer']) for r in universe}
    candidates = {v: dict(original_candidates) for v in VARIANTS}
    rows = []
    model = artifact['bundle']['model']
    # Bounded chunks avoid multiplying thousands of offers by large quadrature grids.
    for begin in range(0, len(target), 64):
        LOG.info('Replaying workload offers %d-%d / %d', begin+1, min(begin+64, len(target)), len(target))
        chunk = target[begin:begin+64]
        features = pd.DataFrame([r['captured']['row'] for r in chunk])
        point = np.array([r['captured']['projection'] for r in chunk])
        curves = {v: model.yardage_curve(features, variant=v, projection=point) for v in VARIANTS}
        summaries = {v: curve_summary(c[0], c[1]) for v, c in curves.items()}
        role = roles(features, artifact['stat']).to_numpy()
        for i, r in enumerate(chunk):
            rec = lookup[r['id']]
            p, cap = r['payload'], r['captured']
            actual = r['actual']
            outcome = None if not np.isfinite(actual) or actual == r['line'] else float((actual > r['line']) == (r['side'] == 'over'))
            try:
                raw = dp.prop_curve(r['stat'], cap['projection'], cap['metrics'], cap['distribution'], cap['row'],
                    archived_candidate(cap['scoring_fingerprint']).__globals__['_side_prob'])
                s = dp.summary(raw)
                row = {k: r[k] for k in ('id', 'game_id', 'player_id', 'season', 'week', 'model_version', 'scoring_version', 'batch', 'side', 'line')}
                row.update(production=float(p['probability']), market=locked_market(rec)['market_probability'],
                    actual=actual, outcome=outcome, role=role[i], exact_micro=r['id'] in micro_ids,
                    fixed_research=r['id'] in fixed_ids, production_point=float(cap['projection']),
                    production_mean=s['mean'], production_p10=s['p10'], production_p90=s['p90'],
                    actual_opportunity=rec.get('actual_targets' if artifact['stat'] == 'receiving_yards' else 'actual_carries'))
                for variant, (v, w, components) in curves.items():
                    candidate = candidate_for_curve(cap, v[i], w[i], r['locked_at'])
                    candidates[variant][r['id']] = candidate
                    side = candidate['side']
                    prob = candidate['probability']
                    curve = dp.curve(v[i], w[i])
                    raw_over = dp.outcomes(curve, r['line'])['conditional_over']
                    row.update({variant+'_'+k: float(summaries[variant][k][i]) for k in ('mean', 'median', 'p10', 'p90')})
                    row[variant+'_opportunity'] = float(components['opportunity'][i])
                    row[variant+'_fixed_rate'] = float(components['fixed_rate'][i])
                    row[variant+'_state_probabilities'] = components['state_probabilities'][i].tolist()
                    row[variant+'_raw'] = raw_over if r['side'] == 'over' else 1-raw_over
                    row[variant+'_final'] = prob if side == r['side'] else 1-prob
                    row[variant+'_trace'] = candidate.get('probability_trace')
                    row[variant+'_selected_probability'] = prob
                    row[variant+'_selected_outcome'] = None if outcome is None else (outcome if side == r['side'] else 1-outcome)
                    row[variant+'_selected_market'] = row['market'] if side == r['side'] else 1-row['market']
                    row[variant+'_selected_side'] = side
                    row[variant+'_selected_price'] = candidate['price']
            except (ValueError, KeyError) as exc:
                excluded[str(exc)] += 1
                # A failed candidate cannot quietly disappear from the ranked universe.
                for variant in VARIANTS:
                    candidates[variant].pop(r['id'], None)
                continue
            rows.append(row)
    ranking_exclusions = {}
    selected, ranking_exclusions['production'] = selected_ids(universe, original_candidates)
    for row in rows:
        row['production_ranked'] = row['id'] in selected
    for variant in VARIANTS:
        selected, ranking_exclusions[variant] = selected_ids(universe, candidates[variant])
        for row in rows:
            row[variant+'_ranked'] = row['id'] in selected
    return pd.DataFrame(rows), dict(excluded), ranking_exclusions


def render(report):
    text = ['# Role Workload: Full-Path FanDuel Comparison', '',
        '| Cohort / pool | Production Brier | Workload final | Tails final | Combined final | Market |',
        '|---|---:|---:|---:|---:|---:|']
    for cohort, scores in report['evaluation']['cohorts'].items():
        for pool in ('all_offers', 'exact_micro', 'fixed_research', 'production_ranked'):
            m = scores[pool]
            values = [f"{m[k]['brier']:.4f} (n={m[k]['rows']})" if m[k].get('rows') else '-'
                for k in ('production', 'workload_only_final', 'tails_only_final', 'combined_final', 'market')]
            text.append('| '+cohort.replace('|', ' / ')+' / '+pool+' | '+' | '.join(values)+' |')
    text += ['', '## Original Selected Picks', '',
        'Original micro and fixed research IDs are evaluated above. Counterfactual rankings are in JSON; empty selections are not a pass.',
        '', '## Scoring Influence', '']
    for cohort, scores in report['evaluation']['cohorts'].items():
        text.append(cohort+': final probability unchanged '+json.dumps(scores['masked_by_scoring']))
    text += ['', 'Exclusions: '+json.dumps(report['exclusions']), '',
        'Ranking exclusions: '+json.dumps(report['ranking_exclusions']), '', *report['limitations']]
    return '\n'.join(text)+'\n'


def run(path, season, weeks):
    production = active_release()
    artifact = joblib.load(path)
    import psycopg2
    from nfl_pipeline.db import PG_DSN
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute('SELECT DISTINCT game_date_et FROM raw.nfl_games WHERE season=%s AND week=ANY(%s) ORDER BY game_date_et',
                    (season, list(weeks)))
        days = [r[0] for r in cur.fetchall()]
    records = []
    for day in days:
        LOG.info('Loading original locks for %s', day)
        records.extend(r for r in load_rows(day) if r['season'] == season and r['week'] in weeks)
    registration = trial.load_registration()
    fixed_ids = {int(r['forecast_id']) for doc in trial.captures(registration) for r in doc.get('selected', [])}
    frame, excluded, rank_excluded = score_records(records, load_ledger(), fixed_ids, artifact)
    report = dict(run_id=artifact['run_id'], stat=artifact['stat'], artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        built_at=datetime.now(timezone.utc).isoformat(), season=season, weeks=weeks, input_rows=len(records),
        exclusions=excluded, ranking_exclusions=rank_excluded, evaluation=evaluate(frame), rows=frame.to_dict('records'),
        production_changed=False, deployment_approved=False, betting_approved=False,
        limitations=['Retrospective development replay, not prospective evidence. Weeks 3-4 informed this repair.',
            'Frozen efficiency center is the original raw point divided by original prior workload, not the book line.',
            'Actual archived scoring code, market adjustments and daily cap are unchanged.',
            'Ranking uses saved eligible offers including non-target stats, not a reconstructed full sportsbook universe.',
            'Original micro and fixed research selections are reported separately from counterfactual rankings.',
            'Distribution coverage refers to the raw stat curve; per-offer final probabilities are not a full CDF.',
            'Scoring cohorts remain separate and repeated offers share player-game weight.',
            'Missing original inputs/outcomes remain excluded/pending; no current inputs replace history.',
            'No production model, frozen trial, Discord, ledger or cash approval changed.'])
    atomic_json(path.parent/'offer_validation.json', clean(report))
    atomic_json(ROOT/f"reports/nfl_role_workload_{artifact['stat']}_offers_latest.json", clean(report))
    atomic_write_text(ROOT/f"reports/nfl_role_workload_{artifact['stat']}_offers_latest.md", render(report))
    if active_release() != production:
        raise RuntimeError('Production changed during read-only validation')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path)
    p.add_argument('--season', type=int)
    p.add_argument('--weeks', type=int, nargs='+')
    p.add_argument('--render-existing', type=Path, help='Render an existing JSON result without querying or rescoring')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if a.render_existing:
        result = json.loads(a.render_existing.read_text(encoding='utf-8'))
        atomic_write_text(a.render_existing.with_suffix('.md'), render(result))
        print('Rendered existing result; no locks or models changed.')
    else:
        if not a.artifact or a.season is None or not a.weeks:
            p.error('--artifact, --season and --weeks are required for validation')
        result = run(a.artifact, a.season, a.weeks)
        print(json.dumps(dict(rows=len(result['rows']), exclusions=result['exclusions'], ranking_exclusions=result['ranking_exclusions'])))
