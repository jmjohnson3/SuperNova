"""Replay quantile forecasts at immutable FanDuel locks; never publish bets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.challenger_models import curve_summary, curve_over
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics, clustered_gain
from nfl_pipeline.modeling.final_probability_validation import load_ledger
from nfl_pipeline.modeling.live_scoring_replay import load_rows, ROOT
from nfl_pipeline.modeling.prospective_market import locked_market
from nfl_pipeline.modeling.receiving_quantiles import CONTRACT
from nfl_pipeline.modeling.receiving_selection_experiment import offer_frame, ranked
from nfl_pipeline.modeling.target_volume_validation import changed_probability
from nfl_pipeline.modeling.train_receiving_quantiles import curves
from nfl_pipeline.modeling import receiving_research_trial as trial

VARIANTS = ('quantile', 'calibrated_blend')


def eligible_error(record, artifact):
    p = record.get('forecast_payload') or {}
    captured = p.get('scoring_replay') or {}
    cutoff = safe_time(p.get('prediction_context_cutoff_utc'))
    if p.get('model_version') != artifact['production_release']:
        return 'production_release_mismatch'
    if not cutoff or str(artifact['training_end']) >= cutoff.date().isoformat():
        return 'training_not_before_lock'
    if not captured.get('row') or captured.get('projection') is None:
        return 'missing_original_scoring_inputs'
    for name in ('targets_avg_5', 'receiving_yards_avg_5'):
        try:
            valid = np.isfinite(float(captured['row'][name]))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            return 'missing_archived_player_history'
    evidence = locked_market(record)
    if evidence['market_probability'] is None:
        return evidence['market_evidence']
    fetched = safe_time(captured['offer'].get('fetched_at_utc'))
    if not fetched or (cutoff-fetched).total_seconds() > 1200:
        return 'stale_original_quote'
    if float(p['line']).is_integer():
        return 'integer_line_not_validated_in_replay'
    return None


def probability_scores(frame):
    settled = frame.loc[frame.outcome.notna()]
    if settled.empty:
        return dict(rows=len(frame), settled=0)
    weights = 1/settled.groupby(['game_id', 'player_id']).prediction_id.transform('size')
    probabilities = {name: probability_metrics(settled[column], settled.outcome, weights)
        for name, column in [('production', 'probability'), ('market', 'market'),
            *[(v+'_raw', v+'_raw_probability') for v in VARIANTS],
            *[(v+'_final', v+'_probability') for v in VARIANTS]]}
    return dict(rows=len(frame), settled=len(settled),
        player_games=len(settled[['game_id', 'player_id']].drop_duplicates()),
        weeks=len(settled[['season', 'week']].drop_duplicates()), probabilities=probabilities)


def evaluate(frame, fixed_ids):
    if frame.empty:
        return dict(status='no_replayable_offers', deployment_approved=False)
    cohorts = {}
    for key, group in frame.groupby(['model_version', 'scoring_version']):
        pools = dict(all_offers=group, exact_micro=group.loc[group.exact_micro],
            fixed_research=group.loc[group.prediction_id.isin(fixed_ids)],
            unchanged_ranking=ranked(group))
        for variant in VARIANTS:
            pools[variant+'_ranked'] = ranked(group, probability=variant+'_probability')
        result = {name: probability_scores(pool) for name, pool in pools.items()}
        if 'calibrated_blend_trace' in group:
            result['scoring_influence'] = dict(rows=len(group),
                zero_model_probability_weight=sum((t.get('market_calibration') or {}).get('probability_trust') == 0
                    for t in group.calibrated_blend_trace),
                final_probability_unchanged=int(np.isclose(group.calibrated_blend_probability,
                    group.probability, atol=1e-12, rtol=0).sum()))
        # First archived forecast per player-game, rather than per book/line.
        unique = group.sort_values(['batch', 'prediction_id']).drop_duplicates(['game_id', 'player_id'])
        result['stat_forecasts'] = dict(production_central=projection_metrics(unique.actual, unique.raw_projection),
            production_central_semantics='Original pre-offer central forecast, not a certified expected mean.',
            locked_pricing_interval=projection_metrics(unique.actual, unique.projection,
                pd.to_numeric(unique.p10, errors='coerce'), pd.to_numeric(unique.p90, errors='coerce')))
        for variant in VARIANTS:
            result['stat_forecasts'][variant] = dict(
                expected=projection_metrics(unique.actual, unique[variant+'_mean'], unique[variant+'_p10'], unique[variant+'_p90']),
                typical=projection_metrics(unique.actual, unique[variant+'_median']))
            settled = group.loc[group.outcome.notna()]
            errors = settled.assign(reference_error=(settled.probability-settled.outcome)**2,
                challenger_error=(settled[variant+'_probability']-settled.outcome)**2)
            result[variant+'_final_brier_gain'] = clustered_gain(errors.groupby(['season', 'week', 'game_id', 'player_id'])[
                ['reference_error', 'challenger_error']].mean().reset_index())
        cohorts['|'.join(key)] = result
    return dict(status='retrospective_diagnostic', cohorts=cohorts,
        deployment_approved=False, betting_approved=False)


def score_records(records, ledger, artifact):
    if artifact.get('contract') != CONTRACT:
        raise ValueError('Wrong receiving quantile artifact contract')
    frame, exclusions = offer_frame(records, ledger)
    excluded = Counter(exclusions)
    if frame.empty:
        return frame, dict(excluded)
    lookup = {int(r['id']): r for r in records}
    selected = []
    for row in frame.to_dict('records'):
        error = eligible_error(lookup[row['prediction_id']], artifact)
        if error:
            excluded[error] += 1
        else:
            selected.append(row)
    if not selected:
        return pd.DataFrame(), dict(excluded)
    # Batch models once; replay still uses each offer's original scoring version.
    features = pd.DataFrame([lookup[r['prediction_id']]['forecast_payload']['scoring_replay']['row'] for r in selected])
    candidates = curves(artifact['bundle'], features)
    summaries = {v: curve_summary(*candidates[v]) for v in VARIANTS}
    result = []
    for i, row in enumerate(selected):
        record = lookup[row['prediction_id']]
        p = record['forecast_payload']
        row['market'] = locked_market(record)['market_probability']
        row['raw_projection'] = float(p['scoring_replay']['projection'])
        try:
            for variant in VARIANTS:
                values, mass = candidates[variant]
                s = summaries[variant]
                final = changed_probability(p, values[i], mass[i], projection_anchor=s['median'][i])
                raw_over = float(curve_over(values[i:i+1], mass[i:i+1], [p['line']])[0])
                row.update({variant+'_'+k: float(s[k][i]) for k in ('mean', 'median', 'p10', 'p90')})
                row[variant+'_raw_probability'] = raw_over if p['side'] == 'over' else 1-raw_over
                row[variant+'_probability'] = final['probability']
                row[variant+'_trace'] = final['trace']
        except (ValueError, KeyError) as exc:
            excluded[str(exc)] += 1
            continue
        result.append(row)
    return pd.DataFrame(result), dict(excluded)


def run(path, season, weeks):
    artifact = joblib.load(path)
    records = [r for r in load_rows(stat='receiving_yards') if r['season'] == season and r['week'] in weeks]
    frame, excluded = score_records(records, load_ledger(), artifact)
    registration = trial.load_registration()
    fixed_ids = {int(r['forecast_id']) for doc in trial.captures(registration) for r in doc.get('selected', [])}
    report = dict(run_id=artifact['run_id'], built_at=datetime.now(timezone.utc).isoformat(),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), season=season, weeks=weeks,
        input_rows=len(records), exclusions=excluded, evaluation=evaluate(frame, fixed_ids),
        rows=frame.to_dict('records'), production_changed=False,
        limitations=['Retrospective replay, not a prospective capture or cash approval.',
            'Only immutable pregame inputs; no current context or reconstructed legacy forecasts.',
            'Same offered side and price for comparisons, including the final scoring adjustments.',
            'Mean and median are separate. The candidate median supplies the central pricing anchor.',
            'Integer lines are excluded until the full replay push path is validated.',
            'Raw forecast intervals are not intervals implied by independently adjusted offer probabilities.',
            'Missing outcomes remain pending; original micro/research IDs are not replaced.',
            'Scoring versions stay separate. Repeated offers share a player-game weight.'])
    atomic_json(path.parent/'offer_validation.json', clean(report))
    atomic_json(ROOT/'reports/nfl_receiving_quantiles_offers_latest.json', clean(report))
    text = ['# Receiving Quantiles: Actual FanDuel Offers', '',
        '| Cohort / pool | Settled | Production | Quantile raw | Quantile final | Blend raw | Blend final | Market |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, result in report['evaluation'].get('cohorts', {}).items():
        for name in ('all_offers', 'exact_micro', 'fixed_research', 'unchanged_ranking', *(v+'_ranked' for v in VARIANTS)):
            entry = result[name]
            metrics = entry.get('probabilities', {})
            values = [f"{metrics[k]['brier']:.4f}" if metrics.get(k, {}).get('rows') else '-'
                for k in ('production', 'quantile_raw', 'quantile_final', 'calibrated_blend_raw', 'calibrated_blend_final', 'market')]
            text.append(f"| {cohort} / {name} | {entry['settled']} | "+' | '.join(values)+' |')
    text += ['', 'Exclusions: '+json.dumps(excluded), '', '## Model Influence']
    for cohort, result in report['evaluation'].get('cohorts', {}).items():
        text.append(cohort+': '+json.dumps(result.get('scoring_influence', {})))
    text += ['', *report['limitations']]
    (ROOT/'reports/nfl_receiving_quantiles_offers_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--weeks', type=int, nargs='+', required=True)
    a = p.parse_args()
    r = run(a.artifact, a.season, a.weeks)
    print(json.dumps(dict(rows=len(r['rows']), exclusions=r['exclusions'], evaluation=r['evaluation'])))
