"""Replay target-only changes at original offers, or capture a passed challenger pregame."""
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

from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import receiving_research_trial as trial
from nfl_pipeline.modeling.receiving_trial_checkpoint import eligible_error
from nfl_pipeline.modeling.live_scoring_replay import load_rows, ROOT
from nfl_pipeline.modeling.prospective_market import locked_market
from nfl_pipeline.modeling.scoring_capture import replay
from nfl_pipeline.modeling.score_accuracy_components import empirical_residuals
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.evaluation import projection_metrics, probability_metrics, clustered_gain
from nfl_pipeline.modeling.receiving_selection_experiment import ranked
from nfl_pipeline.modeling.role_context_training import load_sources, RoleEvidenceIndex
from nfl_pipeline.modeling.yardage_lock_validation import OP_KEYS


def changed_probability(payload, values, weights, projection_anchor=None):
    captured = payload['scoring_replay']
    original = replay(captured)
    if original['side'] != payload['side'] or abs(original['probability']-payload['probability']) > 1e-10:
        raise ValueError('original_scoring_replay_mismatch')
    if float(payload['line']).is_integer():
        raise ValueError('integer_push_curve_not_in_this_trial')
    point = float(np.dot(values, weights)) if projection_anchor is None else float(projection_anchor)
    revised = deepcopy(captured)
    revised['projection'] = point
    revised['distribution'] = dict(kind='empirical_oof_residual',
        residual_quantiles=empirical_residuals(values, weights, point),
        projection_confidence=(captured.get('distribution') or {}).get('projection_confidence', .5))
    out = replay(revised)
    over = out['probability'] if out['side'] == 'over' else 1-out['probability']
    return dict(probability=over if payload['side'] == 'over' else 1-over,
        projection=point, candidate_side=out['side'], trace=out.get('probability_trace'),
        confidence=(captured.get('distribution') or {}).get('projection_confidence', .5))


def scoring_inputs(record, index):
    p = record['forecast_payload']; captured = p.get('scoring_replay') or {}
    features = dict(captured.get('row') or {})
    if number(features.get('targets_avg_5')) is None:
        raise ValueError('missing_archived_target_history')
    old_op = next((number(features.get(k)) for k in OP_KEYS['receiving_yards']
                   if number(features.get(k)) is not None), None)
    if old_op is None or old_op <= 0:
        raise ValueError('missing_original_workload_projection')
    request = dict(features, game_id=record['game_id'], player_id=record['player_id'],
        team_abbr=p.get('team_abbr') or features.get('team_abbr'), season=record['season'], week=record['week'],
        season_type=record.get('season_type') or p.get('season_type') or features.get('season_type'),
        role_lock_cutoff=p['prediction_context_cutoff_utc'], role_lock_id=record['id'],
        start_ts_utc=record['start_ts_utc'])
    features['target_role_evidence'] = index.resolve(request)
    return p, pd.DataFrame([features]), old_op, features['target_role_evidence']


def score_record(record, artifact, index):
    p, frame, old_op, evidence = scoring_inputs(record, index)
    if artifact.get('contract') == 'nfl-target-components-v1':
        return score_components(p, frame, old_op, evidence, artifact)
    model = artifact['bundle']['model']
    output = {}
    for variant in ('control', 'challenger'):
        v, w, c = model.yardage_curve(frame, None, variant, projection=[float(p['projection'])], baseline_targets=[old_op])
        summary = curve_summary(v,w)
        result = changed_probability(p, v[0], w[0])
        output[variant] = dict(result, targets=float(c['targets'][0]), fixed_efficiency=float(c['rate'][0]),
            state_probabilities=c['state_probabilities'][0].tolist(), p10=float(summary['p10'][0]), p90=float(summary['p90'][0]))
    return dict(output, role_evidence=evidence, original_targets=old_op)


def shifted_probability(payload, shift):
    """Change only the center, preserving the actual frozen uncertainty object."""
    captured = payload['scoring_replay']
    original = replay(captured)
    if original['side'] != payload['side'] or abs(original['probability']-payload['probability']) > 1e-10:
        raise ValueError('original_scoring_replay_mismatch')
    if float(payload['line']).is_integer():
        raise ValueError('integer_push_curve_not_in_this_trial')
    revised = deepcopy(captured)
    revised['projection'] = float(captured['projection'])+float(shift)
    result = replay(revised)
    over = result['probability'] if result['side'] == 'over' else 1-result['probability']
    return dict(probability=over if payload['side'] == 'over' else 1-over,
        projection=revised['projection'], candidate_side=result['side'], trace=result.get('probability_trace'))


def score_components(p, frame, old_op, evidence, artifact):
    from nfl_pipeline.modeling.target_components import curves
    distribution = p['scoring_replay'].get('distribution') or {}
    if distribution.get('kind') != 'empirical_oof_residual':
        raise ValueError('existing_uncertainty_not_supported_for_component_ablation')
    variants = curves(artifact['bundle'], frame, projection=[float(p['projection'])],
        baseline_targets=[old_op], residuals=distribution.get('residual_quantiles', []))
    reference_mean = float(np.dot(variants['reference'][0][0], variants['reference'][1][0]))
    output = {}
    for name, (v, w, targets) in variants.items():
        summary = curve_summary(v, w)
        if name in ('reference', 'targets_only'):
            result = shifted_probability(p, float(summary['mean'][0])-reference_mean)
        else:
            anchor = float(p['projection'])+float(summary['mean'][0])-reference_mean
            result = changed_probability(p, v[0], w[0], projection_anchor=anchor)
        output[name] = dict(result, targets=float(targets[0]), expected_yards=float(summary['mean'][0]),
            typical_yards=float(summary['median'][0]), p10=float(summary['p10'][0]), p90=float(summary['p90'][0]),
            fixed_efficiency=float(p['projection'])/max(.5, old_op))
    return dict(output, role_evidence=evidence, original_targets=old_op)


def metrics(frame, columns=('control_probability','challenger_probability')):
    if frame.empty:
        return dict(rows=0)
    settled = frame.loc[frame.outcome.notna()].copy()
    if settled.empty:
        return dict(rows=len(frame), settled=0)
    weight = 1/settled.groupby(['model_version','scoring_version','game_id','player_id']).prediction_id.transform('size')
    return dict(rows=len(frame), settled=len(settled), player_games=len(settled[['game_id','player_id']].drop_duplicates()),
        weeks=len(settled[['season','week']].drop_duplicates()),
        probability={key: probability_metrics(settled[key], settled.outcome, weight)
            for key in ('probability','market',*columns)},
        selected_ids=frame.prediction_id.astype(int).tolist())


def evaluate_rows(frame, fixed_ids, variants=('control', 'challenger')):
    if frame.empty:
        return dict(status='no_replayable_offers', deployment_approved=False)
    by_cohort = {}
    for key, group in frame.groupby(['model_version','scoring_version']):
        distinct = group.drop_duplicates(['game_id','player_id'], keep='first')
        columns = tuple(name+'_probability' for name in variants)
        groups = dict(all_offers=metrics(group, columns),
            fixed_research=metrics(group.loc[group.prediction_id.isin(fixed_ids)], columns),
            current_ranked=metrics(ranked(group), columns))
        for name in variants:
            groups[name+'_ranked'] = metrics(ranked(group, probability=name+'_probability'), columns)
        for variant in ('production',*variants):
            groups[variant+'_yardage'] = projection_metrics(distinct.actual, distinct[variant+'_projection'],
                distinct[variant+'_p10'], distinct[variant+'_p90'])
            groups[variant+'_targets'] = projection_metrics(distinct.actual_targets, distinct[variant+'_targets'])
            if variant+'_expected_yards' in distinct:
                groups[variant+'_expected_yards'] = projection_metrics(distinct.actual, distinct[variant+'_expected_yards'])
                groups[variant+'_typical_yards'] = projection_metrics(distinct.actual, distinct[variant+'_typical_yards'])
        comparable = group.loc[group.outcome.notna()].copy()
        if len(comparable):
            for name in variants:
                p = comparable.assign(reference_error=(comparable.probability-comparable.outcome)**2,
                    challenger_error=(comparable[name+'_probability']-comparable.outcome)**2)
                groups[name+'_brier_gain'] = clustered_gain(p.groupby(['season','week','game_id','player_id'])[
                    ['reference_error','challenger_error']].mean().reset_index())
        by_cohort[str(key)] = groups
    return dict(status='evaluated', cohorts=by_cohort, deployment_approved=False,
        interval_contract='Raw shared-efficiency mixture intervals. Per-line market adjustments are not a certified full CDF.',
        limitations=['Archived eligible offers only, not the complete book universe.',
            'Ranking function and daily cap are unchanged; counterfactual picks are not placed bets.',
            'Original fixed research IDs are evaluated separately and never replaced.',
            'Week 3 informed development and cannot grant deployment approval.'])


def run(artifact_path, season, week, capture=False):
    now = datetime.now(timezone.utc)
    artifact = joblib.load(artifact_path)
    components = artifact.get('contract') == 'nfl-target-components-v1'
    variants = ('reference','targets_only','uncertainty_only','combined') if components else ('control','challenger')
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if capture and components:
        raise ValueError('Component prospective registration is required; diagnostics cannot approve deployment')
    if capture and not artifact.get('historical_screen_passed'):
        raise ValueError('Rejected target-volume model cannot enter prospective scoring')
    registration = trial.load_registration()
    records = [r for r in load_rows(stat='receiving_yards') if r['season'] == season and r['week'] == week]
    fixed_ids = {int(r['forecast_id']) for d in trial.captures(registration) for r in d.get('selected', [])}
    locks, observations = load_sources([r['game_id'] for r in records], now)
    phases = {r['game_id']: r.get('season_type') for r in locks}
    index = RoleEvidenceIndex(observations)
    excluded = Counter(); rows = []
    for rec in records:
        rec['season_type'] = phases.get(rec['game_id'])
        p = rec['forecast_payload']; lock = safe_time(rec['created_at_utc']); start = safe_time(rec['start_ts_utc'])
        error = eligible_error(rec, registration['config'])
        if error:
            excluded[error] += 1; continue
        if p['model_version'] != artifact['production_release'] or artifact['training_end'] >= lock.date().isoformat():
            excluded['training_or_release_mismatch'] += 1; continue
        quote = safe_time((p.get('scoring_replay') or {}).get('offer', {}).get('fetched_at_utc'))
        if capture and (not quote or start <= now or (now-quote).total_seconds() > registration['config']['rules']['max_quote_age_minutes']*60):
            excluded['capture_not_fresh_pregame'] += 1; continue
        try:
            scored = score_record(rec, artifact, index)
        except (ValueError, KeyError) as exc:
            excluded[str(exc)] += 1; continue
        at = datetime.now(timezone.utc)
        if capture and (at >= start or (at-quote).total_seconds() > registration['config']['rules']['max_quote_age_minutes']*60):
            excluded['quote_expired_during_capture'] += 1; continue
        price = float(p['price']); payout = price/100 if price > 0 else 100/-price
        actual = number(rec.get('actual')); line = float(p['line'])
        outcome = None if actual is None or actual == line else float(actual > line if p['side'] == 'over' else actual < line)
        row = dict(prediction_id=rec['id'], game_id=rec['game_id'], player_id=rec['player_id'], season=season, week=week,
            model_version=p['model_version'], scoring_version=p['scoring_replay']['scoring_fingerprint'],
            day=str(p['game_date_et']), batch=p['prediction_context_cutoff_utc'], locked_at=lock.isoformat(),
            captured_at=at.isoformat(), line=line, side=p['side'], book=p['book'], price=price,
            production_projection=p['projection'], production_p10=p.get('projection_p10'), production_p90=p.get('projection_p90'),
            production_targets=scored['original_targets'], probability=float(p['probability']),
            market=locked_market(rec)['market_probability'], actual=actual,
            actual_targets=number(rec.get('actual_targets')) if actual is not None else None,
            outcome=outcome, payout=payout, push_probability=0.,
            eligible=bool(p.get('link') and p.get('drift_guard_pass')), edge=float(p.get('edge') or 0),
            confidence=float(p.get('projection_confidence') or 0), role_evidence=scored['role_evidence'])
        for name in variants:
            row.update({name+'_'+k: v for k,v in scored[name].items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    report = dict(run_id=artifact['run_id'], artifact_sha256=digest, built_at=now.isoformat(), season=season, week=week,
        evidence_mode='pregame_capture' if capture else 'retrospective_diagnostic', exclusions=dict(excluded),
        result=evaluate_rows(frame,fixed_ids,variants), rows=rows, production_changed=False, betting_approved=False,
        acceptance_eligible=False if not capture else now.date().isoformat() >= artifact['prospective_acceptance_start'])
    if capture:
        # No settlement data is persisted with prospective forecasts.
        for row in rows:
            for key in ('outcome','actual','actual_targets'):
                row.pop(key, None)
        atomic_json(artifact_path.parent/'prospective'/now.strftime('%Y%m%dT%H%M%S%fZ.json'), clean(report))
    prefix = 'nfl_target_components' if components else 'nfl_target_volume'
    name = f'{prefix}_offers_{season}_week{week}'
    atomic_json(ROOT/'reports'/(name+'.json'), clean(report))
    text = ['# Target Volume: Exact-Lock Validation', '', report['evidence_mode'], '',
        'Production, fixed selections, scoring adjustments and ranking rules were not changed.', '',
        '| Cohort / pool | Settled | Production Brier | '+' | '.join(variants)+' | Market Brier |',
        '|---|---:|---:|'+'---:|'*len(variants)+'---:|']
    for cohort, groups in report['result'].get('cohorts', {}).items():
        for pool in ('all_offers','fixed_research','current_ranked',*(v+'_ranked' for v in variants)):
            m = groups[pool]; probabilities = m.get('probability', {})
            vals = [f"{probabilities[k]['brier']:.4f}" if probabilities.get(k,{}).get('rows') else '-'
                    for k in ('probability',*(v+'_probability' for v in variants),'market')]
            text.append(f"| {cohort} / {pool} | {m.get('settled',0)} | " + ' | '.join(vals) + ' |')
    text += ['', 'Exclusions: '+json.dumps(dict(excluded)), '',
        'Raw interval coverage is separate from final offered-line probability accuracy.',
        'No deployment approval. Week 3 is development evidence, not an untouched acceptance set.']
    (ROOT/'reports'/f'{prefix}_offers_{season}_week{week}.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--season', type=int, required=True); p.add_argument('--week', type=int, required=True)
    p.add_argument('--capture', action='store_true')
    a = p.parse_args(); result = run(a.artifact, a.season, a.week, a.capture)
    print(json.dumps(dict(rows=len(result['rows']), exclusions=result['exclusions'], mode=result['evidence_mode'])))
