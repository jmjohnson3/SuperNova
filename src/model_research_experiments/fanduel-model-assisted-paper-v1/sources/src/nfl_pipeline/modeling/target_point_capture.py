"""Pinned prospective test for one point output, independent of bet pricing."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, active_release, atomic_json
from nfl_pipeline.modeling.challenger_models import curve_summary
from nfl_pipeline.modeling.evaluation import clustered_gain, projection_metrics
from nfl_pipeline.modeling.target_components import CONTRACT, curves
from nfl_pipeline.modeling.yardage_lock_validation import OP_KEYS

STORE = MODEL_ROOT/'target_components'
QUANTILE_STORE = MODEL_ROOT/'receiving_quantiles'


def point_screen(artifact, variant, output):
    from nfl_pipeline.modeling.receiving_quantiles import CONTRACT as QUANTILE_CONTRACT
    key = 'typical_yards' if artifact.get('contract') == QUANTILE_CONTRACT and output == 'median' else output
    return artifact['output_decisions'][variant][key]['historical_screen_passed']


def register(path, variant='uncertainty_only', output='median', *, store=STORE):
    from nfl_pipeline.modeling.receiving_quantiles import CONTRACT as QUANTILE_CONTRACT
    artifact = joblib.load(path)
    if artifact.get('contract') not in (CONTRACT, QUANTILE_CONTRACT) or artifact['production_release'] != active_release()['release_id']:
        raise ValueError('Point component release/contract mismatch')
    if output != 'median':
        raise ValueError('Current production has a comparable median, not a certified expected mean')
    if not point_screen(artifact, variant, output):
        raise ValueError('Point output did not pass its historical screen')
    document = dict(artifact=str(path.resolve()), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        run_id=artifact['run_id'], variant=variant, output=output,
        production_release=artifact['production_release'], registered_at=datetime.now(timezone.utc).isoformat(),
        acceptance_start=artifact.get('prospective_acceptance_start', datetime.now(timezone.utc).date().isoformat()), minimum_weeks=3,
        status='prospective_only', probability_enabled=False, betting_approved=False)
    target = store/'point_registration.json'
    if target.exists():
        old = json.loads(target.read_text())
        if any(old[k] != document[k] for k in ('artifact_sha256','variant','output','production_release')):
            raise ValueError('A different point component is already pinned')
        return old
    atomic_json(target, document)
    return document


def capture_records(records, *, store=STORE, now=None):
    from nfl_pipeline.modeling.receiving_quantiles import CONTRACT as QUANTILE_CONTRACT
    path = store/'point_registration.json'
    if not path.exists():
        return dict(status='no_registered_point_component', captured=0)
    registration = json.loads(path.read_text())
    model_path = Path(registration['artifact'])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != registration['artifact_sha256']:
        raise ValueError('Pinned point artifact checksum mismatch')
    artifact = joblib.load(model_path)
    quantile_model = artifact.get('contract') == QUANTILE_CONTRACT
    if artifact['production_release'] != active_release()['release_id']:
        raise ValueError('Pinned point component production release changed')
    fixed_clock = now is not None
    now = now or datetime.now(timezone.utc)
    output = registration['output']; variant = registration['variant']
    if not point_screen(artifact, variant, output):
        raise ValueError('Pinned point output no longer matches its approval')
    excluded = Counter(); rows = []
    registered = safe_time(registration['registered_at'])
    for fid, p in records:
        if p.get('stat') != 'receiving_yards':
            continue
        captured = p.get('scoring_replay') or {}
        if not captured.get('scoring_fingerprint'):
            excluded['missing_scoring_version'] += 1; continue
        features = captured.get('row') or {}
        cutoff = safe_time(p.get('prediction_context_cutoff_utc'))
        kickoff = safe_time(features.get('start_ts_utc'))
        if not cutoff or not kickoff or not registered <= cutoff <= now < kickoff or cutoff.date().isoformat() < registration['acceptance_start']:
            excluded['not_post_registration_pregame_lock'] += 1; continue
        if p.get('model_version') != artifact['production_release'] or artifact['training_end'] >= cutoff.date().isoformat():
            excluded['wrong_release_or_training_cutoff'] += 1; continue
        distribution = captured.get('distribution') or {}
        if not features.get('position') or number(features.get('targets_avg_5')) is None:
            excluded['missing_archived_target_history'] += 1; continue
        targets = next((number(features.get(k)) for k in OP_KEYS['receiving_yards'] if number(features.get(k)) is not None), None)
        baseline = number(p.get('projection_p50')) if output == 'median' else None
        if quantile_model:
            # Compare raw independent medians, not the market-anchored pricing center.
            point = number(captured.get('projection'))
            errors = np.asarray(distribution.get('residual_quantiles') or [], dtype=float)
            if point is None or not len(errors) or not np.isfinite(errors).all():
                excluded['missing_raw_reference_distribution'] += 1; continue
            baseline = float(point + np.quantile(errors, .5))
            targets = number(features.get('targets_avg_5'))
            if number(features.get('receiving_yards_avg_5')) is None:
                excluded['missing_archived_yardage_history'] += 1; continue
        # Current production does not certify its central forecast as an expected mean.
        if baseline is None or targets is None or targets <= 0 or distribution.get('kind') != 'empirical_oof_residual':
            excluded['missing_comparable_frozen_point_inputs'] += 1; continue
        if not quantile_model and any(k.startswith('role_') for k in artifact['bundle']['model'].columns) and 'target_role_evidence' not in features:
            excluded['missing_verified_role_features'] += 1; continue
        try:
            if quantile_model:
                from nfl_pipeline.modeling.train_receiving_quantiles import curves as quantile_curves
                values, weights = quantile_curves(artifact['bundle'], pd.DataFrame([features]))[variant]
            else:
                values, weights, _ = curves(artifact['bundle'], pd.DataFrame([features]),
                    projection=[float(p['projection'])], baseline_targets=[targets],
                    residuals=distribution.get('residual_quantiles', []))[variant]
        except ValueError as exc:
            excluded[str(exc)] += 1; continue
        value = float(curve_summary(values, weights)['median'][0])
        at = now if fixed_clock else datetime.now(timezone.utc)
        if at >= kickoff:
            excluded['kickoff_during_capture'] += 1; continue
        rows.append(dict(forecast_id=int(fid), game_id=p['game_id'], player_id=p['player_id'],
            season=p['season'], week=p['week'], model_version=p['model_version'],
            scoring_version=captured.get('scoring_fingerprint'), context_cutoff=cutoff.isoformat(),
            kickoff=kickoff.isoformat(), captured_at=at.isoformat(), reference=baseline,
            candidate=value if quantile_model else max(0., value), output=output, variant=variant, run_id=artifact['run_id'],
            reference_basis='pre_offer_distribution_median' if quantile_model else 'locked_display_median'))
    doc = clean(dict(status='point_prospective_only', registration=registration, rows=rows,
        exclusions=dict(excluded), captured_at=now.isoformat(), production_changed=False, betting_approved=False))
    if rows:
        atomic_json(store/'point_prospective'/now.strftime('%Y%m%dT%H%M%S%fZ.json'), doc)
    return dict(status=doc['status'], captured=len(rows), exclusions=dict(excluded))


def _cohort_metrics(rows, minimum_weeks):
    frame = pd.DataFrame(rows)
    gain = clustered_gain(frame.assign(reference_error=abs(frame.reference-frame.actual),
        challenger_error=abs(frame.candidate-frame.actual)))
    passed = bool(gain['clusters'] >= minimum_weeks and gain['lower_95'] is not None and gain['lower_95'] > 0)
    return dict(status='ready_for_point_deployment_review' if passed else 'not_yet_proven',
        player_games=len(frame), reference=projection_metrics(frame.actual, frame.reference),
        candidate=projection_metrics(frame.actual, frame.candidate), clustered_absolute_error_gain=gain,
        prospective_screen_passed=passed, deployment_approved=False, betting_approved=False,
        probability_enabled=False)


def evaluate(*, store=STORE, now=None, report_name='nfl_target_point_prospective_latest.json'):
    from nfl_pipeline.modeling.live_scoring_replay import load_rows, ROOT
    if not (store/'point_registration.json').exists():
        return dict(status='no_registered_point_component', deployment_approved=False, betting_approved=False)
    registration = json.loads((store/'point_registration.json').read_text())
    now = now or datetime.now(timezone.utc)
    paths = sorted((store/'point_prospective').glob('*.json'))
    records = {int(r['id']):r for r in load_rows(stat='receiving_yards')} if paths else {}
    rows = []; seen = set(); excluded = Counter()
    for path in paths:
        doc = json.loads(path.read_text())
        if doc['registration'] != registration:
            excluded['different_registration'] += len(doc['rows']); continue
        for row in doc['rows']:
            if (row.get('run_id') != registration['run_id'] or
                    row.get('variant') != registration['variant'] or
                    row.get('output') != registration['output'] or
                    row.get('model_version') != registration['production_release']):
                excluded['point_component_identity_mismatch'] += 1; continue
            if not row.get('scoring_version'):
                excluded['missing_scoring_version'] += 1; continue
            key = (row['model_version'], row['scoring_version'], row['game_id'], row['player_id'])
            if key in seen:
                excluded['duplicate_player_game_in_cohort'] += 1
                continue
            seen.add(key)
            kickoff = safe_time(row.get('kickoff'))
            if kickoff is not None and kickoff > now:
                excluded['future_game_pending'] += 1; continue
            record = records.get(row['forecast_id'])
            if record is None or record.get('status') != 'final' or record.get('actual') is None:
                excluded['pending_or_missing_verified_result'] += 1; continue
            p = record.get('forecast_payload') or {}
            if (p.get('model_version') != row['model_version'] or
                    (p.get('scoring_replay') or {}).get('scoring_fingerprint') != row['scoring_version'] or
                    str(record['game_id']) != str(row['game_id']) or str(record['player_id']) != str(row['player_id'])):
                excluded['original_forecast_identity_mismatch'] += 1; continue
            rows.append(dict(row, actual=float(record['actual'])))
    result = dict(status='waiting_for_prospective_results', player_games=len(rows), exclusions=dict(excluded),
        unique_player_games=len({(r['game_id'], r['player_id']) for r in rows}),
        deployment_approved=False, betting_approved=False, probability_enabled=False,
        prospective_screen_passed=False, cohorts=[])
    # Never pool changing production/scoring baselines to pass a component's review.
    for release, scoring in sorted({(r['model_version'], r['scoring_version']) for r in rows}):
        cohort = [r for r in rows if (r['model_version'], r['scoring_version']) == (release, scoring)]
        result['cohorts'].append(dict(production_release=release, scoring_version=scoring,
            **_cohort_metrics(cohort, registration['minimum_weeks'])))
    if len(result['cohorts']) == 1:
        result.update(result['cohorts'][0])
    elif len(result['cohorts']) > 1:
        result['status'] = 'separate_cohort_review_required'
    atomic_json(ROOT/'reports'/report_name, clean(dict(result, registration=registration)))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--register', type=Path)
    p.add_argument('--quantiles', action='store_true')
    args = p.parse_args()
    if args.register:
        result = register(args.register, variant='calibrated_blend', store=QUANTILE_STORE) if args.quantiles else register(args.register)
    elif args.quantiles:
        result = evaluate(store=QUANTILE_STORE, report_name='nfl_receiving_quantile_prospective_latest.json')
    else:
        result = evaluate()
        result['quantile_point'] = evaluate(store=QUANTILE_STORE, report_name='nfl_receiving_quantile_prospective_latest.json')
    print(json.dumps(result, default=str))
