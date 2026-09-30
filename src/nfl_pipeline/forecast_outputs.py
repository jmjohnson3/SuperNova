"""Point forecasts and bet pricing have distinct, immutable output identities."""
from copy import deepcopy
import hashlib
import json
import math

from nfl_pipeline.context_contract import safe_time

CONTRACT = 'nfl-forecast-outputs-v1'
POINT_OUTPUTS = ('expected_yards', 'typical_yards')


def describe(row):
    """Describe existing outputs without relabeling a central forecast as a mean."""
    captured = row.get('scoring_replay') or {}
    distribution = captured.get('distribution') or row.get('forecast_distribution') or {}
    distribution_id = hashlib.sha256(json.dumps(distribution, sort_keys=True, default=str).encode()).hexdigest()
    return dict(contract=CONTRACT,
        scope={k: row.get(k) for k in ('game_id', 'player_id', 'stat', 'model_version', 'prediction_context_cutoff_utc')},
        point_forecast=dict(value=row.get('projection'),
            semantics=row.get('projection_semantics') or 'unspecified_central_forecast',
            model_version=row.get('model_version')),
        expected_yards=None, typical_yards=None,
        pricing=dict(projection_anchor=captured.get('projection', row.get('projection')),
            model_version=row.get('model_version'), distribution_id=distribution_id,
            distribution_kind=distribution.get('kind') or row.get('distribution_kind'),
            scoring_version=captured.get('scoring_fingerprint'),
            line=row.get('line'), side=row.get('side'), price=row.get('price'),
            probability=row.get('probability'), push_probability=row.get('push_probability'), ev=row.get('ev')),
        betting_approval_inherited=False)


def for_lock(row):
    outputs = describe(row)
    previous = row.get('forecast_outputs') or {}
    if any(previous.get(k) is not None for k in POINT_OUTPUTS):
        if previous.get('contract') != CONTRACT or previous.get('scope') != outputs['scope']:
            raise ValueError('Independent point output does not match this forecast context')
        for key in POINT_OUTPUTS:
            outputs[key] = deepcopy(previous.get(key))
    return outputs


def apply_point_forecast(row, *, output, value, model_version, decision):
    """Deploy a validated point output without changing pricing, ranking or tiers."""
    if output not in POINT_OUTPUTS or not str(row.get('stat', '')).endswith('yards'):
        raise ValueError('Not an independent yardage point output')
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError('Invalid yardage point forecast')
    approved = safe_time(decision.get('approved_at'))
    cutoff = safe_time(row.get('prediction_context_cutoff_utc'))
    if not (decision.get('output') == output and decision.get('model_version') == model_version and
            decision.get('production_release') == row.get('model_version') and
            decision.get('historical_screen_passed') is True and
            decision.get('prospective_validation_passed') is True and
            decision.get('deployment_approved') is True and decision.get('evidence_sha256') and
            approved and cutoff and approved <= cutoff):
        raise ValueError('Point forecast requires its own pre-lock deployment decision')
    result = deepcopy(row)
    outputs = deepcopy(result.get('forecast_outputs') or describe(result))
    outputs[output] = dict(value=float(value), model_version=model_version,
        evidence_sha256=decision['evidence_sha256'], approved_at=approved.isoformat())
    result['forecast_outputs'] = outputs
    # The legacy projection remains the pricing anchor, including in replay and EV calculations.
    return result


def display_suffix(row):
    outputs = row.get('forecast_outputs') or {}
    if outputs.get('contract') != CONTRACT:
        return ''
    pieces = []
    for name, label in (('expected_yards', 'Expected yards'), ('typical_yards', 'Typical yards')):
        item = outputs.get(name)
        if item:
            pieces.append(f"{label}={float(item['value']):.1f} [{item['model_version']}; point-only]")
    return ' | '+'; '.join(pieces) if pieces else ''
