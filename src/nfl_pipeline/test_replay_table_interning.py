from copy import deepcopy
import pytest

from nfl_pipeline.modeling import live_scoring_replay as replay
from nfl_pipeline.modeling.live_scoring_replay import intern_forecast_inputs


def test_captured_tables_are_lossless_and_shared():
    table = [dict(key=['stat', '', 'receiving_yards', '', '50_60'], value=dict(rows=20))]
    payload = dict(scoring_replay=dict(row={'known_at_lock': 1}, calibration=table, clv_guards=[]))
    expected = [dict(id=i, forecast_payload=deepcopy(payload)) for i in (1, 2)]
    cache = {}
    restored = [intern_forecast_inputs(row, cache) for row in deepcopy(expected)]
    assert restored == expected
    assert restored[0]['forecast_payload']['scoring_replay']['calibration'] is restored[1]['forecast_payload']['scoring_replay']['calibration']
    copied = deepcopy(restored[0]['forecast_payload'])
    copied['scoring_replay']['calibration'][0]['value']['rows'] = 0
    assert restored == expected


def test_absent_table_is_not_reconstructed():
    row = dict(forecast_payload={'scoring_replay': {'calibration': None}})
    assert intern_forecast_inputs(row, {}) == dict(forecast_payload={'scoring_replay': {'calibration': None}})


def test_player_inputs_are_interned_without_rounding_or_losing_unknowns():
    features = dict(targets=7.123456789, injury=None, historical_values=[1, 2])
    row = dict(forecast_payload=dict(forecast_features=features, scoring_replay=dict(row=deepcopy(features))))
    before = deepcopy(row)
    cache = {}
    first = intern_forecast_inputs(row, cache)
    second = intern_forecast_inputs(deepcopy(before), cache)
    assert first == second == before
    assert first['forecast_payload']['forecast_features'] is second['forecast_payload']['scoring_replay']['row']


def test_hash_collision_never_substitutes_different_captured_inputs(monkeypatch):
    class Digest:
        def digest(self):
            return b'collision'
    monkeypatch.setattr(replay.hashlib, 'sha256', lambda value: Digest())
    row = dict(forecast_payload=dict(forecast_features={'targets': 7}, model_inputs={'targets': 8}))
    with pytest.raises(ValueError, match='replay_input_digest_collision'):
        intern_forecast_inputs(row, {})


def test_interning_preserves_integer_and_float_input_types():
    row = dict(forecast_payload=dict(forecast_features={'targets': 7}, model_inputs={'targets': 7.0}))
    intern_forecast_inputs(row, {})
    assert type(row['forecast_payload']['forecast_features']['targets']) is int
    assert type(row['forecast_payload']['model_inputs']['targets']) is float
