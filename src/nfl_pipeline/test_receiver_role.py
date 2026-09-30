from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.receiver_role_model import (
    RoleHistory, RoleAwareReceiver, RoleConfig, context_inputs, inputs, role_features)


def record(i, *, player='p', team='A', yards=40., targets=5., snaps=50.):
    return {'game_id': f'g{i}', 'player_id': player, 'team_abbr': team, 'opponent_abbr': 'B',
        'position': 'WR', 'season': 2025, 'week': i + 1,
        'game_date_et': date(2025, 9, 1) + timedelta(days=i * 7),
        'targets': targets, 'receiving_yards': yards, 'receiving_air_yards': yards + 10,
        'offense_snaps': snaps, 'pass_attempts': 30.}


def history(n=8):
    return pd.DataFrame([record(i) for i in range(n)])


def test_no_current_or_same_day_outcome_leakage():
    raw = history()
    requests = pd.DataFrame([record(4), record(5)])
    expected = role_features(raw, requests)
    changed = raw.copy()
    changed.loc[changed.game_date_et >= requests.game_date_et.max(), ['targets', 'receiving_yards', 'offense_snaps']] = 10000
    pd.testing.assert_frame_equal(expected, role_features(changed, requests))
    current = raw.copy()
    current.loc[current.game_date_et == requests.game_date_et.min(), ['targets', 'receiving_yards']] = 10000
    pd.testing.assert_frame_equal(expected.iloc[:1], role_features(current, requests.iloc[:1]))
    # A second game on the same date cannot update the first game's inputs.
    extra = dict(record(4), game_id='g4b', targets=500, receiving_yards=10000)
    pd.testing.assert_frame_equal(role_features(raw, requests.iloc[:1]),
        role_features(pd.concat([raw, pd.DataFrame([extra])]), requests.iloc[:1]))


def test_zero_target_partial_game_not_a_zero_ypt_observation():
    state = RoleHistory()
    for i in range(5):
        state.update([record(i)])
    state.update([record(5, yards=0, targets=0, snaps=14)])
    f = state.features(record(6))
    assert f['rr_last_partial'] == 1
    assert f['rr_targets_3'] == pytest.approx(10 / 3)
    assert f['rr_exposure_40'] == 25
    assert f['rr_yards_40'] == 200


def test_small_target_games_have_small_efficiency_weight():
    state = RoleHistory()
    state.update([record(0, yards=80, targets=10)])
    state.update([record(1, yards=0, targets=1)])
    f = state.features(record(2))
    assert f['rr_yards_40'] / f['rr_exposure_40'] == pytest.approx(80 / 11)
    assert f['rr_yards_40'] / f['rr_exposure_40'] != 4  # Not mean of per-game YPT.


def test_prior_strength_limits_one_game_efficiency_swing():
    frame = pd.DataFrame({'position': ['WR', 'WR'], 'rr_exposure_40': [30., 31.],
        'rr_yards_40': [240., 300.]})
    shifts = []
    for strength in (30., 90.):
        model = RoleAwareReceiver(RoleConfig(prior_targets=strength))
        model.global_rate = 8.
        model.position_rates = {'WR': 8.}
        values = model.efficiency_prior(frame)
        shifts.append(values[1] - values[0])
    assert 0 < shifts[1] < shifts[0]


def test_missing_and_future_context_stay_unknown():
    row = record(5)
    row['prediction_context_cutoff_utc'] = '2025-10-06T12:00:00Z'
    unknown = context_inputs(row)
    assert np.isnan(unknown['rr_injury_out'])
    row['context_evidence'] = {'cutoff': '2025-10-06T13:00:00Z', 'injury_status': 'out',
        'injury_observed_at': '2025-10-06T11:00:00Z'}
    assert np.isnan(context_inputs(row)['rr_injury_out'])
    row['context_evidence']['cutoff'] = '2025-10-06T12:00:00Z'
    assert context_inputs(row)['rr_injury_out'] == 1
    X = inputs(pd.DataFrame([unknown]), 'workload')
    assert X.rr_injury_out__missing.iloc[0] == 1
    assert np.isnan(X.rr_injury_out.iloc[0])


def test_team_change_and_partial_context_not_reset_to_zero():
    state = RoleHistory()
    for i in range(4):
        state.update([record(i)])
    f = state.features(record(4, team='C'))
    assert f['rr_team_changed'] == 1
    assert f['rr_current_team_games'] == 0
    assert f['rr_exposure_40'] == 20
    assert np.isnan(f['rr_team_passes_5'])


def test_omission_rebuilds_exposure_role_and_team_history():
    state = RoleHistory()
    for i in range(5):
        state.update([record(i)])
    extreme = record(5, yards=200, targets=20)
    extreme['pass_attempts'] = 60
    state.update([extreme])
    full = state.features(record(6), 1)
    down = state.features(record(6), .25)
    omit = state.features(record(6), 0)
    assert omit['rr_exposure_40'] == 25
    assert omit['rr_yards_40'] == 200
    assert omit['rr_targets_3'] == 5
    assert omit['rr_team_passes_5'] == 30
    assert omit['rr_targets_3'] < down['rr_targets_3'] < full['rr_targets_3']
    assert omit['rr_team_passes_5'] < down['rr_team_passes_5'] < full['rr_team_passes_5']


def test_features_exclude_actuals_and_identity():
    raw = history(); frame = pd.concat([raw, role_features(raw, raw)], axis=1)
    for kind in ('team', 'workload', 'rate', 'all'):
        X = inputs(frame, kind)
        assert all(c.startswith('rr_') for c in X)
        assert 'targets' not in X and 'receiving_yards' not in X and 'player_id' not in X
    assert not any(c.startswith('rr_yards_') for c in inputs(frame, 'workload'))


def test_model_uses_finite_exposure_and_team_budget():
    raw = history(12)
    frame = pd.concat([raw, role_features(raw, raw)], axis=1).iloc[3:].copy()
    frame['team_actual_pass_attempts'] = 30.
    model = RoleAwareReceiver(RoleConfig()).fit(frame)
    output = model.components(frame)
    assert np.isfinite(output['center']).all()
    assert (output['targets'] <= output['team_pass_attempts']).all()
    assert (output['targets'] >= 0).all()
    assert model.efficiency_prior(frame)[-1] == pytest.approx(8.)


def test_duplicates_rejected():
    raw = history()
    with pytest.raises(ValueError, match='one row'):
        role_features(raw, pd.concat([raw.iloc[:1], raw.iloc[:1]]))


def test_selection_uses_only_earlier_weeks(monkeypatch):
    from nfl_pipeline.modeling import train_receiver_role as trainer
    raw = history(20)
    frame = pd.concat([raw, role_features(raw, raw)], axis=1)
    frame = pd.concat([frame.assign(player_id=f'p{i}') for i in range(8)], ignore_index=True)
    calls = []

    class Model:
        def __init__(self, config):
            self.config = config

        def fit(self, earlier):
            self.end = earlier.game_date_et.max()
            return self

        def components(self, later):
            assert self.end < later.game_date_et.min()
            calls.append((self.end, later.game_date_et.min()))
            return {'center': np.full(len(later), 40.), 'targets': np.full(len(later), 5.)}

    monkeypatch.setattr(trainer, 'RoleAwareReceiver', Model)
    config, report = trainer.tune_config(frame, {1.: frame, .25: frame})
    assert len(calls) == 8
    assert report['fit_end'] < report['selection_start']
    assert config.latest_weight in (1., .25)


def test_sensitivity_replays_same_offer_all_variants(monkeypatch):
    from nfl_pipeline.modeling import receiver_role_sensitivity as scorer
    from datetime import datetime, timezone
    raw = history()
    request = record(9)
    request['n_games_prev_3'] = 3
    request['receiving_yards_avg_5'] = 40.
    at = datetime(2025, 11, 3, 10, tzinfo=timezone.utc)
    payload = {**request, 'player_name': 'Test Receiver', 'stat': 'receiving_yards',
        'model_version': 'release', 'forecast_features': request,
        'prediction_context_cutoff_utc': '2025-11-03T09:00:00Z',
        'context_evidence': {'cutoff': '2025-11-03T09:00:00Z'},
        'line': 39.5, 'book': 'book', 'price': -110, 'projection': 40., 'side': 'over',
        'probability': .6, 'scoring_replay': {'test': True}}
    artifact = {'production_release': 'release', 'training_end': '2025-11-01', 'run_id': 'test',
        'bundle': {'config': {'latest_weight': .25}}}

    class Model:
        def components(self, frame):
            return {'targets': np.array([5.]), 'yards_per_target': np.array([8.]),
                'team_pass_attempts': np.array([30.])}

    artifact['bundle']['model'] = Model()
    monkeypatch.setattr(scorer, 'curve', lambda *a, **k: (np.array([[20., 40., 60.]]), np.full((1, 3), 1 / 3)))
    seen = []

    def replay(p, values, weights, projection):
        seen.append((p['line'], p['book'], p['price'], p['prediction_context_cutoff_utc']))
        return {'calibrated_over_probability': .6, 'candidate_side': 'over',
            'live_p10': 20., 'live_p90': 60., 'scoring_stage': 'complete_live_path'}

    monkeypatch.setattr(scorer, 'full_path', replay)
    rows, excluded = scorer.score_records([(1, payload, '2025-11-03T18:00:00Z')], raw, artifact, at)
    assert len(rows) == 1 and not excluded
    assert len(seen) == 3 and len(set(seen)) == 1
    assert rows[0]['pregame_scored'] and not rows[0]['betting_eligible']
    assert set(rows[0]['variants']) == {'full', 'downweighted', 'omitted'}
    assert rows[0]['max_probability_swing_pp'] == 0
    artifact['training_end'] = '2025-11-03'
    rows, excluded = scorer.score_records([(1, payload, '2025-11-03T18:00:00Z')], raw, artifact, at)
    assert not rows and excluded['challenger_training_not_before_original_lock_date'] == 1


def test_scoring_version_failure_never_falls_back(monkeypatch):
    from nfl_pipeline.modeling.score_accuracy_components import full_path
    with pytest.raises(ValueError, match='missing_lock_time'):
        full_path({'line': 35.5}, np.array([30., 50.]), np.array([.5, .5]), 40.)


def test_calibration_workload_rebuilt_without_moving_test_line():
    from nfl_pipeline.modeling.train_receiver_role import probability_frame
    frame = pd.DataFrame({'rr_targets_5': [2.5], 'targets_avg_5': [7.],
        'receiving_yards_avg_5': [50.], 'line': [45.5]})
    transformed = probability_frame(frame)
    assert transformed.targets_avg_5.iloc[0] == 2.5
    assert transformed.line.iloc[0] == 45.5
    assert transformed.receiving_yards_avg_5.iloc[0] == 50.
    assert frame.targets_avg_5.iloc[0] == 7.
