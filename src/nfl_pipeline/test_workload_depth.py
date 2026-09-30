from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.workload_depth_model import historical_features, inputs, states, WorkloadDepthModel


def history(n=15):
    rows = []
    for i in range(n):
        for player in ('a', 'b'):
            rows.append(dict(game_id=f'g{i}', player_id=player, team_abbr='T', opponent_abbr='O',
                position='RB', season=2025, week=i+1, game_date_et=date(2025, 1, 1)+timedelta(days=7*i),
                targets=[1., 5., 12.][i % 3], carries=[2., 12., 24.][i % 3],
                receiving_yards=[3., 35., 96.][i % 3], rushing_yards=[8., 48., 110.][i % 3],
                receptions=[1., 3., 8.][i % 3], receiving_air_yards=[1., 25., 120.][i % 3],
                receiving_yards_after_catch=[2., 15., 30.][i % 3], offense_snaps=50., pass_attempts=20.))
    return pd.DataFrame(rows)


def prepared():
    raw = history()
    frame = pd.concat([raw, historical_features(raw, raw)], axis=1)
    frame['team_actual_pass_attempts'] = 40.
    frame['team_actual_carries'] = frame.carries * 2
    frame['completed_air_yards'] = frame.receiving_yards - frame.receiving_yards_after_catch
    return frame.iloc[8:].copy()


def test_date_batch_excludes_current_outcomes_and_other_same_day_game():
    raw = history(); requests = raw.iloc[12:14].copy()
    expected = historical_features(raw, requests)
    changed = raw.copy()
    changed.loc[changed.game_date_et >= requests.game_date_et.min(), ['targets', 'carries', 'receiving_air_yards']] = 999
    pd.testing.assert_frame_equal(expected, historical_features(changed, requests))
    same_day = requests.assign(game_id='extra', targets=999., carries=999.)
    pd.testing.assert_frame_equal(expected, historical_features(pd.concat([raw, same_day]), requests))


def test_carries_only_game_is_not_dropped_from_workload_history():
    raw = history()
    raw['targets'] = 0.; raw['offense_snaps'] = np.nan
    f = historical_features(raw, raw.iloc[-1:])
    assert f.wd_history.iloc[0] > 0
    assert f.wd_carries_5.iloc[0] > 0


def test_missing_context_is_unknown_not_healthy():
    f = historical_features(history(), history().iloc[-1:])
    assert np.isnan(f.wd_injury_out.iloc[0])
    X = inputs(f, 'workload', 'receiving_yards')
    assert X.wd_injury_out__missing.iloc[0] == 1
    assert np.isnan(X.wd_injury_out.iloc[0])


def test_efficiency_feature_boundary_and_exposure_weighting():
    raw = history(3)
    raw.loc[raw.player_id.eq('a'), ['targets', 'receiving_air_yards']] = [[10, 100], [1, 0], [1, 0]]
    f = historical_features(raw, raw.iloc[-2:-1])
    assert f.wd_depth_exposure_40.iloc[0] == 11
    assert f.wd_depth_sum_40.iloc[0] == 100
    frame = prepared()
    work = inputs(frame, 'workload', 'receiving_yards')
    rate = inputs(frame, 'rate', 'receiving_yards')
    assert not any('_sum_' in c for c in work)
    assert 'wd_depth_sum_40' in rate
    assert 'targets' not in work and 'receiving_yards' not in rate
    assert not any('receiving' in c for c in inputs(frame, 'rate', 'rushing_yards'))


@pytest.mark.parametrize('stat', ['receiving_yards', 'rushing_yards'])
def test_state_mass_and_per_player_workload_bound(stat):
    frame = prepared(); model = WorkloadDepthModel().fit(frame, stat)
    c = model.components(frame)
    np.testing.assert_allclose(c['weights'].sum(axis=1), 1.)
    assert np.isfinite(c['centers']).all()
    assert (c['opportunities'] >= 0).all()
    mean = (c['opportunities'] * c['weights']).sum(axis=1)
    assert (mean <= c['team_volume'] + 1e-8).all()
    assert np.isfinite(model.predict(frame)).all()
    subset = model.components(frame.iloc[:1])
    np.testing.assert_allclose(subset['opportunities'], c['opportunities'][:1])
    np.testing.assert_allclose(subset['weights'], c['weights'][:1])
    np.testing.assert_allclose(subset['centers'], c['centers'][:1])


def test_only_verified_out_removes_workload():
    frame = prepared(); model = WorkloadDepthModel().fit(frame, 'receiving_yards')
    normal = model.predict(frame)
    assert (normal > 0).any()
    unknown = frame.copy(); unknown['wd_injury_out'] = np.nan
    np.testing.assert_allclose(normal, model.predict(unknown))
    confirmed = frame.copy(); confirmed['wd_injury_out'] = 1.
    np.testing.assert_allclose(model.predict(confirmed), 0.)


def test_low_normal_high_targets_are_distinct_labels():
    frame = pd.DataFrame({'targets': [0, 5, 12], 'wd_targets_5': [5, 5, 5]})
    np.testing.assert_array_equal(states(frame, 'receiving_yards'), [0, 1, 2])


def test_zero_targets_never_become_zero_efficiency_evidence():
    raw = history(3)
    raw.loc[(raw.player_id == 'a') & (raw.week == 2), ['targets', 'receiving_yards', 'receiving_air_yards']] = [0, 0, 0]
    f = historical_features(raw, raw.iloc[-2:-1])
    assert f.wd_targets_exposure_40.iloc[0] == 1
    assert f.wd_receiving_yards_sum_40.iloc[0] == 3


def test_duplicate_requests_refused():
    raw = history()
    with pytest.raises(ValueError, match='one row'):
        historical_features(raw, pd.concat([raw.iloc[:1], raw.iloc[:1]]))


def test_efficiency_comparison_uses_identical_observations():
    from nfl_pipeline.modeling.train_workload_depth import component_metrics
    rows = pd.DataFrame(dict(game_id=['a', 'b'], player_id=['p', 'q'], actual=[10., 50.],
        actual_opportunity=[2., 5.], predicted_rate=[5., 0.], actual_depth=[4., 20.], predicted_depth=[4., 0.]))
    history = pd.DataFrame(dict(game_id=['a', 'b'], player_id=['p', 'q'], receiving_yards_avg_5=[8., np.nan],
        targets_avg_5=[2., 5.], wd_depth_sum_5=[6., np.nan], wd_depth_exposure_5=[2., 5.]))
    r = component_metrics(rows, history, 'receiving_yards')
    assert r['efficiency']['model']['rows'] == r['efficiency']['baseline']['rows'] == 1
    assert r['efficiency']['model']['mae'] == 0
    assert r['target_depth']['rows'] == r['target_depth_baseline']['rows'] == 1
