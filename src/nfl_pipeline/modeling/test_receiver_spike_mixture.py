import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import predict_player_props as live
from nfl_pipeline.modeling import train_player_stat_distributions as training


def _params():
    return {
        "state_bias_weight": 0.5,
        "spike_weight_scale": 1.2,
        "low_weight_scale": 0.75,
        "spike_weight_cap": 0.62,
        "states": {
            "low": {"bias": -30.0, "sigma": 8.0},
            "normal": {"bias": -5.0, "sigma": 15.0},
            "target_spike": {"bias": 25.0, "sigma": 25.0},
            "air_yards_spike": {"bias": 35.0, "sigma": 30.0},
            "ypt_tail": {"bias": 45.0, "sigma": 40.0},
        },
    }


@pytest.mark.parametrize("row", [
    {},
    {"targets_avg_5": 2.0, "receiver_teammate_vacancy_score": 0.95, "receiver_target_route_spike_score": np.nan},
    {"targets_avg_5": 8.0, "receiving_yards_avg_5": 70.0, "receiving_air_yards_avg_5": 110.0, "limited_workload_risk_score": 0.8},
])
def test_live_and_training_price_the_same_curve(row):
    params = _params()
    frame = pd.DataFrame([row], index=[42])
    distribution = {"kind": "receiver_spike_mixture_v1", "accepted_distribution": True, "receiver_spike_mixture_params": params}
    weights = training._receiver_spike_mixture_weights(frame, params).iloc[0]
    assert set(weights.index) == set(params["states"])
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    for projection in (1.0, 40.0, 120.0):
        probabilities = []
        for line in (0.5, 30.5, 60.5, 100.5):
            batch = training._receiver_spike_mixture_over_probability(frame, np.array([projection]), np.array([line]), params)[0]
            scalar = live._receiver_spike_mixture_over_probability(projection, line, distribution, row)
            assert batch == pytest.approx(scalar, abs=1e-12)
            assert 0.001 <= scalar <= 0.999
            probabilities.append(scalar)
        assert probabilities == sorted(probabilities, reverse=True)
        batch_mean = training._receiver_spike_mixture_mean(frame, np.array([projection]), params)[0]
        live_mean, live_sigma = live._receiver_spike_mixture_moments(projection, distribution, row)
        assert batch_mean == pytest.approx(max(0.0, live_mean))
        assert live_sigma > 0


def test_missing_one_feature_does_not_erase_known_vacancy_signal():
    frame = pd.DataFrame([{
        "targets_avg_5": 1.0,
        "receiver_teammate_vacancy_score": 0.95,
        "receiver_target_route_spike_score": np.nan,
    }])
    assert training._receiver_spike_signal_frame(frame).iloc[0]["target_spike"] == pytest.approx(0.95)


def test_actual_game_outcomes_cannot_change_live_weights():
    row = {"targets_avg_5": 6.0, "receiving_yards_avg_5": 50.0}
    later = dict(row, targets=18.0, receiving_yards=250.0, receiving_air_yards=220.0, route_participation=1.0)
    assert live._receiver_spike_mixture_weights_single(row, _params()) == live._receiver_spike_mixture_weights_single(later, _params())


@pytest.mark.parametrize("mae,brier,accepted", [
    (19.0, 0.19, True),
    (21.0, 0.19, False),
    (19.0, 0.21, False),
    (20.0, 0.20, False),
    (19.0, float("nan"), False),
])
def test_promotion_requires_both_metrics(mae, brier, accepted):
    assert training._receiver_spike_passes(20.0, mae, 0.20, brier) is accepted


def test_rejected_distribution_does_not_change_live_probability():
    row = {"targets_avg_5": 6.0}
    rejected = {"kind": "receiver_spike_mixture_v1", "accepted_distribution": False, "receiver_spike_mixture_params": _params()}
    expected = live._side_prob("receiving_yards", 50.0, 45.5, {"mae": 20.0}, {}, row)
    assert live._side_prob("receiving_yards", 50.0, 45.5, {"mae": 20.0}, rejected, row) == pytest.approx(expected)


def test_parameter_selection_uses_earlier_weeks_only():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "season": [2025] * 240,
        "week": np.repeat(np.arange(1, 9), 30),
        "position": "WR",
        "targets_avg_5": 6.0,
        "receiving_yards_avg_5": 48.0,
        "targets": rng.integers(1, 12, size=240),
        "receiving_yards": rng.integers(0, 110, size=240),
    })
    pred = np.full(len(frame), 48.0)
    params = training._select_receiver_spike_params(frame, frame.receiving_yards, pred, pred)
    selection = params["selection"]
    assert selection["fit_last_season_week"] == [2025, 4]
    assert selection["validation_season_weeks"] == [[2025, week] for week in range(5, 9)]
    assert selection["holdout_used_for_selection"] is False
    assert selection["candidate_count"] == 315


def test_holdout_evaluation_cannot_retune_candidate():
    params = dict(_params(), status="trained")
    frame = pd.DataFrame([{"targets_avg_5": 6.0}] * 3)
    pred = np.full(3, 48.0)
    original = __import__("copy").deepcopy(params)
    first = training._evaluate_receiver_spike(frame, pd.Series([5.0, 20.0, 30.0]), pred, pred, params, {}, {"mae": 20.0})
    second = training._evaluate_receiver_spike(frame, pd.Series([80.0, 90.0, 100.0]), pred, pred, params, {}, {"mae": 20.0})
    assert params == original
    assert first["receiver_spike_mixture_params"] == second["receiver_spike_mixture_params"]
    assert first["receiver_spike_mixture_mean_mae"] != second["receiver_spike_mixture_mean_mae"]
