import pandas as pd

from .hitter_player_rate_diagnostic import (
    _add_hit_bias_buckets,
    _fit_hit_bias_maps,
    _score_hit_bias_maps,
    choose_rate_blend,
)


def test_rate_blend_accepts_hits_only_when_mae_and_brier_improve() -> None:
    decision = choose_rate_blend("hits", [
        {"alpha": 0.0, "mae_gain": 0.0, "any_brier_gain": 0.0, "bias": -0.02},
        {"alpha": 0.1, "mae_gain": 0.002, "any_brier_gain": 0.001, "bias": -0.02},
        {"alpha": 0.2, "mae_gain": 0.008, "any_brier_gain": 0.001, "bias": -0.03},
    ])

    assert decision["accepted"] is True
    assert decision["alpha"] == 0.2


def test_rate_blend_rejects_home_runs_when_brier_regresses() -> None:
    decision = choose_rate_blend("home_runs", [
        {"alpha": 0.0, "mae_gain": 0.0, "any_brier_gain": 0.0, "bias": 0.02},
        {"alpha": 0.2, "mae_gain": 0.002, "any_brier_gain": -0.001, "bias": 0.02},
    ])

    assert decision["accepted"] is False
    assert decision["alpha"] == 0.0
    assert "any_brier_not_improved" in decision["reason"]


def test_hit_bias_buckets_target_top_order_high_pa_context() -> None:
    df = pd.DataFrame([{
        "lineup_bucket": "slot_1_2",
        "platoon_bucket": "opposite_hand",
        "home_away_bucket": "home",
        "projected_pa": 4.7,
        "player_prior_hit_rate": 0.305,
        "team_implied_runs": 5.1,
        "park_babip_factor": 1.05,
    }])

    out = _add_hit_bias_buckets(df)

    assert out.loc[0, "projected_pa_bucket"] == "projected_pa_high_4_4_plus"
    assert out.loc[0, "player_prior_hit_bucket"] == "prior_hit_rate_high"
    assert out.loc[0, "lineup_pa_bucket"] == "slot_1_2|projected_pa_high_4_4_plus"
    assert out.loc[0, "lineup_prior_bucket"] == "slot_1_2|prior_hit_rate_high"
    assert out.loc[0, "platoon_prior_bucket"] == "opposite_hand|prior_hit_rate_high"


def test_hit_bias_maps_shrink_and_score_repair_residual() -> None:
    train = _add_hit_bias_buckets(pd.DataFrame([
        {
            "player_id": 1,
            "lineup_bucket": "slot_1_2",
            "platoon_bucket": "opposite_hand",
            "home_away_bucket": "home",
            "projected_pa": 4.7,
            "player_prior_hit_rate": 0.305,
            "team_implied_runs": 5.1,
            "park_babip_factor": 1.05,
            "hit_bias_residual": 0.8,
        },
        {
            "player_id": 2,
            "lineup_bucket": "slot_6_9",
            "platoon_bucket": "same_hand",
            "home_away_bucket": "away",
            "projected_pa": 3.3,
            "player_prior_hit_rate": 0.205,
            "team_implied_runs": 3.8,
            "park_babip_factor": 0.94,
            "hit_bias_residual": -0.3,
        },
    ]))
    test = _add_hit_bias_buckets(pd.DataFrame([{
        "player_id": 1,
        "lineup_bucket": "slot_1_2",
        "platoon_bucket": "opposite_hand",
        "home_away_bucket": "home",
        "projected_pa": 4.8,
        "player_prior_hit_rate": 0.310,
        "team_implied_runs": 5.2,
        "park_babip_factor": 1.04,
    }]))

    maps, global_mean = _fit_hit_bias_maps(train)
    scored = _score_hit_bias_maps(test, maps, global_mean)

    assert scored.iloc[0] > 0.0
    assert scored.iloc[0] <= 0.90
