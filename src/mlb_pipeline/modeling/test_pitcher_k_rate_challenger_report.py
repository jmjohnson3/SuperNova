from .pitcher_k_rate_challenger_report import (
    _side_probability_from_count,
    choose_conservative_blend,
)


def test_integer_under_probability_excludes_push() -> None:
    under_integer = _side_probability_from_count(5.0, 5.0, "under")
    under_half = _side_probability_from_count(5.0, 5.5, "under")
    over_integer = _side_probability_from_count(5.0, 5.0, "over")
    over_half = _side_probability_from_count(5.0, 5.5, "over")

    assert under_integer is not None
    assert under_half is not None
    assert over_integer == over_half
    assert under_integer < under_half


def test_conservative_blend_accepts_only_when_mae_and_brier_improve() -> None:
    decision = choose_conservative_blend([
        {"alpha": 0.0, "mae_gain": 0.0, "line_brier_gain": 0.0, "bias": 0.02},
        {"alpha": 0.1, "mae_gain": 0.005, "line_brier_gain": 0.002, "bias": 0.02},
        {"alpha": 0.2, "mae_gain": 0.02, "line_brier_gain": 0.001, "bias": 0.03},
    ])

    assert decision["accepted"] is True
    assert decision["alpha"] == 0.2


def test_conservative_blend_rejects_when_brier_regresses() -> None:
    decision = choose_conservative_blend([
        {"alpha": 0.0, "mae_gain": 0.0, "line_brier_gain": 0.0, "bias": 0.01},
        {"alpha": 0.2, "mae_gain": 0.02, "line_brier_gain": -0.001, "bias": 0.02},
    ])

    assert decision["accepted"] is False
    assert decision["alpha"] == 0.0
    assert "line_brier_not_improved" in decision["reason"]
