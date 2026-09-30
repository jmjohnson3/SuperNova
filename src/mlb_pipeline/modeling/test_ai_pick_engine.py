from __future__ import annotations

from mlb_pipeline.modeling.ai_pick_engine import (
    _apply_selector_metadata,
    ai_pick_to_prop_prediction_row,
    _price_drift_blockers,
    _tier_from_prop,
)


def test_price_drift_accepts_better_american_price() -> None:
    ok, current_ev, min_price, blockers = _price_drift_blockers(
        model_prob=0.55,
        locked_price=110,
        current_price=105,
        minimum_price=-120,
        link="https://sportsbook.example/bet",
        require_current_offer=True,
    )

    assert ok is True
    assert current_ev is not None and current_ev > 0.0
    assert min_price == -120
    assert blockers == []


def test_price_drift_blocks_when_current_price_below_minimum() -> None:
    ok, current_ev, min_price, blockers = _price_drift_blockers(
        model_prob=0.55,
        locked_price=-110,
        current_price=-140,
        minimum_price=-120,
        link="https://sportsbook.example/bet",
        require_current_offer=True,
    )

    assert ok is False
    assert current_ev is not None
    assert min_price == -120
    assert "price_below_minimum" in blockers


def test_tier_from_prop_respects_micro_projection() -> None:
    tier, reasons = _tier_from_prop({
        "bankroll_tier": "micro_projection",
        "bankroll_candidate": False,
        "stat": "pitcher_strikeouts",
        "bet_side": "under",
    })

    assert tier == "micro"
    assert "prediction_tier=micro_projection" in reasons


def test_tier_from_prop_keeps_one_sided_fanduel_separate() -> None:
    tier, reasons = _tier_from_prop({
        "bankroll_tier": "paper",
        "bankroll_candidate": False,
        "stat": "batter_total_bases",
        "bet_side": "over",
        "bookmaker_key": "fanduel",
        "book_line": 1.5,
    })

    assert tier == "one_sided_fanduel"
    assert any(reason.startswith("ledger_class=") for reason in reasons)


def test_ai_pick_to_prop_prediction_row_marks_micro_candidate() -> None:
    row = ai_pick_to_prop_prediction_row({
        "game_date_et": "2026-07-28",
        "game_slug": "2026-07-28-nyy-bos",
        "player_id": 123,
        "player_name": "Example Player",
        "team_abbr": "NYY",
        "stat": "batter_total_bases",
        "side": "over",
        "line": 1.5,
        "current_price": 120,
        "minimum_acceptable_price": 105,
        "model_prob": 0.54,
        "current_ev": 0.19,
        "edge": 0.04,
        "recommendation_tier": "micro",
        "pick_status": "bettable_now",
        "qualifies_now": True,
        "prediction_key": "pred-1",
        "prop_offer_id": 99,
        "link": "https://sportsbook.example/bet",
        "blockers": [],
        "reasons": ["ai_pick_engine_source"],
        "model_meta": {"pair_quality": "same_book"},
    })

    assert row["selector_tier"] == "micro_projection"
    assert row["bankroll_tier"] == "micro_projection"
    assert row["micro_projection_candidate"] is True
    assert row["pred_prob_over"] == 0.54
    assert row["minimum_acceptable_price"] == 105


def test_apply_selector_metadata_promotes_micro_probability(monkeypatch) -> None:
    def fake_score_prediction_row(row, *, ctx, cfg):
        assert row["bet_price"] == -110
        return {
            "selector_tier": "micro_projection",
            "micro_projection_candidate": True,
            "micro_projection_prob_side": 0.574,
            "micro_projection_ev": 0.096,
            "pair_quality": "same_book",
        }

    monkeypatch.setattr(
        "mlb_pipeline.modeling.ai_pick_engine.score_prediction_row",
        fake_score_prediction_row,
    )

    row = _apply_selector_metadata(
        {"bankroll_tier": "watch"},
        current_price=-110,
        ctx=object(),
        cfg=object(),
    )

    assert row["bankroll_tier"] == "micro_projection"
    assert row["micro_projection_prob_side"] == 0.574
    assert row["pair_quality"] == "same_book"
