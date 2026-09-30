from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from mlb_pipeline.modeling.fanduel_one_sided_diagnostic import _raw_fanduel_groups
from mlb_pipeline.modeling.prop_ledger_classification import classify_prop_ledger
from mlb_pipeline.modeling.prop_training_groups import (
    add_player_game_weights,
    dedupe_locked_offer_rows,
    expanding_player_game_folds,
    temporal_player_game_split,
)


def test_player_game_weights_remove_offer_duplication_advantage() -> None:
    rows = pd.DataFrame(
        [
            {"game_date_et": date(2026, 6, 1), "game_slug": "a", "player_id": 1},
            {"game_date_et": date(2026, 6, 1), "game_slug": "a", "player_id": 1},
            {"game_date_et": date(2026, 6, 1), "game_slug": "a", "player_id": 1},
            {"game_date_et": date(2026, 6, 1), "game_slug": "b", "player_id": 2},
        ]
    )
    weighted = add_player_game_weights(rows)
    totals = weighted.groupby("player_game_group")["player_game_weight"].sum()
    assert len(totals) == 2
    assert round(float(totals.iloc[0]), 8) == round(float(totals.iloc[1]), 8)


def test_temporal_split_keeps_player_games_on_one_side() -> None:
    rows = pd.DataFrame(
        [
            {"game_date_et": date(2026, 6, day), "game_slug": f"g{day}", "player_id": player}
            for day in range(1, 11)
            for player in (1, 2)
            for _ in range(2)
        ]
    )
    split = temporal_player_game_split(rows, holdout_days=3, min_train_rows=4, min_holdout_rows=4)
    assert set(split.train["player_game_group"]).isdisjoint(set(split.holdout["player_game_group"]))
    assert split.train["player_game_weight"].notna().all()
    assert split.holdout["player_game_weight"].notna().all()


def test_temporal_split_uses_multi_date_fallback_when_requested_window_is_too_long() -> None:
    rows = pd.DataFrame(
        [
            {"game_date_et": date(2026, 6, day), "game_slug": f"g{day}", "player_id": player}
            for day in range(1, 11)
            for player in range(10)
        ]
    )
    split = temporal_player_game_split(rows, holdout_days=28, min_train_rows=20, min_holdout_rows=20)
    assert split.strategy.startswith("temporal_row_80_20")
    assert split.holdout["game_date_et"].nunique() >= 2
    assert len(split.train) == 80
    assert len(split.holdout) == 20


def test_temporal_split_rebalances_when_holdout_window_consumes_short_history() -> None:
    rows = pd.DataFrame(
        [
            {"game_date_et": date(2026, 6, day), "game_slug": f"g{day}", "player_id": player}
            for day in range(1, 31)
            for player in range(20 if day < 29 else 500)
        ]
    )
    split = temporal_player_game_split(rows, holdout_days=28, min_train_rows=20, min_holdout_rows=20)
    assert split.strategy.startswith("temporal_row_80_20")
    assert len(split.train) / len(rows) >= 0.50
    assert len(split.holdout) / len(rows) <= 0.45


def test_locked_offer_dedupe_collapses_replayed_exact_offer_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "game_date_et": date(2026, 6, 1),
                "game_slug": "g1",
                "player_id": 1,
                "market": "batter_total_bases",
                "side": "over",
                "bookmaker_key": "draftkings",
                "market_line": 1.5,
                "market_price": 120,
                "prop_offer_id": 101,
                "source_created_at": datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
                "run_id": "a",
            },
            {
                "game_date_et": date(2026, 6, 1),
                "game_slug": "g1",
                "player_id": 1,
                "market": "batter_total_bases",
                "side": "over",
                "bookmaker_key": "draftkings",
                "market_line": 1.5,
                "market_price": 120,
                "prop_offer_id": 101,
                "source_created_at": datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
                "run_id": "b",
            },
            {
                "game_date_et": date(2026, 6, 1),
                "game_slug": "g1",
                "player_id": 1,
                "market": "batter_total_bases",
                "side": "over",
                "bookmaker_key": "draftkings",
                "market_line": 2.5,
                "market_price": 220,
                "prop_offer_id": 102,
                "source_created_at": datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
                "run_id": "a",
            },
        ]
    )
    deduped = dedupe_locked_offer_rows(rows)
    assert len(deduped) == 2
    assert deduped.attrs["raw_rows"] == 3
    assert deduped.attrs["deduped_rows"] == 1
    assert set(deduped["prop_offer_id"]) == {101, 102}


def test_expanding_folds_use_non_overlapping_seven_day_windows() -> None:
    rows = pd.DataFrame(
        [
            {"game_date_et": date(2026, 6, day), "game_slug": f"g{day}", "player_id": player}
            for day in range(1, 31)
            for player in range(5)
        ]
    )
    folds = expanding_player_game_folds(
        rows,
        test_window_days=7,
        min_train_dates=9,
        min_train_rows=30,
        min_holdout_rows=20,
    )
    assert len(folds) == 3
    seen_dates: set[date] = set()
    for fold in folds:
        train_dates = set(fold.train["game_date_et"])
        holdout_dates = set(fold.holdout["game_date_et"])
        assert max(train_dates) < min(holdout_dates)
        assert seen_dates.isdisjoint(holdout_dates)
        assert set(fold.train["player_game_group"]).isdisjoint(set(fold.holdout["player_game_group"]))
        seen_dates.update(holdout_dates)


def test_one_sided_fanduel_has_priority_over_lottery() -> None:
    row = {
        "bookmaker_key": "fanduel",
        "market": "batter_home_runs",
        "side": "over",
        "market_line": 1.5,
        "model_tier": "watch",
        "model_meta": {"pair_quality": "one_sided", "selector_tier": "lottery"},
    }
    assert classify_prop_ledger(row) == "one_sided_fanduel"


def test_higher_hit_and_total_base_over_lines_are_lottery_ledgers() -> None:
    base = {
        "bookmaker_key": "draftkings",
        "side": "over",
        "model_tier": "paper",
        "model_meta": {"pair_quality": "same_book"},
    }
    assert classify_prop_ledger({
        **base, "market": "batter_hits", "market_line": 1.5,
    }) == "lottery"
    assert classify_prop_ledger({
        **base, "market": "batter_total_bases", "market_line": 2.5,
    }) == "lottery"
    assert classify_prop_ledger({
        **base, "market": "batter_hits", "market_line": 0.5,
    }) == "paper_common"
    assert classify_prop_ledger({
        **base, "market": "batter_total_bases", "market_line": 1.5,
    }) == "paper_common"


def test_micro_external_agreement_has_own_ledger_bucket() -> None:
    row = {
        "bookmaker_key": "draftkings",
        "market": "pitcher_strikeouts",
        "side": "under",
        "market_line": 4.5,
        "model_tier": "micro_projection",
        "model_meta": {
            "selector_tier": "micro_projection",
            "micro_external_agreement": True,
            "pair_quality": "same_book",
        },
    }
    assert classify_prop_ledger(row) == "micro_external_agreement"


def test_raw_fanduel_diagnostic_preserves_true_sides() -> None:
    payload = [{
        "as_of_date": date(2026, 6, 1),
        "fetched_at_utc": datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
        "payload": {
            "id": "event-1",
            "bookmakers": [{
                "key": "fanduel",
                "markets": [{
                    "key": "batter_total_bases_alternate",
                    "outcomes": [
                        {"name": "Over", "description": "Test Player", "point": 1.5, "price": 120},
                        {"name": "Under", "description": "Test Player", "point": 1.5, "price": -140},
                    ],
                }],
            }],
        },
    }]
    grouped = _raw_fanduel_groups(payload)
    assert len(grouped) == 1
    record = next(iter(grouped.values()))
    assert record["over_price"] == 120.0
    assert record["under_price"] == -140.0
