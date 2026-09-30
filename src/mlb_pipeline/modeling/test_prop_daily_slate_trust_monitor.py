from __future__ import annotations

from datetime import date

from .prop_daily_slate_trust_monitor import evaluate_trust


def _checkpoint(hitter_complete: bool = True, pitcher_complete: bool = True) -> dict:
    return {
        "status": "evaluation_ready_no_micro_buckets",
        "artifact_integrity": {"valid": True},
        "hitter_release": {
            "completed_date_count": 5,
            "dates_remaining": 0,
            "micro_ready_buckets": 0,
            "by_date": [{
                "game_date_et": "2026-07-06",
                "graded_rows": 120,
                "pending_rows": 0 if hitter_complete else 1,
                "void_rows": 8,
                "complete": hitter_complete,
            }],
        },
        "pitcher_release": {
            "completed_date_count": 5,
            "dates_remaining": 0,
            "micro_ready_buckets": 0,
            "by_date": [{
                "game_date_et": "2026-07-06",
                "graded_rows": 28,
                "pending_rows": 0 if pitcher_complete else 1,
                "void_rows": 0,
                "complete": pitcher_complete,
            }],
        },
    }


def _close(final: bool = True, coverage: float = 0.91, stale: float = 0.01) -> dict:
    return {
        "slate_final": final,
        "final_games": 10 if final else 8,
        "games": 10,
        "locked_offer_rows": 1000,
        "valid_closes": int(1000 * coverage),
        "valid_close_coverage": coverage,
        "stale_close_rate": stale,
        "failure_reasons": {"valid_close": int(1000 * coverage)},
        "target_capture": {"events": 10, "all_targets_captured_events": 10},
    }


def test_trust_monitor_passes_clean_final_slate() -> None:
    payload = evaluate_trust(_close(), _checkpoint(), slate_date=date(2026, 7, 6))
    assert payload["status"] == "pass"
    assert payload["required_failures"] == []


def test_trust_monitor_is_provisional_before_games_final() -> None:
    payload = evaluate_trust(_close(final=False), _checkpoint(), slate_date=date(2026, 7, 6))
    assert payload["status"] == "provisional"
    assert [row["name"] for row in payload["required_failures"]] == ["all_games_finalized"]


def test_trust_monitor_fails_final_slate_below_close_coverage() -> None:
    payload = evaluate_trust(_close(coverage=0.877), _checkpoint(), slate_date=date(2026, 7, 6))
    assert payload["status"] == "fail"
    assert "valid_close_coverage" in [row["name"] for row in payload["required_failures"]]
