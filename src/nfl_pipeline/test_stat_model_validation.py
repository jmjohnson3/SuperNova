from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest

from nfl_pipeline.modeling import stat_model_validation as validation
from nfl_pipeline.modeling.stat_model_validation import build_report


def registry():
    return {
        "candidates": {
            "receiving_yards": {"receiving-v2": {
                "component_id": "receiving-v2",
                "outputs": ["expected_mean", "median"],
                "training_end": "2026-09-01",
            }}
        }
    }


def locked_row(index: int, *, result="win"):
    week = index // 40 + 1
    game_day = date(2026, 9, 6) + timedelta(weeks=week - 1)
    start = datetime.combine(game_day, datetime.min.time(), timezone.utc) + timedelta(hours=20)
    lock = start - timedelta(hours=1)
    actual = 50.0 + (index % 7)
    line = actual - 0.5 if index % 2 else actual + 0.5
    outcome = float(actual > line)
    candidate_probability = 0.7 if outcome else 0.3
    return {
        "id": index,
        "game_date_et": game_day,
        "game_id": f"g-{week}-{index // 2}",
        "player_id": f"p-{index}",
        "player_name": f"Player {index}",
        "stat": "receiving_yards",
        "side": "over",
        "line": line,
        "price": -110,
        "book": "fanduel",
        "tier": "micro_projection" if index % 5 == 0 else "paper",
        "created_at_utc": lock,
        "offer_fetched_at": lock - timedelta(minutes=1),
        "start_ts_utc": start,
        "season": 2026,
        "week": week,
        "actual": actual,
        "graded_result": result,
        "forecast_payload": {
            "line": line,
            "side": "over",
            "probability": 0.55,
            "market_no_vig_probability": 0.50,
            "projection": actual + 10,
            "prediction_context_cutoff_utc": (lock - timedelta(seconds=30)).isoformat(),
            "stat_model_challenger": {
                "component_id": "receiving-v2",
                "outputs": ["expected_mean", "median"],
                "same_offer": True,
                "line": line,
                "side": "over",
                "probability": candidate_probability,
                "projection": actual,
                "median": actual,
                "p10": actual - 10,
                "p90": actual + 10,
            },
        },
    }


def test_prospective_validation_uses_identical_offers_and_separates_selection():
    report = build_report([locked_row(i) for i in range(120)], registry())
    cohort = report["cohorts"]["receiving-v2"]
    assert report["eligible_offer_rows"] == 120
    assert cohort["all_offers"]["player_games"] == 120
    assert cohort["all_offers"]["weeks"] == 3
    assert cohort["selected_picks"]["offers"] == 24
    assert cohort["deployment_approved"]
    assert cohort["review_complete"]
    assert set(cohort["approved_outputs"]) == {"expected_mean", "median"}
    assert not cohort["betting_approved"]


def test_unverified_participation_and_mismatched_line_never_count():
    nonparticipant = locked_row(1, result="void_nonparticipant")
    mismatch = locked_row(2)
    mismatch["forecast_payload"]["stat_model_challenger"]["line"] += 1
    report = build_report([nonparticipant, mismatch], registry())
    assert report["eligible_offer_rows"] == 0
    assert report["eligible_player_game_rows"] == 1
    assert report["exclusions"] == {"unverified_participation": 1}
    assert report["probability_exclusions"] == {"shadow_line_mismatch": 1}
    assert not report["cohorts"]["receiving-v2"]["deployment_approved"]


def test_pushes_are_kept_for_projection_but_not_probability():
    row = locked_row(1)
    row["line"] = row["actual"]
    row["forecast_payload"]["line"] = row["actual"]
    row["forecast_payload"]["stat_model_challenger"]["line"] = row["actual"]
    report = build_report([row], registry())
    metrics = report["cohorts"]["receiving-v2"]["all_offers"]
    assert metrics["player_games"] == 1
    assert metrics["probability_rows"] == 0


def test_projection_only_rows_can_prove_point_outputs_without_a_book_line():
    rows = [locked_row(i) for i in range(120)]
    for row in rows:
        row.update(line=None, side=None, price=None, book=None, offer_fetched_at=None, tier="paper")
        payload = row["forecast_payload"]
        payload.update(line=None, side=None, probability=None, market_no_vig_probability=None)
        payload["stat_model_challenger"].update(
            same_offer=False, line=None, side=None, probability=None
        )
    report = build_report(rows, registry())
    cohort = report["cohorts"]["receiving-v2"]
    assert report["eligible_player_game_rows"] == 120
    assert report["eligible_offer_rows"] == 0
    assert report["probability_exclusions"] == {"not_same_offer_shadow": 120}
    assert cohort["all_offers"]["player_games"] == 120
    assert cohort["all_offers"]["probability_rows"] == 0
    assert set(cohort["approved_outputs"]) == {"expected_mean", "median"}
    assert cohort["deployment_approved"]


def test_different_forecast_horizons_do_not_combine_into_deployment_sample():
    rows = [locked_row(i) for i in range(120)]
    for index, row in enumerate(rows):
        start = row["start_ts_utc"]
        minutes = 20 if index < 60 else 120
        lock = start - timedelta(minutes=minutes)
        row["created_at_utc"] = lock
        row["offer_fetched_at"] = lock - timedelta(minutes=1)
        row["forecast_payload"]["prediction_context_cutoff_utc"] = lock.isoformat()
    report = build_report(rows, registry())
    cohort = report["cohorts"]["receiving-v2"]
    assert cohort["all_offers"]["player_games"] == 120
    assert cohort["metrics"]["primary_horizon"] is None
    assert "needs_fixed_horizon_evidence" in cohort["blockers"]
    assert not cohort["review_complete"]
    assert not cohort["deployment_approved"]


def test_forecast_before_candidate_registration_cannot_enter_trial():
    row = locked_row(1)
    registry_doc = registry()
    registry_doc["candidates"]["receiving_yards"]["receiving-v2"]["registered_at"] = (
        row["created_at_utc"] + timedelta(minutes=1)
    ).isoformat()

    report = build_report([row], registry_doc)

    assert report["eligible_shadow_rows"] == 0
    assert report["exclusions"] == {"not_post_registration": 1}


def test_completed_failed_trial_is_retired_only_on_activation_run(monkeypatch, tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    registry_doc = {"candidates": {"receiving_yards": {"receiving-v2": {
        "component_id": "receiving-v2",
        "outputs": ["expected_mean"],
        "evidence": str(evidence),
    }}}}
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {"receiving-v2": {
            "stat": "receiving_yards",
            "metrics": {"player_games": 120, "weeks": 3},
            "approved_outputs": [],
            "deployment_approved": False,
            "review_complete": True,
            "blockers": ["expected_mean_not_proven_prospectively"],
        }},
    }
    cleared = []
    monkeypatch.setattr(
        validation,
        "clear_candidate",
        lambda stat, component_id=None: cleared.append((stat, component_id)) or True,
    )

    unchanged = validation._update_evidence(report, registry_doc, False)
    assert unchanged == {"activated": [], "rejected": []}
    assert cleared == []

    changed = validation._update_evidence(report, registry_doc, True)
    assert changed["activated"] == []
    assert changed["rejected"][0]["component_id"] == "receiving-v2"
    assert cleared == [("receiving_yards", "receiving-v2")]


def test_point_and_probability_challengers_are_evaluated_independently():
    registry_doc = registry()
    registry_doc["candidates"]["receiving_yards"]["receiving-cdf"] = {
        "component_id": "receiving-cdf",
        "outputs": ["probability"],
        "training_end": "2026-09-01",
    }
    rows = [locked_row(i) for i in range(120)]
    for row in rows:
        payload = row["forecast_payload"]
        point = deepcopy(payload["stat_model_challenger"])
        outcome = float(row["actual"] > row["line"])
        probability = deepcopy(point)
        probability.update(
            component_id="receiving-cdf",
            outputs=["probability"],
            probability=.99 if outcome else .01,
            p90=row["actual"] - 1 if row["id"] % 5 == 0 else row["actual"] + 10,
        )
        payload["stat_model_challengers"] = {
            "receiving-v2": point,
            "receiving-cdf": probability,
        }

    report = build_report(rows, registry_doc)

    assert report["eligible_shadow_rows"] == 240
    assert set(report["cohorts"]) == {"receiving-v2", "receiving-cdf"}
    assert set(report["cohorts"]["receiving-v2"]["approved_outputs"]) == {
        "expected_mean", "median"
    }
    assert report["cohorts"]["receiving-cdf"]["approved_outputs"] == ["probability"]
    assert report["cohorts"]["receiving-cdf"]["deployment_approved"]
