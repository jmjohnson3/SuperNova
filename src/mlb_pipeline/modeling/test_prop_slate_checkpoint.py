from __future__ import annotations

from datetime import date

import pandas as pd

from .prop_end_of_slate_close_diagnostic import _reason, _target_capture_rows
from .prop_five_date_checkpoint import _release_summary


def test_close_reason_does_not_treat_missing_boolean_as_valid() -> None:
    row = pd.Series({"clv_valid": float("nan"), "lock_snapshot_id": 10, "clv_unknown_reason": None})
    assert _reason(row) == "no_valid_close_snapshot"


def test_target_capture_requires_each_requested_window() -> None:
    commence = pd.Timestamp("2026-07-06T20:00:00Z")
    snapshots = pd.DataFrame([
        {"event_id": "e1", "commence_time_utc": commence, "home_team": "H", "away_team": "A", "snapshot_at_utc": commence - pd.Timedelta(minutes=120)},
        {"event_id": "e1", "commence_time_utc": commence, "home_team": "H", "away_team": "A", "snapshot_at_utc": commence - pd.Timedelta(minutes=60)},
    ])
    row = _target_capture_rows(snapshots)[0]
    assert row["targets"]["t_minus_120"]["captured"]
    assert row["targets"]["t_minus_60"]["captured"]
    assert not row["targets"]["t_minus_20"]["captured"]
    assert not row["all_targets_captured"]


def test_target_capture_does_not_count_early_t120_snapshot() -> None:
    commence = pd.Timestamp("2026-07-06T20:00:00Z")
    snapshots = pd.DataFrame([
        {"event_id": "e1", "commence_time_utc": commence, "home_team": "H", "away_team": "A", "snapshot_at_utc": commence - pd.Timedelta(minutes=128)},
        {"event_id": "e1", "commence_time_utc": commence, "home_team": "H", "away_team": "A", "snapshot_at_utc": commence - pd.Timedelta(minutes=60)},
        {"event_id": "e1", "commence_time_utc": commence, "home_team": "H", "away_team": "A", "snapshot_at_utc": commence - pd.Timedelta(minutes=20)},
    ])
    row = _target_capture_rows(snapshots)[0]
    assert not row["targets"]["t_minus_120"]["captured"]
    assert row["targets"]["t_minus_60"]["captured"]
    assert row["targets"]["t_minus_20"]["captured"]
    assert not row["all_targets_captured"]


def test_release_checkpoint_excludes_partial_date_and_accepts_voids() -> None:
    release = "release-1"
    rows = [
        {"game_date_et": date(2026, 7, 4), "model_version": release, "stat": "anchor", "result_status": "graded", "checkpoint_status": "graded", "projection_value": 1.0, "baseline_value": 1.5, "actual_value": 1.0},
        {"game_date_et": date(2026, 7, 4), "model_version": release, "stat": "anchor", "result_status": "pending", "checkpoint_status": "void_nonparticipant", "projection_value": 1.0, "baseline_value": 1.5, "actual_value": None},
        {"game_date_et": date(2026, 7, 5), "model_version": release, "stat": "anchor", "result_status": "graded", "checkpoint_status": "graded", "projection_value": 1.0, "baseline_value": 1.5, "actual_value": 1.0},
        {"game_date_et": date(2026, 7, 5), "model_version": release, "stat": "anchor", "result_status": "pending", "checkpoint_status": "pending", "projection_value": 1.0, "baseline_value": 1.5, "actual_value": None},
    ]
    summary = _release_summary(
        pd.DataFrame(rows),
        release_id=release,
        anchor_stat="anchor",
        required_stats=("anchor",),
        game_completion={date(2026, 7, 4): True, date(2026, 7, 5): True},
    )
    assert summary["completed_dates"] == ["2026-07-04"]
    assert summary["completed_date_count"] == 1
    assert summary["void_anchor_rows"] == 1
    assert summary["pending_anchor_rows"] == 1


def test_release_checkpoint_accepts_terminal_non_played_voids() -> None:
    release = "release-1"
    rows = [
        {"game_date_et": date(2026, 7, 10), "model_version": release, "stat": "anchor", "result_status": "graded", "checkpoint_status": "graded", "projection_value": 1.0, "baseline_value": 1.5, "actual_value": 1.0},
        {"game_date_et": date(2026, 7, 10), "model_version": release, "stat": "anchor", "result_status": "void", "checkpoint_status": "void_game_not_played", "projection_value": 1.0, "baseline_value": 1.5, "actual_value": None},
    ]
    summary = _release_summary(
        pd.DataFrame(rows),
        release_id=release,
        anchor_stat="anchor",
        required_stats=("anchor",),
        game_completion={date(2026, 7, 10): True},
    )
    assert summary["completed_dates"] == ["2026-07-10"]
    assert summary["completed_date_count"] == 1
    assert summary["void_anchor_rows"] == 1
    assert summary["pending_anchor_rows"] == 0
