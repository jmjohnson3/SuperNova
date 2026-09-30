from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .prop_close_capture_schedule import find_due_close_targets


def test_due_target_is_returned_only_near_requested_window() -> None:
    now = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": now + timedelta(minutes=120)}]
    due = find_due_close_targets(events, [], now_utc=now)
    assert [(row.event_id, row.target_minutes_before_start) for row in due] == [("game-1", 120)]


def test_existing_targeted_snapshot_suppresses_duplicate_capture() -> None:
    now = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": now + timedelta(minutes=60)}]
    closes = [{"event_id": "game-1", "snapshot_at_utc": now - timedelta(minutes=4)}]
    assert find_due_close_targets(events, closes, now_utc=now) == []


def test_early_t120_snapshot_does_not_suppress_first_valid_close() -> None:
    target = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": target + timedelta(minutes=120)}]
    closes = [{"event_id": "game-1", "snapshot_at_utc": target - timedelta(minutes=8)}]
    due = find_due_close_targets(events, closes, now_utc=target)
    assert [(row.event_id, row.target_minutes_before_start) for row in due] == [("game-1", 120)]


def test_t120_is_not_due_before_valid_close_window_opens() -> None:
    target = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": target + timedelta(minutes=120)}]
    assert find_due_close_targets(events, [], now_utc=target - timedelta(minutes=8)) == []


def test_other_event_snapshot_does_not_suppress_due_capture() -> None:
    now = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": now + timedelta(minutes=20)}]
    closes = [{"event_id": "game-2", "snapshot_at_utc": now}]
    due = find_due_close_targets(events, closes, now_utc=now)
    assert len(due) == 1
    assert due[0].target_minutes_before_start == 20


def test_supplemental_t10_window_is_due() -> None:
    now = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": now + timedelta(minutes=10)}]
    due = find_due_close_targets(events, [], now_utc=now)
    assert [(row.event_id, row.target_minutes_before_start) for row in due] == [("game-1", 10)]


def test_active_close_window_captures_between_named_targets() -> None:
    now = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": now + timedelta(minutes=105)}]
    due = find_due_close_targets(events, [], now_utc=now)
    assert [(row.event_id, row.target_minutes_before_start) for row in due] == [("game-1", 105)]
    assert due[0].capture_reason == "active_close_window"


def test_recent_active_window_capture_suppresses_only_current_tick() -> None:
    now = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    events = [{"event_id": "game-1", "commence_time_utc": now + timedelta(minutes=86)}]
    closes = [{"event_id": "game-1", "snapshot_at_utc": now - timedelta(minutes=4)}]
    assert find_due_close_targets(events, closes, now_utc=now) == []
    due_after_interval = find_due_close_targets(
        events,
        [{"event_id": "game-1", "snapshot_at_utc": now - timedelta(minutes=10)}],
        now_utc=now,
    )
    assert [(row.event_id, row.target_minutes_before_start) for row in due_after_interval] == [("game-1", 90)]
