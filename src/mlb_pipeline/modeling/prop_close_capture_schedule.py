"""Decide whether a targeted prop close snapshot is due for an upcoming game."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import psycopg2

from mlb_pipeline.db import PG_DSN

REQUIRED_TARGET_MINUTES = (120, 60, 20)
DEFAULT_TARGET_MINUTES = (120, 90, 60, 45, 30, 20, 10)
DEFAULT_ACTIVE_WINDOW_MINUTES = 120
DEFAULT_CAPTURE_INTERVAL_MINUTES = 8


@dataclass(frozen=True)
class DueCloseTarget:
    event_id: str
    commence_time_utc: datetime
    target_minutes_before_start: int
    target_time_utc: datetime
    minutes_from_target: float
    capture_reason: str = "active_close_window"


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def find_due_close_targets(
    events: Iterable[dict[str, Any]],
    close_observations: Iterable[dict[str, Any]],
    *,
    now_utc: datetime,
    target_minutes: tuple[int, ...] = DEFAULT_TARGET_MINUTES,
    tolerance_minutes: int = 12,
    active_window_minutes: int = DEFAULT_ACTIVE_WINDOW_MINUTES,
    capture_interval_minutes: int = DEFAULT_CAPTURE_INTERVAL_MINUTES,
) -> list[DueCloseTarget]:
    """Return active-game close captures due on this scheduler tick.

    A valid CLV close can be any same-offer snapshot captured after lock and
    within two hours of first pitch.  The scheduler therefore captures every
    tick for games inside that two-hour window, while still labeling the nearest
    classic target minute for diagnostics.
    """
    now = _utc(now_utc)
    if now is None:
        return []
    tolerance = timedelta(minutes=max(1, int(tolerance_minutes)))
    recent_cutoff = now - timedelta(minutes=max(1, int(capture_interval_minutes)))
    future_slop = now + timedelta(minutes=2)
    observed: dict[str, list[datetime]] = {}
    for row in close_observations:
        event_id = str(row.get("event_id") or "").strip()
        captured = _utc(row.get("snapshot_at_utc"))
        if event_id and captured is not None:
            observed.setdefault(event_id, []).append(captured)

    due: list[DueCloseTarget] = []
    active_window = max(1, int(active_window_minutes))
    for row in events:
        event_id = str(row.get("event_id") or "").strip()
        commence = _utc(row.get("commence_time_utc"))
        if not event_id or commence is None or commence <= now:
            continue
        minutes_to_start = (commence - now).total_seconds() / 60.0
        if not (0.0 < minutes_to_start <= float(active_window)):
            continue
        recently_captured = any(
            recent_cutoff < captured <= future_slop
            for captured in observed.get(event_id, [])
        )
        if recently_captured:
            continue
        nearest_target = min(
            target_minutes or (int(round(minutes_to_start)),),
            key=lambda target: abs(float(target) - minutes_to_start),
        )
        if abs(float(nearest_target) - minutes_to_start) <= float(tolerance_minutes):
            label_minutes = int(nearest_target)
            target_at = commence - timedelta(minutes=label_minutes)
            minutes_from_target = (now - target_at).total_seconds() / 60.0
        else:
            label_minutes = int(round(minutes_to_start))
            target_at = now
            minutes_from_target = 0.0
        due.append(DueCloseTarget(
            event_id=event_id,
            commence_time_utc=commence,
            target_minutes_before_start=label_minutes,
            target_time_utc=target_at,
            minutes_from_target=minutes_from_target,
        ))
    return sorted(due, key=lambda row: (row.target_time_utc, row.event_id))


def load_due_close_targets(
    *,
    slate_date: date,
    now_utc: datetime | None = None,
    pg_dsn: str = PG_DSN,
    target_minutes: tuple[int, ...] = DEFAULT_TARGET_MINUTES,
    tolerance_minutes: int = 12,
    active_window_minutes: int = DEFAULT_ACTIVE_WINDOW_MINUTES,
    capture_interval_minutes: int = DEFAULT_CAPTURE_INTERVAL_MINUTES,
) -> list[DueCloseTarget]:
    now = _utc(now_utc or datetime.now(timezone.utc))
    with psycopg2.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, MAX(commence_time_utc) AS commence_time_utc
                FROM odds.mlb_player_prop_line_snapshots
                WHERE as_of_date = %s
                  AND event_id IS NOT NULL
                  AND commence_time_utc IS NOT NULL
                GROUP BY event_id
                """,
                (slate_date,),
            )
            events = [
                {"event_id": event_id, "commence_time_utc": commence}
                for event_id, commence in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT DISTINCT event_id, snapshot_at_utc
                FROM odds.mlb_player_prop_line_snapshots
                WHERE as_of_date = %s
                  AND snapshot_role = 'close'
                  AND event_id IS NOT NULL
                  AND snapshot_at_utc IS NOT NULL
                """,
                (slate_date,),
            )
            observations = [
                {"event_id": event_id, "snapshot_at_utc": captured}
                for event_id, captured in cur.fetchall()
            ]
    return find_due_close_targets(
        events,
        observations,
        now_utc=now,
        target_minutes=target_minutes,
        tolerance_minutes=tolerance_minutes,
        active_window_minutes=active_window_minutes,
        capture_interval_minutes=capture_interval_minutes,
    )


def decision_payload(targets: Iterable[DueCloseTarget]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        row = asdict(target)
        row["commence_time_utc"] = target.commence_time_utc.isoformat()
        row["target_time_utc"] = target.target_time_utc.isoformat()
        rows.append(row)
    return rows
