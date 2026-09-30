"""Capture targeted prop close snapshots with low-offer retry protection."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN
from mlb_pipeline.subprocess_utils import run_subprocess_tree

from .prop_close_capture_schedule import DueCloseTarget, load_due_close_targets

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _target_payload(targets: list[DueCloseTarget]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        row = asdict(target)
        row["commence_time_utc"] = target.commence_time_utc.isoformat()
        row["target_time_utc"] = target.target_time_utc.isoformat()
        rows.append(row)
    return rows


def _run_module(
    module: str,
    args: tuple[str, ...],
    *,
    env: dict[str, str],
    timeout_s: int,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    rc, stdout, stderr, secs = run_subprocess_tree(
        [sys.executable, "-m", module, *args],
        timeout_s=timeout_s,
        env=env,
    )
    return {
        "module": module,
        "args": list(args),
        "started_at_utc": started,
        "rc": int(rc),
        "ok": rc == 0,
        "seconds": secs,
        "stdout_tail": "\n".join((stdout or "").splitlines()[-30:]),
        "stderr_tail": "\n".join((stderr or "").splitlines()[-30:]),
    }


def _load_active_targets(
    *,
    slate_date: date,
    pg_dsn: str,
    active_window_minutes: int,
) -> list[DueCloseTarget]:
    now = datetime.now(timezone.utc)
    with psycopg2.connect(pg_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT event_id, MAX(commence_time_utc) AS commence_time_utc
                FROM odds.mlb_player_prop_line_snapshots
                WHERE as_of_date = %s
                  AND event_id IS NOT NULL
                  AND commence_time_utc IS NOT NULL
                  AND commence_time_utc > %s
                  AND commence_time_utc <= %s
                GROUP BY event_id
                ORDER BY MAX(commence_time_utc), event_id
                """,
                (
                    slate_date,
                    now,
                    now + timedelta(minutes=max(1, int(active_window_minutes))),
                ),
            )
            rows = cur.fetchall()
    targets: list[DueCloseTarget] = []
    for row in rows:
        commence = row["commence_time_utc"]
        if commence.tzinfo is None:
            commence = commence.replace(tzinfo=timezone.utc)
        commence = commence.astimezone(timezone.utc)
        minutes_to_start = max(1, int(round((commence - now).total_seconds() / 60.0)))
        targets.append(
            DueCloseTarget(
                event_id=str(row["event_id"]),
                commence_time_utc=commence,
                target_minutes_before_start=minutes_to_start,
                target_time_utc=now,
                minutes_from_target=0.0,
                capture_reason="forced_active_close_window",
            )
        )
    return targets


def _fresh_close_quality(
    *,
    pg_dsn: str,
    slate_date: date,
    event_ids: list[str],
    since_utc: datetime,
    min_rows_without_baseline: int,
    min_rows_floor: int,
    min_ratio: float,
) -> dict[str, Any]:
    if not event_ids:
        return {"passed": True, "events": [], "low_events": []}
    since = since_utc - timedelta(minutes=2)
    with psycopg2.connect(pg_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                WITH latest AS (
                    SELECT event_id,
                           MAX(snapshot_at_utc) AS latest_snapshot_at_utc,
                           COUNT(DISTINCT snapshot_at_utc)::int AS fresh_close_times
                    FROM odds.mlb_player_prop_line_snapshots
                    WHERE as_of_date = %s
                      AND snapshot_role = 'close'
                      AND event_id = ANY(%s)
                      AND snapshot_at_utc >= %s
                    GROUP BY event_id
                )
                SELECT l.event_id,
                       COUNT(DISTINCT concat_ws('|',
                           s.bookmaker_key,
                           s.player_name_norm,
                           s.stat,
                           s.line::text
                       ))::int AS fresh_close_rows,
                       MAX(l.fresh_close_times)::int AS fresh_close_times,
                       MAX(l.latest_snapshot_at_utc) AS latest_snapshot_at_utc
                FROM latest l
                JOIN odds.mlb_player_prop_line_snapshots s
                  ON s.event_id = l.event_id
                 AND s.snapshot_at_utc = l.latest_snapshot_at_utc
                 AND s.as_of_date = %s
                 AND s.snapshot_role = 'close'
                GROUP BY l.event_id
                """,
                (slate_date, event_ids, since, slate_date),
            )
            fresh = {str(row["event_id"]): dict(row) for row in cur.fetchall()}
            cur.execute(
                """
                WITH latest AS (
                    SELECT event_id,
                           snapshot_role,
                           MAX(snapshot_at_utc) AS latest_snapshot_at_utc
                    FROM odds.mlb_player_prop_line_snapshots
                    WHERE as_of_date = %s
                      AND event_id = ANY(%s)
                      AND snapshot_role IN ('lock', 'open', 'close')
                      AND snapshot_at_utc < %s
                    GROUP BY event_id, snapshot_role
                ),
                counts AS (
                    SELECT s.event_id,
                           s.snapshot_role,
                           COUNT(DISTINCT concat_ws('|',
                               s.bookmaker_key,
                               s.player_name_norm,
                               s.stat,
                               s.line::text
                           ))::int AS offer_rows
                    FROM odds.mlb_player_prop_line_snapshots s
                    JOIN latest l
                      ON l.event_id = s.event_id
                     AND l.snapshot_role = s.snapshot_role
                     AND l.latest_snapshot_at_utc = s.snapshot_at_utc
                    WHERE s.as_of_date = %s
                    GROUP BY s.event_id, s.snapshot_role
                )
                SELECT event_id,
                       COALESCE(MAX(offer_rows) FILTER (WHERE snapshot_role = 'lock'), 0)::int AS lock_rows,
                       COALESCE(MAX(offer_rows) FILTER (WHERE snapshot_role = 'open'), 0)::int AS open_rows,
                       COALESCE(MAX(offer_rows) FILTER (WHERE snapshot_role = 'close'), 0)::int AS prior_close_rows
                FROM counts
                GROUP BY event_id
                """,
                (slate_date, event_ids, since_utc, slate_date),
            )
            baseline = {str(row["event_id"]): dict(row) for row in cur.fetchall()}

    events: list[dict[str, Any]] = []
    low_events: list[dict[str, Any]] = []
    for event_id in event_ids:
        fresh_row = fresh.get(event_id, {})
        baseline_row = baseline.get(event_id, {})
        baseline_rows = max(
            int(baseline_row.get("lock_rows") or 0),
            int(baseline_row.get("open_rows") or 0),
            int(baseline_row.get("prior_close_rows") or 0),
        )
        if baseline_rows > 0:
            required = max(int(min_rows_floor), int(round(baseline_rows * float(min_ratio))))
        else:
            required = max(int(min_rows_floor), int(min_rows_without_baseline))
        fresh_rows = int(fresh_row.get("fresh_close_rows") or 0)
        record = {
            "event_id": event_id,
            "fresh_close_rows": fresh_rows,
            "fresh_close_times": int(fresh_row.get("fresh_close_times") or 0),
            "latest_snapshot_at_utc": fresh_row.get("latest_snapshot_at_utc"),
            "baseline_rows": baseline_rows,
            "required_rows": required,
            "passed": fresh_rows >= required,
        }
        events.append(record)
        if not record["passed"]:
            low_events.append(record)
    return {
        "passed": not low_events,
        "events": events,
        "low_events": low_events,
        "thresholds": {
            "min_rows_without_baseline": min_rows_without_baseline,
            "min_rows_floor": min_rows_floor,
            "min_ratio": min_ratio,
        },
    }


def _parse_focus_specs(raw_specs: str | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for raw in str(raw_specs or "").split(","):
        text = raw.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            continue
        book, stat, line = parts
        try:
            line_value = float(line)
        except ValueError:
            continue
        specs.append({
            "bookmaker_key": book.lower(),
            "stat": stat,
            "line": line_value,
            "label": f"{book.lower()}:{stat}:{line_value:g}",
        })
    return specs


def _row_matches_focus(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    try:
        line = float(row.get("line"))
    except (TypeError, ValueError):
        return False
    return (
        str(row.get("bookmaker_key") or "").lower() == spec["bookmaker_key"]
        and str(row.get("stat") or "") == spec["stat"]
        and abs(line - float(spec["line"])) <= 1e-9
    )


def _latest_focus_count(rows: list[dict[str, Any]], *, role: str | None = None) -> int:
    work = [
        row for row in rows
        if role is None or str(row.get("snapshot_role") or "") == role
    ]
    if not work:
        return 0
    latest = max(
        row.get("snapshot_at_utc")
        for row in work
        if row.get("snapshot_at_utc") is not None
    )
    if latest is None:
        return 0
    keys = {
        (
            str(row.get("bookmaker_key") or "").lower(),
            str(row.get("player_name_norm") or ""),
            str(row.get("stat") or ""),
            str(row.get("line") or ""),
        )
        for row in work
        if row.get("snapshot_at_utc") == latest
        and (row.get("over_price") is not None or row.get("under_price") is not None)
    }
    return len(keys)


def _fresh_focus_bucket_quality(
    *,
    pg_dsn: str,
    slate_date: date,
    event_ids: list[str],
    since_utc: datetime,
    focus_specs: list[dict[str, Any]],
    min_rows_floor: int,
    min_ratio: float,
) -> dict[str, Any]:
    if not event_ids or not focus_specs:
        return {"passed": True, "events": [], "low_events": [], "focus_specs": focus_specs}
    since = since_utc - timedelta(minutes=2)
    with psycopg2.connect(pg_dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT event_id, snapshot_role, snapshot_at_utc, bookmaker_key,
                       player_name_norm, stat, line::float AS line,
                       over_price, under_price
                FROM odds.mlb_player_prop_line_snapshots
                WHERE as_of_date = %s
                  AND event_id = ANY(%s)
                  AND snapshot_role IN ('lock', 'open', 'close')
                  AND snapshot_at_utc IS NOT NULL
                  AND (
                    (snapshot_role = 'close' AND snapshot_at_utc >= %s)
                    OR snapshot_at_utc < %s
                  )
                """,
                (slate_date, event_ids, since, since_utc),
            )
            all_rows = [dict(row) for row in cur.fetchall()]

    events: list[dict[str, Any]] = []
    low_events: list[dict[str, Any]] = []
    for event_id in event_ids:
        event_passed = True
        focus_rows: list[dict[str, Any]] = []
        for spec in focus_specs:
            matching = [
                row for row in all_rows
                if str(row.get("event_id")) == str(event_id)
                and _row_matches_focus(row, spec)
            ]
            fresh = [
                row for row in matching
                if str(row.get("snapshot_role")) == "close"
                and row.get("snapshot_at_utc") is not None
                and row.get("snapshot_at_utc") >= since
            ]
            prior = [
                row for row in matching
                if row.get("snapshot_at_utc") is not None
                and row.get("snapshot_at_utc") < since_utc
            ]
            fresh_rows = _latest_focus_count(fresh, role="close")
            baseline_rows = max(
                _latest_focus_count(prior, role="lock"),
                _latest_focus_count(prior, role="open"),
                _latest_focus_count(prior, role="close"),
            )
            required = 0
            passed = True
            if baseline_rows > 0:
                required = max(int(min_rows_floor), int(round(baseline_rows * float(min_ratio))))
                passed = fresh_rows >= required
            record = {
                "event_id": event_id,
                "focus": spec["label"],
                "fresh_rows": int(fresh_rows),
                "baseline_rows": int(baseline_rows),
                "required_rows": int(required),
                "enforced": bool(baseline_rows > 0),
                "passed": bool(passed),
            }
            focus_rows.append(record)
            if not passed:
                event_passed = False
        event_record = {
            "event_id": event_id,
            "passed": event_passed,
            "focus_rows": focus_rows,
        }
        events.append(event_record)
        if not event_passed:
            low_events.append(event_record)
    return {
        "passed": not low_events,
        "events": events,
        "low_events": low_events,
        "focus_specs": focus_specs,
        "thresholds": {
            "min_rows_floor": min_rows_floor,
            "min_ratio": min_ratio,
        },
    }


def _write_report(payload: dict[str, Any]) -> Path:
    path = _REPORT_DIR / "mlb_targeted_prop_close_capture_latest.md"
    lines = [
        "# MLB Targeted Prop Close Capture",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status', 'unknown').upper()}**",
        f"Slate: {payload.get('slate_date')}",
        f"Attempts: {payload.get('attempts_run')} / {payload.get('max_attempts')}",
        "",
        "## Targets",
        "",
        "| Event | Minutes To Start | Reason |",
        "|---|---:|---|",
    ]
    for target in payload.get("targets", []):
        lines.append(
            f"| {target.get('event_id')} | {target.get('target_minutes_before_start')} | "
            f"{target.get('capture_reason')} |"
        )
    lines.extend([
        "",
        "## Attempt Quality",
        "",
        "| Attempt | Passed | Low Events |",
        "|---:|---|---:|",
    ])
    for attempt in payload.get("attempts", []):
        quality = attempt.get("quality") or {}
        lines.append(
            f"| {attempt.get('attempt')} | {quality.get('passed')} | "
            f"{len(quality.get('low_events') or [])} |"
        )
    lines.extend([
        "",
        "## Latest Event Counts",
        "",
        "| Event | Fresh Rows | Required | Baseline | Passed |",
        "|---|---:|---:|---:|---|",
    ])
    latest_quality = (payload.get("attempts") or [{}])[-1].get("quality") or {}
    for event in latest_quality.get("events") or []:
        lines.append(
            f"| {event.get('event_id')} | {event.get('fresh_close_rows')} | "
            f"{event.get('required_rows')} | {event.get('baseline_rows')} | {event.get('passed')} |"
        )
    lines.extend([
        "",
        "## Focus Bucket Counts",
        "",
        "| Event | Focus | Fresh Rows | Required | Baseline | Enforced | Passed |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    latest_focus = (payload.get("attempts") or [{}])[-1].get("focus_quality") or {}
    for event in latest_focus.get("events") or []:
        for focus in event.get("focus_rows") or []:
            lines.append(
                f"| {event.get('event_id')} | {focus.get('focus')} | "
                f"{focus.get('fresh_rows')} | {focus.get('required_rows')} | "
                f"{focus.get('baseline_rows')} | {focus.get('enforced')} | {focus.get('passed')} |"
            )
    if payload.get("error"):
        lines.extend(["", "## Error", "", "```", str(payload["error"]), "```"])
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    slate_date = args.date or (
        date.fromisoformat(os.environ["MLB_ET_DATE"])
        if os.getenv("MLB_ET_DATE")
        else datetime.now(tz=_ET).date()
    )
    force = args.force or os.getenv("MLB_FORCE_TARGETED_CLOSE_CAPTURE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    targets = load_due_close_targets(
        slate_date=slate_date,
        pg_dsn=args.pg_dsn,
        active_window_minutes=args.active_window_minutes,
        capture_interval_minutes=args.capture_interval_minutes,
    )
    if force and not targets:
        targets = _load_active_targets(
            slate_date=slate_date,
            pg_dsn=args.pg_dsn,
            active_window_minutes=args.active_window_minutes,
        )
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc),
        "slate_date": slate_date.isoformat(),
        "status": "no_capture_due",
        "force": force,
        "targets": _target_payload(targets),
        "attempts": [],
        "attempts_run": 0,
        "max_attempts": args.max_attempts,
        "focus_specs": _parse_focus_specs(args.focus_buckets),
    }
    if not targets:
        return payload

    event_ids = sorted({target.event_id for target in targets})
    env = os.environ.copy()
    last_quality: dict[str, Any] | None = None
    for attempt_no in range(1, max(1, int(args.max_attempts)) + 1):
        attempt_started = datetime.now(timezone.utc)
        crawl = _run_module(
            "mlb_pipeline.crawler_oddsapi",
            ("--skip-live", "--force-props"),
            env=env,
            timeout_s=args.crawl_timeout_seconds,
        )
        parse = {"ok": False, "rc": None}
        quality: dict[str, Any] = {"passed": False, "events": [], "low_events": event_ids}
        if crawl["ok"]:
            parse = _run_module(
                "mlb_pipeline.parse_oddsapi",
                (
                    "--prop-snapshot-role",
                    "close",
                    "--prop-as-of-date",
                    slate_date.isoformat(),
                    "--prop-snapshot-max-age-minutes",
                    str(args.parse_max_age_minutes),
                ),
                env=env,
                timeout_s=args.parse_timeout_seconds,
            )
        if crawl["ok"] and parse["ok"]:
            quality = _fresh_close_quality(
                pg_dsn=args.pg_dsn,
                slate_date=slate_date,
                event_ids=event_ids,
                since_utc=attempt_started,
                min_rows_without_baseline=args.min_rows_without_baseline,
                min_rows_floor=args.min_rows_floor,
                min_ratio=args.min_ratio,
            )
            focus_quality = _fresh_focus_bucket_quality(
                pg_dsn=args.pg_dsn,
                slate_date=slate_date,
                event_ids=event_ids,
                since_utc=attempt_started,
                focus_specs=payload["focus_specs"],
                min_rows_floor=args.focus_min_rows_floor,
                min_ratio=args.focus_min_ratio,
            )
        else:
            focus_quality = {
                "passed": False,
                "events": [],
                "low_events": event_ids,
                "focus_specs": payload["focus_specs"],
            }
        payload["attempts"].append({
            "attempt": attempt_no,
            "started_at_utc": attempt_started,
            "crawl": crawl,
            "parse": parse,
            "quality": quality,
            "focus_quality": focus_quality,
        })
        payload["attempts_run"] = attempt_no
        last_quality = quality
        if quality.get("passed") and focus_quality.get("passed"):
            payload["status"] = "captured"
            return payload
        if attempt_no < args.max_attempts:
            time.sleep(max(0.0, float(args.retry_sleep_seconds)))
    payload["status"] = (
        "failed_focus_low_offer_count"
        if (last_quality or {}).get("passed")
        else "failed_low_offer_count"
    )
    payload["error"] = {
        "message": "Targeted close capture did not meet fresh offer-count quality thresholds.",
        "last_quality": last_quality,
        "last_focus_quality": (payload.get("attempts") or [{}])[-1].get("focus_quality"),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture targeted MLB prop closes with quality retry.")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--active-window-minutes", type=int, default=_env_int("MLB_TARGETED_CLOSE_ACTIVE_WINDOW_MINUTES", 120))
    parser.add_argument("--capture-interval-minutes", type=int, default=_env_int("MLB_TARGETED_CLOSE_CAPTURE_INTERVAL_MINUTES", 8))
    parser.add_argument("--max-attempts", type=int, default=_env_int("MLB_TARGETED_CLOSE_MAX_ATTEMPTS", 3))
    parser.add_argument("--retry-sleep-seconds", type=float, default=_env_float("MLB_TARGETED_CLOSE_RETRY_SLEEP_SECONDS", 20.0))
    parser.add_argument("--crawl-timeout-seconds", type=int, default=_env_int("MLB_TARGETED_CLOSE_CRAWL_TIMEOUT_SECONDS", 600))
    parser.add_argument("--parse-timeout-seconds", type=int, default=_env_int("MLB_TARGETED_CLOSE_PARSE_TIMEOUT_SECONDS", 300))
    parser.add_argument("--parse-max-age-minutes", type=float, default=_env_float("MLB_TARGETED_CLOSE_PARSE_MAX_AGE_MINUTES", 20.0))
    parser.add_argument("--min-ratio", type=float, default=_env_float("MLB_TARGETED_CLOSE_MIN_OFFER_RATIO", 0.85))
    parser.add_argument("--min-rows-floor", type=int, default=_env_int("MLB_TARGETED_CLOSE_MIN_ROWS_FLOOR", 25))
    parser.add_argument("--min-rows-without-baseline", type=int, default=_env_int("MLB_TARGETED_CLOSE_MIN_ROWS_WITHOUT_BASELINE", 50))
    parser.add_argument(
        "--focus-buckets",
        default=os.getenv("MLB_TARGETED_CLOSE_FOCUS_BUCKETS", "draftkings:batter_total_bases:1.5"),
        help="Comma-separated book:stat:line specs that must have fresh close rows when baseline exists.",
    )
    parser.add_argument("--focus-min-ratio", type=float, default=_env_float("MLB_TARGETED_CLOSE_FOCUS_MIN_RATIO", 0.90))
    parser.add_argument("--focus-min-rows-floor", type=int, default=_env_int("MLB_TARGETED_CLOSE_FOCUS_MIN_ROWS_FLOOR", 10))
    args = parser.parse_args()

    payload = run_capture(args)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        _MODEL_DIR / "targeted_prop_close_capture.json",
        payload,
        default=_json_default,
    )
    report_path = _write_report(payload)
    payload["report_path"] = str(report_path)
    print(json.dumps(payload, indent=2, default=_json_default))
    if payload.get("status") in {"failed_low_offer_count", "failed_focus_low_offer_count"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
