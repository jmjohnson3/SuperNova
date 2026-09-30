"""Explain exact prop close coverage after a slate, without hiding partial slates."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_training_groups import dedupe_locked_offer_rows

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_TARGET_MINUTES = (120, 60, 20)
_TARGET_TOLERANCE_MINUTES = 12

EXAMPLE_SQL = """
SELECT e.id, e.replay_id, e.source_created_at, e.prop_offer_id,
       e.game_date_et, e.game_slug, e.player_id, e.player_name_norm,
       e.market, e.side, e.bookmaker_key, e.market_line, e.market_price,
       e.lock_snapshot_id, e.closing_snapshot_id, e.clv_valid,
       e.clv_unknown_reason, e.pair_quality, e.true_pair_flag,
       l.event_id, l.commence_time_utc
FROM features.mlb_prop_market_training_examples e
LEFT JOIN odds.mlb_player_prop_line_snapshots l ON l.id = e.lock_snapshot_id
WHERE e.game_date_et = %(slate_date)s
ORDER BY e.source_created_at, e.id
"""

SNAPSHOT_SQL = """
SELECT event_id, MAX(commence_time_utc) AS commence_time_utc,
       MAX(home_team) AS home_team, MAX(away_team) AS away_team,
       snapshot_at_utc
FROM odds.mlb_player_prop_line_snapshots
WHERE as_of_date = %(slate_date)s
  AND event_id IS NOT NULL
  AND commence_time_utc IS NOT NULL
  AND snapshot_role = 'close'
GROUP BY event_id, snapshot_at_utc
ORDER BY event_id, snapshot_at_utc
"""


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _game_status(conn, slate_date: date) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(status, 'unknown'), COUNT(*)::int
            FROM raw.mlb_games
            WHERE game_date_et = %s
            GROUP BY COALESCE(status, 'unknown')
            """,
            (slate_date,),
        )
        statuses = {str(status): int(count) for status, count in cur.fetchall()}
    total = sum(statuses.values())
    final = statuses.get("final", 0)
    return {
        "games": total,
        "final_games": final,
        "statuses": statuses,
        "slate_final": bool(total > 0 and final == total),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _reason(row: pd.Series) -> str:
    clv_valid = row.get("clv_valid")
    if pd.notna(clv_valid) and bool(clv_valid):
        return "valid_close"
    if pd.isna(row.get("lock_snapshot_id")):
        return "missing_lock_snapshot"
    value = str(row.get("clv_unknown_reason") or "").strip()
    return value if value and value.lower() != "nan" else "no_valid_close_snapshot"


def _group_rows(df: pd.DataFrame, columns: list[str], *, top_n: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(columns, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        reasons = Counter(group["close_reason"].astype(str))
        valid = int(reasons.get("valid_close", 0))
        total = int(len(group))
        stale = int(reasons.get("stale_close_before_lock", 0))
        record = {column: str(value or "unknown") for column, value in zip(columns, values)}
        record.update({
            "rows": total,
            "valid_closes": valid,
            "valid_close_coverage": _rate(valid, total),
            "stale_close_rate": _rate(stale, total),
            "failed_rows": total - valid,
            "failure_reasons": dict(reasons.most_common()),
        })
        rows.append(record)
    rows.sort(key=lambda row: (-int(row["failed_rows"]), float(row.get("valid_close_coverage") or 0.0)))
    return rows[:top_n]


def _target_capture_rows(snapshots: pd.DataFrame) -> list[dict[str, Any]]:
    if snapshots.empty:
        return []
    work = snapshots.copy()
    work["snapshot_at_utc"] = pd.to_datetime(work["snapshot_at_utc"], utc=True, errors="coerce")
    work["commence_time_utc"] = pd.to_datetime(work["commence_time_utc"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    for event_id, group in work.groupby("event_id", dropna=False):
        commence = group["commence_time_utc"].dropna().max()
        if pd.isna(commence):
            continue
        captured = group["snapshot_at_utc"].dropna()
        label = f"{group['away_team'].dropna().iloc[0] if group['away_team'].notna().any() else '?'} @ {group['home_team'].dropna().iloc[0] if group['home_team'].notna().any() else '?'}"
        targets: dict[str, Any] = {}
        earliest_target = max(_TARGET_MINUTES)
        for minutes in _TARGET_MINUTES:
            target = commence - pd.Timedelta(minutes=minutes)
            raw_offsets = (captured - target).dt.total_seconds() / 60.0
            if minutes == earliest_target:
                eligible = raw_offsets.loc[(raw_offsets >= 0) & (raw_offsets <= _TARGET_TOLERANCE_MINUTES)]
                nearest = float(eligible.min()) if not eligible.empty else None
            else:
                offsets = raw_offsets.abs()
                nearest = float(offsets.min()) if not offsets.empty else None
            targets[f"t_minus_{minutes}"] = {
                "captured": bool(nearest is not None and nearest <= _TARGET_TOLERANCE_MINUTES),
                "nearest_minutes": nearest,
            }
        rows.append({
            "event_id": str(event_id),
            "event": label,
            "commence_time_utc": commence.isoformat(),
            "close_observations": int(captured.nunique()),
            "targets": targets,
            "all_targets_captured": all(rec["captured"] for rec in targets.values()),
        })
    return sorted(rows, key=lambda row: (not row["all_targets_captured"], row["commence_time_utc"]))


def _target_miss_summary(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_window = {f"t_minus_{minutes}": 0 for minutes in _TARGET_MINUTES}
    missed_events: list[dict[str, Any]] = []
    for row in target_rows:
        missing = [
            key for key, rec in (row.get("targets") or {}).items()
            if key in by_window and not bool(rec.get("captured"))
        ]
        for key in missing:
            by_window[key] += 1
        if missing:
            missed_events.append({
                "event_id": row.get("event_id"),
                "event": row.get("event"),
                "commence_time_utc": row.get("commence_time_utc"),
                "missing_windows": missing,
                "close_observations": row.get("close_observations"),
            })
    return {
        "events_with_misses": int(len(missed_events)),
        "by_window": by_window,
        "missed_events": missed_events,
    }


def _outside_window_diagnostic(examples: pd.DataFrame, snapshots: pd.DataFrame) -> dict[str, Any]:
    if examples.empty:
        return {"rows": 0, "reason_counts": {}, "by_book_market": []}
    outside = examples.loc[examples["close_reason"].eq("close_outside_two_hour_window")].copy()
    if outside.empty:
        return {"rows": 0, "reason_counts": {}, "by_book_market": []}
    snap = snapshots.copy()
    if not snap.empty:
        snap["snapshot_at_utc"] = pd.to_datetime(snap["snapshot_at_utc"], utc=True, errors="coerce")
        snap["commence_time_utc"] = pd.to_datetime(snap["commence_time_utc"], utc=True, errors="coerce")
    close_times: dict[str, pd.Series] = {}
    for event_id, group in snap.groupby("event_id", dropna=False):
        close_times[str(event_id)] = group["snapshot_at_utc"].dropna()

    reasons: list[str] = []
    nearest_offsets: list[float | None] = []
    for _, row in outside.iterrows():
        event_id = str(row.get("event_id") or "")
        commence = pd.to_datetime(row.get("commence_time_utc"), utc=True, errors="coerce")
        times = close_times.get(event_id, pd.Series(dtype="datetime64[ns, UTC]")).dropna()
        if pd.isna(commence) or times.empty:
            reasons.append("no_event_close_snapshots")
            nearest_offsets.append(None)
            continue
        minutes_before = (commence - times).dt.total_seconds() / 60.0
        valid_window = minutes_before.loc[(minutes_before >= 0) & (minutes_before <= 120)]
        if not valid_window.empty:
            reasons.append("event_close_window_captured_but_exact_offer_unmatched")
            nearest_offsets.append(float(valid_window.abs().min()))
        elif (minutes_before > 120).any():
            reasons.append("only_early_event_close_snapshots")
            nearest_offsets.append(float(minutes_before.loc[minutes_before > 120].min()))
        elif (minutes_before < 0).any():
            reasons.append("only_after_start_event_close_snapshots")
            nearest_offsets.append(float(minutes_before.max()))
        else:
            reasons.append("no_event_close_snapshots")
            nearest_offsets.append(None)
    outside["outside_window_diagnostic"] = reasons
    outside["nearest_valid_window_offset_minutes"] = nearest_offsets
    reason_counts = Counter(outside["outside_window_diagnostic"].astype(str))
    by_book_market: list[dict[str, Any]] = []
    for key, group in outside.groupby(["bookmaker_key", "market", "outside_window_diagnostic"], dropna=False):
        book, market, reason = key
        by_book_market.append({
            "bookmaker_key": str(book or "unknown"),
            "market": str(market or "unknown"),
            "diagnostic": str(reason),
            "rows": int(len(group)),
        })
    by_book_market.sort(key=lambda row: -int(row["rows"]))
    return {
        "rows": int(len(outside)),
        "reason_counts": dict(reason_counts.most_common()),
        "by_book_market": by_book_market[:30],
    }


def _action_plan(examples: pd.DataFrame, target_rows: list[dict[str, Any]], rows_needed_for_90: int) -> list[dict[str, Any]]:
    if examples.empty:
        return []
    reasons = Counter(examples["close_reason"].astype(str))
    target_misses = _target_miss_summary(target_rows)
    plans = [
        {
            "area": "coverage_gap_to_90",
            "rows": int(rows_needed_for_90),
            "diagnosis": "additional valid exact closes needed for this slate to clear the clean-promotion threshold",
            "next_action": "supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start",
        },
        {
            "area": "required_target_windows",
            "rows": int(target_misses["events_with_misses"]),
            "diagnosis": "events missing at least one required T-120/T-60/T-20 capture",
            "next_action": "verify Task Scheduler history and mutex skips for those event windows",
        },
    ]
    reason_actions = {
        "player_prop_unavailable_at_close": (
            "the book still had the game, but the player prop menu was gone for that player",
            "treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability"
        ),
        "player_market_unavailable_at_close": (
            "the player was present at close but this market was unavailable",
            "rank by book/market and repair feed normalization or demote markets with repeat disappearance"
        ),
        "line_disappeared_at_close": (
            "same player and market existed at close, but the exact line moved or disappeared",
            "store nearest-line movement separately; exact-bucket promotion should keep this as not bookable"
        ),
        "exact_line_unavailable_at_close": (
            "same player and market existed at close, but the exact line was not offered",
            "measure line-move direction and keep exact-line CLV unknown rather than forcing bad CLV"
        ),
        "fallback_other_book_only": (
            "another book had a close but the original book did not",
            "do not use cross-book fallback for promotion; use it only for diagnostics/display"
        ),
        "missing_close_side_price": (
            "close row existed but the needed over/under side price was missing",
            "repair parser side extraction for that book/market before using it for training"
        ),
        "close_outside_two_hour_window": (
            "close was captured too early or after first pitch",
            "scheduler timing problem; targeted captures should reduce this"
        ),
        "stale_close_before_lock": (
            "close snapshot timestamp was before the bet lock",
            "lock/close ordering problem; do not count as CLV"
        ),
        "missing_lock_snapshot": (
            "locked decision did not link to an immutable lock offer",
            "offer ID/link repair needed before the row can be trusted"
        ),
    }
    for reason, count in reasons.most_common():
        if reason == "valid_close" or reason not in reason_actions:
            continue
        diagnosis, action = reason_actions[reason]
        plans.append({
            "area": reason,
            "rows": int(count),
            "diagnosis": diagnosis,
            "next_action": action,
        })
    return plans


def build(slate_date: date, pg_dsn: str = PG_DSN) -> dict[str, Any]:
    with psycopg2.connect(pg_dsn) as conn:
        status = _game_status(conn, slate_date)
        examples = _query_df(conn, EXAMPLE_SQL, {"slate_date": slate_date})
        snapshots = _query_df(conn, SNAPSHOT_SQL, {"slate_date": slate_date})
    if not examples.empty:
        examples = dedupe_locked_offer_rows(examples)
        examples["close_reason"] = examples.apply(_reason, axis=1)
    reasons = Counter(examples.get("close_reason", pd.Series(dtype=str)).astype(str))
    rows = int(len(examples))
    valid = int(reasons.get("valid_close", 0))
    stale = int(reasons.get("stale_close_before_lock", 0))
    target_rows = _target_capture_rows(snapshots)
    valid_coverage = _rate(valid, rows)
    stale_rate = _rate(stale, rows)
    rows_needed_for_90 = max(0, math.ceil(0.90 * rows) - valid) if rows else 0
    final_ready = bool(
        status["slate_final"]
        and rows >= 100
        and valid_coverage is not None and valid_coverage >= 0.90
        and stale_rate is not None and stale_rate <= 0.02
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "slate_date": slate_date.isoformat(),
        "evaluation_status": "final" if status["slate_final"] else "provisional",
        **status,
        "locked_offer_rows": rows,
        "valid_closes": valid,
        "valid_close_coverage": valid_coverage,
        "valid_closes_needed_for_90": rows_needed_for_90,
        "stale_close_rate": stale_rate,
        "strict_clean_slate": final_ready,
        "failure_reasons": dict(reasons.most_common()),
        "target_capture": {
            "events": len(target_rows),
            "all_targets_captured_events": sum(1 for row in target_rows if row["all_targets_captured"]),
            "required_target_misses": _target_miss_summary(target_rows),
            "rows": target_rows,
        },
        "outside_window_diagnostic": _outside_window_diagnostic(examples, snapshots) if not examples.empty else {"rows": 0, "reason_counts": {}, "by_book_market": []},
        "action_plan": _action_plan(examples, target_rows, rows_needed_for_90) if not examples.empty else [],
        "by_event": _group_rows(examples, ["event_id", "game_slug"]) if not examples.empty else [],
        "by_book": _group_rows(examples, ["bookmaker_key"]) if not examples.empty else [],
        "by_market": _group_rows(examples, ["market"]) if not examples.empty else [],
        "by_event_book_market": _group_rows(examples, ["event_id", "bookmaker_key", "market"]) if not examples.empty else [],
    }
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / "prop_end_of_slate_close_diagnostic.json", payload)
    atomic_write_json(_MODEL_DIR / f"prop_end_of_slate_close_{slate_date.isoformat()}.json", payload)
    report = _render(payload)
    latest = _REPORT_DIR / "mlb_prop_end_of_slate_close_latest.md"
    dated = _REPORT_DIR / f"mlb_prop_end_of_slate_close_{slate_date.isoformat()}.md"
    atomic_write_text(latest, report)
    atomic_write_text(dated, report)
    payload["report_path"] = str(latest)
    return payload


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop End-of-Slate Close Diagnostic", "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Slate: {payload['slate_date']}",
        f"Evaluation: **{payload['evaluation_status'].upper()}**",
        f"Games final: {payload['final_games']} / {payload['games']}",
        f"Strict clean slate: **{payload['strict_clean_slate']}**", "",
        f"- Locked executable offers: {payload['locked_offer_rows']}",
        f"- Valid exact closes: {payload['valid_closes']} ({_pct(payload['valid_close_coverage'])})",
        f"- Additional valid closes needed for 90%: {payload.get('valid_closes_needed_for_90', 0)}",
        f"- Stale closes: {_pct(payload['stale_close_rate'])}",
        f"- T-120/T-60/T-20 complete events: {payload['target_capture']['all_targets_captured_events']} / {payload['target_capture']['events']}", "",
        "## Action Plan", "",
        "| Area | Rows | Diagnosis | Next Action |",
        "|---|---:|---|---|",
    ]
    for row in payload.get("action_plan") or []:
        lines.append(
            f"| {row['area']} | {row['rows']} | {row['diagnosis']} | {row['next_action']} |"
        )
    lines.extend([
        "",
        "## Failure Reasons", "",
        "| Reason | Rows |", "|---|---:|",
    ])
    for reason, count in payload["failure_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    outside = payload.get("outside_window_diagnostic") or {}
    lines.extend(["", "## Outside Two-Hour Window Diagnostic", "", "| Diagnostic | Rows |", "|---|---:|"])
    for reason, count in (outside.get("reason_counts") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "| Book | Market | Diagnostic | Rows |", "|---|---|---|---:|"])
    for row in (outside.get("by_book_market") or [])[:20]:
        lines.append(
            f"| {row['bookmaker_key']} | {row['market']} | {row['diagnostic']} | {row['rows']} |"
        )
    lines.extend(["", "## Most Affected Event / Book / Market", "", "| Event ID | Book | Market | Rows | Failed | Valid | Reasons |", "|---|---|---|---:|---:|---:|---|"])
    for row in payload["by_event_book_market"][:40]:
        lines.append(
            f"| {row['event_id']} | {row['bookmaker_key']} | {row['market']} | {row['rows']} | "
            f"{row['failed_rows']} | {_pct(row['valid_close_coverage'])} | "
            f"{json.dumps(row['failure_reasons'], sort_keys=True)} |"
        )
    lines.extend(["", "## Targeted Capture", "", "| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |", "|---|---|---|---|---|---:|"])
    for row in payload["target_capture"]["rows"]:
        targets = row["targets"]
        mark = lambda rec: "yes" if rec["captured"] else "no"
        lines.append(
            f"| {row['event']} | {row['commence_time_utc']} | {mark(targets['t_minus_120'])} | "
            f"{mark(targets['t_minus_60'])} | {mark(targets['t_minus_20'])} | {row['close_observations']} |"
        )
    misses = (payload.get("target_capture") or {}).get("required_target_misses") or {}
    lines.extend(["", "## Required Target Misses", "", "| Window | Missed Events |", "|---|---:|"])
    for window, count in (misses.get("by_window") or {}).items():
        lines.append(f"| {window} | {count} |")
    lines.extend(["", "| Event | Start UTC | Missing Windows | Close Times |", "|---|---|---|---:|"])
    for row in (misses.get("missed_events") or [])[:30]:
        lines.append(
            f"| {row.get('event')} | {row.get('commence_time_utc')} | "
            f"{', '.join(row.get('missing_windows') or [])} | {row.get('close_observations')} |"
        )
    if payload["evaluation_status"] != "final":
        lines.extend(["", "> This slate is still provisional. Coverage must be evaluated again after every game is final."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain MLB prop close coverage for one slate")
    parser.add_argument("--date", help="ET slate date, YYYY-MM-DD. Defaults to today ET.")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    slate_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    payload = build(slate_date, args.pg_dsn)
    print(json.dumps({
        "slate_date": payload["slate_date"],
        "evaluation_status": payload["evaluation_status"],
        "strict_clean_slate": payload["strict_clean_slate"],
        "valid_close_coverage": payload["valid_close_coverage"],
        "report_path": payload["report_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
