"""Daily trust monitor for MLB prop prospective slates.

This report does not promote bets. It answers whether a slate is operationally
usable for the frozen-release evaluation and, later, exact-bucket micro review.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_end_of_slate_close_diagnostic import build as build_close_diagnostic
from .prop_five_date_checkpoint import build as build_five_date_checkpoint

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MIN_VALID_CLOSE_COVERAGE = 0.90
_MAX_STALE_CLOSE_RATE = 0.02


def _rate_ok(value: Any, threshold: float, *, lower_is_better: bool = False) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric <= threshold if lower_is_better else numeric >= threshold


def _date_row(release: dict[str, Any], slate_date: str) -> dict[str, Any] | None:
    for row in release.get("by_date") or []:
        if str(row.get("game_date_et")) == slate_date:
            return row
    return None


def _check(name: str, passed: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "required": bool(required),
        "detail": detail,
    }


def evaluate_trust(
    close_payload: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    *,
    slate_date: date,
) -> dict[str, Any]:
    """Build a deterministic daily trust decision from existing reports."""
    slate = slate_date.isoformat()
    final_games = int(close_payload.get("final_games") or 0)
    games = int(close_payload.get("games") or 0)
    games_final = bool(close_payload.get("slate_final"))
    target_capture = close_payload.get("target_capture") or {}
    target_events = int(target_capture.get("events") or 0)
    captured_events = int(target_capture.get("all_targets_captured_events") or 0)
    hitter = checkpoint_payload.get("hitter_release") or {}
    pitcher = checkpoint_payload.get("pitcher_release") or {}
    hitter_row = _date_row(hitter, slate)
    pitcher_row = _date_row(pitcher, slate)

    checks = [
        _check(
            "all_games_finalized",
            games_final,
            f"{final_games}/{games} games final",
        ),
        _check(
            "valid_close_coverage",
            _rate_ok(close_payload.get("valid_close_coverage"), _MIN_VALID_CLOSE_COVERAGE),
            f"{float(close_payload.get('valid_close_coverage') or 0.0):.1%} valid exact closes; required >= {_MIN_VALID_CLOSE_COVERAGE:.0%}",
        ),
        _check(
            "stale_close_rate",
            _rate_ok(close_payload.get("stale_close_rate"), _MAX_STALE_CLOSE_RATE, lower_is_better=True),
            f"{float(close_payload.get('stale_close_rate') or 0.0):.1%} stale closes; required <= {_MAX_STALE_CLOSE_RATE:.0%}",
        ),
        _check(
            "targeted_close_captures",
            target_events > 0 and captured_events == target_events,
            f"{captured_events}/{target_events} events have T-120/T-60/T-20 captures",
        ),
        _check(
            "hitter_anchor_settled_or_voided",
            bool(hitter_row and hitter_row.get("complete") and int(hitter_row.get("pending_rows") or 0) == 0),
            (
                "missing hitter anchor row"
                if hitter_row is None
                else f"{hitter_row.get('graded_rows', 0)} graded, {hitter_row.get('pending_rows', 0)} pending, {hitter_row.get('void_rows', 0)} void"
            ),
        ),
        _check(
            "pitcher_anchor_settled_or_voided",
            bool(pitcher_row and pitcher_row.get("complete") and int(pitcher_row.get("pending_rows") or 0) == 0),
            (
                "missing pitcher anchor row"
                if pitcher_row is None
                else f"{pitcher_row.get('graded_rows', 0)} graded, {pitcher_row.get('pending_rows', 0)} pending, {pitcher_row.get('void_rows', 0)} void"
            ),
        ),
        _check(
            "frozen_hitter_artifact_valid",
            bool((checkpoint_payload.get("artifact_integrity") or {}).get("valid")),
            str((checkpoint_payload.get("artifact_integrity") or {}).get("reason") or "sha256 matched"),
        ),
    ]
    required_failures = [row for row in checks if row["required"] and not row["passed"]]
    status = "pass"
    if required_failures:
        status = "provisional" if not games_final else "fail"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "slate_date": slate,
        "status": status,
        "checks": checks,
        "required_failures": required_failures,
        "close_summary": {
            "locked_offer_rows": close_payload.get("locked_offer_rows"),
            "valid_closes": close_payload.get("valid_closes"),
            "valid_close_coverage": close_payload.get("valid_close_coverage"),
            "stale_close_rate": close_payload.get("stale_close_rate"),
            "failure_reasons": close_payload.get("failure_reasons") or {},
        },
        "checkpoint_summary": {
            "status": checkpoint_payload.get("status"),
            "hitter_completed_dates": hitter.get("completed_date_count"),
            "hitter_dates_remaining": hitter.get("dates_remaining"),
            "hitter_micro_ready_buckets": hitter.get("micro_ready_buckets"),
            "pitcher_completed_dates": pitcher.get("completed_date_count"),
            "pitcher_dates_remaining": pitcher.get("dates_remaining"),
            "pitcher_micro_ready_buckets": pitcher.get("micro_ready_buckets"),
        },
        "decision": (
            "slate_trusted_for_evaluation"
            if status == "pass"
            else "wait_for_final_results_or_repair_failed_checks"
            if status == "provisional"
            else "do_not_use_slate_for_clean_promotion_evidence"
        ),
    }


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def _render(payload: dict[str, Any]) -> str:
    close = payload.get("close_summary") or {}
    checkpoint = payload.get("checkpoint_summary") or {}
    lines = [
        "# MLB Prop Daily Slate Trust Monitor",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Slate: {payload['slate_date']}",
        f"Status: **{payload['status'].upper()}**",
        f"Decision: `{payload['decision']}`",
        "",
        "## Required Checks",
        "",
        "| Check | Pass | Detail |",
        "|---|---|---|",
    ]
    for row in payload.get("checks") or []:
        lines.append(f"| {row['name']} | {row['passed']} | {row['detail']} |")
    lines.extend([
        "",
        "## Close Quality",
        "",
        f"- Locked executable offers: {close.get('locked_offer_rows')}",
        f"- Valid exact closes: {close.get('valid_closes')} ({_fmt_pct(close.get('valid_close_coverage'))})",
        f"- Stale close rate: {_fmt_pct(close.get('stale_close_rate'))}",
        "",
        "| Close Reason | Rows |",
        "|---|---:|",
    ])
    for reason, count in (close.get("failure_reasons") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend([
        "",
        "## Frozen Release Checkpoint",
        "",
        f"- Checkpoint status: `{checkpoint.get('status')}`",
        f"- Hitter dates: {checkpoint.get('hitter_completed_dates')} completed, {checkpoint.get('hitter_dates_remaining')} remaining",
        f"- Pitcher dates: {checkpoint.get('pitcher_completed_dates')} completed, {checkpoint.get('pitcher_dates_remaining')} remaining",
        f"- Hitter $1 micro buckets: {checkpoint.get('hitter_micro_ready_buckets')}",
        f"- Pitcher $1 micro buckets: {checkpoint.get('pitcher_micro_ready_buckets')}",
    ])
    return "\n".join(lines) + "\n"


def build(
    *,
    slate_date: date,
    model_dir: Path = _MODEL_DIR,
    pg_dsn: str = PG_DSN,
) -> dict[str, Any]:
    close_payload = build_close_diagnostic(slate_date, pg_dsn)
    checkpoint_payload = build_five_date_checkpoint(model_dir, pg_dsn)
    payload = evaluate_trust(close_payload, checkpoint_payload, slate_date=slate_date)
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_daily_slate_trust_monitor.json", payload)
    atomic_write_json(model_dir / f"prop_daily_slate_trust_monitor_{slate_date.isoformat()}.json", payload)
    report = _render(payload)
    latest = _REPORT_DIR / "mlb_prop_daily_slate_trust_latest.md"
    dated = _REPORT_DIR / f"mlb_prop_daily_slate_trust_{slate_date.isoformat()}.md"
    atomic_write_text(latest, report)
    atomic_write_text(dated, report)
    payload["report_path"] = str(latest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop daily slate trust monitor")
    parser.add_argument("--date", help="ET slate date, YYYY-MM-DD. Defaults to today ET.")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    slate_date = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    payload = build(slate_date=slate_date, model_dir=Path(args.model_dir), pg_dsn=args.pg_dsn)
    print(json.dumps({
        "slate_date": payload["slate_date"],
        "status": payload["status"],
        "decision": payload["decision"],
        "failures": [row["name"] for row in payload["required_failures"]],
        "report_path": payload["report_path"],
    }, indent=2))
    if args.fail_on_fail and payload["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
