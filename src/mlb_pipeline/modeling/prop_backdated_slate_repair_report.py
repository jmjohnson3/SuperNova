"""Repairability report for near-miss backdated MLB prop slates."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .model_release import canonical_forecast_phase
from .prop_backdated_slate_eligibility_audit import build as build_eligibility_audit

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

CORE_FORECAST_STATS = {
    "game_run_diff",
    "game_home_win",
    "game_total",
    "pitcher_strikeouts",
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
}
OPPORTUNITY_ONLY_STATS = {
    "pitcher_batters_faced",
    "pitcher_pitch_count",
    "pitcher_innings",
    "hitter_plate_appearances",
    "hitter_plate_appearances_challenger",
}
RECOVERABLE_CLOSE_REASONS = {
    "close_outside_two_hour_window",
    "no_valid_close_snapshot",
    "missing_close_side_price",
}
TRUE_UNAVAILABLE_CLOSE_REASONS = {
    "player_prop_unavailable_at_close",
    "player_market_unavailable_at_close",
    "line_disappeared_at_close",
    "exact_line_unavailable_at_close",
    "fallback_other_book_only",
    "stale_close_before_lock",
}


@dataclass(frozen=True)
class RepairConfig:
    min_valid_clv_coverage: float = 0.90
    max_near_miss_rows: int = 24


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _query_rows(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _pending_rows(conn, dates: list[date], phase: str) -> dict[str, list[dict[str, Any]]]:
    if not dates:
        return {}
    rows = _query_rows(
        conn,
        """
        SELECT game_date_et, stat, forecast_type,
               COALESCE(model_family, 'unknown') AS model_family,
               COALESCE(model_version, 'unknown') AS model_version,
               COUNT(*)::int AS rows
        FROM bets.mlb_daily_forecast_ledger
        WHERE game_date_et = ANY(%(dates)s::date[])
          AND forecast_phase = %(phase)s
          AND result_status = 'pending'
        GROUP BY 1,2,3,4,5
        ORDER BY game_date_et, stat, model_version
        """,
        {"dates": [d.isoformat() for d in dates], "phase": phase},
    )
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row["game_date_et"])].append(row)
    return dict(by_date)


def _model_version_rows(conn, dates: list[date], phase: str) -> dict[str, list[dict[str, Any]]]:
    if not dates:
        return {}
    rows = _query_rows(
        conn,
        """
        SELECT game_date_et,
               COALESCE(NULLIF(TRIM(model_version), ''), 'unknown') AS model_version,
               COUNT(*)::int AS rows
        FROM bets.mlb_daily_forecast_ledger
        WHERE game_date_et = ANY(%(dates)s::date[])
          AND forecast_phase = %(phase)s
        GROUP BY 1,2
        ORDER BY game_date_et, rows DESC
        """,
        {"dates": [d.isoformat() for d in dates], "phase": phase},
    )
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row["game_date_et"])].append(row)
    return dict(by_date)


def _snapshot_late_rows(conn, dates: list[date]) -> dict[str, dict[str, Any]]:
    if not dates:
        return {}
    rows = _query_rows(
        conn,
        """
        SELECT as_of_date AS slate_date,
               COUNT(*) FILTER (
                   WHERE snapshot_role = 'lock'
                     AND selected_side IN ('over', 'under')
               )::int AS side_lock_rows,
               COUNT(*) FILTER (
                   WHERE snapshot_role = 'lock'
                     AND selected_side IN ('over', 'under')
                     AND (snapshot_at_utc >= commence_time_utc OR commence_time_utc IS NULL)
               )::int AS side_locks_not_before_start
        FROM odds.mlb_player_prop_line_snapshots
        WHERE as_of_date = ANY(%(dates)s::date[])
        GROUP BY as_of_date
        """,
        {"dates": [d.isoformat() for d in dates]},
    )
    return {str(row["slate_date"]): row for row in rows}


def _load_close_diagnostic(model_dir: Path, slate: str) -> dict[str, Any]:
    return _load_json(model_dir / f"prop_end_of_slate_close_{slate}.json")


def _close_repair(row: dict[str, Any], model_dir: Path, cfg: RepairConfig) -> dict[str, Any]:
    slate = str(row.get("slate_date"))
    diag = _load_close_diagnostic(model_dir, slate)
    valid_rate = _as_float(row.get("valid_clv_coverage"))
    if valid_rate is None or valid_rate >= cfg.min_valid_clv_coverage:
        return {
            "status": "passed",
            "gap_rows": 0,
            "valid_coverage": valid_rate,
            "repairable_rows": 0,
            "permanent_rows": 0,
            "reason": "valid_close_coverage_passed",
            "failure_reasons": {},
        }
    locked_rows = _as_int(diag.get("locked_offer_rows") or row.get("prop_training_rows"))
    valid_rows = _as_int(diag.get("valid_closes") or row.get("valid_clv_rows"))
    needed = max(0, math.ceil(cfg.min_valid_clv_coverage * locked_rows) - valid_rows)
    reasons = diag.get("failure_reasons") or {}
    repairable_rows = sum(_as_int(reasons.get(reason)) for reason in RECOVERABLE_CLOSE_REASONS)
    permanent_rows = sum(_as_int(reasons.get(reason)) for reason in TRUE_UNAVAILABLE_CLOSE_REASONS)
    if repairable_rows >= needed > 0:
        status = "repairable"
        reason = "enough_recoverable_close_rows_exist"
    elif repairable_rows > 0:
        status = "partially_repairable"
        reason = "some_recoverable_close_rows_but_not_enough_for_90pct"
    else:
        status = "not_repairable"
        reason = "missing_clv_rows_are_true_unavailable_or_no_recoverable_snapshot_reason"
    return {
        "status": status,
        "gap_rows": needed,
        "valid_coverage": valid_rate,
        "locked_rows": locked_rows,
        "valid_rows": valid_rows,
        "repairable_rows": repairable_rows,
        "permanent_rows": permanent_rows,
        "reason": reason,
        "failure_reasons": reasons,
    }


def _pending_repair(pending: list[dict[str, Any]]) -> dict[str, Any]:
    if not pending:
        return {
            "status": "passed",
            "pending_rows": 0,
            "core_pending_rows": 0,
            "opportunity_pending_rows": 0,
            "reason": "no_pending_forecast_rows",
            "rows": [],
        }
    core = sum(_as_int(row.get("rows")) for row in pending if str(row.get("stat")) in CORE_FORECAST_STATS)
    opportunity = sum(_as_int(row.get("rows")) for row in pending if str(row.get("stat")) in OPPORTUNITY_ONLY_STATS)
    total = sum(_as_int(row.get("rows")) for row in pending)
    if core == 0 and opportunity == total:
        status = "repairable"
        reason = "pending_rows_are_opportunity_only; strict promotion can ignore_or_void_ungradeable_opportunity_rows"
    elif core > 0:
        status = "repairable"
        reason = "core_pending_rows_need_result_join_or_void_repair"
    else:
        status = "partially_repairable"
        reason = "pending_rows_are_non_core_unknown_stats"
    return {
        "status": status,
        "pending_rows": total,
        "core_pending_rows": core,
        "opportunity_pending_rows": opportunity,
        "reason": reason,
        "rows": pending,
    }


def _model_version_repair(row: dict[str, Any], versions: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = _as_float(row.get("model_version_coverage"))
    if coverage is not None and coverage >= 0.95:
        return {"status": "passed", "coverage": coverage, "reason": "model_version_coverage_passed", "versions": versions}
    if _as_int(row.get("forecast_rows")) <= 0:
        return {
            "status": "not_repairable",
            "coverage": coverage,
            "reason": "no_immutable_forecast_ledger_rows_for_this_date",
            "versions": versions,
        }
    unversioned = sum(
        _as_int(item.get("rows")) for item in versions
        if str(item.get("model_version") or "").lower() in {"unknown", "current", ""}
    )
    if unversioned > 0:
        return {
            "status": "not_repairable_without_external_artifact_proof",
            "coverage": coverage,
            "reason": "forecast_rows_use_current_or_unknown_model_version",
            "versions": versions,
        }
    return {
        "status": "partially_repairable",
        "coverage": coverage,
        "reason": "model_version_rows_exist_but_do_not_clear_coverage_threshold",
        "versions": versions,
    }


def _side_lock_repair(row: dict[str, Any], late: dict[str, Any]) -> dict[str, Any]:
    late_rows = _as_int(late.get("side_locks_not_before_start") or row.get("side_locks_not_before_start"))
    total = _as_int(late.get("side_lock_rows") or row.get("side_lock_rows"))
    if late_rows <= 0:
        return {"status": "passed", "late_rows": 0, "side_lock_rows": total, "reason": "all_side_locks_before_start"}
    return {
        "status": "partially_repairable",
        "late_rows": late_rows,
        "side_lock_rows": total,
        "reason": "late_or_missing_start_side_locks_must_be_excluded_then_coverage_recomputed",
    }


def _no_ledger_repair(row: dict[str, Any]) -> dict[str, Any]:
    if _as_int(row.get("forecast_rows")) > 0:
        return {"status": "passed", "reason": "forecast_ledger_rows_exist"}
    return {
        "status": "not_repairable",
        "reason": "cannot_create_prospective_forecasts_after_results_without_hindsight_risk",
    }


def _combine_status(blockers: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status")) for item in blockers if item.get("status") and item.get("status") != "passed"}
    if not statuses:
        return "countable"
    if any(status.startswith("not_repairable") for status in statuses):
        return "not_countable"
    if statuses == {"repairable"}:
        return "repairable"
    return "partially_repairable"


def _repair_row(
    row: dict[str, Any],
    *,
    model_dir: Path,
    pending: dict[str, list[dict[str, Any]]],
    versions: dict[str, list[dict[str, Any]]],
    late_locks: dict[str, dict[str, Any]],
    cfg: RepairConfig,
) -> dict[str, Any]:
    slate = str(row.get("slate_date"))
    blocker_details: list[dict[str, Any]] = []
    blockers = list(row.get("blockers") or [])
    if row.get("prop_promotion_countable") and not row.get("eligible_to_count"):
        blocker_details.append({
            "blocker": "strict_only_non_prop_blockers",
            "status": "prop_promotion_countable",
            "reason": "core_prop_forecasts_and_prop_close_evidence_pass; remaining blockers are not prop-promotion blockers",
        })
    if "no_canonical_forecast_ledger_rows" in blockers or "no_graded_forecasts" in blockers:
        blocker_details.append({"blocker": "forecast_ledger_missing", **_no_ledger_repair(row)})
    if any(str(blocker).startswith("model_version_coverage<") for blocker in blockers):
        blocker_details.append({"blocker": "model_version_coverage", **_model_version_repair(row, versions.get(slate, []))})
    pending_blockers = {
        "pending_forecasts_on_settled_games",
        "core_prop_pending_on_settled_games",
        "game_forecast_pending_on_settled_games",
        "opportunity_pending_on_settled_games",
        "other_forecast_pending_on_settled_games",
    }
    if any(blocker in blockers for blocker in pending_blockers):
        blocker_details.append({"blocker": "pending_forecasts_on_settled_games", **_pending_repair(pending.get(slate, []))})
    if "valid_clv_coverage<90%" in blockers:
        blocker_details.append({"blocker": "valid_clv_coverage", **_close_repair(row, model_dir, cfg)})
    if "side_locks_not_before_start" in blockers:
        blocker_details.append({"blocker": "side_locks_not_before_start", **_side_lock_repair(row, late_locks.get(slate, {}))})
    for blocker in blockers:
        if blocker in {
            "no_canonical_forecast_ledger_rows",
            "no_graded_forecasts",
            *pending_blockers,
            "valid_clv_coverage<90%",
            "side_locks_not_before_start",
        } or str(blocker).startswith("model_version_coverage<"):
            continue
        blocker_details.append({
            "blocker": blocker,
            "status": "not_repairable",
            "reason": "not_a_known_repairable_backdate_blocker",
        })
    status = _combine_status(blocker_details)
    if row.get("eligible_to_count"):
        status = "countable"
    elif row.get("prop_promotion_countable"):
        status = "prop_promotion_countable"
    return {
        "slate_date": slate,
        "status": status,
        "strict_countable": bool(row.get("eligible_to_count")),
        "prop_promotion_countable": bool(row.get("prop_promotion_countable")),
        "prop_close_countable": bool(row.get("prop_close_countable")),
        "forecast_countable": bool(row.get("forecast_countable")),
        "prop_rows": _as_int(row.get("prop_training_rows")),
        "forecast_rows": _as_int(row.get("forecast_rows")),
        "valid_clv_coverage": row.get("valid_clv_coverage"),
        "stale_close_rate": row.get("stale_close_rate"),
        "true_pair_rate": row.get("true_pair_rate"),
        "original_blockers": blockers,
        "blocker_details": blocker_details,
        "recommended_action": _recommended_action(status, blocker_details),
    }


def _recommended_action(status: str, details: list[dict[str, Any]]) -> str:
    if status == "countable":
        return "Count this slate as verified-live evidence."
    if status == "prop_promotion_countable":
        return "Count this slate for prop-promotion evidence; keep it out of strict all-layer evidence until non-prop blockers are cleared."
    blockers = {str(item.get("blocker")): item for item in details}
    if status == "repairable" and "pending_forecasts_on_settled_games" in blockers:
        return "Repair audit logic or ledger status for ungradeable opportunity-only pending rows, then rerun eligibility."
    if "valid_clv_coverage" in blockers:
        close = blockers["valid_clv_coverage"]
        if close.get("status") == "not_repairable":
            return "Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue."
        return "Try close-snapshot resolver refresh, then rerun close diagnostic and eligibility."
    if "forecast_ledger_missing" in blockers:
        return "Use this date for CLV/bookability research only; do not count it as forecast-proof evidence."
    if "model_version_coverage" in blockers:
        return "Do not count strict promotion unless immutable model-version evidence exists outside the ledger."
    if "side_locks_not_before_start" in blockers:
        return "Exclude late side locks and recompute close coverage; count only if the remaining slate still clears thresholds."
    return "Keep for diagnostics only."


def build(
    *,
    model_dir: Path = _MODEL_DIR,
    pg_dsn: str = PG_DSN,
    cfg: RepairConfig = RepairConfig(),
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    audit_path = model_dir / "prop_backdated_slate_eligibility_audit.json"
    audit = _load_json(audit_path)
    if not audit:
        today = datetime.now(timezone.utc).date()
        audit = build_eligibility_audit(date_from=today.replace(day=1), date_to=today, model_dir=model_dir, pg_dsn=pg_dsn)
    phase = str(audit.get("canonical_forecast_phase") or canonical_forecast_phase())
    near_misses = list(audit.get("near_misses") or [])
    near_misses = near_misses[: cfg.max_near_miss_rows]
    dates = [_date_value(row.get("slate_date")) for row in near_misses]
    dates = [value for value in dates if value is not None]
    with psycopg2.connect(pg_dsn) as conn:
        pending = _pending_rows(conn, dates, phase)
        versions = _model_version_rows(conn, dates, phase)
        late_locks = _snapshot_late_rows(conn, dates)
    rows = [
        _repair_row(
            row,
            model_dir=model_dir,
            pending=pending,
            versions=versions,
            late_locks=late_locks,
            cfg=cfg,
        )
        for row in near_misses
    ]
    status_counts = Counter(row["status"] for row in rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_audit_generated_at_utc": audit.get("generated_at_utc"),
        "source_date_from": audit.get("date_from"),
        "source_date_to": audit.get("date_to"),
        "canonical_forecast_phase": phase,
        "status_counts": dict(status_counts),
        "rows": rows,
        "strict_eligible_dates": audit.get("eligible_dates") or [],
        "prop_promotion_countable_dates": audit.get("prop_promotion_countable_dates") or [],
        "prop_close_countable_dates": audit.get("prop_close_countable_dates") or [],
        "forecast_countable_dates": audit.get("forecast_countable_dates") or [],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(payload)
    atomic_write_json(model_dir / "prop_backdated_slate_repair_report.json", payload)
    report = _render(payload)
    report_path = _REPORT_DIR / "mlb_prop_backdated_slate_repair_latest.md"
    atomic_write_text(report_path, report)
    payload["report_path"] = str(report_path)
    return payload


def _fmt_pct(value: Any) -> str:
    numeric = _as_float(value)
    return "-" if numeric is None else f"{numeric:.1%}"


def _detail_text(details: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in details:
        blocker = item.get("blocker")
        status = item.get("status")
        reason = item.get("reason")
        if blocker == "valid_clv_coverage":
            chunks.append(
                f"{blocker}: {status}, need {item.get('gap_rows', 0)} rows, "
                f"recoverable {item.get('repairable_rows', 0)}, true-unavailable {item.get('permanent_rows', 0)}"
            )
        elif blocker == "pending_forecasts_on_settled_games":
            chunks.append(
                f"{blocker}: {status}, pending {item.get('pending_rows', 0)} "
                f"(core {item.get('core_pending_rows', 0)}, opportunity {item.get('opportunity_pending_rows', 0)})"
            )
        else:
            chunks.append(f"{blocker}: {status}, {reason}")
    return "; ".join(chunks) or "-"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Backdated Slate Repair Report",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Source audit UTC: {payload.get('source_audit_generated_at_utc')}",
        f"Date range: {payload.get('source_date_from')} to {payload.get('source_date_to')}",
        f"Canonical phase: `{payload.get('canonical_forecast_phase')}`",
        "",
        "This report says whether each near-miss backdated slate can be repaired without inventing lock, close, model-version, or result evidence.",
        "",
        "## Summary",
        "",
        "| Status | Dates |",
        "|---|---:|",
    ]
    for status, count in sorted((payload.get("status_counts") or {}).items()):
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        f"- Strict eligible dates now: {', '.join(payload.get('strict_eligible_dates') or []) or '-'}",
        f"- Prop-promotion countable dates: {', '.join(payload.get('prop_promotion_countable_dates') or []) or '-'}",
        f"- Prop close countable dates: {', '.join(payload.get('prop_close_countable_dates') or []) or '-'}",
        f"- Forecast countable dates: {', '.join(payload.get('forecast_countable_dates') or []) or '-'}",
        "",
        "## Near-Miss Repairability",
        "",
        "| Date | Status | Prop Promotion | Prop Close | Forecast | Prop Rows | Forecast Rows | Valid CLV | Details | Action |",
        "|---|---|---|---|---|---:|---:|---:|---|---|",
    ])
    for row in payload.get("rows") or []:
        lines.append(
            f"| {row['slate_date']} | {row['status']} | {row.get('prop_promotion_countable', False)} | {row['prop_close_countable']} | "
            f"{row['forecast_countable']} | {row['prop_rows']} | {row['forecast_rows']} | "
            f"{_fmt_pct(row.get('valid_clv_coverage'))} | {_detail_text(row.get('blocker_details') or [])} | "
            f"{row.get('recommended_action') or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB backdated slate repairability report")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    payload = build(model_dir=Path(args.model_dir), pg_dsn=args.pg_dsn)
    print(json.dumps({
        "status_counts": payload["status_counts"],
        "report_path": payload["report_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
