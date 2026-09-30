"""Audit historical MLB prop slates for prospective-equivalent eligibility."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .daily_forecast_ledger import TERMINAL_NON_FINAL_GAME_STATUSES
from .model_release import canonical_forecast_phase

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
CORE_PROP_STATS = {
    "pitcher_strikeouts",
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
}
GAME_STATS = {"game_run_diff", "game_home_win", "game_total"}
OPPORTUNITY_STATS = {
    "pitcher_batters_faced",
    "pitcher_pitch_count",
    "pitcher_innings",
    "hitter_plate_appearances",
    "hitter_plate_appearances_challenger",
}


@dataclass(frozen=True)
class BackdatedSlateAuditConfig:
    min_prop_training_rows: int = 100
    min_side_lock_rows: int = 100
    min_valid_clv_coverage: float = 0.90
    max_stale_close_rate: float = 0.02
    max_missing_lock_rate: float = 0.02
    min_model_version_coverage: float = 0.95


def _query_rows(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _rate(num: Any, den: Any) -> float | None:
    try:
        numerator = float(num or 0)
        denominator = float(den or 0)
    except (TypeError, ValueError):
        return None
    return numerator / denominator if denominator > 0 else None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _date_key(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _by_date(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_date_key(row.get("slate_date")): row for row in rows if row.get("slate_date") is not None}


def _games(conn, date_from: date, date_to: date) -> dict[str, dict[str, Any]]:
    return _by_date(_query_rows(
        conn,
        """
        SELECT
            game_date_et AS slate_date,
            COUNT(*)::int AS games,
            COUNT(*) FILTER (WHERE status = 'final')::int AS final_games,
            COUNT(*) FILTER (
                WHERE LOWER(COALESCE(status, '')) = ANY(%(terminal_non_final)s)
            )::int AS terminal_non_final_games,
            COUNT(*) FILTER (
                WHERE status = 'final'
                   OR LOWER(COALESCE(status, '')) = ANY(%(terminal_non_final)s)
            )::int AS settled_games,
            COUNT(*) FILTER (WHERE start_ts_utc IS NULL)::int AS games_missing_start,
            MIN(start_ts_utc) AS first_start_utc,
            MAX(start_ts_utc) AS last_start_utc
        FROM raw.mlb_games
        WHERE game_date_et BETWEEN %(date_from)s AND %(date_to)s
        GROUP BY game_date_et
        """,
        {
            "date_from": date_from,
            "date_to": date_to,
            "terminal_non_final": list(TERMINAL_NON_FINAL_GAME_STATUSES),
        },
    ))


def _forecasts(conn, date_from: date, date_to: date, phase: str) -> dict[str, dict[str, Any]]:
    return _by_date(_query_rows(
        conn,
        """
        WITH canonical AS (
            SELECT DISTINCT ON (
                f.game_date_et, f.game_slug, COALESCE(f.player_id, 0), f.stat,
                COALESCE(f.model_family, 'unknown'), COALESCE(f.model_version, 'unknown')
            )
                f.game_date_et AS slate_date,
                f.locked_at_utc,
                f.forecast_run_id,
                f.forecast_type,
                f.stat,
                f.result_status,
                f.model_version,
                g.start_ts_utc
            FROM bets.mlb_daily_forecast_ledger f
            LEFT JOIN raw.mlb_games g ON g.game_slug = f.game_slug
            WHERE f.game_date_et BETWEEN %(date_from)s AND %(date_to)s
              AND f.forecast_phase = %(phase)s
            ORDER BY f.game_date_et, f.game_slug, COALESCE(f.player_id, 0), f.stat,
                     COALESCE(f.model_family, 'unknown'), COALESCE(f.model_version, 'unknown'),
                     f.locked_at_utc DESC, f.id DESC
        )
        SELECT
            slate_date,
            COUNT(*)::int AS forecast_rows,
            COUNT(*) FILTER (WHERE locked_at_utc < start_ts_utc)::int AS forecast_locks_before_start,
            COUNT(*) FILTER (WHERE locked_at_utc >= start_ts_utc OR start_ts_utc IS NULL)::int AS forecast_locks_not_before_start,
            COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded_forecasts,
            COUNT(*) FILTER (WHERE result_status LIKE 'void%%')::int AS void_forecasts,
            COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending_forecasts,
            COUNT(*) FILTER (WHERE stat = ANY(%(core_prop_stats)s))::int AS core_prop_forecasts,
            COUNT(*) FILTER (
                WHERE stat = ANY(%(core_prop_stats)s)
                  AND locked_at_utc < start_ts_utc
            )::int AS core_prop_locks_before_start,
            COUNT(*) FILTER (
                WHERE stat = ANY(%(core_prop_stats)s)
                  AND (locked_at_utc >= start_ts_utc OR start_ts_utc IS NULL)
            )::int AS core_prop_locks_not_before_start,
            COUNT(*) FILTER (
                WHERE stat = ANY(%(core_prop_stats)s)
                  AND result_status = 'graded'
            )::int AS graded_core_prop_forecasts,
            COUNT(*) FILTER (
                WHERE stat = ANY(%(core_prop_stats)s)
                  AND result_status LIKE 'void%%'
            )::int AS void_core_prop_forecasts,
            COUNT(*) FILTER (
                WHERE stat = ANY(%(core_prop_stats)s)
                  AND COALESCE(NULLIF(TRIM(model_version), ''), 'unknown') NOT IN ('unknown', 'current')
            )::int AS model_versioned_core_prop_forecasts,
            COUNT(*) FILTER (
                WHERE result_status = 'pending'
            )::int AS pending_settled_forecasts,
            COUNT(*) FILTER (
                WHERE result_status = 'pending'
                  AND stat = ANY(%(core_prop_stats)s)
            )::int AS pending_core_prop_forecasts,
            COUNT(*) FILTER (
                WHERE result_status = 'pending'
                  AND (forecast_type = 'game' OR stat = ANY(%(game_stats)s))
            )::int AS pending_game_forecasts,
            COUNT(*) FILTER (
                WHERE result_status = 'pending'
                  AND (forecast_type = 'opportunity' OR stat = ANY(%(opportunity_stats)s))
            )::int AS pending_opportunity_forecasts,
            COUNT(*) FILTER (
                WHERE result_status = 'pending'
                  AND stat <> ALL(%(core_prop_stats)s)
                  AND stat <> ALL(%(game_stats)s)
                  AND stat <> ALL(%(opportunity_stats)s)
            )::int AS pending_other_forecasts,
            COUNT(*) FILTER (
                WHERE COALESCE(NULLIF(TRIM(model_version), ''), 'unknown') NOT IN ('unknown', 'current')
            )::int AS model_versioned_forecasts,
            COUNT(DISTINCT forecast_run_id)::int AS forecast_runs,
            MIN(locked_at_utc) AS first_forecast_lock_utc,
            MAX(locked_at_utc) AS last_forecast_lock_utc
        FROM canonical
        GROUP BY slate_date
        """,
        {
            "date_from": date_from,
            "date_to": date_to,
            "phase": phase,
            "core_prop_stats": sorted(CORE_PROP_STATS),
            "game_stats": sorted(GAME_STATS),
            "opportunity_stats": sorted(OPPORTUNITY_STATS),
        },
    ))


def _snapshots(conn, date_from: date, date_to: date) -> dict[str, dict[str, Any]]:
    return _by_date(_query_rows(
        conn,
        """
        SELECT
            as_of_date AS slate_date,
            COUNT(*) FILTER (WHERE snapshot_role = 'open')::int AS open_snapshot_rows,
            COUNT(*) FILTER (WHERE snapshot_role = 'lock')::int AS lock_snapshot_rows,
            COUNT(*) FILTER (
                WHERE snapshot_role = 'lock'
                  AND selected_side IN ('over', 'under')
            )::int AS side_lock_rows,
            COUNT(*) FILTER (
                WHERE snapshot_role = 'lock'
                  AND selected_side IN ('over', 'under')
                  AND snapshot_at_utc < commence_time_utc
            )::int AS side_locks_before_start,
            COUNT(*) FILTER (
                WHERE snapshot_role = 'lock'
                  AND selected_side IN ('over', 'under')
                  AND (snapshot_at_utc >= commence_time_utc OR commence_time_utc IS NULL)
            )::int AS side_locks_not_before_start,
            COUNT(*) FILTER (WHERE snapshot_role = 'close')::int AS close_snapshot_rows,
            COUNT(DISTINCT snapshot_at_utc) FILTER (WHERE snapshot_role = 'close')::int AS close_times,
            COUNT(DISTINCT run_id) FILTER (WHERE snapshot_role = 'lock')::int AS lock_runs
        FROM odds.mlb_player_prop_line_snapshots
        WHERE as_of_date BETWEEN %(date_from)s AND %(date_to)s
        GROUP BY as_of_date
        """,
        {"date_from": date_from, "date_to": date_to},
    ))


def _close_diagnostics(model_dir: Path, date_from: date, date_to: date) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    day = date_from
    while day <= date_to:
        path = model_dir / f"prop_end_of_slate_close_{day.isoformat()}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            rows[day.isoformat()] = {
                "close_diagnostic_available": True,
                "close_diagnostic_rows": _as_int(payload.get("locked_offer_rows")),
                "close_diagnostic_valid_closes": _as_int(payload.get("valid_closes")),
                "close_diagnostic_valid_coverage": payload.get("valid_close_coverage"),
                "close_diagnostic_stale_rate": payload.get("stale_close_rate"),
                "close_diagnostic_strict_clean": bool(payload.get("strict_clean_slate")),
            }
        day += timedelta(days=1)
    return rows


def _training(conn, date_from: date, date_to: date) -> dict[str, dict[str, Any]]:
    return _by_date(_query_rows(
        conn,
        """
        SELECT
            game_date_et AS slate_date,
            COUNT(*)::int AS prop_training_rows,
            COUNT(DISTINCT lock_snapshot_id) FILTER (WHERE lock_snapshot_id IS NOT NULL)::int AS distinct_lock_snapshots,
            COUNT(*) FILTER (WHERE lock_snapshot_id IS NULL OR clv_unknown_reason = 'missing_lock_snapshot')::int AS missing_lock_rows,
            COUNT(*) FILTER (WHERE clv_valid IS TRUE)::int AS valid_clv_rows,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'stale_close_before_lock')::int AS stale_close_rows,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'close_outside_two_hour_window')::int AS outside_window_rows,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'line_disappeared_at_close')::int AS line_disappeared_rows,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'player_prop_unavailable_at_close')::int AS player_prop_unavailable_rows,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'player_market_unavailable_at_close')::int AS player_market_unavailable_rows,
            COUNT(*) FILTER (WHERE clv_unknown_reason = 'no_valid_close_snapshot')::int AS no_valid_close_rows,
            COUNT(*) FILTER (WHERE true_pair_flag = 1)::int AS true_pair_rows,
            COUNT(*) FILTER (WHERE synthetic_pair_flag = 1)::int AS synthetic_pair_rows,
            COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded_prop_rows
        FROM features.mlb_prop_market_training_examples
        WHERE game_date_et BETWEEN %(date_from)s AND %(date_to)s
        GROUP BY game_date_et
        """,
        {"date_from": date_from, "date_to": date_to},
    ))


def _reasons(row: dict[str, Any], cfg: BackdatedSlateAuditConfig) -> list[str]:
    reasons: list[str] = []
    games = _as_int(row.get("games"))
    settled_games = _as_int(row.get("settled_games"))
    forecast_rows = _as_int(row.get("forecast_rows"))
    prop_rows = _as_int(row.get("prop_training_rows"))
    side_locks = _as_int(row.get("side_lock_rows"))

    if games <= 0:
        reasons.append("no_games")
    elif settled_games < games:
        reasons.append("games_not_settled")
    if _as_int(row.get("games_missing_start")) > 0:
        reasons.append("games_missing_start_time")

    if forecast_rows <= 0:
        reasons.append("no_canonical_forecast_ledger_rows")
    if _as_int(row.get("forecast_locks_not_before_start")) > 0:
        reasons.append("forecast_locks_not_before_start")
    if _as_int(row.get("pending_core_prop_forecasts")) > 0:
        reasons.append("core_prop_pending_on_settled_games")
    if _as_int(row.get("pending_game_forecasts")) > 0:
        reasons.append("game_forecast_pending_on_settled_games")
    if _as_int(row.get("pending_opportunity_forecasts")) > 0:
        reasons.append("opportunity_pending_on_settled_games")
    if _as_int(row.get("pending_other_forecasts")) > 0:
        reasons.append("other_forecast_pending_on_settled_games")
    if _as_int(row.get("graded_forecasts")) <= 0:
        reasons.append("no_graded_forecasts")
    model_version_coverage = row.get("model_version_coverage")
    if model_version_coverage is None or model_version_coverage < cfg.min_model_version_coverage:
        reasons.append(f"model_version_coverage<{cfg.min_model_version_coverage:.0%}")

    if prop_rows < cfg.min_prop_training_rows:
        reasons.append(f"prop_training_rows<{cfg.min_prop_training_rows}")
    if side_locks < cfg.min_side_lock_rows:
        reasons.append(f"side_lock_rows<{cfg.min_side_lock_rows}")
    if _as_int(row.get("side_locks_not_before_start")) > 0:
        reasons.append("side_locks_not_before_start")
    if _as_int(row.get("close_times")) <= 0:
        reasons.append("no_close_snapshot_times")
    valid_clv_coverage = row.get("valid_clv_coverage")
    if valid_clv_coverage is None or valid_clv_coverage < cfg.min_valid_clv_coverage:
        reasons.append(f"valid_clv_coverage<{cfg.min_valid_clv_coverage:.0%}")
    stale_close_rate = row.get("stale_close_rate")
    if stale_close_rate is None or stale_close_rate > cfg.max_stale_close_rate:
        reasons.append(f"stale_close_rate>{cfg.max_stale_close_rate:.0%}")
    missing_lock_rate = row.get("missing_lock_rate")
    if missing_lock_rate is None or missing_lock_rate > cfg.max_missing_lock_rate:
        reasons.append(f"missing_lock_rate>{cfg.max_missing_lock_rate:.0%}")
    if _as_int(row.get("true_pair_rows")) <= 0 and prop_rows > 0:
        reasons.append("true_pair_rows=0")
    return reasons


def _prop_promotion_blockers(row: dict[str, Any], cfg: BackdatedSlateAuditConfig) -> list[str]:
    blockers: list[str] = []
    games = _as_int(row.get("games"))
    settled_games = _as_int(row.get("settled_games"))
    core_rows = _as_int(row.get("core_prop_forecasts"))
    prop_rows = _as_int(row.get("prop_training_rows"))
    side_locks = _as_int(row.get("side_lock_rows"))
    if games <= 0:
        blockers.append("no_games")
    elif settled_games < games:
        blockers.append("games_not_settled")
    if _as_int(row.get("games_missing_start")) > 0:
        blockers.append("games_missing_start_time")
    if core_rows <= 0:
        blockers.append("no_core_prop_forecast_rows")
    if _as_int(row.get("core_prop_locks_not_before_start")) > 0:
        blockers.append("core_prop_locks_not_before_start")
    if _as_int(row.get("pending_core_prop_forecasts")) > 0:
        blockers.append("core_prop_pending_on_settled_games")
    if _as_int(row.get("pending_other_forecasts")) > 0:
        blockers.append("other_forecast_pending_on_settled_games")
    if _as_int(row.get("graded_core_prop_forecasts")) <= 0:
        blockers.append("no_graded_core_prop_forecasts")
    core_version_coverage = row.get("core_prop_model_version_coverage")
    if core_version_coverage is None or core_version_coverage < cfg.min_model_version_coverage:
        blockers.append(f"core_prop_model_version_coverage<{cfg.min_model_version_coverage:.0%}")
    if prop_rows < cfg.min_prop_training_rows:
        blockers.append(f"prop_training_rows<{cfg.min_prop_training_rows}")
    if side_locks < cfg.min_side_lock_rows:
        blockers.append(f"side_lock_rows<{cfg.min_side_lock_rows}")
    if _as_int(row.get("side_locks_not_before_start")) > 0:
        blockers.append("side_locks_not_before_start")
    if _as_int(row.get("close_times")) <= 0:
        blockers.append("no_close_snapshot_times")
    valid_clv_coverage = row.get("valid_clv_coverage")
    if valid_clv_coverage is None or valid_clv_coverage < cfg.min_valid_clv_coverage:
        blockers.append(f"valid_clv_coverage<{cfg.min_valid_clv_coverage:.0%}")
    stale_close_rate = row.get("stale_close_rate")
    if stale_close_rate is None or stale_close_rate > cfg.max_stale_close_rate:
        blockers.append(f"stale_close_rate>{cfg.max_stale_close_rate:.0%}")
    missing_lock_rate = row.get("missing_lock_rate")
    if missing_lock_rate is None or missing_lock_rate > cfg.max_missing_lock_rate:
        blockers.append(f"missing_lock_rate>{cfg.max_missing_lock_rate:.0%}")
    if _as_int(row.get("true_pair_rows")) <= 0 and prop_rows > 0:
        blockers.append("true_pair_rows=0")
    return blockers


def _split_blockers(blockers: list[str]) -> tuple[list[str], list[str]]:
    forecast_prefixes = (
        "no_canonical_forecast_ledger_rows",
        "forecast_locks_not_before_start",
        "core_prop_pending_on_settled_games",
        "game_forecast_pending_on_settled_games",
        "opportunity_pending_on_settled_games",
        "other_forecast_pending_on_settled_games",
        "no_graded_forecasts",
        "model_version_coverage<",
    )
    prop_prefixes = (
        "prop_training_rows<",
        "side_lock_rows<",
        "side_locks_not_before_start",
        "no_close_snapshot_times",
        "valid_clv_coverage<",
        "stale_close_rate>",
        "missing_lock_rate>",
        "true_pair_rows=0",
    )
    game_prefixes = (
        "no_games",
        "games_not_settled",
        "games_missing_start_time",
    )
    forecast_blockers = [
        blocker for blocker in blockers
        if blocker.startswith(forecast_prefixes) or blocker.startswith(game_prefixes)
    ]
    prop_blockers = [
        blocker for blocker in blockers
        if blocker.startswith(prop_prefixes) or blocker.startswith(game_prefixes)
    ]
    return forecast_blockers, prop_blockers


def _row(date_value: date, pieces: list[dict[str, Any]], cfg: BackdatedSlateAuditConfig) -> dict[str, Any]:
    row: dict[str, Any] = {"slate_date": date_value.isoformat()}
    for piece in pieces:
        for key, value in piece.items():
            if key == "slate_date":
                continue
            row[key] = _json_value(value)
    row["slate_date"] = date_value.isoformat()
    row["model_version_coverage"] = _rate(row.get("model_versioned_forecasts"), row.get("forecast_rows"))
    row["core_prop_model_version_coverage"] = _rate(
        row.get("model_versioned_core_prop_forecasts"),
        row.get("core_prop_forecasts"),
    )
    row["forecast_lock_before_start_rate"] = _rate(row.get("forecast_locks_before_start"), row.get("forecast_rows"))
    row["core_prop_lock_before_start_rate"] = _rate(
        row.get("core_prop_locks_before_start"),
        row.get("core_prop_forecasts"),
    )
    row["side_lock_before_start_rate"] = _rate(row.get("side_locks_before_start"), row.get("side_lock_rows"))
    row["training_valid_clv_coverage"] = _rate(row.get("valid_clv_rows"), row.get("prop_training_rows"))
    training_stale_rate = _rate(row.get("stale_close_rows"), row.get("prop_training_rows"))
    if row.get("close_diagnostic_available"):
        row["valid_clv_coverage"] = row.get("close_diagnostic_valid_coverage")
        row["stale_close_rate"] = row.get("close_diagnostic_stale_rate")
        row["close_quality_source"] = "end_of_slate_close_diagnostic"
    else:
        row["valid_clv_coverage"] = row["training_valid_clv_coverage"]
        row["stale_close_rate"] = training_stale_rate
        row["close_quality_source"] = "training_examples_fallback"
    row["missing_lock_rate"] = _rate(row.get("missing_lock_rows"), row.get("prop_training_rows"))
    row["true_pair_rate"] = _rate(row.get("true_pair_rows"), row.get("prop_training_rows"))
    row["synthetic_pair_rate"] = _rate(row.get("synthetic_pair_rows"), row.get("prop_training_rows"))
    reasons = _reasons(row, cfg)
    forecast_blockers, prop_blockers = _split_blockers(reasons)
    prop_promotion_blockers = _prop_promotion_blockers(row, cfg)
    row["forecast_countable"] = not forecast_blockers
    row["prop_close_countable"] = not prop_blockers
    row["prop_promotion_countable"] = not prop_promotion_blockers
    row["forecast_blockers"] = forecast_blockers
    row["prop_blockers"] = prop_blockers
    row["prop_promotion_blockers"] = prop_promotion_blockers
    row["eligible_to_count"] = not reasons
    row["classification"] = "verified_live_backdate" if row["eligible_to_count"] else "not_countable"
    row["blockers"] = reasons
    return row


def build(
    *,
    date_from: date,
    date_to: date,
    phase: str | None = None,
    cfg: BackdatedSlateAuditConfig = BackdatedSlateAuditConfig(),
    model_dir: Path = _MODEL_DIR,
    pg_dsn: str = PG_DSN,
) -> dict[str, Any]:
    phase = phase or canonical_forecast_phase()
    with psycopg2.connect(pg_dsn) as conn:
        games = _games(conn, date_from, date_to)
        forecasts = _forecasts(conn, date_from, date_to, phase)
        snapshots = _snapshots(conn, date_from, date_to)
        training = _training(conn, date_from, date_to)
    close_diagnostics = _close_diagnostics(model_dir, date_from, date_to)
    rows: list[dict[str, Any]] = []
    day = date_from
    while day <= date_to:
        key = day.isoformat()
        rows.append(_row(day, [
            games.get(key, {}),
            forecasts.get(key, {}),
            snapshots.get(key, {}),
            training.get(key, {}),
            close_diagnostics.get(key, {}),
        ], cfg))
        day += timedelta(days=1)
    eligible = [row for row in rows if row["eligible_to_count"]]
    prop_countable = [row for row in rows if row.get("prop_close_countable")]
    forecast_countable = [row for row in rows if row.get("forecast_countable")]
    prop_promotion_countable = [row for row in rows if row.get("prop_promotion_countable")]
    near_misses = sorted(
        [row for row in rows if not row["eligible_to_count"] and _as_int(row.get("prop_training_rows")) > 0],
        key=lambda row: (
            len(row["blockers"]),
            -_as_int(row.get("prop_training_rows")),
            row["slate_date"],
        ),
    )
    blocker_counts: dict[str, int] = {}
    for row in rows:
        for blocker in row["blockers"]:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "canonical_forecast_phase": phase,
        "thresholds": {
            "min_prop_training_rows": cfg.min_prop_training_rows,
            "min_side_lock_rows": cfg.min_side_lock_rows,
            "min_valid_clv_coverage": cfg.min_valid_clv_coverage,
            "max_stale_close_rate": cfg.max_stale_close_rate,
            "max_missing_lock_rate": cfg.max_missing_lock_rate,
            "min_model_version_coverage": cfg.min_model_version_coverage,
        },
        "eligible_count": len(eligible),
        "eligible_dates": [row["slate_date"] for row in eligible],
        "prop_close_countable_count": len(prop_countable),
        "prop_close_countable_dates": [row["slate_date"] for row in prop_countable],
        "forecast_countable_count": len(forecast_countable),
        "forecast_countable_dates": [row["slate_date"] for row in forecast_countable],
        "prop_promotion_countable_count": len(prop_promotion_countable),
        "prop_promotion_countable_dates": [row["slate_date"] for row in prop_promotion_countable],
        "audited_dates": len(rows),
        "dates_with_prop_training": sum(1 for row in rows if _as_int(row.get("prop_training_rows")) > 0),
        "dates_with_canonical_forecasts": sum(1 for row in rows if _as_int(row.get("forecast_rows")) > 0),
        "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))),
        "rows": rows,
        "near_misses": near_misses[:15],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_backdated_slate_eligibility_audit.json", payload)
    report = _render(payload)
    report_path = _REPORT_DIR / "mlb_prop_backdated_slate_eligibility_latest.md"
    atomic_write_text(report_path, report)
    payload["report_path"] = str(report_path)
    return payload


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Backdated Slate Eligibility Audit",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Date range: {payload['date_from']} to {payload['date_to']}",
        f"Canonical forecast phase: `{payload['canonical_forecast_phase']}`",
        f"Eligible verified-live backdates: **{payload['eligible_count']}**",
        f"Eligible dates: {', '.join(payload['eligible_dates']) or '-'}",
        f"Prop-promotion countable dates: {payload['prop_promotion_countable_count']} ({', '.join(payload['prop_promotion_countable_dates']) or '-'})",
        f"Prop close countable dates: {payload['prop_close_countable_count']} ({', '.join(payload['prop_close_countable_dates']) or '-'})",
        f"Forecast-ledger countable dates: {payload['forecast_countable_count']} ({', '.join(payload['forecast_countable_dates']) or '-'})",
        "",
        "A backdated slate counts only when locks, closes, forecasts, grading, and timing can be proven from immutable rows.",
        "",
        "## Slate Results",
        "",
        "| Date | Strict | Prop Promotion | Prop Close | Forecast | Games | Core Prop Rows | Forecast Rows | Prop Rows | Valid CLV | Stale | Missing Lock | True Pair | Blockers |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        if not (
            row.get("eligible_to_count")
            or _as_int(row.get("prop_training_rows")) > 0
            or _as_int(row.get("forecast_rows")) > 0
        ):
            continue
        games = f"{_as_int(row.get('settled_games'))}/{_as_int(row.get('games'))}"
        blockers = ", ".join(row.get("blockers") or []) or "-"
        lines.append(
            f"| {row['slate_date']} | {row['eligible_to_count']} | {row['prop_promotion_countable']} | "
            f"{row['prop_close_countable']} | {row['forecast_countable']} | {games} | "
            f"{_as_int(row.get('core_prop_forecasts'))} | {_as_int(row.get('forecast_rows'))} | {_as_int(row.get('prop_training_rows'))} | "
            f"{_fmt_pct(row.get('valid_clv_coverage'))} | {_fmt_pct(row.get('stale_close_rate'))} | "
            f"{_fmt_pct(row.get('missing_lock_rate'))} | {_fmt_pct(row.get('true_pair_rate'))} | {blockers} |"
        )
    lines.extend([
        "",
        "## Blocker Counts",
        "",
        "| Blocker | Dates |",
        "|---|---:|",
    ])
    for blocker, count in payload["blocker_counts"].items():
        lines.append(f"| {blocker} | {count} |")
    lines.extend([
        "",
        "## Closest Near Misses",
        "",
        "| Date | Prop Rows | Valid CLV | Stale | Missing Lock | Blockers |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in payload["near_misses"]:
        lines.append(
            f"| {row['slate_date']} | {_as_int(row.get('prop_training_rows'))} | "
            f"{_fmt_pct(row.get('valid_clv_coverage'))} | {_fmt_pct(row.get('stale_close_rate'))} | "
            f"{_fmt_pct(row.get('missing_lock_rate'))} | {', '.join(row.get('blockers') or [])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit backdated MLB prop slates for prospective-equivalent eligibility")
    parser.add_argument("--date-from", type=date.fromisoformat)
    parser.add_argument("--date-to", type=date.fromisoformat)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    args = parser.parse_args()
    today_et = datetime.now(_ET).date()
    date_to = args.date_to or (today_et - timedelta(days=1))
    date_from = args.date_from or (date_to - timedelta(days=44))
    payload = build(
        date_from=date_from,
        date_to=date_to,
        phase=args.phase,
        model_dir=Path(args.model_dir),
        pg_dsn=args.pg_dsn,
    )
    print(json.dumps({
        "eligible_count": payload["eligible_count"],
        "eligible_dates": payload["eligible_dates"],
        "prop_promotion_countable_dates": payload["prop_promotion_countable_dates"],
        "prop_close_countable_dates": payload["prop_close_countable_dates"],
        "forecast_countable_dates": payload["forecast_countable_dates"],
        "dates_with_prop_training": payload["dates_with_prop_training"],
        "dates_with_canonical_forecasts": payload["dates_with_canonical_forecasts"],
        "report_path": payload["report_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
