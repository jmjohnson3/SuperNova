"""Operational prop reports for real-money readiness.

These reports sit above the model/training artifacts.  They do not promote a
model or change betting decisions; they explain whether the live slate,
projection ledger, market evidence, shadow challengers, and micro buckets are
healthy enough to trust.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN as _PG_DSN
from mlb_pipeline.modeling.prop_micro_gate_sensitivity_report import build as build_micro_gate_sensitivity

_ET = ZoneInfo("America/New_York")
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _generated_at() -> str:
    return _now_utc().isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _table_exists(conn, qualified: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (qualified,))
        return cur.fetchone()[0] is not None


def _fetch_rows(conn, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    number = _as_float(value)
    return "-" if number is None else f"{number * 100:.{digits}f}%"


def _dt_label(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(_ET).strftime("%m-%d %I:%M %p ET").replace(" 0", " ")
    return str(value)


def _write_artifact(name: str, payload: dict[str, Any]) -> Path:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODEL_DIR / name
    atomic_write_json(path, payload, default=_json_default)
    return path


def _write_report(name: str, text: str) -> Path:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / name
    atomic_write_text(path, text if text.endswith("\n") else text + "\n")
    return path


def build_close_coverage_dashboard(
    *,
    conn,
    slate_date: date,
    fresh_minutes: int = 15,
    close_window_minutes: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": _generated_at(),
        "slate_date": slate_date.isoformat(),
        "status": "ready",
        "fresh_minutes": fresh_minutes,
        "close_window_minutes": close_window_minutes,
        "events": [],
    }
    if not _table_exists(conn, "odds.mlb_player_prop_line_snapshots"):
        payload["status"] = "missing_snapshots_table"
        return payload

    rows = _fetch_rows(
        conn,
        """
        WITH events AS (
            SELECT as_of_date,
                   event_id,
                   MAX(commence_time_utc) AS commence_time_utc,
                   MAX(home_team) AS home_team,
                   MAX(away_team) AS away_team
            FROM odds.mlb_player_prop_line_snapshots
            WHERE as_of_date = %s
              AND event_id IS NOT NULL
              AND commence_time_utc IS NOT NULL
            GROUP BY as_of_date, event_id
        ),
        role_counts AS (
            SELECT event_id,
                   COUNT(*) FILTER (WHERE snapshot_role = 'open')::int AS open_rows,
                   COUNT(*) FILTER (WHERE snapshot_role = 'lock')::int AS lock_rows,
                   COUNT(*) FILTER (WHERE snapshot_role = 'close')::int AS close_rows,
                   COUNT(*) FILTER (
                       WHERE snapshot_role = 'close'
                         AND snapshot_at_utc >= now() - (%s * interval '1 minute')
                   )::int AS fresh_close_rows,
                   COUNT(DISTINCT snapshot_at_utc) FILTER (WHERE snapshot_role = 'close')::int AS close_snapshot_times,
                   MAX(snapshot_at_utc) FILTER (WHERE snapshot_role = 'close') AS last_close_snapshot_at_utc
            FROM odds.mlb_player_prop_line_snapshots
            WHERE as_of_date = %s
            GROUP BY event_id
        ),
        valid AS (
            SELECT l.event_id,
                   COUNT(*)::int AS locked_offer_rows,
                   COUNT(*) FILTER (
                       WHERE EXISTS (
                           SELECT 1
                           FROM odds.mlb_player_prop_line_snapshots c
                           WHERE c.as_of_date = l.as_of_date
                             AND c.snapshot_role = 'close'
                             AND c.event_id = l.event_id
                             AND c.bookmaker_key = l.bookmaker_key
                             AND c.player_name_norm = l.player_name_norm
                             AND c.stat = l.stat
                             AND c.line = l.line
                             AND c.snapshot_at_utc > l.snapshot_at_utc
                             AND c.snapshot_at_utc >= l.commence_time_utc - interval '2 hours'
                             AND c.snapshot_at_utc <= l.commence_time_utc
                             AND (
                                 l.selected_side IS NULL
                                 OR (l.selected_side = 'over' AND c.over_price IS NOT NULL)
                                 OR (l.selected_side = 'under' AND c.under_price IS NOT NULL)
                             )
                       )
                   )::int AS projected_valid_close_rows
            FROM odds.mlb_player_prop_line_snapshots l
            WHERE l.as_of_date = %s
              AND l.snapshot_role = 'lock'
            GROUP BY l.event_id
        )
        SELECT e.event_id,
               e.commence_time_utc,
               e.home_team,
               e.away_team,
               COALESCE(r.open_rows, 0)::int AS open_rows,
               COALESCE(r.lock_rows, 0)::int AS lock_rows,
               COALESCE(r.close_rows, 0)::int AS close_rows,
               COALESCE(r.fresh_close_rows, 0)::int AS fresh_close_rows,
               COALESCE(r.close_snapshot_times, 0)::int AS close_snapshot_times,
               r.last_close_snapshot_at_utc,
               COALESCE(v.locked_offer_rows, 0)::int AS locked_offer_rows,
               COALESCE(v.projected_valid_close_rows, 0)::int AS projected_valid_close_rows
        FROM events e
        LEFT JOIN role_counts r ON r.event_id = e.event_id
        LEFT JOIN valid v ON v.event_id = e.event_id
        ORDER BY e.commence_time_utc, e.event_id
        """,
        (slate_date, fresh_minutes, slate_date, slate_date),
    )
    now = _now_utc()
    capture = _load_json(_MODEL_DIR / "targeted_prop_close_capture.json")
    capture_events = {str(row.get("event_id")) for row in capture.get("targets") or []}
    attempts = capture.get("attempts") or []
    retry_fired = int(capture.get("attempts_run") or len(attempts) or 0) > 1
    events: list[dict[str, Any]] = []
    for row in rows:
        commence = row.get("commence_time_utc")
        if isinstance(commence, datetime):
            if commence.tzinfo is None:
                commence = commence.replace(tzinfo=timezone.utc)
            commence = commence.astimezone(timezone.utc)
        locked = _as_int(row.get("locked_offer_rows"))
        valid = _as_int(row.get("projected_valid_close_rows"))
        active = bool(
            isinstance(commence, datetime)
            and now >= commence - timedelta(minutes=close_window_minutes)
            and now <= commence
        )
        coverage = (valid / locked) if locked else None
        events.append({
            **row,
            "commence_time_utc": commence,
            "inside_close_window": active,
            "projected_valid_close_coverage": coverage,
            "retry_fired": retry_fired and str(row.get("event_id")) in capture_events,
            "minutes_to_first_pitch": (
                (commence - now).total_seconds() / 60.0 if isinstance(commence, datetime) else None
            ),
        })
    active_events = [row for row in events if row["inside_close_window"]]
    locked_rows = sum(_as_int(row.get("locked_offer_rows")) for row in events)
    valid_rows = sum(_as_int(row.get("projected_valid_close_rows")) for row in events)
    payload.update({
        "targeted_capture_status": capture.get("status") or "unknown",
        "targeted_capture_attempts": int(capture.get("attempts_run") or len(attempts) or 0),
        "retry_fired": retry_fired,
        "event_count": len(events),
        "active_close_window_games": len(active_events),
        "locked_offer_rows": locked_rows,
        "projected_valid_close_rows": valid_rows,
        "projected_valid_close_coverage": (valid_rows / locked_rows) if locked_rows else None,
        "events": events,
    })
    return payload


def _render_close_dashboard(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Close Coverage Auto-Dashboard",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Slate: {payload.get('slate_date')}",
        f"Status: **{payload.get('status')}**",
        f"Games inside close window: {payload.get('active_close_window_games', 0)}",
        f"Projected valid-close coverage: {_pct(payload.get('projected_valid_close_coverage'))}",
        f"Targeted close capture: `{payload.get('targeted_capture_status')}`; attempts={payload.get('targeted_capture_attempts', 0)}; retry fired={payload.get('retry_fired', False)}",
        "",
        "| Game | First Pitch | Window | Last Close | Fresh Close Rows | Lock Rows | Projected Valid Close | Coverage | Retry |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("events") or []:
        game = f"{row.get('away_team') or '?'} @ {row.get('home_team') or '?'}"
        lines.append(
            f"| {game} | {_dt_label(row.get('commence_time_utc'))} | {bool(row.get('inside_close_window'))} | "
            f"{_dt_label(row.get('last_close_snapshot_at_utc'))} | {_as_int(row.get('fresh_close_rows'))} | "
            f"{_as_int(row.get('locked_offer_rows'))} | {_as_int(row.get('projected_valid_close_rows'))} | "
            f"{_pct(row.get('projected_valid_close_coverage'))} | {bool(row.get('retry_fired'))} |"
        )
    if not payload.get("events"):
        lines.append("| - | - | - | - | 0 | 0 | 0 | - | - |")
    return "\n".join(lines) + "\n"


def build_projection_ledger_v2(*, conn, lookback_days: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": _generated_at(),
        "lookback_days": lookback_days,
        "status": "ready",
        "summary": [],
        "recent_forecasts": [],
    }
    if not _table_exists(conn, "bets.mlb_daily_forecast_ledger"):
        payload["status"] = "missing_forecast_ledger"
        return payload
    cutoff = datetime.now(_ET).date() - timedelta(days=max(1, lookback_days))
    summary = _fetch_rows(
        conn,
        """
        SELECT forecast_type,
               stat,
               COALESCE(model_family, 'unknown') AS model_family,
               COALESCE(model_version, 'unknown') AS model_version,
               COUNT(*)::int AS rows,
               COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded_rows,
               AVG(absolute_error)::float AS mae,
               AVG(baseline_absolute_error)::float AS baseline_mae,
               AVG(absolute_error - baseline_absolute_error)::float AS mae_minus_baseline,
               AVG(projection_value - actual_value)::float AS bias
        FROM bets.mlb_daily_forecast_ledger
        WHERE game_date_et >= %s
          AND forecast_type IN ('player', 'opportunity')
        GROUP BY forecast_type, stat, model_family, model_version
        ORDER BY forecast_type, stat, rows DESC
        """,
        (cutoff,),
    )
    recent = _fetch_rows(
        conn,
        """
        SELECT game_date_et,
               game_slug,
               player_id,
               player_name,
               team_abbr,
               opponent_abbr,
               stat,
               forecast_type,
               projection_value::float AS projection_value,
               model_only_projection::float AS model_only_projection,
               baseline_value::float AS baseline_value,
               actual_value::float AS actual_value,
               absolute_error::float AS absolute_error,
               baseline_absolute_error::float AS baseline_absolute_error,
               model_family,
               model_version,
               baseline_source,
               result_status,
               locked_at_utc
        FROM bets.mlb_daily_forecast_ledger
        WHERE game_date_et >= %s
          AND forecast_type IN ('player', 'opportunity')
        ORDER BY game_date_et DESC, locked_at_utc DESC, stat, player_name
        LIMIT 80
        """,
        (cutoff,),
    )
    payload.update({
        "cutoff_date": cutoff.isoformat(),
        "summary": summary,
        "recent_forecasts": recent,
        "graded_rows": sum(_as_int(row.get("graded_rows")) for row in summary),
        "total_rows": sum(_as_int(row.get("rows")) for row in summary),
    })
    return payload


def _render_projection_ledger(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Player-Game Projection Ledger v2",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Lookback days: {payload.get('lookback_days')}",
        f"Rows: {payload.get('total_rows', 0)}; graded: {payload.get('graded_rows', 0)}",
        "",
        "## Projection Skill by Version",
        "",
        "| Type | Stat | Model | Version | Rows | Graded | MAE | Baseline MAE | MAE-Baseline | Bias |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary") or []:
        lines.append(
            f"| {row.get('forecast_type')} | {row.get('stat')} | {row.get('model_family')} | "
            f"{row.get('model_version')} | {_as_int(row.get('rows'))} | {_as_int(row.get('graded_rows'))} | "
            f"{_fmt(row.get('mae'))} | {_fmt(row.get('baseline_mae'))} | "
            f"{_fmt(row.get('mae_minus_baseline'))} | {_fmt(row.get('bias'))} |"
        )
    lines.extend([
        "",
        "## Recent Locked Player Forecasts",
        "",
        "| Date | Player | Stat | Projection | Baseline | Actual | Error | Version | Status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in (payload.get("recent_forecasts") or [])[:40]:
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('player_name') or row.get('player_id')} | "
            f"{row.get('stat')} | {_fmt(row.get('projection_value'))} | {_fmt(row.get('baseline_value'))} | "
            f"{_fmt(row.get('actual_value'))} | {_fmt(row.get('absolute_error'))} | "
            f"{row.get('model_version') or '-'} | {row.get('result_status')} |"
        )
    return "\n".join(lines) + "\n"


def build_tb_error_decomposition(*, conn, lookback_days: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": _generated_at(),
        "lookback_days": lookback_days,
        "status": "ready",
        "component_summary": {},
        "line_pricing": [],
        "worst_component_rows": [],
    }
    cutoff = datetime.now(_ET).date() - timedelta(days=max(1, lookback_days))
    if not _table_exists(conn, "features.mlb_hitter_player_game_training"):
        payload["status"] = "missing_hitter_player_game_training"
        return payload
    rows = _fetch_rows(
        conn,
        """
        SELECT game_date_et,
               game_slug,
               player_id,
               player_name,
               team_abbr,
               lineup_slot,
               confirmed_starter,
               projected_pa::float AS projected_pa,
               actual_pa::float AS actual_pa,
               model_pred_hits::float AS model_pred_hits,
               model_pred_total_bases::float AS model_pred_total_bases,
               model_pred_home_runs::float AS model_pred_home_runs,
               actual_hits::float AS actual_hits,
               actual_singles::float AS actual_singles,
               actual_doubles::float AS actual_doubles,
               actual_triples::float AS actual_triples,
               actual_home_runs::float AS actual_home_runs,
               actual_total_bases::float AS actual_total_bases
        FROM features.mlb_hitter_player_game_training
        WHERE game_date_et >= %s
          AND actual_total_bases IS NOT NULL
          AND model_pred_total_bases IS NOT NULL
        """,
        (cutoff,),
    )
    components: list[dict[str, Any]] = []
    for row in rows:
        pred_pa = _as_float(row.get("projected_pa"))
        actual_pa = _as_float(row.get("actual_pa"))
        pred_hits = _as_float(row.get("model_pred_hits"))
        pred_tb = _as_float(row.get("model_pred_total_bases"))
        pred_hr = _as_float(row.get("model_pred_home_runs"))
        actual_hits = _as_float(row.get("actual_hits"))
        actual_tb = _as_float(row.get("actual_total_bases"))
        actual_hr = _as_float(row.get("actual_home_runs"))
        if pred_pa is None or actual_pa is None or pred_hits is None or pred_tb is None or actual_hits is None or actual_tb is None:
            continue
        safe_pred_pa = max(pred_pa, 0.25)
        safe_actual_pa = max(actual_pa, 0.25)
        pred_hit_rate = pred_hits / safe_pred_pa
        actual_hit_rate = actual_hits / safe_actual_pa
        pred_tb_rate = pred_tb / safe_pred_pa
        actual_tb_rate = actual_tb / safe_actual_pa
        pred_extra_tb = max(0.0, pred_tb - pred_hits)
        actual_extra_tb = max(0.0, actual_tb - actual_hits)
        pred_hr_tail_tb = 4.0 * max(0.0, pred_hr or 0.0)
        actual_hr_tail_tb = 4.0 * max(0.0, actual_hr or 0.0)
        record = {
            **row,
            "tb_error": pred_tb - actual_tb,
            "abs_tb_error": abs(pred_tb - actual_tb),
            "pa_error": pred_pa - actual_pa,
            "pa_component_tb_error": pred_tb_rate * (pred_pa - actual_pa),
            "hit_probability_error": pred_hit_rate - actual_hit_rate,
            "hit_component_tb_error": actual_pa * (pred_hit_rate - actual_hit_rate),
            "mix_component_tb_error": pred_extra_tb - actual_extra_tb,
            "hr_tail_component_error": pred_hr_tail_tb - actual_hr_tail_tb,
            "non_hr_4plus_tail_flag": bool(actual_tb >= 4.0 and (actual_hr or 0.0) <= 0.0),
            "hr_driven_4plus_tail_flag": bool((actual_hr or 0.0) > 0.0),
        }
        abs_components = {
            "pa_error": abs(record["pa_component_tb_error"]),
            "hit_probability_error": abs(record["hit_component_tb_error"]),
            "single_double_triple_hr_mix_error": abs(record["mix_component_tb_error"]),
            "4plus_tb_tail_error": abs(record["hr_tail_component_error"]),
        }
        record["dominant_component"] = max(abs_components, key=abs_components.get)
        components.append(record)

    line_pricing: list[dict[str, Any]] = []
    if _table_exists(conn, "features.mlb_prop_market_training_examples"):
        offer_rows = _fetch_rows(
            conn,
            """
            SELECT side,
                   bookmaker_key,
                   line_bucket,
                   line_surface,
                   pair_quality,
                   true_pair_flag::float AS true_pair_flag,
                   synthetic_pair_flag::float AS synthetic_pair_flag,
                   COUNT(*)::int AS rows,
                   AVG(CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 ELSE NULL END)::float AS win_rate,
                   AVG(profit_units)::float AS roi,
                   AVG(POWER(model_prob_side - CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 ELSE NULL END, 2))::float AS brier,
                   COUNT(*) FILTER (WHERE clv_valid IS TRUE)::int AS clv_rows,
                   AVG(CASE WHEN beat_clv_price IS TRUE THEN 1.0 WHEN beat_clv_price IS FALSE THEN 0.0 ELSE NULL END)::float AS clv_beat_rate,
                   AVG(clv_price)::float AS avg_clv_price
            FROM features.mlb_prop_market_training_examples
            WHERE game_date_et >= %s
              AND market = 'batter_total_bases'
              AND result_status = 'graded'
            GROUP BY side, bookmaker_key, line_bucket, line_surface, pair_quality, true_pair_flag, synthetic_pair_flag
            HAVING COUNT(*) >= 5
            ORDER BY rows DESC
            LIMIT 80
            """,
            (cutoff,),
        )
        line_pricing = offer_rows

    dominant_counts: dict[str, int] = {}
    for row in components:
        key = str(row.get("dominant_component") or "unknown")
        dominant_counts[key] = dominant_counts.get(key, 0) + 1
    payload.update({
        "cutoff_date": cutoff.isoformat(),
        "player_game_rows": len(components),
        "component_summary": {
            "tb_mae": _mean([row["abs_tb_error"] for row in components]),
            "tb_bias": _mean([row["tb_error"] for row in components]),
            "pa_mae": _mean([abs(row["pa_error"]) for row in components]),
            "avg_abs_pa_component_tb_error": _mean([abs(row["pa_component_tb_error"]) for row in components]),
            "avg_abs_hit_component_tb_error": _mean([abs(row["hit_component_tb_error"]) for row in components]),
            "avg_abs_mix_component_tb_error": _mean([abs(row["mix_component_tb_error"]) for row in components]),
            "avg_abs_hr_tail_component_error": _mean([abs(row["hr_tail_component_error"]) for row in components]),
            "non_hr_4plus_tail_rows": sum(1 for row in components if row["non_hr_4plus_tail_flag"]),
            "hr_driven_4plus_tail_rows": sum(1 for row in components if row["hr_driven_4plus_tail_flag"]),
            "dominant_component_counts": dominant_counts,
        },
        "line_pricing": line_pricing,
        "worst_component_rows": sorted(components, key=lambda row: row["abs_tb_error"], reverse=True)[:40],
    })
    return payload


def _render_tb_error_decomposition(payload: dict[str, Any]) -> str:
    summary = payload.get("component_summary") or {}
    dominant = summary.get("dominant_component_counts") or {}
    lines = [
        "# MLB TB Error Decomposition",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Lookback days: {payload.get('lookback_days')}",
        f"Player-game rows: {payload.get('player_game_rows', 0)}",
        "",
        "## Component Summary",
        "",
        f"- TB MAE: {_fmt(summary.get('tb_mae'))}",
        f"- TB bias: {_fmt(summary.get('tb_bias'))}",
        f"- PA MAE: {_fmt(summary.get('pa_mae'))}",
        f"- Avg abs PA component TB error: {_fmt(summary.get('avg_abs_pa_component_tb_error'))}",
        f"- Avg abs hit probability component TB error: {_fmt(summary.get('avg_abs_hit_component_tb_error'))}",
        f"- Avg abs single/double/triple/HR mix component TB error: {_fmt(summary.get('avg_abs_mix_component_tb_error'))}",
        f"- Avg abs 4+ HR-tail component error: {_fmt(summary.get('avg_abs_hr_tail_component_error'))}",
        f"- HR-driven 4+ TB rows: {summary.get('hr_driven_4plus_tail_rows', 0)}",
        f"- Non-HR 4+ TB rows: {summary.get('non_hr_4plus_tail_rows', 0)}",
        "",
        "Dominant miss component counts: "
        + (", ".join(f"{key}={value}" for key, value in sorted(dominant.items())) or "-"),
        "",
        "## TB Line Pricing",
        "",
        "| Side | Book | Line | Surface | Pair | Rows | Win | ROI | Brier | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("line_pricing") or []:
        lines.append(
            f"| {row.get('side')} | {row.get('bookmaker_key')} | {row.get('line_bucket')} | "
            f"{row.get('line_surface')} | {row.get('pair_quality')} | {_as_int(row.get('rows'))} | "
            f"{_pct(row.get('win_rate'))} | {_fmt(row.get('roi'))} | {_fmt(row.get('brier'))} | "
            f"{_as_int(row.get('clv_rows'))} | {_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} |"
        )
    lines.extend([
        "",
        "## Worst Player-Game TB Misses",
        "",
        "| Date | Player | Slot | PA Err | Hit Err | Mix Err | Tail Err | TB Pred | TB Actual | Dominant |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("worst_component_rows") or []:
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('player_name') or row.get('player_id')} | "
            f"{_fmt(row.get('lineup_slot'), 0)} | {_fmt(row.get('pa_error'))} | "
            f"{_fmt(row.get('hit_component_tb_error'))} | {_fmt(row.get('mix_component_tb_error'))} | "
            f"{_fmt(row.get('hr_tail_component_error'))} | {_fmt(row.get('model_pred_total_bases'))} | "
            f"{_fmt(row.get('actual_total_bases'))} | {row.get('dominant_component')} |"
        )
    return "\n".join(lines) + "\n"


def build_shadow_challenger_promotion_report() -> dict[str, Any]:
    layer_control = _load_json(_MODEL_DIR / "prop_layer_promotion_control.json")
    checkpoint = _load_json(_MODEL_DIR / "prop_five_date_checkpoint.json")
    k_rate = _load_json(_MODEL_DIR / "pitcher_k_rate_challenger_report.json")
    hitter_rate = _load_json(_MODEL_DIR / "hitter_player_rate_diagnostic.json")
    outcome = _load_json(_MODEL_DIR / "hitter_player_game_outcome_models.json")
    tb_tail = _load_json(_MODEL_DIR / "prop_tb_tail_repair_challenger.json")
    projection_audit = _load_json(_MODEL_DIR / "daily_forecast_projection_audit.json")
    layers = layer_control.get("layers") or {}
    rows: list[dict[str, Any]] = []

    def add_layer(name: str, title: str, artifact: dict[str, Any], blockers: list[str] | None = None) -> None:
        layer = layers.get(name) or {}
        dates = (
            artifact.get("dates")
            or artifact.get("completed_date_count")
            or artifact.get("unique_dates")
            or layer.get("graded_dates")
            or layer.get("completed_dates")
            or 0
        )
        accepted = bool(
            artifact.get("accepted")
            or (artifact.get("challenger_meta") or {}).get("accepted")
            or layer.get("can_auto_integrate")
        )
        mae = (artifact.get("challenger") or {}).get("mae")
        baseline_mae = (artifact.get("baseline") or {}).get("mae")
        brier = ((artifact.get("line_brier") or {}).get("challenger") or {}).get("brier")
        baseline_brier = ((artifact.get("line_brier") or {}).get("baseline") or {}).get("brier")
        if artifact.get("tb_count"):
            mae = ((artifact.get("tb_count") or {}).get("direct_state_expected") or {}).get("mae")
            baseline_mae = ((artifact.get("tb_count") or {}).get("baseline") or {}).get("mae")
            pricing = artifact.get("true_pair_line_pricing") or {}
            brier = pricing.get("selected_blend_brier", pricing.get("calibrated_brier"))
            baseline_brier = pricing.get("baseline_brier")
        row_blockers = list(blockers or [])
        row_blockers.extend(str(value) for value in (layer.get("blockers") or []) if value)
        if not artifact:
            row_blockers.append("artifact_missing")
        if not accepted:
            row_blockers.append("challenger_not_accepted")
        rows.append({
            "layer": name,
            "title": title,
            "status": artifact.get("status") or layer.get("status") or "unknown",
            "current_mode": layer.get("current_mode") or "shadow",
            "target_mode": layer.get("target_mode") or "production",
            "completed_dates": _as_int(dates),
            "rows": _as_int(artifact.get("rows") or artifact.get("evaluation_rows") or 0),
            "mae": mae,
            "baseline_mae": baseline_mae,
            "mae_gain": artifact.get("mae_gain"),
            "brier": brier,
            "baseline_brier": baseline_brier,
            "brier_gain": artifact.get("line_brier_gain"),
            "calibration": artifact.get("calibration_error") or artifact.get("calibration"),
            "passes_promotion": bool(accepted and not row_blockers),
            "blockers": sorted(set(row_blockers)),
        })

    add_layer("hitter_projection", "Hitter opportunity / rates", hitter_rate)
    add_layer("hitter_tb_distribution", "TB conditional/XBH distribution", outcome)
    add_layer("tb_tail_repair", "TB 4+ tail / line calibration challenger", tb_tail)
    add_layer("pitcher_projection", "Pitcher opportunity", projection_audit)
    add_layer("pitcher_k_rate", "Pitcher K-rate conservative challenger", k_rate)
    for name, layer in layers.items():
        if name not in {row["layer"] for row in rows}:
            rows.append({
                "layer": name,
                "title": layer.get("title") or name,
                "status": layer.get("status") or "unknown",
                "current_mode": layer.get("current_mode"),
                "target_mode": layer.get("target_mode"),
                "completed_dates": _as_int(layer.get("graded_dates") or layer.get("completed_dates") or 0),
                "rows": _as_int(layer.get("rows") or layer.get("graded_rows") or 0),
                "passes_promotion": bool(layer.get("can_auto_integrate")),
                "blockers": list(layer.get("blockers") or []),
            })
    return {
        "generated_at_utc": _generated_at(),
        "status": "ready",
        "layer_control_status": layer_control.get("status") or "missing",
        "checkpoint_status": checkpoint.get("status") or "missing",
        "minimum_completed_dates": checkpoint.get("minimum_completed_dates") or 5,
        "shadow_layers": rows,
    }


def _render_shadow_challenger(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Shadow Challenger Promotion Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Layer control: `{payload.get('layer_control_status')}`",
        f"Minimum completed dates: {payload.get('minimum_completed_dates')}",
        "",
        "| Layer | Mode | Dates | Rows | MAE | Baseline MAE | Brier | Baseline Brier | Promotion | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload.get("shadow_layers") or []:
        lines.append(
            f"| {row.get('title')} | {row.get('current_mode')} -> {row.get('target_mode')} | "
            f"{_as_int(row.get('completed_dates'))} | {_as_int(row.get('rows'))} | "
            f"{_fmt(row.get('mae'))} | {_fmt(row.get('baseline_mae'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('baseline_brier'))} | "
            f"{bool(row.get('passes_promotion'))} | {', '.join((row.get('blockers') or [])[:8]) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def build_true_pair_coverage_repair(*, conn, lookback_days: int) -> dict[str, Any]:
    payload = {
        "generated_at_utc": _generated_at(),
        "lookback_days": lookback_days,
        "status": "ready",
        "coverage": [],
    }
    if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
        payload["status"] = "missing_prop_market_training_examples"
        return payload
    cutoff = datetime.now(_ET).date() - timedelta(days=max(1, lookback_days))
    rows = _fetch_rows(
        conn,
        """
        SELECT game_date_et,
               market,
               side,
               bookmaker_key,
               line_bucket,
               line_surface,
               COUNT(*)::int AS rows,
               AVG(COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book', 'cross_book') THEN 1.0 ELSE 0.0 END))::float AS true_pair_rate,
               AVG(COALESCE(same_book_pair_flag::float, CASE WHEN pair_quality = 'same_book' THEN 1.0 ELSE 0.0 END))::float AS same_book_pair_rate,
               AVG(COALESCE(cross_book_pair_flag::float, CASE WHEN pair_quality = 'cross_book' THEN 1.0 ELSE 0.0 END))::float AS cross_book_pair_rate,
               AVG(COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END))::float AS synthetic_pair_rate,
               AVG(CASE WHEN market_prob_source IN ('raw_implied_one_sided', 'one_sided_fanduel_ladder') THEN 1.0 ELSE 0.0 END)::float AS one_sided_market_rate,
               COUNT(*) FILTER (WHERE clv_valid IS TRUE)::int AS clv_rows
        FROM features.mlb_prop_market_training_examples
        WHERE game_date_et >= %s
        GROUP BY game_date_et, market, side, bookmaker_key, line_bucket, line_surface
        HAVING COUNT(*) >= 5
        ORDER BY game_date_et DESC, market, bookmaker_key, rows DESC
        """,
        (cutoff,),
    )
    enriched = []
    for row in rows:
        actions: list[str] = []
        true_rate = _as_float(row.get("true_pair_rate")) or 0.0
        synthetic_rate = _as_float(row.get("synthetic_pair_rate")) or 0.0
        one_sided_rate = _as_float(row.get("one_sided_market_rate")) or 0.0
        same_book_rate = _as_float(row.get("same_book_pair_rate")) or 0.0
        book = str(row.get("bookmaker_key") or "").lower()
        market = str(row.get("market") or "")
        if true_rate < 0.70:
            actions.append("repair_true_pair_coverage_to_70pct")
        if same_book_rate < 0.50:
            actions.append("improve_same_book_pairing")
        if synthetic_rate > 0.05:
            actions.append("demote_synthetic_from_training_and_promotion")
        if one_sided_rate > 0.10:
            actions.append("extract_true_opposite_or_keep_display_only")
        if book == "fanduel" and market.startswith("batter_") and synthetic_rate > 0:
            actions.append("fanduel_hitter_synthetic_display_only")
        enriched.append({**row, "repair_actions": sorted(set(actions))})
    payload.update({
        "cutoff_date": cutoff.isoformat(),
        "coverage": enriched,
        "below_70_true_pair_groups": sum(1 for row in enriched if (_as_float(row.get("true_pair_rate")) or 0.0) < 0.70),
        "synthetic_groups": sum(1 for row in enriched if (_as_float(row.get("synthetic_pair_rate")) or 0.0) > 0.05),
    })
    return payload


def _render_true_pair(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB True-Pair Coverage Repair Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Lookback days: {payload.get('lookback_days')}",
        f"Groups below 70% true-pair coverage: {payload.get('below_70_true_pair_groups', 0)}",
        f"Groups with synthetic evidence: {payload.get('synthetic_groups', 0)}",
        "",
        "| Date | Market | Side | Book | Line | Rows | True Pair | Same Book | Synthetic | One-Sided | Actions |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in (payload.get("coverage") or [])[:120]:
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('market')} | {row.get('side')} | "
            f"{row.get('bookmaker_key')} | {row.get('line_bucket')} | {_as_int(row.get('rows'))} | "
            f"{_pct(row.get('true_pair_rate'))} | {_pct(row.get('same_book_pair_rate'))} | "
            f"{_pct(row.get('synthetic_pair_rate'))} | {_pct(row.get('one_sided_market_rate'))} | "
            f"{', '.join(row.get('repair_actions') or []) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def build_micro_bucket_almost_there() -> dict[str, Any]:
    micro = _load_json(_MODEL_DIR / "prop_micro_promotion_evaluation.json")
    rows = []
    for row in micro.get("buckets") or []:
        clean_needed = max(0, 5 - _as_int(row.get("promotion_clean_dates") or row.get("clean_dates")))
        rows_needed = max(0, 150 - _as_int(row.get("rows")))
        clv_needed = max(0, 30 - _as_int(row.get("clv_rows")))
        clv_beat = _as_float(row.get("clv_beat_rate"))
        clv_short = max(0.0, 0.55 - clv_beat) if clv_beat is not None else 0.55
        avg_clv = _as_float(row.get("avg_clv_price"))
        avg_clv_short = max(0.0, 0.0 - avg_clv) if avg_clv is not None else 1.0
        roi = _as_float(row.get("roi"))
        roi_short = max(0.0, 0.0 - roi) if roi is not None else 1.0
        cal = _as_float(row.get("calibration_error"))
        calibration_short = max(0.0, abs(cal) - 0.05) if cal is not None else 0.05
        coverage = _as_float(row.get("valid_close_coverage"))
        coverage_short = max(0.0, 0.90 - coverage) if coverage is not None else 0.90
        true_pair_needed = 0 if _as_int(row.get("true_pair_rows")) > 0 else 1
        blockers = list(row.get("blockers") or [])
        fixability_score = (
            clean_needed * 8
            + rows_needed / 15.0
            + clv_needed / 6.0
            + clv_short * 100
            + avg_clv_short * 0.5
            + roi_short * 10
            + calibration_short * 100
            + coverage_short * 100
            + true_pair_needed * 25
            + len(blockers) * 2
        )
        notes = []
        if clean_needed:
            notes.append(f"needs {clean_needed} clean date{'s' if clean_needed != 1 else ''}")
        if rows_needed:
            notes.append(f"needs {rows_needed} rows")
        if clv_needed:
            notes.append(f"needs {clv_needed} valid CLV rows")
        if clv_short > 0:
            notes.append(f"CLV beat short {_pct(clv_short, 1)}")
        if avg_clv_short > 0:
            notes.append("avg CLV not positive")
        if roi_short > 0:
            notes.append("ROI not positive")
        if calibration_short > 0:
            notes.append(f"calibration over by {_pct(calibration_short, 1)}")
        if coverage_short > 0:
            notes.append(f"valid close short {_pct(coverage_short, 1)}")
        if true_pair_needed:
            notes.append("needs true-pair proof")
        rows.append({
            **row,
            "fixability_score": fixability_score,
            "rows_needed": rows_needed,
            "clean_dates_needed": clean_needed,
            "clv_rows_needed": clv_needed,
            "clv_beat_short": clv_short,
            "avg_clv_short": avg_clv_short,
            "roi_short": roi_short,
            "calibration_short": calibration_short,
            "valid_close_coverage_short": coverage_short,
            "true_pair_needed": true_pair_needed,
            "almost_there_notes": notes,
        })
    rows.sort(key=lambda row: (
        bool(row.get("micro_ready")) is False,
        float(row.get("fixability_score") or 9999.0),
        -_as_int(row.get("rows")),
    ))
    return {
        "generated_at_utc": _generated_at(),
        "status": "ready" if micro else "missing_micro_promotion_evaluation",
        "micro_ready_count": micro.get("micro_ready_count", 0),
        "source_generated_at_utc": micro.get("generated_at_utc"),
        "closest_buckets": rows[:80],
    }


def _render_micro_almost(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Micro Bucket Almost-There Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Micro-ready buckets: {payload.get('micro_ready_count', 0)}",
        "",
        "| Bucket | Rows | Clean | ROI | CLV Rows | CLV Beat | Avg CLV | Cal Err | Coverage | Score | What It Needs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("closest_buckets") or []:
        lines.append(
            f"| {row.get('bucket')} | {_as_int(row.get('rows'))} | "
            f"{_as_int(row.get('promotion_clean_dates') or row.get('clean_dates'))} | {_fmt(row.get('roi'))} | "
            f"{_as_int(row.get('clv_rows'))} | {_pct(row.get('clv_beat_rate'))} | "
            f"{_fmt(row.get('avg_clv_price'))} | {_pct(row.get('calibration_error'))} | "
            f"{_pct(row.get('valid_close_coverage'))} | {_fmt(row.get('fixability_score'), 1)} | "
            f"{'; '.join((row.get('almost_there_notes') or [])[:6]) or 'passes listed gaps'} |"
        )
    return "\n".join(lines) + "\n"


def run_reports(
    *,
    pg_dsn: str,
    slate_date: date,
    lookback_days: int,
    reports: set[str],
) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "generated_at_utc": _generated_at(),
        "slate_date": slate_date.isoformat(),
        "lookback_days": lookback_days,
        "reports": {},
    }
    with psycopg2.connect(pg_dsn) as conn:
        if "close" in reports:
            payload = build_close_coverage_dashboard(conn=conn, slate_date=slate_date)
            _write_artifact("prop_close_coverage_dashboard.json", payload)
            path = _write_report("mlb_prop_close_coverage_dashboard_latest.md", _render_close_dashboard(payload))
            outputs["reports"]["close_coverage_dashboard"] = {"status": payload.get("status"), "report_path": str(path)}
        if "ledger" in reports:
            payload = build_projection_ledger_v2(conn=conn, lookback_days=lookback_days)
            _write_artifact("player_game_projection_ledger_v2.json", payload)
            path = _write_report("mlb_player_game_projection_ledger_v2_latest.md", _render_projection_ledger(payload))
            outputs["reports"]["player_game_projection_ledger_v2"] = {"status": payload.get("status"), "report_path": str(path)}
        if "tb" in reports:
            payload = build_tb_error_decomposition(conn=conn, lookback_days=lookback_days)
            _write_artifact("tb_error_decomposition_report.json", payload)
            path = _write_report("mlb_tb_error_decomposition_latest.md", _render_tb_error_decomposition(payload))
            outputs["reports"]["tb_error_decomposition"] = {"status": payload.get("status"), "report_path": str(path)}
        if "pair" in reports:
            payload = build_true_pair_coverage_repair(conn=conn, lookback_days=lookback_days)
            _write_artifact("true_pair_coverage_repair_report.json", payload)
            path = _write_report("mlb_true_pair_coverage_repair_latest.md", _render_true_pair(payload))
            outputs["reports"]["true_pair_coverage_repair"] = {"status": payload.get("status"), "report_path": str(path)}
    if "shadow" in reports:
        payload = build_shadow_challenger_promotion_report()
        _write_artifact("shadow_challenger_promotion_report.json", payload)
        path = _write_report("mlb_shadow_challenger_promotion_latest.md", _render_shadow_challenger(payload))
        outputs["reports"]["shadow_challenger_promotion"] = {"status": payload.get("status"), "report_path": str(path)}
    if "micro" in reports:
        payload = build_micro_bucket_almost_there()
        _write_artifact("micro_bucket_almost_there_report.json", payload)
        path = _write_report("mlb_micro_bucket_almost_there_latest.md", _render_micro_almost(payload))
        outputs["reports"]["micro_bucket_almost_there"] = {"status": payload.get("status"), "report_path": str(path)}
    if "sensitivity" in reports:
        payload = build_micro_gate_sensitivity(_MODEL_DIR)
        outputs["reports"]["micro_gate_sensitivity"] = {
            "status": payload.get("status"),
            "report_path": str(payload.get("report_path")),
        }
    _write_artifact("prop_real_money_operational_reports.json", outputs)
    return outputs


def _parse_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(_ET).date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop operational real-money readiness reports")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None)
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument(
        "--report",
        action="append",
        choices=("all", "close", "ledger", "tb", "shadow", "pair", "micro", "sensitivity"),
        default=None,
        help="Report to build; repeatable. Default is all.",
    )
    args = parser.parse_args()
    requested = set(args.report or ["all"])
    if "all" in requested:
        requested = {"close", "ledger", "tb", "shadow", "pair", "micro", "sensitivity"}
    payload = run_reports(
        pg_dsn=args.pg_dsn,
        slate_date=_parse_date(args.date),
        lookback_days=args.lookback_days,
        reports=requested,
    )
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
