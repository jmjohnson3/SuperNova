"""Report $1 micro-projection prop results separately from paper/research rows."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

_ET = ZoneInfo("America/New_York")
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_STARTER_MIN_GRADED = 50
_BANKROLL_MIN_GRADED = 150
_MIN_CLV_ROWS = 30
_STARTER_MIN_DATES = 5
_BANKROLL_MIN_DATES = 10
_MIN_ROI = 0.0
_MIN_CLV_BEAT = 0.55
_MIN_AVG_CLV = 0.0
_MAX_ABS_CALIBRATION_ERROR = 0.05
_MAX_PLAYER_SHARE = 0.25
_MAX_TEAM_SHARE = 0.35


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v * 100.0:+.1f}%" if signed else f"{v * 100.0:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _micro_where() -> str:
    return """
      source = 'prop'
      AND (
        LOWER(COALESCE(model_tier, '')) = 'micro_projection'
        OR LOWER(COALESCE(model_meta->>'selector_tier', '')) = 'micro_projection'
      )
    """


def _summary_sql(group_cols: list[str]) -> str:
    select_cols = ", ".join(group_cols) + "," if group_cols else ""
    group_by = "GROUP BY " + ", ".join(group_cols) if group_cols else ""
    order_by = "ORDER BY graded DESC, pending DESC" if group_cols else ""
    return f"""
        WITH rows AS (
            SELECT
                *,
                CASE WHEN result_status = 'graded' AND COALESCE(push, false) IS FALSE
                     THEN CASE WHEN won IS TRUE THEN 1.0 ELSE 0.0 END
                     ELSE NULL END AS y_win,
                COALESCE(NULLIF(stake_usd, 0), 1.0)::float AS stake_basis
            FROM bets.mlb_model_pick_ledger
            WHERE {_micro_where()}
              AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
        )
        SELECT
            {select_cols}
            COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded,
            COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending,
            COUNT(*) FILTER (WHERE result_status = 'graded' AND won IS TRUE)::int AS wins,
            COUNT(*) FILTER (
                WHERE result_status = 'graded'
                  AND won IS FALSE
                  AND COALESCE(push, false) IS FALSE
            )::int AS losses,
            COUNT(*) FILTER (WHERE result_status = 'graded' AND COALESCE(push, false) IS TRUE)::int AS pushes,
            COALESCE(SUM(profit_units) FILTER (WHERE result_status = 'graded'), 0)::float AS units,
            COALESCE(SUM(profit_units * stake_basis) FILTER (WHERE result_status = 'graded'), 0)::float AS profit_usd_est,
            COALESCE(SUM(stake_basis) FILTER (WHERE result_status = 'graded'), 0)::float AS graded_stake_usd,
            AVG(model_prob::float) FILTER (WHERE y_win IS NOT NULL AND model_prob IS NOT NULL)::float AS avg_model_prob,
            AVG(y_win) FILTER (WHERE y_win IS NOT NULL)::float AS win_rate,
            AVG(POWER(model_prob::float - y_win, 2)) FILTER (WHERE y_win IS NOT NULL AND model_prob IS NOT NULL)::float AS brier,
            COUNT(*) FILTER (WHERE result_status = 'graded' AND clv_valid IS TRUE)::int AS clv_rows,
            AVG(CASE WHEN clv_valid IS TRUE AND clv_price > 0 THEN 1.0 WHEN clv_valid IS TRUE THEN 0.0 ELSE NULL END)::float AS clv_beat_rate,
            AVG(clv_price) FILTER (WHERE result_status = 'graded' AND clv_valid IS TRUE)::float AS avg_clv_price
        FROM rows
        {group_by}
        {order_by}
    """


def _exact_bucket_sql() -> str:
    price_bucket_expr = """
        COALESCE(NULLIF(model_meta->>'price_bucket', ''), CASE
            WHEN market_price IS NULL THEN 'missing_price'
            WHEN market_price > 0 AND market_price < 150 THEN 'plus_100_149'
            WHEN market_price > 0 AND market_price < 250 THEN 'plus_150_249'
            WHEN market_price > 0 AND market_price < 500 THEN 'plus_250_499'
            WHEN market_price > 0 THEN 'plus_500_plus'
            WHEN market_price >= -129 THEN 'fair_lay'
            WHEN market_price >= -149 THEN 'lay_130_149'
            WHEN market_price >= -180 THEN 'lay_150_180'
            ELSE 'heavy_lay'
        END)
    """
    line_bucket_expr = """
        COALESCE(NULLIF(model_meta->>'line_bucket', ''), CASE
            WHEN market = 'pitcher_strikeouts' AND market_line < 4.5 THEN 'K <4.5'
            WHEN market = 'pitcher_strikeouts' AND market_line < 6.5 THEN 'K 4.5-6.0'
            WHEN market = 'pitcher_strikeouts' AND market_line < 8.5 THEN 'K 6.5-8.0'
            WHEN market = 'pitcher_strikeouts' THEN 'K 8.5+'
            WHEN market = 'batter_total_bases' AND market_line < 1.0 THEN 'TB 0.5'
            WHEN market = 'batter_total_bases' AND market_line < 2.0 THEN 'TB 1.5'
            WHEN market = 'batter_total_bases' AND market_line < 3.0 THEN 'TB 2.5'
            WHEN market = 'batter_total_bases' AND market_line < 4.0 THEN 'TB 3.5'
            WHEN market = 'batter_total_bases' THEN 'TB 4.5+'
            WHEN market = 'batter_hits' AND market_line < 1.0 THEN 'H 0.5'
            WHEN market = 'batter_hits' AND market_line < 2.0 THEN 'H 1.5'
            WHEN market = 'batter_hits' AND market_line < 3.0 THEN 'H 2.5'
            WHEN market = 'batter_hits' THEN 'H 3.5+'
            WHEN market = 'batter_home_runs' AND market_line < 1.0 THEN 'HR 0.5'
            WHEN market = 'batter_home_runs' THEN 'HR 1.5+'
            ELSE 'other'
        END)
    """
    return f"""
        WITH rows AS (
            SELECT
                *,
                CASE WHEN result_status = 'graded' AND COALESCE(push, false) IS FALSE
                     THEN CASE WHEN won IS TRUE THEN 1.0 ELSE 0.0 END
                     ELSE NULL END AS y_win,
                COALESCE(NULLIF(stake_usd, 0), 1.0)::float AS stake_basis
            FROM bets.mlb_model_pick_ledger
            WHERE {_micro_where()}
              AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
        ), enriched AS (
            SELECT
                *,
                {line_bucket_expr} AS line_bucket_group,
                {price_bucket_expr} AS price_bucket_group
            FROM rows
        )
        SELECT
            market,
            side,
            COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
            line_bucket_group AS line_bucket,
            price_bucket_group AS price_bucket,
            COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded,
            COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending,
            COUNT(*) FILTER (WHERE result_status = 'graded' AND won IS TRUE)::int AS wins,
            COUNT(*) FILTER (
                WHERE result_status = 'graded'
                  AND won IS FALSE
                  AND COALESCE(push, false) IS FALSE
            )::int AS losses,
            COUNT(*) FILTER (WHERE result_status = 'graded' AND COALESCE(push, false) IS TRUE)::int AS pushes,
            COALESCE(SUM(profit_units) FILTER (WHERE result_status = 'graded'), 0)::float AS units,
            COALESCE(SUM(profit_units * stake_basis) FILTER (WHERE result_status = 'graded'), 0)::float AS profit_usd_est,
            COALESCE(SUM(stake_basis) FILTER (WHERE result_status = 'graded'), 0)::float AS graded_stake_usd,
            AVG(model_prob::float) FILTER (WHERE y_win IS NOT NULL AND model_prob IS NOT NULL)::float AS avg_model_prob,
            AVG(y_win) FILTER (WHERE y_win IS NOT NULL)::float AS win_rate,
            AVG(POWER(model_prob::float - y_win, 2)) FILTER (WHERE y_win IS NOT NULL AND model_prob IS NOT NULL)::float AS brier,
            COUNT(*) FILTER (WHERE result_status = 'graded' AND clv_valid IS TRUE)::int AS clv_rows,
            AVG(CASE WHEN clv_valid IS TRUE AND clv_price > 0 THEN 1.0 WHEN clv_valid IS TRUE THEN 0.0 ELSE NULL END)::float AS clv_beat_rate,
            AVG(clv_price) FILTER (WHERE result_status = 'graded' AND clv_valid IS TRUE)::float AS avg_clv_price
        FROM enriched
        GROUP BY market, side, bookmaker_key, line_bucket_group, price_bucket_group
        ORDER BY graded DESC, pending DESC
    """


def _ladder_sql() -> str:
    price_bucket_expr = """
        COALESCE(NULLIF(model_meta->>'price_bucket', ''), CASE
            WHEN market_price IS NULL THEN 'missing_price'
            WHEN market_price > 0 AND market_price < 150 THEN 'plus_100_149'
            WHEN market_price > 0 AND market_price < 250 THEN 'plus_150_249'
            WHEN market_price > 0 AND market_price < 500 THEN 'plus_250_499'
            WHEN market_price > 0 THEN 'plus_500_plus'
            WHEN market_price >= -129 THEN 'fair_lay'
            WHEN market_price >= -149 THEN 'lay_130_149'
            WHEN market_price >= -180 THEN 'lay_150_180'
            ELSE 'heavy_lay'
        END)
    """
    line_bucket_expr = """
        COALESCE(NULLIF(model_meta->>'line_bucket', ''), CASE
            WHEN market = 'pitcher_strikeouts' AND market_line < 4.5 THEN 'K <4.5'
            WHEN market = 'pitcher_strikeouts' AND market_line < 6.5 THEN 'K 4.5-6.0'
            WHEN market = 'pitcher_strikeouts' AND market_line < 8.5 THEN 'K 6.5-8.0'
            WHEN market = 'pitcher_strikeouts' THEN 'K 8.5+'
            WHEN market = 'batter_total_bases' AND market_line < 1.0 THEN 'TB 0.5'
            WHEN market = 'batter_total_bases' AND market_line < 2.0 THEN 'TB 1.5'
            WHEN market = 'batter_total_bases' AND market_line < 3.0 THEN 'TB 2.5'
            WHEN market = 'batter_total_bases' AND market_line < 4.0 THEN 'TB 3.5'
            WHEN market = 'batter_total_bases' THEN 'TB 4.5+'
            WHEN market = 'batter_hits' AND market_line < 1.0 THEN 'H 0.5'
            WHEN market = 'batter_hits' AND market_line < 2.0 THEN 'H 1.5'
            WHEN market = 'batter_hits' AND market_line < 3.0 THEN 'H 2.5'
            WHEN market = 'batter_hits' THEN 'H 3.5+'
            WHEN market = 'batter_home_runs' AND market_line < 1.0 THEN 'HR 0.5'
            WHEN market = 'batter_home_runs' THEN 'HR 1.5+'
            ELSE 'other'
        END)
    """
    return f"""
        WITH rows AS (
            SELECT
                *,
                CASE WHEN result_status = 'graded' AND COALESCE(push, false) IS FALSE
                     THEN CASE WHEN won IS TRUE THEN 1.0 ELSE 0.0 END
                     ELSE NULL END AS y_win,
                COALESCE(NULLIF(stake_usd, 0), 1.0)::float AS stake_basis
            FROM bets.mlb_model_pick_ledger
            WHERE {_micro_where()}
              AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
        ), enriched AS (
            SELECT
                *,
                {line_bucket_expr} AS line_bucket_group,
                {price_bucket_expr} AS price_bucket_group,
                COALESCE(player_id::text, NULLIF(player_name_norm, ''), NULLIF(player_name, ''), 'unknown') AS player_key,
                COALESCE(NULLIF(team_abbr, ''), 'unknown') AS team_key
            FROM rows
        ), grouped AS (
            SELECT
                market,
                side,
                COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
                line_bucket_group AS line_bucket,
                price_bucket_group AS price_bucket,
                COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded,
                COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending,
                COUNT(DISTINCT game_date_et) FILTER (WHERE result_status = 'graded')::int AS graded_dates,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND won IS TRUE)::int AS wins,
                COUNT(*) FILTER (
                    WHERE result_status = 'graded'
                      AND won IS FALSE
                      AND COALESCE(push, false) IS FALSE
                )::int AS losses,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND COALESCE(push, false) IS TRUE)::int AS pushes,
                COALESCE(SUM(profit_units * stake_basis) FILTER (WHERE result_status = 'graded'), 0)::float AS profit_usd_est,
                COALESCE(SUM(stake_basis) FILTER (WHERE result_status = 'graded'), 0)::float AS graded_stake_usd,
                AVG(model_prob::float) FILTER (WHERE y_win IS NOT NULL AND model_prob IS NOT NULL)::float AS avg_model_prob,
                AVG(y_win) FILTER (WHERE y_win IS NOT NULL)::float AS win_rate,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND clv_valid IS TRUE)::int AS clv_rows,
                AVG(CASE WHEN clv_valid IS TRUE AND clv_price > 0 THEN 1.0 WHEN clv_valid IS TRUE THEN 0.0 ELSE NULL END)::float AS clv_beat_rate,
                AVG(clv_price) FILTER (WHERE result_status = 'graded' AND clv_valid IS TRUE)::float AS avg_clv_price
            FROM enriched
            GROUP BY market, side, bookmaker_key, line_bucket_group, price_bucket_group
        ), player_counts AS (
            SELECT
                market,
                side,
                COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
                line_bucket_group AS line_bucket,
                price_bucket_group AS price_bucket,
                player_key,
                COUNT(*) FILTER (WHERE result_status = 'graded')::int AS player_graded
            FROM enriched
            GROUP BY market, side, bookmaker_key, line_bucket_group, price_bucket_group, player_key
        ), team_counts AS (
            SELECT
                market,
                side,
                COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
                line_bucket_group AS line_bucket,
                price_bucket_group AS price_bucket,
                team_key,
                COUNT(*) FILTER (WHERE result_status = 'graded')::int AS team_graded
            FROM enriched
            GROUP BY market, side, bookmaker_key, line_bucket_group, price_bucket_group, team_key
        )
        SELECT
            g.*,
            (
                SELECT MAX(player_graded)::float / NULLIF(g.graded, 0)
                FROM player_counts pc
                WHERE pc.market = g.market
                  AND pc.side = g.side
                  AND pc.bookmaker_key = g.bookmaker_key
                  AND pc.line_bucket = g.line_bucket
                  AND pc.price_bucket = g.price_bucket
            )::float AS max_player_share,
            (
                SELECT MAX(team_graded)::float / NULLIF(g.graded, 0)
                FROM team_counts tc
                WHERE tc.market = g.market
                  AND tc.side = g.side
                  AND tc.bookmaker_key = g.bookmaker_key
                  AND tc.line_bucket = g.line_bucket
                  AND tc.price_bucket = g.price_bucket
            )::float AS max_team_share
        FROM grouped g
        ORDER BY graded DESC, pending DESC
    """


def _recent_bets_sql() -> str:
    return f"""
        SELECT
            game_date_et,
            inserted_at_utc,
            player_name,
            team_abbr,
            market,
            side,
            COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
            market_line,
            market_price,
            minimum_acceptable_price,
            model_prob,
            ev,
            result_status,
            won,
            push,
            actual_value,
            profit_units,
            clv_valid,
            clv_price,
            closing_price,
            COALESCE(NULLIF(model_meta->>'line_bucket', ''), CASE
                WHEN market = 'pitcher_strikeouts' AND market_line < 4.5 THEN 'K <4.5'
                WHEN market = 'pitcher_strikeouts' AND market_line < 6.5 THEN 'K 4.5-6.0'
                WHEN market = 'pitcher_strikeouts' AND market_line < 8.5 THEN 'K 6.5-8.0'
                WHEN market = 'pitcher_strikeouts' THEN 'K 8.5+'
                WHEN market = 'batter_total_bases' AND market_line < 1.0 THEN 'TB 0.5'
                WHEN market = 'batter_total_bases' AND market_line < 2.0 THEN 'TB 1.5'
                WHEN market = 'batter_total_bases' AND market_line < 3.0 THEN 'TB 2.5'
                WHEN market = 'batter_total_bases' AND market_line < 4.0 THEN 'TB 3.5'
                WHEN market = 'batter_total_bases' THEN 'TB 4.5+'
                WHEN market = 'batter_hits' AND market_line < 1.0 THEN 'H 0.5'
                WHEN market = 'batter_hits' AND market_line < 2.0 THEN 'H 1.5'
                WHEN market = 'batter_hits' AND market_line < 3.0 THEN 'H 2.5'
                WHEN market = 'batter_hits' THEN 'H 3.5+'
                WHEN market = 'batter_home_runs' AND market_line < 1.0 THEN 'HR 0.5'
                WHEN market = 'batter_home_runs' THEN 'HR 1.5+'
                ELSE 'other'
            END) AS line_bucket,
            COALESCE(NULLIF(model_meta->>'price_bucket', ''), CASE
                WHEN market_price IS NULL THEN 'missing_price'
                WHEN market_price > 0 AND market_price < 150 THEN 'plus_100_149'
                WHEN market_price > 0 AND market_price < 250 THEN 'plus_150_249'
                WHEN market_price > 0 AND market_price < 500 THEN 'plus_250_499'
                WHEN market_price > 0 THEN 'plus_500_plus'
                WHEN market_price >= -129 THEN 'fair_lay'
                WHEN market_price >= -149 THEN 'lay_130_149'
                WHEN market_price >= -180 THEN 'lay_150_180'
                ELSE 'heavy_lay'
            END) AS price_bucket
        FROM bets.mlb_model_pick_ledger
        WHERE {_micro_where()}
          AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY game_date_et DESC, inserted_at_utc DESC
        LIMIT 50
    """


def _fetch(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        graded = int(row.get("graded") or 0)
        stake = float(row.get("graded_stake_usd") or 0.0)
        row["roi"] = (float(row.get("profit_usd_est") or 0.0) / stake) if stake > 0 else None
        if row.get("avg_model_prob") is not None and row.get("win_rate") is not None:
            row["calibration_error"] = float(row["avg_model_prob"]) - float(row["win_rate"])
        else:
            row["calibration_error"] = None
        row["record"] = f"{int(row.get('wins') or 0)}-{int(row.get('losses') or 0)}-{int(row.get('pushes') or 0)}"
        row["graded"] = graded
    return rows


def _ladder_blockers(row: dict[str, Any], *, target: str) -> list[str]:
    min_graded = _STARTER_MIN_GRADED if target == "starter" else _BANKROLL_MIN_GRADED
    min_dates = _STARTER_MIN_DATES if target == "starter" else _BANKROLL_MIN_DATES
    blockers: list[str] = []
    graded = int(row.get("graded") or 0)
    clv_rows = int(row.get("clv_rows") or 0)
    dates = int(row.get("graded_dates") or 0)
    roi = row.get("roi")
    clv_beat = row.get("clv_beat_rate")
    avg_clv = row.get("avg_clv_price")
    cal_err = row.get("calibration_error")
    player_share = row.get("max_player_share")
    team_share = row.get("max_team_share")
    if graded < min_graded:
        blockers.append(f"needs_{min_graded - graded}_graded")
    if dates < min_dates:
        blockers.append(f"needs_{min_dates - dates}_dates")
    if clv_rows < _MIN_CLV_ROWS:
        blockers.append(f"needs_{_MIN_CLV_ROWS - clv_rows}_clv_rows")
    if roi is None or float(roi) <= _MIN_ROI:
        blockers.append("roi_not_positive")
    if clv_beat is None or float(clv_beat) < _MIN_CLV_BEAT:
        blockers.append("clv_beat_below_55")
    if avg_clv is None or float(avg_clv) <= _MIN_AVG_CLV:
        blockers.append("avg_clv_not_positive")
    if cal_err is None or abs(float(cal_err)) > _MAX_ABS_CALIBRATION_ERROR:
        blockers.append("calibration_over_5pct")
    if player_share is not None and float(player_share) > _MAX_PLAYER_SHARE:
        blockers.append("player_concentration")
    if team_share is not None and float(team_share) > _MAX_TEAM_SHARE:
        blockers.append("team_concentration")
    return blockers


def _fetch_ladder(conn, params: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _fetch(conn, _ladder_sql(), params)
    for row in rows:
        starter_blockers = _ladder_blockers(row, target="starter")
        bankroll_blockers = _ladder_blockers(row, target="bankroll")
        row["starter_ready"] = not starter_blockers
        row["bankroll_ready"] = not bankroll_blockers
        row["starter_blockers"] = starter_blockers
        row["bankroll_blockers"] = bankroll_blockers
        if row["bankroll_ready"]:
            row["next_ladder_action"] = "bankroll_candidate_review"
        elif row["starter_ready"]:
            row["next_ladder_action"] = "starter_candidate_review"
        else:
            row["next_ladder_action"] = "stay_micro"
    return rows


def _fetch_recent_bets(conn, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_recent_bets_sql(), params)
        return [dict(row) for row in cur.fetchall()]


def build_payload(*, pg_dsn: str, end_date: date, lookback_days: int) -> dict[str, Any]:
    start_date = end_date - timedelta(days=max(1, int(lookback_days)) - 1)
    params = {"start_date": start_date, "end_date": end_date}
    with psycopg2.connect(pg_dsn) as conn:
        overall = _fetch(conn, _summary_sql([]), params)[0]
        by_market = _fetch(conn, _summary_sql(["market", "side"]), params)
        by_book = _fetch(conn, _summary_sql(["market", "side", "bookmaker_key"]), params)
        by_exact_bucket = _fetch(conn, _exact_bucket_sql(), params)
        ladder = _fetch_ladder(conn, params)
        recent_bets = _fetch_recent_bets(conn, params)
        recent = _fetch(conn, _summary_sql(["game_date_et"]), params)
    return {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "lookback_days": int(lookback_days),
        "status": "ready",
        "overall": overall,
        "by_market_side": by_market,
        "by_market_side_book": by_book,
        "by_exact_bucket": by_exact_bucket,
        "ladder_evidence": ladder,
        "recent_bets": recent_bets,
        "by_date": recent,
    }


def render(payload: dict[str, Any]) -> str:
    overall = payload.get("overall") or {}
    lines = [
        "# MLB Prop Micro Ledger Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Window: {payload.get('start_date')} through {payload.get('end_date')}",
        "",
        "## Overall $1 Micro Projection",
        "",
        f"- Graded: {overall.get('graded', 0)}",
        f"- Pending: {overall.get('pending', 0)}",
        f"- Record: {overall.get('record', '0-0-0')}",
        f"- ROI: {_fmt_pct(overall.get('roi'), signed=True)}",
        f"- Units: {_fmt_num(overall.get('units'))}",
        f"- Estimated $ P/L: {_fmt_num(overall.get('profit_usd_est'), 2)}",
        f"- Brier: {_fmt_num(overall.get('brier'))}",
        f"- Calibration error: {_fmt_pct(overall.get('calibration_error'), signed=True)}",
        f"- CLV rows: {overall.get('clv_rows', 0)}",
        f"- CLV beat: {_fmt_pct(overall.get('clv_beat_rate'))}",
        f"- Avg CLV price: {_fmt_num(overall.get('avg_clv_price'))}",
        "",
        "## By Market / Side",
        "",
        "| Market | Side | Graded | Pending | Record | ROI | Brier | Cal Err | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("by_market_side") or []:
        lines.append(
            f"| {row.get('market')} | {row.get('side')} | {row.get('graded')} | {row.get('pending')} | "
            f"{row.get('record')} | {_fmt_pct(row.get('roi'), signed=True)} | {_fmt_num(row.get('brier'))} | "
            f"{_fmt_pct(row.get('calibration_error'), signed=True)} | {row.get('clv_rows')} | "
            f"{_fmt_pct(row.get('clv_beat_rate'))} | {_fmt_num(row.get('avg_clv_price'))} |"
        )
    lines.extend([
        "",
        "## By Exact Bucket",
        "",
        "| Market | Side | Book | Line Bucket | Price Bucket | Graded | Pending | Record | ROI | Brier | CLV Beat | Avg CLV |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ])
    for row in payload.get("by_exact_bucket") or []:
        lines.append(
            f"| {row.get('market')} | {row.get('side')} | {row.get('bookmaker_key')} | "
            f"{row.get('line_bucket')} | {row.get('price_bucket')} | {row.get('graded')} | "
            f"{row.get('pending')} | {row.get('record')} | {_fmt_pct(row.get('roi'), signed=True)} | "
            f"{_fmt_num(row.get('brier'))} | {_fmt_pct(row.get('clv_beat_rate'))} | "
            f"{_fmt_num(row.get('avg_clv_price'))} |"
        )
    lines.extend([
        "",
        "## Ladder Evidence",
        "",
        "| Market | Side | Book | Line Bucket | Price Bucket | Graded | Dates | ROI | CLV Rows | CLV Beat | Avg CLV | Cal Err | Max Player | Max Team | Next | Blockers |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in payload.get("ladder_evidence") or []:
        if row.get("starter_ready"):
            blockers = row.get("bankroll_blockers") or []
        else:
            blockers = row.get("starter_blockers") or []
        lines.append(
            f"| {row.get('market')} | {row.get('side')} | {row.get('bookmaker_key')} | "
            f"{row.get('line_bucket')} | {row.get('price_bucket')} | {row.get('graded')} | "
            f"{row.get('graded_dates')} | {_fmt_pct(row.get('roi'), signed=True)} | {row.get('clv_rows')} | "
            f"{_fmt_pct(row.get('clv_beat_rate'))} | {_fmt_num(row.get('avg_clv_price'))} | "
            f"{_fmt_pct(row.get('calibration_error'), signed=True)} | {_fmt_pct(row.get('max_player_share'))} | "
            f"{_fmt_pct(row.get('max_team_share'))} | {row.get('next_ladder_action')} | {', '.join(blockers[:6])} |"
        )
    lines.extend([
        "",
        "## Recent Micro Bets",
        "",
        "| Date | Player | Team | Market | Side | Book | Line | Price | Min | Prob | EV | Result | Actual | Units | CLV | Close | Bucket |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("recent_bets") or []:
        if row.get("push"):
            result = "push"
        elif row.get("won") is True:
            result = "win"
        elif row.get("won") is False:
            result = "loss"
        else:
            result = row.get("result_status") or "pending"
        bucket = " / ".join(
            str(part)
            for part in (row.get("line_bucket"), row.get("price_bucket"))
            if part
        )
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('player_name')} | {row.get('team_abbr')} | "
            f"{row.get('market')} | {row.get('side')} | {row.get('bookmaker_key')} | "
            f"{_fmt_num(row.get('market_line'), 1)} | {_fmt_num(row.get('market_price'), 0)} | "
            f"{_fmt_num(row.get('minimum_acceptable_price'), 0)} | {_fmt_pct(row.get('model_prob'))} | "
            f"{_fmt_pct(row.get('ev'), signed=True)} | {result} | {_fmt_num(row.get('actual_value'))} | "
            f"{_fmt_num(row.get('profit_units'))} | {_fmt_num(row.get('clv_price'))} | "
            f"{_fmt_num(row.get('closing_price'), 0)} | {bucket} |"
        )
    lines.extend([
        "",
        "## By Date",
        "",
        "| Date | Graded | Pending | Record | ROI | CLV Beat |",
        "|---|---:|---:|---|---:|---:|",
    ])
    for row in payload.get("by_date") or []:
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('graded')} | {row.get('pending')} | "
            f"{row.get('record')} | {_fmt_pct(row.get('roi'), signed=True)} | {_fmt_pct(row.get('clv_beat_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_micro_ledger_report.json"
    report_path = _REPORT_DIR / "mlb_prop_micro_ledger_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, render(payload))
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB $1 micro prop ledger report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--date", default=None, help="End date YYYY-MM-DD ET")
    parser.add_argument("--lookback-days", type=int, default=45)
    args = parser.parse_args()
    end_date = date.fromisoformat(args.date) if args.date else datetime.now(tz=_ET).date()
    payload = build_payload(pg_dsn=args.pg_dsn, end_date=end_date, lookback_days=args.lookback_days)
    json_path, report_path = write_outputs(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "graded": (payload.get("overall") or {}).get("graded", 0),
        "pending": (payload.get("overall") or {}).get("pending", 0),
        "json_path": str(json_path),
        "report_path": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
