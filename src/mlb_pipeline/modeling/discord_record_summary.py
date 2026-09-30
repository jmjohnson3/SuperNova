"""Small Discord-ready W/L summaries for MLB locked ledgers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from .model_release import canonical_forecast_phase

from .prop_ledger_classification import PROP_LEDGER_LABELS

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN


@dataclass(frozen=True)
class RecordSummaryConfig:
    pg_dsn: str = _PG_DSN
    end_date: date | None = None
    lookback_days: int = 30


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _summary_rows(conn, table: str, source: str, start_date: date, end_date: date) -> dict[str, Any]:
    schema, name = table.split(".", 1)
    if not _table_exists(conn, schema, name):
        return {"exists": False}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT
                COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded,
                COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND won IS TRUE)::int AS wins,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND won IS FALSE AND COALESCE(push, false) IS FALSE)::int AS losses,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND COALESCE(push, false) IS TRUE)::int AS pushes,
                COALESCE(SUM(profit_units) FILTER (WHERE result_status = 'graded'), 0)::float AS units
            FROM {table}
            WHERE game_date_et BETWEEN %(start_date)s AND %(end_date)s
              AND source = %(source)s
            """,
            {"start_date": start_date, "end_date": end_date, "source": source},
        )
        row = dict(cur.fetchone() or {})
    row["exists"] = True
    graded = int(row.get("graded") or 0)
    row["roi"] = (float(row.get("units") or 0.0) / graded) if graded else None
    return row


def _empty_summary(*, exists: bool = True) -> dict[str, Any]:
    return {
        "exists": exists,
        "graded": 0,
        "pending": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "units": 0.0,
        "roi": None,
        "valid_clv": 0,
        "clv_beats": 0,
        "avg_clv": None,
    }


def _prop_ledger_rows(conn, start_date: date, end_date: date) -> dict[str, dict[str, Any]]:
    if not _table_exists(conn, "bets", "mlb_model_pick_ledger"):
        return {key: _empty_summary(exists=False) for key, _label in PROP_LEDGER_LABELS}

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH prop_rows AS (
                SELECT
                    result_status,
                    won,
                    push,
                    profit_units,
                    clv_valid,
                    clv_price,
                    LOWER(COALESCE(model_tier, '')) AS model_tier_l,
                    LOWER(CONCAT_WS(
                        ' ',
                        COALESCE(warning_reasons, ''),
                        COALESCE(model_meta->>'selector_reasons', '')
                    )) AS reasons_l,
                    LOWER(COALESCE(bookmaker_key, '')) AS book_l,
                    LOWER(COALESCE(market, stat, '')) AS market_l,
                    LOWER(COALESCE(side, '')) AS side_l,
                    COALESCE(market_line, bet_line)::float AS line_v,
                    LOWER(COALESCE(model_meta->>'selector_tier', '')) AS selector_tier_l,
                    LOWER(COALESCE(model_meta->>'pair_quality', '')) AS pair_quality_l,
                    LOWER(COALESCE(model_meta->>'market_prob_source', '')) AS market_prob_source_l
                FROM bets.mlb_model_pick_ledger
                WHERE source = 'prop'
                  AND game_date_et BETWEEN %(start_date)s AND %(end_date)s
            ),
            categorized AS (
                SELECT
                    *,
                    CASE
                        WHEN book_l = 'fanduel'
                             AND market_l IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
                             AND (
                                 reasons_l LIKE '%%fanduel_synthetic%%'
                                 OR reasons_l LIKE '%%one_sided%%'
                                 OR reasons_l LIKE '%%one-sided%%'
                                 OR reasons_l LIKE '%%synthetic%%'
                                 OR pair_quality_l IN ('synthetic', 'one_sided')
                                 OR market_prob_source_l IN (
                                     'raw_implied_one_sided',
                                     'one_sided',
                                     'one_sided_fanduel_ladder',
                                     'synthetic_fanduel_over_only'
                                 )
                                 OR (
                                     side_l = 'over'
                                     AND pair_quality_l NOT IN ('same_book', 'cross_book')
                                 )
                             )
                            THEN 'one_sided_fanduel'
                        WHEN (
                                 selector_tier_l = 'lottery'
                                 OR reasons_l LIKE '%%alt_line_lottery%%'
                                 OR reasons_l LIKE '%%lottery%%'
                                 OR (
                                     side_l = 'over'
                                     AND (
                                         (market_l = 'batter_hits' AND line_v >= 2.5)
                                         OR (market_l = 'batter_total_bases' AND line_v >= 3.5)
                                         OR (market_l = 'batter_home_runs' AND line_v >= 1.5)
                                     )
                                 )
                             )
                            THEN 'lottery'
                        WHEN selector_tier_l = 'micro_projection'
                             OR model_tier_l = 'micro_projection'
                            THEN 'micro_projection'
                        WHEN model_tier_l = 'micro'
                            THEN 'micro'
                        WHEN model_tier_l IN ('bankroll', 'starter')
                            THEN 'bankroll'
                        WHEN model_tier_l = 'watch'
                            THEN 'watch'
                        ELSE 'paper_common'
                    END AS ledger
                FROM prop_rows
            )
            SELECT
                ledger,
                COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded,
                COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending,
                COUNT(*) FILTER (WHERE result_status = 'graded' AND won IS TRUE)::int AS wins,
                COUNT(*) FILTER (
                    WHERE result_status = 'graded'
                      AND won IS FALSE
                      AND COALESCE(push, false) IS FALSE
                )::int AS losses,
                COUNT(*) FILTER (
                    WHERE result_status = 'graded'
                      AND COALESCE(push, false) IS TRUE
                )::int AS pushes,
                COALESCE(SUM(profit_units) FILTER (WHERE result_status = 'graded'), 0)::float AS units,
                COUNT(*) FILTER (
                    WHERE result_status = 'graded'
                      AND clv_valid IS TRUE
                )::int AS valid_clv,
                COUNT(*) FILTER (
                    WHERE result_status = 'graded'
                      AND clv_valid IS TRUE
                      AND clv_price > 0
                )::int AS clv_beats,
                AVG(clv_price) FILTER (
                    WHERE result_status = 'graded'
                      AND clv_valid IS TRUE
                )::float AS avg_clv
            FROM categorized
            GROUP BY ledger
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        fetched = {str(row["ledger"]): dict(row) for row in cur.fetchall()}

    out = {key: _empty_summary() for key, _label in PROP_LEDGER_LABELS}
    for key, row in fetched.items():
        if key not in out:
            continue
        row["exists"] = True
        graded = int(row.get("graded") or 0)
        row["roi"] = (float(row.get("units") or 0.0) / graded) if graded else None
        out[key].update(row)
    return out


def _forecast_accuracy_rows(conn, start_date: date, end_date: date) -> list[dict[str, Any]]:
    if not _table_exists(conn, "bets", "mlb_daily_forecast_ledger"):
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH canonical AS (
                SELECT DISTINCT ON (game_date_et, game_slug, COALESCE(player_id, 0), stat)
                    game_date_et, game_slug, player_id, stat, forecast_type,
                    result_status, projection_value, baseline_value, actual_value
                FROM bets.mlb_daily_forecast_ledger
                WHERE game_date_et BETWEEN %(start_date)s AND %(end_date)s
                  AND forecast_phase = %(forecast_phase)s
                ORDER BY game_date_et, game_slug, COALESCE(player_id, 0), stat,
                         locked_at_utc, id
            )
            SELECT stat, forecast_type,
                   COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded,
                   COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending,
                   AVG(ABS(projection_value - actual_value))
                       FILTER (WHERE result_status = 'graded')::float AS mae,
                   AVG(ABS(baseline_value - actual_value))
                       FILTER (WHERE result_status = 'graded' AND baseline_value IS NOT NULL)::float AS baseline_mae
            FROM canonical
            GROUP BY stat, forecast_type
            ORDER BY forecast_type, stat
            """,
            {
                "start_date": start_date,
                "end_date": end_date,
                "forecast_phase": canonical_forecast_phase(),
            },
        )
        return [dict(row) for row in cur.fetchall()]


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_units(value: Any) -> str:
    try:
        return f"{float(value):+.2f}u"
    except (TypeError, ValueError):
        return "+0.00u"


def _fmt_pp(value: Any) -> str:
    try:
        return f"{float(value):+.2f}pp"
    except (TypeError, ValueError):
        return "-"


def _line(label: str, rec: dict[str, Any]) -> str:
    if not rec.get("exists"):
        return f"- {label}: no ledger yet"
    wins = int(rec.get("wins") or 0)
    losses = int(rec.get("losses") or 0)
    pushes = int(rec.get("pushes") or 0)
    graded = int(rec.get("graded") or 0)
    pending = int(rec.get("pending") or 0)
    units = _fmt_units(rec.get("units"))
    roi = _fmt_pct(rec.get("roi"), signed=True)
    pending_s = f", {pending} pending" if pending else ""
    clv_s = ""
    valid_clv = int(rec.get("valid_clv") or 0)
    if valid_clv:
        clv_beats = int(rec.get("clv_beats") or 0)
        clv_rate = clv_beats / valid_clv
        avg_clv = _fmt_pp(rec.get("avg_clv"))
        clv_s = f", CLV {clv_beats}/{valid_clv} ({_fmt_pct(clv_rate)}), avg {avg_clv}"
    return f"- {label}: {wins}-{losses}-{pushes} ({graded} graded{pending_s}), {units}, ROI {roi}{clv_s}"


def build_record_summary(cfg: RecordSummaryConfig) -> dict[str, Any]:
    end_date = cfg.end_date or datetime.now(_ET).date()
    start_date = end_date - timedelta(days=max(1, cfg.lookback_days) - 1)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        game_bankroll = _summary_rows(conn, "bets.mlb_bankroll_ledger", "game", start_date, end_date)
        game_model = _summary_rows(conn, "bets.mlb_model_pick_ledger", "game", start_date, end_date)
        prop_shadow = _summary_rows(conn, "bets.mlb_model_pick_ledger", "prop", start_date, end_date)
        prop_ledgers = _prop_ledger_rows(conn, start_date, end_date)
        forecast_accuracy = _forecast_accuracy_rows(conn, start_date, end_date)
    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "lookback_days": cfg.lookback_days,
        "game_bankroll": game_bankroll,
        "game_model": game_model,
        "prop_shadow": prop_shadow,
        "prop_ledgers": prop_ledgers,
        "forecast_accuracy": forecast_accuracy,
    }


def format_record_summary(
    *,
    pg_dsn: str = _PG_DSN,
    end_date: date | None = None,
    lookback_days: int = 30,
    include_game_bankroll: bool = True,
    include_game_model: bool = True,
    include_prop_shadow: bool = True,
) -> str:
    try:
        payload = build_record_summary(RecordSummaryConfig(
            pg_dsn=pg_dsn,
            end_date=end_date,
            lookback_days=lookback_days,
        ))
    except Exception:
        return ""
    lines = [f"**Records ({payload['start_date']} to {payload['end_date']}, graded)**"]
    if include_game_bankroll:
        lines.append(_line("Game bankroll", payload["game_bankroll"]))
    if include_game_model:
        lines.append(_line("Game model picks", payload["game_model"]))
    if include_prop_shadow:
        lines.append("**Player Prop Ledgers**")
        prop_ledgers = payload.get("prop_ledgers") or {}
        for key, label in PROP_LEDGER_LABELS:
            lines.append(_line(label, prop_ledgers.get(key) or _empty_summary(exists=False)))
    forecast_rows = payload.get("forecast_accuracy") or []
    if forecast_rows:
        lines.append("**Locked Forecast Accuracy**")
        for rec in forecast_rows:
            graded = int(rec.get("graded") or 0)
            pending = int(rec.get("pending") or 0)
            mae = _fmt_num(rec.get("mae")) if graded else "-"
            baseline_mae = _fmt_num(rec.get("baseline_mae")) if graded else "-"
            lines.append(
                f"- {rec.get('stat')}: {graded} graded, {pending} pending, "
                f"MAE {mae} (baseline {baseline_mae})"
            )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print MLB record ledger summary")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    print(format_record_summary(end_date=args.end_date, lookback_days=args.lookback_days))


if __name__ == "__main__":
    main()
