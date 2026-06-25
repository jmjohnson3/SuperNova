"""Small Discord-ready W/L summaries for MLB locked ledgers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

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


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_units(value: Any) -> str:
    try:
        return f"{float(value):+.2f}u"
    except (TypeError, ValueError):
        return "+0.00u"


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
    return f"- {label}: {wins}-{losses}-{pushes} ({graded} graded{pending_s}), {units}, ROI {roi}"


def build_record_summary(cfg: RecordSummaryConfig) -> dict[str, Any]:
    end_date = cfg.end_date or datetime.now(_ET).date()
    start_date = end_date - timedelta(days=max(1, cfg.lookback_days) - 1)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        game_bankroll = _summary_rows(conn, "bets.mlb_bankroll_ledger", "game", start_date, end_date)
        game_model = _summary_rows(conn, "bets.mlb_model_pick_ledger", "game", start_date, end_date)
        prop_shadow = _summary_rows(conn, "bets.mlb_model_pick_ledger", "prop", start_date, end_date)
    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "lookback_days": cfg.lookback_days,
        "game_bankroll": game_bankroll,
        "game_model": game_model,
        "prop_shadow": prop_shadow,
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
        lines.append(_line("Props if bet +EV model picks", payload["prop_shadow"]))
    return "\n".join(lines)
