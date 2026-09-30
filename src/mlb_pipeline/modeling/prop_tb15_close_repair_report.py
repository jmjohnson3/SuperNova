"""Focused close-coverage repair report for DraftKings TB over 1.5.

This bucket is close enough to matter, but exact close coverage has been the
main proof blocker.  The broad end-of-slate report is useful; this narrows the
view to the exact bucket we would consider next for micro.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_training_groups import dedupe_locked_offer_rows
from .side_recalibration import price_bucket

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_FOCUS_PRICE_BUCKETS = {"plus_100_149", "plus_150_249", "plus_250_499"}

SQL = """
SELECT
    id,
    replay_id,
    source_created_at,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    market,
    side,
    bookmaker_key,
    market_line::float AS market_line,
    market_price::float AS market_price,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    pair_quality,
    COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book','cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    COALESCE(push, false) AS push,
    CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 ELSE NULL END AS target,
    profit_units::float AS profit_units,
    clv_valid,
    clv_price::float AS clv_price,
    clv_status,
    COALESCE(clv_unknown_reason, '') AS clv_unknown_reason,
    lock_snapshot_id,
    closing_snapshot_id,
    prop_offer_id
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = 'batter_total_bases'
  AND side = 'over'
  AND LOWER(bookmaker_key) = 'draftkings'
  AND ABS(market_line::float - 1.5) <= 1e-9
ORDER BY game_date_et, source_created_at, id
"""


def _table_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('features.mlb_prop_market_training_examples') IS NOT NULL")
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _close_reason(row: pd.Series) -> str:
    if bool(row.get("clv_valid")):
        return "valid_close"
    if pd.isna(row.get("lock_snapshot_id")):
        return "missing_lock_snapshot"
    reason = str(row.get("clv_unknown_reason") or "").strip()
    return reason if reason and reason.lower() != "nan" else "no_valid_close_snapshot"


def _load_rows(pg_dsn: str, lookback_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, int(lookback_days)))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in ("market_line", "market_price", "true_pair_flag", "synthetic_pair_flag", "target", "profit_units", "clv_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bookmaker_key"] = df["bookmaker_key"].fillna("unknown").astype(str).str.lower()
    df["price_bucket"] = [
        bucket if isinstance(bucket, str) and bucket and bucket != "missing_price" else price_bucket(price)
        for bucket, price in zip(df["price_bucket"], df["market_price"])
    ]
    df["clv_valid_bool"] = df["clv_valid"].fillna(False).astype(bool)
    df["close_reason"] = df.apply(_close_reason, axis=1)
    return dedupe_locked_offer_rows(df)


def _metrics(group: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(group))
    valid = int(group["clv_valid_bool"].sum())
    stale = int(group["close_reason"].eq("stale_close_before_lock").sum())
    clv_prices = pd.to_numeric(group.loc[group["clv_valid_bool"], "clv_price"], errors="coerce").dropna()
    graded = group.loc[group["target"].notna() & ~group["push"].fillna(False).astype(bool)].copy()
    reasons = Counter(group["close_reason"].astype(str))
    return {
        "rows": rows,
        "dates": int(group["game_date_et"].nunique()) if rows else 0,
        "valid_closes": valid,
        "valid_close_coverage": _rate(valid, rows),
        "valid_closes_needed_for_90": max(0, math.ceil(0.90 * rows) - valid) if rows else 0,
        "stale_close_rate": _rate(stale, rows),
        "clv_rows": int(len(clv_prices)),
        "clv_beat_rate": float((clv_prices > 0).mean()) if len(clv_prices) else None,
        "avg_clv_price": float(clv_prices.mean()) if len(clv_prices) else None,
        "graded": int(len(graded)),
        "win_rate": float(graded["target"].mean()) if len(graded) else None,
        "roi": float(graded["profit_units"].mean()) if len(graded) and graded["profit_units"].notna().any() else None,
        "failure_reasons": dict(reasons.most_common()),
    }


def _group_rows(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iterable = [(("*",), df)] if not columns else df.groupby(columns, dropna=False)
    for key, group in iterable:
        values = key if isinstance(key, tuple) else (key,)
        rec = _metrics(group)
        rec.update({column: str(value or "unknown") for column, value in zip(columns, values)})
        rows.append(rec)
    rows.sort(key=lambda row: (
        str(row.get("price_bucket") or ""),
        -int(row.get("rows") or 0),
    ))
    return rows


def _action_plan(focus: pd.DataFrame) -> list[dict[str, Any]]:
    if focus.empty:
        return []
    metrics = _metrics(focus)
    reasons = Counter(focus["close_reason"].astype(str))
    plans: list[dict[str, Any]] = []
    if (metrics.get("valid_close_coverage") or 0.0) < 0.90:
        plans.append({
            "area": "valid_close_coverage",
            "rows": metrics["valid_closes_needed_for_90"],
            "diagnosis": "DraftKings TB 1.5 plus-money needs more exact same-book closes before promotion proof counts",
            "next_action": "keep targeted close capture active every 10 minutes inside T-120 and verify offer counts for DK TB 1.5 before first pitch",
        })
    reason_actions = {
        "close_outside_two_hour_window": "scheduler timing/capture window miss; add or verify T-120/T-60/T-20/T-10 event-aware close rows",
        "exact_line_unavailable_at_close": "line moved or exact 1.5 disappeared; count as bookability, not CLV direction",
        "line_disappeared_at_close": "same player/market remained but exact line disappeared; keep bucket proof closed until availability improves",
        "missing_close_side_price": "parser side extraction issue for the close row; inspect raw DK side prices",
        "player_market_unavailable_at_close": "market disappeared for that player; bookability problem",
        "player_prop_unavailable_at_close": "player prop menu disappeared; bookability problem",
        "missing_lock_snapshot": "locked row did not attach an immutable offer ID; repair offer identity before using it as proof",
        "stale_close_before_lock": "ordering bug; close timestamp was before the lock and must not count",
    }
    for reason, count in reasons.most_common():
        if reason == "valid_close":
            continue
        plans.append({
            "area": reason,
            "rows": int(count),
            "diagnosis": "close proof blocker for DK TB 1.5 plus-money",
            "next_action": reason_actions.get(reason, "inspect close resolver taxonomy for this bucket"),
        })
    return plans


def build(*, pg_dsn: str = PG_DSN, lookback_days: int = 365) -> dict[str, Any]:
    df = _load_rows(pg_dsn, lookback_days)
    focus = df.loc[df["price_bucket"].isin(_FOCUS_PRICE_BUCKETS)].copy() if not df.empty else df
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_rows",
        "lookback_days": int(lookback_days),
        "target": "draftkings|batter_total_bases|over|TB 1.5|plus-money",
        "rows": int(len(df)),
        "focus_rows": int(len(focus)) if not focus.empty else 0,
        "overall": _metrics(df) if not df.empty else {},
        "plus_money": _metrics(focus) if not focus.empty else {},
        "by_price_bucket": _group_rows(df, ["price_bucket"]) if not df.empty else [],
        "by_date": _group_rows(focus, ["game_date_et"]) if not focus.empty else [],
        "by_reason": _group_rows(focus, ["close_reason"]) if not focus.empty else [],
        "action_plan": _action_plan(focus) if not focus.empty else [],
    }
    _write_outputs(payload)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _pct(value: Any) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.1f}%"


def _render(payload: dict[str, Any]) -> str:
    overall = payload.get("plus_money") or {}
    lines = [
        "# MLB Prop TB 1.5 Close Repair Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Target: `{payload.get('target')}`",
        f"Status: **{payload.get('status')}**",
        "",
        f"- Plus-money rows: {payload.get('focus_rows', 0)}",
        f"- Valid exact closes: {overall.get('valid_closes', 0)} ({_pct(overall.get('valid_close_coverage'))})",
        f"- Additional valid closes needed for 90%: {overall.get('valid_closes_needed_for_90', 0)}",
        f"- Stale close rate: {_pct(overall.get('stale_close_rate'))}",
        f"- CLV beat: {_pct(overall.get('clv_beat_rate'))}",
        f"- Avg CLV: {_fmt(overall.get('avg_clv_price'))}",
        f"- ROI: {_pct(overall.get('roi'))}",
        "",
        "## Action Plan",
        "",
        "| Area | Rows | Diagnosis | Next Action |",
        "|---|---:|---|---|",
    ]
    for row in payload.get("action_plan") or []:
        lines.append(f"| {row['area']} | {row['rows']} | {row['diagnosis']} | {row['next_action']} |")
    lines.extend([
        "",
        "## By Price Bucket",
        "",
        "| Price Bucket | Rows | Valid | Coverage | Need 90 | CLV Beat | Avg CLV | ROI | Reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in payload.get("by_price_bucket") or []:
        lines.append(
            f"| {row.get('price_bucket')} | {row.get('rows')} | {row.get('valid_closes')} | "
            f"{_pct(row.get('valid_close_coverage'))} | {row.get('valid_closes_needed_for_90')} | "
            f"{_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} | "
            f"{_pct(row.get('roi'))} | {json.dumps(row.get('failure_reasons') or {}, sort_keys=True)} |"
        )
    lines.extend([
        "",
        "## Plus-Money By Date",
        "",
        "| Date | Rows | Valid | Coverage | Need 90 | CLV Beat | Avg CLV | ROI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload.get("by_date") or []:
        lines.append(
            f"| {row.get('game_date_et')} | {row.get('rows')} | {row.get('valid_closes')} | "
            f"{_pct(row.get('valid_close_coverage'))} | {row.get('valid_closes_needed_for_90')} | "
            f"{_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} | {_pct(row.get('roi'))} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_tb15_close_repair.json"
    report_path = _REPORT_DIR / "mlb_prop_tb15_close_repair_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused DK TB 1.5 plus-money close repair report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    args = parser.parse_args()
    payload = build(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days)
    print(json.dumps({
        "status": payload.get("status"),
        "focus_rows": payload.get("focus_rows"),
        "valid_close_coverage": (payload.get("plus_money") or {}).get("valid_close_coverage"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
