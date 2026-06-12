"""Audit target quality for locked MLB prop training rows.

Predictive power depends on clean labels.  This report checks whether every
historical offer-level row has the fields needed for non-leaky prop training:
exact book/line/price, lock snapshot, close snapshot validity, result, and
bookability status.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

from .prop_market_training import ensure_prop_market_training_schema

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class TargetQualityConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    out: str = "mlb_prop_target_quality_latest.md"
    json_out: str = "prop_target_quality_report.json"


SQL = """
SELECT
    id,
    run_id,
    replay_id,
    game_date_et,
    game_slug,
    player_id,
    player_name,
    market,
    side,
    bookmaker_key,
    market_line::float AS market_line,
    market_price::float AS market_price,
    paired_price::float AS paired_price,
    paired_bookmaker_key,
    paired_price_source,
    pair_quality,
    no_vig_market_prob::float AS no_vig_market_prob,
    market_prob_source,
    prop_offer_id,
    lock_snapshot_id,
    source_created_at,
    actual_value::float AS actual_value,
    won,
    push,
    result_status,
    closing_line::float AS closing_line,
    closing_price::float AS closing_price,
    closing_snapshot_id,
    closing_fetched_at_utc,
    clv_valid,
    clv_status,
    clv_unknown_reason,
    clv_match_method
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
ORDER BY game_date_et, id
"""


CORE_REQUIRED_FIELDS = [
    "bookmaker_key",
    "market_line",
    "market_price",
    "prop_offer_id",
    "lock_snapshot_id",
    "source_created_at",
    "actual_value",
    "won",
]

PAIRING_FIELDS = [
    "paired_price",
    "paired_bookmaker_key",
    "paired_price_source",
    "pair_quality",
    "no_vig_market_prob",
    "market_prob_source",
]


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


def _load(cfg: TargetQualityConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_training_schema(conn)
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in [
        "market_line",
        "market_price",
        "paired_price",
        "no_vig_market_prob",
        "actual_value",
        "closing_line",
        "closing_price",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coverage(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    total = max(1, len(df))
    for field in CORE_REQUIRED_FIELDS + PAIRING_FIELDS + [
        "closing_line",
        "closing_price",
        "closing_snapshot_id",
        "closing_fetched_at_utc",
        "clv_valid",
    ]:
        present = df[field].notna() if field in df else pd.Series([False] * len(df))
        rows.append({
            "field": field,
            "present": int(present.sum()),
            "missing": int((~present).sum()),
            "coverage": float(present.sum() / total),
        })
    return rows


def _date_quality(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for date_value, group in df.groupby("game_date_et", dropna=False):
        graded = group["won"].notna()
        price_lock = group[["bookmaker_key", "market_line", "market_price", "lock_snapshot_id"]].notna().all(axis=1)
        offer_id = group["prop_offer_id"].notna()
        pair_source = group["paired_price_source"].fillna("").astype(str)
        pair_quality = group["pair_quality"].fillna("").astype(str)
        same_book_pair = pair_quality.eq("same_book") | pair_source.isin({"prediction_same_book", "same_book_exact_line", "same_book_exact_line_fallback"})
        cross_book_pair = pair_quality.eq("cross_book") | pair_source.isin({"cross_book_exact_line", "cross_book_exact_line_fallback"})
        synthetic_pair = pair_quality.eq("synthetic") | pair_source.eq("synthetic_fanduel_over_only_complement")
        any_pair = group["paired_price"].notna()
        valid_close = group["clv_valid"].fillna(False).astype(bool)
        stale = group["clv_unknown_reason"].fillna("").astype(str).eq("stale_close_before_lock")
        rows.append({
            "date": str(date_value),
            "rows": int(len(group)),
            "offer_id_rate": float(offer_id.mean()) if len(group) else None,
            "price_lock_rate": float(price_lock.mean()) if len(group) else None,
            "same_book_pair_rate": float(same_book_pair.mean()) if len(group) else None,
            "cross_book_pair_rate": float(cross_book_pair.mean()) if len(group) else None,
            "synthetic_pair_rate": float(synthetic_pair.mean()) if len(group) else None,
            "true_pair_rate": float((same_book_pair | cross_book_pair).mean()) if len(group) else None,
            "any_pair_rate": float(any_pair.mean()) if len(group) else None,
            "graded_rate": float(graded.mean()) if len(group) else None,
            "valid_close_rate": float(valid_close.mean()) if len(group) else None,
            "stale_close_rate": float(stale.mean()) if len(group) else None,
        })
    rows.sort(key=lambda rec: rec["date"], reverse=True)
    return rows


def _pairing_quality(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows
    for values, group in df.groupby(["game_date_et", "market", "bookmaker_key"], dropna=False):
        date_value, market, book = values
        source = group["paired_price_source"].fillna("missing").astype(str)
        quality = group["pair_quality"].fillna("unknown").astype(str)
        rows.append({
            "date": str(date_value),
            "market": market,
            "bookmaker_key": book,
            "rows": int(len(group)),
            "same_book_pairs": int(source.isin({"prediction_same_book", "same_book_exact_line", "same_book_exact_line_fallback"}).sum()),
            "cross_book_pairs": int(source.isin({"cross_book_exact_line", "cross_book_exact_line_fallback"}).sum()),
            "synthetic_pairs": int(source.eq("synthetic_fanduel_over_only_complement").sum()),
            "missing_pairs": int(source.eq("missing").sum()),
            "same_book_quality": int(quality.eq("same_book").sum()),
            "cross_book_quality": int(quality.eq("cross_book").sum()),
            "synthetic_quality": int(quality.eq("synthetic").sum()),
            "one_sided_quality": int(quality.eq("one_sided").sum()),
            "raw_implied": int(group["market_prob_source"].fillna("").astype(str).eq("raw_implied").sum()),
            "synthetic_prob": int(group["market_prob_source"].fillna("").astype(str).eq("synthetic_fanduel_over_only").sum()),
        })
    rows.sort(key=lambda rec: (rec["date"], rec["rows"]), reverse=True)
    return rows


def _bad_examples(df: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
    problems = []
    for _, row in df.iterrows():
        missing = [field for field in CORE_REQUIRED_FIELDS if pd.isna(row.get(field))]
        reason = row.get("clv_unknown_reason")
        reason = None if pd.isna(reason) else reason
        paired_source = row.get("paired_price_source")
        paired_source = None if pd.isna(paired_source) else paired_source
        pair_quality = row.get("pair_quality")
        pair_quality = None if pd.isna(pair_quality) else pair_quality
        pairing_note = ""
        if pair_quality in {"one_sided", "unknown"} or pd.isna(row.get("paired_price")):
            pairing_note = "one_sided_or_unpaired"
        elif pair_quality == "cross_book" or paired_source in {"cross_book_exact_line", "cross_book_exact_line_fallback"}:
            pairing_note = "cross_book_pair"
        elif pair_quality == "synthetic" or paired_source == "synthetic_fanduel_over_only_complement":
            pairing_note = "synthetic_fanduel_over_only"
        if missing or pairing_note or reason in {"stale_close_before_lock", "close_outside_two_hour_window", "fallback_other_book_only"}:
            problems.append({
                "date": str(row.get("game_date_et")),
                "player": row.get("player_name"),
                "bet": f"{row.get('market')} {row.get('side')} {row.get('market_line')} {row.get('bookmaker_key')}",
                "missing": missing,
                "pairing_note": pairing_note,
                "clv_status": row.get("clv_status"),
                "clv_unknown_reason": reason,
            })
    return problems[:limit]


def _fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _write_report(payload: dict[str, Any], cfg: TargetQualityConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.out
    lines = [
        "# MLB Prop Target Quality",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Rows: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        "",
        "## Required Field Coverage",
        "",
        "| Field | Present | Missing | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for rec in payload.get("coverage", []):
        lines.append(f"| {rec['field']} | {rec['present']} | {rec['missing']} | {_fmt_pct(rec['coverage'])} |")
    lines.extend([
        "",
        "## CLV / Close Status",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ])
    for status, rows in (payload.get("clv_status_counts") or {}).items():
        lines.append(f"| {status} | {rows} |")
    lines.extend([
        "",
        "## CLV Unknown Reasons",
        "",
        "| Reason | Rows |",
        "|---|---:|",
    ])
    for reason, rows in (payload.get("clv_unknown_reason_counts") or {}).items():
        lines.append(f"| {reason} | {rows} |")
    lines.extend([
        "",
        "## Quality By Date",
        "",
        "| Date | Rows | Offer ID | Price+Lock | True Pair | Same-Book Pair | Cross-Book Pair | Synthetic Pair | Any Pair | Graded | Valid Close | Stale Close |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("date_quality", []):
        lines.append(
            f"| {rec['date']} | {rec['rows']} | {_fmt_pct(rec.get('offer_id_rate'))} | "
            f"{_fmt_pct(rec.get('price_lock_rate'))} | {_fmt_pct(rec.get('true_pair_rate'))} | "
            f"{_fmt_pct(rec.get('same_book_pair_rate'))} | {_fmt_pct(rec.get('cross_book_pair_rate'))} | "
            f"{_fmt_pct(rec.get('synthetic_pair_rate'))} | {_fmt_pct(rec.get('any_pair_rate'))} | "
            f"{_fmt_pct(rec.get('graded_rate'))} | {_fmt_pct(rec.get('valid_close_rate'))} | "
            f"{_fmt_pct(rec.get('stale_close_rate'))} |"
        )
    lines.extend([
        "",
        "## Pairing By Date / Market / Book",
        "",
        "| Date | Market | Book | Rows | Same-Book Pairs | Cross-Book Pairs | Synthetic Pairs | Missing Pairs | Same-Book Quality | Cross-Book Quality | Synthetic Quality | One-Sided Quality | Raw-Implied Prob | Synthetic Prob |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("pairing_quality", [])[:80]:
        lines.append(
            f"| {rec['date']} | {rec['market']} | {rec['bookmaker_key']} | {rec['rows']} | "
            f"{rec['same_book_pairs']} | {rec['cross_book_pairs']} | {rec['synthetic_pairs']} | "
            f"{rec['missing_pairs']} | {rec['same_book_quality']} | {rec['cross_book_quality']} | "
            f"{rec['synthetic_quality']} | {rec['one_sided_quality']} | "
            f"{rec['raw_implied']} | {rec['synthetic_prob']} |"
        )
    lines.extend([
        "",
        "## Problem Examples",
        "",
        "| Date | Player | Bet | Missing Fields | Pairing Note | CLV Status | CLV Reason |",
        "|---|---|---|---|---|---|---|",
    ])
    for rec in payload.get("bad_examples", []):
        lines.append(
            f"| {rec['date']} | {rec['player']} | {rec['bet']} | {', '.join(rec['missing'])} | "
            f"{rec.get('pairing_note') or ''} | {rec.get('clv_status') or ''} | "
            f"{rec.get('clv_unknown_reason') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_report(cfg: TargetQualityConfig) -> dict[str, Any]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
    }
    if df.empty:
        payload["status"] = "no_rows"
    else:
        payload["status"] = "ready"
        payload["coverage"] = _coverage(df)
        payload["date_quality"] = _date_quality(df)
        payload["pairing_quality"] = _pairing_quality(df)
        payload["clv_status_counts"] = {
            str(k): int(v) for k, v in df["clv_status"].fillna("missing").value_counts(dropna=False).items()
        }
        payload["clv_unknown_reason_counts"] = {
            str(k): int(v) for k, v in df["clv_unknown_reason"].fillna("none").value_counts(dropna=False).items()
        }
        payload["bad_examples"] = _bad_examples(df)
    payload["report_path"] = _write_report(payload, cfg)
    (_MODEL_DIR / cfg.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop target quality report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--out", default="mlb_prop_target_quality_latest.md")
    parser.add_argument("--json-out", default="prop_target_quality_report.json")
    args = parser.parse_args()
    payload = build_report(TargetQualityConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        out=args.out,
        json_out=args.json_out,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
