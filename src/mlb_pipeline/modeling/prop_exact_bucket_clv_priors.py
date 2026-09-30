"""Exact-bucket CLV priors for micro prop gating.

This is deliberately narrower than the global CLV classifier.  It answers:
"for this exact market/side/line bucket/price bucket/book, have true-paired
locks historically beaten close?"
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .prop_training_groups import dedupe_locked_offer_rows
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_PRIOR_ROWS = 80.0
_MICRO_MIN_CLV_ROWS = 30
_MICRO_MIN_DATES = 4
_MICRO_MIN_CLV_BEAT = 0.54
_FULL_MIN_CLV_ROWS = 50
_FULL_MIN_DATES = 5
_FULL_MIN_CLV_BEAT = 0.55

SQL = """
SELECT
    id,
    replay_id,
    source_created_at,
    game_date_et,
    game_slug,
    player_id,
    market,
    side,
    COALESCE(line_surface, 'unknown') AS line_surface,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    market_line::float AS market_line,
    market_price::float AS market_price,
    clv_valid,
    clv_price::float AS clv_price,
    clv_unknown_reason,
    COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book','cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    pair_quality,
    prop_offer_id
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND side IN ('over','under')
  AND won IS NOT NULL
  AND COALESCE(push, false) IS FALSE
ORDER BY game_date_et, source_created_at, id
"""


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bucket_key(row: pd.Series) -> str:
    market = str(row.get("market") or "unknown")
    side = str(row.get("side") or "unknown").lower()
    line = _clean_float(row.get("market_line"))
    surface = str(row.get("line_surface") or "unknown")
    if surface == "unknown":
        surface = prop_line_surface(market, side, line)
    line_bucket = str(row.get("line_bucket") or "unknown")
    if line_bucket == "unknown":
        line_bucket = prop_line_bucket(market, line)
    pb = str(row.get("price_bucket") or "missing_price")
    if pb == "missing_price":
        pb = price_bucket(row.get("market_price"))
    book = str(row.get("bookmaker_key") or "unknown").lower()
    return "|".join([market, side, surface, line_bucket, pb, book])


def _load_rows(pg_dsn: str, lookback_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, int(lookback_days)))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in ("market_line", "market_price", "clv_price", "true_pair_flag", "synthetic_pair_flag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["clv_valid"] = df["clv_valid"].fillna(False).astype(bool)
    df["clv_unknown_reason"] = df["clv_unknown_reason"].fillna("none").astype(str)
    df["bookmaker_key"] = df["bookmaker_key"].fillna("unknown").astype(str).str.lower()
    df = df.loc[
        df["true_pair_flag"].fillna(0.0).ge(0.5)
        & df["synthetic_pair_flag"].fillna(0.0).lt(0.5)
    ].copy()
    if df.empty:
        return df
    df = dedupe_locked_offer_rows(df)
    df["bucket_key"] = df.apply(_bucket_key, axis=1)
    valid = df["clv_valid"].astype(bool) & df["clv_price"].notna()
    df["valid_clv_row"] = valid.astype(float)
    df["clv_beat"] = np.nan
    df.loc[valid, "clv_beat"] = (pd.to_numeric(df.loc[valid, "clv_price"], errors="coerce") > 0.0).astype(float)
    return df


def build(*, pg_dsn: str = PG_DSN, lookback_days: int = 365) -> dict[str, Any]:
    df = _load_rows(pg_dsn, lookback_days)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not df.empty else "no_rows",
        "usage": "exact_bucket_true_pair_clv_prior_for_micro_gating",
        "lookback_days": int(lookback_days),
        "rows": int(len(df)),
        "thresholds": {
            "micro_min_clv_rows": _MICRO_MIN_CLV_ROWS,
            "micro_min_dates": _MICRO_MIN_DATES,
            "micro_min_clv_beat_rate": _MICRO_MIN_CLV_BEAT,
            "full_min_clv_rows": _FULL_MIN_CLV_ROWS,
            "full_min_dates": _FULL_MIN_DATES,
            "full_min_clv_beat_rate": _FULL_MIN_CLV_BEAT,
        },
        "global_clv_beat_rate": None,
        "global_avg_clv_price": None,
        "buckets": {},
        "top_buckets": [],
    }
    if df.empty:
        _write_outputs(payload)
        return payload

    valid_df = df.loc[df["valid_clv_row"].fillna(0.0).ge(0.5)].copy()
    global_beat = float(valid_df["clv_beat"].mean()) if not valid_df.empty else None
    global_avg = float(pd.to_numeric(valid_df["clv_price"], errors="coerce").mean()) if not valid_df.empty else None
    payload["global_clv_beat_rate"] = global_beat
    payload["global_avg_clv_price"] = global_avg
    payload["global_valid_close_coverage"] = float(df["valid_clv_row"].mean()) if len(df) else None
    stale_mask = df["clv_unknown_reason"].eq("stale_close_before_lock")
    payload["global_stale_close_rate"] = float(stale_mask.mean()) if len(df) else None

    buckets: dict[str, dict[str, Any]] = {}
    for key, group in df.groupby("bucket_key", dropna=False):
        true_pair_rows = int(len(group))
        valid_group = group.loc[group["valid_clv_row"].fillna(0.0).ge(0.5)].copy()
        rows = int(len(valid_group))
        dates = int(valid_group["game_date_et"].nunique())
        true_pair_dates = int(group["game_date_et"].nunique())
        beat = float(valid_group["clv_beat"].mean()) if rows else None
        avg_clv = float(pd.to_numeric(valid_group["clv_price"], errors="coerce").mean()) if rows else None
        valid_close_coverage = float(rows / true_pair_rows) if true_pair_rows else None
        stale_close_rows = int(group["clv_unknown_reason"].eq("stale_close_before_lock").sum())
        stale_close_rate = float(stale_close_rows / true_pair_rows) if true_pair_rows else None
        unknown_reason_counts = {
            str(reason): int(count)
            for reason, count in group.loc[
                ~group["valid_clv_row"].fillna(0.0).ge(0.5),
                "clv_unknown_reason",
            ].fillna("none").value_counts(dropna=False).items()
        }
        shrink = rows / (rows + _PRIOR_ROWS)
        shrunk_beat = (
            shrink * beat + (1.0 - shrink) * global_beat
            if beat is not None and global_beat is not None
            else None
        )
        shrunk_avg = (
            shrink * avg_clv + (1.0 - shrink) * global_avg
            if avg_clv is not None and global_avg is not None
            else None
        )
        # A $1 micro trial is evidence collection, not bankroll promotion. Once
        # an exact bucket has real sample size, use its own CLV rate for the
        # trial gate; keep the shrunk rate for the stricter full-proof gate.
        micro_clv_beat = beat if rows >= 75 else shrunk_beat
        micro_avg_clv = avg_clv if rows >= 75 else shrunk_avg
        micro_blockers: list[str] = []
        full_blockers: list[str] = []
        close_quality_blockers: list[str] = []
        if rows < _MICRO_MIN_CLV_ROWS:
            micro_blockers.append(f"clv_rows<{_MICRO_MIN_CLV_ROWS}")
        if dates < _MICRO_MIN_DATES:
            micro_blockers.append(f"dates<{_MICRO_MIN_DATES}")
        if micro_clv_beat is None or micro_clv_beat < _MICRO_MIN_CLV_BEAT:
            micro_blockers.append("micro_clv_beat_below_trial_gate")
        if micro_avg_clv is None or micro_avg_clv <= 0.0:
            micro_blockers.append("micro_avg_clv_not_positive")
        if rows < _FULL_MIN_CLV_ROWS:
            full_blockers.append(f"clv_rows<{_FULL_MIN_CLV_ROWS}")
        if dates < _FULL_MIN_DATES:
            full_blockers.append(f"dates<{_FULL_MIN_DATES}")
        if shrunk_beat is None or shrunk_beat < _FULL_MIN_CLV_BEAT:
            full_blockers.append("shrunk_clv_beat_below_full")
        if shrunk_avg is None or shrunk_avg <= 0.0:
            full_blockers.append("shrunk_avg_clv_not_positive")
        if valid_close_coverage is None or valid_close_coverage < 0.90:
            close_quality_blockers.append("valid_close_coverage_below_90")
        if stale_close_rate is not None and stale_close_rate > 0.02:
            close_quality_blockers.append("stale_close_rate_above_2")
        first = group.iloc[0]
        buckets[str(key)] = {
            "bucket": str(key),
            "market": first.get("market"),
            "side": first.get("side"),
            "line_surface": str(key).split("|")[2] if str(key).count("|") >= 5 else None,
            "line_bucket": str(key).split("|")[3] if str(key).count("|") >= 5 else None,
            "price_bucket": str(key).split("|")[4] if str(key).count("|") >= 5 else None,
            "bookmaker_key": first.get("bookmaker_key"),
            "clv_rows": rows,
            "dates": dates,
            "true_pair_rows": true_pair_rows,
            "true_pair_dates": true_pair_dates,
            "valid_close_coverage": valid_close_coverage,
            "stale_close_rows": stale_close_rows,
            "stale_close_rate": stale_close_rate,
            "clv_unknown_reason_counts": unknown_reason_counts,
            "clv_beat_rate": beat,
            "avg_clv_price": avg_clv,
            "shrunk_clv_beat_rate": shrunk_beat,
            "shrunk_avg_clv_price": shrunk_avg,
            "micro_clv_beat_rate": micro_clv_beat,
            "micro_avg_clv_price": micro_avg_clv,
            "micro_clv_confirmed": not micro_blockers,
            "full_clv_confirmed": not full_blockers,
            "micro_blockers": micro_blockers,
            "full_blockers": full_blockers,
            "close_quality_blockers": close_quality_blockers,
        }
    payload["buckets"] = buckets
    payload["top_buckets"] = sorted(
        buckets.values(),
        key=lambda row: (
            not bool(row.get("micro_clv_confirmed")),
            -float(row.get("shrunk_clv_beat_rate") or 0.0),
            -int(row.get("clv_rows") or 0),
        ),
    )[:60]
    _write_outputs(payload)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _clean_float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _pct(value: Any) -> str:
    numeric = _clean_float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.1f}%"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Exact-Bucket CLV Priors",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Evidence: true-paired, non-synthetic, valid-close rows only.",
        "",
        f"- Rows: {payload.get('rows', 0)}",
        f"- Global CLV beat: {_pct(payload.get('global_clv_beat_rate'))}",
        f"- Global avg CLV: {_fmt(payload.get('global_avg_clv_price'))}",
        f"- Global valid close coverage: {_pct(payload.get('global_valid_close_coverage'))}",
        f"- Global stale close rate: {_pct(payload.get('global_stale_close_rate'))}",
        "",
        "| Bucket | True Pair | Valid CLV | Coverage | Stale | Dates | CLV Beat | Avg CLV | Trial Beat | Micro CLV | Close Quality | Blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload.get("top_buckets") or []:
        lines.append(
            f"| {row.get('bucket')} | {row.get('true_pair_rows')} | {row.get('clv_rows')} | "
            f"{_pct(row.get('valid_close_coverage'))} | {_pct(row.get('stale_close_rate'))} | "
            f"{row.get('dates')} | "
            f"{_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} | "
            f"{_pct(row.get('micro_clv_beat_rate'))} | "
            f"{bool(row.get('micro_clv_confirmed'))} | "
            f"{', '.join(row.get('close_quality_blockers') or []) or '-'} | "
            f"{', '.join(row.get('micro_blockers') or []) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_exact_bucket_clv_priors.json"
    report_path = _REPORT_DIR / "mlb_prop_exact_bucket_clv_priors_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact-bucket CLV priors for MLB prop micro gating")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    args = parser.parse_args()
    payload = build(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows"),
        "micro_clv_confirmed_buckets": sum(
            1 for row in (payload.get("buckets") or {}).values()
            if row.get("micro_clv_confirmed")
        ),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
