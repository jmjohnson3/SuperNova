"""Promotion and CLV coverage report for exact MLB prop buckets."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from .prop_market_training import ensure_prop_market_training_schema
from .prop_real_money_eligibility import (
    PROP_REAL_MONEY_ELIGIBILITY_START_DATE,
    parse_eligibility_start_date,
)
from .side_recalibration import prop_line_bucket, prop_line_surface, price_bucket

from mlb_pipeline.db import PG_DSN as _PG_DSN
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class PromotionConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    eligibility_start_date: date = PROP_REAL_MONEY_ELIGIBILITY_START_DATE
    min_rows: int = 150
    min_clv_rows: int = 30
    min_roi: float = 0.0
    min_clv_beat_rate: float = 0.55
    min_avg_clv_price: float = 0.0
    max_abs_calibration_error: float = 0.05
    max_player_share: float = 0.12
    max_team_share: float = 0.25
    max_day_share: float = 0.35
    min_unique_dates: int = 5
    min_clean_unique_dates: int = 5
    top_n: int = 30
    out: str | None = None
    json_out: str | None = None


SQL = """
SELECT
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    market,
    side,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    market_line::float AS market_line,
    market_price::float AS market_price,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(line_surface, 'unknown') AS line_surface,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    model_prob_side::float AS model_prob_side,
    ev::float AS ev,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS won,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    clv_price::float AS clv_price,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price,
    closing_source_row_id,
    closing_fetched_at_utc,
    clv_match_method,
    COALESCE(clv_valid, false) AS clv_valid,
    clv_status,
    clv_unknown_reason
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND side IN ('over','under')
  AND market_line IS NOT NULL
  AND won IS NOT NULL
"""


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


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _bucket_key(row: pd.Series | dict[str, Any]) -> str:
    return "|".join([
        str(row.get("market", "*")),
        str(row.get("side", "*")),
        str(row.get("line_surface", "*")),
        str(row.get("line_bucket", "*")),
        str(row.get("price_bucket", "*")),
        str(row.get("bookmaker_key", "*")),
    ])


def _load(cfg: PromotionConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_training_schema(conn)
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in (
        "market_line", "market_price", "model_prob_side", "ev", "won",
        "profit_units", "clv_price", "beat_clv_price",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    df["line_bucket"] = [
        prop_line_bucket(market, line)
        for market, line in zip(df["market"], df["market_line"])
    ]
    df["line_surface"] = [
        prop_line_surface(market, side, line)
        for market, side, line in zip(df["market"], df["side"], df["market_line"])
    ]
    df["price_bucket"] = df["market_price"].map(price_bucket)
    df["bucket_key"] = df.apply(_bucket_key, axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


def _share(df: pd.DataFrame, col: str) -> float | None:
    counts = df[col].dropna().value_counts() if col in df.columns else pd.Series(dtype=float)
    return float(counts.iloc[0] / len(df)) if len(df) and not counts.empty else None


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _blocker_category(reason: str) -> str:
    if reason.startswith("rows<") or reason.startswith("total_rows<") or reason.startswith("holdout_rows<") or reason.startswith("train_rows<"):
        return "sample"
    if reason.startswith("clv_") or reason.startswith("avg_clv"):
        return "clv"
    if reason.startswith("roi"):
        return "roi"
    if reason.startswith("abs_calibration"):
        return "calibration"
    if "share" in reason:
        return "concentration"
    if reason.startswith("unique_dates") or reason.startswith("clean_unique_dates"):
        return "sample"
    return "other"


def _bucket_summary(key: str, group: pd.DataFrame, cfg: PromotionConfig) -> dict[str, Any]:
    settled = group.loc[~group["push"]].copy()
    graded = int(len(settled))
    wins = int(pd.to_numeric(settled["won"], errors="coerce").fillna(0).sum())
    priced = group["profit_units"].notna()
    units = float(pd.to_numeric(group.loc[priced, "profit_units"], errors="coerce").sum()) if priced.any() else None
    roi = units / int(priced.sum()) if units is not None and int(priced.sum()) else None
    p = pd.to_numeric(settled["model_prob_side"], errors="coerce")
    y = pd.to_numeric(settled["won"], errors="coerce")
    valid = p.notna() & y.notna()
    win_rate = float(wins / graded) if graded else None
    avg_prob = float(p[valid].mean()) if valid.any() else None
    calibration_error = win_rate - avg_prob if win_rate is not None and avg_prob is not None else None
    valid_clv_mask = group["clv_valid"].fillna(False).astype(bool)
    clv = pd.to_numeric(group.loc[valid_clv_mask, "clv_price"], errors="coerce").dropna()
    clv_beat = pd.to_numeric(group.loc[valid_clv_mask, "beat_clv_price"], errors="coerce").dropna()
    max_player_share = _share(group, "player_id")
    max_team_share = _share(group, "team_abbr")
    max_day_share = _share(group, "game_date_et")
    unique_dates = int(group["game_date_et"].dropna().nunique())

    reasons: list[str] = []
    if graded < cfg.min_rows:
        reasons.append(f"rows<{cfg.min_rows}")
    if len(clv) < cfg.min_clv_rows:
        reasons.append(f"clv_rows<{cfg.min_clv_rows}")
    if roi is None or roi <= cfg.min_roi:
        reasons.append(f"roi<={cfg.min_roi:.3f}")
    if clv_beat.empty or float(clv_beat.mean()) < cfg.min_clv_beat_rate:
        reasons.append(f"clv_beat_rate<{cfg.min_clv_beat_rate:.2f}")
    if clv.empty or float(clv.mean()) <= cfg.min_avg_clv_price:
        reasons.append(f"avg_clv_price<={cfg.min_avg_clv_price:.3f}")
    if calibration_error is None or abs(calibration_error) > cfg.max_abs_calibration_error:
        reasons.append(f"abs_calibration_error>{cfg.max_abs_calibration_error:.3f}")
    if max_player_share is not None and max_player_share > cfg.max_player_share:
        reasons.append(f"max_player_share>{cfg.max_player_share:.2f}")
    if max_team_share is not None and max_team_share > cfg.max_team_share:
        reasons.append(f"max_team_share>{cfg.max_team_share:.2f}")
    if max_day_share is not None and max_day_share > cfg.max_day_share:
        reasons.append(f"max_day_share>{cfg.max_day_share:.2f}")
    if unique_dates < cfg.min_unique_dates:
        reasons.append(f"unique_dates<{cfg.min_unique_dates}")

    method_counts = Counter(
        str(v or "missing")
        for v in group.loc[valid_clv_mask, "clv_match_method"].fillna("missing")
    )
    status_counts = Counter(str(v or "unknown") for v in group["clv_status"].fillna("unknown"))
    unknown_counts = Counter(
        str(v or "none")
        for v in group.loc[~valid_clv_mask, "clv_unknown_reason"].fillna("none")
    )
    categories = Counter(_blocker_category(reason) for reason in reasons)
    metric_gaps: list[str] = []
    if graded < cfg.min_rows:
        metric_gaps.append(f"needs {cfg.min_rows - graded} more rows")
    if len(clv) < cfg.min_clv_rows:
        metric_gaps.append(f"needs {cfg.min_clv_rows - len(clv)} valid CLV closes")
    if unique_dates < cfg.min_unique_dates:
        metric_gaps.append(f"needs {cfg.min_unique_dates - unique_dates} more unique dates")
    if roi is None or roi <= cfg.min_roi:
        metric_gaps.append(
            f"ROI needs > {_fmt_pct(cfg.min_roi)}, currently {_fmt_pct(roi, signed=True)}"
        )
    clv_beat_rate = float(clv_beat.mean()) if not clv_beat.empty else None
    if clv_beat_rate is None or clv_beat_rate < cfg.min_clv_beat_rate:
        metric_gaps.append(
            f"CLV beat needs {_fmt_pct(cfg.min_clv_beat_rate)}, currently {_fmt_pct(clv_beat_rate)}"
        )
    avg_clv_price = float(clv.mean()) if not clv.empty else None
    if avg_clv_price is None or avg_clv_price <= cfg.min_avg_clv_price:
        current_avg_clv = (
            "-"
            if avg_clv_price is None
            else f"{_fmt_num(avg_clv_price, 2, signed=True)}pp"
        )
        metric_gaps.append(
            f"avg CLV needs > {cfg.min_avg_clv_price:+.2f}pp, currently {current_avg_clv}"
        )
    if calibration_error is None or abs(calibration_error) > cfg.max_abs_calibration_error:
        metric_gaps.append(
            f"calibration needs within {_fmt_pct(cfg.max_abs_calibration_error)}, currently {_fmt_pct(calibration_error, signed=True)}"
        )
    if max_day_share is not None and max_day_share > cfg.max_day_share:
        metric_gaps.append(
            f"day concentration needs max {_fmt_pct(cfg.max_day_share)}, currently {_fmt_pct(max_day_share)}"
        )
    return {
        "key": key,
        "market": group["market"].iloc[0],
        "side": group["side"].iloc[0],
        "line_surface": group["line_surface"].iloc[0],
        "line_bucket": group["line_bucket"].iloc[0],
        "price_bucket": group["price_bucket"].iloc[0],
        "bookmaker_key": group["bookmaker_key"].iloc[0],
        "status": "promotable" if not reasons else "blocked",
        "graded": graded,
        "rows_needed": max(0, cfg.min_rows - graded),
        "clv_rows": int(len(clv)),
        "clv_rows_needed": max(0, cfg.min_clv_rows - int(len(clv))),
        "unique_dates": unique_dates,
        "unique_dates_needed": max(0, cfg.min_unique_dates - unique_dates),
        "win_rate": win_rate,
        "roi": roi,
        "units": units,
        "avg_model_prob": avg_prob,
        "calibration_error": calibration_error,
        "avg_clv_price": avg_clv_price,
        "clv_beat_rate": clv_beat_rate,
        "max_player_share": max_player_share,
        "max_team_share": max_team_share,
        "max_day_share": max_day_share,
        "match_methods": dict(method_counts),
        "clv_statuses": dict(status_counts),
        "clv_unknown_reasons": dict(unknown_counts),
        "blocker_categories": dict(categories),
        "reasons": reasons,
        "metric_gaps": metric_gaps,
        "promotion_distance": (
            len(reasons) * 1000
            + max(0, cfg.min_rows - graded)
            + max(0, cfg.min_clv_rows - int(len(clv))) * 2
        ),
    }


def _load_ladder_policy(eligibility_start_date: date) -> dict[str, dict[str, Any]]:
    path = _MODEL_DIR / "prop_bucket_reopen_policy.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if str(payload.get("eligibility_start_date") or "") != eligibility_start_date.isoformat():
        return {}
    return {
        str(key): dict(record or {})
        for key, record in (payload.get("ladder_buckets") or {}).items()
    }


def build_payload(cfg: PromotionConfig) -> dict[str, Any]:
    legacy_df = _load(cfg)
    df = legacy_df.loc[
        legacy_df["game_date_et"] >= cfg.eligibility_start_date
    ].copy() if not legacy_df.empty else legacy_df.copy()
    ladder_policy = _load_ladder_policy(cfg.eligibility_start_date)
    rows: list[dict[str, Any]] = []
    if not df.empty:
        for key, group in df.groupby("bucket_key", dropna=False):
            summary = _bucket_summary(str(key), group, cfg)
            ladder = ladder_policy.get(str(key), {})
            summary["ladder_tier"] = ladder.get("ladder_tier", "watch")
            summary["desired_ladder_tier"] = ladder.get("desired_ladder_tier", "watch")
            summary["ladder_block_reason"] = ladder.get("ladder_block_reason")
            summary["promotion_source"] = ladder.get("promotion_source", "closed")
            summary["bootstrap_micro_eligible"] = bool(ladder.get("bootstrap_micro_eligible"))
            summary["bootstrap_micro_reasons"] = list(ladder.get("bootstrap_micro_reasons") or [])
            summary["holdout_unique_clean_dates"] = ladder.get("holdout_unique_clean_dates")
            summary["holdout_clean_rows"] = ladder.get("holdout_clean_rows")
            ladder_reasons = list(ladder.get("model_reasons") or [])
            if ladder_reasons:
                summary["reasons"] = sorted(set(summary["reasons"] + ladder_reasons))
                summary["blocker_categories"] = dict(
                    Counter(_blocker_category(reason) for reason in summary["reasons"])
                )
            clean_dates = _clean_float(summary.get("holdout_unique_clean_dates"))
            if clean_dates is not None and clean_dates < cfg.min_clean_unique_dates:
                summary["metric_gaps"].append(
                    f"needs {cfg.min_clean_unique_dates - int(clean_dates)} more clean dates"
                )
            summary["status"] = "promotable" if not summary["reasons"] else "blocked"
            summary["promotion_distance"] = (
                len(summary["reasons"]) * 1000
                + int(summary.get("rows_needed") or 0)
                + int(summary.get("clv_rows_needed") or 0) * 2
            )
            rows.append(summary)
    rows.sort(key=lambda r: (
        r["promotion_distance"],
        -int(r["graded"]),
        str(r["key"]),
    ))
    alt_rows = [r for r in rows if r["line_surface"] == "alt_tail"]
    common_rows = [r for r in rows if r["line_surface"] != "alt_tail"]
    coverage_counter: Counter[str] = Counter()
    clv_status_counter: Counter[str] = Counter()
    clv_unknown_counter: Counter[str] = Counter()
    for r in rows:
        coverage_counter.update(r["match_methods"])
        clv_status_counter.update(r["clv_statuses"])
        clv_unknown_counter.update(r["clv_unknown_reasons"])
    blocker_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    for r in rows:
        blocker_counter.update(r["reasons"])
        category_counter.update(r["blocker_categories"])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "promotion_scope": "exact_bucket_only",
        "eligibility_start_date": cfg.eligibility_start_date.isoformat(),
        "lookback_days": cfg.lookback_days,
        "rows": int(len(df)),
        "legacy_audit_rows": int(len(legacy_df)),
        "bucket_count": len(rows),
        "promotable_count": sum(1 for r in rows if r["status"] == "promotable"),
        "thresholds": {
            "min_rows": cfg.min_rows,
            "min_clv_rows": cfg.min_clv_rows,
            "min_roi": cfg.min_roi,
            "min_clv_beat_rate": cfg.min_clv_beat_rate,
            "min_avg_clv_price": cfg.min_avg_clv_price,
            "max_abs_calibration_error": cfg.max_abs_calibration_error,
            "max_player_share": cfg.max_player_share,
            "max_team_share": cfg.max_team_share,
            "max_day_share": cfg.max_day_share,
            "min_unique_dates": cfg.min_unique_dates,
            "min_clean_unique_dates": cfg.min_clean_unique_dates,
        },
        "clv_match_methods": dict(coverage_counter.most_common()),
        "clv_status_counts": dict(clv_status_counter.most_common()),
        "clv_unknown_reason_counts": dict(clv_unknown_counter.most_common()),
        "blocker_counts": dict(blocker_counter.most_common()),
        "blocker_category_counts": dict(category_counter.most_common()),
        "closest_common_buckets": common_rows[: cfg.top_n],
        "closest_alt_line_buckets": alt_rows[: cfg.top_n],
        "all_buckets": rows,
    }


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(title for title, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in cols) + " |")
    return "\n".join(lines)


def _display_rows(rows: list[dict[str, Any]], *, min_clean_unique_dates: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        avg_clv = _fmt_num(row.get("avg_clv_price"), 2, signed=True)
        out.append({
            "bucket": row["key"],
            "status": row["status"],
            "ladder": row.get("ladder_tier", "watch"),
            "next_tier": row.get("desired_ladder_tier", "watch"),
            "source": row.get("promotion_source", "closed"),
            "graded": row["graded"],
            "need_rows": row["rows_needed"],
            "clv": f"{row['clv_rows']} / need {row['clv_rows_needed']}",
            "dates": f"{row['unique_dates']} / need {row['unique_dates_needed']}",
            "clean_dates": (
                "-"
                if row.get("holdout_unique_clean_dates") is None
                else f"{int(row.get('holdout_unique_clean_dates') or 0)} / need {min_clean_unique_dates}"
            ),
            "roi": _fmt_pct(row.get("roi"), signed=True),
            "clv_beat": _fmt_pct(row.get("clv_beat_rate")),
            "avg_clv": "-" if avg_clv == "-" else f"{avg_clv}pp",
            "cal": _fmt_pct(row.get("calibration_error"), signed=True),
            "blockers": "; ".join(row.get("reasons") or []),
            "gaps": "; ".join(
                list(row.get("metric_gaps") or [])
                + [
                    reason
                    for reason in row.get("bootstrap_micro_reasons") or []
                    if reason.startswith("bootstrap_")
                ][:4]
            ),
        })
    return out


def build_report(cfg: PromotionConfig) -> str:
    payload = build_payload(cfg)
    lines = [
        "# MLB Prop Bucket Promotion Report",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Lookback days: {payload['lookback_days']}",
        f"Eligibility start: {payload['eligibility_start_date']}",
        f"Eligible training rows: {payload['rows']}",
        f"Legacy audit rows: {payload['legacy_audit_rows']}",
        f"Exact buckets: {payload['bucket_count']}",
        f"Promotable buckets: {payload['promotable_count']}",
        "Scope: exact bucket only (market | side | line surface | line bucket | price bucket | book).",
        "",
        "## CLV Match Coverage",
        "",
        _table(
            [{"method": k, "rows": v} for k, v in payload["clv_match_methods"].items()],
            [("Method", "method"), ("Rows", "rows")],
        ),
        "",
        "## CLV Validity",
        "",
        _table(
            [{"status": k, "rows": v} for k, v in payload["clv_status_counts"].items()],
            [("Status", "status"), ("Rows", "rows")],
        ),
        "",
        "## CLV Unknown Reasons",
        "",
        _table(
            [{"reason": k, "rows": v} for k, v in payload["clv_unknown_reason_counts"].items()],
            [("Reason", "reason"), ("Rows", "rows")],
        ),
        "",
        "## Blocking Metrics",
        "",
        _table(
            [{"metric": k, "buckets": v} for k, v in payload["blocker_category_counts"].items()],
            [("Metric", "metric"), ("Blocks", "buckets")],
        ),
        "",
        "## Closest Common Buckets",
        "",
        _table(
            _display_rows(
                payload["closest_common_buckets"],
                min_clean_unique_dates=cfg.min_clean_unique_dates,
            ),
            [
                ("Bucket", "bucket"),
            ("Status", "status"),
            ("Ladder", "ladder"),
            ("Next", "next_tier"),
            ("Source", "source"),
            ("Graded", "graded"),
                ("Need Rows", "need_rows"),
                ("CLV Rows", "clv"),
                ("Dates", "dates"),
                ("Clean Dates", "clean_dates"),
                ("ROI", "roi"),
                ("CLV Beat", "clv_beat"),
                ("Avg CLV", "avg_clv"),
                ("Cal Err", "cal"),
                ("Metric Gaps", "gaps"),
            ],
        ),
        "",
        "## Alt-Line Lottery Watchlist",
        "",
        _table(
            _display_rows(
                payload["closest_alt_line_buckets"],
                min_clean_unique_dates=cfg.min_clean_unique_dates,
            ),
            [
                ("Bucket", "bucket"),
            ("Status", "status"),
            ("Ladder", "ladder"),
            ("Next", "next_tier"),
            ("Source", "source"),
            ("Graded", "graded"),
                ("Need Rows", "need_rows"),
                ("CLV Rows", "clv"),
                ("Dates", "dates"),
                ("Clean Dates", "clean_dates"),
                ("ROI", "roi"),
                ("CLV Beat", "clv_beat"),
                ("Avg CLV", "avg_clv"),
                ("Cal Err", "cal"),
                ("Metric Gaps", "gaps"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report exact MLB prop buckets closest to bankroll promotion")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument(
        "--eligibility-start-date",
        default=PROP_REAL_MONEY_ELIGIBILITY_START_DATE.isoformat(),
    )
    parser.add_argument("--min-rows", type=int, default=150)
    parser.add_argument("--min-clv-rows", type=int, default=30)
    parser.add_argument("--min-roi", type=float, default=0.0)
    parser.add_argument("--min-clv-beat-rate", type=float, default=0.55)
    parser.add_argument("--min-avg-clv-price", type=float, default=0.0)
    parser.add_argument("--max-abs-calibration-error", type=float, default=0.05)
    parser.add_argument("--max-player-share", type=float, default=0.12)
    parser.add_argument("--max-team-share", type=float, default=0.25)
    parser.add_argument("--max-day-share", type=float, default=0.35)
    parser.add_argument("--min-unique-dates", type=int, default=5)
    parser.add_argument("--min-clean-unique-dates", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=str(_MODEL_DIR / "prop_bucket_promotion_report.json"))
    args = parser.parse_args()
    cfg = PromotionConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        eligibility_start_date=parse_eligibility_start_date(args.eligibility_start_date),
        min_rows=args.min_rows,
        min_clv_rows=args.min_clv_rows,
        min_roi=args.min_roi,
        min_clv_beat_rate=args.min_clv_beat_rate,
        min_avg_clv_price=args.min_avg_clv_price,
        max_abs_calibration_error=args.max_abs_calibration_error,
        max_player_share=args.max_player_share,
        max_team_share=args.max_team_share,
        max_day_share=args.max_day_share,
        min_unique_dates=args.min_unique_dates,
        min_clean_unique_dates=args.min_clean_unique_dates,
        top_n=args.top_n,
        out=args.out,
        json_out=args.json_out,
    )
    payload = build_payload(cfg)
    if cfg.json_out:
        json_path = Path(cfg.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = build_report(cfg)
    if cfg.out:
        out_path = Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
