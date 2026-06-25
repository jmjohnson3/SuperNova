"""Rank MLB prop buckets by what is most repairable.

Promotion reports answer whether an exact bucket can open.  This report is
more operational: it shows why the closest buckets fail across ROI, CLV,
calibration, opportunity, and bookability, then ranks the ones most worth
debugging first.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from .prop_market_training import ensure_prop_market_training_schema
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

from mlb_pipeline.db import PG_DSN as _PG_DSN
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class RepairConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    lookback_days: int = 365
    min_rows: int = 150
    min_clv_rows: int = 30
    min_roi: float = 0.0
    min_clv_beat_rate: float = 0.55
    min_avg_clv_price: float = 0.0
    max_abs_calibration_error: float = 0.05
    min_bookability_rate: float = 0.70
    max_stale_close_rate: float = 0.10
    max_close_window_miss_rate: float = 0.10
    max_hitter_pa_mae: float = 0.80
    max_low_pa_miss_rate: float = 0.12
    max_pitcher_bf_mae: float = 4.50
    top_n: int = 40
    report_file: str = "mlb_prop_bucket_repair_latest.md"
    json_out: str = "prop_bucket_repair_report.json"


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
    COALESCE(pair_quality, 'unknown') AS pair_quality,
    COALESCE(market_prob_source, 'unknown') AS market_prob_source,
    COALESCE(same_book_pair_flag::float, CASE WHEN pair_quality = 'same_book' THEN 1.0 ELSE 0.0 END) AS same_book_pair_flag,
    COALESCE(cross_book_pair_flag::float, CASE WHEN pair_quality = 'cross_book' THEN 1.0 ELSE 0.0 END) AS cross_book_pair_flag,
    COALESCE(synthetic_pair_flag::float, CASE WHEN pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    COALESCE(
        clean_market_pair_flag::float,
        CASE
            WHEN pair_quality IN ('same_book', 'cross_book')
             AND COALESCE(market_prob_source, '') NOT IN ('raw_implied', 'synthetic_fanduel_over_only')
            THEN 1.0
            ELSE 0.0
        END
    ) AS clean_market_pair_flag,
    COALESCE(true_pair_flag::float, CASE WHEN pair_quality IN ('same_book', 'cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    model_prob_side::float AS model_prob_side,
    market_prob_side::float AS market_prob_side,
    ev::float AS ev,
    CASE WHEN won IS TRUE THEN 1 WHEN won IS FALSE THEN 0 ELSE NULL END AS won,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    COALESCE(clv_valid, false) AS clv_valid,
    clv_price::float AS clv_price,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price,
    clv_status,
    clv_unknown_reason,
    confirmed_batting_order::float AS confirmed_batting_order,
    projected_pa::float AS projected_pa,
    actual_pa::float AS actual_pa,
    projected_bf::float AS projected_bf,
    actual_bf::float AS actual_bf,
    projected_pitch_count::float AS projected_pitch_count,
    actual_pitch_count_proxy::float AS actual_pitch_count_proxy
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND side IN ('over','under')
  AND market_line IS NOT NULL
  AND won IS NOT NULL
ORDER BY game_date_et, market, side, bookmaker_key
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


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _rate(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def _brier(group: pd.DataFrame, prob_col: str) -> float | None:
    prob = pd.to_numeric(group[prob_col], errors="coerce")
    target = pd.to_numeric(group["won"], errors="coerce")
    mask = prob.notna() & target.notna()
    if not mask.any():
        return None
    err = prob[mask].clip(1e-6, 1.0 - 1e-6) - target[mask]
    return float((err * err).mean())


def _bucket_key(row: pd.Series | dict[str, Any]) -> str:
    return "|".join([
        str(row.get("market", "*")),
        str(row.get("side", "*")),
        str(row.get("line_surface", "*")),
        str(row.get("line_bucket", "*")),
        str(row.get("price_bucket", "*")),
        str(row.get("bookmaker_key", "*")),
    ])


def _load(cfg: RepairConfig) -> pd.DataFrame:
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
        "market_line", "market_price", "model_prob_side", "market_prob_side",
        "ev", "won", "profit_units", "clv_price", "beat_clv_price",
        "same_book_pair_flag", "cross_book_pair_flag", "synthetic_pair_flag",
        "clean_market_pair_flag", "true_pair_flag",
        "confirmed_batting_order", "projected_pa", "actual_pa",
        "projected_bf", "actual_bf", "projected_pitch_count", "actual_pitch_count_proxy",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    df["clv_valid"] = df["clv_valid"].fillna(False).astype(bool)
    df["line_bucket"] = [prop_line_bucket(m, l) for m, l in zip(df["market"], df["market_line"])]
    df["line_surface"] = [prop_line_surface(m, s, l) for m, s, l in zip(df["market"], df["side"], df["market_line"])]
    df["price_bucket"] = df["market_price"].map(price_bucket)
    df["bucket_key"] = df.apply(_bucket_key, axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


def _load_promotion(cfg: RepairConfig) -> dict[str, dict[str, Any]]:
    path = cfg.model_dir / "prop_bucket_promotion_report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}
    rows = payload.get("all_buckets") or []
    return {str(row.get("key")): dict(row) for row in rows if row.get("key")}


def _opportunity_metrics(group: pd.DataFrame) -> dict[str, Any]:
    market = str(group["market"].iloc[0])
    if market in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
        pred = pd.to_numeric(group["projected_pa"], errors="coerce")
        actual = pd.to_numeric(group["actual_pa"], errors="coerce")
        mask = pred.notna() & actual.notna()
        err = pred[mask] - actual[mask]
        low_pa = mask & (pred >= 3.8) & (actual <= 2.0)
        return {
            "opportunity_type": "hitter_pa",
            "opportunity_rows": int(mask.sum()),
            "opportunity_coverage": float(mask.mean()) if len(group) else None,
            "pa_mae": float(err.abs().mean()) if mask.any() else None,
            "pa_bias": float(err.mean()) if mask.any() else None,
            "low_pa_miss_rate": float(low_pa.sum() / mask.sum()) if mask.any() else None,
            "confirmed_lineup_rate": float(group["confirmed_batting_order"].notna().mean()) if len(group) else None,
        }
    if market == "pitcher_strikeouts":
        pred_bf = pd.to_numeric(group["projected_bf"], errors="coerce")
        actual_bf = pd.to_numeric(group["actual_bf"], errors="coerce")
        bf_mask = pred_bf.notna() & actual_bf.notna()
        pred_pc = pd.to_numeric(group["projected_pitch_count"], errors="coerce")
        actual_pc = pd.to_numeric(group["actual_pitch_count_proxy"], errors="coerce")
        pc_mask = pred_pc.notna() & actual_pc.notna()
        return {
            "opportunity_type": "pitcher_bf_pitch_count",
            "opportunity_rows": int(max(bf_mask.sum(), pc_mask.sum())),
            "opportunity_coverage": float((bf_mask | pc_mask).mean()) if len(group) else None,
            "bf_mae": float((pred_bf[bf_mask] - actual_bf[bf_mask]).abs().mean()) if bf_mask.any() else None,
            "bf_bias": float((pred_bf[bf_mask] - actual_bf[bf_mask]).mean()) if bf_mask.any() else None,
            "pitch_count_mae": float((pred_pc[pc_mask] - actual_pc[pc_mask]).abs().mean()) if pc_mask.any() else None,
        }
    return {"opportunity_type": "unknown"}


def _reason_category(reason: str) -> str:
    if reason.startswith("rows") or reason.startswith("clv_rows") or reason.startswith("unique_dates"):
        return "sample"
    if reason.startswith("bookability") or reason.startswith("stale") or reason.startswith("close_window"):
        return "bookability"
    if reason.startswith("clv_") or reason.startswith("avg_clv"):
        return "clv"
    if reason.startswith("roi"):
        return "roi"
    if reason.startswith("calibration"):
        return "calibration"
    if reason.startswith("opportunity") or reason.startswith("pa_") or reason.startswith("bf_") or reason.startswith("low_pa"):
        return "opportunity"
    if reason.startswith("tb_over") or reason.startswith("hitter_over"):
        return "distribution"
    if reason.startswith("true_pair") or reason.startswith("synthetic_pair") or reason == "market_beats_model_brier":
        return "market_evidence"
    return "other"


def _repair_action(reasons: list[str], rec: dict[str, Any]) -> str:
    cats = {_reason_category(reason) for reason in reasons}
    enough_rows = int(rec.get("graded") or 0) >= 150
    roi = _clean_float(rec.get("roi"))
    clv_beat = _clean_float(rec.get("clv_beat_rate"))
    if cats <= {"sample", "other"}:
        return "collect_more_clean_rows"
    if "bookability" in cats:
        return "repair_bookability_or_close_capture"
    if "market_evidence" in cats:
        return "repair_true_pair_coverage_or_market_model"
    if "opportunity" in cats:
        return "repair_opportunity_features"
    if "distribution" in cats:
        return "repair_hitter_tb_distribution"
    if "calibration" in cats:
        return "recalibrate_bucket_probability"
    if enough_rows and roi is not None and roi < -0.03 and clv_beat is not None and clv_beat < 0.45:
        return "likely_no_bet_bucket"
    if "clv" in cats:
        return "repair_residual_clv_scoring"
    if "roi" in cats:
        return "repair_edge_selection"
    return "inspect_bucket"


def _fixability_score(rec: dict[str, Any], reasons: list[str], cfg: RepairConfig) -> float:
    graded = int(rec.get("graded") or 0)
    clv_rows = int(rec.get("clv_rows") or 0)
    roi = _clean_float(rec.get("roi"))
    cal = _clean_float(rec.get("calibration_error"))
    avg_clv = _clean_float(rec.get("avg_clv_price"))
    clv_beat = _clean_float(rec.get("clv_beat_rate"))
    bookable = _clean_float(rec.get("bookability_rate"))
    true_pair = _clean_float(rec.get("true_pair_rate"))
    synthetic_pair = _clean_float(rec.get("synthetic_pair_rate"))
    brier_gap = _clean_float(rec.get("model_market_brier_gap"))
    score = 0.0
    score += min(25.0, 25.0 * graded / max(1, cfg.min_rows))
    score += min(15.0, 15.0 * clv_rows / max(1, cfg.min_clv_rows))
    if roi is not None:
        score += max(-25.0, min(25.0, (roi + 0.03) * 350.0))
    if cal is not None:
        score += max(-10.0, 12.0 - abs(cal) * 140.0)
    if avg_clv is not None:
        score += max(-12.0, min(12.0, avg_clv * 12.0 + 4.0))
    if clv_beat is not None:
        score += max(-12.0, min(12.0, (clv_beat - 0.45) * 80.0))
    if bookable is not None:
        score += max(-10.0, min(10.0, (bookable - 0.60) * 50.0))
    if true_pair is not None:
        score += max(-8.0, min(8.0, (true_pair - 0.50) * 20.0))
    if synthetic_pair is not None and synthetic_pair > 0.25:
        score -= min(12.0, synthetic_pair * 12.0)
    if brier_gap is not None and brier_gap > 0:
        score -= min(15.0, brier_gap * 180.0)
    if "opportunity" in {_reason_category(r) for r in reasons}:
        score += 4.0
    if "distribution" in {_reason_category(r) for r in reasons}:
        score += 3.0
    if int(rec.get("graded") or 0) >= cfg.min_rows and roi is not None and roi < -0.08:
        score -= 18.0
    if rec.get("line_surface") == "alt_tail":
        score -= 10.0
    score -= max(0, len(reasons) - 4) * 2.0
    return round(score, 3)


def _bucket_summary(key: str, group: pd.DataFrame, promotion: dict[str, Any], cfg: RepairConfig) -> dict[str, Any]:
    settled = group.loc[~group["push"]].copy()
    graded = int(len(settled))
    priced = group["profit_units"].notna()
    units = float(pd.to_numeric(group.loc[priced, "profit_units"], errors="coerce").sum()) if priced.any() else None
    roi = units / int(priced.sum()) if units is not None and int(priced.sum()) else None
    win_rate = _mean(settled["won"])
    avg_prob = _mean(settled["model_prob_side"])
    calibration_error = win_rate - avg_prob if win_rate is not None and avg_prob is not None else None
    valid_clv = group.loc[group["clv_valid"].fillna(False).astype(bool)].copy()
    clv_price = pd.to_numeric(valid_clv["clv_price"], errors="coerce").dropna()
    clv_beat = pd.to_numeric(valid_clv["beat_clv_price"], errors="coerce").dropna()
    unknown = group.loc[~group["clv_valid"].fillna(False).astype(bool), "clv_unknown_reason"].fillna("none").astype(str)
    pair_quality = group["pair_quality"].fillna("unknown").astype(str).str.lower()
    true_pair_rate = _mean(group["true_pair_flag"])
    same_book_pair_rate = _mean(group["same_book_pair_flag"])
    cross_book_pair_rate = _mean(group["cross_book_pair_flag"])
    synthetic_pair_rate = _mean(group["synthetic_pair_flag"])
    clean_market_pair_rate = _mean(group["clean_market_pair_flag"])
    model_brier = _brier(settled, "model_prob_side")
    market_brier = _brier(settled, "market_prob_side")
    model_market_brier_gap = (
        model_brier - market_brier
        if model_brier is not None and market_brier is not None
        else None
    )
    opp = _opportunity_metrics(group)

    reasons: list[str] = []
    if graded < cfg.min_rows:
        reasons.append(f"rows<{cfg.min_rows}")
    if len(clv_price) < cfg.min_clv_rows:
        reasons.append(f"clv_rows<{cfg.min_clv_rows}")
    if roi is None or roi <= cfg.min_roi:
        reasons.append(f"roi<={cfg.min_roi:.3f}")
    clv_beat_rate = float(clv_beat.mean()) if not clv_beat.empty else None
    if clv_beat_rate is None or clv_beat_rate < cfg.min_clv_beat_rate:
        reasons.append(f"clv_beat_rate<{cfg.min_clv_beat_rate:.2f}")
    avg_clv_price = float(clv_price.mean()) if not clv_price.empty else None
    if avg_clv_price is None or avg_clv_price <= cfg.min_avg_clv_price:
        reasons.append(f"avg_clv_price<={cfg.min_avg_clv_price:.3f}")
    if calibration_error is None or abs(calibration_error) > cfg.max_abs_calibration_error:
        reasons.append(f"calibration_abs>{cfg.max_abs_calibration_error:.3f}")
    if true_pair_rate is None or true_pair_rate < 0.50:
        reasons.append("true_pair_rate<0.50")
    if synthetic_pair_rate is not None and synthetic_pair_rate > 0.25:
        reasons.append("synthetic_pair_rate>0.25")
    if model_market_brier_gap is not None and model_market_brier_gap > 0.002:
        reasons.append("market_beats_model_brier")

    bookability_rate = float(group["clv_valid"].mean()) if len(group) else None
    stale_rate = _rate(unknown == "stale_close_before_lock")
    close_window_miss_rate = _rate(unknown == "close_outside_two_hour_window")
    if bookability_rate is None or bookability_rate < cfg.min_bookability_rate:
        reasons.append(f"bookability_rate<{cfg.min_bookability_rate:.2f}")
    if stale_rate is not None and stale_rate > cfg.max_stale_close_rate:
        reasons.append(f"stale_close_rate>{cfg.max_stale_close_rate:.2f}")
    if close_window_miss_rate is not None and close_window_miss_rate > cfg.max_close_window_miss_rate:
        reasons.append(f"close_window_miss_rate>{cfg.max_close_window_miss_rate:.2f}")

    market = str(group["market"].iloc[0])
    side = str(group["side"].iloc[0])
    if market in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
        pa_mae = _clean_float(opp.get("pa_mae"))
        low_pa_miss = _clean_float(opp.get("low_pa_miss_rate"))
        if pa_mae is not None and pa_mae > cfg.max_hitter_pa_mae:
            reasons.append(f"pa_mae>{cfg.max_hitter_pa_mae:.2f}")
        if low_pa_miss is not None and low_pa_miss > cfg.max_low_pa_miss_rate:
            reasons.append(f"low_pa_miss_rate>{cfg.max_low_pa_miss_rate:.2f}")
        if side == "over" and market == "batter_total_bases" and (roi is None or roi <= 0 or (calibration_error is not None and calibration_error < -0.03)):
            reasons.append("tb_over_distribution_repair")
        elif side == "over" and market in {"batter_hits", "batter_home_runs"} and (roi is None or roi <= 0):
            reasons.append("hitter_over_fake_ev_check")
    elif market == "pitcher_strikeouts":
        bf_mae = _clean_float(opp.get("bf_mae"))
        if bf_mae is not None and bf_mae > cfg.max_pitcher_bf_mae:
            reasons.append(f"bf_mae>{cfg.max_pitcher_bf_mae:.2f}")

    rec = {
        "key": key,
        "market": market,
        "side": side,
        "line_surface": str(group["line_surface"].iloc[0]),
        "line_bucket": str(group["line_bucket"].iloc[0]),
        "price_bucket": str(group["price_bucket"].iloc[0]),
        "bookmaker_key": str(group["bookmaker_key"].iloc[0]),
        "graded": graded,
        "rows_needed": max(0, cfg.min_rows - graded),
        "clv_rows": int(len(clv_price)),
        "clv_rows_needed": max(0, cfg.min_clv_rows - int(len(clv_price))),
        "unique_dates": int(group["game_date_et"].nunique()),
        "win_rate": win_rate,
        "roi": roi,
        "units": units,
        "avg_model_prob": avg_prob,
        "calibration_error": calibration_error,
        "true_pair_rate": true_pair_rate,
        "same_book_pair_rate": same_book_pair_rate,
        "cross_book_pair_rate": cross_book_pair_rate,
        "synthetic_pair_rate": synthetic_pair_rate,
        "clean_market_pair_rate": clean_market_pair_rate,
        "pair_quality_counts": dict(Counter(pair_quality).most_common()),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "model_market_brier_gap": model_market_brier_gap,
        "clv_beat_rate": clv_beat_rate,
        "avg_clv_price": avg_clv_price,
        "bookability_rate": bookability_rate,
        "stale_close_rate": stale_rate,
        "close_window_miss_rate": close_window_miss_rate,
        "clv_unknown_reasons": dict(Counter(unknown).most_common()),
        **opp,
        "promotion_reasons": list(promotion.get("reasons") or []),
        "metric_gaps": list(promotion.get("metric_gaps") or []),
        "bootstrap_micro_reasons": list(promotion.get("bootstrap_micro_reasons") or []),
    }
    reasons = sorted(dict.fromkeys(reasons + rec["promotion_reasons"]))
    rec["repair_reasons"] = reasons
    rec["repair_categories"] = dict(Counter(_reason_category(reason) for reason in reasons))
    rec["repair_action"] = _repair_action(reasons, rec)
    rec["fixability_score"] = _fixability_score(rec, reasons, cfg)
    return rec


def build_payload(cfg: RepairConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    promotion = _load_promotion(cfg)
    rows: list[dict[str, Any]] = []
    if not df.empty:
        for key, group in df.groupby("bucket_key", dropna=False):
            rows.append(_bucket_summary(str(key), group, promotion.get(str(key), {}), cfg))
    rows.sort(key=lambda rec: (-float(rec.get("fixability_score") or -999.0), -int(rec.get("graded") or 0), rec["key"]))
    common = [r for r in rows if r.get("line_surface") != "alt_tail"]
    alt = [r for r in rows if r.get("line_surface") == "alt_tail"]
    hitter_tb_over = [
        r for r in rows
        if r.get("market") in {"batter_hits", "batter_total_bases", "batter_home_runs"} and r.get("side") == "over"
    ]
    bookability = [
        r for r in rows
        if "bookability" in (r.get("repair_categories") or {}) or "clv" in (r.get("repair_categories") or {})
    ]
    market_evidence = [
        r for r in rows
        if "market_evidence" in (r.get("repair_categories") or {})
    ]
    no_bet = [
        r for r in rows
        if r.get("repair_action") == "likely_no_bet_bucket"
    ]
    action_counts = Counter(str(r.get("repair_action")) for r in rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if rows else "no_rows",
        "source": "features.mlb_prop_market_training_examples",
        "lookback_days": cfg.lookback_days,
        "rows": int(len(df)),
        "bucket_count": len(rows),
        "thresholds": {
            "min_rows": cfg.min_rows,
            "min_clv_rows": cfg.min_clv_rows,
            "min_roi": cfg.min_roi,
            "min_clv_beat_rate": cfg.min_clv_beat_rate,
            "min_avg_clv_price": cfg.min_avg_clv_price,
            "max_abs_calibration_error": cfg.max_abs_calibration_error,
            "min_bookability_rate": cfg.min_bookability_rate,
            "max_stale_close_rate": cfg.max_stale_close_rate,
            "max_hitter_pa_mae": cfg.max_hitter_pa_mae,
            "max_low_pa_miss_rate": cfg.max_low_pa_miss_rate,
        },
        "repair_action_counts": dict(action_counts.most_common()),
        "most_fixable_common": common[: cfg.top_n],
        "most_fixable_alt": alt[: cfg.top_n],
        "hitter_tb_over_repair": hitter_tb_over[: cfg.top_n],
        "bookability_clv_repair": bookability[: cfg.top_n],
        "market_evidence_repair": market_evidence[: cfg.top_n],
        "likely_no_bet_buckets": no_bet[: cfg.top_n],
        "all_buckets": rows,
    }


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


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(title for title, _ in cols) + " |",
        "| " + " | ".join("---" for _title, _key in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _title, key in cols) + " |")
    return "\n".join(lines)


def _display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        opportunity = "-"
        if row.get("opportunity_type") == "hitter_pa":
            opportunity = (
                f"PA MAE {_fmt_num(row.get('pa_mae'))}, "
                f"low-PA miss {_fmt_pct(row.get('low_pa_miss_rate'))}"
            )
        elif row.get("opportunity_type") == "pitcher_bf_pitch_count":
            opportunity = (
                f"BF MAE {_fmt_num(row.get('bf_mae'))}, "
                f"PC MAE {_fmt_num(row.get('pitch_count_mae'))}"
            )
        gaps = list(row.get("metric_gaps") or [])
        if not gaps:
            gaps = list(row.get("repair_reasons") or [])[:8]
        out.append({
            "bucket": row.get("key"),
            "score": _fmt_num(row.get("fixability_score"), 1),
            "action": row.get("repair_action"),
            "graded": row.get("graded"),
            "roi": _fmt_pct(row.get("roi"), signed=True),
            "clv_beat": _fmt_pct(row.get("clv_beat_rate")),
            "avg_clv": _fmt_num(row.get("avg_clv_price"), 2, signed=True),
            "cal": _fmt_pct(row.get("calibration_error"), signed=True),
            "true_pair": _fmt_pct(row.get("true_pair_rate")),
            "synthetic": _fmt_pct(row.get("synthetic_pair_rate")),
            "brier_gap": _fmt_num(row.get("model_market_brier_gap"), 3, signed=True),
            "bookable": _fmt_pct(row.get("bookability_rate")),
            "stale": _fmt_pct(row.get("stale_close_rate")),
            "opp": opportunity,
            "why": "; ".join(gaps[:8]),
        })
    return out


def write_report(payload: dict[str, Any], cfg: RepairConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    if payload.get("status") != "ready":
        path.write_text(
            "# MLB Prop Bucket Repair Report\n\nNo graded prop buckets were available.\n",
            encoding="utf-8",
        )
        return str(path)
    cols = [
        ("Bucket", "bucket"),
        ("Score", "score"),
        ("Action", "action"),
        ("Rows", "graded"),
        ("ROI", "roi"),
        ("CLV Beat", "clv_beat"),
        ("Avg CLV", "avg_clv"),
        ("Cal", "cal"),
        ("True Pair", "true_pair"),
        ("Synthetic", "synthetic"),
        ("Brier Gap", "brier_gap"),
        ("Bookable", "bookable"),
        ("Stale", "stale"),
        ("Opportunity", "opp"),
        ("Why", "why"),
    ]
    lines = [
        "# MLB Prop Bucket Repair Report",
        "",
        f"- Generated UTC: {payload['generated_at_utc']}",
        f"- Source: {payload['source']}",
        f"- Lookback days: {payload['lookback_days']}",
        f"- Rows: {payload['rows']}",
        f"- Exact buckets: {payload['bucket_count']}",
        "",
        "This report does not reopen bankroll props. It ranks exact buckets by what is most repairable: ROI, CLV, calibration, true-pair market evidence, opportunity, and bookability/close capture.",
        "",
        "## Repair Action Counts",
        "",
        _table(
            [{"action": k, "buckets": v} for k, v in payload.get("repair_action_counts", {}).items()],
            [("Action", "action"), ("Buckets", "buckets")],
        ),
        "",
        "## Most Fixable Common Buckets",
        "",
        _table(_display_rows(payload.get("most_fixable_common") or []), cols),
        "",
        "## Hitter/TB Over Repair Focus",
        "",
        _table(_display_rows(payload.get("hitter_tb_over_repair") or []), cols),
        "",
        "## Bookability And CLV Repair Focus",
        "",
        _table(_display_rows(payload.get("bookability_clv_repair") or []), cols),
        "",
        "## Market Evidence Repair Focus",
        "",
        _table(_display_rows(payload.get("market_evidence_repair") or []), cols),
        "",
        "## Alt-Line Lottery Repair Watchlist",
        "",
        _table(_display_rows(payload.get("most_fixable_alt") or []), cols),
        "",
        "## Likely No-Bet Buckets",
        "",
        _table(_display_rows(payload.get("likely_no_bet_buckets") or []), cols),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop bucket repair report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--report-file", default="mlb_prop_bucket_repair_latest.md")
    parser.add_argument("--json-out", default="prop_bucket_repair_report.json")
    args = parser.parse_args()
    cfg = RepairConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        top_n=args.top_n,
        report_file=args.report_file,
        json_out=args.json_out,
    )
    payload = build_payload(cfg)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / cfg.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(payload, cfg)
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "bucket_count": payload.get("bucket_count", 0),
        "report_path": report_path,
    }, indent=2))


if __name__ == "__main__":
    main()
