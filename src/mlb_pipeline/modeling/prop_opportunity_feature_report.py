"""Validate whether prop opportunity features improve holdout accuracy.

This report compares two simple side-probability evaluators:

* baseline: locked model/market/line/price/book features
* opportunity: baseline plus projected PA/BF/pitch-count and lineup slot

It is diagnostic only. Promotion still requires exact-bucket ROI, CLV,
calibration, concentration, and clean-slate evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .prop_market_training import PropMarketTrainingConfig, refresh_prop_market_training_examples
from .prop_replay import ev_per_unit
from .side_recalibration import prop_line_surface

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_BASE_NUMERIC = [
    "model_prob_side",
    "market_prob_side",
    "prob_edge_vs_market",
    "market_line",
    "abs_price",
    "is_plus_price",
    "count_edge_side",
    "pred_count",
    "pred_value",
]

_OPPORTUNITY_NUMERIC = [
    "confirmed_batting_order",
    "projected_pa",
    "projected_bf",
    "projected_pitch_count",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_team_k_pct_10",
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
    "batter_vs_rp_slg_30",
    "batter_vs_rp_hr_rate_30",
    "pinch_hit_risk",
]

_CATEGORICAL = [
    "market",
    "side",
    "line_surface",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
    "model_family",
    "edge_type",
    "opp_sp_hand",
]


@dataclass(frozen=True)
class OpportunityReportConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_opportunity_feature_report.json"
    report_file: str | None = None
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 150
    min_holdout_rows: int = 40
    min_selected_rows: int = 10
    min_brier_gain: float = 0.001
    min_ev: float = 0.02
    rebuild_training_if_empty: bool = True


SQL = """
SELECT
    replay_id,
    game_date_et,
    market,
    side,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(model_family, 'unknown') AS model_family,
    COALESCE(edge_type, 'unknown') AS edge_type,
    market_line::float AS market_line,
    market_price::float AS market_price,
    ABS(market_price::float) AS abs_price,
    CASE
        WHEN market_price IS NULL THEN NULL
        WHEN market_price::float > 0 THEN 1.0
        ELSE 0.0
    END AS is_plus_price,
    model_prob_side::float AS model_prob_side,
    market_prob_side::float AS market_prob_side,
    prob_edge_vs_market::float AS prob_edge_vs_market,
    count_edge_side::float AS count_edge_side,
    pred_count::float AS pred_count,
    pred_value::float AS pred_value,
    confirmed_batting_order::float AS confirmed_batting_order,
    projected_pa::float AS projected_pa,
    projected_bf::float AS projected_bf,
    projected_pitch_count::float AS projected_pitch_count,
    is_home::float AS is_home,
    team_implied_runs::float AS team_implied_runs,
    opponent_implied_runs::float AS opponent_implied_runs,
    game_total_line::float AS game_total_line,
    opp_sp_hand,
    opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    opp_sp_bb_pct::float AS opp_sp_bb_pct,
    opp_sp_xwoba::float AS opp_sp_xwoba,
    opp_sp_hard_hit_pct::float AS opp_sp_hard_hit_pct,
    opp_sp_whiff_pct::float AS opp_sp_whiff_pct,
    opp_bp_era_10::float AS opp_bp_era_10,
    opp_bp_whip_10::float AS opp_bp_whip_10,
    opp_bp_k9_10::float AS opp_bp_k9_10,
    opp_team_k_pct_10::float AS opp_team_k_pct_10,
    batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    batter_vs_hand_iso_avg_10::float AS batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10::float AS batter_vs_hand_k_rate_10,
    batter_vs_rp_slg_30::float AS batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30::float AS batter_vs_rp_hr_rate_30,
    pinch_hit_risk::float AS pinch_hit_risk,
    actual_pa::float AS actual_pa,
    actual_bf::float AS actual_bf,
    actual_pitch_count_proxy::float AS actual_pitch_count_proxy,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS target,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    CASE WHEN clv_valid IS TRUE THEN clv_price::float ELSE NULL END AS clv_price,
    CASE
        WHEN clv_valid IS TRUE AND beat_clv_price IS TRUE THEN 1
        WHEN clv_valid IS TRUE AND beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND model_prob_side IS NOT NULL
  AND market_line IS NOT NULL
  AND won IS NOT NULL
  AND model_prob_side BETWEEN 0.0 AND 1.0
ORDER BY game_date_et, replay_id, side
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


def _table_row_count(conn, schema: str, table: str) -> int:
    if not _table_exists(conn, schema, table):
        return 0
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        return int(cur.fetchone()[0] or 0)


def _query_df(conn, sql: str, params: dict[str, object]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=columns)


def _load(cfg: OpportunityReportConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if _table_row_count(conn, "features", "mlb_prop_market_training_examples") == 0:
            if not cfg.rebuild_training_if_empty:
                return pd.DataFrame()
            refresh_prop_market_training_examples(PropMarketTrainingConfig(
                pg_dsn=cfg.pg_dsn,
                lookback_days=cfg.lookback_days,
                include_pending=False,
            ))
            if _table_row_count(conn, "features", "mlb_prop_market_training_examples") == 0:
                return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = (
        _BASE_NUMERIC
        + _OPPORTUNITY_NUMERIC
        + [
            "market_price", "target", "profit_units", "clv_price", "beat_clv_price",
            "actual_pa", "actual_bf", "actual_pitch_count_proxy",
        ]
    )
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["line_surface"] = [
        prop_line_surface(market, side, line)
        for market, side, line in zip(df["market"], df["side"], df["market_line"])
    ]
    df["target"] = df["target"].astype(int)
    df["push"] = df["push"].fillna(False).astype(bool)
    for col in _CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)
    return df.replace([np.inf, -np.inf], np.nan)


def _prepare_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    *,
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    categories: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, float], dict[str, list[str]]]:
    means = dict(means or {})
    scales = dict(scales or {})
    categories = {name: list(values) for name, values in (categories or {}).items()}
    parts: list[np.ndarray] = []
    numeric_means: dict[str, float] = {}
    numeric_scales: dict[str, float] = {}

    for name in numeric_features:
        series = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, series.mean() if not series.dropna().empty else 0.0))
        filled = series.fillna(mean)
        std = float(filled.std(ddof=0) or 0.0)
        scale = float(scales.get(name, std if std > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy(dtype=float).reshape(-1, 1))
        numeric_means[name] = mean
        numeric_scales[name] = scale

    cat_values: dict[str, list[str]] = {}
    for name in _CATEGORICAL:
        values = categories.get(name)
        if values is None:
            values = sorted(str(v) for v in df[name].fillna("unknown").astype(str).unique())
        cat_values[name] = values
        series = df[name].fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))

    if not parts:
        return np.zeros((len(df), 0)), numeric_means, numeric_scales, cat_values
    return np.hstack(parts), numeric_means, numeric_scales, cat_values


def _split(df: pd.DataFrame, cfg: OpportunityReportConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train = df.loc[df["game_date_et"] < split].copy()
    holdout = df.loc[df["game_date_et"] >= split].copy()
    if len(train) >= cfg.min_train_rows and len(holdout) >= cfg.min_holdout_rows:
        return train, holdout, f"last_{cfg.holdout_days}_days"
    dates = sorted(df["game_date_et"].unique())
    if len(dates) > 1:
        holdout_date = dates[-1]
        return (
            df.loc[df["game_date_et"] < holdout_date].copy(),
            df.loc[df["game_date_et"] >= holdout_date].copy(),
            "last_available_date",
        )
    return train, holdout, f"last_{cfg.holdout_days}_days"


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _selected_summary(holdout: pd.DataFrame, probs: np.ndarray, cfg: OpportunityReportConfig) -> dict[str, Any]:
    work = holdout.copy()
    work["candidate_prob"] = np.clip(probs, 1e-6, 1 - 1e-6)
    work["candidate_ev"] = [
        ev_per_unit(prob, price)
        for prob, price in zip(work["candidate_prob"], work["market_price"])
    ]
    work = work.dropna(subset=["candidate_ev", "profit_units"])
    if work.empty:
        return {"picks": 0, "roi": None, "win_rate": None, "avg_ev": None, "avg_clv_price": None, "clv_beat_rate": None}
    selected = []
    for _, sub in work.groupby("replay_id", dropna=False):
        idx = pd.to_numeric(sub["candidate_ev"], errors="coerce").idxmax()
        row = work.loc[idx]
        if float(row["candidate_ev"]) >= cfg.min_ev:
            selected.append(row)
    if not selected:
        return {"picks": 0, "roi": None, "win_rate": None, "avg_ev": None, "avg_clv_price": None, "clv_beat_rate": None}
    picks = pd.DataFrame(selected)
    non_push = picks.loc[~picks["push"]]
    clv_mask = picks["clv_price"].notna()
    return {
        "picks": int(len(picks)),
        "roi": _mean(picks["profit_units"]),
        "win_rate": _mean(non_push["target"]) if not non_push.empty else None,
        "avg_ev": _mean(picks["candidate_ev"]),
        "avg_clv_price": _mean(picks.loc[clv_mask, "clv_price"]) if clv_mask.any() else None,
        "clv_beat_rate": _mean(picks.loc[clv_mask, "beat_clv_price"]) if clv_mask.any() else None,
    }


def _coverage(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"projected_pa_rate": None, "projected_bf_rate": None, "pitch_count_rate": None, "batting_order_rate": None}
    return {
        "projected_pa_rate": float(df["projected_pa"].notna().mean()),
        "projected_bf_rate": float(df["projected_bf"].notna().mean()),
        "pitch_count_rate": float(df["projected_pitch_count"].notna().mean()),
        "batting_order_rate": float(df["confirmed_batting_order"].notna().mean()),
    }


def _projection_metric(
    df: pd.DataFrame,
    *,
    key: str,
    label: str,
    pred_col: str,
    actual_col: str,
) -> dict[str, Any]:
    if df.empty:
        return {
            "key": key,
            "label": label,
            "rows": 0,
            "coverage": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "pred_avg": None,
            "actual_avg": None,
        }
    pred = pd.to_numeric(df.get(pred_col), errors="coerce")
    actual = pd.to_numeric(df.get(actual_col), errors="coerce")
    mask = pred.notna() & actual.notna()
    if not mask.any():
        return {
            "key": key,
            "label": label,
            "rows": int(len(df)),
            "coverage": float(pred.notna().mean()) if len(df) else None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "pred_avg": None,
            "actual_avg": None,
        }
    err = pred[mask] - actual[mask]
    return {
        "key": key,
        "label": label,
        "rows": int(mask.sum()),
        "coverage": float(mask.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "pred_avg": float(pred[mask].mean()),
        "actual_avg": float(actual[mask].mean()),
    }


def _projection_accuracy(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hitter = df.loc[df["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])].copy()
    if not hitter.empty:
        rows.append(_projection_metric(
            hitter,
            key="hitters|all|pa",
            label="Hitters PA",
            pred_col="projected_pa",
            actual_col="actual_pa",
        ))
        for market, group in hitter.groupby("market", dropna=False):
            rows.append(_projection_metric(
                group,
                key=f"{market}|pa",
                label=f"{market} PA",
                pred_col="projected_pa",
                actual_col="actual_pa",
            ))
    pitcher = df.loc[df["market"] == "pitcher_strikeouts"].copy()
    if not pitcher.empty:
        rows.append(_projection_metric(
            pitcher,
            key="pitcher_strikeouts|bf",
            label="Pitcher BF",
            pred_col="projected_bf",
            actual_col="actual_bf",
        ))
        rows.append(_projection_metric(
            pitcher,
            key="pitcher_strikeouts|pitch_count",
            label="Pitch Count Proxy",
            pred_col="projected_pitch_count",
            actual_col="actual_pitch_count_proxy",
        ))
    rows.sort(key=lambda row: (row.get("label") or ""))
    return rows


def _batting_order_bucket(value: Any) -> str:
    try:
        order = int(float(value))
    except (TypeError, ValueError):
        return "order_unknown"
    if order <= 2:
        return "order_1_2"
    if order <= 5:
        return "order_3_5"
    if order <= 9:
        return "order_6_9"
    return "order_unknown"


def _pa_bucket(value: Any) -> str:
    try:
        pa = float(value)
    except (TypeError, ValueError):
        return "projected_pa_unknown"
    if not math.isfinite(pa):
        return "projected_pa_unknown"
    if pa < 3.2:
        return "projected_pa_under_3_2"
    if pa < 3.8:
        return "projected_pa_3_2_to_3_7"
    if pa < 4.4:
        return "projected_pa_3_8_to_4_3"
    return "projected_pa_4_4_plus"


def _pinch_bucket(value: Any) -> str:
    try:
        risk = float(value)
    except (TypeError, ValueError):
        return "pinch_unknown"
    if not math.isfinite(risk):
        return "pinch_unknown"
    if risk >= 0.30:
        return "pinch_high"
    if risk >= 0.15:
        return "pinch_medium"
    return "pinch_low"


def _opportunity_breakdown_summary(group: pd.DataFrame) -> dict[str, Any]:
    pred = pd.to_numeric(group.get("projected_pa"), errors="coerce")
    actual = pd.to_numeric(group.get("actual_pa"), errors="coerce")
    mask = pred.notna() & actual.notna()
    err = pred[mask] - actual[mask]
    low_pa_mask = mask & (actual <= 2.0)
    low_pa_miss_mask = mask & (pred >= 3.8) & (actual <= 2.0)
    clv = group.loc[group["beat_clv_price"].notna()]
    return {
        "rows": int(len(group)),
        "projected_pa_coverage": float(pred.notna().mean()) if len(group) else None,
        "actual_pa_coverage": float(actual.notna().mean()) if len(group) else None,
        "avg_projected_pa": float(pred[mask].mean()) if mask.any() else None,
        "avg_actual_pa": float(actual[mask].mean()) if mask.any() else None,
        "pa_mae": float(err.abs().mean()) if mask.any() else None,
        "pa_bias": float(err.mean()) if mask.any() else None,
        "actual_low_pa_rate": float(low_pa_mask.sum() / mask.sum()) if mask.any() else None,
        "low_pa_miss_rate": float(low_pa_miss_mask.sum() / mask.sum()) if mask.any() else None,
        "roi": _mean(group["profit_units"]),
        "clv_beat_rate": _mean(clv["beat_clv_price"]) if not clv.empty else None,
    }


def _hitter_opportunity_breakdowns(df: pd.DataFrame) -> list[dict[str, Any]]:
    hitter = df.loc[df["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])].copy()
    if hitter.empty:
        return []
    hitter["lineup_confirmed_bucket"] = np.where(
        hitter["confirmed_batting_order"].notna(),
        "confirmed_lineup",
        "unconfirmed_lineup",
    )
    hitter["batting_order_bucket"] = hitter["confirmed_batting_order"].map(_batting_order_bucket)
    hitter["projected_pa_bucket"] = hitter["projected_pa"].map(_pa_bucket)
    hitter["pinch_hit_risk_bucket"] = hitter["pinch_hit_risk"].map(_pinch_bucket)

    specs = [
        ("confirmed_lineup", ["lineup_confirmed_bucket"]),
        ("market_confirmed_lineup", ["market", "lineup_confirmed_bucket"]),
        ("batting_order", ["batting_order_bucket"]),
        ("projected_pa", ["projected_pa_bucket"]),
        ("pinch_hit_risk", ["pinch_hit_risk_bucket"]),
    ]
    rows: list[dict[str, Any]] = []
    for level, cols in specs:
        for key, group in hitter.groupby(cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            rec = {
                "level": level,
                "bucket": "|".join(str(v) for v in key_tuple),
                **_opportunity_breakdown_summary(group),
            }
            rows.append(rec)
    rows.sort(key=lambda rec: (rec["level"], -int(rec["rows"]), rec["bucket"]))
    return rows


def _fit_variant(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if train["target"].nunique() < 2:
        return None, {"status": "skipped", "reason": "one_class_train"}
    if holdout["target"].nunique() < 2:
        return None, {"status": "skipped", "reason": "one_class_holdout"}
    X_train, means, scales, categories = _prepare_matrix(train, numeric_features)
    y_train = train["target"].astype(int).to_numpy()
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X_train, y_train)
    X_hold, _, _, _ = _prepare_matrix(
        holdout,
        numeric_features,
        means=means,
        scales=scales,
        categories=categories,
    )
    probs = np.clip(model.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    y_hold = holdout["target"].astype(int).to_numpy()
    return probs, {
        "status": "ready",
        "rows": int(len(holdout)),
        "actual_rate": float(np.mean(y_hold)),
        "avg_prob": float(np.mean(probs)),
        "calibration_error": float(np.mean(y_hold) - np.mean(probs)),
        "brier": float(brier_score_loss(y_hold, probs)),
        "log_loss": float(log_loss(y_hold, probs, labels=[0, 1])),
    }


def _compare_group(df: pd.DataFrame, cfg: OpportunityReportConfig) -> dict[str, Any]:
    train, holdout, split_method = _split(df.loc[~df["push"]].copy(), cfg)
    record: dict[str, Any] = {
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_method": split_method,
        "coverage": _coverage(df),
        "baseline": {},
        "opportunity": {},
        "decision": "insufficient_rows",
        "reasons": [],
    }
    if len(train) < cfg.min_train_rows:
        record["reasons"].append(f"train_rows<{cfg.min_train_rows}")
    if len(holdout) < cfg.min_holdout_rows:
        record["reasons"].append(f"holdout_rows<{cfg.min_holdout_rows}")
    if record["reasons"]:
        return record

    base_probs, base_forecast = _fit_variant(train, holdout, _BASE_NUMERIC)
    opp_probs, opp_forecast = _fit_variant(train, holdout, _BASE_NUMERIC + _OPPORTUNITY_NUMERIC)
    record["baseline"]["forecast"] = base_forecast
    record["opportunity"]["forecast"] = opp_forecast
    if base_probs is None or opp_probs is None:
        record["decision"] = "skipped"
        record["reasons"].extend([
            base_forecast.get("reason") or "",
            opp_forecast.get("reason") or "",
        ])
        record["reasons"] = [reason for reason in record["reasons"] if reason]
        return record

    record["baseline"]["selection"] = _selected_summary(holdout, base_probs, cfg)
    record["opportunity"]["selection"] = _selected_summary(holdout, opp_probs, cfg)
    base_brier = base_forecast.get("brier")
    opp_brier = opp_forecast.get("brier")
    brier_gain = (base_brier - opp_brier) if base_brier is not None and opp_brier is not None else None
    record["brier_gain"] = brier_gain

    reasons: list[str] = []
    if brier_gain is None or brier_gain < cfg.min_brier_gain:
        reasons.append("brier_not_improved")
    base_sel = record["baseline"]["selection"]
    opp_sel = record["opportunity"]["selection"]
    if int(opp_sel.get("picks") or 0) < cfg.min_selected_rows:
        reasons.append("selected_rows_too_small")
    if int(base_sel.get("picks") or 0) >= cfg.min_selected_rows and int(opp_sel.get("picks") or 0) >= cfg.min_selected_rows:
        if base_sel.get("roi") is not None and opp_sel.get("roi") is not None and opp_sel["roi"] < base_sel["roi"]:
            reasons.append("roi_worse")
        if (
            base_sel.get("avg_clv_price") is not None
            and opp_sel.get("avg_clv_price") is not None
            and opp_sel["avg_clv_price"] < base_sel["avg_clv_price"]
        ):
            reasons.append("avg_clv_worse")
        if (
            base_sel.get("clv_beat_rate") is not None
            and opp_sel.get("clv_beat_rate") is not None
            and opp_sel["clv_beat_rate"] < base_sel["clv_beat_rate"]
        ):
            reasons.append("clv_beat_worse")
    record["decision"] = "opportunity_helped" if not reasons else "keep_baseline"
    record["reasons"] = reasons
    return record


def _group_rows(df: pd.DataFrame, cfg: OpportunityReportConfig, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values, group in df.groupby(group_cols, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        rec = dict(zip(group_cols, values))
        rec["key"] = "|".join(str(rec.get(col, "*")) for col in group_cols)
        rec.update(_compare_group(group, cfg))
        rows.append(rec)
    rows.sort(key=lambda r: (-int(r.get("holdout_rows") or 0), r["key"]))
    return rows


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _variant_values(rec: dict[str, Any], variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = rec.get(variant) or {}
    return data.get("forecast") or {}, data.get("selection") or {}


def _table(rows: list[dict[str, Any]], top_n: int = 40) -> list[str]:
    lines = [
        "| Bucket | Rows | Train | Holdout | Decision | Brier Gain | Base Brier | Opp Brier | Base ROI | Opp ROI | Base CLV | Opp CLV | PA Cov | BF Cov | Reasons |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rec in rows[:top_n]:
        base_f, base_s = _variant_values(rec, "baseline")
        opp_f, opp_s = _variant_values(rec, "opportunity")
        cov = rec.get("coverage") or {}
        lines.append(
            f"| {rec['key']} | {rec.get('rows', 0)} | {rec.get('train_rows', 0)} | {rec.get('holdout_rows', 0)} | "
            f"{rec.get('decision')} | {_fmt_num(rec.get('brier_gain'), signed=True)} | "
            f"{_fmt_num(base_f.get('brier'))} | {_fmt_num(opp_f.get('brier'))} | "
            f"{_fmt_pct(base_s.get('roi'), signed=True)} | {_fmt_pct(opp_s.get('roi'), signed=True)} | "
            f"{_fmt_num(base_s.get('avg_clv_price'), 2, signed=True)} | {_fmt_num(opp_s.get('avg_clv_price'), 2, signed=True)} | "
            f"{_fmt_pct(cov.get('projected_pa_rate'))} | {_fmt_pct(cov.get('projected_bf_rate'))} | "
            f"{'; '.join(rec.get('reasons') or [])} |"
        )
    return lines


def _projection_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Metric | Rows | Coverage | MAE | RMSE | Bias | Pred Avg | Actual Avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in rows:
        lines.append(
            f"| {rec.get('label')} | {rec.get('rows', 0)} | {_fmt_pct(rec.get('coverage'))} | "
            f"{_fmt_num(rec.get('mae'))} | {_fmt_num(rec.get('rmse'))} | "
            f"{_fmt_num(rec.get('bias'), signed=True)} | {_fmt_num(rec.get('pred_avg'))} | "
            f"{_fmt_num(rec.get('actual_avg'))} |"
        )
    return lines


def _opportunity_breakdown_table(rows: list[dict[str, Any]], top_n: int = 80) -> list[str]:
    lines = [
        "| Level | Bucket | Rows | PA Cov | PA MAE | PA Bias | Avg PA | Actual PA | Low-PA | Low-PA Miss | ROI | CLV Beat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in rows[:top_n]:
        lines.append(
            f"| {rec.get('level')} | {rec.get('bucket')} | {rec.get('rows', 0)} | "
            f"{_fmt_pct(rec.get('projected_pa_coverage'))} | {_fmt_num(rec.get('pa_mae'))} | "
            f"{_fmt_num(rec.get('pa_bias'), signed=True)} | {_fmt_num(rec.get('avg_projected_pa'), 2)} | "
            f"{_fmt_num(rec.get('avg_actual_pa'), 2)} | {_fmt_pct(rec.get('actual_low_pa_rate'))} | "
            f"{_fmt_pct(rec.get('low_pa_miss_rate'))} | {_fmt_pct(rec.get('roi'), signed=True)} | "
            f"{_fmt_pct(rec.get('clv_beat_rate'))} |"
        )
    return lines


def _write_report(payload: dict[str, Any], cfg: OpportunityReportConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_name = cfg.report_file or "mlb_prop_opportunity_feature_latest.md"
    path = _REPORT_DIR / report_name
    if payload.get("status") != "ready":
        path.write_text(
            "# MLB Prop Opportunity Feature Report\n\n"
            "No graded prop market-training rows were available. "
            "The report tried to rebuild `features.mlb_prop_market_training_examples` first.\n",
            encoding="utf-8",
        )
        return str(path)
    lines = [
        "# MLB Prop Opportunity Feature Report",
        "",
        f"- Generated UTC: {payload['generated_at_utc']}",
        f"- Source: {payload['source']}",
        f"- Date range: {payload['date_min']} to {payload['date_max']}",
        f"- Rows: {payload['rows']}",
        f"- Unique dates: {payload['unique_dates']}",
        f"- Holdout days: {payload['holdout_days']}",
        f"- Minimum Brier gain: {cfg.min_brier_gain:.3f}",
        "",
        "This is a holdout diagnostic. It does not reopen bankroll buckets by itself.",
        "",
        "## Projection Accuracy",
        "",
        *_projection_table(payload.get("projection_accuracy") or []),
        "",
        "## Hitter Opportunity Breakdowns",
        "",
        "These rows isolate confirmed lineup, batting-order, projected-PA, and pinch-hit risk effects. `Low-PA Miss` means the model expected at least 3.8 PA and the hitter finished with 2 or fewer.",
        "",
        *_opportunity_breakdown_table(payload.get("hitter_opportunity_breakdowns") or []),
        "",
        "## Overall",
        "",
        *_table([payload["overall"]], top_n=1),
        "",
        "## Market",
        "",
        *_table(payload["market"], top_n=20),
        "",
        "## Market And Side",
        "",
        *_table(payload["market_side"], top_n=40),
        "",
        "## Rule",
        "",
        "Opportunity features are favored only when they improve holdout Brier and do not make ROI/CLV worse on enough selected rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_payload(cfg: OpportunityReportConfig) -> dict[str, Any]:
    df = _load(cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
        "unique_dates": int(df["game_date_et"].nunique()) if not df.empty else 0,
        "overall": {},
        "market": [],
        "market_side": [],
        "projection_accuracy": [],
        "hitter_opportunity_breakdowns": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        return payload
    payload["projection_accuracy"] = _projection_accuracy(df)
    payload["hitter_opportunity_breakdowns"] = _hitter_opportunity_breakdowns(df)
    payload["overall"] = {"key": "overall", **_compare_group(df, cfg)}
    payload["market"] = _group_rows(df, cfg, ["market"])
    payload["market_side"] = _group_rows(df, cfg, ["market", "side"])
    payload["status"] = "ready"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MLB prop opportunity features on holdout rows")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_opportunity_feature_report.json")
    parser.add_argument("--report-file", default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=150)
    parser.add_argument("--min-holdout-rows", type=int, default=40)
    parser.add_argument("--min-selected-rows", type=int, default=10)
    parser.add_argument("--min-brier-gain", type=float, default=0.001)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--no-rebuild-training-if-empty", action="store_true")
    args = parser.parse_args()
    cfg = OpportunityReportConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        report_file=args.report_file,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_selected_rows=args.min_selected_rows,
        min_brier_gain=args.min_brier_gain,
        min_ev=args.min_ev,
        rebuild_training_if_empty=not args.no_rebuild_training_if_empty,
    )
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(cfg)
    report_path = _write_report(payload, cfg)
    json_path = cfg.model_dir / cfg.out_file
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": report_path,
        "json_path": str(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
