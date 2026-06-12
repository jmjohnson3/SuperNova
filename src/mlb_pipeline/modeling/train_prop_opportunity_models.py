"""Train shadow opportunity models for MLB player props.

The prop side models are only as good as their opportunity inputs.  This
script learns the opportunity layer directly from locked historical rows:

* hitters: actual PA and low-PA/removal risk
* pitchers: actual BF, IP, and pitch-count proxy

The artifacts are shadow-only.  They feed distribution/model-selection audits
and provide a cleaner way to measure whether opportunity modeling is improving
predictive power before any bankroll promotion.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


HITTER_NUMERIC = [
    "confirmed_batting_order",
    "projected_pa",
    "pa_games",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_sp_hand_l",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "opp_team_k_pct_10",
    "opp_team_avg_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
    "batter_vs_hand_games_10",
    "batter_vs_rp_ba_30",
    "batter_vs_rp_slg_30",
    "batter_vs_rp_hr_rate_30",
    "batter_vs_rp_k_rate_30",
    "pinch_hit_risk",
]

HITTER_PA_V2_NUMERIC = [
    "lineup_confirmed_flag",
    "batting_order_missing_flag",
    "lineup_slot_pa_prior",
    "lineup_slot_low_pa_prior",
    "top_order_flag",
    "middle_order_flag",
    "bottom_order_flag",
    "projected_pa_slot_delta",
    "projected_pa_x_slot_prior",
    "team_runs_x_top_order",
    "team_runs_x_bottom_order",
    "platoon_sample_flag",
    "vs_hand_hit_per_pa",
    "vs_hand_tb_per_pa",
    "vs_hand_hr_per_pa",
    "vs_hand_iso_per_pa",
    "bullpen_quality_risk",
    "starter_quality_risk",
    "bench_or_removal_risk",
]

HITTER_NUMERIC = [*HITTER_NUMERIC, *HITTER_PA_V2_NUMERIC]

PITCHER_NUMERIC = [
    "projected_ip",
    "projected_bf",
    "projected_pitch_count",
    "pitcher_starts",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_team_k_pct_10",
    "opp_team_avg_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
]

HITTER_CATEGORICAL = ["confirmed_lineup_source", "opp_sp_hand", "team_abbr", "opponent_abbr"]
PITCHER_CATEGORICAL = ["team_abbr", "opponent_abbr"]


_LINEUP_PA_PRIORS = {
    1: 4.65,
    2: 4.55,
    3: 4.45,
    4: 4.35,
    5: 4.20,
    6: 4.05,
    7: 3.90,
    8: 3.75,
    9: 3.60,
}
_LINEUP_LOW_PA_PRIORS = {
    1: 0.04,
    2: 0.05,
    3: 0.05,
    4: 0.06,
    5: 0.07,
    6: 0.09,
    7: 0.12,
    8: 0.15,
    9: 0.18,
}


SQL = """
SELECT
    game_date_et,
    game_slug,
    player_id,
    player_name,
    team_abbr,
    opponent_abbr,
    market,
    confirmed_batting_order::float AS confirmed_batting_order,
    COALESCE(confirmed_lineup_source, 'unknown') AS confirmed_lineup_source,
    projected_pa::float AS projected_pa,
    pa_games::float AS pa_games,
    projected_ip::float AS projected_ip,
    projected_bf::float AS projected_bf,
    projected_pitch_count::float AS projected_pitch_count,
    pitcher_starts::float AS pitcher_starts,
    is_home::float AS is_home,
    team_implied_runs::float AS team_implied_runs,
    opponent_implied_runs::float AS opponent_implied_runs,
    game_total_line::float AS game_total_line,
    opp_sp_hand,
    opp_sp_hand_l::float AS opp_sp_hand_l,
    opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    opp_sp_bb_pct::float AS opp_sp_bb_pct,
    opp_sp_xwoba::float AS opp_sp_xwoba,
    opp_sp_hard_hit_pct::float AS opp_sp_hard_hit_pct,
    opp_sp_whiff_pct::float AS opp_sp_whiff_pct,
    opp_bp_era_10::float AS opp_bp_era_10,
    opp_bp_whip_10::float AS opp_bp_whip_10,
    opp_bp_k9_10::float AS opp_bp_k9_10,
    opp_bp_ip_last_3::float AS opp_bp_ip_last_3,
    opp_bp_ip_last_7::float AS opp_bp_ip_last_7,
    opp_team_k_pct_10::float AS opp_team_k_pct_10,
    opp_team_avg_10::float AS opp_team_avg_10,
    opp_team_obp_10::float AS opp_team_obp_10,
    opp_team_slg_10::float AS opp_team_slg_10,
    batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    batter_vs_hand_iso_avg_10::float AS batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10::float AS batter_vs_hand_k_rate_10,
    batter_vs_hand_games_10::float AS batter_vs_hand_games_10,
    batter_vs_rp_ba_30::float AS batter_vs_rp_ba_30,
    batter_vs_rp_slg_30::float AS batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30::float AS batter_vs_rp_hr_rate_30,
    batter_vs_rp_k_rate_30::float AS batter_vs_rp_k_rate_30,
    pinch_hit_risk::float AS pinch_hit_risk,
    actual_pa::float AS actual_pa,
    actual_bf::float AS actual_bf,
    actual_ip::float AS actual_ip,
    actual_pitch_count_proxy::float AS actual_pitch_count_proxy,
    low_pa_flag::float AS low_pa_flag
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND result_status = 'graded'
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
ORDER BY game_date_et, game_slug, player_id, market
"""


@dataclass(frozen=True)
class OpportunityConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_opportunity_models.json"
    report_file: str = "mlb_prop_opportunity_models_latest.md"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 100
    min_holdout_rows: int = 30


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
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _numeric_series(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def add_hitter_pa_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add opportunity features that describe lineup slot and playing-time risk."""
    if df.empty:
        return df
    out = df.copy()
    order = _numeric_series(out, "confirmed_batting_order")
    projected_pa = _numeric_series(out, "projected_pa")
    team_runs = _numeric_series(out, "team_implied_runs")
    pinch_risk = _numeric_series(out, "pinch_hit_risk", 0.0).fillna(0.0).clip(0.0, 1.0)

    rounded_slot = order.round()
    valid_slot = rounded_slot.between(1, 9)
    slot_int = rounded_slot.where(valid_slot).astype("Int64")
    slot_prior = slot_int.map(_LINEUP_PA_PRIORS).astype("float64").fillna(4.05)
    low_pa_prior = slot_int.map(_LINEUP_LOW_PA_PRIORS).astype("float64").fillna(0.14)

    out["lineup_confirmed_flag"] = valid_slot.astype(float)
    out["batting_order_missing_flag"] = (~valid_slot).astype(float)
    out["lineup_slot_pa_prior"] = slot_prior
    out["lineup_slot_low_pa_prior"] = low_pa_prior
    out["top_order_flag"] = slot_int.between(1, 4).fillna(False).astype(float)
    out["middle_order_flag"] = slot_int.between(5, 6).fillna(False).astype(float)
    out["bottom_order_flag"] = slot_int.between(7, 9).fillna(False).astype(float)
    out["projected_pa_slot_delta"] = projected_pa - slot_prior
    out["projected_pa_x_slot_prior"] = projected_pa * slot_prior
    out["team_runs_x_top_order"] = team_runs * out["top_order_flag"]
    out["team_runs_x_bottom_order"] = team_runs * out["bottom_order_flag"]

    vs_hand_games = _numeric_series(out, "batter_vs_hand_games_10", 0.0).fillna(0.0)
    pa_denom = projected_pa.where(projected_pa > 0.5, slot_prior).clip(lower=0.5)
    out["platoon_sample_flag"] = (vs_hand_games >= 5).astype(float)
    out["vs_hand_hit_per_pa"] = (_numeric_series(out, "batter_vs_hand_hits_avg_10") / pa_denom).clip(0.0, 1.0)
    out["vs_hand_tb_per_pa"] = (_numeric_series(out, "batter_vs_hand_tb_avg_10") / pa_denom).clip(0.0, 4.0)
    out["vs_hand_hr_per_pa"] = (_numeric_series(out, "batter_vs_hand_hr_avg_10") / pa_denom).clip(0.0, 1.0)
    out["vs_hand_iso_per_pa"] = (_numeric_series(out, "batter_vs_hand_iso_avg_10") / pa_denom).clip(0.0, 2.0)

    bp_era = _numeric_series(out, "opp_bp_era_10")
    bp_whip = _numeric_series(out, "opp_bp_whip_10")
    sp_xwoba = _numeric_series(out, "opp_sp_xwoba")
    sp_hard = _numeric_series(out, "opp_sp_hard_hit_pct")
    out["bullpen_quality_risk"] = (
        ((bp_era - 4.20) / 1.50).fillna(0.0)
        + ((bp_whip - 1.30) / 0.25).fillna(0.0)
    ).clip(-3.0, 3.0)
    out["starter_quality_risk"] = (
        ((sp_xwoba - 0.320) / 0.050).fillna(0.0)
        + ((sp_hard - 0.390) / 0.070).fillna(0.0)
    ).clip(-3.0, 3.0)
    out["bench_or_removal_risk"] = (
        0.55 * out["batting_order_missing_flag"]
        + 0.25 * out["bottom_order_flag"]
        + 0.20 * pinch_risk
    ).clip(0.0, 1.0)
    for col in HITTER_PA_V2_NUMERIC:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    return out


def _load(cfg: OpportunityConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = sorted(set(HITTER_NUMERIC + PITCHER_NUMERIC + [
        "actual_pa",
        "actual_bf",
        "actual_ip",
        "actual_pitch_count_proxy",
        "low_pa_flag",
    ]))
    for col in numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_hitter_pa_v2_features(df)
    for col in sorted(set(HITTER_CATEGORICAL + PITCHER_CATEGORICAL)):
        df[col] = df[col].fillna("unknown").astype(str)
    return df.replace([np.inf, -np.inf], np.nan)


def _split(df: pd.DataFrame, cfg: OpportunityConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train = df.loc[df["game_date_et"] < split].copy()
    holdout = df.loc[df["game_date_et"] >= split].copy()
    if len(train) >= cfg.min_train_rows and len(holdout) >= cfg.min_holdout_rows:
        return train, holdout, f"last_{cfg.holdout_days}_days"
    dates = sorted(df["game_date_et"].unique())
    if len(dates) > 1:
        holdout_date = dates[-1]
        return df.loc[df["game_date_et"] < holdout_date].copy(), df.loc[df["game_date_et"] >= holdout_date].copy(), "last_available_date"
    return train, holdout, f"last_{cfg.holdout_days}_days"


def _prepare(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    *,
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    cats: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, float], dict[str, float], dict[str, list[str]]]:
    means = dict(means or {})
    scales = dict(scales or {})
    cats = {key: list(value) for key, value in (cats or {}).items()}
    parts: list[np.ndarray] = []
    names: list[str] = []
    out_means: dict[str, float] = {}
    out_scales: dict[str, float] = {}
    for name in numeric:
        s = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, s.mean() if not s.dropna().empty else 0.0))
        filled = s.fillna(mean)
        std = float(filled.std(ddof=0) or 0.0)
        scale = float(scales.get(name, std if std > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy().reshape(-1, 1))
        names.append(name)
        out_means[name] = mean
        out_scales[name] = scale
    out_cats: dict[str, list[str]] = {}
    for name in categorical:
        values = cats.get(name)
        if values is None:
            values = sorted(str(v) for v in df.get(name, pd.Series([], dtype=str)).fillna("unknown").unique())
        out_cats[name] = values
        series = df.get(name, pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{name}={value}")
    return np.hstack(parts) if parts else np.zeros((len(df), 0)), names, out_means, out_scales, out_cats


def _score_linear(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    numeric = list(model.get("numeric_features") or [])
    categorical = list(model.get("categorical_features") or [])
    X, names, _, _, _ = _prepare(
        df,
        numeric,
        categorical,
        means=model.get("numeric_means") or {},
        scales=model.get("numeric_scales") or {},
        cats=model.get("categorical_values") or {},
    )
    coef_map = model.get("coef") or {}
    coef = np.array([float(coef_map.get(name, 0.0)) for name in names], dtype="float64")
    raw = float(model.get("intercept", 0.0)) + X.dot(coef)
    if model.get("kind") == "classifier":
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
    return raw


def _baseline_metrics_reg(df: pd.DataFrame, target: str, baseline: str) -> dict[str, Any]:
    work = df.dropna(subset=[target, baseline])
    if work.empty:
        return {"rows": 0}
    y = work[target].astype(float)
    p = work[baseline].astype(float)
    rmse = math.sqrt(mean_squared_error(y, p))
    return {
        "rows": int(len(work)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(rmse),
        "bias": float((p - y).mean()),
        "r2": float(r2_score(y, p)) if len(work) > 1 else None,
    }


def _model_metrics_reg(df: pd.DataFrame, target: str, pred: np.ndarray) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    if work.empty:
        return {"rows": 0}
    y = work[target].astype(float)
    p = pd.Series(pred, index=df.index).loc[work.index].astype(float)
    rmse = math.sqrt(mean_squared_error(y, p))
    return {
        "rows": int(len(work)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(rmse),
        "bias": float((p - y).mean()),
        "r2": float(r2_score(y, p)) if len(work) > 1 else None,
    }


def _model_metrics_cls(df: pd.DataFrame, target: str, prob: np.ndarray) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    if work.empty:
        return {"rows": 0}
    y = work[target].astype(int)
    p = pd.Series(prob, index=df.index).loc[work.index].astype(float).clip(1e-6, 1 - 1e-6)
    return {
        "rows": int(len(work)),
        "actual_rate": float(y.mean()),
        "avg_prob": float(p.mean()),
        "calibration_error": float(y.mean() - p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if y.nunique() == 2 else None,
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
    }


def _fit_regression(
    df: pd.DataFrame,
    target: str,
    baseline: str,
    numeric: list[str],
    categorical: list[str],
    cfg: OpportunityConfig,
) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    train, holdout, split = _split(work, cfg) if not work.empty else (work, work, "none")
    payload: dict[str, Any] = {
        "kind": "regression",
        "target": target,
        "baseline_feature": baseline,
        "rows": int(len(work)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_strategy": split,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    payload["baseline_holdout"] = _baseline_metrics_reg(holdout, target, baseline)
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
        payload["status"] = "insufficient_rows"
        return payload
    X_train, names, means, scales, cats = _prepare(train, numeric, categorical)
    y_train = train[target].astype(float).to_numpy()
    model = Ridge(alpha=2.0)
    model.fit(X_train, y_train)
    X_hold, _, _, _, _ = _prepare(holdout, numeric, categorical, means=means, scales=scales, cats=cats)
    pred = model.predict(X_hold)
    if target == "actual_pa":
        pred = np.clip(pred, 0.0, 7.0)
    elif target in {"actual_bf", "actual_pitch_count_proxy"}:
        pred = np.clip(pred, 0.0, 130.0)
    elif target == "actual_ip":
        pred = np.clip(pred, 0.0, 9.0)
    holdout_metrics = _model_metrics_reg(holdout, target, pred)
    base_metrics = payload.get("baseline_holdout") or {}
    use_for_distribution = (
        holdout_metrics.get("mae") is not None
        and base_metrics.get("mae") is not None
        and holdout_metrics["mae"] < base_metrics["mae"]
        and (
            holdout_metrics.get("rmse") is None
            or base_metrics.get("rmse") is None
            or holdout_metrics["rmse"] <= base_metrics["rmse"]
        )
    )
    payload.update({
        "status": "trained",
        "use_for_distribution": bool(use_for_distribution),
        "intercept": float(model.intercept_),
        "coef": {name: float(value) for name, value in zip(names, model.coef_) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "holdout": holdout_metrics,
    })
    return payload


def _fit_classifier(
    df: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    cfg: OpportunityConfig,
) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    train, holdout, split = _split(work, cfg) if not work.empty else (work, work, "none")
    payload: dict[str, Any] = {
        "kind": "classifier",
        "target": target,
        "rows": int(len(work)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_strategy": split,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows or train[target].nunique() < 2:
        payload["status"] = "insufficient_rows"
        return payload
    X_train, names, means, scales, cats = _prepare(train, numeric, categorical)
    y_train = train[target].astype(int).to_numpy()
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X_train, y_train)
    X_hold, _, _, _, _ = _prepare(holdout, numeric, categorical, means=means, scales=scales, cats=cats)
    prob = np.clip(model.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    payload.update({
        "status": "trained",
        "intercept": float(model.intercept_[0]),
        "coef": {name: float(value) for name, value in zip(names, model.coef_[0]) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "holdout": _model_metrics_cls(holdout, target, prob),
    })
    return payload


def _lineup_slot_impact(hitters: pd.DataFrame) -> list[dict[str, Any]]:
    work = hitters.dropna(subset=["confirmed_batting_order", "actual_pa"]).copy()
    if work.empty:
        return []
    rows = []
    for slot, group in work.groupby(work["confirmed_batting_order"].round().astype(int)):
        rows.append({
            "slot": int(slot),
            "rows": int(len(group)),
            "avg_actual_pa": float(group["actual_pa"].mean()),
            "low_pa_rate": float(group["low_pa_flag"].mean()) if group["low_pa_flag"].notna().any() else None,
        })
    rows.sort(key=lambda rec: rec["slot"])
    return rows


def _top_coefficients(model: dict[str, Any], limit: int = 12) -> list[tuple[str, float]]:
    coefs = [(name, float(value)) for name, value in (model.get("coef") or {}).items()]
    coefs.sort(key=lambda item: abs(item[1]), reverse=True)
    return coefs[:limit]


def _fmt_num(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def _fmt_pct(value: Any, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    v = float(value) * 100.0
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def _write_report(payload: dict[str, Any], cfg: OpportunityConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Prop Opportunity Models",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Rows: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Status: {payload.get('status')}",
        "",
        "## Regression Holdouts",
        "",
        "| Model | Rows | Base MAE | Model MAE | Base RMSE | Model RMSE | Model Bias | R2 | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("hitter_pa", "pitcher_bf", "pitcher_ip", "pitcher_pitch_count_proxy"):
        rec = payload.get("models", {}).get(name) or {}
        base = rec.get("baseline_holdout") or {}
        hold = rec.get("holdout") or {}
        decision = "model_helped" if rec.get("use_for_distribution") else (
            "keep_baseline" if rec.get("status") == "trained" else rec.get("status", "unknown")
        )
        lines.append(
            f"| {name} | {hold.get('rows', rec.get('holdout_rows', 0))} | "
            f"{_fmt_num(base.get('mae'))} | {_fmt_num(hold.get('mae'))} | "
            f"{_fmt_num(base.get('rmse'))} | {_fmt_num(hold.get('rmse'))} | "
            f"{_fmt_num(hold.get('bias'), signed=True)} | {_fmt_num(hold.get('r2'))} | {decision} |"
        )

    low_pa = payload.get("models", {}).get("hitter_low_pa") or {}
    hold = low_pa.get("holdout") or {}
    lines.extend([
        "",
        "## Low-PA / Removal Risk",
        "",
        "| Rows | Actual | Avg Prob | Brier | Log Loss | AUC | Status |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {hold.get('rows', low_pa.get('holdout_rows', 0))} | {_fmt_pct(hold.get('actual_rate'))} | "
            f"{_fmt_pct(hold.get('avg_prob'))} | {_fmt_num(hold.get('brier'))} | "
            f"{_fmt_num(hold.get('log_loss'))} | {_fmt_num(hold.get('auc'))} | {low_pa.get('status')} |"
        ),
        "",
        "## Lineup Slot Impact",
        "",
        "| Slot | Rows | Avg Actual PA | Low-PA Rate |",
        "|---:|---:|---:|---:|",
    ])
    for rec in payload.get("lineup_slot_impact", []):
        lines.append(
            f"| {rec['slot']} | {rec['rows']} | {_fmt_num(rec.get('avg_actual_pa'))} | {_fmt_pct(rec.get('low_pa_rate'))} |"
        )

    lines.extend(["", "## Largest Coefficients", ""])
    for name, rec in (payload.get("models") or {}).items():
        lines.extend([f"### {name}", "", "| Feature | Coef |", "|---|---:|"])
        for feature, coef in _top_coefficients(rec):
            lines.append(f"| {feature} | {_fmt_num(coef, signed=True)} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def train(cfg: OpportunityConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "usage": "shadow_only",
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
        "models": {},
        "lineup_slot_impact": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    hitters = (
        df.loc[df["market"].isin(["batter_hits", "batter_total_bases", "batter_home_runs"])]
        .drop_duplicates(["game_slug", "player_id"])
        .copy()
    )
    pitchers = (
        df.loc[df["market"] == "pitcher_strikeouts"]
        .drop_duplicates(["game_slug", "player_id"])
        .copy()
    )
    payload["hitter_player_games"] = int(len(hitters))
    payload["pitcher_player_games"] = int(len(pitchers))
    payload["lineup_slot_impact"] = _lineup_slot_impact(hitters)

    payload["models"]["hitter_pa"] = _fit_regression(
        hitters, "actual_pa", "projected_pa", HITTER_NUMERIC, HITTER_CATEGORICAL, cfg
    )
    payload["models"]["hitter_low_pa"] = _fit_classifier(
        hitters, "low_pa_flag", HITTER_NUMERIC, HITTER_CATEGORICAL, cfg
    )
    payload["models"]["pitcher_bf"] = _fit_regression(
        pitchers, "actual_bf", "projected_bf", PITCHER_NUMERIC, PITCHER_CATEGORICAL, cfg
    )
    payload["models"]["pitcher_ip"] = _fit_regression(
        pitchers, "actual_ip", "projected_ip", PITCHER_NUMERIC, PITCHER_CATEGORICAL, cfg
    )
    payload["models"]["pitcher_pitch_count_proxy"] = _fit_regression(
        pitchers, "actual_pitch_count_proxy", "projected_pitch_count", PITCHER_NUMERIC, PITCHER_CATEGORICAL, cfg
    )
    payload["status"] = "ready"
    payload["report_path"] = _write_report(payload, cfg)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop opportunity shadow models")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    payload = train(OpportunityConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "hitter_player_games": payload.get("hitter_player_games", 0),
        "pitcher_player_games": payload.get("pitcher_player_games", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
