"""Compare MLB prop probability variants on locked side examples.

This shadow report compares:
  - model_only: the locked player-model side probability
  - model_plus_prior: player model blended with historical market-side prior
  - market_no_vig: no-vig market probability baseline
  - direct_side_model: optional locked direct-side classifier, when trained

The report is intentionally diagnostic. It does not reopen bankroll buckets;
it tells us which probability source is earning that right.
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
from sklearn.metrics import brier_score_loss, log_loss

from .prop_betting_layer import apply_prop_market_side_prior
from .prop_replay import ev_per_unit
from .side_recalibration import clean_float, logit, prop_line_surface, sigmoid

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_DIRECT_NUMERIC_FEATURES = [
    "model_prob_side",
    "market_prob_side",
    "prob_edge_vs_market",
    "market_line",
    "abs_price",
    "is_plus_price",
    "count_edge_side",
    "pred_count",
    "pred_value",
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

_DIRECT_CATEGORICAL_FEATURES = [
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
class PropProbabilityComparisonConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_probability_variant_comparison.json"
    report_file: str | None = None
    lookback_days: int = 365
    min_ev: float = 0.02
    prior_file: str = "prop_market_side_priors.json"
    direct_model_file: str = "prop_direct_side_models.json"
    prior_max_blend: float = 0.35
    min_group_rows: int = 40
    min_selected_rows: int = 10
    min_brier_gain: float = 0.001
    min_clv_gain: float = 0.0


SQL = """
SELECT
    id,
    replay_id,
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
    paired_price::float AS paired_price,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(model_family, 'unknown') AS model_family,
    COALESCE(edge_type, 'unknown') AS edge_type,
    pred_value::float AS pred_value,
    pred_count::float AS pred_count,
    model_prob_side::float AS model_prob_side,
    market_prob_side::float AS market_prob_side,
    prob_edge_vs_market::float AS prob_edge_vs_market,
    count_edge_side::float AS count_edge_side,
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
    CASE WHEN market_price IS NULL THEN NULL ELSE ABS(market_price::float) END AS abs_price,
    CASE
        WHEN market_price IS NULL THEN NULL
        WHEN market_price::float > 0 THEN 1.0
        ELSE 0.0
    END AS is_plus_price,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS target,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    clv_price::float AS clv_price,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price,
    clv_line::float AS clv_line,
    CASE
        WHEN beat_clv_line IS TRUE THEN 1
        WHEN beat_clv_line IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_line
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


def _query_df(conn, sql: str, params: dict[str, object]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=columns)


def _load_rows(cfg: PropProbabilityComparisonConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    numeric = [
        "market_line",
        "market_price",
        "paired_price",
        "pred_value",
        "pred_count",
        "model_prob_side",
        "market_prob_side",
        "prob_edge_vs_market",
        "count_edge_side",
        "abs_price",
        "is_plus_price",
        "target",
        "profit_units",
        "clv_price",
        "beat_clv_price",
        "clv_line",
        "beat_clv_line",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["line_surface"] = [
        prop_line_surface(market, side, line)
        for market, side, line in zip(df["market"], df["side"], df["market_line"])
    ]
    df["target"] = df["target"].astype(int)
    df["push"] = df["push"].fillna(False).astype(bool)
    for col in _DIRECT_CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    return df.replace([np.inf, -np.inf], np.nan)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _direct_key(
    market: str = "*",
    side: str = "*",
    line_surface: str = "*",
    line_bucket: str = "*",
    price_bucket: str = "*",
    bookmaker_key: str = "*",
    model_family: str = "*",
) -> str:
    return "|".join([
        market or "*",
        side or "*",
        line_surface or "*",
        line_bucket or "*",
        price_bucket or "*",
        bookmaker_key or "*",
        model_family or "*",
    ])


def _direct_key_legacy(
    market: str = "*",
    side: str = "*",
    line_bucket: str = "*",
    price_bucket: str = "*",
    bookmaker_key: str = "*",
    model_family: str = "*",
) -> str:
    return "|".join([
        market or "*",
        side or "*",
        line_bucket or "*",
        price_bucket or "*",
        bookmaker_key or "*",
        model_family or "*",
    ])


def _lookup_direct_model(payload: dict, row: pd.Series) -> tuple[str | None, dict | None]:
    models = payload.get("models") or {}
    if not models:
        return None, None
    values = {
        "market": str(row.get("market") or "unknown"),
        "side": str(row.get("side") or "unknown"),
        "line_surface": str(row.get("line_surface") or "unknown"),
        "line_bucket": str(row.get("line_bucket") or "unknown"),
        "price_bucket": str(row.get("price_bucket") or "missing_price"),
        "bookmaker_key": str(row.get("bookmaker_key") or "unknown"),
        "model_family": str(row.get("model_family") or "unknown"),
    }
    priorities = [
        (values["market"], values["side"], values["line_surface"], values["line_bucket"], values["price_bucket"], values["bookmaker_key"], values["model_family"]),
        (values["market"], values["side"], values["line_surface"], values["line_bucket"], values["price_bucket"], values["bookmaker_key"], "*"),
        (values["market"], values["side"], values["line_surface"], values["line_bucket"], values["price_bucket"], "*", "*"),
        (values["market"], values["side"], values["line_surface"], values["line_bucket"], "*", "*", "*"),
        (values["market"], values["side"], values["line_surface"], "*", "*", "*", "*"),
        (values["market"], values["side"], "*", "*", "*", "*", "*"),
        ("*", values["side"], values["line_surface"], "*", "*", "*", "*"),
        ("*", values["side"], "*", "*", "*", "*", "*"),
        ("*", "*", "*", "*", "*", "*", "*"),
    ]
    for key_parts in priorities:
        key = _direct_key(*key_parts)
        rec = models.get(key)
        if rec:
            return key, rec
    legacy_priorities = [
        (values["market"], values["side"], values["line_bucket"], values["price_bucket"], values["bookmaker_key"], values["model_family"]),
        (values["market"], values["side"], values["line_bucket"], values["price_bucket"], values["bookmaker_key"], "*"),
        (values["market"], values["side"], values["line_bucket"], values["price_bucket"], "*", "*"),
        (values["market"], values["side"], values["line_bucket"], "*", "*", "*"),
        (values["market"], values["side"], "*", "*", "*", "*"),
        ("*", values["side"], "*", "*", "*", "*"),
        ("*", "*", "*", "*", "*", "*"),
    ]
    for key_parts in legacy_priorities:
        key = _direct_key_legacy(*key_parts)
        rec = models.get(key)
        if rec:
            return key, rec
    return None, None


def _score_linear(features: dict[str, Any], rec: dict, numeric_features: list[str], categorical_features: list[str]) -> float | None:
    intercept = clean_float(rec.get("intercept"))
    if intercept is None:
        return None
    coef = rec.get("coef") or {}
    means = rec.get("numeric_means") or {}
    scales = rec.get("numeric_scales") or {}
    z = intercept
    for name in numeric_features:
        value = clean_float(features.get(name))
        if value is None:
            value = clean_float(means.get(name)) or 0.0
        mean = clean_float(means.get(name)) or 0.0
        scale = clean_float(scales.get(name)) or 1.0
        if scale <= 1e-9:
            scale = 1.0
        z += float(coef.get(name, 0.0)) * ((value - mean) / scale)
    for name in categorical_features:
        value = str(features.get(name) or "unknown")
        z += float(coef.get(f"{name}={value}", 0.0))
    return max(1e-6, min(1.0 - 1e-6, sigmoid(z)))


def _score_direct_models(df: pd.DataFrame, payload: dict) -> tuple[pd.Series, pd.Series]:
    probs: list[float | None] = []
    keys: list[str | None] = []
    for _, row in df.iterrows():
        key, rec = _lookup_direct_model(payload, row)
        if not rec:
            probs.append(None)
            keys.append(None)
            continue
        features = {name: row.get(name) for name in _DIRECT_NUMERIC_FEATURES + _DIRECT_CATEGORICAL_FEATURES}
        p_direct = _score_linear(features, rec, _DIRECT_NUMERIC_FEATURES, _DIRECT_CATEGORICAL_FEATURES)
        p_raw = clean_float(row.get("model_prob_side"))
        if p_direct is None or p_raw is None:
            probs.append(None)
            keys.append(key)
            continue
        weight = clean_float(rec.get("blend_weight"))
        if weight is None:
            weight = 1.0
        weight = max(0.0, min(1.0, weight))
        p = sigmoid(logit(max(1e-6, min(1 - 1e-6, p_raw))) + weight * (logit(p_direct) - logit(max(1e-6, min(1 - 1e-6, p_raw)))))
        probs.append(max(1e-6, min(1.0 - 1e-6, p)))
        keys.append(key)
    return pd.Series(probs, index=df.index, dtype="float64"), pd.Series(keys, index=df.index, dtype="object")


def _apply_variants(df: pd.DataFrame, prior_payload: dict, direct_payload: dict, cfg: PropProbabilityComparisonConfig) -> pd.DataFrame:
    out = df.copy()
    out["p_model_only"] = pd.to_numeric(out["model_prob_side"], errors="coerce").clip(1e-6, 1 - 1e-6)
    out["p_market_no_vig"] = pd.to_numeric(out["market_prob_side"], errors="coerce").clip(1e-6, 1 - 1e-6)
    prior_probs = []
    prior_keys = []
    for _, row in out.iterrows():
        side = str(row.get("side") or "")
        price = row.get("market_price")
        paired = row.get("paired_price")
        over_price = price if side == "over" else paired
        under_price = price if side == "under" else paired
        p, key = apply_prop_market_side_prior(
            row.get("model_prob_side"),
            prior_payload,
            market=str(row.get("market") or ""),
            side=side,
            line=row.get("market_line"),
            price=price,
            over_price=over_price,
            under_price=under_price,
            bookmaker_key=str(row.get("bookmaker_key") or "unknown"),
            max_blend_weight=cfg.prior_max_blend,
        )
        prior_probs.append(p)
        prior_keys.append(key)
    out["p_model_plus_prior"] = pd.Series(prior_probs, index=out.index, dtype="float64").clip(1e-6, 1 - 1e-6)
    out["prior_key"] = prior_keys
    direct_probs, direct_keys = _score_direct_models(out, direct_payload)
    out["p_direct_side_model"] = direct_probs.clip(1e-6, 1 - 1e-6)
    out["direct_key"] = direct_keys
    return out


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _forecast_summary(df: pd.DataFrame, prob_col: str) -> dict:
    work = df.loc[~df["push"]].copy()
    p = pd.to_numeric(work.get(prob_col), errors="coerce")
    y = pd.to_numeric(work.get("target"), errors="coerce")
    mask = p.notna() & y.notna()
    if not mask.any():
        return {
            "rows": 0,
            "actual_rate": None,
            "avg_prob": None,
            "calibration_error": None,
            "brier": None,
            "log_loss": None,
        }
    pv = p.loc[mask].clip(1e-6, 1 - 1e-6).to_numpy()
    yv = y.loc[mask].astype(int).to_numpy()
    actual = float(np.mean(yv))
    avg_prob = float(np.mean(pv))
    return {
        "rows": int(mask.sum()),
        "actual_rate": actual,
        "avg_prob": avg_prob,
        "calibration_error": actual - avg_prob,
        "brier": float(brier_score_loss(yv, pv)),
        "log_loss": float(log_loss(yv, pv, labels=[0, 1])),
    }


def _ev_for_row(prob: Any, price: Any) -> float | None:
    p = clean_float(prob)
    if p is None:
        return None
    return ev_per_unit(p, price)


def _selected_rows(df: pd.DataFrame, prob_col: str, min_ev: float) -> pd.DataFrame:
    work = df.copy()
    work["variant_ev"] = [
        _ev_for_row(prob, price)
        for prob, price in zip(work.get(prob_col), work.get("market_price"))
    ]
    work = work.dropna(subset=["variant_ev", "profit_units"])
    if work.empty:
        return work
    selected = []
    for _, sub in work.groupby("replay_id", dropna=False):
        best_idx = sub["variant_ev"].astype(float).idxmax()
        best = work.loc[best_idx]
        if float(best["variant_ev"]) >= min_ev:
            selected.append(best)
    if not selected:
        return work.iloc[0:0].copy()
    return pd.DataFrame(selected)


def _selection_summary(df: pd.DataFrame, prob_col: str, min_ev: float) -> dict:
    selected = _selected_rows(df, prob_col, min_ev)
    if selected.empty:
        return {
            "selected_rows": 0,
            "win_rate": None,
            "units": None,
            "roi": None,
            "avg_ev": None,
            "avg_clv_price": None,
            "clv_price_beat_rate": None,
            "avg_clv_line": None,
            "clv_line_beat_rate": None,
        }
    non_push = selected.loc[~selected["push"]]
    units = _mean(selected["profit_units"]) * len(selected) if _mean(selected["profit_units"]) is not None else None
    clv_price = pd.to_numeric(selected.get("beat_clv_price"), errors="coerce")
    clv_line = pd.to_numeric(selected.get("beat_clv_line"), errors="coerce")
    return {
        "selected_rows": int(len(selected)),
        "win_rate": _mean(non_push["target"]) if not non_push.empty else None,
        "units": units,
        "roi": _mean(selected["profit_units"]),
        "avg_ev": _mean(selected["variant_ev"]),
        "avg_clv_price": _mean(selected["clv_price"]),
        "clv_price_beat_rate": _mean(clv_price.dropna()) if clv_price.notna().any() else None,
        "avg_clv_line": _mean(selected["clv_line"]),
        "clv_line_beat_rate": _mean(clv_line.dropna()) if clv_line.notna().any() else None,
    }


def _variant_summary(df: pd.DataFrame, prob_col: str, cfg: PropProbabilityComparisonConfig) -> dict:
    return {
        "forecast": _forecast_summary(df, prob_col),
        "selection": _selection_summary(df, prob_col, cfg.min_ev),
    }


def _recommend(summary: dict, cfg: PropProbabilityComparisonConfig) -> str:
    model = summary.get("model_only") or {}
    prior = summary.get("model_plus_prior") or {}
    mf = model.get("forecast") or {}
    pf = prior.get("forecast") or {}
    ms = model.get("selection") or {}
    ps = prior.get("selection") or {}
    if (pf.get("rows") or 0) < cfg.min_group_rows:
        return "insufficient_rows"
    mb = mf.get("brier")
    pb = pf.get("brier")
    if mb is None or pb is None or (mb - pb) < cfg.min_brier_gain:
        return "keep_model_only"
    if (ms.get("selected_rows") or 0) >= cfg.min_selected_rows and (ps.get("selected_rows") or 0) >= cfg.min_selected_rows:
        if (ps.get("roi") is None or ms.get("roi") is None or ps["roi"] < ms["roi"]):
            return "keep_model_only_roi"
        if (
            ps.get("clv_price_beat_rate") is not None
            and ms.get("clv_price_beat_rate") is not None
            and ps["clv_price_beat_rate"] < ms["clv_price_beat_rate"]
        ):
            return "keep_model_only_clv"
    if abs(pf.get("calibration_error") or 0.0) > abs(mf.get("calibration_error") or 0.0):
        return "keep_model_only_calibration"
    return "use_model_plus_prior"


def _eligible_variant_against_model(summary: dict, name: str, cfg: PropProbabilityComparisonConfig) -> tuple[bool, list[str]]:
    baseline = summary.get("model_only") or {}
    candidate = summary.get(name) or {}
    bf = baseline.get("forecast") or {}
    cf = candidate.get("forecast") or {}
    bs = baseline.get("selection") or {}
    cs = candidate.get("selection") or {}
    reasons: list[str] = []

    bb = bf.get("brier")
    cb = cf.get("brier")
    if bb is None or cb is None:
        reasons.append("missing_brier")
    elif (bb - cb) < cfg.min_brier_gain:
        reasons.append("brier_not_improved")

    base_picks = int(bs.get("selected_rows") or 0)
    cand_picks = int(cs.get("selected_rows") or 0)
    if cand_picks < cfg.min_selected_rows:
        reasons.append("candidate_selected_rows_too_small")
    if base_picks >= cfg.min_selected_rows and cand_picks >= cfg.min_selected_rows:
        broi = bs.get("roi")
        croi = cs.get("roi")
        if broi is None or croi is None:
            reasons.append("missing_roi")
        elif croi < broi:
            reasons.append("roi_not_improved")

        bbeat = bs.get("clv_price_beat_rate")
        cbeat = cs.get("clv_price_beat_rate")
        if bbeat is None or cbeat is None:
            reasons.append("missing_clv_beat_rate")
        elif cbeat < bbeat:
            reasons.append("clv_beat_rate_not_improved")

        bclv = bs.get("avg_clv_price")
        cclv = cs.get("avg_clv_price")
        if bclv is None or cclv is None:
            reasons.append("missing_avg_clv_price")
        elif (cclv - bclv) < cfg.min_clv_gain:
            reasons.append("avg_clv_price_not_improved")

    if abs(cf.get("calibration_error") or 0.0) > abs(bf.get("calibration_error") or 0.0) + 0.01:
        reasons.append("calibration_worse")
    return not reasons, reasons


def _choose_shadow_variant(summary: dict, cfg: PropProbabilityComparisonConfig) -> dict:
    decisions: dict[str, dict] = {}
    eligible: list[tuple[float, str]] = []
    for name in summary:
        if name == "model_only":
            continue
        ok, reasons = _eligible_variant_against_model(summary, name, cfg)
        rec = summary.get(name) or {}
        forecast = rec.get("forecast") or {}
        selection = rec.get("selection") or {}
        decisions[name] = {
            "eligible": ok,
            "reasons": reasons,
        }
        if ok:
            score = float(forecast.get("brier") or 9.0) - float(selection.get("roi") or 0.0) * 0.01
            eligible.append((score, name))
    if not eligible:
        return {
            "selected": "model_only",
            "reason": "No shadow variant improved Brier, ROI, and CLV versus model_only.",
            "candidates": decisions,
        }
    eligible.sort()
    return {
        "selected": eligible[0][1],
        "reason": "Selected variant improved Brier, ROI, and CLV versus model_only.",
        "candidates": decisions,
    }


def _summaries_for(df: pd.DataFrame, cfg: PropProbabilityComparisonConfig) -> dict:
    variants = {
        "model_only": "p_model_only",
        "model_plus_prior": "p_model_plus_prior",
        "market_no_vig": "p_market_no_vig",
    }
    if pd.to_numeric(df.get("p_direct_side_model"), errors="coerce").notna().any():
        variants["direct_side_model"] = "p_direct_side_model"
    return {name: _variant_summary(df, col, cfg) for name, col in variants.items()}


def _group_summaries(df: pd.DataFrame, cfg: PropProbabilityComparisonConfig, group_cols: list[str]) -> list[dict]:
    rows: list[dict] = []
    for values, sub in df.groupby(group_cols, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        value_dict = dict(zip(group_cols, values))
        summary = _summaries_for(sub, cfg)
        rows.append({
            "key": "|".join(str(value_dict.get(col, "*")) for col in group_cols),
            **value_dict,
            "rows": int(len(sub)),
            "recommendation": _recommend(summary, cfg),
            "variants": summary,
        })
    rows.sort(key=lambda r: (r.get("market", ""), r.get("side", ""), r.get("line_bucket", ""), r.get("price_bucket", "")))
    return rows


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _fmt_clv(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}"


def _fmt_num(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _variant_table(summary: dict) -> list[str]:
    lines = [
        "| Variant | Rows | Brier | Cal err | Picks | ROI | CLV beat | Avg CLV price |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in summary.items():
        f = rec["forecast"]
        s = rec["selection"]
        lines.append(
            f"| {name} | {f['rows']} | {_fmt_num(f['brier'])} | {_fmt_pct(f['calibration_error'])} | "
            f"{s['selected_rows']} | {_fmt_pct(s['roi'])} | {_fmt_pct(s['clv_price_beat_rate'])} | {_fmt_clv(s['avg_clv_price'])} |"
        )
    return lines


def _write_report(payload: dict, cfg: PropProbabilityComparisonConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_name = cfg.report_file or f"mlb_prop_probability_shadow_{datetime.now(timezone.utc).date().isoformat()}.md"
    path = _REPORT_DIR / report_name
    lines = [
        f"# MLB Prop Probability Shadow - {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Scope",
        "",
        f"- Locked side rows: {payload.get('rows', 0)}",
        f"- Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"- Unique graded dates: {payload.get('unique_dates', 0)}",
        f"- Minimum EV for simulated picks: {cfg.min_ev:.3f}",
        f"- Shadow winner: {payload.get('shadow_winner', {}).get('selected', 'model_only')}",
        "",
        "## Overall",
        "",
        *_variant_table(payload.get("overall", {})),
        "",
        "## Market/Side Recommendations",
        "",
        "| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rec in payload.get("market_side", []):
        variants = rec.get("variants", {})
        model = variants.get("model_only", {})
        prior = variants.get("model_plus_prior", {})
        mf = model.get("forecast", {})
        pf = prior.get("forecast", {})
        ms = model.get("selection", {})
        ps = prior.get("selection", {})
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {rec['recommendation']} | "
            f"{_fmt_num(mf.get('brier'))} | {_fmt_num(pf.get('brier'))} | "
            f"{_fmt_pct(ms.get('roi'))} | {_fmt_pct(ps.get('roi'))} | {_fmt_pct(ps.get('clv_price_beat_rate'))} |"
        )
    lines.extend([
        "",
        "## Line Surface Recommendations",
        "",
        "| Bucket | Rows | Recommendation | Model Brier | Prior Brier | Model ROI | Prior ROI | Prior CLV beat |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("line_surface", []):
        variants = rec.get("variants", {})
        model = variants.get("model_only", {})
        prior = variants.get("model_plus_prior", {})
        mf = model.get("forecast", {})
        pf = prior.get("forecast", {})
        ms = model.get("selection", {})
        ps = prior.get("selection", {})
        lines.append(
            f"| {rec['key']} | {rec['rows']} | {rec['recommendation']} | "
            f"{_fmt_num(mf.get('brier'))} | {_fmt_num(pf.get('brier'))} | "
            f"{_fmt_pct(ms.get('roi'))} | {_fmt_pct(ps.get('roi'))} | {_fmt_pct(ps.get('clv_price_beat_rate'))} |"
        )
    lines.extend([
        "",
        "## Shadow Winner",
        "",
        payload.get("shadow_winner", {}).get("reason", "No shadow winner available."),
        "",
        "| Candidate | Eligible | Reasons |",
        "|---|---:|---|",
    ])
    for name, rec in (payload.get("shadow_winner", {}).get("candidates") or {}).items():
        reasons = ", ".join(rec.get("reasons") or ["passes"])
        lines.append(f"| {name} | {str(bool(rec.get('eligible'))).lower()} | {reasons} |")
    lines.extend([
        "",
        "## Rule",
        "",
        "A variant is promoted only when it improves Brier score, ROI, and CLV versus model_only. Small selected-pick samples are treated as diagnostic only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def compare(cfg: PropProbabilityComparisonConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(cfg)
    prior_payload = _load_json(cfg.model_dir / cfg.prior_file)
    direct_payload = _load_json(cfg.model_dir / cfg.direct_model_file)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "lookback_days": cfg.lookback_days,
        "min_ev": cfg.min_ev,
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
        "unique_dates": int(df["game_date_et"].nunique()) if not df.empty else 0,
        "prior_models": len(prior_payload.get("models") or {}),
        "direct_side_models": len(direct_payload.get("models") or {}),
        "overall": {},
        "shadow_winner": {},
        "market_side": [],
        "line_surface": [],
        "bucket": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    scored = _apply_variants(df, prior_payload, direct_payload, cfg)
    payload["overall"] = _summaries_for(scored, cfg)
    payload["shadow_winner"] = _choose_shadow_variant(payload["overall"], cfg)
    payload["market_side"] = _group_summaries(scored, cfg, ["market", "side"])
    payload["line_surface"] = _group_summaries(scored, cfg, ["market", "side", "line_surface"])
    payload["bucket"] = _group_summaries(scored, cfg, ["market", "side", "line_surface", "line_bucket", "price_bucket"])
    payload["status"] = "ready"
    payload["report_path"] = _write_report(payload, cfg)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLB prop probability variants on locked side examples")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_probability_variant_comparison.json")
    parser.add_argument("--report-file", default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--prior-file", default="prop_market_side_priors.json")
    parser.add_argument("--direct-model-file", default="prop_direct_side_models.json")
    parser.add_argument("--prior-max-blend", type=float, default=0.35)
    parser.add_argument("--min-group-rows", type=int, default=40)
    parser.add_argument("--min-selected-rows", type=int, default=10)
    parser.add_argument("--min-brier-gain", type=float, default=0.001)
    parser.add_argument("--min-clv-gain", type=float, default=0.0)
    args = parser.parse_args()
    payload = compare(PropProbabilityComparisonConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        report_file=args.report_file,
        lookback_days=args.lookback_days,
        min_ev=args.min_ev,
        prior_file=args.prior_file,
        direct_model_file=args.direct_model_file,
        prior_max_blend=args.prior_max_blend,
        min_group_rows=args.min_group_rows,
        min_selected_rows=args.min_selected_rows,
        min_brier_gain=args.min_brier_gain,
        min_clv_gain=args.min_clv_gain,
    ))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
