"""Diagnose hitter hits/HR player-rate errors on one row per player-game."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg2
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN
from mlb_pipeline.modeling.train_hitter_player_game_outcome_models import add_leakage_safe_player_priors

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_BLEND_ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
_HIT_BIAS_ALPHAS = (0.0, 0.15, 0.25, 0.35, 0.50, 0.65)
_MIN_TRAIN_ROWS = 1000
_FOLD_TEST_DAYS = 10
_MIN_MAE_GAIN = {
    "hits": 0.005,
    "home_runs": 0.0005,
}
_MIN_BRIER_GAIN = {
    "hits": 0.00025,
    "home_runs": 0.0005,
}
_MAX_BIAS_WORSENING = {
    "hits": 0.02,
    "home_runs": 0.01,
}
_HIT_BIAS_GROUP_SPECS = (
    ("player_id", 0.34, 24.0),
    ("lineup_pa_bucket", 0.18, 120.0),
    ("lineup_prior_bucket", 0.18, 120.0),
    ("platoon_prior_bucket", 0.13, 140.0),
    ("park_hit_bucket", 0.08, 160.0),
    ("run_environment_bucket", 0.05, 180.0),
    ("home_away_bucket", 0.04, 180.0),
)

NUMERIC_FEATURES = [
    "lineup_slot",
    "confirmed_starter_num",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "park_run_factor",
    "park_hr_factor",
    "park_babip_factor",
    "temperature_f",
    "wind_speed_mph",
    "wind_sin",
    "wind_cos",
    "is_dome",
    "is_day_game",
    "own_lineup_xwoba_avg",
    "own_lineup_xslg_avg",
    "own_lineup_barrel_avg",
    "own_lineup_hard_hit_avg",
    "lineup_confirmed_flag",
    "confirmed_team_lineup_slots",
    "team_lineup_confirmed_flag",
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
    "batter_sc_barrel_rate",
    "batter_sc_hard_hit_pct",
    "batter_sc_avg_exit_velo",
    "batter_sc_avg_launch_angle",
    "batter_sc_sweet_spot_pct",
    "batter_sc_fb_pct",
    "batter_sc_gb_pct",
    "batter_sc_ld_pct",
    "batter_sc_xba",
    "batter_sc_xslg",
    "batter_sc_xwoba",
    "batter_sc_xiso",
    "batter_sc_brl_pa",
    "batter_sprint_speed",
    "batter_disc_whiff_pct",
    "batter_disc_k_pct",
    "batter_disc_bb_pct",
    "opp_sp_sc_barrel_rate",
    "opp_sp_sc_hard_hit_pct",
    "opp_sp_sc_avg_exit_velo",
    "opp_sp_sc_avg_launch_angle",
    "opp_sp_sc_xba",
    "opp_sp_sc_xslg",
    "opp_sp_sc_xwoba",
    "opp_sp_sc_xiso",
    "opp_sp_fastball_family_pct",
    "opp_sp_pitch_diversity",
    "projected_pa",
    "pa_games",
    "base_hits_per_pa",
    "base_hr_per_pa",
    "player_prior_pa",
    "player_prior_hit_rate",
    "player_prior_xbh_given_hit",
    "player_prior_hr_given_xbh",
    "player_prior_walk_given_non_hit",
]
CATEGORICAL_FEATURES = [
    "team_abbr",
    "opponent_abbr",
    "lineup_source",
    "primary_position",
    "batter_hand",
    "opp_sp_hand",
]

SQL = """
SELECT game_date_et, game_slug, player_id, player_name,
       team_abbr, opponent_abbr, lineup_slot, lineup_source,
       CASE WHEN confirmed_starter IS TRUE THEN 1.0 ELSE 0.0 END AS confirmed_starter_num,
       primary_position, batter_hand, opp_sp_hand,
       is_home::float, projected_pa::float, actual_pa::float, low_pa_flag::float,
       model_pred_hits::float, actual_hits::float,
       actual_singles::float, actual_doubles::float, actual_triples::float,
       actual_walks::float,
       model_pred_home_runs::float, actual_home_runs::float,
       team_implied_runs::float, opponent_implied_runs::float, game_total_line::float,
       park_run_factor::float, park_hr_factor::float, park_babip_factor::float,
       temperature_f::float, wind_speed_mph::float, wind_sin::float, wind_cos::float,
       is_dome::float, is_day_game::float,
       own_lineup_xwoba_avg::float, own_lineup_xslg_avg::float,
       own_lineup_barrel_avg::float, own_lineup_hard_hit_avg::float,
       lineup_confirmed_flag::float, confirmed_team_lineup_slots::float,
       team_lineup_confirmed_flag::float,
       opp_sp_k_pct_10::float, opp_sp_bb_pct::float, opp_sp_xwoba::float,
       opp_sp_hard_hit_pct::float, opp_sp_whiff_pct::float,
       opp_bp_era_10::float, opp_bp_whip_10::float, opp_bp_k9_10::float,
       opp_bp_ip_last_3::float, opp_bp_ip_last_7::float,
       opp_team_k_pct_10::float, opp_team_avg_10::float,
       opp_team_obp_10::float, opp_team_slg_10::float,
       batter_vs_hand_hits_avg_10::float, batter_vs_hand_tb_avg_10::float,
       batter_vs_hand_hr_avg_10::float, batter_vs_hand_iso_avg_10::float,
       batter_vs_hand_k_rate_10::float, batter_vs_hand_games_10::float,
       batter_vs_rp_ba_30::float, batter_vs_rp_slg_30::float,
       batter_vs_rp_hr_rate_30::float, batter_vs_rp_k_rate_30::float,
       batter_sc_barrel_rate::float, batter_sc_hard_hit_pct::float,
       batter_sc_avg_exit_velo::float, batter_sc_avg_launch_angle::float,
       batter_sc_sweet_spot_pct::float, batter_sc_fb_pct::float,
       batter_sc_gb_pct::float, batter_sc_ld_pct::float,
       batter_sc_xba::float, batter_sc_xslg::float, batter_sc_xwoba::float,
       batter_sc_xiso::float, batter_sc_brl_pa::float, batter_sprint_speed::float,
       batter_disc_whiff_pct::float, batter_disc_k_pct::float,
       batter_disc_bb_pct::float,
       opp_sp_sc_barrel_rate::float, opp_sp_sc_hard_hit_pct::float,
       opp_sp_sc_avg_exit_velo::float, opp_sp_sc_avg_launch_angle::float,
       opp_sp_sc_xba::float, opp_sp_sc_xslg::float,
       opp_sp_sc_xwoba::float, opp_sp_sc_xiso::float,
       opp_sp_fastball_family_pct::float, opp_sp_pitch_diversity::float
FROM features.mlb_hitter_player_game_training
WHERE game_date_et >= %(cutoff)s
  AND actual_pa IS NOT NULL
ORDER BY game_date_et, game_slug, player_id
"""


def _bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    projected_pa = pd.to_numeric(out.get("projected_pa"), errors="coerce").replace(0, np.nan)
    out["base_hits_per_pa"] = pd.to_numeric(out.get("model_pred_hits"), errors="coerce") / projected_pa
    out["base_hr_per_pa"] = pd.to_numeric(out.get("model_pred_home_runs"), errors="coerce") / projected_pa
    slot = pd.to_numeric(out["lineup_slot"], errors="coerce")
    out["lineup_bucket"] = np.select(
        [slot.between(1, 2), slot.between(3, 5), slot.between(6, 9)],
        ["slot_1_2", "slot_3_5", "slot_6_9"],
        default="slot_unknown",
    )
    batter_hand = out["batter_hand"].fillna("unknown").astype(str).str.upper()
    pitcher_hand = out["opp_sp_hand"].fillna("unknown").astype(str).str.upper()
    out["platoon_bucket"] = np.where(
        batter_hand.isin(["L", "R"]) & pitcher_hand.isin(["L", "R"]),
        np.where(batter_hand.eq(pitcher_hand), "same_hand", "opposite_hand"),
        "unknown_hand",
    )
    barrel = pd.to_numeric(out["batter_sc_barrel_rate"], errors="coerce")
    out["barrel_bucket"] = np.select(
        [barrel < 5.0, barrel.between(5.0, 10.0, inclusive="left"), barrel >= 10.0],
        ["barrel_low", "barrel_average", "barrel_high"],
        default="barrel_missing",
    )
    xslg = pd.to_numeric(out["batter_sc_xslg"], errors="coerce")
    out["xslg_bucket"] = np.select(
        [xslg < 0.350, xslg.between(0.350, 0.500, inclusive="left"), xslg >= 0.500],
        ["xslg_low", "xslg_average", "xslg_high"],
        default="xslg_missing",
    )
    opp_xwoba = pd.to_numeric(out["opp_sp_sc_xwoba"], errors="coerce")
    out["starter_quality_bucket"] = np.select(
        [opp_xwoba < 0.300, opp_xwoba.between(0.300, 0.340, inclusive="left"), opp_xwoba >= 0.340],
        ["starter_strong", "starter_average", "starter_weak"],
        default="starter_quality_missing",
    )
    out["home_away_bucket"] = np.where(pd.to_numeric(out["is_home"], errors="coerce") >= 0.5, "home", "away")
    out["pa_outcome_bucket"] = np.where(pd.to_numeric(out["actual_pa"], errors="coerce") <= 2.0, "actual_pa_0_2", "actual_pa_3_plus")
    return out


def _add_hit_bias_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add live-recreatable hit-rate calibration buckets."""
    out = df.copy()
    projected_pa = pd.to_numeric(out.get("projected_pa"), errors="coerce")
    out["projected_pa_bucket"] = np.select(
        [
            projected_pa >= 4.4,
            projected_pa.between(3.8, 4.4, inclusive="left"),
            projected_pa < 3.8,
        ],
        ["projected_pa_high_4_4_plus", "projected_pa_mid_3_8_4_4", "projected_pa_low_under_3_8"],
        default="projected_pa_missing",
    )
    prior = pd.to_numeric(out.get("player_prior_hit_rate"), errors="coerce")
    out["player_prior_hit_bucket"] = np.select(
        [
            prior >= 0.280,
            prior.between(0.235, 0.280, inclusive="left"),
            prior < 0.235,
        ],
        ["prior_hit_rate_high", "prior_hit_rate_mid", "prior_hit_rate_low"],
        default="prior_hit_rate_missing",
    )
    team_runs = pd.to_numeric(out.get("team_implied_runs"), errors="coerce")
    out["run_environment_bucket"] = np.select(
        [
            team_runs >= 4.8,
            team_runs.between(4.0, 4.8, inclusive="left"),
            team_runs < 4.0,
        ],
        ["team_total_high", "team_total_mid", "team_total_low"],
        default="team_total_missing",
    )
    park = pd.to_numeric(out.get("park_babip_factor"), errors="coerce")
    park = park.fillna(pd.to_numeric(out.get("park_run_factor"), errors="coerce"))
    out["park_hit_bucket"] = np.select(
        [
            park >= 1.03,
            park.between(0.97, 1.03, inclusive="left"),
            park < 0.97,
        ],
        ["hit_park_boost", "hit_park_neutral", "hit_park_suppress"],
        default="hit_park_missing",
    )
    for column in ("lineup_bucket", "platoon_bucket", "home_away_bucket"):
        if column not in out:
            out[column] = "missing"
        out[column] = out[column].fillna("missing").astype(str)
    out["lineup_pa_bucket"] = out["lineup_bucket"].astype(str) + "|" + out["projected_pa_bucket"].astype(str)
    out["lineup_prior_bucket"] = out["lineup_bucket"].astype(str) + "|" + out["player_prior_hit_bucket"].astype(str)
    out["platoon_prior_bucket"] = out["platoon_bucket"].astype(str) + "|" + out["player_prior_hit_bucket"].astype(str)
    return out


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", _one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _regressor_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LGBMRegressor(
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=5,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.15,
            reg_lambda=1.25,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _classifier_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline([
        ("features", _preprocessor(numeric_features, categorical_features)),
        ("model", LGBMClassifier(
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=5,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=1.50,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])


def _count_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, Any]:
    valid = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if valid.empty:
        return {"rows": 0}
    error = valid["pred"] - valid["actual"]
    return {
        "rows": int(len(valid)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def _event_probability_from_count(count: pd.Series) -> pd.Series:
    values = pd.to_numeric(count, errors="coerce").clip(lower=0.0, upper=8.0)
    return 1.0 - np.exp(-values)


def _any_brier(actual: pd.Series, pred_count: pd.Series) -> dict[str, Any]:
    valid = pd.DataFrame({"actual": actual, "pred": pred_count}).dropna()
    if valid.empty:
        return {"rows": 0, "brier": None}
    target = valid["actual"].gt(0).astype(int)
    prob = _event_probability_from_count(valid["pred"]).clip(1e-6, 1.0 - 1e-6)
    return {
        "rows": int(len(valid)),
        "brier": float(brier_score_loss(target, prob)),
        "actual_any_rate": float(target.mean()),
        "pred_any_rate": float(prob.mean()),
    }


def _rate_metrics(df: pd.DataFrame, actual_col: str, pred: pd.Series) -> dict[str, Any]:
    valid = pd.DataFrame({
        "actual": pd.to_numeric(df[actual_col], errors="coerce"),
        "pred": pd.to_numeric(pred, errors="coerce"),
        "actual_pa": pd.to_numeric(df["actual_pa"], errors="coerce"),
        "projected_pa": pd.to_numeric(df["projected_pa"], errors="coerce"),
    }).dropna()
    valid = valid.loc[(valid["actual_pa"] > 0) & (valid["projected_pa"] > 0)]
    if valid.empty:
        return {"rows": 0}
    pred_rate = valid["pred"] / valid["projected_pa"]
    actual_rate = valid["actual"] / valid["actual_pa"]
    return {
        "rows": int(len(valid)),
        "pred_rate": float(pred_rate.mean()),
        "actual_rate": float(actual_rate.mean()),
        "rate_bias": float((pred_rate - actual_rate).mean()),
    }


def _available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in NUMERIC_FEATURES if column in df.columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in df.columns]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    numeric = [column for column in numeric if df[column].notna().any()]
    for column in categorical:
        df[column] = df[column].fillna("unknown").astype(str)
    return numeric, categorical


def _raw_walk_forward_predictions(
    df: pd.DataFrame,
    *,
    stat_key: str,
    actual_col: str,
    upper: float,
) -> tuple[pd.Series, dict[str, Any]]:
    work = df.dropna(subset=[actual_col, "projected_pa"]).copy()
    numeric, categorical = _available_features(work)
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for start in range(0, len(dates), _FOLD_TEST_DAYS):
        test_dates = dates[start:start + _FOLD_TEST_DAYS]
        if not test_dates:
            continue
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < test_dates[0]
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date.isin(test_dates)
        if int(train_mask.sum()) < _MIN_TRAIN_ROWS or int(test_mask.sum()) == 0:
            continue
        train = work.loc[train_mask]
        test = work.loc[test_mask]
        if stat_key == "home_runs":
            y = train[actual_col].gt(0).astype(int)
            if y.nunique() < 2:
                continue
            model = _classifier_pipeline(numeric, categorical)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
                model.fit(train[numeric + categorical], y)
                raw = model.predict_proba(test[numeric + categorical])[:, 1]
        else:
            model = _regressor_pipeline(numeric, categorical)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
                warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
                model.fit(train[numeric + categorical], train[actual_col])
                raw = model.predict(test[numeric + categorical])
        pred.loc[test.index] = np.clip(raw, 0.0, upper)
        folds.append({
            "holdout_start": str(test_dates[0]),
            "holdout_end": str(test_dates[-1]),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
        })
    return pred, {
        "enabled": bool(folds),
        "folds": folds,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }


def _candidate_record(
    *,
    stat_key: str,
    df: pd.DataFrame,
    actual_col: str,
    baseline_col: str,
    candidate: pd.Series,
    baseline_metrics: dict[str, Any],
    baseline_brier: dict[str, Any],
) -> dict[str, Any]:
    count = _count_metrics(df[actual_col], candidate)
    any_brier = _any_brier(df[actual_col], candidate)
    rate = _rate_metrics(df, actual_col, candidate)
    mae_gain = (
        float(baseline_metrics["mae"] - count["mae"])
        if baseline_metrics.get("mae") is not None and count.get("mae") is not None
        else None
    )
    brier_gain = (
        float(baseline_brier["brier"] - any_brier["brier"])
        if baseline_brier.get("brier") is not None and any_brier.get("brier") is not None
        else None
    )
    baseline = pd.to_numeric(df[baseline_col], errors="coerce")
    return {
        "stat": stat_key,
        "rows": count.get("rows", 0),
        "mae": count.get("mae"),
        "rmse": count.get("rmse"),
        "bias": count.get("bias"),
        "mae_gain": mae_gain,
        "any_brier": any_brier.get("brier"),
        "any_brier_gain": brier_gain,
        "actual_any_rate": any_brier.get("actual_any_rate"),
        "pred_any_rate": any_brier.get("pred_any_rate"),
        "rate_bias": rate.get("rate_bias"),
        "mean_shift_vs_baseline": float((candidate - baseline).mean()),
    }


def choose_rate_blend(stat_key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((row for row in records if abs(float(row.get("alpha") or 0.0)) <= 1e-12), None)
    if baseline is None:
        return {"accepted": False, "alpha": 0.0, "reason": "missing_baseline_candidate"}
    accepted: list[dict[str, Any]] = []
    baseline_abs_bias = abs(float(baseline.get("bias") or 0.0))
    for row in records:
        alpha = float(row.get("alpha") or 0.0)
        if alpha <= 0:
            continue
        mae_gain = row.get("mae_gain")
        brier_gain = row.get("any_brier_gain")
        bias = abs(float(row.get("bias") or 0.0))
        if (
            mae_gain is not None
            and brier_gain is not None
            and mae_gain >= _MIN_MAE_GAIN[stat_key]
            and brier_gain >= _MIN_BRIER_GAIN[stat_key]
            and bias <= baseline_abs_bias + _MAX_BIAS_WORSENING[stat_key]
        ):
            accepted.append(row)
    if not accepted:
        positives = [row for row in records if float(row.get("alpha") or 0.0) > 0]
        reasons = []
        if not any((row.get("mae_gain") is not None and row["mae_gain"] >= _MIN_MAE_GAIN[stat_key]) for row in positives):
            reasons.append("mae_not_improved")
        if not any((row.get("any_brier_gain") is not None and row["any_brier_gain"] >= _MIN_BRIER_GAIN[stat_key]) for row in positives):
            reasons.append("any_brier_not_improved")
        if not reasons:
            reasons.append("bias_or_threshold_gate_failed")
        return {"accepted": False, "alpha": 0.0, "reason": ",".join(reasons)}
    best = sorted(
        accepted,
        key=lambda row: (
            -float(row.get("any_brier_gain") or 0.0),
            -float(row.get("mae_gain") or 0.0),
            float(row.get("alpha") or 0.0),
        ),
    )[0]
    return {"accepted": True, "alpha": float(best["alpha"]), "reason": "mae_and_any_brier_improved"}


def _shrunk_mean_map(train: pd.DataFrame, key: str, target: str, shrink_rows: float) -> tuple[dict[str, float], float]:
    values = pd.to_numeric(train[target], errors="coerce")
    global_mean = float(values.mean()) if values.notna().any() else 0.0
    mapping: dict[str, float] = {}
    for value, group in train.groupby(key, dropna=False):
        group_values = pd.to_numeric(group[target], errors="coerce").dropna()
        if group_values.empty:
            continue
        n = float(len(group_values))
        weight = n / (n + shrink_rows)
        mapping[str(value)] = float(weight * group_values.mean() + (1.0 - weight) * global_mean)
    return mapping, global_mean


def _apply_map(df: pd.DataFrame, key: str, mapping: dict[str, float], fallback: float) -> pd.Series:
    return df[key].astype(str).map(mapping).fillna(float(fallback)).astype(float)


def _fit_hit_bias_maps(train: pd.DataFrame) -> tuple[list[dict[str, Any]], float]:
    maps: list[dict[str, Any]] = []
    values = pd.to_numeric(train.get("hit_bias_residual"), errors="coerce")
    global_mean = float(values.mean()) if values.notna().any() else 0.0
    for key, weight, shrink_rows in _HIT_BIAS_GROUP_SPECS:
        if key not in train:
            continue
        mapping, fallback = _shrunk_mean_map(train, key, "hit_bias_residual", shrink_rows)
        maps.append({
            "key": key,
            "weight": float(weight),
            "shrink_rows": float(shrink_rows),
            "fallback": float(fallback),
            "values": mapping,
        })
    return maps, global_mean


def _score_hit_bias_maps(test: pd.DataFrame, maps: list[dict[str, Any]], global_mean: float) -> pd.Series:
    if test.empty:
        return pd.Series(dtype=float, index=test.index)
    residual = pd.Series(0.0, index=test.index, dtype=float)
    total_weight = 0.0
    for spec in maps:
        key = str(spec.get("key") or "")
        if key not in test:
            continue
        weight = float(spec.get("weight") or 0.0)
        if weight <= 0:
            continue
        residual = residual + weight * _apply_map(
            test,
            key,
            spec.get("values") or {},
            float(spec.get("fallback") if spec.get("fallback") is not None else global_mean),
        )
        total_weight += weight
    if total_weight <= 0:
        return pd.Series(float(global_mean), index=test.index, dtype=float)
    return (residual / total_weight).clip(lower=-0.90, upper=0.90)


def _hit_bias_calibration_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a leakage-safe hits residual calibration layer.

    The layer repairs count bias on one row per player-game. It is intentionally
    small and heavily shrunk, so top-order or high-PA corrections must prove
    themselves in date-forward folds before live scoring can use them.
    """
    required = {"game_date_et", "actual_hits", "model_pred_hits", "projected_pa"}
    if df.empty or not required.issubset(set(df.columns)):
        return {
            "status": "missing_required_columns",
            "accepted": False,
            "selected_alpha": 0.0,
            "candidate_records": [],
        }
    work = _add_hit_bias_buckets(df).dropna(subset=["actual_hits", "model_pred_hits", "projected_pa"]).copy()
    work = work.loc[pd.to_numeric(work["projected_pa"], errors="coerce") > 0].copy()
    if work.empty:
        return {
            "status": "no_training_rows",
            "accepted": False,
            "selected_alpha": 0.0,
            "candidate_records": [],
        }
    work["actual_hits"] = pd.to_numeric(work["actual_hits"], errors="coerce")
    work["model_pred_hits"] = pd.to_numeric(work["model_pred_hits"], errors="coerce").clip(lower=0.0, upper=5.0)
    work["hit_bias_residual"] = (work["actual_hits"] - work["model_pred_hits"]).clip(lower=-2.0, upper=2.0)
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    repaired = pd.Series(np.nan, index=work.index, dtype=float)
    raw_residual = pd.Series(np.nan, index=work.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for start in range(0, len(dates), _FOLD_TEST_DAYS):
        test_dates = dates[start:start + _FOLD_TEST_DAYS]
        if not test_dates:
            continue
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < test_dates[0]
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date.isin(test_dates)
        if int(train_mask.sum()) < _MIN_TRAIN_ROWS or int(test_mask.sum()) == 0:
            continue
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        maps, global_mean = _fit_hit_bias_maps(train)
        residual = _score_hit_bias_maps(test, maps, global_mean)
        raw_residual.loc[test.index] = residual
        repaired.loc[test.index] = (test["model_pred_hits"] + residual).clip(lower=0.0, upper=5.0)
        folds.append({
            "holdout_start": str(test_dates[0]),
            "holdout_end": str(test_dates[-1]),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
            "global_residual": float(global_mean),
        })

    valid = repaired.notna()
    if not bool(valid.any()):
        return {
            "status": "no_oof_calibration_rows",
            "accepted": False,
            "selected_alpha": 0.0,
            "folds": folds,
            "candidate_records": [],
        }
    valid_work = work.loc[valid].copy()
    baseline = pd.to_numeric(valid_work["model_pred_hits"], errors="coerce").clip(lower=0.0, upper=5.0)
    corrected = pd.to_numeric(repaired.loc[valid], errors="coerce").clip(lower=0.0, upper=5.0)
    actual = pd.to_numeric(valid_work["actual_hits"], errors="coerce")
    baseline_metrics = _count_metrics(actual, baseline)
    baseline_brier = _any_brier(actual, baseline)
    records: list[dict[str, Any]] = []
    for alpha in _HIT_BIAS_ALPHAS:
        candidate = (baseline + alpha * (corrected - baseline)).clip(lower=0.0, upper=5.0)
        records.append({
            "source": "hit_bias_calibration_v2",
            "alpha": float(alpha),
            **_candidate_record(
                stat_key="hits",
                df=valid_work,
                actual_col="actual_hits",
                baseline_col="model_pred_hits",
                candidate=candidate,
                baseline_metrics=baseline_metrics,
                baseline_brier=baseline_brier,
            ),
        })
    decision = choose_rate_blend("hits", records)
    selected_alpha = float(decision.get("alpha") or 0.0)
    production_maps, production_global = _fit_hit_bias_maps(work)
    selected = (baseline + selected_alpha * (corrected - baseline)).clip(lower=0.0, upper=5.0)
    return {
        "status": "ready",
        "accepted": bool(decision.get("accepted")),
        "selected_alpha": selected_alpha,
        "reason": decision.get("reason"),
        "rows": int(len(valid_work)),
        "dates": int(pd.to_datetime(valid_work["game_date_et"]).dt.date.nunique()),
        "folds": folds,
        "candidate_records": records,
        "baseline": {
            **baseline_metrics,
            "any_brier": baseline_brier.get("brier"),
            **_rate_metrics(valid_work, "actual_hits", baseline),
        },
        "selected": {
            **_count_metrics(actual, selected),
            "any_brier": _any_brier(actual, selected).get("brier"),
            **_rate_metrics(valid_work, "actual_hits", selected),
        },
        "production_maps": production_maps,
        "production_global_residual": float(production_global),
        "raw_residual_summary": {
            "mean": float(pd.to_numeric(raw_residual.loc[valid], errors="coerce").mean()),
            "p10": float(pd.to_numeric(raw_residual.loc[valid], errors="coerce").quantile(0.10)),
            "p90": float(pd.to_numeric(raw_residual.loc[valid], errors="coerce").quantile(0.90)),
        },
        "live_usage": (
            "eligible_for_shadow_and_future_live_gate"
            if bool(decision.get("accepted"))
            else "diagnostic_only_until_oof_improves"
        ),
    }


def _eb_residual_predictions(
    df: pd.DataFrame,
    *,
    actual_col: str,
    baseline_col: str,
    upper: float,
) -> tuple[pd.Series, dict[str, Any]]:
    work = df.dropna(subset=[actual_col, baseline_col]).copy()
    work["baseline_residual"] = (
        pd.to_numeric(work[actual_col], errors="coerce")
        - pd.to_numeric(work[baseline_col], errors="coerce")
    ).clip(lower=-upper, upper=upper)
    dates = sorted(pd.to_datetime(work["game_date_et"]).dt.date.unique())
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    folds: list[dict[str, Any]] = []
    for start in range(0, len(dates), _FOLD_TEST_DAYS):
        test_dates = dates[start:start + _FOLD_TEST_DAYS]
        if not test_dates:
            continue
        train_mask = pd.to_datetime(work["game_date_et"]).dt.date < test_dates[0]
        test_mask = pd.to_datetime(work["game_date_et"]).dt.date.isin(test_dates)
        if int(train_mask.sum()) < _MIN_TRAIN_ROWS or int(test_mask.sum()) == 0:
            continue
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        player_map, global_mean = _shrunk_mean_map(train, "player_id", "baseline_residual", 12.0)
        lineup_map, _ = _shrunk_mean_map(train, "lineup_bucket", "baseline_residual", 80.0)
        platoon_map, _ = _shrunk_mean_map(train, "platoon_bucket", "baseline_residual", 80.0)
        barrel_map, _ = _shrunk_mean_map(train, "barrel_bucket", "baseline_residual", 80.0)
        xslg_map, _ = _shrunk_mean_map(train, "xslg_bucket", "baseline_residual", 80.0)
        residual = (
            0.45 * _apply_map(test, "player_id", player_map, global_mean)
            + 0.15 * _apply_map(test, "lineup_bucket", lineup_map, global_mean)
            + 0.12 * _apply_map(test, "platoon_bucket", platoon_map, global_mean)
            + 0.14 * _apply_map(test, "barrel_bucket", barrel_map, global_mean)
            + 0.14 * _apply_map(test, "xslg_bucket", xslg_map, global_mean)
        ).clip(lower=-min(1.25, upper), upper=min(1.25, upper))
        pred.loc[test.index] = (
            pd.to_numeric(test[baseline_col], errors="coerce") + residual
        ).clip(lower=0.0, upper=upper)
        folds.append({
            "holdout_start": str(test_dates[0]),
            "holdout_end": str(test_dates[-1]),
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(test_mask.sum()),
            "global_residual": global_mean,
        })
    return pred, {
        "enabled": bool(folds),
        "model": "empirical_bayes_player_context_residual",
        "target": f"{actual_col}_minus_{baseline_col}",
        "folds": folds,
    }


def _challenger_summary(
    df: pd.DataFrame,
    *,
    stat_key: str,
    actual_col: str,
    baseline_col: str,
    upper: float,
) -> dict[str, Any]:
    raw, meta = _raw_walk_forward_predictions(df, stat_key=stat_key, actual_col=actual_col, upper=upper)
    eb_raw, eb_meta = _eb_residual_predictions(df, actual_col=actual_col, baseline_col=baseline_col, upper=upper)
    valid = raw.notna() & pd.to_numeric(df[baseline_col], errors="coerce").notna() & pd.to_numeric(df[actual_col], errors="coerce").notna()
    eb_valid = eb_raw.notna() & pd.to_numeric(df[baseline_col], errors="coerce").notna() & pd.to_numeric(df[actual_col], errors="coerce").notna()
    if not valid.any():
        return {
            "status": "no_oof_predictions",
            "accepted": False,
            "selected_alpha": 0.0,
            "raw_meta": meta,
            "eb_residual_meta": eb_meta,
            "candidate_records": [],
        }
    work = df.loc[valid].copy()
    raw = raw.loc[valid]
    baseline = pd.to_numeric(work[baseline_col], errors="coerce").clip(lower=0.0, upper=upper)
    actual = pd.to_numeric(work[actual_col], errors="coerce")
    baseline_metrics = _count_metrics(actual, baseline)
    baseline_brier = _any_brier(actual, baseline)
    raw_metrics = _count_metrics(actual, raw)
    raw_brier = _any_brier(actual, raw)

    def blend_records(source_name: str, source_pred: pd.Series, source_work: pd.DataFrame) -> list[dict[str, Any]]:
        source_baseline = pd.to_numeric(source_work[baseline_col], errors="coerce").clip(lower=0.0, upper=upper)
        source_actual = pd.to_numeric(source_work[actual_col], errors="coerce")
        source_base_metrics = _count_metrics(source_actual, source_baseline)
        source_base_brier = _any_brier(source_actual, source_baseline)
        out: list[dict[str, Any]] = []
        for alpha in _BLEND_ALPHAS:
            candidate = (source_baseline + alpha * (source_pred - source_baseline)).clip(lower=0.0, upper=upper)
            out.append({
                "source": source_name,
                "alpha": float(alpha),
                **_candidate_record(
                    stat_key=stat_key,
                    df=source_work,
                    actual_col=actual_col,
                    baseline_col=baseline_col,
                    candidate=candidate,
                    baseline_metrics=source_base_metrics,
                    baseline_brier=source_base_brier,
                ),
            })
        return out

    records = blend_records("lightgbm_rate", raw, work)
    lightgbm_records = list(records)
    decision = choose_rate_blend(stat_key, records)
    eb_records: list[dict[str, Any]] = []
    eb_decision = {"accepted": False, "alpha": 0.0, "reason": "no_oof_eb_predictions"}
    if eb_valid.any():
        eb_work = df.loc[eb_valid].copy()
        eb_series = eb_raw.loc[eb_valid]
        eb_records = blend_records("empirical_bayes_residual", eb_series, eb_work)
        eb_decision = choose_rate_blend(stat_key, eb_records)
    # Prefer the main OOF challenger when it passes on the canonical fold set.
    # EB is a fallback residual repair, not a replacement selected on a
    # different subset of rows.
    if bool(decision.get("accepted")):
        selected_source = "lightgbm_rate"
    elif bool(eb_decision.get("accepted")):
        selected_source = "empirical_bayes_residual"
        decision = eb_decision
        records = eb_records
    else:
        selected_source = "lightgbm_rate"
    alpha = float(decision.get("alpha") or 0.0)
    selected_source_pred = raw if selected_source == "lightgbm_rate" else eb_raw.loc[valid].fillna(raw)
    selected = (baseline + alpha * (selected_source_pred - baseline)).clip(lower=0.0, upper=upper)
    return {
        "status": "ready",
        "accepted": bool(decision.get("accepted")),
        "selected_source": selected_source,
        "selected_alpha": alpha,
        "reason": decision.get("reason"),
        "rows": int(len(work)),
        "dates": int(pd.to_datetime(work["game_date_et"]).dt.date.nunique()),
        "raw_meta": meta,
        "eb_residual_meta": eb_meta,
        "baseline": {
            **baseline_metrics,
            "any_brier": baseline_brier.get("brier"),
            **_rate_metrics(work, actual_col, baseline),
        },
        "raw_challenger": {
            **raw_metrics,
            "any_brier": raw_brier.get("brier"),
            **_rate_metrics(work, actual_col, raw),
        },
        "selected": {
            **_count_metrics(actual, selected),
            "any_brier": _any_brier(actual, selected).get("brier"),
            **_rate_metrics(work, actual_col, selected),
        },
        "mae_gain": records[0].get("mae") - _count_metrics(actual, selected).get("mae")
        if records and records[0].get("mae") is not None else None,
        "candidate_records": lightgbm_records,
        "eb_candidate_records": eb_records,
    }


def _summaries(df: pd.DataFrame, stat: str, predicted: str, actual: str) -> dict[str, Any]:
    work = df.dropna(subset=[predicted, actual, "projected_pa", "actual_pa"]).copy()
    work = work.loc[(work["projected_pa"] > 0) & (work["actual_pa"] > 0)]
    work["count_error"] = work[predicted] - work[actual]
    work["pred_rate"] = work[predicted] / work["projected_pa"]
    work["actual_rate"] = work[actual] / work["actual_pa"]
    work["rate_error"] = work["pred_rate"] - work["actual_rate"]

    def summarize(group: pd.DataFrame, level: str, bucket: str) -> dict[str, Any]:
        return {
            "stat": stat,
            "level": level,
            "bucket": bucket,
            "rows": int(len(group)),
            "dates": int(group["game_date_et"].nunique()),
            "count_mae": float(group["count_error"].abs().mean()),
            "count_bias": float(group["count_error"].mean()),
            "pred_rate": float(group["pred_rate"].mean()),
            "actual_rate": float(group["actual_rate"].mean()),
            "rate_bias": float(group["rate_error"].mean()),
        }

    overall = summarize(work, "overall", "all") if not work.empty else {"stat": stat, "rows": 0}
    rows: list[dict[str, Any]] = []
    for level in (
        "lineup_bucket", "platoon_bucket", "barrel_bucket", "xslg_bucket",
        "starter_quality_bucket", "home_away_bucket", "pa_outcome_bucket",
    ):
        for bucket, group in work.groupby(level, dropna=False):
            if len(group) >= 30:
                rows.append(summarize(group, level, str(bucket)))
    rows.sort(key=lambda row: (-abs(float(row["rate_bias"])), -int(row["rows"])))
    return {"overall": overall, "slices": rows}


def _hit_rate_error_decomposition(df: pd.DataFrame) -> dict[str, Any]:
    """Explain hits misses as opportunity, player-rate, and context components."""
    required = {"model_pred_hits", "actual_hits", "projected_pa", "actual_pa"}
    if df.empty or not required.issubset(set(df.columns)):
        return {"rows": 0}

    work = df.copy()
    for column in [
        "model_pred_hits", "actual_hits", "projected_pa", "actual_pa",
        "lineup_slot", "team_implied_runs", "park_run_factor", "park_babip_factor",
        "player_prior_hit_rate",
    ]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["model_pred_hits", "actual_hits", "projected_pa", "actual_pa"])
    work = work.loc[(work["projected_pa"] > 0) & (work["actual_pa"] > 0)].copy()
    if work.empty:
        return {"rows": 0}

    work["pred_rate"] = (work["model_pred_hits"] / work["projected_pa"]).clip(lower=0.0, upper=1.0)
    work["actual_rate"] = (work["actual_hits"] / work["actual_pa"]).clip(lower=0.0, upper=1.0)
    work["count_error"] = work["model_pred_hits"] - work["actual_hits"]
    work["pa_component_error"] = (work["projected_pa"] - work["actual_pa"]) * work["pred_rate"]
    work["rate_component_error"] = work["actual_pa"] * (work["pred_rate"] - work["actual_rate"])
    work["dominant_error"] = np.where(
        work["pa_component_error"].abs() >= work["rate_component_error"].abs(),
        "pa_error",
        "player_rate_error",
    )
    work.loc[work["count_error"].abs() < 0.15, "dominant_error"] = "small_error"

    slot = pd.to_numeric(work.get("lineup_slot"), errors="coerce")
    work["lineup_detail_bucket"] = np.select(
        [slot.between(1, 2), slot.between(3, 5), slot.between(6, 9)],
        ["top_order_1_2", "middle_order_3_5", "bottom_order_6_9"],
        default="slot_missing",
    )
    projected_pa = pd.to_numeric(work.get("projected_pa"), errors="coerce")
    work["projected_pa_bucket"] = np.select(
        [projected_pa >= 4.4, projected_pa.between(3.8, 4.4, inclusive="left"), projected_pa < 3.8],
        ["projected_pa_high_4_4_plus", "projected_pa_mid_3_8_4_4", "projected_pa_low_under_3_8"],
        default="projected_pa_missing",
    )
    prior = pd.to_numeric(work.get("player_prior_hit_rate"), errors="coerce")
    work["player_prior_hit_bucket"] = np.select(
        [prior >= 0.280, prior.between(0.235, 0.280, inclusive="left"), prior < 0.235],
        ["prior_hit_rate_high", "prior_hit_rate_mid", "prior_hit_rate_low"],
        default="prior_hit_rate_missing",
    )
    batter_hand = work.get("batter_hand", pd.Series(index=work.index, dtype=object)).fillna("unknown").astype(str).str.upper()
    pitcher_hand = work.get("opp_sp_hand", pd.Series(index=work.index, dtype=object)).fillna("unknown").astype(str).str.upper()
    work["handedness_bucket"] = np.where(
        batter_hand.isin(["L", "R"]) & pitcher_hand.isin(["L", "R"]),
        np.where(batter_hand.eq(pitcher_hand), "same_hand", "opposite_hand"),
        "handedness_missing",
    )
    team_runs = pd.to_numeric(work.get("team_implied_runs"), errors="coerce")
    work["run_environment_bucket"] = np.select(
        [team_runs >= 4.8, team_runs.between(4.0, 4.8, inclusive="left"), team_runs < 4.0],
        ["team_total_high", "team_total_mid", "team_total_low"],
        default="team_total_missing",
    )
    park = pd.to_numeric(work.get("park_babip_factor"), errors="coerce")
    park = park.fillna(pd.to_numeric(work.get("park_run_factor"), errors="coerce"))
    work["park_hit_bucket"] = np.select(
        [park >= 1.03, park.between(0.97, 1.03, inclusive="left"), park < 0.97],
        ["hit_park_boost", "hit_park_neutral", "hit_park_suppress"],
        default="hit_park_missing",
    )
    work["under_projection_flag"] = work["count_error"] <= -0.25
    work["top_order_under_projection"] = (
        work["under_projection_flag"] & work["lineup_detail_bucket"].eq("top_order_1_2")
    )
    work["high_pa_under_projection"] = (
        work["under_projection_flag"] & work["projected_pa_bucket"].eq("projected_pa_high_4_4_plus")
    )
    work["high_prior_under_projection"] = (
        work["under_projection_flag"] & work["player_prior_hit_bucket"].eq("prior_hit_rate_high")
    )
    work["missing_context_flag"] = (
        work["handedness_bucket"].eq("handedness_missing")
        | work["run_environment_bucket"].eq("team_total_missing")
        | work["park_hit_bucket"].eq("hit_park_missing")
    )
    if "player_prior_hit_rate" in work.columns:
        work["prior_pred_hits"] = work["player_prior_hit_rate"] * work["projected_pa"]
        work["prior_count_error"] = work["prior_pred_hits"] - work["actual_hits"]
        work["model_vs_prior_hits_delta"] = work["model_pred_hits"] - work["prior_pred_hits"]
    else:
        work["prior_pred_hits"] = np.nan
        work["prior_count_error"] = np.nan
        work["model_vs_prior_hits_delta"] = np.nan

    def _safe_mean(series: pd.Series) -> Optional[float]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.mean())

    def _slice_summary(group: pd.DataFrame, level: str, bucket: str) -> dict[str, Any]:
        return {
            "level": level,
            "bucket": bucket,
            "rows": int(len(group)),
            "dates": int(pd.to_datetime(group["game_date_et"]).dt.date.nunique()) if "game_date_et" in group else 0,
            "count_mae": _safe_mean(group["count_error"].abs()),
            "count_bias": _safe_mean(group["count_error"]),
            "pred_rate": _safe_mean(group["pred_rate"]),
            "actual_rate": _safe_mean(group["actual_rate"]),
            "rate_bias": _safe_mean(group["pred_rate"] - group["actual_rate"]),
            "pa_component_bias": _safe_mean(group["pa_component_error"]),
            "rate_component_bias": _safe_mean(group["rate_component_error"]),
            "under_projection_rate": _safe_mean(group["under_projection_flag"].astype(float)),
        }

    overall = _slice_summary(work, "overall", "all")
    overall.update({
        "top_order_under_projection_rows": int(work["top_order_under_projection"].sum()),
        "high_pa_under_projection_rows": int(work["high_pa_under_projection"].sum()),
        "high_prior_under_projection_rows": int(work["high_prior_under_projection"].sum()),
        "missing_context_rows": int(work["missing_context_flag"].sum()),
    })

    dominant_rows: list[dict[str, Any]] = []
    for bucket, group in work.groupby("dominant_error", dropna=False):
        dominant_rows.append(_slice_summary(group, "dominant_error", str(bucket)))
    dominant_rows.sort(key=lambda row: (-int(row["rows"]), str(row["bucket"])))

    slice_rows: list[dict[str, Any]] = []
    for level in (
        "lineup_detail_bucket",
        "projected_pa_bucket",
        "player_prior_hit_bucket",
        "handedness_bucket",
        "run_environment_bucket",
        "park_hit_bucket",
    ):
        for bucket, group in work.groupby(level, dropna=False):
            if len(group) >= 25:
                slice_rows.append(_slice_summary(group, level, str(bucket)))
    slice_rows.sort(
        key=lambda row: (
            -float(row.get("under_projection_rate") or 0.0),
            -abs(float(row.get("rate_bias") or 0.0)),
            -int(row["rows"]),
        )
    )

    top_under = work.sort_values("count_error", ascending=True).head(25)
    top_rows: list[dict[str, Any]] = []
    for row in top_under.to_dict(orient="records"):
        top_rows.append({
            "game_date_et": str(row.get("game_date_et")),
            "game_slug": row.get("game_slug"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "team_abbr": row.get("team_abbr"),
            "opponent_abbr": row.get("opponent_abbr"),
            "lineup_slot": _safe_mean(pd.Series([row.get("lineup_slot")])),
            "projected_pa": _safe_mean(pd.Series([row.get("projected_pa")])),
            "actual_pa": _safe_mean(pd.Series([row.get("actual_pa")])),
            "pred_hits": _safe_mean(pd.Series([row.get("model_pred_hits")])),
            "actual_hits": _safe_mean(pd.Series([row.get("actual_hits")])),
            "count_error": _safe_mean(pd.Series([row.get("count_error")])),
            "pred_rate": _safe_mean(pd.Series([row.get("pred_rate")])),
            "actual_rate": _safe_mean(pd.Series([row.get("actual_rate")])),
            "player_prior_hit_rate": _safe_mean(pd.Series([row.get("player_prior_hit_rate")])),
            "prior_pred_hits": _safe_mean(pd.Series([row.get("prior_pred_hits")])),
            "model_vs_prior_hits_delta": _safe_mean(pd.Series([row.get("model_vs_prior_hits_delta")])),
            "dominant_error": row.get("dominant_error"),
            "lineup_bucket": row.get("lineup_detail_bucket"),
            "projected_pa_bucket": row.get("projected_pa_bucket"),
            "player_prior_hit_bucket": row.get("player_prior_hit_bucket"),
            "handedness_bucket": row.get("handedness_bucket"),
            "run_environment_bucket": row.get("run_environment_bucket"),
            "park_hit_bucket": row.get("park_hit_bucket"),
        })

    return {
        "rows": int(len(work)),
        "overall": overall,
        "dominant_error_counts": dominant_rows,
        "slices": slice_rows,
        "top_under_projected": top_rows,
    }


def _fmt(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}"


def build(lookback_days: int = 365, pg_dsn: str = PG_DSN) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, lookback_days))
    with psycopg2.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL, {"cutoff": cutoff})
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    df = _bucketize(pd.DataFrame(rows, columns=columns)) if rows else pd.DataFrame()
    if not df.empty:
        df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
        df, _player_prior_state = add_leakage_safe_player_priors(df)
    hits_bias_calibration = _hit_bias_calibration_summary(df) if not df.empty else {}
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "one_row_per_player_game",
        "usage": "challenger_diagnostic_only",
        "rows": int(len(df)),
        "hits": _summaries(df, "batter_hits", "model_pred_hits", "actual_hits") if not df.empty else {},
        "hits_error_decomposition": _hit_rate_error_decomposition(df) if not df.empty else {},
        "home_runs": _summaries(df, "batter_home_runs", "model_pred_home_runs", "actual_home_runs") if not df.empty else {},
        "challengers": {
            "hits": _challenger_summary(
                df,
                stat_key="hits",
                actual_col="actual_hits",
                baseline_col="model_pred_hits",
                upper=5.0,
            ) if not df.empty else {},
            "home_runs": _challenger_summary(
                df,
                stat_key="home_runs",
                actual_col="actual_home_runs",
                baseline_col="model_pred_home_runs",
                upper=3.0,
            ) if not df.empty else {},
        },
        "hits_bias_calibration_v2": hits_bias_calibration,
    }
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / "hitter_player_rate_diagnostic.json", payload)
    atomic_write_json(_MODEL_DIR / "hitter_hits_bias_calibration_v2.json", hits_bias_calibration)
    lines = [
        "# MLB Hitter Player-Rate Diagnostic", "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "Evidence: one row per player-game; this report cannot replace production models.", "",
        "## Conservative Player-Rate Challengers", "",
        "| Stat | Accepted | Source | Reason | Rows | Dates | Alpha | Base MAE | Raw MAE | Selected MAE | Base Any Brier | Raw Any Brier | Selected Any Brier | Selected Rate Bias |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("hits", "Hits"), ("home_runs", "Home Runs")):
        rec = (payload.get("challengers") or {}).get(key) or {}
        base = rec.get("baseline") or {}
        raw = rec.get("raw_challenger") or {}
        selected = rec.get("selected") or {}
        lines.append(
            f"| {label} | {bool(rec.get('accepted'))} | {rec.get('selected_source') or '-'} | "
            f"{rec.get('reason') or rec.get('status') or '-'} | "
            f"{rec.get('rows', 0)} | {rec.get('dates', 0)} | {_fmt(rec.get('selected_alpha'), 2)} | "
            f"{_fmt(base.get('mae'))} | {_fmt(raw.get('mae'))} | {_fmt(selected.get('mae'))} | "
            f"{_fmt(base.get('any_brier'), 5)} | {_fmt(raw.get('any_brier'), 5)} | "
            f"{_fmt(selected.get('any_brier'), 5)} | {_fmt(selected.get('rate_bias'), 5, signed=True)} |"
        )
    lines.extend([
        "",
        "| Stat | Source | Alpha | MAE | MAE Gain | Bias | Any Brier | Brier Gain | Rate Bias | Mean Shift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("hits", "Hits"), ("home_runs", "Home Runs")):
        for row in ((payload.get("challengers") or {}).get(key) or {}).get("candidate_records") or []:
            lines.append(
                f"| {label} | {row.get('source') or 'lightgbm_rate'} | {_fmt(row.get('alpha'), 2)} | {_fmt(row.get('mae'))} | "
                f"{_fmt(row.get('mae_gain'), signed=True)} | {_fmt(row.get('bias'), signed=True)} | "
                f"{_fmt(row.get('any_brier'), 5)} | {_fmt(row.get('any_brier_gain'), 5, signed=True)} | "
                f"{_fmt(row.get('rate_bias'), 5, signed=True)} | {_fmt(row.get('mean_shift_vs_baseline'), 4, signed=True)} |"
            )
        for row in ((payload.get("challengers") or {}).get(key) or {}).get("eb_candidate_records") or []:
            lines.append(
                f"| {label} | {row.get('source') or 'empirical_bayes_residual'} | {_fmt(row.get('alpha'), 2)} | {_fmt(row.get('mae'))} | "
                f"{_fmt(row.get('mae_gain'), signed=True)} | {_fmt(row.get('bias'), signed=True)} | "
                f"{_fmt(row.get('any_brier'), 5)} | {_fmt(row.get('any_brier_gain'), 5, signed=True)} | "
                f"{_fmt(row.get('rate_bias'), 5, signed=True)} | {_fmt(row.get('mean_shift_vs_baseline'), 4, signed=True)} |"
            )
    lines.append("")
    cal = payload.get("hits_bias_calibration_v2") or {}
    cal_base = cal.get("baseline") or {}
    cal_sel = cal.get("selected") or {}
    lines.extend([
        "## Hits Bias Calibration V2",
        "",
        "This is a player-game residual repair by lineup slot, projected PA, handedness, player prior, park, and run environment.",
        "",
        "| Accepted | Reason | Rows | Dates | Alpha | Base MAE | Selected MAE | MAE Gain | Base Brier | Selected Brier | Brier Gain | Selected Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    cal_mae_gain = (
        float(cal_base["mae"] - cal_sel["mae"])
        if cal_base.get("mae") is not None and cal_sel.get("mae") is not None
        else None
    )
    cal_brier_gain = (
        float(cal_base["any_brier"] - cal_sel["any_brier"])
        if cal_base.get("any_brier") is not None and cal_sel.get("any_brier") is not None
        else None
    )
    lines.append(
        f"| {bool(cal.get('accepted'))} | {cal.get('reason') or cal.get('status') or '-'} | "
        f"{cal.get('rows', 0)} | {cal.get('dates', 0)} | {_fmt(cal.get('selected_alpha'), 2)} | "
        f"{_fmt(cal_base.get('mae'))} | {_fmt(cal_sel.get('mae'))} | {_fmt(cal_mae_gain, signed=True)} | "
        f"{_fmt(cal_base.get('any_brier'), 5)} | {_fmt(cal_sel.get('any_brier'), 5)} | "
        f"{_fmt(cal_brier_gain, 5, signed=True)} | {_fmt(cal_sel.get('bias'), signed=True)} |"
    )
    lines.extend([
        "",
        "| Source | Alpha | MAE | MAE Gain | Bias | Any Brier | Brier Gain | Rate Bias | Mean Shift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cal.get("candidate_records") or []:
        lines.append(
            f"| {row.get('source') or 'hit_bias_calibration_v2'} | {_fmt(row.get('alpha'), 2)} | "
            f"{_fmt(row.get('mae'))} | {_fmt(row.get('mae_gain'), signed=True)} | "
            f"{_fmt(row.get('bias'), signed=True)} | {_fmt(row.get('any_brier'), 5)} | "
            f"{_fmt(row.get('any_brier_gain'), 5, signed=True)} | "
            f"{_fmt(row.get('rate_bias'), 5, signed=True)} | "
            f"{_fmt(row.get('mean_shift_vs_baseline'), 4, signed=True)} |"
        )
    lines.append("")
    for key, label in (("hits", "Hits"), ("home_runs", "Home Runs")):
        section = payload.get(key) or {}
        overall = section.get("overall") or {}
        lines.extend([
            f"## {label}", "",
            f"Overall rows: {overall.get('rows', 0)}; MAE: {overall.get('count_mae', '-')}; count bias: {overall.get('count_bias', '-')}; rate bias: {overall.get('rate_bias', '-')}", "",
            "| Feature | Bucket | Rows | Dates | Count MAE | Count Bias | Pred/PA | Actual/PA | Rate Bias |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (section.get("slices") or [])[:30]:
            lines.append(
                f"| {row['level']} | {row['bucket']} | {row['rows']} | {row['dates']} | "
                f"{row['count_mae']:.3f} | {row['count_bias']:+.3f} | {row['pred_rate']:.4f} | "
                f"{row['actual_rate']:.4f} | {row['rate_bias']:+.4f} |"
            )
        lines.append("")
        if key == "hits":
            decomp = payload.get("hits_error_decomposition") or {}
            decomp_overall = decomp.get("overall") or {}
            lines.extend([
                "### Hits Error Decomposition",
                "",
                "Miss decomposition: count miss = PA/opportunity component + per-PA hit-rate component.",
                "",
                f"Rows: {decomp.get('rows', 0)}; count bias: {_fmt(decomp_overall.get('count_bias'), signed=True)}; "
                f"rate bias: {_fmt(decomp_overall.get('rate_bias'), 5, signed=True)}; "
                f"PA component bias: {_fmt(decomp_overall.get('pa_component_bias'), signed=True)}; "
                f"rate component bias: {_fmt(decomp_overall.get('rate_component_bias'), signed=True)}",
                "",
                f"Top-order under-projection rows: {decomp_overall.get('top_order_under_projection_rows', 0)}; "
                f"high-PA under-projection rows: {decomp_overall.get('high_pa_under_projection_rows', 0)}; "
                f"high-prior under-projection rows: {decomp_overall.get('high_prior_under_projection_rows', 0)}; "
                f"missing-context rows: {decomp_overall.get('missing_context_rows', 0)}",
                "",
                "| Component | Rows | Count MAE | Count Bias | Pred/PA | Actual/PA | Rate Bias | PA Bias | Rate Bias Component | Under-Proj Rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in decomp.get("dominant_error_counts") or []:
                lines.append(
                    f"| {row.get('bucket')} | {row.get('rows', 0)} | {_fmt(row.get('count_mae'))} | "
                    f"{_fmt(row.get('count_bias'), signed=True)} | {_fmt(row.get('pred_rate'), 4)} | "
                    f"{_fmt(row.get('actual_rate'), 4)} | {_fmt(row.get('rate_bias'), 5, signed=True)} | "
                    f"{_fmt(row.get('pa_component_bias'), signed=True)} | "
                    f"{_fmt(row.get('rate_component_bias'), signed=True)} | "
                    f"{_fmt(row.get('under_projection_rate'), 3)} |"
                )
            lines.extend([
                "",
                "| Slice | Bucket | Rows | Count Bias | Rate Bias | PA Bias | Rate Component Bias | Under-Proj Rate |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ])
            for row in (decomp.get("slices") or [])[:35]:
                lines.append(
                    f"| {row.get('level')} | {row.get('bucket')} | {row.get('rows', 0)} | "
                    f"{_fmt(row.get('count_bias'), signed=True)} | "
                    f"{_fmt(row.get('rate_bias'), 5, signed=True)} | "
                    f"{_fmt(row.get('pa_component_bias'), signed=True)} | "
                    f"{_fmt(row.get('rate_component_bias'), signed=True)} | "
                    f"{_fmt(row.get('under_projection_rate'), 3)} |"
                )
            lines.extend([
                "",
                "| Player | Date | Slot | Proj PA | Actual PA | Pred H | Actual H | Error | Prior H/PA | Prior Pred H | Dominant | Context |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ])
            for row in (decomp.get("top_under_projected") or [])[:15]:
                context = ", ".join(
                    str(row.get(key) or "-")
                    for key in ("lineup_bucket", "projected_pa_bucket", "player_prior_hit_bucket", "handedness_bucket")
                )
                lines.append(
                    f"| {row.get('player_name') or row.get('player_id')} | {row.get('game_date_et')} | "
                    f"{_fmt(row.get('lineup_slot'), 0)} | {_fmt(row.get('projected_pa'), 2)} | "
                    f"{_fmt(row.get('actual_pa'), 2)} | {_fmt(row.get('pred_hits'), 2)} | "
                    f"{_fmt(row.get('actual_hits'), 0)} | {_fmt(row.get('count_error'), 2, signed=True)} | "
                    f"{_fmt(row.get('player_prior_hit_rate'), 4)} | {_fmt(row.get('prior_pred_hits'), 2)} | "
                    f"{row.get('dominant_error') or '-'} | {context} |"
                )
            lines.append("")
    report_path = _REPORT_DIR / "mlb_hitter_player_rate_diagnostic_latest.md"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    payload["report_path"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose hitter player-rate errors by feature slice")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    payload = build(max(1, args.lookback_days), args.pg_dsn)
    print(json.dumps({"rows": payload["rows"], "report_path": payload["report_path"]}, indent=2))


if __name__ == "__main__":
    main()
