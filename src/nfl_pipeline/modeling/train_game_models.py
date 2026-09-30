"""Train NFL spread and total projection models."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from nfl_pipeline.db import PG_DSN

log = logging.getLogger("nfl_pipeline.modeling.train_game_models")

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "game_bets"


@dataclass(frozen=True)
class TrainGameConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    min_rows: int = 200
    holdout_weeks: int = 4
    random_state: int = 42
    out_file: str = "nfl_game_models.joblib"
    report_file: str = "nfl_game_models_report.json"


SQL_TRAIN = """
SELECT *
FROM features.nfl_game_training_features
WHERE home_margin IS NOT NULL
  AND total_points_actual IS NOT NULL
  AND game_date_et IS NOT NULL
ORDER BY season, week, game_id
"""

TARGETS = ("home_margin", "total_points_actual")
ID_COLS = {
    "game_id", "game_date_et", "start_ts_utc", "home_team_abbr", "away_team_abbr",
    "created_at_utc", "updated_at_utc",
}
LEAKY_COLS = {"home_score", "away_score", *TARGETS}

GAME_TARGET_POLICIES: dict[str, tuple[str, ...]] = {
    "home_margin": (
        "market_spread", "home_margin", "away_margin", "offense_pf_diff", "defense_pa_diff",
        "play_volume_diff", "pass_rate_diff", "ypp_diff", "ypa_diff", "ypc_diff",
        "td_rate_diff", "red_zone_td_rate_diff", "rest_diff", "injury_risk_diff",
        "qb_injury_risk_diff", "skill_injury_score_diff", "ol_injury_score_diff",
        "home_implied_points", "away_implied_points", "season", "week",
    ),
    "total_points_actual": (
        "market_total", "market_total_centered", "home_total", "away_total",
        "offense_pf_sum", "defense_pa_sum", "pace_total_sum", "play_volume_sum",
        "pass_rate_sum", "ypp_sum", "ypa_sum", "ypc_sum", "td_rate_sum",
        "red_zone_td_rate_sum", "red_zone_volume_sum", "red_zone_scoring_signal",
        "yardage_volume_signal", "td_volume_signal", "pass_explosive_signal",
        "rush_efficiency_balance_signal", "pace_market_total_interaction",
        "td_rate_market_total_interaction", "yardage_market_residual_signal",
        "td_market_residual_signal", "red_zone_market_residual_signal",
        "pace_pass_explosive_trend_signal", "red_zone_td_trend_sum",
        "total_suppression_signal", "total_upside_signal", "market_total_gap_signal",
        "total_pace_efficiency_v2_signal", "total_weather_injury_drag_v2",
        "total_market_residual_v2_signal", "total_red_zone_explosive_v2_signal",
        "weather", "wind", "temp", "cold", "roof", "surface",
        "injury_total_drag", "qb_injury_total", "skill_injury_total", "ol_injury_total",
        "season", "week",
    ),
}


def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if is_numeric_dtype(out[col]) or is_bool_dtype(out[col]) or is_datetime64_any_dtype(out[col]):
            continue
        if out[col].dtype == "object" or str(out[col].dtype).startswith("string"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def make_game_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[col for col in ID_COLS | LEAKY_COLS if col in df.columns]).copy()
    def num(name: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(X.get(name, pd.Series(default, index=X.index)), errors="coerce").fillna(default)

    if {"home_pf_avg_5", "away_pf_avg_5"}.issubset(X.columns):
        X["offense_pf_sum_avg_5"] = pd.to_numeric(X["home_pf_avg_5"], errors="coerce") + pd.to_numeric(X["away_pf_avg_5"], errors="coerce")
        X["offense_pf_diff_avg_5"] = pd.to_numeric(X["home_pf_avg_5"], errors="coerce") - pd.to_numeric(X["away_pf_avg_5"], errors="coerce")
    if {"home_pa_avg_5", "away_pa_avg_5"}.issubset(X.columns):
        X["defense_pa_sum_avg_5"] = pd.to_numeric(X["home_pa_avg_5"], errors="coerce") + pd.to_numeric(X["away_pa_avg_5"], errors="coerce")
        X["defense_pa_diff_avg_5"] = pd.to_numeric(X["home_pa_avg_5"], errors="coerce") - pd.to_numeric(X["away_pa_avg_5"], errors="coerce")
    if {"home_total_avg_5", "away_total_avg_5"}.issubset(X.columns):
        X["pace_total_sum_avg_5"] = pd.to_numeric(X["home_total_avg_5"], errors="coerce") + pd.to_numeric(X["away_total_avg_5"], errors="coerce")
    if {"home_plays_avg_5", "away_plays_avg_5"}.issubset(X.columns):
        X["play_volume_sum_avg_5"] = pd.to_numeric(X["home_plays_avg_5"], errors="coerce") + pd.to_numeric(X["away_plays_avg_5"], errors="coerce")
        X["play_volume_diff_avg_5"] = pd.to_numeric(X["home_plays_avg_5"], errors="coerce") - pd.to_numeric(X["away_plays_avg_5"], errors="coerce")
    if {"home_pass_rate_avg_5", "away_pass_rate_avg_5"}.issubset(X.columns):
        X["pass_rate_sum_avg_5"] = pd.to_numeric(X["home_pass_rate_avg_5"], errors="coerce") + pd.to_numeric(X["away_pass_rate_avg_5"], errors="coerce")
        X["pass_rate_diff_avg_5"] = pd.to_numeric(X["home_pass_rate_avg_5"], errors="coerce") - pd.to_numeric(X["away_pass_rate_avg_5"], errors="coerce")
    if {"home_yards_per_play_avg_5", "away_yards_per_play_avg_5"}.issubset(X.columns):
        X["ypp_sum_avg_5"] = pd.to_numeric(X["home_yards_per_play_avg_5"], errors="coerce") + pd.to_numeric(X["away_yards_per_play_avg_5"], errors="coerce")
        X["ypp_diff_avg_5"] = pd.to_numeric(X["home_yards_per_play_avg_5"], errors="coerce") - pd.to_numeric(X["away_yards_per_play_avg_5"], errors="coerce")
    if {"home_yards_per_pass_avg_5", "away_yards_per_pass_avg_5"}.issubset(X.columns):
        X["ypa_sum_avg_5"] = pd.to_numeric(X["home_yards_per_pass_avg_5"], errors="coerce") + pd.to_numeric(X["away_yards_per_pass_avg_5"], errors="coerce")
        X["ypa_diff_avg_5"] = pd.to_numeric(X["home_yards_per_pass_avg_5"], errors="coerce") - pd.to_numeric(X["away_yards_per_pass_avg_5"], errors="coerce")
    if {"home_yards_per_carry_avg_5", "away_yards_per_carry_avg_5"}.issubset(X.columns):
        X["ypc_sum_avg_5"] = pd.to_numeric(X["home_yards_per_carry_avg_5"], errors="coerce") + pd.to_numeric(X["away_yards_per_carry_avg_5"], errors="coerce")
        X["ypc_diff_avg_5"] = pd.to_numeric(X["home_yards_per_carry_avg_5"], errors="coerce") - pd.to_numeric(X["away_yards_per_carry_avg_5"], errors="coerce")
    if {"home_tds_per_play_avg_5", "away_tds_per_play_avg_5"}.issubset(X.columns):
        X["td_rate_sum_avg_5"] = pd.to_numeric(X["home_tds_per_play_avg_5"], errors="coerce") + pd.to_numeric(X["away_tds_per_play_avg_5"], errors="coerce")
        X["td_rate_diff_avg_5"] = pd.to_numeric(X["home_tds_per_play_avg_5"], errors="coerce") - pd.to_numeric(X["away_tds_per_play_avg_5"], errors="coerce")
    if {"home_red_zone_td_rate_avg_5", "away_red_zone_td_rate_avg_5"}.issubset(X.columns):
        X["red_zone_td_rate_sum_avg_5"] = num("home_red_zone_td_rate_avg_5") + num("away_red_zone_td_rate_avg_5")
        X["red_zone_td_rate_diff_avg_5"] = num("home_red_zone_td_rate_avg_5") - num("away_red_zone_td_rate_avg_5")
    if {"home_red_zone_td_rate_avg_3", "home_red_zone_td_rate_avg_10", "away_red_zone_td_rate_avg_3", "away_red_zone_td_rate_avg_10"}.issubset(X.columns):
        X["red_zone_td_trend_sum"] = (
            num("home_red_zone_td_rate_avg_3") - num("home_red_zone_td_rate_avg_10")
            + num("away_red_zone_td_rate_avg_3") - num("away_red_zone_td_rate_avg_10")
        )
    if {"home_red_zone_plays_avg_5", "away_red_zone_plays_avg_5"}.issubset(X.columns):
        X["red_zone_volume_sum_avg_5"] = num("home_red_zone_plays_avg_5") + num("away_red_zone_plays_avg_5")
    if {"home_qb_injury_risk", "away_qb_injury_risk"}.issubset(X.columns):
        X["qb_injury_total"] = num("home_qb_injury_risk") + num("away_qb_injury_risk")
        X["qb_injury_risk_diff"] = num("home_qb_injury_risk") - num("away_qb_injury_risk")
    if {"home_ol_injury_score", "away_ol_injury_score"}.issubset(X.columns):
        X["ol_injury_total"] = num("home_ol_injury_score") + num("away_ol_injury_score")
        X["ol_injury_score_diff"] = num("home_ol_injury_score") - num("away_ol_injury_score")
    if {"home_skill_injury_score", "away_skill_injury_score"}.issubset(X.columns):
        X["skill_injury_total"] = num("home_skill_injury_score") + num("away_skill_injury_score")
        X["skill_injury_score_diff"] = num("home_skill_injury_score") - num("away_skill_injury_score")
    if {"home_total_injury_score", "away_total_injury_score"}.issubset(X.columns):
        X["injury_total_drag"] = (num("home_total_injury_score") + num("away_total_injury_score")).clip(0.0, 8.0)
        X["injury_risk_diff"] = num("home_total_injury_score") - num("away_total_injury_score")
    if {"home_plays_avg_3", "home_plays_avg_10", "away_plays_avg_3", "away_plays_avg_10"}.issubset(X.columns):
        X["play_volume_trend_sum"] = (num("home_plays_avg_3") - num("home_plays_avg_10")) + (num("away_plays_avg_3") - num("away_plays_avg_10"))
    if {"home_pass_rate_avg_3", "home_pass_rate_avg_10", "away_pass_rate_avg_3", "away_pass_rate_avg_10"}.issubset(X.columns):
        X["pass_rate_trend_sum"] = (num("home_pass_rate_avg_3") - num("home_pass_rate_avg_10")) + (num("away_pass_rate_avg_3") - num("away_pass_rate_avg_10"))
    if {"home_yards_per_play_avg_3", "home_yards_per_play_avg_10", "away_yards_per_play_avg_3", "away_yards_per_play_avg_10"}.issubset(X.columns):
        X["explosive_efficiency_trend_sum"] = (num("home_yards_per_play_avg_3") - num("home_yards_per_play_avg_10")) + (num("away_yards_per_play_avg_3") - num("away_yards_per_play_avg_10"))
    if {"play_volume_trend_sum", "pass_rate_trend_sum", "explosive_efficiency_trend_sum"}.issubset(X.columns):
        X["pace_pass_explosive_trend_signal"] = (
            0.45 * num("play_volume_trend_sum")
            + 12.0 * num("pass_rate_trend_sum")
            + 7.5 * num("explosive_efficiency_trend_sum")
        )
    if {"play_volume_sum_avg_5", "ypp_sum_avg_5"}.issubset(X.columns):
        X["yardage_volume_signal"] = num("play_volume_sum_avg_5") * num("ypp_sum_avg_5")
    if {"play_volume_sum_avg_5", "td_rate_sum_avg_5"}.issubset(X.columns):
        X["td_volume_signal"] = num("play_volume_sum_avg_5") * num("td_rate_sum_avg_5")
    if {"red_zone_volume_sum_avg_5", "red_zone_td_rate_sum_avg_5"}.issubset(X.columns):
        X["red_zone_scoring_signal"] = num("red_zone_volume_sum_avg_5") * num("red_zone_td_rate_sum_avg_5")
    if {"pass_rate_sum_avg_5", "ypa_sum_avg_5"}.issubset(X.columns):
        X["pass_explosive_signal"] = num("pass_rate_sum_avg_5") * num("ypa_sum_avg_5")
    if {"ypc_sum_avg_5", "pass_rate_sum_avg_5"}.issubset(X.columns):
        X["rush_efficiency_balance_signal"] = num("ypc_sum_avg_5") * (2.0 - num("pass_rate_sum_avg_5").clip(0.0, 2.0))
    if {"home_rest_days", "away_rest_days"}.issubset(X.columns):
        X["rest_diff"] = pd.to_numeric(X["home_rest_days"], errors="coerce") - pd.to_numeric(X["away_rest_days"], errors="coerce")
    if "market_spread_home" in X.columns:
        spread = pd.to_numeric(X["market_spread_home"], errors="coerce")
        X["market_spread_abs"] = spread.abs()
    if "market_total" in X.columns:
        total = pd.to_numeric(X["market_total"], errors="coerce")
        X["market_total_centered"] = total - total.median()
    if {"market_total", "market_spread_home"}.issubset(X.columns):
        total = pd.to_numeric(X["market_total"], errors="coerce")
        spread = pd.to_numeric(X["market_spread_home"], errors="coerce")
        X["home_implied_points"] = total / 2.0 - spread / 2.0
        X["away_implied_points"] = total / 2.0 + spread / 2.0
        if "td_rate_sum_avg_5" in X.columns:
            X["td_rate_market_total_interaction"] = X["td_rate_sum_avg_5"] * (total / 44.0).clip(0.70, 1.35)
        if "play_volume_sum_avg_5" in X.columns:
            X["pace_market_total_interaction"] = X["play_volume_sum_avg_5"] * (total / 44.0).clip(0.70, 1.35)
        if "yardage_volume_signal" in X.columns:
            X["yardage_market_residual_signal"] = X["yardage_volume_signal"] - (total * 14.0)
        if "td_volume_signal" in X.columns:
            X["td_market_residual_signal"] = X["td_volume_signal"] - (total / 7.0)
        if "red_zone_scoring_signal" in X.columns:
            X["red_zone_market_residual_signal"] = X["red_zone_scoring_signal"] - (total / 14.0)
        if {"pace_pass_explosive_trend_signal", "red_zone_td_trend_sum"}.issubset(X.columns):
            X["market_total_gap_signal"] = (
                num("pace_pass_explosive_trend_signal")
                + 18.0 * num("red_zone_td_trend_sum")
                - num("market_total_centered")
            )
    if "wind" in X.columns:
        wind = pd.to_numeric(X["wind"], errors="coerce")
        X["wind_over_12"] = (wind >= 12).astype(float)
        X["wind_over_18"] = (wind >= 18).astype(float)
        if "market_total" in X.columns:
            X["wind_market_total_drag"] = (wind.fillna(0.0).clip(lower=0.0) / 20.0).clip(0.0, 1.0) * pd.to_numeric(X["market_total"], errors="coerce")
    if "temp" in X.columns:
        temp = pd.to_numeric(X["temp"], errors="coerce")
        X["cold_under_35"] = (temp <= 35).astype(float)
        X["temperature_scoring_drag"] = np.where(temp <= 32, (32 - temp.fillna(32.0)).clip(0.0, 35.0) / 35.0, 0.0)
    if {"wind_market_total_drag", "temperature_scoring_drag"}.issubset(X.columns):
        X["weather_under_signal"] = num("wind_market_total_drag") + 8.0 * num("temperature_scoring_drag")
    if {"weather_under_signal", "injury_total_drag"}.issubset(X.columns):
        X["total_environment_drag_signal"] = num("weather_under_signal") + 1.5 * num("injury_total_drag")
    if {"total_environment_drag_signal", "market_total_centered"}.issubset(X.columns):
        X["total_suppression_signal"] = (
            num("total_environment_drag_signal")
            + 1.5 * num("qb_injury_total")
            + 0.8 * num("ol_injury_total")
            + 0.20 * num("market_total_centered").clip(lower=0.0)
        )
    if {"pace_pass_explosive_trend_signal", "red_zone_td_trend_sum"}.issubset(X.columns):
        X["total_upside_signal"] = (
            num("pace_pass_explosive_trend_signal")
            + 20.0 * num("red_zone_td_trend_sum")
            - 0.75 * num("injury_total_drag")
            - 0.50 * num("weather_under_signal")
        )
    if {"play_volume_sum_avg_5", "ypp_sum_avg_5", "td_rate_sum_avg_5"}.issubset(X.columns):
        pace = (num("play_volume_sum_avg_5") - 124.0) / 16.0
        efficiency = num("ypp_sum_avg_5") - 10.8
        td_rate = 95.0 * (num("td_rate_sum_avg_5") - 0.045)
        X["total_pace_efficiency_v2_signal"] = (
            5.5 * pace
            + 2.2 * efficiency
            + td_rate
            + 0.35 * num("pace_pass_explosive_trend_signal")
        )
    if {"weather_under_signal", "injury_total_drag"}.issubset(X.columns):
        X["total_weather_injury_drag_v2"] = (
            num("weather_under_signal")
            + 1.4 * num("injury_total_drag")
            + 1.8 * num("qb_injury_total")
            + 0.9 * num("ol_injury_total")
        )
    if {"red_zone_scoring_signal", "pass_explosive_signal"}.issubset(X.columns):
        X["total_red_zone_explosive_v2_signal"] = (
            6.5 * num("red_zone_scoring_signal")
            + 2.0 * num("pass_explosive_signal")
            + 0.45 * num("red_zone_td_trend_sum")
            + 0.18 * num("explosive_efficiency_trend_sum")
        )
    if "market_total" in X.columns:
        model_total_signal = (
            0.09 * num("yardage_volume_signal")
            + 4.2 * num("td_volume_signal")
            + 1.6 * num("red_zone_scoring_signal")
            + 0.30 * num("total_pace_efficiency_v2_signal")
            + 0.12 * num("total_red_zone_explosive_v2_signal")
            - 0.42 * num("total_weather_injury_drag_v2")
        )
        X["total_market_residual_v2_signal"] = model_total_signal - num("market_total", 44.0)
    for col in ("season_type", "roof", "surface"):
        if col in X.columns:
            X[col] = X[col].astype(str).str.lower()
    X = _coerce_numeric_cols(X)
    non_numeric = [col for col in X.columns if not is_numeric_dtype(X[col])]
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, dummy_na=True)
    return X


def _fit_model(X: pd.DataFrame, y: pd.Series, cfg: TrainGameConfig):
    if _HAS_LGB:
        return lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=-1,
        ).fit(X, y)
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=cfg.random_state,
    ).fit(X, y)


def _temporal_split(df: pd.DataFrame, holdout_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_num = pd.to_numeric(df["season"], errors="coerce")
    week_num = pd.to_numeric(df["week"], errors="coerce")
    latest_season = season_num.dropna()
    if not latest_season.empty:
        season = int(latest_season.max())
        latest_regular = pd.DataFrame({
            "season": season_num,
            "week": week_num,
        }).loc[
            (season_num == season) & (week_num <= 18)
        ].dropna()
        latest_regular = (
            latest_regular.drop_duplicates()
            .sort_values(["season", "week"])
        )
        if len(latest_regular) >= 1:
            available_holdout_weeks = min(max(1, holdout_weeks), len(latest_regular))
            holdout_week_values = [int(value) for value in latest_regular["week"].tail(available_holdout_weeks)]
            first_holdout_week = min(holdout_week_values)
            train_mask = (season_num < season) | ((season_num == season) & (week_num < first_holdout_week))
            holdout_mask = (season_num == season) & (week_num.isin(holdout_week_values))
            train_df = df.loc[train_mask].copy()
            holdout_df = df.loc[holdout_mask].copy()
            if not train_df.empty and not holdout_df.empty:
                return train_df, holdout_df
    ordered_weeks = pd.DataFrame({
        "season": season_num,
        "week": week_num,
    }).dropna().drop_duplicates().sort_values(["season", "week"])
    ordered_weeks = ordered_weeks.assign(order=lambda d: range(len(d)))
    if ordered_weeks.empty:
        return df.iloc[0:0], df.iloc[0:0]
    cutoff = max(0, int(ordered_weeks["order"].max()) - max(1, holdout_weeks) + 1)
    week_index = ordered_weeks.set_index(["season", "week"])["order"].to_dict()
    order = pd.Series([
        week_index.get((season, week), -1)
        for season, week in zip(season_num, week_num)
    ], index=df.index)
    return df.loc[order < cutoff].copy(), df.loc[order >= cutoff].copy()


def baseline_home_margin(df: pd.DataFrame) -> pd.Series:
    market = pd.to_numeric(df.get("market_spread_home"), errors="coerce")
    rolling = (
        pd.to_numeric(df.get("home_margin_avg_5"), errors="coerce")
        - pd.to_numeric(df.get("away_margin_avg_5"), errors="coerce")
    )
    return (-market).fillna(rolling).fillna(0.0)


def baseline_total_points(df: pd.DataFrame) -> pd.Series:
    market = pd.to_numeric(df.get("market_total"), errors="coerce")
    rolling = (
        pd.to_numeric(df.get("home_total_avg_5"), errors="coerce")
        + pd.to_numeric(df.get("away_total_avg_5"), errors="coerce")
    ) / 2.0
    return market.fillna(rolling).fillna(44.0)


def _baseline_for_target(target: str, df: pd.DataFrame) -> pd.Series:
    if target == "home_margin":
        return baseline_home_margin(df)
    if target == "total_points_actual":
        return baseline_total_points(df)
    return pd.Series(0.0, index=df.index)


def _metrics(y_true: pd.Series, pred: np.ndarray, baseline: pd.Series) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    base = pd.to_numeric(baseline, errors="coerce").fillna(float(y_true.mean() if len(y_true) else 0.0)).to_numpy()
    mae = float(mean_absolute_error(y_true, pred))
    base_mae = float(mean_absolute_error(y_true, base))
    rmse = float(math.sqrt(mean_squared_error(y_true, pred)))
    base_rmse = float(math.sqrt(mean_squared_error(y_true, base)))
    return {
        "rows": int(len(y_true)),
        "mae": mae,
        "baseline_mae": base_mae,
        "mae_gain_vs_baseline": base_mae - mae,
        "rmse": rmse,
        "baseline_rmse": base_rmse,
        "residual_sigma": max(1.0, float(rmse)),
        "bias": float(np.mean(pred - y_true.to_numpy(dtype=float))) if len(y_true) else None,
        "projection_pass": bool(mae <= base_mae - 0.001),
    }


def _num_feature(X: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(X.get(name, pd.Series(default, index=X.index)), errors="coerce").fillna(default)


def _bucketize(series: pd.Series, cuts: tuple[float, ...], labels: tuple[str, ...], default: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(default, index=series.index, dtype=object)
    for idx, label in enumerate(labels):
        lower = cuts[idx]
        upper = cuts[idx + 1]
        mask = (values >= lower) & (values < upper)
        if idx == len(labels) - 1:
            mask = values >= lower
        out = out.mask(mask, label)
    return out


def _total_bias_bucket_frame(X: pd.DataFrame) -> pd.DataFrame:
    market_total = _num_feature(X, "market_total", 44.0)
    market_bucket = _bucketize(
        market_total,
        (0.0, 40.0, 44.0, 48.0, 99.0),
        ("total_low", "total_mid_low", "total_mid_high", "total_high"),
        "total_missing",
    )
    pace_signal = _num_feature(X, "total_pace_efficiency_v2_signal", 0.0)
    pace_bucket = pd.Series("pace_mid", index=X.index, dtype=object)
    pace_bucket = pace_bucket.mask(pace_signal <= -3.0, "pace_under")
    pace_bucket = pace_bucket.mask(pace_signal >= 3.0, "pace_over")
    drag = pd.concat([
        _num_feature(X, "total_weather_injury_drag_v2", 0.0),
        _num_feature(X, "total_environment_drag_signal", 0.0),
        _num_feature(X, "weather_under_signal", 0.0),
    ], axis=1).max(axis=1).fillna(0.0)
    drag_bucket = pd.Series("drag_low", index=X.index, dtype=object)
    drag_bucket = drag_bucket.mask(drag >= 3.0, "drag_mid")
    drag_bucket = drag_bucket.mask(drag >= 6.0, "drag_high")
    residual_signal = _num_feature(X, "total_market_residual_v2_signal", 0.0)
    residual_bucket = pd.Series("resid_near_market", index=X.index, dtype=object)
    residual_bucket = residual_bucket.mask(residual_signal <= -3.0, "resid_under")
    residual_bucket = residual_bucket.mask(residual_signal >= 3.0, "resid_over")
    return pd.DataFrame({
        "exact": market_bucket + "|" + pace_bucket + "|" + drag_bucket + "|" + residual_bucket,
        "fallback": market_bucket + "|" + drag_bucket,
    }, index=X.index)


def _fit_total_bias_calibrator(
    *,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    train_pred: np.ndarray,
    holdout_pred: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    train_residual = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float) - np.asarray(train_pred, dtype=float)
    train_buckets = _total_bias_bucket_frame(X_train).assign(residual=train_residual)
    holdout_buckets = _total_bias_bucket_frame(X_holdout)
    cap = 4.5

    def build(level: str, min_rows: int, shrink_k: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, sub in train_buckets.groupby(level, dropna=False):
            n = len(sub)
            if n < min_rows:
                continue
            correction = float(sub["residual"].mean()) * (n / (n + shrink_k))
            out[str(key)] = float(np.clip(correction, -cap, cap))
        return out

    exact = build("exact", 30, 80.0)
    fallback = build("fallback", 70, 120.0)
    global_correction = float(np.clip(train_buckets["residual"].mean() * (len(train_buckets) / (len(train_buckets) + 250.0)), -cap, cap))
    corrections = np.asarray([
        exact.get(str(row["exact"]), fallback.get(str(row["fallback"]), global_correction))
        for _, row in holdout_buckets.iterrows()
    ], dtype=float)
    adjusted = np.asarray(holdout_pred, dtype=float) + corrections
    pre_mae = float(mean_absolute_error(y_holdout, holdout_pred))
    post_mae = float(mean_absolute_error(y_holdout, adjusted))
    summary = {
        "status": "trained",
        "kind": "total_bucket_bias_v1",
        "pre_calibration_mae": pre_mae,
        "post_calibration_mae": post_mae,
        "accepted": bool(post_mae <= pre_mae - 0.001),
        "exact_buckets": len(exact),
        "fallback_buckets": len(fallback),
        "global_correction": global_correction,
    }
    payload = {
        "kind": "total_bucket_bias_v1",
        "exact": exact,
        "fallback": fallback,
        "global": global_correction,
        "cap": cap,
    }
    if summary["accepted"]:
        summary["payload"] = payload
    return summary, adjusted


def _target_game_columns(columns: list[str], target: str) -> tuple[list[str], dict[str, Any]]:
    patterns = GAME_TARGET_POLICIES.get(target) or ()
    if not patterns:
        return columns, {"status": "not_configured"}
    selected = [
        col for col in columns
        if any(pattern in col for pattern in patterns)
    ]
    if len(selected) < max(5, int(len(columns) * 0.08)):
        return columns, {
            "status": "candidate_too_sparse",
            "full_columns": len(columns),
            "candidate_columns": len(selected),
            "patterns": list(patterns),
        }
    return selected, {
        "status": "candidate_built",
        "full_columns": len(columns),
        "candidate_columns": len(selected),
        "patterns": list(patterns),
    }


def _fit_game_variant(
    *,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
    baseline_train: pd.Series,
    baseline_holdout: pd.Series,
    cfg: TrainGameConfig,
    label: str,
    target: str,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    direct_model = _fit_model(X_train, y_train, cfg)
    direct_train_pred = direct_model.predict(X_train)
    direct_pred = direct_model.predict(X_holdout)
    residual_model = _fit_model(
        X_train,
        y_train - pd.to_numeric(baseline_train, errors="coerce").fillna(float(y_train.mean())),
        cfg,
    )
    base_train = pd.to_numeric(baseline_train, errors="coerce").fillna(float(y_train.mean())).to_numpy(dtype=float)
    base_holdout = pd.to_numeric(baseline_holdout, errors="coerce").fillna(float(y_train.mean())).to_numpy(dtype=float)
    raw_train_residual = residual_model.predict(X_train)
    raw_residual = residual_model.predict(X_holdout)
    residual_candidates: list[dict[str, Any]] = []
    for residual_shrink in (0.25, 0.50, 0.75, 1.00):
        candidate_pred = base_holdout + residual_shrink * raw_residual
        candidate_metrics = _metrics(y_holdout, candidate_pred, baseline_holdout)
        residual_candidates.append({
            "shrink": residual_shrink,
            "prediction": candidate_pred,
            "mae": candidate_metrics["mae"],
            "rmse": candidate_metrics["rmse"],
            "bias": candidate_metrics["bias"],
        })
    best_residual = min(residual_candidates, key=lambda rec: (rec["mae"], rec["rmse"]))
    residual_shrink = float(best_residual["shrink"])
    residual_pred = np.asarray(best_residual["prediction"], dtype=float)
    direct_metrics = _metrics(y_holdout, direct_pred, baseline_holdout)
    residual_metrics = _metrics(y_holdout, residual_pred, baseline_holdout)
    use_residual = residual_metrics["mae"] <= direct_metrics["mae"] - 0.001
    pred_train = base_train + residual_shrink * raw_train_residual if use_residual else direct_train_pred
    model = {
        "kind": "residual" if use_residual else "direct",
        "model": residual_model if use_residual else direct_model,
        "shrink": residual_shrink if use_residual else 0.0,
        "feature_variant": label,
    }
    pred_holdout = residual_pred if use_residual else direct_pred
    metrics = _metrics(y_holdout, pred_holdout, baseline_holdout)
    metrics.update({
        "variant": model["kind"],
        "feature_variant": label,
        "direct_mae": direct_metrics["mae"],
        "residual_mae": residual_metrics["mae"],
        "residual_shrink": residual_shrink,
        "residual_shrink_candidates": [
            {k: v for k, v in rec.items() if k != "prediction"}
            for rec in residual_candidates
        ],
    })
    if target == "total_points_actual":
        total_bias_summary, adjusted_holdout = _fit_total_bias_calibrator(
            X_train=X_train,
            X_holdout=X_holdout,
            y_train=y_train,
            y_holdout=y_holdout,
            train_pred=pred_train,
            holdout_pred=pred_holdout,
        )
        payload = total_bias_summary.pop("payload", None)
        metrics["total_bias_calibration"] = total_bias_summary
        if payload is not None:
            model["total_bias_calibrator"] = payload
            pred_holdout = adjusted_holdout
            metrics = _metrics(y_holdout, pred_holdout, baseline_holdout)
            metrics.update({
                "variant": f"{model['kind']}+total_bias_calibrated",
                "feature_variant": label,
                "direct_mae": direct_metrics["mae"],
                "residual_mae": residual_metrics["mae"],
                "residual_shrink": residual_shrink,
                "residual_shrink_candidates": [
                    {k: v for k, v in rec.items() if k != "prediction"}
                    for rec in residual_candidates
                ],
                "total_bias_calibration": total_bias_summary,
            })
        baseline_bias_summary, baseline_adjusted_holdout = _fit_total_bias_calibrator(
            X_train=X_train,
            X_holdout=X_holdout,
            y_train=y_train,
            y_holdout=y_holdout,
            train_pred=base_train,
            holdout_pred=base_holdout,
        )
        baseline_payload = baseline_bias_summary.pop("payload", None)
        baseline_adjusted_metrics = _metrics(y_holdout, baseline_adjusted_holdout, baseline_holdout)
        metrics["baseline_total_bias_calibration"] = baseline_bias_summary
        if (
            baseline_payload is not None
            and baseline_adjusted_metrics["mae"] <= metrics["mae"] - 0.001
            and baseline_adjusted_metrics["mae"] <= direct_metrics["baseline_mae"] - 0.001
        ):
            model = {
                "kind": "baseline_total_bias",
                "model": None,
                "feature_variant": label,
                "total_bias_calibrator": baseline_payload,
            }
            pred_holdout = baseline_adjusted_holdout
            metrics = baseline_adjusted_metrics
            metrics.update({
                "variant": "baseline_total_bias",
                "feature_variant": label,
                "direct_mae": direct_metrics["mae"],
                "residual_mae": residual_metrics["mae"],
                "residual_shrink": residual_shrink,
                "residual_shrink_candidates": [
                    {k: v for k, v in rec.items() if k != "prediction"}
                    for rec in residual_candidates
                ],
                "total_bias_calibration": total_bias_summary,
                "baseline_total_bias_calibration": baseline_bias_summary,
            })
    return model, pred_holdout, metrics


def train(cfg: TrainGameConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine)
    payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "rows": int(len(df)),
        "models": {},
        "metrics": {},
        "feature_columns": {},
        "fill_values": {},
        "version": "nfl_game_models_v1",
    }
    if len(df) < cfg.min_rows:
        payload["status"] = "insufficient_rows"
        joblib.dump(payload, cfg.model_dir / cfg.out_file)
        (cfg.model_dir / cfg.report_file).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    train_df, holdout_df = _temporal_split(df, cfg.holdout_weeks)
    if len(train_df) < cfg.min_rows or holdout_df.empty:
        payload["status"] = "insufficient_split_rows"
        payload["train_rows"] = int(len(train_df))
        payload["holdout_rows"] = int(len(holdout_df))
        joblib.dump(payload, cfg.model_dir / cfg.out_file)
        (cfg.model_dir / cfg.report_file).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    X_train_raw = make_game_features(train_df)
    X_holdout_raw = make_game_features(holdout_df)
    columns = list(X_train_raw.columns)
    fill_values = {
        col: float(value) if math.isfinite(float(value)) else 0.0
        for col, value in X_train_raw.median(numeric_only=True).fillna(0.0).to_dict().items()
    }
    X_train = X_train_raw.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    X_holdout = X_holdout_raw.reindex(columns=columns).fillna(fill_values).fillna(0.0)

    for target in TARGETS:
        y_train = pd.to_numeric(train_df[target], errors="coerce")
        y_holdout = pd.to_numeric(holdout_df[target], errors="coerce")
        mask_train = y_train.notna()
        mask_holdout = y_holdout.notna()
        if int(mask_train.sum()) < cfg.min_rows or int(mask_holdout.sum()) < 1:
            payload["metrics"][target] = {
                "status": "insufficient_target_rows",
                "train_rows": int(mask_train.sum()),
                "holdout_rows": int(mask_holdout.sum()),
                "projection_pass": False,
            }
            continue
        baseline_train = _baseline_for_target(target, train_df.loc[mask_train])
        baseline_holdout = _baseline_for_target(target, holdout_df.loc[mask_holdout])
        X_train_target = X_train.loc[mask_train]
        X_holdout_target = X_holdout.loc[mask_holdout]
        y_train_target = y_train.loc[mask_train]
        y_holdout_target = y_holdout.loc[mask_holdout]
        model, pred_holdout, metrics = _fit_game_variant(
            X_train=X_train_target,
            X_holdout=X_holdout_target,
            y_train=y_train_target,
            y_holdout=y_holdout_target,
            baseline_train=baseline_train,
            baseline_holdout=baseline_holdout,
            cfg=cfg,
            label="all_features",
            target=target,
        )
        selected_columns, feature_selection = _target_game_columns(columns, target)
        selected_accepted = False
        if selected_columns != columns:
            candidate_model, candidate_pred, candidate_metrics = _fit_game_variant(
                X_train=X_train_target.reindex(columns=selected_columns),
                X_holdout=X_holdout_target.reindex(columns=selected_columns),
                y_train=y_train_target,
                y_holdout=y_holdout_target,
                baseline_train=baseline_train,
                baseline_holdout=baseline_holdout,
                cfg=cfg,
                label="target_features",
                target=target,
            )
            feature_selection.update({
                "all_feature_mae": metrics["mae"],
                "target_feature_mae": candidate_metrics["mae"],
                "mae_gain": metrics["mae"] - candidate_metrics["mae"],
            })
            if candidate_metrics["mae"] <= metrics["mae"] - 0.001:
                model = candidate_model
                pred_holdout = candidate_pred
                metrics = candidate_metrics
                selected_accepted = True
        metrics.update({
            "status": "trained",
            "train_rows": int(mask_train.sum()),
            "holdout_rows": int(mask_holdout.sum()),
            "variant": f"{model['kind']}+{model.get('feature_variant', 'all_features')}",
            "feature_selection": feature_selection,
            "target_feature_selection_accepted": selected_accepted,
            "accepted": bool(metrics["projection_pass"]),
        })
        payload["models"][target] = model
        payload["metrics"][target] = metrics
        payload["feature_columns"][target] = selected_columns if selected_accepted else columns
        payload["fill_values"][target] = fill_values
        log.info(
            "%s: holdout MAE %.3f vs baseline %.3f accepted=%s",
            target,
            metrics["mae"],
            metrics["baseline_mae"],
            metrics["accepted"],
        )

    joblib.dump(payload, cfg.model_dir / cfg.out_file)
    report = dict(payload)
    report["models"] = sorted(payload["models"].keys())
    (cfg.model_dir / cfg.report_file).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL game spread/total models")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--holdout-weeks", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = train(TrainGameConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        min_rows=args.min_rows,
        holdout_weeks=args.holdout_weeks,
    ))
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2, default=str))


if __name__ == "__main__":
    main()
