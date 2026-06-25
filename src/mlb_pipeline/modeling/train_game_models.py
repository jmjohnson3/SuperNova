# src/mlb_pipeline/modeling/train_game_models.py
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .features import add_game_derived_features

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

log = logging.getLogger("mlb_pipeline.modeling.train_game_models")


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = PG_DSN

    # Minimum number of market rows required to train residual models
    min_market_rows: int = 15
    run_walk_forward: bool = True
    # MLB has a 162-game season; need more warmup data than NBA (82 games).
    min_train_days: int = 120
    test_window_days: int = 14
    step_days: int = 21          # was 14; 33% fewer walk-forward folds, still ample OOF coverage

    # XGBoost params (defaults — overridden by Optuna when enabled)
    n_estimators: int = 2000
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    # Regularization
    min_child_weight: int = 10
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 3.0
    # Early stopping
    early_stopping_rounds: int = 30  # was 50; bad Optuna trials abort sooner, no quality impact
    # Huber loss slope
    huber_slope: float = 3.0

    # Optuna tuning
    run_optuna: bool = True
    optuna_n_trials: int = 30    # was 60; TPE converges by trial 20-25, halves Optuna cost
    optuna_n_folds: int = 3      # was 5; 600 fits → 180 fits (70% reduction)
    # Recency weighting half-life (days): sample weight halves every N days of age.
    # Age is measured from the most recent training date in each fold.
    recency_half_life_days: int = 45


SQL_GAME_TRAINING_FEATURES = """
SELECT gtf.*,
       elo.home_elo, elo.away_elo, elo.elo_diff, elo.elo_win_prob_home
FROM features.mlb_game_training_features gtf
LEFT JOIN features.mlb_game_elo_features elo
  ON elo.season = gtf.season AND elo.game_slug = gtf.game_slug
WHERE gtf.run_diff IS NOT NULL
ORDER BY gtf.game_date_et, gtf.game_slug
"""


def load_training_frame(conn) -> pd.DataFrame:
    df = pd.read_sql(SQL_GAME_TRAINING_FEATURES, conn)
    if df.empty:
        raise RuntimeError("No rows returned from features.mlb_game_training_features (df is empty).")

    df["game_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce")
    if df["game_date_et"].isna().any():
        bad = df[df["game_date_et"].isna()].head(5)
        raise RuntimeError(f"Some game_date_et values could not be parsed. Examples:\n{bad}")

    elo_null_pct = df["elo_diff"].isna().mean()
    if elo_null_pct > 0.5:
        import logging
        logging.getLogger(__name__).warning(
            "%.0f%% of training rows have NULL elo features — "
            "run compute_elo before training for best results.",
            elo_null_pct * 100,
        )

    return df.sort_values(["game_date_et", "game_slug"]).reset_index(drop=True)


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def build_model(
    cfg: TrainConfig,
    params_override: Optional[Dict] = None,
    objective: str = "reg:pseudohubererror",
    n_estimators: Optional[int] = None,
    use_early_stopping: bool = True,
) -> XGBRegressor:
    """Build XGBRegressor with config defaults, optionally overridden by Optuna params.

    Both run line and total models use Huber loss (reg:pseudohubererror).
    Optuna tunes huber_slope so gradients stay informative for totals (~9 runs).

    Set use_early_stopping=False for final production models: train to a fixed
    n_estimators derived from CV best_iteration statistics rather than using a
    held-out eval split, which triggers premature stopping due to distribution shift
    between earlier training data and the most-recent games used as the eval set.
    """
    p = {
        "n_estimators": n_estimators if n_estimators is not None else cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "objective": objective,
        "min_child_weight": cfg.min_child_weight,
        "gamma": cfg.gamma,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "eval_metric": "mae",
        "random_state": cfg.random_state,
        "n_jobs": -1,
    }
    if use_early_stopping:
        p["early_stopping_rounds"] = cfg.early_stopping_rounds
    # huber_slope only applies to Huber-family objectives
    if objective == "reg:pseudohubererror":
        p["huber_slope"] = cfg.huber_slope
    if params_override:
        override = dict(params_override)
        if objective != "reg:pseudohubererror":
            override.pop("huber_slope", None)
        p.update(override)
    return XGBRegressor(**p)


# ── Metrics ──────────────────────────────────────────────────────────────────
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def _directional_accuracy_margin(y_true_margin: np.ndarray, y_pred_margin: np.ndarray) -> float:
    # Correct sign of run differential (home - away)
    return float(np.mean(np.sign(y_true_margin) == np.sign(y_pred_margin)))


def _calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if len(y_true) < 3:
        return (np.nan, np.nan)
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    var = float(np.var(x))
    if var <= 1e-12:
        return (np.nan, np.nan)
    b = float(np.cov(x, y, ddof=0)[0, 1] / var)
    a = float(np.mean(y) - b * np.mean(x))
    return b, a


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def directional_accuracy_margin(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Did we predict the winner (sign of run differential)?"""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def calibration_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    std_true = float(np.std(y_true)) if len(y_true) else 0.0
    std_pred = float(np.std(y_pred)) if len(y_pred) else 0.0
    std_ratio = float(std_pred / std_true) if std_true > 0 else float("nan")

    if len(y_true) >= 2 and np.std(y_pred) > 1e-9:
        b, a = np.polyfit(y_pred, y_true, deg=1)
        slope = float(b)
        intercept = float(a)
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "std_true": std_true,
        "std_pred": std_pred,
        "std_ratio": std_ratio,
        "slope": slope,
        "intercept": intercept,
    }


def make_xy_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build X with NaNs preserved (no median fill here).
    We'll fit medians on *train only* inside each walk-forward fold (prevents leakage).

    Targets:
      y_run_diff  — home_score - away_score, clipped to [-15, 15]
      y_total     — home_score + away_score, clipped to [1, 30]
      y_f5        — total_f5 (first-5-innings total), clipped to [0, 25]; NaN where unavailable
    """
    y_run_diff = df["run_diff"].astype(float).clip(-15.0, 15.0)
    y_total = df["total_runs"].astype(float).clip(1.0, 30.0)
    y_f5 = (
        df["total_f5"].astype(float).clip(0.0, 25.0)
        if "total_f5" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    drop_cols = {
        "game_slug",
        "game_date_et",
        "run_diff",
        "total_runs",
        # leakage — postgame actuals
        "home_score",
        "away_score",
        "home_win",
        # residuals are postgame (actual - market) — must not be direct-model features
        "run_line_residual",
        "total_residual",
        # F5 targets — postgame actuals
        "home_f5_runs",
        "away_f5_runs",
        "total_f5",
        "f5_run_diff",
        # raw name columns
        "home_sp_raw_name",
        "away_sp_raw_name",
        "home_sp_id",
        "away_sp_id",
        "status",
        # text identifiers — OHE causes team-specific noise (overfitting with ~15 games/team early season)
        # team quality is captured by rolling stat features; venue is captured by venue_id + park factor cols
        "home_team_abbr",
        "away_team_abbr",
        "venue_name",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Drop any additional postgame/leaky columns by prefix
    leaky_prefixes = ("box_", "final_", "actual_", "result_", "postgame_")
    leaky_exact = {
        "home_score", "away_score", "total_runs", "run_diff", "home_win",
        "run_line_residual", "total_residual",
    }
    leaky_cols = [c for c in X.columns if c in leaky_exact or c.startswith(leaky_prefixes)]
    if leaky_cols:
        log.warning("Dropping %d potentially leaky cols: %s", len(leaky_cols), leaky_cols[:20])
        X = X.drop(columns=leaky_cols)

    # Season position: days elapsed since April 1 of the season year.
    # MLB season starts ~late March / early April; lets model discount noisy early-season stats.
    if "game_date_et" in df.columns:
        gdt = pd.to_datetime(df["game_date_et"])
        # Season year: games in Jan–Mar belong to prior calendar year's spring training edge case
        # but realistically MLB games are Apr–Oct so use year directly.
        season_start = pd.to_datetime(gdt.dt.year.astype(str) + "-04-01")
        X["season_days_elapsed"] = (gdt - season_start).dt.days.values

    # Timing features
    if "start_ts_utc" in X.columns:
        ts = pd.to_datetime(X["start_ts_utc"], errors="coerce", utc=True)
        X["start_hour_utc"] = ts.dt.hour
        X["start_dow_utc"] = ts.dt.dayofweek
        # Day game if first pitch before 5 PM ET (day/night splits are real in MLB)
        ts_et = ts.dt.tz_convert("America/New_York")
        X["is_day_game"] = (ts_et.dt.hour < 17).astype(int)
        X = X.drop(columns=["start_ts_utc"])

    # Ensure b2b flags are 0/1
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # One-hot encode season only (team abbrs dropped — cause overfitting in small datasets)
    cat_cols = ["season"]
    for c in cat_cols:
        if c not in X.columns and c in df.columns:
            X[c] = df[c]
    cat_cols_present = [c for c in cat_cols if c in X.columns]
    if cat_cols_present:
        X = pd.get_dummies(X, columns=cat_cols_present, drop_first=False, dummy_na=False)

    # Coerce remaining strings/objects to numeric where possible
    X = _coerce_numeric_cols(X)

    # Last-resort one-hot for any remaining non-numeric columns
    non_numeric_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if non_numeric_cols:
        log.warning("One-hot encoding remaining non-numeric cols: %s", non_numeric_cols)
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False, dummy_na=True)

    # Derived interaction features — single source of truth in features.add_game_derived_features.
    X = add_game_derived_features(X)

    # Drop confirmed zero-importance features (per feature_importance.json after last retrain).
    # Note: ump_runs_per_game is intentionally kept because ump_rpg_x_park references it.
    _prune_zero_importance = [
        "elo_win_prob_home",           # redundant with elo_diff (logistic transform)
        "h2h_edge", "h2h_familiarity", # season H2H record too sparse; ELO captures it
        "sp_short_rest_asymmetry",     # captured by individual sp_days_rest features
        "home_sp_is_short_rest", "away_sp_is_short_rest",
        "home_sp_last_short",   "away_sp_last_short",
        "home_is_b2b", "away_is_b2b", "rest_diff",
        "is_cold_outdoor",             # captured by temp_cold_penalty (continuous)
        "precip_scoring_risk",         # overlaps with precip_prob_pct
        "win_pct_diff",                # captured by win_pct_last_5_diff / _10_diff
        "home_division_rank", "away_division_rank", "division_rank_edge",
        "over_implied_prob",           # redundant; total_vig_direction is directional
        "team_era_10_diff", "team_whip_10_diff",
        "batting_runs_avg_10_diff",
        "sp_fip_5_diff",               # captured by sp_composite_edge
        "home_games_vs_lhp", "home_games_vs_rhp",
        "away_games_vs_lhp", "away_games_vs_rhp",
        # Raw batting-vs-hand split columns (derived platoon features replace them)
        "home_sp_pitch_hand_L", "away_sp_pitch_hand_L",
        "home_team_avg_vs_lhp", "home_team_avg_vs_rhp",
        "home_team_obp_vs_lhp", "home_team_obp_vs_rhp",
        "home_team_slg_vs_lhp", "home_team_slg_vs_rhp",
        "away_team_avg_vs_lhp", "away_team_avg_vs_rhp",
        "away_team_obp_vs_lhp", "away_team_obp_vs_rhp",
        "away_team_slg_vs_lhp", "away_team_slg_vs_rhp",
        # Other confirmed 0.0-importance features
        "rest_days_advantage",
        "market_total",                # aliased by total_line (already in matrix)
        "market_run_line",             # always -1.5, zero variance
        "sp_short_asymmetry",
        "home_is_opener", "away_is_opener", "opener_asymmetry",
        "home_sp_era_shrunk", "away_sp_era_shrunk", "sp_era_shrunk_diff",
        "is_dome",
        "home_sp_short_last", "away_sp_short_last",
        # Near-zero importance (<0.001): replace raw columns with derived interactions
        "run_line_home",           # always -1.5; zero variance (market_run_line already pruned)
        "venue_id",                # park effects captured by park_run_factor / park_hr_factor
        "park_hr_factor",          # raw value; interaction versions (team_hr_x_park) carry signal
        "h2h_home_win_pct_ytd",    # season H2H too sparse; Elo captures team strength
        "away_wins_last_5",        # captured by win_pct_last_5_diff / runs_trend_diff
        "market_home_win_prob",    # redundant with total_vig_direction + Elo features
        "n_ump_games_prev_5",      # ump experience; ump_rpg_avg_10 is the informative signal
        "home_runs_avg_5",         # captured by runs_trend_diff (5v20) and runs_avg_10
        "run_line_home_price",     # market_home_win_prob was derived from this; both ~0
        "over_price",              # total_vig_direction is the informative derived version
        "total_line_move",         # raw move; abs_total_line_move / direction replace it
        "run_line_move",           # raw move; abs_run_line_move / direction replace it
        # Confirmed 0.0 importance — platoon source data (team_avg_vs_lhp etc.) is all NULL
        # in the database, so derived platoon features compute to 0 and carry no signal.
        "platoon_matchup_diff", "home_hand_matchup_edge", "away_hand_matchup_edge",
        "home_obp_vs_sp_hand", "away_obp_vs_sp_hand",
        # Confirmed 0.0 importance — binary flags redundant with continuous versions
        "home_sp_last_high_workload", "away_sp_last_high_workload",
        "home_sp_k_trend_5v10", "away_sp_k_trend_5v10",
        "home_sp_last_quality", "away_sp_last_quality",
        "home_win_pct_last_5",         # captured by win_pct_last_5_diff
        "run_line_move_large", "total_line_move_large",   # binary; continuous versions used
        "season_2026-regular",         # only 2026 games; effectively constant
        "ump_runs_per_game",           # ump_rpg_x_park interaction is the informative version
        "market_total_vs_avg",         # removed from features.py; constant shift = no new info
        # PROXY LEAKAGE: always 1.0 in training (completed games), median-imputed at inference.
        # Top-2 features by importance but detect train-vs-predict time, not real signal.
        "home_lineup_completeness", "away_lineup_completeness",
    ]
    X = X.drop(columns=[c for c in _prune_zero_importance if c in X.columns])

    still_bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if still_bad:
        raise RuntimeError(f"Non-numeric columns remain after encoding: {still_bad[:20]}")

    return X, y_run_diff, y_total, y_f5


def fit_fill_stats(X_train: pd.DataFrame) -> Dict[str, float]:
    """Return per-column medians (train only) for NaN filling."""
    meds = X_train.median(numeric_only=True)
    return {str(k): float(v) for k, v in meds.items()}


def apply_fill(X: pd.DataFrame, medians: Dict[str, float], columns: List[str]) -> pd.DataFrame:
    """Align to `columns`, fill NaNs with train medians, then final fill of 0.0."""
    X2 = X.reindex(columns=columns)
    for c, m in medians.items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(m)
    return X2.fillna(0.0)


def temporal_eval_split(
    dates: pd.Series, eval_frac: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (fit_mask, eval_mask) splitting by date quantile."""
    cutoff = dates.quantile(1.0 - eval_frac)
    fit_mask = (dates < cutoff).values
    eval_mask = (dates >= cutoff).values
    return fit_mask, eval_mask


def walk_forward_folds(
    df: pd.DataFrame,
    *,
    min_train_days: int,
    test_window_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of (train_end_date, test_end_date) boundaries.
    Train uses dates < train_end_date.
    Test uses [train_end_date, test_end_date).
    """
    dates = df["game_date_et"]
    start_date = dates.min().normalize()
    end_date = dates.max().normalize() + pd.Timedelta(days=1)

    first_train_end = start_date + pd.Timedelta(days=min_train_days)
    if first_train_end >= end_date:
        return []

    folds = []
    train_end = first_train_end
    while True:
        test_end = train_end + pd.Timedelta(days=test_window_days)
        if test_end > end_date:
            break
        folds.append((train_end, test_end))
        train_end = train_end + pd.Timedelta(days=step_days)
    return folds


def compute_recency_weights(
    dates: pd.Series,
    *,
    half_life_days: int,
) -> np.ndarray:
    """Return normalized recency weights using true half-life decay.

    Weight formula:
      w = exp(-ln(2) * age_days / half_life_days)

    where age_days is measured from the most recent date in `dates`.
    """
    if len(dates) == 0:
        return np.array([], dtype=float)

    safe_half_life = max(int(half_life_days), 1)
    d = pd.to_datetime(dates)
    max_dt = d.max()
    age_days = (max_dt - d).dt.days.astype(float).to_numpy()
    w = np.exp(-np.log(2.0) * age_days / float(safe_half_life))

    # Keep the mean at 1.0 so regularization behavior stays comparable.
    mean_w = float(np.mean(w))
    if mean_w > 0:
        w = w / mean_w
    return w


# ── Optuna hyperparameter tuning ─────────────────────────────────────────────
def _optuna_objective(
    trial,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_target: pd.Series,
    feature_cols: List[str],
    folds: List[Tuple[pd.Timestamp, pd.Timestamp]],
    target_name: str,
    recency_half_life_days: int,
    objective: str = "reg:pseudohubererror",
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    }
    if objective == "reg:pseudohubererror":
        params["huber_slope"] = trial.suggest_float("huber_slope", 1.0, 6.0)

    mae_scores = []

    for train_end, test_end in folds:
        train_mask = df["game_date_et"] < train_end
        test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < 50 or n_test == 0:
            continue

        X_train_raw = X_raw.loc[train_mask]
        X_test_raw = X_raw.loc[test_mask]

        medians = fit_fill_stats(X_train_raw)
        X_train = apply_fill(X_train_raw, medians, feature_cols)
        X_test = apply_fill(X_test_raw, medians, feature_cols)

        y_train = y_target.loc[train_mask]
        y_test = y_target.loc[test_mask]

        train_dates = df.loc[train_mask, "game_date_et"]
        fit_rel, eval_rel = temporal_eval_split(train_dates)
        recency_weights = compute_recency_weights(
            train_dates,
            half_life_days=recency_half_life_days,
        )
        recency_weights_fit = recency_weights[fit_rel]
        recency_weights_eval = recency_weights[eval_rel]

        model = XGBRegressor(
            n_estimators=2000,
            objective=objective,
            early_stopping_rounds=50,
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(
            X_train.iloc[fit_rel], y_train.iloc[fit_rel],
            sample_weight=recency_weights_fit,
            eval_set=[(X_train.iloc[eval_rel], y_train.iloc[eval_rel])],
            sample_weight_eval_set=[recency_weights_eval],
            verbose=False,
        )

        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        mae_scores.append(mae)

    if not mae_scores:
        return float("inf")
    return float(np.mean(mae_scores))


RUN_LINE_OBJECTIVE      = "reg:pseudohubererror"
# Direct total target is ~9 runs (far from 0); squarederror avoids near-zero Huber
# hessians that prevent splits when base_score=0.5. Residual target is ≈0 ±4 runs —
# Huber is ideal there and Optuna tunes huber_slope in the residual-only path.
TOTAL_DIRECT_OBJECTIVE  = "reg:squarederror"
TOTAL_RESID_OBJECTIVE   = "reg:pseudohubererror"


def run_optuna_tuning(
    cfg: TrainConfig,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_run_diff: pd.Series,
    y_total: pd.Series,
    feature_cols: List[str],
) -> Tuple[Dict, Dict]:
    """
    Run Optuna hyperparameter search for both run-line and total models.
    Returns (best_run_line_params, best_total_params).

    Run line uses Huber loss (robust to blowout outliers; targets near 0).
    Direct total uses squarederror (targets ~9 runs far from 0; avoids near-zero
    Huber hessians). Residual total uses Huber (targets ≈0 ±4 runs; ideal for Huber).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_folds = walk_forward_folds(
        df,
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
    )
    tune_folds = all_folds[-cfg.optuna_n_folds:] if len(all_folds) > cfg.optuna_n_folds else all_folds

    if not tune_folds:
        log.warning("No folds available for Optuna tuning. Using default params.")
        return {}, {}

    log.info("Running Optuna tuning for RUN LINE model (%d trials, %d folds)...",
             cfg.optuna_n_trials, len(tune_folds))
    run_line_study = optuna.create_study(direction="minimize", study_name="run_line")
    run_line_study.optimize(
        lambda trial: _optuna_objective(
            trial, df, X_raw, y_run_diff, feature_cols, tune_folds, "run_line",
            recency_half_life_days=cfg.recency_half_life_days,
            objective=RUN_LINE_OBJECTIVE,
        ),
        n_trials=cfg.optuna_n_trials,
        show_progress_bar=False,
    )
    best_run_line = run_line_study.best_params
    log.info("RUN LINE best params (MAE=%.3f): %s", run_line_study.best_value, best_run_line)

    log.info("Running Optuna tuning for TOTAL model (%d trials, %d folds)...",
             cfg.optuna_n_trials, len(tune_folds))
    total_study = optuna.create_study(direction="minimize", study_name="total")
    total_study.optimize(
        lambda trial: _optuna_objective(
            trial, df, X_raw, y_total, feature_cols, tune_folds, "total",
            recency_half_life_days=cfg.recency_half_life_days,
            objective=TOTAL_DIRECT_OBJECTIVE,
        ),
        n_trials=cfg.optuna_n_trials,
        show_progress_bar=False,
    )
    best_total = total_study.best_params
    log.info("TOTAL best params (MAE=%.3f): %s", total_study.best_value, best_total)

    return best_run_line, best_total


def _select_top_features(
    X_all: pd.DataFrame,
    y_run_diff: pd.Series,
    y_total: pd.Series,
    feature_cols: list,
    model_dir: Path,
    cum_thresh: float = 0.95,
    min_keep: int = 60,
    max_keep: int = 120,
) -> tuple:
    """Fit two cheap selector XGBs; rank by mean gain; return (pruned_feature_cols, X_pruned).
    Always retains one-hot dummies (season_ prefix) and market line anchors (run_line_home,
    total_line, open_run_line_home, open_total_line, total_line_move, run_line_move).
    Market anchors are critical for computing correct edges at inference; their low XGB
    importance underestimates their value because they're often imputed with the median."""
    sel_s = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    sel_t = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    sel_s.fit(X_all, y_run_diff)
    sel_t.fit(X_all, y_total)

    def _norm(arr):
        tot = arr.sum() or 1.0
        return arr / tot

    imp_s = _norm(sel_s.feature_importances_)
    imp_t = _norm(sel_t.feature_importances_)
    avg   = {f: float((imp_s[i] + imp_t[i]) / 2.0) for i, f in enumerate(feature_cols)}

    sorted_imp = sorted(avg.items(), key=lambda x: -x[1])
    (model_dir / "feature_importance.json").write_text(
        json.dumps({k: round(v, 6) for k, v in sorted_imp}, indent=2), encoding="utf-8"
    )

    # Always keep season dummies and market line anchors; the latter have deceptively
    # low XGB importance because ~50% of rows have imputed median values, but they are
    # essential for edge = pred - market_line to be meaningful.
    _market_anchors = {
        "run_line_home", "total_line",
        "open_run_line_home", "open_total_line",
        "total_line_move", "run_line_move",
    }
    dummies = {c for c in feature_cols if c.startswith("season_")} | {
        c for c in _market_anchors if c in feature_cols
    }

    non_struct = [(c, avg.get(c, 0.0)) for c in feature_cols if c not in dummies]
    non_struct.sort(key=lambda x: -x[1])
    total_gain = sum(v for _, v in non_struct) or 1.0

    selected, cum = [], 0.0
    for feat, g in non_struct:
        selected.append(feat)
        cum += g / total_gain
        if cum >= cum_thresh and len(selected) >= min_keep:
            break
    if len(selected) < min_keep:
        selected = [f for f, _ in non_struct[:min_keep]]
    elif len(selected) > max_keep:
        selected = [f for f, _ in non_struct[:max_keep]]

    keep = dummies | set(selected)
    pruned = [c for c in feature_cols if c in keep]

    log.info(
        "Feature pruning: %d → %d features (%d one-hot + %d signal, %.1f%% cum gain)",
        len(feature_cols), len(pruned), len(dummies), len(selected), cum_thresh * 100,
    )
    return pruned, X_all[pruned]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = TrainConfig()
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = load_training_frame(conn)
        log.info("Loaded %d rows from mlb_game_training_features", len(df))

        # Build RAW X (NaNs preserved) once
        X_raw, y_run_diff, y_total, y_f5 = make_xy_raw(df)
        feature_cols = list(X_raw.columns)
        f5_mask = y_f5.notna()
        log.info("F5 target available for %d / %d rows", int(f5_mask.sum()), len(df))

        # Optuna hyperparameter tuning
        best_run_line_params: Dict = {}
        best_total_params: Dict = {}
        if cfg.run_optuna:
            try:
                best_run_line_params, best_total_params = run_optuna_tuning(
                    cfg, df, X_raw, y_run_diff, y_total, feature_cols
                )
            except ImportError:
                log.warning("optuna not installed. Skipping tuning. pip install optuna")
            except Exception as e:
                log.warning("Optuna tuning failed: %s. Using default params.", e)

        # Walk-forward folds
        folds = walk_forward_folds(
            df,
            min_train_days=cfg.min_train_days,
            test_window_days=cfg.test_window_days,
            step_days=cfg.step_days,
        )
        if not folds:
            raise RuntimeError(
                "No folds produced. Try reducing min_train_days or ensure data spans enough dates.")

        # Direct model aggregates
        run_line_true_all: List[float] = []
        run_line_pred_all: List[float] = []
        total_true_all: List[float] = []
        total_pred_all: List[float] = []

        # Residual model aggregates
        resid_rl_true_all: List[float] = []
        resid_rl_pred_all: List[float] = []
        resid_total_true_all: List[float] = []
        resid_total_pred_all: List[float] = []

        # Fair-comparison baselines for quality gate (same population as resid model)
        direct_market_rl_pred_all:  List[float] = []
        direct_market_tot_pred_all: List[float] = []
        market_rl_baseline_all:     List[float] = []
        market_tot_baseline_all:    List[float] = []

        # ATS (run line cover) tracking
        ats_correct_all: List[bool] = []
        ats_hc_correct_all: List[bool] = []  # high-confidence |edge| >= 1.5 runs

        # Track fold best_iterations so final model trains to the right depth
        rl_best_iters: List[int] = []
        tot_best_iters: List[int] = []
        resid_rl_best_iters: List[int] = []
        resid_tot_best_iters: List[int] = []
        f5_best_iters: List[int] = []

        # F5 (first 5 innings) aggregate
        f5_true_all: List[float] = []
        f5_pred_all: List[float] = []

        fold_rows = 0

        log.info(
            "Starting walk-forward eval | min_train_days=%d test_window_days=%d step_days=%d",
            cfg.min_train_days,
            cfg.test_window_days,
            cfg.step_days,
        )

        for k, (train_end, test_end) in enumerate(folds, start=1):
            train_mask = df["game_date_et"] < train_end
            test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())
            if n_train == 0 or n_test == 0:
                continue

            X_train_raw = X_raw.loc[train_mask]
            X_test_raw = X_raw.loc[test_mask]

            # Fit medians on TRAIN ONLY to avoid leakage
            medians = fit_fill_stats(X_train_raw)
            X_train = apply_fill(X_train_raw, medians, feature_cols)
            X_test = apply_fill(X_test_raw, medians, feature_cols)

            y_rl_train = y_run_diff.loc[train_mask]
            y_rl_test  = y_run_diff.loc[test_mask]
            y_tot_train = y_total.loc[train_mask]
            y_tot_test  = y_total.loc[test_mask]

            # Temporal eval split for early stopping (last 20% of train by date)
            train_dates = df.loc[train_mask, "game_date_et"]
            fit_rel, eval_rel = temporal_eval_split(train_dates)
            X_fit  = X_train.iloc[fit_rel]
            X_eval = X_train.iloc[eval_rel]

            # ── Recency weighting for direct models ────────────────────────────────
            # True half-life decay from most-recent training date.
            _train_dates = df.loc[train_mask, "game_date_et"]
            _recency_weights = compute_recency_weights(
                _train_dates,
                half_life_days=cfg.recency_half_life_days,
            )
            _recency_weights_fit = _recency_weights[fit_rel]

            # DIRECT MODELS
            rl_model  = build_model(cfg, params_override=best_run_line_params, objective=RUN_LINE_OBJECTIVE)
            tot_model = build_model(cfg, params_override=best_total_params,    objective=TOTAL_DIRECT_OBJECTIVE)

            rl_model.fit(
                X_fit, y_rl_train.iloc[fit_rel],
                sample_weight=_recency_weights_fit,
                eval_set=[(X_eval, y_rl_train.iloc[eval_rel])],
                verbose=False,
            )
            tot_model.fit(
                X_fit, y_tot_train.iloc[fit_rel],
                sample_weight=_recency_weights_fit,
                eval_set=[(X_eval, y_tot_train.iloc[eval_rel])],
                verbose=False,
            )

            rl_pred  = rl_model.predict(X_test)
            tot_pred = tot_model.predict(X_test)

            if getattr(rl_model, "best_iteration", None) is not None:
                rl_best_iters.append(rl_model.best_iteration)
            if getattr(tot_model, "best_iteration", None) is not None:
                tot_best_iters.append(tot_model.best_iteration)

            # F5 MODEL (first 5 innings total; only on rows with F5 data)
            f5_train_mask = train_mask & f5_mask
            f5_test_mask  = test_mask  & f5_mask
            if f5_train_mask.sum() >= 50 and f5_test_mask.sum() >= 3:
                y_f5_train = y_f5.loc[f5_train_mask]
                y_f5_test  = y_f5.loc[f5_test_mask]
                X_f5_train = apply_fill(X_raw.loc[f5_train_mask], medians, feature_cols)
                X_f5_test  = apply_fill(X_raw.loc[f5_test_mask],  medians, feature_cols)
                f5_fit_rel, f5_eval_rel = temporal_eval_split(df.loc[f5_train_mask, "game_date_et"])
                f5_model = build_model(cfg, params_override=best_total_params,
                                       objective=TOTAL_DIRECT_OBJECTIVE)
                if f5_fit_rel.sum() > 10 and f5_eval_rel.sum() > 5:
                    f5_model.fit(
                        X_f5_train.iloc[f5_fit_rel], y_f5_train.iloc[f5_fit_rel],
                        eval_set=[(X_f5_train.iloc[f5_eval_rel], y_f5_train.iloc[f5_eval_rel])],
                        verbose=False,
                    )
                    f5_pred = f5_model.predict(X_f5_test)
                    if getattr(f5_model, "best_iteration", None) is not None:
                        f5_best_iters.append(f5_model.best_iteration)
                    f5_true_all.extend(y_f5_test.to_list())
                    f5_pred_all.extend(f5_pred.tolist())

            # Direct metrics
            rl_mae,  rl_rmse  = evaluate_regression(y_rl_test.to_numpy(),  rl_pred)
            tot_mae, tot_rmse = evaluate_regression(y_tot_test.to_numpy(), tot_pred)
            rl_dir = directional_accuracy_margin(y_rl_test.to_numpy(), rl_pred)
            cal = calibration_stats(y_rl_test.to_numpy(), rl_pred)

            msg = (
                f"Fold {k:02d} | train_end={train_end.date()} "
                f"test=[{train_end.date()}..{test_end.date()}) rows train={n_train} test={n_test} | "
                f"DIRECT RL MAE={rl_mae:.3f} RMSE={rl_rmse:.3f} DIR={rl_dir:.3f} "
                f"std_ratio={cal['std_ratio']:.3f} slope={cal['slope']:.3f} | "
                f"DIRECT TOTAL MAE={tot_mae:.3f} RMSE={tot_rmse:.3f}"
            )

            # ATS cover rate
            market_rl_test = df.loc[test_mask, "run_line_home"].values if "run_line_home" in df.columns else np.full(n_test, np.nan)
            has_market_mask = ~np.isnan(market_rl_test.astype(float))
            if has_market_mask.sum() >= 3:
                mkt_rl = market_rl_test[has_market_mask].astype(float)
                edge = rl_pred[has_market_mask] + mkt_rl
                actual_rd = y_rl_test.to_numpy()[has_market_mask]
                actual_covers = actual_rd > -mkt_rl
                pred_home = edge > 0
                ats_correct = pred_home == actual_covers
                ats_n = len(ats_correct)
                ats_pct = float(ats_correct.mean())
                ats_roi = (ats_correct.sum() * 100 - (~ats_correct).sum() * 110) / (ats_n * 110)
                hc_mask = np.abs(edge) >= 1.5
                hc_n = int(hc_mask.sum())
                hc_ats_pct = float(ats_correct[hc_mask].mean()) if hc_n >= 2 else float("nan")
                log.info(
                    "  ATS n=%d pct=%.3f roi=%.3f | HC(>=1.5r) n=%d pct=%s",
                    ats_n, ats_pct, ats_roi, hc_n,
                    f"{hc_ats_pct:.3f}" if not np.isnan(hc_ats_pct) else "n/a",
                )
                ats_correct_all.extend(ats_correct.tolist())
                if hc_n >= 1:
                    ats_hc_correct_all.extend(ats_correct[hc_mask].tolist())

            fold_rows += n_test
            run_line_true_all.extend(y_rl_test.to_list())
            run_line_pred_all.extend(rl_pred.tolist())
            total_true_all.extend(y_tot_test.to_list())
            total_pred_all.extend(tot_pred.tolist())

            # RESIDUAL MODELS (market line + XGB correction)
            has_market_test = (
                df.loc[test_mask, "run_line_home"].notna()
                & df.loc[test_mask, "total_line"].notna()
            ) if ("run_line_home" in df.columns and "total_line" in df.columns) else pd.Series(False, index=df.loc[test_mask].index)

            has_market_train = (
                df.loc[train_mask, "run_line_home"].notna()
                & df.loc[train_mask, "total_line"].notna()
                & df.loc[train_mask, "run_line_residual"].notna()
                & df.loc[train_mask, "total_residual"].notna()
            ) if all(c in df.columns for c in ("run_line_home", "total_line", "run_line_residual", "total_residual")) else pd.Series(False, index=df.loc[train_mask].index)

            n_resid_test  = int(has_market_test.sum())
            n_resid_train = int(has_market_train.sum())

            if n_resid_train >= cfg.min_market_rows and n_resid_test >= 5:
                y_rl_resid_train  = df.loc[train_mask & has_market_train, "run_line_residual"].astype(float)
                y_tot_resid_train = df.loc[train_mask & has_market_train, "total_residual"].astype(float)

                X_train_resid = X_train.loc[has_market_train]
                X_test_resid  = X_test.loc[has_market_test]

                resid_train_dates = df.loc[train_mask & has_market_train, "game_date_et"]
                resid_fit, resid_eval = temporal_eval_split(resid_train_dates)
                resid_weights = compute_recency_weights(
                    resid_train_dates,
                    half_life_days=cfg.recency_half_life_days,
                )
                resid_weights_fit = resid_weights[resid_fit]

                rl_resid_model  = build_model(cfg, params_override=best_run_line_params, objective=RUN_LINE_OBJECTIVE)
                tot_resid_model = build_model(cfg, params_override=best_total_params,    objective=TOTAL_RESID_OBJECTIVE)

                if resid_fit.sum() > 10 and resid_eval.sum() > 5:
                    rl_resid_model.fit(
                        X_train_resid.iloc[resid_fit], y_rl_resid_train.iloc[resid_fit],
                        sample_weight=resid_weights_fit,
                        eval_set=[(X_train_resid.iloc[resid_eval], y_rl_resid_train.iloc[resid_eval])],
                        verbose=False,
                    )
                    tot_resid_model.fit(
                        X_train_resid.iloc[resid_fit], y_tot_resid_train.iloc[resid_fit],
                        sample_weight=resid_weights_fit,
                        eval_set=[(X_train_resid.iloc[resid_eval], y_tot_resid_train.iloc[resid_eval])],
                        verbose=False,
                    )

                    pred_rl_resid  = rl_resid_model.predict(X_test_resid)
                    pred_tot_resid = tot_resid_model.predict(X_test_resid)

                    if getattr(rl_resid_model, "best_iteration", None) is not None:
                        resid_rl_best_iters.append(rl_resid_model.best_iteration)
                    if getattr(tot_resid_model, "best_iteration", None) is not None:
                        resid_tot_best_iters.append(tot_resid_model.best_iteration)

                    market_rl  = df.loc[test_mask & has_market_test, "run_line_home"].astype(float).values
                    market_tot = df.loc[test_mask & has_market_test, "total_line"].astype(float).values

                    pred_rl_direct_market  = rl_model.predict(X_test_resid)
                    pred_tot_direct_market = tot_model.predict(X_test_resid)
                    direct_market_rl_pred_all.extend(pred_rl_direct_market.tolist())
                    direct_market_tot_pred_all.extend(pred_tot_direct_market.tolist())
                    market_rl_baseline_all.extend(market_rl.tolist())
                    market_tot_baseline_all.extend(market_tot.tolist())

                    pred_rl_recon  = market_rl  + pred_rl_resid
                    pred_tot_recon = market_tot + pred_tot_resid

                    y_rl_true  = df.loc[test_mask & has_market_test, "run_diff"].astype(float).values
                    y_tot_true = df.loc[test_mask & has_market_test, "total_runs"].astype(float).values

                    resid_rl_true_all.extend(y_rl_true.tolist())
                    resid_rl_pred_all.extend(pred_rl_recon.tolist())
                    resid_total_true_all.extend(y_tot_true.tolist())
                    resid_total_pred_all.extend(pred_tot_recon.tolist())

                    r_mae_rl  = mean_absolute_error(y_rl_true,  pred_rl_recon)
                    r_rmse_rl = _rmse(y_rl_true, pred_rl_recon)
                    r_dir_rl  = _directional_accuracy_margin(y_rl_true, pred_rl_recon)
                    r_mae_tot = mean_absolute_error(y_tot_true, pred_tot_recon)

                    msg += (
                        f" | RESID(recon) n={n_resid_test} "
                        f"RL MAE={r_mae_rl:.3f} RMSE={r_rmse_rl:.3f} DIR={r_dir_rl:.3f} | "
                        f"TOTAL MAE={r_mae_tot:.3f}"
                    )

            log.info(msg)

        if fold_rows == 0:
            raise RuntimeError("All folds had 0 test rows. Something is off with fold boundaries.")

        # Overall walk-forward metrics
        rl_true_np  = np.asarray(run_line_true_all, dtype=float)
        rl_pred_np  = np.asarray(run_line_pred_all, dtype=float)
        tot_true_np = np.asarray(total_true_all,    dtype=float)
        tot_pred_np = np.asarray(total_pred_all,    dtype=float)

        rl_mae,  rl_rmse  = evaluate_regression(rl_true_np, rl_pred_np)
        tot_mae, tot_rmse = evaluate_regression(tot_true_np, tot_pred_np)
        rl_dir = directional_accuracy_margin(rl_true_np, rl_pred_np)
        cal = calibration_stats(rl_true_np, rl_pred_np)

        oof_total_mean_bias = float(np.mean(tot_pred_np - tot_true_np))
        oof_rl_mean_bias    = float(np.mean(rl_pred_np  - rl_true_np))
        log.info("OOF MEAN BIAS | total=%.3f runs  rl=%.3f runs",
                 oof_total_mean_bias, oof_rl_mean_bias)

        log.info(
            "WALK-FORWARD DIRECT OVERALL | rows=%d | RL MAE=%.3f RMSE=%.3f DIR=%.3f "
            "std_ratio=%.3f slope=%.3f intercept=%.3f | TOTAL MAE=%.3f RMSE=%.3f",
            len(rl_true_np),
            rl_mae, rl_rmse, rl_dir,
            cal["std_ratio"], cal["slope"], cal["intercept"],
            tot_mae, tot_rmse,
        )

        if ats_correct_all:
            ats_arr = np.asarray(ats_correct_all)
            ats_n_total = len(ats_arr)
            ats_pct_total = float(ats_arr.mean())
            ats_roi_total = (ats_arr.sum() * 100 - (~ats_arr).sum() * 110) / (ats_n_total * 110)
            log.info(
                "WALK-FORWARD ATS OVERALL | n=%d wins=%d pct=%.3f roi=%.3f",
                ats_n_total, int(ats_arr.sum()), ats_pct_total, ats_roi_total,
            )
            if ats_hc_correct_all:
                hc_arr = np.asarray(ats_hc_correct_all)
                log.info(
                    "WALK-FORWARD ATS HC(>=1.5r) | n=%d wins=%d pct=%.3f roi=%.3f",
                    len(hc_arr), int(hc_arr.sum()), float(hc_arr.mean()),
                    (hc_arr.sum() * 100 - (~hc_arr).sum() * 110) / (len(hc_arr) * 110),
                )

        # CI calibration: empirical error percentiles
        rl_abs  = np.abs(rl_true_np  - rl_pred_np)
        tot_abs = np.abs(tot_true_np - tot_pred_np)
        rl_p68,  rl_p90  = float(np.percentile(rl_abs,  68)), float(np.percentile(rl_abs,  90))
        tot_p68, tot_p90 = float(np.percentile(tot_abs, 68)), float(np.percentile(tot_abs, 90))
        log.info(
            "CI CALIBRATION DIRECT | RL ±MAE covers %.1f%% (p68=%.2f p90=%.2f RMSE=%.2f) | "
            "TOTAL ±MAE covers %.1f%% (p68=%.2f p90=%.2f RMSE=%.2f)",
            (rl_abs < rl_mae).mean() * 100, rl_p68, rl_p90, rl_rmse,
            (tot_abs < tot_mae).mean() * 100, tot_p68, tot_p90, tot_rmse,
        )

        calib = {
            "direct_spread_mae":  float(rl_mae),   # "spread" key alias for NBA compat
            "direct_spread_rmse": float(rl_rmse),
            "direct_spread_p68":  rl_p68,
            "direct_spread_p90":  rl_p90,
            "direct_rl_mae":      float(rl_mae),
            "direct_rl_rmse":     float(rl_rmse),
            "direct_rl_dir":      float(rl_dir),
            "direct_rl_p68":      rl_p68,
            "direct_rl_p90":      rl_p90,
            "direct_total_mae":   float(tot_mae),
            "direct_total_rmse":  float(tot_rmse),
            "direct_total_p68":   tot_p68,
            "direct_total_p90":   tot_p90,
            "oof_total_mean_bias": oof_total_mean_bias,
            "oof_rl_mean_bias":    oof_rl_mean_bias,
        }

        # F5 walk-forward MAE
        if f5_true_all:
            f5_true_np = np.asarray(f5_true_all, dtype=float)
            f5_pred_np = np.asarray(f5_pred_all, dtype=float)
            f5_mae, _ = evaluate_regression(f5_true_np, f5_pred_np)
            calib["direct_f5_mae"] = float(f5_mae)
            log.info("WALK-FORWARD F5 TOTAL | rows=%d | MAE=%.3f", len(f5_true_np), f5_mae)

        if resid_rl_true_all:
            r_true  = np.asarray(resid_rl_true_all,    dtype=float)
            r_pred  = np.asarray(resid_rl_pred_all,    dtype=float)
            rt_true = np.asarray(resid_total_true_all, dtype=float)
            rt_pred = np.asarray(resid_total_pred_all, dtype=float)

            r_dir = _directional_accuracy_margin(r_true, r_pred)
            r_cal = calibration_stats(r_true, r_pred)
            r_mae, r_rmse   = evaluate_regression(r_true,  r_pred)
            rt_mae, rt_rmse = evaluate_regression(rt_true, rt_pred)

            log.info(
                "WALK-FORWARD RESID(recon) OVERALL | rows=%d | RL MAE=%.3f RMSE=%.3f DIR=%.3f "
                "std_ratio=%.3f slope=%.3f | TOTAL MAE=%.3f RMSE=%.3f",
                len(r_true), r_mae, r_rmse, r_dir,
                r_cal["std_ratio"], r_cal["slope"],
                rt_mae, rt_rmse,
            )
            rs_abs  = np.abs(r_true  - r_pred)
            rtt_abs = np.abs(rt_true - rt_pred)
            rs_p68, rs_p90   = float(np.percentile(rs_abs,  68)), float(np.percentile(rs_abs,  90))
            rtt_p68, rtt_p90 = float(np.percentile(rtt_abs, 68)), float(np.percentile(rtt_abs, 90))

            calib.update({
                "resid_spread_mae":  float(r_mae),
                "resid_spread_rmse": float(r_rmse),
                "resid_spread_p68":  rs_p68,
                "resid_spread_p90":  rs_p90,
                "resid_rl_mae":      float(r_mae),
                "resid_rl_rmse":     float(r_rmse),
                "resid_rl_p68":      rs_p68,
                "resid_rl_p90":      rs_p90,
                "resid_total_mae":   float(rt_mae),
                "resid_total_rmse":  float(rt_rmse),
                "resid_total_p68":   rtt_p68,
                "resid_total_p90":   rtt_p90,
            })

            if direct_market_rl_pred_all:
                dm_rl_mae  = float(mean_absolute_error(resid_rl_true_all,    direct_market_rl_pred_all))
                dm_tot_mae = float(mean_absolute_error(resid_total_true_all, direct_market_tot_pred_all))
                mkt_rl_mae  = float(mean_absolute_error(resid_rl_true_all,    market_rl_baseline_all))
                mkt_tot_mae = float(mean_absolute_error(resid_total_true_all, market_tot_baseline_all))

                calib.update({
                    "direct_spread_mae_market": dm_rl_mae,
                    "direct_total_mae_market":  dm_tot_mae,
                    "market_spread_mae":        mkt_rl_mae,
                    "market_total_mae":         mkt_tot_mae,
                })
                log.info(
                    "QUALITY GATE BASELINES (market-data rows=%d) | "
                    "direct_mkt: rl=%.3f total=%.3f | "
                    "raw_market: rl=%.3f total=%.3f | "
                    "resid(recon): rl=%.3f total=%.3f",
                    len(resid_rl_true_all),
                    dm_rl_mae, dm_tot_mae,
                    mkt_rl_mae, mkt_tot_mae,
                    float(r_mae), float(rt_mae),
                )

                best_baseline_rl  = min(dm_rl_mae,  mkt_rl_mae)
                best_baseline_tot = min(dm_tot_mae, mkt_tot_mae)

                blend_alpha_rl  = 0.0
                blend_alpha_tot = 0.0
                if float(r_mae) < best_baseline_rl * 1.005:
                    w_d = 1.0 / dm_rl_mae
                    w_r = 1.0 / float(r_mae)
                    blend_alpha_rl = round(w_r / (w_d + w_r), 4)
                if float(rt_mae) < best_baseline_tot * 1.005:
                    w_d = 1.0 / dm_tot_mae
                    w_r = 1.0 / float(rt_mae)
                    blend_alpha_tot = round(w_r / (w_d + w_r), 4)

                log.info(
                    "BLEND WEIGHTS | rl_alpha=%.3f total_alpha=%.3f "
                    "(0.0 = direct-only, 1.0 = resid-only)",
                    blend_alpha_rl, blend_alpha_tot,
                )
                calib["blend_alpha_spread"] = blend_alpha_rl
                calib["blend_alpha_total"]  = blend_alpha_tot
        else:
            r_mae = r_rmse = rt_mae = rt_rmse = None

        calib_path = model_dir / "calibration.json"
        calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        log.info("Saved CI calibration to %s", calib_path)

        # Derive final model depth from CV best_iteration statistics
        _rl_iters  = rl_best_iters  if rl_best_iters  else [cfg.n_estimators]
        _tot_iters = tot_best_iters if tot_best_iters else [cfg.n_estimators]
        rl_n_est  = max(int(np.percentile(_rl_iters,  75) * 1.2), 100)
        tot_n_est = max(int(np.percentile(_tot_iters, 75) * 1.2), 100)
        log.info(
            "CV best_iteration | RL median=%d p75=%d → final n_est=%d | "
            "TOTAL median=%d p75=%d → final n_est=%d",
            int(np.median(_rl_iters)),  int(np.percentile(_rl_iters,  75)), rl_n_est,
            int(np.median(_tot_iters)), int(np.percentile(_tot_iters, 75)), tot_n_est,
        )

        # Train FINAL models on ALL rows
        medians_all = fit_fill_stats(X_raw)
        X_all = apply_fill(X_raw, medians_all, feature_cols)

        # Prune to top features
        feature_cols, X_all = _select_top_features(
            X_all, y_run_diff, y_total, feature_cols, model_dir
        )

        # FINAL DIRECT MODELS (no early stopping — full data, fixed depth)
        rl_final = build_model(cfg, params_override=best_run_line_params,
                               objective=RUN_LINE_OBJECTIVE,
                               n_estimators=rl_n_est,
                               use_early_stopping=False)
        tot_final = build_model(cfg, params_override=best_total_params,
                                objective=TOTAL_DIRECT_OBJECTIVE,
                                n_estimators=tot_n_est,
                                use_early_stopping=False)

        log.info("Fitting FINAL DIRECT run-line model on all %d rows (n_estimators=%d)...",
                 len(X_all), rl_n_est)
        rl_final.fit(X_all, y_run_diff, verbose=False)

        log.info("Fitting FINAL DIRECT total model on all %d rows (n_estimators=%d)...",
                 len(X_all), tot_n_est)
        tot_final.fit(X_all, y_total, verbose=False)

        rl_final.save_model(str(model_dir / "run_line_direct_xgb.json"))
        tot_final.save_model(str(model_dir / "total_direct_xgb.json"))
        log.info("Saved direct models to %s", model_dir)

        # LightGBM direct ensemble (if available)
        _lgb_params = dict(
            objective="huber",
            alpha=0.9,
            metric="huber",
            num_leaves=127,   # was 63; 2^7−1 matches depth-7 XGB but grows asymmetrically
            learning_rate=0.05,
            n_estimators=500,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            n_jobs=-1,
            random_state=cfg.random_state,
        )
        if _HAS_LGB:
            log.info("Training LightGBM direct game models on %d rows...", len(X_all))
            for _lgb_target, _lgb_y, _lgb_name in [
                ("run_line", y_run_diff, "run_line_direct_lgb.txt"),
                ("total",    y_total,   "total_direct_lgb.txt"),
            ]:
                try:
                    _lgb_m = lgb.LGBMRegressor(**_lgb_params)
                    _lgb_m.fit(X_all, _lgb_y)
                    _lgb_m.booster_.save_model(str(model_dir / _lgb_name))
                    log.info("Saved LGB %s model → %s", _lgb_target, _lgb_name)
                except Exception as _exc:
                    log.warning("LGB %s training failed: %s", _lgb_target, _exc)
        else:
            log.info("lightgbm not installed — skipping LGB ensemble (pip install lightgbm)")

        # ── Quantile LGB models (q10/25/50/75/90) ─────────────────────────────────
        _QUANTILES = [10, 25, 50, 75, 90]
        if _HAS_LGB:
            log.info("Training LightGBM quantile models on %d rows...", len(X_all))
            for _q_int in _QUANTILES:
                _q = _q_int / 100.0
                _q_params = {
                    **_lgb_params,
                    "objective": "quantile",
                    "alpha": _q,
                    "metric": "quantile",
                }
                for _lgb_target, _lgb_y, _lgb_stem in [
                    ("run_line", y_run_diff, "run_line"),
                    ("total",    y_total,   "total"),
                ]:
                    try:
                        _qm = lgb.LGBMRegressor(**_q_params)
                        _qm.fit(X_all, _lgb_y)
                        _fname = f"{_lgb_stem}_q{_q_int:02d}_lgb.txt"
                        _qm.booster_.save_model(str(model_dir / _fname))
                        log.info("Saved LGB quantile q%02d %s → %s", _q_int, _lgb_target, _fname)
                    except Exception as _exc:
                        log.warning("LGB quantile q%02d %s failed: %s", _q_int, _lgb_target, _exc)

        # FINAL F5 MODEL (first 5 innings total)
        _n_f5_all = int(f5_mask.sum())
        if _n_f5_all >= 100:
            X_f5_all = X_all.loc[f5_mask]
            y_f5_all = y_f5.loc[f5_mask]
            _f5_n_est = max(int(np.percentile(f5_best_iters, 75) * 1.2), 100) if f5_best_iters else cfg.n_estimators
            f5_final = build_model(cfg, params_override=best_total_params,
                                   objective=TOTAL_DIRECT_OBJECTIVE,
                                   n_estimators=_f5_n_est,
                                   use_early_stopping=False)
            log.info("Fitting FINAL F5 model on %d rows (n_estimators=%d)...", _n_f5_all, _f5_n_est)
            f5_final.fit(X_f5_all, y_f5_all, verbose=False)
            f5_final.save_model(str(model_dir / "total_f5_direct_xgb.json"))
            log.info("Saved F5 direct model → total_f5_direct_xgb.json")
            if _HAS_LGB:
                try:
                    _f5_lgb = lgb.LGBMRegressor(**_lgb_params)
                    _f5_lgb.fit(X_f5_all, y_f5_all)
                    _f5_lgb.booster_.save_model(str(model_dir / "total_f5_direct_lgb.txt"))
                    log.info("Saved F5 LGB model → total_f5_direct_lgb.txt")
                except Exception as _exc:
                    log.warning("F5 LGB training failed: %s", _exc)
        else:
            log.info("Skipping F5 model — only %d rows with F5 data (need 100)", _n_f5_all)

        # FINAL RESIDUAL MODELS
        if all(c in df.columns for c in ("run_line_home", "total_line", "run_line_residual", "total_residual")):
            has_market_all = (
                df["run_line_home"].notna()
                & df["total_line"].notna()
                & df["run_line_residual"].notna()
                & df["total_residual"].notna()
            )
        else:
            has_market_all = pd.Series(False, index=df.index)

        n_market_all = int(has_market_all.sum())

        if n_market_all >= cfg.min_market_rows:
            log.info("Training FINAL RESIDUAL models on %d rows with market data...", n_market_all)

            y_rl_resid_all  = df.loc[has_market_all, "run_line_residual"].astype(float)
            y_tot_resid_all = df.loc[has_market_all, "total_residual"].astype(float)
            X_resid_all     = X_all.loc[has_market_all]

            _rs_iters = resid_rl_best_iters  if resid_rl_best_iters  else [cfg.n_estimators]
            _rt_iters = resid_tot_best_iters if resid_tot_best_iters else [cfg.n_estimators]
            resid_rl_n_est  = max(int(np.percentile(_rs_iters, 75) * 1.2), 100)
            resid_tot_n_est = max(int(np.percentile(_rt_iters, 75) * 1.2), 100)

            rl_resid_final = build_model(cfg, params_override=best_run_line_params,
                                         objective=RUN_LINE_OBJECTIVE,
                                         n_estimators=resid_rl_n_est,
                                         use_early_stopping=False)
            tot_resid_final = build_model(cfg, params_override=best_total_params,
                                          objective=TOTAL_RESID_OBJECTIVE,
                                          n_estimators=resid_tot_n_est,
                                          use_early_stopping=False)

            log.info("Fitting FINAL RESIDUAL run-line model on %d rows (n_estimators=%d)...",
                     n_market_all, resid_rl_n_est)
            rl_resid_final.fit(X_resid_all, y_rl_resid_all, verbose=False)

            log.info("Fitting FINAL RESIDUAL total model on %d rows (n_estimators=%d)...",
                     n_market_all, resid_tot_n_est)
            tot_resid_final.fit(X_resid_all, y_tot_resid_all, verbose=False)

            rl_resid_final.save_model(str(model_dir / "run_line_resid_xgb.json"))
            tot_resid_final.save_model(str(model_dir / "total_resid_xgb.json"))
            log.info("Saved RESIDUAL models to %s", model_dir)
        else:
            log.warning("Only %d rows have market data (need %d). Skipping residual models.",
                        n_market_all, cfg.min_market_rows)

        # Save schema artifacts
        selected_medians = {c: medians_all[c] for c in feature_cols}
        (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
        (model_dir / "feature_medians.json").write_text(json.dumps(selected_medians), encoding="utf-8")

        # Save Optuna best params for reproducibility
        if best_run_line_params or best_total_params:
            optuna_results = {
                "run_line_params": best_run_line_params,
                "total_params":    best_total_params,
            }
            (model_dir / "optuna_best_params.json").write_text(
                json.dumps(optuna_results, indent=2), encoding="utf-8"
            )
            log.info("Saved Optuna best params to %s", model_dir / "optuna_best_params.json")

        log.info("Saved models + feature schema to %s", model_dir)


if __name__ == "__main__":
    main()
