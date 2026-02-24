# src/nba_pipeline/modeling/train_game_models.py
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

log = logging.getLogger("nba_pipeline.modeling.train_game_models")


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    # Minimum number of market rows required to train residual models
    min_market_rows: int = 50
    run_walk_forward: bool = True
    min_train_days: int = 60
    test_window_days: int = 7
    step_days: int = 7

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
    early_stopping_rounds: int = 50
    # Huber loss
    huber_slope: float = 5.0

    # Optuna tuning
    run_optuna: bool = True
    optuna_n_trials: int = 40
    optuna_n_folds: int = 5      # walk-forward folds used for tuning


SQL_GAME_TRAINING_FEATURES = """
SELECT *
FROM features.game_training_features
WHERE margin IS NOT NULL
ORDER BY game_date_et, game_slug
"""


def load_training_frame(conn) -> pd.DataFrame:
    df = pd.read_sql(SQL_GAME_TRAINING_FEATURES, conn)
    if df.empty:
        raise RuntimeError("No rows returned from features.game_training_features (df is empty).")

    df["game_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce")
    if df["game_date_et"].isna().any():
        bad = df[df["game_date_et"].isna()].head(5)
        raise RuntimeError(f"Some game_date_et values could not be parsed. Examples:\n{bad}")

    return df.sort_values(["game_date_et", "game_slug"]).reset_index(drop=True)


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build (X, y_spread, y_total) for DIRECT models.
    - Drops leaky score columns.
    - One-hot encodes season/home/away team IDs (as you already chose).
    - Keeps market columns if present (they are valid pregame features).
    """

    y_spread = df["margin"].astype(float)
    y_total = df["total_points"].astype(float)

    keep_cats = ["season", "home_team_abbr", "away_team_abbr"]

    drop_cols = {
        "game_slug",
        "game_date_et",
        "margin",
        "total_points",
        # leakage:
        "home_score",
        "away_score",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # timing features
    if "start_ts_utc" in X.columns:
        ts = pd.to_datetime(X["start_ts_utc"], errors="coerce", utc=True)
        X["start_hour_utc"] = ts.dt.hour
        X["start_dow_utc"] = ts.dt.dayofweek
        X = X.drop(columns=["start_ts_utc"])

    # b2b as 0/1
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # ensure cats exist then one-hot
    for c in keep_cats:
        if c not in X.columns and c in df.columns:
            X[c] = df[c]

    cat_cols = [c for c in keep_cats if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    X = _coerce_numeric_cols(X)

    # fill numeric NaNs with global medians
    numeric_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median(numeric_only=True))

    # last-resort one-hot
    non_numeric_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if non_numeric_cols:
        log.warning("One-hot encoding remaining non-numeric cols: %s", non_numeric_cols)
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False, dummy_na=True)

    X = X.fillna(0.0)

    still_bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if still_bad:
        raise RuntimeError(f"Non-numeric columns remain after encoding: {still_bad[:20]}")

    return X, y_spread, y_total


def build_model(cfg: TrainConfig, params_override: Optional[Dict] = None) -> XGBRegressor:
    """Build XGBRegressor with config defaults, optionally overridden by Optuna params."""
    p = {
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "objective": "reg:pseudohubererror",
        "huber_slope": cfg.huber_slope,
        "min_child_weight": cfg.min_child_weight,
        "gamma": cfg.gamma,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "eval_metric": "mae",
        "random_state": cfg.random_state,
        "n_jobs": -1,
    }
    if params_override:
        p.update(params_override)
    return XGBRegressor(**p)


# -------------------------
# Metrics
# -------------------------
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def _directional_accuracy_margin(y_true_margin: np.ndarray, y_pred_margin: np.ndarray) -> float:
    # Correct sign of margin (home - away)
    return float(np.mean(np.sign(y_true_margin) == np.sign(y_pred_margin)))


def _calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    # y_true ~ a + b*y_pred
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


def _std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s_true = float(np.std(y_true))
    s_pred = float(np.std(y_pred))
    if s_true <= 1e-12:
        return np.nan
    return s_pred / s_true


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    log.info("%s | MAE=%.4f RMSE=%.4f", name, mae, rmse)
    return mae, rmse


def _save_schema_artifacts(model_dir: Path, feature_cols: list[str], feature_medians: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (model_dir / "feature_medians.json").write_text(json.dumps(feature_medians), encoding="utf-8")



def make_xy_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X with NaNs preserved (no median fill here).
    We'll fit medians on *train only* inside each walk-forward fold (prevents leakage).
    """
    y_spread = df["margin"].astype(float)
    y_total = df["total_points"].astype(float)

    keep_cats = ["season", "home_team_abbr", "away_team_abbr"]

    drop_cols = {
        "game_slug",
        "game_date_et",
        "margin",
        "total_points",
        # leakage (postgame):
        "home_score",
        "away_score",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Safety: drop obviously postgame/leaky columns if they ever appear
    leaky_prefixes = (
        "box_", "pbp_", "final_", "actual_", "result_",
    )
    leaky_exact = {
        "home_score", "away_score", "total_points", "margin",
        "home_points", "away_points",
    }
    leaky_cols = [c for c in X.columns if c in leaky_exact or c.startswith(leaky_prefixes)]
    if leaky_cols:
        log.warning("Dropping %d potentially leaky cols: %s", len(leaky_cols), leaky_cols[:20])
        X = X.drop(columns=leaky_cols)

    # Timing features (optional). Keep if present.
    if "start_ts_utc" in X.columns:
        ts = pd.to_datetime(X["start_ts_utc"], errors="coerce", utc=True)
        X["start_hour_utc"] = ts.dt.hour
        X["start_dow_utc"] = ts.dt.dayofweek
        X = X.drop(columns=["start_ts_utc"])

    # Ensure b2b flags are 0/1
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # One-hot encode season + teams (stable schema)
    for c in keep_cats:
        if c not in X.columns and c in df.columns:
            X[c] = df[c]
    cat_cols = [c for c in keep_cats if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # Coerce remaining strings/objects to numeric where possible
    X = _coerce_numeric_cols(X)

    # Last-resort one-hot for any remaining non-numeric columns
    non_numeric_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if non_numeric_cols:
        log.warning("One-hot encoding remaining non-numeric cols: %s", non_numeric_cols)
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False, dummy_na=True)

    # --- Derived interaction features ---
    if "home_rest_days" in X.columns and "away_rest_days" in X.columns:
        X["rest_advantage_home"] = X["home_rest_days"] - X["away_rest_days"]

    if "home_pts_for_avg_10" in X.columns and "home_pts_against_avg_10" in X.columns:
        X["home_net_rating_10"] = X["home_pts_for_avg_10"] - X["home_pts_against_avg_10"]
    if "away_pts_for_avg_10" in X.columns and "away_pts_against_avg_10" in X.columns:
        X["away_net_rating_10"] = X["away_pts_for_avg_10"] - X["away_pts_against_avg_10"]
    if "home_net_rating_10" in X.columns and "away_net_rating_10" in X.columns:
        X["net_rating_diff_10"] = X["home_net_rating_10"] - X["away_net_rating_10"]

    if "home_pace_avg_5" in X.columns and "away_pace_avg_5" in X.columns:
        X["pace_diff_5"] = X["home_pace_avg_5"] - X["away_pace_avg_5"]

    if "home_pts_for_avg_5" in X.columns and "away_pts_for_avg_5" in X.columns:
        X["pts_for_diff_5"] = X["home_pts_for_avg_5"] - X["away_pts_for_avg_5"]

    # --- NEW: Efficiency differentials ---
    if "home_efg_pct_avg_5" in X.columns and "away_efg_pct_avg_5" in X.columns:
        X["efg_diff_5"] = X["home_efg_pct_avg_5"] - X["away_efg_pct_avg_5"]
    if "home_ts_pct_avg_5" in X.columns and "away_ts_pct_avg_5" in X.columns:
        X["ts_diff_5"] = X["home_ts_pct_avg_5"] - X["away_ts_pct_avg_5"]
    if "home_fg3a_rate_avg_5" in X.columns and "away_fg3a_rate_avg_5" in X.columns:
        X["fg3a_rate_diff_5"] = X["home_fg3a_rate_avg_5"] - X["away_fg3a_rate_avg_5"]
    if "home_tov_rate_avg_5" in X.columns and "away_tov_rate_avg_5" in X.columns:
        X["tov_rate_diff_5"] = X["home_tov_rate_avg_5"] - X["away_tov_rate_avg_5"]

    # --- NEW: Injury impact differential ---
    if "home_injured_pts_lost" in X.columns and "away_injured_pts_lost" in X.columns:
        X["injury_pts_diff"] = X["away_injured_pts_lost"] - X["home_injured_pts_lost"]

    # --- NEW: Clutch differential ---
    if "home_clutch_net_avg_10" in X.columns and "away_clutch_net_avg_10" in X.columns:
        X["clutch_net_diff_10"] = X["home_clutch_net_avg_10"] - X["away_clutch_net_avg_10"]

    # --- V005: Odds juice derived ---
    if "spread_home_implied_prob" in X.columns:
        X["spread_implied_edge"] = X.get("spread_home_implied_prob", 0.5) - 0.5
    if "dk_spread_juice_move" in X.columns:
        pass  # raw column already in feature set, no derivation needed

    # --- V006: Team style differentials ---
    if "home_stocks_avg_10" in X.columns and "away_stocks_avg_10" in X.columns:
        X["stocks_diff_10"] = X["home_stocks_avg_10"] - X["away_stocks_avg_10"]
    if "home_ast_tov_ratio_10" in X.columns and "away_ast_tov_ratio_10" in X.columns:
        X["ast_tov_ratio_diff_10"] = X["home_ast_tov_ratio_10"] - X["away_ast_tov_ratio_10"]
    if "home_pts_paint_avg_10" in X.columns and "away_pts_paint_avg_10" in X.columns:
        X["paint_pts_diff_10"] = X["home_pts_paint_avg_10"] - X["away_pts_paint_avg_10"]
    if "home_pts_fast_break_avg_10" in X.columns and "away_pts_fast_break_avg_10" in X.columns:
        X["fast_break_diff_10"] = X["home_pts_fast_break_avg_10"] - X["away_pts_fast_break_avg_10"]
    if "home_bench_pct_10" in X.columns and "away_bench_pct_10" in X.columns:
        X["bench_depth_diff_10"] = X["home_bench_pct_10"] - X["away_bench_pct_10"]
    if "home_fouls_avg_10" in X.columns and "away_fouls_avg_10" in X.columns:
        X["fouls_diff_10"] = X["home_fouls_avg_10"] - X["away_fouls_avg_10"]

    # --- V008: Lineup stability differential ---
    if "home_starter_continuity_avg_10" in X.columns and "away_starter_continuity_avg_10" in X.columns:
        X["continuity_diff_10"] = X["home_starter_continuity_avg_10"] - X["away_starter_continuity_avg_10"]

    # --- V009: Standings differentials ---
    if "home_streak" in X.columns and "away_streak" in X.columns:
        X["streak_diff"] = X["home_streak"] - X["away_streak"]
    if "home_last10_pct" in X.columns and "away_last10_pct" in X.columns:
        X["last10_pct_diff"] = X["home_last10_pct"] - X["away_last10_pct"]
    if "home_home_record_pct" in X.columns and "away_away_record_pct" in X.columns:
        X["venue_record_diff"] = X["home_home_record_pct"] - X["away_away_record_pct"]

    # --- V011: PBP differentials ---
    if "home_three_pt_rate_avg_10" in X.columns and "away_three_pt_rate_avg_10" in X.columns:
        X["three_pt_rate_diff_10"] = X["home_three_pt_rate_avg_10"] - X["away_three_pt_rate_avg_10"]

    # At this point: numeric columns, but may contain NaNs.
    still_bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if still_bad:
        raise RuntimeError(f"Non-numeric columns remain after encoding: {still_bad[:20]}")

    return X, y_spread, y_total


def fit_fill_stats(X_train: pd.DataFrame) -> Dict[str, float]:
    """Return per-column medians (train only) for NaN filling."""
    # median(numeric_only=True) returns Series indexed by columns
    meds = X_train.median(numeric_only=True)
    return {str(k): float(v) for k, v in meds.items()}


def apply_fill(X: pd.DataFrame, medians: Dict[str, float], columns: List[str]) -> pd.DataFrame:
    """
    Align to `columns`, fill NaNs with train medians, then final fill of 0.0.
    """
    X2 = X.reindex(columns=columns)
    # fill with train medians
    for c, m in medians.items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(m)
    # any columns that never had a median (all NaN in train) -> 0
    return X2.fillna(0.0)


def temporal_eval_split(
    dates: pd.Series, eval_frac: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (fit_mask, eval_mask) splitting by date quantile."""
    cutoff = dates.quantile(1.0 - eval_frac)
    fit_mask = (dates < cutoff).values
    eval_mask = (dates >= cutoff).values
    return fit_mask, eval_mask


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def directional_accuracy_margin(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spread direction: did we predict the winner (sign of margin)?
    Treat 0 as no-side; exclude true==0 games from denominator (rare).
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def calibration_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calibration for margin predictions.
    - std_ratio close to 1 means variance matches reality
    - slope/intercept from y_true ≈ a + b*y_pred; b close to 1 is ideal
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    std_true = float(np.std(y_true)) if len(y_true) else 0.0
    std_pred = float(np.std(y_pred)) if len(y_pred) else 0.0
    std_ratio = float(std_pred / std_true) if std_true > 0 else float("nan")

    # Linear fit: y_true = a + b*y_pred
    if len(y_true) >= 2 and np.std(y_pred) > 1e-9:
        b, a = np.polyfit(y_pred, y_true, deg=1)  # returns [slope, intercept]
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


# =========================================================================
# Optuna hyperparameter tuning
# =========================================================================
def _optuna_objective(
    trial,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_target: pd.Series,
    feature_cols: List[str],
    folds: List[Tuple[pd.Timestamp, pd.Timestamp]],
    target_name: str,
) -> float:
    """
    Optuna objective: walk-forward MAE averaged across a subset of folds.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "huber_slope": trial.suggest_float("huber_slope", 2.0, 10.0),
    }

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

        # Temporal eval split for early stopping
        train_dates = df.loc[train_mask, "game_date_et"]
        fit_rel, eval_rel = temporal_eval_split(train_dates)

        model = XGBRegressor(
            n_estimators=2000,
            objective="reg:pseudohubererror",
            early_stopping_rounds=50,
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
            **params,
        )

        model.fit(
            X_train.iloc[fit_rel], y_train.iloc[fit_rel],
            eval_set=[(X_train.iloc[eval_rel], y_train.iloc[eval_rel])],
            verbose=False,
        )

        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        mae_scores.append(mae)

    if not mae_scores:
        return float("inf")

    return float(np.mean(mae_scores))


def run_optuna_tuning(
    cfg: TrainConfig,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_spread: pd.Series,
    y_total: pd.Series,
    feature_cols: List[str],
) -> Tuple[Dict, Dict]:
    """
    Run Optuna hyperparameter search for both spread and total models.
    Returns (best_spread_params, best_total_params).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use last N folds for tuning (faster than all folds)
    all_folds = walk_forward_folds(
        df,
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
    )
    # Take last optuna_n_folds folds (most representative of recent data)
    tune_folds = all_folds[-cfg.optuna_n_folds:] if len(all_folds) > cfg.optuna_n_folds else all_folds

    if not tune_folds:
        log.warning("No folds available for Optuna tuning. Using default params.")
        return {}, {}

    log.info("Running Optuna tuning for SPREAD model (%d trials, %d folds)...",
             cfg.optuna_n_trials, len(tune_folds))
    spread_study = optuna.create_study(direction="minimize", study_name="spread")
    spread_study.optimize(
        lambda trial: _optuna_objective(
            trial, df, X_raw, y_spread, feature_cols, tune_folds, "spread"
        ),
        n_trials=cfg.optuna_n_trials,
        show_progress_bar=False,
    )
    best_spread = spread_study.best_params
    log.info("SPREAD best params (MAE=%.3f): %s", spread_study.best_value, best_spread)

    log.info("Running Optuna tuning for TOTAL model (%d trials, %d folds)...",
             cfg.optuna_n_trials, len(tune_folds))
    total_study = optuna.create_study(direction="minimize", study_name="total")
    total_study.optimize(
        lambda trial: _optuna_objective(
            trial, df, X_raw, y_total, feature_cols, tune_folds, "total"
        ),
        n_trials=cfg.optuna_n_trials,
        show_progress_bar=False,
    )
    best_total = total_study.best_params
    log.info("TOTAL best params (MAE=%.3f): %s", total_study.best_value, best_total)

    return best_spread, best_total


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
        log.info("Loaded %d rows from game_training_features", len(df))

        # --- Build RAW X (NaNs preserved) once ---
        X_raw, y_spread, y_total = make_xy_raw(df)
        feature_cols = list(X_raw.columns)

        # --- Optuna hyperparameter tuning ---
        best_spread_params: Dict = {}
        best_total_params: Dict = {}
        if cfg.run_optuna:
            try:
                best_spread_params, best_total_params = run_optuna_tuning(
                    cfg, df, X_raw, y_spread, y_total, feature_cols
                )
            except ImportError:
                log.warning("optuna not installed. Skipping tuning. pip install optuna")
            except Exception as e:
                log.warning("Optuna tuning failed: %s. Using default params.", e)

        # --- Walk-forward folds ---
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
        spread_true_all: List[float] = []
        spread_pred_all: List[float] = []
        total_true_all: List[float] = []
        total_pred_all: List[float] = []

        # Residual model aggregates
        resid_spread_true_all: List[float] = []
        resid_spread_pred_all: List[float] = []
        resid_total_true_all: List[float] = []
        resid_total_pred_all: List[float] = []

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

            y_spread_train = y_spread.loc[train_mask]
            y_spread_test = y_spread.loc[test_mask]
            y_total_train = y_total.loc[train_mask]
            y_total_test = y_total.loc[test_mask]

            # Temporal eval split for early stopping (last 15% of train by date)
            train_dates = df.loc[train_mask, "game_date_et"]
            fit_rel, eval_rel = temporal_eval_split(train_dates)
            X_fit = X_train.iloc[fit_rel]
            X_eval = X_train.iloc[eval_rel]

            # --- DIRECT MODELS ---
            spread_model = build_model(cfg, params_override=best_spread_params)
            total_model = build_model(cfg, params_override=best_total_params)

            spread_model.fit(
                X_fit, y_spread_train.iloc[fit_rel],
                eval_set=[(X_eval, y_spread_train.iloc[eval_rel])],
                verbose=False,
            )
            total_model.fit(
                X_fit, y_total_train.iloc[fit_rel],
                eval_set=[(X_eval, y_total_train.iloc[eval_rel])],
                verbose=False,
            )

            spread_pred = spread_model.predict(X_test)
            total_pred = total_model.predict(X_test)

            # Direct metrics
            s_mae, s_rmse = evaluate_regression(y_spread_test.to_numpy(), spread_pred)
            t_mae, t_rmse = evaluate_regression(y_total_test.to_numpy(), total_pred)
            s_dir = directional_accuracy_margin(y_spread_test.to_numpy(), spread_pred)
            cal = calibration_stats(y_spread_test.to_numpy(), spread_pred)

            msg = (
                f"Fold {k:02d} | train_end={train_end.date()} "
                f"test=[{train_end.date()}..{test_end.date()}) rows train={n_train} test={n_test} | "
                f"DIRECT SPREAD MAE={s_mae:.3f} RMSE={s_rmse:.3f} DIR={s_dir:.3f} std_ratio={cal['std_ratio']:.3f} slope={cal['slope']:.3f} | "
                f"DIRECT TOTAL MAE={t_mae:.3f} RMSE={t_rmse:.3f}"
            )

            fold_rows += n_test
            spread_true_all.extend(y_spread_test.to_list())
            spread_pred_all.extend(spread_pred.tolist())
            total_true_all.extend(y_total_test.to_list())
            total_pred_all.extend(total_pred.tolist())

            # --- RESIDUAL MODELS (market line + XGB correction) ---
            has_market_test = (
                df.loc[test_mask, "market_spread_home"].notna()
                & df.loc[test_mask, "market_total"].notna()
            )
            has_market_train = (
                df.loc[train_mask, "market_spread_home"].notna()
                & df.loc[train_mask, "market_total"].notna()
                & df.loc[train_mask, "spread_residual"].notna()
                & df.loc[train_mask, "total_residual"].notna()
            )

            n_resid_test = int(has_market_test.sum())
            n_resid_train = int(has_market_train.sum())

            if n_resid_train >= cfg.min_market_rows and n_resid_test >= 5:
                # Residual targets
                y_spread_resid_train = df.loc[train_mask & has_market_train, "spread_residual"].astype(float)
                y_total_resid_train = df.loc[train_mask & has_market_train, "total_residual"].astype(float)

                X_train_resid = X_train.loc[has_market_train.values[train_mask.values]]
                X_test_resid = X_test.loc[has_market_test.values[test_mask.values]]

                # Temporal eval split for residual training
                resid_train_dates = df.loc[train_mask & has_market_train, "game_date_et"]
                resid_fit, resid_eval = temporal_eval_split(resid_train_dates)

                spread_resid_model = build_model(cfg, params_override=best_spread_params)
                total_resid_model = build_model(cfg, params_override=best_total_params)

                if resid_fit.sum() > 10 and resid_eval.sum() > 5:
                    spread_resid_model.fit(
                        X_train_resid.iloc[resid_fit], y_spread_resid_train.iloc[resid_fit],
                        eval_set=[(X_train_resid.iloc[resid_eval], y_spread_resid_train.iloc[resid_eval])],
                        verbose=False,
                    )
                    total_resid_model.fit(
                        X_train_resid.iloc[resid_fit], y_total_resid_train.iloc[resid_fit],
                        eval_set=[(X_train_resid.iloc[resid_eval], y_total_resid_train.iloc[resid_eval])],
                        verbose=False,
                    )

                    pred_spread_resid = spread_resid_model.predict(X_test_resid)
                    pred_total_resid = total_resid_model.predict(X_test_resid)

                    market_spread = df.loc[test_mask & has_market_test, "market_spread_home"].astype(float).values
                    market_total = df.loc[test_mask & has_market_test, "market_total"].astype(float).values

                    # Reconstruct: prediction = market line + predicted residual
                    pred_margin_recon = market_spread + pred_spread_resid
                    pred_total_recon = market_total + pred_total_resid

                    y_m_true = df.loc[test_mask & has_market_test, "margin"].astype(float).values
                    y_t_true = df.loc[test_mask & has_market_test, "total_points"].astype(float).values

                    resid_spread_true_all.extend(y_m_true.tolist())
                    resid_spread_pred_all.extend(pred_margin_recon.tolist())
                    resid_total_true_all.extend(y_t_true.tolist())
                    resid_total_pred_all.extend(pred_total_recon.tolist())

                    r_mae_m = mean_absolute_error(y_m_true, pred_margin_recon)
                    r_rmse_m = _rmse(y_m_true, pred_margin_recon)
                    r_dir_m = _directional_accuracy_margin(y_m_true, pred_margin_recon)
                    r_mae_t = mean_absolute_error(y_t_true, pred_total_recon)

                    msg += (
                        f" | RESID(recon) n={n_resid_test} "
                        f"SPREAD MAE={r_mae_m:.3f} RMSE={r_rmse_m:.3f} DIR={r_dir_m:.3f} | "
                        f"TOTAL MAE={r_mae_t:.3f}"
                    )

            log.info(msg)

        if fold_rows == 0:
            raise RuntimeError("All folds had 0 test rows. Something is off with fold boundaries.")

        # --- Overall walk-forward metrics ---
        spread_true_np = np.asarray(spread_true_all, dtype=float)
        spread_pred_np = np.asarray(spread_pred_all, dtype=float)
        total_true_np = np.asarray(total_true_all, dtype=float)
        total_pred_np = np.asarray(total_pred_all, dtype=float)

        s_mae, s_rmse = evaluate_regression(spread_true_np, spread_pred_np)
        t_mae, t_rmse = evaluate_regression(total_true_np, total_pred_np)
        s_dir = directional_accuracy_margin(spread_true_np, spread_pred_np)
        cal = calibration_stats(spread_true_np, spread_pred_np)

        log.info(
            "WALK-FORWARD DIRECT OVERALL | rows=%d | SPREAD MAE=%.3f RMSE=%.3f DIR=%.3f "
            "std_ratio=%.3f slope=%.3f intercept=%.3f | TOTAL MAE=%.3f RMSE=%.3f",
            len(spread_true_np),
            s_mae,
            s_rmse,
            s_dir,
            cal["std_ratio"],
            cal["slope"],
            cal["intercept"],
            t_mae,
            t_rmse,
        )

        if resid_spread_true_all:
            r_true = np.asarray(resid_spread_true_all, dtype=float)
            r_pred = np.asarray(resid_spread_pred_all, dtype=float)
            r_dir = directional_accuracy_margin(r_true, r_pred)
            r_cal = calibration_stats(r_true, r_pred)
            r_mae, r_rmse = evaluate_regression(r_true, r_pred)
            rt_mae, rt_rmse = evaluate_regression(
                np.asarray(resid_total_true_all, dtype=float),
                np.asarray(resid_total_pred_all, dtype=float),
            )
            log.info(
                "WALK-FORWARD RESID(recon) OVERALL | rows=%d | SPREAD MAE=%.3f RMSE=%.3f DIR=%.3f "
                "std_ratio=%.3f slope=%.3f intercept=%.3f | TOTAL MAE=%.3f RMSE=%.3f",
                len(r_true), r_mae, r_rmse, r_dir,
                r_cal["std_ratio"], r_cal["slope"], r_cal["intercept"],
                rt_mae, rt_rmse,
            )

        # --- Train FINAL models on ALL rows (production) ---
        medians_all = fit_fill_stats(X_raw)
        X_all = apply_fill(X_raw, medians_all, feature_cols)

        # Temporal eval split for early stopping on final models
        all_dates = df["game_date_et"]
        fit_final, eval_final = temporal_eval_split(all_dates)
        X_fit_final = X_all.iloc[fit_final]
        X_eval_final = X_all.iloc[eval_final]

        # --- FINAL DIRECT MODELS ---
        spread_final = build_model(cfg, params_override=best_spread_params)
        total_final = build_model(cfg, params_override=best_total_params)

        log.info("Fitting FINAL DIRECT spread model on all rows...")
        spread_final.fit(
            X_fit_final, y_spread.iloc[fit_final],
            eval_set=[(X_eval_final, y_spread.iloc[eval_final])],
            verbose=False,
        )
        log.info("Spread best iteration: %d", spread_final.best_iteration)

        log.info("Fitting FINAL DIRECT total model on all rows...")
        total_final.fit(
            X_fit_final, y_total.iloc[fit_final],
            eval_set=[(X_eval_final, y_total.iloc[eval_final])],
            verbose=False,
        )
        log.info("Total best iteration: %d", total_final.best_iteration)

        spread_final.save_model(str(model_dir / "spread_direct_xgb.json"))
        total_final.save_model(str(model_dir / "total_direct_xgb.json"))

        # --- FINAL RESIDUAL MODELS ---
        has_market_all = (
            df["market_spread_home"].notna()
            & df["market_total"].notna()
            & df["spread_residual"].notna()
            & df["total_residual"].notna()
        )
        n_market_all = int(has_market_all.sum())

        if n_market_all >= cfg.min_market_rows:
            log.info("Training FINAL RESIDUAL models on %d rows with market data...", n_market_all)

            y_spread_resid_all = df.loc[has_market_all, "spread_residual"].astype(float)
            y_total_resid_all = df.loc[has_market_all, "total_residual"].astype(float)
            X_resid_all = X_all.loc[has_market_all]

            resid_dates = df.loc[has_market_all, "game_date_et"]
            resid_fit, resid_eval = temporal_eval_split(resid_dates)

            if resid_fit.sum() > 10 and resid_eval.sum() > 5:
                spread_resid_final = build_model(cfg, params_override=best_spread_params)
                total_resid_final = build_model(cfg, params_override=best_total_params)

                spread_resid_final.fit(
                    X_resid_all.iloc[resid_fit], y_spread_resid_all.iloc[resid_fit],
                    eval_set=[(X_resid_all.iloc[resid_eval], y_spread_resid_all.iloc[resid_eval])],
                    verbose=False,
                )
                log.info("Spread residual best iteration: %d", spread_resid_final.best_iteration)

                total_resid_final.fit(
                    X_resid_all.iloc[resid_fit], y_total_resid_all.iloc[resid_fit],
                    eval_set=[(X_resid_all.iloc[resid_eval], y_total_resid_all.iloc[resid_eval])],
                    verbose=False,
                )
                log.info("Total residual best iteration: %d", total_resid_final.best_iteration)

                spread_resid_final.save_model(str(model_dir / "spread_resid_xgb.json"))
                total_resid_final.save_model(str(model_dir / "total_resid_xgb.json"))
                log.info("Saved RESIDUAL models to %s", model_dir)
            else:
                log.warning("Not enough residual data for fit/eval split. Skipping residual models.")
        else:
            log.warning("Only %d rows have market data (need %d). Skipping residual models.",
                        n_market_all, cfg.min_market_rows)

        # Save schema artifacts
        (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
        (model_dir / "feature_medians.json").write_text(json.dumps(medians_all), encoding="utf-8")

        # Save Optuna best params for reproducibility
        if best_spread_params or best_total_params:
            optuna_results = {
                "spread_params": best_spread_params,
                "total_params": best_total_params,
            }
            (model_dir / "optuna_best_params.json").write_text(
                json.dumps(optuna_results, indent=2), encoding="utf-8"
            )
            log.info("Saved Optuna best params to %s", model_dir / "optuna_best_params.json")

        log.info("Saved models + feature schema to %s", model_dir)

if __name__ == "__main__":
    main()
