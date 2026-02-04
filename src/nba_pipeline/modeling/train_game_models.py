# src/nba_pipeline/modeling/train_game_models.py
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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
    min_market_rows: int = 1
    run_walk_forward: bool = True
    min_train_days: int = 60
    test_window_days: int = 7
    step_days: int = 7
    # XGBoost params
    n_estimators: int = 700
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


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


def build_model(cfg: TrainConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=-1,
        reg_lambda=1.0,
    )


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
        "box_", "pbp_", "game_", "final_", "actual_", "result_",
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



# -------------------------
# Walk-forward evaluation
# -------------------------
def walk_forward_eval(
    cfg: TrainConfig,
    df: pd.DataFrame,
    X: pd.DataFrame,
    feature_cols: List[str],
    feature_medians: Dict[str, float],
) -> None:
    """
    Evaluate BOTH:
      - Direct models: predict margin/total_points
      - Residual models: predict (margin - market_spread_home), (total_points - market_total)
        then reconstruct using market lines

    For residual eval we only score games that have market_spread_home/market_total available.
    """
    min_date = df["game_date_et"].min()
    max_date = df["game_date_et"].max()

    min_train_end = min_date + timedelta(days=cfg.min_train_days)
    test_window = timedelta(days=cfg.test_window_days)
    step = timedelta(days=cfg.step_days)

    fold = 0

    # Aggregates
    direct_margin_true_all, direct_margin_pred_all = [], []
    direct_total_true_all, direct_total_pred_all = [], []

    resid_margin_true_all, resid_margin_pred_all = [], []
    resid_total_true_all, resid_total_pred_all = [], []

    # We'll walk by date boundaries
    train_end = min_train_end
    while train_end + test_window <= max_date + timedelta(days=1):
        test_start = train_end
        test_end = train_end + test_window

        train_mask = df["game_date_et"] < train_end
        test_mask = (df["game_date_et"] >= test_start) & (df["game_date_et"] < test_end)

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            train_end += step
            continue

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        # Targets
        y_margin_train = df.loc[train_mask, "margin"].astype(float)
        y_total_train = df.loc[train_mask, "total_points"].astype(float)

        y_margin_test = df.loc[test_mask, "margin"].astype(float)
        y_total_test = df.loc[test_mask, "total_points"].astype(float)

        # Direct models
        m_direct = build_model(cfg)
        t_direct = build_model(cfg)
        m_direct.fit(X_train, y_margin_train)
        t_direct.fit(X_train, y_total_train)
        pred_margin_direct = m_direct.predict(X_test)
        pred_total_direct = t_direct.predict(X_test)

        # record direct
        direct_margin_true_all.extend(y_margin_test.values.tolist())
        direct_margin_pred_all.extend(pred_margin_direct.tolist())
        direct_total_true_all.extend(y_total_test.values.tolist())
        direct_total_pred_all.extend(pred_total_direct.tolist())

        # Residual models: only rows that have market lines
        has_market = (
            df.loc[test_mask, "market_spread_home"].notna()
            & df.loc[test_mask, "market_total"].notna()
        )
        # Train residuals only on rows with market
        train_has_market = (
            df.loc[train_mask, "market_spread_home"].notna()
            & df.loc[train_mask, "market_total"].notna()
            & df.loc[train_mask, "spread_residual"].notna()
            & df.loc[train_mask, "total_residual"].notna()
        )

        resid_scored = int(has_market.sum())

        pred_margin_resid_final = None
        pred_total_resid_final = None

        if train_has_market.sum() >= 50 and resid_scored >= 10:
            y_spread_resid_train = df.loc[train_mask & train_has_market, "spread_residual"].astype(float)
            y_total_resid_train = df.loc[train_mask & train_has_market, "total_residual"].astype(float)

            # We must align X_train for the residual training subset
            X_train_resid = X.loc[train_mask & train_has_market]
            X_test_resid = X.loc[test_mask & has_market]

            m_resid = build_model(cfg)
            t_resid = build_model(cfg)
            m_resid.fit(X_train_resid, y_spread_resid_train)
            t_resid.fit(X_train_resid, y_total_resid_train)

            pred_spread_resid = m_resid.predict(X_test_resid)
            pred_total_resid = t_resid.predict(X_test_resid)

            market_spread = df.loc[test_mask & has_market, "market_spread_home"].astype(float).values
            market_total = df.loc[test_mask & has_market, "market_total"].astype(float).values

            pred_margin_resid_final = market_spread + pred_spread_resid
            pred_total_resid_final = market_total + pred_total_resid

            # record residual reconstructed preds
            resid_margin_true_all.extend(df.loc[test_mask & has_market, "margin"].astype(float).values.tolist())
            resid_margin_pred_all.extend(pred_margin_resid_final.tolist())
            resid_total_true_all.extend(df.loc[test_mask & has_market, "total_points"].astype(float).values.tolist())
            resid_total_pred_all.extend(pred_total_resid_final.tolist())

        # Fold logging (use residual if we have it, else direct)
        fold += 1

        # direct metrics
        mae_m = mean_absolute_error(y_margin_test, pred_margin_direct)
        rmse_m = _rmse(y_margin_test.values, pred_margin_direct)
        dir_m = _directional_accuracy_margin(y_margin_test.values, pred_margin_direct)
        slope_m, intercept_m = _calibration_slope_intercept(y_margin_test.values, pred_margin_direct)
        std_ratio_m = _std_ratio(y_margin_test.values, pred_margin_direct)

        mae_t = mean_absolute_error(y_total_test, pred_total_direct)
        rmse_t = _rmse(y_total_test.values, pred_total_direct)

        msg = (
            f"Fold {fold:02d} | train_end={train_end.date()} "
            f"test=[{test_start.date()}..{test_end.date()}) rows train={int(train_mask.sum())} test={int(test_mask.sum())} | "
            f"DIRECT SPREAD MAE={mae_m:.3f} RMSE={rmse_m:.3f} DIR={dir_m:.3f} std_ratio={std_ratio_m:.3f} slope={slope_m:.3f} | "
            f"DIRECT TOTAL MAE={mae_t:.3f} RMSE={rmse_t:.3f}"
        )

        if pred_margin_resid_final is not None:
            y_m_true = df.loc[test_mask & has_market, "margin"].astype(float).values
            y_t_true = df.loc[test_mask & has_market, "total_points"].astype(float).values

            mae_m_r = mean_absolute_error(y_m_true, pred_margin_resid_final)
            rmse_m_r = _rmse(y_m_true, pred_margin_resid_final)
            dir_m_r = _directional_accuracy_margin(y_m_true, pred_margin_resid_final)
            slope_m_r, _ = _calibration_slope_intercept(y_m_true, pred_margin_resid_final)
            std_ratio_m_r = _std_ratio(y_m_true, pred_margin_resid_final)

            mae_t_r = mean_absolute_error(y_t_true, pred_total_resid_final)
            rmse_t_r = _rmse(y_t_true, pred_total_resid_final)

            msg += (
                f" | RESID(recon) n={resid_scored} "
                f"SPREAD MAE={mae_m_r:.3f} RMSE={rmse_m_r:.3f} DIR={dir_m_r:.3f} std_ratio={std_ratio_m_r:.3f} slope={slope_m_r:.3f} | "
                f"TOTAL MAE={mae_t_r:.3f} RMSE={rmse_t_r:.3f}"
            )

        log.info(msg)

        train_end += step

    # Overall summary
    def _summ(name: str, y_true: List[float], y_pred: List[float]) -> str:
        y_true_a = np.asarray(y_true, dtype=float)
        y_pred_a = np.asarray(y_pred, dtype=float)
        mae = float(mean_absolute_error(y_true_a, y_pred_a))
        rmse = _rmse(y_true_a, y_pred_a)
        return f"{name} MAE={mae:.3f} RMSE={rmse:.3f}"

    if direct_margin_true_all:
        y_true = np.asarray(direct_margin_true_all, dtype=float)
        y_pred = np.asarray(direct_margin_pred_all, dtype=float)
        dir_acc = _directional_accuracy_margin(y_true, y_pred)
        slope, intercept = _calibration_slope_intercept(y_true, y_pred)
        std_ratio = _std_ratio(y_true, y_pred)
        log.info(
            "WALK-FORWARD DIRECT OVERALL | rows=%d | SPREAD %s DIR=%.3f std_ratio=%.3f slope=%.3f intercept=%.3f | TOTAL %s",
            len(y_true),
            _summ("SPREAD", direct_margin_true_all, direct_margin_pred_all),
            dir_acc,
            std_ratio,
            slope,
            intercept,
            _summ("TOTAL", direct_total_true_all, direct_total_pred_all),
        )

    if resid_margin_true_all:
        y_true = np.asarray(resid_margin_true_all, dtype=float)
        y_pred = np.asarray(resid_margin_pred_all, dtype=float)
        dir_acc = _directional_accuracy_margin(y_true, y_pred)
        slope, intercept = _calibration_slope_intercept(y_true, y_pred)
        std_ratio = _std_ratio(y_true, y_pred)
        log.info(
            "WALK-FORWARD RESID(recon) OVERALL | rows=%d | SPREAD %s DIR=%.3f std_ratio=%.3f slope=%.3f intercept=%.3f | TOTAL %s",
            len(y_true),
            _summ("SPREAD", resid_margin_true_all, resid_margin_pred_all),
            dir_acc,
            std_ratio,
            slope,
            intercept,
            _summ("TOTAL", resid_total_true_all, resid_total_pred_all),
        )

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

        spread_true_all: List[float] = []
        spread_pred_all: List[float] = []
        total_true_all: List[float] = []
        total_pred_all: List[float] = []

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

            spread_model = build_model(cfg)
            total_model = build_model(cfg)

            spread_model.fit(X_train, y_spread_train)
            total_model.fit(X_train, y_total_train)

            spread_pred = spread_model.predict(X_test)
            total_pred = total_model.predict(X_test)

            # Fold metrics
            s_mae, s_rmse = evaluate_regression(y_spread_test.to_numpy(), spread_pred)
            t_mae, t_rmse = evaluate_regression(y_total_test.to_numpy(), total_pred)
            s_dir = directional_accuracy_margin(y_spread_test.to_numpy(), spread_pred)
            cal = calibration_stats(y_spread_test.to_numpy(), spread_pred)

            log.info(
                "Fold %02d | train_end=%s test=[%s..%s) rows train=%d test=%d | "
                "DIRECT SPREAD MAE=%.3f RMSE=%.3f DIR=%.3f std_ratio=%.3f slope=%.3f | "
                "DIRECT TOTAL MAE=%.3f RMSE=%.3f",
                k,
                train_end.date(),
                train_end.date(),
                test_end.date(),
                n_train,
                n_test,
                s_mae,
                s_rmse,
                s_dir,
                cal["std_ratio"],
                cal["slope"],
                t_mae,
                t_rmse,
            )

            fold_rows += n_test
            spread_true_all.extend(y_spread_test.to_list())
            spread_pred_all.extend(spread_pred.tolist())
            total_true_all.extend(y_total_test.to_list())
            total_pred_all.extend(total_pred.tolist())

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

        # --- Train FINAL models on ALL rows (production) ---
        medians_all = fit_fill_stats(X_raw)
        X_all = apply_fill(X_raw, medians_all, feature_cols)

        spread_final = build_model(cfg)
        total_final = build_model(cfg)

        log.info("Fitting FINAL DIRECT spread model on all rows...")
        spread_final.fit(X_all, y_spread)

        log.info("Fitting FINAL DIRECT total model on all rows...")
        total_final.fit(X_all, y_total)

        spread_final.save_model(str(model_dir / "spread_direct_xgb.json"))
        total_final.save_model(str(model_dir / "total_direct_xgb.json"))

        (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
        (model_dir / "feature_medians.json").write_text(json.dumps(medians_all), encoding="utf-8")

        log.info("Saved models + feature schema to %s", model_dir)

if __name__ == "__main__":
    main()
