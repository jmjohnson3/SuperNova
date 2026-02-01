import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

log = logging.getLogger("nba_pipeline.modeling.train_game_models")


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    # Walk-forward settings
    min_train_days: int = 60          # require at least this many days of history before first test window
    test_window_days: int = 7         # predict the next week
    step_days: int = 7                # slide forward by one week

    # XGBoost params
    n_estimators: int = 600
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
        raise RuntimeError(f"Some game_date_et values could not be parsed to datetime. Examples:\n{bad}")

    df = df.sort_values(["game_date_et", "game_slug"]).reset_index(drop=True)
    return df


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


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Convert object/string columns to numeric where possible; non-convertible -> NaN."""
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = TrainConfig()
    log.info(
        "Starting walk-forward eval | min_train_days=%d test_window_days=%d step_days=%d",
        cfg.min_train_days,
        cfg.test_window_days,
        cfg.step_days,
    )

    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = load_training_frame(conn)
        log.info("Loaded %d rows from game_training_features", len(df))

        X_raw, y_spread, y_total = make_xy_raw(df)
        all_cols = list(X_raw.columns)

        folds = walk_forward_folds(
            df,
            min_train_days=cfg.min_train_days,
            test_window_days=cfg.test_window_days,
            step_days=cfg.step_days,
        )
        if not folds:
            raise RuntimeError("No folds produced. Try reducing min_train_days or ensure data spans enough dates.")

        # Collect per-row predictions for overall metrics
        spread_true_all: List[float] = []
        spread_pred_all: List[float] = []
        total_true_all: List[float] = []
        total_pred_all: List[float] = []

        # Fold metrics (for logging)
        fold_rows = 0

        for k, (train_end, test_end) in enumerate(folds, start=1):
            train_mask = df["game_date_et"] < train_end
            test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())
            if n_train == 0 or n_test == 0:
                continue

            X_train_raw = X_raw.loc[train_mask]
            X_test_raw = X_raw.loc[test_mask]

            medians = fit_fill_stats(X_train_raw)
            X_train = apply_fill(X_train_raw, medians, all_cols)
            X_test = apply_fill(X_test_raw, medians, all_cols)

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
                "SPREAD MAE=%.3f RMSE=%.3f DIR=%.3f std_ratio=%.3f slope=%.3f | "
                "TOTAL MAE=%.3f RMSE=%.3f",
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

        # Overall metrics across all fold predictions
        spread_true_np = np.asarray(spread_true_all, dtype=float)
        spread_pred_np = np.asarray(spread_pred_all, dtype=float)
        total_true_np = np.asarray(total_true_all, dtype=float)
        total_pred_np = np.asarray(total_pred_all, dtype=float)

        s_mae, s_rmse = evaluate_regression(spread_true_np, spread_pred_np)
        t_mae, t_rmse = evaluate_regression(total_true_np, total_pred_np)
        s_dir = directional_accuracy_margin(spread_true_np, spread_pred_np)
        cal = calibration_stats(spread_true_np, spread_pred_np)

        log.info(
            "WALK-FORWARD OVERALL | rows=%d | SPREAD MAE=%.3f RMSE=%.3f DIR=%.3f "
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

        # ---- Train final models on ALL data (for use in predict_today) ----
        # IMPORTANT: fill NaNs using medians from *all* data since this is the production model.
        medians_all = fit_fill_stats(X_raw)
        X_all = apply_fill(X_raw, medians_all, all_cols)

        spread_final = build_model(cfg)
        total_final = build_model(cfg)

        log.info("Fitting FINAL spread model on all rows...")
        spread_final.fit(X_all, y_spread)

        log.info("Fitting FINAL total model on all rows...")
        total_final.fit(X_all, y_total)

        MODEL_DIR = Path(__file__).resolve().parent / "models"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        spread_final.save_model(str(MODEL_DIR / "spread_xgb.json"))
        total_final.save_model(str(MODEL_DIR / "total_xgb.json"))

        (MODEL_DIR / "feature_columns.json").write_text(json.dumps(all_cols), encoding="utf-8")
        (MODEL_DIR / "feature_medians.json").write_text(json.dumps(medians_all), encoding="utf-8")

        log.info("Saved models + feature schema to %s", MODEL_DIR)

        # Feature importances (best-effort)
        try:
            importances = pd.Series(spread_final.feature_importances_, index=all_cols).sort_values(ascending=False)
            log.info("Top spread features:\n%s", importances.head(25).to_string())
        except Exception:
            log.warning("Could not print feature importances", exc_info=True)


if __name__ == "__main__":
    main()
