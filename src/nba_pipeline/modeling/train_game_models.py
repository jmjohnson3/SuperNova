import logging
from dataclasses import dataclass
from typing import Tuple

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

    # --- A) time-based split as a fraction (last N% is test) ---
    test_frac: float = 0.20  # 20% most-recent games held out

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

    # Ensure game_date_et is datetime-like
    df["game_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce")
    if df["game_date_et"].isna().any():
        bad = df[df["game_date_et"].isna()].head(5)
        raise RuntimeError(f"Some game_date_et values could not be parsed to datetime. Examples:\n{bad}")

    # Sort for time split stability
    df = df.sort_values(["game_date_et", "game_slug"]).reset_index(drop=True)
    return df


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert object/string columns to numeric; non-convertible -> NaN.
    """
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue

        # Convert pandas string dtype or object to numeric when possible
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    A) Drop leaky columns (scores)
    B) Keep + one-hot season/home/away team identity
    C) Return numeric matrix with safe filling
    """
    # targets
    y_spread = df["margin"].astype(float)
    y_total = df["total_points"].astype(float)

    # ---- columns we never want as features ----
    # keep these for one-hot and then drop later:
    keep_cats = ["season", "home_team_abbr", "away_team_abbr"]

    drop_cols = {
        "game_slug",
        "game_date_et",
        "margin",
        "total_points",
        # leakage:
        "home_score",
        "away_score",
        # raw timestamp column (we'll derive hour/dow)
        # (we keep it temporarily if present so we can derive features)
    }

    # Build X from all columns except drop_cols
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # --- Derive timing features from start_ts_utc if present ---
    if "start_ts_utc" in X.columns:
        ts = pd.to_datetime(X["start_ts_utc"], errors="coerce", utc=True)
        X["start_hour_utc"] = ts.dt.hour
        X["start_dow_utc"] = ts.dt.dayofweek  # 0=Mon
        X = X.drop(columns=["start_ts_utc"])

    # --- Ensure b2b flags are numeric 0/1 ---
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # --- B) One-hot encode season + home/away teams ---
    # Ensure categorical cols exist before encoding
    for c in keep_cats:
        if c not in X.columns and c in df.columns:
            X[c] = df[c]

    cat_cols = [c for c in keep_cats if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # --- Coerce remaining object/string to numeric (non-convertible -> NaN) ---
    X = _coerce_numeric_cols(X)

    # --- Fill numeric NaNs with train-agnostic medians (global) ---
    numeric_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median(numeric_only=True))

    # --- If anything still non-numeric, one-hot encode it as last resort ---
    non_numeric_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if non_numeric_cols:
        log.warning("One-hot encoding remaining non-numeric cols: %s", non_numeric_cols)
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False, dummy_na=True)

    # final fill
    X = X.fillna(0.0)

    # Safety: all numeric
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


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    log.info("%s | MAE=%.4f RMSE=%.4f", name, mae, rmse)
    return mae, rmse


def baseline_mae(y_train: pd.Series, y_test: pd.Series) -> float:
    """
    Predict the training mean for every test row (dumb baseline).
    """
    mu = float(y_train.mean())
    return float(np.mean(np.abs(y_test - mu)))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = TrainConfig()
    log.info("Starting training. test_frac=%.2f", cfg.test_frac)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = load_training_frame(conn)
        log.info("Loaded %d rows from game_training_features", len(df))

        X, y_spread, y_total = make_xy(df)

        # --- A) time-based 80/20 split by sorted row order ---
        n = len(df)
        if not (0.0 < cfg.test_frac < 1.0):
            raise ValueError("test_frac must be between 0 and 1.")

        cut = int(n * (1.0 - cfg.test_frac))
        cut = max(1, min(cut, n - 1))  # ensure non-empty train and test

        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_spread_train, y_spread_test = y_spread.iloc[:cut], y_spread.iloc[cut:]
        y_total_train, y_total_test = y_total.iloc[:cut], y_total.iloc[cut:]

        log.info(
            "Split sizes | train=%d test=%d | features=%d",
            len(X_train),
            len(X_test),
            X.shape[1],
        )

        # Baselines
        spread_base = baseline_mae(y_spread_train, y_spread_test)
        total_base = baseline_mae(y_total_train, y_total_test)
        log.info("Baseline (predict-train-mean) | SPREAD_MAE=%.4f TOTAL_MAE=%.4f", spread_base, total_base)

        spread_model = build_model(cfg)
        total_model = build_model(cfg)

        log.info("Fitting spread model...")
        spread_model.fit(X_train, y_spread_train)

        log.info("Fitting total model...")
        total_model.fit(X_train, y_total_train)

        from pathlib import Path
        import json

        MODEL_DIR = Path(__file__).resolve().parent / "models"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        spread_model.save_model(str(MODEL_DIR / "spread_xgb.json"))
        total_model.save_model(str(MODEL_DIR / "total_xgb.json"))

        (MODEL_DIR / "feature_columns.json").write_text(
            json.dumps(list(X.columns)),
            encoding="utf-8",
        )
        log.info("Saved models + feature columns to %s", MODEL_DIR)

        spread_preds = spread_model.predict(X_test)
        total_preds = total_model.predict(X_test)

        evaluate("SPREAD", y_spread_test, spread_preds)
        evaluate("TOTAL", y_total_test, total_preds)

        # Feature importances (best-effort)
        try:
            importances = pd.Series(spread_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            log.info("Top spread features:\n%s", importances.head(25).to_string())
        except Exception:
            log.warning("Could not print feature importances", exc_info=True)


if __name__ == "__main__":
    main()
