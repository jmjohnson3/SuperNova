import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor

log = logging.getLogger("nba_pipeline.modeling.train_player_prop_models")


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = Path(__file__).resolve().parent / "models" / "player_props"

    # basic quality filters
    min_prev_10: int = 3
    min_minutes_avg_10: float = 10.0

    # XGBoost params
    n_estimators: int = 900
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


SQL_PLAYER_TRAIN = """
SELECT *
FROM features.player_training_features
WHERE points IS NOT NULL
  AND rebounds IS NOT NULL
  AND assists IS NOT NULL
  AND n_games_prev_10 >= :min_prev_10
  AND (min_avg_10 IS NULL OR min_avg_10 >= :min_min_avg_10)
ORDER BY game_date_et, game_slug, player_id
"""


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def make_xy_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    X with NaNs preserved. Targets: points/rebounds/assists.
    """
    y_pts = df["points"].astype(float)
    y_reb = df["rebounds"].astype(float)
    y_ast = df["assists"].astype(float)

    drop_cols = {
        "game_slug",
        "game_date_et",
        "start_ts_utc",
        "points",
        "rebounds",
        "assists",
        "minutes",  # target-ish / postgame
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # categorical identifiers
    keep_cats = ["season", "team_abbr", "opponent_abbr", "is_home"]
    for c in keep_cats:
        if c in df.columns and c not in X.columns:
            X[c] = df[c]

    cat_cols = [c for c in ["season", "team_abbr", "opponent_abbr"] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # booleans -> int
    for bcol in ("is_home", "home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric_cols(X)

    # last-resort one-hot for any remaining non-numeric
    non_numeric_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if non_numeric_cols:
        log.warning("One-hot encoding remaining non-numeric cols: %s", non_numeric_cols)
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False, dummy_na=True)

    still_bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if still_bad:
        raise RuntimeError(f"Non-numeric columns remain after encoding: {still_bad[:20]}")

    return X, y_pts, y_reb, y_ast


def fit_fill_stats(X_train: pd.DataFrame) -> Dict[str, float]:
    meds = X_train.median(numeric_only=True)
    return {str(k): float(v) for k, v in meds.items()}


def apply_fill(X: pd.DataFrame, medians: Dict[str, float], columns: List[str]) -> pd.DataFrame:
    X2 = X.reindex(columns=columns)
    for c, m in medians.items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(m)
    return X2.fillna(0.0)


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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _eval(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    log.info("%s | MAE=%.3f RMSE=%.3f", name, mae, rmse)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = TrainConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(cfg.pg_dsn)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(SQL_PLAYER_TRAIN),
            conn,
            params={"min_prev_10": cfg.min_prev_10, "min_min_avg_10": cfg.min_minutes_avg_10},
        )

    if df.empty:
        raise RuntimeError("No rows returned from features.player_training_features after filters.")

    log.info("Loaded %d player-game rows", len(df))

    X_raw, y_pts, y_reb, y_ast = make_xy_raw(df)
    feature_cols = list(X_raw.columns)
    medians = fit_fill_stats(X_raw)
    X_all = apply_fill(X_raw, medians, feature_cols)

    # Train final models (simple first version)
    m_pts = build_model(cfg)
    m_reb = build_model(cfg)
    m_ast = build_model(cfg)

    log.info("Training PTS model...")
    m_pts.fit(X_all, y_pts)
    _eval("PTS (train)", y_pts.to_numpy(), m_pts.predict(X_all))

    log.info("Training REB model...")
    m_reb.fit(X_all, y_reb)
    _eval("REB (train)", y_reb.to_numpy(), m_reb.predict(X_all))

    log.info("Training AST model...")
    m_ast.fit(X_all, y_ast)
    _eval("AST (train)", y_ast.to_numpy(), m_ast.predict(X_all))

    m_pts.save_model(str(cfg.model_dir / "points_xgb.json"))
    m_reb.save_model(str(cfg.model_dir / "rebounds_xgb.json"))
    m_ast.save_model(str(cfg.model_dir / "assists_xgb.json"))

    (cfg.model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (cfg.model_dir / "feature_medians.json").write_text(json.dumps(medians), encoding="utf-8")

    log.info("Saved player prop models + schema to %s", cfg.model_dir)


if __name__ == "__main__":
    main()
