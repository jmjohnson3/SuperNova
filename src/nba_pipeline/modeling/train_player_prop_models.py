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
    n_estimators: int = 2000
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    # Regularization
    min_child_weight: int = 15
    gamma: float = 0.1
    reg_alpha: float = 0.5
    reg_lambda: float = 5.0
    # Early stopping
    early_stopping_rounds: int = 50
    # Huber slopes per stat
    huber_slope_pts: float = 4.0
    huber_slope_reb: float = 2.0
    huber_slope_ast: float = 2.0
    # Walk-forward
    run_walk_forward: bool = True
    min_train_days: int = 60
    test_window_days: int = 7
    step_days: int = 7


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


def _add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived interaction features from existing columns."""
    # Per-minute efficiency
    if "pts_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["pts_per_min_10"] = X["pts_avg_10"] / X["min_avg_10"].clip(lower=1.0)
    if "reb_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["reb_per_min_10"] = X["reb_avg_10"] / X["min_avg_10"].clip(lower=1.0)
    if "ast_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["ast_per_min_10"] = X["ast_avg_10"] / X["min_avg_10"].clip(lower=1.0)

    # Recent trend: 5-game vs 10-game (momentum)
    if "pts_avg_5" in X.columns and "pts_avg_10" in X.columns:
        X["pts_trend_5v10"] = X["pts_avg_5"] - X["pts_avg_10"]
    if "reb_avg_5" in X.columns and "reb_avg_10" in X.columns:
        X["reb_trend_5v10"] = X["reb_avg_5"] - X["reb_avg_10"]
    if "ast_avg_5" in X.columns and "ast_avg_10" in X.columns:
        X["ast_trend_5v10"] = X["ast_avg_5"] - X["ast_avg_10"]

    # Minutes trend (increasing/decreasing role)
    if "min_avg_5" in X.columns and "min_avg_10" in X.columns:
        X["min_trend_5v10"] = X["min_avg_5"] - X["min_avg_10"]

    # Coefficient of variation (consistency)
    if "pts_sd_10" in X.columns and "pts_avg_10" in X.columns:
        X["pts_cv_10"] = X["pts_sd_10"] / X["pts_avg_10"].clip(lower=0.5)
    if "reb_sd_10" in X.columns and "reb_avg_10" in X.columns:
        X["reb_cv_10"] = X["reb_sd_10"] / X["reb_avg_10"].clip(lower=0.5)
    if "ast_sd_10" in X.columns and "ast_avg_10" in X.columns:
        X["ast_cv_10"] = X["ast_sd_10"] / X["ast_avg_10"].clip(lower=0.5)

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
        "minutes",    # target-ish / postgame
        "player_id",  # arbitrary identifier, not ordinal — causes train/serve skew
        "player_name",
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

    # Derived interaction features
    X = _add_derived_features(X)

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


def build_model(cfg: TrainConfig, huber_slope: float = 4.0) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="reg:pseudohubererror",
        huber_slope=huber_slope,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        early_stopping_rounds=cfg.early_stopping_rounds,
        eval_metric="mae",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _eval(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    log.info("%s | MAE=%.3f RMSE=%.3f", name, mae, rmse)


def temporal_eval_split(
    dates: pd.Series, eval_frac: float = 0.15
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

    # Parse dates for walk-forward and temporal splits
    df["game_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce")

    X_raw, y_pts, y_reb, y_ast = make_xy_raw(df)
    feature_cols = list(X_raw.columns)

    stat_configs = [
        ("PTS", y_pts, cfg.huber_slope_pts),
        ("REB", y_reb, cfg.huber_slope_reb),
        ("AST", y_ast, cfg.huber_slope_ast),
    ]

    # --- Walk-forward validation ---
    if cfg.run_walk_forward:
        folds = walk_forward_folds(
            df,
            min_train_days=cfg.min_train_days,
            test_window_days=cfg.test_window_days,
            step_days=cfg.step_days,
        )

        if not folds:
            log.warning("No walk-forward folds produced. Skipping validation.")
        else:
            log.info(
                "Starting walk-forward eval | folds=%d min_train_days=%d test_window_days=%d step_days=%d",
                len(folds), cfg.min_train_days, cfg.test_window_days, cfg.step_days,
            )

            # Aggregates per stat
            agg: Dict[str, Tuple[List[float], List[float]]] = {
                name: ([], []) for name, _, _ in stat_configs
            }

            for k, (train_end, test_end) in enumerate(folds, start=1):
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

                # Temporal eval split within training for early stopping
                train_dates = df.loc[train_mask, "game_date_et"]
                fit_rel, eval_rel = temporal_eval_split(train_dates)
                X_fit = X_train.iloc[fit_rel]
                X_eval = X_train.iloc[eval_rel]

                fold_msgs = []
                for stat_name, y_full, huber_slope in stat_configs:
                    y_train = y_full.loc[train_mask]
                    y_test = y_full.loc[test_mask]

                    model = build_model(cfg, huber_slope=huber_slope)
                    model.fit(
                        X_fit, y_train.iloc[fit_rel],
                        eval_set=[(X_eval, y_train.iloc[eval_rel])],
                        verbose=False,
                    )
                    pred = model.predict(X_test)

                    mae = float(mean_absolute_error(y_test, pred))
                    rmse = _rmse(y_test.to_numpy(), pred)
                    fold_msgs.append(f"{stat_name} MAE={mae:.3f} RMSE={rmse:.3f}")

                    agg[stat_name][0].extend(y_test.to_list())
                    agg[stat_name][1].extend(pred.tolist())

                log.info(
                    "Fold %02d | train_end=%s test=[%s..%s) train=%d test=%d | %s",
                    k, train_end.date(), train_end.date(), test_end.date(),
                    n_train, n_test, " | ".join(fold_msgs),
                )

            # Overall walk-forward summary
            overall_msgs = []
            for stat_name, _, _ in stat_configs:
                true_all, pred_all = agg[stat_name]
                if true_all:
                    y_t = np.asarray(true_all, dtype=float)
                    y_p = np.asarray(pred_all, dtype=float)
                    mae = float(mean_absolute_error(y_t, y_p))
                    rmse = _rmse(y_t, y_p)
                    overall_msgs.append(f"{stat_name} MAE={mae:.3f} RMSE={rmse:.3f}")

            if overall_msgs:
                log.info("WALK-FORWARD OVERALL | rows=%d | %s",
                         len(agg["PTS"][0]), " | ".join(overall_msgs))

    # --- Train FINAL models on ALL rows (production) ---
    medians_all = fit_fill_stats(X_raw)
    X_all = apply_fill(X_raw, medians_all, feature_cols)

    # Temporal eval split for early stopping on final models
    all_dates = df["game_date_et"]
    fit_final, eval_final = temporal_eval_split(all_dates)
    X_fit_final = X_all.iloc[fit_final]
    X_eval_final = X_all.iloc[eval_final]

    models = {}
    save_names = {"PTS": "points_xgb.json", "REB": "rebounds_xgb.json", "AST": "assists_xgb.json"}

    for stat_name, y_full, huber_slope in stat_configs:
        log.info("Training FINAL %s model...", stat_name)
        m = build_model(cfg, huber_slope=huber_slope)
        m.fit(
            X_fit_final, y_full.iloc[fit_final],
            eval_set=[(X_eval_final, y_full.iloc[eval_final])],
            verbose=False,
        )
        log.info("%s best iteration: %d", stat_name, m.best_iteration)
        _eval(f"{stat_name} (train)", y_full.to_numpy(), m.predict(X_all))
        m.save_model(str(cfg.model_dir / save_names[stat_name]))
        models[stat_name] = m

    (cfg.model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (cfg.model_dir / "feature_medians.json").write_text(json.dumps(medians_all), encoding="utf-8")

    log.info("Saved player prop models + schema to %s", cfg.model_dir)


if __name__ == "__main__":
    main()
