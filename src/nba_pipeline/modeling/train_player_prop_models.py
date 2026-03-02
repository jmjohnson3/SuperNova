import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor

from .features import add_player_prop_derived_features

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
    # Optuna tuning
    run_optuna: bool = True
    optuna_n_trials: int = 50
    optuna_n_folds: int = 5


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
        "minutes",    # target-ish / postgame
        "player_id",  # arbitrary identifier, not ordinal — causes train/serve skew
        "player_name",
    }

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    if "game_date_et" in df.columns:
        gdt = pd.to_datetime(df["game_date_et"])
        season_start_year = gdt.dt.year.where(gdt.dt.month >= 7, gdt.dt.year - 1)
        season_start = pd.to_datetime(season_start_year.astype(str) + "-10-01")
        X["season_days_elapsed"] = (gdt - season_start).dt.days.values

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

    # Derived interaction features — single source of truth in features.add_player_prop_derived_features.
    X = add_player_prop_derived_features(X)

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


def build_model(
    cfg: TrainConfig,
    huber_slope: float = 4.0,
    params_override: Optional[Dict] = None,
    n_estimators: Optional[int] = None,
    use_early_stopping: bool = True,
) -> XGBRegressor:
    """Build XGBRegressor with config defaults, optionally overridden by Optuna params.

    Set use_early_stopping=False for final production models: train to a fixed
    n_estimators derived from CV best_iteration statistics rather than using a
    held-out eval split, which triggers premature stopping due to distribution shift
    between earlier training data and the most-recent games used as the eval set.
    """
    p = dict(
        n_estimators=n_estimators if n_estimators is not None else cfg.n_estimators,
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
        eval_metric="mae",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    if use_early_stopping:
        p["early_stopping_rounds"] = cfg.early_stopping_rounds
    if params_override:
        # Per-stat Optuna params include tuned huber_slope — apply all directly.
        p.update(params_override)
    return XGBRegressor(**p)


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


def run_optuna_tuning_props(
    cfg: TrainConfig,
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_target: pd.Series,
    feature_cols: List[str],
    stat_name: str = "PTS",
) -> Dict:
    """
    Tune player-prop hyperparameters via walk-forward MAE on y_target.
    Runs a separate Optuna study per stat so REB/AST get their own optimal
    depth, regularization, and huber_slope rather than reusing PTS params.
    Returns full best_params including tuned huber_slope for this stat.
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
        log.warning("No walk-forward folds available for Optuna (%s). Using default params.", stat_name)
        return {}

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 15.0, log=True),
            "huber_slope":      trial.suggest_float("huber_slope", 1.0, 8.0),
        }
        mae_scores = []
        for train_end, test_end in tune_folds:
            train_mask = df["game_date_et"] < train_end
            test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)
            n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
            if n_train < 50 or n_test == 0:
                continue
            medians = fit_fill_stats(X_raw.loc[train_mask])
            X_tr = apply_fill(X_raw.loc[train_mask], medians, feature_cols)
            X_te = apply_fill(X_raw.loc[test_mask], medians, feature_cols)
            y_tr = y_target.loc[train_mask]
            y_te = y_target.loc[test_mask]
            train_dates = df.loc[train_mask, "game_date_et"]
            fit_rel, eval_rel = temporal_eval_split(train_dates)
            m = XGBRegressor(
                n_estimators=2000,
                objective="reg:pseudohubererror",
                early_stopping_rounds=50,
                eval_metric="mae",
                random_state=42,
                n_jobs=-1,
                **params,
            )
            m.fit(
                X_tr.iloc[fit_rel], y_tr.iloc[fit_rel],
                eval_set=[(X_tr.iloc[eval_rel], y_tr.iloc[eval_rel])],
                verbose=False,
            )
            mae_scores.append(float(mean_absolute_error(y_te, m.predict(X_te))))
        return float(np.mean(mae_scores)) if mae_scores else float("inf")

    log.info(
        "Running Optuna tuning for %s (%d trials, %d folds)...",
        stat_name, cfg.optuna_n_trials, len(tune_folds),
    )
    study = optuna.create_study(direction="minimize", study_name=f"player_props_{stat_name.lower()}")
    study.optimize(objective, n_trials=cfg.optuna_n_trials, show_progress_bar=False)
    best_params = study.best_params
    log.info("%s best params (MAE=%.3f): %s", stat_name, study.best_value, best_params)
    return best_params


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

    # --- Optuna hyperparameter tuning (separate study per stat) ---
    best_params_pts: Dict = {}
    best_params_reb: Dict = {}
    best_params_ast: Dict = {}
    if cfg.run_optuna:
        try:
            best_params_pts = run_optuna_tuning_props(cfg, df, X_raw, y_pts, feature_cols, "PTS")
            best_params_reb = run_optuna_tuning_props(cfg, df, X_raw, y_reb, feature_cols, "REB")
            best_params_ast = run_optuna_tuning_props(cfg, df, X_raw, y_ast, feature_cols, "AST")
        except ImportError:
            log.warning("optuna not installed. Skipping tuning. pip install optuna")
        except Exception as e:
            log.warning("Optuna tuning failed: %s. Using default params.", e)

    # (stat_name, y_target, huber_slope_default, best_params)
    stat_configs = [
        ("PTS", y_pts, cfg.huber_slope_pts, best_params_pts),
        ("REB", y_reb, cfg.huber_slope_reb, best_params_reb),
        ("AST", y_ast, cfg.huber_slope_ast, best_params_ast),
    ]

    # best_iters initialized here so final model section can reference it regardless
    # of whether walk-forward ran (e.g. run_walk_forward=False or no folds produced).
    best_iters: Dict[str, List[int]] = {name: [] for name, _, _, _ in stat_configs}

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

            # Aggregates per stat (best_iters defined above the walk-forward block)
            agg: Dict[str, Tuple[List[float], List[float]]] = {
                name: ([], []) for name, _, _, _ in stat_configs
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
                for stat_name, y_full, huber_slope, params_override in stat_configs:
                    y_train = y_full.loc[train_mask]
                    y_test = y_full.loc[test_mask]

                    model = build_model(cfg, huber_slope=huber_slope, params_override=params_override)
                    model.fit(
                        X_fit, y_train.iloc[fit_rel],
                        eval_set=[(X_eval, y_train.iloc[eval_rel])],
                        verbose=False,
                    )
                    if getattr(model, "best_iteration", None) is not None:
                        best_iters[stat_name].append(model.best_iteration)
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
            wf_mae: Dict[str, float] = {}
            for stat_name, _, _, _ in stat_configs:
                true_all, pred_all = agg[stat_name]
                if true_all:
                    y_t = np.asarray(true_all, dtype=float)
                    y_p = np.asarray(pred_all, dtype=float)
                    mae = float(mean_absolute_error(y_t, y_p))
                    rmse = _rmse(y_t, y_p)
                    overall_msgs.append(f"{stat_name} MAE={mae:.3f} RMSE={rmse:.3f}")
                    wf_mae[stat_name.lower()] = mae

            if overall_msgs:
                log.info("WALK-FORWARD OVERALL | rows=%d | %s",
                         len(agg["PTS"][0]), " | ".join(overall_msgs))

            # Save per-stat walk-forward MAEs for use as confidence intervals at predict time
            if wf_mae:
                mae_path = cfg.model_dir / "backtest_mae.json"
                mae_path.write_text(json.dumps(wf_mae, indent=2), encoding="utf-8")
                log.info("Saved walk-forward MAE to %s", mae_path)

    # --- Train FINAL models on ALL rows (production) ---
    medians_all = fit_fill_stats(X_raw)
    X_all = apply_fill(X_raw, medians_all, feature_cols)

    # Derive per-stat n_estimators from CV best_iteration statistics.
    # p75 × 1.2 buffer: final model trains on more data than any fold, so needs more trees.
    # p75 (not median) guards against folds where early stopping fired too aggressively.
    stat_n_est: Dict[str, int] = {}
    for stat_name, _, _, _ in stat_configs:
        iters = best_iters.get(stat_name, [])
        _iters = iters if iters else [cfg.n_estimators]
        n_est = max(int(np.percentile(_iters, 75) * 1.2), 100)
        stat_n_est[stat_name] = n_est
        log.info(
            "CV best_iteration %s | median=%d p75=%d → final n_estimators=%d",
            stat_name, int(np.median(_iters)), int(np.percentile(_iters, 75)), n_est,
        )

    save_names = {"PTS": "points_xgb.json", "REB": "rebounds_xgb.json", "AST": "assists_xgb.json"}

    for stat_name, y_full, huber_slope, params_override in stat_configs:
        n_est = stat_n_est[stat_name]
        log.info("Fitting FINAL %s model on all %d rows (n_estimators=%d, no early stopping)...",
                 stat_name, len(X_all), n_est)
        m = build_model(cfg, huber_slope=huber_slope, params_override=params_override,
                        n_estimators=n_est, use_early_stopping=False)
        m.fit(X_all, y_full, verbose=False)
        _eval(f"{stat_name} (train)", y_full.to_numpy(), m.predict(X_all))
        m.save_model(str(cfg.model_dir / save_names[stat_name]))

    (cfg.model_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (cfg.model_dir / "feature_medians.json").write_text(json.dumps(medians_all), encoding="utf-8")

    log.info("Saved player prop models + schema to %s", cfg.model_dir)


if __name__ == "__main__":
    main()
