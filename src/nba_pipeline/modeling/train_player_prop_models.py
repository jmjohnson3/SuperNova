import datetime
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

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

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
    optuna_max_age_days: int = 7  # skip re-tune if cached params are fresher than this
    # Minutes stacking
    use_minutes_stacking: bool = True
    run_multioutput_test: bool = True


SQL_PLAYER_TRAIN = """
SELECT ptf.*, pg.minutes::float AS minutes
FROM features.player_training_features ptf
JOIN raw.nba_player_gamelogs pg
  ON pg.player_id = ptf.player_id
 AND pg.game_slug = ptf.game_slug
WHERE ptf.points IS NOT NULL
  AND ptf.rebounds IS NOT NULL
  AND ptf.assists IS NOT NULL
  AND ptf.n_games_prev_10 >= :min_prev_10
  AND (ptf.min_avg_10 IS NULL OR ptf.min_avg_10 >= :min_min_avg_10)
  AND pg.minutes IS NOT NULL
ORDER BY ptf.game_date_et, ptf.game_slug, ptf.player_id
"""


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _load_player_positions(engine) -> dict:
    """Returns {player_id: most_common_position} from box score history.
    MODE() handles versatile players; only the 5 main positions returned."""
    sql = text("""
        SELECT player_id, MODE() WITHIN GROUP (ORDER BY position) AS primary_position
        FROM raw.nba_boxscore_player_stats
        WHERE position IN ('PG', 'SG', 'SF', 'PF', 'C')
        GROUP BY player_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return {r[0]: r[1] for r in rows}


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

    # team_abbr and opponent_abbr OHE removed — causes overfitting
    # (e.g. team_abbr_HOU, opponent_abbr_OKL ranked top-10 in assists model)
    # primary_position is fine: only 5 stable semantic values, no team leakage
    cat_cols = [c for c in ["season", "primary_position"] if c in X.columns]
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


def build_minutes_model(
    cfg: TrainConfig,
    n_estimators: Optional[int] = None,
    use_early_stopping: bool = True,
) -> XGBRegressor:
    """Minutes model uses squared-error objective.

    pseudohubererror cannot be used for minutes: its auto base_score
    initialises to ~198 (far from the true mean of ~25), which saturates
    gradients on the first iteration and causes early-stopping to fire
    immediately (best_iteration=0), yielding constant predictions.
    """
    p = dict(
        n_estimators=n_estimators if n_estimators is not None else cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="reg:squarederror",
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        eval_metric="rmse",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    if use_early_stopping:
        p["early_stopping_rounds"] = cfg.early_stopping_rounds
    return XGBRegressor(**p)


def build_model(
    cfg: TrainConfig,
    huber_slope: float = 4.0,  # kept for backward compat, ignored (objective=absoluteerror)
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
        objective="reg:absoluteerror",  # MAE = predicts median, avoids mean-bias vs book lines
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
        # Strip huber_slope from legacy Optuna params (no longer used with reg:absoluteerror)
        filtered = {k: v for k, v in params_override.items() if k != "huber_slope"}
        p.update(filtered)
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
                objective="reg:absoluteerror",  # MAE = predicts median
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

    # Item 10: primary_position from box score history
    positions = _load_player_positions(engine)
    df["primary_position"] = df["player_id"].map(positions).fillna("UNK")
    log.info(
        "Position coverage: %d/%d players mapped (%.1f%%)",
        int((df["primary_position"] != "UNK").sum()), len(df),
        100.0 * (df["primary_position"] != "UNK").mean(),
    )

    X_raw, y_pts, y_reb, y_ast = make_xy_raw(df)
    y_min = df["minutes"].astype(float)
    feature_cols_base = list(X_raw.columns)

    # --- Minutes model: walk-forward OOF stacking features ---
    fallback_min = df["min_avg_10"].fillna(df["min_avg_5"]).fillna(20.0).values
    min_best_iters: List[int] = []
    if cfg.use_minutes_stacking:
        # A minutes model is trained on each fold's history; its out-of-fold (OOF)
        # predictions are injected as features into PTS/REB/AST models.
        # This is leakage-free: each fold uses only prior data to predict minutes.
        oof_minutes = np.full(len(df), np.nan)
        min_folds = walk_forward_folds(
            df, min_train_days=cfg.min_train_days,
            test_window_days=cfg.test_window_days, step_days=cfg.step_days,
        )
        log.info("Minutes stacking: running %d OOF folds...", len(min_folds))
        for train_end, test_end in min_folds:
            train_mask_m = df["game_date_et"] < train_end
            test_mask_m = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)
            n_tr, n_te = int(train_mask_m.sum()), int(test_mask_m.sum())
            if n_tr < 50 or n_te == 0:
                continue
            med_m = fit_fill_stats(X_raw.loc[train_mask_m])
            X_tr_m = apply_fill(X_raw.loc[train_mask_m], med_m, feature_cols_base)
            X_te_m = apply_fill(X_raw.loc[test_mask_m], med_m, feature_cols_base)
            tr_dates_m = df.loc[train_mask_m, "game_date_et"]
            fit_m, eval_m = temporal_eval_split(tr_dates_m)
            m_min = build_minutes_model(cfg)
            m_min.fit(
                X_tr_m.iloc[fit_m], y_min.loc[train_mask_m].iloc[fit_m],
                eval_set=[(X_tr_m.iloc[eval_m], y_min.loc[train_mask_m].iloc[eval_m])],
                verbose=False,
            )
            if getattr(m_min, "best_iteration", None) is not None:
                min_best_iters.append(m_min.best_iteration)
            oof_minutes[np.where(test_mask_m)[0]] = m_min.predict(X_te_m)

        oof_covered = ~np.isnan(oof_minutes)
        if oof_covered.any():
            min_oof_mae = float(mean_absolute_error(y_min.values[oof_covered], oof_minutes[oof_covered]))
            log.info("Minutes OOF | coverage=%.1f%% MAE=%.3f", oof_covered.mean() * 100, min_oof_mae)

        oof_filled = np.clip(
            np.where(np.isnan(oof_minutes), fallback_min, oof_minutes), 0.0, 48.0
        ).astype(float)
        X_aug = X_raw.copy()
        X_aug["pred_minutes"] = oof_filled
        X_aug["pred_min_vs_avg"] = oof_filled - fallback_min
        feature_cols = list(X_aug.columns)
        log.info(
            "Feature matrix: %d base + 2 minutes stacking = %d total",
            len(feature_cols_base), len(feature_cols),
        )
    else:
        X_aug = X_raw
        feature_cols = feature_cols_base
        log.info("Minutes stacking disabled. Feature matrix: %d base features.", len(feature_cols))

    # --- Optuna hyperparameter tuning (separate study per stat) ---
    best_params_pts: Dict = {}
    best_params_reb: Dict = {}
    best_params_ast: Dict = {}
    _optuna_cache_path = cfg.model_dir / "optuna_best_params_props.json"

    def _optuna_cache_is_fresh() -> bool:
        if not _optuna_cache_path.exists():
            return False
        age_s = datetime.datetime.now().timestamp() - _optuna_cache_path.stat().st_mtime
        return age_s < cfg.optuna_max_age_days * 86400

    if cfg.run_optuna:
        if _optuna_cache_is_fresh():
            try:
                cached = json.loads(_optuna_cache_path.read_text())
                best_params_pts = cached.get("pts", {})
                best_params_reb = cached.get("reb", {})
                best_params_ast = cached.get("ast", {})
                cache_age_h = (_optuna_cache_path.stat().st_mtime)
                cache_age_h = (datetime.datetime.now().timestamp() - cache_age_h) / 3600
                log.info(
                    "Loaded cached Optuna params (%.1fh old, max_age=%dd) — skipping re-tune.",
                    cache_age_h, cfg.optuna_max_age_days,
                )
            except Exception as e:
                log.warning("Failed to load cached Optuna params: %s. Re-running tuning.", e)
                best_params_pts = best_params_reb = best_params_ast = {}

        if not best_params_pts:
            try:
                best_params_pts = run_optuna_tuning_props(cfg, df, X_aug, y_pts, feature_cols, "PTS")
                best_params_reb = run_optuna_tuning_props(cfg, df, X_aug, y_reb, feature_cols, "REB")
                best_params_ast = run_optuna_tuning_props(cfg, df, X_aug, y_ast, feature_cols, "AST")
                # Persist so the next N days can skip re-tuning
                _optuna_cache_path.write_text(json.dumps({
                    "pts": best_params_pts,
                    "reb": best_params_reb,
                    "ast": best_params_ast,
                }, indent=2))
                log.info("Saved Optuna params to %s", _optuna_cache_path)
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
    # calib/wf_mae/multi_mae initialized here so residual training can access them after walk-forward block.
    calib: Dict = {}
    wf_mae: Dict = {}
    multi_mae: Dict[str, float] = {}

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
            # Item 8: Track OOF predictions on market-data rows for quality gate
            market_oof_direct:   Dict[str, List] = {name: [] for name, _, _, _ in stat_configs}
            market_oof_baseline: Dict[str, List] = {name: [] for name, _, _, _ in stat_configs}

            for k, (train_end, test_end) in enumerate(folds, start=1):
                train_mask = df["game_date_et"] < train_end
                test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

                n_train = int(train_mask.sum())
                n_test = int(test_mask.sum())
                if n_train < 50 or n_test == 0:
                    continue

                X_train_raw = X_aug.loc[train_mask]
                X_test_raw = X_aug.loc[test_mask]

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

                    # Item 8: Track market-data rows for residual quality gate
                    _bl_col = {"PTS": "prev_book_line_pts", "REB": "prev_book_line_reb",
                               "AST": "prev_book_line_ast"}[stat_name]
                    if _bl_col in df.columns:
                        _test_bl = df.loc[test_mask, _bl_col].values
                        _mkt = ~pd.isna(_test_bl)
                        if _mkt.any():
                            market_oof_direct[stat_name].extend(
                                zip(y_test.values[_mkt], pred[_mkt]))
                            market_oof_baseline[stat_name].extend(
                                zip(y_test.values[_mkt], _test_bl[_mkt].astype(float)))

                log.info(
                    "Fold %02d | train_end=%s test=[%s..%s) train=%d test=%d | %s",
                    k, train_end.date(), train_end.date(), test_end.date(),
                    n_train, n_test, " | ".join(fold_msgs),
                )

            # Overall walk-forward summary + calibration quantiles
            overall_msgs = []
            wf_mae: Dict[str, float] = {}
            calib: Dict[str, float] = {}
            for stat_name, _, _, _ in stat_configs:
                true_all, pred_all = agg[stat_name]
                if true_all:
                    y_t = np.asarray(true_all, dtype=float)
                    y_p = np.asarray(pred_all, dtype=float)
                    mae = float(mean_absolute_error(y_t, y_p))
                    rmse = _rmse(y_t, y_p)
                    abs_errs = np.abs(y_t - y_p)
                    p68 = float(np.percentile(abs_errs, 68))
                    p90 = float(np.percentile(abs_errs, 90))
                    overall_msgs.append(
                        f"{stat_name} MAE={mae:.3f} RMSE={rmse:.3f} p68={p68:.3f} p90={p90:.3f}"
                    )
                    wf_mae[stat_name.lower()] = mae
                    calib[f"ci_p68_{stat_name.lower()}"] = p68
                    calib[f"ci_p90_{stat_name.lower()}"] = p90

            if overall_msgs:
                log.info("WALK-FORWARD OVERALL | rows=%d | %s",
                         len(agg["PTS"][0]), " | ".join(overall_msgs))

            # Item 8: Market subset MAE (direct model vs. book line baseline)
            for stat_name, _, _, _ in stat_configs:
                sl = stat_name.lower()
                if market_oof_direct[stat_name]:
                    acts_d, preds_d = zip(*market_oof_direct[stat_name])
                    acts_b, bls_b   = zip(*market_oof_baseline[stat_name])
                    direct_mkt_mae = float(mean_absolute_error(list(acts_d), list(preds_d)))
                    book_line_mae  = float(mean_absolute_error(list(acts_b), list(bls_b)))
                    log.info(
                        "MARKET SUBSET %s | direct_mae=%.3f  book_line_mae=%.3f",
                        stat_name, direct_mkt_mae, book_line_mae,
                    )
                    calib[f"direct_market_mae_{sl}"] = round(direct_mkt_mae, 4)
                    calib[f"book_line_mae_{sl}"]     = round(book_line_mae, 4)

            # --- Item 11: Multi-output LGB walk-forward test ---
            # LGB 4.x Dataset API enforces 1D labels, so we use sklearn's MultiOutputRegressor
            # wrapping LGBMRegressor.  Each stat gets an independent model, but the wrapper
            # trains them in parallel and exposes a single .predict() → (n, 3) interface,
            # which plugs naturally into the 3-way ensemble at inference.
            if cfg.run_multioutput_test and _HAS_LGB and folds:
                from sklearn.multioutput import MultiOutputRegressor
                log.info("Running multi-output LGB walk-forward test (MultiOutputRegressor)...")
                mo_agg: Dict[str, Tuple[List[float], List[float]]] = {
                    "PTS": ([], []), "REB": ([], []), "AST": ([], [])
                }
                for train_end, test_end in folds:
                    train_mask = df["game_date_et"] < train_end
                    test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)
                    if int(train_mask.sum()) < 50 or int(test_mask.sum()) == 0:
                        continue
                    medians_mo = fit_fill_stats(X_aug.loc[train_mask])
                    X_tr_mo = apply_fill(X_aug.loc[train_mask], medians_mo, feature_cols)
                    X_te_mo = apply_fill(X_aug.loc[test_mask], medians_mo, feature_cols)
                    y_tr_mo = np.column_stack([
                        y_pts.loc[train_mask].values,
                        y_reb.loc[train_mask].values,
                        y_ast.loc[train_mask].values,
                    ])
                    try:
                        m_mo = MultiOutputRegressor(lgb.LGBMRegressor(
                            objective="regression_l1",
                            metric="mae",
                            num_leaves=63,
                            learning_rate=0.05,
                            n_estimators=300,
                            min_child_samples=20,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=0.1,
                            reg_lambda=1.0,
                            verbose=-1,
                            n_jobs=1,
                            random_state=cfg.random_state,
                        ), n_jobs=-1)
                        m_mo.fit(X_tr_mo, y_tr_mo)
                        pred_mo = m_mo.predict(X_te_mo)  # shape (n_test, 3)
                        for i, (stat_name, y_stat) in enumerate([
                            ("PTS", y_pts), ("REB", y_reb), ("AST", y_ast)
                        ]):
                            y_te_stat = y_stat.loc[test_mask]
                            mo_agg[stat_name][0].extend(y_te_stat.tolist())
                            mo_agg[stat_name][1].extend(pred_mo[:, i].tolist())
                    except Exception as exc:
                        log.warning("Multi-output LGB fold failed: %s", exc)
                        break

                mo_msgs = []
                for stat_name, y_stat in [("PTS", y_pts), ("REB", y_reb), ("AST", y_ast)]:
                    true_all, pred_all = mo_agg[stat_name]
                    if true_all:
                        mo_mae_val = float(mean_absolute_error(true_all, pred_all))
                        multi_mae[stat_name.lower()] = mo_mae_val
                        existing = wf_mae.get(stat_name.lower(), float("inf"))
                        delta = mo_mae_val - existing
                        mo_msgs.append(
                            f"{stat_name}: multi={mo_mae_val:.3f} xgb={existing:.3f} delta={delta:+.3f}"
                        )
                if mo_msgs:
                    log.info("MULTI-OUTPUT LGB COMPARISON | %s", " | ".join(mo_msgs))

                for sl in ("pts", "reb", "ast"):
                    if sl in multi_mae:
                        calib[f"multi_output_mae_{sl}"] = round(multi_mae[sl], 4)

                # Gate: use multi-output ensemble if it doesn't regress on any stat by >1%
                use_multi = bool(multi_mae) and all(
                    multi_mae.get(sl, float("inf")) <= wf_mae.get(sl, float("inf")) * 1.01
                    for sl in ["pts", "reb", "ast"]
                )
                calib["use_multioutput_ensemble"] = use_multi
                log.info("Multi-output ensemble gate: %s", use_multi)

            # Save per-stat walk-forward MAEs + calibrated CI quantiles
            if wf_mae:
                payload = {**wf_mae, **calib}
                mae_path = cfg.model_dir / "backtest_mae.json"
                mae_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                log.info("Saved walk-forward MAE + CI calibration to %s", mae_path)

    # --- Train FINAL models on ALL rows (production) ---
    if cfg.use_minutes_stacking:
        # Step 1: Fit final minutes model on base features (no minutes stacking — it's the base).
        medians_all_base = fit_fill_stats(X_raw)
        X_all_base = apply_fill(X_raw, medians_all_base, feature_cols_base)
        _min_n_est = max(int(np.percentile(min_best_iters, 75) * 1.2), 100) if min_best_iters else cfg.n_estimators
        log.info(
            "Fitting FINAL MIN model (%d rows, n_estimators=%d, no early stopping)...",
            len(X_all_base), _min_n_est,
        )
        m_min_final = build_minutes_model(cfg, n_estimators=_min_n_est, use_early_stopping=False)
        m_min_final.fit(X_all_base, y_min, verbose=False)
        _eval("MIN (train)", y_min.to_numpy(), m_min_final.predict(X_all_base))
        m_min_final.save_model(str(cfg.model_dir / "minutes_xgb.json"))

        # Step 2: Augment with final minutes predictions for PTS/REB/AST training.
        pred_min_all = np.clip(m_min_final.predict(X_all_base), 0.0, 48.0)
        X_aug_final = X_raw.copy()
        X_aug_final["pred_minutes"] = pred_min_all
        X_aug_final["pred_min_vs_avg"] = pred_min_all - fallback_min
        medians_all = fit_fill_stats(X_aug_final)
        X_all = apply_fill(X_aug_final, medians_all, feature_cols)
    else:
        medians_all_base = fit_fill_stats(X_raw)
        X_aug_final = X_raw
        medians_all = medians_all_base
        X_all = apply_fill(X_aug_final, medians_all, feature_cols)

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
    if cfg.use_minutes_stacking:
        (cfg.model_dir / "feature_columns_base.json").write_text(json.dumps(feature_cols_base), encoding="utf-8")
        (cfg.model_dir / "feature_medians_base.json").write_text(json.dumps(medians_all_base), encoding="utf-8")

    log.info("Saved player prop models + schema to %s", cfg.model_dir)

    # --- LightGBM ensemble (if available) ---
    # Train a parallel LGB model on the same final X_all data.  At inference,
    # predictions are averaged 50/50 with XGBoost, reducing variance and bias.
    if _HAS_LGB:
        log.info("Training LightGBM ensemble models on %d rows...", len(X_all))
        lgb_save_names = {"PTS": "lgb_points.txt", "REB": "lgb_rebounds.txt", "AST": "lgb_assists.txt"}
        lgb_trained = 0
        for stat_name, y_full, _, _ in stat_configs:
            try:
                lgb_params = dict(
                    objective="regression_l1",  # MAE = predicts median, matches XGB objective
                    metric="mae",
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=2000,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbose=-1,
                    n_jobs=-1,
                    random_state=cfg.random_state,
                )
                m_lgb = lgb.LGBMRegressor(**lgb_params)
                m_lgb.fit(X_all, y_full)
                _eval(f"{stat_name} LGB (train)", y_full.to_numpy(), m_lgb.predict(X_all))
                m_lgb.booster_.save_model(str(cfg.model_dir / lgb_save_names[stat_name]))
                lgb_trained += 1
                log.info("Saved LGB %s model to %s", stat_name, lgb_save_names[stat_name])
            except Exception as exc:
                log.warning("LightGBM training failed for %s: %s", stat_name, exc)
        if lgb_trained == 3:
            log.info("LightGBM ensemble complete — XGB+LGB predictions will be averaged at inference")
    else:
        log.info("lightgbm not installed — skipping LGB ensemble (pip install lightgbm)")

    # --- Item 11: Final multi-output LGB model (MultiOutputRegressor, saved via joblib) ---
    if cfg.run_multioutput_test and _HAS_LGB and multi_mae:
        from sklearn.multioutput import MultiOutputRegressor
        import joblib
        y_all_stacked = np.column_stack([y_pts.values, y_reb.values, y_ast.values])
        try:
            m_mo_final = MultiOutputRegressor(lgb.LGBMRegressor(
                objective="regression_l1",
                metric="mae",
                num_leaves=63,
                learning_rate=0.05,
                n_estimators=300,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbose=-1,
                n_jobs=1,
                random_state=cfg.random_state,
            ), n_jobs=-1)
            m_mo_final.fit(X_all, y_all_stacked)
            joblib.dump(m_mo_final, str(cfg.model_dir / "lgb_multi.pkl"))
            log.info("Saved multi-output LGB model → lgb_multi.pkl")
        except Exception as exc:
            log.warning("Final multi-output LGB failed: %s", exc)

    # --- Item 8: Book line residual prop models ---
    # Train a residual model that predicts (actual − book_line) on market rows.
    # At inference: pred = book_line + resid_model(X) when a book line is available.
    book_line_cols   = {"PTS": "prev_book_line_pts", "REB": "prev_book_line_reb", "AST": "prev_book_line_ast"}
    resid_save_names = {"PTS": "prop_resid_pts_xgb.json", "REB": "prop_resid_reb_xgb.json",
                        "AST": "prop_resid_ast_xgb.json"}
    resid_quality: Dict[str, bool] = {}

    for stat_name, y_full, huber_slope, params_override in stat_configs:
        sl = stat_name.lower()
        bl_col = book_line_cols[stat_name]
        if bl_col not in df.columns:
            log.warning("RESID %s: book line column %s not in df, skipping.", stat_name, bl_col)
            resid_quality[stat_name] = False
            continue

        market_mask = df[bl_col].notna()
        n_mkt = int(market_mask.sum())
        if n_mkt < 1000:
            log.warning("RESID %s: only %d market rows (<1000), skipping.", stat_name, n_mkt)
            resid_quality[stat_name] = False
            continue

        # Quality gate: skip if direct model already beats book-line baseline by >3%
        direct_mkt = calib.get(f"direct_market_mae_{sl}")
        book_mkt   = calib.get(f"book_line_mae_{sl}")
        if direct_mkt is not None and book_mkt is not None and direct_mkt <= book_mkt * 0.97:
            log.info(
                "RESID %s: direct already beats book line by >3%% (%.3f vs %.3f). Gated out.",
                stat_name, direct_mkt, book_mkt,
            )
            resid_quality[stat_name] = False
            continue

        y_dev = (y_full - df[bl_col].astype(float)).loc[market_mask]
        X_mkt = apply_fill(X_aug_final.loc[market_mask], medians_all, feature_cols)

        n_est = stat_n_est[stat_name]
        log.info(
            "Fitting FINAL RESID %s on %d market rows (n_estimators=%d)...",
            stat_name, n_mkt, n_est,
        )
        m_resid = build_model(cfg, huber_slope=huber_slope, params_override=params_override,
                              n_estimators=n_est, use_early_stopping=False)
        m_resid.fit(X_mkt, y_dev, verbose=False)
        m_resid.save_model(str(cfg.model_dir / resid_save_names[stat_name]))
        resid_quality[stat_name] = True
        log.info("Saved RESID %s → %s", stat_name, resid_save_names[stat_name])

    calib["resid_quality"] = resid_quality
    log.info("Residual quality: %s", resid_quality)

    # Re-save backtest_mae.json with resid_quality appended
    if wf_mae or calib:
        mae_path = cfg.model_dir / "backtest_mae.json"
        payload = {**wf_mae, **calib}
        mae_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Updated backtest_mae.json with resid_quality")


if __name__ == "__main__":
    main()
