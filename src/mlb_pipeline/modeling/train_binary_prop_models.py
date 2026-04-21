# src/mlb_pipeline/modeling/train_binary_prop_models.py
"""
Train MLB player prop binary classifiers: P(actual > book_line).

Root-cause fix for OVER prediction problems: instead of regressing on E[stat]
and comparing to a book line, train directly on the betting question:
"Does this player exceed the offered line?"

book_line is a required feature — the model learns context like
"hits_avg_10=1.3 with a line of 0.5 is very different from a line of 1.5".

Trained on rows where a DraftKings prop line is available for that game date.
After 2025 backfill (Mar–Jun data) + 2026 live season, expect ~25k rows for hits.

Walk-forward CV: min_train_days=60, test_window=14d, step=14d.
Reports Brier score (primary) + AUC (secondary).

Artifacts saved to models/player_props/:
  hits_clf_xgb.json,         lgb_hits_clf.txt
  total_bases_clf_xgb.json,  lgb_total_bases_clf.txt
  home_runs_clf_xgb.json,    lgb_home_runs_clf.txt
  walks_clf_xgb.json,        lgb_walks_clf.txt
  strikeouts_clf_xgb.json,   lgb_strikeouts_clf.txt
  feature_columns_clf_batters.json,  feature_medians_clf_batters.json
  feature_columns_clf_pitchers.json, feature_medians_clf_pitchers.json
  backtest_clf.json
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features
from .train_player_prop_models import (
    SQL_BATTER_TRAIN,
    SQL_PITCHER_TRAIN,
    _BATTER_META,
    _BATTER_TARGETS,
    _PITCHER_META,
    _PITCHER_TARGETS,
    _coerce_numeric,
    fit_medians,
    apply_medians,
)

log = logging.getLogger("mlb_pipeline.modeling.train_binary_prop_models")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"

# Map training-data column names → (Odds API stat name, preferred bookmaker)
# HR uses FanDuel — DraftKings only offered it ~281 times vs FD's 22k rows.
_BATTER_STAT_MAP = {
    "hits":         ("batter_hits",        "draftkings"),
    "total_bases":  ("batter_total_bases", "draftkings"),
    "home_runs":    ("batter_home_runs",   "fanduel"),
    "walks_batter": ("batter_walks",       "draftkings"),
}
_PITCHER_STAT_MAP = {
    "strikeouts": ("pitcher_strikeouts", "draftkings"),
}

# Break-even probability at standard -110 juice
_BREAKEVEN_PROB = 0.524

_SQL_PLAYER_NAMES = """
SELECT DISTINCT player_id,
       lower(regexp_replace(
           encode(convert_to(
               trim(first_name || ' ' || last_name), 'LATIN1'
           ), 'escape'),
           '[^a-z ]', '', 'g'
       )) AS name_norm
FROM raw.mlb_boxscore_player_stats
WHERE first_name IS NOT NULL AND last_name IS NOT NULL
"""

_SQL_PROP_LINES = """
SELECT as_of_date, player_name_norm, stat, bookmaker_key, AVG(line) AS line
FROM odds.mlb_player_prop_lines
WHERE bookmaker_key = ANY(%(bookmakers)s)
  AND stat = ANY(%(stats)s)
  AND line IS NOT NULL
GROUP BY as_of_date, player_name_norm, stat, bookmaker_key
"""


@dataclass(frozen=True)
class ClfConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR

    # Walk-forward — shorter window than regression (binary has less data initially)
    min_train_days: int = 60
    test_window_days: int = 14
    step_days: int = 14

    # XGBoost defaults
    n_estimators: int = 1000
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    gamma: float = 0.1
    reg_alpha: float = 0.5
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 50
    random_state: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization (mirrors parse_oddsapi._normalize_name)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    """Strip accents, remove non-alpha, lowercase — same as Odds API parser."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + join
# ─────────────────────────────────────────────────────────────────────────────

def _load_player_names(conn) -> Dict[int, str]:
    """Return player_id → normalized_name from boxscore player stats."""
    rows = pd.read_sql(
        "SELECT DISTINCT player_id, first_name, last_name "
        "FROM raw.mlb_boxscore_player_stats "
        "WHERE first_name IS NOT NULL AND last_name IS NOT NULL",
        conn,
    )
    result: Dict[int, str] = {}
    for _, r in rows.iterrows():
        pid = int(r["player_id"])
        name = _norm_name(f"{r['first_name']} {r['last_name']}")
        if name:
            result[pid] = name
    return result


def _load_prop_lines(conn, stat_bookmaker_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Load prop lines for given (stat, bookmaker) pairs.
    Returns DataFrame with columns [as_of_date, player_name_norm, stat, bookmaker_key, line].
    """
    stats = list({s for s, _ in stat_bookmaker_pairs})
    bookmakers = list({b for _, b in stat_bookmaker_pairs})
    with conn.cursor() as cur:
        cur.execute(_SQL_PROP_LINES, {"stats": stats, "bookmakers": bookmakers})
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["as_of_date", "player_name_norm", "stat", "bookmaker_key", "line"])
    df = pd.DataFrame(rows, columns=["as_of_date", "player_name_norm", "stat", "bookmaker_key", "line"])
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    df["line"] = df["line"].astype(float)
    return df


def _join_lines_for_stat(
    df_train: pd.DataFrame,
    name_map: Dict[int, str],
    lines_df: pd.DataFrame,
    target_col: str,
    odds_stat: str,
    bookmaker: str,
) -> pd.DataFrame:
    """
    Vectorized join: add book_line + over_hit to training rows for one stat.
    Only rows with a matching book_line are returned.
    """
    stat_lines = lines_df[
        (lines_df["stat"] == odds_stat) & (lines_df["bookmaker_key"] == bookmaker)
    ][["as_of_date", "player_name_norm", "line"]].copy()
    if stat_lines.empty:
        log.warning("No prop lines found for stat=%s bookmaker=%s", odds_stat, bookmaker)
        return pd.DataFrame()

    # Add normalized name to training df
    df_work = df_train.copy()
    df_work["_norm_name"] = df_work["player_id"].map(
        lambda pid: name_map.get(int(pid), None)
    )
    df_work = df_work.dropna(subset=["_norm_name"])
    df_work["_date"] = pd.to_datetime(df_work["game_date_et"]).dt.date

    # Vectorized merge on (date, normalized name)
    stat_lines = stat_lines.rename(columns={"as_of_date": "_date", "player_name_norm": "_norm_name"})
    merged = df_work.merge(stat_lines, on=["_date", "_norm_name"], how="inner")

    if merged.empty:
        return pd.DataFrame()

    merged["book_line"] = merged["line"].astype(float)
    merged["over_hit"] = (merged[target_col].astype(float) > merged["book_line"]).astype(int)
    merged = merged.drop(columns=["_norm_name", "_date", "line"])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Feature prep
# ─────────────────────────────────────────────────────────────────────────────

def _prep_X_clf(
    df: pd.DataFrame,
    target_cols: List[str],
    meta_cols: List[str],
    extra_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop meta/target cols, keep book_line as feature, OHE season, add derived features.
    book_line stays in X — it is the key new signal for binary classification.
    """
    drop_cols = set(target_cols) | set(meta_cols) | set(extra_drop or [])
    # Never drop book_line — it's our primary new feature
    drop_cols.discard("book_line")
    # Drop over_hit (the binary target) from features
    drop_cols.add("over_hit")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    if "season" in X.columns:
        X = pd.get_dummies(X, columns=["season"], drop_first=False, dummy_na=False)

    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric(X)

    bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if bad:
        X = pd.get_dummies(X, columns=bad, drop_first=False, dummy_na=True)

    X = add_player_prop_derived_features(X)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_xgb_clf(cfg: ClfConfig, n_est: Optional[int] = None,
                   early_stop: bool = True,
                   params_override: Optional[Dict] = None) -> XGBRegressor:
    """XGBRegressor with binary:logistic outputs probabilities from predict()."""
    p = dict(
        n_estimators=n_est or cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    if early_stop:
        p["early_stopping_rounds"] = cfg.early_stopping_rounds
    if params_override:
        p.update(params_override)
    return XGBRegressor(**p)


def _build_lgb_clf(n_est: int = 1000, early_stop: bool = True,
                   params_override: Optional[Dict] = None):
    if not _HAS_LGB:
        return None
    p = dict(
        n_estimators=n_est,
        num_leaves=31,
        learning_rate=0.05,
        objective="binary",
        metric="binary_logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params_override:
        _lgb_keys = {"learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}
        p.update({k: v for k, v in params_override.items() if k in _lgb_keys})
    if early_stop:
        p["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    return lgb.LGBMRegressor(**p)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward CV
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_folds(
    df: pd.DataFrame,
    min_train_days: int,
    test_window_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(df["game_date_et"])
    start = dates.min().normalize()
    end = dates.max().normalize() + pd.Timedelta(days=1)
    first_end = start + pd.Timedelta(days=min_train_days)
    if first_end >= end:
        return []
    folds = []
    train_end = first_end
    while True:
        test_end = train_end + pd.Timedelta(days=test_window_days)
        if test_end > end:
            break
        folds.append((train_end, test_end))
        train_end = train_end + pd.Timedelta(days=step_days)
    return folds


def _run_walk_forward_clf(
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    medians: Dict[str, float],
    cfg: ClfConfig,
    stat_name: str,
) -> Tuple[float, float]:
    """Walk-forward CV for binary classifier. Returns (brier_score, auc)."""
    folds = _walk_forward_folds(df, cfg.min_train_days, cfg.test_window_days, cfg.step_days)
    if not folds:
        log.warning("No walk-forward folds for %s (need %d days of prop line data)",
                    stat_name, cfg.min_train_days)
        return float("nan"), float("nan")

    oof_preds, oof_actual = [], []

    for train_end, test_end in folds:
        tr_mask = pd.to_datetime(df["game_date_et"]) < train_end
        te_mask = (pd.to_datetime(df["game_date_et"]) >= train_end) & \
                  (pd.to_datetime(df["game_date_et"]) < test_end)
        if tr_mask.sum() < 50 or te_mask.sum() == 0:
            continue

        X_tr = apply_medians(X_raw[tr_mask], medians, feature_cols)
        X_te = apply_medians(X_raw[te_mask], medians, feature_cols)
        y_tr = y[tr_mask]
        y_te = y[te_mask]

        # Skip fold if only one class in train or test
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        cutoff = X_tr.index[int(len(X_tr) * 0.85)]
        fit_mask = X_tr.index < cutoff
        eval_mask = X_tr.index >= cutoff
        if fit_mask.sum() < 30 or eval_mask.sum() == 0:
            fit_mask = slice(None)
            eval_mask = None

        xgb = _build_xgb_clf(cfg, early_stop=True)
        if eval_mask is not None:
            xgb.fit(X_tr[fit_mask], y_tr[fit_mask],
                    eval_set=[(X_tr[eval_mask], y_tr[eval_mask])],
                    verbose=False)
        else:
            xgb.fit(X_tr, y_tr, verbose=False)

        preds = np.clip(xgb.predict(X_te), 1e-6, 1 - 1e-6)

        if _HAS_LGB:
            lgb_model = _build_lgb_clf(early_stop=True)
            if eval_mask is not None:
                lgb_model.fit(X_tr[fit_mask], y_tr[fit_mask],
                              eval_set=[(X_tr[eval_mask], y_tr[eval_mask])])
            else:
                lgb_model.fit(X_tr, y_tr)
            lgb_preds = np.clip(lgb_model.predict(X_te), 1e-6, 1 - 1e-6)
            preds = (preds + lgb_preds) / 2.0

        oof_preds.extend(preds)
        oof_actual.extend(y_te.values)

    if len(oof_preds) < 20:
        return float("nan"), float("nan")

    oof_preds = np.array(oof_preds)
    oof_actual = np.array(oof_actual)

    brier = float(brier_score_loss(oof_actual, oof_preds))
    try:
        auc = float(roc_auc_score(oof_actual, oof_preds))
    except ValueError:
        auc = float("nan")

    over_rate = float(oof_actual.mean())
    log.info(
        "Walk-forward CLF %s | Brier=%.4f AUC=%.4f | %d OOF rows, %d folds | OVER%%=%.1f%%",
        stat_name, brier, auc, len(oof_actual), len(folds), over_rate * 100,
    )
    return brier, auc


def _fit_final_clf(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ClfConfig,
    stat_name: str,
    n_estimators: Optional[int] = None,
) -> Tuple[XGBRegressor, Optional[object]]:
    """Fit final binary XGB + LGB on all data."""
    n_est = n_estimators or cfg.n_estimators
    log.info("Fitting final CLF %s XGB (n=%d rows, n_estimators=%d)", stat_name, len(X), n_est)
    xgb = _build_xgb_clf(cfg, n_est=n_est, early_stop=False)
    xgb.fit(X, y, verbose=False)

    lgb_model = None
    if _HAS_LGB:
        log.info("Fitting final CLF %s LGB", stat_name)
        lgb_model = _build_lgb_clf(n_est=n_est, early_stop=False)
        lgb_model.fit(X, y)

    return xgb, lgb_model


# ─────────────────────────────────────────────────────────────────────────────
# Train batter binary classifiers
# ─────────────────────────────────────────────────────────────────────────────

def train_batter_clf(cfg: ClfConfig) -> Dict:
    conn = psycopg2.connect(cfg.pg_dsn)

    # Load base training data (same as regression)
    log.info("Loading batter training data...")
    df_base = pd.read_sql(SQL_BATTER_TRAIN, conn)
    df_base["game_date_et"] = pd.to_datetime(df_base["game_date_et"])

    # Load player name lookup and prop lines
    log.info("Loading player names and prop lines...")
    name_map = _load_player_names(conn)
    stat_bk_pairs = list(_BATTER_STAT_MAP.values())
    lines_df = _load_prop_lines(conn, stat_bk_pairs)
    conn.close()

    log.info("Batter base rows: %d | name_map entries: %d | prop line rows: %d",
             len(df_base), len(name_map), len(lines_df))

    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {}
    feature_cols_saved = None
    medians_saved = None

    for target_col, (odds_stat, bookmaker) in _BATTER_STAT_MAP.items():
        log.info("=== Binary CLF: %s → %s ===", target_col, odds_stat)

        df_stat = _join_lines_for_stat(df_base, name_map, lines_df, target_col, odds_stat, bookmaker)
        if df_stat.empty or len(df_stat) < 200:
            log.warning("Skipping %s — only %d rows after prop line join", target_col, len(df_stat))
            results[f"brier_{target_col}"] = float("nan")
            results[f"auc_{target_col}"] = float("nan")
            continue

        log.info("  %s: %d rows after prop line join, date range %s to %s",
                 target_col, len(df_stat),
                 df_stat["game_date_et"].min().date(),
                 df_stat["game_date_et"].max().date())
        log.info("  OVER rate: %.1f%% (line avg=%.2f)",
                 df_stat["over_hit"].mean() * 100,
                 df_stat["book_line"].mean())

        y = df_stat["over_hit"].astype(float)
        extra_drop = [c for c in _BATTER_TARGETS if c != target_col] + ["over_hit"]
        X_raw = _prep_X_clf(df_stat, _BATTER_TARGETS, _BATTER_META, extra_drop=extra_drop)

        feature_cols = list(X_raw.columns)
        medians = fit_medians(X_raw)
        X_filled = apply_medians(X_raw, medians, feature_cols)

        # Walk-forward evaluation
        brier, auc = _run_walk_forward_clf(df_stat, X_raw, y, feature_cols, medians, cfg, target_col)
        results[f"brier_{target_col}"] = brier
        results[f"auc_{target_col}"] = auc

        # Final model
        xgb, lgb_model = _fit_final_clf(X_filled, y, cfg, target_col)

        # Save models
        xgb.save_model(str(model_dir / f"{target_col}_clf_xgb.json"))
        if lgb_model is not None:
            lgb_model.booster_.save_model(str(model_dir / f"lgb_{target_col}_clf.txt"))

        # Save feature cols / medians (same across all batter stats since same X)
        if feature_cols_saved is None:
            feature_cols_saved = feature_cols
            medians_saved = medians

    # Save shared feature metadata
    if feature_cols_saved:
        (model_dir / "feature_columns_clf_batters.json").write_text(
            json.dumps(feature_cols_saved), encoding="utf-8"
        )
        (model_dir / "feature_medians_clf_batters.json").write_text(
            json.dumps(medians_saved), encoding="utf-8"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Train pitcher binary classifier (strikeouts)
# ─────────────────────────────────────────────────────────────────────────────

def train_pitcher_clf(cfg: ClfConfig) -> Dict:
    conn = psycopg2.connect(cfg.pg_dsn)

    log.info("Loading pitcher training data...")
    df_base = pd.read_sql(SQL_PITCHER_TRAIN, conn)
    df_base["game_date_et"] = pd.to_datetime(df_base["game_date_et"])

    log.info("Loading pitcher names and prop lines...")
    name_map = _load_player_names(conn)
    lines_df = _load_prop_lines(conn, list(_PITCHER_STAT_MAP.values()))
    conn.close()


    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {}

    for target_col, (odds_stat, bookmaker) in _PITCHER_STAT_MAP.items():
        log.info("=== Binary CLF: %s → %s ===", target_col, odds_stat)

        df_stat = _join_lines_for_stat(df_base, name_map, lines_df, target_col, odds_stat, bookmaker)
        if df_stat.empty or len(df_stat) < 100:
            log.warning("Skipping %s — only %d rows after prop line join", target_col, len(df_stat))
            results[f"brier_{target_col}"] = float("nan")
            results[f"auc_{target_col}"] = float("nan")
            continue

        log.info("  %s: %d rows, OVER rate=%.1f%%, line avg=%.2f",
                 target_col, len(df_stat),
                 df_stat["over_hit"].mean() * 100,
                 df_stat["book_line"].mean())

        y = df_stat["over_hit"].astype(float)
        extra_drop = [c for c in _PITCHER_TARGETS if c != target_col] + ["over_hit"]
        X_raw = _prep_X_clf(df_stat, _PITCHER_TARGETS, _PITCHER_META, extra_drop=extra_drop)

        feature_cols = list(X_raw.columns)
        medians = fit_medians(X_raw)
        X_filled = apply_medians(X_raw, medians, feature_cols)

        brier, auc = _run_walk_forward_clf(df_stat, X_raw, y, feature_cols, medians, cfg, target_col)
        results[f"brier_{target_col}"] = brier
        results[f"auc_{target_col}"] = auc

        xgb, lgb_model = _fit_final_clf(X_filled, y, cfg, target_col)

        xgb.save_model(str(model_dir / f"{target_col}_clf_xgb.json"))
        if lgb_model is not None:
            lgb_model.booster_.save_model(str(model_dir / f"lgb_{target_col}_clf.txt"))

        (model_dir / "feature_columns_clf_pitchers.json").write_text(
            json.dumps(feature_cols), encoding="utf-8"
        )
        (model_dir / "feature_medians_clf_pitchers.json").write_text(
            json.dumps(medians), encoding="utf-8"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = ClfConfig()

    log.info("=== MLB binary prop model training ===")

    pitcher_results = train_pitcher_clf(cfg)
    batter_results  = train_batter_clf(cfg)

    all_results = {**pitcher_results, **batter_results}

    (cfg.model_dir / "backtest_clf.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    log.info("Saved backtest_clf.json: %s", all_results)
    log.info("MLB binary prop training complete.")


if __name__ == "__main__":
    main()
