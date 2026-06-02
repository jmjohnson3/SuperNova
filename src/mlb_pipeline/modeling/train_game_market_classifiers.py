"""Train optional MLB game-market classifiers.

These models estimate:
  - P(home team covers the listed run line)
  - P(game total goes over the listed total)

They are intentionally optional. predict_today.py will use them as probability
confirmation gates when the artifacts exist, and fall back to the regression
edge workflow when they do not.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from .train_game_models import (
    TrainConfig,
    apply_fill,
    compute_recency_weights,
    fit_fill_stats,
    load_training_frame,
    make_xy_raw,
)

log = logging.getLogger("mlb_pipeline.modeling.train_game_market_classifiers")


@dataclass(frozen=True)
class MarketClassifierConfig:
    pg_dsn: str = TrainConfig().pg_dsn
    model_dir: Path = Path(__file__).resolve().parent / "models"
    holdout_days: int = 28
    min_train_rows: int = 200
    min_holdout_rows: int = 25
    recency_half_life_days: int = 45
    n_estimators: int = 700
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: int = 8
    reg_alpha: float = 0.1
    reg_lambda: float = 3.0
    random_state: int = 42


def _load_feature_schema(model_dir: Path, X_raw: pd.DataFrame) -> list[str]:
    path = model_dir / "feature_columns.json"
    if path.exists():
        cols = json.loads(path.read_text(encoding="utf-8"))
        return [str(c) for c in cols]
    return list(X_raw.columns)


def _build_classifier(cfg: MarketClassifierConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def _metric_payload(y_true: pd.Series, p: np.ndarray) -> Dict:
    y = y_true.astype(int).to_numpy()
    pred = (p >= 0.5).astype(int)
    payload = {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "accuracy_50": float(accuracy_score(y, pred)) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])) if len(y) else None,
    }
    if len(np.unique(y)) == 2:
        payload["auc"] = float(roc_auc_score(y, p))
    else:
        payload["auc"] = None
    return payload


def _train_one(
    cfg: MarketClassifierConfig,
    *,
    name: str,
    X_raw: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    feature_cols: list[str],
    feature_medians: Dict[str, float],
    out_file: str,
) -> Dict:
    valid = y.notna()
    X_raw = X_raw.loc[valid].copy()
    y = y.loc[valid].astype(int)
    dates = pd.to_datetime(dates.loc[valid])

    split = dates.max() - timedelta(days=cfg.holdout_days)
    train_mask = dates < split
    holdout_mask = dates >= split
    n_train = int(train_mask.sum())
    n_holdout = int(holdout_mask.sum())
    if n_train < cfg.min_train_rows or n_holdout < cfg.min_holdout_rows:
        return {
            "status": "skipped",
            "reason": "not_enough_rows",
            "train_rows": n_train,
            "holdout_rows": n_holdout,
        }

    medians = fit_fill_stats(X_raw.loc[train_mask])
    X_train = apply_fill(X_raw.loc[train_mask], medians, feature_cols)
    X_holdout = apply_fill(X_raw.loc[holdout_mask], medians, feature_cols)
    y_train = y.loc[train_mask]
    y_holdout = y.loc[holdout_mask]
    weights = compute_recency_weights(
        dates.loc[train_mask],
        half_life_days=cfg.recency_half_life_days,
    )

    model = _build_classifier(cfg)
    model.fit(X_train, y_train, sample_weight=weights, verbose=False)
    p_holdout = model.predict_proba(X_holdout)[:, 1]
    metrics = _metric_payload(y_holdout, p_holdout)

    # Refit on all valid rows for production, with medians from the full sample.
    X_all = apply_fill(X_raw, feature_medians, feature_cols)
    weights_all = compute_recency_weights(dates, half_life_days=cfg.recency_half_life_days)
    final_model = _build_classifier(cfg)
    final_model.fit(X_all, y, sample_weight=weights_all, verbose=False)
    final_model.save_model(str(cfg.model_dir / out_file))

    return {
        "status": "trained",
        "target": name,
        "train_rows": n_train,
        "holdout_rows": n_holdout,
        "holdout": metrics,
    }


def train(cfg: MarketClassifierConfig) -> Dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = load_training_frame(conn)

    X_raw, y_run_diff, y_total, _y_f5 = make_xy_raw(df)
    feature_cols = _load_feature_schema(cfg.model_dir, X_raw)
    feature_medians = fit_fill_stats(X_raw)

    run_line = pd.to_numeric(df.get("run_line_home"), errors="coerce")
    total_line = pd.to_numeric(df.get("total_line"), errors="coerce")
    y_home_cover = ((y_run_diff.astype(float) + run_line.astype(float)) > 0).astype(float)
    y_home_cover[run_line.isna()] = np.nan
    y_total_over = (y_total.astype(float) > total_line.astype(float)).astype(float)
    y_total_over[total_line.isna()] = np.nan

    payload = {
        "feature_schema": "models/feature_columns.json",
        "holdout_days": cfg.holdout_days,
        "recency_half_life_days": cfg.recency_half_life_days,
        "targets": {},
    }

    payload["targets"]["run_line_home_cover"] = _train_one(
        cfg,
        name="run_line_home_cover",
        X_raw=X_raw,
        y=y_home_cover,
        dates=df["game_date_et"],
        feature_cols=feature_cols,
        feature_medians=feature_medians,
        out_file="run_line_cover_clf_xgb.json",
    )
    payload["targets"]["total_over"] = _train_one(
        cfg,
        name="total_over",
        X_raw=X_raw,
        y=y_total_over,
        dates=df["game_date_et"],
        feature_cols=feature_cols,
        feature_medians=feature_medians,
        out_file="total_over_clf_xgb.json",
    )

    (cfg.model_dir / "game_market_clf_feature_columns.json").write_text(
        json.dumps(feature_cols), encoding="utf-8"
    )
    (cfg.model_dir / "game_market_clf_feature_medians.json").write_text(
        json.dumps(feature_medians), encoding="utf-8"
    )
    (cfg.model_dir / "game_market_clf_backtest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = train(MarketClassifierConfig())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
