"""Train direct MLB prop side classifiers from locked side examples.

Each model predicts whether a specific offered side wins after seeing the
player model probability, offered line, price, book, and bucket metadata.
This is intentionally trained from locked model-pick examples, not from broad
market history, so it can become real-money evidence only after enough shadow
ledger rows accumulate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"

_NUMERIC_FEATURES = [
    "model_prob_side",
    "market_prob_side",
    "prob_edge_vs_market",
    "market_line",
    "abs_price",
    "is_plus_price",
    "count_edge_side",
    "pred_count",
    "pred_value",
]

_CATEGORICAL_FEATURES = [
    "market",
    "side",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
    "model_family",
    "edge_type",
]


@dataclass(frozen=True)
class PropDirectSideConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_direct_side_models.json"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 150
    min_holdout_rows: int = 40


SQL = """
SELECT
    game_date_et,
    market,
    side,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(model_family, 'unknown') AS model_family,
    COALESCE(edge_type, 'unknown') AS edge_type,
    model_prob_side::float AS model_prob_side,
    market_prob_side::float AS market_prob_side,
    prob_edge_vs_market::float AS prob_edge_vs_market,
    market_line::float AS market_line,
    ABS(market_price::float) AS abs_price,
    CASE
        WHEN market_price IS NULL THEN NULL
        WHEN market_price::float > 0 THEN 1.0
        ELSE 0.0
    END AS is_plus_price,
    count_edge_side::float AS count_edge_side,
    pred_count::float AS pred_count,
    pred_value::float AS pred_value,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS target,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    clv_price::float AS clv_price,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND model_prob_side IS NOT NULL
  AND market_line IS NOT NULL
  AND won IS NOT NULL
  AND model_prob_side BETWEEN 0.0 AND 1.0
"""


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _load(cfg: PropDirectSideConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = pd.read_sql(SQL, conn, params={"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in _NUMERIC_FEATURES + ["target", "profit_units", "clv_price", "beat_clv_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["model_prob_side", "target"])
    df["model_prob_side"] = df["model_prob_side"].clip(1e-6, 1.0 - 1e-6)
    df["target"] = df["target"].astype(int)
    df["push"] = df["push"].fillna(False).astype(bool)
    for col in _CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def _key(
    market: str = "*",
    side: str = "*",
    line_bucket: str = "*",
    price_bucket: str = "*",
    bookmaker_key: str = "*",
    model_family: str = "*",
) -> str:
    return "|".join([
        market or "*",
        side or "*",
        line_bucket or "*",
        price_bucket or "*",
        bookmaker_key or "*",
        model_family or "*",
    ])


def _prepare_matrix(df: pd.DataFrame, *, means=None, scales=None, cats=None):
    means = dict(means or {})
    scales = dict(scales or {})
    cats = {k: list(v) for k, v in (cats or {}).items()}
    parts = []
    feature_names = []
    numeric_means = {}
    numeric_scales = {}
    for name in _NUMERIC_FEATURES:
        s = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, s.mean() if not s.dropna().empty else 0.0))
        filled = s.fillna(mean)
        std = float(filled.std(ddof=0) or 0.0)
        scale = float(scales.get(name, std if std > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy(dtype=float).reshape(-1, 1))
        feature_names.append(name)
        numeric_means[name] = mean
        numeric_scales[name] = scale
    cat_values = {}
    for name in _CATEGORICAL_FEATURES:
        values = cats.get(name)
        if values is None:
            values = sorted(str(v) for v in df[name].fillna("unknown").astype(str).unique())
        cat_values[name] = values
        series = df[name].fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))
            feature_names.append(f"{name}={value}")
    X = np.hstack(parts) if parts else np.zeros((len(df), 0))
    return X, feature_names, numeric_means, numeric_scales, cat_values


def _split_mask(df: pd.DataFrame, cfg: PropDirectSideConfig) -> tuple[pd.Series, pd.Series, str]:
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train_mask = df["game_date_et"] < split
    holdout_mask = df["game_date_et"] >= split
    if train_mask.any():
        return train_mask, holdout_mask, f"last_{cfg.holdout_days}_days"
    dates = sorted(df["game_date_et"].unique())
    if len(dates) > 1:
        holdout_date = dates[-1]
        return df["game_date_et"] < holdout_date, df["game_date_et"] >= holdout_date, "last_available_date"
    return train_mask, holdout_mask, f"last_{cfg.holdout_days}_days"


def _fit_group(train: pd.DataFrame, holdout: pd.DataFrame, all_rows: pd.DataFrame) -> dict | None:
    if len(train) < 40 or len(holdout) < 12:
        return None
    if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        return None
    X_train, feature_names, means, scales, cats = _prepare_matrix(train)
    y_train = train["target"].astype(int).to_numpy()
    lr = LogisticRegression(max_iter=3000, solver="lbfgs")
    lr.fit(X_train, y_train)

    X_hold, _, _, _, _ = _prepare_matrix(holdout, means=means, scales=scales, cats=cats)
    raw = holdout["model_prob_side"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    y_hold = holdout["target"].astype(int).to_numpy()
    pred_hold = np.clip(lr.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    raw_brier = float(brier_score_loss(y_hold, raw))
    model_brier = float(brier_score_loss(y_hold, pred_hold))
    improvement = raw_brier - model_brier
    base_weight = min(0.85, len(holdout) / (len(holdout) + 160.0))
    if improvement < -0.006:
        blend = base_weight * 0.15
    elif improvement < 0.001:
        blend = base_weight * 0.45
    else:
        blend = base_weight

    X_all, feature_names, means, scales, cats = _prepare_matrix(all_rows)
    y_all = all_rows["target"].astype(int).to_numpy()
    final = LogisticRegression(max_iter=3000, solver="lbfgs")
    final.fit(X_all, y_all)
    coef = {
        name: float(value)
        for name, value in zip(feature_names, final.coef_[0])
        if abs(float(value)) > 1e-12
    }
    profit = pd.to_numeric(holdout.get("profit_units"), errors="coerce")
    clv = pd.to_numeric(holdout.get("beat_clv_price"), errors="coerce").dropna()
    return {
        "method": "locked_direct_side_logistic",
        "intercept": float(final.intercept_[0]),
        "coef": coef,
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "blend_weight": float(blend),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "actual_rate_holdout": float(np.mean(y_hold)),
        "avg_raw_holdout": float(np.mean(raw)),
        "avg_model_holdout": float(np.mean(pred_hold)),
        "holdout_roi": float(profit.dropna().mean()) if not profit.dropna().empty else None,
        "holdout_clv_beat_rate": float(clv.mean()) if not clv.empty else None,
        "brier_raw_holdout": raw_brier,
        "brier_model_holdout": model_brier,
        "log_loss_raw_holdout": float(log_loss(y_hold, raw, labels=[0, 1])),
        "log_loss_model_holdout": float(log_loss(y_hold, pred_hold, labels=[0, 1])),
        "auc_raw_holdout": float(roc_auc_score(y_hold, raw)) if len(np.unique(y_hold)) == 2 else None,
        "auc_model_holdout": float(roc_auc_score(y_hold, pred_hold)) if len(np.unique(y_hold)) == 2 else None,
    }


def _group_specs() -> Iterable[tuple[tuple[str, ...], int, int]]:
    return [
        (("market", "side", "line_bucket", "price_bucket", "bookmaker_key", "model_family"), 180, 45),
        (("market", "side", "line_bucket", "price_bucket", "bookmaker_key"), 180, 45),
        (("market", "side", "line_bucket", "price_bucket"), 180, 45),
        (("market", "side", "line_bucket"), 180, 45),
        (("market", "side"), 150, 40),
        (("side",), 260, 80),
        (tuple(), 400, 120),
    ]


def _key_for_group(group_cols: tuple[str, ...], values) -> str:
    if not group_cols:
        return _key()
    values = values if isinstance(values, tuple) else (values,)
    d = dict(zip(group_cols, values))
    return _key(
        d.get("market", "*"),
        d.get("side", "*"),
        d.get("line_bucket", "*"),
        d.get("price_bucket", "*"),
        d.get("bookmaker_key", "*"),
        d.get("model_family", "*"),
    )


def train(cfg: PropDirectSideConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "rows": int(len(df)),
        "bankroll_ready": False,
        "usage": "shadow_only",
        "promotion_rule": (
            "Direct side models must improve probability shadow Brier, ROI, and CLV, "
            "then pass bucket reopen thresholds before affecting bankroll picks."
        ),
        "models": {},
        "backtest": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    train_mask, holdout_mask, split_strategy = _split_mask(df, cfg)
    payload["split_strategy"] = split_strategy
    payload["train_rows"] = int(train_mask.sum())
    payload["holdout_rows"] = int(holdout_mask.sum())
    for group_cols, min_train, min_holdout in _group_specs():
        grouped = [((), df)] if not group_cols else list(df.groupby(list(group_cols), dropna=False))
        for values, sub in grouped:
            tr = sub.loc[train_mask.loc[sub.index]]
            ho = sub.loc[holdout_mask.loc[sub.index]]
            if len(tr) < max(min_train, cfg.min_train_rows) or len(ho) < max(min_holdout, cfg.min_holdout_rows):
                continue
            rec = _fit_group(tr, ho, sub)
            if rec is None:
                continue
            key = _key_for_group(group_cols, values)
            rec["key"] = key
            rec["group_columns"] = list(group_cols)
            payload["models"][key] = rec
            payload["backtest"].append({
                k: rec[k]
                for k in (
                    "key",
                    "train_rows",
                    "holdout_rows",
                    "actual_rate_holdout",
                    "avg_raw_holdout",
                    "avg_model_holdout",
                    "holdout_roi",
                    "holdout_clv_beat_rate",
                    "brier_raw_holdout",
                    "brier_model_holdout",
                    "blend_weight",
                )
            })
    payload["status"] = "trained" if payload["models"] else "no_models"
    payload["backtest"].sort(key=lambda r: r["key"])
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct MLB prop side models from locked side examples")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_direct_side_models.json")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=150)
    parser.add_argument("--min-holdout-rows", type=int, default=40)
    args = parser.parse_args()
    payload = train(PropDirectSideConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
    ))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
