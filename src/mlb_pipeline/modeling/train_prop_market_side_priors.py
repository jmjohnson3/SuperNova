"""Train historical market-side priors for MLB props.

These models learn P(side wins) from the offered line/price/book context. They
are not a replacement for SuperNova's player model edge; they are a stable
market calibration prior used to diagnose and eventually guard bucket reopening.
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

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"

_NUMERIC_FEATURES = [
    "raw_p_side",
    "market_line",
    "abs_price",
    "is_plus_price",
]

_CATEGORICAL_FEATURES = [
    "market",
    "side",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
]


@dataclass(frozen=True)
class PropMarketSidePriorConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_market_side_priors.json"
    lookback_days: int = 540
    holdout_days: int = 28
    min_train_rows: int = 500
    min_holdout_rows: int = 80


SQL = """
SELECT
    game_date_et,
    market,
    side,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    market_prob_side::float AS raw_p_side,
    market_line::float AS market_line,
    ABS(market_price::float) AS abs_price,
    CASE
        WHEN market_price IS NULL THEN NULL
        WHEN market_price::float > 0 THEN 1.0
        ELSE 0.0
    END AS is_plus_price,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS target,
    profit_units::float AS profit_units
FROM features.mlb_prop_market_history_examples
WHERE game_date_et >= %(cutoff)s
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND market_price IS NOT NULL
  AND market_prob_side IS NOT NULL
  AND won IS NOT NULL
  AND market_prob_side BETWEEN 0.0 AND 1.0
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


def _load(cfg: PropMarketSidePriorConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_history_examples"):
            return pd.DataFrame()
        df = pd.read_sql(SQL, conn, params={"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in _NUMERIC_FEATURES + ["target", "profit_units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_p_side", "target"])
    df["raw_p_side"] = df["raw_p_side"].clip(1e-6, 1.0 - 1e-6)
    df["target"] = df["target"].astype(int)
    for col in _CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def _key(market: str = "*", side: str = "*", line_bucket: str = "*", price_bucket: str = "*", bookmaker_key: str = "*") -> str:
    return "|".join([market or "*", side or "*", line_bucket or "*", price_bucket or "*", bookmaker_key or "*"])


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


def _fit_group(train: pd.DataFrame, holdout: pd.DataFrame, all_rows: pd.DataFrame) -> dict | None:
    if len(train) < 80 or len(holdout) < 20:
        return None
    if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        return None
    X_train, feature_names, means, scales, cats = _prepare_matrix(train)
    y_train = train["target"].astype(int).to_numpy()
    lr = LogisticRegression(max_iter=3000, solver="lbfgs")
    lr.fit(X_train, y_train)

    X_hold, _, _, _, _ = _prepare_matrix(holdout, means=means, scales=scales, cats=cats)
    raw = holdout["raw_p_side"].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    y_hold = holdout["target"].astype(int).to_numpy()
    pred_hold = np.clip(lr.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)
    raw_brier = float(brier_score_loss(y_hold, raw))
    model_brier = float(brier_score_loss(y_hold, pred_hold))
    improvement = raw_brier - model_brier
    base_weight = min(0.85, len(holdout) / (len(holdout) + 250.0))
    if improvement < -0.004:
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
    return {
        "method": "historical_market_side_prior",
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
        "brier_raw_holdout": raw_brier,
        "brier_model_holdout": model_brier,
        "log_loss_raw_holdout": float(log_loss(y_hold, raw, labels=[0, 1])),
        "log_loss_model_holdout": float(log_loss(y_hold, pred_hold, labels=[0, 1])),
        "auc_raw_holdout": float(roc_auc_score(y_hold, raw)) if len(np.unique(y_hold)) == 2 else None,
        "auc_model_holdout": float(roc_auc_score(y_hold, pred_hold)) if len(np.unique(y_hold)) == 2 else None,
    }


def _group_specs() -> Iterable[tuple[tuple[str, ...], int, int]]:
    return [
        (("market", "side", "line_bucket", "price_bucket", "bookmaker_key"), 700, 100),
        (("market", "side", "line_bucket", "bookmaker_key"), 900, 120),
        (("market", "side", "line_bucket"), 1200, 160),
        (("market", "side", "bookmaker_key"), 1200, 160),
        (("market", "side"), 1600, 220),
        (("side", "bookmaker_key"), 2200, 280),
        (("side",), 2600, 350),
        (tuple(), 4000, 500),
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
    )


def train(cfg: PropMarketSidePriorConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_history_examples",
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "rows": int(len(df)),
        "models": {},
        "backtest": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train_mask = df["game_date_et"] < split
    holdout_mask = df["game_date_et"] >= split
    for group_cols, min_train, min_holdout in _group_specs():
        grouped = [((), df)] if not group_cols else list(df.groupby(list(group_cols), dropna=False))
        for values, sub in grouped:
            tr = sub.loc[train_mask.loc[sub.index]]
            ho = sub.loc[holdout_mask.loc[sub.index]]
            if len(tr) < min(min_train, cfg.min_train_rows) or len(ho) < min(min_holdout, cfg.min_holdout_rows):
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
    parser = argparse.ArgumentParser(description="Train historical MLB prop market-side priors")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_market_side_priors.json")
    parser.add_argument("--lookback-days", type=int, default=540)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-holdout-rows", type=int, default=80)
    args = parser.parse_args()
    payload = train(PropMarketSidePriorConfig(
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
