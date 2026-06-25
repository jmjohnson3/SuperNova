"""Train side/line/price recalibrators for MLB prop model probabilities.

This calibrates the probability of the selected betting side:
  - pitcher_strikeouts over / under
  - batter_hits over / under
  - batter_total_bases over / under
  - batter_home_runs over / under

It uses graded historical prediction rows, so it improves the live betting
probability without needing new data feeds or a full model retrain. Each graded
P(over) row contributes both an over-side and under-side calibration example.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .side_recalibration import calibration_key, logit, price_bucket, prop_line_bucket, sigmoid

log = logging.getLogger("mlb_pipeline.modeling.train_prop_side_recalibrators")

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class PropSideRecalConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_side_recalibrators.json"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 120
    min_holdout_rows: int = 20


SQL = """
SELECT
    game_date_et,
    stat,
    bet_side AS selected_side,
    book_line::float AS book_line,
    over_price::float AS over_price,
    under_price::float AS under_price,
    bet_price::float AS bet_price,
    pred_prob_over::float AS pred_prob_over,
    edge_type,
    COALESCE(model_family, 'unknown') AS model_family,
    line_bucket,
    over_hit
FROM bets.mlb_prop_predictions
WHERE game_date_et >= %(cutoff)s
  AND stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND pred_prob_over IS NOT NULL
  AND book_line IS NOT NULL
  AND over_hit IS NOT NULL
  AND pred_prob_over BETWEEN 0.0 AND 1.0
"""

SQL_REPLAY = """
SELECT
    game_date_et,
    stat AS stat,
    side AS selected_side,
    market_line::float AS book_line,
    over_price::float AS over_price,
    under_price::float AS under_price,
    market_price::float AS bet_price,
    model_prob_over::float AS pred_prob_over,
    edge_type,
    COALESCE(model_family, 'unknown') AS model_family,
    line_bucket,
    over_hit
FROM bets.mlb_prop_prediction_replay
WHERE game_date_et >= %(cutoff)s
  AND stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND model_prob_over IS NOT NULL
  AND market_line IS NOT NULL
  AND over_hit IS NOT NULL
  AND model_prob_over BETWEEN 0.0 AND 1.0
"""

SQL_MARKET_TRAINING = """
SELECT
    game_date_et,
    market AS stat,
    side,
    market_line::float AS book_line,
    market_price::float AS side_price,
    model_prob_side::float AS raw_p_side,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS target,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(model_family, 'unknown') AS model_family
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


def _load(cfg: PropSideRecalConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = pd.DataFrame()
        if _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            df = pd.read_sql(SQL_MARKET_TRAINING, conn, params={"cutoff": cutoff})
            if not df.empty:
                df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
                df["market"] = df["stat"].astype(str)
                df["side"] = df["side"].fillna("").astype(str).str.lower()
                df["book_line"] = pd.to_numeric(df["book_line"], errors="coerce")
                df["side_price"] = pd.to_numeric(df["side_price"], errors="coerce")
                df["raw_p_side"] = pd.to_numeric(df["raw_p_side"], errors="coerce")
                df["target"] = pd.to_numeric(df["target"], errors="coerce")
                df["line_bucket"] = [
                    lb if isinstance(lb, str) and lb and lb != "unknown" else prop_line_bucket(stat, line)
                    for stat, line, lb in zip(df["stat"], df["book_line"], df["line_bucket"])
                ]
                df["price_bucket"] = [
                    pb if isinstance(pb, str) and pb else price_bucket(price)
                    for pb, price in zip(df["price_bucket"], df["side_price"])
                ]
                df["model_family"] = df["model_family"].fillna("unknown").astype(str)
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_p_side", "target"])
                df["raw_p_side"] = df["raw_p_side"].clip(1e-6, 1 - 1e-6)
                df["target"] = df["target"].astype(int)
                return df
        if _table_exists(conn, "bets", "mlb_prop_prediction_replay"):
            df = pd.read_sql(SQL_REPLAY, conn, params={"cutoff": cutoff})
        if df.empty:
            df = pd.read_sql(SQL, conn, params={"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["selected_side"] = df["selected_side"].fillna("").astype(str).str.lower()
    df["market"] = df["stat"].astype(str)
    df["line_bucket"] = [
        lb if isinstance(lb, str) and lb and lb != "unknown" else prop_line_bucket(stat, line)
        for stat, line, lb in zip(df["stat"], df["book_line"], df["line_bucket"])
    ]
    df["model_family"] = df["model_family"].fillna("unknown").astype(str)
    df["pred_prob_over"] = pd.to_numeric(df["pred_prob_over"], errors="coerce")
    over_hit = df["over_hit"].astype(bool)

    over = df.copy()
    over["side"] = "over"
    over["raw_p_side"] = over["pred_prob_over"]
    over["target"] = over_hit.astype(int)
    over["side_price"] = over["over_price"].where(
        over["over_price"].notna(),
        over["bet_price"].where(over["selected_side"].eq("over")),
    )

    under = df.copy()
    under["side"] = "under"
    under["raw_p_side"] = 1.0 - under["pred_prob_over"]
    under["target"] = (~over_hit).astype(int)
    under["side_price"] = under["under_price"].where(
        under["under_price"].notna(),
        under["bet_price"].where(under["selected_side"].eq("under")),
    )

    out = pd.concat([over, under], ignore_index=True)
    out["price_bucket"] = out["side_price"].map(price_bucket)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["raw_p_side", "target"])
    out["raw_p_side"] = out["raw_p_side"].clip(1e-6, 1 - 1e-6)
    return out


def _fit_platt(train: pd.DataFrame, fit_all: pd.DataFrame) -> tuple[dict, np.ndarray]:
    x_train = np.array([logit(v) for v in train["raw_p_side"]], dtype=float).reshape(-1, 1)
    y_train = train["target"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        rate = float(np.mean(y_train)) if len(y_train) else 0.5
        return {"method": "constant", "actual_rate": rate}, np.repeat(rate, len(train))
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(x_train, y_train)

    x_all = np.array([logit(v) for v in fit_all["raw_p_side"]], dtype=float).reshape(-1, 1)
    y_all = fit_all["target"].astype(int).to_numpy()
    lr_all = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr_all.fit(x_all, y_all)
    cal = {
        "method": "platt",
        "a": float(lr_all.coef_[0][0]),
        "b": float(lr_all.intercept_[0]),
    }
    train_pred = 1.0 / (1.0 + np.exp(-(lr.coef_[0][0] * x_train.ravel() + lr.intercept_[0])))
    return cal, np.clip(train_pred, 1e-6, 1 - 1e-6)


def _evaluate(train: pd.DataFrame, holdout: pd.DataFrame, all_rows: pd.DataFrame) -> dict | None:
    if len(train) < 20 or len(holdout) < 5:
        return None
    if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        return None
    cal, _ = _fit_platt(train, all_rows)
    # Evaluate with train-only parameters to avoid holdout leakage.
    x_train = np.array([logit(v) for v in train["raw_p_side"]], dtype=float).reshape(-1, 1)
    y_train = train["target"].astype(int).to_numpy()
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(x_train, y_train)
    x_hold = np.array([logit(v) for v in holdout["raw_p_side"]], dtype=float)
    raw = holdout["raw_p_side"].astype(float).to_numpy()
    y = holdout["target"].astype(int).to_numpy()
    cal_hold = np.clip(sigmoid(0.0), 1e-6, 1 - 1e-6) * np.ones_like(raw)
    cal_hold = 1.0 / (1.0 + np.exp(-(lr.coef_[0][0] * x_hold + lr.intercept_[0])))
    cal_hold = np.clip(cal_hold, 1e-6, 1 - 1e-6)

    raw_brier = float(brier_score_loss(y, raw))
    cal_brier = float(brier_score_loss(y, cal_hold))
    improvement = raw_brier - cal_brier
    base_weight = min(0.85, len(holdout) / (len(holdout) + 120.0))
    if improvement < -0.003:
        blend = base_weight * 0.20
    elif improvement < 0.001:
        blend = base_weight * 0.50
    else:
        blend = base_weight

    cal.update({
        "blend_weight": float(blend),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "actual_rate_train": float(train["target"].mean()),
        "actual_rate_holdout": float(np.mean(y)),
        "avg_raw_holdout": float(np.mean(raw)),
        "avg_cal_holdout": float(np.mean(cal_hold)),
        "brier_raw_holdout": raw_brier,
        "brier_cal_holdout": cal_brier,
        "log_loss_raw_holdout": float(log_loss(y, np.clip(raw, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "log_loss_cal_holdout": float(log_loss(y, np.clip(cal_hold, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "auc_raw_holdout": float(roc_auc_score(y, raw)) if len(np.unique(y)) == 2 else None,
        "auc_cal_holdout": float(roc_auc_score(y, cal_hold)) if len(np.unique(y)) == 2 else None,
    })
    return cal


def _group_specs() -> Iterable[tuple[tuple[str, ...], int, int]]:
    return [
        (("market", "side", "line_bucket", "price_bucket", "model_family"), 80, 12),
        (("market", "side", "line_bucket", "model_family"), 100, 15),
        (("market", "side", "line_bucket"), 100, 15),
        (("market", "side", "price_bucket"), 100, 15),
        (("market", "side", "model_family"), 120, 20),
        (("market", "side"), 120, 20),
    ]


def _key_for_group(group_cols: tuple[str, ...], values: tuple) -> str:
    item = dict(zip(group_cols, values if isinstance(values, tuple) else (values,)))
    return calibration_key(
        item.get("market", "*"),
        item.get("side", "*"),
        item.get("line_bucket", "*"),
        item.get("price_bucket", "*"),
        item.get("model_family", "*"),
    )


def train(cfg: PropSideRecalConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "rows": int(len(df)),
        "calibrators": {},
        "backtest": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        return payload
    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train_mask = df["game_date_et"] < split
    holdout_mask = df["game_date_et"] >= split

    for group_cols, min_train, min_holdout in _group_specs():
        for values, sub in df.groupby(list(group_cols), dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            tr = sub.loc[train_mask.loc[sub.index]]
            ho = sub.loc[holdout_mask.loc[sub.index]]
            if len(tr) < min(min_train, cfg.min_train_rows) or len(ho) < min(min_holdout, cfg.min_holdout_rows):
                continue
            rec = _evaluate(tr, ho, sub)
            if rec is None:
                continue
            key = _key_for_group(group_cols, values)
            rec["key"] = key
            rec["group_columns"] = list(group_cols)
            payload["calibrators"][key] = rec
            payload["backtest"].append({
                k: rec[k]
                for k in (
                    "key",
                    "train_rows",
                    "holdout_rows",
                    "actual_rate_holdout",
                    "avg_raw_holdout",
                    "avg_cal_holdout",
                    "brier_raw_holdout",
                    "brier_cal_holdout",
                    "blend_weight",
                )
            })

    payload["status"] = "trained" if payload["calibrators"] else "no_calibrators"
    payload["backtest"].sort(key=lambda r: (r["key"]))
    out = cfg.model_dir / cfg.out_file
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop side recalibrators")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_side_recalibrators.json")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--min-holdout-rows", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = train(PropSideRecalConfig(
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
