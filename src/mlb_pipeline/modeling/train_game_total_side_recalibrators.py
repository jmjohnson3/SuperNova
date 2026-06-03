"""Train side/line/price recalibrators for MLB game totals.

This targets the known weak area directly: total overs and unders, including
high-total unders.  It calibrates the selected side probability using the
offered total line, price bucket, and whether the raw probability came from the
direct classifier or edge/sigma fallback.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .side_recalibration import calibration_key, game_total_line_bucket, logit, price_bucket

log = logging.getLogger("mlb_pipeline.modeling.train_game_total_side_recalibrators")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class GameTotalRecalConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "game_total_side_recalibrators.json"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 120
    min_holdout_rows: int = 20


SQL = """
SELECT
    game_date_et,
    market_total::float AS market_total,
    market_total_price::float AS market_total_price,
    edge_total::float AS edge_total,
    p_total_over_clf::float AS p_total_over_clf,
    win_prob_total::float AS win_prob_total,
    sigma_q_total::float AS sigma_q_total,
    actual_total::float AS actual_total
FROM bets.mlb_game_predictions
WHERE game_date_et >= %(cutoff)s
  AND market_total IS NOT NULL
  AND edge_total IS NOT NULL
  AND actual_total IS NOT NULL
"""


def _edge_sigma_prob(edge, sigma) -> float | None:
    try:
        e = abs(float(edge))
        s = float(sigma) if sigma is not None and not pd.isna(sigma) else 3.0
        if s <= 0:
            return None
        raw = 1.0 / (1.0 + math.exp(-e / s))
        return 0.5 + (raw - 0.5) * 0.60
    except Exception:
        return None


def _load(cfg: GameTotalRecalConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = pd.read_sql(SQL, conn, params={"cutoff": cutoff})
    if df.empty:
        return df
    rows = []
    for _, r in df.iterrows():
        edge = r.get("edge_total")
        if pd.isna(edge) or abs(float(edge)) < 1e-9:
            continue
        side = "over" if float(edge) > 0 else "under"
        p_over = r.get("p_total_over_clf")
        model_family = "total_over_clf"
        if p_over is None or pd.isna(p_over):
            p_side = r.get("win_prob_total")
            if p_side is None or pd.isna(p_side):
                p_side = _edge_sigma_prob(edge, r.get("sigma_q_total"))
            if p_side is None:
                continue
            p_over = p_side if side == "over" else 1.0 - float(p_side)
            model_family = "edge_sigma"
        p_side = float(p_over) if side == "over" else 1.0 - float(p_over)
        over_hit = float(r["actual_total"]) > float(r["market_total"])
        target = over_hit if side == "over" else not over_hit
        rows.append({
            "game_date_et": pd.to_datetime(r["game_date_et"]).date(),
            "market": "game_total",
            "side": side,
            "line_bucket": game_total_line_bucket(r.get("market_total")),
            "price_bucket": price_bucket(r.get("market_total_price")),
            "model_family": model_family,
            "raw_p_side": min(1 - 1e-6, max(1e-6, p_side)),
            "target": int(target),
        })
    return pd.DataFrame(rows)


def _fit_eval(train: pd.DataFrame, holdout: pd.DataFrame, all_rows: pd.DataFrame) -> dict | None:
    if len(train) < 20 or len(holdout) < 5:
        return None
    if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        return None
    x_train = np.array([logit(v) for v in train["raw_p_side"]], dtype=float).reshape(-1, 1)
    y_train = train["target"].astype(int).to_numpy()
    x_hold = np.array([logit(v) for v in holdout["raw_p_side"]], dtype=float)
    y_hold = holdout["target"].astype(int).to_numpy()
    raw_hold = holdout["raw_p_side"].astype(float).to_numpy()

    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(x_train, y_train)
    p_hold = np.clip(1.0 / (1.0 + np.exp(-(lr.coef_[0][0] * x_hold + lr.intercept_[0]))), 1e-6, 1 - 1e-6)

    x_all = np.array([logit(v) for v in all_rows["raw_p_side"]], dtype=float).reshape(-1, 1)
    y_all = all_rows["target"].astype(int).to_numpy()
    lr_all = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr_all.fit(x_all, y_all)

    raw_brier = float(brier_score_loss(y_hold, raw_hold))
    cal_brier = float(brier_score_loss(y_hold, p_hold))
    improvement = raw_brier - cal_brier
    base_weight = min(0.85, len(holdout) / (len(holdout) + 120.0))
    if improvement < -0.003:
        blend = base_weight * 0.20
    elif improvement < 0.001:
        blend = base_weight * 0.50
    else:
        blend = base_weight
    return {
        "method": "platt",
        "a": float(lr_all.coef_[0][0]),
        "b": float(lr_all.intercept_[0]),
        "blend_weight": float(blend),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "actual_rate_holdout": float(np.mean(y_hold)),
        "avg_raw_holdout": float(np.mean(raw_hold)),
        "avg_cal_holdout": float(np.mean(p_hold)),
        "brier_raw_holdout": raw_brier,
        "brier_cal_holdout": cal_brier,
        "log_loss_raw_holdout": float(log_loss(y_hold, np.clip(raw_hold, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "log_loss_cal_holdout": float(log_loss(y_hold, np.clip(p_hold, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "auc_raw_holdout": float(roc_auc_score(y_hold, raw_hold)) if len(np.unique(y_hold)) == 2 else None,
        "auc_cal_holdout": float(roc_auc_score(y_hold, p_hold)) if len(np.unique(y_hold)) == 2 else None,
    }


def _key(group_cols: tuple[str, ...], values: tuple) -> str:
    item = dict(zip(group_cols, values if isinstance(values, tuple) else (values,)))
    return calibration_key(
        item.get("market", "game_total"),
        item.get("side", "*"),
        item.get("line_bucket", "*"),
        item.get("price_bucket", "*"),
        item.get("model_family", "*"),
    )


def train(cfg: GameTotalRecalConfig) -> dict:
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
    specs = [
        (("market", "side", "line_bucket", "price_bucket", "model_family"), 80, 12),
        (("market", "side", "line_bucket", "model_family"), 100, 15),
        (("market", "side", "line_bucket"), 100, 15),
        (("market", "side", "model_family"), 120, 20),
        (("market", "side"), 120, 20),
    ]
    for group_cols, min_train, min_hold in specs:
        for values, sub in df.groupby(list(group_cols), dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            tr = sub.loc[train_mask.loc[sub.index]]
            ho = sub.loc[holdout_mask.loc[sub.index]]
            if len(tr) < min(min_train, cfg.min_train_rows) or len(ho) < min(min_hold, cfg.min_holdout_rows):
                continue
            rec = _fit_eval(tr, ho, sub)
            if rec is None:
                continue
            key = _key(group_cols, values)
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
    payload["backtest"].sort(key=lambda r: r["key"])
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB game-total side recalibrators")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="game_total_side_recalibrators.json")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--min-holdout-rows", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = train(GameTotalRecalConfig(
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
