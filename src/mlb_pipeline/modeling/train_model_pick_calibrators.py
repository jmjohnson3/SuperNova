"""Train line/price-aware direct win classifiers for MLB model picks.

These are meta-calibrators: they learn whether the model-selected side wins
using the offered line, price, model edge/probability, EV, market, side, and
bucket features. They do not require new data feeds.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

log = logging.getLogger("mlb_pipeline.modeling.train_model_pick_calibrators")

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "model_pick_calibrators"


@dataclass(frozen=True)
class CalibratorConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 150
    min_holdout_rows: int = 25
    min_ev: float = 0.0
    random_state: int = 42


PROP_SQL = """
SELECT
    game_date_et,
    'prop'::text AS source,
    stat AS market,
    stat,
    bet_side AS side,
    book_line::float AS market_line,
    bet_price::float AS market_price,
    pred_value::float AS pred_value,
    pred_count::float AS pred_count,
    pred_prob_over::float AS pred_prob_over,
    edge::float AS edge,
    ev::float AS ev,
    breakeven_prob::float AS breakeven_prob,
    edge_type,
    model_family,
    line_bucket,
    over_hit
FROM bets.mlb_prop_predictions
WHERE game_date_et >= %(cutoff)s
  AND over_hit IS NOT NULL
  AND bet_side IN ('over','under')
  AND book_line IS NOT NULL
  AND edge IS NOT NULL
  AND stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
"""


GAME_SQL = """
SELECT
    game_date_et,
    'game'::text AS source,
    game_slug,
    pred_run_diff::float AS pred_run_diff,
    pred_total::float AS pred_total,
    market_run_line::float AS market_run_line,
    market_total::float AS market_total,
    edge_run_line::float AS edge_run_line,
    edge_total::float AS edge_total,
    market_rl_price::float AS market_rl_price,
    market_total_price::float AS market_total_price,
    win_prob_rl::float AS win_prob_rl,
    win_prob_total::float AS win_prob_total,
    p_home_cover_clf::float AS p_home_cover_clf,
    p_total_over_clf::float AS p_total_over_clf,
    sigma_q_rl::float AS sigma_q_rl,
    sigma_q_total::float AS sigma_q_total,
    actual_run_diff::float AS actual_run_diff,
    actual_total::float AS actual_total
FROM bets.mlb_game_predictions
WHERE game_date_et >= %(cutoff)s
  AND actual_run_diff IS NOT NULL
  AND actual_total IS NOT NULL
"""


def _american_to_prob(price) -> float | None:
    try:
        p = float(price)
    except Exception:
        return None
    if p == 0 or np.isnan(p):
        return None
    return 100.0 / (p + 100.0) if p > 0 else abs(p) / (abs(p) + 100.0)


def _american_profit_mult(price) -> float | None:
    try:
        p = float(price)
    except Exception:
        return None
    if p == 0 or np.isnan(p):
        return None
    return p / 100.0 if p > 0 else 100.0 / abs(p)


def _line_bucket(market: str, line) -> str:
    try:
        line = float(line)
    except Exception:
        return "missing_line"
    if market == "pitcher_strikeouts":
        if line < 4.5:
            return "K <4.5"
        if line < 6.5:
            return "K 4.5-6"
        if line < 8.5:
            return "K 6.5-8"
        return "K 8.5+"
    if market == "batter_total_bases":
        if line < 1.0:
            return "TB 0.5"
        if line < 2.0:
            return "TB 1.5"
        return "TB 2.5+"
    if market == "batter_hits":
        if line < 1.0:
            return "H 0.5"
        if line < 2.0:
            return "H 1.5"
        return "H 2.5+"
    if market == "batter_home_runs":
        return "HR 0.5" if line < 1.0 else "HR 1.5+"
    if market == "run_line":
        return f"RL {line:+.1f}"
    if market == "total":
        if line < 8.0:
            return "total <8"
        if line < 9.5:
            return "total 8-9"
        if line < 11.0:
            return "total 9.5-10.5"
        return "total 11+"
    return "other"


def _price_bucket(price) -> str:
    p = _american_to_prob(price)
    if p is None:
        return "missing_price"
    price = float(price)
    if price > 0:
        return "plus_money"
    if price >= -129:
        return "fair_lay"
    if price >= -149:
        return "lay_130_149"
    if price >= -180:
        return "lay_150_180"
    return "heavy_lay"


def _ev_per_unit(p_win, price) -> float | None:
    try:
        p = float(p_win)
    except Exception:
        return None
    mult = _american_profit_mult(price)
    if mult is None or np.isnan(p):
        return None
    return p * mult - (1.0 - p)


def _load_frames(cfg: CalibratorConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        props = pd.read_sql(PROP_SQL, conn, params={"cutoff": cutoff})
        games = pd.read_sql(GAME_SQL, conn, params={"cutoff": cutoff})

    rows: list[dict] = []
    for _, r in props.iterrows():
        side = str(r.get("side") or "").lower()
        if side not in {"over", "under"}:
            continue
        p_over = r.get("pred_prob_over")
        p_side = p_over if side == "over" else (1.0 - p_over if pd.notna(p_over) else np.nan)
        won = bool(r["over_hit"]) if side == "over" else not bool(r["over_hit"])
        ev = r.get("ev")
        rows.append({
            "game_date_et": r["game_date_et"],
            "source": "prop",
            "market": r["market"],
            "stat": r["stat"],
            "side": side,
            "market_side": f"prop:{r['market']}/{side}",
            "market_line": r.get("market_line"),
            "market_price": r.get("market_price"),
            "pred_value": r.get("pred_value"),
            "pred_count": r.get("pred_count"),
            "model_prob": p_side,
            "edge": r.get("edge"),
            "abs_edge": abs(float(r.get("edge") or 0.0)),
            "ev": ev,
            "breakeven_prob": r.get("breakeven_prob"),
            "edge_type": r.get("edge_type"),
            "model_family": r.get("model_family"),
            "line_bucket": r.get("line_bucket") or _line_bucket(str(r["market"]), r.get("market_line")),
            "price_bucket": _price_bucket(r.get("market_price")),
            "target": int(won),
        })

    for _, r in games.iterrows():
        edge_rl = r.get("edge_run_line")
        line_rl = r.get("market_run_line")
        if pd.notna(edge_rl) and pd.notna(line_rl) and abs(float(edge_rl)) > 1e-9:
            side = "home" if float(edge_rl) > 0 else "away"
            home_covered = float(r["actual_run_diff"]) > -float(line_rl)
            p_home = r.get("p_home_cover_clf")
            p_side = p_home if side == "home" else (1.0 - p_home if pd.notna(p_home) else r.get("win_prob_rl"))
            price = r.get("market_rl_price")
            rows.append({
                "game_date_et": r["game_date_et"],
                "source": "game",
                "market": "run_line",
                "stat": None,
                "side": side,
                "market_side": f"game:run_line/{side}",
                "market_line": line_rl,
                "market_price": price,
                "pred_value": r.get("pred_run_diff"),
                "pred_count": np.nan,
                "model_prob": p_side,
                "edge": edge_rl,
                "abs_edge": abs(float(edge_rl)),
                "ev": _ev_per_unit(p_side, price),
                "breakeven_prob": _american_to_prob(price),
                "edge_type": "run_line",
                "model_family": "game_regression",
                "line_bucket": _line_bucket("run_line", line_rl),
                "price_bucket": _price_bucket(price),
                "target": int(home_covered if side == "home" else not home_covered),
            })
        edge_total = r.get("edge_total")
        line_total = r.get("market_total")
        if pd.notna(edge_total) and pd.notna(line_total) and abs(float(edge_total)) > 1e-9:
            side = "over" if float(edge_total) > 0 else "under"
            over_hit = float(r["actual_total"]) > float(line_total)
            p_over = r.get("p_total_over_clf")
            p_side = p_over if side == "over" else (1.0 - p_over if pd.notna(p_over) else r.get("win_prob_total"))
            price = r.get("market_total_price")
            rows.append({
                "game_date_et": r["game_date_et"],
                "source": "game",
                "market": "total",
                "stat": None,
                "side": side,
                "market_side": f"game:total/{side}",
                "market_line": line_total,
                "market_price": price,
                "pred_value": r.get("pred_total"),
                "pred_count": np.nan,
                "model_prob": p_side,
                "edge": edge_total,
                "abs_edge": abs(float(edge_total)),
                "ev": _ev_per_unit(p_side, price),
                "breakeven_prob": _american_to_prob(price),
                "edge_type": "total",
                "model_family": "game_regression",
                "line_bucket": _line_bucket("total", line_total),
                "price_bucket": _price_bucket(price),
                "target": int(over_hit if side == "over" else not over_hit),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df = df[pd.to_numeric(df["ev"], errors="coerce").fillna(0.0) >= cfg.min_ev].copy()
    return df


def _feature_frame(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str], Dict[str, float]]:
    work = df.copy()
    for col in ["market_line", "market_price", "pred_value", "pred_count", "model_prob", "edge", "abs_edge", "ev", "breakeven_prob"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work["price_profit_mult"] = work["market_price"].map(_american_profit_mult)
    work["edge_x_price"] = work["abs_edge"] * work["breakeven_prob"].fillna(0.524)
    work["prob_minus_breakeven"] = work["model_prob"] - work["breakeven_prob"]
    cats = ["source", "market", "side", "market_side", "line_bucket", "price_bucket", "edge_type", "model_family"]
    X = pd.concat(
        [
            work[[
                "market_line",
                "market_price",
                "pred_value",
                "pred_count",
                "model_prob",
                "edge",
                "abs_edge",
                "ev",
                "breakeven_prob",
                "price_profit_mult",
                "edge_x_price",
                "prob_minus_breakeven",
            ]],
            pd.get_dummies(work[cats].fillna("unknown"), columns=cats, dtype=float),
        ],
        axis=1,
    )
    if feature_cols is None:
        feature_cols = list(X.columns)
    X = X.reindex(columns=feature_cols)
    medians = {
        col: float(X[col].median()) if pd.notna(X[col].median()) else 0.0
        for col in feature_cols
    }
    X = X.fillna(medians)
    return X, feature_cols, medians


def _model(cfg: CalibratorConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=3.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def _metrics(y_true: pd.Series, p: np.ndarray) -> dict:
    y = y_true.astype(int).to_numpy()
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "accuracy_50": float(accuracy_score(y, pred)) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])) if len(y) else None,
        "avg_prob": float(np.mean(p)) if len(y) else None,
        "cal_error": float(np.mean(y) - np.mean(p)) if len(y) else None,
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return out


def _slice_metrics(df: pd.DataFrame, y: pd.Series, p: np.ndarray, col: str, min_rows: int = 20) -> list[dict]:
    rows = []
    tmp = df[[col]].copy()
    tmp["target"] = y.to_numpy()
    tmp["p"] = p
    for key, sub in tmp.groupby(col, dropna=False):
        if len(sub) < min_rows:
            continue
        rows.append({"bucket": str(key), **_metrics(sub["target"], sub["p"].to_numpy())})
    rows.sort(key=lambda r: (-r["n"], r["bucket"]))
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def train(cfg: CalibratorConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load_frames(cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "min_ev": cfg.min_ev,
        "rows": int(len(df)),
        "global": {},
        "market_side_backtest": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        return payload

    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train_mask = df["game_date_et"] < split
    holdout_mask = df["game_date_et"] >= split
    if int(train_mask.sum()) < cfg.min_train_rows or int(holdout_mask.sum()) < cfg.min_holdout_rows:
        payload["status"] = "not_enough_rows"
        payload["train_rows"] = int(train_mask.sum())
        payload["holdout_rows"] = int(holdout_mask.sum())
        return payload

    X_train, feature_cols, medians = _feature_frame(df.loc[train_mask])
    X_holdout, _, _ = _feature_frame(df.loc[holdout_mask], feature_cols)
    y_train = df.loc[train_mask, "target"].astype(int)
    y_holdout = df.loc[holdout_mask, "target"].astype(int)
    model = _model(cfg)
    model.fit(X_train, y_train, verbose=False)
    p_holdout = model.predict_proba(X_holdout)[:, 1]
    payload["global"] = {
        "status": "trained",
        "train_rows": int(train_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "holdout": _metrics(y_holdout, p_holdout),
        "by_market_side": _slice_metrics(df.loc[holdout_mask], y_holdout, p_holdout, "market_side"),
        "by_line_bucket": _slice_metrics(df.loc[holdout_mask], y_holdout, p_holdout, "line_bucket"),
        "by_price_bucket": _slice_metrics(df.loc[holdout_mask], y_holdout, p_holdout, "price_bucket"),
    }

    X_all, feature_cols, medians = _feature_frame(df, feature_cols)
    final_model = _model(cfg)
    final_model.fit(X_all, df["target"].astype(int), verbose=False)
    final_model.save_model(str(cfg.model_dir / "model_pick_win_xgb.json"))
    (cfg.model_dir / "model_pick_feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (cfg.model_dir / "model_pick_feature_medians.json").write_text(json.dumps(medians, indent=2), encoding="utf-8")

    for market_side, sub in df.groupby("market_side"):
        dates = sub["game_date_et"]
        split_s = max(dates) - timedelta(days=cfg.holdout_days)
        train_s = dates < split_s
        holdout_s = dates >= split_s
        if int(train_s.sum()) < cfg.min_train_rows or int(holdout_s.sum()) < cfg.min_holdout_rows:
            continue
        Xs_train, cols_s, meds_s = _feature_frame(sub.loc[train_s])
        Xs_holdout, _, _ = _feature_frame(sub.loc[holdout_s], cols_s)
        ys_train = sub.loc[train_s, "target"].astype(int)
        ys_holdout = sub.loc[holdout_s, "target"].astype(int)
        if ys_train.nunique() < 2 or ys_holdout.nunique() < 2:
            continue
        m = _model(cfg)
        m.fit(Xs_train, ys_train, verbose=False)
        ps = m.predict_proba(Xs_holdout)[:, 1]
        rec = {
            "market_side": market_side,
            "train_rows": int(train_s.sum()),
            "holdout_rows": int(holdout_s.sum()),
            "holdout": _metrics(ys_holdout, ps),
        }
        payload["market_side_backtest"].append(rec)
        Xs_all, cols_s, meds_s = _feature_frame(sub, cols_s)
        mf = _model(cfg)
        mf.fit(Xs_all, sub["target"].astype(int), verbose=False)
        stem = _safe_name(market_side)
        mf.save_model(str(cfg.model_dir / f"{stem}_win_xgb.json"))
        (cfg.model_dir / f"{stem}_feature_columns.json").write_text(json.dumps(cols_s), encoding="utf-8")
        (cfg.model_dir / f"{stem}_feature_medians.json").write_text(json.dumps(meds_s, indent=2), encoding="utf-8")

    (cfg.model_dir / "model_pick_calibrator_backtest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB model-pick direct win calibrators")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=150)
    parser.add_argument("--min-holdout-rows", type=int, default=25)
    parser.add_argument("--min-ev", type=float, default=0.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = train(CalibratorConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_ev=args.min_ev,
    ))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
