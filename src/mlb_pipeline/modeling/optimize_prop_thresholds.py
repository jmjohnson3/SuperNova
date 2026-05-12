"""
Optimize prop thresholds (and min_ev when available) using train/holdout gating.

Writes model overrides consumed by predict_player_props.py:
  models/player_props/prop_thresholds.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import psycopg2

from .predict_player_props import PredictConfig

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass
class OptimizeConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_thresholds.json"
    lookback_days: int = 180
    holdout_days: int = 21
    min_train_bets: int = 40
    min_holdout_bets: int = 20


def _roi_pct(win_rate: float) -> float:
    # -110 pricing assumption: +100 on win, -110 on loss.
    return (win_rate * 100.0 - (1.0 - win_rate) * 110.0) / 110.0 * 100.0


def _load_rows(conn, cutoff: date) -> pd.DataFrame:
    q = """
    SELECT game_date_et, stat, edge, over_hit
    FROM bets.mlb_prop_predictions
    WHERE game_date_et >= %(cutoff)s
      AND edge IS NOT NULL
      AND over_hit IS NOT NULL
      AND stat IN (
        'pitcher_strikeouts',
        'batter_hits',
        'batter_total_bases',
        'batter_home_runs',
        'batter_walks'
      )
    """
    return pd.read_sql(q, conn, params={"cutoff": cutoff})


def _candidate_thresholds(edges: pd.Series) -> list[float]:
    vals = pd.to_numeric(edges, errors="coerce").abs().dropna()
    if vals.empty:
        return []
    base = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50]
    qs = [float(np.quantile(vals, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9)]
    cands = sorted({round(x, 3) for x in [*base, *qs] if x >= 0.0})
    return cands


def _pick_threshold(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    fallback: float,
    min_train_bets: int,
    min_holdout_bets: int,
) -> tuple[float, Dict]:
    cands = _candidate_thresholds(train["edge"])
    if not cands:
        return fallback, {"reason": "no_candidates"}

    best = None
    for t in cands:
        tr = train[train["edge"].abs() >= t]
        n = len(tr)
        if n < min_train_bets:
            continue
        wr = float(tr["over_hit"].mean())
        roi = _roi_pct(wr)
        rec = {"t": t, "n": n, "wr": wr, "roi": roi}
        if best is None or rec["roi"] > best["roi"] or (rec["roi"] == best["roi"] and rec["n"] > best["n"]):
            best = rec

    if best is None:
        return fallback, {"reason": "no_train_candidate_passed_min_bets"}

    ho = holdout[holdout["edge"].abs() >= best["t"]]
    n_ho = len(ho)
    if n_ho < min_holdout_bets:
        return fallback, {"reason": "holdout_too_small", "chosen_train": best, "holdout_n": n_ho}
    wr_ho = float(ho["over_hit"].mean())
    roi_ho = _roi_pct(wr_ho)
    if roi_ho <= 0:
        # Return a sentinel threshold (999) that prevents any bets from firing.
        # Using fallback here would deploy bets against a stat the holdout shows
        # has negative ROI — effectively guaranteed money loss.
        return 999.0, {
            "reason": "holdout_roi_non_positive",
            "chosen_train": best,
            "holdout": {"n": n_ho, "wr": wr_ho, "roi": roi_ho},
        }
    return float(best["t"]), {
        "reason": "accepted",
        "train": best,
        "holdout": {"n": n_ho, "wr": wr_ho, "roi": roi_ho},
    }


def optimize(cfg: OptimizeConfig) -> dict:
    conn = psycopg2.connect(cfg.pg_dsn)
    today = datetime.utcnow().date()
    cutoff = today - timedelta(days=cfg.lookback_days)
    df = _load_rows(conn, cutoff)
    conn.close()

    base = PredictConfig()
    out = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "min_train_bets": cfg.min_train_bets,
        "min_holdout_bets": cfg.min_holdout_bets,
        "threshold_strikeouts": base.threshold_strikeouts,
        "threshold_hits": base.threshold_hits,
        "threshold_total_bases": base.threshold_total_bases,
        "threshold_home_runs_over": base.threshold_home_runs_over,
        "threshold_home_runs_under": base.threshold_home_runs_under,
        "threshold_walks": base.threshold_walks,
        "threshold_clf": base.threshold_clf,
        "min_ev": base.min_ev,
        "_diagnostics": {},
    }
    if df.empty:
        out["_diagnostics"]["status"] = "no_rows"
        return out

    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    max_day = max(df["game_date_et"])
    split = max_day - timedelta(days=cfg.holdout_days)
    train = df[df["game_date_et"] < split].copy()
    holdout = df[df["game_date_et"] >= split].copy()

    mapping = {
        "pitcher_strikeouts": ("threshold_strikeouts", base.threshold_strikeouts),
        "batter_hits": ("threshold_hits", base.threshold_hits),
        "batter_total_bases": ("threshold_total_bases", base.threshold_total_bases),
        "batter_home_runs": ("threshold_home_runs_over", base.threshold_home_runs_over),
        "batter_walks": ("threshold_walks", base.threshold_walks),
    }

    for stat, (attr, fallback) in mapping.items():
        tr_s = train[train["stat"] == stat]
        ho_s = holdout[holdout["stat"] == stat]
        t_opt, diag = _pick_threshold(
            tr_s,
            ho_s,
            fallback=float(fallback),
            min_train_bets=cfg.min_train_bets,
            min_holdout_bets=cfg.min_holdout_bets,
        )
        out[attr] = t_opt
        out["_diagnostics"][stat] = diag

    # Keep HR over/under symmetric when auto-optimized from generic history.
    out["threshold_home_runs_under"] = out["threshold_home_runs_over"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Optimize prop thresholds with holdout gating")
    p.add_argument("--pg-dsn", default=_PG_DSN)
    p.add_argument("--model-dir", default=str(_MODEL_DIR))
    p.add_argument("--out-file", default="prop_thresholds.json")
    p.add_argument("--lookback-days", type=int, default=180)
    p.add_argument("--holdout-days", type=int, default=21)
    p.add_argument("--min-train-bets", type=int, default=40)
    p.add_argument("--min-holdout-bets", type=int, default=20)
    args = p.parse_args()

    cfg = OptimizeConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_bets=args.min_train_bets,
        min_holdout_bets=args.min_holdout_bets,
    )
    payload = optimize(cfg)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.model_dir / cfg.out_file
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote threshold overrides: {out_path}")


if __name__ == "__main__":
    main()
