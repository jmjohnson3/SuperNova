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

from .predict_player_props import PredictConfig, _ensure_schema

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


def _side_win_rate(df: pd.DataFrame) -> float:
    """Win rate for the model's side, not just raw over-hit rate."""
    if df.empty:
        return float("nan")
    over_hit = df["over_hit"].astype(bool)
    if "bet_side" in df.columns:
        side = df["bet_side"].astype("string").str.lower()
        edge = pd.to_numeric(df["edge"], errors="coerce")
        side = side.fillna(pd.Series(np.where(edge >= 0, "over", "under"), index=df.index))
        side_won = ((side == "over") & over_hit) | ((side == "under") & ~over_hit)
        return float(side_won.mean())
    edge = pd.to_numeric(df["edge"], errors="coerce")
    side_won = ((edge >= 0) & over_hit) | ((edge < 0) & ~over_hit)
    return float(side_won.mean())


def _load_rows(conn, cutoff: date) -> pd.DataFrame:
    q = """
    WITH typed AS (
      SELECT
        game_date_et,
        stat,
        edge::float AS raw_edge,
        over_hit,
        bet_side,
        line_bucket,
        ev::float AS ev,
        COALESCE(pred_prob_over, pred_value)::float AS p_over,
        CASE
          WHEN edge_type IS NOT NULL THEN edge_type
          WHEN book_line IS NOT NULL
           AND pred_value IS NOT NULL
           AND ABS((pred_value::float - book_line::float) - edge::float) <= 0.02
            THEN 'count'
          WHEN pred_value IS NOT NULL AND pred_value BETWEEN 0.0 AND 1.0
            THEN 'probability'
          ELSE 'unknown'
        END AS edge_type
      FROM bets.mlb_prop_predictions
      WHERE game_date_et >= %(cutoff)s
        AND edge IS NOT NULL
        AND over_hit IS NOT NULL
        AND stat IN (
          'pitcher_strikeouts',
          'batter_hits',
          'batter_total_bases',
          'batter_home_runs'
        )
    ),
    scored AS (
      SELECT
        *,
        CASE
          WHEN edge_type = 'probability' AND p_over IS NOT NULL THEN
            CASE
              WHEN p_over - %(breakeven)s > 0
               AND p_over - %(breakeven)s >= (1.0 - p_over) - %(breakeven)s
                THEN p_over - %(breakeven)s
              WHEN (1.0 - p_over) - %(breakeven)s > 0
                THEN -((1.0 - p_over) - %(breakeven)s)
              ELSE 0.0
            END
          ELSE raw_edge
        END AS effective_edge
      FROM typed
    )
    SELECT
      game_date_et,
      stat,
      effective_edge AS edge,
      over_hit,
      COALESCE(
        bet_side,
        CASE WHEN effective_edge > 0 THEN 'over'
             WHEN effective_edge < 0 THEN 'under'
             ELSE NULL END
      ) AS bet_side,
      line_bucket,
      ev,
      edge_type
    FROM scored
    WHERE effective_edge <> 0
    """
    return pd.read_sql(q, conn, params={"cutoff": cutoff, "breakeven": 0.524})


def _side_bucket_metrics(df: pd.DataFrame, min_bets: int) -> list[Dict]:
    if df.empty:
        return []
    out: list[Dict] = []
    work = df.copy()
    work["line_bucket"] = work["line_bucket"].fillna("unknown")
    for (stat, side, bucket, edge_type), g in work.groupby(["stat", "bet_side", "line_bucket", "edge_type"], dropna=False):
        n = len(g)
        if n < min_bets:
            continue
        wr = _side_win_rate(g)
        out.append({
            "stat": str(stat),
            "side": str(side),
            "line_bucket": str(bucket),
            "edge_type": str(edge_type),
            "n": n,
            "wr": wr,
            "roi": _roi_pct(wr),
        })
    return sorted(out, key=lambda r: (r["stat"], r["side"], r["line_bucket"], r["edge_type"]))


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
        return 999.0, {"reason": "no_candidates_disabled", "fallback": fallback}

    best = None
    for t in cands:
        tr = train[train["edge"].abs() >= t]
        n = len(tr)
        if n < min_train_bets:
            continue
        wr = _side_win_rate(tr)
        roi = _roi_pct(wr)
        rec = {"t": t, "n": n, "wr": wr, "roi": roi}
        if best is None or rec["roi"] > best["roi"] or (rec["roi"] == best["roi"] and rec["n"] > best["n"]):
            best = rec

    if best is None:
        return 999.0, {"reason": "no_train_candidate_passed_min_bets_disabled", "fallback": fallback}
    if best["roi"] <= 0:
        return 999.0, {
            "reason": "train_roi_non_positive_disabled",
            "chosen_train": best,
            "fallback": fallback,
        }

    ho = holdout[holdout["edge"].abs() >= best["t"]]
    n_ho = len(ho)
    if n_ho < min_holdout_bets:
        return 999.0, {
            "reason": "holdout_too_small_disabled",
            "chosen_train": best,
            "holdout_n": n_ho,
            "fallback": fallback,
        }
    wr_ho = _side_win_rate(ho)
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


def _pick_min_ev(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    fallback: float,
    min_train_bets: int,
    min_holdout_bets: int,
) -> tuple[float, Dict]:
    if train.empty or "ev" not in train.columns:
        return fallback, {"reason": "no_ev_rows"}
    tr_ev = pd.to_numeric(train["ev"], errors="coerce")
    cands = sorted({0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, *[
        round(float(np.quantile(tr_ev.dropna(), q)), 3)
        for q in (0.5, 0.6, 0.7, 0.8)
        if not tr_ev.dropna().empty
    ]})
    best = None
    for t in cands:
        tr = train[pd.to_numeric(train["ev"], errors="coerce") >= t]
        n = len(tr)
        if n < min_train_bets:
            continue
        wr = _side_win_rate(tr)
        roi = _roi_pct(wr)
        rec = {"t": t, "n": n, "wr": wr, "roi": roi}
        if best is None or rec["roi"] > best["roi"] or (rec["roi"] == best["roi"] and rec["n"] > best["n"]):
            best = rec
    if best is None:
        return fallback, {"reason": "no_train_candidate_passed_min_bets"}

    ho = holdout[pd.to_numeric(holdout["ev"], errors="coerce") >= best["t"]]
    n_ho = len(ho)
    if n_ho < min_holdout_bets:
        return fallback, {"reason": "holdout_too_small", "chosen_train": best, "holdout_n": n_ho}
    wr_ho = _side_win_rate(ho)
    roi_ho = _roi_pct(wr_ho)
    if roi_ho <= 0:
        return fallback, {
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
    _ensure_schema(conn)
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
        "threshold_strikeouts_over": base.threshold_strikeouts_over,
        "threshold_strikeouts_under": base.threshold_strikeouts_under,
        "threshold_hits": base.threshold_hits,
        "threshold_total_bases": base.threshold_total_bases,
        "threshold_total_bases_over": base.threshold_total_bases_over,
        "threshold_total_bases_under": base.threshold_total_bases_under,
        "threshold_home_runs_over": base.threshold_home_runs_over,
        "threshold_home_runs_under": base.threshold_home_runs_under,
        "threshold_clf": base.threshold_clf,
        "min_ev": base.min_ev,
        "_diagnostics": {},
    }
    if df.empty:
        out["_diagnostics"]["status"] = "no_rows"
        return out

    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["is_bettable"] = ~(
        df["stat"].isin(base.fd_over_only)
        & (df["bet_side"].astype("string").str.lower() == "under")
    )
    threshold_df = df[df["is_bettable"]].copy()
    if threshold_df.empty:
        out["_diagnostics"]["status"] = "no_bettable_rows"
        out["_diagnostics"]["side_bucket_metrics"] = _side_bucket_metrics(df, cfg.min_train_bets)
        return out

    max_day = max(threshold_df["game_date_et"])
    split = max_day - timedelta(days=cfg.holdout_days)
    train = threshold_df[threshold_df["game_date_et"] < split].copy()
    holdout = threshold_df[threshold_df["game_date_et"] >= split].copy()
    train_count = train[train["edge_type"] == "count"].copy()
    holdout_count = holdout[holdout["edge_type"] == "count"].copy()
    train_prob = train[train["edge_type"] == "probability"].copy()
    holdout_prob = holdout[holdout["edge_type"] == "probability"].copy()

    mapping = {
        "pitcher_strikeouts": ("threshold_strikeouts", base.threshold_strikeouts),
        "batter_hits":         ("threshold_hits",           base.threshold_hits),
        "batter_total_bases":  ("threshold_total_bases",    base.threshold_total_bases),
        "batter_home_runs":    ("threshold_home_runs_over", base.threshold_home_runs_over),
    }

    for stat, (attr, fallback) in mapping.items():
        tr_s = train_count[train_count["stat"] == stat]
        ho_s = holdout_count[holdout_count["stat"] == stat]
        t_opt, diag = _pick_threshold(
            tr_s,
            ho_s,
            fallback=float(fallback),
            min_train_bets=cfg.min_train_bets,
            min_holdout_bets=cfg.min_holdout_bets,
        )
        out[attr] = t_opt
        out["_diagnostics"][stat] = diag

    side_mapping = {
        "pitcher_strikeouts_over": (
            "pitcher_strikeouts",
            "over",
            "threshold_strikeouts_over",
            base.threshold_strikeouts,
        ),
        "pitcher_strikeouts_under": (
            "pitcher_strikeouts",
            "under",
            "threshold_strikeouts_under",
            base.threshold_strikeouts,
        ),
        "batter_total_bases_over": (
            "batter_total_bases",
            "over",
            "threshold_total_bases_over",
            base.threshold_total_bases,
        ),
        "batter_total_bases_under": (
            "batter_total_bases",
            "under",
            "threshold_total_bases_under",
            base.threshold_total_bases,
        ),
    }
    for diag_key, (stat, side, attr, fallback) in side_mapping.items():
        tr_s = train_count[
            (train_count["stat"] == stat)
            & (train_count["bet_side"].astype("string").str.lower() == side)
        ]
        ho_s = holdout_count[
            (holdout_count["stat"] == stat)
            & (holdout_count["bet_side"].astype("string").str.lower() == side)
        ]
        t_opt, diag = _pick_threshold(
            tr_s,
            ho_s,
            fallback=float(fallback),
            min_train_bets=cfg.min_train_bets,
            min_holdout_bets=cfg.min_holdout_bets,
        )
        out[attr] = t_opt
        out["_diagnostics"][diag_key] = diag

    t_clf, diag_clf = _pick_threshold(
        train_prob,
        holdout_prob,
        fallback=float(base.threshold_clf),
        min_train_bets=cfg.min_train_bets,
        min_holdout_bets=cfg.min_holdout_bets,
    )
    out["threshold_clf"] = t_clf
    out["_diagnostics"]["classifier_probability"] = diag_clf

    ev_train = train[train["ev"].notna()].copy()
    ev_holdout = holdout[holdout["ev"].notna()].copy()
    min_ev, diag_ev = _pick_min_ev(
        ev_train,
        ev_holdout,
        fallback=float(base.min_ev),
        min_train_bets=cfg.min_train_bets,
        min_holdout_bets=cfg.min_holdout_bets,
    )
    out["min_ev"] = min_ev
    out["_diagnostics"]["min_ev"] = diag_ev
    out["_diagnostics"]["side_bucket_metrics"] = _side_bucket_metrics(threshold_df, cfg.min_train_bets)
    out["_diagnostics"]["excluded_unbettable_rows"] = int((~df["is_bettable"]).sum())

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

    # Apply manual overrides — any key under "_manual_overrides" in the existing
    # file wins over the optimizer's computed value.  This lets you pin a threshold
    # (e.g. after a model retrain) without waiting for the holdout window to refill.
    out_path = cfg.model_dir / cfg.out_file
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for k, v in existing.get("_manual_overrides", {}).items():
                if k in payload:
                    payload[k] = v
            if existing.get("_manual_overrides"):
                payload["_manual_overrides"] = existing["_manual_overrides"]
        except Exception:
            pass

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote threshold overrides: {out_path}")


if __name__ == "__main__":
    main()
