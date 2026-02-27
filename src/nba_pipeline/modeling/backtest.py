"""
backtest.py — Walk-forward backtesting and calibration report for game + prop models.

Usage:
    python -m nba_pipeline.modeling.backtest              # game spread/total only
    python -m nba_pipeline.modeling.backtest --props      # also run player prop backtest
    python -m nba_pipeline.modeling.backtest --props-only # skip games, only props
    python -m nba_pipeline.modeling.backtest --min-train-days 90
    python -m nba_pipeline.modeling.backtest --output backtest_results.csv

What it does:
- Replays the same walk-forward CV used in training (no data leakage)
- Uses saved Optuna best params if available, else falls back to defaults
- Reports per-fold and overall metrics:
    Spread:  MAE, RMSE, bias, directional accuracy, calibration slope
    Total:   MAE, RMSE, bias, calibration slope
    ATS:     cover rate + ROI at -110 (only when market spread data is available)
    Props:   MAE/RMSE/bias per stat (PTS, REB, AST), calibration bins
- Prints calibration bins (predicted range vs actual outcome rate)
- Optionally writes all per-game predictions to CSV for further analysis
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2

from .train_game_models import (
    SPREAD_OBJECTIVE,
    TOTAL_OBJECTIVE,
    TrainConfig,
    apply_fill,
    build_model,
    fit_fill_stats,
    load_training_frame,
    make_xy_raw,
    temporal_eval_split,
    walk_forward_folds,
)

log = logging.getLogger("nba_pipeline.modeling.backtest")

MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    min_train_days: int = 60
    test_window_days: int = 7
    step_days: int = 7
    # ATS edge threshold for "high confidence" flagging
    hc_edge_pts: float = 3.0
    output_csv: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_best_params() -> tuple[Dict, Dict]:
    """Load saved Optuna params or return empty dicts (use defaults)."""
    p = MODEL_DIR / "optuna_best_params.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("spread_params", {}), data.get("total_params", {})
    return {}, {}


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean prediction minus mean actuals (positive = over-predicts)."""
    return float(np.mean(y_pred) - np.mean(y_true))


def _directional(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of games where predicted sign of margin matches actual."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def _calib_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Slope from linear fit y_true ~ a + b*y_pred.
    b=1 is perfect; b<1 means predictions are too spread out.
    """
    if len(y_pred) < 3 or float(np.std(y_pred)) < 1e-9:
        return float("nan")
    b, _ = np.polyfit(y_pred, y_true, deg=1)
    return float(b)


def _ats_stats(
    y_true_margin: np.ndarray,
    y_pred_margin: np.ndarray,
    market_spread: np.ndarray,
    hc_edge: float = 3.0,
) -> Dict:
    """
    ATS (against the spread) statistics.

    A bet is placed when |pred_margin - market_spread| >= hc_edge.
    We bet the side our model favors vs the spread.

    At -110 juice:
        ROI = (wins*100 - losses*110) / (n_bets*110)
    """
    valid = ~np.isnan(market_spread.astype(float))
    if valid.sum() == 0:
        return {"n": 0}

    ms = market_spread[valid].astype(float)
    yt = y_true_margin[valid]
    yp = y_pred_margin[valid]

    # Did the home team cover?
    home_covered = yt > ms

    # Edge: how much our model disagrees with the spread
    edge = yp - ms

    # All games with spread: did we pick the right side?
    pred_home = edge > 0
    all_correct = pred_home == home_covered
    n_all = len(all_correct)
    pct_all = float(all_correct.mean()) if n_all > 0 else float("nan")
    roi_all = (
        float((all_correct.sum() * 100 - (~all_correct).sum() * 110) / (n_all * 110))
        if n_all > 0
        else float("nan")
    )

    # High-confidence subset: |edge| >= hc_edge
    hc_mask = np.abs(edge) >= hc_edge
    n_hc = int(hc_mask.sum())
    if n_hc >= 2:
        hc_correct = all_correct[hc_mask]
        pct_hc = float(hc_correct.mean())
        roi_hc = float(
            (hc_correct.sum() * 100 - (~hc_correct).sum() * 110) / (n_hc * 110)
        )
    else:
        pct_hc = float("nan")
        roi_hc = float("nan")

    return {
        "n": n_all,
        "pct": pct_all,
        "roi": roi_all,
        "n_hc": n_hc,
        "pct_hc": pct_hc,
        "roi_hc": roi_hc,
    }


def _calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5,
    label: str = "margin",
) -> str:
    """
    Split predictions into n_bins quantile buckets and report mean actual vs mean pred.
    Useful for detecting systematic over/under-prediction in certain ranges.
    """
    if len(y_pred) < n_bins * 3:
        return f"  (not enough data for {label} calibration bins)"

    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, labels=False, duplicates="drop")

    lines = [f"\n  {label} calibration bins (pred range -> mean pred vs mean actual):"]
    for b in sorted(df["bin"].dropna().unique()):
        grp = df[df["bin"] == b]
        lo = grp["pred"].min()
        hi = grp["pred"].max()
        mp = grp["pred"].mean()
        ma = grp["actual"].mean()
        diff = ma - mp
        lines.append(
            f"    [{lo:+.1f} .. {hi:+.1f}]  n={len(grp):4d}  "
            f"pred={mp:+.2f}  actual={ma:+.2f}  diff={diff:+.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------
def run_backtest(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Run walk-forward CV, collect all test-fold predictions, return DataFrame.

    Columns: game_slug, game_date_et, home_team_abbr, away_team_abbr,
             actual_margin, pred_margin, actual_total, pred_total,
             market_spread_home, market_total
    """
    train_cfg = TrainConfig(
        pg_dsn=cfg.pg_dsn,
        run_optuna=False,
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
    )

    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = load_training_frame(conn)

    log.info(
        "Loaded %d rows | %s to %s",
        len(df),
        df["game_date_et"].min().date(),
        df["game_date_et"].max().date(),
    )

    # Build feature matrix once (NaNs preserved)
    X_raw, y_spread, y_total = make_xy_raw(df)
    feature_cols = list(X_raw.columns)

    best_spread_params, best_total_params = _load_best_params()
    if best_spread_params:
        log.info("Using saved Optuna spread params.")
    if best_total_params:
        log.info("Using saved Optuna total params.")

    folds = walk_forward_folds(
        df,
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
    )
    if not folds:
        raise RuntimeError("No folds produced. Reduce --min-train-days or add more data.")

    log.info(
        "Walk-forward: %d folds | min_train=%d test_window=%d step=%d",
        len(folds),
        cfg.min_train_days,
        cfg.test_window_days,
        cfg.step_days,
    )

    # Accumulate per-game results
    records: List[Dict] = []

    for k, (train_end, test_end) in enumerate(folds, start=1):
        train_mask = df["game_date_et"] < train_end
        test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < 30 or n_test == 0:
            continue

        X_train_raw = X_raw.loc[train_mask]
        X_test_raw = X_raw.loc[test_mask]

        medians = fit_fill_stats(X_train_raw)
        X_train = apply_fill(X_train_raw, medians, feature_cols)
        X_test = apply_fill(X_test_raw, medians, feature_cols)

        y_spread_train = y_spread.loc[train_mask]
        y_total_train = y_total.loc[train_mask]

        # Temporal eval split for early stopping
        train_dates = df.loc[train_mask, "game_date_et"]
        fit_rel, eval_rel = temporal_eval_split(train_dates)

        spread_model = build_model(
            train_cfg, params_override=best_spread_params, objective=SPREAD_OBJECTIVE
        )
        total_model = build_model(
            train_cfg, params_override=best_total_params, objective=TOTAL_OBJECTIVE
        )

        spread_model.fit(
            X_train.iloc[fit_rel],
            y_spread_train.iloc[fit_rel],
            eval_set=[(X_train.iloc[eval_rel], y_spread_train.iloc[eval_rel])],
            verbose=False,
        )
        total_model.fit(
            X_train.iloc[fit_rel],
            y_total_train.iloc[fit_rel],
            eval_set=[(X_train.iloc[eval_rel], y_total_train.iloc[eval_rel])],
            verbose=False,
        )

        spread_pred = spread_model.predict(X_test)
        total_pred = total_model.predict(X_test)

        test_rows = df.loc[test_mask].reset_index(drop=True)

        for i in range(n_test):
            row = test_rows.iloc[i]
            records.append(
                {
                    "fold": k,
                    "game_date_et": row["game_date_et"],
                    "game_slug": row["game_slug"],
                    "home_team_abbr": row["home_team_abbr"],
                    "away_team_abbr": row["away_team_abbr"],
                    "actual_margin": float(row["margin"]),
                    "pred_margin": float(spread_pred[i]),
                    "actual_total": float(row["total_points"]),
                    "pred_total": float(total_pred[i]),
                    "market_spread_home": (
                        float(row["market_spread_home"])
                        if pd.notna(row.get("market_spread_home"))
                        else float("nan")
                    ),
                    "market_total": (
                        float(row["market_total"])
                        if pd.notna(row.get("market_total"))
                        else float("nan")
                    ),
                    "n_train": n_train,
                }
            )

        fold_spread_mae = float(np.mean(np.abs(y_spread.loc[test_mask].values - spread_pred)))
        fold_total_mae = float(np.mean(np.abs(y_total.loc[test_mask].values - total_pred)))
        log.info(
            "Fold %3d | train_end=%s  n_train=%4d  n_test=%3d | "
            "SPREAD MAE=%.3f | TOTAL MAE=%.3f",
            k,
            train_end.date(),
            n_train,
            n_test,
            fold_spread_mae,
            fold_total_mae,
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(results: pd.DataFrame, cfg: BacktestConfig) -> None:
    if results.empty:
        print("No results to report.")
        return

    n = len(results)
    print(f"\n{'='*65}")
    print(f"  BACKTEST CALIBRATION REPORT  ({n} games, {results['fold'].nunique()} folds)")
    print(f"{'='*65}")
    print(
        f"  Date range: {results['game_date_et'].min().date()} to "
        f"{results['game_date_et'].max().date()}"
    )

    y_margin = results["actual_margin"].values
    p_margin = results["pred_margin"].values
    y_total = results["actual_total"].values
    p_total = results["pred_total"].values

    # ---- Spread / Margin ----
    spread_mae = float(np.mean(np.abs(y_margin - p_margin)))
    spread_rmse = float(np.sqrt(np.mean((y_margin - p_margin) ** 2)))
    spread_bias = _bias(y_margin, p_margin)
    spread_dir = _directional(y_margin, p_margin)
    spread_slope = _calib_slope(y_margin, p_margin)

    print(f"\n  SPREAD (home margin)")
    print(f"    MAE  = {spread_mae:.3f} pts")
    print(f"    RMSE = {spread_rmse:.3f} pts")
    print(f"    Bias = {spread_bias:+.3f} pts  (+ = over-predicts home)")
    print(f"    Directional accuracy = {spread_dir:.1%}  (did we pick the winner?)")
    print(f"    Calibration slope    = {spread_slope:.3f}  (ideal = 1.00)")

    # ---- Total ----
    total_mae = float(np.mean(np.abs(y_total - p_total)))
    total_rmse = float(np.sqrt(np.mean((y_total - p_total) ** 2)))
    total_bias = _bias(y_total, p_total)
    total_slope = _calib_slope(y_total, p_total)

    print(f"\n  TOTAL (combined score)")
    print(f"    MAE  = {total_mae:.3f} pts")
    print(f"    RMSE = {total_rmse:.3f} pts")
    print(f"    Bias = {total_bias:+.3f} pts  (+ = over-predicts total)")
    print(f"    Calibration slope    = {total_slope:.3f}  (ideal = 1.00)")

    # ---- ATS ----
    market_spread = results["market_spread_home"].values
    market_total_vals = results["market_total"].values
    n_market = int(np.sum(~np.isnan(market_spread.astype(float))))

    print(f"\n  ATS (against the spread)  -- {n_market} games with market data")
    if n_market >= 5:
        ats = _ats_stats(y_margin, p_margin, market_spread, cfg.hc_edge_pts)
        print(
            f"    All games:          n={ats['n']:4d}  "
            f"cover={ats['pct']:.1%}  ROI={ats['roi']:+.1%}"
        )
        if ats["n_hc"] >= 2:
            print(
                f"    High-conf (≥{cfg.hc_edge_pts:.0f}pt edge): "
                f"n={ats['n_hc']:4d}  "
                f"cover={ats['pct_hc']:.1%}  ROI={ats['roi_hc']:+.1%}"
            )
        else:
            print(f"    High-conf (≥{cfg.hc_edge_pts:.0f}pt edge): fewer than 2 qualifying games.")

        # Over/Under ATS
        n_total_mkt = int(np.sum(~np.isnan(market_total_vals.astype(float))))
        if n_total_mkt >= 5:
            valid_t = ~np.isnan(market_total_vals.astype(float))
            mt = market_total_vals[valid_t].astype(float)
            at = y_total[valid_t]
            pt = p_total[valid_t]
            pred_over = pt > mt
            actual_over = at > mt
            ou_correct = pred_over == actual_over
            n_ou = len(ou_correct)
            ou_pct = float(ou_correct.mean())
            ou_roi = float(
                (ou_correct.sum() * 100 - (~ou_correct).sum() * 110) / (n_ou * 110)
            )
            print(
                f"    Over/Under:         n={n_ou:4d}  "
                f"correct={ou_pct:.1%}  ROI={ou_roi:+.1%}"
            )
    else:
        print(
            f"    No market data in test folds. "
            f"Run crawler to get odds, then retrain."
        )

    # ---- Calibration bins ----
    print(_calibration_bins(y_margin, p_margin, n_bins=5, label="spread"))
    print(_calibration_bins(y_total, p_total, n_bins=5, label="total"))

    # ---- Season-level check ----
    results["season_year"] = pd.to_datetime(results["game_date_et"]).dt.year
    print(f"\n  BY SEASON YEAR (spread MAE | total MAE):")
    for yr, grp in results.groupby("season_year"):
        sm = float(np.mean(np.abs(grp["actual_margin"] - grp["pred_margin"])))
        tm = float(np.mean(np.abs(grp["actual_total"] - grp["pred_total"])))
        print(f"    {yr}: n={len(grp):4d}  SPREAD={sm:.3f}  TOTAL={tm:.3f}")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# Player prop backtest
# ---------------------------------------------------------------------------
PROP_BEST_PARAMS_PATH = MODEL_DIR / "player_props" / "optuna_best_params.json"

SQL_PROP_TRAIN = """
SELECT *
FROM features.player_training_features
WHERE points IS NOT NULL
  AND rebounds IS NOT NULL
  AND assists IS NOT NULL
  AND n_games_prev_10 >= 3
  AND (min_avg_10 IS NULL OR min_avg_10 >= 10.0)
ORDER BY game_date_et, game_slug, player_id
"""


def _load_prop_best_params() -> Dict:
    if PROP_BEST_PARAMS_PATH.exists():
        return json.loads(PROP_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return {}


def run_prop_backtest(cfg: BacktestConfig) -> pd.DataFrame:
    """
    Walk-forward CV for player prop models (PTS, REB, AST).
    Returns DataFrame with per-player-game predictions.
    """
    from . import train_player_prop_models as ptm

    prop_cfg = ptm.TrainConfig(
        pg_dsn=cfg.pg_dsn,
        run_optuna=False,
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
    )

    engine = __import__("sqlalchemy").create_engine(cfg.pg_dsn)
    with engine.connect() as conn:
        df = pd.read_sql(__import__("sqlalchemy").text(SQL_PROP_TRAIN), conn)
    df["game_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce")
    df = df.sort_values(["game_date_et", "game_slug", "player_id"]).reset_index(drop=True)

    log.info(
        "Props: loaded %d player-game rows | %s to %s",
        len(df),
        df["game_date_et"].min().date(),
        df["game_date_et"].max().date(),
    )

    X_raw, y_pts, y_reb, y_ast = ptm.make_xy_raw(df)
    feature_cols = list(X_raw.columns)

    best_params = _load_prop_best_params()
    if best_params:
        log.info("Props: using saved Optuna params.")

    folds = ptm.walk_forward_folds(
        df,
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days,
        step_days=cfg.step_days,
    )
    if not folds:
        raise RuntimeError("No prop folds produced.")

    log.info("Props walk-forward: %d folds", len(folds))

    records: List[Dict] = []

    for k, (train_end, test_end) in enumerate(folds, start=1):
        train_mask = df["game_date_et"] < train_end
        test_mask = (df["game_date_et"] >= train_end) & (df["game_date_et"] < test_end)

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < 100 or n_test == 0:
            continue

        X_train_raw = X_raw.loc[train_mask]
        X_test_raw = X_raw.loc[test_mask]

        medians = ptm.fit_fill_stats(X_train_raw)
        X_train = ptm.apply_fill(X_train_raw, medians, feature_cols)
        X_test = ptm.apply_fill(X_test_raw, medians, feature_cols)

        train_dates = df.loc[train_mask, "game_date_et"]
        fit_rel, eval_rel = ptm.temporal_eval_split(train_dates)

        pts_model = ptm.build_model(prop_cfg, huber_slope=prop_cfg.huber_slope_pts, params_override=best_params)
        reb_model = ptm.build_model(prop_cfg, huber_slope=prop_cfg.huber_slope_reb, params_override=best_params)
        ast_model = ptm.build_model(prop_cfg, huber_slope=prop_cfg.huber_slope_ast, params_override=best_params)

        def _fit(model, y_target):
            y_tr = y_target.loc[train_mask]
            model.fit(
                X_train.iloc[fit_rel], y_tr.iloc[fit_rel],
                eval_set=[(X_train.iloc[eval_rel], y_tr.iloc[eval_rel])],
                verbose=False,
            )
            return model

        pts_model = _fit(pts_model, y_pts)
        reb_model = _fit(reb_model, y_reb)
        ast_model = _fit(ast_model, y_ast)

        pts_pred = pts_model.predict(X_test)
        reb_pred = reb_model.predict(X_test)
        ast_pred = ast_model.predict(X_test)

        test_rows = df.loc[test_mask].reset_index(drop=True)

        for i in range(n_test):
            row = test_rows.iloc[i]
            records.append({
                "fold": k,
                "game_date_et": row["game_date_et"],
                "game_slug": row.get("game_slug", ""),
                "player_id": row.get("player_id", ""),
                "team_abbr": row.get("team_abbr", ""),
                "actual_pts": float(row["points"]),
                "pred_pts": float(pts_pred[i]),
                "actual_reb": float(row["rebounds"]),
                "pred_reb": float(reb_pred[i]),
                "actual_ast": float(row["assists"]),
                "pred_ast": float(ast_pred[i]),
                "n_train": n_train,
            })

        fold_pts_mae = float(np.mean(np.abs(y_pts.loc[test_mask].values - pts_pred)))
        fold_reb_mae = float(np.mean(np.abs(y_reb.loc[test_mask].values - reb_pred)))
        fold_ast_mae = float(np.mean(np.abs(y_ast.loc[test_mask].values - ast_pred)))
        log.info(
            "Props Fold %3d | train_end=%s n_train=%5d n_test=%4d | "
            "PTS=%.3f REB=%.3f AST=%.3f",
            k, train_end.date(), n_train, n_test,
            fold_pts_mae, fold_reb_mae, fold_ast_mae,
        )

    return pd.DataFrame(records)


def print_prop_report(results: pd.DataFrame) -> None:
    if results.empty:
        print("No prop results to report.")
        return

    n = len(results)
    print(f"\n{'='*65}")
    print(f"  PLAYER PROP BACKTEST REPORT  ({n} player-games, {results['fold'].nunique()} folds)")
    print(f"{'='*65}")
    print(
        f"  Date range: {results['game_date_et'].min().date()} to "
        f"{results['game_date_et'].max().date()}"
    )

    for stat, actual_col, pred_col in [
        ("POINTS",  "actual_pts", "pred_pts"),
        ("REBOUNDS", "actual_reb", "pred_reb"),
        ("ASSISTS",  "actual_ast", "pred_ast"),
    ]:
        y = results[actual_col].values
        p = results[pred_col].values
        mae = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        bias = _bias(y, p)
        slope = _calib_slope(y, p)

        print(f"\n  {stat}")
        print(f"    MAE   = {mae:.3f}")
        print(f"    RMSE  = {rmse:.3f}")
        print(f"    Bias  = {bias:+.3f}  (+ = over-predicts)")
        print(f"    Calib slope = {slope:.3f}  (ideal = 1.00)")
        print(_calibration_bins(y, p, n_bins=5, label=stat.lower()))

    # By season year
    results["season_year"] = pd.to_datetime(results["game_date_et"]).dt.year
    print(f"\n  BY SEASON YEAR (PTS MAE | REB MAE | AST MAE):")
    for yr, grp in results.groupby("season_year"):
        pm = float(np.mean(np.abs(grp["actual_pts"] - grp["pred_pts"])))
        rm = float(np.mean(np.abs(grp["actual_reb"] - grp["pred_reb"])))
        am = float(np.mean(np.abs(grp["actual_ast"] - grp["pred_ast"])))
        print(f"    {yr}: n={len(grp):6d}  PTS={pm:.3f}  REB={rm:.3f}  AST={am:.3f}")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Walk-forward backtest + calibration report")
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=60,
        help="Minimum days of training data before first test fold (default: 60)",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=7,
        help="Size of each test window in days (default: 7)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=7,
        help="Days to advance between folds (default: 7)",
    )
    parser.add_argument(
        "--hc-edge",
        type=float,
        default=3.0,
        help="Minimum |pred - spread| to flag as high-confidence ATS bet (default: 3.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write per-game predictions CSV (optional)",
    )
    parser.add_argument(
        "--props",
        action="store_true",
        help="Also run player prop (PTS/REB/AST) walk-forward backtest",
    )
    parser.add_argument(
        "--props-only",
        action="store_true",
        help="Skip game backtest, only run player prop backtest",
    )
    parser.add_argument(
        "--props-output",
        type=str,
        default=None,
        help="Path to write per-player-game prop predictions CSV (optional)",
    )
    args = parser.parse_args()

    cfg = BacktestConfig(
        min_train_days=args.min_train_days,
        test_window_days=args.test_window,
        step_days=args.step,
        hc_edge_pts=args.hc_edge,
        output_csv=args.output,
    )

    run_games = not args.props_only
    run_props = args.props or args.props_only

    if run_games:
        results = run_backtest(cfg)
        if results.empty:
            log.error("No game backtest results produced. Check data availability.")
            if not run_props:
                sys.exit(1)
        else:
            print_report(results, cfg)
            if cfg.output_csv:
                out_path = Path(cfg.output_csv)
                results.to_csv(out_path, index=False)
                log.info("Per-game predictions written to %s", out_path)

    if run_props:
        prop_results = run_prop_backtest(cfg)
        if prop_results.empty:
            log.error("No prop backtest results produced. Check data availability.")
        else:
            print_prop_report(prop_results)
            if args.props_output:
                prop_path = Path(args.props_output)
                prop_results.to_csv(prop_path, index=False)
                log.info("Per-player-game prop predictions written to %s", prop_path)


if __name__ == "__main__":
    main()
