# src/mlb_pipeline/modeling/scan_thresholds.py
"""
Side-aware MLB edge-threshold backtest.

This script scans run-line and total edge cutoffs, then breaks results down by
the actual saved bet side. That matters because "abs(edge)" alone can hide a
good side and a bad side inside the same market.

Usage:
    python -m mlb_pipeline.modeling.scan_thresholds
    python -m mlb_pipeline.modeling.scan_thresholds --days 180
    python -m mlb_pipeline.modeling.scan_thresholds --min-bets 15
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.scan_thresholds")

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN
_CURRENT_RL_THRESHOLD = 1.5
_CURRENT_TOT_THRESHOLD = 2.0

_SCAN_MIN = 0.50
_SCAN_MAX = 4.00
_SCAN_STEP = 0.25


def _thresholds() -> list[float]:
    out = []
    t = _SCAN_MIN
    while t <= _SCAN_MAX + 0.001:
        out.append(round(t, 2))
        t += _SCAN_STEP
    return out


def _pct(num: float, denom: float) -> float:
    if not denom:
        return float("nan")
    return num / denom * 100.0


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return True


def _fnum(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or _is_nan(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _american_profit(price) -> float:
    """Net profit on a 1-unit stake if the bet wins."""
    if price is None:
        return 100.0 / 110.0
    p = float(price)
    if p > 0:
        return p / 100.0
    if p < 0:
        return 100.0 / abs(p)
    return 100.0 / 110.0


def _unit_profit(covered, price) -> float:
    return _american_profit(price) if bool(covered) else -1.0


def _marker(threshold: float, current: float) -> str:
    return " <-- current" if abs(threshold - current) < 0.01 else ""


def _find_optimal(rows: list[dict], min_bets: int) -> float | None:
    candidates = [r for r in rows if r["n"] >= min_bets and not _is_nan(r["roi"])]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["roi"])["threshold"]


def _side_rows(rows: Iterable[dict], market: str, side: str | None = None) -> list[dict]:
    if market == "run_line":
        return [
            r for r in rows
            if r["run_line_bet_side"] is not None
            and r["run_line_covered"] is not None
            and (side is None or r["run_line_bet_side"] == side)
        ]
    if market == "total":
        return [
            r for r in rows
            if r["total_bet_side"] is not None
            and r["total_covered"] is not None
            and (side is None or r["total_bet_side"] == side)
        ]
    raise ValueError(f"unknown market: {market}")


def _edge_abs(row: dict, market: str) -> float | None:
    key = "edge_run_line" if market == "run_line" else "edge_total"
    value = row.get(key)
    return None if value is None else abs(float(value))


def _scan(rows: list[dict], market: str, side: str | None, thresholds: list[float]) -> list[dict]:
    base = _side_rows(rows, market, side)
    covered_col = "run_line_covered" if market == "run_line" else "total_covered"
    price_col = "market_rl_price" if market == "run_line" else "market_total_price"
    clv_col = "clv_rl_price" if market == "run_line" else "clv_total_price"

    results = []
    for threshold in thresholds:
        subset = [r for r in base if (_edge_abs(r, market) or 0.0) >= threshold]
        n = len(subset)
        wins = sum(1 for r in subset if r[covered_col])
        profits = [_unit_profit(r[covered_col], r.get(price_col)) for r in subset]
        roi = (sum(profits) / n) * 100.0 if n else float("nan")

        clv_subset = [r for r in subset if r.get(clv_col) is not None]
        clv_n = len(clv_subset)
        clv_beat = sum(1 for r in clv_subset if float(r[clv_col]) > 0)
        avg_clv = (
            sum(float(r[clv_col]) for r in clv_subset) / clv_n
            if clv_n else float("nan")
        )
        avg_price = (
            sum(float(r[price_col]) for r in subset if r.get(price_col) is not None)
            / sum(1 for r in subset if r.get(price_col) is not None)
            if any(r.get(price_col) is not None for r in subset)
            else float("nan")
        )
        results.append({
            "threshold": threshold,
            "n": n,
            "wins": wins,
            "win_pct": _pct(wins, n),
            "roi": roi,
            "clv_n": clv_n,
            "clv_beat": clv_beat,
            "avg_clv": avg_clv,
            "avg_price": avg_price,
        })
    return results


def _print_scan_table(title: str, results: list[dict], current: float, min_bets: int) -> None:
    optimal = _find_optimal(results, min_bets)
    print(f"\n  {title}")
    print(f"  min_bets filter for optimal: {min_bets}")
    print(f"  ROI uses stored American prices when present; otherwise assumes -110.")
    print()
    print(
        f"  {'Thr':>5}  {'Bets':>5}  {'W-L':<9}  {'Win%':>6}  {'ROI':>8}  "
        f"{'pCLV%':>6}  {'AvgpCLV':>8}  {'AvgPx':>7}"
    )
    print(
        f"  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}  "
        f"{'-'*6}  {'-'*8}  {'-'*7}"
    )
    for row in results:
        n = row["n"]
        if n == 0:
            continue
        wins = row["wins"]
        opt = "* " if optimal is not None and abs(row["threshold"] - optimal) < 0.01 else "  "
        clv_pct = _pct(row["clv_beat"], row["clv_n"])
        print(
            f"  {opt}{row['threshold']:>4.2f}  {n:>5}  {wins}-{n-wins:<7}  "
            f"{_fnum(row['win_pct'], 1, '%'):>6}  {_fnum(row['roi'], 1, '%'):>8}  "
            f"{_fnum(clv_pct, 0, '%'):>6}  {_fnum(row['avg_clv'], 2):>8}  "
            f"{_fnum(row['avg_price'], 0):>7}{_marker(row['threshold'], current)}"
        )
    if optimal is not None:
        best = next(r for r in results if abs(r["threshold"] - optimal) < 0.01)
        print(
            f"\n  >> Optimal threshold: {optimal:.2f} "
            f"({best['wins']}-{best['n'] - best['wins']}, ROI {best['roi']:+.1f}%)"
        )
    print(f"  >> Current threshold: {current:.2f}")


def _bucket_total(row: dict) -> str:
    line = row.get("market_total")
    if line is None:
        return "no total"
    line = float(line)
    if line >= 10.5:
        return "10.5+"
    if line >= 9.5:
        return "9.5-10.0"
    if line >= 8.5:
        return "8.5-9.0"
    return "<=8.0"


def _bucket_run_line(row: dict) -> str:
    side = row.get("run_line_bet_side")
    market_line = row.get("market_run_line")
    price = row.get("market_rl_price")
    if side == "away" and market_line is not None and float(market_line) > 0:
        return "away favorite"
    if price is not None and float(price) < -180:
        return "heavy juice"
    pred = row.get("pred_run_diff")
    if pred is None:
        return "no pred"
    pred = float(pred)
    if pred <= -3.0:
        return "pred away by 3+"
    if pred < 0:
        return "pred away close"
    if pred < 1.5:
        return "pred home close"
    return "pred home by 1.5+"


def _print_bucket_table(rows: list[dict], market: str, current: float) -> None:
    covered_col = "run_line_covered" if market == "run_line" else "total_covered"
    price_col = "market_rl_price" if market == "run_line" else "market_total_price"
    edge_col = "edge_run_line" if market == "run_line" else "edge_total"
    side_col = "run_line_bet_side" if market == "run_line" else "total_bet_side"
    clv_col = "clv_rl_price" if market == "run_line" else "clv_total_price"
    bucket_fn = _bucket_run_line if market == "run_line" else _bucket_total

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get(side_col) is None or row.get(covered_col) is None:
            continue
        if (_edge_abs(row, market) or 0.0) < current:
            continue
        key = (str(row[side_col]), bucket_fn(row))
        groups.setdefault(key, []).append(row)

    print(f"\n  {market.replace('_', ' ').upper()} SIDE/BUCKET DIAGNOSIS at current threshold")
    print(
        f"  {'Side':<6}  {'Bucket':<18}  {'Bets':>5}  {'Win%':>6}  {'ROI':>8}  "
        f"{'AvgEdge':>8}  {'AvgPx':>7}  {'AvgpCLV':>8}"
    )
    print(
        f"  {'-'*6}  {'-'*18}  {'-'*5}  {'-'*6}  {'-'*8}  "
        f"{'-'*8}  {'-'*7}  {'-'*8}"
    )
    for (side, bucket), subset in sorted(groups.items()):
        n = len(subset)
        wins = sum(1 for r in subset if r[covered_col])
        roi = sum(_unit_profit(r[covered_col], r.get(price_col)) for r in subset) / n * 100.0
        avg_edge = sum(abs(float(r[edge_col])) for r in subset) / n
        price_vals = [float(r[price_col]) for r in subset if r.get(price_col) is not None]
        clv_vals = [float(r[clv_col]) for r in subset if r.get(clv_col) is not None]
        avg_price = sum(price_vals) / len(price_vals) if price_vals else float("nan")
        avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else float("nan")
        print(
            f"  {side:<6}  {bucket:<18}  {n:>5}  {_fnum(_pct(wins, n), 1, '%'):>6}  "
            f"{_fnum(roi, 1, '%'):>8}  {_fnum(avg_edge, 2):>8}  "
            f"{_fnum(avg_price, 0):>7}  {_fnum(avg_clv, 2):>8}"
        )


def run_scan(conn, days: int = 180, min_bets: int = 10) -> None:
    cutoff = (datetime.now(_ET).date() - timedelta(days=days)).isoformat()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                p.pred_run_diff,
                p.pred_total,
                p.market_run_line,
                p.market_total,
                p.edge_run_line,
                p.edge_total,
                p.actual_run_diff,
                p.actual_total,
                p.run_line_bet_side,
                p.total_bet_side,
                p.run_line_covered,
                p.total_covered,
                p.market_rl_price,
                p.market_total_price,
                p.clv_rl_price,
                p.clv_total_price
            FROM bets.mlb_game_predictions p
            WHERE p.game_date_et >= %s
              AND (
                    (p.run_line_bet_side IS NOT NULL AND p.run_line_covered IS NOT NULL)
                 OR (p.total_bet_side IS NOT NULL AND p.total_covered IS NOT NULL)
              )
            """,
            (cutoff,),
        )
        rows = list(cur.fetchall())

    if not rows:
        print(f"No graded MLB predictions found since {cutoff}.")
        return

    thresholds = _thresholds()
    print(f"\n{'=' * 78}")
    print(f"  MLB SIDE-AWARE THRESHOLD BACKTEST  (last {days}d | {len(rows)} rows)")
    print(f"{'=' * 78}")

    rl_all = _scan(rows, "run_line", None, thresholds)
    tot_all = _scan(rows, "total", None, thresholds)
    _print_scan_table("RUN LINE: all saved sides", rl_all, _CURRENT_RL_THRESHOLD, min_bets)
    for side in ("home", "away"):
        _print_scan_table(f"RUN LINE: {side.upper()} only", _scan(rows, "run_line", side, thresholds), _CURRENT_RL_THRESHOLD, min_bets)

    _print_scan_table("TOTAL: all saved sides", tot_all, _CURRENT_TOT_THRESHOLD, min_bets)
    for side in ("over", "under"):
        _print_scan_table(f"TOTAL: {side.upper()} only", _scan(rows, "total", side, thresholds), _CURRENT_TOT_THRESHOLD, min_bets)

    _print_bucket_table(rows, "run_line", _CURRENT_RL_THRESHOLD)
    _print_bucket_table(rows, "total", _CURRENT_TOT_THRESHOLD)

    print("""
  HOW TO USE THIS
  ---------------
  1. Treat the all-market table as a broad check only.
  2. Use the side tables to decide whether a side is actually investable.
  3. Use the bucket table to find why a side is weak before changing live gates.
  4. Rerun after each meaningful model or market-gate change.

  Current live defaults:
    min_edge_run_line = 1.50
    min_edge_total    = 2.00
""")
    print(f"{'=' * 78}\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="MLB side-aware edge-threshold backtest")
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days")
    parser.add_argument("--min-bets", type=int, default=10, help="Minimum bets required to flag an optimal threshold")
    args = parser.parse_args()

    with psycopg2.connect(PG_DSN) as conn:
        run_scan(conn, days=args.days, min_bets=args.min_bets)


if __name__ == "__main__":
    main()
