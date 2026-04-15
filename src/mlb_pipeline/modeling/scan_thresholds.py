# src/mlb_pipeline/modeling/scan_thresholds.py
"""
MLB edge-threshold backtest: scans every edge cutoff from 0.5 to 4.0 runs
and reports W/L, ROI, CLV beat%, and sample size for run-line and total bets.

Use this to find the optimal thresholds — the sweet spot where edge is large
enough to be reliable but not so large that you're down to 5 bets/season.

Usage:
    python -m mlb_pipeline.modeling.scan_thresholds
    python -m mlb_pipeline.modeling.scan_thresholds --days 180
    python -m mlb_pipeline.modeling.scan_thresholds --min-bets 15
"""
import argparse
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.scan_thresholds")

_ET     = ZoneInfo("America/New_York")
PG_DSN  = "postgresql://josh:password@localhost:5432/nba"

# Current live thresholds — shown with a marker in the output
_CURRENT_RL_THRESHOLD  = 1.5
_CURRENT_TOT_THRESHOLD = 6.0

# Threshold scan range and step
_SCAN_MIN  = 0.50
_SCAN_MAX  = 4.00
_SCAN_STEP = 0.25


def _roi(wins: int, n: int) -> float:
    if n == 0:
        return float("nan")
    return (wins * 100 - (n - wins) * 110) / (n * 110) * 100


def _pct(num: int, denom: int) -> float:
    if denom == 0:
        return float("nan")
    return num / denom * 100


def _marker(threshold: float, current: float) -> str:
    """Return ' <-- current' if this threshold matches the live setting."""
    return " <-- current" if abs(threshold - current) < 0.01 else ""


def _find_optimal(rows: list[dict], min_bets: int) -> float | None:
    """Return the threshold with best ROI among rows with n >= min_bets."""
    candidates = [r for r in rows if r["n"] >= min_bets and not _is_nan(r["roi"])]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["roi"])["threshold"]


def _is_nan(v) -> bool:
    import math
    try:
        return math.isnan(v)
    except TypeError:
        return True


def run_scan(conn, days: int = 180, min_bets: int = 10) -> None:
    cutoff = (datetime.now(_ET).date() - timedelta(days=days)).isoformat()

    # Pull all graded game predictions in one query
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                p.edge_run_line,
                p.edge_total,
                p.run_line_covered,
                p.total_covered,
                p.clv_rl_price,
                p.clv_total_price,
                CASE
                    WHEN g.start_ts_utc IS NOT NULL
                    THEN EXTRACT(HOUR FROM g.start_ts_utc AT TIME ZONE 'America/New_York') < 17
                    ELSE NULL
                END AS is_day_game
            FROM bets.mlb_game_predictions p
            LEFT JOIN raw.mlb_games g ON g.game_slug = p.game_slug
            WHERE p.game_date_et >= %s
              AND (p.run_line_covered IS NOT NULL OR p.total_covered IS NOT NULL)
        """, (cutoff,))
        rows = cur.fetchall()

    if not rows:
        print(f"No graded MLB predictions found since {cutoff}.")
        return

    total_graded = len(rows)
    print(f"\n{'='*72}")
    print(f"  MLB THRESHOLD BACKTEST  (last {days}d  |  {total_graded} graded games)")
    print(f"{'='*72}")

    # ── Build per-threshold stats ─────────────────────────────────────────────
    thresholds = []
    t = _SCAN_MIN
    while t <= _SCAN_MAX + 0.001:
        thresholds.append(round(t, 2))
        t += _SCAN_STEP

    def _scan(edge_col: str, covered_col: str, price_clv_col: str, current_thresh: float):
        # price_clv_col: probability-point CLV (positive = beat the close).
        # NOTE: MLB run lines rarely move (locked at ±1.5); line-point CLV is almost
        # always 0, making CLV% meaningless. Price CLV (vig movement) is the correct
        # metric for MLB.
        results = []
        for thr in thresholds:
            subset = [
                r for r in rows
                if r[edge_col] is not None
                and r[covered_col] is not None
                and abs(float(r[edge_col])) >= thr
            ]
            n    = len(subset)
            wins = sum(1 for r in subset if r[covered_col])
            roi  = _roi(wins, n)
            win_pct = _pct(wins, n)

            clv_subset = [r for r in subset if r[price_clv_col] is not None]
            clv_n    = len(clv_subset)
            clv_beat = sum(1 for r in clv_subset if float(r[price_clv_col]) > 0)
            avg_clv  = (sum(float(r[price_clv_col]) for r in clv_subset) / clv_n
                        if clv_n else float("nan"))

            results.append({
                "threshold": thr,
                "n":         n,
                "wins":      wins,
                "win_pct":   win_pct,
                "roi":       roi,
                "clv_n":     clv_n,
                "clv_beat":  clv_beat,
                "avg_clv":   avg_clv,
            })
        return results

    rl_results  = _scan("edge_run_line", "run_line_covered", "clv_rl_price",    _CURRENT_RL_THRESHOLD)
    tot_results = _scan("edge_total",    "total_covered",    "clv_total_price", _CURRENT_TOT_THRESHOLD)

    optimal_rl  = _find_optimal(rl_results,  min_bets)
    optimal_tot = _find_optimal(tot_results, min_bets)

    # ── Print run-line table ──────────────────────────────────────────────────
    print(f"\n  RUN LINE  (edge >= threshold)  |  optimal threshold shown with *")
    print(f"  min_bets filter for optimal: {min_bets}")
    print(f"  pCLV% = price CLV beat rate (% of bets where we got better vig than close)")
    print()
    print(f"  {'Thr':>5}  {'Bets':>5}  {'W-L':<9}  {'Win%':>6}  {'ROI':>8}  "
          f"{'pCLVn':>5}  {'pCLV%':>6}  {'AvgpCLV':>8}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}  "
          f"{'-'*5}  {'-'*6}  {'-'*8}")
    for r in rl_results:
        n, w = r["n"], r["wins"]
        if n == 0:
            continue
        marker = _marker(r["threshold"], _CURRENT_RL_THRESHOLD)
        opt    = "* " if optimal_rl is not None and abs(r["threshold"] - optimal_rl) < 0.01 else "  "
        clv_pct = _pct(r["clv_beat"], r["clv_n"])
        avg_clv_str = f"{r['avg_clv']:+.2f}" if not _is_nan(r["avg_clv"]) else "  N/A"
        win_str = f"{r['win_pct']:.1f}%" if not _is_nan(r["win_pct"]) else " N/A"
        roi_str = f"{r['roi']:+.1f}%"   if not _is_nan(r["roi"])     else "  N/A"
        clv_pct_str = f"{clv_pct:.0f}%" if not _is_nan(clv_pct)      else "N/A"
        print(f"  {opt}{r['threshold']:>4.2f}  {n:>5}  {w}-{n-w:<7}  {win_str:>6}  "
              f"{roi_str:>8}  {r['clv_n']:>5}  {clv_pct_str:>6}  {avg_clv_str:>8}"
              f"{marker}")

    if optimal_rl is not None:
        best = next(r for r in rl_results if abs(r["threshold"] - optimal_rl) < 0.01)
        print(f"\n  >> Optimal run-line threshold: {optimal_rl:.2f} runs  "
              f"({best['wins']}-{best['n']-best['wins']}, ROI {best['roi']:+.1f}%)")
    print(f"  >> Current run-line threshold: {_CURRENT_RL_THRESHOLD:.2f} runs")

    # ── Print total table ─────────────────────────────────────────────────────
    print(f"\n  TOTAL  (edge >= threshold)")
    print()
    print(f"  {'Thr':>5}  {'Bets':>5}  {'W-L':<9}  {'Win%':>6}  {'ROI':>8}  "
          f"{'pCLVn':>5}  {'pCLV%':>6}  {'AvgpCLV':>8}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}  "
          f"{'-'*5}  {'-'*6}  {'-'*8}")
    for r in tot_results:
        n, w = r["n"], r["wins"]
        if n == 0:
            continue
        marker = _marker(r["threshold"], _CURRENT_TOT_THRESHOLD)
        opt    = "* " if optimal_tot is not None and abs(r["threshold"] - optimal_tot) < 0.01 else "  "
        clv_pct = _pct(r["clv_beat"], r["clv_n"])
        avg_clv_str = f"{r['avg_clv']:+.2f}" if not _is_nan(r["avg_clv"]) else "  N/A"
        win_str = f"{r['win_pct']:.1f}%" if not _is_nan(r["win_pct"]) else " N/A"
        roi_str = f"{r['roi']:+.1f}%"   if not _is_nan(r["roi"])     else "  N/A"
        clv_pct_str = f"{clv_pct:.0f}%" if not _is_nan(clv_pct)      else "N/A"
        print(f"  {opt}{r['threshold']:>4.2f}  {n:>5}  {w}-{n-w:<7}  {win_str:>6}  "
              f"{roi_str:>8}  {r['clv_n']:>5}  {clv_pct_str:>6}  {avg_clv_str:>8}"
              f"{marker}")

    if optimal_tot is not None:
        best = next(r for r in tot_results if abs(r["threshold"] - optimal_tot) < 0.01)
        print(f"\n  >> Optimal total threshold: {optimal_tot:.2f} runs  "
              f"({best['wins']}-{best['n']-best['wins']}, ROI {best['roi']:+.1f}%)")
    print(f"  >> Current total threshold:  {_CURRENT_TOT_THRESHOLD:.2f} runs")

    # ── Combined ROI table (run-line + total together at same threshold) ──────
    print(f"\n  COMBINED  (run-line + total, same threshold applied to both)")
    print()
    print(f"  {'Thr':>5}  {'Bets':>5}  {'W-L':<9}  {'Win%':>6}  {'ROI':>8}  {'AvgpCLV':>8}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*8}")

    combined_results = []
    for rl, tot in zip(rl_results, tot_results):
        assert abs(rl["threshold"] - tot["threshold"]) < 0.001
        n = rl["n"] + tot["n"]
        w = rl["wins"] + tot["wins"]
        roi = _roi(w, n)
        clv_vals = []
        for r in rows:
            if r["edge_run_line"] is not None and r["run_line_covered"] is not None \
               and abs(float(r["edge_run_line"])) >= rl["threshold"] \
               and r["clv_rl_price"] is not None:
                clv_vals.append(float(r["clv_rl_price"]))
            if r["edge_total"] is not None and r["total_covered"] is not None \
               and abs(float(r["edge_total"])) >= rl["threshold"] \
               and r["clv_total_price"] is not None:
                clv_vals.append(float(r["clv_total_price"]))
        avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else float("nan")
        combined_results.append({
            "threshold": rl["threshold"], "n": n, "wins": w, "roi": roi, "avg_clv": avg_clv
        })

    optimal_combined = _find_optimal(combined_results, min_bets)

    for r in combined_results:
        n, w = r["n"], r["wins"]
        if n == 0:
            continue
        opt = "* " if optimal_combined is not None and abs(r["threshold"] - optimal_combined) < 0.01 else "  "
        win_str = f"{_pct(w,n):.1f}%" if n else " N/A"
        roi_str = f"{r['roi']:+.1f}%" if not _is_nan(r["roi"]) else "  N/A"
        avg_clv_str = f"{r['avg_clv']:+.2f}" if not _is_nan(r["avg_clv"]) else "  N/A"
        # marker if both current thresholds are the same, else skip
        print(f"  {opt}{r['threshold']:>4.2f}  {n:>5}  {w}-{n-w:<7}  {win_str:>6}  "
              f"{roi_str:>8}  {avg_clv_str:>8}")

    if optimal_combined is not None:
        best = next(r for r in combined_results if abs(r["threshold"] - optimal_combined) < 0.01)
        print(f"\n  >> Optimal combined threshold: {optimal_combined:.2f} runs  "
              f"({best['wins']}-{best['n']-best['wins']}, ROI {best['roi']:+.1f}%)")

    # ── How to apply the results ──────────────────────────────────────────────
    print(f"""
  HOW TO UPDATE THRESHOLDS
  ─────────────────────────
  1. Identify the optimal threshold from the tables above (marked *).
  2. Update predict_today.py:
       _MIN_EDGE_RUN_LINE = <new value>   (currently {_CURRENT_RL_THRESHOLD})
       _MIN_EDGE_TOTAL    = <new value>   (currently {_CURRENT_TOT_THRESHOLD})
  3. Rerun predictions: python -m mlb_pipeline.modeling.predict_today
  4. Re-run this scan after 2+ weeks of new graded data to validate.

  NOTE: Prefer the CLV beat% trend over raw ROI when sample size < 50 bets.
  CLV > 50% beat rate is a stronger long-run signal than short-run W/L records.
""")
    print(f"{'='*72}\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="MLB edge-threshold backtest")
    parser.add_argument("--days",     type=int, default=180,
                        help="Lookback window in days (default 180)")
    parser.add_argument("--min-bets", type=int, default=10,
                        help="Minimum bets required to flag a threshold as optimal (default 10)")
    args = parser.parse_args()

    with psycopg2.connect(PG_DSN) as conn:
        run_scan(conn, days=args.days, min_bets=args.min_bets)


if __name__ == "__main__":
    main()