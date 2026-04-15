# src/mlb_pipeline/modeling/scan_prop_thresholds.py
"""
MLB player-prop edge-threshold backtest.

For each stat, scans every |edge| cutoff and reports W/L, ROI, win%,
and OVER vs UNDER breakdown so you can see where (if anywhere) the
model has a genuine edge.

Usage:
    python -m mlb_pipeline.modeling.scan_prop_thresholds
    python -m mlb_pipeline.modeling.scan_prop_thresholds --days 180
    python -m mlb_pipeline.modeling.scan_prop_thresholds --stat batter_hits
    python -m mlb_pipeline.modeling.scan_prop_thresholds --min-bets 20
"""
import argparse
import logging
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.scan_prop_thresholds")

_ET    = ZoneInfo("America/New_York")
PG_DSN = "postgresql://josh:password@localhost:5432/nba"

# Per-stat scan config: threshold range + current live value
# Ranges chosen to match realistic edge magnitudes for each stat unit.
_STAT_CONFIGS: dict[str, dict] = {
    "pitcher_strikeouts": {"min": 0.25, "max": 3.00, "step": 0.25, "current": 2.00, "label": "Strikeouts"},
    "batter_hits":        {"min": 0.05, "max": 0.75, "step": 0.05, "current": 0.75, "label": "Hits"},
    "batter_total_bases": {"min": 0.10, "max": 1.50, "step": 0.10, "current": 0.60, "label": "Total Bases"},
    "batter_home_runs":   {"min": 0.05, "max": 0.50, "step": 0.05, "current": 0.45, "label": "Home Runs"},
    "batter_walks":       {"min": 0.05, "max": 0.50, "step": 0.05, "current": 0.30, "label": "Walks"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roi(wins: int, n: int) -> float:
    if n == 0:
        return float("nan")
    return (wins * 100 - (n - wins) * 110) / (n * 110) * 100


def _pct(num: int, denom: int) -> float:
    if denom == 0:
        return float("nan")
    return num / denom * 100


def _fmt_roi(v: float) -> str:
    return f"{v:+.1f}%" if not math.isnan(v) else "  N/A"


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%" if not math.isnan(v) else " N/A"


def _find_optimal(rows: list[dict], min_bets: int) -> float | None:
    """Return the threshold with the best ROI among rows with n >= min_bets."""
    candidates = [r for r in rows if r["n"] >= min_bets and not math.isnan(r["roi"])]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["roi"])["threshold"]


def _thresholds(cfg: dict) -> list[float]:
    out = []
    t = cfg["min"]
    while t <= cfg["max"] + 0.001:
        out.append(round(t, 4))
        t += cfg["step"]
    return out


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def _scan_stat(rows: list[dict], cfg: dict, min_bets: int) -> list[dict]:
    """Compute per-threshold stats for a single prop stat."""
    results = []
    for thr in _thresholds(cfg):
        # All bets above this threshold (regardless of direction)
        subset = [
            r for r in rows
            if r["edge"] is not None
            and r["over_hit"] is not None
            and abs(float(r["edge"])) >= thr
        ]
        n = len(subset)

        # Win condition: if edge > 0 we bet OVER → win when over_hit; else bet UNDER
        def _won(r) -> bool:
            e = float(r["edge"])
            oh = bool(r["over_hit"])
            return oh if e > 0 else not oh

        wins     = sum(1 for r in subset if _won(r))
        roi      = _roi(wins, n)
        win_pct  = _pct(wins, n)

        # OVER-only slice
        over_sub = [r for r in subset if float(r["edge"]) > 0]
        over_w   = sum(1 for r in over_sub if r["over_hit"])
        over_roi = _roi(over_w, len(over_sub))

        # UNDER-only slice
        under_sub = [r for r in subset if float(r["edge"]) < 0]
        under_w   = sum(1 for r in under_sub if not r["over_hit"])
        under_roi = _roi(under_w, len(under_sub))

        results.append({
            "threshold": thr,
            "n":         n,
            "wins":      wins,
            "win_pct":   win_pct,
            "roi":       roi,
            "over_n":    len(over_sub),
            "over_wins": over_w,
            "over_roi":  over_roi,
            "under_n":   len(under_sub),
            "under_wins":under_w,
            "under_roi": under_roi,
        })
    return results


def _print_stat_table(stat: str, cfg: dict, results: list[dict], min_bets: int) -> None:
    label   = cfg["label"]
    current = cfg["current"]
    optimal = _find_optimal(results, min_bets)

    print(f"\n  {label.upper()}  (|edge| >= threshold)")
    print(f"  Current threshold: {current}  |  min_bets for optimal: {min_bets}")
    print()
    print(f"  {'Thr':>6}  {'Bets':>5}  {'W-L':<9}  {'Win%':>6}  {'ROI':>8}"
          f"  {'OVERn':>6}  {'OVERroi':>8}"
          f"  {'UNDn':>5}  {'UNDroi':>8}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*8}"
          f"  {'-'*6}  {'-'*8}"
          f"  {'-'*5}  {'-'*8}")

    for r in results:
        n, w = r["n"], r["wins"]
        if n == 0:
            continue
        is_current = abs(r["threshold"] - current) < 0.001
        is_optimal = optimal is not None and abs(r["threshold"] - optimal) < 0.001
        marker = " <-- current" if is_current else ""
        opt    = "* " if is_optimal else "  "

        over_roi_str  = _fmt_roi(r["over_roi"])
        under_roi_str = _fmt_roi(r["under_roi"])

        print(
            f"  {opt}{r['threshold']:>5.2f}  {n:>5}  {w}-{n-w:<7}"
            f"  {_fmt_pct(r['win_pct']):>6}  {_fmt_roi(r['roi']):>8}"
            f"  {r['over_n']:>6}  {over_roi_str:>8}"
            f"  {r['under_n']:>5}  {under_roi_str:>8}"
            f"{marker}"
        )

    if optimal is not None:
        best = next(r for r in results if abs(r["threshold"] - optimal) < 0.001)
        print(
            f"\n  >> Optimal {label} threshold: {optimal:.2f}"
            f"  ({best['wins']}-{best['n']-best['wins']}, ROI {best['roi']:+.1f}%)"
        )
    else:
        print(f"\n  >> No threshold met min_bets={min_bets} filter.")
    print(f"  >> Current {label} threshold: {current:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scan(conn, days: int = 180, min_bets: int = 10,
             stat_filter: str | None = None) -> None:
    cutoff = (datetime.now(_ET).date() - timedelta(days=days)).isoformat()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT stat, edge, over_hit
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %s
              AND over_hit IS NOT NULL
              AND edge IS NOT NULL
        """, (cutoff,))
        all_rows = cur.fetchall()

    if not all_rows:
        print(f"No graded MLB prop predictions found since {cutoff}.")
        return

    # Group by stat
    by_stat: dict[str, list] = {}
    for r in all_rows:
        by_stat.setdefault(r["stat"], []).append(r)

    total_graded = len(all_rows)
    print(f"\n{'='*72}")
    print(f"  MLB PROP THRESHOLD BACKTEST  (last {days}d  |  {total_graded} graded props)")
    print(f"{'='*72}")

    print(f"\n  {'Stat':<25}  {'Graded':>7}  {'Win%':>6}  {'ROI':>8}  Notes")
    print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*8}  -----")
    for stat, cfg in _STAT_CONFIGS.items():
        if stat_filter and stat != stat_filter:
            continue
        rows = by_stat.get(stat, [])
        n = len(rows)
        if n == 0:
            print(f"  {cfg['label']:<25}  {'—':>7}  {'—':>6}  {'—':>8}  no graded data")
            continue

        def _won(r) -> bool:
            e = float(r["edge"])
            oh = bool(r["over_hit"])
            return oh if e > 0 else not oh

        wins = sum(1 for r in rows if _won(r))
        roi  = _roi(wins, n)

        all_over  = all(float(r["edge"]) > 0 for r in rows)
        all_under = all(float(r["edge"]) < 0 for r in rows)
        note = ""
        if all_under:
            note = "[!] model always bets UNDER"
        elif all_over:
            note = "[!] model always bets OVER"

        print(f"  {cfg['label']:<25}  {n:>7}  {_fmt_pct(_pct(wins,n)):>6}  {_fmt_roi(roi):>8}  {note}")

    # Per-stat detail
    for stat, cfg in _STAT_CONFIGS.items():
        if stat_filter and stat != stat_filter:
            continue
        rows = by_stat.get(stat, [])
        if not rows:
            continue
        results = _scan_stat(rows, cfg, min_bets)
        _print_stat_table(stat, cfg, results, min_bets)

    # Recommendation block
    print(f"""
  HOW TO UPDATE THRESHOLDS
  -------------------------
  1. Identify the optimal threshold from the tables above (marked *).
  2. Update predict_player_props.py (PropConfig):
       threshold_strikeouts  = <new>   (currently {_STAT_CONFIGS['pitcher_strikeouts']['current']})
       threshold_hits        = <new>   (currently {_STAT_CONFIGS['batter_hits']['current']})
       threshold_total_bases = <new>   (currently {_STAT_CONFIGS['batter_total_bases']['current']})
       threshold_home_runs   = <new>   (currently {_STAT_CONFIGS['batter_home_runs']['current']})
       threshold_walks       = <new>   (currently {_STAT_CONFIGS['batter_walks']['current']})
  3. For stats with no positive ROI at any threshold, consider disabling
     by raising the threshold above its observed max edge.
  4. Re-run after 3-4 weeks of new graded data to validate changes.

  NOTE: Win% > 52.4% is breakeven at -110 juice. Prefer stats where
  both win% AND ROI are positive at the optimal threshold.
""")
    print(f"{'='*72}\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="MLB prop edge-threshold backtest")
    parser.add_argument("--days",     type=int,   default=180,
                        help="Lookback window in days (default 180)")
    parser.add_argument("--min-bets", type=int,   default=10,
                        help="Minimum bets to flag a threshold as optimal (default 10)")
    parser.add_argument("--stat",     type=str,   default=None,
                        help="Limit to one stat (e.g. batter_hits)")
    args = parser.parse_args()

    with psycopg2.connect(PG_DSN) as conn:
        run_scan(conn, days=args.days, min_bets=args.min_bets,
                 stat_filter=args.stat)


if __name__ == "__main__":
    main()
