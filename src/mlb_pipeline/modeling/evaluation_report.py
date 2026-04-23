"""
MLB model change evaluation report (pre vs post split date).

Compares hit rate and ROI across:
  - game predictions (run line / total)
  - edge buckets
  - props by stat and side

Usage:
  python -m mlb_pipeline.modeling.evaluation_report --split-date 2026-04-20
  python -m mlb_pipeline.modeling.evaluation_report --split-date 2026-04-20 --days 180
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
# Markets where UNDER side is not realistically bettable for this workflow.
_UNDER_UNBETTABLE_STATS = {"batter_home_runs", "batter_walks"}


def _roi(wins: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return (wins * 100 - (n - wins) * 110) / (n * 110) * 100


def _pct(num: int, den: int) -> float:
    return (num / den * 100.0) if den else 0.0


def _seg_label(d, split_date):
    return "POST" if d >= split_date else "PRE"


def run_report(conn, *, split_date, days: int, min_bets: int) -> None:
    cutoff = (datetime.now(_ET).date() - timedelta(days=days))
    print(f"\nMLB EVALUATION REPORT | cutoff={cutoff} | split_date={split_date} | min_bets={min_bets}")
    print("=" * 92)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            WITH game_rows AS (
                SELECT game_date_et, 'run_line'::text AS market, edge_run_line AS edge, run_line_covered AS won
                FROM bets.mlb_game_predictions
                WHERE game_date_et >= %(cutoff)s
                  AND run_line_covered IS NOT NULL
                  AND edge_run_line IS NOT NULL
                UNION ALL
                SELECT game_date_et, 'total'::text, edge_total AS edge, total_covered AS won
                FROM bets.mlb_game_predictions
                WHERE game_date_et >= %(cutoff)s
                  AND total_covered IS NOT NULL
                  AND edge_total IS NOT NULL
            )
            SELECT game_date_et, market, edge, won
            FROM game_rows
            """,
            {"cutoff": cutoff},
        )
        game_rows = cur.fetchall()

    if game_rows:
        g_dates = [r["game_date_et"] for r in game_rows]
        g_pre = sum(1 for d in g_dates if d < split_date)
        g_post = sum(1 for d in g_dates if d >= split_date)
        print(
            f"Game rows loaded: {len(game_rows)} "
            f"(PRE={g_pre}, POST={g_post}) | range={min(g_dates)}..{max(g_dates)}"
        )
        if g_post == 0:
            print(
                "WARNING: No POST game rows for this split date. "
                "Use an earlier --split-date or confirm recent predictions were saved."
            )

    # ── Overall game markets (PRE vs POST) ───────────────────────────────────
    agg = {}
    for r in game_rows:
        seg = _seg_label(r["game_date_et"], split_date)
        key = (seg, r["market"])
        if key not in agg:
            agg[key] = {"n": 0, "w": 0}
        agg[key]["n"] += 1
        agg[key]["w"] += int(bool(r["won"]))

    print("\n[Game Predictions | PRE vs POST]")
    print(f"{'Segment':<8} {'Market':<10} {'Bets':>6} {'W-L':>11} {'Win%':>8} {'ROI':>8}")
    for seg in ("PRE", "POST"):
        for market in ("run_line", "total"):
            a = agg.get((seg, market), {"n": 0, "w": 0})
            n, w = a["n"], a["w"]
            if n < min_bets:
                continue
            print(f"{seg:<8} {market:<10} {n:>6} {w}-{n-w:>7} {_pct(w,n):>7.1f}% {_roi(w,n):>7.1f}%")

    # ── Game edge buckets ─────────────────────────────────────────────────────
    def _bucket(x: float) -> str:
        ax = abs(float(x))
        if ax < 1.0:
            return "0.0-1.0"
        if ax < 1.5:
            return "1.0-1.5"
        if ax < 2.0:
            return "1.5-2.0"
        if ax < 2.5:
            return "2.0-2.5"
        if ax < 3.0:
            return "2.5-3.0"
        return "3.0+"

    buckets = {}
    for r in game_rows:
        seg = _seg_label(r["game_date_et"], split_date)
        key = (seg, r["market"], _bucket(r["edge"]))
        if key not in buckets:
            buckets[key] = {"n": 0, "w": 0}
        buckets[key]["n"] += 1
        buckets[key]["w"] += int(bool(r["won"]))

    print("\n[Game Predictions by Edge Bucket]")
    print(f"{'Segment':<8} {'Market':<10} {'Edge':<8} {'Bets':>6} {'W-L':>11} {'Win%':>8} {'ROI':>8}")
    bucket_order = ("0.0-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0+")
    for seg in ("PRE", "POST"):
        for market in ("run_line", "total"):
            for b in bucket_order:
                a = buckets.get((seg, market, b), {"n": 0, "w": 0})
                n, w = a["n"], a["w"]
                if n < min_bets:
                    continue
                print(f"{seg:<8} {market:<10} {b:<8} {n:>6} {w}-{n-w:>7} {_pct(w,n):>7.1f}% {_roi(w,n):>7.1f}%")

    # ── Props by stat and side ────────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT game_date_et, stat, edge, over_hit
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %(cutoff)s
              AND over_hit IS NOT NULL
              AND edge IS NOT NULL
            """,
            {"cutoff": cutoff},
        )
        prop_rows = cur.fetchall()

    if prop_rows:
        p_dates = [r["game_date_et"] for r in prop_rows]
        p_pre = sum(1 for d in p_dates if d < split_date)
        p_post = sum(1 for d in p_dates if d >= split_date)
        print(
            f"Prop rows loaded: {len(prop_rows)} "
            f"(PRE={p_pre}, POST={p_post}) | range={min(p_dates)}..{max(p_dates)}"
        )
        if p_post == 0:
            print(
                "WARNING: No POST prop rows for this split date. "
                "Use an earlier --split-date or confirm recent predictions were saved."
            )

    prop_agg = {}
    side_agg = {}
    for r in prop_rows:
        seg = _seg_label(r["game_date_et"], split_date)
        stat = r["stat"]
        edge = float(r["edge"])
        if stat in _UNDER_UNBETTABLE_STATS and edge < 0:
            # Exclude unbettable UNDER rows from headline performance stats.
            continue
        over_hit = bool(r["over_hit"])
        won = over_hit if edge > 0 else (not over_hit)
        side = "over" if edge > 0 else "under"

        k1 = (seg, stat)
        if k1 not in prop_agg:
            prop_agg[k1] = {"n": 0, "w": 0}
        prop_agg[k1]["n"] += 1
        prop_agg[k1]["w"] += int(won)

        k2 = (seg, stat, side)
        if k2 not in side_agg:
            side_agg[k2] = {"n": 0, "w": 0}
        side_agg[k2]["n"] += 1
        side_agg[k2]["w"] += int(won)

    print("\n[Props by Stat | PRE vs POST]  (HR/Walk UNDER excluded as unbettable)")
    print(f"{'Segment':<8} {'Stat':<22} {'Bets':>6} {'W-L':>11} {'Win%':>8} {'ROI':>8}")
    for seg in ("PRE", "POST"):
        for stat in sorted({k[1] for k in prop_agg.keys()}):
            a = prop_agg.get((seg, stat), {"n": 0, "w": 0})
            n, w = a["n"], a["w"]
            if n < min_bets:
                continue
            print(f"{seg:<8} {stat:<22} {n:>6} {w}-{n-w:>7} {_pct(w,n):>7.1f}% {_roi(w,n):>7.1f}%")

    print("\n[Props by Stat + Side | PRE vs POST]  (HR/Walk UNDER excluded as unbettable)")
    print(f"{'Segment':<8} {'Stat':<22} {'Side':<6} {'Bets':>6} {'W-L':>11} {'Win%':>8} {'ROI':>8}")
    for seg in ("PRE", "POST"):
        for stat in sorted({k[1] for k in side_agg.keys()}):
            for side in ("over", "under"):
                a = side_agg.get((seg, stat, side), {"n": 0, "w": 0})
                n, w = a["n"], a["w"]
                if n < min_bets:
                    continue
                print(f"{seg:<8} {stat:<22} {side:<6} {n:>6} {w}-{n-w:>7} {_pct(w,n):>7.1f}% {_roi(w,n):>7.1f}%")

    # ── Over-side diagnostics for bettable props ─────────────────────────────
    # Focus on why OVER sides may underperform (esp. TB and K), and on markets
    # where users can only bet OVER (HR, walks at many books).
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                game_date_et,
                stat,
                pred_value,
                book_line,
                edge,
                over_hit,
                actual_value
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %(cutoff)s
              AND stat IN ('pitcher_strikeouts', 'batter_total_bases', 'batter_home_runs', 'batter_walks')
              AND over_hit IS NOT NULL
              AND edge IS NOT NULL
              AND book_line IS NOT NULL
            """,
            {"cutoff": cutoff},
        )
        over_diag = cur.fetchall()

    # OVER-only summary
    over_stats = {}
    for r in over_diag:
        if float(r["edge"]) <= 0:
            continue
        seg = _seg_label(r["game_date_et"], split_date)
        stat = r["stat"]
        key = (seg, stat)
        if key not in over_stats:
            over_stats[key] = {
                "n": 0, "w": 0, "edge_sum": 0.0,
                "pred_sum": 0.0, "line_sum": 0.0,
                "cal_n": 0, "pred_minus_actual_sum": 0.0,
            }
        a = over_stats[key]
        a["n"] += 1
        a["w"] += int(bool(r["over_hit"]))
        a["edge_sum"] += float(r["edge"])
        a["pred_sum"] += float(r["pred_value"] or 0.0)
        a["line_sum"] += float(r["book_line"] or 0.0)
        if r["actual_value"] is not None:
            a["cal_n"] += 1
            a["pred_minus_actual_sum"] += float(r["pred_value"]) - float(r["actual_value"])

    print("\n[OVER-side Diagnostics | Why OVER may be weak]")
    print(
        f"{'Segment':<8} {'Stat':<20} {'Bets':>6} {'W-L':>11} {'Win%':>8} {'ROI':>8} "
        f"{'AvgEdge':>8} {'AvgPred':>8} {'AvgLine':>8} {'Pred-Act':>9}"
    )
    for seg in ("PRE", "POST"):
        for stat in ("pitcher_strikeouts", "batter_total_bases", "batter_home_runs", "batter_walks"):
            a = over_stats.get((seg, stat), None)
            if not a:
                continue
            n, w = a["n"], a["w"]
            if n < min_bets:
                continue
            avg_edge = a["edge_sum"] / n
            avg_pred = a["pred_sum"] / n
            avg_line = a["line_sum"] / n
            pred_act = (a["pred_minus_actual_sum"] / a["cal_n"]) if a["cal_n"] else float("nan")
            pred_act_s = f"{pred_act:+.2f}" if a["cal_n"] else "   n/a"
            print(
                f"{seg:<8} {stat:<20} {n:>6} {w}-{n-w:>7} {_pct(w,n):>7.1f}% {_roi(w,n):>7.1f}% "
                f"{avg_edge:>8.2f} {avg_pred:>8.2f} {avg_line:>8.2f} {pred_act_s:>9}"
            )

    # OVER-only by line bucket (helps detect systematic miss by offered line tier)
    def _line_bucket(stat: str, line: float) -> str:
        if stat == "batter_total_bases":
            if line < 1.0:
                return "TB 0.5"
            if line < 2.0:
                return "TB 1.5"
            return "TB 2.5+"
        if stat == "pitcher_strikeouts":
            if line < 4.5:
                return "K <4.5"
            if line < 6.5:
                return "K 4.5-6.0"
            if line < 8.5:
                return "K 6.5-8.0"
            return "K 8.5+"
        if stat == "batter_home_runs":
            return "HR 0.5"
        if stat == "batter_walks":
            if line < 1.0:
                return "BB 0.5"
            return "BB 1.5+"
        return "other"

    bucket_agg = {}
    for r in over_diag:
        if float(r["edge"]) <= 0:
            continue
        seg = _seg_label(r["game_date_et"], split_date)
        stat = r["stat"]
        b = _line_bucket(stat, float(r["book_line"]))
        key = (seg, stat, b)
        if key not in bucket_agg:
            bucket_agg[key] = {"n": 0, "w": 0}
        bucket_agg[key]["n"] += 1
        bucket_agg[key]["w"] += int(bool(r["over_hit"]))

    print("\n[OVER-side by Line Bucket]")
    print(f"{'Segment':<8} {'Stat':<20} {'LineBucket':<12} {'Bets':>6} {'W-L':>11} {'Win%':>8} {'ROI':>8}")
    for seg in ("PRE", "POST"):
        for stat in ("pitcher_strikeouts", "batter_total_bases", "batter_home_runs", "batter_walks"):
            rows = [(k, v) for k, v in bucket_agg.items() if k[0] == seg and k[1] == stat]
            for (k_seg, k_stat, k_bucket), a in sorted(rows, key=lambda x: x[0][2]):
                n, w = a["n"], a["w"]
                if n < min_bets:
                    continue
                print(f"{k_seg:<8} {k_stat:<20} {k_bucket:<12} {n:>6} {w}-{n-w:>7} {_pct(w,n):>7.1f}% {_roi(w,n):>7.1f}%")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB model change evaluation report (pre vs post split date).")
    parser.add_argument("--split-date", required=True, help="YYYY-MM-DD. Dates >= split are POST.")
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days (default: 180).")
    parser.add_argument("--min-bets", type=int, default=20, help="Hide rows below this sample size.")
    args = parser.parse_args()

    split_date = datetime.strptime(args.split_date, "%Y-%m-%d").date()
    with psycopg2.connect(_PG_DSN) as conn:
        run_report(conn, split_date=split_date, days=args.days, min_bets=args.min_bets)


if __name__ == "__main__":
    main()
