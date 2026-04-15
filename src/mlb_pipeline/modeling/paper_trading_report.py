# src/mlb_pipeline/modeling/paper_trading_report.py
"""
MLB paper trading dashboard: W/L, ROI, CLV, weekly breakdown, prop grading.

Run standalone:
    python -m mlb_pipeline.modeling.paper_trading_report
    python -m mlb_pipeline.modeling.paper_trading_report --days 30
"""
import argparse
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.modeling.paper_trading_report")

_ET = ZoneInfo("America/New_York")
PG_DSN = "postgresql://josh:password@localhost:5432/nba"


def _roi(wins: int, n: int) -> float:
    """ROI at standard -110 juice."""
    if n == 0:
        return 0.0
    return (wins * 100 - (n - wins) * 110) / (n * 110) * 100


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num / denom * 100:.1f}%"


def print_report(conn, days: int = 90) -> None:
    cutoff = (datetime.now(_ET).date() - timedelta(days=days)).isoformat()

    # ── Overall summary ──────────────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                                       AS total_preds,
                COUNT(*) FILTER (WHERE run_line_bet_side IS NOT NULL)          AS rl_flagged,
                COUNT(*) FILTER (WHERE total_bet_side IS NOT NULL)             AS tot_flagged,
                COUNT(*) FILTER (WHERE run_line_covered IS NOT NULL)           AS rl_graded,
                SUM(CASE WHEN run_line_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE run_line_covered IS NOT NULL)                 AS rl_wins,
                COUNT(*) FILTER (WHERE total_covered IS NOT NULL)              AS tot_graded,
                SUM(CASE WHEN total_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE total_covered IS NOT NULL)                    AS tot_wins,
                AVG(edge_run_line) FILTER (WHERE edge_run_line IS NOT NULL)    AS avg_edge_rl,
                AVG(edge_total)    FILTER (WHERE edge_total IS NOT NULL)       AS avg_edge_tot,
                COUNT(*) FILTER (WHERE clv_run_line IS NOT NULL)               AS clv_rl_n,
                SUM(CASE WHEN clv_run_line > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_run_line IS NOT NULL)                     AS clv_rl_beat,
                AVG(clv_run_line)  FILTER (WHERE clv_run_line IS NOT NULL)     AS avg_clv_rl,
                COUNT(*) FILTER (WHERE clv_total IS NOT NULL)                  AS clv_tot_n,
                SUM(CASE WHEN clv_total > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_total IS NOT NULL)                        AS clv_tot_beat,
                AVG(clv_total)     FILTER (WHERE clv_total IS NOT NULL)        AS avg_clv_tot,
                COUNT(*) FILTER (WHERE clv_rl_price IS NOT NULL)               AS price_clv_rl_n,
                SUM(CASE WHEN clv_rl_price > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_rl_price IS NOT NULL)                    AS price_clv_rl_beat,
                AVG(clv_rl_price)  FILTER (WHERE clv_rl_price IS NOT NULL)    AS avg_price_clv_rl,
                COUNT(*) FILTER (WHERE clv_total_price IS NOT NULL)            AS price_clv_tot_n,
                AVG(clv_total_price) FILTER (WHERE clv_total_price IS NOT NULL) AS avg_price_clv_tot,
                MIN(game_date_et)                                               AS first_date,
                MAX(game_date_et)                                               AS last_date
            FROM bets.mlb_game_predictions
            WHERE game_date_et >= %s
        """, (cutoff,))
        ov = cur.fetchone()

    if not ov or not ov["total_preds"]:
        print(f"No MLB predictions found since {cutoff}.")
        return

    rl_n    = int(ov["rl_graded"]  or 0)
    rl_w    = int(ov["rl_wins"]    or 0)
    tot_n   = int(ov["tot_graded"] or 0)
    tot_w   = int(ov["tot_wins"]   or 0)
    clv_rl_n  = int(ov["clv_rl_n"]   or 0)
    clv_rl_b  = int(ov["clv_rl_beat"] or 0)
    clv_tot_n = int(ov["clv_tot_n"]  or 0)
    clv_tot_b = int(ov["clv_tot_beat"] or 0)
    price_clv_rl_n   = int(ov["price_clv_rl_n"]   or 0)
    price_clv_rl_b   = int(ov["price_clv_rl_beat"] or 0)
    price_clv_tot_n  = int(ov["price_clv_tot_n"]  or 0)

    print(f"\n{'='*65}")
    print(f"  MLB PAPER TRADING REPORT  ({ov['first_date']} to {ov['last_date']},  last {days}d)")
    print(f"{'='*65}")
    print(f"  Total predictions:  {ov['total_preds']}")
    print(f"  Run line flagged:   {ov['rl_flagged']}  |  Total flagged: {ov['tot_flagged']}")
    print()
    print(f"  {'Metric':<22}  {'Run Line':<18}  {'Total'}")
    print(f"  {'-'*22}  {'-'*18}  {'-'*18}")
    print(f"  {'Record':<22}  {rl_w}-{rl_n-rl_w:<12}  {tot_w}-{tot_n-tot_w}")
    print(f"  {'Win %':<22}  {_pct(rl_w, rl_n):<18}  {_pct(tot_w, tot_n)}")
    print(f"  {'ROI (-110)':<22}  {_roi(rl_w, rl_n):+.1f}%{'':<13}  {_roi(tot_w, tot_n):+.1f}%")
    avg_erl = float(ov["avg_edge_rl"]  or 0)
    avg_etot = float(ov["avg_edge_tot"] or 0)
    print(f"  {'Avg model edge':<22}  {avg_erl:+.2f} runs{'':<10}  {avg_etot:+.2f} runs")

    print()
    print("  Closing Line Value (CLV)  [positive = beat the close]")
    if clv_rl_n > 0:
        avg_crl = float(ov["avg_clv_rl"] or 0)
        print(f"  Run Line (pts): beat close {clv_rl_b}/{clv_rl_n} ({_pct(clv_rl_b, clv_rl_n)})  avg CLV = {avg_crl:+.2f} runs")
    else:
        print("  Run Line (pts): no CLV data yet")
    if price_clv_rl_n > 0:
        avg_price_crl = float(ov["avg_price_clv_rl"] or 0)
        print(f"  Run Line (price): beat close {price_clv_rl_b}/{price_clv_rl_n} ({_pct(price_clv_rl_b, price_clv_rl_n)})  avg CLV = {avg_price_crl:+.2f}%")
    if clv_tot_n > 0:
        avg_ctot = float(ov["avg_clv_tot"] or 0)
        print(f"  Total    (pts): beat close {clv_tot_b}/{clv_tot_n} ({_pct(clv_tot_b, clv_tot_n)})  avg CLV = {avg_ctot:+.2f} runs")
    if price_clv_tot_n > 0:
        avg_price_ctot = float(ov["avg_price_clv_tot"] or 0)
        print(f"  Total    (price): {price_clv_tot_n} graded  avg CLV = {avg_price_ctot:+.2f}%")

    # ── Game bets by edge bucket ──────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                bet_type,
                edge_bucket,
                COUNT(*)                                                AS n,
                SUM(CASE WHEN covered THEN 1 ELSE 0 END)               AS wins,
                AVG(CASE WHEN clv IS NOT NULL THEN clv END)            AS avg_clv
            FROM (
                SELECT
                    'run_line'                                          AS bet_type,
                    CASE
                        WHEN ABS(edge_run_line) < 1.0 THEN '0.0-1.0'
                        WHEN ABS(edge_run_line) < 1.5 THEN '1.0-1.5'
                        WHEN ABS(edge_run_line) < 2.0 THEN '1.5-2.0'
                        WHEN ABS(edge_run_line) < 2.5 THEN '2.0-2.5'
                        WHEN ABS(edge_run_line) < 3.0 THEN '2.5-3.0'
                        ELSE '3.0+'
                    END                                                 AS edge_bucket,
                    run_line_covered                                    AS covered,
                    clv_run_line                                        AS clv
                FROM bets.mlb_game_predictions
                WHERE game_date_et >= %s
                  AND run_line_covered IS NOT NULL
                  AND edge_run_line IS NOT NULL
                UNION ALL
                SELECT
                    'total',
                    CASE
                        WHEN ABS(edge_total) < 1.0 THEN '0.0-1.0'
                        WHEN ABS(edge_total) < 1.5 THEN '1.0-1.5'
                        WHEN ABS(edge_total) < 2.0 THEN '1.5-2.0'
                        WHEN ABS(edge_total) < 2.5 THEN '2.0-2.5'
                        WHEN ABS(edge_total) < 3.0 THEN '2.5-3.0'
                        ELSE '3.0+'
                    END,
                    total_covered,
                    clv_total
                FROM bets.mlb_game_predictions
                WHERE game_date_et >= %s
                  AND total_covered IS NOT NULL
                  AND edge_total IS NOT NULL
            ) sub
            GROUP BY 1, 2
            ORDER BY 1, 2
        """, (cutoff, cutoff))
        bucket_rows = cur.fetchall()

    if bucket_rows:
        print(f"\n  Game Bets by Edge Bucket  (last {days}d)")
        print(f"  {'Type':<10}  {'Edge':>7}  {'W-L':<12}  {'Win%':>6}  {'ROI':>8}  {'AvgCLV':>8}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}")
        for r in bucket_rows:
            n, w = int(r["n"]), int(r["wins"])
            clv_str = f"{float(r['avg_clv']):+.2f}" if r["avg_clv"] is not None else "  N/A"
            print(f"  {r['bet_type']:<10}  {r['edge_bucket']:>7}  {w}-{n-w:<8}  "
                  f"{_pct(w, n):>6}  {_roi(w, n):>+7.1f}%  {clv_str:>8}")

    # ── Bet side breakdown ────────────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                run_line_bet_side                                       AS side,
                COUNT(*) FILTER (WHERE run_line_covered IS NOT NULL)   AS rl_n,
                SUM(CASE WHEN run_line_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE run_line_covered IS NOT NULL)         AS rl_w,
                total_bet_side                                          AS tot_side,
                COUNT(*) FILTER (WHERE total_covered IS NOT NULL)      AS tot_n,
                SUM(CASE WHEN total_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE total_covered IS NOT NULL)            AS tot_w
            FROM bets.mlb_game_predictions
            WHERE game_date_et >= %s
            GROUP BY 1, 4
            ORDER BY 1, 4
        """, (cutoff,))
        side_rows = cur.fetchall()

    if side_rows:
        seen_rl, seen_tot = set(), set()
        rl_lines, tot_lines = [], []
        for r in side_rows:
            s = r["side"]
            if s and s not in seen_rl:
                seen_rl.add(s)
                n, w = int(r["rl_n"] or 0), int(r["rl_w"] or 0)
                if n:
                    rl_lines.append(f"{s:<6} {w}-{n-w} ({_pct(w,n)}) ROI:{_roi(w,n):+.1f}%")
            ts = r["tot_side"]
            if ts and ts not in seen_tot:
                seen_tot.add(ts)
                n, w = int(r["tot_n"] or 0), int(r["tot_w"] or 0)
                if n:
                    tot_lines.append(f"{ts:<6} {w}-{n-w} ({_pct(w,n)}) ROI:{_roi(w,n):+.1f}%")
        if rl_lines or tot_lines:
            print(f"\n  Bet Side Breakdown")
            if rl_lines:
                print(f"  Run Line:  " + "   |   ".join(rl_lines))
            if tot_lines:
                print(f"  Total:     " + "   |   ".join(tot_lines))

    # ── Day game vs night game split ──────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                (EXTRACT(HOUR FROM g.start_ts_utc AT TIME ZONE 'America/New_York') < 17)
                    AS is_day_game,
                COUNT(*) FILTER (WHERE p.run_line_covered IS NOT NULL)  AS rl_n,
                SUM(CASE WHEN p.run_line_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE p.run_line_covered IS NOT NULL)        AS rl_w,
                COUNT(*) FILTER (WHERE p.total_covered IS NOT NULL)     AS tot_n,
                SUM(CASE WHEN p.total_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE p.total_covered IS NOT NULL)           AS tot_w
            FROM bets.mlb_game_predictions p
            JOIN raw.mlb_games g ON g.game_slug = p.game_slug
            WHERE p.game_date_et >= %s
              AND g.start_ts_utc IS NOT NULL
              AND (p.run_line_covered IS NOT NULL OR p.total_covered IS NOT NULL)
            GROUP BY 1
            ORDER BY 1
        """, (cutoff,))
        day_rows = cur.fetchall()

    if day_rows and len(day_rows) > 1:
        print(f"\n  Day vs Night Game Split")
        print(f"  {'Context':<10}  {'Run Line':<22}  {'Total'}")
        print(f"  {'-'*10}  {'-'*22}  {'-'*22}")
        for r in day_rows:
            label = "Day" if r["is_day_game"] else "Night"
            rln, rlw = int(r["rl_n"] or 0), int(r["rl_w"] or 0)
            ton, tow = int(r["tot_n"] or 0), int(r["tot_w"] or 0)
            rl_str  = f"{rlw}-{rln-rlw} ({_pct(rlw,rln)}) {_roi(rlw,rln):+.1f}%" if rln else "N/A"
            tot_str = f"{tow}-{ton-tow} ({_pct(tow,ton)}) {_roi(tow,ton):+.1f}%" if ton else "N/A"
            print(f"  {label:<10}  {rl_str:<22}  {tot_str}")

    # ── Weekly breakdown ─────────────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                DATE_TRUNC('week', game_date_et)::date                        AS week_start,
                COUNT(*) FILTER (WHERE run_line_covered IS NOT NULL)           AS rl_n,
                SUM(CASE WHEN run_line_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE run_line_covered IS NOT NULL)                 AS rl_w,
                COUNT(*) FILTER (WHERE total_covered IS NOT NULL)              AS tot_n,
                SUM(CASE WHEN total_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE total_covered IS NOT NULL)                    AS tot_w,
                AVG(clv_run_line) FILTER (WHERE clv_run_line IS NOT NULL)      AS avg_clv_rl,
                AVG(clv_total)    FILTER (WHERE clv_total IS NOT NULL)         AS avg_clv_tot
            FROM bets.mlb_game_predictions
            WHERE game_date_et >= %s
              AND (run_line_bet_side IS NOT NULL OR total_bet_side IS NOT NULL)
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 14
        """, (cutoff,))
        weeks = cur.fetchall()

    if weeks:
        print(f"\n  {'Week':<12}  {'Run Line':<16}  {'Total':<16}  {'CLV-RL':>7}  {'CLV-Tot':>7}")
        print(f"  {'-'*12}  {'-'*16}  {'-'*16}  {'-'*7}  {'-'*7}")
        for w in weeks:
            rln = int(w["rl_n"] or 0)
            rlw = int(w["rl_w"] or 0)
            ton = int(w["tot_n"] or 0)
            tow = int(w["tot_w"] or 0)
            crl = f"{float(w['avg_clv_rl']):+.2f}" if w["avg_clv_rl"] is not None else "  N/A "
            ctot = f"{float(w['avg_clv_tot']):+.2f}" if w["avg_clv_tot"] is not None else "  N/A "
            rl_str  = f"{rlw}-{rln-rlw} ({_pct(rlw, rln)})" if rln else "N/A"
            tot_str = f"{tow}-{ton-tow} ({_pct(tow, ton)})" if ton else "N/A"
            print(f"  {str(w['week_start']):<12}  {rl_str:<16}  {tot_str:<16}  {crl:>7}  {ctot:>7}")

    # ── Prop predictions summary ─────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                stat,
                COUNT(*)                                            AS total,
                COUNT(*) FILTER (WHERE actual_value IS NOT NULL)   AS graded,
                AVG(ABS(pred_value - actual_value))
                    FILTER (WHERE actual_value IS NOT NULL)         AS mae,
                AVG((over_hit::int))
                    FILTER (WHERE over_hit IS NOT NULL)             AS over_rate
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %s
            GROUP BY stat
            ORDER BY stat
        """, (cutoff,))
        prop_stats = cur.fetchall()

    if prop_stats and any(int(r["graded"] or 0) > 0 for r in prop_stats):
        print(f"\n  Prop Predictions by Stat  (last {days}d)")
        print(f"  {'Stat':<25}  {'Graded':>6}  {'MAE':>6}  {'OverRate':>9}")
        print(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*9}")
        for r in prop_stats:
            graded = int(r["graded"] or 0)
            if graded == 0:
                continue
            mae = float(r["mae"] or 0)
            over_rate = float(r["over_rate"] or 0) if r["over_rate"] is not None else None
            over_str = f"{over_rate*100:.1f}%" if over_rate is not None else "N/A"
            print(f"  {r['stat']:<25}  {graded:>6}  {mae:>6.2f}  {over_str:>9}")

    # ── Over/under bias by stat ──────────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT stat,
                   COUNT(*)              AS n,
                   AVG(over_hit::int)    AS actual_over_rate,
                   AVG(edge)             AS avg_edge
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %s
              AND over_hit IS NOT NULL
              AND edge IS NOT NULL
            GROUP BY stat
            ORDER BY stat
        """, (cutoff,))
        bias_rows = cur.fetchall()

    if bias_rows:
        print(f"\n  Over/Under Bias  [50% = fair]")
        print(f"  {'Stat':<25}  {'n':>5}  {'ActualOver%':>11}  {'AvgEdge':>9}")
        print(f"  {'-'*25}  {'-'*5}  {'-'*11}  {'-'*9}")
        for b in bias_rows:
            n = int(b["n"])
            over_rate = float(b["actual_over_rate"] or 0)
            avg_edge  = float(b["avg_edge"] or 0)
            flag = "  (inflated!)" if over_rate < 0.42 else ""
            print(f"  {b['stat']:<25}  {n:>5}  {over_rate*100:>10.1f}%  {avg_edge:>+9.3f}{flag}")

    # ── Prop bet grading by edge bucket ──────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                stat,
                CASE WHEN ABS(edge) < 0.5 THEN '0-0.5'
                     WHEN ABS(edge) < 1.0 THEN '0.5-1'
                     WHEN ABS(edge) < 2.0 THEN '1-2'
                     ELSE '2+' END                                    AS edge_bucket,
                CASE WHEN edge > 0 THEN 'over' ELSE 'under' END       AS direction,
                COUNT(*)                                               AS n,
                SUM(CASE WHEN (edge > 0 AND over_hit)
                           OR (edge < 0 AND NOT over_hit)
                          THEN 1 ELSE 0 END)                          AS wins
            FROM bets.mlb_prop_predictions
            WHERE game_date_et >= %s
              AND edge IS NOT NULL
              AND over_hit IS NOT NULL
            GROUP BY 1, 2, 3
            ORDER BY 1, 2
        """, (cutoff,))
        grade_rows = cur.fetchall()

    if grade_rows:
        print(f"\n  Prop Bet Grading by Edge Bucket  (last {days}d)")
        print(f"  {'Stat':<25}  {'Edge':>6}  {'Dir':<6}  {'W-L':<12}  {'Win%'}")
        print(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*6}")
        for r in grade_rows:
            n, w = int(r["n"]), int(r["wins"])
            print(f"  {r['stat']:<25}  {r['edge_bucket']:>6}  {r['direction']:<6}  "
                  f"{w}-{n-w:<8}  {_pct(w, n)}")

    print(f"\n{'='*65}\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="MLB paper trading dashboard")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days (default 90)")
    args = parser.parse_args()

    with psycopg2.connect(PG_DSN) as conn:
        print_report(conn, days=args.days)


if __name__ == "__main__":
    main()
