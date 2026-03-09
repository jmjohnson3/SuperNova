# src/nba_pipeline/modeling/paper_trading_report.py
"""
Paper trading dashboard: comprehensive W/L, ROI, CLV, and weekly breakdown.

Run standalone:
    python -m nba_pipeline.modeling.paper_trading_report

Or with a specific lookback:
    python -m nba_pipeline.modeling.paper_trading_report --days 30
"""
import argparse
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

log = logging.getLogger("nba_pipeline.modeling.paper_trading_report")

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
    return f"{num/denom*100:.1f}%"


def print_report(conn, days: int = 90) -> None:
    cutoff = (datetime.now(_ET).date() - timedelta(days=days)).isoformat()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Overall summary
        cur.execute("""
            SELECT
                COUNT(*)                                                    AS total_preds,
                COUNT(*) FILTER (WHERE spread_bet_side IS NOT NULL)        AS spread_flagged,
                COUNT(*) FILTER (WHERE total_bet_side IS NOT NULL)         AS total_flagged,
                COUNT(*) FILTER (WHERE spread_covered IS NOT NULL)         AS spread_graded,
                SUM(CASE WHEN spread_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE spread_covered IS NOT NULL)               AS spread_wins,
                COUNT(*) FILTER (WHERE total_correct IS NOT NULL)          AS total_graded,
                SUM(CASE WHEN total_correct THEN 1 ELSE 0 END)
                    FILTER (WHERE total_correct IS NOT NULL)                AS total_wins,
                COUNT(*) FILTER (WHERE direction_correct IS NOT NULL)      AS dir_graded,
                SUM(CASE WHEN direction_correct THEN 1 ELSE 0 END)
                    FILTER (WHERE direction_correct IS NOT NULL)            AS dir_wins,
                AVG(edge_spread)  FILTER (WHERE edge_spread IS NOT NULL)   AS avg_edge_spread,
                AVG(edge_total)   FILTER (WHERE edge_total IS NOT NULL)    AS avg_edge_total,
                COUNT(*) FILTER (WHERE clv_spread IS NOT NULL)             AS clv_spread_n,
                SUM(CASE WHEN clv_spread > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_spread IS NOT NULL)                   AS clv_spread_beat,
                AVG(clv_spread)   FILTER (WHERE clv_spread IS NOT NULL)    AS avg_clv_spread,
                COUNT(*) FILTER (WHERE clv_total IS NOT NULL)              AS clv_total_n,
                SUM(CASE WHEN clv_total > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_total IS NOT NULL)                    AS clv_total_beat,
                AVG(clv_total)    FILTER (WHERE clv_total IS NOT NULL)     AS avg_clv_total,
                MIN(game_date_et)                                           AS first_date,
                MAX(game_date_et)                                           AS last_date
            FROM bets.game_predictions
            WHERE game_date_et >= %s
        """, (cutoff,))
        ov = cur.fetchone()

    if not ov or not ov["total_preds"]:
        print(f"No predictions found since {cutoff}.")
        return

    sp_n = int(ov["spread_graded"] or 0)
    sp_w = int(ov["spread_wins"] or 0)
    to_n = int(ov["total_graded"] or 0)
    to_w = int(ov["total_wins"] or 0)
    di_n = int(ov["dir_graded"] or 0)
    di_w = int(ov["dir_wins"] or 0)
    cl_sn = int(ov["clv_spread_n"] or 0)
    cl_sb = int(ov["clv_spread_beat"] or 0)
    cl_tn = int(ov["clv_total_n"] or 0)
    cl_tb = int(ov["clv_total_beat"] or 0)

    print(f"\n{'='*65}")
    print(f"  PAPER TRADING REPORT  ({ov['first_date']} to {ov['last_date']},  last {days}d)")
    print(f"{'='*65}")
    print(f"  Total predictions: {ov['total_preds']}")
    print(f"  Spread flagged:    {ov['spread_flagged']}  |  Total flagged: {ov['total_flagged']}")
    print()
    print(f"  {'Metric':<22}  {'Spread':<18}  {'Total':<18}  {'Direction'}")
    print(f"  {'-'*22}  {'-'*18}  {'-'*18}  {'-'*15}")
    print(f"  {'Record':<22}  {sp_w}-{sp_n-sp_w:<12}  {to_w}-{to_n-to_w:<12}  {di_w}/{di_n}")
    print(f"  {'Win %':<22}  {_pct(sp_w, sp_n):<18}  {_pct(to_w, to_n):<18}  {_pct(di_w, di_n)}")
    print(f"  {'ROI (-110)':<22}  {_roi(sp_w, sp_n):+.1f}%{'':<13}  {_roi(to_w, to_n):+.1f}%")
    avg_es = float(ov["avg_edge_spread"] or 0)
    avg_et = float(ov["avg_edge_total"] or 0)
    print(f"  {'Avg model edge':<22}  {avg_es:+.2f} pts{'':<11}  {avg_et:+.2f} pts")

    print()
    print(f"  Closing Line Value (CLV)  [positive = beat the close]")
    if cl_sn > 0:
        avg_cs = float(ov["avg_clv_spread"] or 0)
        print(f"  Spread: beat close {cl_sb}/{cl_sn} ({_pct(cl_sb, cl_sn)})  avg CLV = {avg_cs:+.2f} pts")
    else:
        print(f"  Spread: no CLV data (closing lines not yet in DB)")
    if cl_tn > 0:
        avg_ct = float(ov["avg_clv_total"] or 0)
        print(f"  Total:  beat close {cl_tb}/{cl_tn} ({_pct(cl_tb, cl_tn)})  avg CLV = {avg_ct:+.2f} pts")
    else:
        print(f"  Total:  no CLV data")

    # Weekly breakdown
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                DATE_TRUNC('week', game_date_et)::date                      AS week_start,
                COUNT(*) FILTER (WHERE spread_covered IS NOT NULL)           AS sp_n,
                SUM(CASE WHEN spread_covered THEN 1 ELSE 0 END)
                    FILTER (WHERE spread_covered IS NOT NULL)                 AS sp_w,
                COUNT(*) FILTER (WHERE total_correct IS NOT NULL)            AS to_n,
                SUM(CASE WHEN total_correct THEN 1 ELSE 0 END)
                    FILTER (WHERE total_correct IS NOT NULL)                  AS to_w,
                AVG(clv_spread)  FILTER (WHERE clv_spread IS NOT NULL)       AS avg_clv_s,
                AVG(clv_total)   FILTER (WHERE clv_total IS NOT NULL)        AS avg_clv_t
            FROM bets.game_predictions
            WHERE game_date_et >= %s
              AND (spread_bet_side IS NOT NULL OR total_bet_side IS NOT NULL)
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 12
        """, (cutoff,))
        weeks = cur.fetchall()

    if weeks:
        print(f"\n  {'Week':<12}  {'Spread':<14}  {'Total':<14}  {'CLV-Sprd':>8}  {'CLV-Tot':>7}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*7}")
        for w in weeks:
            spn = int(w["sp_n"] or 0)
            spw = int(w["sp_w"] or 0)
            ton = int(w["to_n"] or 0)
            tow = int(w["to_w"] or 0)
            clv_s = f"{float(w['avg_clv_s']):+.2f}" if w["avg_clv_s"] is not None else "  N/A  "
            clv_t = f"{float(w['avg_clv_t']):+.2f}" if w["avg_clv_t"] is not None else "  N/A  "
            sp_str = f"{spw}-{spn-spw} ({_pct(spw, spn)})" if spn else "N/A"
            to_str = f"{tow}-{ton-tow} ({_pct(tow, ton)})" if ton else "N/A"
            print(f"  {str(w['week_start']):<12}  {sp_str:<14}  {to_str:<14}  {clv_s:>8}  {clv_t:>7}")

    # Prop predictions summary
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                              AS total_props,
                COUNT(*) FILTER (WHERE actual_points IS NOT NULL)    AS graded_props,
                AVG(ABS(pred_points - actual_points))
                    FILTER (WHERE actual_points IS NOT NULL)          AS pts_mae,
                AVG(ABS(pred_rebounds - actual_rebounds))
                    FILTER (WHERE actual_rebounds IS NOT NULL)        AS reb_mae,
                AVG(ABS(pred_assists - actual_assists))
                    FILTER (WHERE actual_assists IS NOT NULL)         AS ast_mae
            FROM bets.prop_predictions
            WHERE game_date_et >= %s
        """, (cutoff,))
        pr = cur.fetchone()

    if pr and pr["graded_props"]:
        print(f"\n  Prop Predictions ({pr['graded_props']}/{pr['total_props']} graded)")
        pts_mae = float(pr["pts_mae"] or 0)
        reb_mae = float(pr["reb_mae"] or 0)
        ast_mae = float(pr["ast_mae"] or 0)
        print(f"  MAE  PTS: {pts_mae:.2f}  REB: {reb_mae:.2f}  AST: {ast_mae:.2f}")

    print(f"\n{'='*65}\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="Paper trading dashboard")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days (default 90)")
    args = parser.parse_args()

    with psycopg2.connect(PG_DSN) as conn:
        print_report(conn, days=args.days)


if __name__ == "__main__":
    main()
