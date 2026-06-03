# src/mlb_pipeline/modeling/update_outcomes.py
"""
Fill actual scores and ATS results for completed MLB games in bets.mlb_game_predictions,
and actual stats for bets.mlb_prop_predictions.

Run after games complete (nightly):
    python -m mlb_pipeline.modeling.update_outcomes
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from .bankroll_ledger import grade_bankroll_ledger
from .model_pick_ledger import grade_model_pick_ledger

log = logging.getLogger("mlb_pipeline.modeling.update_outcomes")

_ET = ZoneInfo("America/New_York")
PG_DSN = "postgresql://josh:password@localhost:5432/nba"

def _american_to_prob(price: float) -> float:
    """Convert American odds price to raw implied probability (no vig removal)."""
    if price >= 100:
        return 100.0 / (price + 100.0)
    else:
        return abs(price) / (abs(price) + 100.0)


def _price_clv(entry_price: float, closing_price: float) -> float:
    """Price CLV in probability percentage points.

    Positive = we got a better price than the closing line (beat the close).
    entry_price and closing_price should be for the SAME side we bet.
    """
    return round((_american_to_prob(closing_price) - _american_to_prob(entry_price)) * 100, 2)


# Stat column mapping: prop stat name → mlb_player_gamelogs column
_STAT_COL = {
    "pitcher_strikeouts": "strikeouts_pitcher",
    "batter_hits":        "hits",
    "batter_home_runs":   "home_runs",
    "batter_total_bases": "total_bases",
    "batter_walks":       "walks_batter",
}


def update_game_outcomes(conn) -> int:
    """Fill actuals and ATS results for completed MLB games.

    For each row in bets.mlb_game_predictions where actual_home_score IS NULL
    and the game has a final score in raw.mlb_games, compute:
      - actual_home_score, actual_away_score, actual_run_diff, actual_total
      - run_line_covered  (did the bet side win vs the run line?)
      - total_covered     (did the bet side win vs the total?)
      - closing_run_line, closing_total  (latest fetched line from odds.mlb_game_lines)
      - clv_run_line, clv_total          (Closing Line Value — positive = beat the close)

    Returns number of rows updated.
    """
    today = datetime.now(_ET).date()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, game_date_et,
                   home_team_abbr, away_team_abbr,
                   market_run_line, market_total,
                   run_line_bet_side, total_bet_side,
                   pred_run_diff,
                   market_rl_price, market_total_price
            FROM bets.mlb_game_predictions
            WHERE actual_home_score IS NULL
              AND game_date_et <= %s
        """, (today,))
        pending = cur.fetchall()

    if not pending:
        log.info("No pending MLB game predictions to update.")
        return 0

    slugs = [r["game_slug"] for r in pending]

    # Pull final scores
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, home_score, away_score
            FROM raw.mlb_games
            WHERE game_slug = ANY(%s)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        """, (slugs,))
        finals = {r["game_slug"]: r for r in cur.fetchall()}

    if not finals:
        log.info("No final MLB games found for %d pending predictions.", len(pending))
        return 0

    # Fetch closing lines: latest fetched snapshot per (home_team, away_team, as_of_date).
    # Use FanDuel if available, else any bookmaker (DISTINCT ON picks the first after ORDER BY).
    dates = list({r["game_date_et"] for r in pending})
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (home_team, away_team, as_of_date)
                    home_team, away_team, as_of_date,
                    spread_home_points  AS closing_run_line,
                    total_points        AS closing_total,
                    spread_home_price   AS closing_rl_home_price,
                    spread_away_price   AS closing_rl_away_price,
                    total_over_price    AS closing_total_over_price,
                    total_under_price   AS closing_total_under_price
                FROM odds.mlb_game_lines
                WHERE as_of_date = ANY(%s::date[])
                  AND spread_home_points IS NOT NULL
                ORDER BY
                    home_team, away_team, as_of_date,
                    -- prefer FanDuel as canonical close; else latest fetch
                    CASE WHEN bookmaker_key = 'fanduel' THEN 0 ELSE 1 END,
                    fetched_at_utc DESC
            """, (dates,))
            closing_lines = {
                (r["home_team"], r["away_team"], r["as_of_date"]): r
                for r in cur.fetchall()
            }
    except Exception as exc:
        log.warning("Could not fetch MLB closing lines for CLV: %s", exc)
        closing_lines = {}

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            slug = row["game_slug"]
            if slug not in finals:
                continue

            g = finals[slug]
            home_score = float(g["home_score"])
            away_score = float(g["away_score"])
            actual_run_diff = home_score - away_score
            actual_total    = home_score + away_score

            # ── Run line ATS ────────────────────────────────────────────────
            # market_run_line follows Odds API sign convention:
            #   negative = home is FAVORITE (e.g. -1.5 means home gives 1.5 runs)
            #   positive = home is UNDERDOG (e.g. +1.5 means home gets 1.5 runs)
            # Home covers when: actual_run_diff > -market_run_line
            run_line_covered = None
            if row["market_run_line"] is not None and row["run_line_bet_side"] is not None:
                mkt_rl = float(row["market_run_line"])
                home_covered = actual_run_diff > -mkt_rl
                if row["run_line_bet_side"] == "home":
                    run_line_covered = home_covered
                else:
                    run_line_covered = not home_covered

            # ── Total ────────────────────────────────────────────────────────
            total_covered = None
            if row["market_total"] is not None and row["total_bet_side"] is not None:
                mkt_t = float(row["market_total"])
                if row["total_bet_side"] == "over":
                    total_covered = actual_total > mkt_t
                else:
                    total_covered = actual_total < mkt_t

            # ── CLV ──────────────────────────────────────────────────────────
            # Positive CLV = we beat the closing line (sharp signal).
            # home bet: clv = market_run_line - closing_run_line
            # away bet: clv = closing_run_line - market_run_line
            # over bet: clv = closing_total - market_total
            # under bet: clv = market_total - closing_total
            cl_key = (row["home_team_abbr"], row["away_team_abbr"], row["game_date_et"])
            cl = closing_lines.get(cl_key)

            closing_rl = float(cl["closing_run_line"]) if cl and cl["closing_run_line"] is not None else None
            closing_tot = float(cl["closing_total"]) if cl and cl["closing_total"] is not None else None

            mkt_rl = float(row["market_run_line"]) if row["market_run_line"] is not None else None
            mkt_t  = float(row["market_total"])    if row["market_total"]    is not None else None

            clv_rl = None
            if closing_rl is not None and mkt_rl is not None and row["run_line_bet_side"] is not None:
                if row["run_line_bet_side"] == "home":
                    clv_rl = round(mkt_rl - closing_rl, 2)
                else:
                    clv_rl = round(closing_rl - mkt_rl, 2)

            clv_tot = None
            if closing_tot is not None and mkt_t is not None and row["total_bet_side"] is not None:
                if row["total_bet_side"] == "over":
                    clv_tot = round(closing_tot - mkt_t, 2)
                else:
                    clv_tot = round(mkt_t - closing_tot, 2)

            # Price-based CLV (in probability % points; positive = beat the close)
            closing_rl_price = None
            clv_rl_price = None
            closing_total_price = None
            clv_total_price = None
            entry_rl_price = row.get("market_rl_price")
            entry_tot_price = row.get("market_total_price")

            if cl is not None and row["run_line_bet_side"] is not None:
                if row["run_line_bet_side"] == "home":
                    closing_rl_price = cl.get("closing_rl_home_price")
                else:
                    closing_rl_price = cl.get("closing_rl_away_price")
            if cl is not None and row["total_bet_side"] is not None:
                if row["total_bet_side"] == "over":
                    closing_total_price = cl.get("closing_total_over_price")
                else:
                    closing_total_price = cl.get("closing_total_under_price")

            if closing_rl_price is not None and entry_rl_price is not None:
                clv_rl_price = _price_clv(float(entry_rl_price), float(closing_rl_price))
            if closing_total_price is not None and entry_tot_price is not None:
                clv_total_price = _price_clv(float(entry_tot_price), float(closing_total_price))

            cur.execute("""
                UPDATE bets.mlb_game_predictions
                SET actual_home_score  = %s,
                    actual_away_score  = %s,
                    actual_run_diff    = %s,
                    actual_total       = %s,
                    run_line_covered   = %s,
                    total_covered      = %s,
                    closing_run_line   = %s,
                    closing_total      = %s,
                    clv_run_line       = %s,
                    clv_total          = %s,
                    closing_rl_price   = %s,
                    closing_total_price= %s,
                    clv_rl_price       = %s,
                    clv_total_price    = %s,
                    updated_at_utc     = NOW()
                WHERE game_slug = %s
            """, (
                home_score, away_score, actual_run_diff, actual_total,
                run_line_covered, total_covered,
                closing_rl, closing_tot,
                clv_rl, clv_tot,
                closing_rl_price, closing_total_price,
                clv_rl_price, clv_total_price,
                row["game_slug"],
            ))
            updated += 1

    conn.commit()
    log.info("update_game_outcomes: updated %d rows", updated)
    return updated


def backfill_clv(conn) -> int:
    """Recompute CLV for already-graded rows where clv_run_line IS NULL
    but a closing line is now available (e.g. evening crawl ran after grading).

    Returns number of rows updated.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, game_date_et, home_team_abbr, away_team_abbr,
                   market_run_line, market_total,
                   run_line_bet_side, total_bet_side,
                   market_rl_price, market_total_price
            FROM bets.mlb_game_predictions
            WHERE actual_home_score IS NOT NULL
              AND (clv_run_line IS NULL OR clv_rl_price IS NULL)
              AND (run_line_bet_side IS NOT NULL OR total_bet_side IS NOT NULL)
        """)
        rows = cur.fetchall()

    if not rows:
        log.info("backfill_clv: nothing to update.")
        return 0

    dates = list({r["game_date_et"] for r in rows})
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (home_team, away_team, as_of_date)
                    home_team, away_team, as_of_date,
                    spread_home_points  AS closing_run_line,
                    total_points        AS closing_total,
                    spread_home_price   AS closing_rl_home_price,
                    spread_away_price   AS closing_rl_away_price,
                    total_over_price    AS closing_total_over_price,
                    total_under_price   AS closing_total_under_price
                FROM odds.mlb_game_lines
                WHERE as_of_date = ANY(%s::date[])
                  AND spread_home_points IS NOT NULL
                ORDER BY
                    home_team, away_team, as_of_date,
                    CASE WHEN bookmaker_key = 'fanduel' THEN 0 ELSE 1 END,
                    fetched_at_utc DESC
            """, (dates,))
            closing_lines = {
                (r["home_team"], r["away_team"], r["as_of_date"]): r
                for r in cur.fetchall()
            }
    except Exception as exc:
        log.warning("backfill_clv: could not fetch closing lines: %s", exc)
        return 0

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            cl_key = (row["home_team_abbr"], row["away_team_abbr"], row["game_date_et"])
            cl = closing_lines.get(cl_key)
            if not cl:
                continue

            closing_rl  = float(cl["closing_run_line"]) if cl["closing_run_line"]  is not None else None
            closing_tot = float(cl["closing_total"])    if cl["closing_total"]     is not None else None
            mkt_rl = float(row["market_run_line"]) if row["market_run_line"] is not None else None
            mkt_t  = float(row["market_total"])    if row["market_total"]    is not None else None

            clv_rl = clv_tot = None
            if closing_rl is not None and mkt_rl is not None and row["run_line_bet_side"] is not None:
                if row["run_line_bet_side"] == "home":
                    clv_rl = round(mkt_rl - closing_rl, 2)
                else:
                    clv_rl = round(closing_rl - mkt_rl, 2)
            if closing_tot is not None and mkt_t is not None and row["total_bet_side"] is not None:
                if row["total_bet_side"] == "over":
                    clv_tot = round(closing_tot - mkt_t, 2)
                else:
                    clv_tot = round(mkt_t - closing_tot, 2)

            # Price CLV for historical rows that have stored entry prices
            entry_rl_price  = row.get("market_rl_price")
            entry_tot_price = row.get("market_total_price")
            closing_rl_price = closing_total_price = None
            clv_rl_price = clv_total_price = None
            if cl is not None and row["run_line_bet_side"] is not None:
                closing_rl_price = (cl.get("closing_rl_home_price")
                                    if row["run_line_bet_side"] == "home"
                                    else cl.get("closing_rl_away_price"))
            if cl is not None and row["total_bet_side"] is not None:
                closing_total_price = (cl.get("closing_total_over_price")
                                       if row["total_bet_side"] == "over"
                                       else cl.get("closing_total_under_price"))
            if closing_rl_price is not None and entry_rl_price is not None:
                clv_rl_price = _price_clv(float(entry_rl_price), float(closing_rl_price))
            if closing_total_price is not None and entry_tot_price is not None:
                clv_total_price = _price_clv(float(entry_tot_price), float(closing_total_price))

            if clv_rl is None and clv_tot is None and clv_rl_price is None:
                continue

            cur.execute("""
                UPDATE bets.mlb_game_predictions
                SET closing_run_line  = COALESCE(closing_run_line, %s),
                    closing_total     = COALESCE(closing_total, %s),
                    clv_run_line      = %s,
                    clv_total         = %s,
                    closing_rl_price  = COALESCE(closing_rl_price, %s),
                    closing_total_price = COALESCE(closing_total_price, %s),
                    clv_rl_price      = %s,
                    clv_total_price   = %s,
                    updated_at_utc    = NOW()
                WHERE game_slug = %s
            """, (closing_rl, closing_tot, clv_rl, clv_tot,
                  closing_rl_price, closing_total_price,
                  clv_rl_price, clv_total_price,
                  row["game_slug"]))
            updated += 1

    conn.commit()
    log.info("backfill_clv: updated %d rows", updated)
    return updated


def update_prop_outcomes(conn) -> int:
    """Fill actual stat values and over_hit flag for completed MLB prop predictions.

    Pulls actuals from raw.mlb_player_gamelogs for games marked 'final'.
    Supported stats: pitcher_strikeouts, batter_hits, batter_home_runs,
                     batter_total_bases, batter_walks.

    Returns number of rows updated.
    """
    today = datetime.now(_ET).date()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, game_slug, player_id, stat, book_line
            FROM bets.mlb_prop_predictions
            WHERE actual_value IS NULL
              AND game_date_et <= %s
        """, (today,))
        pending = cur.fetchall()

    if not pending:
        log.info("No pending MLB prop predictions to update.")
        return 0

    slugs      = list({r["game_slug"]  for r in pending})
    player_ids = list({int(r["player_id"]) for r in pending})

    # Pull all relevant gamelogs from finished games in one query
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT gl.game_slug, gl.player_id,
                   gl.strikeouts_pitcher,
                   gl.hits,
                   gl.home_runs,
                   gl.total_bases,
                   gl.walks_batter
            FROM raw.mlb_player_gamelogs gl
            JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
            WHERE gl.game_slug = ANY(%s)
              AND gl.player_id = ANY(%s)
              AND g.status = 'final'
        """, (slugs, player_ids))
        # Key: (game_slug, player_id)
        actuals = {
            (r["game_slug"], int(r["player_id"])): r
            for r in cur.fetchall()
        }

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            key = (row["game_slug"], int(row["player_id"]))
            if key not in actuals:
                continue

            a = actuals[key]
            stat = row["stat"]
            gl_col = _STAT_COL.get(stat)
            if gl_col is None:
                log.warning("Unknown stat '%s' for prop id=%s — skipping", stat, row["id"])
                continue

            raw_val = a[gl_col]
            if raw_val is None:
                continue

            actual_value = float(raw_val)
            over_hit = None
            if row["book_line"] is not None:
                over_hit = actual_value > float(row["book_line"])

            cur.execute("""
                UPDATE bets.mlb_prop_predictions
                SET actual_value = %s,
                    over_hit     = %s
                WHERE id = %s
            """, (actual_value, over_hit, row["id"]))
            updated += 1

    conn.commit()
    log.info("update_prop_outcomes: updated %d prop rows", updated)
    return updated


def print_running_record(conn) -> None:
    """Print a one-line MLB ATS/total record summary with CLV stats."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE run_line_covered IS NOT NULL)        AS rl_n,
                SUM(CASE WHEN run_line_covered = TRUE THEN 1 ELSE 0 END)    AS rl_wins,
                COUNT(*) FILTER (WHERE total_covered IS NOT NULL)           AS tot_n,
                SUM(CASE WHEN total_covered = TRUE THEN 1 ELSE 0 END)       AS tot_wins,
                COUNT(*) FILTER (WHERE clv_run_line IS NOT NULL)            AS clv_rl_n,
                SUM(CASE WHEN clv_run_line > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_run_line IS NOT NULL)                  AS clv_rl_beat,
                AVG(clv_run_line) FILTER (WHERE clv_run_line IS NOT NULL)   AS avg_clv_rl,
                COUNT(*) FILTER (WHERE clv_total IS NOT NULL)               AS clv_tot_n,
                AVG(clv_total)    FILTER (WHERE clv_total IS NOT NULL)      AS avg_clv_tot,
                COUNT(*) FILTER (WHERE clv_rl_price IS NOT NULL)            AS price_clv_rl_n,
                AVG(clv_rl_price) FILTER (WHERE clv_rl_price IS NOT NULL)   AS avg_price_clv_rl
            FROM bets.mlb_game_predictions
            WHERE run_line_bet_side IS NOT NULL OR total_bet_side IS NOT NULL
        """)
        row = cur.fetchone()

    if not row:
        return

    rl_n     = int(row["rl_n"]   or 0)
    rl_w     = int(row["rl_wins"] or 0)
    tot_n    = int(row["tot_n"]  or 0)
    tot_w    = int(row["tot_wins"] or 0)
    clv_rl_n = int(row["clv_rl_n"] or 0)
    clv_rl_b = int(row["clv_rl_beat"] or 0)
    avg_clv_rl  = float(row["avg_clv_rl"]  or 0.0)
    clv_tot_n = int(row["clv_tot_n"] or 0)
    avg_clv_tot = float(row["avg_clv_tot"] or 0.0)
    price_clv_rl_n  = int(row["price_clv_rl_n"] or 0)
    avg_price_clv_rl = float(row["avg_price_clv_rl"] or 0.0)

    rl_pct  = (rl_w / rl_n * 100)   if rl_n  > 0 else 0.0
    tot_pct = (tot_w / tot_n * 100) if tot_n > 0 else 0.0
    rl_roi  = ((rl_w * 100  - (rl_n  - rl_w)  * 110) / (rl_n  * 110) * 100) if rl_n  > 0 else 0.0
    tot_roi = ((tot_w * 100 - (tot_n - tot_w) * 110) / (tot_n * 110) * 100) if tot_n > 0 else 0.0
    clv_rl_pct = (clv_rl_b / clv_rl_n * 100) if clv_rl_n > 0 else 0.0

    print(
        f"MLB Run Line: {rl_w}-{rl_n - rl_w} ({rl_pct:.1f}%) ROI: {rl_roi:+.1f}%"
        f" | Total: {tot_w}-{tot_n - tot_w} ({tot_pct:.1f}%) ROI: {tot_roi:+.1f}%"
    )
    if clv_rl_n > 0:
        print(
            f"MLB CLV Run Line: beat close {clv_rl_b}/{clv_rl_n} ({clv_rl_pct:.0f}%)"
            f" avg CLV={avg_clv_rl:+.2f} runs"
            + (f" | CLV Total avg={avg_clv_tot:+.2f} runs" if clv_tot_n > 0 else "")
        )
    if price_clv_rl_n > 0:
        print(f"MLB Price CLV Run Line: {price_clv_rl_n} bets  avg={avg_price_clv_rl:+.2f}%")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with psycopg2.connect(PG_DSN) as conn:
        n_games = update_game_outcomes(conn)
        log.info("Updated %d MLB game outcome rows", n_games)

        n_clv = backfill_clv(conn)
        if n_clv:
            log.info("Backfilled CLV for %d historical rows", n_clv)

        n_props = update_prop_outcomes(conn)
        log.info("Updated %d MLB prop outcome rows", n_props)

        n_ledger = grade_bankroll_ledger(conn)
        log.info("Graded %d MLB bankroll ledger rows", n_ledger)

        n_model_ledger = grade_model_pick_ledger(conn)
        log.info("Graded %d MLB model-pick ledger rows", n_model_ledger)

        print_running_record(conn)


if __name__ == "__main__":
    main()
