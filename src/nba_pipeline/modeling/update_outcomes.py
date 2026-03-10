# src/nba_pipeline/modeling/update_outcomes.py
"""
Fill actual scores and ATS results for completed games in bets.game_predictions,
and actual stats for bets.prop_predictions.

Run after games complete:
    python -m nba_pipeline.modeling.update_outcomes
"""
import logging
import math
from datetime import datetime, date
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

log = logging.getLogger("nba_pipeline.modeling.update_outcomes")

_ET = ZoneInfo("America/New_York")
PG_DSN = "postgresql://josh:password@localhost:5432/nba"


def _ensure_schema(conn) -> None:
    """Create bets schema and tables if they don't exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE SCHEMA IF NOT EXISTS bets;
            CREATE TABLE IF NOT EXISTS bets.game_predictions (
                id                  SERIAL PRIMARY KEY,
                predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                game_date_et        DATE        NOT NULL,
                game_slug           TEXT        NOT NULL,
                season              TEXT        NOT NULL,
                home_team_abbr      TEXT        NOT NULL,
                away_team_abbr      TEXT        NOT NULL,
                pred_margin_home    NUMERIC,
                pred_total          NUMERIC,
                used_residual_model BOOLEAN     DEFAULT FALSE,
                market_spread_home  NUMERIC,
                market_total        NUMERIC,
                edge_spread         NUMERIC,
                edge_total          NUMERIC,
                actual_margin_home  NUMERIC,
                actual_total        NUMERIC,
                spread_bet_side     TEXT,
                total_bet_side      TEXT,
                spread_covered      BOOLEAN,
                total_correct       BOOLEAN,
                direction_correct   BOOLEAN,
                UNIQUE (game_date_et, game_slug)
            );
            -- Add columns for older tables that predate this schema version
            ALTER TABLE bets.game_predictions
                ADD COLUMN IF NOT EXISTS direction_correct     BOOLEAN,
                ADD COLUMN IF NOT EXISTS kelly_fraction_spread NUMERIC,
                ADD COLUMN IF NOT EXISTS kelly_fraction_total  NUMERIC,
                ADD COLUMN IF NOT EXISTS win_prob_spread       NUMERIC,
                ADD COLUMN IF NOT EXISTS win_prob_total        NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_spread_home   NUMERIC,
                ADD COLUMN IF NOT EXISTS closing_total         NUMERIC,
                ADD COLUMN IF NOT EXISTS clv_spread            NUMERIC,
                ADD COLUMN IF NOT EXISTS clv_total             NUMERIC;
            CREATE TABLE IF NOT EXISTS bets.prop_predictions (
                id                  SERIAL PRIMARY KEY,
                predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                game_date_et        DATE        NOT NULL,
                game_slug           TEXT        NOT NULL,
                player_id           BIGINT      NOT NULL,
                player_name         TEXT,
                team_abbr           TEXT,
                pred_points         NUMERIC,
                pred_rebounds       NUMERIC,
                pred_assists        NUMERIC,
                actual_points       NUMERIC,
                actual_rebounds     NUMERIC,
                actual_assists      NUMERIC,
                UNIQUE (game_date_et, game_slug, player_id)
            );
        """)
    conn.commit()


def _kelly(
    edge_pts: float,
    juice: int = -110,
    shrink: float = 0.60,
    sigma: float = 14.0,
) -> tuple[float, float]:
    """Estimate full Kelly fraction and win probability for a spread/total bet.
    Mirrors the same function in predict_today.py.
    """
    b = 100 / abs(juice)
    p_raw = 1 / (1 + math.exp(-edge_pts / sigma))
    p = 0.5 + (p_raw - 0.5) * shrink
    kelly = max(0.0, (b * p - (1 - p)) / b)
    return kelly, p


def _backfill_kelly(conn) -> int:
    """Backfill kelly_fraction and win_prob for historical rows that have edges but NULL kelly.

    Uses default sigmas: spread=14.0, total=20.0.
    Returns number of rows updated.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, edge_spread, edge_total
            FROM bets.game_predictions
            WHERE (edge_spread IS NOT NULL OR edge_total IS NOT NULL)
              AND (kelly_fraction_spread IS NULL AND kelly_fraction_total IS NULL)
        """)
        rows = cur.fetchall()

    if not rows:
        log.info("No rows to backfill for Kelly.")
        return 0

    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            kf_s = kf_t = wp_s = wp_t = None
            if row["edge_spread"] is not None:
                kf_s, wp_s = _kelly(abs(float(row["edge_spread"])), sigma=14.0)
                kf_s = round(kf_s, 4)
                wp_s = round(wp_s, 4)
            if row["edge_total"] is not None:
                kf_t, wp_t = _kelly(abs(float(row["edge_total"])), sigma=20.0)
                kf_t = round(kf_t, 4)
                wp_t = round(wp_t, 4)
            cur.execute("""
                UPDATE bets.game_predictions
                SET kelly_fraction_spread = %s,
                    kelly_fraction_total  = %s,
                    win_prob_spread       = %s,
                    win_prob_total        = %s
                WHERE id = %s
            """, (kf_s, kf_t, wp_s, wp_t, row["id"]))
            updated += 1

    conn.commit()
    log.info("Backfilled Kelly/win_prob for %d rows", updated)
    return updated


def update_game_outcomes(conn) -> int:
    """
    For each row in bets.game_predictions where actual_margin_home IS NULL
    and the game is final, fill actuals and compute ATS result.
    Returns number of rows updated.
    """
    today = datetime.now(_ET).date()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT gp.id, gp.game_slug, gp.game_date_et,
                   gp.market_spread_home, gp.market_total,
                   gp.spread_bet_side, gp.total_bet_side,
                   gp.pred_margin_home,
                   gp.home_team_abbr, gp.away_team_abbr
            FROM bets.game_predictions gp
            WHERE gp.actual_margin_home IS NULL
              AND gp.game_date_et <= %s
        """, (today,))
        pending = cur.fetchall()

    if not pending:
        log.info("No pending game predictions to update.")
        return 0

    slugs = [r["game_slug"] for r in pending]
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_slug, home_score, away_score
            FROM raw.nba_games
            WHERE game_slug = ANY(%s)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
        """, (slugs,))
        finals = {r["game_slug"]: r for r in cur.fetchall()}

    # Fetch closing lines for CLV computation.
    # Match on (home_team_abbr, away_team_abbr, as_of_date) using game_date_et.
    # Take the most favorable close per game (best bookmaker closing line).
    dates = list({r["game_date_et"] for r in pending})  # keep as date objects
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (home_team_abbr, away_team_abbr, as_of_date)
                    home_team_abbr, away_team_abbr, as_of_date,
                    close_spread_home_points, close_total
                FROM odds.nba_game_lines_open_close
                WHERE as_of_date = ANY(%s::date[])
                ORDER BY home_team_abbr, away_team_abbr, as_of_date, close_fetched_at_utc DESC
            """, (dates,))
            closing_rows = cur.fetchall()
        # Key: (home_team_abbr, away_team_abbr, game_date_et as date obj)
        closing_lines = {
            (r["home_team_abbr"], r["away_team_abbr"], r["as_of_date"]): r
            for r in closing_rows
        }
    except Exception as exc:
        log.warning("Could not fetch closing lines for CLV: %s", exc)
        closing_lines = {}

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            slug = row["game_slug"]
            if slug not in finals:
                continue

            g = finals[slug]
            actual_margin = float(g["home_score"]) - float(g["away_score"])
            actual_total = float(g["home_score"]) + float(g["away_score"])

            # ATS: home covered if actual_margin > -market_spread_home
            # market_spread_home follows Odds API sign convention:
            #   negative = home is FAVORITE (gives pts, e.g. -8.5)
            #   positive = home is UNDERDOG (gets pts, e.g. +7.5)
            # Home covers when: actual_margin + market_spread_home > 0
            #                 = actual_margin > -market_spread_home
            spread_covered = None
            if row["market_spread_home"] is not None and row["spread_bet_side"] is not None:
                mkt_s = float(row["market_spread_home"])
                covered = actual_margin > -mkt_s
                if row["spread_bet_side"] == "home":
                    spread_covered = covered
                else:
                    spread_covered = not covered

            # Total: correct if prediction direction matches actual
            total_correct = None
            if row["market_total"] is not None and row["total_bet_side"] is not None:
                mkt_t = float(row["market_total"])
                if row["total_bet_side"] == "over":
                    total_correct = actual_total > mkt_t
                else:
                    total_correct = actual_total < mkt_t

            # Directional accuracy: did the model correctly predict home win vs. away win?
            pred_margin = float(row["pred_margin_home"]) if row["pred_margin_home"] is not None else None
            direction_correct = None
            if pred_margin is not None:
                direction_correct = (pred_margin > 0) == (actual_margin > 0)

            # CLV: Closing Line Value — positive = we beat the closing line (sharp signal).
            # home bet: CLV = market_spread_home - closing_spread_home
            # away bet: CLV = closing_spread_home - market_spread_home
            # over bet: CLV = closing_total - market_total
            # under bet: CLV = market_total - closing_total
            closing_spread = None
            closing_total = None
            clv_spread = None
            clv_total = None
            cl_key = (row["home_team_abbr"], row["away_team_abbr"], row["game_date_et"])
            cl = closing_lines.get(cl_key)
            if cl:
                if cl["close_spread_home_points"] is not None:
                    closing_spread = float(cl["close_spread_home_points"])
                if cl["close_total"] is not None:
                    closing_total = float(cl["close_total"])

            mkt_s = float(row["market_spread_home"]) if row["market_spread_home"] is not None else None
            mkt_t = float(row["market_total"]) if row["market_total"] is not None else None

            if closing_spread is not None and mkt_s is not None and row["spread_bet_side"] is not None:
                if row["spread_bet_side"] == "home":
                    clv_spread = round(mkt_s - closing_spread, 2)
                else:
                    clv_spread = round(closing_spread - mkt_s, 2)

            if closing_total is not None and mkt_t is not None and row["total_bet_side"] is not None:
                if row["total_bet_side"] == "over":
                    clv_total = round(closing_total - mkt_t, 2)
                else:
                    clv_total = round(mkt_t - closing_total, 2)

            cur.execute("""
                UPDATE bets.game_predictions
                SET actual_margin_home = %s,
                    actual_total       = %s,
                    spread_covered     = %s,
                    total_correct      = %s,
                    direction_correct  = %s,
                    closing_spread_home = %s,
                    closing_total       = %s,
                    clv_spread          = %s,
                    clv_total           = %s
                WHERE id = %s
            """, (actual_margin, actual_total, spread_covered, total_correct, direction_correct,
                  closing_spread, closing_total, clv_spread, clv_total, row["id"]))
            updated += 1

    conn.commit()
    return updated


def update_prop_outcomes(conn) -> int:
    """
    For each row in bets.prop_predictions where actual_points IS NULL
    and the game is done, fill actual stats from raw.nba_player_gamelogs.
    Returns number of rows updated.
    """
    today = datetime.now(_ET).date()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT pp.id, pp.game_slug, pp.player_id
            FROM bets.prop_predictions pp
            WHERE pp.actual_points IS NULL
              AND pp.game_date_et <= %s
        """, (today,))
        pending = cur.fetchall()

    if not pending:
        log.info("No pending prop predictions to update.")
        return 0

    slugs = list({r["game_slug"] for r in pending})
    player_ids = list({r["player_id"] for r in pending})

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT gl.game_slug, gl.player_id, gl.points, gl.rebounds, gl.assists
            FROM raw.nba_player_gamelogs gl
            JOIN raw.nba_games g
              ON g.game_slug = gl.game_slug
            WHERE gl.game_slug = ANY(%s)
              AND gl.player_id = ANY(%s)
              AND g.status = 'final'
        """, (slugs, player_ids))
        actuals = {(r["game_slug"], int(r["player_id"])): r for r in cur.fetchall()}

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            key = (row["game_slug"], int(row["player_id"]))
            if key not in actuals:
                continue
            a = actuals[key]
            cur.execute("""
                UPDATE bets.prop_predictions
                SET actual_points   = %s,
                    actual_rebounds = %s,
                    actual_assists  = %s
                WHERE id = %s
            """, (a["points"], a["rebounds"], a["assists"], row["id"]))
            updated += 1

    conn.commit()
    return updated


def print_running_record(conn) -> None:
    """Print running ATS and total record from bets.game_predictions, including CLV stats."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE spread_covered IS NOT NULL)        AS spread_n,
                SUM(CASE WHEN spread_covered = TRUE THEN 1 ELSE 0 END)    AS spread_wins,
                COUNT(*) FILTER (WHERE total_correct IS NOT NULL)         AS total_n,
                SUM(CASE WHEN total_correct = TRUE THEN 1 ELSE 0 END)     AS total_wins,
                COUNT(*) FILTER (WHERE direction_correct IS NOT NULL)     AS dir_n,
                SUM(CASE WHEN direction_correct = TRUE THEN 1 ELSE 0 END) AS dir_wins,
                COUNT(*) FILTER (WHERE clv_spread IS NOT NULL)            AS clv_spread_n,
                SUM(CASE WHEN clv_spread > 0 THEN 1 ELSE 0 END)
                    FILTER (WHERE clv_spread IS NOT NULL)                  AS clv_spread_wins,
                AVG(clv_spread) FILTER (WHERE clv_spread IS NOT NULL)     AS avg_clv_spread,
                COUNT(*) FILTER (WHERE clv_total IS NOT NULL)             AS clv_total_n,
                AVG(clv_total) FILTER (WHERE clv_total IS NOT NULL)       AS avg_clv_total
            FROM bets.game_predictions
            WHERE spread_bet_side IS NOT NULL OR total_bet_side IS NOT NULL
        """)
        row = cur.fetchone()

    if not row:
        return

    spread_n = int(row["spread_n"] or 0)
    spread_wins = int(row["spread_wins"] or 0)
    total_n = int(row["total_n"] or 0)
    total_wins = int(row["total_wins"] or 0)
    dir_n = int(row["dir_n"] or 0)
    dir_wins = int(row["dir_wins"] or 0)
    clv_spread_n = int(row["clv_spread_n"] or 0)
    clv_spread_wins = int(row["clv_spread_wins"] or 0)
    avg_clv_spread = float(row["avg_clv_spread"] or 0.0)
    clv_total_n = int(row["clv_total_n"] or 0)
    avg_clv_total = float(row["avg_clv_total"] or 0.0)

    spread_pct = (spread_wins / spread_n * 100) if spread_n > 0 else 0.0
    total_pct = (total_wins / total_n * 100) if total_n > 0 else 0.0
    dir_pct = (dir_wins / dir_n * 100) if dir_n > 0 else 0.0
    clv_spread_pct = (clv_spread_wins / clv_spread_n * 100) if clv_spread_n > 0 else 0.0

    # ROI at -110: win $100 per win, lose $110 per loss
    spread_roi = ((spread_wins * 100 - (spread_n - spread_wins) * 110) / (spread_n * 110) * 100) if spread_n > 0 else 0.0
    total_roi = ((total_wins * 100 - (total_n - total_wins) * 110) / (total_n * 110) * 100) if total_n > 0 else 0.0

    print(
        f"Spread: {spread_wins}-{spread_n - spread_wins} ({spread_pct:.1f}%) | ROI: {spread_roi:+.1f}%"
        f" | Total: {total_wins}-{total_n - total_wins} ({total_pct:.1f}%) | ROI: {total_roi:+.1f}%"
        f" | Direction: {dir_wins}/{dir_n} ({dir_pct:.1f}%)"
    )
    if clv_spread_n > 0:
        print(
            f"CLV Spread: beat close {clv_spread_wins}/{clv_spread_n} ({clv_spread_pct:.0f}%)"
            f" | avg CLV={avg_clv_spread:+.2f} pts"
            + (f" | CLV Total avg={avg_clv_total:+.2f} pts" if clv_total_n > 0 else "")
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with psycopg2.connect(PG_DSN) as conn:
        _ensure_schema(conn)

        n_kelly = _backfill_kelly(conn)
        if n_kelly:
            log.info("Backfilled Kelly for %d historical rows", n_kelly)

        n_games = update_game_outcomes(conn)
        log.info("Updated %d game outcome rows", n_games)

        n_props = update_prop_outcomes(conn)
        log.info("Updated %d prop outcome rows", n_props)

        print_running_record(conn)


if __name__ == "__main__":
    main()
