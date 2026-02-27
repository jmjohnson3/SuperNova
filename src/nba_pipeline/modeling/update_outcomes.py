# src/nba_pipeline/modeling/update_outcomes.py
"""
Fill actual scores and ATS results for completed games in bets.game_predictions,
and actual stats for bets.prop_predictions.

Run after games complete:
    python -m nba_pipeline.modeling.update_outcomes
"""
import logging
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
                UNIQUE (game_date_et, game_slug)
            );
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
                   gp.spread_bet_side, gp.total_bet_side
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

    updated = 0
    with conn.cursor() as cur:
        for row in pending:
            slug = row["game_slug"]
            if slug not in finals:
                continue

            g = finals[slug]
            actual_margin = float(g["home_score"]) - float(g["away_score"])
            actual_total = float(g["home_score"]) + float(g["away_score"])

            # ATS: home covered if actual_margin > market_spread_home
            spread_covered = None
            if row["market_spread_home"] is not None and row["spread_bet_side"] is not None:
                mkt_s = float(row["market_spread_home"])
                covered = actual_margin > mkt_s
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

            cur.execute("""
                UPDATE bets.game_predictions
                SET actual_margin_home = %s,
                    actual_total       = %s,
                    spread_covered     = %s,
                    total_correct      = %s
                WHERE id = %s
            """, (actual_margin, actual_total, spread_covered, total_correct, row["id"]))
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
    """Print running ATS and total record from bets.game_predictions."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE spread_covered IS NOT NULL)       AS spread_n,
                SUM(CASE WHEN spread_covered = TRUE THEN 1 ELSE 0 END)   AS spread_wins,
                COUNT(*) FILTER (WHERE total_correct IS NOT NULL)        AS total_n,
                SUM(CASE WHEN total_correct = TRUE THEN 1 ELSE 0 END)    AS total_wins
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

    spread_pct = (spread_wins / spread_n * 100) if spread_n > 0 else 0.0
    total_pct = (total_wins / total_n * 100) if total_n > 0 else 0.0

    # ROI at -110: win $100 per win, lose $110 per loss
    spread_roi = ((spread_wins * 100 - (spread_n - spread_wins) * 110) / (spread_n * 110) * 100) if spread_n > 0 else 0.0
    total_roi = ((total_wins * 100 - (total_n - total_wins) * 110) / (total_n * 110) * 100) if total_n > 0 else 0.0

    print(
        f"Spread: {spread_wins}-{spread_n - spread_wins} ({spread_pct:.1f}%) | ROI: {spread_roi:+.1f}%"
        f" | Total: {total_wins}-{total_n - total_wins} ({total_pct:.1f}%) | ROI: {total_roi:+.1f}%"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    with psycopg2.connect(PG_DSN) as conn:
        _ensure_schema(conn)

        n_games = update_game_outcomes(conn)
        log.info("Updated %d game outcome rows", n_games)

        n_props = update_prop_outcomes(conn)
        log.info("Updated %d prop outcome rows", n_props)

        print_running_record(conn)


if __name__ == "__main__":
    main()
