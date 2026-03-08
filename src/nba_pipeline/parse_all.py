import logging
from pathlib import Path

import psycopg2

from nba_pipeline.parse_games import main as parse_games
from nba_pipeline.parse_meta import main as parse_meta
from nba_pipeline.parse_player_gamelogs import main as parse_player_gamelogs
from nba_pipeline.parse_lineup import main as parse_lineup
from nba_pipeline.parse_boxscore import main as parse_boxscore
from nba_pipeline.parse_pbp import main as parse_pbp
from nba_pipeline.parse_referees import main as parse_referees
from nba_pipeline.parse_oddsapi import parse_prop_odds, main as parse_game_odds, parse_game_odds_historical

log = logging.getLogger("nba_pipeline.parse_all")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

# ---------------------------------------------------------------------------
# Materialized-view helpers
# ---------------------------------------------------------------------------
# game_training_features and game_prediction_features are complex views with
# many window-function CTEs that take ~80-90 s each on a full table scan.
# We convert them to materialized views (refreshed here after each parse run)
# so that prediction queries run in milliseconds instead of minutes.
#
# On FIRST run: creates the matviews, rebuilds wrapper views, takes ~3 min.
# On subsequent runs: just refreshes the matviews, still ~3 min.
# Predictions then complete in < 5 s.
# ---------------------------------------------------------------------------

def _apply_view_fixes(pg_dsn: str) -> None:
    """Apply SQL view bug fixes. Idempotent — safe to run on every parse_all.

    Fixes applied:
    - V011: three_pt_rate_avg_10 was always 0 (wrong event_type filter;
      now uses raw_json->'fieldGoalAttempt'->>'points' = 3)
    - V013: referee fan-out producing 26% duplicate player-game rows
      (team_abbr removed from player_referee_foul_history GROUP BY).
      Requires DROP CASCADE because column list shrinks.
    - V016: opp_pts_allowed_role_10 up to 140 in season-opening games
      (NULL when window has < 3 games of data)

    V013 drops player_referee_foul_history CASCADE which cascades to
    player_game_referee_foul_risk and player_training_features.
    V022 is re-applied afterwards to recreate player_training_features
    with all its columns (including the prop line book-line prior).
    """
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        # V011: CREATE OR REPLACE is fine (same column list, different filter)
        cur.execute((_SQL_DIR / "V011_pbp_features.sql").read_text(encoding="utf-8"))
        log.info("Applied V011 PBP 3PT rate fix")

        # V016: CREATE OR REPLACE is fine (output column names unchanged)
        cur.execute((_SQL_DIR / "V016_opponent_position_defense.sql").read_text(encoding="utf-8"))
        log.info("Applied V016 opp_position_defense sparse-sample fix")

        # V013: column list shrinks (team_abbr removed) — must DROP CASCADE first.
        # This also drops player_game_referee_foul_risk and player_training_features.
        cur.execute("DROP VIEW IF EXISTS features.player_referee_foul_history CASCADE")
        cur.execute((_SQL_DIR / "V013_referee_features.sql").read_text(encoding="utf-8"))
        log.info("Applied V013 referee fan-out fix")

        # V022: recreate player_training_features (includes DROP VIEW IF EXISTS CASCADE)
        cur.execute((_SQL_DIR / "V022_prop_line_features.sql").read_text(encoding="utf-8"))
        log.info("Recreated player_training_features from V022")

        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("Failed to apply view fixes — continuing with existing views")
    finally:
        conn.close()


_GAME_VIEWS = [
    (
        "game_training_features",
        "game_training_features_mat",
        [
            "CREATE INDEX IF NOT EXISTS idx_gtf_mat_season_slug"
            "    ON features.game_training_features_mat (season, game_slug)",
            "CREATE INDEX IF NOT EXISTS idx_gtf_mat_date"
            "    ON features.game_training_features_mat (game_date_et)",
        ],
    ),
    (
        "game_prediction_features",
        "game_prediction_features_mat",
        [
            "CREATE INDEX IF NOT EXISTS idx_gpf_mat_date"
            "    ON features.game_prediction_features_mat (game_date_et)",
            "CREATE INDEX IF NOT EXISTS idx_gpf_mat_season_slug"
            "    ON features.game_prediction_features_mat (season, game_slug)",
        ],
    ),
]


def _materialize_game_features(pg_dsn: str) -> None:
    """Ensure matviews exist, then refresh them.

    First call (~3 min):
      1. pg_get_viewdef → complex SQL
      2. CREATE MATERIALIZED VIEW <name>_mat AS <complex SQL> WITH NO DATA
      3. Add indexes
      4. DROP VIEW <name> CASCADE  (also drops player_training_features if downstream)
      5. CREATE VIEW <name> AS SELECT * FROM <name>_mat   (thin wrapper)
      6. Recreate any dropped downstream views
      7. REFRESH MATERIALIZED VIEW

    Subsequent calls (~90 s each):
      - REFRESH MATERIALIZED VIEW only (step 7)
    """
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        for view_name, mat_name, index_sqls in _GAME_VIEWS:
            # ------------------------------------------------------------------
            # 1. Check if matview already exists
            # ------------------------------------------------------------------
            cur.execute(
                """
                SELECT relkind FROM pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'features' AND c.relname = %s
                """,
                (mat_name,),
            )
            row = cur.fetchone()

            if row is None:
                # ------------------------------------------------------------------
                # 2. First run: create the matview from the existing view's SQL
                # ------------------------------------------------------------------
                log.info("Creating materialized view features.%s (one-time, ~90 s) ...", mat_name)

                cur.execute("SELECT pg_get_viewdef(%s, true)", (f"features.{view_name}",))
                # pg_get_viewdef may include a trailing semicolon — strip it so the
                # CREATE MATERIALIZED VIEW ... AS <sql> WITH NO DATA syntax is valid.
                view_sql = cur.fetchone()[0].rstrip().rstrip(";")

                # Save player_training_features SQL before CASCADE drop (if it depends on this view)
                ptf_sql = None
                if view_name == "game_training_features":
                    cur.execute(
                        """
                        SELECT pg_get_viewdef('features.player_training_features', true)
                        WHERE EXISTS (
                            SELECT 1 FROM pg_class c
                            JOIN pg_namespace n ON c.relnamespace = n.oid
                            WHERE n.nspname = 'features' AND c.relname = 'player_training_features'
                        )
                        """
                    )
                    r2 = cur.fetchone()
                    if r2:
                        ptf_sql = r2[0].rstrip().rstrip(";")

                cur.execute(
                    f"CREATE MATERIALIZED VIEW features.{mat_name} AS {view_sql} WITH NO DATA"
                )
                for idx_sql in index_sqls:
                    cur.execute(idx_sql)

                cur.execute(f"DROP VIEW IF EXISTS features.{view_name} CASCADE")
                cur.execute(
                    f"CREATE VIEW features.{view_name} AS SELECT * FROM features.{mat_name}"
                )

                if ptf_sql:
                    cur.execute(
                        f"CREATE VIEW features.player_training_features AS {ptf_sql}"
                    )

                conn.commit()
                log.info("  Wrapper views rebuilt for features.%s", view_name)

            # ------------------------------------------------------------------
            # 3. Refresh (always — picks up new data from this parse run)
            # ------------------------------------------------------------------
            log.info("Refreshing features.%s ...", mat_name)
            cur.execute(f"REFRESH MATERIALIZED VIEW features.{mat_name}")
            conn.commit()
            log.info("  Done refreshing features.%s", mat_name)

    except Exception:
        conn.rollback()
        log.exception(
            "Failed to materialize/refresh game features — predictions will fall back to slow views"
        )
    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    log.info("Starting FULL parse pipeline")

    # Order matters a little (dimensions first)
    parse_meta()              # venues, teams, standings, injuries
    parse_games()             # nba_games
    parse_player_gamelogs()   # training backbone
    parse_lineup()            # availability / starters
    parse_boxscore()          # game + player boxscores
    parse_pbp()               # advanced features
    parse_referees()               # referee assignments from boxscore payloads
    parse_game_odds()              # live game lines (nba_odds)
    parse_game_odds_historical()   # backfilled historical game lines (nba_odds_historical)
    parse_prop_odds()              # prop lines from odds API

    _apply_view_fixes(_PG_DSN)
    _materialize_game_features(_PG_DSN)

    log.info("ALL PARSERS COMPLETE")

if __name__ == "__main__":
    main()
