import logging
from pathlib import Path

import psycopg2

from mlb_pipeline.parse_meta import main as parse_meta
from mlb_pipeline.parse_games import main as parse_games, sync_scores_from_boxscores
from mlb_pipeline.parse_boxscore import main as parse_boxscore
from mlb_pipeline.parse_player_gamelogs import main as parse_player_gamelogs
from mlb_pipeline.parse_starting_pitchers import main as parse_starting_pitchers
from mlb_pipeline.parse_oddsapi import main as parse_game_odds

log = logging.getLogger("mlb_pipeline.parse_all")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_SQL_DIR = Path(__file__).resolve().parents[2] / "sql"

_MLB_SQL_VIEWS = [
    "MLB001_mlb_batting_rolling.sql",
    "MLB002_mlb_pitching_rolling.sql",
    "MLB003_mlb_pitcher_rolling.sql",
    "MLB004_mlb_ballpark_factors.sql",
    "MLB005_mlb_standings_rest.sql",
    "MLB006_mlb_game_features.sql",
    "MLB008_mlb_player_batting_rolling.sql",
]

_MLB_MATVIEW_REFRESH = [
    # MLB007: create + refresh rolling materialized views for fast prediction queries
    "MLB007_mlb_materialized_rolling.sql",
]

_MLB_MATVIEW_REFRESH_SQL = """
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_batting_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_pitching_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_pitcher_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_standings_rest_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_player_batting_rolling_mat;
"""


def _apply_sql_views(pg_dsn: str) -> None:
    """Apply MLB SQL views in order. Idempotent — safe to run on every parse_all.

    Each file is applied in a try/except so a failure in one view does not
    abort the remaining views.  Errors are logged and the function continues.
    All successful executions are committed together at the end.
    """
    conn = None
    try:
        conn = psycopg2.connect(pg_dsn)
    except Exception:
        log.exception("_apply_sql_views: failed to connect to database")
        return

    conn.autocommit = False
    cur = conn.cursor()

    for filename in _MLB_SQL_VIEWS:
        sql_path = _SQL_DIR / filename
        try:
            sql = sql_path.read_text(encoding="utf-8")
            cur.execute(sql)
            log.info("Applied %s", filename)
        except Exception:
            log.exception("Failed to apply %s — continuing with remaining views", filename)
            conn.rollback()
            # Re-acquire cursor after rollback so subsequent views can still run
            cur = conn.cursor()

    try:
        conn.commit()
        log.info("All MLB SQL views applied and committed")
    except Exception:
        log.exception("Failed to commit SQL views")
        conn.rollback()
    finally:
        conn.close()


def _refresh_matviews(pg_dsn: str) -> None:
    """Create (if needed) and refresh MLB rolling materialized views."""
    conn = None
    try:
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        cur = conn.cursor()

        # Apply MLB007 SQL (CREATE MATERIALIZED VIEW IF NOT EXISTS + indexes)
        for filename in _MLB_MATVIEW_REFRESH:
            sql_path = _SQL_DIR / filename
            try:
                sql = sql_path.read_text(encoding="utf-8")
                # Split into individual statements and run each
                for stmt in sql.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
                log.info("Applied %s", filename)
            except Exception:
                log.exception("Failed to apply %s", filename)

        # Refresh all mat views
        for stmt in _MLB_MATVIEW_REFRESH_SQL.strip().split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                cur.execute(stmt)
                log.info("Refreshed: %s", stmt.split()[-1])
            except Exception:
                log.exception("Failed to refresh: %s", stmt)
    except Exception:
        log.exception("_refresh_matviews: failed to connect")
    finally:
        if conn:
            conn.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    log.info("Starting FULL MLB parse pipeline")

    # Order matters — dimensions before facts
    parse_meta()              # venues, teams, standings, injuries
    parse_games()             # mlb_games
    parse_player_gamelogs()   # player stat backbone
    parse_boxscore()          # game + player boxscores
    parse_starting_pitchers() # confirmed/probable starting pitchers
    parse_game_odds()         # game lines from Odds API

    # Sync scores again: parse_boxscore may have added new rows that parse_games
    # (run before boxscore) didn't see yet.
    with psycopg2.connect(_PG_DSN) as _c:
        n = sync_scores_from_boxscores(_c)
        _c.commit()
        if n:
            log.info("Post-boxscore score sync: updated %d games", n)

    _apply_sql_views(_PG_DSN)

    # Create/refresh materialized rolling views for fast prediction queries
    _refresh_matviews(_PG_DSN)

    log.info("ALL MLB PARSERS COMPLETE")


if __name__ == "__main__":
    main()
