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

    # Each view is committed independently so a failure in one view does not
    # roll back previously-applied views (important: MLB006 depends on MLB003/005
    # columns that must already be committed before MLB006 can reference them).
    conn.autocommit = True
    cur = conn.cursor()

    for filename in _MLB_SQL_VIEWS:
        sql_path = _SQL_DIR / filename
        try:
            sql = sql_path.read_text(encoding="utf-8")
            cur.execute(sql)
            log.info("Applied %s", filename)
        except Exception:
            log.exception("Failed to apply %s — continuing with remaining views", filename)

    log.info("All MLB SQL views applied")
    conn.close()


_MATVIEW_TO_VIEW = {
    "mlb_team_batting_rolling_mat":  "mlb_team_batting_rolling",
    "mlb_team_pitching_rolling_mat": "mlb_team_pitching_rolling",
    "mlb_pitcher_rolling_mat":       "mlb_pitcher_rolling",
    "mlb_standings_rest_mat":        "mlb_standings_rest",
    "mlb_player_batting_rolling_mat": "mlb_player_batting_rolling",
}


def _matview_needs_recreate(cur, matview: str, base_view: str) -> bool:
    """Return True if the matview's column count differs from its underlying view.

    Used to detect schema changes (new columns added to the view) so the matview
    can be dropped and recreated rather than just refreshed.
    """
    try:
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'features' AND table_name = %s",
            (matview,),
        )
        mat_cols = (cur.fetchone() or (0,))[0]
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = 'features' AND table_name = %s",
            (base_view,),
        )
        view_cols = (cur.fetchone() or (0,))[0]
        return mat_cols == 0 or mat_cols != view_cols
    except Exception:
        return True  # err on the side of recreation


def _refresh_matviews(pg_dsn: str) -> None:
    """Create (if needed) and refresh MLB rolling materialized views.

    Auto-detects schema changes (column count mismatch) and drops/recreates
    affected matviews.  If any matview is dropped, MLB006 is re-applied
    afterwards so dependent views (mlb_game_training/prediction_features)
    are recreated with the updated schema.
    """
    conn = None
    try:
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        cur = conn.cursor()

        # ── Drop matviews whose column count is out of sync with the base view ──
        dropped_any = False
        for matview, base_view in _MATVIEW_TO_VIEW.items():
            if _matview_needs_recreate(cur, matview, base_view):
                try:
                    cur.execute(
                        f"DROP MATERIALIZED VIEW IF EXISTS features.{matview} CASCADE"
                    )
                    log.info("Dropped %s (schema change detected)", matview)
                    dropped_any = True
                except Exception:
                    log.exception("Failed to drop %s", matview)

        # ── Apply MLB007 SQL (CREATE MATERIALIZED VIEW IF NOT EXISTS + indexes) ──
        for filename in _MLB_MATVIEW_REFRESH:
            sql_path = _SQL_DIR / filename
            try:
                sql = sql_path.read_text(encoding="utf-8")
                for stmt in sql.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
                log.info("Applied %s", filename)
            except Exception:
                log.exception("Failed to apply %s", filename)

        # ── Refresh all mat views ────────────────────────────────────────────────
        for stmt in _MLB_MATVIEW_REFRESH_SQL.strip().split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                cur.execute(stmt)
                log.info("Refreshed: %s", stmt.split()[-1])
            except Exception:
                log.exception("Failed to refresh: %s", stmt)

        # ── If any matview was dropped (cascading to game feature views),
        #    re-apply MLB006 so those views are recreated with updated schemas ──
        if dropped_any:
            sql_path = _SQL_DIR / "MLB006_mlb_game_features.sql"
            conn.autocommit = False
            cur2 = conn.cursor()
            try:
                cur2.execute(sql_path.read_text(encoding="utf-8"))
                conn.commit()
                log.info("Re-applied MLB006 after matview schema change")
            except Exception:
                conn.rollback()
                log.exception("Failed to re-apply MLB006 after matview drop")
            finally:
                conn.autocommit = True

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
