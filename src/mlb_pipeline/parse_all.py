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
    "MLB019_mlb_reliever_rolling.sql",
    # MLB006 depends on rolling matviews (e.g. mlb_pitcher_rolling_mat).
    # Apply it after _refresh_matviews() via _MLB_POST_MATVIEW_VIEWS.
    "MLB008_mlb_player_batting_rolling.sql",
    "MLB009_mlb_umpire_rolling.sql",   # DDL + umpire rolling view
    "MLB010_mlb_weather_ddl.sql",      # weather table DDL
    "MLB012_mlb_batting_vs_hand.sql",  # handedness DDL + base view
    "MLB013_mlb_batting_cross_season_rolling.sql",  # cross-season rolling (no season partition)
    "MLB014_mlb_player_prev_season_stats.sql",       # full prior-season aggregate stats
    "MLB015_mlb_batter_vs_sp.sql",                   # batter vs specific SP career H2H base view
    "MLB016_mlb_elo_features.sql",                    # Elo rating features (depends on raw.mlb_elo)
    "MLB017_mlb_sp_venue_stats.sql",                  # SP career stats per venue (Group F)
    "MLB018_mlb_team_batting_vs_hand.sql",            # team batting vs LHP/RHP (Group F)
]

# Applied AFTER _refresh_matviews() — MLB011 depends on mlb_player_batting_rolling_mat.
# MLB006 is re-applied here so lineup quality columns take effect.
_MLB_POST_MATVIEW_VIEWS = [
    "MLB011_mlb_lineup_quality.sql",
    "MLB006_mlb_game_features.sql",
]

_MLB_MATVIEW_REFRESH = [
    # MLB007: create + refresh rolling materialized views for fast prediction queries
    "MLB007_mlb_materialized_rolling.sql",
    # MLB016: career batter vs SP H2H aggregates per game
    "MLB016_mlb_game_h2h_batting.sql",
]

_MLB_MATVIEW_REFRESH_SQL = """
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_batting_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_pitching_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_pitcher_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_standings_rest_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_player_batting_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_umpire_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_batting_vs_hand_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_player_batting_rolling_cross_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_player_prev_season_stats_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_batter_vs_sp_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_venue_stats_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_batting_vs_hand_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_reliever_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_game_h2h_batting_mat;
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

    failed: list[str] = []
    for filename in _MLB_SQL_VIEWS:
        sql_path = _SQL_DIR / filename
        try:
            sql = sql_path.read_text(encoding="utf-8")
            cur.execute(sql)
            log.info("Applied %s", filename)
        except Exception:
            log.exception("Failed to apply %s — continuing with remaining views", filename)
            failed.append(filename)

    if failed:
        raise RuntimeError(
            "One or more MLB SQL view files failed to apply: "
            + ", ".join(failed)
        )

    log.info("All MLB SQL views applied")
    conn.close()


_MATVIEW_TO_VIEW = {
    "mlb_team_batting_rolling_mat":       "mlb_team_batting_rolling",
    "mlb_team_pitching_rolling_mat":      "mlb_team_pitching_rolling",
    "mlb_pitcher_rolling_mat":            "mlb_pitcher_rolling",
    "mlb_standings_rest_mat":             "mlb_standings_rest",
    "mlb_player_batting_rolling_mat":     "mlb_player_batting_rolling",
    "mlb_umpire_rolling_mat":             "mlb_umpire_rolling",
    "mlb_batting_vs_hand_mat":            "mlb_batting_vs_hand",
    "mlb_player_batting_rolling_cross_mat": "mlb_player_batting_rolling_cross",
    "mlb_player_prev_season_stats_mat":   "mlb_player_prev_season_stats",
    "mlb_batter_vs_sp_mat":               "mlb_batter_vs_sp",
    "mlb_sp_venue_stats_mat":             "mlb_sp_venue_stats",        # Group F
    "mlb_team_batting_vs_hand_mat":       "mlb_team_batting_vs_hand",  # Group F
    "mlb_reliever_rolling_mat":           "mlb_reliever_rolling",      # Group G
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

        failed_ops: list[str] = []

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
                    failed_ops.append(f"drop:{matview}")

        # ── Apply MLB007 SQL (CREATE MATERIALIZED VIEW IF NOT EXISTS + indexes) ──
        # Use per-statement try/except so a single failed CREATE (e.g. because a
        # cascade-dropped base view hasn't been re-applied yet) doesn't abort all
        # subsequent matview creates in the same file.
        def _apply_matview_sql(filename: str) -> None:
            sql_path = _SQL_DIR / filename
            sql = sql_path.read_text(encoding="utf-8")
            ok = skipped = 0
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    cur.execute(stmt)
                    ok += 1
                except Exception as exc:
                    log.warning("Statement in %s failed (skipping): %s", filename, exc)
                    skipped += 1
                    failed_ops.append(f"{filename}:{stmt[:80]}")
            log.info("Applied %s (%d ok, %d skipped)", filename, ok, skipped)

        for filename in _MLB_MATVIEW_REFRESH:
            _apply_matview_sql(filename)

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
                failed_ops.append(f"refresh:{stmt}")

        # ── If any matview was dropped (cascading to game feature views),
        #    re-apply MLB011 (lineup quality) then MLB006 so those views are
        #    recreated with updated schemas. MLB006 references mlb_lineup_quality
        #    so MLB011 must come first.
        #    Re-run MLB007 first to ensure any matviews whose base views were
        #    cascade-dropped-and-since-restored are now created. ──
        if dropped_any:
            # Re-apply MLB007 to pick up any matviews that failed on the first pass
            # (their base views may now exist after the refresh cycle above).
            _apply_matview_sql("MLB007_mlb_materialized_rolling.sql")

            conn.autocommit = False
            cur2 = conn.cursor()
            try:
                for fname in ("MLB011_mlb_lineup_quality.sql", "MLB006_mlb_game_features.sql"):
                    cur2.execute((_SQL_DIR / fname).read_text(encoding="utf-8"))
                conn.commit()
                log.info("Re-applied MLB011 + MLB006 after matview schema change")
            except Exception:
                conn.rollback()
                log.exception("Failed to re-apply MLB011/MLB006 after matview drop")
                failed_ops.append("reapply:MLB011+MLB006")
            finally:
                conn.autocommit = True

        if failed_ops:
            raise RuntimeError(
                "One or more matview operations failed: "
                + " | ".join(failed_ops[:20])
                + (" | ..." if len(failed_ops) > 20 else "")
            )

    except Exception:
        log.exception("_refresh_matviews: failed to connect")
    finally:
        if conn:
            conn.close()


def _apply_post_matview_views(pg_dsn: str) -> None:
    """Apply SQL files that depend on matviews (MLB011 lineup quality + MLB006 re-apply).

    Called after _refresh_matviews() so that mlb_player_batting_rolling_mat and
    mlb_umpire_rolling_mat already exist when these views are created.
    """
    conn = None
    try:
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        cur = conn.cursor()
        failed: list[str] = []
        for filename in _MLB_POST_MATVIEW_VIEWS:
            sql_path = _SQL_DIR / filename
            try:
                sql = sql_path.read_text(encoding="utf-8")
                cur.execute(sql)
                log.info("Applied (post-matview) %s", filename)
            except Exception:
                log.exception("Failed to apply (post-matview) %s — continuing", filename)
                failed.append(filename)
        if failed:
            raise RuntimeError(
                "Post-matview SQL apply failed for: " + ", ".join(failed)
            )
    except Exception:
        log.exception("_apply_post_matview_views: failed to connect")
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
    parse_boxscore()          # game + player boxscores (also extracts umpires daily)
    parse_starting_pitchers() # confirmed/probable starting pitchers
    parse_game_odds()         # game lines from Odds API

    # Sync scores again: parse_boxscore may have added new rows that parse_games
    # (run before boxscore) didn't see yet.
    with psycopg2.connect(_PG_DSN) as _c:
        n = sync_scores_from_boxscores(_c)
        _c.commit()
        if n:
            log.info("Post-boxscore score sync: updated %d games", n)

    _apply_sql_views(_PG_DSN)   # applies MLB001-010 (creates raw.mlb_weather table)

    # Fetch weather for all games without data (archive for historical, forecast for today)
    # Must run AFTER _apply_sql_views so raw.mlb_weather table exists.
    try:
        from mlb_pipeline.crawler_weather import fetch_all_missing_weather, refresh_todays_weather
        with psycopg2.connect(_PG_DSN) as _c:
            n_wx = fetch_all_missing_weather(_c)
            _c.commit()
            if n_wx:
                log.info("Fetched weather for %d games", n_wx)
        # Always refresh today's forecast (most recent Open-Meteo forecast before predictions run)
        with psycopg2.connect(_PG_DSN) as _c:
            n_refresh = refresh_todays_weather(_c)
            _c.commit()
            if n_refresh:
                log.info("Refreshed today's weather forecast for %d games", n_refresh)
    except Exception:
        log.exception("Weather crawl failed — continuing without weather data")

    # Create/refresh materialized rolling views for fast prediction queries
    _refresh_matviews(_PG_DSN)

    # Apply views that depend on matviews (lineup quality + MLB006 re-apply)
    _apply_post_matview_views(_PG_DSN)

    log.info("ALL MLB PARSERS COMPLETE")


if __name__ == "__main__":
    main()
