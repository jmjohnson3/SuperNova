import logging
from pathlib import Path

import psycopg2
from concurrent.futures import ThreadPoolExecutor

from mlb_pipeline.parse_meta import main as parse_meta
from mlb_pipeline.parse_games import main as parse_games, sync_scores_from_boxscores
from mlb_pipeline.parse_boxscore import main as parse_boxscore
from mlb_pipeline.parse_player_gamelogs import main as parse_player_gamelogs
from mlb_pipeline.parse_starting_pitchers import main as parse_starting_pitchers
from mlb_pipeline.parse_lineup import main as parse_lineup
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
    "MLB021_mlb_sp_velocity_rolling.sql",             # SP per-start velocity rolling view (Group N)
    "MLB022_mlb_sp_hand_k_pct.sql",                   # SP K% vs LHB/RHB (Retrain 2 #7)
    "MLB023_mlb_batter_venue_stats.sql",              # Batter career stats at specific venue (Retrain 2 #9)
    "MLB024_mlb_sp_vs_team.sql",                       # SP career stats vs specific opposing team
    "MLB025_mlb_batter_umpire.sql",                    # Batter career stats with specific home plate umpire
    "MLB026_mlb_batter_vs_rp.sql",                     # Batter rolling stats in bullpen games (opp SP < 5 IP)
    "MLB027_mlb_sp_lob_rate.sql",                      # SP LOB% / strand rate (career + rolling 10 starts)
    "MLB028_mlb_park_babip_factor.sql",                # Park-specific BABIP factor from actual gamelogs
    "MLB029_mlb_team_offensive_momentum.sql",          # Team runs scored last 3/5 games (offensive momentum)
    "MLB030_mlb_batter_babip_rolling.sql",             # Batter rolling BABIP (luck/regression signal for hits)
    "MLB031_mlb_sp_babip_rolling.sql",                 # SP rolling BABIP-against (luck/regression signal for Ks)
    "MLB032_mlb_team_der_rolling.sql",                 # Team Defensive Efficiency Rating (DER) rolling 20 games
    "MLB033_mlb_sp_hand_hr_rate.sql",                  # SP HR rate vs LHB/RHB (Feature 13)
]

# Applied AFTER _refresh_matviews() — MLB011 depends on mlb_player_batting_rolling_mat.
# MLB006 is re-applied here so lineup quality + catcher framing columns take effect.
_MLB_POST_MATVIEW_VIEWS = [
    "MLB011_mlb_lineup_quality.sql",
    "MLB020_mlb_team_catcher_framing.sql",  # catcher framing view (plain VIEW, cheap join)
    "MLB006_mlb_game_features.sql",
]

_MLB_MATVIEW_REFRESH = [
    # MLB007: create + refresh rolling materialized views for fast prediction queries
    "MLB007_mlb_materialized_rolling.sql",
    # MLB016: career batter vs SP H2H aggregates per game
    "MLB016_mlb_game_h2h_batting.sql",
    # MLB021b: matview for SP velocity rolling (fast training/prediction joins)
    "MLB021b_mlb_sp_velocity_rolling_mat.sql",
    # MLB022b/023b: matviews for SP K% by hand and batter venue stats (fast training joins)
    "MLB022b_mlb_sp_hand_k_pct_mat.sql",
    "MLB023b_mlb_batter_venue_stats_mat.sql",
    "MLB024b_mlb_sp_vs_team_mat.sql",
    "MLB025b_mlb_batter_umpire_mat.sql",
    "MLB026b_mlb_batter_vs_rp_mat.sql",
    "MLB027b_mlb_sp_lob_rate_mat.sql",
    "MLB029b_mlb_team_offensive_momentum_mat.sql",
    "MLB030b_mlb_batter_babip_rolling_mat.sql",
    "MLB031b_mlb_sp_babip_rolling_mat.sql",
    "MLB032b_mlb_team_der_rolling_mat.sql",
    "MLB033b_mlb_sp_hand_hr_rate_mat.sql",
]

_MLB_MATVIEW_REFRESH_SQL = """
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_velocity_rolling_mat;
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
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_hand_k_pct_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_batter_venue_stats_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_vs_team_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_batter_umpire_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_batter_vs_rp_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_lob_rate_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_offensive_momentum_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_batter_babip_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_babip_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_team_der_rolling_mat;
REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_sp_hand_hr_rate_mat;
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
    "mlb_sp_velocity_rolling_mat":        "mlb_sp_velocity_rolling",
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
    "mlb_sp_hand_k_pct_mat":             "mlb_sp_hand_k_pct",         # Retrain 2 #7
    "mlb_batter_venue_stats_mat":        "mlb_batter_venue_stats",    # Retrain 2 #9
    "mlb_sp_vs_team_mat":                "mlb_sp_vs_team",
    "mlb_batter_umpire_mat":             "mlb_batter_umpire",
    "mlb_batter_vs_rp_mat":             "mlb_batter_vs_rp",
    "mlb_sp_lob_rate_mat":              "mlb_sp_lob_rate",
    "mlb_team_offensive_momentum_mat":  "mlb_team_offensive_momentum",
    "mlb_batter_babip_rolling_mat":     "mlb_batter_babip_rolling",
    "mlb_sp_babip_rolling_mat":         "mlb_sp_babip_rolling",
    "mlb_team_der_rolling_mat":         "mlb_team_der_rolling",
    "mlb_sp_hand_hr_rate_mat":          "mlb_sp_hand_hr_rate",
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


def _refresh_one_matview(pg_dsn: str, matview_fqn: str) -> tuple:
    """Refresh one materialized view in its own connection.

    Returns ``(matview_fqn, None)`` on success or ``(matview_fqn, exc)`` on failure.
    Runs in a thread pool — each matview gets an independent DB connection so all
    25 refreshes execute concurrently instead of sequentially.
    """
    try:
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {matview_fqn}")
        conn.close()
        return matview_fqn, None
    except Exception as exc:
        return matview_fqn, exc


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

        # ── Refresh all matviews in parallel (each in its own DB connection) ────
        # Sequential refresh of 25 matviews on one connection took ~3–5 min wall
        # time (each CONCURRENTLY refresh holds a lock for its duration).  Parallel
        # threads each open their own connection so all refreshes run simultaneously,
        # cutting wall time to the slowest single refresh (~15–30 s).
        _matview_fqns = [
            stmt.strip().split()[-1]
            for stmt in _MLB_MATVIEW_REFRESH_SQL.strip().split(";")
            if stmt.strip()
        ]
        log.info("Refreshing %d matviews with up to 12 parallel workers", len(_matview_fqns))
        with ThreadPoolExecutor(max_workers=min(len(_matview_fqns), 12)) as _tex:
            for fqn, err in _tex.map(
                _refresh_one_matview,
                [pg_dsn] * len(_matview_fqns),
                _matview_fqns,
            ):
                if err is not None:
                    log.error("Failed to refresh %s: %s", fqn, err)
                    failed_ops.append(f"refresh:{fqn}")
                else:
                    log.info("Refreshed: %s", fqn)

        # ── Always re-apply MLB007 + MLB011 + MLB006 after the refresh cycle. ──
        # MLB007 picks up any matviews that failed on the first pass (their base
        # views may now exist after the refresh).  MLB011 + MLB006 recreate
        # mlb_lineup_quality and mlb_game_prediction/training_features, which
        # depend directly on matviews and are silently dropped by any CASCADE
        # drop — even one triggered by a concurrent pipeline run.  Running
        # unconditionally (not just when dropped_any) ensures correctness even
        # when two pipeline runs overlap.
        _apply_matview_sql("MLB007_mlb_materialized_rolling.sql")

        conn.autocommit = False
        cur2 = conn.cursor()
        try:
            for fname in ("MLB011_mlb_lineup_quality.sql", "MLB006_mlb_game_features.sql"):
                cur2.execute((_SQL_DIR / fname).read_text(encoding="utf-8"))
            conn.commit()
            log.info("Re-applied MLB007 + MLB011 + MLB006 (unconditional, dropped_any=%s)", dropped_any)
        except Exception:
            conn.rollback()
            log.exception("Failed to re-apply MLB007/MLB011/MLB006")
            failed_ops.append("reapply:MLB007+MLB011+MLB006")
        finally:
            conn.autocommit = True

        # ── Create + refresh mlb_lineup_quality_mat (depends on MLB011 VIEW) ──────
        # Must run after MLB011 is re-applied above.  Converts the slow VIEW join
        # in pitcher training/prediction SQL into a fast index lookup.
        try:
            _apply_matview_sql("MLB011b_mlb_lineup_quality_mat.sql")
            cur.execute(
                "REFRESH MATERIALIZED VIEW CONCURRENTLY features.mlb_lineup_quality_mat"
            )
            log.info("Refreshed mlb_lineup_quality_mat")
        except Exception:
            log.exception("Failed to create/refresh mlb_lineup_quality_mat")
            failed_ops.append("mlb_lineup_quality_mat")

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
    parse_lineup()            # pre-game confirmed batting orders
    parse_game_odds()         # game lines from Odds API

    # Sync scores again: parse_boxscore may have added new rows that parse_games
    # (run before boxscore) didn't see yet.
    with psycopg2.connect(_PG_DSN) as _c:
        n = sync_scores_from_boxscores(_c)
        _c.commit()
        if n:
            log.info("Post-boxscore score sync: updated %d games", n)

    # Fetch SP per-start velocity from Baseball Savant (Group N features)
    # Must run before _apply_sql_views so raw.mlb_sp_start_velocity exists when
    # MLB021 (which references it) is applied.
    try:
        from mlb_pipeline.crawler_statcast_velocity import fetch_all_sp_velocity
        with psycopg2.connect(_PG_DSN) as _c:
            n_v = fetch_all_sp_velocity(_c)
            _c.commit()
            if n_v:
                log.info("Fetched velocity data for %d pitcher-games", n_v)
    except Exception:
        log.exception(
            "SP velocity crawl failed — today's predictions will run WITHOUT velocity trend "
            "features (mlb_sp_velocity_rolling). Check Baseball Savant connectivity."
        )

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
        log.exception(
            "Weather crawl failed — today's predictions will run WITHOUT weather features "
            "(wind_speed_mph, temp_f, etc.). Check Open-Meteo connectivity."
        )

    # Create/refresh materialized rolling views for fast prediction queries
    _refresh_matviews(_PG_DSN)

    # Apply views that depend on matviews (lineup quality + MLB006 re-apply)
    _apply_post_matview_views(_PG_DSN)

    log.info("ALL MLB PARSERS COMPLETE")


if __name__ == "__main__":
    main()
