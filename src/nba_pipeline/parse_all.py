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

_TEAM_FORM_FEATURES_SQL = """
CREATE OR REPLACE VIEW features.team_form_features AS
SELECT
    season,
    team_abbr,
    game_slug,
    game_date_et,
    AVG(points_for)     OVER w5  AS pts_for_avg_5,
    AVG(points_against) OVER w5  AS pts_against_avg_5,
    AVG(points_for)     OVER w10 AS pts_for_avg_10,
    AVG(points_against) OVER w10 AS pts_against_avg_10,
    STDDEV_SAMP(points_for) OVER w10 AS pts_for_sd_10,
    AVG(points_for)     OVER w20 AS pts_for_avg_20,
    AVG(points_against) OVER w20 AS pts_against_avg_20
FROM features.team_game_spine
WINDOW
    w5  AS (PARTITION BY season, team_abbr ORDER BY game_date_et, game_slug ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, team_abbr ORDER BY game_date_et, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    w20 AS (PARTITION BY season, team_abbr ORDER BY game_date_et, game_slug ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
"""

_TEAM_PREGAME_ROLLING_SQL = """
CREATE OR REPLACE VIEW features.team_pregame_rolling_boxscore AS
WITH joined AS (
    SELECT
        s.season, s.team_abbr, s.game_slug, s.game_date_et, s.start_ts_utc,
        m.poss_est, m.fg_pct, m.ft_pct, m.tov, m.oreb
    FROM features.team_game_spine s
    LEFT JOIN features.team_boxscore_metrics m
      ON m.season = s.season AND m.team_abbr = s.team_abbr AND m.game_slug = s.game_slug
),
ordered AS (
    SELECT *,
        COALESCE(start_ts_utc, game_date_et::timestamp with time zone) AS order_ts
    FROM joined
)
SELECT
    season, team_abbr, game_slug, game_date_et,
    COUNT(poss_est) OVER w5  AS pace_n_5,
    COUNT(poss_est) OVER w10 AS pace_n_10,
    AVG(poss_est)   OVER w5  AS pace_avg_5,
    AVG(poss_est)   OVER w10 AS pace_avg_10,
    AVG(fg_pct)     OVER w5  AS fg_pct_avg_5,
    AVG(ft_pct)     OVER w5  AS ft_pct_avg_5,
    AVG(tov)        OVER w5  AS tov_avg_5,
    AVG(oreb)       OVER w5  AS oreb_avg_5,
    AVG(poss_est)   OVER w20 AS pace_avg_20
FROM ordered
WINDOW
    w5  AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    w20 AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
"""


def _apply_view_fixes(pg_dsn: str) -> None:
    """Apply SQL view bug fixes. Idempotent — safe to run on every parse_all.

    Fixes applied:
    - team_form_features: add pts_for_avg_20 / pts_against_avg_20 (w20 window)
    - team_pregame_rolling_boxscore: add pace_avg_20 (w20 window)
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
        # Ensure team_form_features and team_pregame_rolling_boxscore have _avg_20 columns.
        # CREATE OR REPLACE is safe — same base columns, new w20 columns appended.
        cur.execute(_TEAM_FORM_FEATURES_SQL)
        log.info("Applied team_form_features w20 update")
        cur.execute(_TEAM_PREGAME_ROLLING_SQL)
        log.info("Applied team_pregame_rolling_boxscore w20 update")

        # V006/V001: CREATE OR REPLACE is fine (new _avg_20 columns added)
        cur.execute((_SQL_DIR / "V006_team_style_profile.sql").read_text(encoding="utf-8"))
        log.info("Applied V006 team_style_profile w20 update")
        cur.execute((_SQL_DIR / "V001_new_feature_views.sql").read_text(encoding="utf-8"))
        log.info("Applied V001 new_feature_views w20 + home/away splits update")

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

        # V014: adds new _avg_20 + home/away split columns.
        # DROPs the wrapper views game_training/prediction_features CASCADE.
        # The matviews (_mat) are NOT dropped because the wrapper depends ON the
        # matview, not vice versa.  _matview_needs_recreate() in
        # _materialize_game_features() will detect the column count mismatch
        # between the old matview and the new full V014 view and trigger recreation.
        # Also drops player_training_features via CASCADE from game_training_features.
        cur.execute((_SQL_DIR / "V014_updated_game_views.sql").read_text(encoding="utf-8"))
        log.info("Applied V014 game_views w20 update")

        # V023: player shot profile view (PBP-derived shot quality features).
        # Must run BEFORE V022 since V022 LEFT JOINs features.player_shot_profile.
        cur.execute((_SQL_DIR / "V023_player_shot_profile.sql").read_text(encoding="utf-8"))
        log.info("Applied V023 player_shot_profile (PBP shot quality features)")

        # V024: opponent shot defense view (PBP-derived team defensive shot profile).
        # Must run BEFORE V022 since V022 LEFT JOINs features.opponent_shot_defense.
        cur.execute((_SQL_DIR / "V024_opponent_shot_defense.sql").read_text(encoding="utf-8"))
        log.info("Applied V024 opponent_shot_defense (PBP opponent defensive shot profile)")

        # V022: recreate player_training_features (includes DROP VIEW IF EXISTS CASCADE).
        # Must run AFTER V014 since V014 drops player_training_features via CASCADE.
        # Must run AFTER V023 + V024 since it LEFT JOINs both views.
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


def _matview_needs_recreate(conn, matview_name: str, view_name: str) -> bool:
    """Return True if the matview column count differs from the underlying view.

    When the underlying view gains new columns (e.g. after a SQL update), the
    materialized view's schema is stale.  Dropping it forces the first-run path
    in _materialize_game_features to recreate it with the correct schema.

    IMPORTANT: information_schema.columns does NOT include materialized views —
    use pg_attribute + pg_class instead (covers both views and matviews).
    """
    query = """
        SELECT COUNT(*)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'features' AND c.relname = %s
          AND a.attnum > 0 AND NOT a.attisdropped
    """
    with conn.cursor() as cur:
        cur.execute(query, (matview_name,))
        mat_cols = cur.fetchone()[0]
        cur.execute(query, (view_name,))
        view_cols = cur.fetchone()[0]
    return mat_cols != view_cols


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
            # 2b. Schema-change detection: if the underlying view has more
            #     columns than the matview (e.g. after a SQL update), drop the
            #     matview so the first-run path above recreates it correctly.
            # ------------------------------------------------------------------
            elif _matview_needs_recreate(conn, mat_name, view_name):
                log.info(
                    "Schema change detected for features.%s — dropping matview for recreation",
                    mat_name,
                )
                # Save player_training_features SQL before CASCADE drop
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

                cur.execute(f"DROP MATERIALIZED VIEW IF EXISTS features.{mat_name} CASCADE")
                conn.commit()

                # Recreate from the (now-updated) underlying view SQL
                cur.execute("SELECT pg_get_viewdef(%s, true)", (f"features.{view_name}",))
                view_sql = cur.fetchone()[0].rstrip().rstrip(";")
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
                log.info("  Recreated features.%s with updated schema", mat_name)

            # ------------------------------------------------------------------
            # 3. Refresh (always — picks up new data from this parse run)
            # ------------------------------------------------------------------
            log.info("Refreshing features.%s ...", mat_name)
            cur.execute(f"REFRESH MATERIALIZED VIEW features.{mat_name}")
            conn.commit()
            log.info("  Done refreshing features.%s", mat_name)

            # ------------------------------------------------------------------
            # 4. Restore thin wrapper (always).
            # _apply_view_fixes runs CREATE OR REPLACE VIEW on game_training_features
            # and game_prediction_features every parse_all run, replacing the thin
            # wrapper with the full view SQL.  This ensures the wrapper always
            # points to the matview for fast queries.
            # CREATE OR REPLACE VIEW (not DROP+CREATE) so player_training_features
            # and other downstream views are NOT cascade-dropped.
            # ------------------------------------------------------------------
            cur.execute(
                f"CREATE OR REPLACE VIEW features.{view_name}"
                f" AS SELECT * FROM features.{mat_name}"
            )
            conn.commit()

    except Exception:
        conn.rollback()
        log.exception(
            "Failed to materialize/refresh game features — predictions will fall back to slow views"
        )
    finally:
        conn.close()


def _audit_prop_name_coverage(pg_dsn: str) -> None:
    """Log the fraction of player-game training rows that have a matched book line.

    Low coverage (<15%) usually means the name-normalization join is broken or
    the prop-line backfill hasn't been run yet.
    """
    conn = psycopg2.connect(pg_dsn)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*)                                                    AS total,
                COUNT(*) FILTER (WHERE prev_book_line_pts IS NOT NULL)      AS matched
            FROM features.player_training_features
        """)
        row = cur.fetchone()
        if row and row[0]:
            total, matched = row
            pct = 100.0 * matched / total
            log.info("Prop line name-match coverage: %d/%d rows (%.1f%%)", matched, total, pct)
            if pct < 15.0:
                log.warning(
                    "Prop line coverage %.1f%% is below 15%% threshold — "
                    "check name normalization or run: python -m nba_pipeline.crawler_oddsapi_backfill --prop-lines",
                    pct,
                )
        else:
            log.info("player_training_features is empty — skipping prop name-match audit")
    except Exception:
        log.exception("Could not audit prop line name-match coverage (safe to ignore if view doesn't exist)")
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
    _audit_prop_name_coverage(_PG_DSN)

    log.info("ALL PARSERS COMPLETE")

if __name__ == "__main__":
    main()
