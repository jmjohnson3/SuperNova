-- ============================================================================
-- V023: Player Shot Profile — PBP-Derived Shot Quality Features
-- ============================================================================
-- Extracts shot profile metrics from raw play-by-play data (raw.nba_pbp_plays)
-- using the fieldGoalAttempt event type.  These features encode shot difficulty,
-- self-creation tendency, and foul-draw likelihood — signals unavailable from
-- box-score data alone.
--
-- New view: features.player_shot_profile
--
-- Features produced (rolling pregame averages per player per game):
--   paint_shot_rate_avg_10/5    — % FGA that are paint shots (layup/dunk/hook/etc.)
--   pullup_shot_rate_avg_10     — % FGA that are off-dribble (pullup/step-back/fadeaway)
--   driving_shot_rate_avg_10    — % FGA tagged "Driving*" (foul-draw proxy)
--   catch_and_shoot_rate_avg_10/5 — % FGA with an assisting player (assisted shot)
--   three_pt_rate_pbp_avg_10    — % FGA that are 3-pointers (PBP-sourced)
--   blocked_rate_avg_10         — % FGA that were blocked (shot contestedness proxy)
--
-- Key implementation notes:
--   - ::numeric::int on player IDs handles MSF float IDs like "10090.0"
--   - LOWER() applied before LIKE for case-insensitive shot_type matching
--   - assisting_json IS NOT NULL (text ->> extract) correctly identifies assisted shots
--   - Window ROWS BETWEEN N PRECEDING AND 1 PRECEDING = strictly pregame (no leakage)
--   - Window partitioned by player_id only (no season), matching V007 pattern
--   - XGBoost handles NULL natively for early-season / no-PBP-data rows
-- ============================================================================

CREATE OR REPLACE VIEW features.player_shot_profile AS

WITH raw_shots AS (
    SELECT
        p.season,
        p.game_slug,
        (p.raw_json->'fieldGoalAttempt'->'shootingPlayer'->>'id')::numeric::int AS player_id,
        p.raw_json->'fieldGoalAttempt'->>'result'          AS result,
        (p.raw_json->'fieldGoalAttempt'->>'points')::numeric AS points,
        p.raw_json->'fieldGoalAttempt'->>'shotType'        AS shot_type,
        p.raw_json->'fieldGoalAttempt'->>'assistingPlayer' AS assisting_json
    FROM raw.nba_pbp_plays p
    WHERE p.event_type = 'fieldGoalAttempt'
      AND p.raw_json->'fieldGoalAttempt'->'shootingPlayer' IS NOT NULL
      AND (p.raw_json->'fieldGoalAttempt'->'shootingPlayer'->>'id') IS NOT NULL
),
shot_flags AS (
    SELECT
        season, game_slug, player_id,
        -- Paint shots: layup, dunk, hook, cutting, alley oop, putback, finger roll
        CASE WHEN LOWER(shot_type) LIKE '%layup%'
              OR LOWER(shot_type) LIKE '%dunk%'
              OR LOWER(shot_type) LIKE '%hook%'
              OR LOWER(shot_type) LIKE '%cutting%'
              OR LOWER(shot_type) LIKE '%alley%'
              OR LOWER(shot_type) LIKE '%putback%'
              OR LOWER(shot_type) LIKE '%finger%'
             THEN 1 ELSE 0 END AS is_paint,
        -- Pull-up / off-dribble: self-created harder shots
        CASE WHEN LOWER(shot_type) LIKE '%pullup%'
              OR LOWER(shot_type) LIKE '%step back%'
              OR LOWER(shot_type) LIKE '%fadeaway%'
              OR LOWER(shot_type) LIKE '%turnaround%'
             THEN 1 ELSE 0 END AS is_pullup,
        -- Driving shots (foul-draw indicator)
        CASE WHEN LOWER(shot_type) LIKE 'driving%'
             THEN 1 ELSE 0 END AS is_driving,
        -- Catch-and-shoot: assisting player present (text ->> returns NULL for JSON null)
        CASE WHEN assisting_json IS NOT NULL THEN 1 ELSE 0 END AS is_assisted,
        -- Blocked shots (shot contestedness proxy)
        CASE WHEN result = 'BLOCKED' THEN 1 ELSE 0 END AS is_blocked,
        -- Three-pointer (points stored as float in JSONB)
        CASE WHEN points = 3 THEN 1 ELSE 0 END AS is_three
    FROM raw_shots
    WHERE player_id IS NOT NULL
),
per_game AS (
    SELECT
        season, game_slug, player_id,
        COUNT(*)            AS pbp_fga,
        SUM(is_three)       AS pbp_3pa,
        SUM(is_paint)       AS pbp_paint_fga,
        SUM(is_pullup)      AS pbp_pullup_fga,
        SUM(is_driving)     AS pbp_driving_fga,
        SUM(is_assisted)    AS pbp_assisted_fga,
        SUM(is_blocked)     AS pbp_blocked_fga
    FROM shot_flags
    GROUP BY season, game_slug, player_id
),
with_rates AS (
    SELECT
        pg.*,
        g.game_date_et,
        COALESCE(gl.start_ts_utc, g.game_date_et::timestamp with time zone) AS order_ts,
        CASE WHEN pbp_fga > 0 THEN pbp_paint_fga::numeric    / pbp_fga ELSE NULL END AS paint_shot_rate,
        CASE WHEN pbp_fga > 0 THEN pbp_pullup_fga::numeric   / pbp_fga ELSE NULL END AS pullup_shot_rate,
        CASE WHEN pbp_fga > 0 THEN pbp_driving_fga::numeric  / pbp_fga ELSE NULL END AS driving_shot_rate,
        CASE WHEN pbp_fga > 0 THEN pbp_assisted_fga::numeric / pbp_fga ELSE NULL END AS catch_and_shoot_rate,
        CASE WHEN pbp_fga > 0 THEN pbp_3pa::numeric           / pbp_fga ELSE NULL END AS three_pt_rate_pbp,
        CASE WHEN pbp_fga > 0 THEN pbp_blocked_fga::numeric  / pbp_fga ELSE NULL END AS blocked_rate
    FROM per_game pg
    JOIN raw.nba_games g
      ON g.season = pg.season AND g.game_slug = pg.game_slug AND g.status = 'final'
    LEFT JOIN raw.nba_player_gamelogs gl
      ON gl.season = pg.season AND gl.game_slug = pg.game_slug AND gl.player_id = pg.player_id
)

SELECT
    season, game_slug, player_id, game_date_et,
    -- Rolling 10-game pregame averages (strictly prior to this game)
    AVG(paint_shot_rate)        OVER w10 AS paint_shot_rate_avg_10,
    AVG(pullup_shot_rate)       OVER w10 AS pullup_shot_rate_avg_10,
    AVG(driving_shot_rate)      OVER w10 AS driving_shot_rate_avg_10,
    AVG(catch_and_shoot_rate)   OVER w10 AS catch_and_shoot_rate_avg_10,
    AVG(three_pt_rate_pbp)      OVER w10 AS three_pt_rate_pbp_avg_10,
    AVG(blocked_rate)           OVER w10 AS blocked_rate_avg_10,
    -- Rolling 5-game averages for shorter-window trend
    AVG(paint_shot_rate)        OVER w5  AS paint_shot_rate_avg_5,
    AVG(catch_and_shoot_rate)   OVER w5  AS catch_and_shoot_rate_avg_5
FROM with_rates
WINDOW
    w5  AS (PARTITION BY player_id ORDER BY order_ts ROWS BETWEEN  5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY player_id ORDER BY order_ts ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);
