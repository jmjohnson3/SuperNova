-- ============================================================================
-- V024: Opponent Shot Defense — PBP-Derived Team-Level Defensive Shot Profile
-- ============================================================================
-- Computes per-team rolling shot-type defense rates from play-by-play data.
-- For each game, measures what types of shots each team ALLOWED (paint shots,
-- 3-pointers, driving shots, etc.) as a fraction of total FGA allowed.
--
-- New view: features.opponent_shot_defense
--
-- Features produced (rolling pregame averages per defending team per game):
--   opp_paint_allowed_avg_10/5      — % FGA allowed that are paint shots
--   opp_pullup_allowed_avg_10       — % FGA allowed that are off-dribble
--   opp_driving_allowed_avg_10      — % FGA allowed tagged "Driving*"
--   opp_catch_shoot_allowed_avg_10  — % FGA allowed that are assisted (catch-and-shoot)
--   opp_3pt_allowed_avg_10/5        — % FGA allowed that are 3-pointers
--   opp_blocked_rate_avg_10         — % FGA allowed that the defense blocked
--
-- Key implementation notes:
--   - Joins player shots → player gamelogs to identify shooter's team
--   - Defending team = the OTHER team in the game (not the shooter's team)
--   - ::numeric::int on player IDs handles MSF float IDs like "10090.0"
--   - UPPER() on team_abbr ensures consistent case with opponent_abbr in V022
--   - Window ROWS BETWEEN N PRECEDING AND 1 PRECEDING = strictly pregame
--   - Partitioned by (season, defending_team) for correct rolling context
-- ============================================================================

CREATE OR REPLACE VIEW features.opponent_shot_defense AS

WITH raw_shots AS (
    -- Same PBP extraction as V023: fieldGoalAttempt events with a shootingPlayer
    SELECT
        p.season,
        p.game_slug,
        (p.raw_json->'fieldGoalAttempt'->'shootingPlayer'->>'id')::numeric::int AS player_id,
        p.raw_json->'fieldGoalAttempt'->>'shotType'        AS shot_type,
        p.raw_json->'fieldGoalAttempt'->>'result'          AS result,
        (p.raw_json->'fieldGoalAttempt'->>'points')::numeric AS points,
        p.raw_json->'fieldGoalAttempt'->>'assistingPlayer' AS assisting_json
    FROM raw.nba_pbp_plays p
    WHERE p.event_type = 'fieldGoalAttempt'
      AND p.raw_json->'fieldGoalAttempt'->'shootingPlayer' IS NOT NULL
      AND (p.raw_json->'fieldGoalAttempt'->'shootingPlayer'->>'id') IS NOT NULL
),
shot_with_teams AS (
    -- Join player gamelogs to get shooter's team, then derive the defending team
    SELECT
        rs.season,
        rs.game_slug,
        rs.shot_type,
        rs.result,
        rs.points,
        rs.assisting_json,
        UPPER(gl.team_abbr) AS shooting_team,
        CASE WHEN UPPER(gl.team_abbr) = g.home_team_abbr
             THEN g.away_team_abbr
             ELSE g.home_team_abbr
        END AS defending_team
    FROM raw_shots rs
    JOIN raw.nba_player_gamelogs gl
      ON gl.season = rs.season AND gl.game_slug = rs.game_slug AND gl.player_id = rs.player_id
    JOIN raw.nba_games g
      ON g.season = rs.season AND g.game_slug = rs.game_slug AND g.status = 'final'
    WHERE rs.player_id IS NOT NULL
),
shot_flags AS (
    -- Same LOWER()/LIKE classification as V023 but aggregated at team-defense level
    SELECT
        season, game_slug, defending_team,
        -- Paint shots: layup, dunk, hook, cutting, alley oop, putback, finger roll
        CASE WHEN LOWER(shot_type) LIKE '%layup%'
              OR LOWER(shot_type) LIKE '%dunk%'
              OR LOWER(shot_type) LIKE '%hook%'
              OR LOWER(shot_type) LIKE '%cutting%'
              OR LOWER(shot_type) LIKE '%alley%'
              OR LOWER(shot_type) LIKE '%putback%'
              OR LOWER(shot_type) LIKE '%finger%'
             THEN 1 ELSE 0 END AS is_paint,
        -- Pull-up / off-dribble shots
        CASE WHEN LOWER(shot_type) LIKE '%pullup%'
              OR LOWER(shot_type) LIKE '%step back%'
              OR LOWER(shot_type) LIKE '%fadeaway%'
              OR LOWER(shot_type) LIKE '%turnaround%'
             THEN 1 ELSE 0 END AS is_pullup,
        -- Driving shots (foul-draw indicator)
        CASE WHEN LOWER(shot_type) LIKE 'driving%'
             THEN 1 ELSE 0 END AS is_driving,
        -- Catch-and-shoot: assisting player present
        CASE WHEN assisting_json IS NOT NULL THEN 1 ELSE 0 END AS is_assisted,
        -- Blocked shots (this team's block rate on defense)
        CASE WHEN result = 'BLOCKED' THEN 1 ELSE 0 END AS is_blocked,
        -- Three-pointer
        CASE WHEN points = 3 THEN 1 ELSE 0 END AS is_three
    FROM shot_with_teams
),
per_game_defense AS (
    -- Aggregate shot counts per (game, defending team)
    SELECT
        season, game_slug, defending_team,
        COUNT(*)               AS fga_allowed,
        SUM(is_paint)          AS paint_fga_allowed,
        SUM(is_pullup)         AS pullup_fga_allowed,
        SUM(is_driving)        AS driving_fga_allowed,
        SUM(is_assisted)       AS catch_shoot_fga_allowed,
        SUM(is_blocked)        AS blocked_fga,
        SUM(is_three)          AS three_fga_allowed
    FROM shot_flags
    GROUP BY season, game_slug, defending_team
),
with_rates AS (
    -- Convert counts to rates; pull game_date_et and order_ts from raw.nba_games
    SELECT
        pd.*,
        g.game_date_et,
        COALESCE(g.start_ts_utc, g.game_date_et::timestamp with time zone) AS order_ts,
        CASE WHEN pd.fga_allowed > 0 THEN pd.paint_fga_allowed::numeric        / pd.fga_allowed ELSE NULL END AS opp_paint_rate,
        CASE WHEN pd.fga_allowed > 0 THEN pd.pullup_fga_allowed::numeric       / pd.fga_allowed ELSE NULL END AS opp_pullup_rate,
        CASE WHEN pd.fga_allowed > 0 THEN pd.driving_fga_allowed::numeric      / pd.fga_allowed ELSE NULL END AS opp_driving_rate,
        CASE WHEN pd.fga_allowed > 0 THEN pd.catch_shoot_fga_allowed::numeric  / pd.fga_allowed ELSE NULL END AS opp_catch_shoot_rate,
        CASE WHEN pd.fga_allowed > 0 THEN pd.three_fga_allowed::numeric        / pd.fga_allowed ELSE NULL END AS opp_3pt_rate,
        CASE WHEN pd.fga_allowed > 0 THEN pd.blocked_fga::numeric              / pd.fga_allowed ELSE NULL END AS opp_blocked_rate
    FROM per_game_defense pd
    JOIN raw.nba_games g
      ON g.season = pd.season AND g.game_slug = pd.game_slug
)

SELECT
    season, game_slug, defending_team AS opponent_abbr, game_date_et,
    AVG(opp_paint_rate)        OVER w10 AS opp_paint_allowed_avg_10,
    AVG(opp_pullup_rate)       OVER w10 AS opp_pullup_allowed_avg_10,
    AVG(opp_driving_rate)      OVER w10 AS opp_driving_allowed_avg_10,
    AVG(opp_catch_shoot_rate)  OVER w10 AS opp_catch_shoot_allowed_avg_10,
    AVG(opp_3pt_rate)          OVER w10 AS opp_3pt_allowed_avg_10,
    AVG(opp_blocked_rate)      OVER w10 AS opp_blocked_rate_avg_10,
    AVG(opp_paint_rate)        OVER w5  AS opp_paint_allowed_avg_5,
    AVG(opp_3pt_rate)          OVER w5  AS opp_3pt_allowed_avg_5
FROM with_rates
WINDOW
    w5  AS (PARTITION BY season, defending_team
            ORDER BY order_ts, game_slug ROWS BETWEEN  5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, defending_team
            ORDER BY order_ts, game_slug ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);
