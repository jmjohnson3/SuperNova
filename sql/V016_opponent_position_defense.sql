-- ============================================================================
-- V016: Opponent Position Defense
-- ============================================================================
-- Computes how many points/rebounds/assists each team allows to players of
-- each role bucket (G / F / C) over their last 10 games.
--
-- Role buckets are derived from career FGA/48-min rate — no position table
-- needed:
--   G  (guard / ball-handler)  : FGA/48 >= 10
--   F  (forward / wing)        : FGA/48 4–9
--   C  (big / center)          : FGA/48 < 4
--
-- FIX (v2): The original view windowed over player-level rows, so
-- "10 PRECEDING" spanned only 1–2 games worth of players. Now we aggregate
-- to one row per (opponent, role, game) first, then window over those
-- game-level rows so "10 PRECEDING" means exactly 10 prior games.
--
-- Used by: features.player_training_features (V018)
--          predict_player_props.py inference SQL
-- ============================================================================

CREATE OR REPLACE VIEW features.opp_position_defense AS

WITH player_roles AS (
    -- One role bucket per player based on career FGA rate
    SELECT
        player_id,
        CASE
            WHEN AVG(CASE WHEN minutes > 5 THEN fga * 48.0 / NULLIF(minutes, 0) ELSE NULL END) >= 10 THEN 'G'
            WHEN AVG(CASE WHEN minutes > 5 THEN fga * 48.0 / NULLIF(minutes, 0) ELSE NULL END) >= 4  THEN 'F'
            ELSE 'C'
        END AS role
    FROM raw.nba_player_gamelogs
    WHERE minutes IS NOT NULL AND minutes > 5
    GROUP BY player_id
),

game_player_stats AS (
    SELECT
        p.season,
        p.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        UPPER(p.team_abbr)      AS team_abbr,
        UPPER(p.opponent_abbr)  AS opponent_abbr,
        p.player_id,
        COALESCE(pr.role, 'F')  AS role,   -- default forward if no FGA history
        p.points::numeric       AS points,
        p.rebounds::numeric     AS rebounds,
        p.assists::numeric      AS assists
    FROM raw.nba_player_gamelogs p
    JOIN raw.nba_games g
        ON g.season = p.season AND g.game_slug = p.game_slug
    LEFT JOIN player_roles pr ON pr.player_id = p.player_id
    WHERE p.minutes IS NOT NULL AND p.minutes > 5
      AND g.status = 'final'
),

-- Aggregate to one row per (opponent, role, game).
-- This is the correct granularity for the rolling window — each preceding
-- row represents one full game, not one player's line.
game_level_defense AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        start_ts_utc,
        opponent_abbr,
        role,
        SUM(points)   AS game_pts_allowed,
        SUM(rebounds) AS game_reb_allowed,
        SUM(assists)  AS game_ast_allowed
    FROM game_player_stats
    GROUP BY season, game_slug, game_date_et, start_ts_utc, opponent_abbr, role
),

-- Rolling 10-game averages at game granularity.
-- Now "10 PRECEDING" = 10 prior games, not 10 prior player rows.
rolling_defense AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        start_ts_utc,
        opponent_abbr,
        role,
        AVG(game_pts_allowed) OVER (
            PARTITION BY opponent_abbr, role
            ORDER BY game_date_et, start_ts_utc
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS opp_pts_allowed_role_10,

        AVG(game_reb_allowed) OVER (
            PARTITION BY opponent_abbr, role
            ORDER BY game_date_et, start_ts_utc
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS opp_reb_allowed_role_10,

        AVG(game_ast_allowed) OVER (
            PARTITION BY opponent_abbr, role
            ORDER BY game_date_et, start_ts_utc
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS opp_ast_allowed_role_10
    FROM game_level_defense
)

-- Join rolling game-level averages back to player-level rows
SELECT
    gps.season,
    gps.game_slug,
    gps.game_date_et,
    gps.start_ts_utc,
    gps.player_id,
    gps.team_abbr,
    gps.opponent_abbr,
    gps.role,
    rd.opp_pts_allowed_role_10,
    rd.opp_reb_allowed_role_10,
    rd.opp_ast_allowed_role_10

FROM game_player_stats gps
LEFT JOIN rolling_defense rd
    ON  rd.season        = gps.season
    AND rd.game_slug     = gps.game_slug
    AND rd.opponent_abbr = gps.opponent_abbr
    AND rd.role          = gps.role;
