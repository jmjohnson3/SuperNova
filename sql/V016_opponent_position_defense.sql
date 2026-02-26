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
)

SELECT
    season,
    game_slug,
    game_date_et,
    start_ts_utc,
    player_id,
    team_abbr,
    opponent_abbr,
    role,

    -- Rolling 10-game avg of stats ALLOWED by this player's opponent to their role bucket.
    -- Window partitions on opponent_abbr + role so we measure how THIS defence handles
    -- THIS type of player over the 10 games preceding the current row.
    AVG(points)   OVER (
        PARTITION BY opponent_abbr, role
        ORDER BY game_date_et, start_ts_utc
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS opp_pts_allowed_role_10,

    AVG(rebounds) OVER (
        PARTITION BY opponent_abbr, role
        ORDER BY game_date_et, start_ts_utc
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS opp_reb_allowed_role_10,

    AVG(assists)  OVER (
        PARTITION BY opponent_abbr, role
        ORDER BY game_date_et, start_ts_utc
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS opp_ast_allowed_role_10

FROM game_player_stats;
