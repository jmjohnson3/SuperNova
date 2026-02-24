-- ============================================================================
-- V007: Player Expanded Rolling Stats from JSONB
-- ============================================================================
-- Extracts additional player box-score stats from the `stats` JSONB column in
-- raw.nba_player_gamelogs that are NOT already surfaced as named columns
-- (minutes, points, rebounds, assists, threes_made, fga, fta).
--
-- New stats extracted:
--   steals, blocks, turnovers, offensive/defensive rebounds,
--   field goals made, three-point attempts, free throws made,
--   plus/minus, personal fouls
--
-- The view computes 5-game and 10-game rolling averages using window functions
-- that EXCLUDE the current game (ROWS BETWEEN N PRECEDING AND 1 PRECEDING)
-- to prevent data leakage during training.
--
-- Depends on: raw.nba_player_gamelogs, raw.nba_games
-- ============================================================================


CREATE OR REPLACE VIEW features.player_expanded_rolling AS

-- ---------------------------------------------------------------------------
-- CTE 1: base_stats
-- Extract JSONB fields into typed numeric columns. Only include games where
-- the player actually played (minutes > 0) and the game is final.
-- ---------------------------------------------------------------------------
WITH base_stats AS (
    SELECT
        pg.season,
        pg.game_slug,
        g.game_date_et,
        pg.player_id,
        UPPER(pg.team_abbr)     AS team_abbr,
        pg.start_ts_utc,

        -- Already-extracted columns we need for ratio calculations
        pg.minutes::numeric                                         AS minutes,
        pg.assists::numeric                                         AS assists,
        pg.threes_made::numeric                                     AS threes_made,
        pg.fga::numeric                                             AS fga,

        -- ===== JSONB EXTRACTIONS =====

        -- Steals  (defense -> stl)
        NULLIF(pg.stats->'defense'->>'stl', '')::numeric            AS stl,

        -- Blocks  (defense -> blk)
        NULLIF(pg.stats->'defense'->>'blk', '')::numeric            AS blk,

        -- Turnovers  (could live under defense OR offense depending on feed version)
        COALESCE(
            NULLIF(pg.stats->'defense'->>'tov', '')::numeric,
            NULLIF(pg.stats->'offense'->>'tov', '')::numeric
        )                                                           AS tov,

        -- Offensive rebounds
        NULLIF(pg.stats->'rebounds'->>'offReb', '')::numeric        AS off_reb,

        -- Defensive rebounds
        NULLIF(pg.stats->'rebounds'->>'defReb', '')::numeric        AS def_reb,

        -- Field goals made
        NULLIF(pg.stats->'fieldGoals'->>'fgMade', '')::numeric      AS fg_made,

        -- Three-point attempts
        NULLIF(pg.stats->'fieldGoals'->>'fg3PtAtt', '')::numeric    AS fg3_att,

        -- Free throws made
        NULLIF(pg.stats->'freeThrows'->>'ftMade', '')::numeric      AS ft_made,

        -- Free throw attempts (for FT%)
        NULLIF(pg.stats->'freeThrows'->>'ftAtt', '')::numeric       AS ft_att,

        -- Plus/minus
        NULLIF(pg.stats->'miscellaneous'->>'plusMinus', '')::numeric AS plus_minus,

        -- Personal fouls
        NULLIF(pg.stats->'miscellaneous'->>'foulsTotal', '')::numeric AS fouls

    FROM raw.nba_player_gamelogs pg
    JOIN raw.nba_games g
        ON  g.season    = pg.season
        AND g.game_slug = pg.game_slug
    WHERE pg.minutes IS NOT NULL
      AND pg.minutes > 0
      AND g.status = 'final'
),

-- ---------------------------------------------------------------------------
-- CTE 2: with_ratios
-- Pre-compute per-game shooting percentages & composite stats so we can
-- take rolling averages of the ratios themselves (not ratio of averages).
-- ---------------------------------------------------------------------------
with_ratios AS (
    SELECT
        bs.*,

        -- Stocks (steals + blocks) per game
        COALESCE(bs.stl, 0) + COALESCE(bs.blk, 0)                  AS stl_plus_blk,

        -- FG%  = fgMade / fga  (use the named fga column for the denominator)
        CASE WHEN bs.fga > 0
             THEN bs.fg_made / bs.fga
             ELSE NULL
        END                                                          AS fg_pct,

        -- 3PT% = threes_made / fg3PtAtt
        CASE WHEN bs.fg3_att > 0
             THEN bs.threes_made / bs.fg3_att
             ELSE NULL
        END                                                          AS fg3_pct,

        -- FT%  = ftMade / ftAtt
        CASE WHEN bs.ft_att > 0
             THEN bs.ft_made / bs.ft_att
             ELSE NULL
        END                                                          AS ft_pct,

        -- Assist-to-turnover ratio
        CASE WHEN bs.tov > 0
             THEN bs.assists / bs.tov
             ELSE NULL
        END                                                          AS ast_to_tov

    FROM base_stats bs
)

-- ---------------------------------------------------------------------------
-- FINAL SELECT: Rolling averages over 5 and 10 prior games.
--
-- Window frames use ROWS BETWEEN N PRECEDING AND 1 PRECEDING so that the
-- current game's stats are NEVER included in the rolling average (prevents
-- data leakage when used as pre-game features for prediction).
-- ---------------------------------------------------------------------------
SELECT
    season,
    game_slug,
    game_date_et,
    player_id,
    team_abbr,

    -- ===== STEALS =====
    AVG(stl)            OVER w5     AS stl_avg_5,
    AVG(stl)            OVER w10    AS stl_avg_10,

    -- ===== BLOCKS =====
    AVG(blk)            OVER w5     AS blk_avg_5,
    AVG(blk)            OVER w10    AS blk_avg_10,

    -- ===== TURNOVERS =====
    AVG(tov)            OVER w5     AS tov_avg_5,
    AVG(tov)            OVER w10    AS tov_avg_10,

    -- ===== STOCKS (steals + blocks combined) =====
    AVG(stl_plus_blk)   OVER w10   AS stl_plus_blk_avg_10,

    -- ===== REBOUND SPLIT =====
    AVG(off_reb)        OVER w10    AS off_reb_avg_10,
    AVG(def_reb)        OVER w10    AS def_reb_avg_10,

    -- ===== SHOOTING PERCENTAGES (rolling avg of per-game pcts) =====
    AVG(fg_pct)         OVER w10    AS fg_pct_avg_10,
    AVG(fg3_pct)        OVER w10    AS fg3_pct_avg_10,
    AVG(ft_pct)         OVER w10    AS ft_pct_avg_10,

    -- ===== PLUS/MINUS =====
    AVG(plus_minus)     OVER w5     AS plus_minus_avg_5,
    AVG(plus_minus)     OVER w10    AS plus_minus_avg_10,

    -- ===== PERSONAL FOULS (key for referee analysis) =====
    AVG(fouls)          OVER w5     AS fouls_avg_5,
    AVG(fouls)          OVER w10    AS fouls_avg_10,

    -- ===== ASSIST-TO-TURNOVER RATIO =====
    AVG(ast_to_tov)     OVER w10    AS ast_to_tov_ratio_avg_10

FROM with_ratios

WINDOW
    w5  AS (
        PARTITION BY player_id
        ORDER BY start_ts_utc
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ),
    w10 AS (
        PARTITION BY player_id
        ORDER BY start_ts_utc
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    );
