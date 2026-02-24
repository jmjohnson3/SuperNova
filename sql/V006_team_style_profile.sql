-- ============================================================================
-- V006: Team Style Profile Features
-- ============================================================================
-- Extracts previously unused counting stats from the JSONB `stats` column in
-- raw.nba_boxscore_team_stats, then computes rolling 5-game and 10-game
-- pregame averages plus derived rate stats that characterise a team's style
-- of play (transition tendency, inside scoring, depth, ball security, etc.).
--
-- Source table:  raw.nba_boxscore_team_stats  (stats JSONB)
-- Spine:         features.team_game_spine     (ordering & game_date_et)
--
-- Newly extracted JSONB keys (not used in V001 team_advanced_efficiency):
--   defense  -> stl, blk, blkAgainst, defReb
--   rebounds -> reb
--   offense  -> ast, ptsOffTov, pts2ndChance, ptsFastBreak,
--               ptsPaint (or ptsPaintTotal), ptsBench (or ptsBenchTotal)
--   miscellaneous -> foulsTotal, plusMinus
--
-- Rolling windows exclude the current game (ROWS BETWEEN N PRECEDING AND
-- 1 PRECEDING) so that every feature is strictly pregame, avoiding leakage.
-- ============================================================================

CREATE OR REPLACE VIEW features.team_style_profile AS

-- ---------------------------------------------------------------------------
-- CTE 1: Extract raw counting stats from the JSONB column
-- ---------------------------------------------------------------------------
WITH box AS (
    SELECT
        s.season,
        s.game_slug,
        s.team_abbr,
        sp.game_date_et,
        sp.start_ts_utc,

        -- === Defense ===
        NULLIF((s.stats->'defense'->>'stl'), '')::numeric            AS stl,
        NULLIF((s.stats->'defense'->>'blk'), '')::numeric            AS blk,
        NULLIF((s.stats->'defense'->>'blkAgainst'), '')::numeric     AS blk_against,
        NULLIF((s.stats->'defense'->>'defReb'), '')::numeric         AS def_reb,
        NULLIF((s.stats->'defense'->>'tov'), '')::numeric            AS tov,

        -- === Rebounds ===
        NULLIF((s.stats->'rebounds'->>'reb'), '')::numeric           AS reb,
        NULLIF((s.stats->'rebounds'->>'offReb'), '')::numeric        AS off_reb,

        -- === Offense ===
        NULLIF((s.stats->'offense'->>'ast'), '')::numeric            AS ast,
        NULLIF((s.stats->'offense'->>'pts'), '')::numeric            AS pts,
        NULLIF((s.stats->'offense'->>'ptsOffTov'), '')::numeric      AS pts_off_tov,
        NULLIF((s.stats->'offense'->>'pts2ndChance'), '')::numeric   AS pts_2nd_chance,
        NULLIF((s.stats->'offense'->>'ptsFastBreak'), '')::numeric   AS pts_fast_break,
        -- Handle potential key naming variations (ptsPaint vs ptsPaintTotal)
        COALESCE(
            NULLIF((s.stats->'offense'->>'ptsPaint'), ''),
            NULLIF((s.stats->'offense'->>'ptsPaintTotal'), '')
        )::numeric                                                   AS pts_paint,
        -- Handle potential key naming variations (ptsBench vs ptsBenchTotal)
        COALESCE(
            NULLIF((s.stats->'offense'->>'ptsBench'), ''),
            NULLIF((s.stats->'offense'->>'ptsBenchTotal'), '')
        )::numeric                                                   AS pts_bench,

        -- === Miscellaneous ===
        NULLIF((s.stats->'miscellaneous'->>'foulsTotal'), '')::numeric AS fouls,
        NULLIF((s.stats->'miscellaneous'->>'plusMinus'), '')::numeric   AS plus_minus

    FROM raw.nba_boxscore_team_stats s
    JOIN features.team_game_spine sp
      ON sp.season    = s.season
     AND sp.game_slug = s.game_slug
     AND sp.team_abbr = s.team_abbr
),

-- ---------------------------------------------------------------------------
-- CTE 2: Add a deterministic ordering timestamp (same pattern as V001)
-- ---------------------------------------------------------------------------
ordered AS (
    SELECT
        *,
        COALESCE(start_ts_utc, game_date_et::timestamp with time zone) AS order_ts
    FROM box
)

-- ---------------------------------------------------------------------------
-- Final SELECT: rolling 5-game and 10-game pregame averages + derived rates
-- ---------------------------------------------------------------------------
SELECT
    season,
    game_slug,
    team_abbr,
    game_date_et,

    -- =======================================================================
    -- ROLLING 5-GAME AVERAGES (pregame, excludes current game)
    -- =======================================================================
    AVG(stl)            OVER w5  AS stl_avg_5,
    AVG(blk)            OVER w5  AS blk_avg_5,
    AVG(ast)            OVER w5  AS ast_avg_5,
    AVG(reb)            OVER w5  AS reb_avg_5,
    AVG(off_reb)        OVER w5  AS off_reb_avg_5,
    AVG(def_reb)        OVER w5  AS def_reb_avg_5,
    AVG(pts_off_tov)    OVER w5  AS pts_off_tov_avg_5,
    AVG(pts_2nd_chance) OVER w5  AS pts_2nd_chance_avg_5,
    AVG(pts_fast_break) OVER w5  AS pts_fast_break_avg_5,
    AVG(pts_paint)      OVER w5  AS pts_paint_avg_5,
    AVG(pts_bench)      OVER w5  AS pts_bench_avg_5,
    AVG(fouls)          OVER w5  AS fouls_avg_5,
    AVG(plus_minus)     OVER w5  AS plus_minus_avg_5,

    -- =======================================================================
    -- ROLLING 10-GAME AVERAGES (pregame, excludes current game)
    -- =======================================================================
    AVG(stl)            OVER w10 AS stl_avg_10,
    AVG(blk)            OVER w10 AS blk_avg_10,
    AVG(ast)            OVER w10 AS ast_avg_10,
    AVG(reb)            OVER w10 AS reb_avg_10,
    AVG(off_reb)        OVER w10 AS off_reb_avg_10,
    AVG(def_reb)        OVER w10 AS def_reb_avg_10,
    AVG(pts_off_tov)    OVER w10 AS pts_off_tov_avg_10,
    AVG(pts_2nd_chance) OVER w10 AS pts_2nd_chance_avg_10,
    AVG(pts_fast_break) OVER w10 AS pts_fast_break_avg_10,
    AVG(pts_paint)      OVER w10 AS pts_paint_avg_10,
    AVG(pts_bench)      OVER w10 AS pts_bench_avg_10,
    AVG(fouls)          OVER w10 AS fouls_avg_10,
    AVG(plus_minus)     OVER w10 AS plus_minus_avg_10,

    -- =======================================================================
    -- DERIVED RATE STATS (10-game window for stability)
    -- =======================================================================

    -- Stocks: steals + blocks (defensive disruption metric)
    AVG(stl)  OVER w10 + AVG(blk) OVER w10            AS stl_plus_blk_avg_10,

    -- Assist-to-turnover ratio (ball security / playmaking quality)
    -- Guard against zero turnovers with NULLIF
    CASE
        WHEN AVG(tov) OVER w10 IS NOT NULL
         AND AVG(tov) OVER w10 > 0
        THEN AVG(ast) OVER w10 / AVG(tov) OVER w10
        ELSE NULL
    END                                                AS ast_to_tov_ratio_avg_10,

    -- Fast-break points as % of total points (transition tendency)
    CASE
        WHEN AVG(pts) OVER w10 IS NOT NULL
         AND AVG(pts) OVER w10 > 0
        THEN AVG(pts_fast_break) OVER w10
             / NULLIF(AVG(pts) OVER w10, 0)
        ELSE NULL
    END                                                AS fast_break_pct_avg_10,

    -- Points in the paint as % of total points (inside scoring tendency)
    CASE
        WHEN AVG(pts) OVER w10 IS NOT NULL
         AND AVG(pts) OVER w10 > 0
        THEN AVG(pts_paint) OVER w10
             / NULLIF(AVG(pts) OVER w10, 0)
        ELSE NULL
    END                                                AS paint_pct_avg_10,

    -- Bench points as % of total points (roster depth indicator)
    CASE
        WHEN AVG(pts) OVER w10 IS NOT NULL
         AND AVG(pts) OVER w10 > 0
        THEN AVG(pts_bench) OVER w10
             / NULLIF(AVG(pts) OVER w10, 0)
        ELSE NULL
    END                                                AS bench_pct_avg_10

FROM ordered

-- ---------------------------------------------------------------------------
-- Window definitions: pregame rolling windows partitioned by season & team
-- order_ts provides sub-day precision; game_slug breaks ties deterministically
-- ROWS BETWEEN N PRECEDING AND 1 PRECEDING ensures we never include the
-- current game's stats (strict pregame constraint).
-- ---------------------------------------------------------------------------
WINDOW
    w5  AS (
        PARTITION BY season, team_abbr
        ORDER BY order_ts, game_slug
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ),
    w10 AS (
        PARTITION BY season, team_abbr
        ORDER BY order_ts, game_slug
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    );
