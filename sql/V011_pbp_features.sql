-- ============================================================================
-- V011: Play-by-Play Derived Features
-- ============================================================================
-- The PBP data (raw.nba_pbp_plays) is currently used ONLY for the clutch
-- performance view. This unlocks:
--   1. team_pbp_profile   -- play-type distribution (3PT rate, foul rate, etc.)
--   2. team_quarter_scoring -- quarter-by-quarter scoring tendencies
-- ============================================================================


-- ============================================================================
-- VIEW 1: Team PBP Profile — play-type distribution rolling averages
-- ============================================================================
CREATE OR REPLACE VIEW features.team_pbp_profile AS

WITH game_plays AS (
    -- Count play types per team per game
    SELECT
        p.season,
        p.game_slug,
        UPPER(p.team_abbr) AS team_abbr,
        COUNT(*) FILTER (WHERE p.event_type ILIKE '%three%'
                            OR p.event_type ILIKE '%3pt%'
                            OR p.event_type ILIKE '%3P%')          AS three_pt_plays,
        COUNT(*) FILTER (WHERE p.event_type ILIKE '%two%'
                            OR p.event_type ILIKE '%2pt%'
                            OR p.event_type ILIKE '%2P%')          AS two_pt_plays,
        COUNT(*) FILTER (WHERE p.event_type ILIKE '%foul%')        AS foul_plays,
        COUNT(*) FILTER (WHERE p.event_type ILIKE '%turnover%'
                            OR p.event_type ILIKE '%tov%')         AS turnover_plays,
        COUNT(*) FILTER (WHERE p.event_type ILIKE '%timeout%')     AS timeout_plays,
        COUNT(*) FILTER (WHERE p.event_type ILIKE '%rebound%'
                            OR p.event_type ILIKE '%reb%')         AS rebound_plays,
        COUNT(*)                                                    AS total_plays
    FROM raw.nba_pbp_plays p
    WHERE p.team_abbr IS NOT NULL
    GROUP BY p.season, p.game_slug, UPPER(p.team_abbr)
),

with_rates AS (
    SELECT
        gp.*,
        sp.game_date_et,
        sp.start_ts_utc,
        CASE WHEN total_plays > 0 THEN three_pt_plays::numeric / total_plays ELSE NULL END  AS three_pt_rate,
        CASE WHEN total_plays > 0 THEN foul_plays::numeric / total_plays     ELSE NULL END  AS foul_rate,
        CASE WHEN total_plays > 0 THEN turnover_plays::numeric / total_plays ELSE NULL END  AS turnover_rate,
        timeout_plays::numeric                                                               AS timeouts,
        COALESCE(sp.start_ts_utc, sp.game_date_et::timestamp with time zone)                AS order_ts
    FROM game_plays gp
    JOIN features.team_game_spine sp
      ON sp.season = gp.season
     AND sp.game_slug = gp.game_slug
     AND sp.team_abbr = gp.team_abbr
)

SELECT
    season,
    game_slug,
    team_abbr,
    game_date_et,

    -- Raw counts (current game, for reference)
    total_plays,

    -- Rolling 10-game averages (pregame)
    AVG(three_pt_rate)   OVER w10 AS three_pt_rate_avg_10,
    AVG(foul_rate)       OVER w10 AS foul_rate_avg_10,
    AVG(turnover_rate)   OVER w10 AS turnover_rate_avg_10,
    AVG(timeouts)        OVER w10 AS timeouts_avg_10,
    AVG(total_plays)     OVER w10 AS total_plays_avg_10,

    -- Rolling 5-game averages (pregame)
    AVG(three_pt_rate)   OVER w5  AS three_pt_rate_avg_5,
    AVG(foul_rate)       OVER w5  AS foul_rate_avg_5,
    AVG(turnover_rate)   OVER w5  AS turnover_rate_avg_5

FROM with_rates

WINDOW
    w5  AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);


-- ============================================================================
-- VIEW 2: Team Quarter Scoring — quarter-by-quarter scoring tendencies
-- ============================================================================
CREATE OR REPLACE VIEW features.team_quarter_scoring AS

WITH quarter_scores AS (
    -- Get the max running score per period for each team (home/away)
    -- Then subtract previous period's max to get per-quarter points
    SELECT
        p.season,
        p.game_slug,
        g.home_team_abbr,
        g.away_team_abbr,
        p.period,
        MAX(p.points_home) AS home_running_max,
        MAX(p.points_away) AS away_running_max
    FROM raw.nba_pbp_plays p
    JOIN raw.nba_games g ON g.game_slug = p.game_slug AND g.season = p.season
    WHERE p.period BETWEEN 1 AND 4  -- regulation only
      AND (p.points_home IS NOT NULL OR p.points_away IS NOT NULL)
    GROUP BY p.season, p.game_slug, g.home_team_abbr, g.away_team_abbr, p.period
),

-- Pivot to get per-quarter points
game_quarters AS (
    SELECT
        qs.season,
        qs.game_slug,
        qs.home_team_abbr,
        qs.away_team_abbr,
        qs.period,
        qs.home_running_max - COALESCE(
            LAG(qs.home_running_max) OVER (
                PARTITION BY qs.season, qs.game_slug
                ORDER BY qs.period
            ), 0
        ) AS home_q_pts,
        qs.away_running_max - COALESCE(
            LAG(qs.away_running_max) OVER (
                PARTITION BY qs.season, qs.game_slug
                ORDER BY qs.period
            ), 0
        ) AS away_q_pts
    FROM quarter_scores qs
),

-- Pivot into one row per game-team
team_quarter_flat AS (
    -- Home team rows
    SELECT
        gq.season,
        gq.game_slug,
        gq.home_team_abbr AS team_abbr,
        MAX(CASE WHEN gq.period = 1 THEN gq.home_q_pts END) AS q1_pts,
        MAX(CASE WHEN gq.period = 2 THEN gq.home_q_pts END) AS q2_pts,
        MAX(CASE WHEN gq.period = 3 THEN gq.home_q_pts END) AS q3_pts,
        MAX(CASE WHEN gq.period = 4 THEN gq.home_q_pts END) AS q4_pts
    FROM game_quarters gq
    GROUP BY gq.season, gq.game_slug, gq.home_team_abbr

    UNION ALL

    -- Away team rows
    SELECT
        gq.season,
        gq.game_slug,
        gq.away_team_abbr AS team_abbr,
        MAX(CASE WHEN gq.period = 1 THEN gq.away_q_pts END) AS q1_pts,
        MAX(CASE WHEN gq.period = 2 THEN gq.away_q_pts END) AS q2_pts,
        MAX(CASE WHEN gq.period = 3 THEN gq.away_q_pts END) AS q3_pts,
        MAX(CASE WHEN gq.period = 4 THEN gq.away_q_pts END) AS q4_pts
    FROM game_quarters gq
    GROUP BY gq.season, gq.game_slug, gq.away_team_abbr
),

-- Check for overtime
ot_check AS (
    SELECT season, game_slug,
           MAX(period) AS max_period,
           GREATEST(MAX(period) - 4, 0) AS ot_periods
    FROM raw.nba_pbp_plays
    GROUP BY season, game_slug
),

with_totals AS (
    SELECT
        tqf.*,
        sp.game_date_et,
        COALESCE(sp.start_ts_utc, sp.game_date_et::timestamp with time zone) AS order_ts,
        COALESCE(tqf.q1_pts, 0) + COALESCE(tqf.q2_pts, 0)
          + COALESCE(tqf.q3_pts, 0) + COALESCE(tqf.q4_pts, 0) AS total_reg_pts,
        -- Quarter percentages
        CASE WHEN COALESCE(tqf.q1_pts, 0) + COALESCE(tqf.q2_pts, 0)
                 + COALESCE(tqf.q3_pts, 0) + COALESCE(tqf.q4_pts, 0) > 0
             THEN tqf.q1_pts::numeric / NULLIF(
                    COALESCE(tqf.q1_pts, 0) + COALESCE(tqf.q2_pts, 0)
                  + COALESCE(tqf.q3_pts, 0) + COALESCE(tqf.q4_pts, 0), 0)
             ELSE NULL END AS q1_pct,
        CASE WHEN COALESCE(tqf.q1_pts, 0) + COALESCE(tqf.q2_pts, 0)
                 + COALESCE(tqf.q3_pts, 0) + COALESCE(tqf.q4_pts, 0) > 0
             THEN tqf.q4_pts::numeric / NULLIF(
                    COALESCE(tqf.q1_pts, 0) + COALESCE(tqf.q2_pts, 0)
                  + COALESCE(tqf.q3_pts, 0) + COALESCE(tqf.q4_pts, 0), 0)
             ELSE NULL END AS q4_pct,
        COALESCE(ot.ot_periods, 0) AS ot_periods
    FROM team_quarter_flat tqf
    JOIN features.team_game_spine sp
      ON sp.season = tqf.season AND sp.game_slug = tqf.game_slug AND sp.team_abbr = tqf.team_abbr
    LEFT JOIN ot_check ot
      ON ot.season = tqf.season AND ot.game_slug = tqf.game_slug
)

SELECT
    season,
    game_slug,
    team_abbr,
    game_date_et,

    -- Rolling 10-game quarter averages (pregame)
    AVG(q1_pts)          OVER w10 AS q1_pts_avg_10,
    AVG(q2_pts)          OVER w10 AS q2_pts_avg_10,
    AVG(q3_pts)          OVER w10 AS q3_pts_avg_10,
    AVG(q4_pts)          OVER w10 AS q4_pts_avg_10,

    -- Quarter distribution tendencies
    AVG(q1_pct)          OVER w10 AS q1_pct_avg_10,
    AVG(q4_pct)          OVER w10 AS q4_pct_avg_10,

    -- First half vs second half scoring pct
    AVG(CASE WHEN total_reg_pts > 0
         THEN (COALESCE(q1_pts, 0) + COALESCE(q2_pts, 0))::numeric / total_reg_pts
         ELSE NULL END)  OVER w10 AS first_half_pct_avg_10,

    AVG(CASE WHEN total_reg_pts > 0
         THEN (COALESCE(q3_pts, 0) + COALESCE(q4_pts, 0))::numeric / total_reg_pts
         ELSE NULL END)  OVER w10 AS second_half_pct_avg_10,

    -- Overtime tendency (count of OT games in last 10)
    SUM(CASE WHEN ot_periods > 0 THEN 1 ELSE 0 END) OVER w10 AS ot_game_count_10,

    -- Slow starter / strong closer flags (rolling)
    AVG(CASE WHEN q1_pct < 0.22 THEN 1.0 ELSE 0.0 END) OVER w10 AS slow_starter_pct_10,
    AVG(CASE WHEN q4_pct > 0.28 THEN 1.0 ELSE 0.0 END) OVER w10 AS strong_closer_pct_10

FROM with_totals

WINDOW
    w10 AS (PARTITION BY season, team_abbr ORDER BY order_ts, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);
