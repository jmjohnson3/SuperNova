-- ============================================================================
-- V012: Attendance, Overtime & Game Duration Features
-- ============================================================================
-- Uses previously unused columns from raw.nba_boxscore_games:
--   • attendance (int)
--   • ended_ts_utc (timestamptz) — for game duration / overtime detection
-- Also cross-references PBP for overtime period counts.
-- ============================================================================

CREATE OR REPLACE VIEW features.game_context_features AS

WITH game_base AS (
    SELECT
        bg.game_slug,
        bg.season,
        g.game_date_et,
        g.start_ts_utc,
        UPPER(g.home_team_abbr)  AS home_team_abbr,
        UPPER(g.away_team_abbr)  AS away_team_abbr,
        bg.attendance,
        bg.start_ts_utc  AS box_start,
        bg.ended_ts_utc  AS box_end,
        -- Game duration in minutes
        CASE WHEN bg.ended_ts_utc IS NOT NULL AND bg.start_ts_utc IS NOT NULL
             THEN EXTRACT(EPOCH FROM (bg.ended_ts_utc - bg.start_ts_utc)) / 60.0
             ELSE NULL
        END AS game_duration_min
    FROM raw.nba_boxscore_games bg
    JOIN raw.nba_games g
      ON g.game_slug = bg.game_slug AND g.season = bg.season
),

-- Overtime from PBP
ot_from_pbp AS (
    SELECT
        season,
        game_slug,
        GREATEST(MAX(period) - 4, 0) AS ot_periods
    FROM raw.nba_pbp_plays
    GROUP BY season, game_slug
),

-- Venue capacity for attendance % (try to extract from JSONB)
venue_cap AS (
    SELECT DISTINCT ON (UPPER(home_team_abbr))
        UPPER(home_team_abbr) AS team_abbr,
        COALESCE(
            NULLIF(capacities->>'maxCapacity', '')::int,
            NULLIF(capacities->>'capacity', '')::int,
            NULLIF(capacities->>'basketball', '')::int,
            19000  -- NBA default if not found
        ) AS capacity
    FROM raw.nba_venues
    WHERE home_team_abbr IS NOT NULL
    ORDER BY UPPER(home_team_abbr), updated_at_utc DESC
),

enriched AS (
    SELECT
        gb.*,
        COALESCE(ot.ot_periods, 0) AS ot_periods,
        COALESCE(ot.ot_periods, 0) > 0
            OR gb.game_duration_min > 150  -- 2.5 hrs = likely OT
            AS is_overtime,
        vc.capacity AS venue_capacity,
        CASE WHEN gb.attendance IS NOT NULL AND vc.capacity IS NOT NULL AND vc.capacity > 0
             THEN gb.attendance::numeric / vc.capacity
             ELSE NULL
        END AS attendance_pct_capacity
    FROM game_base gb
    LEFT JOIN ot_from_pbp ot
      ON ot.season = gb.season AND ot.game_slug = gb.game_slug
    LEFT JOIN venue_cap vc
      ON vc.team_abbr = gb.home_team_abbr
),

-- Create team-game rows for rolling calculations
team_rows AS (
    -- Home team perspective
    SELECT season, game_slug, game_date_et, start_ts_utc,
           home_team_abbr AS team_abbr,
           attendance, attendance_pct_capacity, game_duration_min,
           is_overtime, ot_periods,
           TRUE AS is_home_row
    FROM enriched
    UNION ALL
    -- Away team perspective
    SELECT season, game_slug, game_date_et, start_ts_utc,
           away_team_abbr AS team_abbr,
           attendance, attendance_pct_capacity, game_duration_min,
           is_overtime, ot_periods,
           FALSE AS is_home_row
    FROM enriched
),

team_rolling AS (
    SELECT
        tr.*,
        COALESCE(tr.start_ts_utc, tr.game_date_et::timestamp with time zone) AS order_ts,
        -- Rolling OT tendency
        SUM(CASE WHEN tr.is_overtime THEN 1 ELSE 0 END) OVER (
            PARTITION BY tr.team_abbr
            ORDER BY COALESCE(tr.start_ts_utc, tr.game_date_et::timestamp with time zone), tr.game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS ot_tendency_10,
        -- Rolling avg game duration
        AVG(tr.game_duration_min) OVER (
            PARTITION BY tr.team_abbr
            ORDER BY COALESCE(tr.start_ts_utc, tr.game_date_et::timestamp with time zone), tr.game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS avg_game_duration_10
    FROM team_rows tr
),

-- Rolling attendance (home games only for this team)
home_attendance_rolling AS (
    SELECT
        tr.game_slug,
        tr.team_abbr,
        AVG(tr.attendance) OVER (
            PARTITION BY tr.team_abbr
            ORDER BY COALESCE(tr.start_ts_utc, tr.game_date_et::timestamp with time zone), tr.game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS home_avg_attendance_10
    FROM team_rows tr
    WHERE tr.is_home_row = TRUE
)

SELECT
    e.season,
    e.game_slug,
    e.game_date_et,
    e.home_team_abbr,
    e.away_team_abbr,

    -- Attendance
    e.attendance,
    ROUND(e.attendance_pct_capacity::numeric, 3) AS attendance_pct_capacity,
    ROUND(har.home_avg_attendance_10::numeric, 0) AS home_avg_attendance_10,
    CASE WHEN har.home_avg_attendance_10 IS NOT NULL AND har.home_avg_attendance_10 > 0
         THEN ROUND(e.attendance::numeric / har.home_avg_attendance_10, 3)
         ELSE NULL
    END AS attendance_vs_avg,

    -- Overtime
    e.is_overtime::int AS is_overtime,
    e.ot_periods,
    htr.ot_tendency_10 AS home_ot_tendency_10,
    atr.ot_tendency_10 AS away_ot_tendency_10,

    -- Game duration
    ROUND(e.game_duration_min::numeric, 1) AS game_duration_min,
    ROUND(htr.avg_game_duration_10::numeric, 1) AS home_avg_game_duration_10,
    ROUND(atr.avg_game_duration_10::numeric, 1) AS away_avg_game_duration_10

FROM enriched e
LEFT JOIN team_rolling htr
  ON htr.game_slug = e.game_slug AND htr.team_abbr = e.home_team_abbr AND htr.is_home_row = TRUE
LEFT JOIN team_rolling atr
  ON atr.game_slug = e.game_slug AND atr.team_abbr = e.away_team_abbr AND atr.is_home_row = FALSE
LEFT JOIN home_attendance_rolling har
  ON har.game_slug = e.game_slug AND har.team_abbr = e.home_team_abbr;
