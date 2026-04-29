-- MLB019: Individual reliever rest / workload rolling view
-- Counts distinct relievers used per 1/2/3-day windows prior to each game.
-- This captures bullpen depth depletion better than total IP alone:
-- a team that used 5 relievers yesterday is in a worse spot than one that
-- threw a single mop-up arm, even if total IP is identical.
--
-- Source: raw.mlb_player_gamelogs where is_starter = FALSE AND innings_pitched > 0
-- Grain: one row per (game_slug, team_abbr) — only final games appear as anchors.

CREATE OR REPLACE VIEW features.mlb_reliever_rolling AS
WITH reliever_season_era AS (
    SELECT
        player_id,
        game_slug,
        SUM(COALESCE(earned_runs, 0)) OVER w * 9.0
            / NULLIF(SUM(COALESCE(innings_pitched, 0)) OVER w, 0) AS ytd_era
    FROM raw.mlb_player_gamelogs
    WHERE is_starter = FALSE
      AND COALESCE(innings_pitched, 0) > 0
    WINDOW w AS (
        PARTITION BY player_id,
            EXTRACT(YEAR FROM game_date_et)::INT
        ORDER BY game_date_et
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
reliever_apps AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        gl.player_id,
        COALESCE(gl.innings_pitched, 0) AS ip,
        rse.ytd_era                      AS era
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl ON gl.game_slug = g.game_slug
    LEFT JOIN reliever_season_era rse
           ON rse.player_id = gl.player_id
          AND rse.game_slug = gl.game_slug
    WHERE g.status = 'final'
      AND gl.is_starter = FALSE
      AND COALESCE(gl.innings_pitched, 0) > 0
),
anchor AS (
    SELECT DISTINCT season, game_slug, game_date_et, team_abbr
    FROM reliever_apps
)
SELECT
    a.game_slug,
    a.game_date_et,
    a.season,
    a.team_abbr,

    COUNT(DISTINCT CASE
        WHEN prior.game_date_et >= a.game_date_et - INTERVAL '1 day'
         AND prior.game_date_et <  a.game_date_et
        THEN prior.player_id END)                           AS bp_relievers_last_1d,

    COUNT(DISTINCT CASE
        WHEN prior.game_date_et >= a.game_date_et - INTERVAL '2 days'
         AND prior.game_date_et <  a.game_date_et
        THEN prior.player_id END)                           AS bp_relievers_last_2d,

    COUNT(DISTINCT CASE
        WHEN prior.game_date_et >= a.game_date_et - INTERVAL '3 days'
         AND prior.game_date_et <  a.game_date_et
        THEN prior.player_id END)                           AS bp_relievers_last_3d,

    COALESCE(SUM(CASE
        WHEN prior.game_date_et >= a.game_date_et - INTERVAL '1 day'
         AND prior.game_date_et <  a.game_date_et
        THEN prior.ip END), 0)                              AS bp_ip_last_1d,

    AVG(CASE
        WHEN prior.game_date_et >= a.game_date_et - INTERVAL '3 days'
         AND prior.game_date_et <  a.game_date_et
        THEN prior.era END)                                 AS bp_avg_era_last_3d

FROM anchor a
LEFT JOIN reliever_apps prior
    ON prior.team_abbr    = a.team_abbr
   AND prior.season       = a.season
   AND prior.game_date_et >= a.game_date_et - INTERVAL '3 days'
   AND prior.game_date_et <  a.game_date_et
GROUP BY a.game_slug, a.game_date_et, a.season, a.team_abbr;
