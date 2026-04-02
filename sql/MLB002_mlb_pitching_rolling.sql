-- MLB002: Team pitching rolling stats (starter + bullpen splits)
-- Leakage-safe: all windows use ROWS BETWEEN N PRECEDING AND 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_team_pitching_rolling AS
WITH game_pitching AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        -- Starter aggregates
        SUM(CASE WHEN gl.is_starter THEN COALESCE(gl.innings_pitched, 0) ELSE 0 END)        AS sp_ip,
        SUM(CASE WHEN gl.is_starter THEN COALESCE(gl.earned_runs, 0) ELSE 0 END)            AS sp_er,
        SUM(CASE WHEN gl.is_starter THEN COALESCE(gl.strikeouts_pitcher, 0) ELSE 0 END)     AS sp_k,
        SUM(CASE WHEN gl.is_starter THEN COALESCE(gl.walks_allowed, 0) ELSE 0 END)          AS sp_bb,
        SUM(CASE WHEN gl.is_starter THEN COALESCE(gl.hits_allowed, 0) ELSE 0 END)           AS sp_h,
        SUM(CASE WHEN gl.is_starter THEN COALESCE(gl.home_runs_allowed, 0) ELSE 0 END)      AS sp_hr,
        -- Bullpen aggregates
        SUM(CASE WHEN NOT gl.is_starter THEN COALESCE(gl.innings_pitched, 0) ELSE 0 END)    AS bp_ip,
        SUM(CASE WHEN NOT gl.is_starter THEN COALESCE(gl.earned_runs, 0) ELSE 0 END)        AS bp_er,
        SUM(CASE WHEN NOT gl.is_starter THEN COALESCE(gl.strikeouts_pitcher, 0) ELSE 0 END) AS bp_k,
        SUM(CASE WHEN NOT gl.is_starter THEN COALESCE(gl.walks_allowed, 0) ELSE 0 END)      AS bp_bb,
        SUM(CASE WHEN NOT gl.is_starter THEN COALESCE(gl.hits_allowed, 0) ELSE 0 END)       AS bp_h,
        SUM(CASE WHEN NOT gl.is_starter THEN COALESCE(gl.home_runs_allowed, 0) ELSE 0 END)  AS bp_hr,
        -- Total team pitching
        SUM(COALESCE(gl.innings_pitched, 0))   AS total_ip,
        SUM(COALESCE(gl.earned_runs, 0))       AS total_er,
        SUM(COALESCE(gl.strikeouts_pitcher, 0)) AS total_k,
        SUM(COALESCE(gl.walks_allowed, 0))     AS total_bb,
        SUM(COALESCE(gl.hits_allowed, 0))      AS total_h,
        SUM(COALESCE(gl.home_runs_allowed, 0)) AS total_hr,
        -- Runs allowed by this team's pitching staff = opponent's runs scored
        CASE
            WHEN gl.team_abbr = g.home_team_abbr THEN g.away_score
            ELSE g.home_score
        END AS runs_allowed
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = g.game_slug
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
      AND gl.innings_pitched IS NOT NULL
    GROUP BY
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        g.home_team_abbr,
        g.home_score,
        g.away_score
),
derived AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        runs_allowed,
        sp_ip,
        sp_er,
        sp_k,
        sp_bb,
        sp_h,
        sp_hr,
        bp_ip,
        bp_er,
        bp_k,
        bp_bb,
        bp_h,
        bp_hr,
        total_ip,
        total_er,
        total_k,
        total_bb,
        total_h,
        total_hr,
        -- Team ERA (9 * ER / IP)
        CASE WHEN total_ip > 0
             THEN 9.0 * total_er / total_ip
             ELSE NULL END AS era,
        -- Team WHIP ((H + BB) / IP)
        CASE WHEN total_ip > 0
             THEN (total_h + total_bb)::float / total_ip
             ELSE NULL END AS whip,
        -- K/9
        CASE WHEN total_ip > 0
             THEN 9.0 * total_k / total_ip
             ELSE NULL END AS k9,
        -- BB/9
        CASE WHEN total_ip > 0
             THEN 9.0 * total_bb / total_ip
             ELSE NULL END AS bb9,
        -- HR/9
        CASE WHEN total_ip > 0
             THEN 9.0 * total_hr / total_ip
             ELSE NULL END AS hr9,
        -- Starter ERA
        CASE WHEN sp_ip > 0
             THEN 9.0 * sp_er / sp_ip
             ELSE NULL END AS sp_era,
        -- Starter WHIP
        CASE WHEN sp_ip > 0
             THEN (sp_h + sp_bb)::float / sp_ip
             ELSE NULL END AS sp_whip,
        -- Starter K/9
        CASE WHEN sp_ip > 0
             THEN 9.0 * sp_k / sp_ip
             ELSE NULL END AS sp_k9,
        -- Bullpen ERA
        CASE WHEN bp_ip > 0
             THEN 9.0 * bp_er / bp_ip
             ELSE NULL END AS bp_era,
        -- Bullpen WHIP
        CASE WHEN bp_ip > 0
             THEN (bp_h + bp_bb)::float / bp_ip
             ELSE NULL END AS bp_whip
    FROM game_pitching
),
-- Bullpen fatigue: sum of bullpen IP in last 3 calendar days
bullpen_fatigue AS (
    SELECT
        gp.season,
        gp.game_slug,
        gp.game_date_et,
        gp.team_abbr,
        -- Sum of bullpen IP for games in [game_date_et - 3 days, game_date_et - 1 day]
        COALESCE(SUM(prior.bp_ip), 0) AS bullpen_ip_last_3
    FROM game_pitching gp
    LEFT JOIN game_pitching prior
        ON prior.team_abbr   = gp.team_abbr
       AND prior.season      = gp.season
       AND prior.game_date_et >= gp.game_date_et - INTERVAL '3 days'
       AND prior.game_date_et <  gp.game_date_et
    GROUP BY
        gp.season,
        gp.game_slug,
        gp.game_date_et,
        gp.team_abbr
),
-- 7-day bullpen fatigue: captures full-week workload (3-day window misses games Mon/Tue
-- when predicting weekend series).  Also computes rolling 7-day bullpen ERA.
bullpen_7d AS (
    SELECT
        gp.season,
        gp.game_slug,
        gp.game_date_et,
        gp.team_abbr,
        COALESCE(SUM(prior.bp_ip), 0)           AS bullpen_ip_last_7,
        CASE WHEN SUM(prior.bp_ip) > 0
             THEN 9.0 * SUM(prior.bp_er) / SUM(prior.bp_ip)
             ELSE NULL END                       AS bp_era_7d
    FROM game_pitching gp
    LEFT JOIN game_pitching prior
        ON prior.team_abbr   = gp.team_abbr
       AND prior.season      = gp.season
       AND prior.game_date_et >= gp.game_date_et - INTERVAL '7 days'
       AND prior.game_date_et <  gp.game_date_et
    GROUP BY
        gp.season,
        gp.game_slug,
        gp.game_date_et,
        gp.team_abbr
),
-- SP short outing flag: did this team's SP get knocked out early (<4 IP) in their
-- most recent game?  Signals the bullpen was overused last outing and may be depleted.
sp_last_outing AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        LAG(sp_ip) OVER (
            PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
        ) AS prev_sp_ip
    FROM game_pitching
)
SELECT
    d.season,
    d.team_abbr,
    d.game_slug,
    d.game_date_et,

    -- 5-game rolling windows
    AVG(d.runs_allowed) OVER w5  AS runs_allowed_avg_5,
    AVG(d.era)          OVER w5  AS era_5,
    AVG(d.whip)         OVER w5  AS whip_5,
    AVG(d.k9)           OVER w5  AS k9_5,
    AVG(d.bb9)          OVER w5  AS bb9_5,
    AVG(d.hr9)          OVER w5  AS hr9_5,
    AVG(d.sp_era)       OVER w5  AS sp_era_5,
    AVG(d.sp_whip)      OVER w5  AS sp_whip_5,
    AVG(d.sp_k9)        OVER w5  AS sp_k9_5,
    AVG(d.bp_era)       OVER w5  AS bp_era_5,
    AVG(d.bp_whip)      OVER w5  AS bp_whip_5,

    -- 10-game rolling windows
    AVG(d.runs_allowed)          OVER w10 AS runs_allowed_avg_10,
    AVG(d.era)                   OVER w10 AS era_10,
    AVG(d.whip)                  OVER w10 AS whip_10,
    AVG(d.k9)                    OVER w10 AS k9_10,
    AVG(d.bb9)                   OVER w10 AS bb9_10,
    AVG(d.hr9)                   OVER w10 AS hr9_10,
    AVG(d.sp_era)                OVER w10 AS sp_era_10,
    AVG(d.sp_whip)               OVER w10 AS sp_whip_10,
    AVG(d.sp_k9)                 OVER w10 AS sp_k9_10,
    AVG(d.bp_era)                OVER w10 AS bp_era_10,
    AVG(d.bp_whip)               OVER w10 AS bp_whip_10,
    STDDEV_SAMP(d.runs_allowed)  OVER w10 AS runs_allowed_sd_10,

    -- Bullpen fatigue (calendar-day based, not rolling window)
    bf.bullpen_ip_last_3,

    -- 7-day bullpen fatigue + recent ERA
    b7d.bullpen_ip_last_7,
    b7d.bp_era_7d,

    -- SP short outing: 1 if last start was < 4 IP (signals depleted bullpen)
    CASE WHEN slo.prev_sp_ip IS NOT NULL AND slo.prev_sp_ip < 4.0
         THEN 1 ELSE 0 END AS sp_short_last

FROM derived d
JOIN bullpen_fatigue bf
    ON bf.game_slug  = d.game_slug
   AND bf.team_abbr  = d.team_abbr
JOIN bullpen_7d b7d
    ON b7d.game_slug = d.game_slug
   AND b7d.team_abbr = d.team_abbr
JOIN sp_last_outing slo
    ON slo.game_slug = d.game_slug
   AND slo.team_abbr = d.team_abbr
WINDOW
    w5  AS (PARTITION BY d.season, d.team_abbr
            ORDER BY d.game_date_et, d.game_slug
            ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY d.season, d.team_abbr
            ORDER BY d.game_date_et, d.game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
;
