-- MLB003: Individual starting pitcher rolling stats (last 5 starts)
-- Leakage-safe: all windows use ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
-- FIP approximation: (13*HR + 3*BB - 2*K) / IP + 3.2  (no HBP in raw data)
CREATE OR REPLACE VIEW features.mlb_pitcher_rolling AS
WITH starter_gamelogs AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.player_id,
        gl.team_abbr,
        COALESCE(gl.innings_pitched, 0)    AS ip,
        COALESCE(gl.earned_runs, 0)        AS er,
        COALESCE(gl.strikeouts_pitcher, 0) AS k,
        COALESCE(gl.walks_allowed, 0)      AS bb,
        COALESCE(gl.hits_allowed, 0)       AS h,
        COALESCE(gl.home_runs_allowed, 0)  AS hr
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = g.game_slug
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
      AND gl.is_starter = TRUE
      AND gl.innings_pitched IS NOT NULL
      AND gl.innings_pitched > 0
),
derived AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        player_id,
        team_abbr,
        ip,
        er,
        k,
        bb,
        h,
        hr,
        -- ERA for this start: 9 * ER / IP
        CASE WHEN ip > 0
             THEN 9.0 * er / ip
             ELSE NULL END AS start_era,
        -- WHIP for this start: (H + BB) / IP
        CASE WHEN ip > 0
             THEN (h + bb)::float / ip
             ELSE NULL END AS start_whip,
        -- K% per batter faced approx: K / (IP * 3 + H + BB)  [batters faced estimate]
        CASE WHEN (ip * 3 + h + bb) > 0
             THEN k::float / (ip * 3 + h + bb)
             ELSE NULL END AS start_k_pct,
        -- BB% per batter faced approx
        CASE WHEN (ip * 3 + h + bb) > 0
             THEN bb::float / (ip * 3 + h + bb)
             ELSE NULL END AS start_bb_pct,
        -- FIP (no HBP): (13*HR + 3*BB - 2*K) / IP + 3.2
        CASE WHEN ip > 0
             THEN (13.0 * hr + 3.0 * bb - 2.0 * k) / ip + 3.2
             ELSE NULL END AS start_fip,
        -- K/9
        CASE WHEN ip > 0
             THEN 9.0 * k / ip
             ELSE NULL END AS start_k9,
        -- BB/9
        CASE WHEN ip > 0
             THEN 9.0 * bb / ip
             ELSE NULL END AS start_bb9,
        -- HR/9
        CASE WHEN ip > 0
             THEN 9.0 * hr / ip
             ELSE NULL END AS start_hr9
    FROM starter_gamelogs
)
SELECT
    season,
    player_id,
    team_abbr,
    game_slug,
    game_date_et,

    -- Raw last-start values (most recent prior start)
    FIRST_VALUE(ip)          OVER w1 AS last_start_ip,
    FIRST_VALUE(start_era)   OVER w1 AS last_start_era,
    FIRST_VALUE(k)           OVER w1 AS last_start_k,
    FIRST_VALUE(start_fip)   OVER w1 AS last_start_fip,

    -- 5-start rolling averages (last 5 starts, leakage-safe)
    AVG(ip)            OVER w5 AS ip_avg_5,
    AVG(er)            OVER w5 AS er_avg_5,
    AVG(start_era)     OVER w5 AS era_5,
    AVG(start_whip)    OVER w5 AS whip_5,
    AVG(start_k_pct)   OVER w5 AS k_pct_5,
    AVG(start_bb_pct)  OVER w5 AS bb_pct_5,
    AVG(start_fip)     OVER w5 AS fip_5,
    AVG(start_k9)      OVER w5 AS k9_5,
    AVG(start_bb9)     OVER w5 AS bb9_5,
    AVG(start_hr9)     OVER w5 AS hr9_5,

    -- 10-start rolling averages
    AVG(start_era)             OVER w10 AS era_10,
    AVG(start_whip)            OVER w10 AS whip_10,
    AVG(start_k_pct)           OVER w10 AS k_pct_10,
    AVG(start_bb_pct)          OVER w10 AS bb_pct_10,
    AVG(start_fip)             OVER w10 AS fip_10,
    AVG(start_k9)              OVER w10 AS k9_10,
    AVG(start_bb9)             OVER w10 AS bb9_10,
    AVG(start_hr9)             OVER w10 AS hr9_10,
    STDDEV_SAMP(start_era)     OVER w10 AS era_sd_10,
    STDDEV_SAMP(start_fip)     OVER w10 AS fip_sd_10,

    -- Career/season rolling (20 starts) for baseline
    AVG(start_era)   OVER w20 AS era_20,
    AVG(start_fip)   OVER w20 AS fip_20,
    AVG(ip)          OVER w20 AS ip_avg_20,

    -- Count of starts in last 5 (to know how many samples back the avg)
    COUNT(*) OVER w5  AS starts_in_window_5,
    COUNT(*) OVER w10 AS starts_in_window_10

FROM derived
WINDOW
    -- Last 1 preceding start (for most-recent start values)
    w1  AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING),
    -- Last 5 starts
    w5  AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    -- Last 10 starts
    w10 AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    -- Last 20 starts
    w20 AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
;
