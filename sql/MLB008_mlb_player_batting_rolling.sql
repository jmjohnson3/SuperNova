-- MLB008: Individual batter rolling stats
-- Leakage-safe: all rolling windows use ROWS BETWEEN N PRECEDING AND 1 PRECEDING
-- Partitioned by (season, player_id) to avoid cross-season bleeding.
-- Windows: 5, 10, 20 games.
CREATE OR REPLACE VIEW features.mlb_player_batting_rolling AS
WITH batter_gamelogs AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        g.start_ts_utc,
        gl.player_id,
        gl.team_abbr,
        CASE WHEN gl.team_abbr = g.home_team_abbr
             THEN g.away_team_abbr
             ELSE g.home_team_abbr END           AS opponent_abbr,
        CASE WHEN gl.team_abbr = g.home_team_abbr
             THEN TRUE ELSE FALSE END             AS is_home,
        COALESCE(gl.at_bats, 0)                  AS ab,
        COALESCE(gl.hits, 0)                     AS h,
        COALESCE(gl.doubles, 0)                  AS d2,
        COALESCE(gl.triples, 0)                  AS d3,
        COALESCE(gl.home_runs, 0)                AS hr,
        COALESCE(gl.total_bases, 0)              AS tb,
        COALESCE(gl.walks_batter, 0)             AS bb,
        COALESCE(gl.strikeouts_batter, 0)        AS k_bat,
        COALESCE(bps.batting_order, 5)           AS batting_order
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl ON gl.game_slug = g.game_slug
    LEFT JOIN raw.mlb_boxscore_player_stats bps
        ON bps.game_slug = gl.game_slug
        AND bps.player_id = gl.player_id
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
      AND gl.at_bats IS NOT NULL
      AND gl.at_bats > 0
),
derived AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        start_ts_utc,
        player_id,
        team_abbr,
        opponent_abbr,
        is_home,
        ab,
        h,
        hr,
        tb,
        bb,
        k_bat,
        batting_order,
        -- Per-game rate stats
        CASE WHEN ab > 0
             THEN h::float / ab
             ELSE NULL END                         AS game_avg,
        -- ISO: (TB - H) / AB
        CASE WHEN ab > 0
             THEN (tb - h)::float / ab
             ELSE NULL END                         AS game_iso,
        -- K rate (K / AB)
        CASE WHEN ab > 0
             THEN k_bat::float / ab
             ELSE NULL END                         AS game_k_rate,
        -- BB rate (BB / plate appearances ≈ AB + BB)
        CASE WHEN (ab + bb) > 0
             THEN bb::float / (ab + bb)
             ELSE NULL END                         AS game_bb_rate,
        -- HR rate (HR / AB)
        CASE WHEN ab > 0
             THEN hr::float / ab
             ELSE NULL END                         AS game_hr_rate,
        -- Extra-base hit weight: TB minus H (0=single, 1=double, 2=triple, 3=HR)
        -- Captures XBH production on a continuous scale without needing raw 2B/3B counts
        (tb - h)                                   AS xbh_weight,
        -- Big-game flag: 1 when player achieves 2+ TB (direct proxy for OVER 1.5 line)
        CASE WHEN tb >= 2 THEN 1.0 ELSE 0.0 END    AS is_big_tb_game
    FROM batter_gamelogs
)
SELECT
    season,
    player_id,
    team_abbr,
    opponent_abbr,
    game_slug,
    game_date_et,
    start_ts_utc,
    is_home,

    -- Sample size gate: games in last 10 window (leakage-safe count)
    COUNT(*)         OVER w10                        AS n_games_prev_10,

    -- Hits rolling
    AVG(h)           OVER w5                         AS hits_avg_5,
    AVG(h)           OVER w10                        AS hits_avg_10,
    AVG(h)           OVER w20                        AS hits_avg_20,
    STDDEV_SAMP(h)   OVER w10                        AS hits_sd_10,

    -- Home runs rolling
    AVG(hr)          OVER w5                         AS hr_avg_5,
    AVG(hr)          OVER w10                        AS hr_avg_10,
    AVG(hr)          OVER w20                        AS hr_avg_20,
    STDDEV_SAMP(hr)  OVER w10                        AS hr_sd_10,

    -- Total bases rolling
    AVG(tb)          OVER w5                         AS tb_avg_5,
    AVG(tb)          OVER w10                        AS tb_avg_10,
    AVG(tb)          OVER w20                        AS tb_avg_20,
    STDDEV_SAMP(tb)  OVER w10                        AS tb_sd_10,

    -- At bats rolling (plate appearance proxy)
    AVG(ab)          OVER w5                         AS ab_avg_5,
    AVG(ab)          OVER w10                        AS ab_avg_10,

    -- Rate stats rolling (10-game)
    AVG(game_avg)    OVER w10                        AS avg_avg_10,
    AVG(game_k_rate) OVER w10                        AS k_rate_avg_10,
    AVG(game_bb_rate)OVER w10                        AS bb_rate_avg_10,
    AVG(game_iso)    OVER w10                        AS iso_avg_10,

    -- HR rate rolling
    AVG(game_hr_rate)OVER w5                         AS hr_rate_avg_5,
    AVG(game_hr_rate)OVER w10                        AS hr_rate_avg_10,

    -- Rest days: days since previous game for this player (same season)
    EXTRACT(EPOCH FROM (
        start_ts_utc
        - LAG(start_ts_utc) OVER (
            PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
          )
    )) / 86400.0                                     AS rest_days,

    -- Lineup slot rolling averages
    AVG(batting_order) OVER w5                       AS batting_order_avg_5,
    AVG(batting_order) OVER w10                      AS batting_order_avg_10,

    -- Absolute walk counts rolling (for walks prop model — raw scale signal)
    AVG(bb)          OVER w5                         AS bb_avg_5,
    AVG(bb)          OVER w10                        AS bb_avg_10,
    AVG(bb)          OVER w20                        AS bb_avg_20,
    STDDEV_SAMP(bb)  OVER w10                        AS bb_sd_10,

    -- Absolute batter strikeout counts rolling (K tendency)
    AVG(k_bat)       OVER w5                         AS k_avg_5,
    AVG(k_bat)       OVER w10                        AS k_avg_10,

    -- Home/away conditional rolling (last 20 games, FILTER by venue flag)
    -- Follows same pattern as MLB003 home/away ERA splits
    AVG(h)  FILTER (WHERE is_home = TRUE)  OVER w20  AS hits_home_avg_20,
    AVG(h)  FILTER (WHERE is_home = FALSE) OVER w20  AS hits_away_avg_20,
    AVG(tb) FILTER (WHERE is_home = TRUE)  OVER w20  AS tb_home_avg_20,
    AVG(tb) FILTER (WHERE is_home = FALSE) OVER w20  AS tb_away_avg_20,
    AVG(hr) FILTER (WHERE is_home = TRUE)  OVER w20  AS hr_home_avg_20,
    AVG(hr) FILTER (WHERE is_home = FALSE) OVER w20  AS hr_away_avg_20,
    AVG(bb) FILTER (WHERE is_home = TRUE)  OVER w20  AS bb_home_avg_20,
    AVG(bb) FILTER (WHERE is_home = FALSE) OVER w20  AS bb_away_avg_20,

    -- TB hot streak / ceiling / big-game frequency
    -- Appended at end per PostgreSQL CREATE OR REPLACE VIEW constraint
    AVG(tb)             OVER w3   AS tb_avg_3,
    MAX(tb)             OVER w10  AS tb_max_10,
    AVG(is_big_tb_game) OVER w10  AS tb_over2_rate_10,
    AVG(xbh_weight)     OVER w10  AS tb_xbh_weight_avg_10,

    -- HR volatility signals (appended; new columns must go at end for CREATE OR REPLACE VIEW)
    STDDEV_SAMP(hr)  OVER w5                         AS hr_sd_5,
    MAX(hr)          OVER w10                        AS hr_max_10

FROM derived
WINDOW
    w3  AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 3  PRECEDING AND 1 PRECEDING),
    w5  AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    w20 AS (PARTITION BY season, player_id
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
;
