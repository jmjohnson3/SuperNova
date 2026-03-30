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
        COALESCE(gl.strikeouts_batter, 0)        AS k_bat
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl ON gl.game_slug = g.game_slug
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
             ELSE NULL END                         AS game_hr_rate
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
    )) / 86400.0                                     AS rest_days

FROM derived
WINDOW
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
