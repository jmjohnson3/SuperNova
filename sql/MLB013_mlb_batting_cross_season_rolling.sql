-- MLB013: Cross-season batter rolling stats
-- Unlike MLB008 (partitioned by season+player), this partitions by player_id only,
-- so the rolling windows span season boundaries.
-- This gives meaningful features at season start using prior-year data,
-- solving the cold-start problem for early-season predictions.
-- Columns are suffixed _cs (cross-season) to distinguish from MLB008.
CREATE OR REPLACE VIEW features.mlb_player_batting_rolling_cross AS
WITH batter_gamelogs AS (
    SELECT
        g.game_slug,
        g.game_date_et,
        gl.player_id,
        COALESCE(gl.at_bats, 0)              AS ab,
        COALESCE(gl.hits, 0)                 AS h,
        COALESCE(gl.home_runs, 0)            AS hr,
        COALESCE(gl.total_bases, 0)          AS tb,
        COALESCE(gl.walks_batter, 0)         AS bb,
        COALESCE(gl.strikeouts_batter, 0)    AS k_bat
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl ON gl.game_slug = g.game_slug
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
      AND gl.at_bats IS NOT NULL
      AND gl.at_bats > 0
),
derived AS (
    SELECT
        game_slug,
        game_date_et,
        player_id,
        ab, h, hr, tb, bb, k_bat,
        CASE WHEN ab > 0 THEN (tb - h)::float / ab       ELSE NULL END AS game_iso,
        CASE WHEN ab > 0 THEN k_bat::float / ab           ELSE NULL END AS game_k_rate,
        CASE WHEN (ab + bb) > 0 THEN bb::float / (ab + bb) ELSE NULL END AS game_bb_rate,
        CASE WHEN ab > 0 THEN hr::float / ab              ELSE NULL END AS game_hr_rate
    FROM batter_gamelogs
)
SELECT
    player_id,
    game_slug,
    game_date_et,

    -- Sample size: games in cross-season window
    COUNT(*)          OVER w10_cs   AS n_games_prev_10_cs,

    -- Hits cross-season rolling
    AVG(h)            OVER w10_cs   AS hits_avg_10_cs,
    AVG(h)            OVER w20_cs   AS hits_avg_20_cs,

    -- Total bases cross-season rolling
    AVG(tb)           OVER w10_cs   AS tb_avg_10_cs,
    AVG(tb)           OVER w20_cs   AS tb_avg_20_cs,

    -- Home runs cross-season rolling
    AVG(hr)           OVER w10_cs   AS hr_avg_10_cs,
    AVG(game_hr_rate) OVER w10_cs   AS hr_rate_avg_10_cs,

    -- At bats (plate appearance proxy)
    AVG(ab)           OVER w10_cs   AS ab_avg_10_cs,

    -- Rate stats
    AVG(game_k_rate)  OVER w10_cs   AS k_rate_avg_10_cs,
    AVG(game_bb_rate) OVER w10_cs   AS bb_rate_avg_10_cs,
    AVG(game_iso)     OVER w10_cs   AS iso_avg_10_cs

FROM derived
WINDOW
    w10_cs AS (PARTITION BY player_id
               ORDER BY game_date_et, game_slug
               ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    w20_cs AS (PARTITION BY player_id
               ORDER BY game_date_et, game_slug
               ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
;
