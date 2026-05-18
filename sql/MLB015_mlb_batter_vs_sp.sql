-- MLB015: Batter vs specific starting pitcher — career H2H rolling stats
-- Leakage-safe: ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
-- One row per (batter_id, pitcher_id, game_slug) = career cumulative stats
-- up to but not including this game. NULL ratios when no prior history.
CREATE OR REPLACE VIEW features.mlb_batter_vs_sp AS
WITH matchups AS (
    SELECT
        gl.player_id                        AS batter_id,
        sp.player_id                        AS pitcher_id,
        g.game_slug,
        g.game_date_et,
        COALESCE(gl.hits, 0)                AS hits,
        COALESCE(gl.at_bats, 0)             AS ab,
        COALESCE(gl.walks_batter, 0)        AS bb,
        COALESCE(gl.strikeouts_batter, 0)   AS k_bat,
        COALESCE(gl.home_runs, 0)           AS hr,
        COALESCE(gl.total_bases, 0)         AS tb
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    JOIN raw.mlb_starting_pitchers sp
        ON  sp.game_slug = g.game_slug
        AND sp.team_abbr = CASE
                WHEN gl.team_abbr = g.home_team_abbr THEN g.away_team_abbr
                ELSE g.home_team_abbr
            END
    WHERE g.status = 'final'
      AND gl.at_bats > 0
),
rolling AS (
    SELECT
        batter_id, pitcher_id, game_slug, game_date_et,
        COUNT(*)                        OVER w  AS h2h_games,
        COALESCE(SUM(hits)   OVER w, 0)         AS h2h_h,
        COALESCE(SUM(ab)     OVER w, 0)         AS h2h_ab,
        COALESCE(SUM(bb)     OVER w, 0)         AS h2h_bb,
        COALESCE(SUM(k_bat)  OVER w, 0)         AS h2h_k,
        COALESCE(SUM(hr)     OVER w, 0)         AS h2h_hr,
        COALESCE(SUM(tb)     OVER w, 0)         AS h2h_tb,
        -- Last-3 matchup recency (captures hot/cold streaks vs this SP)
        COALESCE(SUM(hits) OVER w3, 0)          AS h2h_h_last3,
        COALESCE(SUM(ab)   OVER w3, 0)          AS h2h_ab_last3,
        COALESCE(SUM(tb)   OVER w3, 0)          AS h2h_tb_last3,
        COALESCE(SUM(hr)   OVER w3, 0)          AS h2h_hr_last3
    FROM matchups
    WINDOW w  AS (PARTITION BY batter_id, pitcher_id ORDER BY game_date_et, game_slug
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
           w3 AS (PARTITION BY batter_id, pitcher_id ORDER BY game_date_et, game_slug
                  ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
)
SELECT
    batter_id, pitcher_id, game_slug, game_date_et,
    h2h_games, h2h_h, h2h_ab, h2h_bb, h2h_k, h2h_hr, h2h_tb,
    CASE WHEN h2h_ab > 0 THEN h2h_h::float / h2h_ab ELSE NULL END          AS h2h_ba,
    CASE WHEN h2h_ab + h2h_bb > 0
         THEN (h2h_h + h2h_bb)::float / (h2h_ab + h2h_bb)
         ELSE NULL END                                                       AS h2h_obp,
    CASE WHEN h2h_ab > 0 THEN h2h_tb::float / h2h_ab ELSE NULL END          AS h2h_slg,
    CASE WHEN h2h_ab > 0 THEN h2h_k::float  / h2h_ab ELSE NULL END          AS h2h_k_rate,
    CASE WHEN h2h_ab > 0 THEN (h2h_tb - h2h_h)::float / h2h_ab ELSE NULL END AS h2h_iso,
    -- Last-3 matchup recency ratios
    CASE WHEN h2h_ab_last3 > 0 THEN h2h_h_last3::float  / h2h_ab_last3 ELSE NULL END AS h2h_ba_last3,
    CASE WHEN h2h_ab_last3 > 0 THEN h2h_tb_last3::float / h2h_ab_last3 ELSE NULL END AS h2h_slg_last3,
    CASE WHEN h2h_ab_last3 > 0 THEN h2h_hr_last3::float / h2h_ab_last3 ELSE NULL END AS h2h_hr_rate_last3,
    h2h_ab_last3
FROM rolling;
