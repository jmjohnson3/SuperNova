-- MLB027: SP LOB% / strand rate — career and rolling (last 10 starts)
-- Keys: (player_id, game_slug)
-- Formula: LOB% = 100 * (H_allowed + BB_allowed - R_allowed) / (H_allowed + BB_allowed)
-- Leakage-safe: ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_sp_lob_rate AS
WITH starter_games AS (
    SELECT
        pgl.player_id,
        pgl.game_slug,
        g.game_date_et,
        COALESCE(pgl.hits_allowed, 0)  AS h,
        COALESCE(pgl.walks_allowed, 0) AS bb,
        COALESCE(pgl.runs_allowed, 0)  AS r
    FROM raw.mlb_player_gamelogs pgl
    JOIN raw.mlb_games g ON g.game_slug = pgl.game_slug
    WHERE g.status = 'final'
      AND pgl.innings_pitched >= 1.0
      AND pgl.is_starter = TRUE
),
rolling AS (
    SELECT
        player_id, game_slug, game_date_et,
        COALESCE(SUM(h + bb) OVER w,   0) AS career_br,
        COALESCE(SUM(r)      OVER w,   0) AS career_r,
        COALESCE(SUM(h + bb) OVER w10, 0) AS br_10,
        COALESCE(SUM(r)      OVER w10, 0) AS r_10
    FROM starter_games
    WINDOW w   AS (PARTITION BY player_id ORDER BY game_date_et, game_slug
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
           w10 AS (PARTITION BY player_id ORDER BY game_date_et, game_slug
                   ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING)
)
SELECT
    player_id, game_slug, game_date_et,
    CASE WHEN career_br > 0
         THEN 100.0 * (career_br - career_r)::float / career_br
         ELSE NULL END AS sp_lob_pct_career,
    CASE WHEN br_10 > 0
         THEN 100.0 * (br_10 - r_10)::float / br_10
         ELSE NULL END AS sp_lob_pct_10
FROM rolling;
