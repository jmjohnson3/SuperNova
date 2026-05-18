-- MLB023: Batter career stats at specific venue
-- Plain VIEW — LATERAL-joined in train/predict. Leakage-safe: ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING.
DROP VIEW IF EXISTS features.mlb_batter_venue_stats CASCADE;
CREATE VIEW features.mlb_batter_venue_stats AS
WITH batter_venue AS (
    SELECT
        g.game_slug, g.game_date_et, g.venue_id, gl.player_id,
        COALESCE(gl.hits,         0) AS h,
        COALESCE(gl.at_bats,      0) AS ab,
        COALESCE(gl.home_runs,    0) AS hr,
        COALESCE(gl.total_bases,  0) AS tb,
        COALESCE(gl.walks_batter, 0) AS bb
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl ON gl.game_slug = g.game_slug
    WHERE g.status = 'final' AND g.home_score IS NOT NULL
      AND gl.at_bats > 0 AND g.venue_id IS NOT NULL
)
SELECT
    player_id, venue_id, game_slug, game_date_et,
    COUNT(*) OVER w                                         AS batter_n_games_at_venue,
    SUM(h)  OVER w / NULLIF(SUM(ab)      OVER w, 0)        AS batter_venue_ba,
    SUM(hr) OVER w / NULLIF(SUM(ab)      OVER w, 0)        AS batter_venue_hr_rate,
    SUM(tb) OVER w / NULLIF(SUM(ab)      OVER w, 0)        AS batter_venue_slg,
    SUM(bb) OVER w / NULLIF(SUM(ab + bb) OVER w, 0)        AS batter_venue_bb_rate
FROM batter_venue
WINDOW w AS (
    PARTITION BY player_id, venue_id
    ORDER BY game_date_et, game_slug
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
;
