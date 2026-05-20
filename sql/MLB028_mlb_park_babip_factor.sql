-- MLB028: Park-specific BABIP factor derived from actual gamelogs
-- Tiny 30-row result — no matview needed
-- BABIP = (H - HR) / (AB - K - HR)
CREATE OR REPLACE VIEW features.mlb_park_babip_factor AS
SELECT
    g.venue_id,
    AVG(
        CASE WHEN (COALESCE(gl.at_bats, 0)
                   - COALESCE(gl.strikeouts_batter, 0)
                   - COALESCE(gl.home_runs, 0)) > 0
             THEN (COALESCE(gl.hits, 0) - COALESCE(gl.home_runs, 0))::float
                / (COALESCE(gl.at_bats, 0)
                   - COALESCE(gl.strikeouts_batter, 0)
                   - COALESCE(gl.home_runs, 0))
             ELSE NULL END
    )                         AS park_babip_avg,
    COUNT(*) FILTER (WHERE COALESCE(gl.at_bats, 0) > 0) AS n_records
FROM raw.mlb_player_gamelogs gl
JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
WHERE g.status = 'final'
  AND gl.at_bats > 0
  AND g.venue_id IS NOT NULL
GROUP BY g.venue_id;
