-- MLB014: Per-player full prior-season aggregate stats
-- Used as a stable 162-game prior to supplement early-season rolling features.
-- Filtered to at_bats >= 3 (starting lineup appearances only) so backup/
-- pinch-hit games don't dilute the averages.
-- Join in training: pss.season = prior_season(b.season)
-- Join in inference: pss.season = prior season of current date's year
CREATE OR REPLACE VIEW features.mlb_player_prev_season_stats AS
SELECT
    gl.player_id,
    g.season,
    COUNT(*)                                                          AS prev_games,
    ROUND(AVG(gl.hits::float)::numeric, 4)                           AS prev_hits_avg,
    ROUND(AVG(gl.total_bases::float)::numeric, 4)                    AS prev_tb_avg,
    ROUND(AVG(gl.home_runs::float)::numeric, 4)                      AS prev_hr_avg,
    ROUND(AVG(gl.at_bats::float)::numeric, 4)                        AS prev_ab_avg,
    ROUND(AVG(
        CASE WHEN gl.at_bats::int > 0
             THEN gl.strikeouts_batter::float / gl.at_bats::int
             ELSE NULL END
    )::numeric, 4)                                                    AS prev_k_rate,
    ROUND(AVG(
        CASE WHEN (gl.at_bats::int + COALESCE(gl.walks_batter::int, 0)) > 0
             THEN gl.walks_batter::float /
                  (gl.at_bats::int + COALESCE(gl.walks_batter::int, 0))
             ELSE NULL END
    )::numeric, 4)                                                    AS prev_bb_rate,
    ROUND(AVG(
        CASE WHEN gl.at_bats::int > 0
             THEN (gl.total_bases::int - gl.hits::int)::float /
                  gl.at_bats::int
             ELSE NULL END
    )::numeric, 4)                                                    AS prev_iso,
    ROUND(AVG(
        CASE WHEN gl.at_bats::int > 0
             THEN gl.home_runs::float / gl.at_bats::int
             ELSE NULL END
    )::numeric, 4)                                                    AS prev_hr_rate
FROM raw.mlb_player_gamelogs gl
JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
WHERE g.status = 'final'
  AND gl.at_bats IS NOT NULL
  AND gl.at_bats::int >= 3
GROUP BY gl.player_id, g.season
;
