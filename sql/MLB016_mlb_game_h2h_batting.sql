-- MLB016: Career batter vs SP H2H batting aggregates per game
--
-- For each game, aggregates career head-to-head batting stats for:
--   home_vs_awaysp: home batters vs the away starting pitcher
--   away_vs_homesp: away batters vs the home starting pitcher
--
-- Minimum 5 career AB required to count a matchup (avoids 1-AB noise).
-- home_h2h_n / away_h2h_n = number of batters with ≥5 career AB vs today's SP.
-- h2h_slg_edge = home lineup SLG advantage over the away SP vs away lineup over home SP.
--
-- Depends on:
--   features.mlb_batter_vs_sp  (MLB015 view — leakage-safe career H2H)
--   raw.mlb_games
--   raw.mlb_starting_pitchers
--
-- Applied by parse_all._refresh_matviews() via _MLB_MATVIEW_REFRESH.
-- Refreshed each run via _MLB_MATVIEW_REFRESH_SQL.

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_game_h2h_batting_mat AS
WITH
home_vs_awaysp AS (
    SELECT
        bvsp.game_slug,
        AVG(CASE WHEN bvsp.h2h_ab >= 5 THEN bvsp.h2h_ba  END) AS home_h2h_ba,
        AVG(CASE WHEN bvsp.h2h_ab >= 5 THEN bvsp.h2h_obp END) AS home_h2h_obp,
        AVG(CASE WHEN bvsp.h2h_ab >= 5 THEN bvsp.h2h_slg END) AS home_h2h_slg,
        COUNT(CASE WHEN bvsp.h2h_ab >= 5 THEN 1 END)          AS home_h2h_n
    FROM features.mlb_batter_vs_sp bvsp
    JOIN raw.mlb_games g ON g.game_slug = bvsp.game_slug
    JOIN raw.mlb_starting_pitchers asp
        ON  asp.game_slug  = bvsp.game_slug
        AND asp.team_abbr  = g.away_team_abbr
        AND asp.player_id  = bvsp.pitcher_id
    WHERE bvsp.h2h_games >= 1
    GROUP BY bvsp.game_slug
),
away_vs_homesp AS (
    SELECT
        bvsp.game_slug,
        AVG(CASE WHEN bvsp.h2h_ab >= 5 THEN bvsp.h2h_ba  END) AS away_h2h_ba,
        AVG(CASE WHEN bvsp.h2h_ab >= 5 THEN bvsp.h2h_obp END) AS away_h2h_obp,
        AVG(CASE WHEN bvsp.h2h_ab >= 5 THEN bvsp.h2h_slg END) AS away_h2h_slg,
        COUNT(CASE WHEN bvsp.h2h_ab >= 5 THEN 1 END)          AS away_h2h_n
    FROM features.mlb_batter_vs_sp bvsp
    JOIN raw.mlb_games g ON g.game_slug = bvsp.game_slug
    JOIN raw.mlb_starting_pitchers hsp
        ON  hsp.game_slug  = bvsp.game_slug
        AND hsp.team_abbr  = g.home_team_abbr
        AND hsp.player_id  = bvsp.pitcher_id
    WHERE bvsp.h2h_games >= 1
    GROUP BY bvsp.game_slug
)
SELECT
    g.game_slug,
    h.home_h2h_ba,
    h.home_h2h_obp,
    h.home_h2h_slg,
    COALESCE(h.home_h2h_n, 0)                                      AS home_h2h_n,
    a.away_h2h_ba,
    a.away_h2h_obp,
    a.away_h2h_slg,
    COALESCE(a.away_h2h_n, 0)                                      AS away_h2h_n,
    COALESCE(h.home_h2h_slg, 0) - COALESCE(a.away_h2h_slg, 0)     AS h2h_slg_edge
FROM raw.mlb_games g
LEFT JOIN home_vs_awaysp h ON h.game_slug = g.game_slug
LEFT JOIN away_vs_homesp a ON a.game_slug = g.game_slug
WHERE g.status = 'final'
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS mlb_game_h2h_batting_mat_game_slug_idx
    ON features.mlb_game_h2h_batting_mat (game_slug);
