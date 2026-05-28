DROP VIEW IF EXISTS features.mlb_sp_hand_hr_rate CASCADE;
CREATE VIEW features.mlb_sp_hand_hr_rate AS
WITH game_hand_stats AS (
    SELECT
        g.game_slug,
        g.game_date_et,
        pgl.player_id                                                          AS pitcher_id,
        SUM(bgl.at_bats)    FILTER (WHERE bh.bat_side = 'L')                  AS ab_vs_lhb,
        SUM(bgl.home_runs)  FILTER (WHERE bh.bat_side = 'L')                  AS hr_vs_lhb,
        SUM(bgl.at_bats)    FILTER (WHERE bh.bat_side = 'R')                  AS ab_vs_rhb,
        SUM(bgl.home_runs)  FILTER (WHERE bh.bat_side = 'R')                  AS hr_vs_rhb
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs pgl
        ON  pgl.game_slug       = g.game_slug
        AND pgl.is_starter      = TRUE
        AND pgl.innings_pitched >= 1.0
    JOIN raw.mlb_player_gamelogs bgl
        ON  bgl.game_slug  = g.game_slug
        AND bgl.team_abbr  != pgl.team_abbr
        AND bgl.at_bats    > 0
    LEFT JOIN raw.mlb_player_handedness bh ON bh.player_id = bgl.player_id
    WHERE g.status     = 'final'
      AND g.home_score IS NOT NULL
    GROUP BY g.game_slug, g.game_date_et, pgl.player_id
)
SELECT
    pitcher_id, game_slug, game_date_et,
    SUM(hr_vs_lhb) OVER w25 / NULLIF(SUM(ab_vs_lhb) OVER w25, 0)  AS sp_hr_rate_vs_lhb_25,
    SUM(hr_vs_rhb) OVER w25 / NULLIF(SUM(ab_vs_rhb) OVER w25, 0)  AS sp_hr_rate_vs_rhb_25,
    SUM(hr_vs_lhb) OVER w10 / NULLIF(SUM(ab_vs_lhb) OVER w10, 0)  AS sp_hr_rate_vs_lhb_10,
    SUM(hr_vs_rhb) OVER w10 / NULLIF(SUM(ab_vs_rhb) OVER w10, 0)  AS sp_hr_rate_vs_rhb_10,
    SUM(ab_vs_lhb) OVER w25                                         AS sp_n_ab_vs_lhb_25,
    SUM(ab_vs_rhb) OVER w25                                         AS sp_n_ab_vs_rhb_25
FROM game_hand_stats
WINDOW
    w25 AS (PARTITION BY pitcher_id ORDER BY game_date_et, game_slug
            ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY pitcher_id ORDER BY game_date_et, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);
