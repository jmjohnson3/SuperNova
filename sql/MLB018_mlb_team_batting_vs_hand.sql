-- MLB018: Team-level batting rolling stats split by opposing SP handedness (LHP vs RHP)
-- For each team per game, computes a rolling 40-game batting line (AVG, OBP, SLG)
-- split by whether they faced a left-handed or right-handed starting pitcher.
-- Leakage-safe: ROWS BETWEEN 40 PRECEDING AND 1 PRECEDING.
-- Used in MLB006 to add platoon-advantage features when SP handedness is known.
--
-- Why this matters: teams systematically hit better or worse vs one pitcher hand.
-- Knowing a team faces a LHP today, and they hit .285/.350/.480 vs LHP vs .255 overall,
-- is a meaningful run-scoring signal that season-average stats miss.
--
-- Requires: raw.mlb_player_handedness (populated by crawler_statsapi hand lookup).
-- When handedness is unknown for an SP, no row is produced for that game.
DROP VIEW IF EXISTS features.mlb_team_batting_vs_hand CASCADE;
CREATE VIEW features.mlb_team_batting_vs_hand AS
WITH
-- Best (actual > probable) SP record per game+team
best_sp AS (
    SELECT DISTINCT ON (game_slug, team_abbr)
        game_slug,
        team_abbr,
        player_id AS sp_player_id
    FROM raw.mlb_starting_pitchers
    WHERE player_id IS NOT NULL
    ORDER BY
        game_slug,
        team_abbr,
        CASE source
            WHEN 'actual'   THEN 0
            WHEN 'probable' THEN 1
            ELSE 2
        END,
        player_id
),
-- Team batting line per game: sum all player batting stats for each team,
-- tagged with the opposing SP's handedness.
team_game_batting AS (
    SELECT
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        opp_hand.pitch_hand                                 AS opp_sp_hand,
        -- Batting aggregate for this team this game
        SUM(COALESCE(gl.at_bats,       0))                 AS ab,
        SUM(COALESCE(gl.hits,          0))                 AS h,
        SUM(COALESCE(gl.home_runs,     0))                 AS hr,
        SUM(COALESCE(gl.walks_batter,  0))                 AS bb,
        SUM(COALESCE(gl.total_bases,   0))                 AS tb
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = g.game_slug
    -- Opposing SP (from the other team)
    JOIN best_sp opp_sp
        ON opp_sp.game_slug   = g.game_slug
       AND opp_sp.team_abbr  != gl.team_abbr
    -- Opposing SP handedness
    JOIN raw.mlb_player_handedness opp_hand
        ON opp_hand.player_id = opp_sp.sp_player_id
       AND opp_hand.pitch_hand IN ('L', 'R')
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
    GROUP BY
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        opp_hand.pitch_hand
)
SELECT
    tgb.team_abbr,
    tgb.game_slug,
    tgb.game_date_et,

    -- Rolling batting average vs LHP and RHP (last 40 games including that hand type)
    AVG(CASE WHEN tgb.opp_sp_hand = 'L' AND tgb.ab > 0
             THEN tgb.h::FLOAT / tgb.ab  ELSE NULL END) OVER w AS team_avg_vs_lhp,
    AVG(CASE WHEN tgb.opp_sp_hand = 'R' AND tgb.ab > 0
             THEN tgb.h::FLOAT / tgb.ab  ELSE NULL END) OVER w AS team_avg_vs_rhp,

    -- OBP approximation (H + BB) / (AB + BB)
    AVG(CASE WHEN tgb.opp_sp_hand = 'L' AND (tgb.ab + tgb.bb) > 0
             THEN (tgb.h + tgb.bb)::FLOAT / (tgb.ab + tgb.bb)
             ELSE NULL END) OVER w                              AS team_obp_vs_lhp,
    AVG(CASE WHEN tgb.opp_sp_hand = 'R' AND (tgb.ab + tgb.bb) > 0
             THEN (tgb.h + tgb.bb)::FLOAT / (tgb.ab + tgb.bb)
             ELSE NULL END) OVER w                              AS team_obp_vs_rhp,

    -- SLG = total bases / AB
    AVG(CASE WHEN tgb.opp_sp_hand = 'L' AND tgb.ab > 0
             THEN tgb.tb::FLOAT / tgb.ab  ELSE NULL END) OVER w AS team_slg_vs_lhp,
    AVG(CASE WHEN tgb.opp_sp_hand = 'R' AND tgb.ab > 0
             THEN tgb.tb::FLOAT / tgb.ab  ELSE NULL END) OVER w AS team_slg_vs_rhp,

    -- Sample sizes (how many games vs each hand in the 40-game window)
    COUNT(CASE WHEN tgb.opp_sp_hand = 'L' THEN 1 END) OVER w   AS games_vs_lhp,
    COUNT(CASE WHEN tgb.opp_sp_hand = 'R' THEN 1 END) OVER w   AS games_vs_rhp

FROM team_game_batting tgb
WINDOW w AS (
    PARTITION BY tgb.team_abbr
    ORDER BY tgb.game_date_et, tgb.game_slug
    -- Up to 40 prior games regardless of hand mix (AVG ignores NULLs for the non-matching hand)
    ROWS BETWEEN 40 PRECEDING AND 1 PRECEDING
)
;