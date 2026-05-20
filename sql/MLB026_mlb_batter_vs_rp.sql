-- MLB026: Batter stats vs relief pitching (bullpen games where opp SP < 5 IP)
-- Keys: (batter_id, game_slug)
-- Leakage-safe: ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_batter_vs_rp AS
WITH batter_games AS (
    SELECT
        gl.player_id                        AS batter_id,
        g.game_slug,
        g.game_date_et,
        CASE WHEN gl.team_abbr = g.home_team_abbr
             THEN g.away_team_abbr ELSE g.home_team_abbr END AS opp_team_abbr,
        COALESCE(gl.hits, 0)                AS h,
        COALESCE(gl.at_bats, 0)             AS ab,
        COALESCE(gl.home_runs, 0)           AS hr,
        COALESCE(gl.total_bases, 0)         AS tb,
        COALESCE(gl.strikeouts_batter, 0)   AS k
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    WHERE g.status = 'final'
      AND gl.at_bats > 0
),
sp_ip AS (
    SELECT
        sp.game_slug,
        sp.team_abbr  AS pitching_team,
        COALESCE(pgl.innings_pitched, 0) AS sp_ip
    FROM raw.mlb_starting_pitchers sp
    JOIN raw.mlb_player_gamelogs pgl
        ON pgl.game_slug = sp.game_slug AND pgl.player_id = sp.player_id
    WHERE sp.source = 'actual'
),
with_sp_ip AS (
    SELECT
        bg.*,
        COALESCE(si.sp_ip, 0)                                    AS opp_sp_ip,
        CASE WHEN COALESCE(si.sp_ip, 0) < 5 THEN 1 ELSE 0 END   AS is_bullpen_game
    FROM batter_games bg
    LEFT JOIN sp_ip si ON si.game_slug = bg.game_slug AND si.pitching_team = bg.opp_team_abbr
),
rolling AS (
    SELECT
        batter_id, game_slug, game_date_et,
        COUNT(*)                                                               OVER w30 AS bvr_games_30,
        COALESCE(SUM(is_bullpen_game)                                          OVER w30, 0) AS bvr_bp_games_30,
        COALESCE(SUM(CASE WHEN is_bullpen_game=1 THEN h  ELSE 0 END)          OVER w30, 0) AS bvr_h_30,
        COALESCE(SUM(CASE WHEN is_bullpen_game=1 THEN ab ELSE 0 END)          OVER w30, 0) AS bvr_ab_30,
        COALESCE(SUM(CASE WHEN is_bullpen_game=1 THEN hr ELSE 0 END)          OVER w30, 0) AS bvr_hr_30,
        COALESCE(SUM(CASE WHEN is_bullpen_game=1 THEN tb ELSE 0 END)          OVER w30, 0) AS bvr_tb_30,
        COALESCE(SUM(CASE WHEN is_bullpen_game=1 THEN k  ELSE 0 END)          OVER w30, 0) AS bvr_k_30
    FROM with_sp_ip
    WINDOW w30 AS (PARTITION BY batter_id ORDER BY game_date_et, game_slug
                   ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING)
)
SELECT
    batter_id, game_slug, game_date_et,
    bvr_games_30, bvr_bp_games_30, bvr_ab_30,
    CASE WHEN bvr_ab_30 > 0 THEN bvr_h_30::float  / bvr_ab_30 ELSE NULL END AS bvr_ba_30,
    CASE WHEN bvr_ab_30 > 0 THEN bvr_hr_30::float / bvr_ab_30 ELSE NULL END AS bvr_hr_rate_30,
    CASE WHEN bvr_ab_30 > 0 THEN bvr_tb_30::float / bvr_ab_30 ELSE NULL END AS bvr_slg_30,
    CASE WHEN bvr_ab_30 > 0 THEN bvr_k_30::float  / bvr_ab_30 ELSE NULL END AS bvr_k_rate_30
FROM rolling;
