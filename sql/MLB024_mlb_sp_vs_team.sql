-- MLB024: SP career stats vs specific opposing team (leakage-safe window)
-- Keys: (pitcher_id, opp_team_abbr, game_slug)
-- Pattern mirrors MLB015 mlb_batter_vs_sp.

CREATE OR REPLACE VIEW features.mlb_sp_vs_team AS
WITH matchups AS (
    SELECT
        sp.player_id                    AS pitcher_id,
        CASE WHEN sp.team_abbr = g.home_team_abbr
             THEN g.away_team_abbr
             ELSE g.home_team_abbr END  AS opp_team_abbr,
        g.game_slug,
        g.game_date_et,
        COALESCE(pgl.innings_pitched, 0)    AS ip,
        COALESCE(pgl.earned_runs, 0)        AS er,
        COALESCE(pgl.strikeouts_pitcher, 0) AS k,
        COALESCE(pgl.hits_allowed, 0)       AS h,
        COALESCE(pgl.walks_allowed, 0)      AS bb
    FROM raw.mlb_starting_pitchers sp
    JOIN raw.mlb_games g ON g.game_slug = sp.game_slug
    JOIN raw.mlb_player_gamelogs pgl
        ON pgl.game_slug = g.game_slug AND pgl.player_id = sp.player_id
    WHERE g.status = 'final'
      AND pgl.innings_pitched >= 1.0
      AND sp.source = 'actual'
),
rolling AS (
    SELECT
        pitcher_id, opp_team_abbr, game_slug, game_date_et,
        COUNT(*)                                OVER w  AS svt_games,
        COALESCE(SUM(ip)  OVER w, 0.0)                  AS svt_ip,
        COALESCE(SUM(er)  OVER w, 0)                    AS svt_er,
        COALESCE(SUM(k)   OVER w, 0)                    AS svt_k,
        COALESCE(SUM(h)   OVER w, 0)                    AS svt_h,
        COALESCE(SUM(bb)  OVER w, 0)                    AS svt_bb,
        COALESCE(SUM(k)   OVER w3, 0)                   AS svt_k_last3,
        COALESCE(SUM(er)  OVER w3, 0)                   AS svt_er_last3,
        COALESCE(SUM(ip)  OVER w3, 0.0)                 AS svt_ip_last3
    FROM matchups
    WINDOW w  AS (PARTITION BY pitcher_id, opp_team_abbr ORDER BY game_date_et, game_slug
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
           w3 AS (PARTITION BY pitcher_id, opp_team_abbr ORDER BY game_date_et, game_slug
                  ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
)
SELECT
    pitcher_id, opp_team_abbr, game_slug, game_date_et,
    svt_games,
    CASE WHEN svt_ip > 0 THEN (svt_er * 9.0) / svt_ip ELSE NULL END       AS svt_era,
    CASE WHEN svt_ip > 0 THEN (svt_k  * 9.0) / svt_ip ELSE NULL END       AS svt_k9,
    CASE WHEN svt_k + svt_h + svt_bb > 0
         THEN svt_k::float / (svt_k + svt_h + svt_bb) ELSE NULL END       AS svt_k_pct,
    CASE WHEN svt_ip_last3 > 0 THEN (svt_er_last3 * 9.0) / svt_ip_last3
         ELSE NULL END                                                      AS svt_era_last3,
    CASE WHEN svt_ip_last3 > 0 THEN (svt_k_last3  * 9.0) / svt_ip_last3
         ELSE NULL END                                                      AS svt_k9_last3
FROM rolling;
