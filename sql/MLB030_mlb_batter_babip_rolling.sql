-- MLB030: Batter rolling BABIP (H-HR)/(AB-K-HR) — luck/regression signal for hits
-- Leakage-safe: ROWS BETWEEN ... 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_batter_babip_rolling AS
WITH batter_games AS (
    SELECT
        gl.player_id,
        g.game_slug,
        g.game_date_et,
        COALESCE(gl.hits, 0)               AS h,
        COALESCE(gl.home_runs, 0)          AS hr,
        COALESCE(gl.at_bats, 0)            AS ab,
        COALESCE(gl.strikeouts_batter, 0)  AS k
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    WHERE g.status = 'final'
      AND gl.at_bats > 0
),
rolling AS (
    SELECT
        player_id,
        game_slug,
        game_date_et,
        -- balls in play (denominator): AB - K - HR
        COALESCE(SUM(GREATEST(ab - k - hr, 0)) OVER w10, 0) AS bip_10,
        COALESCE(SUM(GREATEST(h  - hr,     0)) OVER w10, 0) AS hits_on_bip_10,
        COALESCE(SUM(GREATEST(ab - k - hr, 0)) OVER wc,  0) AS bip_career,
        COALESCE(SUM(GREATEST(h  - hr,     0)) OVER wc,  0) AS hits_on_bip_career,
        COUNT(*)                           OVER w10          AS babip_games_10
    FROM batter_games
    WINDOW
        w10 AS (PARTITION BY player_id ORDER BY game_date_et, game_slug
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
        wc  AS (PARTITION BY player_id ORDER BY game_date_et, game_slug
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    player_id,
    game_slug,
    game_date_et,
    babip_games_10,
    CASE WHEN bip_10     > 0 THEN hits_on_bip_10::float    / bip_10     ELSE NULL END AS batter_babip_10,
    CASE WHEN bip_career > 0 THEN hits_on_bip_career::float / bip_career ELSE NULL END AS batter_babip_career
FROM rolling;
