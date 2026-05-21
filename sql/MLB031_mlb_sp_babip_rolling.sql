-- MLB031: SP rolling BABIP-against (H_allowed-HR_allowed)/(BIP) — luck/regression for Ks
-- BIP estimated as: outs_recorded + hits - HR ≈ ROUND(IP*3) + H - K - HR
-- Leakage-safe: ROWS BETWEEN ... 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_sp_babip_rolling AS
WITH starter_games AS (
    SELECT
        pgl.player_id,
        pgl.game_slug,
        g.game_date_et,
        COALESCE(pgl.hits_allowed, 0)       AS h,
        COALESCE(pgl.home_runs_allowed, 0)  AS hr,
        COALESCE(pgl.strikeouts_pitcher, 0) AS k,
        -- BIP denominator: outs on BIP + hits on BIP
        -- outs_recorded = ROUND(ip * 3); BIP = outs_on_contact + balls_in_play_hits
        -- = (outs_recorded - k) + (h - hr) = outs_recorded + h - k - hr
        GREATEST(ROUND(COALESCE(pgl.innings_pitched, 0) * 3) + COALESCE(pgl.hits_allowed, 0)
                 - COALESCE(pgl.strikeouts_pitcher, 0) - COALESCE(pgl.home_runs_allowed, 0), 0) AS bip
    FROM raw.mlb_player_gamelogs pgl
    JOIN raw.mlb_games g ON g.game_slug = pgl.game_slug
    WHERE g.status = 'final'
      AND pgl.is_starter = TRUE
      AND pgl.innings_pitched >= 1.0
),
rolling AS (
    SELECT
        player_id,
        game_slug,
        game_date_et,
        COALESCE(SUM(bip)         OVER w10, 0) AS bip_10,
        COALESCE(SUM(GREATEST(h - hr, 0)) OVER w10, 0) AS hits_on_bip_10,
        COALESCE(SUM(bip)         OVER wc,  0) AS bip_career,
        COALESCE(SUM(GREATEST(h - hr, 0)) OVER wc,  0) AS hits_on_bip_career,
        COUNT(*)                  OVER w10      AS sp_babip_starts_10
    FROM starter_games
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
    sp_babip_starts_10,
    CASE WHEN bip_10     > 0 THEN hits_on_bip_10::float    / bip_10     ELSE NULL END AS sp_babip_against_10,
    CASE WHEN bip_career > 0 THEN hits_on_bip_career::float / bip_career ELSE NULL END AS sp_babip_against_career
FROM rolling;
