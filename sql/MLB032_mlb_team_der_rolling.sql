-- MLB032: Team Defensive Efficiency Rating (DER) rolling
-- DER = 1 - BABIP_against; BIP estimated as ROUND(IP*3) + H - K - HR
-- Leakage-safe: ROWS BETWEEN ... 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_team_der_rolling AS
WITH team_pitching_games AS (
    -- Sum all pitchers' allowed stats per team per game (defensive context)
    SELECT
        g.game_date_et,
        g.game_slug,
        g.season,
        g.home_team_abbr AS team_abbr,
        COALESCE(SUM(pgl.hits_allowed), 0)       AS h,
        COALESCE(SUM(pgl.home_runs_allowed), 0)  AS hr,
        COALESCE(SUM(pgl.strikeouts_pitcher), 0) AS k,
        COALESCE(SUM(ROUND(COALESCE(pgl.innings_pitched, 0) * 3)), 0) AS outs_recorded
    FROM raw.mlb_player_gamelogs pgl
    JOIN raw.mlb_games g ON g.game_slug = pgl.game_slug
    WHERE g.status = 'final'
      AND pgl.team_abbr = g.home_team_abbr
      AND pgl.innings_pitched > 0
    GROUP BY g.game_date_et, g.game_slug, g.season, g.home_team_abbr

    UNION ALL

    SELECT
        g.game_date_et,
        g.game_slug,
        g.season,
        g.away_team_abbr AS team_abbr,
        COALESCE(SUM(pgl.hits_allowed), 0),
        COALESCE(SUM(pgl.home_runs_allowed), 0),
        COALESCE(SUM(pgl.strikeouts_pitcher), 0),
        COALESCE(SUM(ROUND(COALESCE(pgl.innings_pitched, 0) * 3)), 0)
    FROM raw.mlb_player_gamelogs pgl
    JOIN raw.mlb_games g ON g.game_slug = pgl.game_slug
    WHERE g.status = 'final'
      AND pgl.team_abbr = g.away_team_abbr
      AND pgl.innings_pitched > 0
    GROUP BY g.game_date_et, g.game_slug, g.season, g.away_team_abbr
),
with_bip AS (
    SELECT
        team_abbr,
        game_slug,
        game_date_et,
        season,
        -- BIP = outs on contact + hits on contact = (outs_recorded - K) + (H - HR)
        --     = outs_recorded + H - K - HR
        GREATEST(outs_recorded + h - k - hr, 0) AS bip,
        GREATEST(h - hr, 0)                     AS hits_on_bip
    FROM team_pitching_games
),
rolling AS (
    SELECT
        team_abbr,
        game_slug,
        game_date_et,
        season,
        COALESCE(SUM(bip)         OVER w20, 0) AS bip_20,
        COALESCE(SUM(hits_on_bip) OVER w20, 0) AS hits_bip_20,
        COALESCE(SUM(bip)         OVER wc,  0) AS bip_career,
        COALESCE(SUM(hits_on_bip) OVER wc,  0) AS hits_bip_career,
        COUNT(*)                  OVER w20      AS der_games_20
    FROM with_bip
    WINDOW
        w20 AS (PARTITION BY team_abbr, season ORDER BY game_date_et, game_slug
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        wc  AS (PARTITION BY team_abbr ORDER BY game_date_et, game_slug
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    team_abbr,
    game_slug,
    game_date_et,
    season,
    der_games_20,
    -- DER = 1 - BABIP_against; higher = better defense
    CASE WHEN bip_20     > 0 THEN 1.0 - hits_bip_20::float    / bip_20     ELSE NULL END AS team_der_20,
    CASE WHEN bip_career > 0 THEN 1.0 - hits_bip_career::float / bip_career ELSE NULL END AS team_der_career
FROM rolling;
