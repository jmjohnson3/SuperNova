-- MLB020: Starting catcher framing stats per (game_slug, team_abbr).
-- Grain: one row per (game_slug, team_abbr).
-- NULL for upcoming games (boxscore not yet available) → median-imputed by model.
CREATE OR REPLACE VIEW features.mlb_team_catcher_framing AS
SELECT
    bps.game_slug,
    bps.team_abbr,
    cf.framing_rv_per_100  AS catcher_framing_rv,
    cf.framing_rate        AS catcher_framing_rate,
    cf.framing_pitches     AS catcher_framing_pitches
FROM (
    SELECT DISTINCT ON (game_slug, team_abbr)
        game_slug,
        team_abbr,
        player_id
    FROM raw.mlb_boxscore_player_stats
    WHERE primary_position = 'C'
    ORDER BY game_slug, team_abbr, batting_order
) bps
JOIN raw.mlb_games g ON g.game_slug = bps.game_slug
LEFT JOIN raw.mlb_statcast_catcher_framing cf
    ON  cf.player_id   = bps.player_id
    AND cf.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT;
