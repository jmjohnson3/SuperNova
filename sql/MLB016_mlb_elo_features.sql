-- MLB016_mlb_elo_features.sql
-- Elo features for MLB game training & prediction.
-- Mirrors the NBA V017_elo_features.sql pattern.
--
-- Provides:
--   home_elo, away_elo, elo_diff, elo_win_prob_home
--
-- For completed games: uses elo_pre (leakage-safe).
-- For upcoming games:  uses latest elo_post (most recent completed game).
-- Falls back to 1500 if no Elo history.

DROP VIEW IF EXISTS features.mlb_game_elo_features CASCADE;

CREATE OR REPLACE VIEW features.mlb_game_elo_features AS

WITH latest_elo AS (
    -- Most recent post-game Elo per team (for upcoming/unplayed games)
    SELECT DISTINCT ON (team_abbr)
        team_abbr,
        elo_post AS latest_elo
    FROM raw.mlb_elo
    ORDER BY team_abbr, game_date_et DESC, game_slug DESC
)
SELECT
    g.season,
    g.game_slug,
    g.game_date_et,
    -- Home Elo: prefer elo_pre from this game; fallback to latest; default 1500
    COALESCE(he.elo_pre, lh.latest_elo, 1500.0) AS home_elo,
    -- Away Elo: same logic
    COALESCE(ae.elo_pre, la.latest_elo, 1500.0) AS away_elo,
    -- Elo difference (home - away)
    COALESCE(he.elo_pre, lh.latest_elo, 1500.0)
      - COALESCE(ae.elo_pre, la.latest_elo, 1500.0) AS elo_diff,
    -- Expected home win probability (with home advantage = 24)
    1.0 / (1.0 + POWER(10.0,
        (COALESCE(ae.elo_pre, la.latest_elo, 1500.0)
         - (COALESCE(he.elo_pre, lh.latest_elo, 1500.0) + 24.0))
        / 400.0
    )) AS elo_win_prob_home
FROM raw.mlb_games g
LEFT JOIN raw.mlb_elo he
    ON he.game_slug = g.game_slug AND he.team_abbr = g.home_team_abbr
LEFT JOIN raw.mlb_elo ae
    ON ae.game_slug = g.game_slug AND ae.team_abbr = g.away_team_abbr
LEFT JOIN latest_elo lh
    ON lh.team_abbr = g.home_team_abbr
LEFT JOIN latest_elo la
    ON la.team_abbr = g.away_team_abbr
;
