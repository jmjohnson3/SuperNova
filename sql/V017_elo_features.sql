-- ============================================================================
-- V017: Elo Rating Features
-- ============================================================================
-- Creates features.game_elo_features — pre-game Elo ratings and derived
-- features for both training and inference.
--
-- Data source: raw.nba_elo (populated by nba_pipeline.compute_elo)
--
-- Features:
--   home_elo          : home team pre-game Elo (exact for completed games,
--                       latest elo_post for upcoming games, 1500 fallback)
--   away_elo          : away team pre-game Elo (same logic)
--   elo_diff          : home_elo - away_elo
--   elo_win_prob_home : P(home wins) = 1/(1+10^((away-(home+100))/400))
--
-- Leakage-safe: elo_pre is computed BEFORE the game result (compute_elo.py
-- stores both elo_pre and elo_post; this view reads elo_pre for training).
--
-- Usage: JOIN this view into training/inference queries by (season, game_slug).
-- ============================================================================

CREATE OR REPLACE VIEW features.game_elo_features AS
WITH latest_elo AS (
    -- Most recent elo_post per team — used as fallback for upcoming games
    -- that don't yet have a row in raw.nba_elo.
    SELECT DISTINCT ON (team_abbr)
        team_abbr,
        elo_post AS latest_elo_post
    FROM   raw.nba_elo
    ORDER  BY team_abbr, game_date_et DESC
)
SELECT
    g.season,
    g.game_slug,
    g.game_date_et,
    g.home_team_abbr,
    g.away_team_abbr,

    -- Pre-game home Elo: exact row if present, otherwise latest known
    COALESCE(he.elo_pre, lhe.latest_elo_post, 1500.0)::float AS home_elo,

    -- Pre-game away Elo
    COALESCE(ae.elo_pre, lae.latest_elo_post, 1500.0)::float AS away_elo,

    -- Derived
    (COALESCE(he.elo_pre, lhe.latest_elo_post, 1500.0)
     - COALESCE(ae.elo_pre, lae.latest_elo_post, 1500.0))::float AS elo_diff,

    (1.0 / (
        1.0 + power(10.0, (
            COALESCE(ae.elo_pre, lae.latest_elo_post, 1500.0)
            - (COALESCE(he.elo_pre, lhe.latest_elo_post, 1500.0) + 100.0)
        ) / 400.0)
    ))::float AS elo_win_prob_home

FROM raw.nba_games g
-- Exact pre-game Elo (completed games)
LEFT JOIN raw.nba_elo he
    ON  he.game_slug  = g.game_slug
    AND he.team_abbr  = g.home_team_abbr
LEFT JOIN raw.nba_elo ae
    ON  ae.game_slug  = g.game_slug
    AND ae.team_abbr  = g.away_team_abbr
-- Latest Elo fallback (upcoming games)
LEFT JOIN latest_elo lhe ON lhe.team_abbr = g.home_team_abbr
LEFT JOIN latest_elo lae ON lae.team_abbr = g.away_team_abbr;
