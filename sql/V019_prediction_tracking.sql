-- V019: Prediction tracking tables
-- Applied via psycopg2 in update_outcomes.py / predict scripts (not a migration runner).

CREATE SCHEMA IF NOT EXISTS bets;

CREATE TABLE IF NOT EXISTS bets.game_predictions (
    id                  SERIAL PRIMARY KEY,
    predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_date_et        DATE        NOT NULL,
    game_slug           TEXT        NOT NULL,
    season              TEXT        NOT NULL,
    home_team_abbr      TEXT        NOT NULL,
    away_team_abbr      TEXT        NOT NULL,
    -- Model output
    pred_margin_home    NUMERIC,   -- home minus away
    pred_total          NUMERIC,
    used_residual_model BOOLEAN    DEFAULT FALSE,
    -- Market at prediction time
    market_spread_home  NUMERIC,
    market_total        NUMERIC,
    -- Edges (positive = we like home vs spread; positive = we like over vs total)
    edge_spread         NUMERIC,   -- pred_margin - market_spread_home
    edge_total          NUMERIC,   -- pred_total - market_total
    -- Filled in by update_outcomes.py after game completes
    actual_margin_home  NUMERIC,
    actual_total        NUMERIC,
    spread_bet_side     TEXT,      -- 'home' or 'away' or NULL if no edge
    total_bet_side      TEXT,      -- 'over' or 'under' or NULL if no edge
    spread_covered      BOOLEAN,
    total_correct       BOOLEAN,
    UNIQUE (game_date_et, game_slug)
);

CREATE TABLE IF NOT EXISTS bets.prop_predictions (
    id                  SERIAL PRIMARY KEY,
    predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_date_et        DATE        NOT NULL,
    game_slug           TEXT        NOT NULL,
    player_id           BIGINT      NOT NULL,
    player_name         TEXT,
    team_abbr           TEXT,
    pred_points         NUMERIC,
    pred_rebounds       NUMERIC,
    pred_assists        NUMERIC,
    -- Filled in by update_outcomes.py after game completes
    actual_points       NUMERIC,
    actual_rebounds     NUMERIC,
    actual_assists      NUMERIC,
    UNIQUE (game_date_et, game_slug, player_id)
);
