-- V021: player prop lines from The Odds API
-- Applied inline by parse_oddsapi.parse_prop_odds() on first run.

CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.nba_player_prop_lines (
    id                SERIAL PRIMARY KEY,
    as_of_date        DATE        NOT NULL,
    fetched_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_id          TEXT        NOT NULL,
    commence_time_utc TEXT,
    bookmaker_key     TEXT        NOT NULL,
    home_team         TEXT,
    away_team         TEXT,
    player_name       TEXT        NOT NULL,
    player_name_norm  TEXT        NOT NULL,  -- lowercased, accents stripped, no punctuation
    stat              TEXT        NOT NULL,  -- 'points', 'rebounds', 'assists'
    line              NUMERIC,
    over_price        INTEGER,              -- American odds e.g. -115
    under_price       INTEGER,             -- American odds e.g. -105
    updated_at_utc    TIMESTAMPTZ,
    UNIQUE (as_of_date, event_id, bookmaker_key, player_name_norm, stat)
);

CREATE INDEX IF NOT EXISTS idx_prop_lines_date_bk
    ON odds.nba_player_prop_lines (as_of_date, bookmaker_key);
