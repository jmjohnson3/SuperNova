-- MLB000_schema_bootstrap.sql
-- Run once (or idempotently) to create all MLB raw + odds + bets tables.
-- Assumes raw, odds, bets, features schemas already exist from NBA bootstrap.

-- ============================================================
-- raw.mlb_teams
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_teams (
    team_id         INTEGER PRIMARY KEY,
    team_abbr       TEXT    NOT NULL UNIQUE,
    team_name       TEXT    NOT NULL,
    city            TEXT,
    division        TEXT,   -- e.g. 'AL East', 'NL West'
    league          TEXT,   -- 'AL' or 'NL'
    updated_at_utc  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- raw.mlb_venues
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_venues (
    venue_id        INTEGER PRIMARY KEY,
    venue_name      TEXT    NOT NULL,
    city            TEXT,
    state           TEXT,
    country         TEXT    DEFAULT 'US',
    roof_type       TEXT,   -- 'open', 'retractable', 'dome'
    turf_type       TEXT,   -- 'grass', 'artificial'
    capacity        INTEGER,
    latitude        NUMERIC(9, 6),
    longitude       NUMERIC(9, 6),
    updated_at_utc  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- raw.mlb_games
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_games (
    game_slug           TEXT    PRIMARY KEY,   -- YYYYMMDD-AWAY-HOME
    season              TEXT    NOT NULL,
    game_date_et        DATE    NOT NULL,
    start_ts_utc        TIMESTAMPTZ,
    home_team_abbr      TEXT    NOT NULL,
    away_team_abbr      TEXT    NOT NULL,
    venue_id            INTEGER,
    status              TEXT    NOT NULL DEFAULT 'scheduled',
    home_score          INTEGER,
    away_score          INTEGER,
    home_sp_id          INTEGER,
    away_sp_id          INTEGER,
    source_fetched_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mlb_games_date
    ON raw.mlb_games (game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_games_season
    ON raw.mlb_games (season);

-- ============================================================
-- raw.mlb_boxscore_games
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_games (
    game_slug           TEXT    PRIMARY KEY,
    home_runs           INTEGER,
    away_runs           INTEGER,
    home_hits           INTEGER,
    away_hits           INTEGER,
    home_errors         INTEGER,
    away_errors         INTEGER,
    innings_played      NUMERIC(4, 1),  -- 9.0, 12.0, etc.
    game_duration_min   INTEGER,
    played_status       TEXT,           -- 'COMPLETED', 'IN_PROGRESS', etc.
    fetched_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- raw.mlb_boxscore_team_stats
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_team_stats (
    game_slug   TEXT    NOT NULL,
    team_abbr   TEXT    NOT NULL,
    stats       JSONB,
    PRIMARY KEY (game_slug, team_abbr)
);

-- ============================================================
-- raw.mlb_boxscore_player_stats
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_boxscore_player_stats (
    game_slug           TEXT NOT NULL,
    player_id           INTEGER NOT NULL,
    season              TEXT,
    team_abbr           TEXT,
    team_id             INTEGER,
    is_home             BOOLEAN,
    first_name          TEXT,
    last_name           TEXT,
    primary_position    TEXT,
    batting_order       INTEGER,
    stats               JSONB,
    source_fetched_at_utc TIMESTAMPTZ,
    updated_at_utc      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_slug, player_id)
);

CREATE INDEX IF NOT EXISTS idx_mlb_bs_player_stats_player
    ON raw.mlb_boxscore_player_stats (player_id);

-- Performance indexes for frequently-joined columns
CREATE INDEX IF NOT EXISTS idx_mlb_games_venue_id
    ON raw.mlb_games (venue_id);

CREATE INDEX IF NOT EXISTS idx_mlb_games_home_sp_id
    ON raw.mlb_games (home_sp_id)
    WHERE home_sp_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mlb_games_away_sp_id
    ON raw.mlb_games (away_sp_id)
    WHERE away_sp_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mlb_games_game_date_et
    ON raw.mlb_games (game_date_et);

CREATE INDEX IF NOT EXISTS idx_mlb_games_status
    ON raw.mlb_games (status);

CREATE INDEX IF NOT EXISTS idx_mlb_bs_player_stats_team
    ON raw.mlb_boxscore_player_stats (team_abbr);

-- ============================================================
-- raw.mlb_player_gamelogs
-- (batting + pitching stats per player per game)
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_player_gamelogs (
    season          TEXT    NOT NULL,
    game_slug       TEXT    NOT NULL,
    player_id       INTEGER NOT NULL,
    team_abbr       TEXT    NOT NULL,
    game_date_et    DATE    NOT NULL,
    -- pitching
    is_starter      BOOLEAN DEFAULT FALSE,
    innings_pitched NUMERIC(5, 2),
    hits_allowed    INTEGER,
    runs_allowed    INTEGER,
    earned_runs     INTEGER,
    walks_allowed   INTEGER,
    strikeouts_pitcher INTEGER,
    home_runs_allowed  INTEGER,
    -- batting
    at_bats         INTEGER,
    hits            INTEGER,
    doubles         INTEGER,
    triples         INTEGER,
    home_runs       INTEGER,
    rbi             INTEGER,
    walks_batter    INTEGER,
    strikeouts_batter INTEGER,
    stolen_bases    INTEGER,
    total_bases     INTEGER,
    fetched_at_utc  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, game_slug, player_id)
);

CREATE INDEX IF NOT EXISTS idx_mlb_gamelogs_player
    ON raw.mlb_player_gamelogs (player_id, game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_gamelogs_team
    ON raw.mlb_player_gamelogs (team_abbr, game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_player_gamelogs_team
    ON raw.mlb_player_gamelogs (team_abbr);
CREATE INDEX IF NOT EXISTS idx_mlb_player_gamelogs_game_date
    ON raw.mlb_player_gamelogs (game_date_et);

-- ============================================================
-- raw.mlb_starting_pitchers
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_starting_pitchers (
    game_slug   TEXT    NOT NULL,
    team_abbr   TEXT    NOT NULL,
    player_id   INTEGER NOT NULL,
    player_name TEXT,
    source      TEXT    NOT NULL DEFAULT 'announced',  -- 'announced' or 'actual'
    fetched_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_slug, team_abbr)
);

-- ============================================================
-- raw.mlb_standings
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_standings (
    season          TEXT    NOT NULL,
    as_of_date      DATE    NOT NULL,
    team_abbr       TEXT    NOT NULL,
    wins            INTEGER,
    losses          INTEGER,
    win_pct         NUMERIC(5, 3),
    run_diff        INTEGER,
    division_rank   INTEGER,
    league_rank     INTEGER,
    updated_at_utc  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, as_of_date, team_abbr)
);

-- ============================================================
-- raw.mlb_injuries
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_injuries (
    player_id           INTEGER PRIMARY KEY,
    player_name         TEXT,
    team_abbr           TEXT,
    status              TEXT,   -- 'IL10', 'IL60', 'DTD', 'OUT'
    injury_description  TEXT,
    expected_return     DATE,
    updated_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- odds.mlb_game_lines
-- ============================================================
CREATE TABLE IF NOT EXISTS odds.mlb_game_lines (
    id                  BIGSERIAL PRIMARY KEY,
    fetched_at_utc      TIMESTAMPTZ NOT NULL,
    as_of_date          DATE NOT NULL,
    event_id            TEXT NOT NULL,
    bookmaker_key       TEXT NOT NULL,
    home_team_raw       TEXT,
    away_team_raw       TEXT,
    home_team_abbr      TEXT,
    away_team_abbr      TEXT,
    -- run line (always ±1.5)
    run_line_home       NUMERIC(4, 1) DEFAULT -1.5,
    run_line_home_price INTEGER,
    run_line_away_price INTEGER,
    -- totals
    total_line          NUMERIC(5, 1),
    over_price          INTEGER,
    under_price         INTEGER,
    -- open lines (set on first insert, not updated)
    open_run_line_home  NUMERIC(4, 1),
    open_total          NUMERIC(5, 1),
    UNIQUE (as_of_date, event_id, bookmaker_key)
);

CREATE INDEX IF NOT EXISTS idx_mlb_game_lines_date
    ON odds.mlb_game_lines (as_of_date);

-- ============================================================
-- odds.mlb_player_prop_lines
-- ============================================================
CREATE TABLE IF NOT EXISTS odds.mlb_player_prop_lines (
    id                  BIGSERIAL PRIMARY KEY,
    as_of_date          DATE NOT NULL,
    bookmaker_key       TEXT NOT NULL,
    player_name_raw     TEXT NOT NULL,
    player_name_norm    TEXT NOT NULL,
    stat                TEXT NOT NULL,  -- 'pitcher_strikeouts', 'batter_hits', 'batter_home_runs', 'batter_total_bases'
    line                NUMERIC(6, 1),
    over_price          INTEGER,
    under_price         INTEGER,
    open_line           NUMERIC(6, 1),  -- set on first insert
    over_link           TEXT,
    under_link          TEXT,
    fetched_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (as_of_date, bookmaker_key, player_name_norm, stat)
);

CREATE INDEX IF NOT EXISTS idx_mlb_prop_lines_date_player
    ON odds.mlb_player_prop_lines (as_of_date, player_name_norm);

-- ============================================================
-- bets.mlb_game_predictions
-- ============================================================
CREATE TABLE IF NOT EXISTS bets.mlb_game_predictions (
    game_slug           TEXT    PRIMARY KEY,
    game_date_et        DATE,
    home_team_abbr      TEXT,
    away_team_abbr      TEXT,
    home_sp_name        TEXT,
    away_sp_name        TEXT,
    pred_run_diff       NUMERIC(6, 2),
    pred_total          NUMERIC(6, 2),
    market_run_line     NUMERIC(4, 1),
    market_total        NUMERIC(5, 1),
    edge_run_line       NUMERIC(6, 2),
    edge_total          NUMERIC(6, 2),
    run_line_bet_side   TEXT,   -- 'home' or 'away' or NULL
    total_bet_side      TEXT,   -- 'over' or 'under' or NULL
    kelly_fraction_run_line NUMERIC(6, 4),
    kelly_fraction_total    NUMERIC(6, 4),
    win_prob_run_line   NUMERIC(5, 3),
    win_prob_total      NUMERIC(5, 3),
    actual_home_score   INTEGER,
    actual_away_score   INTEGER,
    run_line_covered    BOOLEAN,
    total_covered       BOOLEAN,
    closing_run_line    NUMERIC(4, 1),
    closing_total       NUMERIC(5, 1),
    clv_run_line        NUMERIC(5, 2),
    clv_total           NUMERIC(5, 2),
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- bets.mlb_prop_predictions
-- ============================================================
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    game_slug           TEXT    NOT NULL,
    game_date_et        DATE,
    player_id           INTEGER,
    player_name         TEXT    NOT NULL,
    team_abbr           TEXT,
    stat                TEXT    NOT NULL,  -- 'pitcher_strikeouts', 'batter_hits', etc.
    prediction          NUMERIC(6, 2),
    book_line           NUMERIC(6, 1),
    edge                NUMERIC(6, 2),
    kelly_fraction      NUMERIC(6, 4),
    bet_side            TEXT,              -- 'over' or 'under' or NULL
    actual_value        NUMERIC(6, 1),
    over_hit            BOOLEAN,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (game_slug, player_id, stat)
);
