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
    source_game_id      BIGINT,
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
ALTER TABLE raw.mlb_games ADD COLUMN IF NOT EXISTS source_game_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_mlb_games_source_game_id
    ON raw.mlb_games (source_game_id)
    WHERE source_game_id IS NOT NULL;

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
    fetched_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    home_f5_runs        INTEGER,        -- runs scored through end of 5th inning
    away_f5_runs        INTEGER
);
-- Idempotent additions for installations created before home_f5_runs was added
ALTER TABLE raw.mlb_boxscore_games ADD COLUMN IF NOT EXISTS home_f5_runs INTEGER;
ALTER TABLE raw.mlb_boxscore_games ADD COLUMN IF NOT EXISTS away_f5_runs INTEGER;

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
    provider              TEXT        NOT NULL,
    as_of_date            DATE        NOT NULL,
    fetched_at_utc        TIMESTAMPTZ NOT NULL,
    event_id              TEXT        NOT NULL,
    commence_time_utc     TEXT,
    bookmaker_key         TEXT        NOT NULL,
    bookmaker_title       TEXT,
    home_team             TEXT,
    away_team             TEXT,
    -- run line (always ±1.5)
    spread_home_points    NUMERIC(6,2),
    spread_home_price     INTEGER,
    spread_away_points    NUMERIC(6,2),
    spread_away_price     INTEGER,
    -- totals
    total_points          NUMERIC(6,2),
    total_over_price      INTEGER,
    total_under_price     INTEGER,
    spread_home_link      TEXT,
    spread_away_link      TEXT,
    total_over_link       TEXT,
    total_under_link      TEXT,
    updated_at_utc        TIMESTAMPTZ,
    PRIMARY KEY (provider, fetched_at_utc, event_id, bookmaker_key)
);

CREATE INDEX IF NOT EXISTS idx_mlb_game_lines_date_bk
    ON odds.mlb_game_lines (as_of_date, bookmaker_key);

-- ============================================================
-- odds.mlb_player_prop_lines
-- ============================================================
CREATE TABLE IF NOT EXISTS odds.mlb_player_prop_lines (
    id                SERIAL PRIMARY KEY,
    as_of_date        DATE        NOT NULL,
    fetched_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_id          TEXT        NOT NULL,
    commence_time_utc TEXT,
    bookmaker_key     TEXT        NOT NULL,
    home_team         TEXT,
    away_team         TEXT,
    player_name       TEXT        NOT NULL,
    player_name_norm  TEXT        NOT NULL,
    stat              TEXT        NOT NULL,
    line              NUMERIC,
    over_price        INTEGER,
    under_price       INTEGER,
    over_link         TEXT,
    under_link        TEXT,
    open_line         NUMERIC,
    updated_at_utc    TIMESTAMPTZ,
    UNIQUE (as_of_date, event_id, bookmaker_key, player_name_norm, stat, line)
);

CREATE INDEX IF NOT EXISTS idx_mlb_prop_lines_date_bk
    ON odds.mlb_player_prop_lines (as_of_date, bookmaker_key);

-- ============================================================
-- odds.mlb_player_prop_line_snapshots
-- Immutable open / prediction-lock / close observations.
-- ============================================================
CREATE TABLE IF NOT EXISTS odds.mlb_player_prop_line_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    snapshot_key        TEXT NOT NULL UNIQUE,
    snapshot_role       TEXT NOT NULL CHECK (snapshot_role IN ('open', 'lock', 'close')),
    snapshot_at_utc     TIMESTAMPTZ NOT NULL,
    source_prop_line_id BIGINT,
    prediction_id       INTEGER,
    prediction_key      TEXT,
    run_id              TEXT,
    as_of_date          DATE NOT NULL,
    event_id            TEXT,
    commence_time_utc   TIMESTAMPTZ,
    bookmaker_key       TEXT NOT NULL,
    home_team           TEXT,
    away_team           TEXT,
    player_name         TEXT NOT NULL,
    player_name_norm    TEXT NOT NULL,
    stat                TEXT NOT NULL,
    selected_side       TEXT CHECK (selected_side IN ('over', 'under')),
    line                NUMERIC NOT NULL,
    over_price          INTEGER,
    under_price         INTEGER,
    over_link           TEXT,
    under_link          TEXT,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_offer
    ON odds.mlb_player_prop_line_snapshots
    (as_of_date, player_name_norm, stat, bookmaker_key, line, snapshot_role);
CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_prop_snapshots_open_offer
    ON odds.mlb_player_prop_line_snapshots
    (as_of_date, event_id, bookmaker_key, player_name_norm, stat, line)
    WHERE snapshot_role = 'open';
CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_prediction
    ON odds.mlb_player_prop_line_snapshots (prediction_id, snapshot_role);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_snapshots_run_prediction_key
    ON odds.mlb_player_prop_line_snapshots (run_id, prediction_key, snapshot_role);
CREATE OR REPLACE FUNCTION odds.reject_mlb_prop_snapshot_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'odds.mlb_player_prop_line_snapshots is immutable';
END;
$$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_reject_mlb_prop_snapshot_mutation'
          AND tgrelid = 'odds.mlb_player_prop_line_snapshots'::regclass
          AND NOT tgisinternal
    ) THEN
        CREATE TRIGGER trg_reject_mlb_prop_snapshot_mutation
        BEFORE UPDATE OR DELETE OR TRUNCATE
        ON odds.mlb_player_prop_line_snapshots
        FOR EACH STATEMENT
        EXECUTE FUNCTION odds.reject_mlb_prop_snapshot_mutation();
    END IF;
END;
$$;

-- ============================================================
-- bets.mlb_game_predictions
-- ============================================================
CREATE TABLE IF NOT EXISTS bets.mlb_game_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    game_date_et        DATE    NOT NULL,
    game_slug           TEXT    NOT NULL,
    season              TEXT    NOT NULL,
    home_team_abbr      TEXT    NOT NULL,
    away_team_abbr      TEXT    NOT NULL,
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
    kelly_fraction_rl   NUMERIC(6, 4),
    kelly_fraction_total NUMERIC(6, 4),
    win_prob_rl         NUMERIC(5, 3),
    win_prob_total      NUMERIC(5, 3),
    used_residual_model BOOLEAN DEFAULT FALSE,
    actual_home_score   INTEGER,
    actual_away_score   INTEGER,
    actual_run_diff     NUMERIC(5,1),
    actual_total        NUMERIC(5,1),
    run_line_covered    BOOLEAN,
    total_covered       BOOLEAN,
    closing_run_line    NUMERIC(4, 1),
    closing_total       NUMERIC(5, 1),
    clv_run_line        NUMERIC(5, 2),
    clv_total           NUMERIC(5, 2),
    market_rl_price     NUMERIC,        -- entry juice for run-line bet side
    market_total_price  NUMERIC,        -- entry juice for total bet side
    closing_rl_price    NUMERIC,        -- closing juice for run-line bet side
    closing_total_price NUMERIC,        -- closing juice for total bet side
    clv_rl_price        NUMERIC,        -- price CLV in implied prob % pts
    clv_total_price     NUMERIC,        -- price CLV in implied prob % pts
    clv_rl_valid        BOOLEAN,
    clv_rl_status       TEXT,
    clv_rl_unknown_reason TEXT,
    clv_total_valid     BOOLEAN,
    clv_total_status    TEXT,
    clv_total_unknown_reason TEXT,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (game_date_et, game_slug)
);
-- Idempotent additions for installations created before price CLV was added
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS predicted_at_utc TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS season TEXT;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS used_residual_model BOOLEAN DEFAULT FALSE;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS kelly_fraction_rl NUMERIC(6, 4);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS win_prob_rl NUMERIC(5, 3);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS actual_home_score INTEGER;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS actual_away_score INTEGER;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS run_line_covered BOOLEAN;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS total_covered BOOLEAN;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS closing_run_line NUMERIC(4, 1);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS closing_total NUMERIC(5, 1);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_run_line NUMERIC(5, 2);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_total NUMERIC(5, 2);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS market_rl_price     NUMERIC;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS market_total_price  NUMERIC;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS closing_rl_price    NUMERIC;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS closing_total_price NUMERIC;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_rl_price        NUMERIC;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_total_price     NUMERIC;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_rl_valid BOOLEAN;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_rl_status TEXT;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_rl_unknown_reason TEXT;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_total_valid BOOLEAN;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_total_status TEXT;
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS clv_total_unknown_reason TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_game_predictions_date_slug
    ON bets.mlb_game_predictions (game_date_et, game_slug);
-- Idempotent additions for actual_run_diff / actual_total (Bug #1 fix)
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS actual_run_diff NUMERIC(5,1);
ALTER TABLE bets.mlb_game_predictions ADD COLUMN IF NOT EXISTS actual_total    NUMERIC(5,1);

-- ============================================================
-- bets.mlb_prop_predictions
-- ============================================================
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    game_date_et        DATE    NOT NULL,
    game_slug           TEXT    NOT NULL,
    player_id           INTEGER NOT NULL,
    player_name         TEXT,
    team_abbr           TEXT,
    stat                TEXT    NOT NULL,  -- 'pitcher_strikeouts', 'batter_hits', etc.
    prediction_key      TEXT,
    prop_offer_id       BIGINT,
    prop_offer_source_row_id INTEGER,
    prediction          NUMERIC(6, 2),
    pred_value          NUMERIC,
    pred_count          NUMERIC,
    pred_prob_over      NUMERIC,
    book_line           NUMERIC(6, 1),
    edge                NUMERIC(6, 2),
    edge_type           TEXT,
    model_family        TEXT,
    kelly_fraction      NUMERIC(6, 4),
    bet_side            TEXT,              -- 'over' or 'under' or NULL
    line_bucket         TEXT,
    over_price          NUMERIC,
    under_price         NUMERIC,
    bet_price           NUMERIC,
    minimum_acceptable_price NUMERIC,
    breakeven_prob      NUMERIC,
    ev                  NUMERIC,
    bookmaker_key       TEXT,
    bet_link            TEXT,
    bankroll_tier       TEXT,
    bankroll_candidate  BOOLEAN,
    bankroll_reasons    TEXT,
    stake_pct           NUMERIC,
    stake_usd           NUMERIC,
    actual_value        NUMERIC(6, 1),
    over_hit            BOOLEAN,
    closing_line        NUMERIC,
    closing_price       NUMERIC,
    clv_line            NUMERIC,
    clv_price           NUMERIC,
    beat_clv_line       BOOLEAN,
    beat_clv_price      BOOLEAN,
    closing_source_row_id BIGINT,
    closing_snapshot_id BIGINT,
    closing_fetched_at_utc TIMESTAMPTZ,
    clv_match_method    TEXT,
    clv_valid           BOOLEAN,
    clv_status          TEXT,
    clv_unknown_reason  TEXT,
    lock_snapshot_id    BIGINT,
    locked_at_utc       TIMESTAMPTZ,
    run_id              TEXT,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_at       TIMESTAMPTZ,
    stale_reason        TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE bets.mlb_prop_predictions
    ADD COLUMN IF NOT EXISTS run_id TEXT,
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS stale_reason TEXT,
    ADD COLUMN IF NOT EXISTS closing_source_row_id BIGINT,
    ADD COLUMN IF NOT EXISTS closing_snapshot_id BIGINT,
    ADD COLUMN IF NOT EXISTS closing_fetched_at_utc TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS clv_match_method TEXT,
    ADD COLUMN IF NOT EXISTS clv_valid BOOLEAN,
    ADD COLUMN IF NOT EXISTS clv_status TEXT,
    ADD COLUMN IF NOT EXISTS clv_unknown_reason TEXT,
    ADD COLUMN IF NOT EXISTS lock_snapshot_id BIGINT,
    ADD COLUMN IF NOT EXISTS locked_at_utc TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS minimum_acceptable_price NUMERIC,
    ADD COLUMN IF NOT EXISTS stake_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE bets.mlb_prop_predictions
    DROP CONSTRAINT IF EXISTS mlb_prop_predictions_game_slug_player_id_stat_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_mlb_prop_predictions_prediction_key
    ON bets.mlb_prop_predictions (prediction_key);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_offer
    ON bets.mlb_prop_predictions (prop_offer_id);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_date_market
    ON bets.mlb_prop_predictions (game_date_et, stat, bet_side);
CREATE INDEX IF NOT EXISTS idx_mlb_prop_predictions_active_date
    ON bets.mlb_prop_predictions (game_date_et, is_active);

-- ============================================================
-- bets.mlb_lottery_picks
-- One row per selected lottery leg per day.
-- actual_value / over_hit filled by _grade_lottery_picks() on next run.
-- ============================================================
CREATE TABLE IF NOT EXISTS bets.mlb_lottery_picks (
    id              BIGSERIAL PRIMARY KEY,
    game_date_et    DATE        NOT NULL,
    game_slug       TEXT        NOT NULL,
    player_id       INTEGER     NOT NULL,
    player_name     TEXT        NOT NULL,
    team_abbr       TEXT,
    stat            TEXT        NOT NULL,   -- 'pitcher_strikeouts', 'batter_hits', etc.
    pred_value      NUMERIC(7, 3),          -- regression model prediction
    book_line       NUMERIC(6, 1) NOT NULL, -- alt line selected (e.g. 2.5, 3.5)
    p_over          NUMERIC(6, 4),          -- Poisson P(stat > book_line)
    ev              NUMERIC(8, 4),          -- raw EV at offered odds
    streak_mult     NUMERIC(6, 4),          -- recency multiplier used for ranking
    ranked_ev       NUMERIC(8, 4),          -- ev * streak_mult (sort key)
    over_odds       INTEGER,                -- American odds offered (e.g. 500)
    actual_value    NUMERIC(6, 1),          -- filled after game completes
    over_hit        BOOLEAN,                -- filled after game completes
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (game_date_et, game_slug, player_id, stat)
);
CREATE INDEX IF NOT EXISTS idx_mlb_lottery_picks_date
    ON bets.mlb_lottery_picks (game_date_et);

-- ============================================================
-- raw.mlb_elo
-- (Elo ratings per team per game, computed by compute_elo.py)
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_elo (
    game_slug    TEXT   NOT NULL,
    team_abbr    TEXT   NOT NULL,
    season       TEXT   NOT NULL,
    game_date_et DATE   NOT NULL,
    elo_pre      FLOAT  NOT NULL,
    elo_post     FLOAT  NOT NULL,
    PRIMARY KEY (game_slug, team_abbr)
);
