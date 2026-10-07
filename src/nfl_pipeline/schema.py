"""DDL helpers for the NFL pipeline."""
from __future__ import annotations

import psycopg2

from nfl_pipeline.db import PG_DSN


NFL_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS odds;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS bets;

CREATE TABLE IF NOT EXISTS raw.api_responses (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    season TEXT,
    game_slug TEXT,
    as_of_date DATE,
    url TEXT NOT NULL,
    fetched_at_utc TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    payload_sha256 TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, endpoint, url)
);

CREATE TABLE IF NOT EXISTS raw.nfl_api_responses (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    snapshot_role TEXT NOT NULL DEFAULT 'live',
    season TEXT,
    game_slug TEXT,
    as_of_date DATE,
    url TEXT NOT NULL,
    fetched_at_utc TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    payload_sha256 TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nfl_api_responses_endpoint_date
    ON raw.nfl_api_responses (provider, endpoint, as_of_date, snapshot_role, fetched_at_utc);

CREATE TABLE IF NOT EXISTS raw.nfl_games (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    season_type TEXT,
    game_date_et DATE,
    start_ts_utc TIMESTAMPTZ,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    home_score NUMERIC,
    away_score NUMERIC,
    spread_line NUMERIC,
    total_line NUMERIC,
    roof TEXT,
    surface TEXT,
    temp NUMERIC,
    wind NUMERIC,
    status TEXT,
    source TEXT,
    raw_json JSONB,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS season_type TEXT;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS home_score NUMERIC;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS away_score NUMERIC;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS spread_line NUMERIC;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS total_line NUMERIC;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS roof TEXT;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS surface TEXT;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS temp NUMERIC;
ALTER TABLE raw.nfl_games ADD COLUMN IF NOT EXISTS wind NUMERIC;

CREATE TABLE IF NOT EXISTS raw.nfl_player_gamelogs (
    season INTEGER,
    week INTEGER,
    game_id TEXT,
    game_date_et DATE,
    player_id TEXT,
    player_name TEXT,
    team_abbr TEXT,
    opponent_abbr TEXT,
    position TEXT,
    is_home BOOLEAN,
    passing_yards NUMERIC,
    passing_tds NUMERIC,
    rushing_yards NUMERIC,
    rushing_tds NUMERIC,
    receiving_yards NUMERIC,
    receiving_tds NUMERIC,
    carries NUMERIC,
    targets NUMERIC,
    receptions NUMERIC,
    pass_attempts NUMERIC,
    route_participation NUMERIC,
    snap_share NUMERIC,
    source TEXT,
    raw_json JSONB,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, week, game_id, player_id, team_abbr)
);

CREATE INDEX IF NOT EXISTS idx_nfl_player_gamelogs_player_date
    ON raw.nfl_player_gamelogs (player_id, game_date_et);
CREATE INDEX IF NOT EXISTS idx_nfl_player_gamelogs_team_date
    ON raw.nfl_player_gamelogs (team_abbr, game_date_et);

CREATE TABLE IF NOT EXISTS raw.nfl_rosters (
    season INTEGER NOT NULL,
    week INTEGER,
    game_type TEXT,
    team_abbr TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT,
    player_name_norm TEXT,
    position TEXT,
    depth_chart_position TEXT,
    jersey_number INTEGER,
    roster_status TEXT,
    status_description_abbr TEXT,
    years_exp NUMERIC,
    height NUMERIC,
    weight NUMERIC,
    birth_date DATE,
    college TEXT,
    espn_id TEXT,
    sportradar_id TEXT,
    pfr_id TEXT,
    source TEXT,
    source_url TEXT,
    raw_json JSONB,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, week, game_type, team_abbr, player_id)
);

CREATE INDEX IF NOT EXISTS idx_nfl_rosters_latest_player
    ON raw.nfl_rosters (season, player_name_norm, team_abbr);
CREATE INDEX IF NOT EXISTS idx_nfl_rosters_team_position
    ON raw.nfl_rosters (season, team_abbr, position, roster_status);

CREATE TABLE IF NOT EXISTS raw.nfl_depth_charts (
    row_hash TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    snapshot_ts_utc TIMESTAMPTZ,
    team_abbr TEXT NOT NULL,
    player_id TEXT,
    player_name TEXT,
    player_name_norm TEXT,
    espn_id TEXT,
    pos_grp TEXT,
    pos_name TEXT,
    pos_abb TEXT,
    pos_slot INTEGER,
    pos_rank INTEGER,
    source TEXT,
    source_url TEXT,
    raw_json JSONB,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nfl_depth_latest_player
    ON raw.nfl_depth_charts (season, player_name_norm, team_abbr, snapshot_ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_nfl_depth_team_position
    ON raw.nfl_depth_charts (season, team_abbr, pos_abb, pos_rank);

CREATE TABLE IF NOT EXISTS raw.nfl_injuries (
    row_hash TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    season_type TEXT,
    game_type TEXT,
    week INTEGER,
    team_abbr TEXT NOT NULL,
    player_id TEXT,
    player_name TEXT,
    player_name_norm TEXT,
    position TEXT,
    report_primary_injury TEXT,
    report_secondary_injury TEXT,
    report_status TEXT,
    practice_primary_injury TEXT,
    practice_secondary_injury TEXT,
    practice_status TEXT,
    source TEXT,
    source_url TEXT,
    raw_json JSONB,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nfl_injuries_player
    ON raw.nfl_injuries (season, week, player_name_norm, team_abbr);
CREATE INDEX IF NOT EXISTS idx_nfl_injuries_team_status
    ON raw.nfl_injuries (season, week, team_abbr, report_status);

CREATE TABLE IF NOT EXISTS odds.nfl_player_prop_lines (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'oddsapi',
    snapshot_role TEXT NOT NULL DEFAULT 'live',
    as_of_date DATE NOT NULL,
    fetched_at_utc TIMESTAMPTZ NOT NULL,
    event_id TEXT,
    commence_time_utc TIMESTAMPTZ,
    bookmaker_key TEXT,
    bookmaker_title TEXT,
    home_team TEXT,
    away_team TEXT,
    player_name TEXT,
    player_name_norm TEXT,
    market_key TEXT,
    stat TEXT,
    line NUMERIC,
    over_price INTEGER,
    under_price INTEGER,
    over_link TEXT,
    under_link TEXT,
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, fetched_at_utc, event_id, bookmaker_key, player_name_norm, market_key, line)
);

ALTER TABLE odds.nfl_player_prop_lines ADD COLUMN IF NOT EXISTS snapshot_role TEXT NOT NULL DEFAULT 'live';

ALTER TABLE raw.nfl_depth_charts ADD COLUMN IF NOT EXISTS week INTEGER;
ALTER TABLE raw.nfl_depth_charts ADD COLUMN IF NOT EXISTS game_type TEXT;

CREATE INDEX IF NOT EXISTS idx_nfl_prop_lines_date_stat
    ON odds.nfl_player_prop_lines (as_of_date, stat);
CREATE INDEX IF NOT EXISTS idx_nfl_prop_lines_player
    ON odds.nfl_player_prop_lines (as_of_date, player_name_norm, stat);
CREATE INDEX IF NOT EXISTS idx_nfl_prop_lines_snapshot_role
    ON odds.nfl_player_prop_lines (as_of_date, snapshot_role, bookmaker_key);

CREATE TABLE IF NOT EXISTS odds.nfl_game_lines (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'oddsapi',
    snapshot_role TEXT NOT NULL DEFAULT 'live',
    as_of_date DATE NOT NULL,
    fetched_at_utc TIMESTAMPTZ NOT NULL,
    event_id TEXT,
    commence_time_utc TIMESTAMPTZ,
    bookmaker_key TEXT,
    bookmaker_title TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    spread_home_points NUMERIC,
    spread_home_price INTEGER,
    spread_away_points NUMERIC,
    spread_away_price INTEGER,
    total_points NUMERIC,
    total_over_price INTEGER,
    total_under_price INTEGER,
    spread_home_link TEXT,
    spread_away_link TEXT,
    total_over_link TEXT,
    total_under_link TEXT,
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, fetched_at_utc, event_id, bookmaker_key)
);

ALTER TABLE odds.nfl_game_lines ADD COLUMN IF NOT EXISTS snapshot_role TEXT NOT NULL DEFAULT 'live';

CREATE INDEX IF NOT EXISTS idx_nfl_game_lines_date
    ON odds.nfl_game_lines (as_of_date, bookmaker_key);
CREATE INDEX IF NOT EXISTS idx_nfl_game_lines_event
    ON odds.nfl_game_lines (event_id, fetched_at_utc);
CREATE INDEX IF NOT EXISTS idx_nfl_game_lines_snapshot_role
    ON odds.nfl_game_lines (as_of_date, snapshot_role, bookmaker_key);

CREATE TABLE IF NOT EXISTS features.nfl_player_game_training_features (
    season INTEGER,
    week INTEGER,
    game_id TEXT,
    game_date_et DATE,
    player_id TEXT,
    player_name TEXT,
    team_abbr TEXT,
    opponent_abbr TEXT,
    position TEXT,
    is_home BOOLEAN,
    n_games_prev_3 INTEGER,
    n_games_prev_5 INTEGER,
    n_games_prev_10 INTEGER,
    rest_days NUMERIC,
    team_game_number INTEGER,
    opp_game_number INTEGER,
    passing_yards NUMERIC,
    passing_tds NUMERIC,
    rushing_yards NUMERIC,
    rushing_tds NUMERIC,
    receiving_yards NUMERIC,
    receiving_tds NUMERIC,
    carries NUMERIC,
    targets NUMERIC,
    receptions NUMERIC,
    pass_attempts NUMERIC,
    route_participation NUMERIC,
    snap_share NUMERIC,
    roster_status TEXT,
    roster_is_active BOOLEAN,
    depth_chart_position TEXT,
    depth_pos_abb TEXT,
    depth_pos_rank NUMERIC,
    depth_pos_slot INTEGER,
    injury_report_status TEXT,
    injury_practice_status TEXT,
    game_total_line NUMERIC,
    team_spread_line NUMERIC,
    team_implied_points NUMERIC,
    opponent_implied_points NUMERIC,
    passing_yards_avg_3 NUMERIC,
    passing_yards_avg_5 NUMERIC,
    passing_yards_avg_10 NUMERIC,
    passing_tds_avg_3 NUMERIC,
    passing_tds_avg_5 NUMERIC,
    passing_tds_avg_10 NUMERIC,
    rushing_yards_avg_3 NUMERIC,
    rushing_yards_avg_5 NUMERIC,
    rushing_yards_avg_10 NUMERIC,
    rushing_tds_avg_3 NUMERIC,
    rushing_tds_avg_5 NUMERIC,
    rushing_tds_avg_10 NUMERIC,
    receiving_yards_avg_3 NUMERIC,
    receiving_yards_avg_5 NUMERIC,
    receiving_yards_avg_10 NUMERIC,
    receiving_tds_avg_3 NUMERIC,
    receiving_tds_avg_5 NUMERIC,
    receiving_tds_avg_10 NUMERIC,
    carries_avg_3 NUMERIC,
    carries_avg_5 NUMERIC,
    carries_avg_10 NUMERIC,
    targets_avg_3 NUMERIC,
    targets_avg_5 NUMERIC,
    targets_avg_10 NUMERIC,
    receptions_avg_3 NUMERIC,
    receptions_avg_5 NUMERIC,
    receptions_avg_10 NUMERIC,
    pass_attempts_avg_3 NUMERIC,
    pass_attempts_avg_5 NUMERIC,
    pass_attempts_avg_10 NUMERIC,
    route_participation_avg_3 NUMERIC,
    route_participation_avg_5 NUMERIC,
    route_participation_avg_10 NUMERIC,
    snap_share_avg_3 NUMERIC,
    snap_share_avg_5 NUMERIC,
    snap_share_avg_10 NUMERIC,
    opp_allowed_passing_yards_avg_5 NUMERIC,
    opp_allowed_rushing_yards_avg_5 NUMERIC,
    opp_allowed_receiving_yards_avg_5 NUMERIC,
    opp_allowed_passing_tds_avg_5 NUMERIC,
    opp_allowed_rushing_tds_avg_5 NUMERIC,
    opp_allowed_receiving_tds_avg_5 NUMERIC,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, player_id, team_abbr)
);

ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS route_participation NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS snap_share NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS roster_status TEXT;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS roster_is_active BOOLEAN;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS depth_chart_position TEXT;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS depth_pos_abb TEXT;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS depth_pos_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS depth_pos_slot INTEGER;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS injury_report_status TEXT;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS injury_practice_status TEXT;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS game_total_line NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_spread_line NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_implied_points NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS opponent_implied_points NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS route_participation_avg_3 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS route_participation_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS route_participation_avg_10 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS snap_share_avg_3 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS snap_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS snap_share_avg_10 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_pass_attempts_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_carries_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_targets_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_receptions_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_passing_yards_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_rushing_yards_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS team_player_receiving_yards_avg5_sum NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS pass_attempts_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS carries_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS targets_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS receptions_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS passing_yards_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS rushing_yards_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS receiving_yards_share_avg_5 NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS pass_attempts_role_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS carries_role_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS targets_role_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS receptions_role_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS passing_yards_role_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS rushing_yards_role_rank NUMERIC;
ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS receiving_yards_role_rank NUMERIC;

CREATE TABLE IF NOT EXISTS features.nfl_game_training_features (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    season_type TEXT,
    game_date_et DATE,
    start_ts_utc TIMESTAMPTZ,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    home_score NUMERIC,
    away_score NUMERIC,
    home_margin NUMERIC,
    total_points_actual NUMERIC,
    market_spread_home NUMERIC,
    market_total NUMERIC,
    roof TEXT,
    surface TEXT,
    temp NUMERIC,
    wind NUMERIC,
    home_game_number INTEGER,
    away_game_number INTEGER,
    home_rest_days NUMERIC,
    away_rest_days NUMERIC,
    home_pf_avg_3 NUMERIC,
    home_pf_avg_5 NUMERIC,
    home_pf_avg_10 NUMERIC,
    home_pa_avg_3 NUMERIC,
    home_pa_avg_5 NUMERIC,
    home_pa_avg_10 NUMERIC,
    home_margin_avg_3 NUMERIC,
    home_margin_avg_5 NUMERIC,
    home_margin_avg_10 NUMERIC,
    home_total_avg_3 NUMERIC,
    home_total_avg_5 NUMERIC,
    home_total_avg_10 NUMERIC,
    away_pf_avg_3 NUMERIC,
    away_pf_avg_5 NUMERIC,
    away_pf_avg_10 NUMERIC,
    away_pa_avg_3 NUMERIC,
    away_pa_avg_5 NUMERIC,
    away_pa_avg_10 NUMERIC,
    away_margin_avg_3 NUMERIC,
    away_margin_avg_5 NUMERIC,
    away_margin_avg_10 NUMERIC,
    away_total_avg_3 NUMERIC,
    away_total_avg_5 NUMERIC,
    away_total_avg_10 NUMERIC,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nfl_game_features_date
    ON features.nfl_game_training_features (game_date_et);

CREATE TABLE IF NOT EXISTS bets.nfl_player_prop_predictions (
    id BIGSERIAL PRIMARY KEY,
    game_date_et DATE NOT NULL,
    season INTEGER,
    week INTEGER,
    game_id TEXT,
    player_id TEXT,
    player_name TEXT,
    team_abbr TEXT,
    opponent_abbr TEXT,
    position TEXT,
    stat TEXT NOT NULL,
    projection NUMERIC,
    baseline_projection NUMERIC,
    model_version TEXT,
    line NUMERIC,
    side TEXT,
    price INTEGER,
    book TEXT,
    link TEXT,
    probability NUMERIC,
    ev NUMERIC,
    edge NUMERIC,
    tier TEXT NOT NULL DEFAULT 'paper',
    reasons TEXT,
    prediction_key TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (game_date_et, player_id, stat, line, side, book)
);

ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS roster_status TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS depth_chart_position TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS depth_pos_abb TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS depth_pos_rank NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS depth_pos_slot INTEGER;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS injury_report_status TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS injury_practice_status TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS offer_player_name TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS offer_player_name_norm TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS projection_p10 NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS projection_p50 NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS projection_p90 NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS distribution_kind TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS market_no_vig_probability NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS model_market_edge NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS projection_confidence NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS exact_line_model_probability NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS exact_line_clv_probability NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS exact_line_model_version TEXT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS over_probability NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS under_probability NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS raw_over_probability NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS minimum_american_price INTEGER;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS drift_guard_pass BOOLEAN;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS usage_context_quality NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS limited_usage_risk NUMERIC;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS spike_usage_probability NUMERIC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_nfl_player_prop_predictions_key
    ON bets.nfl_player_prop_predictions (prediction_key)
    WHERE prediction_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS bets.nfl_game_predictions (
    id BIGSERIAL PRIMARY KEY,
    game_date_et DATE NOT NULL,
    season INTEGER,
    week INTEGER,
    game_id TEXT,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    book TEXT,
    line NUMERIC,
    price INTEGER,
    link TEXT,
    predicted_home_margin NUMERIC,
    predicted_total_points NUMERIC,
    baseline_home_margin NUMERIC,
    baseline_total_points NUMERIC,
    probability NUMERIC,
    ev NUMERIC,
    edge NUMERIC,
    tier TEXT NOT NULL DEFAULT 'paper',
    model_version TEXT,
    reasons TEXT,
    prediction_key TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (game_date_et, game_id, market, side, book)
);

CREATE TABLE IF NOT EXISTS bets.nfl_game_prediction_results (
    prediction_id BIGINT PRIMARY KEY,
    game_date_et DATE NOT NULL,
    game_id TEXT,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    book TEXT,
    line NUMERIC,
    price INTEGER,
    actual_home_margin NUMERIC,
    actual_total_points NUMERIC,
    result TEXT,
    profit_per_unit NUMERIC,
    graded_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bets.nfl_player_prop_prediction_results (
    prediction_id BIGINT PRIMARY KEY,
    game_date_et DATE NOT NULL,
    game_id TEXT,
    player_id TEXT,
    player_name TEXT,
    stat TEXT NOT NULL,
    side TEXT,
    line NUMERIC,
    price INTEGER,
    actual_stat NUMERIC,
    result TEXT,
    profit_per_unit NUMERIC,
    graded_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bets.nfl_prediction_clv (
    source_kind TEXT NOT NULL,
    prediction_id BIGINT NOT NULL,
    game_date_et DATE NOT NULL,
    book TEXT,
    market TEXT,
    stat TEXT,
    side TEXT,
    locked_line NUMERIC,
    locked_price INTEGER,
    close_line NUMERIC,
    close_price INTEGER,
    close_fetched_at_utc TIMESTAMPTZ,
    clv_prob_delta NUMERIC,
    clv_status TEXT NOT NULL,
    commence_time_utc TIMESTAMPTZ,
    minutes_to_start NUMERIC,
    line_available_at_close BOOLEAN,
    valid_close_snapshot_captured BOOLEAN,
    close_quality_reason TEXT,
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_kind, prediction_id)
);

ALTER TABLE bets.nfl_prediction_clv ADD COLUMN IF NOT EXISTS commence_time_utc TIMESTAMPTZ;
ALTER TABLE bets.nfl_prediction_clv ADD COLUMN IF NOT EXISTS minutes_to_start NUMERIC;
ALTER TABLE bets.nfl_prediction_clv ADD COLUMN IF NOT EXISTS line_available_at_close BOOLEAN;
ALTER TABLE bets.nfl_prediction_clv ADD COLUMN IF NOT EXISTS valid_close_snapshot_captured BOOLEAN;
ALTER TABLE bets.nfl_prediction_clv ADD COLUMN IF NOT EXISTS close_quality_reason TEXT;

CREATE TABLE IF NOT EXISTS bets.nfl_bet_ledger (
    ledger_id BIGSERIAL PRIMARY KEY,
    source_kind TEXT NOT NULL,
    prediction_id BIGINT NOT NULL,
    locked_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_date_et DATE NOT NULL,
    tier TEXT NOT NULL,
    stake NUMERIC NOT NULL DEFAULT 1.0,
    book TEXT,
    market TEXT,
    stat TEXT,
    side TEXT,
    line NUMERIC,
    price INTEGER,
    link TEXT,
    model_version TEXT,
    prediction_key TEXT,
    result TEXT,
    profit NUMERIC,
    clv_status TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_kind, prediction_id, tier)
);
"""

NFL_PLAYER_USAGE_COLUMNS = (
    "routes_run",
    "pass_route_opportunities",
    "pass_route_opportunity_share",
    "target_share",
    "air_yards_share",
    "wopr",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "targets_per_route_run",
    "yards_per_route_run",
    "first_read_targets",
    "first_read_target_share",
    "end_zone_targets",
    "end_zone_target_share",
    "offense_snaps",
    "offense_snap_share",
    "red_zone_carries",
    "red_zone_targets",
    "red_zone_receptions",
    "red_zone_pass_attempts",
    "red_zone_pass_tds",
    "red_zone_rush_tds",
    "red_zone_rec_tds",
    "red_zone_touches",
    "goal_line_carries",
    "goal_line_targets",
)

NFL_PLAYER_FEATURE_TEXT_COLUMNS = (
    "season_type",
)

NFL_PLAYER_DERIVED_COLUMNS = (
    "is_week_18",
    "is_late_season",
    "is_postseason",
    "starter_confidence",
    "rest_risk_score",
    "injury_downgrade_score",
    "same_week_teammate_skill_injury_count",
    "same_week_teammate_skill_injury_score",
    "same_week_teammate_receiver_injury_score",
    "same_week_teammate_receiver_out_count",
    "same_week_context_confidence_score",
    "teammate_skill_injury_pressure_score",
    "teammate_receiver_injury_pressure_score",
    "receiver_teammate_vacancy_score",
    "receiver_target_eruption_score",
    "air_yards_spike_path_score",
    "receiver_air_yards_eruption_score",
    "receiver_explosive_spike_score",
    "receiver_target_eruption_anchor_targets",
    "receiver_air_yards_spike_anchor_yards",
    "receiver_contextual_spike_score",
    "depth_rank_delta",
    "depth_rank_worse",
    "depth_rank_better",
    "depth_rank_change_abs",
    "has_true_route_history",
    "estimated_routes_avg_5",
    "route_participation_proxy_avg_5",
    "targets_per_route_proxy_avg_5",
    "yards_per_route_proxy_avg_5",
    "route_target_intensity_avg_5",
    "route_weighted_target_expectation_avg_5",
    "route_weighted_yard_expectation_avg_5",
    "red_zone_targets_per_route_proxy_avg_5",
    "first_read_proxy_avg_5",
    "snap_share_trend_3_10",
    "snap_share_volatility_5",
    "targets_trend_3_10",
    "carries_trend_3_10",
    "receptions_trend_3_10",
    "receiving_yards_trend_3_10",
    "rushing_yards_trend_3_10",
    "target_share_trend_3_10",
    "air_yards_share_trend_3_10",
    "wopr_trend_3_10",
    "route_share_trend_3_10",
    "recent_snap_drop_score",
    "recent_snap_rise_score",
    "recent_target_spike_score",
    "recent_carry_spike_score",
    "recent_route_spike_score",
    "usage_volatility_score",
    "limited_workload_risk_score",
    "full_workload_score",
    "high_usage_fragility_score",
    "backup_role_score",
    "starter_role_stability_score",
    "projected_starter_score",
    "depth_movement_risk_score",
    "weird_usage_risk_score",
    "role_continuity_score",
    "normal_usage_path_score",
    "high_pass_attempt_score",
    "high_carry_score",
    "high_target_score",
    "spike_snap_share_score",
    "game_script_pass_boost",
    "game_script_rush_boost",
    "spike_target_opportunity_score",
    "spike_carry_opportunity_score",
    "spike_pass_attempt_opportunity_score",
    "target_spike_path_score",
    "carry_spike_path_score",
    "pass_spike_path_score",
    "receiver_route_quality_score",
    "receiver_yards_rate_signal",
    "receiver_route_env_score",
    "receiver_spike_yards_score",
    "rb_rush_role_env_score",
    "rb_carry_trend_env_score",
    "rb_spike_rush_score",
    "td_goal_line_env_score",
    "receiving_td_role_score",
    "receiving_td_route_redzone_score",
    "receiving_td_any_score",
    "receiver_usage_spike_signal",
    "rb_usage_spike_signal",
    "qb_volume_spike_signal",
    "role_change_upside_score",
    "workload_floor_score",
    "receiver_spike_volume_score",
    "receiver_spike_volume_anchor_yards",
    "receiver_target_command_score",
    "receiver_route_spike_readiness_score",
    "receiver_target_route_spike_score",
    "receiver_target_route_spike_anchor_targets",
    "receiver_target_route_spike_anchor_yards",
    "workload_downside_v2_score",
    "workload_upside_v2_score",
    "receiver_projected_targets_v2",
    "receiver_high_value_target_score",
    "receiver_target_spike_v2_score",
    "receiver_air_yards_spike_v2_score",
    "receiver_ypt_efficiency_spike_score",
    "receiver_spike_yards_anchor_v2",
    "rb_projected_carries_v2",
    "rb_carry_spike_v2_score",
    "rb_rush_yards_anchor_v2",
    "receiving_td_rare_event_score_v2",
    "same_week_usage_confidence_v3_score",
    "receiver_live_spike_v3_score",
    "receiver_projected_targets_v3",
    "receiver_spike_yards_anchor_v3",
    "rb_live_carry_v3_score",
    "rb_projected_carries_v3",
    "rb_rush_yards_anchor_v3",
    "yardage_projection_volatility_v3_score",
    "receiving_usage_history_quality_score",
    "rb_usage_history_quality_score",
    "td_usage_history_quality_score",
    "live_usage_context_quality_v4_score",
    "receiver_spike_under_correction_v4_score",
    "receiver_spike_yards_anchor_v4",
    "rb_carry_under_correction_v4_score",
    "rb_rush_yards_anchor_v4",
    "yardage_projection_volatility_v4_score",
    "pass_attempts_std_5",
    "pass_attempts_std_10",
    "carries_std_5",
    "carries_std_10",
    "targets_std_5",
    "targets_std_10",
    "receptions_std_5",
    "receptions_std_10",
    "rushing_yards_std_5",
    "rushing_yards_std_10",
    "receiving_yards_std_5",
    "receiving_yards_std_10",
    "rushing_tds_std_5",
    "rushing_tds_std_10",
    "receiving_tds_std_5",
    "receiving_tds_std_10",
    "routes_run_std_5",
    "routes_run_std_10",
    "pass_route_opportunities_std_5",
    "pass_route_opportunities_std_10",
    "pass_route_opportunity_share_std_5",
    "pass_route_opportunity_share_std_10",
    "snap_share_std_5",
    "snap_share_std_10",
    "offense_snap_share_std_5",
    "offense_snap_share_std_10",
    "red_zone_carries_std_5",
    "red_zone_carries_std_10",
    "red_zone_targets_std_5",
    "red_zone_targets_std_10",
    "goal_line_carries_std_5",
    "goal_line_carries_std_10",
    "goal_line_targets_std_5",
    "goal_line_targets_std_10",
)

NFL_GAME_CONTEXT_COLUMNS = (
    "home_plays_avg_3",
    "home_plays_avg_5",
    "home_plays_avg_10",
    "away_plays_avg_3",
    "away_plays_avg_5",
    "away_plays_avg_10",
    "home_pass_rate_avg_3",
    "home_pass_rate_avg_5",
    "home_pass_rate_avg_10",
    "away_pass_rate_avg_3",
    "away_pass_rate_avg_5",
    "away_pass_rate_avg_10",
    "home_yards_per_play_avg_3",
    "home_yards_per_play_avg_5",
    "home_yards_per_play_avg_10",
    "away_yards_per_play_avg_3",
    "away_yards_per_play_avg_5",
    "away_yards_per_play_avg_10",
    "home_yards_per_pass_avg_3",
    "home_yards_per_pass_avg_5",
    "home_yards_per_pass_avg_10",
    "away_yards_per_pass_avg_3",
    "away_yards_per_pass_avg_5",
    "away_yards_per_pass_avg_10",
    "home_yards_per_carry_avg_3",
    "home_yards_per_carry_avg_5",
    "home_yards_per_carry_avg_10",
    "away_yards_per_carry_avg_3",
    "away_yards_per_carry_avg_5",
    "away_yards_per_carry_avg_10",
    "home_tds_per_play_avg_3",
    "home_tds_per_play_avg_5",
    "home_tds_per_play_avg_10",
    "away_tds_per_play_avg_3",
    "away_tds_per_play_avg_5",
    "away_tds_per_play_avg_10",
    "home_red_zone_td_rate_avg_3",
    "home_red_zone_td_rate_avg_5",
    "home_red_zone_td_rate_avg_10",
    "away_red_zone_td_rate_avg_3",
    "away_red_zone_td_rate_avg_5",
    "away_red_zone_td_rate_avg_10",
    "home_red_zone_plays_avg_3",
    "home_red_zone_plays_avg_5",
    "home_red_zone_plays_avg_10",
    "away_red_zone_plays_avg_3",
    "away_red_zone_plays_avg_5",
    "away_red_zone_plays_avg_10",
    "home_qb_injury_risk",
    "away_qb_injury_risk",
    "home_ol_injury_score",
    "away_ol_injury_score",
    "home_skill_injury_score",
    "away_skill_injury_score",
    "home_total_injury_score",
    "away_total_injury_score",
)

NFL_SCHEMA_SQL += "\n" + "\n".join(
    f"ALTER TABLE raw.nfl_player_gamelogs ADD COLUMN IF NOT EXISTS {col} NUMERIC;"
    for col in NFL_PLAYER_USAGE_COLUMNS
)
NFL_SCHEMA_SQL += "\n" + "\n".join(
    f"ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS {col} NUMERIC;"
    for col in NFL_PLAYER_USAGE_COLUMNS
)
NFL_SCHEMA_SQL += "\n" + "\n".join(
    f"ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS {col}_avg_{window} NUMERIC;"
    for col in NFL_PLAYER_USAGE_COLUMNS
    for window in (3, 5, 10)
)
NFL_SCHEMA_SQL += "\n" + "\n".join(
    f"ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS {col} TEXT;"
    for col in NFL_PLAYER_FEATURE_TEXT_COLUMNS
)
NFL_SCHEMA_SQL += "\n" + "\n".join(
    f"ALTER TABLE features.nfl_player_game_training_features ADD COLUMN IF NOT EXISTS {col} NUMERIC;"
    for col in NFL_PLAYER_DERIVED_COLUMNS
)
NFL_SCHEMA_SQL += "\n" + "\n".join(
    f"ALTER TABLE features.nfl_game_training_features ADD COLUMN IF NOT EXISTS {col} NUMERIC;"
    for col in NFL_GAME_CONTEXT_COLUMNS
)


INTEGRITY_SQL = """
CREATE TABLE IF NOT EXISTS raw.nfl_schema_versions (version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT NOW());
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS integrity_version TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS offer_id BIGINT;
ALTER TABLE bets.nfl_player_prop_predictions ADD COLUMN IF NOT EXISTS forecast_payload JSONB;
ALTER TABLE bets.nfl_game_predictions ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE bets.nfl_game_predictions ADD COLUMN IF NOT EXISTS integrity_version TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE bets.nfl_game_predictions ADD COLUMN IF NOT EXISTS forecast_payload JSONB;
ALTER TABLE bets.nfl_bet_ledger ADD COLUMN IF NOT EXISTS execution_status TEXT NOT NULL DEFAULT 'simulated';
ALTER TABLE bets.nfl_bet_ledger ADD COLUMN IF NOT EXISTS execution_confirmed_at TIMESTAMPTZ;
DO $$ DECLARE r RECORD; BEGIN
  FOR r IN SELECT conrelid::regclass AS tbl, conname FROM pg_constraint
    WHERE conrelid IN ('bets.nfl_player_prop_predictions'::regclass, 'bets.nfl_game_predictions'::regclass)
      AND contype = 'u'
  LOOP EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', r.tbl, r.conname); END LOOP;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS idx_nfl_game_prediction_revision ON bets.nfl_game_predictions(prediction_key);
CREATE OR REPLACE FUNCTION bets.nfl_immutable_forecast() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF OLD.integrity_version <> 'legacy' THEN
   IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Locked NFL forecasts cannot be deleted'; END IF;
   IF (to_jsonb(OLD) - 'is_current' - 'updated_at_utc') IS DISTINCT FROM
      (to_jsonb(NEW) - 'is_current' - 'updated_at_utc') THEN
     RAISE EXCEPTION 'Locked NFL forecast fields are immutable';
   END IF;
 END IF;
 IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS nfl_immutable_forecast ON bets.nfl_player_prop_predictions;
CREATE TRIGGER nfl_immutable_forecast BEFORE UPDATE OR DELETE ON bets.nfl_player_prop_predictions
FOR EACH ROW EXECUTE FUNCTION bets.nfl_immutable_forecast();
DROP TRIGGER IF EXISTS nfl_immutable_forecast ON bets.nfl_game_predictions;
CREATE TRIGGER nfl_immutable_forecast BEFORE UPDATE OR DELETE ON bets.nfl_game_predictions
FOR EACH ROW EXECUTE FUNCTION bets.nfl_immutable_forecast();
CREATE TABLE IF NOT EXISTS raw.nfl_context_observations (
 kind TEXT NOT NULL, row_id TEXT NOT NULL, observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 payload JSONB NOT NULL, PRIMARY KEY(kind, row_id, observed_at)
);
-- Bookkeeping columns are not content: re-importing an unchanged row rewrites created_at_utc and
-- raw_json, which made 92% of roster observations duplicates of their predecessor and grew the
-- as-of log by ~162k rows/week for no information. raw_json is also dropped from the stored payload
-- (nothing reads it back through the *_at() builders), halving the bytes an as-of rebuild must read.
CREATE OR REPLACE FUNCTION raw.nfl_capture_context() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE body JSONB; BEGIN
 -- body decides whether anything changed; the stored payload keeps updated_at_utc, which the
 -- *_at() builders surface as roster/depth/injury_observed_at and scoring reads.
 body := to_jsonb(NEW) - 'updated_at_utc' - 'created_at_utc' - 'raw_json';
 IF TG_OP = 'INSERT' OR body IS DISTINCT FROM (to_jsonb(OLD) - 'updated_at_utc' - 'created_at_utc' - 'raw_json') THEN
   INSERT INTO raw.nfl_context_observations(kind,row_id,payload)
   VALUES(TG_TABLE_NAME, md5(body::text), to_jsonb(NEW) - 'raw_json') ON CONFLICT DO NOTHING;
 ELSE
   NEW.updated_at_utc := OLD.updated_at_utc;
 END IF;
 RETURN NEW;
END $$;
DO $$ DECLARE t TEXT; BEGIN
 FOREACH t IN ARRAY ARRAY['nfl_rosters','nfl_depth_charts','nfl_injuries'] LOOP
  -- Seed one observation per entity, but only on a log that has none for this kind. Re-seeding a
  -- live log injects rows dated at the base row's updated_at_utc, which can outrank real captured
  -- history and silently change which depth-chart entry wins an equal-pos_rank tie.
  CONTINUE WHEN EXISTS (SELECT 1 FROM raw.nfl_context_observations WHERE kind = t);
  -- row_id and payload must match what the trigger writes, or the seed row looks like a change.
  EXECUTE format('INSERT INTO raw.nfl_context_observations(kind,row_id,observed_at,payload) SELECT %L,md5((to_jsonb(x) - ''updated_at_utc'' - ''created_at_utc'' - ''raw_json'')::text),updated_at_utc,to_jsonb(x) - ''raw_json'' FROM (SELECT DISTINCT ON (season,COALESCE(player_id,player_name_norm),team_abbr) * FROM raw.%I WHERE season >= extract(year FROM NOW()) - 1 ORDER BY season,COALESCE(player_id,player_name_norm),team_abbr,updated_at_utc DESC) x ON CONFLICT DO NOTHING',t,t);
  EXECUTE format('DROP TRIGGER IF EXISTS nfl_capture_context ON raw.%I',t);
  EXECUTE format('CREATE TRIGGER nfl_capture_context BEFORE INSERT OR UPDATE ON raw.%I FOR EACH ROW EXECUTE FUNCTION raw.nfl_capture_context()',t);
 END LOOP;
END $$;
CREATE INDEX IF NOT EXISTS idx_nfl_context_asof ON raw.nfl_context_observations(kind, observed_at);
CREATE OR REPLACE FUNCTION raw.nfl_rosters_at(cutoff TIMESTAMPTZ) RETURNS SETOF raw.nfl_rosters LANGUAGE sql STABLE AS $$
 SELECT (jsonb_populate_record(NULL::raw.nfl_rosters, payload)).*
 FROM raw.nfl_context_observations WHERE kind='nfl_rosters' AND observed_at <= cutoff;
$$;
CREATE OR REPLACE FUNCTION raw.nfl_injuries_at(cutoff TIMESTAMPTZ) RETURNS SETOF raw.nfl_injuries LANGUAGE sql STABLE AS $$
 SELECT (jsonb_populate_record(NULL::raw.nfl_injuries, payload)).*
 FROM raw.nfl_context_observations WHERE kind='nfl_injuries' AND observed_at <= cutoff;
$$;
CREATE OR REPLACE FUNCTION raw.nfl_depth_charts_at(cutoff TIMESTAMPTZ) RETURNS SETOF raw.nfl_depth_charts LANGUAGE sql STABLE AS $$
 SELECT (jsonb_populate_record(NULL::raw.nfl_depth_charts, payload)).*
 FROM raw.nfl_context_observations WHERE kind='nfl_depth_charts' AND observed_at <= cutoff;
$$;
INSERT INTO raw.nfl_schema_versions(version) VALUES('20261007_context_log_compaction') ON CONFLICT DO NOTHING;
"""


def changed_where(table: str, cols: tuple[str, ...]) -> str:
    """ON CONFLICT ... DO UPDATE ... WHERE clause that skips no-op rewrites."""
    return (f"({', '.join(f'{table}.{c}' for c in cols)}) "
            f"IS DISTINCT FROM ({', '.join(f'EXCLUDED.{c}' for c in cols)})")


PROP_LINE_CHANGED = changed_where("odds.nfl_player_prop_lines", (
    "snapshot_role", "commence_time_utc", "bookmaker_title", "home_team", "away_team",
    "player_name", "stat", "over_price", "under_price", "over_link", "under_link",
))
GAME_LINE_CHANGED = changed_where("odds.nfl_game_lines", (
    "snapshot_role", "commence_time_utc", "bookmaker_title", "home_team", "away_team",
    "home_team_abbr", "away_team_abbr",
    "spread_home_points", "spread_home_price", "spread_away_points", "spread_away_price",
    "total_points", "total_over_price", "total_under_price",
    "spread_home_link", "spread_away_link", "total_over_link", "total_under_link",
))


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('raw.nfl_schema_versions')")
        if cur.fetchone()[0]:
            cur.execute("SELECT 1 FROM raw.nfl_schema_versions WHERE version = '20261007_context_log_compaction'")
            if cur.fetchone():
                conn.commit()
                return
        cur.execute("SET LOCAL lock_timeout = '5s'")
        cur.execute("SET LOCAL statement_timeout = '60s'")
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('nfl_pipeline_schema'))")
        cur.execute(NFL_SCHEMA_SQL)
        cur.execute(INTEGRITY_SQL)
    conn.commit()


def main() -> None:
    with psycopg2.connect(PG_DSN) as conn:
        ensure_schema(conn)
        from nfl_pipeline.cash_execution import schema
        schema(conn)
    print("NFL schema ready")


if __name__ == "__main__":
    main()
