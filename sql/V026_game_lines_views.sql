-- V026: Persist odds.team_name_map + derived views
-- These objects were previously only in the database (not in any SQL file).
-- Run this once after a DB rebuild to restore the full 30-team name map and
-- the two views that power market-line features in game_training_features.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Team name map: Odds API full name  →  MySportsFeeds abbreviation
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS odds.team_name_map (
    team_name TEXT NOT NULL,
    team_abbr TEXT NOT NULL,
    PRIMARY KEY (team_name)
);

-- Full 30-team seed (idempotent)
INSERT INTO odds.team_name_map (team_name, team_abbr) VALUES
    ('Atlanta Hawks',           'ATL'),
    ('Boston Celtics',          'BOS'),
    ('Brooklyn Nets',           'BRO'),
    ('Charlotte Hornets',       'CHA'),
    ('Chicago Bulls',           'CHI'),
    ('Cleveland Cavaliers',     'CLE'),
    ('Dallas Mavericks',        'DAL'),
    ('Denver Nuggets',          'DEN'),
    ('Detroit Pistons',         'DET'),
    ('Golden State Warriors',   'GSW'),
    ('Houston Rockets',         'HOU'),
    ('Indiana Pacers',          'IND'),
    ('Los Angeles Clippers',    'LAC'),
    ('Los Angeles Lakers',      'LAL'),
    ('Memphis Grizzlies',       'MEM'),
    ('Miami Heat',              'MIA'),
    ('Milwaukee Bucks',         'MIL'),
    ('Minnesota Timberwolves',  'MIN'),
    ('New Orleans Pelicans',    'NOP'),
    ('New York Knicks',         'NYK'),
    ('Oklahoma City Thunder',   'OKL'),   -- NOTE: OKL not OKC (MySportsFeeds convention)
    ('Orlando Magic',           'ORL'),
    ('Philadelphia 76ers',      'PHI'),
    ('Phoenix Suns',            'PHX'),
    ('Portland Trail Blazers',  'POR'),
    ('Sacramento Kings',        'SAC'),
    ('San Antonio Spurs',       'SAS'),
    ('Toronto Raptors',         'TOR'),
    ('Utah Jazz',               'UTA'),
    ('Washington Wizards',      'WAS')
ON CONFLICT (team_name) DO UPDATE SET team_abbr = EXCLUDED.team_abbr;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. nba_game_lines_norm: add abbreviations + market margin
--    market_margin_prior = -1 × spread_home_points (home favorite = positive)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW odds.nba_game_lines_norm AS
SELECT
    l.provider,
    l.as_of_date,
    l.fetched_at_utc,
    l.event_id,
    l.commence_time_utc,
    l.bookmaker_key,
    l.bookmaker_title,
    l.home_team,
    l.away_team,
    l.spread_home_points,
    l.spread_home_price,
    l.spread_away_points,
    l.spread_away_price,
    l.total_points,
    l.total_over_price,
    l.total_under_price,
    l.created_at_utc,
    l.updated_at_utc,
    hm.team_abbr AS home_team_abbr,
    am.team_abbr AS away_team_abbr,
    -1.0 * l.spread_home_points AS market_margin_prior
FROM odds.nba_game_lines l
LEFT JOIN odds.team_name_map hm ON hm.team_name = l.home_team
LEFT JOIN odds.team_name_map am ON am.team_name = l.away_team;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. nba_game_lines_open_close: first + last fetch per game per day
--    Partitions by (game_date_et, bookmaker_key, home_team_abbr, away_team_abbr)
--    so that game date (not fetch date) is the grouping key.
--    Filters out rows where abbreviations are NULL (unmapped teams).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW odds.nba_game_lines_open_close AS
WITH ranked AS (
    SELECT
        n.provider,
        n.as_of_date,
        n.fetched_at_utc,
        n.event_id,
        n.commence_time_utc,
        n.bookmaker_key,
        n.bookmaker_title,
        n.home_team,
        n.away_team,
        n.spread_home_points,
        n.spread_home_price,
        n.spread_away_points,
        n.spread_away_price,
        n.total_points,
        n.total_over_price,
        n.total_under_price,
        n.created_at_utc,
        n.updated_at_utc,
        n.home_team_abbr,
        n.away_team_abbr,
        n.market_margin_prior,
        (n.commence_time_utc AT TIME ZONE 'America/New_York')::date AS game_date_et,
        ROW_NUMBER() OVER (
            PARTITION BY (n.commence_time_utc AT TIME ZONE 'America/New_York')::date,
                         n.bookmaker_key, n.home_team_abbr, n.away_team_abbr
            ORDER BY n.fetched_at_utc
        ) AS rn_open,
        ROW_NUMBER() OVER (
            PARTITION BY (n.commence_time_utc AT TIME ZONE 'America/New_York')::date,
                         n.bookmaker_key, n.home_team_abbr, n.away_team_abbr
            ORDER BY n.fetched_at_utc DESC
        ) AS rn_close
    FROM odds.nba_game_lines_norm n
    WHERE n.home_team_abbr IS NOT NULL
      AND n.away_team_abbr IS NOT NULL
)
SELECT
    o.game_date_et                    AS as_of_date,
    c.bookmaker_key,
    c.home_team_abbr,
    c.away_team_abbr,
    o.fetched_at_utc                  AS open_fetched_at_utc,
    o.spread_home_points              AS open_spread_home_points,
    o.market_margin_prior             AS open_market_margin_prior,
    o.total_points                    AS open_total,
    c.fetched_at_utc                  AS close_fetched_at_utc,
    c.spread_home_points              AS close_spread_home_points,
    c.market_margin_prior             AS close_market_margin_prior,
    c.total_points                    AS close_total,
    c.market_margin_prior - o.market_margin_prior  AS line_move_margin,
    c.total_points        - o.total_points         AS line_move_total
FROM ranked o
JOIN ranked c
  ON  c.game_date_et    = o.game_date_et
  AND c.bookmaker_key   = o.bookmaker_key
  AND c.home_team_abbr  = o.home_team_abbr
  AND c.away_team_abbr  = o.away_team_abbr
WHERE o.rn_open = 1
  AND c.rn_close = 1;
