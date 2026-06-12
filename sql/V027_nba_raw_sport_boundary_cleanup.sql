-- Remove NBA raw-store rows that were loaded from MLB MySportsFeeds payloads.
--
-- Run the corrected nba_pipeline.parse_games first so legitimate NBA games
-- whose game_slug collided with an MLB game are restored before this cleanup.

BEGIN;

-- Predictions made from an MLB-style season are not trustworthy training data.
DELETE FROM bets.game_predictions p
WHERE p.season IN (
    SELECT DISTINCT season
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND url LIKE '%/mlb/%'
      AND season IS NOT NULL
)
  AND p.season !~ '^[0-9]{4}-[0-9]{4}-';

-- These tables all carry a season key, so MLB-style season slugs are definitive.
DELETE FROM raw.nba_elo r
WHERE r.season IN (
    SELECT DISTINCT season FROM raw.api_responses
    WHERE provider = 'mysportsfeeds' AND url LIKE '%/mlb/%'
)
  AND r.season !~ '^[0-9]{4}-[0-9]{4}-';

DELETE FROM raw.nba_game_lineups r
WHERE r.season IN (
    SELECT DISTINCT season FROM raw.api_responses
    WHERE provider = 'mysportsfeeds' AND url LIKE '%/mlb/%'
)
  AND r.season !~ '^[0-9]{4}-[0-9]{4}-';

DELETE FROM raw.nba_player_gamelogs r
WHERE r.season IN (
    SELECT DISTINCT season FROM raw.api_responses
    WHERE provider = 'mysportsfeeds' AND url LIKE '%/mlb/%'
)
  AND r.season !~ '^[0-9]{4}-[0-9]{4}-';

DELETE FROM raw.nba_standings r
WHERE r.season IN (
    SELECT DISTINCT season FROM raw.api_responses
    WHERE provider = 'mysportsfeeds' AND url LIKE '%/mlb/%'
)
  AND r.season !~ '^[0-9]{4}-[0-9]{4}-';

DELETE FROM raw.nba_games r
WHERE r.season IN (
    SELECT DISTINCT season FROM raw.api_responses
    WHERE provider = 'mysportsfeeds' AND url LIKE '%/mlb/%'
)
  AND r.season !~ '^[0-9]{4}-[0-9]{4}-';

-- Player and venue IDs are global MySportsFeeds identifiers. Remove only IDs
-- seen in MLB payloads and never seen in an NBA payload.
WITH nba_ids AS (
    SELECT DISTINCT (player->>'id')::bigint AS player_id
    FROM raw.api_responses r
    CROSS JOIN LATERAL jsonb_array_elements(r.payload->'players') player
    WHERE r.provider = 'mysportsfeeds'
      AND r.endpoint = 'injuries'
      AND r.url LIKE '%/nba/%'
),
mlb_only_ids AS (
    SELECT DISTINCT (player->>'id')::bigint AS player_id
    FROM raw.api_responses r
    CROSS JOIN LATERAL jsonb_array_elements(r.payload->'players') player
    WHERE r.provider = 'mysportsfeeds'
      AND r.endpoint = 'injuries'
      AND r.url LIKE '%/mlb/%'
      AND NOT EXISTS (
          SELECT 1
          FROM nba_ids n
          WHERE n.player_id = (player->>'id')::bigint
      )
)
DELETE FROM raw.nba_injuries_history h
USING mlb_only_ids m
WHERE h.player_id = m.player_id;

WITH nba_ids AS (
    SELECT DISTINCT (player->>'id')::bigint AS player_id
    FROM raw.api_responses r
    CROSS JOIN LATERAL jsonb_array_elements(r.payload->'players') player
    WHERE r.provider = 'mysportsfeeds'
      AND r.endpoint = 'injuries'
      AND r.url LIKE '%/nba/%'
),
mlb_only_ids AS (
    SELECT DISTINCT (player->>'id')::bigint AS player_id
    FROM raw.api_responses r
    CROSS JOIN LATERAL jsonb_array_elements(r.payload->'players') player
    WHERE r.provider = 'mysportsfeeds'
      AND r.endpoint = 'injuries'
      AND r.url LIKE '%/mlb/%'
      AND NOT EXISTS (
          SELECT 1
          FROM nba_ids n
          WHERE n.player_id = (player->>'id')::bigint
      )
)
DELETE FROM raw.nba_injuries i
USING mlb_only_ids m
WHERE i.player_id = m.player_id;

WITH nba_ids AS (
    SELECT DISTINCT (item->'venue'->>'id')::integer AS venue_id
    FROM raw.api_responses r
    CROSS JOIN LATERAL jsonb_array_elements(r.payload->'venues') item
    WHERE r.provider = 'mysportsfeeds'
      AND r.endpoint = 'venues'
      AND r.url LIKE '%/nba/%'
),
mlb_only_ids AS (
    SELECT DISTINCT (item->'venue'->>'id')::integer AS venue_id
    FROM raw.api_responses r
    CROSS JOIN LATERAL jsonb_array_elements(r.payload->'venues') item
    WHERE r.provider = 'mysportsfeeds'
      AND r.endpoint = 'venues'
      AND r.url LIKE '%/mlb/%'
      AND NOT EXISTS (
          SELECT 1
          FROM nba_ids n
          WHERE n.venue_id = (item->'venue'->>'id')::integer
      )
)
DELETE FROM raw.nba_venues v
USING mlb_only_ids m
WHERE v.venue_id = m.venue_id;

COMMIT;
