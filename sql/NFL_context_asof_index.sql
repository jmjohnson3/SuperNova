-- Maintenance-only, outside prediction transactions. Run with autocommit.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_context_season_asof
ON raw.nfl_context_observations (kind, ((payload->>'season')::int), observed_at DESC);
