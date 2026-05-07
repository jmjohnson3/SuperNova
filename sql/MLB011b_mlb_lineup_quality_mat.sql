-- MLB011b: Materialized view wrapping mlb_lineup_quality (MLB011).
-- Provides fast (game_slug, team_abbr) index lookups for pitcher training/prediction SQL.
-- Applied by _refresh_matviews() AFTER MLB011 re-apply so the base VIEW exists.

CREATE MATERIALIZED VIEW IF NOT EXISTS features.mlb_lineup_quality_mat
AS SELECT * FROM features.mlb_lineup_quality
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS mlb_lineup_quality_mat_pk
    ON features.mlb_lineup_quality_mat (game_slug, team_abbr);
