-- MLB017: Per-SP per-venue career rolling stats
-- For each starting pitcher, computes career ERA, WHIP, K9, and FIP at each
-- specific venue from prior starts at that ballpark only.
-- Leakage-safe: ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING.
-- Used in MLB006 to add park-familiarity signal to individual SP features.
--
-- Why this matters: some pitchers are demonstrably better/worse at specific parks
-- (e.g., a flyball pitcher at Coors Field vs. a groundball pitcher at Fenway).
-- This complements the generic home/away ERA split with park-specific history.
DROP VIEW IF EXISTS features.mlb_sp_venue_stats CASCADE;
CREATE VIEW features.mlb_sp_venue_stats AS
WITH starter_venue AS (
    SELECT
        g.game_slug,
        g.game_date_et,
        g.venue_id,
        gl.player_id,
        COALESCE(gl.innings_pitched, 0)    AS ip,
        COALESCE(gl.earned_runs, 0)        AS er,
        COALESCE(gl.strikeouts_pitcher, 0) AS k,
        COALESCE(gl.walks_allowed, 0)      AS bb,
        COALESCE(gl.hits_allowed, 0)       AS h,
        COALESCE(gl.home_runs_allowed, 0)  AS hr
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = g.game_slug
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
      AND gl.is_starter = TRUE
      AND gl.innings_pitched IS NOT NULL
      AND gl.innings_pitched > 0
      AND g.venue_id IS NOT NULL
)
SELECT
    sv.player_id,
    sv.venue_id,
    sv.game_slug,
    sv.game_date_et,

    -- Number of prior starts at this specific venue (sample size for reliability weighting)
    COUNT(*) OVER w                                                          AS n_starts_at_venue,

    -- Career ERA at this venue (all prior starts, leakage-safe)
    AVG(CASE WHEN sv.ip > 0 THEN 9.0 * sv.er / sv.ip ELSE NULL END) OVER w  AS venue_era,

    -- Career WHIP at this venue
    AVG(CASE WHEN sv.ip > 0
             THEN (sv.h + sv.bb)::FLOAT / sv.ip
             ELSE NULL END) OVER w                                           AS venue_whip,

    -- Career K9 at this venue
    AVG(CASE WHEN sv.ip > 0
             THEN 9.0 * sv.k / sv.ip
             ELSE NULL END) OVER w                                           AS venue_k9,

    -- Career FIP at this venue: (13*HR + 3*BB - 2*K) / IP + 3.2
    AVG(CASE WHEN sv.ip > 0
             THEN (13.0 * sv.hr + 3.0 * sv.bb - 2.0 * sv.k) / sv.ip + 3.2
             ELSE NULL END) OVER w                                           AS venue_fip,

    -- Average IP per start at this venue (workload context)
    AVG(sv.ip) OVER w                                                        AS venue_ip_avg

FROM starter_venue sv
WINDOW w AS (
    PARTITION BY sv.player_id, sv.venue_id
    ORDER BY sv.game_date_et, sv.game_slug
    -- All prior starts at this venue before the current game
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
;