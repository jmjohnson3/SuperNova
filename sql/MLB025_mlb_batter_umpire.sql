-- MLB025: Batter career stats with specific home plate umpire (leakage-safe window)
-- Keys: (batter_id, umpire_id, game_slug)

CREATE OR REPLACE VIEW features.mlb_batter_umpire AS
WITH game_ump_batter AS (
    SELECT
        gl.player_id                       AS batter_id,
        u.umpire_id,
        g.game_slug,
        g.game_date_et,
        COALESCE(gl.at_bats, 0)           AS ab,
        COALESCE(gl.hits, 0)              AS h,
        COALESCE(gl.walks_batter, 0)      AS bb,
        COALESCE(gl.strikeouts_batter, 0) AS k
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    JOIN raw.mlb_game_umpires u
        ON u.game_slug = g.game_slug AND u.ump_position = 'Home Plate'
    WHERE g.status = 'final' AND gl.at_bats > 0
),
rolling AS (
    SELECT
        batter_id, umpire_id, game_slug, game_date_et,
        COUNT(*)                             OVER w AS btu_games,
        COALESCE(SUM(ab) OVER w, 0)                 AS btu_ab,
        COALESCE(SUM(h)  OVER w, 0)                 AS btu_h,
        COALESCE(SUM(bb) OVER w, 0)                 AS btu_bb,
        COALESCE(SUM(k)  OVER w, 0)                 AS btu_k
    FROM game_ump_batter
    WINDOW w AS (PARTITION BY batter_id, umpire_id ORDER BY game_date_et, game_slug
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    batter_id, umpire_id, game_slug, game_date_et,
    btu_games, btu_ab,
    CASE WHEN btu_ab > 0 THEN btu_h::float  / btu_ab               ELSE NULL END AS btu_ba,
    CASE WHEN btu_ab > 0 THEN btu_k::float  / btu_ab               ELSE NULL END AS btu_k_rate,
    CASE WHEN btu_ab + btu_bb > 0
         THEN btu_bb::float / (btu_ab + btu_bb)                    ELSE NULL END AS btu_bb_rate
FROM rolling;
