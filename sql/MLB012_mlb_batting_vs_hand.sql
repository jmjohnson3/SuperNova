-- MLB012: Batter vs SP handedness splits
-- Adds raw.mlb_player_handedness lookup table (bat_side / pitch_hand)
-- and features.mlb_batting_vs_hand view with rolling 40-game splits
-- vs LHP and RHP opponents.
-- Leakage-safe: ROWS BETWEEN 40 PRECEDING AND 1 PRECEDING.

-- ============================================================
-- 1. Handedness lookup table
-- ============================================================
CREATE TABLE IF NOT EXISTS raw.mlb_player_handedness (
    player_id   INTEGER PRIMARY KEY,
    bat_side    TEXT,       -- L, R, S (switch)
    pitch_hand  TEXT,       -- L, R
    updated_at  TIMESTAMPTZ DEFAULT now()
);


-- ============================================================
-- 2. One-SP-per-game helper (re-used by view below)
-- ============================================================

-- ============================================================
-- 3. Batter vs SP handedness rolling view
-- ============================================================
CREATE OR REPLACE VIEW features.mlb_batting_vs_hand AS
WITH best_sp AS (
    -- Pick the single best SP record per game/team:
    -- priority: actual sources first, then probable; break ties by player_id.
    SELECT DISTINCT ON (game_slug, team_abbr)
        game_slug,
        team_abbr,
        player_id AS sp_player_id
    FROM raw.mlb_starting_pitchers
    WHERE player_id IS NOT NULL
    ORDER BY
        game_slug,
        team_abbr,
        CASE source
            WHEN 'actual'   THEN 0
            WHEN 'probable' THEN 1
            ELSE 2
        END,
        player_id
),
batter_games AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.player_id,
        gl.team_abbr,
        COALESCE(gl.at_bats,           0) AS ab,
        COALESCE(gl.hits,              0) AS h,
        COALESCE(gl.home_runs,         0) AS hr,
        COALESCE(gl.total_bases,       0) AS tb,
        COALESCE(gl.walks_batter,      0) AS bb,
        COALESCE(gl.strikeouts_batter, 0) AS k_bat,
        ph.pitch_hand                     AS opp_sp_hand
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = g.game_slug
    -- Look up the opposing team's starting pitcher
    JOIN best_sp sp
        ON  sp.game_slug  = g.game_slug
        AND sp.team_abbr  = CASE
                                WHEN gl.team_abbr = g.home_team_abbr THEN g.away_team_abbr
                                ELSE g.home_team_abbr
                            END
    LEFT JOIN raw.mlb_player_handedness ph
        ON ph.player_id = sp.sp_player_id
    WHERE g.status        = 'final'
      AND g.home_score    IS NOT NULL
      AND gl.at_bats      IS NOT NULL
      AND gl.at_bats      > 0
),
derived AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        player_id,
        team_abbr,
        ab,
        h,
        hr,
        tb,
        bb,
        k_bat,
        opp_sp_hand,
        CASE WHEN ab > 0 THEN h::float / ab              ELSE NULL END AS game_avg,
        CASE WHEN ab > 0 THEN (tb - h)::float / ab       ELSE NULL END AS game_iso,
        CASE WHEN ab > 0 THEN k_bat::float / ab          ELSE NULL END AS game_k_rate,
        CASE WHEN (ab + bb) > 0 THEN bb::float / (ab+bb) ELSE NULL END AS game_bb_rate
    FROM batter_games
)
SELECT
    d.season,
    d.player_id,
    d.team_abbr,
    d.game_slug,
    d.game_date_et,
    d.opp_sp_hand,

    -- Rolling 40-game vs LHP (FILTER keeps only LHP-opponent rows in window)
    AVG(d.h)            FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS hits_avg_40_vs_lhp,
    AVG(d.tb)           FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS tb_avg_40_vs_lhp,
    AVG(d.hr)           FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS hr_avg_40_vs_lhp,
    AVG(d.game_k_rate)  FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS k_rate_avg_40_vs_lhp,
    AVG(d.game_bb_rate) FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS bb_rate_avg_40_vs_lhp,
    AVG(d.game_iso)     FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS iso_avg_40_vs_lhp,
    AVG(d.game_avg)     FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS avg_avg_40_vs_lhp,
    COUNT(*)            FILTER (WHERE d.opp_sp_hand = 'L') OVER w40  AS n_games_vs_lhp_40,

    -- Rolling 40-game vs RHP
    AVG(d.h)            FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS hits_avg_40_vs_rhp,
    AVG(d.tb)           FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS tb_avg_40_vs_rhp,
    AVG(d.hr)           FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS hr_avg_40_vs_rhp,
    AVG(d.game_k_rate)  FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS k_rate_avg_40_vs_rhp,
    AVG(d.game_bb_rate) FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS bb_rate_avg_40_vs_rhp,
    AVG(d.game_iso)     FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS iso_avg_40_vs_rhp,
    AVG(d.game_avg)     FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS avg_avg_40_vs_rhp,
    COUNT(*)            FILTER (WHERE d.opp_sp_hand = 'R') OVER w40  AS n_games_vs_rhp_40,

    -- Batter's own handedness (same for every row of a player; passed through for OHE)
    bh.bat_side,

    -- Rolling 10-game vs LHP (recent form — faster response to in-season handedness shifts)
    -- NOTE: appended after bat_side to satisfy CREATE OR REPLACE VIEW column-order rules
    AVG(d.h)           FILTER (WHERE d.opp_sp_hand = 'L') OVER w10 AS hits_avg_10_vs_lhp,
    AVG(d.tb)          FILTER (WHERE d.opp_sp_hand = 'L') OVER w10 AS tb_avg_10_vs_lhp,
    AVG(d.hr)          FILTER (WHERE d.opp_sp_hand = 'L') OVER w10 AS hr_avg_10_vs_lhp,
    AVG(d.game_k_rate) FILTER (WHERE d.opp_sp_hand = 'L') OVER w10 AS k_rate_avg_10_vs_lhp,
    AVG(d.game_iso)    FILTER (WHERE d.opp_sp_hand = 'L') OVER w10 AS iso_avg_10_vs_lhp,
    COUNT(*)           FILTER (WHERE d.opp_sp_hand = 'L') OVER w10 AS n_games_vs_lhp_10,
    -- Rolling 10-game vs RHP
    AVG(d.h)           FILTER (WHERE d.opp_sp_hand = 'R') OVER w10 AS hits_avg_10_vs_rhp,
    AVG(d.tb)          FILTER (WHERE d.opp_sp_hand = 'R') OVER w10 AS tb_avg_10_vs_rhp,
    AVG(d.hr)          FILTER (WHERE d.opp_sp_hand = 'R') OVER w10 AS hr_avg_10_vs_rhp,
    AVG(d.game_k_rate) FILTER (WHERE d.opp_sp_hand = 'R') OVER w10 AS k_rate_avg_10_vs_rhp,
    AVG(d.game_iso)    FILTER (WHERE d.opp_sp_hand = 'R') OVER w10 AS iso_avg_10_vs_rhp,
    COUNT(*)           FILTER (WHERE d.opp_sp_hand = 'R') OVER w10 AS n_games_vs_rhp_10

FROM derived d
LEFT JOIN raw.mlb_player_handedness bh ON bh.player_id = d.player_id

WINDOW
    w40 AS (
        PARTITION BY d.season, d.player_id
        ORDER BY     d.game_date_et, d.game_slug
        ROWS BETWEEN 40 PRECEDING AND 1 PRECEDING
    ),
    w10 AS (
        PARTITION BY d.season, d.player_id
        ORDER BY     d.game_date_et, d.game_slug
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    )
;
