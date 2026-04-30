-- MLB001: Team batting rolling stats
-- Leakage-safe: all windows use ROWS BETWEEN N PRECEDING AND 1 PRECEDING
CREATE OR REPLACE VIEW features.mlb_team_batting_rolling AS
WITH team_batting AS (
    SELECT
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        SUM(gl.hits)              AS team_hits,
        SUM(gl.at_bats)           AS team_ab,
        SUM(gl.home_runs)         AS team_hr,
        SUM(gl.rbi)               AS team_rbi,
        SUM(gl.walks_batter)      AS team_bb,
        SUM(gl.strikeouts_batter) AS team_k,
        SUM(gl.total_bases)       AS team_tb,
        SUM(gl.doubles)           AS team_2b,
        SUM(gl.triples)           AS team_3b,
        SUM(COALESCE(gl.stolen_bases, 0)) AS team_sb,
        CASE
            WHEN gl.team_abbr = g.home_team_abbr THEN g.home_score
            ELSE g.away_score
        END AS runs_scored
    FROM raw.mlb_games g
    JOIN raw.mlb_player_gamelogs gl
        ON gl.game_slug = g.game_slug
    WHERE g.status = 'final'
      AND g.home_score IS NOT NULL
      AND gl.at_bats IS NOT NULL
    GROUP BY
        g.season,
        g.game_slug,
        g.game_date_et,
        gl.team_abbr,
        g.home_team_abbr,
        g.home_score,
        g.away_score
),
derived AS (
    SELECT
        season,
        game_slug,
        game_date_et,
        team_abbr,
        runs_scored,
        team_hr,
        team_rbi,
        team_hits,
        team_ab,
        team_bb,
        team_k,
        team_tb,
        team_sb,
        -- Batting average: H / AB
        CASE WHEN team_ab > 0
             THEN team_hits::float / team_ab
             ELSE NULL END AS avg,
        -- On-base approx (H + BB) / (AB + BB)
        CASE WHEN (team_ab + team_bb) > 0
             THEN (team_hits + team_bb)::float / (team_ab + team_bb)
             ELSE NULL END AS obp,
        -- Slugging: TB / AB
        CASE WHEN team_ab > 0
             THEN team_tb::float / team_ab
             ELSE NULL END AS slg,
        -- ISO (Isolated Power): (TB - H) / AB
        CASE WHEN team_ab > 0
             THEN (team_tb - team_hits)::float / team_ab
             ELSE NULL END AS iso,
        -- K%: K / AB
        CASE WHEN team_ab > 0
             THEN team_k::float / team_ab
             ELSE NULL END AS k_pct,
        -- BB%: BB / (AB + BB)
        CASE WHEN (team_ab + team_bb) > 0
             THEN team_bb::float / (team_ab + team_bb)
             ELSE NULL END AS bb_pct
    FROM team_batting
)
SELECT
    season,
    team_abbr,
    game_slug,
    game_date_et,

    -- 5-game rolling windows
    AVG(runs_scored) OVER w5          AS runs_avg_5,
    AVG(team_hr)     OVER w5          AS hr_avg_5,
    AVG(avg)         OVER w5          AS avg_avg_5,
    AVG(obp)         OVER w5          AS obp_avg_5,
    AVG(slg)         OVER w5          AS slg_avg_5,
    AVG(iso)         OVER w5          AS iso_avg_5,
    AVG(k_pct)       OVER w5          AS k_pct_avg_5,
    AVG(bb_pct)      OVER w5          AS bb_pct_avg_5,
    SUM(runs_scored) OVER w5          AS runs_sum_5,

    -- 10-game rolling windows
    AVG(runs_scored)          OVER w10 AS runs_avg_10,
    AVG(team_hr)              OVER w10 AS hr_avg_10,
    AVG(avg)                  OVER w10 AS avg_avg_10,
    AVG(obp)                  OVER w10 AS obp_avg_10,
    AVG(slg)                  OVER w10 AS slg_avg_10,
    AVG(iso)                  OVER w10 AS iso_avg_10,
    AVG(k_pct)                OVER w10 AS k_pct_avg_10,
    AVG(bb_pct)               OVER w10 AS bb_pct_avg_10,
    STDDEV_SAMP(runs_scored)  OVER w10 AS runs_sd_10,

    -- 20-game rolling windows
    AVG(runs_scored) OVER w20         AS runs_avg_20,
    AVG(team_hr)     OVER w20         AS hr_avg_20,
    AVG(avg)         OVER w20         AS avg_avg_20,
    AVG(obp)         OVER w20         AS obp_avg_20,
    AVG(slg)         OVER w20         AS slg_avg_20,
    AVG(iso)         OVER w20         AS iso_avg_20,
    AVG(k_pct)       OVER w20         AS k_pct_avg_20,
    AVG(bb_pct)      OVER w20         AS bb_pct_avg_20,
    STDDEV_SAMP(runs_scored) OVER w20 AS runs_sd_20,

    -- SB rolling (appended at end — CREATE OR REPLACE VIEW requires new columns last)
    AVG(team_sb)     OVER w5          AS sb_avg_5,
    AVG(team_sb)     OVER w10         AS sb_avg_10,
    -- SB% proxy: SB / AB over last 10 games (aggressiveness signal)
    CASE WHEN SUM(team_ab) OVER w10 > 0
         THEN SUM(team_sb) OVER w10::float / NULLIF(SUM(team_ab) OVER w10, 0)
         ELSE NULL END                AS sb_pct_10

FROM derived
WINDOW
    w5  AS (PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10 AS (PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    w20 AS (PARTITION BY season, team_abbr
            ORDER BY game_date_et, game_slug
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
;
