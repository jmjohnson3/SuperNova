-- MLB006: Main game feature views for training and prediction
-- Two views:
--   features.mlb_game_training_features   -- completed games (status='final')
--   features.mlb_game_prediction_features -- upcoming games (status IN ('scheduled','in_progress'))
--
-- Both views join:
--   MLB001: team batting rolling (home + away)
--   MLB002: team pitching rolling (home + away)
--   MLB003: individual SP rolling (home_sp + away_sp)
--   MLB004: ballpark factors
--   MLB005: standings + rest days (home + away)
--   odds.mlb_game_lines: market run line + total (LEFT JOIN)
--
-- Column naming convention:
--   home_* / away_* for team-level features
--   home_sp_* / away_sp_* for starting pitcher features

-- ============================================================
-- TRAINING VIEW: one row per completed game
-- ============================================================
CREATE OR REPLACE VIEW features.mlb_game_training_features AS
WITH
-- Best (most recent) odds line per game
market_lines AS (
    SELECT DISTINCT ON (home_team, away_team, as_of_date)
        home_team,
        away_team,
        as_of_date,
        spread_home_points  AS run_line_home,
        spread_home_price   AS run_line_home_price,
        spread_away_price   AS run_line_away_price,
        total_points        AS total_line,
        total_over_price    AS over_price,
        total_under_price   AS under_price
    FROM odds.mlb_game_lines
    WHERE bookmaker_key IN ('draftkings', 'fanduel')
    ORDER BY
        home_team,
        away_team,
        as_of_date,
        -- Prefer fanduel, then draftkings
        CASE bookmaker_key WHEN 'fanduel' THEN 0 ELSE 1 END
),
-- Opening line: earliest crawl per game
market_lines_open AS (
    SELECT DISTINCT ON (home_team, away_team, as_of_date)
        home_team,
        away_team,
        as_of_date,
        spread_home_points  AS open_run_line_home,
        total_points        AS open_total_line
    FROM odds.mlb_game_lines
    WHERE bookmaker_key IN ('draftkings', 'fanduel')
    ORDER BY
        home_team,
        away_team,
        as_of_date,
        -- Oldest crawl first for opening line
        event_id ASC
)
SELECT
    -- ---- Game identifiers ----
    g.game_slug,
    g.season,
    g.game_date_et,
    g.start_ts_utc,
    g.home_team_abbr,
    g.away_team_abbr,
    g.venue_id,
    v.name                                  AS venue_name,

    -- ---- Targets (training only) ----
    g.home_score,
    g.away_score,
    g.home_score - g.away_score            AS run_diff,        -- positive = home win
    g.home_score + g.away_score            AS total_runs,
    CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_win,

    -- ---- Market lines ----
    ml.run_line_home,
    ml.run_line_home_price,
    ml.run_line_away_price,
    ml.total_line,
    ml.over_price,
    ml.under_price,
    mlo.open_run_line_home,
    mlo.open_total_line,
    ml.total_line - mlo.open_total_line     AS total_line_move,
    ml.run_line_home - mlo.open_run_line_home AS run_line_move,

    -- ---- Ballpark factors ----
    bf.run_factor                           AS park_run_factor,
    bf.hr_factor                            AS park_hr_factor,

    -- ---- Home team batting rolling ----
    hb.runs_avg_5                           AS home_runs_avg_5,
    hb.hr_avg_5                             AS home_hr_avg_5,
    hb.avg_avg_5                            AS home_avg_avg_5,
    hb.obp_avg_5                            AS home_obp_avg_5,
    hb.slg_avg_5                            AS home_slg_avg_5,
    hb.iso_avg_5                            AS home_iso_avg_5,
    hb.k_pct_avg_5                          AS home_k_pct_avg_5,
    hb.bb_pct_avg_5                         AS home_bb_pct_avg_5,
    hb.runs_avg_10                          AS home_runs_avg_10,
    hb.hr_avg_10                            AS home_hr_avg_10,
    hb.avg_avg_10                           AS home_avg_avg_10,
    hb.obp_avg_10                           AS home_obp_avg_10,
    hb.slg_avg_10                           AS home_slg_avg_10,
    hb.iso_avg_10                           AS home_iso_avg_10,
    hb.k_pct_avg_10                         AS home_k_pct_avg_10,
    hb.bb_pct_avg_10                        AS home_bb_pct_avg_10,
    hb.runs_avg_20                          AS home_runs_avg_20,
    hb.runs_sd_10                           AS home_runs_sd_10,

    -- ---- Away team batting rolling ----
    ab.runs_avg_5                           AS away_runs_avg_5,
    ab.hr_avg_5                             AS away_hr_avg_5,
    ab.avg_avg_5                            AS away_avg_avg_5,
    ab.obp_avg_5                            AS away_obp_avg_5,
    ab.slg_avg_5                            AS away_slg_avg_5,
    ab.iso_avg_5                            AS away_iso_avg_5,
    ab.k_pct_avg_5                          AS away_k_pct_avg_5,
    ab.bb_pct_avg_5                         AS away_bb_pct_avg_5,
    ab.runs_avg_10                          AS away_runs_avg_10,
    ab.hr_avg_10                            AS away_hr_avg_10,
    ab.avg_avg_10                           AS away_avg_avg_10,
    ab.obp_avg_10                           AS away_obp_avg_10,
    ab.slg_avg_10                           AS away_slg_avg_10,
    ab.iso_avg_10                           AS away_iso_avg_10,
    ab.k_pct_avg_10                         AS away_k_pct_avg_10,
    ab.bb_pct_avg_10                        AS away_bb_pct_avg_10,
    ab.runs_avg_20                          AS away_runs_avg_20,
    ab.runs_sd_10                           AS away_runs_sd_10,

    -- ---- Home team pitching rolling ----
    hp.runs_allowed_avg_5                   AS home_runs_allowed_avg_5,
    hp.era_5                                AS home_era_5,
    hp.whip_5                               AS home_whip_5,
    hp.k9_5                                 AS home_k9_5,
    hp.bb9_5                                AS home_bb9_5,
    hp.hr9_5                                AS home_hr9_5,
    hp.sp_era_5                             AS home_sp_era_5,
    hp.sp_whip_5                            AS home_sp_whip_5,
    hp.sp_k9_5                              AS home_team_sp_k9_5,
    hp.bp_era_5                             AS home_bp_era_5,
    hp.bp_whip_5                            AS home_bp_whip_5,
    hp.runs_allowed_avg_10                  AS home_runs_allowed_avg_10,
    hp.era_10                               AS home_era_10,
    hp.whip_10                              AS home_whip_10,
    hp.k9_10                                AS home_k9_10,
    hp.bb9_10                               AS home_bb9_10,
    hp.hr9_10                               AS home_hr9_10,
    hp.sp_era_10                            AS home_sp_era_10,
    hp.sp_whip_10                           AS home_sp_whip_10,
    hp.sp_k9_10                             AS home_team_sp_k9_10,
    hp.bp_era_10                            AS home_bp_era_10,
    hp.bp_whip_10                           AS home_bp_whip_10,
    hp.bullpen_ip_last_3                    AS home_bullpen_ip_last_3,

    -- ---- Away team pitching rolling ----
    ap.runs_allowed_avg_5                   AS away_runs_allowed_avg_5,
    ap.era_5                                AS away_era_5,
    ap.whip_5                               AS away_whip_5,
    ap.k9_5                                 AS away_k9_5,
    ap.bb9_5                                AS away_bb9_5,
    ap.hr9_5                                AS away_hr9_5,
    ap.sp_era_5                             AS away_sp_era_5,
    ap.sp_whip_5                            AS away_sp_whip_5,
    ap.sp_k9_5                              AS away_team_sp_k9_5,
    ap.bp_era_5                             AS away_bp_era_5,
    ap.bp_whip_5                            AS away_bp_whip_5,
    ap.runs_allowed_avg_10                  AS away_runs_allowed_avg_10,
    ap.era_10                               AS away_era_10,
    ap.whip_10                              AS away_whip_10,
    ap.k9_10                                AS away_k9_10,
    ap.bb9_10                               AS away_bb9_10,
    ap.hr9_10                               AS away_hr9_10,
    ap.sp_era_10                            AS away_sp_era_10,
    ap.sp_whip_10                           AS away_sp_whip_10,
    ap.sp_k9_10                             AS away_team_sp_k9_10,
    ap.bp_era_10                            AS away_bp_era_10,
    ap.bp_whip_10                           AS away_bp_whip_10,
    ap.bullpen_ip_last_3                    AS away_bullpen_ip_last_3,

    -- ---- Home SP rolling ----
    hsp.era_5                               AS home_sp_career_era_5,
    hsp.whip_5                              AS home_sp_career_whip_5,
    hsp.k_pct_5                             AS home_sp_k_pct_5,
    hsp.bb_pct_5                            AS home_sp_bb_pct_5,
    hsp.fip_5                               AS home_sp_fip_5,
    hsp.k9_5                                AS home_sp_k9_5,
    hsp.bb9_5                               AS home_sp_bb9_5,
    hsp.hr9_5                               AS home_sp_hr9_5,
    hsp.ip_avg_5                            AS home_sp_ip_avg_5,
    hsp.era_10                              AS home_sp_career_era_10,
    hsp.fip_10                              AS home_sp_fip_10,
    hsp.starts_in_window_5                  AS home_sp_starts_in_window,

    -- ---- Away SP rolling ----
    asp.era_5                               AS away_sp_career_era_5,
    asp.whip_5                              AS away_sp_career_whip_5,
    asp.k_pct_5                             AS away_sp_k_pct_5,
    asp.bb_pct_5                            AS away_sp_bb_pct_5,
    asp.fip_5                               AS away_sp_fip_5,
    asp.k9_5                                AS away_sp_k9_5,
    asp.bb9_5                               AS away_sp_bb9_5,
    asp.hr9_5                               AS away_sp_hr9_5,
    asp.ip_avg_5                            AS away_sp_ip_avg_5,
    asp.era_10                              AS away_sp_career_era_10,
    asp.fip_10                              AS away_sp_fip_10,
    asp.starts_in_window_5                  AS away_sp_starts_in_window,

    -- ---- Standings + rest (home) ----
    hsr.wins                                AS home_wins,
    hsr.losses                              AS home_losses,
    hsr.win_pct                             AS home_win_pct,
    hsr.run_diff                            AS home_run_diff,
    hsr.run_diff_per_game                   AS home_run_diff_per_game,
    hsr.division_rank                       AS home_division_rank,
    hsr.rest_days                           AS home_rest_days,
    hsr.is_b2b                              AS home_is_b2b,
    hsr.games_played                        AS home_games_played,

    -- ---- Standings + rest (away) ----
    asr.wins                                AS away_wins,
    asr.losses                              AS away_losses,
    asr.win_pct                             AS away_win_pct,
    asr.run_diff                            AS away_run_diff,
    asr.run_diff_per_game                   AS away_run_diff_per_game,
    asr.division_rank                       AS away_division_rank,
    asr.rest_days                           AS away_rest_days,
    asr.is_b2b                              AS away_is_b2b,
    asr.games_played                        AS away_games_played,

    -- ---- Derived differential features ----
    -- Batting edge
    hb.runs_avg_10 - ab.runs_avg_10         AS runs_scored_edge_10,
    hb.obp_avg_10  - ab.obp_avg_10          AS obp_edge_10,
    hb.slg_avg_10  - ab.slg_avg_10          AS slg_edge_10,
    -- Pitching edge (lower ERA = better, so away_era - home_era = positive for home)
    ap.era_10    - hp.era_10                AS era_edge_10,
    ap.whip_10   - hp.whip_10              AS whip_edge_10,
    -- SP matchup edge
    asp.fip_5    - hsp.fip_5               AS sp_fip_edge_5,
    asp.era_5    - hsp.era_5               AS sp_era_edge_5,
    -- Bullpen fatigue edge (home bullpen less taxed = positive for home)
    ap.bullpen_ip_last_3 - hp.bullpen_ip_last_3 AS bullpen_fatigue_edge,
    -- Rest advantage
    hsr.rest_days - asr.rest_days           AS rest_days_advantage,
    -- Win pct edge
    hsr.win_pct   - asr.win_pct            AS win_pct_edge,
    hsr.run_diff_per_game - asr.run_diff_per_game AS run_diff_per_game_edge,
    -- Division rank edge (lower = better, so away - home = positive for home)
    COALESCE(asr.division_rank, 5) - COALESCE(hsr.division_rank, 5) AS division_rank_edge,
    -- Offense vs defense: home bats vs away pitching (runs_avg_10 + allowed_avg_10) / 2 = implied total
    (hb.runs_avg_10 + ap.runs_allowed_avg_10) / 2.0  AS home_implied_runs,
    (ab.runs_avg_10 + hp.runs_allowed_avg_10) / 2.0  AS away_implied_runs,
    -- Park-adjusted implied total
    bf.run_factor * ((hb.runs_avg_10 + ap.runs_allowed_avg_10)
                   + (ab.runs_avg_10 + hp.runs_allowed_avg_10)) / 2.0 AS park_adj_implied_total

FROM raw.mlb_games g

-- Venue info
LEFT JOIN raw.mlb_venues v
    ON v.venue_id = g.venue_id

-- Ballpark factors (join by home team since they always play in their home park)
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr

-- Home team batting rolling
LEFT JOIN features.mlb_team_batting_rolling hb
    ON hb.game_slug = g.game_slug
   AND hb.team_abbr = g.home_team_abbr

-- Away team batting rolling
LEFT JOIN features.mlb_team_batting_rolling ab
    ON ab.game_slug = g.game_slug
   AND ab.team_abbr = g.away_team_abbr

-- Home team pitching rolling
LEFT JOIN features.mlb_team_pitching_rolling hp
    ON hp.game_slug = g.game_slug
   AND hp.team_abbr = g.home_team_abbr

-- Away team pitching rolling
LEFT JOIN features.mlb_team_pitching_rolling ap
    ON ap.game_slug = g.game_slug
   AND ap.team_abbr = g.away_team_abbr

-- Home starting pitcher (from scheduled SP table)
LEFT JOIN raw.mlb_starting_pitchers hsp_sched
    ON hsp_sched.game_slug = g.game_slug
   AND hsp_sched.team_abbr = g.home_team_abbr

-- Home SP rolling stats
LEFT JOIN features.mlb_pitcher_rolling hsp
    ON hsp.game_slug  = g.game_slug
   AND hsp.player_id  = COALESCE(hsp_sched.player_id, g.home_sp_id)

-- Away starting pitcher
LEFT JOIN raw.mlb_starting_pitchers asp_sched
    ON asp_sched.game_slug = g.game_slug
   AND asp_sched.team_abbr = g.away_team_abbr

-- Away SP rolling stats
LEFT JOIN features.mlb_pitcher_rolling asp
    ON asp.game_slug  = g.game_slug
   AND asp.player_id  = COALESCE(asp_sched.player_id, g.away_sp_id)

-- Home standings + rest
LEFT JOIN features.mlb_standings_rest hsr
    ON hsr.game_slug = g.game_slug
   AND hsr.team_abbr = g.home_team_abbr

-- Away standings + rest
LEFT JOIN features.mlb_standings_rest asr
    ON asr.game_slug = g.game_slug
   AND asr.team_abbr = g.away_team_abbr

-- Market lines (most recent line before game time)
LEFT JOIN market_lines ml
    ON ml.home_team = g.home_team_abbr
   AND ml.away_team = g.away_team_abbr
   AND ml.as_of_date     = g.game_date_et

LEFT JOIN market_lines_open mlo
    ON mlo.home_team = g.home_team_abbr
   AND mlo.away_team = g.away_team_abbr
   AND mlo.as_of_date     = g.game_date_et

WHERE g.status = 'final'
  AND g.home_score IS NOT NULL
  AND g.away_score IS NOT NULL
;

-- ============================================================
-- PREDICTION VIEW: today's and tomorrow's games
-- ============================================================
CREATE OR REPLACE VIEW features.mlb_game_prediction_features AS
WITH
market_lines AS (
    SELECT DISTINCT ON (home_team, away_team, as_of_date)
        home_team,
        away_team,
        as_of_date,
        spread_home_points  AS run_line_home,
        spread_home_price   AS run_line_home_price,
        spread_away_price   AS run_line_away_price,
        total_points        AS total_line,
        total_over_price    AS over_price,
        total_under_price   AS under_price
    FROM odds.mlb_game_lines
    WHERE bookmaker_key IN ('draftkings', 'fanduel')
    ORDER BY
        home_team,
        away_team,
        as_of_date,
        CASE bookmaker_key WHEN 'fanduel' THEN 0 ELSE 1 END
)
SELECT
    -- ---- Game identifiers ----
    g.game_slug,
    g.season,
    g.game_date_et,
    g.start_ts_utc,
    g.home_team_abbr,
    g.away_team_abbr,
    g.venue_id,
    v.name                                  AS venue_name,
    g.status,

    -- ---- Market lines ----
    ml.run_line_home,
    ml.run_line_home_price,
    ml.run_line_away_price,
    ml.total_line,
    ml.over_price,
    ml.under_price,

    -- ---- Ballpark factors ----
    bf.run_factor                           AS park_run_factor,
    bf.hr_factor                            AS park_hr_factor,

    -- ---- Home team batting rolling ----
    hb.runs_avg_5                           AS home_runs_avg_5,
    hb.hr_avg_5                             AS home_hr_avg_5,
    hb.avg_avg_5                            AS home_avg_avg_5,
    hb.obp_avg_5                            AS home_obp_avg_5,
    hb.slg_avg_5                            AS home_slg_avg_5,
    hb.iso_avg_5                            AS home_iso_avg_5,
    hb.k_pct_avg_5                          AS home_k_pct_avg_5,
    hb.bb_pct_avg_5                         AS home_bb_pct_avg_5,
    hb.runs_avg_10                          AS home_runs_avg_10,
    hb.hr_avg_10                            AS home_hr_avg_10,
    hb.avg_avg_10                           AS home_avg_avg_10,
    hb.obp_avg_10                           AS home_obp_avg_10,
    hb.slg_avg_10                           AS home_slg_avg_10,
    hb.iso_avg_10                           AS home_iso_avg_10,
    hb.k_pct_avg_10                         AS home_k_pct_avg_10,
    hb.bb_pct_avg_10                        AS home_bb_pct_avg_10,
    hb.runs_avg_20                          AS home_runs_avg_20,
    hb.runs_sd_10                           AS home_runs_sd_10,

    -- ---- Away team batting rolling ----
    ab.runs_avg_5                           AS away_runs_avg_5,
    ab.hr_avg_5                             AS away_hr_avg_5,
    ab.avg_avg_5                            AS away_avg_avg_5,
    ab.obp_avg_5                            AS away_obp_avg_5,
    ab.slg_avg_5                            AS away_slg_avg_5,
    ab.iso_avg_5                            AS away_iso_avg_5,
    ab.k_pct_avg_5                          AS away_k_pct_avg_5,
    ab.bb_pct_avg_5                         AS away_bb_pct_avg_5,
    ab.runs_avg_10                          AS away_runs_avg_10,
    ab.hr_avg_10                            AS away_hr_avg_10,
    ab.avg_avg_10                           AS away_avg_avg_10,
    ab.obp_avg_10                           AS away_obp_avg_10,
    ab.slg_avg_10                           AS away_slg_avg_10,
    ab.iso_avg_10                           AS away_iso_avg_10,
    ab.k_pct_avg_10                         AS away_k_pct_avg_10,
    ab.bb_pct_avg_10                        AS away_bb_pct_avg_10,
    ab.runs_avg_20                          AS away_runs_avg_20,
    ab.runs_sd_10                           AS away_runs_sd_10,

    -- ---- Home team pitching rolling ----
    hp.runs_allowed_avg_5                   AS home_runs_allowed_avg_5,
    hp.era_5                                AS home_era_5,
    hp.whip_5                               AS home_whip_5,
    hp.k9_5                                 AS home_k9_5,
    hp.bb9_5                                AS home_bb9_5,
    hp.hr9_5                                AS home_hr9_5,
    hp.sp_era_5                             AS home_sp_era_5,
    hp.sp_whip_5                            AS home_sp_whip_5,
    hp.sp_k9_5                              AS home_team_sp_k9_5,
    hp.bp_era_5                             AS home_bp_era_5,
    hp.bp_whip_5                            AS home_bp_whip_5,
    hp.runs_allowed_avg_10                  AS home_runs_allowed_avg_10,
    hp.era_10                               AS home_era_10,
    hp.whip_10                              AS home_whip_10,
    hp.k9_10                                AS home_k9_10,
    hp.bb9_10                               AS home_bb9_10,
    hp.hr9_10                               AS home_hr9_10,
    hp.sp_era_10                            AS home_sp_era_10,
    hp.sp_whip_10                           AS home_sp_whip_10,
    hp.sp_k9_10                             AS home_team_sp_k9_10,
    hp.bp_era_10                            AS home_bp_era_10,
    hp.bp_whip_10                           AS home_bp_whip_10,
    hp.bullpen_ip_last_3                    AS home_bullpen_ip_last_3,

    -- ---- Away team pitching rolling ----
    ap.runs_allowed_avg_5                   AS away_runs_allowed_avg_5,
    ap.era_5                                AS away_era_5,
    ap.whip_5                               AS away_whip_5,
    ap.k9_5                                 AS away_k9_5,
    ap.bb9_5                                AS away_bb9_5,
    ap.hr9_5                                AS away_hr9_5,
    ap.sp_era_5                             AS away_sp_era_5,
    ap.sp_whip_5                            AS away_sp_whip_5,
    ap.sp_k9_5                              AS away_team_sp_k9_5,
    ap.bp_era_5                             AS away_bp_era_5,
    ap.bp_whip_5                            AS away_bp_whip_5,
    ap.runs_allowed_avg_10                  AS away_runs_allowed_avg_10,
    ap.era_10                               AS away_era_10,
    ap.whip_10                              AS away_whip_10,
    ap.k9_10                                AS away_k9_10,
    ap.bb9_10                               AS away_bb9_10,
    ap.hr9_10                               AS away_hr9_10,
    ap.sp_era_10                            AS away_sp_era_10,
    ap.sp_whip_10                           AS away_sp_whip_10,
    ap.sp_k9_10                             AS away_team_sp_k9_10,
    ap.bp_era_10                            AS away_bp_era_10,
    ap.bp_whip_10                           AS away_bp_whip_10,
    ap.bullpen_ip_last_3                    AS away_bullpen_ip_last_3,

    -- ---- Home SP rolling ----
    hsp.era_5                               AS home_sp_career_era_5,
    hsp.whip_5                              AS home_sp_career_whip_5,
    hsp.k_pct_5                             AS home_sp_k_pct_5,
    hsp.bb_pct_5                            AS home_sp_bb_pct_5,
    hsp.fip_5                               AS home_sp_fip_5,
    hsp.k9_5                                AS home_sp_k9_5,
    hsp.bb9_5                               AS home_sp_bb9_5,
    hsp.hr9_5                               AS home_sp_hr9_5,
    hsp.ip_avg_5                            AS home_sp_ip_avg_5,
    hsp.era_10                              AS home_sp_career_era_10,
    hsp.fip_10                              AS home_sp_fip_10,
    hsp.starts_in_window_5                  AS home_sp_starts_in_window,

    -- ---- Away SP rolling ----
    asp.era_5                               AS away_sp_career_era_5,
    asp.whip_5                              AS away_sp_career_whip_5,
    asp.k_pct_5                             AS away_sp_k_pct_5,
    asp.bb_pct_5                            AS away_sp_bb_pct_5,
    asp.fip_5                               AS away_sp_fip_5,
    asp.k9_5                                AS away_sp_k9_5,
    asp.bb9_5                               AS away_sp_bb9_5,
    asp.hr9_5                               AS away_sp_hr9_5,
    asp.ip_avg_5                            AS away_sp_ip_avg_5,
    asp.era_10                              AS away_sp_career_era_10,
    asp.fip_10                              AS away_sp_fip_10,
    asp.starts_in_window_5                  AS away_sp_starts_in_window,

    -- ---- Standings + rest (home) ----
    hsr.wins                                AS home_wins,
    hsr.losses                              AS home_losses,
    hsr.win_pct                             AS home_win_pct,
    hsr.run_diff                            AS home_run_diff,
    hsr.run_diff_per_game                   AS home_run_diff_per_game,
    hsr.division_rank                       AS home_division_rank,
    hsr.rest_days                           AS home_rest_days,
    hsr.is_b2b                              AS home_is_b2b,
    hsr.games_played                        AS home_games_played,

    -- ---- Standings + rest (away) ----
    asr.wins                                AS away_wins,
    asr.losses                              AS away_losses,
    asr.win_pct                             AS away_win_pct,
    asr.run_diff                            AS away_run_diff,
    asr.run_diff_per_game                   AS away_run_diff_per_game,
    asr.division_rank                       AS away_division_rank,
    asr.rest_days                           AS away_rest_days,
    asr.is_b2b                              AS away_is_b2b,
    asr.games_played                        AS away_games_played,

    -- ---- Derived differential features ----
    hb.runs_avg_10 - ab.runs_avg_10         AS runs_scored_edge_10,
    hb.obp_avg_10  - ab.obp_avg_10          AS obp_edge_10,
    hb.slg_avg_10  - ab.slg_avg_10          AS slg_edge_10,
    ap.era_10    - hp.era_10                AS era_edge_10,
    ap.whip_10   - hp.whip_10              AS whip_edge_10,
    asp.fip_5    - hsp.fip_5               AS sp_fip_edge_5,
    asp.era_5    - hsp.era_5               AS sp_era_edge_5,
    ap.bullpen_ip_last_3 - hp.bullpen_ip_last_3 AS bullpen_fatigue_edge,
    hsr.rest_days - asr.rest_days           AS rest_days_advantage,
    hsr.win_pct   - asr.win_pct            AS win_pct_edge,
    hsr.run_diff_per_game - asr.run_diff_per_game AS run_diff_per_game_edge,
    COALESCE(asr.division_rank, 5) - COALESCE(hsr.division_rank, 5) AS division_rank_edge,
    (hb.runs_avg_10 + ap.runs_allowed_avg_10) / 2.0  AS home_implied_runs,
    (ab.runs_avg_10 + hp.runs_allowed_avg_10) / 2.0  AS away_implied_runs,
    bf.run_factor * ((hb.runs_avg_10 + ap.runs_allowed_avg_10)
                   + (ab.runs_avg_10 + hp.runs_allowed_avg_10)) / 2.0 AS park_adj_implied_total

FROM raw.mlb_games g

LEFT JOIN raw.mlb_venues v
    ON v.venue_id = g.venue_id

-- Ballpark factors (join by home team since they always play in their home park)
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_batting_rolling hb
    ON hb.game_slug = g.game_slug
   AND hb.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_batting_rolling ab
    ON ab.game_slug = g.game_slug
   AND ab.team_abbr = g.away_team_abbr

LEFT JOIN features.mlb_team_pitching_rolling hp
    ON hp.game_slug = g.game_slug
   AND hp.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_pitching_rolling ap
    ON ap.game_slug = g.game_slug
   AND ap.team_abbr = g.away_team_abbr

LEFT JOIN raw.mlb_starting_pitchers hsp_sched
    ON hsp_sched.game_slug = g.game_slug
   AND hsp_sched.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_pitcher_rolling hsp
    ON hsp.game_slug  = g.game_slug
   AND hsp.player_id  = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN raw.mlb_starting_pitchers asp_sched
    ON asp_sched.game_slug = g.game_slug
   AND asp_sched.team_abbr = g.away_team_abbr

LEFT JOIN features.mlb_pitcher_rolling asp
    ON asp.game_slug  = g.game_slug
   AND asp.player_id  = COALESCE(asp_sched.player_id, g.away_sp_id)

LEFT JOIN features.mlb_standings_rest hsr
    ON hsr.game_slug = g.game_slug
   AND hsr.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_standings_rest asr
    ON asr.game_slug = g.game_slug
   AND asr.team_abbr = g.away_team_abbr

LEFT JOIN market_lines ml
    ON ml.home_team  = g.home_team_abbr
   AND ml.away_team  = g.away_team_abbr
   AND ml.as_of_date = g.game_date_et

WHERE g.status IN ('scheduled', 'in_progress')
  AND g.game_date_et >= CURRENT_DATE
  AND g.game_date_et <= CURRENT_DATE + INTERVAL '1 day'
;
