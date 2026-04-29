-- MLB006: Main game feature views for training and prediction
-- Group C additions (MLB009/010/011):
--   Umpire:  home plate ump rolling K9/BB9/RPG (mlb_umpire_rolling_mat)
--   Weather: temperature, wind, precipitation (raw.mlb_weather)
--   Lineup:  actual batting-order quality OBP/SLG proxies (mlb_lineup_quality)
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
-- Group B additions:
--   SP: days_rest, is_short_rest, home/away ERA splits
--   Standings: wins_last_5/10, win_pct_last_5/10, run_diff_avg_last_5
--   H2H: head-to-head season record between the two teams

-- ============================================================
-- TRAINING VIEW: one row per completed game
-- ============================================================
CREATE OR REPLACE VIEW features.mlb_game_training_features AS
WITH
-- Best (most recent) odds line per game
market_lines AS (
    SELECT DISTINCT ON (home_team, away_team, as_of_date)
        home_team                   AS home_team_abbr,
        away_team                   AS away_team_abbr,
        as_of_date,
        spread_home_points          AS run_line_home,
        spread_home_price           AS run_line_home_price,
        spread_away_price           AS run_line_away_price,
        total_points                AS total_line,
        total_over_price            AS over_price,
        total_under_price           AS under_price
    FROM odds.mlb_game_lines
    WHERE bookmaker_key IN ('draftkings', 'fanduel')
    ORDER BY
        home_team,
        away_team,
        as_of_date,
        CASE bookmaker_key WHEN 'fanduel' THEN 0 ELSE 1 END
),
-- Opening line: earliest crawl per game
market_lines_open AS (
    SELECT DISTINCT ON (home_team, away_team, as_of_date)
        home_team                   AS home_team_abbr,
        away_team                   AS away_team_abbr,
        as_of_date,
        spread_home_points          AS open_run_line_home,
        total_points                AS open_total_line
    FROM odds.mlb_game_lines
    WHERE bookmaker_key IN ('draftkings', 'fanduel')
    ORDER BY
        home_team,
        away_team,
        as_of_date,
        event_id ASC
),
-- Head-to-head season record: for each game, how many prior games between
-- these two teams this season, and how many did the home team win?
h2h AS (
    SELECT
        g1.game_slug,
        COUNT(*) FILTER (WHERE g2.status = 'final')       AS h2h_games_ytd,
        SUM(CASE
            WHEN g2.home_team_abbr = g1.home_team_abbr
             AND g2.home_score > g2.away_score THEN 1
            WHEN g2.away_team_abbr = g1.home_team_abbr
             AND g2.away_score > g2.home_score THEN 1
            ELSE 0
        END) FILTER (WHERE g2.status = 'final')           AS h2h_home_team_wins
    FROM raw.mlb_games g1
    LEFT JOIN raw.mlb_games g2
        ON (   (g2.home_team_abbr = g1.home_team_abbr AND g2.away_team_abbr = g1.away_team_abbr)
            OR (g2.home_team_abbr = g1.away_team_abbr AND g2.away_team_abbr = g1.home_team_abbr))
       AND g2.game_date_et < g1.game_date_et
       AND g2.season       = g1.season
    GROUP BY g1.game_slug
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
    g.home_score - g.away_score            AS run_diff,
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
    hb.sb_avg_5                             AS home_sb_avg_5,
    hb.sb_avg_10                            AS home_sb_avg_10,
    hb.sb_pct_10                            AS home_sb_pct_10,

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
    ab.sb_avg_5                             AS away_sb_avg_5,
    ab.sb_avg_10                            AS away_sb_avg_10,
    ab.sb_pct_10                            AS away_sb_pct_10,

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
    hp.bp_k9_5                              AS home_bp_k9_5,
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
    hp.bp_k9_10                             AS home_bp_k9_10,
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
    ap.bp_k9_5                              AS away_bp_k9_5,
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
    ap.bp_k9_10                             AS away_bp_k9_10,
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
    -- Group B: SP rest + home/away splits
    hsp.days_since_last_start               AS home_sp_days_rest,
    hsp.is_short_rest                       AS home_sp_is_short_rest,
    hsp.era_home_10                         AS home_sp_era_home_10,
    hsp.era_away_10                         AS home_sp_era_away_10,
    hsp.k9_home_10                          AS home_sp_k9_home_10,
    hsp.k9_away_10                          AS home_sp_k9_away_10,
    hsp.fip_home_10                         AS home_sp_fip_home_10,
    hsp.fip_away_10                         AS home_sp_fip_away_10,
    -- SP 20-start baseline (regression-to-mean anchor)
    hsp.era_20                              AS home_sp_era_20,
    hsp.fip_20                              AS home_sp_fip_20,
    -- Last-start pitch quality (complements last_start_ip for full last-outing picture)
    hsp.last_start_k                        AS home_sp_last_k,
    hsp.last_start_bb                       AS home_sp_last_bb,
    hsp.last_start_fip                      AS home_sp_last_fip,

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
    -- Group B: SP rest + home/away splits
    asp.days_since_last_start               AS away_sp_days_rest,
    asp.is_short_rest                       AS away_sp_is_short_rest,
    asp.era_home_10                         AS away_sp_era_home_10,
    asp.era_away_10                         AS away_sp_era_away_10,
    asp.k9_home_10                          AS away_sp_k9_home_10,
    asp.k9_away_10                          AS away_sp_k9_away_10,
    asp.fip_home_10                         AS away_sp_fip_home_10,
    asp.fip_away_10                         AS away_sp_fip_away_10,
    -- SP 20-start baseline (regression-to-mean anchor)
    asp.era_20                              AS away_sp_era_20,
    asp.fip_20                              AS away_sp_fip_20,
    -- Last-start pitch quality (complements last_start_ip for full last-outing picture)
    asp.last_start_k                        AS away_sp_last_k,
    asp.last_start_bb                       AS away_sp_last_bb,
    asp.last_start_fip                      AS away_sp_last_fip,

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
    -- Group B: rolling form
    hsr.wins_last_5                         AS home_wins_last_5,
    hsr.win_pct_last_5                      AS home_win_pct_last_5,
    hsr.wins_last_10                        AS home_wins_last_10,
    hsr.win_pct_last_10                     AS home_win_pct_last_10,
    hsr.run_diff_avg_last_5                 AS home_run_diff_avg_last_5,

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
    -- Group B: rolling form
    asr.wins_last_5                         AS away_wins_last_5,
    asr.win_pct_last_5                      AS away_win_pct_last_5,
    asr.wins_last_10                        AS away_wins_last_10,
    asr.win_pct_last_10                     AS away_win_pct_last_10,
    asr.run_diff_avg_last_5                 AS away_run_diff_avg_last_5,

    -- ---- Head-to-head season record ----
    h2h.h2h_games_ytd,
    CASE WHEN h2h.h2h_games_ytd > 0
         THEN h2h.h2h_home_team_wins::float / h2h.h2h_games_ytd
         ELSE NULL END                      AS h2h_home_win_pct_ytd,

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
                   + (ab.runs_avg_10 + hp.runs_allowed_avg_10)) / 2.0 AS park_adj_implied_total,

    -- ---- Group C: Umpire rolling stats ----
    COALESCE(ump.n_ump_games_prev_5, 0)             AS n_ump_games_prev_5,
    ump.ump_rpg_5,
    ump.ump_k9_5,
    ump.ump_bb9_5,
    ump.ump_rpg_10,
    ump.ump_k9_10,
    ump.ump_bb9_10,

    -- ---- Group C: Weather ----
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0)
    END                                             AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0)
    END                                             AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0)
    END                                             AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)        AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END AS is_dome,

    -- ---- Group C: Lineup quality (home) ----
    hlq.lineup_avg_avg_10       AS home_lineup_avg_avg_10,
    hlq.lineup_slg_avg_10       AS home_lineup_slg_avg_10,
    hlq.lineup_iso_avg_10       AS home_lineup_iso_avg_10,
    hlq.top4_slg_avg_10         AS home_top4_slg_avg_10,
    hlq.lineup_data_completeness AS home_lineup_completeness,

    -- ---- Group C: Lineup quality (away) ----
    alq.lineup_avg_avg_10       AS away_lineup_avg_avg_10,
    alq.lineup_slg_avg_10       AS away_lineup_slg_avg_10,
    alq.lineup_iso_avg_10       AS away_lineup_iso_avg_10,
    alq.top4_slg_avg_10         AS away_top4_slg_avg_10,
    alq.lineup_data_completeness AS away_lineup_completeness,

    -- ---- Group C: Lineup differential features ----
    COALESCE(hlq.lineup_avg_avg_10, 0) - COALESCE(alq.lineup_avg_avg_10, 0) AS lineup_avg_edge,
    COALESCE(hlq.lineup_slg_avg_10, 0) - COALESCE(alq.lineup_slg_avg_10, 0) AS lineup_slg_edge,

    -- ---- Residual targets (MUST be appended last — CREATE OR REPLACE VIEW forbids column reordering) ----
    -- Training targets for market-anchored residual models.
    -- Residual = actual - market_line; reconstruction = market_line + pred_residual.
    (g.home_score - g.away_score) - ml.run_line_home AS run_line_residual,
    (g.home_score + g.away_score) - ml.total_line    AS total_residual,

    -- ---- Group D: Bullpen fatigue 7-day window + SP short outing (appended last) ----
    hp.bullpen_ip_last_7                             AS home_bullpen_ip_last_7,
    hp.bp_era_7d                                     AS home_bp_era_7d,
    hp.sp_short_last                                 AS home_sp_short_last,
    ap.bullpen_ip_last_7                             AS away_bullpen_ip_last_7,
    ap.bp_era_7d                                     AS away_bp_era_7d,
    ap.sp_short_last                                 AS away_sp_short_last,
    ap.bullpen_ip_last_7 - hp.bullpen_ip_last_7      AS bullpen_fatigue_7d_edge,

    -- ---- Group E: Series context (appended last — column reordering forbidden) ----
    -- Game number within the current home series between these two teams.
    -- 1 = opener, 2 = middle game, 3+ = series finale. Captures bullpen depth effects.
    (
        SELECT COUNT(*)
        FROM raw.mlb_games g2
        WHERE g2.home_team_abbr = g.home_team_abbr
          AND g2.away_team_abbr = g.away_team_abbr
          AND g2.game_date_et  <= g.game_date_et
          AND g2.game_date_et  >= g.game_date_et - INTERVAL '4 days'
          AND g2.status IN ('final', 'scheduled', 'in_progress')
    )::INTEGER                                       AS series_game_number,

    -- ---- Group F: SP k_pct_10 (individual SP last 10 starts K%) ----
    -- Enables K% trend (5v10) calculation in features.py.
    hsp.k_pct_10                                     AS home_sp_k_pct_10,
    asp.k_pct_10                                     AS away_sp_k_pct_10,

    -- ---- Group F: SP venue familiarity (career ERA/FIP at this ballpark) ----
    -- Reliability-weighted in features.py by n_starts_at_venue.
    hsp_venue.n_starts_at_venue                      AS home_sp_venue_starts,
    hsp_venue.venue_era                              AS home_sp_venue_era,
    hsp_venue.venue_k9                               AS home_sp_venue_k9,
    hsp_venue.venue_fip                              AS home_sp_venue_fip,
    asp_venue.n_starts_at_venue                      AS away_sp_venue_starts,
    asp_venue.venue_era                              AS away_sp_venue_era,
    asp_venue.venue_k9                               AS away_sp_venue_k9,
    asp_venue.venue_fip                              AS away_sp_venue_fip,

    -- ---- Group F: SP handedness (1 = left-handed, 0 = right-handed) ----
    CASE WHEN hsp_hand.pitch_hand = 'L' THEN 1 ELSE 0 END AS home_sp_pitch_hand_L,
    CASE WHEN asp_hand.pitch_hand = 'L' THEN 1 ELSE 0 END AS away_sp_pitch_hand_L,

    -- ---- Group F: Team batting vs opposing SP handedness (rolling 40 games) ----
    -- Home team batting stats split by hand (vs today's away SP hand = away_sp_pitch_hand_L)
    htbh.team_avg_vs_lhp                             AS home_team_avg_vs_lhp,
    htbh.team_avg_vs_rhp                             AS home_team_avg_vs_rhp,
    htbh.team_obp_vs_lhp                             AS home_team_obp_vs_lhp,
    htbh.team_obp_vs_rhp                             AS home_team_obp_vs_rhp,
    htbh.team_slg_vs_lhp                             AS home_team_slg_vs_lhp,
    htbh.team_slg_vs_rhp                             AS home_team_slg_vs_rhp,
    htbh.games_vs_lhp                                AS home_games_vs_lhp,
    htbh.games_vs_rhp                                AS home_games_vs_rhp,
    -- Away team batting stats split by hand (vs today's home SP hand = home_sp_pitch_hand_L)
    atbh.team_avg_vs_lhp                             AS away_team_avg_vs_lhp,
    atbh.team_avg_vs_rhp                             AS away_team_avg_vs_rhp,
    atbh.team_obp_vs_lhp                             AS away_team_obp_vs_lhp,
    atbh.team_obp_vs_rhp                             AS away_team_obp_vs_rhp,
    atbh.team_slg_vs_lhp                             AS away_team_slg_vs_lhp,
    atbh.team_slg_vs_rhp                             AS away_team_slg_vs_rhp,
    atbh.games_vs_lhp                                AS away_games_vs_lhp,
    atbh.games_vs_rhp                                AS away_games_vs_rhp,

    -- ---- Group G: Individual reliever rest/workload ----
    hrr.bp_relievers_last_1d                         AS home_bp_relievers_last_1d,
    hrr.bp_relievers_last_2d                         AS home_bp_relievers_last_2d,
    hrr.bp_relievers_last_3d                         AS home_bp_relievers_last_3d,
    hrr.bp_ip_last_1d                                AS home_bp_ip_last_1d,
    hrr.bp_avg_era_last_3d                           AS home_bp_avg_era_last_3d,
    arr.bp_relievers_last_1d                         AS away_bp_relievers_last_1d,
    arr.bp_relievers_last_2d                         AS away_bp_relievers_last_2d,
    arr.bp_relievers_last_3d                         AS away_bp_relievers_last_3d,
    arr.bp_ip_last_1d                                AS away_bp_ip_last_1d,
    arr.bp_avg_era_last_3d                           AS away_bp_avg_era_last_3d,

    -- ---- SP workload: last-start innings pitched (Item #4) ----
    hsp.last_start_ip                               AS home_sp_last_ip,
    asp.last_start_ip                               AS away_sp_last_ip,

    -- ---- Group H: SP Statcast (quality of contact allowed) ----
    sc_home_sp.barrel_batted_rate   AS home_sp_sc_barrel_rate,
    sc_home_sp.hard_hit_percent     AS home_sp_sc_hard_hit_pct,
    sc_home_sp.xwoba                AS home_sp_sc_xwoba,
    sc_home_sp.avg_exit_velocity    AS home_sp_sc_exit_velo,
    pa_home_sp.sl_whiff_pct         AS home_sp_sl_whiff_pct,
    pa_home_sp.ch_whiff_pct         AS home_sp_ch_whiff_pct,
    pa_home_sp.fb_put_away          AS home_sp_fb_put_away,
    sc_away_sp.barrel_batted_rate   AS away_sp_sc_barrel_rate,
    sc_away_sp.hard_hit_percent     AS away_sp_sc_hard_hit_pct,
    sc_away_sp.xwoba                AS away_sp_sc_xwoba,
    sc_away_sp.avg_exit_velocity    AS away_sp_sc_exit_velo,
    pa_away_sp.sl_whiff_pct         AS away_sp_sl_whiff_pct,
    pa_away_sp.ch_whiff_pct         AS away_sp_ch_whiff_pct,
    pa_away_sp.fb_put_away          AS away_sp_fb_put_away,

    -- ---- Group I: SP Statcast discipline (season-level K/BB/whiff anchors) ----
    -- Stable season-level profile — complements rolling ERA/FIP with command signal.
    disc_home_sp.k_pct              AS home_sp_sc_k_pct,
    disc_home_sp.bb_pct             AS home_sp_sc_bb_pct,
    disc_home_sp.whiff_pct          AS home_sp_sc_whiff_pct,
    disc_home_sp.oz_swing_pct       AS home_sp_sc_oz_swing_pct,
    disc_away_sp.k_pct              AS away_sp_sc_k_pct,
    disc_away_sp.bb_pct             AS away_sp_sc_bb_pct,
    disc_away_sp.whiff_pct          AS away_sp_sc_whiff_pct,
    disc_away_sp.oz_swing_pct       AS away_sp_sc_oz_swing_pct,

    -- ---- Group J: F5 targets (training only) ----
    bg.home_f5_runs                                              AS home_f5_runs,
    bg.away_f5_runs                                              AS away_f5_runs,
    bg.home_f5_runs + bg.away_f5_runs                           AS total_f5,
    bg.home_f5_runs - bg.away_f5_runs                           AS f5_run_diff,

    -- ---- Group K: Career batter vs SP H2H ----
    h2h_batting.home_h2h_ba, h2h_batting.home_h2h_obp, h2h_batting.home_h2h_slg, h2h_batting.home_h2h_n,
    h2h_batting.away_h2h_ba, h2h_batting.away_h2h_obp, h2h_batting.away_h2h_slg, h2h_batting.away_h2h_n,
    h2h_batting.h2h_slg_edge,

    -- ---- Group L: Lineup Statcast quality ----
    hlq.lineup_xwoba_avg       AS home_lineup_xwoba_avg,
    hlq.lineup_xslg_avg        AS home_lineup_xslg_avg,
    hlq.lineup_barrel_avg      AS home_lineup_barrel_avg,
    hlq.lineup_hard_hit_avg    AS home_lineup_hard_hit_avg,
    alq.lineup_xwoba_avg       AS away_lineup_xwoba_avg,
    alq.lineup_xslg_avg        AS away_lineup_xslg_avg,
    alq.lineup_barrel_avg      AS away_lineup_barrel_avg,
    alq.lineup_hard_hit_avg    AS away_lineup_hard_hit_avg,

    -- ---- Group M: Catcher framing (NULL for upcoming games — median-imputed) ----
    hcf.catcher_framing_rv                           AS home_catcher_framing_rv,
    hcf.catcher_framing_rate                         AS home_catcher_framing_rate,
    acf.catcher_framing_rv                           AS away_catcher_framing_rv,
    acf.catcher_framing_rate                         AS away_catcher_framing_rate,

    -- ---- Group N: SP velocity trend (NULL when no velocity data — median-imputed) ----
    velo_home_sp.fb_velo_trend_5                     AS home_fb_velo_trend_5,
    velo_home_sp.fb_velo_avg_5                       AS home_fb_velo_avg_5,
    velo_away_sp.fb_velo_trend_5                     AS away_fb_velo_trend_5,
    velo_away_sp.fb_velo_avg_5                       AS away_fb_velo_avg_5

FROM raw.mlb_games g

LEFT JOIN raw.mlb_venues v
    ON v.venue_id = g.venue_id

LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_batting_rolling_mat hb
    ON hb.game_slug = g.game_slug
   AND hb.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_batting_rolling_mat ab
    ON ab.game_slug = g.game_slug
   AND ab.team_abbr = g.away_team_abbr

LEFT JOIN features.mlb_team_pitching_rolling_mat hp
    ON hp.game_slug = g.game_slug
   AND hp.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_pitching_rolling_mat ap
    ON ap.game_slug = g.game_slug
   AND ap.team_abbr = g.away_team_abbr

LEFT JOIN raw.mlb_starting_pitchers hsp_sched
    ON hsp_sched.game_slug = g.game_slug
   AND hsp_sched.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_pitcher_rolling_mat hsp
    ON hsp.game_slug  = g.game_slug
   AND hsp.player_id  = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN raw.mlb_starting_pitchers asp_sched
    ON asp_sched.game_slug = g.game_slug
   AND asp_sched.team_abbr = g.away_team_abbr

LEFT JOIN features.mlb_pitcher_rolling_mat asp
    ON asp.game_slug  = g.game_slug
   AND asp.player_id  = COALESCE(asp_sched.player_id, g.away_sp_id)

LEFT JOIN features.mlb_standings_rest_mat hsr
    ON hsr.game_slug = g.game_slug
   AND hsr.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_standings_rest_mat asr
    ON asr.game_slug = g.game_slug
   AND asr.team_abbr = g.away_team_abbr

LEFT JOIN market_lines ml
    ON ml.home_team_abbr = g.home_team_abbr
   AND ml.away_team_abbr = g.away_team_abbr
   AND ml.as_of_date     = g.game_date_et

LEFT JOIN market_lines_open mlo
    ON mlo.home_team_abbr = g.home_team_abbr
   AND mlo.away_team_abbr = g.away_team_abbr
   AND mlo.as_of_date     = g.game_date_et

LEFT JOIN h2h
    ON h2h.game_slug = g.game_slug

-- Group C joins
LEFT JOIN features.mlb_umpire_rolling_mat ump
    ON ump.game_slug = g.game_slug

LEFT JOIN raw.mlb_weather wx
    ON wx.game_slug = g.game_slug

LEFT JOIN features.mlb_lineup_quality hlq
    ON hlq.game_slug = g.game_slug
   AND hlq.is_home = TRUE

LEFT JOIN features.mlb_lineup_quality alq
    ON alq.game_slug = g.game_slug
   AND alq.is_home = FALSE

-- Group F: SP venue stats (training view joins on game_slug + player_id)
LEFT JOIN features.mlb_sp_venue_stats_mat hsp_venue
    ON hsp_venue.game_slug = g.game_slug
   AND hsp_venue.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN features.mlb_sp_venue_stats_mat asp_venue
    ON asp_venue.game_slug = g.game_slug
   AND asp_venue.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)

-- Group F: SP handedness
LEFT JOIN raw.mlb_player_handedness hsp_hand
    ON hsp_hand.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN raw.mlb_player_handedness asp_hand
    ON asp_hand.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)

-- Group F: Team batting vs SP handedness (training view: join on game_slug + team_abbr)
LEFT JOIN features.mlb_team_batting_vs_hand_mat htbh
    ON htbh.game_slug = g.game_slug
   AND htbh.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_batting_vs_hand_mat atbh
    ON atbh.game_slug = g.game_slug
   AND atbh.team_abbr = g.away_team_abbr

-- Group G: Individual reliever rest (training view: join on game_slug + team_abbr)
LEFT JOIN features.mlb_reliever_rolling_mat hrr
    ON hrr.game_slug = g.game_slug
   AND hrr.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_reliever_rolling_mat arr
    ON arr.game_slug = g.game_slug
   AND arr.team_abbr = g.away_team_abbr

-- Group H: SP Statcast joins
LEFT JOIN raw.mlb_statcast_pitching sc_home_sp
    ON sc_home_sp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND sc_home_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_home_sp
    ON pa_home_sp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND pa_home_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitching sc_away_sp
    ON sc_away_sp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND sc_away_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_away_sp
    ON pa_away_sp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND pa_away_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

-- Group I: SP discipline
LEFT JOIN raw.mlb_statcast_pitcher_discipline disc_home_sp
    ON disc_home_sp.player_id  = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND disc_home_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitcher_discipline disc_away_sp
    ON disc_away_sp.player_id  = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND disc_away_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

-- Group J: F5 targets (boxscore has first-5-innings scores)
LEFT JOIN raw.mlb_boxscore_games bg
    ON bg.game_slug = g.game_slug

-- Group K: Career batter vs SP H2H aggregates
LEFT JOIN features.mlb_game_h2h_batting_mat h2h_batting
    ON h2h_batting.game_slug = g.game_slug

-- Group M: Catcher framing
LEFT JOIN features.mlb_team_catcher_framing hcf
    ON hcf.game_slug = g.game_slug
   AND hcf.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_catcher_framing acf
    ON acf.game_slug = g.game_slug
   AND acf.team_abbr = g.away_team_abbr

-- Group N: SP velocity trend (join on player_id + game_date)
LEFT JOIN features.mlb_sp_velocity_rolling velo_home_sp
    ON velo_home_sp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND velo_home_sp.game_date = g.game_date_et::DATE

LEFT JOIN features.mlb_sp_velocity_rolling velo_away_sp
    ON velo_away_sp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND velo_away_sp.game_date = g.game_date_et::DATE

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
        home_team                   AS home_team_abbr,
        away_team                   AS away_team_abbr,
        as_of_date,
        spread_home_points          AS run_line_home,
        spread_home_price           AS run_line_home_price,
        spread_away_price           AS run_line_away_price,
        total_points                AS total_line,
        total_over_price            AS over_price,
        total_under_price           AS under_price
    FROM odds.mlb_game_lines
    WHERE bookmaker_key IN ('draftkings', 'fanduel')
    ORDER BY
        home_team,
        away_team,
        as_of_date,
        CASE bookmaker_key WHEN 'fanduel' THEN 0 ELSE 1 END
),
-- Latest rolling stats per team
latest_batting AS (
    SELECT DISTINCT ON (team_abbr)
        *
    FROM features.mlb_team_batting_rolling_mat
    ORDER BY team_abbr, game_date_et DESC, game_slug DESC
),
latest_pitching AS (
    SELECT DISTINCT ON (team_abbr)
        *
    FROM features.mlb_team_pitching_rolling_mat
    ORDER BY team_abbr, game_date_et DESC, game_slug DESC
),
-- Latest rolling stats per pitcher (most recent start)
latest_pitcher AS (
    SELECT DISTINCT ON (player_id)
        *
    FROM features.mlb_pitcher_rolling_mat
    ORDER BY player_id, game_date_et DESC, game_slug DESC
),
-- Latest SP venue stats per (player, venue) — for prediction, pick the most
-- recent row the pitcher has at the specific venue they'll pitch at today.
latest_sp_venue AS (
    SELECT DISTINCT ON (player_id, venue_id)
        player_id,
        venue_id,
        n_starts_at_venue,
        venue_era,
        venue_k9,
        venue_fip
    FROM features.mlb_sp_venue_stats_mat
    ORDER BY player_id, venue_id, game_date_et DESC, game_slug DESC
),
-- Latest team batting vs SP hand per team
latest_batting_vs_hand AS (
    SELECT DISTINCT ON (team_abbr)
        team_abbr,
        team_avg_vs_lhp,
        team_avg_vs_rhp,
        team_obp_vs_lhp,
        team_obp_vs_rhp,
        team_slg_vs_lhp,
        team_slg_vs_rhp,
        games_vs_lhp,
        games_vs_rhp
    FROM features.mlb_team_batting_vs_hand_mat
    ORDER BY team_abbr, game_date_et DESC, game_slug DESC
),
-- Group G: Latest individual reliever rest/workload per team
latest_reliever AS (
    SELECT DISTINCT ON (team_abbr)
        *
    FROM features.mlb_reliever_rolling_mat
    ORDER BY team_abbr, game_date_et DESC, game_slug DESC
),
-- Group N: Latest SP velocity (most recent start per pitcher)
latest_velocity AS (
    SELECT DISTINCT ON (player_id)
        player_id,
        game_date,
        ff_avg_speed,
        si_avg_speed,
        fb_velo_avg_5,
        fb_velo_trend_5,
        si_velo_avg_5,
        si_velo_trend_5
    FROM features.mlb_sp_velocity_rolling
    ORDER BY player_id, game_date DESC
),
-- Head-to-head season record for upcoming games
h2h AS (
    SELECT
        g1.game_slug,
        COUNT(*) FILTER (WHERE g2.status = 'final')       AS h2h_games_ytd,
        SUM(CASE
            WHEN g2.home_team_abbr = g1.home_team_abbr
             AND g2.home_score > g2.away_score THEN 1
            WHEN g2.away_team_abbr = g1.home_team_abbr
             AND g2.away_score > g2.home_score THEN 1
            ELSE 0
        END) FILTER (WHERE g2.status = 'final')           AS h2h_home_team_wins
    FROM raw.mlb_games g1
    LEFT JOIN raw.mlb_games g2
        ON (   (g2.home_team_abbr = g1.home_team_abbr AND g2.away_team_abbr = g1.away_team_abbr)
            OR (g2.home_team_abbr = g1.away_team_abbr AND g2.away_team_abbr = g1.home_team_abbr))
       AND g2.game_date_et < g1.game_date_et
       AND g2.season       = g1.season
    GROUP BY g1.game_slug
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
    hb.sb_avg_5                             AS home_sb_avg_5,
    hb.sb_avg_10                            AS home_sb_avg_10,
    hb.sb_pct_10                            AS home_sb_pct_10,

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
    ab.sb_avg_5                             AS away_sb_avg_5,
    ab.sb_avg_10                            AS away_sb_avg_10,
    ab.sb_pct_10                            AS away_sb_pct_10,

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
    hp.bp_k9_5                              AS home_bp_k9_5,
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
    hp.bp_k9_10                             AS home_bp_k9_10,
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
    ap.bp_k9_5                              AS away_bp_k9_5,
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
    ap.bp_k9_10                             AS away_bp_k9_10,
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
    -- Group B: SP rest (days since last start before today) + home/away splits
    (g.game_date_et - hsp.game_date_et)     AS home_sp_days_rest,
    CASE WHEN (g.game_date_et - hsp.game_date_et) <= 4 THEN 1 ELSE 0 END
                                            AS home_sp_is_short_rest,
    hsp.era_home_10                         AS home_sp_era_home_10,
    hsp.era_away_10                         AS home_sp_era_away_10,
    hsp.k9_home_10                          AS home_sp_k9_home_10,
    hsp.k9_away_10                          AS home_sp_k9_away_10,
    hsp.fip_home_10                         AS home_sp_fip_home_10,
    hsp.fip_away_10                         AS home_sp_fip_away_10,
    -- SP 20-start baseline (regression-to-mean anchor)
    hsp.era_20                              AS home_sp_era_20,
    hsp.fip_20                              AS home_sp_fip_20,
    -- Last-start pitch quality (complements last_start_ip for full last-outing picture)
    hsp.last_start_k                        AS home_sp_last_k,
    hsp.last_start_bb                       AS home_sp_last_bb,
    hsp.last_start_fip                      AS home_sp_last_fip,

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
    -- Group B: SP rest + home/away splits
    (g.game_date_et - asp.game_date_et)     AS away_sp_days_rest,
    CASE WHEN (g.game_date_et - asp.game_date_et) <= 4 THEN 1 ELSE 0 END
                                            AS away_sp_is_short_rest,
    asp.era_home_10                         AS away_sp_era_home_10,
    asp.era_away_10                         AS away_sp_era_away_10,
    asp.k9_home_10                          AS away_sp_k9_home_10,
    asp.k9_away_10                          AS away_sp_k9_away_10,
    asp.fip_home_10                         AS away_sp_fip_home_10,
    asp.fip_away_10                         AS away_sp_fip_away_10,
    -- SP 20-start baseline (regression-to-mean anchor)
    asp.era_20                              AS away_sp_era_20,
    asp.fip_20                              AS away_sp_fip_20,
    -- Last-start pitch quality (complements last_start_ip for full last-outing picture)
    asp.last_start_k                        AS away_sp_last_k,
    asp.last_start_bb                       AS away_sp_last_bb,
    asp.last_start_fip                      AS away_sp_last_fip,

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
    -- Group B: rolling form
    hsr.wins_last_5                         AS home_wins_last_5,
    hsr.win_pct_last_5                      AS home_win_pct_last_5,
    hsr.wins_last_10                        AS home_wins_last_10,
    hsr.win_pct_last_10                     AS home_win_pct_last_10,
    hsr.run_diff_avg_last_5                 AS home_run_diff_avg_last_5,

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
    -- Group B: rolling form
    asr.wins_last_5                         AS away_wins_last_5,
    asr.win_pct_last_5                      AS away_win_pct_last_5,
    asr.wins_last_10                        AS away_wins_last_10,
    asr.win_pct_last_10                     AS away_win_pct_last_10,
    asr.run_diff_avg_last_5                 AS away_run_diff_avg_last_5,

    -- ---- Head-to-head season record ----
    h2h.h2h_games_ytd,
    CASE WHEN h2h.h2h_games_ytd > 0
         THEN h2h.h2h_home_team_wins::float / h2h.h2h_games_ytd
         ELSE NULL END                      AS h2h_home_win_pct_ytd,

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
                   + (ab.runs_avg_10 + hp.runs_allowed_avg_10)) / 2.0 AS park_adj_implied_total,

    -- ---- Group C: Umpire rolling stats ----
    -- NULL for upcoming games (umpire only known from completed boxscores)
    COALESCE(ump.n_ump_games_prev_5, 0)             AS n_ump_games_prev_5,
    ump.ump_rpg_5,
    ump.ump_k9_5,
    ump.ump_bb9_5,
    ump.ump_rpg_10,
    ump.ump_k9_10,
    ump.ump_bb9_10,

    -- ---- Group C: Weather ----
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0)
    END                                             AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0)
    END                                             AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0)
    END                                             AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)        AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END AS is_dome,

    -- ---- Group C: Lineup quality (NULL for upcoming games — median-imputed) ----
    hlq.lineup_avg_avg_10       AS home_lineup_avg_avg_10,
    hlq.lineup_slg_avg_10       AS home_lineup_slg_avg_10,
    hlq.lineup_iso_avg_10       AS home_lineup_iso_avg_10,
    hlq.top4_slg_avg_10         AS home_top4_slg_avg_10,
    hlq.lineup_data_completeness AS home_lineup_completeness,
    alq.lineup_avg_avg_10       AS away_lineup_avg_avg_10,
    alq.lineup_slg_avg_10       AS away_lineup_slg_avg_10,
    alq.lineup_iso_avg_10       AS away_lineup_iso_avg_10,
    alq.top4_slg_avg_10         AS away_top4_slg_avg_10,
    alq.lineup_data_completeness AS away_lineup_completeness,
    COALESCE(hlq.lineup_avg_avg_10, 0) - COALESCE(alq.lineup_avg_avg_10, 0) AS lineup_avg_edge,
    COALESCE(hlq.lineup_slg_avg_10, 0) - COALESCE(alq.lineup_slg_avg_10, 0) AS lineup_slg_edge,

    -- ---- Group D: Bullpen fatigue 7-day window + SP short outing (appended last) ----
    hp.bullpen_ip_last_7                             AS home_bullpen_ip_last_7,
    hp.bp_era_7d                                     AS home_bp_era_7d,
    hp.sp_short_last                                 AS home_sp_short_last,
    ap.bullpen_ip_last_7                             AS away_bullpen_ip_last_7,
    ap.bp_era_7d                                     AS away_bp_era_7d,
    ap.sp_short_last                                 AS away_sp_short_last,
    ap.bullpen_ip_last_7 - hp.bullpen_ip_last_7      AS bullpen_fatigue_7d_edge,

    -- ---- Group E: Series context ----
    (
        SELECT COUNT(*)
        FROM raw.mlb_games g2
        WHERE g2.home_team_abbr = g.home_team_abbr
          AND g2.away_team_abbr = g.away_team_abbr
          AND g2.game_date_et  <= g.game_date_et
          AND g2.game_date_et  >= g.game_date_et - INTERVAL '4 days'
          AND g2.status IN ('final', 'scheduled', 'in_progress')
    )::INTEGER                                       AS series_game_number,

    -- ---- Group F: SP k_pct_10 ----
    hsp.k_pct_10                                     AS home_sp_k_pct_10,
    asp.k_pct_10                                     AS away_sp_k_pct_10,

    -- ---- Group F: SP venue familiarity ----
    hsp_venue.n_starts_at_venue                      AS home_sp_venue_starts,
    hsp_venue.venue_era                              AS home_sp_venue_era,
    hsp_venue.venue_k9                               AS home_sp_venue_k9,
    hsp_venue.venue_fip                              AS home_sp_venue_fip,
    asp_venue.n_starts_at_venue                      AS away_sp_venue_starts,
    asp_venue.venue_era                              AS away_sp_venue_era,
    asp_venue.venue_k9                               AS away_sp_venue_k9,
    asp_venue.venue_fip                              AS away_sp_venue_fip,

    -- ---- Group F: SP handedness ----
    CASE WHEN hsp_hand.pitch_hand = 'L' THEN 1 ELSE 0 END AS home_sp_pitch_hand_L,
    CASE WHEN asp_hand.pitch_hand = 'L' THEN 1 ELSE 0 END AS away_sp_pitch_hand_L,

    -- ---- Group F: Team batting vs SP handedness ----
    htbh.team_avg_vs_lhp                             AS home_team_avg_vs_lhp,
    htbh.team_avg_vs_rhp                             AS home_team_avg_vs_rhp,
    htbh.team_obp_vs_lhp                             AS home_team_obp_vs_lhp,
    htbh.team_obp_vs_rhp                             AS home_team_obp_vs_rhp,
    htbh.team_slg_vs_lhp                             AS home_team_slg_vs_lhp,
    htbh.team_slg_vs_rhp                             AS home_team_slg_vs_rhp,
    htbh.games_vs_lhp                                AS home_games_vs_lhp,
    htbh.games_vs_rhp                                AS home_games_vs_rhp,
    atbh.team_avg_vs_lhp                             AS away_team_avg_vs_lhp,
    atbh.team_avg_vs_rhp                             AS away_team_avg_vs_rhp,
    atbh.team_obp_vs_lhp                             AS away_team_obp_vs_lhp,
    atbh.team_obp_vs_rhp                             AS away_team_obp_vs_rhp,
    atbh.team_slg_vs_lhp                             AS away_team_slg_vs_lhp,
    atbh.team_slg_vs_rhp                             AS away_team_slg_vs_rhp,
    atbh.games_vs_lhp                                AS away_games_vs_lhp,
    atbh.games_vs_rhp                                AS away_games_vs_rhp,

    -- ---- Group G: Individual reliever rest/workload ----
    hrr.bp_relievers_last_1d                         AS home_bp_relievers_last_1d,
    hrr.bp_relievers_last_2d                         AS home_bp_relievers_last_2d,
    hrr.bp_relievers_last_3d                         AS home_bp_relievers_last_3d,
    hrr.bp_ip_last_1d                                AS home_bp_ip_last_1d,
    hrr.bp_avg_era_last_3d                           AS home_bp_avg_era_last_3d,
    arr.bp_relievers_last_1d                         AS away_bp_relievers_last_1d,
    arr.bp_relievers_last_2d                         AS away_bp_relievers_last_2d,
    arr.bp_relievers_last_3d                         AS away_bp_relievers_last_3d,
    arr.bp_ip_last_1d                                AS away_bp_ip_last_1d,
    arr.bp_avg_era_last_3d                           AS away_bp_avg_era_last_3d,

    -- ---- SP workload: last-start innings pitched (Item #4) ----
    hsp.last_start_ip                               AS home_sp_last_ip,
    asp.last_start_ip                               AS away_sp_last_ip,

    -- ---- Group H: SP Statcast (quality of contact allowed) ----
    sc_home_sp.barrel_batted_rate   AS home_sp_sc_barrel_rate,
    sc_home_sp.hard_hit_percent     AS home_sp_sc_hard_hit_pct,
    sc_home_sp.xwoba                AS home_sp_sc_xwoba,
    sc_home_sp.avg_exit_velocity    AS home_sp_sc_exit_velo,
    pa_home_sp.sl_whiff_pct         AS home_sp_sl_whiff_pct,
    pa_home_sp.ch_whiff_pct         AS home_sp_ch_whiff_pct,
    pa_home_sp.fb_put_away          AS home_sp_fb_put_away,
    sc_away_sp.barrel_batted_rate   AS away_sp_sc_barrel_rate,
    sc_away_sp.hard_hit_percent     AS away_sp_sc_hard_hit_pct,
    sc_away_sp.xwoba                AS away_sp_sc_xwoba,
    sc_away_sp.avg_exit_velocity    AS away_sp_sc_exit_velo,
    pa_away_sp.sl_whiff_pct         AS away_sp_sl_whiff_pct,
    pa_away_sp.ch_whiff_pct         AS away_sp_ch_whiff_pct,
    pa_away_sp.fb_put_away          AS away_sp_fb_put_away,

    -- ---- Group I: SP Statcast discipline (season-level K/BB/whiff anchors) ----
    disc_home_sp.k_pct              AS home_sp_sc_k_pct,
    disc_home_sp.bb_pct             AS home_sp_sc_bb_pct,
    disc_home_sp.whiff_pct          AS home_sp_sc_whiff_pct,
    disc_home_sp.oz_swing_pct       AS home_sp_sc_oz_swing_pct,
    disc_away_sp.k_pct              AS away_sp_sc_k_pct,
    disc_away_sp.bb_pct             AS away_sp_sc_bb_pct,
    disc_away_sp.whiff_pct          AS away_sp_sc_whiff_pct,
    disc_away_sp.oz_swing_pct       AS away_sp_sc_oz_swing_pct,

    -- ---- Group L: Lineup Statcast quality (NULL for upcoming games — median-imputed) ----
    hlq.lineup_xwoba_avg       AS home_lineup_xwoba_avg,
    hlq.lineup_xslg_avg        AS home_lineup_xslg_avg,
    hlq.lineup_barrel_avg      AS home_lineup_barrel_avg,
    hlq.lineup_hard_hit_avg    AS home_lineup_hard_hit_avg,
    alq.lineup_xwoba_avg       AS away_lineup_xwoba_avg,
    alq.lineup_xslg_avg        AS away_lineup_xslg_avg,
    alq.lineup_barrel_avg      AS away_lineup_barrel_avg,
    alq.lineup_hard_hit_avg    AS away_lineup_hard_hit_avg,

    -- ---- Group M: Catcher framing (NULL for upcoming games — median-imputed) ----
    hcf.catcher_framing_rv                           AS home_catcher_framing_rv,
    hcf.catcher_framing_rate                         AS home_catcher_framing_rate,
    acf.catcher_framing_rv                           AS away_catcher_framing_rv,
    acf.catcher_framing_rate                         AS away_catcher_framing_rate,

    -- ---- Group N: SP velocity trend (NULL when no velocity data — median-imputed) ----
    velo_home_sp.fb_velo_trend_5                     AS home_fb_velo_trend_5,
    velo_home_sp.fb_velo_avg_5                       AS home_fb_velo_avg_5,
    velo_away_sp.fb_velo_trend_5                     AS away_fb_velo_trend_5,
    velo_away_sp.fb_velo_avg_5                       AS away_fb_velo_avg_5

FROM raw.mlb_games g

LEFT JOIN raw.mlb_venues v
    ON v.venue_id = g.venue_id

LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr

LEFT JOIN latest_batting hb ON hb.team_abbr = g.home_team_abbr
LEFT JOIN latest_batting ab ON ab.team_abbr = g.away_team_abbr
LEFT JOIN latest_pitching hp ON hp.team_abbr = g.home_team_abbr
LEFT JOIN latest_pitching ap ON ap.team_abbr = g.away_team_abbr

LEFT JOIN raw.mlb_starting_pitchers hsp_sched
    ON hsp_sched.game_slug = g.game_slug
   AND hsp_sched.team_abbr = g.home_team_abbr

LEFT JOIN latest_pitcher hsp
    ON hsp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN raw.mlb_starting_pitchers asp_sched
    ON asp_sched.game_slug = g.game_slug
   AND asp_sched.team_abbr = g.away_team_abbr

LEFT JOIN latest_pitcher asp
    ON asp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)

LEFT JOIN features.mlb_standings_rest_mat hsr
    ON hsr.game_slug = g.game_slug
   AND hsr.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_standings_rest_mat asr
    ON asr.game_slug = g.game_slug
   AND asr.team_abbr = g.away_team_abbr

LEFT JOIN market_lines ml
    ON ml.home_team_abbr = g.home_team_abbr
   AND ml.away_team_abbr = g.away_team_abbr
   AND ml.as_of_date     = g.game_date_et

LEFT JOIN h2h
    ON h2h.game_slug = g.game_slug

-- Group C joins
LEFT JOIN features.mlb_umpire_rolling_mat ump
    ON ump.game_slug = g.game_slug

LEFT JOIN raw.mlb_weather wx
    ON wx.game_slug = g.game_slug

LEFT JOIN features.mlb_lineup_quality hlq
    ON hlq.game_slug = g.game_slug
   AND hlq.is_home = TRUE

LEFT JOIN features.mlb_lineup_quality alq
    ON alq.game_slug = g.game_slug
   AND alq.is_home = FALSE

-- Group F: SP venue stats (prediction view uses latest per player+venue)
LEFT JOIN latest_sp_venue hsp_venue
    ON hsp_venue.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND hsp_venue.venue_id  = g.venue_id

LEFT JOIN latest_sp_venue asp_venue
    ON asp_venue.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND asp_venue.venue_id  = g.venue_id

-- Group F: SP handedness
LEFT JOIN raw.mlb_player_handedness hsp_hand
    ON hsp_hand.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN raw.mlb_player_handedness asp_hand
    ON asp_hand.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)

-- Group F: Team batting vs SP handedness (prediction: latest per team)
LEFT JOIN latest_batting_vs_hand htbh ON htbh.team_abbr = g.home_team_abbr
LEFT JOIN latest_batting_vs_hand atbh ON atbh.team_abbr = g.away_team_abbr

-- Group G: Individual reliever rest (prediction: latest per team)
LEFT JOIN latest_reliever hrr ON hrr.team_abbr = g.home_team_abbr
LEFT JOIN latest_reliever arr ON arr.team_abbr = g.away_team_abbr

-- Group H: SP Statcast joins
LEFT JOIN raw.mlb_statcast_pitching sc_home_sp
    ON sc_home_sp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND sc_home_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_home_sp
    ON pa_home_sp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND pa_home_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitching sc_away_sp
    ON sc_away_sp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND sc_away_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_away_sp
    ON pa_away_sp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND pa_away_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

-- Group I: SP discipline
LEFT JOIN raw.mlb_statcast_pitcher_discipline disc_home_sp
    ON disc_home_sp.player_id  = COALESCE(hsp_sched.player_id, g.home_sp_id)
   AND disc_home_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

LEFT JOIN raw.mlb_statcast_pitcher_discipline disc_away_sp
    ON disc_away_sp.player_id  = COALESCE(asp_sched.player_id, g.away_sp_id)
   AND disc_away_sp.season_year = EXTRACT(YEAR FROM g.game_date_et)::INT

-- Group M: Catcher framing
LEFT JOIN features.mlb_team_catcher_framing hcf
    ON hcf.game_slug = g.game_slug
   AND hcf.team_abbr = g.home_team_abbr

LEFT JOIN features.mlb_team_catcher_framing acf
    ON acf.game_slug = g.game_slug
   AND acf.team_abbr = g.away_team_abbr

-- Group N: SP velocity trend (prediction: latest start per pitcher)
LEFT JOIN latest_velocity velo_home_sp
    ON velo_home_sp.player_id = COALESCE(hsp_sched.player_id, g.home_sp_id)

LEFT JOIN latest_velocity velo_away_sp
    ON velo_away_sp.player_id = COALESCE(asp_sched.player_id, g.away_sp_id)

WHERE g.status IN ('scheduled', 'in_progress')
  AND g.game_date_et >= CURRENT_DATE
  AND g.game_date_et <= CURRENT_DATE + INTERVAL '1 day'
;
