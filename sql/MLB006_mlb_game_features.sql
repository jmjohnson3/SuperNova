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
    -- Group B: SP rest + home/away splits
    hsp.days_since_last_start               AS home_sp_days_rest,
    hsp.is_short_rest                       AS home_sp_is_short_rest,
    hsp.era_home_10                         AS home_sp_era_home_10,
    hsp.era_away_10                         AS home_sp_era_away_10,
    hsp.k9_home_10                          AS home_sp_k9_home_10,
    hsp.k9_away_10                          AS home_sp_k9_away_10,
    hsp.fip_home_10                         AS home_sp_fip_home_10,
    hsp.fip_away_10                         AS home_sp_fip_away_10,

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
    atbh.games_vs_rhp                                AS away_games_vs_rhp

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
    atbh.games_vs_rhp                                AS away_games_vs_rhp

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

WHERE g.status IN ('scheduled', 'in_progress')
  AND g.game_date_et >= CURRENT_DATE
  AND g.game_date_et <= CURRENT_DATE + INTERVAL '1 day'
;
