# src/mlb_pipeline/modeling/predict_player_props.py
"""
MLB player prop predictions for today's slate.

Loads pitcher/batter prop models and generates predictions for:
  - pitcher_strikeouts   (starting pitchers from raw.mlb_starting_pitchers)
  - batter_hits          (batters with ab_avg_10 >= 1.5 playing today)
  - batter_total_bases
  - batter_home_runs
  - batter_walks

Edge formula:  edge = pred - book_line
Bet signal:    |edge| >= threshold (K: 1.0, H: 0.5, TB: 0.5, HR: 0.25, BB: 0.3)

Discord output (DISCORD_FORMAT=1): compact grouped by game.
DB: bets.mlb_prop_predictions — one row per (game_date_et, game_slug, player_id, stat).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure stdout is UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features, build_fd_parlay_url

log = logging.getLogger("mlb_pipeline.modeling.predict_player_props")
_ET = ZoneInfo("America/New_York")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    et_date: date | None = None
    # Minimum samples to include a batter
    # Set to 1 to allow early-season predictions (e.g., Opening Day)
    min_ab_avg_10: float = 1.0
    min_n_games: int = 1
    # Bet thresholds (|edge| >=)
    threshold_strikeouts: float = 1.0
    # Raised from 0.4 → 0.5: 0.4-0.5 edge bucket was marginal; ≥0.5 shows 65%+ win rate
    threshold_hits: float = 0.5
    threshold_total_bases: float = 0.5
    threshold_home_runs: float = 0.25
    threshold_walks: float = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot SQL
# ─────────────────────────────────────────────────────────────────────────────

SQL_PITCHER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
),
today_starters AS (
    SELECT
        sp.game_slug,
        sp.player_id,
        sp.player_name,
        sp.team_abbr,
        CASE WHEN sp.team_abbr = gt.home_team_abbr THEN TRUE ELSE FALSE END AS is_home,
        CASE WHEN sp.team_abbr = gt.home_team_abbr
             THEN gt.away_team_abbr ELSE gt.home_team_abbr END AS opponent_abbr,
        gt.start_ts_utc,
        gt.game_date_et
    FROM raw.mlb_starting_pitchers sp
    JOIN games_today gt ON gt.game_slug = sp.game_slug
    WHERE sp.player_id IS NOT NULL
)
SELECT
    ts.game_slug,
    ts.game_date_et,
    ts.start_ts_utc,
    ts.player_id,
    ts.player_name,
    ts.team_abbr,
    ts.is_home,
    ts.opponent_abbr,
    -- Most recent rolling pitcher stats prior to today
    pr.season,
    pr.ip_avg_5,
    pr.k_pct_5,
    pr.k9_5,
    pr.era_5,
    pr.whip_5,
    pr.fip_5,
    pr.bb_pct_5,
    pr.hr9_5,
    pr.k_pct_10,
    pr.k9_10,
    pr.era_10,
    pr.whip_10,
    pr.fip_10,
    pr.starts_in_window_5,
    pr.starts_in_window_10,
    pr.last_start_k,
    pr.last_start_ip,
    pr.last_start_bb,
    -- Group B: SP rest + home/away splits (MLB003)
    pr.days_since_last_start AS sp_days_since_last_start,
    pr.is_short_rest,
    pr.era_home_10,
    pr.era_away_10,
    pr.k9_home_10,
    pr.k9_away_10,
    pr.fip_home_10,
    pr.fip_away_10,
    -- Opponent batting context
    ob.k_pct_avg_10     AS opp_k_pct_avg_10,
    ob.bb_pct_avg_10    AS opp_bb_pct_avg_10,
    ob.avg_avg_10       AS opp_avg_avg_10,
    ob.hr_avg_10        AS opp_hr_avg_10,
    ob.iso_avg_10       AS opp_iso_avg_10,
    ob.slg_avg_10       AS opp_slg_avg_10,
    -- Park factors
    bf.run_factor       AS park_run_factor,
    bf.hr_factor        AS park_hr_factor,
    -- Weather (dome-zeroed)
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0) END        AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)                     AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END             AS is_dome,
    -- Day game flag (dome = always 0; day = ET hour < 17)
    CASE WHEN v.roof_type = 'dome' THEN 0
         WHEN EXTRACT(HOUR FROM ts.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1
         ELSE 0 END                                               AS is_day_game,
    -- Market total (game-level run environment)
    mkt_odds.market_total                                        AS market_total,
    -- Pitcher handedness
    ph.pitch_hand                                                AS pitcher_hand,
    -- Opponent lineup quality (NULL for upcoming games, median-imputed by model)
    lq_opp.lineup_avg_avg_10                                     AS opp_lineup_avg_avg_10,
    lq_opp.lineup_iso_avg_10                                     AS opp_lineup_iso_avg_10,
    lq_opp.top4_slg_avg_10                                       AS opp_top4_slg_avg_10,
    -- Statcast: pitcher's own batted-ball-against profile
    sc_p.barrel_batted_rate  AS sc_barrel_rate,
    sc_p.hard_hit_percent    AS sc_hard_hit_pct,
    sc_p.avg_exit_velocity   AS sc_avg_exit_velo,
    sc_p.groundballs_percent AS sc_gb_pct,
    sc_p.flyballs_percent    AS sc_fb_pct,
    sc_p.xba                 AS sc_xba,
    sc_p.xslg                AS sc_xslg,
    sc_p.xwoba               AS sc_xwoba,
    -- Extended Statcast: pitcher's own arsenal whiff/K profile (crawler_statcast_extended)
    pa_self.fb_whiff_pct          AS sc_sp_fb_whiff_pct,
    pa_self.fb_k_pct              AS sc_sp_fb_k_pct,
    pa_self.fb_put_away           AS sc_sp_fb_put_away,
    pa_self.sl_percent            AS sc_sp_sl_pct,
    pa_self.sl_whiff_pct          AS sc_sp_sl_whiff_pct,
    pa_self.sl_k_pct              AS sc_sp_sl_k_pct,
    pa_self.sl_run_value_per_100  AS sc_sp_sl_run_value_per_100,
    pa_self.ch_percent            AS sc_sp_ch_pct,
    pa_self.ch_whiff_pct          AS sc_sp_ch_whiff_pct,
    pa_self.ch_k_pct              AS sc_sp_ch_k_pct,
    -- Arsenal: FB/SI usage + SI whiff/K + diversity (previously unused)
    pa_self.fb_percent            AS sc_sp_fb_pct,
    pa_self.fb_hard_hit_pct       AS sc_sp_fb_hard_hit_pct,
    pa_self.fb_xwoba              AS sc_sp_fb_xwoba,
    pa_self.fb_run_value_per_100  AS sc_sp_fb_run_value_per_100,
    pa_self.si_percent            AS sc_sp_si_pct,
    pa_self.si_whiff_pct          AS sc_sp_si_whiff_pct,
    pa_self.si_k_pct              AS sc_sp_si_k_pct,
    pa_self.si_hard_hit_pct       AS sc_sp_si_hard_hit_pct,
    pa_self.fastball_family_pct   AS sc_sp_fastball_family_pct,
    pa_self.pitch_diversity       AS sc_sp_pitch_diversity,
    -- Pitcher plate discipline (induced chase rate, whiff, zone pct)
    pd_p.oz_swing_pct             AS sc_sp_oz_swing_pct,
    pd_p.iz_contact_pct           AS sc_sp_iz_contact_pct,
    pd_p.oz_contact_pct           AS sc_sp_oz_contact_pct,
    pd_p.whiff_pct                AS sc_sp_disc_whiff_pct,
    -- Catcher framing (run value per 100 borderline pitches; + = better framer)
    cf.framing_rv_per_100         AS catcher_framing_rv,
    cf.framing_rate               AS catcher_framing_rate
FROM today_starters ts
-- Most recent rolling stats
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_pitcher_rolling_mat pr
    WHERE pr.player_id = ts.player_id
      AND pr.game_date_et < %(game_date)s
    ORDER BY pr.game_date_et DESC, pr.game_slug DESC
    LIMIT 1
) pr ON TRUE
-- Opponent team batting (most recent for opponent)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_team_batting_rolling_mat ob
    WHERE ob.team_abbr = ts.opponent_abbr
      AND ob.game_date_et < %(game_date)s
    ORDER BY ob.game_date_et DESC, ob.game_slug DESC
    LIMIT 1
) ob ON TRUE
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = (
        SELECT home_team_abbr FROM raw.mlb_games
        WHERE game_slug = ts.game_slug LIMIT 1
    )
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = ts.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = ts.game_slug LIMIT 1
)
LEFT JOIN raw.mlb_player_handedness ph ON ph.player_id = ts.player_id
-- Opponent batting lineup quality (NULL for upcoming games)
LEFT JOIN features.mlb_lineup_quality lq_opp
    ON lq_opp.game_slug  = ts.game_slug
    AND lq_opp.team_abbr = ts.opponent_abbr
-- Statcast pitcher profile (season-level)
LEFT JOIN raw.mlb_statcast_pitching sc_p
    ON sc_p.player_id = ts.player_id
    AND sc_p.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Extended Statcast: pitcher's own arsenal whiff/K profile
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa_self
    ON pa_self.player_id = ts.player_id
    AND pa_self.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Market total (game-level run environment signal)
LEFT JOIN LATERAL (
    SELECT o.total_points AS market_total
    FROM odds.mlb_game_lines o
    JOIN raw.mlb_games mg ON mg.home_team_abbr = o.home_team
    WHERE mg.game_slug = ts.game_slug
      AND o.as_of_date = ts.game_date_et
      AND o.total_points IS NOT NULL
    ORDER BY o.fetched_at_utc DESC
    LIMIT 1
) mkt_odds ON TRUE
-- Pitcher plate discipline (own season-level discipline profile)
LEFT JOIN raw.mlb_statcast_pitcher_discipline pd_p
    ON pd_p.player_id = ts.player_id
    AND pd_p.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
-- Catcher framing: find the most recent catcher from this pitcher's team, then join framing stats
LEFT JOIN LATERAL (
    SELECT bps.player_id AS catcher_id
    FROM raw.mlb_boxscore_player_stats bps
    JOIN raw.mlb_games mg ON mg.game_slug = bps.game_slug
    WHERE bps.primary_position = 'C'
      AND bps.team_abbr = ts.team_abbr
      AND mg.game_date_et < %(game_date)s
    ORDER BY mg.game_date_et DESC
    LIMIT 1
) cat_recent ON TRUE
LEFT JOIN raw.mlb_statcast_catcher_framing cf
    ON cf.player_id = cat_recent.catcher_id
    AND cf.season_year = EXTRACT(YEAR FROM ts.game_date_et)::INT
ORDER BY ts.start_ts_utc, ts.game_slug, ts.player_id
"""

SQL_BATTER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
),
teams_today AS (
    SELECT home_team_abbr AS team_abbr, away_team_abbr AS opponent_abbr,
           TRUE AS is_home, game_slug, start_ts_utc, game_date_et
    FROM games_today
    UNION ALL
    SELECT away_team_abbr, home_team_abbr, FALSE, game_slug, start_ts_utc, game_date_et
    FROM games_today
),
-- Players who appeared for today's teams recently (last 14 days)
recent_players AS (
    SELECT DISTINCT gl.player_id, gl.team_abbr
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    WHERE gl.team_abbr IN (SELECT team_abbr FROM teams_today)
      AND g.game_date_et >= %(game_date)s::date - INTERVAL '14 days'
      AND gl.at_bats > 0
)
SELECT
    tt.game_slug,
    tt.start_ts_utc,
    tt.game_date_et,
    rp.player_id,
    tt.team_abbr,
    tt.opponent_abbr,
    tt.is_home,
    -- Latest rolling batter stats
    br.season,
    br.hits_avg_5,  br.hits_avg_10,  br.hits_avg_20,  br.hits_sd_10,
    br.hr_avg_5,    br.hr_avg_10,    br.hr_avg_20,
    br.tb_avg_5,    br.tb_avg_10,    br.tb_avg_20,    br.tb_sd_10,
    br.ab_avg_5,    br.ab_avg_10,
    br.avg_avg_10,  br.k_rate_avg_10, br.bb_rate_avg_10, br.iso_avg_10,
    br.hr_rate_avg_5, br.hr_rate_avg_10,
    br.n_games_prev_10,
    br.rest_days,
    -- Absolute walk/K count rolling
    br.bb_avg_5,    br.bb_avg_10,    br.bb_avg_20,    br.bb_sd_10,
    br.k_avg_5,     br.k_avg_10,
    -- Home/away conditional rolling
    br.hits_home_avg_20, br.hits_away_avg_20,
    br.tb_home_avg_20,   br.tb_away_avg_20,
    br.hr_home_avg_20,   br.hr_away_avg_20,
    br.bb_home_avg_20,   br.bb_away_avg_20,
    -- Opponent SP stats
    sp_r.era_5     AS opp_sp_era_5,
    sp_r.fip_5     AS opp_sp_fip_5,
    sp_r.k_pct_5   AS opp_sp_k_pct_5,
    sp_r.k9_5      AS opp_sp_k9_5,
    sp_r.bb_pct_5  AS opp_sp_bb_pct_5,
    sp_r.whip_5    AS opp_sp_whip_5,
    sp_r.ip_avg_5  AS opp_sp_ip_avg_5,
    -- Group B: opponent SP rest + home/away splits
    sp_r.days_since_last_start AS opp_sp_days_since_last_start,
    sp_r.is_short_rest         AS opp_sp_is_short_rest,
    sp_r.era_home_10           AS opp_sp_era_home_10,
    sp_r.era_away_10           AS opp_sp_era_away_10,
    -- Park factors
    bf.run_factor  AS park_run_factor,
    bf.hr_factor   AS park_hr_factor,
    -- Umpire
    ur.ump_bb9_10  AS ump_bb9_avg_10,
    ur.ump_k9_10   AS ump_k9_avg_10,
    -- Weather (dome-zeroed)
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0) END        AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)                     AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END             AS is_dome,
    -- Day game flag (dome = always 0; day = ET hour < 17)
    CASE WHEN v.roof_type = 'dome' THEN 0
         WHEN EXTRACT(HOUR FROM tt.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1
         ELSE 0 END                                               AS is_day_game,
    -- Market total (game-level run environment)
    mkt_odds.market_total                                        AS market_total,
    -- Lineup slot
    br.batting_order_avg_5,
    br.batting_order_avg_10,
    -- Batter + opponent SP handedness (OHE'd by _prep_features)
    bh.bat_side                AS batter_hand,
    opp_ph.pitch_hand          AS opp_sp_hand,
    -- Matched-hand stats (the split that matches today's opponent's throwing hand)
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.hits_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.hits_avg_40_vs_rhp END AS hits_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.tb_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.tb_avg_40_vs_rhp END AS tb_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.k_rate_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.k_rate_avg_40_vs_rhp END AS k_rate_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.iso_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.iso_avg_40_vs_rhp END AS iso_avg_40_vs_hand,
    -- Split differential (positive = better vs LHP)
    COALESCE(bvh.hits_avg_40_vs_lhp, 0) - COALESCE(bvh.hits_avg_40_vs_rhp, 0) AS hits_hand_split_40,
    COALESCE(bvh.tb_avg_40_vs_lhp,   0) - COALESCE(bvh.tb_avg_40_vs_rhp,   0) AS tb_hand_split_40,
    -- Sample sizes
    bvh.n_games_vs_lhp_40,
    bvh.n_games_vs_rhp_40,
    -- Cross-season rolling features (MLB013)
    bcs.n_games_prev_10_cs,
    bcs.hits_avg_10_cs,  bcs.hits_avg_20_cs,
    bcs.tb_avg_10_cs,    bcs.tb_avg_20_cs,
    bcs.hr_avg_10_cs,    bcs.hr_rate_avg_10_cs,
    bcs.ab_avg_10_cs,
    bcs.k_rate_avg_10_cs, bcs.bb_rate_avg_10_cs, bcs.iso_avg_10_cs,
    -- Prior full-season stats (MLB014)
    pss.prev_games,
    pss.prev_hits_avg,  pss.prev_tb_avg,  pss.prev_hr_avg,
    pss.prev_ab_avg,    pss.prev_k_rate,  pss.prev_bb_rate,
    pss.prev_iso,       pss.prev_hr_rate,
    -- Career H2H stats vs today's SP (MLB015, most recent entry before game date)
    h2h.h2h_games,
    h2h.h2h_ba,
    h2h.h2h_obp,
    h2h.h2h_slg,
    h2h.h2h_k_rate,
    h2h.h2h_iso,
    -- Own-team lineup quality (NULL for upcoming games, median-imputed by model)
    lq_own.lineup_slg_avg_10                                     AS own_lineup_slg_avg_10,
    lq_own.lineup_iso_avg_10                                     AS own_lineup_iso_avg_10,
    lq_own.top4_slg_avg_10                                       AS own_top4_slg_avg_10,
    -- Statcast: batter's own batted-ball profile (season-level)
    sc_b.barrel_batted_rate  AS sc_barrel_rate,
    sc_b.hard_hit_percent    AS sc_hard_hit_pct,
    sc_b.avg_exit_velocity   AS sc_avg_exit_velo,
    sc_b.avg_launch_angle    AS sc_avg_launch_angle,
    sc_b.sweet_spot_percent  AS sc_sweet_spot_pct,
    sc_b.flyballs_percent    AS sc_fb_pct,
    sc_b.groundballs_percent AS sc_gb_pct,
    sc_b.linedrives_percent  AS sc_ld_pct,
    sc_b.xba                 AS sc_xba,
    sc_b.xslg                AS sc_xslg,
    sc_b.xwoba               AS sc_xwoba,
    sc_b.xiso                AS sc_xiso,
    -- Extended Statcast: spray angle + barrels/PA
    sc_b.pull_percent        AS sc_pull_pct,
    sc_b.opposite_percent    AS sc_opposite_pct,
    sc_b.popup_percent       AS sc_popup_pct,
    sc_b.brl_pa              AS sc_brl_pa,
    -- Extended Statcast: sprint speed
    ss.sprint_speed          AS sprint_speed,
    -- Statcast: opposing SP's batted-ball-against profile
    sc_opp_p.barrel_batted_rate  AS opp_sp_sc_barrel_rate,
    sc_opp_p.hard_hit_percent    AS opp_sp_sc_hard_hit_pct,
    sc_opp_p.avg_exit_velocity   AS opp_sp_sc_avg_exit_velo,
    sc_opp_p.groundballs_percent AS opp_sp_sc_gb_pct,
    sc_opp_p.xba                 AS opp_sp_sc_xba,
    sc_opp_p.xslg                AS opp_sp_sc_xslg,
    sc_opp_p.xwoba               AS opp_sp_sc_xwoba,
    -- Extended Statcast: opposing SP arsenal (crawler_statcast_extended)
    pa.fb_percent            AS opp_sp_fb_pct,
    pa.fb_hard_hit_pct       AS opp_sp_fb_hard_hit_pct,
    pa.fb_xwoba              AS opp_sp_fb_xwoba,
    pa.fb_run_value_per_100  AS opp_sp_fb_run_value_per_100,
    pa.fb_whiff_pct          AS opp_sp_fb_whiff_pct,
    pa.fb_k_pct              AS opp_sp_fb_k_pct,
    pa.sl_percent            AS opp_sp_sl_pct,
    pa.sl_whiff_pct          AS opp_sp_sl_whiff_pct,
    pa.sl_k_pct              AS opp_sp_sl_k_pct,
    pa.sl_xwoba              AS opp_sp_sl_xwoba,
    pa.ch_percent            AS opp_sp_ch_pct,
    pa.ch_whiff_pct          AS opp_sp_ch_whiff_pct,
    pa.ch_k_pct              AS opp_sp_ch_k_pct,
    pa.fastball_family_pct   AS opp_sp_fastball_family_pct,
    pa.pitch_diversity       AS opp_sp_pitch_diversity,
    -- Arsenal: SI + SL/CH quality metrics (previously unused)
    pa.si_percent            AS opp_sp_si_pct,
    pa.si_whiff_pct          AS opp_sp_si_whiff_pct,
    pa.si_k_pct              AS opp_sp_si_k_pct,
    pa.si_hard_hit_pct       AS opp_sp_si_hard_hit_pct,
    pa.sl_hard_hit_pct       AS opp_sp_sl_hard_hit_pct,
    pa.sl_run_value_per_100  AS opp_sp_sl_run_value_per_100,
    pa.ch_hard_hit_pct       AS opp_sp_ch_hard_hit_pct,
    pa.ch_run_value_per_100  AS opp_sp_ch_run_value_per_100,
    -- Batter plate discipline (chase rate, contact rates, swing-and-miss)
    pd_b.oz_swing_pct        AS sc_b_oz_swing_pct,
    pd_b.iz_contact_pct      AS sc_b_iz_contact_pct,
    pd_b.oz_contact_pct      AS sc_b_oz_contact_pct,
    pd_b.whiff_pct           AS sc_b_disc_whiff_pct,
    pd_b.out_zone_pct        AS sc_b_out_zone_pct
FROM teams_today tt
JOIN recent_players rp ON rp.team_abbr = tt.team_abbr
-- Most recent rolling batter stats prior to today
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_player_batting_rolling_mat br
    WHERE br.player_id = rp.player_id
      AND br.game_date_et < %(game_date)s
    ORDER BY br.game_date_et DESC, br.game_slug DESC
    LIMIT 1
) br ON TRUE
-- Opposing starting pitcher
LEFT JOIN raw.mlb_starting_pitchers sp
    ON sp.game_slug = tt.game_slug
    AND sp.team_abbr = tt.opponent_abbr
-- Opposing SP rolling stats
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_pitcher_rolling_mat sp_r
    WHERE sp_r.player_id = sp.player_id
      AND sp_r.game_date_et < %(game_date)s
    ORDER BY sp_r.game_date_et DESC, sp_r.game_slug DESC
    LIMIT 1
) sp_r ON TRUE
-- Opponent SP handedness
LEFT JOIN raw.mlb_player_handedness opp_ph ON opp_ph.player_id = sp.player_id
-- Batter's own handedness
LEFT JOIN raw.mlb_player_handedness bh ON bh.player_id = rp.player_id
-- Most recent batter-vs-hand rolling stats prior to today (LATERAL for latest snapshot)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_batting_vs_hand_mat
    WHERE player_id = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) bvh ON TRUE
-- Cross-season rolling stats (no season boundary) — gives prior-year data early in season
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_player_batting_rolling_cross_mat
    WHERE player_id = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) bcs ON TRUE
-- Prior full-season stats — stable 162-game prior for each player
LEFT JOIN features.mlb_player_prev_season_stats_mat pss
    ON pss.player_id = rp.player_id
    AND pss.season = %(prior_season)s
-- Career H2H stats vs today's SP (most recent entry before game date)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_batter_vs_sp_mat
    WHERE batter_id    = rp.player_id
      AND pitcher_id   = sp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) h2h ON TRUE
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = (
        SELECT home_team_abbr FROM games_today
        WHERE game_slug = tt.game_slug LIMIT 1
    )
-- Home plate umpire rolling stats
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = tt.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_umpire_rolling_mat
    WHERE umpire_id = gu.umpire_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC LIMIT 1
) ur ON TRUE
-- Weather
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = tt.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = tt.game_slug LIMIT 1
)
-- Own-team lineup quality (NULL for upcoming games)
LEFT JOIN features.mlb_lineup_quality lq_own
    ON lq_own.game_slug  = tt.game_slug
    AND lq_own.team_abbr = tt.team_abbr
-- Statcast: batter's own batted-ball profile (season-level)
LEFT JOIN raw.mlb_statcast_batting sc_b
    ON sc_b.player_id = rp.player_id
    AND sc_b.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Statcast: opposing SP's batted-ball-against profile
LEFT JOIN raw.mlb_statcast_pitching sc_opp_p
    ON sc_opp_p.player_id = sp.player_id
    AND sc_opp_p.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Extended Statcast: batter sprint speed
LEFT JOIN raw.mlb_statcast_sprint_speed ss
    ON ss.player_id = rp.player_id
    AND ss.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Extended Statcast: opposing SP fastball arsenal
LEFT JOIN raw.mlb_statcast_pitcher_arsenal pa
    ON pa.player_id = sp.player_id
    AND pa.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
-- Market total (game-level run environment signal)
LEFT JOIN LATERAL (
    SELECT o.total_points AS market_total
    FROM odds.mlb_game_lines o
    JOIN games_today gt ON gt.home_team_abbr = o.home_team
    WHERE gt.game_slug = tt.game_slug
      AND o.as_of_date = tt.game_date_et
      AND o.total_points IS NOT NULL
    ORDER BY o.fetched_at_utc DESC
    LIMIT 1
) mkt_odds ON TRUE
-- Batter plate discipline (chase rate, contact rates, swing-and-miss)
LEFT JOIN raw.mlb_statcast_batter_discipline pd_b
    ON pd_b.player_id = rp.player_id
    AND pd_b.season_year = EXTRACT(YEAR FROM tt.game_date_et)::INT
WHERE br.ab_avg_10 >= %(min_ab_avg_10)s
  AND br.n_games_prev_10 >= %(min_n_games)s
ORDER BY tt.start_ts_utc, tt.game_slug, rp.player_id
"""

SQL_PROP_LINES = """
SELECT
    player_name_norm,
    stat,
    bookmaker_key,
    line,
    over_price,
    under_price,
    over_link,
    under_link
FROM odds.mlb_player_prop_lines
WHERE as_of_date = %(game_date)s
  AND bookmaker_key = 'fanduel'
ORDER BY player_name_norm, stat
"""

SQL_PLAYER_NAMES = """
SELECT DISTINCT player_id,
       COALESCE(player_name, 'id=' || player_id::text) AS player_name
FROM raw.mlb_starting_pitchers
UNION ALL
SELECT DISTINCT player_id,
       player_name
FROM raw.mlb_starting_pitchers
WHERE player_name IS NOT NULL
"""


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pitcher_artifacts(model_dir: Path):
    """Returns (xgb_k, lgb_k, feature_cols, medians, backtest)."""
    xgb_k = XGBRegressor()
    xgb_k.load_model(str(model_dir / "strikeouts_xgb.json"))

    lgb_k = None
    lgb_path = model_dir / "lgb_strikeouts.txt"
    if _HAS_LGB and lgb_path.exists():
        lgb_k = lgb.Booster(model_file=str(lgb_path))

    feat = json.loads((model_dir / "feature_columns_pitchers.json").read_text())
    meds = json.loads((model_dir / "feature_medians_pitchers.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    return xgb_k, lgb_k, feat, meds, bt


def _load_batter_artifacts(model_dir: Path):
    """Returns (xgb_hits, lgb_hits, xgb_tb, lgb_tb, xgb_hr, lgb_hr,
                xgb_walks, lgb_walks, feature_cols, medians, backtest)."""
    xgb_h = XGBRegressor()
    xgb_h.load_model(str(model_dir / "hits_xgb.json"))

    xgb_tb = XGBRegressor()
    xgb_tb.load_model(str(model_dir / "total_bases_xgb.json"))

    xgb_hr = XGBRegressor()
    xgb_hr.load_model(str(model_dir / "home_runs_xgb.json"))

    xgb_walks = XGBRegressor()
    xgb_walks.load_model(str(model_dir / "walks_xgb.json"))

    lgb_h = lgb_tb_m = lgb_hr = lgb_walks = None
    if _HAS_LGB:
        h_path    = model_dir / "lgb_hits.txt"
        tb_path   = model_dir / "lgb_total_bases.txt"
        hr_path   = model_dir / "lgb_home_runs.txt"
        walk_path = model_dir / "lgb_walks.txt"
        if h_path.exists():
            lgb_h = lgb.Booster(model_file=str(h_path))
        if tb_path.exists():
            lgb_tb_m = lgb.Booster(model_file=str(tb_path))
        if hr_path.exists():
            lgb_hr = lgb.Booster(model_file=str(hr_path))
        if walk_path.exists():
            lgb_walks = lgb.Booster(model_file=str(walk_path))

    feat = json.loads((model_dir / "feature_columns_batters.json").read_text())
    meds = json.loads((model_dir / "feature_medians_batters.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    return xgb_h, lgb_h, xgb_tb, lgb_tb_m, xgb_hr, lgb_hr, xgb_walks, lgb_walks, feat, meds, bt


# ─────────────────────────────────────────────────────────────────────────────
# Feature prep
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _prep_features(
    df: pd.DataFrame,
    meta_cols: List[str],
    feature_cols: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    """Drop meta, OHE season, coerce, add derived, fill medians, align to feature_cols."""
    X = df.drop(columns=[c for c in meta_cols if c in df.columns]).copy()

    if "season" in X.columns:
        X = pd.get_dummies(X, columns=["season"], drop_first=False, dummy_na=False)
    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric(X)

    bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if bad:
        X = pd.get_dummies(X, columns=bad, drop_first=False, dummy_na=True)

    X = add_player_prop_derived_features(X)

    # Align to training schema
    X = X.reindex(columns=feature_cols)
    for c, m in medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    return X.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization (for matching prop lines)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lower-case, strip accents, remove punctuation, collapse whitespace."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-z0-9\s]", "", ascii_str.lower())
    return re.sub(r"\s+", " ", ascii_str).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────

def _kelly_prop(edge: float, sigma: float, max_kelly: float = 0.05) -> float:
    """Simple Kelly fraction for a prop bet.
    edge = pred - line (positive = over), sigma = CI p68.
    Approximates win probability from edge / sigma ratio.
    """
    if sigma <= 0:
        return 0.0
    z = edge / sigma
    # Use logistic approximation: p ≈ 1 / (1 + exp(-z))
    p = 1.0 / (1.0 + math.exp(-z))
    # Kelly fraction = (p * 2 - 1) for 1:1 odds (simplified for -110 juice)
    k = max(0.0, (p - 0.5238) / 0.4762)  # break-even is 52.38% at -110
    return min(k, max_kelly)


# ─────────────────────────────────────────────────────────────────────────────
# DB table
# ─────────────────────────────────────────────────────────────────────────────

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id               SERIAL PRIMARY KEY,
    game_date_et     DATE        NOT NULL,
    game_slug        TEXT        NOT NULL,
    player_id        BIGINT      NOT NULL,
    player_name      TEXT,
    team_abbr        TEXT,
    stat             TEXT        NOT NULL,
    pred_value       NUMERIC,
    book_line        NUMERIC,
    edge             NUMERIC,
    kelly_fraction   NUMERIC,
    actual_value     NUMERIC,
    over_hit         BOOLEAN,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (game_date_et, game_slug, player_id, stat)
);
"""

_UPSERT_SQL = """
INSERT INTO bets.mlb_prop_predictions
    (game_date_et, game_slug, player_id, player_name, team_abbr, stat,
     pred_value, book_line, edge, kelly_fraction)
VALUES
    (%(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(team_abbr)s,
     %(stat)s, %(pred_value)s, %(book_line)s, %(edge)s, %(kelly_fraction)s)
ON CONFLICT (game_date_et, game_slug, player_id, stat) DO UPDATE SET
    player_name    = EXCLUDED.player_name,
    pred_value     = EXCLUDED.pred_value,
    book_line      = EXCLUDED.book_line,
    edge           = EXCLUDED.edge,
    kelly_fraction = EXCLUDED.kelly_fraction
"""


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_ENSURE_TABLE_SQL)
    conn.commit()


def _save_predictions(conn, rows: List[Dict]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(_UPSERT_SQL, row)
    conn.commit()
    log.info("Saved %d prop prediction rows", len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Prop line loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_prop_lines(conn, game_date: date) -> Dict[Tuple[str, str], Dict]:
    """
    Returns {(player_name_norm, stat): {line, over_link, under_link}}.
    FanDuel only. Batter props come from FD alternate markets (Over-only);
    pitcher strikeouts come from FD standard market (Over + Under).
    """
    df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
    if df.empty:
        return {}

    def _clean(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return str(v) if v else None

    result: Dict[Tuple[str, str], Dict] = {}
    for _, row in df.iterrows():
        key = (str(row["player_name_norm"]), str(row["stat"]))
        result[key] = {
            "bookmaker_key": "fanduel",
            "line": float(row["line"]) if row["line"] is not None else None,
            "over_link": _clean(row.get("over_link")),
            "under_link": _clean(row.get("under_link")),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_ensemble(
    X: pd.DataFrame,
    xgb_model: XGBRegressor,
    lgb_model,
) -> np.ndarray:
    preds = xgb_model.predict(X)
    if lgb_model is not None:
        lgb_preds = lgb_model.predict(X.values)
        preds = (preds + lgb_preds) / 2.0
    return preds


_PITCHER_META = ["game_slug", "game_date_et", "start_ts_utc", "player_id",
                 "player_name", "team_abbr", "opponent_abbr"]
_BATTER_META = ["game_slug", "game_date_et", "start_ts_utc", "player_id",
                "team_abbr", "opponent_abbr"]


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_edge(edge: Optional[float], threshold: float, direction: str = "OVER") -> str:
    if edge is None or math.isnan(edge):
        return "[no line]"
    abs_e = abs(edge)
    direction = "OVER" if edge > 0 else "UNDER"
    if abs_e >= threshold:
        return f"★ {direction} +{abs_e:.2f}"
    return f"[{direction} +{abs_e:.2f}]" if abs_e > 0 else "[no edge]"


def _collect_all_prop_links(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
) -> List[str]:
    """Collect one FD bet link per prop, taking the model's predicted direction.

    Used to build the All Props Parlay (every player, model's side, regardless of edge).
    """
    links: List[str] = []

    for row in all_pitcher_rows:
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld.get("line") is None:
            continue
        edge = pred_k - ld["line"]
        link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
        if link:
            links.append(link)

    for row in all_batter_rows:
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        for pred_col, stat_key in [
            ("pred_hits",        "batter_hits"),
            ("pred_total_bases", "batter_total_bases"),
            ("pred_home_runs",   "batter_home_runs"),
            ("pred_walks",       "batter_walks"),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld.get("line") is None:
                continue
            edge = pred_v - ld["line"]
            link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
            if link:
                links.append(link)

    return links


def _print_discord(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    game_map: Dict[str, Dict],  # game_slug -> {home, away, start_ts_utc}
    cfg: PredictConfig,
) -> List[str]:
    """Print per-game prop output. Returns edge-play links for parlay.

    DISCORD_FORMAT=1  →  compact mode: edge plays only, one line each, links
                         clickable. Games with no edges are skipped entirely.
    (no env var)      →  full table mode: all players in aligned columns.
    """
    is_discord = os.getenv("DISCORD_FORMAT") == "1"
    fd_links: List[str] = []

    def _link_name(full_name: str) -> str:
        """'Fernando Tatis Jr.' → 'Tatis Jr.',  'Michael King' → 'King'."""
        parts = full_name.split()
        if len(parts) >= 2 and parts[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii", "iv"):
            return f"{parts[-2]} {parts[-1]}"
        return parts[-1] if parts else full_name

    # ── Table-mode helpers (used only when not is_discord) ────────────────────
    NW, PW, LW, EW = 22, 5, 6, 6
    SEP = "─" * (NW + PW + LW + EW)

    def _hdr(section: str, stat: str) -> str:
        return f"{section:<{NW}}{stat:>{PW}}{'LINE':>{LW}}{'EDGE':>{EW}}"

    def _row(name: str, pred: float, pred_fmt: str, line, edge, starred: bool) -> str:
        pfx = "* " if starred else "  "
        nm = (pfx + name[:20]).ljust(NW)
        ps = pred_fmt.format(pred)
        if line is not None:
            ls = ("O" if edge > 0 else "U") + f"{line:.1f}"
            es = f"{edge:+.2f}"
        else:
            ls = es = ""
        return f"{nm}{ps:>{PW}}{ls:>{LW}}{es:>{EW}}"

    # ── Game loop ──────────────────────────────────────────────────────────────
    games_seen = sorted(
        set(r["game_slug"] for r in all_pitcher_rows + all_batter_rows),
        key=lambda s: (game_map.get(s, {}).get("start_ts_utc", ""), s),
    )

    for slug in games_seen:
        gm = game_map.get(slug, {})
        home, away = gm.get("home", "???"), gm.get("away", "???")
        start_ts = gm.get("start_ts_utc")
        time_str = ""
        if start_ts:
            try:
                dt_et = pd.Timestamp(start_ts).tz_convert(_ET)
                hour = dt_et.hour % 12 or 12
                time_str = f"{hour}:{dt_et.strftime('%M %p')} ET"
            except Exception:
                pass

        sp_rows = [r for r in all_pitcher_rows if r["game_slug"] == slug]
        bat_rows = [r for r in all_batter_rows if r["game_slug"] == slug]
        game_header = f"**{away} @ {home}**" + (f" · {time_str}" if time_str else "")

        if is_discord:
            pass  # collected below — Discord mode does one pass over all games

        else:
            # ── Full table mode ────────────────────────────────────────────────
            print(game_header)
            tbl: List[str] = []

            if sp_rows:
                tbl += [_hdr("PITCHER", "K"), SEP]
                for row in sp_rows:
                    name = row.get("player_name", f"id={row['player_id']}")
                    pred_k = row.get("pred_strikeouts")
                    if pred_k is None:
                        continue
                    norm = _normalize_name(name)
                    ld = prop_lines.get((norm, "pitcher_strikeouts"))
                    line = ld["line"] if ld else None
                    edge = (pred_k - line) if line is not None else None
                    has_edge = edge is not None and abs(edge) >= cfg.threshold_strikeouts
                    tbl.append(_row(name, pred_k, "{:.1f}", line, edge, has_edge))
                    if has_edge and ld:
                        lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
                        if lnk:
                            fd_links.append(lnk)

            if bat_rows:
                if tbl:
                    tbl.append("")
                tbl += [_hdr("HITS", "H"), SEP]
                for row in bat_rows:
                    name = row.get("player_name", f"id={row['player_id']}")
                    norm = _normalize_name(name)
                    pred_h = row.get("pred_hits")
                    if pred_h is None:
                        continue
                    _ci = math.sqrt(10.0 / max(row.get("n_games_prev_10") or 1, 1))
                    ld = prop_lines.get((norm, "batter_hits"))
                    line = ld["line"] if ld else None
                    edge = (pred_h - line) if line is not None else None
                    has_edge = edge is not None and abs(edge) >= cfg.threshold_hits * _ci
                    tbl.append(_row(name, pred_h, "{:.2f}", line, edge, has_edge))
                    if has_edge and ld:
                        lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
                        if lnk:
                            fd_links.append(lnk)

            for stat_lbl, hdr_lbl, pred_col, stat_key, thresh, fmt in [
                ("TB", "TOTAL BASES", "pred_total_bases", "batter_total_bases", cfg.threshold_total_bases, "{:.2f}"),
                ("HR", "HOME RUNS",   "pred_home_runs",   "batter_home_runs",   cfg.threshold_home_runs,   "{:.3f}"),
                ("BB", "WALKS",       "pred_walks",       "batter_walks",       cfg.threshold_walks,        "{:.2f}"),
            ]:
                stat_edges: List[tuple] = []
                for row in bat_rows:
                    name = row.get("player_name", f"id={row['player_id']}")
                    norm = _normalize_name(name)
                    pred_v = row.get(pred_col)
                    if pred_v is None:
                        continue
                    _ci = math.sqrt(10.0 / max(row.get("n_games_prev_10") or 1, 1))
                    ld = prop_lines.get((norm, stat_key))
                    if not ld or ld.get("line") is None:
                        continue
                    line = ld["line"]
                    edge = pred_v - line
                    if abs(edge) >= thresh * _ci:
                        stat_edges.append((name, pred_v, line, edge, ld))
                if stat_edges:
                    tbl += ["", _hdr(hdr_lbl, stat_lbl), SEP]
                    for name, pred_v, line, edge, ld in stat_edges:
                        tbl.append(_row(name, pred_v, fmt, line, edge, True))
                        lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
                        if lnk:
                            fd_links.append(lnk)

            for line_txt in tbl:
                print(line_txt)
            print("")

    if not is_discord:
        return fd_links

    # ── Discord mode: collect all edge plays across every game, then print ─────
    k_plays:  List[Dict] = []
    tb_plays: List[Dict] = []
    h_plays:  List[Dict] = []
    hr_plays: List[Dict] = []

    for row in all_pitcher_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(name)
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld.get("line") is None:
            continue
        line = ld["line"]
        edge = pred_k - line
        if abs(edge) < cfg.threshold_strikeouts:
            continue
        lnk = ld.get("over_link") if edge > 0 else ld.get("under_link")
        k_plays.append({
            "name": name, "team": row.get("team_abbr", ""),
            "pred": pred_k, "line": line, "edge": edge,
            "lnk": lnk, "book": "FD",
        })

    # Batter sections: sorted by raw prediction (not edge).
    # TB  → who will rack up the most total bases
    # H   → who is most likely to get a hit
    # HR  → who is most likely to homer
    # FD alternate markets are Over-only so no bet links; FD line shown for reference.
    for row in all_batter_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        norm = _normalize_name(name)
        team = row.get("team_abbr", "")
        for pred_col, stat_key, plays_list in [
            ("pred_hits",        "batter_hits",        h_plays),
            ("pred_total_bases", "batter_total_bases", tb_plays),
            ("pred_home_runs",   "batter_home_runs",   hr_plays),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            line = ld["line"] if ld else None
            plays_list.append({
                "name": name, "team": team,
                "pred": pred_v, "line": line,
            })

    k_plays.sort(key=lambda x: abs(x["edge"]), reverse=True)
    h_plays.sort(key=lambda x: x["pred"], reverse=True)
    tb_plays.sort(key=lambda x: x["pred"], reverse=True)
    hr_plays.sort(key=lambda x: x["pred"], reverse=True)

    def _section_k(title: str, plays: List[Dict], stat: str, pred_fmt: str) -> None:
        """K section: edge-based, with FD bet links."""
        if not plays:
            return
        print(f"**{title}**")
        for p in plays[:10]:
            d = "O" if p["edge"] > 0 else "U"
            ls = f"{d}{p['line']:.1f}"
            ps = pred_fmt.format(p["pred"])
            team_str = f" ({p['team']})" if p["team"] else ""
            link_str = f" [Bet FD](<{p['lnk']}>)" if p["lnk"] else ""
            print(f"★ {_link_name(p['name'])}{team_str} {stat} {ls} → {ps}{link_str}")
        print("")

    def _section_batter(title: str, plays: List[Dict], stat: str, pred_fmt: str) -> None:
        """Batter section: prediction-sorted, FD line shown for reference, no bet link."""
        if not plays:
            return
        print(f"**{title}**")
        for p in plays[:10]:
            ps = pred_fmt.format(p["pred"])
            team_str = f" ({p['team']})" if p["team"] else ""
            line_str = f" O{p['line']:.1f}" if p["line"] is not None else ""
            print(f"★ {_link_name(p['name'])}{team_str} {stat}{line_str} → {ps}")
        print("")

    def _parlay(title: str, links: List) -> None:
        dedup = list(dict.fromkeys(l for l in links if l))
        if not dedup:
            return
        n_chunks = math.ceil(len(dedup) / 25)
        for i in range(0, len(dedup), 25):
            url = build_fd_parlay_url(dedup[i:i + 25])
            if url:
                sfx = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
                print(f"**{title}{sfx}** [FD]({url})")

    _section_k("Top Strikeouts",  k_plays,  "K",  "{:.1f}")
    _section_batter("Top Total Bases", tb_plays, "TB", "{:.2f}")
    _section_batter("Top Hits",        h_plays,  "H",  "{:.2f}")
    _section_batter("Top Home Runs",   hr_plays, "HR", "{:.2f}")

    _parlay("All Ks Parlay",   [p["lnk"] for p in k_plays])
    _parlay("Best Props Parlay", [p["lnk"] for p in k_plays[:10]])

    return []  # parlays already printed; outer parlay logic skipped


def _print_best_bets(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    cfg: PredictConfig,
) -> List[str]:
    """Print best bets ranked by |edge| as a table. Returns FD links for parlay."""
    is_discord = os.getenv("DISCORD_FORMAT") == "1"
    fd_links: List[str] = []
    best: List[Dict] = []

    for row in all_pitcher_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(name)
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld["line"] is None:
            continue
        edge = pred_k - ld["line"]
        if abs(edge) >= cfg.threshold_strikeouts:
            best.append({
                "name": name, "stat": "K", "pred": pred_k,
                "line": ld["line"], "edge": edge,
                "bet_link": ld.get("over_link") if edge > 0 else ld.get("under_link"),
                "team": row.get("team_abbr", ""),
            })

    for row in all_batter_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        norm = _normalize_name(name)
        _n_g = row.get("n_games_prev_10") or 0
        _ci_scale = math.sqrt(10.0 / max(_n_g, 1))
        for stat, pred_col, stat_key, threshold in [
            ("H",  "pred_hits",        "batter_hits",        cfg.threshold_hits),
            ("TB", "pred_total_bases",  "batter_total_bases", cfg.threshold_total_bases),
            ("HR", "pred_home_runs",    "batter_home_runs",   cfg.threshold_home_runs),
            ("BB", "pred_walks",        "batter_walks",       cfg.threshold_walks),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld["line"] is None:
                continue
            edge = pred_v - ld["line"]
            if abs(edge) >= threshold * _ci_scale:
                best.append({
                    "name": name, "stat": stat, "pred": pred_v,
                    "line": ld["line"], "edge": edge,
                    "bet_link": ld.get("over_link") if edge > 0 else ld.get("under_link"),
                    "bookmaker_key": ld.get("bookmaker_key", ""),
                    "team": row.get("team_abbr", ""),
                })

    best.sort(key=lambda r: abs(r["edge"]), reverse=True)

    if best:
        print("─" * 40)
        print("**Best Props (ranked by |edge|)**")

        for b in best[:10]:
            parts = b["name"].split()
            if len(parts) >= 2 and parts[-1].rstrip(".").lower() in ("jr", "sr", "ii", "iii", "iv"):
                short = f"{parts[-2]} {parts[-1]} ({b['team']})"
            else:
                short = f"{parts[-1]} ({b['team']})"
            ps = f"{b['pred']:.1f}" if b["stat"] == "K" else f"{b['pred']:.2f}"
            d = "O" if b["edge"] > 0 else "U"
            ls = f"{d}{b['line']:.1f}"
            es = f"{b['edge']:+.2f}"
            lnk = b.get("bet_link")
            link_str = f" [Bet FD](<{lnk}>)" if lnk else " [FD]"
            print(f"★ {short} {b['stat']} {ls} → {ps}{link_str}")
            if lnk:
                fd_links.append(lnk)

    return fd_links


def _print_top_hr_hitters(
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    top_n: int = 10,
) -> None:
    """Print the top predicted HR hitters for the day, sorted by pred_home_runs desc."""
    hr_rows = [r for r in all_batter_rows if r.get("pred_home_runs") is not None]
    if not hr_rows:
        return
    hr_rows.sort(key=lambda r: r["pred_home_runs"], reverse=True)

    print("─" * 40)
    print(f"**Top HR Hitters Today (top {min(top_n, len(hr_rows))})**")
    for r in hr_rows[:top_n]:
        name  = r.get("player_name", f"id={r['player_id']}")
        norm  = _normalize_name(name)
        team  = r.get("team_abbr", "?")
        opp   = r.get("opponent_abbr", "?")
        phr   = r["pred_home_runs"]

        ld    = prop_lines.get((norm, "batter_home_runs"))
        line  = ld["line"] if ld else None

        if line is not None:
            edge = phr - line
            dir_str = "O" if edge > 0 else "U"
            bet_link = ld.get("over_link") if edge > 0 else ld.get("under_link")
            link_str = f"  [Bet FD](<{bet_link}>)" if bet_link else ""
            print(
                f"  {name} ({team} vs {opp}) — pred {phr:.3f} | "
                f"{dir_str}{line:.1f} edge {edge:+.2f}{link_str}"
            )
        else:
            print(f"  {name} ({team} vs {opp}) — pred {phr:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_props(cfg: PredictConfig) -> None:
    et_date = cfg.et_date or datetime.now(tz=_ET).date()

    # ── Load models ───────────────────────────────────────────────────────────
    model_dir = cfg.model_dir
    pitcher_artifacts_ok = (
        (model_dir / "strikeouts_xgb.json").exists() and
        (model_dir / "feature_columns_pitchers.json").exists()
    )
    batter_artifacts_ok = (
        (model_dir / "hits_xgb.json").exists() and
        (model_dir / "total_bases_xgb.json").exists() and
        (model_dir / "home_runs_xgb.json").exists() and
        (model_dir / "walks_xgb.json").exists() and
        (model_dir / "feature_columns_batters.json").exists()
    )

    xgb_k = lgb_k = feat_p = meds_p = bt = None
    xgb_h = lgb_h = xgb_tb_m = lgb_tb_m = None
    xgb_hr = lgb_hr = xgb_walks_m = lgb_walks_m = feat_b = meds_b = None

    if pitcher_artifacts_ok:
        try:
            xgb_k, lgb_k, feat_p, meds_p, bt = _load_pitcher_artifacts(model_dir)
        except Exception:
            log.exception("Failed to load pitcher artifacts")
            pitcher_artifacts_ok = False

    if batter_artifacts_ok:
        try:
            (xgb_h, lgb_h, xgb_tb_m, lgb_tb_m,
             xgb_hr, lgb_hr, xgb_walks_m, lgb_walks_m,
             feat_b, meds_b, bt) = _load_batter_artifacts(model_dir)
        except Exception:
            log.exception("Failed to load batter artifacts")
            batter_artifacts_ok = False

    if not pitcher_artifacts_ok and not batter_artifacts_ok:
        log.warning("No prop models found. Run train_player_prop_models first.")
        print("_(No prop models — run train_player_prop_models first)_")
        return

    # ── Connect and fetch data ─────────────────────────────────────────────
    conn = psycopg2.connect(cfg.pg_dsn)
    _ensure_schema(conn)

    prop_lines = _load_prop_lines(conn, et_date)
    log.info("Loaded %d prop line entries for %s", len(prop_lines), et_date)

    # ── Pitcher predictions ────────────────────────────────────────────────
    all_pitcher_rows: List[Dict] = []
    db_rows: List[Dict] = []

    if pitcher_artifacts_ok:
        df_p = pd.read_sql(SQL_PITCHER_SNAPSHOTS, conn, params={"game_date": et_date})
        log.info("Pitcher snapshots: %d rows", len(df_p))

        if not df_p.empty:
            X_p = _prep_features(df_p, _PITCHER_META, feat_p, meds_p)
            pred_k = _predict_ensemble(X_p, xgb_k, lgb_k)
            sigma_k = bt.get("ci_strikeouts") if bt else None

            for i, (_, row) in enumerate(df_p.iterrows()):
                pk = max(0.0, float(pred_k[i]))
                name = row.get("player_name", f"id={row['player_id']}")
                norm = _normalize_name(name)
                ld = prop_lines.get((norm, "pitcher_strikeouts"))
                line = ld["line"] if ld else None
                edge = (pk - line) if line is not None else None
                kel = _kelly_prop(abs(edge), sigma_k or cfg.threshold_strikeouts * 2) \
                      if edge is not None else 0.0

                r = {
                    "game_slug": row["game_slug"],
                    "game_date_et": et_date,
                    "player_id": int(row["player_id"]),
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "is_home": row.get("is_home"),
                    "opponent_abbr": row.get("opponent_abbr"),
                    "start_ts_utc": row.get("start_ts_utc"),
                    "pred_strikeouts": pk,
                }
                all_pitcher_rows.append(r)
                db_rows.append({
                    "game_date_et": et_date,
                    "game_slug": row["game_slug"],
                    "player_id": int(row["player_id"]),
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "stat": "pitcher_strikeouts",
                    "pred_value": round(pk, 3),
                    "book_line": line,
                    "edge": round(edge, 3) if edge is not None else None,
                    "kelly_fraction": round(kel, 4),
                })

    # ── Batter predictions ─────────────────────────────────────────────────
    all_batter_rows: List[Dict] = []

    if batter_artifacts_ok:
        prior_season = f"{et_date.year - 1}-regular"
        df_b = pd.read_sql(
            SQL_BATTER_SNAPSHOTS, conn,
            params={
                "game_date": et_date,
                "prior_season": prior_season,
                "min_ab_avg_10": cfg.min_ab_avg_10,
                "min_n_games": cfg.min_n_games,
            }
        )
        log.info("Batter snapshots: %d rows", len(df_b))

        # Fetch player names from boxscore (first + last name) and SP table as fallback
        name_df = pd.read_sql(
            """SELECT player_id,
                  TRIM(first_name || ' ' || last_name) AS player_name
               FROM (
                   SELECT DISTINCT ON (player_id)
                       player_id,
                       first_name,
                       last_name
                   FROM raw.mlb_boxscore_player_stats
                   WHERE first_name IS NOT NULL AND last_name IS NOT NULL
                   ORDER BY player_id, game_slug DESC
               ) sub
               UNION ALL
               SELECT player_id, player_name
               FROM raw.mlb_starting_pitchers
               WHERE player_name IS NOT NULL""",
            conn
        )
        # Deduplicate: keep first occurrence
        name_df = name_df.drop_duplicates(subset=["player_id"], keep="first")
        name_map = dict(zip(name_df["player_id"], name_df["player_name"]))

        if not df_b.empty:
            X_b = _prep_features(df_b, _BATTER_META, feat_b, meds_b)
            pred_h     = _predict_ensemble(X_b, xgb_h,      lgb_h)
            pred_tb    = _predict_ensemble(X_b, xgb_tb_m,   lgb_tb_m)
            pred_hr    = _predict_ensemble(X_b, xgb_hr,     lgb_hr)
            pred_walks = _predict_ensemble(X_b, xgb_walks_m, lgb_walks_m)
            sigma_h     = bt.get("ci_hits")        if bt else None
            sigma_tb    = bt.get("ci_total_bases") if bt else None
            sigma_hr    = bt.get("ci_home_runs")   if bt else None
            sigma_walks = bt.get("ci_walks")       if bt else None

            seen = set()
            for i, (_, row) in enumerate(df_b.iterrows()):
                pid = int(row["player_id"])
                slug = row["game_slug"]
                key = (slug, pid)
                if key in seen:
                    continue
                seen.add(key)

                name = name_map.get(pid, f"id={pid}")
                norm = _normalize_name(name)
                ph    = max(0.0, float(pred_h[i]))
                ptb   = max(0.0, float(pred_tb[i]))
                phr   = max(0.0, float(pred_hr[i]))
                pwalk = max(0.0, float(pred_walks[i]))

                r = {
                    "game_slug": slug,
                    "game_date_et": et_date,
                    "player_id": pid,
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "is_home": row.get("is_home"),
                    "opponent_abbr": row.get("opponent_abbr"),
                    "start_ts_utc": row.get("start_ts_utc"),
                    "n_games_prev_10": int(row.get("n_games_prev_10") or 0),
                    "pred_hits": ph,
                    "pred_total_bases": ptb,
                    "pred_home_runs": phr,
                    "pred_walks": pwalk,
                }
                all_batter_rows.append(r)

                for stat_key, stat_label, pred_v, sigma in [
                    ("batter_hits",        "batter_hits",        ph,    sigma_h),
                    ("batter_total_bases", "batter_total_bases", ptb,   sigma_tb),
                    ("batter_home_runs",   "batter_home_runs",   phr,   sigma_hr),
                    ("batter_walks",       "batter_walks",       pwalk, sigma_walks),
                ]:
                    ld = prop_lines.get((norm, stat_key))
                    line = ld["line"] if ld else None
                    edge = (pred_v - line) if line is not None else None
                    kel = _kelly_prop(abs(edge), sigma or 0.5) if edge is not None else 0.0
                    db_rows.append({
                        "game_date_et": et_date,
                        "game_slug": slug,
                        "player_id": pid,
                        "player_name": name,
                        "team_abbr": row.get("team_abbr"),
                        "stat": stat_label,
                        "pred_value": round(pred_v, 3),
                        "book_line": line,
                        "edge": round(edge, 3) if edge is not None else None,
                        "kelly_fraction": round(kel, 4),
                    })

    # ── Save to DB ────────────────────────────────────────────────────────
    _save_predictions(conn, db_rows)
    conn.close()

    if not all_pitcher_rows and not all_batter_rows:
        print(f"_(No prop data for {et_date})_")
        return

    # ── Build game map for grouping ───────────────────────────────────────
    game_map: Dict[str, Dict] = {}
    for row in all_pitcher_rows + all_batter_rows:
        slug = row["game_slug"]
        if slug not in game_map:
            parts = slug.split("-")
            away = parts[1] if len(parts) > 2 else "?"
            home = parts[2] if len(parts) > 2 else "?"
            game_map[slug] = {
                "home": home,
                "away": away,
                "start_ts_utc": row.get("start_ts_utc"),
            }

    # ── Print output ──────────────────────────────────────────────────────
    header = f"⚾ **MLB Props — {et_date.strftime('%b')} {et_date.day}**"
    print(header)
    print("")

    is_discord = os.getenv("DISCORD_FORMAT") == "1"

    _print_discord(all_pitcher_rows, all_batter_rows, prop_lines, game_map, cfg)

    if not is_discord:
        fd_links = _print_best_bets(all_pitcher_rows, all_batter_rows, prop_lines, cfg)

        # Top HR hitters leaderboard
        _print_top_hr_hitters(all_batter_rows, prop_lines)

        # Best props parlay (high-edge bets only) — chunked at 25 legs
        seen_bp: set[str] = set()
        fd_links_dedup = [l for l in fd_links if l not in seen_bp and not seen_bp.add(l)]  # type: ignore[func-returns-value]
        n_bp = len(fd_links_dedup)
        n_bp_chunks = math.ceil(n_bp / 25) if n_bp else 0
        for i in range(0, n_bp, 25):
            chunk = fd_links_dedup[i:i + 25]
            parlay_url = build_fd_parlay_url(chunk)
            if parlay_url:
                suffix = f" {i // 25 + 1}/{n_bp_chunks}" if n_bp_chunks > 1 else ""
                print(f"\n**Best Props Parlay{suffix}** [FD]({parlay_url})")

        # All props parlay — every player, model's predicted direction, chunked at 25 legs
        all_links = _collect_all_prop_links(all_pitcher_rows, all_batter_rows, prop_lines)
        seen: set[str] = set()
        all_links_dedup = [l for l in all_links if l not in seen and not seen.add(l)]  # type: ignore[func-returns-value]
        n_all = len(all_links_dedup)
        n_chunks = math.ceil(n_all / 25) if n_all else 0
        for i in range(0, n_all, 25):
            chunk = all_links_dedup[i:i + 25]
            ap = build_fd_parlay_url(chunk)
            if ap:
                suffix = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
                print(f"\n**All Props Parlay{suffix}** [FD]({ap})")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    et_date = None
    if args.date:
        from datetime import date as _date
        et_date = _date.fromisoformat(args.date)
    elif os.getenv("MLB_ET_DATE"):
        from datetime import date as _date
        et_date = _date.fromisoformat(os.getenv("MLB_ET_DATE"))

    cfg = PredictConfig(et_date=et_date)
    predict_props(cfg)


if __name__ == "__main__":
    main()
