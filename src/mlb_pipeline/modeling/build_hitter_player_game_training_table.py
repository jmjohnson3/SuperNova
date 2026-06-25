"""Build one-row-per-hitter-game training examples.

This table is intentionally player-game first, not offer first.  Actual plate
appearances, lineup slot, and hitter outcomes come from boxscore/gamelog data;
offer-level rows only join in as optional projection context for validation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2

from .prop_market_training import ensure_game_training_features

from mlb_pipeline.db import PG_DSN as _PG_DSN
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


@dataclass(frozen=True)
class HitterPlayerGameTrainingConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 365
    date_from: date | None = None
    date_to: date | None = None
    report_file: str | None = None


DDL = """
CREATE SCHEMA IF NOT EXISTS features;
CREATE TABLE IF NOT EXISTS features.mlb_hitter_player_game_training (
    id BIGSERIAL PRIMARY KEY,
    game_date_et DATE NOT NULL,
    game_slug TEXT NOT NULL,
    player_id BIGINT NOT NULL,
    player_name TEXT,
    player_name_norm TEXT,
    team_abbr TEXT,
    opponent_abbr TEXT,
    is_home NUMERIC,
    lineup_slot NUMERIC,
    lineup_source TEXT,
    confirmed_starter BOOLEAN,
    starter_status_source TEXT,
    primary_position TEXT,
    batter_hand TEXT,
    opp_sp_id BIGINT,
    opp_sp_hand TEXT,
    opp_sp_hand_l NUMERIC,
    team_implied_runs NUMERIC,
    opponent_implied_runs NUMERIC,
    game_total_line NUMERIC,
    venue_id INTEGER,
    park_run_factor NUMERIC,
    park_hr_factor NUMERIC,
    park_babip_factor NUMERIC,
    temperature_f NUMERIC,
    wind_speed_mph NUMERIC,
    wind_sin NUMERIC,
    wind_cos NUMERIC,
    precip_prob_pct NUMERIC,
    is_dome NUMERIC,
    is_day_game NUMERIC,
    weather_pregame_flag NUMERIC,
    own_lineup_xwoba_avg NUMERIC,
    own_lineup_xslg_avg NUMERIC,
    own_lineup_barrel_avg NUMERIC,
    own_lineup_hard_hit_avg NUMERIC,
    own_lineup_k_pct_cv NUMERIC,
    own_lineup_pct_lhb NUMERIC,
    lineup_confirmed_flag NUMERIC,
    confirmed_team_lineup_slots NUMERIC,
    team_lineup_confirmed_flag NUMERIC,
    lineup_boxscore_proxy_flag NUMERIC,
    lineup_slot_x_team_implied_runs NUMERIC,
    opp_sp_k_pct_10 NUMERIC,
    opp_sp_bb_pct NUMERIC,
    opp_sp_xwoba NUMERIC,
    opp_sp_hard_hit_pct NUMERIC,
    opp_sp_whiff_pct NUMERIC,
    opp_bp_era_10 NUMERIC,
    opp_bp_whip_10 NUMERIC,
    opp_bp_k9_10 NUMERIC,
    opp_bp_ip_last_3 NUMERIC,
    opp_bp_ip_last_7 NUMERIC,
    opp_team_k_pct_10 NUMERIC,
    opp_team_avg_10 NUMERIC,
    opp_team_obp_10 NUMERIC,
    opp_team_slg_10 NUMERIC,
    batter_vs_hand_hits_avg_10 NUMERIC,
    batter_vs_hand_tb_avg_10 NUMERIC,
    batter_vs_hand_hr_avg_10 NUMERIC,
    batter_vs_hand_iso_avg_10 NUMERIC,
    batter_vs_hand_k_rate_10 NUMERIC,
    batter_vs_hand_games_10 NUMERIC,
    batter_vs_rp_ba_30 NUMERIC,
    batter_vs_rp_slg_30 NUMERIC,
    batter_vs_rp_hr_rate_30 NUMERIC,
    batter_vs_rp_k_rate_30 NUMERIC,
    batter_sc_barrel_rate NUMERIC,
    batter_sc_hard_hit_pct NUMERIC,
    batter_sc_avg_exit_velo NUMERIC,
    batter_sc_avg_launch_angle NUMERIC,
    batter_sc_sweet_spot_pct NUMERIC,
    batter_sc_fb_pct NUMERIC,
    batter_sc_gb_pct NUMERIC,
    batter_sc_ld_pct NUMERIC,
    batter_sc_xba NUMERIC,
    batter_sc_xslg NUMERIC,
    batter_sc_xwoba NUMERIC,
    batter_sc_xiso NUMERIC,
    batter_sc_pull_pct NUMERIC,
    batter_sc_opposite_pct NUMERIC,
    batter_sc_popup_pct NUMERIC,
    batter_sc_brl_pa NUMERIC,
    batter_sprint_speed NUMERIC,
    batter_disc_oz_swing_pct NUMERIC,
    batter_disc_iz_contact_pct NUMERIC,
    batter_disc_oz_contact_pct NUMERIC,
    batter_disc_whiff_pct NUMERIC,
    batter_disc_out_zone_pct NUMERIC,
    batter_disc_k_pct NUMERIC,
    batter_disc_bb_pct NUMERIC,
    opp_sp_sc_barrel_rate NUMERIC,
    opp_sp_sc_hard_hit_pct NUMERIC,
    opp_sp_sc_avg_exit_velo NUMERIC,
    opp_sp_sc_avg_launch_angle NUMERIC,
    opp_sp_sc_xba NUMERIC,
    opp_sp_sc_xslg NUMERIC,
    opp_sp_sc_xwoba NUMERIC,
    opp_sp_sc_xiso NUMERIC,
    opp_sp_fb_pct NUMERIC,
    opp_sp_fb_hard_hit_pct NUMERIC,
    opp_sp_fb_xwoba NUMERIC,
    opp_sp_fb_run_value_per_100 NUMERIC,
    opp_sp_fb_whiff_pct NUMERIC,
    opp_sp_fb_k_pct NUMERIC,
    opp_sp_si_pct NUMERIC,
    opp_sp_si_hard_hit_pct NUMERIC,
    opp_sp_si_xwoba NUMERIC,
    opp_sp_si_whiff_pct NUMERIC,
    opp_sp_si_k_pct NUMERIC,
    opp_sp_sl_pct NUMERIC,
    opp_sp_sl_hard_hit_pct NUMERIC,
    opp_sp_sl_xwoba NUMERIC,
    opp_sp_sl_run_value_per_100 NUMERIC,
    opp_sp_sl_whiff_pct NUMERIC,
    opp_sp_sl_k_pct NUMERIC,
    opp_sp_ch_pct NUMERIC,
    opp_sp_ch_hard_hit_pct NUMERIC,
    opp_sp_ch_xwoba NUMERIC,
    opp_sp_ch_run_value_per_100 NUMERIC,
    opp_sp_ch_whiff_pct NUMERIC,
    opp_sp_ch_k_pct NUMERIC,
    opp_sp_fastball_family_pct NUMERIC,
    opp_sp_pitch_diversity NUMERIC,
    projected_pa NUMERIC,
    pa_games INTEGER,
    model_pred_hits NUMERIC,
    model_pred_total_bases NUMERIC,
    model_pred_home_runs NUMERIC,
    prop_example_rows INTEGER,
    prop_offer_rows INTEGER,
    paired_price_rate NUMERIC,
    same_book_pair_rate NUMERIC,
    valid_clv_rate NUMERIC,
    first_lock_at_utc TIMESTAMPTZ,
    actual_pa NUMERIC,
    actual_at_bats NUMERIC,
    actual_hits NUMERIC,
    actual_singles NUMERIC,
    actual_doubles NUMERIC,
    actual_triples NUMERIC,
    actual_home_runs NUMERIC,
    actual_total_bases NUMERIC,
    actual_walks NUMERIC,
    actual_strikeouts NUMERIC,
    low_pa_flag NUMERIC,
    source_updated_at TIMESTAMPTZ,
    row_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (game_slug, player_id)
);
CREATE INDEX IF NOT EXISTS idx_mlb_hitter_pg_training_date
    ON features.mlb_hitter_player_game_training (game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_hitter_pg_training_player
    ON features.mlb_hitter_player_game_training (player_id, game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_hitter_pg_training_lineup
    ON features.mlb_hitter_player_game_training (lineup_slot, confirmed_starter);

ALTER TABLE features.mlb_hitter_player_game_training
    ADD COLUMN IF NOT EXISTS venue_id INTEGER,
    ADD COLUMN IF NOT EXISTS park_run_factor NUMERIC,
    ADD COLUMN IF NOT EXISTS park_hr_factor NUMERIC,
    ADD COLUMN IF NOT EXISTS park_babip_factor NUMERIC,
    ADD COLUMN IF NOT EXISTS temperature_f NUMERIC,
    ADD COLUMN IF NOT EXISTS wind_speed_mph NUMERIC,
    ADD COLUMN IF NOT EXISTS wind_sin NUMERIC,
    ADD COLUMN IF NOT EXISTS wind_cos NUMERIC,
    ADD COLUMN IF NOT EXISTS precip_prob_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS is_dome NUMERIC,
    ADD COLUMN IF NOT EXISTS is_day_game NUMERIC,
    ADD COLUMN IF NOT EXISTS weather_pregame_flag NUMERIC,
    ADD COLUMN IF NOT EXISTS own_lineup_xwoba_avg NUMERIC,
    ADD COLUMN IF NOT EXISTS own_lineup_xslg_avg NUMERIC,
    ADD COLUMN IF NOT EXISTS own_lineup_barrel_avg NUMERIC,
    ADD COLUMN IF NOT EXISTS own_lineup_hard_hit_avg NUMERIC,
    ADD COLUMN IF NOT EXISTS own_lineup_k_pct_cv NUMERIC,
    ADD COLUMN IF NOT EXISTS own_lineup_pct_lhb NUMERIC,
    ADD COLUMN IF NOT EXISTS lineup_confirmed_flag NUMERIC,
    ADD COLUMN IF NOT EXISTS confirmed_team_lineup_slots NUMERIC,
    ADD COLUMN IF NOT EXISTS team_lineup_confirmed_flag NUMERIC,
    ADD COLUMN IF NOT EXISTS lineup_boxscore_proxy_flag NUMERIC,
    ADD COLUMN IF NOT EXISTS lineup_slot_x_team_implied_runs NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_barrel_rate NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_hard_hit_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_avg_exit_velo NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_avg_launch_angle NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_sweet_spot_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_fb_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_gb_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_ld_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_xba NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_xslg NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_xwoba NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_xiso NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_pull_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_opposite_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_popup_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sc_brl_pa NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_sprint_speed NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_oz_swing_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_iz_contact_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_oz_contact_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_whiff_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_out_zone_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_k_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS batter_disc_bb_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_barrel_rate NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_hard_hit_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_avg_exit_velo NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_avg_launch_angle NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_xba NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_xslg NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_xwoba NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sc_xiso NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fb_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fb_hard_hit_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fb_xwoba NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fb_run_value_per_100 NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fb_whiff_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fb_k_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_si_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_si_hard_hit_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_si_xwoba NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_si_whiff_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_si_k_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sl_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sl_hard_hit_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sl_xwoba NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sl_run_value_per_100 NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sl_whiff_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_sl_k_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_ch_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_ch_hard_hit_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_ch_xwoba NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_ch_run_value_per_100 NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_ch_whiff_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_ch_k_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_fastball_family_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS opp_sp_pitch_diversity NUMERIC;
"""


BUILD_SQL = """
WITH base AS (
    SELECT
        COALESCE(g.game_date_et, gl.game_date_et) AS game_date_et,
        bps.game_slug,
        bps.player_id::bigint AS player_id,
        NULLIF(TRIM(CONCAT_WS(' ', bps.first_name, bps.last_name)), '') AS player_name,
        TRIM(LOWER(REGEXP_REPLACE(
            COALESCE(NULLIF(CONCAT_WS(' ', bps.first_name, bps.last_name), ''), ''),
            '[^a-z0-9]+', ' ', 'g'
        ))) AS player_name_norm,
        bps.team_abbr,
        CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN g.away_team_abbr
            WHEN bps.team_abbr = g.away_team_abbr THEN g.home_team_abbr
            ELSE NULL
        END AS opponent_abbr,
        CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN 1.0
            WHEN bps.team_abbr = g.away_team_abbr THEN 0.0
            ELSE NULL
        END AS is_home,
        COALESCE(lu.batting_order, bps.batting_order)::float AS lineup_slot,
        CASE
            WHEN lu.batting_order IS NOT NULL THEN lu.lineup_source
            WHEN bps.batting_order IS NOT NULL THEN 'boxscore_actual'
            ELSE NULL
        END AS lineup_source,
        CASE
            WHEN lu.batting_order BETWEEN 1 AND 9 THEN TRUE
            WHEN lu.batting_order IS NULL AND bps.batting_order BETWEEN 1 AND 9
                 AND COALESCE(NULLIF(bps.stats #>> '{batting,plateAppearances}', '')::float, 0.0) >= 3.0
                 THEN TRUE
            ELSE FALSE
        END AS confirmed_starter,
        CASE
            WHEN lu.batting_order IS NOT NULL THEN 'raw_lineups_name_match'
            WHEN bps.batting_order IS NOT NULL THEN 'boxscore_batting_order_pa_proxy'
            ELSE 'unknown'
        END AS starter_status_source,
        bps.primary_position,
        bh.bat_side AS batter_hand,
        CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN g.away_sp_id
            WHEN bps.team_abbr = g.away_team_abbr THEN g.home_sp_id
            ELSE NULL
        END AS opp_sp_id,
        CASE
            WHEN bps.team_abbr = g.home_team_abbr AND gf.away_sp_pitch_hand_l = 1 THEN 'L'
            WHEN bps.team_abbr = g.home_team_abbr AND gf.away_sp_pitch_hand_l = 0 THEN 'R'
            WHEN bps.team_abbr = g.away_team_abbr AND gf.home_sp_pitch_hand_l = 1 THEN 'L'
            WHEN bps.team_abbr = g.away_team_abbr AND gf.home_sp_pitch_hand_l = 0 THEN 'R'
            ELSE opp_ph.pitch_hand
        END AS opp_sp_hand,
        CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_sp_pitch_hand_l
            WHEN bps.team_abbr = g.away_team_abbr THEN gf.home_sp_pitch_hand_l
            ELSE NULL
        END AS opp_sp_hand_l,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.home_implied_runs ELSE gf.away_implied_runs END AS team_implied_runs,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_implied_runs ELSE gf.home_implied_runs END AS opponent_implied_runs,
        gf.total_line AS game_total_line,
        g.venue_id::int AS venue_id,
        gf.park_run_factor,
        gf.park_hr_factor,
        pbf.park_babip_avg AS park_babip_factor,
        wx.temperature_f,
        CASE WHEN v.roof_type = 'dome' THEN 0.0 ELSE wx.wind_speed_mph::float END AS wind_speed_mph,
        CASE WHEN v.roof_type = 'dome' THEN 0.0 ELSE SIN(RADIANS(wx.wind_direction_deg::float)) END AS wind_sin,
        CASE WHEN v.roof_type = 'dome' THEN 0.0 ELSE COS(RADIANS(wx.wind_direction_deg::float)) END AS wind_cos,
        wx.precip_prob_pct::float AS precip_prob_pct,
        CASE WHEN v.roof_type = 'dome' THEN 1.0 ELSE 0.0 END AS is_dome,
        CASE
            WHEN v.roof_type = 'dome' THEN 0.0
            WHEN EXTRACT(HOUR FROM g.start_ts_utc AT TIME ZONE 'America/New_York') < 17 THEN 1.0
            ELSE 0.0
        END AS is_day_game,
        CASE WHEN wx.game_slug IS NOT NULL THEN 1.0 ELSE 0.0 END AS weather_pregame_flag,
        lq_own.lineup_xwoba_avg AS own_lineup_xwoba_avg,
        lq_own.lineup_xslg_avg AS own_lineup_xslg_avg,
        lq_own.lineup_barrel_avg AS own_lineup_barrel_avg,
        lq_own.lineup_hard_hit_avg AS own_lineup_hard_hit_avg,
        lq_own.lineup_k_pct_cv AS own_lineup_k_pct_cv,
        lq_own.pct_lhb AS own_lineup_pct_lhb,
        CASE WHEN lu.batting_order BETWEEN 1 AND 9 THEN 1.0 ELSE 0.0 END AS lineup_confirmed_flag,
        team_lu.confirmed_team_lineup_slots::float AS confirmed_team_lineup_slots,
        CASE WHEN COALESCE(team_lu.confirmed_team_lineup_slots, 0) >= 7 THEN 1.0 ELSE 0.0 END AS team_lineup_confirmed_flag,
        CASE WHEN lu.batting_order IS NULL AND bps.batting_order BETWEEN 1 AND 9 THEN 1.0 ELSE 0.0 END AS lineup_boxscore_proxy_flag,
        COALESCE(lu.batting_order, bps.batting_order)::float
            * CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.home_implied_runs ELSE gf.away_implied_runs END
            AS lineup_slot_x_team_implied_runs,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_sp_k_pct_10 ELSE gf.home_sp_k_pct_10 END AS opp_sp_k_pct_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_bb_pct ELSE gf.home_sp_sc_bb_pct END AS opp_sp_bb_pct,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_xwoba ELSE gf.home_sp_sc_xwoba END AS opp_sp_xwoba,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_hard_hit_pct ELSE gf.home_sp_sc_hard_hit_pct END AS opp_sp_hard_hit_pct,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_sp_sc_whiff_pct ELSE gf.home_sp_sc_whiff_pct END AS opp_sp_whiff_pct,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_bp_era_10 ELSE gf.home_bp_era_10 END AS opp_bp_era_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_bp_whip_10 ELSE gf.home_bp_whip_10 END AS opp_bp_whip_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_bp_k9_10 ELSE gf.home_bp_k9_10 END AS opp_bp_k9_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_bullpen_ip_last_3 ELSE gf.home_bullpen_ip_last_3 END AS opp_bp_ip_last_3,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_bullpen_ip_last_7 ELSE gf.home_bullpen_ip_last_7 END AS opp_bp_ip_last_7,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_k_pct_avg_10 ELSE gf.home_k_pct_avg_10 END AS opp_team_k_pct_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_avg_avg_10 ELSE gf.home_avg_avg_10 END AS opp_team_avg_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_obp_avg_10 ELSE gf.home_obp_avg_10 END AS opp_team_obp_10,
        CASE WHEN bps.team_abbr = g.home_team_abbr THEN gf.away_slg_avg_10 ELSE gf.home_slg_avg_10 END AS opp_team_slg_10,
        CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.hits_avg_10_vs_lhp ELSE bvh.hits_avg_10_vs_rhp END AS batter_vs_hand_hits_avg_10,
        CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.tb_avg_10_vs_lhp ELSE bvh.tb_avg_10_vs_rhp END AS batter_vs_hand_tb_avg_10,
        CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.hr_avg_10_vs_lhp ELSE bvh.hr_avg_10_vs_rhp END AS batter_vs_hand_hr_avg_10,
        CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.iso_avg_10_vs_lhp ELSE bvh.iso_avg_10_vs_rhp END AS batter_vs_hand_iso_avg_10,
        CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.k_rate_avg_10_vs_lhp ELSE bvh.k_rate_avg_10_vs_rhp END AS batter_vs_hand_k_rate_10,
        CASE WHEN bvh.opp_sp_hand = 'L' THEN bvh.n_games_vs_lhp_10 ELSE bvh.n_games_vs_rhp_10 END AS batter_vs_hand_games_10,
        bvr.bvr_ba_30 AS batter_vs_rp_ba_30,
        bvr.bvr_slg_30 AS batter_vs_rp_slg_30,
        bvr.bvr_hr_rate_30 AS batter_vs_rp_hr_rate_30,
        bvr.bvr_k_rate_30 AS batter_vs_rp_k_rate_30,
        sc_b.barrel_batted_rate AS batter_sc_barrel_rate,
        sc_b.hard_hit_percent AS batter_sc_hard_hit_pct,
        sc_b.avg_exit_velocity AS batter_sc_avg_exit_velo,
        sc_b.avg_launch_angle AS batter_sc_avg_launch_angle,
        sc_b.sweet_spot_percent AS batter_sc_sweet_spot_pct,
        sc_b.flyballs_percent AS batter_sc_fb_pct,
        sc_b.groundballs_percent AS batter_sc_gb_pct,
        sc_b.linedrives_percent AS batter_sc_ld_pct,
        sc_b.xba AS batter_sc_xba,
        sc_b.xslg AS batter_sc_xslg,
        sc_b.xwoba AS batter_sc_xwoba,
        sc_b.xiso AS batter_sc_xiso,
        sc_b.pull_percent AS batter_sc_pull_pct,
        sc_b.opposite_percent AS batter_sc_opposite_pct,
        sc_b.popup_percent AS batter_sc_popup_pct,
        sc_b.brl_pa AS batter_sc_brl_pa,
        ss.sprint_speed AS batter_sprint_speed,
        pd_b.oz_swing_pct AS batter_disc_oz_swing_pct,
        pd_b.iz_contact_pct AS batter_disc_iz_contact_pct,
        pd_b.oz_contact_pct AS batter_disc_oz_contact_pct,
        pd_b.whiff_pct AS batter_disc_whiff_pct,
        pd_b.out_zone_pct AS batter_disc_out_zone_pct,
        pd_b.k_pct AS batter_disc_k_pct,
        pd_b.bb_pct AS batter_disc_bb_pct,
        sc_opp_p.barrel_batted_rate AS opp_sp_sc_barrel_rate,
        sc_opp_p.hard_hit_percent AS opp_sp_sc_hard_hit_pct,
        sc_opp_p.avg_exit_velocity AS opp_sp_sc_avg_exit_velo,
        sc_opp_p.avg_launch_angle AS opp_sp_sc_avg_launch_angle,
        sc_opp_p.xba AS opp_sp_sc_xba,
        sc_opp_p.xslg AS opp_sp_sc_xslg,
        sc_opp_p.xwoba AS opp_sp_sc_xwoba,
        sc_opp_p.xiso AS opp_sp_sc_xiso,
        pa_opp.fb_percent AS opp_sp_fb_pct,
        pa_opp.fb_hard_hit_pct AS opp_sp_fb_hard_hit_pct,
        pa_opp.fb_xwoba AS opp_sp_fb_xwoba,
        pa_opp.fb_run_value_per_100 AS opp_sp_fb_run_value_per_100,
        pa_opp.fb_whiff_pct AS opp_sp_fb_whiff_pct,
        pa_opp.fb_k_pct AS opp_sp_fb_k_pct,
        pa_opp.si_percent AS opp_sp_si_pct,
        pa_opp.si_hard_hit_pct AS opp_sp_si_hard_hit_pct,
        pa_opp.si_xwoba AS opp_sp_si_xwoba,
        pa_opp.si_whiff_pct AS opp_sp_si_whiff_pct,
        pa_opp.si_k_pct AS opp_sp_si_k_pct,
        pa_opp.sl_percent AS opp_sp_sl_pct,
        pa_opp.sl_hard_hit_pct AS opp_sp_sl_hard_hit_pct,
        pa_opp.sl_xwoba AS opp_sp_sl_xwoba,
        pa_opp.sl_run_value_per_100 AS opp_sp_sl_run_value_per_100,
        pa_opp.sl_whiff_pct AS opp_sp_sl_whiff_pct,
        pa_opp.sl_k_pct AS opp_sp_sl_k_pct,
        pa_opp.ch_percent AS opp_sp_ch_pct,
        pa_opp.ch_hard_hit_pct AS opp_sp_ch_hard_hit_pct,
        pa_opp.ch_xwoba AS opp_sp_ch_xwoba,
        pa_opp.ch_run_value_per_100 AS opp_sp_ch_run_value_per_100,
        pa_opp.ch_whiff_pct AS opp_sp_ch_whiff_pct,
        pa_opp.ch_k_pct AS opp_sp_ch_k_pct,
        pa_opp.fastball_family_pct AS opp_sp_fastball_family_pct,
        pa_opp.pitch_diversity AS opp_sp_pitch_diversity,
        hist_pa.projected_pa,
        hist_pa.pa_games,
        COALESCE(NULLIF(bps.stats #>> '{batting,plateAppearances}', '')::float,
                 GREATEST(COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0), 0)::float) AS actual_pa,
        COALESCE(NULLIF(bps.stats #>> '{batting,atBats}', '')::float, gl.at_bats::float) AS actual_at_bats,
        COALESCE(NULLIF(bps.stats #>> '{batting,hits}', '')::float, gl.hits::float) AS actual_hits,
        COALESCE(NULLIF(bps.stats #>> '{batting,doubles}', '')::float, gl.doubles::float) AS actual_doubles,
        COALESCE(NULLIF(bps.stats #>> '{batting,triples}', '')::float, gl.triples::float) AS actual_triples,
        COALESCE(NULLIF(bps.stats #>> '{batting,homeRuns}', '')::float, gl.home_runs::float) AS actual_home_runs,
        COALESCE(NULLIF(bps.stats #>> '{batting,totalBases}', '')::float, gl.total_bases::float) AS actual_total_bases,
        COALESCE(NULLIF(bps.stats #>> '{batting,baseOnBalls}', '')::float, gl.walks_batter::float) AS actual_walks,
        COALESCE(NULLIF(bps.stats #>> '{batting,strikeOuts}', '')::float, gl.strikeouts_batter::float) AS actual_strikeouts,
        GREATEST(
            COALESCE(NULLIF(bps.stats #>> '{batting,hits}', '')::float, gl.hits::float, 0.0)
          - COALESCE(NULLIF(bps.stats #>> '{batting,doubles}', '')::float, gl.doubles::float, 0.0)
          - COALESCE(NULLIF(bps.stats #>> '{batting,triples}', '')::float, gl.triples::float, 0.0)
          - COALESCE(NULLIF(bps.stats #>> '{batting,homeRuns}', '')::float, gl.home_runs::float, 0.0),
            0.0
        ) AS actual_singles,
        bps.source_fetched_at_utc AS source_updated_at
    FROM raw.mlb_boxscore_player_stats bps
    LEFT JOIN raw.mlb_games g ON g.game_slug = bps.game_slug
    LEFT JOIN raw.mlb_venues v ON v.venue_id = g.venue_id
    LEFT JOIN raw.mlb_weather wx
      ON wx.game_slug = bps.game_slug
     AND wx.fetched_at_utc <= g.start_ts_utc
    LEFT JOIN raw.mlb_player_gamelogs gl
      ON gl.game_slug = bps.game_slug
     AND gl.player_id = bps.player_id
    LEFT JOIN features.mlb_game_training_features gf ON gf.game_slug = bps.game_slug
    LEFT JOIN raw.mlb_player_handedness bh ON bh.player_id = bps.player_id
    LEFT JOIN raw.mlb_player_handedness opp_ph
      ON opp_ph.player_id = CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN g.away_sp_id
            WHEN bps.team_abbr = g.away_team_abbr THEN g.home_sp_id
            ELSE NULL
         END
    LEFT JOIN raw.mlb_lineups lu
      ON lu.game_slug = bps.game_slug
     AND lu.team_abbr = bps.team_abbr
     AND lu.player_name_norm = TRIM(LOWER(REGEXP_REPLACE(
            COALESCE(NULLIF(CONCAT_WS(' ', bps.first_name, bps.last_name), ''), ''),
            '[^a-z0-9]+', ' ', 'g'
         )))
    LEFT JOIN LATERAL (
        SELECT COUNT(*)::int AS confirmed_team_lineup_slots
        FROM raw.mlb_lineups lu_team
        WHERE lu_team.game_slug = bps.game_slug
          AND lu_team.team_abbr = bps.team_abbr
          AND lu_team.batting_order BETWEEN 1 AND 9
    ) team_lu ON TRUE
    LEFT JOIN features.mlb_batting_vs_hand bvh
      ON bvh.game_slug = bps.game_slug
     AND bvh.team_abbr = bps.team_abbr
     AND bvh.player_id = bps.player_id
    LEFT JOIN features.mlb_batter_vs_rp bvr
      ON bvr.game_slug = bps.game_slug
     AND bvr.batter_id = bps.player_id
    LEFT JOIN features.mlb_lineup_quality lq_own
      ON lq_own.game_slug = bps.game_slug
     AND lq_own.team_abbr = bps.team_abbr
    LEFT JOIN features.mlb_park_babip_factor pbf
      ON pbf.venue_id = g.venue_id
    LEFT JOIN LATERAL (
        SELECT
            SUM(barrel_batted_rate * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS barrel_batted_rate,
            SUM(hard_hit_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS hard_hit_percent,
            SUM(avg_exit_velocity * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS avg_exit_velocity,
            SUM(avg_launch_angle * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS avg_launch_angle,
            SUM(sweet_spot_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS sweet_spot_percent,
            SUM(LEAST(flyballs_percent, 100.0) * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS flyballs_percent,
            SUM(LEAST(groundballs_percent, 100.0) * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS groundballs_percent,
            SUM(LEAST(linedrives_percent, 100.0) * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS linedrives_percent,
            SUM(xba * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xba,
            SUM(xslg * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xslg,
            SUM(xwoba * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xwoba,
            SUM(xiso * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xiso,
            SUM(pull_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS pull_percent,
            SUM(opposite_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS opposite_percent,
            SUM(popup_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS popup_percent,
            SUM(brl_pa * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS brl_pa
        FROM raw.mlb_statcast_batting
        WHERE player_id = bps.player_id
          AND (
              season_year < EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
              OR (
                  season_year = EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
                  AND fetched_at_utc::date <= COALESCE(g.game_date_et, gl.game_date_et)
              )
          )
    ) sc_b ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            SUM(barrel_batted_rate * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS barrel_batted_rate,
            SUM(hard_hit_percent * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS hard_hit_percent,
            SUM(avg_exit_velocity * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS avg_exit_velocity,
            SUM(avg_launch_angle * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS avg_launch_angle,
            SUM(xba * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xba,
            SUM(xslg * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xslg,
            SUM(xwoba * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xwoba,
            SUM(xiso * batted_ball_events) / NULLIF(SUM(batted_ball_events), 0) AS xiso
        FROM raw.mlb_statcast_pitching
        WHERE player_id = CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN g.away_sp_id
            WHEN bps.team_abbr = g.away_team_abbr THEN g.home_sp_id
            ELSE NULL
        END
          AND (
              season_year < EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
              OR (
                  season_year = EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
                  AND fetched_at_utc::date <= COALESCE(g.game_date_et, gl.game_date_et)
              )
          )
    ) sc_opp_p ON TRUE
    LEFT JOIN LATERAL (
        SELECT sprint_speed
        FROM raw.mlb_statcast_sprint_speed
        WHERE player_id = bps.player_id
          AND (
              season_year < EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
              OR (
                  season_year = EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
                  AND fetched_at_utc::date <= COALESCE(g.game_date_et, gl.game_date_et)
              )
          )
        ORDER BY season_year DESC
        LIMIT 1
    ) ss ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            oz_swing_pct,
            iz_contact_pct,
            oz_contact_pct,
            whiff_pct,
            out_zone_pct,
            k_pct,
            bb_pct
        FROM raw.mlb_statcast_batter_discipline
        WHERE player_id = bps.player_id
          AND (
              season_year < EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
              OR (
                  season_year = EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
                  AND fetched_at_utc::date <= COALESCE(g.game_date_et, gl.game_date_et)
              )
          )
        ORDER BY season_year DESC
        LIMIT 1
    ) pd_b ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            fb_percent,
            fb_hard_hit_pct,
            fb_xwoba,
            fb_run_value_per_100,
            fb_whiff_pct,
            fb_k_pct,
            si_percent,
            si_hard_hit_pct,
            si_xwoba,
            si_whiff_pct,
            si_k_pct,
            sl_percent,
            sl_hard_hit_pct,
            sl_xwoba,
            sl_run_value_per_100,
            sl_whiff_pct,
            sl_k_pct,
            ch_percent,
            ch_hard_hit_pct,
            ch_xwoba,
            ch_run_value_per_100,
            ch_whiff_pct,
            ch_k_pct,
            fastball_family_pct,
            pitch_diversity
        FROM raw.mlb_statcast_pitcher_arsenal
        WHERE player_id = CASE
            WHEN bps.team_abbr = g.home_team_abbr THEN g.away_sp_id
            WHEN bps.team_abbr = g.away_team_abbr THEN g.home_sp_id
            ELSE NULL
        END
          AND (
              season_year < EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
              OR (
                  season_year = EXTRACT(YEAR FROM COALESCE(g.game_date_et, gl.game_date_et))::int
                  AND fetched_at_utc::date <= COALESCE(g.game_date_et, gl.game_date_et)
              )
          )
        ORDER BY season_year DESC, fetched_at_utc DESC NULLS LAST
        LIMIT 1
    ) pa_opp ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            AVG(pa_est)::float AS projected_pa,
            COUNT(*)::int AS pa_games
        FROM (
            SELECT GREATEST(COALESCE(gl2.at_bats, 0) + COALESCE(gl2.walks_batter, 0), 0) AS pa_est
            FROM raw.mlb_player_gamelogs gl2
            JOIN raw.mlb_games g2 ON g2.game_slug = gl2.game_slug
            WHERE gl2.player_id = bps.player_id
              AND gl2.team_abbr = bps.team_abbr
              AND g2.status = 'final'
              AND g2.game_date_et < COALESCE(g.game_date_et, gl.game_date_et)
              AND (COALESCE(gl2.at_bats, 0) + COALESCE(gl2.walks_batter, 0)) > 0
            ORDER BY g2.game_date_et DESC, gl2.game_slug DESC
            LIMIT 10
        ) recent_pa
    ) hist_pa ON TRUE
    WHERE COALESCE(g.game_date_et, gl.game_date_et) >= %(date_from)s
      AND (%(date_to)s::date IS NULL OR COALESCE(g.game_date_et, gl.game_date_et) <= %(date_to)s::date)
      AND bps.stats ? 'batting'
),
prop_agg AS (
    SELECT
        game_slug,
        player_id,
        AVG(projected_pa)::float AS projected_pa,
        MAX(pa_games)::int AS pa_games,
        AVG(pred_count) FILTER (WHERE market = 'batter_hits')::float AS model_pred_hits,
        AVG(pred_count) FILTER (WHERE market = 'batter_total_bases')::float AS model_pred_total_bases,
        AVG(pred_count) FILTER (WHERE market = 'batter_home_runs')::float AS model_pred_home_runs,
        COUNT(*)::int AS prop_example_rows,
        COUNT(DISTINCT prop_offer_id)::int AS prop_offer_rows,
        AVG(CASE WHEN paired_price IS NOT NULL THEN 1.0 ELSE 0.0 END)::float AS paired_price_rate,
        AVG(CASE WHEN pair_quality = 'same_book' THEN 1.0 ELSE 0.0 END)::float AS same_book_pair_rate,
        AVG(CASE WHEN clv_valid IS TRUE THEN 1.0 ELSE 0.0 END)::float AS valid_clv_rate,
        MIN(source_created_at) AS first_lock_at_utc
    FROM features.mlb_prop_market_training_examples
    WHERE market IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
      AND game_date_et >= %(date_from)s
      AND (%(date_to)s::date IS NULL OR game_date_et <= %(date_to)s::date)
    GROUP BY game_slug, player_id
)
INSERT INTO features.mlb_hitter_player_game_training (
    game_date_et, game_slug, player_id, player_name, player_name_norm,
    team_abbr, opponent_abbr, is_home, lineup_slot, lineup_source,
    confirmed_starter, starter_status_source, primary_position, batter_hand,
    opp_sp_id, opp_sp_hand, opp_sp_hand_l, team_implied_runs, opponent_implied_runs,
    game_total_line, venue_id, park_run_factor, park_hr_factor, park_babip_factor,
    temperature_f, wind_speed_mph, wind_sin, wind_cos, precip_prob_pct,
    is_dome, is_day_game, weather_pregame_flag,
    own_lineup_xwoba_avg, own_lineup_xslg_avg, own_lineup_barrel_avg,
    own_lineup_hard_hit_avg, own_lineup_k_pct_cv, own_lineup_pct_lhb,
    lineup_confirmed_flag, confirmed_team_lineup_slots, team_lineup_confirmed_flag,
    lineup_boxscore_proxy_flag, lineup_slot_x_team_implied_runs,
    opp_sp_k_pct_10, opp_sp_bb_pct, opp_sp_xwoba,
    opp_sp_hard_hit_pct, opp_sp_whiff_pct, opp_bp_era_10, opp_bp_whip_10,
    opp_bp_k9_10, opp_bp_ip_last_3, opp_bp_ip_last_7, opp_team_k_pct_10,
    opp_team_avg_10, opp_team_obp_10, opp_team_slg_10,
    batter_vs_hand_hits_avg_10, batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10, batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10, batter_vs_hand_games_10,
    batter_vs_rp_ba_30, batter_vs_rp_slg_30, batter_vs_rp_hr_rate_30,
    batter_vs_rp_k_rate_30, batter_sc_barrel_rate, batter_sc_hard_hit_pct,
    batter_sc_avg_exit_velo, batter_sc_avg_launch_angle, batter_sc_sweet_spot_pct,
    batter_sc_fb_pct, batter_sc_gb_pct, batter_sc_ld_pct, batter_sc_xba,
    batter_sc_xslg, batter_sc_xwoba, batter_sc_xiso, batter_sc_pull_pct,
    batter_sc_opposite_pct, batter_sc_popup_pct, batter_sc_brl_pa,
    batter_sprint_speed, batter_disc_oz_swing_pct, batter_disc_iz_contact_pct,
    batter_disc_oz_contact_pct, batter_disc_whiff_pct, batter_disc_out_zone_pct,
    batter_disc_k_pct, batter_disc_bb_pct, opp_sp_sc_barrel_rate,
    opp_sp_sc_hard_hit_pct, opp_sp_sc_avg_exit_velo, opp_sp_sc_avg_launch_angle,
    opp_sp_sc_xba, opp_sp_sc_xslg, opp_sp_sc_xwoba, opp_sp_sc_xiso,
    opp_sp_fb_pct, opp_sp_fb_hard_hit_pct, opp_sp_fb_xwoba,
    opp_sp_fb_run_value_per_100, opp_sp_fb_whiff_pct, opp_sp_fb_k_pct,
    opp_sp_si_pct, opp_sp_si_hard_hit_pct, opp_sp_si_xwoba,
    opp_sp_si_whiff_pct, opp_sp_si_k_pct, opp_sp_sl_pct,
    opp_sp_sl_hard_hit_pct, opp_sp_sl_xwoba, opp_sp_sl_run_value_per_100,
    opp_sp_sl_whiff_pct, opp_sp_sl_k_pct, opp_sp_ch_pct,
    opp_sp_ch_hard_hit_pct, opp_sp_ch_xwoba, opp_sp_ch_run_value_per_100,
    opp_sp_ch_whiff_pct, opp_sp_ch_k_pct, opp_sp_fastball_family_pct,
    opp_sp_pitch_diversity,
    projected_pa, pa_games, model_pred_hits,
    model_pred_total_bases, model_pred_home_runs, prop_example_rows,
    prop_offer_rows, paired_price_rate, same_book_pair_rate, valid_clv_rate,
    first_lock_at_utc, actual_pa, actual_at_bats, actual_hits, actual_singles,
    actual_doubles, actual_triples, actual_home_runs, actual_total_bases,
    actual_walks, actual_strikeouts, low_pa_flag, source_updated_at, row_updated_at
)
SELECT
    b.game_date_et, b.game_slug, b.player_id, b.player_name, b.player_name_norm,
    b.team_abbr, b.opponent_abbr, b.is_home, b.lineup_slot, b.lineup_source,
    b.confirmed_starter, b.starter_status_source, b.primary_position, b.batter_hand,
    b.opp_sp_id, b.opp_sp_hand, b.opp_sp_hand_l, b.team_implied_runs, b.opponent_implied_runs,
    b.game_total_line, b.venue_id, b.park_run_factor, b.park_hr_factor, b.park_babip_factor,
    b.temperature_f, b.wind_speed_mph, b.wind_sin, b.wind_cos, b.precip_prob_pct,
    b.is_dome, b.is_day_game, b.weather_pregame_flag,
    b.own_lineup_xwoba_avg, b.own_lineup_xslg_avg, b.own_lineup_barrel_avg,
    b.own_lineup_hard_hit_avg, b.own_lineup_k_pct_cv, b.own_lineup_pct_lhb,
    b.lineup_confirmed_flag, b.confirmed_team_lineup_slots, b.team_lineup_confirmed_flag,
    b.lineup_boxscore_proxy_flag, b.lineup_slot_x_team_implied_runs,
    b.opp_sp_k_pct_10, b.opp_sp_bb_pct, b.opp_sp_xwoba,
    b.opp_sp_hard_hit_pct, b.opp_sp_whiff_pct, b.opp_bp_era_10, b.opp_bp_whip_10,
    b.opp_bp_k9_10, b.opp_bp_ip_last_3, b.opp_bp_ip_last_7, b.opp_team_k_pct_10,
    b.opp_team_avg_10, b.opp_team_obp_10, b.opp_team_slg_10,
    b.batter_vs_hand_hits_avg_10, b.batter_vs_hand_tb_avg_10,
    b.batter_vs_hand_hr_avg_10, b.batter_vs_hand_iso_avg_10,
    b.batter_vs_hand_k_rate_10, b.batter_vs_hand_games_10,
    b.batter_vs_rp_ba_30, b.batter_vs_rp_slg_30, b.batter_vs_rp_hr_rate_30,
    b.batter_vs_rp_k_rate_30, b.batter_sc_barrel_rate, b.batter_sc_hard_hit_pct,
    b.batter_sc_avg_exit_velo, b.batter_sc_avg_launch_angle, b.batter_sc_sweet_spot_pct,
    b.batter_sc_fb_pct, b.batter_sc_gb_pct, b.batter_sc_ld_pct, b.batter_sc_xba,
    b.batter_sc_xslg, b.batter_sc_xwoba, b.batter_sc_xiso, b.batter_sc_pull_pct,
    b.batter_sc_opposite_pct, b.batter_sc_popup_pct, b.batter_sc_brl_pa,
    b.batter_sprint_speed, b.batter_disc_oz_swing_pct, b.batter_disc_iz_contact_pct,
    b.batter_disc_oz_contact_pct, b.batter_disc_whiff_pct, b.batter_disc_out_zone_pct,
    b.batter_disc_k_pct, b.batter_disc_bb_pct, b.opp_sp_sc_barrel_rate,
    b.opp_sp_sc_hard_hit_pct, b.opp_sp_sc_avg_exit_velo, b.opp_sp_sc_avg_launch_angle,
    b.opp_sp_sc_xba, b.opp_sp_sc_xslg, b.opp_sp_sc_xwoba, b.opp_sp_sc_xiso,
    b.opp_sp_fb_pct, b.opp_sp_fb_hard_hit_pct, b.opp_sp_fb_xwoba,
    b.opp_sp_fb_run_value_per_100, b.opp_sp_fb_whiff_pct, b.opp_sp_fb_k_pct,
    b.opp_sp_si_pct, b.opp_sp_si_hard_hit_pct, b.opp_sp_si_xwoba,
    b.opp_sp_si_whiff_pct, b.opp_sp_si_k_pct, b.opp_sp_sl_pct,
    b.opp_sp_sl_hard_hit_pct, b.opp_sp_sl_xwoba, b.opp_sp_sl_run_value_per_100,
    b.opp_sp_sl_whiff_pct, b.opp_sp_sl_k_pct, b.opp_sp_ch_pct,
    b.opp_sp_ch_hard_hit_pct, b.opp_sp_ch_xwoba, b.opp_sp_ch_run_value_per_100,
    b.opp_sp_ch_whiff_pct, b.opp_sp_ch_k_pct, b.opp_sp_fastball_family_pct,
    b.opp_sp_pitch_diversity,
    COALESCE(p.projected_pa, b.projected_pa), COALESCE(p.pa_games, b.pa_games), p.model_pred_hits,
    p.model_pred_total_bases, p.model_pred_home_runs, COALESCE(p.prop_example_rows, 0),
    COALESCE(p.prop_offer_rows, 0), p.paired_price_rate, p.same_book_pair_rate, p.valid_clv_rate,
    p.first_lock_at_utc, b.actual_pa, b.actual_at_bats, b.actual_hits, b.actual_singles,
    b.actual_doubles, b.actual_triples, b.actual_home_runs, b.actual_total_bases,
    b.actual_walks, b.actual_strikeouts,
    CASE WHEN b.actual_pa <= 2 THEN 1.0 ELSE 0.0 END AS low_pa_flag,
    b.source_updated_at, now()
FROM base b
LEFT JOIN prop_agg p
  ON p.game_slug = b.game_slug
 AND p.player_id = b.player_id
WHERE b.game_date_et IS NOT NULL
  AND b.actual_pa IS NOT NULL
ON CONFLICT (game_slug, player_id) DO UPDATE SET
    game_date_et = EXCLUDED.game_date_et,
    player_name = EXCLUDED.player_name,
    player_name_norm = EXCLUDED.player_name_norm,
    team_abbr = EXCLUDED.team_abbr,
    opponent_abbr = EXCLUDED.opponent_abbr,
    is_home = EXCLUDED.is_home,
    lineup_slot = EXCLUDED.lineup_slot,
    lineup_source = EXCLUDED.lineup_source,
    confirmed_starter = EXCLUDED.confirmed_starter,
    starter_status_source = EXCLUDED.starter_status_source,
    primary_position = EXCLUDED.primary_position,
    batter_hand = EXCLUDED.batter_hand,
    opp_sp_id = EXCLUDED.opp_sp_id,
    opp_sp_hand = EXCLUDED.opp_sp_hand,
    opp_sp_hand_l = EXCLUDED.opp_sp_hand_l,
    team_implied_runs = EXCLUDED.team_implied_runs,
    opponent_implied_runs = EXCLUDED.opponent_implied_runs,
    game_total_line = EXCLUDED.game_total_line,
    venue_id = EXCLUDED.venue_id,
    park_run_factor = EXCLUDED.park_run_factor,
    park_hr_factor = EXCLUDED.park_hr_factor,
    park_babip_factor = EXCLUDED.park_babip_factor,
    temperature_f = EXCLUDED.temperature_f,
    wind_speed_mph = EXCLUDED.wind_speed_mph,
    wind_sin = EXCLUDED.wind_sin,
    wind_cos = EXCLUDED.wind_cos,
    precip_prob_pct = EXCLUDED.precip_prob_pct,
    is_dome = EXCLUDED.is_dome,
    is_day_game = EXCLUDED.is_day_game,
    weather_pregame_flag = EXCLUDED.weather_pregame_flag,
    own_lineup_xwoba_avg = EXCLUDED.own_lineup_xwoba_avg,
    own_lineup_xslg_avg = EXCLUDED.own_lineup_xslg_avg,
    own_lineup_barrel_avg = EXCLUDED.own_lineup_barrel_avg,
    own_lineup_hard_hit_avg = EXCLUDED.own_lineup_hard_hit_avg,
    own_lineup_k_pct_cv = EXCLUDED.own_lineup_k_pct_cv,
    own_lineup_pct_lhb = EXCLUDED.own_lineup_pct_lhb,
    lineup_confirmed_flag = EXCLUDED.lineup_confirmed_flag,
    confirmed_team_lineup_slots = EXCLUDED.confirmed_team_lineup_slots,
    team_lineup_confirmed_flag = EXCLUDED.team_lineup_confirmed_flag,
    lineup_boxscore_proxy_flag = EXCLUDED.lineup_boxscore_proxy_flag,
    lineup_slot_x_team_implied_runs = EXCLUDED.lineup_slot_x_team_implied_runs,
    opp_sp_k_pct_10 = EXCLUDED.opp_sp_k_pct_10,
    opp_sp_bb_pct = EXCLUDED.opp_sp_bb_pct,
    opp_sp_xwoba = EXCLUDED.opp_sp_xwoba,
    opp_sp_hard_hit_pct = EXCLUDED.opp_sp_hard_hit_pct,
    opp_sp_whiff_pct = EXCLUDED.opp_sp_whiff_pct,
    opp_bp_era_10 = EXCLUDED.opp_bp_era_10,
    opp_bp_whip_10 = EXCLUDED.opp_bp_whip_10,
    opp_bp_k9_10 = EXCLUDED.opp_bp_k9_10,
    opp_bp_ip_last_3 = EXCLUDED.opp_bp_ip_last_3,
    opp_bp_ip_last_7 = EXCLUDED.opp_bp_ip_last_7,
    opp_team_k_pct_10 = EXCLUDED.opp_team_k_pct_10,
    opp_team_avg_10 = EXCLUDED.opp_team_avg_10,
    opp_team_obp_10 = EXCLUDED.opp_team_obp_10,
    opp_team_slg_10 = EXCLUDED.opp_team_slg_10,
    batter_vs_hand_hits_avg_10 = EXCLUDED.batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10 = EXCLUDED.batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10 = EXCLUDED.batter_vs_hand_hr_avg_10,
    batter_vs_hand_iso_avg_10 = EXCLUDED.batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10 = EXCLUDED.batter_vs_hand_k_rate_10,
    batter_vs_hand_games_10 = EXCLUDED.batter_vs_hand_games_10,
    batter_vs_rp_ba_30 = EXCLUDED.batter_vs_rp_ba_30,
    batter_vs_rp_slg_30 = EXCLUDED.batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30 = EXCLUDED.batter_vs_rp_hr_rate_30,
    batter_vs_rp_k_rate_30 = EXCLUDED.batter_vs_rp_k_rate_30,
    batter_sc_barrel_rate = EXCLUDED.batter_sc_barrel_rate,
    batter_sc_hard_hit_pct = EXCLUDED.batter_sc_hard_hit_pct,
    batter_sc_avg_exit_velo = EXCLUDED.batter_sc_avg_exit_velo,
    batter_sc_avg_launch_angle = EXCLUDED.batter_sc_avg_launch_angle,
    batter_sc_sweet_spot_pct = EXCLUDED.batter_sc_sweet_spot_pct,
    batter_sc_fb_pct = EXCLUDED.batter_sc_fb_pct,
    batter_sc_gb_pct = EXCLUDED.batter_sc_gb_pct,
    batter_sc_ld_pct = EXCLUDED.batter_sc_ld_pct,
    batter_sc_xba = EXCLUDED.batter_sc_xba,
    batter_sc_xslg = EXCLUDED.batter_sc_xslg,
    batter_sc_xwoba = EXCLUDED.batter_sc_xwoba,
    batter_sc_xiso = EXCLUDED.batter_sc_xiso,
    batter_sc_pull_pct = EXCLUDED.batter_sc_pull_pct,
    batter_sc_opposite_pct = EXCLUDED.batter_sc_opposite_pct,
    batter_sc_popup_pct = EXCLUDED.batter_sc_popup_pct,
    batter_sc_brl_pa = EXCLUDED.batter_sc_brl_pa,
    batter_sprint_speed = EXCLUDED.batter_sprint_speed,
    batter_disc_oz_swing_pct = EXCLUDED.batter_disc_oz_swing_pct,
    batter_disc_iz_contact_pct = EXCLUDED.batter_disc_iz_contact_pct,
    batter_disc_oz_contact_pct = EXCLUDED.batter_disc_oz_contact_pct,
    batter_disc_whiff_pct = EXCLUDED.batter_disc_whiff_pct,
    batter_disc_out_zone_pct = EXCLUDED.batter_disc_out_zone_pct,
    batter_disc_k_pct = EXCLUDED.batter_disc_k_pct,
    batter_disc_bb_pct = EXCLUDED.batter_disc_bb_pct,
    opp_sp_sc_barrel_rate = EXCLUDED.opp_sp_sc_barrel_rate,
    opp_sp_sc_hard_hit_pct = EXCLUDED.opp_sp_sc_hard_hit_pct,
    opp_sp_sc_avg_exit_velo = EXCLUDED.opp_sp_sc_avg_exit_velo,
    opp_sp_sc_avg_launch_angle = EXCLUDED.opp_sp_sc_avg_launch_angle,
    opp_sp_sc_xba = EXCLUDED.opp_sp_sc_xba,
    opp_sp_sc_xslg = EXCLUDED.opp_sp_sc_xslg,
    opp_sp_sc_xwoba = EXCLUDED.opp_sp_sc_xwoba,
    opp_sp_sc_xiso = EXCLUDED.opp_sp_sc_xiso,
    opp_sp_fb_pct = EXCLUDED.opp_sp_fb_pct,
    opp_sp_fb_hard_hit_pct = EXCLUDED.opp_sp_fb_hard_hit_pct,
    opp_sp_fb_xwoba = EXCLUDED.opp_sp_fb_xwoba,
    opp_sp_fb_run_value_per_100 = EXCLUDED.opp_sp_fb_run_value_per_100,
    opp_sp_fb_whiff_pct = EXCLUDED.opp_sp_fb_whiff_pct,
    opp_sp_fb_k_pct = EXCLUDED.opp_sp_fb_k_pct,
    opp_sp_si_pct = EXCLUDED.opp_sp_si_pct,
    opp_sp_si_hard_hit_pct = EXCLUDED.opp_sp_si_hard_hit_pct,
    opp_sp_si_xwoba = EXCLUDED.opp_sp_si_xwoba,
    opp_sp_si_whiff_pct = EXCLUDED.opp_sp_si_whiff_pct,
    opp_sp_si_k_pct = EXCLUDED.opp_sp_si_k_pct,
    opp_sp_sl_pct = EXCLUDED.opp_sp_sl_pct,
    opp_sp_sl_hard_hit_pct = EXCLUDED.opp_sp_sl_hard_hit_pct,
    opp_sp_sl_xwoba = EXCLUDED.opp_sp_sl_xwoba,
    opp_sp_sl_run_value_per_100 = EXCLUDED.opp_sp_sl_run_value_per_100,
    opp_sp_sl_whiff_pct = EXCLUDED.opp_sp_sl_whiff_pct,
    opp_sp_sl_k_pct = EXCLUDED.opp_sp_sl_k_pct,
    opp_sp_ch_pct = EXCLUDED.opp_sp_ch_pct,
    opp_sp_ch_hard_hit_pct = EXCLUDED.opp_sp_ch_hard_hit_pct,
    opp_sp_ch_xwoba = EXCLUDED.opp_sp_ch_xwoba,
    opp_sp_ch_run_value_per_100 = EXCLUDED.opp_sp_ch_run_value_per_100,
    opp_sp_ch_whiff_pct = EXCLUDED.opp_sp_ch_whiff_pct,
    opp_sp_ch_k_pct = EXCLUDED.opp_sp_ch_k_pct,
    opp_sp_fastball_family_pct = EXCLUDED.opp_sp_fastball_family_pct,
    opp_sp_pitch_diversity = EXCLUDED.opp_sp_pitch_diversity,
    projected_pa = EXCLUDED.projected_pa,
    pa_games = EXCLUDED.pa_games,
    model_pred_hits = EXCLUDED.model_pred_hits,
    model_pred_total_bases = EXCLUDED.model_pred_total_bases,
    model_pred_home_runs = EXCLUDED.model_pred_home_runs,
    prop_example_rows = EXCLUDED.prop_example_rows,
    prop_offer_rows = EXCLUDED.prop_offer_rows,
    paired_price_rate = EXCLUDED.paired_price_rate,
    same_book_pair_rate = EXCLUDED.same_book_pair_rate,
    valid_clv_rate = EXCLUDED.valid_clv_rate,
    first_lock_at_utc = EXCLUDED.first_lock_at_utc,
    actual_pa = EXCLUDED.actual_pa,
    actual_at_bats = EXCLUDED.actual_at_bats,
    actual_hits = EXCLUDED.actual_hits,
    actual_singles = EXCLUDED.actual_singles,
    actual_doubles = EXCLUDED.actual_doubles,
    actual_triples = EXCLUDED.actual_triples,
    actual_home_runs = EXCLUDED.actual_home_runs,
    actual_total_bases = EXCLUDED.actual_total_bases,
    actual_walks = EXCLUDED.actual_walks,
    actual_strikeouts = EXCLUDED.actual_strikeouts,
    low_pa_flag = EXCLUDED.low_pa_flag,
    source_updated_at = EXCLUDED.source_updated_at,
    row_updated_at = now();
"""


SUMMARY_SQL = """
SELECT
    COUNT(*) AS rows,
    COUNT(projected_pa) AS rows_with_projected_pa,
    COUNT(lineup_slot) AS rows_with_lineup_slot,
    COUNT(CASE WHEN confirmed_starter IS TRUE THEN 1 END) AS confirmed_starters,
    COUNT(CASE WHEN prop_example_rows > 0 THEN 1 END) AS rows_with_prop_examples,
    AVG(ABS(actual_pa - projected_pa)) FILTER (WHERE projected_pa IS NOT NULL) AS projected_pa_mae,
    AVG(actual_pa - projected_pa) FILTER (WHERE projected_pa IS NOT NULL) AS projected_pa_bias,
    AVG(actual_hits) AS avg_hits,
    AVG(actual_total_bases) AS avg_total_bases,
    AVG(actual_home_runs) AS avg_home_runs
FROM features.mlb_hitter_player_game_training
WHERE game_date_et >= %(date_from)s
  AND (%(date_to)s::date IS NULL OR game_date_et <= %(date_to)s::date)
"""


SLOT_SQL = """
WITH grouped AS (
    SELECT
        COALESCE(lineup_slot::int::text, 'missing') AS lineup_slot,
        CASE
            WHEN COALESCE(lineup_slot::int::text, 'missing') = 'missing' THEN 99
            ELSE COALESCE(lineup_slot::int::text, '99')::int
        END AS slot_sort,
        COUNT(*) AS rows,
        AVG(actual_pa) AS avg_actual_pa,
        AVG(projected_pa) AS avg_projected_pa,
        AVG(actual_pa - projected_pa) FILTER (WHERE projected_pa IS NOT NULL) AS pa_bias,
        AVG(actual_hits) AS avg_hits,
        AVG(actual_total_bases) AS avg_total_bases,
        AVG(actual_home_runs) AS avg_home_runs
    FROM features.mlb_hitter_player_game_training
    WHERE game_date_et >= %(date_from)s
      AND (%(date_to)s::date IS NULL OR game_date_et <= %(date_to)s::date)
    GROUP BY
        COALESCE(lineup_slot::int::text, 'missing'),
        CASE
            WHEN COALESCE(lineup_slot::int::text, 'missing') = 'missing' THEN 99
            ELSE COALESCE(lineup_slot::int::text, '99')::int
        END
)
SELECT lineup_slot, rows, avg_actual_pa, avg_projected_pa, pa_bias, avg_hits, avg_total_bases, avg_home_runs
FROM grouped
ORDER BY slot_sort
"""


def _date_from(cfg: HitterPlayerGameTrainingConfig) -> date:
    return cfg.date_from or (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days))


def _query_one(conn, sql: str, params: dict[str, Any]) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
    return dict(zip(columns, row)) if row else {}


def _query_many(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in rows]


def _write_report(payload: dict[str, Any], report_file: str | None) -> None:
    name = report_file or "mlb_hitter_player_game_training_latest.md"
    path = _REPORT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload.get("summary", {})
    slots = payload.get("lineup_slots", [])

    def fmt_num(value: Any, digits: int = 2) -> str:
        try:
            if value is None:
                return "-"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    def fmt_pct(num: Any, den: Any) -> str:
        try:
            den_f = float(den or 0)
            if den_f <= 0:
                return "-"
            return f"{100.0 * float(num or 0) / den_f:.1f}%"
        except Exception:
            return "-"

    lines = [
        "# MLB Hitter Player-Game Training Table",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Date range: {payload.get('date_from')} to {payload.get('date_to') or 'latest'}",
        "",
        "## Coverage",
        "",
        f"- Rows: {summary.get('rows', 0)}",
        f"- Lineup slot coverage: {fmt_pct(summary.get('rows_with_lineup_slot'), summary.get('rows'))}",
        f"- Projected PA coverage: {fmt_pct(summary.get('rows_with_projected_pa'), summary.get('rows'))}",
        f"- Prop-example coverage: {fmt_pct(summary.get('rows_with_prop_examples'), summary.get('rows'))}",
        f"- Projected PA MAE: {fmt_num(summary.get('projected_pa_mae'))}",
        f"- Projected PA bias: {fmt_num(summary.get('projected_pa_bias'), 2)}",
        "",
        "## By Lineup Slot",
        "",
        "| Slot | Rows | Actual PA | Projected PA | PA Bias | Hits | TB | HR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in slots:
        lines.append(
            f"| {rec.get('lineup_slot')} | {rec.get('rows', 0)} | "
            f"{fmt_num(rec.get('avg_actual_pa'))} | {fmt_num(rec.get('avg_projected_pa'))} | "
            f"{fmt_num(rec.get('pa_bias'))} | {fmt_num(rec.get('avg_hits'))} | "
            f"{fmt_num(rec.get('avg_total_bases'))} | {fmt_num(rec.get('avg_home_runs'), 3)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_hitter_player_game_training(cfg: HitterPlayerGameTrainingConfig) -> dict[str, Any]:
    date_from = _date_from(cfg)
    params = {"date_from": date_from, "date_to": cfg.date_to}
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_game_training_features(conn)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '2s'")
            cur.execute("SET LOCAL statement_timeout = '8min'")
            cur.execute(DDL)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM features.mlb_hitter_player_game_training
                WHERE game_date_et >= %(date_from)s
                  AND (%(date_to)s::date IS NULL OR game_date_et <= %(date_to)s::date)
                """,
                params,
            )
            deleted = cur.rowcount
            cur.execute(BUILD_SQL, params)
            inserted = cur.rowcount
        conn.commit()
        summary = _query_one(conn, SUMMARY_SQL, params)
        slots = _query_many(conn, SLOT_SQL, params)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_from": str(date_from),
        "date_to": str(cfg.date_to) if cfg.date_to else None,
        "deleted": deleted,
        "inserted": inserted,
        "summary": summary,
        "lineup_slots": slots,
    }
    _write_report(payload, cfg.report_file)
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        import decimal

        if isinstance(value, decimal.Decimal):
            return float(value)
    except Exception:
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hitter player-game training table.")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    cfg = HitterPlayerGameTrainingConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        date_from=date.fromisoformat(args.date_from) if args.date_from else None,
        date_to=date.fromisoformat(args.date_to) if args.date_to else None,
        report_file=args.report_file,
    )
    print(json.dumps(refresh_hitter_player_game_training(cfg), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
