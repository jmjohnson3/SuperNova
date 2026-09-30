"""Train NFL player opportunity models on player-game rows."""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sqlalchemy import create_engine, text

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.modeling.train_player_stat_models import (
    _baseline_suite,
    _best_baseline,
    _fit_model,
    _make_features,
    _metric_payload,
    _temporal_split,
)

log = logging.getLogger("nfl_pipeline.modeling.train_player_opportunity_models")

ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_MD_REPORT = ROOT / "reports" / "nfl_player_opportunity_models_latest.md"


@dataclass(frozen=True)
class OpportunitySpec:
    name: str
    target: str
    label: str
    positions: tuple[str, ...]
    baseline_col: str
    count_like: bool = True


OPPORTUNITY_SPECS: tuple[OpportunitySpec, ...] = (
    OpportunitySpec("qb_pass_attempts", "pass_attempts", "QB pass attempts/dropbacks proxy", ("QB",), "pass_attempts_avg_5"),
    OpportunitySpec("qb_rush_carries", "carries", "QB designed/scramble carry opportunity", ("QB",), "carries_avg_5"),
    OpportunitySpec("rb_carries", "carries", "RB rushing opportunity", ("RB",), "carries_avg_5"),
    OpportunitySpec("rb_targets", "targets", "RB receiving opportunity", ("RB",), "targets_avg_5"),
    OpportunitySpec("receiver_targets", "targets", "WR/TE target opportunity", ("WR", "TE"), "targets_avg_5"),
    OpportunitySpec("receiver_receptions", "receptions", "WR/TE reception opportunity", ("WR", "TE"), "receptions_avg_5"),
    OpportunitySpec("qb_passing_td_role", "passing_tds", "QB passing TD role", ("QB",), "passing_tds_avg_5"),
    OpportunitySpec("skill_td_role", "skill_tds", "RB/WR/TE touchdown role", ("RB", "WR", "TE"), "skill_tds_avg_5"),
    OpportunitySpec("route_participation", "route_participation", "Route participation", ("RB", "WR", "TE"), "route_participation_avg_5", False),
    OpportunitySpec("pass_route_opportunity_share", "pass_route_opportunity_share", "Pass-route opportunity share", ("RB", "WR", "TE"), "pass_route_opportunity_share_avg_5", False),
    OpportunitySpec("receiver_target_share", "target_share", "RB/WR/TE target-share opportunity", ("RB", "WR", "TE"), "targets_share_avg_5", False),
    OpportunitySpec("receiver_air_yards_share", "air_yards_share", "RB/WR/TE air-yards share opportunity", ("RB", "WR", "TE"), "air_yards_share_avg_5", False),
    OpportunitySpec("receiver_air_yards", "receiving_air_yards", "RB/WR/TE air-yards volume opportunity", ("RB", "WR", "TE"), "receiving_air_yards_avg_5", False),
    OpportunitySpec("snap_share", "snap_share", "Offensive snap share", ("QB", "RB", "WR", "TE"), "snap_share_avg_5", False),
    OpportunitySpec("full_workload_probability", "actual_full_workload", "Full-workload probability", ("QB", "RB", "WR", "TE"), "full_workload_score", False),
    OpportunitySpec("limited_usage_risk", "actual_limited_usage", "Limited/rest/weird-usage risk", ("QB", "RB", "WR", "TE"), "limited_workload_risk_score", False),
    OpportunitySpec("high_pass_attempt_probability", "actual_high_pass_attempts", "Spike pass-attempt probability", ("QB",), "high_pass_attempt_score", False),
    OpportunitySpec("high_carry_probability", "actual_high_carries", "Spike carry-volume probability", ("RB", "QB"), "high_carry_score", False),
    OpportunitySpec("high_target_probability", "actual_high_targets", "Spike target-volume probability", ("RB", "WR", "TE"), "high_target_score", False),
    OpportunitySpec("receiver_spike_volume_probability", "actual_receiver_spike_volume", "RB/WR/TE route-driven spike target-volume probability", ("RB", "WR", "TE"), "receiver_spike_volume_score", False),
    OpportunitySpec("receiver_target_route_spike_probability", "actual_receiver_target_route_spike", "RB/WR/TE true target-route spike probability", ("RB", "WR", "TE"), "receiver_target_route_spike_score", False),
    OpportunitySpec("receiver_target_spike_v2_probability", "actual_receiver_target_route_spike", "RB/WR/TE target/air-yard spike probability v2", ("RB", "WR", "TE"), "receiver_target_spike_v2_score", False),
    OpportunitySpec("receiver_air_yards_spike_probability", "actual_receiver_air_yards_spike", "RB/WR/TE air-yards spike probability", ("RB", "WR", "TE"), "receiver_air_yards_spike_v2_score", False),
    OpportunitySpec("receiver_spike_under_correction_v4_probability", "actual_receiver_target_route_spike", "RB/WR/TE spike-underprojection probability v4", ("RB", "WR", "TE"), "receiver_spike_under_correction_v4_score", False),
    OpportunitySpec("rb_carry_spike_v2_probability", "actual_high_carries", "RB carry spike probability v2", ("RB",), "rb_carry_spike_v2_score", False),
    OpportunitySpec("rb_carry_under_correction_v4_probability", "actual_high_carries", "RB carry underprojection probability v4", ("RB",), "rb_carry_under_correction_v4_score", False),
    OpportunitySpec("spike_snap_share_probability", "actual_spike_snap_share", "Spike snap-share probability", ("QB", "RB", "WR", "TE"), "spike_snap_share_score", False),
)

UPSIDE_SIGNAL_MODELS = {
    "high_pass_attempt_probability",
    "high_carry_probability",
    "high_target_probability",
    "receiver_spike_volume_probability",
    "receiver_target_route_spike_probability",
    "receiver_target_spike_v2_probability",
    "receiver_air_yards_spike_probability",
    "receiver_spike_under_correction_v4_probability",
    "rb_carry_spike_v2_probability",
    "rb_carry_under_correction_v4_probability",
    "spike_snap_share_probability",
}

OPPORTUNITY_FEATURE_POLICIES: dict[str, tuple[str, ...]] = {
    "qb_pass_attempts": (
        "pass_attempts", "passing_yards", "high_pass_attempt", "pass_spike_path",
        "qb_volume_spike_signal", "game_script_pass", "team_spread", "team_implied",
        "game_total", "opp_allowed_passing", "starter_confidence", "projected_starter",
        "limited_workload", "full_workload", "injury_", "depth_",
    ),
    "qb_rush_carries": (
        "carries", "rushing_yards", "high_carry", "carry_spike_path", "game_script_rush",
        "starter_confidence", "projected_starter", "limited_workload", "full_workload",
        "injury_", "depth_",
    ),
    "rb_carries": (
        "carries", "rushing_yards", "carry", "rb_", "game_script_rush", "team_spread",
        "team_implied", "snap_share", "offense_snap", "starter_confidence",
        "projected_starter", "limited_workload", "full_workload", "rest_risk",
        "weird_usage", "depth_", "injury_", "role_change_upside",
    ),
    "rb_targets": (
        "target", "route", "receiving_yards", "pass_route", "snap_share",
        "game_script_pass", "team_implied", "starter_confidence", "limited_workload",
        "full_workload", "depth_", "injury_", "role_change_upside",
    ),
    "receiver_targets": (
        "target", "route", "air_yards", "wopr", "first_read", "receiving_yards",
        "receiver_", "game_script_pass", "team_implied", "snap_share",
        "starter_confidence", "limited_workload", "full_workload", "depth_",
        "injury_", "same_week_", "teammate_", "vacancy", "role_change_upside",
    ),
    "receiver_receptions": (
        "target", "reception", "route", "air_yards", "wopr", "first_read",
        "receiver_", "game_script_pass", "team_implied", "starter_confidence",
        "limited_workload", "full_workload", "depth_", "injury_",
    ),
    "qb_passing_td_role": (
        "passing_tds", "pass_attempts", "red_zone_pass", "team_implied",
        "game_total", "opp_allowed_passing_tds", "starter_confidence",
        "projected_starter", "injury_", "depth_",
    ),
    "skill_td_role": (
        "td_", "_tds", "red_zone", "goal_line", "end_zone", "team_implied",
        "first_read", "route", "target", "carry", "starter_confidence",
        "limited_workload", "full_workload", "depth_", "injury_",
    ),
    "route_participation": (
        "route", "pass_route", "snap_share", "target", "receiver_", "depth_",
        "starter_confidence", "limited_workload", "full_workload", "injury_",
    ),
    "pass_route_opportunity_share": (
        "route", "pass_route", "snap_share", "target", "receiver_", "depth_",
        "starter_confidence", "limited_workload", "full_workload", "injury_",
    ),
    "receiver_target_share": (
        "target_share", "targets_share", "target", "route", "pass_route",
        "receiver_target", "receiver_usage", "receiver_", "wopr", "first_read",
        "game_script_pass", "team_implied", "starter_confidence",
        "limited_workload", "full_workload", "workload_floor", "depth_",
        "injury_", "same_week_", "teammate_", "vacancy", "role_change_upside",
    ),
    "receiver_air_yards_share": (
        "air_yards_share", "receiving_air_yards", "air_yards", "wopr",
        "first_read", "target_share", "targets_share", "route", "pass_route",
        "receiver_air", "receiver_explosive", "receiver_contextual",
        "game_script_pass", "team_implied", "starter_confidence",
        "limited_workload", "full_workload", "workload_floor", "depth_",
        "injury_", "same_week_", "teammate_", "vacancy", "role_change_upside",
    ),
    "receiver_air_yards": (
        "receiving_air_yards", "air_yards_share", "air_yards", "wopr",
        "first_read", "target_share", "targets_share", "target", "route",
        "pass_route", "estimated_routes", "receiver_air", "receiver_yards_rate",
        "receiver_explosive", "route_env_receiving_yards", "game_script_pass",
        "team_implied", "starter_confidence", "limited_workload",
        "full_workload", "workload_floor", "depth_", "injury_",
        "same_week_", "teammate_", "vacancy", "role_change_upside",
    ),
    "snap_share": (
        "snap", "offense_snap", "starter_confidence", "projected_starter", "depth_",
        "injury_", "limited_workload", "full_workload", "rest_risk", "weird_usage",
        "role_continuity",
    ),
    "full_workload_probability": (
        "snap", "offense_snap", "starter_confidence", "projected_starter", "depth_",
        "injury_", "limited_workload", "rest_risk", "weird_usage", "role_continuity",
        "normal_usage", "workload_floor",
    ),
    "limited_usage_risk": (
        "snap", "offense_snap", "starter_confidence", "depth_", "injury_",
        "practice_status", "rest_risk", "week_18", "late_season", "weird_usage",
        "usage_volatility", "backup_role", "depth_movement",
    ),
    "high_pass_attempt_probability": (
        "pass_attempts", "high_pass_attempt", "pass_spike_path", "qb_volume_spike_signal",
        "game_script_pass", "team_spread", "game_total", "team_implied",
        "starter_confidence", "full_workload", "limited_workload",
    ),
    "high_carry_probability": (
        "carries", "high_carry", "carry_spike_path", "rb_usage_spike_signal",
        "recent_carry_spike", "game_script_rush", "team_spread", "team_implied",
        "snap_share", "starter_confidence", "full_workload", "limited_workload",
        "role_change_upside", "rb_live_carry_v3", "rb_projected_carries_v3",
        "rb_rush_yards_anchor_v3", "same_week_usage_confidence_v3",
        "yardage_projection_volatility_v3", "depth_", "injury_",
    ),
    "high_target_probability": (
        "target", "high_target", "target_spike_path", "receiver_usage_spike_signal",
        "recent_target_spike", "recent_route_spike", "route", "air_yards", "wopr",
        "first_read", "game_script_pass", "team_implied", "snap_share",
        "starter_confidence", "full_workload", "limited_workload", "role_change_upside",
        "receiver_live_spike_v3", "receiver_projected_targets_v3",
        "receiver_spike_yards_anchor_v3", "same_week_usage_confidence_v3",
        "yardage_projection_volatility_v3", "depth_", "injury_",
    ),
    "receiver_spike_volume_probability": (
        "receiver_spike_volume", "receiver_usage_spike_signal", "target_spike_path",
        "spike_target", "high_target", "recent_target_spike", "recent_route_spike",
        "route", "pass_route", "target_share", "targets_share", "target",
        "air_yards", "wopr", "first_read", "depth_rank_better", "depth_rank_worse",
        "depth_movement", "starter_confidence", "projected_starter",
        "starter_role_stability", "role_continuity", "role_change_upside",
        "workload_floor", "full_workload", "limited_workload", "injury_",
        "practice_status", "same_week_", "teammate_", "vacancy",
        "game_script_pass", "team_implied", "snap_share",
        "offense_snap",
    ),
    "receiver_target_route_spike_probability": (
        "receiver_target_route_spike", "receiver_target_command",
        "receiver_route_spike_readiness", "receiver_spike_volume",
        "receiver_usage_spike_signal", "target_spike_path", "spike_target",
        "high_target", "recent_target_spike", "recent_route_spike",
        "recent_snap_rise", "route", "pass_route", "route_participation",
        "estimated_routes", "target_share", "targets_share", "target",
        "targets_per_route", "air_yards", "wopr", "first_read",
        "depth_rank_better", "depth_rank_worse", "depth_movement",
        "starter_confidence", "projected_starter", "starter_role_stability",
        "role_continuity", "role_change_upside", "workload_floor",
        "full_workload", "limited_workload", "injury_", "practice_status",
        "same_week_", "teammate_", "vacancy", "receiver_contextual_spike",
        "game_script_pass", "team_implied", "snap_share", "offense_snap",
    ),
    "receiver_target_spike_v2_probability": (
        "receiver_target_spike_v2", "receiver_air_yards_spike_v2",
        "receiver_ypt_efficiency_spike", "receiver_spike_yards_anchor_v2",
        "receiver_projected_targets_v2", "receiver_high_value_target",
        "receiver_live_spike_v3", "receiver_projected_targets_v3",
        "receiver_spike_yards_anchor_v3", "same_week_usage_confidence_v3",
        "yardage_projection_volatility_v3",
        "receiving_usage_history_quality", "live_usage_context_quality_v4",
        "receiver_spike_under_correction_v4", "receiver_spike_yards_anchor_v4",
        "yardage_projection_volatility_v4",
        "receiver_target_route_spike", "receiver_target_command",
        "receiver_route_spike_readiness", "receiver_spike_volume",
        "receiver_usage_spike_signal", "target_spike_path", "spike_target",
        "high_target", "recent_target_spike", "recent_route_spike",
        "recent_snap_rise", "route", "pass_route", "route_participation",
        "estimated_routes", "target_share", "targets_share", "target",
        "targets_per_route", "air_yards", "wopr", "first_read",
        "depth_rank_better", "depth_rank_worse", "depth_movement",
        "starter_confidence", "projected_starter", "starter_role_stability",
        "role_continuity", "role_change_upside", "workload_floor",
        "full_workload", "limited_workload", "workload_downside_v2",
        "workload_upside_v2", "injury_", "practice_status", "same_week_",
        "teammate_", "vacancy", "receiver_contextual_spike",
        "game_script_pass", "team_implied", "snap_share", "offense_snap",
    ),
    "receiver_air_yards_spike_probability": (
        "receiver_air_yards_spike", "receiver_air_yards_spike_v2",
        "air_yards_spike_path", "receiver_air_yards_eruption",
        "receiver_explosive_spike", "receiver_air_yards_spike_anchor",
        "receiving_air_yards", "air_yards_share", "air_yards",
        "wopr", "first_read", "target_share", "targets_share", "target",
        "route", "pass_route", "estimated_routes", "receiver_target_route_spike",
        "receiver_target_spike_v2", "receiver_target_command",
        "receiver_route_spike_readiness", "receiver_usage_spike",
        "receiver_ypt_efficiency_spike", "receiver_contextual_spike",
        "receiver_live_spike_v3", "receiver_spike_under_correction_v4",
        "same_week_usage_confidence_v3", "live_usage_context_quality_v4",
        "yardage_projection_volatility", "depth_rank_better",
        "depth_rank_worse", "depth_movement", "starter_confidence",
        "projected_starter", "starter_role_stability", "role_continuity",
        "role_change_upside", "workload_floor", "full_workload",
        "limited_workload", "workload_downside_v2", "workload_upside_v2",
        "injury_", "practice_status", "same_week_", "teammate_", "vacancy",
        "game_script_pass", "team_implied", "snap_share", "offense_snap",
    ),
    "receiver_spike_under_correction_v4_probability": (
        "receiver_spike_under_correction_v4", "receiver_spike_yards_anchor_v4",
        "receiver_live_spike_v3", "receiver_projected_targets_v3",
        "receiver_target_spike_v2", "receiver_air_yards_spike_v2",
        "receiver_ypt_efficiency_spike", "receiver_target_route_spike",
        "receiver_target_command", "receiver_route_spike_readiness",
        "receiver_spike_volume", "receiver_usage_spike_signal",
        "target_spike_path", "high_target", "recent_target_spike",
        "recent_route_spike", "route", "pass_route", "estimated_routes",
        "target_share", "targets_share", "target", "targets_per_route",
        "air_yards", "wopr", "first_read", "receiving_usage_history_quality",
        "live_usage_context_quality_v4", "same_week_usage_confidence_v3",
        "depth_rank_better", "depth_rank_worse", "depth_movement",
        "starter_confidence", "projected_starter", "starter_role_stability",
        "role_continuity", "role_change_upside", "workload_floor",
        "full_workload", "limited_workload", "workload_downside_v2",
        "workload_upside_v2", "injury_", "practice_status", "same_week_",
        "teammate_", "vacancy", "receiver_contextual_spike",
        "game_script_pass", "team_implied", "snap_share", "offense_snap",
    ),
    "rb_carry_spike_v2_probability": (
        "rb_carry_spike_v2", "rb_projected_carries_v2", "rb_rush_yards_anchor_v2",
        "rb_live_carry_v3", "rb_projected_carries_v3", "rb_rush_yards_anchor_v3",
        "same_week_usage_confidence_v3", "yardage_projection_volatility_v3",
        "rb_usage_history_quality", "live_usage_context_quality_v4",
        "rb_carry_under_correction_v4", "rb_rush_yards_anchor_v4",
        "yardage_projection_volatility_v4",
        "high_carry", "carry_spike_path", "spike_carry", "rb_usage_spike_signal",
        "recent_carry_spike", "carries", "carry", "rushing_yards",
        "game_script_rush", "team_spread", "team_implied", "snap_share",
        "offense_snap", "depth_rank_better", "depth_rank_worse",
        "depth_movement", "starter_confidence", "projected_starter",
        "starter_role_stability", "role_change_upside", "workload_floor",
        "full_workload", "limited_workload", "workload_downside_v2",
        "workload_upside_v2", "rest_risk", "weird_usage", "injury_",
        "practice_status",
    ),
    "rb_carry_under_correction_v4_probability": (
        "rb_carry_under_correction_v4", "rb_rush_yards_anchor_v4",
        "rb_carry_spike_v2", "rb_projected_carries_v2", "rb_rush_yards_anchor_v2",
        "rb_live_carry_v3", "rb_projected_carries_v3", "rb_rush_yards_anchor_v3",
        "rb_usage_history_quality", "live_usage_context_quality_v4",
        "same_week_usage_confidence_v3", "yardage_projection_volatility_v4",
        "high_carry", "carry_spike_path", "spike_carry", "rb_usage_spike_signal",
        "recent_carry_spike", "carries", "carry", "rushing_yards",
        "game_script_rush", "team_spread", "team_implied", "snap_share",
        "offense_snap", "depth_rank_better", "depth_rank_worse",
        "depth_movement", "starter_confidence", "projected_starter",
        "starter_role_stability", "role_change_upside", "workload_floor",
        "full_workload", "limited_workload", "workload_downside_v2",
        "workload_upside_v2", "rest_risk", "weird_usage", "injury_",
        "practice_status",
    ),
    "spike_snap_share_probability": (
        "snap", "offense_snap", "spike_snap", "recent_snap_rise", "depth_rank_better",
        "role_change_upside", "starter_confidence", "full_workload", "limited_workload",
        "injury_", "depth_",
    ),
}

OPPORTUNITY_ALWAYS_KEEP = ("position_", "season", "week", "n_games_prev", "is_home")


@dataclass(frozen=True)
class TrainOpportunityConfig:
    pg_dsn: str = PG_DSN
    model_dir: Path = _MODEL_DIR
    min_prev_games: int = 3
    min_rows: int = 200
    holdout_weeks: int = 4
    random_state: int = 42
    out_file: str = "nfl_player_opportunity_models.joblib"
    report_file: str = "nfl_player_opportunity_models_report.json"
    md_report_file: Path = DEFAULT_MD_REPORT


SQL_TRAIN = """
SELECT *
FROM features.nfl_player_game_training_features
WHERE n_games_prev_3 >= :min_prev_games
  AND game_date_et IS NOT NULL
ORDER BY season, week, game_id, player_id
"""

LEAKAGE_TARGET_COLS = {
    "skill_tds",
    "actual_full_workload",
    "actual_limited_usage",
    "actual_high_pass_attempts",
    "actual_high_carries",
    "actual_high_targets",
    "actual_receiver_spike_volume",
    "actual_receiver_target_route_spike",
    "actual_receiver_air_yards_spike",
    "actual_spike_snap_share",
}


def _prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def num(name: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(out.get(name, pd.Series(default, index=out.index)), errors="coerce").fillna(default)

    def sigmoid(series: pd.Series) -> pd.Series:
        arr = np.clip(pd.to_numeric(series, errors="coerce").fillna(0.0), -35.0, 35.0)
        return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=out.index, dtype=float)

    rushing = pd.to_numeric(out.get("rushing_tds"), errors="coerce").fillna(0.0)
    receiving = pd.to_numeric(out.get("receiving_tds"), errors="coerce").fillna(0.0)
    out["skill_tds"] = (rushing + receiving).clip(lower=0.0)
    rush_avg = pd.to_numeric(out.get("rushing_tds_avg_5"), errors="coerce").fillna(0.0)
    rec_avg = pd.to_numeric(out.get("receiving_tds_avg_5"), errors="coerce").fillna(0.0)
    out["skill_tds_avg_5"] = (rush_avg + rec_avg).clip(lower=0.0)
    snap = pd.to_numeric(out.get("offense_snap_share"), errors="coerce")
    snap = snap.fillna(pd.to_numeric(out.get("snap_share"), errors="coerce"))
    pos = out.get("position", pd.Series("", index=out.index)).astype(str).str.upper()
    full_threshold = pd.Series(0.55, index=out.index, dtype=float)
    full_threshold.loc[pos == "QB"] = 0.75
    full_threshold.loc[pos == "RB"] = 0.45
    full_threshold.loc[pos.isin(["WR", "TE"])] = 0.55
    limited_threshold = pd.Series(0.35, index=out.index, dtype=float)
    limited_threshold.loc[pos == "QB"] = 0.50
    limited_threshold.loc[pos == "RB"] = 0.25
    limited_threshold.loc[pos.isin(["WR", "TE"])] = 0.35
    out["actual_full_workload"] = np.where(snap.notna(), (snap >= full_threshold).astype(float), np.nan)
    out["actual_limited_usage"] = np.where(snap.notna(), (snap <= limited_threshold).astype(float), np.nan)
    pass_attempts = pd.to_numeric(out.get("pass_attempts"), errors="coerce")
    carries = pd.to_numeric(out.get("carries"), errors="coerce")
    targets = pd.to_numeric(out.get("targets"), errors="coerce")
    pass_avg = num("pass_attempts_avg_5", 0.0)
    carries_avg = num("carries_avg_5", 0.0)
    targets_avg = num("targets_avg_5", 0.0)
    snap_avg = num("offense_snap_share_avg_5", 0.0).where(num("offense_snap_share_avg_5", 0.0) > 0, num("snap_share_avg_5", 0.0))

    out["high_pass_attempt_score"] = pd.concat([
        sigmoid((pass_avg - 31.0) / 5.5).clip(0.02, 0.92),
        pd.to_numeric(out.get("spike_pass_attempt_opportunity_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("pass_spike_path_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("qb_volume_spike_signal", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
    ], axis=1).max(axis=1).clip(0.02, 0.92)
    out["high_carry_score"] = pd.concat([
        sigmoid((carries_avg - 11.0) / 3.8).clip(0.02, 0.92),
        pd.to_numeric(out.get("spike_carry_opportunity_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("carry_spike_path_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("rb_usage_spike_signal", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        0.75 * pd.to_numeric(out.get("role_change_upside_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
    ], axis=1).max(axis=1).clip(0.02, 0.92)
    out["high_target_score"] = pd.concat([
        sigmoid((targets_avg - 6.0) / 2.2).clip(0.02, 0.92),
        pd.to_numeric(out.get("spike_target_opportunity_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("target_spike_path_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("receiver_usage_spike_signal", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
        0.75 * pd.to_numeric(out.get("role_change_upside_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0),
    ], axis=1).max(axis=1).clip(0.02, 0.92)
    route_expectation = num("route_weighted_target_expectation_avg_5", 0.0)
    target_share_prior = num("targets_share_avg_5", 0.0).clip(0.0, 1.0)
    first_read_prior = num("first_read_proxy_avg_5", 0.0).clip(0.0, 1.0)
    pass_boost_score = ((num("game_script_pass_boost", 1.0).clip(0.65, 1.35) - 0.65) / 0.70).clip(0.0, 1.0)
    workload_floor = pd.to_numeric(out.get("workload_floor_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    workload_floor = workload_floor.fillna(
        (num("full_workload_score", 0.65).clip(0.0, 1.0) * (1.0 - num("limited_workload_risk_score", 0.0).clip(0.0, 1.0))).clip(0.0, 1.0)
    )
    receiver_health_gate = (
        (0.50 + 0.50 * workload_floor.clip(0.0, 1.0))
        * (1.0 - 0.42 * num("limited_workload_risk_score", 0.0).clip(0.0, 1.0))
        * (1.0 - 0.30 * num("injury_downgrade_score", 0.0).clip(0.0, 1.0))
        * (1.0 - 0.24 * num("depth_movement_risk_score", 0.0).clip(0.0, 1.0))
        * (1.0 - 0.20 * num("weird_usage_risk_score", 0.0).clip(0.0, 1.0))
    ).clip(0.15, 1.0)
    receiver_pos_gate = pos.isin(["RB", "WR", "TE"]).astype(float)
    target_earning_score = sigmoid((num("targets_per_route_proxy_avg_5", 0.0) - 0.22) / 0.055).clip(0.02, 0.92)
    wopr_volume_score = sigmoid((num("wopr_avg_5", 0.0) - 0.48) / 0.11).clip(0.02, 0.92)
    target_trend_score = sigmoid((num("targets_trend_3_10", 0.0) - 0.55) / 0.65).clip(0.02, 0.92)
    target_share_trend_score = sigmoid((num("target_share_trend_3_10", 0.0) - 0.018) / 0.026).clip(0.02, 0.92)
    air_share_score = sigmoid((num("air_yards_share_avg_5", 0.0).clip(0.0, 1.0) - 0.24) / 0.065).clip(0.02, 0.92)
    air_share_trend_score = sigmoid((num("air_yards_share_trend_3_10", 0.0) - 0.022) / 0.032).clip(0.02, 0.92)
    air_volume_score = sigmoid((num("receiving_air_yards_avg_5", 0.0).clip(lower=0.0) - 58.0) / 24.0).clip(0.02, 0.92)
    air_yards_rate_score = sigmoid((num("receiver_yards_rate_signal", 0.0) - 48.0) / 20.0).clip(0.02, 0.92)
    receiver_vacancy = pd.to_numeric(out.get("receiver_teammate_vacancy_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    eruption_calc = (
        receiver_pos_gate
        * (
            0.24 * out["high_target_score"].astype(float).clip(0.0, 1.0)
            + 0.18 * sigmoid((route_expectation - 5.8) / 1.8).clip(0.02, 0.92)
            + 0.15 * sigmoid((target_share_prior - 0.18) / 0.045).clip(0.02, 0.92)
            + 0.14 * target_earning_score
            + 0.11 * target_trend_score
            + 0.09 * target_share_trend_score
            + 0.06 * receiver_vacancy
            + 0.03 * pass_boost_score
        )
        * receiver_health_gate
    ).clip(0.02, 0.94)
    existing_eruption = pd.to_numeric(out.get("receiver_target_eruption_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_target_eruption_score"] = existing_eruption.fillna(eruption_calc).clip(0.02, 0.94)
    air_path_calc = (
        receiver_pos_gate
        * (
            0.23 * air_volume_score
            + 0.20 * air_share_score
            + 0.17 * air_share_trend_score
            + 0.15 * wopr_volume_score
            + 0.10 * sigmoid((first_read_prior - 0.17) / 0.050).clip(0.02, 0.92)
            + 0.09 * air_yards_rate_score
            + 0.06 * pass_boost_score
        )
        * receiver_health_gate
    ).clip(0.02, 0.94)
    existing_air_path = pd.to_numeric(out.get("air_yards_spike_path_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["air_yards_spike_path_score"] = existing_air_path.fillna(air_path_calc).clip(0.02, 0.94)
    air_eruption_calc = (
        receiver_pos_gate
        * pd.concat([
            out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
            0.90 * air_volume_score,
            0.88 * air_share_trend_score,
            0.84 * wopr_volume_score,
        ], axis=1).max(axis=1).fillna(0.0)
        * receiver_health_gate
    ).clip(0.02, 0.95)
    existing_air_eruption = pd.to_numeric(out.get("receiver_air_yards_eruption_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_air_yards_eruption_score"] = existing_air_eruption.fillna(air_eruption_calc).clip(0.02, 0.95)
    explosive_calc = (
        receiver_pos_gate
        * (
            0.40 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.34 * out["receiver_air_yards_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.16 * receiver_vacancy
            + 0.10 * pass_boost_score
        )
        * receiver_health_gate
    ).clip(0.02, 0.95)
    existing_explosive = pd.to_numeric(out.get("receiver_explosive_spike_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_explosive_spike_score"] = existing_explosive.fillna(explosive_calc).clip(0.02, 0.95)
    out["target_spike_path_score"] = pd.concat([
        pd.to_numeric(out.get("target_spike_path_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0),
        0.90 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0),
        0.72 * out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    target_eruption_anchor_calc = (
        receiver_pos_gate
        * pd.concat([
            route_expectation.clip(lower=0.0),
            targets_avg.clip(lower=0.0) + num("targets_trend_3_10", 0.0).clip(lower=0.0),
            num("estimated_routes_avg_5", 0.0).clip(lower=0.0) * num("targets_per_route_proxy_avg_5", 0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.96 + 0.24 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    existing_target_eruption_anchor = pd.to_numeric(out.get("receiver_target_eruption_anchor_targets", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_target_eruption_anchor_targets"] = existing_target_eruption_anchor.fillna(target_eruption_anchor_calc).clip(lower=0.0)
    air_anchor_calc = (
        receiver_pos_gate
        * pd.concat([
            num("receiver_yards_rate_signal", 0.0).clip(lower=0.0),
            num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
            0.78 * num("receiving_air_yards_avg_5", 0.0).clip(lower=0.0),
            num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.92 + 0.22 * out["receiver_air_yards_eruption_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    existing_air_anchor = pd.to_numeric(out.get("receiver_air_yards_spike_anchor_yards", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_air_yards_spike_anchor_yards"] = existing_air_anchor.fillna(air_anchor_calc).clip(lower=0.0)
    out["receiver_usage_spike_signal"] = pd.concat([
        pd.to_numeric(out.get("receiver_usage_spike_signal", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0),
        0.88 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0),
        0.82 * out["receiver_explosive_spike_score"].astype(float).clip(0.0, 1.0),
        0.70 * out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    receiver_score_calc = (
        receiver_pos_gate
        * (
            0.18 * out["high_target_score"].astype(float).clip(0.0, 1.0)
            + 0.18 * sigmoid((route_expectation - 5.8) / 1.8).clip(0.0, 1.0)
            + 0.13 * sigmoid((target_share_prior - 0.18) / 0.045).clip(0.0, 1.0)
            + 0.12 * sigmoid((first_read_prior - 0.17) / 0.050).clip(0.0, 1.0)
            + 0.13 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * out["receiver_explosive_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.09 * pd.to_numeric(out.get("receiver_usage_spike_signal", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            + 0.05 * pass_boost_score
            + 0.07 * pd.to_numeric(out.get("role_change_upside_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            + 0.05 * receiver_vacancy
        )
        * receiver_health_gate
    ).clip(0.02, 0.92)
    existing_receiver_score = pd.to_numeric(out.get("receiver_spike_volume_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_spike_volume_score"] = existing_receiver_score.fillna(receiver_score_calc).clip(0.02, 0.92)
    receiver_anchor = pd.concat([
        out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
        num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
        num("receiver_yards_rate_signal", 0.0).clip(lower=0.0),
        num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
    ], axis=1).max(axis=1).fillna(0.0)
    existing_receiver_anchor = pd.to_numeric(out.get("receiver_spike_volume_anchor_yards", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_spike_volume_anchor_yards"] = existing_receiver_anchor.fillna(
        receiver_pos_gate
        * receiver_anchor
        * (0.82 + 0.30 * out["receiver_spike_volume_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    route_cut = pd.Series(0.72, index=out.index, dtype=float)
    route_cut.loc[pos == "TE"] = 0.66
    route_cut.loc[pos == "RB"] = 0.42
    route_stability_score = sigmoid((num("route_participation_proxy_avg_5", 0.0) - route_cut) / 0.090).clip(0.02, 0.92)
    route_depth_score = sigmoid((num("estimated_routes_avg_5", 0.0) - 30.0) / 6.0).clip(0.02, 0.92)
    command_calc = (
        receiver_pos_gate
        * (
            0.28 * out["high_target_score"].astype(float).clip(0.0, 1.0)
            + 0.21 * sigmoid((target_share_prior - 0.18) / 0.045).clip(0.02, 0.92)
            + 0.18 * sigmoid((first_read_prior - 0.17) / 0.050).clip(0.02, 0.92)
            + 0.18 * target_earning_score
            + 0.15 * wopr_volume_score
        )
        * (0.70 + 0.30 * num("role_continuity_score", 0.55).clip(0.0, 1.0))
        * receiver_health_gate
    ).clip(0.02, 0.94)
    existing_command = pd.to_numeric(out.get("receiver_target_command_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_target_command_score"] = existing_command.fillna(command_calc).clip(0.02, 0.94)
    route_ready_calc = (
        receiver_pos_gate
        * (
            0.30 * route_stability_score
            + 0.24 * route_depth_score
            + 0.16 * num("recent_route_spike_score", 0.0).clip(0.0, 1.0)
            + 0.14 * num("recent_snap_rise_score", 0.0).clip(0.0, 1.0)
            + 0.10 * num("role_change_upside_score", 0.0).clip(0.0, 1.0)
            + 0.06 * num("starter_role_stability_score", 0.55).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.94)
    existing_route_ready = pd.to_numeric(out.get("receiver_route_spike_readiness_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_route_spike_readiness_score"] = existing_route_ready.fillna(route_ready_calc).clip(0.02, 0.94)
    target_route_calc = (
        receiver_pos_gate
        * (
            0.27 * out["receiver_target_command_score"].astype(float).clip(0.0, 1.0)
            + 0.20 * out["receiver_route_spike_readiness_score"].astype(float).clip(0.0, 1.0)
            + 0.13 * out["receiver_spike_volume_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * sigmoid((route_expectation - 5.8) / 1.8).clip(0.02, 0.92)
            + 0.10 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.09 * pass_boost_score
            + 0.06 * num("target_spike_path_score", 0.0).clip(0.0, 1.0)
            + 0.05 * num("role_change_upside_score", 0.0).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.95)
    existing_target_route = pd.to_numeric(out.get("receiver_target_route_spike_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_target_route_spike_score"] = existing_target_route.fillna(target_route_calc).clip(0.02, 0.95)
    target_anchor_calc = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_target_eruption_anchor_targets"].astype(float).clip(lower=0.0),
            route_expectation.clip(lower=0.0),
            targets_avg.clip(lower=0.0) + 0.75 * num("targets_trend_3_10", 0.0).clip(lower=0.0),
            num("estimated_routes_avg_5", 0.0).clip(lower=0.0) * num("targets_per_route_proxy_avg_5", 0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.92 + 0.22 * out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    existing_target_anchor = pd.to_numeric(out.get("receiver_target_route_spike_anchor_targets", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_target_route_spike_anchor_targets"] = existing_target_anchor.fillna(target_anchor_calc).clip(lower=0.0)
    yards_per_target_proxy = (
        num("receiving_yards_avg_5", 0.0).replace(0.0, np.nan)
        / targets_avg.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    yards_per_target_proxy = yards_per_target_proxy.fillna(
        num("yards_per_route_proxy_avg_5", 0.0).clip(lower=0.0)
        / num("targets_per_route_proxy_avg_5", 0.0).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(8.4).clip(3.5, 18.0)
    yards_anchor_calc = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_spike_volume_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_target_route_spike_anchor_targets"].astype(float).clip(lower=0.0) * yards_per_target_proxy,
            out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
            num("receiver_yards_rate_signal", 0.0).clip(lower=0.0),
            num("route_weighted_yard_expectation_avg_5", 0.0).clip(lower=0.0),
            num("receiving_yards_avg_5", 0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.90 + 0.18 * out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    existing_yards_anchor = pd.to_numeric(out.get("receiver_target_route_spike_anchor_yards", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["receiver_target_route_spike_anchor_yards"] = existing_yards_anchor.fillna(yards_anchor_calc).clip(lower=0.0)
    spike_snap_cut = pd.Series(0.72, index=out.index, dtype=float)
    spike_snap_cut.loc[pos == "QB"] = 0.92
    spike_snap_cut.loc[pos == "RB"] = 0.58
    spike_snap_cut.loc[pos == "TE"] = 0.70
    out["spike_snap_share_score"] = sigmoid((snap_avg - spike_snap_cut) / 0.11).clip(0.02, 0.92)

    pass_threshold = np.maximum(34.0, pass_avg.to_numpy(dtype=float) + 6.0)
    carry_threshold = np.maximum(12.0, carries_avg.to_numpy(dtype=float) + 4.0)
    target_threshold = np.maximum(7.0, targets_avg.to_numpy(dtype=float) + 3.0)
    receiver_target_threshold = np.maximum(7.0, targets_avg.to_numpy(dtype=float) + 2.5)
    receiver_share_threshold = np.maximum(0.23, (target_share_prior + 0.055).to_numpy(dtype=float))
    receiver_share_threshold = np.minimum(receiver_share_threshold, 0.36)
    target_share_actual = pd.to_numeric(out.get("target_share"), errors="coerce")
    route_part_actual = pd.to_numeric(out.get("route_participation"), errors="coerce")
    route_part_actual = route_part_actual.fillna(pd.to_numeric(out.get("pass_route_opportunity_share"), errors="coerce"))
    receiver_spike = (
        (targets.to_numpy(dtype=float) >= receiver_target_threshold)
        | ((target_share_actual.fillna(-1.0).to_numpy(dtype=float) >= receiver_share_threshold) & (targets.fillna(0.0).to_numpy(dtype=float) >= 5.0))
        | ((targets.fillna(0.0).to_numpy(dtype=float) >= 7.0) & (route_part_actual.fillna(0.0).to_numpy(dtype=float) >= 0.72))
    )
    route_prior = num("pass_route_opportunity_share_avg_5", 0.0)
    route_prior = route_prior.where(route_prior > 0, num("route_participation_avg_5", 0.0))
    route_prior = route_prior.where(route_prior > 0, snap_avg)
    route_floor = np.where(pos == "RB", 0.42, np.where(pos == "TE", 0.66, 0.72))
    route_ceiling = np.where(pos == "RB", 0.74, 0.92)
    strict_route_threshold = np.minimum(
        np.maximum(route_prior.to_numpy(dtype=float) + 0.08, route_floor),
        route_ceiling,
    )
    strict_route_spike = (
        (route_part_actual.fillna(-1.0).to_numpy(dtype=float) >= strict_route_threshold)
        | (route_part_actual.fillna(-1.0).to_numpy(dtype=float) >= np.where(pos == "RB", 0.48, np.where(pos == "TE", 0.74, 0.80)))
    )
    strict_snap_spike = (
        snap.fillna(-1.0).to_numpy(dtype=float)
        >= np.minimum(np.maximum(snap_avg.to_numpy(dtype=float) + 0.08, 0.68), 0.92)
    )
    strict_target_threshold = np.maximum(np.where(pos == "RB", 6.0, 8.0), targets_avg.to_numpy(dtype=float) + np.where(pos == "RB", 2.0, 3.0))
    strict_share_threshold = np.minimum(
        np.maximum(target_share_prior.to_numpy(dtype=float) + np.where(pos == "RB", 0.045, 0.065), np.where(pos == "RB", 0.18, 0.245)),
        0.38,
    )
    strict_target_spike = (
        (targets.to_numpy(dtype=float) >= strict_target_threshold)
        | (
            (target_share_actual.fillna(-1.0).to_numpy(dtype=float) >= strict_share_threshold)
            & (targets.fillna(0.0).to_numpy(dtype=float) >= np.where(pos == "RB", 4.0, 6.0))
        )
    )
    air_actual = pd.to_numeric(out.get("receiving_air_yards"), errors="coerce")
    air_prior = num("receiving_air_yards_avg_5", 0.0)
    air_share_actual = pd.to_numeric(out.get("air_yards_share"), errors="coerce")
    air_share_prior = num("air_yards_share_avg_5", 0.0).clip(0.0, 1.0)
    air_threshold = np.maximum(70.0, air_prior.to_numpy(dtype=float) + 38.0)
    air_share_threshold = np.minimum(
        np.maximum(air_share_prior.to_numpy(dtype=float) + 0.075, 0.27),
        0.48,
    )
    strict_air_spike = (
        (air_actual.fillna(-1.0).to_numpy(dtype=float) >= air_threshold)
        | (
            (air_share_actual.fillna(-1.0).to_numpy(dtype=float) >= air_share_threshold)
            & (air_actual.fillna(0.0).to_numpy(dtype=float) >= 45.0)
        )
    )
    receiver_target_route_spike = (
        (strict_target_spike | strict_air_spike)
        & (strict_route_spike | strict_snap_spike | (target_share_actual.fillna(-1.0).to_numpy(dtype=float) >= 0.30))
    )
    spike_threshold = np.minimum(np.maximum((spike_snap_cut + 0.08).to_numpy(dtype=float), 0.62), 0.95)
    out["actual_high_pass_attempts"] = np.where(pass_attempts.notna(), (pass_attempts.to_numpy(dtype=float) >= pass_threshold).astype(float), np.nan)
    out["actual_high_carries"] = np.where(carries.notna(), (carries.to_numpy(dtype=float) >= carry_threshold).astype(float), np.nan)
    out["actual_high_targets"] = np.where(targets.notna(), (targets.to_numpy(dtype=float) >= target_threshold).astype(float), np.nan)
    out["actual_receiver_spike_volume"] = np.where(targets.notna() & pos.isin(["RB", "WR", "TE"]), receiver_spike.astype(float), np.nan)
    out["actual_receiver_target_route_spike"] = np.where(targets.notna() & pos.isin(["RB", "WR", "TE"]), receiver_target_route_spike.astype(float), np.nan)
    out["actual_receiver_air_yards_spike"] = np.where(
        (air_actual.notna() | air_share_actual.notna()) & pos.isin(["RB", "WR", "TE"]),
        strict_air_spike.astype(float),
        np.nan,
    )
    out["actual_spike_snap_share"] = np.where(snap.notna(), (snap.to_numpy(dtype=float) >= spike_threshold).astype(float), np.nan)
    return out


def _align_features(train_df: pd.DataFrame, holdout_df: pd.DataFrame, *, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, float]]:
    drop_cols = sorted(LEAKAGE_TARGET_COLS | {target_col})
    X_train_raw = _make_features(train_df.drop(columns=drop_cols, errors="ignore"))
    X_holdout_raw = _make_features(holdout_df.drop(columns=drop_cols, errors="ignore"))
    columns = list(X_train_raw.columns)
    fill_values = {
        col: float(value) if math.isfinite(float(value)) else 0.0
        for col, value in X_train_raw.median(numeric_only=True).fillna(0.0).to_dict().items()
    }
    X_train = X_train_raw.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    X_holdout = X_holdout_raw.reindex(columns=columns).fillna(fill_values).fillna(0.0)
    return X_train, X_holdout, columns, fill_values


def _select_opportunity_columns(columns: list[str], name: str) -> tuple[list[str], dict[str, Any]]:
    patterns = OPPORTUNITY_FEATURE_POLICIES.get(name)
    if not patterns:
        return columns, {"status": "not_configured"}
    selected = [
        col for col in columns
        if any(pattern in col for pattern in OPPORTUNITY_ALWAYS_KEEP)
        or any(pattern in col for pattern in patterns)
    ]
    if len(selected) < max(5, int(len(columns) * 0.08)):
        return columns, {
            "status": "candidate_too_sparse",
            "full_columns": len(columns),
            "candidate_columns": len(selected),
            "patterns": list(patterns),
        }
    return selected, {
        "status": "candidate_built",
        "full_columns": len(columns),
        "candidate_columns": len(selected),
        "patterns": list(patterns),
    }


def _metric_row(name: str, rec: dict[str, Any]) -> str:
    accepted = "yes" if rec.get("accepted") or rec.get("projection_pass") else "no"
    upside_signal = "yes" if rec.get("upside_signal_pass") else "no"
    return (
        f"| {name} | {int(rec.get('rows') or rec.get('holdout_rows') or 0)} | "
        f"{float(rec.get('mae') or 0):.3f} | {float(rec.get('baseline_mae') or 0):.3f} | "
        f"{float(rec.get('mae_gain_vs_baseline') or 0):+.3f} | "
        f"{float(rec.get('bias') or 0):+.3f} | {accepted} | {upside_signal} | {rec.get('status', '-')} |"
    )


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Player Opportunity Models",
        "",
        "These models estimate opportunity before pricing player props: pass attempts, carries, targets, receptions, TD role, and full-workload/rest risk.",
        "Route and snap-share targets are included, but they require a source with non-null route/snap data.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Training rows: {payload.get('rows')}",
        f"- Trained at: {payload.get('trained_at_utc')}",
        "",
        "| Model | Holdout Rows | MAE | Baseline MAE | Gain | Bias | Accepted | Upside Signal | Status |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for name, rec in (payload.get("metrics") or {}).items():
        if isinstance(rec, dict):
            lines.append(_metric_row(name, rec))
    lines.extend([
        "",
        "## Data Coverage",
        "",
        "| Field | Non-null Rows | Coverage |",
        "|---|---:|---:|",
    ])
    for field, rec in (payload.get("coverage") or {}).items():
        lines.append(f"| {field} | {rec.get('non_null', 0)} | {float(rec.get('coverage') or 0):.1%} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def train(cfg: TrainOpportunityConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(cfg.pg_dsn)
    df = pd.read_sql(text(SQL_TRAIN), engine, params={"min_prev_games": cfg.min_prev_games})
    df = _prepare_targets(df)
    payload: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "rows": int(len(df)),
        "models": {},
        "metrics": {},
        "feature_columns": {},
        "fill_values": {},
        "coverage": {},
    }
    for field in (
        "pass_attempts",
        "carries",
        "targets",
        "receptions",
        "route_participation",
        "pass_route_opportunities",
        "pass_route_opportunity_share",
        "snap_share",
        "offense_snap_share",
        "target_share",
        "air_yards_share",
        "receiving_air_yards",
        "wopr",
        "targets_per_route_run",
        "yards_per_route_run",
        "first_read_targets",
        "first_read_target_share",
        "end_zone_targets",
        "end_zone_target_share",
        "red_zone_carries",
        "red_zone_targets",
        "red_zone_pass_attempts",
        "red_zone_touches",
        "goal_line_carries",
        "goal_line_targets",
        "team_implied_points",
        "depth_pos_rank",
        "starter_confidence",
        "rest_risk_score",
        "full_workload_score",
        "limited_workload_risk_score",
        "high_usage_fragility_score",
        "usage_volatility_score",
        "backup_role_score",
        "starter_role_stability_score",
        "recent_snap_drop_score",
        "recent_snap_rise_score",
        "recent_target_spike_score",
        "recent_carry_spike_score",
        "recent_route_spike_score",
        "actual_full_workload",
        "actual_limited_usage",
        "high_pass_attempt_score",
        "high_carry_score",
        "high_target_score",
        "receiver_target_eruption_score",
        "air_yards_spike_path_score",
        "receiver_air_yards_eruption_score",
        "receiver_explosive_spike_score",
        "receiver_spike_volume_score",
        "receiver_spike_volume_anchor_yards",
        "receiver_target_command_score",
        "receiver_route_spike_readiness_score",
        "receiver_target_route_spike_score",
        "receiver_contextual_spike_score",
        "receiver_target_spike_v2_score",
        "receiver_air_yards_spike_v2_score",
        "receiver_ypt_efficiency_spike_score",
        "receiver_spike_yards_anchor_v2",
        "receiver_projected_targets_v2",
        "receiver_high_value_target_score",
        "rb_carry_spike_v2_score",
        "rb_projected_carries_v2",
        "rb_rush_yards_anchor_v2",
        "same_week_usage_confidence_v3_score",
        "receiver_live_spike_v3_score",
        "receiver_projected_targets_v3",
        "receiver_spike_yards_anchor_v3",
        "rb_live_carry_v3_score",
        "rb_projected_carries_v3",
        "rb_rush_yards_anchor_v3",
        "yardage_projection_volatility_v3_score",
        "receiving_usage_history_quality_score",
        "rb_usage_history_quality_score",
        "td_usage_history_quality_score",
        "live_usage_context_quality_v4_score",
        "receiver_spike_under_correction_v4_score",
        "receiver_spike_yards_anchor_v4",
        "rb_carry_under_correction_v4_score",
        "rb_rush_yards_anchor_v4",
        "yardage_projection_volatility_v4_score",
        "workload_downside_v2_score",
        "workload_upside_v2_score",
        "receiver_teammate_vacancy_score",
        "teammate_receiver_injury_pressure_score",
        "same_week_teammate_skill_injury_count",
        "same_week_teammate_skill_injury_score",
        "same_week_teammate_receiver_injury_score",
        "same_week_teammate_receiver_out_count",
        "receiver_target_eruption_anchor_targets",
        "receiver_air_yards_spike_anchor_yards",
        "receiver_target_route_spike_anchor_targets",
        "receiver_target_route_spike_anchor_yards",
        "spike_snap_share_score",
        "spike_target_opportunity_score",
        "spike_carry_opportunity_score",
        "spike_pass_attempt_opportunity_score",
        "target_spike_path_score",
        "carry_spike_path_score",
        "pass_spike_path_score",
        "normal_usage_path_score",
        "weird_usage_risk_score",
        "role_continuity_score",
        "projected_starter_score",
        "actual_high_pass_attempts",
        "actual_high_carries",
        "actual_high_targets",
        "actual_receiver_spike_volume",
        "actual_receiver_target_route_spike",
        "actual_receiver_air_yards_spike",
        "actual_spike_snap_share",
    ):
        series = pd.to_numeric(df.get(field, pd.Series(index=df.index)), errors="coerce")
        payload["coverage"][field] = {
            "non_null": int(series.notna().sum()),
            "coverage": float(series.notna().mean()) if len(series) else 0.0,
        }
    if df.empty:
        payload["status"] = "no_training_rows"
        joblib.dump(payload, cfg.model_dir / cfg.out_file)
        (cfg.model_dir / cfg.report_file).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _write_markdown(payload, cfg.md_report_file)
        return payload

    for spec in OPPORTUNITY_SPECS:
        sub = df.loc[df["position"].astype(str).str.upper().isin(spec.positions)].copy()
        sub = sub.loc[pd.to_numeric(sub.get(spec.target), errors="coerce").notna()].copy()
        if len(sub) < cfg.min_rows:
            payload["metrics"][spec.name] = {
                "status": "insufficient_rows",
                "rows": int(len(sub)),
                "min_rows": cfg.min_rows,
                "projection_pass": False,
            }
            continue
        train_df, holdout_df = _temporal_split(sub, cfg.holdout_weeks)
        if len(train_df) < cfg.min_rows or holdout_df.empty:
            payload["metrics"][spec.name] = {
                "status": "insufficient_split_rows",
                "train_rows": int(len(train_df)),
                "holdout_rows": int(len(holdout_df)),
                "projection_pass": False,
            }
            continue
        X_train, X_holdout, columns, fill_values = _align_features(train_df, holdout_df, target_col=spec.target)
        selected_columns, feature_selection = _select_opportunity_columns(columns, spec.name)
        if selected_columns != columns:
            X_train = X_train.reindex(columns=selected_columns)
            X_holdout = X_holdout.reindex(columns=selected_columns)
            fill_values = {col: fill_values.get(col, 0.0) for col in selected_columns}
            columns = selected_columns
        y_train = pd.to_numeric(train_df[spec.target], errors="coerce").fillna(0.0).clip(lower=0.0)
        y_holdout = pd.to_numeric(holdout_df[spec.target], errors="coerce").fillna(0.0).clip(lower=0.0)
        probability_target = spec.target.startswith("actual_")
        if probability_target:
            y_train = y_train.clip(0.0, 1.0)
            y_holdout = y_holdout.clip(0.0, 1.0)
        direct_model = _fit_model(X_train, y_train, count_like=spec.count_like, cfg=cfg)
        direct_pred = np.clip(direct_model.predict(X_holdout), 0.0, None)
        if probability_target:
            direct_pred = np.clip(direct_pred, 0.0, 1.0)
        residual_shrink = 0.60
        train_base = pd.to_numeric(train_df.get(spec.baseline_col), errors="coerce").fillna(float(y_train.mean() if len(y_train) else 0.0))
        holdout_base = pd.to_numeric(holdout_df.get(spec.baseline_col), errors="coerce").fillna(float(y_train.mean() if len(y_train) else 0.0))
        residual_model = _fit_model(X_train, y_train - train_base, count_like=False, cfg=cfg)
        residual_pred = np.clip(holdout_base.to_numpy(dtype=float) + residual_shrink * residual_model.predict(X_holdout), 0.0, None)
        if probability_target:
            residual_pred = np.clip(residual_pred, 0.0, 1.0)
        baseline_suite = _baseline_suite(spec.target, train_df, holdout_df)
        if spec.baseline_col not in baseline_suite and spec.baseline_col in holdout_df.columns:
            baseline_suite[spec.baseline_col] = holdout_base
        baseline_name, baseline, baseline_summary = _best_baseline(y_holdout, baseline_suite)
        direct_metrics = _metric_payload(y_holdout, direct_pred, baseline)
        residual_metrics = _metric_payload(y_holdout, residual_pred, baseline)
        use_residual = residual_metrics["mae"] <= direct_metrics["mae"] - 0.001
        model = {
            "kind": "residual" if use_residual else "direct",
            "model": residual_model if use_residual else direct_model,
            "baseline_col": spec.baseline_col,
            "shrink": residual_shrink if use_residual else 0.0,
        }
        pred = residual_pred if use_residual else direct_pred
        probability_candidates: list[dict[str, Any]] = []
        metrics = _metric_payload(y_holdout, pred, baseline)
        if probability_target:
            baseline_prob = pd.to_numeric(baseline, errors="coerce").fillna(float(y_train.mean() if len(y_train) else 0.0)).clip(0.0, 1.0)
            base_prob_arr = baseline_prob.to_numpy(dtype=float)
            raw_candidates = [
                ("direct", direct_model, direct_pred, 0.0),
                ("residual", residual_model, residual_pred, residual_shrink),
            ]
            for variant, candidate_model, candidate_pred, shrink in raw_candidates:
                for blend_weight in (1.0, 0.80, 0.65, 0.50, 0.35, 0.20):
                    blended = np.clip(
                        blend_weight * np.asarray(candidate_pred, dtype=float)
                        + (1.0 - blend_weight) * base_prob_arr,
                        0.0,
                        1.0,
                    )
                    candidate_metrics = _metric_payload(y_holdout, blended, baseline)
                    candidate_metrics["brier"] = float(
                        brier_score_loss(y_holdout.astype(int), np.clip(blended, 0.001, 0.999))
                    )
                    probability_candidates.append({
                        "variant": variant,
                        "blend_weight": blend_weight,
                        "shrink": shrink,
                        "model": candidate_model,
                        "prediction": blended,
                        "mae": candidate_metrics["mae"],
                        "brier": candidate_metrics["brier"],
                        "bias": candidate_metrics["bias"],
                    })
            best_probability = min(probability_candidates, key=lambda row: (row["brier"], row["mae"]))
            model = {
                "kind": str(best_probability["variant"]),
                "model": best_probability["model"],
                "baseline_col": spec.baseline_col,
                "shrink": float(best_probability["shrink"]),
                "probability_blend_weight": float(best_probability["blend_weight"]),
            }
            pred = np.asarray(best_probability["prediction"], dtype=float)
            metrics = _metric_payload(y_holdout, pred, baseline)
            metrics["brier"] = float(best_probability["brier"])
            metrics["baseline_brier"] = float(brier_score_loss(y_holdout.astype(int), baseline_prob.clip(0.001, 0.999)))
            metrics["brier_gain_vs_baseline"] = metrics["baseline_brier"] - metrics["brier"]
            metrics["projection_pass"] = bool(metrics["brier"] <= metrics["baseline_brier"] - 0.001)
            metrics["probability_blend_weight"] = float(best_probability["blend_weight"])
            if spec.name in UPSIDE_SIGNAL_MODELS:
                upside_pool = [
                    row for row in probability_candidates
                    if float(row["brier"]) <= metrics["baseline_brier"] + 0.010
                ]
                best_upside = min(upside_pool or probability_candidates, key=lambda row: (row["mae"], row["brier"]))
                metrics["upside_signal_pass"] = bool(
                    best_upside["mae"] <= metrics["baseline_mae"] - 0.015
                    and best_upside["brier"] <= metrics["baseline_brier"] + 0.010
                )
                metrics["upside_signal_variant"] = str(best_upside["variant"])
                metrics["upside_signal_blend_weight"] = float(best_upside["blend_weight"])
                metrics["upside_signal_brier"] = float(best_upside["brier"])
                metrics["upside_signal_mae"] = float(best_upside["mae"])
                if metrics["upside_signal_pass"]:
                    model["upside_kind"] = str(best_upside["variant"])
                    model["upside_model"] = best_upside["model"]
                    model["upside_shrink"] = float(best_upside["shrink"])
                    model["upside_probability_blend_weight"] = float(best_upside["blend_weight"])
            else:
                metrics["upside_signal_pass"] = False
            metrics["probability_candidates"] = [
                {k: v for k, v in row.items() if k not in {"model", "prediction"}}
                for row in probability_candidates
            ]
        metrics.update({
            "status": "trained",
            "label": spec.label,
            "target": spec.target,
            "positions": list(spec.positions),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "baseline_column": baseline_name,
            "baseline_suite": baseline_summary,
            "variant": model["kind"],
            "direct_mae": direct_metrics["mae"],
            "residual_mae": residual_metrics["mae"],
            "residual_shrink": residual_shrink,
            "feature_selection": feature_selection,
            "accepted": bool(metrics["projection_pass"]),
            "live_signal_accepted": bool(metrics["projection_pass"] or metrics.get("upside_signal_pass")),
        })
        payload["models"][spec.name] = model
        payload["metrics"][spec.name] = metrics
        payload["feature_columns"][spec.name] = columns
        payload["fill_values"][spec.name] = fill_values
        log.info(
            "%s: holdout MAE %.3f vs baseline %.3f accepted=%s",
            spec.name,
            metrics["mae"],
            metrics["baseline_mae"],
            metrics["accepted"],
        )

    joblib.dump(payload, cfg.model_dir / cfg.out_file)
    report = dict(payload)
    report["models"] = sorted(payload["models"].keys())
    (cfg.model_dir / cfg.report_file).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, cfg.md_report_file)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFL player opportunity models")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--holdout-weeks", type=int, default=4)
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD_REPORT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = train(TrainOpportunityConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        min_rows=args.min_rows,
        holdout_weeks=args.holdout_weeks,
        md_report_file=Path(args.md_report_file),
    ))
    print(json.dumps(
        {k: v for k, v in result.items() if k not in {"models", "feature_columns", "fill_values"}},
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
