"""
Shared feature engineering functions.

add_game_derived_features        — used by train_game_models.make_xy_raw()
                                   and predict_today._prep_features()
add_player_prop_derived_features — used by train_player_prop_models.make_xy_raw()
                                   and predict_player_props._prep_X()

Adding a new derived feature?  Edit the function here ONCE.  Both training
and inference pick it up automatically.  No need to touch two files.
"""

import pandas as pd


def add_game_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived interaction features for game models in-place.

    Assumes X already has:
      - b2b flags encoded as int (0/1)
      - categorical columns (season, team abbrs) already one-hot encoded
      - all raw feature columns present (NaNs are fine — guards check membership)

    Returns X with new columns appended.
    """
    # Rest advantage
    if "home_rest_days" in X.columns and "away_rest_days" in X.columns:
        X["rest_advantage_home"] = X["home_rest_days"] - X["away_rest_days"]

    # Opponent-adjusted net ratings
    if "home_pts_for_avg_10" in X.columns and "home_pts_against_avg_10" in X.columns:
        X["home_net_rating_10"] = X["home_pts_for_avg_10"] - X["home_pts_against_avg_10"]
    if "away_pts_for_avg_10" in X.columns and "away_pts_against_avg_10" in X.columns:
        X["away_net_rating_10"] = X["away_pts_for_avg_10"] - X["away_pts_against_avg_10"]
    if "home_net_rating_10" in X.columns and "away_net_rating_10" in X.columns:
        X["net_rating_diff_10"] = X["home_net_rating_10"] - X["away_net_rating_10"]

    # Pace
    if "home_pace_avg_5" in X.columns and "away_pace_avg_5" in X.columns:
        X["pace_diff_5"] = X["home_pace_avg_5"] - X["away_pace_avg_5"]

    # Scoring volume
    if "home_pts_for_avg_5" in X.columns and "away_pts_for_avg_5" in X.columns:
        X["pts_for_diff_5"] = X["home_pts_for_avg_5"] - X["away_pts_for_avg_5"]

    # Shooting efficiency differentials
    if "home_efg_pct_avg_5" in X.columns and "away_efg_pct_avg_5" in X.columns:
        X["efg_diff_5"] = X["home_efg_pct_avg_5"] - X["away_efg_pct_avg_5"]
    if "home_ts_pct_avg_5" in X.columns and "away_ts_pct_avg_5" in X.columns:
        X["ts_diff_5"] = X["home_ts_pct_avg_5"] - X["away_ts_pct_avg_5"]
    if "home_fg3a_rate_avg_5" in X.columns and "away_fg3a_rate_avg_5" in X.columns:
        X["fg3a_rate_diff_5"] = X["home_fg3a_rate_avg_5"] - X["away_fg3a_rate_avg_5"]
    if "home_tov_rate_avg_5" in X.columns and "away_tov_rate_avg_5" in X.columns:
        X["tov_rate_diff_5"] = X["home_tov_rate_avg_5"] - X["away_tov_rate_avg_5"]

    # Injury impact
    if "home_injured_pts_lost" in X.columns and "away_injured_pts_lost" in X.columns:
        X["injury_pts_diff"] = X["away_injured_pts_lost"] - X["home_injured_pts_lost"]

    # Clutch performance
    if "home_clutch_net_avg_10" in X.columns and "away_clutch_net_avg_10" in X.columns:
        X["clutch_net_diff_10"] = X["home_clutch_net_avg_10"] - X["away_clutch_net_avg_10"]

    # Odds-juice signal (V005)
    if "spread_home_implied_prob" in X.columns:
        X["spread_implied_edge"] = X["spread_home_implied_prob"] - 0.5

    # Team style differentials (V006)
    if "home_stocks_avg_10" in X.columns and "away_stocks_avg_10" in X.columns:
        X["stocks_diff_10"] = X["home_stocks_avg_10"] - X["away_stocks_avg_10"]
    if "home_ast_tov_ratio_10" in X.columns and "away_ast_tov_ratio_10" in X.columns:
        X["ast_tov_ratio_diff_10"] = X["home_ast_tov_ratio_10"] - X["away_ast_tov_ratio_10"]
    if "home_pts_paint_avg_10" in X.columns and "away_pts_paint_avg_10" in X.columns:
        X["paint_pts_diff_10"] = X["home_pts_paint_avg_10"] - X["away_pts_paint_avg_10"]
    if "home_pts_fast_break_avg_10" in X.columns and "away_pts_fast_break_avg_10" in X.columns:
        X["fast_break_diff_10"] = X["home_pts_fast_break_avg_10"] - X["away_pts_fast_break_avg_10"]
    if "home_bench_pct_10" in X.columns and "away_bench_pct_10" in X.columns:
        X["bench_depth_diff_10"] = X["home_bench_pct_10"] - X["away_bench_pct_10"]
    if "home_fouls_avg_10" in X.columns and "away_fouls_avg_10" in X.columns:
        X["fouls_diff_10"] = X["home_fouls_avg_10"] - X["away_fouls_avg_10"]

    # Lineup stability (V008)
    if "home_starter_continuity_avg_10" in X.columns and "away_starter_continuity_avg_10" in X.columns:
        X["continuity_diff_10"] = X["home_starter_continuity_avg_10"] - X["away_starter_continuity_avg_10"]

    # Standings differentials (V009)
    if "home_streak" in X.columns and "away_streak" in X.columns:
        X["streak_diff"] = X["home_streak"] - X["away_streak"]
    if "home_last10_pct" in X.columns and "away_last10_pct" in X.columns:
        X["last10_pct_diff"] = X["home_last10_pct"] - X["away_last10_pct"]
    if "home_home_record_pct" in X.columns and "away_away_record_pct" in X.columns:
        X["venue_record_diff"] = X["home_home_record_pct"] - X["away_away_record_pct"]

    # PBP differentials (V011)
    if "home_three_pt_rate_avg_10" in X.columns and "away_three_pt_rate_avg_10" in X.columns:
        X["three_pt_rate_diff_10"] = X["home_three_pt_rate_avg_10"] - X["away_three_pt_rate_avg_10"]

    # B2B net disadvantage (+1 = home on b2b / away rested, -1 = reverse)
    if "home_is_b2b" in X.columns and "away_is_b2b" in X.columns:
        X["b2b_net_disadvantage"] = X["home_is_b2b"] - X["away_is_b2b"]

    # Travel fatigue (V010) — raw columns already in gtf.*, add compound interactions
    if "is_cross_country" in X.columns and "away_is_b2b" in X.columns:
        X["cross_country_b2b"] = X["is_cross_country"].fillna(0).astype(float) * X["away_is_b2b"].astype(float)
    if "travel_distance_miles" in X.columns and "away_is_b2b" in X.columns:
        X["travel_b2b_fatigue"] = X["travel_distance_miles"].fillna(0) * X["away_is_b2b"].astype(float) / 1000.0
    if "home_altitude_ft" in X.columns and "travel_distance_miles" in X.columns:
        X["altitude_travel_stress"] = (X["home_altitude_ft"].fillna(0) / 5280.0) * X["travel_distance_miles"].fillna(0) / 1000.0
    if "away_total_travel_miles_5" in X.columns and "away_rest_days" in X.columns:
        X["travel_load_per_rest"] = X["away_total_travel_miles_5"].fillna(0) / X["away_rest_days"].clip(lower=1.0)

    # Referee foul over/under bias: positive = crew calls more fouls than these teams
    # typically generate → more FTAs → over bias.  Raw crew_avg_fouls_per_game alone
    # gives no matchup-relative signal.
    if (
        "crew_avg_fouls_per_game" in X.columns
        and "home_fouls_avg_10" in X.columns
        and "away_fouls_avg_10" in X.columns
    ):
        X["ref_foul_ot_signal"] = (
            X["crew_avg_fouls_per_game"] - (X["home_fouls_avg_10"] + X["away_fouls_avg_10"])
        )

    return X


def add_player_prop_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived interaction features for player prop models in-place.

    Assumes categorical columns (season, team_abbr, opponent_abbr) have already
    been one-hot encoded, and b2b/is_home flags are int.

    Returns X with new columns appended.
    """
    # Per-minute efficiency
    if "pts_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["pts_per_min_10"] = X["pts_avg_10"] / X["min_avg_10"].clip(lower=1.0)
    if "reb_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["reb_per_min_10"] = X["reb_avg_10"] / X["min_avg_10"].clip(lower=1.0)
    if "ast_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["ast_per_min_10"] = X["ast_avg_10"] / X["min_avg_10"].clip(lower=1.0)

    # Recent trend: 5-game vs 10-game (momentum)
    if "pts_avg_5" in X.columns and "pts_avg_10" in X.columns:
        X["pts_trend_5v10"] = X["pts_avg_5"] - X["pts_avg_10"]
    if "reb_avg_5" in X.columns and "reb_avg_10" in X.columns:
        X["reb_trend_5v10"] = X["reb_avg_5"] - X["reb_avg_10"]
    if "ast_avg_5" in X.columns and "ast_avg_10" in X.columns:
        X["ast_trend_5v10"] = X["ast_avg_5"] - X["ast_avg_10"]

    # Minutes trend
    if "min_avg_5" in X.columns and "min_avg_10" in X.columns:
        X["min_trend_5v10"] = X["min_avg_5"] - X["min_avg_10"]

    # Coefficient of variation (consistency)
    if "pts_sd_10" in X.columns and "pts_avg_10" in X.columns:
        X["pts_cv_10"] = X["pts_sd_10"] / X["pts_avg_10"].clip(lower=0.5)
    if "reb_sd_10" in X.columns and "reb_avg_10" in X.columns:
        X["reb_cv_10"] = X["reb_sd_10"] / X["reb_avg_10"].clip(lower=0.5)
    if "ast_sd_10" in X.columns and "ast_avg_10" in X.columns:
        X["ast_cv_10"] = X["ast_sd_10"] / X["ast_avg_10"].clip(lower=0.5)

    # Matchup-scaled projections
    if "pts_per_min_10" in X.columns and "game_pace_est_5" in X.columns:
        X["pts_pace_interaction"] = X["pts_per_min_10"] * X["game_pace_est_5"]
    if "reb_per_min_10" in X.columns and "opp_pace_avg_10" in X.columns:
        X["reb_pace_interaction"] = X["reb_per_min_10"] * X["opp_pace_avg_10"]

    # Implied share of team total
    if "pts_avg_10" in X.columns and "team_implied_total" in X.columns:
        X["implied_pts_share"] = X["pts_avg_10"] / X["team_implied_total"].clip(lower=80.0)

    # V007: Expanded player stat interactions
    if "stl_plus_blk_avg_10" in X.columns and "opp_pace_avg_10" in X.columns:
        X["stocks_pace_interaction"] = X["stl_plus_blk_avg_10"] * X["opp_pace_avg_10"] / 100.0
    if "off_reb_avg_10" in X.columns and "def_reb_avg_10" in X.columns:
        total_reb = X["off_reb_avg_10"] + X["def_reb_avg_10"]
        X["off_reb_pct_10"] = X["off_reb_avg_10"] / total_reb.clip(lower=0.5)
    if "tov_avg_10" in X.columns and "opp_stl_avg_10" in X.columns:
        X["tov_vs_opp_stl"] = X["tov_avg_10"] * X["opp_stl_avg_10"] / 10.0
    if "fouls_avg_5" in X.columns and "fouls_avg_10" in X.columns:
        X["fouls_trend_5v10"] = X["fouls_avg_5"] - X["fouls_avg_10"]
    if "plus_minus_avg_5" in X.columns and "plus_minus_avg_10" in X.columns:
        X["pm_trend_5v10"] = X["plus_minus_avg_5"] - X["plus_minus_avg_10"]
    if "fg_pct_avg_10" in X.columns and "opp_blk_avg_10" in X.columns:
        X["fg_pct_vs_opp_blk"] = X["fg_pct_avg_10"] - X["opp_blk_avg_10"] / 10.0

    # V013: Referee foul risk interactions
    if "avg_foul_uplift_crew" in X.columns and "fouls_avg_10" in X.columns:
        X["ref_adjusted_fouls"] = X["fouls_avg_10"] + X["avg_foul_uplift_crew"].fillna(0)
    if "avg_foul_per_36_uplift_crew" in X.columns and "min_avg_10" in X.columns:
        X["ref_foul_min_risk"] = X["avg_foul_per_36_uplift_crew"].fillna(0) * X["min_avg_10"] / 36.0

    # V018: Teammate injury impact
    if "teammate_pts_out" in X.columns and "team_implied_total" in X.columns:
        X["teammate_pts_share_lost"] = X["teammate_pts_out"] / X["team_implied_total"].clip(lower=80.0)
    if "teammate_pts_out" in X.columns and "pts_avg_10" in X.columns:
        # How much of the missing pts load could this player absorb (usage bump proxy)
        X["potential_usage_bump"] = X["teammate_pts_out"] * (X["pts_avg_10"] / X["pts_avg_10"].clip(lower=1.0))

    # V016: Opponent position defense matchup edges
    if "pts_avg_10" in X.columns and "opp_pts_allowed_role_10" in X.columns:
        X["pts_vs_opp_role_edge"] = X["pts_avg_10"] - X["opp_pts_allowed_role_10"]
    if "reb_avg_10" in X.columns and "opp_reb_allowed_role_10" in X.columns:
        X["reb_vs_opp_role_edge"] = X["reb_avg_10"] - X["opp_reb_allowed_role_10"]
    if "ast_avg_10" in X.columns and "opp_ast_allowed_role_10" in X.columns:
        X["ast_vs_opp_role_edge"] = X["ast_avg_10"] - X["opp_ast_allowed_role_10"]

    return X
