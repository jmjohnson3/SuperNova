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

    # Pythagorean expectation (NBA exponent ≈ 14, from Daryl Morey / 82games research).
    # More predictive of future wins than current W-L record because it corrects for
    # close-game luck.  Points-for^14 / (points-for^14 + points-against^14) → [0, 1].
    # Using 5-game (recent form), 10-game (true-talent), and 20-game (season baseline) windows.
    _EXP = 14.0
    for win in (5, 10, 20):
        hf = f"home_pts_for_avg_{win}"
        ha = f"home_pts_against_avg_{win}"
        af = f"away_pts_for_avg_{win}"
        aa = f"away_pts_against_avg_{win}"
        if hf in X.columns and ha in X.columns:
            hf_pow = X[hf].clip(lower=1.0) ** _EXP
            ha_pow = X[ha].clip(lower=1.0) ** _EXP
            X[f"home_pythag_{win}"] = hf_pow / (hf_pow + ha_pow)
        if af in X.columns and aa in X.columns:
            af_pow = X[af].clip(lower=1.0) ** _EXP
            aa_pow = X[aa].clip(lower=1.0) ** _EXP
            X[f"away_pythag_{win}"] = af_pow / (af_pow + aa_pow)
        hp = f"home_pythag_{win}"
        ap = f"away_pythag_{win}"
        if hp in X.columns and ap in X.columns:
            X[f"pythag_diff_{win}"] = X[hp] - X[ap]

    # Pythagorean vs actual win-pct: positive = team is "better than their record"
    # (won fewer close games than their scoring says they should).  Signals regression
    # to mean — a team outperforming pythag is due for a correction.
    if "home_pythag_10" in X.columns and "home_win_pct" in X.columns:
        X["home_pythag_vs_record"] = X["home_pythag_10"] - X["home_win_pct"]
    if "away_pythag_10" in X.columns and "away_win_pct" in X.columns:
        X["away_pythag_vs_record"] = X["away_pythag_10"] - X["away_win_pct"]
    if "home_pythag_vs_record" in X.columns and "away_pythag_vs_record" in X.columns:
        # Positive = home team is more underrated by record than away team
        X["pythag_record_edge"] = X["home_pythag_vs_record"] - X["away_pythag_vs_record"]

    # Pace
    if "home_pace_avg_5" in X.columns and "away_pace_avg_5" in X.columns:
        X["pace_diff_5"] = X["home_pace_avg_5"] - X["away_pace_avg_5"]
    if "home_pace_avg_20" in X.columns and "away_pace_avg_20" in X.columns:
        X["pace_diff_20"] = X["home_pace_avg_20"] - X["away_pace_avg_20"]

    # Scoring volume
    if "home_pts_for_avg_5" in X.columns and "away_pts_for_avg_5" in X.columns:
        X["pts_for_diff_5"] = X["home_pts_for_avg_5"] - X["away_pts_for_avg_5"]

    # Shooting efficiency differentials (multi-window)
    for window in (5, 10, 20):
        for stat in ("efg_pct", "ts_pct", "tov_rate", "fg3_pct", "fg3a_rate"):
            h, a = f"home_{stat}_avg_{window}", f"away_{stat}_avg_{window}"
            if h in X.columns and a in X.columns:
                X[f"{stat}_diff_{window}"] = X[h] - X[a]

    # Net rating and pts diffs for 20-game window (5 and 10 are computed above via net_rating_10)
    for window in (20,):
        hf = f"home_pts_for_avg_{window}"
        ha = f"home_pts_against_avg_{window}"
        af = f"away_pts_for_avg_{window}"
        aa = f"away_pts_against_avg_{window}"
        if hf in X.columns and ha in X.columns:
            X[f"home_net_rating_{window}"] = X[hf] - X[ha]
        if af in X.columns and aa in X.columns:
            X[f"away_net_rating_{window}"] = X[af] - X[aa]
        if f"home_net_rating_{window}" in X.columns and f"away_net_rating_{window}" in X.columns:
            X[f"net_rating_diff_{window}"] = X[f"home_net_rating_{window}"] - X[f"away_net_rating_{window}"]
        if hf in X.columns and af in X.columns:
            X[f"pts_for_diff_{window}"] = X[hf] - X[af]

    # Venue-context diffs: how much does the home team over-perform at home vs. overall?
    # Positive home_venue_premium = team gets meaningful home boost.
    if "home_home_pts_for_avg_10" in X.columns and "home_pts_for_avg_10" in X.columns:
        X["home_venue_premium"] = X["home_home_pts_for_avg_10"] - X["home_pts_for_avg_10"]
    if "away_away_pts_for_avg_10" in X.columns and "away_pts_for_avg_10" in X.columns:
        X["away_road_premium"] = X["away_away_pts_for_avg_10"] - X["away_pts_for_avg_10"]
    if "home_venue_premium" in X.columns and "away_road_premium" in X.columns:
        X["venue_context_diff"] = X["home_venue_premium"] - X["away_road_premium"]

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

    # Line movement magnitude (sharp money signal) — large absolute moves indicate
    # consensus sharp action regardless of direction.
    if "market_line_move_margin" in X.columns:
        X["line_move_abs_spread"] = X["market_line_move_margin"].abs()
    if "market_line_move_total" in X.columns:
        X["line_move_abs_total"] = X["market_line_move_total"].abs()

    # Sharp momentum: line moved AND implied probability agrees with move direction.
    # A game where spread moved toward home AND book implies home likely = strong sharp signal.
    if "market_line_move_margin" in X.columns and "spread_home_implied_prob" in X.columns:
        X["sharp_momentum"] = (
            X["market_line_move_margin"] * (X["spread_home_implied_prob"] - 0.5)
        )

    # Steam signal: large spread move coinciding with rest advantage amplifies the edge.
    if "line_move_abs_spread" in X.columns and "rest_advantage_home" in X.columns:
        X["steam_x_rest"] = (
            X["line_move_abs_spread"].fillna(0) * X["rest_advantage_home"].fillna(0)
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
    if "min_avg_3" in X.columns and "min_avg_5" in X.columns:
        X["min_trend_3v5"] = X["min_avg_3"] - X["min_avg_5"]

    # 3-game hot/cold streaks (raw difference)
    if "pts_avg_3" in X.columns and "pts_avg_10" in X.columns:
        X["pts_trend_3v10"] = X["pts_avg_3"] - X["pts_avg_10"]
    if "reb_avg_3" in X.columns and "reb_avg_10" in X.columns:
        X["reb_trend_3v10"] = X["reb_avg_3"] - X["reb_avg_10"]
    if "ast_avg_3" in X.columns and "ast_avg_10" in X.columns:
        X["ast_trend_3v10"] = X["ast_avg_3"] - X["ast_avg_10"]

    # Hot/cold ratio — scale-invariant form: 1.2 means 20% above baseline regardless of role.
    # Captures the same signal as trend diffs but normalized: a +5-pt bump means more for a
    # 5 ppg player than for a 25 ppg player.
    if "pts_avg_3" in X.columns and "pts_avg_10" in X.columns:
        X["pts_hot_ratio"] = X["pts_avg_3"] / X["pts_avg_10"].clip(lower=1.0)
    if "reb_avg_3" in X.columns and "reb_avg_10" in X.columns:
        X["reb_hot_ratio"] = X["reb_avg_3"] / X["reb_avg_10"].clip(lower=0.5)
    if "ast_avg_3" in X.columns and "ast_avg_10" in X.columns:
        X["ast_hot_ratio"] = X["ast_avg_3"] / X["ast_avg_10"].clip(lower=0.5)
    if "min_avg_3" in X.columns and "min_avg_10" in X.columns:
        X["min_hot_ratio"] = X["min_avg_3"] / X["min_avg_10"].clip(lower=10.0)

    # Trend significance: is the recent move meaningful relative to this player's own variance?
    # Large trend / small SD = a real streak; large trend / large SD = noisy player who swings often.
    # Behaves like a z-score of the 3-game trend vs the 10-game baseline.
    if "pts_trend_3v10" in X.columns and "pts_sd_10" in X.columns:
        X["pts_trend_sig"] = X["pts_trend_3v10"] / X["pts_sd_10"].clip(lower=0.5)
    if "reb_trend_3v10" in X.columns and "reb_sd_10" in X.columns:
        X["reb_trend_sig"] = X["reb_trend_3v10"] / X["reb_sd_10"].clip(lower=0.5)
    if "ast_trend_3v10" in X.columns and "ast_sd_10" in X.columns:
        X["ast_trend_sig"] = X["ast_trend_3v10"] / X["ast_sd_10"].clip(lower=0.5)

    # Hot-player × fast-game: a player on a scoring streak benefits even more in a fast game
    if "pts_hot_ratio" in X.columns and "game_pace_est_5" in X.columns:
        X["pts_hot_x_pace"] = X["pts_hot_ratio"] * X["game_pace_est_5"] / 100.0
    if "ast_hot_ratio" in X.columns and "game_pace_est_5" in X.columns:
        X["ast_hot_x_pace"] = X["ast_hot_ratio"] * X["game_pace_est_5"] / 100.0

    # Usage momentum
    if "usage_proxy_avg_5" in X.columns and "usage_proxy_avg_10" in X.columns:
        X["usage_trend_5v10"] = X["usage_proxy_avg_5"] - X["usage_proxy_avg_10"]

    # ---------------------------------------------------------------------------
    # EMA rolling approximations via lag-bin decomposition.
    # The SQL provides overlapping windows (avg_3, avg_5, avg_10).  We recover
    # the non-overlapping bin averages and apply EMA weights (λ=0.85):
    #   games 1-3  (most recent):  sum of λ^0..λ^2 = 2.5725,  per game weight ≈ 0.8575
    #   games 4-5:                 sum of λ^3..λ^4 = 1.1361,  per game weight ≈ 0.5681
    #   games 6-10 (oldest):       sum of λ^5..λ^9 = 1.6455,  per game weight ≈ 0.3291
    #   total normaliser = 5.354
    # EMA_10 ≈ (avg_3 × 2.5725 + avg_4_5 × 1.1361 + avg_6_10 × 1.6455) / 5.354
    # ---------------------------------------------------------------------------
    _EMA_NORM = 5.354
    for stat in ("pts", "reb", "ast", "min"):
        a3  = f"{stat}_avg_3"
        a5  = f"{stat}_avg_5"
        a10 = f"{stat}_avg_10"
        if a3 in X.columns and a5 in X.columns and a10 in X.columns:
            # Bin averages (can be NaN when window is partially filled)
            bin_4_5  = (X[a5]  * 5 - X[a3]  * 3) / 2.0
            bin_6_10 = (X[a10] * 10 - X[a5] * 5) / 5.0
            X[f"{stat}_ema_10"] = (
                X[a3] * 2.5725 + bin_4_5 * 1.1361 + bin_6_10 * 1.6455
            ) / _EMA_NORM
            # EMA trend vs simple average (captures recency premium/discount)
            X[f"{stat}_ema_vs_avg"] = X[f"{stat}_ema_10"] - X[a10]

    # Coefficient of variation (consistency)
    if "pts_sd_10" in X.columns and "pts_avg_10" in X.columns:
        X["pts_cv_10"] = X["pts_sd_10"] / X["pts_avg_10"].clip(lower=0.5)
    if "reb_sd_10" in X.columns and "reb_avg_10" in X.columns:
        X["reb_cv_10"] = X["reb_sd_10"] / X["reb_avg_10"].clip(lower=0.5)
    if "ast_sd_10" in X.columns and "ast_avg_10" in X.columns:
        X["ast_cv_10"] = X["ast_sd_10"] / X["ast_avg_10"].clip(lower=0.5)
    # Minute share volatility: high min_cv_10 = unpredictable role, harder to forecast
    if "min_sd_10" in X.columns and "min_avg_10" in X.columns:
        X["min_cv_10"] = X["min_sd_10"] / X["min_avg_10"].clip(lower=5.0)

    # Matchup-scaled projections (pace × per-min efficiency)
    if "pts_per_min_10" in X.columns and "game_pace_est_5" in X.columns:
        X["pts_pace_interaction"] = X["pts_per_min_10"] * X["game_pace_est_5"]
    if "reb_per_min_10" in X.columns and "opp_pace_avg_10" in X.columns:
        X["reb_pace_interaction"] = X["reb_per_min_10"] * X["opp_pace_avg_10"]
    if "ast_per_min_10" in X.columns and "game_pace_est_5" in X.columns:
        # Assists rise with pace — more possessions = more chances to dish
        X["ast_pace_interaction"] = X["ast_per_min_10"] * X["game_pace_est_5"]

    # Pace mismatch: derive player's team pace from game average and opponent pace.
    # game_pace_est_5 ≈ (team_pace + opp_pace) / 2
    # → team_pace_derived = 2 × game_pace_est_5 - opp_pace_avg_5
    # Positive pace_mismatch = our team is faster than the opponent's preferred tempo
    if "game_pace_est_5" in X.columns and "opp_pace_avg_5" in X.columns:
        X["team_pace_derived"] = (2.0 * X["game_pace_est_5"] - X["opp_pace_avg_5"]).clip(lower=80.0)
        X["pace_mismatch"] = X["team_pace_derived"] - X["opp_pace_avg_5"]

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
    if "teammate_pts_out" in X.columns and "pts_avg_10" in X.columns and "team_implied_total" in X.columns:
        # How much of the missing pts load this player absorbs, weighted by their own scoring share.
        # Higher-usage players get proportionally more of the vacated load.
        _scoring_share = X["pts_avg_10"] / X["team_implied_total"].clip(lower=80.0)
        X["potential_usage_bump"] = X["teammate_pts_out"] * _scoring_share
    if "potential_usage_bump" in X.columns and "pts_trend_5v10" in X.columns:
        # Compound signal: player is trending UP *and* a teammate is out → usage bump likely real
        X["injury_boost_confirmed"] = X["potential_usage_bump"] * X["pts_trend_5v10"].clip(lower=0)

    # V016: Opponent position defense matchup edges
    if "pts_avg_10" in X.columns and "opp_pts_allowed_role_10" in X.columns:
        X["pts_vs_opp_role_edge"] = X["pts_avg_10"] - X["opp_pts_allowed_role_10"]
    if "reb_avg_10" in X.columns and "opp_reb_allowed_role_10" in X.columns:
        X["reb_vs_opp_role_edge"] = X["reb_avg_10"] - X["opp_reb_allowed_role_10"]
    if "ast_avg_10" in X.columns and "opp_ast_allowed_role_10" in X.columns:
        X["ast_vs_opp_role_edge"] = X["ast_avg_10"] - X["opp_ast_allowed_role_10"]

    # ---------------------------------------------------------------------------
    # Home/away split features.
    # pts_avg_10_home / pts_avg_10_away come from SQL conditional windows.
    # ---------------------------------------------------------------------------
    # Raw home advantage (positive = player scores more at home)
    if "pts_avg_10_home" in X.columns and "pts_avg_10_away" in X.columns:
        X["pts_home_adv"] = X["pts_avg_10_home"] - X["pts_avg_10_away"]
    if "reb_avg_10_home" in X.columns and "reb_avg_10_away" in X.columns:
        X["reb_home_adv"] = X["reb_avg_10_home"] - X["reb_avg_10_away"]

    # Venue-specific prediction: pick the split matching tonight's venue
    if "pts_avg_10_home" in X.columns and "pts_avg_10_away" in X.columns and "is_home" in X.columns:
        X["pts_venue_split"] = X["pts_avg_10_home"].where(
            X["is_home"].astype(bool), X["pts_avg_10_away"]
        )
    if "reb_avg_10_home" in X.columns and "reb_avg_10_away" in X.columns and "is_home" in X.columns:
        X["reb_venue_split"] = X["reb_avg_10_home"].where(
            X["is_home"].astype(bool), X["reb_avg_10_away"]
        )
    if "min_avg_10_home" in X.columns and "min_avg_10_away" in X.columns and "is_home" in X.columns:
        X["min_venue_split"] = X["min_avg_10_home"].where(
            X["is_home"].astype(bool), X["min_avg_10_away"]
        )

    # ---------------------------------------------------------------------------
    # Synthetic team implied total — fills NULLs when market odds are unavailable.
    # Formula: team_off_rtg / (team_off_rtg + opp_def_rtg) × league_avg_total
    # League avg ≈ 222 points per game (2024-25 season).
    # ---------------------------------------------------------------------------
    if "team_off_rtg_10" in X.columns and "opp_def_rtg_10" in X.columns:
        _off = X["team_off_rtg_10"].clip(lower=90.0)
        _def = X["opp_def_rtg_10"].clip(lower=90.0)
        X["team_implied_total_synth"] = (_off / (_off + _def) * 222.0).clip(80.0, 140.0)
        if "team_implied_total" in X.columns:
            X["team_implied_total_filled"] = X["team_implied_total"].fillna(
                X["team_implied_total_synth"]
            )

    # V022: DraftKings book line as a prior feature.
    # The raw book lines (prev_book_line_*) are passed as-is to the model.
    # These interaction features encode how the book line compares to the player's
    # recent production — positive = book expects above-average output.
    if "prev_book_line_pts" in X.columns and "pts_avg_10" in X.columns:
        X["book_line_vs_pts_avg"] = X["prev_book_line_pts"] - X["pts_avg_10"]
    if "prev_book_line_reb" in X.columns and "reb_avg_10" in X.columns:
        X["book_line_vs_reb_avg"] = X["prev_book_line_reb"] - X["reb_avg_10"]
    if "prev_book_line_ast" in X.columns and "ast_avg_10" in X.columns:
        X["book_line_vs_ast_avg"] = X["prev_book_line_ast"] - X["ast_avg_10"]

    # Book line as a share of team implied total (role / usage signal)
    if "prev_book_line_pts" in X.columns and "team_implied_total" in X.columns:
        X["book_pts_share"] = X["prev_book_line_pts"] / X["team_implied_total"].clip(lower=80.0)

    # Book line trend: how much has the book shifted relative to the player's 3-game form
    if "prev_book_line_pts" in X.columns and "pts_avg_3" in X.columns:
        X["book_line_vs_pts_hot"] = X["prev_book_line_pts"] - X["pts_avg_3"]

    # V023 × V024: shot-type matchup differentials
    # Positive = player shoots this type more than defense typically allows → advantage
    _shot_matchups = [
        ("paint_shot_rate_avg_10",      "opp_paint_allowed_avg_10"),
        ("driving_shot_rate_avg_10",    "opp_driving_allowed_avg_10"),
        ("catch_and_shoot_rate_avg_10", "opp_catch_shoot_allowed_avg_10"),
    ]
    for player_col, opp_col in _shot_matchups:
        if player_col in X.columns and opp_col in X.columns:
            X[f"matchup_{player_col[:5]}_{opp_col[:9]}"] = X[player_col] - X[opp_col]

    return X
