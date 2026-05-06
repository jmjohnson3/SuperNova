"""
Shared feature engineering functions for MLB game models.

add_game_derived_features — used by train_game_models.make_xy_raw()
                            and predict_today._prep_features()

Adding a new derived feature?  Edit the function here ONCE.  Both training
and inference pick it up automatically.  No need to touch two files.
"""

import numpy as np
import pandas as pd


def add_game_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived interaction features for MLB game models.

    Assumes X already has:
      - raw rolling-window stat columns (NaNs are fine — guards check membership)
      - categorical columns (season, team abbrs) already one-hot encoded
      - b2b / rest flags encoded as int (0/1)

    Returns X with new columns appended.
    """
    X = X.copy()

    # ── Baseball Pythagorean expectation ─────────────────────────────────────
    # Standard formula: win_pct_pythag = RF^1.82 / (RF^1.82 + RA^1.82)
    # Uses each team's own runs-scored (RF) and runs-allowed (RA) rolling averages.
    _EXP = 1.82

    def _pythag(rf: pd.Series, ra: pd.Series) -> pd.Series:
        rf_pow = rf.clip(lower=0.01) ** _EXP
        ra_pow = ra.clip(lower=0.01) ** _EXP
        return rf_pow / (rf_pow + ra_pow)

    _pythag_cols = (
        "home_runs_avg_10", "home_runs_allowed_avg_10",
        "away_runs_avg_10", "away_runs_allowed_avg_10",
    )
    if all(c in X.columns for c in _pythag_cols):
        home_rf = X["home_runs_avg_10"]
        home_ra = X["home_runs_allowed_avg_10"]
        away_rf = X["away_runs_avg_10"]
        away_ra = X["away_runs_allowed_avg_10"]
        X["home_pythag"] = _pythag(home_rf, home_ra)
        X["away_pythag"] = _pythag(away_rf, away_ra)
        X["pythag_diff"] = X["home_pythag"] - X["away_pythag"]

    # Pythagorean vs actual win-pct: positive = team is "better than their record".
    # Signals regression to mean — a team outperforming pythag is due for a correction.
    if "home_pythag" in X.columns and "home_win_pct" in X.columns:
        X["home_pythag_vs_record"] = X["home_pythag"] - X["home_win_pct"]
    if "away_pythag" in X.columns and "away_win_pct" in X.columns:
        X["away_pythag_vs_record"] = X["away_pythag"] - X["away_win_pct"]
    if "home_pythag_vs_record" in X.columns and "away_pythag_vs_record" in X.columns:
        X["pythag_record_edge"] = X["home_pythag_vs_record"] - X["away_pythag_vs_record"]

    # ── SP matchup edges ──────────────────────────────────────────────────────
    # For rate stats where lower is better (ERA, WHIP, FIP): away_col - home_col > 0
    # means home SP has an advantage.
    for col in ("era_5", "whip_5", "fip_5"):
        home_col = f"home_sp_{col}"
        away_col = f"away_sp_{col}"
        if home_col in X.columns and away_col in X.columns:
            X[f"sp_{col}_diff"] = X[away_col] - X[home_col]

    # For K% — higher is better for the pitcher:
    for col in ("k_pct_5",):
        home_col = f"home_sp_{col}"
        away_col = f"away_sp_{col}"
        if home_col in X.columns and away_col in X.columns:
            X[f"sp_{col}_diff"] = X[home_col] - X[away_col]

    # Composite SP quality score: ERA edge + FIP edge (normalized)
    if "sp_era_5_diff" in X.columns and "sp_fip_5_diff" in X.columns:
        X["sp_composite_edge"] = (X["sp_era_5_diff"] + X["sp_fip_5_diff"]) / 2.0

    # ── Team batting differentials ────────────────────────────────────────────
    for col in ("runs_avg_10", "hr_avg_10", "avg_avg_10", "iso_avg_10",
                "runs_avg_5", "hr_avg_5", "avg_avg_5"):
        hc = f"home_{col}"
        ac = f"away_{col}"
        if hc in X.columns and ac in X.columns:
            X[f"batting_{col}_diff"] = X[hc] - X[ac]

    # ── Team pitching differentials ───────────────────────────────────────────
    # ERA / WHIP: lower is better → away minus home = home edge
    for col in ("era_10", "era_5", "whip_10", "whip_5", "fip_10"):
        hc = f"home_{col}"
        ac = f"away_{col}"
        if hc in X.columns and ac in X.columns:
            X[f"team_{col}_diff"] = X[ac] - X[hc]

    # ── Run differential per game edge ───────────────────────────────────────
    if "home_run_diff_per_game" in X.columns and "away_run_diff_per_game" in X.columns:
        X["run_diff_per_game_edge"] = X["home_run_diff_per_game"] - X["away_run_diff_per_game"]

    # ── Win% differential ────────────────────────────────────────────────────
    if "home_win_pct" in X.columns and "away_win_pct" in X.columns:
        X["win_pct_diff"] = X["home_win_pct"] - X["away_win_pct"]

    # ── Park factor passthrough ───────────────────────────────────────────────
    if "run_factor" in X.columns:
        X["park_run_factor"] = X["run_factor"]

    # ── Rest advantage ────────────────────────────────────────────────────────
    if "home_rest_days" in X.columns and "away_rest_days" in X.columns:
        X["rest_diff"] = X["home_rest_days"] - X["away_rest_days"]
        X["home_is_b2b"] = (X["home_rest_days"] <= 1).astype(int)
        X["away_is_b2b"] = (X["away_rest_days"] <= 1).astype(int)
        X["b2b_asymmetry"] = X["home_is_b2b"] - X["away_is_b2b"]

    # ── Market line passthroughs ──────────────────────────────────────────────
    if "total_line" in X.columns:
        X["market_total"] = X["total_line"]
    if "run_line_home" in X.columns:
        X["market_run_line"] = X["run_line_home"]

    # ── Market total deviation from league average ────────────────────────────
    # More signal than raw total_line (centers on ~9.0 runs/game)
    if "total_line" in X.columns:
        X["market_total_vs_avg"] = pd.to_numeric(X["total_line"], errors="coerce") - 9.0

    # ── Line movement feature engineering ─────────────────────────────────────
    # Raw move columns have ~0 importance; abs(), direction, and magnitude flags
    # are far more informative (sharp vs stale signal).
    if "total_line_move" in X.columns:
        _tm = pd.to_numeric(X["total_line_move"], errors="coerce").fillna(0.0)
        X["abs_total_line_move"]       = _tm.abs()
        X["total_line_move_direction"] = np.sign(_tm).astype(int)
        X["total_line_move_large"]     = (_tm.abs() >= 0.5).astype(int)
    if "run_line_move" in X.columns:
        _rm = pd.to_numeric(X["run_line_move"], errors="coerce").fillna(0.0)
        X["abs_run_line_move"]         = _rm.abs()
        X["run_line_move_direction"]   = np.sign(_rm).astype(int)
        X["run_line_move_large"]       = (_rm.abs() >= 0.05).astype(int)

    # ── OPS (OBP + SLG) differential ─────────────────────────────────────────
    # OPS is the most common single-number offensive quality proxy.
    for _w in ("5", "10"):
        for _side in ("home", "away"):
            _obp = f"{_side}_obp_avg_{_w}"
            _slg = f"{_side}_slg_avg_{_w}"
            if _obp in X.columns and _slg in X.columns:
                X[f"{_side}_ops_{_w}"] = X[_obp] + X[_slg]
        h, a = f"home_ops_{_w}", f"away_ops_{_w}"
        if h in X.columns and a in X.columns:
            X[f"ops_diff_{_w}"] = X[h] - X[a]

    # ── Runs scoring trend (hot / cold offense arc) ───────────────────────────
    # Positive = team is outscoring its season baseline over last 5 games.
    for _side in ("home", "away"):
        _r5, _r20 = f"{_side}_runs_avg_5", f"{_side}_runs_avg_20"
        if _r5 in X.columns and _r20 in X.columns:
            X[f"{_side}_runs_trend_5v20"] = X[_r5] - X[_r20]
    if "home_runs_trend_5v20" in X.columns and "away_runs_trend_5v20" in X.columns:
        X["runs_trend_diff"] = X["home_runs_trend_5v20"] - X["away_runs_trend_5v20"]

    # ── Team ERA trend (improving = negative value) ───────────────────────────
    for _side in ("home", "away"):
        _e5, _e10 = f"{_side}_era_5", f"{_side}_era_10"
        if _e5 in X.columns and _e10 in X.columns:
            X[f"{_side}_era_trend_5v10"] = X[_e5] - X[_e10]
    # Positive diff = away staff worsening relative to home (home edge)
    if "home_era_trend_5v10" in X.columns and "away_era_trend_5v10" in X.columns:
        X["era_trend_diff"] = X["away_era_trend_5v10"] - X["home_era_trend_5v10"]

    # ── Opponent-quality adjusted SP ERA ─────────────────────────────────────
    # SP facing above-avg offense gets credit: adj_era = raw_era - opp_quality * 0.25
    _LEAGUE_AVG_R = 4.5
    for _sp, _opp_r in [("home", "away"), ("away", "home")]:
        _era = f"{_sp}_sp_era_5"
        _opp = f"{_opp_r}_runs_avg_5"
        if _era in X.columns and _opp in X.columns:
            _era_v = pd.to_numeric(X[_era], errors="coerce")
            _opp_v = pd.to_numeric(X[_opp], errors="coerce").fillna(_LEAGUE_AVG_R)
            X[f"{_sp}_sp_era_opp_adj_5"] = _era_v - (_opp_v - _LEAGUE_AVG_R) * 0.25
    if "home_sp_era_opp_adj_5" in X.columns and "away_sp_era_opp_adj_5" in X.columns:
        X["sp_era_opp_adj_diff"] = X["home_sp_era_opp_adj_5"] - X["away_sp_era_opp_adj_5"]

    # ── SP K-rate trend (5 vs 10 starts) ─────────────────────────────────────
    # Positive = SP on upswing in Ks; captures form/momentum vs raw K%.
    for _side in ("home", "away"):
        _k5  = f"{_side}_sp_k_pct_5"
        _k10 = f"{_side}_sp_k_pct_10"
        if _k5 in X.columns and _k10 in X.columns:
            X[f"{_side}_sp_k_pct_trend"] = (
                pd.to_numeric(X[_k5],  errors="coerce")
                - pd.to_numeric(X[_k10], errors="coerce")
            )
    if "home_sp_k_pct_trend" in X.columns and "away_sp_k_pct_trend" in X.columns:
        X["sp_k_pct_trend_diff"] = X["home_sp_k_pct_trend"] - X["away_sp_k_pct_trend"]

    # ── SP K/BB ratio (dominance vs command) ─────────────────────────────────
    for _side in ("home", "away"):
        _k, _bb = f"{_side}_sp_k_pct_5", f"{_side}_sp_bb_pct_5"
        if _k in X.columns and _bb in X.columns:
            X[f"{_side}_sp_k_bb_ratio_5"] = X[_k] / X[_bb].clip(lower=0.01)
    if "home_sp_k_bb_ratio_5" in X.columns and "away_sp_k_bb_ratio_5" in X.columns:
        X["sp_k_bb_ratio_diff"] = X["home_sp_k_bb_ratio_5"] - X["away_sp_k_bb_ratio_5"]

    # ── Team pitching K/9 ÷ BB/9 (staff control quality) ─────────────────────
    for _side in ("home", "away"):
        _k9, _bb9 = f"{_side}_k9_5", f"{_side}_bb9_5"
        if _k9 in X.columns and _bb9 in X.columns:
            X[f"{_side}_team_k_bb_ratio_5"] = X[_k9] / X[_bb9].clip(lower=0.1)
    if "home_team_k_bb_ratio_5" in X.columns and "away_team_k_bb_ratio_5" in X.columns:
        X["team_k_bb_ratio_diff"] = X["home_team_k_bb_ratio_5"] - X["away_team_k_bb_ratio_5"]

    # ── Bullpen exposure (fraction of game SP doesn't cover) ─────────────────
    # 0 = SP goes all 9 (no bullpen needed), 1 = SP gets no outs (all bullpen).
    for _side in ("home", "away"):
        _ip = f"{_side}_sp_ip_avg_5"
        if _ip in X.columns:
            X[f"{_side}_bullpen_exposure"] = (1.0 - X[_ip].clip(0, 9) / 9.0).clip(lower=0.0)

    # ── SP last-start workload (IP-based fatigue proxy; ~16.5 pitches/inning) ─
    for _side in ("home", "away"):
        _lip = f"{_side}_sp_last_ip"
        if _lip in X.columns:
            _ip_v = pd.to_numeric(X[_lip], errors="coerce").fillna(0.0)
            X[f"{_side}_sp_last_high_workload"] = (_ip_v > 6.0).astype(int)
            X[f"{_side}_sp_pitches_est"] = _ip_v * 16.5
    if "home_sp_pitches_est" in X.columns and "away_sp_pitches_est" in X.columns:
        X["sp_pitches_est_diff"] = X["home_sp_pitches_est"] - X["away_sp_pitches_est"]

    # ── SP depth × BP ERA (short SP + bad bullpen = runs) ────────────────────
    for _side in ("home", "away"):
        _exp, _bpera = f"{_side}_bullpen_exposure", f"{_side}_bp_era_5"
        if _exp in X.columns and _bpera in X.columns:
            X[f"{_side}_pen_exposure_x_era"] = X[_exp] * X[_bpera]
    # Positive = away team is more exposed (home pen advantage)
    if "home_pen_exposure_x_era" in X.columns and "away_pen_exposure_x_era" in X.columns:
        X["pen_exposure_era_diff"] = X["away_pen_exposure_x_era"] - X["home_pen_exposure_x_era"]

    # ── Bullpen ERA differential ──────────────────────────────────────────────
    # Positive = away BP worse than home (late-inning advantage for home).
    for _w in ("5", "10"):
        h, a = f"home_bp_era_{_w}", f"away_bp_era_{_w}"
        if h in X.columns and a in X.columns:
            X[f"bp_era_diff_{_w}"] = X[a] - X[h]

    # ── Market vig-implied win probability ────────────────────────────────────
    # Convert American odds → raw implied prob → de-vig (normalize so home+away = 1).
    def _american_to_prob(odds: pd.Series) -> pd.Series:
        odds_n = pd.to_numeric(odds, errors="coerce")
        pos = odds_n > 0
        neg = odds_n < 0
        p = pd.Series(np.nan, index=odds_n.index)
        p[pos] = 100.0 / (odds_n[pos] + 100.0)
        p[neg] = odds_n[neg].abs() / (odds_n[neg].abs() + 100.0)
        return p

    if "run_line_home_price" in X.columns and "run_line_away_price" in X.columns:
        _raw_h = _american_to_prob(X["run_line_home_price"])
        _raw_a = _american_to_prob(X["run_line_away_price"])
        _total = (_raw_h + _raw_a).replace(0, np.nan)
        X["market_home_win_prob"] = _raw_h / _total

    # ── Vig direction for totals ──────────────────────────────────────────────
    # Positive = over is juiced (sharp money on over); negative = under juiced.
    if "over_price" in X.columns and "under_price" in X.columns:
        _raw_ov = _american_to_prob(X["over_price"])
        _raw_un = _american_to_prob(X["under_price"])
        _tot_vig = (_raw_ov + _raw_un).replace(0, np.nan)
        X["total_vig_direction"] = _raw_ov - _raw_un   # positive = over juiced
        X["over_implied_prob"]   = _raw_ov / _tot_vig  # de-vigored P(over)

    # ── Park × offense interaction ────────────────────────────────────────────
    # Amplifies offensive projections by park run-scoring environment.
    _park = X.get("park_run_factor") if "park_run_factor" in X.columns else X.get("run_factor")
    if _park is not None:
        for _side in ("home", "away"):
            _r10 = f"{_side}_runs_avg_10"
            if _r10 in X.columns:
                X[f"{_side}_park_adj_runs_10"] = X[_r10] * _park
        if "home_park_adj_runs_10" in X.columns and "away_park_adj_runs_10" in X.columns:
            X["park_adj_runs_diff"] = X["home_park_adj_runs_10"] - X["away_park_adj_runs_10"]

    # ── Park HR factor × team HR rate ─────────────────────────────────────────
    # More targeted than park_run_factor × runs (captures HR-park amplification).
    # Raw park_hr_factor has ~0 importance alone; this interaction has real signal.
    if "park_hr_factor" in X.columns:
        _phr = pd.to_numeric(X["park_hr_factor"], errors="coerce").fillna(1.0)
        for _side in ("home", "away"):
            _hr10 = f"{_side}_hr_avg_10"
            if _hr10 in X.columns:
                X[f"{_side}_team_hr_x_park"] = X[_hr10] * _phr
        if "home_team_hr_x_park" in X.columns and "away_team_hr_x_park" in X.columns:
            X["team_hr_park_diff"] = X["home_team_hr_x_park"] - X["away_team_hr_x_park"]

    # ── Park-adjusted SP ERA ──────────────────────────────────────────────────
    # Normalize ERA by park run factor to remove ballpark bias.
    _park_f = pd.to_numeric(
        X["park_run_factor"] if "park_run_factor" in X.columns else pd.Series(1.0, index=X.index),
        errors="coerce",
    ).fillna(1.0).clip(lower=0.7, upper=1.4)
    for _side in ("home", "away"):
        _era = f"{_side}_sp_era_5"
        if _era in X.columns:
            X[f"{_side}_sp_era_park_adj_5"] = pd.to_numeric(X[_era], errors="coerce") / _park_f
    if "home_sp_era_park_adj_5" in X.columns and "away_sp_era_park_adj_5" in X.columns:
        X["sp_era_park_adj_diff"] = X["home_sp_era_park_adj_5"] - X["away_sp_era_park_adj_5"]

    # ── Pythagorean × opposing SP quality ────────────────────────────────────
    # Strong team facing weak SP → amplified win probability.
    # High away_sp_fip_5 = weak away pitcher = home offense benefits more.
    if "home_pythag" in X.columns and "away_sp_fip_5" in X.columns:
        X["home_pythag_x_opp_sp_fip"] = X["home_pythag"] * X["away_sp_fip_5"]
    if "away_pythag" in X.columns and "home_sp_fip_5" in X.columns:
        X["away_pythag_x_opp_sp_fip"] = X["away_pythag"] * X["home_sp_fip_5"]
    if "home_pythag_x_opp_sp_fip" in X.columns and "away_pythag_x_opp_sp_fip" in X.columns:
        X["pythag_sp_quality_diff"] = X["home_pythag_x_opp_sp_fip"] - X["away_pythag_x_opp_sp_fip"]

    # ── SP days rest differential (Group B) ──────────────────────────────────
    # Positive = home SP has had more rest (advantage for home).
    if "home_sp_days_rest" in X.columns and "away_sp_days_rest" in X.columns:
        X["sp_days_rest_diff"] = (
            pd.to_numeric(X["home_sp_days_rest"], errors="coerce")
            - pd.to_numeric(X["away_sp_days_rest"], errors="coerce")
        )

    # Short-rest asymmetry: +1 = away SP on short rest, -1 = home SP on short rest
    if "home_sp_is_short_rest" in X.columns and "away_sp_is_short_rest" in X.columns:
        X["sp_short_rest_asymmetry"] = (
            pd.to_numeric(X["away_sp_is_short_rest"], errors="coerce").fillna(0)
            - pd.to_numeric(X["home_sp_is_short_rest"], errors="coerce").fillna(0)
        )

    # ── SP home/away ERA splits (Group B) ────────────────────────────────────
    # era_home - era_away: negative = pitcher is better at home (common for home SPs).
    # The "split edge" = away SP's home disadvantage minus home SP's home advantage.
    for _side in ("home", "away"):
        _eh = f"{_side}_sp_era_home_10"
        _ea = f"{_side}_sp_era_away_10"
        if _eh in X.columns and _ea in X.columns:
            X[f"{_side}_sp_home_away_era_split"] = (
                pd.to_numeric(X[_eh], errors="coerce")
                - pd.to_numeric(X[_ea], errors="coerce")
            )

    # Positive = away SP performs relatively worse away from home (home has edge)
    if "home_sp_home_away_era_split" in X.columns and "away_sp_home_away_era_split" in X.columns:
        X["sp_home_away_split_edge"] = (
            X["away_sp_home_away_era_split"] - X["home_sp_home_away_era_split"]
        )

    # ── Rolling win% differential — recent form (Group B) ────────────────────
    # Captures hot/cold streaks over the last 5 and 10 games.
    if "home_win_pct_last_5" in X.columns and "away_win_pct_last_5" in X.columns:
        X["win_pct_last_5_diff"] = (
            pd.to_numeric(X["home_win_pct_last_5"], errors="coerce")
            - pd.to_numeric(X["away_win_pct_last_5"], errors="coerce")
        )
    if "home_win_pct_last_10" in X.columns and "away_win_pct_last_10" in X.columns:
        X["win_pct_last_10_diff"] = (
            pd.to_numeric(X["home_win_pct_last_10"], errors="coerce")
            - pd.to_numeric(X["away_win_pct_last_10"], errors="coerce")
        )

    # Rolling run diff per game (last 5): positive = home team has been outscoring opponents
    if "home_run_diff_avg_last_5" in X.columns and "away_run_diff_avg_last_5" in X.columns:
        X["run_diff_avg_last_5_edge"] = (
            pd.to_numeric(X["home_run_diff_avg_last_5"], errors="coerce")
            - pd.to_numeric(X["away_run_diff_avg_last_5"], errors="coerce")
        )

    # Recent form vs season form: is the home team trending up relative to season baseline?
    if "home_win_pct_last_5" in X.columns and "home_win_pct" in X.columns:
        X["home_form_vs_season"] = (
            pd.to_numeric(X["home_win_pct_last_5"], errors="coerce")
            - pd.to_numeric(X["home_win_pct"], errors="coerce")
        )
    if "away_win_pct_last_5" in X.columns and "away_win_pct" in X.columns:
        X["away_form_vs_season"] = (
            pd.to_numeric(X["away_win_pct_last_5"], errors="coerce")
            - pd.to_numeric(X["away_win_pct"], errors="coerce")
        )
    if "home_form_vs_season" in X.columns and "away_form_vs_season" in X.columns:
        X["form_momentum_diff"] = X["home_form_vs_season"] - X["away_form_vs_season"]

    # ── H2H season record edge (Group B) ────────────────────────────────────
    # Deviation from 50/50 in head-to-head games this season.
    # Only meaningful after a few series; NULL-safe via fillna.
    if "h2h_home_win_pct_ytd" in X.columns:
        X["h2h_edge"] = pd.to_numeric(X["h2h_home_win_pct_ytd"], errors="coerce") - 0.5

    # ── H2H familiarity (Improvement 1) ─────────────────────────────────────────
    # Scaled 0-1 measure of how many times these teams have met this season
    if "h2h_games_ytd" in X.columns:
        X["h2h_familiarity"] = X["h2h_games_ytd"].clip(upper=19) / 19.0

    # ── Bullpen fatigue (Group D) ─────────────────────────────────────────────
    # SP short outing asymmetry: positive = home SP was knocked out early last game
    # (home bullpen overused; typically -1 for away advantage, +1 for home disadvantage)
    if "home_sp_short_last" in X.columns and "away_sp_short_last" in X.columns:
        X["sp_short_asymmetry"] = (
            pd.to_numeric(X["home_sp_short_last"], errors="coerce").fillna(0)
            - pd.to_numeric(X["away_sp_short_last"], errors="coerce").fillna(0)
        )

    # Pen depletion index: 7-day IP × (1 + short-start flag)
    # Short start means bullpen worked extra innings → compounding fatigue
    for _side in ("home", "away"):
        _ip7   = f"{_side}_bullpen_ip_last_7"
        _short = f"{_side}_sp_short_last"
        if _ip7 in X.columns:
            _ip  = pd.to_numeric(X[_ip7], errors="coerce").fillna(0.0)
            _sh  = pd.to_numeric(X[_short], errors="coerce").fillna(0.0) if _short in X.columns else 0.0
            X[f"{_side}_pen_depletion"] = _ip * (1.0 + _sh)

    # Depletion edge: positive = away team's bullpen more depleted (home advantage)
    if "home_pen_depletion" in X.columns and "away_pen_depletion" in X.columns:
        X["pen_depletion_edge"] = X["away_pen_depletion"] - X["home_pen_depletion"]

    # ── 3a: Quality-Weighted Bullpen Depletion ────────────────────────────────
    # ERA-weighted depletion: a depleted pen is worse if it also has high ERA.
    # Falls back to 5-start ERA if the new 3-day ERA column is not present.
    for _side in ("home", "away"):
        _dep  = f"{_side}_pen_depletion"
        _era3 = f"{_side}_bp_avg_era_last_3d"
        _era5 = f"{_side}_bp_era_5"
        if _dep in X.columns:
            _d = pd.to_numeric(X[_dep], errors="coerce").fillna(0.0)
            _e_col = _era3 if _era3 in X.columns else _era5
            _e = pd.to_numeric(X[_e_col], errors="coerce").fillna(4.0) if _e_col in X.columns else pd.Series(4.0, index=X.index)
            X[f"{_side}_pen_qual_depletion"] = _d * _e / 4.0
    if "home_pen_qual_depletion" in X.columns and "away_pen_qual_depletion" in X.columns:
        X["pen_qual_depletion_edge"] = X["away_pen_qual_depletion"] - X["home_pen_qual_depletion"]

    # ── 3b: ERA-Quality Differential of Arms Used Last 3 Days ─────────────────
    # Positive = home BP has higher average ERA among arms used recently (disadvantage)
    if "home_bp_avg_era_last_3d" in X.columns and "away_bp_avg_era_last_3d" in X.columns:
        _hera3 = pd.to_numeric(X["home_bp_avg_era_last_3d"], errors="coerce").fillna(4.0)
        _aera3 = pd.to_numeric(X["away_bp_avg_era_last_3d"], errors="coerce").fillna(4.0)
        X["bp_avg_era_last_3d_diff"] = _hera3 - _aera3

    # Quality × fatigue interaction: fatigued AND leaky pen = extra trouble
    # bp_era_7d (higher = worse pen); multiply by 7-day IP (more usage = more depleted)
    for _side in ("home", "away"):
        _era7 = f"{_side}_bp_era_7d"
        _ip7  = f"{_side}_bullpen_ip_last_7"
        if _era7 in X.columns and _ip7 in X.columns:
            X[f"{_side}_bp_era7_x_ip7"] = (
                pd.to_numeric(X[_era7], errors="coerce").fillna(4.5)   # league-avg if NULL
                * pd.to_numeric(X[_ip7], errors="coerce").fillna(0.0)
            )

    # ── Bullpen fatigue × rest interaction (Improvement 3) ──────────────────────
    # Bullpen IP per day of rest (higher = more stressed)
    for _side in ("home", "away"):
        _bp7 = f"{_side}_bullpen_ip_last_7"
        _rest = f"{_side}_rest_days"
        if _bp7 in X.columns and _rest in X.columns:
            X[f"{_side}_bp_usage_per_rest"] = X[_bp7] / X[_rest].clip(lower=0.5)

    # ── Individual reliever usage edge (Group G) ─────────────────────────────────
    # Positive = away team used more relievers recently (home team has fresher pen)
    for _w, _days in (("1d", "1"), ("2d", "2"), ("3d", "3")):
        h = f"home_bp_relievers_last_{_w}"
        a = f"away_bp_relievers_last_{_w}"
        if h in X.columns and a in X.columns:
            X[f"bp_relievers_edge_{_w}"] = (
                pd.to_numeric(X[a], errors="coerce").fillna(0)
                - pd.to_numeric(X[h], errors="coerce").fillna(0)
            )

    # Yesterday's reliever count × yesterday's IP: "hot pen" signal
    # (many relievers AND high IP yesterday = deeply depleted bullpen)
    for _side in ("home", "away"):
        _cnt = f"{_side}_bp_relievers_last_1d"
        _ip  = f"{_side}_bp_ip_last_1d"
        if _cnt in X.columns and _ip in X.columns:
            X[f"{_side}_bp_hot_1d"] = (
                pd.to_numeric(X[_cnt], errors="coerce").fillna(0)
                * pd.to_numeric(X[_ip],  errors="coerce").fillna(0)
            )

    # ── SP quality × opponent batting interaction (Improvement 3) ────────────────
    # ERA × opposing batting average: pitcher quality vs opponent quality
    for _h, _a in [("home", "away"), ("away", "home")]:
        _sp_era = f"{_h}_sp_career_era_5"
        _opp_avg = f"{_a}_avg_avg_10"
        if _sp_era in X.columns and _opp_avg in X.columns:
            X[f"{_h}_sp_era_x_opp_avg"] = X[_sp_era] * X[_opp_avg] * 10  # scale for visibility

    # ── Park-adjusted run features (Improvement 5) ──────────────────────────────
    # Offensive features adjusted for park run-scoring environment
    if "park_run_factor" in X.columns:
        for _side in ("home", "away"):
            _runs = f"{_side}_runs_avg_10"
            if _runs in X.columns:
                X[f"{_side}_park_adj_runs_10"] = X[_runs] * X["park_run_factor"]

    # ── Elo-based interactions ────────────────────────────────────────────────
    # Elo advantage × pitching quality: strong team with good pitcher = amplified edge
    if "elo_diff" in X.columns:
        for _side, _sign in [("home", 1), ("away", -1)]:
            _sp_era = f"{_side}_sp_era_5"
            if _sp_era in X.columns:
                # Negative product = team is favoured AND has low ERA (good combo)
                X[f"{_side}_elo_x_sp_quality"] = (_sign * X["elo_diff"]) / (X[_sp_era].clip(lower=1.0))

    # ── SP recent form trends (5-start vs 10-start) ──────────────────────────
    # Captures hot/cold SP stretches that season-average ERA misses entirely.
    # Negative = pitcher improving (ERA lower in last 5 vs last 10 starts).
    # Positive = pitcher declining (ERA higher recently = danger signal).
    for _side in ("home", "away"):
        _era5  = f"{_side}_sp_career_era_5"
        _era10 = f"{_side}_sp_career_era_10"
        if _era5 in X.columns and _era10 in X.columns:
            X[f"{_side}_sp_era_trend_5v10"] = (
                pd.to_numeric(X[_era5],  errors="coerce")
                - pd.to_numeric(X[_era10], errors="coerce")
            )
        _fip5  = f"{_side}_sp_fip_5"
        _fip10 = f"{_side}_sp_fip_10"
        if _fip5 in X.columns and _fip10 in X.columns:
            X[f"{_side}_sp_fip_trend_5v10"] = (
                pd.to_numeric(X[_fip5],  errors="coerce")
                - pd.to_numeric(X[_fip10], errors="coerce")
            )
        _k5  = f"{_side}_sp_k_pct_5"
        _k10 = f"{_side}_sp_k_pct_10"
        if _k5 in X.columns and _k10 in X.columns:
            X[f"{_side}_sp_k_trend_5v10"] = (
                pd.to_numeric(X[_k5],  errors="coerce")
                - pd.to_numeric(X[_k10], errors="coerce")
            )

    # ERA trend differential: positive = away SP getting worse vs home SP (home form edge)
    if "home_sp_era_trend_5v10" in X.columns and "away_sp_era_trend_5v10" in X.columns:
        X["sp_era_trend_diff"] = (
            X["away_sp_era_trend_5v10"] - X["home_sp_era_trend_5v10"]
        )
    if "home_sp_fip_trend_5v10" in X.columns and "away_sp_fip_trend_5v10" in X.columns:
        X["sp_fip_trend_diff"] = (
            X["away_sp_fip_trend_5v10"] - X["home_sp_fip_trend_5v10"]
        )

    # Combined SP form edge (average of ERA and FIP trend differentials)
    if "sp_era_trend_diff" in X.columns and "sp_fip_trend_diff" in X.columns:
        X["sp_form_edge"] = (X["sp_era_trend_diff"] + X["sp_fip_trend_diff"]) / 2.0
    elif "sp_era_trend_diff" in X.columns:
        X["sp_form_edge"] = X["sp_era_trend_diff"]

    # ── SP last-outing workload (Item #4) ────────────────────────────────────────
    # last_start_ip = innings TODAY's SP threw in their most recent prior start.
    # Short outing: was pulled before 5th inning → may signal poor form or injury.
    # IP / rest days: high = more accumulated fatigue entering today.
    for _side in ("home", "away"):
        _ip   = f"{_side}_sp_last_ip"
        _rest = f"{_side}_sp_days_rest"
        if _ip in X.columns:
            _lip = pd.to_numeric(X[_ip], errors="coerce").fillna(5.5)  # ~league avg
            X[f"{_side}_sp_last_short"] = (_lip < 5.0).astype(int)
            if _rest in X.columns:
                _r = pd.to_numeric(X[_rest], errors="coerce").clip(lower=1.0).fillna(5.0)
                X[f"{_side}_sp_ip_per_rest"] = _lip / _r

    # Differential: positive = away SP carried more workload per rest day (home edge)
    if "home_sp_ip_per_rest" in X.columns and "away_sp_ip_per_rest" in X.columns:
        X["sp_workload_edge"] = X["away_sp_ip_per_rest"] - X["home_sp_ip_per_rest"]

    # ── Opener / bullpen-game detection ──────────────────────────────────────
    # An "opener" SP throws ≤ 2 IP (or has no recent starts). In opener games the
    # entire run-environment model changes: there is no traditional SP suppression,
    # and the opposing lineup bats against relievers from inning 2 onward.
    # Signal: sp_last_ip < 2.0 OR starts_in_window_5 == 0.
    for _side in ("home", "away"):
        _lip    = f"{_side}_sp_last_ip"
        _starts = f"{_side}_sp_starts_in_window_5"
        _is_opener = pd.Series(0, index=X.index)
        if _lip in X.columns:
            _lip_val = pd.to_numeric(X[_lip], errors="coerce")
            _is_opener = _is_opener | (_lip_val < 2.0).fillna(False)
        if _starts in X.columns:
            _starts_val = pd.to_numeric(X[_starts], errors="coerce").fillna(1.0)
            _is_opener = _is_opener | (_starts_val == 0)
        if _lip in X.columns or _starts in X.columns:
            X[f"{_side}_is_opener"] = _is_opener.astype(int)
    if "home_is_opener" in X.columns and "away_is_opener" in X.columns:
        # Positive = away team using opener (home team faces bullpen from early → easier scoring)
        X["opener_asymmetry"] = X["away_is_opener"] - X["home_is_opener"]

    # ── SP 20-start regression anchor ─────────────────────────────────────────
    # 5-start ERA minus 20-start baseline: positive = recently worse than career norm
    # → expect regression toward mean (caution flag). Negative = on a hot streak.
    for _side in ("home", "away"):
        _e5  = f"{_side}_sp_career_era_5"
        _e20 = f"{_side}_sp_era_20"
        _f5  = f"{_side}_sp_fip_5"
        _f20 = f"{_side}_sp_fip_20"
        if _e20 in X.columns and _e5 in X.columns:
            X[f"{_side}_sp_era_5v20"] = (
                pd.to_numeric(X[_e5],  errors="coerce").fillna(4.0)
                - pd.to_numeric(X[_e20], errors="coerce").fillna(4.0)
            )
        if _f20 in X.columns and _f5 in X.columns:
            X[f"{_side}_sp_fip_5v20"] = (
                pd.to_numeric(X[_f5],  errors="coerce").fillna(4.0)
                - pd.to_numeric(X[_f20], errors="coerce").fillna(4.0)
            )
    # Differential: positive = away SP more likely to regress (home pitching edge)
    if "home_sp_era_5v20" in X.columns and "away_sp_era_5v20" in X.columns:
        X["sp_era_regression_diff"] = (
            X["away_sp_era_5v20"] - X["home_sp_era_5v20"]
        )

    # ── 3c: SP ERA Bayesian Shrinkage ─────────────────────────────────────────
    # 5-start ERA shrunk 80% toward league average (4.20) to reduce noise from
    # small sample sizes early in the season.
    LEAGUE_ERA = 4.20
    for _side in ("home", "away"):
        _e5 = f"{_side}_sp_era_5"
        if _e5 in X.columns:
            _era = pd.to_numeric(X[_e5], errors="coerce").fillna(LEAGUE_ERA)
            X[f"{_side}_sp_era_shrunk"] = (_era * 20.0 + LEAGUE_ERA * 80.0) / 100.0
    if "home_sp_era_shrunk" in X.columns and "away_sp_era_shrunk" in X.columns:
        X["sp_era_shrunk_diff"] = X["away_sp_era_shrunk"] - X["home_sp_era_shrunk"]

    # ── Last-start quality composite ──────────────────────────────────────────
    # FIP of last outing = best single-number quality proxy for that start.
    # K/IP rate in last start = command × stuff signal.
    for _side in ("home", "away"):
        _lfip = f"{_side}_sp_last_fip"
        _lk   = f"{_side}_sp_last_k"
        _lip  = f"{_side}_sp_last_ip"
        if _lfip in X.columns:
            X[f"{_side}_sp_last_quality"] = (
                pd.to_numeric(X[_lfip], errors="coerce").fillna(4.0)
            )
        if _lk in X.columns and _lip in X.columns:
            _lk_val  = pd.to_numeric(X[_lk],  errors="coerce").fillna(0.0)
            _lip_val = pd.to_numeric(X[_lip], errors="coerce").clip(lower=0.1).fillna(5.5)
            X[f"{_side}_sp_last_k_rate"] = _lk_val / _lip_val
    # Positive = away SP had worse last-outing FIP (home pitching edge today)
    if "home_sp_last_quality" in X.columns and "away_sp_last_quality" in X.columns:
        X["sp_last_quality_diff"] = (
            pd.to_numeric(X["away_sp_last_quality"], errors="coerce").fillna(4.0)
            - pd.to_numeric(X["home_sp_last_quality"], errors="coerce").fillna(4.0)
        )

    # ── SP venue familiarity (career ERA at this specific ballpark) ───────────
    # Pitchers perform materially differently at specific parks (flyball pitcher at
    # Coors vs. groundball pitcher at Petco). Reliability-weighted by prior starts.
    for _side in ("home", "away"):
        _v_era  = f"{_side}_sp_venue_era"
        _c_era  = f"{_side}_sp_career_era_5"
        _starts = f"{_side}_sp_venue_starts"
        if _v_era in X.columns and _c_era in X.columns:
            # Scale reliability 0→1 as starts 0→15 (cap at 15 to avoid single outlier dominance)
            _rel = (
                pd.to_numeric(X[_starts], errors="coerce").clip(upper=15.0) / 15.0
            ).fillna(0.0) if _starts in X.columns else pd.Series(0.0, index=X.index)
            # Positive = pitcher performs better at this venue vs recent career ERA
            X[f"{_side}_sp_venue_advantage"] = (
                pd.to_numeric(X[_c_era],  errors="coerce").fillna(4.0)
                - pd.to_numeric(X[_v_era], errors="coerce").fillna(4.0)
            ) * _rel

    # Home advantage vs away disadvantage at this park
    if "home_sp_venue_advantage" in X.columns and "away_sp_venue_advantage" in X.columns:
        X["sp_venue_familiarity_diff"] = (
            X["home_sp_venue_advantage"] - X["away_sp_venue_advantage"]
        )

    # ── Umpire × SP strikeout interactions for game model ────────────────────
    # The umpire's strike zone directly affects how many Ks and walks happen.
    # This is the #1 umpire-driven signal for totals.
    if "ump_k9_5" in X.columns:
        _ump_k9 = pd.to_numeric(X["ump_k9_5"], errors="coerce").fillna(7.5)
        for _side in ("home", "away"):
            _sp_k = f"{_side}_sp_k_pct_5"
            if _sp_k in X.columns:
                # Wide-zone ump amplifies a strikeout pitcher's K ability → fewer baserunners
                X[f"ump_k9_x_{_side}_sp_k"] = (
                    _ump_k9 * pd.to_numeric(X[_sp_k], errors="coerce").fillna(0.22)
                )
        # Average K environment for the whole game (both SPs × ump zone)
        _h = "ump_k9_x_home_sp_k"
        _a = "ump_k9_x_away_sp_k"
        if _h in X.columns and _a in X.columns:
            X["ump_sp_k_environment"] = (X[_h] + X[_a]) / 2.0

    if "ump_bb9_5" in X.columns:
        _ump_bb9 = pd.to_numeric(X["ump_bb9_5"], errors="coerce").fillna(3.0)
        for _side in ("home", "away"):
            _team_bb = f"{_side}_bb_pct_avg_10"
            if _team_bb in X.columns:
                # Generous ump + walk-prone team = more baserunners = more runs
                X[f"ump_bb9_x_{_side}_team_bb"] = (
                    _ump_bb9 * pd.to_numeric(X[_team_bb], errors="coerce").fillna(0.09)
                )

    # Direct ump scoring environment (strong predictor for game totals)
    if "ump_rpg_5" in X.columns:
        X["ump_runs_per_game"] = pd.to_numeric(X["ump_rpg_5"], errors="coerce")
        # Umpire scoring environment amplified by park run factor
        if "park_run_factor" in X.columns:
            X["ump_rpg_x_park"] = (
                X["ump_runs_per_game"]
                * pd.to_numeric(X["park_run_factor"], errors="coerce").fillna(1.0)
            )

    # ── Combined total-suppression environment ────────────────────────────────
    # These are ADDITIVE features for the total model — not home-vs-away differentials.
    # Both starters' K-rate: high combined K% = fewer baserunners across the whole game.
    _hspk = "home_sp_k_pct_5"
    _aspk = "away_sp_k_pct_5"
    if _hspk in X.columns and _aspk in X.columns:
        X["combined_sp_k_pct"] = (
            pd.to_numeric(X[_hspk], errors="coerce").fillna(0.22)
            + pd.to_numeric(X[_aspk], errors="coerce").fillna(0.22)
        ) / 2.0

    # Both bullpens' K/9: high combined bullpen K9 suppresses late-inning run scoring.
    _hbpk = "home_bp_k9_5"
    _abpk = "away_bp_k9_5"
    if _hbpk in X.columns and _abpk in X.columns:
        X["combined_bp_k9"] = (
            pd.to_numeric(X[_hbpk], errors="coerce").fillna(8.5)
            + pd.to_numeric(X[_abpk], errors="coerce").fillna(8.5)
        ) / 2.0
        # Bullpen K differential for run-line: positive = home bullpen has higher K9
        X["bp_k9_edge"] = (
            pd.to_numeric(X[_hbpk], errors="coerce").fillna(8.5)
            - pd.to_numeric(X[_abpk], errors="coerce").fillna(8.5)
        )

    # Full-game K environment: SP + bullpen combined × umpire zone
    if "combined_sp_k_pct" in X.columns and "combined_bp_k9" in X.columns:
        X["total_k_env"] = X["combined_sp_k_pct"] * X["combined_bp_k9"]
        if "ump_sp_k_environment" in X.columns:
            X["total_k_suppression"] = (
                X["total_k_env"]
                * pd.to_numeric(X["ump_sp_k_environment"], errors="coerce").fillna(1.5)
            )

    # ── 3d: Park-Adjusted K Environment ─────────────────────────────────────
    # In HR-friendly parks (pf > 1), batters are slightly less K-averse (swinging
    # for the fences), so raw K-env overstates suppression in those parks.
    if "total_k_env" in X.columns and "park_run_factor" in X.columns:
        _pf = pd.to_numeric(X["park_run_factor"], errors="coerce").fillna(1.0)
        X["total_k_env_park_adj"] = (
            pd.to_numeric(X["total_k_env"], errors="coerce")
            / (1.0 + 0.15 * (_pf - 1.0)).clip(lower=0.5)
        )

    # ── Combined offensive environment (additive signals for total model) ──────
    # These are SUM (not difference) of home + away stats — the kind of signal
    # that predicts total run scoring rather than which team wins.
    _additive_pairs = [
        ("home_runs_avg_5",  "away_runs_avg_5",  "combined_runs_avg_5",  4.0),
        ("home_runs_avg_10", "away_runs_avg_10", "combined_runs_avg_10", 4.0),
        ("home_obp_avg_10",  "away_obp_avg_10",  "combined_obp_avg_10",  0.32),
        ("home_slg_avg_10",  "away_slg_avg_10",  "combined_slg_avg_10",  0.40),
        ("home_ops_10",      "away_ops_10",      "combined_ops_10",      0.72),
    ]
    for _hc, _ac, _name, _fill in _additive_pairs:
        if _hc in X.columns and _ac in X.columns:
            X[_name] = (
                pd.to_numeric(X[_hc], errors="coerce").fillna(_fill)
                + pd.to_numeric(X[_ac], errors="coerce").fillna(_fill)
            )

    # ── Stolen base / team speed features ────────────────────────────────────
    # SB rate reflects team aggressiveness on the basepaths. Faster/more aggressive
    # teams manufacture runs beyond what batting stats alone capture.
    for _side in ("home", "away"):
        _sb5  = f"{_side}_sb_avg_5"
        _sb10 = f"{_side}_sb_avg_10"
        if _sb5 in X.columns and _sb10 in X.columns:
            X[f"{_side}_sb_trend"] = (
                pd.to_numeric(X[_sb5],  errors="coerce").fillna(0.5)
                - pd.to_numeric(X[_sb10], errors="coerce").fillna(0.5)
            )
    if "home_sb_avg_10" in X.columns and "away_sb_avg_10" in X.columns:
        X["sb_rate_diff"] = (
            pd.to_numeric(X["home_sb_avg_10"], errors="coerce").fillna(0.5)
            - pd.to_numeric(X["away_sb_avg_10"], errors="coerce").fillna(0.5)
        )
        X["combined_sb_avg_10"] = (
            pd.to_numeric(X["home_sb_avg_10"], errors="coerce").fillna(0.5)
            + pd.to_numeric(X["away_sb_avg_10"], errors="coerce").fillna(0.5)
        )

    # Combined SP ERA (both pitchers struggling = more runs expected)
    for _suffix, _fill in [("5", 4.0), ("10", 4.0)]:
        _hsp = f"home_sp_era_{_suffix}"
        _asp = f"away_sp_era_{_suffix}"
        if _hsp in X.columns and _asp in X.columns:
            X[f"combined_sp_era_{_suffix}"] = (
                pd.to_numeric(X[_hsp], errors="coerce").fillna(_fill)
                + pd.to_numeric(X[_asp], errors="coerce").fillna(_fill)
            )

    # Park-adjusted total run environment
    if "park_run_factor" in X.columns and "combined_runs_avg_10" in X.columns:
        X["park_adj_total_env"] = (
            pd.to_numeric(X["combined_runs_avg_10"], errors="coerce").fillna(8.5)
            * pd.to_numeric(X["park_run_factor"], errors="coerce").fillna(1.0)
        )

    # ── 3e: Top-4 Lineup × Opponent Bullpen Interaction ──────────────────────
    # Top-4 sluggers facing a weak/tired opponent bullpen = elevated run scoring.
    if "home_top4_slg_avg_10" in X.columns and "away_bp_era_5" in X.columns:
        _hslg = pd.to_numeric(X["home_top4_slg_avg_10"], errors="coerce").fillna(0.4)
        _abp  = pd.to_numeric(X["away_bp_era_5"], errors="coerce").fillna(4.0).clip(lower=1.0)
        X["home_top4_x_opp_bp"] = _hslg * (4.0 / _abp)
    if "away_top4_slg_avg_10" in X.columns and "home_bp_era_5" in X.columns:
        _aslg = pd.to_numeric(X["away_top4_slg_avg_10"], errors="coerce").fillna(0.4)
        _hbp  = pd.to_numeric(X["home_bp_era_5"], errors="coerce").fillna(4.0).clip(lower=1.0)
        X["away_top4_x_opp_bp"] = _aslg * (4.0 / _hbp)
    if "home_top4_x_opp_bp" in X.columns and "away_top4_x_opp_bp" in X.columns:
        X["top4_bp_matchup_diff"] = X["home_top4_x_opp_bp"] - X["away_top4_x_opp_bp"]

    # ── SP Statcast quality features ─────────────────────────────────────────
    # combined_* = additive (both SPs' quality) → signal for total runs model
    # diff_*     = differential (home SP worse than away) → signal for run-line model
    _sc_game_pairs = [
        ("home_sp_sc_xwoba",        "away_sp_sc_xwoba",        "sp_sc_xwoba",        0.320),
        ("home_sp_sc_barrel_rate",  "away_sp_sc_barrel_rate",  "sp_sc_barrel_rate",   6.0),
        ("home_sp_sc_hard_hit_pct", "away_sp_sc_hard_hit_pct", "sp_sc_hard_hit_pct", 35.0),
        ("home_sp_sc_exit_velo",    "away_sp_sc_exit_velo",    "sp_sc_exit_velo",    88.5),
    ]
    for _h, _a, _stem, _fill in _sc_game_pairs:
        if _h in X.columns and _a in X.columns:
            _hv = pd.to_numeric(X[_h], errors="coerce").fillna(_fill)
            _av = pd.to_numeric(X[_a], errors="coerce").fillna(_fill)
            X[f"combined_{_stem}"] = _hv + _av   # higher = more runs expected
            X[f"diff_{_stem}"]     = _hv - _av   # positive = home SP allows more

    # Combined whiff environment (both SPs dominant → fewer total runs)
    for _h, _a, _name, _fill in [
        ("home_sp_sl_whiff_pct", "away_sp_sl_whiff_pct", "combined_sl_whiff_pct", 25.0),
        ("home_sp_ch_whiff_pct", "away_sp_ch_whiff_pct", "combined_ch_whiff_pct", 20.0),
        ("home_sp_fb_put_away",  "away_sp_fb_put_away",  "combined_fb_put_away",  15.0),
    ]:
        if _h in X.columns and _a in X.columns:
            X[_name] = (
                pd.to_numeric(X[_h], errors="coerce").fillna(_fill)
                + pd.to_numeric(X[_a], errors="coerce").fillna(_fill)
            )

    # ── SP Statcast discipline (season-level K/BB/whiff anchors) ─────────────
    # Stable season totals complement rolling ERA/FIP: a SP with K%=30, BB%=5
    # suppresses runs regardless of recent ERA variance.
    # combined_* → total runs model (high K + low BB = low-scoring game)
    # diff_*     → run-line model (home SP command edge over away)
    _disc_pairs = [
        ("home_sp_sc_k_pct",        "away_sp_sc_k_pct",        "sp_sc_k_pct",        22.0),
        ("home_sp_sc_bb_pct",       "away_sp_sc_bb_pct",       "sp_sc_bb_pct",        8.0),
        ("home_sp_sc_whiff_pct",    "away_sp_sc_whiff_pct",    "sp_sc_whiff_pct",    24.0),
        ("home_sp_sc_oz_swing_pct", "away_sp_sc_oz_swing_pct", "sp_sc_oz_swing_pct", 30.0),
    ]
    for _h, _a, _stem, _fill in _disc_pairs:
        if _h in X.columns and _a in X.columns:
            _hv = pd.to_numeric(X[_h], errors="coerce").fillna(_fill)
            _av = pd.to_numeric(X[_a], errors="coerce").fillna(_fill)
            X[f"combined_{_stem}"] = _hv + _av
            X[f"diff_{_stem}"]     = _hv - _av

    # K/BB ratio from discipline stats — command efficiency (K% / BB%)
    if "home_sp_sc_k_pct" in X.columns and "home_sp_sc_bb_pct" in X.columns:
        _hk = pd.to_numeric(X["home_sp_sc_k_pct"], errors="coerce").fillna(22.0)
        _hb = pd.to_numeric(X["home_sp_sc_bb_pct"], errors="coerce").fillna(8.0).clip(lower=1.0)
        X["home_sp_sc_k_bb_ratio"] = _hk / _hb
    if "away_sp_sc_k_pct" in X.columns and "away_sp_sc_bb_pct" in X.columns:
        _ak = pd.to_numeric(X["away_sp_sc_k_pct"], errors="coerce").fillna(22.0)
        _ab = pd.to_numeric(X["away_sp_sc_bb_pct"], errors="coerce").fillna(8.0).clip(lower=1.0)
        X["away_sp_sc_k_bb_ratio"] = _ak / _ab
    if "home_sp_sc_k_bb_ratio" in X.columns and "away_sp_sc_k_bb_ratio" in X.columns:
        X["diff_sp_sc_k_bb_ratio"] = X["home_sp_sc_k_bb_ratio"] - X["away_sp_sc_k_bb_ratio"]

    # ── Venue altitude → air density → run scoring boost ─────────────────────
    # Thinner air at altitude reduces drag on batted balls; Coors Field effect.
    # Air density relative to sea level: ρ/ρ₀ ≈ exp(-alt_ft / 25_000).
    # Effect: each 1000 ft above sea level adds ~1.5% to ball carry distance.
    # We express this as a multiplicative run-scoring factor on top of park_run_factor.
    _PARK_ALTITUDE_FT = {
        "COL": 5280,   # Coors Field, Denver
        "ARI": 1082,   # Chase Field, Phoenix
        "TEX": 551,    # Globe Life Field, Arlington
        "ATL": 1050,   # Truist Park, Atlanta
        "CIN": 869,    # Great American Ball Park, Cincinnati
        "STL": 466,    # Busch Stadium, St. Louis
        "KC":  909,    # Kauffman Stadium, Kansas City
        "MIN": 841,    # Target Field, Minneapolis
        "CHC": 595,    # Wrigley Field, Chicago
        "CWS": 595,    # Guaranteed Rate Field, Chicago
        "MIL": 635,    # American Family Field, Milwaukee
        "PIT": 730,    # PNC Park, Pittsburgh
        "DEN": 5280,   # alias if team abbr varies
    }
    # Default sea-level parks get 0 ft (SF, NYM, NYY, BOS, LAD, SD, SEA, etc.)
    if "home_team_abbr" in X.columns:
        _alt = X["home_team_abbr"].map(_PARK_ALTITUDE_FT).fillna(0.0).astype(float)
        # Air density factor: 1.0 at sea level, lower at altitude → ball carries farther
        # Simple linear approximation: −0.003% per foot (calibrated to Coors empirically)
        X["venue_altitude_ft"] = _alt
        X["altitude_run_boost"] = (_alt / 5280.0).clip(upper=1.0) * 0.18
        # Interaction: altitude effect amplified in warm parks (hot + thin air)
        if "temperature_f" in X.columns:
            _temp = pd.to_numeric(X["temperature_f"], errors="coerce").fillna(72.0)
            X["altitude_x_temp"] = X["altitude_run_boost"] * (_temp / 72.0).clip(lower=0.5)

    # ── Weather interactions for game totals ─────────────────────────────────
    # Cold temperatures suppress scoring — ball doesn't carry in cold air.
    if "temperature_f" in X.columns:
        _temp = pd.to_numeric(X["temperature_f"], errors="coerce").fillna(72.0)
        _dome = pd.to_numeric(
            X["is_dome"] if "is_dome" in X.columns else pd.Series(0.0, index=X.index),
            errors="coerce",
        ).fillna(0.0)
        # Each degree below 60°F in an outdoor park reduces scoring slightly
        X["temp_cold_penalty"] = (60.0 - _temp).clip(lower=0.0) * (1.0 - _dome)
        # Binary flag: < 50°F outdoor = significant suppressor (~0.5 fewer runs vs warm avg)
        X["is_cold_outdoor"] = ((_temp < 50.0) & (_dome < 0.5)).astype(int)

    # Wind: wind_cos ≈ -1 means wind blowing OUT to CF (tailwind = more HRs + runs);
    # wind_cos ≈ +1 means wind blowing IN from CF (headwind = suppresses scoring).
    if "wind_cos" in X.columns:
        _wind_cos = pd.to_numeric(X["wind_cos"], errors="coerce").fillna(0.0)
        _wind_spd = pd.to_numeric(
            X["wind_speed_mph"] if "wind_speed_mph" in X.columns else pd.Series(0.0, index=X.index),
            errors="coerce",
        ).fillna(0.0)
        _pf = pd.to_numeric(
            X["park_run_factor"] if "park_run_factor" in X.columns else pd.Series(1.0, index=X.index),
            errors="coerce",
        ).fillna(1.0)
        # Negative cos (tailwind) × speed × park factor = run-scoring boost
        X["wind_scoring_boost"] = -_wind_cos * _wind_spd * _pf / 10.0  # /10 to scale

    # Precipitation risk suppresses totals (rain delays, shortened games)
    if "precip_prob_pct" in X.columns:
        _precip = pd.to_numeric(X["precip_prob_pct"], errors="coerce").fillna(0.0)
        _dome = pd.to_numeric(
            X["is_dome"] if "is_dome" in X.columns else pd.Series(0.0, index=X.index),
            errors="coerce",
        ).fillna(0.0)
        X["precip_scoring_risk"] = _precip * (1.0 - _dome) / 100.0

    # ── SP handedness vs opposing lineup (platoon advantage) ─────────────────
    # Teams systematically hit better or worse vs one pitcher handedness.
    # We select the relevant hand-split batting stat based on today's opposing SP.
    # Reliability weighted by games_vs_lhp / games_vs_rhp (sample size).
    for _bat_side, _sp_side in [("home", "away"), ("away", "home")]:
        _sp_hand_L  = f"{_sp_side}_sp_pitch_hand_l"   # 1 if today's SP is LHP (lowercase — matches PostgreSQL)
        _avg_vs_lhp = f"{_bat_side}_team_avg_vs_lhp"
        _avg_vs_rhp = f"{_bat_side}_team_avg_vs_rhp"
        _obp_vs_lhp = f"{_bat_side}_team_obp_vs_lhp"
        _obp_vs_rhp = f"{_bat_side}_team_obp_vs_rhp"
        _g_lhp      = f"{_bat_side}_games_vs_lhp"
        _g_rhp      = f"{_bat_side}_games_vs_rhp"
        _overall    = f"{_bat_side}_avg_avg_10"        # overall 10-game batting avg

        if _sp_hand_L not in X.columns or _avg_vs_lhp not in X.columns:
            continue

        _is_lhp = pd.to_numeric(X[_sp_hand_L], errors="coerce").fillna(0.0)
        _overall_avg = pd.to_numeric(
            X[_overall] if _overall in X.columns else pd.Series(0.255, index=X.index),
            errors="coerce",
        ).fillna(0.255)

        # Reliability weights: scale 0→1 as sample games 0→10
        _g_vs_l = pd.to_numeric(
            X[_g_lhp] if _g_lhp in X.columns else pd.Series(0.0, index=X.index),
            errors="coerce",
        ).fillna(0.0)
        _g_vs_r = pd.to_numeric(
            X[_g_rhp] if _g_rhp in X.columns else pd.Series(0.0, index=X.index),
            errors="coerce",
        ).fillna(0.0)
        _rel_l = (_g_vs_l / 10.0).clip(upper=1.0)
        _rel_r = (_g_vs_r / 10.0).clip(upper=1.0)

        # Weighted batting avg vs the actual opposing SP handedness today
        _avgl = pd.to_numeric(X[_avg_vs_lhp], errors="coerce").fillna(_overall_avg)
        _avgr = pd.to_numeric(X[_avg_vs_rhp], errors="coerce").fillna(_overall_avg)
        _bat_vs_sp_hand = (
            _is_lhp * (_overall_avg + (_avgl - _overall_avg) * _rel_l)
            + (1.0 - _is_lhp) * (_overall_avg + (_avgr - _overall_avg) * _rel_r)
        )
        # Delta vs overall batting: positive = team has platoon advantage today
        X[f"{_bat_side}_hand_matchup_edge"] = _bat_vs_sp_hand - _overall_avg

        # OPS-based version for stronger signal
        if _obp_vs_lhp in X.columns and _obp_vs_rhp in X.columns:
            _obpl = pd.to_numeric(X[_obp_vs_lhp], errors="coerce")
            _obpr = pd.to_numeric(X[_obp_vs_rhp], errors="coerce")
            _obp_vs_hand = _is_lhp * _obpl.fillna(0.32) + (1.0 - _is_lhp) * _obpr.fillna(0.32)
            X[f"{_bat_side}_obp_vs_sp_hand"] = _obp_vs_hand

    # Combined platoon edge: home batting advantage vs away SP hand minus away's vs home SP
    if "home_hand_matchup_edge" in X.columns and "away_hand_matchup_edge" in X.columns:
        X["platoon_matchup_diff"] = X["home_hand_matchup_edge"] - X["away_hand_matchup_edge"]

    # ── Lineup Statcast quality edge (home - away) ────────────────────────────
    # Positive = home lineup has better Statcast contact quality vs away SP.
    for _h, _a, _stem, _fill in [
        ("home_lineup_xwoba_avg",    "away_lineup_xwoba_avg",    "lineup_xwoba",    0.315),
        ("home_lineup_barrel_avg",   "away_lineup_barrel_avg",   "lineup_barrel",   6.5),
        ("home_lineup_hard_hit_avg", "away_lineup_hard_hit_avg", "lineup_hard_hit", 37.0),
    ]:
        if _h in X.columns and _a in X.columns:
            X[f"diff_{_stem}"] = (
                pd.to_numeric(X[_h], errors="coerce").fillna(_fill)
                - pd.to_numeric(X[_a], errors="coerce").fillna(_fill)
            )

    # ── 3f: H2H Sample-Size Confidence Weighting ─────────────────────────────
    # Weight H2H slugging stats by number of matchup games (0→1 as n→15).
    # Prevents single-game flukes from producing large H2H edges.
    if "home_h2h_slg" in X.columns and "home_h2h_n" in X.columns:
        _wh = (pd.to_numeric(X["home_h2h_n"], errors="coerce").fillna(0.0) / 15.0).clip(upper=1.0)
        _wa = (pd.to_numeric(X["away_h2h_n"], errors="coerce").fillna(0.0) / 15.0).clip(upper=1.0)
        X["home_h2h_slg_wtd"] = pd.to_numeric(X["home_h2h_slg"], errors="coerce").fillna(0.0) * _wh
        X["away_h2h_slg_wtd"] = pd.to_numeric(X["away_h2h_slg"], errors="coerce").fillna(0.0) * _wa
        X["h2h_slg_edge_wtd"] = X["home_h2h_slg_wtd"] - X["away_h2h_slg_wtd"]

    # ── Catcher framing differential (game model) ─────────────────────────────
    # framing_rv > 0 = catcher converts more borderline pitches to strikes.
    # Differential favours the home team's pitcher when home catcher is better.
    if "home_catcher_framing_rv" in X.columns and "away_catcher_framing_rv" in X.columns:
        _hfr = pd.to_numeric(X["home_catcher_framing_rv"], errors="coerce").fillna(0.0)
        _afr = pd.to_numeric(X["away_catcher_framing_rv"], errors="coerce").fillna(0.0)
        X["catcher_framing_rv_diff"] = _hfr - _afr
        X["combined_catcher_framing_rv"] = _hfr + _afr

    # ── SP velocity trend differential ──────────────────────────────────────────
    # Positive fb_velo_trend_diff = home SP throwing harder than their 5-start norm
    if "home_fb_velo_trend_5" in X.columns and "away_fb_velo_trend_5" in X.columns:
        _hvt = pd.to_numeric(X["home_fb_velo_trend_5"], errors="coerce").fillna(0.0)
        _avt = pd.to_numeric(X["away_fb_velo_trend_5"], errors="coerce").fillna(0.0)
        X["fb_velo_trend_diff"]     = _hvt - _avt
        X["abs_fb_velo_trend_home"] = _hvt.abs()
        X["abs_fb_velo_trend_away"] = _avt.abs()

    return X


def add_player_prop_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived interaction features for MLB player prop models.

    Works for both pitcher and batter DataFrames by checking column presence.
    Call this after OHE / coercion so all inputs are numeric.

    Pitcher columns trigger pitcher-specific features (k_pct_5, era_5, k9_5, …).
    Batter columns trigger batter-specific features (hits_avg_5, tb_avg_5, …).
    """
    X = X.copy()

    # ── Batter features ──────────────────────────────────────────────────────
    # Trend: recent (5g) vs medium-term (10g) rolling averages
    for stat in ("hits", "hr", "tb"):
        avg5  = f"{stat}_avg_5"
        avg10 = f"{stat}_avg_10"
        sd10  = f"{stat}_sd_10"
        if avg5 in X.columns and avg10 in X.columns:
            X[f"{stat}_trend_5v10"] = X[avg5] - X[avg10]
            # Hot ratio — scale-invariant: 1.0 = neutral, >1 = hot streak
            denom = X[avg10].where(X[avg10].abs() > 1e-6, other=np.nan)
            X[f"{stat}_hot_ratio"] = X[avg5] / denom
        # Trend z-score (significance relative to player volatility)
        if f"{stat}_trend_5v10" in X.columns and sd10 in X.columns:
            denom = X[sd10].where(X[sd10] > 1e-6, other=np.nan)
            X[f"{stat}_trend_sig"] = X[f"{stat}_trend_5v10"] / denom

    # Park-adjusted hit rates
    if "hr_avg_10" in X.columns and "park_hr_factor" in X.columns:
        X["hr_park_adj_10"] = X["hr_avg_10"] * X["park_hr_factor"]
    if "tb_avg_10" in X.columns and "park_run_factor" in X.columns:
        X["tb_park_adj_10"] = X["tb_avg_10"] * X["park_run_factor"]

    # AB-weighted cumulative HR rate features (more reliable than per-game avg for rare events)
    # hr_rate_cumul_10 uses SUM(hr)/SUM(ab) — down-weights 1-AB pinch-hit HRs vs AVG(game_hr_rate)
    if "hr_rate_cumul_10" in X.columns:
        _cumul_10 = pd.to_numeric(X["hr_rate_cumul_10"], errors="coerce")
        # Trend: are cumulative HRs accelerating or cooling over short vs long window?
        if "hr_rate_cumul_5" in X.columns and "hr_rate_cumul_20" in X.columns:
            _cumul_5  = pd.to_numeric(X["hr_rate_cumul_5"],  errors="coerce").fillna(0.0)
            _cumul_20 = pd.to_numeric(X["hr_rate_cumul_20"], errors="coerce").fillna(0.0)
            X["hr_rate_cumul_trend_5v20"] = _cumul_5 - _cumul_20
        # Park-amplified cumulative rate: best single HR signal
        if "park_hr_factor" in X.columns:
            _phr = pd.to_numeric(X["park_hr_factor"], errors="coerce").fillna(1.0)
            X["hr_rate_cumul_x_park"] = _cumul_10.fillna(0.0) * _phr

    # TB ceiling/volatility features — help predict OVER (big-game likelihood),
    # not just average TB. A player with high CV or high max has a fatter tail distribution.
    if "tb_sd_10" in X.columns and "tb_avg_10" in X.columns:
        _tb_mean = pd.to_numeric(X["tb_avg_10"], errors="coerce").fillna(1.2)
        _tb_sd   = pd.to_numeric(X["tb_sd_10"],  errors="coerce").fillna(0.8)
        # Coefficient of variation: high = volatile player with occasional big games
        X["tb_cv_10"] = _tb_sd / _tb_mean.clip(lower=0.1)

    # Power × opportunity interaction: ISO × expected AB count today
    # Captures expected extra-base-hit production (power player + more PAs = more TB)
    if "iso_avg_10" in X.columns and "ab_avg_10" in X.columns:
        X["tb_power_opportunity"] = (
            pd.to_numeric(X["iso_avg_10"], errors="coerce").fillna(0.12)
            * pd.to_numeric(X["ab_avg_10"], errors="coerce").fillna(3.0)
        )

    # Statcast xSLG × park HR factor (expected slugging amplified by this specific park)
    if "sc_xslg" in X.columns and "park_hr_factor" in X.columns:
        X["sc_xslg_x_park"] = (
            pd.to_numeric(X["sc_xslg"],         errors="coerce").fillna(0.38)
            * pd.to_numeric(X["park_hr_factor"], errors="coerce").fillna(1.0)
        )

    # Barrels per PA × today's expected AB count (expected barrel count this game)
    if "sc_brl_pa" in X.columns and "ab_avg_10" in X.columns:
        X["sc_brl_x_ab"] = (
            pd.to_numeric(X["sc_brl_pa"],   errors="coerce").fillna(0.035)
            * pd.to_numeric(X["ab_avg_10"], errors="coerce").fillna(3.0)
        )

    # Batter × Opponent SP interaction: batters facing strikeout-heavy pitchers get fewer hits
    if "hits_avg_10" in X.columns and "opp_sp_era_5" in X.columns:
        X["hits_vs_opp_era"] = X["hits_avg_10"] * (X["opp_sp_era_5"] / 4.0)
    if "tb_avg_10" in X.columns and "opp_sp_fip_5" in X.columns:
        X["tb_vs_opp_fip"] = X["tb_avg_10"] * (X["opp_sp_fip_5"] / 4.0)

    # ── Opponent bullpen quality (batter props) ───────────────────────────────
    # ~40 % of a batter's PAs are vs relievers; a weak or fatigued opponent BP
    # meaningfully raises expected H/TB/HR for batters later in the lineup.
    if "opp_bp_era_5" in X.columns:
        _bp_era = pd.to_numeric(X["opp_bp_era_5"], errors="coerce").fillna(4.20)
        # Deviation from league-average ERA (4.20); positive = hitter-friendly BP
        X["opp_bp_era_adj"] = _bp_era - 4.20
        # Rolling hit/TB rates scaled by BP quality (mirrors existing hits_vs_opp_era)
        if "hits_avg_10" in X.columns:
            X["hits_vs_opp_bp_era"] = X["hits_avg_10"].fillna(0.0) * (_bp_era / 4.0)
        if "tb_avg_10" in X.columns:
            X["tb_vs_opp_bp_era"] = X["tb_avg_10"].fillna(0.0) * (_bp_era / 4.0)

    # Fatigue × quality: ERA weighted by recent workload (tired + bad BP is most hittable)
    if "opp_bp_era_7d" in X.columns and "opp_bp_ip_last_7" in X.columns:
        X["opp_bp_era7_x_ip7"] = (
            pd.to_numeric(X["opp_bp_era_7d"],      errors="coerce").fillna(4.20)
            * pd.to_numeric(X["opp_bp_ip_last_7"], errors="coerce").fillna(0.0)
        )

    # ── Pitcher features ─────────────────────────────────────────────────────
    # K% × opponent K vulnerability (opponent batter K rate — higher = easier to K)
    if "k_pct_5" in X.columns and "opp_k_pct_avg_10" in X.columns:
        X["sp_k_pct_vs_opp_k_rate"] = X["k_pct_5"] * X["opp_k_pct_avg_10"]

    # Strikeout rate trend (recent vs 10-start)
    if "k9_5" in X.columns and "k9_10" in X.columns:
        X["k9_trend_5v10"] = X["k9_5"] - X["k9_10"]
    if "k_pct_5" in X.columns and "k_pct_10" in X.columns:
        X["k_pct_trend_5v10"] = X["k_pct_5"] - X["k_pct_10"]

    # ERA trend — negative means improving
    if "era_5" in X.columns and "era_10" in X.columns:
        X["era_trend_5v10"] = X["era_5"] - X["era_10"]

    # FIP trend
    if "fip_5" in X.columns and "fip_10" in X.columns:
        X["fip_trend_5v10"] = X["fip_5"] - X["fip_10"]

    # Pitcher × park: K% × park HR factor (pitcher in HR-friendly park faces fewer weak contacts)
    if "k_pct_5" in X.columns and "park_hr_factor" in X.columns:
        X["k_pct_x_park_hr"] = X["k_pct_5"] * X["park_hr_factor"]

    # ── Weather features ──────────────────────────────────────────────────────
    if "wind_speed_mph" in X.columns and "park_hr_factor" in X.columns:
        X["wind_x_park_hr"] = X["wind_speed_mph"] * X["park_hr_factor"]

    if "temperature_f" in X.columns:
        X["is_cold_game"]  = (X["temperature_f"].fillna(72.0) < 50.0).astype(int)
        X["temp_below_60"] = (60.0 - X["temperature_f"].fillna(72.0)).clip(lower=0.0)

    # ── Walk count trend + umpire interaction ─────────────────────────────────
    if "bb_avg_5" in X.columns and "bb_avg_10" in X.columns:
        X["bb_trend_5v10"] = X["bb_avg_5"] - X["bb_avg_10"]
        denom = X["bb_avg_10"].where(X["bb_avg_10"].abs() > 1e-6, other=np.nan)
        X["bb_hot_ratio"] = X["bb_avg_5"] / denom
    if "bb_trend_5v10" in X.columns and "bb_sd_10" in X.columns:
        denom = X["bb_sd_10"].where(X["bb_sd_10"] > 1e-6, other=np.nan)
        X["bb_trend_sig"] = X["bb_trend_5v10"] / denom

    # ── SP K-rate trend (5 vs 10 starts) ─────────────────────────────────────
    # Positive = pitcher on upswing in Ks (improving form).
    if "k_pct_5" in X.columns and "k_pct_10" in X.columns:
        X["sp_k_pct_trend_5v10"] = (
            pd.to_numeric(X["k_pct_5"],  errors="coerce")
            - pd.to_numeric(X["k_pct_10"], errors="coerce")
        )
    if "k9_5" in X.columns and "k9_10" in X.columns:
        X["sp_k9_trend_5v10"] = (
            pd.to_numeric(X["k9_5"],  errors="coerce")
            - pd.to_numeric(X["k9_10"], errors="coerce")
        )
    # Trending pitcher × wide-zone ump amplifies the K probability further
    if "sp_k_pct_trend_5v10" in X.columns and "ump_k9_avg_10" in X.columns:
        X["k_trend_x_ump_k9"] = (
            X["sp_k_pct_trend_5v10"].fillna(0.0) * X["ump_k9_avg_10"].fillna(0.0)
        )

    # ── Umpire zone-tendency interactions ────────────────────────────────────
    if "ump_bb9_avg_10" in X.columns and "bb_rate_avg_10" in X.columns:
        X["ump_bb9_x_batter_bb_rate"] = X["ump_bb9_avg_10"] * X["bb_rate_avg_10"]
    if "ump_bb9_avg_10" in X.columns and "bb_avg_10" in X.columns:
        X["ump_bb9_x_bb_avg_10"] = X["ump_bb9_avg_10"] * X["bb_avg_10"]
    # Pitcher strikeout model: ump K9 × pitcher K rate (wide-zone ump amplifies K ability)
    if "ump_k9_avg_10" in X.columns and "k_pct_5" in X.columns:
        X["ump_k9_x_sp_k_pct"] = X["ump_k9_avg_10"] * X["k_pct_5"]
    if "ump_k9_avg_10" in X.columns and "opp_k_pct_avg_10" in X.columns:
        X["ump_k9_x_opp_k_pct"] = X["ump_k9_avg_10"] * X["opp_k_pct_avg_10"]
    # Statcast-based K interactions: ump tendencies × pitch-arsenal quality metrics
    if "ump_k9_avg_10" in X.columns:
        if "sc_sp_disc_whiff_pct" in X.columns:
            X["ump_k9_x_sp_whiff_pct"] = X["ump_k9_avg_10"] * X["sc_sp_disc_whiff_pct"]
        if "sc_sp_sl_whiff_pct" in X.columns:
            X["ump_k9_x_sp_sl_whiff_pct"] = X["ump_k9_avg_10"] * X["sc_sp_sl_whiff_pct"]
        if "k9_5" in X.columns:
            X["ump_k9_x_sp_k9_5"] = X["ump_k9_avg_10"] * X["k9_5"]
        # Triple interaction: wide-zone ump + high-K pitcher vs strikeout-prone lineup
        if "k_pct_5" in X.columns and "opp_k_pct_avg_10" in X.columns:
            X["ump_k9_x_sp_opp_k_triple"] = (
                X["ump_k9_avg_10"] * X["k_pct_5"] * X["opp_k_pct_avg_10"]
            )

    # ── Lineup slot ───────────────────────────────────────────────────────────
    if "batting_order_avg_10" in X.columns:
        X["is_top_of_order"] = (X["batting_order_avg_10"].fillna(5.0) <= 2.5).astype(int)
        X["is_cleanup"]      = X["batting_order_avg_10"].fillna(5.0).between(3.0, 5.0).astype(int)

    # ── Cross-season delta features ───────────────────────────────────────────
    # How much is the player's current-season performance deviating from their
    # cross-season (multi-year) baseline?  Positive = hot vs career trend.
    for stat, cs_col, in_col in [
        ("hits", "hits_avg_10_cs", "hits_avg_10"),
        ("tb",   "tb_avg_10_cs",   "tb_avg_10"),
        ("hr",   "hr_avg_10_cs",   "hr_avg_10"),
    ]:
        if cs_col in X.columns and in_col in X.columns:
            X[f"{stat}_cs_delta"] = X[in_col].fillna(X[cs_col]) - X[cs_col]

    # Prior-season vs cross-season baseline comparison
    # prev_hits_avg reflects full 162-game prior season; cs features are recent 10-game
    for stat, prev_col, cs_col in [
        ("hits", "prev_hits_avg", "hits_avg_10_cs"),
        ("tb",   "prev_tb_avg",   "tb_avg_10_cs"),
    ]:
        if prev_col in X.columns and cs_col in X.columns:
            # Positive = recent form better than full prior-season avg
            X[f"{stat}_cs_vs_prev"] = X[cs_col].fillna(X[prev_col]) - X[prev_col].fillna(X[cs_col])

    # ── Reliability-weighted platoon splits ──────────────────────────────────
    # Dampens vs-hand split stats for batters with few PA against the relevant
    # pitcher handedness.  Uses the OHE'd opp_sp_hand_L/R columns to identify
    # which split is active today, then scales by min(n_games / 20, 1.0).
    if "n_games_vs_lhp_40" in X.columns and "n_games_vs_rhp_40" in X.columns:
        n_vs_hand = pd.Series(0.0, index=X.index)
        if "opp_sp_hand_L" in X.columns:
            n_vs_hand += X["opp_sp_hand_L"].fillna(0.0) * X["n_games_vs_lhp_40"].fillna(0.0)
        if "opp_sp_hand_R" in X.columns:
            n_vs_hand += X["opp_sp_hand_R"].fillna(0.0) * X["n_games_vs_rhp_40"].fillna(0.0)
        rel = (n_vs_hand / 20.0).clip(upper=1.0)

        for stat_col, new_col in [
            ("hits_avg_40_vs_hand",   "hits_vs_hand_weighted"),
            ("tb_avg_40_vs_hand",     "tb_vs_hand_weighted"),
            ("k_rate_avg_40_vs_hand", "k_rate_vs_hand_weighted"),
            ("iso_avg_40_vs_hand",    "iso_vs_hand_weighted"),
        ]:
            if stat_col in X.columns:
                X[new_col] = X[stat_col].fillna(0.0) * rel

        for split_col, new_col in [
            ("hits_hand_split_40", "hits_hand_split_weighted"),
            ("tb_hand_split_40",   "tb_hand_split_weighted"),
        ]:
            if split_col in X.columns:
                n_min = pd.concat(
                    [X["n_games_vs_lhp_40"].fillna(0.0), X["n_games_vs_rhp_40"].fillna(0.0)],
                    axis=1,
                ).min(axis=1)
                X[new_col] = X[split_col].fillna(0.0) * (n_min / 15.0).clip(upper=1.0)

    # ── Batter vs specific pitcher H2H career stats (MLB015) ─────────────────
    if "h2h_games" in X.columns:
        _games = pd.to_numeric(X["h2h_games"], errors="coerce").fillna(0.0)

        # Reliability weight: 0→1 as h2h_games grows 0→5
        # Prevents 1-2 game samples from swamping generic features
        _rel = (_games / 5.0).clip(upper=1.0)
        X["h2h_reliability"] = _rel

        # OPS computed here (sum of OBP + SLG from SQL cols)
        if "h2h_obp" in X.columns and "h2h_slg" in X.columns:
            X["h2h_ops"] = (
                pd.to_numeric(X["h2h_obp"], errors="coerce").fillna(0.0)
                + pd.to_numeric(X["h2h_slg"], errors="coerce").fillna(0.0)
            )

        # Reliability-weighted stats (0 when no history, full value at 5+ games)
        for _raw, _weighted in [
            ("h2h_ba",     "h2h_ba_weighted"),
            ("h2h_ops",    "h2h_ops_weighted"),
            ("h2h_k_rate", "h2h_k_rate_weighted"),
            ("h2h_iso",    "h2h_iso_weighted"),
        ]:
            if _raw in X.columns:
                X[_weighted] = (
                    pd.to_numeric(X[_raw], errors="coerce").fillna(0.0) * _rel
                )

        # Delta vs batter's rolling baseline — "is this pitcher harder/easier than average?"
        if "h2h_ops" in X.columns and "obp_avg_10" in X.columns and "iso_avg_10" in X.columns:
            _ops_baseline = (
                pd.to_numeric(X["obp_avg_10"], errors="coerce").fillna(0.0)
                + pd.to_numeric(X["iso_avg_10"], errors="coerce").fillna(0.0)
            )
            X["h2h_ops_delta"] = (
                pd.to_numeric(X["h2h_ops"], errors="coerce").fillna(0.0) - _ops_baseline
            ) * _rel  # only meaningful when there's history

        if "h2h_k_rate" in X.columns and "k_rate_avg_10" in X.columns:
            _k_baseline = pd.to_numeric(X["k_rate_avg_10"], errors="coerce").fillna(0.0)
            X["h2h_k_rate_delta"] = (
                pd.to_numeric(X["h2h_k_rate"], errors="coerce").fillna(0.0) - _k_baseline
            ) * _rel

        # Binary flag: 5+ matchup games = meaningful H2H sample
        X["h2h_has_history"] = (_games >= 5.0).astype(int)

    # ── Home/Away venue-matched splits ───────────────────────────────────────
    # For each stat, select the home or away rolling average based on today's location.
    # Falls back to overall avg_10 when the venue-specific split is NULL (few games).
    for _stat in ("hits", "tb", "hr", "bb"):
        _home_col = f"{_stat}_home_avg_20"
        _away_col = f"{_stat}_away_avg_20"
        _fallback = f"{_stat}_avg_10"
        if _home_col in X.columns and _away_col in X.columns and "is_home" in X.columns:
            _is_home = X["is_home"].astype(float).fillna(0.0)
            _fb = X[_fallback].fillna(0.0) if _fallback in X.columns else 0.0
            X[f"{_stat}_venue_adj"] = np.where(
                _is_home > 0.5,
                X[_home_col].fillna(_fb),
                X[_away_col].fillna(_fb),
            )
            X[f"{_stat}_home_away_split"] = (
                X[_home_col].fillna(0.0) - X[_away_col].fillna(0.0)
            )

    # ── Statcast batted-ball features (batter) ─────────────────────────────
    # These are the strongest HR/TB predictors available — barrel rate and
    # hard-hit rate directly correlate with extra-base hit outcomes.

    # Barrel rate × park HR factor: elite power in a hitter's park
    if "sc_barrel_rate" in X.columns and "park_hr_factor" in X.columns:
        X["sc_barrel_x_park_hr"] = X["sc_barrel_rate"].fillna(0.0) * X["park_hr_factor"].fillna(1.0)

    # Hard hit % × flyball %: hard + elevated = HR/TB potential
    if "sc_hard_hit_pct" in X.columns and "sc_fb_pct" in X.columns:
        _fb_clipped = pd.to_numeric(X["sc_fb_pct"], errors="coerce").clip(upper=100.0).fillna(0.0)
        X["sc_hard_hit_x_fb"] = X["sc_hard_hit_pct"].fillna(0.0) * _fb_clipped / 100.0

    # Exit velocity vs league average (~88 mph): how much above/below
    if "sc_avg_exit_velo" in X.columns:
        X["sc_exit_velo_above_avg"] = X["sc_avg_exit_velo"].fillna(88.0) - 88.0

    # xSLG - actual SLG proxy: how much underlying power vs what rolling stats show
    if "sc_xslg" in X.columns and "iso_avg_10" in X.columns:
        # xSLG ~ xBA + xISO; compare expected power to observed power (ISO)
        X["sc_xslg_vs_iso"] = X["sc_xslg"].fillna(0.0) - X["iso_avg_10"].fillna(0.0)

    # xwOBA vs rolling avg: overall quality-of-contact indicator
    if "sc_xwoba" in X.columns and "avg_avg_10" in X.columns:
        X["sc_xwoba_vs_avg"] = X["sc_xwoba"].fillna(0.0) - X["avg_avg_10"].fillna(0.0)

    # xISO as direct power proxy (stronger than ISO from box scores)
    if "sc_xiso" in X.columns and "park_hr_factor" in X.columns:
        X["sc_xiso_park_adj"] = X["sc_xiso"].fillna(0.0) * X["park_hr_factor"].fillna(1.0)

    # Launch angle sweet spot × exit velocity: optimal HR conditions
    if "sc_sweet_spot_pct" in X.columns and "sc_avg_exit_velo" in X.columns:
        X["sc_sweet_spot_x_velo"] = (
            X["sc_sweet_spot_pct"].fillna(0.0) / 100.0
            * (X["sc_avg_exit_velo"].fillna(88.0) - 85.0)
        )

    # ── Statcast matchup features (batter vs opposing SP's batted-ball profile) ──
    # Batter barrel rate vs SP barrel-allowed rate: mismatch detector
    if "sc_barrel_rate" in X.columns and "opp_sp_sc_barrel_rate" in X.columns:
        X["sc_barrel_matchup"] = (
            X["sc_barrel_rate"].fillna(0.0) - X["opp_sp_sc_barrel_rate"].fillna(0.0)
        )

    # Batter hard-hit vs SP hard-hit-allowed: contact quality mismatch
    if "sc_hard_hit_pct" in X.columns and "opp_sp_sc_hard_hit_pct" in X.columns:
        X["sc_hard_hit_matchup"] = (
            X["sc_hard_hit_pct"].fillna(0.0) - X["opp_sp_sc_hard_hit_pct"].fillna(0.0)
        )

    # SP groundball% affects TB/HR potential: high GB% pitcher = fewer extra-base hits
    if "opp_sp_sc_gb_pct" in X.columns:
        X["opp_sp_gb_tendency"] = X["opp_sp_sc_gb_pct"].fillna(45.0) / 100.0

    # SP xwOBA-against: overall hittability (higher = easier to hit)
    if "opp_sp_sc_xwoba" in X.columns and "sc_xwoba" in X.columns:
        X["sc_hittability_combo"] = X["sc_xwoba"].fillna(0.0) + X["opp_sp_sc_xwoba"].fillna(0.0)

    # ── Statcast batter × pitcher multiplicative cross-products ───────────────
    # Additive matchup diffs above capture direction; these products capture the
    # conditional magnitude (e.g., elite barrel rate means nothing vs extreme GB SP).
    # Barrel rate × (1 - SP GB%): power batter vs fly-ball-inducing pitcher
    if "sc_barrel_rate" in X.columns and "opp_sp_sc_gb_pct" in X.columns:
        X["sc_barrel_x_opp_fb"] = (
            X["sc_barrel_rate"].fillna(0.0)
            * (1.0 - X["opp_sp_sc_gb_pct"].fillna(45.0) / 100.0)
        )
    # Hard-hit product: both batter contact quality AND pitcher-allowed contact matter
    if "sc_hard_hit_pct" in X.columns and "opp_sp_sc_hard_hit_pct" in X.columns:
        X["sc_hard_hit_x_opp"] = (
            X["sc_hard_hit_pct"].fillna(0.0) / 100.0
            * X["opp_sp_sc_hard_hit_pct"].fillna(0.0) / 100.0
        )
    # xSLG × SP xwOBA-against: expected slugging in context of pitcher hittability
    if "sc_xslg" in X.columns and "opp_sp_sc_xwoba" in X.columns:
        X["sc_xslg_x_opp_xwoba"] = (
            X["sc_xslg"].fillna(0.0) * X["opp_sp_sc_xwoba"].fillna(0.0)
        )
    # Batter xwOBA × SP xwOBA-against: combined wOBA matchup quality
    if "sc_xwoba" in X.columns and "opp_sp_sc_xwoba" in X.columns:
        X["sc_xwoba_x_opp_xwoba"] = (
            X["sc_xwoba"].fillna(0.0) * X["opp_sp_sc_xwoba"].fillna(0.0)
        )

    # ── HR environment: pitcher HR/9 × batter fly-ball tendencies ────────────
    # opp_sp_hr9_5 is the single strongest predictor of pitcher HR propensity.
    # ~1.0 HR/9 is league average; deviation from that is the signal.
    _LEAGUE_HR9 = 1.0
    if "opp_sp_hr9_5" in X.columns:
        _hr9 = pd.to_numeric(X["opp_sp_hr9_5"], errors="coerce").fillna(_LEAGUE_HR9)
        X["opp_sp_hr9_vs_avg"] = _hr9 - _LEAGUE_HR9   # positive = pitcher gives up more HRs than avg

        # Flyball batter vs HR-prone pitcher: highest HR probability combination
        # clip(upper=100) guards against corrupted raw values > 100 in Statcast table
        if "sc_fb_pct" in X.columns:
            _fb = pd.to_numeric(X["sc_fb_pct"], errors="coerce").clip(upper=100.0).fillna(30.0) / 100.0
            X["fb_x_opp_hr9"] = _fb * _hr9

        # Barrel rate × pitcher HR/9: elite contact quality meets HR-prone pitcher
        if "sc_barrel_rate" in X.columns:
            X["barrel_x_opp_hr9"] = X["sc_barrel_rate"].fillna(0.0) * _hr9

        # xISO × pitcher HR/9: expected isolated power in a HR-friendly matchup
        if "sc_xiso" in X.columns:
            X["xiso_x_opp_hr9"] = X["sc_xiso"].fillna(0.0) * _hr9

        # Pitcher flyball rate allowed × batter flyball rate: both flying-ball types = max HR probability
        # opp_sp_sc_fb_pct is the pitcher's BBE-weighted flyball% allowed (capped at 100 in SQL)
        if "opp_sp_sc_fb_pct" in X.columns and "sc_fb_pct" in X.columns:
            _opp_fb = pd.to_numeric(X["opp_sp_sc_fb_pct"], errors="coerce").clip(upper=100.0).fillna(35.0) / 100.0
            _bat_fb = pd.to_numeric(X["sc_fb_pct"],        errors="coerce").clip(upper=100.0).fillna(30.0) / 100.0
            X["fb_batter_x_pitcher_fb"] = _bat_fb * _opp_fb  # joint flyball probability

        # AB-weighted HR rate × pitcher HR/9: volume-corrected power vs HR-prone pitcher
        if "hr_rate_cumul_10" in X.columns:
            X["hr_rate_cumul_x_opp_hr9"] = (
                pd.to_numeric(X["hr_rate_cumul_10"], errors="coerce").fillna(0.0) * _hr9
            )

    # ── Launch angle HR zone × exit velocity ─────────────────────────────────
    # HR sweet spot: ~25-35° launch angle + 90+ mph EV
    if "sc_avg_launch_angle" in X.columns and "sc_avg_exit_velo" in X.columns:
        _la = pd.to_numeric(X["sc_avg_launch_angle"], errors="coerce").fillna(20.0)
        _ev = pd.to_numeric(X["sc_avg_exit_velo"], errors="coerce").fillna(86.0)
        X["sc_in_hr_la_zone"] = ((_la >= 25.0) & (_la <= 35.0)).astype(int)
        # Continuous: EV scaled by how close to ideal launch angle (peaks at 30°)
        _la_penalty = 1.0 - ((_la - 30.0).abs().clip(upper=15.0) / 15.0)
        X["sc_la_ev_composite"] = _ev * _la_penalty / 100.0

    # ── HR volatility: coefficient of variation (hot ceiling signal) ──────────
    # hr_sd_10 / hr_avg_10 = how erratic is this batter's HR output?
    # High CV = feast-or-famine (good for OVER bets at high lines)
    if "hr_sd_10" in X.columns and "hr_avg_10" in X.columns:
        _avg = X["hr_avg_10"].replace(0, np.nan)
        X["hr_cv_10"] = X["hr_sd_10"].fillna(0.0) / _avg
    if "hr_sd_5" in X.columns and "hr_avg_5" in X.columns:
        _avg5 = X["hr_avg_5"].replace(0, np.nan)
        X["hr_cv_5"] = X["hr_sd_5"].fillna(0.0) / _avg5
    # HR ceiling: max HR in last 10 games vs rolling avg (how often does batter go off?)
    if "hr_max_10" in X.columns and "hr_avg_10" in X.columns:
        X["hr_ceiling_ratio"] = X["hr_max_10"].fillna(0.0) / X["hr_avg_10"].clip(lower=0.01)

    # ── Statcast pitcher features (for strikeout model) ──────────────────────
    # Pitcher barrel-allowed rate: low = dominant stuff, correlates with K potential
    if "sc_barrel_rate" in X.columns and "k_pct_5" in X.columns:
        X["sc_barrel_vs_k_rate"] = X["sc_barrel_rate"].fillna(0.0) / (X["k_pct_5"].fillna(0.2) + 0.01)

    # SP hard-hit-against × opponent K-proneness
    if "sc_hard_hit_pct" in X.columns and "opp_k_pct_avg_10" in X.columns:
        X["sc_hard_hit_vs_opp_k"] = X["sc_hard_hit_pct"].fillna(0.0) * X["opp_k_pct_avg_10"].fillna(0.0)

    # ── Shared ───────────────────────────────────────────────────────────────
    # Back-to-back flag (rest_days ≤ 1)
    if "rest_days" in X.columns:
        X["is_b2b"] = (X["rest_days"].fillna(10.0) <= 1).astype(int)

    # ── 4a: Batter Usage Regularity + Batting Order Signals ──────────────────
    # Regular starters with ≥7 games played and ≥3 AB/game get full PA exposure.
    if "ab_avg_10" in X.columns and "n_games_prev_10" in X.columns:
        _ab = pd.to_numeric(X["ab_avg_10"], errors="coerce").fillna(0.0)
        _g  = pd.to_numeric(X["n_games_prev_10"], errors="coerce").fillna(0.0)
        X["is_regular_starter"] = ((_g >= 7.0) & (_ab >= 3.0)).astype(int)
        X["ab_per_game_played"]  = _ab / _g.clip(lower=0.5)
    if "batting_order_avg_10" in X.columns:
        _slot = pd.to_numeric(X["batting_order_avg_10"], errors="coerce").fillna(5.0)
        X["is_top_of_order"]    = (_slot <= 4.0).astype(int)
        X["is_bottom_of_order"] = (_slot >= 7.0).astype(int)
        X["pa_frequency_factor"] = (10.0 - _slot).clip(lower=1.0) / 9.0

    # ── 4b: Lineup Slot × Weak Bullpen Interaction ────────────────────────────
    # Top-of-order batters are more likely to bat in later innings against the bullpen.
    if "opp_bp_era_5" in X.columns and "batting_order_avg_10" in X.columns:
        _bp   = pd.to_numeric(X["opp_bp_era_5"], errors="coerce").fillna(4.0).clip(lower=1.0)
        _slot = pd.to_numeric(X["batting_order_avg_10"], errors="coerce").fillna(5.0)
        X["top_order_x_weak_bp"] = (_slot <= 3.5).astype(float) * (_bp / 4.0)

    # ── 4c: K9 Bayesian Shrinkage (pitcher K model) ───────────────────────────
    # Shrink pitcher K stats toward league averages for small sample sizes.
    LEAGUE_K9   = 8.5
    LEAGUE_KPCT = 0.22
    if "k9_5" in X.columns and "ip_avg_5" in X.columns and "starts_in_window_5" in X.columns:
        _ip5 = (pd.to_numeric(X["ip_avg_5"], errors="coerce").fillna(5.0)
                * pd.to_numeric(X["starts_in_window_5"], errors="coerce").fillna(1.0))
        _k9  = pd.to_numeric(X["k9_5"], errors="coerce").fillna(LEAGUE_K9)
        X["k9_shrunk"] = (_k9 * _ip5 + LEAGUE_K9 * 20.0) / (_ip5 + 20.0)
    if "k_pct_5" in X.columns and "ip_avg_5" in X.columns and "starts_in_window_5" in X.columns:
        _ip5  = (pd.to_numeric(X["ip_avg_5"], errors="coerce").fillna(5.0)
                 * pd.to_numeric(X["starts_in_window_5"], errors="coerce").fillna(1.0))
        _kpct = pd.to_numeric(X["k_pct_5"], errors="coerce").fillna(LEAGUE_KPCT)
        X["k_pct_shrunk"] = (_kpct * _ip5 + LEAGUE_KPCT * 20.0) / (_ip5 + 20.0)

    # ── 4d: Catcher Framing × Pitcher Handedness ─────────────────────────────
    # LHP benefit less from framing (their delivery angle differs); scale accordingly.
    if "catcher_framing_rate" in X.columns:
        _fr = pd.to_numeric(X["catcher_framing_rate"], errors="coerce").fillna(0.0)
        if "pitcher_hand_l" in X.columns:
            _lhp = pd.to_numeric(X["pitcher_hand_l"], errors="coerce").fillna(0.0)
            X["framing_hand_benefit"] = _fr * (1.0 - 0.25 * _lhp)
        else:
            X["framing_hand_benefit"] = _fr

    # ── Pitcher vs opponent lineup K-rate variance ────────────────────────────
    # Lineup K-rate CV distinguishes "all contact" vs "mixed" lineups.
    # A high-K pitcher facing a HIGH-CV lineup has more K opportunities than
    # the raw avg K rate suggests (variance creates matchup-specific upside).
    if "k_pct_5" in X.columns and "opp_lineup_k_pct_cv" in X.columns:
        _sp_k = pd.to_numeric(X["k_pct_5"], errors="coerce").fillna(0.22)
        _cv   = pd.to_numeric(X["opp_lineup_k_pct_cv"], errors="coerce").fillna(0.0)
        X["sp_k_pct_x_opp_cv"] = _sp_k * (1.0 + _cv)
    if "opp_lineup_k_pct_std" in X.columns:
        # Keep raw std as standalone feature for the model to weight independently
        X["opp_lineup_k_pct_std"] = pd.to_numeric(
            X["opp_lineup_k_pct_std"], errors="coerce"
        ).fillna(0.05)

    return X


def build_fd_parlay_url(links) -> str | None:
    """Combine individual FanDuel addToBetslip links into a multi-leg parlay URL."""
    from urllib.parse import urlparse, parse_qs
    legs = []
    for link in (links or []):
        if not link:
            continue
        try:
            qs = parse_qs(urlparse(link).query)
            m = qs.get("marketId", [None])[0]
            s = qs.get("selectionId", [None])[0]
            if m and s:
                legs.append((m, s))
        except Exception:
            continue
    if not legs:
        return None
    base = "https://sportsbook.fanduel.com/addToBetslip"
    params = "&".join(f"marketId[{i}]={m}&selectionId[{i}]={s}" for i, (m, s) in enumerate(legs))
    return f"{base}?{params}"
