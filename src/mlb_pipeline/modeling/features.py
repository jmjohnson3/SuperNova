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
    # Exponent 1.82 is the standard baseball value (vs NBA's 14).
    # win_pct_pythag = RF^1.82 / (RF^1.82 + RA^1.82)
    # Uses runs_avg_10 as offensive proxy; opponent's runs_avg_10 as defensive proxy.
    _EXP = 1.82

    home_rf = X.get("home_runs_avg_10", pd.Series(dtype=float))
    away_rf = X.get("away_runs_avg_10", pd.Series(dtype=float))

    def _pythag(rf: pd.Series, ra: pd.Series) -> pd.Series:
        rf_pow = rf.clip(lower=0.01) ** _EXP
        ra_pow = ra.clip(lower=0.01) ** _EXP
        return rf_pow / (rf_pow + ra_pow)

    if "home_runs_avg_10" in X.columns and "away_runs_avg_10" in X.columns:
        # Home offense (home_rf) vs home defense (away_rf = runs the away team scores,
        # which equals runs allowed by the home team)
        X["home_pythag"] = _pythag(home_rf, away_rf)
        X["away_pythag"] = _pythag(away_rf, home_rf)
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

    # Batter × Opponent SP interaction: batters facing strikeout-heavy pitchers get fewer hits
    if "hits_avg_10" in X.columns and "opp_sp_era_5" in X.columns:
        X["hits_vs_opp_era"] = X["hits_avg_10"] * (X["opp_sp_era_5"] / 4.0)
    if "tb_avg_10" in X.columns and "opp_sp_fip_5" in X.columns:
        X["tb_vs_opp_fip"] = X["tb_avg_10"] * (X["opp_sp_fip_5"] / 4.0)

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

    # ── Umpire walk-tendency ──────────────────────────────────────────────────
    if "ump_bb9_avg_10" in X.columns and "bb_rate_avg_10" in X.columns:
        X["ump_bb9_x_batter_bb_rate"] = X["ump_bb9_avg_10"] * X["bb_rate_avg_10"]

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

    # ── Shared ───────────────────────────────────────────────────────────────
    # Back-to-back flag (rest_days ≤ 1)
    if "rest_days" in X.columns:
        X["is_b2b"] = (X["rest_days"].fillna(10.0) <= 1).astype(int)

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
