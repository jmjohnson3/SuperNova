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
