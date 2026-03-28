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
