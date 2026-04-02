# src/mlb_pipeline/modeling/train_player_prop_models.py
"""
Train MLB player prop models: pitcher strikeouts, batter hits, batter total bases.

Three separate XGBoost + LightGBM models (50/50 ensemble where LGB is available).
Walk-forward CV: min_train_days=120, test_window=14, step=14.

Artifacts saved to models/player_props/:
  strikeouts_xgb.json, hits_xgb.json, total_bases_xgb.json
  lgb_strikeouts.txt, lgb_hits.txt, lgb_total_bases.txt
  feature_columns_pitchers.json, feature_columns_batters.json
  feature_medians_pitchers.json, feature_medians_batters.json
  backtest_mae.json
"""
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features

log = logging.getLogger("mlb_pipeline.modeling.train_player_prop_models")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class TrainConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR

    # Walk-forward (same cadence as MLB game model)
    min_train_days: int = 120
    test_window_days: int = 14
    step_days: int = 14

    # XGBoost
    n_estimators: int = 2000
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    gamma: float = 0.1
    reg_alpha: float = 0.5
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 50
    random_state: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Training SQL
# ─────────────────────────────────────────────────────────────────────────────

SQL_PITCHER_TRAIN = """
SELECT
    p.season,
    p.game_slug,
    p.game_date_et,
    p.player_id,
    p.team_abbr,
    CASE WHEN p.team_abbr = g.home_team_abbr THEN TRUE ELSE FALSE END AS is_home,
    -- SP rolling features (leakage-safe: exclude current game)
    p.ip_avg_5,
    p.k_pct_5,
    p.k9_5,
    p.era_5,
    p.whip_5,
    p.fip_5,
    p.bb_pct_5,
    p.hr9_5,
    p.k_pct_10,
    p.k9_10,
    p.era_10,
    p.whip_10,
    p.fip_10,
    p.starts_in_window_5,
    p.starts_in_window_10,
    p.last_start_k,
    p.last_start_ip,
    -- Group B: SP rest + home/away performance splits (MLB003)
    p.days_since_last_start AS sp_days_since_last_start,
    p.is_short_rest,
    p.era_home_10,
    p.era_away_10,
    p.k9_home_10,
    p.k9_away_10,
    p.fip_home_10,
    p.fip_away_10,
    -- Opponent team batting context (how K-prone are the opposing batters?)
    ob.k_pct_avg_10    AS opp_k_pct_avg_10,
    ob.bb_pct_avg_10   AS opp_bb_pct_avg_10,
    ob.avg_avg_10      AS opp_avg_avg_10,
    ob.hr_avg_10       AS opp_hr_avg_10,
    ob.iso_avg_10      AS opp_iso_avg_10,
    ob.slg_avg_10      AS opp_slg_avg_10,
    -- Park factors
    bf.run_factor      AS park_run_factor,
    bf.hr_factor       AS park_hr_factor,
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
    -- Pitcher handedness
    ph.pitch_hand                                                AS pitcher_hand,
    -- Target
    pgl.strikeouts_pitcher AS strikeouts
FROM features.mlb_pitcher_rolling_mat p
JOIN raw.mlb_games g
    ON g.game_slug = p.game_slug
JOIN raw.mlb_player_gamelogs pgl
    ON pgl.game_slug = p.game_slug
    AND pgl.player_id = p.player_id
LEFT JOIN features.mlb_team_batting_rolling_mat ob
    ON ob.game_slug = p.game_slug
    AND ob.team_abbr = CASE
        WHEN p.team_abbr = g.home_team_abbr THEN g.away_team_abbr
        ELSE g.home_team_abbr
    END
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = g.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id   = g.venue_id
LEFT JOIN raw.mlb_player_handedness ph ON ph.player_id = p.player_id
WHERE g.status = 'final'
  AND pgl.innings_pitched >= 3.0
  AND p.starts_in_window_10 >= 3
  AND pgl.strikeouts_pitcher IS NOT NULL
ORDER BY p.game_date_et, p.game_slug, p.player_id
"""

SQL_BATTER_TRAIN = """
SELECT
    b.season,
    b.game_slug,
    b.game_date_et,
    b.player_id,
    b.team_abbr,
    b.is_home,
    b.rest_days,
    -- Batter rolling features
    b.hits_avg_5,  b.hits_avg_10,  b.hits_avg_20,  b.hits_sd_10,
    b.hr_avg_5,    b.hr_avg_10,    b.hr_avg_20,
    b.tb_avg_5,    b.tb_avg_10,    b.tb_avg_20,    b.tb_sd_10,
    b.ab_avg_5,    b.ab_avg_10,
    b.avg_avg_10,  b.k_rate_avg_10, b.bb_rate_avg_10, b.iso_avg_10,
    b.hr_rate_avg_5, b.hr_rate_avg_10,
    b.n_games_prev_10,
    -- Opponent SP context
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
    -- Lineup slot
    b.batting_order_avg_5,
    b.batting_order_avg_10,
    -- Batter + opponent SP handedness (OHE'd by _prep_X)
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
    -- Sample sizes (model learns when to trust the split)
    bvh.n_games_vs_lhp_40,
    bvh.n_games_vs_rhp_40,
    -- Cross-season rolling features (MLB013) — populated even in early season
    bcs.n_games_prev_10_cs,
    bcs.hits_avg_10_cs,  bcs.hits_avg_20_cs,
    bcs.tb_avg_10_cs,    bcs.tb_avg_20_cs,
    bcs.hr_avg_10_cs,    bcs.hr_rate_avg_10_cs,
    bcs.ab_avg_10_cs,
    bcs.k_rate_avg_10_cs, bcs.bb_rate_avg_10_cs, bcs.iso_avg_10_cs,
    -- Prior full-season stats (MLB014) — stable 162-game prior
    pss.prev_games,
    pss.prev_hits_avg,  pss.prev_tb_avg,  pss.prev_hr_avg,
    pss.prev_ab_avg,    pss.prev_k_rate,  pss.prev_bb_rate,
    pss.prev_iso,       pss.prev_hr_rate,
    -- Career H2H vs this specific SP (MLB015) — leakage-safe via matview window
    h2h.h2h_games,
    h2h.h2h_ba,
    h2h.h2h_obp,
    h2h.h2h_slg,
    h2h.h2h_k_rate,
    h2h.h2h_iso,
    -- Targets
    gl.hits        AS hits,
    gl.total_bases AS total_bases,
    gl.home_runs   AS home_runs,
    gl.walks_batter AS walks_batter
FROM features.mlb_player_batting_rolling_mat b
JOIN raw.mlb_games g
    ON g.game_slug = b.game_slug
JOIN raw.mlb_player_gamelogs gl
    ON gl.game_slug = b.game_slug
    AND gl.player_id = b.player_id
-- Opposing starting pitcher
LEFT JOIN raw.mlb_starting_pitchers sp
    ON sp.game_slug = b.game_slug
    AND sp.team_abbr = CASE
        WHEN b.team_abbr = g.home_team_abbr THEN g.away_team_abbr
        ELSE g.home_team_abbr
    END
-- SP rolling stats for this game
LEFT JOIN features.mlb_pitcher_rolling_mat sp_r
    ON sp_r.game_slug = b.game_slug
    AND sp_r.player_id = sp.player_id
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = g.home_team_abbr
-- Home plate umpire rolling stats
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = b.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN features.mlb_umpire_rolling_mat ur
    ON ur.game_slug = b.game_slug AND ur.umpire_id = gu.umpire_id
-- Weather (dome-safe)
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = g.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id   = g.venue_id
-- Opponent SP handedness
LEFT JOIN raw.mlb_player_handedness opp_ph
    ON opp_ph.player_id = sp.player_id
-- Batter's own handedness
LEFT JOIN raw.mlb_player_handedness bh
    ON bh.player_id = b.player_id
-- Batter vs hand rolling stats (leakage-safe: joined by exact game_slug)
LEFT JOIN features.mlb_batting_vs_hand_mat bvh
    ON bvh.game_slug = b.game_slug
    AND bvh.player_id = b.player_id
-- Cross-season rolling stats (MLB013) — same game_slug join, no season boundary
LEFT JOIN features.mlb_player_batting_rolling_cross_mat bcs
    ON bcs.game_slug = b.game_slug
    AND bcs.player_id = b.player_id
-- Prior full-season stats (MLB014) — one season back
LEFT JOIN features.mlb_player_prev_season_stats_mat pss
    ON pss.player_id = b.player_id
    AND pss.season = CASE g.season
        WHEN '2025-regular' THEN '2024-regular'
        WHEN '2026-regular' THEN '2025-regular'
        ELSE NULL
    END
-- Career H2H stats vs today's specific SP (MLB015, leakage-safe via matview window)
LEFT JOIN features.mlb_batter_vs_sp_mat h2h
    ON  h2h.game_slug  = b.game_slug
    AND h2h.batter_id  = b.player_id
    AND h2h.pitcher_id = sp.player_id
WHERE g.status = 'final'
  AND b.ab_avg_10 >= 2.5
  AND b.n_games_prev_10 >= 3
  AND gl.hits IS NOT NULL
  AND gl.total_bases IS NOT NULL
ORDER BY b.game_date_et, b.game_slug, b.player_id
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _prep_X(df: pd.DataFrame, target_cols: List[str], meta_cols: List[str]) -> pd.DataFrame:
    """Drop meta / target cols, OHE season, coerce to numeric, add derived features."""
    drop_cols = set(target_cols) | set(meta_cols)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Season OHE
    if "season" in X.columns:
        X = pd.get_dummies(X, columns=["season"], drop_first=False, dummy_na=False)

    # is_home → int
    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric(X)

    # Remaining non-numeric (shouldn't happen after above)
    bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if bad:
        X = pd.get_dummies(X, columns=bad, drop_first=False, dummy_na=True)

    X = add_player_prop_derived_features(X)
    return X


def fit_medians(X: pd.DataFrame) -> Dict[str, float]:
    return {str(k): float(v) for k, v in X.median(numeric_only=True).items()}


def apply_medians(X: pd.DataFrame, medians: Dict[str, float], cols: List[str]) -> pd.DataFrame:
    X2 = X.reindex(columns=cols)
    for c, m in medians.items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(m)
    return X2.fillna(0.0)


def _build_xgb(cfg: TrainConfig, n_est: Optional[int] = None,
               early_stop: bool = True,
               objective: str = "reg:absoluteerror") -> XGBRegressor:
    eval_metric = "poisson-nloglik" if objective == "count:poisson" else "mae"
    p = dict(
        n_estimators=n_est or cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    if early_stop:
        p["early_stopping_rounds"] = cfg.early_stopping_rounds
    return XGBRegressor(**p)


def _build_lgb(n_est: int = 2000, early_stop: bool = True,
               objective: str = "regression_l1"):
    if not _HAS_LGB:
        return None
    metric = "poisson" if objective == "poisson" else "mae"
    p = dict(
        n_estimators=n_est,
        num_leaves=31,
        learning_rate=0.05,
        objective=objective,
        metric=metric,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if early_stop:
        p["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    return lgb.LGBMRegressor(**p)


def _walk_forward_folds(
    df: pd.DataFrame,
    min_train_days: int,
    test_window_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = pd.to_datetime(df["game_date_et"])
    start = dates.min().normalize()
    end = dates.max().normalize() + pd.Timedelta(days=1)
    first_end = start + pd.Timedelta(days=min_train_days)
    if first_end >= end:
        return []
    folds = []
    train_end = first_end
    while True:
        test_end = train_end + pd.Timedelta(days=test_window_days)
        if test_end > end:
            break
        folds.append((train_end, test_end))
        train_end = train_end + pd.Timedelta(days=step_days)
    return folds


def _run_walk_forward(
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    medians: Dict[str, float],
    cfg: TrainConfig,
    stat_name: str,
    objective: str = "reg:absoluteerror",
) -> Tuple[float, float]:
    """Run walk-forward CV. Returns (mae, p68_ci)."""
    folds = _walk_forward_folds(df, cfg.min_train_days, cfg.test_window_days, cfg.step_days)
    if not folds:
        log.warning("No walk-forward folds for %s", stat_name)
        return float("nan"), float("nan")

    oof_preds, oof_actual = [], []
    best_iters = []

    for train_end, test_end in folds:
        tr_mask = pd.to_datetime(df["game_date_et"]) < train_end
        te_mask = (pd.to_datetime(df["game_date_et"]) >= train_end) & \
                  (pd.to_datetime(df["game_date_et"]) < test_end)
        if tr_mask.sum() < 50 or te_mask.sum() == 0:
            continue

        X_tr = apply_medians(X_raw[tr_mask], medians, feature_cols)
        X_te = apply_medians(X_raw[te_mask], medians, feature_cols)
        y_tr = y[tr_mask]
        y_te = y[te_mask]

        # XGBoost with early stopping on last 15% of train
        cutoff = X_tr.index[int(len(X_tr) * 0.85)]
        fit_mask = X_tr.index < cutoff
        eval_mask = X_tr.index >= cutoff
        if fit_mask.sum() < 30 or eval_mask.sum() == 0:
            fit_mask = slice(None)
            eval_mask = None

        lgb_obj = "poisson" if objective == "count:poisson" else "regression_l1"
        xgb = _build_xgb(cfg, early_stop=True, objective=objective)
        if eval_mask is not None:
            xgb.fit(
                X_tr[fit_mask], y_tr[fit_mask],
                eval_set=[(X_tr[eval_mask], y_tr[eval_mask])],
                verbose=False,
            )
        else:
            xgb.fit(X_tr, y_tr, verbose=False)
        best_iters.append(xgb.best_iteration if hasattr(xgb, "best_iteration") and
                          xgb.best_iteration > 0 else cfg.n_estimators)

        preds = xgb.predict(X_te)

        if _HAS_LGB:
            lgb_model = _build_lgb(early_stop=True, objective=lgb_obj)
            if eval_mask is not None:
                lgb_model.fit(
                    X_tr[fit_mask], y_tr[fit_mask],
                    eval_set=[(X_tr[eval_mask], y_tr[eval_mask])],
                )
            else:
                lgb_model.fit(X_tr, y_tr)
            preds = (preds + lgb_model.predict(X_te)) / 2.0

        oof_preds.extend(preds)
        oof_actual.extend(y_te.values)

    if not oof_preds:
        return float("nan"), float("nan")

    oof_preds = np.array(oof_preds)
    oof_actual = np.array(oof_actual)
    mae = float(mean_absolute_error(oof_actual, oof_preds))
    errors = np.abs(oof_actual - oof_preds)
    p68 = float(np.percentile(errors, 68))  # ~1σ for MAE-like CI

    log.info(
        "Walk-forward %s | MAE=%.3f p68=%.3f | %d OOF rows, %d folds, best_iter~%d",
        stat_name, mae, p68,
        len(oof_actual), len(folds),
        int(np.median(best_iters)) if best_iters else cfg.n_estimators,
    )
    return mae, p68


def _fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainConfig,
    stat_name: str,
    n_estimators: Optional[int] = None,
    objective: str = "reg:absoluteerror",
) -> Tuple[XGBRegressor, Optional[object]]:
    """Fit final XGB (+LGB) on all data, no early stopping."""
    n_est = n_estimators or cfg.n_estimators
    lgb_obj = "poisson" if objective == "count:poisson" else "regression_l1"
    log.info("Fitting final %s XGB (n=%d rows, n_estimators=%d)", stat_name, len(X), n_est)
    xgb = _build_xgb(cfg, n_est=n_est, early_stop=False, objective=objective)
    xgb.fit(X, y, verbose=False)

    lgb_model = None
    if _HAS_LGB:
        log.info("Fitting final %s LGB", stat_name)
        lgb_model = _build_lgb(n_est=n_est, early_stop=False, objective=lgb_obj)
        lgb_model.fit(X, y)

    return xgb, lgb_model


# ─────────────────────────────────────────────────────────────────────────────
# Train pitcher model (strikeouts)
# ─────────────────────────────────────────────────────────────────────────────

_PITCHER_META = ["game_slug", "game_date_et", "player_id", "team_abbr"]
_PITCHER_TARGETS = ["strikeouts"]


def train_pitcher_models(cfg: TrainConfig) -> Dict:
    conn = psycopg2.connect(cfg.pg_dsn)
    df = pd.read_sql(SQL_PITCHER_TRAIN, conn)
    conn.close()

    if df.empty:
        raise RuntimeError("No pitcher training data returned.")

    df["game_date_et"] = pd.to_datetime(df["game_date_et"])
    log.info("Pitcher training data: %d rows, %s → %s",
             len(df), df["game_date_et"].min().date(), df["game_date_et"].max().date())

    y_k = df["strikeouts"].astype(float)
    X_raw = _prep_X(df, _PITCHER_TARGETS, _PITCHER_META)
    feature_cols = list(X_raw.columns)

    # Fit medians on all data (final model uses all data; walk-forward uses per-fold)
    medians = fit_medians(X_raw)
    X_filled = apply_medians(X_raw, medians, feature_cols)

    # Walk-forward evaluation
    wf_mae, wf_p68 = _run_walk_forward(
        df, X_raw, y_k, feature_cols, medians, cfg, "strikeouts"
    )

    # Final model
    xgb, lgb_model = _fit_final_model(X_filled, y_k, cfg, "strikeouts")

    # Save
    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb.save_model(str(model_dir / "strikeouts_xgb.json"))
    if lgb_model is not None:
        lgb_model.booster_.save_model(str(model_dir / "lgb_strikeouts.txt"))

    (model_dir / "feature_columns_pitchers.json").write_text(
        json.dumps(feature_cols), encoding="utf-8"
    )
    (model_dir / "feature_medians_pitchers.json").write_text(
        json.dumps(medians), encoding="utf-8"
    )

    return {"mae_strikeouts": wf_mae, "ci_strikeouts": wf_p68, "n_rows": len(df)}


# ─────────────────────────────────────────────────────────────────────────────
# Train batter models (hits + total_bases)
# ─────────────────────────────────────────────────────────────────────────────

_BATTER_META = ["game_slug", "game_date_et", "player_id", "team_abbr"]
_BATTER_TARGETS = ["hits", "total_bases", "home_runs", "walks_batter"]


def train_batter_models(cfg: TrainConfig) -> Dict:
    conn = psycopg2.connect(cfg.pg_dsn)
    df = pd.read_sql(SQL_BATTER_TRAIN, conn)
    conn.close()

    if df.empty:
        raise RuntimeError("No batter training data returned.")

    df["game_date_et"] = pd.to_datetime(df["game_date_et"])
    log.info("Batter training data: %d rows, %s → %s",
             len(df), df["game_date_et"].min().date(), df["game_date_et"].max().date())

    y_hits  = df["hits"].astype(float)
    y_tb    = df["total_bases"].astype(float)
    y_hr    = df["home_runs"].fillna(0).astype(float)
    y_walks = df["walks_batter"].fillna(0).astype(float)
    X_raw = _prep_X(df, _BATTER_TARGETS, _BATTER_META)
    feature_cols = list(X_raw.columns)

    medians = fit_medians(X_raw)
    X_filled = apply_medians(X_raw, medians, feature_cols)

    # Walk-forward for all four stats
    hits_mae,  hits_p68  = _run_walk_forward(df, X_raw, y_hits,  feature_cols, medians, cfg, "hits",        objective="count:poisson")
    tb_mae,    tb_p68    = _run_walk_forward(df, X_raw, y_tb,    feature_cols, medians, cfg, "total_bases", objective="count:poisson")
    hr_mae,    hr_p68    = _run_walk_forward(df, X_raw, y_hr,    feature_cols, medians, cfg, "home_runs",    objective="count:poisson")
    walks_mae, walks_p68 = _run_walk_forward(df, X_raw, y_walks, feature_cols, medians, cfg, "walks_batter")

    # Final models
    xgb_hits,  lgb_hits  = _fit_final_model(X_filled, y_hits,  cfg, "hits",        objective="count:poisson")
    xgb_tb,    lgb_tb    = _fit_final_model(X_filled, y_tb,    cfg, "total_bases", objective="count:poisson")
    xgb_hr,    lgb_hr    = _fit_final_model(X_filled, y_hr,    cfg, "home_runs",    objective="count:poisson")
    xgb_walks, lgb_walks = _fit_final_model(X_filled, y_walks, cfg, "walks_batter")

    # Save
    model_dir = cfg.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    xgb_hits.save_model(str(model_dir / "hits_xgb.json"))
    xgb_tb.save_model(str(model_dir / "total_bases_xgb.json"))
    xgb_hr.save_model(str(model_dir / "home_runs_xgb.json"))
    xgb_walks.save_model(str(model_dir / "walks_xgb.json"))

    if lgb_hits is not None:
        lgb_hits.booster_.save_model(str(model_dir / "lgb_hits.txt"))
    if lgb_tb is not None:
        lgb_tb.booster_.save_model(str(model_dir / "lgb_total_bases.txt"))
    if lgb_hr is not None:
        lgb_hr.booster_.save_model(str(model_dir / "lgb_home_runs.txt"))
    if lgb_walks is not None:
        lgb_walks.booster_.save_model(str(model_dir / "lgb_walks.txt"))

    (model_dir / "feature_columns_batters.json").write_text(
        json.dumps(feature_cols), encoding="utf-8"
    )
    (model_dir / "feature_medians_batters.json").write_text(
        json.dumps(medians), encoding="utf-8"
    )

    return {
        "mae_hits": hits_mae,   "ci_hits": hits_p68,
        "mae_total_bases": tb_mae, "ci_total_bases": tb_p68,
        "mae_home_runs": hr_mae,   "ci_home_runs": hr_p68,
        "mae_walks": walks_mae,    "ci_walks": walks_p68,
        "n_rows": len(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = TrainConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    log.info("=== Training pitcher prop model (strikeouts) ===")
    try:
        r = train_pitcher_models(cfg)
        results.update(r)
        log.info("Pitcher: MAE=%.3f, CI(p68)=%.3f, n=%d",
                 r["mae_strikeouts"], r["ci_strikeouts"], r["n_rows"])
    except Exception:
        log.exception("Pitcher model training failed")
        results["mae_strikeouts"] = None
        results["ci_strikeouts"] = None

    log.info("=== Training batter prop models (hits + total_bases + home_runs + walks) ===")
    try:
        r = train_batter_models(cfg)
        results.update(r)
        log.info(
            "Hits: MAE=%.3f | TB: MAE=%.3f | HR: MAE=%.3f | Walks: MAE=%.3f | n=%d",
            r["mae_hits"], r["mae_total_bases"], r["mae_home_runs"], r["mae_walks"],
            r["n_rows"],
        )
    except Exception:
        log.exception("Batter model training failed")
        for k in ("mae_hits", "ci_hits", "mae_total_bases", "ci_total_bases",
                  "mae_home_runs", "ci_home_runs", "mae_walks", "ci_walks"):
            results[k] = None

    # Save backtest summary
    (cfg.model_dir / "backtest_mae.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    log.info("Saved backtest_mae.json: %s", results)
    log.info("MLB player prop training complete.")


if __name__ == "__main__":
    main()
