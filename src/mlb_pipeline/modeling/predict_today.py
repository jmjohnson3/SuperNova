import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBClassifier, XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from sqlalchemy import create_engine, text

from .bankroll_layers import BankrollAssessment, assess_bankroll_layer, bankroll_tag
from .bankroll_ledger import insert_game_bankroll_ledger
from .model_pick_ledger import insert_game_model_pick_ledger
from .features import add_game_derived_features, build_fd_parlay_url
from .side_recalibration import (
    apply_side_calibrator,
    game_total_line_bucket as _cal_game_total_line_bucket,
    price_bucket as _cal_price_bucket,
)

# Ensure stdout is UTF-8 on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger("mlb_pipeline.modeling.predict_today")

_ET = ZoneInfo("America/New_York")
_BREAKEVEN_PROB = 0.524


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = Path(__file__).resolve().parent / "models"
    season: str | None = None
    et_date: date | None = None
    # Minimum |edge| in runs to flag a run-line bet
    min_edge_run_line: float = 1.5
    # Away run lines are not blanket-bad, but price and role matter. Recent
    # graded history is close to breakeven overall, with the weakness
    # concentrated in away-favorite covers and heavy juice. Allow away sides
    # only after those guards pass.
    allow_away_run_line_bets: bool = True
    allow_away_favorite_run_line_bets: bool = False
    max_away_dog_run_line_lay_price: int = -130
    max_run_line_lay_price: int = -180
    # Minimum |edge| in runs to flag a total bet (over OR under)
    # 2026-05-25: raised 1.5→2.0. Analysis showed 90d WR of only 38-47% for edges
    # 1.0–2.0 (overs systematically losing; pred_vs_mkt=+1.07 structural over-bias).
    # Edges 2.5+ show 58-75% WR. 2.0 is the conservative cut until the over-bias
    # is diagnosed and fixed in the model.
    min_edge_total: float = 2.0
    # Unders are not blanket-bad, but the weak bucket is high totals (10.5+).
    # Keep the side off by default until the non-high-total sample is larger;
    # if enabled, the high-total guard below still applies.
    allow_total_under_bets: bool = True
    max_total_under_market_line: float = 99.0
    max_total_lay_price: int = -180
    total_side_recalibrators_file: str = "game_total_side_recalibrators.json"
    # Optional direct market classifiers must clear this probability edge when
    # artifacts are present. If no classifier artifacts exist, run-edge logic is
    # used by itself.
    min_prob_edge_game: float = 0.03
    min_game_clf_auc: float = 0.55
    max_game_clf_brier: float = 0.26
    # Shadow-bankroll layer: never suppresses output; labels real-money readiness.
    bankroll_shadow_mode: bool = True
    bankroll_max_stake_pct: float = 0.005
    bankroll_max_daily_exposure_pct: float = 0.02
    min_game_ev: float = 0.02
    top_n_game_bets: int = 10


SQL_GAMES_FOR_DATE = """
SELECT gpf.*,
       elo.home_elo, elo.away_elo, elo.elo_diff, elo.elo_win_prob_home
FROM features.mlb_game_prediction_features gpf
LEFT JOIN features.mlb_game_elo_features elo
  ON elo.season = gpf.season AND elo.game_slug = gpf.game_slug
WHERE gpf.game_date_et = :game_date
ORDER BY gpf.start_ts_utc, gpf.game_slug
"""

SQL_STARTING_PITCHERS = """
SELECT sp.game_slug,
       CASE WHEN sp.team_abbr = g.home_team_abbr THEN 'home' ELSE 'away' END AS side,
       sp.player_name AS pitcher_name
FROM raw.mlb_starting_pitchers sp
JOIN raw.mlb_games g ON g.game_slug = sp.game_slug
WHERE g.game_date_et = :game_date
  AND sp.player_name IS NOT NULL
"""


SQL_FANDUEL_LINKS = """
SELECT home_team  AS home_abbr,
       away_team  AS away_abbr,
       spread_home_link,
       spread_away_link,
       total_over_link,
       total_under_link,
       spread_home_price,
       spread_away_price,
       total_over_price,
       total_under_price
FROM odds.mlb_game_lines
WHERE as_of_date = :d
  AND bookmaker_key = 'fanduel'
  AND (spread_home_link IS NOT NULL OR total_over_link IS NOT NULL)
ORDER BY fetched_at_utc DESC
"""


def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _prep_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_medians: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (id_df, X_aligned).

    Mirrors make_xy_raw() from train_game_models.py:
      - Drop postgame / id columns
      - Compute season_days_elapsed
      - Derive start_hour_utc / start_dow_utc
      - One-hot season
      - Add derived interaction features
      - Align to feature_cols schema
      - Fill NaNs with training medians
    """
    id_cols = ["season", "game_slug", "game_date_et", "start_ts_utc",
               "home_team_abbr", "away_team_abbr"]
    id_df = df[[c for c in id_cols if c in df.columns]].copy()

    X = df.drop(columns=[c for c in id_cols if c in df.columns]).copy()

    # Season position: days elapsed since April 1
    if "game_date_et" in df.columns:
        gdt = pd.to_datetime(df["game_date_et"])
        season_start = pd.to_datetime(gdt.dt.year.astype(str) + "-04-01")
        X["season_days_elapsed"] = (gdt - season_start).dt.days.values

    # Timing
    if "start_ts_utc" in df.columns:
        ts = pd.to_datetime(df["start_ts_utc"], errors="coerce", utc=True)
        X["start_hour_utc"] = ts.dt.hour
        X["start_dow_utc"]  = ts.dt.dayofweek
        ts_et = ts.dt.tz_convert("America/New_York")
        X["is_day_game"] = (ts_et.dt.hour < 17).astype(int)

    # b2b flags → 0/1
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # One-hot season (must match training schema)
    cat_cols = []
    if "season" in df.columns:
        X["season"] = df["season"]
        cat_cols.append("season")
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # Coerce numeric-ish strings
    X = _coerce_numeric_cols(X)

    # Derived interaction features — single source of truth in features.py
    X = add_game_derived_features(X)

    # Align to training schema
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        X = X.drop(columns=extra)

    X = X[feature_cols]

    # Fill with training medians
    for c, med in feature_medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(med)

    # Final fill (one-hot dummies, etc.)
    X = X.fillna(0.0)

    return id_df, X


def _load_models(
    cfg: PredictConfig,
) -> tuple[dict[str, object], list[str], dict[str, float]]:
    model_dir = cfg.model_dir
    feature_cols_path = model_dir / "feature_columns.json"
    medians_path      = model_dir / "feature_medians.json"

    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing {feature_cols_path}. Run train_game_models.py first.")
    if not medians_path.exists():
        raise FileNotFoundError(f"Missing {medians_path}. Run train_game_models.py first.")

    feature_cols    = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    feature_medians = json.loads(medians_path.read_text(encoding="utf-8"))

    paths = {
        "rl_direct":    model_dir / "run_line_direct_xgb.json",
        "total_direct": model_dir / "total_direct_xgb.json",
        "rl_resid":     model_dir / "run_line_resid_xgb.json",
        "total_resid":  model_dir / "total_resid_xgb.json",
        "f5_direct":    model_dir / "total_f5_direct_xgb.json",
    }

    models: dict = {}
    for k, p in paths.items():
        if p.exists():
            m = XGBRegressor()
            m.load_model(str(p))
            models[k] = m

    clf_cols_path = model_dir / "game_market_clf_feature_columns.json"
    clf_meds_path = model_dir / "game_market_clf_feature_medians.json"
    if clf_cols_path.exists() and clf_meds_path.exists():
        try:
            backtest_path = model_dir / "game_market_clf_backtest.json"
            backtest = json.loads(backtest_path.read_text(encoding="utf-8")) if backtest_path.exists() else {}

            def _clf_quality_ok(target_name: str) -> bool:
                rec = ((backtest.get("targets") or {}).get(target_name) or {})
                if rec.get("status") != "trained":
                    return False
                holdout = rec.get("holdout") or {}
                auc = holdout.get("auc")
                brier = holdout.get("brier")
                return (
                    isinstance(auc, (int, float))
                    and isinstance(brier, (int, float))
                    and float(auc) >= cfg.min_game_clf_auc
                    and float(brier) <= cfg.max_game_clf_brier
                )

            models["_game_clf_feature_cols"] = json.loads(clf_cols_path.read_text(encoding="utf-8"))
            models["_game_clf_feature_medians"] = json.loads(clf_meds_path.read_text(encoding="utf-8"))
            for key, filename in [
                ("run_line_cover_clf", "run_line_cover_clf_xgb.json"),
                ("total_over_clf", "total_over_clf_xgb.json"),
            ]:
                target_name = "run_line_home_cover" if key == "run_line_cover_clf" else "total_over"
                if not _clf_quality_ok(target_name):
                    log.info("Skipping %s: classifier backtest did not pass quality gate", key)
                    continue
                path = model_dir / filename
                if path.exists():
                    clf = XGBClassifier()
                    clf.load_model(str(path))
                    models[key] = clf
            loaded_clf = [k for k in ("run_line_cover_clf", "total_over_clf") if k in models]
            if loaded_clf:
                log.info("Loaded game market classifiers: %s", loaded_clf)
        except Exception as exc:
            log.warning("Could not load game market classifier artifacts: %s", exc)

    if _HAS_LGB:
        for lgb_key, lgb_name in [("rl_direct_lgb",    "run_line_direct_lgb.txt"),
                                   ("total_direct_lgb", "total_direct_lgb.txt"),
                                   ("f5_direct_lgb",    "total_f5_direct_lgb.txt")]:
            lgb_path = model_dir / lgb_name
            if lgb_path.exists():
                try:
                    models[lgb_key] = lgb.Booster(model_file=str(lgb_path))
                except Exception as _e:
                    log.debug("Could not load %s: %s", lgb_name, _e)
        # Quantile models for game-specific sigma
        _QUANTILE_INTS = [10, 25, 50, 75, 90]
        for _qi in _QUANTILE_INTS:
            for _qstem, _qkey_pfx in [("run_line", "rl"), ("total", "tot")]:
                _qpath = model_dir / f"{_qstem}_q{_qi:02d}_lgb.txt"
                if _qpath.exists():
                    try:
                        models[f"{_qkey_pfx}_q{_qi:02d}"] = lgb.Booster(model_file=str(_qpath))
                    except Exception as _e:
                        log.debug("Could not load %s: %s", _qpath.name, _e)
        _q_loaded = [k for k in models if k.startswith(("rl_q", "tot_q"))]
        if _q_loaded:
            log.info("Loaded quantile LGB models: %d total", len(_q_loaded))

    if "rl_direct" not in models or "total_direct" not in models:
        raise FileNotFoundError(f"Missing direct models in {model_dir}. Run training first.")

    lgb_loaded = [k for k in models if k.endswith("_lgb")]
    if lgb_loaded:
        log.info("Loaded LGB game models: %s", lgb_loaded)

    return models, feature_cols, feature_medians


def _load_calibration(model_dir: Path) -> dict:
    p = model_dir / "calibration.json"
    defaults = {
        "direct_spread_rmse": 3.5,   # typical MLB run-diff RMSE ~3-4 runs
        "direct_total_rmse":  3.0,
        "resid_spread_rmse":  2.5,
        "resid_total_rmse":   2.5,
    }
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {**defaults, **d}
    return defaults


def _compute_blend_weight_run_line(calib: dict) -> float:
    """Inverse-MAE blend weight for the run-line residual model.

    Quality gate: residual model must be within 2% of the best baseline MAE.
    Falls back to 0.0 (direct-only) when gate fails or values are missing.
    """
    saved_alpha = calib.get("blend_alpha_spread")
    if isinstance(saved_alpha, (int, float)) and 0.0 <= float(saved_alpha) <= 1.0:
        return float(saved_alpha)

    direct_mae_market = calib.get("direct_spread_mae_market",
                                   calib.get("direct_rl_mae", calib.get("direct_spread_mae", 4.0)))
    market_mae = calib.get("market_spread_mae", direct_mae_market)
    resid_mae  = calib.get("resid_spread_mae",  calib.get("resid_rl_mae", 99.0))
    if direct_mae_market <= 0 or resid_mae <= 0:
        return 0.0
    best_baseline = min(direct_mae_market, market_mae)
    if resid_mae >= best_baseline * 1.02:
        log.info(
            "Run-line residual quality gate failed "
            "(resid MAE %.3f >= 102%% of best baseline %.3f). Direct only.",
            resid_mae, best_baseline * 1.02,
        )
        return 0.0
    w_d = 1.0 / direct_mae_market
    w_r = 1.0 / resid_mae
    return round(w_r / (w_d + w_r), 4)


def _compute_blend_weight_total(calib: dict) -> float:
    """Inverse-MAE blend weight for the total residual model."""
    saved_alpha = calib.get("blend_alpha_total")
    if isinstance(saved_alpha, (int, float)) and 0.0 <= float(saved_alpha) <= 1.0:
        return float(saved_alpha)

    direct_mae_market = calib.get("direct_total_mae_market",
                                   calib.get("direct_total_mae", 4.0))
    market_mae = calib.get("market_total_mae", direct_mae_market)
    resid_mae  = calib.get("resid_total_mae", 99.0)
    if direct_mae_market <= 0 or resid_mae <= 0:
        return 0.0
    best_baseline = min(direct_mae_market, market_mae)
    if resid_mae >= best_baseline * 1.02:
        log.info(
            "Total residual quality gate failed "
            "(resid MAE %.3f >= 102%% of best baseline %.3f). Direct only.",
            resid_mae, best_baseline * 1.02,
        )
        return 0.0
    w_d = 1.0 / direct_mae_market
    w_r = 1.0 / resid_mae
    return round(w_r / (w_d + w_r), 4)


def _compute_live_bias(engine) -> tuple[float, float]:
    """Rolling mean prediction error from recent graded MLB games.

    Returns (rl_bias, total_bias). Requires >= 10 graded games in last 30 days;
    otherwise returns (0.0, 0.0).
    """
    sql = text("""
        SELECT
            COUNT(*)                                   AS n,
            AVG(pred_run_diff - actual_run_diff)       AS rl_bias,
            AVG(pred_total    - actual_total)          AS total_bias
        FROM bets.mlb_game_predictions
        WHERE actual_run_diff IS NOT NULL
          AND game_date_et >= CURRENT_DATE - INTERVAL '30 days'
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if row is None or row[0] is None or int(row[0]) < 10:
            log.info("Auto bias correction: not enough data (need 10+ graded games in last 30 days).")
            return 0.0, 0.0
        rb = float(row[1] or 0.0)
        tb = float(row[2] or 0.0)
        log.info("Auto bias correction: n=%d  rl_bias=%.2f  total_bias=%.2f", int(row[0]), rb, tb)
        return rb, tb
    except Exception as exc:
        log.warning("Could not compute live bias: %s", exc)
        return 0.0, 0.0


def _compute_team_bias(engine, min_games: int = 4, shrinkage: float = 0.4) -> dict[str, float]:
    """Rolling per-team mean prediction error (last 30 days).

    Positive value = team has been over-rated (model predicted better than reality).
    Requires min_games per team to apply correction; otherwise omits that team.

    Attribution:
      home team gets +error  (pred_run_diff - actual_run_diff)
      away team gets -error  (they look better when home team is over-predicted)
    Both perspectives pooled per team, then shrunk toward 0.
    """
    sql = text("""
        SELECT home_team_abbr, away_team_abbr,
               (pred_run_diff - actual_run_diff) AS error
        FROM bets.mlb_game_predictions
        WHERE actual_run_diff IS NOT NULL
          AND pred_run_diff   IS NOT NULL
          AND game_date_et >= CURRENT_DATE - INTERVAL '30 days'
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        if not rows:
            return {}
        from collections import defaultdict
        team_errors: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            err = float(r[2] or 0.0)
            team_errors[r[0]].append(err)    # home: +error
            team_errors[r[1]].append(-err)   # away: -error
        result: dict[str, float] = {}
        for team, errs in team_errors.items():
            if len(errs) >= min_games:
                result[team] = shrinkage * (sum(errs) / len(errs))
        if result:
            log.info("Team bias correction: %d teams  top: %s",
                     len(result),
                     sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        return result
    except Exception as exc:
        log.warning("Could not compute team bias: %s", exc)
        return {}


def _kelly(
    edge_runs: float,
    juice: int = -110,
    shrink: float = 0.60,
    sigma: float = 3.5,
) -> tuple[float, float]:
    """
    Estimate full Kelly fraction and win probability for a run-line / total bet.

    edge_runs : |pred - market_line| (positive, regardless of direction)
    sigma     : calibrated RMSE from walk-forward CV (saved in models/calibration.json).
                Default 3.5 ≈ typical MLB run-diff RMSE.
    """
    b = 100 / abs(juice)
    p_raw = 1 / (1 + math.exp(-edge_runs / sigma))
    p = 0.5 + (p_raw - 0.5) * shrink
    kelly = max(0.0, (b * p - (1 - p)) / b)
    return kelly, p


def _american_profit_mult(price: int | float | None) -> float | None:
    if price is None:
        return None
    try:
        p = float(price)
        if pd.isna(p) or p == 0:
            return None
        return p / 100.0 if p > 0 else 100.0 / abs(p)
    except Exception:
        return None


def _price_adjusted_ev(p_win: float | None, price: int | float | None) -> float | None:
    p = _clean_prob(p_win)
    mult = _american_profit_mult(price)
    if p is None or mult is None:
        return None
    return p * mult - (1.0 - p)


def _kelly_from_prob(
    p_win: float | None,
    market_price: int | float | None = None,
    max_kelly: float = 0.05,
) -> tuple[float | None, float | None]:
    p = _clean_prob(p_win)
    if p is None:
        return None, None
    mult = _american_profit_mult(market_price)
    if mult is not None:
        k = max(0.0, (mult * p - (1.0 - p)) / mult)
    else:
        k = max(0.0, (p - _BREAKEVEN_PROB) / (1.0 - _BREAKEVEN_PROB))
    return min(k, max_kelly), p


def _clean_prob(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return None


def _load_game_total_side_recalibrators(model_dir: Path, file_name: str) -> dict:
    path = model_dir / file_name
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse game total side recalibrators at %s: %s", path, exc)
        return {}
    n = len(raw.get("calibrators") or {})
    if n:
        log.info("Loaded game total side recalibrators from %s (%d buckets)", path, n)
    return raw


def _apply_total_side_recalibration(
    *,
    side: str,
    raw_p_side: float | None,
    market_total,
    market_price,
    model_family: str,
    recalibrators: dict,
) -> tuple[float | None, str | None]:
    if raw_p_side is None:
        return None, None
    return apply_side_calibrator(
        raw_p_side,
        recalibrators,
        market="game_total",
        side=side,
        line_bucket=_cal_game_total_line_bucket(market_total),
        price_bucket_value=_cal_price_bucket(market_price),
        model_family=model_family or "unknown",
    )


def _calibrated_total_prob_kelly(
    *,
    side: str,
    raw_p_side: float | None,
    edge_abs: float,
    sigma: float,
    market_total,
    market_price,
    recalibrators: dict,
) -> tuple[float | None, float | None, str | None]:
    family = "total_over_clf" if _clean_prob(raw_p_side) is not None else "edge_sigma"
    kelly, p_win = _kelly_from_prob(raw_p_side, market_price)
    if kelly is None or p_win is None:
        _, p_win = _kelly(edge_abs, sigma=sigma)
        family = "edge_sigma"
    p_win, cal_key = _apply_total_side_recalibration(
        side=side,
        raw_p_side=p_win,
        market_total=market_total,
        market_price=market_price,
        model_family=family,
        recalibrators=recalibrators,
    )
    if p_win is not None:
        kelly, p_win = _kelly_from_prob(p_win, market_price)
    return kelly, p_win, cal_key


def _prob_edge_for_side(p_over: float | None, side: str) -> float | None:
    p = _clean_prob(p_over)
    if p is None:
        return None
    if side in {"home", "over"}:
        return p - _BREAKEVEN_PROB
    if side in {"away", "under"}:
        return (1.0 - p) - _BREAKEVEN_PROB
    return None


def _prob_gate_allowed(p_over: float | None, side: str, cfg: PredictConfig) -> bool:
    edge = _prob_edge_for_side(p_over, side)
    return True if edge is None else edge >= cfg.min_prob_edge_game


def _price_lay_allowed(price: int | float | None, max_lay_price: int | float) -> bool:
    if price is None:
        return True
    try:
        return float(price) >= float(max_lay_price)
    except Exception:
        return True


def _signal_run_line_price(fd_row, edge_run_line: float | None) -> int | None:
    if fd_row is None or edge_run_line is None or pd.isna(edge_run_line):
        return None
    return (
        getattr(fd_row, "spread_home_price", None)
        if float(edge_run_line) > 0
        else getattr(fd_row, "spread_away_price", None)
    )


def _signal_total_price(fd_row, edge_total: float | None) -> int | None:
    if fd_row is None or edge_total is None or pd.isna(edge_total):
        return None
    return (
        getattr(fd_row, "total_over_price", None)
        if float(edge_total) > 0
        else getattr(fd_row, "total_under_price", None)
    )


def _run_line_label_for_side(market_run_line: float | None, side: str) -> str:
    """Format the displayed spread for the selected side.

    market_run_line is stored as the home team's spread. Away-side labels must
    use the inverse line.
    """
    if market_run_line is None or pd.isna(market_run_line):
        return "n/a"
    line = float(market_run_line)
    if side == "away":
        line = -line
    return f"{line:+.1f}"


def _run_line_signal_allowed(
    edge_run_line: float | None,
    cfg: PredictConfig,
    p_home_cover: float | None = None,
    market_price: int | float | None = None,
    market_run_line: float | None = None,
) -> bool:
    if edge_run_line is None or pd.isna(edge_run_line):
        return False
    edge = float(edge_run_line)
    if abs(edge) < cfg.min_edge_run_line:
        return False
    if not _price_lay_allowed(market_price, cfg.max_run_line_lay_price):
        return False
    side = "home" if edge > 0 else "away"
    if side == "away" and not cfg.allow_away_run_line_bets:
        return False
    if (
        side == "away"
        and not cfg.allow_away_favorite_run_line_bets
        and market_run_line is not None
        and pd.notna(market_run_line)
        and float(market_run_line) > 0
    ):
        return False
    return _prob_gate_allowed(p_home_cover, side, cfg)


def _total_signal_allowed(
    edge_total: float | None,
    cfg: PredictConfig,
    p_total_over: float | None = None,
    market_total: float | None = None,
    market_price: int | float | None = None,
) -> bool:
    if edge_total is None or pd.isna(edge_total):
        return False
    edge = float(edge_total)
    if edge >= cfg.min_edge_total:
        return _prob_gate_allowed(p_total_over, "over", cfg)
    if not cfg.allow_total_under_bets:
        return False
    if market_total is not None and pd.notna(market_total) and float(market_total) > cfg.max_total_under_market_line:
        return False
    if not _price_lay_allowed(market_price, cfg.max_total_lay_price):
        return False
    return (
        -edge >= cfg.min_edge_total
        and _prob_gate_allowed(p_total_over, "under", cfg)
    )


def _game_bankroll_assessment(
    *,
    market: str,
    side: str | None,
    cfg: PredictConfig,
    kelly_fraction: float | None,
    win_prob: float | None,
    market_price: int | float | None,
    both_sp_known: bool,
    market_line: float | None = None,
) -> BankrollAssessment:
    """Label a game signal for shadow-bankroll readiness."""
    hard: list[str] = []
    soft: list[str] = []

    if not both_sp_known:
        hard.append("sp_tbd")

    clean_price = None
    if market_price is not None:
        try:
            clean_price = None if pd.isna(market_price) else float(market_price)
        except Exception:
            clean_price = None

    if clean_price is None:
        hard.append("missing_price")
    elif market == "run_line" and not _price_lay_allowed(clean_price, cfg.max_run_line_lay_price):
        hard.append("heavy_juice")
    elif market == "total" and not _price_lay_allowed(clean_price, cfg.max_total_lay_price):
        hard.append("heavy_juice")

    if market == "run_line":
        try:
            if (
                market_line is not None
                and pd.notna(market_line)
                and abs(abs(float(market_line)) - 1.5) > 1e-9
            ):
                hard.append("non_standard_run_line")
        except Exception:
            pass

    if market == "run_line" and side == "away":
        try:
            if market_line is not None and pd.notna(market_line) and float(market_line) > 0:
                hard.append("away_favorite_run_line")
            if (
                market_line is not None
                and pd.notna(market_line)
                and float(market_line) < 0
                and clean_price is not None
                and clean_price < cfg.max_away_dog_run_line_lay_price
            ):
                hard.append("away_dog_lay_price")
        except Exception:
            pass

    if market == "total" and side == "under":
        if not cfg.allow_total_under_bets:
            hard.append("total_under_disabled")
        try:
            if (
                market_line is not None
                and pd.notna(market_line)
                and float(market_line) > cfg.max_total_under_market_line
            ):
                hard.append("high_total_under")
        except Exception:
            pass

    if win_prob is None:
        soft.append("missing_win_probability")
    elif clean_price is not None:
        price_ev = _price_adjusted_ev(win_prob, clean_price)
        if price_ev is None:
            soft.append("missing_price_ev")
        elif price_ev < 0:
            soft.append("negative_price_ev")
        elif price_ev < cfg.min_game_ev:
            soft.append("below_min_price_ev")
    if kelly_fraction is None or kelly_fraction <= 0:
        soft.append("zero_kelly")

    return assess_bankroll_layer(
        has_signal=side is not None,
        hard_blocks=hard,
        soft_warnings=soft,
        kelly_fraction=kelly_fraction,
        max_stake_pct=cfg.bankroll_max_stake_pct,
    )


def _append_bankroll_reason(existing: str, reason: str) -> str:
    parts = [p.strip() for p in (existing or "").split(";") if p.strip()]
    if reason not in parts:
        parts.append(reason)
    return "; ".join(parts)


def _cap_game_bankroll_rows(rows: list[dict], cfg: PredictConfig) -> list[dict]:
    candidates: list[tuple[float, int, str]] = []
    for idx, row in enumerate(rows):
        if row.get("bankroll_candidate_rl"):
            candidates.append((abs(float(row.get("edge_run_line") or 0.0)), idx, "rl"))
        if row.get("bankroll_candidate_total"):
            candidates.append((abs(float(row.get("edge_total") or 0.0)), idx, "total"))

    used = 0.0
    for _score, idx, market in sorted(candidates, reverse=True):
        row = rows[idx]
        if market == "rl":
            stake_key = "stake_pct_rl"
            candidate_key = "bankroll_candidate_rl"
            tier_key = "bankroll_tier_rl"
            reasons_key = "bankroll_reasons_rl"
        else:
            stake_key = "stake_pct_total"
            candidate_key = "bankroll_candidate_total"
            tier_key = "bankroll_tier_total"
            reasons_key = "bankroll_reasons_total"

        stake = float(row.get(stake_key) or 0.0)
        if used + stake <= cfg.bankroll_max_daily_exposure_pct + 1e-12:
            used += stake
            continue
        row[candidate_key] = False
        row[tier_key] = "paper"
        row[reasons_key] = _append_bankroll_reason(
            row.get(reasons_key) or "",
            "daily_exposure_cap",
        )
        row[stake_key] = 0.0
    return rows


def _existing_prop_bankroll_exposure(engine, et_day: date) -> float:
    try:
        with engine.begin() as conn:
            val = conn.execute(
                text(
                    """
                    SELECT COALESCE(SUM(
                        CASE WHEN bankroll_candidate
                             THEN COALESCE(stake_pct, 0)
                             ELSE 0 END
                    ), 0)
                    FROM bets.mlb_prop_predictions
                    WHERE game_date_et = :game_date
                    """
                ),
                {"game_date": et_day},
            ).scalar()
            return float(val or 0.0)
    except Exception:
        log.debug("Could not load existing prop bankroll exposure", exc_info=True)
        return 0.0


def _p_over_from_quantiles(q_preds: dict, market_line: float) -> float:
    """Interpolate P(value > market_line) from a 5-point quantile CDF."""
    qs = sorted(q_preds)
    vals = [q_preds[q] for q in qs]
    probs = [q / 100.0 for q in qs]
    cdf = float(np.interp(market_line, vals, probs, left=0.0, right=1.0))
    return 1.0 - cdf


def _sigma_from_quantiles(q_preds: dict, fallback: float = 3.5) -> float:
    """Game-specific sigma from IQR (q75 - q25) / 1.349 (normal approximation)."""
    if 25 in q_preds and 75 in q_preds:
        iqr = q_preds[75] - q_preds[25]
        return max(float(iqr) / 1.349, 1.0)
    return fallback


def _ensure_bets_schema(engine) -> None:
    """Create bets.mlb_game_predictions table if it doesn't exist."""
    ddl = """
    CREATE SCHEMA IF NOT EXISTS bets;
    CREATE TABLE IF NOT EXISTS bets.mlb_game_predictions (
        id                    SERIAL PRIMARY KEY,
        predicted_at_utc      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        game_date_et          DATE        NOT NULL,
        game_slug             TEXT        NOT NULL,
        season                TEXT        NOT NULL,
        home_team_abbr        TEXT        NOT NULL,
        away_team_abbr        TEXT        NOT NULL,
        home_sp_name          TEXT,
        away_sp_name          TEXT,
        pred_run_diff         NUMERIC,
        pred_total            NUMERIC,
        used_residual_model   BOOLEAN     DEFAULT FALSE,
        market_run_line       NUMERIC,
        market_total          NUMERIC,
        edge_run_line         NUMERIC,
        edge_total            NUMERIC,
        actual_run_diff       NUMERIC,
        actual_total          NUMERIC,
        run_line_bet_side     TEXT,
        total_bet_side        TEXT,
        run_line_covered      BOOLEAN,
        total_correct         BOOLEAN,
        direction_correct     BOOLEAN,
        kelly_fraction_rl     NUMERIC,
        kelly_fraction_total  NUMERIC,
        win_prob_rl           NUMERIC,
        win_prob_total        NUMERIC,
        UNIQUE (game_date_et, game_slug)
    );
    ALTER TABLE bets.mlb_game_predictions
        ADD COLUMN IF NOT EXISTS sigma_q_rl      FLOAT,
        ADD COLUMN IF NOT EXISTS sigma_q_total   FLOAT,
        ADD COLUMN IF NOT EXISTS p_over_rl_q     FLOAT,
        ADD COLUMN IF NOT EXISTS p_over_total_q  FLOAT,
        ADD COLUMN IF NOT EXISTS market_rl_price INTEGER,
        ADD COLUMN IF NOT EXISTS market_total_price INTEGER,
        ADD COLUMN IF NOT EXISTS p_home_cover_clf FLOAT,
        ADD COLUMN IF NOT EXISTS p_total_over_clf FLOAT,
        ADD COLUMN IF NOT EXISTS edge_run_line_prob FLOAT,
        ADD COLUMN IF NOT EXISTS edge_total_prob FLOAT,
        ADD COLUMN IF NOT EXISTS bankroll_tier_rl TEXT,
        ADD COLUMN IF NOT EXISTS bankroll_tier_total TEXT,
        ADD COLUMN IF NOT EXISTS bankroll_candidate_rl BOOLEAN,
        ADD COLUMN IF NOT EXISTS bankroll_candidate_total BOOLEAN,
        ADD COLUMN IF NOT EXISTS bankroll_reasons_rl TEXT,
        ADD COLUMN IF NOT EXISTS bankroll_reasons_total TEXT,
        ADD COLUMN IF NOT EXISTS stake_pct_rl NUMERIC,
        ADD COLUMN IF NOT EXISTS stake_pct_total NUMERIC;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _save_predictions(
    out: pd.DataFrame,
    engine,
    et_day,
    cfg: PredictConfig,
    calib: dict | None = None,
    w_rl: float = 0.0,
    w_total: float = 0.0,
    fd_links: dict | None = None,
    total_side_recalibrators: dict | None = None,
) -> None:
    """Upsert game predictions into bets.mlb_game_predictions."""
    _ensure_bets_schema(engine)
    upsert_sql = text("""
        INSERT INTO bets.mlb_game_predictions
            (game_date_et, game_slug, season, home_team_abbr, away_team_abbr,
             home_sp_name, away_sp_name,
             pred_run_diff, pred_total, used_residual_model,
             market_run_line, market_total, edge_run_line, edge_total,
             run_line_bet_side, total_bet_side,
             kelly_fraction_rl, kelly_fraction_total,
             win_prob_rl, win_prob_total,
             market_rl_price, market_total_price,
             sigma_q_rl, sigma_q_total, p_over_rl_q, p_over_total_q,
             p_home_cover_clf, p_total_over_clf, edge_run_line_prob, edge_total_prob,
             bankroll_tier_rl, bankroll_tier_total,
             bankroll_candidate_rl, bankroll_candidate_total,
             bankroll_reasons_rl, bankroll_reasons_total,
             stake_pct_rl, stake_pct_total)
        VALUES
            (:game_date_et, :game_slug, :season, :home_team_abbr, :away_team_abbr,
             :home_sp_name, :away_sp_name,
             :pred_run_diff, :pred_total, :used_residual_model,
             :market_run_line, :market_total, :edge_run_line, :edge_total,
             :run_line_bet_side, :total_bet_side,
             :kelly_fraction_rl, :kelly_fraction_total,
             :win_prob_rl, :win_prob_total,
             :market_rl_price, :market_total_price,
             :sigma_q_rl, :sigma_q_total, :p_over_rl_q, :p_over_total_q,
             :p_home_cover_clf, :p_total_over_clf, :edge_run_line_prob, :edge_total_prob,
             :bankroll_tier_rl, :bankroll_tier_total,
             :bankroll_candidate_rl, :bankroll_candidate_total,
             :bankroll_reasons_rl, :bankroll_reasons_total,
             :stake_pct_rl, :stake_pct_total)
        ON CONFLICT (game_date_et, game_slug) DO UPDATE SET
            predicted_at_utc    = NOW(),
            home_sp_name        = EXCLUDED.home_sp_name,
            away_sp_name        = EXCLUDED.away_sp_name,
            pred_run_diff       = EXCLUDED.pred_run_diff,
            pred_total          = EXCLUDED.pred_total,
            used_residual_model = EXCLUDED.used_residual_model,
            market_run_line     = EXCLUDED.market_run_line,
            market_total        = EXCLUDED.market_total,
            edge_run_line       = EXCLUDED.edge_run_line,
            edge_total          = EXCLUDED.edge_total,
            run_line_bet_side   = EXCLUDED.run_line_bet_side,
            total_bet_side      = EXCLUDED.total_bet_side,
            kelly_fraction_rl   = EXCLUDED.kelly_fraction_rl,
            kelly_fraction_total= EXCLUDED.kelly_fraction_total,
            win_prob_rl         = EXCLUDED.win_prob_rl,
            win_prob_total      = EXCLUDED.win_prob_total,
            market_rl_price     = EXCLUDED.market_rl_price,
            market_total_price  = EXCLUDED.market_total_price,
            sigma_q_rl          = EXCLUDED.sigma_q_rl,
            sigma_q_total       = EXCLUDED.sigma_q_total,
            p_over_rl_q         = EXCLUDED.p_over_rl_q,
            p_over_total_q      = EXCLUDED.p_over_total_q,
            p_home_cover_clf    = EXCLUDED.p_home_cover_clf,
            p_total_over_clf    = EXCLUDED.p_total_over_clf,
            edge_run_line_prob  = EXCLUDED.edge_run_line_prob,
            edge_total_prob     = EXCLUDED.edge_total_prob,
            bankroll_tier_rl    = EXCLUDED.bankroll_tier_rl,
            bankroll_tier_total = EXCLUDED.bankroll_tier_total,
            bankroll_candidate_rl = EXCLUDED.bankroll_candidate_rl,
            bankroll_candidate_total = EXCLUDED.bankroll_candidate_total,
            bankroll_reasons_rl = EXCLUDED.bankroll_reasons_rl,
            bankroll_reasons_total = EXCLUDED.bankroll_reasons_total,
            stake_pct_rl        = EXCLUDED.stake_pct_rl,
            stake_pct_total     = EXCLUDED.stake_pct_total
    """)

    _calib = calib or {}
    rows = []
    for _, r in out.iterrows():
        edge_rl = float(r["edge_run_line"]) if pd.notna(r.get("edge_run_line")) else None
        edge_t  = float(r["edge_total"])    if pd.notna(r.get("edge_total"))    else None
        p_home_cover = _clean_prob(r.get("p_home_cover_clf"))
        p_total_over = _clean_prob(r.get("p_total_over_clf"))
        rl_bet = None
        tot_bet = None
        kf_rl = kf_t = wp_rl = wp_t = None

        used_blend = bool(r.get("used_market_recon", False))
        _sq_rl = r.get("sigma_q_rl")
        _sq_t  = r.get("sigma_q_total")
        sigma_rl = (float(_sq_rl) if (_sq_rl is not None and pd.notna(_sq_rl)) else
                    _calib.get("resid_spread_rmse" if (used_blend and w_rl > 0) else "direct_spread_rmse", 3.5))
        sigma_t  = (float(_sq_t) if (_sq_t is not None and pd.notna(_sq_t)) else
                    _calib.get("resid_total_rmse"  if (used_blend and w_total > 0) else "direct_total_rmse",  3.0))

        both_sp = bool(r.get("both_sp_known", True))
        _fdr = (fd_links or {}).get((r["home_team_abbr"], r["away_team_abbr"]))
        rl_signal_price = _signal_run_line_price(_fdr, edge_rl)
        total_signal_price = _signal_total_price(_fdr, edge_t)

        if edge_rl is not None and both_sp and _run_line_signal_allowed(
            edge_rl,
            cfg,
            p_home_cover,
            rl_signal_price,
            r.get("market_run_line"),
        ):
            rl_bet = "home" if edge_rl > 0 else "away"
            p_side = p_home_cover if rl_bet == "home" else (1.0 - p_home_cover if p_home_cover is not None else None)
            kf_rl, wp_rl = _kelly_from_prob(p_side, rl_signal_price)
            if kf_rl is None or wp_rl is None:
                _, wp_rl = _kelly(abs(edge_rl), sigma=sigma_rl)
                kf_rl, wp_rl = _kelly_from_prob(wp_rl, rl_signal_price)

        if edge_t is not None and both_sp:
            if edge_t >= cfg.min_edge_total and _total_signal_allowed(
                edge_t,
                cfg,
                p_total_over,
                r.get("market_total"),
                total_signal_price,
            ):
                tot_bet = "over"
                kf_t, wp_t, _total_cal_key = _calibrated_total_prob_kelly(
                    side="over",
                    raw_p_side=p_total_over,
                    edge_abs=edge_t,
                    sigma=sigma_t,
                    market_total=r.get("market_total"),
                    market_price=total_signal_price,
                    recalibrators=total_side_recalibrators or {},
                )
            elif _total_signal_allowed(
                edge_t,
                cfg,
                p_total_over,
                r.get("market_total"),
                total_signal_price,
            ):
                tot_bet = "under"
                p_side = 1.0 - p_total_over if p_total_over is not None else None
                kf_t, wp_t, _total_cal_key = _calibrated_total_prob_kelly(
                    side="under",
                    raw_p_side=p_side,
                    edge_abs=-edge_t,
                    sigma=sigma_t,
                    market_total=r.get("market_total"),
                    market_price=total_signal_price,
                    recalibrators=total_side_recalibrators or {},
                )
            # Totals are side-calibrated from locked historical model picks before bankroll sizing.

        # Entry prices for price-based CLV tracking
        mkt_rl_price = None
        mkt_tot_price = None
        if _fdr is not None:
            if rl_bet == "home":
                mkt_rl_price = getattr(_fdr, "spread_home_price", None)
            elif rl_bet == "away":
                mkt_rl_price = getattr(_fdr, "spread_away_price", None)
            if tot_bet == "over":
                mkt_tot_price = getattr(_fdr, "total_over_price", None)
            elif tot_bet == "under":
                mkt_tot_price = getattr(_fdr, "total_under_price", None)

        rl_bankroll = _game_bankroll_assessment(
            market="run_line",
            side=rl_bet,
            cfg=cfg,
            kelly_fraction=kf_rl,
            win_prob=wp_rl,
            market_price=mkt_rl_price,
            both_sp_known=both_sp,
            market_line=r.get("market_run_line"),
        )
        total_bankroll = _game_bankroll_assessment(
            market="total",
            side=tot_bet,
            cfg=cfg,
            kelly_fraction=kf_t,
            win_prob=wp_t,
            market_price=mkt_tot_price,
            both_sp_known=both_sp,
            market_line=r.get("market_total"),
        )

        rows.append({
            "game_date_et":       et_day,
            "game_slug":          r["game_slug"],
            "season":             r["season"],
            "home_team_abbr":     r["home_team_abbr"],
            "away_team_abbr":     r["away_team_abbr"],
            "home_sp_name":       r.get("home_sp_name"),
            "away_sp_name":       r.get("away_sp_name"),
            "pred_run_diff":      float(r["pred_run_diff"]),
            "pred_total":         float(r["pred_total"]),
            "used_residual_model": bool(r.get("used_market_recon", False)),
            "market_run_line":    float(r["market_run_line"]) if pd.notna(r.get("market_run_line")) else None,
            "market_total":       float(r["market_total"])   if pd.notna(r.get("market_total"))    else None,
            "edge_run_line":      edge_rl,
            "edge_total":         edge_t,
            "run_line_bet_side":  rl_bet,
            "total_bet_side":     tot_bet,
            "kelly_fraction_rl":  round(kf_rl, 4) if kf_rl  is not None else None,
            "kelly_fraction_total": round(kf_t, 4) if kf_t  is not None else None,
            "win_prob_rl":        round(wp_rl, 4)  if wp_rl  is not None else None,
            "win_prob_total":     round(wp_t, 4)   if wp_t   is not None else None,
            "market_rl_price":    int(mkt_rl_price)  if mkt_rl_price  is not None else None,
            "market_total_price": int(mkt_tot_price) if mkt_tot_price is not None else None,
            "sigma_q_rl":     round(float(r["sigma_q_rl"]),    3) if pd.notna(r.get("sigma_q_rl"))    else None,
            "sigma_q_total":  round(float(r["sigma_q_total"]), 3) if pd.notna(r.get("sigma_q_total")) else None,
            "p_over_rl_q":    round(float(r["p_over_rl_q"]),   4) if pd.notna(r.get("p_over_rl_q"))   else None,
            "p_over_total_q": round(float(r["p_over_total_q"]), 4) if pd.notna(r.get("p_over_total_q")) else None,
            "p_home_cover_clf": round(float(p_home_cover), 4) if p_home_cover is not None else None,
            "p_total_over_clf": round(float(p_total_over), 4) if p_total_over is not None else None,
            "both_sp_known": both_sp,
            "edge_run_line_prob": round(float(r["edge_run_line_prob"]), 4) if pd.notna(r.get("edge_run_line_prob")) else None,
            "edge_total_prob": round(float(r["edge_total_prob"]), 4) if pd.notna(r.get("edge_total_prob")) else None,
            "bankroll_tier_rl": rl_bankroll.tier,
            "bankroll_tier_total": total_bankroll.tier,
            "bankroll_candidate_rl": rl_bankroll.candidate,
            "bankroll_candidate_total": total_bankroll.candidate,
            "bankroll_reasons_rl": rl_bankroll.reasons,
            "bankroll_reasons_total": total_bankroll.reasons,
            "stake_pct_rl": round(rl_bankroll.stake_pct, 4),
            "stake_pct_total": round(total_bankroll.stake_pct, 4),
        })

    if rows:
        rows = _cap_game_bankroll_rows(rows, cfg)
        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)
        log.info("Saved %d MLB game predictions to bets.mlb_game_predictions", len(rows))
        try:
            with psycopg2.connect(cfg.pg_dsn) as ledger_conn:
                locked = insert_game_bankroll_ledger(ledger_conn, rows, fd_links=fd_links, cfg=cfg)
            log.info("Locked %d MLB game bankroll ledger rows", locked)
        except Exception:
            log.exception("Failed to lock MLB game bankroll ledger rows")
        try:
            with psycopg2.connect(cfg.pg_dsn) as ledger_conn:
                locked = insert_game_model_pick_ledger(ledger_conn, rows, fd_links=fd_links, cfg=cfg)
            log.info("Locked %d MLB game model-pick ledger rows", locked)
        except Exception:
            log.exception("Failed to lock MLB game model-pick ledger rows")


def _fmt_run_diff(run_diff_home: float, home: str, away: str) -> str:
    """Format run differential as a spread label (positive = home favored)."""
    if run_diff_home >= 0:
        return f"{home} -{abs(run_diff_home):.1f}"
    return f"{away} -{abs(run_diff_home):.1f}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Predict today's MLB game run lines and totals")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date in YYYY-MM-DD format (ET). Defaults to today.")
    args = parser.parse_args()

    cfg = PredictConfig()
    if args.date:
        et_day = date.fromisoformat(args.date)
    else:
        et_day = cfg.et_date or datetime.now(_ET).date()

    log.info("Predicting MLB games for ET date=%s", et_day)

    models, feature_cols, feature_medians = _load_models(cfg)
    calib = _load_calibration(cfg.model_dir)
    total_side_recalibrators = _load_game_total_side_recalibrators(
        cfg.model_dir,
        cfg.total_side_recalibrators_file,
    )

    engine = create_engine(cfg.pg_dsn)

    # Load prediction features
    with engine.connect() as conn:
        df = pd.read_sql(
            text(SQL_GAMES_FOR_DATE),
            conn,
            params={"game_date": et_day},
        )

    if df.empty:
        log.warning("No games found for %s in features.mlb_game_prediction_features", et_day)
        return

    # Load starting pitcher names from raw table
    sp_map: dict[str, dict[str, str]] = {}  # game_slug → {"home": name, "away": name}
    try:
        with engine.connect() as conn:
            sp_rows = conn.execute(
                text(SQL_STARTING_PITCHERS),
                {"game_date": et_day},
            ).fetchall()
        for row in sp_rows:
            slug = row.game_slug
            if slug not in sp_map:
                sp_map[slug] = {}
            sp_map[slug][row.side] = row.pitcher_name
        if sp_map:
            log.info("Loaded starting pitchers for %d games", len(sp_map))
    except Exception as exc:
        log.debug("Could not load starting pitchers (table may not exist yet): %s", exc)

    # Load FanDuel deeplinks for today's games (present after includeLinks=true crawls)
    fd_links: dict[tuple[str, str], object] = {}
    try:
        with engine.connect() as conn:
            fd_rows = conn.execute(text(SQL_FANDUEL_LINKS), {"d": et_day}).fetchall()
        for row in fd_rows:
            key = (row.home_abbr, row.away_abbr)
            if key not in fd_links:
                fd_links[key] = row
        if fd_links:
            log.info("Loaded FanDuel deeplinks for %d games", len(fd_links))
    except Exception as exc:
        log.debug("Could not load FanDuel links: %s", exc)

    id_df, X = _prep_features(df, feature_cols=feature_cols, feature_medians=feature_medians)

    # Generate direct predictions
    rl_direct  = models["rl_direct"]
    tot_direct = models["total_direct"]

    pred_run_diff_direct = rl_direct.predict(X)
    pred_total_direct    = tot_direct.predict(X)

    # Blend LightGBM 50/50 with XGB for direct predictions if LGB models loaded
    if "rl_direct_lgb" in models:
        lgb_rl_pred = models["rl_direct_lgb"].predict(X.values)
        pred_run_diff_direct = 0.5 * pred_run_diff_direct + 0.5 * lgb_rl_pred
        log.info("Blended LGB+XGB for direct run-line prediction")
    if "total_direct_lgb" in models:
        lgb_tot_pred = models["total_direct_lgb"].predict(X.values)
        pred_total_direct = 0.5 * pred_total_direct + 0.5 * lgb_tot_pred
        log.info("Blended LGB+XGB for direct total prediction")

    pred_run_diff_final = pred_run_diff_direct.copy()
    pred_total_final    = pred_total_direct.copy()
    used_market = np.zeros(len(df), dtype=bool)

    has_resid_models = ("rl_resid" in models) and ("total_resid" in models)

    w_rl    = _compute_blend_weight_run_line(calib)
    w_total = _compute_blend_weight_total(calib)

    if has_resid_models and "run_line_home" in df.columns and "total_line" in df.columns:
        mkt_rl  = pd.to_numeric(df["run_line_home"], errors="coerce")
        mkt_tot = pd.to_numeric(df["total_line"],    errors="coerce")
        ok = mkt_rl.notna() & mkt_tot.notna()

        if ok.any():
            rl_resid  = models["rl_resid"]
            tot_resid = models["total_resid"]

            pred_rl_resid  = rl_resid.predict(X.loc[ok])
            pred_tot_resid = tot_resid.predict(X.loc[ok])

            resid_recon_rl  = mkt_rl.loc[ok].astype(float).values  + pred_rl_resid
            resid_recon_tot = mkt_tot.loc[ok].astype(float).values + pred_tot_resid

            pred_run_diff_final = pred_run_diff_final.copy()
            pred_total_final    = pred_total_final.copy()

            pred_run_diff_final[ok.values] = (
                (1.0 - w_rl) * pred_run_diff_direct[ok.values] + w_rl * resid_recon_rl
            )
            pred_total_final[ok.values] = (
                (1.0 - w_total) * pred_total_direct[ok.values] + w_total * resid_recon_tot
            )

            log.info(
                "Blended predictions for %d/%d games (rl: %.0f%% resid | total: %.0f%% resid)",
                ok.sum(), len(df), w_rl * 100, w_total * 100,
            )
            used_market = ok.values

    # Bias correction
    bias_rl, bias_total = _compute_live_bias(engine)
    pred_run_diff_final = pred_run_diff_final - bias_rl
    pred_total_final    = pred_total_final    - bias_total

    # Per-team bias correction
    team_bias = _compute_team_bias(engine)
    if team_bias:
        home_teams = id_df["home_team_abbr"].values
        away_teams = id_df["away_team_abbr"].values
        for i, (ht, at) in enumerate(zip(home_teams, away_teams)):
            home_adj = team_bias.get(ht, 0.0)
            away_adj = team_bias.get(at, 0.0)
            pred_run_diff_final[i] -= (home_adj - away_adj)

    # Shrink extreme predictions — reduces overconfidence on high-edge bets.
    # 10% linear shrinkage of predictions exceeding ±2.5 runs.
    # At |pred|=3.5: shrinks to 3.40. At |pred|=4.0: shrinks to 3.85.
    _SHRINK_ABOVE = 2.5
    _SHRINK_RATE  = 0.10
    _abs = np.abs(pred_run_diff_final)
    _excess = np.maximum(_abs - _SHRINK_ABOVE, 0.0)
    pred_run_diff_final = np.sign(pred_run_diff_final) * (_abs - _SHRINK_RATE * _excess)

    # Clip predictions to reasonable MLB ranges
    pred_run_diff_final = np.clip(pred_run_diff_final, -15.0, 15.0)
    pred_total_final    = np.clip(pred_total_final,     1.0,  30.0)

    p_home_cover_clf = None
    p_total_over_clf = None
    clf_feature_cols = models.get("_game_clf_feature_cols")
    clf_feature_medians = models.get("_game_clf_feature_medians")
    if clf_feature_cols and clf_feature_medians and (
        "run_line_cover_clf" in models or "total_over_clf" in models
    ):
        _, X_game_clf = _prep_features(
            df,
            feature_cols=list(clf_feature_cols),
            feature_medians=dict(clf_feature_medians),
        )
        if "run_line_cover_clf" in models:
            p_home_cover_clf = models["run_line_cover_clf"].predict_proba(X_game_clf)[:, 1]
        if "total_over_clf" in models:
            p_total_over_clf = models["total_over_clf"].predict_proba(X_game_clf)[:, 1]

    # F5 total prediction (optional — no market line comparison yet)
    pred_f5_final = None
    if "f5_direct" in models:
        pred_f5 = models["f5_direct"].predict(X)
        if "f5_direct_lgb" in models:
            pred_f5 = 0.5 * pred_f5 + 0.5 * models["f5_direct_lgb"].predict(X.values)
            log.info("Blended LGB+XGB for F5 prediction")
        pred_f5_final = np.clip(pred_f5, 0.0, 20.0)

    # Quantile predictions for game-specific sigma and P(over/cover)
    _QUANTILE_INTS = [10, 25, 50, 75, 90]
    _q_rl_arrays  = {qi: models[f"rl_q{qi:02d}"].predict(X.values)
                     for qi in _QUANTILE_INTS if f"rl_q{qi:02d}" in models}
    _q_tot_arrays = {qi: models[f"tot_q{qi:02d}"].predict(X.values)
                     for qi in _QUANTILE_INTS if f"tot_q{qi:02d}" in models}
    if _q_rl_arrays or _q_tot_arrays:
        log.info("Computed quantile predictions: %d rl quantiles, %d tot quantiles",
                 len(_q_rl_arrays), len(_q_tot_arrays))

    # Build output frame
    out = id_df.copy()
    out["pred_run_diff"]    = np.round(pred_run_diff_final, 2)
    out["pred_total"]       = np.round(pred_total_final,    2)
    if pred_f5_final is not None:
        out["pred_f5_total"] = np.round(pred_f5_final, 2)
    out["used_market_recon"] = used_market
    out["p_home_cover_clf"] = np.round(p_home_cover_clf, 4) if p_home_cover_clf is not None else np.nan
    out["p_total_over_clf"] = np.round(p_total_over_clf, 4) if p_total_over_clf is not None else np.nan

    # Per-game sigma from quantile IQR
    _n = len(out)
    if _q_rl_arrays:
        out["sigma_q_rl"] = [
            _sigma_from_quantiles({qi: float(_q_rl_arrays[qi][i]) for qi in _q_rl_arrays},
                                  fallback=calib.get("direct_spread_rmse", 3.5))
            for i in range(_n)
        ]
    else:
        out["sigma_q_rl"] = np.nan

    if _q_tot_arrays:
        out["sigma_q_total"] = [
            _sigma_from_quantiles({qi: float(_q_tot_arrays[qi][i]) for qi in _q_tot_arrays},
                                  fallback=calib.get("direct_total_rmse", 3.0))
            for i in range(_n)
        ]
    else:
        out["sigma_q_total"] = np.nan

    # Attach SP names
    out["home_sp_name"] = out["game_slug"].map(lambda s: sp_map.get(s, {}).get("home"))
    out["away_sp_name"] = out["game_slug"].map(lambda s: sp_map.get(s, {}).get("away"))
    # Gate bets on confirmed SPs — if either SP is unknown the model is flying blind
    out["both_sp_known"] = out["game_slug"].map(
        lambda s: bool(sp_map.get(s, {}).get("home") and sp_map.get(s, {}).get("away"))
    )

    # Compute edges vs market lines
    if "run_line_home" in df.columns:
        out["market_run_line"] = pd.to_numeric(df["run_line_home"].values, errors="coerce")
        out["market_total"]    = pd.to_numeric(df["total_line"].values,    errors="coerce") if "total_line" in df.columns else np.nan
        # edge_run_line > 0 → home covers -1.5 (home wins by more than 1.5)
        out["edge_run_line"] = np.where(
            out["market_run_line"].notna(),
            out["pred_run_diff"] + out["market_run_line"],
            np.nan,
        )
        out["edge_total"] = np.where(
            out["market_total"].notna(),
            out["pred_total"] - out["market_total"],
            np.nan,
        )
    else:
        out["market_run_line"] = np.nan
        out["market_total"]    = np.nan
        out["edge_run_line"]   = np.nan
        out["edge_total"]      = np.nan

    out["edge_run_line_prob"] = [
        _prob_edge_for_side(
            out.iloc[i].get("p_home_cover_clf"),
            "home" if pd.notna(out.iloc[i].get("edge_run_line")) and float(out.iloc[i]["edge_run_line"]) > 0 else "away",
        )
        if pd.notna(out.iloc[i].get("edge_run_line")) else np.nan
        for i in range(len(out))
    ]
    out["edge_total_prob"] = [
        _prob_edge_for_side(
            out.iloc[i].get("p_total_over_clf"),
            "over" if pd.notna(out.iloc[i].get("edge_total")) and float(out.iloc[i]["edge_total"]) > 0 else "under",
        )
        if pd.notna(out.iloc[i].get("edge_total")) else np.nan
        for i in range(len(out))
    ]
    out["signal_rl_price"] = [
        _signal_run_line_price(
            fd_links.get((out.iloc[i]["home_team_abbr"], out.iloc[i]["away_team_abbr"])),
            out.iloc[i].get("edge_run_line"),
        )
        for i in range(len(out))
    ]
    out["signal_total_price"] = [
        _signal_total_price(
            fd_links.get((out.iloc[i]["home_team_abbr"], out.iloc[i]["away_team_abbr"])),
            out.iloc[i].get("edge_total"),
        )
        for i in range(len(out))
    ]

    # P(over/cover) from quantile CDF
    if _q_rl_arrays:
        out["p_over_rl_q"] = [
            (_p_over_from_quantiles({qi: float(_q_rl_arrays[qi][i]) for qi in _q_rl_arrays},
                                    float(out.iloc[i]["market_run_line"]))
             if pd.notna(out.iloc[i]["market_run_line"]) else np.nan)
            for i in range(_n)
        ]
    else:
        out["p_over_rl_q"] = np.nan

    if _q_tot_arrays:
        out["p_over_total_q"] = [
            (_p_over_from_quantiles({qi: float(_q_tot_arrays[qi][i]) for qi in _q_tot_arrays},
                                    float(out.iloc[i]["market_total"]))
             if pd.notna(out.iloc[i]["market_total"]) else np.nan)
            for i in range(_n)
        ]
    else:
        out["p_over_total_q"] = np.nan

    # Count bet signals (only for games with confirmed SPs)
    sp_mask = out["both_sp_known"] if "both_sp_known" in out.columns else pd.Series(True, index=out.index)
    n_rl_bets = (
        int(sum(_run_line_signal_allowed(
                    v,
                    cfg,
                    out["p_home_cover_clf"].iloc[i],
                    out["signal_rl_price"].iloc[i],
                    out["market_run_line"].iloc[i],
                ) and bool(sp_mask.iloc[i])
                for i, v in enumerate(out["edge_run_line"])))
        if "edge_run_line" in out else 0
    )
    n_total_bets = (
        int(sum(_total_signal_allowed(
                    v,
                    cfg,
                    out["p_total_over_clf"].iloc[i],
                    out["market_total"].iloc[i],
                    out["signal_total_price"].iloc[i],
                ) and bool(sp_mask.iloc[i])
                for i, v in enumerate(out["edge_total"])))
        if "edge_total" in out else 0
    )
    n_high_edge  = n_rl_bets + n_total_bets

    if out["used_market_recon"].any():
        model_note = (
            f"rl resid {w_rl:.0%} | total resid {w_total:.0%}"
            if (w_rl > 0 or w_total > 0)
            else "direct only (resid quality gate failed)"
        )
    else:
        model_note = "direct only"

    discord = os.getenv("DISCORD_FORMAT") == "1"

    summary_line = (
        f"{'⚾ ' if discord else ''}{et_day} — {len(out)} games  "
        f"{n_high_edge} high-edge bets ({n_rl_bets} run-line, {n_total_bets} total)  "
        f"[{model_note}]"
    )
    print(summary_line)

    # Discord: print "BETS TODAY" summary block before per-game detail
    compact_discord = False
    if discord:
        compact_discord = True
        _bets_today: list[dict] = []
        _rl_bet_links: list[str] = []
        _total_bet_links: list[str] = []
        for _, _r in out.iterrows():
            _edge_rl = _r.get("edge_run_line")
            _edge_t  = _r.get("edge_total")
            _p_home_clf = _clean_prob(_r.get("p_home_cover_clf"))
            _p_total_clf = _clean_prob(_r.get("p_total_over_clf"))
            _both_sp = bool(_r.get("both_sp_known", True))
            _home2   = _r["home_team_abbr"]
            _away2   = _r["away_team_abbr"]
            _fd2     = fd_links.get((_home2, _away2))
            _ub      = bool(_r.get("used_market_recon", False))
            _sq_rl2  = _r.get("sigma_q_rl")
            _sq_t2   = _r.get("sigma_q_total")
            _srl     = (float(_sq_rl2) if (_sq_rl2 is not None and pd.notna(_sq_rl2)) else
                        calib.get("resid_spread_rmse" if (_ub and w_rl > 0) else "direct_spread_rmse", 3.5))
            _st      = (float(_sq_t2) if (_sq_t2 is not None and pd.notna(_sq_t2)) else
                        calib.get("resid_total_rmse"  if (_ub and w_total > 0) else "direct_total_rmse",  3.0))
            _sr2     = _r.get("start_ts_utc")
            if pd.notna(_sr2):
                _t2 = pd.to_datetime(_sr2, utc=True).tz_convert(_ET).strftime("%I:%M %p ET").lstrip("0")
            else:
                _t2 = "TBD"
            _mrl2 = _r.get("market_run_line")
            _mt2  = _r.get("market_total")

            _rl_price2 = _signal_run_line_price(_fd2, _edge_rl)
            _tot_price2 = _signal_total_price(_fd2, _edge_t)
            _rl_bankroll_gate = _run_line_signal_allowed(_edge_rl, cfg, _p_home_clf, _rl_price2, _mrl2)
            _total_bankroll_gate = _total_signal_allowed(_edge_t, cfg, _p_total_clf, _mt2, _tot_price2)

            if _both_sp and pd.notna(_edge_rl) and abs(float(_edge_rl)) > 1e-9:
                _e  = float(_edge_rl)
                _bt = _home2 if _e > 0 else _away2
                _vs = _away2 if _e > 0 else _home2
                _ml = _run_line_label_for_side(_mrl2, "home" if _e > 0 else "away")
                _p_side = _p_home_clf if _e > 0 else (1.0 - _p_home_clf if _p_home_clf is not None else None)
                _k, _p = _kelly_from_prob(_p_side, _rl_price2)
                if _k is None or _p is None:
                    _, _p = _kelly(abs(_e), sigma=_srl)
                    _k, _p = _kelly_from_prob(_p, _rl_price2)
                _bankroll = _game_bankroll_assessment(
                    market="run_line",
                    side="home" if _e > 0 else "away",
                    cfg=cfg,
                    kelly_fraction=_k,
                    win_prob=_p,
                    market_price=_rl_price2,
                    both_sp_known=_both_sp,
                    market_line=_mrl2,
                )
                if not _rl_bankroll_gate:
                    _bankroll = BankrollAssessment(
                        tier="paper",
                        candidate=False,
                        reasons=_append_bankroll_reason(_bankroll.reasons, "outside_bankroll_gate"),
                        stake_pct=0.0,
                    )
                _lnk = (_fd2.spread_home_link if _e > 0 else _fd2.spread_away_link) if _fd2 else None
                if _lnk:
                    _rl_bet_links.append(_lnk)
                _bets_today.append({
                    "desc": f"**{_bt} {_ml}** (vs {_vs} · {_t2})",
                    "edge": abs(_e), "p": _p, "qk": (_k / 4) * 1000, "link": _lnk,
                    "assessment": _bankroll,
                    "bankroll": bankroll_tag(_bankroll),
                    "stake": _bankroll.stake_pct * 1000,
                })
            if (
                _both_sp
                and pd.notna(_edge_t)
                and float(_edge_t) >= 0
            ):
                _e   = float(_edge_t)
                _mtl = f"{float(_mt2):.1f}" if pd.notna(_mt2) else "?"
                _k, _p, _cal_key = _calibrated_total_prob_kelly(
                    side="over",
                    raw_p_side=_p_total_clf,
                    edge_abs=_e,
                    sigma=_st,
                    market_total=_mt2,
                    market_price=_tot_price2,
                    recalibrators=total_side_recalibrators,
                )
                _bankroll = _game_bankroll_assessment(
                    market="total",
                    side="over",
                    cfg=cfg,
                    kelly_fraction=_k,
                    win_prob=_p,
                    market_price=_tot_price2,
                    both_sp_known=_both_sp,
                    market_line=_mt2,
                )
                if not _total_bankroll_gate:
                    _bankroll = BankrollAssessment(
                        tier="paper",
                        candidate=False,
                        reasons=_append_bankroll_reason(_bankroll.reasons, "outside_bankroll_gate"),
                        stake_pct=0.0,
                    )
                _lnk = _fd2.total_over_link if _fd2 else None
                if _lnk:
                    _total_bet_links.append(_lnk)
                _bets_today.append({
                    "desc": f"**OVER {_mtl}** ({_away2} @ {_home2} · {_t2})",
                    "edge": _e, "p": _p, "qk": (_k / 4) * 1000, "link": _lnk,
                    "assessment": _bankroll,
                    "bankroll": bankroll_tag(_bankroll),
                    "stake": _bankroll.stake_pct * 1000,
                })
            elif (
                _both_sp
                and pd.notna(_edge_t)
                and float(_edge_t) < 0
            ):
                _e   = -float(_edge_t)
                _mtl = f"{float(_mt2):.1f}" if pd.notna(_mt2) else "?"
                _p_side = 1.0 - _p_total_clf if _p_total_clf is not None else None
                _k, _p, _cal_key = _calibrated_total_prob_kelly(
                    side="under",
                    raw_p_side=_p_side,
                    edge_abs=_e,
                    sigma=_st,
                    market_total=_mt2,
                    market_price=_tot_price2,
                    recalibrators=total_side_recalibrators,
                )
                _bankroll = _game_bankroll_assessment(
                    market="total",
                    side="under",
                    cfg=cfg,
                    kelly_fraction=_k,
                    win_prob=_p,
                    market_price=_tot_price2,
                    both_sp_known=_both_sp,
                    market_line=_mt2,
                )
                if not _total_bankroll_gate:
                    _bankroll = BankrollAssessment(
                        tier="paper",
                        candidate=False,
                        reasons=_append_bankroll_reason(_bankroll.reasons, "outside_bankroll_gate"),
                        stake_pct=0.0,
                    )
                _lnk = _fd2.total_under_link if _fd2 else None
                if _lnk:
                    _total_bet_links.append(_lnk)
                _bets_today.append({
                    "desc": f"**UNDER {_mtl}** ({_away2} @ {_home2} · {_t2})",
                    "edge": _e, "p": _p, "qk": (_k / 4) * 1000, "link": _lnk,
                    "assessment": _bankroll,
                    "bankroll": bankroll_tag(_bankroll),
                    "stake": _bankroll.stake_pct * 1000,
                })

        _bets_today.sort(key=lambda x: x["edge"], reverse=True)
        if _bets_today:
            _bankroll_bets = [b for b in _bets_today if b["assessment"].candidate]
            _model_bets = _bets_today[:cfg.top_n_game_bets]
            _paper_bets = [b for b in _model_bets if not b["assessment"].candidate]

            def _print_game_bet(_b: dict, *, include_link: bool) -> None:
                _ls = f"  [Bet FD](<{_b['link']}>)" if include_link and _b["link"] else ""
                _p = f"p={_b['p']:.0%}" if _b.get("p") is not None else "p=?"
                print(
                    f"- {_b['desc']}  +{_b['edge']:.2f}  {_p}  "
                    f"[{bankroll_tag(_b['assessment'])}] stake=${_b['assessment'].stake_pct * 1000:.0f}/$1k{_ls}"
                )

            def _print_chunked_parlays(title: str, links: list[str]) -> None:
                dedup = list(dict.fromkeys([l for l in links if l]))
                if len(dedup) < 2:
                    return
                n_chunks = math.ceil(len(dedup) / 25)
                for i in range(0, len(dedup), 25):
                    url = build_fd_parlay_url(dedup[i:i + 25])
                    if not url:
                        continue
                    sfx = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
                    print(f"\n**{title}{sfx}** [FD]({url})")

            if _bankroll_bets:
                print(f"\n**BANKROLL BETS ({len(_bankroll_bets)})**")
                for _b in _bankroll_bets:
                    _print_game_bet(_b, include_link=True)
            else:
                print("\n**BANKROLL BETS**")
                print("- No bankroll-qualified game bets today")

            _print_chunked_parlays(
                "Bankroll Game Bets Parlay",
                [b["link"] for b in _bankroll_bets if b.get("link")],
            )

            _game_exposure = sum(float(b["assessment"].stake_pct or 0.0) for b in _bankroll_bets)
            _prop_exposure = _existing_prop_bankroll_exposure(engine, et_day)
            _combined_exposure = _game_exposure + _prop_exposure
            _cap = cfg.bankroll_max_daily_exposure_pct
            if _combined_exposure > _cap + 1e-12:
                print(
                    f"- GLOBAL CAP WARNING: games + props total {_combined_exposure:.2%} "
                    f"vs daily cap {_cap:.2%} (over by {_combined_exposure - _cap:.2%})"
                )

            if _paper_bets:
                print(f"\n**PAPER / RESEARCH ({len(_paper_bets)} of {len(_model_bets)} model picks)**")
                for _b in _paper_bets:
                    _print_game_bet(_b, include_link=False)
        else:
            print("\n**No edge bets today**")
        if False and _bets_today:
            print(f"\n**BETS TODAY ({len(_bets_today)})**")
            for _b in _bets_today:
                _ls = f"  [Bet FD](<{_b['link']}>)" if _b["link"] else ""
                print(f"• {_b['desc']}  +{_b['edge']:.2f}  p={_b['p']:.0%}  [{_b['bankroll']}] stake=${_b['stake']:.0f}/$1k{_ls}")

            def _print_chunked_parlays(title: str, links: list[str]) -> None:
                dedup = list(dict.fromkeys([l for l in links if l]))
                if not dedup:
                    return
                n_chunks = math.ceil(len(dedup) / 25)
                for i in range(0, len(dedup), 25):
                    url = build_fd_parlay_url(dedup[i:i + 25])
                    if not url:
                        continue
                    sfx = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
                    print(f"\n**{title}{sfx}** [FD]({url})")

            # Compact mobile mode: one combined parlay of all high-edge run line + total bets.
            _print_chunked_parlays("All Run Line + Total Bets Parlay", _rl_bet_links + _total_bet_links)
        elif False:
            print("\n**No edge bets today**")
        print("")

    best_links: list[str | None] = []      # FD links for high-edge bets (best bets parlay)

    if compact_discord:
        # In compact Discord mode, summary list above is the primary output.
        # Skip verbose per-game sections that are hard to read on mobile.
        try:
            _save_predictions(
                out,
                engine,
                et_day,
                cfg,
                calib=calib,
                w_rl=w_rl,
                w_total=w_total,
                fd_links=fd_links,
                total_side_recalibrators=total_side_recalibrators,
            )
        except Exception:
            log.exception("Failed to save predictions")
        return

    for _, r in out.iterrows():
        start_raw = r.get("start_ts_utc")
        if pd.notna(start_raw):
            start = pd.to_datetime(start_raw, utc=True).tz_convert(_ET)
            time_str = start.strftime("%I:%M %p ET").lstrip("0")
        else:
            time_str = "TBD"

        home = r["home_team_abbr"]
        away = r["away_team_abbr"]

        home_sp = r.get("home_sp_name") or "TBD"
        away_sp = r.get("away_sp_name") or "TBD"

        _ld = fd_links.get((home, away))  # FanDuel deeplink row for this game

        if discord:
            print(f"\n**{away} @ {home}** · {time_str}")
        else:
            print(f"\n{away} @ {home}  {time_str}")

        if not (home_sp == "TBD" and away_sp == "TBD"):
            print(f"  SP: {home_sp} (home) vs {away_sp} (away)")

        pred_rd  = float(r["pred_run_diff"])
        pred_tot = float(r["pred_total"])

        # Pick sigma: prefer game-specific quantile sigma, fall back to calibration RMSE
        used_blend = bool(r.get("used_market_recon", False))
        _sq_rl = r.get("sigma_q_rl")
        _sq_t  = r.get("sigma_q_total")
        sigma_rl = (float(_sq_rl) if (_sq_rl is not None and pd.notna(_sq_rl)) else
                    calib.get("resid_spread_rmse" if (used_blend and w_rl > 0) else "direct_spread_rmse", 3.5))
        sigma_t  = (float(_sq_t) if (_sq_t is not None and pd.notna(_sq_t)) else
                    calib.get("resid_total_rmse"  if (used_blend and w_total > 0) else "direct_total_rmse",  3.0))

        # Pred label
        run_line_label = _fmt_run_diff(pred_rd, home, away)
        _f5_str = ""
        if "pred_f5_total" in r.index and pd.notna(r.get("pred_f5_total")):
            _f5_str = f" | F5: {float(r['pred_f5_total']):.1f}"
        pred_label = f"Pred: {run_line_label} | Total: {pred_tot:.1f}{_f5_str}"
        print(f"  {pred_label}")

        # Run line edge
        edge_rl   = r.get("edge_run_line")
        mkt_rl    = r.get("market_run_line")
        both_sp   = bool(r.get("both_sp_known", True))
        p_home_clf = _clean_prob(r.get("p_home_cover_clf"))

        _rl_price = _signal_run_line_price(_ld, edge_rl)

        if both_sp and _run_line_signal_allowed(edge_rl, cfg, p_home_clf, _rl_price, mkt_rl):
            e_rl = float(edge_rl)
            bet_side = "HOME" if e_rl > 0 else "AWAY"
            bet_team = home if e_rl > 0 else away
            p_side = p_home_clf if e_rl > 0 else (1.0 - p_home_clf if p_home_clf is not None else None)
            kelly, p_win = _kelly_from_prob(p_side, _rl_price)
            if kelly is None or p_win is None:
                _, p_win = _kelly(abs(e_rl), sigma=sigma_rl)
                kelly, p_win = _kelly_from_prob(p_win, _rl_price)
            qk_bet = (kelly / 4) * 1000
            side_key = "home" if e_rl > 0 else "away"
            bankroll = _game_bankroll_assessment(
                market="run_line",
                side=side_key,
                cfg=cfg,
                kelly_fraction=kelly,
                win_prob=p_win,
                market_price=_rl_price,
                both_sp_known=both_sp,
                market_line=mkt_rl,
            )
            bankroll_label = bankroll_tag(bankroll)
            mkt_label = _run_line_label_for_side(mkt_rl, "home" if e_rl > 0 else "away")
            # FD link: home covers → spread_home_link; away covers → spread_away_link
            _sl = (_ld.spread_home_link if e_rl > 0 else _ld.spread_away_link) if _ld else None
            best_links.append(_sl)
            _link_str = f"  [Bet FD](<{_sl}>)" if (_sl and discord) else ""
            if discord:
                print(f"  Run line: {bet_team} {mkt_label}  * **EDGE +{abs(e_rl):.2f}  [bet {bet_side}] [{bankroll_label}]**{_link_str}")
            else:
                print(f"  Run line: {bet_team} {mkt_label}  * EDGE +{abs(e_rl):.2f}  [bet {bet_side}] [{bankroll_label}]")
                print(f"    Kelly: p={p_win:.1%}  full={kelly:.1%}  raw 1/4 Kelly = ${qk_bet:.0f}; bankroll stake = ${bankroll.stake_pct * 1000:.0f} per $1,000")
        elif pd.notna(edge_rl) and abs(float(edge_rl)) >= cfg.min_edge_run_line and not both_sp:
            side = "home" if float(edge_rl) > 0 else "away"
            mkt_label = _run_line_label_for_side(mkt_rl, side)
            pred_side_label = home if side == "home" else away
            _sl_no_sp = ((_ld.spread_home_link if float(edge_rl) > 0 else _ld.spread_away_link) if _ld else None)
            print(f"  Run line: {pred_side_label} {mkt_label}  (edge +{abs(float(edge_rl)):.2f} — SP TBD, bet suppressed)")
        elif pd.notna(mkt_rl):
            side = "home" if pred_rd >= 0 else "away"
            mkt_label = _run_line_label_for_side(mkt_rl, side)
            pred_side_label = home if side == "home" else away
            _sl_no_edge = (_ld.spread_home_link if side == "home" else _ld.spread_away_link) if _ld else None
            _link_str = f"  [FD](<{_sl_no_edge}>)" if (_sl_no_edge and discord) else ""
            print(f"  Run line: {pred_side_label} {mkt_label}{_link_str}")
        else:
            print(f"  Pred run diff: {pred_rd:+.1f}")

        # Total edge
        edge_t  = r.get("edge_total")
        mkt_tot = r.get("market_total")
        p_total_clf = _clean_prob(r.get("p_total_over_clf"))

        _tot_price = _signal_total_price(_ld, edge_t)

        if (
            both_sp
            and _total_signal_allowed(edge_t, cfg, p_total_clf, mkt_tot, _tot_price)
            and pd.notna(edge_t)
            and float(edge_t) >= 0
        ):
            e_t = float(edge_t)
            kelly, p_win, _cal_key = _calibrated_total_prob_kelly(
                side="over",
                raw_p_side=p_total_clf,
                edge_abs=e_t,
                sigma=sigma_t,
                market_total=mkt_tot,
                market_price=_tot_price,
                recalibrators=total_side_recalibrators,
            )
            qk_bet = (kelly / 4) * 1000
            bankroll = _game_bankroll_assessment(
                market="total",
                side="over",
                cfg=cfg,
                kelly_fraction=kelly,
                win_prob=p_win,
                market_price=_tot_price,
                both_sp_known=both_sp,
                market_line=mkt_tot,
            )
            bankroll_label = bankroll_tag(bankroll)
            mkt_t_label = f"{float(mkt_tot):.1f}" if pd.notna(mkt_tot) else "n/a"
            _tl = _ld.total_over_link if _ld else None
            best_links.append(_tl)
            _link_str = f"  [Bet FD](<{_tl}>)" if (_tl and discord) else ""
            if discord:
                print(f"  Total: OVER {mkt_t_label}  * **EDGE +{e_t:.2f}  [bet OVER] [{bankroll_label}]**{_link_str}")
            else:
                print(f"  Total: OVER {mkt_t_label}  * EDGE +{e_t:.2f}  [bet OVER] [{bankroll_label}]")
                print(f"    Kelly: p={p_win:.1%}  full={kelly:.1%}  raw 1/4 Kelly = ${qk_bet:.0f}; bankroll stake = ${bankroll.stake_pct * 1000:.0f} per $1,000")
        elif (
            both_sp
            and _total_signal_allowed(edge_t, cfg, p_total_clf, mkt_tot, _tot_price)
            and pd.notna(edge_t)
            and float(edge_t) < 0
        ):
            e_t = float(edge_t)
            p_side = 1.0 - p_total_clf if p_total_clf is not None else None
            kelly, p_win, _cal_key = _calibrated_total_prob_kelly(
                side="under",
                raw_p_side=p_side,
                edge_abs=-e_t,
                sigma=sigma_t,
                market_total=mkt_tot,
                market_price=_tot_price,
                recalibrators=total_side_recalibrators,
            )
            qk_bet = (kelly / 4) * 1000
            bankroll = _game_bankroll_assessment(
                market="total",
                side="under",
                cfg=cfg,
                kelly_fraction=kelly,
                win_prob=p_win,
                market_price=_tot_price,
                both_sp_known=both_sp,
                market_line=mkt_tot,
            )
            bankroll_label = bankroll_tag(bankroll)
            mkt_t_label = f"{float(mkt_tot):.1f}" if pd.notna(mkt_tot) else "n/a"
            _tl = _ld.total_under_link if _ld else None
            best_links.append(_tl)
            _link_str = f"  [Bet FD](<{_tl}>)" if (_tl and discord) else ""
            if discord:
                print(f"  Total: UNDER {mkt_t_label}  * **EDGE +{-e_t:.2f}  [bet UNDER] [{bankroll_label}]**{_link_str}")
            else:
                print(f"  Total: UNDER {mkt_t_label}  * EDGE +{-e_t:.2f}  [bet UNDER] [{bankroll_label}]")
                print(f"    Kelly: p={p_win:.1%}  full={kelly:.1%}  raw 1/4 Kelly = ${qk_bet:.0f}; bankroll stake = ${bankroll.stake_pct * 1000:.0f} per $1,000")
        elif pd.notna(edge_t) and abs(float(edge_t)) >= cfg.min_edge_total and not both_sp:
            mkt_t_label = f"{float(mkt_tot):.1f}" if pd.notna(mkt_tot) else "n/a"
            direction = "OVER" if float(edge_t) > 0 else "UNDER"
            print(f"  Total: {direction} {mkt_t_label}  (edge +{abs(float(edge_t)):.2f} — SP TBD, bet suppressed)")
        elif pd.notna(mkt_tot):
            mkt_t_label = f"{float(mkt_tot):.1f}"
            pred_ou = "O" if pred_tot > float(mkt_tot) else "U"
            _tl_no_edge = (_ld.total_over_link if pred_tot > float(mkt_tot) else _ld.total_under_link) if (_ld and pd.notna(mkt_tot)) else None
            _link_str = f"  [FD](<{_tl_no_edge}>)" if (_tl_no_edge and discord) else ""
            print(f"  Total: {pred_ou}{mkt_t_label}{_link_str}")
        else:
            print(f"  Pred total: {pred_tot:.1f}")

    # Parlay URLs — chunked at 25 legs
    if discord:
        def _game_parlay(title: str, links: list) -> None:
            dedup = list(dict.fromkeys(l for l in links if l))
            if not dedup:
                return
            n_chunks = math.ceil(len(dedup) / 25)
            for i in range(0, len(dedup), 25):
                url = build_fd_parlay_url(dedup[i:i + 25])
                if url:
                    sfx = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
                    print(f"\n**{title}{sfx}** [FD]({url})")

        _game_parlay("Best Bets Parlay", best_links)

    # Save predictions to DB
    try:
        _save_predictions(
            out,
            engine,
            et_day,
            cfg,
            calib=calib,
            w_rl=w_rl,
            w_total=w_total,
            fd_links=fd_links,
            total_side_recalibrators=total_side_recalibrators,
        )
    except Exception as exc:
        log.warning("Could not save predictions to DB: %s", exc)


if __name__ == "__main__":
    main()
