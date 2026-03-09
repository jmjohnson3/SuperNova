import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBRegressor
from sqlalchemy import create_engine, text

from .features import add_game_derived_features

log = logging.getLogger("nba_pipeline.modeling.predict_today")

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = Path(__file__).resolve().parent / "models"
    season: str | None = None
    et_date: date | None = None
    min_edge_spread: float = 6.0  # minimum |pred - market| to flag a spread bet
    min_edge_total: float = 5.0   # minimum |pred - market| to flag a total bet


def _compute_blend_weight_spread(calib: dict) -> float:
    """Inverse-MAE blend weight for the *spread* residual model.

    Quality gate: only blend if resid MAE is ≥10% better (lower) than direct.
    Falls back to 0.0 (direct-only) when gate fails or values are missing.
    """
    direct_mae = calib.get("direct_spread_mae", 10.5)
    resid_mae = calib.get("resid_spread_mae", 99.0)
    if direct_mae <= 0 or resid_mae <= 0:
        return 0.0
    if resid_mae >= direct_mae * 0.90:
        log.info(
            "Spread residual quality gate failed (resid MAE %.3f >= 90%% of direct %.3f). Direct only.",
            resid_mae, direct_mae,
        )
        return 0.0
    w_d = 1.0 / direct_mae
    w_r = 1.0 / resid_mae
    return round(w_r / (w_d + w_r), 4)


def _compute_blend_weight_total(calib: dict) -> float:
    """Inverse-MAE blend weight for the *total* residual model.

    Quality gate: only blend if resid MAE is ≥10% better (lower) than direct.
    Falls back to 0.0 (direct-only) when gate fails or values are missing.
    """
    direct_mae = calib.get("direct_total_mae", 15.0)
    resid_mae = calib.get("resid_total_mae", 99.0)
    if direct_mae <= 0 or resid_mae <= 0:
        return 0.0
    if resid_mae >= direct_mae * 0.90:
        log.info(
            "Total residual quality gate failed (resid MAE %.3f >= 90%% of direct %.3f). Direct only.",
            resid_mae, direct_mae,
        )
        return 0.0
    w_d = 1.0 / direct_mae
    w_r = 1.0 / resid_mae
    return round(w_r / (w_d + w_r), 4)


def _compute_live_bias(engine) -> tuple[float, float]:
    """Compute rolling mean prediction error from recent graded games.

    Returns (spread_bias, total_bias) — subtract from raw predictions to correct.
    Positive bias means the model over-predicted (pred > actual on average).
    Requires ≥10 graded games in the last 30 days; otherwise returns (0.0, 0.0).
    """
    sql = text("""
        SELECT
            COUNT(*)                               AS n,
            AVG(pred_margin_home - actual_margin_home) AS spread_bias,
            AVG(pred_total       - actual_total)       AS total_bias
        FROM bets.game_predictions
        WHERE actual_margin_home IS NOT NULL
          AND game_date_et >= CURRENT_DATE - INTERVAL '30 days'
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if row is None or row[0] is None or int(row[0]) < 10:
            log.info("Auto bias correction: not enough data (need 10+ graded games in last 30 days).")
            return 0.0, 0.0
        sb = float(row[1] or 0.0)
        tb = float(row[2] or 0.0)
        log.info(
            "Auto bias correction: n=%d  spread_bias=%.2f  total_bias=%.2f",
            int(row[0]), sb, tb,
        )
        return sb, tb
    except Exception as exc:
        log.warning("Could not compute live bias: %s", exc)
        return 0.0, 0.0


def _compute_per_team_bias(engine) -> dict[str, float]:
    """Rolling spread bias indexed by home_team_abbr (last 60 days, min 5 games).

    Returns {home_team_abbr: spread_bias}.  Teams with insufficient data are
    omitted; callers should fall back to the global bias for those.
    """
    sql = text("""
        SELECT home_team_abbr,
               AVG(pred_margin_home - actual_margin_home) AS spread_bias,
               COUNT(*) AS n
        FROM bets.game_predictions
        WHERE actual_margin_home IS NOT NULL
          AND game_date_et >= CURRENT_DATE - INTERVAL '60 days'
        GROUP BY home_team_abbr
        HAVING COUNT(*) >= 5
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        result = {r.home_team_abbr: float(r.spread_bias or 0.0) for r in rows}
        log.info("Per-team spread bias loaded for %d teams", len(result))
        return result
    except Exception as exc:
        log.warning("Could not compute per-team spread bias: %s", exc)
        return {}


SQL_GAMES_FOR_DATE = """
SELECT gpf.*,
       h2h.h2h_meetings_5, h2h.h2h_home_margin_avg5, h2h.h2h_home_win_pct5,
       elo.home_elo, elo.away_elo, elo.elo_diff, elo.elo_win_prob_home
FROM features.game_prediction_features gpf
LEFT JOIN features.team_h2h_features h2h
  ON h2h.season = gpf.season AND h2h.game_slug = gpf.game_slug
LEFT JOIN features.game_elo_features elo
  ON elo.season = gpf.season AND elo.game_slug = gpf.game_slug
WHERE gpf.game_date_et = :game_date
  AND gpf.start_ts_utc IS NOT NULL
  AND (:season IS NULL OR gpf.season = :season)
ORDER BY gpf.start_ts_utc, gpf.game_slug
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
    Returns (id_df, X_aligned)
    - One-hot season/home/away teams (matches training)
    - Derive start_hour_utc/start_dow_utc
    - Fill NaNs using feature_medians
    - Align columns exactly to feature_cols
    """
    id_cols = ["season", "game_slug", "game_date_et", "start_ts_utc", "home_team_abbr", "away_team_abbr"]
    id_df = df[id_cols].copy()

    # Start with all non-id columns
    X = df.drop(columns=[c for c in id_cols if c in df.columns]).copy()

    # Season position: days elapsed since Oct 1 of the season-start year (mirrors training).
    if "game_date_et" in df.columns:
        gdt = pd.to_datetime(df["game_date_et"])
        season_start_year = gdt.dt.year.where(gdt.dt.month >= 7, gdt.dt.year - 1)
        season_start = pd.to_datetime(season_start_year.astype(str) + "-10-01")
        X["season_days_elapsed"] = (gdt - season_start).dt.days.values

    # Derive timing
    if "start_ts_utc" in df.columns:
        ts = pd.to_datetime(df["start_ts_utc"], errors="coerce", utc=True)
        X["start_hour_utc"] = ts.dt.hour
        X["start_dow_utc"] = ts.dt.dayofweek

    # b2b to 0/1
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # one-hot season/home/away
    cat_cols = []
    for c in ("season", "home_team_abbr", "away_team_abbr"):
        if c in df.columns:
            X[c] = df[c]
            cat_cols.append(c)
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # coerce numeric-ish strings
    X = _coerce_numeric_cols(X)

    # Derived interaction features — single source of truth in features.add_game_derived_features.
    X = add_game_derived_features(X)

    # align to training columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        X = X.drop(columns=extra)

    X = X[feature_cols]

    # fill with training medians where available
    for c, med in feature_medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(med)

    # final fill (one-hot columns etc.)
    X = X.fillna(0.0)

    return id_df, X


def _load_models(cfg: PredictConfig) -> tuple[dict[str, XGBRegressor], list[str], dict[str, float]]:
    model_dir = cfg.model_dir
    feature_cols_path = model_dir / "feature_columns.json"
    medians_path = model_dir / "feature_medians.json"

    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing {feature_cols_path}. Run training first.")
    if not medians_path.exists():
        raise FileNotFoundError(f"Missing {medians_path}. Run training first (medians save enabled).")

    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    feature_medians = json.loads(medians_path.read_text(encoding="utf-8"))

    # Load whichever models exist
    paths = {
        "spread_direct": model_dir / "spread_direct_xgb.json",
        "total_direct": model_dir / "total_direct_xgb.json",
        "spread_resid": model_dir / "spread_resid_xgb.json",
        "total_resid": model_dir / "total_resid_xgb.json",
    }

    models: dict[str, XGBRegressor] = {}
    for k, p in paths.items():
        if p.exists():
            m = XGBRegressor()
            m.load_model(str(p))
            models[k] = m

    if "spread_direct" not in models or "total_direct" not in models:
        raise FileNotFoundError(f"Missing direct models in {model_dir}. Run training first.")

    return models, feature_cols, feature_medians


def _load_calibration(model_dir: Path) -> dict:
    """Load walk-forward calibration stats (RMSE, p68, p90) per model type.

    Falls back to conservative defaults (direct RMSE ≈ 14 spread / 20 total)
    if the file doesn't exist (trained before calibration was added).
    """
    p = model_dir / "calibration.json"
    defaults = {
        "direct_spread_rmse": 14.0,
        "direct_total_rmse":  20.0,
        "resid_spread_rmse":  6.5,
        "resid_total_rmse":   6.0,
    }
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {**defaults, **d}
    return defaults


def _validate_game_predictions(
    pred_margin: np.ndarray,
    pred_total: np.ndarray,
    id_df: pd.DataFrame,
) -> None:
    """Warn when raw model outputs fall outside sensible NBA ranges before clipping."""
    for i, (m, t) in enumerate(zip(pred_margin, pred_total)):
        slug = id_df.iloc[i].get("game_slug", i)
        if abs(m) > 35:
            log.warning("Extreme margin prediction game=%s raw_margin=%.1f (will clip to ±40)", slug, m)
        if not (200 <= t <= 260):
            log.warning("Unusual total prediction game=%s raw_total=%.1f (outside 200-260)", slug, t)


def _fmt_spread_from_margin(margin_home_minus_away: float, home: str, away: str) -> str:
    if margin_home_minus_away >= 0:
        return f"{home} -{abs(margin_home_minus_away):.1f}"
    return f"{away} -{abs(margin_home_minus_away):.1f}"


def _ensure_bets_schema(engine) -> None:
    """Create bets.game_predictions table if it doesn't exist."""
    ddl = """
    CREATE SCHEMA IF NOT EXISTS bets;
    CREATE TABLE IF NOT EXISTS bets.game_predictions (
        id                    SERIAL PRIMARY KEY,
        predicted_at_utc      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        game_date_et          DATE        NOT NULL,
        game_slug             TEXT        NOT NULL,
        season                TEXT        NOT NULL,
        home_team_abbr        TEXT        NOT NULL,
        away_team_abbr        TEXT        NOT NULL,
        pred_margin_home      NUMERIC,
        pred_total            NUMERIC,
        used_residual_model   BOOLEAN     DEFAULT FALSE,
        market_spread_home    NUMERIC,
        market_total          NUMERIC,
        edge_spread           NUMERIC,
        edge_total            NUMERIC,
        actual_margin_home    NUMERIC,
        actual_total          NUMERIC,
        spread_bet_side       TEXT,
        total_bet_side        TEXT,
        spread_covered        BOOLEAN,
        total_correct         BOOLEAN,
        direction_correct     BOOLEAN,
        kelly_fraction_spread NUMERIC,
        kelly_fraction_total  NUMERIC,
        win_prob_spread       NUMERIC,
        win_prob_total        NUMERIC,
        UNIQUE (game_date_et, game_slug)
    );
    -- Add columns for older tables that predate this schema version
    ALTER TABLE bets.game_predictions
        ADD COLUMN IF NOT EXISTS kelly_fraction_spread NUMERIC,
        ADD COLUMN IF NOT EXISTS kelly_fraction_total  NUMERIC,
        ADD COLUMN IF NOT EXISTS win_prob_spread       NUMERIC,
        ADD COLUMN IF NOT EXISTS win_prob_total        NUMERIC,
        ADD COLUMN IF NOT EXISTS direction_correct     BOOLEAN,
        ADD COLUMN IF NOT EXISTS closing_spread_home   NUMERIC,
        ADD COLUMN IF NOT EXISTS closing_total         NUMERIC,
        ADD COLUMN IF NOT EXISTS clv_spread            NUMERIC,
        ADD COLUMN IF NOT EXISTS clv_total             NUMERIC;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _save_predictions(
    out: pd.DataFrame,
    engine,
    et_day,
    calib: dict | None = None,
    w_spread: float = 0.0,
    w_total: float = 0.0,
) -> None:
    """Upsert game predictions into bets.game_predictions."""
    _ensure_bets_schema(engine)
    upsert_sql = text("""
        INSERT INTO bets.game_predictions
            (game_date_et, game_slug, season, home_team_abbr, away_team_abbr,
             pred_margin_home, pred_total, used_residual_model,
             market_spread_home, market_total, edge_spread, edge_total,
             spread_bet_side, total_bet_side,
             kelly_fraction_spread, kelly_fraction_total,
             win_prob_spread, win_prob_total)
        VALUES
            (:game_date_et, :game_slug, :season, :home_team_abbr, :away_team_abbr,
             :pred_margin_home, :pred_total, :used_residual_model,
             :market_spread_home, :market_total, :edge_spread, :edge_total,
             :spread_bet_side, :total_bet_side,
             :kelly_fraction_spread, :kelly_fraction_total,
             :win_prob_spread, :win_prob_total)
        ON CONFLICT (game_date_et, game_slug) DO UPDATE SET
            predicted_at_utc      = NOW(),
            pred_margin_home      = EXCLUDED.pred_margin_home,
            pred_total            = EXCLUDED.pred_total,
            used_residual_model   = EXCLUDED.used_residual_model,
            market_spread_home    = EXCLUDED.market_spread_home,
            market_total          = EXCLUDED.market_total,
            edge_spread           = EXCLUDED.edge_spread,
            edge_total            = EXCLUDED.edge_total,
            spread_bet_side       = EXCLUDED.spread_bet_side,
            total_bet_side        = EXCLUDED.total_bet_side,
            kelly_fraction_spread = EXCLUDED.kelly_fraction_spread,
            kelly_fraction_total  = EXCLUDED.kelly_fraction_total,
            win_prob_spread       = EXCLUDED.win_prob_spread,
            win_prob_total        = EXCLUDED.win_prob_total
    """)

    rows = []
    for _, r in out.iterrows():
        edge_s = float(r["edge_spread"]) if pd.notna(r.get("edge_spread")) else None
        edge_t = float(r["edge_total"]) if pd.notna(r.get("edge_total")) else None
        spread_bet = None
        total_bet = None
        kf_s = kf_t = wp_s = wp_t = None
        used_blend = bool(r.get("used_market_recon", False))
        _calib = calib or {}
        sigma_s = _calib.get(
            "resid_spread_rmse" if (used_blend and w_spread > 0) else "direct_spread_rmse", 14.0
        )
        sigma_t = _calib.get(
            "resid_total_rmse" if (used_blend and w_total > 0) else "direct_total_rmse", 20.0
        )
        if edge_s is not None:
            spread_bet = "home" if edge_s > 0 else "away"
            kf_s, wp_s = _kelly(abs(edge_s), sigma=sigma_s)
        if edge_t is not None:
            total_bet = "over" if edge_t > 0 else "under"
            kf_t, wp_t = _kelly(abs(edge_t), sigma=sigma_t)
        rows.append({
            "game_date_et": et_day,
            "game_slug": r["game_slug"],
            "season": r["season"],
            "home_team_abbr": r["home_team_abbr"],
            "away_team_abbr": r["away_team_abbr"],
            "pred_margin_home": float(r["pred_margin_home_minus_away"]),
            "pred_total": float(r["pred_total_points"]),
            "used_residual_model": bool(r["used_market_recon"]),
            "market_spread_home": float(r["market_spread_home"]) if pd.notna(r.get("market_spread_home")) else None,
            "market_total": float(r["market_total"]) if pd.notna(r.get("market_total")) else None,
            "edge_spread": edge_s,
            "edge_total": edge_t,
            "spread_bet_side": spread_bet,
            "total_bet_side": total_bet,
            "kelly_fraction_spread": round(kf_s, 4) if kf_s is not None else None,
            "kelly_fraction_total":  round(kf_t, 4) if kf_t is not None else None,
            "win_prob_spread": round(wp_s, 4) if wp_s is not None else None,
            "win_prob_total":  round(wp_t, 4) if wp_t is not None else None,
        })

    if rows:
        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)
        log.info("Saved %d game predictions to bets.game_predictions", len(rows))


def _check_injury_staleness(engine, warn_hours: float = 12.0) -> None:
    """Print a warning if injury data has not been refreshed within warn_hours."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT MAX(source_fetched_at_utc) FROM raw.nba_injuries")
            ).fetchone()
        if row is None or row[0] is None:
            print("\n  *** WARNING: No injury data found in DB. Run crawler first. ***")
            return
        last_fetch = row[0]
        if last_fetch.tzinfo is None:
            last_fetch = last_fetch.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last_fetch).total_seconds() / 3600
        if age_hours > warn_hours:
            print(
                f"\n  *** STALE INJURY DATA: last updated {age_hours:.1f}h ago "
                f"(>{warn_hours:.0f}h threshold)."
                f"\n      Run: python -m nba_pipeline.crawler to refresh. ***\n"
            )
        else:
            log.info("Injury data age: %.1fh (fresh)", age_hours)
    except Exception as exc:
        log.warning("Could not check injury data staleness: %s", exc)


def _kelly(
    edge_pts: float,
    juice: int = -110,
    shrink: float = 0.35,
    sigma: float = 14.0,
) -> tuple[float, float]:
    """
    Estimate full Kelly fraction and win probability for a spread/total bet.

    edge_pts : |pred - market_line| (positive, regardless of direction)
    juice    : standard American odds (-110 = bet $110 to win $100)
    shrink   : how far we pull win-prob toward 50% to account for model uncertainty.
               0.35 means we use 35% of the logistic signal above coin-flip.
    sigma    : calibrated RMSE from walk-forward CV (saved in models/calibration.json).
               Controls how fast p grows with edge. Default 14.0 = direct spread RMSE.
               Old default was 7.0 which overstated win-prob by ~2×.

    Returns (full_kelly_fraction, estimated_win_prob).
    Caller should apply fractional Kelly (typically 1/4) for safety.
    """
    b = 100 / abs(juice)
    p_raw = 1 / (1 + math.exp(-edge_pts / sigma))
    p = 0.5 + (p_raw - 0.5) * shrink
    kelly = max(0.0, (b * p - (1 - p)) / b)
    return kelly, p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = PredictConfig()
    et_day = cfg.et_date or datetime.now(_ET).date()
    log.info("Predicting for ET date=%s season=%s", et_day, cfg.season)

    models, feature_cols, feature_medians = _load_models(cfg)
    calib = _load_calibration(cfg.model_dir)

    engine = create_engine(cfg.pg_dsn)
    _check_injury_staleness(engine)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(SQL_GAMES_FOR_DATE),
            conn,
            params={"game_date": et_day, "season": cfg.season},
        )
        if df.empty:
            log.warning("No games found for %s in features.game_prediction_features", et_day)
            return

        id_df, X = _prep_features(df, feature_cols=feature_cols, feature_medians=feature_medians)

        # Predictions:
        # - If residual models exist AND market lines exist, blend:
        #     pred = (1 - w) * direct  +  w * (market_line + resid_pred)
        #   where w = _RESID_BLEND_WEIGHT (0.70 by default)
        # - Else use direct model only
        spread_direct = models["spread_direct"]
        total_direct = models["total_direct"]

        has_resid_models = ("spread_resid" in models) and ("total_resid" in models)

        pred_margin_direct = spread_direct.predict(X)
        pred_total_direct = total_direct.predict(X)

        # default output = direct
        pred_margin_final = pred_margin_direct
        pred_total_final = pred_total_direct
        used_market = np.zeros(len(df), dtype=bool)

        # Compute blend weights independently for spread and total
        w_spread = _compute_blend_weight_spread(calib)
        w_total = _compute_blend_weight_total(calib)

        if has_resid_models and ("market_spread_home" in df.columns) and ("market_total" in df.columns):
            mkt_spread = pd.to_numeric(df["market_spread_home"], errors="coerce")
            mkt_total = pd.to_numeric(df["market_total"], errors="coerce")
            ok = mkt_spread.notna() & mkt_total.notna()

            if ok.any():
                spread_resid = models["spread_resid"]
                total_resid = models["total_resid"]

                pred_spread_resid = spread_resid.predict(X.loc[ok])
                pred_total_resid = total_resid.predict(X.loc[ok])

                # Residual reconstruction: anchor to market line, shift by model residual
                resid_recon_spread = mkt_spread.loc[ok].astype(float).values + pred_spread_resid
                resid_recon_total = mkt_total.loc[ok].astype(float).values + pred_total_resid

                # Blend separately: spread and total can have different quality gates.
                # w=0 → pure direct, w=1 → pure residual
                pred_margin_final = pred_margin_final.copy()
                pred_total_final = pred_total_final.copy()

                pred_margin_final[ok.values] = (
                    (1.0 - w_spread) * pred_margin_direct[ok.values] + w_spread * resid_recon_spread
                )
                pred_total_final[ok.values] = (
                    (1.0 - w_total) * pred_total_direct[ok.values] + w_total * resid_recon_total
                )

                log.info(
                    "Blended predictions for %d/%d games (spread: %.0f%% resid | total: %.0f%% resid)",
                    ok.sum(), len(df), w_spread * 100, w_total * 100,
                )
                used_market = ok.values

        # Bias correction:
        #  Spread — use per-team rolling bias where ≥5 games exist (60-day window),
        #           global rolling bias as fallback for teams with less data.
        #  Total  — global rolling bias only (symmetric — no per-team advantage).
        team_biases = _compute_per_team_bias(engine)
        bias_spread_global, bias_total = _compute_live_bias(engine)
        spread_bias_vec = np.array(
            [team_biases.get(str(h), bias_spread_global) for h in id_df["home_team_abbr"]],
            dtype=float,
        )
        pred_margin_final = pred_margin_final - spread_bias_vec
        pred_total_final = pred_total_final - bias_total

        # Validate raw predictions before clipping
        _validate_game_predictions(pred_margin_final, pred_total_final, id_df)

        # Clip predictions to reasonable NBA ranges
        pred_margin_final = np.clip(pred_margin_final, -40.0, 40.0)
        pred_total_final = np.clip(pred_total_final, 170.0, 280.0)

        out = id_df.copy()
        out["pred_margin_home_minus_away"] = np.round(pred_margin_final, 2)
        out["pred_total_points"] = np.round(pred_total_final, 2)
        out["used_market_recon"] = used_market

        out["pred_spread_label"] = [
            _fmt_spread_from_margin(m, h, a)
            for m, h, a in zip(out["pred_margin_home_minus_away"], out["home_team_abbr"], out["away_team_abbr"])
        ]

        # Compute edges vs market lines
        if "market_spread_home" in df.columns:
            out["market_spread_home"] = pd.to_numeric(df["market_spread_home"].values, errors="coerce")
            out["market_total"] = pd.to_numeric(df["market_total"].values, errors="coerce")
            out["edge_spread"] = np.where(
                out["market_spread_home"].notna(),
                out["pred_margin_home_minus_away"] - out["market_spread_home"],
                np.nan,
            )
            out["edge_total"] = np.where(
                out["market_total"].notna(),
                out["pred_total_points"] - out["market_total"],
                np.nan,
            )
        else:
            out["market_spread_home"] = np.nan
            out["market_total"] = np.nan
            out["edge_spread"] = np.nan
            out["edge_total"] = np.nan

        # Load today's line movement for steam detection (line already moved in model's direction).
        steam_games: set[str] = set()
        try:
            steam_sql = text("""
                SELECT home_team_abbr, away_team_abbr, line_move_margin, line_move_total
                FROM odds.nba_game_lines_open_close
                WHERE as_of_date = :d
            """)
            with engine.connect() as _conn:
                steam_rows = _conn.execute(steam_sql, {"d": et_day}).fetchall()
            steam_map = {(r.home_team_abbr, r.away_team_abbr): r for r in steam_rows}
            # Flag game if |line_move_margin| >= 1.0 and direction aligns with model prediction.
            for _, r in out.iterrows():
                sm = steam_map.get((r["home_team_abbr"], r["away_team_abbr"]))
                if sm is None:
                    continue
                lm = sm.line_move_margin  # positive = line moved toward home (home more favored)
                edge_s = r.get("edge_spread")
                if lm is not None and edge_s is not None and pd.notna(edge_s) and abs(float(lm)) >= 1.0:
                    # Model edge > 0 = model likes home. Steam = line moved toward home (lm > 0).
                    if float(edge_s) * float(lm) > 0:
                        steam_games.add(r["game_slug"])
        except Exception as _exc:
            log.debug("Could not load line movement for steam detection: %s", _exc)

        # Print
        discord = os.getenv("DISCORD_FORMAT") == "1"

        # Count bets that exceed the edge threshold
        n_spread_bets = int(out["edge_spread"].abs().ge(cfg.min_edge_spread).sum()) if "edge_spread" in out else 0
        n_total_bets  = int(out["edge_total"].abs().ge(cfg.min_edge_total).sum()) if "edge_total" in out else 0
        n_high_edge = n_spread_bets + n_total_bets
        if out["used_market_recon"].any():
            model_note = (
                f"spread resid {w_spread:.0%} | total resid {w_total:.0%}"
                if (w_spread > 0 or w_total > 0)
                else "direct only (resid quality gate failed)"
            )
        else:
            model_note = "direct only"
        if discord:
            print(f"**{et_day}** — {len(out)} games · {n_high_edge} high-edge bets ({n_spread_bets} spread, {n_total_bets} total) · {model_note}")
        else:
            print(f"{et_day} — {len(out)} games  {n_high_edge} high-edge bets ({n_spread_bets} spread, {n_total_bets} total)  [{model_note}]")

        for _, r in out.iterrows():
            start = pd.to_datetime(r["start_ts_utc"], utc=True).tz_convert(_ET)
            steam_tag = " 🔥STEAM" if r["game_slug"] in steam_games else ""
            if discord:
                print(f"\n**{r['away_team_abbr']} @ {r['home_team_abbr']}** · {start:%I:%M %p ET}{steam_tag}")
            else:
                print(f"\n{r['away_team_abbr']} @ {r['home_team_abbr']}  {start:%I:%M %p ET}{steam_tag}")

            # Choose calibrated sigma based on which model was actually blended
            used_blend = bool(r.get("used_market_recon", False))
            sigma_s = calib.get(
                "resid_spread_rmse" if (used_blend and w_spread > 0) else "direct_spread_rmse", 14.0
            )
            sigma_t = calib.get(
                "resid_total_rmse" if (used_blend and w_total > 0) else "direct_total_rmse", 20.0
            )

            # Spread line
            edge_s = r.get("edge_spread")
            if pd.notna(edge_s) and abs(float(edge_s)) >= cfg.min_edge_spread:
                es = float(edge_s)
                edge_dir = "+" if es > 0 else ""
                bet_side = "HOME" if es > 0 else "AWAY"
                kelly, p_win = _kelly(abs(es), sigma=sigma_s)
                qk_bet = (kelly / 4) * 1000
                if discord:
                    print(f"  {r['pred_spread_label']}  ⚡ EDGE {edge_dir}{es:.1f} [{bet_side}] p={p_win:.0%} ¼K=${qk_bet:.0f}/$1k")
                else:
                    print(f"  {r['pred_spread_label']}  * EDGE {edge_dir}{es:.1f} pts  [bet {bet_side}]")
                    print(f"    Kelly: p={p_win:.1%}  full={kelly:.1%}  1/4 Kelly = ${qk_bet:.0f} per $1,000 bankroll")
            else:
                print(f"  {r['pred_spread_label']}")

            # Total line
            edge_t = r.get("edge_total")
            if pd.notna(edge_t) and abs(float(edge_t)) >= cfg.min_edge_total:
                et_ = float(edge_t)
                edge_dir = "+" if et_ > 0 else ""
                over_under = "Over" if et_ > 0 else "Under"
                kelly, p_win = _kelly(abs(et_), sigma=sigma_t)
                qk_bet = (kelly / 4) * 1000
                if discord:
                    print(f"  {over_under} {r['pred_total_points']:.1f}  ⚡ EDGE {edge_dir}{et_:.1f} [{over_under.upper()}] p={p_win:.0%} ¼K=${qk_bet:.0f}/$1k")
                else:
                    print(f"  {over_under} {r['pred_total_points']:.1f}  * EDGE {edge_dir}{et_:.1f} pts")
                    print(f"    Kelly: p={p_win:.1%}  full={kelly:.1%}  1/4 Kelly = ${qk_bet:.0f} per $1,000 bankroll")
            else:
                print(f"  Pred total: {r['pred_total_points']:.1f}")

        # Save predictions to DB
        try:
            _save_predictions(out, engine, et_day, calib=calib, w_spread=w_spread, w_total=w_total)
        except Exception as exc:
            log.warning("Could not save predictions to DB: %s", exc)


if __name__ == "__main__":
    main()
