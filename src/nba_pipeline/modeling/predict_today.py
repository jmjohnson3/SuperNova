import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBRegressor
from sqlalchemy import create_engine, text

log = logging.getLogger("nba_pipeline.modeling.predict_today")

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = Path(__file__).resolve().parent / "models"
    season: str | None = None
    et_date: date | None = None


SQL_GAMES_FOR_DATE = """
SELECT *
FROM features.game_prediction_features
WHERE game_date_et = :game_date
  AND start_ts_utc IS NOT NULL
  AND (:season IS NULL OR season = :season)
ORDER BY start_ts_utc, game_slug
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

    # --- Derived interaction features (must match training) ---
    if "home_rest_days" in X.columns and "away_rest_days" in X.columns:
        X["rest_advantage_home"] = X["home_rest_days"] - X["away_rest_days"]

    if "home_pts_for_avg_10" in X.columns and "home_pts_against_avg_10" in X.columns:
        X["home_net_rating_10"] = X["home_pts_for_avg_10"] - X["home_pts_against_avg_10"]
    if "away_pts_for_avg_10" in X.columns and "away_pts_against_avg_10" in X.columns:
        X["away_net_rating_10"] = X["away_pts_for_avg_10"] - X["away_pts_against_avg_10"]
    if "home_net_rating_10" in X.columns and "away_net_rating_10" in X.columns:
        X["net_rating_diff_10"] = X["home_net_rating_10"] - X["away_net_rating_10"]

    if "home_pace_avg_5" in X.columns and "away_pace_avg_5" in X.columns:
        X["pace_diff_5"] = X["home_pace_avg_5"] - X["away_pace_avg_5"]

    if "home_pts_for_avg_5" in X.columns and "away_pts_for_avg_5" in X.columns:
        X["pts_for_diff_5"] = X["home_pts_for_avg_5"] - X["away_pts_for_avg_5"]

    # --- NEW: Efficiency differentials ---
    if "home_efg_pct_avg_5" in X.columns and "away_efg_pct_avg_5" in X.columns:
        X["efg_diff_5"] = X["home_efg_pct_avg_5"] - X["away_efg_pct_avg_5"]
    if "home_ts_pct_avg_5" in X.columns and "away_ts_pct_avg_5" in X.columns:
        X["ts_diff_5"] = X["home_ts_pct_avg_5"] - X["away_ts_pct_avg_5"]
    if "home_fg3a_rate_avg_5" in X.columns and "away_fg3a_rate_avg_5" in X.columns:
        X["fg3a_rate_diff_5"] = X["home_fg3a_rate_avg_5"] - X["away_fg3a_rate_avg_5"]
    if "home_tov_rate_avg_5" in X.columns and "away_tov_rate_avg_5" in X.columns:
        X["tov_rate_diff_5"] = X["home_tov_rate_avg_5"] - X["away_tov_rate_avg_5"]

    # --- NEW: Injury impact differential ---
    if "home_injured_pts_lost" in X.columns and "away_injured_pts_lost" in X.columns:
        X["injury_pts_diff"] = X["away_injured_pts_lost"] - X["home_injured_pts_lost"]

    # --- NEW: Clutch differential ---
    if "home_clutch_net_avg_10" in X.columns and "away_clutch_net_avg_10" in X.columns:
        X["clutch_net_diff_10"] = X["home_clutch_net_avg_10"] - X["away_clutch_net_avg_10"]

    # --- V005: Odds juice derived ---
    if "spread_home_implied_prob" in X.columns:
        X["spread_implied_edge"] = X.get("spread_home_implied_prob", 0.5) - 0.5

    # --- V006: Team style differentials ---
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

    # --- V008: Lineup stability differential ---
    if "home_starter_continuity_avg_10" in X.columns and "away_starter_continuity_avg_10" in X.columns:
        X["continuity_diff_10"] = X["home_starter_continuity_avg_10"] - X["away_starter_continuity_avg_10"]

    # --- V009: Standings differentials ---
    if "home_streak" in X.columns and "away_streak" in X.columns:
        X["streak_diff"] = X["home_streak"] - X["away_streak"]
    if "home_last10_pct" in X.columns and "away_last10_pct" in X.columns:
        X["last10_pct_diff"] = X["home_last10_pct"] - X["away_last10_pct"]
    if "home_home_record_pct" in X.columns and "away_away_record_pct" in X.columns:
        X["venue_record_diff"] = X["home_home_record_pct"] - X["away_away_record_pct"]

    # --- V011: PBP differentials ---
    if "home_three_pt_rate_avg_10" in X.columns and "away_three_pt_rate_avg_10" in X.columns:
        X["three_pt_rate_diff_10"] = X["home_three_pt_rate_avg_10"] - X["away_three_pt_rate_avg_10"]

    # --- B2B × rest interaction ---
    if "home_is_b2b" in X.columns and "away_is_b2b" in X.columns:
        X["b2b_net_disadvantage"] = X["home_is_b2b"] - X["away_is_b2b"]

    # --- Referee foul over/under bias signal ---
    if (
        "crew_avg_fouls_per_game" in X.columns
        and "home_fouls_avg_10" in X.columns
        and "away_fouls_avg_10" in X.columns
    ):
        X["ref_foul_ot_signal"] = (
            X["crew_avg_fouls_per_game"] - (X["home_fouls_avg_10"] + X["away_fouls_avg_10"])
        )

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


def _fmt_spread_from_margin(margin_home_minus_away: float, home: str, away: str) -> str:
    if margin_home_minus_away >= 0:
        return f"{home} -{abs(margin_home_minus_away):.1f}"
    return f"{away} -{abs(margin_home_minus_away):.1f}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = PredictConfig()
    et_day = cfg.et_date or datetime.now(_ET).date()
    log.info("Predicting for ET date=%s season=%s", et_day, cfg.season)

    models, feature_cols, feature_medians = _load_models(cfg)

    engine = create_engine(cfg.pg_dsn)

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
        # - If residual models exist AND market lines exist, reconstruct:
        #     pred_margin = market_spread_home + pred_resid
        #     pred_total  = market_total + pred_resid
        # - Else fallback to direct models
        spread_direct = models["spread_direct"]
        total_direct = models["total_direct"]

        has_resid_models = ("spread_resid" in models) and ("total_resid" in models)

        pred_margin_direct = spread_direct.predict(X)
        pred_total_direct = total_direct.predict(X)

        # default output = direct
        pred_margin_final = pred_margin_direct
        pred_total_final = pred_total_direct
        used_market = np.zeros(len(df), dtype=bool)

        if has_resid_models and ("market_spread_home" in df.columns) and ("market_total" in df.columns):
            mkt_spread = pd.to_numeric(df["market_spread_home"], errors="coerce")
            mkt_total = pd.to_numeric(df["market_total"], errors="coerce")
            ok = mkt_spread.notna() & mkt_total.notna()

            if ok.any():
                spread_resid = models["spread_resid"]
                total_resid = models["total_resid"]

                pred_spread_resid = spread_resid.predict(X.loc[ok])
                pred_total_resid = total_resid.predict(X.loc[ok])

                pred_margin_final = pred_margin_final.copy()
                pred_total_final = pred_total_final.copy()

                pred_margin_final[ok.values] = mkt_spread.loc[ok].astype(float).values + pred_spread_resid
                pred_total_final[ok.values] = mkt_total.loc[ok].astype(float).values + pred_total_resid

                used_market = ok.values

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

        # Print
        for _, r in out.iterrows():
            start = pd.to_datetime(r["start_ts_utc"], utc=True).tz_convert(_ET)
            tag = "mkt+resid" if bool(r["used_market_recon"]) else "direct"
            print(
                f"{start:%Y-%m-%d %I:%M%p ET} | {r['away_team_abbr']} @ {r['home_team_abbr']} | "
                f"Spread: {r['pred_spread_label']} | Total: {r['pred_total_points']:.1f} | {tag}"
            )


if __name__ == "__main__":
    main()
