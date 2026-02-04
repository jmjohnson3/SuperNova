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
