import json
import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBRegressor

log = logging.getLogger("nba_pipeline.modeling.predict_today")

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = Path(__file__).resolve().parent / "models"
    season: str | None = None  # optional filter, e.g. "2025-2026-regular"
    et_date: date | None = None  # default today ET


SQL_TODAY = """
SELECT *
FROM features.game_prediction_features
WHERE game_date_et = %(game_date)s
  AND start_ts_utc IS NOT NULL
  AND (%(season)s IS NULL OR season = %(season)s)
ORDER BY start_ts_utc, game_slug
"""


def _prep_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (id_df, X)
    id_df: columns used for printing
    X: numeric model matrix (with one-hot for season/home/away)
    """
    id_cols = ["season", "game_slug", "game_date_et", "start_ts_utc", "home_team_abbr", "away_team_abbr"]
    id_df = df[id_cols].copy()

    X = df.drop(columns=[c for c in id_cols if c in df.columns]).copy()

    # Derive start time features (from the id_df copy)
    ts = pd.to_datetime(id_df["start_ts_utc"], errors="coerce", utc=True)
    X["start_hour_utc"] = ts.dt.hour
    X["start_dow_utc"] = ts.dt.dayofweek

    # Ensure b2b numeric
    for bcol in ("home_is_b2b", "away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    # One-hot season/team abbr (match training)
    cat_cols = []
    for c in ("season", "home_team_abbr", "away_team_abbr"):
        if c in df.columns:
            X[c] = df[c]
            cat_cols.append(c)

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # Coerce remaining object/string columns to numeric where possible
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # IMPORTANT: do NOT fill with batch medians here.
    # We'll fill using training medians AFTER we align to feature_cols.
    return id_df, X


def _load_models(cfg: PredictConfig) -> tuple[XGBRegressor, XGBRegressor, list[str], dict[str, float]]:
    feature_cols_path = cfg.model_dir / "feature_columns.json"
    feature_medians_path = cfg.model_dir / "feature_medians.json"
    spread_path = cfg.model_dir / "spread_xgb.json"
    total_path = cfg.model_dir / "total_xgb.json"

    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing {feature_cols_path}. Run training with feature_columns save enabled.")
    if not spread_path.exists() or not total_path.exists():
        raise FileNotFoundError(f"Missing model files in {cfg.model_dir}. Run training and save models first.")

    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    # Medians are strongly recommended; if missing, we'll fall back to 0-fills
    feature_medians: dict[str, float] = {}
    if feature_medians_path.exists():
        raw = json.loads(feature_medians_path.read_text(encoding="utf-8"))
        # ensure floats
        feature_medians = {str(k): float(v) for k, v in raw.items() if v is not None}
    else:
        log.warning("Missing %s. Will fill NaNs with 0.0 only (may hurt accuracy).", feature_medians_path)

    spread_model = XGBRegressor()
    spread_model.load_model(str(spread_path))

    total_model = XGBRegressor()
    total_model.load_model(str(total_path))

    return spread_model, total_model, feature_cols, feature_medians


def _align_and_fill(
    X: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_medians: dict[str, float],
) -> pd.DataFrame:
    """
    Align columns to training schema, fill NaNs using training medians, then fill remaining with 0.
    """
    # Add any missing columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan  # keep as NaN so median fill can apply

    # Drop any extra columns not seen in training
    extra_cols = [c for c in X.columns if c not in feature_cols]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    # Reorder
    X = X[feature_cols]

    # Fill using training medians (only where provided)
    if feature_medians:
        for c, m in feature_medians.items():
            if c in X.columns:
                X[c] = X[c].fillna(m)

    # Final safety
    X = X.fillna(0.0)
    return X


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = PredictConfig()
    et_today = cfg.et_date or datetime.now(_ET).date()

    log.info("Predicting for ET date=%s season=%s", et_today, cfg.season)

    spread_model, total_model, feature_cols, feature_medians = _load_models(cfg)

    with psycopg2.connect(cfg.pg_dsn) as conn:
        df = pd.read_sql(SQL_TODAY, conn, params={"game_date": et_today, "season": cfg.season})
        if df.empty:
            log.warning("No games found for %s in features.game_prediction_features", et_today)
            return

        id_df, X = _prep_features(df)

        # Align + fill exactly like training
        X = _align_and_fill(X, feature_cols=feature_cols, feature_medians=feature_medians)

        spread_pred = spread_model.predict(X)  # predicted margin = home - away
        total_pred = total_model.predict(X)

        out = id_df.copy()
        out["pred_margin_home_minus_away"] = np.round(spread_pred, 2)
        out["pred_total_points"] = np.round(total_pred, 2)

        def fmt_spread(m: float, home: str, away: str) -> str:
            if m >= 0:
                return f"{home} -{abs(m):.1f}"
            return f"{away} -{abs(m):.1f}"

        out["pred_spread_label"] = [
            fmt_spread(m, h, a)
            for m, h, a in zip(out["pred_margin_home_minus_away"], out["home_team_abbr"], out["away_team_abbr"])
        ]

        for _, r in out.iterrows():
            start = pd.to_datetime(r["start_ts_utc"], utc=True).tz_convert(_ET)
            print(
                f"{start:%Y-%m-%d %I:%M%p ET} | {r['away_team_abbr']} @ {r['home_team_abbr']} | "
                f"Spread: {r['pred_spread_label']} | Total: {r['pred_total_points']:.1f}"
            )


if __name__ == "__main__":
    main()
