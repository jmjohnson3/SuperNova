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
    min_edge_pts: float = 3.0    # minimum |pred - market| to flag as high-confidence


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
        id                  SERIAL PRIMARY KEY,
        predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        game_date_et        DATE        NOT NULL,
        game_slug           TEXT        NOT NULL,
        season              TEXT        NOT NULL,
        home_team_abbr      TEXT        NOT NULL,
        away_team_abbr      TEXT        NOT NULL,
        pred_margin_home    NUMERIC,
        pred_total          NUMERIC,
        used_residual_model BOOLEAN     DEFAULT FALSE,
        market_spread_home  NUMERIC,
        market_total        NUMERIC,
        edge_spread         NUMERIC,
        edge_total          NUMERIC,
        actual_margin_home  NUMERIC,
        actual_total        NUMERIC,
        spread_bet_side     TEXT,
        total_bet_side      TEXT,
        spread_covered      BOOLEAN,
        total_correct       BOOLEAN,
        UNIQUE (game_date_et, game_slug)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _save_predictions(out: pd.DataFrame, engine, et_day) -> None:
    """Upsert game predictions into bets.game_predictions."""
    _ensure_bets_schema(engine)
    upsert_sql = text("""
        INSERT INTO bets.game_predictions
            (game_date_et, game_slug, season, home_team_abbr, away_team_abbr,
             pred_margin_home, pred_total, used_residual_model,
             market_spread_home, market_total, edge_spread, edge_total,
             spread_bet_side, total_bet_side)
        VALUES
            (:game_date_et, :game_slug, :season, :home_team_abbr, :away_team_abbr,
             :pred_margin_home, :pred_total, :used_residual_model,
             :market_spread_home, :market_total, :edge_spread, :edge_total,
             :spread_bet_side, :total_bet_side)
        ON CONFLICT (game_date_et, game_slug) DO UPDATE SET
            predicted_at_utc    = NOW(),
            pred_margin_home    = EXCLUDED.pred_margin_home,
            pred_total          = EXCLUDED.pred_total,
            used_residual_model = EXCLUDED.used_residual_model,
            market_spread_home  = EXCLUDED.market_spread_home,
            market_total        = EXCLUDED.market_total,
            edge_spread         = EXCLUDED.edge_spread,
            edge_total          = EXCLUDED.edge_total,
            spread_bet_side     = EXCLUDED.spread_bet_side,
            total_bet_side      = EXCLUDED.total_bet_side
    """)

    rows = []
    for _, r in out.iterrows():
        edge_s = float(r["edge_spread"]) if pd.notna(r.get("edge_spread")) else None
        edge_t = float(r["edge_total"]) if pd.notna(r.get("edge_total")) else None
        spread_bet = None
        total_bet = None
        if edge_s is not None:
            spread_bet = "home" if edge_s > 0 else "away"
        if edge_t is not None:
            total_bet = "over" if edge_t > 0 else "under"
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


def _kelly(edge_pts: float, juice: int = -110, shrink: float = 0.35) -> tuple[float, float]:
    """
    Estimate full Kelly fraction and win probability for a spread/total bet.

    edge_pts : |pred_margin - market_spread| (positive, regardless of direction)
    juice    : standard American odds (-110 = bet $110 to win $100)
    shrink   : how far we pull win-prob toward 50% to account for model uncertainty.
               0.35 means we use 35% of the logistic signal above coin-flip.
               At shrink=0.35: edge=3 → p≈0.52, edge=7 → p≈0.55, edge=14 → p≈0.60

    Returns (full_kelly_fraction, estimated_win_prob).
    Caller should apply fractional Kelly (typically 1/4) for safety.
    """
    b = 100 / abs(juice)           # net odds: win $b per $1 risked
    sigma = 7.0                    # ~model RMSE; controls how fast p grows with edge
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

        # Print
        discord = os.getenv("DISCORD_FORMAT") == "1"
        for _, r in out.iterrows():
            start = pd.to_datetime(r["start_ts_utc"], utc=True).tz_convert(_ET)
            if discord:
                print(f"\n**{r['away_team_abbr']} @ {r['home_team_abbr']}** · {start:%I:%M %p ET}")
            else:
                print(f"\n{r['away_team_abbr']} @ {r['home_team_abbr']}  {start:%I:%M %p ET}")

            # Spread line
            edge_s = r.get("edge_spread")
            if pd.notna(edge_s) and abs(float(edge_s)) >= cfg.min_edge_pts:
                es = float(edge_s)
                edge_dir = "+" if es > 0 else ""
                bet_side = "HOME" if es > 0 else "AWAY"
                kelly, p_win = _kelly(abs(es))
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
            if pd.notna(edge_t) and abs(float(edge_t)) >= cfg.min_edge_pts:
                et_ = float(edge_t)
                edge_dir = "+" if et_ > 0 else ""
                over_under = "Over" if et_ > 0 else "Under"
                kelly, p_win = _kelly(abs(et_))
                qk_bet = (kelly / 4) * 1000
                if discord:
                    print(f"  {over_under} {r['pred_total_points']:.1f}  ⚡ EDGE {edge_dir}{et_:.1f} [{over_under.upper()}] p={p_win:.0%} ¼K=${qk_bet:.0f}/$1k")
                else:
                    print(f"  {over_under} {r['pred_total_points']:.1f}  * EDGE {edge_dir}{et_:.1f} pts")
                    print(f"    Kelly: p={p_win:.1%}  full={kelly:.1%}  1/4 Kelly = ${qk_bet:.0f} per $1,000 bankroll")
            else:
                print(f"  Over {r['pred_total_points']:.1f}")

        # Save predictions to DB
        try:
            _save_predictions(out, engine, et_day)
        except Exception as exc:
            log.warning("Could not save predictions to DB: %s", exc)


if __name__ == "__main__":
    main()
