import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor

log = logging.getLogger("nba_pipeline.modeling.predict_player_props")

_ET = ZoneInfo("America/New_York")

@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = Path(__file__).resolve().parent / "models" / "player_props"
    et_date: date | None = None
    min_proj_minutes: float = 18.0          # filter threshold
    starter_minutes_threshold: float = 28.0 # flag threshold
    starter_min_prev_games: int = 5         # sample size requirement

SQL_TODAYS_GAMES = """
SELECT *
FROM features.game_prediction_features
WHERE game_date_et = :game_date
  AND start_ts_utc IS NOT NULL
ORDER BY start_ts_utc, game_slug
"""

# Build "as-of today" player features using all prior games.
# Strategy:
#  - get today's teams/opponents from features.game_prediction_features
#  - for those teams, compute rolling stats for each player over previous games
#  - choose each player's latest row prior to today (acts as a pregame snapshot)
SQL_PLAYER_SNAPSHOTS_FOR_DATE = """
WITH games_today AS (
    SELECT
      season,
      game_slug,
      game_date_et,
      start_ts_utc,
      UPPER(home_team_abbr) AS home_team_abbr,
      UPPER(away_team_abbr) AS away_team_abbr,
      market_total,
      market_spread_home,
      game_pace_est_5,
      home_rest_days,
      home_is_b2b,
      away_rest_days,
      away_is_b2b
    FROM features.game_prediction_features
    WHERE game_date_et = :game_date
      AND start_ts_utc IS NOT NULL
),
teams_today AS (
    SELECT season, UPPER(home_team_abbr) AS team_abbr, UPPER(away_team_abbr) AS opponent_abbr, TRUE AS is_home, game_slug
    FROM games_today
    UNION ALL
    SELECT season, UPPER(away_team_abbr) AS team_abbr, UPPER(home_team_abbr) AS opponent_abbr, FALSE AS is_home, game_slug
    FROM games_today
),
hist AS (
    SELECT
      p.season,
      p.game_slug,
      g.start_ts_utc,
      p.player_id,
      COALESCE(pl.player_name, 'player_id=' || p.player_id::text) AS player_name,
      UPPER(p.team_abbr) AS team_abbr,
      UPPER(p.opponent_abbr) AS opponent_abbr,
      p.is_home,
      p.status,
      p.minutes,
      p.points,
      p.rebounds,
      p.assists,
      p.threes_made,
      p.fga,
      p.fta
    FROM raw.nba_player_gamelogs p
    JOIN raw.nba_games g
      ON g.season = p.season
     AND g.game_slug = p.game_slug
    LEFT JOIN raw.nba_players pl
      ON pl.player_id = p.player_id
    WHERE g.start_ts_utc < (SELECT MIN(start_ts_utc) FROM games_today)
      AND p.minutes IS NOT NULL
      AND p.minutes > 0
),
feat AS (
    SELECT
      h.*,
      LAG(h.start_ts_utc) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc) AS prev_start_ts_utc,

      COUNT(*) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS n_games_prev_10,

      AVG(h.minutes)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS min_avg_5,
      AVG(h.minutes)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS min_avg_10,

      AVG(h.points)   OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS pts_avg_5,
      AVG(h.points)   OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS pts_avg_10,

      AVG(h.rebounds) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS reb_avg_5,
      AVG(h.rebounds) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS reb_avg_10,

      AVG(h.assists)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS ast_avg_5,
      AVG(h.assists)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS ast_avg_10,

      AVG(h.fga)      OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS fga_avg_10,
      AVG(h.fta)      OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS fta_avg_10,
      AVG(h.threes_made) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS threes_avg_10,

      STDDEV_SAMP(h.points)   OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS pts_sd_10,
      STDDEV_SAMP(h.rebounds) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS reb_sd_10,
      STDDEV_SAMP(h.assists)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS ast_sd_10
    FROM hist h
),
latest_per_player_team AS (
    SELECT DISTINCT ON (player_id, team_abbr)
      season,
      player_id,
      player_name,
      team_abbr,
      start_ts_utc,
      n_games_prev_10,
      EXTRACT(EPOCH FROM (start_ts_utc - prev_start_ts_utc)) / 86400.0 AS player_rest_days,
      min_avg_5, min_avg_10,
      pts_avg_5, pts_avg_10,
      reb_avg_5, reb_avg_10,
      ast_avg_5, ast_avg_10,
      fga_avg_10, fta_avg_10, threes_avg_10,
      pts_sd_10, reb_sd_10, ast_sd_10
    FROM feat
    ORDER BY player_id, team_abbr, start_ts_utc DESC
),
joined AS (
    SELECT
      t.season,
      t.game_slug,
      gt.game_date_et,
      gt.start_ts_utc,
      t.team_abbr,
      t.opponent_abbr,
      t.is_home,
      lp.player_id,
      lp.player_name,
      COALESCE(lp.min_avg_5, lp.min_avg_10) AS proj_minutes,
      CASE
        WHEN COALESCE(lp.min_avg_5, lp.min_avg_10) >= :starter_min_threshold
         AND lp.n_games_prev_10 >= :starter_min_prev_games
        THEN TRUE ELSE FALSE
      END AS is_proj_starter,
      lp.n_games_prev_10,
      lp.player_rest_days,
      lp.min_avg_5, lp.min_avg_10,
      lp.pts_avg_5, lp.pts_avg_10,
      lp.reb_avg_5, lp.reb_avg_10,
      lp.ast_avg_5, lp.ast_avg_10,
      lp.fga_avg_10, lp.fta_avg_10, lp.threes_avg_10,
      lp.pts_sd_10, lp.reb_sd_10, lp.ast_sd_10,

      gt.game_pace_est_5,
      gt.market_total,
      gt.market_spread_home,
      gt.home_rest_days,
      gt.home_is_b2b,
      gt.away_rest_days,
      gt.away_is_b2b,

      CASE
        WHEN gt.market_total IS NULL OR gt.market_spread_home IS NULL THEN NULL::numeric
        WHEN t.team_abbr = gt.home_team_abbr THEN (gt.market_total / 2.0) - (gt.market_spread_home / 2.0)
        WHEN t.team_abbr = gt.away_team_abbr THEN (gt.market_total / 2.0) + (gt.market_spread_home / 2.0)
        ELSE NULL::numeric
      END AS team_implied_total
    FROM teams_today t
    JOIN games_today gt
      ON gt.season = t.season
     AND gt.game_slug = t.game_slug
    JOIN latest_per_player_team lp
      ON lp.season = t.season
     AND lp.team_abbr = t.team_abbr
)
SELECT *
FROM joined
WHERE n_games_prev_10 >= 3
  AND (min_avg_10 IS NULL OR min_avg_10 >= 10.0)
  AND COALESCE(min_avg_5, min_avg_10) >= :min_proj_minutes
ORDER BY start_ts_utc, game_slug, team_abbr, player_id
"""

def _coerce_numeric_cols(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def _load_artifacts(cfg: PredictConfig) -> tuple[dict[str, XGBRegressor], list[str], dict[str, float]]:
    model_dir = cfg.model_dir
    feat_path = model_dir / "feature_columns.json"
    med_path = model_dir / "feature_medians.json"

    if not feat_path.exists() or not med_path.exists():
        raise FileNotFoundError(f"Missing artifacts in {model_dir}. Run train_player_prop_models first.")

    feature_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    medians = json.loads(med_path.read_text(encoding="utf-8"))

    def load_model(p: Path) -> XGBRegressor:
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        m = XGBRegressor()
        m.load_model(str(p))
        return m

    models = {
        "points": load_model(model_dir / "points_xgb.json"),
        "rebounds": load_model(model_dir / "rebounds_xgb.json"),
        "assists": load_model(model_dir / "assists_xgb.json"),
    }
    return models, feature_cols, medians

def _add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived interaction features (must match training)."""
    if "pts_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["pts_per_min_10"] = X["pts_avg_10"] / X["min_avg_10"].clip(lower=1.0)
    if "reb_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["reb_per_min_10"] = X["reb_avg_10"] / X["min_avg_10"].clip(lower=1.0)
    if "ast_avg_10" in X.columns and "min_avg_10" in X.columns:
        X["ast_per_min_10"] = X["ast_avg_10"] / X["min_avg_10"].clip(lower=1.0)

    if "pts_avg_5" in X.columns and "pts_avg_10" in X.columns:
        X["pts_trend_5v10"] = X["pts_avg_5"] - X["pts_avg_10"]
    if "reb_avg_5" in X.columns and "reb_avg_10" in X.columns:
        X["reb_trend_5v10"] = X["reb_avg_5"] - X["reb_avg_10"]
    if "ast_avg_5" in X.columns and "ast_avg_10" in X.columns:
        X["ast_trend_5v10"] = X["ast_avg_5"] - X["ast_avg_10"]

    if "min_avg_5" in X.columns and "min_avg_10" in X.columns:
        X["min_trend_5v10"] = X["min_avg_5"] - X["min_avg_10"]

    if "pts_sd_10" in X.columns and "pts_avg_10" in X.columns:
        X["pts_cv_10"] = X["pts_sd_10"] / X["pts_avg_10"].clip(lower=0.5)
    if "reb_sd_10" in X.columns and "reb_avg_10" in X.columns:
        X["reb_cv_10"] = X["reb_sd_10"] / X["reb_avg_10"].clip(lower=0.5)
    if "ast_sd_10" in X.columns and "ast_avg_10" in X.columns:
        X["ast_cv_10"] = X["ast_sd_10"] / X["ast_avg_10"].clip(lower=0.5)

    return X


def _prep_X(df: pd.DataFrame, feature_cols: list[str], medians: dict[str, float]) -> pd.DataFrame:
    id_cols = {
        "season","game_slug","game_date_et","start_ts_utc",
        "team_abbr","opponent_abbr","is_home","player_id",
        "player_name",
    }
    X = df.drop(columns=[c for c in id_cols if c in df.columns]).copy()

    # one-hot for team/opponent/season
    cat_cols = [c for c in ("season","team_abbr","opponent_abbr") if c in df.columns]
    for c in cat_cols:
        X[c] = df[c]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dummy_na=False)

    # bools -> int
    for bcol in ("is_home","home_is_b2b","away_is_b2b"):
        if bcol in X.columns:
            X[bcol] = X[bcol].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric_cols(X)

    # Derived interaction features (must match training)
    X = _add_derived_features(X)

    # align schema
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        X = X.drop(columns=extra)
    X = X[feature_cols]

    # fill
    for c, m in medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    return X.fillna(0.0)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = PredictConfig()
    et_day = cfg.et_date or datetime.now(_ET).date()
    log.info("Predicting player props for ET date=%s", et_day)

    models, feature_cols, medians = _load_artifacts(cfg)
    engine = create_engine(cfg.pg_dsn)

    with engine.connect() as conn:
        games = pd.read_sql(text(SQL_TODAYS_GAMES), conn, params={"game_date": et_day})
        if games.empty:
            log.warning("No games found for %s", et_day)
            return

        df = pd.read_sql(
            text(SQL_PLAYER_SNAPSHOTS_FOR_DATE),
            conn,
            params={
                "game_date": et_day,
                "min_proj_minutes": cfg.min_proj_minutes,
                "starter_min_threshold": cfg.starter_minutes_threshold,
                "starter_min_prev_games": cfg.starter_min_prev_games,
            },
        )

        if df.empty:
            log.warning("No player snapshots found for %s (need history in raw.nba_player_gamelogs)", et_day)
            return

    X = _prep_X(df, feature_cols, medians)

    df_out = df[
        ["start_ts_utc", "game_slug", "team_abbr", "opponent_abbr", "is_home", "player_id", "player_name",
         "proj_minutes", "is_proj_starter"]
    ].copy()

    df_out["pred_points"] = np.clip(models["points"].predict(X), 0.0, 60.0)
    df_out["pred_rebounds"] = np.clip(models["rebounds"].predict(X), 0.0, 25.0)
    df_out["pred_assists"] = np.clip(models["assists"].predict(X), 0.0, 20.0)

    # pretty print grouped by game
    df_out["start_ts_utc"] = pd.to_datetime(df_out["start_ts_utc"], utc=True).dt.tz_convert(_ET)
    df_out = df_out.sort_values(["start_ts_utc","game_slug","team_abbr","pred_points"], ascending=[True,True,True,False])

    for (ts, slug), g in df_out.groupby(["start_ts_utc", "game_slug"], sort=False):
        print(f"\n{ts:%Y-%m-%d %I:%M%p ET} | {slug}")

        g_sorted = g.sort_values(["team_abbr", "pred_points"], ascending=[True, False])

        for _, r in g_sorted.iterrows():
            ha = "HOME" if bool(r["is_home"]) else "AWAY"
            name = (r.get("player_name") or "").strip() or f"player_id={int(r['player_id'])}"
            star = "★" if bool(r.get("is_proj_starter", False)) else " "

            pm = r.get("proj_minutes")
            pm_txt = f"{pm:.1f}" if pd.notna(pm) else "NA"

            print(
                f"  {r['team_abbr']} vs {r['opponent_abbr']} ({ha}) | {star} {name} | PTS {r['pred_points']:.1f} | REB {r['pred_rebounds']:.1f} | AST {r['pred_assists']:.1f}"
            )



if __name__ == "__main__":
    main()
