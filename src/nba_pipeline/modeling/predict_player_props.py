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
#  - LEFT JOIN V013 referee foul risk and V006 opponent style for full feature parity with training
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
      away_is_b2b,
      home_off_rtg_avg_10,
      home_def_rtg_avg_10,
      away_off_rtg_avg_10,
      away_def_rtg_avg_10,
      home_pace_avg_5 AS home_pace_5,
      away_pace_avg_5 AS away_pace_5,
      home_pace_avg_10 AS home_pace_10,
      away_pace_avg_10 AS away_pace_10
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
      p.fta,
      -- V007: expanded stats from JSONB
      NULLIF(p.stats->'defense'->>'stl', '')::numeric            AS stl,
      NULLIF(p.stats->'defense'->>'blk', '')::numeric            AS blk,
      COALESCE(
          NULLIF(p.stats->'defense'->>'tov', '')::numeric,
          NULLIF(p.stats->'offense'->>'tov', '')::numeric
      )                                                           AS tov,
      NULLIF(p.stats->'rebounds'->>'offReb', '')::numeric        AS off_reb,
      NULLIF(p.stats->'rebounds'->>'defReb', '')::numeric        AS def_reb,
      NULLIF(p.stats->'fieldGoals'->>'fgMade', '')::numeric      AS fg_made,
      NULLIF(p.stats->'miscellaneous'->>'plusMinus', '')::numeric AS plus_minus,
      NULLIF(p.stats->'miscellaneous'->>'foulsTotal', '')::numeric AS fouls
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
      STDDEV_SAMP(h.assists)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS ast_sd_10,

      AVG(
        CASE WHEN h.minutes > 0 THEN (h.fga + 0.44 * h.fta) / h.minutes ELSE NULL END
      ) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS usage_proxy_avg_10,

      -- V007 rolling expanded stats
      AVG(h.stl)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS stl_avg_5,
      AVG(h.stl)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS stl_avg_10,
      AVG(h.blk)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS blk_avg_5,
      AVG(h.blk)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS blk_avg_10,
      AVG(h.tov)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS tov_avg_10,
      AVG(COALESCE(h.stl, 0) + COALESCE(h.blk, 0))
                        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS stl_plus_blk_avg_10,
      AVG(h.off_reb)    OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS off_reb_avg_10,
      AVG(h.def_reb)    OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS def_reb_avg_10,
      AVG(CASE WHEN h.fga > 0 THEN h.fg_made / h.fga ELSE NULL END)
                        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS fg_pct_avg_10,
      AVG(h.plus_minus) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS plus_minus_avg_5,
      AVG(h.plus_minus) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS plus_minus_avg_10,
      AVG(h.fouls)      OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS fouls_avg_5,
      AVG(h.fouls)      OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS fouls_avg_10
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
      pts_sd_10, reb_sd_10, ast_sd_10,
      usage_proxy_avg_10,
      -- V007 expanded stats
      stl_avg_5, stl_avg_10,
      blk_avg_5, blk_avg_10,
      tov_avg_10,
      stl_plus_blk_avg_10,
      off_reb_avg_10, def_reb_avg_10,
      fg_pct_avg_10,
      plus_minus_avg_5, plus_minus_avg_10,
      fouls_avg_5, fouls_avg_10
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
      lp.usage_proxy_avg_10,
      -- V007 expanded player stats
      lp.stl_avg_5, lp.stl_avg_10,
      lp.blk_avg_5, lp.blk_avg_10,
      lp.tov_avg_10,
      lp.stl_plus_blk_avg_10,
      lp.off_reb_avg_10, lp.def_reb_avg_10,
      lp.fg_pct_avg_10,
      lp.plus_minus_avg_5, lp.plus_minus_avg_10,
      lp.fouls_avg_5, lp.fouls_avg_10,

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
      END AS team_implied_total,

      CASE
        WHEN t.team_abbr = gt.home_team_abbr THEN gt.away_def_rtg_avg_10
        WHEN t.team_abbr = gt.away_team_abbr THEN gt.home_def_rtg_avg_10
        ELSE NULL::numeric
      END AS opp_def_rtg_10,
      CASE
        WHEN t.team_abbr = gt.home_team_abbr THEN gt.away_off_rtg_avg_10
        WHEN t.team_abbr = gt.away_team_abbr THEN gt.home_off_rtg_avg_10
        ELSE NULL::numeric
      END AS opp_off_rtg_10,
      CASE
        WHEN t.team_abbr = gt.home_team_abbr THEN gt.away_pace_5
        WHEN t.team_abbr = gt.away_team_abbr THEN gt.home_pace_5
        ELSE NULL::numeric
      END AS opp_pace_avg_5,
      CASE
        WHEN t.team_abbr = gt.home_team_abbr THEN gt.away_pace_10
        WHEN t.team_abbr = gt.away_team_abbr THEN gt.home_pace_10
        ELSE NULL::numeric
      END AS opp_pace_avg_10,
      gt.market_total AS game_market_total,
      gt.market_spread_home AS game_market_spread,

      -- V013 referee foul risk
      rfr.avg_foul_uplift_crew,
      rfr.avg_foul_per_36_uplift_crew,

      -- V006 opponent style (for tov_vs_opp_stl, fg_pct_vs_opp_blk derived features)
      ostyle.stl_avg_10   AS opp_stl_avg_10,
      ostyle.blk_avg_10   AS opp_blk_avg_10,
      ostyle.fouls_avg_10 AS opp_fouls_avg_10

    FROM teams_today t
    JOIN games_today gt
      ON gt.season = t.season
     AND gt.game_slug = t.game_slug
    JOIN latest_per_player_team lp
      ON lp.season = t.season
     AND lp.team_abbr = t.team_abbr
    LEFT JOIN features.player_game_referee_foul_risk rfr
      ON rfr.game_slug = t.game_slug
     AND rfr.season   = t.season
     AND rfr.player_id = lp.player_id
    LEFT JOIN features.team_style_profile ostyle
      ON ostyle.season    = t.season
     AND ostyle.team_abbr = t.opponent_abbr
     AND ostyle.game_slug = t.game_slug
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

    # NEW: Matchup-scaled projections (player rate × game environment)
    if "pts_per_min_10" in X.columns and "game_pace_est_5" in X.columns:
        X["pts_pace_interaction"] = X["pts_per_min_10"] * X["game_pace_est_5"]
    if "reb_per_min_10" in X.columns and "opp_pace_avg_10" in X.columns:
        X["reb_pace_interaction"] = X["reb_per_min_10"] * X["opp_pace_avg_10"]

    # NEW: Implied share of team total
    if "pts_avg_10" in X.columns and "team_implied_total" in X.columns:
        X["implied_pts_share"] = X["pts_avg_10"] / X["team_implied_total"].clip(lower=80.0)

    # --- V007: Expanded player stat interactions ---
    if "stl_plus_blk_avg_10" in X.columns and "opp_pace_avg_10" in X.columns:
        X["stocks_pace_interaction"] = X["stl_plus_blk_avg_10"] * X["opp_pace_avg_10"] / 100.0
    if "off_reb_avg_10" in X.columns and "def_reb_avg_10" in X.columns:
        total_reb = X["off_reb_avg_10"] + X["def_reb_avg_10"]
        X["off_reb_pct_10"] = X["off_reb_avg_10"] / total_reb.clip(lower=0.5)
    if "tov_avg_10" in X.columns and "opp_stl_avg_10" in X.columns:
        X["tov_vs_opp_stl"] = X["tov_avg_10"] * X["opp_stl_avg_10"] / 10.0
    if "fouls_avg_5" in X.columns and "fouls_avg_10" in X.columns:
        X["fouls_trend_5v10"] = X["fouls_avg_5"] - X["fouls_avg_10"]
    if "plus_minus_avg_5" in X.columns and "plus_minus_avg_10" in X.columns:
        X["pm_trend_5v10"] = X["plus_minus_avg_5"] - X["plus_minus_avg_10"]
    if "fg_pct_avg_10" in X.columns and "opp_blk_avg_10" in X.columns:
        X["fg_pct_vs_opp_blk"] = X["fg_pct_avg_10"] - X["opp_blk_avg_10"] / 10.0

    # --- V013: Referee foul risk ---
    if "avg_foul_uplift_crew" in X.columns and "fouls_avg_10" in X.columns:
        X["ref_adjusted_fouls"] = X["fouls_avg_10"] + X["avg_foul_uplift_crew"].fillna(0)
    if "avg_foul_per_36_uplift_crew" in X.columns and "min_avg_10" in X.columns:
        X["ref_foul_min_risk"] = X["avg_foul_per_36_uplift_crew"].fillna(0) * X["min_avg_10"] / 36.0

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
