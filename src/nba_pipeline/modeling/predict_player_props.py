import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor

from .features import add_player_prop_derived_features

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
    # Best bets filter
    top_n_best_bets: int = 15               # how many to highlight
    best_bets_min_games: int = 7            # minimum n_games_prev_10
    best_bets_min_minutes: float = 22.0     # minimum min_avg_10

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
      away_pace_avg_10 AS away_pace_10,
      -- teammate injury impact (V018)
      COALESCE(home_injured_pts_lost, 0) AS home_injured_pts_lost,
      COALESCE(away_injured_pts_lost, 0) AS away_injured_pts_lost,
      COALESCE(home_injured_out_count, 0) AS home_injured_out_count,
      COALESCE(away_injured_out_count, 0) AS away_injured_out_count
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

      -- 3-game window (hot/cold streaks)
      AVG(h.minutes)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS min_avg_3,
      AVG(h.points)   OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS pts_avg_3,
      AVG(h.rebounds) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS reb_avg_3,
      AVG(h.assists)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS ast_avg_3,

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
      ) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS usage_proxy_avg_5,
      AVG(
        CASE WHEN h.minutes > 0 THEN (h.fga + 0.44 * h.fta) / h.minutes ELSE NULL END
      ) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS usage_proxy_avg_10,

      -- V007 rolling expanded stats
      AVG(h.stl)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS stl_avg_5,
      AVG(h.stl)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS stl_avg_10,
      AVG(h.blk)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS blk_avg_5,
      AVG(h.blk)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS blk_avg_10,
      AVG(h.tov)        OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING) AS tov_avg_5,
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
    -- DISTINCT ON includes season so a player who played for the same team across
    -- multiple seasons gets one row per season.  ORDER BY season DESC ensures we
    -- pick the current-season row first when the joined CTE filters on lp.season = t.season.
    SELECT DISTINCT ON (player_id, team_abbr, season)
      season,
      player_id,
      player_name,
      team_abbr,
      start_ts_utc,
      n_games_prev_10,
      EXTRACT(EPOCH FROM (start_ts_utc - prev_start_ts_utc)) / 86400.0 AS player_rest_days,
      min_avg_3, min_avg_5, min_avg_10,
      pts_avg_3, pts_avg_5, pts_avg_10,
      reb_avg_3, reb_avg_5, reb_avg_10,
      ast_avg_3, ast_avg_5, ast_avg_10,
      fga_avg_10, fta_avg_10, threes_avg_10,
      pts_sd_10, reb_sd_10, ast_sd_10,
      usage_proxy_avg_5, usage_proxy_avg_10,
      -- V007 expanded stats
      stl_avg_5, stl_avg_10,
      blk_avg_5, blk_avg_10,
      tov_avg_5, tov_avg_10,
      stl_plus_blk_avg_10,
      off_reb_avg_10, def_reb_avg_10,
      fg_pct_avg_10,
      plus_minus_avg_5, plus_minus_avg_10,
      fouls_avg_5, fouls_avg_10
    FROM feat
    ORDER BY player_id, team_abbr, season DESC, start_ts_utc DESC
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
      lp.min_avg_3, lp.min_avg_5, lp.min_avg_10,
      lp.pts_avg_3, lp.pts_avg_5, lp.pts_avg_10,
      lp.reb_avg_3, lp.reb_avg_5, lp.reb_avg_10,
      lp.ast_avg_3, lp.ast_avg_5, lp.ast_avg_10,
      lp.fga_avg_10, lp.fta_avg_10, lp.threes_avg_10,
      lp.pts_sd_10, lp.reb_sd_10, lp.ast_sd_10,
      lp.usage_proxy_avg_5, lp.usage_proxy_avg_10,
      -- V007 expanded player stats
      lp.stl_avg_5, lp.stl_avg_10,
      lp.blk_avg_5, lp.blk_avg_10,
      lp.tov_avg_5, lp.tov_avg_10,
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
      ostyle.fouls_avg_10 AS opp_fouls_avg_10,

      -- V018: teammate injury impact
      CASE WHEN t.team_abbr = gt.home_team_abbr
           THEN gt.home_injured_pts_lost
           ELSE gt.away_injured_pts_lost
      END AS teammate_pts_out,
      CASE WHEN t.team_abbr = gt.home_team_abbr
           THEN gt.home_injured_out_count
           ELSE gt.away_injured_out_count
      END AS teammate_out_count,

      -- V016: opponent position defense (latest rolling window before today)
      opd.opp_pts_allowed_role_10,
      opd.opp_reb_allowed_role_10,
      opd.opp_ast_allowed_role_10

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
    LEFT JOIN LATERAL (
        SELECT pd.opp_pts_allowed_role_10,
               pd.opp_reb_allowed_role_10,
               pd.opp_ast_allowed_role_10
        FROM features.opp_position_defense pd
        WHERE pd.opponent_abbr = t.opponent_abbr
          AND pd.role = CASE
                  WHEN lp.fga_avg_10 IS NOT NULL AND lp.min_avg_10 > 0
                       AND (lp.fga_avg_10 * 48.0 / NULLIF(lp.min_avg_10, 0)) >= 10 THEN 'G'
                  WHEN lp.fga_avg_10 IS NOT NULL AND lp.min_avg_10 > 0
                       AND (lp.fga_avg_10 * 48.0 / NULLIF(lp.min_avg_10, 0)) >= 4  THEN 'F'
                  ELSE 'C' END
          AND pd.game_date_et < :game_date
        ORDER BY pd.game_date_et DESC, pd.start_ts_utc DESC
        LIMIT 1
    ) opd ON TRUE
)
SELECT *
FROM joined j
WHERE n_games_prev_10 >= 3
  AND (min_avg_10 IS NULL OR min_avg_10 >= 10.0)
  AND COALESCE(min_avg_5, min_avg_10) >= :min_proj_minutes
  AND NOT EXISTS (
    SELECT 1 FROM raw.nba_injuries inj
    WHERE inj.player_id = j.player_id
      AND UPPER(COALESCE(inj.playing_probability, '')) IN ('OUT', 'DOUBTFUL', 'QUESTIONABLE')
  )
ORDER BY start_ts_utc, game_slug, team_abbr, player_id
"""

def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove non-alpha-space characters."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _load_prop_lines(engine, et_day: date, bookmaker: str = "draftkings") -> dict:
    """Load prop lines from odds.nba_player_prop_lines for et_day.

    Returns dict: {(player_name_norm, stat): (line, over_price, under_price)}
    Gracefully returns empty dict if the table doesn't exist yet.
    """
    sql = text("""
        SELECT player_name_norm, stat, line, over_price, under_price
        FROM odds.nba_player_prop_lines
        WHERE as_of_date = :date AND bookmaker_key = :bk
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"date": et_day, "bk": bookmaker}).fetchall()
        result = {}
        for r in rows:
            key = (r.player_name_norm, r.stat)
            result[key] = (r.line, r.over_price, r.under_price)
        log.info("Loaded %d prop lines from DB for %s (%s)", len(result), et_day, bookmaker)
        return result
    except Exception as exc:
        log.warning("Could not load prop lines (table may not exist yet): %s", exc)
        return {}


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

    # Derived interaction features — single source of truth in features.add_player_prop_derived_features.
    X = add_player_prop_derived_features(X)

    # align schema
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    extra = [c for c in X.columns if c not in feature_cols]
    if extra:
        X = X.drop(columns=extra)
    X = X[feature_cols]

    # Fill with training medians first (avoids pulling rate stats like fg_pct to 0).
    # For bench players, the population median (e.g. ~0.47 FG%) is far better than 0.
    # The final 0.0 fallback only hits one-hot dummy columns and any column
    # whose training median was itself NaN (shouldn't occur for real numeric features).
    for c, m in medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    return X.fillna(0.0)

def _ensure_prop_table(engine) -> None:
    """Create bets.prop_predictions table if it doesn't exist."""
    ddl = text("""
    CREATE SCHEMA IF NOT EXISTS bets;
    CREATE TABLE IF NOT EXISTS bets.prop_predictions (
        id                  SERIAL PRIMARY KEY,
        predicted_at_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        game_date_et        DATE        NOT NULL,
        game_slug           TEXT        NOT NULL,
        player_id           BIGINT      NOT NULL,
        player_name         TEXT,
        team_abbr           TEXT,
        pred_points         NUMERIC,
        pred_rebounds       NUMERIC,
        pred_assists        NUMERIC,
        actual_points       NUMERIC,
        actual_rebounds     NUMERIC,
        actual_assists      NUMERIC,
        UNIQUE (game_date_et, game_slug, player_id)
    );
    """)
    with engine.begin() as conn:
        conn.execute(ddl)


def _save_prop_predictions(df_out: pd.DataFrame, engine, et_day) -> None:
    """Upsert player prop predictions into bets.prop_predictions."""
    _ensure_prop_table(engine)
    upsert_sql = text("""
        INSERT INTO bets.prop_predictions
            (game_date_et, game_slug, player_id, player_name, team_abbr,
             pred_points, pred_rebounds, pred_assists)
        VALUES
            (:game_date_et, :game_slug, :player_id, :player_name, :team_abbr,
             :pred_points, :pred_rebounds, :pred_assists)
        ON CONFLICT (game_date_et, game_slug, player_id) DO UPDATE SET
            predicted_at_utc = NOW(),
            player_name      = EXCLUDED.player_name,
            team_abbr        = EXCLUDED.team_abbr,
            pred_points      = EXCLUDED.pred_points,
            pred_rebounds    = EXCLUDED.pred_rebounds,
            pred_assists     = EXCLUDED.pred_assists
    """)

    rows = []
    for _, r in df_out.iterrows():
        rows.append({
            "game_date_et": et_day,
            "game_slug": r["game_slug"],
            "player_id": int(r["player_id"]),
            "player_name": str(r.get("player_name") or ""),
            "team_abbr": str(r.get("team_abbr") or ""),
            "pred_points": float(r["pred_points"]),
            "pred_rebounds": float(r["pred_rebounds"]),
            "pred_assists": float(r["pred_assists"]),
        })

    if rows:
        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)
        log.info("Saved %d prop predictions to bets.prop_predictions", len(rows))


def _validate_player_predictions(
    pred_pts: np.ndarray,
    pred_reb: np.ndarray,
    pred_ast: np.ndarray,
    player_names: pd.Series,
) -> None:
    """Warn when raw model outputs fall outside sensible per-game NBA ranges before clipping."""
    for i, (pts, reb, ast) in enumerate(zip(pred_pts, pred_reb, pred_ast)):
        name = player_names.iloc[i] if i < len(player_names) else i
        if not (0 <= pts <= 60):
            log.warning("Extreme PTS prediction player=%s raw=%.1f (outside 0-60)", name, pts)
        if not (0 <= reb <= 25):
            log.warning("Extreme REB prediction player=%s raw=%.1f (outside 0-25)", name, reb)
        if not (0 <= ast <= 20):
            log.warning("Extreme AST prediction player=%s raw=%.1f (outside 0-20)", name, ast)


def _rank_best_props(df_raw: pd.DataFrame, df_out: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    """
    Score and rank prop predictions by model confidence + matchup quality.

    Scoring components (all 0-1):
      sample_score    — more games in lookback = more reliable (weight 0.25)
      consistency     — lower pts CV = more predictable scorer    (weight 0.35)
      min_stability   — stable minutes = predictable role         (weight 0.25)
      matchup_quality — player avg vs. what opp allows to role    (weight 0.15)

    Hard filters applied first:
      n_games_prev_10 >= cfg.best_bets_min_games
      min_avg_10      >= cfg.best_bets_min_minutes
    """
    mask = (
        (df_raw["n_games_prev_10"] >= cfg.best_bets_min_games) &
        (df_raw["min_avg_10"] >= cfg.best_bets_min_minutes)
    )
    df_r = df_out[mask].copy()
    raw = df_raw[mask].copy()

    if df_r.empty:
        return df_r

    # Sample size (0-1)
    df_r["_sample"] = (raw["n_games_prev_10"] / 10.0).clip(0, 1).values

    # Consistency: lower CV = higher score
    pts_cv = (raw["pts_sd_10"] / raw["pts_avg_10"].clip(lower=0.5)).clip(0, 3)
    df_r["_consistency"] = (1.0 / (1.0 + pts_cv)).values

    # Minutes stability: small trend = predictable role
    min_trend = (raw["min_avg_5"] - raw["min_avg_10"]).abs()
    df_r["_min_stability"] = (1.0 / (1.0 + min_trend / raw["min_avg_10"].clip(lower=1.0))).values

    # Matchup quality: player avg vs. what opponent allows to their role
    if "opp_pts_allowed_role_10" in raw.columns and raw["opp_pts_allowed_role_10"].notna().any():
        edge = raw["pts_avg_10"] - raw["opp_pts_allowed_role_10"].fillna(raw["pts_avg_10"])
        # scale: +5 edge -> 0.75, 0 edge -> 0.5, -5 edge -> 0.25
        df_r["_matchup"] = (0.5 + edge / (raw["pts_avg_10"].clip(lower=1.0) * 2)).clip(0, 1).values
    else:
        df_r["_matchup"] = 0.5

    df_r["confidence"] = (
        0.25 * df_r["_sample"] +
        0.35 * df_r["_consistency"] +
        0.25 * df_r["_min_stability"] +
        0.15 * df_r["_matchup"]
    )

    # Carry through context columns for display
    df_r["pts_avg_10"]  = raw["pts_avg_10"].values
    df_r["pts_avg_3"]   = raw["pts_avg_3"].values  if "pts_avg_3"  in raw.columns else np.nan
    df_r["min_avg_10"]  = raw["min_avg_10"].values
    df_r["opp_def_rtg"] = raw["opp_def_rtg_10"].values if "opp_def_rtg_10" in raw.columns else np.nan

    return df_r.nlargest(cfg.top_n_best_bets, "confidence").drop(
        columns=["_sample", "_consistency", "_min_stability", "_matchup"], errors="ignore"
    )


def _print_best_bets(
    best: pd.DataFrame,
    pts_mae: float,
    reb_mae: float,
    ast_mae: float,
    prop_lines: dict | None = None,
) -> None:
    """
    Print actionable bet calls for top-confidence prop players.

    When DraftKings lines are available (from odds.nba_player_prop_lines):
      BET OVER  if book line < (model pred - MAE)
      BET UNDER if book line > (model pred + MAE)
      no bet    if line is inside the confidence interval

    When no book line is available, falls back to decision rules:
      OVER if line < {pred - MAE}  |  UNDER if line > {pred + MAE}
    """
    if best.empty:
        return

    if prop_lines is None:
        prop_lines = {}

    n_with_lines = sum(
        1 for _, r in best.iterrows()
        if prop_lines.get((_normalize_name(str(r.get("player_name") or "")), "points"))
    )

    print("\n" + "=" * 65)
    print(f"  BEST PROP BETS  (top {len(best)} by model confidence)")
    if n_with_lines > 0:
        print(f"  DK lines loaded for {n_with_lines}/{len(best)} players")
    else:
        print(f"  No DK lines in DB — showing decision rules (look up line manually)")
    print("=" * 65)

    for _, r in best.iterrows():
        conf = r["confidence"]
        stars = "***" if conf >= 0.78 else "** " if conf >= 0.68 else "*  "

        name = (r.get("player_name") or f"id={int(r['player_id'])}").strip()
        name_ascii = name.encode("ascii", errors="replace").decode("ascii")
        name_norm = _normalize_name(name)
        opp  = r.get("opponent_abbr", "")

        pp = float(r["pred_points"])
        pr = float(r["pred_rebounds"])
        pa = float(r["pred_assists"])

        # Confidence interval bounds (1 MAE ≈ 68% of outcomes)
        pts_lo, pts_hi = max(0.0, pp - pts_mae), pp + pts_mae
        reb_lo, reb_hi = max(0.0, pr - reb_mae), pr + reb_mae
        ast_lo, ast_hi = max(0.0, pa - ast_mae), pa + ast_mae

        # Context tags
        pts3  = r.get("pts_avg_3", np.nan)
        pts10 = r.get("pts_avg_10", np.nan)
        if pd.notna(pts3) and pd.notna(pts10) and pts10 > 0:
            trend = "HOT" if pts3 > pts10 * 1.10 else "COLD" if pts3 < pts10 * 0.90 else ""
        else:
            trend = ""

        def_rtg = r.get("opp_def_rtg", np.nan)
        if pd.notna(def_rtg):
            opp_str = "vs weak D" if def_rtg > 115 else "vs strong D" if def_rtg < 108 else "vs avg D"
        else:
            opp_str = ""

        tags = "  ".join(t for t in [trend, opp_str] if t)
        tags_str = f"  {tags}" if tags else ""

        print(
            f"\n  {stars} {name_ascii} ({r['team_abbr']} vs {opp})"
            f"  proj {r['proj_minutes']:.0f}min{tags_str}  [conf={conf:.2f}]"
        )

        for pred, lo, hi, stat_label, stat_key in [
            (pp, pts_lo, pts_hi, "PTS", "points"),
            (pr, reb_lo, reb_hi, "REB", "rebounds"),
            (pa, ast_lo, ast_hi, "AST", "assists"),
        ]:
            line_data = prop_lines.get((name_norm, stat_key))
            if line_data and line_data[0] is not None:
                book_line = float(line_data[0])
                edge = pred - book_line
                if edge > 0 and book_line < lo:
                    call = f"★ BET OVER   edge=+{edge:.1f}"
                elif edge < 0 and book_line > hi:
                    call = f"★ BET UNDER  edge={edge:.1f}"
                else:
                    call = f"no bet — inside CI {lo:.1f}-{hi:.1f}"
                print(f"         {stat_label:<3}  model={pred:.1f}  DK={book_line:.1f}  {call}")
            else:
                print(f"         {stat_label:<3}  model={pred:.1f}  OVER if line < {lo:.1f}  |  UNDER if line > {hi:.1f}")

    print()


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


def _load_prop_mae(model_dir: Path) -> dict:
    """Load per-stat walk-forward MAE from training. Falls back to defaults if missing."""
    p = model_dir / "backtest_mae.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # Defaults from last known backtest run (2024-12-27 to 2026-01-22, 31663 player-games)
    return {"pts": 4.745, "reb": 1.955, "ast": 1.391}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = PredictConfig()
    et_day = cfg.et_date or datetime.now(_ET).date()
    log.info("Predicting player props for ET date=%s", et_day)

    models, feature_cols, medians = _load_artifacts(cfg)
    engine = create_engine(cfg.pg_dsn)
    _check_injury_staleness(engine)
    prop_lines = _load_prop_lines(engine, et_day)

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
         "proj_minutes", "is_proj_starter", "n_games_prev_10"]
    ].copy()

    raw_pts = models["points"].predict(X)
    raw_reb = models["rebounds"].predict(X)
    raw_ast = models["assists"].predict(X)

    _validate_player_predictions(raw_pts, raw_reb, raw_ast, df["player_name"])

    df_out["pred_points"] = np.clip(raw_pts, 0.0, 60.0)
    df_out["pred_rebounds"] = np.clip(raw_reb, 0.0, 25.0)
    df_out["pred_assists"] = np.clip(raw_ast, 0.0, 20.0)

    # Load walk-forward MAE for confidence intervals (±1 MAE = ~68% of outcomes)
    mae = _load_prop_mae(cfg.model_dir)
    pts_mae = mae.get("pts", 4.745)
    reb_mae = mae.get("reb", 1.955)
    ast_mae = mae.get("ast", 1.391)

    # pretty print grouped by game
    df_out["start_ts_utc"] = pd.to_datetime(df_out["start_ts_utc"], utc=True).dt.tz_convert(_ET)
    df_out = df_out.sort_values(["start_ts_utc","game_slug","team_abbr","pred_points"], ascending=[True,True,True,False])

    for (ts, slug), g in df_out.groupby(["start_ts_utc", "game_slug"], sort=False):
        # derive away @ home from game_slug (format: YYYYMMDD-AWAY-HOME)
        parts = str(slug).split("-")
        matchup = f"{parts[1]} @ {parts[2]}" if len(parts) >= 3 else slug
        print(f"\n{matchup}  {ts:%I:%M %p ET}")

        g_sorted = g.sort_values(["team_abbr", "pred_points"], ascending=[True, False])

        for _, r in g_sorted.iterrows():
            name = (r.get("player_name") or "").strip() or f"player_id={int(r['player_id'])}"
            name = name.encode("ascii", errors="replace").decode("ascii")
            name_norm = _normalize_name((r.get("player_name") or "").strip())

            pts_line = prop_lines.get((name_norm, "points"))
            reb_line = prop_lines.get((name_norm, "rebounds"))
            ast_line = prop_lines.get((name_norm, "assists"))

            def _fmt(pred: float, line_data) -> str:
                if line_data and line_data[0] is not None:
                    return f"{pred:.1f} (DK {float(line_data[0]):.1f})"
                return f"{pred:.1f}"

            print(
                f"  {name}"
                f"  {_fmt(r['pred_points'], pts_line)} PTS"
                f"  {_fmt(r['pred_rebounds'], reb_line)} REB"
                f"  {_fmt(r['pred_assists'], ast_line)} AST"
            )

    # Best bets section
    best = _rank_best_props(df, df_out, cfg)
    _print_best_bets(best, pts_mae=pts_mae, reb_mae=reb_mae, ast_mae=ast_mae, prop_lines=prop_lines)

    # Save predictions to DB
    try:
        _save_prop_predictions(df_out, engine, et_day)
    except Exception as exc:
        log.warning("Could not save prop predictions to DB: %s", exc)


if __name__ == "__main__":
    main()
