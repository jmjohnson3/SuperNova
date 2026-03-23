import json
import logging
import math
import os
import re
import sys
import unicodedata

# Ensure stdout is UTF-8 on Windows (player names may contain non-ASCII chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features, build_fd_parlay_url

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
      NULLIF(p.stats->'miscellaneous'->>'fouls', '')::numeric AS fouls
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
      STDDEV_SAMP(h.minutes)  OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS min_sd_10,

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
      AVG(h.fouls)      OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS fouls_avg_10,

      -- Home/away split averages (last 10 games at each venue type)
      AVG(CASE WHEN h.is_home THEN h.points  ELSE NULL END) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS pts_avg_10_home,
      AVG(CASE WHEN NOT h.is_home THEN h.points  ELSE NULL END) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS pts_avg_10_away,
      AVG(CASE WHEN h.is_home THEN h.rebounds ELSE NULL END) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS reb_avg_10_home,
      AVG(CASE WHEN NOT h.is_home THEN h.rebounds ELSE NULL END) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS reb_avg_10_away,
      AVG(CASE WHEN h.is_home THEN h.minutes  ELSE NULL END) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS min_avg_10_home,
      AVG(CASE WHEN NOT h.is_home THEN h.minutes  ELSE NULL END) OVER (PARTITION BY h.player_id ORDER BY h.start_ts_utc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS min_avg_10_away
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
      pts_sd_10, reb_sd_10, ast_sd_10, min_sd_10,
      usage_proxy_avg_5, usage_proxy_avg_10,
      -- V007 expanded stats
      stl_avg_5, stl_avg_10,
      blk_avg_5, blk_avg_10,
      tov_avg_5, tov_avg_10,
      stl_plus_blk_avg_10,
      off_reb_avg_10, def_reb_avg_10,
      fg_pct_avg_10,
      plus_minus_avg_5, plus_minus_avg_10,
      fouls_avg_5, fouls_avg_10,
      -- Home/away splits
      pts_avg_10_home, pts_avg_10_away,
      reb_avg_10_home, reb_avg_10_away,
      min_avg_10_home, min_avg_10_away
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
      lp.pts_sd_10, lp.reb_sd_10, lp.ast_sd_10, lp.min_sd_10,
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
      -- Home/away splits
      lp.pts_avg_10_home, lp.pts_avg_10_away,
      lp.reb_avg_10_home, lp.reb_avg_10_away,
      lp.min_avg_10_home, lp.min_avg_10_away,

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

      -- Player's own team offensive rating (for synthetic team implied total)
      CASE
        WHEN t.team_abbr = gt.home_team_abbr THEN gt.home_off_rtg_avg_10
        WHEN t.team_abbr = gt.away_team_abbr THEN gt.away_off_rtg_avg_10
        ELSE NULL::numeric
      END AS team_off_rtg_10,

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
      opd.opp_ast_allowed_role_10,

      -- V022: DraftKings prop line prior (most recent closing line before game date)
      prop_prior.prev_book_line_pts,
      prop_prior.prev_book_line_reb,
      prop_prior.prev_book_line_ast,

      -- V023: player shot profile (most recent pregame rolling snapshot before today)
      psp.paint_shot_rate_avg_10,
      psp.pullup_shot_rate_avg_10,
      psp.driving_shot_rate_avg_10,
      psp.catch_and_shoot_rate_avg_10,
      psp.three_pt_rate_pbp_avg_10,
      psp.blocked_rate_avg_10,
      psp.paint_shot_rate_avg_5,
      psp.catch_and_shoot_rate_avg_5,

      -- V024: opponent shot defense (most recent pregame rolling snapshot before today)
      osd.opp_paint_allowed_avg_10,
      osd.opp_pullup_allowed_avg_10,
      osd.opp_driving_allowed_avg_10,
      osd.opp_catch_shoot_allowed_avg_10,
      osd.opp_3pt_allowed_avg_10,
      osd.opp_blocked_rate_avg_10,
      osd.opp_paint_allowed_avg_5,
      osd.opp_3pt_allowed_avg_5,

      -- V025: on/off splits (most recent pregame snapshot)
      poo.on_net_per36_avg_10,
      poo.on_net_per36_avg_5,
      poo.on_off_diff_avg_10,
      poo.on_off_diff_avg_5

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

    -- V022: most recent DraftKings prop line strictly before game date (leakage guard).
    -- Mirrors the same join in features.player_training_features so train/serve features match.
    -- Uses the most recent as_of_date (not MAX line value) to avoid selecting historical outliers.
    LEFT JOIN LATERAL (
        SELECT
            MAX(CASE WHEN pl.stat = 'points'   THEN pl.line END) AS prev_book_line_pts,
            MAX(CASE WHEN pl.stat = 'rebounds' THEN pl.line END) AS prev_book_line_reb,
            MAX(CASE WHEN pl.stat = 'assists'  THEN pl.line END) AS prev_book_line_ast
        FROM odds.nba_player_prop_lines pl
        WHERE pl.player_name_norm = LOWER(REGEXP_REPLACE(
                  unaccent(lp.player_name), '[^a-z ]', '', 'g'))
          AND pl.bookmaker_key = 'draftkings'
          AND pl.as_of_date = (
              SELECT MAX(pl2.as_of_date)
              FROM odds.nba_player_prop_lines pl2
              WHERE pl2.player_name_norm = LOWER(REGEXP_REPLACE(
                        unaccent(lp.player_name), '[^a-z ]', '', 'g'))
                AND pl2.as_of_date < :game_date
                AND pl2.bookmaker_key = 'draftkings'
          )
    ) prop_prior ON TRUE

    -- V023: player shot profile (precomputed snap table, indexed by player_id)
    LEFT JOIN features.player_shot_profile_snap psp
      ON psp.player_id = lp.player_id
     AND psp.game_date_et < :game_date

    -- V024: opponent shot defense (precomputed snap table, indexed by opponent+season)
    LEFT JOIN features.opponent_shot_defense_snap osd
      ON osd.opponent_abbr = t.opponent_abbr
     AND osd.season        = t.season
     AND osd.game_date_et  < :game_date

    -- V025: on/off splits (precomputed snap table, indexed by player_id)
    LEFT JOIN features.player_on_off_snap poo
      ON poo.player_id    = lp.player_id
     AND poo.game_date_et < :game_date
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


def _load_prop_lines(
    engine, et_day: date, bookmakers: list[str] | None = None
) -> dict:
    """Load prop lines from odds.nba_player_prop_lines for et_day.

    Loads all requested bookmakers and returns the best line per (player, stat):
      {(player_name_norm, stat): {
          "best_over":  (line, price, bookmaker_key),   # lowest line  → best for OVER bets
          "best_under": (line, price, bookmaker_key),   # highest line → best for UNDER bets
          "draftkings": (line, over_price, under_price) | None,
          "fanduel":    (line, over_price, under_price) | None,
          ...
      }}
    Gracefully returns empty dict if the table doesn't exist yet.
    """
    if bookmakers is None:
        bookmakers = ["draftkings", "fanduel"]

    sql = text("""
        SELECT player_name_norm, stat, bookmaker_key, line, over_price, under_price,
               over_link, under_link
        FROM odds.nba_player_prop_lines
        WHERE as_of_date = :date AND bookmaker_key = ANY(:bks)
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"date": et_day, "bks": bookmakers}).fetchall()
        # Group by (player_name_norm, stat)
        groups: dict[tuple, dict] = {}
        fd_links_local: dict[tuple, tuple] = {}
        for r in rows:
            key = (r.player_name_norm, r.stat)
            if key not in groups:
                groups[key] = {}
            groups[key][r.bookmaker_key] = (r.line, r.over_price, r.under_price)
            if r.bookmaker_key == "fanduel" and (getattr(r, "over_link", None) or getattr(r, "under_link", None)):
                fd_links_local[key] = (r.over_link, r.under_link)

        # Compute best line for OVER (lowest) and UNDER (highest) across bookmakers
        result = {}
        for key, bk_data in groups.items():
            entry: dict = dict(bk_data)  # bookmaker_key → (line, ov, un)
            best_over = best_under = None
            for bk, (line, ov, un) in bk_data.items():
                if line is None:
                    continue
                line_f = float(line)
                if best_over is None or line_f < best_over[0]:
                    best_over = (line_f, ov, bk)
                if best_under is None or line_f > best_under[0]:
                    best_under = (line_f, un, bk)
            entry["best_over"] = best_over
            entry["best_under"] = best_under
            result[key] = entry

        # Attach FD links to each entry
        for key, entry in result.items():
            ov_l, un_l = fd_links_local.get(key, (None, None))
            entry["fd_over_link"] = ov_l
            entry["fd_under_link"] = un_l

        n = sum(len(v) - 2 for v in result.values())  # approximate # book rows
        log.info("Loaded prop lines for %d player/stat combos from %s", len(result), bookmakers)
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


def _load_player_positions(engine) -> dict:
    """Returns {player_id: most_common_position} from box score history."""
    sql = text("""
        SELECT player_id, MODE() WITHIN GROUP (ORDER BY position) AS primary_position
        FROM raw.nba_boxscore_player_stats
        WHERE position IN ('PG', 'SG', 'SF', 'PF', 'C')
        GROUP BY player_id
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception as exc:
        log.warning("Could not load player positions: %s", exc)
        return {}


def _load_artifacts(cfg: PredictConfig):
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

    # Optional minutes stacking model (added after initial training run)
    minutes_model: XGBRegressor | None = None
    feature_cols_base: list[str] | None = None
    medians_base: dict[str, float] | None = None
    min_path = model_dir / "minutes_xgb.json"
    base_feat_path = model_dir / "feature_columns_base.json"
    base_med_path = model_dir / "feature_medians_base.json"
    if min_path.exists() and base_feat_path.exists() and base_med_path.exists():
        m_min = XGBRegressor()
        m_min.load_model(str(min_path))
        minutes_model = m_min
        feature_cols_base = json.loads(base_feat_path.read_text(encoding="utf-8"))
        medians_base = json.loads(base_med_path.read_text(encoding="utf-8"))
        log.info("Loaded minutes stacking model from %s", min_path)
    else:
        log.info("No minutes stacking model found — using base features only for PTS/REB/AST")

    # Optional LightGBM ensemble models
    lgb_models: dict | None = None
    if _HAS_LGB:
        lgb_paths = {
            "points":   model_dir / "lgb_points.txt",
            "rebounds": model_dir / "lgb_rebounds.txt",
            "assists":  model_dir / "lgb_assists.txt",
        }
        if all(p.exists() for p in lgb_paths.values()):
            lgb_models = {}
            for stat, p in lgb_paths.items():
                bst = lgb.Booster(model_file=str(p))
                lgb_models[stat] = bst
            log.info("Loaded LightGBM ensemble models — predictions will be XGB+LGB averaged")
        else:
            log.info("No LightGBM ensemble models found (run train_player_prop_models to generate)")

    # Item 8: Optional residual prop models
    resid_models: dict[str, XGBRegressor] = {}
    for stat, fname in [("points",   "prop_resid_pts_xgb.json"),
                        ("rebounds", "prop_resid_reb_xgb.json"),
                        ("assists",  "prop_resid_ast_xgb.json")]:
        p = model_dir / fname
        if p.exists():
            m_r = XGBRegressor()
            m_r.load_model(str(p))
            resid_models[stat] = m_r
            log.info("Loaded residual prop model: %s", fname)

    # Item 11: Optional multi-output LGB ensemble (MultiOutputRegressor, joblib-serialized)
    lgb_multi_model = None
    multi_path = model_dir / "lgb_multi.pkl"
    if multi_path.exists():
        mae_data = _load_prop_mae(model_dir)
        if mae_data.get("use_multioutput_ensemble", False):
            try:
                import joblib
                lgb_multi_model = joblib.load(str(multi_path))
                log.info("Loaded multi-output LGB model for ensemble")
            except Exception as exc:
                log.warning("Could not load multi-output LGB model: %s", exc)

    return models, feature_cols, medians, minutes_model, feature_cols_base, medians_base, lgb_models, resid_models, lgb_multi_model

def _prep_X(df: pd.DataFrame, feature_cols: list[str], medians: dict[str, float]) -> pd.DataFrame:
    id_cols = {
        "season","game_slug","game_date_et","start_ts_utc",
        "team_abbr","opponent_abbr","is_home","player_id",
        "player_name",
    }
    X = df.drop(columns=[c for c in id_cols if c in df.columns]).copy()

    if "game_date_et" in df.columns:
        gdt = pd.to_datetime(df["game_date_et"])
        season_start_year = gdt.dt.year.where(gdt.dt.month >= 7, gdt.dt.year - 1)
        season_start = pd.to_datetime(season_start_year.astype(str) + "-10-01")
        X["season_days_elapsed"] = (gdt - season_start).dt.days.values

    # one-hot for team/opponent/season/position
    cat_cols = [c for c in ("season","team_abbr","opponent_abbr","primary_position") if c in df.columns]
    for c in cat_cols:
        if c not in X.columns:
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
    ALTER TABLE bets.prop_predictions
        ADD COLUMN IF NOT EXISTS book_line_pts   NUMERIC,
        ADD COLUMN IF NOT EXISTS book_line_reb   NUMERIC,
        ADD COLUMN IF NOT EXISTS book_line_ast   NUMERIC,
        ADD COLUMN IF NOT EXISTS edge_pts        NUMERIC,
        ADD COLUMN IF NOT EXISTS edge_reb        NUMERIC,
        ADD COLUMN IF NOT EXISTS edge_ast        NUMERIC,
        ADD COLUMN IF NOT EXISTS kelly_pts       NUMERIC,
        ADD COLUMN IF NOT EXISTS kelly_reb       NUMERIC,
        ADD COLUMN IF NOT EXISTS kelly_ast       NUMERIC,
        ADD COLUMN IF NOT EXISTS pts_over_hit    BOOLEAN,
        ADD COLUMN IF NOT EXISTS reb_over_hit    BOOLEAN,
        ADD COLUMN IF NOT EXISTS ast_over_hit    BOOLEAN;
    """)
    with engine.begin() as conn:
        conn.execute(ddl)


def _float_or_none(v) -> float | None:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _save_prop_predictions(df_out: pd.DataFrame, engine, et_day) -> None:
    """Upsert player prop predictions into bets.prop_predictions."""
    _ensure_prop_table(engine)
    upsert_sql = text("""
        INSERT INTO bets.prop_predictions
            (game_date_et, game_slug, player_id, player_name, team_abbr,
             pred_points, pred_rebounds, pred_assists,
             book_line_pts, book_line_reb, book_line_ast,
             edge_pts, edge_reb, edge_ast,
             kelly_pts, kelly_reb, kelly_ast)
        VALUES
            (:game_date_et, :game_slug, :player_id, :player_name, :team_abbr,
             :pred_points, :pred_rebounds, :pred_assists,
             :book_line_pts, :book_line_reb, :book_line_ast,
             :edge_pts, :edge_reb, :edge_ast,
             :kelly_pts, :kelly_reb, :kelly_ast)
        ON CONFLICT (game_date_et, game_slug, player_id) DO UPDATE SET
            predicted_at_utc = NOW(),
            player_name      = EXCLUDED.player_name,
            team_abbr        = EXCLUDED.team_abbr,
            pred_points      = EXCLUDED.pred_points,
            pred_rebounds    = EXCLUDED.pred_rebounds,
            pred_assists     = EXCLUDED.pred_assists,
            book_line_pts    = EXCLUDED.book_line_pts,
            book_line_reb    = EXCLUDED.book_line_reb,
            book_line_ast    = EXCLUDED.book_line_ast,
            edge_pts         = EXCLUDED.edge_pts,
            edge_reb         = EXCLUDED.edge_reb,
            edge_ast         = EXCLUDED.edge_ast,
            kelly_pts        = EXCLUDED.kelly_pts,
            kelly_reb        = EXCLUDED.kelly_reb,
            kelly_ast        = EXCLUDED.kelly_ast
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
            "book_line_pts":  _float_or_none(r.get("prev_book_line_pts")),
            "book_line_reb":  _float_or_none(r.get("prev_book_line_reb")),
            "book_line_ast":  _float_or_none(r.get("prev_book_line_ast")),
            "edge_pts":       _float_or_none(r.get("edge_pts")),
            "edge_reb":       _float_or_none(r.get("edge_reb")),
            "edge_ast":       _float_or_none(r.get("edge_ast")),
            "kelly_pts":      _float_or_none(r.get("kelly_pts")),
            "kelly_reb":      _float_or_none(r.get("kelly_reb")),
            "kelly_ast":      _float_or_none(r.get("kelly_ast")),
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

    return df_r.nlargest(cfg.top_n_best_bets, "confidence")


def _kelly_prop(
    edge_pts: float,
    juice: int = -110,
    shrink: float = 0.60,
    sigma: float = 5.5,
) -> tuple[float, float]:
    """Kelly fraction and win probability for a player prop bet.

    edge_pts : |pred - line| (positive, regardless of direction)
    juice    : American odds on the bet side (e.g. -115)
    shrink   : pull win-prob toward 50% to account for model uncertainty
    sigma    : empirical p68 CI half-width from backtest_mae.json (ci_p68_*)

    Returns (full_kelly_fraction, estimated_win_prob).
    Callers should apply fractional Kelly (typically 1/4) when sizing bets.
    """
    b = 100 / abs(juice)
    p_raw = 1 / (1 + math.exp(-edge_pts / sigma))
    p = 0.5 + (p_raw - 0.5) * shrink
    kelly = max(0.0, (b * p - (1 - p)) / b)
    return kelly, p


def _print_discord_best_bets(
    df_out: pd.DataFrame,
    prop_lines: dict,
    pts_ci: float,
    reb_ci: float,
    ast_ci: float,
    max_legs: int = 8,
) -> None:
    """Print best prop bets for Discord, scanning ALL predicted players.

    Unlike _print_best_bets() which uses only the top-confidence subset,
    this scans every prediction and ranks purely by edge vs book line,
    then builds a FanDuel parlay URL from the top legs.
    """
    ci_map = {"points": pts_ci, "rebounds": reb_ci, "assists": ast_ci}
    candidates: list[tuple] = []  # (edge, name, team, opp, side, line, stat_label, stat_key, fd_link, kelly_pct)

    for _, r in df_out.iterrows():
        name = (r.get("player_name") or "").strip()
        if not name:
            continue
        name_norm = _normalize_name(name)
        team = str(r.get("team_abbr", ""))
        opp = str(r.get("opponent_abbr", ""))

        for pred_col, stat_key, stat_label in [
            ("pred_points",   "points",   "PTS"),
            ("pred_rebounds", "rebounds", "REB"),
            ("pred_assists",  "assists",  "AST"),
        ]:
            pred = float(r.get(pred_col, np.nan))
            if pd.isna(pred):
                continue
            entry = prop_lines.get((name_norm, stat_key))
            if not entry or not isinstance(entry, dict):
                continue
            ci = ci_map.get(stat_key, pts_ci)
            bo = entry.get("best_over")
            bu = entry.get("best_under")
            fd_over  = entry.get("fd_over_link")
            fd_under = entry.get("fd_under_link")

            if bo and bo[0] is not None and fd_over:
                over_line = float(bo[0])
                edge = pred - over_line
                if edge > 0:
                    juice = int(bo[1]) if bo[1] else -110
                    kelly, _ = _kelly_prop(edge, juice=juice, sigma=ci)
                    candidates.append((edge, name, team, opp, "OVER",  over_line,
                                       stat_label, stat_key, fd_over,  kelly / 4 * 100))

            if bu and bu[0] is not None and fd_under:
                under_line = float(bu[0])
                edge = under_line - pred
                if edge > 0:
                    juice = int(bu[1]) if bu[1] else -110
                    kelly, _ = _kelly_prop(edge, juice=juice, sigma=ci)
                    candidates.append((edge, name, team, opp, "UNDER", under_line,
                                       stat_label, stat_key, fd_under, kelly / 4 * 100))

    print("\n**BEST PROP BETS**")
    if not candidates:
        print("*No FD lines available for today's slate*\n")
        return

    # Deduplicate: keep highest edge per (name, stat_key), then rank
    seen: dict[tuple, tuple] = {}
    for c in sorted(candidates, key=lambda x: -x[0]):
        key = (c[1], c[7])
        if key not in seen:
            seen[key] = c
    ranked = sorted(seen.values(), key=lambda x: -x[0])[:max_legs]

    parlay_links: list[str] = []
    qp_props: list[str] = []
    for (edge, name, team, opp, side, line, stat_label, stat_key, fd_link, kelly_pct) in ranked:
        print(f"**{name}** ({team} vs {opp})  {side} {line:.1f} {stat_label} ({kelly_pct:.1f}%k) [Bet FD]({fd_link})")
        parlay_links.append(fd_link)
        qp_props.append(f"{name} ({team} vs {opp}) {side} {line:.1f} {stat_label}")

    if qp_props:
        print("---QUICKPICK:props---")
        for p in qp_props:
            print(p)
        print("---/QUICKPICK---")

    if parlay_links:
        parlay_url = build_fd_parlay_url(parlay_links)
        if parlay_url:
            n = len(parlay_links)
            leg_str = "leg" if n == 1 else "legs"
            print(f"\n[Best Props Parlay ({n} {leg_str})]({parlay_url})")

    print()


def _print_best_bets(
    best: pd.DataFrame,
    pts_ci: float,
    reb_ci: float,
    ast_ci: float,
    prop_lines: dict | None = None,
    discord: bool = False,
) -> list[str]:
    """
    Print actionable bet calls for top-confidence prop players.

    CI half-widths (pts_ci/reb_ci/ast_ci) are the empirical 68th percentile
    of |error| from walk-forward CV — i.e. the true 68% coverage interval.

    When DraftKings lines are available (from odds.nba_player_prop_lines):
      BET OVER  if book line < (model pred - ci)
      BET UNDER if book line > (model pred + ci)
      no bet    if line is inside the confidence interval

    When no book line is available, falls back to decision rules:
      OVER if line < {pred - ci}  |  UNDER if line > {pred + ci}
    """
    if best.empty:
        return []

    if prop_lines is None:
        prop_lines = {}

    qp_props: list[str] = []
    fd_bet_links: list[str] = []

    n_with_lines = sum(
        1 for _, r in best.iterrows()
        if prop_lines.get((_normalize_name(str(r.get("player_name") or "")), "points"))
    )
    books_present: set[str] = set()
    for entry in prop_lines.values():
        if isinstance(entry, dict):
            books_present.update(k for k in entry if k not in ("best_over", "best_under"))
    books_str = "/".join(sorted(books_present)).upper() or "DK"

    if discord:
        print(f"\n**BEST PROP BETS**")
    else:
        print("\n" + "=" * 65)
        print(f"  BEST PROP BETS  (top {len(best)} by model confidence)")
        if n_with_lines > 0:
            print(f"  {books_str} lines loaded for {n_with_lines}/{len(best)} players")
        else:
            print(f"  No book lines in DB — showing decision rules (look up line manually)")
        print("=" * 65)

    for _, r in best.iterrows():
        conf = r["confidence"]
        if discord:
            stars = "⭐⭐⭐" if conf >= 0.78 else "⭐⭐" if conf >= 0.68 else "⭐"
        else:
            stars = "***" if conf >= 0.78 else "** " if conf >= 0.68 else "*  "

        name = (r.get("player_name") or f"id={int(r['player_id'])}").strip()
        name_ascii = name.encode("ascii", errors="replace").decode("ascii")
        name_norm = _normalize_name(name)
        opp  = r.get("opponent_abbr", "")

        pp = float(r["pred_points"])
        pr = float(r["pred_rebounds"])
        pa = float(r["pred_assists"])

        # Confidence interval bounds (empirically calibrated 68% coverage)
        pts_lo, pts_hi = max(0.0, pp - pts_ci), pp + pts_ci
        reb_lo, reb_hi = max(0.0, pr - reb_ci), pr + reb_ci
        ast_lo, ast_hi = max(0.0, pa - ast_ci), pa + ast_ci

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

        # Sub-score breakdown (sample / consistency / min_stability / matchup)
        sub_s   = r.get("_sample",        np.nan)
        sub_c   = r.get("_consistency",   np.nan)
        sub_ms  = r.get("_min_stability", np.nan)
        sub_mq  = r.get("_matchup",       np.nan)
        if all(pd.notna(v) for v in [sub_s, sub_c, sub_ms, sub_mq]):
            sub_str = f"samp={sub_s:.2f} cons={sub_c:.2f} min={sub_ms:.2f} mtch={sub_mq:.2f}"
        else:
            sub_str = ""

        if not discord:
            print(
                f"\n  {stars} {name_ascii} ({r['team_abbr']} vs {opp})"
                f"  proj {r['proj_minutes']:.0f}min{tags_str}  [conf={conf:.2f}]"
            )
            if sub_str:
                print(f"         breakdown: {sub_str}")

        stat_lines_discord: list[str] = []
        for pred, lo, hi, ci, stat_label, stat_key in [
            (pp, pts_lo, pts_hi, pts_ci, "PTS", "points"),
            (pr, reb_lo, reb_hi, reb_ci, "REB", "rebounds"),
            (pa, ast_lo, ast_hi, ast_ci, "AST", "assists"),
        ]:
            entry = prop_lines.get((name_norm, stat_key))
            # entry is now a dict: {bk_key: (line, ov_price, un_price), "best_over": ..., "best_under": ...}
            has_lines = bool(entry and isinstance(entry, dict) and entry.get("best_over") is not None)

            def _fmt_juice(price) -> str:
                if price is None:
                    return ""
                p = int(price)
                return f"{p:+d}" if p >= 0 else f"{p}"

            def _fmt_lines_summary(entry: dict, for_over: bool) -> str:
                """Return e.g. 'DK=27.5 (o-115) FD=28.0 (o-120)' for display."""
                parts = []
                for bk, data in entry.items():
                    if bk in ("best_over", "best_under"):
                        continue
                    if not isinstance(data, tuple) or data[0] is None:
                        continue
                    ln, ov_p, un_p = data
                    price = ov_p if for_over else un_p
                    bk_label = bk.replace("draftkings", "DK").replace("fanduel", "FD").upper()
                    juice_str = f" (o{_fmt_juice(price)})" if for_over else f" (u{_fmt_juice(price)})"
                    parts.append(f"{bk_label}={float(ln):.1f}{juice_str}")
                return "  ".join(parts)

            if has_lines:
                best_over = entry["best_over"]    # (line, price, bk)
                best_under = entry["best_under"]  # (line, price, bk)
                # Determine signal based on model vs best lines
                over_line = float(best_over[0])
                under_line = float(best_under[0])
                over_edge = pred - over_line
                under_edge = pred - under_line
                bet_over = over_edge > 0 and over_line < lo
                bet_under = under_edge < 0 and under_line > hi

                # sigma = p68 CI for this stat (same value used for bet thresholds)
                _sigma_map = {"points": pts_ci, "rebounds": reb_ci, "assists": ast_ci}
                _sigma = _sigma_map.get(stat_key, pts_ci)

                if bet_over:
                    bk_label = best_over[2].replace("draftkings", "DK").replace("fanduel", "FD").upper()
                    juice_str = _fmt_juice(best_over[1])
                    lines_display = _fmt_lines_summary(entry, for_over=True)
                    break_even = 100 / (100 + abs(best_over[1])) if best_over[1] else None
                    be_str = f" BE={break_even:.1%}" if break_even else ""
                    _juice = int(best_over[1]) if best_over[1] else -110
                    kelly, p_win = _kelly_prop(over_edge, juice=_juice, sigma=_sigma)
                    qk_pct = kelly / 4 * 100  # 1/4 Kelly as %
                    if discord:
                        link = entry.get("fd_over_link")
                        link_str = f" [Bet FD]({link})" if link else ""
                        stat_lines_discord.append(f"OVER {over_line:.1f} {stat_label} ({qk_pct:.1f}%k){link_str}")
                        if link:
                            fd_bet_links.append(link)
                    else:
                        print(f"         {stat_label:<3}  model={pred:.1f}  {lines_display}  >> BET OVER @ {bk_label}  edge=+{over_edge:.1f}{be_str}  kelly={qk_pct:.1f}%")
                elif bet_under:
                    bk_label = best_under[2].replace("draftkings", "DK").replace("fanduel", "FD").upper()
                    juice_str = _fmt_juice(best_under[1])
                    lines_display = _fmt_lines_summary(entry, for_over=False)
                    break_even = 100 / (100 + abs(best_under[1])) if best_under[1] else None
                    be_str = f" BE={break_even:.1%}" if break_even else ""
                    _juice = int(best_under[1]) if best_under[1] else -110
                    kelly, p_win = _kelly_prop(abs(under_edge), juice=_juice, sigma=_sigma)
                    qk_pct = kelly / 4 * 100  # 1/4 Kelly as %
                    if discord:
                        link = entry.get("fd_under_link")
                        link_str = f" [Bet FD]({link})" if link else ""
                        stat_lines_discord.append(f"UNDER {under_line:.1f} {stat_label} ({qk_pct:.1f}%k){link_str}")
                        if link:
                            fd_bet_links.append(link)
                    else:
                        print(f"         {stat_label:<3}  model={pred:.1f}  {lines_display}  >> BET UNDER @ {bk_label}  edge={under_edge:.1f}{be_str}  kelly={qk_pct:.1f}%")
                else:
                    lines_display = _fmt_lines_summary(entry, for_over=True)
                    if not discord:
                        print(f"         {stat_label:<3}  model={pred:.1f}  {lines_display}  no bet — inside CI {lo:.1f}-{hi:.1f}")
            else:
                if not discord:
                    print(f"         {stat_label:<3}  model={pred:.1f} ±{ci:.1f}  OVER if line < {lo:.1f}  |  UNDER if line > {hi:.1f}")

        if discord and stat_lines_discord:
            player_line = f"**{name}** ({r['team_abbr']} vs {opp})  " + "  ".join(stat_lines_discord)
            print(player_line)
            qp_props.append(f"{name} ({r['team_abbr']} vs {opp}) " + "  ".join(stat_lines_discord))

    if discord and qp_props:
        print("---QUICKPICK:props---")
        for p in qp_props:
            print(p)
        print("---/QUICKPICK---")

    if discord and fd_bet_links:
        parlay_url = build_fd_parlay_url(fd_bet_links)
        if parlay_url:
            n = len(fd_bet_links)
            leg_str = "leg" if n == 1 else "legs"
            print(f"\n[Best Props Parlay ({n} {leg_str})]({parlay_url})")

    print()
    return fd_bet_links


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
    """Load per-stat walk-forward MAE + calibrated CI quantiles from training.

    Keys returned:
      pts, reb, ast              — walk-forward MAE (kept for reference)
      ci_p68_pts/reb/ast         — empirical 68th percentile of |error| (true 68% CI half-width)
      ci_p90_pts/reb/ast         — empirical 90th percentile of |error|

    Falls back to 1.25 × MAE for p68 (theoretical Normal approximation) if quantiles
    are not in the file (i.e. trained before this calibration was added).
    """
    p = model_dir / "backtest_mae.json"
    # Defaults from last known backtest run
    defaults = {"pts": 4.745, "reb": 1.955, "ast": 1.391}
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else defaults
    # Back-fill p68/p90 from MAE if missing (pre-calibration models)
    for stat in ("pts", "reb", "ast"):
        mae = d.get(stat, defaults[stat])
        if f"ci_p68_{stat}" not in d:
            d[f"ci_p68_{stat}"] = round(mae * 1.25, 3)   # Normal: MAE ≈ 0.798σ → σ ≈ 1.25 MAE
        if f"ci_p90_{stat}" not in d:
            d[f"ci_p90_{stat}"] = round(mae * 2.06, 3)   # Normal: 1.645σ ≈ 2.06 MAE
    return d


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = PredictConfig()
    et_day = cfg.et_date or datetime.now(_ET).date()
    log.info("Predicting player props for ET date=%s", et_day)

    models, feature_cols, medians, minutes_model, feature_cols_base, medians_base, lgb_models, resid_models, lgb_multi_model = _load_artifacts(cfg)
    engine = create_engine(cfg.pg_dsn)
    if os.getenv("DISCORD_FORMAT") != "1":
        _check_injury_staleness(engine, warn_hours=4.0)
    prop_lines = _load_prop_lines(engine, et_day)
    # Item 10: Load player positions for inference
    positions = _load_player_positions(engine)

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

    # Warn about players whose last recorded game is unusually old (load management, illness gaps)
    if "player_rest_days" in df.columns and os.getenv("DISCORD_FORMAT") != "1":
        stale_mask = df["player_rest_days"].fillna(0) > 5
        stale_cols = ["player_name", "team_abbr", "player_rest_days"]
        if "player_id" in df.columns:
            stale_cols = ["player_id"] + stale_cols
        stale_players = df.loc[stale_mask, stale_cols].drop_duplicates("player_id" if "player_id" in df.columns else "player_name")
        if not stale_players.empty:
            print(f"\n  *** STALE PLAYER DATA WARNING: {len(stale_players)} player(s) last played 5+ days ago:")
            for _, row in stale_players.iterrows():
                print(f"      {row.get('player_name', '?')} ({row.get('team_abbr', '?')}) — {row['player_rest_days']:.0f} days since last game")
            print("      Predictions for these players may be less reliable.\n")

    # Override prev_book_line_* with today's actual DK line when available.
    # Today's line is more accurate than the lagged value from yesterday's as_of_date.
    # The model already uses prev_book_line_* as a feature — no retrain needed.
    if prop_lines and not df.empty:
        df = df.copy()
        for col, stat in [
            ("prev_book_line_pts", "points"),
            ("prev_book_line_reb", "rebounds"),
            ("prev_book_line_ast", "assists"),
        ]:
            if col not in df.columns:
                df[col] = np.nan

            def _get_line(player_name: str, _stat: str = stat) -> float:
                key = (_normalize_name(str(player_name)), _stat)
                v = prop_lines.get(key)
                if not v or not isinstance(v, dict):
                    return np.nan
                # Use best_over line as representative model input feature
                best_over = v.get("best_over")
                if best_over and best_over[0] is not None:
                    return float(best_over[0])
                return np.nan

            today_lines = df["player_name"].apply(_get_line)
            n_overridden = int(today_lines.notna().sum())
            if n_overridden > 0:
                df[col] = today_lines.where(today_lines.notna(), df[col])
                log.info(
                    "Overrode %s with today's DK line for %d/%d players",
                    col, n_overridden, len(df),
                )

    # Item 10: Add primary_position to df for OHE in _prep_X
    if positions and "player_id" in df.columns:
        df = df.copy()
        df["primary_position"] = df["player_id"].map(positions).fillna("UNK")
        n_mapped = int((df["primary_position"] != "UNK").sum())
        log.info("Position feature: %d/%d players mapped", n_mapped, len(df))

    # Apply minutes stacking model: predict minutes first, inject as features for PTS/REB/AST.
    if minutes_model is not None and feature_cols_base is not None and medians_base is not None:
        X_base = _prep_X(df, feature_cols_base, medians_base)
        pred_min_raw = minutes_model.predict(X_base)
        pred_min = np.clip(pred_min_raw, 0.0, 48.0)
        fallback_min = df["min_avg_10"].fillna(df["min_avg_5"]).fillna(20.0).values
        df = df.copy()
        df["pred_minutes"] = pred_min
        df["pred_min_vs_avg"] = pred_min - fallback_min
        log.info(
            "Minutes stacking: min=%.1f median=%.1f max=%.1f",
            pred_min.min(), float(np.median(pred_min)), pred_min.max(),
        )

    X = _prep_X(df, feature_cols, medians)

    _bl_cols = [c for c in ("prev_book_line_pts", "prev_book_line_reb", "prev_book_line_ast")
                if c in df.columns]
    df_out = df[
        ["start_ts_utc", "game_slug", "team_abbr", "opponent_abbr", "is_home", "player_id", "player_name",
         "proj_minutes", "is_proj_starter", "n_games_prev_10"] + _bl_cols
    ].copy()

    xgb_pts = models["points"].predict(X)
    xgb_reb = models["rebounds"].predict(X)
    xgb_ast = models["assists"].predict(X)

    # LightGBM ensemble: average XGB + LGB predictions if available
    if lgb_models is not None:
        lgb_pts = lgb_models["points"].predict(X.values)
        lgb_reb = lgb_models["rebounds"].predict(X.values)
        lgb_ast = lgb_models["assists"].predict(X.values)
        raw_pts = (xgb_pts + lgb_pts) / 2.0
        raw_reb = (xgb_reb + lgb_reb) / 2.0
        raw_ast = (xgb_ast + lgb_ast) / 2.0
        log.info("XGB+LGB ensemble active")
    else:
        raw_pts, raw_reb, raw_ast = xgb_pts, xgb_reb, xgb_ast

    # Item 11: XGB + LGB_single + LGB_multi (1/3 each when all present)
    if lgb_multi_model is not None:
        pred_mo_inf = lgb_multi_model.predict(X.values)  # shape (n, 3)
        if lgb_models is not None:
            # 3-way ensemble: xgb + lgb_single + lgb_multi
            raw_pts = (xgb_pts + lgb_pts + pred_mo_inf[:, 0]) / 3.0
            raw_reb = (xgb_reb + lgb_reb + pred_mo_inf[:, 1]) / 3.0
            raw_ast = (xgb_ast + lgb_ast + pred_mo_inf[:, 2]) / 3.0
            log.info("XGB+LGB+LGB_multi 3-way ensemble active")
        else:
            # XGB + LGB_multi only
            raw_pts = (xgb_pts + pred_mo_inf[:, 0]) / 2.0
            raw_reb = (xgb_reb + pred_mo_inf[:, 1]) / 2.0
            raw_ast = (xgb_ast + pred_mo_inf[:, 2]) / 2.0
            log.info("XGB+LGB_multi 2-way ensemble active")

    # Item 8: Apply residual prop models where book lines are available
    if resid_models:
        _book_line_inf = {
            "points":   df["prev_book_line_pts"].values if "prev_book_line_pts" in df.columns else None,
            "rebounds": df["prev_book_line_reb"].values if "prev_book_line_reb" in df.columns else None,
            "assists":  df["prev_book_line_ast"].values if "prev_book_line_ast" in df.columns else None,
        }
        for _stat, _pred_arr in [("points", raw_pts), ("rebounds", raw_reb), ("assists", raw_ast)]:
            if _stat not in resid_models or _book_line_inf[_stat] is None:
                continue
            _bl = _book_line_inf[_stat].astype(float)
            _has_line = ~np.isnan(_bl)
            if not _has_line.any():
                continue
            _dev = resid_models[_stat].predict(X)
            _pred_resid = _bl + _dev
            _pred_arr[_has_line] = _pred_resid[_has_line]
            log.info(
                "Residual applied to %s: %d/%d players have book lines",
                _stat, int(_has_line.sum()), len(_has_line),
            )

    _validate_player_predictions(raw_pts, raw_reb, raw_ast, df["player_name"])

    df_out["pred_points"] = np.clip(raw_pts, 0.0, 60.0)
    df_out["pred_rebounds"] = np.clip(raw_reb, 0.0, 25.0)
    df_out["pred_assists"] = np.clip(raw_ast, 0.0, 20.0)

    # Load walk-forward MAE + calibrated CI quantiles (empirical 68% coverage per stat)
    mae = _load_prop_mae(cfg.model_dir)
    pts_mae = mae.get("pts", 4.745)
    reb_mae = mae.get("reb", 1.955)
    ast_mae = mae.get("ast", 1.391)
    # Empirically calibrated CI half-widths: p68 is the true 68% interval,
    # not ±1 MAE which only covers ~57% for Normal distributions.
    pts_ci = mae.get("ci_p68_pts", round(pts_mae * 1.25, 3))
    reb_ci = mae.get("ci_p68_reb", round(reb_mae * 1.25, 3))
    ast_ci = mae.get("ci_p68_ast", round(ast_mae * 1.25, 3))

    # Item 12: Store book lines, edges, Kelly for grading
    for pred_col, bl_col, edge_col, kelly_col, sigma in [
        ("pred_points",   "prev_book_line_pts", "edge_pts", "kelly_pts", pts_ci),
        ("pred_rebounds", "prev_book_line_reb", "edge_reb", "kelly_reb", reb_ci),
        ("pred_assists",  "prev_book_line_ast", "edge_ast", "kelly_ast", ast_ci),
    ]:
        if bl_col in df_out.columns:
            bl = pd.to_numeric(df_out[bl_col], errors="coerce")
            edge = df_out[pred_col] - bl
            df_out[edge_col] = edge.where(bl.notna(), other=np.nan)
            kelly_vals = np.where(
                bl.notna(),
                [_kelly_prop(abs(e), sigma=sigma)[0] if not np.isnan(e) else np.nan
                 for e in edge.values],
                np.nan,
            )
            df_out[kelly_col] = kelly_vals

    # pretty print grouped by game
    df_out["start_ts_utc"] = pd.to_datetime(df_out["start_ts_utc"], utc=True).dt.tz_convert(_ET)
    df_out = df_out.sort_values(["start_ts_utc","game_slug","team_abbr","pred_points"], ascending=[True,True,True,False])

    discord = os.getenv("DISCORD_FORMAT") == "1"
    if not discord:
        for (ts, slug), g in df_out.groupby(["start_ts_utc", "game_slug"], sort=False):
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
                    if line_data and isinstance(line_data, dict):
                        bo = line_data.get("best_over")
                        if bo and bo[0] is not None:
                            bk_lbl = bo[2].replace("draftkings", "DK").replace("fanduel", "FD").upper()
                            return f"{pred:.1f} ({bk_lbl} {float(bo[0]):.1f})"
                    return f"{pred:.1f}"

                print(
                    f"  {name}"
                    f"  {_fmt(r['pred_points'], pts_line)} PTS"
                    f"  {_fmt(r['pred_rebounds'], reb_line)} REB"
                    f"  {_fmt(r['pred_assists'], ast_line)} AST"
                )

    # Full Discord listing (projected starters only, grouped by game)
    if discord:
        for (ts, slug), g in df_out.groupby(["start_ts_utc", "game_slug"], sort=False):
            parts = str(slug).split("-")
            matchup = f"{parts[1].upper()} @ {parts[2].upper()}" if len(parts) >= 3 else slug
            print(f"\n**{matchup}** · {ts:%I:%M %p ET}")
            g_all = g.sort_values(
                ["team_abbr", "pred_points"], ascending=[True, False]
            )
            for _, r in g_all.iterrows():
                name = (r.get("player_name") or "").strip() or f"id={int(r['player_id'])}"
                name_norm = _normalize_name((r.get("player_name") or "").strip())
                pp = float(r["pred_points"])
                pr = float(r["pred_rebounds"])
                pa = float(r["pred_assists"])
                pts_lo = max(0.0, pp - pts_ci)
                pts_hi = pp + pts_ci
                reb_lo = max(0.0, pr - reb_ci)
                reb_hi = pr + reb_ci
                ast_lo = max(0.0, pa - ast_ci)
                ast_hi = pa + ast_ci
                bet_parts = []
                fd_links: list[str] = []
                for pred, lo, hi, stat_label, stat_key in [
                    (pp, pts_lo, pts_hi, "PTS", "points"),
                    (pr, reb_lo, reb_hi, "REB", "rebounds"),
                    (pa, ast_lo, ast_hi, "AST", "assists"),
                ]:
                    entry = prop_lines.get((name_norm, stat_key))
                    if not entry or not isinstance(entry, dict):
                        continue
                    bo = entry.get("best_over")
                    bu = entry.get("best_under")
                    _sigma = {"points": pts_ci, "rebounds": reb_ci, "assists": ast_ci}.get(stat_key, pts_ci)
                    has_bet = False
                    if bo and bo[0] is not None and pred > float(bo[0]) and float(bo[0]) < lo:
                        _edge = pred - float(bo[0])
                        _juice = int(bo[1]) if bo[1] else -110
                        _kelly, _ = _kelly_prop(_edge, juice=_juice, sigma=_sigma)
                        _qk = _kelly / 4 * 100
                        link = entry.get("fd_over_link")
                        bet_parts.append(f"OVER {float(bo[0]):.1f} {stat_label} ({_qk:.1f}%k)" + (f" [Bet FD]({link})" if link else ""))
                        has_bet = True
                    elif bu and bu[0] is not None and pred < float(bu[0]) and float(bu[0]) > hi:
                        _edge = float(bu[0]) - pred
                        _juice = int(bu[1]) if bu[1] else -110
                        _kelly, _ = _kelly_prop(_edge, juice=_juice, sigma=_sigma)
                        _qk = _kelly / 4 * 100
                        link = entry.get("fd_under_link")
                        bet_parts.append(f"UNDER {float(bu[0]):.1f} {stat_label} ({_qk:.1f}%k)" + (f" [Bet FD]({link})" if link else ""))
                        has_bet = True
                    if not has_bet:
                        # No edge — add a plain directional FD link if available
                        if pred >= (float(bo[0]) if bo and bo[0] is not None else 9999) and entry.get("fd_over_link"):
                            fd_links.append(f"[{stat_label}↑]({entry['fd_over_link']})")
                        elif pred <= (float(bu[0]) if bu and bu[0] is not None else 0) and entry.get("fd_under_link"):
                            fd_links.append(f"[{stat_label}↓]({entry['fd_under_link']})")
                bet_str = "  " + "  ".join(bet_parts) if bet_parts else ""
                link_str = "  " + " ".join(fd_links) if fd_links else ""
                print(f"**{name}** ({r['team_abbr']} vs {r['opponent_abbr']})  {pp:.1f} PTS / {pr:.1f} REB / {pa:.1f} AST{bet_str}{link_str}")

    # Best bets section
    if discord:
        # Discord: scan ALL predicted players by edge vs book line, not just the top-confidence subset
        _print_discord_best_bets(df_out, prop_lines=prop_lines, pts_ci=pts_ci, reb_ci=reb_ci, ast_ci=ast_ci)
    else:
        best = _rank_best_props(df, df_out, cfg)
        _print_best_bets(best, pts_ci=pts_ci, reb_ci=reb_ci, ast_ci=ast_ci, prop_lines=prop_lines, discord=False)

    # Save predictions to DB
    try:
        _save_prop_predictions(df_out, engine, et_day)
    except Exception as exc:
        log.warning("Could not save prop predictions to DB: %s", exc)


if __name__ == "__main__":
    main()
