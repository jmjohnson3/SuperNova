# src/mlb_pipeline/modeling/predict_player_props.py
"""
MLB player prop predictions for today's slate.

Loads pitcher/batter prop models and generates predictions for:
  - pitcher_strikeouts   (starting pitchers from raw.mlb_starting_pitchers)
  - batter_hits          (batters with ab_avg_10 >= 1.5 playing today)
  - batter_total_bases

Edge formula:  edge = pred - book_line
Bet signal:    |edge| >= threshold (K: 1.0, H: 0.4, TB: 0.5)

Discord output (DISCORD_FORMAT=1): compact grouped by game.
DB: bets.mlb_prop_predictions — one row per (game_date_et, game_slug, player_id, stat).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure stdout is UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from .features import add_player_prop_derived_features, build_fd_parlay_url

log = logging.getLogger("mlb_pipeline.modeling.predict_player_props")
_ET = ZoneInfo("America/New_York")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class PredictConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    et_date: date | None = None
    # Minimum samples to include a batter
    # Set to 1 to allow early-season predictions (e.g., Opening Day)
    min_ab_avg_10: float = 1.0
    min_n_games: int = 1
    # Bet thresholds (|edge| >=)
    threshold_strikeouts: float = 1.0
    threshold_hits: float = 0.4
    threshold_total_bases: float = 0.5
    threshold_home_runs: float = 0.25
    threshold_walks: float = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot SQL
# ─────────────────────────────────────────────────────────────────────────────

SQL_PITCHER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
),
today_starters AS (
    SELECT
        sp.game_slug,
        sp.player_id,
        sp.player_name,
        sp.team_abbr,
        CASE WHEN sp.team_abbr = gt.home_team_abbr THEN TRUE ELSE FALSE END AS is_home,
        CASE WHEN sp.team_abbr = gt.home_team_abbr
             THEN gt.away_team_abbr ELSE gt.home_team_abbr END AS opponent_abbr,
        gt.start_ts_utc,
        gt.game_date_et
    FROM raw.mlb_starting_pitchers sp
    JOIN games_today gt ON gt.game_slug = sp.game_slug
    WHERE sp.player_id IS NOT NULL
)
SELECT
    ts.game_slug,
    ts.game_date_et,
    ts.start_ts_utc,
    ts.player_id,
    ts.player_name,
    ts.team_abbr,
    ts.is_home,
    ts.opponent_abbr,
    -- Most recent rolling pitcher stats prior to today
    pr.season,
    pr.ip_avg_5,
    pr.k_pct_5,
    pr.k9_5,
    pr.era_5,
    pr.whip_5,
    pr.fip_5,
    pr.bb_pct_5,
    pr.hr9_5,
    pr.k_pct_10,
    pr.k9_10,
    pr.era_10,
    pr.whip_10,
    pr.fip_10,
    pr.starts_in_window_5,
    pr.starts_in_window_10,
    pr.last_start_k,
    pr.last_start_ip,
    -- Group B: SP rest + home/away splits (MLB003)
    pr.days_since_last_start AS sp_days_since_last_start,
    pr.is_short_rest,
    pr.era_home_10,
    pr.era_away_10,
    pr.k9_home_10,
    pr.k9_away_10,
    pr.fip_home_10,
    pr.fip_away_10,
    -- Opponent batting context
    ob.k_pct_avg_10     AS opp_k_pct_avg_10,
    ob.bb_pct_avg_10    AS opp_bb_pct_avg_10,
    ob.avg_avg_10       AS opp_avg_avg_10,
    ob.hr_avg_10        AS opp_hr_avg_10,
    ob.iso_avg_10       AS opp_iso_avg_10,
    ob.slg_avg_10       AS opp_slg_avg_10,
    -- Park factors
    bf.run_factor       AS park_run_factor,
    bf.hr_factor        AS park_hr_factor,
    -- Weather (dome-zeroed)
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0) END        AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)                     AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END             AS is_dome,
    -- Pitcher handedness
    ph.pitch_hand                                                AS pitcher_hand
FROM today_starters ts
-- Most recent rolling stats
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_pitcher_rolling_mat pr
    WHERE pr.player_id = ts.player_id
      AND pr.game_date_et < %(game_date)s
    ORDER BY pr.game_date_et DESC, pr.game_slug DESC
    LIMIT 1
) pr ON TRUE
-- Opponent team batting (most recent for opponent)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_team_batting_rolling_mat ob
    WHERE ob.team_abbr = ts.opponent_abbr
      AND ob.game_date_et < %(game_date)s
    ORDER BY ob.game_date_et DESC, ob.game_slug DESC
    LIMIT 1
) ob ON TRUE
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = (
        SELECT home_team_abbr FROM raw.mlb_games
        WHERE game_slug = ts.game_slug LIMIT 1
    )
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = ts.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = ts.game_slug LIMIT 1
)
LEFT JOIN raw.mlb_player_handedness ph ON ph.player_id = ts.player_id
ORDER BY ts.start_ts_utc, ts.game_slug, ts.player_id
"""

SQL_BATTER_SNAPSHOTS = """
WITH games_today AS (
    SELECT game_slug, home_team_abbr, away_team_abbr, start_ts_utc, game_date_et
    FROM raw.mlb_games
    WHERE game_date_et = %(game_date)s
),
teams_today AS (
    SELECT home_team_abbr AS team_abbr, away_team_abbr AS opponent_abbr,
           TRUE AS is_home, game_slug, start_ts_utc, game_date_et
    FROM games_today
    UNION ALL
    SELECT away_team_abbr, home_team_abbr, FALSE, game_slug, start_ts_utc, game_date_et
    FROM games_today
),
-- Players who appeared for today's teams recently (last 14 days)
recent_players AS (
    SELECT DISTINCT gl.player_id, gl.team_abbr
    FROM raw.mlb_player_gamelogs gl
    JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
    WHERE gl.team_abbr IN (SELECT team_abbr FROM teams_today)
      AND g.game_date_et >= %(game_date)s::date - INTERVAL '14 days'
      AND gl.at_bats > 0
)
SELECT
    tt.game_slug,
    tt.start_ts_utc,
    tt.game_date_et,
    rp.player_id,
    tt.team_abbr,
    tt.opponent_abbr,
    tt.is_home,
    -- Latest rolling batter stats
    br.season,
    br.hits_avg_5,  br.hits_avg_10,  br.hits_avg_20,  br.hits_sd_10,
    br.hr_avg_5,    br.hr_avg_10,    br.hr_avg_20,
    br.tb_avg_5,    br.tb_avg_10,    br.tb_avg_20,    br.tb_sd_10,
    br.ab_avg_5,    br.ab_avg_10,
    br.avg_avg_10,  br.k_rate_avg_10, br.bb_rate_avg_10, br.iso_avg_10,
    br.hr_rate_avg_5, br.hr_rate_avg_10,
    br.n_games_prev_10,
    br.rest_days,
    -- Opponent SP stats
    sp_r.era_5     AS opp_sp_era_5,
    sp_r.fip_5     AS opp_sp_fip_5,
    sp_r.k_pct_5   AS opp_sp_k_pct_5,
    sp_r.k9_5      AS opp_sp_k9_5,
    sp_r.bb_pct_5  AS opp_sp_bb_pct_5,
    sp_r.whip_5    AS opp_sp_whip_5,
    sp_r.ip_avg_5  AS opp_sp_ip_avg_5,
    -- Group B: opponent SP rest + home/away splits
    sp_r.days_since_last_start AS opp_sp_days_since_last_start,
    sp_r.is_short_rest         AS opp_sp_is_short_rest,
    sp_r.era_home_10           AS opp_sp_era_home_10,
    sp_r.era_away_10           AS opp_sp_era_away_10,
    -- Park factors
    bf.run_factor  AS park_run_factor,
    bf.hr_factor   AS park_hr_factor,
    -- Umpire
    ur.ump_bb9_10  AS ump_bb9_avg_10,
    ur.ump_k9_10   AS ump_k9_avg_10,
    -- Weather (dome-zeroed)
    wx.temperature_f,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(wx.wind_speed_mph::FLOAT, 0.0) END        AS wind_speed_mph,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(SIN(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_sin,
    CASE WHEN v.roof_type = 'dome' THEN 0.0
         ELSE COALESCE(COS(RADIANS(wx.wind_direction_deg::FLOAT)), 0.0) END AS wind_cos,
    COALESCE(wx.precip_prob_pct::FLOAT, 0.0)                     AS precip_prob_pct,
    CASE WHEN v.roof_type = 'dome' THEN 1 ELSE 0 END             AS is_dome,
    -- Lineup slot
    br.batting_order_avg_5,
    br.batting_order_avg_10,
    -- Batter + opponent SP handedness (OHE'd by _prep_features)
    bh.bat_side                AS batter_hand,
    opp_ph.pitch_hand          AS opp_sp_hand,
    -- Matched-hand stats (the split that matches today's opponent's throwing hand)
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.hits_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.hits_avg_40_vs_rhp END AS hits_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.tb_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.tb_avg_40_vs_rhp END AS tb_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.k_rate_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.k_rate_avg_40_vs_rhp END AS k_rate_avg_40_vs_hand,
    CASE WHEN opp_ph.pitch_hand = 'L' THEN bvh.iso_avg_40_vs_lhp
         WHEN opp_ph.pitch_hand = 'R' THEN bvh.iso_avg_40_vs_rhp END AS iso_avg_40_vs_hand,
    -- Split differential (positive = better vs LHP)
    COALESCE(bvh.hits_avg_40_vs_lhp, 0) - COALESCE(bvh.hits_avg_40_vs_rhp, 0) AS hits_hand_split_40,
    COALESCE(bvh.tb_avg_40_vs_lhp,   0) - COALESCE(bvh.tb_avg_40_vs_rhp,   0) AS tb_hand_split_40,
    -- Sample sizes
    bvh.n_games_vs_lhp_40,
    bvh.n_games_vs_rhp_40,
    -- Cross-season rolling features (MLB013)
    bcs.n_games_prev_10_cs,
    bcs.hits_avg_10_cs,  bcs.hits_avg_20_cs,
    bcs.tb_avg_10_cs,    bcs.tb_avg_20_cs,
    bcs.hr_avg_10_cs,    bcs.hr_rate_avg_10_cs,
    bcs.ab_avg_10_cs,
    bcs.k_rate_avg_10_cs, bcs.bb_rate_avg_10_cs, bcs.iso_avg_10_cs,
    -- Prior full-season stats (MLB014)
    pss.prev_games,
    pss.prev_hits_avg,  pss.prev_tb_avg,  pss.prev_hr_avg,
    pss.prev_ab_avg,    pss.prev_k_rate,  pss.prev_bb_rate,
    pss.prev_iso,       pss.prev_hr_rate,
    -- Career H2H stats vs today's SP (MLB015, most recent entry before game date)
    h2h.h2h_games,
    h2h.h2h_ba,
    h2h.h2h_obp,
    h2h.h2h_slg,
    h2h.h2h_k_rate,
    h2h.h2h_iso
FROM teams_today tt
JOIN recent_players rp ON rp.team_abbr = tt.team_abbr
-- Most recent rolling batter stats prior to today
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_player_batting_rolling_mat br
    WHERE br.player_id = rp.player_id
      AND br.game_date_et < %(game_date)s
    ORDER BY br.game_date_et DESC, br.game_slug DESC
    LIMIT 1
) br ON TRUE
-- Opposing starting pitcher
LEFT JOIN raw.mlb_starting_pitchers sp
    ON sp.game_slug = tt.game_slug
    AND sp.team_abbr = tt.opponent_abbr
-- Opposing SP rolling stats
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_pitcher_rolling_mat sp_r
    WHERE sp_r.player_id = sp.player_id
      AND sp_r.game_date_et < %(game_date)s
    ORDER BY sp_r.game_date_et DESC, sp_r.game_slug DESC
    LIMIT 1
) sp_r ON TRUE
-- Opponent SP handedness
LEFT JOIN raw.mlb_player_handedness opp_ph ON opp_ph.player_id = sp.player_id
-- Batter's own handedness
LEFT JOIN raw.mlb_player_handedness bh ON bh.player_id = rp.player_id
-- Most recent batter-vs-hand rolling stats prior to today (LATERAL for latest snapshot)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_batting_vs_hand_mat
    WHERE player_id = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) bvh ON TRUE
-- Cross-season rolling stats (no season boundary) — gives prior-year data early in season
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_player_batting_rolling_cross_mat
    WHERE player_id = rp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC, game_slug DESC
    LIMIT 1
) bcs ON TRUE
-- Prior full-season stats — stable 162-game prior for each player
LEFT JOIN features.mlb_player_prev_season_stats_mat pss
    ON pss.player_id = rp.player_id
    AND pss.season = %(prior_season)s
-- Career H2H stats vs today's SP (most recent entry before game date)
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_batter_vs_sp_mat
    WHERE batter_id    = rp.player_id
      AND pitcher_id   = sp.player_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC
    LIMIT 1
) h2h ON TRUE
LEFT JOIN features.mlb_ballpark_factors bf
    ON bf.team_abbr = (
        SELECT home_team_abbr FROM games_today
        WHERE game_slug = tt.game_slug LIMIT 1
    )
-- Home plate umpire rolling stats
LEFT JOIN raw.mlb_game_umpires gu
    ON gu.game_slug = tt.game_slug AND gu.ump_position = 'Home Plate'
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_umpire_rolling_mat
    WHERE umpire_id = gu.umpire_id
      AND game_date_et < %(game_date)s
    ORDER BY game_date_et DESC LIMIT 1
) ur ON TRUE
-- Weather
LEFT JOIN raw.mlb_weather wx ON wx.game_slug = tt.game_slug
LEFT JOIN raw.mlb_venues  v  ON v.venue_id = (
    SELECT venue_id FROM raw.mlb_games WHERE game_slug = tt.game_slug LIMIT 1
)
WHERE br.ab_avg_10 >= %(min_ab_avg_10)s
  AND br.n_games_prev_10 >= %(min_n_games)s
ORDER BY tt.start_ts_utc, tt.game_slug, rp.player_id
"""

SQL_PROP_LINES = """
SELECT
    player_name_norm,
    stat,
    bookmaker_key,
    line,
    over_price,
    under_price,
    over_link,
    under_link
FROM odds.mlb_player_prop_lines
WHERE as_of_date = %(game_date)s
ORDER BY bookmaker_key, player_name_norm, stat
"""

SQL_PLAYER_NAMES = """
SELECT DISTINCT player_id,
       COALESCE(player_name, 'id=' || player_id::text) AS player_name
FROM raw.mlb_starting_pitchers
UNION ALL
SELECT DISTINCT player_id,
       player_name
FROM raw.mlb_starting_pitchers
WHERE player_name IS NOT NULL
"""


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pitcher_artifacts(model_dir: Path):
    """Returns (xgb_k, lgb_k, feature_cols, medians, backtest)."""
    xgb_k = XGBRegressor()
    xgb_k.load_model(str(model_dir / "strikeouts_xgb.json"))

    lgb_k = None
    lgb_path = model_dir / "lgb_strikeouts.txt"
    if _HAS_LGB and lgb_path.exists():
        lgb_k = lgb.Booster(model_file=str(lgb_path))

    feat = json.loads((model_dir / "feature_columns_pitchers.json").read_text())
    meds = json.loads((model_dir / "feature_medians_pitchers.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    return xgb_k, lgb_k, feat, meds, bt


def _load_batter_artifacts(model_dir: Path):
    """Returns (xgb_hits, lgb_hits, xgb_tb, lgb_tb, xgb_hr, lgb_hr,
                xgb_walks, lgb_walks, feature_cols, medians, backtest)."""
    xgb_h = XGBRegressor()
    xgb_h.load_model(str(model_dir / "hits_xgb.json"))

    xgb_tb = XGBRegressor()
    xgb_tb.load_model(str(model_dir / "total_bases_xgb.json"))

    xgb_hr = XGBRegressor()
    xgb_hr.load_model(str(model_dir / "home_runs_xgb.json"))

    xgb_walks = XGBRegressor()
    xgb_walks.load_model(str(model_dir / "walks_xgb.json"))

    lgb_h = lgb_tb_m = lgb_hr = lgb_walks = None
    if _HAS_LGB:
        h_path    = model_dir / "lgb_hits.txt"
        tb_path   = model_dir / "lgb_total_bases.txt"
        hr_path   = model_dir / "lgb_home_runs.txt"
        walk_path = model_dir / "lgb_walks.txt"
        if h_path.exists():
            lgb_h = lgb.Booster(model_file=str(h_path))
        if tb_path.exists():
            lgb_tb_m = lgb.Booster(model_file=str(tb_path))
        if hr_path.exists():
            lgb_hr = lgb.Booster(model_file=str(hr_path))
        if walk_path.exists():
            lgb_walks = lgb.Booster(model_file=str(walk_path))

    feat = json.loads((model_dir / "feature_columns_batters.json").read_text())
    meds = json.loads((model_dir / "feature_medians_batters.json").read_text())

    bt_path = model_dir / "backtest_mae.json"
    bt = json.loads(bt_path.read_text()) if bt_path.exists() else {}

    return xgb_h, lgb_h, xgb_tb, lgb_tb_m, xgb_hr, lgb_hr, xgb_walks, lgb_walks, feat, meds, bt


# ─────────────────────────────────────────────────────────────────────────────
# Feature prep
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in list(X.columns):
        if is_numeric_dtype(X[c]) or is_bool_dtype(X[c]) or is_datetime64_any_dtype(X[c]):
            continue
        if X[c].dtype == "object" or str(X[c].dtype).startswith("string"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _prep_features(
    df: pd.DataFrame,
    meta_cols: List[str],
    feature_cols: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    """Drop meta, OHE season, coerce, add derived, fill medians, align to feature_cols."""
    X = df.drop(columns=[c for c in meta_cols if c in df.columns]).copy()

    if "season" in X.columns:
        X = pd.get_dummies(X, columns=["season"], drop_first=False, dummy_na=False)
    if "is_home" in X.columns:
        X["is_home"] = X["is_home"].astype("boolean").fillna(False).astype(int)

    X = _coerce_numeric(X)

    bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
    if bad:
        X = pd.get_dummies(X, columns=bad, drop_first=False, dummy_na=True)

    X = add_player_prop_derived_features(X)

    # Align to training schema
    X = X.reindex(columns=feature_cols)
    for c, m in medians.items():
        if c in X.columns:
            X[c] = X[c].fillna(m)
    return X.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization (for matching prop lines)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Lower-case, strip accents, remove punctuation, collapse whitespace."""
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_str = re.sub(r"[^a-z0-9\s]", "", ascii_str.lower())
    return re.sub(r"\s+", " ", ascii_str).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────

def _kelly_prop(edge: float, sigma: float, max_kelly: float = 0.05) -> float:
    """Simple Kelly fraction for a prop bet.
    edge = pred - line (positive = over), sigma = CI p68.
    Approximates win probability from edge / sigma ratio.
    """
    if sigma <= 0:
        return 0.0
    z = edge / sigma
    # Use logistic approximation: p ≈ 1 / (1 + exp(-z))
    p = 1.0 / (1.0 + math.exp(-z))
    # Kelly fraction = (p * 2 - 1) for 1:1 odds (simplified for -110 juice)
    k = max(0.0, (p - 0.5238) / 0.4762)  # break-even is 52.38% at -110
    return min(k, max_kelly)


# ─────────────────────────────────────────────────────────────────────────────
# DB table
# ─────────────────────────────────────────────────────────────────────────────

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bets.mlb_prop_predictions (
    id               SERIAL PRIMARY KEY,
    game_date_et     DATE        NOT NULL,
    game_slug        TEXT        NOT NULL,
    player_id        BIGINT      NOT NULL,
    player_name      TEXT,
    team_abbr        TEXT,
    stat             TEXT        NOT NULL,
    pred_value       NUMERIC,
    book_line        NUMERIC,
    edge             NUMERIC,
    kelly_fraction   NUMERIC,
    actual_value     NUMERIC,
    over_hit         BOOLEAN,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (game_date_et, game_slug, player_id, stat)
);
"""

_UPSERT_SQL = """
INSERT INTO bets.mlb_prop_predictions
    (game_date_et, game_slug, player_id, player_name, team_abbr, stat,
     pred_value, book_line, edge, kelly_fraction)
VALUES
    (%(game_date_et)s, %(game_slug)s, %(player_id)s, %(player_name)s, %(team_abbr)s,
     %(stat)s, %(pred_value)s, %(book_line)s, %(edge)s, %(kelly_fraction)s)
ON CONFLICT (game_date_et, game_slug, player_id, stat) DO UPDATE SET
    player_name    = EXCLUDED.player_name,
    pred_value     = EXCLUDED.pred_value,
    book_line      = EXCLUDED.book_line,
    edge           = EXCLUDED.edge,
    kelly_fraction = EXCLUDED.kelly_fraction
"""


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_ENSURE_TABLE_SQL)
    conn.commit()


def _save_predictions(conn, rows: List[Dict]) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        for row in rows:
            cur.execute(_UPSERT_SQL, row)
    conn.commit()
    log.info("Saved %d prop prediction rows", len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Prop line loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_prop_lines(conn, game_date: date) -> Dict[Tuple[str, str], Dict]:
    """
    Returns {(player_name_norm, stat): {line, over_link, under_link}}.
    Prefers FanDuel, falls back to DraftKings.
    """
    df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
    if df.empty:
        return {}

    result: Dict[Tuple[str, str], Dict] = {}
    priority = {"fanduel": 0, "draftkings": 1}

    for _, row in df.iterrows():
        bk = str(row.get("bookmaker_key", ""))
        if bk not in priority:
            continue
        key = (str(row["player_name_norm"]), str(row["stat"]))
        existing = result.get(key)
        if existing is None or priority[bk] < priority.get(existing["bookmaker_key"], 99):
            def _clean(v):
                if v is None:
                    return None
                try:
                    if pd.isna(v):
                        return None
                except (TypeError, ValueError):
                    pass
                return str(v) if v else None
            result[key] = {
                "bookmaker_key": bk,
                "line": float(row["line"]) if row["line"] is not None else None,
                "over_link": _clean(row.get("over_link")),
                "under_link": _clean(row.get("under_link")),
            }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_ensemble(
    X: pd.DataFrame,
    xgb_model: XGBRegressor,
    lgb_model,
) -> np.ndarray:
    preds = xgb_model.predict(X)
    if lgb_model is not None:
        lgb_preds = lgb_model.predict(X.values)
        preds = (preds + lgb_preds) / 2.0
    return preds


_PITCHER_META = ["game_slug", "game_date_et", "start_ts_utc", "player_id",
                 "player_name", "team_abbr", "opponent_abbr"]
_BATTER_META = ["game_slug", "game_date_et", "start_ts_utc", "player_id",
                "team_abbr", "opponent_abbr"]


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_edge(edge: Optional[float], threshold: float, direction: str = "OVER") -> str:
    if edge is None or math.isnan(edge):
        return "[no line]"
    abs_e = abs(edge)
    direction = "OVER" if edge > 0 else "UNDER"
    if abs_e >= threshold:
        return f"★ {direction} +{abs_e:.2f}"
    return f"[{direction} +{abs_e:.2f}]" if abs_e > 0 else "[no edge]"


def _collect_all_prop_links(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
) -> List[str]:
    """Collect one FD bet link per prop, taking the model's predicted direction.

    Used to build the All Props Parlay (every player, model's side, regardless of edge).
    """
    links: List[str] = []

    for row in all_pitcher_rows:
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld.get("line") is None:
            continue
        edge = pred_k - ld["line"]
        link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
        if link:
            links.append(link)

    for row in all_batter_rows:
        norm = _normalize_name(row.get("player_name", f"id={row['player_id']}"))
        for pred_col, stat_key in [
            ("pred_hits",        "batter_hits"),
            ("pred_total_bases", "batter_total_bases"),
            ("pred_home_runs",   "batter_home_runs"),
            ("pred_walks",       "batter_walks"),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld.get("line") is None:
                continue
            edge = pred_v - ld["line"]
            link = ld.get("over_link") if edge >= 0 else ld.get("under_link")
            if link:
                links.append(link)

    return links


def _print_discord(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    game_map: Dict[str, Dict],  # game_slug -> {home, away, start_ts}
    cfg: PredictConfig,
) -> List[str]:
    """Print Discord-formatted output. Returns list of FD bet links for parlay."""
    is_discord = os.getenv("DISCORD_FORMAT") == "1"
    fd_links: List[str] = []

    # Group by game
    games_seen = sorted(
        set(r["game_slug"] for r in all_pitcher_rows + all_batter_rows),
        key=lambda s: (game_map.get(s, {}).get("start_ts_utc", ""), s)
    )

    for slug in games_seen:
        gm = game_map.get(slug, {})
        home = gm.get("home", "???")
        away = gm.get("away", "???")
        start_ts = gm.get("start_ts_utc")
        if start_ts:
            try:
                dt_et = pd.Timestamp(start_ts).tz_convert(_ET)
                hour = dt_et.hour % 12 or 12
                time_str = f"{hour}:{dt_et.strftime('%M %p')} ET"
            except Exception:
                time_str = ""
        else:
            time_str = ""

        header = f"**{away} @ {home}**" + (f" · {time_str}" if time_str else "")
        print(header)

        # Pitchers for this game
        for row in all_pitcher_rows:
            if row["game_slug"] != slug:
                continue
            name = row.get("player_name", f"id={row['player_id']}")
            pred_k = row.get("pred_strikeouts")
            if pred_k is None:
                continue

            norm = _normalize_name(name)
            line_data = prop_lines.get((norm, "pitcher_strikeouts"))
            line = line_data["line"] if line_data else None

            edge = (pred_k - line) if (line is not None) else None
            has_edge = edge is not None and abs(edge) >= cfg.threshold_strikeouts

            if line is not None:
                bk = line_data.get("bookmaker_key", "")
                book_lbl = "FD" if bk == "fanduel" else "DK"
                dir_str = "O" if edge > 0 else "U"
                k_part = f"K {pred_k:.1f}/{dir_str}{line:.1f}"
                if has_edge:
                    bet_link = (line_data.get("over_link") if edge > 0 else line_data.get("under_link"))
                    link_str = f" [Bet {book_lbl}](<{bet_link}>)" if bet_link else ""
                    if bet_link:
                        fd_links.append(bet_link)
                    print(f"  SP {name}: ★ {k_part} +{abs(edge):.1f}{link_str}")
                else:
                    bet_link = line_data.get("over_link") or line_data.get("under_link")
                    link_str = f" [{book_lbl}](<{bet_link}>)" if bet_link else ""
                    print(f"  SP {name}: {k_part}{link_str}")
            else:
                print(f"  SP {name}: K {pred_k:.1f}")

        # Batters for this game — one compact line per player
        for row in all_batter_rows:
            if row["game_slug"] != slug:
                continue
            name = row.get("player_name", f"id={row['player_id']}")
            norm = _normalize_name(name)

            # CI scaling: widen effective threshold for thin-sample early-season players
            _n_g = row.get("n_games_prev_10") or 0
            _ci_scale = math.sqrt(10.0 / max(_n_g, 1))

            parts: List[str] = []
            for stat_lbl, pred_col, stat_key, threshold in [
                ("H",  "pred_hits",        "batter_hits",         cfg.threshold_hits),
                ("TB", "pred_total_bases",  "batter_total_bases",  cfg.threshold_total_bases),
                ("HR", "pred_home_runs",    "batter_home_runs",    cfg.threshold_home_runs),
                ("BB", "pred_walks",        "batter_walks",        cfg.threshold_walks),
            ]:
                pred_v = row.get(pred_col)
                if pred_v is None:
                    continue

                line_data = prop_lines.get((norm, stat_key))
                line = line_data["line"] if line_data else None

                # Skip HR/BB projections when there's no book line — too noisy
                if line is None and stat_lbl in ("HR", "BB"):
                    continue

                edge = (pred_v - line) if (line is not None) else None
                has_edge = edge is not None and abs(edge) >= threshold * _ci_scale

                if line is not None:
                    bk = line_data.get("bookmaker_key", "")
                    book_lbl = "FD" if bk == "fanduel" else "DK"
                    dir_str = "O" if edge > 0 else "U"
                    if has_edge:
                        bet_link = (line_data.get("over_link") if edge > 0 else line_data.get("under_link"))
                        link_str = f" [Bet {book_lbl}](<{bet_link}>)" if bet_link else ""
                        if bet_link:
                            fd_links.append(bet_link)
                        parts.append(f"★ {stat_lbl} {pred_v:.2f}/{dir_str}{line:.1f} +{abs(edge):.2f}{link_str}")
                    else:
                        bet_link = line_data.get("over_link") or line_data.get("under_link")
                        link_str = f" [{book_lbl}](<{bet_link}>)" if bet_link else ""
                        parts.append(f"{stat_lbl} {pred_v:.2f}/{dir_str}{line:.1f}{link_str}")
                else:
                    parts.append(f"{stat_lbl} {pred_v:.2f}")

            if parts:
                print(f"  {name}: {' | '.join(parts)}")

        print("")

    return fd_links


def _print_best_bets(
    all_pitcher_rows: List[Dict],
    all_batter_rows: List[Dict],
    prop_lines: Dict,
    cfg: PredictConfig,
) -> List[str]:
    """Print best bets ranked by |edge|. Returns FD links for parlay."""
    fd_links: List[str] = []
    best: List[Dict] = []

    for row in all_pitcher_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        pred_k = row.get("pred_strikeouts")
        if pred_k is None:
            continue
        norm = _normalize_name(name)
        ld = prop_lines.get((norm, "pitcher_strikeouts"))
        if not ld or ld["line"] is None:
            continue
        edge = pred_k - ld["line"]
        if abs(edge) >= cfg.threshold_strikeouts:
            best.append({
                "name": name, "stat": "K", "pred": pred_k,
                "line": ld["line"], "edge": edge,
                "bet_link": ld.get("over_link") if edge > 0 else ld.get("under_link"),
                "bookmaker_key": ld.get("bookmaker_key", ""),
                "team": row.get("team_abbr", ""),
            })

    for row in all_batter_rows:
        name = row.get("player_name", f"id={row['player_id']}")
        norm = _normalize_name(name)
        _n_g = row.get("n_games_prev_10") or 0
        _ci_scale = math.sqrt(10.0 / max(_n_g, 1))
        for stat, pred_col, stat_key, threshold in [
            ("H",  "pred_hits",        "batter_hits",        cfg.threshold_hits),
            ("TB", "pred_total_bases",  "batter_total_bases", cfg.threshold_total_bases),
            ("HR", "pred_home_runs",    "batter_home_runs",   cfg.threshold_home_runs),
            ("BB", "pred_walks",        "batter_walks",       cfg.threshold_walks),
        ]:
            pred_v = row.get(pred_col)
            if pred_v is None:
                continue
            ld = prop_lines.get((norm, stat_key))
            if not ld or ld["line"] is None:
                continue
            edge = pred_v - ld["line"]
            if abs(edge) >= threshold * _ci_scale:
                best.append({
                    "name": name, "stat": stat, "pred": pred_v,
                    "line": ld["line"], "edge": edge,
                    "bet_link": ld.get("over_link") if edge > 0 else ld.get("under_link"),
                    "bookmaker_key": ld.get("bookmaker_key", ""),
                    "team": row.get("team_abbr", ""),
                })

    best.sort(key=lambda r: abs(r["edge"]), reverse=True)

    if best:
        print("─" * 40)
        print("**Best Props (ranked by edge)**")
        for b in best[:10]:
            direction = "OVER" if b["edge"] > 0 else "UNDER"
            book_lbl = "FD" if b.get("bookmaker_key") == "fanduel" else "DK"
            bet_link = b.get("bet_link")
            link_str = f"  [Bet {book_lbl}](<{bet_link}>)" if bet_link else ""
            print(f"  {b['name']} ({b['team']}) {b['stat']} {direction} {b['line']} "
                  f"| pred {b['pred']:.2f} | edge {b['edge']:+.2f}{link_str}")
            if bet_link:
                fd_links.append(bet_link)

    return fd_links


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_props(cfg: PredictConfig) -> None:
    et_date = cfg.et_date or datetime.now(tz=_ET).date()

    # ── Load models ───────────────────────────────────────────────────────────
    model_dir = cfg.model_dir
    pitcher_artifacts_ok = (
        (model_dir / "strikeouts_xgb.json").exists() and
        (model_dir / "feature_columns_pitchers.json").exists()
    )
    batter_artifacts_ok = (
        (model_dir / "hits_xgb.json").exists() and
        (model_dir / "total_bases_xgb.json").exists() and
        (model_dir / "home_runs_xgb.json").exists() and
        (model_dir / "walks_xgb.json").exists() and
        (model_dir / "feature_columns_batters.json").exists()
    )

    xgb_k = lgb_k = feat_p = meds_p = bt = None
    xgb_h = lgb_h = xgb_tb_m = lgb_tb_m = None
    xgb_hr = lgb_hr = xgb_walks_m = lgb_walks_m = feat_b = meds_b = None

    if pitcher_artifacts_ok:
        try:
            xgb_k, lgb_k, feat_p, meds_p, bt = _load_pitcher_artifacts(model_dir)
        except Exception:
            log.exception("Failed to load pitcher artifacts")
            pitcher_artifacts_ok = False

    if batter_artifacts_ok:
        try:
            (xgb_h, lgb_h, xgb_tb_m, lgb_tb_m,
             xgb_hr, lgb_hr, xgb_walks_m, lgb_walks_m,
             feat_b, meds_b, bt) = _load_batter_artifacts(model_dir)
        except Exception:
            log.exception("Failed to load batter artifacts")
            batter_artifacts_ok = False

    if not pitcher_artifacts_ok and not batter_artifacts_ok:
        log.warning("No prop models found. Run train_player_prop_models first.")
        print("_(No prop models — run train_player_prop_models first)_")
        return

    # ── Connect and fetch data ─────────────────────────────────────────────
    conn = psycopg2.connect(cfg.pg_dsn)
    _ensure_schema(conn)

    prop_lines = _load_prop_lines(conn, et_date)
    log.info("Loaded %d prop line entries for %s", len(prop_lines), et_date)

    # ── Pitcher predictions ────────────────────────────────────────────────
    all_pitcher_rows: List[Dict] = []
    db_rows: List[Dict] = []

    if pitcher_artifacts_ok:
        df_p = pd.read_sql(SQL_PITCHER_SNAPSHOTS, conn, params={"game_date": et_date})
        log.info("Pitcher snapshots: %d rows", len(df_p))

        if not df_p.empty:
            X_p = _prep_features(df_p, _PITCHER_META, feat_p, meds_p)
            pred_k = _predict_ensemble(X_p, xgb_k, lgb_k)
            sigma_k = bt.get("ci_strikeouts") if bt else None

            for i, (_, row) in enumerate(df_p.iterrows()):
                pk = max(0.0, float(pred_k[i]))
                name = row.get("player_name", f"id={row['player_id']}")
                norm = _normalize_name(name)
                ld = prop_lines.get((norm, "pitcher_strikeouts"))
                line = ld["line"] if ld else None
                edge = (pk - line) if line is not None else None
                kel = _kelly_prop(abs(edge), sigma_k or cfg.threshold_strikeouts * 2) \
                      if edge is not None else 0.0

                r = {
                    "game_slug": row["game_slug"],
                    "game_date_et": et_date,
                    "player_id": int(row["player_id"]),
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "is_home": row.get("is_home"),
                    "opponent_abbr": row.get("opponent_abbr"),
                    "start_ts_utc": row.get("start_ts_utc"),
                    "pred_strikeouts": pk,
                }
                all_pitcher_rows.append(r)
                db_rows.append({
                    "game_date_et": et_date,
                    "game_slug": row["game_slug"],
                    "player_id": int(row["player_id"]),
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "stat": "pitcher_strikeouts",
                    "pred_value": round(pk, 3),
                    "book_line": line,
                    "edge": round(edge, 3) if edge is not None else None,
                    "kelly_fraction": round(kel, 4),
                })

    # ── Batter predictions ─────────────────────────────────────────────────
    all_batter_rows: List[Dict] = []

    if batter_artifacts_ok:
        prior_season = f"{et_date.year - 1}-regular"
        df_b = pd.read_sql(
            SQL_BATTER_SNAPSHOTS, conn,
            params={
                "game_date": et_date,
                "prior_season": prior_season,
                "min_ab_avg_10": cfg.min_ab_avg_10,
                "min_n_games": cfg.min_n_games,
            }
        )
        log.info("Batter snapshots: %d rows", len(df_b))

        # Fetch player names from boxscore (first + last name) and SP table as fallback
        name_df = pd.read_sql(
            """SELECT player_id,
                  TRIM(first_name || ' ' || last_name) AS player_name
               FROM (
                   SELECT DISTINCT ON (player_id)
                       player_id,
                       first_name,
                       last_name
                   FROM raw.mlb_boxscore_player_stats
                   WHERE first_name IS NOT NULL AND last_name IS NOT NULL
                   ORDER BY player_id, game_slug DESC
               ) sub
               UNION ALL
               SELECT player_id, player_name
               FROM raw.mlb_starting_pitchers
               WHERE player_name IS NOT NULL""",
            conn
        )
        # Deduplicate: keep first occurrence
        name_df = name_df.drop_duplicates(subset=["player_id"], keep="first")
        name_map = dict(zip(name_df["player_id"], name_df["player_name"]))

        if not df_b.empty:
            X_b = _prep_features(df_b, _BATTER_META, feat_b, meds_b)
            pred_h     = _predict_ensemble(X_b, xgb_h,      lgb_h)
            pred_tb    = _predict_ensemble(X_b, xgb_tb_m,   lgb_tb_m)
            pred_hr    = _predict_ensemble(X_b, xgb_hr,     lgb_hr)
            pred_walks = _predict_ensemble(X_b, xgb_walks_m, lgb_walks_m)
            sigma_h     = bt.get("ci_hits")        if bt else None
            sigma_tb    = bt.get("ci_total_bases") if bt else None
            sigma_hr    = bt.get("ci_home_runs")   if bt else None
            sigma_walks = bt.get("ci_walks")       if bt else None

            seen = set()
            for i, (_, row) in enumerate(df_b.iterrows()):
                pid = int(row["player_id"])
                slug = row["game_slug"]
                key = (slug, pid)
                if key in seen:
                    continue
                seen.add(key)

                name = name_map.get(pid, f"id={pid}")
                norm = _normalize_name(name)
                ph    = max(0.0, float(pred_h[i]))
                ptb   = max(0.0, float(pred_tb[i]))
                phr   = max(0.0, float(pred_hr[i]))
                pwalk = max(0.0, float(pred_walks[i]))

                r = {
                    "game_slug": slug,
                    "game_date_et": et_date,
                    "player_id": pid,
                    "player_name": name,
                    "team_abbr": row.get("team_abbr"),
                    "is_home": row.get("is_home"),
                    "opponent_abbr": row.get("opponent_abbr"),
                    "start_ts_utc": row.get("start_ts_utc"),
                    "n_games_prev_10": int(row.get("n_games_prev_10") or 0),
                    "pred_hits": ph,
                    "pred_total_bases": ptb,
                    "pred_home_runs": phr,
                    "pred_walks": pwalk,
                }
                all_batter_rows.append(r)

                for stat_key, stat_label, pred_v, sigma in [
                    ("batter_hits",        "batter_hits",        ph,    sigma_h),
                    ("batter_total_bases", "batter_total_bases", ptb,   sigma_tb),
                    ("batter_home_runs",   "batter_home_runs",   phr,   sigma_hr),
                    ("batter_walks",       "batter_walks",       pwalk, sigma_walks),
                ]:
                    ld = prop_lines.get((norm, stat_key))
                    line = ld["line"] if ld else None
                    edge = (pred_v - line) if line is not None else None
                    kel = _kelly_prop(abs(edge), sigma or 0.5) if edge is not None else 0.0
                    db_rows.append({
                        "game_date_et": et_date,
                        "game_slug": slug,
                        "player_id": pid,
                        "player_name": name,
                        "team_abbr": row.get("team_abbr"),
                        "stat": stat_label,
                        "pred_value": round(pred_v, 3),
                        "book_line": line,
                        "edge": round(edge, 3) if edge is not None else None,
                        "kelly_fraction": round(kel, 4),
                    })

    # ── Save to DB ────────────────────────────────────────────────────────
    _save_predictions(conn, db_rows)
    conn.close()

    if not all_pitcher_rows and not all_batter_rows:
        print(f"_(No prop data for {et_date})_")
        return

    # ── Build game map for grouping ───────────────────────────────────────
    game_map: Dict[str, Dict] = {}
    for row in all_pitcher_rows + all_batter_rows:
        slug = row["game_slug"]
        if slug not in game_map:
            parts = slug.split("-")
            away = parts[1] if len(parts) > 2 else "?"
            home = parts[2] if len(parts) > 2 else "?"
            game_map[slug] = {
                "home": home,
                "away": away,
                "start_ts_utc": row.get("start_ts_utc"),
            }

    # ── Print output ──────────────────────────────────────────────────────
    header = f"⚾ **MLB Props — {et_date.strftime('%b')} {et_date.day}**"
    print(header)
    print("")

    fd_links = _print_discord(
        all_pitcher_rows, all_batter_rows, prop_lines, game_map, cfg
    )
    best_links = _print_best_bets(
        all_pitcher_rows, all_batter_rows, prop_lines, cfg
    )
    fd_links.extend(best_links)

    # Best props parlay (high-edge bets only)
    parlay_url = build_fd_parlay_url(fd_links)
    if parlay_url:
        print(f"\n**Best Props Parlay** [FD]({parlay_url})")

    # All props parlay — every player, model's predicted direction, chunked at 25 legs
    all_links = _collect_all_prop_links(all_pitcher_rows, all_batter_rows, prop_lines)
    # Deduplicate while preserving order
    seen: set[str] = set()
    all_links_dedup = [l for l in all_links if l not in seen and not seen.add(l)]  # type: ignore[func-returns-value]
    n_all = len(all_links_dedup)
    n_chunks = math.ceil(n_all / 25) if n_all else 0
    for i in range(0, n_all, 25):
        chunk = all_links_dedup[i:i + 25]
        ap = build_fd_parlay_url(chunk)
        if ap:
            suffix = f" {i // 25 + 1}/{n_chunks}" if n_chunks > 1 else ""
            print(f"\n**All Props Parlay{suffix}** [FD]({ap})")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    et_date = None
    if args.date:
        from datetime import date as _date
        et_date = _date.fromisoformat(args.date)
    elif os.getenv("MLB_ET_DATE"):
        from datetime import date as _date
        et_date = _date.fromisoformat(os.getenv("MLB_ET_DATE"))

    cfg = PredictConfig(et_date=et_date)
    predict_props(cfg)


if __name__ == "__main__":
    main()
