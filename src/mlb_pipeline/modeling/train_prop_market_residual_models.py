"""Train shadow market-residual models for MLB prop sides.

The residual model learns when the player model is adding useful information
over the no-vig market baseline, and when the market should be trusted instead.
It also writes exact-bucket model-selection recommendations.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .prop_replay import ev_per_unit
from .prop_training_groups import (
    dedupe_locked_offer_rows,
    expanding_player_game_folds,
    grouping_summary,
    sample_weights,
    temporal_player_game_split,
)

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_NUMERIC = [
    "model_prob_side",
    "market_prob_side",
    "prob_edge_vs_market",
    "market_line",
    "abs_price",
    "is_plus_price",
    "count_edge_side",
    "pred_count",
    "pred_value",
    "confirmed_batting_order",
    "projected_pa",
    "projected_bf",
    "projected_pitch_count",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_team_k_pct_10",
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
    "batter_vs_rp_slg_30",
    "batter_vs_rp_hr_rate_30",
    "pinch_hit_risk",
    "same_book_pair_flag",
    "cross_book_pair_flag",
    "synthetic_pair_flag",
    "clean_market_pair_flag",
    "true_pair_flag",
    "minutes_to_first_pitch_at_lock",
    "lock_price_age_minutes",
    "open_price_implied",
    "lock_price_implied",
    "open_to_lock_prob_move",
    "open_to_lock_line_move_side",
    "consensus_prob_at_lock",
    "consensus_price_dispersion",
    "consensus_book_count",
    "book_lead_lag_prob",
    "best_consensus_prob",
    "worst_consensus_prob",
    "lock_offer_available",
    "lock_same_book_pair_available",
]

_CLV_V1_NUMERIC = [
    name for name in _NUMERIC
    if name not in {
        "open_price_implied", "lock_price_implied", "open_to_lock_prob_move",
        "open_to_lock_line_move_side", "consensus_prob_at_lock",
        "consensus_price_dispersion", "consensus_book_count", "book_lead_lag_prob",
        "best_consensus_prob", "worst_consensus_prob", "lock_offer_available",
        "lock_same_book_pair_available",
    }
]

_CLV_V3_EXTRA_NUMERIC = [
    "open_to_lock_minutes",
    "minutes_to_first_pitch_log",
    "lock_price_age_log",
    "open_to_lock_move_per_hour",
    "consensus_range",
    "book_lead_lag_z",
    "model_market_abs_disagreement",
    "model_move_agreement",
    "line_price_move_agreement",
    "recent_book_clv_mean",
    "recent_book_clv_beat_rate",
    "recent_book_clv_rows",
]
_CLV_V3_NUMERIC = [*_NUMERIC, *_CLV_V3_EXTRA_NUMERIC]

_CLV_V4_EXTRA_NUMERIC = [
    "lock_consensus_gap",
    "open_consensus_gap",
    "movement_dispersion_ratio",
    "time_pressure",
    "consensus_book_count_log",
    "availability_pair_interaction",
]
_CLV_V4_NUMERIC = [*_CLV_V3_NUMERIC, *_CLV_V4_EXTRA_NUMERIC]

_CLV_V5_EXTRA_NUMERIC = [
    "price_move_abs",
    "line_move_abs",
    "lead_lag_abs",
    "consensus_depth_score",
    "market_pressure_score",
    "model_edge_times_move",
    "model_edge_times_lead_lag",
    "edge_consensus_gap",
    "book_lagging_market_flag",
    "one_hour_move_pressure",
    "dispersion_x_time_pressure",
    "available_consensus_depth",
    "same_book_consensus_depth",
]
_CLV_V5_NUMERIC = [*_CLV_V4_NUMERIC, *_CLV_V5_EXTRA_NUMERIC]

_CATEGORICAL = [
    "market",
    "side",
    "line_surface",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
    "pair_quality",
    "paired_price_source",
    "market_prob_source",
    "model_family",
    "edge_type",
    "opp_sp_hand",
]

_VARIANTS = {
    "model_only": "model_prob_side",
    "market_no_vig": "market_prob_side",
    "market_residual": "p_market_residual",
}


@dataclass(frozen=True)
class MarketResidualConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_market_residual_models.json"
    report_file: str = "mlb_prop_market_residual_latest.md"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 250
    min_holdout_rows: int = 60
    min_bucket_rows: int = 20
    min_brier_gain: float = 0.001
    min_residual_brier_gain: float = 0.003
    min_residual_selected_rows: int = 20
    min_residual_clv_beat_rate: float = 0.55
    max_residual_calibration_error: float = 0.05
    min_ev: float = 0.02
    walk_forward_test_days: int = 7
    walk_forward_min_train_dates: int = 9


SQL = """
WITH lock_prices AS (
    SELECT
        s.id,
        s.run_id,
        s.as_of_date,
        s.event_id,
        s.player_name_norm,
        s.stat,
        s.selected_side,
        s.line,
        s.bookmaker_key,
        s.snapshot_at_utc,
        CASE WHEN s.selected_side = 'over' THEN s.over_price ELSE s.under_price END::float AS side_price,
        CASE
            WHEN (CASE WHEN s.selected_side = 'over' THEN s.over_price ELSE s.under_price END) > 0
            THEN 100.0 / ((CASE WHEN s.selected_side = 'over' THEN s.over_price ELSE s.under_price END) + 100.0)
            WHEN (CASE WHEN s.selected_side = 'over' THEN s.over_price ELSE s.under_price END) < 0
            THEN -(CASE WHEN s.selected_side = 'over' THEN s.over_price ELSE s.under_price END)
                 / (-(CASE WHEN s.selected_side = 'over' THEN s.over_price ELSE s.under_price END) + 100.0)
            ELSE NULL
        END AS side_implied,
        CASE WHEN s.over_price IS NOT NULL AND s.under_price IS NOT NULL THEN 1.0 ELSE 0.0 END AS same_book_pair_available
    FROM odds.mlb_player_prop_line_snapshots s
    WHERE s.snapshot_role = 'lock'
),
lock_consensus AS (
    SELECT
        run_id, as_of_date, event_id, player_name_norm, stat, selected_side, line,
        AVG(side_implied) AS consensus_prob,
        STDDEV_POP(side_implied) AS price_dispersion,
        COUNT(DISTINCT bookmaker_key)::float AS book_count,
        MIN(side_implied) AS best_prob,
        MAX(side_implied) AS worst_prob
    FROM lock_prices
    WHERE side_implied IS NOT NULL
    GROUP BY run_id, as_of_date, event_id, player_name_norm, stat, selected_side, line
)
SELECT
    e.id,
    e.replay_id,
    e.source_created_at,
    e.prop_offer_id,
    e.lock_snapshot_id,
    e.game_date_et,
    e.game_slug,
    e.player_id,
    e.player_name_norm,
    e.market,
    e.side,
    COALESCE(e.line_surface, 'unknown') AS line_surface,
    COALESCE(e.line_bucket, 'unknown') AS line_bucket,
    COALESCE(e.price_bucket, 'missing_price') AS price_bucket,
    COALESCE(e.bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(e.pair_quality, 'unknown') AS pair_quality,
    COALESCE(e.paired_price_source, 'unknown') AS paired_price_source,
    COALESCE(e.market_prob_source, 'unknown') AS market_prob_source,
    COALESCE(e.same_book_pair_flag::float, CASE WHEN e.pair_quality = 'same_book' THEN 1.0 ELSE 0.0 END) AS same_book_pair_flag,
    COALESCE(e.cross_book_pair_flag::float, CASE WHEN e.pair_quality = 'cross_book' THEN 1.0 ELSE 0.0 END) AS cross_book_pair_flag,
    COALESCE(e.synthetic_pair_flag::float, CASE WHEN e.pair_quality = 'synthetic' THEN 1.0 ELSE 0.0 END) AS synthetic_pair_flag,
    COALESCE(
        e.clean_market_pair_flag::float,
        CASE
            WHEN e.pair_quality IN ('same_book', 'cross_book')
             AND COALESCE(e.market_prob_source, '') NOT IN ('raw_implied', 'synthetic_fanduel_over_only')
            THEN 1.0
            ELSE 0.0
        END
    ) AS clean_market_pair_flag,
    COALESCE(e.true_pair_flag::float, CASE WHEN e.pair_quality IN ('same_book', 'cross_book') THEN 1.0 ELSE 0.0 END) AS true_pair_flag,
    COALESCE(e.model_family, 'unknown') AS model_family,
    COALESCE(e.edge_type, 'unknown') AS edge_type,
    e.minutes_to_first_pitch_at_lock::float AS minutes_to_first_pitch_at_lock,
    e.lock_price_age_minutes::float AS lock_price_age_minutes,
    open_snap.open_implied::float AS open_price_implied,
    lock_row.side_implied::float AS lock_price_implied,
    (lock_row.side_implied - open_snap.open_implied)::float AS open_to_lock_prob_move,
    CASE
        WHEN e.side = 'over' THEN (open_snap.open_line - e.market_line)::float
        WHEN e.side = 'under' THEN (e.market_line - open_snap.open_line)::float
        ELSE NULL
    END AS open_to_lock_line_move_side,
    EXTRACT(EPOCH FROM (lock_row.snapshot_at_utc - open_snap.open_snapshot_at_utc)) / 60.0 AS open_to_lock_minutes,
    consensus.consensus_prob::float AS consensus_prob_at_lock,
    consensus.price_dispersion::float AS consensus_price_dispersion,
    consensus.book_count::float AS consensus_book_count,
    (lock_row.side_implied - consensus.consensus_prob)::float AS book_lead_lag_prob,
    consensus.best_prob::float AS best_consensus_prob,
    consensus.worst_prob::float AS worst_consensus_prob,
    CASE WHEN lock_row.id IS NOT NULL AND lock_row.side_price IS NOT NULL THEN 1.0 ELSE 0.0 END AS lock_offer_available,
    COALESCE(lock_row.same_book_pair_available, 0.0)::float AS lock_same_book_pair_available,
    e.market_line::float AS market_line,
    e.market_price::float AS market_price,
    ABS(e.market_price::float) AS abs_price,
    CASE WHEN e.market_price::float > 0 THEN 1.0 WHEN e.market_price IS NOT NULL THEN 0.0 ELSE NULL END AS is_plus_price,
    e.model_prob_side::float AS model_prob_side,
    e.market_prob_side::float AS market_prob_side,
    e.prob_edge_vs_market::float AS prob_edge_vs_market,
    e.count_edge_side::float AS count_edge_side,
    e.pred_count::float AS pred_count,
    e.pred_value::float AS pred_value,
    e.confirmed_batting_order::float AS confirmed_batting_order,
    e.projected_pa::float AS projected_pa,
    e.projected_bf::float AS projected_bf,
    e.projected_pitch_count::float AS projected_pitch_count,
    e.is_home::float AS is_home,
    e.team_implied_runs::float AS team_implied_runs,
    e.opponent_implied_runs::float AS opponent_implied_runs,
    e.game_total_line::float AS game_total_line,
    e.opp_sp_hand,
    e.opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    e.opp_sp_bb_pct::float AS opp_sp_bb_pct,
    e.opp_sp_xwoba::float AS opp_sp_xwoba,
    e.opp_sp_hard_hit_pct::float AS opp_sp_hard_hit_pct,
    e.opp_sp_whiff_pct::float AS opp_sp_whiff_pct,
    e.opp_bp_era_10::float AS opp_bp_era_10,
    e.opp_bp_whip_10::float AS opp_bp_whip_10,
    e.opp_bp_k9_10::float AS opp_bp_k9_10,
    e.opp_team_k_pct_10::float AS opp_team_k_pct_10,
    e.batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    e.batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    e.batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    e.batter_vs_hand_iso_avg_10::float AS batter_vs_hand_iso_avg_10,
    e.batter_vs_hand_k_rate_10::float AS batter_vs_hand_k_rate_10,
    e.batter_vs_rp_slg_30::float AS batter_vs_rp_slg_30,
    e.batter_vs_rp_hr_rate_30::float AS batter_vs_rp_hr_rate_30,
    e.pinch_hit_risk::float AS pinch_hit_risk,
    CASE WHEN e.won IS TRUE THEN 1 WHEN e.won IS FALSE THEN 0 ELSE NULL END AS target,
    COALESCE(e.push, false) AS push,
    e.profit_units::float AS profit_units,
    e.clv_price::float AS clv_price,
    CASE WHEN e.beat_clv_price IS TRUE THEN 1 WHEN e.beat_clv_price IS FALSE THEN 0 ELSE NULL END AS beat_clv_price
FROM features.mlb_prop_market_training_examples e
JOIN raw.mlb_games g_final
  ON g_final.game_slug = e.game_slug
 AND g_final.status = 'final'
LEFT JOIN lock_prices lock_row ON lock_row.id = e.lock_snapshot_id
LEFT JOIN lock_consensus consensus
  ON consensus.run_id IS NOT DISTINCT FROM lock_row.run_id
 AND consensus.as_of_date = lock_row.as_of_date
 AND consensus.event_id IS NOT DISTINCT FROM lock_row.event_id
 AND consensus.player_name_norm = lock_row.player_name_norm
 AND consensus.stat = lock_row.stat
 AND consensus.selected_side = lock_row.selected_side
 AND consensus.line = lock_row.line
LEFT JOIN LATERAL (
    SELECT
        os.line::float AS open_line,
        os.snapshot_at_utc AS open_snapshot_at_utc,
        CASE
            WHEN e.side = 'over' AND os.over_price > 0 THEN 100.0 / (os.over_price + 100.0)
            WHEN e.side = 'over' AND os.over_price < 0 THEN -os.over_price / (-os.over_price + 100.0)
            WHEN e.side = 'under' AND os.under_price > 0 THEN 100.0 / (os.under_price + 100.0)
            WHEN e.side = 'under' AND os.under_price < 0 THEN -os.under_price / (-os.under_price + 100.0)
            ELSE NULL
        END::float AS open_implied
    FROM odds.mlb_player_prop_line_snapshots os
    WHERE os.snapshot_role = 'open'
      AND os.as_of_date = e.game_date_et
      AND os.event_id IS NOT DISTINCT FROM lock_row.event_id
      AND os.player_name_norm = e.player_name_norm
      AND os.stat = e.market
      AND os.bookmaker_key = e.bookmaker_key
      AND os.snapshot_at_utc <= lock_row.snapshot_at_utc
    ORDER BY
        CASE WHEN os.line = e.market_line THEN 0 ELSE 1 END,
        os.snapshot_at_utc DESC
    LIMIT 1
) open_snap ON TRUE
WHERE e.game_date_et >= %(cutoff)s
  AND e.result_status = 'graded'
  AND e.market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND e.side IN ('over','under')
  AND e.model_prob_side IS NOT NULL
  AND e.market_prob_side IS NOT NULL
  AND e.market_line IS NOT NULL
  AND e.won IS NOT NULL
  AND COALESCE(
        e.clean_market_pair_flag::float,
        CASE
            WHEN e.pair_quality IN ('same_book', 'cross_book')
             AND COALESCE(e.market_prob_source, '') NOT IN ('raw_implied', 'synthetic_fanduel_over_only')
            THEN 1.0
            ELSE 0.0
        END
      ) = 1.0
  AND COALESCE(e.true_pair_flag::float, 0.0) >= 0.5
  AND COALESCE(e.synthetic_pair_flag::float, 0.0) < 0.5
  AND NOT (
        LOWER(COALESCE(e.bookmaker_key, '')) = 'fanduel'
    AND e.market IN ('batter_hits','batter_total_bases','batter_home_runs')
    AND (
          COALESCE(e.pair_quality, '') = 'synthetic'
       OR COALESCE(e.market_prob_source, '') = 'synthetic_fanduel_over_only'
       OR COALESCE(e.paired_price_source, '') = 'synthetic_fanduel_over_only_complement'
       OR COALESCE(e.synthetic_pair_flag::float, 0.0) >= 0.5
    )
  )
ORDER BY e.game_date_et, e.replay_id
"""


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _load(cfg: MarketResidualConfig) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in _CLV_V5_NUMERIC + ["market_price", "target", "profit_units", "clv_price", "beat_clv_price"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df["push"] = df["push"].fillna(False).astype(bool)
    df = df.loc[~df["push"]].dropna(subset=["target", "model_prob_side", "market_prob_side"])
    df["target"] = df["target"].astype(int)
    df = dedupe_locked_offer_rows(df)
    dedupe_attrs = dict(df.attrs)
    for col in _CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)
    out = _add_clv_v3_features(df)
    out.attrs.update(dedupe_attrs)
    return out


def _add_clv_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(
        columns=["recent_book_clv_mean", "recent_book_clv_beat_rate", "recent_book_clv_rows"],
        errors="ignore",
    )
    minutes = pd.to_numeric(out["minutes_to_first_pitch_at_lock"], errors="coerce").clip(lower=0.0)
    age = pd.to_numeric(out["lock_price_age_minutes"], errors="coerce").clip(lower=0.0)
    elapsed = pd.to_numeric(out["open_to_lock_minutes"], errors="coerce").clip(lower=1.0)
    move = pd.to_numeric(out["open_to_lock_prob_move"], errors="coerce")
    line_move = pd.to_numeric(out["open_to_lock_line_move_side"], errors="coerce")
    lead_lag = pd.to_numeric(out["book_lead_lag_prob"], errors="coerce")
    dispersion = pd.to_numeric(out["consensus_price_dispersion"], errors="coerce").abs()
    model_prob = pd.to_numeric(out["model_prob_side"], errors="coerce")
    market_prob = pd.to_numeric(out["market_prob_side"], errors="coerce")
    out["minutes_to_first_pitch_log"] = np.log1p(minutes)
    out["lock_price_age_log"] = np.log1p(age)
    out["open_to_lock_move_per_hour"] = move / (elapsed / 60.0).clip(lower=1.0 / 60.0)
    out["consensus_range"] = (
        pd.to_numeric(out["worst_consensus_prob"], errors="coerce")
        - pd.to_numeric(out["best_consensus_prob"], errors="coerce")
    ).abs()
    out["book_lead_lag_z"] = lead_lag / dispersion.clip(lower=0.005)
    out["model_market_abs_disagreement"] = (model_prob - market_prob).abs()
    out["model_move_agreement"] = np.sign(model_prob - market_prob) * np.sign(move)
    out["line_price_move_agreement"] = np.sign(line_move) * np.sign(move)
    lock_implied = pd.to_numeric(out["lock_price_implied"], errors="coerce")
    open_implied = pd.to_numeric(out["open_price_implied"], errors="coerce")
    consensus = pd.to_numeric(out["consensus_prob_at_lock"], errors="coerce")
    book_count = pd.to_numeric(out["consensus_book_count"], errors="coerce").clip(lower=0.0)
    available = pd.to_numeric(out["lock_offer_available"], errors="coerce").fillna(0.0)
    paired = pd.to_numeric(out["lock_same_book_pair_available"], errors="coerce").fillna(0.0)
    out["lock_consensus_gap"] = lock_implied - consensus
    out["open_consensus_gap"] = open_implied - consensus
    out["movement_dispersion_ratio"] = move / dispersion.clip(lower=0.005)
    out["time_pressure"] = 1.0 / (1.0 + minutes / 60.0)
    out["consensus_book_count_log"] = np.log1p(book_count)
    out["availability_pair_interaction"] = available * paired
    out["clv_abs_prob_points"] = pd.to_numeric(out["clv_price"], errors="coerce").abs()
    model_edge = model_prob - market_prob
    out["price_move_abs"] = move.abs()
    out["line_move_abs"] = line_move.abs()
    out["lead_lag_abs"] = lead_lag.abs()
    out["consensus_depth_score"] = np.log1p(book_count) * (1.0 - dispersion.fillna(0.0).clip(0.0, 0.25))
    out["market_pressure_score"] = move.fillna(0.0).abs() * out["time_pressure"]
    out["model_edge_times_move"] = model_edge * move
    out["model_edge_times_lead_lag"] = model_edge * lead_lag
    out["edge_consensus_gap"] = model_prob - consensus
    out["book_lagging_market_flag"] = (
        np.sign(model_edge.fillna(0.0)) == np.sign((-lead_lag).fillna(0.0))
    ).astype(float)
    out["one_hour_move_pressure"] = out["open_to_lock_move_per_hour"].fillna(0.0) * out["time_pressure"]
    out["dispersion_x_time_pressure"] = dispersion.fillna(0.0) * out["time_pressure"]
    out["available_consensus_depth"] = available * out["consensus_depth_score"]
    out["same_book_consensus_depth"] = paired * out["consensus_depth_score"]

    group_cols = ["bookmaker_key", "market", "side"]
    history = out.loc[out["clv_price"].notna(), [*group_cols, "game_date_et", "clv_price", "beat_clv_price"]].copy()
    if history.empty:
        out["recent_book_clv_mean"] = np.nan
        out["recent_book_clv_beat_rate"] = np.nan
        out["recent_book_clv_rows"] = 0.0
        return out
    history["beat_known"] = history["beat_clv_price"].notna().astype(float)
    history["beat_value"] = history["beat_clv_price"].fillna(0.0)
    daily = history.groupby([*group_cols, "game_date_et"], as_index=False).agg(
        clv_sum=("clv_price", "sum"),
        clv_rows=("clv_price", "count"),
        beat_sum=("beat_value", "sum"),
        beat_rows=("beat_known", "sum"),
    ).sort_values([*group_cols, "game_date_et"])
    for value in ("clv_sum", "clv_rows", "beat_sum", "beat_rows"):
        daily[f"prior_{value}"] = daily.groupby(group_cols, sort=False)[value].cumsum() - daily[value]
    daily["recent_book_clv_mean"] = daily["prior_clv_sum"] / daily["prior_clv_rows"].replace(0.0, np.nan)
    daily["recent_book_clv_beat_rate"] = daily["prior_beat_sum"] / daily["prior_beat_rows"].replace(0.0, np.nan)
    daily["recent_book_clv_rows"] = daily["prior_clv_rows"].astype(float)
    return out.merge(
        daily[[*group_cols, "game_date_et", "recent_book_clv_mean", "recent_book_clv_beat_rate", "recent_book_clv_rows"]],
        on=[*group_cols, "game_date_et"],
        how="left",
    )


def _prepare(
    df: pd.DataFrame,
    *,
    means=None,
    scales=None,
    cats=None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
):
    numeric_features = list(numeric_features or _NUMERIC)
    categorical_features = list(categorical_features or _CATEGORICAL)
    means = dict(means or {})
    scales = dict(scales or {})
    cats = {k: list(v) for k, v in (cats or {}).items()}
    parts = []
    names = []
    out_means = {}
    out_scales = {}
    for name in numeric_features:
        s = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, s.mean() if not s.dropna().empty else 0.0))
        filled = s.fillna(mean)
        scale = float(scales.get(name, filled.std(ddof=0) if float(filled.std(ddof=0) or 0.0) > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy().reshape(-1, 1))
        names.append(name)
        out_means[name] = mean
        out_scales[name] = scale
    out_cats = {}
    for name in categorical_features:
        values = cats.get(name)
        if values is None:
            values = sorted(str(v) for v in df[name].fillna("unknown").unique())
        out_cats[name] = values
        series = df[name].fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))
            names.append(f"{name}={value}")
    return np.hstack(parts) if parts else np.zeros((len(df), 0)), names, out_means, out_scales, out_cats


def _fit_logistic_record(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    method: str,
) -> tuple[dict[str, Any], np.ndarray]:
    X_train, names, means, scales, cats = _prepare(
        train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(
        X_train,
        train[target].astype(int).to_numpy(),
        sample_weight=sample_weights(train),
    )
    X_hold, _, _, _, _ = _prepare(
        holdout,
        means=means,
        scales=scales,
        cats=cats,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    probability = np.clip(model.predict_proba(X_hold)[:, 1], 1e-6, 1.0 - 1e-6)
    return {
        "method": method,
        "intercept": float(model.intercept_[0]),
        "coef": {name: float(value) for name, value in zip(names, model.coef_[0]) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }, probability


def _fit_linear_record(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    method: str,
) -> tuple[dict[str, Any], np.ndarray]:
    X_train, names, means, scales, cats = _prepare(
        train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model = Ridge(alpha=20.0)
    model.fit(X_train, train[target].astype(float).to_numpy(), sample_weight=sample_weights(train))
    X_hold, _, _, _, _ = _prepare(
        holdout,
        means=means,
        scales=scales,
        cats=cats,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    prediction = model.predict(X_hold)
    return {
        "method": method,
        "intercept": float(model.intercept_),
        "coef": {name: float(value) for name, value in zip(names, model.coef_) if abs(float(value)) > 1e-12},
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }, np.asarray(prediction, dtype=float)


def _clv_magnitude_summary(
    holdout: pd.DataFrame,
    prediction: np.ndarray,
    baseline: float,
) -> dict[str, Any]:
    target = pd.to_numeric(holdout["clv_price"], errors="coerce").to_numpy(dtype=float)
    weights = sample_weights(holdout)
    model_mae = float(np.average(np.abs(target - prediction), weights=weights))
    baseline_mae = float(np.average(np.abs(target - baseline), weights=weights))
    sign_accuracy = float(np.average((np.sign(prediction) == np.sign(target)).astype(float), weights=weights))
    positive = (target > 0).astype(int)
    auc = None
    if len(np.unique(positive)) == 2:
        try:
            auc = float(roc_auc_score(positive, prediction, sample_weight=weights))
        except Exception:
            auc = None
    return {
        "rows": int(len(holdout)),
        "player_games": int(grouping_summary(holdout).get("player_games") or 0),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_gain": baseline_mae - model_mae,
        "sign_accuracy": sign_accuracy,
        "positive_clv_auc": auc,
        "actual_mean": float(np.average(target, weights=weights)),
        "predicted_mean": float(np.average(prediction, weights=weights)),
    }


def _fit_conditional_clv_v4(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    direction_probability: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Predict signed CLV from direction probability and conditional size.

    ``clv_price`` is already an implied-probability movement in percentage
    points. Modeling its absolute size separately on positive and negative
    rows avoids asking one ridge model to learn a discontinuous signed target.
    """
    train = train.dropna(subset=["clv_price", "clv_abs_prob_points"]).copy()
    holdout = holdout.dropna(subset=["clv_price"]).copy()
    positive = train.loc[train["clv_price"] > 0].copy()
    non_positive = train.loc[train["clv_price"] <= 0].copy()
    minimum = 40
    if len(positive) < minimum or len(non_positive) < minimum or holdout.empty:
        return {
            "method": "conditional_probability_space_clv_v4",
            "enabled": False,
            "reason": "insufficient_direction_rows",
            "positive_rows": int(len(positive)),
            "non_positive_rows": int(len(non_positive)),
        }, np.full(len(holdout), np.nan)

    positive_record, positive_size = _fit_linear_record(
        positive, holdout, "clv_abs_prob_points", _CLV_V4_NUMERIC, _CATEGORICAL,
        "positive_clv_abs_ridge_v4",
    )
    negative_record, negative_size = _fit_linear_record(
        non_positive, holdout, "clv_abs_prob_points", _CLV_V4_NUMERIC, _CATEGORICAL,
        "non_positive_clv_abs_ridge_v4",
    )
    positive_size = np.clip(positive_size, 0.0, 25.0)
    negative_size = np.clip(negative_size, 0.0, 25.0)
    direction = np.clip(np.asarray(direction_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    expected = direction * positive_size - (1.0 - direction) * negative_size
    baseline_direction = float(np.average(
        (train["clv_price"] > 0).astype(float), weights=sample_weights(train)
    ))
    positive_mean = float(np.average(
        positive["clv_abs_prob_points"], weights=sample_weights(positive)
    ))
    negative_mean = float(np.average(
        non_positive["clv_abs_prob_points"], weights=sample_weights(non_positive)
    ))
    baseline = baseline_direction * positive_mean - (1.0 - baseline_direction) * negative_mean
    metrics = _clv_magnitude_summary(holdout, expected, baseline)
    enabled = bool(
        (metrics.get("mae_gain") or 0.0) > 0.0
        and (metrics.get("sign_accuracy") or 0.0) >= 0.50
    )
    return {
        "method": "conditional_probability_space_clv_v4",
        "enabled": enabled,
        "reason": "holdout_mae_and_direction_improved" if enabled else "holdout_gate_failed",
        "probability_space_target": "close_implied_probability_minus_lock_implied_probability_pp",
        "positive_rows": int(len(positive)),
        "non_positive_rows": int(len(non_positive)),
        "positive_magnitude": positive_record,
        "non_positive_magnitude": negative_record,
        "baseline_direction_probability": baseline_direction,
        "baseline_positive_magnitude": positive_mean,
        "baseline_non_positive_magnitude": negative_mean,
        "holdout": metrics,
    }, expected


def _expanding_residual_oof(
    df: pd.DataFrame,
    cfg: MarketResidualConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    folds = expanding_player_game_folds(
        df,
        test_window_days=cfg.walk_forward_test_days,
        min_train_dates=cfg.walk_forward_min_train_dates,
        min_train_rows=cfg.min_train_rows,
        min_holdout_rows=cfg.min_holdout_rows,
    )
    outputs: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for fold in folds:
        holdout = fold.holdout.copy()
        global_record, global_probability = _fit_logistic_record(
            fold.train, holdout, "target", _NUMERIC, _CATEGORICAL, "market_residual_logistic"
        )
        holdout["p_market_residual"] = global_probability
        clv_train = fold.train.dropna(subset=["beat_clv_price", "clv_price"]).copy()
        clv_holdout = holdout.dropna(subset=["beat_clv_price", "clv_price"]).copy()
        clv_rows = 0
        if (
            len(clv_train) >= max(120, cfg.min_train_rows // 2)
            and len(clv_holdout) >= 25
            and clv_train["beat_clv_price"].nunique() == 2
        ):
            for name, features in (
                ("p_clv_v1", _CLV_V1_NUMERIC),
                ("p_clv_v2", _NUMERIC),
                ("p_clv_v3", _CLV_V3_NUMERIC),
                ("p_clv_v4", _CLV_V4_NUMERIC),
                ("p_clv_v5", _CLV_V5_NUMERIC),
            ):
                _, probability = _fit_logistic_record(
                    clv_train, clv_holdout, "beat_clv_price", features, _CATEGORICAL, name
                )
                holdout.loc[clv_holdout.index, name] = probability
            _, expected = _fit_conditional_clv_v4(
                clv_train,
                clv_holdout,
                holdout.loc[clv_holdout.index, "p_clv_v4"].to_numpy(dtype=float),
            )
            holdout.loc[clv_holdout.index, "expected_clv_v4"] = expected
            clv_rows = int(len(clv_holdout))
        holdout["walk_forward_fold"] = fold.fold_index
        outputs.append(holdout)
        summaries.append({
            "fold": fold.fold_index,
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "holdout_start": str(fold.holdout_start),
            "holdout_end": str(fold.holdout_end),
            "train_rows": int(len(fold.train)),
            "holdout_rows": int(len(fold.holdout)),
            "clv_holdout_rows": clv_rows,
            "purged_rows": int(fold.purged_rows),
            "global_feature_count": len(global_record.get("coef") or {}),
        })
    return (pd.concat(outputs, ignore_index=False), summaries) if outputs else (pd.DataFrame(), summaries)


def _clv_history_state(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    state: dict[str, dict[str, float]] = {}
    work = df.dropna(subset=["clv_price"]).copy()
    for key, group in work.groupby(["bookmaker_key", "market", "side"], dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        beats = pd.to_numeric(group["beat_clv_price"], errors="coerce").dropna()
        state["|".join(str(value) for value in key)] = {
            "recent_book_clv_mean": float(pd.to_numeric(group["clv_price"], errors="coerce").mean()),
            "recent_book_clv_beat_rate": float(beats.mean()) if not beats.empty else 0.5,
            "recent_book_clv_rows": float(len(group)),
        }
    return state


def _split(df: pd.DataFrame, cfg: MarketResidualConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    split = temporal_player_game_split(
        df,
        holdout_days=cfg.holdout_days,
        min_train_rows=cfg.min_train_rows,
        min_holdout_rows=cfg.min_holdout_rows,
    )
    return split.train, split.holdout, split.strategy


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _weighted_mean(df: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(df.get(column), errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None
    weights = sample_weights(df.loc[valid])
    return float(np.average(values.loc[valid], weights=weights))


def _ev(prob: Any, price: Any) -> float | None:
    try:
        return ev_per_unit(float(prob), price)
    except Exception:
        return None


def _forecast_summary(df: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    work = df.dropna(subset=[prob_col, "target"])
    if work.empty:
        return {"rows": 0}
    p = work[prob_col].astype(float).clip(1e-6, 1 - 1e-6)
    y = work["target"].astype(int)
    weights = sample_weights(work)
    actual_rate = float(np.average(y, weights=weights))
    avg_prob = float(np.average(p, weights=weights))
    return {
        "rows": int(len(work)),
        "player_games": int(work["player_game_group"].nunique()),
        "actual_rate": actual_rate,
        "avg_prob": avg_prob,
        "calibration_error": actual_rate - avg_prob,
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weights)) if y.nunique() == 2 else None,
        "auc": float(roc_auc_score(y, p, sample_weight=weights)) if y.nunique() == 2 else None,
    }


def _binary_forecast_summary(df: pd.DataFrame, prob_col: str, target_col: str) -> dict[str, Any]:
    work = df.dropna(subset=[prob_col, target_col])
    if work.empty:
        return {"rows": 0}
    p = work[prob_col].astype(float).clip(1e-6, 1 - 1e-6)
    y = work[target_col].astype(int)
    weights = sample_weights(work)
    actual_rate = float(np.average(y, weights=weights))
    avg_prob = float(np.average(p, weights=weights))
    return {
        "rows": int(len(work)),
        "player_games": int(work["player_game_group"].nunique()),
        "actual_rate": actual_rate,
        "avg_prob": avg_prob,
        "calibration_error": actual_rate - avg_prob,
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=weights)) if y.nunique() == 2 else None,
        "auc": float(roc_auc_score(y, p, sample_weight=weights)) if y.nunique() == 2 else None,
    }


def _selection_summary(df: pd.DataFrame, prob_col: str, cfg: MarketResidualConfig) -> dict[str, Any]:
    work = df.copy()
    work["variant_ev"] = [_ev(prob, price) for prob, price in zip(work[prob_col], work["market_price"])]
    selected = work.loc[pd.to_numeric(work["variant_ev"], errors="coerce") >= cfg.min_ev]
    if selected.empty:
        return {"selected_rows": 0, "roi": None, "win_rate": None, "avg_ev": None, "clv_beat_rate": None, "avg_clv_price": None}
    clv = selected.dropna(subset=["beat_clv_price"])
    return {
        "selected_rows": int(len(selected)),
        "selected_player_games": int(selected["player_game_group"].nunique()),
        "roi": _weighted_mean(selected, "profit_units"),
        "win_rate": _weighted_mean(selected, "target"),
        "avg_ev": _weighted_mean(selected, "variant_ev"),
        "clv_beat_rate": _weighted_mean(clv, "beat_clv_price") if not clv.empty else None,
        "avg_clv_price": _weighted_mean(clv, "clv_price") if not clv.empty else None,
    }


def _variant_summary(df: pd.DataFrame, cfg: MarketResidualConfig) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "forecast": _forecast_summary(df, col),
            "selection": _selection_summary(df, col, cfg),
        }
        for name, col in _VARIANTS.items()
        if col in df.columns
    }


def _best_bucket_variant(group: pd.DataFrame, cfg: MarketResidualConfig) -> dict[str, Any]:
    variants = _variant_summary(group, cfg)
    model_brier = variants.get("model_only", {}).get("forecast", {}).get("brier")
    best_name = None
    best_brier = None
    for name, rec in variants.items():
        brier = rec.get("forecast", {}).get("brier")
        if brier is None:
            continue
        if best_brier is None or brier < best_brier:
            best_brier = brier
            best_name = name
    roi = _weighted_mean(group, "profit_units")
    best_selection = variants.get(best_name or "", {}).get("selection", {})
    best_selected_roi = best_selection.get("roi")
    best_selected_clv_beat = best_selection.get("clv_beat_rate")
    best_selected_avg_clv = best_selection.get("avg_clv_price")
    residual_forecast = variants.get("market_residual", {}).get("forecast", {})
    residual_selection = variants.get("market_residual", {}).get("selection", {})
    residual_selected_rows = int(residual_selection.get("selected_rows") or 0)
    residual_roi = residual_selection.get("roi")
    residual_clv_beat = residual_selection.get("clv_beat_rate")
    residual_avg_clv = residual_selection.get("avg_clv_price")
    residual_calibration_error = residual_forecast.get("calibration_error")
    residual_brier = residual_forecast.get("brier")
    residual_brier_gain = (
        model_brier - residual_brier
        if model_brier is not None and residual_brier is not None
        else None
    )
    residual_proof_blockers: list[str] = []
    if residual_selected_rows < cfg.min_residual_selected_rows:
        residual_proof_blockers.append("residual_selected_sample")
    if residual_brier_gain is None or residual_brier_gain < cfg.min_residual_brier_gain:
        residual_proof_blockers.append("residual_brier_gain")
    if residual_roi is None or residual_roi <= 0:
        residual_proof_blockers.append("residual_selected_roi")
    if residual_clv_beat is None or residual_clv_beat < cfg.min_residual_clv_beat_rate:
        residual_proof_blockers.append("residual_clv_beat")
    if residual_avg_clv is None or residual_avg_clv < 0:
        residual_proof_blockers.append("residual_avg_clv")
    if residual_calibration_error is None or abs(float(residual_calibration_error)) > cfg.max_residual_calibration_error:
        residual_proof_blockers.append("residual_calibration")
    if len(group) < cfg.min_bucket_rows:
        decision = "no_bet_sample"
    elif roi is not None and roi <= 0:
        decision = "no_bet_negative_roi"
    elif best_selected_roi is not None and best_selected_roi <= 0:
        decision = "no_bet_selected_negative_roi"
    elif best_selected_clv_beat is not None and best_selected_clv_beat < 0.50:
        decision = "no_bet_bad_clv"
    elif best_selected_avg_clv is not None and best_selected_avg_clv < 0:
        decision = "no_bet_negative_clv"
    elif best_name == "market_residual" and not residual_proof_blockers:
        decision = "use_market_residual"
    elif best_name == "market_residual":
        decision = "no_bet_residual_unproven"
    elif best_name == "market_no_vig" and model_brier is not None and best_brier is not None and (model_brier - best_brier) >= cfg.min_brier_gain:
        decision = "use_market_baseline"
    elif best_name == "model_only":
        decision = "keep_model_only"
    else:
        decision = "no_bet_no_edge"
    return {
        "rows": int(len(group)),
        "decision": decision,
        "roi": roi,
        "best_variant": best_name,
        "model_brier": model_brier,
        "market_brier": variants.get("market_no_vig", {}).get("forecast", {}).get("brier"),
        "residual_brier": residual_brier,
        "residual_brier_gain": residual_brier_gain,
        "residual_selected_rows": residual_selected_rows,
        "residual_calibration_error": residual_calibration_error,
        "residual_proof_blockers": residual_proof_blockers,
        "best_roi_selected": best_selected_roi,
        "best_clv_beat_selected": best_selected_clv_beat,
        "best_avg_clv_selected": best_selected_avg_clv,
        "model_roi_selected": variants.get("model_only", {}).get("selection", {}).get("roi"),
        "residual_roi_selected": residual_roi,
        "residual_clv_beat": residual_clv_beat,
        "residual_avg_clv": residual_avg_clv,
    }


def _bucket_recommendations(df: pd.DataFrame, cfg: MarketResidualConfig) -> list[dict[str, Any]]:
    group_cols = ["market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"]
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({"bucket": "|".join(str(v) for v in key), **_best_bucket_variant(group, cfg)})
    rows.sort(key=lambda rec: (rec["decision"].startswith("no_bet"), -(rec.get("rows") or 0)))
    return rows


def _fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _write_text_with_lock_fallback(path: Path, text: str) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _write_report(payload: dict[str, Any], cfg: MarketResidualConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Prop Market Residual Models",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Rows: {payload.get('rows', 0)}",
        f"Raw rows before locked-offer dedupe: {payload.get('raw_rows', payload.get('rows', 0))}",
        f"Collapsed duplicate locked-offer rows: {payload.get('deduped_rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Status: {payload.get('status')}",
        "",
        "## Expanding Walk-Forward OOF Variants",
        "",
        "| Variant | Rows | Brier | Cal Err | Selected | ROI | CLV Beat |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in (payload.get("holdout_variants") or {}).items():
        f = rec.get("forecast") or {}
        s = rec.get("selection") or {}
        lines.append(
            f"| {name} | {f.get('rows', 0)} | {_fmt_num(f.get('brier'))} | {_fmt_pct(f.get('calibration_error'))} | "
            f"{s.get('selected_rows', 0)} | {_fmt_pct(s.get('roi'))} | {_fmt_pct(s.get('clv_beat_rate'))} |"
        )
    clv_target = payload.get("clv_target") or {}
    if clv_target:
        f = clv_target.get("holdout") or {}
        lines.extend([
            "",
            "## CLV Target",
            "",
            "| Rows | Beat Rate | Avg Prob | Brier | Cal Err | AUC | Status |",
            "|---:|---:|---:|---:|---:|---:|---|",
            (
                f"| {f.get('rows', 0)} | {_fmt_pct(f.get('actual_rate'))} | {_fmt_pct(f.get('avg_prob'))} | "
                f"{_fmt_num(f.get('brier'))} | {_fmt_pct(f.get('calibration_error'))} | "
                f"{_fmt_num(f.get('auc'))} | {clv_target.get('status')} |"
            ),
        ])
        comparisons = clv_target.get("variant_comparison") or {}
        if comparisons:
            lines.extend([
                "",
                "| CLV Variant | Rows | Brier | AUC | Cal Err | Selected |",
                "|---|---:|---:|---:|---:|---|",
            ])
            for variant, rec in comparisons.items():
                lines.append(
                    f"| {variant} | {rec.get('rows', 0)} | {_fmt_num(rec.get('brier'))} | "
                    f"{_fmt_num(rec.get('auc'))} | {_fmt_pct(rec.get('calibration_error'))} | "
                    f"{variant == clv_target.get('selected_variant')} |"
                )
            coverage = clv_target.get("feature_coverage") or {}
            lines.extend([
                "",
                "CLV direction variants are evaluated with expanding walk-forward folds and trained only from true, non-synthetic paired offers.",
                f"Open-to-lock coverage: {_fmt_pct(coverage.get('open_to_lock_prob_move'))}; "
                f"consensus coverage: {_fmt_pct(coverage.get('consensus_prob_at_lock'))}; "
                f"true multi-book consensus: {_fmt_pct(coverage.get('consensus_multibook_rate'))}.",
            ])
        magnitude = clv_target.get("magnitude") or {}
        if magnitude:
            lines.extend([
                "",
                "## Expected CLV Magnitude v4",
                "",
                f"- Enabled: {clv_target.get('magnitude_enabled', False)}",
                f"- Holdout rows: {magnitude.get('rows', 0)}",
                f"- Model MAE / baseline MAE: {_fmt_num(magnitude.get('model_mae'))} / {_fmt_num(magnitude.get('baseline_mae'))}",
                f"- MAE gain: {_fmt_num(magnitude.get('mae_gain'))}",
                f"- Sign accuracy: {_fmt_pct(magnitude.get('sign_accuracy'))}",
                f"- Positive-CLV AUC: {_fmt_num(magnitude.get('positive_clv_auc'))}",
                "- Target: implied-probability movement; positive and non-positive movement sizes are modeled separately.",
            ])
    lines.extend([
        "",
        "## Exact Bucket Model Selection",
        "",
        "| Bucket | Rows | Decision | Best | ROI | Best Sel ROI | Best Sel CLV | Model Brier | Market Brier | Residual Brier | Residual Gain | Residual Sel | Residual ROI | Residual CLV | Residual Avg CLV | Residual Cal | Proof Blockers |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for rec in payload.get("bucket_recommendations", [])[:60]:
        lines.append(
            f"| {rec['bucket']} | {rec['rows']} | {rec['decision']} | {rec.get('best_variant') or '-'} | "
            f"{_fmt_pct(rec.get('roi'))} | {_fmt_pct(rec.get('best_roi_selected'))} | "
            f"{_fmt_pct(rec.get('best_clv_beat_selected'))} | "
            f"{_fmt_num(rec.get('model_brier'))} | {_fmt_num(rec.get('market_brier'))} | "
            f"{_fmt_num(rec.get('residual_brier'))} | {_fmt_num(rec.get('residual_brier_gain'))} | "
            f"{rec.get('residual_selected_rows', 0)} | {_fmt_pct(rec.get('residual_roi_selected'))} | "
            f"{_fmt_pct(rec.get('residual_clv_beat'))} | {_fmt_num(rec.get('residual_avg_clv'), 2)} | "
            f"{_fmt_pct(rec.get('residual_calibration_error'))} | "
            f"{'; '.join(rec.get('residual_proof_blockers') or [])} |"
        )
    return str(_write_text_with_lock_fallback(path, "\n".join(lines) + "\n"))


def train(cfg: MarketResidualConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "usage": "shadow_only",
        "raw_rows": int(df.attrs.get("raw_rows", len(df))) if not df.empty else 0,
        "deduped_rows": int(df.attrs.get("deduped_rows", 0)) if not df.empty else 0,
        "rows": int(len(df)),
        "date_min": str(min(df["game_date_et"])) if not df.empty else None,
        "date_max": str(max(df["game_date_et"])) if not df.empty else None,
        "models": {},
        "grouped_player_game_training": grouping_summary(df),
    }
    if df.empty or df["target"].nunique() < 2:
        payload["status"] = "insufficient_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    train_df, holdout_df, split = _split(df, cfg)
    payload["split_strategy"] = split
    payload["train_rows"] = int(len(train_df))
    payload["holdout_rows"] = int(len(holdout_df))
    payload["train_grouping"] = grouping_summary(train_df)
    payload["holdout_grouping"] = grouping_summary(holdout_df)
    if len(train_df) < cfg.min_train_rows or len(holdout_df) < cfg.min_holdout_rows or train_df["target"].nunique() < 2:
        payload["status"] = "insufficient_rows"
        payload["report_path"] = _write_report(payload, cfg)
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    evaluation, fold_summaries = _expanding_residual_oof(df, cfg)
    if evaluation.empty:
        evaluation = holdout_df.copy()
        _, probability = _fit_logistic_record(
            train_df, evaluation, "target", _NUMERIC, _CATEGORICAL, "market_residual_logistic"
        )
        evaluation["p_market_residual"] = probability
    payload["split_strategy"] = (
        f"expanding_{cfg.walk_forward_test_days}_day_player_game_purged"
        if fold_summaries else split
    )
    payload["walk_forward"] = {
        "method": "expanding_window",
        "test_window_days": cfg.walk_forward_test_days,
        "fold_count": int(len(fold_summaries)),
        "oof_rows": int(len(evaluation)),
        "folds": fold_summaries,
        "fallback_used": not bool(fold_summaries),
    }
    payload["holdout_rows"] = int(len(evaluation))
    payload["holdout_grouping"] = grouping_summary(evaluation)
    payload["holdout_variants"] = _variant_summary(evaluation, cfg)
    payload["bucket_recommendations"] = _bucket_recommendations(evaluation, cfg)
    production_global, _ = _fit_logistic_record(
        df, df.iloc[:1], "target", _NUMERIC, _CATEGORICAL, "market_residual_logistic"
    )
    production_global["trained_on_all_settled_rows"] = True
    production_global["selected_by"] = "expanding_walk_forward_brier"
    payload["models"]["global"] = production_global

    clv_train = df.dropna(subset=["beat_clv_price", "clv_price"]).copy()
    clv_holdout = evaluation.dropna(subset=["beat_clv_price", "clv_price"]).copy()
    payload["clv_target"] = {
        "target": "beat_clv_price",
        "train_rows": int(len(clv_train)),
        "holdout_rows": int(len(clv_holdout)),
        "status": "insufficient_rows",
    }
    if (
        len(clv_train) >= cfg.min_train_rows
        and len(clv_holdout) >= max(30, cfg.min_holdout_rows // 2)
        and clv_train["beat_clv_price"].nunique() == 2
    ):
        clv_holdout = clv_holdout.copy()
        feature_sets = {
            "clv_v1": ("p_clv_v1", _CLV_V1_NUMERIC, "clv_beat_logistic_v1"),
            "clv_v2": ("p_clv_v2", _NUMERIC, "clv_beat_logistic_v2"),
            "clv_v3": ("p_clv_v3", _CLV_V3_NUMERIC, "clv_beat_logistic_v3"),
            "clv_v4": ("p_clv_v4", _CLV_V4_NUMERIC, "clv_direction_logistic_v4"),
            "clv_v5": ("p_clv_v5", _CLV_V5_NUMERIC, "clv_direction_logistic_v5"),
        }
        if not fold_summaries:
            fallback_train = train_df.dropna(subset=["beat_clv_price", "clv_price"]).copy()
            for probability_col, features, method in feature_sets.values():
                _, probability = _fit_logistic_record(
                    fallback_train, clv_holdout, "beat_clv_price", features, _CATEGORICAL, method
                )
                clv_holdout[probability_col] = probability
        for probability_col, features, method in feature_sets.values():
            if probability_col not in clv_holdout.columns or clv_holdout[probability_col].isna().all():
                _, probability = _fit_logistic_record(
                    clv_train, clv_holdout, "beat_clv_price", features, _CATEGORICAL, method
                )
                clv_holdout[probability_col] = probability
        if "expected_clv_v4" not in clv_holdout.columns or clv_holdout["expected_clv_v4"].isna().all():
            _, expected = _fit_conditional_clv_v4(
                clv_train,
                clv_holdout,
                clv_holdout["p_clv_v4"].to_numpy(dtype=float),
            )
            clv_holdout["expected_clv_v4"] = expected
        v1_metrics = _binary_forecast_summary(clv_holdout, "p_clv_v1", "beat_clv_price")
        v2_metrics = _binary_forecast_summary(clv_holdout, "p_clv_v2", "beat_clv_price")
        v3_metrics = _binary_forecast_summary(clv_holdout, "p_clv_v3", "beat_clv_price")
        v4_metrics = _binary_forecast_summary(clv_holdout, "p_clv_v4", "beat_clv_price")
        v5_metrics = _binary_forecast_summary(clv_holdout, "p_clv_v5", "beat_clv_price")
        candidates = {
            "clv_v1": v1_metrics,
            "clv_v2": v2_metrics,
            "clv_v3": v3_metrics,
            "clv_v4": v4_metrics,
            "clv_v5": v5_metrics,
        }
        best_auc = max(
            (float(rec["auc"]) for rec in candidates.values() if rec.get("auc") is not None),
            default=None,
        )
        eligible = {
            name: rec for name, rec in candidates.items()
            if rec.get("brier") is not None
            and (best_auc is None or rec.get("auc") is None or float(rec["auc"]) >= best_auc - 0.01)
        }
        selected_variant, _selected_metrics = min(
            eligible.items() if eligible else candidates.items(),
            key=lambda item: float(item[1].get("brier") or 1.0),
        )
        selected_probability_col, selected_features, selected_method = feature_sets[selected_variant]
        clv_holdout["p_clv_beat"] = clv_holdout[selected_probability_col]
        selected_record, _ = _fit_logistic_record(
            clv_train, clv_train.iloc[:1], "beat_clv_price", selected_features, _CATEGORICAL, selected_method
        )
        coverage = {
            feature: float(clv_train[feature].notna().mean()) if feature in clv_train else 0.0
            for feature in [
                "open_to_lock_prob_move", "open_to_lock_line_move_side", "consensus_prob_at_lock",
                "consensus_price_dispersion", "book_lead_lag_prob", "lock_offer_available",
                "open_to_lock_minutes", "open_to_lock_move_per_hour", "book_lead_lag_z",
                "recent_book_clv_mean", "recent_book_clv_beat_rate",
                "price_move_abs", "lead_lag_abs", "consensus_depth_score",
                "market_pressure_score", "model_edge_times_move",
            ]
        }
        coverage["consensus_multibook_rate"] = float(
            (pd.to_numeric(clv_train["consensus_book_count"], errors="coerce") >= 2).mean()
        )
        coverage["consensus_nonzero_dispersion_rate"] = float(
            (pd.to_numeric(clv_train["consensus_price_dispersion"], errors="coerce") > 0).mean()
        )
        payload["clv_target"] = {
            "target": "beat_clv_price",
            "train_rows": int(len(clv_train)),
            "holdout_rows": int(len(clv_holdout)),
            "status": "ready",
            "evidence": "expanding_walk_forward_true_pair_non_synthetic_only",
            "selected_variant": selected_variant,
            "variant_comparison": candidates,
            "feature_coverage": coverage,
            "holdout": _binary_forecast_summary(clv_holdout, "p_clv_beat", "beat_clv_price"),
        }
        selected_record["selected_by_holdout"] = selected_variant
        selected_record["selected_by"] = "expanding_walk_forward_brier_with_auc_guard"
        selected_record["trained_on_all_settled_rows"] = True
        selected_record["true_pair_only"] = True
        payload["models"]["clv_beat"] = selected_record
        magnitude_holdout = clv_holdout.dropna(subset=["expected_clv_v4"]).copy()
        if len(magnitude_holdout) >= max(30, cfg.min_holdout_rows // 2):
            baseline_clv = _weighted_mean(clv_train, "clv_price") or 0.0
            magnitude_metrics = _clv_magnitude_summary(
                magnitude_holdout,
                magnitude_holdout["expected_clv_v4"].to_numpy(dtype=float),
                baseline_clv,
            )
            production_magnitude, _ = _fit_conditional_clv_v4(
                clv_train,
                clv_train.iloc[:1],
                np.asarray([float(clv_train["beat_clv_price"].mean())]),
            )
            magnitude_enabled = bool(
                (magnitude_metrics.get("mae_gain") or 0.0) > 0.0
                and (magnitude_metrics.get("sign_accuracy") or 0.0) >= 0.50
            )
            production_magnitude.update({
                "enabled": magnitude_enabled,
                "reason": "walk_forward_gate_passed" if magnitude_enabled else "walk_forward_gate_failed",
                "true_pair_only": True,
                "trained_on_all_settled_rows": True,
                "holdout": magnitude_metrics,
            })
            magnitude_record = production_magnitude
            payload["models"]["clv_magnitude"] = magnitude_record
            payload["clv_target"]["magnitude"] = magnitude_metrics
            payload["clv_target"]["magnitude_enabled"] = magnitude_enabled
        payload["clv_history_state"] = _clv_history_state(df)
    payload["status"] = "ready"
    payload["report_path"] = _write_report(payload, cfg)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop market-residual shadow models")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    payload = train(MarketResidualConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
