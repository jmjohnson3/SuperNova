"""Build prop bucket reopen policy from side-level market training examples.

This does not make live bets by itself.  It writes an auditable artifact that
marks which market/side/line/price buckets have enough graded evidence to be
considered for bankroll reopening.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2

from .prop_clean_slate import CleanSlateThresholds, clean_date_set, load_clean_slate_rows
from .prop_market_history import ensure_prop_market_history_schema
from .prop_market_training import ensure_prop_market_training_schema
from .prop_real_money_eligibility import (
    PROP_REAL_MONEY_ELIGIBILITY_START_DATE,
    parse_eligibility_start_date,
)
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_MARKETS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")


@dataclass(frozen=True)
class PropBucketReopenConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_bucket_reopen_policy.json"
    lookback_days: int = 365
    holdout_days: int = 28
    eligibility_start_date: date = PROP_REAL_MONEY_ELIGIBILITY_START_DATE
    min_total_rows: int = 150
    min_train_rows: int = 150
    min_holdout_rows: int = 40
    min_priced_rate: float = 0.95
    min_train_roi: float = -0.02
    min_holdout_roi: float = 0.0
    min_avg_ev: float = 0.0
    min_avg_clv_price: float = 0.0
    min_clv_beat_rate: float = 0.55
    min_clv_price_rows: int = 30
    max_abs_calibration_error: float = 0.05
    min_unique_players: int = 20
    min_unique_teams: int = 6
    min_unique_dates: int = 5
    min_clean_unique_dates: int = 5
    clean_slate_min_side_locks: int = 100
    clean_slate_min_valid_side_locks: int = 100
    clean_slate_min_valid_coverage: float = 0.90
    clean_slate_max_missing_lock_rate: float = 0.02
    clean_slate_max_stale_close_rate: float = 0.02
    max_player_share: float = 0.12
    max_team_share: float = 0.25
    max_game_date_share: float = 0.35
    starter_min_total_rows: int = 250
    starter_min_holdout_rows: int = 60
    starter_min_clv_price_rows: int = 50
    starter_min_unique_dates: int = 8
    starter_min_clean_unique_dates: int = 8
    starter_max_abs_calibration_error: float = 0.04
    bankroll_min_total_rows: int = 500
    bankroll_min_holdout_rows: int = 100
    bankroll_min_clv_price_rows: int = 100
    bankroll_min_unique_dates: int = 12
    bankroll_min_clean_unique_dates: int = 12
    bankroll_max_abs_calibration_error: float = 0.03
    history_lookback_days: int = 540
    min_history_rows: int = 500
    min_history_priced_rate: float = 0.80
    min_history_roi: float = -0.15
    max_history_abs_market_calibration_error: float = 0.12
    enforce_history_guard: bool = False
    force_reopen_all: bool = False
    research_only: bool = False
    enable_bootstrap_micro: bool = True
    bootstrap_common_only: bool = True
    bootstrap_min_rows: int = 150
    bootstrap_min_priced_rate: float = 0.95
    bootstrap_min_roi: float = 0.0
    bootstrap_min_avg_ev: float = 0.0
    bootstrap_min_avg_clv_price: float = 0.0
    bootstrap_min_clv_beat_rate: float = 0.55
    bootstrap_min_clv_price_rows: int = 30
    bootstrap_max_abs_calibration_error: float = 0.05
    bootstrap_min_unique_players: int = 20
    bootstrap_min_unique_teams: int = 6
    bootstrap_min_unique_dates: int = 5
    bootstrap_min_clean_unique_dates: int = 5
    bootstrap_max_player_share: float = 0.12
    bootstrap_max_team_share: float = 0.25
    bootstrap_max_game_date_share: float = 0.35


SQL = """
SELECT
    game_date_et,
    game_slug,
    player_id,
    team_abbr,
    market,
    side,
    COALESCE(bookmaker_key, '*') AS bookmaker_key,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(line_surface, 'unknown') AS line_surface,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    COALESCE(model_family, 'unknown') AS model_family,
    market_line::float AS market_line,
    market_price::float AS market_price,
    model_prob_side::float AS model_prob_side,
    market_prob_side::float AS market_prob_side,
    prob_edge_vs_market::float AS prob_edge_vs_market,
    ev::float AS ev,
    kelly_fraction::float AS kelly_fraction,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS won,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units,
    CASE WHEN clv_valid IS TRUE THEN clv_price::float ELSE NULL END AS clv_price,
    CASE
        WHEN clv_valid IS TRUE AND beat_clv_price IS TRUE THEN 1
        WHEN clv_valid IS TRUE AND beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price,
    CASE WHEN clv_valid IS TRUE THEN clv_line::float ELSE NULL END AS clv_line,
    CASE
        WHEN clv_valid IS TRUE AND beat_clv_line IS TRUE THEN 1
        WHEN clv_valid IS TRUE AND beat_clv_line IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_line,
    COALESCE(clv_valid, false) AS clv_valid,
    clv_unknown_reason
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = ANY(%(markets)s)
  AND model_prob_side IS NOT NULL
  AND market_prob_side IS NOT NULL
  AND market_line IS NOT NULL
  AND won IS NOT NULL
  AND model_prob_side BETWEEN 0.0 AND 1.0
  AND COALESCE(
        clean_market_pair_flag::float,
        CASE
            WHEN pair_quality IN ('same_book', 'cross_book')
             AND COALESCE(market_prob_source, '') NOT IN ('raw_implied', 'synthetic_fanduel_over_only')
            THEN 1.0
            ELSE 0.0
        END
      ) = 1.0
  AND NOT (
        LOWER(COALESCE(bookmaker_key, '')) = 'fanduel'
    AND market IN ('batter_hits','batter_total_bases','batter_home_runs')
    AND (
          COALESCE(pair_quality, '') = 'synthetic'
       OR COALESCE(market_prob_source, '') = 'synthetic_fanduel_over_only'
       OR COALESCE(paired_price_source, '') = 'synthetic_fanduel_over_only_complement'
       OR COALESCE(synthetic_pair_flag::float, 0.0) >= 0.5
    )
  )
"""

HISTORY_SQL = """
SELECT
    game_date_et,
    market,
    side,
    COALESCE(bookmaker_key, '*') AS bookmaker_key,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
    COALESCE(line_surface, 'unknown') AS line_surface,
    COALESCE(price_bucket, 'missing_price') AS price_bucket,
    market_line::float AS market_line,
    market_price::float AS market_price,
    market_prob_side::float AS market_prob_side,
    CASE WHEN won IS TRUE THEN 1 ELSE 0 END AS won,
    COALESCE(push, false) AS push,
    profit_units::float AS profit_units
FROM features.mlb_prop_market_history_examples
WHERE game_date_et >= %(cutoff)s
  AND market = ANY(%(markets)s)
  AND market_line IS NOT NULL
  AND won IS NOT NULL
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


def _query_df(conn, sql: str, params: dict[str, object]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=columns)


def _load(cfg: PropBucketReopenConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_training_schema(conn)
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = _query_df(conn, SQL, {"cutoff": cutoff, "markets": list(_MARKETS)})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["line_bucket"] = [
        prop_line_bucket(market, line)
        for market, line in zip(df["market"], df["market_line"])
    ]
    df["line_surface"] = [
        prop_line_surface(market, side, line)
        for market, side, line in zip(df["market"], df["side"], df["market_line"])
    ]
    df["price_bucket"] = df["market_price"].map(price_bucket)
    for col in (
        "market_line",
        "market_price",
        "model_prob_side",
        "market_prob_side",
        "prob_edge_vs_market",
        "ev",
        "kelly_fraction",
        "won",
        "profit_units",
        "clv_price",
        "beat_clv_price",
        "clv_line",
        "beat_clv_line",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    return df.replace([np.inf, -np.inf], np.nan)


def _load_history(cfg: PropBucketReopenConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.history_lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_market_history_schema(conn)
        if not _table_exists(conn, "features", "mlb_prop_market_history_examples"):
            return pd.DataFrame()
        df = _query_df(conn, HISTORY_SQL, {"cutoff": cutoff, "markets": list(_MARKETS)})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["line_bucket"] = [
        prop_line_bucket(market, line)
        for market, line in zip(df["market"], df["market_line"])
    ]
    df["line_surface"] = [
        prop_line_surface(market, side, line)
        for market, side, line in zip(df["market"], df["side"], df["market_line"])
    ]
    df["price_bucket"] = df["market_price"].map(price_bucket)
    for col in ("market_line", "market_price", "market_prob_side", "won", "profit_units"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    return df.replace([np.inf, -np.inf], np.nan)


def _load_clean_dates(cfg: PropBucketReopenConfig, *, date_to: date | None = None) -> tuple[set[date], list[dict]]:
    cutoff = max(
        datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days),
        cfg.eligibility_start_date,
    )
    end_date = date_to or datetime.now(timezone.utc).date()
    if end_date < cutoff:
        return set(), []
    thresholds = CleanSlateThresholds(
        min_side_locks=cfg.clean_slate_min_side_locks,
        min_valid_side_locks=cfg.clean_slate_min_valid_side_locks,
        min_valid_coverage=cfg.clean_slate_min_valid_coverage,
        max_missing_lock_rate=cfg.clean_slate_max_missing_lock_rate,
        max_stale_close_rate=cfg.clean_slate_max_stale_close_rate,
    )
    with psycopg2.connect(cfg.pg_dsn) as conn:
        rows = load_clean_slate_rows(
            conn,
            date_from=cutoff,
            date_to=end_date,
            thresholds=thresholds,
        )
    return clean_date_set(rows), rows


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _sum(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


def _max_share(rows: pd.DataFrame, column: str) -> float | None:
    if rows.empty or column not in rows.columns:
        return None
    counts = rows[column].dropna().value_counts()
    if counts.empty:
        return None
    return float(counts.iloc[0] / len(rows))


def _nunique(rows: pd.DataFrame, column: str) -> int:
    if rows.empty or column not in rows.columns:
        return 0
    return int(rows[column].dropna().nunique())


def _summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "rows": 0,
            "priced_rows": 0,
            "priced_rate": None,
            "win_rate": None,
            "units": None,
            "roi": None,
            "avg_model_prob": None,
            "calibration_error": None,
            "avg_market_prob": None,
            "avg_prob_edge": None,
            "avg_ev": None,
            "avg_kelly_fraction": None,
            "avg_clv_price": None,
            "clv_beat_rate": None,
            "avg_clv_line": None,
            "clv_line_beat_rate": None,
            "clv_price_rows": 0,
            "clv_line_rows": 0,
            "valid_close_coverage": None,
            "stale_close_rate": None,
            "unique_players": 0,
            "unique_teams": 0,
            "unique_dates": 0,
            "unique_clean_dates": 0,
            "clean_rows": 0,
            "max_player_share": None,
            "max_team_share": None,
            "max_game_date_share": None,
        }
    graded = rows.loc[~rows["push"]].copy()
    priced = rows["profit_units"].notna()
    clv_price = rows["beat_clv_price"].notna()
    clv_line = rows["beat_clv_line"].notna()
    valid_close = rows["clv_valid"].fillna(False).astype(bool)
    units = _sum(rows.loc[priced, "profit_units"])
    row_count = int(len(rows))
    priced_rows = int(priced.sum())
    win_rate = _mean(graded["won"]) if not graded.empty else None
    avg_model = _mean(rows["model_prob_side"])
    roi = (units / priced_rows) if units is not None and priced_rows else None
    calibration_error = (win_rate - avg_model) if win_rate is not None and avg_model is not None else None
    return {
        "rows": row_count,
        "clean_rows": int(rows.get("clean_slate", pd.Series(False, index=rows.index)).fillna(False).astype(bool).sum()),
        "priced_rows": priced_rows,
        "priced_rate": float(priced_rows / row_count) if row_count else None,
        "win_rate": win_rate,
        "units": units,
        "roi": roi,
        "avg_model_prob": avg_model,
        "calibration_error": calibration_error,
        "avg_market_prob": _mean(rows["market_prob_side"]),
        "avg_prob_edge": _mean(rows["prob_edge_vs_market"]),
        "avg_ev": _mean(rows["ev"]),
        "avg_kelly_fraction": _mean(rows["kelly_fraction"]),
        "avg_clv_price": _mean(rows["clv_price"]),
        "clv_beat_rate": _mean(rows.loc[clv_price, "beat_clv_price"]) if clv_price.any() else None,
        "avg_clv_line": _mean(rows["clv_line"]),
        "clv_line_beat_rate": _mean(rows.loc[clv_line, "beat_clv_line"]) if clv_line.any() else None,
        "clv_price_rows": int(clv_price.sum()),
        "clv_line_rows": int(clv_line.sum()),
        "valid_close_coverage": float(valid_close.mean()) if row_count else None,
        "stale_close_rate": float(
            (rows["clv_unknown_reason"] == "stale_close_before_lock").mean()
        ) if row_count else None,
        "unique_players": _nunique(rows, "player_id"),
        "unique_teams": _nunique(rows, "team_abbr"),
        "unique_dates": _nunique(rows, "game_date_et"),
        "unique_clean_dates": _nunique(
            rows.loc[
                rows.get("clean_slate", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
            ],
            "game_date_et",
        ),
        "max_player_share": _max_share(rows, "player_id"),
        "max_team_share": _max_share(rows, "team_abbr"),
        "max_game_date_share": _max_share(rows, "game_date_et"),
    }


def _history_summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "rows": 0,
            "priced_rows": 0,
            "priced_rate": None,
            "win_rate": None,
            "units": None,
            "roi": None,
            "avg_market_prob": None,
            "market_calibration_error": None,
        }
    graded = rows.loc[~rows["push"]].copy()
    priced = rows["profit_units"].notna()
    units = _sum(rows.loc[priced, "profit_units"])
    row_count = int(len(rows))
    priced_rows = int(priced.sum())
    win_rate = _mean(graded["won"]) if not graded.empty else None
    avg_market_prob = _mean(rows["market_prob_side"])
    roi = (units / priced_rows) if units is not None and priced_rows else None
    return {
        "rows": row_count,
        "priced_rows": priced_rows,
        "priced_rate": float(priced_rows / row_count) if row_count else None,
        "win_rate": win_rate,
        "units": units,
        "roi": roi,
        "avg_market_prob": avg_market_prob,
        "market_calibration_error": (
            win_rate - avg_market_prob
            if win_rate is not None and avg_market_prob is not None
            else None
        ),
    }


def _lt(value: float | None, threshold: float) -> bool:
    return value is None or float(value) < threshold


def _gt_abs(value: float | None, threshold: float) -> bool:
    return value is None or abs(float(value)) > threshold


def _gt(value: float | None, threshold: float) -> bool:
    return value is None or float(value) > threshold


def _policy_reasons(train: dict, holdout: dict, cfg: PropBucketReopenConfig) -> list[str]:
    reasons: list[str] = []
    total_rows = int(train["rows"] or 0) + int(holdout["rows"] or 0)
    if total_rows < cfg.min_total_rows:
        reasons.append(f"total_rows<{cfg.min_total_rows}")
    if train["rows"] < cfg.min_train_rows:
        reasons.append(f"train_rows<{cfg.min_train_rows}")
    if holdout["rows"] < cfg.min_holdout_rows:
        reasons.append(f"holdout_rows<{cfg.min_holdout_rows}")
    if _lt(holdout["priced_rate"], cfg.min_priced_rate):
        reasons.append(f"priced_rate<{cfg.min_priced_rate:.2f}")
    if _lt(train["roi"], cfg.min_train_roi):
        reasons.append(f"train_roi<{cfg.min_train_roi:.3f}")
    if _lt(holdout["roi"], cfg.min_holdout_roi):
        reasons.append(f"holdout_roi<{cfg.min_holdout_roi:.3f}")
    if _lt(holdout["avg_ev"], cfg.min_avg_ev):
        reasons.append(f"avg_ev<{cfg.min_avg_ev:.3f}")
    if _lt(holdout["avg_clv_price"], cfg.min_avg_clv_price):
        reasons.append(f"avg_clv_price<{cfg.min_avg_clv_price:.3f}")
    if holdout["clv_price_rows"] < cfg.min_clv_price_rows:
        reasons.append(f"clv_price_rows<{cfg.min_clv_price_rows}")
    if _lt(holdout["valid_close_coverage"], cfg.clean_slate_min_valid_coverage):
        reasons.append(f"valid_close_coverage<{cfg.clean_slate_min_valid_coverage:.2f}")
    if _gt(holdout["stale_close_rate"], cfg.clean_slate_max_stale_close_rate):
        reasons.append(f"stale_close_rate>{cfg.clean_slate_max_stale_close_rate:.2f}")
    if _lt(holdout["clv_beat_rate"], cfg.min_clv_beat_rate):
        reasons.append(f"clv_beat_rate<{cfg.min_clv_beat_rate:.2f}")
    if _gt_abs(holdout["calibration_error"], cfg.max_abs_calibration_error):
        reasons.append(f"abs_calibration_error>{cfg.max_abs_calibration_error:.3f}")
    if holdout["unique_players"] < cfg.min_unique_players:
        reasons.append(f"unique_players<{cfg.min_unique_players}")
    if holdout["unique_teams"] < cfg.min_unique_teams:
        reasons.append(f"unique_teams<{cfg.min_unique_teams}")
    if holdout["unique_dates"] < cfg.min_unique_dates:
        reasons.append(f"unique_dates<{cfg.min_unique_dates}")
    if holdout["unique_clean_dates"] < cfg.min_clean_unique_dates:
        reasons.append(f"clean_unique_dates<{cfg.min_clean_unique_dates}")
    if _gt(holdout["max_player_share"], cfg.max_player_share):
        reasons.append(f"max_player_share>{cfg.max_player_share:.2f}")
    if _gt(holdout["max_team_share"], cfg.max_team_share):
        reasons.append(f"max_team_share>{cfg.max_team_share:.2f}")
    if _gt(holdout["max_game_date_share"], cfg.max_game_date_share):
        reasons.append(f"max_game_date_share>{cfg.max_game_date_share:.2f}")
    return reasons


def _bootstrap_micro_reasons(
    summary: dict,
    cfg: PropBucketReopenConfig,
    values: dict,
) -> list[str]:
    reasons: list[str] = []
    if not cfg.enable_bootstrap_micro:
        reasons.append("bootstrap_micro_disabled")
    if cfg.bootstrap_common_only and values.get("line_surface") == "alt_tail":
        reasons.append("bootstrap_common_only")
    if summary["rows"] < cfg.bootstrap_min_rows:
        reasons.append(f"bootstrap_rows<{cfg.bootstrap_min_rows}")
    if _lt(summary["priced_rate"], cfg.bootstrap_min_priced_rate):
        reasons.append(f"bootstrap_priced_rate<{cfg.bootstrap_min_priced_rate:.2f}")
    if _lt(summary["roi"], cfg.bootstrap_min_roi):
        reasons.append(f"bootstrap_roi<{cfg.bootstrap_min_roi:.3f}")
    if _lt(summary["avg_ev"], cfg.bootstrap_min_avg_ev):
        reasons.append(f"bootstrap_avg_ev<{cfg.bootstrap_min_avg_ev:.3f}")
    if _lt(summary["avg_clv_price"], cfg.bootstrap_min_avg_clv_price):
        reasons.append(f"bootstrap_avg_clv_price<{cfg.bootstrap_min_avg_clv_price:.3f}")
    if summary["clv_price_rows"] < cfg.bootstrap_min_clv_price_rows:
        reasons.append(f"bootstrap_clv_price_rows<{cfg.bootstrap_min_clv_price_rows}")
    if _lt(summary["valid_close_coverage"], cfg.clean_slate_min_valid_coverage):
        reasons.append(f"bootstrap_valid_close_coverage<{cfg.clean_slate_min_valid_coverage:.2f}")
    if _gt(summary["stale_close_rate"], cfg.clean_slate_max_stale_close_rate):
        reasons.append(f"bootstrap_stale_close_rate>{cfg.clean_slate_max_stale_close_rate:.2f}")
    if _lt(summary["clv_beat_rate"], cfg.bootstrap_min_clv_beat_rate):
        reasons.append(f"bootstrap_clv_beat_rate<{cfg.bootstrap_min_clv_beat_rate:.2f}")
    if _gt_abs(summary["calibration_error"], cfg.bootstrap_max_abs_calibration_error):
        reasons.append(f"bootstrap_abs_calibration_error>{cfg.bootstrap_max_abs_calibration_error:.3f}")
    if summary["unique_players"] < cfg.bootstrap_min_unique_players:
        reasons.append(f"bootstrap_unique_players<{cfg.bootstrap_min_unique_players}")
    if summary["unique_teams"] < cfg.bootstrap_min_unique_teams:
        reasons.append(f"bootstrap_unique_teams<{cfg.bootstrap_min_unique_teams}")
    if summary["unique_dates"] < cfg.bootstrap_min_unique_dates:
        reasons.append(f"bootstrap_unique_dates<{cfg.bootstrap_min_unique_dates}")
    if summary["unique_clean_dates"] < cfg.bootstrap_min_clean_unique_dates:
        reasons.append(f"bootstrap_clean_unique_dates<{cfg.bootstrap_min_clean_unique_dates}")
    if _gt(summary["max_player_share"], cfg.bootstrap_max_player_share):
        reasons.append(f"bootstrap_max_player_share>{cfg.bootstrap_max_player_share:.2f}")
    if _gt(summary["max_team_share"], cfg.bootstrap_max_team_share):
        reasons.append(f"bootstrap_max_team_share>{cfg.bootstrap_max_team_share:.2f}")
    if _gt(summary["max_game_date_share"], cfg.bootstrap_max_game_date_share):
        reasons.append(f"bootstrap_max_game_date_share>{cfg.bootstrap_max_game_date_share:.2f}")
    return reasons


_LADDER_TIERS = ("watch", "micro", "starter", "bankroll")
_LADDER_RANK = {tier: idx for idx, tier in enumerate(_LADDER_TIERS)}


def _additional_ladder_reasons(
    train: dict,
    holdout: dict,
    cfg: PropBucketReopenConfig,
    tier: str,
) -> list[str]:
    if tier == "micro":
        return []
    total_rows = int(train["rows"] or 0) + int(holdout["rows"] or 0)
    if tier == "starter":
        thresholds = (
            ("total_rows", total_rows, cfg.starter_min_total_rows),
            ("holdout_rows", int(holdout["rows"] or 0), cfg.starter_min_holdout_rows),
            ("clv_price_rows", int(holdout["clv_price_rows"] or 0), cfg.starter_min_clv_price_rows),
            ("unique_dates", int(holdout["unique_dates"] or 0), cfg.starter_min_unique_dates),
            ("clean_unique_dates", int(holdout["unique_clean_dates"] or 0), cfg.starter_min_clean_unique_dates),
        )
        max_cal = cfg.starter_max_abs_calibration_error
    else:
        thresholds = (
            ("total_rows", total_rows, cfg.bankroll_min_total_rows),
            ("holdout_rows", int(holdout["rows"] or 0), cfg.bankroll_min_holdout_rows),
            ("clv_price_rows", int(holdout["clv_price_rows"] or 0), cfg.bankroll_min_clv_price_rows),
            ("unique_dates", int(holdout["unique_dates"] or 0), cfg.bankroll_min_unique_dates),
            ("clean_unique_dates", int(holdout["unique_clean_dates"] or 0), cfg.bankroll_min_clean_unique_dates),
        )
        max_cal = cfg.bankroll_max_abs_calibration_error
    reasons = [f"{name}<{minimum}" for name, value, minimum in thresholds if value < minimum]
    if _gt_abs(holdout.get("calibration_error"), max_cal):
        reasons.append(f"abs_calibration_error>{max_cal:.3f}")
    return reasons


def _desired_ladder_tier(
    train: dict,
    holdout: dict,
    cfg: PropBucketReopenConfig,
    base_reasons: list[str],
) -> tuple[str, dict[str, list[str]]]:
    reasons_by_tier = {
        "micro": list(base_reasons),
        "starter": list(base_reasons) + _additional_ladder_reasons(train, holdout, cfg, "starter"),
        "bankroll": list(base_reasons) + _additional_ladder_reasons(train, holdout, cfg, "bankroll"),
    }
    for tier in ("bankroll", "starter", "micro"):
        if not reasons_by_tier[tier]:
            return tier, reasons_by_tier
    return "watch", reasons_by_tier


def _graduate_one_rung(previous: str, desired: str) -> str:
    previous = previous if previous in _LADDER_RANK else "watch"
    desired = desired if desired in _LADDER_RANK else "watch"
    if _LADDER_RANK[desired] <= _LADDER_RANK[previous]:
        return desired
    return _LADDER_TIERS[min(_LADDER_RANK[previous] + 1, _LADDER_RANK[desired])]


def _load_previous_ladder(path: Path, eligibility_start_date: date) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if str(payload.get("eligibility_start_date") or "") != eligibility_start_date.isoformat():
        return {}
    out: dict[str, dict] = {}
    for key, record in (payload.get("ladder_buckets") or {}).items():
        out[str(key)] = dict(record or {})
    for key, record in (payload.get("reopen_buckets") or {}).items():
        out.setdefault(str(key), dict(record or {}))
    return out


def _history_guard_reasons(history: dict, cfg: PropBucketReopenConfig) -> list[str]:
    reasons: list[str] = []
    if history["rows"] < cfg.min_history_rows:
        reasons.append(f"history_rows<{cfg.min_history_rows}")
    if _lt(history["priced_rate"], cfg.min_history_priced_rate):
        reasons.append(f"history_priced_rate<{cfg.min_history_priced_rate:.2f}")
    if _lt(history["roi"], cfg.min_history_roi):
        reasons.append(f"history_roi<{cfg.min_history_roi:.3f}")
    if _gt_abs(history["market_calibration_error"], cfg.max_history_abs_market_calibration_error):
        reasons.append(f"history_abs_market_calibration_error>{cfg.max_history_abs_market_calibration_error:.3f}")
    return reasons


def _group_specs() -> Iterable[tuple[str, tuple[str, ...]]]:
    return [
        ("bucket", ("market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key")),
        ("line_bucket", ("market", "side", "line_surface", "line_bucket", "bookmaker_key")),
        ("market_side_surface", ("market", "side", "line_surface")),
        ("market_side", ("market", "side")),
    ]


def _bucket_key(values: dict) -> str:
    return "|".join([
        str(values.get("market", "*")),
        str(values.get("side", "*")),
        str(values.get("line_surface", "*")),
        str(values.get("line_bucket", "*")),
        str(values.get("price_bucket", "*")),
        str(values.get("bookmaker_key", "*")),
    ])


def _is_tail_alt_bucket(values: dict) -> bool:
    return values.get("line_surface") == "alt_tail"


def _history_index(history_df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    if history_df.empty:
        return {}
    index: dict[tuple[str, str], dict] = {}
    for level, group_cols in _group_specs():
        for values, sub in history_df.groupby(list(group_cols), dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            value_dict = dict(zip(group_cols, values))
            index[(level, _bucket_key(value_dict))] = _history_summary(sub)
    return index


def train(cfg: PropBucketReopenConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.model_dir / cfg.out_file
    previous_ladder = _load_previous_ladder(output_path, cfg.eligibility_start_date)
    legacy_df = _load(cfg)
    history_df = _load_history(cfg)
    evidence_date_to = max(legacy_df["game_date_et"]) if not legacy_df.empty else datetime.now(timezone.utc).date()
    clean_dates, clean_slate_rows = _load_clean_dates(cfg, date_to=evidence_date_to)
    if not legacy_df.empty:
        legacy_df["clean_slate"] = legacy_df["game_date_et"].isin(clean_dates)
    df = legacy_df.loc[
        legacy_df["game_date_et"] >= cfg.eligibility_start_date
    ].copy() if not legacy_df.empty else legacy_df.copy()
    history_by_level = _history_index(history_df)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
        "promotion_scope": "exact_bucket_only",
        "eligibility_start_date": cfg.eligibility_start_date.isoformat(),
        "eligibility_rule": "promotion_evidence_only_on_or_after_fixed_start_date",
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "thresholds": {
            "min_total_rows": cfg.min_total_rows,
            "min_train_rows": cfg.min_train_rows,
            "min_holdout_rows": cfg.min_holdout_rows,
            "min_priced_rate": cfg.min_priced_rate,
            "min_train_roi": cfg.min_train_roi,
            "min_holdout_roi": cfg.min_holdout_roi,
            "min_avg_ev": cfg.min_avg_ev,
            "min_avg_clv_price": cfg.min_avg_clv_price,
            "min_clv_beat_rate": cfg.min_clv_beat_rate,
            "min_clv_price_rows": cfg.min_clv_price_rows,
            "max_abs_calibration_error": cfg.max_abs_calibration_error,
            "min_unique_players": cfg.min_unique_players,
            "min_unique_teams": cfg.min_unique_teams,
            "min_unique_dates": cfg.min_unique_dates,
            "min_clean_unique_dates": cfg.min_clean_unique_dates,
            "clean_slate_min_side_locks": cfg.clean_slate_min_side_locks,
            "clean_slate_min_valid_side_locks": cfg.clean_slate_min_valid_side_locks,
            "clean_slate_min_valid_coverage": cfg.clean_slate_min_valid_coverage,
            "clean_slate_max_missing_lock_rate": cfg.clean_slate_max_missing_lock_rate,
            "clean_slate_max_stale_close_rate": cfg.clean_slate_max_stale_close_rate,
            "max_player_share": cfg.max_player_share,
            "max_team_share": cfg.max_team_share,
            "max_game_date_share": cfg.max_game_date_share,
            "starter_min_total_rows": cfg.starter_min_total_rows,
            "starter_min_holdout_rows": cfg.starter_min_holdout_rows,
            "starter_min_clv_price_rows": cfg.starter_min_clv_price_rows,
            "starter_min_unique_dates": cfg.starter_min_unique_dates,
            "starter_min_clean_unique_dates": cfg.starter_min_clean_unique_dates,
            "starter_max_abs_calibration_error": cfg.starter_max_abs_calibration_error,
            "bankroll_min_total_rows": cfg.bankroll_min_total_rows,
            "bankroll_min_holdout_rows": cfg.bankroll_min_holdout_rows,
            "bankroll_min_clv_price_rows": cfg.bankroll_min_clv_price_rows,
            "bankroll_min_unique_dates": cfg.bankroll_min_unique_dates,
            "bankroll_min_clean_unique_dates": cfg.bankroll_min_clean_unique_dates,
            "bankroll_max_abs_calibration_error": cfg.bankroll_max_abs_calibration_error,
            "history_lookback_days": cfg.history_lookback_days,
            "min_history_rows": cfg.min_history_rows,
            "min_history_priced_rate": cfg.min_history_priced_rate,
            "min_history_roi": cfg.min_history_roi,
            "max_history_abs_market_calibration_error": cfg.max_history_abs_market_calibration_error,
            "enforce_history_guard": cfg.enforce_history_guard,
            "force_reopen_all": cfg.force_reopen_all,
            "research_only": cfg.research_only,
            "enable_bootstrap_micro": cfg.enable_bootstrap_micro,
            "bootstrap_common_only": cfg.bootstrap_common_only,
            "bootstrap_min_rows": cfg.bootstrap_min_rows,
            "bootstrap_min_priced_rate": cfg.bootstrap_min_priced_rate,
            "bootstrap_min_roi": cfg.bootstrap_min_roi,
            "bootstrap_min_avg_ev": cfg.bootstrap_min_avg_ev,
            "bootstrap_min_avg_clv_price": cfg.bootstrap_min_avg_clv_price,
            "bootstrap_min_clv_beat_rate": cfg.bootstrap_min_clv_beat_rate,
            "bootstrap_min_clv_price_rows": cfg.bootstrap_min_clv_price_rows,
            "bootstrap_max_abs_calibration_error": cfg.bootstrap_max_abs_calibration_error,
            "bootstrap_min_unique_players": cfg.bootstrap_min_unique_players,
            "bootstrap_min_unique_teams": cfg.bootstrap_min_unique_teams,
            "bootstrap_min_unique_dates": cfg.bootstrap_min_unique_dates,
            "bootstrap_min_clean_unique_dates": cfg.bootstrap_min_clean_unique_dates,
            "bootstrap_max_player_share": cfg.bootstrap_max_player_share,
            "bootstrap_max_team_share": cfg.bootstrap_max_team_share,
            "bootstrap_max_game_date_share": cfg.bootstrap_max_game_date_share,
        },
        "rows": int(len(df)),
        "eligible_rows": int(len(df)),
        "legacy_audit_rows": int(len(legacy_df)),
        "legacy_audit_summary": _summary(legacy_df) if not legacy_df.empty else _summary(pd.DataFrame()),
        "history_rows": int(len(history_df)),
        "clean_slate_count": len(clean_dates),
        "clean_slate_rows": clean_slate_rows,
        "force_reopen_all": cfg.force_reopen_all,
        "research_only": cfg.research_only,
        "reopen_buckets": {},
        "ladder_buckets": {},
        "closed_buckets": {},
        "diagnostics": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    split = max(df["game_date_et"]) - timedelta(days=cfg.holdout_days)
    train_mask = df["game_date_et"] < split
    holdout_mask = df["game_date_et"] >= split

    for level, group_cols in _group_specs():
        for values, sub in df.groupby(list(group_cols), dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            value_dict = dict(zip(group_cols, values))
            train_summary = _summary(sub.loc[train_mask.loc[sub.index]])
            holdout_summary = _summary(sub.loc[holdout_mask.loc[sub.index]])
            all_summary = _summary(sub)
            reasons = _policy_reasons(train_summary, holdout_summary, cfg)
            history_summary = history_by_level.get(
                (level, _bucket_key(value_dict)),
                _history_summary(pd.DataFrame()),
            )
            history_reasons = _history_guard_reasons(history_summary, cfg)
            if cfg.enforce_history_guard:
                reasons.extend(history_reasons)
            if _is_tail_alt_bucket(value_dict) and reasons:
                reasons = ["tail_alt_requires_proof", *[r for r in reasons if r != "tail_alt_requires_proof"]]
            key = _bucket_key(value_dict)
            desired_tier, reasons_by_tier = _desired_ladder_tier(
                train_summary,
                holdout_summary,
                cfg,
                reasons,
            )
            bootstrap_reasons = (
                _bootstrap_micro_reasons(all_summary, cfg, value_dict)
                if level == "bucket"
                else ["bootstrap_exact_bucket_only"]
            )
            bootstrap_micro_eligible = (
                level == "bucket"
                and cfg.enable_bootstrap_micro
                and desired_tier == "watch"
                and int(train_summary.get("rows") or 0) < cfg.min_train_rows
                and not bootstrap_reasons
            )
            promotion_source = "full_policy" if desired_tier != "watch" else "closed"
            if bootstrap_micro_eligible:
                desired_tier = "micro"
                reasons_by_tier["micro"] = []
                reasons_by_tier["starter"] = sorted(set(
                    list(reasons_by_tier.get("starter") or []) + ["bootstrap_micro_only"]
                ))
                reasons_by_tier["bankroll"] = sorted(set(
                    list(reasons_by_tier.get("bankroll") or []) + ["bootstrap_micro_only"]
                ))
                promotion_source = "bootstrap_micro"
            if cfg.force_reopen_all and not cfg.research_only:
                desired_tier = "bankroll"
                promotion_source = "forced"
            previous_record = previous_ladder.get(key, {})
            previous_tier = str(previous_record.get("ladder_tier") or "watch")
            previous_evidence_dates = int(previous_record.get("ladder_evidence_unique_dates") or 0)
            current_evidence_dates = int(holdout_summary.get("unique_dates") or 0)
            awaiting_new_slate = (
                bool(previous_record)
                and _LADDER_RANK.get(desired_tier, 0) > _LADDER_RANK.get(previous_tier, 0)
                and current_evidence_dates <= previous_evidence_dates
            )
            if cfg.research_only:
                ladder_tier = "watch"
            elif awaiting_new_slate:
                ladder_tier = previous_tier
            else:
                ladder_tier = _graduate_one_rung(previous_tier, desired_tier)
            ladder_changed = ladder_tier != previous_tier
            last_ladder_change_at_utc = (
                payload["generated_at_utc"]
                if ladder_changed
                else previous_record.get("last_ladder_change_at_utc")
            )
            status = (
                "research_only"
                if cfg.research_only
                else f"{ladder_tier}_eligible"
                if ladder_tier != "watch"
                else "closed"
            )
            record = {
                "key": key,
                "level": level,
                "market": value_dict.get("market", "*"),
                "side": value_dict.get("side", "*"),
                "line_surface": value_dict.get("line_surface", "*"),
                "line_bucket": value_dict.get("line_bucket", "*"),
                "price_bucket": value_dict.get("price_bucket", "*"),
                "bookmaker_key": value_dict.get("bookmaker_key", "*"),
                "status": status,
                "ladder_tier": ladder_tier,
                "desired_ladder_tier": desired_tier,
                "previous_ladder_tier": previous_tier,
                "awaiting_new_slate": awaiting_new_slate,
                "ladder_block_reason": "awaiting_new_slate_for_next_rung" if awaiting_new_slate else None,
                "ladder_evidence_unique_dates": current_evidence_dates,
                "last_ladder_change_at_utc": last_ladder_change_at_utc,
                "ladder_reasons": reasons_by_tier,
                "promotion_source": promotion_source,
                "bootstrap_micro_eligible": bootstrap_micro_eligible,
                "bootstrap_micro_reasons": bootstrap_reasons,
                "reasons": reasons,
                "model_reasons": reasons,
                "history_reasons": history_reasons,
                "train": train_summary,
                "holdout": holdout_summary,
                "bootstrap": all_summary,
                "history": history_summary,
            }
            payload["diagnostics"].append(record)
            if level == "bucket":
                ladder_record = {
                        "market": record["market"],
                        "side": record["side"],
                        "line_surface": record["line_surface"],
                        "line_bucket": record["line_bucket"],
                        "price_bucket": record["price_bucket"],
                        "bookmaker_key": record["bookmaker_key"],
                        "status": status,
                        "ladder_tier": ladder_tier,
                        "desired_ladder_tier": desired_tier,
                        "previous_ladder_tier": previous_tier,
                        "awaiting_new_slate": awaiting_new_slate,
                        "ladder_block_reason": "awaiting_new_slate_for_next_rung" if awaiting_new_slate else None,
                        "ladder_evidence_unique_dates": current_evidence_dates,
                        "last_ladder_change_at_utc": last_ladder_change_at_utc,
                        "ladder_reasons": reasons_by_tier,
                        "promotion_source": promotion_source,
                        "bootstrap_micro_eligible": bootstrap_micro_eligible,
                        "bootstrap_micro_reasons": bootstrap_reasons,
                        "research_only": cfg.research_only,
                        "train_rows": train_summary["rows"],
                        "holdout_rows": holdout_summary["rows"],
                        "holdout_roi": holdout_summary["roi"],
                        "holdout_win_rate": holdout_summary["win_rate"],
                        "holdout_avg_model_prob": holdout_summary["avg_model_prob"],
                        "holdout_calibration_error": holdout_summary["calibration_error"],
                        "holdout_avg_clv_price": holdout_summary["avg_clv_price"],
                        "holdout_clv_beat_rate": holdout_summary["clv_beat_rate"],
                        "holdout_clv_price_rows": holdout_summary["clv_price_rows"],
                        "holdout_valid_close_coverage": holdout_summary["valid_close_coverage"],
                        "holdout_stale_close_rate": holdout_summary["stale_close_rate"],
                        "holdout_unique_players": holdout_summary["unique_players"],
                        "holdout_unique_teams": holdout_summary["unique_teams"],
                        "holdout_unique_dates": holdout_summary["unique_dates"],
                        "holdout_unique_clean_dates": holdout_summary["unique_clean_dates"],
                        "holdout_clean_rows": holdout_summary["clean_rows"],
                        "holdout_max_player_share": holdout_summary["max_player_share"],
                        "holdout_max_team_share": holdout_summary["max_team_share"],
                        "holdout_max_game_date_share": holdout_summary["max_game_date_share"],
                        "bootstrap_rows": all_summary["rows"],
                        "bootstrap_roi": all_summary["roi"],
                        "bootstrap_win_rate": all_summary["win_rate"],
                        "bootstrap_avg_model_prob": all_summary["avg_model_prob"],
                        "bootstrap_calibration_error": all_summary["calibration_error"],
                        "bootstrap_avg_ev": all_summary["avg_ev"],
                        "bootstrap_avg_clv_price": all_summary["avg_clv_price"],
                        "bootstrap_clv_beat_rate": all_summary["clv_beat_rate"],
                        "bootstrap_clv_price_rows": all_summary["clv_price_rows"],
                        "bootstrap_valid_close_coverage": all_summary["valid_close_coverage"],
                        "bootstrap_stale_close_rate": all_summary["stale_close_rate"],
                        "bootstrap_unique_players": all_summary["unique_players"],
                        "bootstrap_unique_teams": all_summary["unique_teams"],
                        "bootstrap_unique_dates": all_summary["unique_dates"],
                        "bootstrap_unique_clean_dates": all_summary["unique_clean_dates"],
                        "bootstrap_max_player_share": all_summary["max_player_share"],
                        "bootstrap_max_team_share": all_summary["max_team_share"],
                        "bootstrap_max_game_date_share": all_summary["max_game_date_share"],
                        "history_rows": history_summary["rows"],
                        "history_roi": history_summary["roi"],
                        "history_win_rate": history_summary["win_rate"],
                        "history_avg_market_prob": history_summary["avg_market_prob"],
                        "history_market_calibration_error": history_summary["market_calibration_error"],
                        "model_reasons": reasons,
                        "history_reasons": history_reasons,
                    }
                payload["ladder_buckets"][key] = ladder_record
                if ladder_tier in {"micro", "starter", "bankroll"} and not cfg.research_only:
                    payload["reopen_buckets"][key] = ladder_record
                else:
                    payload["closed_buckets"][key] = reasons

    payload["diagnostics"].sort(key=lambda r: (
        r["level"], r["market"], r["side"], r.get("line_surface", "*"), r["line_bucket"], r["price_bucket"]
    ))
    if cfg.force_reopen_all:
        payload["status"] = "forced_reopen_all_research" if cfg.research_only else "forced_reopen_all"
    else:
        payload["status"] = "ready_for_bucket_review" if payload["reopen_buckets"] else "all_buckets_closed"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop bucket reopen policy")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_bucket_reopen_policy.json")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument(
        "--eligibility-start-date",
        default=PROP_REAL_MONEY_ELIGIBILITY_START_DATE.isoformat(),
        help="Fixed first slate allowed to count toward real-money promotion.",
    )
    parser.add_argument("--min-total-rows", type=int, default=150)
    parser.add_argument("--min-train-rows", type=int, default=150)
    parser.add_argument("--min-holdout-rows", type=int, default=40)
    parser.add_argument("--min-priced-rate", type=float, default=0.95)
    parser.add_argument("--min-train-roi", type=float, default=-0.02)
    parser.add_argument("--min-holdout-roi", type=float, default=0.0)
    parser.add_argument("--min-avg-ev", type=float, default=0.0)
    parser.add_argument("--min-avg-clv-price", type=float, default=0.0)
    parser.add_argument("--min-clv-beat-rate", type=float, default=0.55)
    parser.add_argument("--min-clv-price-rows", type=int, default=30)
    parser.add_argument("--max-abs-calibration-error", type=float, default=0.05)
    parser.add_argument("--min-unique-players", type=int, default=20)
    parser.add_argument("--min-unique-teams", type=int, default=6)
    parser.add_argument("--min-unique-dates", type=int, default=5)
    parser.add_argument("--min-clean-unique-dates", type=int, default=5)
    parser.add_argument("--clean-slate-min-side-locks", type=int, default=100)
    parser.add_argument("--clean-slate-min-valid-side-locks", type=int, default=100)
    parser.add_argument("--clean-slate-min-valid-coverage", type=float, default=0.90)
    parser.add_argument("--clean-slate-max-missing-lock-rate", type=float, default=0.02)
    parser.add_argument("--clean-slate-max-stale-close-rate", type=float, default=0.02)
    parser.add_argument("--max-player-share", type=float, default=0.12)
    parser.add_argument("--max-team-share", type=float, default=0.25)
    parser.add_argument("--max-game-date-share", type=float, default=0.35)
    parser.add_argument("--starter-min-total-rows", type=int, default=250)
    parser.add_argument("--starter-min-holdout-rows", type=int, default=60)
    parser.add_argument("--starter-min-clv-price-rows", type=int, default=50)
    parser.add_argument("--starter-min-unique-dates", type=int, default=8)
    parser.add_argument("--starter-min-clean-unique-dates", type=int, default=8)
    parser.add_argument("--starter-max-abs-calibration-error", type=float, default=0.04)
    parser.add_argument("--bankroll-min-total-rows", type=int, default=500)
    parser.add_argument("--bankroll-min-holdout-rows", type=int, default=100)
    parser.add_argument("--bankroll-min-clv-price-rows", type=int, default=100)
    parser.add_argument("--bankroll-min-unique-dates", type=int, default=12)
    parser.add_argument("--bankroll-min-clean-unique-dates", type=int, default=12)
    parser.add_argument("--bankroll-max-abs-calibration-error", type=float, default=0.03)
    parser.add_argument("--history-lookback-days", type=int, default=540)
    parser.add_argument("--min-history-rows", type=int, default=500)
    parser.add_argument("--min-history-priced-rate", type=float, default=0.80)
    parser.add_argument("--min-history-roi", type=float, default=-0.15)
    parser.add_argument("--max-history-abs-market-calibration-error", type=float, default=0.12)
    parser.add_argument("--enforce-history-guard", action="store_true")
    parser.add_argument("--force-reopen-all", action="store_true")
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--disable-bootstrap-micro", action="store_true")
    parser.add_argument("--allow-bootstrap-alt-tail", action="store_true")
    parser.add_argument("--bootstrap-min-rows", type=int, default=150)
    parser.add_argument("--bootstrap-min-priced-rate", type=float, default=0.95)
    parser.add_argument("--bootstrap-min-roi", type=float, default=0.0)
    parser.add_argument("--bootstrap-min-avg-ev", type=float, default=0.0)
    parser.add_argument("--bootstrap-min-avg-clv-price", type=float, default=0.0)
    parser.add_argument("--bootstrap-min-clv-beat-rate", type=float, default=0.55)
    parser.add_argument("--bootstrap-min-clv-price-rows", type=int, default=30)
    parser.add_argument("--bootstrap-max-abs-calibration-error", type=float, default=0.05)
    parser.add_argument("--bootstrap-min-unique-players", type=int, default=20)
    parser.add_argument("--bootstrap-min-unique-teams", type=int, default=6)
    parser.add_argument("--bootstrap-min-unique-dates", type=int, default=5)
    parser.add_argument("--bootstrap-min-clean-unique-dates", type=int, default=5)
    parser.add_argument("--bootstrap-max-player-share", type=float, default=0.12)
    parser.add_argument("--bootstrap-max-team-share", type=float, default=0.25)
    parser.add_argument("--bootstrap-max-game-date-share", type=float, default=0.35)
    args = parser.parse_args()
    payload = train(PropBucketReopenConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        eligibility_start_date=parse_eligibility_start_date(args.eligibility_start_date),
        min_total_rows=args.min_total_rows,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_priced_rate=args.min_priced_rate,
        min_train_roi=args.min_train_roi,
        min_holdout_roi=args.min_holdout_roi,
        min_avg_ev=args.min_avg_ev,
        min_avg_clv_price=args.min_avg_clv_price,
        min_clv_beat_rate=args.min_clv_beat_rate,
        min_clv_price_rows=args.min_clv_price_rows,
        max_abs_calibration_error=args.max_abs_calibration_error,
        min_unique_players=args.min_unique_players,
        min_unique_teams=args.min_unique_teams,
        min_unique_dates=args.min_unique_dates,
        min_clean_unique_dates=args.min_clean_unique_dates,
        clean_slate_min_side_locks=args.clean_slate_min_side_locks,
        clean_slate_min_valid_side_locks=args.clean_slate_min_valid_side_locks,
        clean_slate_min_valid_coverage=args.clean_slate_min_valid_coverage,
        clean_slate_max_missing_lock_rate=args.clean_slate_max_missing_lock_rate,
        clean_slate_max_stale_close_rate=args.clean_slate_max_stale_close_rate,
        max_player_share=args.max_player_share,
        max_team_share=args.max_team_share,
        max_game_date_share=args.max_game_date_share,
        starter_min_total_rows=args.starter_min_total_rows,
        starter_min_holdout_rows=args.starter_min_holdout_rows,
        starter_min_clv_price_rows=args.starter_min_clv_price_rows,
        starter_min_unique_dates=args.starter_min_unique_dates,
        starter_min_clean_unique_dates=args.starter_min_clean_unique_dates,
        starter_max_abs_calibration_error=args.starter_max_abs_calibration_error,
        bankroll_min_total_rows=args.bankroll_min_total_rows,
        bankroll_min_holdout_rows=args.bankroll_min_holdout_rows,
        bankroll_min_clv_price_rows=args.bankroll_min_clv_price_rows,
        bankroll_min_unique_dates=args.bankroll_min_unique_dates,
        bankroll_min_clean_unique_dates=args.bankroll_min_clean_unique_dates,
        bankroll_max_abs_calibration_error=args.bankroll_max_abs_calibration_error,
        history_lookback_days=args.history_lookback_days,
        min_history_rows=args.min_history_rows,
        min_history_priced_rate=args.min_history_priced_rate,
        min_history_roi=args.min_history_roi,
        max_history_abs_market_calibration_error=args.max_history_abs_market_calibration_error,
        enforce_history_guard=args.enforce_history_guard,
        force_reopen_all=args.force_reopen_all,
        research_only=args.research_only,
        enable_bootstrap_micro=not args.disable_bootstrap_micro,
        bootstrap_common_only=not args.allow_bootstrap_alt_tail,
        bootstrap_min_rows=args.bootstrap_min_rows,
        bootstrap_min_priced_rate=args.bootstrap_min_priced_rate,
        bootstrap_min_roi=args.bootstrap_min_roi,
        bootstrap_min_avg_ev=args.bootstrap_min_avg_ev,
        bootstrap_min_avg_clv_price=args.bootstrap_min_avg_clv_price,
        bootstrap_min_clv_beat_rate=args.bootstrap_min_clv_beat_rate,
        bootstrap_min_clv_price_rows=args.bootstrap_min_clv_price_rows,
        bootstrap_max_abs_calibration_error=args.bootstrap_max_abs_calibration_error,
        bootstrap_min_unique_players=args.bootstrap_min_unique_players,
        bootstrap_min_unique_teams=args.bootstrap_min_unique_teams,
        bootstrap_min_unique_dates=args.bootstrap_min_unique_dates,
        bootstrap_min_clean_unique_dates=args.bootstrap_min_clean_unique_dates,
        bootstrap_max_player_share=args.bootstrap_max_player_share,
        bootstrap_max_team_share=args.bootstrap_max_team_share,
        bootstrap_max_game_date_share=args.bootstrap_max_game_date_share,
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
