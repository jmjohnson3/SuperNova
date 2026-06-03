"""Build prop bucket reopen policy from side-level market training examples.

This does not make live bets by itself.  It writes an auditable artifact that
marks which market/side/line/price buckets have enough graded evidence to be
considered for bankroll reopening.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2

from .side_recalibration import prop_line_bucket

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_MARKETS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")


@dataclass(frozen=True)
class PropBucketReopenConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_bucket_reopen_policy.json"
    lookback_days: int = 365
    holdout_days: int = 28
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
    max_player_share: float = 0.12
    max_team_share: float = 0.25
    max_game_date_share: float = 0.35
    history_lookback_days: int = 540
    min_history_rows: int = 500
    min_history_priced_rate: float = 0.80
    min_history_roi: float = -0.15
    max_history_abs_market_calibration_error: float = 0.12
    enforce_history_guard: bool = False
    force_reopen_all: bool = False
    research_only: bool = False


SQL = """
SELECT
    game_date_et,
    game_slug,
    player_id,
    team_abbr,
    market,
    side,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
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
    clv_price::float AS clv_price,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_price,
    clv_line::float AS clv_line,
    CASE
        WHEN beat_clv_line IS TRUE THEN 1
        WHEN beat_clv_line IS FALSE THEN 0
        ELSE NULL
    END AS beat_clv_line
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = ANY(%(markets)s)
  AND model_prob_side IS NOT NULL
  AND market_line IS NOT NULL
  AND won IS NOT NULL
  AND model_prob_side BETWEEN 0.0 AND 1.0
"""

HISTORY_SQL = """
SELECT
    game_date_et,
    market,
    side,
    COALESCE(line_bucket, 'unknown') AS line_bucket,
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


def _load(cfg: PropBucketReopenConfig) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            return pd.DataFrame()
        df = pd.read_sql(SQL, conn, params={"cutoff": cutoff, "markets": list(_MARKETS)})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["line_bucket"] = [
        lb if isinstance(lb, str) and lb and lb != "unknown" else prop_line_bucket(market, line)
        for market, line, lb in zip(df["market"], df["market_line"], df["line_bucket"])
    ]
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
        if not _table_exists(conn, "features", "mlb_prop_market_history_examples"):
            return pd.DataFrame()
        df = pd.read_sql(HISTORY_SQL, conn, params={"cutoff": cutoff, "markets": list(_MARKETS)})
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    df["line_bucket"] = [
        lb if isinstance(lb, str) and lb and lb != "unknown" else prop_line_bucket(market, line)
        for market, line, lb in zip(df["market"], df["market_line"], df["line_bucket"])
    ]
    for col in ("market_line", "market_price", "market_prob_side", "won", "profit_units"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["push"] = df["push"].fillna(False).astype(bool)
    return df.replace([np.inf, -np.inf], np.nan)


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
            "unique_players": 0,
            "unique_teams": 0,
            "unique_dates": 0,
            "max_player_share": None,
            "max_team_share": None,
            "max_game_date_share": None,
        }
    graded = rows.loc[~rows["push"]].copy()
    priced = rows["profit_units"].notna()
    clv_price = rows["beat_clv_price"].notna()
    clv_line = rows["beat_clv_line"].notna()
    units = _sum(rows.loc[priced, "profit_units"])
    row_count = int(len(rows))
    priced_rows = int(priced.sum())
    win_rate = _mean(graded["won"]) if not graded.empty else None
    avg_model = _mean(rows["model_prob_side"])
    roi = (units / priced_rows) if units is not None and priced_rows else None
    calibration_error = (win_rate - avg_model) if win_rate is not None and avg_model is not None else None
    return {
        "rows": row_count,
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
        "unique_players": _nunique(rows, "player_id"),
        "unique_teams": _nunique(rows, "team_abbr"),
        "unique_dates": _nunique(rows, "game_date_et"),
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
    if _gt(holdout["max_player_share"], cfg.max_player_share):
        reasons.append(f"max_player_share>{cfg.max_player_share:.2f}")
    if _gt(holdout["max_team_share"], cfg.max_team_share):
        reasons.append(f"max_team_share>{cfg.max_team_share:.2f}")
    if _gt(holdout["max_game_date_share"], cfg.max_game_date_share):
        reasons.append(f"max_game_date_share>{cfg.max_game_date_share:.2f}")
    return reasons


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
        ("bucket", ("market", "side", "line_bucket", "price_bucket")),
        ("line_bucket", ("market", "side", "line_bucket")),
        ("market_side", ("market", "side")),
    ]


def _bucket_key(values: dict) -> str:
    return "|".join([
        str(values.get("market", "*")),
        str(values.get("side", "*")),
        str(values.get("line_bucket", "*")),
        str(values.get("price_bucket", "*")),
    ])


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
    df = _load(cfg)
    history_df = _load_history(cfg)
    history_by_level = _history_index(history_df)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "features.mlb_prop_market_training_examples",
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
            "max_player_share": cfg.max_player_share,
            "max_team_share": cfg.max_team_share,
            "max_game_date_share": cfg.max_game_date_share,
            "history_lookback_days": cfg.history_lookback_days,
            "min_history_rows": cfg.min_history_rows,
            "min_history_priced_rate": cfg.min_history_priced_rate,
            "min_history_roi": cfg.min_history_roi,
            "max_history_abs_market_calibration_error": cfg.max_history_abs_market_calibration_error,
            "enforce_history_guard": cfg.enforce_history_guard,
            "force_reopen_all": cfg.force_reopen_all,
            "research_only": cfg.research_only,
        },
        "rows": int(len(df)),
        "history_rows": int(len(history_df)),
        "force_reopen_all": cfg.force_reopen_all,
        "research_only": cfg.research_only,
        "reopen_buckets": {},
        "closed_buckets": {},
        "diagnostics": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
            reasons = _policy_reasons(train_summary, holdout_summary, cfg)
            history_summary = history_by_level.get(
                (level, _bucket_key(value_dict)),
                _history_summary(pd.DataFrame()),
            )
            history_reasons = _history_guard_reasons(history_summary, cfg)
            if cfg.enforce_history_guard:
                reasons.extend(history_reasons)
            status = "forced_reopen" if cfg.force_reopen_all else "reopen_candidate" if not reasons else "closed"
            key = _bucket_key(value_dict)
            record = {
                "key": key,
                "level": level,
                "market": value_dict.get("market", "*"),
                "side": value_dict.get("side", "*"),
                "line_bucket": value_dict.get("line_bucket", "*"),
                "price_bucket": value_dict.get("price_bucket", "*"),
                "status": status,
                "reasons": reasons,
                "model_reasons": reasons,
                "history_reasons": history_reasons,
                "train": train_summary,
                "holdout": holdout_summary,
                "history": history_summary,
            }
            payload["diagnostics"].append(record)
            if level == "bucket":
                if status in {"reopen_candidate", "forced_reopen"}:
                    payload["reopen_buckets"][key] = {
                        "market": record["market"],
                        "side": record["side"],
                        "line_bucket": record["line_bucket"],
                        "price_bucket": record["price_bucket"],
                        "status": status,
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
                        "holdout_unique_players": holdout_summary["unique_players"],
                        "holdout_unique_teams": holdout_summary["unique_teams"],
                        "holdout_unique_dates": holdout_summary["unique_dates"],
                        "holdout_max_player_share": holdout_summary["max_player_share"],
                        "holdout_max_team_share": holdout_summary["max_team_share"],
                        "holdout_max_game_date_share": holdout_summary["max_game_date_share"],
                        "history_rows": history_summary["rows"],
                        "history_roi": history_summary["roi"],
                        "history_win_rate": history_summary["win_rate"],
                        "history_avg_market_prob": history_summary["avg_market_prob"],
                        "history_market_calibration_error": history_summary["market_calibration_error"],
                        "model_reasons": reasons,
                        "history_reasons": history_reasons,
                    }
                else:
                    payload["closed_buckets"][key] = reasons

    payload["diagnostics"].sort(key=lambda r: (r["level"], r["market"], r["side"], r["line_bucket"], r["price_bucket"]))
    if cfg.force_reopen_all:
        payload["status"] = "forced_reopen_all_research" if cfg.research_only else "forced_reopen_all"
    else:
        payload["status"] = "ready_for_bucket_review" if payload["reopen_buckets"] else "all_buckets_closed"
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop bucket reopen policy")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_bucket_reopen_policy.json")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
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
    parser.add_argument("--max-player-share", type=float, default=0.12)
    parser.add_argument("--max-team-share", type=float, default=0.25)
    parser.add_argument("--max-game-date-share", type=float, default=0.35)
    parser.add_argument("--history-lookback-days", type=int, default=540)
    parser.add_argument("--min-history-rows", type=int, default=500)
    parser.add_argument("--min-history-priced-rate", type=float, default=0.80)
    parser.add_argument("--min-history-roi", type=float, default=-0.15)
    parser.add_argument("--max-history-abs-market-calibration-error", type=float, default=0.12)
    parser.add_argument("--enforce-history-guard", action="store_true")
    parser.add_argument("--force-reopen-all", action="store_true")
    parser.add_argument("--research-only", action="store_true")
    args = parser.parse_args()
    payload = train(PropBucketReopenConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
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
        max_player_share=args.max_player_share,
        max_team_share=args.max_team_share,
        max_game_date_share=args.max_game_date_share,
        history_lookback_days=args.history_lookback_days,
        min_history_rows=args.min_history_rows,
        min_history_priced_rate=args.min_history_priced_rate,
        min_history_roi=args.min_history_roi,
        max_history_abs_market_calibration_error=args.max_history_abs_market_calibration_error,
        enforce_history_guard=args.enforce_history_guard,
        force_reopen_all=args.force_reopen_all,
        research_only=args.research_only,
    ))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
