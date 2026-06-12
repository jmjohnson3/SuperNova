"""Train the MLB prop betting layer.

This is separate from projection models.  Projection models estimate outcomes;
this layer estimates whether a specific offered side is mispriced versus the
market after seeing line, price, model family, and raw model probability.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .prop_betting_layer import _CATEGORICAL_FEATURES, _NUMERIC_FEATURES, betting_layer_key
from .prop_replay import american_to_prob, ensure_prop_replay_schema, no_vig_probs
from .side_recalibration import price_bucket, prop_line_bucket

log = logging.getLogger("mlb_pipeline.modeling.train_prop_betting_layer")

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"


@dataclass(frozen=True)
class BettingLayerConfig:
    pg_dsn: str = _PG_DSN
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_betting_layer.json"
    run_id: str | None = None
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 180
    min_holdout_rows: int = 30


_REPLAY_SQL = """
SELECT
    game_date_et,
    stat AS market,
    market_line::float AS market_line,
    over_price::float AS over_price,
    under_price::float AS under_price,
    model_prob_over::float AS model_prob_over,
    COALESCE(model_family, 'unknown') AS model_family,
    line_bucket,
    over_hit
FROM bets.mlb_prop_prediction_replay
WHERE game_date_et >= %(cutoff)s
  AND (%(run_id)s IS NULL OR run_id = %(run_id)s)
  AND stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND model_prob_over IS NOT NULL
  AND market_line IS NOT NULL
  AND over_hit IS NOT NULL
"""

_LIVE_SQL = """
SELECT
    game_date_et,
    stat AS market,
    book_line::float AS market_line,
    over_price::float AS over_price,
    under_price::float AS under_price,
    pred_prob_over::float AS model_prob_over,
    COALESCE(model_family, 'unknown') AS model_family,
    line_bucket,
    over_hit
FROM bets.mlb_prop_predictions
WHERE game_date_et >= %(cutoff)s
  AND stat IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND pred_prob_over IS NOT NULL
  AND book_line IS NOT NULL
  AND over_hit IS NOT NULL
"""

_MARKET_TABLE_SQL = """
SELECT
    game_date_et,
    market,
    side,
    COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
    COALESCE(line_surface, 'unknown') AS line_surface,
    line_bucket,
    price_bucket,
    COALESCE(model_family, 'unknown') AS model_family,
    model_prob_side::float AS raw_p_side,
    market_prob_side::float AS market_prob_side,
    prob_edge_vs_market::float AS prob_edge_vs_market,
    market_line::float AS market_line,
    confirmed_batting_order::float AS confirmed_batting_order,
    projected_pa::float AS projected_pa,
    projected_bf::float AS projected_bf,
    projected_pitch_count::float AS projected_pitch_count,
    is_home::float AS is_home,
    team_implied_runs::float AS team_implied_runs,
    opponent_implied_runs::float AS opponent_implied_runs,
    game_total_line::float AS game_total_line,
    opp_sp_hand,
    opp_sp_k_pct_10::float AS opp_sp_k_pct_10,
    opp_sp_bb_pct::float AS opp_sp_bb_pct,
    opp_sp_xwoba::float AS opp_sp_xwoba,
    opp_sp_hard_hit_pct::float AS opp_sp_hard_hit_pct,
    opp_sp_whiff_pct::float AS opp_sp_whiff_pct,
    opp_bp_era_10::float AS opp_bp_era_10,
    opp_bp_whip_10::float AS opp_bp_whip_10,
    opp_bp_k9_10::float AS opp_bp_k9_10,
    opp_team_k_pct_10::float AS opp_team_k_pct_10,
    batter_vs_hand_hits_avg_10::float AS batter_vs_hand_hits_avg_10,
    batter_vs_hand_tb_avg_10::float AS batter_vs_hand_tb_avg_10,
    batter_vs_hand_hr_avg_10::float AS batter_vs_hand_hr_avg_10,
    batter_vs_hand_iso_avg_10::float AS batter_vs_hand_iso_avg_10,
    batter_vs_hand_k_rate_10::float AS batter_vs_hand_k_rate_10,
    batter_vs_rp_slg_30::float AS batter_vs_rp_slg_30,
    batter_vs_rp_hr_rate_30::float AS batter_vs_rp_hr_rate_30,
    pinch_hit_risk::float AS pinch_hit_risk,
    ABS(market_price::float) AS abs_price,
    CASE
        WHEN market_price IS NULL THEN NULL
        WHEN market_price::float > 0 THEN 1.0
        ELSE 0.0
    END AS is_plus_price,
    CASE
        WHEN won IS TRUE THEN 1
        WHEN won IS FALSE THEN 0
        ELSE NULL
    END AS target,
    CASE
        WHEN beat_clv_price IS TRUE THEN 1
        WHEN beat_clv_price IS FALSE THEN 0
        ELSE NULL
    END AS clv_target
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND (%(run_id)s IS NULL OR run_id = %(run_id)s)
  AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
  AND model_prob_side IS NOT NULL
  AND market_line IS NOT NULL
  AND model_prob_side BETWEEN 0.0 AND 1.0
  AND (won IS NOT NULL OR beat_clv_price IS NOT NULL)
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


def _read_sql(conn, sql: str, params: dict) -> pd.DataFrame:
    return pd.read_sql(sql, conn, params=params)


def _load_base_rows(cfg: BettingLayerConfig) -> tuple[pd.DataFrame, str]:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        source = "live_predictions"
        df = pd.DataFrame()
        if _table_exists(conn, "bets", "mlb_prop_prediction_replay"):
            df = _read_sql(conn, _REPLAY_SQL, {"cutoff": cutoff, "run_id": cfg.run_id})
            if not df.empty:
                source = "replay"
        if df.empty:
            df = _read_sql(conn, _LIVE_SQL, {"cutoff": cutoff})
    if df.empty:
        return df, source
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    return df, source


def _load_side_rows(cfg: BettingLayerConfig) -> tuple[pd.DataFrame, str, int]:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)).isoformat()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            df = _read_sql(conn, _MARKET_TABLE_SQL, {"cutoff": cutoff, "run_id": cfg.run_id})
            if not df.empty:
                df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
                df = df.replace([np.inf, -np.inf], np.nan)
                df["target"] = pd.to_numeric(df["target"], errors="coerce")
                df["clv_target"] = pd.to_numeric(df["clv_target"], errors="coerce")
                return df, "market_training_table", int(len(df) // 2)
    base, source = _load_base_rows(cfg)
    return _expand_sides(base), source, int(len(base))


def _expand_sides(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base
    rows = []
    for _, r in base.iterrows():
        p_over = float(r["model_prob_over"])
        over_price = r.get("over_price")
        under_price = r.get("under_price")
        nv_over, nv_under = no_vig_probs(over_price, under_price)
        line = float(r["market_line"])
        market = str(r["market"])
        lb = r.get("line_bucket")
        if not isinstance(lb, str) or not lb or lb == "unknown":
            lb = prop_line_bucket(market, line)
        for side in ("over", "under"):
            p_side = p_over if side == "over" else 1.0 - p_over
            price = over_price if side == "over" else under_price
            market_prob = nv_over if side == "over" else nv_under
            if market_prob is None:
                market_prob = american_to_prob(price)
            target = bool(r["over_hit"]) if side == "over" else (not bool(r["over_hit"]))
            rows.append({
                "game_date_et": r["game_date_et"],
                "market": market,
                "side": side,
                "line_bucket": lb,
                "price_bucket": price_bucket(price),
                "model_family": str(r.get("model_family") or "unknown"),
                "raw_p_side": max(1e-6, min(1.0 - 1e-6, p_side)),
                "market_prob_side": market_prob,
                "prob_edge_vs_market": (p_side - market_prob) if market_prob is not None else np.nan,
                "market_line": line,
                "abs_price": abs(float(price)) if price is not None and pd.notna(price) else np.nan,
                "is_plus_price": 1.0 if price is not None and pd.notna(price) and float(price) > 0 else 0.0 if price is not None and pd.notna(price) else np.nan,
                "target": int(target),
            })
    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["raw_p_side", "target"])
    return out


def _prepare_matrix(df: pd.DataFrame, *, means=None, scales=None, cats=None):
    means = dict(means or {})
    scales = dict(scales or {})
    cats = {k: list(v) for k, v in (cats or {}).items()}
    parts = []
    feature_names = []
    numeric_means = {}
    numeric_scales = {}
    for name in _NUMERIC_FEATURES:
        s = pd.to_numeric(df.get(name), errors="coerce")
        mean = float(means.get(name, s.mean() if not s.dropna().empty else 0.0))
        filled = s.fillna(mean)
        scale = float(scales.get(name, filled.std(ddof=0) if float(filled.std(ddof=0) or 0.0) > 1e-9 else 1.0))
        if scale <= 1e-9:
            scale = 1.0
        parts.append(((filled - mean) / scale).to_numpy(dtype=float).reshape(-1, 1))
        feature_names.append(name)
        numeric_means[name] = mean
        numeric_scales[name] = scale
    cat_values = {}
    for name in _CATEGORICAL_FEATURES:
        values = cats.get(name)
        base_series = (
            df[name]
            if name in df.columns
            else pd.Series(["unknown"] * len(df), index=df.index)
        )
        if values is None:
            values = sorted(str(v) for v in base_series.fillna("unknown").astype(str).unique())
        cat_values[name] = values
        series = base_series.fillna("unknown").astype(str)
        for value in values:
            parts.append((series == value).astype(float).to_numpy().reshape(-1, 1))
            feature_names.append(f"{name}={value}")
    X = np.hstack(parts) if parts else np.zeros((len(df), 0))
    return X, feature_names, numeric_means, numeric_scales, cat_values


def _fit_group(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    all_rows: pd.DataFrame,
    *,
    baseline_col: str | None = "raw_p_side",
) -> dict | None:
    if len(train) < 40 or len(holdout) < 8:
        return None
    if train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        return None
    X_train, feature_names, means, scales, cats = _prepare_matrix(train)
    y_train = train["target"].astype(int).to_numpy()
    lr = LogisticRegression(max_iter=3000, solver="lbfgs")
    lr.fit(X_train, y_train)

    X_hold, _, _, _, _ = _prepare_matrix(holdout, means=means, scales=scales, cats=cats)
    y_hold = holdout["target"].astype(int).to_numpy()
    if baseline_col:
        raw = holdout[baseline_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
        baseline_kind = baseline_col
        baseline_rate = None
    else:
        baseline_rate = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
        raw = np.repeat(baseline_rate, len(holdout))
        baseline_kind = "train_target_rate"
    pred_hold = np.clip(lr.predict_proba(X_hold)[:, 1], 1e-6, 1 - 1e-6)

    raw_brier = float(brier_score_loss(y_hold, raw))
    model_brier = float(brier_score_loss(y_hold, pred_hold))
    improvement = raw_brier - model_brier
    base_weight = min(0.85, len(holdout) / (len(holdout) + 160.0))
    if improvement < -0.006:
        blend = base_weight * 0.15
    elif improvement < 0.001:
        blend = base_weight * 0.45
    else:
        blend = base_weight

    X_all, feature_names, means, scales, cats = _prepare_matrix(all_rows)
    y_all = all_rows["target"].astype(int).to_numpy()
    final = LogisticRegression(max_iter=3000, solver="lbfgs")
    final.fit(X_all, y_all)
    coef = {
        name: float(value)
        for name, value in zip(feature_names, final.coef_[0])
        if abs(float(value)) > 1e-12
    }
    return {
        "method": "logistic_betting_layer",
        "intercept": float(final.intercept_[0]),
        "coef": coef,
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_values": cats,
        "blend_weight": float(blend),
        "baseline_kind": baseline_kind,
        "baseline_rate_train": baseline_rate,
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "actual_rate_holdout": float(np.mean(y_hold)),
        "avg_raw_holdout": float(np.mean(raw)),
        "avg_model_holdout": float(np.mean(pred_hold)),
        "brier_raw_holdout": raw_brier,
        "brier_model_holdout": model_brier,
        "log_loss_raw_holdout": float(log_loss(y_hold, raw, labels=[0, 1])),
        "log_loss_model_holdout": float(log_loss(y_hold, pred_hold, labels=[0, 1])),
        "auc_raw_holdout": float(roc_auc_score(y_hold, raw)) if len(np.unique(y_hold)) == 2 else None,
        "auc_model_holdout": float(roc_auc_score(y_hold, pred_hold)) if len(np.unique(y_hold)) == 2 else None,
    }


def _group_specs() -> Iterable[tuple[tuple[str, ...], int, int]]:
    return [
        (("market", "side", "line_bucket", "model_family"), 140, 25),
        (("market", "side", "line_bucket"), 160, 30),
        (("market", "side", "model_family"), 180, 30),
        (("market", "side"), 180, 30),
        (("side", "model_family"), 220, 40),
        (("side",), 240, 45),
        (tuple(), 300, 60),
    ]


def _key(group_cols: tuple[str, ...], values) -> str:
    if not group_cols:
        return betting_layer_key("*", "*", "*", "*")
    values = values if isinstance(values, tuple) else (values,)
    d = dict(zip(group_cols, values))
    return betting_layer_key(
        d.get("market", "*"),
        d.get("side", "*"),
        d.get("line_bucket", "*"),
        d.get("model_family", "*"),
    )


def train(cfg: BettingLayerConfig) -> dict:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_prop_replay_schema(conn)
    df, source, base_rows = _load_side_rows(cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": source,
        "run_id": cfg.run_id,
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "base_rows": int(base_rows),
        "side_rows": int(len(df)),
        "models": {},
        "backtest": [],
        "clv_models": {},
        "clv_backtest": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _train_target_models(
        source_df: pd.DataFrame,
        target_col: str,
        *,
        baseline_col: str | None = "raw_p_side",
    ) -> tuple[dict, list[dict], int]:
        work = source_df.copy()
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work.dropna(subset=[target_col])
        if work.empty:
            return {}, [], 0
        work["target"] = work[target_col].astype(int)
        split = max(work["game_date_et"]) - timedelta(days=cfg.holdout_days)
        train_mask = work["game_date_et"] < split
        holdout_mask = work["game_date_et"] >= split
        models: dict[str, dict] = {}
        backtest: list[dict] = []
        for group_cols, min_train, min_holdout in _group_specs():
            grouped = [((), work)] if not group_cols else list(work.groupby(list(group_cols), dropna=False))
            for values, sub in grouped:
                tr = sub.loc[train_mask.loc[sub.index]]
                ho = sub.loc[holdout_mask.loc[sub.index]]
                if len(tr) < min(min_train, cfg.min_train_rows) or len(ho) < min(min_holdout, cfg.min_holdout_rows):
                    continue
                rec = _fit_group(tr, ho, sub, baseline_col=baseline_col)
                if rec is None:
                    continue
                key = _key(group_cols, values)
                rec["key"] = key
                rec["group_columns"] = list(group_cols)
                rec["target"] = target_col
                models[key] = rec
                backtest.append({
                    "key": key,
                    "target": target_col,
                    "train_rows": rec["train_rows"],
                    "holdout_rows": rec["holdout_rows"],
                    "actual_rate_holdout": rec["actual_rate_holdout"],
                    "avg_raw_holdout": rec["avg_raw_holdout"],
                    "avg_model_holdout": rec["avg_model_holdout"],
                    "brier_raw_holdout": rec["brier_raw_holdout"],
                    "brier_model_holdout": rec["brier_model_holdout"],
                    "blend_weight": rec["blend_weight"],
                })
        return models, backtest, len(work)

    win_models, win_backtest, win_rows = _train_target_models(df, "target")
    clv_models, clv_backtest, clv_rows = _train_target_models(df, "clv_target", baseline_col=None)
    payload["models"] = win_models
    payload["backtest"] = sorted(win_backtest, key=lambda r: r["key"])
    payload["clv_models"] = clv_models
    payload["clv_backtest"] = sorted(clv_backtest, key=lambda r: r["key"])
    payload["win_rows"] = int(win_rows)
    payload["clv_rows"] = int(clv_rows)
    payload["status"] = "trained" if payload["models"] else "no_models"
    payload["clv_status"] = "trained" if payload["clv_models"] else "no_clv_models"
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB prop betting layer")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_betting_layer.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--min-train-rows", type=int, default=180)
    parser.add_argument("--min-holdout-rows", type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    payload = train(BettingLayerConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        run_id=args.run_id,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
    ))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
