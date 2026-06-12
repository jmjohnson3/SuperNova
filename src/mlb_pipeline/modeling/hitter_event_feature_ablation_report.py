"""Feature ablation report for hitter opportunity and event models.

The production hitter event artifact tells us whether the current feature set
works.  This report answers the more useful repair question: which feature
families actually move holdout PA MAE, event Brier/log loss, and hit/TB/HR MAE.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train_hitter_player_game_outcome_models import (
    CATEGORICAL_FEATURES,
    EVENT_CLASSES,
    HitterOutcomeModelConfig,
    _build_event_training_examples,
    _count_metrics,
    _event_count_matrix,
    _event_log_loss_from_counts,
    _event_projection_metrics,
    _classifier_pipeline,
    _load,
    _regression_pipeline,
    _split,
    prepare_hitter_outcome_features,
)

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

BASELINE_NUMERIC = [
    "lineup_slot",
    "confirmed_starter_num",
    "is_home",
    "team_implied_runs",
    "opponent_implied_runs",
    "game_total_line",
    "opp_sp_hand_l",
    "opp_sp_k_pct_10",
    "opp_sp_bb_pct",
    "opp_sp_xwoba",
    "opp_sp_hard_hit_pct",
    "opp_sp_whiff_pct",
    "opp_bp_era_10",
    "opp_bp_whip_10",
    "opp_bp_k9_10",
    "opp_bp_ip_last_3",
    "opp_bp_ip_last_7",
    "opp_team_k_pct_10",
    "opp_team_avg_10",
    "opp_team_obp_10",
    "opp_team_slg_10",
    "batter_vs_hand_hits_avg_10",
    "batter_vs_hand_tb_avg_10",
    "batter_vs_hand_hr_avg_10",
    "batter_vs_hand_iso_avg_10",
    "batter_vs_hand_k_rate_10",
    "batter_vs_hand_games_10",
    "batter_vs_rp_ba_30",
    "batter_vs_rp_slg_30",
    "batter_vs_rp_hr_rate_30",
    "batter_vs_rp_k_rate_30",
    "projected_pa",
    "pa_games",
]

FEATURE_GROUPS: list[tuple[str, list[str]]] = [
    ("baseline", []),
    ("+lineup", [
        "own_lineup_xwoba_avg",
        "own_lineup_xslg_avg",
        "own_lineup_barrel_avg",
        "own_lineup_hard_hit_avg",
        "own_lineup_k_pct_cv",
        "own_lineup_pct_lhb",
        "lineup_confirmed_flag",
        "confirmed_team_lineup_slots",
        "team_lineup_confirmed_flag",
        "lineup_boxscore_proxy_flag",
        "lineup_slot_x_team_implied_runs",
    ]),
    ("+park", [
        "park_run_factor",
        "park_hr_factor",
        "park_babip_factor",
    ]),
    ("+batter_statcast", [
        "batter_sc_barrel_rate",
        "batter_sc_hard_hit_pct",
        "batter_sc_avg_exit_velo",
        "batter_sc_avg_launch_angle",
        "batter_sc_sweet_spot_pct",
        "batter_sc_fb_pct",
        "batter_sc_gb_pct",
        "batter_sc_ld_pct",
        "batter_sc_xba",
        "batter_sc_xslg",
        "batter_sc_xwoba",
        "batter_sc_xiso",
        "batter_sc_brl_pa",
        "batter_sprint_speed",
    ]),
    ("+pitcher_statcast", [
        "opp_sp_sc_barrel_rate",
        "opp_sp_sc_hard_hit_pct",
        "opp_sp_sc_avg_exit_velo",
        "opp_sp_sc_avg_launch_angle",
        "opp_sp_sc_xba",
        "opp_sp_sc_xslg",
        "opp_sp_sc_xwoba",
        "opp_sp_sc_xiso",
        "opp_sp_fb_pct",
        "opp_sp_fb_hard_hit_pct",
        "opp_sp_fb_xwoba",
        "opp_sp_fb_run_value_per_100",
        "opp_sp_fb_whiff_pct",
        "opp_sp_fb_k_pct",
        "opp_sp_si_pct",
        "opp_sp_si_hard_hit_pct",
        "opp_sp_si_xwoba",
        "opp_sp_si_whiff_pct",
        "opp_sp_si_k_pct",
        "opp_sp_sl_pct",
        "opp_sp_sl_hard_hit_pct",
        "opp_sp_sl_xwoba",
        "opp_sp_sl_run_value_per_100",
        "opp_sp_sl_whiff_pct",
        "opp_sp_sl_k_pct",
        "opp_sp_ch_pct",
        "opp_sp_ch_hard_hit_pct",
        "opp_sp_ch_xwoba",
        "opp_sp_ch_run_value_per_100",
        "opp_sp_ch_whiff_pct",
        "opp_sp_ch_k_pct",
        "opp_sp_fastball_family_pct",
        "opp_sp_pitch_diversity",
    ]),
    ("+discipline", [
        "batter_disc_oz_swing_pct",
        "batter_disc_iz_contact_pct",
        "batter_disc_oz_contact_pct",
        "batter_disc_whiff_pct",
        "batter_disc_out_zone_pct",
        "batter_disc_k_pct",
        "batter_disc_bb_pct",
    ]),
    ("+market", [
        "model_pred_hits",
        "model_pred_total_bases",
        "model_pred_home_runs",
        "prop_example_rows",
        "prop_offer_rows",
        "paired_price_rate",
        "same_book_pair_rate",
    ]),
]


@dataclass(frozen=True)
class FeatureAblationConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    model_dir: Path = _MODEL_DIR
    report_file: str = "mlb_hitter_event_feature_ablation_latest.md"
    out_file: str = "hitter_event_feature_ablation.json"
    lookback_days: int = 365
    holdout_days: int = 28
    min_train_rows: int = 1000
    min_holdout_rows: int = 200


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _weighted_multiclass_brier(counts: pd.DataFrame, probs: pd.DataFrame) -> float | None:
    total = float(counts[EVENT_CLASSES].sum(axis=1).sum())
    if total <= 0:
        return None
    loss = 0.0
    p_mat = np.column_stack([
        pd.to_numeric(probs[f"p_{cls}"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for cls in EVENT_CLASSES
    ])
    p_sq = np.sum(p_mat * p_mat, axis=1)
    for i, cls in enumerate(EVENT_CLASSES):
        w = pd.to_numeric(counts[cls], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        loss += float(np.sum(w * (p_sq - 2.0 * p_mat[:, i] + 1.0)))
    return loss / total


def _evaluate(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    train_features = prepare_hitter_outcome_features(train, numeric_features, categorical_features)
    holdout_features = prepare_hitter_outcome_features(holdout, numeric_features, categorical_features)
    pa_model = _regression_pipeline(10.0, numeric_features, categorical_features)
    pa_model.fit(train_features[numeric_features + categorical_features], train["actual_pa"])
    pa_pred = np.clip(pa_model.predict(holdout_features[numeric_features + categorical_features]), 0.0, 6.5)

    event_train = train[train["actual_pa"] > 0].copy()
    event_holdout = holdout[holdout["actual_pa"] > 0].copy()
    X_event, y_event, w_event = _build_event_training_examples(event_train, numeric_features, categorical_features)
    event_payload: dict[str, Any] = {"train_event_rows": int(len(y_event)), "holdout_rows": int(len(event_holdout))}
    if len(y_event) > 0 and len(np.unique(y_event)) >= 4 and len(event_holdout) > 0:
        event_model = _classifier_pipeline(numeric_features, categorical_features)
        event_model.fit(X_event[numeric_features + categorical_features], y_event, model__sample_weight=w_event)
        event_features = prepare_hitter_outcome_features(event_holdout, numeric_features, categorical_features)
        raw = event_model.predict_proba(event_features[numeric_features + categorical_features])
        probs = pd.DataFrame(0.0, index=event_holdout.index, columns=[f"p_{cls}" for cls in EVENT_CLASSES])
        for i, cls in enumerate(event_model.classes_):
            if cls in EVENT_CLASSES:
                probs[f"p_{cls}"] = raw[:, i]
        probs = probs.div(probs.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / len(EVENT_CLASSES))
        event_pa_pred = pd.Series(pa_pred, index=holdout.index).loc[event_holdout.index].to_numpy(dtype=float)
        counts = _event_count_matrix(event_holdout)
        event_payload.update(_event_projection_metrics(event_holdout, probs, event_pa_pred))
        event_payload["weighted_event_brier"] = _weighted_multiclass_brier(counts, probs)
        event_payload["weighted_event_log_loss"] = _event_log_loss_from_counts(counts, probs)
    return {
        "pa": _count_metrics(holdout["actual_pa"], pa_pred),
        "event": event_payload,
    }


def build_report(cfg: FeatureAblationConfig) -> dict[str, Any]:
    df = _load(HitterOutcomeModelConfig(
        pg_dsn=cfg.pg_dsn,
        lookback_days=cfg.lookback_days,
        holdout_days=cfg.holdout_days,
        min_train_rows=cfg.min_train_rows,
        min_holdout_rows=cfg.min_holdout_rows,
    ))
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "features.mlb_hitter_player_game_training",
        "rows": int(len(df)),
        "lookback_days": cfg.lookback_days,
        "holdout_days": cfg.holdout_days,
        "status": "ok",
        "ablation": [],
    }
    if df.empty:
        payload["status"] = "no_rows"
        _write_outputs(payload, cfg)
        return payload
    train, holdout = _split(df, cfg.holdout_days)
    payload["train_rows"] = int(len(train))
    payload["holdout_rows"] = int(len(holdout))
    if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
        payload["status"] = "insufficient_rows"
        _write_outputs(payload, cfg)
        return payload
    numeric: list[str] = []
    previous: dict[str, Any] | None = None
    for label, additions in FEATURE_GROUPS:
        if label == "baseline":
            numeric = list(BASELINE_NUMERIC)
        else:
            numeric = list(dict.fromkeys([*numeric, *additions]))
        usable_numeric = [
            col for col in numeric
            if col in train.columns and pd.to_numeric(train[col], errors="coerce").notna().any()
        ]
        dropped_numeric = [col for col in numeric if col not in usable_numeric]
        rec = {
            "feature_set": label,
            "numeric_features": list(usable_numeric),
            "dropped_all_null_numeric_features": dropped_numeric,
            "metrics": _evaluate(train, holdout, usable_numeric, CATEGORICAL_FEATURES),
        }
        if previous:
            cur_pa = ((rec["metrics"].get("pa") or {}).get("mae"))
            prev_pa = (((previous.get("metrics") or {}).get("pa") or {}).get("mae"))
            cur_brier = (((rec["metrics"].get("event") or {}).get("weighted_event_brier")))
            prev_brier = ((((previous.get("metrics") or {}).get("event") or {}).get("weighted_event_brier")))
            rec["delta_vs_previous"] = {
                "pa_mae_gain": (float(prev_pa) - float(cur_pa)) if prev_pa is not None and cur_pa is not None else None,
                "event_brier_gain": (
                    float(prev_brier) - float(cur_brier)
                    if prev_brier is not None and cur_brier is not None
                    else None
                ),
            }
        payload["ablation"].append(rec)
        previous = rec
    _write_outputs(payload, cfg)
    return payload


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _write_outputs(payload: dict[str, Any], cfg: FeatureAblationConfig) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    path = _REPORT_DIR / cfg.report_file
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MLB Hitter Event Feature Ablation",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Rows: {payload.get('rows', 0)} | Train: {payload.get('train_rows', 0)} | Holdout: {payload.get('holdout_rows', 0)}",
        f"Status: {payload.get('status')}",
        "",
        "The `+market` row is diagnostic context only; bankroll gating should still rely on locked offer-level reports.",
        "",
        "| Feature Set | PA MAE | Event Brier | Event Log Loss | Hits MAE | TB MAE | HR MAE | PA Gain | Brier Gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in payload.get("ablation", []):
        metrics = rec.get("metrics") or {}
        pa = metrics.get("pa") or {}
        event = metrics.get("event") or {}
        delta = rec.get("delta_vs_previous") or {}
        lines.append(
            f"| {rec.get('feature_set')} | {_fmt(pa.get('mae'), 3)} | "
            f"{_fmt(event.get('weighted_event_brier'), 5)} | {_fmt(event.get('weighted_event_log_loss'), 5)} | "
            f"{_fmt((event.get('hits') or {}).get('mae'), 3)} | "
            f"{_fmt((event.get('total_bases') or {}).get('mae'), 3)} | "
            f"{_fmt((event.get('home_runs') or {}).get('mae'), 3)} | "
            f"{_fmt(delta.get('pa_mae_gain'), 4)} | {_fmt(delta.get('event_brier_gain'), 5)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hitter event feature ablation report")
    parser.add_argument("--pg-dsn", default=FeatureAblationConfig.pg_dsn)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--report-file", default="mlb_hitter_event_feature_ablation_latest.md")
    parser.add_argument("--out-file", default="hitter_event_feature_ablation.json")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    payload = build_report(FeatureAblationConfig(
        pg_dsn=args.pg_dsn,
        model_dir=Path(args.model_dir),
        report_file=args.report_file,
        out_file=args.out_file,
        lookback_days=args.lookback_days,
        holdout_days=args.holdout_days,
    ))
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
