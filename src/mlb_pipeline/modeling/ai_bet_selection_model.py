"""Train and score a learned MLB prop bet-selection model.

The projection layer answers "what will the player do?"  This model learns
"when should we trust that projection against this exact offered line/price?"
from historical locked, graded offer-level rows.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psycopg2
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import joblib
except Exception:  # pragma: no cover - deployed env has joblib via sklearn stack
    joblib = None  # type: ignore

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text, atomic_write_via
from mlb_pipeline.db import PG_DSN

from .prop_replay import american_to_prob, ev_per_unit
from .prop_training_groups import (
    add_player_game_weights,
    dedupe_locked_offer_rows,
    expanding_player_game_folds,
    grouping_summary,
    sample_weights,
    temporal_player_game_split,
)

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_ARTIFACT_FILE = "ai_bet_selection_model.joblib"
_META_FILE = "ai_bet_selection_model.json"
_REPORT_FILE = "mlb_ai_bet_selection_model_latest.md"

_MARKETS = (
    "pitcher_strikeouts",
    "batter_hits",
    "batter_hits_runs_rbis",
    "batter_total_bases",
    "batter_home_runs",
)

_TARGET_WIN = "target_win"
_TARGET_GOOD_BET = "target_good_bet"
_TARGET_CLV = "target_clv"
_TARGET_AVAILABLE = "target_line_available_at_close"
_TARGET_CLOSE_CAPTURE = "target_valid_close_snapshot_captured"
_TARGET_DAILY_RANK = "target_daily_top_pick"

_PRIMARY_TARGETS = (
    _TARGET_GOOD_BET,
    _TARGET_WIN,
    _TARGET_CLV,
    _TARGET_AVAILABLE,
    _TARGET_DAILY_RANK,
)

_TARGET_PAYLOAD_KEYS = {
    _TARGET_GOOD_BET: "good_bet",
    _TARGET_WIN: "win",
    _TARGET_CLV: "clv",
    _TARGET_AVAILABLE: "line_available",
    _TARGET_CLOSE_CAPTURE: "close_capture",
    _TARGET_DAILY_RANK: "daily_rank",
}

_TARGET_MODEL_KEYS = {
    _TARGET_GOOD_BET: "good_bet_model",
    _TARGET_WIN: "win_model",
    _TARGET_CLV: "clv_model",
    _TARGET_AVAILABLE: "line_available_model",
    _TARGET_CLOSE_CAPTURE: "close_capture_model",
    _TARGET_DAILY_RANK: "daily_rank_model",
}

_VALID_CLOSE_STATUSES = {"valid_movement", "true_no_movement"}
_UNAVAILABLE_CLOSE_REASONS = {
    "line_disappeared_at_close",
    "player_market_unavailable_at_close",
    "player_prop_unavailable_at_close",
}
_CAPTURE_FAILURE_REASONS = {
    "close_outside_two_hour_window",
    "stale_close_before_lock",
    "fallback_other_book_only",
    "no_valid_close_snapshot",
}

_NUMERIC_FEATURES = [
    "model_prob_side",
    "market_prob_side",
    "raw_market_prob",
    "no_vig_market_prob",
    "prob_edge_vs_market",
    "model_market_gap_abs",
    "model_market_gap_signed",
    "market_price_prob_gap",
    "no_vig_prob_gap",
    "market_line",
    "market_price",
    "price_implied_prob",
    "paired_price",
    "paired_price_implied_prob",
    "pred_value",
    "pred_count",
    "count_edge_side",
    "abs_count_edge_side",
    "ev",
    "model_confidence",
    "plus_money_flag",
    "favorite_price_flag",
    "book_is_fanduel",
    "book_is_draftkings",
    "line_tb15_flag",
    "line_hits05_flag",
    "line_hr05_flag",
    "line_k_common_flag",
    "same_book_pair_flag",
    "cross_book_pair_flag",
    "synthetic_pair_flag",
    "clean_market_pair_flag",
    "true_pair_flag",
    "minutes_to_first_pitch_at_lock",
    "lock_price_age_minutes",
    "confirmed_batting_order",
    "projected_pa",
    "pa_low_probability",
    "pa_normal_mean",
    "pa_games",
    "projected_ip",
    "projected_bf",
    "projected_pitch_count",
    "pitcher_starts",
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
    "pinch_hit_risk",
    "clv_beat_prob",
    "expected_clv_price",
    "residual_clv_beat_prob",
    "event_side_line_clv_beat_prob",
    "bookable_prob",
    "line_available_prob",
    "close_capture_prob",
    "bucket_roi",
    "bucket_clv_beat_rate",
    "bucket_avg_clv",
    "micro_clv_beat_prob",
    "exact_bucket_clv_beat_prob",
    "exact_bucket_avg_clv",
    "micro_truth_filter_roi",
    "micro_truth_filter_graded",
    "micro_truth_filter_clv_beat_rate",
    "micro_truth_filter_avg_clv",
    "external_agreement_count",
    "external_disagreement_count",
    "external_agreement_strength",
]

_CATEGORICAL_FEATURES = [
    "market",
    "side",
    "bookmaker_key",
    "source",
    "paired_bookmaker_key",
    "paired_price_source",
    "pair_quality",
    "market_prob_source",
    "ai_market_family",
    "book_market_side",
    "clean_evidence_tier",
    "fanduel_evidence_tier",
    "price_bucket",
    "line_bucket",
    "line_surface",
    "model_family",
    "edge_type",
    "confirmed_lineup_source",
    "team_abbr",
    "opponent_abbr",
    "opp_sp_hand",
    "external_match_level",
    "clv_status",
    "clv_unknown_reason",
]


@dataclass(frozen=True)
class TrainConfig:
    lookback_days: int = 120
    min_rows: int = 500
    min_train_rows: int = 300
    min_holdout_rows: int = 80
    min_train_dates: int = 5
    fold_days: int = 5
    max_folds: int = 6
    min_brier_gain: float = 0.001
    include_gbm: bool = False
    max_rows: int = 0
    max_market_families: int = 10
    family_min_rows: int = 180
    family_min_train_rows: int = 90
    family_min_holdout_rows: int = 30
    serious_true_pairs_only: bool = True
    model_dir: Path = _MODEL_DIR
    report_dir: Path = _REPORT_DIR


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return math.isfinite(float(value)) and float(value) != 0.0
        except Exception:
            return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_prob(value: Any) -> float | None:
    out = _float(value)
    if out is None:
        return None
    return max(1e-6, min(1.0 - 1e-6, out))


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _line_token(market: Any, line: Any) -> str:
    market_s = _norm_text(market)
    line_f = _float(line)
    if line_f is None:
        return "line_missing"
    if market_s == "batter_total_bases":
        if abs(line_f - 1.5) < 1e-6:
            return "tb_1.5"
        if line_f >= 3.5:
            return "tb_3.5_plus"
        return "tb_other"
    if market_s in {"batter_hits", "batter_hits_runs_rbis"}:
        if abs(line_f - 0.5) < 1e-6:
            return "hits_0.5"
        if abs(line_f - 1.5) < 1e-6:
            return "line_1.5"
        return "hits_alt"
    if market_s == "batter_home_runs":
        if abs(line_f - 0.5) < 1e-6:
            return "hr_0.5"
        return "hr_alt"
    if market_s == "pitcher_strikeouts":
        if line_f < 4.5:
            return "k_low"
        if line_f <= 6.0:
            return "k_common"
        return "k_high"
    return f"line_{line_f:g}"


def _market_family(book: Any, market: Any, side: Any, line: Any) -> str:
    book_s = _norm_text(book) or "unknown_book"
    market_s = _norm_text(market) or "unknown_market"
    side_s = _norm_text(side) or "unknown_side"
    return f"{book_s}|{market_s}|{side_s}|{_line_token(market_s, line)}"


def _book_market_side(book: Any, market: Any, side: Any) -> str:
    return f"{_norm_text(book) or 'unknown_book'}|{_norm_text(market) or 'unknown_market'}|{_norm_text(side) or 'unknown_side'}"


def _clean_evidence_tier(
    *,
    book: Any,
    pair_quality: Any,
    synthetic_flag: Any,
    true_pair_flag: Any,
    paired_price: Any,
) -> str:
    book_s = _norm_text(book)
    pair_s = _norm_text(pair_quality)
    true_pair = _bool(true_pair_flag)
    synthetic = _bool(synthetic_flag)
    has_pair = _float(paired_price) is not None
    if true_pair and not synthetic and pair_s in {"same_book", "clean_same_book", "true_pair", ""} and has_pair:
        return "true_same_book_pair"
    if true_pair and not synthetic and has_pair:
        return "true_pair"
    if book_s == "fanduel" and synthetic:
        return "fanduel_synthetic_display_only"
    if book_s == "fanduel" and pair_s in {"one_sided", "synthetic", "missing", ""}:
        return "fanduel_one_sided_display_only"
    if synthetic:
        return "synthetic_display_only"
    if has_pair:
        return "paired_uncertain"
    return "one_sided"


def _fanduel_evidence_tier(evidence_tier: Any, book: Any) -> str:
    if _norm_text(book) != "fanduel":
        return "not_fanduel"
    tier = _norm_text(evidence_tier)
    if tier in {"true_same_book_pair", "true_pair"}:
        return "fanduel_true_pair"
    if "synthetic" in tier:
        return "fanduel_synthetic_display_only"
    return "fanduel_one_sided_display_only"


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return None
    if weights is None:
        return float(np.mean(values[mask]))
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    denom = float(np.sum(weights[mask]))
    if denom <= 0:
        return float(np.mean(values[mask]))
    return float(np.sum(values[mask] * weights[mask]) / denom)


def _weighted_brier(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return None
    return _weighted_mean((p[mask] - y[mask]) ** 2, None if weights is None else weights[mask])


def _weighted_log_loss(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    mask = np.isfinite(p)
    if not mask.any() or len(set(y[mask])) < 2:
        return None
    try:
        return float(log_loss(y[mask], p[mask], sample_weight=None if weights is None else weights[mask]))
    except Exception:
        return None


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(p)
    if not mask.any() or len(set(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], p[mask]))
    except Exception:
        return None


def _calibration_error(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None, bins: int = 10) -> float | None:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return None
    y = y[mask]
    p = p[mask]
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    total_w = float(np.sum(w))
    if total_w <= 0:
        return None
    err = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (p >= left) & (p <= right if right >= 1.0 else p < right)
        if not in_bin.any():
            continue
        bw = float(np.sum(w[in_bin]))
        if bw <= 0:
            continue
        err += bw / total_w * abs(float(np.average(y[in_bin], weights=w[in_bin])) - float(np.average(p[in_bin], weights=w[in_bin])))
    return float(err)


def _price_to_implied(value: Any) -> float | None:
    try:
        return american_to_prob(value)
    except Exception:
        return None


def _selected_roi(df: pd.DataFrame, probs: np.ndarray, *, min_ev: float = 0.0) -> dict[str, Any]:
    prices = pd.to_numeric(df.get("market_price"), errors="coerce")
    evs = np.array([
        ev_per_unit(prob, price) if pd.notna(price) else np.nan
        for prob, price in zip(probs, prices)
    ], dtype=float)
    selected = np.isfinite(evs) & (evs > min_ev)
    profit = pd.to_numeric(df.get("profit_units"), errors="coerce").to_numpy(dtype=float)
    won = pd.Series(df.get("won")).map(lambda value: bool(value) if value is not None and not pd.isna(value) else np.nan)
    if not selected.any():
        return {"selected": 0, "roi": None, "win_rate": None, "avg_ev": None}
    return {
        "selected": int(selected.sum()),
        "roi": _weighted_mean(profit[selected]),
        "win_rate": _weighted_mean(won.to_numpy(dtype=float)[selected]),
        "avg_ev": _weighted_mean(evs[selected]),
    }


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=8, sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore")


def _simple_imputer(*, strategy: str, fill_value: Any | None = None) -> SimpleImputer:
    kwargs: dict[str, Any] = {"strategy": strategy}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    try:
        return SimpleImputer(**kwargs, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(**kwargs)


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", _simple_imputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", _simple_imputer(strategy="constant", fill_value="missing")),
        ("onehot", _one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, _NUMERIC_FEATURES),
            ("cat", categorical, _CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _candidate_models(*, include_gbm: bool = False) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "regularized_logistic": Pipeline([
            ("prep", _preprocessor()),
            ("model", LogisticRegression(C=0.65, max_iter=500, solver="lbfgs")),
        ]),
    }
    if include_gbm:
        models["gradient_boosted_meta"] = Pipeline([
            ("prep", _preprocessor()),
            ("model", GradientBoostingClassifier(
                n_estimators=70,
                learning_rate=0.045,
                max_depth=2,
                subsample=0.85,
                random_state=42,
            )),
        ])
    return models


def _fit(model: Pipeline, df: pd.DataFrame, target: str) -> Pipeline:
    y = df[target].astype(int).to_numpy()
    weights = sample_weights(df)
    if weights is not None:
        model.fit(df, y, model__sample_weight=weights)
    else:
        model.fit(df, y)
    return model


def _predict(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    return np.asarray(model.predict_proba(df)[:, 1], dtype=float)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _NUMERIC_FEATURES:
        if col not in out:
            out[col] = np.nan
    for col in _CATEGORICAL_FEATURES:
        if col not in out:
            out[col] = "missing"

    for col in _NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in _CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("missing").astype(str).str.lower().str.strip().replace({"": "missing"})

    out["price_implied_prob"] = out["market_price"].map(_price_to_implied)
    out["paired_price_implied_prob"] = out["paired_price"].map(_price_to_implied)
    if "prob_edge_vs_market" not in out or out["prob_edge_vs_market"].isna().all():
        out["prob_edge_vs_market"] = out["model_prob_side"] - out["market_prob_side"]
    out["model_market_gap_signed"] = out["model_prob_side"] - out["market_prob_side"]
    out["model_market_gap_abs"] = out["model_market_gap_signed"].abs()
    out["market_price_prob_gap"] = out["model_prob_side"] - out["price_implied_prob"]
    out["no_vig_prob_gap"] = out["model_prob_side"] - out["no_vig_market_prob"]
    out["abs_count_edge_side"] = out["count_edge_side"].abs()
    out["model_confidence"] = (out["model_prob_side"] - 0.5).abs()
    out["plus_money_flag"] = (out["market_price"] > 0).astype(float)
    out["favorite_price_flag"] = (out["market_price"] < 0).astype(float)
    out["book_is_fanduel"] = out["bookmaker_key"].map(lambda value: 1.0 if _norm_text(value) == "fanduel" else 0.0)
    out["book_is_draftkings"] = out["bookmaker_key"].map(lambda value: 1.0 if _norm_text(value) == "draftkings" else 0.0)
    out["line_tb15_flag"] = [
        1.0 if _line_token(market, line) == "tb_1.5" else 0.0
        for market, line in zip(out["market"], out["market_line"])
    ]
    out["line_hits05_flag"] = [
        1.0 if _line_token(market, line) == "hits_0.5" else 0.0
        for market, line in zip(out["market"], out["market_line"])
    ]
    out["line_hr05_flag"] = [
        1.0 if _line_token(market, line) == "hr_0.5" else 0.0
        for market, line in zip(out["market"], out["market_line"])
    ]
    out["line_k_common_flag"] = [
        1.0 if _line_token(market, line) == "k_common" else 0.0
        for market, line in zip(out["market"], out["market_line"])
    ]

    for flag in (
        "same_book_pair_flag",
        "cross_book_pair_flag",
        "synthetic_pair_flag",
        "clean_market_pair_flag",
        "true_pair_flag",
    ):
        out[flag] = pd.to_numeric(out[flag], errors="coerce").fillna(0.0)
    out["ai_market_family"] = [
        _market_family(book, market, side, line)
        for book, market, side, line in zip(
            out["bookmaker_key"],
            out["market"],
            out["side"],
            out["market_line"],
        )
    ]
    out["book_market_side"] = [
        _book_market_side(book, market, side)
        for book, market, side in zip(out["bookmaker_key"], out["market"], out["side"])
    ]
    out["clean_evidence_tier"] = [
        _clean_evidence_tier(
            book=book,
            pair_quality=pair_quality,
            synthetic_flag=synthetic,
            true_pair_flag=true_pair,
            paired_price=paired,
        )
        for book, pair_quality, synthetic, true_pair, paired in zip(
            out["bookmaker_key"],
            out["pair_quality"],
            out["synthetic_pair_flag"],
            out["true_pair_flag"],
            out["paired_price"],
        )
    ]
    out["fanduel_evidence_tier"] = [
        _fanduel_evidence_tier(tier, book)
        for tier, book in zip(out["clean_evidence_tier"], out["bookmaker_key"])
    ]
    return out


def _feature_dict_from_ai_row(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = row.get("model_meta") if isinstance(row.get("model_meta"), Mapping) else {}

    def pick(*keys: str) -> Any:
        for key in keys:
            if key in row and row.get(key) is not None:
                return row.get(key)
            if key in meta and meta.get(key) is not None:
                return meta.get(key)
        return None

    model_prob = pick("model_prob", "model_prob_side")
    market_prob = pick("market_prob", "market_prob_side")
    line = pick("line", "market_line")
    price = pick("current_price", "market_price", "locked_price")
    side = pick("side", "bet_side")
    explicit_paired_price = pick("paired_price")
    opposite_price = (
        pick("under_price")
        if _norm_text(side) == "over"
        else pick("over_price") if _norm_text(side) == "under" else None
    )
    paired_price = explicit_paired_price if explicit_paired_price is not None else opposite_price
    pair_quality = pick("pair_quality")
    pair_quality_s = _norm_text(pair_quality)
    inferred_same_book_pair = pair_quality_s == "same_book" and paired_price is not None
    feature = {
        "market": pick("market", "stat"),
        "side": side,
        "bookmaker_key": pick("book", "bookmaker_key"),
        "source": pick("source"),
        "market_line": line,
        "market_price": price,
        "paired_price": paired_price,
        "pair_quality": pair_quality,
        "same_book_pair_flag": pick("same_book_pair_flag") if pick("same_book_pair_flag") is not None else (1.0 if inferred_same_book_pair else 0.0),
        "true_pair_flag": pick("true_pair_flag") if pick("true_pair_flag") is not None else (1.0 if inferred_same_book_pair else 0.0),
        "synthetic_pair_flag": pick("synthetic_pair_flag") if pick("synthetic_pair_flag") is not None else (1.0 if pair_quality_s in {"synthetic", "one_sided"} else 0.0),
        "clean_market_pair_flag": pick("clean_market_pair_flag") if pick("clean_market_pair_flag") is not None else (1.0 if inferred_same_book_pair else 0.0),
        "model_prob_side": model_prob,
        "market_prob_side": market_prob,
        "raw_market_prob": pick("raw_market_prob"),
        "no_vig_market_prob": pick("no_vig_market_prob"),
        "market_prob_source": pick("market_prob_source"),
        "prob_edge_vs_market": (
            (_float(model_prob) - _float(market_prob))
            if _float(model_prob) is not None and _float(market_prob) is not None
            else pick("prob_edge_vs_market")
        ),
        "pred_value": pick("pred_value", "projection_pred_value"),
        "pred_count": pick("pred_count", "projection_count"),
        "count_edge_side": pick("count_edge_side"),
        "ev": pick("current_ev", "ev"),
        "external_agreement_count": len(pick("external_platforms") or []) if isinstance(pick("external_platforms"), list) else pick("external_agreement_count"),
        "external_disagreement_count": pick("external_disagreement_count"),
        "external_agreement_strength": pick("external_agreement_strength"),
    }
    for key in _NUMERIC_FEATURES + _CATEGORICAL_FEATURES:
        if key not in feature:
            feature[key] = pick(key)
    return feature


def _frame_from_ai_row(row: Mapping[str, Any]) -> pd.DataFrame:
    return _prepare_features(pd.DataFrame([_feature_dict_from_ai_row(row)]))


def _add_training_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    won = out["won"].astype(bool)
    profit = pd.to_numeric(out.get("profit_units"), errors="coerce")
    out[_TARGET_WIN] = won.astype(int)
    out[_TARGET_CLV] = out["beat_clv_price"].map(
        lambda value: int(bool(value)) if value is not None and not pd.isna(value) else np.nan
    )

    clv_valid = out["clv_valid"].map(_bool) if "clv_valid" in out else pd.Series(False, index=out.index)
    clv_status = out.get("clv_status", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    clv_reason = out.get("clv_unknown_reason", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()

    good = pd.Series(np.nan, index=out.index, dtype=float)
    good_mask = clv_valid & out[_TARGET_CLV].notna() & profit.notna()
    good.loc[good_mask] = ((profit.loc[good_mask] > 0.0) & (out.loc[good_mask, _TARGET_CLV].astype(float) > 0.0)).astype(int)
    out[_TARGET_GOOD_BET] = good

    available = pd.Series(np.nan, index=out.index, dtype=float)
    available.loc[clv_status.isin(_VALID_CLOSE_STATUSES)] = 1.0
    unavailable_mask = clv_reason.isin(_UNAVAILABLE_CLOSE_REASONS)
    available.loc[unavailable_mask] = 0.0
    out[_TARGET_AVAILABLE] = available

    captured = pd.Series(np.nan, index=out.index, dtype=float)
    captured.loc[clv_status.isin(_VALID_CLOSE_STATUSES)] = 1.0
    captured.loc[clv_reason.isin(_CAPTURE_FAILURE_REASONS | _UNAVAILABLE_CLOSE_REASONS)] = 0.0
    out[_TARGET_CLOSE_CAPTURE] = captured

    out["realized_bet_utility"] = profit.fillna(0.0)
    out.loc[out[_TARGET_CLV].notna(), "realized_bet_utility"] += 0.35 * out.loc[out[_TARGET_CLV].notna(), _TARGET_CLV].astype(float)
    out.loc[clv_valid & out.get("clv_price", pd.Series(np.nan, index=out.index)).notna(), "realized_bet_utility"] += (
        0.01 * pd.to_numeric(out.loc[clv_valid, "clv_price"], errors="coerce").fillna(0.0).clip(-25.0, 25.0)
    )
    out[_TARGET_DAILY_RANK] = 0.0
    dates = pd.to_datetime(out["game_date_et"], errors="coerce").dt.date
    for _, idx in out.groupby(dates, dropna=False).groups.items():
        idx_list = list(idx)
        day_util = out.loc[idx_list, "realized_bet_utility"]
        if len(day_util) < 10:
            continue
        cutoff = float(day_util.quantile(0.85))
        out.loc[idx_list, _TARGET_DAILY_RANK] = ((day_util >= cutoff) & (day_util > 0.0)).astype(float)
    return out


def _serious_training_frame(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if not cfg.serious_true_pairs_only:
        return df.copy()
    clean = (
        (pd.to_numeric(df.get("true_pair_flag"), errors="coerce").fillna(0.0) > 0.0)
        & (pd.to_numeric(df.get("synthetic_pair_flag"), errors="coerce").fillna(0.0) <= 0.0)
        & pd.to_numeric(df.get("paired_price"), errors="coerce").notna()
    )
    return df.loc[clean].copy()


def _read_training_frame(conn, cfg: TrainConfig) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = '120s'")
        cur.execute("SELECT to_regclass('features.mlb_prop_market_training_examples')")
        if cur.fetchone()[0] is None:
            return pd.DataFrame()
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'features'
              AND table_name = 'mlb_prop_market_training_examples'
            """
        )
        available_columns = {str(row[0]) for row in cur.fetchall()}
    columns = [
        "id",
        "source",
        "game_date_et",
        "game_slug",
        "player_id",
        "player_name_norm",
        "team_abbr",
        "market",
        "side",
        "bookmaker_key",
        "market_line",
        "market_price",
        "paired_price",
        "paired_bookmaker_key",
        "paired_price_source",
        "pair_quality",
        "same_book_pair_flag",
        "cross_book_pair_flag",
        "synthetic_pair_flag",
        "clean_market_pair_flag",
        "true_pair_flag",
        "minutes_to_first_pitch_at_lock",
        "lock_price_age_minutes",
        "raw_market_prob",
        "no_vig_market_prob",
        "market_prob_side",
        "market_prob_source",
        "price_bucket",
        "line_bucket",
        "line_surface",
        "model_family",
        "edge_type",
        "pred_value",
        "pred_count",
        "model_prob_side",
        "count_edge_side",
        "prob_edge_vs_market",
        "confirmed_batting_order",
        "confirmed_lineup_source",
        "projected_pa",
        "pa_low_probability",
        "pa_normal_mean",
        "pa_games",
        "projected_ip",
        "projected_bf",
        "projected_pitch_count",
        "pitcher_starts",
        "is_home",
        "opponent_abbr",
        "opp_sp_hand",
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
        "pinch_hit_risk",
        "team_implied_runs",
        "opponent_implied_runs",
        "game_total_line",
        "ev",
        "won",
        "push",
        "profit_units",
        "closing_price",
        "clv_line",
        "clv_price",
        "beat_clv_line",
        "clv_valid",
        "beat_clv_price",
        "clv_status",
        "clv_unknown_reason",
        "result_status",
        "source_created_at",
        "prop_offer_id",
        "replay_id",
        "run_id",
    ]
    required_columns = {
        "id",
        "game_date_et",
        "market",
        "model_prob_side",
        "market_price",
        "won",
        "push",
    }
    missing_required = sorted(required_columns - available_columns)
    if missing_required:
        raise RuntimeError(
            "features.mlb_prop_market_training_examples is missing required columns: "
            + ", ".join(missing_required)
        )
    selected_columns = [
        col if col in available_columns else f"NULL AS {col}"
        for col in columns
    ]
    col_sql = ",\n        ".join(selected_columns)
    filters = [
        "won IS NOT NULL",
        "COALESCE(push, FALSE) IS FALSE",
        "market = ANY(%(markets)s)",
        "model_prob_side IS NOT NULL",
        "market_price IS NOT NULL",
    ]
    params: dict[str, Any] = {"markets": list(_MARKETS)}
    if cfg.lookback_days > 0:
        filters.append("game_date_et >= CURRENT_DATE - (%(lookback_days)s::int * INTERVAL '1 day')")
        params["lookback_days"] = int(cfg.lookback_days)
    sql = f"""
        SELECT
            {col_sql}
        FROM features.mlb_prop_market_training_examples
        WHERE {' AND '.join(filters)}
        ORDER BY game_date_et, id
    """
    df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return df
    df = dedupe_locked_offer_rows(df)
    raw_rows = int(df.attrs.get("raw_rows", len(df)))
    deduped_rows = int(df.attrs.get("deduped_rows", 0))
    if cfg.max_rows > 0 and len(df) > cfg.max_rows:
        df = df.sort_values(["game_date_et", "id"]).tail(int(cfg.max_rows)).copy()
    df = _add_training_targets(df)
    df = _prepare_features(df)
    df = add_player_game_weights(df)
    df.attrs["raw_rows"] = raw_rows
    df.attrs["deduped_rows"] = deduped_rows
    return df


def _baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    y_mean = float(pd.to_numeric(df.get("target_win"), errors="coerce").mean()) if not df.empty else 0.5
    model_prob = pd.to_numeric(df.get("model_prob_side"), errors="coerce").fillna(y_mean).clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
    market_prob = pd.to_numeric(df.get("market_prob_side"), errors="coerce").fillna(y_mean).clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
    blend = np.clip(0.55 * model_prob + 0.45 * market_prob, 1e-6, 1 - 1e-6)
    return {
        "model_prob_side": model_prob,
        "market_prob_side": market_prob,
        "model_market_blend": blend,
    }


def _metrics(df: pd.DataFrame, target: str, probs: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    weights = sample_weights(df)
    return {
        "rows": int(len(df)),
        "positive_rate": _weighted_mean(y, weights),
        "brier": _weighted_brier(y, probs, weights),
        "log_loss": _weighted_log_loss(y, probs, weights),
        "auc": _auc(y, probs),
        "calibration_error": _calibration_error(y, probs, weights),
        "selection": _selected_roi(df, probs) if target == "target_win" else None,
    }


def _calibration_bins(
    df: pd.DataFrame,
    target: str,
    probs: np.ndarray,
    *,
    bins: int = 6,
) -> list[dict[str, Any]]:
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    p = np.asarray(probs, dtype=float)
    weights = sample_weights(df)
    if weights is None:
        weights = np.ones(len(df), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(weights)
    if not mask.any():
        return []
    y = y[mask]
    p = p[mask]
    weights = weights[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict[str, Any]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (p >= left) & (p <= right if right >= 1.0 else p < right)
        if not in_bin.any():
            continue
        w = weights[in_bin]
        weight_sum = float(np.sum(w))
        if weight_sum <= 0:
            continue
        out.append({
            "left": float(left),
            "right": float(right),
            "rows": int(in_bin.sum()),
            "weight": weight_sum,
            "predicted": float(np.average(p[in_bin], weights=w)),
            "actual": float(np.average(y[in_bin], weights=w)),
        })
    return out


def _apply_probability_calibration(prob: float | None, bins: list[Mapping[str, Any]] | None) -> float | None:
    p = _safe_prob(prob)
    if p is None or not bins:
        return p
    for rec in bins:
        left = _float(rec.get("left"))
        right = _float(rec.get("right"))
        if left is None or right is None:
            continue
        if p < left or (p > right if right >= 1.0 else p >= right):
            continue
        actual = _safe_prob(rec.get("actual"))
        predicted = _safe_prob(rec.get("predicted"))
        rows = _float(rec.get("rows")) or 0.0
        if actual is None or predicted is None or rows <= 0:
            return p
        shrink = rows / (rows + 60.0)
        return max(1e-6, min(1.0 - 1e-6, (1.0 - shrink) * p + shrink * actual))
    return p


def _walk_forward_predictions(
    df: pd.DataFrame,
    *,
    target: str,
    cfg: TrainConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str]:
    folds = expanding_player_game_folds(
        df,
        test_window_days=cfg.fold_days,
        step_days=cfg.fold_days,
        min_train_dates=cfg.min_train_dates,
        min_train_rows=cfg.min_train_rows,
        min_holdout_rows=cfg.min_holdout_rows,
        max_folds=cfg.max_folds,
    )
    strategy = "expanding_player_game_folds"
    if not folds:
        split = temporal_player_game_split(
            df,
            holdout_days=max(1, cfg.fold_days),
            min_train_rows=cfg.min_train_rows,
            min_holdout_rows=cfg.min_holdout_rows,
        )
        if split.train.empty or split.holdout.empty:
            return {}, [], "insufficient_temporal_split"
        folds = [type("Fold", (), {
            "fold_index": 1,
            "train": split.train,
            "holdout": split.holdout,
            "train_start": None,
            "train_end": None,
            "holdout_start": None,
            "holdout_end": None,
            "purged_rows": split.purged_rows,
        })()]
        strategy = split.strategy

    candidates = _candidate_models(include_gbm=cfg.include_gbm)
    predictions: dict[str, list[pd.Series]] = {name: [] for name in candidates}
    holdout_parts: list[pd.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []
    for fold in folds:
        train = fold.train.dropna(subset=[target]).copy()
        holdout = fold.holdout.dropna(subset=[target]).copy()
        if len(train) < cfg.min_train_rows or len(holdout) < cfg.min_holdout_rows:
            continue
        if train[target].nunique() < 2:
            continue
        holdout_parts.append(holdout)
        fold_meta.append({
            "fold_index": int(fold.fold_index),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "holdout_start": str(fold.holdout_start),
            "holdout_end": str(fold.holdout_end),
            "purged_rows": int(fold.purged_rows),
        })
        for name, model in _candidate_models(include_gbm=cfg.include_gbm).items():
            fitted = _fit(model, train, target)
            predictions[name].append(pd.Series(_predict(fitted, holdout), index=holdout.index))

    if not holdout_parts:
        return {}, [], f"{strategy}_no_valid_folds"
    holdout_all = pd.concat(holdout_parts, axis=0).sort_index()
    out: dict[str, np.ndarray] = {"__holdout_index__": holdout_all.index.to_numpy()}
    for name, parts in predictions.items():
        if parts:
            out[name] = pd.concat(parts).reindex(holdout_all.index).to_numpy(dtype=float)
    return out, fold_meta, strategy


def _train_target(df: pd.DataFrame, *, target: str, cfg: TrainConfig) -> dict[str, Any]:
    work = df.dropna(subset=[target]).copy()
    if len(work) < cfg.min_rows or work[target].nunique() < 2:
        return {
            "status": "insufficient_rows",
            "rows": int(len(work)),
            "positive_rate": float(work[target].mean()) if len(work) else None,
            "enabled": False,
            "model": None,
        }
    wf, folds, strategy = _walk_forward_predictions(work, target=target, cfg=cfg)
    if not wf:
        return {
            "status": strategy,
            "rows": int(len(work)),
            "positive_rate": float(work[target].mean()) if len(work) else None,
            "enabled": False,
            "model": None,
        }
    holdout = work.loc[wf["__holdout_index__"]].copy()
    baselines = _baseline_predictions(holdout) if target in {_TARGET_WIN, _TARGET_GOOD_BET, _TARGET_DAILY_RANK} else {
        "global_mean": np.full(len(holdout), float(work[target].mean()), dtype=float)
    }
    metrics = {name: _metrics(holdout, target, probs) for name, probs in baselines.items()}
    for name, probs in wf.items():
        if name == "__holdout_index__":
            continue
        metrics[name] = _metrics(holdout, target, probs)

    candidates = _candidate_models(include_gbm=cfg.include_gbm)
    baseline_names = [name for name in metrics if name not in candidates]
    baseline_best = min(
        (metrics[name]["brier"], name)
        for name in baseline_names
        if metrics[name].get("brier") is not None
    )
    model_best = min(
        (metrics[name]["brier"], name)
        for name in candidates
        if name in metrics and metrics[name].get("brier") is not None
    )
    brier_gain = float(baseline_best[0] - model_best[0])
    selected_model_name = model_best[1]
    enabled = brier_gain >= cfg.min_brier_gain
    selected_probs = wf.get(selected_model_name)
    calibration = _calibration_bins(holdout, target, selected_probs) if selected_probs is not None else []
    final_model = _candidate_models(include_gbm=cfg.include_gbm)[selected_model_name]
    _fit(final_model, work, target)
    return {
        "status": "enabled" if enabled else "shadow_brier_not_better",
        "enabled": bool(enabled),
        "rows": int(len(work)),
        "dates": int(pd.to_datetime(work["game_date_et"], errors="coerce").dt.date.nunique()),
        "positive_rate": float(work[target].mean()),
        "target": target,
        "selected_model": selected_model_name,
        "split_strategy": strategy,
        "folds": folds,
        "grouping": grouping_summary(work),
        "baseline_best": {"name": baseline_best[1], "brier": float(baseline_best[0])},
        "model_best": {"name": model_best[1], "brier": float(model_best[0])},
        "brier_gain": brier_gain,
        "calibration_bins": calibration,
        "metrics": metrics,
        "model": final_model,
    }


def _child_cfg(
    cfg: TrainConfig,
    *,
    min_rows: int | None = None,
    min_train_rows: int | None = None,
    min_holdout_rows: int | None = None,
    max_folds: int | None = None,
) -> TrainConfig:
    return TrainConfig(
        lookback_days=cfg.lookback_days,
        min_rows=cfg.min_rows if min_rows is None else min_rows,
        min_train_rows=cfg.min_train_rows if min_train_rows is None else min_train_rows,
        min_holdout_rows=cfg.min_holdout_rows if min_holdout_rows is None else min_holdout_rows,
        min_train_dates=cfg.min_train_dates,
        fold_days=cfg.fold_days,
        max_folds=cfg.max_folds if max_folds is None else max_folds,
        min_brier_gain=cfg.min_brier_gain,
        include_gbm=cfg.include_gbm,
        max_rows=cfg.max_rows,
        max_market_families=cfg.max_market_families,
        family_min_rows=cfg.family_min_rows,
        family_min_train_rows=cfg.family_min_train_rows,
        family_min_holdout_rows=cfg.family_min_holdout_rows,
        serious_true_pairs_only=cfg.serious_true_pairs_only,
        model_dir=cfg.model_dir,
        report_dir=cfg.report_dir,
    )


def _without_model(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in rec.items() if key != "model"}


def _priority_market_families() -> list[str]:
    return [
        "draftkings|pitcher_strikeouts|under|k_low",
        "draftkings|pitcher_strikeouts|under|k_common",
        "draftkings|batter_total_bases|over|tb_1.5",
        "fanduel|pitcher_strikeouts|under|k_low",
        "fanduel|pitcher_strikeouts|under|k_common",
        "fanduel|batter_total_bases|over|tb_1.5",
        "fanduel|batter_home_runs|over|hr_0.5",
        "draftkings|batter_home_runs|over|hr_0.5",
    ]


def _train_market_family_models(df: pd.DataFrame, cfg: TrainConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if df.empty or "ai_market_family" not in df:
        return {}, {}
    counts = df.groupby("ai_market_family", dropna=False).size().sort_values(ascending=False)
    ordered: list[str] = []
    for family in _priority_market_families():
        if family in counts.index and family not in ordered:
            ordered.append(family)
    for family in counts.index:
        if family not in ordered:
            ordered.append(str(family))
        if len(ordered) >= cfg.max_market_families:
            break

    family_cfg = _child_cfg(
        cfg,
        min_rows=cfg.family_min_rows,
        min_train_rows=cfg.family_min_train_rows,
        min_holdout_rows=cfg.family_min_holdout_rows,
        max_folds=min(cfg.max_folds, 4),
    )
    payload: dict[str, Any] = {}
    artifact: dict[str, Any] = {}
    for family in ordered[: cfg.max_market_families]:
        fam_df = df.loc[df["ai_market_family"] == family].copy()
        target_payload: dict[str, Any] = {}
        target_models: dict[str, Any] = {}
        for target in (_TARGET_GOOD_BET, _TARGET_WIN, _TARGET_CLV, _TARGET_AVAILABLE):
            rec = _train_target(fam_df, target=target, cfg=family_cfg)
            target_payload[target] = _without_model(rec)
            if rec.get("model") is not None:
                target_models[target] = rec.get("model")
        enabled_targets = [
            target for target, rec in target_payload.items()
            if isinstance(rec, Mapping) and bool(rec.get("enabled"))
        ]
        payload[family] = {
            "rows": int(len(fam_df)),
            "dates": int(pd.to_datetime(fam_df["game_date_et"], errors="coerce").dt.date.nunique()),
            "book": family.split("|")[0] if "|" in family else None,
            "clean_evidence": fam_df.groupby("clean_evidence_tier", dropna=False).size().sort_values(ascending=False).to_dict(),
            "enabled": bool(enabled_targets),
            "enabled_targets": enabled_targets,
            "targets": target_payload,
        }
        artifact[family] = {
            **payload[family],
            "models": target_models,
        }
    return payload, artifact


def _evidence_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    out = {
        "by_evidence_tier": df.groupby("clean_evidence_tier", dropna=False).size().sort_values(ascending=False).to_dict(),
        "by_book": df.groupby("bookmaker_key", dropna=False).size().sort_values(ascending=False).to_dict(),
    }
    fd = df.loc[df["bookmaker_key"].astype(str).str.lower() == "fanduel"].copy()
    if not fd.empty:
        out["fanduel"] = {
            "rows": int(len(fd)),
            "by_evidence_tier": fd.groupby("fanduel_evidence_tier", dropna=False).size().sort_values(ascending=False).to_dict(),
            "by_market": fd.groupby("market", dropna=False).size().sort_values(ascending=False).to_dict(),
        }
    return out


def train_ai_bet_selection_model(conn, cfg: TrainConfig) -> dict[str, Any]:
    df = _read_training_frame(conn, cfg)
    raw_rows = int(df.attrs.get("raw_rows", len(df))) if hasattr(df, "attrs") else int(len(df))
    payload: dict[str, Any] = {
        "model_version": f"mlb_ai_bet_selection_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_days": int(cfg.lookback_days),
        "include_gbm": bool(cfg.include_gbm),
        "max_rows": int(cfg.max_rows),
        "max_market_families": int(cfg.max_market_families),
        "serious_true_pairs_only": bool(cfg.serious_true_pairs_only),
        "raw_rows": raw_rows,
        "rows": int(len(df)),
        "deduped_rows_removed": int(raw_rows - len(df)),
        "numeric_features": list(_NUMERIC_FEATURES),
        "categorical_features": list(_CATEGORICAL_FEATURES),
        "good_bet": None,
        "win": None,
        "clv": None,
        "line_available": None,
        "daily_rank": None,
        "market_families": {},
        "enabled": False,
    }
    if df.empty:
        payload["status"] = "no_training_rows"
        _write_outputs(payload, cfg)
        return payload

    payload["date_min"] = str(pd.to_datetime(df["game_date_et"], errors="coerce").dt.date.min())
    payload["date_max"] = str(pd.to_datetime(df["game_date_et"], errors="coerce").dt.date.max())
    payload["by_market"] = df.groupby("market", dropna=False).size().sort_values(ascending=False).to_dict()
    payload["by_pair_quality"] = df.groupby("pair_quality", dropna=False).size().sort_values(ascending=False).to_dict()
    payload["evidence"] = _evidence_summary(df)

    serious_df = _serious_training_frame(df, cfg)
    payload["serious_rows"] = int(len(serious_df))
    payload["serious_evidence"] = _evidence_summary(serious_df)

    target_cfg = _child_cfg(cfg)
    good_bet = _train_target(serious_df, target=_TARGET_GOOD_BET, cfg=target_cfg)
    win = _train_target(serious_df, target=_TARGET_WIN, cfg=target_cfg)
    clv_valid_mask = serious_df["clv_valid"].map(_bool) if "clv_valid" in serious_df else pd.Series(False, index=serious_df.index)
    clv_df = serious_df.loc[clv_valid_mask & serious_df[_TARGET_CLV].notna()].copy()
    clv = _train_target(clv_df, target=_TARGET_CLV, cfg=_child_cfg(
        cfg,
        min_rows=max(250, min(cfg.min_rows, 500)),
        min_train_rows=max(150, min(cfg.min_train_rows, 300)),
        min_holdout_rows=max(40, min(cfg.min_holdout_rows, 80)),
    ))
    available = _train_target(serious_df, target=_TARGET_AVAILABLE, cfg=_child_cfg(
        cfg,
        min_rows=max(250, min(cfg.min_rows, 500)),
        min_train_rows=max(150, min(cfg.min_train_rows, 300)),
        min_holdout_rows=max(40, min(cfg.min_holdout_rows, 80)),
    ))
    daily_rank = _train_target(serious_df, target=_TARGET_DAILY_RANK, cfg=_child_cfg(
        cfg,
        min_rows=max(250, min(cfg.min_rows, 500)),
        min_train_rows=max(150, min(cfg.min_train_rows, 300)),
        min_holdout_rows=max(40, min(cfg.min_holdout_rows, 80)),
    ))
    family_payload, family_artifacts = _train_market_family_models(serious_df, cfg)
    payload["good_bet"] = _without_model(good_bet)
    payload["win"] = _without_model(win)
    payload["clv"] = _without_model(clv)
    payload["line_available"] = _without_model(available)
    payload["daily_rank"] = _without_model(daily_rank)
    payload["market_families"] = family_payload
    payload["enabled"] = bool(good_bet.get("enabled") or win.get("enabled"))
    payload["status"] = "enabled" if payload["enabled"] else "shadow"

    artifact = {
        **payload,
        "good_bet_model": good_bet.get("model"),
        "win_model": win.get("model"),
        "clv_model": clv.get("model"),
        "line_available_model": available.get("model"),
        "daily_rank_model": daily_rank.get("model"),
        "market_family_models": family_artifacts,
    }
    _write_outputs(payload, cfg, artifact=artifact)
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    numeric = _float(value)
    if numeric is None:
        return "-"
    return f"{numeric * 100.0:.{digits}f}%"


def _model_line(name: str, rec: Mapping[str, Any] | None) -> str:
    if not rec:
        return f"- {name}: no result"
    return (
        f"- {name}: status={rec.get('status')} rows={rec.get('rows')} "
        f"dates={rec.get('dates', '-')} selected={rec.get('selected_model', '-')} "
        f"brier_gain={_fmt(rec.get('brier_gain'))}"
    )


def _report(payload: Mapping[str, Any]) -> str:
    good = payload.get("good_bet") if isinstance(payload.get("good_bet"), Mapping) else {}
    win = payload.get("win") if isinstance(payload.get("win"), Mapping) else {}
    clv = payload.get("clv") if isinstance(payload.get("clv"), Mapping) else {}
    available = payload.get("line_available") if isinstance(payload.get("line_available"), Mapping) else {}
    daily_rank = payload.get("daily_rank") if isinstance(payload.get("daily_rank"), Mapping) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else {}
    serious_evidence = payload.get("serious_evidence") if isinstance(payload.get("serious_evidence"), Mapping) else {}
    fd = evidence.get("fanduel") if isinstance(evidence.get("fanduel"), Mapping) else {}
    fd_serious = serious_evidence.get("fanduel") if isinstance(serious_evidence.get("fanduel"), Mapping) else {}
    lines = [
        "# MLB AI Bet Selection Model",
        "",
        f"- Generated: {payload.get('generated_at_utc')}",
        f"- Status: {payload.get('status')} enabled={payload.get('enabled')}",
        f"- Candidate set: regularized_logistic{' + gradient_boosted_meta' if payload.get('include_gbm') else ''}",
        f"- Rows: {payload.get('rows')} after dedupe ({payload.get('deduped_rows_removed')} duplicates removed)",
        f"- Serious clean rows: {payload.get('serious_rows')} true-paired rows used for real training",
        f"- Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"- FanDuel evidence: all={fd.get('by_evidence_tier', {})} serious={fd_serious.get('by_evidence_tier', {})}",
        "",
        "## Targets",
        _model_line("Good-bet classifier", good),
        _model_line("Win classifier", win),
        _model_line("CLV classifier", clv),
        _model_line("Line-available classifier", available),
        _model_line("Daily rank classifier", daily_rank),
        "",
        "## Good-Bet Metrics",
    ]
    good_metrics = good.get("metrics") if isinstance(good.get("metrics"), Mapping) else {}
    for name, rec in good_metrics.items():
        if not isinstance(rec, Mapping):
            continue
        sel = rec.get("selection") if isinstance(rec.get("selection"), Mapping) else {}
        lines.append(
            f"- {name}: Brier={_fmt(rec.get('brier'))} AUC={_fmt(rec.get('auc'))} "
            f"CalErr={_fmt_pct(rec.get('calibration_error'))} "
            f"selected={sel.get('selected', '-')} ROI={_fmt_pct(sel.get('roi'))} "
            f"win={_fmt_pct(sel.get('win_rate'))}"
        )
    lines.extend([
        "",
        "## Win Metrics",
    ])
    metrics = win.get("metrics") if isinstance(win.get("metrics"), Mapping) else {}
    for name, rec in metrics.items():
        if not isinstance(rec, Mapping):
            continue
        sel = rec.get("selection") if isinstance(rec.get("selection"), Mapping) else {}
        lines.append(
            f"- {name}: Brier={_fmt(rec.get('brier'))} AUC={_fmt(rec.get('auc'))} "
            f"CalErr={_fmt_pct(rec.get('calibration_error'))} "
            f"selected={sel.get('selected', '-')} ROI={_fmt_pct(sel.get('roi'))} "
            f"win={_fmt_pct(sel.get('win_rate'))}"
        )
    lines.extend([
        "",
        "## CLV Metrics",
    ])
    clv_metrics = clv.get("metrics") if isinstance(clv.get("metrics"), Mapping) else {}
    for name, rec in clv_metrics.items():
        if not isinstance(rec, Mapping):
            continue
        lines.append(
            f"- {name}: Brier={_fmt(rec.get('brier'))} AUC={_fmt(rec.get('auc'))} "
            f"CalErr={_fmt_pct(rec.get('calibration_error'))}"
        )
    lines.extend([
        "",
        "## Market Families",
    ])
    families = payload.get("market_families") if isinstance(payload.get("market_families"), Mapping) else {}
    if not families:
        lines.append("- No market-family models trained.")
    for family, rec in list(families.items())[:20]:
        if not isinstance(rec, Mapping):
            continue
        enabled_targets = ",".join(str(item) for item in rec.get("enabled_targets") or []) or "-"
        good_rec = ((rec.get("targets") or {}).get(_TARGET_GOOD_BET) or {}) if isinstance(rec.get("targets"), Mapping) else {}
        lines.append(
            f"- {family}: rows={rec.get('rows')} dates={rec.get('dates')} enabled={rec.get('enabled')} "
            f"targets={enabled_targets} good_status={good_rec.get('status')} "
            f"good_gain={_fmt(good_rec.get('brier_gain'))}"
        )
    lines.extend([
        "",
        "## Notes",
        "- This model is trained only from pre-existing locked/graded rows.",
        "- Serious real-money training uses clean true-paired rows; FanDuel synthetic/one-sided rows are display/watch evidence only.",
        "- Good-bet means the row won and beat closing price on a valid close row.",
        "- Daily rank learns which historical slate rows were among the strongest realized betting opportunities.",
        "- It does not use closing line, final result, or actual stat values as live features.",
        "- Enabled means the learned good-bet or win model beat the best simple baseline on grouped walk-forward Brier.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(
    payload: Mapping[str, Any],
    cfg: TrainConfig,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.model_dir / _META_FILE, payload, default=_json_default)
    atomic_write_text(cfg.report_dir / _REPORT_FILE, _report(payload))
    if artifact is not None and joblib is not None:
        atomic_write_via(
            cfg.model_dir / _ARTIFACT_FILE,
            lambda temporary: joblib.dump(dict(artifact), temporary),
            retry_all_errors=True,
        )


def load_ai_bet_selection_artifact(model_dir: Path = _MODEL_DIR) -> dict[str, Any] | None:
    if joblib is None:
        return None
    path = Path(model_dir) / _ARTIFACT_FILE
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    if not isinstance(artifact, dict):
        return None
    return artifact


def _target_meta_from_artifact(artifact: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    key = _TARGET_PAYLOAD_KEYS.get(target)
    rec = artifact.get(key) if key else None
    return rec if isinstance(rec, Mapping) else {}


def _score_target_model(
    model: Any,
    meta: Mapping[str, Any],
    frame: pd.DataFrame,
) -> float | None:
    if model is None:
        return None
    try:
        raw = float(_predict(model, frame)[0])
    except Exception:
        return None
    bins = meta.get("calibration_bins") if isinstance(meta, Mapping) else None
    return _apply_probability_calibration(raw, bins if isinstance(bins, list) else None)


def _score_global_target(
    artifact: Mapping[str, Any],
    target: str,
    frame: pd.DataFrame,
) -> tuple[float | None, bool, str]:
    meta = _target_meta_from_artifact(artifact, target)
    prob = _score_target_model(artifact.get(_TARGET_MODEL_KEYS.get(target, "")), meta, frame)
    return prob, bool(meta.get("enabled")), str(meta.get("status") or "unknown")


def _score_family_target(
    family_artifact: Mapping[str, Any] | None,
    target: str,
    frame: pd.DataFrame,
) -> tuple[float | None, bool, str]:
    if not isinstance(family_artifact, Mapping):
        return None, False, "family_missing"
    targets = family_artifact.get("targets") if isinstance(family_artifact.get("targets"), Mapping) else {}
    meta = targets.get(target) if isinstance(targets.get(target), Mapping) else {}
    models = family_artifact.get("models") if isinstance(family_artifact.get("models"), Mapping) else {}
    prob = _score_target_model(models.get(target), meta, frame)
    return prob, bool(meta.get("enabled")), str(meta.get("status") or "unknown")


def _score_target_array(
    model: Any,
    meta: Mapping[str, Any],
    frame: pd.DataFrame,
) -> list[float | None]:
    if model is None or frame.empty:
        return [None] * len(frame)
    try:
        raw = _predict(model, frame)
    except Exception:
        return [None] * len(frame)
    bins = meta.get("calibration_bins") if isinstance(meta, Mapping) else None
    return [
        _apply_probability_calibration(float(prob), bins if isinstance(bins, list) else None)
        for prob in raw
    ]


def _score_target_arrays(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for target in (_TARGET_GOOD_BET, _TARGET_WIN, _TARGET_CLV, _TARGET_AVAILABLE, _TARGET_DAILY_RANK):
        meta = _target_meta_from_artifact(artifact, target)
        out[target] = {
            "probs": _score_target_array(artifact.get(_TARGET_MODEL_KEYS.get(target, "")), meta, frame),
            "enabled": bool(meta.get("enabled")),
            "status": str(meta.get("status") or "unknown"),
        }
    return out


def score_ai_bet_selection_rows(
    rows: list[Mapping[str, Any]],
    artifact: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if not artifact:
        return [{"ai_ml_status": "artifact_missing", "ai_ml_enabled": False} for _ in rows]
    frame = _prepare_features(pd.DataFrame([_feature_dict_from_ai_row(row) for row in rows]))
    global_scores = _score_target_arrays(artifact, frame)
    family_models = artifact.get("market_family_models") if isinstance(artifact.get("market_family_models"), Mapping) else {}

    family_scores: dict[str, dict[str, dict[str, Any]]] = {}
    family_values = frame["ai_market_family"].astype(str)
    for family, idx in frame.groupby(family_values, dropna=False).groups.items():
        family_artifact = family_models.get(str(family)) if isinstance(family_models, Mapping) else None
        if not isinstance(family_artifact, Mapping):
            continue
        sub = frame.loc[list(idx)]
        targets = family_artifact.get("targets") if isinstance(family_artifact.get("targets"), Mapping) else {}
        models = family_artifact.get("models") if isinstance(family_artifact.get("models"), Mapping) else {}
        fam_out: dict[str, dict[str, Any]] = {}
        for target in (_TARGET_GOOD_BET, _TARGET_WIN, _TARGET_CLV, _TARGET_AVAILABLE):
            meta = targets.get(target) if isinstance(targets.get(target), Mapping) else {}
            probs = _score_target_array(models.get(target), meta, sub)
            fam_out[target] = {
                "index": list(idx),
                "probs": probs,
                "enabled": bool(meta.get("enabled")),
                "status": str(meta.get("status") or "unknown"),
            }
        family_scores[str(family)] = fam_out

    outputs: list[dict[str, Any]] = []
    clv_meta = _target_meta_from_artifact(artifact, _TARGET_CLV)
    clv_prior = _float(clv_meta.get("positive_rate"))
    for pos, row in enumerate(rows):
        frame_row = frame.iloc[pos].to_dict()
        family = str(frame_row.get("ai_market_family") or "unknown")
        evidence_tier = str(frame_row.get("clean_evidence_tier") or "unknown")
        fanduel_tier = str(frame_row.get("fanduel_evidence_tier") or "not_fanduel")
        real_money_evidence = evidence_tier in {"true_same_book_pair", "true_pair"}
        fam = family_scores.get(family) or {}

        def fam_prob(target: str) -> tuple[float | None, bool, str]:
            rec = fam.get(target) if isinstance(fam.get(target), Mapping) else None
            if not rec:
                return None, False, "family_missing"
            try:
                local_pos = list(rec.get("index") or []).index(frame.index[pos])
            except ValueError:
                return None, bool(rec.get("enabled")), str(rec.get("status") or "unknown")
            probs = rec.get("probs") or []
            return (
                probs[local_pos] if local_pos < len(probs) else None,
                bool(rec.get("enabled")),
                str(rec.get("status") or "unknown"),
            )

        family_good, family_good_enabled, family_good_status = fam_prob(_TARGET_GOOD_BET)
        family_win, family_win_enabled, family_win_status = fam_prob(_TARGET_WIN)
        family_clv, family_clv_enabled, family_clv_status = fam_prob(_TARGET_CLV)
        family_avail, family_avail_enabled, _ = fam_prob(_TARGET_AVAILABLE)

        global_good = global_scores[_TARGET_GOOD_BET]["probs"][pos]
        global_win = global_scores[_TARGET_WIN]["probs"][pos]
        global_clv = global_scores[_TARGET_CLV]["probs"][pos]
        global_avail = global_scores[_TARGET_AVAILABLE]["probs"][pos]
        global_rank = global_scores[_TARGET_DAILY_RANK]["probs"][pos]
        global_good_enabled = bool(global_scores[_TARGET_GOOD_BET]["enabled"])
        global_win_enabled = bool(global_scores[_TARGET_WIN]["enabled"])
        global_good_status = str(global_scores[_TARGET_GOOD_BET]["status"])
        global_win_status = str(global_scores[_TARGET_WIN]["status"])

        use_family = bool(fam) and (
            family_good_enabled or family_win_enabled or family_clv_enabled or family_avail_enabled
        )
        good_prob = family_good if use_family and family_good is not None else global_good
        win_prob = family_win if use_family and family_win is not None else global_win
        clv_prob = family_clv if use_family and family_clv is not None else global_clv
        avail_prob = family_avail if use_family and family_avail is not None else global_avail
        source = "family" if use_family else "global_clean"
        enabled = bool(real_money_evidence and (use_family or global_good_enabled or global_win_enabled))
        if clv_prob is None:
            clv_prob = clv_prior
        if good_prob is None and win_prob is not None and clv_prob is not None:
            good_prob = max(1e-6, min(1.0 - 1e-6, win_prob * clv_prob))
        if win_prob is None:
            win_prob = good_prob

        price = _float(row.get("current_price")) if row.get("current_price") is not None else _float(row.get("locked_price"))
        market_prob = _float(row.get("market_prob"))
        ai_ev = ev_per_unit(win_prob, price) if win_prob is not None and price is not None else None
        ai_edge = win_prob - market_prob if win_prob is not None and market_prob is not None else None
        out: dict[str, Any] = {
            "ai_ml_model_version": artifact.get("model_version"),
            "ai_ml_status": artifact.get("status") or "unknown",
            "ai_ml_artifact_enabled": bool(artifact.get("enabled")),
            "ai_ml_market_family": family,
            "ai_ml_clean_evidence_tier": evidence_tier,
            "ai_ml_fanduel_evidence_tier": fanduel_tier,
            "ai_ml_real_money_evidence": bool(real_money_evidence),
            "ai_ml_enabled": bool(enabled),
            "ai_ml_source": source if (win_prob is not None or good_prob is not None) else "none",
            "ai_ml_family_status": family_good_status if use_family else "not_used",
            "ai_ml_global_status": global_good_status or global_win_status,
        }
        if win_prob is not None or good_prob is not None:
            win_part = win_prob if win_prob is not None else good_prob or 0.50
            good_part = good_prob if good_prob is not None else max(1e-6, min(1.0 - 1e-6, win_part * (clv_prob or 0.50)))
            ev_part = max(-0.20, min(0.35, ai_ev or 0.0))
            clv_part = clv_prob if clv_prob is not None else 0.50
            avail_part = avail_prob if avail_prob is not None else 0.75
            rank_part = global_rank if global_rank is not None else 0.10
            ai_score = 100.0 * (
                0.35 * win_part
                + 0.22 * good_part
                + 0.18 * clv_part
                + 0.10 * avail_part
                + 0.08 * ((ev_part + 0.20) / 0.55)
                + 0.07 * rank_part
            )
            if not real_money_evidence:
                ai_score -= 12.0 if fanduel_tier.startswith("fanduel") else 8.0
            out.update({
                "ai_ml_good_bet_prob": good_prob,
                "ai_ml_no_bet_prob": (1.0 - good_prob) if good_prob is not None else None,
                "ai_ml_win_prob": win_prob,
                "ai_ml_clv_beat_prob": clv_prob,
                "ai_ml_line_available_prob": avail_prob,
                "ai_ml_daily_top_prob": global_rank,
                "ai_ml_edge": ai_edge,
                "ai_ml_ev": ai_ev,
                "ai_ml_score": round(float(ai_score), 4),
            })
        outputs.append(out)
    return outputs


def score_ai_bet_selection_row(
    row: Mapping[str, Any],
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scored = score_ai_bet_selection_rows([row], artifact)
    return scored[0] if scored else {"ai_ml_status": "artifact_missing", "ai_ml_enabled": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned MLB AI bet-selection model")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--min-holdout-rows", type=int, default=80)
    parser.add_argument("--min-train-dates", type=int, default=5)
    parser.add_argument("--fold-days", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--min-brier-gain", type=float, default=0.001)
    parser.add_argument("--include-gbm", action="store_true", help="Also test a slower boosted meta-model candidate")
    parser.add_argument("--max-rows", type=int, default=0, help="Use only the most recent N deduped rows when positive")
    parser.add_argument("--max-market-families", type=int, default=10)
    parser.add_argument("--family-min-rows", type=int, default=180)
    parser.add_argument("--include-synthetic-training", action="store_true", help="Allow synthetic/one-sided rows into serious training")
    args = parser.parse_args()

    cfg = TrainConfig(
        lookback_days=args.lookback_days,
        min_rows=args.min_rows,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_train_dates=args.min_train_dates,
        fold_days=args.fold_days,
        max_folds=args.max_folds,
        min_brier_gain=args.min_brier_gain,
        include_gbm=args.include_gbm,
        max_rows=args.max_rows,
        max_market_families=args.max_market_families,
        family_min_rows=args.family_min_rows,
        serious_true_pairs_only=not args.include_synthetic_training,
    )
    with psycopg2.connect(PG_DSN) as conn:
        payload = train_ai_bet_selection_model(conn, cfg)
    print(_report(payload))


if __name__ == "__main__":
    main()
