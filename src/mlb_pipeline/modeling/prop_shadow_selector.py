"""Residual/CLV-aware shadow selector for MLB player props.

This module does not reopen bankroll props by itself.  It scores active prop
rows with the artifacts we already train: walk-forward policy, market-residual
probability, CLV beat probability, bookability, and exact-bucket trust.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

from .external_pick_ledger import attach_external_agreement
from .prop_candidate_engine import exceeds_non_lottery_line_cap
from .prop_k_distribution import score_k_v3_over_probability
from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_ET = ZoneInfo("America/New_York")
from mlb_pipeline.db import PG_DSN as _PG_DSN
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


@dataclass(frozen=True)
class ShadowSelectorConfig:
    pg_dsn: str = _PG_DSN
    report_date: date | None = None
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_shadow_selector_report.json"
    report_file: str = "mlb_prop_shadow_selector_latest.md"
    min_ev: float = 0.02
    min_clv_beat_prob: float = 0.55
    min_bookable_prob: float = 0.60
    min_close_capture_prob: float = 0.60
    min_line_available_prob: float = 0.60
    min_bucket_clv_beat_rate: float = 0.55
    min_hitter_projected_pa: float = 4.0
    min_pitcher_projected_bf: float = 20.0
    enable_micro_projection: bool = True
    micro_projection_min_prob_edge: float = 0.05
    micro_projection_min_ev: float = 0.03
    micro_projection_min_clv_beat_prob: float = 0.52
    micro_projection_min_bookable_prob: float = 0.75
    micro_projection_min_line_available_prob: float = 0.75
    micro_projection_min_close_capture_prob: float = 0.75
    micro_projection_min_bucket_clv_beat_rate: float = 0.50
    micro_projection_min_bucket_roi: float = -0.03
    micro_projection_min_selector_score: float = -0.10
    enable_external_agreement_micro: bool = True
    external_agreement_min_strength: float = 0.75
    top_n: int = 40


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    return v


def american_to_prob(price: Any) -> float | None:
    p = _clean_float(price)
    if p is None or p == 0:
        return None
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def ev_per_unit(prob: Any, price: Any) -> float | None:
    p = _clean_float(prob)
    pr = _clean_float(price)
    if p is None or pr is None or pr == 0:
        return None
    payout = pr / 100.0 if pr > 0 else 100.0 / abs(pr)
    return p * payout - (1.0 - p)


def _poisson_over_prob(mean: Any, line: Any) -> float | None:
    lam = _clean_float(mean)
    ln = _clean_float(line)
    if lam is None or ln is None or lam < 0:
        return None
    lam = max(0.0001, min(lam, 20.0))
    threshold = max(0, int(math.floor(ln)) + 1)
    cumulative = 0.0
    prob = math.exp(-lam)
    for k in range(threshold):
        if k == 0:
            prob = math.exp(-lam)
        elif k > 0:
            prob *= lam / k
        cumulative += prob
    return max(1e-6, min(1.0 - 1e-6, 1.0 - cumulative))


def _binom_over_prob(line: Any, n: int, p: float) -> float | None:
    ln = _clean_float(line)
    if ln is None:
        return None
    n = max(1, min(8, int(n)))
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    threshold = math.floor(ln)
    prob = 0.0
    for k in range(threshold + 1, n + 1):
        prob += math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
    return max(1e-6, min(1.0 - 1e-6, prob))


def _rate_like(value: Any, default: float) -> float:
    v = _clean_float(value)
    if v is None:
        return default
    if v > 1.0:
        v /= 100.0
    return max(0.0, min(1.0, v))


def _per_pa_hitter_outcomes(
    pa: Any,
    *,
    expected_hits: Any,
    expected_tb: Any,
    expected_hr: Any,
    walk_rate: float,
) -> dict[str, float] | None:
    pa_f = _clean_float(pa)
    if pa_f is None or pa_f < 1.0:
        return None
    n = max(1, min(7, int(round(pa_f))))
    hits_f = _clean_float(expected_hits)
    tb_f = _clean_float(expected_tb)
    hr_f = _clean_float(expected_hr)
    if hits_f is None and tb_f is not None:
        hits_f = tb_f / 1.45
    if tb_f is None and hits_f is not None:
        tb_f = hits_f * 1.45
    hits = max(1e-6, min(float(hits_f or 1e-6), float(n) * 0.64))
    tb = max(hits, min(float(tb_f or hits * 1.45), float(n) * 2.10))
    hr = hr_f if hr_f is not None else min(tb / 4.0, hits * 0.14)
    hr = max(0.0, min(float(hr or 0.0), hits * 0.50, float(n) * 0.18))

    non_hr_hits = max(0.0, hits - hr)
    non_hr_tb = max(non_hr_hits, min(tb - 4.0 * hr, non_hr_hits * 3.0 if non_hr_hits > 0 else 0.0))
    extra_non_hr_bases = max(0.0, non_hr_tb - non_hr_hits)
    triples = max(0.0, min(0.025 * n, non_hr_hits * 0.06, extra_non_hr_bases / 2.0))
    doubles = max(0.0, min(non_hr_hits - triples, extra_non_hr_bases - 2.0 * triples))
    singles = max(0.0, non_hr_hits - doubles - triples)

    p_walk = max(0.035, min(0.160, walk_rate))
    p_single = max(0.0, singles / n)
    p_double = max(0.0, doubles / n)
    p_triple = max(0.0, triples / n)
    p_hr = max(0.0, hr / n)
    non_zero = p_walk + p_single + p_double + p_triple + p_hr
    if non_zero > 0.95:
        scale = 0.95 / non_zero
        p_walk *= scale
        p_single *= scale
        p_double *= scale
        p_triple *= scale
        p_hr *= scale
        non_zero = 0.95
    return {
        "pa_events": float(n),
        "p_out": 1.0 - non_zero,
        "p_walk": p_walk,
        "p_single": p_single,
        "p_double": p_double,
        "p_triple": p_triple,
        "p_hr": p_hr,
        "p_hit": p_single + p_double + p_triple + p_hr,
    }


def _compound_tb_over_prob(
    line: Any,
    pa: Any,
    expected_tb: Any,
    expected_hits: Any = None,
    expected_hr: Any = None,
    walk_rate: float = 0.085,
) -> float | None:
    ln = _clean_float(line)
    pa_f = _clean_float(pa)
    tb_f = _clean_float(expected_tb)
    if ln is None or pa_f is None or tb_f is None or pa_f < 1.0 or tb_f < 0:
        return None
    probs = _per_pa_hitter_outcomes(
        pa_f,
        expected_hits=expected_hits,
        expected_tb=tb_f,
        expected_hr=expected_hr,
        walk_rate=walk_rate,
    )
    if probs is None:
        return None
    n = int(probs["pa_events"])
    per_pa = {
        0: probs["p_out"] + probs["p_walk"],
        1: probs["p_single"],
        2: probs["p_double"],
        3: probs["p_triple"],
        4: probs["p_hr"],
    }
    dist = {0: 1.0}
    for _ in range(n):
        nxt: dict[int, float] = {}
        for current, p_current in dist.items():
            for add, p_add in per_pa.items():
                nxt[current + add] = nxt.get(current + add, 0.0) + p_current * p_add
        dist = nxt
    threshold = math.floor(ln)
    return max(1e-6, min(1.0 - 1e-6, sum(prob for total, prob in dist.items() if total > threshold)))


def _distribution_side_prob(row: dict[str, Any], side: str, line: Any) -> float | None:
    stat = str(row.get("stat") or row.get("market") or "")
    mean = _clean_float(row.get("pred_count") or row.get("pred_value"))
    walk_rate = max(0.035, min(0.160, 0.70 * _rate_like(row.get("opp_sp_bb_pct"), 0.085) + 0.30 * 0.085))
    p_over: float | None = None
    if stat == "batter_hits":
        pa = _clean_float(row.get("projected_pa"))
        if pa is not None and mean is not None and pa >= 1.0:
            probs = _per_pa_hitter_outcomes(
                pa,
                expected_hits=mean,
                expected_tb=row.get("batter_vs_hand_tb_avg_10"),
                expected_hr=row.get("batter_vs_hand_hr_avg_10"),
                walk_rate=walk_rate,
            )
            if probs is not None:
                p_over = _binom_over_prob(line, int(probs["pa_events"]), probs["p_hit"])
    elif stat == "batter_total_bases":
        p_over = _compound_tb_over_prob(
            line,
            row.get("projected_pa"),
            mean,
            row.get("batter_vs_hand_hits_avg_10"),
            row.get("batter_vs_hand_hr_avg_10"),
            walk_rate,
        )
    elif stat == "batter_home_runs":
        pa = _clean_float(row.get("projected_pa"))
        if pa is not None and mean is not None and pa >= 1.0:
            probs = _per_pa_hitter_outcomes(
                pa,
                expected_hits=row.get("batter_vs_hand_hits_avg_10"),
                expected_tb=row.get("batter_vs_hand_tb_avg_10"),
                expected_hr=mean,
                walk_rate=walk_rate,
            )
            if probs is not None:
                p_over = _binom_over_prob(line, int(probs["pa_events"]), probs["p_hr"])
    if p_over is None:
        p_over = _poisson_over_prob(mean, line)
    if p_over is None:
        return None
    return p_over if side == "over" else 1.0 - p_over if side == "under" else None


_DISTRIBUTION_CALIBRATION_GROUP_SPECS = (
    ("market", "side", "line_surface", "line_bucket", "price_bucket", "bookmaker_key"),
    ("market", "side", "line_surface", "line_bucket", "price_bucket"),
    ("market", "side", "line_surface", "line_bucket"),
    ("market", "side", "line_bucket"),
    ("market", "side", "line_surface"),
    ("market", "side"),
)


def _apply_serialized_probability_calibrator(probability: float, model: dict[str, Any]) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(probability)))
    if model.get("method") == "beta":
        coef = list(model.get("coef") or [0.0, 0.0])
        while len(coef) < 2:
            coef.append(0.0)
        z = (
            float(model.get("intercept") or 0.0)
            + float(coef[0]) * math.log(p)
            + float(coef[1]) * math.log1p(-p)
        )
        calibrated = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))
        return max(1e-6, min(1.0 - 1e-6, calibrated))
    if model.get("method") == "isotonic":
        xs = [float(value) for value in (model.get("x_thresholds") or [])]
        ys = [float(value) for value in (model.get("y_thresholds") or [])]
        if len(xs) >= 2 and len(xs) == len(ys):
            if p <= xs[0]:
                return max(1e-6, min(1.0 - 1e-6, ys[0]))
            if p >= xs[-1]:
                return max(1e-6, min(1.0 - 1e-6, ys[-1]))
            for idx in range(1, len(xs)):
                if p <= xs[idx]:
                    span = max(1e-12, xs[idx] - xs[idx - 1])
                    weight = (p - xs[idx - 1]) / span
                    value = ys[idx - 1] + weight * (ys[idx] - ys[idx - 1])
                    return max(1e-6, min(1.0 - 1e-6, value))
    return p


def _apply_distribution_probability_calibration(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    probability: float | None,
) -> tuple[float | None, str | None]:
    p = _clean_float(probability)
    if p is None:
        return probability, None
    if str(feature_row.get("market") or "") != "pitcher_strikeouts":
        return p, None
    payload = ((ctx.distribution.get("distribution_calibrators") or {}).get("probability") or {})
    groups = payload.get("groups") or {}
    if not groups:
        return p, None
    market = str(feature_row.get("market") or "unknown")
    for cols in _DISTRIBUTION_CALIBRATION_GROUP_SPECS:
        values = [str(feature_row.get(col) or "unknown") for col in cols]
        group_key = "|".join([*cols, *values])
        for lookup_key in (group_key, f"{market}|{group_key}"):
            rec = groups.get(lookup_key) or {}
            if rec.get("enabled") and rec.get("holdout_enabled") is not False and rec.get("model"):
                return _apply_serialized_probability_calibrator(p, rec["model"]), lookup_key
    return p, None


def no_vig_side_prob(side: str, over_price: Any, under_price: Any, bet_price: Any) -> tuple[float | None, str]:
    over_raw = american_to_prob(over_price)
    under_raw = american_to_prob(under_price)
    side_v = (side or "").lower()
    if over_raw is not None and under_raw is not None and over_raw + under_raw > 0:
        over = over_raw / (over_raw + under_raw)
        under = under_raw / (over_raw + under_raw)
        return (over if side_v == "over" else under), "no_vig_prediction_pair"
    return american_to_prob(bet_price), "raw_implied"


def _pair_quality(row: dict[str, Any], market_prob_source: str | None = None) -> str:
    current = str(row.get("pair_quality") or "").strip().lower()
    if current in {"same_book", "cross_book", "synthetic", "one_sided", "ladder"}:
        return current
    source = str(row.get("paired_price_source") or "").strip().lower()
    market_source = str(market_prob_source or row.get("market_prob_source") or "").strip().lower()
    if "synthetic" in source or "synthetic" in market_source:
        return "synthetic"
    if market_source == "no_vig_prediction_pair":
        return "same_book"
    if "same_book" in source or market_source == "no_vig_same_book":
        return "same_book"
    if "cross_book" in source or market_source == "no_vig_cross_book_exact_line":
        return "cross_book"
    if market_source in {"one_sided_fanduel_ladder", "ladder_implied"}:
        return "ladder"
    if market_source in {"raw_implied", "raw_implied_one_sided", "one_sided"}:
        return "one_sided"
    return "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


class SelectorContext:
    def __init__(self, model_dir: Path = _MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.walk_forward = _load_json(self.model_dir / "prop_walk_forward_accuracy_report.json")
        self.residual = _load_json(self.model_dir / "prop_market_residual_models.json")
        self.clv_history_state = self.residual.get("clv_history_state") or {}
        self.distribution = _load_json(self.model_dir / "prop_distribution_models.json")
        self.bookability = _load_json(self.model_dir / "prop_bookability_model.json")
        self.trust = _load_json(self.model_dir / "prop_bucket_trust_scores.json")
        self.tb15_line_calibration = _load_json(self.model_dir / "prop_tb15_line_calibrators.json")
        self.k_under_repair = _load_json(self.model_dir / "prop_k_under_repair.json")
        self.micro_probability_calibrator = _load_json(self.model_dir / "prop_micro_probability_calibrator.json")
        self.exact_bucket_clv_priors = _load_json(self.model_dir / "prop_exact_bucket_clv_priors.json")
        self.micro_promotion_evaluation = _load_json(self.model_dir / "prop_micro_promotion_evaluation.json")
        self.promotion = _load_json(self.model_dir / "prop_bucket_promotion_report.json")
        self.tb_hr_line_gates = (
            (self.distribution.get("tb_hr_line_production_gates") or {}).get("groups") or {}
        )

        self.live_policy = self.walk_forward.get("live_policy") or {}
        self.residual_buckets = {
            str(rec.get("bucket")): dict(rec)
            for rec in (self.residual.get("bucket_recommendations") or [])
            if rec.get("bucket")
        }
        self.distribution_buckets = {
            str(rec.get("bucket")): dict(rec)
            for rec in (self.distribution.get("bucket_model_selection") or [])
            if rec.get("bucket")
        }
        self.trust_scores = {
            str(key): dict(value or {})
            for key, value in (self.trust.get("bucket_scores") or {}).items()
        }
        self.exact_bucket_clv = {
            str(key): dict(value or {})
            for key, value in (self.exact_bucket_clv_priors.get("buckets") or {}).items()
        }
        self.micro_trial_buckets = {
            str(row.get("bucket")): dict(row)
            for row in (self.micro_promotion_evaluation.get("buckets") or [])
            if row.get("bucket")
        }
        self.bookability_targets = self.bookability.get("targets") or {}
        capture_target = self.bookability_targets.get("valid_close_snapshot_captured") or {}
        availability_target = self.bookability_targets.get("line_available_at_close") or {}
        holdout = capture_target.get("holdout") or self.bookability.get("holdout") or {}
        self.bookability_model_usable = bool(
            holdout.get("model_usable", True)
            and (capture_target.get("selected_scoring_method") or self.bookability.get("selected_scoring_method", "logistic")) == "logistic"
        )
        availability_holdout = availability_target.get("holdout") or {}
        self.line_availability_model_usable = bool(
            availability_holdout.get("model_usable")
            and availability_target.get("selected_scoring_method") == "logistic"
        )
        self.bookability_rates = self.bookability.get("empirical_bookability_rates") or {}
        self.default_bookability_rate = _clean_float(holdout.get("actual_bookable_rate"))
        self.default_close_capture_rate = self.default_bookability_rate
        self.default_line_available_rate = _clean_float(availability_holdout.get("actual_bookable_rate"))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logistic_score(model: dict[str, Any] | None, row: dict[str, Any]) -> float | None:
    if not model:
        return None
    coef = model.get("coef") or {}
    if not isinstance(coef, dict):
        return None
    try:
        z = float(model.get("intercept") or 0.0)
    except (TypeError, ValueError):
        z = 0.0
    means = model.get("numeric_means") or {}
    scales = model.get("numeric_scales") or {}
    numeric_features = model.get("numeric_features") or list((model.get("numeric_means") or {}).keys())
    categorical_features = model.get("categorical_features") or list((model.get("categorical_values") or {}).keys())
    for name in numeric_features:
        mean = _clean_float(means.get(name))
        if mean is None:
            mean = 0.0
        scale = _clean_float(scales.get(name))
        if scale is None or abs(scale) <= 1e-9:
            scale = 1.0
        value = _clean_float(row.get(name))
        if value is None:
            value = mean
        z += float(coef.get(name) or 0.0) * ((value - mean) / scale)
    for name in categorical_features:
        value = str(row.get(name) if row.get(name) is not None else "unknown")
        z += float(coef.get(f"{name}={value}") or 0.0)
    return max(1e-6, min(1.0 - 1e-6, _sigmoid(z)))


def _linear_record_score(model: dict[str, Any] | None, row: dict[str, Any]) -> float | None:
    if not model:
        return None
    coef = model.get("coef") or {}
    try:
        value = float(model.get("intercept") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    means = model.get("numeric_means") or {}
    scales = model.get("numeric_scales") or {}
    for name in model.get("numeric_features") or []:
        mean = _clean_float(means.get(name)) or 0.0
        scale = _clean_float(scales.get(name)) or 1.0
        if abs(scale) <= 1e-9:
            scale = 1.0
        feature = _clean_float(row.get(name))
        if feature is None:
            feature = mean
        value += float(coef.get(name) or 0.0) * ((feature - mean) / scale)
    for name in model.get("categorical_features") or []:
        category = str(row.get(name) if row.get(name) is not None else "unknown")
        value += float(coef.get(f"{name}={category}") or 0.0)
    return value


def _linear_score(model: dict[str, Any] | None, row: dict[str, Any]) -> float | None:
    if not model or not model.get("enabled"):
        return None
    return _linear_record_score(model, row)


def _expected_clv_score(
    model: dict[str, Any] | None,
    row: dict[str, Any],
    direction_probability: float | None,
) -> float | None:
    if not model or not model.get("enabled"):
        return None
    if model.get("method") != "conditional_probability_space_clv_v4":
        return _linear_record_score(model, row)
    if direction_probability is None:
        return None
    positive = _linear_record_score(model.get("positive_magnitude"), row)
    non_positive = _linear_record_score(model.get("non_positive_magnitude"), row)
    if positive is None or non_positive is None:
        return None
    p = max(1e-6, min(1.0 - 1e-6, float(direction_probability)))
    return p * max(0.0, min(25.0, positive)) - (1.0 - p) * max(0.0, min(25.0, non_positive))


def _clv_v3_row(ctx: SelectorContext, row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    minutes = max(0.0, _clean_float(out.get("minutes_to_first_pitch_at_lock")) or 0.0)
    age = max(0.0, _clean_float(out.get("lock_price_age_minutes")) or 0.0)
    elapsed = max(1.0, _clean_float(out.get("open_to_lock_minutes")) or 0.0)
    move = _clean_float(out.get("open_to_lock_prob_move"))
    line_move = _clean_float(out.get("open_to_lock_line_move_side"))
    lead_lag = _clean_float(out.get("book_lead_lag_prob"))
    dispersion = abs(_clean_float(out.get("consensus_price_dispersion")) or 0.0)
    model_prob = _clean_float(out.get("model_prob_side"))
    market_prob = _clean_float(out.get("market_prob_side"))
    best = _clean_float(out.get("best_consensus_prob"))
    worst = _clean_float(out.get("worst_consensus_prob"))
    lock_implied = _clean_float(out.get("lock_price_implied"))
    open_implied = _clean_float(out.get("open_price_implied"))
    consensus = _clean_float(out.get("consensus_prob_at_lock"))
    out.update({
        "minutes_to_first_pitch_log": math.log1p(minutes),
        "lock_price_age_log": math.log1p(age),
        "open_to_lock_move_per_hour": move / max(elapsed / 60.0, 1.0 / 60.0) if move is not None else None,
        "consensus_range": abs(worst - best) if worst is not None and best is not None else None,
        "book_lead_lag_z": lead_lag / max(dispersion, 0.005) if lead_lag is not None else None,
        "model_market_abs_disagreement": abs(model_prob - market_prob) if model_prob is not None and market_prob is not None else None,
        "model_move_agreement": (
            math.copysign(1.0, model_prob - market_prob) * math.copysign(1.0, move)
            if model_prob is not None and market_prob is not None and move not in (None, 0.0) and model_prob != market_prob
            else 0.0
        ),
        "line_price_move_agreement": (
            math.copysign(1.0, line_move) * math.copysign(1.0, move)
            if line_move not in (None, 0.0) and move not in (None, 0.0)
            else 0.0
        ),
        "lock_consensus_gap": lock_implied - consensus if lock_implied is not None and consensus is not None else None,
        "open_consensus_gap": open_implied - consensus if open_implied is not None and consensus is not None else None,
        "movement_dispersion_ratio": (move / max(dispersion, 0.005)) if move is not None else None,
        "time_pressure": 1.0 / (1.0 + minutes / 60.0),
        "consensus_book_count_log": math.log1p(max(0.0, _clean_float(out.get("consensus_book_count")) or 0.0)),
        "availability_pair_interaction": (
            (_clean_float(out.get("lock_offer_available")) or 0.0)
            * (_clean_float(out.get("lock_same_book_pair_available")) or 0.0)
        ),
    })
    history_key = "|".join([
        str(out.get("bookmaker_key") or "unknown"),
        str(out.get("market") or "unknown"),
        str(out.get("side") or "unknown"),
    ])
    for name, value in (ctx.clv_history_state.get(history_key) or {}).items():
        if out.get(name) is None:
            out[name] = value
    return out


def _event_side_line_model(ctx: SelectorContext) -> dict[str, Any] | None:
    model = (
        ((ctx.distribution.get("distribution_calibrators") or {}).get("side_line_models") or {})
        .get("win_probability")
    )
    if not isinstance(model, dict) or model.get("status") != "trained":
        return None
    brier_model = _clean_float(model.get("brier_model_holdout"))
    brier_baseline = _clean_float(model.get("brier_baseline_holdout"))
    if brier_model is not None and brier_baseline is not None and brier_model > brier_baseline - 0.001:
        return None
    return model


def _event_side_line_clv_model(ctx: SelectorContext) -> dict[str, Any] | None:
    model = (
        ((ctx.distribution.get("distribution_calibrators") or {}).get("side_line_models") or {})
        .get("clv_beat_probability")
    )
    if not isinstance(model, dict) or model.get("status") != "trained":
        return None
    brier_model = _clean_float(model.get("brier_model_holdout"))
    brier_baseline = _clean_float(model.get("brier_baseline_holdout"))
    if brier_model is not None and brier_baseline is not None and brier_model > brier_baseline - 0.001:
        return None
    return model


def _event_side_line_preferred(ctx: SelectorContext) -> bool:
    overall = ctx.distribution.get("overall") or {}
    event_brier = _clean_float(((overall.get("event_side_line") or {}).get("forecast") or {}).get("brier"))
    if event_brier is None:
        return False
    other_briers: list[float] = []
    for variant in ("model_only", "market_no_vig", "distribution", "distribution_empirical_blend"):
        brier = _clean_float(((overall.get(variant) or {}).get("forecast") or {}).get("brier"))
        if brier is not None:
            other_briers.append(brier)
    return bool(other_briers) and event_brier <= min(other_briers) - 0.001


def _event_side_line_score(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    distribution_prob: float | None,
) -> float | None:
    model = _event_side_line_model(ctx)
    if model is None:
        return None
    row = dict(feature_row)
    dist = distribution_prob
    model_prob = _clean_float(row.get("model_prob_side"))
    market_prob = _clean_float(row.get("market_prob_side"))
    if dist is None:
        dist = model_prob
    blend_values = [v for v in (dist, model_prob, market_prob) if v is not None]
    blend = sum(blend_values) / len(blend_values) if blend_values else None
    row.update({
        "p_distribution_side": dist,
        "p_distribution_calibrated": dist,
        "p_distribution_blend": blend,
        "p_empirical_bucket": market_prob if market_prob is not None else blend,
        "opp_model_pa": row.get("opp_model_pa") or row.get("projected_pa"),
    })
    return _logistic_score(model, row)


def _event_side_line_clv_score(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    distribution_prob: float | None,
) -> float | None:
    model = _event_side_line_clv_model(ctx)
    if model is None:
        return None
    row = dict(feature_row)
    dist = distribution_prob
    model_prob = _clean_float(row.get("model_prob_side"))
    market_prob = _clean_float(row.get("market_prob_side"))
    if dist is None:
        dist = model_prob
    blend_values = [v for v in (dist, model_prob, market_prob) if v is not None]
    blend = sum(blend_values) / len(blend_values) if blend_values else None
    row.update({
        "p_distribution_side": dist,
        "p_distribution_calibrated": dist,
        "p_distribution_blend": blend,
        "p_empirical_bucket": market_prob if market_prob is not None else blend,
        "opp_model_pa": row.get("opp_model_pa") or row.get("projected_pa"),
    })
    return _logistic_score(model, row)


def _bookability_empirical_components(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    bucket_key: str,
) -> tuple[float | None, float | None, str | None]:
    parts = bucket_key.split("|")
    if len(parts) < 6:
        return ctx.default_close_capture_rate, ctx.default_line_available_rate, "holdout_default"
    stat, side, surface, line_bucket_value, price_bucket_value, book = parts[:6]
    lookups = [
        ("exact_bucket", bucket_key),
        ("line_surface_book", "|".join([stat, side, surface, book])),
        ("line_surface", "|".join([stat, side, surface])),
        ("market_side_book", "|".join([stat, side, book])),
        ("market_side", "|".join([stat, side])),
    ]
    for level, key in lookups:
        rec = ((ctx.bookability_rates.get(level) or {}).get(key) or {})
        close_rate = _clean_float(rec.get("close_capture_rate"))
        if close_rate is None:
            close_rate = _clean_float(rec.get("bookable_rate"))
        line_rate = _clean_float(rec.get("line_available_rate"))
        if close_rate is not None and int(rec.get("rows") or 0) >= 10:
            return (
                max(1e-6, min(1.0 - 1e-6, close_rate)),
                max(1e-6, min(1.0 - 1e-6, line_rate)) if line_rate is not None else None,
                level,
            )
    return ctx.default_close_capture_rate, ctx.default_line_available_rate, "holdout_default"


def _bookability_score(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    bucket_key: str,
) -> tuple[float | None, str | None, float | None, float | None]:
    empirical_close, empirical_line, empirical_source = _bookability_empirical_components(ctx, feature_row, bucket_key)
    models = ctx.bookability.get("models") or {}
    close_capture_prob = empirical_close
    line_available_prob = empirical_line
    close_source = empirical_source
    line_source = empirical_source if empirical_line is not None else None
    if ctx.bookability_model_usable:
        logistic = _logistic_score(models.get("valid_close_snapshot_captured") or models.get("global"), feature_row)
        if logistic is not None:
            close_capture_prob = logistic
            close_source = "close_capture_logistic"
    if ctx.line_availability_model_usable:
        logistic = _logistic_score(models.get("line_available_at_close"), feature_row)
        if logistic is not None:
            line_available_prob = logistic
            line_source = "line_available_logistic"
    components = [p for p in (close_capture_prob, line_available_prob) if p is not None]
    combined = min(components) if components else None
    source_parts = [f"capture:{close_source or 'none'}"]
    if line_available_prob is not None:
        source_parts.append(f"line:{line_source or 'empirical'}")
    return combined, ";".join(source_parts), line_available_prob, close_capture_prob


def _side_model_prob(row: dict[str, Any]) -> float | None:
    p_over = _clean_float(row.get("pred_prob_over") or row.get("model_prob_over"))
    side = str(row.get("bet_side") or row.get("side") or "").lower()
    if p_over is None:
        p_side = _clean_float(row.get("model_prob_side"))
        return p_side
    return p_over if side == "over" else 1.0 - p_over if side == "under" else None


def _book_key(row: dict[str, Any]) -> str:
    return str(row.get("bookmaker_key") or row.get("book") or "unknown").lower()


def _prediction_price(row: dict[str, Any]) -> float | None:
    return _clean_float(row.get("bet_price") or row.get("market_price") or row.get("price"))


def _prediction_line(row: dict[str, Any]) -> float | None:
    return _clean_float(row.get("book_line") or row.get("market_line") or row.get("line"))


def _line_surface(stat: str, side: str, line: Any, row: dict[str, Any]) -> str:
    return str(row.get("line_surface") or prop_line_surface(stat, side, line))


def _line_bucket(stat: str, line: Any, row: dict[str, Any]) -> str:
    current = str(row.get("line_bucket") or "")
    return current if current and current != "unknown" else prop_line_bucket(stat, line)


def _price_bucket(price: Any, row: dict[str, Any]) -> str:
    current = str(row.get("price_bucket") or "")
    return current if current and current != "unknown" else price_bucket(price)


def exact_bucket_key(row: dict[str, Any]) -> str:
    stat = str(row.get("stat") or row.get("market") or "")
    side = str(row.get("bet_side") or row.get("side") or "").lower()
    line = _prediction_line(row)
    price = _prediction_price(row)
    return "|".join([
        stat,
        side,
        _line_surface(stat, side, line, row),
        _line_bucket(stat, line, row),
        _price_bucket(price, row),
        _book_key(row),
    ])


def _walk_forward_record(ctx: SelectorContext, bucket_key: str) -> tuple[str, dict[str, Any] | None]:
    parts = bucket_key.split("|")
    if len(parts) < 6:
        return "none", None
    stat, side, surface, line_bucket_value, price_bucket_value, book = parts[:6]
    exact = f"{stat}|{side}|{surface}|{line_bucket_value}|{price_bucket_value}|{book}"
    line_surface_key = f"{stat}|{side}|{surface}"
    market_side_key = f"{stat}|{side}"
    for level, key in (
        ("exact_bucket", exact),
        ("line_surface", line_surface_key),
        ("market_side", market_side_key),
    ):
        rec = (ctx.live_policy.get(level) or {}).get(key)
        if rec:
            return level, dict(rec)
    return "none", None


def _residual_bucket_record(ctx: SelectorContext, bucket_key: str) -> dict[str, Any] | None:
    return ctx.residual_buckets.get(bucket_key)


def _distribution_bucket_record(ctx: SelectorContext, bucket_key: str) -> dict[str, Any] | None:
    return ctx.distribution_buckets.get(bucket_key)


def _model_row(row: dict[str, Any], bucket_key: str) -> dict[str, Any]:
    context = row.get("opportunity_context")
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (TypeError, ValueError, json.JSONDecodeError):
            context = {}
    if isinstance(context, dict):
        merged = dict(row)
        merged.update({key: value for key, value in context.items() if value is not None})
        row = merged
    stat, side, surface, line_bucket_value, price_bucket_value, book = (bucket_key.split("|") + [""] * 6)[:6]
    price = _prediction_price(row)
    model_prob = _side_model_prob(row)
    training_market_prob = _clean_float(row.get("training_market_prob_side") or row.get("market_prob_side"))
    paired_market_prob, market_prob_source = no_vig_side_prob(
        side,
        row.get("over_price"),
        row.get("under_price"),
        price,
    )
    market_prob = training_market_prob if training_market_prob is not None else paired_market_prob
    market_source = str(row.get("market_prob_source") or market_prob_source)
    pair_quality = _pair_quality(row, market_source)
    same_book_pair_flag = 1.0 if pair_quality == "same_book" else 0.0
    cross_book_pair_flag = 1.0 if pair_quality == "cross_book" else 0.0
    synthetic_pair_flag = 1.0 if pair_quality == "synthetic" else 0.0
    true_pair_flag = 1.0 if pair_quality in {"same_book", "cross_book"} else 0.0
    clean_market_pair_flag = (
        1.0
        if true_pair_flag
        and market_source not in {"raw_implied", "raw_implied_one_sided", "synthetic_fanduel_over_only", "one_sided_fanduel_ladder"}
        else 0.0
    )
    line = _prediction_line(row)
    pred_count = _clean_float(row.get("pred_count"))
    count_edge = None
    if pred_count is not None and line is not None:
        count_edge = pred_count - line if side == "over" else line - pred_count
    return {
        **row,
        "market": stat,
        "stat": stat,
        "side": side,
        "bet_side": side,
        "line_surface": surface,
        "line_bucket": line_bucket_value,
        "price_bucket": price_bucket_value,
        "bookmaker_key": book,
        "market_line": line,
        "market_price": price,
        "abs_price": abs(price) if price is not None else None,
        "is_plus_price": 1.0 if price is not None and price > 0 else (0.0 if price is not None else None),
        "model_prob_side": model_prob,
        "market_prob_side": market_prob,
        "market_prob_source": market_source,
        "paired_price_source": row.get("paired_price_source"),
        "pair_quality": pair_quality,
        "same_book_pair_flag": _clean_float(row.get("same_book_pair_flag")) if row.get("same_book_pair_flag") is not None else same_book_pair_flag,
        "cross_book_pair_flag": _clean_float(row.get("cross_book_pair_flag")) if row.get("cross_book_pair_flag") is not None else cross_book_pair_flag,
        "synthetic_pair_flag": _clean_float(row.get("synthetic_pair_flag")) if row.get("synthetic_pair_flag") is not None else synthetic_pair_flag,
        "clean_market_pair_flag": _clean_float(row.get("clean_market_pair_flag")) if row.get("clean_market_pair_flag") is not None else clean_market_pair_flag,
        "true_pair_flag": _clean_float(row.get("true_pair_flag")) if row.get("true_pair_flag") is not None else true_pair_flag,
        "minutes_to_first_pitch_at_lock": _clean_float(row.get("minutes_to_first_pitch_at_lock")),
        "lock_price_age_minutes": _clean_float(row.get("lock_price_age_minutes")),
        "distribution_prob_side": _distribution_side_prob(row, side, line),
        "prob_edge_vs_market": (
            model_prob - market_prob
            if model_prob is not None and market_prob is not None
            else None
        ),
        "count_edge_side": count_edge,
        "edge_type": row.get("edge_type") or "unknown",
        "model_family": row.get("model_family") or "unknown",
        "clv_unknown_reason": row.get("clv_unknown_reason") or "unknown",
    }


def _policy_prob(
    variant: str,
    model_prob: float | None,
    market_prob: float | None,
    residual_prob: float | None,
    distribution_prob: float | None,
    event_side_line_prob: float | None,
    k_v3_prob: float | None,
    policy_rec: dict[str, Any] | None,
) -> float | None:
    if variant == "market_no_vig":
        return market_prob
    if variant == "market_residual":
        return residual_prob
    if variant == "distribution":
        return distribution_prob
    if variant == "event_side_line":
        return event_side_line_prob
    if variant == "k_v3":
        return k_v3_prob
    if variant == "distribution_blend":
        if distribution_prob is None:
            return market_prob if market_prob is not None else model_prob
        if market_prob is None:
            return distribution_prob
        return 0.55 * distribution_prob + 0.45 * market_prob
    if variant == "walk_forward_blend":
        if model_prob is None:
            return market_prob
        if market_prob is None:
            return model_prob
        weight = _clean_float((policy_rec or {}).get("model_weight"))
        if weight is None:
            weight = 0.6
        weight = max(0.0, min(1.0, weight))
        return weight * model_prob + (1.0 - weight) * market_prob
    return model_prob


def _micro_projection_probability(
    *,
    stat: str,
    model_family: str | None = None,
    model_prob: float | None,
    distribution_prob: float | None,
    event_side_line_prob: float | None,
    k_v3_prob: float | None,
) -> tuple[float | None, str | None]:
    """Probability source for $1 projection-led testing, separate from bankroll proof."""
    if stat == "pitcher_strikeouts" and k_v3_prob is not None:
        return k_v3_prob, "k_v3_projection"
    if stat == "pitcher_strikeouts" and distribution_prob is not None:
        return distribution_prob, "k_distribution_calibrated"
    family = str(model_family or "").lower()
    if stat in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
        if stat == "batter_total_bases" and family == "tb_tail_state" and model_prob is not None:
            return model_prob, "tb_tail_state_projection"
        if distribution_prob is not None:
            return distribution_prob, "distribution_projection"
        if event_side_line_prob is not None:
            return event_side_line_prob, "event_side_line_projection"
        if stat == "batter_total_bases":
            return None, None
    if model_prob is not None:
        return model_prob, "model_projection"
    return None, None


def _micro_calibration_keys(feature_row: dict[str, Any]) -> list[str]:
    market = str(feature_row.get("market") or feature_row.get("stat") or "unknown")
    side = str(feature_row.get("side") or feature_row.get("bet_side") or "unknown").lower()
    line_bucket_value = str(feature_row.get("line_bucket") or prop_line_bucket(market, feature_row.get("market_line")))
    price_bucket_value = str(feature_row.get("price_bucket") or price_bucket(feature_row.get("market_price")))
    book = _book_key(feature_row)
    model_family = str(feature_row.get("model_family") or "unknown").lower()
    return [
        "|".join(["exact", market, side, line_bucket_value, price_bucket_value, book]),
        "|".join(["approved_model", market, side, line_bucket_value, book, model_family]),
        "|".join(["book_line", market, side, line_bucket_value, book]),
        "|".join(["line_price", market, side, line_bucket_value, price_bucket_value]),
        "|".join(["line", market, side, line_bucket_value]),
        "|".join(["market_side", market, side]),
        "global|*",
    ]


def _micro_unproven_cap(feature_row: dict[str, Any], probability: float) -> dict[str, Any]:
    market = str(feature_row.get("market") or feature_row.get("stat") or "unknown")
    side = str(feature_row.get("side") or feature_row.get("bet_side") or "unknown").lower()
    line = _clean_float(feature_row.get("market_line") or feature_row.get("book_line"))
    line_bucket_value = str(feature_row.get("line_bucket") or prop_line_bucket(market, line))
    cap = 0.68
    target = 0.58
    shrink = 0.45
    reason = "unproven_high_confidence_micro_cap"
    if market == "batter_total_bases" and side == "over" and line is not None and abs(line - 1.5) <= 1e-9:
        cap = 0.62 if probability >= 0.75 else 0.64
        target = 0.57
        shrink = 0.55
        reason = "unproven_tb15_micro_confidence_cap"
    elif market == "pitcher_strikeouts" and side == "under":
        cap = 0.64 if probability >= 0.75 else 0.66
        target = 0.56
        shrink = 0.50
        reason = "unproven_k_under_micro_confidence_cap"
    elif market in {"batter_hits", "batter_home_runs"}:
        cap = 0.66
        target = 0.56
        shrink = 0.50
        reason = "unproven_hitter_micro_confidence_cap"
    return {
        "enabled": True,
        "proven": False,
        "target_probability": target,
        "shrink_weight": shrink,
        "max_probability": cap,
        "method": "mandatory_unproven_micro_shrink",
        "level": "unproven_default",
        "line_bucket": line_bucket_value,
        "reason": reason,
    }


def _apply_micro_shrink_rule(probability: float, rec: dict[str, Any]) -> float:
    target = _clean_float(rec.get("target_probability"))
    weight = _clean_float(rec.get("shrink_weight"))
    cap = _clean_float(rec.get("max_probability"))
    if target is None or weight is None:
        return probability
    weight = max(0.0, min(0.95, weight))
    adjusted = probability
    if probability > target:
        adjusted = probability - weight * (probability - target)
    if not rec.get("proven") and probability >= 0.70 and cap is not None:
        adjusted = min(adjusted, cap)
    return max(1e-6, min(1.0 - 1e-6, min(probability, adjusted)))


def _apply_micro_probability_calibration(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    probability: float | None,
    *,
    allow_broad_fallback: bool = True,
) -> tuple[float | None, str | None, dict[str, Any]]:
    p = _clean_float(probability)
    if p is None:
        return probability, None, {}
    payload = ctx.micro_probability_calibrator or {}
    default_rec = _micro_unproven_cap(feature_row, p)
    if not payload.get("enabled"):
        adjusted = _apply_micro_shrink_rule(p, default_rec) if p >= 0.70 else p
        return adjusted, "unproven_default|micro_probability_cap" if adjusted < p else None, default_rec if adjusted < p else {}
    groups = payload.get("groups") or {}
    keys = _micro_calibration_keys(feature_row)
    if not allow_broad_fallback:
        approved_keys = [key for key in keys if key.startswith("approved_model|")]
        keys = approved_keys + [
            key for key in keys
            if not key.startswith("approved_model|")
        ]
    for key in keys:
        rec = groups.get(key) or {}
        if not rec.get("enabled"):
            continue
        if not rec.get("proven") and p >= 0.70:
            rec = dict(rec)
            rec["max_probability"] = min(
                _clean_float(rec.get("max_probability")) or default_rec["max_probability"],
                default_rec["max_probability"],
            )
            rec["target_probability"] = min(
                _clean_float(rec.get("target_probability")) or default_rec["target_probability"],
                default_rec["target_probability"],
            )
            rec["shrink_weight"] = max(
                _clean_float(rec.get("shrink_weight")) or default_rec["shrink_weight"],
                default_rec["shrink_weight"],
            )
            rec["fallback_reason"] = default_rec["reason"]
        adjusted = _apply_micro_shrink_rule(p, rec)
        # This calibrator is deliberately one-way. Micro results can reduce
        # fake confidence, but they do not inflate a probability into a bet.
        return adjusted, key, rec
    adjusted = _apply_micro_shrink_rule(p, default_rec) if p >= 0.70 else p
    return adjusted, "unproven_default|micro_probability_cap" if adjusted < p else None, default_rec if adjusted < p else {}


def _apply_tb15_high_pa_power_micro_cap(
    feature_row: dict[str, Any],
    probability: float | None,
    calibration: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any]]:
    """Reduce unproven TB 1.5 confidence for high-PA power profiles."""
    p = _clean_float(probability)
    if p is None:
        return probability, {}
    market = str(feature_row.get("market") or feature_row.get("stat") or "")
    side = str(feature_row.get("side") or feature_row.get("bet_side") or "").lower()
    line = _clean_float(feature_row.get("market_line") or feature_row.get("book_line"))
    if market != "batter_total_bases" or side != "over" or line is None or abs(line - 1.5) > 1e-9:
        return p, {}
    if bool((calibration or {}).get("proven")):
        return p, {}

    projected_pa = _clean_float(feature_row.get("projected_pa"))
    iso = _clean_float(feature_row.get("batter_vs_hand_iso_avg_10"))
    hr_rate = _clean_float(feature_row.get("batter_vs_hand_hr_avg_10"))
    pred_tb = _clean_float(feature_row.get("pred_count") or feature_row.get("pred_value"))
    if projected_pa is None or projected_pa < 4.25:
        return p, {}

    power_flags: list[str] = []
    if iso is not None and iso >= 0.180:
        power_flags.append("iso")
    if hr_rate is not None and hr_rate >= 0.100:
        power_flags.append("hr_rate")
    if pred_tb is not None and pred_tb >= 2.0:
        power_flags.append("mean_tb")
    if not power_flags:
        return p, {}

    cap = 0.64
    if projected_pa >= 4.60 or len(power_flags) >= 2:
        cap = 0.60
    if projected_pa >= 4.75 and len(power_flags) >= 2:
        cap = 0.58
    capped = min(p, cap)
    return capped, {
        "applied": capped < p,
        "cap": cap,
        "projected_pa": projected_pa,
        "power_flags": ",".join(power_flags),
        "reason": "tb15_high_pa_power_unproven_tail_cap",
    }


def _approved_micro_trial_model(
    feature_row: dict[str, Any],
    *,
    stat: str,
    side: str,
    line: float | None,
    surface: str,
    pair_quality: str,
    k_under_micro_allowed: bool = False,
) -> tuple[bool, str | None, str]:
    """Narrow allowlist for $1 micro tests that collect proof before bankroll."""
    model_family = str(feature_row.get("model_family") or "unknown").lower()
    book = _book_key(feature_row)
    line_bucket_value = str(feature_row.get("line_bucket") or prop_line_bucket(stat, line))
    key = "|".join(["approved_model", stat, side, line_bucket_value, book, model_family])
    if (
        stat == "pitcher_strikeouts"
        and side == "under"
        and book == "draftkings"
        and pair_quality == "same_book"
        and surface == "common"
        and k_under_micro_allowed
    ):
        return True, key, "dk_k_under_repair_gate"
    if (
        stat == "batter_total_bases"
        and side == "over"
        and line is not None
        and abs(float(line) - 1.5) <= 1e-9
        and book == "draftkings"
        and pair_quality == "same_book"
        and surface == "common"
        and model_family == "tb_tail_state"
    ):
        return True, key, "dk_tb15_over_tb_tail_state"
    return False, key, "micro_trial_model_family_not_approved"


def _micro_truth_filter(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    approved_model_key: str | None,
) -> tuple[bool, str, str | None, dict[str, Any]]:
    """Use real locked micro results to keep or stop a trial bucket."""
    payload = ctx.micro_probability_calibrator or {}
    if not payload.get("enabled"):
        return True, "collecting_micro_truth", "micro_truth_no_results_yet", {}
    groups = payload.get("groups") or {}
    market = str(feature_row.get("market") or feature_row.get("stat") or "unknown")
    side = str(feature_row.get("side") or feature_row.get("bet_side") or "unknown").lower()
    line_bucket_value = str(feature_row.get("line_bucket") or prop_line_bucket(market, feature_row.get("market_line")))
    book = _book_key(feature_row)
    lookup_keys = [approved_model_key]
    rec: dict[str, Any] = {}
    rec_key = None
    for key in lookup_keys:
        if not key:
            continue
        candidate = groups.get(key) or {}
        if candidate.get("enabled"):
            rec = candidate
            rec_key = key
            break
    if not rec:
        return True, "collecting_micro_truth", "micro_truth_no_bucket_results_yet", {}

    graded = int(rec.get("graded") or 0)
    roi = _clean_float(rec.get("roi"))
    clv_rows = int(rec.get("clv_rows") or 0)
    clv_beat = _clean_float(rec.get("clv_beat_rate"))
    avg_clv = _clean_float(rec.get("avg_clv_price"))
    rec = dict(rec)
    rec["micro_truth_key"] = rec_key
    if graded < 15:
        return True, "collecting_micro_truth", "micro_truth_sample_building", rec

    losing_roi = roi is not None and roi < -0.03
    losing_clv = (
        clv_rows >= 10
        and clv_beat is not None
        and clv_beat < 0.50
        and (avg_clv is None or avg_clv <= 0.0)
    )
    if losing_roi or losing_clv:
        return False, "stopped_micro_truth", "micro_truth_filter_losing", rec

    if (
        graded >= 25
        and roi is not None and roi > 0.0
        and clv_rows >= 15
        and clv_beat is not None and clv_beat >= 0.55
        and avg_clv is not None and avg_clv > 0.0
    ):
        return True, "graduation_candidate", "micro_truth_graduation_candidate", rec

    return True, "collecting_micro_truth", "micro_truth_not_enough_for_graduation", rec


def _tb15_prob_bin(value: Any) -> str:
    p = _clean_float(value)
    if p is None:
        return "missing_prob"
    if p < 0.40:
        return "p<40"
    if p < 0.55:
        return "p40_55"
    if p < 0.70:
        return "p55_70"
    return "p70_plus"


def _tb15_apply_offset(prob: float, offset: Any) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(prob)))
    logit_value = math.log(p / (1.0 - p)) + float(offset or 0.0)
    return max(1e-6, min(1.0 - 1e-6, 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit_value))))))


def _apply_tb15_line_calibration(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
    probability: float | None,
) -> tuple[float | None, str | None]:
    p = _clean_float(probability)
    if p is None:
        return probability, None
    if str(feature_row.get("market") or "") != "batter_total_bases":
        return p, None
    if str(feature_row.get("side") or "").lower() != "over":
        return p, None
    if _book_key(feature_row) != "draftkings":
        return p, None
    if _pair_quality(feature_row) not in {"same_book", "cross_book"}:
        return p, None
    line = _clean_float(feature_row.get("market_line"))
    if line is None or abs(line - 1.5) > 1e-9:
        return p, None
    payload = ctx.tb15_line_calibration or {}
    calibrators = payload.get("calibrators") or {}
    if not calibrators:
        return p, None
    book = _book_key(feature_row)
    pb = str(feature_row.get("price_bucket") or price_bucket(feature_row.get("market_price") or feature_row.get("price")))
    keys = [f"{book}|{pb}", f"{book}|*", f"*|{pb}", "*|*"]
    for key in keys:
        cal = calibrators.get(key)
        if not cal:
            continue
        if not cal.get("enabled"):
            return p, None
        bin_cal = (cal.get("bins") or {}).get(_tb15_prob_bin(p)) or {}
        offset = bin_cal.get("offset", cal.get("offset"))
        return _tb15_apply_offset(p, offset), key
    return p, None


def _k_under_gate(
    ctx: SelectorContext,
    feature_row: dict[str, Any],
) -> tuple[bool, str | None, list[str]]:
    if str(feature_row.get("market") or "") != "pitcher_strikeouts":
        return True, None, []
    if str(feature_row.get("side") or "").lower() != "under":
        return True, None, []
    payload = ctx.k_under_repair or {}
    gates = payload.get("gates") or {}
    if not gates:
        return False, None, ["k_under_repair_gate_missing"]
    line_bucket_value = str(feature_row.get("line_bucket") or prop_line_bucket("pitcher_strikeouts", feature_row.get("market_line")))
    book = _book_key(feature_row)
    line = _clean_float(feature_row.get("market_line"))
    projected_bf = _clean_float(feature_row.get("projected_bf"))
    projected_pitch_count = _clean_float(feature_row.get("projected_pitch_count"))
    opp_team_k = _clean_float(feature_row.get("opp_team_k_pct_10") or feature_row.get("opp_team_k_pct_avg_10"))
    days_rest = _clean_float(feature_row.get("pitcher_days_rest") or feature_row.get("days_rest"))
    last_start_ip = _clean_float(feature_row.get("pitcher_last_ip") or feature_row.get("last_start_ip"))
    context_blockers: list[str] = []
    if projected_bf is None:
        context_blockers.append("projected_bf_missing")
    if projected_pitch_count is None:
        context_blockers.append("projected_pitch_count_missing")
    if line is not None and line < 4.5:
        if projected_bf is not None and projected_bf >= 23.5:
            context_blockers.append("low_line_under_high_bf_projection")
        if projected_pitch_count is not None and projected_pitch_count >= 88.0:
            context_blockers.append("low_line_under_high_pitch_count_projection")
        if opp_team_k is not None and opp_team_k >= 0.245:
            context_blockers.append("low_line_under_high_opponent_k_context")
        if days_rest is not None and days_rest >= 5.0 and last_start_ip is not None and last_start_ip >= 6.0:
            context_blockers.append("low_line_under_recent_long_leash")
    keys = [f"{line_bucket_value}|{book}", f"{line_bucket_value}|*", f"*|{book}", "*|*"]
    for key in keys:
        gate = gates.get(key)
        if not gate:
            continue
        blockers = [str(value) for value in (gate.get("blockers") or [])]
        blockers.extend(context_blockers)
        return bool(gate.get("micro_allowed")) and not context_blockers, key, blockers
    return False, None, ["k_under_repair_gate_missing", *context_blockers]


def _is_tail_alt(stat: str, side: str, line: float | None, surface: str) -> bool:
    if surface == "alt_tail":
        return True
    if side != "over" or line is None:
        return False
    return (
        (stat == "batter_hits" and line >= 1.5)
        or (stat == "batter_total_bases" and line >= 2.5)
        or (stat == "batter_home_runs" and line >= 1.5)
    )


def _is_fanduel_synthetic_hitter_evidence(feature_row: dict[str, Any]) -> bool:
    stat = str(feature_row.get("market") or feature_row.get("stat") or "")
    if stat not in {"batter_hits", "batter_total_bases", "batter_home_runs"}:
        return False
    if _book_key(feature_row) != "fanduel":
        return False
    pair_quality = str(feature_row.get("pair_quality") or "").lower()
    market_source = str(feature_row.get("market_prob_source") or "").lower()
    pair_source = str(feature_row.get("paired_price_source") or "").lower()
    synthetic_flag = _clean_float(feature_row.get("synthetic_pair_flag")) or 0.0
    return (
        pair_quality == "synthetic"
        or market_source == "synthetic_fanduel_over_only"
        or pair_source == "synthetic_fanduel_over_only_complement"
        or synthetic_flag >= 0.5
    )


def _external_platforms_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if str(item).strip())
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return ", ".join(str(item) for item in parsed if str(item).strip())
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return text
    return str(value)


def _external_agreement_micro_lane(
    feature_row: dict[str, Any],
    *,
    cfg: ShadowSelectorConfig,
    market_evidence_confirms: bool,
    synthetic_fanduel_evidence: bool,
    tail_alt: bool,
    line_cap_exceeded: bool,
) -> tuple[bool, str, list[str]]:
    blockers: list[str] = []
    if not cfg.enable_external_agreement_micro:
        return False, "external_agreement_micro_disabled", ["external_agreement_micro_disabled"]
    count = _clean_int(feature_row.get("external_agreement_count")) or 0
    strength = _clean_float(feature_row.get("external_agreement_strength")) or 0.0
    match_level = str(feature_row.get("external_match_level") or "")
    disagreements = _clean_int(feature_row.get("external_disagreement_count")) or 0
    platforms = _external_platforms_text(feature_row.get("external_platforms"))
    if count <= 0:
        blockers.append("external_agreement_missing")
    if strength < cfg.external_agreement_min_strength:
        blockers.append("external_agreement_not_exact_enough")
    if match_level not in {"exact_line_same_book", "exact_line_bookless", "exact_line_cross_book"}:
        blockers.append("external_agreement_line_not_exact")
    if disagreements > count:
        blockers.append("external_disagreement_outnumbers_agreement")
    if not market_evidence_confirms:
        blockers.append("external_agreement_needs_true_pair_market")
    if synthetic_fanduel_evidence:
        blockers.append("external_agreement_ignores_synthetic_fanduel")
    if tail_alt:
        blockers.append("external_agreement_no_lottery")
    if line_cap_exceeded:
        blockers.append("external_agreement_line_cap")
    if not platforms:
        blockers.append("external_agreement_platform_unknown")
    if blockers:
        return False, blockers[0], blockers
    return True, "external_model_agreement_micro_lane", []


def score_prediction_row(
    row: dict[str, Any],
    *,
    ctx: SelectorContext | None = None,
    cfg: ShadowSelectorConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or ShadowSelectorConfig()
    ctx = ctx or SelectorContext(cfg.model_dir)
    bucket_key = exact_bucket_key(row)
    feature_row = _model_row(row, bucket_key)
    feature_row = _clv_v3_row(ctx, feature_row)
    model_prob = _clean_float(feature_row.get("model_prob_side"))
    market_prob = _clean_float(feature_row.get("market_prob_side"))
    distribution_prob = _clean_float(feature_row.get("distribution_prob_side"))
    k_v3_prob = None
    if str(feature_row.get("market") or "") == "pitcher_strikeouts":
        k_v3_artifact = ctx.distribution.get("k_v3") or {}
        k_v3_over = score_k_v3_over_probability(
            feature_row,
            feature_row.get("market_line"),
            k_v3_artifact,
        )
        if k_v3_over is not None and k_v3_artifact.get("enabled"):
            k_v3_prob = 1.0 - k_v3_over if str(feature_row.get("side") or "") == "under" else k_v3_over
            distribution_prob = k_v3_prob
    distribution_calibration_key = None
    if str(feature_row.get("market") or "") == "pitcher_strikeouts":
        distribution_prob, distribution_calibration_key = _apply_distribution_probability_calibration(
            ctx,
            feature_row,
            distribution_prob,
        )
    event_side_line_prob = _event_side_line_score(ctx, feature_row, distribution_prob)
    tb15_calibration_key = None
    if str(feature_row.get("market") or "") == "batter_total_bases" and str(feature_row.get("side") or "") == "over":
        if distribution_prob is not None:
            distribution_prob, tb15_calibration_key = _apply_tb15_line_calibration(ctx, feature_row, distribution_prob)
        if event_side_line_prob is not None:
            event_side_line_prob, event_key = _apply_tb15_line_calibration(ctx, feature_row, event_side_line_prob)
            tb15_calibration_key = tb15_calibration_key or event_key
        if model_prob is not None and distribution_prob is None and event_side_line_prob is None:
            model_prob, tb15_calibration_key = _apply_tb15_line_calibration(ctx, feature_row, model_prob)
    price = _prediction_price(feature_row)
    breakeven = american_to_prob(price)
    residual_prob = _logistic_score((ctx.residual.get("models") or {}).get("global"), feature_row)
    residual_clv_prob = _logistic_score((ctx.residual.get("models") or {}).get("clv_beat"), feature_row)
    expected_clv_price = _expected_clv_score(
        (ctx.residual.get("models") or {}).get("clv_magnitude"),
        feature_row,
        residual_clv_prob,
    )
    event_side_line_clv_prob = _event_side_line_clv_score(ctx, feature_row, distribution_prob)
    clv_prob_candidates = [
        value for value in (residual_clv_prob, event_side_line_clv_prob)
        if value is not None
    ]
    clv_prob = min(clv_prob_candidates) if clv_prob_candidates else None
    bookable_prob, bookability_source, line_available_prob, close_capture_prob = _bookability_score(ctx, feature_row, bucket_key)

    policy_level, policy_rec = _walk_forward_record(ctx, bucket_key)
    residual_bucket = _residual_bucket_record(ctx, bucket_key)
    distribution_bucket = _distribution_bucket_record(ctx, bucket_key)
    trust = ctx.trust_scores.get(bucket_key) or {}
    exact_bucket_clv = ctx.exact_bucket_clv.get(bucket_key) or {}
    micro_eval_bucket = ctx.micro_trial_buckets.get(bucket_key) or {}
    exact_bucket_clv_prob = _clean_float(
        exact_bucket_clv.get("micro_clv_beat_rate")
        if exact_bucket_clv.get("micro_clv_beat_rate") is not None
        else exact_bucket_clv.get("shrunk_clv_beat_rate")
        if exact_bucket_clv.get("shrunk_clv_beat_rate") is not None
        else exact_bucket_clv.get("clv_beat_rate")
    )
    exact_bucket_avg_clv = _clean_float(
        exact_bucket_clv.get("micro_avg_clv_price")
        if exact_bucket_clv.get("micro_avg_clv_price") is not None
        else exact_bucket_clv.get("shrunk_avg_clv_price")
        if exact_bucket_clv.get("shrunk_avg_clv_price") is not None
        else exact_bucket_clv.get("avg_clv_price")
    )
    exact_bucket_clv_micro_confirmed = bool(exact_bucket_clv.get("micro_clv_confirmed"))
    micro_trial_ready = bool(micro_eval_bucket.get("micro_trial_ready"))
    pair_quality = str(feature_row.get("pair_quality") or "unknown").lower()
    market_prob_source = str(feature_row.get("market_prob_source") or "unknown").lower()
    synthetic_fanduel_evidence = _is_fanduel_synthetic_hitter_evidence(feature_row)
    ladder_market_confirms = (
        pair_quality == "ladder"
        and market_prob_source == "one_sided_fanduel_ladder"
        and not synthetic_fanduel_evidence
    )
    market_pair_confirms = (
        pair_quality in {"same_book", "cross_book"}
        and market_prob_source not in {"raw_implied", "synthetic_fanduel_over_only"}
        and not synthetic_fanduel_evidence
    )
    market_evidence_confirms = market_pair_confirms or ladder_market_confirms
    variant = str((policy_rec or {}).get("variant") or "model_only")
    if residual_bucket and str(residual_bucket.get("decision") or "").startswith("use_market_residual"):
        variant = "market_residual"
    elif distribution_bucket and market_evidence_confirms:
        distribution_decision = str(distribution_bucket.get("decision") or "")
        if distribution_decision == "use_event_curve_side_line" and event_side_line_prob is not None:
            variant = "event_side_line"
        elif distribution_decision == "use_market_only" and market_prob is not None:
            variant = "market_no_vig"
        elif distribution_decision == "use_distribution_market_blend" and distribution_prob is not None:
            variant = "distribution_blend"
        elif distribution_decision == "use_distribution" and distribution_prob is not None:
            variant = "distribution"
        elif distribution_decision == "use_k_v3" and k_v3_prob is not None:
            variant = "k_v3"
    elif (
        event_side_line_prob is not None
        and market_evidence_confirms
        and _event_side_line_preferred(ctx)
        and variant in {"model_only", "distribution", "walk_forward_blend"}
    ):
        variant = "event_side_line"
    selected_prob = _policy_prob(
        variant,
        model_prob,
        market_prob,
        residual_prob,
        distribution_prob,
        event_side_line_prob,
        k_v3_prob,
        policy_rec,
    )
    selected_ev = ev_per_unit(selected_prob, price)

    stat = str(feature_row.get("market") or "")
    side = str(feature_row.get("side") or "")
    line = _prediction_line(feature_row)
    surface = str(feature_row.get("line_surface") or "unknown")
    tail_alt = _is_tail_alt(stat, side, line, surface)
    line_cap_exceeded = exceeds_non_lottery_line_cap(stat, line)
    residual_decision = str((residual_bucket or {}).get("decision") or "")
    no_bet_decision = residual_decision.startswith("no_bet")
    trust_status = str(trust.get("status") or "closed")
    trust_score = _clean_float(trust.get("trust_score")) or 0.0
    bucket_roi = _clean_float(trust.get("roi"))
    bucket_clv_beat_rate = _clean_float(trust.get("clv_price_beat_rate"))
    if bucket_clv_beat_rate is None:
        bucket_clv_beat_rate = _clean_float(trust.get("bucket_clv_beat_rate"))
    bucket_avg_clv = _clean_float(trust.get("avg_clv_price"))
    projected_pa = _clean_float(feature_row.get("projected_pa"))
    projected_bf = _clean_float(feature_row.get("projected_bf"))
    confirmed_order = _clean_float(feature_row.get("confirmed_batting_order"))
    hitter_market = stat in {"batter_hits", "batter_total_bases", "batter_home_runs"}
    pitcher_market = stat == "pitcher_strikeouts"
    opportunity_confirms = True
    if hitter_market:
        opportunity_confirms = (
            projected_pa is not None
            and projected_pa >= cfg.min_hitter_projected_pa
            and confirmed_order is not None
        )
    elif pitcher_market:
        opportunity_confirms = projected_bf is not None and projected_bf >= cfg.min_pitcher_projected_bf
    bucket_confirms = (
        trust_status in {"bankroll", "starter", "micro"}
        and bucket_roi is not None
        and bucket_roi >= 0.0
        and bucket_clv_beat_rate is not None
        and bucket_clv_beat_rate >= cfg.min_bucket_clv_beat_rate
        and bucket_avg_clv is not None
        and bucket_avg_clv >= 0.0
    )
    line_bucket_value = str(feature_row.get("line_bucket") or row.get("line_bucket") or "")
    if not line_bucket_value and bucket_key.count("|") >= 5:
        line_bucket_value = bucket_key.split("|")[3]
    line_gate_key = "|".join([
        stat,
        side,
        line_bucket_value or "unknown",
    ])
    line_production_gate = ctx.tb_hr_line_gates.get(line_gate_key) or {}
    tb_hr_line_confirms = (
        stat not in {"batter_total_bases", "batter_home_runs"}
        or bool(line_production_gate.get("passes"))
    )
    (
        external_agreement_micro_allowed,
        external_agreement_micro_reason,
        external_agreement_micro_blockers,
    ) = _external_agreement_micro_lane(
        feature_row,
        cfg=cfg,
        market_evidence_confirms=market_evidence_confirms,
        synthetic_fanduel_evidence=synthetic_fanduel_evidence,
        tail_alt=tail_alt,
        line_cap_exceeded=line_cap_exceeded,
    )
    external_agreement_count = _clean_int(feature_row.get("external_agreement_count")) or 0
    external_disagreement_count = _clean_int(feature_row.get("external_disagreement_count")) or 0
    external_agreement_strength = _clean_float(feature_row.get("external_agreement_strength")) or 0.0
    external_platforms = _external_platforms_text(feature_row.get("external_platforms"))

    model_wins = (
        selected_prob is not None
        and breakeven is not None
        and selected_ev is not None
        and selected_ev >= cfg.min_ev
        and selected_prob > breakeven
    )
    residual_ev = ev_per_unit(residual_prob, price)
    residual_wins = (
        residual_prob is not None
        and breakeven is not None
        and residual_ev is not None
        and residual_ev >= cfg.min_ev
        and residual_prob > breakeven
    )
    clv_wins = clv_prob is not None and clv_prob >= cfg.min_clv_beat_prob
    clv_magnitude_confirms = expected_clv_price is None or expected_clv_price > 0.0
    bookable = bookable_prob is not None and bookable_prob >= cfg.min_bookable_prob
    close_capture_confirms = close_capture_prob is not None and close_capture_prob >= cfg.min_close_capture_prob
    line_available_confirms = line_available_prob is None or line_available_prob >= cfg.min_line_available_prob
    real_candidate = (
        model_wins
        and residual_wins
        and clv_wins
        and clv_magnitude_confirms
        and bookable
        and close_capture_confirms
        and line_available_confirms
        and bucket_confirms
        and opportunity_confirms
        and market_evidence_confirms
        and tb_hr_line_confirms
        and not tail_alt
        and not line_cap_exceeded
        and not no_bet_decision
    )
    k_under_micro_allowed, k_under_gate_key, k_under_gate_blockers = _k_under_gate(ctx, feature_row)
    approved_micro_trial_model, micro_approved_model_key, micro_approved_model_reason = _approved_micro_trial_model(
        feature_row,
        stat=stat,
        side=side,
        line=line,
        surface=surface,
        pair_quality=pair_quality,
        k_under_micro_allowed=k_under_micro_allowed,
    )
    micro_projection_prob, micro_projection_source = _micro_projection_probability(
        stat=stat,
        model_family=feature_row.get("model_family"),
        model_prob=model_prob,
        distribution_prob=distribution_prob,
        event_side_line_prob=event_side_line_prob,
        k_v3_prob=k_v3_prob,
    )
    micro_projection_raw_prob = micro_projection_prob
    micro_probability_calibration_key = None
    micro_probability_calibration: dict[str, Any] = {}
    micro_projection_prob, micro_probability_calibration_key, micro_probability_calibration = (
        _apply_micro_probability_calibration(
            ctx,
            feature_row,
            micro_projection_prob,
            allow_broad_fallback=not approved_micro_trial_model,
        )
    )
    micro_tb15_high_pa_power_cap: dict[str, Any] = {}
    micro_projection_prob, micro_tb15_high_pa_power_cap = _apply_tb15_high_pa_power_micro_cap(
        feature_row,
        micro_projection_prob,
        micro_probability_calibration,
    )
    if micro_probability_calibration_key and micro_projection_source:
        micro_projection_source = f"{micro_projection_source}+micro_calibrated"
    if micro_tb15_high_pa_power_cap.get("applied") and micro_projection_source:
        micro_projection_source = f"{micro_projection_source}+tb15_high_pa_power_capped"
    micro_projection_ev = ev_per_unit(micro_projection_prob, price)
    micro_projection_edge = (
        micro_projection_prob - breakeven
        if micro_projection_prob is not None and breakeven is not None
        else None
    )
    micro_opportunity_confirms = True
    if hitter_market:
        micro_opportunity_confirms = projected_pa is not None and projected_pa >= max(3.2, cfg.min_hitter_projected_pa - 0.8)
    elif pitcher_market:
        micro_opportunity_confirms = projected_bf is not None and projected_bf >= max(16.0, cfg.min_pitcher_projected_bf - 4.0)
    micro_market_evidence_confirms = market_pair_confirms
    micro_bookable = bookable_prob is None or bookable_prob >= cfg.micro_projection_min_bookable_prob
    micro_close_capture = (
        close_capture_prob is None
        or close_capture_prob >= cfg.micro_projection_min_close_capture_prob
    )
    micro_line_available = (
        line_available_prob is None
        or line_available_prob >= cfg.micro_projection_min_line_available_prob
    )
    model_micro_clv_confirms = (
        clv_prob is not None
        and clv_prob >= cfg.micro_projection_min_clv_beat_prob
        and clv_magnitude_confirms
    )
    exact_bucket_micro_clv_confirms = (
        micro_trial_ready
        and exact_bucket_clv_micro_confirmed
        and exact_bucket_clv_prob is not None
        and exact_bucket_clv_prob >= cfg.micro_projection_min_bucket_clv_beat_rate
        and exact_bucket_avg_clv is not None
        and exact_bucket_avg_clv > 0.0
    )
    micro_clv_prob_candidates = [
        value for value in (clv_prob, exact_bucket_clv_prob)
        if value is not None
    ]
    micro_clv_prob = max(micro_clv_prob_candidates) if micro_clv_prob_candidates else None
    exact_bucket_micro_clv_not_bad = (
        exact_bucket_clv_prob is None
        or (
            exact_bucket_clv_prob >= cfg.micro_projection_min_bucket_clv_beat_rate
            and (exact_bucket_avg_clv is None or exact_bucket_avg_clv >= 0.0)
        )
    )
    micro_clv_confirms = model_micro_clv_confirms and exact_bucket_micro_clv_not_bad
    micro_bucket_history_confirms = (
        micro_trial_ready
        or (
            not no_bet_decision
            and (bucket_roi is None or bucket_roi >= cfg.micro_projection_min_bucket_roi)
            and (
                bucket_clv_beat_rate is None
                or bucket_clv_beat_rate >= cfg.micro_projection_min_bucket_clv_beat_rate
            )
            and (bucket_avg_clv is None or bucket_avg_clv >= 0.0)
        )
    )
    micro_no_bet_blocks = no_bet_decision and not micro_trial_ready
    tb15_dk_micro_focus_ok = True
    if stat == "batter_total_bases" and side == "over":
        tb15_dk_micro_focus_ok = (
            line is not None
            and abs(float(line) - 1.5) <= 1e-9
            and _book_key(feature_row) == "draftkings"
            and pair_quality == "same_book"
            and surface == "common"
            and distribution_prob is not None
        )
    (
        micro_truth_filter_allows,
        micro_truth_filter_status,
        micro_truth_filter_reason,
        micro_truth_filter_record,
    ) = _micro_truth_filter(
        ctx,
        feature_row,
        micro_approved_model_key
        if approved_micro_trial_model
        else f"external_agreement|{bucket_key}",
    )
    relaxed_micro_trial_lane = (
        approved_micro_trial_model
        or external_agreement_micro_allowed
    ) and micro_truth_filter_allows
    strict_micro_projection_proof = (
        micro_clv_confirms
        and micro_bucket_history_confirms
        and not micro_no_bet_blocks
    )
    micro_projection_required_prob_edge = 0.0 if relaxed_micro_trial_lane else cfg.micro_projection_min_prob_edge
    micro_projection_required_ev = 0.0 if relaxed_micro_trial_lane else cfg.micro_projection_min_ev
    micro_operational_confirms = (
        micro_bookable
        and micro_close_capture
        and micro_line_available
    ) or relaxed_micro_trial_lane
    micro_projection_candidate = (
        bool(cfg.enable_micro_projection)
        and not real_candidate
        and micro_projection_prob is not None
        and breakeven is not None
        and micro_projection_edge is not None
        and micro_projection_edge > micro_projection_required_prob_edge
        and micro_projection_ev is not None
        and micro_projection_ev > micro_projection_required_ev
        and micro_market_evidence_confirms
        and micro_operational_confirms
        and (micro_opportunity_confirms or relaxed_micro_trial_lane)
        and (strict_micro_projection_proof or relaxed_micro_trial_lane)
        and k_under_micro_allowed
        and tb15_dk_micro_focus_ok
        and not tail_alt
        and not line_cap_exceeded
        and not synthetic_fanduel_evidence
    )

    reasons: list[str] = []
    if not model_wins:
        reasons.append("model_or_price_no_edge")
    if not residual_wins:
        reasons.append("market_residual_not_confirming")
    if not clv_wins:
        reasons.append("clv_model_not_confirming")
    if not clv_magnitude_confirms:
        reasons.append("expected_clv_not_positive")
    if (
        residual_clv_prob is not None
        and event_side_line_clv_prob is not None
        and min(residual_clv_prob, event_side_line_clv_prob) < cfg.min_clv_beat_prob
    ):
        reasons.append("clv_models_disagree_or_weak")
    if not bookable:
        reasons.append("bookability_not_confirming")
    if not close_capture_confirms:
        reasons.append("close_capture_not_confirming")
    if not line_available_confirms:
        reasons.append("line_availability_not_confirming")
    if trust_status not in {"bankroll", "starter", "micro"}:
        reasons.append(f"bucket_{trust_status or 'closed'}")
    elif not bucket_confirms:
        reasons.append("bucket_history_not_confirming")
    if not opportunity_confirms:
        reasons.append("opportunity_not_confirming")
    if not market_evidence_confirms:
        reasons.append(f"pair_quality_{pair_quality or 'unknown'}")
    elif pair_quality == "cross_book":
        reasons.append("cross_book_market_pair")
    elif ladder_market_confirms:
        reasons.append("one_sided_ladder_market_evidence")
    if synthetic_fanduel_evidence:
        reasons.append("fanduel_synthetic_market_evidence")
    if not tb_hr_line_confirms:
        reasons.append("tb_hr_line_production_gate_failed")
        reasons.extend(str(value) for value in (line_production_gate.get("reasons") or []))
    if bucket_roi is not None and bucket_roi < 0:
        reasons.append("bucket_roi_negative")
    if bucket_clv_beat_rate is not None and bucket_clv_beat_rate < cfg.min_bucket_clv_beat_rate:
        reasons.append("bucket_clv_beat_low")
    if bucket_avg_clv is not None and bucket_avg_clv < 0:
        reasons.append("bucket_avg_clv_negative")
    if model_prob is not None and market_prob is not None and residual_prob is None and abs(model_prob - market_prob) >= 0.12:
        reasons.append("unconfirmed_market_disagreement")
    if tail_alt:
        reasons.append("alt_line_lottery")
    if line_cap_exceeded:
        reasons.append("non_lottery_line_cap")
    if no_bet_decision and not micro_trial_ready:
        reasons.append(residual_decision)
        for blocker in (residual_bucket or {}).get("residual_proof_blockers") or []:
            reasons.append(str(blocker))
    elif no_bet_decision and micro_trial_ready:
        reasons.append("micro_trial_exact_bucket_overrode_no_bet")
    distribution_decision = str((distribution_bucket or {}).get("decision") or "")
    if distribution_decision:
        reasons.append(f"distribution_bucket_{distribution_decision}")
    if micro_projection_candidate:
        reasons.append("micro_projection_candidate")
        reasons.append("micro_projection_not_bankroll_proven")
        if relaxed_micro_trial_lane:
            if approved_micro_trial_model:
                reasons.append("micro_trial_approved_model")
                reasons.append(micro_approved_model_reason)
            if external_agreement_micro_allowed:
                reasons.append("external_model_agreement_micro_lane")
                reasons.append("micro_external_agreement_not_bankroll_proven")
                if external_platforms:
                    reasons.append(f"external_agreement_platforms_{external_platforms.replace(', ', '_')}")
            if not micro_clv_confirms:
                reasons.append("micro_trial_collecting_clv_not_required")
            if not micro_bucket_history_confirms:
                reasons.append("micro_trial_collecting_roi_not_required")
            if not micro_bookable:
                reasons.append("micro_trial_bookability_model_not_required")
            if not micro_close_capture:
                reasons.append("micro_trial_close_capture_model_not_required")
            if not micro_line_available:
                reasons.append("micro_trial_line_availability_model_not_required")
            if micro_no_bet_blocks:
                reasons.append("micro_trial_collecting_overrode_no_bet")
            if micro_truth_filter_reason:
                reasons.append(micro_truth_filter_reason)
        if micro_trial_ready:
            reasons.append("micro_trial_exact_bucket")
        if exact_bucket_micro_clv_confirms and not model_micro_clv_confirms:
            reasons.append("micro_trial_exact_bucket_clv_prior")
    elif cfg.enable_micro_projection and micro_projection_prob is not None and not real_candidate:
        if micro_projection_edge is None or micro_projection_edge <= micro_projection_required_prob_edge:
            reasons.append("micro_projection_edge_too_small")
        if micro_projection_ev is None or micro_projection_ev <= micro_projection_required_ev:
            reasons.append("micro_projection_ev_too_small")
        if not micro_market_evidence_confirms:
            reasons.append("micro_projection_needs_true_pair")
        if not micro_opportunity_confirms:
            reasons.append("micro_projection_opportunity_weak")
        if not approved_micro_trial_model and not micro_bookable:
            reasons.append("micro_projection_bookability_weak")
        if approved_micro_trial_model and not micro_bookable:
            reasons.append("micro_trial_bookability_model_not_required")
        if not approved_micro_trial_model and not micro_close_capture:
            reasons.append("micro_projection_close_capture_weak")
        if approved_micro_trial_model and not micro_close_capture:
            reasons.append("micro_trial_close_capture_model_not_required")
        if not approved_micro_trial_model and not micro_line_available:
            reasons.append("micro_projection_line_availability_weak")
        if approved_micro_trial_model and not micro_line_available:
            reasons.append("micro_trial_line_availability_model_not_required")
        if not approved_micro_trial_model and not micro_clv_confirms:
            reasons.append("micro_projection_clv_weak")
        if not approved_micro_trial_model and not exact_bucket_micro_clv_not_bad:
            reasons.append("micro_projection_exact_bucket_clv_bad")
        if not tb15_dk_micro_focus_ok:
            reasons.append("micro_projection_tb15_dk_true_pair_only")
        if not approved_micro_trial_model:
            reasons.append(micro_approved_model_reason)
        if external_agreement_count and not external_agreement_micro_allowed:
            reasons.extend(external_agreement_micro_blockers)
        if (approved_micro_trial_model or external_agreement_micro_allowed) and not micro_truth_filter_allows:
            reasons.append(micro_truth_filter_reason or "micro_truth_filter_blocked")
        if micro_tb15_high_pa_power_cap.get("applied"):
            reasons.append("micro_projection_tb15_high_pa_power_cap")
        if not approved_micro_trial_model and micro_no_bet_blocks:
            reasons.append("micro_projection_bucket_no_bet")
        if bucket_roi is not None and bucket_roi < cfg.micro_projection_min_bucket_roi:
            reasons.append("micro_projection_bucket_roi_weak")
        if (
            bucket_clv_beat_rate is not None
            and bucket_clv_beat_rate < cfg.micro_projection_min_bucket_clv_beat_rate
        ):
            reasons.append("micro_projection_bucket_clv_weak")
        if bucket_avg_clv is not None and bucket_avg_clv < 0:
            reasons.append("micro_projection_bucket_avg_clv_weak")
        if not k_under_micro_allowed:
            reasons.append("micro_projection_k_under_repair_gate_failed")
            reasons.extend(k_under_gate_blockers)
    if _clean_bool(row.get("bankroll_candidate")) and not real_candidate:
        reasons.append("bankroll_downgrade_recommended")

    score_ev = selected_ev if selected_ev is not None else (_clean_float(row.get("ev")) or -0.25)
    score = max(-0.35, min(0.45, score_ev))
    score_clv_prob = micro_clv_prob if micro_trial_ready and micro_clv_prob is not None else clv_prob
    score_clv_wins = clv_wins or exact_bucket_micro_clv_confirms or relaxed_micro_trial_lane
    if score_clv_prob is not None:
        score += 0.25 * (score_clv_prob - 0.50)
    else:
        score -= 0.05
    if model_wins and not score_clv_wins:
        score -= 0.35
    if expected_clv_price is not None:
        score += max(-0.15, min(0.15, expected_clv_price * 0.02))
    if model_wins and not residual_wins:
        score -= 0.30
    if bookable_prob is not None:
        score += 0.10 * (bookable_prob - 0.50)
    else:
        score -= 0.03
    if close_capture_prob is not None and close_capture_prob < cfg.min_close_capture_prob:
        score -= 0.12
    if line_available_prob is not None and line_available_prob < cfg.min_line_available_prob:
        score -= 0.12
    score += 0.05 * max(0.0, min(1.0, trust_score / 100.0))
    if bucket_roi is not None and bucket_roi < 0:
        score -= min(0.20, abs(bucket_roi) * 0.75)
    if bucket_clv_beat_rate is not None and bucket_clv_beat_rate < 0.50:
        score -= min(0.20, (0.50 - bucket_clv_beat_rate) * 0.75)
    if bucket_avg_clv is not None and bucket_avg_clv < 0:
        score -= min(0.10, abs(bucket_avg_clv) * 0.05)
    if not opportunity_confirms:
        score -= 0.12
    if pair_quality == "same_book":
        score += 0.03
    elif pair_quality == "cross_book":
        score -= 0.03
    elif pair_quality == "ladder":
        score -= 0.08
    elif pair_quality in {"synthetic", "one_sided", "unknown"}:
        score -= 0.30
    if synthetic_fanduel_evidence:
        score -= 0.50
    if external_agreement_micro_allowed:
        score += 0.08 + min(0.05, max(0.0, external_agreement_strength - 0.75) * 0.20)
    elif external_agreement_count:
        score += 0.02
    if external_disagreement_count:
        score -= min(0.10, 0.03 * external_disagreement_count)
    if no_bet_decision:
        score -= 0.05 if micro_trial_ready else 0.25
    if tail_alt and trust_status not in {"bankroll", "starter", "micro"}:
        score -= 0.50
    if not _clean_bool(row.get("price_drift_ok", True)):
        score -= 0.50

    if score < cfg.micro_projection_min_selector_score and not relaxed_micro_trial_lane:
        if micro_projection_candidate:
            reasons = [
                reason for reason in reasons
                if reason not in {"micro_projection_candidate", "micro_projection_not_bankroll_proven"}
            ]
        micro_projection_candidate = False
        if (
            cfg.enable_micro_projection
            and micro_projection_prob is not None
            and not real_candidate
        ):
            reasons.append("micro_projection_selector_score_weak")
    elif score < cfg.micro_projection_min_selector_score and relaxed_micro_trial_lane:
        reasons.append("micro_trial_selector_score_observed")

    diagnostic_paper = (
        model_wins
        and not tail_alt
        and not synthetic_fanduel_evidence
        and not no_bet_decision
    )

    tier = "watch"
    if line_cap_exceeded and side == "over":
        tier = "lottery"
    elif line_cap_exceeded:
        tier = "no_bet"
    elif real_candidate:
        tier = trust_status if trust_status in {"micro", "starter", "bankroll"} else "watch"
    elif micro_projection_candidate:
        tier = "micro_projection"
    elif tail_alt:
        tier = "lottery"
    elif diagnostic_paper:
        tier = "paper"
    elif model_wins and opportunity_confirms and pair_quality in {"synthetic", "one_sided", "unknown", "ladder"} and not synthetic_fanduel_evidence:
        tier = "lottery"
    elif no_bet_decision:
        tier = "no_bet"
    elif model_wins:
        tier = "no_bet"

    return {
        "prediction_key": row.get("prediction_key"),
        "prop_offer_id": row.get("prop_offer_id"),
        "player_name": row.get("player_name"),
        "team_abbr": row.get("team_abbr"),
        "market": stat,
        "side": side,
        "line": line,
        "price": price,
        "bookmaker_key": _book_key(feature_row),
        "bucket_key": bucket_key,
        "line_surface": surface,
        "line_bucket": feature_row.get("line_bucket"),
        "price_bucket": feature_row.get("price_bucket"),
        "policy_level": policy_level,
        "policy_variant": variant,
        "residual_bucket_decision": residual_decision or None,
        "residual_proof_blockers": (residual_bucket or {}).get("residual_proof_blockers") or [],
        "distribution_bucket_decision": distribution_decision or None,
        "model_prob_side": model_prob,
        "market_prob_side": market_prob,
        "market_prob_source": feature_row.get("market_prob_source"),
        "pair_quality": pair_quality,
        "residual_prob_side": residual_prob,
        "residual_ev": residual_ev,
        "residual_confirms": residual_wins,
        "distribution_prob_side": distribution_prob,
        "distribution_calibration_key": distribution_calibration_key,
        "k_v3_prob_side": k_v3_prob,
        "event_side_line_prob_side": event_side_line_prob,
        "tb15_calibration_key": tb15_calibration_key,
        "clv_beat_prob": clv_prob,
        "micro_clv_beat_prob": micro_clv_prob,
        "model_micro_clv_confirms": model_micro_clv_confirms,
        "exact_bucket_clv_beat_prob": exact_bucket_clv_prob,
        "exact_bucket_avg_clv": exact_bucket_avg_clv,
        "exact_bucket_clv_micro_confirmed": exact_bucket_clv_micro_confirmed,
        "expected_clv_price": expected_clv_price,
        "clv_magnitude_confirms": clv_magnitude_confirms,
        "residual_clv_beat_prob": residual_clv_prob,
        "event_side_line_clv_beat_prob": event_side_line_clv_prob,
        "bookable_prob": bookable_prob,
        "line_available_prob": line_available_prob,
        "close_capture_prob": close_capture_prob,
        "bookability_source": bookability_source,
        "selector_prob_side": selected_prob,
        "tb_hr_line_production_gate": bool(tb_hr_line_confirms),
        "selector_ev": selected_ev,
        "micro_projection_candidate": micro_projection_candidate,
        "micro_projection_prob_side": micro_projection_prob,
        "micro_projection_raw_prob_side": micro_projection_raw_prob,
        "micro_projection_prob_source": micro_projection_source,
        "micro_probability_calibration_key": micro_probability_calibration_key,
        "micro_probability_calibration_status": (
            "applied" if micro_probability_calibration_key else "not_applied"
        ),
        "micro_probability_calibration_target": micro_probability_calibration.get("target_probability"),
        "micro_probability_calibration_cap": micro_probability_calibration.get("max_probability"),
        "micro_probability_calibration_shrink": micro_probability_calibration.get("shrink_weight"),
        "micro_tb15_high_pa_power_cap_applied": bool(micro_tb15_high_pa_power_cap.get("applied")),
        "micro_tb15_high_pa_power_cap": micro_tb15_high_pa_power_cap.get("cap"),
        "micro_tb15_high_pa_power_cap_reason": micro_tb15_high_pa_power_cap.get("reason"),
        "micro_tb15_high_pa_power_flags": micro_tb15_high_pa_power_cap.get("power_flags"),
        "micro_projection_edge": micro_projection_edge,
        "micro_projection_ev": micro_projection_ev,
        "micro_projection_required_prob_edge": micro_projection_required_prob_edge,
        "micro_projection_required_ev": micro_projection_required_ev,
        "micro_projection_clv_confirms": micro_clv_confirms,
        "micro_projection_bucket_history_confirms": micro_bucket_history_confirms,
        "micro_approved_model": approved_micro_trial_model,
        "micro_approved_model_key": micro_approved_model_key,
        "micro_approved_model_reason": micro_approved_model_reason,
        "micro_external_agreement": external_agreement_micro_allowed,
        "micro_external_agreement_reason": external_agreement_micro_reason,
        "micro_external_agreement_blockers": external_agreement_micro_blockers,
        "external_agreement_count": external_agreement_count,
        "external_disagreement_count": external_disagreement_count,
        "external_agreement_strength": external_agreement_strength,
        "external_match_level": feature_row.get("external_match_level"),
        "external_platforms": external_platforms,
        "external_best_grade": feature_row.get("external_best_grade"),
        "external_max_ev": _clean_float(feature_row.get("external_max_ev")),
        "external_max_probability": _clean_float(feature_row.get("external_max_probability")),
        "micro_relaxed_trial_lane": relaxed_micro_trial_lane,
        "micro_truth_filter_status": micro_truth_filter_status,
        "micro_truth_filter_reason": micro_truth_filter_reason,
        "micro_truth_filter_key": micro_truth_filter_record.get("micro_truth_key"),
        "micro_truth_filter_graded": micro_truth_filter_record.get("graded"),
        "micro_truth_filter_record": micro_truth_filter_record.get("record"),
        "micro_truth_filter_roi": micro_truth_filter_record.get("roi"),
        "micro_truth_filter_clv_rows": micro_truth_filter_record.get("clv_rows"),
        "micro_truth_filter_clv_beat_rate": micro_truth_filter_record.get("clv_beat_rate"),
        "micro_truth_filter_avg_clv": micro_truth_filter_record.get("avg_clv_price"),
        "micro_trial_ready": micro_trial_ready,
        "micro_trial_blockers": micro_eval_bucket.get("micro_trial_blockers") or [],
        "k_under_repair_gate_key": k_under_gate_key,
        "k_under_repair_micro_allowed": k_under_micro_allowed,
        "k_under_repair_blockers": k_under_gate_blockers,
        "selector_score": score,
        "selector_tier": tier,
        "selector_real_candidate": real_candidate,
        "bucket_trust_status": trust_status,
        "bucket_trust_score": trust_score,
        "bucket_roi": trust.get("roi"),
        "bucket_clv_beat_rate": trust.get("clv_price_beat_rate"),
        "bucket_avg_clv": trust.get("avg_clv_price"),
        "opportunity_confirms": opportunity_confirms,
        "tail_alt": tail_alt,
        "line_cap_exceeded": line_cap_exceeded,
        "no_bet_decision": no_bet_decision,
        "selector_reasons": sorted(dict.fromkeys(reasons)),
    }


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


def _table_has_columns(conn, schema: str, table: str, columns: set[str]) -> bool:
    if not columns:
        return True
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            """,
            (schema, table),
        )
        existing = {str(row[0]) for row in cur.fetchall()}
    return columns.issubset(existing)


_ACTIVE_SQL_BASE = """
SELECT
    pp.id,
    pp.game_date_et,
    pp.game_slug,
    pp.player_id,
    pp.player_name,
    pp.team_abbr,
    pp.stat,
    pp.bet_side,
    pp.pred_value::float AS pred_value,
    pp.pred_count::float AS pred_count,
    pp.pred_prob_over::float AS pred_prob_over,
    pp.book_line::float AS book_line,
    pp.edge::float AS edge,
    pp.edge_type,
    pp.model_family,
    pp.over_price::float AS over_price,
    pp.under_price::float AS under_price,
    pp.bet_price::float AS bet_price,
    pp.ev::float AS ev,
    pp.bookmaker_key,
    pp.prediction_key,
    pp.prop_offer_id,
    pp.bankroll_candidate,
    pp.bankroll_tier,
    pp.bankroll_reasons,
    TRUE AS price_drift_ok
FROM bets.mlb_prop_predictions pp
WHERE pp.game_date_et = %(report_date)s
  AND COALESCE(pp.is_active, TRUE) IS TRUE
  AND pp.bet_side IN ('over','under')
"""

_ACTIVE_SQL_JOIN = """
WITH active_predictions AS (
    SELECT
        pp.id,
        pp.game_date_et,
        pp.game_slug,
        pp.player_id,
        pp.player_name,
        pp.team_abbr,
        pp.stat,
        pp.bet_side,
        pp.pred_value::float AS pred_value,
        pp.pred_count::float AS pred_count,
        pp.pred_prob_over::float AS pred_prob_over,
        pp.book_line::float AS book_line,
        pp.edge::float AS edge,
        pp.edge_type,
        pp.model_family,
        pp.over_price::float AS over_price,
        pp.under_price::float AS under_price,
        pp.bet_price::float AS bet_price,
        pp.ev::float AS ev,
        pp.bookmaker_key,
        pp.prediction_key,
        pp.prop_offer_id,
        pp.bankroll_candidate,
        pp.bankroll_tier,
        pp.bankroll_reasons,
        pp.opportunity_context,
        TRUE AS price_drift_ok
    FROM bets.mlb_prop_predictions pp
    WHERE pp.game_date_et = %(report_date)s
      AND COALESCE(pp.is_active, TRUE) IS TRUE
      AND pp.bet_side IN ('over','under')
),
active_prediction_keys AS (
    SELECT DISTINCT prediction_key
    FROM active_predictions
    WHERE prediction_key IS NOT NULL
),
latest_training AS (
    SELECT DISTINCT ON (e.prediction_key)
        e.prediction_key,
        e.market_prob_side::float AS training_market_prob_side,
        e.market_prob_source,
        e.paired_price_source,
        e.pair_quality,
        e.same_book_pair_flag::float AS same_book_pair_flag,
        e.cross_book_pair_flag::float AS cross_book_pair_flag,
        e.synthetic_pair_flag::float AS synthetic_pair_flag,
        e.clean_market_pair_flag::float AS clean_market_pair_flag,
        e.true_pair_flag::float AS true_pair_flag,
        e.minutes_to_first_pitch_at_lock::float AS minutes_to_first_pitch_at_lock,
        e.lock_price_age_minutes::float AS lock_price_age_minutes,
        e.line_surface,
        e.line_bucket,
        e.price_bucket,
        e.count_edge_side::float AS count_edge_side,
        e.prob_edge_vs_market::float AS prob_edge_vs_market,
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
        e.pinch_hit_risk::float AS pinch_hit_risk
    FROM features.mlb_prop_market_training_examples e
    JOIN active_prediction_keys k
      ON k.prediction_key = e.prediction_key
    WHERE e.game_date_et = %(report_date)s
    ORDER BY e.prediction_key, e.example_updated_at DESC, e.id DESC
)
SELECT
    pp.*,
    e.training_market_prob_side,
    e.market_prob_source,
    e.paired_price_source,
    e.pair_quality,
    e.same_book_pair_flag,
    e.cross_book_pair_flag,
    e.synthetic_pair_flag,
    e.clean_market_pair_flag,
    e.true_pair_flag,
    e.minutes_to_first_pitch_at_lock,
    e.lock_price_age_minutes,
    e.line_surface,
    e.line_bucket,
    e.price_bucket,
    e.count_edge_side,
    e.prob_edge_vs_market,
    e.confirmed_batting_order,
    e.projected_pa,
    e.projected_bf,
    e.projected_pitch_count,
    e.is_home,
    e.team_implied_runs,
    e.opponent_implied_runs,
    e.game_total_line,
    e.opp_sp_hand,
    e.opp_sp_k_pct_10,
    e.opp_sp_bb_pct,
    e.opp_sp_xwoba,
    e.opp_sp_hard_hit_pct,
    e.opp_sp_whiff_pct,
    e.opp_bp_era_10,
    e.opp_bp_whip_10,
    e.opp_bp_k9_10,
    e.opp_team_k_pct_10,
    e.batter_vs_hand_hits_avg_10,
    e.batter_vs_hand_tb_avg_10,
    e.batter_vs_hand_hr_avg_10,
    e.batter_vs_hand_iso_avg_10,
    e.batter_vs_hand_k_rate_10,
    e.batter_vs_rp_slg_30,
    e.batter_vs_rp_hr_rate_30,
    e.pinch_hit_risk
FROM active_predictions pp
LEFT JOIN latest_training e
  ON e.prediction_key = pp.prediction_key
"""


def load_active_rows(conn, report_date: date) -> list[dict[str, Any]]:
    if not _table_exists(conn, "bets", "mlb_prop_predictions"):
        return []
    has_training_table = _table_exists(conn, "features", "mlb_prop_market_training_examples")
    has_training_columns = has_training_table and _table_has_columns(
        conn,
        "features",
        "mlb_prop_market_training_examples",
        {
            "market_prob_source", "paired_price_source", "pair_quality",
            "same_book_pair_flag", "cross_book_pair_flag", "synthetic_pair_flag",
            "clean_market_pair_flag", "true_pair_flag",
            "minutes_to_first_pitch_at_lock", "lock_price_age_minutes",
        },
    )
    sql = (
        _ACTIVE_SQL_JOIN
        if has_training_columns
        else _ACTIVE_SQL_BASE
    )
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"report_date": report_date})
        rows = [dict(row) for row in cur.fetchall()]
    try:
        attach_external_agreement(conn, rows, game_date=report_date)
    except Exception:
        # External comparisons are useful evidence, but selector operation must
        # remain available when no export/API feed has been loaded yet.
        pass
    return rows


def _fmt_pct(value: Any, *, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    v = _clean_float(value)
    if v is None:
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _write_text_with_lock_fallback(path: Path, text: str) -> Path:
    try:
        atomic_write_text(path, text)
        return path
    except PermissionError:
        stamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
        fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        atomic_write_text(fallback, text)
        return fallback


def build_payload(cfg: ShadowSelectorConfig) -> dict[str, Any]:
    report_date = cfg.report_date or datetime.now(_ET).date()
    ctx = SelectorContext(cfg.model_dir)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        rows = load_active_rows(conn, report_date)
    scored = [score_prediction_row(row, ctx=ctx, cfg=cfg) for row in rows]
    scored.sort(key=lambda row: (_clean_float(row.get("selector_score")) or -999.0), reverse=True)
    common_paper = [
        row for row in scored
        if row.get("selector_tier") == "paper" and not row.get("tail_alt")
    ]
    micro_projection = [
        row for row in scored
        if row.get("selector_tier") == "micro_projection"
    ]
    micro_projection.sort(
        key=lambda row: (
            _clean_float(row.get("micro_projection_ev")) or -999.0,
            _clean_float(row.get("micro_projection_edge")) or -999.0,
            _clean_float(row.get("micro_projection_prob_side")) or -999.0,
        ),
        reverse=True,
    )
    no_bet = [
        row for row in scored
        if row.get("selector_tier") == "no_bet"
    ]
    lottery = [
        row for row in scored
        if row.get("selector_tier") == "lottery" or row.get("tail_alt")
    ]
    closest = list(ctx.promotion.get("closest_common_buckets") or [])
    if len(closest) < cfg.top_n:
        closest.extend(ctx.promotion.get("closest_alt_line_buckets") or [])
    return {
        "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
        "report_date": str(report_date),
        "active_rows": len(rows),
        "scored_rows": len(scored),
        "real_candidate_rows": sum(1 for row in scored if row.get("selector_real_candidate")),
        "micro_projection_rows": sum(1 for row in scored if row.get("selector_tier") == "micro_projection"),
        "paper_rows": sum(1 for row in scored if row.get("selector_tier") == "paper"),
        "lottery_rows": sum(1 for row in scored if row.get("selector_tier") == "lottery"),
        "no_bet_rows": sum(1 for row in scored if row.get("selector_tier") == "no_bet"),
        "external_agreement_rows": sum(1 for row in scored if row.get("external_agreement_count")),
        "external_micro_rows": sum(1 for row in scored if row.get("micro_external_agreement")),
        "top_rows": scored[: cfg.top_n],
        "micro_projection_top_rows": micro_projection[: cfg.top_n],
        "best_common_paper_rows": common_paper[: cfg.top_n],
        "no_bet_top_rows": no_bet[: cfg.top_n],
        "lottery_top_rows": lottery[: cfg.top_n],
        "closest_to_promotion_buckets": closest[: cfg.top_n],
    }


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _label, _key in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for _label, key in columns) + " |")
    return "\n".join(lines)


def write_report(payload: dict[str, Any], cfg: ShadowSelectorConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    def _display_rows(section_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for row in section_rows:
            rows.append({
                "player": row.get("player_name"),
                "stat": row.get("market"),
                "side": row.get("side"),
                "line": _fmt_num(row.get("line"), 1),
                "price": _fmt_num(row.get("price"), 0, signed=True),
                "book": row.get("bookmaker_key"),
                "tier": row.get("selector_tier"),
                "variant": row.get("policy_variant"),
                "p": _fmt_pct(row.get("selector_prob_side")),
                "ev": _fmt_pct(row.get("selector_ev"), signed=True),
                "micro_p": _fmt_pct(row.get("micro_projection_prob_side")),
                "micro_ev": _fmt_pct(row.get("micro_projection_ev"), signed=True),
                "micro_edge": _fmt_pct(row.get("micro_projection_edge"), signed=True),
                "micro_src": row.get("micro_projection_prob_source"),
                "external": row.get("external_platforms"),
                "external_match": row.get("external_match_level"),
                "clv": _fmt_pct(row.get("clv_beat_prob")),
                "bookable": _fmt_pct(row.get("bookable_prob")),
                "line_avail": _fmt_pct(row.get("line_available_prob")),
                "close_cap": _fmt_pct(row.get("close_capture_prob")),
                "book_src": row.get("bookability_source"),
                "pair": row.get("pair_quality"),
                "trust": row.get("bucket_trust_status"),
                "score": _fmt_num(row.get("selector_score"), 3, signed=True),
                "reasons": "; ".join(row.get("selector_reasons") or []),
            })
        return rows

    row_columns = [
        ("Player", "player"),
        ("Stat", "stat"),
        ("Side", "side"),
        ("Line", "line"),
        ("Price", "price"),
        ("Book", "book"),
        ("Tier", "tier"),
        ("Variant", "variant"),
        ("P", "p"),
        ("EV", "ev"),
        ("CLV", "clv"),
        ("Bookable", "bookable"),
        ("Line Avail", "line_avail"),
        ("Close Cap", "close_cap"),
        ("Book Src", "book_src"),
        ("Pair", "pair"),
        ("Trust", "trust"),
        ("Score", "score"),
        ("Reasons", "reasons"),
    ]
    micro_columns = [
        ("Player", "player"),
        ("Stat", "stat"),
        ("Side", "side"),
        ("Line", "line"),
        ("Price", "price"),
        ("Book", "book"),
        ("Projection P", "micro_p"),
        ("Proj Edge", "micro_edge"),
        ("Proj EV", "micro_ev"),
        ("Source", "micro_src"),
        ("External", "external"),
        ("Ext Match", "external_match"),
        ("CLV", "clv"),
        ("Pair", "pair"),
        ("Reasons", "reasons"),
    ]

    bucket_rows = []
    for row in payload.get("closest_to_promotion_buckets") or []:
        bucket_rows.append({
            "bucket": row.get("key"),
            "graded": row.get("graded"),
            "roi": _fmt_pct(row.get("roi"), signed=True),
            "clv": _fmt_pct(row.get("clv_beat_rate")),
            "avg_clv": _fmt_num(row.get("avg_clv_price"), 2, signed=True),
            "cal": _fmt_pct(row.get("calibration_error"), signed=True),
            "dates": row.get("unique_dates"),
            "gaps": "; ".join(row.get("metric_gaps") or row.get("reasons") or []),
        })

    text = "\n".join([
        "# MLB Prop Shadow Selector",
        "",
        f"Date: {payload.get('report_date')}",
        f"Active rows: {payload.get('active_rows')}",
        f"Scored rows: {payload.get('scored_rows')}",
        f"Real candidates: {payload.get('real_candidate_rows')}",
        f"$1 MICRO TEST rows: {payload.get('micro_projection_rows')}",
        f"External-agreement rows: {payload.get('external_agreement_rows')}",
        f"External-agreement micro rows: {payload.get('external_micro_rows')}",
        f"Paper rows: {payload.get('paper_rows')}",
        f"Lottery rows: {payload.get('lottery_rows')}",
        f"No-bet rows: {payload.get('no_bet_rows')}",
        "",
        "## Best Common-Line Paper Props",
        "",
        _table(_display_rows(payload.get("best_common_paper_rows") or []), row_columns),
        "",
        "## $1 MICRO TEST Plays",
        "",
        "Projection-led $1 testing candidates. These are not bankroll-proven buckets.",
        "",
        _table(_display_rows(payload.get("micro_projection_top_rows") or []), micro_columns),
        "",
        "## No-Bet Rows",
        "",
        _table(_display_rows(payload.get("no_bet_top_rows") or []), row_columns),
        "",
        "## Alt-Line Lottery Rows",
        "",
        _table(_display_rows(payload.get("lottery_top_rows") or []), row_columns),
        "",
        "## Closest-To-Promotion Buckets",
        "",
        _table(bucket_rows, [
            ("Bucket", "bucket"),
            ("Graded", "graded"),
            ("ROI", "roi"),
            ("CLV Beat", "clv"),
            ("Avg CLV", "avg_clv"),
            ("Cal Err", "cal"),
            ("Dates", "dates"),
            ("Gaps", "gaps"),
        ]),
        "",
        "## Overall Top Rows",
        "",
        _table(_display_rows(payload.get("top_rows") or []), row_columns),
        "",
    ])
    return str(_write_text_with_lock_fallback(path, text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual/CLV-aware MLB prop shadow selector report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--date", default=None)
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--min-clv-beat-prob", type=float, default=0.55)
    parser.add_argument("--min-bookable-prob", type=float, default=0.60)
    parser.add_argument("--min-bucket-clv-beat-rate", type=float, default=0.55)
    parser.add_argument("--min-hitter-projected-pa", type=float, default=3.2)
    parser.add_argument("--min-pitcher-projected-bf", type=float, default=16.0)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--json-out", default="prop_shadow_selector_report.json")
    parser.add_argument("--report-file", default="mlb_prop_shadow_selector_latest.md")
    args = parser.parse_args()
    cfg = ShadowSelectorConfig(
        pg_dsn=args.pg_dsn,
        report_date=date.fromisoformat(args.date) if args.date else None,
        model_dir=Path(args.model_dir),
        min_ev=args.min_ev,
        min_clv_beat_prob=args.min_clv_beat_prob,
        min_bookable_prob=args.min_bookable_prob,
        min_bucket_clv_beat_rate=args.min_bucket_clv_beat_rate,
        min_hitter_projected_pa=args.min_hitter_projected_pa,
        min_pitcher_projected_bf=args.min_pitcher_projected_bf,
        top_n=args.top_n,
        out_file=args.json_out,
        report_file=args.report_file,
    )
    payload = build_payload(cfg)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.model_dir / cfg.out_file, payload)
    report_path = write_report(payload, cfg)
    print(json.dumps({
        "status": "ok",
        "active_rows": payload.get("active_rows"),
        "real_candidate_rows": payload.get("real_candidate_rows"),
        "report_path": report_path,
    }, indent=2))


if __name__ == "__main__":
    main()
