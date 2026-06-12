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

from .side_recalibration import price_bucket, prop_line_bucket, prop_line_surface

_ET = ZoneInfo("America/New_York")
_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
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
    min_bucket_clv_beat_rate: float = 0.55
    min_hitter_projected_pa: float = 3.2
    min_pitcher_projected_bf: float = 16.0
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
    if current in {"same_book", "cross_book", "synthetic", "one_sided"}:
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
    if market_source in {"raw_implied", "one_sided"}:
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
        self.bookability = _load_json(self.model_dir / "prop_bookability_model.json")
        self.trust = _load_json(self.model_dir / "prop_bucket_trust_scores.json")
        self.promotion = _load_json(self.model_dir / "prop_bucket_promotion_report.json")

        self.live_policy = self.walk_forward.get("live_policy") or {}
        self.residual_buckets = {
            str(rec.get("bucket")): dict(rec)
            for rec in (self.residual.get("bucket_recommendations") or [])
            if rec.get("bucket")
        }
        self.trust_scores = {
            str(key): dict(value or {})
            for key, value in (self.trust.get("bucket_scores") or {}).items()
        }
        holdout = self.bookability.get("holdout") or {}
        self.bookability_model_usable = bool(
            holdout.get("model_usable", True)
            and self.bookability.get("selected_scoring_method", "logistic") == "logistic"
        )
        self.bookability_rates = self.bookability.get("empirical_bookability_rates") or {}
        self.default_bookability_rate = _clean_float(holdout.get("actual_bookable_rate"))


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
    for name in model.get("numeric_features") or []:
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
    for name in model.get("categorical_features") or []:
        value = str(row.get(name) if row.get(name) is not None else "unknown")
        z += float(coef.get(f"{name}={value}") or 0.0)
    return max(1e-6, min(1.0 - 1e-6, _sigmoid(z)))


def _bookability_empirical_score(ctx: SelectorContext, feature_row: dict[str, Any], bucket_key: str) -> tuple[float | None, str | None]:
    parts = bucket_key.split("|")
    if len(parts) < 6:
        return ctx.default_bookability_rate, "holdout_default"
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
        rate = _clean_float(rec.get("bookable_rate"))
        if rate is not None and int(rec.get("rows") or 0) >= 10:
            return max(1e-6, min(1.0 - 1e-6, rate)), level
    return ctx.default_bookability_rate, "holdout_default"


def _bookability_score(ctx: SelectorContext, feature_row: dict[str, Any], bucket_key: str) -> tuple[float | None, str | None]:
    empirical, empirical_source = _bookability_empirical_score(ctx, feature_row, bucket_key)
    if not ctx.bookability_model_usable:
        return empirical, empirical_source
    logistic = _logistic_score((ctx.bookability.get("models") or {}).get("global"), feature_row)
    if logistic is None:
        return empirical, empirical_source
    return logistic, "logistic"


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


def _model_row(row: dict[str, Any], bucket_key: str) -> dict[str, Any]:
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
        "pair_quality": pair_quality,
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
    policy_rec: dict[str, Any] | None,
) -> float | None:
    if variant == "market_no_vig":
        return market_prob
    if variant == "market_residual":
        return residual_prob
    if variant == "distribution":
        return distribution_prob
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


def _is_tail_alt(stat: str, side: str, line: float | None, surface: str) -> bool:
    if surface == "alt_tail":
        return True
    if side != "over" or line is None:
        return False
    return (
        (stat == "batter_hits" and line >= 2.5)
        or (stat == "batter_total_bases" and line >= 3.5)
        or (stat == "batter_home_runs" and line >= 1.5)
    )


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
    model_prob = _clean_float(feature_row.get("model_prob_side"))
    market_prob = _clean_float(feature_row.get("market_prob_side"))
    distribution_prob = _clean_float(feature_row.get("distribution_prob_side"))
    price = _prediction_price(feature_row)
    breakeven = american_to_prob(price)
    residual_prob = _logistic_score((ctx.residual.get("models") or {}).get("global"), feature_row)
    clv_prob = _logistic_score((ctx.residual.get("models") or {}).get("clv_beat"), feature_row)
    bookable_prob, bookability_source = _bookability_score(ctx, feature_row, bucket_key)

    policy_level, policy_rec = _walk_forward_record(ctx, bucket_key)
    residual_bucket = _residual_bucket_record(ctx, bucket_key)
    trust = ctx.trust_scores.get(bucket_key) or {}
    variant = str((policy_rec or {}).get("variant") or "model_only")
    if residual_bucket and str(residual_bucket.get("decision") or "").startswith("use_market_residual"):
        variant = "market_residual"
    selected_prob = _policy_prob(variant, model_prob, market_prob, residual_prob, distribution_prob, policy_rec)
    selected_ev = ev_per_unit(selected_prob, price)

    stat = str(feature_row.get("market") or "")
    side = str(feature_row.get("side") or "")
    line = _prediction_line(feature_row)
    surface = str(feature_row.get("line_surface") or "unknown")
    tail_alt = _is_tail_alt(stat, side, line, surface)
    residual_decision = str((residual_bucket or {}).get("decision") or "")
    no_bet_decision = residual_decision.startswith("no_bet")
    pair_quality = str(feature_row.get("pair_quality") or "unknown").lower()
    market_prob_source = str(feature_row.get("market_prob_source") or "unknown").lower()
    market_pair_confirms = (
        pair_quality in {"same_book", "cross_book"}
        and market_prob_source not in {"raw_implied", "synthetic_fanduel_over_only"}
    )
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

    model_wins = (
        selected_prob is not None
        and breakeven is not None
        and selected_ev is not None
        and selected_ev >= cfg.min_ev
        and selected_prob > breakeven
    )
    clv_wins = clv_prob is not None and clv_prob >= cfg.min_clv_beat_prob
    bookable = bookable_prob is not None and bookable_prob >= cfg.min_bookable_prob
    real_candidate = (
        model_wins
        and clv_wins
        and bookable
        and bucket_confirms
        and opportunity_confirms
        and market_pair_confirms
        and not tail_alt
        and not no_bet_decision
    )

    reasons: list[str] = []
    if not model_wins:
        reasons.append("model_or_price_no_edge")
    if not clv_wins:
        reasons.append("clv_model_not_confirming")
    if not bookable:
        reasons.append("bookability_not_confirming")
    if trust_status not in {"bankroll", "starter", "micro"}:
        reasons.append(f"bucket_{trust_status or 'closed'}")
    elif not bucket_confirms:
        reasons.append("bucket_history_not_confirming")
    if not opportunity_confirms:
        reasons.append("opportunity_not_confirming")
    if not market_pair_confirms:
        reasons.append(f"pair_quality_{pair_quality or 'unknown'}")
    elif pair_quality == "cross_book":
        reasons.append("cross_book_market_pair")
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
    if no_bet_decision:
        reasons.append(residual_decision)
        for blocker in (residual_bucket or {}).get("residual_proof_blockers") or []:
            reasons.append(str(blocker))
    if _clean_bool(row.get("bankroll_candidate")) and not real_candidate:
        reasons.append("bankroll_downgrade_recommended")

    score_ev = selected_ev if selected_ev is not None else (_clean_float(row.get("ev")) or -0.25)
    score = max(-0.35, min(0.45, score_ev))
    if clv_prob is not None:
        score += 0.25 * (clv_prob - 0.50)
    else:
        score -= 0.05
    if bookable_prob is not None:
        score += 0.10 * (bookable_prob - 0.50)
    else:
        score -= 0.03
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
    elif pair_quality in {"synthetic", "one_sided", "unknown"}:
        score -= 0.18
    if no_bet_decision:
        score -= 0.25
    if tail_alt and trust_status not in {"bankroll", "starter", "micro"}:
        score -= 0.50
    if not _clean_bool(row.get("price_drift_ok", True)):
        score -= 0.50

    tier = "watch"
    if real_candidate:
        tier = trust_status if trust_status in {"micro", "starter", "bankroll"} else "watch"
    elif tail_alt:
        tier = "lottery"
    elif no_bet_decision:
        tier = "no_bet"
    elif model_wins and opportunity_confirms:
        tier = "paper"
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
        "model_prob_side": model_prob,
        "market_prob_side": market_prob,
        "market_prob_source": feature_row.get("market_prob_source"),
        "pair_quality": pair_quality,
        "residual_prob_side": residual_prob,
        "distribution_prob_side": distribution_prob,
        "clv_beat_prob": clv_prob,
        "bookable_prob": bookable_prob,
        "bookability_source": bookability_source,
        "selector_prob_side": selected_prob,
        "selector_ev": selected_ev,
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
    TRUE AS price_drift_ok,
    e.market_prob_side::float AS training_market_prob_side,
    e.market_prob_source,
    e.paired_price_source,
    e.pair_quality,
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
FROM bets.mlb_prop_predictions pp
LEFT JOIN LATERAL (
    SELECT *
    FROM features.mlb_prop_market_training_examples e
    WHERE e.prediction_key = pp.prediction_key
    ORDER BY e.example_updated_at DESC, e.id DESC
    LIMIT 1
) e ON TRUE
WHERE pp.game_date_et = %(report_date)s
  AND COALESCE(pp.is_active, TRUE) IS TRUE
  AND pp.bet_side IN ('over','under')
"""


def load_active_rows(conn, report_date: date) -> list[dict[str, Any]]:
    if not _table_exists(conn, "bets", "mlb_prop_predictions"):
        return []
    has_training_table = _table_exists(conn, "features", "mlb_prop_market_training_examples")
    has_training_columns = has_training_table and _table_has_columns(
        conn,
        "features",
        "mlb_prop_market_training_examples",
        {"market_prob_source", "paired_price_source", "pair_quality"},
    )
    sql = (
        _ACTIVE_SQL_JOIN
        if has_training_columns
        else _ACTIVE_SQL_BASE
    )
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, {"report_date": report_date})
        return [dict(row) for row in cur.fetchall()]


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
    no_bet = [
        row for row in scored
        if row.get("selector_tier") == "no_bet" or row.get("no_bet_decision")
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
        "paper_rows": sum(1 for row in scored if row.get("selector_tier") == "paper"),
        "lottery_rows": sum(1 for row in scored if row.get("selector_tier") == "lottery"),
        "no_bet_rows": sum(1 for row in scored if row.get("selector_tier") == "no_bet"),
        "top_rows": scored[: cfg.top_n],
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
                "clv": _fmt_pct(row.get("clv_beat_prob")),
                "bookable": _fmt_pct(row.get("bookable_prob")),
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
        ("Book Src", "book_src"),
        ("Pair", "pair"),
        ("Trust", "trust"),
        ("Score", "score"),
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
        f"Paper rows: {payload.get('paper_rows')}",
        f"Lottery rows: {payload.get('lottery_rows')}",
        f"No-bet rows: {payload.get('no_bet_rows')}",
        "",
        "## Best Common-Line Paper Props",
        "",
        _table(_display_rows(payload.get("best_common_paper_rows") or []), row_columns),
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
    path.write_text(text, encoding="utf-8")
    return str(path)


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
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = write_report(payload, cfg)
    print(json.dumps({
        "status": "ok",
        "active_rows": payload.get("active_rows"),
        "real_candidate_rows": payload.get("real_candidate_rows"),
        "report_path": report_path,
    }, indent=2))


if __name__ == "__main__":
    main()
