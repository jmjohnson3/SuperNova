"""Runtime helpers for the MLB prop betting layer.

The projection model estimates player outcomes.  This betting layer estimates
whether a specific offered side is mispriced after seeing model probability,
book line, price, line bucket, price bucket, side, and model family.
"""
from __future__ import annotations

import math
from typing import Any

from .prop_replay import american_to_prob, no_vig_probs
from .side_recalibration import clean_float, logit, price_bucket, prop_line_bucket, sigmoid

_NUMERIC_FEATURES = [
    "raw_p_side",
    "market_prob_side",
    "prob_edge_vs_market",
    "market_line",
    "abs_price",
    "is_plus_price",
]

_CATEGORICAL_FEATURES = [
    "market",
    "side",
    "line_bucket",
    "price_bucket",
    "model_family",
]

_MARKET_PRIOR_NUMERIC_FEATURES = [
    "raw_p_side",
    "market_line",
    "abs_price",
    "is_plus_price",
]

_MARKET_PRIOR_CATEGORICAL_FEATURES = [
    "market",
    "side",
    "line_bucket",
    "price_bucket",
    "bookmaker_key",
]


def betting_layer_key(
    market: str,
    side: str,
    line_bucket: str = "*",
    model_family: str = "*",
) -> str:
    return "|".join([
        str(market or "*"),
        str(side or "*"),
        str(line_bucket or "*"),
        str(model_family or "*"),
    ])


def market_side_prior_key(
    market: str,
    side: str,
    line_bucket: str = "*",
    price_bucket: str = "*",
    bookmaker_key: str = "*",
) -> str:
    return "|".join([
        str(market or "*"),
        str(side or "*"),
        str(line_bucket or "*"),
        str(price_bucket or "*"),
        str(bookmaker_key or "*"),
    ])


def _lookup_model(
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line_bucket: str,
    model_family: str,
    section: str = "models",
) -> tuple[str | None, dict[str, Any] | None]:
    if not payload:
        return None, None
    models = payload.get(section) or {}
    priorities = [
        (market, side, line_bucket, model_family),
        (market, side, line_bucket, "*"),
        (market, side, "*", model_family),
        (market, side, "*", "*"),
        ("*", side, "*", model_family),
        ("*", side, "*", "*"),
        ("*", "*", "*", "*"),
    ]
    for m, s, lb, mf in priorities:
        key = betting_layer_key(m, s, lb, mf)
        rec = models.get(key)
        if rec:
            return key, rec
    return None, None


def _lookup_market_side_prior(
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line_bucket: str,
    price_bucket: str,
    bookmaker_key: str,
) -> tuple[str | None, dict[str, Any] | None]:
    if not payload:
        return None, None
    models = payload.get("models") or {}
    priorities = [
        (market, side, line_bucket, price_bucket, bookmaker_key),
        (market, side, line_bucket, price_bucket, "*"),
        (market, side, line_bucket, "*", bookmaker_key),
        (market, side, line_bucket, "*", "*"),
        (market, side, "*", "*", bookmaker_key),
        (market, side, "*", "*", "*"),
        ("*", side, "*", "*", bookmaker_key),
        ("*", side, "*", "*", "*"),
        ("*", "*", "*", "*", "*"),
    ]
    for m, s, lb, pb, bk in priorities:
        key = market_side_prior_key(m, s, lb, pb, bk)
        rec = models.get(key)
        if rec:
            return key, rec
    return None, None


def build_betting_features(
    *,
    market: str,
    side: str,
    line,
    price,
    raw_p_side,
    over_price=None,
    under_price=None,
    model_family: str = "unknown",
    bookmaker_key: str = "unknown",
) -> dict[str, Any]:
    raw = clean_float(raw_p_side)
    line_v = clean_float(line)
    price_v = clean_float(price)
    market_prob = american_to_prob(price_v)
    nv_over, nv_under = no_vig_probs(over_price, under_price)
    no_vig_side = nv_over if side == "over" else nv_under
    if no_vig_side is None:
        no_vig_side = market_prob
    return {
        "market": str(market or "unknown"),
        "side": str(side or "unknown"),
        "line_bucket": prop_line_bucket(str(market or ""), line_v),
        "price_bucket": price_bucket(price_v),
        "bookmaker_key": str(bookmaker_key or "unknown"),
        "model_family": str(model_family or "unknown"),
        "raw_p_side": raw,
        "market_prob_side": no_vig_side,
        "prob_edge_vs_market": (raw - no_vig_side) if raw is not None and no_vig_side is not None else None,
        "market_line": line_v,
        "abs_price": abs(price_v) if price_v is not None else None,
        "is_plus_price": 1.0 if price_v is not None and price_v > 0 else 0.0 if price_v is not None else None,
    }


def _score_model(
    features: dict[str, Any],
    rec: dict[str, Any],
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> float | None:
    coef = rec.get("coef") or {}
    intercept = clean_float(rec.get("intercept"))
    if intercept is None:
        return None
    numeric_features = numeric_features or _NUMERIC_FEATURES
    categorical_features = categorical_features or _CATEGORICAL_FEATURES
    z = intercept
    means = rec.get("numeric_means") or {}
    scales = rec.get("numeric_scales") or {}
    for name in numeric_features:
        value = clean_float(features.get(name))
        if value is None:
            value = clean_float(means.get(name)) or 0.0
        scale = clean_float(scales.get(name)) or 1.0
        mean = clean_float(means.get(name)) or 0.0
        z += float(coef.get(name, 0.0)) * ((value - mean) / scale)
    for name in categorical_features:
        value = str(features.get(name) or "unknown")
        z += float(coef.get(f"{name}={value}", 0.0))
    return max(1e-6, min(1.0 - 1e-6, sigmoid(z)))


def apply_prop_market_side_prior(
    model_p_side,
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line,
    price,
    over_price=None,
    under_price=None,
    bookmaker_key: str = "unknown",
    max_blend_weight: float = 0.35,
) -> tuple[float | None, str | None]:
    """Blend a player-model side probability toward a historical market prior.

    The prior is trained on market/no-vig probability, line, price bucket, and
    book.  It should be a conservative anchor, not a replacement for the player
    model, so its artifact blend weight is capped by ``max_blend_weight``.
    """
    p_model = clean_float(model_p_side)
    if p_model is None:
        return None, None
    p_model = max(1e-6, min(1.0 - 1e-6, p_model))
    line_v = clean_float(line)
    price_v = clean_float(price)
    market_prob = american_to_prob(price_v)
    nv_over, nv_under = no_vig_probs(over_price, under_price)
    no_vig_side = nv_over if side == "over" else nv_under
    if no_vig_side is not None:
        market_prob = no_vig_side
    if market_prob is None:
        return p_model, None
    features = {
        "market": str(market or "unknown"),
        "side": str(side or "unknown"),
        "line_bucket": prop_line_bucket(str(market or ""), line_v),
        "price_bucket": price_bucket(price_v),
        "bookmaker_key": str(bookmaker_key or "unknown"),
        "raw_p_side": max(1e-6, min(1.0 - 1e-6, market_prob)),
        "market_line": line_v,
        "abs_price": abs(price_v) if price_v is not None else None,
        "is_plus_price": 1.0 if price_v is not None and price_v > 0 else 0.0 if price_v is not None else None,
    }
    key, rec = _lookup_market_side_prior(
        payload,
        market=str(market or "unknown"),
        side=str(side or "unknown"),
        line_bucket=str(features["line_bucket"]),
        price_bucket=str(features["price_bucket"]),
        bookmaker_key=str(features["bookmaker_key"]),
    )
    if not rec:
        return p_model, None
    prior_p = _score_model(
        features,
        rec,
        numeric_features=_MARKET_PRIOR_NUMERIC_FEATURES,
        categorical_features=_MARKET_PRIOR_CATEGORICAL_FEATURES,
    )
    if prior_p is None:
        return p_model, None
    weight = clean_float(rec.get("blend_weight"))
    if weight is None:
        weight = 1.0
    weight = max(0.0, min(float(max_blend_weight), weight * float(max_blend_weight)))
    out = sigmoid(logit(p_model) + weight * (logit(prior_p) - logit(p_model)))
    return max(1e-6, min(1.0 - 1e-6, out)), key


def apply_prop_betting_layer(
    raw_p_side,
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line,
    price,
    over_price=None,
    under_price=None,
    model_family: str = "unknown",
    bookmaker_key: str = "unknown",
) -> tuple[float | None, str | None]:
    raw = clean_float(raw_p_side)
    if raw is None:
        return None, None
    raw = max(1e-6, min(1.0 - 1e-6, raw))
    features = build_betting_features(
        market=market,
        side=side,
        line=line,
        price=price,
        raw_p_side=raw,
        over_price=over_price,
        under_price=under_price,
        model_family=model_family,
        bookmaker_key=bookmaker_key,
    )
    key, rec = _lookup_model(
        payload,
        market=str(market or "unknown"),
        side=str(side or "unknown"),
        line_bucket=str(features["line_bucket"]),
        model_family=str(model_family or "unknown"),
        section="models",
    )
    if not rec:
        return raw, None
    p_model = _score_model(features, rec)
    if p_model is None:
        return raw, None
    weight = clean_float(rec.get("blend_weight"))
    if weight is None:
        weight = 1.0
    weight = max(0.0, min(1.0, weight))
    # Blend in logit space so the layer can correct overconfidence without
    # crushing high/low probabilities linearly.
    out = sigmoid(logit(raw) + weight * (logit(p_model) - logit(raw)))
    return max(1e-6, min(1.0 - 1e-6, out)), key


def apply_prop_clv_layer(
    raw_p_side,
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line,
    price,
    over_price=None,
    under_price=None,
    model_family: str = "unknown",
    bookmaker_key: str = "unknown",
) -> tuple[float | None, str | None]:
    """Score the chance that this side beats the closing price."""
    raw = clean_float(raw_p_side)
    if raw is None:
        return None, None
    features = build_betting_features(
        market=market,
        side=side,
        line=line,
        price=price,
        raw_p_side=max(1e-6, min(1.0 - 1e-6, raw)),
        over_price=over_price,
        under_price=under_price,
        model_family=model_family,
        bookmaker_key=bookmaker_key,
    )
    key, rec = _lookup_model(
        payload,
        market=str(market or "unknown"),
        side=str(side or "unknown"),
        line_bucket=str(features["line_bucket"]),
        model_family=str(model_family or "unknown"),
        section="clv_models",
    )
    if not rec:
        return None, None
    p_model = _score_model(features, rec)
    if p_model is None:
        return None, None
    weight = clean_float(rec.get("blend_weight"))
    if weight is None:
        weight = 1.0
    weight = max(0.0, min(1.0, weight))
    baseline = clean_float(rec.get("baseline_rate_train"))
    if baseline is None:
        baseline = 0.5
    baseline = max(1e-6, min(1.0 - 1e-6, baseline))
    out = sigmoid(logit(baseline) + weight * (logit(p_model) - logit(baseline)))
    return max(1e-6, min(1.0 - 1e-6, out)), key
