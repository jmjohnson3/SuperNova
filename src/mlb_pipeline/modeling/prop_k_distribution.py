"""Lightweight scoring helpers for the pitcher strikeout v3 distribution."""
from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def score_k_rate(row: dict[str, Any], artifact: dict[str, Any]) -> float | None:
    model = artifact.get("rate_model") or {}
    if not model:
        return None
    value = float(model.get("intercept") or 0.0)
    means = model.get("numeric_means") or {}
    scales = model.get("numeric_scales") or {}
    coefficients = model.get("coef") or {}
    standardized: dict[str, float] = {}
    for name in model.get("numeric_features") or []:
        mean = _number(means.get(name)) or 0.0
        scale = _number(scales.get(name)) or 1.0
        if abs(scale) <= 1e-9:
            scale = 1.0
        feature = _number(row.get(name))
        if feature is None:
            feature = mean
        standardized[name] = (feature - mean) / scale
        value += float(coefficients.get(name) or 0.0) * standardized[name]
    for term in model.get("derived_terms") or []:
        term_name = str(term.get("name") or "")
        left = standardized.get(str(term.get("left") or ""), 0.0)
        kind = str(term.get("kind") or "")
        if kind == "square":
            term_value = left * left
        elif kind == "interaction":
            right = standardized.get(str(term.get("right") or ""), 0.0)
            term_value = left * right
        else:
            continue
        value += float(coefficients.get(term_name) or 0.0) * term_value
    return max(1e-5, min(1.0 - 1e-5, _sigmoid(value)))


def _normal_bf_pmf(mean: float, sigma: float) -> dict[int, float]:
    sigma = max(1.25, min(8.0, float(sigma)))
    lo = max(8, int(math.floor(mean - 4.0 * sigma)))
    hi = min(40, int(math.ceil(mean + 4.0 * sigma)))
    values: dict[int, float] = {}
    root_two = math.sqrt(2.0)
    for bf in range(lo, hi + 1):
        upper = 0.5 * (1.0 + math.erf(((bf + 0.5) - mean) / (sigma * root_two)))
        lower = 0.5 * (1.0 + math.erf(((bf - 0.5) - mean) / (sigma * root_two)))
        values[bf] = max(0.0, upper - lower)
    total = sum(values.values()) or 1.0
    return {bf: probability / total for bf, probability in values.items()}


def _beta_binomial_pmf(k: int, n: int, probability: float, concentration: float) -> float:
    probability = max(1e-6, min(1.0 - 1e-6, float(probability)))
    concentration = max(2.0, float(concentration))
    alpha = probability * concentration
    beta = (1.0 - probability) * concentration
    log_value = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + math.lgamma(k + alpha)
        + math.lgamma(n - k + beta)
        - math.lgamma(n + alpha + beta)
        + math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )
    return math.exp(log_value)


def score_k_v3_over_probability(
    row: dict[str, Any],
    line: Any,
    artifact: dict[str, Any],
) -> float | None:
    if not artifact or artifact.get("status") != "trained":
        return None
    line_value = _number(line)
    projected_bf = _number(row.get("opp_model_bf")) or _number(row.get("projected_bf"))
    if line_value is None or projected_bf is None or projected_bf <= 0:
        return None
    rate = score_k_rate(row, artifact)
    if rate is None:
        return None
    bf_mean = projected_bf + float(artifact.get("bf_bias") or 0.0)
    bf_sigma = float(artifact.get("bf_sigma") or 3.5)
    concentration = float(artifact.get("beta_concentration") or 80.0)
    threshold = math.floor(line_value)
    over = 0.0
    for bf, bf_probability in _normal_bf_pmf(bf_mean, bf_sigma).items():
        conditional = sum(
            _beta_binomial_pmf(k, bf, rate, concentration)
            for k in range(max(0, threshold + 1), bf + 1)
        )
        over += bf_probability * conditional
    return max(1e-6, min(1.0 - 1e-6, over))
