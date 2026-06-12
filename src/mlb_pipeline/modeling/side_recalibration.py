"""Side-specific probability recalibration helpers.

These helpers calibrate the probability of the selected betting side, not just
P(over).  That lets an under at one line/price learn a different correction
than an over at another line/price.
"""
from __future__ import annotations

import math
from typing import Any


def clean_float(value) -> float | None:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(z)))


def price_bucket(price) -> str:
    p = clean_float(price)
    if p is None:
        return "missing_price"
    if p > 0:
        if p < 150:
            return "plus_100_149"
        if p < 250:
            return "plus_150_249"
        if p < 500:
            return "plus_250_499"
        return "plus_500_plus"
    if p >= -129:
        return "fair_lay"
    if p >= -149:
        return "lay_130_149"
    if p >= -180:
        return "lay_150_180"
    return "heavy_lay"


def prop_line_bucket(stat: str, line) -> str:
    line_v = clean_float(line)
    if line_v is None:
        return "missing_line"
    if stat == "pitcher_strikeouts":
        if line_v < 4.5:
            return "K <4.5"
        if line_v < 6.5:
            return "K 4.5-6.0"
        if line_v < 8.5:
            return "K 6.5-8.0"
        return "K 8.5+"
    if stat == "batter_total_bases":
        if line_v < 1.0:
            return "TB 0.5"
        if line_v < 2.0:
            return "TB 1.5"
        if line_v < 3.0:
            return "TB 2.5"
        if line_v < 4.0:
            return "TB 3.5"
        return "TB 4.5+"
    if stat == "batter_hits":
        if line_v < 1.0:
            return "H 0.5"
        if line_v < 2.0:
            return "H 1.5"
        if line_v < 3.0:
            return "H 2.5"
        return "H 3.5+"
    if stat == "batter_home_runs":
        return "HR 0.5" if line_v < 1.0 else "HR 1.5+"
    return "other"


def prop_line_surface(stat: str, side: str | None, line) -> str:
    """Classify an offered prop line into common/canonical vs high-variance alt tail."""
    line_v = clean_float(line)
    side_v = (side or "").lower()
    if line_v is None:
        return "missing_line"
    if side_v != "over":
        return "common"
    if stat == "batter_hits" and line_v >= 2.5:
        return "alt_tail"
    if stat == "batter_total_bases" and line_v >= 3.5:
        return "alt_tail"
    if stat == "batter_home_runs" and line_v >= 1.5:
        return "alt_tail"
    return "common"


def game_total_line_bucket(line) -> str:
    line_v = clean_float(line)
    if line_v is None:
        return "missing_line"
    if line_v < 8.0:
        return "total <8"
    if line_v < 9.5:
        return "total 8-9"
    if line_v < 11.0:
        return "total 9.5-10.5"
    return "total 11+"


def calibration_key(
    market: str,
    side: str,
    line_bucket: str = "*",
    price_bucket_value: str = "*",
    model_family: str = "*",
) -> str:
    return "|".join([
        str(market or "*"),
        str(side or "*"),
        str(line_bucket or "*"),
        str(price_bucket_value or "*"),
        str(model_family or "*"),
    ])


def lookup_calibrator(
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line_bucket: str,
    price_bucket_value: str,
    model_family: str,
) -> tuple[str | None, dict[str, Any] | None]:
    if not payload:
        return None, None
    calibrators = payload.get("calibrators") or {}
    priorities = [
        (line_bucket, price_bucket_value, model_family),
        (line_bucket, "*", model_family),
        (line_bucket, price_bucket_value, "*"),
        (line_bucket, "*", "*"),
        ("*", price_bucket_value, model_family),
        ("*", "*", model_family),
        ("*", price_bucket_value, "*"),
        ("*", "*", "*"),
    ]
    for lb, pb, mf in priorities:
        key = calibration_key(market, side, lb, pb, mf)
        cal = calibrators.get(key)
        if cal:
            return key, cal
    return None, None


def apply_side_calibrator(
    raw_p_side,
    payload: dict[str, Any] | None,
    *,
    market: str,
    side: str,
    line_bucket: str,
    price_bucket_value: str,
    model_family: str,
) -> tuple[float | None, str | None]:
    raw = clean_float(raw_p_side)
    if raw is None:
        return None, None
    raw = min(1.0 - 1e-6, max(1e-6, raw))
    key, cal = lookup_calibrator(
        payload,
        market=market,
        side=side,
        line_bucket=line_bucket,
        price_bucket_value=price_bucket_value,
        model_family=model_family,
    )
    if not cal:
        return raw, None
    method = str(cal.get("method") or "").lower()
    p_cal = raw
    if method == "platt":
        try:
            p_cal = sigmoid(float(cal.get("a")) * logit(raw) + float(cal.get("b")))
        except Exception:
            p_cal = raw
    elif method == "constant":
        p_cal = clean_float(cal.get("actual_rate")) or raw
    weight = clean_float(cal.get("blend_weight"))
    if weight is None:
        weight = 1.0
    weight = min(1.0, max(0.0, weight))
    out = raw + weight * (p_cal - raw)
    return min(1.0 - 1e-6, max(1e-6, out)), key
