"""Price math for comparing FanDuel to a sharp book: no-vig, nearby-line conversion, EV and minimum price.

Sharp books often hang a slightly different line (45.5 vs 46.5). Exact-line-only comparison throws
those away, so a sharp no-vig probability is moved to FanDuel's line with a normal approximation:
the sharp quote fixes the median, and a stat-level spread (sigma) sets how fast probability changes
per unit of line. Conversions are only allowed within a small window per stat; spreads are never
shifted (key numbers 3/7 make a smooth curve wrong).
"""
from __future__ import annotations

import math

from scipy.stats import norm

# Typical spread of the outcome around a fair line, and the largest line gap we will convert across.
LINE_MODEL = {
    "receiving_yards": dict(sigma=26.0, max_gap=3.0),
    "rushing_yards": dict(sigma=22.0, max_gap=3.0),
    "passing_yards": dict(sigma=55.0, max_gap=6.0),
    "receptions": dict(sigma=1.9, max_gap=0.0),   # counts: exact line only
    "passing_tds": dict(sigma=0.95, max_gap=0.0),
    "pass_completions": dict(sigma=5.0, max_gap=0.0),
    "pass_attempts": dict(sigma=6.0, max_gap=0.0),
    "rush_attempts": dict(sigma=4.0, max_gap=0.0),
    "total": dict(sigma=10.5, max_gap=1.0),
    "spread": dict(sigma=13.5, max_gap=0.0),      # key numbers: exact line only
    "anytime_td": dict(sigma=1.0, max_gap=0.0),   # yes/no at a notional 0.5 line: exact only
}


def implied(price: float | None) -> float | None:
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or abs(price) < 100:
        return None
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


DEVIG_METHOD = "power"


def no_vig_over(over_price, under_price, method: str = DEVIG_METHOD) -> float | None:
    """Fair P(over) from a two-sided quote, with the book's margin removed.

    Proportional de-vig (`over / (over + under)`) assumes the book spreads its margin evenly across
    both sides. It does not: the unlikely side carries more of it, so dividing the overround out
    evenly leaves longshots looking better than they are. On an anytime-TD price of +380 / -520 that
    error is worth about two points of probability, which is the whole edge.

    The power method instead solves `over**k + under**k == 1`, which shrinks the longer price harder.
    For a near-even prop (-125/+105) the two agree to within 0.2 points, so switching costs the
    yardage markets nothing; it only bites where it should.
    """
    over, under = implied(over_price), implied(under_price)
    if over is None or under is None or over + under <= 0:
        return None
    proportional = over / (over + under)
    if method == "proportional" or not (0.0 < over < 1.0 and 0.0 < under < 1.0):
        return proportional
    try:
        from scipy.optimize import brentq
        k = brentq(lambda k: over ** k + under ** k - 1.0, 0.05, 20.0, xtol=1e-12)
    except (ValueError, RuntimeError, ImportError):
        return proportional  # no sign change in the bracket (e.g. a quote with no margin at all)
    return float(over ** k)


def over_probability_at(stat: str, sharp_line: float, sharp_over: float, target_line: float) -> float | None:
    """Sharp P(over sharp_line) moved to target_line, or None if the gap is too large for this stat."""
    model = LINE_MODEL.get(stat)
    if model is None or sharp_over is None or not 0.0 < sharp_over < 1.0:
        return None
    gap = float(target_line) - float(sharp_line)
    if abs(gap) < 1e-9:
        return float(sharp_over)
    if abs(gap) > model["max_gap"] + 1e-9:
        return None
    sigma = model["sigma"]
    mean = float(sharp_line) + sigma * norm.ppf(sharp_over)  # P(X > line) = sharp_over
    return float(1.0 - norm.cdf(float(target_line), loc=mean, scale=sigma))


# How far a closing line may be moved back to the bet's line when measuring movement (wider than
# LINE_MODEL max_gap, which bounds pricing a bet off a different sharp line).
MAX_CLOSE_SHIFT = {"receiving_yards": 10.0, "rushing_yards": 10.0, "passing_yards": 20.0, "receptions": 1.0, "total": 3.0}


def shift_probability(stat: str, from_line: float, p_over: float | None, to_line: float) -> float | None:
    """No-vig P(over from_line) moved to to_line, for measuring line movement (FanDuel moves props by
    line at a fixed price). None beyond MAX_CLOSE_SHIFT or for exact-line-only stats."""
    model = LINE_MODEL.get(stat)
    if model is None or p_over is None or not 0.0 < p_over < 1.0:
        return None
    if abs(float(to_line) - float(from_line)) < 1e-9:
        return float(p_over)
    if abs(float(to_line) - float(from_line)) > MAX_CLOSE_SHIFT.get(stat, 0.0) or model["max_gap"] == 0.0:
        return None
    mean = float(from_line) + model["sigma"] * norm.ppf(p_over)
    return float(1.0 - norm.cdf(float(to_line), loc=mean, scale=model["sigma"]))


def ev(probability: float, price) -> float | None:
    p = implied(price)
    if p is None or probability is None:
        return None
    payout = (1.0 - p) / p  # profit per unit staked
    return probability * payout - (1.0 - probability)


def minimum_price(probability: float, min_ev: float = 0.01) -> int | None:
    """Worst American price that still has at least min_ev at this probability."""
    if probability is None or not 0.0 < probability < 1.0:
        return None
    payout = (1.0 + min_ev) / probability - 1.0  # profit per unit needed
    if payout <= 0:
        return None
    price = 100.0 * payout if payout >= 1.0 else -100.0 / payout
    return int(math.ceil(price)) if price > 0 else int(math.ceil(price))
