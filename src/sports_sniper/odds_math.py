"""Price and probability arithmetic. Pure functions; every probability is a number from 0 to 1.

American odds: +150 pays 150 on 100, -150 needs 150 to win 100. Decimal odds D pay D per 1 staked,
stake included, so the break-even probability is 1 / D and the expected return per 1 staked is
p * (D - 1) - (1 - p) = p * D - 1.
"""
from __future__ import annotations

import math
from typing import Sequence

DEVIG_METHODS = ("power", "multiplicative")


def american_to_decimal(american: float) -> float:
    american = float(american)
    if not math.isfinite(american) or abs(american) < 100:
        raise ValueError(f"not an American price: {american}")
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def decimal_to_american(decimal: float) -> int:
    decimal = float(decimal)
    if not math.isfinite(decimal) or decimal <= 1:
        raise ValueError(f"decimal odds must be above 1: {decimal}")
    return int(round((decimal - 1) * 100)) if decimal >= 2 else int(round(-100 / (decimal - 1)))


def implied_probability(american: float) -> float:
    """The break-even probability of a price, sportsbook margin included."""
    return 1 / american_to_decimal(american)


def no_vig(prices: Sequence[float], method: str = "power") -> list[float]:
    """Fair probabilities for a complete market (every outcome priced), margin removed.

    multiplicative: divide each implied probability by their sum. Simple, but it leaves too much
    probability on longshots.
    power: raise each implied probability to the power k that makes them sum to 1. It takes more
    margin out of longshots than favourites, which matches how books shade prices, and is the
    method the rest of this repository uses.

    A one-sided price cannot be de-vigged: at least two outcomes are required.
    """
    if method not in DEVIG_METHODS:
        raise ValueError(f"unknown de-vig method: {method}")
    implied = [implied_probability(p) for p in prices]
    if len(implied) < 2:
        raise ValueError("a no-vig probability needs every side of the market")
    total = sum(implied)
    if method == "multiplicative" or abs(total - 1) < 1e-12:
        return [p / total for p in implied]
    low, high = (1.0, 50.0) if total > 1 else (1e-6, 1.0)   # total > 1 is the normal case (a margin)
    for _ in range(100):
        k = (low + high) / 2
        if sum(p ** k for p in implied) > 1:
            low = k
        else:
            high = k
    fair = [p ** ((low + high) / 2) for p in implied]
    scale = sum(fair)
    return [p / scale for p in fair]


def expected_roi(probability: float, decimal: float) -> float:
    """Expected profit per 1 staked: p * (D - 1) - (1 - p)."""
    return probability * (decimal - 1) - (1 - probability)


def expected_value(probability: float, decimal: float, stake: float = 1.0) -> float:
    return stake * expected_roi(probability, decimal)


def fair_decimal(probability: float) -> float:
    if not 0 < probability < 1:
        raise ValueError(f"probability must be between 0 and 1: {probability}")
    return 1 / probability


def fair_american(probability: float) -> int:
    return decimal_to_american(fair_decimal(probability))


def conservative_probability(probability: float, standard_error: float, z: float = 1.0) -> float:
    """The estimate lowered by z standard errors (never below zero)."""
    return max(0.0, probability - z * max(0.0, standard_error))


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a win rate; well behaved at small n and at rates near 0 or 1."""
    if n <= 0:
        return 0.0, 1.0
    rate = wins / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probabilities against 0/1 outcomes (0 is perfect, 0.25 is a coin flip)."""
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("need one outcome per probability")
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(probabilities)


def calibration_table(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> list[dict]:
    """Predicted against observed frequency by probability bin; a calibrated model has them equal."""
    if len(probabilities) != len(outcomes):
        raise ValueError("need one outcome per probability")
    rows = []
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [(p, o) for p, o in zip(probabilities, outcomes) if low <= p < high or (b == bins - 1 and p == 1)]
        if members:
            wins = sum(o for _, o in members)
            rows.append(dict(low=low, high=high, n=len(members), predicted=sum(p for p, _ in members) / len(members),
                             observed=wins / len(members), interval=wilson_interval(wins, len(members))))
    return rows
