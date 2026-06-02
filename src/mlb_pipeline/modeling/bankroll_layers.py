"""Shared shadow-bankroll classification helpers.

These helpers intentionally do not suppress picks. They label each model signal
so the scripts can keep showing volume while separating paper-tracked plays
from bankroll candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BankrollAssessment:
    tier: str
    candidate: bool
    reasons: str
    stake_pct: float


def _clean_reasons(reasons: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for reason in reasons or []:
        label = str(reason).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def capped_quarter_kelly_pct(
    kelly_fraction: float | int | None,
    *,
    max_stake_pct: float = 0.005,
) -> float:
    """Return a capped 1/4-Kelly stake as a fraction of bankroll."""
    if kelly_fraction is None:
        return 0.0
    try:
        kelly = float(kelly_fraction)
    except (TypeError, ValueError):
        return 0.0
    if kelly <= 0:
        return 0.0
    return max(0.0, min(float(max_stake_pct), kelly / 4.0))


def assess_bankroll_layer(
    *,
    has_signal: bool,
    hard_blocks: Iterable[str] | None = None,
    soft_warnings: Iterable[str] | None = None,
    kelly_fraction: float | int | None = None,
    max_stake_pct: float = 0.005,
    no_signal_reason: str = "no_model_edge",
) -> BankrollAssessment:
    """Classify a signal without changing whether the signal is displayed."""
    if not has_signal:
        return BankrollAssessment(
            tier="research",
            candidate=False,
            reasons=no_signal_reason or "no_model_edge",
            stake_pct=0.0,
        )

    hard = _clean_reasons(hard_blocks)
    soft = _clean_reasons(soft_warnings)
    if hard:
        return BankrollAssessment(
            tier="paper",
            candidate=False,
            reasons="; ".join(hard),
            stake_pct=0.0,
        )
    if soft:
        return BankrollAssessment(
            tier="paper",
            candidate=False,
            reasons="; ".join(soft),
            stake_pct=0.0,
        )

    return BankrollAssessment(
        tier="bankroll_candidate",
        candidate=True,
        reasons="",
        stake_pct=capped_quarter_kelly_pct(
            kelly_fraction,
            max_stake_pct=max_stake_pct,
        ),
    )


def bankroll_tag(assessment: BankrollAssessment) -> str:
    if assessment.candidate:
        return f"BANKROLL {assessment.stake_pct * 100:.2f}%"
    suffix = f": {assessment.reasons}" if assessment.reasons else ""
    return f"{assessment.tier.upper()}{suffix}"
