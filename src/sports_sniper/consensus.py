"""A fair probability for one bet, from what the other sportsbooks charge for the identical bet.

Each reference book must price both sides at the same line; its margin is removed (odds_math.no_vig) and the
books are averaged. The book being judged never takes part in its own reference. A one-sided market gets no
probability at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sports_sniper import odds_math
from sports_sniper.normalize import OPPOSITE, Quote


@dataclass(frozen=True)
class Consensus:
    probability: float            # of the quoted side
    standard_error: float
    books: tuple[str, ...]
    has_sharp: bool
    spread: float                 # weighted standard deviation across the reference books

    @property
    def n_books(self) -> int:
        return len(self.books)


def fair_probability(quote: Quote, opposite: Quote, method: str) -> float:
    """The quoted side's probability at one book once its margin is removed."""
    return odds_math.no_vig([quote.price, opposite.price], method)[0]


def consensus_for(quote: Quote, books: dict[str, dict[str, Quote]], *, method: str = "power",
                  weights: dict[str, float] | None = None, sharp_books: tuple[str, ...] = ("pinnacle",),
                  max_skew_seconds: float = 180, uncertainty_floor: float = 0.015,
                  soft_only_multiplier: float = 2.0) -> Consensus | None:
    """`books` maps book -> side -> quote for this exact selection. None when no other book prices both sides.

    The standard error is the disagreement between the reference books (weighted standard deviation over the
    square root of the effective number of books), never less than `uncertainty_floor`: one book cannot
    disagree with itself, and removing a margin is itself an estimate. With no sharp book among the references
    the floor is multiplied by `soft_only_multiplier`, because retail books copy each other.
    """
    weights = weights or {}
    used = []
    for book, sides in books.items():
        ours, theirs = sides.get(quote.side), sides.get(OPPOSITE[quote.side])
        if book == quote.book or ours is None or theirs is None:
            continue
        if abs((ours.last_update - quote.last_update).total_seconds()) > max_skew_seconds:
            continue
        used.append((book, fair_probability(ours, theirs, method), float(weights.get(book, 1.0))))
    if not used:
        return None
    total = sum(w for _, _, w in used)
    mean = sum(p * w for _, p, w in used) / total
    variance = sum(w * (p - mean) ** 2 for _, p, w in used) / total
    effective = total ** 2 / sum(w ** 2 for _, _, w in used)
    has_sharp = any(book in sharp_books for book, _, _ in used)
    floor = uncertainty_floor * (1.0 if has_sharp else soft_only_multiplier)
    return Consensus(probability=mean, standard_error=max(floor, math.sqrt(variance / effective)),
                     books=tuple(sorted(book for book, _, _ in used)), has_sharp=has_sharp, spread=math.sqrt(variance))
