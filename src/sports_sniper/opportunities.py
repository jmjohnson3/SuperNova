"""Find and rank prices that look too generous against what other sportsbooks charge for the identical bet.

Every estimate here is market-consensus-derived: no prediction model is involved in this version. An estimated
edge is an estimate. It is ranked by its conservative value (the probability lowered by its uncertainty), and
a bet with no two-sided reference gets no probability and no EV at all.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass

from sports_sniper import odds_math
from sports_sniper.config import Settings
from sports_sniper.consensus import consensus_for
from sports_sniper.markets import book_label, market_label
from sports_sniper.normalize import Quote

SOURCE_MARKET = "market_consensus"
EVIDENCE_ORDER = {"strong": 0, "moderate": 1, "speculative": 2, "insufficient": 3}


@dataclass
class Opportunity:
    sport: str
    event_id: str
    event_name: str
    commence_time: object
    book: str
    market: str
    player: str
    player_key: str
    line: float | None
    side: str
    price: int
    decimal: float
    observed_at: object
    last_update: object
    link: str | None
    status: str                      # priced | insufficient_data
    source: str | None               # where the probability came from
    probability: float | None
    standard_error: float | None
    break_even: float
    fair_price: int | None
    ev: float | None
    conservative_ev: float | None
    reference_books: tuple[str, ...]
    has_sharp: bool
    evidence: str                    # strong | moderate | speculative | insufficient
    is_longshot: bool
    is_best_price: bool
    books_quoting: int
    premium_vs_median: float | None  # this price against the middle price at the other books to bet at
    minutes_to_start: float
    explanation: str
    model_probability: float | None = None   # the prediction model's calibrated chance of this side, when it has one
    model_version: str | None = None

    def record(self) -> dict:
        out = asdict(self)
        out["reference_books"] = ",".join(self.reference_books)
        return out


def describe(side: str, line: float | None) -> str:
    return side.capitalize() if line is None else f"{side.capitalize()} {line:g}"


def evidence_tier(ev: float, conservative_ev: float, has_sharp: bool, n_books: int) -> str:
    """strong: holds after the uncertainty haircut, with a sharp book and a second reference agreeing.
    moderate: holds after the haircut with a sharp book, or with three retail books.
    speculative: positive only before the haircut, or resting on one or two retail books."""
    if conservative_ev > 0 and has_sharp and n_books >= 2:
        return "strong"
    if conservative_ev > 0 and (has_sharp or n_books >= 3):
        return "moderate"
    return "speculative"


def detect(quotes: list[Quote], settings: Settings) -> list[Opportunity]:
    """Ranked opportunities from one batch of quotes: priced ones first (best conservative EV first), then
    one-sided longshot price outliers that could not be given a probability."""
    by_selection: dict[tuple, dict[str, dict[str, Quote]]] = defaultdict(lambda: defaultdict(dict))
    for q in quotes:
        by_selection[q.selection + (q.observed_at,)][q.book][q.side] = q
    sharp = tuple(settings.sharp_books)
    bettable = set(settings.bet_books)       # other books still serve as references, never as places to bet
    found: list[Opportunity] = []
    for books in by_selection.values():
        for book, sides in books.items():
            if book in sharp or (bettable and book not in bettable):
                continue
            for side, q in sides.items():
                minutes = (q.commence_time - q.observed_at).total_seconds() / 60
                if minutes <= 0:
                    continue            # the game had started: not a pre-game price
                rivals = [b[side].decimal for name, b in books.items()
                          if name not in sharp and name != book and side in b and (not bettable or name in bettable)]
                is_best = all(q.decimal >= r - 1e-9 for r in rivals)
                premium = q.decimal / statistics.median(rivals) - 1 if rivals else None
                longshot = q.price >= settings.longshot_min_american
                base = dict(sport=q.sport, event_id=q.event_id, event_name=q.event_name, commence_time=q.commence_time,
                            book=book, market=q.market, player=q.player, player_key=q.player_key, line=q.line, side=side,
                            price=q.price, decimal=q.decimal, observed_at=q.observed_at, last_update=q.last_update,
                            link=q.link, break_even=1 / q.decimal, is_longshot=longshot, is_best_price=is_best,
                            books_quoting=len(rivals) + 1, premium_vs_median=premium, minutes_to_start=minutes)
                ref = consensus_for(q, books, method=settings.devig_method, weights=settings.reference_weights,
                                    sharp_books=sharp, max_skew_seconds=settings.max_book_skew_seconds,
                                    uncertainty_floor=settings.uncertainty_floor,
                                    soft_only_multiplier=settings.soft_only_multiplier)
                if ref is not None and ref.n_books >= settings.min_reference_books:
                    ev = odds_math.expected_roi(ref.probability, q.decimal)
                    if ev <= 0:
                        continue
                    low = odds_math.conservative_probability(ref.probability, ref.standard_error, settings.confidence_z)
                    conservative = odds_math.expected_roi(low, q.decimal)
                    fair = odds_math.fair_american(ref.probability)
                    names = ", ".join(book_label(b) for b in ref.books)
                    text = (f"{book_label(book)} pays {q.price:+d} on {q.player} {describe(side, q.line)} "
                            f"({market_label(q.market)}). The same bet is worth {fair:+d} going by {names} with their margin "
                            f"removed. Estimated chance {ref.probability:.1%} (give or take {ref.standard_error:.1%}) against "
                            f"a break-even of {1 / q.decimal:.1%}.")
                    if len(rivals) and is_best:
                        text += f" Best price of {len(rivals) + 1} books."
                    if not ref.has_sharp:
                        text += " No sharp book prices this line, so the reference is retail books only."
                    tier = evidence_tier(ev, conservative, ref.has_sharp, ref.n_books)
                    if ev > settings.max_plausible_ev:
                        tier = "speculative"
                        text += (f" An edge above {settings.max_plausible_ev:.0%} usually means a stale price or news the "
                                 "other books have already acted on, so this is treated as speculative.")
                    found.append(Opportunity(**base, status="priced", source=SOURCE_MARKET, probability=ref.probability,
                                             standard_error=ref.standard_error, fair_price=fair, ev=ev,
                                             conservative_ev=conservative, reference_books=ref.books, has_sharp=ref.has_sharp,
                                             evidence=tier, explanation=text))
                elif longshot and is_best and premium is not None and premium >= settings.outlier_min_premium:
                    two_sided = any(len(b) == 2 for name, b in books.items() if name != book)
                    why = ("No other book prices both sides close enough in time" if two_sided
                           else "No book prices the other side of this bet")
                    text = (f"{book_label(book)} pays {q.price:+d} on {q.player} {describe(side, q.line)} "
                            f"({market_label(q.market)}), {premium:.0%} more than the middle price at {len(rivals)} other "
                            f"book{'s' if len(rivals) != 1 else ''}. {why}, so there is no fair probability and no EV: "
                            "insufficient data.")
                    found.append(Opportunity(**base, status="insufficient_data", source=None, probability=None,
                                             standard_error=None, fair_price=None, ev=None, conservative_ev=None,
                                             reference_books=(), has_sharp=False, evidence="insufficient", explanation=text))
    return rank(found)


def current(found: list[Opportunity], latest_seen: dict[str, object]) -> list[Opportunity]:
    """Only what the newest look at each game flagged ({event_id: observed_at}). A bet flagged in an earlier
    snapshot and not in the newest one is gone, not a second opportunity."""
    return rank([o for o in found if o.observed_at == latest_seen.get(o.event_id)])


def rank(found: list[Opportunity]) -> list[Opportunity]:
    """Priced bets by conservative EV, best first; unpriced outliers last, biggest price premium first."""
    priced = sorted((o for o in found if o.status == "priced"), key=lambda o: (-o.conservative_ev, -o.ev))
    rest = sorted((o for o in found if o.status != "priced"), key=lambda o: -(o.premium_vs_median or 0))
    return priced + rest
