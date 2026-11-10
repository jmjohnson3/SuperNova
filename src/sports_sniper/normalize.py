"""Turn one Odds API event into clean, comparable price quotes.

A quote is one side of one player market at one book. Two quotes are the same bet only when sport, event,
market, player and line all match; alternate lines are never merged with each other.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from sports_sniper import odds_math

OPPOSITE = {"over": "under", "under": "over", "yes": "no", "no": "yes"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass(frozen=True)
class Quote:
    sport: str
    event_id: str
    event_name: str
    commence_time: datetime
    book: str
    market: str
    player: str
    player_key: str
    line: float | None        # None for yes/no markets such as anytime touchdown
    side: str                 # over | under | yes | no
    price: int                # American
    decimal: float
    last_update: datetime
    observed_at: datetime
    link: str | None = None

    @property
    def selection(self) -> tuple:
        """What is being bet on, without the book or the side."""
        return (self.sport, self.event_id, self.market, self.player_key, self.line)


def player_key(name: str) -> str:
    """One key per player however a book spells him: accents, punctuation, case and Jr./III dropped."""
    plain = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    words = [w for w in re.sub(r"[^a-z0-9 ]", "", plain.replace("-", " ")).split() if w not in _SUFFIXES]
    return " ".join(words)


def parse_time(value) -> datetime | None:
    """A timestamp as UTC. Every time in this package is UTC: SQLite keeps the clock reading and drops the
    offset, so a time stored in any other zone would come back wrong."""
    if isinstance(value, str) and value:
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def normalize_event(event: dict, sport: str, observed_at: datetime, *, max_age_seconds: float | None = None,
                    ) -> tuple[list[Quote], Counter]:
    """(quotes, counts of what was kept and why the rest was dropped). Never raises on a malformed payload."""
    counts: Counter = Counter()
    observed_at = parse_time(observed_at)
    if isinstance(event, dict) and "bookmakers" not in event and isinstance(event.get("data"), dict):
        event = event["data"]          # historical payloads wrap the event
    commence = parse_time(event.get("commence_time")) if isinstance(event, dict) else None
    if not isinstance(event, dict) or not event.get("id") or commence is None:
        counts["invalid_event"] += 1
        return [], counts
    name = f"{event.get('away_team') or '?'} @ {event.get('home_team') or '?'}"
    found: dict[tuple, Quote] = {}
    for book in event.get("bookmakers") or []:
        if not isinstance(book, dict) or not book.get("key"):
            counts["invalid_book"] += 1
            continue
        for market in book.get("markets") or []:
            if not isinstance(market, dict) or not market.get("key"):
                counts["invalid_market"] += 1
                continue
            updated = parse_time(market.get("last_update")) or parse_time(book.get("last_update"))
            if updated is None or updated > observed_at:
                counts["no_timestamp"] += len(market.get("outcomes") or [])
                continue
            if max_age_seconds is not None and (observed_at - updated).total_seconds() > max_age_seconds:
                counts["stale"] += len(market.get("outcomes") or [])
                continue
            for outcome in market.get("outcomes") or []:
                side = str(outcome.get("name") or "").strip().lower()
                player = str(outcome.get("description") or "").strip()
                if side not in OPPOSITE or not player:
                    counts["not_player_market"] += 1    # game lines, or a shape this version does not read
                    continue
                key = player_key(player)
                try:
                    price = int(outcome["price"])
                    decimal = odds_math.american_to_decimal(price)
                    line = None if outcome.get("point") is None else float(outcome["point"])
                except (KeyError, TypeError, ValueError):
                    counts["invalid_price"] += 1
                    continue
                if not key or (side in ("over", "under") and line is None):
                    counts["invalid_price"] += 1
                    continue
                quote = Quote(sport=sport, event_id=str(event["id"]), event_name=name, commence_time=commence,
                              book=str(book["key"]), market=str(market["key"]), player=player, player_key=key, line=line,
                              side=side, price=price, decimal=decimal, last_update=updated, observed_at=observed_at,
                              link=outcome.get("link") or None)
                identity = (quote.book, quote.market, quote.player_key, quote.line, quote.side)
                if identity in found:
                    counts["duplicate"] += 1
                found[identity] = quote
    counts["kept"] = len(found)
    return list(found.values()), counts
