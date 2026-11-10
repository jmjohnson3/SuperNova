"""Settle paper bets from final statistics, and measure each one against the closing price.

A bet is only settled when its game is known to be complete. A player with no line in a complete box score is
a void (sportsbooks refund a prop on a player who does not play), but only when his name is known on one of
the two teams; an unrecognised name stays open rather than being guessed.

Closing line value here is the edge of the price that was taken, measured against the fair probability at the
last look before the game: the other books' no-vig price for the identical bet, the same method that flagged it.
Positive means the market moved toward the bet.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from sports_sniper import db, odds_math
from sports_sniper.config import Settings
from sports_sniper.consensus import consensus_for
from sports_sniper.normalize import Quote
from sports_sniper.providers import results

# market -> the statistics added together to settle it. Alternate lines settle on the same statistic.
STATS = {
    "nba": {"player_points": ("pts",), "player_rebounds": ("reb",), "player_assists": ("ast",), "player_threes": ("fg3m",),
            "player_points_rebounds_assists": ("pts", "reb", "ast")},
    "nfl": {"player_reception_yds": ("receiving_yards",), "player_rush_yds": ("rushing_yards",),
            "player_receptions": ("receptions",), "player_pass_tds": ("passing_tds",), "player_pass_yds": ("passing_yards",)},
}
OPEN = "open"
CLOSE_RETRY_HOURS = 24      # how long after the start a settled bet is still checked for a closing price


def stat_names(sport: str, market: str) -> tuple[str, ...] | None:
    return STATS.get(sport, {}).get(market.removesuffix("_alternate"))


def grade(side: str, line: float | None, value: float) -> str | None:
    """won | lost | push for an over/under at a line; None for a side this version cannot settle."""
    if side not in ("over", "under") or line is None:
        return None
    if value == line:
        return "push"
    return "won" if (value > line) == (side == "over") else "lost"


def profit(status: str, stake: float, decimal: float) -> float:
    return {"won": stake * (decimal - 1), "lost": -stake}.get(status, 0.0)


def decide(bet, game: results.GameResult | None) -> tuple[str, float | None, str]:
    """(status, statistic, note) for one bet given what is known about its game."""
    names = stat_names(bet.sport, bet.market)
    if names is None or bet.side not in ("over", "under"):
        return OPEN, None, "this market cannot be settled yet"
    if game is None or not game.complete:
        return OPEN, None, game.reason if game else "no results source available"
    if bet.player_key in game.ambiguous:
        return OPEN, None, "two players in this game share the name"
    line = game.stats.get(bet.player_key)
    if line is None:
        if bet.player_key in game.known:
            return "void", None, "did not play"
        return OPEN, None, "player not found in the results"
    value = sum(line[n] for n in names)
    return grade(bet.side, bet.line, value), value, "final"


def closing_value(bet, prices: list, settings: Settings) -> dict | None:
    """The closing look at a bet from stored prices (rows for its game and market), or None when there is none.

    The close is the last snapshot before the game starts, inside the closing window and later than the price the
    bet was taken at. Returns the fair probability then (other books, margin removed), the entry price's edge
    against it, and the bet's own book's price then when it still quoted the same line.
    """
    start = db.as_utc(bet.commence_time)
    taken = db.as_utc(bet.price_observed_at)
    window = start - timedelta(minutes=settings.closing_window_minutes)
    seen = [db.as_utc(p.observed_at) for p in prices]
    late = [t for t in seen if window <= t < start and t > taken]
    if not late:
        return None
    close = max(late)
    books: dict[str, dict[str, Quote]] = defaultdict(dict)
    for p, at in zip(prices, seen):
        if at == close and p.player_key == bet.player_key and p.line_key == bet.line_key:
            books[p.book][p.side] = Quote(sport=p.sport, event_id=p.event_id, event_name=p.event_name, commence_time=start,
                                          book=p.book, market=p.market, player=p.player, player_key=p.player_key, line=p.line,
                                          side=p.side, price=p.price, decimal=odds_math.american_to_decimal(p.price),
                                          last_update=db.as_utc(p.last_update), observed_at=at)
    own = books.get(bet.book, {}).get(bet.side)
    probe = own or Quote(sport=bet.sport, event_id=bet.event_id, event_name=bet.event_name, commence_time=start, book=bet.book,
                         market=bet.market, player=bet.player, player_key=bet.player_key, line=bet.line, side=bet.side,
                         price=bet.price, decimal=bet.decimal, last_update=close, observed_at=close)
    ref = consensus_for(probe, books, method=settings.devig_method, weights=settings.reference_weights,
                        sharp_books=tuple(settings.sharp_books), uncertainty_floor=settings.uncertainty_floor,
                        soft_only_multiplier=settings.soft_only_multiplier,
                        max_skew_seconds=settings.max_book_skew_seconds if own else settings.max_quote_age_seconds)
    if ref is None and own is None:
        return None
    return dict(observed_at=close, price=None if own is None else own.price,
                probability=None if ref is None else ref.probability,
                value=None if ref is None else odds_math.expected_roi(ref.probability, bet.decimal))


def settle(engine: Engine, settings: Settings, *, now: datetime | None = None, load_results=None) -> dict:
    """Fill in closing values and settle every open paper bet whose game has started. Returns counts."""
    now = now or datetime.now(timezone.utc)
    load_results = load_results or results.load
    db.init_db(engine)
    counts: Counter = Counter()
    with Session(engine) as session, session.begin():
        started = [b for b in session.scalars(select(db.PaperBet).where(db.PaperBet.data_source != "sample"))
                   if db.as_utc(b.commence_time) <= now]
        recent = now - timedelta(hours=CLOSE_RETRY_HOURS)
        for bet in started:
            # No price is stored after a game starts, so a settled bet's close is looked for once, not every pass.
            if bet.closing_observed_at is None and (bet.status == OPEN or db.as_utc(bet.commence_time) >= recent):
                prices = session.scalars(select(db.PriceRow).where(db.PriceRow.event_id == bet.event_id,
                                                                   db.PriceRow.market == bet.market,
                                                                   db.PriceRow.player_key == bet.player_key)).all()
                close = closing_value(bet, prices, settings)
                if close is not None:
                    bet.closing_observed_at, bet.closing_price = close["observed_at"], close["price"]
                    bet.closing_probability, bet.closing_line_value = close["probability"], close["value"]
                    counts["closing_values"] += 1
        still_open = [b for b in started if b.status == OPEN]
        games = {b.event_id: results.Game(b.sport, b.event_id, *b.event_name.split(" @ ")[::-1], db.as_utc(b.commence_time))
                 for b in still_open if " @ " in b.event_name}
        try:
            found = load_results(settings.results_dsn, list(games.values())) if games else {}
        except Exception as error:               # the results database is optional
            found, counts["results_unavailable"] = {}, 1
            note = f"results unavailable ({type(error).__name__})"
            for bet in still_open:
                bet.settle_note = note
        else:
            for bet in still_open:
                status, value, note = decide(bet, found.get(bet.event_id))
                bet.settle_note = note
                if status != OPEN:
                    bet.status, bet.result_value, bet.settled_at = status, value, now
                    bet.profit = profit(status, bet.stake, bet.decimal)
                counts[status] += 1
        counts["checked"] = len(still_open)
    return dict(counts)
