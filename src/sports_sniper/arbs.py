"""Arbitrages and middles between sportsbooks, from the prop prices the dashboard already stores. Information only.

Both are an over at one book and an under at another on the same player and statistic, staked so the payout is the
same whichever one wins:

    arbitrage   the two lines are the same and the prices return more than was staked whatever happens
    middle      the under's line is higher than the over's: a result between the lines wins both bets. A free middle
                cannot lose; a paid middle loses a little when the result lands outside the lines.

Nothing here estimates a result. The "books' own chance" of a middle is only the difference between the two books'
no-vig prices, a rough guide. Prices are the last stored snapshot and may have moved; nothing places a bet.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sports_sniper import odds_math

SAME_SNAPSHOT = timedelta(minutes=2)
MAX_QUOTE_AGE = timedelta(minutes=5)          # a book's price last changed this long before the snapshot at most
EXCLUDED_BOOKS = ("pinnacle",)                    # a reference price, not a book the user can bet at
ALTERNATE = "_alternate"
KEY = ["event_id", "stat", "player_key"]
COLUMNS = ["kind", "sport", "event_id", "event_name", "commence_time", "observed_at", "stat", "player", "player_key", "over_book", "over_line",
           "over_price", "over_link", "under_book", "under_line", "under_price", "under_link", "width", "over_share", "miss_return",
           "hit_return", "break_even", "books_chance"]


def current(prices: pd.DataFrame, now: datetime, max_age: timedelta, upcoming_only: bool = True) -> pd.DataFrame:
    """Each game's most recent snapshot taken before its start, when that snapshot is no older than max_age."""
    if prices.empty:
        return prices
    p = prices[(prices.observed_at < prices.commence_time) & (prices.observed_at - prices.last_update <= MAX_QUOTE_AGE)]
    if upcoming_only:
        p = p[p.commence_time > now]
    latest = p.groupby("event_id").observed_at.transform("max")
    p = p[(p.observed_at >= latest - SAME_SNAPSHOT) & (latest >= now - max_age)]
    p = p.sort_values("observed_at").drop_duplicates(["event_id", "book", "market", "player_key", "line_key", "side"], keep="last")
    return p.assign(stat=p.market.str.replace(ALTERNATE, "", regex=False)).reset_index(drop=True)


def no_vig_over(prices: pd.DataFrame) -> pd.DataFrame:
    """A book's own no-vig chance of the over where it prices both sides of a line."""
    cols = ["event_id", "market", "player_key", "book", "line"]
    over, under = (prices[prices.side == s][cols + ["price"]] for s in ("over", "under"))
    both = over.merge(under, on=cols, suffixes=("_over", "_under"))
    a, b = (both[c].map(odds_math.implied_probability) for c in ("price_over", "price_under"))
    return both.assign(p_over=a / (a + b))[["event_id", "player_key", "book", "line", "p_over"]].drop_duplicates(["event_id", "player_key", "book", "line"])


def find(prices: pd.DataFrame, books: list[str] | None = None, max_cost: float = 0.05) -> pd.DataFrame:
    """Every over and under at two different books that is an arbitrage or a middle costing at most max_cost."""
    if prices.empty:
        return pd.DataFrame(columns=COLUMNS)
    p = prices[~prices.book.isin(EXCLUDED_BOOKS) & prices.line.notna() & prices.side.isin(["over", "under"])]
    if books is not None:
        p = p[p.book.isin(books)]
    legs = KEY + ["book", "line", "price", "link", "market"]
    over = p[p.side == "over"][legs + ["sport", "event_name", "commence_time", "observed_at", "player"]]
    pairs = over.merge(p[p.side == "under"][legs], on=KEY, suffixes=("_over", "_under"))
    pairs = pairs[(pairs.book_over != pairs.book_under) & (pairs.line_under >= pairs.line_over)]
    if pairs.empty:
        return pd.DataFrame(columns=COLUMNS)
    over_dec, under_dec = (pairs[c].map(odds_math.american_to_decimal) for c in ("price_over", "price_under"))
    pairs = pairs.assign(over_share=under_dec / (over_dec + under_dec),                     # of the total staked, so either win pays the same
                         miss_return=over_dec * under_dec / (over_dec + under_dec) - 1.0, width=pairs.line_under - pairs.line_over)
    pairs["hit_return"] = np.where(pairs.width > 0, 2.0 * (pairs.miss_return + 1.0) - 1.0, np.nan)
    pairs["kind"] = np.select([(pairs.width == 0) & (pairs.miss_return > 0), (pairs.width > 0) & (pairs.miss_return >= 0), pairs.width > 0],
                              ["Arbitrage", "Free middle", "Middle"], "")
    pairs = pairs[(pairs.kind != "") & (pairs.miss_return >= -max_cost)]
    pairs["break_even"] = np.where(pairs.miss_return < 0, -pairs.miss_return / (pairs.hit_return - pairs.miss_return), 0.0)
    chance = no_vig_over(p)
    for side in ("over", "under"):
        pairs = pairs.merge(chance.rename(columns={"book": f"book_{side}", "line": f"line_{side}", "p_over": f"p_{side}"}),
                            on=["event_id", "player_key", f"book_{side}", f"line_{side}"], how="left")
    pairs["books_chance"] = np.where(pairs.width > 0, (pairs.p_over - pairs.p_under).where(pairs.p_over > pairs.p_under), np.nan)
    pairs = pairs.sort_values("miss_return", ascending=False).drop_duplicates(KEY + ["line_over", "line_under"])
    pairs = pairs.rename(columns={f"{c}_{s}": f"{s}_{c}" for c in ("book", "line", "price", "link") for s in ("over", "under")})
    order = pd.Categorical(pairs.kind, ["Arbitrage", "Free middle", "Middle"], ordered=True)
    return pairs.assign(_order=order).sort_values(["_order", "miss_return"], ascending=[True, False])[COLUMNS].reset_index(drop=True)


def stakes(row, total: float = 100.0) -> tuple[float, float]:
    """How a total stake is split between the over and the under so that either bet winning pays the same."""
    return round(total * row.over_share, 2), round(total * (1.0 - row.over_share), 2)
