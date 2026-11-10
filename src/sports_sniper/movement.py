"""Line movement from the stored price history: first observed price against the latest one."""
from __future__ import annotations

import pandas as pd

from sports_sniper import odds_math

COLUMNS = ["sport", "event_name", "commence_time", "book", "market", "player", "side", "line", "first_price", "last_price",
           "first_seen", "last_seen", "observations", "break_even_change", "best_now", "best_book"]


def moves(quotes: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """One row per book and bet (same line) seen at least twice with a changed price.

    `break_even_change` is the change in the price's break-even probability: negative means the price got
    longer (better for the bettor), positive means it shortened. `best_now` is the best latest price for that
    bet at any book in the history, and which book has it.
    """
    if quotes.empty:
        return pd.DataFrame(columns=COLUMNS)
    key = ["sport", "event_id", "book", "market", "player_key", "line_key", "side"]
    ordered = quotes.sort_values("last_update")
    grouped = ordered.groupby(key, sort=False)
    out = grouped.agg(event_name=("event_name", "last"), commence_time=("commence_time", "last"), player=("player", "last"),
                      line=("line", "last"), first_price=("price", "first"), last_price=("price", "last"),
                      first_seen=("last_update", "first"), last_seen=("last_update", "last"),
                      observations=("price", "size")).reset_index()
    out["break_even_change"] = [odds_math.implied_probability(b) - odds_math.implied_probability(a)
                                for a, b in zip(out.first_price, out.last_price)]
    bet = ["sport", "event_id", "market", "player_key", "line_key", "side"]
    best = out.sort_values("last_price", key=lambda s: s.map(odds_math.american_to_decimal)).groupby(bet, sort=False).tail(1)
    out = out.merge(best[bet + ["last_price", "book"]].rename(columns={"last_price": "best_now", "book": "best_book"}), on=bet)
    changed = out[(out.observations > 1) & (out.first_price != out.last_price) & (out.break_even_change.abs() >= threshold)]
    return changed.sort_values("break_even_change", key=lambda s: -s.abs())[COLUMNS].reset_index(drop=True)
