"""The Arbitrage & Middles tab. Display only: it reads stored prices, flags nothing and places nothing."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from sports_sniper import arbs, db, phone_view
from sports_sniper.markets import book_label, market_label

DEFAULT_BOOKS = ["fanduel", "draftkings"]
SHOWN = 12
TOTAL = 100.0


def american(price) -> str:
    return f"{int(price):+d}"


def number(line) -> str:
    return f"{line:g}"


def link(value, state: str) -> str | None:
    return None if value is None or pd.isna(value) or not value else str(value).replace("{state}", state.lower())


def describe(row) -> str:
    over, under = arbs.stakes(row, TOTAL)
    text = (f"**{row.player}** · {market_label(row.stat)} · {row.event_name}  \n"
            f"Over {number(row.over_line)} at {book_label(row.over_book)} **{american(row.over_price)}** (${over:.2f}) and "
            f"under {number(row.under_line)} at {book_label(row.under_book)} **{american(row.under_price)}** (${under:.2f})  \n")
    if row.kind == "Arbitrage":
        return text + f"Returns **{row.miss_return:+.2%}** of the ${TOTAL:.0f} staked whatever happens."
    middle = f"lands from {number(row.over_line)} to {number(row.under_line)}"
    chance = "" if pd.isna(row.books_chance) else f" The books' own prices put that at about {row.books_chance:.0%}."
    if row.kind == "Free middle":
        return text + f"Returns {row.miss_return:+.2%} at worst and **{row.hit_return:+.0%}** if the result {middle}.{chance}"
    return text + (f"Costs **{-row.miss_return:.1%}** if it misses, returns **{row.hit_return:+.0%}** if the result {middle}. "
                   f"Pays over time only if that happens more than {row.break_even:.1%} of the time.{chance}")


def rows(found: pd.DataFrame, now, state: str, prefix: str) -> None:
    for i, row in enumerate(found.head(SHOWN).itertuples()):
        text, go_over, go_under = st.container(key=f"{prefix}_{i}").columns([6, 1.6, 1.6], vertical_alignment="center")
        minutes = (row.commence_time - now).total_seconds() / 60
        seen = (now - row.observed_at).total_seconds() / 60
        when = f"Starts in {minutes:.0f} min" if minutes > 0 else "Started"
        text.markdown(describe(row) + f"  \n{when} · prices seen {seen:.0f} min ago")
        for column, book, value in ((go_over, row.over_book, row.over_link), (go_under, row.under_book, row.under_link)):
            target = link(value, state)
            if target:
                column.link_button(f"Open {book_label(book)}", target)
    if len(found) > SHOWN:
        st.caption(f"Showing the first {SHOWN} of {len(found)}.")


def table(found: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "Type": r.kind, "Player": r.player, "Statistic": market_label(r.stat), "Game": r.event_name,
        "Over": f"{number(r.over_line)} at {book_label(r.over_book)} {american(r.over_price)}",
        "Under": f"{number(r.under_line)} at {book_label(r.under_book)} {american(r.under_price)}",
        "If it misses": f"{r.miss_return:+.2%}", "If it lands between": "-" if pd.isna(r.hit_return) else f"{r.hit_return:+.0%}",
        "Needed to pay": "-" if r.kind != "Middle" else f"{r.break_even:.1%}",
        "Books' own chance": "-" if pd.isna(r.books_chance) else f"{r.books_chance:.0%}"} for r in found.itertuples()])


@st.cache_data(ttl=55, show_spinner=False)
def stored(_engine, hours: float, stamp: str) -> pd.DataFrame:
    return db.load_prices(_engine, since=pd.Timestamp(stamp).to_pydatetime() - timedelta(hours=hours))


def render(engine, settings, now) -> None:
    st.subheader("Arbitrage & Middles")
    st.caption("An over at one book and an under at another on the same player. Read from the prices already stored, so it costs no "
               "credits; they are stored only in the last 40 minutes before an NBA game and the last hour before an NFL game. "
               "Information only: nothing here places a bet, and a price may have moved by the time you open the book.")
    stamp = pd.Timestamp(now).floor("min").isoformat()
    prices = stored(engine, 1.0, stamp)
    available = sorted(b for b in prices.book.unique() if b not in arbs.EXCLUDED_BOOKS) if not prices.empty else DEFAULT_BOOKS
    left, right = st.columns([3, 2])
    books = left.multiselect("Books you can bet at", available, default=[b for b in DEFAULT_BOOKS if b in available], format_func=book_label,
                             key="arb_books", help="Both bets must be at books in this list.")
    max_cost = right.slider("Most a middle may cost if it misses", 0.0, 10.0, 5.0, 0.5, format="%.1f%%", key="arb_cost") / 100
    if len(books) < 2:
        st.info("Choose at least two books: the two bets are placed at different books.")
        return
    live = arbs.find(arbs.current(prices, now, timedelta(minutes=settings.fresh_minutes)), books, max_cost)
    riskless = live[live.kind != "Middle"]
    st.markdown(f"#### Cannot lose, seen in the last {settings.fresh_minutes:.0f} minutes ({len(riskless)})")
    if riskless.empty:
        st.caption("None right now.")
    else:
        rows(riskless, now, settings.link_state, "arb_safe")
    middles = live[live.kind == "Middle"].sort_values("break_even")
    st.markdown(f"#### Middles, seen in the last {settings.fresh_minutes:.0f} minutes ({len(middles)})")
    if middles.empty:
        st.caption("None right now within the cost you set.")
    else:
        st.caption("Cheapest to try first. A middle is a small likely loss for a chance at winning both bets. Tested on 23 days of stored "
                   "prices (16 Nov 2026): FanDuel and DraftKings middles costing up to 5% landed 1.6% of the time against the 4.7% needed "
                   "and lost 2.9% of the money staked. Same-line arbitrages returned 1.4%.")
        rows(middles, now, settings.link_state, "arb_mid")
    if st.toggle("Show every game's last stored prices in the past 36 hours", key="arb_history"):
        earlier = arbs.find(arbs.current(stored(engine, 36.0, stamp), now, timedelta(hours=36), upcoming_only=False), books, max_cost)
        if earlier.empty:
            st.caption("Nothing in the stored prices.")
        else:
            st.caption(f"{(earlier.kind != 'Middle').sum()} that could not lose and {(earlier.kind == 'Middle').sum()} middles in each "
                       "game's final stored snapshot. Most of these games have started; this shows how often they turn up.")
            phone_view.table(table(earlier), sub=["Game"], keep=["Type", "Over", "Under", "If it misses", "If it lands between", "Needed to pay"],
                             link=None)
