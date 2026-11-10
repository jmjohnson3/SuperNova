"""Sports Betting Sniper dashboard (Streamlit). Research and paper trading only.

    streamlit run src/sports_sniper/dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from sports_sniper import db, odds_math
from sports_sniper.config import get_settings
from sports_sniper.markets import book_label, market_label
from sports_sniper.movement import moves
from sports_sniper.opportunities import EVIDENCE_ORDER, describe
from sports_sniper.scanner import run_scan

st.set_page_config(page_title="Sports Betting Sniper", page_icon="🎯", layout="wide")
settings = get_settings()
SOURCE_NOTE = {
    "sample": ("warning", "**Hypothetical sample data.** These teams, players and prices are invented so the app works "
                          "offline. Nothing on this page is a real opportunity."),
    "saved": ("info", "**Real prices saved by the scheduled scanners** (no API credits used). They are captured in the "
                      "hour before each game, so outside that window everything here is history."),
    "live": ("info", "**Live prices from The Odds API.**"),
}


@st.cache_resource
def engine():
    eng = db.make_engine(settings.database_url)
    db.init_db(eng)
    return eng


def american(price) -> str:
    return "-" if pd.isna(price) else f"{int(price):+d}"


def pct(value, signed: bool = False) -> str:
    return "-" if pd.isna(value) else (f"{value:+.1%}" if signed else f"{value:.1%}")


def ago(when, now) -> str:
    minutes = (now - when).total_seconds() / 60
    return "just now" if minutes < 1 else f"{minutes:.0f} min ago" if minutes < 90 else f"{minutes / 60:.1f} h ago"


def local(when) -> str:
    return pd.Timestamp(when).to_pydatetime().astimezone().strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def label(o) -> str:
    return f"{o.player} — {describe(o.side, None if pd.isna(o.line) else o.line)} {market_label(o.market)}"


def scan_and_reload(source: str) -> None:
    with st.spinner("Scanning..."):
        result = run_scan(settings, engine(), source)
    st.session_state["scan_message"] = (f"Scan {result.scan_id}: {result.events} games, {result.quotes} prices, "
                                        f"{len(result.current)} flagged." + (f" {result.note}." if result.note else ""))
    st.rerun()


now = datetime.now(timezone.utc)
with Session(engine()) as session:
    scan = db.latest_scan(session)

# --- sidebar ---------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Scan")
    if st.button("Scan saved prices (free)", width="stretch", help="Real snapshots the scheduled jobs already saved. "
                                                                   "Falls back to sample data when there are none."):
        scan_and_reload("auto")
    if st.button("Scan sample data", width="stretch"):
        scan_and_reload("sample")
    spend = st.checkbox("Allow a live scan (spends Odds API credits)")
    if st.button("Scan live now", width="stretch", disabled=not spend,
                 help=f"One credit per market per game, at most {settings.max_credits_per_scan} per scan; "
                      f"stops at {settings.credit_floor} credits remaining."):
        scan_and_reload("live")
    if "scan_message" in st.session_state:
        st.caption(st.session_state["scan_message"])

st.title("🎯 Sports Betting Sniper")
st.caption("Research and paper trading only. An estimated edge is an estimate, never a guaranteed profit, "
           "and nothing here places a bet.")
if scan is None:
    st.info("No scans yet. Use **Scan saved prices** or **Scan sample data** in the sidebar.")
    st.stop()

kind, text = SOURCE_NOTE[scan.source]
getattr(st, kind)(text)
frame = db.load_opportunities(engine(), since=now - timedelta(hours=36))
frame = frame[frame.data_source == scan.source] if not frame.empty else frame
if not frame.empty:
    age = (now - frame.observed_at).dt.total_seconds() / 60
    frame["available"] = ~frame.superseded & (frame.commence_time > now) & (age <= settings.max_snapshot_age_minutes)
else:
    frame["available"] = pd.Series(dtype=bool)

with st.sidebar:
    st.header("Filters")
    sports = st.multiselect("Sport", sorted(frame.sport.unique()), format_func=str.upper)
    markets = st.multiselect("Market", sorted(frame.market.unique()), format_func=market_label)
    books = st.multiselect("Sportsbook", sorted(frame.book.unique()), format_func=book_label)
    min_odds = st.number_input("Minimum odds (American)", value=-10000, step=50)
    min_edge = st.slider("Minimum conservative EV", 0.0, 0.15, float(settings.min_conservative_ev), 0.005, format="%.3f")
    tiers = st.multiselect("Evidence", ["strong", "moderate", "speculative"], default=["strong", "moderate"],
                           help="Strong: holds after the uncertainty haircut, with a sharp book and a second reference. "
                                "Moderate: holds with a sharp book or three retail books. Speculative: thin references, "
                                "or positive only before the haircut.")
    best_only = st.checkbox("Best price only", value=True, help="Hide a bet at a book when another book pays more.")
    show_expired = st.checkbox("Include prices no longer available", value=False)


def filtered(data: pd.DataFrame) -> pd.DataFrame:
    if sports:
        data = data[data.sport.isin(sports)]
    if markets:
        data = data[data.market.isin(markets)]
    if books:
        data = data[data.book.isin(books)]
    data = data[data.price >= min_odds]
    if best_only:
        data = data[data.is_best_price]
    return data if show_expired else data[data.available]


def table(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Player / market": [label(o) for o in data.itertuples()],
        "Game": [f"{o.event_name} ({local(o.commence_time)})" for o in data.itertuples()],
        "Book": data.book.map(book_label).to_list(),
        "Odds": data.price.map(american).to_list(),
        "Fair odds": data.fair_price.map(american).to_list(),
        "Est. chance": data.probability.map(pct).to_list(),
        "Break-even": data.break_even.map(pct).to_list(),
        "Est. EV": data.ev.map(lambda v: pct(v, True)).to_list(),
        "Conservative EV": data.conservative_ev.map(lambda v: pct(v, True)).to_list(),
        "Evidence": data.evidence.str.capitalize().to_list(),
        "Estimate from": ["Market consensus" if s == "market_consensus" else "-" for s in data.source],
        "Price seen": [("" if o.available else "EXPIRED · ") + ago(o.observed_at, now) for o in data.itertuples()],
    })


def details(data: pd.DataFrame, key: str) -> None:
    """Why one row was flagged, and a button to save it as a paper bet."""
    choice = st.selectbox("Explain and save", data.index, key=key,
                          format_func=lambda i: f"{label(data.loc[i])} at {book_label(data.loc[i].book)} {american(data.loc[i].price)}")
    row = data.loc[choice]
    st.write(row.explanation)
    if not row.available:
        st.caption(f"This price was seen {ago(row.observed_at, now)} and may be gone.")
    if pd.notna(row.link) and row.link and scan.source != "sample":
        st.caption(f"[Open at {book_label(row.book)}]({row.link})")
    stake = st.number_input("Hypothetical stake", min_value=0.01, value=float(settings.paper_stake), key=f"{key}_stake")
    if st.button("Save as paper bet", key=f"{key}_save"):
        with Session(engine()) as session, session.begin():
            bet, created = db.place_paper_bet(session, session.get(db.OpportunityRow, int(row.id)), stake=stake)
            message = f"paper bet {bet.id} at {book_label(bet.book)} {american(bet.price)}"
        (st.success if created else st.info)(("Saved " if created else "Already saved as ") + message + " (simulated).")


priced = frame[frame.status == "priced"].sort_values(["conservative_ev", "ev"], ascending=False)
unpriced = frame[frame.status != "priced"].sort_values("premium_vs_median", ascending=False)
last_seen = frame.observed_at.max() if not frame.empty else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last prices seen", "-" if last_seen is None else ago(last_seen, now))
c2.metric("Available now", int((priced.available & (priced.conservative_ev > min_edge)).sum()))
c3.metric("Flagged in the last 36 h", len(priced))
c4.metric("Last scan", f"#{scan.id} · {scan.source}")

board_tab, longshot_tab, moves_tab, paper_tab = st.tabs(["Opportunity Board", "Longshot Hunter", "Line Movement", "Paper Bets"])

with board_tab:
    st.subheader("Today's opportunities")
    board = filtered(priced)
    board = board[(board.conservative_ev > min_edge) & board.evidence.isin(tiers)]
    if board.empty:
        st.info("No opportunities clear the bar right now. That is a normal result: most prices are fair or worse."
                + ("" if show_expired or priced.empty else " Tick **Include prices no longer available** to review earlier ones."))
    else:
        st.dataframe(table(board), hide_index=True, width="stretch")
        st.caption("Ranked by conservative EV: the estimated chance lowered by its uncertainty before the EV is worked out. "
                   "Every estimate on this page comes from other sportsbooks' prices for the identical bet, with their "
                   "margin removed; no prediction model is involved yet.")
        details(board, "board")

with longshot_tab:
    st.subheader("Longshot Hunter")
    floor = st.number_input("Minimum payout odds (American)", value=int(settings.longshot_min_american), step=50)
    longs = filtered(priced)
    longs = longs[(longs.price >= floor) & longs.evidence.isin(tiers) & (longs.conservative_ev > min_edge)]
    if longs.empty:
        st.info("No longshot has a supportable positive edge right now.")
    else:
        st.dataframe(table(longs), hide_index=True, width="stretch")
        details(longs, "longshot")
    outliers = filtered(unpriced)
    outliers = outliers[outliers.price >= floor]
    st.markdown("#### Insufficient data")
    st.caption("High payouts priced well above the other books, where no book prices the other side of the bet. Without "
               "both sides there is no fair probability, so these get no EV and are not recommendations.")
    if outliers.empty:
        st.write("None.")
    else:
        st.dataframe(pd.DataFrame({
            "Player / market": [label(o) for o in outliers.itertuples()],
            "Game": outliers.event_name.to_list(), "Book": outliers.book.map(book_label).to_list(),
            "Odds": outliers.price.map(american).to_list(), "Break-even": outliers.break_even.map(pct).to_list(),
            "Above the middle price": outliers.premium_vs_median.map(lambda v: pct(v, True)).to_list(),
            "Books quoting": outliers.books_quoting.to_list(), "Status": "Insufficient data",
            "Price seen": [("" if o.available else "EXPIRED · ") + ago(o.observed_at, now) for o in outliers.itertuples()],
        }), hide_index=True, width="stretch")

with moves_tab:
    st.subheader("Line movement")
    history = db.load_quotes(engine(), since=now - timedelta(hours=36))
    with Session(engine()) as session:
        own = set(session.scalars(select(db.Scan.id).where(db.Scan.source == scan.source)))
    history = history[history.scan_id.isin(own)] if not history.empty else history
    changed = moves(history, settings.line_move_threshold)
    if changed.empty:
        st.info("No price has moved by a meaningful amount between the stored snapshots.")
    else:
        if not show_expired:
            changed = changed[changed.commence_time > now]
        st.dataframe(pd.DataFrame({
            "Player / market": [label(m) for m in changed.itertuples()],
            "Game": changed.event_name.to_list(), "Book": changed.book.map(book_label).to_list(),
            "First seen": changed.first_price.map(american).to_list(), "Latest": changed.last_price.map(american).to_list(),
            "Break-even change": changed.break_even_change.map(lambda v: pct(v, True)).to_list(),
            "Best latest price": [f"{american(m.best_now)} at {book_label(m.best_book)}" for m in changed.itertuples()],
            "Latest seen": [("" if now - m.last_seen <= timedelta(minutes=settings.max_snapshot_age_minutes) else "STALE · ")
                            + ago(m.last_seen, now) for m in changed.itertuples()],
        }), hide_index=True, width="stretch")
        st.caption("First seen is the first price this app stored, not the book's opening price. A negative break-even "
                   "change means the payout got longer. Closing prices are not captured yet.")

with paper_tab:
    st.subheader("Paper bets")
    st.caption("Simulated entries at prices that were actually observed. No real money is involved.")
    bets = db.load_paper_bets(engine())
    if bets.empty:
        st.info("No paper bets yet. Save one from the Opportunity Board.")
    else:
        settled = bets[bets.status.isin(["won", "lost", "push"])]
        a, b, c, d = st.columns(4)
        a.metric("Paper bets", len(bets))
        b.metric("Open", int((bets.status == "open").sum()))
        c.metric("Record (W-L-P)", f"{(bets.status == 'won').sum()}-{(bets.status == 'lost').sum()}-{(bets.status == 'push').sum()}")
        staked = settled.stake.sum()
        d.metric("Simulated profit", "-" if settled.empty else f"{settled.profit.sum():+.2f} ({settled.profit.sum() / staked:+.1%})")
        if not settled.empty:
            low, high = odds_math.wilson_interval(int((settled.status == "won").sum()), int(settled.status.isin(["won", "lost"]).sum()))
            st.caption(f"Win rate 95% interval: {low:.0%} to {high:.0%}.")
        st.dataframe(pd.DataFrame({
            "Saved": bets.created_at.map(local).to_list(),
            "Bet": [label(b) for b in bets.itertuples()], "Game": bets.event_name.to_list(),
            "Book": bets.book.map(book_label).to_list(), "Odds": bets.price.map(american).to_list(),
            "Est. chance": bets.probability.map(pct).to_list(), "Conservative EV": bets.conservative_ev.map(lambda v: pct(v, True)).to_list(),
            "Stake": bets.stake.to_list(), "Status": bets.status.str.capitalize().to_list(),
            "Data": ["Hypothetical sample" if s == "sample" else "Real prices" for s in bets.data_source],
            "Strategy": bets.strategy.to_list(),
        }), hide_index=True, width="stretch")
        st.caption("Automatic settlement from final statistics is not built yet, so bets stay open.")
