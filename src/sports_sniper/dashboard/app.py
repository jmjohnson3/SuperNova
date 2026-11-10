"""Sports Betting Sniper dashboard (Streamlit). Research and paper trading only.

    streamlit run src/sports_sniper/dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from sports_sniper import db, performance
from sports_sniper.config import get_settings
from sports_sniper.markets import book_label, market_label
from sports_sniper.movement import moves
from sports_sniper.opportunities import EVIDENCE_ORDER, describe
from sports_sniper.scanner import run_scan
from sports_sniper.settlement import settle

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
                                        f"{len(result.current)} flagged."
                                        + (f" {result.paper_bets} new paper bets." if result.paper_bets else "")
                                        + (f" {result.note}." if result.note else ""))
    st.rerun()


def settle_and_reload() -> None:
    with st.spinner("Settling..."):
        run_scan(settings, engine(), "saved")           # store the latest snapshots first, so closing prices exist
        counts = settle(engine(), settings)
    done = ", ".join(f"{counts.get(k, 0)} {k}" for k in ("won", "lost", "push", "void"))
    st.session_state["scan_message"] = (f"Settled: {done}; {counts.get('open', 0)} still open."
                                        + (" The results database could not be read." if counts.get("results_unavailable") else ""))
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
    if st.button("Settle paper bets", width="stretch", help="Grade bets whose games are final and add closing line value. "
                                                            "Reads saved results; uses no API credits."):
        settle_and_reload()
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

board_tab, longshot_tab, moves_tab, perf_tab, paper_tab = st.tabs(
    ["Opportunity Board", "Longshot Hunter", "Line Movement", "Performance", "Paper Bets"])

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
    history = db.load_prices(engine(), since=now - timedelta(hours=36))
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
                   "change means the payout got longer.")

STRATEGY_LABEL = {db.STRATEGY_AUTO: "Automatic: every flagged bet that clears the bar", db.STRATEGY_MANUAL: "Saved by hand"}
STATUS_LABEL = {"open": "Open", "won": "Won", "lost": "Lost", "push": "Push", "void": "Void"}


def money(value) -> str:
    return "-" if pd.isna(value) else f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def plus_minus(value, error) -> str:
    return "-" if value is None else pct(value, True) + ("" if error is None else f" ± {error:.1%}")


def results_table(rows: list[tuple[str, dict]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Group": name, "Bets": r["bets"], "Open": r["open"], "Record (W-L-P)": f"{r['won']}-{r['lost']}-{r['push']}", "Void": r["void"],
        "Win rate": "-" if r["win_rate"] is None else f"{r['win_rate']:.1%}",
        "Simulated profit": money(r["profit"]) if r["staked"] else "-", "ROI": plus_minus(r["roi"], r["roi_se"]),
        "Est. EV": "-" if r["estimated_ev"] is None else pct(r["estimated_ev"], True),
        "Closing line value": plus_minus(r["closing_value"], r["closing_se"]), "With a close": r["closing_count"],
    } for name, r in rows])


all_bets = db.load_paper_bets(engine())

with perf_tab:
    st.subheader("Performance")
    st.caption("Simulated results of paper bets at prices that were actually observed. No real money is involved, and a "
               "past paper result does not promise a future one.")
    real = all_bets[all_bets.data_source != "sample"] if not all_bets.empty else all_bets
    if real.empty:
        st.info("No paper bets on real prices yet. Scan saved prices: every flagged bet that clears the bar is entered "
                "automatically, then **Settle paper bets** grades it once its game is final.")
    else:
        names = sorted(real.strategy.unique())
        chosen = st.selectbox("Strategy", names, format_func=lambda n: STRATEGY_LABEL.get(n, n))
        with_replays = st.checkbox("Include replayed bets", value=True,
                                   help="A replayed bet was entered long after its price was seen, from saved snapshots. Its "
                                        "price was real, but the rule was applied after the fact; forward bets are the clean test.")
        view = real[real.strategy == chosen]
        view = view if with_replays else view[~view.replayed]
        total = performance.summarize(view)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Paper bets", total["bets"], help=f"{total['open']} open, {total['void']} void")
        m2.metric("Record (W-L-P)", f"{total['won']}-{total['lost']}-{total['push']}",
                  help=None if total["win_interval"] is None else
                  f"Win rate {total['win_rate']:.1%}; 95% interval {total['win_interval'][0]:.0%} to {total['win_interval'][1]:.0%}")
        m3.metric("Simulated profit", money(total["profit"]) if total["staked"] else "-", plus_minus(total["roi"], total["roi_se"]),
                  delta_color="off")
        m4.metric("Closing line value", plus_minus(total["closing_value"], total["closing_se"]),
                  None if total["beat_close"] is None else f"{total['beat_close']:.0%} beat the close", delta_color="off")
        st.caption("ROI is simulated profit over the stakes of settled bets; the ± is one standard error with whole games as "
                   "the draws. Closing line value is the edge of the price taken against the fair price at the last look before "
                   "the game (other books, margin removed). It settles down far faster than results do: a few hundred bets "
                   "can still swing 10% either way on luck alone.")
        rows = [("All", total), ("Forward (entered when seen)", performance.summarize(view[~view.replayed])),
                ("Replayed from saved snapshots", performance.summarize(view[view.replayed]))]
        st.dataframe(results_table([r for r in rows if r[1]["bets"]]), hide_index=True, width="stretch")
        for title, column, name in (("By sport", "sport", str.upper), ("By market", "market", market_label),
                                    ("By sportsbook", "book", book_label), ("By side", "side", str.capitalize),
                                    ("By evidence", "evidence", str.capitalize)):
            st.markdown(f"#### {title}")
            st.dataframe(results_table([(name(key), performance.summarize(group)) for key, group in view.groupby(column)]),
                         hide_index=True, width="stretch")
        st.markdown("#### Calibration")
        bands = performance.calibration(view)
        if total["won"] + total["lost"] < 30:
            st.write("Not enough settled bets to judge calibration yet (30 needed).")
        else:
            st.dataframe(pd.DataFrame([{"Estimated chance": f"{b['low']:.0%} to {b['high']:.0%}", "Bets": b["n"],
                                        "Average estimate": pct(b["predicted"]), "Actually won": pct(b["observed"]),
                                        "95% interval": f"{b['interval'][0]:.0%} to {b['interval'][1]:.0%}"} for b in bands]),
                         hide_index=True, width="stretch")
            st.caption(f"Brier score {total['brier']:.4f} for these estimates against {total['break_even_brier']:.4f} for the "
                       "prices' own break-even chance (lower is better). An estimate that cannot beat the price it was "
                       "compared with is not adding information.")

with paper_tab:
    st.subheader("Paper bets")
    st.caption("Simulated entries at prices that were actually observed. No real money is involved.")
    if all_bets.empty:
        st.info("No paper bets yet. Save one from the Opportunity Board, or scan saved prices to enter flagged bets automatically.")
    else:
        shown = all_bets.head(500)
        st.dataframe(pd.DataFrame({
            "Entered": shown.created_at.map(local).to_list(),
            "Bet": [label(b) for b in shown.itertuples()], "Game": shown.event_name.to_list(),
            "Book": shown.book.map(book_label).to_list(), "Odds": shown.price.map(american).to_list(),
            "Est. chance": shown.probability.map(pct).to_list(),
            "Conservative EV": shown.conservative_ev.map(lambda v: pct(v, True)).to_list(),
            "Stake": shown.stake.to_list(), "Status": shown.status.map(STATUS_LABEL).to_list(),
            "Result": ["-" if pd.isna(v) else f"{v:g}" for v in shown.result_value],
            "Profit": shown.profit.map(money).to_list(),
            "Closing price": shown.closing_price.map(american).to_list(),
            "Closing line value": shown.closing_line_value.map(lambda v: pct(v, True)).to_list(),
            "Note": shown.settle_note.fillna("").to_list(),
            "Data": ["Hypothetical sample" if d == "sample" else "Replayed" if r else "Real prices"
                     for d, r in zip(shown.data_source, shown.replayed)],
            "Strategy": shown.strategy.map(lambda n: "Automatic" if n == db.STRATEGY_AUTO else "By hand").to_list(),
        }), hide_index=True, width="stretch")
        if len(all_bets) > len(shown):
            st.caption(f"Showing the newest {len(shown)} of {len(all_bets)}.")
        st.caption("Bets on hypothetical sample data are never settled. A bet stays open until its game is final and complete "
                   "in the results source; the Note column says why.")
