"""Sports Betting Sniper dashboard (Streamlit). Research and paper trading only.

    streamlit run src/sports_sniper/dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from sports_sniper import backtest, bankroll, db, model_compare, performance
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


def bet_link(link, source: str) -> str | None:
    """The sportsbook's own link for a bet, or None. Sample data has no real links."""
    if source == "sample" or pd.isna(link) or not link:
        return None
    return str(link).replace("{state}", settings.link_state.lower())


LINK_COLUMN = {"Bet": st.column_config.LinkColumn("Bet", display_text="Open", help="Opens this bet at the sportsbook. "
                                                  "The price there may have changed since it was seen.")}


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
if settings.bet_books and not frame.empty:          # rows stored before the bet-book setting existed
    frame = frame[frame.book.isin(settings.bet_books)]
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
    min_ev = st.slider("Minimum estimated EV", 0.0, 0.15, float(settings.min_ev), 0.005, format="%.3f",
                       help="Flagged bets with an estimated edge under 4% have lost on paper so far; lower this to see them.")
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
        "Bet": [bet_link(o.link, o.data_source) for o in data.itertuples()],
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
    link = bet_link(row.link, row.data_source)
    if link:
        st.link_button(f"Open at {book_label(row.book)}", link)
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
c2.metric("Available now", int((priced.available & (priced.conservative_ev > min_edge) & (priced.ev >= min_ev)).sum()))
c3.metric("Flagged in the last 36 h", len(priced))
c4.metric("Last scan", f"#{scan.id} · {scan.source}")

board_tab, longshot_tab, moves_tab, perf_tab, paper_tab, backtest_tab, model_tab = st.tabs(
    ["Opportunity Board", "Longshot Hunter", "Line Movement", "Performance", "Paper Bets", "Backtest", "Model Comparison"])

with board_tab:
    st.subheader("Today's opportunities")
    board = filtered(priced)
    board = board[(board.conservative_ev > min_edge) & (board.ev >= min_ev) & board.evidence.isin(tiers)]
    if board.empty:
        st.info("No opportunities clear the bar right now. That is a normal result: most prices are fair or worse."
                + ("" if show_expired or priced.empty else " Tick **Include prices no longer available** to review earlier ones."))
    else:
        st.dataframe(table(board), hide_index=True, width="stretch", column_config=LINK_COLUMN)
        st.caption("Ranked by conservative EV: the estimated chance lowered by its uncertainty before the EV is worked out. "
                   "Every estimate on this page comes from other sportsbooks' prices for the identical bet, with their "
                   "margin removed; no prediction model is involved yet.")
        details(board, "board")

with longshot_tab:
    st.subheader("Longshot Hunter")
    floor = st.number_input("Minimum payout odds (American)", value=int(settings.longshot_min_american), step=50)
    longs = filtered(priced)
    longs = longs[(longs.price >= floor) & longs.evidence.isin(tiers) & (longs.conservative_ev > min_edge) & (longs.ev >= min_ev)]
    if longs.empty:
        st.info("No longshot has a supportable positive edge right now.")
    else:
        st.dataframe(table(longs), hide_index=True, width="stretch", column_config=LINK_COLUMN)
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
            "Bet": [bet_link(o.link, o.data_source) for o in outliers.itertuples()],
            "Odds": outliers.price.map(american).to_list(), "Break-even": outliers.break_even.map(pct).to_list(),
            "Above the middle price": outliers.premium_vs_median.map(lambda v: pct(v, True)).to_list(),
            "Books quoting": outliers.books_quoting.to_list(), "Status": "Insufficient data",
            "Price seen": [("" if o.available else "EXPIRED · ") + ago(o.observed_at, now) for o in outliers.itertuples()],
        }), hide_index=True, width="stretch", column_config=LINK_COLUMN)

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
            "Bet": [bet_link(m.link, scan.source) for m in changed.itertuples()],
            "First seen": changed.first_price.map(american).to_list(), "Latest": changed.last_price.map(american).to_list(),
            "Break-even change": changed.break_even_change.map(lambda v: pct(v, True)).to_list(),
            "Best latest price": [f"{american(m.best_now)} at {book_label(m.best_book)}" for m in changed.itertuples()],
            "Latest seen": [("" if now - m.last_seen <= timedelta(minutes=settings.max_snapshot_age_minutes) else "STALE · ")
                            + ago(m.last_seen, now) for m in changed.itertuples()],
        }), hide_index=True, width="stretch", column_config=LINK_COLUMN)
        st.caption("First seen is the first price this app stored, not the book's opening price. A negative break-even "
                   "change means the payout got longer.")

STRATEGY_LABEL = {db.STRATEGY_AUTO: "Automatic v2: edge of 4% or more, at your bet books",
                  db.STRATEGY_AUTO_V1: "Automatic v1 (to Nov 10): any positive conservative edge, every book",
                  db.STRATEGY_MANUAL: "Saved by hand"}
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
            "Book": shown.book.map(book_label).to_list(),
            "Bet": [bet_link(link, source) for link, source in zip(shown.link, shown.data_source)],
            "Odds": shown.price.map(american).to_list(),
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
            "Strategy": shown.strategy.map(lambda n: "By hand" if n == db.STRATEGY_MANUAL else "Automatic").to_list(),
        }), hide_index=True, width="stretch", column_config=LINK_COLUMN)
        if len(all_bets) > len(shown):
            st.caption(f"Showing the newest {len(shown)} of {len(all_bets)}.")
        st.caption("Bets on hypothetical sample data are never settled. A bet stays open until its game is final and complete "
                   "in the results source; the Note column says why.")


# --- Phase 3: backtest, bankroll and model against market ------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Replaying the stored snapshots (a couple of minutes the first time)...")
def replay():
    """Every flagged price in every stored snapshot, graded, and one row per two-sided line with the model's chance."""
    found, offers = backtest.candidates(engine(), settings, cache=backtest.default_cache(settings))
    return found, model_compare.attach_models(offers, settings)


def backtest_table(rows) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rule": name, "Settled": m["won"] + m["lost"] + m["push"], "Record (W-L-P)": f"{m['won']}-{m['lost']}-{m['push']}",
        "ROI": plus_minus(m["roi"], m["roi_se"]),
        "95% interval": "-" if m["roi_interval"] is None else f"{m['roi_interval'][0]:+.0%} to {m['roi_interval'][1]:+.0%}",
        "Est. EV": "-" if m["estimated_ev"] is None else pct(m["estimated_ev"], True),
        "Closing line value": plus_minus(m["closing_value"], m["closing_se"]), "With a close": m["closing_count"],
        "Largest fall (units)": round(m["max_drawdown"], 1), "Longest losing run": m["longest_losing_streak"],
    } for name, m in rows])


with backtest_tab:
    st.subheader("Backtest")
    st.caption("Replays every stored snapshot in time order. A rule sees only what was in a snapshot at that moment, enters "
               "once per bet at the price that was stored, and is graded from final statistics. History on paper, not a forecast.")
    if st.button("Load the backtest", key="bt_load") or st.session_state.get("bt_on"):
        st.session_state["bt_on"] = True
        found, offers = replay()
        if found.empty:
            st.info("No stored real prices to replay yet. Use **Scan saved prices** first.")
        else:
            defaults = tuple(settings.bet_books)
            st.caption(f"Snapshots {local(found.observed_at.min())} to {local(found.observed_at.max())}: "
                       f"{found.event_id.nunique()} games, {len(found)} flagged prices.")
            st.markdown("#### Your rule")
            b1, b2, b3, b4 = st.columns(4)
            bt_edge = b1.slider("Minimum estimated edge", 0.0, 0.10, float(settings.min_ev), 0.005, format="%.3f", key="bt_edge")
            bt_books = b2.multiselect("Books", sorted(found.book.unique()), default=[b for b in defaults if b in set(found.book)],
                                      format_func=book_label, key="bt_books")
            bt_sides = b3.multiselect("Side", ["over", "under"], format_func=str.capitalize, key="bt_sides")
            bt_sports = b4.multiselect("Sport", sorted(found.sport.unique()), format_func=str.upper, key="bt_sports")
            bt_delay = st.checkbox("Enter one look later (about ten minutes), and skip the bet if the line is gone", key="bt_delay",
                                   help="A check on whether the price was really there to bet.")
            rule = backtest.Strategy("Your rule", min_ev=bt_edge, books=tuple(bt_books) or backtest.ANY_BOOK, sides=tuple(bt_sides),
                                     sports=tuple(bt_sports), entry_delay=int(bt_delay))
            mine = backtest.run(found, rule, defaults)
            m = backtest.metrics(mine)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Settled bets", m["won"] + m["lost"] + m["push"], help=f"{m['open']} open, {m['void']} void, over {m['days']} game days")
            k2.metric("Record (W-L-P)", f"{m['won']}-{m['lost']}-{m['push']}")
            k3.metric("Simulated ROI", plus_minus(m["roi"], m["roi_se"]),
                      None if m["roi_interval"] is None else f"95%: {m['roi_interval'][0]:+.0%} to {m['roi_interval'][1]:+.0%}", delta_color="off")
            k4.metric("Closing line value", plus_minus(m["closing_value"], m["closing_se"]),
                      f"largest fall {m['max_drawdown']:.1f} units", delta_color="off")
            st.markdown("#### Rules compared")
            st.dataframe(backtest_table([(s.name, r) for s, r in backtest.compare(found, backtest.presets(settings), defaults)]
                                        + [("Your rule", m)]), hide_index=True, width="stretch")
            st.caption("ROI is profit over $1 stakes; ± is one standard error and the interval is 95%, with whole games as the "
                       "draws. With around a hundred bets the interval is wide: none of these rules is shown to win or lose yet.")
            st.markdown("#### Walk-forward")
            objective = st.radio("Choose the minimum edge on earlier days by", ["roi", "closing_value"], horizontal=True,
                                 format_func={"roi": "Results (ROI)", "closing_value": "Closing line value"}.get, key="bt_objective")
            forward = backtest.walk_forward(found, backtest.replace(rule, name="walk-forward"),
                                            [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08], default_books=defaults, objective=objective)
            if not forward["folds"]:
                st.write("Not enough days stored yet (seven are needed before the first test window).")
            else:
                st.dataframe(pd.DataFrame([{
                    "Later days": f"{f['test_from']:%b %d} to {f['test_to']:%b %d}", "Edge chosen from earlier days": f"{f['chosen']:.1%}",
                    "Settled": f["test_won"] + f["test_lost"] + f["test_push"], "Record (W-L-P)": f"{f['test_won']}-{f['test_lost']}-{f['test_push']}",
                    "ROI": pct(f["test_roi"], True), "Closing line value": pct(f["test_closing_value"], True)} for f in forward["folds"]]),
                    hide_index=True, width="stretch")
                w = backtest.metrics(forward["bets"])
                st.write(f"All later days together: **{w['won']}-{w['lost']}-{w['push']}**, ROI **{plus_minus(w['roi'], w['roi_se'])}**, "
                         f"closing line value {plus_minus(w['closing_value'], w['closing_se'])}. The minimum edge was never "
                         "chosen on the days it is judged on.")
            st.markdown("#### By period and market")
            st.dataframe(backtest_table(backtest.by_period(mine) + [(market_label(k), backtest.metrics(g)) for k, g in mine.groupby("market")]),
                         hide_index=True, width="stretch")
            st.markdown("#### Paper bankroll")
            st.caption("Hypothetical money run through your rule's settled bets, a game day at a time. Nothing here places a bet.")
            s1, s2, s3, s4 = st.columns(4)
            staking = s1.selectbox("Staking", ["flat", "kelly"], format_func={"flat": "Flat stake", "kelly": "Fractional Kelly"}.get, key="br_staking")
            start_money = s2.number_input("Starting bankroll", min_value=1.0, value=100.0, key="br_start")
            flat = s3.number_input("Flat stake", min_value=0.1, value=1.0, key="br_flat")
            share = s4.slider("Share of Kelly", 0.05, 1.0, 0.25, 0.05, key="br_kelly")
            l1, l2, l3, l4 = st.columns(4)
            cap = l1.number_input("Most on one bet", min_value=0.1, value=5.0, key="br_cap")
            daily = l2.number_input("Most in one day", min_value=0.1, value=20.0, key="br_daily")
            per_game = l3.number_input("Most on one game", min_value=0.1, value=3.0, key="br_game")
            stop = l4.slider("Stop after a fall of", 0.0, 0.9, 0.0, 0.05, key="br_stop", help="0 means never stop.")
            sim = bankroll.simulate(mine, bankroll.Rules(start=start_money, staking=staking, flat_stake=flat, kelly_fraction=share,
                                                         max_stake=cap, max_daily=daily, max_per_game=per_game,
                                                         stop_drawdown=stop or None))
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Ending bankroll", f"${sim['final']:,.2f}", money(sim["profit"]), delta_color="off")
            r2.metric("Bets placed", sim["placed"], f"${sim['staked']:,.2f} staked", delta_color="off")
            r3.metric("Largest fall", f"${sim['max_drawdown']:,.2f}", f"{sim['max_drawdown_pct']:.0%} from the high", delta_color="off")
            r4.metric("Lowest point", f"${sim['lowest']:,.2f}")
            if len(sim["curve"]):
                st.line_chart(sim["curve"].set_index("day").bankroll)
            if sim["skipped"]:
                st.caption("Left out: " + ", ".join(f"{n} {why}" for why, n in sim["skipped"].items())
                           + (f". Stopped after {sim['stopped']}." if sim["stopped"] else "."))

with model_tab:
    st.subheader("Model comparison")
    st.caption("Which probability source was closest to what happened? Each two-sided line is scored once, at the first look "
               "where a bet book and a sharp book both priced it. The model's chance comes from a forecast made before the "
               "price was seen (NBA projections; NFL forecasts are not a separate source, they are blended toward FanDuel).")
    if st.button("Load the comparison", key="mc_load") or st.session_state.get("mc_on"):
        st.session_state["mc_on"] = True
        _, offers = replay()
        table_all = model_compare.scores(offers)
        if table_all.empty:
            st.info("No settled lines with a model forecast made before the price was seen yet.")
        else:
            both = pd.concat([table_all, model_compare.scores(offers, "market")])
            st.dataframe(pd.DataFrame([{
                "Lines": "All" if s.group == "All" else market_label(s.group), "Settled": s.lines, "Over won": pct(s.over_rate),
                "Bet book's own price": f"{s.book_brier:.4f}", "Market consensus": f"{s.consensus_brier:.4f}",
                "Prediction model": f"{s.model_brier:.4f}",
                "Closest": min(model_compare.SOURCES, key=lambda k: getattr(s, f"{k}_brier")).capitalize()} for s in both.itertuples()]),
                hide_index=True, width="stretch")
            st.caption("Brier score: the average squared miss of the probability, lower is better. A coin flip scores 0.2500.")
            st.markdown("#### If you bet each source's picks")
            st.dataframe(pd.DataFrame([{
                "Source": model_compare.SOURCES[p.source], "Edge needed": f"{p.threshold:.0%}+", "Settled": p.bets,
                "Record (W-L)": f"{p.won}-{p.lost}", "ROI": plus_minus(None if pd.isna(p.roi) else p.roi, None if pd.isna(p.roi_se) else p.roi_se),
                "Its own estimate": pct(p.estimated_ev, True)} for p in model_compare.picks_table(offers).itertuples()]),
                hide_index=True, width="stretch")
            st.caption("Each source bets the side it prices above break-even by at least the edge shown, at the bet book's price, on "
                       "the same lines. A model that loses here is not adding information to the market's own prices.")
            st.markdown("#### Model calibration")
            st.dataframe(pd.DataFrame([{"Model's chance of the over": f"{b['low']:.0%} to {b['high']:.0%}", "Lines": b["n"],
                                        "Average estimate": pct(b["predicted"]), "Over actually won": pct(b["observed"]),
                                        "95% interval": f"{b['interval'][0]:.0%} to {b['interval'][1]:.0%}"}
                                       for b in model_compare.calibration(offers, "model", 10)]), hide_index=True, width="stretch")
