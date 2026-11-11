"""Sports Betting Sniper dashboard (Streamlit). Research and paper trading only.

    streamlit run src/sports_sniper/dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from sports_sniper import backtest, bankroll, breakout, db, ladder, model_compare, parlay, performance, research
from sports_sniper import odds_math
from sports_sniper.config import get_settings
from sports_sniper.markets import book_label, market_label
from sports_sniper.movement import moves
from sports_sniper.providers import schedule
from sports_sniper.opportunities import EVIDENCE_ORDER, describe
from sports_sniper.scanner import run_scan
from sports_sniper.settlement import profit as settle_profit
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
st.session_state["page_runs"] = st.session_state.get("page_runs", 0) + 1
with st.sidebar:
    auto_refresh = st.toggle("Refresh every minute", value=True, key="auto_refresh",
                             help="Reloads the page each minute so new prices show up without touching it. The scheduled "
                                  "scanners store them; this only re-reads what is stored and uses no API credits.")


    @st.fragment(run_every=60 if auto_refresh else None)
    def heartbeat() -> None:
        """Runs with the page, then alone once a minute; a run of its own asks for the whole page again."""
        if st.session_state.get("heartbeat_run") == st.session_state["page_runs"]:
            st.rerun()
        st.session_state["heartbeat_run"] = st.session_state["page_runs"]
        st.caption(f"Page read {datetime.now().astimezone():%I:%M:%S %p}".replace(" 0", " "))


    heartbeat()
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


def money(value) -> str:
    return "-" if pd.isna(value) else f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def plus_minus(value, error) -> str:
    return "-" if value is None else pct(value, True) + ("" if error is None else f" ± {error:.1%}")


def model_view(probability, break_even) -> str:
    """What the prediction model says about a bet's side, when it has a calibrated forecast for it."""
    if pd.isna(probability):
        return "-"
    return f"{'Agrees' if probability > break_even else 'Disagrees'} ({probability:.0%})"


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
        "Model view": [model_view(p, b) for p, b in zip(data.model_probability, data.break_even)],
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
    g1, g2, g3 = st.columns([1, 1, 2])
    seen_price = g3.number_input("Price you saw, if it had changed (American; 0 = unchanged)", value=0, step=5, key=f"{key}_seen")
    if g1.button("Got it", key=f"{key}_got", help="The price was still there when you opened the book."):
        save_check(int(row.id), "got", int(seen_price) or None)
    if g2.button("Gone", key=f"{key}_gone", help="The price had moved or the line was no longer offered."):
        save_check(int(row.id), "gone")
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

# --- bet now: prices seen in the last few minutes, and whether they were really there -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def next_games() -> list[dict]:
    try:
        return schedule.upcoming(settings.results_dsn, datetime.now(timezone.utc))
    except Exception:
        return []


def check_key(row) -> tuple:
    return (row.event_id, row.book, row.market, row.player_key, row.line_key, row.side)


def save_check(opportunity_id: int, outcome: str, price_seen: int | None = None) -> None:
    try:
        with Session(engine()) as session, session.begin():
            db.record_check(session, session.get(db.OpportunityRow, opportunity_id), outcome, price_seen=price_seen)
    except ValueError as problem:
        st.warning(str(problem).capitalize() + ".")
        return
    st.rerun()


CHECK_LABEL = {"got": "✔ Got it", "gone": "✖ Gone"}
all_checks = db.load_checks(engine())
checked = {check_key(c): c.outcome for c in all_checks[all_checks.latest].itertuples()} if not all_checks.empty else {}
fresh_age = (now - priced.observed_at).dt.total_seconds() / 60 if not priced.empty else pd.Series(dtype=float)
bet_now = priced[priced.available & (fresh_age <= settings.fresh_minutes) & (priced.conservative_ev > min_edge) & (priced.ev >= min_ev)
                 & priced.evidence.isin(tiers) & priced.is_best_price] if not priced.empty else priced
for column, wanted in (("sport", sports), ("market", markets), ("book", books)):
    if wanted and not bet_now.empty:
        bet_now = bet_now[bet_now[column].isin(wanted)]
bet_now = bet_now.sort_values("commence_time") if not bet_now.empty else bet_now

st.subheader(f"Bet now ({len(bet_now)})" if len(bet_now) else "Bet now")
if bet_now.empty:
    st.caption(f"No flagged price has been seen in the last {settings.fresh_minutes:.0f} minutes. Prices are stored in the last "
               "40 minutes before an NBA game and the last hour before an NFL game, every ten minutes.")
else:
    st.caption(f"Flagged prices seen in the last {settings.fresh_minutes:.0f} minutes, soonest game first. Open the book, then say "
               "whether the price was still there: that is the only way to learn if these bets can really be placed. "
               "Nothing here places a bet.")
    for o in bet_now.head(12).itertuples():
        starts = (o.commence_time - now).total_seconds() / 60
        left, when, go, yes, no = st.columns([5, 2.2, 1.6, 1.2, 1.2], vertical_alignment="center")
        left.markdown(f"**{label(o)}** · {book_label(o.book)} **{american(o.price)}** (fair {american(o.fair_price)}, "
                      f"edge {pct(o.ev, True)}) · model: {model_view(o.model_probability, o.break_even).lower()}")
        when.markdown(f"Starts in **{starts:.0f} min** · seen {ago(o.observed_at, now)}")
        link = bet_link(o.link, o.data_source)
        if link:
            go.link_button(f"Open {book_label(o.book)}", link)
        answer = checked.get(check_key(o))
        if answer:
            yes.markdown(CHECK_LABEL[answer])
            if no.button("Change", key=f"now_change_{o.id}", help="Adds a correction. Your first answer is kept as the measurement."):
                save_check(int(o.id), "gone" if answer == "got" else "got")
        else:
            if yes.button("Got it", key=f"now_got_{o.id}", help="The price was still there when you opened the book."):
                save_check(int(o.id), "got")
            if no.button("Gone", key=f"now_gone_{o.id}", help="The price had moved or the line was no longer offered."):
                save_check(int(o.id), "gone")
games_ahead = next_games()
if games_ahead:
    soon = [g for g in games_ahead if g["start"] > now][:8]
    if soon:
        with st.expander("When to look", expanded=bet_now.empty):
            st.dataframe(pd.DataFrame([{"Game": f"{g['sport'].upper()} · {g['name']}", "Starts": local(g["start"]),
                                        "Prices from about": local(g["prices_from"]),
                                        "In": "now" if g["prices_from"] <= now else f"{(g['prices_from'] - now).total_seconds() / 60:.0f} min"}
                                       for g in soon]), hide_index=True, width="stretch")

board_tab, longshot_tab, radar_tab, parlay_tab, moves_tab, perf_tab, paper_tab, backtest_tab, model_tab = st.tabs(
    ["Opportunity Board", "Longshot Hunter", "Breakout Radar", "Parlay Lab", "Line Movement", "Performance", "Paper Bets",
     "Backtest", "Model Comparison"])


@st.cache_data(ttl=600, show_spinner="Replaying the stored snapshots (a couple of minutes the first time)...")
def replay():
    """Every flagged price in every stored snapshot, graded, and one row per two-sided line with the model's chance."""
    found, offers = backtest.candidates(engine(), settings, cache=backtest.default_cache(settings))
    found, offers, _ = research.with_models(found, offers, settings)
    return found, offers


@st.cache_data(ttl=1800, show_spinner="Reading past lines and results...")
def ladder_ratios():
    try:
        extra = ladder.nfl_line_history(settings.results_dsn)
    except Exception:
        extra = None
    return ladder.combined_library(replay()[1], extra)


@st.cache_data(ttl=1800, show_spinner="Scoring the ladder model on stored alternates...")
def ladder_evidence():
    return research.ladder_check(engine(), replay()[1], settings)


@st.cache_data(ttl=1800, show_spinner="Reading this season's game logs...")
def game_logs():
    try:
        return breakout.load_history(settings.results_dsn)
    except Exception:
        return {}


def latest_look(hours: float, upcoming_only: bool = True) -> pd.DataFrame:
    """The newest stored snapshot of each game seen in the last `hours`."""
    recent = db.load_prices(engine(), since=now - timedelta(hours=hours))
    if recent.empty:
        return recent
    if upcoming_only:
        recent = recent[recent.commence_time > now]
    return recent[recent.observed_at == recent.groupby("event_id").observed_at.transform("max")]

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
                   "Every estimate comes from other sportsbooks' prices for the identical bet, with their margin removed. "
                   "**Model view** is a second opinion only: the prediction model's calibrated chance of the same side, "
                   "\"Agrees\" when it is above the price's break-even. It does not change what is flagged or how it is ranked.")
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

    st.markdown("#### Model-priced longshots (speculative)")
    st.caption("One-sided alternate lines given a chance by the ladder model: the sharp book's main line for the same player, "
               "plus how far players have finished from their lines in the past. On the alternates stored so far the model's "
               "\"underpriced\" longshots lost money (see the evidence below), so these are for study, never entered as paper bets.")
    if st.button("Price longshots with the ladder model", key="ladder_load") or st.session_state.get("ladder_on"):
        st.session_state["ladder_on"] = True
        ratios = ladder_ratios()
        modelled = ladder.current(latest_look(3, not show_expired), ratios, settings, int(floor))
        if modelled.empty:
            st.info("No alternate lines in the latest stored prices. The scheduled NFL scanner no longer collects them; a live "
                    "scan with alternate markets configured would.")
        else:
            st.dataframe(pd.DataFrame({
                "Player / market": [f"{m.player} — Over {m.line:g} {market_label(m.market)}" for m in modelled.itertuples()],
                "Game": modelled.event_name.to_list(), "Book": modelled.book.map(book_label).to_list(),
                "Bet": [bet_link(m.link, scan.source) for m in modelled.itertuples()],
                "Odds": modelled.price.map(american).to_list(),
                "Main line": [f"{m.anchor_line:g} ({m.anchor_probability:.0%} over)" for m in modelled.itertuples()],
                "Model's chance": [f"{m.probability:.1%} ± {m.standard_error:.1%}" for m in modelled.itertuples()],
                "Break-even": (1 / modelled.decimal).map(pct).to_list(),
                "Model EV": modelled.ev.map(lambda v: pct(v, True)).to_list(),
                "Conservative EV": modelled.conservative_ev.map(lambda v: pct(v, True)).to_list(),
                "Estimate from": "Ladder model", "Evidence": "Speculative",
            }).head(100), hide_index=True, width="stretch", column_config=LINK_COLUMN)
        check = ladder_evidence()
        if check["lines"]:
            st.markdown("**Has the ladder model paid?**")
            st.dataframe(pd.DataFrame([{"Bets": name, "Settled": m["won"] + m["lost"] + m["push"], "Record (W-L)": f"{m['won']}-{m['lost']}",
                                        "ROI": plus_minus(m["roi"], m["roi_se"]), "The model's own estimate": pct(m["estimated_ev"], True)}
                                       for name, m in check["groups"]]), hide_index=True, width="stretch")
            st.caption(f"{check['lines']} settled alternates from {check['games']} games, each priced only from results known before "
                       "its game. A model whose picks lose is describing the ladder, not beating it.")

with radar_tab:
    st.subheader("Breakout Radar")
    st.caption("Players whose recent opportunity (minutes and shot attempts; targets, carries and snap share) is clearly "
               "above their own average earlier this season, with every recent game above it. Built from game logs only: "
               "injuries, lineups and news are not read here. A signal is information about the player, not an edge.")
    if st.button("Load the radar", key="radar_load") or st.session_state.get("radar_on"):
        st.session_state["radar_on"] = True
        logs = game_logs()
        latest_prices = db.load_prices(engine(), since=now - timedelta(hours=3))
        if not latest_prices.empty:
            latest_prices = latest_prices[latest_prices.commence_time > now]
            newest = latest_prices.groupby("event_id").observed_at.transform("max")
            latest_prices = latest_prices[(latest_prices.observed_at == newest) & latest_prices.book.isin(settings.bet_books)]
        flagged = breakout.radar(logs, latest_prices, now.astimezone(breakout.results.EASTERN).date())
        if not logs:
            st.info("The game-log database could not be read, so there is nothing to show.")
        elif flagged.empty:
            st.info("No player clears the bar right now.")
        else:
            only_playing = st.checkbox("Only players with a stored line for a game still to start", value=bool(flagged.playing.any()),
                                       key="radar_playing")
            shown = flagged[flagged.playing] if only_playing else flagged
            number = lambda row, v: f"{v:.0%}" if row.metric == "snap_share" else f"{v:.1f}"      # noqa: E731
            st.dataframe(pd.DataFrame({
                "Player": shown.player.to_list(), "Sport": shown.sport.str.upper().to_list(), "Team": shown.team.to_list(),
                "Signal": shown.label.str.capitalize().to_list(),
                "Earlier average": [number(r, r.baseline) for r in shown.itertuples()],
                "Recent average": [number(r, r.recent) for r in shown.itertuples()],
                "Change": shown.share.map(lambda v: pct(v, True)).to_list(),
                "Last game": shown.last_game.map(str).to_list(),
                "Lines seen now": ["; ".join(f"{market_label(p['market'])} over {p['line']:g} {american(p['price'])} at {book_label(p['book'])}"
                                             for p in props) or "-" for props in shown.props],
            }).head(60), hide_index=True, width="stretch")
            if len(shown):
                pick = st.selectbox("Explain", shown.index, key="radar_pick",
                                    format_func=lambda i: f"{shown.loc[i].player} ({shown.loc[i].label})")
                st.write(shown.loc[pick].explanation)
                for prop in shown.loc[pick].props:
                    link = bet_link(prop["link"], scan.source)
                    if link:
                        st.link_button(f"{market_label(prop['market'])} over {prop['line']:g} {american(prop['price'])} at "
                                       f"{book_label(prop['book'])}", link)
        st.markdown("#### Has the signal paid?")
        if st.button("Load the evidence", key="radar_evidence") or st.session_state.get("radar_evidence_on"):
            st.session_state["radar_evidence_on"] = True
            checked = breakout.validate(replay()[1], logs)
            if checked.empty:
                st.write("No settled lines to test it on yet.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Sport": b.sport.upper(), "Lines": "Breakout signal" if b.breakout else "No signal", "Count": b.lines,
                    "Players": b.players, "Over won": pct(b.over_rate), "Market's chance of the over": pct(b.market_over),
                    "Betting every over": plus_minus(b.over_roi, None if pd.isna(b.over_roi_se) else b.over_roi_se)}
                    for b in checked.itertuples()]), hide_index=True, width="stretch")
                st.caption("Every stored two-sided line, split by whether the player had a signal from games before that day. "
                           "If flagged players do not go over more often than the price implies, the books already know.")

with parlay_tab:
    st.subheader("Parlay Lab")
    st.caption("The chance that several bets in one game all win, allowing for how they move together. Multiplying the legs "
               "assumes they are independent; in one game they are not. Each leg's own chance is the market consensus (other "
               "books, margin removed). A book's same-game parlay price is not in the odds feed, so type it in to compare. "
               "Research only.")
    links = parlay.load(parlay.default_path(settings.database_url))
    if not links:
        st.info("The links between statistics have not been measured yet. Run `python -m sports_sniper.cli evaluate` once "
                "(the scheduled pass also does it daily).")
    elif st.button("Load the legs", key="parlay_load") or st.session_state.get("parlay_on"):
        st.session_state["parlay_on"] = True
        look = latest_look(36 if show_expired else 3, not show_expired)
        if look.empty:
            st.info("No stored prices for games still to start. Tick **Include prices no longer available** to study recent games.")
        else:
            games = look.drop_duplicates("event_id").sort_values("commence_time")
            game = st.selectbox("Game", games.event_id, key="parlay_game",
                                format_func=lambda e: f"{games.set_index('event_id').loc[e].event_name} "
                                                      f"({local(games.set_index('event_id').loc[e].commence_time)})")
            rows = look[look.event_id == game]
            two_sided = pd.DataFrame(backtest._offers([backtest._quote(r) for r in rows.itertuples(index=False)], settings))
            if two_sided.empty:
                st.info("No two-sided line in this game has a sharp reference right now.")
            else:
                team_of = parlay.teams_from(game_logs())
                options = {}
                for o in two_sided.itertuples():
                    for side, chance in (("over", o.consensus_probability), ("under", 1 - o.consensus_probability)):
                        options[f"{o.player} — {side.capitalize()} {o.line:g} {market_label(o.market)} ({chance:.0%})"] = dict(
                            probability=chance, event_id=o.event_id, player_key=o.player_key, market=o.market, side=side,
                            team=team_of.get((o.sport, o.player_key)))
                picked = st.multiselect("Legs (2 to 4)", sorted(options), max_selections=4, key="parlay_legs")
                offered = st.number_input("The book's price for this parlay (American; 0 if you have none)", value=0, step=10, key="parlay_price")
                if len(picked) >= 2:
                    try:
                        out = parlay.evaluate([options[name] for name in picked], links, rows.sport.iloc[0], int(offered) or None)
                    except ValueError as problem:
                        st.warning(str(problem).capitalize() + ".")
                    else:
                        p1, p2, p3, p4 = st.columns(4)
                        p1.metric("If the legs were independent", pct(out["independent"]),
                                  f"fair {american(odds_math.decimal_to_american(out['independent_decimal']))}", delta_color="off")
                        p2.metric("Allowing for the links", pct(out["joint"]),
                                  f"fair {american(odds_math.decimal_to_american(out['fair_decimal']))}", delta_color="off")
                        p3.metric("EV at the book's price", "-" if out["ev"] is None else pct(out["ev"], True))
                        p4.metric("EV if you assumed independence", "-" if out["ev"] is None else pct(out["ev_if_independent"], True))
                        st.dataframe(pd.DataFrame([{"Leg": picked[n["a"]], "With": picked[n["b"]],
                                                    "Link": f"{n['correlation']:+.2f}", "Why": n["why"]} for n in out["links"]]),
                                     hide_index=True, width="stretch")
                        st.caption("A positive link makes both legs more likely to win together than multiplying suggests; a "
                                   "negative one less. The links are averages over many players and games, so a particular pair "
                                   "can differ. Books price same-game parlays knowing this.")
                else:
                    st.write("Pick at least two legs.")

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


def results_table(rows: list[tuple[str, dict]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Group": name, "Bets": r["bets"], "Open": r["open"], "Record (W-L-P)": f"{r['won']}-{r['lost']}-{r['push']}", "Void": r["void"],
        "Win rate": "-" if r["win_rate"] is None else f"{r['win_rate']:.1%}",
        "Simulated profit": money(r["profit"]) if r["staked"] else "-", "ROI": plus_minus(r["roi"], r["roi_se"]),
        "Est. EV": "-" if r["estimated_ev"] is None else pct(r["estimated_ev"], True),
        "Closing line value": plus_minus(r["closing_value"], r["closing_se"]), "With a close": r["closing_count"],
    } for name, r in rows])


all_bets = db.load_paper_bets(engine())

HYPOTHESIS_BETS = 100        # settled forward bets needed in each group before the model-agrees split means anything

with perf_tab:
    st.subheader("Performance")
    real = all_bets[all_bets.data_source != "sample"] if not all_bets.empty else all_bets
    if real.empty:
        st.info("No paper bets on real prices yet. Scan saved prices: every flagged bet that clears the bar is entered "
                "automatically, then **Settle paper bets** grades it once its game is final.")
    else:
        names = sorted(real.strategy.unique())
        chosen = st.selectbox("Strategy", names, index=names.index(db.STRATEGY_AUTO) if db.STRATEGY_AUTO in names else 0,
                              format_func=lambda n: STRATEGY_LABEL.get(n, n),
                              help="Each rule keeps its own record. A change of rule starts a new one rather than rewriting the old.")
        with_replays = st.checkbox("Include replayed bets", value=False,
                                   help="A replayed bet was entered long after its price was seen, from saved snapshots. Its "
                                        "price was real, but the rule was applied after the fact. Forward bets, entered when "
                                        "the price was seen, are the clean test and the default here.")
        everything = real[real.strategy == chosen]
        view = everything if with_replays else everything[~everything.replayed]
        real_checks = all_checks[all_checks.data_source != "sample"] if not all_checks.empty else all_checks

        st.markdown("### Demonstrated: prices confirmed at the sportsbook")
        if real_checks.empty:
            st.info("**Nothing is demonstrated yet: 0 flagged prices have been checked at a sportsbook.** Everything below is "
                    "research on paper, which assumes every flagged price could be bet. Use **Got it** / **Gone** in the Bet "
                    "now panel after opening a bet to start measuring it.")
        else:
            a = performance.availability(real_checks)
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Prices checked", a["checks"], f"{a['corrected']} corrected later" if a["corrected"] else None, delta_color="off")
            a2.metric("Still there", f"{a['got_rate']:.0%}", f"95%: {a['interval'][0]:.0%} to {a['interval'][1]:.0%}", delta_color="off")
            a3.metric("Got it at a different price", a["changed"], f"{a['worse']} worse", delta_color="off")
            a4.metric("Typical time to check", f"{a['median_seconds'] / 60:.1f} min", "after the price was stored", delta_color="off")
            firsts = real_checks[real_checks["first"]]
            st.dataframe(pd.DataFrame([{"Book": book_label(k), "Checked": len(g), "Still there": pct((g.outcome == "got").mean()),
                                        "Gone": int((g.outcome == "gone").sum())} for k, g in firsts.groupby("book")]),
                         hide_index=True, width="stretch")
            confirmed = performance.checked_returns(everything, real_checks, settle_profit)
            proven = performance.summarize(confirmed) if len(confirmed) else None
            if proven is None or not proven["bets"]:
                st.write("None of the confirmed prices belongs to a paper bet of this strategy yet.")
            else:
                d1, d2, d3 = st.columns(3)
                d1.metric("Confirmed bets", proven["bets"], f"{proven['open']} open", delta_color="off")
                d2.metric("Confirmed record (W-L-P)", f"{proven['won']}-{proven['lost']}-{proven['push']}")
                d3.metric("Return at checked prices", plus_minus(proven["roi"], proven["roi_se"]),
                          money(proven["profit"]) if proven["staked"] else None, delta_color="off")
                st.dataframe(results_table([("All confirmed", proven)]
                                           + [(f"Sport: {k.upper()}", performance.summarize(g)) for k, g in confirmed.groupby("sport")]
                                           + [(f"Market: {market_label(k)}", performance.summarize(g)) for k, g in confirmed.groupby("market")]
                                           + [(f"Book: {book_label(k)}", performance.summarize(g)) for k, g in confirmed.groupby("book")]),
                             hide_index=True, width="stretch")
            st.caption("Only each bet's first answer counts toward \"still there\"; later corrections are kept beside it. Return at "
                       "checked prices re-grades the confirmed bets at the price you actually saw. ± is one standard error.")

        st.markdown("### Research: paper bets, assuming every flagged price could be bet")
        st.caption(("Forward bets only. " if not with_replays else "Forward and replayed bets together. ")
                   + "Simulated results at prices that were observed; no real money, and no check that the price could be bet.")
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
        rows = [("All", performance.summarize(everything)), ("Forward (entered when seen)", performance.summarize(everything[~everything.replayed])),
                ("Replayed from saved snapshots", performance.summarize(everything[everything.replayed]))]
        st.dataframe(results_table([r for r in rows if r[1]["bets"]]), hide_index=True, width="stretch")
        first_answer = {check_key(c): c.outcome for c in real_checks[real_checks["first"]].itertuples()} if not real_checks.empty else {}
        view = view.assign(model=["No model forecast" if pd.isna(p) else "Model agrees" if p > 1 / d else "Model disagrees"
                                  for p, d in zip(view.model_probability, view.decimal)],
                           check=[{"got": "Price was there (Got it)", "gone": "Price was gone"}.get(first_answer.get(check_key(b)), "Not checked")
                                  for b in view.itertuples()])
        for title, column, name in (("By sport", "sport", str.upper), ("By market", "market", market_label),
                                    ("By sportsbook", "book", book_label), ("By side", "side", str.capitalize),
                                    ("By evidence", "evidence", str.capitalize),
                                    ("By whether the price was there", "check", str)):
            st.markdown(f"#### {title}")
            st.dataframe(results_table([(name(key), performance.summarize(group)) for key, group in view.groupby(column)]),
                         hide_index=True, width="stretch")

        st.markdown("### Probability accuracy")
        parts = [(f"{title}: {name(row[column])}", row) for title, column, name in (("Sport", "sport", str.upper), ("Market", "market", market_label),
                                                                                  ("Book", "book", book_label))
                 for row in performance.accuracy(view, column).to_dict("records")]
        if total["won"] + total["lost"] < 30:
            st.write("Not enough settled bets to judge calibration yet (30 needed).")
        else:
            st.dataframe(pd.DataFrame([{"Group": name, "Decided bets": r["bets"], "Average estimate": pct(r["estimate"]),
                                        "Actually won": pct(r["won"]), "95% interval": f"{r['interval'][0]:.0%} to {r['interval'][1]:.0%}",
                                        "Brier, our estimate": f"{r['brier']:.4f}", "Brier, the price's break-even": f"{r['break_even_brier']:.4f}"}
                                       for name, r in parts]), hide_index=True, width="stretch")
            bands = performance.calibration(view)
            st.dataframe(pd.DataFrame([{"Estimated chance": f"{b['low']:.0%} to {b['high']:.0%}", "Bets": b["n"],
                                        "Average estimate": pct(b["predicted"]), "Actually won": pct(b["observed"]),
                                        "95% interval": f"{b['interval'][0]:.0%} to {b['interval'][1]:.0%}"} for b in bands]),
                         hide_index=True, width="stretch")
            st.caption(f"Brier score {total['brier']:.4f} for these estimates against {total['break_even_brier']:.4f} for the "
                       "prices' own break-even chance (lower is better). An estimate that cannot beat the price it was "
                       "compared with is not adding information.")

        st.markdown("### Hypothesis under test: bets the model agrees with do better")
        forward = everything[~everything.replayed]
        forward = forward.assign(model=["none" if pd.isna(p) else "agrees" if p > 1 / d else "disagrees"
                                        for p, d in zip(forward.model_probability, forward.decimal)])
        tested = []
        for name, group in [("All markets", forward)] + [(market_label(k), g) for k, g in forward.groupby("market")]:
            yes, no = performance.summarize(group[group.model == "agrees"]), performance.summarize(group[group.model == "disagrees"])
            tested.append({"Market": name, "Agrees: settled": yes["won"] + yes["lost"] + yes["push"], "Agrees: record": f"{yes['won']}-{yes['lost']}-{yes['push']}",
                           "Agrees: ROI": plus_minus(yes["roi"], yes["roi_se"]), "Disagrees: settled": no["won"] + no["lost"] + no["push"],
                           "Disagrees: record": f"{no['won']}-{no['lost']}-{no['push']}", "Disagrees: ROI": plus_minus(no["roi"], no["roi_se"])})
        st.dataframe(pd.DataFrame(tested), hide_index=True, width="stretch")
        enough = tested[0]["Agrees: settled"] >= HYPOTHESIS_BETS and tested[0]["Disagrees: settled"] >= HYPOTHESIS_BETS
        st.caption("Stated on 2026-11-11 after a replay in which agreed bets went 22-17 and the rest 21-23, a gap well inside "
                   "chance. Only forward bets entered since the model's view was recorded count here, so it cannot be fitted to "
                   f"the bets that suggested it. It needs {HYPOTHESIS_BETS} settled bets in each group before it says anything"
                   + ("." if enough else "; it does not have them yet.") + " The model's view does not affect what is flagged.")

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
        table_all = model_compare.scores(offers, "sport")
        if table_all.empty:
            st.info("No settled lines with a model forecast made before the price was seen yet.")
        else:
            both = pd.concat([table_all, model_compare.scores(offers, "market")])
            st.dataframe(pd.DataFrame([{
                "Lines": s.group.upper() if s.group in ("nba", "nfl", "mlb") else market_label(s.group), "Settled": s.lines,
                "Over won": pct(s.over_rate), "Bet book's own price": f"{s.book_brier:.4f}",
                "Market consensus": f"{s.consensus_brier:.4f}", "Prediction model": f"{s.model_brier:.4f}",
                "Closest": min(model_compare.BASE, key=lambda k: getattr(s, f"{k}_brier")).capitalize()} for s in both.itertuples()]),
                hide_index=True, width="stretch")
            st.caption("Brier score: the average squared miss of the probability, lower is better. A coin flip scores 0.2500. "
                       "NFL covers receptions and passing touchdowns only; the yardage projections are anchored to the line.")
            fitted = model_compare.scores(offers, "sport", tuple(model_compare.SOURCES))
            if not fitted.empty:
                st.markdown("#### After calibration")
                st.dataframe(pd.DataFrame([{
                    "Lines": s.group.upper(), "Settled": s.lines, "Market consensus": f"{s.consensus_brier:.4f}",
                    "Prediction model": f"{s.model_brier:.4f}", "Model, calibrated": f"{s.calibrated_brier:.4f}",
                    "Market moved by the model": f"{s.blend_brier:.4f}"} for s in fitted.itertuples()]), hide_index=True, width="stretch")
                st.caption("The calibration and the model's weight against the market are fitted on earlier days only, then "
                           "scored on later ones. If moving the market's chance by the model does not beat the market alone, "
                           "the model is not adding information.")
            checked_candidate = research.candidate_check(offers, settings)
            if not checked_candidate["scores"].empty:
                st.markdown("#### NBA candidate and calibration by statistic")
                st.dataframe(pd.DataFrame([dict({"Lines": "All NBA" if c.group == "All" else market_label(c.group), "Settled": c.lines},
                                                **{label_: f"{getattr(c, key + '_brier'):.4f}" for key, label_ in model_compare.CANDIDATE_SOURCES.items()})
                                           for c in checked_candidate["scores"].itertuples()]), hide_index=True, width="stretch")
                st.caption("The candidate keeps the live model's projections but takes its scale and spread from earlier days' "
                           "results, then is calibrated by statistic on earlier days and scored on later ones. Lower is better; "
                           "the market consensus column is the one to beat.")
                if not checked_candidate["weights"].empty:
                    st.dataframe(pd.DataFrame([{"Statistic": market_label(w.market), "Lines of its own": w.own_lines,
                                                "Share of its own fit": f"{w.own_weight:.0%}", "Confidence": f"{w.confidence:.2f}",
                                                "Weight against the market": f"{w.weight_against_market:.2f}"}
                                               for w in checked_candidate["weights"].itertuples()]), hide_index=True, width="stretch")
                    st.caption("Confidence multiplies the candidate's log-odds (1 = take it at its word, 0 = it says nothing). The "
                               "weight against the market is never negative and may stay at zero. A statistic with few lines leans "
                               "on the fit for the whole sport.")
                found_out = checked_candidate["diagnostics"]
                if found_out:
                    st.dataframe(pd.DataFrame([{"Statistic": w.stat, "Player-games": w.player_games, "Average miss": f"{w.mae:.2f}",
                                                "Miss if minutes were known": f"{w.mae_minutes_known:.2f}",
                                                "Share owed to minutes": f"{w.minutes_share:.0%}"} for w in found_out["workload"].itertuples()]),
                                 hide_index=True, width="stretch")
                    st.caption("How much of the live model's miss is playing time: its projection rescaled to the minutes actually "
                               "played. Usual minutes are the average of the previous five games, a stand-in.")
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
                                       for b in model_compare.calibration_table(offers, "model", 10)]), hide_index=True, width="stretch")
