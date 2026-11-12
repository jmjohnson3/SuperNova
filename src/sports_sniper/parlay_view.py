"""Dashboard tab: the Parlay Lab. Research only; the links open a sportsbook's betslip and place nothing."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from sports_sniper import breakout, db, odds_math, parlay, parlay_legs, phone_view, predictions
from sports_sniper.markets import book_label, market_label

PRICE_HOURS = 12             # how far back to look for a stored prop line when reading projections


def when(stamp) -> str:
    return pd.Timestamp(stamp).to_pydatetime().astimezone().strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def american(price) -> str:
    return "-" if price is None or pd.isna(price) else f"{int(price):+d}"


def pct(value, signed: bool = False) -> str:
    return "-" if value is None or pd.isna(value) else (f"{value:+.1%}" if signed else f"{value:.1%}")


def chance(value) -> str:
    """A parlay's chance: two decimals once it is under 1%, where one would read as zero."""
    return "-" if value is None or pd.isna(value) else f"{value:.2%}" if value < 0.01 else f"{value:.1%}"


def leg_name(leg: dict, with_game: bool) -> str:
    name = f"{leg['player']} — {leg['side'].capitalize()} {leg['line']:g} {market_label(leg['market'])}"
    return f"{name} · {leg['event_name']}" if with_game else name


def hits_text(found) -> str:
    return "-" if found is None else f"{found[0]} of {found[1]}"


@st.cache_data(ttl=1800, show_spinner="Reading recent game logs...")
def recent_games(dsn, day) -> dict:
    try:
        return parlay_legs.by_player(parlay_legs.load_recent(dsn, day))
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def teams(dsn) -> dict:
    try:
        return parlay.teams_from(breakout.load_history(dsn))
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def projections(dsn, minute: datetime, _engine) -> pd.DataFrame:
    try:
        return predictions.props(dsn, minute, None)
    except Exception:
        return pd.DataFrame()


def latest_look(engine, now: datetime, hours: float, upcoming_only: bool) -> pd.DataFrame:
    recent = db.load_prices(engine, since=now - timedelta(hours=hours))
    if recent.empty:
        return recent
    if upcoming_only:
        recent = recent[recent.commence_time > now]
    return recent[recent.observed_at == recent.groupby("event_id").observed_at.transform("max")]


def legs_table(legs: list[dict], history: dict, props: pd.DataFrame, link_of) -> pd.DataFrame:
    rows = []
    for leg in legs:
        games = history.get((leg["sport"], leg["player_key"]))
        projected, model = parlay_legs.projection_for(props, leg)
        rows.append({"Leg": leg_name(leg, False), "Game": leg["event_name"], "Price": american(leg["price"]),
                     "Market chance": pct(leg["probability"]), "Projection": "-" if projected is None else f"{projected:.1f}",
                     "Model chance": pct(model),
                     "Last 5": hits_text(parlay_legs.hit_count(games, leg["market"], leg["line"], leg["side"], 5)),
                     "Last 10": hits_text(parlay_legs.hit_count(games, leg["market"], leg["line"], leg["side"], 10)),
                     "Bet": link_of(leg["link"])})
    return pd.DataFrame(rows)


def render(settings, engine, now: datetime, show_expired: bool, bet_link) -> None:
    st.subheader("Parlay Lab")
    st.caption("The chance that several bets all win, allowing for how legs in the same game move together. Multiplying the "
               "legs assumes they are independent; in one game they are not. Each leg's own chance is the market consensus "
               "(other books, margin removed). Research only.")
    links = parlay.load(parlay.default_path(settings.database_url))
    if not links:
        st.info("The links between statistics have not been measured yet. Run `python -m sports_sniper.cli evaluate` once "
                "(the scheduled pass also does it daily).")
        return
    if not (st.button("Load the legs", key="parlay_load") or st.session_state.get("parlay_on")):
        return
    st.session_state["parlay_on"] = True
    look = latest_look(engine, now, 36 if show_expired else 3, not show_expired)
    if look.empty:
        st.info("No stored prices for games still to start. Tick **Include prices no longer available** to study recent games.")
        return
    books = [b for b in (settings.bet_books or sorted(look.book.unique())) if b in set(look.book)]
    if not books:
        st.info("None of your sportsbooks has a stored price in these games.")
        return
    games = look.drop_duplicates("event_id").sort_values("commence_time")
    games = pd.concat([games[games.commence_time > now], games[games.commence_time <= now].iloc[::-1]]).set_index("event_id")   # next game first
    c1, c2 = st.columns([1, 3])
    book = c1.selectbox("Sportsbook", books, format_func=book_label, key="parlay_book",
                        help="A parlay is placed at one book, so the legs, prices and link all come from this one.")
    if any(e not in games.index for e in st.session_state.get("parlay_games", [])):
        st.session_state["parlay_games"] = [e for e in st.session_state["parlay_games"] if e in games.index]
    chosen = c2.multiselect("Games", list(games.index), default=list(games.index[:1]), key="parlay_games",
                            format_func=lambda e: f"{games.loc[e].event_name} ({when(games.loc[e].commence_time)})")
    if not chosen:
        st.write("Pick at least one game.")
        return
    found = parlay_legs.book_legs(look[look.event_id.isin(chosen)], settings, book)
    if not found:
        st.info(f"No two-sided line at {book_label(book)} in these games has a sharp reference right now.")
        return
    team_of = teams(settings.results_dsn)
    several = len(chosen) > 1
    options = {}
    for leg in sorted(found, key=lambda x: (x["event_name"], x["player"], x["market"], x["line"], x["side"])):
        options[leg_name(leg, several)] = dict(leg, team=team_of.get((leg["sport"], leg["player_key"])))
    if any(name not in options for name in st.session_state.get("parlay_legs", [])):          # a line that moved or left
        st.session_state["parlay_legs"] = [name for name in st.session_state["parlay_legs"] if name in options]
    picked = st.multiselect(f"Legs (2 to {parlay_legs.MAX_LEGS})", list(options), max_selections=parlay_legs.MAX_LEGS, key="parlay_legs")
    legs = [options[name] for name in picked]
    if not legs:
        st.write("Pick at least two legs.")
        return

    history = recent_games(settings.results_dsn, now.date())
    props = projections(settings.results_dsn, now.replace(second=0, microsecond=0), engine)
    phone_view.table(legs_table(legs, history, props, lambda link: bet_link(link, "saved")), sub=["Game"],
                     column_config={"Bet": st.column_config.LinkColumn("Bet", display_text="Open")})
    st.caption("**Last 5 / Last 10**: how many of the player's most recent games this side of this line would have won. "
               "**Projection** and **Model chance** are the stored model forecast (Predictions tab) at this leg's line; the "
               "models have not beaten the market in testing, and a hot streak is already in the book's price. "
               "**Market chance** is what the other books' prices imply.")
    if len(legs) < 2:
        st.write("Pick at least two legs.")
        return

    multiplied = parlay_legs.multiplied_decimal(legs)
    same_game = len({leg["event_id"] for leg in legs}) < len(legs)
    typed = st.number_input(f"{book_label(book)}'s price for this parlay (American; 0 to use the legs multiplied)", value=0, step=10,
                            key="parlay_price",
                            help="The odds feed has no same-game parlay prices. When legs share a game the book reprices the "
                                 "parlay, so type in what its betslip shows.")
    price = int(typed) or parlay_legs.american(multiplied)
    try:
        out = parlay.evaluate(legs, links, legs[0]["sport"], price)
    except ValueError as problem:
        st.warning(str(problem).capitalize() + ".")
        return
    p1, p2, p3, p4 = st.columns(4)
    unpriced = same_game and not typed               # the book reprices a same-game parlay: the legs multiplied is not its price
    p1.metric("If the legs were independent", chance(out["independent"]),
              f"fair {american(odds_math.decimal_to_american(out['independent_decimal']))}", delta_color="off")
    p2.metric("Allowing for the links", chance(out["joint"]),
              f"fair {american(odds_math.decimal_to_american(out['fair_decimal']))}", delta_color="off")
    p3.metric("Legs multiplied" if unpriced else "Price used", american(price),
              "not what the book pays" if unpriced else "typed in" if typed else "legs multiplied", delta_color="off")
    p4.metric("EV at that price", "-" if unpriced else pct(out["ev"], True),
              "needs the book's price" if unpriced else f"{pct(out['ev_if_independent'], True)} if independent", delta_color="off")
    if unpriced:
        st.warning("Some legs share a game, so the book sets its own price for this parlay, usually below the legs "
                   "multiplied. Open the parlay, read the price in its betslip and type it in above to see the EV.")

    address = parlay_legs.parlay_link(book, [bet_link(leg["link"], "saved") for leg in legs])
    if address:
        st.link_button(f"Open this parlay at {book_label(book)}", address)
        st.caption("Opens the legs in the sportsbook's betslip; nothing is placed. Prices there may have moved, and the book "
                   "may refuse a combination."
                   + (" DraftKings' combined link is built from its single-bet links and has not been confirmed to load "
                      "every leg; check the slip." if book == "draftkings" else ""))
    else:
        st.caption(f"No combined link could be built for {book_label(book)} from these legs; use each leg's own link above.")

    tied = [n for n in out["links"] if n["correlation"]]
    if tied:
        st.dataframe(pd.DataFrame([{"Leg": picked[n["a"]], "With": picked[n["b"]], "Link": f"{n['correlation']:+.2f}", "Why": n["why"]}
                                   for n in tied]), hide_index=True, width="stretch")
        st.caption("A positive link makes both legs more likely to win together than multiplying suggests; a negative one "
                   "less. The links are averages over many players and games, so a particular pair can differ. Legs in "
                   "different games are treated as independent.")
    else:
        st.caption("No measured link between these legs (different games, or pairs with too little history): they are "
                   "treated as independent.")
