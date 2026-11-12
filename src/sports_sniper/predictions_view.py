"""Dashboard tab: the stored model forecasts beside the latest line. Information only; it flags nothing,
enters no paper bet and never places a wager."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from sports_sniper import db, phone_view, predictions
from sports_sniper.markets import book_label

SPORTS = {"nba": "NBA", "nfl": "NFL", "ncaaf": "NCAAF"}
PRICE_HOURS = 12             # how far back to look for a stored prop line
GAME_PHONE = ["Model spread", "Line", "Spread lean", "Model total", "Total line", "Total lean", "Missing players"]
PROP_PHONE = ["Stat", "Projection", "Line", "Gap", "Model: over", "Market: over", "Likely range", "Line from"]


def when(stamp) -> str:
    return pd.Timestamp(stamp).to_pydatetime().astimezone().strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def short(team: str, sport: str) -> str:
    """NBA teams by nickname (Indiana Pacers -> Pacers); college and NFL names as stored."""
    return team.split()[-1] if sport == "nba" and " " in team else team


def number(value, digits: int = 1) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}f}"


def spread_text(home: str, away: str, spread, sport: str = "") -> str:
    """A home spread as the favourite's line: -3.5 -> 'Home -3.5', +3.5 -> 'Away -3.5'."""
    if pd.isna(spread):
        return "-"
    if spread == 0:
        return "Pick'em"
    return f"{short(home if spread < 0 else away, sport)} -{abs(spread):.1f}"


def spread_lean(g) -> str:
    if pd.isna(g.spread_gap) or abs(g.spread_gap) < 0.05:
        return "-"
    team, line = (g.home, g.spread) if g.spread_gap > 0 else (g.away, -g.spread)
    return f"{short(team, g.sport)} {line:+.1f} (by {abs(g.spread_gap):.1f})"


def total_lean(g) -> str:
    if pd.isna(g.total_gap) or abs(g.total_gap) < 0.05:
        return "-"
    return f"{'Over' if g.total_gap > 0 else 'Under'} {g.total:.1f} (by {abs(g.total_gap):.1f})"


def games_table(games: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"Sport": SPORTS.get(g.sport, g.sport.upper()), "Game": f"{g.away} at {g.home}", "Starts": when(g.start),
                          "Model spread": spread_text(g.home, g.away, -g.pred_margin, g.sport), "Line": spread_text(g.home, g.away, g.spread, g.sport),
                          "Spread lean": spread_lean(g), "Model total": number(g.pred_total), "Total line": number(g.total),
                          "Total lean": total_lean(g), "Line from": "-" if pd.isna(g.book) or not g.book else book_label(g.book),
                          "Forecast made": when(g.made_at), "Missing players": g.note or ""} for g in games.itertuples()])


def props_table(props: pd.DataFrame) -> pd.DataFrame:
    def chance(value) -> str:
        return "-" if pd.isna(value) else f"{value:.0%}"
    return pd.DataFrame([{"Sport": SPORTS.get(p.sport, p.sport.upper()), "Player": p.player, "Team": p.team, "Game": p.game,
                          "Starts": when(p.start), "Stat": p.stat, "Projection": number(p.projection),
                          "Likely range": "-" if pd.isna(p.low) or pd.isna(p.high) else f"{p.low:.1f} to {p.high:.1f}",
                          "Line": number(p.line), "Gap": "-" if pd.isna(p.gap) else f"{p.gap:+.1f}",
                          "Model: over": chance(p.model_over), "Market: over": chance(p.market_over),
                          "Line from": "-" if pd.isna(p.book) or not p.book else book_label(p.book),
                          "Forecast made": when(p.made_at)} for p in props.itertuples()])


@st.cache_data(ttl=300, show_spinner="Reading the stored forecasts...")
def load_games(dsn, minute: datetime) -> pd.DataFrame:
    return predictions.games(dsn, minute)


@st.cache_data(ttl=300, show_spinner="Reading the stored player projections...")
def load_props(dsn, minute: datetime, _engine) -> pd.DataFrame:
    prices = db.load_prices(_engine, since=minute - timedelta(hours=PRICE_HOURS))
    if len(prices):
        prices = prices[(prices.sport == "nba") & (prices.commence_time > minute)]
    return predictions.props(dsn, minute, prices)


@st.cache_data(ttl=3600, show_spinner=False)
def load_record(dsn) -> dict | None:
    return predictions.ncaaf_record(dsn)


def record_lines(record: dict | None) -> list[str]:
    lines = []
    if record:
        won, lost = record["spread_record"]
        over, under = record["total_record"]
        lines.append(f"**NCAAF**, measured here on {record['games']} finished games: the model missed the final margin by "
                     f"{record['margin_model']:.1f} points on average against {record['margin_line']:.1f} for the line, and the total by "
                     f"{record['total_model']:.1f} against {record['total_line']:.1f}. Its side of the spread went {won}-{lost} and its "
                     f"side of the total {over}-{under}.")
    else:
        lines.append("**NCAAF**: no finished games with a stored forecast to measure yet. Earlier tests found the ratings do not "
                     "beat the closing line.")
    lines.append("**NBA**: in the stored tests the game model does not beat DraftKings' line, and the player model's "
                 "probabilities are worse than the books' own prices (Model Comparison tab).")
    lines.append("**NFL**: the production game forecast is the market line itself, so its gap is zero by design. Player "
                 "projections are anchored to FanDuel's line; the raw model has not beaten that line.")
    return lines


def render(settings, engine, now: datetime) -> None:
    st.subheader("Predictions")
    st.caption("What the models forecast for the coming games, beside the latest stored line. These are the forecasts the "
               "scheduled pipelines already save; this page only reads them. A lean is where the model sits against the "
               "line, not a flagged bet: none of these models has beaten the market in testing, so nothing here is entered "
               "as a paper bet.")
    minute = now.replace(second=0, microsecond=0)
    games, props = load_games(settings.results_dsn, minute), load_props(settings.results_dsn, minute, engine)
    present = [s for s in SPORTS if s in set(games.sport) | set(props.sport)]
    if not present:
        st.info("No stored forecasts for coming games right now. They appear after the morning runs: NCAAF at 7:30 AM MT, NFL "
                "with its daily run, NBA as lineups and injury news arrive.")
        return
    chosen = st.multiselect("Sports", present, default=present, format_func=SPORTS.get, key="predictions_sports")
    games, props = games[games.sport.isin(chosen)], props[props.sport.isin(chosen)]

    st.markdown("### Games: spreads and totals")
    if games.empty:
        st.write("No game forecasts for the chosen sports.")
    else:
        phone_view.table(games_table(games), title="Game", sub=["Sport", "Starts"], keep=GAME_PHONE)
        waiting = int(games.spread.isna().sum())
        st.caption("Model spread and Line both name the favourite. Spread lean is the side the model prefers at that line and by "
                   "how many points; Total lean likewise."
                   + (f" {waiting} game{'s have' if waiting != 1 else ' has'} no stored line yet: NBA lines are saved from about "
                      "40 minutes before tip-off." if waiting else ""))
    with st.expander("How these forecasts have done"):
        for line in record_lines(load_record(settings.results_dsn)):
            st.markdown(line)

    st.markdown("### Player props")
    if props.empty:
        st.write("No player projections for the chosen sports.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    stats = sorted(props.stat.unique())
    wanted = c1.multiselect("Statistic", stats, default=[], placeholder="All statistics", key="predictions_stats")
    name = c2.text_input("Player", "", placeholder="Search by name", key="predictions_player")
    lined = c3.toggle("Lines only", value=bool(props.line.notna().any()), key="predictions_lined",
                      help="Show only projections that have a stored sportsbook line to compare with.")
    view = props
    if wanted:
        view = view[view.stat.isin(wanted)]
    if name.strip():
        view = view[view.player.str.contains(name.strip(), case=False, regex=False)]
    if lined:
        view = view[view.line.notna()]
    disagreement = (view.model_over - view.market_over).abs()
    view = view.assign(order=disagreement.fillna(-1)).sort_values(["order", "start", "player"], ascending=[False, True, True])
    st.write(f"{len(view)} projections" + (f", {int(props.line.notna().sum())} of {len(props)} with a line" if not lined else ""))
    phone_view.table(props_table(view.head(500)), title="Player", sub=["Team", "Game", "Starts"], keep=PROP_PHONE)
    st.caption("Sorted by how far the model's chance of the over is from the market's (the book's two prices with the margin "
               "removed). Likely range is the model's 10th to 90th percentile. NBA lines come from the stored price snapshots "
               "(FanDuel first, then DraftKings), which start about 40 minutes before tip-off. NFL projections are anchored "
               "to FanDuel's line, so their gap is small by design. In testing, the books' prices were "
               "closer to what happened than the models'.")
