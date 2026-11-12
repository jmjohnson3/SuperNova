"""What the Parlay Lab shows for each leg and for the whole parlay: one book's prices and links, the stored
projection, and how often the player cleared this line in his last games. Read-only; a link only opens the
sportsbook's betslip, it never places anything."""
from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import numpy as np
import pandas as pd

from sports_sniper import backtest, odds_math
from sports_sniper.normalize import player_key
from sports_sniper.parlay import STATS
from sports_sniper.providers import saved_quotes

MAX_LEGS = 12
HISTORY_DAYS = 500           # far enough back that the last 10 games can reach into the previous season
HISTORY = {
    "nba": ("nbaprops.player_games",
            """SELECT game_date, player_name, pts, reb, ast, fg3m FROM nbaprops.player_games
               WHERE minutes > 0 AND game_date >= %s""",
            ["pts", "reb", "ast", "fg3m"]),
    "nfl": ("raw.nfl_player_gamelogs",
            """SELECT g.game_date_et, COALESCE(r.player_name, g.player_name), g.receiving_yards, g.receptions, g.rushing_yards,
                      g.passing_yards, g.passing_tds
               FROM raw.nfl_player_gamelogs g
               LEFT JOIN LATERAL (SELECT player_name FROM raw.nfl_rosters r WHERE r.player_id = g.player_id
                                  ORDER BY r.season DESC, r.week DESC NULLS LAST LIMIT 1) r ON TRUE
               WHERE g.source LIKE 'nflverse%%' AND g.game_date_et >= %s""",
            ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "passing_tds"]),
}


# --- last games ---------------------------------------------------------------------------------------------
def load_recent(dsn: str | None, today: date, connect=None) -> dict[str, pd.DataFrame]:
    """Recent game logs per sport, one row per player-game, with a column per prop market (read-only)."""
    dsn = dsn or saved_quotes.default_dsn()
    out: dict[str, pd.DataFrame] = {}
    if dsn is None:
        return out
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    try:
        conn = connect(dsn)
    except Exception:
        return out
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for sport, (table, sql, stats) in HISTORY.items():
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    continue
                cur.execute(sql, (today - timedelta(days=HISTORY_DAYS),))
                frame = pd.DataFrame(cur.fetchall(), columns=["day", "player", *[STATS[sport][s] for s in stats]])
                for column in frame.columns[2:]:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame["player_key"] = frame.player.fillna("").map({name: player_key(name) for name in frame.player.fillna("").unique()})
                out[sport] = frame[frame.player_key != ""].sort_values("day").reset_index(drop=True)
    finally:
        conn.close()
    return out


def by_player(history: dict[str, pd.DataFrame]) -> dict[tuple, pd.DataFrame]:
    return {(sport, key): games for sport, frame in history.items() for key, games in frame.groupby("player_key")}


def hit_count(games: pd.DataFrame | None, market: str, line: float, side: str, last: int) -> tuple[int, int] | None:
    """(times this side of this line won, games counted) over the player's most recent `last` games with a value.
    None when the market is not in his logs."""
    market = market.removesuffix("_alternate")
    if games is None or market not in games:
        return None
    values = games[market].dropna().tail(last)
    if values.empty:
        return None
    won = values > line if side == "over" else values < line
    return int(won.sum()), int(len(values))


# --- one book's legs -------------------------------------------------------------------------------------------
def book_legs(rows: pd.DataFrame, settings, book: str) -> list[dict]:
    """Both sides of every two-sided line `book` prices that has a sharp reference, with the book's price and link
    and the market-consensus chance (other books, margin removed)."""
    if rows.empty:
        return []
    only = settings.model_copy(update={"bet_books": [book]})
    quotes = [backtest._quote(r) for r in rows.itertuples(index=False)]
    at_book = {(q.event_id, q.market, q.player_key, q.line, q.side): q for q in quotes if q.book == book}
    out = []
    for offer in backtest._offers(quotes, only):
        for side, chance in (("over", offer["consensus_probability"]), ("under", 1 - offer["consensus_probability"])):
            q = at_book.get((offer["event_id"], offer["market"], offer["player_key"], offer["line"], side))
            if q is None:
                continue
            out.append(dict(sport=offer["sport"], event_id=offer["event_id"], event_name=offer["event_name"],
                            commence_time=offer["commence_time"], player=offer["player"], player_key=offer["player_key"],
                            market=offer["market"], line=float(offer["line"]), side=side, probability=float(chance), book=book,
                            price=q.price, decimal=q.decimal, link=q.link))
    return out


def multiplied_decimal(legs: list[dict]) -> float:
    """The payout if the book simply multiplies the legs' prices. Books do that for legs from different games;
    a same-game parlay is repriced by the book and usually pays less."""
    return float(np.prod([leg["decimal"] for leg in legs]))


# --- one link for the whole parlay --------------------------------------------------------------------------
def _query(link) -> tuple[str, list[tuple[str, str]]] | None:
    if not isinstance(link, str) or not link.startswith("https://"):
        return None
    try:
        url = urlsplit(link)
        return f"{url.scheme}://{url.netloc}{url.path}", parse_qsl(url.query, keep_blank_values=True)
    except ValueError:
        return None


def parlay_link(book: str, links: list) -> str | None:
    """One link that opens every leg in the book's betslip, built from each leg's own stored link. None when a leg
    has no usable link or the book's format is not known."""
    parsed = [_query(link) for link in links]
    if len(parsed) < 2 or any(p is None for p in parsed):
        return None
    if book == "fanduel":
        try:
            from nfl_pipeline.fanduel_links import parlay_betslip_url
            return parlay_betslip_url(links)
        except Exception:
            return None
    if book == "draftkings":
        outcomes = [dict(query).get("outcomes") for _, query in parsed]
        if not all(outcomes) or len(set(outcomes)) != len(outcomes):
            return None
        return "https://sportsbook.draftkings.com/?outcomes=" + "+".join(quote(o, safe="") for o in outcomes)
    if book == "espnbet":
        fields = ("market_selection_id", "odds_numerator", "odds_denominator")
        query = []
        for index, (_, pairs) in enumerate(parsed):
            leg = {key.split("[")[0]: value for key, value in pairs}
            if not leg.get("market_selection_id"):
                return None
            query += [(f"{field}[{index}]", leg[field]) for field in fields if field in leg]
        return parsed[0][0] + "?" + urlencode(query, safe="[]")
    return None


# --- the stored projection for a leg ----------------------------------------------------------------------------
def projection_for(props: pd.DataFrame | None, leg: dict) -> tuple[float | None, float | None]:
    """(projected value, the model's chance that this leg wins) from the Predictions reader's player props.
    The NBA chance is worked out at the leg's own line; the NFL one is stored only for the line it was made at."""
    if props is None or props.empty or "player_key" not in props:
        return None, None
    market = leg["market"].removesuffix("_alternate")
    found = props[(props.sport == leg["sport"]) & (props.player_key == leg["player_key"]) & (props.market == market)]
    if found.empty:
        return None, None
    row = found.iloc[0]
    over = None
    if leg["sport"] == "nba" and pd.notna(row.disp_a):
        try:
            from nba_props.model import p_over
            over = float(p_over(np.array([float(row.projection)]), np.array([leg["line"]]), float(row.disp_a), float(row.disp_b))[0])
        except Exception:                   # the NBA pipeline is optional
            over = None
    elif pd.notna(row.model_over) and pd.notna(row.line) and float(row.line) == leg["line"]:
        over = float(row.model_over)
    chance = None if over is None else over if leg["side"] == "over" else 1 - over
    return (None if pd.isna(row.projection) else float(row.projection)), chance


def american(decimal: float) -> int:
    return int(odds_math.decimal_to_american(decimal))
