"""Breakout radar: players whose recent opportunity is clearly above their own season baseline.

Opportunity, not results: minutes and shot attempts in the NBA; targets, carries and snap share in the NFL. A
player is flagged when the average of his last few games is above the average of his earlier games this season by
both an absolute amount and a share, and every one of those recent games was above the baseline (so one unusual
game cannot do it). Everything said about a player is built from those numbers. Injuries, lineups and news are
not read here, so nothing is claimed about them.

`validate` asks the only question that matters for betting: on settled lines, did flagged players go over more
often than everyone else, and did betting their overs at the book's price make money? A signal the price already
includes is information about the player, not an edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from sports_sniper import performance
from sports_sniper.normalize import player_key
from sports_sniper.providers import results, saved_quotes


@dataclass(frozen=True)
class Signal:
    metric: str                 # column in the history
    label: str                  # how it is described
    unit: str
    min_gain: float             # recent minus baseline, at least this much
    min_share: float            # and at least this share of the baseline
    markets: tuple[str, ...]    # props the signal bears on


RULES = {
    "nba": dict(recent=3, baseline=4, signals=(
        Signal("minutes", "minutes", "a game", 4.0, 0.15, ("player_points", "player_rebounds", "player_assists", "player_threes")),
        Signal("fga", "shot attempts", "a game", 3.0, 0.25, ("player_points", "player_threes")))),
    "nfl": dict(recent=2, baseline=3, signals=(
        Signal("targets", "targets", "a game", 2.0, 0.30, ("player_receptions", "player_reception_yds")),
        Signal("carries", "carries", "a game", 4.0, 0.30, ("player_rush_yds",)),
        Signal("snap_share", "share of offensive snaps", "", 0.15, 0.20, ("player_receptions", "player_reception_yds", "player_rush_yds")))),
}
COLUMNS = ["sport", "player_key", "player", "team", "metric", "label", "baseline", "recent", "gain", "share", "strength",
           "baseline_games", "last_game", "markets", "explanation"]


def load_history(dsn: str | None, connect=None) -> dict[str, pd.DataFrame]:
    """This season's game logs per sport from the existing pipelines' Postgres (read-only). A sport whose table
    is missing is simply absent."""
    dsn = dsn or saved_quotes.default_dsn()
    out: dict[str, pd.DataFrame] = {}
    if dsn is None:
        return out
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            queries = {
                "nba": ("nbaprops.player_games",
                        """SELECT game_date, player_name, team, minutes, fga FROM nbaprops.player_games
                           WHERE season = (SELECT max(season) FROM nbaprops.player_games) AND minutes > 0""",
                        ["day", "player", "team", "minutes", "fga"]),
                "nfl": ("raw.nfl_player_gamelogs",
                        """SELECT g.game_date_et, COALESCE(r.player_name, g.player_name), g.team_abbr, g.targets, g.carries,
                                  COALESCE(g.offense_snap_share, g.snap_share)
                           FROM raw.nfl_player_gamelogs g
                           LEFT JOIN LATERAL (SELECT player_name FROM raw.nfl_rosters r WHERE r.player_id = g.player_id
                                              ORDER BY r.season DESC, r.week DESC NULLS LAST LIMIT 1) r ON TRUE
                           WHERE g.season = (SELECT max(season) FROM raw.nfl_player_gamelogs) AND g.source LIKE 'nflverse%%'""",
                        ["day", "player", "team", "targets", "carries", "snap_share"]),
            }
            for sport, (table, sql, columns) in queries.items():
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    continue
                cur.execute(sql)
                frame = pd.DataFrame(cur.fetchall(), columns=columns)
                for column in columns[3:]:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame["player_key"] = frame.player.fillna("").map(player_key)
                out[sport] = frame[frame.player_key != ""].sort_values("day").reset_index(drop=True)
    finally:
        conn.close()
    return out


def signals(history: pd.DataFrame, sport: str, as_of: date) -> pd.DataFrame:
    """Players flagged using only games before `as_of`. One row per player and signal, strongest first."""
    rules = RULES.get(sport)
    if rules is None or history is None or history.empty:
        return pd.DataFrame(columns=COLUMNS)
    recent_n, baseline_n = rules["recent"], rules["baseline"]
    past = history[history.day < as_of]
    rows = []
    for key, games in past.groupby("player_key", sort=False):
        if len(games) < recent_n + baseline_n:
            continue
        early, late = games.iloc[:-recent_n], games.iloc[-recent_n:]
        for signal in rules["signals"]:
            before, after = early[signal.metric].dropna(), late[signal.metric].dropna()
            if len(before) < baseline_n or len(after) < recent_n:
                continue
            base, now = float(before.mean()), float(after.mean())
            gain = now - base
            if base <= 0 or gain < signal.min_gain or gain < signal.min_share * base or float(after.min()) <= base:
                continue
            spread = max(float(before.std(ddof=1)) if len(before) > 1 else 0.0, 0.10 * base)
            last = games.iloc[-1]
            number = (lambda v: f"{v:.0%}") if signal.metric == "snap_share" else (lambda v: f"{v:.1f}")
            unit = f" {signal.unit}" if signal.unit else ""
            text = (f"{last.player}: {signal.label} {number(now)}{unit} over his last {recent_n} games against {number(base)}{unit} "
                    f"in his {len(before)} games before that ({gain / base:+.0%}); each of the last {recent_n} was above that average.")
            rows.append(dict(sport=sport, player_key=key, player=last.player, team=last.team, metric=signal.metric, label=signal.label,
                             baseline=base, recent=now, gain=gain, share=gain / base, strength=gain / (spread / recent_n ** 0.5),
                             baseline_games=len(before), last_game=last.day, markets=signal.markets, explanation=text))
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def flag_offers(offers: pd.DataFrame, history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mark each line whose player had a signal bearing on that market, using only games before the line's game."""
    offers = offers.copy()
    offers["breakout"] = False
    if offers.empty:
        return offers
    day = offers.commence_time.dt.tz_convert(results.EASTERN).dt.date
    for sport, frame in history.items():
        for today in sorted(day[offers.sport == sport].unique()):
            found = signals(frame, sport, today)
            wanted = {(s.player_key, market) for s in found.itertuples() for market in s.markets}
            rows = (offers.sport == sport) & (day == today)
            offers.loc[rows, "breakout"] = [(k, m.removesuffix("_alternate")) in wanted
                                            for k, m in zip(offers.loc[rows, "player_key"], offers.loc[rows, "market"])]
    return offers


def validate(offers: pd.DataFrame, history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flagged lines against the rest: how often the over won, and what betting every over at the book's price
    returned. One row per sport and group."""
    flagged = flag_offers(offers, history)
    settled = flagged[flagged.status.isin(performance.SETTLED)] if len(flagged) else flagged
    rows = []
    for (sport, hit), group in (settled.groupby(["sport", "breakout"]) if len(settled) else []):
        decided = group[group.status.isin(performance.DECIDED)]
        profit = pd.Series([d - 1 if s == "won" else -1.0 if s == "lost" else 0.0 for s, d in zip(group.status, group.over_decimal)],
                           index=group.index)
        rows.append(dict(sport=sport, breakout=bool(hit), lines=len(group), players=group.player_key.nunique(),
                         over_rate=float((decided.status == "won").mean()) if len(decided) else None,
                         market_over=float(group.consensus_probability.mean()), over_roi=float(profit.mean()),
                         over_roi_se=performance.clustered_se(profit, group.event_id)))
    return pd.DataFrame(rows, columns=["sport", "breakout", "lines", "players", "over_rate", "market_over", "over_roi", "over_roi_se"])


def radar(history: dict[str, pd.DataFrame], upcoming: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Today's flagged players. `upcoming` is the latest stored prices for games that have not started; a flagged
    player with a line there gets the best over price for each market his signal bears on."""
    parts = [signals(frame, sport, as_of) for sport, frame in history.items()]
    found = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    found["props"] = [[] for _ in range(len(found))]
    found["playing"] = False
    if found.empty or upcoming is None or upcoming.empty:
        return found
    overs = upcoming[upcoming.side == "over"]
    best = overs.sort_values("price", ascending=False).drop_duplicates(["sport", "player_key", "market", "line_key"])
    by_player = {key: group for key, group in best.groupby(["sport", "player_key"])}
    props, playing = [], []
    for s in found.itertuples():
        lines = by_player.get((s.sport, s.player_key))
        playing.append(lines is not None)
        props.append([] if lines is None else [dict(market=p.market, line=p.line, price=int(p.price), book=p.book, link=p.link,
                                                    event_name=p.event_name, commence_time=p.commence_time)
                                               for p in lines.itertuples() if p.market in s.markets])
    found["props"], found["playing"] = props, playing
    return found.sort_values(["playing", "strength"], ascending=[False, False]).reset_index(drop=True)
