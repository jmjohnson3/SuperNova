"""Parlay Lab: the chance that several bets in one game all win, allowing for how they move together.

Multiplying the legs' chances assumes they are independent. In one game they are not: a player's points and
assists rise together, and two teammates share the same possessions. This module measures those links from game
logs and uses them when working out a parlay's chance.

Links (`correlations`): each player's statistic in a game is turned into a surprise, how far it was from his own
season average in units of his own spread. The correlation of surprises is then measured for the same player's
two statistics, for teammates, and for opponents, by statistic pair.

Joint chance (`joint_probability`): a Gaussian copula. Each leg wins when a standard normal draw lands in the
slice matching its own chance; the draws are correlated by the measured links (reversed for an under); the share
of simulated games in which every leg wins is the parlay's chance. A book's same-game parlay price is not in the
odds feed, so the price to compare with is typed in. Research only.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from sports_sniper.normalize import player_key
from sports_sniper.providers import saved_quotes

MIN_GAMES = 8               # a player-season needs this many games before its surprises are used
MIN_PAIRS = 200             # a link measured on fewer pairs is treated as unknown (zero)
TOP_PLAYERS = 8             # per team and game, by playing time or volume, for teammate and opponent pairs
STATS = {"nba": {"pts": "player_points", "reb": "player_rebounds", "ast": "player_assists", "fg3m": "player_threes"},
         "nfl": {"receiving_yards": "player_reception_yds", "receptions": "player_receptions", "rushing_yards": "player_rush_yds",
                 "passing_yards": "player_pass_yds", "passing_tds": "player_pass_tds"}}
QUERIES = {
    "nba": ("nbaprops.player_games",
            """SELECT season, game_id, team, player_name, minutes, pts, reb, ast, fg3m FROM nbaprops.player_games WHERE minutes >= 10""",
            ["season", "game_id", "team", "player", "weight", "pts", "reb", "ast", "fg3m"]),
    "nfl": ("raw.nfl_player_gamelogs",
            """SELECT season, game_id, team_abbr, player_id, COALESCE(targets, 0) + COALESCE(carries, 0) + COALESCE(pass_attempts, 0),
                      receiving_yards, receptions, rushing_yards, passing_yards, passing_tds
               FROM raw.nfl_player_gamelogs WHERE source LIKE 'nflverse%%' AND season >= 2021
                 AND COALESCE(targets, 0) + COALESCE(carries, 0) + COALESCE(pass_attempts, 0) >= 3""",
            ["season", "game_id", "team", "player", "weight", "receiving_yards", "receptions", "rushing_yards", "passing_yards", "passing_tds"]),
}


def load_logs(dsn: str | None, connect=None) -> dict[str, pd.DataFrame]:
    """Several seasons of game logs per sport (read-only). A sport whose table is missing is absent."""
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
            for sport, (table, sql, columns) in QUERIES.items():
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    continue
                cur.execute(sql)
                frame = pd.DataFrame(cur.fetchall(), columns=columns)
                for column in columns[4:]:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                out[sport] = frame
    finally:
        conn.close()
    return out


def surprises(logs: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Each statistic as (value - the player's season average) / his season spread, for player-seasons with enough games."""
    out = logs[["season", "game_id", "team", "player", "weight"]].copy()
    group = logs.groupby(["season", "player"])
    enough = group[stats[0]].transform("size") >= MIN_GAMES
    for stat in stats:
        mean, spread = group[stat].transform("mean"), group[stat].transform("std")
        out[stat] = ((logs[stat] - mean) / spread).where(enough & (spread > 0))
    return out


def correlations(logs: pd.DataFrame, sport: str) -> dict[str, dict]:
    """{"relation|stat_a|stat_b": {"r": correlation, "n": pairs}} for same player, teammates and opponents."""
    stats = list(STATS[sport])
    z = surprises(logs, stats)
    sums: dict[tuple, list] = {}

    def add(relation: str, a: str, b: str, x: np.ndarray, y: np.ndarray) -> None:
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.any():
            first, second = sorted((a, b))
            if first != a:
                x, y = y, x
            total = sums.setdefault((relation, first, second), [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            x, y = x[ok], y[ok]
            for n, value in enumerate((len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum())):
                total[n] += value
    for a, b in combinations(stats, 2):
        add("same", a, b, z[a].to_numpy(), z[b].to_numpy())
    top = z.sort_values("weight", ascending=False).groupby(["game_id", "team"], sort=False).head(TOP_PLAYERS)
    for _, game in top.groupby("game_id", sort=False):
        teams = game.team.to_numpy()
        values = {s: game[s].to_numpy() for s in stats}
        i, j = np.triu_indices(len(game), k=1)
        mates = teams[i] == teams[j]
        for relation, mask in (("teammates", mates), ("opponents", ~mates)):
            if not mask.any():
                continue
            for a in stats:
                for b in stats:
                    if a <= b:
                        add(relation, a, b, values[a][i[mask]], values[b][j[mask]])
                        if a != b:
                            add(relation, a, b, values[a][j[mask]], values[b][i[mask]])
    out = {}
    for (relation, a, b), (n, sx, sy, sxx, syy, sxy) in sums.items():
        spread = np.sqrt(max(sxx - sx * sx / n, 0) * max(syy - sy * sy / n, 0))
        if n >= MIN_PAIRS and spread > 0:
            first, second = sorted((STATS[sport][a], STATS[sport][b]))
            out[f"{relation}|{first}|{second}"] = dict(r=float((sxy - sx * sy / n) / spread), n=int(n))
    return out


def build(dsn: str | None, connect=None) -> dict:
    """{sport: correlations} from the stored game logs."""
    return {sport: correlations(frame, sport) for sport, frame in load_logs(dsn, connect).items() if sport in STATS}


def save(table: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table), encoding="utf-8")


def load(path: Path | None) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def default_path(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite:///") or ":memory:" in database_url:
        return None
    return Path(database_url.removeprefix("sqlite:///")).with_name("correlations.json")


def link(table: dict, sport: str, a: dict, b: dict) -> tuple[float, str]:
    """(correlation between two legs winning, why). Legs are dicts with event_id, player_key, team, market, side."""
    if a["event_id"] != b["event_id"]:
        return 0.0, "different games: treated as independent"
    same_player = a["player_key"] == b["player_key"]
    if same_player and a["market"] == b["market"]:
        raise ValueError("two legs on the same player and statistic cannot be combined here")
    if not same_player and (not a.get("team") or not b.get("team")):
        return 0.0, "team unknown: treated as independent"
    relation = "same" if same_player else "teammates" if a["team"] == b["team"] else "opponents"
    first, second = sorted((a["market"].removesuffix("_alternate"), b["market"].removesuffix("_alternate")))
    found = table.get(sport, {}).get(f"{relation}|{first}|{second}")
    if found is None:
        return 0.0, f"no measured link for {relation}: treated as independent"
    sign = 1.0 if a["side"] == b["side"] else -1.0          # an over and an under pull in opposite directions
    words = {"same": "same player", "teammates": "teammates", "opponents": "opponents"}[relation]
    return sign * found["r"], f"{words}, measured on {found['n']:,} pairs"


def joint_probability(probabilities: list[float], matrix: np.ndarray, draws: int = 200_000, seed: int = 7) -> float:
    """The chance every leg wins under a Gaussian copula with the given correlation matrix."""
    matrix = np.asarray(matrix, dtype=float)
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    fixed = vectors @ np.diag(np.clip(values, 1e-6, None)) @ vectors.T        # nearest usable matrix when the links disagree
    scale = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(scale, scale)
    normal = NormalDist()
    cuts = np.array([normal.inv_cdf(min(max(p, 1e-6), 1 - 1e-6)) for p in probabilities])
    sample = np.random.default_rng(seed).standard_normal((draws, len(probabilities))) @ np.linalg.cholesky(fixed).T
    return float((sample < cuts).all(axis=1).mean())


def evaluate(legs: list[dict], table: dict, sport: str, price: int | None = None) -> dict:
    """A parlay's chance with and without the measured links, its fair odds, and its EV at a typed-in price.
    Each leg needs probability, event_id, player_key, team, market and side."""
    if len(legs) < 2:
        raise ValueError("a parlay needs at least two legs")
    n = len(legs)
    matrix, notes = np.eye(n), []
    for i, j in combinations(range(n), 2):
        r, why = link(table, legs[i].get("sport", sport), legs[i], legs[j])
        matrix[i, j] = matrix[j, i] = r
        notes.append(dict(a=i, b=j, correlation=r, why=why))
    chances = [float(leg["probability"]) for leg in legs]
    independent = float(np.prod(chances))
    joint = joint_probability(chances, matrix) if any(note["correlation"] for note in notes) else independent
    out = dict(independent=independent, joint=joint, links=notes, fair_decimal=1 / joint if joint > 0 else None,
               independent_decimal=1 / independent if independent > 0 else None, ev=None)
    if price is not None:
        decimal = 1 + (100 / -price if price < 0 else price / 100)
        out.update(decimal=decimal, ev=joint * decimal - 1, ev_if_independent=independent * decimal - 1)
    return out


def teams_from(history: dict[str, pd.DataFrame]) -> dict[tuple, str]:
    """{(sport, player key): team in his most recent game} from the breakout radar's game logs."""
    out = {}
    for sport, frame in history.items():
        latest = frame.sort_values("day").drop_duplicates("player_key", keep="last")
        out.update({(sport, key): team for key, team in zip(latest.player_key, latest.team)})
    return out


__all__ = ["build", "correlations", "evaluate", "joint_probability", "link", "load", "save", "default_path", "teams_from", "player_key"]
