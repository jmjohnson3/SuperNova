"""NBA probability candidate: the live projections with their uncertainty measured on later, unseen games.

The live model (nba-props-v2) sets each statistic's spread from the games it was trained on, which makes it look
surer than it is. This candidate keeps the live model's projected means and replaces the rest with what the
forecasts actually did afterwards. For a game on day D it uses only forecasts and results from days before D:

  scale       results / projections for that statistic (is the mean too high or too low?)
  dispersion  the spread of results around the scaled projection, as variance = a * mean + b * mean^2

With too few earlier player-games for a statistic it falls back to the live model's own numbers. Nothing here
changes the live model or any pinned file; it reads stored projections and box scores (read-only) and is scored
beside the live model in `model_compare`.

`workload_split` asks how much of the projection error is playing time: it compares each error with what it
would have been had the player's minutes been known exactly.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_sniper.providers import saved_quotes
from sports_sniper.providers.models import EASTERN, NBA_STAT

VERSION = "nba-props-v2-oos-dispersion"
MIN_PAIRS = 400             # earlier player-games of a statistic needed before its own scale and dispersion are used
RECENT_GAMES = 5            # games behind the "usual minutes" a projection is assumed to have had in mind
STATS = tuple(NBA_STAT.values())


_LOADED: dict = {}          # the last season read, so one run does not read it twice


@dataclass(frozen=True)
class Params:
    scale: float
    a: float
    b: float
    pairs: int
    fitted: bool            # False: too few earlier games, the live model's own numbers


def load(dsn: str | None, days: list, connect=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(projections, results). Projections: every stored run for the season so far up to the last day asked for.
    Results: this season's box scores with a name key matching the projections'."""
    empty = (pd.DataFrame(columns=["day", "name_key", "stat", "run_at", "tip", "mu", "a", "b"]),
             pd.DataFrame(columns=["day", "name_key", "minutes", *STATS]))
    dsn = dsn or saved_quotes.default_dsn()
    if dsn is None or not days:
        return empty
    remembered = _LOADED.get((dsn, max(days))) if connect is None else None
    if remembered is not None:               # the report reads the same season twice in one run
        return remembered
    if connect is None:
        import psycopg2
        connect = lambda d: psycopg2.connect(d, options="-c timezone=UTC")      # noqa: E731
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for table in ("nbaprops.projections", "nbaprops.player_games"):
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    return empty
            cur.execute("""SELECT game_date, name_key, stat, run_at_utc, tip_utc, mu, disp_a, disp_b FROM nbaprops.projections
                           WHERE game_date <= %s ORDER BY game_date, name_key, stat, run_at_utc""", (max(days),))
            projections = pd.DataFrame(cur.fetchall(), columns=empty[0].columns)
            cur.execute("""SELECT game_date, player_name, minutes, pts, reb, ast, fg3m FROM nbaprops.player_games
                           WHERE season = (SELECT max(season) FROM nbaprops.player_games) AND game_date <= %s""", (max(days),))
            results = pd.DataFrame(cur.fetchall(), columns=["day", "player", "minutes", *STATS])
    finally:
        conn.close()
    try:
        from nba_props.model import match_name
    except Exception:                       # the NBA pipeline is optional
        return empty
    for column in ("mu", "a", "b"):
        projections[column] = pd.to_numeric(projections[column], errors="coerce")
    projections["run_at"] = pd.to_datetime(projections.run_at, utc=True)
    projections["tip"] = pd.to_datetime(projections.tip, utc=True)
    for column in ("minutes", *STATS):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results["name_key"] = results.player.fillna("").map(match_name)
    out = projections, results[["day", "name_key", "minutes", *STATS]]
    _LOADED.clear()
    _LOADED[(dsn, max(days))] = out
    return out


def pairs(projections: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per player, statistic and game he played: the last projection made before the game against what
    he did. Columns day, name_key, stat, mu, a, b, actual, minutes."""
    columns = ["day", "name_key", "stat", "mu", "a", "b", "actual", "minutes"]
    if projections.empty or results.empty:
        return pd.DataFrame(columns=columns)
    before = projections[projections.tip.isna() | (projections.run_at < projections.tip)]
    last = before.drop_duplicates(["day", "name_key", "stat"], keep="last")
    played = results[results.minutes > 0].drop_duplicates(["day", "name_key"])
    long = played.melt(id_vars=["day", "name_key", "minutes"], value_vars=list(STATS), var_name="stat", value_name="actual")
    return last.merge(long, on=["day", "name_key", "stat"], how="inner")[columns].dropna(subset=["mu", "actual"])


def fit(sample: pd.DataFrame) -> Params | None:
    """Scale and dispersion from projection/result pairs of one statistic; None with fewer than MIN_PAIRS."""
    if len(sample) < MIN_PAIRS:
        return None
    mu, actual = sample.mu.to_numpy(dtype=float), sample.actual.to_numpy(dtype=float)
    scale = float(actual.sum() / mu.sum()) if mu.sum() > 0 else 1.0
    mean = np.clip(mu * scale, 0.05, None)
    squared = (actual - mean) ** 2
    weight = 1 / mean                                   # squared errors grow with the mean; this evens their say
    design = np.column_stack([mean, mean ** 2]) * weight[:, None]
    a, b = np.linalg.lstsq(design, squared * weight, rcond=None)[0]
    if b < 0 or a <= 0:                                 # no room for a mean-squared term: a single spread factor
        a, b = float((squared / mean).mean()), 0.0
    return Params(scale=scale, a=float(max(a, 0.2)), b=float(max(b, 0.0)), pairs=len(sample), fitted=True)


def params_by_day(history: pd.DataFrame, days: list) -> dict[tuple, Params]:
    """{(day, stat): what the candidate uses for a game on that day}, each from pairs of earlier days only."""
    out: dict[tuple, Params] = {}
    for stat in STATS:
        mine = history[history.stat == stat]
        for day in days:
            earlier = mine[mine.day < day]
            found = fit(earlier)
            if found is None:
                a = float(earlier.a.iloc[-1]) if len(earlier) else float(mine.a.iloc[0]) if len(mine) else 1.0
                b = float(earlier.b.iloc[-1]) if len(earlier) else float(mine.b.iloc[0]) if len(mine) else 0.0
                found = Params(scale=1.0, a=a, b=b, pairs=len(earlier), fitted=False)
            out[(day, stat)] = found
    return out


def over_probabilities(dsn: str | None, offers: pd.DataFrame, connect=None, data=None) -> tuple[list[float | None], dict]:
    """(the candidate's chance of the over for each offer, the parameters it used by (day, stat)). None where the
    live model has no forecast made before the price was seen. `data` supplies (projections, results) directly."""
    out: list[float | None] = [None] * len(offers)
    wanted = offers[(offers.sport == "nba") & offers.market.isin(NBA_STAT)] if len(offers) else offers
    if wanted.empty:
        return out, {}
    try:
        from nba_props.model import match_name, p_over
    except Exception:
        return out, {}
    day_of = wanted.commence_time.map(lambda t: t.astimezone(EASTERN).date())
    days = sorted(set(day_of))
    projections, results = data if data is not None else load(dsn, days, connect)
    if projections.empty:
        return out, {}
    params = params_by_day(pairs(projections, results), days)
    runs = {key: (group.run_at.tolist(), group.mu.tolist()) for key, group in projections.groupby(["day", "name_key", "stat"], sort=False)}
    position = {index: n for n, index in enumerate(offers.index)}
    for offer, day in zip(wanted.itertuples(), day_of):
        stat = NBA_STAT[offer.market]
        found = runs.get((day, match_name(offer.player), stat))
        if found is None:
            continue
        before = bisect_left(found[0], pd.Timestamp(offer.observed_at))
        if before == 0:
            continue
        p = params[(day, stat)]
        out[position[offer.Index]] = float(p_over(np.array([found[1][before - 1] * p.scale]), np.array([float(offer.line)]), p.a, p.b)[0])
    return out, params


def spread_check(history: pd.DataFrame) -> pd.DataFrame:
    """Is the live model's stated spread right? Per statistic: the average of (error^2 / stated variance). 1 means
    the stated uncertainty matches what happened; above 1 means the model was surer than it had reason to be."""
    rows = []
    for stat, group in history.groupby("stat"):
        mean = np.clip(group.mu.to_numpy(dtype=float), 0.05, None)
        stated = group.a.to_numpy(dtype=float) * mean + group.b.to_numpy(dtype=float) * mean ** 2
        error = group.actual.to_numpy(dtype=float) - mean
        rows.append(dict(stat=stat, player_games=len(group), bias=float(error.mean()), result_over_projection=float(group.actual.sum() / group.mu.sum()),
                         spread_ratio=float((error ** 2 / stated).mean()), mae=float(np.abs(error).mean())))
    return pd.DataFrame(rows, columns=["stat", "player_games", "bias", "result_over_projection", "spread_ratio", "mae"])


def workload_split(history: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """How much of each statistic's projection error is playing time.

    The projection is taken to assume the player's usual minutes (his average over his previous RECENT_GAMES
    games). Rescaling it to the minutes he actually played gives the error that would have remained had minutes
    been known exactly; the difference is the part owed to minutes. A proxy, since the live model stores no
    minutes forecast of its own, but the same proxy is applied to every statistic.
    """
    columns = ["stat", "player_games", "mae", "mae_minutes_known", "minutes_share", "minutes_mae"]
    if history.empty or results.empty:
        return pd.DataFrame(columns=columns)
    played = results[results.minutes > 0].sort_values("day")
    usual = played.groupby("name_key").minutes.transform(lambda m: m.shift(1).rolling(RECENT_GAMES, min_periods=3).mean())
    expected = played.assign(usual=usual)[["day", "name_key", "usual"]].dropna()
    joined = history.merge(expected, on=["day", "name_key"], how="inner")
    rows = []
    for stat, group in joined.groupby("stat"):
        error = (group.actual - group.mu).abs()
        known = (group.actual - group.mu * group.minutes / group.usual).abs()
        rows.append(dict(stat=stat, player_games=len(group), mae=float(error.mean()), mae_minutes_known=float(known.mean()),
                         minutes_share=float(1 - known.mean() / error.mean()) if error.mean() else None,
                         minutes_mae=float((group.minutes - group.usual).abs().mean())))
    return pd.DataFrame(rows, columns=columns)


def diagnostics(dsn: str | None, days: list, connect=None) -> dict:
    """What the report says about the candidate: the live model's spread against results, the workload split, and
    the scale and dispersion the candidate would use for a game after the last day."""
    projections, results = load(dsn, days, connect)
    history = pairs(projections, results)
    if history.empty:
        return {}
    latest = {stat: fit(history[history.stat == stat]) for stat in STATS}
    return dict(spread=spread_check(history), workload=workload_split(history, results), days=int(history.day.nunique()),
                params=pd.DataFrame([dict(stat=stat, scale=p.scale, a=p.a, b=p.b, pairs=p.pairs,
                                          live_a=float(history[history.stat == stat].a.iloc[-1])) for stat, p in latest.items() if p]))
