"""Adapters: probabilities from the repository's existing prediction models. Read-only.

One adapter per model. Each takes offers (rows with sport, commence_time, player, market, line, observed_at) and
returns the model's chance of the over for each, or None when it has no forecast made before the price was seen.
A forecast made after the price was seen is never used.

NBA: nbaprops.projections (model nba-props-v2: a mean and its dispersion per player and statistic), turned into
a chance at the line with nba_props.model.p_over.

NFL: bets.nfl_player_prop_predictions.raw_over_probability, the pinned production model's own chance before it is
blended toward FanDuel's price. Only receptions and passing touchdowns are used: the yardage projections are
anchored to the line itself, so their raw chance is 50% by construction and says nothing.

MLB: no adapter. The MLB pipeline has had no data since its statistics feed lapsed, and the season is over.
"""
from __future__ import annotations

from bisect import bisect_left
from zoneinfo import ZoneInfo

import pandas as pd

from sports_sniper.normalize import player_key
from sports_sniper.providers import saved_quotes

EASTERN = ZoneInfo("America/New_York")
NBA_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m"}
NBA_MODEL = "nba-props-v2"
NFL_STAT = {"player_receptions": "receptions", "player_pass_tds": "passing_tds"}
NFL_MODEL = "nfl-production-raw"


def _connect(dsn):
    import psycopg2
    return psycopg2.connect(dsn, options="-c timezone=UTC")


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")


def _rows(dsn, table: str, sql: str, params: tuple, connect) -> list:
    conn = (connect or _connect)(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if cur.fetchone()[0] is None:
                return []
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def _latest_before(runs: dict, key: tuple, seen) -> object | None:
    """The newest stored value for `key` made strictly before `seen`."""
    if key not in runs:
        return None
    times, values = runs[key]
    before = bisect_left(times, pd.Timestamp(seen))
    return values[before - 1] if before else None


def nba_over_probabilities(dsn: str | None, offers: pd.DataFrame, connect=None) -> list[float | None]:
    """The NBA projection model's chance of the over for each offer; None where it has no earlier forecast."""
    out: list[float | None] = [None] * len(offers)
    dsn = dsn or saved_quotes.default_dsn()
    wanted = offers[(offers.sport == "nba") & offers.market.isin(NBA_STAT)] if len(offers) else offers
    if dsn is None or wanted.empty:
        return out
    try:
        import numpy as np
        from nba_props.model import match_name, p_over
    except Exception:                       # the NBA pipeline is optional
        return out
    days = sorted({t.astimezone(EASTERN).date() for t in wanted.commence_time})
    rows = _rows(dsn, "nbaprops.projections",
                 """SELECT game_date, name_key, stat, run_at_utc, mu, disp_a, disp_b FROM nbaprops.projections
                    WHERE game_date = ANY(%s) ORDER BY game_date, name_key, stat, run_at_utc""", (days,), connect)
    runs: dict[tuple, tuple[list, list]] = {}
    for day, name, stat, run_at, mu, a, b in rows:
        times, values = runs.setdefault((day, name, stat), ([], []))
        times.append(_utc(run_at))
        values.append((float(mu), float(a), float(b)))
    position = {index: n for n, index in enumerate(offers.index)}
    for offer in wanted.itertuples():
        key = (offer.commence_time.astimezone(EASTERN).date(), match_name(offer.player), NBA_STAT[offer.market])
        found = _latest_before(runs, key, offer.observed_at)
        if found is not None:
            mu, a, b = found
            out[position[offer.Index]] = float(p_over(np.array([mu]), np.array([float(offer.line)]), a, b)[0])
    return out


def nfl_over_probabilities(dsn: str | None, offers: pd.DataFrame, connect=None) -> list[float | None]:
    """The NFL production model's own chance of the over (receptions, passing touchdowns), from a forecast stored
    for that exact line before the price was seen."""
    out: list[float | None] = [None] * len(offers)
    dsn = dsn or saved_quotes.default_dsn()
    wanted = offers[(offers.sport == "nfl") & offers.market.isin(NFL_STAT)] if len(offers) else offers
    if dsn is None or wanted.empty:
        return out
    days = sorted({t.astimezone(EASTERN).date() for t in wanted.commence_time})
    rows = _rows(dsn, "bets.nfl_player_prop_predictions",
                 """SELECT game_date_et, COALESCE(offer_player_name, player_name), stat, line, created_at_utc, raw_over_probability
                    FROM bets.nfl_player_prop_predictions
                    WHERE game_date_et = ANY(%s) AND stat = ANY(%s) AND raw_over_probability IS NOT NULL AND line IS NOT NULL
                    ORDER BY created_at_utc""", (days, sorted(set(NFL_STAT.values()))), connect)
    runs: dict[tuple, tuple[list, list]] = {}
    for day, name, stat, line, made, probability in rows:
        times, values = runs.setdefault((day, player_key(name or ""), stat, float(line)), ([], []))
        times.append(_utc(made))
        values.append(float(probability))
    position = {index: n for n, index in enumerate(offers.index)}
    for offer in wanted.itertuples():
        key = (offer.commence_time.astimezone(EASTERN).date(), player_key(offer.player), NFL_STAT[offer.market], float(offer.line))
        found = _latest_before(runs, key, offer.observed_at)
        if found is not None:
            out[position[offer.Index]] = found
    return out


ADAPTERS = ((NBA_MODEL, nba_over_probabilities), (NFL_MODEL, nfl_over_probabilities))


def over_probabilities(dsn: str | None, offers: pd.DataFrame, adapters=ADAPTERS) -> tuple[list[float | None], list[str | None]]:
    """(chance of the over, model version) for each offer from whichever adapter covers it. An adapter that
    cannot reach its data covers nothing; it never stops the others."""
    values: list[float | None] = [None] * len(offers)
    versions: list[str | None] = [None] * len(offers)
    for version, adapter in adapters:
        try:
            got = adapter(dsn, offers)
        except Exception:
            continue
        for n, value in enumerate(got):
            if value is not None and values[n] is None:
                values[n], versions[n] = float(value), version
    return values, versions
