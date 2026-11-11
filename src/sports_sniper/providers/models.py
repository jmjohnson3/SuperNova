"""Adapters: probabilities from the repository's existing prediction models. Read-only.

One adapter per model. Each takes offers (rows with sport, commence_time, player, market, line, observed_at) and
returns the model's chance of the over for each, or None when it has no forecast made before the price was seen.
A forecast made after the price was seen is never used.

NBA: nbaprops.projections (model nba-props-v2: a mean and its dispersion per player and statistic), turned into
a chance at the line with nba_props.model.p_over. NFL forecasts are not wired in: the stored probabilities in
bets.nfl_player_prop_predictions are already blended toward FanDuel's own price, so they are not a separate
source to compare with the market.
"""
from __future__ import annotations

from bisect import bisect_left
from zoneinfo import ZoneInfo

import pandas as pd

from sports_sniper.providers import saved_quotes

EASTERN = ZoneInfo("America/New_York")
NBA_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m"}
NBA_MODEL = "nba-props-v2"


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
    if connect is None:
        import psycopg2
        connect = lambda d: psycopg2.connect(d, options="-c timezone=UTC")      # noqa: E731
    days = sorted({t.astimezone(EASTERN).date() for t in wanted.commence_time})
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('nbaprops.projections')")
            if cur.fetchone()[0] is None:
                return out
            cur.execute("""SELECT game_date, name_key, stat, run_at_utc, mu, disp_a, disp_b FROM nbaprops.projections
                           WHERE game_date = ANY(%s) ORDER BY game_date, name_key, stat, run_at_utc""", (days,))
            rows = cur.fetchall()
    finally:
        conn.close()
    runs: dict[tuple, tuple[list, list]] = {}
    for day, name, stat, run_at, mu, a, b in rows:
        times, values = runs.setdefault((day, name, stat), ([], []))
        times.append(pd.Timestamp(run_at).tz_convert("UTC") if pd.Timestamp(run_at).tzinfo else pd.Timestamp(run_at, tz="UTC"))
        values.append((float(mu), float(a), float(b)))
    position = {index: n for n, index in enumerate(offers.index)}
    for offer in wanted.itertuples():
        key = (offer.commence_time.astimezone(EASTERN).date(), match_name(offer.player), NBA_STAT[offer.market])
        if key not in runs:
            continue
        times, values = runs[key]
        before = bisect_left(times, pd.Timestamp(offer.observed_at))       # forecasts strictly earlier than the price
        if before == 0:
            continue
        mu, a, b = values[before - 1]
        out[position[offer.Index]] = float(p_over(np.array([mu]), np.array([float(offer.line)]), a, b)[0])
    return out
