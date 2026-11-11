"""Longshot ladder model: a probability for one-sided alternate lines (for example 80+ receiving yards at +900).

An alternate line has no other side, so the market gives no fair price for it directly. The model borrows two
things the market does give:

  1. the anchor: the sharp book's two-sided main line for the same player and statistic in the same snapshot,
     with its margin removed (say a 48% chance of more than 24.5 yards);
  2. the shape: how far players have actually finished from their main lines, as the ratio result / line, on
     every settled main line of that market from earlier days.

The stored ratios are rescaled so that the anchor's chance comes out right at the main line, and the share of
them beyond the alternate line is the estimate. Its uncertainty is the sampling error of that share, widened
because one shape is being used for every player. It is a model estimate, always labelled speculative, and
`evaluate` scores it walk-forward: each alternate is priced only from results that were known before its game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sports_sniper import backtest, odds_math, performance
from sports_sniper.config import Settings
from sports_sniper.providers import results, saved_quotes

SOURCE = "ladder_model"
MIN_RATIOS = 150            # settled main lines of a market needed before the shape is used
SE_INFLATION = 1.5          # one shape for every player understates the real uncertainty
SUFFIX = "_alternate"
KEY = ["event_id", "observed_at", "market", "player_key"]


def library(offers: pd.DataFrame) -> pd.DataFrame:
    """One ratio (result / line) per settled main line: columns day, market, ratio."""
    columns = ["day", "market", "ratio"]
    if offers.empty or "result_value" not in offers:
        return pd.DataFrame(columns=columns)
    use = offers[offers.status.isin(performance.SETTLED) & offers.result_value.notna() & (offers.line > 0)]
    return pd.DataFrame(dict(day=use.commence_time.dt.tz_convert(results.EASTERN).dt.date, market=use.market,
                             ratio=use.result_value.astype(float) / use.line.astype(float)), columns=columns).reset_index(drop=True)


NFL_LINE_MARKETS = {"receiving_yards": "player_reception_yds", "receptions": "player_receptions", "rushing_yards": "player_rush_yds"}


def nfl_line_history(dsn: str | None, connect=None) -> pd.DataFrame:
    """Ratios from the NFL pipeline's own record of FanDuel main lines since the start of the season (the last line
    stored for each player and game) against official final statistics. Read-only; empty when it is not there."""
    columns = ["day", "market", "ratio"]
    dsn = dsn or saved_quotes.default_dsn()
    if dsn is None:
        return pd.DataFrame(columns=columns)
    if connect is None:
        import psycopg2
        connect = psycopg2.connect
    conn = connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for table in ("bets.nfl_player_prop_predictions", "raw.nfl_player_gamelogs", "raw.nfl_games"):
                cur.execute("SELECT to_regclass(%s)", (table,))
                if cur.fetchone()[0] is None:
                    return pd.DataFrame(columns=columns)
            cur.execute("""
                SELECT DISTINCT ON (p.game_id, p.player_id, p.stat) p.game_date_et, p.stat, p.line,
                       CASE p.stat WHEN 'receiving_yards' THEN g.receiving_yards WHEN 'receptions' THEN g.receptions
                                   ELSE g.rushing_yards END
                FROM bets.nfl_player_prop_predictions p
                JOIN raw.nfl_games s ON s.game_id = p.game_id AND lower(s.status) = 'final' AND s.home_score IS NOT NULL
                JOIN raw.nfl_player_gamelogs g ON g.game_id = p.game_id AND g.player_id = p.player_id AND g.source LIKE 'nflverse%%'
                WHERE p.stat = ANY(%s) AND p.line > 0
                ORDER BY p.game_id, p.player_id, p.stat, p.created_at_utc DESC""", (sorted(NFL_LINE_MARKETS),))
            rows = cur.fetchall()
    finally:
        conn.close()
    frame = pd.DataFrame([(day, NFL_LINE_MARKETS[stat], float(value) / float(line)) for day, stat, line, value in rows if value is not None],
                         columns=columns)
    return frame


def combined_library(offers: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """The stored snapshots' own ratios, with a longer outside history replacing them for the markets it covers."""
    own = library(offers)
    if extra is None or extra.empty:
        return own
    return pd.concat([own[~own.market.isin(set(extra.market))], extra], ignore_index=True)


def estimate(ratios: np.ndarray, anchor_probability, anchor_line, line) -> tuple[np.ndarray, np.ndarray]:
    """(chance of finishing over `line`, its standard error) for each row, from sorted `ratios`.

    The ratios are scaled so that the share above the anchor line equals the anchor's chance, then counted beyond
    the alternate line. Counts are smoothed by half an observation so no chance is exactly 0 or 1.
    """
    ratios = np.sort(np.asarray(ratios, dtype=float))
    p0 = np.clip(np.asarray(anchor_probability, dtype=float), 0.02, 0.98)
    at_anchor = np.quantile(ratios, 1 - p0)                     # the ratio that p0 of the stored results beat
    scale = 1 / np.maximum(at_anchor, 1e-6)
    target = np.asarray(line, dtype=float) / np.asarray(anchor_line, dtype=float)
    beyond = len(ratios) - np.searchsorted(ratios, target / scale, side="right")
    p = (beyond + 0.5) / (len(ratios) + 1)
    return p, SE_INFLATION * np.sqrt(p * (1 - p) / len(ratios))


def anchors(prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The sharp book's two-sided main line per snapshot, player and market, margin removed."""
    main = prices[prices.book.isin(settings.sharp_books) & ~prices.market.str.endswith(SUFFIX) & prices.line.notna()]
    wide = main.pivot_table(index=KEY + ["line"], columns="side", values="price", aggfunc="first").reset_index()
    if "over" not in wide or "under" not in wide:
        return pd.DataFrame(columns=KEY + ["anchor_line", "anchor_probability"])
    wide = wide.dropna(subset=["over", "under"]).drop_duplicates(KEY)
    wide["anchor_probability"] = [odds_math.no_vig([o, u], settings.devig_method)[0] for o, u in zip(wide.over, wide.under)]
    return wide.rename(columns={"line": "anchor_line"})[KEY + ["anchor_line", "anchor_probability"]]


def alternates(prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Every one-sided alternate over at a bet book, joined to its anchor in the same snapshot."""
    books = settings.bet_books or sorted(set(prices.book) - set(settings.sharp_books))
    alts = prices[prices.market.str.endswith(SUFFIX) & (prices.side == "over") & prices.book.isin(books) & prices.line.notna()
                  & (prices.commence_time > prices.observed_at)].copy()
    if alts.empty:
        return alts.assign(anchor_line=pd.Series(dtype=float), anchor_probability=pd.Series(dtype=float), base_market=pd.Series(dtype=str))
    alts["base_market"] = alts.market.str.removesuffix(SUFFIX)
    joined = alts.merge(anchors(prices, settings).rename(columns={"market": "base_market"}), how="inner",
                        on=["event_id", "observed_at", "base_market", "player_key"])
    joined["decimal"] = joined.price.map(odds_math.american_to_decimal)
    joined["minutes_to_start"] = (joined.commence_time - joined.observed_at).dt.total_seconds() / 60
    return joined


def price(alts: pd.DataFrame, ratios: pd.DataFrame, settings: Settings, *, walk_forward: bool = True) -> pd.DataFrame:
    """Add probability, standard_error, ev and conservative_ev. With `walk_forward`, each alternate uses only
    ratios from days before its own game; otherwise every stored ratio (today's prices)."""
    alts = alts.copy()
    for column in ("probability", "standard_error", "ev", "conservative_ev"):
        alts[column] = np.nan
    alts["ratios_used"] = 0
    if alts.empty or ratios.empty:
        return alts
    day = alts.commence_time.dt.tz_convert(results.EASTERN).dt.date
    for market, mine in ratios.groupby("market"):
        rows_market = alts.base_market == market
        for today in (sorted(day[rows_market].unique()) if walk_forward else [None]):
            known = mine if today is None else mine[mine.day < today]
            rows = rows_market if today is None else rows_market & (day == today)
            if len(known) < MIN_RATIOS or not rows.any():
                continue
            p, se = estimate(known.ratio.to_numpy(), alts.loc[rows, "anchor_probability"], alts.loc[rows, "anchor_line"], alts.loc[rows, "line"])
            alts.loc[rows, "probability"], alts.loc[rows, "standard_error"], alts.loc[rows, "ratios_used"] = p, se, len(known)
    low = (alts.probability - settings.confidence_z * alts.standard_error).clip(lower=0)
    alts["ev"] = alts.probability * alts.decimal - 1
    alts["conservative_ev"] = low * alts.decimal - 1
    return alts


def evaluate(prices: pd.DataFrame, ratios: pd.DataFrame, settings: Settings, load_results=None) -> pd.DataFrame:
    """Every stored alternate, once (its first look at each book), priced walk-forward from `ratios` and graded."""
    alts = alternates(prices, settings)
    if alts.empty:
        return alts.assign(status=pd.Series(dtype=str))
    first = alts.sort_values("observed_at").drop_duplicates(["event_id", "book", "market", "player_key", "line_key"])
    priced = price(first, ratios, settings)
    priced = priced[priced.probability.notna()]
    return backtest.grade(priced, settings, load_results)


def _bets(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(stake=1.0, closing_line_value=None,
                       profit=[d - 1 if s == "won" else -1.0 if s == "lost" else 0.0 for s, d in zip(rows.status, rows.decimal)])


def summary(evaluated: pd.DataFrame, min_ev: float = 0.04, longshot_american: int = 200) -> dict:
    """Is the model right, and do its picks pay? Calibration by probability band, then every longshot blind against
    the longshots the model calls underpriced, with one bet per player, market and line (the best price)."""
    if evaluated.empty:
        return dict(lines=0, games=0, calibration=[], groups=[])
    settled = evaluated[evaluated.status.isin(performance.SETTLED)]
    decided = settled[settled.status.isin(performance.DECIDED)]
    edges = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0000001]
    bands = []
    for low, high in zip(edges, edges[1:]):
        band = decided[(decided.probability >= low) & (decided.probability < high)]
        if len(band):
            wins = int((band.status == "won").sum())
            bands.append(dict(low=low, high=min(high, 1.0), n=len(band), predicted=float(band.probability.mean()), observed=wins / len(band),
                              break_even=float((1 / band.decimal).mean()), interval=odds_math.wilson_interval(wins, len(band))))
    best = settled.sort_values("decimal", ascending=False).drop_duplicates(["event_id", "market", "player_key", "line_key"])
    longs = best[best.price >= longshot_american]
    groups = [("Every alternate, blind", best), (f"Every longshot (+{longshot_american} or longer), blind", longs),
              (f"Longshots the model has at {min_ev:.0%}+", longs[longs.ev >= min_ev]),
              ("Longshots the model has positive after its haircut", longs[longs.conservative_ev > 0]),
              (f"Any alternate the model has at {min_ev:.0%}+", best[best.ev >= min_ev])]
    return dict(lines=len(settled), games=int(settled.event_id.nunique()), calibration=bands, groups=[(name, performance.summarize(_bets(rows))) for name, rows in groups])


def current(prices: pd.DataFrame, ratios: pd.DataFrame, settings: Settings, longshot_american: int | None = None) -> pd.DataFrame:
    """Longshot alternates in the given (latest) prices, priced from every stored ratio. Ranked by conservative EV."""
    floor = settings.longshot_min_american if longshot_american is None else longshot_american
    alts = alternates(prices, settings) if len(prices) else prices
    if alts.empty:
        return alts
    priced = price(alts[alts.price >= floor], ratios, settings, walk_forward=False)
    priced = priced[priced.probability.notna()].assign(source=SOURCE, evidence="speculative")
    return priced.sort_values("conservative_ev", ascending=False).reset_index(drop=True)
