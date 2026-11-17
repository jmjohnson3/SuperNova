"""How often do middles land, and would they have paid? A test of the Arbitrage & Middles tab on stored prices.

Every stored look at a game before its start is searched the way the tab searches (`arbs.find`), each bet is taken
once at the prices of the look where it was first seen, and graded with the game's final statistics:

    return = what the two bets paid together for each $1 staked across them, minus the $1

A middle "lands" when the result falls strictly between the two lines, so both bets win. A push on one line refunds
that bet. A player who did not play is a refund on both and is left out. The interval resamples whole games.

It assumes every price could be bet at the moment it was stored; prices are stored only in the last hour before a
game, so this says nothing about earlier in the day. Report only: it flags nothing and places nothing.

    python -m sports_sniper.arbs_backtest
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sports_sniper import arbs, backtest, odds_math
from sports_sniper.config import Settings, get_settings
from sports_sniper.markets import book_label, market_label

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "sniper_middles_latest.md"
MAX_COST = 0.10                       # the widest search; the report cuts it down to each cost cap
TAB_COST = 0.05                       # the tab's default cost cap
TAB_BOOKS = ("fanduel", "draftkings")
SEP = "|"
BET = ["event", "stat", "player_key"]
RESAMPLES = 2000
COSTS = [("Free (cannot lose)", -1e-9, 1.0), ("Costs up to 2%", -0.02, -1e-9), ("Costs 2% to 5%", -0.05, -0.02), ("Costs 5% to 10%", -0.10, -0.05)]


def sightings(prices: pd.DataFrame, books=None, max_cost: float = MAX_COST) -> pd.DataFrame:
    """Every arbitrage and middle in every stored look before the start, each pair of prices at its first sighting."""
    if prices.empty:
        return pd.DataFrame(columns=arbs.COLUMNS + ["event"])
    p = prices[(prices.observed_at < prices.commence_time) & (prices.observed_at - prices.last_update <= arbs.MAX_QUOTE_AGE)]
    p = p.assign(event=p.event_id, stat=p.market.str.replace(arbs.ALTERNATE, "", regex=False),
                 event_id=p.event_id + SEP + p.observed_at.astype("int64").astype(str))          # one look at one game
    found = arbs.find(p, None if books is None else list(books), max_cost)
    found = found.assign(event=found.event_id.str.split(SEP).str[0])
    pair = BET + ["over_book", "over_line", "under_book", "under_line"]
    return found.sort_values("observed_at").drop_duplicates(pair).reset_index(drop=True)


def settle(found: pd.DataFrame, settings: Settings, load_results=None) -> pd.DataFrame:
    """Add the statistic, whether it landed between the lines, and the return per $1 staked across both bets."""
    if found.empty:
        return found.assign(value=pd.Series(dtype=float), landed=pd.Series(dtype=bool), ret=pd.Series(dtype=float))
    asked = found.assign(event_id=found.event, market=found.stat, line=found.over_line)
    graded = backtest.grade(asked, settings, load_results)
    out = found.assign(value=graded.result_value.to_numpy(), status=graded.status.to_numpy())
    out = out[out.status.isin(["won", "lost", "push"])].copy()
    over_dec, under_dec = (out[c].map(odds_math.american_to_decimal) for c in ("over_price", "under_price"))
    over_pays = np.select([out.value > out.over_line, out.value == out.over_line], [over_dec, 1.0], 0.0)
    under_pays = np.select([out.value < out.under_line, out.value == out.under_line], [under_dec, 1.0], 0.0)
    out["landed"] = (out.value > out.over_line) & (out.value < out.under_line)
    out["ret"] = out.over_share * over_pays + (1.0 - out.over_share) * under_pays - 1.0
    return out.drop(columns="status").reset_index(drop=True)


def one_each(settled: pd.DataFrame, last: bool = False) -> pd.DataFrame:
    """One bet per player, statistic and game: the first seen (or the last), the cheapest when a look shows several."""
    s = settled.sort_values(["observed_at", "miss_return"], ascending=[not last, False])
    return s.drop_duplicates(BET).reset_index(drop=True)


def interval(bets: pd.DataFrame, column: str = "ret", seed: int = 7) -> tuple[float, float] | None:
    """95% interval for the mean, resampling whole games (a game's bets rise and fall together)."""
    groups = [g[column].to_numpy(float) for _, g in bets.groupby("event")]
    if len(groups) < 5:
        return None
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    pick = np.random.default_rng(seed).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def measure(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return dict(bets=0)
    return dict(bets=len(bets), games=int(bets.event.nunique()), landed=float(bets.landed.mean()), landed_interval=interval(bets.assign(x=bets.landed.astype(float)), "x"),
                needed=float(bets.break_even.mean()), books_chance=None if bets.books_chance.isna().all() else float(bets.books_chance.mean()),
                cost=float(-bets.miss_return.mean()), ret=float(bets.ret.mean()), ret_interval=interval(bets), width=float(bets.width.mean()))


def summarise(settled: pd.DataFrame, two: pd.DataFrame, found: int) -> dict:
    """The report: the tab's own setting first (`two`: FanDuel and DraftKings only), then where the result comes from."""
    def middles(frame, low=-TAB_COST):
        return one_each(frame[(frame.width > 0) & (frame.miss_return >= low)])

    tab, everyone = middles(two), middles(settled, low=-MAX_COST)
    out = dict(found=found, settled=len(settled), games=int(settled.event.nunique()) if len(settled) else 0,
               days=int(settled.commence_time.dt.date.nunique()) if len(settled) else 0,
               headline=[("FanDuel and DraftKings, cost up to 5% (the tab's setting)", measure(tab)),
                         ("Any two books, cost up to 5%", measure(middles(settled))),
                         ("Any two books, cost up to 10%", measure(everyone))],
               arbitrage=[("FanDuel and DraftKings", measure(one_each(two[two.width == 0]))),
                          ("Any two books", measure(one_each(settled[settled.width == 0])))],
               by_cost=[(name, measure(everyone[(everyone.miss_return >= low) & (everyone.miss_return < high)])) for name, low, high in COSTS],
               by_sport=[(sport.upper(), measure(g)) for sport, g in middles(settled).groupby("sport")],
               by_stat=[(f"{sport.upper()} {market_label(stat)}", measure(g)) for (sport, stat), g in middles(settled).groupby(["sport", "stat"])],
               by_width=[(f"{width:g}", measure(g)) for width, g in middles(settled).groupby("width") if len(g) >= 10],
               by_books=[(f"Over at {book_label(o)}, under at {book_label(u)}", measure(g))
                         for (o, u), g in tab.groupby(["over_book", "under_book"])])
    return out


def pct(value, signed: bool = False) -> str:
    return "-" if value is None or value != value else (f"{value:+.1%}" if signed else f"{value:.1%}")


def span(pair, signed: bool = False) -> str:
    return "-" if not pair else f"{pct(pair[0], signed)} to {pct(pair[1], signed)}"


HEAD = ("| {} | Bets | Games | Landed | 95% interval | Needed to pay | Books' own chance | Cost if it misses | Return per $1 staked | 95% interval |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def line(name: str, m: dict) -> str:
    if not m.get("bets"):
        return f"| {name} | 0 | - | - | - | - | - | - | - | - |"
    return (f"| {name} | {m['bets']} | {m['games']} | {pct(m['landed'])} | {span(m['landed_interval'])} | {pct(m['needed'])} | "
            f"{pct(m['books_chance'])} | {pct(m['cost'])} | {pct(m['ret'], True)} | {span(m['ret_interval'], True)} |")


def markdown(r: dict, now: datetime) -> str:
    out = [f"# Middles and arbitrages against results ({now:%Y-%m-%d})", "",
           f"{r['found']} pairs of prices found in the stored looks; {r['settled']} settled over {r['games']} games on {r['days']} days. "
           "One bet per player, statistic and game, at the prices of the look where it was first seen. A middle is an over at one book "
           "and an under on a higher line at another; it lands when the result falls between the lines. Player props only. It assumes "
           "every stored price could have been bet.", "", "## Middles", "", HEAD.format("Rule")]
    out += [line(n, m) for n, m in r["headline"]]
    out += ["", "\"Needed to pay\" is the average share of the time the middle must land to break even. \"Books' own chance\" is the gap "
            "between the two books' no-vig prices where both price both sides.", "", "## Same line at two books (arbitrage)", "",
            "Here \"cost if it misses\" is negative: it is the sure return.", "",
            HEAD.format("Books")]
    out += [line(n, m) for n, m in r["arbitrage"]]
    out += ["", "An arbitrage returns the same whatever happens, so its return here only confirms the grading."]
    for title, key, first in (("By cost (any two books)", "by_cost", "Cost"), ("By sport (any two books, cost up to 5%)", "by_sport", "Sport"),
                              ("By statistic (any two books, cost up to 5%)", "by_stat", "Statistic"),
                              ("By gap between the lines (any two books, cost up to 5%)", "by_width", "Gap"),
                              ("By side (FanDuel and DraftKings, cost up to 5%)", "by_books", "Side")):
        out += ["", f"## {title}", "", HEAD.format(first)] + [line(n, m) for n, m in r[key]]
    return "\n".join(out) + "\n"


def run(engine=None, settings: Settings | None = None, *, load_results=None, write: bool = True, now: datetime | None = None) -> dict:
    from sports_sniper import db
    settings = settings or get_settings()
    engine = engine or db.make_engine(settings.database_url)
    now = now or datetime.now(timezone.utc)
    prices = backtest.load_prices(engine)
    found = sightings(prices)
    report = summarise(settle(found, settings, load_results), settle(sightings(prices, TAB_BOOKS), settings, load_results), len(found))
    if write:
        REPORT.write_text(markdown(report, now), encoding="utf-8")
        REPORT.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(markdown(run(), datetime.now(timezone.utc)))
