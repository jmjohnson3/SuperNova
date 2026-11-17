"""Middles against results: first sightings, grading of both bets, one bet per player and the report."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sports_sniper import arbs_backtest as ab
from sports_sniper.config import Settings
from sports_sniper.providers.results import GameResult

START = datetime(2026, 11, 16, 23, 30, tzinfo=timezone.utc)
DEC = 1 + 100 / 110


def price(book, side, line, american, player="A Player", minutes_before=30, event="e1", market="player_points"):
    seen = START - timedelta(minutes=minutes_before)
    return dict(sport="nba", event_id=event, event_name="Away @ Home", commence_time=START, book=book, market=market, player=player,
                player_key=player.lower(), line=line, line_key=str(line), side=side, price=american, last_update=seen - timedelta(minutes=1),
                observed_at=seen, link=None)


def middle(player, event="e1", low=10.5, high=12.5, **more):
    return [price("fanduel", "over", low, -110, player, event=event, **more), price("draftkings", "under", high, -110, player, event=event, **more)]


def results(points: dict):
    def load(dsn, games):
        return {g.event_id: GameResult(True, "final", {k: {"pts": v, "reb": 0, "ast": 0, "fg3m": 0} for k, v in points.items()}, set(),
                                       set(points) | {"sat out"}) for g in games}
    return load


def test_each_pair_of_prices_is_taken_once_at_its_first_sighting():
    rows = middle("A Player", minutes_before=30) + middle("A Player", minutes_before=20)                 # the same prices at a later look
    rows += [price("fanduel", "over", 10.5, 100, minutes_before=10), price("draftkings", "under", 12.5, -110, minutes_before=10)]
    rows += middle("A Player", minutes_before=-5)                                                        # after the start
    found = ab.sightings(pd.DataFrame(rows), ab.TAB_BOOKS)
    assert len(found) == 1 and found.over_price.iloc[0] == -110 and found.event.iloc[0] == "e1"
    assert found.observed_at.iloc[0] == START - timedelta(minutes=30)
    assert ab.sightings(pd.DataFrame(rows).iloc[0:0]).empty and ab.sightings(pd.DataFrame(rows), ["fanduel"]).empty


def test_both_bets_are_graded_and_a_player_who_sat_out_is_left_out():
    rows = middle("Lands") + middle("Under Only") + middle("Over Only") + middle("Sat Out") + middle("Pushes", low=10, high=12)
    found = ab.sightings(pd.DataFrame(rows), ab.TAB_BOOKS)
    settled = ab.settle(found, Settings(), results({"lands": 11, "under only": 8, "over only": 20, "pushes": 10})).set_index("player")
    assert set(settled.index) == {"Lands", "Under Only", "Over Only", "Pushes"}
    assert settled.loc["Lands", "landed"] and settled.loc["Lands", "ret"] == pytest.approx(DEC - 1)
    for name in ("Under Only", "Over Only"):
        assert not settled.loc[name, "landed"] and settled.loc[name, "ret"] == pytest.approx(DEC / 2 - 1)
    assert not settled.loc["Pushes", "landed"] and settled.loc["Pushes", "ret"] == pytest.approx((1 + DEC) / 2 - 1)   # over refunded, under won


def test_one_bet_per_player_and_the_report_reads_back():
    rows = []
    for i in range(6):
        rows += middle("Lands", event=f"g{i}") + middle("Misses", event=f"g{i}")
        rows += middle("Lands", event=f"g{i}", low=9.5, high=12.5, minutes_before=20)                    # a wider one seen later: not a second bet
        rows += [price("draftkings", "over", 2.5, 143, "Arb", event=f"g{i}", market="player_assists"),
                 price("fanduel", "under", 2.5, -128, "Arb", event=f"g{i}", market="player_assists")]
    load = results({"lands": 11, "misses": 30, "arb": 0})
    load_all = lambda dsn, games: {k: GameResult(True, "final", {p: {**s, "ast": 1.0} for p, s in g.stats.items()}, set(), g.known)
                                   for k, g in load(dsn, games).items()}
    prices = pd.DataFrame(rows)
    two = ab.settle(ab.sightings(prices, ab.TAB_BOOKS), Settings(), load_all)
    report = ab.summarise(two, two, 0)
    tab = report["headline"][0][1]
    assert tab["bets"] == 12 and tab["games"] == 6 and tab["landed"] == pytest.approx(0.5) and tab["needed"] == pytest.approx(1 / 21)
    assert tab["ret"] == pytest.approx((DEC - 1 + DEC / 2 - 1) / 2) and tab["ret_interval"][0] == pytest.approx(tab["ret"])   # every game alike
    arb = report["arbitrage"][0][1]
    assert arb["bets"] == 6 and arb["ret"] == pytest.approx(arb["cost"] * -1) and arb["ret"] > 0
    assert ab.one_each(two[two.player == "Lands"], last=True).over_line.eq(9.5).all()
    text = ab.markdown(report, START)
    assert "| FanDuel and DraftKings, cost up to 5% (the tab's setting) | 12 | 6 | 50.0% |" in text and "| NBA Points |" in text
    assert ab.measure(two.iloc[0:0]) == dict(bets=0) and "| x | 0 |" in ab.line("x", dict(bets=0))
