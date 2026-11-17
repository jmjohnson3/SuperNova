"""Arbitrages and middles: the current snapshot, pairing across books, the stake split and the text shown."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sports_sniper import arbs, arbs_view

NOW = datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc)


def price(book, side, line, american, player="Al Horford", market="player_assists", minutes_ago=3, event="e1", starts_in=30, link=None,
          quote_age=1):
    seen = NOW - timedelta(minutes=minutes_ago)
    return dict(sport="nba", event_id=event, event_name="Away @ Home", commence_time=NOW + timedelta(minutes=starts_in), book=book,
                market=market, player=player, player_key=player.lower(), line=line, line_key=str(line), side=side, price=american,
                last_update=seen - timedelta(minutes=quote_age), observed_at=seen, link=link)


def frame(rows):
    return pd.DataFrame(rows)


def test_the_current_snapshot_is_each_games_latest_look_before_the_start():
    prices = frame([price("fanduel", "over", 2.5, 140), price("fanduel", "over", 2.5, 120, minutes_ago=13),          # an older look
                    price("draftkings", "under", 2.5, -128, minutes_ago=4),                                             # same snapshot
                    price("betmgm", "over", 2.5, 150, quote_age=9),                                                     # a stale price
                    price("fanduel", "over", 4.5, 100, event="old", minutes_ago=40),
                    price("fanduel", "over", 4.5, 100, event="started", starts_in=-5, minutes_ago=8)])
    live = arbs.current(prices, NOW, timedelta(minutes=10))
    assert sorted(zip(live.book, live.price)) == [("draftkings", -128), ("fanduel", 140)] and (live.stat == "player_assists").all()
    assert set(arbs.current(prices, NOW, timedelta(hours=36), upcoming_only=False).event_id) == {"e1", "old", "started"}
    assert arbs.current(prices.iloc[0:0], NOW, timedelta(minutes=10)).empty


def test_an_arbitrage_is_the_same_line_at_two_books_returning_more_than_staked():
    prices = arbs.current(frame([price("draftkings", "over", 2.5, 143), price("draftkings", "under", 2.5, -180),
                                 price("fanduel", "under", 2.5, -128), price("fanduel", "over", 2.5, 104),
                                 price("pinnacle", "under", 2.5, 100),                                               # never a leg
                                 price("betmgm", "under", 2.5, -120)]), NOW, timedelta(minutes=10))
    found = arbs.find(prices, ["fanduel", "draftkings"])
    assert len(found) == 1
    arb = found.iloc[0]
    over, under = 2.43, 1 + 100 / 128
    assert (arb.kind, arb.over_book, arb.under_book) == ("Arbitrage", "draftkings", "fanduel")
    assert arb.miss_return == pytest.approx(over * under / (over + under) - 1) and arb.miss_return > 0 and arb.break_even == 0
    stake_over, stake_under = arbs.stakes(arb, 100)
    assert stake_over + stake_under == pytest.approx(100) and stake_over * over == pytest.approx(stake_under * under, abs=0.05)
    everyone = arbs.find(prices)
    assert set(everyone.under_book) == {"betmgm"} and "pinnacle" not in set(everyone.over_book) | set(everyone.under_book)   # the best pair per line
    assert arbs.find(prices, ["fanduel"]).empty and arbs.find(prices.iloc[0:0]).empty


def test_a_middle_is_an_under_on_a_higher_line():
    prices = arbs.current(frame([price("fanduel", "over", 10.5, -110, market="player_points"), price("fanduel", "under", 10.5, -110, market="player_points"),
                                 price("draftkings", "under", 12.5, -110, market="player_points"), price("draftkings", "over", 12.5, -110, market="player_points"),
                                 price("draftkings", "over", 9.5, 105, market="player_points_alternate"),
                                 price("betmgm", "under", 11.5, 130, market="player_points")]), NOW, timedelta(minutes=10))
    found = arbs.find(prices, ["fanduel", "draftkings"], max_cost=0.05)
    paid = found[(found.over_line == 10.5) & (found.under_line == 12.5)].iloc[0]
    dec = 1 + 100 / 110
    assert paid.kind == "Middle" and paid.width == 2 and paid.miss_return == pytest.approx(dec / 2 - 1) and paid.hit_return == pytest.approx(dec - 1)
    assert paid.break_even == pytest.approx(-paid.miss_return / (paid.hit_return - paid.miss_return)) and pd.isna(paid.books_chance)   # each book calls its line even
    assert (found.under_line > found.over_line).all() and (found.over_book != found.under_book).all() and len(found) == 2
    assert arbs.find(prices, ["fanduel", "draftkings"], max_cost=0.01).empty
    free = arbs.find(prices, ["fanduel", "betmgm", "draftkings"])
    best = free.iloc[0]
    assert best.kind == "Free middle" and (best.over_book, best.over_line, best.under_book, best.under_line) == ("draftkings", 9.5, "betmgm", 11.5)
    assert best.miss_return > 0 and best.hit_return == pytest.approx(2 * (best.miss_return + 1) - 1) and best.break_even == 0


def test_the_text_says_what_each_one_pays():
    prices = arbs.current(frame([price("draftkings", "over", 2.5, 143, link="https://dk.example/{state}/x"), price("fanduel", "under", 2.5, -128),
                                 price("fanduel", "over", 10.5, -110, market="player_points", player="B"),
                                 price("draftkings", "under", 12.5, -110, market="player_points", player="B")]), NOW, timedelta(minutes=10))
    found = arbs.find(prices, ["fanduel", "draftkings"])
    arb, middle = found.iloc[0], found.iloc[1]
    assert "Returns **+2.78%** of the $100 staked whatever happens." in arbs_view.describe(arb) and "DraftKings **+143**" in arbs_view.describe(arb)
    text = arbs_view.describe(middle)
    assert "Costs **4.5%** if it misses" in text and "lands from 10.5 to 12.5" in text and "more than 4.8% of the time" in text
    assert arbs_view.link(arb.over_link, "CO") == "https://dk.example/co/x" and arbs_view.link(arb.under_link, "CO") is None
    shown = arbs_view.table(found)
    assert shown.Type.tolist() == ["Arbitrage", "Middle"] and shown["Needed to pay"].tolist() == ["-", "4.8%"]
    assert shown.Over.iloc[0] == "2.5 at DraftKings +143"
