from datetime import timedelta

import pytest

from sharp_pipeline import arb_watch
from sharp_pipeline.test_sharp_multi import NOW, bookmaker, event, market


def prop(over, under, at=NOW, point=22.5, player="Test Player"):
    return market("player_points", [dict(name="Over", description=player, point=point, price=over, link="https://book.example/o"),
                                    dict(name="Under", description=player, point=point, price=under)], at)


def test_opposite_sides_at_two_books_returning_more_than_staked():
    ev = event([bookmaker("fanduel", [prop(120, -150)]), bookmaker("draftkings", [prop(-135, 105)])])
    (arb,) = arb_watch.find_arbs(ev, NOW)
    assert [(leg["book"], leg["label"], leg["price"]) for leg in arb["legs"]] == [
        ("fanduel", "OVER 22.5", 120), ("draftkings", "UNDER 22.5", 105)]
    cost = 100 / 220 + 100 / 205
    assert arb["guaranteed_return"] == pytest.approx(1 / cost - 1)
    assert sum(leg["share"] for leg in arb["legs"]) == pytest.approx(1)
    # Either result pays the same: each leg's share times its decimal price.
    assert arb["legs"][0]["share"] * 2.20 == pytest.approx(arb["legs"][1]["share"] * 2.05)
    text = arb_watch.card("nba", ev, arb, NOW)
    assert "FanDuel: OVER 22.5 **+120**" in text and "<https://book.example/o>" in text and "Information only" in text


def test_no_arbitrage_without_a_gap_a_shared_line_or_two_books():
    assert arb_watch.find_arbs(event([bookmaker("fanduel", [prop(-110, -110)]), bookmaker("draftkings", [prop(-110, -110)])]), NOW) == []
    assert arb_watch.find_arbs(event([bookmaker("fanduel", [prop(120, -150)]), bookmaker("draftkings", [prop(-135, 105, point=23.5)])]), NOW) == []
    # The same side at both books, or a third book, is never an arbitrage here.
    assert arb_watch.find_arbs(event([bookmaker("fanduel", [prop(120, -150)]), bookmaker("betmgm", [prop(-135, 105)])]), NOW) == []


def test_stale_or_skewed_quotes_are_ignored():
    old = NOW - timedelta(seconds=arb_watch.MAX_QUOTE_AGE + 1)
    assert arb_watch.find_arbs(event([bookmaker("fanduel", [prop(120, -150, old)], old), bookmaker("draftkings", [prop(-135, 105)])]), NOW) == []
    skew = NOW - timedelta(seconds=arb_watch.MAX_QUOTE_SKEW + 1)
    assert arb_watch.find_arbs(event([bookmaker("fanduel", [prop(120, -150, skew)], skew), bookmaker("draftkings", [prop(-135, 105)])]), NOW) == []


def test_spreads_pair_the_home_line_with_the_away_line():
    spread = lambda home, away, line: market("spreads", [dict(name="Milwaukee Bucks", price=home, point=line),
                                                         dict(name="Chicago Bulls", price=away, point=-line)])
    ev = event([bookmaker("fanduel", [spread(105, -125, -6.5)]), bookmaker("draftkings", [spread(-125, 104, -6.5)])])
    (arb,) = arb_watch.find_arbs(ev, NOW)
    assert [leg["label"] for leg in arb["legs"]] == ["Milwaukee Bucks -6.5", "Chicago Bulls +6.5"]
    # A different home line at the other book is a middle, not an arbitrage.
    assert arb_watch.find_arbs(event([bookmaker("fanduel", [spread(105, -125, -6.5)]), bookmaker("draftkings", [spread(-125, 104, -7.5)])]), NOW) == []


class Cursor:
    def __init__(self, db):
        self.db, self.rowcount, self.result = db, 0, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        if "to_regclass" in sql:
            self.result = [("bets.sharp_multi_quotes", None)]
        elif "FROM bets.sharp_multi_quotes" in sql:
            self.result = self.db.snapshots
        elif sql.lstrip().startswith("INSERT"):
            self.rowcount = 0 if params[0] in self.db.rows else 1
            self.db.rows.setdefault(params[0], params[-1])
        elif sql.lstrip().startswith("UPDATE"):
            self.db.rows[params[2]] = params[0]

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


class Database:
    def __init__(self, snapshots):
        self.snapshots, self.rows = snapshots, {}

    def cursor(self):
        return Cursor(self)

    def commit(self):
        pass


def test_each_arbitrage_is_posted_once_and_old_snapshots_are_not_posted():
    ev = event([bookmaker("fanduel", [prop(120, -150)]), bookmaker("draftkings", [prop(-135, 105)])])
    db, posted = Database([("nba", NOW, ev)]), []
    poster = lambda sport, text: posted.append((sport, text)) or "1"
    first = arb_watch.run(db, now=NOW + timedelta(seconds=30), poster=poster)
    again = arb_watch.run(db, now=NOW + timedelta(seconds=60), poster=poster)
    assert (first["sent"], again["sent"], len(posted)) == (1, 0, 1) and list(db.rows.values()) == ["sent"]
    late = Database([("nba", NOW, ev)])
    result = arb_watch.run(late, now=NOW + timedelta(seconds=arb_watch.MAX_SNAPSHOT_AGE + 1), poster=poster)
    assert (result["stale"], result["sent"], len(posted)) == (1, 0, 1) and list(late.rows.values()) == ["stale"]
