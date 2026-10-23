from datetime import datetime, timedelta, timezone

import pytest

from nfl_pipeline import sharp_parlays as sp
from nfl_pipeline.modeling.sharp_parlay_report import grade_parlay

NOW = datetime(2026, 10, 25, 16, tzinfo=timezone.utc)


def leg(alert_id, game, book="fanduel", price=100, fair=0.52, ev=None):
    ev = ev if ev is not None else fair * sp.decimal_price(price) - 1
    return dict(alert_id=alert_id, game_id=game, book=book, fd_price=price, fair_probability=fair, ev=ev,
                commence_time_utc=NOW + timedelta(minutes=30), home_team="CHI", away_team="NE",
                player_name=f"P{alert_id}", player_name_norm=f"p{alert_id}", stat="receiving_yards",
                side="over", fd_line=45.5, fd_link=None)


def test_independent_leg_edges_compound():
    priced = sp.price([leg(1, "a", price=100, fair=0.52), leg(2, "b", price=100, fair=0.52)])
    assert priced["decimal_price"] == pytest.approx(4.0)
    assert priced["fair_probability"] == pytest.approx(0.2704)
    assert priced["ev"] == pytest.approx(0.0816)  # two +4% legs make about +8%


def test_pairs_must_share_a_book_and_span_games():
    same_game = [leg(1, "a", fair=0.6), leg(2, "a", fair=0.6)]
    assert sp.choose(same_game) is None
    mixed = [leg(1, "a", book="fanduel", fair=0.6), leg(2, "b", book="draftkings", fair=0.6),
             leg(3, "c", book="fanduel", fair=0.53), leg(4, "d", book="draftkings", fair=0.55)]
    assert [l["alert_id"] for l in sp.choose(mixed)] == [2, 4]  # the best same-book pair
    assert sp.american_price(4.0) == 300 and sp.american_price(1.5) == -200


def test_parlay_text_says_research_and_gives_the_combined_price():
    legs = [leg(1, "a"), leg(2, "b")]
    text = sp.format_parlay(legs, sp.price(legs), "https://account.sportsbook.fanduel.com/x")
    assert "RESEARCH PARLAY - not a bet" in text and "+300" in text and "Add both to slip" in text


def test_grading_follows_book_settlement():
    win = dict(bet_price=100, ev_at_sharp_close=0.04, result="win")
    loss, push = dict(win, result="loss"), dict(win, result="push")
    assert grade_parlay([win, win])["units"] == pytest.approx(3.0)
    assert grade_parlay([win, loss])["units"] == -1.0
    assert grade_parlay([win, push])["units"] == pytest.approx(1.0)  # the push drops out
    assert grade_parlay([win, dict(win, result=None)])["result"] is None
    assert grade_parlay([win, win])["ev_at_sharp_close"] == pytest.approx(1.04 * 1.04 - 1)
    assert grade_parlay([win, dict(win, ev_at_sharp_close=None)])["ev_at_sharp_close"] is None
