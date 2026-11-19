"""NFL Report tab: the right slate is shown first, every line and prop becomes a bet row, links stay on the book's domain."""
from datetime import date, datetime, timezone
from pathlib import Path

from sports_sniper import nfl_report_view as view

GAME = {"game_id": "2026_11_NE_DET", "matchup": "NE @ DET", "away": "NE", "home": "DET", "kickoff_utc": "2026-11-22T18:00:00+00:00",
        "lines": {"draftkings": {"latest": {"seen": "Sun 08:30 ET", "home_spread": -2.5, "home_price": -115, "away_price": -105, "total": 50.5,
                                             "over_price": -110, "under_price": -110,
                                             "links": {"spread_home": "https://sportsbook.draftkings.com/event/1?outcomes=0HC1",
                                                       "total_over": "https://evil.example/?outcomes=1"}}}},
        "props_fanduel": [{"player": "A.Brown", "stat": "receiving_yards", "line": 50.5, "side_priced": "over", "price": -113,
                           "model_projection": 55.2, "link": None}]}


def test_the_next_slate_is_shown_first():
    paths = [Path(f"2026_w11_{d}.json") for d in ("2026-11-23", "2026-11-22", "2026-11-19")]
    assert view.default_index(paths, date(2026, 11, 20)) == 1 and view.default_index(paths, date(2026, 11, 24)) == 0
    assert view.list_packets(Path("does-not-exist")) == []


def test_every_line_side_and_prop_is_a_bet_row():
    rows = view.bet_rows(GAME)
    assert [r["Bet"] for r in rows] == ["DET -2.5", "NE +2.5", "Over 50.5", "Under 50.5", "A.Brown over 50.5 receiving yards"]
    assert rows[1]["edge"] == {"book": "draftkings", "stat": "spread", "side": "away", "line": 2.5, "link": None, "player": None}
    assert rows[1]["Odds"] == "-105" and rows[4]["Book"] == "FanDuel"


def test_links_stay_on_the_books_own_domain_and_fall_back_to_its_page():
    rows = view.bet_rows(GAME)
    assert view.resolve(rows[0]["edge"], "NE", "DET") == ("https://sportsbook.draftkings.com/event/1?outcomes=0HC1", "selection")
    assert view.resolve(rows[2]["edge"], "NE", "DET") == (view.BOOK_PAGE["draftkings"], "page")       # foreign host refused


def test_started_games_get_no_links():
    assert view.started(GAME, date(2026, 11, 22), datetime(2026, 11, 22, 18, 1, tzinfo=timezone.utc))
    assert not view.started(GAME, date(2026, 11, 22), datetime(2026, 11, 22, 17, 0, tzinfo=timezone.utc))
    assert view.started({}, date(2026, 11, 15), datetime(2026, 11, 19, tzinfo=timezone.utc))
