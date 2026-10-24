from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from sharp_pipeline import report, scan

NOW = datetime(2026, 10, 23, 23, 0, tzinfo=timezone.utc)
START = NOW + timedelta(minutes=30)


def market(key, outcomes, at=NOW):
    return {"key": key, "last_update": at.isoformat(), "outcomes": outcomes}


def bookmaker(key, markets, at=NOW):
    return {"key": key, "last_update": at.isoformat(), "markets": markets}


def event(books, eid="e1"):
    return {"id": eid, "commence_time": START.isoformat(), "home_team": "Milwaukee Bucks",
            "away_team": "Chicago Bulls", "bookmakers": books}


def lines(spread_home=-6.5, home_spread=-110, away_spread=-110, ml=(-250, 210), total=(225.5, -110, -110)):
    return [market("h2h", [dict(name="Milwaukee Bucks", price=ml[0]), dict(name="Chicago Bulls", price=ml[1])]),
            market("spreads", [dict(name="Milwaukee Bucks", price=home_spread, point=spread_home),
                               dict(name="Chicago Bulls", price=away_spread, point=-spread_home)]),
            market("totals", [dict(name="Over", price=total[1], point=total[0]),
                              dict(name="Under", price=total[2], point=total[0])])]


def test_pairs_key_spreads_on_the_home_line_and_keep_every_market():
    quotes = scan.pairs(dict(bookmaker("fanduel", lines()), _home="Milwaukee Bucks", _away="Chicago Bulls"), NOW)
    assert set(quotes) == {("h2h", "game", 0.0), ("spread", "game", -6.5), ("total", "game", 225.5)}
    assert quotes[("spread", "game", -6.5)]["a"] == -110


def test_gaps_on_every_side_at_the_exact_line_only():
    ev_ = event([bookmaker("draftkings", lines(home_spread=105, away_spread=-125, total=(225.5, -130, +110))),
                 bookmaker("pinnacle", lines(home_spread=-108, away_spread=-104, total=(225.5, -105, -105))),
                 bookmaker("fanduel", lines(spread_home=-7.5, home_spread=120, away_spread=-140))])
    gaps = scan.find_gaps(ev_, NOW)
    found = {(g["book"], g["market"], g["side"], g["line"]) for g in gaps}
    assert ("draftkings", "spread", "home", -6.5) in found       # +105 on the home side Pinnacle prices near even
    assert ("draftkings", "total", "under", 225.5) in found      # unders count
    assert not any(g["book"] == "fanduel" and g["market"] == "spread" for g in gaps)  # -7.5 has no exact Pinnacle line


def test_away_spread_gap_reports_the_away_line():
    ev_ = event([bookmaker("draftkings", lines(home_spread=-140, away_spread=120)),
                 bookmaker("pinnacle", lines(home_spread=-110, away_spread=-110))])
    away = next(g for g in scan.find_gaps(ev_, NOW) if g["market"] == "spread")
    assert away["side"] == "away" and away["line"] == 6.5


def test_props_and_freshness_and_skew():
    pts = lambda over, under, at=NOW: bookmaker("fanduel" if over != -115 else "pinnacle", [market("player_points", [
        dict(name="Over", description="Giannis Antetokounmpo", price=over, point=30.5),
        dict(name="Under", description="Giannis Antetokounmpo", price=under, point=30.5)], at)], at)
    sharp = bookmaker("pinnacle", [market("player_points", [
        dict(name="Over", description="Giannis Antetokounmpo", price=-115, point=30.5),
        dict(name="Under", description="Giannis Antetokounmpo", price=-105, point=30.5)])])
    gaps = scan.find_gaps(event([pts(+110, -140), sharp]), NOW)
    assert gaps and gaps[0]["market"] == "points" and gaps[0]["side"] == "over" and gaps[0]["player_norm"]
    stale = NOW - timedelta(seconds=301)
    assert scan.find_gaps(event([pts(+110, -140, stale), sharp]), NOW) == []
    skewed = NOW - timedelta(seconds=150)
    assert scan.find_gaps(event([pts(+110, -140, skewed), sharp]), NOW) == []


def test_tiers_follow_the_nfl_window():
    assert scan.tier(0.05, 30) == "decision" and scan.tier(0.05, 50) == "early"
    assert scan.tier(0.05, 10) == "late" and scan.tier(0.02, 30) == "logged"


def test_credits_hold_back_nfl_slates_until_the_reset():
    today = date(2026, 10, 23)  # Friday; reset on the 3rd
    reserve = scan.nfl_reserve(today, scan.reset_date(today, 3))
    assert reserve == 2 * scan.NFL_SUNDAY_RESERVE + 3 * scan.NFL_PRIMETIME_RESERVE
    assert scan.daily_allowance(4940, 0, today, 3) == pytest.approx((4940 - 150 - reserve) / 11)
    assert scan.daily_allowance(1000, 0, today, 3) == 0.0


def test_plan_polls_lines_once_per_sport_then_props_spaced():
    events = {"nba": [dict(id="a", commence_time=(NOW + timedelta(minutes=40)).isoformat()),
                      dict(id="b", commence_time=(NOW + timedelta(minutes=100)).isoformat())],
              "ncaaf": [dict(id="c", commence_time=(NOW + timedelta(minutes=55)).isoformat())]}
    due = scan.plan(events, {}, NOW)
    kinds = [(d[0], d[1], d[2]["id"] if d[0] == "props" else None) for d in due]
    assert kinds == [("lines", "nba", None), ("lines", "ncaaf", None), ("props", "nba", "a")]
    recent = {"props_last_poll": {"a": (NOW - timedelta(minutes=5)).isoformat()}}
    assert not any(d[0] == "props" for d in scan.plan(events, recent, NOW))


def test_close_grades_against_pinnacles_last_quote_at_the_same_line():
    close_event = event([bookmaker("pinnacle", lines(home_spread=-115, away_spread=-105), at=START - timedelta(minutes=5))])
    closes = {("nba", "e1"): (START - timedelta(minutes=5), close_event)}
    row = dict(sport="nba", event_id="e1", market="spread", side="home", line=-6.5, price=105, player_norm="")
    fair = scan.sharp_math.no_vig_over(-115, -105)
    assert report.close_ev(row, closes) == pytest.approx(scan.sharp_math.ev(fair, 105))
    assert report.close_ev(dict(row, side="away", line=6.5, price=-110), closes) == pytest.approx(
        scan.sharp_math.ev(1 - fair, -110))
    moved = scan.shifted_probability("spread", -6.5, fair, -7.5, 12.5)
    assert report.close_ev(dict(row, line=-7.5), closes) == pytest.approx(scan.sharp_math.ev(moved, 105))
    assert report.close_ev(dict(row, line=-9.5), closes) is None  # moved 3 points: unknown, not zero


def test_digest_lists_research_decisions_for_the_day():
    df = pd.DataFrame([dict(sport="nba", day=date(2026, 10, 23), tier="decision", ev=0.05, ev_at_close=0.03,
                            player=None, home_team="Milwaukee Bucks", away_team="Chicago Bulls", market="spread",
                            side="away", line=6.5, book="draftkings", price=105)])
    text = report.digest_text(df, "nba", date(2026, 10, 23))
    assert "research only" in text and "Chicago Bulls +6.5 spread at draftkings +105" in text
    assert report.digest_text(df, "ncaaf", date(2026, 10, 23)) is None


def test_a_half_point_off_spread_is_priced_by_shifting_and_labelled():
    ev_ = event([bookmaker("fanduel", lines(spread_home=-6.0, home_spread=100, away_spread=-120)),
                 bookmaker("pinnacle", lines(spread_home=-6.5, home_spread=-110, away_spread=-110))])
    gaps = [g for g in scan.find_gaps(ev_, NOW, sport="nba") if g["market"] == "spread"]
    # Getting -6 instead of -6.5 at the same price is the edge.
    assert gaps and gaps[0]["side"] == "home" and gaps[0]["line"] == -6.0 and gaps[0]["method"] == "shifted"
    assert scan.find_gaps(ev_, NOW, sport=None) == [g for g in scan.find_gaps(ev_, NOW, sport=None)]  # no shift without a sport
    assert not [g for g in scan.find_gaps(ev_, NOW) if g["market"] == "spread"]


def test_college_spreads_are_never_shifted_across_key_numbers():
    near = event([bookmaker("fanduel", lines(spread_home=-3.5, home_spread=145, away_spread=-175)),
                  bookmaker("pinnacle", lines(spread_home=-2.5, home_spread=-110, away_spread=-110))])
    assert not [g for g in scan.find_gaps(near, NOW, sport="ncaaf") if g["market"] == "spread"]
    assert [g for g in scan.find_gaps(near, NOW, sport="nba") if g["market"] == "spread"]  # basketball still shifts


def test_alternate_first_half_and_team_total_markets_are_read():
    alt = market("alternate_spreads", [dict(name="Milwaukee Bucks", price=150, point=-9.5),
                                       dict(name="Chicago Bulls", price=-180, point=9.5)])
    h1 = market("totals_h1", [dict(name="Over", price=-110, point=112.5), dict(name="Under", price=-110, point=112.5)])
    team = market("team_totals", [dict(name="Over", description="Milwaukee Bucks", price=-115, point=118.5),
                                  dict(name="Under", description="Milwaukee Bucks", price=-105, point=118.5)])
    quotes = scan.pairs(dict(bookmaker("fanduel", [alt, h1, team]), _home="Milwaukee Bucks", _away="Chicago Bulls"), NOW)
    assert ("spread", "game", -9.5) in quotes and ("total_h1", "game", 112.5) in quotes
    assert ("team_total", "milwaukee bucks", 118.5) in quotes


def test_shift_is_monotone_and_exact_at_zero_gap():
    assert scan.shifted_probability("total", 225.5, 0.5, 225.5, 18.0) == pytest.approx(0.5)
    assert scan.shifted_probability("total", 225.5, 0.5, 226.5, 18.0) < 0.5
    assert scan.shifted_probability("spread", -6.5, 0.5, -6.0, 12.5) > 0.5  # home -6 covers more often


def test_lines_are_polled_about_every_twenty_minutes():
    events = {"ncaaf": [dict(id="c", commence_time=(NOW + timedelta(minutes=30)).isoformat())], "nba": []}
    assert scan.plan(events, {"lines_last_poll": {"ncaaf": (NOW - timedelta(minutes=10)).isoformat()}}, NOW) == []
    assert scan.plan(events, {"lines_last_poll": {"ncaaf": (NOW - timedelta(minutes=20)).isoformat()}}, NOW)


def test_a_scanned_day_without_decisions_still_posts():
    empty = pd.DataFrame(columns=["sport", "day", "tier", "ev"])
    text = report.digest_text(empty, "ncaaf", date(2026, 10, 23), scanned=4)
    assert "4 games scanned" in text and "no gap reached +3%" in text
    assert report.digest_text(empty, "ncaaf", date(2026, 10, 23), scanned=0) is None  # nothing ran: nothing to say
