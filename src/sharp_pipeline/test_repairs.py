from datetime import date, timedelta

import pandas as pd
import pytest

from sharp_pipeline import outcomes, report, scan
from sharp_pipeline.test_sharp_multi import NOW, START, bookmaker, event, lines, market


def test_market_timestamp_overrides_fresh_parent_and_drives_skew():
    sharp = bookmaker("pinnacle", lines())
    for age in (60, 3):  # stale, then fresh but beyond the two-minute cross-book skew
        soft_lines = lines(home_spread=110)
        for m in soft_lines:
            m["last_update"] = (NOW - timedelta(minutes=age)).isoformat()
        assert scan.find_gaps(event([sharp, bookmaker("fanduel", soft_lines)]), NOW) == []
    assert scan._fresh([NOW.isoformat(), None], NOW) == NOW
    assert scan._fresh([NOW.isoformat(), "invalid"], NOW) is None
    assert scan._fresh([NOW.isoformat(), (NOW + timedelta(seconds=1)).isoformat()], NOW) is None


def test_estimated_closes_cannot_satisfy_review_even_when_counts_and_ev_pass():
    close_time = START - timedelta(minutes=5)
    close = event([bookmaker("pinnacle", [market("totals", [dict(name="Over", price=-110, point=50.5),
                         dict(name="Under", price=-110, point=50.5)], at=close_time)], at=close_time)])
    raw, closes = [], {}
    for i in range(50):
        eid = f"game{i}"
        raw.append(dict(sport="ncaaf", event_id=eid, market="total", side="over", line=49.5, price=110,
                        player_norm="", ev=.05, tier="decision", book="fanduel", decided_at=NOW,
                        commence_time_utc=START + timedelta(days=i % 7)))
        closes[("ncaaf", eid, "lines")] = (close_time, close)
    df = report.decisions_frame(pd.DataFrame(raw), closes)
    stats = report.summarize(df)["ncaaf"]
    assert stats["overall"]["decisions"] == 50 and stats["overall"]["days"] == 7
    assert stats["overall"]["estimated_closes"] == 50
    assert stats["overall"]["estimated_close_ev"] > .015
    assert stats["overall"]["close_coverage"] == 0 and stats["status"] == "research"
    assert set(df.close_method) == {"shifted"} and df.ev_at_close.isna().all()
    assert "Estimated closes: 50" in report.markdown(report.summarize(df))


@pytest.mark.parametrize("players,scores", [
    ({}, {"home": 100, "away": 90}),
    ({"pts": None, "reb": 5, "ast": 4}, {"home": 100, "away": 90}),
    ({}, {"home": 2}),
    ({"pts": float("nan")}, {"home": 100, "away": 90}),
])
def test_missing_player_or_stat_remains_pending(monkeypatch, players, scores):
    day = date(2026, 10, 28)
    monkeypatch.setattr(outcomes, "nba_actuals", lambda *args: (
        {(day, "a"): players} if players else {}, {(day, k): v for k, v in scores.items()}, {day}))
    monkeypatch.setattr(outcomes, "ncaaf_scores", lambda *args: [])
    monkeypatch.setattr(outcomes, "latest_projections", lambda *args: pd.DataFrame())
    df = pd.DataFrame([dict(sport="nba", day=day, market="points", player="A", home_team="home",
                           away_team="away", side="under", line=20.5, price=-110, fair_probability=.5)])
    graded = outcomes.attach(None, df)
    assert graded.result.isna().all() and graded.profit.isna().all()


@pytest.mark.parametrize("name,expected", [
    ("Hawaii Rainbow Warriors", "Hawai'i"), ("San Jose State Spartans", "San José State"),
    ("UMass Minutemen", "Massachusetts"), ("Miami (OH) RedHawks", "Miami (OH)"),
    ("Unknown University", None),
])
def test_verified_school_aliases_and_longest_prefix(name, expected):
    assert outcomes.match_school(name, ["Hawai'i", "San José State", "Massachusetts", "Miami", "Miami (OH)"]) == expected
