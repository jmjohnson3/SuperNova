from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd

from nba_props import ingest, live


def logs():
    return pd.DataFrame([dict(GAME_ID="0022600001", PLAYER_ID=1, GAME_DATE="2026-10-20", PLAYER_NAME="A",
                             TEAM_ABBREVIATION="BOS", MATCHUP="BOS vs. NYK", MIN=30.0, PTS=20, REB=5, AST=3,
                             FG3M=2, FGA=15, FTA=4, TOV=2, OREB=1, STL=1, BLK=0, PLUS_MINUS=5.0)])


def test_identical_and_complementary_duplicates_are_safe_for_one_upsert():
    df = logs()
    rows = ingest.to_rows(pd.concat([df] * 3), "2026-27", "Regular Season")
    assert len(rows) == 1
    missing = df.assign(AST=None)
    assert ingest.to_rows(pd.concat([missing, df]), "2026-27", "Regular Season") == rows


def test_conflicting_values_are_quarantined_instead_of_arbitrarily_selected():
    rows, conflicts = ingest.prepare_rows(pd.concat([logs(), logs().assign(PTS=21)]), "2026-27", "Regular Season")
    assert rows == []
    assert len(conflicts) == 1
    assert {c["pts"] for c in conflicts[0]["candidates"]} == {20, 21}


def test_news_retries_unchanged_digest_once_daily_fits_are_available(monkeypatch, tmp_path):
    now = datetime(2026, 10, 28, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(live, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(live, "FITS", tmp_path)
    monkeypatch.setattr(live, "schedule", lambda day: pd.DataFrame([dict(tip_utc=pd.Timestamp(now) + pd.Timedelta(hours=3), home="BOS", away="NYK")]))
    monkeypatch.setattr(live.psycopg2, "connect", MagicMock())
    monkeypatch.setattr(live.availability, "fetch_lineups", lambda day: [])
    monkeypatch.setattr(live.availability, "fetch_injuries", lambda: [])
    monkeypatch.setattr(live.availability, "save", lambda *args: None)
    monkeypatch.setattr(live.availability, "news_digest", lambda *args: "unchanged")
    project = MagicMock(return_value=dict(players=10, games=1))
    monkeypatch.setattr(live, "project", project)
    assert live.news(now)["status"] == "no_fits_yet"
    assert live.news(now)["status"] == "no_fits_yet"
    assert not live.STATE.exists() and project.call_count == 0
    live._fit_path(now.date()).touch()
    project.return_value = dict(status="no_projections")
    assert live.news(now)["status"] == "no_projections" and not live.STATE.exists()
    project.return_value = dict(players=10, games=1)
    assert live.news(now)["status"] == "baseline"
    assert live.news(now)["status"] == "no_change"
    assert project.call_count == 2
