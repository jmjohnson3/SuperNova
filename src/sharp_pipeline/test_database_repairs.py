"""Opt-in PostgreSQL regressions. Every change is isolated in a schema and rolled back."""
import os
import uuid
from datetime import timedelta

import pandas as pd
import psycopg2
import pytest

from nba_props import ingest
from nba_props.test_operational_repairs import logs
from nfl_pipeline.db import PG_DSN
from sharp_pipeline import report, scan
from sharp_pipeline.test_sharp_multi import NOW, bookmaker, event, lines

pytestmark = pytest.mark.skipif(os.getenv("RUN_DATABASE_REPAIR_TESTS") != "1", reason="opt-in local database test")


@pytest.fixture
def database():
    conn = psycopg2.connect(PG_DSN, connect_timeout=5)
    schema = "repair_test_" + uuid.uuid4().hex
    with conn.cursor() as cur:
        cur.execute(f'CREATE SCHEMA "{schema}"')
    def rewrite(query):
        return query.replace("bets.", schema + ".").replace("nbaprops.", schema + ".").replace("nbaprops;", schema + ";")
    class Cursor:
        def __init__(self, *args, **kwargs):
            self.inner = conn.cursor(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.inner.close()
        def execute(self, query, params=None):
            if isinstance(query, bytes):
                query = rewrite(query.decode()).encode()
            else:
                query = rewrite(query)
            return self.inner.execute(query, params)
        def __getattr__(self, name):
            return getattr(self.inner, name)
    class Connection:
        def cursor(self, *args, **kwargs):
            return Cursor(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def commit(self):
            pass  # Even code under test cannot commit fixture data.
    try:
        yield Connection()
    finally:
        conn.rollback()
        conn.close()


def test_legacy_unique_key_migrates_and_same_offer_can_qualify_later(database):
    with database.cursor() as cur:
        cur.execute(scan.SCHEMA_SQL)
        # Emulate the old production uniqueness constraint before running the migration.
        cur.execute("""ALTER TABLE bets.sharp_multi_decisions ADD CONSTRAINT legacy_offer_unique
                       UNIQUE (sport, event_id, market, player_norm, side, line, book, price)""")
    early = NOW - timedelta(minutes=25)  # T-55; NOW is T-30
    def snapshot(at):
        books = [bookmaker("fanduel", lines(home_spread=110)), bookmaker("pinnacle", lines())]
        for b in books:
            b["last_update"] = at.isoformat()
            for m in b["markets"]:
                m["last_update"] = at.isoformat()
        return event(books)
    scan.store(database, "nba", snapshot(early), early)
    with database.cursor() as cur:
        cur.execute(scan.SCHEMA_SQL)
        cur.execute(scan.SCHEMA_SQL)  # migration is repeatable
    scan.store(database, "nba", snapshot(NOW), NOW)
    scan.store(database, "nba", snapshot(NOW), NOW)  # identical retry must not duplicate
    scan.store(database, "nba", snapshot(NOW + timedelta(minutes=5)), NOW + timedelta(minutes=5))
    with database.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM bets.sharp_multi_decisions ORDER BY decided_at")
        raw = pd.DataFrame(cur.fetchall())
    assert list(raw.tier) == ["early", "decision", "decision"]
    assert raw.observation_id.nunique() == 3
    selected = report.decisions_frame(raw, {})
    assert list(selected.tier) == ["early", "decision"]
    assert selected[selected.tier.eq("decision")].decided_at.iloc[0] == NOW


def test_ingest_quarantines_conflicts_and_recovers_when_source_agrees(database, monkeypatch):
    monkeypatch.setattr(ingest.psycopg2, "connect", lambda *args, **kwargs: database)
    monkeypatch.setattr(ingest, "SEASON_TYPES", ("Regular Season",))
    monkeypatch.setattr(ingest.time, "sleep", lambda *args: None)
    payload = pd.concat([logs(), logs()])
    monkeypatch.setattr(ingest, "fetch", lambda *args: payload)
    assert ingest.ingest([2026])["2026-27 Regular Season"] == 1
    payload = pd.concat([logs(), logs().assign(PTS=21)])
    assert ingest.ingest([2026])["2026-27 Regular Season"] == 0
    assert ingest.conflicted_games(database) == ["0022600001"]
    payload = logs().assign(PTS=21)
    assert ingest.ingest([2026])["2026-27 Regular Season"] == 1
    assert ingest.conflicted_games(database) == []
    with database.cursor() as cur:
        cur.execute("SELECT pts FROM nbaprops.player_games")
        assert cur.fetchall() == [(21,)]
