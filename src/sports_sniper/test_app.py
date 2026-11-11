"""The scanner, database, saved-quotes adapter, command line and dashboard working together."""
import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from sports_sniper import cli, db, scanner
from sports_sniper.config import Settings, get_settings
from sports_sniper.providers import sample, saved_quotes
from sports_sniper.providers.odds_api import OddsApiClient

NOW = datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, database_url=f"sqlite:///{(tmp_path / 'sniper.db').as_posix()}", odds_api_key="test-key")


@pytest.fixture
def engine(settings):
    eng = db.make_engine(settings.database_url)
    db.init_db(eng)
    yield eng
    eng.dispose()


def count(engine, model) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model))


# --- scanning and storing --------------------------------------------------------------------------
def test_a_sample_scan_is_stored_and_a_repeat_adds_nothing(settings, engine):
    first = scanner.run_scan(settings, engine, "sample", now=NOW)
    assert (first.source, first.events, first.quotes, first.new_quotes) == ("sample", 2, 114, 114)
    assert first.note == "hypothetical sample data" and len(first.opportunities) == count(engine, db.OpportunityRow) == 22
    assert len(first.current) == 12 and {o.observed_at for o in first.current} == {NOW}
    again = scanner.run_scan(settings, engine, "sample", now=NOW)
    assert again.new_quotes == 0 and count(engine, db.PriceRow) == 114 and count(engine, db.OpportunityRow) == 22
    assert count(engine, db.Scan) == 2
    frame = db.load_opportunities(engine)
    assert len(frame) == 12 and set(frame.data_source) == {"sample"} and frame.observed_at.max() == NOW
    assert not frame.superseded.any()                    # each bet's newest look is the current one
    everything = db.load_opportunities(engine, latest_only=False)
    assert len(everything) == 22 and everything.superseded.sum() == 10 and (everything[everything.superseded].observed_at < NOW).all()
    later = scanner.run_scan(settings, engine, "sample", now=NOW + timedelta(minutes=5), snapshots=[])
    assert later.current == [] and not db.load_opportunities(engine).superseded.any()      # an empty scan changes nothing
    assert len(db.load_opportunities(engine, latest_only=False)) == 22
    assert str(frame.observed_at.dt.tz) == "UTC" and frame[frame.status == "insufficient_data"].probability.isna().all()


def test_stale_prices_are_not_stored_as_current(settings, engine):
    snaps = sample.load(NOW - timedelta(hours=2))
    late = [type(s)(s.sport, NOW, s.event) for s in snaps]
    result = scanner.run_scan(settings, engine, "sample", now=NOW, snapshots=late)
    assert result.quotes == 0 and result.dropped["stale"] == 114 and result.opportunities == []


def test_paper_bets_use_the_observed_price_and_are_not_duplicated(settings, engine):
    scanner.run_scan(settings, engine, "sample", now=NOW)
    with Session(engine) as session, session.begin():
        row = session.scalars(select(db.OpportunityRow).where(db.OpportunityRow.player == "Sample Player H",
                                                              db.OpportunityRow.side == "over",
                                                              db.OpportunityRow.observed_at == NOW)).one()
        bet, created = db.place_paper_bet(session, row, stake=2.0, now=NOW)
        assert created and (bet.price, bet.book, bet.stake, bet.status, bet.entry_no) == (340, "draftkings", 2.0, "open", 1)
        assert bet.price_observed_at.replace(tzinfo=timezone.utc) == NOW and bet.probability == row.probability
        assert bet.strategy == db.STRATEGY and bet.data_source == "sample" and bet.profit is None
        same, created = db.place_paper_bet(session, row, stake=5.0)
        assert not created and same.id == bet.id and same.stake == 2.0
        row.link = "https://book.example/bet"
        extra, created = db.place_paper_bet(session, row, stake=1.0, allow_duplicate=True)
        assert extra.link == "https://book.example/bet" and bet.link is None
        assert created and extra.entry_no == 2
        with pytest.raises(ValueError):
            db.place_paper_bet(session, row, stake=0, allow_duplicate=True)
    saved = db.load_paper_bets(engine)
    assert len(saved) == 2 and set(saved.link) == {"https://book.example/bet"}     # the older bet takes its opportunity's link


# --- live source -----------------------------------------------------------------------------------
def live_client(settings, events, calls, **kw):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        headers = {"x-requests-remaining": "9000", "x-requests-used": "100", "x-requests-last": "0"}
        if request.url.path.endswith("/sports"):
            return httpx.Response(200, json=[], headers=headers)
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json=events if "basketball_nba" in request.url.path else [], headers=headers)
        cost = str(len(request.url.params["markets"].split(",")))
        event_id = request.url.path.split("/")[-2]
        return httpx.Response(200, json=dict(id=event_id, commence_time=events[0]["commence_time"], bookmakers=[]),
                              headers=dict(headers, **{"x-requests-last": cost}))
    return OddsApiClient(settings.api_key(), transport=httpx.MockTransport(handler), **kw)


def test_live_scan_only_asks_for_upcoming_games_inside_the_window(settings):
    stamp = lambda hours: (NOW + timedelta(hours=hours)).isoformat()
    events = [dict(id="started", commence_time=stamp(-1)), dict(id="soon", commence_time=stamp(2)),
              dict(id="next-week", commence_time=stamp(150)), dict(id="no-time")]
    calls = []
    client = live_client(settings, events, calls)
    snapshots, meta = asyncio.run(scanner.fetch_live(settings, ["nfl", "nba"], NOW, client))
    assert [s.event["id"] for s in snapshots] == ["soon"] and snapshots[0].sport == "nba"
    assert [c for c in calls if c.endswith("/odds")] == ["/v4/sports/basketball_nba/events/soon/odds"]
    assert meta == dict(credits_used=4, credits_remaining=9000, note=None)


def test_live_scan_stops_at_the_credit_budget(settings):
    events = [dict(id=f"g{i}", commence_time=(NOW + timedelta(hours=i + 1)).isoformat()) for i in range(3)]
    calls = []
    client = live_client(settings, events, calls, max_credits=8)
    snapshots, meta = asyncio.run(scanner.fetch_live(settings, ["nba"], NOW, client))
    assert [s.event["id"] for s in snapshots] == ["g0", "g1"] and meta["credits_used"] == 8
    assert "stopped early" in meta["note"] and len([c for c in calls if c.endswith("/odds")]) == 2


# --- saved snapshots -------------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, tables, rows):
        self.tables, self.rows, self.result = tables, rows, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        if "to_regclass" in sql:
            self.result = [(params[0] if params[0] in self.tables else None,)]
        else:
            self.result = self.rows["nba" if "sharp_multi_quotes" in sql else "nfl"]

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


class FakeConnection:
    def __init__(self, tables, rows):
        self.tables, self.rows, self.readonly, self.closed = tables, rows, False, False

    def set_session(self, readonly):
        self.readonly = readonly

    def cursor(self):
        return FakeCursor(self.tables, self.rows)

    def close(self):
        self.closed = True


def test_saved_snapshots_are_read_only_unwrapped_and_in_time_order():
    game = dict(id="g1", bookmakers=[dict(key="fanduel", markets=[])])
    rows = dict(nba=[(NOW, game), (NOW - timedelta(minutes=5), dict(id="lines", bookmakers=[]))],
                nfl=[(NOW - timedelta(minutes=10), dict(data=dict(game, id="g2"), timestamp="t"))])
    conn = FakeConnection({"bets.sharp_multi_quotes", "raw.nfl_api_responses"}, rows)
    found = saved_quotes.load("dsn", ["nfl", "nba", "mlb"], NOW - timedelta(hours=1), connect=lambda dsn: conn)
    assert [(s.sport, s.event["id"]) for s in found] == [("nfl", "g2"), ("nba", "g1")]      # empty game-line poll dropped
    assert conn.readonly and conn.closed
    missing = FakeConnection({"bets.sharp_multi_quotes"}, rows)
    assert {s.sport for s in saved_quotes.load("dsn", ["nfl", "nba"], NOW, connect=lambda dsn: missing)} == {"nba"}


def test_auto_uses_saved_prices_when_there_are_any_and_never_goes_live(settings, monkeypatch):
    snaps = sample.load(NOW)
    monkeypatch.setattr(scanner, "fetch_live", lambda *a, **k: pytest.fail("auto must not spend credits"))
    monkeypatch.setattr(saved_quotes, "load", lambda dsn, sports, since: snaps)
    assert scanner.collect("auto", settings, ["nba"], NOW)[:2] == ("saved", snaps)
    monkeypatch.setattr(saved_quotes, "load", lambda dsn, sports, since: [])
    source, found, meta = scanner.collect("auto", settings, ["nba"], NOW)
    assert source == "sample" and len(found) == 4 and meta["note"] == "hypothetical sample data"
    assert scanner.collect("saved", settings, ["nba"], NOW) == ("saved", [], dict(note="no saved snapshots in the look-back window"))

    def broken(dsn, sports, since):
        raise ConnectionError("database is down")
    monkeypatch.setattr(saved_quotes, "load", broken)
    assert scanner.collect("auto", settings, ["nba"], NOW)[0] == "sample"
    with pytest.raises(ValueError):
        scanner.collect("telepathy", settings, ["nba"], NOW)


# --- command line ----------------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SNIPER_DATABASE_URL", f"sqlite:///{(tmp_path / 'cli.db').as_posix()}")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_command_line_scans_ranks_and_saves_a_paper_bet(env):
    runner = CliRunner()
    assert "No scans yet" in runner.invoke(cli.app, ["top"]).output
    out = runner.invoke(cli.app, ["scan", "--source", "sample"])
    assert out.exit_code == 0 and "HYPOTHETICAL SAMPLE DATA" in out.output and "6 with a positive conservative EV" in out.output
    assert "Sample Player H Over 4.5 (Three-pointers) at DraftKings +340" in out.output
    top = runner.invoke(cli.app, ["top"])
    first_id = int(top.output.split("[")[1].split("]")[0])
    saved = runner.invoke(cli.app, ["paper-bet", str(first_id)])
    assert saved.exit_code == 0 and "Saved paper bet 1" in saved.output and "(simulated)" in saved.output
    assert "Already saved as paper bet 1" in runner.invoke(cli.app, ["paper-bet", str(first_id)]).output
    assert runner.invoke(cli.app, ["paper-bet", "99999"]).exit_code != 0
    assert runner.invoke(cli.app, ["scan", "--source", "nope"]).exit_code != 0


# --- dashboard -------------------------------------------------------------------------------------
def dashboard():
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    return AppTest.from_file(str(cli.Path(cli.__file__).resolve().parent / "dashboard" / "app.py"), default_timeout=90)


def test_dashboard_starts_empty_then_shows_a_labelled_sample_scan(env):
    app = dashboard().run()
    assert not app.exception and "No scans yet" in app.info[0].value
    app.button[[b.label for b in app.button].index("Scan sample data")].click().run()
    assert not app.exception and app.title[0].value.endswith("Sports Betting Sniper")
    assert "Hypothetical sample data" in app.warning[0].value
    assert [t.label for t in app.tabs] == ["Opportunity Board", "Longshot Hunter", "Line Movement", "Performance", "Paper Bets"]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Available now"] == "6" and metrics["Last scan"].endswith("sample")
    board = app.dataframe[0].value
    assert list(board["Odds"])[0] == "+340" and set(board["Evidence"]) == {"Strong"}      # speculative hidden by default
    assert set(board["Estimate from"]) == {"Market consensus"} and not board["Price seen"].str.contains("EXPIRED").any()
    outliers = next(d.value for d in app.dataframe if "Status" in d.value)
    assert set(outliers["Status"]) == {"Insufficient data"} and len(outliers) == 2


def test_dashboard_saves_a_paper_bet_once(env):
    engine = db.make_engine(get_settings().database_url)
    scanner.run_scan(get_settings(), engine, "sample")
    app = dashboard().run()
    app.button(key="board_save").click().run()
    assert not app.exception and "Saved paper bet 1" in app.success[0].value
    app.button(key="board_save").click().run()
    assert "Already saved as paper bet 1" in " ".join(i.value for i in app.info)
    assert len(db.load_paper_bets(engine)) == 1
    engine.dispose()
