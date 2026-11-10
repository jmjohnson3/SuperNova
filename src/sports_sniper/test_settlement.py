"""Settling paper bets, closing line value, automatic paper entries and the performance summary."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from sports_sniper import cli, db, performance, scanner, settlement
from sports_sniper.config import Settings, get_settings
from sports_sniper.providers import Snapshot, results, sample, saved_quotes
from sports_sniper.providers.results import Game, GameResult

NOW = datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)
NBA, NFL = "sample-nba-001", "sample-nfl-001"


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, database_url=f"sqlite:///{(tmp_path / 'sniper.db').as_posix()}")


@pytest.fixture
def engine(settings):
    eng = db.make_engine(settings.database_url)
    db.init_db(eng)
    yield eng
    eng.dispose()


def restamp(snap: Snapshot, when: datetime, prices: dict | None = None) -> Snapshot:
    """The same game looked at again at `when`, optionally with some prices changed: {(book, market, player, name): price}."""
    event = copy.deepcopy(snap.event)
    stamp = (when - timedelta(seconds=40)).isoformat()
    for book in event["bookmakers"]:
        book["last_update"] = stamp
        for market in book["markets"]:
            market["last_update"] = stamp
            for outcome in market["outcomes"]:
                key = (book["key"], market["key"], outcome["description"], outcome["name"])
                outcome["price"] = (prices or {}).get(key, outcome["price"])
    return Snapshot(snap.sport, when, event)


def real_snapshots():
    """The sample games presented as saved real snapshots, plus a last look at the NBA game 10 minutes before it starts
    in which Pinnacle has moved toward Sample Player A's over."""
    snaps = sample.load(NOW)
    nba_now = next(s for s in snaps if s.sport == "nba" and s.observed_at == NOW)
    close = restamp(nba_now, NOW + timedelta(minutes=25), {("pinnacle", "player_points", "Sample Player A", "Over"): -150,
                                                           ("pinnacle", "player_points", "Sample Player A", "Under"): 125,
                                                           ("draftkings", "player_points", "Sample Player A", "Over"): -115})
    return snaps + [close]


def scan(settings, engine, now=NOW + timedelta(minutes=26)):
    return scanner.run_scan(settings, engine, "saved", now=now, snapshots=real_snapshots())


def bets(engine) -> dict:
    with Session(engine) as session:
        return {(b.player, b.side): b for b in session.scalars(select(db.PaperBet))}


FINAL = GameResult(True, "final", stats={"sample player a": dict(pts=30, reb=4, ast=5, fg3m=2),
                                         "sample player b": dict(pts=11, reb=8, ast=1, fg3m=0)},
                   known={"sample player a", "sample player b", "sample player h"})


def fake_results(dsn, games):
    return {g.event_id: FINAL if g.event_id == NBA else GameResult(reason="the game is not final yet") for g in games}


# --- grading rules ---------------------------------------------------------------------------------
def test_grading_and_profit():
    assert [settlement.grade("over", 24.5, v) for v in (25, 24, 24.5)] == ["won", "lost", "push"]
    assert [settlement.grade("under", 8.5, v) for v in (8, 9)] == ["won", "lost"]
    assert settlement.grade("under", 8.0, 8.0) == "push" and settlement.grade("yes", None, 1) is None
    assert settlement.profit("won", 2.0, 4.4) == pytest.approx(6.8) and settlement.profit("lost", 2.0, 4.4) == -2.0
    assert settlement.profit("push", 2.0, 4.4) == 0.0 and settlement.profit("void", 2.0, 4.4) == 0.0
    assert settlement.stat_names("nba", "player_points_alternate") == ("pts",)
    assert settlement.stat_names("nba", "player_points_rebounds_assists") == ("pts", "reb", "ast")
    assert settlement.stat_names("nfl", "player_reception_yds") == ("receiving_yards",)
    assert settlement.stat_names("nfl", "player_anytime_td") is None and settlement.stat_names("mlb", "batter_hits") is None


def bet(**over):
    base = dict(sport="nba", market="player_points", side="over", line=24.5, player_key="sample player a")
    return SimpleNamespace(**dict(base, **over))


def test_a_bet_is_only_settled_from_a_complete_game_and_a_known_player():
    assert settlement.decide(bet(), FINAL) == ("won", 30.0, "final")
    assert settlement.decide(bet(side="under"), FINAL) == ("lost", 30.0, "final")
    assert settlement.decide(bet(market="player_points_rebounds_assists", line=38.5), FINAL) == ("won", 39.0, "final")
    assert settlement.decide(bet(), None) == ("open", None, "no results source available")
    assert settlement.decide(bet(), GameResult(reason="the box score is not complete yet")) == ("open", None, "the box score is not complete yet")
    assert settlement.decide(bet(player_key="sample player h"), FINAL) == ("void", None, "did not play")
    assert settlement.decide(bet(player_key="somebody else"), FINAL) == ("open", None, "player not found in the results")
    twins = GameResult(True, "final", stats=FINAL.stats, ambiguous={"sample player a"}, known=FINAL.known)
    assert settlement.decide(bet(), twins)[0] == "open"
    assert settlement.decide(bet(sport="nfl", market="player_anytime_td", side="yes", line=None), FINAL) == (
        "open", None, "this market cannot be settled yet")


# --- results adapter -------------------------------------------------------------------------------
class Cursor:
    def __init__(self, answers):
        self.answers, self.result = answers, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.result = next((rows for needle, rows in self.answers if needle in sql), [])

    def fetchone(self):
        return self.result[0] if self.result else (None,)

    def fetchall(self):
        return self.result


class Connection:
    def __init__(self, answers):
        self.answers, self.readonly, self.closed = answers, False, False

    def set_session(self, readonly):
        self.readonly = readonly

    def cursor(self):
        return Cursor(self.answers)

    def close(self):
        self.closed = True


def nba_rows(home_minutes=240.0):
    return [("g1", "DEN", "Nikola Jokić", home_minutes - 200, 31, 12, 9, 1), ("g1", "DEN", "Jamal Murray", 200.0, 22, 3, 6, 4),
            ("g1", "UTA", "Lauri Markkanen", 240.0, 18, 7, 2, 3)]


def nba_game():
    return Game("nba", "e1", "Denver Nuggets", "Utah Jazz", datetime(2026, 11, 10, 2, 0, tzinfo=timezone.utc))


def test_nba_results_need_a_whole_box_score():
    answers = [("ingest_conflicts')", [(None,)]), ("SELECT DISTINCT player_name", [("Nikola Jokić",), ("Aaron Gordon",)]),
               ("FROM nbaprops.player_games\n", nba_rows())]
    conn = Connection(answers)
    game = results.load("dsn", [nba_game()], connect=lambda dsn: conn)["e1"]
    assert game.complete and conn.readonly and conn.closed
    assert game.stats["nikola jokic"] == dict(pts=31.0, reb=12.0, ast=9.0, fg3m=1.0) and "aaron gordon" in game.known
    assert nba_game().day.isoformat() == "2026-11-09"                      # the evening before in Eastern time
    partial = results.load("dsn", [nba_game()], connect=lambda dsn: Connection(
        [("ingest_conflicts')", [(None,)]), ("FROM nbaprops.player_games\n", nba_rows(home_minutes=215.0))]))["e1"]
    assert not partial.complete and partial.reason == "the box score is not complete yet" and partial.stats == {}
    conflicted = results.load("dsn", [nba_game()], connect=lambda dsn: Connection(
        [("ingest_conflicts')", [("nbaprops.ingest_conflicts",)]), ("WHERE NOT resolved", [("g1",)]),
         ("FROM nbaprops.player_games\n", nba_rows())]))["e1"]
    assert not conflicted.complete and "conflict" in conflicted.reason
    nothing = results.load("dsn", [nba_game()], connect=lambda dsn: Connection([("ingest_conflicts')", [(None,)])]))["e1"]
    assert not nothing.complete and nothing.reason == "no result for this game yet"
    unknown = Game("nba", "e2", "Springfield Isotopes", "Utah Jazz", NOW)
    assert results.load("dsn", [unknown], connect=lambda dsn: Connection([]))["e2"].reason == "team name not recognised"
    assert results.load("dsn", [Game("mlb", "e3", "A", "B", NOW)], connect=lambda dsn: Connection([]))["e3"].complete is False


def test_nfl_results_need_a_final_score_and_use_roster_names():
    game = Game("nfl", "e1", "Minnesota Vikings", "Buffalo Bills", datetime(2026, 11, 10, 1, 15, tzinfo=timezone.utc))
    box = [("Khalil Shakir", 58.0, 0.0, 5.0, 0.0, 0.0), ("Josh Allen", 0.0, 31.0, 0.0, 2.0, 244.0)]
    final = [("FROM raw.nfl_games", [("2026_09_BUF_MIN", 2026, "final", 24, 27)]), ("FROM raw.nfl_player_gamelogs", box),
             ("FROM raw.nfl_rosters WHERE season", [("Khalil Shakir",), ("Dalton Kincaid",)])]
    found = results.load("dsn", [game], connect=lambda dsn: Connection(final))["e1"]
    assert found.complete and found.stats["khalil shakir"]["receiving_yards"] == 58.0 and found.stats["josh allen"]["passing_tds"] == 2.0
    assert found.known == {"khalil shakir", "dalton kincaid"}
    pending = [("FROM raw.nfl_games", [("2026_09_BUF_MIN", 2026, "scheduled", None, None)])]
    assert results.load("dsn", [game], connect=lambda dsn: Connection(pending))["e1"].reason == "the game is not final yet"
    no_box = [("FROM raw.nfl_games", [("2026_09_BUF_MIN", 2026, "final", 24, 27)])]
    assert results.load("dsn", [game], connect=lambda dsn: Connection(no_box))["e1"].reason == "official box score not loaded yet"
    assert results.load(None, [], connect=lambda dsn: pytest.fail("no games, no query")) == {}


# --- automatic paper bets --------------------------------------------------------------------------
def test_flagged_bets_are_entered_once_at_the_first_price_that_qualified(settings, engine):
    first = scan(settings, engine)
    found = bets(engine)
    assert first.paper_bets == len(found) == 5 and {b.strategy for b in found.values()} == {db.STRATEGY_AUTO}
    assert set(found) == {("Sample Player H", "over"), ("Sample Player B", "over"), ("Sample Player A", "over"),
                          ("Sample Player F", "under"), ("Sample Player G", "over")}     # the retail-only one is speculative
    h, a = found[("Sample Player H", "over")], found[("Sample Player A", "over")]
    assert (h.price, h.book, db.as_utc(h.price_observed_at)) == (340, "draftkings", NOW - timedelta(minutes=30))
    assert (a.price, db.as_utc(a.price_observed_at)) == (105, NOW)          # -110 half an hour earlier did not qualify
    assert not any(b.replayed for b in found.values()) and {b.status for b in found.values()} == {"open"}
    assert scan(settings, engine).paper_bets == 0 and len(bets(engine)) == 5


def test_sample_data_and_switched_off_settings_enter_nothing(settings, engine):
    assert scanner.run_scan(settings, engine, "sample", now=NOW).paper_bets == 0 and bets(engine) == {}
    off = Settings(_env_file=None, database_url=settings.database_url, auto_paper=False)
    assert scan(off, engine).paper_bets == 0 and bets(engine) == {}
    late = scanner.run_scan(settings, engine, "saved", now=NOW + timedelta(hours=3), snapshots=real_snapshots())
    assert late.paper_bets == 0                                            # those opportunities belong to the earlier scan
    fresh = db.make_engine(settings.database_url.replace("sniper.db", "replay.db"))
    scanner.run_scan(settings, fresh, "saved", now=NOW + timedelta(hours=3), snapshots=real_snapshots())
    assert all(b.replayed for b in bets(fresh).values()) and len(bets(fresh)) == 5
    fresh.dispose()


# --- closing line value and settlement -------------------------------------------------------------
def test_settlement_grades_final_games_and_measures_the_close(settings, engine):
    scan(settings, engine)
    counts = settlement.settle(engine, settings, now=NOW + timedelta(hours=4), load_results=fake_results)
    assert counts == dict(closing_values=3, won=1, lost=1, void=1, open=2, checked=5)
    found = bets(engine)
    a, b, h = found[("Sample Player A", "over")], found[("Sample Player B", "over")], found[("Sample Player H", "over")]
    assert (a.status, a.result_value, a.profit) == ("won", 30.0, pytest.approx(1.05))
    assert (b.status, b.result_value, b.profit) == ("lost", 8.0, -1.0)
    assert (h.status, h.result_value, h.profit, h.settle_note) == ("void", None, 0.0, "did not play")
    nfl = found[("Sample Player F", "under")]
    assert (nfl.status, nfl.profit, nfl.settle_note, nfl.closing_line_value) == ("open", None, "the game is not final yet", None)
    assert db.as_utc(a.closing_observed_at) == NOW + timedelta(minutes=25) and a.closing_price == -115
    assert a.closing_line_value > a.ev > 0                     # Pinnacle moved toward the over after the bet
    assert a.closing_line_value == pytest.approx(a.closing_probability * a.decimal - 1)
    assert b.closing_price == 135 and b.closing_line_value == pytest.approx(b.ev)      # nothing moved for this one
    again = settlement.settle(engine, settings, now=NOW + timedelta(hours=5), load_results=fake_results)
    assert again == dict(open=2, checked=2) and bets(engine)[("Sample Player A", "over")].profit == pytest.approx(1.05)


def test_snapshots_in_another_time_zone_are_stored_as_utc_and_still_find_the_close(settings, engine):
    pacific = timezone(timedelta(hours=-8))
    local = [Snapshot(s.sport, s.observed_at.astimezone(pacific), s.event) for s in real_snapshots()]
    scanner.run_scan(settings, engine, "saved", now=(NOW + timedelta(minutes=26)).astimezone(pacific), snapshots=local)
    assert db.load_prices(engine).observed_at.max() == NOW + timedelta(minutes=25)
    counts = settlement.settle(engine, settings, now=NOW + timedelta(hours=4), load_results=fake_results)
    assert counts["closing_values"] == 3 and not any(b.replayed for b in bets(engine).values())
    assert db.as_utc(bets(engine)[("Sample Player A", "over")].price_observed_at) == NOW


def test_no_closing_value_without_a_later_look_inside_the_window(settings, engine):
    scanner.run_scan(settings, engine, "saved", now=NOW, snapshots=sample.load(NOW))     # last look is 35 minutes before tip
    counts = settlement.settle(engine, settings, now=NOW + timedelta(hours=4), load_results=fake_results)
    assert "closing_values" not in counts and all(b.closing_line_value is None for b in bets(engine).values())
    assert settlement.settle(engine, settings, now=NOW - timedelta(hours=1), load_results=fake_results) == dict(checked=0)


def test_sample_bets_and_missing_results_stay_open(settings, engine):
    hypothetical = db.make_engine(settings.database_url.replace("sniper.db", "sample.db"))
    scanner.run_scan(settings, hypothetical, "sample", now=NOW)
    with Session(hypothetical) as session, session.begin():
        row = session.scalars(select(db.OpportunityRow).where(db.OpportunityRow.status == "priced")).first()
        db.place_paper_bet(session, row, stake=1.0, now=NOW)
    assert settlement.settle(hypothetical, settings, now=NOW + timedelta(days=2), load_results=fake_results) == dict(checked=0)
    assert {b.status for b in bets(hypothetical).values()} == {"open"}
    hypothetical.dispose()
    scan(settings, engine)

    def down(dsn, games):
        raise ConnectionError("results database is down")
    counts = settlement.settle(engine, settings, now=NOW + timedelta(hours=4), load_results=down)
    assert counts["results_unavailable"] == 1 and counts["checked"] == 5 and "won" not in counts
    real = [b for b in bets(engine).values() if b.data_source != "sample"]
    assert {b.status for b in real} == {"open"} and {b.settle_note for b in real} == {"results unavailable (ConnectionError)"}


# --- performance -----------------------------------------------------------------------------------
def frame(rows):
    cols = ["event_id", "status", "stake", "decimal", "profit", "ev", "probability", "closing_line_value", "sport"]
    return pd.DataFrame(rows, columns=cols)


def test_performance_summary_counts_only_settled_stakes_and_clusters_by_game():
    data = frame([("g1", "won", 1.0, 2.0, 1.0, 0.05, 0.55, 0.04, "nba"), ("g1", "lost", 1.0, 2.0, -1.0, 0.05, 0.55, 0.02, "nba"),
                  ("g2", "won", 2.0, 3.0, 4.0, 0.10, 0.40, -0.01, "nfl"), ("g2", "push", 1.0, 2.0, 0.0, 0.03, 0.52, None, "nfl"),
                  ("g3", "void", 1.0, 2.0, 0.0, 0.04, 0.53, 0.03, "nba"), ("g4", "open", 1.0, 2.0, None, 0.04, 0.53, None, "nba")])
    s = performance.summarize(data)
    assert (s["bets"], s["open"], s["won"], s["lost"], s["push"], s["void"]) == (6, 1, 2, 1, 1, 1)
    assert s["win_rate"] == pytest.approx(2 / 3) and s["win_interval"][0] < 2 / 3 < s["win_interval"][1]
    assert s["staked"] == 5.0 and s["profit"] == 4.0 and s["roi"] == pytest.approx(0.8)
    assert s["closing_value"] == pytest.approx(0.02) and s["closing_count"] == 4 and s["beat_close"] == 0.75
    assert s["estimated_ev"] == pytest.approx((0.05 + 0.05 + 0.10 + 0.03) / 4)
    per_stake = pd.Series([1.0, -1.0, 2.0, 0.0])
    by_game = [(1.0 - 0.5) + (-1.0 - 0.5), (2.0 - 0.5) + (0.0 - 0.5)]
    assert per_stake.mean() == 0.5 and s["roi_se"] == pytest.approx((by_game[0] ** 2 + by_game[1] ** 2) ** 0.5 / 4)
    assert s["brier"] == pytest.approx(((0.55 - 1) ** 2 + 0.55 ** 2 + (0.40 - 1) ** 2) / 3)
    assert s["break_even_brier"] == pytest.approx(((0.5 - 1) ** 2 + 0.5 ** 2 + (1 / 3 - 1) ** 2) / 3)
    empty = performance.summarize(data.iloc[:0])
    assert empty["bets"] == 0 and empty["roi"] is None and empty["closing_value"] is None and empty["win_rate"] is None
    by_sport = performance.breakdown(data, "sport").set_index("sport")
    assert by_sport.loc["nfl", "profit"] == 4.0 and by_sport.loc["nba", "bets"] == 4
    bands = performance.calibration(data, bins=2)
    assert [(b["n"], b["observed"]) for b in bands] == [(1, 1.0), (2, 0.5)]


# --- upgrading an older database -------------------------------------------------------------------
def test_an_older_database_is_brought_up_to_date(settings, engine):
    with engine.begin() as conn:
        for column in ("closing_probability", "closing_observed_at", "settle_note", "replayed"):
            conn.execute(text(f"ALTER TABLE sniper_paper_bets DROP COLUMN {column}"))
        conn.execute(text("CREATE TABLE sniper_quotes (id INTEGER PRIMARY KEY, price INTEGER)"))
    db.init_db(engine)
    look = inspect(engine)
    assert not look.has_table("sniper_quotes") and look.has_table("sniper_prices")
    assert {"closing_probability", "closing_observed_at", "settle_note", "replayed"} <= {c["name"] for c in look.get_columns("sniper_paper_bets")}
    assert scan(settings, engine).paper_bets == 5 and not any(b.replayed for b in bets(engine).values())


# --- command line and dashboard --------------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SNIPER_DATABASE_URL", f"sqlite:///{(tmp_path / 'cli.db').as_posix()}")
    monkeypatch.setattr(saved_quotes, "load", lambda dsn, sports, since: [])
    monkeypatch.setattr(results, "load", fake_results)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def seeded():
    """A database with the sample games scanned as real prices and settled four hours later."""
    config = get_settings()
    engine = db.make_engine(config.database_url)
    scan(config, engine, now=datetime.now(timezone.utc) - timedelta(hours=6))
    return engine


def test_settle_command_reports_the_record(env):
    engine = seeded()
    with Session(engine) as session, session.begin():           # the games in this database started long ago
        for row in session.scalars(select(db.PaperBet)):
            row.commence_time = datetime.now(timezone.utc) - timedelta(hours=5)
    out = CliRunner().invoke(cli.app, ["settle"])
    assert out.exit_code == 0, out.output
    assert "Checked 5 open paper bets: 1 won, 1 lost, 0 push, 1 void, 2 still open." in out.output
    assert f"{db.STRATEGY_AUTO}: 5 bets, 1-1-0 (1 void, 2 open), simulated profit +0.05 on 2 staked (+2.5%)" in out.output
    engine.dispose()


def test_dashboard_shows_settled_performance(env):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    engine = seeded()
    with Session(engine) as session, session.begin():
        for row in session.scalars(select(db.PaperBet)):
            row.commence_time = datetime.now(timezone.utc) - timedelta(hours=5)
    st.cache_resource.clear()
    app = AppTest.from_file(str(cli.Path(cli.__file__).resolve().parent / "dashboard" / "app.py"), default_timeout=90).run()
    assert not app.exception
    app.button[[b.label for b in app.button].index("Settle paper bets")].click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Paper bets"] == "5" and metrics["Record (W-L-P)"] == "1-1-0" and metrics["Simulated profit"] == "+$0.05"
    tables = [d.value for d in app.dataframe]
    summary = next(t for t in tables if "Group" in t and "All" in set(t["Group"]))
    assert summary.set_index("Group").loc["All", "Void"] == 1
    paper = next(t for t in tables if "Note" in t)
    assert set(paper["Status"]) == {"Won", "Lost", "Void", "Open"} and "did not play" in set(paper["Note"])
    assert "Not enough settled bets" in " ".join(m.value for m in app.markdown)
    engine.dispose()
