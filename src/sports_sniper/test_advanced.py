"""Phase 4: the longshot ladder model, alerts, the Parlay Lab and the automatic evaluation."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from sports_sniper import alerts, breakout, cli, db, ladder, parlay, research
from sports_sniper.config import Settings, get_settings
from sports_sniper.providers import models
from sports_sniper.test_research import Connection
from sports_sniper.test_settlement import NOW, engine, env, fake_results, scan, seeded, settings  # noqa: F401

START = pd.Timestamp(datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc))
SEEN = START - timedelta(minutes=30)
NO_LINES = pd.DataFrame(columns=["day", "market", "ratio"])


# --- ladder model ------------------------------------------------------------------------------------
def test_the_ladder_estimate_is_anchored_to_the_main_line_and_falls_with_the_line():
    ratios = np.random.default_rng(1).gamma(2.0, 0.5, 4000)                 # results as a share of the line, median about 0.84
    p, se = ladder.estimate(ratios, [0.5, 0.5, 0.5, 0.35], [40.0, 40.0, 40.0, 40.0], [40.0, 80.0, 120.0, 40.0])
    assert p[0] == pytest.approx(0.5, abs=0.01) and p[3] == pytest.approx(0.35, abs=0.01)      # the anchor's own chance at its own line
    assert p[0] > p[1] > p[2] > 0 and 0.03 < p[1] < 0.20 and (se > 0).all() and se[2] < se[0]
    few, wide = ladder.estimate(ratios[:200], [0.5], [40.0], [80.0])
    assert wide[0] > se[1]                                                    # fewer past results: a wider error
    assert 0 < ladder.estimate(ratios, [0.5], [40.0], [4000.0])[0][0] < 0.001  # never exactly zero


def price_row(book, market, line, side, price, player="Sample Receiver", seen=SEEN, event="g1"):
    return dict(sport="nfl", event_id=event, event_name="Away Team @ Home Team", commence_time=START, book=book, market=market,
                player=player, player_key=player.lower(), line=line, line_key=f"{line:g}", side=side, price=price,
                observed_at=pd.Timestamp(seen), last_update=pd.Timestamp(seen), link=None)


def stored_prices() -> pd.DataFrame:
    return pd.DataFrame([
        price_row("pinnacle", "player_reception_yds", 40.5, "over", -110), price_row("pinnacle", "player_reception_yds", 40.5, "under", -110),
        price_row("fanduel", "player_reception_yds_alternate", 80.5, "over", 900),
        price_row("draftkings", "player_reception_yds_alternate", 80.5, "over", 700),
        price_row("fanduel", "player_reception_yds_alternate", 80.5, "over", 1000, seen=SEEN + timedelta(minutes=10)),   # a later look
        price_row("betmgm", "player_reception_yds_alternate", 80.5, "over", 2000),                                        # not a bet book
        price_row("fanduel", "player_reception_yds_alternate", 60.5, "over", 300, player="No Anchor"),                    # no main line
        price_row("fanduel", "player_rush_yds_alternate", 60.5, "over", 300),                                             # no anchor for rushing
    ])


def past_ratios(n: int, day: date, market: str = "player_reception_yds") -> pd.DataFrame:
    return pd.DataFrame(dict(day=day, market=market, ratio=np.random.default_rng(2).gamma(2.0, 0.5, n)))


def test_alternates_are_priced_only_with_an_anchor_and_only_from_earlier_results(settings):  # noqa: F811
    prices = stored_prices()
    alts = ladder.alternates(prices, settings)
    assert set(alts.book) == {"fanduel", "draftkings"} and set(alts.player) == {"Sample Receiver"} and len(alts) == 2     # the later look has no main line beside it
    assert alts.anchor_probability.iloc[0] == pytest.approx(0.5) and set(alts.anchor_line) == {40.5}
    game_day = START.tz_convert("America/New_York").date()
    same_day = ladder.price(alts, past_ratios(500, game_day), settings)
    assert same_day.probability.isna().all()                                   # results from the day itself are not known yet
    thin = ladder.price(alts, past_ratios(100, game_day - timedelta(days=7)), settings)
    assert thin.probability.isna().all()                                       # too few past results
    priced = ladder.price(alts, past_ratios(500, game_day - timedelta(days=7)), settings)
    assert priced.probability.notna().all() and priced.ratios_used.eq(500).all()
    assert (priced.conservative_ev < priced.ev).all() and priced.ev.iloc[0] == pytest.approx(priced.probability.iloc[0] * priced.decimal.iloc[0] - 1)
    today = ladder.current(prices, past_ratios(500, game_day), settings, 200)   # today's prices use every stored result
    assert len(today) == 2 and set(today.evidence) == {"speculative"} and set(today.source) == {"ladder_model"}
    assert ladder.current(prices.iloc[0:0], NO_LINES, settings).empty and ladder.price(alts, NO_LINES, settings).probability.isna().all()


def test_the_ladder_is_scored_once_per_line_and_reports_whether_its_picks_paid(settings):  # noqa: F811
    from sports_sniper.providers.results import GameResult
    final = GameResult(True, "final", stats={"sample receiver": dict(receiving_yards=95.0, rushing_yards=0, receptions=6, passing_tds=0, passing_yards=0)})
    game_day = START.tz_convert("America/New_York").date()
    out = ladder.evaluate(stored_prices(), past_ratios(500, game_day - timedelta(days=7)), settings, load_results=lambda dsn, games: {"g1": final})
    assert len(out) == 2 and set(out.status) == {"won"} and set(out.price) == {900, 700}        # the first look at each book
    check = ladder.summary(out, min_ev=0.04, longshot_american=200)
    assert check["lines"] == 2 and check["games"] == 1 and sum(b["n"] for b in check["calibration"]) == 2
    blind = dict(check["groups"])["Every longshot (+200 or longer), blind"]
    assert blind["won"] == 1 and blind["roi"] == pytest.approx(9.0)             # one bet per line, at the best price
    assert ladder.summary(out.iloc[0:0])["lines"] == 0 and ladder.evaluate(stored_prices().iloc[0:0], NO_LINES, settings).empty


def test_past_lines_come_from_the_nfl_record_and_replace_the_stored_ones():
    rows = [(date(2026, 9, 13), "receiving_yards", 50.0, 75.0), (date(2026, 9, 13), "receptions", 4.5, 9.0), (date(2026, 9, 13), "rushing_yards", 60.5, None)]
    history = ladder.nfl_line_history("dsn", connect=lambda dsn: Connection(rows))
    assert history.to_dict("records") == [dict(day=date(2026, 9, 13), market="player_reception_yds", ratio=1.5),
                                          dict(day=date(2026, 9, 13), market="player_receptions", ratio=2.0)]
    offers = pd.DataFrame([dict(commence_time=START, market=m, status="won", result_value=30.0, line=20.0) for m in ("player_reception_yds", "player_points")]
                          + [dict(commence_time=START, market="player_points", status="open", result_value=None, line=20.0)])
    both = ladder.combined_library(offers, history)
    assert both.groupby("market").ratio.first().to_dict() == {"player_points": 1.5, "player_reception_yds": 1.5, "player_receptions": 2.0}
    assert len(both) == 3 and len(ladder.combined_library(offers, None)) == 2 and ladder.library(offers.iloc[0:0]).empty


# --- alerts -------------------------------------------------------------------------------------------
def alert_settings(base: Settings, **over) -> Settings:
    return base.model_copy(update=dict(alert_webhook_url="https://discord.test/api/webhooks/secret-token",
                                       **{"alert_max_age_minutes": 30.0, **over}))


def test_alerts_are_off_without_an_address_and_sent_once_for_fresh_bets(settings, engine):  # noqa: F811
    now = NOW + timedelta(minutes=26)
    scan(settings, engine, now=now)
    assert alerts.send(engine, settings, now=now) == dict(status="off", sent=0)
    posted = []
    on = alert_settings(settings, alert_max_per_pass=2)
    first = alerts.send(engine, on, now=now, post=lambda url, text: posted.append((url, text)))
    assert first == dict(status="sent", sent=2) and len(posted) == 1
    url, text = posted[0]
    assert url.endswith("secret-token") and "secret-token" not in text and "research only" in text and text.count("•") == 2
    assert "starts in" in text and "fair" in text and "est. edge" in text
    with Session(engine) as session:
        marked = [b for b in session.scalars(select(db.PaperBet)) if b.alerted_at is not None]
        assert len(marked) == 2 and all(b.strategy == db.STRATEGY_AUTO and not b.replayed for b in marked)
    alerts.send(engine, on, now=now, post=lambda url, text: posted.append((url, text)))
    late = alerts.send(engine, alert_settings(settings), now=now + timedelta(minutes=30), post=lambda url, text: posted.append(text))
    assert late == dict(status="nothing new", sent=0)                           # the prices are no longer fresh
    with Session(engine) as session:
        assert all(b.alerted_at is None or db.as_utc(b.alerted_at) == now for b in session.scalars(select(db.PaperBet)))


def test_each_sport_can_have_its_own_channel(settings, engine):  # noqa: F811
    now = NOW + timedelta(minutes=26)
    scan(settings, engine, now=now)
    with Session(engine) as session:
        sports = {b.sport for b in alerts.pending(session, alert_settings(settings, alert_max_per_pass=50), now)}
    posted = []
    only_nba = settings.model_copy(update=dict(alert_webhook_url_nba="https://discord.test/nba-token", alert_max_age_minutes=30.0, alert_max_per_pass=50))
    out = alerts.send(engine, only_nba, now=now, post=lambda url, text: posted.append(url))
    assert posted == ["https://discord.test/nba-token"] and out["status"] == "sent"
    with Session(engine) as session:
        assert {b.sport for b in session.scalars(select(db.PaperBet)) if b.alerted_at is not None} == {"nba"}     # no channel for the rest: not sent
    both = only_nba.model_copy(update=dict(alert_webhook_url="https://discord.test/general-token"))
    more = alerts.send(engine, both, now=now, post=lambda url, text: posted.append(url))
    assert (more["sent"] > 0 and posted[1:] == ["https://discord.test/general-token"]) if sports - {"nba"} else more["status"] == "nothing new"
    assert alerts.address(both, "nba").endswith("nba-token") and alerts.address(both, "nfl").endswith("general-token")


def test_a_failed_alert_is_retried_and_never_shows_the_address(settings, engine):  # noqa: F811
    now = NOW + timedelta(minutes=26)
    scan(settings, engine, now=now)

    def down(url, text):
        raise ConnectionError(f"could not reach {url}")
    failed = alerts.send(engine, alert_settings(settings), now=now, post=down)
    assert failed == dict(status="failed (ConnectionError)", sent=0)
    with Session(engine) as session:
        assert all(b.alerted_at is None for b in session.scalars(select(db.PaperBet)))
    assert alerts.send(engine, alert_settings(settings), now=now, post=lambda url, text: None)["sent"] > 0
    replayed = alerts.send(engine, alert_settings(settings, alert_min_ev=0.9), now=now, post=lambda url, text: None)
    assert replayed["status"] == "nothing new"


# --- parlay lab -----------------------------------------------------------------------------------------
def test_joint_probability_is_the_product_when_independent_and_moves_with_the_link():
    independent = parlay.joint_probability([0.5, 0.4], np.eye(2))
    assert independent == pytest.approx(0.20, abs=0.005)
    together = parlay.joint_probability([0.5, 0.4], np.array([[1, 0.6], [0.6, 1]]))
    apart = parlay.joint_probability([0.5, 0.4], np.array([[1, -0.6], [-0.6, 1]]))
    assert apart < independent < together < 0.4 and parlay.joint_probability([0.5, 0.5], np.array([[1, 0.999], [0.999, 1]])) == pytest.approx(0.5, abs=0.02)
    clashing = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])       # links that cannot all be true together
    assert 0 <= parlay.joint_probability([0.5, 0.5, 0.5], clashing) <= 0.5


def game_logs(games: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for g in range(games):
        pace = rng.normal()                                                    # a fast game lifts everyone
        for team in ("AAA", "BBB"):
            share = rng.normal()                                               # one teammate's gain is the other's loss
            for n, sign in enumerate((1, -1)):
                form = rng.normal()
                rows.append(dict(season="2026-27", game_id=f"g{g}", team=team, player=f"{team}-{n}", weight=30.0,
                                 pts=20 + 4 * form + 2 * pace + 3 * sign * share + rng.normal(), reb=6 + rng.normal(),
                                 ast=5 + 1.5 * form + rng.normal(), fg3m=2 + 0.8 * form + rng.normal()))
    return pd.DataFrame(rows)


def test_links_are_measured_for_the_same_player_teammates_and_opponents(monkeypatch):
    monkeypatch.setattr(parlay, "MIN_PAIRS", 50)
    table = parlay.correlations(game_logs(), "nba")
    assert table["same|player_assists|player_points"]["r"] > 0.4 and abs(table["same|player_points|player_rebounds"]["r"]) < 0.25
    assert table["teammates|player_points|player_points"]["r"] < -0.1          # they share the ball
    assert table["opponents|player_points|player_points"]["r"] > 0.02          # the pace of the game lifts both sides
    assert table["teammates|player_points|player_points"]["n"] == 120 and "same|player_points|player_points" not in table
    monkeypatch.setattr(parlay, "MIN_PAIRS", 10 ** 6)
    assert parlay.correlations(game_logs(), "nba") == {}                        # too few pairs: unknown, not guessed


def leg(player, market, side="over", team="AAA", probability=0.5, event="g1"):
    return dict(probability=probability, event_id=event, player_key=player, team=team, market=market, side=side)


def test_a_parlay_uses_the_links_and_reverses_them_for_an_under():
    table = {"nba": {"same|player_assists|player_points": dict(r=0.5, n=1000), "teammates|player_points|player_points": dict(r=-0.3, n=1000)}}
    same = parlay.evaluate([leg("a", "player_points"), leg("a", "player_assists")], table, "nba", price=300)
    assert same["joint"] > same["independent"] == pytest.approx(0.25) and same["ev"] > same["ev_if_independent"] == pytest.approx(0.0)
    assert same["links"][0]["correlation"] == 0.5 and "same player" in same["links"][0]["why"]
    mixed = parlay.evaluate([leg("a", "player_points"), leg("a", "player_assists", side="under")], table, "nba")
    assert mixed["joint"] < 0.25 and mixed["links"][0]["correlation"] == -0.5 and mixed["ev"] is None
    mates = parlay.evaluate([leg("a", "player_points"), leg("b", "player_points")], table, "nba")
    assert mates["joint"] < 0.25 and "teammates" in mates["links"][0]["why"]
    for other, why in ((leg("c", "player_points", team="BBB"), "no measured link"), (leg("c", "player_points", team=None), "team unknown"),
                       (leg("c", "player_points", event="g2"), "different games")):
        out = parlay.evaluate([leg("a", "player_points"), other], table, "nba")
        assert out["joint"] == out["independent"] and why in out["links"][0]["why"]
    with pytest.raises(ValueError):
        parlay.evaluate([leg("a", "player_points"), leg("a", "player_points", side="under")], table, "nba")
    with pytest.raises(ValueError):
        parlay.evaluate([leg("a", "player_points")], table, "nba")


def test_links_are_saved_and_teams_come_from_the_latest_game(tmp_path):
    path = parlay.default_path(f"sqlite:///{(tmp_path / 'sniper.db').as_posix()}")
    assert parlay.load(path) == {} and parlay.default_path("sqlite:///:memory:") is None
    parlay.save({"nba": {"same|a|b": dict(r=0.2, n=500)}}, path)
    assert parlay.load(path)["nba"]["same|a|b"]["r"] == 0.2
    history = {"nba": pd.DataFrame([dict(day=date(2026, 11, 1), player_key="traded player", team="AAA"),
                                    dict(day=date(2026, 11, 9), player_key="traded player", team="BBB")])}
    assert parlay.teams_from(history) == {("nba", "traded player"): "BBB"}


# --- automatic evaluation -------------------------------------------------------------------------------
@pytest.fixture
def offline(monkeypatch):
    """No model, game-log or line-history database: everything is worked out from the Sniper's own tables."""
    monkeypatch.setattr(models, "ADAPTERS", ())
    monkeypatch.setattr(breakout, "load_history", lambda dsn, connect=None: {})
    monkeypatch.setattr(ladder, "nfl_line_history", lambda dsn, connect=None: NO_LINES)
    monkeypatch.setattr(parlay, "build", lambda dsn, connect=None: {"nba": {"same|a|b": dict(r=0.1, n=999)}})


def test_the_evaluation_is_due_once_and_then_only_when_it_is_old(settings, engine, offline):  # noqa: F811
    scan(settings, engine)
    assert research.due(settings)
    done = research.evaluate(engine, settings, load_results=fake_results, write_report=False)
    assert done["flagged"] > 0 and done["calibration_saved"] is False and done["parlay_links"] == {"nba": 1} and "report" not in done
    assert not research.due(settings) and parlay.load(parlay.default_path(settings.database_url))["nba"]
    marker = parlay.default_path(settings.database_url).with_name("evaluation.json")
    old = (datetime.now(timezone.utc) - timedelta(hours=30)).timestamp()
    os.utime(marker, (old, old))
    assert research.due(settings) and not research.due(settings.model_copy(update={"evaluate_every_hours": 48.0}))
    assert not research.due(settings.model_copy(update={"database_url": "sqlite:///:memory:"}))


def test_watch_runs_the_daily_evaluation_and_the_evaluate_command(env, offline, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(research, "REPORT", tmp_path / "report.md")
    monkeypatch.setattr(research.write, "__defaults__", (tmp_path / "report.md",))
    seeded().dispose()
    out = CliRunner().invoke(cli.app, ["watch"])
    assert out.exit_code == 0 and "daily evaluation:" in out.output and out.output.strip().endswith("done")
    again = CliRunner().invoke(cli.app, ["watch"])
    assert again.exit_code == 0 and "daily evaluation" not in again.output           # not due again today
    manual = CliRunner().invoke(cli.app, ["evaluate"])
    assert manual.exit_code == 0 and "Replayed" in manual.output and "Parlay links: {'nba': 1}" in manual.output
    assert (tmp_path / "report.md").exists() and "## Longshot ladder model" in (tmp_path / "report.md").read_text(encoding="utf-8")

    def broken(engine, settings, **kwargs):
        raise RuntimeError("no database")
    monkeypatch.setattr(research, "evaluate", broken)
    monkeypatch.setattr(research, "due", lambda settings, now=None: True)
    failed = CliRunner().invoke(cli.app, ["watch"])
    assert failed.exit_code == 0 and "daily evaluation failed (RuntimeError)" in failed.output    # the pass itself still finishes
    get_settings.cache_clear()


def test_dashboard_parlay_lab_and_ladder_section(env, offline):  # noqa: F811
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    seeded().dispose()
    st.cache_resource.clear()
    st.cache_data.clear()
    page = str(cli.Path(cli.__file__).resolve().parent / "dashboard" / "app.py")
    app = AppTest.from_file(page, default_timeout=120).run()
    assert not app.exception and any("have not been measured yet" in i.value for i in app.info)
    parlay.save({"nba": {"same|player_assists|player_points": dict(r=0.4, n=5000)}}, parlay.default_path(get_settings().database_url))
    app = AppTest.from_file(page, default_timeout=120).run()
    app.checkbox[[c.label for c in app.checkbox].index("Include prices no longer available")].check().run()
    for label in ("Load the legs", "Price longshots with the ladder model", "Load the radar"):
        app.button[[b.label for b in app.button].index(label)].click().run()
        assert not app.exception, label
    assert any(s.label == "Game" for s in app.selectbox) and any(m.label == "Legs (2 to 4)" for m in app.multiselect)
