"""Phase 3: backtest, walk-forward, paper bankroll and model against market."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from typer.testing import CliRunner

from sports_sniper import backtest, bankroll, breakout, cli, db, model_compare, research
from sports_sniper.backtest import Strategy
from sports_sniper.config import Settings
from sports_sniper.providers import models
from sports_sniper.test_settlement import NBA, NOW, engine, env, fake_results, real_snapshots, scan, seeded, settings  # noqa: F401

START = datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)


def flagged(n: int, *, day: int = 0, ev: float = 0.05, status: str = "won", price: int = -110, book: str = "fanduel",
            side: str = "over", minutes: int = 30, clv: float | None = 0.02, delay_price: int | None = None, **over) -> dict:
    """One flagged price for player n in the game on `day`."""
    start = START + timedelta(days=day)
    decimal = 1 + (100 / -price if price < 0 else price / 100)
    row = dict(sport="nba", event_id=f"game-{day}", event_name="Away @ Home", commence_time=start, book=book,
               market="player_points", player=f"Player {n}", player_key=f"player {n}", line=20.5, line_key="20.5", side=side,
               price=price, decimal=decimal, observed_at=start - timedelta(minutes=minutes), probability=(1 + ev) / decimal,
               standard_error=0.015, ev=ev, conservative_ev=ev - 0.01, evidence="strong", minutes_to_start=float(minutes),
               closing_line_value=clv, status=status, delay_price=delay_price,
               delay_decimal=None if delay_price is None else 1 + (100 / -delay_price if delay_price < 0 else delay_price / 100),
               delay_observed_at=None if delay_price is None else start - timedelta(minutes=minutes - 10),
               delay_closing_line_value=None if delay_price is None else 0.01)
    return dict(row, **over)


def frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for column in ("commence_time", "observed_at", "delay_observed_at"):
        out[column] = pd.to_datetime(out[column], utc=True)
    for column in ("delay_price", "delay_decimal", "delay_closing_line_value", "closing_line_value"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


# --- applying a rule -------------------------------------------------------------------------------
def test_a_rule_enters_once_at_the_first_price_that_met_it_and_at_the_best_book():
    found = frame([flagged(1, ev=0.02, minutes=40), flagged(1, ev=0.05, minutes=30, price=-105), flagged(1, ev=0.06, minutes=20, price=100),
                   flagged(1, ev=0.05, minutes=30, price=-102, book="draftkings"), flagged(2, ev=0.05, book="betmgm")])
    bets = backtest.run(found, Strategy(min_ev=0.04), ("fanduel", "draftkings"))
    assert len(bets) == 1                                           # player 2 is at a book that is not bet at
    assert bets.iloc[0].book == "draftkings" and bets.iloc[0].price == -102 and bets.iloc[0].minutes_to_start == 30
    loose = backtest.run(found, Strategy(min_ev=0.0), ("fanduel", "draftkings"))
    assert loose.iloc[0].minutes_to_start == 40                     # a looser rule enters at the earlier look
    assert len(backtest.run(found, Strategy(min_ev=0.04, books=backtest.ANY_BOOK))) == 2
    assert backtest.run(found.iloc[0:0], Strategy()).empty and backtest.metrics(backtest.run(found.iloc[0:0], Strategy()))["bets"] == 0


def test_filters_and_profit():
    found = frame([flagged(1, side="under", status="lost"), flagged(2, status="won", price=150), flagged(3, status="push"),
                   flagged(4, status="open"), flagged(5, status="void", sport="nfl")])
    bets = backtest.run(found, Strategy(min_ev=0.04), ("fanduel",))
    assert bets.set_index("player_key").profit.round(2).to_dict() == {"player 1": -1.0, "player 2": 1.5, "player 3": 0.0,
                                                                     "player 5": 0.0, "player 4": pytest.approx(float("nan"), nan_ok=True)}
    m = backtest.metrics(bets)
    assert (m["won"], m["lost"], m["push"], m["void"], m["open"]) == (1, 1, 1, 1, 1) and m["roi"] == pytest.approx(0.5 / 3)
    assert set(backtest.run(found, Strategy(sides=("under",)), ("fanduel",)).player_key) == {"player 1"}
    assert set(backtest.run(found, Strategy(sports=("nfl",)), ("fanduel",)).player_key) == {"player 5"}
    assert set(backtest.run(found, Strategy(min_price=100), ("fanduel",)).player_key) == {"player 2"}


def test_a_delayed_entry_takes_the_later_price_or_no_bet():
    found = frame([flagged(1, ev=0.12, price=110, delay_price=100, status="won"),      # still there, a little worse
                   flagged(2, ev=0.12, price=110, delay_price=None),                    # the line was gone
                   flagged(3, ev=0.12, price=110, delay_price=-130)])                   # still there, but no longer an edge
    bets = backtest.run(found, Strategy(min_ev=0.04, entry_delay=1), ("fanduel",))
    assert list(bets.player_key) == ["player 1"]
    bet = bets.iloc[0]
    assert bet.price == 100 and bet.flagged_price == 110 and bet.profit == pytest.approx(1.0)
    assert bet.price_observed_at == bet.observed_at + timedelta(minutes=10) and bet.closing_line_value == 0.01
    assert bet.ev == pytest.approx(bet.probability * 2.0 - 1)       # the estimate, at the price actually taken


def test_drawdown_and_losing_streak():
    assert backtest.drawdown(pd.Series([1.0, -1, -1, 2, -1])) == 2.0 and backtest.drawdown(pd.Series([-1.0, -1])) == 2.0
    assert backtest.losing_streak(["won", "lost", "push", "lost", "lost", "won", "lost"]) == 3
    found = frame([flagged(n, day=n // 2, status="lost" if n % 3 else "won") for n in range(12)])
    m = backtest.metrics(backtest.run(found, Strategy(), ("fanduel",)))
    assert m["days"] == 6 and m["games"] == 6 and m["longest_losing_streak"] == 2 and m["roi_interval"][0] < m["roi"] < m["roi_interval"][1]
    assert [name for name, _ in backtest.by_period(backtest.run(found, Strategy(), ("fanduel",)), days=3)] == ["Nov 01 to Nov 03", "Nov 04 to Nov 06"]


def test_walk_forward_chooses_the_edge_on_earlier_days_only():
    rows = []
    for day in range(8):                                             # early days: only the big edges win
        rows += [flagged(day * 10 + n, day=day, ev=0.06, status="won") for n in range(3)]
        rows += [flagged(day * 10 + 5 + n, day=day, ev=0.02, status="lost") for n in range(3)]
    for day in range(8, 12):                                         # later days: the reverse
        rows += [flagged(day * 10 + n, day=day, ev=0.06, status="lost") for n in range(3)]
        rows += [flagged(day * 10 + 5 + n, day=day, ev=0.02, status="won") for n in range(3)]
    found = frame(rows)
    out = backtest.walk_forward(found, Strategy(min_ev=0.0), [0.0, 0.05], default_books=("fanduel",), min_train_days=8,
                                test_days=2, min_train_bets=10)
    first, second = out["folds"]
    assert first["chosen"] == 0.05 and first["tuned"] and first["test_won"] == 0 and first["test_lost"] == 6   # judged on days it never saw
    assert second["chosen"] == 0.05 and (second["train_to"] - first["train_to"]).days == 2
    assert len(out["bets"]) == 12 and backtest.metrics(out["bets"])["roi"] == -1.0
    thin = backtest.walk_forward(found, Strategy(min_ev=0.03), [0.0, 0.05], default_books=("fanduel",), min_train_days=8,
                                 test_days=2, min_train_bets=500)
    assert thin["folds"][0]["chosen"] == 0.03 and not thin["folds"][0]["tuned"]       # too few bets to choose: the rule's own bar
    with pytest.raises(ValueError):
        backtest.walk_forward(found, Strategy(), [0.0], objective="luck")


# --- from stored snapshots --------------------------------------------------------------------------
def test_candidates_come_from_stored_snapshots_and_are_kept_per_game(settings, engine, tmp_path, monkeypatch):  # noqa: F811
    scan(settings, engine)
    cache = tmp_path / "cache.pkl"
    found, offers = backtest.candidates(engine, settings, cache=cache, load_results=fake_results)
    assert len(found) and set(found.status) <= {"won", "lost", "push", "void", "open"} and cache.exists()
    assert "betmgm" in set(found.book)                                # every book is flagged; a rule narrows it
    nba = found[(found.event_id == NBA) & (found.player == "Sample Player A") & (found.side == "over")]
    assert set(nba.status) == {"won"} and nba.closing_line_value.notna().any()       # 30 points against 24.5, with a later look
    assert set(found[found.event_id != NBA].status) == {"open"}        # that game is not final
    assert len(offers) and {"book_probability", "consensus_probability", "over_decimal", "under_decimal"} <= set(offers)
    assert offers.groupby(["event_id", "market", "player_key", "line_key"]).size().max() == 1
    bets = backtest.run(found, backtest.presets(settings)[1], tuple(settings.bet_books))
    assert len(bets) and set(bets.book) <= set(settings.bet_books) and not bets.duplicated(backtest.SELECTION).any()

    calls = []
    real = backtest.build_event
    monkeypatch.setattr(backtest, "build_event", lambda rows, s: calls.append(rows.event_id.iloc[0]) or real(rows, s))
    again, _ = backtest.candidates(engine, settings, cache=cache, load_results=fake_results)
    assert calls == [] and len(again) == len(found)                    # nothing new: nothing worked out again
    changed = settings.model_copy(update={"uncertainty_floor": 0.02})
    backtest.candidates(engine, changed, cache=cache, load_results=fake_results)
    assert len(calls) == found.event_id.nunique()                      # a pricing setting changed: everything is redone

    def down(dsn, games):
        raise RuntimeError("no database")
    assert set(backtest.candidates(engine, settings, load_results=down)[0].status) == {"open"}


def test_sample_data_is_never_backtested(settings, engine):  # noqa: F811
    from sports_sniper import scanner
    scanner.run_scan(settings, engine, "sample", now=NOW)
    assert backtest.load_prices(engine).empty and backtest.candidates(engine, settings, load_results=fake_results)[0].empty


# --- bankroll ----------------------------------------------------------------------------------------
def test_flat_staking_respects_the_limits():
    found = frame([flagged(n, day=0, status="won") for n in range(6)] + [flagged(10 + n, day=1, status="lost") for n in range(2)])
    bets = backtest.run(found, Strategy(), ("fanduel",))
    out = bankroll.simulate(bets, bankroll.Rules(start=50, flat_stake=2, max_per_game=5, max_daily=20, max_per_player=None))
    assert out["placed"] == 5 and out["skipped"] == {"game limit": 3}            # 2 + 2 + 1 fills the first game, 2 + 2 the second
    assert out["staked"] == pytest.approx(9.0) and out["final"] == pytest.approx(50 + 5 * (100 / 110) - 4)
    assert out["max_drawdown"] == pytest.approx(4.0) and list(out["curve"].staked) == [5.0, 4.0]
    daily = bankroll.simulate(bets, bankroll.Rules(flat_stake=2, max_per_game=None, max_daily=3, max_per_player=None))
    assert daily["skipped"] == {"daily limit": 4} and daily["placed"] == 4
    assert bankroll.simulate(bets.iloc[0:0])["final"] == 100.0 and bankroll.simulate(frame([flagged(1, status="open")]))["placed"] == 0


def test_kelly_stakes_shrink_with_the_bankroll_and_never_chase_a_loss():
    assert bankroll.kelly_fraction(0.55, 2.0) == pytest.approx(0.10) and bankroll.kelly_fraction(0.45, 2.0) == 0.0
    rows = [flagged(day, day=day, ev=0.10, price=100, status="lost", standard_error=0.0) for day in range(5)]
    rules = bankroll.Rules(staking="kelly", kelly_fraction=1.0, max_stake=None, max_daily=None, max_per_game=None, max_per_player=None)
    out = bankroll.simulate(backtest.run(frame(rows), Strategy(), ("fanduel",)), rules)
    stakes = list(out["bets"].stake)
    assert stakes[0] == pytest.approx(10.0) and all(b < a for a, b in zip(stakes, stakes[1:]))   # a loss never raises the next stake
    thin = bankroll.simulate(backtest.run(frame([flagged(1, ev=0.02, price=100, standard_error=0.02, status="won")]), Strategy(min_ev=0.0), ("fanduel",)),
                             bankroll.Rules(staking="kelly"))
    assert thin["placed"] == 0 and "stake too small (no edge after the haircut)" in thin["skipped"]
    stopped = bankroll.simulate(backtest.run(frame(rows), Strategy(), ("fanduel",)), bankroll.Rules(flat_stake=5, stop_drawdown=0.08, max_per_game=None, max_per_player=None))
    assert stopped["placed"] == 2 and stopped["stopped"] is not None and stopped["skipped"] == {"stopped after drawdown": 3}
    with pytest.raises(ValueError):
        bankroll.simulate(backtest.run(frame(rows), Strategy(), ("fanduel",)), bankroll.Rules(staking="martingale"))


# --- model against market -----------------------------------------------------------------------------
def offers_frame(rows) -> pd.DataFrame:
    out = pd.DataFrame([dict(sport="nba", event_id=f"g{n}", event_name="Away @ Home", commence_time=START, market="player_points",
                             player=f"Player {n}", player_key=f"player {n}", line=20.5, line_key="20.5", book="fanduel",
                             observed_at=START - timedelta(minutes=30), over_decimal=2.0, under_decimal=1.8,
                             book_probability=0.48, consensus_probability=c, model_probability=m, status=s)
                        for n, (c, m, s) in enumerate(rows)])
    return out


def test_sources_are_scored_on_the_same_settled_lines():
    offers = offers_frame([(0.6, 0.9, "won"), (0.4, 0.1, "lost"), (0.5, None, "won"), (0.5, 0.9, "open"), (0.5, 0.2, "push")])
    row = model_compare.scores(offers).iloc[0]
    assert row.lines == 2 and row.model_brier == pytest.approx(0.01) and row.consensus_brier == pytest.approx(0.16)
    assert row.book_brier > row.consensus_brier > row.model_brier and row.model_log_loss < row.consensus_log_loss
    assert model_compare.scores(offers.iloc[0:0].assign(model_probability=[])).empty
    picks = model_compare.picks(offers, "model", 0.04)
    assert list(picks.side) == ["over", "under", "under"] and list(picks.status) == ["won", "won", "push"]
    table = model_compare.picks_table(offers, thresholds=(0.04,), sources=("consensus", "model")).set_index("source")
    assert table.loc["model", "bets"] == 3 and table.loc["model", "roi"] == pytest.approx((1.0 + 0.8) / 3)
    assert table.loc["consensus", "won"] == 2                       # 0.6 over at 2.0, 0.6 under at 1.8; the 0.5s have no edge
    assert [b["n"] for b in model_compare.calibration_table(offers, "model", 2)] == [1, 1]


class Cursor:
    def __init__(self, rows):
        self.rows, self.result = rows, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.result = [("nbaprops.projections",)] if "to_regclass" in sql else self.rows

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


class Connection:
    def __init__(self, rows):
        self.rows = rows

    def set_session(self, readonly):
        assert readonly

    def cursor(self):
        return Cursor(self.rows)

    def close(self):
        pass


def test_the_nba_model_uses_only_forecasts_made_before_the_price_was_seen():
    pytest.importorskip("nba_props.model")
    seen = START - timedelta(minutes=30)
    day = START.astimezone(models.EASTERN).date()
    rows = [(day, "player 0", "pts", seen - timedelta(hours=5), 18.0, 1.5, 0.0),
            (day, "player 0", "pts", seen - timedelta(hours=1), 26.0, 1.5, 0.0),        # the latest one before the price
            (day, "player 0", "pts", seen + timedelta(minutes=5), 40.0, 1.5, 0.0),       # made after: never used
            (day, "player 1", "pts", seen + timedelta(minutes=5), 40.0, 1.5, 0.0)]       # only a later one: no forecast
    offers = offers_frame([(0.5, None, "won"), (0.5, None, "won"), (0.5, None, "won")]).drop(columns="model_probability")
    offers.loc[2, ["sport", "market"]] = ["nfl", "player_rush_yds"]
    got = models.nba_over_probabilities("dsn", offers, connect=lambda dsn: Connection(rows))
    assert 0.75 < got[0] < 0.95 and got[1] is None and got[2] is None       # mean 26 against a line of 20.5
    attached, fits = model_compare.attach_models(offers, Settings(_env_file=None), adapters=((models.NBA_MODEL, lambda dsn, o: got),))
    assert attached.model_version.iloc[0] == models.NBA_MODEL and attached.model_version.iloc[1:].isna().all()
    assert fits == {} and attached.calibrated_probability.isna().all()        # three lines are far too few to calibrate on
    assert models.nba_over_probabilities(None, offers.iloc[0:0]) == []


# --- report, command line and dashboard ---------------------------------------------------------------
def test_report_covers_rules_walk_forward_bankroll_and_models(settings, engine):  # noqa: F811
    scan(settings, engine)
    report = research.build(engine, settings, load_results=fake_results, history={},
                            adapters=(("test-model", lambda dsn, offers: [0.7] * len(offers)),))
    text = research.markdown(report)
    for heading in ("## Rules compared", "## By minimum estimated edge", "## Walk-forward", "## Paper bankroll", "## Model against market",
                    "## Breakout signals"):
        assert heading in text
    assert "Automatic v2 (4%+, bet books)" in text and "not a forecast" in text and len(report["strategies"]) == 9
    assert research.save_calibration(report, settings) is None           # nothing fitted on a handful of lines: nothing saved
    empty = db.make_engine("sqlite:///:memory:")
    db.init_db(empty)
    assert "No stored real prices yet" in research.markdown(research.build(empty, settings, load_results=fake_results, history={}, adapters=()))


def test_backtest_and_bankroll_commands(env, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(models, "ADAPTERS", ())
    monkeypatch.setattr(breakout, "load_history", lambda dsn, connect=None: {})
    seeded().dispose()
    out = CliRunner().invoke(cli.app, ["backtest", "--no-write", "--min-ev", "0.01", "--side", "over"])
    assert out.exit_code == 0, out.output
    assert "Automatic v2 (4%+, bet books)" in out.output and "Custom rule" in out.output and "Walk-forward" in out.output
    assert "Paper bankroll, Flat $1" in out.output and "Report:" not in out.output
    money = CliRunner().invoke(cli.app, ["bankroll", "--staking", "kelly"])
    assert money.exit_code == 0 and "Hypothetical $100.00" in money.output
    assert CliRunner().invoke(cli.app, ["bankroll", "--staking", "double"]).exit_code != 0


def test_dashboard_backtest_and_model_tabs(env, monkeypatch):  # noqa: F811
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(models, "ADAPTERS", ())
    monkeypatch.setattr(breakout, "load_history", lambda dsn, connect=None: {})
    seeded().dispose()
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(cli.Path(cli.__file__).resolve().parent / "dashboard" / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.button[[b.label for b in app.button].index("Load the backtest")].click().run()
    assert not app.exception
    labels = {m.label for m in app.metric}
    assert {"Settled bets", "Simulated ROI", "Ending bankroll", "Largest fall"} <= labels
    rules = next(d.value for d in app.dataframe if "Rule" in d.value and "Your rule" in set(d.value["Rule"]))
    assert "Automatic v2 (4%+, bet books)" in set(rules["Rule"])
    app.button[[b.label for b in app.button].index("Load the comparison")].click().run()
    assert not app.exception
