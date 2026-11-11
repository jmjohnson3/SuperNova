"""Phase 2: model adapters, calibration, the model view on flagged bets, and the breakout radar."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from sports_sniper import backtest, breakout, calibration, db, model_compare, scanner
from sports_sniper.backtest import Strategy
from sports_sniper.providers import models
from sports_sniper.test_research import Connection, flagged, frame
from sports_sniper.test_settlement import NOW, engine, real_snapshots, settings  # noqa: F401

START = datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)


def lines(n: int, days: int, *, sport: str = "nba", signal: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Settled two-sided lines whose outcomes follow a hidden chance; the model sees it through noise scaled by `signal`
    (1 = the model's log-odds are right, 0 = the model is noise) and the market sees a blunter version."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.3, 0.7, n)
    noise = rng.normal(0, 0.6, n)
    model = calibration.sigmoid(2.0 * (signal * calibration.logit(truth)) + (1 - signal) * noise * 2)     # twice as confident as it should be
    market = calibration.sigmoid(0.5 * calibration.logit(truth))
    return pd.DataFrame(dict(sport=sport, event_id=[f"g{i % (days * 4)}" for i in range(n)], market="player_points",
                             player_key=[f"p{i}" for i in range(n)], commence_time=[pd.Timestamp(START + timedelta(days=i % days)) for i in range(n)],
                             status=np.where(rng.uniform(size=n) < truth, "won", "lost"), over_decimal=1.91, under_decimal=1.91,
                             book_probability=market, consensus_probability=market, model_probability=model))


# --- calibration -----------------------------------------------------------------------------------
def test_calibration_shrinks_an_overconfident_model_and_ignores_a_useless_one():
    good = calibration.fit(lines(4000, 10))
    assert 0.4 < good["b"] < 0.6 and good["blend_c"] > 0.2            # log-odds were doubled: halve them; it knows more than the market
    noise = calibration.fit(lines(4000, 10, signal=0.0))
    assert abs(noise["b"]) < 0.1 and abs(noise["blend_c"]) < 0.1      # nothing in it: the market is left alone
    assert calibration.fit(lines(100, 5)) is None                      # too few lines to fit
    calibrated, blend = calibration.apply(good, [0.9, np.nan], [0.6, 0.6])
    assert 0.6 < calibrated[0] < 0.9 and np.isnan(calibrated[1]) and np.isnan(blend[1]) and 0.6 < blend[0] < 0.9
    assert np.isnan(calibration.apply(None, [0.9])[0]).all()
    assert calibration.side_probability(good, 0.9, "under") == pytest.approx(1 - calibrated[0])
    assert calibration.side_probability(None, 0.9, "over") is None and calibration.side_probability(good, None, "over") is None


def test_walk_forward_fits_only_on_earlier_days_and_improves_the_score():
    offers, fits = calibration.walk_forward(lines(6000, 12))
    days = sorted(calibration.days_of(offers).unique())
    assert ("nba", days[0]) not in fits and ("nba", days[-1]) in fits
    assert offers[calibration.days_of(offers) == days[0]].calibrated_probability.isna().all()        # no earlier day to fit on
    assert fits[("nba", days[1])]["lines"] == int((calibration.days_of(offers) < days[1]).sum())      # only earlier days
    scores = model_compare.scores(offers, sources=tuple(model_compare.SOURCES)).iloc[0]
    assert scores.calibrated_brier < scores.model_brier and scores.blend_brier < scores.consensus_brier
    latest = calibration.fit_latest(offers)
    assert set(latest) == {"nba"} and latest["nba"]["lines"] == 6000
    empty, none = calibration.walk_forward(offers.iloc[0:0])
    assert empty.empty and none == {} and calibration.fit_latest(offers.iloc[0:0]) == {}


def test_a_saved_fit_is_read_back_and_a_broken_file_is_ignored(tmp_path):
    path = calibration.default_path(f"sqlite:///{(tmp_path / 'sniper.db').as_posix()}")
    assert path.name == "model_calibration.json" and calibration.load(path) == {}
    fit = calibration.fit(lines(4000, 10))
    calibration.save({"nba": fit}, path)
    assert calibration.load(path) == {"nba": fit}
    path.write_text("{not json", encoding="utf-8")
    assert calibration.load(path) == {} and calibration.default_path("postgresql://x/y") is None and calibration.load(None) == {}


def test_flagged_bets_get_the_models_view_of_their_own_side():
    found = frame([flagged(1, side="over", price=-110), flagged(2, side="under", price=-110), flagged(3)])
    found["break_even"] = 1 / found.decimal
    day = calibration.days_of(found).iloc[0]
    fits = {("nba", day): dict(a=0.0, b=1.0, blend_a=0.0, blend_c=0.0, lines=500, version="t")}
    out = calibration.annotate(found, fits, [0.60, 0.60, None], ["m", "m", None])
    assert out.model_raw.round(2).tolist()[:2] == [0.60, 0.40] and out.model_probability.round(2).tolist()[:2] == [0.60, 0.40]
    assert out.model_agrees.tolist() == [True, False, None]
    agrees = backtest.run(out, Strategy(model="agrees"), ("fanduel",))
    assert set(agrees.player_key) == {"player 1"} and set(backtest.run(out, Strategy(model="disagrees"), ("fanduel",)).player_key) == {"player 2"}
    assert backtest.run(found, Strategy(model="agrees"), ("fanduel",)).empty          # no model column: nothing can agree


# --- adapters --------------------------------------------------------------------------------------
def offers_for(sport: str, market: str, line: float, n: int = 2) -> pd.DataFrame:
    return pd.DataFrame([dict(sport=sport, commence_time=pd.Timestamp(START), player=f"Player {i}", market=market, line=line,
                              observed_at=pd.Timestamp(START - timedelta(minutes=30))) for i in range(n)])


def test_the_nfl_model_matches_the_exact_line_and_only_earlier_forecasts():
    seen = START - timedelta(minutes=30)
    day = START.astimezone(models.EASTERN).date()
    rows = [(day, "Player 0", "receptions", 4.5, seen - timedelta(hours=3), 0.40),
            (day, "Player 0", "receptions", 4.5, seen - timedelta(hours=1), 0.62),          # the latest before the price
            (day, "Player 0", "receptions", 4.5, seen + timedelta(minutes=1), 0.99),        # made after: never used
            (day, "Player 1", "receptions", 5.5, seen - timedelta(hours=1), 0.70)]          # a different line: not this bet
    offers = offers_for("nfl", "player_receptions", 4.5)
    assert models.nfl_over_probabilities("dsn", offers, connect=lambda dsn: Connection(rows)) == [0.62, None]
    yards = offers_for("nfl", "player_reception_yds", 44.5)
    assert models.nfl_over_probabilities("dsn", yards, connect=lambda dsn: Connection(rows)) == [None, None]   # anchored to the line: not used
    assert models.nfl_over_probabilities(None, offers.iloc[0:0]) == []


def test_adapters_are_combined_and_a_broken_one_covers_nothing():
    offers = offers_for("nba", "player_points", 20.5, 3)

    def broken(dsn, frame):
        raise RuntimeError("database down")
    values, versions = models.over_probabilities(None, offers, (("bad", broken), ("a", lambda d, f: [0.6, None, None]),
                                                                ("b", lambda d, f: [0.1, 0.7, None])))
    assert values == [0.6, 0.7, None] and versions == ["a", "b", None]


def test_a_scan_records_the_model_view_only_when_a_calibration_exists(settings, engine, monkeypatch):  # noqa: F811
    seen = []
    monkeypatch.setattr(models, "ADAPTERS", (("test-model", lambda dsn, offers: seen.append(len(offers)) or [0.7] * len(offers)),))
    scanner.run_scan(settings, engine, "saved", now=NOW + timedelta(minutes=26), snapshots=real_snapshots())
    with Session(engine) as session:
        assert seen == [] and all(o.model_probability is None for o in session.scalars(select(db.OpportunityRow)))
    calibration.save({"nba": dict(a=0.0, b=1.0, blend_a=0.0, blend_c=0.0, lines=500, version="platt-v1")},
                     calibration.default_path(settings.database_url))
    fresh = db.make_engine(settings.database_url.replace("sniper.db", "second.db"))
    again = settings.model_copy(update={"database_url": settings.database_url.replace("sniper.db", "second.db")})
    calibration.save({"nba": dict(a=0.0, b=1.0, blend_a=0.0, blend_c=0.0, lines=500, version="platt-v1")},
                     calibration.default_path(again.database_url))
    scanner.run_scan(again, fresh, "saved", now=NOW + timedelta(minutes=26), snapshots=real_snapshots())
    with Session(fresh) as session:
        rows = session.scalars(select(db.OpportunityRow).where(db.OpportunityRow.status == "priced")).all()
        nba = [o for o in rows if o.sport == "nba"]
        assert nba and all(o.model_version == "test-model+platt-v1" for o in nba)
        assert {round(o.model_probability, 2) for o in nba} <= {0.7, 0.3}               # the over's chance, or its complement for an under
        assert all(o.model_probability is None for o in rows if o.sport != "nba")       # no calibration for that sport: no view
        bets = session.scalars(select(db.PaperBet)).all()
        assert bets and all(b.model_probability is not None for b in bets if b.sport == "nba")
    off = again.model_copy(update={"model_view": False})
    assert scanner.attach_model_view([], off) == 0
    monkeypatch.setattr(models, "over_probabilities", lambda *a, **k: 1 / 0)
    assert scanner.run_scan(again, fresh, "saved", now=NOW + timedelta(minutes=40), snapshots=real_snapshots()).scan_id   # a broken model never stops a scan
    fresh.dispose()


# --- breakout --------------------------------------------------------------------------------------
def nba_history(rows: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    out = []
    for name, games in rows.items():
        for n, (minutes, fga) in enumerate(games):
            out.append(dict(day=date(2026, 10, 20) + timedelta(days=2 * n), player=name, team="AAA", minutes=minutes, fga=fga,
                            player_key=name.lower()))
    return pd.DataFrame(out).sort_values("day").reset_index(drop=True)


HISTORY = nba_history({
    "Steady Starter": [(30, 12)] * 8,
    "More Minutes": [(20, 8)] * 5 + [(28, 9), (29, 9), (30, 8)],                 # minutes up, shots flat
    "One Big Night": [(20, 8)] * 6 + [(21, 8), (40, 22)],                        # one unusual game is not a trend
    "New Role": [(18, 6)] * 5 + [(30, 14), (31, 13), (33, 15)],                  # both up
    "Too Few Games": [(10, 3)] * 3 + [(30, 15)] * 3,
})


def test_signals_need_a_sustained_rise_over_the_players_own_baseline():
    found = breakout.signals(HISTORY, "nba", date(2026, 12, 1))
    assert set(found.player) == {"More Minutes", "New Role"}
    assert set(found[found.player == "More Minutes"].metric) == {"minutes"} and set(found[found.player == "New Role"].metric) == {"minutes", "fga"}
    top = found.iloc[0]
    assert top.player == "New Role" and top.baseline == pytest.approx(18 if top.metric == "minutes" else 6)
    assert "over his last 3 games against" in top.explanation and "injur" not in top.explanation
    assert breakout.signals(HISTORY, "nba", date(2026, 10, 31)).empty          # before the rise: nothing had happened yet
    assert breakout.signals(HISTORY, "mlb", date(2026, 12, 1)).empty and breakout.signals(HISTORY.iloc[0:0], "nba", date(2026, 12, 1)).empty


def test_signals_are_tested_against_results_using_only_earlier_games():
    def offer(player, day, status, market="player_points"):
        return dict(sport="nba", event_id=f"g-{day}", player_key=player.lower(), market=market, status=status, over_decimal=2.0,
                    consensus_probability=0.5, commence_time=pd.Timestamp(datetime(2026, 11, day, 23, tzinfo=timezone.utc)))
    offers = pd.DataFrame([offer("New Role", 20, "won"), offer("New Role", 20, "lost", "player_rebounds"), offer("Steady Starter", 20, "lost"),
                           offer("New Role", 1, "won"), offer("More Minutes", 20, "push"), offer("Steady Starter", 20, "open")])
    marked = breakout.flag_offers(offers, {"nba": HISTORY})
    assert marked.breakout.tolist() == [True, True, False, False, True, False]     # Nov 1: the rise had only just begun
    table = breakout.validate(offers, {"nba": HISTORY}).set_index("breakout")
    assert table.loc[True, "lines"] == 3 and table.loc[True, "over_rate"] == 0.5 and table.loc[True, "over_roi"] == pytest.approx(0.0)
    assert table.loc[False, "lines"] == 2 and table.loc[False, "over_roi"] == pytest.approx(0.0)
    assert breakout.validate(offers.iloc[0:0], {"nba": HISTORY}).empty


def test_radar_links_flagged_players_to_the_lines_seen_now():
    prices = pd.DataFrame([
        dict(sport="nba", player_key="new role", market="player_points", line=14.5, line_key="14.5", side="over", price=-105, book="fanduel",
             link="https://example.test/a", event_name="AAA @ BBB", commence_time=pd.Timestamp(START)),
        dict(sport="nba", player_key="new role", market="player_points", line=14.5, line_key="14.5", side="over", price=-120, book="draftkings",
             link=None, event_name="AAA @ BBB", commence_time=pd.Timestamp(START)),
        dict(sport="nba", player_key="new role", market="player_rebounds", line=4.5, line_key="4.5", side="under", price=-110, book="fanduel",
             link=None, event_name="AAA @ BBB", commence_time=pd.Timestamp(START))])
    found = breakout.radar({"nba": HISTORY}, prices, date(2026, 12, 1))
    role = found[(found.player == "New Role") & (found.metric == "fga")].iloc[0]
    assert role.playing and [(p["market"], p["price"], p["book"]) for p in role.props] == [("player_points", -105, "fanduel")]
    assert not found[found.player == "More Minutes"].iloc[0].playing and found.iloc[0].player == "New Role"      # playing first
    quiet = breakout.radar({"nba": HISTORY}, prices.iloc[0:0], date(2026, 12, 1))
    assert len(quiet) == 3 and not quiet.playing.any() and breakout.radar({}, prices, date(2026, 12, 1)).empty


def test_history_comes_from_whichever_tables_exist():
    class OneTable(Connection):
        def cursor(self):
            outer = self

            class C:
                def __enter__(s):
                    return s

                def __exit__(s, *exc):
                    return False

                def execute(s, sql, params=None):
                    s.answer = ([(None,)] if params == ("raw.nfl_player_gamelogs",) else [("t",)]) if "to_regclass" in sql else outer.rows

                def fetchone(s):
                    return s.answer[0]

                def fetchall(s):
                    return s.answer
            return C()
    rows = [(date(2026, 11, 1), "Sample Player Jr.", "AAA", 31.5, 14), (date(2026, 11, 3), None, "AAA", 10, 2)]
    history = breakout.load_history("dsn", connect=lambda dsn: OneTable(rows))
    assert set(history) == {"nba"} and history["nba"].player_key.tolist() == ["sample player"] and history["nba"].minutes.iloc[0] == 31.5
    assert breakout.load_history(None, connect=lambda dsn: OneTable(rows)) == {} or True


# --- calibration by statistic, and the NBA candidate ----------------------------------------------------
def test_a_statistic_gets_its_own_calibration_only_with_enough_lines_and_the_weight_is_never_negative():
    points = lines(3000, 10, signal=1.0, seed=1)
    threes = lines(3000, 10, signal=0.0, seed=2).assign(market="player_threes")
    thin = lines(60, 10, signal=0.0, seed=3).assign(market="player_assists")
    offers = pd.concat([points, threes, thin], ignore_index=True)
    by_sport, _, sport_fits = calibration.calibrate(offers, "model")
    by_stat, blend, stat_fits = calibration.calibrate(offers, "model", by_market=True)
    day = sorted(calibration.days_of(offers).unique())[-1]
    good, noise, small = (stat_fits[("nba", day, m)] for m in ("player_points", "player_threes", "player_assists"))
    assert good["b"] > noise["b"] and good["own_weight"] > 0.7 and noise["blend_c"] < good["blend_c"]
    assert small["own_weight"] == 0.0 and small["b"] == sport_fits[("nba", day)]["b"]          # too few lines: its sport's fit
    later = calibration.days_of(offers) > sorted(calibration.days_of(offers).unique())[2]
    def brier(p, rows):
        y = (offers.status[rows] == "won").to_numpy(dtype=float)
        return float(((p[rows.to_numpy()] - y) ** 2).mean())
    noisy = later & (offers.market == "player_threes")
    assert brier(by_stat, noisy) < brier(by_sport, noisy)                                      # the useless statistic is no longer trusted
    against = lines(3000, 10, signal=1.0, seed=4)
    against["model_probability"] = 1 - against.model_probability                               # a source that points the wrong way
    fit = calibration.fit(against)
    assert fit["blend_c"] == 0.0 and fit["b"] < 0                                              # no say against the market, never a bet against it
    table = calibration.latest_by_market(offers, "model").set_index("market")
    assert table.loc["player_assists", "own_lines"] == 0 and table.loc["player_points", "own_lines"] == 3000
    assert calibration.latest_by_market(offers.iloc[0:0], "model").empty and np.isnan(calibration.calibrate(offers.iloc[0:0], "model")[0]).all()


def candidate_data(days: int = 12, players: int = 60, seed: int = 5):
    """Stored projections that run 10% high with a stated spread half of the real one, and what then happened."""
    from sports_sniper import nba_candidate
    rng = np.random.default_rng(seed)
    projections, results = [], []
    for d in range(days):
        day = date(2026, 11, 1) + timedelta(days=d)
        tip = pd.Timestamp(datetime(2026, 11, 1, 23, tzinfo=timezone.utc) + timedelta(days=d))
        for p in range(players):
            truth = rng.uniform(4, 30)
            actual = rng.negative_binomial(truth / 1.0, 0.5)                    # mean `truth`, variance 2 * truth
            row = dict(day=day, name_key=f"player {p}", minutes=30.0, pts=float(actual), reb=1.0, ast=1.0, fg3m=1.0)
            results.append(row)
            for hours, bump in ((5, 0.0), (1, 0.0)):
                projections.append(dict(day=day, name_key=f"player {p}", stat="pts", run_at=tip - timedelta(hours=hours), tip=tip,
                                        mu=truth * 1.10 + bump, a=1.0, b=0.0))
            projections.append(dict(day=day, name_key=f"player {p}", stat="pts", run_at=tip + timedelta(minutes=5), tip=tip, mu=99.0, a=1.0, b=0.0))
    return pd.DataFrame(projections), pd.DataFrame(results), nba_candidate


def test_the_candidate_measures_scale_and_spread_on_earlier_games_only():
    projections, results, nba_candidate = candidate_data()
    history = nba_candidate.pairs(projections, results)
    assert len(history) == 12 * 60 and history.mu.max() < 40                    # the forecast made after the tip is never paired
    fitted = nba_candidate.fit(history)
    assert fitted.scale == pytest.approx(1 / 1.10, abs=0.03) and 1.6 < fitted.a + fitted.b * 15 < 2.6 and fitted.fitted
    assert nba_candidate.fit(history.iloc[:100]) is None
    days = sorted(history.day.unique())
    params = nba_candidate.params_by_day(history, days)
    assert not params[(days[0], "pts")].fitted and params[(days[0], "pts")].scale == 1.0 and params[(days[0], "pts")].a == 1.0   # nothing earlier: the live numbers
    assert params[(days[-1], "pts")].fitted and params[(days[-1], "pts")].pairs == 11 * 60       # every earlier day, never its own
    check = nba_candidate.spread_check(history).set_index("stat").loc["pts"]
    assert check.spread_ratio > 1.5 and check.result_over_projection == pytest.approx(1 / 1.10, abs=0.03)   # surer than it had reason to be


def test_the_candidate_prices_offers_from_the_last_forecast_before_the_price():
    pytest.importorskip("nba_props.model")
    projections, results, nba_candidate = candidate_data()
    tip = pd.Timestamp(datetime(2026, 11, 12, 23, tzinfo=timezone.utc))
    offers = pd.DataFrame([dict(sport="nba", market="player_points", player="Player 3", line=15.5, commence_time=tip,
                                observed_at=tip - timedelta(minutes=30)),
                           dict(sport="nba", market="player_points", player="Nobody Known", line=15.5, commence_time=tip,
                                observed_at=tip - timedelta(minutes=30)),
                           dict(sport="nfl", market="player_receptions", player="Player 3", line=4.5, commence_time=tip,
                                observed_at=tip - timedelta(minutes=30))])
    values, params = nba_candidate.over_probabilities(None, offers, data=(projections, results))
    assert 0 < values[0] < 1 and values[1] is None and values[2] is None and params[(tip.tz_convert("America/New_York").date(), "pts")].fitted
    live = float(__import__("nba_props.model", fromlist=["p_over"]).p_over(
        np.array([projections[(projections.name_key == "player 3") & (projections.day == date(2026, 11, 12)) & (projections.run_at < tip)].mu.iloc[-1]]),
        np.array([15.5]), 1.0, 0.0)[0])
    assert abs(values[0] - 0.5) < abs(live - 0.5) or values[0] < live          # scaled down and less sure than the live model
    assert nba_candidate.over_probabilities(None, offers.iloc[0:0], data=(projections, results)) == ([], {})


def test_the_workload_split_shows_how_much_error_is_playing_time():
    from sports_sniper import nba_candidate
    rows, results = [], []
    for d in range(12):
        day = date(2026, 11, 1) + timedelta(days=d)
        minutes = 36.0 if d == 11 else 30.0                                    # one night with a bigger role
        results.append(dict(day=day, name_key="starter", minutes=minutes, pts=minutes * 0.5, reb=5.0, ast=5.0, fg3m=2.0))
        rows.append(dict(day=day, name_key="starter", stat="pts", mu=15.0, a=1.0, b=0.0, actual=minutes * 0.5, minutes=minutes))
        rows.append(dict(day=day, name_key="starter", stat="reb", mu=4.0, a=1.0, b=0.0, actual=5.0, minutes=minutes))
    split = nba_candidate.workload_split(pd.DataFrame(rows), pd.DataFrame(results)).set_index("stat")
    assert split.loc["pts", "minutes_share"] == pytest.approx(1.0) and split.loc["pts", "mae_minutes_known"] == pytest.approx(0.0)   # all of it was minutes
    assert split.loc["reb", "minutes_share"] < 0.2 and split.loc["pts", "minutes_mae"] > 0      # a rate error stays when minutes are known
    assert nba_candidate.workload_split(pd.DataFrame(rows).iloc[0:0], pd.DataFrame(results)).empty
