from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nba_props import selector

NOW = datetime(2026, 11, 10, 0, 30, tzinfo=timezone.utc)


class Flat:
    """A booster that adds a fixed amount to the book's own log-odds."""
    def __init__(self, shift=0.0):
        self.shift = shift

    def predict(self, X, raw_score=True):
        return np.full(len(X), self.shift)


def frame(**over):
    row = dict(stat="pts", line=20.5, over_price=-110, under_price=-110, pred=21.0, p_model=0.52)
    return pd.DataFrame([dict(row, **over)])


def test_decimal_prices():
    assert selector.decimal(150) == pytest.approx(2.5)
    assert selector.decimal(-200) == pytest.approx(1.5)


def test_design_starts_from_the_books_own_price():
    X = selector.design(frame())
    assert X.p_market[0] == pytest.approx(0.5)
    assert X.logit_mkt[0] == pytest.approx(0.0)
    assert X.hold[0] == pytest.approx(2 * 110 / 210 - 1)
    assert (X.is_pts[0], X.is_reb[0], X.is_ast[0]) == (1, 0, 0)
    assert X.p_gap[0] == pytest.approx(0.02)
    assert X.pred_minus_line[0] == pytest.approx(0.5)
    assert set(selector.FEATURES) <= set(X.columns)


def test_a_model_that_learns_nothing_returns_the_books_price():
    X = selector.design(frame(over_price=-150, under_price=120))
    assert selector.probability(Flat(), X)[0] == pytest.approx(X.p_market[0])
    assert selector.probability(Flat(0.3), X)[0] > X.p_market[0]


def test_best_side_is_the_higher_ev_side():
    X = selector.design(frame(over_price=-110, under_price=-110))
    under = selector.best_side(X, np.array([0.40])).iloc[0]
    assert under.side == "under" and under.probability == pytest.approx(0.60)
    assert under.ev == pytest.approx(0.60 * (1 + 100 / 110) - 1)
    assert selector.best_side(X, np.array([0.60])).iloc[0].side == "over"


def test_at_the_books_price_every_bet_loses_the_hold():
    X = selector.design(frame())
    assert selector.best_side(X, selector.probability(Flat(), X)).iloc[0].ev < 0


def test_settle_and_record_use_the_offered_price_and_whole_days():
    m = pd.DataFrame(dict(over=[1, 0, 0], game_date=pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
                          season="2025-26", stat="pts"))
    sides = pd.DataFrame(dict(side=["over", "over", "under"], dec=[2.0, 2.0, 1.5], ev=0.1))
    bets = selector.settle(m, sides)
    assert bets.profit.tolist() == [1.0, -1.0, 0.5]
    rec = selector.record(bets)
    assert rec["bets"] == 3 and rec["roi"] == pytest.approx(0.5 / 3) and rec["unders"] == pytest.approx(1 / 3)
    day_sums = np.array([(1 - 0.5 / 3) + (-1 - 0.5 / 3), 0.5 - 0.5 / 3])
    assert rec["se"] == pytest.approx(np.sqrt((day_sums ** 2).sum()) / 3)
    assert selector.record(bets.iloc[:0])["bets"] == 0


def test_projection_must_be_made_before_the_quote():
    runs = [(NOW - timedelta(hours=5), 20.0, 1.0, 0.1), (NOW - timedelta(hours=1), 22.0, 1.0, 0.1),
            (NOW + timedelta(minutes=1), 30.0, 1.0, 0.1)]
    assert selector.projection_before(runs, NOW)[1] == 22.0
    assert selector.projection_before(runs[2:], NOW) is None
    assert selector.projection_before([(NOW, 30.0, 1.0, 0.1)], NOW) is None


def market(key, name, point, over, under, stamp):
    return dict(key=key, last_update=stamp, outcomes=[
        dict(name="Over", description=name, point=point, price=over, link="over-link"),
        dict(name="Under", description=name, point=point, price=under, link="under-link")])


def event():
    stamp = (NOW - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    book = lambda key, markets: dict(key=key, last_update=stamp, markets=markets)
    return dict(id="e1", home_team="Denver Nuggets", away_team="Utah Jazz", bookmakers=[
        book("fanduel", [market("player_points", "Nikola Jokic", 27.5, -115, -105, stamp),
                         market("player_points", "Jamal Murray", 20.0, -110, -110, stamp)]),
        book("draftkings", [market("player_points", "Nikola Jokic", 28.5, 100, -120, stamp)]),
        book("betmgm", [market("player_points", "Nikola Jokic", 27.5, -110, -110, stamp)]),
        book("pinnacle", [market("player_points", "Nikola Jokic", 27.5, -120, 100, stamp)])])


def test_offers_keep_half_point_bet_book_lines_and_pinnacle_only_at_the_same_line():
    found = {(o["book"], o["line"]): o for o in selector.offers(event(), NOW)}
    assert set(found) == {("fanduel", 27.5), ("draftkings", 28.5)}       # whole-number line and BetMGM dropped
    fd, dk = found[("fanduel", 27.5)], found[("draftkings", 28.5)]
    assert (fd["over_price"], fd["under_price"], fd["stat"], fd["over_link"]) == (-115, -105, "pts", "over-link")
    assert fd["sharp_fair_over"] > 0.5 and dk["sharp_fair_over"] is None
    assert fd["name_key"] == dk["name_key"]


def test_stale_quotes_are_not_scored():
    assert selector.offers(event(), NOW + timedelta(minutes=10)) == []


def test_score_needs_an_earlier_projection_and_prices_the_chosen_side():
    rows = selector.offers(event(), NOW)
    key = rows[0]["name_key"]
    day = NOW.date()
    late = {(day, key, "pts"): [(NOW + timedelta(minutes=5), 30.0, 1.0, 0.05)]}
    assert selector.score(rows, late, day, NOW, Flat()).empty
    early = {(day, key, "pts"): [(NOW - timedelta(hours=2), 30.0, 1.0, 0.05)]}
    scored = selector.score(rows, early, day, NOW, Flat(1.0))            # model leans hard to the over
    assert set(scored.side) == {"over"}
    fd = scored[scored.book == "fanduel"].iloc[0]
    assert fd.price == -115 and fd.link == "over-link" and fd.pred == 30.0
    assert fd.sharp_ev == pytest.approx(fd.sharp_fair * (1 + 100 / 115) - 1)
    assert pd.isna(scored[scored.book == "draftkings"].iloc[0].sharp_fair)
    under = selector.score(rows, early, day, NOW, Flat(-1.0))
    assert set(under.side) == {"under"} and under[under.book == "fanduel"].iloc[0].price == -105


def test_decide_keeps_one_pick_per_player_and_stat_above_the_threshold():
    scored = pd.DataFrame(dict(name_key=["a", "a", "b", "c"], market=["points"] * 4, book=["fanduel", "draftkings", "fanduel", "espnbet"],
                               ev=[0.07, 0.09, 0.05, 0.061]))
    picks = selector.decide(scored)
    assert sorted(zip(picks.name_key, picks.book)) == [("a", "draftkings"), ("c", "espnbet")]
    assert selector.decide(scored.iloc[:0]).empty


def test_model_file_must_match_its_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(selector, "MODEL_PATH", tmp_path / "model.txt")
    monkeypatch.setattr(selector, "MANIFEST", tmp_path / "manifest.json")
    assert selector.load_model() == (None, "model_missing")
    selector.MODEL_PATH.write_text("tree")
    selector.MANIFEST.write_text('{"version": "%s", "model_sha256": "not-this-file"}' % selector.VERSION)
    assert selector.load_model() == (None, "model_changed")
    assert selector.capture(None, NOW) == dict(status="model_changed", picks=0)


def test_tally_reports_results_and_the_edge_against_pinnacle():
    picks = pd.DataFrame(dict(result=["win", "loss", None], profit=[0.9, -1.0, None], ev=[0.07, 0.08, 0.09],
                              sharp_ev=[-0.04, None, -0.06], close_ev=[-0.05, -0.03, None],
                              day=[NOW.date(), NOW.date(), NOW.date() + timedelta(days=1)]))
    t = selector.tally(picks)
    assert (t["picks"], t["settled"], t["wins"], t["losses"]) == (3, 2, 1, 1)
    assert t["units"] == pytest.approx(-0.1) and t["roi"] == pytest.approx(-0.05)
    assert t["sharp_ev_at_pick"] == pytest.approx(-0.05) and t["with_sharp"] == 2
    assert t["close_ev"] == pytest.approx(-0.04) and t["with_close"] == 2
    text = selector.markdown(selector.summarize(picks.assign(replayed=True, book="fanduel", side="over", market="points"),
                                                dict(version=selector.VERSION), NOW))
    assert "Edge vs Pinnacle at close" in text and "1-1" in text
