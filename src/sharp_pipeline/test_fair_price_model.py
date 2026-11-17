"""Fitted fair price: the shared line, the fit, walk-forward by day and the bets (no database)."""
import numpy as np
import pandas as pd
import pytest

from sharp_pipeline import fair_price_model as fp

TIP = pd.Timestamp("2026-11-01 23:00", tz="UTC")


def quote(book, line, over, under, minutes=40, player="p1", actual=12.0, seconds=0, day=0, game="g1"):
    tip = TIP + pd.Timedelta(days=day)
    return dict(sport="nba", game=game, player=player, stat="pts", book=book, line=line, over=over, under=under,
                observed_at=tip - pd.Timedelta(minutes=minutes) + pd.Timedelta(seconds=seconds), commence=tip, actual=actual)


def test_a_row_is_the_first_look_on_the_line_most_books_share():
    quotes = pd.DataFrame([
        quote("pinnacle", 10.5, -120, 100), quote("fanduel", 10.5, -110, -110, seconds=30), quote("draftkings", 10.5, -105, -115, seconds=40),
        quote("betmgm", 11.5, 110, -130, seconds=40),                                     # another line: left out
        quote("fanduel", 10.5, 150, -180, minutes=30),                                    # a later look
        quote("espnbet", 10.5, -110, -110, minutes=60),                                   # before the window
        quote("pinnacle", 5.5, -110, -110, player="p2"), quote("fanduel", 5.5, -110, -110, player="p2"),          # two books only
        quote("pinnacle", 7.5, -110, -110, player="p3", actual=7.5), quote("fanduel", 8.5, -110, -110, player="p3", actual=7.5),
        quote("draftkings", 8.5, -110, -110, player="p3", actual=7.5), quote("betmgm", 7.5, -110, -110, player="p3", actual=7.5),
        quote("espnbet", 7.5, -110, -110, player="p3", actual=7.5)])
    long = fp.looks(quotes)
    assert sorted(long[long.player == "p1"].book) == ["draftkings", "fanduel", "pinnacle"] and set(long.player) == {"p1", "p3"}
    assert long[long.player == "p1"].over[long.book == "fanduel"].iloc[0] == -110 and (long[long.player == "p3"].line == 7.5).all()
    rows = fp.table(long)
    one = rows[rows.player == "p1"].iloc[0]
    assert one.books == 3 and one.over_hit == 1 and one.dev_betmgm == 0 and one.group == "nba pts"
    assert one.dev_pinnacle + one.dev_fanduel + one.dev_draftkings == pytest.approx(0) and one.dev_pinnacle > 0
    assert np.isnan(rows[rows.player == "p3"].over_hit.iloc[0])                            # a push


def simulated(n=4000, days=20, seed=4):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 0.4, n)
    sharp, soft = truth + rng.normal(0, 0.05, n), truth + rng.normal(0, 0.3, n)
    consensus = (sharp + soft) / 2
    return pd.DataFrame(dict(sport="nba", game=[f"g{i % 80}" for i in range(n)], player=[f"p{i}" for i in range(n)], stat="pts",
                             day=[pd.Timestamp("2026-10-01").date() + pd.Timedelta(days=int(i * days / n)) for i in range(n)],
                             consensus=consensus, dev_sharp=sharp - consensus, dev_soft=soft - consensus, group="nba pts",
                             over_hit=(rng.uniform(size=n) < 1 / (1 + np.exp(-truth))).astype(float)))


def test_the_fit_finds_the_book_that_is_right_and_stays_at_the_average_without_evidence():
    rows = simulated()
    fit = fp.Fit(rows, ["sharp", "soft"], shrink=1.0)
    w = fit.weights()
    assert w["sharp"] > 0.3 and w["soft"] < -0.3 and abs(w["nba pts (toward the over)"]) < 0.15
    assert fp.log_loss(fit.predict(rows), rows.over_hit).mean() < fp.log_loss(1 / (1 + np.exp(-rows.consensus)), rows.over_hit).mean()
    flat = fp.Fit(rows.head(20), ["sharp", "soft"], shrink=1e6).weights()
    assert flat["average of the books"] == pytest.approx(1, abs=1e-3) and flat["sharp"] == pytest.approx(0, abs=1e-3)
    assert fp.Fit(rows.assign(over_hit=np.nan), ["sharp", "soft"]).coef.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_each_day_is_forecast_from_earlier_days_only():
    rows = simulated(n=2000, days=10)
    oof = fp.walk_forward(rows, ["sharp", "soft"], min_days=4)
    days = sorted(rows.day.unique())
    assert sorted(oof.day.unique()) == days[4:] and len(oof) == (rows.day >= days[4]).sum()
    first = rows[rows.day == days[4]]
    assert oof[oof.day == days[4]].p_model.to_numpy() == pytest.approx(fp.Fit(rows[rows.day < days[4]], ["sharp", "soft"]).predict(first))
    later = rows.assign(over_hit=np.where(rows.day >= days[4], 1 - rows.over_hit, rows.over_hit))          # change today's results
    assert fp.walk_forward(later, ["sharp", "soft"], min_days=4)[lambda f: f.day == days[4]].p_model.to_numpy() == pytest.approx(
        oof[oof.day == days[4]].p_model.to_numpy())
    score = fp.against(oof, 1 / (1 + np.exp(-oof.consensus)), draws=100)
    assert score["rows"] == len(oof) and score["diff"] < 0 and score["low"] < score["diff"] < score["high"]


def test_bets_are_the_better_side_of_a_book_at_the_fair_chance():
    quotes = pd.DataFrame([quote("pinnacle", 10.5, -130, 110), quote("fanduel", 10.5, 100, -120), quote("draftkings", 10.5, -160, 135)])
    long = fp.looks(quotes)
    fair = fp.table(long).assign(p_model=0.56)
    bets = fp.offers(long, fair, "p_model").set_index("book")
    assert bets.loc["fanduel", "side"] == 1 and bets.loc["fanduel", "ev"] == pytest.approx(0.56 * 2 - 1) and bets.loc["fanduel", "profit"] == pytest.approx(1.0)
    assert bets.loc["draftkings", "side"] == -1 and bets.loc["draftkings", "profit"] == -1
    others = fp.leave_one_out(long).set_index("book")
    rest = fp.gv.logit(long[long.book != "fanduel"].p).mean()
    assert others.loc["fanduel", "p_others"] == pytest.approx(1 / (1 + np.exp(-rest)))


def test_the_report_reads():
    rng = np.random.default_rng(2)
    quotes = []
    for day in range(10):
        for i in range(30):
            actual = float(rng.integers(5, 16))
            for book, (over, under) in dict(pinnacle=(-115, -105), fanduel=(100, -120), draftkings=(-110, -110), betmgm=(-120, 100)).items():
                quotes.append(quote(book, 10.5, over, under, player=f"p{i}", actual=actual, day=day, game=f"g{day}_{i % 6}"))
    report = fp.news_lag.clean(fp.summarise(fp.looks(pd.DataFrame(quotes))))
    assert report["sample"]["rows"] == 300 and report["sample"]["forecast_days"] == 3 and report["sample"]["forecast_rows"] == 90
    text = fp.markdown(report)
    assert "## What the fit learned" in text and "| average of the books |" in text and "## Bets at 3%+ above: Pinnacle's price" in text
