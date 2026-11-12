"""Price-gap validity: pairing with Pinnacle, the better side, the weight on Pinnacle and returns (no database)."""
import numpy as np
import pandas as pd
import pytest

from sharp_pipeline import gap_validity as gv

TIP = pd.Timestamp("2026-11-01 00:00", tz="UTC")


def quote(book, minutes, line, over, under, actual=12.0, player="p1", seconds=0):
    return dict(sport="nba", game="g1", player=player, stat="pts", book=book, line=line, over=over, under=under,
                observed_at=TIP - pd.Timedelta(minutes=minutes) + pd.Timedelta(seconds=seconds), commence=TIP, actual=actual)


def test_a_book_price_is_paired_with_pinnacle_at_the_same_line_in_the_same_look():
    quotes = pd.DataFrame([quote("pinnacle", 40, 10.5, -130, 110), quote("fanduel", 40, 10.5, 100, -120, seconds=20),
                           quote("draftkings", 40, 11.5, -110, -110, seconds=20),          # another line: no pair
                           quote("fanduel", 20, 10.5, -110, -110),                          # no Pinnacle price in that look
                           quote("pinnacle", -5, 10.5, -130, 110), quote("fanduel", -5, 10.5, 100, -120, seconds=5)])   # after the start
    rows = gv.paired(quotes)
    assert len(rows) == 1 and rows.book.iloc[0] == "fanduel" and rows.minutes.iloc[0] == pytest.approx(40 - 20 / 60)
    fair = (130 / 230) / (130 / 230 + 100 / 210)
    row = rows.iloc[0]
    assert row.p_ref == pytest.approx(fair) and row.side == 1 and row.ev == pytest.approx(fair * 2.0 - 1.0)
    assert row.profit == pytest.approx(1.0) and row.over_hit == 1                           # 12 goes over 10.5 at +100
    under = gv.paired(pd.DataFrame([quote("pinnacle", 40, 10.5, 110, -130), quote("fanduel", 40, 10.5, -120, 100, actual=10.5)])).iloc[0]
    assert under.side == -1 and under.profit == 0 and np.isnan(under.over_hit)              # a push
    void = gv.paired(pd.DataFrame([quote("pinnacle", 40, 10.5, 110, -130), quote("fanduel", 40, 10.5, -120, 100, actual=np.nan)])).iloc[0]
    assert np.isnan(void.profit)


def test_the_window_keeps_the_first_look_per_player_and_book():
    quotes = pd.DataFrame([q for m in (50, 40, 30, 10) for q in (quote("pinnacle", m, 10.5, -110, -110), quote("fanduel", m, 10.5, -105, -115))])
    rows = gv.paired(quotes)
    assert gv.first_in(rows, *gv.WINDOW).minutes.tolist() == [40.0] and gv.first_in(rows, 5.0, 15.0).minutes.tolist() == [10.0]


def test_the_weight_tells_whose_price_matched_the_results():
    rng = np.random.default_rng(5)
    n = 6000
    truth = rng.uniform(0.35, 0.65, n)
    noisy = np.clip(truth + rng.normal(0, 0.04, n), 0.05, 0.95)
    hit = (rng.uniform(size=n) < truth).astype(float)
    assert gv.fit_weight(noisy, truth, hit) > 0.6                                           # the reference is the truth
    assert gv.fit_weight(truth, noisy, hit) < 0.4                                           # the book is
    rows = pd.DataFrame(dict(sport="nba", game=[f"g{i % 30}" for i in range(n)], p_book=noisy, p_ref=truth, over_hit=hit))
    w = gv.weight(rows, draws=50)
    assert w["rows"] == n and w["games"] == 30 and w["low"] < w["weight"] < w["high"] and gv.weight(rows.head(50)) is None


def test_returns_and_the_report():
    rows = pd.DataFrame(dict(sport="nba", game=[f"g{i % 12}" for i in range(48)], ev=0.04, profit=[1.0, -1.0, 1.0, 0.0] * 12))
    r = gv.returns(rows, draws=100)
    assert r["bets"] == 48 and r["roi"] == pytest.approx(0.25) and r["hit"] == pytest.approx(2 / 3) and r["claimed"] == pytest.approx(0.04)
    assert gv.returns(rows.iloc[0:0])["roi"] is None
    quotes = pd.DataFrame([q for i in range(12) for q in (
        {**quote("pinnacle", 40, 10.5, -130, 110), "game": f"g{i}"}, {**quote("fanduel", 40, 10.5, 100, -120, actual=12.0 if i % 2 else 9.0), "game": f"g{i}"})])
    text = gv.markdown(gv.news_lag.clean(gv.summarise(gv.paired(quotes))))
    assert "## Whose price is right" in text and "| All | 12 |" in text and "| 45 to 35 minutes | 12 |" in text
