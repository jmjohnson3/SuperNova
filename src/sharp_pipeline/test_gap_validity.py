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
    report = gv.summarise(gv.paired(quotes))
    earlier = [dict(date="2026-01-01", bets=3, claimed=0.04, roi=0.1, roi_low=None, roi_high=None, weight=None, weight_low=None, weight_high=None),
               dict(gv.headline(report), bets=1)]                                           # an earlier run today is replaced
    report["history"] = gv.with_history(report, earlier)
    assert [h["bets"] for h in report["history"]] == [3, 12] and report["history"][1]["date"] == report["built"][:10]
    text = gv.markdown(gv.news_lag.clean(report))
    assert "## Progress, run by run (about 1,500 prop bets" in text and "| 2026-01-01 | 3 | +4.0% | +10.0% | - | 0 | - |" in text


def market(book, key, outcomes, minutes=40, game="e1"):
    return dict(sport="nba", game=game, observed_at=TIP - pd.Timedelta(minutes=minutes), commence=TIP, home_team="Home", away_team="Away",
                book=book, market=key, outcomes=outcomes)


def test_game_lines_are_held_from_the_home_side():
    side = lambda name, price, point=None: dict(name=name, price=price, **({} if point is None else {"point": point}))
    rows = gv.line_rows([
        market("pinnacle", "spreads", [side("Away", -105, 3.5), side("Home", -105, -3.5)]),
        market("fanduel", "spreads", [side("Home", 110, -3.5), side("Away", -130, 3.5)]),
        market("draftkings", "spreads", [side("Home", -110, -4.0), side("Away", -110, 4.0)]),      # another line: no pair
        market("pinnacle", "h2h", [side("Home", -160), side("Away", 140)]),
        market("fanduel", "h2h", [side("Home", -150), side("Away", 130)]),
        market("pinnacle", "totals", [side("Over", -110, 220.5), side("Under", -110, 220.5)]),
        market("fanduel", "totals", [side("Over", -120, 220.5), side("Under", 100, 220.5)]),
        market("fanduel", "totals", [side("Over", -120, 220.5)]),                                   # one side only
        market("fanduel", "spreads", [side("Home", -110, -3.5), side("Away", -110, 4.5)]),          # the two sides disagree
        market("fanduel", "team_totals", [side("Over", -110, 110.5), side("Under", -110, 110.5)])])
    assert len(rows) == 7 and rows[rows.stat == "spread"].line.tolist() == [3.5, 3.5, 4.0] and (rows[rows.stat == "moneyline"].line == 0).all()
    fd = rows[(rows.book == "fanduel") & (rows.stat == "spread")].iloc[0]
    assert (fd.over, fd.under) == (110.0, -130.0)                                                   # over is the home price
    scores = pd.DataFrame(dict(game=["e1"], home_points=[110.0], away_points=[106.0]))
    p = gv.paired(gv.with_scores(rows, scores)).set_index("stat")
    assert sorted(p.index) == ["moneyline", "spread", "total"] and (p.kind == "Game lines").all()
    assert p.loc["spread", "side"] == 1 and p.loc["spread", "profit"] == pytest.approx(1.10)        # home by 4 covers 3.5 at +110
    assert p.loc["moneyline", "over_hit"] == 1 and p.loc["total", "side"] == -1 and p.loc["total", "profit"] == pytest.approx(1.0)
    push = gv.paired(gv.with_scores(rows[rows.stat == "spread"].assign(line=4.0), scores))
    assert (push.profit == 0).all()
    unsettled = gv.paired(gv.with_scores(rows, scores.assign(game="other")))
    assert unsettled.profit.isna().all() and len(unsettled) == 3
    props = gv.paired(pd.DataFrame([quote("pinnacle", 40, 10.5, -130, 110), quote("fanduel", 40, 10.5, 100, -120)]))
    report = gv.summarise(pd.concat([p.reset_index(), props], ignore_index=True))
    assert set(report["sample"]) == {"NBA game lines", "NBA player props"} and report["flagged"]["Game lines"]["bets"] >= 1
    line = gv.headline(report)
    assert line["line_bets"] == report["flagged"]["Game lines"]["bets"] and line["bets"] == report["flagged"]["Player props"]["bets"]
    assert line["line_roi"] == report["flagged"]["Game lines"]["roi"]
