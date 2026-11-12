"""News lag: prices as implied projections, news moves, the price before and after each, and the bets (no database)."""
import numpy as np
import pandas as pd
import pytest

from nba_props import news_lag as nl

TIP = pd.Timestamp("2026-11-01 00:00", tz="UTC")
DAY = pd.Timestamp("2026-10-31")
at = lambda minutes: TIP - pd.Timedelta(minutes=minutes)


def projections() -> pd.DataFrame:
    """Team AAA: the star (1) is dropped at T-38, his backup (2) starts and the regular (3) gains; player 7 has no news."""
    runs = [  # minutes before tip, trigger, {player: (mu, starter)}
        (300, "scheduled", {1: (25.0, True), 2: (8.0, False), 3: (15.0, True), 7: (20.0, True)}),
        (44, "news", {1: (25.0, True), 2: (8.0, False), 3: (15.6, True), 7: (20.0, True)}),
        (38, "news", {2: (12.0, True), 3: (16.5, True), 7: (20.0, True)}),
        (18, "news", {2: (12.1, True), 3: (16.0, True), 7: (20.0, True)}),
        (3, "news", {2: (12.1, True), 3: (17.0, True), 7: (20.0, True)}),
    ]
    rows = [dict(run_at=at(m), game_date=DAY, game_id="g1", tip_utc=TIP, player_id=p, name_key=f"player {p}", team="BBB" if p == 7 else "AAA",
                 stat="pts", mu=mu, a=4.0, b=0.0, starter=starter, trigger=trigger) for m, trigger, players in runs for p, (mu, starter) in players.items()]
    rows += [dict(run_at=at(m), game_date=DAY, game_id="g2", tip_utc=TIP, player_id=9, name_key="player 9", team="CCC", stat="pts", mu=mu,
                  a=4.0, b=0.0, starter=True, trigger="scheduled") for m, mu in ((300, 10.0), (120, 12.0))]
    return pd.DataFrame(rows)


def look(minutes: float, book: str, prices: dict) -> tuple:
    """One saved look: {player: (line, over, under)}."""
    outcomes = [dict(name=side, point=line, price=price, description=f"Player {p}") for p, (line, over, under) in prices.items()
                for side, price in (("Over", over), ("Under", under))]
    return ("e1", at(minutes), TIP, dict(bookmakers=[dict(key=book, markets=[dict(key="player_points", outcomes=outcomes),
                                                                              dict(key="h2h", outcomes=[dict(name="AAA", price=-150)])])]))


def looks() -> list[tuple]:
    even = lambda line: (line, -110, -110)
    return [look(45, "fanduel", {3: even(14.5), 7: even(20.5)}),
            look(35, "fanduel", {2: even(9.5), 3: even(15.5), 7: even(20.5)}), look(35, "pinnacle", {3: (15.5, -105, -115)}),
            look(25, "fanduel", {3: (15.5, -120, 100)}),
            look(15, "fanduel", {3: (15.5, -115, -105)}),
            look(5, "fanduel", {2: even(10.5), 3: even(16.5), 7: even(21.5)}), look(5, "pinnacle", {3: (15.5, -130, 110)}),
            look(-5, "fanduel", {3: even(18.5)})]                                              # after tip: not used


def frames():
    proj = projections()
    quotes = nl.quote_table(nl.parse_quotes(looks()), proj)
    moves = nl.news_moves(proj)
    return proj, quotes, moves, nl.timeline(moves, quotes)


def test_a_price_and_the_projection_it_implies_are_inverses():
    line, p, a, b = np.array([20.5, 4.5, 1.5]), np.array([0.5, 0.58, 0.41]), np.array([4.0, 1.5, 1.3]), np.array([0.0, 0.001, 0.0])
    mu = nl.implied_mean(line, p, a, b)
    assert mu[0] == pytest.approx(20.5) and mu[1] > 4.5 and mu[2] < 1.5
    assert nl.over_probability(mu, line, a, b) == pytest.approx(p, abs=1e-6)
    assert nl.implied_mean(21.5, 0.5, 4.0, 0.0) > mu[0] and nl.implied_mean(20.5, 0.55, 4.0, 0.0) > mu[0]   # a higher line or a shorter over: a higher projection
    assert nl.american_probability([-110, 150]) == pytest.approx([110 / 210, 0.4]) and nl.decimal([-110, 150]) == pytest.approx([1 + 100 / 110, 2.5])


def test_saved_looks_become_one_two_sided_price_per_player_and_book():
    payload = dict(bookmakers=[dict(key="fanduel", markets=[
        dict(key="player_points", outcomes=[dict(name="Over", point=20.5, price=-110, description="Michael Porter Jr."),
                                            dict(name="Under", point=20.5, price=-110, description="Michael Porter Jr."),
                                            dict(name="Over", point=24.5, price=250, description="Michael Porter Jr."),
                                            dict(name="Under", point=24.5, price=-330, description="Michael Porter Jr."),
                                            dict(name="Over", point=9.5, price=-105, description="One Sided")]),
        dict(key="totals", outcomes=[dict(name="Over", point=220.5, price=-110, description="Game")])])])
    q = nl.parse_quotes([("e1", at(30), TIP, payload)])
    assert len(q) == 1 and (q.name_key[0], q.stat[0], q.line[0], q.book[0]) == ("michael porter", "pts", 20.5, "fanduel")
    assert q.p[0] == pytest.approx(0.5)
    _, quotes, _, _ = frames()
    assert quotes.observed_at.max() == at(5) and set(quotes.player_id) == {2, 3, 7}                 # the look after tip is dropped
    assert quotes[(quotes.player_id == 3) & (quotes.observed_at == at(45))].imp.iloc[0] == pytest.approx(14.5)


def test_news_moves_are_projection_changes_in_news_runs_with_their_kind():
    _, _, moves, _ = frames()
    assert set(moves.game_id) == {"g1"}                                                              # a scheduled refit is not news
    by = moves.set_index(["player_id", "run_at"])
    assert by.loc[(2, at(38))].kind == "role" and by.loc[(2, at(38))].move == pytest.approx(0.5)
    assert by.loc[(3, at(38))].kind == "teammates" and by.loc[(3, at(44))].kind == "other"          # the star left the projections at T-38
    assert by.loc[(3, at(18))].direction == -1 and (2, at(18)) not in by.index                      # under 3%: not a move
    assert 7 not in set(moves.player_id)


def test_each_move_gets_the_price_before_we_saw_it_the_first_after_and_the_last():
    _, _, _, t = frames()
    fd = t[t.book == "fanduel"].set_index(["player_id", "run_at"])
    gain = fd.loc[(3, at(38))]
    assert (gain.pre_observed_at, gain.first_observed_at, gain.close_observed_at) == (at(45), at(35), at(5)) and gain.age == pytest.approx(3.0)
    assert gain.before_look == pytest.approx(15.5 / 14.5 - 1) and gain.after_look == pytest.approx(16.5 / 15.5 - 1)
    assert gain.whole == pytest.approx(16.5 / 14.5 - 1)
    assert (3, at(44)) not in fd.index                                                               # shares its first price with the later move
    assert (3, at(3)) not in fd.index                                                                # no price after it
    drop = fd.loc[(3, at(18))]                                                                       # the model moved down; the line then went up
    assert drop.first_observed_at == at(15) and drop.after_look < 0 and drop.pre_observed_at == at(25)
    backup = fd.loc[(2, at(38))]
    assert np.isnan(backup.before_look) and backup.after_look == pytest.approx(10.5 / 9.5 - 1)      # no price before the news
    assert set(t.book) == {"fanduel", "pinnacle"}


def test_news_bets_are_judged_at_the_reference_close_and_by_the_result():
    _, quotes, _, t = frames()
    actuals = pd.DataFrame(dict(game_id="g1", player_id=[3, 2], stat="pts", actual=[20.0, 9.0]))
    bets = nl.news_bets(t, quotes, actuals).set_index(["player_id", "side"])
    assert set(bets.book) == {"fanduel"} and len(bets) == 3
    over = bets.loc[(3, 1.0)]
    fair = (130 / 230) / (130 / 230 + 100 / 210)                                                    # Pinnacle's last price, same line
    assert over.line == 15.5 and over.ev == pytest.approx(fair * (1 + 100 / 110) - 1) and over.profit == pytest.approx(100 / 110) and over.won == 1
    under = bets.loc[(3, -1.0)]
    assert under["under"] == -105 and under.ev == pytest.approx((1 - fair) * (1 + 100 / 105) - 1) and under.profit == -1
    assert bets.loc[(2, 1.0)].profit == -1 and np.isnan(bets.loc[(2, 1.0)].ev)                     # no reference price for him
    early = nl.news_bets(t, quotes, actuals, at="pre").set_index(["player_id", "side"])
    assert early.loc[(3, 1.0)].line == 14.5 and early.loc[(3, 1.0)].ev > over.ev and (2, 1.0) not in early.index
    void = nl.news_bets(t, quotes, actuals[actuals.player_id == 2])
    assert void[void.player_id == 3].profit.isna().all()                                            # he did not play: no result
    last = nl.graded(pd.DataFrame([dict(game_id="g1", player_id=3, stat="pts", side=1.0, line=15.5, over=-110, under=-110, at=at(5), a=4.0, b=0.0)]),
                     quotes, actuals)
    assert np.isnan(last.ev.iloc[0])                                                                 # the reference close is not later than the bet


def test_the_control_is_players_without_news_and_the_report_reads():
    _, quotes, moves, t = frames()
    actuals = pd.DataFrame(dict(game_id="g1", player_id=[3, 2, 7], stat="pts", actual=[20.0, 9.0, 21.0]))
    drift, blind = nl.control(moves, quotes, actuals)
    assert set(drift.player_id) == {7} and drift.drift.iloc[0] == pytest.approx(21.5 / 20.5 - 1)
    assert len(blind) == 2 and blind.set_index("side").profit.to_dict() == {1.0: pytest.approx(100 / 110), -1.0: -1.0}
    bets, early = nl.news_bets(t, quotes, actuals), nl.news_bets(t, quotes, actuals, at="pre")
    report = nl.summarise(moves, quotes, t, bets, drift, blind, early)
    assert report["sample"]["fresh"] == 3 and report["fresh_by_book"]["fanduel"]["with_price_before"] == 2
    assert report["bets"]["Fresh, over side"]["bets"] == 2 and list(report["speed"].values())[0]["bets"] == 2
    text = nl.markdown(nl.clean(report))
    assert "## Fresh news: where the market's move happened" in text and "What being one look faster would be worth" in text
    days = pd.DataFrame(dict(game_date=pd.date_range("2026-11-01", periods=6).repeat(2), x=[0.01, 0.03] * 6))
    full, few = nl.mean_interval(days, "x"), nl.mean_interval(days.head(4), "x")
    assert full["mean"] == pytest.approx(0.02) and full["low"] == pytest.approx(0.02) and few["low"] is None and nl.mean_interval(days.iloc[0:0], "x") is None
