import pytest

from nfl_pipeline.modeling import sharp_gap_backtest as bt


def snap(fd_line, fd_over, fd_under, pin_line, pin_over, pin_under, player="Terry McLaurin"):
    pair = lambda line, o, u: [dict(name="Over", description=player, point=line, price=o),
                               dict(name="Under", description=player, point=line, price=u)]
    mk = lambda key, outs: {"key": key, "markets": [{"key": "player_reception_yds", "outcomes": outs}]}
    return {"bookmakers": [mk("fanduel", pair(fd_line, fd_over, fd_under)), mk("pinnacle", pair(pin_line, pin_over, pin_under))]}


def test_fanduel_line_moves_are_graded_at_the_bets_line():
    # T-90: FanDuel 45.5 while Pinnacle is 47.5 -> over is cheap. Close: FanDuel moved to 49.5 at the same price.
    snaps = {90: snap(45.5, -112, -112, 47.5, -115, -115), 45: {}, 15: {},
             bt.CLOSE_MINUTES: snap(49.5, -112, -112, 47.5, -132, 100)}
    rows = {r["side"]: r for r in bt.sides(dict(game_id="g", week=3), snaps)}
    over, under = rows["over"], rows["under"]
    assert over["ev"] > 0.0 and over["fd_close_line"] == 49.5
    assert over["fd_clv"] > 0.05 and under["fd_clv"] == pytest.approx(-over["fd_clv"])  # FanDuel moved toward the over
    assert over["ev_at_sharp_close"] > over["ev"] > 0  # Pinnacle steamed the over too


def test_shift_limits():
    assert bt._shift("receiving_yards", 45.5, 0.5, 45.5) == 0.5
    assert bt._shift("receiving_yards", 45.5, 0.5, 49.5) < 0.5
    assert bt._shift("receiving_yards", 45.5, 0.5, 60.5) is None  # beyond the movement cap
    assert bt._shift("receptions", 4.5, 0.5, 5.5) is None  # counts: exact line only
