"""Game-line test: the line's sign, walk-forward by season, the weight beside the line and the pick record (no database)."""
import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import game_line_test as gl


def games(seasons=(2008, 2009, 2010, 2011), per=40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = len(seasons) * per
    strength = rng.normal(0, 6, n)
    return pd.DataFrame(dict(game_id=[f"g{i}" for i in range(n)], season=np.repeat(seasons, per), week=np.tile(np.arange(per) % 18 + 1, len(seasons)),
                             home_margin=np.round(strength + rng.normal(0, 10, n)), total_points_actual=np.round(44 + rng.normal(0, 10, n)),
                             market_spread_home=-np.round(strength * 2) / 2, market_total=44.5, home_margin_avg_5=strength + rng.normal(0, 3, n),
                             home_pf_avg_5=22.0 + rng.normal(0, 2, n), roof="dome"))


def test_the_line_margin_is_the_home_margin_the_spread_expects():
    df = gl.with_line_margin(games())
    assert np.corrcoef(df.line_margin, df.home_margin)[0, 1] > 0 and (df.line_margin == -df.market_spread_home).all()
    flipped = gl.with_line_margin(games().assign(market_spread_home=lambda f: -f.market_spread_home))
    assert (flipped.line_margin == df.line_margin).all()
    assert gl.feature_columns(df) == ["home_margin_avg_5", "home_pf_avg_5"]                # no ids, results or lines


def test_each_season_is_forecast_from_earlier_seasons_only():
    df = gl.with_line_margin(games())
    seen = []

    class Mean:
        def __init__(self, X, y):
            seen.append(len(X))
            self.value = float(y.mean())

        def predict(self, X):
            return np.full(len(X), self.value)

    oof = gl.walk_forward(df, 2010, fitter=Mean)
    assert sorted(oof.season.unique()) == [2010, 2011] and seen == [80] * 4 + [120] * 4      # two markets, two models, per season
    first = oof[oof.season == 2010]
    assert first.spread_direct.iloc[0] == pytest.approx(df[df.season < 2010].home_margin.mean())
    beyond = (df[df.season < 2010].home_margin - df[df.season < 2010].line_margin).mean()
    assert (first.spread_residual - first.line_margin).iloc[0] == pytest.approx(beyond)


def test_weight_and_picks_are_measured_against_the_line():
    rows = pd.DataFrame(dict(season=[2020, 2020, 2021, 2021, 2021], y=[10.0, -3.0, 7.0, 0.0, 3.0], line=[3.0, 3.0, 3.0, -3.0, 3.0],
                             model=[7.0, 7.0, -1.0, 1.0, 4.0]))
    w = gl.weight_beside_line(rows.y.to_numpy(), rows.line.to_numpy(), rows.line.to_numpy() + 2 * (rows.y - rows.line).to_numpy())
    assert w["weight"] == pytest.approx(0.5) and w["se"] == pytest.approx(0.0, abs=1e-9)
    p = gl.picks(rows, "y", "line", "model", 4.0)                  # home, home, away, home; the fifth is too close to the line
    assert (p["picks"], p["wins"], p["losses"], p["pushes"]) == (4, 2, 2, 0) and p["rate"] == 0.5
    assert p["roi"] == pytest.approx((2 * gl.WIN - 2) / 4) and p["seasons"] == 2 and p["winning_seasons"] == 0
    push = gl.picks(rows.assign(y=[3.0, -3.0, 7.0, 0.0, 3.0]), "y", "line", "model", 4.0)
    assert push["pushes"] == 1 and push["wins"] + push["losses"] == 3
    assert gl.picks(rows, "y", "line", "model", 50.0)["rate"] is None


def test_the_report_reads():
    df = gl.with_line_margin(games())
    oof = df[df.season >= 2010].assign(spread_direct=lambda f: f.line_margin + 3, spread_residual=lambda f: f.line_margin + 0.6,
                                       total_direct=45.0, total_residual=44.0)
    doc = dict(built_at="now", first_season=2010, last_season=2011, markets=gl.evaluate(oof))
    text = gl.markdown(doc)
    assert "## Home margin (80 games)" in text and "| Direct | 2+ | 80 |" in text and "| Residual | 0.5+ | 80 |" in text
    assert doc["markets"]["total"]["models"]["direct"]["picks"][0]["picks"] == 0
