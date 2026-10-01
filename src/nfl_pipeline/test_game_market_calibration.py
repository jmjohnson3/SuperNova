import pandas as pd
import pytest

from nfl_pipeline.modeling import predict_today as games


def line(**prices):
    base = dict(spread_home_points=-3.0, spread_away_points=3.0, spread_home_price=-120, spread_away_price=100,
                total_points=44.5, total_over_price=-110, total_under_price=-110)
    return pd.Series(dict(base, **prices))


def cal(spread, total):
    return {"version": "test", "markets": {"spread": {"probability_trust": spread}, "total": {"probability_trust": total}}}


def test_zero_trust_prices_each_side_at_fanduel_no_vig():
    pick = games._best_spread_candidate({}, 10.0, 0.0, 13.5, line(), cal(0.0, 0.0))
    home_raw, away_raw = 120 / 220, 100 / 200
    expected = (home_raw if pick["side"] == "home" else away_raw) / (home_raw + away_raw)
    assert pick["probability"] == pytest.approx(expected)
    assert pick["model_probability"] != pytest.approx(pick["probability"])
    assert pick["market_no_vig_probability"] == pytest.approx(expected) and pick["market_trust"] == 0.0
    assert pick["ev"] < 0  # at the market price only the vig remains


def test_full_trust_keeps_model_probability_and_missing_file_is_identity(monkeypatch, tmp_path):
    kept = games._best_total_candidate({}, 52.0, 44.0, 10.5, line(), cal(1.0, 1.0))
    assert kept["probability"] == pytest.approx(kept["model_probability"])
    monkeypatch.setattr(games, "MARKET_CALIBRATION_PATH", tmp_path / "missing.json")
    assert games._load_market_calibration() == {}
    assert games._market_trust({}, "total") == 1.0


def test_half_trust_lands_between_market_and_model():
    pick = games._best_total_candidate({}, 52.0, 44.0, 10.5, line(), cal(1.0, 0.5))
    lo, hi = sorted((pick["market_no_vig_probability"], pick["model_probability"]))
    assert lo < pick["probability"] < hi
