import pytest

from nfl_pipeline import sharp_math as m


def test_no_vig_and_ev():
    assert m.no_vig_over(-110, -110) == pytest.approx(0.5)
    assert m.ev(0.5, -110) == pytest.approx(-0.04545, abs=1e-4)
    assert m.ev(0.55, -110) > 0.04


def test_nearby_line_conversion_moves_probability_the_right_way():
    p = m.no_vig_over(-110, -110)
    higher = m.over_probability_at("receiving_yards", 45.5, p, 46.5)
    lower = m.over_probability_at("receiving_yards", 45.5, p, 44.5)
    assert higher < 0.5 < lower and abs(0.5 - higher) == pytest.approx(abs(lower - 0.5), abs=1e-9)
    assert 0.48 < higher < 0.49  # one yard on a ~26-yard spread is ~1.5 points of probability
    assert m.over_probability_at("receiving_yards", 45.5, p, 45.5) == pytest.approx(0.5)


def test_conversion_refuses_large_gaps_spreads_and_counts():
    assert m.over_probability_at("receiving_yards", 45.5, 0.5, 49.5) is None
    assert m.over_probability_at("spread", -3.0, 0.5, -3.5) is None
    assert m.over_probability_at("receptions", 4.5, 0.5, 5.5) is None
    assert m.over_probability_at("receptions", 4.5, 0.55, 4.5) == pytest.approx(0.55)


@pytest.mark.parametrize("p", [0.45, 0.5, 0.55, 0.6])
def test_minimum_price_is_the_breakeven_plus_margin(p):
    price = m.minimum_price(p, min_ev=0.01)
    assert m.ev(p, price) >= 0.01 - 1e-9
    assert m.ev(p, price) < 0.01 + 0.01  # tight: no more than ~1 extra point of EV of slack
