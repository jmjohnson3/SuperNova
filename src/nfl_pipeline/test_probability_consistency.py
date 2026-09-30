from datetime import datetime, timezone

import numpy as np
import pytest
from scipy.stats import poisson

from nfl_pipeline.modeling import predict_player_props as props
from nfl_pipeline.modeling.selected_pick_diagnostics import monotonicity_violations


@pytest.mark.parametrize('probability', [0.01, .1, .333333333333, .4, .5, .50001, .55, .6, .69, .8, .9, .99])
def test_minimum_price_is_first_strictly_positive_ev_tick(probability):
    price = props._break_even_american_price(probability)
    previous = -101 if price == 100 else price-1
    assert abs(price) >= 100
    assert props._price_has_positive_ev(probability, price)
    assert not props._price_has_positive_ev(probability, previous)


def test_rounding_regressions_and_invalid_inputs():
    assert props._break_even_american_price(.69) == -222
    assert props._break_even_american_price(.6) == -149
    assert props._break_even_american_price(.5) == 101
    for value in (None, float('nan'), float('inf'), 0, 1, -.1, 1.1):
        assert props._break_even_american_price(value) is None
    for price in (None, float('nan'), 0, 99, -99):
        assert props._ev_per_unit(.6, price) is None


@pytest.mark.parametrize('stat,mean', [('passing_tds', 1.5), ('rushing_tds', .7), ('receiving_tds', .3),
                                     ('receiving_yards', 45), ('rushing_yards', 55), ('passing_yards', 240)])
@pytest.mark.parametrize('empirical', [False, True])
def test_unconditional_tails_are_monotone_and_push_mass_sums_to_one(stat, mean, empirical):
    dist = {'kind': 'empirical_oof_residual', 'residual_quantiles': [-40, -20, -1, 0, 1, 20, 70]} if empirical else {}
    lines = np.arange(0, mean*2+10, .5)
    outcomes = np.array([props._line_outcomes(stat, mean, line, {}, dist) for line in lines])
    assert np.all(np.diff(outcomes[:, 0]) <= 1e-10)
    assert np.all(np.diff(outcomes[:, 1]) >= -1e-10)
    assert np.allclose(outcomes.sum(axis=1), 1)
    assert np.all(outcomes >= -1e-12)
    assert np.all(outcomes[1::2, 2] == 0)


def test_td_any_head_preserves_monotonic_tail_and_zero_push():
    values = [props._line_outcomes('receiving_tds', .3, line, {'td_probability_accepted': True},
                                  row={'_td_any_probability': .2}) for line in [0, .5, 1, 1.5, 2]]
    assert values[0] == pytest.approx((.2, 0, .8))
    assert values[1] == pytest.approx((.2, .8, 0))
    assert all(b[0] <= a[0] for a, b in zip(values, values[1:]))


def test_integer_td_push_is_not_an_under_win():
    over, under, push = props._line_outcomes('passing_tds', 1.5, 1, {})
    assert over == pytest.approx(poisson.sf(1, 1.5))
    assert under == pytest.approx(poisson.pmf(0, 1.5))
    assert push == pytest.approx(poisson.pmf(1, 1.5))
    conditional = over/(1-push)
    assert props._ev_per_unit(conditional, -110, push) == pytest.approx(over*100/110-under)
    assert not props._price_has_positive_ev(.9, 110, 1)


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.setattr(props, '_offer_probability_calibration', lambda **kw: (kw['raw_p_over'], []))
    monkeypatch.setattr(props, '_apply_recent_probability_calibration', lambda p, **kw: (p, []))
    def build(line=1, over=110, under=-110):
        return props._candidate_from_offer({'position': 'QB'}, 'passing_tds', 1.5, 1.5, {},
            {'line': line, 'over_price': over, 'under_price': under, 'bookmaker_key': 'fanduel'})
    return build


def test_candidate_final_probability_ev_minimum_and_display_agree(candidate, monkeypatch):
    monkeypatch.setattr(props, '_exact_line_overlay', lambda artifact, **kw: (.7, .7, None, 'exact_line_model_blend_test'))
    row = candidate()
    assert row['probability'] == .7
    assert row['win_probability'] + row['loss_probability'] + row['push_probability'] == pytest.approx(1)
    assert row['over_probability'] + row['under_probability'] + row['push_probability'] == pytest.approx(1)
    assert row[row['side']+'_probability'] == pytest.approx(row['win_probability'])
    assert row['ev'] == pytest.approx(props._ev_per_unit(row['probability'], row['price'], row['push_probability']))
    assert props._price_has_positive_ev(row['probability'], row['minimum_american_price'], row['push_probability'])
    assert 'no push' in props._row_probability_summary(row) and 'Push=' in props._row_probability_summary(row)


def test_zero_ev_is_not_treated_as_missing(candidate, monkeypatch):
    monkeypatch.setattr(props, '_offer_probability_calibration', lambda **kw: (.5, []))
    row = candidate(line=.5, over=100, under=-110)
    assert row['side'] == 'over'
    assert row['ev'] == 0
    assert row['drift_guard_pass'] is False


def test_half_lines_unchanged_by_push_scaling(candidate):
    row = candidate(line=1.5)
    assert row['push_probability'] == 0
    assert row['win_probability'] == row['probability']


def test_calibration_cannot_invent_under_zero_td_wins():
    row = props._candidate_from_offer({'position': 'QB'}, 'passing_tds', 1.5, 1.5, {},
        {'line': 0, 'over_price': 110, 'under_price': -110, 'bookmaker_key': 'fanduel'})
    assert row['under_probability'] == pytest.approx(0)
    assert row['over_probability'] + row['push_probability'] == pytest.approx(1)
    assert row['tier'] == 'paper' and not row['drift_guard_pass']


def test_recent_calibration_inversion_is_detected_not_silently_rewritten():
    common = dict(stat='receiving_yards', side='over', book='fanduel', market_probability=.5,
                  projection=50, baseline=50, position='WR')
    context = props._calibration_context_key(stat='receiving_yards', position='WR', line=30.5, projection=50, baseline=50)
    calibration = {('exact_context', 'fanduel', context, 'over', props._probability_bucket(.60)):
                   {'rows': 40, 'avg_probability': .8, 'empirical_win_rate': .4, 'win_rate': .4}}
    probabilities = [props._apply_recent_probability_calibration(p, line=line, calibration=calibration, **common)[0]
                     for p, line in [(.6, 30.5), (.59, 40.5)]]
    records = []
    for i, (line, p) in enumerate(zip([30.5, 40.5], probabilities)):
        records.append(dict(id=i, game_id='g', player_id='p', stat='receiving_yards', book='fanduel', line=line,
            created_at_utc=datetime(2026, 9, 20, 16, tzinfo=timezone.utc),
            start_ts_utc=datetime(2026, 9, 20, 17, tzinfo=timezone.utc), offer_fetched_at='2026-09-20T15:55:00Z',
            forecast_payload=dict(line=line, over_probability=p, under_probability=1-p,
                prediction_context_cutoff_utc='2026-09-20T16:00:00Z', model_version='release')))
    report = monotonicity_violations(records)
    assert report['comparable_pairs'] == 1 and len(report['violations']) == 1
    assert records[0]['forecast_payload']['over_probability'] == probabilities[0]
    records[1]['forecast_payload']['prediction_context_cutoff_utc'] = '2026-09-20T15:59:00Z'
    assert monotonicity_violations(records)['comparable_pairs'] == 0
