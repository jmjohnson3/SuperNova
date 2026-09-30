import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.challenger_models import curve_over, curve_summary
from nfl_pipeline.modeling.model_benchmark import (
    SimplePoint, raw_curve, select_outputs, frame_identity, complete_calibration_controls)
from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration


def enabled_calibrator():
    c = TailPreservingCalibration()
    c.knots = np.array([.1, .2, .5, .8, .9])
    c.values = np.array([.1, .14, .35, .7, .9])
    c.strength = .75
    c.enabled = True
    return c


def test_tail_identity_monotonicity_and_complete_curve():
    c = enabled_calibrator()
    p = np.linspace(0, 1, 1001)
    new = c.predict(p)
    assert np.all(np.diff(new) > 0)
    assert np.array_equal(new[(p <= .1) | (p >= .9)], p[(p <= .1) | (p >= .9)])
    v = np.array([np.arange(101), np.repeat(np.arange(51), 2)[:101]], dtype=float)
    w = np.full(v.shape, 1/101)
    cv, cw = c.transform(v[:, ::-1], w)
    assert np.all(cw >= 0)
    assert np.allclose(cw.sum(axis=1), 1)
    before, after = curve_summary(v, w), curve_summary(cv, cw)
    assert np.array_equal(before['p10'], after['p10'])
    assert np.array_equal(before['p90'], after['p90'])
    for line in (-1, 0, 10, 25, 42.5, 70, 101):
        raw = curve_over(v, w, [line, line])
        assert np.allclose(curve_over(cv, cw, [line, line]), c.predict(raw))


def test_single_atom_and_invalid_distributions():
    c = enabled_calibrator()
    v, w = c.transform([[0., 0.]], [[.25, .75]])
    assert curve_over(v, w, [0])[0] == 0
    with pytest.raises(ValueError):
        c.transform([[1]], [[-1]])
    with pytest.raises(ValueError):
        c.predict([np.nan])


def cal_frame(week, outcome=.7):
    rng = np.random.default_rng(42)
    p = np.linspace(.2, .8, 1000)
    return pd.DataFrame(dict(season=2025, week=week, player_game=np.arange(len(p)),
        probability=p, outcome=(rng.random(len(p)) < p*outcome).astype(int), weight=1.))


def test_calibration_rejects_overlapping_or_future_blocks():
    with pytest.raises(ValueError):
        TailPreservingCalibration().fit(cal_frame(1), cal_frame(1), cal_frame(3))
    with pytest.raises(ValueError):
        TailPreservingCalibration().fit(cal_frame(3), cal_frame(2), cal_frame(1))


def test_later_gate_can_disable_good_fit():
    c = TailPreservingCalibration().fit(cal_frame(1), cal_frame(2), cal_frame(3, 1.5))
    assert not c.enabled
    assert np.array_equal(c.predict([.2, .5, .8]), [.2, .5, .8])


def test_player_game_identity_is_order_independent():
    f = pd.DataFrame(dict(season=2025, week=np.arange(1, 21), game_id=np.arange(20),
                         player_id='x', game_date_et=pd.date_range('2025-01-01', periods=20, freq='7D')))
    assert frame_identity(f) == frame_identity(f.iloc[::-1])


def test_output_selection_is_separate_and_bad_tails_not_approved():
    scores = {
        'mean': dict(expected=dict(rmse=2, bias=0, coverage_80=.6), typical=dict(mae=3), probability=dict(brier=.1, calibration_error=.01)),
        'median': dict(expected=dict(rmse=4, bias=0, coverage_80=.95), typical=dict(mae=1), probability=dict(brier=.15, calibration_error=.02)),
        'probability': dict(expected=dict(rmse=3, bias=0, coverage_80=.8), typical=dict(mae=2), probability=dict(brier=.2, calibration_error=.03))}
    assert select_outputs(scores) == dict(expected='mean', typical='median', probability='probability', probability_fallback=None)
    del scores['probability']
    assert select_outputs(scores)['probability'] is None


def test_ensemble_is_mixture_not_mean_only():
    frame = pd.DataFrame(dict(position=['WR'], receiving_yards_avg_5=[10]))
    point = SimplePoint('receiving_yards', 'rolling')
    low = dict(kind='pooled', model=point, errors=np.array([-10., -8.]))
    high = dict(kind='pooled', model=point, errors=np.array([90., 110.]))
    v, w = raw_curve(dict(kind='ensemble', members=[low, high]), frame, 'receiving_yards')
    assert curve_over(v, w, [50])[0] == .5
    assert curve_summary(v, w)['p90'][0] == 120


def test_rejected_calibration_does_not_drop_outer_rows():
    raw = pd.DataFrame(dict(fold=['a', 'b', 'b'], family=['reference', 'reference', 'reference+tail_cal'],
                            player_id=[1, 2, 2], probability=[.3, .4, .35]))
    rows, lines = complete_calibration_controls(raw, raw)
    assert rows.groupby('family').size().to_dict() == {'reference': 2, 'reference+tail_cal': 2}
    assert rows.loc[rows.fold.eq('a') & rows.family.eq('reference+tail_cal'), 'probability'].iloc[0] == .3
    pd.testing.assert_frame_equal(rows, lines)
