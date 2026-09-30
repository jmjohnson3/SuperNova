import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.challenger_models import player_features, curve_over, curve_summary
from nfl_pipeline.modeling.workload_repair import (
    WorkloadRepairModel, AsymmetricWorkloadResidual, repair_inputs)
from nfl_pipeline.test_workload_depth import prepared


@pytest.mark.parametrize('stat', ['receiving_yards', 'rushing_yards'])
def test_repair_states_are_bounded_and_offer_pool_independent(stat):
    frame = prepared()
    model = WorkloadRepairModel().fit(frame, stat)
    c = model.components(frame)
    np.testing.assert_allclose(c['weights'].sum(axis=1), 1)
    assert (np.diff(c['opportunities'], axis=1) >= 0).all()
    assert (c['opportunities'] <= c['team_volume'][:, None]).all()
    assert np.isfinite(c['centers']).all()
    np.testing.assert_allclose(c['centers'][:1], model.components(frame.iloc[:1])['centers'])
    poisoned = frame.copy()
    poisoned[['targets', 'carries', 'receiving_yards', 'rushing_yards']] = 9999.
    np.testing.assert_allclose(c['centers'], model.components(poisoned)['centers'])
    unavailable = frame.assign(wd_injury_out=1.)
    np.testing.assert_allclose(model.predict(unavailable), 0.)
    unknown = frame.assign(wd_injury_out=np.nan)
    np.testing.assert_allclose(model.predict(frame), model.predict(unknown))


def test_missing_and_duplicated_training_rows_are_not_silently_imputed():
    frame = prepared()
    with pytest.raises(ValueError, match='unique'):
        WorkloadRepairModel().fit(pd.concat([frame, frame.iloc[:1]]), 'rushing_yards')
    with pytest.raises(ValueError, match='Missing opportunity'):
        WorkloadRepairModel().fit(frame.assign(carries=np.nan), 'rushing_yards')


def test_features_use_prior_role_not_current_stats_or_price():
    frame = prepared().assign(price=900, actual_stat=100, projection=100)
    X = repair_inputs(frame, 'rushing_yards', 'rate')
    assert not {'price', 'actual_stat', 'projection', 'carries', 'rushing_yards'} & set(X)
    work = repair_inputs(frame, 'rushing_yards', 'workload')
    assert work.wd_injury_out__missing.iloc[0] == 1


def test_asymmetric_uncertainty_is_coherent_and_not_fixed_confidence():
    rng = np.random.default_rng(41)
    X = pd.DataFrame({'role': np.tile([0., 1.], 200)})
    errors = rng.normal(0, np.where(X.role == 0, 3, 25)) + rng.exponential(8, len(X))
    u = AsymmetricWorkloadResidual().fit_scale(X.iloc[:200], errors[:200], np.full(200, 40.), 2.)
    u.fit_residuals(X.iloc[200:], errors[200:])
    values, weights = u.mixture(X.iloc[:2], np.array([40., 40.]), None)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert not np.allclose(u.scales(X)[0], u.scales(X)[1])
    assert np.isfinite(u.confidence(X)).all()
    assert not np.allclose(u.confidence(X), .5)
    probabilities = np.array([curve_over(values, weights, [line, line]) for line in range(0, 101)])
    assert (np.diff(probabilities, axis=0) <= 1e-12).all()
    s = curve_summary(values, weights)
    assert (s['p10'] <= s['median']).all() and (s['median'] <= s['p90']).all()


def test_ablations_require_exactly_paired_lines():
    from nfl_pipeline.modeling.train_workload_repair import paired_gain
    rows = pd.DataFrame(dict(season=[2025, 2025], week=[1, 2], player_game=['a', 'b'],
        line=[10.5, 10.5], outcome=[1., 0.], calibrated_probability=[.5, .5]))
    better = rows.assign(calibrated_probability=[.8, .2])
    assert paired_gain(rows, better)['mean_gain'] > 0
    with pytest.raises(ValueError, match='identical'):
        paired_gain(rows, better.iloc[:1])


def test_brier_gain_cannot_justify_overwide_intervals_or_worse_calibration():
    from nfl_pipeline.modeling.train_workload_depth import calibration_curve_passes
    before = dict(brier=.25, calibration_error=.04, coverage_80=.78, interval_score=50.)
    good = dict(brier=.24, calibration_error=.03, coverage_80=.80, interval_score=49.)
    assert calibration_curve_passes(before, good)
    assert not calibration_curve_passes(before, dict(good, coverage_80=.95))
    assert not calibration_curve_passes(before, dict(good, interval_score=80.))
    assert not calibration_curve_passes(before, dict(good, calibration_error=.06))
