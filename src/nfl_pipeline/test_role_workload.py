import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.challenger_models import curve_over
from nfl_pipeline.modeling.role_workload import RoleWorkload, VARIANTS, features, roles, count_crps, verified_context
from nfl_pipeline.modeling.train_role_workload import decisions, summarize


class Reference:
    def predict(self, frame):
        return frame.targets_avg_5.to_numpy()*7.


def sample(start='2023-01-01', n=30):
    i = np.arange(n)
    count = (i % 12).astype(float)
    return pd.DataFrame(dict(game_id=[start+'-'+str(k) for k in i], player_id='p',
        game_date_et=pd.date_range(start, periods=n).date, position=np.where(i % 2, 'WR', 'RB'),
        targets=count, receiving_yards=count*(4+i % 7), targets_avg_5=np.where(i % 2, 6., 2.),
        targets_avg_10=4., targets_avg_3=5., targets_std_5=3., carries=count,
        rushing_yards=count*4., carries_avg_5=np.where(i % 2, 12., 3.)))


def fitted(stat='receiving_yards'):
    return (RoleWorkload().fit(sample(), stat).fit_count_tails(sample('2023-03-01'))
        .fit_efficiency_tails(sample('2023-05-01'), Reference()).tune(sample('2023-07-01')))


@pytest.mark.parametrize('stat', ['receiving_yards', 'rushing_yards'])
def test_features_ignore_outcomes_prices_and_unverified_context(stat):
    frame = sample()
    changed = frame.copy()
    for column in ('targets', 'carries', 'receiving_yards', 'rushing_yards', 'routes_run', 'injury_status',
                   'actual', 'line', 'price', 'team_actual_pass_attempts'):
        changed[column] = 999.
    changed['target_role_evidence'] = [{'injury_status': 'out', 'expected_starter': False}]*len(frame)
    pd.testing.assert_frame_equal(features(frame, stat), features(changed, stat))
    assert features(frame, stat).role_injury_status_out.isna().all()


def test_context_rejects_after_lock_sources():
    from nfl_pipeline.modeling.role_context_training import CONTRACT
    context = dict(contract=CONTRACT, cutoff='2026-09-24T20:00:00Z', status='asof_observed',
        injury_status='out', provenance={'nfl_injuries': {'observed_at': '2026-09-24T21:00:00Z'}})
    assert verified_context(context) == {}
    context['provenance']['nfl_injuries']['observed_at'] = '2026-09-24T19:00:00Z'
    assert verified_context(context)['injury_status'] == 'out'


def test_training_identity_missing_targets_and_chronology():
    frame = sample()
    with pytest.raises(ValueError, match='unique player-games'):
        RoleWorkload().fit(pd.concat([frame, frame.iloc[:1]]))
    bad = frame.copy()
    bad.loc[0, 'targets'] = np.nan
    with pytest.raises(ValueError, match='observed counts'):
        RoleWorkload().fit(bad)
    model = RoleWorkload().fit(frame)
    with pytest.raises(ValueError, match='later'):
        model.fit_count_tails(frame)
    later = sample('2023-03-01')
    later.loc[0, 'game_id'] = frame.game_id.iloc[0]
    with pytest.raises(ValueError, match='overlapping games'):
        model.fit_count_tails(later)
    model.fit_count_tails(sample('2023-03-01'))
    with pytest.raises(ValueError, match='once in order'):
        model.fit_count_tails(sample('2023-05-01'))


@pytest.mark.parametrize('stat', ['receiving_yards', 'rushing_yards'])
def test_workload_count_mass_states_and_fixed_efficiency(stat):
    model = fitted(stat)
    later = sample('2023-09-01')
    expected_rate = None
    for variant in VARIANTS:
        v, w, c = model.yardage_curve(later, Reference(), variant)
        assert np.isfinite(v).all() and (w >= 0).all()
        np.testing.assert_allclose(w.sum(axis=1), 1.)
        np.testing.assert_allclose(c['state_probabilities'].sum(axis=1), 1.)
        if expected_rate is None:
            expected_rate = c['fixed_rate']
        np.testing.assert_array_equal(c['fixed_rate'], expected_rate)
        np.testing.assert_allclose((v*w).sum(axis=1), c['opportunity']*c['fixed_rate'], atol=1e-9)
        over = np.array([curve_over(v, w, np.full(len(v), line)) for line in (0., 10., 20., 40., 80.)])
        assert (np.diff(over, axis=0) <= 1e-12).all()
        for line in (0., 10., 20.5):
            np.testing.assert_allclose(((v > line)*w).sum(axis=1)+((v < line)*w).sum(axis=1)+((v == line)*w).sum(axis=1), 1.)
    counts, weights = model.count_curve(later, True, True)
    assert (counts >= 0).all() and (counts == np.rint(counts)).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    for tail in model.efficiency_tails.values():
        assert abs(tail.mean()) < 1e-10


def test_prediction_independent_of_later_outcomes_and_offered_pool():
    model = fitted()
    frame = sample('2023-09-01')
    a = model.yardage_curve(frame, Reference())
    changed = frame.copy()
    changed['targets'] = 100.
    changed['receiving_yards'] = 1000.
    b = model.yardage_curve(changed, Reference())
    np.testing.assert_array_equal(a[0], b[0])
    single = model.yardage_curve(frame.iloc[:1], Reference())
    np.testing.assert_array_equal(a[0][0], single[0][0])
    np.testing.assert_array_equal(a[1][0], single[1][0])
    assert roles(frame, 'receiving_yards').nunique() == 2
    frame['targets_avg_5'] = np.nan
    assert roles(frame, 'receiving_yards').eq('unknown').all()
    frame['position'] = 'QB'
    with pytest.raises(ValueError, match='no workload training support'):
        model.yardage_curve(frame, Reference())


def test_zero_opportunity_has_zero_yards_and_tails_are_role_specific():
    model = fitted()
    model.alpha = 0.
    model.count_residuals = {k: np.zeros_like(v) for k, v in model.count_residuals.items()}
    frame = sample('2023-09-01')
    frame['targets_avg_5'] = 0.
    v, w, c = model.yardage_curve(frame, Reference(), 'control')
    assert (v == 0).all() and (c['opportunity'] == 0).all()
    assert not np.allclose(model.efficiency_tails['higher'], model.efficiency_tails['lower'])


def test_count_crps_matches_pairwise_definition():
    v = np.array([[0., 2., 7.]])
    w = np.array([[.3, .5, .2]])
    y = np.array([3.])
    expected = (w*abs(v-y[:, None])).sum()- .5*np.sum(w[0, :, None]*w[0, None, :]*abs(v[0, :, None]-v[0, None, :]))
    np.testing.assert_allclose(count_crps(v, w, y), expected)


def test_point_acceptance_does_not_require_probability_acceptance():
    rows, lines = [], []
    for variant in ('reference', *VARIANTS):
        for i in range(24):
            better = variant == 'combined'
            rows.append(dict(game_id=str(i), player_id='p', season=2025, week=i//4,
                actual=20., actual_opportunity=4., opportunity=4. if better else 5.,
                mean=20. if better else 25., median=20. if better else 25., p10=0., p90=50., variant=variant))
            lines.append(dict(player_game=str(i), season=2025, week=i//4, variant=variant,
                line=20.5, outcome=i % 2, probability=.99 if better else .5, weight=1.))
    r, l = pd.DataFrame(rows), pd.DataFrame(lines)
    decision = decisions(r, l, summarize(r, l))['combined']
    assert decision['expected_yards']['historical_screen_passed']
    assert not decision['probability']['historical_screen_passed']
    assert not decision['expected_yards']['deployment_approved']
