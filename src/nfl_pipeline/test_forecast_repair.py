from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import distribution_policy as dp
from nfl_pipeline.modeling import forecast_repair as repair
from nfl_pipeline.modeling import predict_today as games
from nfl_pipeline.modeling.role_workload import RoleWorkload, roles


def sample(n=90, stat='rushing_yards'):
    op = 'carries' if stat == 'rushing_yards' else 'pass_attempts'
    count = np.arange(n)%20+1
    f = pd.DataFrame(dict(game_id=np.arange(n), player_id=np.arange(n),
        game_date_et=pd.date_range('2024-01-01', periods=n).date,
        season=2024, week=np.arange(n)//7+1,
        position='RB' if stat == 'rushing_yards' else 'QB', n_games_prev_10=10,
        is_home=np.arange(n)%2, rest_days=7))
    f[op] = count; f[stat] = count*4+np.arange(n)%3
    for w in (5, 10):
        f[f'{op}_avg_{w}'] = count*.8
        f[f'{stat}_avg_{w}'] = count*3.5
    return f


def test_efficiency_is_separate_from_current_opportunity_and_outcome():
    f = sample()
    model = repair.StableEfficiency().fit(f, 'rushing_yards').tune(f)
    a = model.predict(f)
    changed = f.assign(carries=10000, rushing_yards=-50000, market_total=900,
        injury_status='out', home_score=999)
    np.testing.assert_array_equal(a, model.predict(changed))
    assert not any(c in model.columns for c in ('carries', 'rushing_yards', 'market_total', 'injury_status'))


def test_efficiency_prior_shrinks_small_sample_and_rejects_unknown_position():
    f = sample(); model = repair.StableEfficiency().fit(f, 'rushing_yards')
    low = f.iloc[:1].assign(carries_avg_10=.2, rushing_yards_avg_10=4., n_games_prev_10=1)
    high = low.assign(carries_avg_10=20., rushing_yards_avg_10=400., n_games_prev_10=10)
    assert abs(model.prior(low)[0]-model.priors['RB']) < abs(model.prior(high)[0]-model.priors['RB'])
    with pytest.raises(ValueError, match='without training support'):
        model.predict(low.assign(position='WR'))


def test_passing_low_volume_roles_and_zero_attempts_have_zero_yards():
    f = sample(2, 'passing_yards').assign(pass_attempts_avg_5=[10., 30.])
    assert roles(f, 'passing_yards').tolist() == ['lower', 'higher']
    m = RoleWorkload(); m.stat = 'passing_yards'; m.phase = 'tuned'; m.position_prior = {'QB': 25.}
    m.global_prior = 25.
    m.count_curve = lambda *a: (np.array([[0., 10.], [0., 30.]]), np.full((2, 2), .5))
    m.efficiency_tails = {'all': np.zeros(41)}
    v, w, parts = m.yardage_curve(f, projection=np.array([50., 150.]), rates=np.array([6., 7.]))
    np.testing.assert_array_equal(v[:, :41], 0.)
    np.testing.assert_allclose(w.sum(axis=1), 1.)
    np.testing.assert_allclose(parts['state_probabilities'].sum(axis=1), 1.)
    assert v[0, -1] == 60. and v[1, -1] == 210.


def test_total_features_are_small_lagged_symmetric_and_ignore_hindsight():
    f = pd.DataFrame({f'{side}_{name}_avg_{w}': [float(w)]
        for side in ('home', 'away') for name in ('pf', 'pa', 'plays', 'pass_rate', 'yards_per_play', 'red_zone_td_rate')
        for w in (5, 10)})
    f['home_rest_days'] = 7; f['away_rest_days'] = 6
    x = repair.total_inputs(f)
    assert len(x.columns) == 13
    changed = f.assign(total_points_actual=1000, market_total=-100, wind=999, home_score=0)
    pd.testing.assert_frame_equal(x, repair.total_inputs(changed))
    assert repair.total_inputs(pd.DataFrame(index=[0])).isna().all().all()


def test_cdf_tail_repair_is_coherent_at_every_line_and_keeps_pushes():
    c = dp.curve([-5, 0, 20, 70, 300, 600], [.01, .15, .2, .35, .25, .04])
    calibrated = dp.calibrate(c, dict(a=.9, b=1.2))
    masses = [dp.outcomes(calibrated, x) for x in range(-10, 610)]
    assert np.all(np.diff([r['over'] for r in masses]) <= 1e-12)
    assert all(r['over']+r['under']+r['push'] == pytest.approx(1.) for r in masses)
    assert dp.outcomes(calibrated, 0)['push'] > 0
    assert max(calibrated['values']) == 600


def test_repair_capture_is_time_checked_and_cannot_activate_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(dp, 'STORE', tmp_path)
    artifact = dict(contract=repair.CONTRACT, market='rushing_yards', training_end='2026-09-17',
        production_release='r', model=SimpleNamespace(rate=SimpleNamespace(priors={'RB': 4.})))
    monkeypatch.setattr(dp, 'source_artifact', lambda *a: artifact)
    d = dict(kind='forecast_repair', contract=repair.CONTRACT, path='x', sha256='s', stat='rushing_yards',
        variant='joint', training_end='2026-09-17', production_release='r')
    row = sample(1).iloc[0].to_dict()
    assert dp.repair_error(d, row, 50, 'r', '2026-09-18T12:00Z') is None
    assert dp.repair_error(d, row, 50, 'r', '2026-09-17T12:00Z') == 'repair_training_not_before_lock'
    assert dp.repair_error(d, dict(row, carries_avg_10=None), 50, 'r', '2026-09-18T12:00Z') == 'repair_missing_original_inputs'
    registered = dp.register_source(d)
    assert dp.register_source(d) == registered
    monkeypatch.setattr(dp, 'source_curve', lambda *a: dp.normal_curve(50, 25))
    q = dict(bookmaker_key='fanduel', line=49.5, over_price=-110, under_price=-110)
    contexts = dp.prop_contexts(dp.normal_curve(40, 10), 'rushing_yards', 'r', 'fanduel', [q], row, {}, {},
        projection=50, cutoff='2026-09-18T12:00Z')
    assert len(contexts) == 2
    assert contexts[1]['source_artifact'] == d
    assert all(c['deployed'] is None and c['challenger'] is None for c in contexts)
    assert not dp.load_registry('active')


def test_games_capture_source_curve_without_changing_probability(monkeypatch):
    monkeypatch.setattr(dp, 'load_registry', lambda *a: {})
    game = dict(bookmaker_key='fanduel', _model_release='r', quote_fetched_at_utc='2026-10-08T12:00Z')
    choices = [dict(line=44.5, price=-110, probability=.6, side='over'),
               dict(line=44.5, price=-110, probability=.4, side='under')]
    before = deepcopy(choices)
    games._distribution_policy_choices(choices, game, {}, 47., 14., 'total')
    assert [c['probability'] for c in choices] == [c['probability'] for c in before]
    assert choices[0]['distribution_policy_capture']['anchor']['probability'] == .5
    assert len(choices[0]['distribution_policy_variants']) == 1


def test_bad_original_game_inputs_remain_unknown(monkeypatch):
    artifact = dict(contract=repair.CONTRACT, market='total', training_end='2026-09-17', production_release='r',
        model=SimpleNamespace(columns=['pf_5']))
    monkeypatch.setattr(dp, 'source_artifact', lambda *a: artifact)
    d = dict(contract=repair.CONTRACT, stat='total', training_end='2026-09-17', path='x', sha256='s')
    assert dp.repair_error(d, {}, 45., 'r', '2026-10-08T12:00Z') == 'repair_missing_original_game_inputs'


def test_calibration_rejected_if_later_downside_or_tail_coverage_worsens():
    model = repair.PlayerRepair(); model.stat = 'rushing_yards'
    f = sample(30).assign(rushing_yards=50.)
    def curves(frame, variant):
        values = [0., 1., 2.] if variant == 'joint_calibrated' else [30., 50., 70.]
        return np.tile(values, (len(frame), 1)), np.tile([.1, .8, .1], (len(frame), 1))
    model.curve = curves
    result = model.gate_calibration(f)
    assert not result['accepted']
    assert result['scope'] == 'later_inner_gate_not_outer_test'
    assert result['scores']['joint_calibrated']['above'] == 1.


def test_dense_curve_gives_intermediate_yards_mass_but_keeps_zero_atom():
    c = repair.dense_yard_curve([0, 10, 20], [.2, .4, .4])
    assert dp.outcomes(c, 0)['push'] == pytest.approx(.2)
    assert dp.outcomes(c, 5)['push'] > 0
    assert sum(c['weights']) == pytest.approx(1.)


def test_forecast_deployment_is_not_blocked_by_absent_cash_picks():
    from nfl_pipeline.modeling.validate_distribution_policy import deployment_blockers, selection_blockers
    r = dict(scope='prospective_same_policy_same_offers', market='rushing_yards',
        challenger=dict(weeks=4, player_games=100, brier=.19, calibration_error=.01,
            distribution=dict(coverage_80=.8)), production=dict(brier=.22, calibration_error=.03),
        brier_gain=dict(lower_95=.002), production_coverage=.78,
        production_ranked=dict(rows=0), challenger_ranked=dict(rows=0),
        ranking_exclusions={}, unchanged_ranking=True, cash_approved=False)
    assert deployment_blockers(r) == []
    assert selection_blockers(r) == ['selected_pick_sample_missing']
    assert not r['cash_approved']


def test_failed_shadow_source_is_explicit_but_active_failure_cannot_fallback(monkeypatch):
    source = 'forecast_repair:x:joint'
    key = dp.policy_key('rushing_yards', 'r', source=source)
    entry = dict(key=key, source=source, source_artifact=dict(kind='forecast_repair'))
    monkeypatch.setattr(dp, 'load_registry', lambda *a: {key: entry})
    monkeypatch.setattr(dp, 'repair_error', lambda *a: 'repair_missing_original_inputs')
    contexts = dp.prop_contexts(dp.normal_curve(40, 15), 'rushing_yards', 'r', 'fanduel', [], {}, {}, {})
    assert len(contexts) == 1
    assert contexts[0]['source_exclusions'][source] == 'repair_missing_original_inputs'
    with pytest.raises(ValueError, match='repair_missing_original_inputs'):
        dp.prop_contexts(dp.normal_curve(40, 15), 'rushing_yards', 'r', 'fanduel', [], {}, {}, {key: entry})


def test_count_blend_is_fixed_before_efficiency_and_cdf_blocks():
    def block(offset):
        f = sample(50)
        f.game_id += offset; f.player_id += offset
        f.game_date_et = pd.date_range('2020-01-01', periods=offset+50).date[offset:]
        return f
    model = RoleWorkload().fit(block(0), 'rushing_yards').fit_count_tails(block(60)).fit_count_weight(block(120))
    alpha = model.alpha
    reference = SimpleNamespace(predict=lambda f: np.full(len(f), 40.))
    model.fit_efficiency_tails(block(180), reference).finish_count_first()
    assert model.alpha == alpha
    assert model.count_tuning_end < model.last_end
    assert model.phase == 'tuned'
    with pytest.raises(ValueError, match='phases must run once'):
        model.fit_count_weight(block(240))
