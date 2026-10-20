from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import distribution_policy as dp
from nfl_pipeline.modeling import validate_distribution_policy as validation


def test_count_mass_push_and_complement():
    c = dp.curve([0, 1, 2, 3], [.1, .2, .3, .4])
    o = dp.outcomes(c, 2)
    assert o['push'] == pytest.approx(.3)
    assert o['conditional_over'] == pytest.approx(.4/.7)
    assert sum(o[k] for k in ('over', 'under', 'push')) == pytest.approx(1)
    assert dp.outcomes(c, 2.5)['push'] == 0


@pytest.mark.parametrize('stat', dp.MARKETS[:7])
def test_every_prop_stat_has_a_valid_distribution(stat):
    c = dp.prop_curve(stat, 3., {}, {}, {})
    assert sum(c['weights']) == pytest.approx(1)
    assert all(w >= 0 for w in c['weights'])


@pytest.mark.parametrize('a,b,w', [(1, 1, 0), (.7, 1.4, .3), (2, .5, .7), (.8, .9, 1)])
def test_whole_cdf_is_monotonic_after_calibration_and_market_blend(a, b, w):
    c = dp.normal_curve(50, 23)
    anchor = dict(line=49.5, probability=.55)
    final = dp.score(c, dict(a=a, b=b, model_weight=w), anchor)
    over = [dp.outcomes(final, line)['over'] for line in np.arange(0, 120, .5)]
    assert np.max(np.diff(over)) <= 1e-12
    assert sum(final['weights']) == pytest.approx(1)
    if w == 0:
        assert dp.outcomes(final, anchor['line'])['conditional_over'] == pytest.approx(.55)


def test_anchor_supports_integer_push_without_converting_push_to_loss():
    c = dp.normal_curve(4, 2, nonnegative=True)
    market = dp.market_curve(c, dict(line=4., probability=.6))
    p = dp.outcomes(market, 4)
    assert p['push'] > 0
    assert p['conditional_over'] == pytest.approx(.6)


def test_synthetic_pair_cannot_anchor_and_anchor_is_order_independent():
    base = dict(bookmaker_key='fanduel', over_price=-110, under_price=-110)
    offers = [dict(base, line=50.5, offer_id=2), dict(base, line=49.5, offer_id=1),
              dict(base, line=50., offer_id=3, pair_quality='synthetic')]
    assert dp.anchor_for(offers, 50) == dp.anchor_for(offers[::-1], 50)
    assert dp.anchor_for(offers, 50)['line'] == 49.5
    assert dp.anchor_for([offers[-1]], 50) is None


def test_cdf_fit_does_not_count_duplicate_offers_as_independent_players():
    c = dp.normal_curve(25, 10)
    rows = [dict(key=str(i), curve=c, actual=i+10) for i in range(40)]
    assert dp.fit_cdf(rows) == dp.fit_cdf(rows+rows[:1]*300)


def test_registry_tampering_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, 'STORE', tmp_path)
    dp.save_registry({'a': {'x': 1}}, 'active')
    assert dp.load_registry('active')['a']['x'] == 1
    path = tmp_path/'active.json'
    path.write_text(path.read_text().replace('"x": 1', '"x": 2'))
    with pytest.raises(ValueError, match='checksum'):
        dp.load_registry('active')


def test_no_deployment_from_history_or_boolean_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, 'STORE', tmp_path)
    with pytest.raises(ValueError, match='deployment_rejected'):
        dp.deploy(dict(policy_id='fake', key='fake'), dict(passed=True))
    assert not (tmp_path/'active.json').exists()


def test_future_policy_cannot_enter_lock():
    key = dp.policy_key('receiving_yards', 'release')
    policy = dict(contract=dp.CONTRACT, key=key, created_at='2030-01-01T00:00:00+00:00')
    with pytest.raises(ValueError, match='future'):
        dp.context(dp.normal_curve(50, 20), None, key, {key: policy}, {}, '2026-10-07T00:00:00+00:00')


def test_calibration_does_not_inherit_legacy_zero_model_trust():
    c = dp.normal_curve(80, 20)
    own = dp.score(c, dict(a=1, b=1, model_weight=1), dict(line=49.5, probability=.5))
    assert dp.outcomes(own, 49.5)['conditional_over'] > .9


def test_game_choices_capture_both_curve_and_unchanged_production():
    from nfl_pipeline.modeling.predict_today import _best_total_candidate, _best_spread_candidate
    game = dict(home_team_abbr='A', away_team_abbr='B', bookmaker_key='fanduel', _model_release='r')
    quote = pd.Series(dict(total_points=44.5, total_over_price=-110, total_under_price=-110,
                           spread_home_points=-3, spread_away_points=3, spread_home_price=-110, spread_away_price=-110))
    total = _best_total_candidate(game, 48, 44, 12, quote)
    spread = _best_spread_candidate(game, 4, 3, 14, quote)
    assert total['distribution_policy_capture']['deployed'] is None
    assert total['distribution_challenger_score'] is None
    assert spread['game_outcome_distribution']['push'] > 0
    assert spread['game_outcome_policy_capture']['anchor'] is None


def test_legacy_game_calibration_write_is_disabled(monkeypatch):
    from nfl_pipeline.modeling.fit_game_market_calibration import main
    monkeypatch.setattr('sys.argv', ['fit_game_market_calibration', '--write'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_fit_uses_earlier_weeks_for_shape_and_later_week_for_blend(monkeypatch):
    frame = pd.DataFrame([dict(key=f'{week}-{i}', market='receiving_yards', season=2026, week=week,
        actual=30+i, locked_at=f'2026-09-{week*7:02}T12:00Z', label_available_at=f'2026-09-{week*7+1:02}T12:00Z',
        curve=dp.normal_curve(45, 15), anchor=dict(line=44.5, probability=.5), line=44.5,
        side='over', outcome=float(30+i > 44.5)) for week in (1, 2, 3) for i in range(25)])
    seen = []
    original = dp.fit_cdf
    def capture(rows):
        seen.append({r['week'] for r in rows})
        return original(rows)
    monkeypatch.setattr(dp, 'fit_cdf', capture)
    result = validation.fit_policy(frame)
    assert result['tuning_period'] == (2026, 3)
    assert seen == [{1, 2}]
    changed = frame.copy(); changed.loc[changed.week==3, 'actual'] = 1000
    validation.fit_policy(changed)
    assert seen[-1] == {1, 2}


def test_live_policy_survives_legacy_zero_trust_and_does_not_authorize_cash(monkeypatch):
    from nfl_pipeline.modeling import predict_player_props as props
    monkeypatch.setattr(props, '_micro_projection_status', lambda **kw: ('micro_projection', 'test'))
    c = dp.curve([0, 1, 2, 3], [.1, .2, .3, .4])
    score = dict(dp.outcomes(c, 2), summary=dp.summary(c), policy_id='pinned')
    metrics = dict(market_calibration={'stats': {'passing_tds': {'probability_trust': 0.}}})
    policy = dict(policy_id='pinned', source='raw_distribution', model_weight=.35)
    quote = dict(line=2., over_price=110, under_price=-110, bookmaker_key='fanduel',
                 distribution_policy_score=score, distribution_policy_capture={'deployed': policy},
                 monotone_fanduel_no_vig_over_probability=100/210,
                 sharp_reference={'consensus_over_probability': .62, 'consensus_book_count': 1,
                                  'consensus_dispersion': 0.0},
                 over_link='https://sportsbook.fanduel.com/s', _execution_verified=True)
    context = dict(position='QB', live_usage_context_quality_v4_score=.9,
        limited_workload_risk_score=.1, starter_role_stability_score=.8,
        same_week_injury_context_known=1., route_data_source='measured_history')
    r = props._candidate_from_offer(context, 'passing_tds', 1.5, 1.5, metrics, quote)
    assert 100/210 < r['conditional_over_probability'] < .4/.7
    assert r['push_probability'] == pytest.approx(.3)
    assert r['ev'] == pytest.approx(props._ev_per_unit(r['probability'], r['price'], .3))
    assert r['projection'] == 1.5
    assert r['calibrated_distribution_mean'] == pytest.approx(2.)
    assert r['tier'] == 'micro_projection'
    assert r['forecast_policy_pre_cash_tier'] == 'micro_projection'


def test_current_fit_can_use_observed_labels_without_faking_historical_availability():
    frame = pd.DataFrame([dict(key=f'{week}-{i}', market='game_outcome', season=2026, week=week,
        actual=i-12, locked_at=f'2026-09-{week*7:02}T12:00Z', label_available_at='2026-10-06T12:00Z',
        curve=dp.normal_curve(0, 15), anchor=None, line=0., side='home', outcome=float(i > 12))
        for week in (1, 2) for i in range(25)])
    assert validation.fit_policy(frame) is None
    fitted = validation.fit_policy(frame, datetime(2026, 10, 7, tzinfo=timezone.utc))
    assert fitted['fit_scope'] == 'current_time_development'
    assert fitted['model_weight'] == 1


def test_changed_outcomes_do_not_change_pick_ranking(monkeypatch):
    from nfl_pipeline.modeling import predict_player_props as props
    def fake(*args, **kwargs):
        offer = args[5]
        return dict(tier='micro_projection', side='over', book='fanduel', link='https://sportsbook.fanduel.com/s',
                    game_id='g', player_id=offer['player'], line=20.5, probability=.6, price=110,
                    drift_guard_pass=True, ev=offer['score'], model_market_edge=.1, projection_confidence=.5,
                    actionable_score=max(.001, offer['score']), execution_verified=True,
                    fanduel_signal={'approved': True})
    monkeypatch.setattr(props, '_candidate_from_offer', fake)
    rows = [dict(id=i, market='receiving_yards', day='2026-10-07', batch='batch', side='over',
        production_probability=.6, actual=50., line=20.5, captured=dict(row={}, stat='receiving_yards', projection=30,
        baseline=30, metrics={}, offer={'player': str(i), 'score': i/100}, distribution={}, min_ev=.02,
        calibration=[], clv_guards=[])) for i in range(8)]
    selected, errors = validation.unchanged_selection(rows)
    changed = deepcopy(rows)
    for r in changed:
        r['actual'] = 0.
    again, _ = validation.unchanged_selection(changed)
    assert {r['id'] for r in selected} == {r['id'] for r in again} == {5,6,7}
    assert not errors
