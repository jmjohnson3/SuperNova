from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling import distribution_policy as dp
from nfl_pipeline.modeling import validate_distribution_policy as validation
from nfl_pipeline.modeling import predict_player_props as props


def calibration_frame():
    return pd.DataFrame([dict(key=f'{week}-{i}', market='receiving_yards', source='role_workload:test:combined',
        season=2026, week=week, actual=30+i, locked_at=f'2026-09-{week*7:02}T12:00Z',
        label_available_at=f'2026-09-{week*7+1:02}T12:00Z', curve=dp.normal_curve(45, 15),
        anchor=dict(line=44.5, probability=.5), line=44.5, side='over', outcome=float(30+i>44.5))
        for week in (1, 2, 3) for i in range(25)])


def test_regularized_blend_counts_player_games_not_offer_rows():
    f = calibration_frame()
    a = validation.fit_policy(f)
    b = validation.fit_policy(pd.concat([f, f, f], ignore_index=True))
    assert a['model_weight'] == b['model_weight']
    assert a['blend_regularization'] == pytest.approx(.01*50/75)
    assert a['blend_regularization'] == b['blend_regularization']
    assert a['tuning_player_games'] == 25
    assert a['calibration_player_games'] == 50
    assert set(a['blend_comparisons']) == {0., .25, .5, .75, 1.}
    assert a['ranking_support'] == .2


def test_outer_test_labels_cannot_choose_calibration_or_blend():
    f = calibration_frame()
    a, _ = validation.forward_evaluate(f.assign(production_probability=.5, market_probability=.5))
    f.loc[f.week==3, ['actual', 'outcome']] = [500, 1.]
    b, _ = validation.forward_evaluate(f.assign(production_probability=.5, market_probability=.5))
    assert a['folds'][-1]['policy'] == b['folds'][-1]['policy']
    assert a['scope'] == 'historical_development_only'


def test_source_requires_original_history_training_before_lock_and_supported_position(monkeypatch):
    artifact = dict(contract='c', stat='receiving_yards', training_end='2026-09-17', production_release='r',
                    bundle={'model': SimpleNamespace(position_prior={'WR': 4})})
    monkeypatch.setattr(dp, 'source_artifact', lambda *a: artifact)
    descriptor = dict(path='x', sha256='s', contract='c', stat='receiving_yards', training_end='2026-09-17')
    row = dict(position='WR', targets_avg_5=5, receiving_yards_avg_5=50)
    assert dp.workload_error(descriptor, row, 50, 'r', '2026-09-18T12:00Z') is None
    assert dp.workload_error(descriptor, row, 50, 'r', '2026-09-17T12:00Z') == 'workload_training_not_before_lock'
    assert dp.workload_error(descriptor, dict(row, position='QB'), 50, 'r', '2026-09-18T12:00Z') == 'workload_position_without_training_support'
    assert dp.workload_error(descriptor, row, None, 'r', '2026-09-18T12:00Z') == 'workload_missing_original_inputs'


def test_source_uses_raw_point_not_line_anchored_point(monkeypatch):
    seen = []
    class Model:
        def yardage_curve(self, frame, variant, projection):
            seen.append((variant, float(projection[0])))
            return np.array([[0, 40, 80]]), np.array([[.3, .4, .3]]), {}
    monkeypatch.setattr(dp, 'source_artifact', lambda *a: {'bundle': {'model': Model()}})
    c = dp.source_curve(dict(kind='role_workload', path='x', sha256='s', variant='combined'), {}, 63.)
    assert seen == [('combined', 63.)]
    assert dp.summary(c)['mean'] == 40.


def test_live_capture_keeps_existing_trial_and_does_not_activate_challenger(monkeypatch):
    raw = dp.normal_curve(40, 12)
    source = 'role_workload:test:combined'
    key = dp.policy_key('receiving_yards', 'release', source=source)
    policy = dict(contract=dp.CONTRACT, key=key, source=source, created_at='2026-01-01T00:00:00+00:00',
                  policy_id='new', a=1., b=1., model_weight=.5,
                  source_artifact=dict(kind='role_workload'))
    registry = {key: policy, 'unrelated_trial': {'key': 'unrelated_trial', 'source': 'original'}}
    before = deepcopy(registry)
    monkeypatch.setattr(dp, 'workload_error', lambda *a: None)
    monkeypatch.setattr(dp, 'source_curve', lambda *a: dp.normal_curve(60, 20))
    q = dict(bookmaker_key='fanduel', over_price=-110, under_price=-110, line=49.5)
    contexts = dp.prop_contexts(raw, 'receiving_yards', 'release', 'fanduel', [q], {}, registry, {},
                               projection=60, cutoff='2026-10-08T12:00Z')
    assert registry == before
    assert len(contexts) == 2
    assert all(c['deployed'] is None for c in contexts)
    assert dp.captured_score(contexts[1], 'challenger', 49.5)['policy_id'] == 'new'


def test_registered_stat_component_uses_the_exact_live_shadow_curve():
    raw = dp.normal_curve(40, 12)
    component = dp.normal_curve(62, 18)
    source = 'stat_component:oe-v2'
    key = dp.policy_key('receiving_yards', 'release', source=source)
    policy = dict(contract=dp.CONTRACT, key=key, source=source,
        created_at='2026-01-01T00:00:00+00:00', policy_id='pinned',
        a=1., b=1., model_weight=.25,
        source_artifact={'kind': 'stat_component', 'component_id': 'oe-v2'})
    q = dict(bookmaker_key='fanduel', over_price=-110, under_price=-110, line=49.5)
    contexts = dp.prop_contexts(
        raw, 'receiving_yards', 'release', 'fanduel', [q], {}, {key: policy}, {},
        projection=40, cutoff='2026-10-08T12:00Z', source_curves={source: component})
    assert len(contexts) == 2
    assert contexts[1]['curve'] == component
    assert contexts[1]['challenger']['policy_id'] == 'pinned'


@pytest.mark.parametrize('line', [39.5, 40., 40.5, 65.5])
def test_final_probability_drives_ev_price_and_discord_even_with_zero_legacy_trust(monkeypatch, line):
    monkeypatch.setattr(props, '_micro_projection_status', lambda **kw: ('micro_projection', 'test'))
    c = dp.curve([0, 40, 80, 120], [.1, .2, .5, .2])
    c = dp.score(c, dict(a=.9, b=1.1, model_weight=.75), dict(line=40.5, probability=.5))
    score = dict(dp.outcomes(c, line), summary=dp.summary(c), policy_id='own_calibration')
    policy = dict(policy_id='own_calibration', source='stat_component:oe', model_weight=.35)
    q = dict(line=line, over_price=110, under_price=-110, bookmaker_key='fanduel',
             over_link='https://sportsbook.fanduel.com/s', under_link='https://sportsbook.fanduel.com/s',
             distribution_policy_score=score,
             distribution_policy_capture={'deployed': policy},
             monotone_fanduel_no_vig_over_probability=100/210/(100/210+110/210),
             sharp_reference={'consensus_over_probability': .60, 'consensus_book_count': 1,
                              'consensus_dispersion': 0.0}, _execution_verified=True)
    m = dict(market_calibration={'stats': {'receiving_yards': {'probability_trust': 0., 'projection_trust': 0.}}})
    features = dict(position='WR', live_usage_context_quality_v4_score=.9,
        limited_workload_risk_score=.1, starter_role_stability_score=.8,
        same_week_injury_context_known=1., route_data_source='measured_history',
        _challenger_stat_component_id='oe', _challenger_stat_opportunity=7.,
        _challenger_stat_efficiency=8.)
    r = props._candidate_from_offer(features, 'receiving_yards', 62, 55, m, q)
    payout = r['price']/100 if r['price']>0 else -100/r['price']
    assert r['ev'] == pytest.approx(r['win_probability']*payout-r['loss_probability'])
    assert r['win_probability']+r['loss_probability']+r['push_probability'] == pytest.approx(1.)
    market_over = q['monotone_fanduel_no_vig_over_probability']
    assert market_over <= r['conditional_over_probability'] <= min(score['conditional_over'], market_over+.04)
    assert props._ev_per_unit(r['probability'], r['minimum_american_price'], r['push_probability']) > 0
    message = props._row_probability_summary(r)
    assert f"={r['probability']:.0%}" in message
    assert f"EV={r['ev']:+.1%}" in message
    assert r['tier'] == 'micro_projection'
    assert r['fanduel_signal']['approved'] is True


def test_conservative_ranking_never_mutates_probability_or_price():
    r = dict(probability=.6, price=-110, push_probability=.1, market_no_vig_probability=.5,
             ev=props._ev_per_unit(.6, -110, .1))
    before = dict(r)
    assert validation.conservative_rank_score(r, 0.) < validation.conservative_rank_score(r, .5) < r['ev']
    assert r == before


def test_no_quota_and_ranking_only_penalty_does_not_reprice_selected_pick(monkeypatch):
    def fake(*args, **kwargs):
        q = args[5]
        p = q['p']
        return dict(tier='micro_projection', forecast_policy_pre_cash_tier='micro_projection', side='over',
            probability=p, price=-110, ev=props._ev_per_unit(p, -110), push_probability=0.,
            market_no_vig_probability=.5, model_market_edge=p-.5, projection_confidence=.5,
            book='fanduel', link='https://sportsbook.fanduel.com/s', line=40.5, drift_guard_pass=True,
            game_id='g', player_id=q['player'], actionable_score=.3, execution_verified=True,
            fanduel_signal={'approved': True})
    monkeypatch.setattr(props, '_candidate_from_offer', fake)
    rows = [dict(id=i, market='receiving_yards', side='over', day='d', batch='b', actual=70., line=40.5,
        production_probability=.7, market_probability=.5, scored_curve=dp.normal_curve(60, 20),
        policy_ranking_support=0. if i else 1., captured=dict(row={}, stat='receiving_yards', projection=60,
        baseline=50, metrics={}, offer=dict(p=.7, player=i), distribution={}, min_ev=.02, calibration=[], clv_guards=[])) for i in range(8)]
    plain, _ = validation.unchanged_selection(rows, True)
    conservative, _ = validation.unchanged_selection(rows, True, True)
    assert len(plain) == 3
    assert len(conservative) == 1
    assert conservative[0]['selected_probability'] == .7
    assert conservative[0]['selected_ev'] == props._ev_per_unit(.7, -110)
    changed = deepcopy(rows)
    for r in changed:
        r['actual'] = 0.
    again, _ = validation.unchanged_selection(changed, True, True)
    assert [r['id'] for r in again] == [r['id'] for r in conservative]


def test_failed_replay_excludes_whole_batch(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('missing input')
    monkeypatch.setattr(props, '_candidate_from_offer', fail)
    r = dict(id=1, market='receiving_yards', day='d', batch='b', captured=dict(offer={}, row={}, stat='receiving_yards',
        projection=50, baseline=40, metrics={}, distribution={}, min_ev=.02, calibration=[], clv_guards=[]))
    selected, excluded = validation.unchanged_selection([r])
    assert selected == []
    assert excluded['incomplete_ranking_batch'] == 1


def test_selection_replay_uses_lock_clock_without_changing_live_clock(monkeypatch):
    from nfl_pipeline.modeling import scoring_versions
    fn = props._candidate_from_offer
    helper = props._game_has_started
    monkeypatch.setattr(scoring_versions, 'archived_candidate', lambda _: fn)
    monkeypatch.setattr(props, '_micro_projection_status', lambda **kw:
        ('paper', 'started') if kw['game_started'] else ('micro_projection', 'pregame'))
    quote = dict(line=40.5, over_price=110, under_price=-110, bookmaker_key='fanduel',
                 over_link='https://sportsbook.fanduel.com/s', under_link='https://sportsbook.fanduel.com/s')
    features = dict(position='WR', start_ts_utc='2010-01-01T20:00Z')
    cap = dict(scoring_fingerprint='saved', row=features, stat='receiving_yards', projection=60, baseline=40,
               metrics={}, offer=quote, distribution={}, min_ev=.02, calibration=[], clv_guards=[])
    r = validation.candidate_at_lock(dict(captured=cap, locked_at='2010-01-01T19:00Z'), quote)
    assert r['tier'] == 'micro_projection'
    assert props._game_has_started is helper
    assert props._game_has_started(features)
    late = validation.candidate_at_lock(dict(captured=cap, locked_at='2010-01-01T21:00Z'), quote)
    assert late['tier'] == 'paper'


def test_forecast_deployment_does_not_require_old_model_to_have_cash_selections():
    review = dict(scope='prospective_same_policy_same_offers', market='receiving_yards',
        challenger=dict(weeks=3, player_games=80, brier=.2, calibration_error=.02, distribution={'coverage_80': .8}),
        production=dict(brier=.25, calibration_error=.04), brier_gain={'lower_95': .01}, production_coverage=.78,
        production_ranked={'rows': 0}, challenger_ranked=dict(rows=25, brier=.2, calibration_error=.03),
        selected_comparison_scope='identical_challenger_selected_offers', selected_player_games=25,
        production_on_challenger_selected=dict(rows=25, brier=.26, calibration_error=.06),
        ranking_exclusions={}, unchanged_ranking=True, cash_approved=False)
    assert validation.deployment_blockers(review) == []
    assert review['cash_approved'] is False
    review['production_on_challenger_selected']['brier'] = .18
    assert 'selected_probabilities_worse' in validation.deployment_blockers(review)
