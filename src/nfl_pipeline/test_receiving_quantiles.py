import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.challenger_models import curve_over, curve_summary
from nfl_pipeline.modeling.receiving_quantiles import ReceivingQuantiles, features, blend, QUANTILES
from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration
from nfl_pipeline.modeling.train_receiving_quantiles import compare
from nfl_pipeline.modeling import receiving_quantile_validation as validation


def sample(n=30, start='2024-01-01'):
    i = np.arange(n)
    return pd.DataFrame(dict(game_id=[start+'-'+str(k) for k in i], player_id='p',
        game_date_et=pd.date_range(start, periods=n).date, position=np.where(i % 2, 'WR', 'RB'),
        receiving_yards=(i % 9)*10., targets_avg_5=4., targets_avg_10=5., targets_avg_3=3.,
        receiving_yards_avg_5=30., receiving_yards_avg_10=40., targets_std_5=2.))


def test_no_current_outcomes_or_odds_in_features():
    frame = sample()
    changed = frame.copy()
    for key in ('targets', 'receiving_yards', 'actual', 'line', 'price', 'probability',
                'injury_status', 'routes_run', 'receiving_air_yards'):
        changed[key] = 999.
    pd.testing.assert_frame_equal(features(frame), features(changed))
    model = ReceivingQuantiles().fit(frame)
    np.testing.assert_array_equal(model.curve(frame)[0], model.curve(changed)[0])
    assert features(frame)['receiving_air_yards_avg_5__missing'].eq(1).all()


def test_sparse_fit_uses_quantiles_not_mean():
    frame = sample()
    model = ReceivingQuantiles().fit(frame)
    np.testing.assert_allclose(model.raw(frame)[0], np.quantile(frame.receiving_yards, QUANTILES))
    assert model.raw(frame)[0, 0] < model.raw(frame)[0, -1]


def test_duplicate_missing_and_nonchronological_correction_rejected():
    frame = sample()
    with pytest.raises(ValueError, match='unique player-games'):
        ReceivingQuantiles().fit(pd.concat([frame, frame.iloc[:1]]))
    missing = frame.copy()
    missing.loc[0, 'receiving_yards'] = np.nan
    with pytest.raises(ValueError, match='observed receiving'):
        ReceivingQuantiles().fit(missing)
    model = ReceivingQuantiles().fit(frame)
    with pytest.raises(ValueError, match='follow model fitting'):
        model.correct(frame)
    later = sample(start='2024-03-01')
    later.loc[0, 'game_id'] = frame.game_id.iloc[0]
    with pytest.raises(ValueError, match='overlapping games'):
        model.correct(later)


def test_calibrated_curve_is_coherent_at_all_lines_and_pushes():
    model = ReceivingQuantiles().fit(sample()).correct(sample(start='2024-03-01'))
    v, w = model.curve(sample(start='2024-05-01'))
    calibration = TailPreservingCalibration()
    calibration.knots = np.array([.1, .4, .7, .9])
    calibration.values = np.array([.1, .3, .65, .9])
    calibration.strength = .5
    calibration.enabled = True
    v, w = calibration.transform(*blend((v, w), (v-2, w)))
    assert np.isfinite(v).all() and (w >= 0).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.)
    probabilities = np.array([curve_over(v, w, np.full(len(v), line)) for line in np.arange(-5, 150, .5)])
    assert (np.diff(probabilities, axis=0) <= 1e-12).all()
    for line in (0., 10., 20.5, 70.5):
        over = ((v > line)*w).sum(axis=1)
        under = ((v < line)*w).sum(axis=1)
        push = ((v == line)*w).sum(axis=1)
        np.testing.assert_allclose(over+under+push, 1.)
    summary = curve_summary(v, w)
    assert (summary['p10'] <= summary['median']).all()
    assert (summary['median'] <= summary['p90']).all()


def test_later_outcomes_do_not_change_fixed_forecast():
    model = ReceivingQuantiles().fit(sample()).correct(sample(start='2024-03-01'))
    later = sample(start='2024-05-01')
    v, _ = model.curve(later)
    later.receiving_yards = -500.
    np.testing.assert_array_equal(v, model.curve(later)[0])
    with pytest.raises(ValueError, match='fraction'):
        blend((v, np.ones_like(v)), (v, np.ones_like(v)), 2.)


def test_point_screen_does_not_inherit_probability_or_cash_decision():
    rows, lines = [], []
    for name in ('reference', 'conditional', 'quantile', 'blend', 'calibrated_blend'):
        for i in range(30):
            better = name in ('quantile', 'blend', 'calibrated_blend')
            value = 20. if better else 25.
            rows.append(dict(family=name, game_id=str(i), player_id='p', season=2025, week=i//5,
                actual=20., mean=value, median=value, p10=value-5, p90=value+5))
            lines.append(dict(family=name, player_game=str(i), season=2025, week=i//5,
                line=20.5, outcome=i % 2, probability=.99 if better else .5, weight=1.))
    _, _, decisions = compare(pd.DataFrame(rows), pd.DataFrame(lines))
    result = decisions['quantile']
    assert result['expected_yards']['historical_screen_passed']
    assert result['typical_yards']['historical_screen_passed']
    assert not result['probability']['historical_screen_passed']
    assert not result['expected_yards']['deployment_approved']
    assert not result['expected_yards']['betting_approved']


def test_replay_requires_training_before_lock_and_archived_history(monkeypatch):
    monkeypatch.setattr(validation, 'locked_market', lambda record: {'market_probability': .5})
    artifact = dict(production_release='release', training_end='2026-09-17')
    record = dict(forecast_payload=dict(model_version='release', line=40.5,
        prediction_context_cutoff_utc='2026-09-24T20:00:00Z', scoring_replay=dict(projection=45.,
            row={'targets_avg_5': 6., 'receiving_yards_avg_5': 40.},
            offer={'fetched_at_utc': '2026-09-24T19:59:00Z'})))
    assert validation.eligible_error(record, artifact) is None
    assert validation.eligible_error(record, dict(artifact, training_end='2026-09-24')) == 'training_not_before_lock'
    record['forecast_payload']['scoring_replay']['row']['targets_avg_5'] = None
    assert validation.eligible_error(record, artifact) == 'missing_archived_player_history'


def test_selected_probabilities_are_evaluated_separately_without_changing_ids():
    rows = []
    for i in range(6):
        row = dict(model_version='release', scoring_version='scoring', prediction_id=i,
            game_id=str(i), player_id='p', season=2026, week=3+i//3, day='2026-09-24',
            batch='2026-09-24T20:00:00Z', actual=10., outcome=0., probability=.5, market=.5,
            exact_micro=i == 0, eligible=True, payout=2., push_probability=0., raw_projection=20.,
            projection=20., p10=0., p90=50.)
        for variant in validation.VARIANTS:
            row.update({variant+'_raw_probability': .9 if i == 0 else .1,
                variant+'_probability': .8 if i == 0 else .2,
                variant+'_mean': 15., variant+'_median': 10., variant+'_p10': 0., variant+'_p90': 30.})
        rows.append(row)
    result = validation.evaluate(pd.DataFrame(rows), {0})
    metrics = result['cohorts']['release|scoring']
    assert metrics['all_offers']['probabilities']['quantile_final']['brier'] < .25
    assert metrics['exact_micro']['probabilities']['quantile_final']['brier'] > .25
    assert metrics['fixed_research']['settled'] == 1
    assert not result['deployment_approved'] and not result['betting_approved']


def test_quantile_point_capture_is_pinned_independent_and_pregame(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import joblib
    import json
    from nfl_pipeline.modeling import target_point_capture as capture, train_receiving_quantiles as trainer
    from nfl_pipeline.modeling.receiving_quantiles import CONTRACT
    monkeypatch.setattr(capture, 'active_release', lambda: {'release_id': 'production'})
    artifact = dict(contract=CONTRACT, production_release='production', run_id='quantile-1',
        training_end='2026-09-17', bundle={}, output_decisions={'calibrated_blend': {
            'typical_yards': {'historical_screen_passed': True},
            'probability': {'historical_screen_passed': False}}})
    path = tmp_path/'model.joblib'
    joblib.dump(artifact, path)
    registration = capture.register(path, variant='calibrated_blend', store=tmp_path)
    assert not registration['probability_enabled'] and not registration['betting_approved']
    monkeypatch.setattr(trainer, 'curves', lambda *a: {
        'calibrated_blend': (np.array([[10., 30., 70.]]), np.array([[.25, .5, .25]]))})
    row = dict(game_id='g', player_id='p', stat='receiving_yards', model_version='production',
        season=2099, week=4, projection=45., projection_p50=45., probability=.55,
        prediction_context_cutoff_utc='2099-10-01T20:00:00Z', scoring_replay={
            'projection': 60., 'scoring_fingerprint': 'stable-scoring',
            'row': {'start_ts_utc': '2099-10-01T23:00:00Z', 'position': 'WR',
                    'targets_avg_5': 5., 'receiving_yards_avg_5': 55.},
            'distribution': {'kind': 'empirical_oof_residual', 'residual_quantiles': [-20, -10, 20]}})
    now = datetime(2099, 10, 1, 20, 1, tzinfo=timezone.utc)
    assert capture.capture_records([(1, row)], store=tmp_path, now=now)['captured'] == 1
    doc = json.loads(next((tmp_path/'point_prospective').glob('*.json')).read_text())
    assert doc['rows'][0]['reference'] == 50.  # Raw model median, not the line-anchored 45.
    assert doc['rows'][0]['candidate'] == 30.
    assert 'probability' not in doc['rows'][0] and 'actual' not in doc['rows'][0]
    assert row['projection'] == 45. and row['probability'] == .55
    assert capture.capture_records([(1, row)], store=tmp_path,
        now=datetime(2099, 10, 2, tzinfo=timezone.utc))['captured'] == 0
    artifact['run_id'] = 'different'
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match='already pinned'):
        capture.register(path, variant='calibrated_blend', store=tmp_path)
