from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline import forecast_outputs
from nfl_pipeline.modeling.target_components import curves, output_decisions, VARIANTS
from nfl_pipeline.modeling.target_volume_validation import shifted_probability
from nfl_pipeline.test_target_volume import fixture, Reference
from nfl_pipeline.modeling.target_volume import TargetVolume


def bundle():
    data = fixture()
    model = TargetVolume().fit(data).fit_residuals(data.iloc[:100], Reference())
    model.alpha = .5
    return dict(model=model, reference=Reference(), point_alpha=.5, reference_errors=np.array([-10., 0., 20.])), data.iloc[:4]


def test_factorial_experiment_isolates_center_and_uncertainty():
    b, frame = bundle()
    result = curves(b, frame)
    ref, rw, baseline = result['reference']
    point, pw, targets = result['targets_only']
    shape, sw, unchanged = result['uncertainty_only']
    both, bw, _ = result['combined']
    np.testing.assert_array_equal(rw, pw)
    np.testing.assert_allclose(np.diff(point, axis=1), np.diff(ref, axis=1))
    np.testing.assert_array_equal(unchanged, baseline)
    np.testing.assert_allclose((shape*sw).sum(axis=1), (ref*rw).sum(axis=1))
    np.testing.assert_allclose((both*bw).sum(axis=1), (point*pw).sum(axis=1))
    np.testing.assert_allclose((point-ref)[:, 0], (targets-baseline)*8)
    changed = frame.copy()
    changed[['targets', 'receiving_yards']] = 999
    np.testing.assert_array_equal(curves(b, changed)['targets_only'][0], point)


def test_point_alpha_zero_preserves_reference_exactly():
    b, frame = bundle(); b['point_alpha'] = 0.
    result = curves(b, frame)
    np.testing.assert_array_equal(result['targets_only'][0], result['reference'][0])
    np.testing.assert_allclose(result['combined'][0], result['uncertainty_only'][0])


def test_unknown_original_uncertainty_is_not_reconstructed():
    b, frame = bundle()
    with pytest.raises(ValueError, match='uncertainty'):
        curves(b, frame, residuals=[])
    with pytest.raises(ValueError, match='positive target'):
        curves(b, frame, baseline_targets=[0]*len(frame))


def test_projection_pass_does_not_require_probability_or_cash_pass():
    rows = []; lines = []
    for variant in VARIANTS:
        for i in range(80):
            actual = float(i % 20+30)
            center = actual+3 if variant in ('reference', 'uncertainty_only') else actual+.5
            rows.append(dict(variant=variant, game_id=str(i), player_id='p', season=2025, week=i//10+1,
                actual=actual, mean=center, median=center, p10=center-5, p90=center+5,
                actual_targets=5., predicted_targets=5.1, control_targets=6., long_targets=6., share_targets=6.))
            lines.append(dict(variant=variant, player_game=str(i), season=2025, week=i//10+1,
                line=35.5, outcome=float(i % 2), probability=.5 if variant=='reference' else .99, weight=1.))
    decisions = output_decisions(pd.DataFrame(rows), pd.DataFrame(lines))
    assert decisions['targets_only']['expected_mean']['historical_screen_passed']
    assert not decisions['targets_only']['probability']['historical_screen_passed']
    assert not decisions['targets_only']['expected_mean']['deployment_approved']
    assert not decisions['targets_only']['expected_mean']['betting_approved']


def test_shift_replays_frozen_uncertainty_without_changing_price_or_calibration(monkeypatch):
    from nfl_pipeline.modeling import target_volume_validation as validation
    payload = dict(line=45.5, side='over', probability=.55,
        scoring_replay=dict(projection=50., distribution={'kind':'empirical_oof_residual','residual_quantiles':[-20.,0.,40.]},
            offer={'over_price':-110}, calibration=['original']))
    original = deepcopy(payload); calls = []
    def replay(captured):
        calls.append(deepcopy(captured))
        return dict(side='over', probability=.55)
    monkeypatch.setattr(validation, 'replay', replay)
    shifted_probability(payload, 3)
    assert calls[-1]['projection'] == 53.
    assert calls[-1]['distribution'] == original['scoring_replay']['distribution']
    assert calls[-1]['offer'] == original['scoring_replay']['offer']
    assert payload == original


def forecast():
    return dict(game_id='g', player_id='p', stat='receiving_yards', model_version='production',
        prediction_context_cutoff_utc='2026-10-01T20:00:00Z', projection=50.,
        projection_semantics='central_forecast_not_certified_mean', line=45.5, side='over', price=-110,
        probability=.57, ev=.088, tier='paper', minimum_american_price=-120)


def approval():
    return dict(output='expected_yards', model_version='component-1', production_release='production',
        historical_screen_passed=True, prospective_validation_passed=True, deployment_approved=True,
        approved_at='2026-09-30T12:00:00Z', evidence_sha256='test-evidence', betting_approved=False)


def test_point_output_deployment_does_not_change_bet_pricing_or_tier():
    row = forecast(); before = deepcopy(row)
    new = forecast_outputs.apply_point_forecast(row, output='expected_yards', value=54.,
        model_version='component-1', decision=approval())
    assert row == before
    assert {k:new[k] for k in row} == before
    assert new['forecast_outputs']['expected_yards']['value'] == 54.
    assert new['forecast_outputs']['pricing']['projection_anchor'] == 50.
    assert 'Expected yards=54.0' in forecast_outputs.display_suffix(new)
    assert '[component-1; point-only]' in forecast_outputs.display_suffix(new)


@pytest.mark.parametrize('changes', [dict(prospective_validation_passed=False), dict(output='probability'),
    dict(production_release='different'), dict(approved_at='2026-10-02T00:00:00Z')])
def test_historical_or_wrong_output_approval_cannot_deploy_point(changes):
    with pytest.raises(ValueError, match='deployment decision'):
        forecast_outputs.apply_point_forecast(forecast(), output='expected_yards', value=54.,
            model_version='component-1', decision=dict(approval(), **changes))


def test_lock_identity_refreshes_price_without_relabeling_legacy_projection():
    row = forecast(); row['forecast_outputs'] = forecast_outputs.describe(row)
    row['price'] = 105
    outputs = forecast_outputs.for_lock(row)
    assert outputs['pricing']['price'] == 105
    assert outputs['expected_yards'] is None
    assert outputs['point_forecast']['semantics'] == 'central_forecast_not_certified_mean'


def test_point_forecast_cannot_transfer_to_another_player_or_lock():
    row = forecast_outputs.apply_point_forecast(forecast(), output='expected_yards', value=54.,
        model_version='component-1', decision=approval())
    row['player_id'] = 'other'
    with pytest.raises(ValueError, match='forecast context'):
        forecast_outputs.for_lock(row)


def test_median_capture_is_independent_of_failed_probability_screen(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import joblib
    import json
    from nfl_pipeline.modeling import target_point_capture as capture
    monkeypatch.setattr(capture, 'active_release', lambda: {'release_id':'production'})
    artifact = dict(contract='nfl-target-components-v1', production_release='production', run_id='component-1',
        training_end='2026-09-18', prospective_acceptance_start='2099-10-01',
        bundle={'model':SimpleNamespace(columns=[])},
        output_decisions={'uncertainty_only':{'median':{'historical_screen_passed':True},
                                             'probability':{'historical_screen_passed':False}}})
    path = tmp_path/'model.joblib'; joblib.dump(artifact,path)
    capture.register(path, store=tmp_path)
    monkeypatch.setattr(capture, 'curves', lambda *a,**k: {
        'uncertainty_only':(np.array([[20.,30.,40.]]),np.array([[.25,.5,.25]]),np.array([5.]))})
    row = dict(forecast(), season=2099, week=4, projection_p50=25.,
        prediction_context_cutoff_utc='2099-10-01T20:00:00Z', scoring_replay={
            'row':{'start_ts_utc':'2099-10-01T23:00:00Z','_pred_receiver_targets':5.,'position':'WR','targets_avg_5':5.,
                   'receiver_projected_targets_v3':5.},
            'distribution':{'kind':'empirical_oof_residual','residual_quantiles':[-10,0,10]}})
    now = datetime(2099,10,1,20,1,tzinfo=timezone.utc)
    result = capture.capture_records([(1,row)],store=tmp_path,now=now)
    assert result['captured'] == 1
    document = json.loads(next((tmp_path/'point_prospective').glob('*.json')).read_text())
    assert document['rows'][0]['candidate'] == 30.
    assert document['betting_approved'] is False and document['production_changed'] is False
    assert all(k not in document['rows'][0] for k in ('actual','outcome','probability'))
    row['prediction_context_cutoff_utc'] = '2020-10-01T20:00:00Z'
    assert capture.capture_records([(1,row)],store=tmp_path,now=now)['captured'] == 0
    assert capture.capture_records([(1,dict(row,prediction_context_cutoff_utc='2099-10-01T20:00:00Z'))],
        store=tmp_path,now=datetime(2099,10,2,tzinfo=timezone.utc))['captured'] == 0


def test_point_capture_registration_is_pinned(tmp_path, monkeypatch):
    import joblib
    from nfl_pipeline.modeling import target_point_capture as capture
    monkeypatch.setattr(capture,'active_release',lambda:{'release_id':'production'})
    artifact = dict(contract='nfl-target-components-v1',production_release='production',run_id='test',
        prospective_acceptance_start='2099-10-01',output_decisions={
            'uncertainty_only':{'median':{'historical_screen_passed':False}}})
    path = tmp_path/'model.joblib'; joblib.dump(artifact,path)
    with pytest.raises(ValueError,match='historical screen'):
        capture.register(path,store=tmp_path)
    artifact['output_decisions']['uncertainty_only']['median']['historical_screen_passed']=True
    joblib.dump(artifact,path)
    first=capture.register(path,store=tmp_path)
    assert capture.register(path,store=tmp_path)==first
    artifact['run_id']='different'; joblib.dump(artifact,path)
    with pytest.raises(ValueError,match='already pinned'):
        capture.register(path,store=tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        capture.capture_records([],store=tmp_path)
