from copy import deepcopy
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline.modeling.role_context_training import RoleEvidenceIndex, join_role_context
from nfl_pipeline.modeling.target_volume import TargetVolume, target_features
from nfl_pipeline import cash_readiness

LOCK = datetime(2026, 9, 24, 20, tzinfo=timezone.utc)


def observation(kind='nfl_depth_charts', player='p', at=None, **payload):
    return dict(kind=kind, row_id=str(payload), observed_at=at or LOCK-timedelta(hours=1),
        payload=dict(season=2026, week=3, player_id=player, team_abbr='DAL', source='test',
                     pos_rank=1, pos_abb='WR', **payload))


def request():
    return dict(game_id='g', player_id='p', team_abbr='DAL', season=2026, week=3,
        role_lock_cutoff=LOCK, role_lock_id=7, start_ts_utc=LOCK+timedelta(hours=2))


def test_asof_join_rejects_future_source_and_late_observation():
    old = observation(at=LOCK-timedelta(days=1)); old['payload']['pos_rank'] = 2
    future = observation(at=LOCK+timedelta(minutes=1))
    future_source = observation(snapshot_ts_utc=LOCK+timedelta(days=1))
    result = RoleEvidenceIndex([old, future, future_source]).resolve(request())
    assert result['depth_rank'] == 2 and result['expected_starter'] is False
    assert result['provenance']['nfl_depth_charts']['observed_at'] == old['observed_at'].isoformat()


def test_own_unknown_does_not_erase_known_teammate_or_imply_health():
    r = observation('nfl_injuries', player='q', report_status='Out', position='WR')
    result = RoleEvidenceIndex([r]).resolve(request())
    assert result['injury_status'] is None
    assert result['teammate_absences'] == 1
    assert result['teammate_coverage'] == 'partial_observed_reports'
    r['payload']['week'] = 2
    assert RoleEvidenceIndex([r]).resolve(request())['teammate_absences'] is None


def test_team_phase_and_actual_lock_identity_are_required():
    r = observation('nfl_injuries', report_status='Out', season_type='POST')
    assert RoleEvidenceIndex([r]).resolve(dict(request(), season_type='REG'))['injury_status'] is None
    r = observation(); r['payload']['team_abbr'] = 'MIA'
    assert RoleEvidenceIndex([r]).resolve(request())['depth_rank'] is None
    frame = pd.DataFrame([request()]).drop(columns=['role_lock_cutoff', 'role_lock_id'])
    joined, health = join_role_context(frame, [], [observation()])
    assert joined.target_role_evidence.iloc[0]['status'] == 'missing_verified_lock'
    assert health['overall']['verified_locks'] == 0


def test_earliest_lock_prevents_later_context_from_backfilling():
    frame = pd.DataFrame([request()])
    locks = [dict(request(), created_at_utc=LOCK, source_cutoff=LOCK-timedelta(hours=2)),
             dict(request(), role_lock_id=8, created_at_utc=LOCK+timedelta(hours=1), source_cutoff=LOCK)]
    joined, _ = join_role_context(frame, locks, [observation()])
    assert joined.target_role_evidence.iloc[0]['depth_rank'] is None
    assert joined.target_role_evidence.iloc[0]['lock_id'] == 7


def fixture(n=350):
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(dict(game_id=[f'g{i}' for i in range(n)], player_id=[f'p{i}' for i in range(n)],
        team_abbr='DAL', position='WR', targets_avg_3=rng.uniform(2, 9, n), targets_avg_5=rng.uniform(2, 9, n),
        targets_avg_10=rng.uniform(2, 9, n), target_share_avg_3=.17, target_share_avg_10=.15,
        targets=rng.poisson(5, n), receiving_yards=rng.uniform(0, 100, n), team_actual_pass_attempts=34))
    return frame


class Reference:
    def predict(self, frame):
        return frame.targets_avg_5.to_numpy()*8


def test_target_only_ablation_keeps_efficiency_identical_and_ignores_outcomes():
    data = fixture()
    model = TargetVolume().fit(data).fit_residuals(data.iloc[:100], Reference())
    model.alpha = .5
    test = data.iloc[:4].copy()
    a = model.yardage_curve(test, Reference(), 'control')
    b = model.yardage_curve(test, Reference(), 'challenger')
    assert np.array_equal(a[2]['rate'], b[2]['rate'])
    np.testing.assert_allclose(b[1].sum(axis=1), 1)
    original = deepcopy(test)
    test[['targets', 'receiving_yards', 'team_actual_pass_attempts']] = 999
    test['injury_report_status'] = 'Out'
    changed = model.yardage_curve(test, Reference(), 'challenger')
    np.testing.assert_array_equal(b[0], changed[0])
    assert not any(c.startswith('role_') for c in model.columns)
    assert all(np.isnan(x) for x in target_features(original).role_injury_status_out)
    model.alpha = 0
    np.testing.assert_array_equal(model.yardage_curve(original, Reference(), 'challenger')[0], a[0])


def test_missing_and_duplicate_training_targets_fail():
    data = fixture()
    with pytest.raises(ValueError):
        TargetVolume().fit(pd.concat([data, data.iloc[:1]]))
    data.loc[0, 'targets'] = np.nan
    with pytest.raises(ValueError):
        TargetVolume().fit(data)


def test_cash_writer_lock_retries_without_loosening_policy(monkeypatch):
    calls = []
    class Cursor:
        def execute(self, sql, params):
            calls.append((sql, params))
        def fetchone(self):
            return (len(calls) >= 3,)
    monkeypatch.setattr(cash_readiness.time, 'sleep', lambda _: None)
    cash_readiness.acquire_review_lock(Cursor())
    assert len(calls) == 3 and all('pg_try_advisory_xact_lock' in r[0] for r in calls)


def test_cash_writer_busy_is_failure_not_approval(monkeypatch):
    class Cursor:
        def execute(self, *args):
            pass
        def fetchone(self):
            return (False,)
    with pytest.raises(TimeoutError, match='no checkpoint'):
        cash_readiness.acquire_review_lock(Cursor(), timeout_s=0)


def test_cash_history_loading_happens_before_writer_transaction(monkeypatch):
    calls = []
    monkeypatch.setattr(cash_readiness.policy, 'load_registration', lambda: {'sha256':'p', 'config':{'trial_sha256':'t'}})
    monkeypatch.setattr(cash_readiness.trial, 'load_registration', lambda: {'sha256':'t'})
    def load(*args):
        calls.append('load')
        return [], [], ([], [], {}, 0, 0)
    monkeypatch.setattr(cash_readiness, 'load_review_inputs', load)
    def connect(*args):
        calls.append('connect')
        raise RuntimeError('stop at transaction boundary')
    monkeypatch.setattr(cash_readiness.psycopg2, 'connect', connect)
    with pytest.raises(RuntimeError):
        cash_readiness.report(LOCK.date())
    assert calls == ['load', 'connect']


def test_full_path_keeps_price_calibration_confidence_and_original_lock(monkeypatch):
    from nfl_pipeline.modeling import target_volume_validation as validation
    captured = dict(distribution={'projection_confidence': .63},
        offer={'over_price': -110, 'under_price': -110, 'line': 45.5}, calibration=['unchanged'],
        clv_guards=['unchanged'], projection=50.)
    p = dict(scoring_replay=captured, side='over', probability=.6, line=45.5)
    original = deepcopy(p); calls = []
    def replay(row):
        calls.append(deepcopy(row))
        return dict(side='over', probability=.6, probability_trace={'final': .6})
    monkeypatch.setattr(validation, 'replay', replay)
    result = validation.changed_probability(p, np.array([20.,60.]), np.array([.5,.5]))
    assert p == original and result['confidence'] == .63
    assert calls[-1]['offer'] == captured['offer']
    assert calls[-1]['calibration'] == captured['calibration']
    assert calls[-1]['clv_guards'] == captured['clv_guards']
    assert calls[-1]['projection'] == 40


def test_rejected_challenger_cannot_start_prospective_capture(tmp_path):
    import joblib
    from nfl_pipeline.modeling.target_volume_validation import run
    path = tmp_path/'model.joblib'
    joblib.dump({'historical_screen_passed': False}, path)
    with pytest.raises(ValueError, match='Rejected'):
        run(path, 2026, 4, capture=True)


def test_original_target_baseline_override_keeps_frozen_efficiency():
    data = fixture(); model = TargetVolume().fit(data).fit_residuals(data.iloc[:100], Reference())
    frame = data.iloc[:2]; target = [7.,3.]; projection = [56.,24.]
    model.alpha = .5
    _, _, control = model.yardage_curve(frame,None,'control',projection,target)
    _, _, candidate = model.yardage_curve(frame,None,'challenger',projection,target)
    np.testing.assert_array_equal(control['rate'], [8.,8.])
    np.testing.assert_array_equal(candidate['rate'], control['rate'])
