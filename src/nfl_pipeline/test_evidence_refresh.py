from datetime import date, datetime, timedelta, timezone
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from nfl_pipeline import clv_report as clv
from nfl_pipeline import run_close_and_grade as runner
from nfl_pipeline.modeling import benchmark_offers as benchmark
from nfl_pipeline.modeling.exact_line_evidence import audit_evidence
from nfl_pipeline.modeling.train_prop_exact_line_models import _prepare_frame, _split_by_date, _weights


def evidence(**kwargs):
    row = dict(id=1, game_date_et=date(2026, 9, 20), game_id='g', player_id='p', stat='receiving_yards',
        side='over', book='fanduel', line=30.5, price=-110, model_probability=.55,
        created_at_utc='2026-09-20T15:00:00Z', start_ts_utc='2026-09-20T17:00:00Z',
        offer_fetched_at='2026-09-20T14:59:00Z', integrity_version='nfl-asof-v2',
        model_version='release', forecast_payload=dict(model_version='release', line=30.5,
        prediction_context_cutoff_utc='2026-09-20T14:59:30Z'), matched_offer_id=9,
        offer_book='fanduel', offer_stat='receiving_yards', offer_line=30.5,
        offer_player='a', offer_player_name_norm='a', offer_home='DAL', offer_away='NYG',
        home_team_abbr='DAL', away_team_abbr='NYG', snapshot_role='lock',
        lock_over_price=-110, lock_under_price=-110, status='final', participated=True,
        result='win', actual=50, label_available_at='2026-09-21T08:00:00Z', season=2026, week=2)
    row.update(kwargs)
    return row


def test_audit_accounts_for_every_row_and_deduplicates_valid_locks_only():
    rows = [evidence(id=0, price=None), evidence(), evidence(id=2),
            evidence(id=3, player_id='x', lock_under_price=None), evidence(id=4, result='push')]
    frame, report = audit_evidence(pd.DataFrame(rows))
    assert frame.id.tolist() == [1]
    assert report['scanned_rows'] == report['eligible_rows'] + sum(report['exclusions'].values())
    assert report['exclusions'] == dict(invalid_locked_price=1, later_revision_same_decision=1,
                                       missing_true_pair=1, push_not_binary=1)


@pytest.mark.parametrize('change,reason', [
    ({'participated': False}, 'participation_unverified'),
    ({'status': 'scheduled'}, 'game_not_final'),
    ({'offer_line': 29.5}, 'exact_offer_identity_mismatch'),
    ({'price': -115}, 'locked_price_mismatch'),
    ({'offer_home': 'MIA'}, 'offer_game_mismatch'),
    ({'created_at_utc': '2026-09-20T18:00:00Z'}, 'lock_not_before_start'),
    ({'offer_fetched_at': '2026-09-20T15:01:00Z'}, 'invalid_lock_offer_timing'),
    ({'result': None}, 'missing_settled_result'),
    ({'identity_fetched_at': '2026-09-20T15:00:00Z'}, 'offer_game_identity_after_cutoff'),
])
def test_bad_evidence_is_explained_not_silently_dropped(change, reason):
    frame, report = audit_evidence(pd.DataFrame([evidence(**change)]))
    assert frame.empty and report['exclusions'] == {reason: 1}


def test_unknown_clv_not_a_negative_training_label():
    df = pd.DataFrame([dict(result='win', clv_status=status, clv_prob_delta=value,
        price=-110, lock_over_price=-110, lock_under_price=-110, stat='receiving_yards',
        side='over', line=30.5, model_probability=.55, model_edge=10)
        for status,value in [('missing_close', None), ('valid_close', 0.), ('valid_close', .01)]])
    out = _prepare_frame(df)
    assert np.isnan(out.target_clv_beat.iloc[0])
    assert out.target_clv_beat.iloc[1:].tolist() == [0., 1.]


def test_split_groups_weeks_and_respects_label_availability():
    rows = [evidence(id=i, week=i, game_id=f'g{i}',
        created_at_utc=f'2026-09-{i*7:02}T15:00:00Z',
        label_available_at=f'2026-09-{i*7+1:02}T12:00:00Z') for i in (1,2,3)]
    rows[1]['label_available_at'] = '2026-09-23T12:00:00Z'
    train, holdout = _split_by_date(pd.DataFrame(rows))
    assert train.id.tolist() == [1] and holdout.id.tolist() == [3]
    assert _split_by_date(pd.DataFrame(rows[:2]))[0].empty
    copies = pd.DataFrame([evidence(id=i) for i in range(10)])
    assert _weights(copies).sum() == pytest.approx(1.)


def close_record(**kwargs):
    start = datetime(2026,9,21,20,tzinfo=timezone.utc)
    row = dict(side='over',line=13.5,locked_line=13.5,over_price=-110,under_price=-110,
        start_ts_utc=start,commence_time_utc=start,created_at_utc=start-timedelta(hours=3),
        fetched_at_utc=start-timedelta(minutes=125), integrity_version='nfl-asof-v2',
        lock_offer_id=1,fresh_book_rows=20,fresh_market_rows=5,observed_lines=[11.5,12.5])
    row.update(kwargs); return row


def test_moved_line_is_not_a_capture_failure_or_fabricated_price_clv():
    q = clv.classify_prop_close(close_record())
    assert q['status'] == 'exact_line_unavailable_in_captured_feed'
    assert not q['valid'] and q['available'] is None
    q = clv.classify_prop_close(close_record(fresh_market_rows=0))
    assert q['status'] == 'player_market_not_observed_in_capture'
    q = clv.classify_prop_close(close_record(fresh_market_rows=0, fresh_book_rows=0))
    assert q['status'] == 'close_outside_two_hour_window'


@pytest.mark.parametrize('minutes,status', [(119,'valid_close'),(120,'valid_close'),
    (121,'close_outside_two_hour_window'), (0,'close_after_kickoff'),(-1,'close_after_kickoff')])
def test_strict_close_window(minutes,status):
    row=close_record(fresh_market_rows=0,fresh_book_rows=0)
    row['fetched_at_utc']=row['start_ts_utc']-timedelta(minutes=minutes)
    assert clv.classify_prop_close(row)['status'] == status


@pytest.fixture(autouse=True)
def isolated_settlement(monkeypatch, tmp_path):
    """Settlement gating reads the DB and a state file; keep tests off both."""
    signature = {'value': dict(final_games='1')}
    monkeypatch.setattr(runner, 'SETTLEMENT_STATE', tmp_path/'settlement_state.json')
    monkeypatch.setattr(runner, '_results_pending', lambda now: False)
    monkeypatch.setattr(runner, '_settlement_signature', lambda: dict(signature['value']))
    return signature


def test_unchanged_settlement_inputs_skip_the_ten_minute_sweep(monkeypatch, isolated_settlement):
    calls=[]
    monkeypatch.setattr(runner, '_has_close_work', lambda d: True)
    monkeypatch.setattr(runner, '_locked_prop_count', lambda d: 1)
    monkeypatch.setattr(runner, '_active_close_games', lambda *a,**k: [])
    monkeypatch.setattr(runner, '_run', lambda step: calls.append(step.module) or (0,'{}',''))
    first = runner.run_for_date(date(2026,9,21))
    assert first['status']=='ok' and 'nfl_pipeline.import_nflverse' in calls  # first run: no import recorded yet
    calls.clear()
    second = runner.run_for_date(date(2026,9,21))
    # Nothing runs on a skipped sweep; the sharp watcher has its own task and its own mutex.
    assert second['close_window']['mode']=='settlement_skipped_no_new_inputs' and calls==[]
    isolated_settlement['value'] = dict(final_games='2')  # e.g. a new final result or a user cash confirmation
    third = runner.run_for_date(date(2026,9,21))
    assert 'nfl_pipeline.grade_predictions' in calls
    assert 'nfl_pipeline.import_nflverse' not in calls  # results import is not due again yet
    assert third['settlement']['results_import_due'] is False


def test_pending_results_force_import_and_failed_runs_are_not_marked_complete(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, '_has_close_work', lambda d: True)
    monkeypatch.setattr(runner, '_locked_prop_count', lambda d: 1)
    monkeypatch.setattr(runner, '_active_close_games', lambda *a,**k: [])
    monkeypatch.setattr(runner, '_results_pending', lambda now: True)
    monkeypatch.setattr(runner, '_run', lambda step: calls.append(step.module) or (
        (1,'','boom') if step.module=='nfl_pipeline.grade_predictions' else (0,'{}','')))
    assert runner.run_for_date(date(2026,9,21))['status']=='failed'
    calls.clear()
    runner.run_for_date(date(2026,9,21))  # same inputs, but the last run failed: rerun everything
    assert 'nfl_pipeline.import_nflverse' in calls and 'nfl_pipeline.grade_predictions' in calls


def test_close_before_lock_stays_invalid():
    row=close_record(fresh_book_rows=0,fresh_market_rows=0)
    row['fetched_at_utc']=row['created_at_utc']
    assert clv.classify_prop_close(row)['status'] == 'stale_close_before_lock'


def cross_provider_close(**changes):
    row = close_record(lock_provider='oddsapi_watch', close_provider='sportsgameodds',
        lock_event_id='provider-a', close_event_id='provider-b',
        game_home_team='NO', game_away_team='ATL',
        lock_home_team='New Orleans Saints', lock_away_team='Atlanta Falcons',
        close_home_team='New Orleans Saints', close_away_team='Atlanta Falcons')
    row['fetched_at_utc'] = row['start_ts_utc'] - timedelta(minutes=10)
    row['lock_commence_time_utc'] = row['start_ts_utc'] + timedelta(minutes=3)
    row.update(changes)
    return row


def test_exact_same_book_close_can_cross_provider_event_ids():
    quality = clv.classify_prop_close(cross_provider_close())
    assert quality['valid'] and quality['available'] is True
    assert quality['match_method'] == 'verified_same_book_matchup_cross_provider'


@pytest.mark.parametrize('change', [
    {'close_home_team': 'DAL'}, {'lock_away_team': 'DAL'}, {'game_home_team': None},
    {'lock_commence_time_utc': None}, {'commence_time_utc': None},
    {'commence_time_utc': datetime(2026, 9, 21, 20, 6, tzinfo=timezone.utc)},
    {'lock_commence_time_utc': datetime(2026, 9, 21, 20, 6, tzinfo=timezone.utc)},
])
def test_cross_provider_close_rejects_ambiguous_game_or_kickoff(change):
    quality = clv.classify_prop_close(cross_provider_close(**change))
    assert not quality['valid'] and quality['available'] is None
    assert quality['status'] == 'cross_provider_game_identity_unverified'


def test_cross_provider_close_never_substitutes_an_alternate_line():
    quality = clv.classify_prop_close(cross_provider_close(line=14.5))
    assert not quality['valid'] and quality['status'] == 'exact_line_mismatch'


def test_cross_provider_identity_does_not_relax_close_timing():
    row = cross_provider_close()
    row['fetched_at_utc'] = row['created_at_utc']
    row.update(fresh_book_rows=0, fresh_market_rows=0)
    assert clv.classify_prop_close(row)['status'] == 'stale_close_before_lock'


def test_capture_hot_path_retries_empty_book_and_does_not_train(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, '_has_close_work', lambda d: True)
    monkeypatch.setattr(runner, '_locked_prop_count', lambda d: 2)
    monkeypatch.setattr(runner, '_active_close_games', lambda *a,**k: [dict(game_id='g',minutes_to_start=20)])
    checks=iter([[dict(fresh_rows=0)], [dict(fresh_rows=5)]])
    monkeypatch.setattr(runner, '_capture_health', lambda *a: next(checks))
    def run(step):
        calls.append(step.module)
        return 0,json.dumps(dict(player_prop_events_saved=1)),''
    monkeypatch.setattr(runner, '_run', run)
    result=runner.run_for_date(date(2026,9,21))
    assert result['status']=='ok' and result['close_window']['retry_fired']
    assert calls.count('nfl_pipeline.crawler_oddsapi')==2
    assert 'nfl_pipeline.import_nflverse' not in calls
    assert 'nfl_pipeline.modeling.train_prop_exact_line_models' not in calls
    assert 'nfl_pipeline.modeling.receiving_trial_checkpoint' not in calls


def test_capture_still_empty_returns_failure(monkeypatch):
    monkeypatch.setattr(runner, '_has_close_work', lambda d: True)
    monkeypatch.setattr(runner, '_locked_prop_count', lambda d: 1)
    monkeypatch.setattr(runner, '_active_close_games', lambda *a,**k: [dict(game_id='g',minutes_to_start=20)])
    monkeypatch.setattr(runner, '_capture_health', lambda *a: [dict(fresh_rows=0)])
    monkeypatch.setattr(runner, '_run', lambda s: (0,'{"player_prop_events_saved":1}',''))
    assert runner.run_for_date(date(2026,9,21))['status']=='failed'


def test_review_never_grants_betting_permission():
    metrics=dict(rows=200,weeks=5,final_brier_gain={'lower_95':.002},
                 production={'calibration_error':.04},final={'calibration_error':.03},
                 live_curve={'coverage_80':.8},production_curve={'coverage_80':.75})
    review=benchmark.deployment_review(dict(real_offers=metrics,exact_micro=metrics))
    assert review['status']=='ready_for_component_review'
    assert not review['deployment_approved'] and not review['betting_approved']
    worse=deepcopy(metrics); worse['live_curve']['coverage_80']=.7
    review=benchmark.deployment_review(dict(real_offers=worse,exact_micro=metrics))
    assert 'real_offers:distribution_coverage_not_confirmed' in review['blockers']


def test_settlement_path_refreshes_all_dates_without_new_locks(monkeypatch):
    calls=[]
    monkeypatch.setattr(runner, '_has_close_work', lambda d: True)
    monkeypatch.setattr(runner, '_locked_prop_count', lambda d: 1)
    monkeypatch.setattr(runner, '_active_close_games', lambda *a,**k: [])
    def run(step):
        calls.append(step)
        return 0,'{}',''
    monkeypatch.setattr(runner, '_run', run)
    result=runner.run_for_date(date(2026,9,21))
    assert result['status']=='ok'
    grade=next(s for s in calls if s.module=='nfl_pipeline.grade_predictions')
    ledger=next(s for s in calls if s.module=='nfl_pipeline.lock_ledger')
    assert grade.args==() and ledger.args==('--refresh-only',)
    assert any(s.module=='nfl_pipeline.modeling.train_prop_exact_line_models' for s in calls)
    checkpoint=next(s for s in calls if s.module=='nfl_pipeline.modeling.receiving_trial_checkpoint')
    assert checkpoint.critical and checkpoint.args==('--date','2026-09-21')
    modules=[s.module for s in calls]
    assert modules.index('nfl_pipeline.grade_predictions') < modules.index(checkpoint.module)
    assert modules.index('nfl_pipeline.clv_report') < modules.index(checkpoint.module)


def test_checkpoint_failure_cannot_be_reported_as_scheduler_success(monkeypatch):
    monkeypatch.setattr(runner,'_has_close_work',lambda d:True)
    monkeypatch.setattr(runner,'_locked_prop_count',lambda d:1)
    monkeypatch.setattr(runner,'_active_close_games',lambda *a,**k:[])
    monkeypatch.setattr(runner,'_run',lambda step:(
        (1,'capture_incomplete','') if step.module=='nfl_pipeline.modeling.receiving_trial_checkpoint' else (0,'{}','')))
    result=runner.run_for_date(date(2026,9,24))
    assert result['status']=='failed'
    assert result['steps'][-1]['label']=='NFL Receiving Trial Checkpoint'


def test_zero_or_nonfinite_close_price_is_unknown():
    for price in (0, float('nan'), 99):
        row=close_record(over_price=price)
        row['fetched_at_utc']=row['start_ts_utc']-timedelta(minutes=10)
        q=clv.classify_prop_close(row)
        assert not q['valid'] and q['status']=='same_line_missing_side_price'


def test_pinned_trial_cannot_silently_follow_latest(monkeypatch,tmp_path):
    monkeypatch.setattr(benchmark,'STORE',tmp_path)
    monkeypatch.setattr(benchmark,'load_artifact',lambda **k: ({'run_id':'trial'}, {'run_id':'trial','sha256':'abc'}))
    benchmark.pin_current()
    assert json.loads((tmp_path/'active_trial.json').read_text())['run_id']=='trial'
    monkeypatch.setattr(benchmark,'load_artifact',lambda **k: ({'run_id':'next'}, {'run_id':'next','sha256':'def'}))
    with pytest.raises(ValueError,match='different trial'):
        benchmark.pin_current()
