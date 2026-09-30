from datetime import datetime, timezone

from nfl_pipeline.modeling import score_locked_micro_uncertainty as scorer


NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def record():
    return dict(ledger_id=9, prediction_id=11, execution_status='simulated',
        locked_at_utc='2026-09-18T12:00:00Z', created_at_utc='2026-09-18T11:59:00Z',
        start_ts_utc='2026-09-20T17:00:00Z', ledger_book='fanduel', ledger_line=30.5,
        ledger_side='over', ledger_price=-114, ledger_model_version='production',
        forecast_payload=dict(stat='receiving_yards', book='fanduel', side='over', line=30.5,
                              price=-114, model_version='production'))


def artifact():
    return dict(run_id='components-test', training_end='2026-09-17',
                models={'receiving_conditional': {'enabled_outputs': ['probability']}})


def test_exact_ledger_id_used_and_distinct_micro_cohort(monkeypatch):
    captured = []
    def score(records, model):
        captured.extend(records)
        assert list(model['models']) == ['receiving_conditional']
        return {'receiving_conditional': [dict(forecast_id=records[0][0], challenger_run='old')]}, {}
    monkeypatch.setattr(scorer, 'score_records', score)
    rows, excluded = scorer.score_locks([record()], artifact(), NOW)
    assert captured[0][0] == 11
    assert rows[0]['ledger_id'] == 9
    assert rows[0]['challenger_run'].endswith('-locked_micro')
    assert rows[0]['execution_status'] == 'simulated'
    assert not excluded


def test_after_kickoff_and_after_training_dates_refused(monkeypatch):
    monkeypatch.setattr(scorer, 'score_records', lambda *_: (_ for _ in ()).throw(AssertionError('Must not score')))
    r = record(); r['start_ts_utc'] = '2026-09-20T11:00:00Z'
    assert scorer.score_locks([r], artifact(), NOW)[1]['not_prospective'] == 1
    a = artifact(); a['training_end'] = '2026-09-20'
    assert scorer.score_locks([record()], a, NOW)[1]['not_prospective'] == 1


def test_changed_price_or_line_is_not_same_executable_pick():
    r = record(); r['ledger_price'] = 120
    rows, issues = scorer.score_locks([r], artifact(), NOW)
    assert not rows and issues['ledger_offer_mismatch'] == 1
    r = record(); r['forecast_payload']['line'] = None
    assert scorer.score_locks([r], artifact(), NOW)[1]['ledger_offer_mismatch'] == 1


def test_scoring_cannot_precede_ledger_lock():
    r = record(); r['locked_at_utc'] = '2026-09-20T13:00:00Z'
    rows, issues = scorer.score_locks([r], artifact(), NOW)
    assert not rows and issues['not_prospective'] == 1
