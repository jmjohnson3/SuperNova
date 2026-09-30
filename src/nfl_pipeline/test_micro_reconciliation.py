from copy import deepcopy
from datetime import datetime, timezone

import pytest

from nfl_pipeline.modeling.micro_reconciliation import classify, reconcile
from nfl_pipeline.test_prospective_market_trial import paired_record

NOW=datetime(2026,9,23,tzinfo=timezone.utc)


def entry():
    r=paired_record();p=r['forecast_payload']
    return dict(r,ledger_id=1,prediction_id=r['id'],game_date_et='2026-09-22',
        integrity_version='nfl-asof-v2',ledger_result='loss',graded_result='loss',actual_stat=10.,
        ledger_book=p['book'],ledger_side=p['side'],ledger_line=p['line'],ledger_price=p['price'],
        ledger_model_version=p['model_version'],locked_at_utc='2026-09-22T10:02:00Z',execution_status='simulated')


def test_verified_probability_can_be_scored_without_inventing_replay():
    r=entry();r['forecast_payload'].pop('scoring_replay')
    assert classify(r,NOW)[0]=='evaluable'
    before=deepcopy(r);doc=reconcile([r],[1],NOW)
    assert doc['categories']=={'evaluable':1} and not doc['evaluable_but_unscored']
    assert r==before and not doc['locks_changed']


@pytest.mark.parametrize('changes,expected',[
    ({'id':None,'ledger_result':'voided_stale_card'},('void_or_push','recorded_voided_stale_card')),
    ({'id':None},('missing_evaluation_inputs','original_prediction_missing')),
    ({'integrity_version':'legacy','forecast_payload':None},('missing_evaluation_inputs','legacy_missing_immutable_lock_payload')),
    ({'actual_stat':None},('missing_result','final_result_or_participation_unresolved')),
    ({'ledger_price':150},('missing_evaluation_inputs','ledger_offer_identity_mismatch')),
    ({'locked_at_utc':'2026-09-23T10:00:00Z'},('missing_evaluation_inputs','ledger_lock_timing_invalid')),
    ({'ledger_result':'win'},('missing_result','grading_or_ledger_result_mismatch')),
    ({'status':'scheduled'},('pending','awaiting_final_result')),
])
def test_every_exclusion_has_an_explicit_category(changes,expected):
    assert classify(dict(entry(),**changes),NOW)==expected


def test_pending_not_started_and_bad_probability():
    r=entry();r['status']='scheduled'
    assert classify(r,datetime(2026,9,21,tzinfo=timezone.utc))[1]=='not_started'
    r=entry();r['forecast_payload']['probability']=float('nan')
    assert classify(r,NOW)==('missing_evaluation_inputs','invalid_locked_probability_or_offer')


def test_reconciliation_accounts_for_old_void_and_new_rows_separately():
    rows=[entry(),dict(entry(),prediction_id=2,integrity_version='legacy',forecast_payload=None),
          dict(entry(),prediction_id=3,id=None,ledger_result='voided_stale_card')]
    doc=reconcile(rows,[1],NOW)
    assert doc['ledger_rows']==doc['accounted_rows']==sum(doc['categories'].values())==3
    assert not doc['evaluable_but_unscored'] and not doc['legacy_rows_added_to_training']
