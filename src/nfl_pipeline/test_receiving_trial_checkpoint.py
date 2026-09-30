from copy import deepcopy
from datetime import date, datetime, timezone

import pytest

from nfl_pipeline.modeling import receiving_trial_checkpoint as checkpoint
from nfl_pipeline.modeling import benchmark_offers as benchmark
from nfl_pipeline.test_prospective_market_trial import paired_record, config, shadow
from nfl_pipeline.test_benchmark_offers import document

DAY = date(2026,9,22)
NOW = datetime(2026,9,22,10,10,tzinfo=timezone.utc)


def inputs():
    rec = paired_record(); rec['is_current'] = True
    doc = document()
    doc.update(scored_at=NOW.isoformat(), source_manifest={'run_id':'benchmark-test','sha256':'hash'})
    doc['rows'] = [shadow()]
    selection = dict(forecast_id=1,day=str(DAY))
    return [rec],[doc],[],{'config':config(),'sha256':'registered'},[selection],{}


def build(data=None, now=NOW):
    return checkpoint.build_checkpoint(DAY,*(data or inputs()),now)


def test_matched_scopes_have_actual_final_probabilities_and_no_automatic_bets():
    doc = build()
    assert doc['capture']['missing_current_forecast_ids']==[]
    assert doc['capture']['paired_challenger_locks']==1
    for scope in ('all_eligible','fixed_research'):
        m = doc['day_metrics'][scope]['real_offers']
        assert m['rows']==1
        assert m['production']['brier']==pytest.approx(.65**2)
        assert m['final']['brier']==pytest.approx(.60**2)
        assert m['matched_market']['rows']==1
        assert m['live_curve']['coverage_80'] is not None
    assert not doc['betting_approved'] and not doc['production_changed']
    assert doc['forecast_next_action']=='keep_frozen_collect_evidence'
    assert 'independent' in ' '.join(doc['forecast_deployment_review']['blockers'])


def test_unsettled_games_cannot_look_like_a_failed_model_or_close():
    data=inputs(); data[0][0].update(actual=None,status='scheduled')
    doc=build(data)
    assert doc['status']=='awaiting_settlement'
    assert doc['settlement']=={'pending_game':1}
    assert doc['close']['fixed_research']['valid_coverage'] is None
    assert not doc['day_metrics']['fixed_research']


@pytest.mark.parametrize('change', ['missing','wrong_line','late','wrong_pin'])
def test_missed_or_invalid_capture_is_an_explicit_failure(change):
    data=inputs()
    if change=='missing': data[1].clear()
    if change=='wrong_line': data[1][0]['rows'][0]['source_line']=40.5
    if change=='late': data[1][0]['scored_at']='2026-09-22T20:01:00Z'
    if change=='wrong_pin': data[1][0]['source_manifest']['sha256']='other'
    doc=build(data)
    assert doc['status']=='capture_incomplete'
    assert doc['capture']['missing_current_forecast_ids']==[1]
    assert not doc['day_metrics']['all_eligible']


def test_stale_original_quotes_are_accounted_for_not_replaced_with_new_prices():
    data=inputs()
    rec=data[0][0]; rec['offer_fetched_at']='2026-09-22T01:00:00Z'
    rec['forecast_payload']['scoring_replay']['offer']['fetched_at_utc']=rec['offer_fetched_at']
    original=deepcopy(rec)
    doc=build(data)
    assert doc['status']=='no_eligible_offers'
    assert doc['capture']['eligibility_exclusions']['stale_original_quote']==1
    assert rec==original


def close():
    return dict(valid_close_snapshot_captured=True,clv_prob_delta=0.,close_quality_reason='valid_close',
        book='fanduel',stat='receiving_yards',side='over',locked_line=30.5,close_line=30.5,
        close_fetched_at_utc='2026-09-22T19:40:00Z')


def test_missing_close_unknown_flat_close_known_and_wrong_line_rejected():
    now=datetime(2026,9,22,21,tzinfo=timezone.utc)
    data=inputs()
    missing=build(data,now)['close']['fixed_research']
    assert missing['unknown']==1 and missing['average_clv'] is None
    data[-1][1]=close()
    flat=build(data,now)['close']['fixed_research']
    assert flat['valid']==1 and flat['average_clv']==0 and flat['target_met']
    data[-1][1]['close_line']=31.5
    wrong=build(data,now)['close']['fixed_research']
    assert wrong['valid']==0 and wrong['average_clv'] is None
    assert wrong['unknown_reasons']=={'close_label_identity_or_timing_mismatch':1}


def test_price_becoming_stale_today_does_not_delete_honest_original_evidence():
    doc=build(now=datetime(2026,9,22,15,tzinfo=timezone.utc))
    assert doc['capture']['current_quotes_fresh_now']==0
    assert doc['capture']['paired_challenger_locks']==1
    assert doc['day_metrics']['fixed_research']['real_offers']['rows']==1


def test_proven_forecast_can_reach_review_without_roi_clv_or_cash_approval():
    m=dict(rows=100,weeks=4,production={'calibration_error':.03},final={'calibration_error':.02},
        final_brier_gain={'lower_95':.001},live_curve={'coverage_80':.8},production_curve={'coverage_80':.8})
    review=benchmark.deployment_review({'real_offers':m})
    assert review['status']=='ready_for_component_review'
    assert not review['blockers'] and review['market_evidence_blockers']
    assert not review['betting_approved']


def test_capture_gaps_remain_visible_after_day_rollover():
    data=inputs(); data[1].clear(); data[0][0]['is_current']=False
    doc=build(data,datetime(2026,9,23,tzinfo=timezone.utc))
    assert doc['capture']['missing_historical_forecast_ids']==[1]
    assert 'missing_prospective_capture_history' in doc['forecast_deployment_review']['blockers']


def test_nonfinite_clv_is_unknown_not_positive_proof():
    data=inputs();data[-1][1]=dict(close(),clv_prob_delta=float('nan'))
    doc=build(data,datetime(2026,9,23,tzinfo=timezone.utc))
    assert doc['close']['fixed_research']['valid']==0
    assert doc['close']['fixed_research']['average_clv'] is None


def test_verified_inactive_is_void_not_pending_or_winning_under():
    data=inputs();data[0][0].update(actual=None,graded_result='void_nonparticipant')
    doc=build(data)
    assert doc['settlement']=={'void_nonparticipant':1}
    assert doc['pending_variant_rows']==0
    assert not doc['day_metrics']['all_eligible']
