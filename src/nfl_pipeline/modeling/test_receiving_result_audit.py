from nfl_pipeline.modeling import receiving_result_audit as audit
from nfl_pipeline.modeling import benchmark_offers


def test_missing_grade_is_a_mismatch_even_when_result_source_is_complete(monkeypatch):
    monkeypatch.setattr(audit, 'eligible_error', lambda r,c: None)
    row=dict(id=1, game_id='g', player_id='p', player_name='P', status='final', actual=0,
             line=10.5, side='under', graded_result=None)
    result=audit.summarize([row], {})
    assert result['unresolved_final_rows']==0
    assert result['grading_mismatches'][0]['expected_grade']=='win'


def test_missing_participation_stays_unresolved_and_revisions_are_counted_separately(monkeypatch):
    monkeypatch.setattr(audit, 'eligible_error', lambda r,c: None)
    row=dict(id=1, game_id='g', player_id='p', player_name='P', status='final', actual=None,
             result_game_id='g', actual_offense_snaps=0, graded_result='pending_participation_review')
    result=audit.summarize([row, dict(row,id=2)], {})
    assert result['unresolved_final_rows']==2
    assert result['unique_unresolved_player_games']==1
    assert result['grading_mismatches']==[]


def test_future_game_is_not_a_missing_final_result(monkeypatch):
    monkeypatch.setattr(audit, 'eligible_error', lambda r,c: 'invalid_quote' if r['id']==2 else None)
    row=dict(id=1, game_id='g', player_id='p', player_name='P', status='scheduled', actual=None)
    result=audit.summarize([row, dict(row,id=2)], {})
    assert result['states']=={'pending_game':1}
    assert result['unresolved_final_rows']==0
    assert result['exclusions']=={'invalid_quote':1}


def test_stored_but_expired_capture_stays_excluded(monkeypatch):
    monkeypatch.setattr(benchmark_offers,'original_offer_matches',lambda *args: True)
    config=dict(run_id='run',artifact_sha256='sha',variant='candidate',rules={'max_quote_age_minutes':20})
    records=[dict(id=1,forecast_payload={'scoring_replay':{'offer':{'fetched_at_utc':'2026-10-18T16:00:00Z'}}})]
    doc=dict(run_id='run',source_manifest={'sha256':'sha'},scored_at='2026-10-18T16:21:00Z',
             rows=[dict(forecast_id=1,variant='candidate')])
    result=audit.check_excluded_captures(records,[doc],config,{1})
    assert result['status']=='verified_exclusion'
    assert result['rejected_capture_reasons']=={'quote_expired_before_challenger_capture':1}
    doc['scored_at']='2026-10-18T16:01:00Z'
    result=audit.check_excluded_captures(records,[doc],config,{1})
    assert result['status']=='needs_review'
    assert result['unexpected_valid_capture_ids']==[1]
