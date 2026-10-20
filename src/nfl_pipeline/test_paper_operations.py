"""Operational status distinguishes pending windows, exclusions and source evidence."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

from nfl_pipeline import paper_operations as audit

DAY=date(2026,10,19)
NOW=datetime(2026,10,19,20,tzinfo=timezone.utc)
START=NOW+timedelta(hours=1)


def inputs():
    game=dict(game_id='game',start_ts_utc=START,status='scheduled',home_team_abbr='SF',away_team_abbr='WAS')
    quote=dict(provider='oddsapi',home_team='SF',away_team='WAS',commence_time_utc=START,
               fetched_at_utc=NOW-timedelta(minutes=5),over_price=-110,under_price=-110)
    forecast=dict(id=1,game_id='game',player_id='player',stat='receiving_yards',is_current=True,
                  status='scheduled',created_at_utc=NOW-timedelta(minutes=3),quote_at=quote['fetched_at_utc'],
                  generation_sha='frozen',captured_provider='oddsapi',graded_result=None,actual=None)
    captured=dict(forecast_id=1,game_id='game',player_id='player',stat='receiving_yards',day=str(DAY),
                  decision_id='decision',selected=True,line=50.5,side='over')
    receipt=dict(date=str(DAY),game_ids=['game'],ended_at=NOW.isoformat(),
                 freeze_verification=dict(experiments=dict(receiving=dict(sha256='frozen'))),
                 steps=[dict(exit_code=0)],capture_result=dict(captured=1,exclusions={}))
    legacy=dict(day='2026-10-18',capture=dict(missing_current_forecast_ids=list(range(100,113)),
                                           missing_historical_forecast_ids=list(range(100,120)),captured_forecast_ids=[99]))
    return dict(games=[game],quotes=[quote],forecasts=[forecast],documents=[dict(rows=[captured])],
                closes={},receipt=receipt,legacy=legacy,freezes=dict(status='verified'),registration_sha='frozen')


def build(data=None,now=NOW):
    return audit.build(DAY,now=now,**(data or inputs()))


def test_current_capture_does_not_claim_future_close_or_grade_completed():
    doc=build();stages=doc['games'][0]['stages']
    assert doc['status']=='pending' and stages['capture']['status']=='captured'
    assert stages['exact_close']['status']==stages['grading']['status']=='not_due'
    assert doc['legacy_missing']['previously_current_missing_count']==13
    assert doc['legacy_missing']['historical_missing_count']==20
    assert doc['legacy_missing']['excluded_from_paper_evidence']
    assert doc['earliest_kickoff']==START.isoformat()
    assert not doc['betting_approved'] and not doc['production_changed']


def test_stale_wrong_date_wrong_cohort_and_future_receipts_cannot_prove_empty_capture():
    for changes in ({'ended_at':(NOW-timedelta(minutes=21)).isoformat()}, {'date':'2026-10-18'},
                    {'freeze_verification':{}}, {'ended_at':(NOW+timedelta(minutes=1)).isoformat()}):
        data=inputs();data['documents']=[];data['receipt'].update(changes)
        doc=build(data)
        assert doc['games'][0]['stages']['capture']['status']=='unknown'
        assert doc['status']=='needs_attention'
        assert not doc['source_report_success_used']


def test_current_zero_selection_receipt_retains_exclusions_without_fabricating_rows():
    data=inputs();data['documents']=[]
    data['receipt']['capture_result']=dict(captured=0,exclusions={'same_week_injury_context_unknown':9})
    doc=build(data)
    assert doc['games'][0]['stages']['capture']['status']=='examined_no_eligible_captures'
    assert doc['games'][0]['stages']['capture']['rows']==0
    assert doc['collection_receipt']['exclusions']=={'same_week_injury_context_unknown':9}
    assert doc['status']!='complete'


def test_stale_quotes_require_refresh_only_when_the_window_is_due():
    data=inputs();data['quotes'][0]['fetched_at_utc']=NOW-timedelta(hours=1)
    assert build(data)['games'][0]['stages']['quotes']['status']=='needs_refresh'
    data['games'][0]['start_ts_utc']=NOW+timedelta(hours=4)
    assert build(data)['games'][0]['stages']['quotes']['status']=='not_due'


def test_final_missing_participation_and_close_stay_unknown():
    data=inputs();data['games'][0]['status']='final';data['forecasts'][0].update(status='final',actual=0)
    doc=build(data,START+timedelta(hours=4));stages=doc['games'][0]['stages']
    assert stages['grading']['status']=='pending_result_or_participation'
    assert stages['grading']['accounting']=={'pending_participation_or_result':1}
    assert stages['exact_close']['status']=='missing_exact_close'
    assert doc['status']=='needs_attention'


def test_only_verified_final_grade_and_exact_close_complete_the_cycle():
    data=inputs();data['games'][0]['status']='final'
    data['forecasts'][0].update(status='final',graded_result='win',actual=60)
    data['closes']={'decision':{'valid':True}}
    assert build(data,START+timedelta(hours=4))['status']=='complete'
    data['freezes']=dict(status='blocked',reason='source changed')
    assert build(data,START+timedelta(hours=4))['status']=='needs_attention'


def test_legacy_missing_ids_cannot_enter_new_evidence():
    data=inputs();data['documents'][0]['rows'][0]['forecast_id']=100
    doc=build(data)
    assert 'historical_missing_capture_admitted' in doc['blockers']
    assert doc['legacy_missing']['unexpected_admitted_ids']==[100]
    assert not doc['legacy_missing']['excluded_from_paper_evidence']


def test_empty_schedule_points_to_next_slate_and_never_completes():
    data=inputs();data['games']=[];data['next_game']=dict(start_ts_utc=START+timedelta(days=3))
    doc=build(data)
    assert doc['status']=='no_games' and doc['next_action']=='wait_for_pregame_window'
    assert doc['earliest_kickoff']==(START+timedelta(days=3)).isoformat()
