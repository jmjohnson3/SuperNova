from datetime import datetime, timedelta, timezone

import pytest

from nfl_pipeline.receiving_close_audit import diagnose, raw_sgo_evidence

START = datetime(2026,10,30,0,15,tzinfo=timezone.utc)
QUOTE = dict(provider='sportsgameodds',event_id='e',bookmaker_key='fanduel',
             player_name_norm='christian watson',stat='receiving_yards',market_key='player_reception_yds')
ROW = dict(quote=QUOTE,decision_id='d',forecast_id=1,player_name='C.Watson',book='fanduel',stat='receiving_yards',
           side='under',line=69.5,price=-114,market_probability=.5,
           captured_at=START-timedelta(days=2),start_ts_utc=START)


def quote(**changes):
    return dict(QUOTE,id=2,line=69.5,over_price=-114,under_price=-114,
                commence_time_utc=START,fetched_at_utc=START-timedelta(minutes=5),snapshot_role='close',**changes)


def test_valid_close_and_one_sided_reason():
    q=quote()
    assert diagnose(ROW,[q],[],START)['valid']
    q['under_price']=None
    r=diagnose(ROW,[q],[],START)
    assert not r['valid'] and r['reason']=='selected_side_price_missing'
    q.update(over_price=None,under_price=-114)
    assert diagnose(ROW,[q],[],START)['reason']=='opposing_price_or_pair_invalid'


def test_another_line_does_not_repair_the_captured_line():
    q=dict(quote(),line=70.5)
    r=diagnose(ROW,[q],[],START)
    assert not r['valid'] and r['reason']=='exact_line_not_observed'
    assert r['paired_lines']==[70.5]


@pytest.mark.parametrize('change', [dict(provider='another'),dict(event_id='another'),
    dict(fetched_at_utc=START),dict(fetched_at_utc=START-timedelta(minutes=21)),dict(snapshot_role='lock')])
def test_wrong_identity_timing_or_role_cannot_supply_a_close(change):
    assert diagnose(ROW,[dict(quote(),**change)],[],START)['reason']=='no_matching_market_in_close_window'


def test_future_selection_is_pending_not_failed():
    assert diagnose(ROW,[],[],START-timedelta(hours=1))['reason']=='awaiting_kickoff'


def snapshot():
    alt=dict(overUnder='69.5',odds='-114',available=False,lastUpdatedAt='2026-10-27T20:20:07Z')
    book=dict(overUnder='70.5',odds='-113',available=True,altLines=[alt])
    odd=dict(playerName='Christian Watson',byBookmaker=dict(fanduel=book))
    event=dict(eventID='e',odds={'receiving_yards-player-game-ou-under':odd})
    return dict(id=44,payload_sha256='hash',fetched_at_utc=START-timedelta(minutes=5),payload=dict(data=[event]))


def test_inactive_raw_quote_explains_gap_but_never_counts_as_evidence():
    raw=snapshot()
    evidence=raw_sgo_evidence(ROW,[raw])
    assert len(evidence)==1 and evidence[0]['available'] is False
    r=diagnose(ROW,[dict(quote(),under_price=None)],[raw],START)
    assert r['provider_marks_selected_side_unavailable'] and not r['valid']
    assert r['live_sportsbook_availability']=='unknown'
    assert not raw_sgo_evidence(ROW,[dict(raw,fetched_at_utc=START)])
    raw['payload']['data'][0]['eventID']='wrong'
    assert not raw_sgo_evidence(ROW,[raw])
