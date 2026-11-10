"""Direct-feed checks must never manufacture manual execution evidence."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from email.utils import format_datetime
import json

import pytest
import requests

from nba_props import fanduel_live as live

NOW=datetime(2026,10,30,23,tzinfo=timezone.utc)


def alert(**changes):
    return dict(dict(alert_id=1,book='fanduel',player='Test Player',market='points',side='under',line=22.5,
        price=110,away_team='Chicago Bulls',home_team='Milwaukee Bucks',
        native_tip=NOW+timedelta(minutes=20),commence_time_utc=NOW+timedelta(minutes=30),
        sent_at=NOW-timedelta(seconds=30)),**changes)


def page():
    return dict(attachments=dict(events={'1':dict(eventId=1,name='Chicago Bulls @ Milwaukee Bucks',
        openDate=(NOW+timedelta(minutes=30)).isoformat(),inPlay=False)},
        markets={'m1':dict(eventId=1,marketId='m1',marketName='Test Player - Points',
            marketStatus='OPEN',inPlay=False,runners=[dict(runnerName='Test Player Under',selectionId=123,
                handicap=22.5,runnerStatus='ACTIVE',result=dict(type='UNDER'),
                winRunnerOdds=dict(americanDisplayOdds=dict(americanOddsInt=-130)),
                previousWinRunnerOdds=dict(americanDisplayOdds=dict(americanOddsInt=120)))])}),
        layout=dict(tabs={'168':dict(title='Player Points',cards=[dict(id='c1')])},
            cards={'c1':dict(attachmentsFullyLoaded=True,display=[dict(rows=[dict(eventId=1,marketIds=['m1'])])])}))


def market(body):return body['attachments']['markets']['m1']
def runner(body):return market(body)['runners'][0]
def quote(body=None,**changes):return live.quote_from_page(alert(**changes),'1',body or page(),NOW)


def test_exact_live_price_can_be_worse_than_alert_and_previous_price():
    q=quote()
    assert (q['status'],q['line'],q['price'])==('available',22.5,-130)
    assert q['market_id']=='m1' and q['selection_id']=='123'


def test_alternate_uses_named_threshold_not_placeholder_handicap():
    p=page();market(p)['marketName']='Test Player - Alt Points'
    runner(p).update(runnerName='Test Player Under 22.5',handicap=0)
    assert quote(p)['line']==22.5
    runner(p)['runnerName']='Test Player Under'
    assert quote(p)['status']=='unknown'


@pytest.mark.parametrize('mutate,reason',[
    (lambda p:p['layout']['cards']['c1'].update(attachmentsFullyLoaded=False),'market_tab_incomplete'),
    (lambda p:p['attachments']['markets'].clear(),'market_attachments_missing'),
    (lambda p:p['attachments']['events']['1'].update(inPlay=True),'game_started'),
    (lambda p:p['attachments']['events']['1'].update(inPlay=None),'pregame_status_unverified'),
    (lambda p:market(p).update(eventId=2),'market_event_mismatch'),
    (lambda p:runner(p)['result'].update(type='OVER'),'runner_side_mismatch'),
    (lambda p:runner(p).update(runnerName='Test Player Under 23.5'),'runner_line_mismatch'),
    (lambda p:runner(p).update(winRunnerOdds=None),'quote_status_or_price_unverified'),
    (lambda p:runner(p).update(selectionId=None),'quote_status_or_price_unverified'),
    (lambda p:market(p).update(runners=None),'response_structure_unverified'),
])
def test_ambiguous_response_stays_unknown(mutate,reason):
    p=page();mutate(p)
    q=quote(p)
    assert q['status']=='unknown' and q['reason']==reason
    assert q['price'] is None and q['line'] is None


def test_no_matching_team_does_not_match_two_unknown_teams():
    p=page();p['attachments']['events']['1']['name']='Unknown Away @ Unknown Home'
    assert quote(p,away_team='Other Away',home_team='Other Home')['status']=='unknown'
    assert live.team_code('CHI')==live.team_code('Chicago Bulls')=='CHI'
    assert quote(native_tip=None)['reason']=='native_schedule_unverified'
    assert quote(native_tip=NOW)['reason']=='game_started'


def test_missing_offer_requires_complete_catalog_and_different_line_stays_different():
    p=page();market(p)['marketName']='Different Player - Points'
    assert quote(p)['status']=='not_listed'
    p=page();runner(p)['handicap']=23.5
    q=quote(p)
    assert (q['status'],q['line'],q['price'])==('changed_line',23.5,-130)
    p['layout']['cards']['c1']['display']=[]
    assert quote(p)['status']=='unknown'


def test_conflicting_duplicates_never_choose_better_price():
    p=page();other=deepcopy(runner(p));other['winRunnerOdds']['americanDisplayOdds']['americanOddsInt']=120
    market(p)['runners'].append(other)
    assert quote(p)['reason']=='conflicting_exact_quotes'


def test_suspended_has_no_executable_odds():
    p=page();market(p)['marketStatus']='SUSPENDED'
    q=quote(p)
    assert q['status']=='suspended' and q['price'] is None and q['line'] is None


class Response:
    status_code=200
    headers={'Date':format_datetime(NOW,usegmt=True),'Content-Type':'application/json'}
    def json(self):return page()


class Session:
    def __init__(self,response):self.response=response;self.calls=[]
    def get(self,*args,**kw):self.calls.append((args,kw));return self.response


def test_client_saves_fresh_response_without_credentials(tmp_path):
    s=Session(Response());client=live.Client('co',session=s,clock=lambda:NOW,folder=tmp_path)
    body,receipt=client.get('event-page',{'eventId':'1','tab':'player-points'})
    saved=json.loads(next(tmp_path.iterdir()).read_text())
    assert receipt['body_sha256']==saved['meta']['body_sha256']
    assert '_ak' not in saved['meta']['parameters'] and saved['body']==body
    assert s.calls[0][1]['allow_redirects'] is False
    assert client.calls==1


@pytest.mark.parametrize('headers,reason',[
    ({'Age':'60'},'cached_response_stale'),({'Age':'not-a-number'},'cache_age_unverified'),
    ({'Date':format_datetime(NOW-timedelta(minutes=1),usegmt=True)},'server_response_stale'),
    ({'Date':''},'response_format_unverified'),({'Content-Type':'text/html'},'non_json_response'),
])
def test_response_freshness_and_format_fail_closed(tmp_path,headers,reason):
    response=Response();response.headers=dict(response.headers,**headers)
    client=live.Client('co',session=Session(response),clock=lambda:NOW,folder=tmp_path)
    with pytest.raises(live.FeedError,match=reason):client.get('event-page',{})


@pytest.mark.parametrize('code,reason',[(302,'http_302'),(403,'access_blocked'),(429,'rate_limited')])
def test_no_redirect_or_block_bypass(tmp_path,code,reason):
    response=Response();response.status_code=code
    client=live.Client('co',session=Session(response),clock=lambda:NOW,folder=tmp_path)
    with pytest.raises(live.FeedError,match=reason):client.get('event-page',{})
    assert not list(tmp_path.iterdir())


def test_network_failure_is_unknown_and_never_disappeared(tmp_path):
    class Failed:
        def get(self,*args,**kw):raise requests.Timeout('do not expose URL')
    client=live.Client('co',session=Failed(),clock=lambda:NOW,folder=tmp_path)
    result=live.observe(alert(),client,clock=lambda:NOW)
    assert result['status']=='unknown' and result['reason']=='network_Timeout'
    assert 'observed_at' not in result and result['source']=='automated_fanduel_public_feed'


def test_requests_must_follow_original_alert_and_budget_is_bounded(tmp_path):
    client=live.Client('co',session=Session(Response()),clock=lambda:NOW,folder=tmp_path)
    assert live.observe(alert(sent_at=NOW+timedelta(seconds=1)),client,clock=lambda:NOW)['reason']=='snapshot_before_alert'
    client.calls=live.MAX_REQUESTS
    with pytest.raises(live.FeedError,match='request_budget_exhausted'):client.get('event-page',{})
