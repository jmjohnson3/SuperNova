from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import model_research
import nfl_spread_collection as c

NOW = datetime(2026, 11, 5, 23, 40, tzinfo=timezone.utc)
GAME = dict(game_id='g', home='BAL', away='JAX', start=NOW+timedelta(minutes=35))
DOC = dict(sha256='test', registered_at=(NOW-timedelta(minutes=10)).isoformat())


def event():
    books = []
    for name, home, away in [('fanduel', 110, -130), ('pinnacle', -130, 110), ('draftkings', -125, 105)]:
        books.append(dict(key=name, last_update=NOW.isoformat(), markets=[dict(key='spreads',
            last_update=(NOW-timedelta(seconds=30)).isoformat(), outcomes=[
                dict(name='Baltimore Ravens', point=-3.5, price=home),
                dict(name='Jacksonville Jaguars', point=3.5, price=away)])]))
    return dict(id='event', home_team='Baltimore Ravens', away_team='Jacksonville Jaguars',
                commence_time=GAME['start'].isoformat(), bookmakers=books)


def test_only_final_hour_pregame_and_at_most_one_poll_per_five_minutes():
    assert c.due_games([GAME], {}, NOW) == [GAME]
    assert not c.due_games([GAME], dict(last_attempt=NOW.isoformat()), NOW)
    assert not c.due_games([dict(GAME,start=NOW)], {}, NOW)
    assert not c.due_games([dict(GAME,start=NOW+timedelta(minutes=61))], {}, NOW)


def test_identity_requires_both_teams_exact_kickoff_and_unambiguous_game():
    e = event()
    assert c.match_event(e, [GAME]) == GAME
    assert c.match_event(e, [GAME,GAME]) is None
    assert c.match_event(dict(e,home_team='Denver Broncos'), [GAME]) is None
    assert c.match_event(dict(e,commence_time=(GAME['start']+timedelta(minutes=1)).isoformat()), [GAME]) is None


def test_market_clock_beats_book_clock_and_missing_opponent_is_rejected():
    e = event()
    e['bookmakers'][0]['markets'][0]['last_update']=(NOW-timedelta(hours=1)).isoformat()
    rows = c.observations(e,GAME,NOW,DOC)
    assert {q['validation_error'] for q in rows if q['book']=='fanduel'} == {'source_quote_stale'}
    e['bookmakers'][1]['markets'][0]['outcomes'].pop()
    assert next(q for q in c.observations(e,GAME,NOW,DOC) if q['book']=='pinnacle')['validation_error']=='missing_opposing_spread'


def test_conflicting_duplicates_are_not_resolved_arbitrarily():
    e=event();m=e['bookmakers'][0]['markets'][0]
    m['outcomes'].append(dict(m['outcomes'][0],price=150))
    assert not any(q['book']=='fanduel' for q in c.observations(e,GAME,NOW,DOC))
    e=event();e['bookmakers'].append(deepcopy(e['bookmakers'][0]))
    assert not any(q['book']=='fanduel' for q in c.observations(e,GAME,NOW,DOC))


def test_new_quotes_feed_existing_frozen_spread_selection_without_touching_cash(monkeypatch):
    quotes=c.observations(event(),GAME,NOW,DOC)
    assert len(quotes)==6 and all(q['validation_error'] is None for q in quotes)
    for q in quotes:
        assert q['observation_id']==c.watch._digest({k:v for k,v in q.items() if k!='observation_id'})
    def rows(conn,sql,params=()):
        if 'quote_observations' in sql:return [dict(payload=q) for q in quotes]
        return [dict(home_team_abbr='BAL',away_team_abbr='JAX')]
    monkeypatch.setattr(model_research,'rows',rows)
    offers=model_research.nfl_spread_candidates(None,NOW-timedelta(minutes=5),NOW)
    assert len(offers)==1 and offers[0]['side']=='home' and offers[0]['line']==-3.5
    assert model_research.eligible(offers[0],DOC['registered_at'],NOW)
    decision=model_research.assess(offers[0],dict(run_at_utc=NOW-timedelta(minutes=2),pred_margin=5.,pred_total=42,
        identity_verified=True,model_version='frozen'))
    assert decision['selected'] and decision['directional_margin']==1.5


def test_daily_credit_cap_stops_before_network(monkeypatch,tmp_path):
    monkeypatch.setattr(c.watch,'_games',lambda now:[GAME])
    path=tmp_path/'state.json'
    path.write_text(json.dumps(dict(daily_calls={NOW.astimezone(c.watch._ET).date().isoformat():48})))
    class NoNetwork:
        def get(self,*a,**kw):raise AssertionError('No request allowed')
    assert c.once(None,DOC,now=NOW,session=NoNetwork(),state_path=path)['status']=='daily_credit_cap'


def test_credit_floor_and_unknown_balance_never_buy_quotes(monkeypatch,tmp_path):
    monkeypatch.setattr(c.watch,'_games',lambda now:[GAME])
    monkeypatch.setattr(c,'_saved_windows_env',lambda name:'test-key')
    class Session:
        def __init__(self,headers):self.headers=headers;self.calls=0
        def get(self,url,**kw):
            self.calls+=1
            assert url.endswith('/events')
            class Response:
                status_code=200
            response=Response();response.headers=self.headers
            return response
    for headers,status in [({},'credit_balance_unknown'),({'x-requests-remaining':'1000'},'credit_floor')]:
        session=Session(headers)
        assert c.once(None,DOC,now=NOW,session=session,state_path=tmp_path/'state.json')['status']==status
        assert session.calls==1


def test_timeout_counts_paid_attempt_and_does_not_retry_immediately(monkeypatch,tmp_path):
    import requests
    import pytest
    monkeypatch.setattr(c.watch,'_games',lambda now:[GAME])
    monkeypatch.setattr(c,'_saved_windows_env',lambda name:'test-key')
    class Session:
        def __init__(self):self.calls=0
        def get(self,url,**kw):
            self.calls+=1
            if url.endswith('/odds'):raise requests.Timeout()
            class Response:
                status_code=200
                headers={'x-requests-remaining':'5000'}
                def json(self):return [event()]
            return Response()
    session=Session();path=tmp_path/'state.json'
    with pytest.raises(requests.Timeout):c.once(None,DOC,now=NOW,session=session,state_path=path)
    assert c.once(None,DOC,now=NOW,session=session,state_path=path)['status']=='not_due'
    assert session.calls==2
    assert list(json.loads(path.read_text())['daily_calls'].values())==[1]
