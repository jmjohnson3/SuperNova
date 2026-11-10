from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json

import pytest
import nfl_decision_board as b

NOW=datetime(2026,11,5,23,40,tzinfo=timezone.utc)


def game():
    return dict(game_id='g',home_team_abbr='BAL',away_team_abbr='JAX',start_ts_utc=NOW+timedelta(minutes=35),
        last_quote=NOW,quote_times={'spread':NOW},forecast=None,decisions=[])


def decision():
    return dict(id=111,delivery_status='sent',sent_at=NOW-timedelta(minutes=1),payload=dict(
        offer=dict(player='Receiver',side='over',line=45.5,market='receiving_yards',price=-110,
                   ev=.04,decided_at=NOW-timedelta(minutes=2)),
        decision=dict(selected=False,reason='model_nonpositive_ev')))


def test_empty_board_distinguishes_waiting_no_pick_and_outage():
    g=game()
    assert 'window open' in b.render(g,NOW)
    assert 'No current qualifying' in b.render(g,NOW)
    waiting=b.render(g,NOW-timedelta(hours=2))
    assert 'Waiting for the final-hour' in waiting
    assert 'Not evaluated yet' in waiting and 'No current qualifying' not in waiting
    collecting=b.render(g,NOW-timedelta(minutes=15))
    assert 'Not evaluated yet' in collecting and 'No current qualifying' not in collecting
    g['last_quote']=None
    missing=b.render(g,NOW)
    assert 'missing or delayed' in missing
    assert 'Unable to assess' in missing and 'No current qualifying' not in missing
    blocked=b.render(g,NOW,blocked='drift')
    assert 'Selections unavailable' in blocked and 'No current qualifying' not in blocked
    assert 'Game started' in b.render(g,NOW+timedelta(hours=1))


def test_model_veto_is_visible_but_stale_quotes_cannot_become_live_picks():
    g=game();g['decisions']=[decision()]
    text=b.render(g,NOW)
    assert 'Receiver OVER 45.5' in text and 'model veto' in text and 'No cash recommendation' in text
    assert 'original alert time applies' in text
    assert 'Receiver OVER' not in b.render(g,NOW+timedelta(minutes=4))
    assert 'Earlier selections remain' in b.render(g,NOW+timedelta(minutes=4))
    assert 'Receiver OVER' not in b.render(g,NOW,blocked='drift')


def test_many_picks_fit_discord_limit_and_do_not_reset_check_deadlines():
    g=game();g['decisions']=[dict(decision(),id=i) for i in range(30)]
    text=b.render(g,NOW)
    assert len(text)<=2000 and 'Additional offers' in text
    assert 'Only an original PAPER PRICE CHECK starts a check window' in text


class Response:
    status_code=200
    def __init__(self,payload):self.payload=payload
    def json(self):return self.payload


class Session:
    def __init__(self):self.posts=0;self.patches=0;self.content=''
    def get(self,url,**kwargs):
        if '/messages/' in url:return Response(dict(content=self.content))
        return Response(dict(id='webhook',guild_id='guild',channel_id='channel'))
    def post(self,url,**kwargs):
        self.posts+=1;self.content=kwargs['json']['content']
        return Response(dict(id='message',channel_id='channel'))
    def patch(self,url,**kwargs):
        self.patches+=1;self.content=kwargs['json']['content']
        return Response(dict(id='message',channel_id='channel'))


def test_update_same_game_message_and_never_duplicate_uncertain_send(tmp_path):
    s=Session();path=tmp_path/'state.json';g=game();g['content']=b.render(g,NOW)
    board=dict(games=[g]);b.publish(board,'https://example.test/webhook',state_path=path,session=s)
    b.publish(board,'https://example.test/webhook',state_path=path,session=s)
    assert s.posts==1 and s.patches==0
    g['content']+='\nUpdated.';b.publish(board,'https://example.test/webhook',state_path=path,session=s)
    assert s.posts==1 and s.patches==1
    state=json.loads(path.read_text());state['games']['g']=dict(status='pending_create');path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='uncertain'):
        b.publish(board,'https://example.test/webhook',state_path=path,session=s)
    assert s.posts==1
