from datetime import timedelta
from copy import deepcopy
from unittest.mock import MagicMock
import os

import pandas as pd
import pytest

from edge_research import market,trial,reporting,feeds
from nba_props import research_data as data
from sharp_pipeline.test_sharp_multi import NOW,event,bookmaker,lines
from sharp_pipeline.test_database_repairs import database

DOC=dict(sha256='d'*64,registered_at=(NOW-timedelta(minutes=1)).isoformat())


def pair():
    e=event([bookmaker(b,lines(home_spread=140 if b=='fanduel' else -110,away_spread=-170 if b=='fanduel' else -110))
        for b in ('fanduel','draftkings','pinnacle')])
    return e,next(o for o in market.offers(e,'nba',NOW) if o['market']=='spread')


def art():return dict(weights={'weights':{},'game_error':{}},rotation={})


def test_zero_model_weight_retains_consensus_even_when_model_disagrees():
    _,o=pair();r=trial.score(o,dict(probability=.01,context_verified=True),None,art())
    assert r['selection']['consensus']['side']=='home'
    assert r['probabilities']['blend_production']==r['probabilities']['consensus']
    assert r['weights']['blend_production']['weight']==0
    assert r['selection']['production']['side']=='away'
    o.update(market='points',participation_verified=False)
    assert not trial.score(o,dict(probability=.99,context_verified=True),None,art())['selection']['consensus']['selected']


def test_exact_close_does_not_rescue_newer_moved_reference_lines(monkeypatch):
    _,o=pair();tip=o['commence_time_utc'];at=tip-timedelta(minutes=5)
    old=event([bookmaker(b,lines()) for b in ('fanduel','draftkings','pinnacle')])
    new=deepcopy(old)
    for b in new['bookmakers']:
        for m in b['markets']:
            m['last_update']=at.isoformat()
            if m['key']=='spreads':
                for outcome in m['outcomes']:outcome['point']+=-1 if outcome['point']<0 else 1
    monkeypatch.setattr(feeds,'snapshots',lambda *args:[dict(at=NOW,event=old),dict(at=at,event=new)])
    result=reporting.exact_close(None,o,dict(tip=tip),{})
    assert result['probability'] is None and result['method']=='missing_exact_consensus'


def test_public_feed_checks_do_not_satisfy_manual_execution(monkeypatch):
    _,o=pair();o.update(market='points',player_norm='test player',player='Test Player',line=20.5)
    selection=market.selection(o,.7)
    alert=dict(o,side=selection['side'],price=selection['price'],sent_at=NOW)
    feed=dict(alert=alert,source='automated_fanduel_public_feed',timely=True,status='available',reason='exact_quote',
        line=selection['line'],price=-130,observed_at=NOW+timedelta(seconds=30))
    monkeypatch.setattr(data,'rows',lambda *args:[dict(alert_id=1,delivery_status='sent',sent_at=NOW,feed=feed)])
    result=reporting.execution(None,o,selection,dict(probability=.55,method='observed_exact_line'))
    assert result['feed_checked_price_ev']<0 and result['feed_timely']
    assert not result['timely_check'] and result['checked_price_ev'] is None


def test_expected_starting_five_is_not_treated_as_entire_rotation(monkeypatch):
    from nba_props.test_research import histories
    monkeypatch.setattr(data,'now',lambda:NOW)
    monkeypatch.setattr(data,'history',lambda *args:histories())
    monkeypatch.setattr(trial.base,'attach_past_roles',lambda p,*args:p)
    game=dict(game_id='g1',game_date=NOW.date(),home_team='BOS',away_team='NYK',tip_utc=NOW+timedelta(minutes=30))
    def rows(conn,sql,params):
        return [game] if 'game_projections' in sql else [dict(player_id=i) for i in range(5)]
    monkeypatch.setattr(data,'rows',rows)
    result=trial.project(None,DOC,{},save=False)
    assert result['forecasts']==0 and result['status']=='no_verified_rosters'
    assert all(r['reason']=='incomplete_roster_wait_for_bench' for r in result['excluded'])


def test_unknown_checks_and_correlated_rows_cannot_pass_review():
    rows=[]
    for i in range(100):
        r=dict(event_id='one game',day=NOW.date(),outcome=1.,close_probability=.6,consensus=.6)
        for label in ('consensus','production','rotation_mean','rotation_mixture','blend_production','blend_rotation'):
            r[label]=.6;r[label+'_selected']=True;r[label+'_execution']={};r[label+'_close_ev']=.1;r[label+'_paper_profit']=1.
        rows.append(r)
    result=reporting.summary(pd.DataFrame(rows))
    assert result['games']==1 and not result['betting_authorized']
    for v in result['variants'].values():
        assert v['manual_checked_closes']==0 and not v['review_only']
        assert not v['gates']['positive_execution_bounds']


def test_cost_estimate_requires_coverage_and_does_not_raise_stakes():
    f=pd.DataFrame([dict(event_id=str(i),day=NOW.date()+timedelta(days=i//5),opportunity_ev=.03) for i in range(150)])
    result=reporting.economics(f,30)
    assert result['projected_monthly_gross']==pytest.approx(4.5)
    assert result['projected_monthly_net']==pytest.approx(-25.5)
    assert result['comparison_stake_usd']==1 and result['actual_profit'] is None
    f.loc[:30,'opportunity_ev']=None
    assert reporting.economics(f,30)['projected_monthly_net'] is None
    assert reporting.economics(f,5)['projected_monthly_net'] is None


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')
def test_unqualified_observation_can_later_qualify_without_rewriting_decision(database,monkeypatch):
    trial.init(database);e,o=pair();clock=[NOW]
    monkeypatch.setattr(data,'now',lambda:clock[0])
    packet=dict(id='first',at=NOW,event=e)
    monkeypatch.setattr(feeds,'snapshots',lambda *args:[packet])
    monkeypatch.setattr(trial,'verified_schedule',lambda *args:dict(game_id='g1',tip=NOW+timedelta(minutes=30)))
    monkeypatch.setattr(feeds,'native_forecast',lambda *args:dict(probability=None,forecast=None,context_verified=False))
    monkeypatch.setattr(trial,'rotation_forecast',lambda *args:None)
    e['bookmakers']=e['bookmakers'][:2]
    trial.capture(database,DOC,art(),['nba'])
    assert not data.rows(database,'SELECT * FROM bets.edge_market_decisions')
    packet.update(id='second',at=NOW+timedelta(minutes=1));clock[0]=packet['at']
    e['bookmakers'].append(bookmaker('pinnacle',lines()))
    trial.capture(database,DOC,art(),['nba'])
    rows=data.rows(database,'SELECT payload FROM bets.edge_market_decisions')
    assert len(rows)==2  # one spread and one total; h2h deliberately unsupported
    assert all(r['payload']['decision']['native']['probability'] is None for r in rows)
    monkeypatch.setattr(feeds,'native_forecast',lambda *args:dict(probability=.99,forecast=None,context_verified=True))
    trial.capture(database,DOC,art(),['nba'])
    again=data.rows(database,'SELECT payload FROM bets.edge_market_decisions')
    assert again==rows
    assert len(data.rows(database,'SELECT * FROM bets.edge_market_observations'))==4


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')
def test_capture_rechecks_completion_deadline(database,monkeypatch):
    trial.init(database);e,_=pair();clock=[NOW]
    monkeypatch.setattr(data,'now',lambda:clock[0])
    monkeypatch.setattr(feeds,'snapshots',lambda *args:[dict(id='p',at=NOW,event=e)])
    monkeypatch.setattr(trial,'verified_schedule',lambda *args:dict(game_id='g',tip=NOW+timedelta(minutes=30)))
    monkeypatch.setattr(trial,'rotation_forecast',lambda *args:None)
    def slow(*args):
        clock[0]=NOW+timedelta(minutes=6)
        return dict(probability=None,forecast=None,context_verified=False)
    monkeypatch.setattr(feeds,'native_forecast',slow)
    assert trial.capture(database,DOC,art(),['nba'])['late']>0
    assert not data.rows(database,'SELECT * FROM bets.edge_market_decisions')


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')
def test_report_retains_pending_denominators_and_never_authorizes(database,monkeypatch,tmp_path):
    import psycopg2.extras
    trial.init(database);_,o=pair();d=trial.score(o,dict(probability=.6,forecast={'pred_margin':2.},context_verified=True),None,art())
    monkeypatch.setattr(trial,'ROOT',tmp_path)
    monkeypatch.setattr(feeds,'nba_result_index',lambda *args:{})
    monkeypatch.setattr(feeds,'outcome',lambda conn,o,*args:dict(outcome=1. if o['event_id']=='e1' else None,actual=10. if o['event_id']=='e1' else None,result_status='test_fixture'))
    monkeypatch.setattr(reporting,'exact_close',lambda *args:dict(probability=.6,method='observed_exact_line'))
    monkeypatch.setattr(reporting,'execution',lambda *args:dict(timely_check=False,exact_offer_available=False,checked_price_ev=None,feed_status='unknown',feed_timely=False,feed_checked_price_ev=None))
    for eid in ('e1','pending'):
        payload=trial.frozen.clean(dict(offer=dict(o,event_id=eid),decision=d,schedule={'tip':o['commence_time_utc']}))
        with database.cursor() as cur:
            cur.execute('''INSERT INTO bets.edge_market_decisions(experiment_sha256,sport,event_id,market,player_norm,captured_at,payload)
                VALUES(%s,'nba',%s,'spread','',%s,%s)''',(DOC['sha256'],eid,NOW,psycopg2.extras.Json(payload)))
    result=reporting.build(database,DOC);group=result['groups']['nba:spread']
    assert group['decisions']==2 and group['settled']==1 and group['pending']==1
    assert group['native_stat_accuracy']['mae']==8.
    assert not result['automatic_promotion'] and not result['betting_authorized']
    assert group['variants']['consensus']['manual_checked_closes']==0
