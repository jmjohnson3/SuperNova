"""Research ledgers/context tests in a transaction that always rolls back."""
from datetime import timedelta
from zoneinfo import ZoneInfo
import os
import numpy as np
import psycopg2
import psycopg2.extras
import pytest

from sharp_pipeline.test_database_repairs import database
from nba_props import research_data as data,research_trial as trial,availability,live
from nba_props.test_research import NOW,game,histories,Fake,artifact

pytestmark=pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')
DOC=dict(sha256='e'*64,registered_at=(NOW-timedelta(minutes=5)).isoformat())


def save_game(conn,identity,at,players,teams):
    payload=dict(game_id='g1',game_date='2025-01-01',players=players,teams=teams,final_verified=True)
    with conn.cursor() as cur:
        cur.execute('INSERT INTO nbaprops.research_games VALUES(%s,%s,%s,%s,%s)',(identity,'g1','2025-01-01',at,psycopg2.extras.Json(payload)))


def test_result_versions_are_asof_and_append_only(database,monkeypatch,tmp_path):
    data.init(database);monkeypatch.setattr(data,'DATA',tmp_path)
    p,t=game();save_game(database,'old',NOW-timedelta(minutes=1),p,t)
    p2=[dict(x,fg3a=99.) for x in p];save_game(database,'later',NOW+timedelta(minutes=1),p2,t)
    first,_=data.history(database,NOW);again,_=data.history(database,NOW)
    assert (first.fg3a==4.).all() and first.equals(again)
    later,_=data.history(database,NOW+timedelta(minutes=2));assert (later.fg3a==99.).all()
    with database.cursor() as cur:
        cur.execute('SAVEPOINT immutability')
        with pytest.raises(psycopg2.errors.RaiseException):cur.execute('DELETE FROM nbaprops.research_games')
        cur.execute('ROLLBACK TO SAVEPOINT immutability')


def test_news_and_caps_require_observation_and_recording_before_forecast(database):
    data.init(database)
    with database.cursor() as cur:
        cur.execute(availability.SCHEMA_SQL)
        cur.execute('CREATE TABLE nbaprops.player_games(player_id bigint,player_name text,game_date date)')
        cur.execute("INSERT INTO nbaprops.player_games VALUES(1,'Test Player','2026-10-29')")
        for at,starter,roster in ((NOW-timedelta(minutes=1),True,'Active'),(NOW+timedelta(minutes=1),False,'Inactive')):
            cur.execute('''INSERT INTO nbaprops.lineup_snapshots(captured_at_utc,game_date,game_id,team,player_id,player_name,name_key,starter,roster_status,lineup_status)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(at,'2026-10-30','g1','BOS',1,'Test Player','test player',starter,roster,'Confirmed'))
        cur.execute('''INSERT INTO nbaprops.research_context(game_id,player_id,team,observed_at,recorded_at,minutes_cap,source,note)
            VALUES('g1',1,'BOS',%s,%s,20,'verified-source','minutes restriction')''',(NOW-timedelta(minutes=1),NOW+timedelta(minutes=1)))
    context=data.context(database,'g1','BOS',1,NOW)
    assert context['starter'] is True and context['availability']=='active' and context['minutes_cap'] is None
    context=data.context(database,'g1','BOS',1,NOW+timedelta(minutes=2))
    assert context['availability']=='out' and context['minutes_cap']==20


def test_forecast_ledger_never_rewrites_or_backfills(database):
    trial.init(database)
    f=dict(game_id='g1',player_id=1,stat='pts',forecast_at=NOW,tip_utc=NOW+timedelta(minutes=30),mu=20)
    assert trial.save_forecasts(database,DOC,[f])==1
    assert trial.save_forecasts(database,DOC,[dict(f,mu=99)])==0
    assert trial.save_forecasts(database,DOC,[dict(f,forecast_at=NOW-timedelta(days=1))])==0
    assert trial.save_forecasts(database,DOC,[dict(f,forecast_at=NOW+timedelta(hours=1))])==0


def test_report_workload_error_and_missing_checks_remain_separate(database,monkeypatch,tmp_path):
    trial.init(database);monkeypatch.setattr(trial,'ROOT',tmp_path)
    p,t=game();save_game(database,'r',NOW,p,t)
    offer=dict(sport='nba',book='fanduel',event_id='e1',market='points',player_norm='player0',player='Player 0',
        side='over',line=14.5,price=110,decided_at=NOW-timedelta(minutes=30),commence_time_utc=NOW,
        fair_probability=.5,home_team='Boston Celtics',away_team='New York Knicks')
    forecast=dict(game_id='g1',player_id=10,team='BOS',mu=20.,predicted_minutes=30.)
    payload=dict(offer=offer,decision=dict(forecast=forecast,selected=True,probability=.6,reason='positive_ev'))
    payload['decision'].update(raw_probability=.65,raw_selected=True)
    payload['frozen']=dict(forecast={'mu':22.},model_probability=.7,selected=True)
    with database.cursor() as cur:
        cur.execute('''INSERT INTO bets.nba_research_trial(experiment_sha256,event_id,market,player_norm,captured_at,payload)
            VALUES(%s,'e1','points','player0',%s,%s)''',(DOC['sha256'],NOW,psycopg2.extras.Json(trial.clean(payload))))
        missing=dict(offer=dict(offer,event_id='pending'),decision=dict(payload['decision'],forecast=dict(forecast,game_id='unresolved')))
        cur.execute('''INSERT INTO bets.nba_research_trial(experiment_sha256,event_id,market,player_norm,captured_at,payload)
            VALUES(%s,'pending','points','player0',%s,%s)''',(DOC['sha256'],NOW,psycopg2.extras.Json(trial.clean(missing))))
    monkeypatch.setattr(trial,'exact_close',lambda *args:dict(method='exact',probability=.55,ev=.155))
    result=trial.build_report(database,DOC);r=result['records'][0]
    assert r['result']=='win' and r['actual_minutes']==48
    assert abs(r['workload_error']+r['rate_error']-r['forecast_error'])<1e-9
    assert not r['timely_check'] and not result['betting_authorized']
    review=result['groups']['props']['execution_review']
    assert review['status']=='collecting' and review['checked_prices']==0
    assert review['decisions']==2 and review['pending_results']==1
    assert result['groups']['props']['vs_raw']['rows']==1 and result['groups']['props']['vs_frozen']['rows']==1
    assert r['absolute_error_gain']==2.


def test_context_does_not_assume_active_or_reuse_an_omitted_injury(database):
    data.init(database)
    with database.cursor() as cur:
        cur.execute(availability.SCHEMA_SQL)
        cur.execute('CREATE TABLE nbaprops.player_games(player_id bigint,player_name text,game_date date)')
        cur.execute("INSERT INTO nbaprops.player_games VALUES(1,'Test Player','2026-10-29')")
        cur.execute('''INSERT INTO nbaprops.lineup_snapshots(captured_at_utc,game_date,game_id,team,player_id,player_name,name_key,starter,lineup_status)
            VALUES(%s,'2026-10-30','g1','BOS',1,'Test Player','test player',true,'Confirmed')''',(NOW-timedelta(minutes=2),))
        for at,who,status in ((NOW-timedelta(minutes=1),'test player','Out'),(NOW,'different player','Out')):
            cur.execute('INSERT INTO nbaprops.injury_snapshots(captured_at_utc,team,player_name,name_key,status) VALUES(%s,%s,%s,%s,%s)',(at,'BOS',who,who,status))
    assert data.context(database,'g1','BOS',1,NOW-timedelta(seconds=30))['availability']=='out'
    assert data.context(database,'g1','BOS',1,NOW)['availability']=='unknown'


class ConstantGame:
    def __init__(self,value):self.value=value
    def predict(self,x):return np.full(len(x),self.value)


def test_projection_saves_components_only_when_completed_before_tip(database,monkeypatch):
    trial.init(database)
    with database.cursor() as cur:
        cur.execute(availability.SCHEMA_SQL);cur.execute(live.SCHEMA_SQL)
        cur.execute('CREATE TABLE nbaprops.player_games(player_id bigint,player_name text,game_date date)')
        tip=NOW+timedelta(minutes=30);day=NOW.astimezone(ZoneInfo('America/New_York')).date()
        cur.execute('''INSERT INTO nbaprops.game_projections(run_at_utc,game_date,game_id,tip_utc,home_team,away_team,pred_margin,pred_total)
            VALUES(%s,%s,'g1',%s,'BOS','NYK',6,220)''',(NOW-timedelta(minutes=1),day,tip))
        cur.execute('''INSERT INTO nbaprops.projections(run_at_utc,game_date,game_id,tip_utc,player_id,player_name,name_key,team,opponent,stat,mu,disp_a,disp_b)
            VALUES(%s,%s,'g1',%s,10,'Player 0','player 0','BOS','NYK','pts',999,1,0)''',(NOW-timedelta(minutes=1),day,tip))
    p,t=histories();monkeypatch.setattr(data,'history',lambda *args:(p,t))
    a=artifact();a['minutes_error']=dict(q10=-5,q90=5)
    a['fit']=dict(minutes=Fake(),rates={s:Fake(True) for s in ('pts','reb','ast','fg3a')},games={'margin':ConstantGame(5),'total':ConstantGame(210)})
    for s in ('margin','total'):a['game_error'][s]['bias']=0.
    times=iter([NOW,NOW+timedelta(seconds=2)])
    monkeypatch.setattr(data,'now',lambda:next(times))
    assert trial.project(database,DOC,a)['forecasts']==6
    saved=data.rows(database,'SELECT payload FROM bets.nba_research_forecasts ORDER BY stat')
    points=next(r['payload'] for r in saved if r['payload']['stat']=='pts')
    assert points['mu']==2.5 and points['predicted_minutes']==25 and points['mu']!=999
    assert points['predicted_three_attempts']==2.5 and not points['available']
    assert points['forecast_at']==(NOW+timedelta(seconds=2)).isoformat()
    total=next(r['payload'] for r in saved if r['payload']['stat']=='total')
    assert total['pred_home']+total['pred_away']==total['mu']==210
    times=iter([NOW,NOW+timedelta(minutes=31)])
    assert trial.project(database,DOC,a)['forecasts']==0


def test_first_capture_is_immutable_and_cannot_be_backfilled(database,monkeypatch):
    trial.init(database)
    o=dict(sport='nba',book='fanduel',event_id='e1',market='points',player_norm='player0',player='Player 0',
        parent_check_id=1,side='over',line=20.5,price=110,decided_at=NOW,commence_time_utc=NOW+timedelta(minutes=30),
        fair_probability=.55,ev=.05,method='exact',home_team='Boston Celtics',away_team='New York Knicks')
    monkeypatch.setattr(data,'now',lambda:NOW)
    monkeypatch.setattr(trial.parent,'candidates',lambda *args:[o])
    monkeypatch.setattr(trial.parent,'quote_times',lambda conn,row:row)
    monkeypatch.setattr(trial.parent,'forecast_for',lambda *args:None)
    assert trial.capture(database,DOC,artifact())['captured']==1
    f=dict(game_id='g1',player_id=10,player_name='Player 0',team='BOS',opponent='NYK',stat='pts',
        forecast_at=NOW-timedelta(minutes=1),tip_utc=NOW+timedelta(minutes=30),mu=24.,available=True)
    trial.save_forecasts(database,DOC,[f])
    assert trial.capture(database,DOC,artifact())['captured']==0
    first=data.rows(database,'SELECT payload FROM bets.nba_research_trial')[0]['payload']['decision']
    assert first['forecast'] is None and not first['selected']
    o['event_id']='e2'
    assert trial.capture(database,DOC,artifact())['captured']==1
    second=data.rows(database,"SELECT payload FROM bets.nba_research_trial WHERE event_id='e2'")[0]['payload']['decision']
    assert second['selected'] and second['raw_selected']
    o['event_id']='too_late'
    def late_lookup(*args):
        monkeypatch.setattr(data,'now',lambda:NOW+timedelta(minutes=6))
        return None
    monkeypatch.setattr(trial.parent,'forecast_for',late_lookup)
    assert trial.capture(database,DOC,artifact())['captured']==0


def test_manual_props_check_requires_sent_status_and_keeps_worse_price(database):
    with database.cursor() as cur:
        cur.execute('CREATE TABLE bets.nba_paper_price_alerts(alert_id bigint,delivery_status text,sent_at timestamptz)')
        cur.execute('''CREATE TABLE bets.nba_paper_price_checks(alert_id bigint,checked_at timestamptz,recorded_at timestamptz,
            status text,displayed_line numeric,displayed_price int,evidence_source text)''')
        cur.execute("INSERT INTO bets.nba_paper_price_alerts VALUES(1,'sent',%s)",(NOW,))
        cur.execute("INSERT INTO bets.nba_paper_price_checks VALUES(1,%s,%s,'available',20.5,-130,'manual_live')",(NOW+timedelta(minutes=1),NOW+timedelta(minutes=2)))
    o=dict(parent_check_id=1,line=20.5,price=110,commence_time_utc=NOW+timedelta(minutes=30))
    r=trial.manual_check(database,DOC,o)
    evidence=trial.check_evidence(dict(o,**r,close_method='exact',close_probability=.55))
    assert evidence['timely_check'] and evidence['exact_offer_available'] and evidence['checked_price_ev']<0
    assert r['displayed_price']==-130


def test_game_checks_require_identical_parent_offer(database):
    with database.cursor() as cur:cur.execute(trial.parent.SCHEMA)
    o=dict(sport='nba',book='fanduel',event_id='e1',market='total',player_norm='',side='over',line=220.5,price=110,
        decision_id=42,decided_at=NOW,commence_time_utc=NOW+timedelta(minutes=30))
    parent_doc=dict(sha256='p'*64);doc=dict(DOC,config={'parent_sha256':parent_doc['sha256']})
    identity=trial.parent.save(database,parent_doc,o,{},NOW)
    with database.cursor() as cur:
        cur.execute("INSERT INTO bets.model_research_delivery(id,delivery_status,sent_at) VALUES(%s,'sent',%s)",(identity,NOW))
        cur.execute("INSERT INTO bets.model_research_checks(id,checked_at,recorded_at,status,displayed_line,displayed_price) VALUES(%s,%s,%s,'available',220.5,-110)",
                    (identity,NOW+timedelta(minutes=1),NOW+timedelta(minutes=2)))
    assert trial.manual_check(database,doc,o)['displayed_price']==-110
    assert trial.manual_check(database,doc,dict(o,line=221.5))=={}
    assert trial.manual_check(database,doc,dict(o,decided_at=NOW+timedelta(seconds=1)))=={}
