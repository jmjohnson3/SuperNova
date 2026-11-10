from copy import deepcopy
from datetime import datetime,timedelta,timezone
import numpy as np
import pandas as pd
import pytest

from nba_props import research_data as data,research_models as model,research_trial as trial,research_evaluation as evaluation

NOW=datetime(2026,10,31,4,tzinfo=timezone.utc)


def game():
    players=[];teams=[]
    for home,team,other in ((True,'BOS','NYK'),(False,'NYK','BOS')):
        part=[]
        for i in range(5):
            p=dict(game_id='g1',player_id=i+(10 if home else 20),player_name='Player '+str(i),game_date='2025-01-01',
                team=team,opponent=other,home=home,minutes=48.,pts=15. if home else 13.,reb=4.,ast=2.,
                fg3m=2.,fg3a=4.,fga=12.,fgm=6. if home else 5.,fta=2.,ftm=1.,tov=1.,oreb=1.,wl='W' if home else 'L')
            players.append(p);part.append(p)
        t=dict(part[0],player_id=None,player_name=None,minutes=240.)
        for s in data.STATS:t[s]=sum(p[s] for p in part)
        teams.append(t)
    return players,teams


def test_reconciled_attempts_and_conflicts_never_turn_into_zero():
    p,t=game();valid,bad=data.reconcile(p,t)
    assert len(valid)==1 and not bad and valid[0]['players'][0]['fg3a']==4
    for change in ('missing','wrong','duplicate','minutes','tie'):
        pp,tt=deepcopy(p),deepcopy(t)
        if change=='missing':pp.pop()
        if change=='wrong':pp[0]['fg3a']=1
        if change=='duplicate':pp.append(dict(pp[0],fg3a=5))
        if change=='minutes':pp[0]['minutes']=None
        if change=='tie':tt[0]['pts']=tt[1]['pts']
        assert not data.reconcile(pp,tt)[0]
    assert len(data.reconcile(p+[deepcopy(p[0])],t)[0])==1  # exact repeats are not conflicts


def histories(n=15):
    players=[];teams=[]
    for i in range(n):
        p,t=game()
        for r in p+t:
            r.update(game_id=str(i),game_date=str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=i)).date()))
            if 'player_id' in r and r['player_id'] is not None:r['actual_starter']=float(r['player_id']%2==0)
        players+=p;teams+=t
    return pd.DataFrame(players),pd.DataFrame(teams)


def test_current_and_future_outcomes_do_not_change_prior_features():
    p,t=histories();ps,ts=model.states(p,t)
    target=p[p.game_id=='10'].copy();a=model.features(target,ps,ts)
    changed=p.copy();changed.loc[changed.game_date>='2025-01-10',['pts','minutes','fg3a','fg3m','actual_starter']]=0
    ct=t.copy();ct.loc[ct.game_date>='2025-01-10','pts']=999
    ps2,ts2=model.states(changed,ct);b=model.features(target,ps2,ts2)
    pd.testing.assert_frame_equal(a[model.FEATURES],b[model.FEATURES])
    assert (a.history_date<=pd.Timestamp('2025-01-09')).all()


def test_shooting_small_sample_shrinks_toward_fixed_prior():
    p,t=histories(1);p['fg3m']=4.;p['fg3a']=4.
    ps,_=model.states(p,t)
    assert ps.fg3_pct.between(.36,.4).all()


class Fake:
    def __init__(self,rate=False):self.rate=rate
    def predict(self,x):return np.full(len(x),.1) if self.rate else 20+10*x.role.to_numpy()


def test_confirmed_roles_and_verified_cap_change_minutes_not_labels():
    p,t=histories();ps,ts=model.states(p,t);f=model.features(p[p.game_id=='10'],ps,ts)
    f['confirmed_starter']=True;f['minutes_cap']=24.
    fit={'minutes':Fake(),'rates':{s:Fake(True) for s in ('pts','reb','ast','fg3a')}}
    forecast=model.predict(fit,f)
    assert (forecast.pred_minutes==24).all() and np.allclose(forecast.pred_pts,2.4)
    assert np.allclose(forecast.pred_fg3m,forecast.pred_fg3a*f.fg3_pct)
    f['minutes_cap']=np.nan;f['confirmed_starter']=False
    assert (model.predict(fit,f).pred_minutes==20).all()


def test_calibration_monotone_and_count_probabilities_coherent():
    p=np.linspace(.05,.95,500);y=np.tile([0,1],250)
    cal=model.fit_calibration(p,y)
    assert cal['slope']>0 and np.all(np.diff(model.calibrate(p,cal))>=0)
    disp={'a':2.,'b':.1};lines=np.arange(.5,10,.5)
    probs=model.calibrate(model.over(3.,lines,disp),cal)
    assert np.all(np.diff(probs)<=1e-12)
    assert model.fit_calibration(p[:20],y[:20])['status']=='insufficient_quotes_identity_mapping'
    with pytest.raises(ValueError):model.fit_dispersion(np.ones(10),np.ones(10))


def test_separate_calibration_can_reduce_overconfidence_without_test_labels():
    p=np.r_[np.full(200,.95),np.full(200,.05)]
    y=np.r_[np.tile([1,1,1,0],50),np.tile([0,0,0,1],50)]
    cal=model.fit_calibration(p,y)
    assert .6<model.calibrate(.95,cal)<.9


def offer(**kw):return dict(dict(market='points',side='over',line=20.5,price=110,decided_at=NOW),**kw)
def forecast(**kw):return dict(dict(forecast_at=NOW-timedelta(minutes=1),tip_utc=NOW+timedelta(minutes=20),mu=24.,available=True),**kw)
def artifact():return dict(dispersion={'pts':{'a':1.,'b':0.}},calibration={'pts':{'slope':1.,'intercept':0.}},game_error={'margin':{'sd':10.},'total':{'sd':15.}})


def test_capture_rejects_future_stale_uncertain_and_started_forecasts():
    assert trial.score(offer(),forecast(),artifact(),NOW)['selected']
    for f in (None,forecast(forecast_at=NOW+timedelta(seconds=1)),forecast(forecast_at=NOW-timedelta(minutes=31)),forecast(tip_utc=NOW),forecast(available=False)):
        assert not trial.score(offer(),f,artifact(),NOW)['selected']


def test_game_side_signs_match_same_line_opposite_probabilities():
    a=trial.score(offer(market='spread',side='home',line=-5.5),forecast(mu=5.),artifact(),NOW)
    b=trial.score(offer(market='spread',side='away',line=5.5),forecast(mu=5.),artifact(),NOW)
    assert abs(a['probability']+b['probability']-1)<1e-12 and a['probability']<.5


def test_synthetic_historical_quote_timestamps_and_wrong_games_are_excluded():
    p,t=histories();ps,ts=model.states(p,t);f=model.features(p[p.game_id=='10'],ps,ts)
    for s in model.STATS:f['pred_'+s]=2.
    q=pd.DataFrame([dict(quote_id=1,event_id='e',player_name='Player 0',home_team='Boston Celtics',away_team='New York Knicks',
        stat='points',line=2.5,over_price=-110,under_price=-110,book='fanduel',
        fetched_at_utc='2025-01-12T00:00:00Z',updated_at_utc='2025-01-11T22:00:00Z',commence_time_utc='2025-01-11T23:00:00Z')])
    assert evaluation.match_props(f,q).empty


def test_evaluation_with_no_historical_offers_still_reports_stat_errors():
    p,t=histories();ps,ts=model.states(p,t);f=model.features(p[p.game_id=='10'],ps,ts)
    for s in ('minutes',*model.STATS,'fg3a'):f['pred_'+s]=2.
    g=model.game_features(t[t.game_id=='10'],ts)
    for s in ('margin','total'):g['pred_'+s]=g[s]
    a=artifact()
    for s in ('margin','total'):a['game_error'][s]['bias']=0.
    r=evaluation.evaluate(f,g,pd.DataFrame(),pd.DataFrame(),a)
    assert r['props']=={} and r['stat_forecasts']['fg3m']['rows']==10
    assert r['games']['total']['mae']==0 and not r['betting_authorized']


def test_busy_research_step_retries_then_reports_failure(monkeypatch):
    from unittest.mock import MagicMock
    conn=MagicMock();cursor=conn.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(trial.time,'sleep',lambda delay:None)
    cursor.fetchone.side_effect=[(False,),(True,)]
    trial.acquire_lock(conn)
    assert cursor.fetchone.call_count==2
    cursor.fetchone.side_effect=None;cursor.fetchone.return_value=(False,)
    with pytest.raises(TimeoutError):trial.acquire_lock(conn,wait_seconds=0)
