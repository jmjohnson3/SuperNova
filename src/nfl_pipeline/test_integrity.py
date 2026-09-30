from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
import asyncio

import joblib
import numpy as np
import pandas as pd
import pytest

from nfl_pipeline import integrity, forecast_store
from nfl_pipeline.clv_report import _close_quality
from nfl_pipeline.features import TARGET_STATS, _add_player_rolling
from nfl_pipeline.modeling.predict_player_props import _snapshot_players, _side_prob, _projection_distribution_summary
from nfl_pipeline.modeling.train_game_models import baseline_home_margin
from nfl_pipeline.modeling.train_validated_release import expanding_folds, feature_columns
from nfl_pipeline.modeling.release_models import ConservativeStatModel
from nfl_pipeline import run_daily_and_notify as daily


def test_spread_is_home_handicap_not_expected_margin():
    frame = pd.DataFrame({'market_spread_home':[-7.0,3.0,0.0]})
    assert baseline_home_margin(frame).tolist() == [7.0,-3.0,0.0]


def player_history():
    rows=[]
    for player, targets in [('a',[7,8,9]),('b',[3,4,5])]:
        for week, value in enumerate(targets,1):
            rows.append({**dict.fromkeys(TARGET_STATS,0.0), 'player_id':player,'player_name':player,
                         'position':'WR','team_abbr':'KC','opponent_abbr':'DEN','is_home':True,
                         'game_id':f'g{week}','season':2026,'week':week,
                         'game_date_et':date(2026,8,week*7),'targets':value})
    return pd.DataFrame(rows)


def game_frame():
    return pd.DataFrame([{'game_id':'future','season':2026,'week':4,'home_team_abbr':'KC',
                          'away_team_abbr':'DEN','start_ts_utc':datetime(2099,9,1,tzinfo=timezone.utc),
                          'market_total':45,'market_spread_home':-3}])


def test_offer_subset_cannot_change_raw_roles_or_rolling_features():
    h=player_history()
    full=_snapshot_players(h.copy(),game_frame(),date(2026,9,1),[],366,pd.DataFrame())
    offered=_snapshot_players(h.copy(),game_frame(),date(2026,9,1),[{'player_name':'a'}],366,pd.DataFrame())
    pd.testing.assert_frame_equal(full,offered)
    a=full.set_index('player_id').loc['a']
    assert a.targets_share_avg_5 == pytest.approx(2/3)
    assert a.team_game_number == 3
    assert a.targets_std_5 == pytest.approx(1.0)


def test_live_rolling_matches_historical_replay():
    h=player_history()
    live=_snapshot_players(h.copy(),game_frame(),date(2026,9,1),[],366,pd.DataFrame()).set_index('player_id')
    future=h.groupby('player_id').tail(1).copy()
    future['game_id']='future'; future['week']=4; future['game_date_et']=date(2026,9,1)
    replay=_add_player_rolling(pd.concat([h,future],ignore_index=True))
    replay=replay.loc[replay.game_id=='future'].set_index('player_id')
    for column in ['targets_avg_3','targets_avg_5','targets_std_5','targets_std_10','rest_days']:
        np.testing.assert_allclose(live[column],replay[column],equal_nan=True)


@pytest.mark.parametrize('minutes,status',[(121,'close_outside_two_hour_window'),(120,'valid_close'),
                                          (20,'valid_close'),(0,'close_after_kickoff'),(-1,'close_after_kickoff')])
def test_close_window_is_strictly_pregame(minutes,status):
    start=datetime(2026,9,20,17,tzinfo=timezone.utc)
    result=_close_quality(close_time=start-timedelta(minutes=minutes), lock_time=start-timedelta(hours=4),
                          commence_time=start,close_price=-110,line_available_at_close=True)
    assert result[0]==status
    assert result[2] is (status=='valid_close')


def test_equal_lock_close_and_missing_capture_do_not_prove_clv():
    at=datetime(2026,9,20,16,tzinfo=timezone.utc)
    result=_close_quality(close_time=at,lock_time=at,commence_time=at+timedelta(hours=1),
                          close_price=-110,line_available_at_close=True)
    assert result[0]=='stale_close_before_lock'
    assert not result[2]
    result=_close_quality(close_time=None,lock_time=at,commence_time=at+timedelta(hours=1),
                          close_price=None,line_available_at_close=None)
    assert result[0]=='missing_close'


def test_expanding_folds_purge_entire_weeks():
    df=pd.DataFrame([{'season':s,'week':w,'game_id':f'{s}-{w}'} for s in [2024,2025] for w in range(1,19) for _ in range(20)])
    folds=list(expanding_folds(df,2025))
    assert len(folds)==5
    for train,test in folds:
        assert set(train.game_id).isdisjoint(test.game_id)
        assert max(zip(train.season,train.week)) < min(zip(test.season,test.week))


def test_features_are_target_specific_and_exclude_unproven_context():
    X=pd.DataFrame(columns=['passing_yards_avg_5','carries_avg_5','targets_avg_5','market_total',
                            'team_implied_points','injury_score','rest_days','targets_share_avg_5'])
    assert feature_columns(X,'passing_yards') == ['passing_yards_avg_5','rest_days']
    assert feature_columns(X,'rushing_yards') == ['carries_avg_5','rest_days']
    assert feature_columns(X,'receiving_yards') == ['targets_avg_5','rest_days']


def test_atomic_write_failure_keeps_previous_artifact(tmp_path,monkeypatch):
    path=tmp_path/'active.json'
    integrity.atomic_json(path,{'version':'old'})
    def fail(*args):
        raise OSError('simulated interrupted replace')
    monkeypatch.setattr(integrity.os,'replace',fail)
    with pytest.raises(OSError):
        integrity.atomic_json(path,{'version':'new'})
    assert 'old' in path.read_text()
    assert list(tmp_path.iterdir()) == [path]


def test_release_estimator_has_importable_module_and_roundtrips(tmp_path):
    from sklearn.dummy import DummyRegressor
    X=pd.DataFrame({'passing_yards_avg_5':[100.,200.]})
    model=ConservativeStatModel(DummyRegressor(strategy='constant',constant=10).fit(X,[10,10]),'passing_yards_avg_5')
    assert model.__class__.__module__ != '__main__'
    path=tmp_path/'model.joblib'; integrity.atomic_joblib(path,model)
    np.testing.assert_allclose(joblib.load(path).predict(X),[105,205])


def test_td_baseline_and_empirical_yardage_pricing():
    assert _side_prob('passing_tds',2.0,.5,{}) == pytest.approx(1-np.exp(-2))
    dist={'kind':'empirical_oof_residual','residual_quantiles':[-10,0,10]}
    assert _side_prob('receiving_yards',50,55,{},dist) == pytest.approx(1/3)
    summary=_projection_distribution_summary('passing_tds',2,{},dist)
    assert summary['distribution_kind'] != 'empirical_oof_residual'


class Cursor:
    def __init__(self, upcoming):
        self.upcoming=upcoming; self.calls=[]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def execute(self,*args): self.calls.append(args)
    def fetchall(self): return [(g,) for g in self.upcoming]


class Connection:
    def __init__(self, upcoming): self.cur=Cursor(upcoming); self.commits=0
    def cursor(self): return self.cur
    def commit(self): self.commits+=1


def test_forecast_reruns_append_distinct_decisions(monkeypatch):
    conn=Connection(['future']); captured=[]
    monkeypatch.setattr(forecast_store.psycopg2.extras,'execute_values',lambda c,q,rows:captured.extend(rows))
    fields=['game_date_et','game_id','projection','price','prediction_key']
    row={'game_date_et':date(2099,9,1),'game_id':'future','projection':np.float64(42),'price':-110}
    forecast_store.save_forecasts(conn,'nfl_player_prop_predictions',[row],fields)
    first=row['prediction_key']
    row['price']=105
    forecast_store.save_forecasts(conn,'nfl_player_prop_predictions',[row],fields)
    assert row['prediction_key']!=first
    assert captured[0][3]==-110 and captured[1][3]==105
    assert conn.commits==2
    assert all('DELETE' not in call[0] for call in conn.cur.calls)


def test_forecast_after_start_is_rejected_before_insert():
    conn=Connection([])
    with pytest.raises(RuntimeError,match='after kickoff'):
        forecast_store.save_forecasts(conn,'nfl_player_prop_predictions',[{'game_date_et':date.today(),'game_id':'started'}],[])
    assert conn.commits==0


def test_discord_failure_is_not_silent(monkeypatch):
    monkeypatch.setattr(daily,'_webhook_url',lambda:'')
    with pytest.raises(RuntimeError,match='not configured'):
        asyncio.run(daily._post('test'))


def test_publisher_demotes_unlocked_micro(monkeypatch):
    from nfl_pipeline.publish_forecasts import display_rows
    conn=Connection([])
    conn.cur.fetchall=lambda:[({'tier':'micro_projection','player_name':'test'},False)]
    rows=display_rows(conn,date.today(),'prop')
    assert rows[0]['tier']=='paper'
    assert not rows[0]['ledger_locked']


def test_sgo_nested_teams_and_team_totals_never_replace_game_total():
    from nfl_pipeline.crawler_oddsapi import _canonical_event_from_sgo
    event={'eventID':'event','teams':{'home':{'names':{'long':'New York Jets'}},
                                   'away':{'names':{'long':'Green Bay Packers'}}},
           'status':{'startsAt':'2099-09-20T17:00:00Z'},'odds':{}}
    for entity,line in [('all',44.5),('home',20.5),('away',24.5)]:
        for side in ['over','under']:
            event['odds'][f'points-{entity}-game-ou-{side}']={'byBookmaker':{'draftkings':{'odds':'-110','overUnder':line,'available':True}}}
    event['odds']['points-all-reg-ou-over']={'byBookmaker':{'draftkings':{'odds':'-110','overUnder':40.5,'available':True}}}
    result=_canonical_event_from_sgo(event)
    assert result['home_team']=='New York Jets'
    assert result['away_team']=='Green Bay Packers'
    totals=result['bookmakers'][0]['markets'][0]
    assert len(totals['outcomes'])==2
    assert {r['point'] for r in totals['outcomes']}=={44.5}


def test_sgo_unavailable_prices_are_not_bookable():
    from nfl_pipeline.crawler_oddsapi import _canonical_event_from_sgo
    event={'eventID':'event','odds':{'points-all-game-ou-over':{'byBookmaker':{
        'fanduel':{'odds':'-110','overUnder':44.5,'available':False}}}}}
    assert _canonical_event_from_sgo(event) is None


def test_all_null_usage_values_have_explicit_numeric_types(monkeypatch):
    from nfl_pipeline import import_usage_context as usage
    conn=Connection([]); conn.cur.rowcount=1
    capture={}
    def execute(cur,sql,rows,**kwargs): capture.update(kwargs)
    monkeypatch.setattr(usage.psycopg2.extras,'execute_values',execute)
    usage._update_advanced_usage(conn,[(2026,1,'g','p','KC',*([None]*13))])
    assert capture['template'].count('::numeric')==13


@pytest.mark.parametrize('snaps,actions,expected', [(0,0,'pending_participation_review'),(12,0,'win'),(None,2,'win')])
def test_zero_stat_grading_requires_verified_participation(monkeypatch,snaps,actions,expected):
    from nfl_pipeline import grade_predictions as grading
    conn=Connection([])
    conn.cursor=lambda **kwargs:conn.cur
    conn.cur.fetchall=lambda:[{'id':1,'game_date_et':date(2026,9,10),'game_id':'g',
        'player_id':'p','player_name':'Player','stat':'receiving_yards','side':'under',
        'line':10.5,'price':-110,'receiving_yards':0,'offense_snaps':snaps,'offensive_actions':actions}]
    captured=[]
    monkeypatch.setattr(grading.psycopg2.extras,'execute_values',lambda c,q,rows,**kwargs:captured.extend(rows))
    assert grading.grade_player_prop_predictions(conn,grading.GradeConfig())==1
    assert captured[0][-2]==expected
    if expected=='pending_participation_review':
        assert captured[0][-1] is None


def test_prior_micro_is_distinct_from_current_betting_instruction():
    from nfl_pipeline.publish_forecasts import previously_locked_props
    from nfl_pipeline.modeling.predict_player_props import _row_price_summary
    conn=Connection([])
    conn.cur.fetchall=lambda:[({'tier':'micro_projection','price':-110,'book':'fanduel',
                              'link':'https://sportsbook.fanduel.com/offer'},)]
    row=previously_locked_props(conn,date.today())[0]
    assert 'ORDER BY l.ledger_id' in conn.cur.calls[-1][0]
    assert row['tier']=='locked_micro'
    assert 'historical quote' in _row_price_summary(row)
    assert 'Cur=' not in _row_price_summary(row)


@pytest.mark.parametrize('offers,predictions,buckets,expected',[
    ([],[],[],'waiting_for_props'),
    ([{}],[{}],[{'graded_rows':0}],'waiting_for_settlement'),
    ([{}],[{}],[{'graded_rows':2,'valid_clv_rows':0}],'waiting_for_valid_clv'),
    ([{}],[{}],[{'graded_rows':2,'valid_clv_rows':2}],'evidence_collected_not_bankroll_approval'),
])
def test_exact_line_report_does_not_claim_unearned_readiness(offers,predictions,buckets,expected):
    from nfl_pipeline.modeling.prop_exact_line_proof import evidence_status
    assert evidence_status(offers,predictions,buckets)==expected


def test_any_td_proof_does_not_approve_alternate_td_lines():
    from nfl_pipeline.modeling.predict_player_props import _micro_projection_status
    tier,reason=_micro_projection_status(stat='rushing_tds',
        metrics={'selection_target':'P(any TD), not count MAE','accepted_live':True,'projection_accepted':True},
        offer={'line':1.5,'over_price':100,'under_price':-110},probability=.6,ev=.2,
        market_probability=.5,link='https://example.com/bet',min_ev=.02)
    assert tier=='paper'
    assert 'alternate_line_distribution_unvalidated' in reason
