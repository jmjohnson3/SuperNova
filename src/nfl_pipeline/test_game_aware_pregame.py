import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nfl_pipeline import game_scope, run_pregame as runner, forecast_store
from nfl_pipeline.test_integrity import Connection

NOW = datetime(2099, 9, 27, 15, 30, tzinfo=timezone.utc)


def game(name='early', minutes=90):
    start = NOW + timedelta(minutes=minutes)
    return dict(game_id=name, start_ts_utc=start,
                game_date_et=start.astimezone(ZoneInfo('America/New_York')).date())


def state():
    return dict(contract=runner.CONTRACT, games={})


def successful(games, *_):
    return 0, dict(status='ok', pregame=True, game_ids=[g['game_id'] for g in games],
        publications=[dict(game_id=g['game_id'],sent_at=NOW.isoformat()) for g in games])


def test_due_wave_only_and_success_is_not_repeated():
    games = [game(),game('same_wave',85),game('afternoon',270),game('night',525),game('monday',1965)]
    history = state(); calls=[]; writes=[]
    def execute(games,*args):
        calls.append([g['game_id'] for g in games])
        return successful(games)
    result = runner.process(games,history,execute=execute,save=lambda *a:writes.append(a),clock=lambda:NOW)
    assert result['status']=='completed' and calls==[['early','same_wave']]
    assert len(writes)==2
    result = runner.process(games,history,execute=execute,save=lambda *a:None,clock=lambda:NOW)
    assert result['status']=='idle' and not result['pipeline_invoked'] and len(calls)==1


@pytest.mark.parametrize('minutes,action', [(91,'waiting_for_window'),(90,'due'),(21,'due'),
    (20,'missed_pregame_window'),(-5,'missed_pregame_window')])
def test_window_boundaries(minutes,action):
    _, rows = runner.plan([game(minutes=minutes)],state(),NOW)
    assert rows[0]['action']==action


@pytest.mark.parametrize('local_start', ['2099-09-28T18:15:00','2099-12-28T18:15:00','2099-09-30T11:00:00'])
def test_any_weekday_and_dst_use_actual_kickoff(local_start):
    kickoff=datetime.fromisoformat(local_start).replace(tzinfo=ZoneInfo('America/Denver')).astimezone(timezone.utc)
    g=game(); g['start_ts_utc']=kickoff
    groups, rows=runner.plan([g],state(),kickoff-timedelta(minutes=85))
    assert len(groups)==1 and rows[0]['action']=='due'
    assert datetime.fromisoformat(rows[0]['target_at'])==kickoff-timedelta(minutes=90)


def test_no_games_never_executes_or_writes_state():
    def forbidden(*a):
        pytest.fail('idle runner performed work')
    result=runner.process([],state(),execute=forbidden,save=forbidden,clock=lambda:NOW)
    assert result['status']=='idle' and not result['pipeline_invoked']


def test_known_failure_retries_but_uncertain_delivery_does_not():
    g=game(); history=state()
    result=runner.process([g],history,execute=lambda *a:(1,dict(steps=[dict(step='NFL Fresh Lock Quotes')])),
        save=lambda *a:None,clock=lambda:NOW)
    assert result['status']=='failed'
    assert runner.plan([g],history,NOW)[1][0]['action']=='retry_backoff'
    assert runner.plan([g],history,NOW+timedelta(minutes=10))[1][0]['action']=='due'
    history['games'][runner.key(g)]['attempts']=3
    assert runner.plan([g],history,NOW+timedelta(minutes=10))[1][0]['action']=='retry_limit_reached'
    history['games'][runner.key(g)]['status']='running'
    assert runner.plan([g],history,NOW+timedelta(minutes=10))[1][0]['action']=='review_required'


@pytest.mark.parametrize('rc,doc', [(124,None),(1,None),(1,{'steps':[{'step':'NFL Matchup Cards'}]}),
    (1,{'publications':[{'game_id':'early','sent_at':NOW.isoformat()}]})])
def test_unconfirmed_sends_need_review(rc,doc):
    assert runner.classify_attempt([game()],rc,doc)=='review_required'


def test_exception_persists_review_and_reschedule_has_new_identity():
    history=state(); g=game()
    def crash(*a):
        raise RuntimeError('child crashed')
    result=runner.process([g],history,execute=crash,save=lambda *a:None,clock=lambda:NOW)
    assert result['status']=='failed' and result['runs'][0]['status']=='review_required'
    assert runner.plan([game(minutes=80)],history,NOW)[1][0]['action']=='due'


def test_bad_state_is_not_silently_reset(tmp_path,monkeypatch):
    path=tmp_path/'state.json'; path.write_text('{}')
    monkeypatch.setattr(runner,'STATE',path)
    with pytest.raises(ValueError,match='Invalid pregame state'):
        runner.load_state()


def test_scope_filters_and_persistence_does_not_clear_other_games(monkeypatch):
    monkeypatch.setenv(game_scope.ENV,json.dumps(['early']))
    rows=[dict(game_id='early',game_date_et=NOW.date()),dict(game_id='night',game_date_et=NOW.date())]
    assert len(game_scope.filter_frame(pd.DataFrame(rows)))==1
    assert game_scope.filter_records(rows)==rows[:1]
    conn=Connection(['early','night'])
    monkeypatch.setattr(forecast_store.psycopg2.extras,'execute_values',lambda *a:None)
    with pytest.raises(ValueError,match='outside'):
        forecast_store.save_forecasts(conn,'nfl_game_predictions',rows,[])
    assert not conn.cur.calls
    forecast_store.save_forecasts(conn,'nfl_game_predictions',rows[:1],['game_id'])
    update=next(c for c in conn.cur.calls if c[0].startswith('UPDATE'))
    assert 'game_id=ANY(%s)' in update[0] and update[1][1]==['early']


def test_fast_pipeline_order_and_scope(monkeypatch,tmp_path):
    from nfl_pipeline import run_daily_and_notify as daily
    monkeypatch.setattr('sys.argv',['daily','--pregame','--date','2099-09-27','--game-id','early','--run-id','test'])
    monkeypatch.setenv(game_scope.ENV,'[]')
    monkeypatch.setattr(daily,'_repo_root',lambda:tmp_path)
    monkeypatch.setattr(daily,'active_release',lambda:{'release_id':'frozen'})
    monkeypatch.setenv('NFL_MODEL_RELEASE_ID','frozen')
    steps=[]
    def execute(step):
        assert game_scope.game_ids()==['early']
        steps.append(step)
        return 0,'{}',''
    async def post(*a):
        pass
    monkeypatch.setattr(daily,'_run',execute)
    monkeypatch.setattr(daily,'_post_matchups',post)
    asyncio.run(daily.main())
    modules=[s.module for s in steps]
    assert modules[0]=='nfl_pipeline.import_context' and steps[0].critical
    assert 'nfl_pipeline.run_training' not in modules and 'nfl_pipeline.schema' not in modules
    assert modules.index('nfl_pipeline.refresh_lock_quotes')<modules.index('nfl_pipeline.modeling.predict_today')
    assert next(s for s in steps if s.module=='nfl_pipeline.refresh_lock_quotes').critical  # pregame retries for fresh quotes
    assert modules.index('nfl_pipeline.modeling.benchmark_offers')<modules.index('nfl_pipeline.publish_forecasts')
    doc=json.loads((tmp_path/'reports/nfl_daily_run_test.json').read_text())
    assert doc['pregame'] and doc['game_ids']==['early']


def test_morning_run_publishes_projections_when_quotes_are_missing(monkeypatch,tmp_path):
    from nfl_pipeline import run_daily_and_notify as daily
    monkeypatch.setattr('sys.argv',['daily','--date','2099-09-27','--skip-train','--run-id','morning'])
    monkeypatch.setattr(daily,'_repo_root',lambda:tmp_path)
    monkeypatch.setattr(daily,'active_release',lambda:{'release_id':'frozen'})
    monkeypatch.setenv('NFL_MODEL_RELEASE_ID','frozen')
    steps=[]; posted=[]
    def execute(step):
        steps.append(step.module)
        return (1,'{"status":"no_fresh_quotes"}','') if step.module=='nfl_pipeline.refresh_lock_quotes' else (0,'{}','')
    async def post_section(header, body):
        posted.append(header)
    async def post(*a):
        posted.append('cards')
    monkeypatch.setattr(daily,'_run',execute)
    monkeypatch.setattr(daily,'_post_section',post_section)
    monkeypatch.setattr(daily,'_post_matchups',post)
    with pytest.raises(SystemExit):
        asyncio.run(daily.main())  # still reported as a failed run: quotes were missing
    assert 'nfl_pipeline.modeling.predict_player_props' in steps and 'cards' in posted
    assert any('Fresh Lock Quotes' in h for h in posted)


def test_task_replaces_fixed_refresh_with_ten_minute_checker():
    root=Path(__file__).resolve().parents[2]
    tree=ET.parse(root/'scripts/tasks/NFL-PrimeTime-Refresh.xml')
    ns={'t':'http://schemas.microsoft.com/windows/2004/02/mit/task'}
    assert tree.find('.//t:Repetition/t:Interval',ns).text=='PT10M'
    assert tree.find('.//t:Exec/t:Arguments',ns).text=='--game-aware'


def test_scoped_quote_health_cannot_borrow_freshness_from_other_game(monkeypatch):
    from contextlib import nullcontext
    from nfl_pipeline import refresh_lock_quotes as quotes
    monkeypatch.setenv(game_scope.ENV,json.dumps(['early']))
    conn=Connection([])
    start=NOW+timedelta(minutes=90)
    def fetch():
        sql=conn.cur.calls[-1][0]
        if 'raw.nfl_games' in sql:
            return [('GB','ATL',start)]
        return [('Green Bay Packers','Atlanta Falcons',start,NOW-timedelta(hours=8)),
                ('Dallas Cowboys','New York Giants',start,NOW-timedelta(minutes=1))]
    conn.cur.fetchall=fetch
    monkeypatch.setattr(quotes.psycopg2,'connect',lambda *a:nullcontext(conn))
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            return NOW
    monkeypatch.setattr(quotes,'datetime',Clock)
    result=quotes.quote_health(NOW.date())
    assert all(v['observed_rows']==1 and v['fresh_rows']==0 for v in result.values())


def test_matchup_scope_is_applied_after_global_selection(monkeypatch,tmp_path):
    from contextlib import nullcontext
    from nfl_pipeline import publish_forecasts as publish, discord_matchups
    monkeypatch.setenv(game_scope.ENV,json.dumps(['early']))
    monkeypatch.setattr(publish,'__file__',str(tmp_path/'src/nfl_pipeline/publish_forecasts.py'))
    monkeypatch.setattr(publish,'release_artifact',lambda *a:{'version':'frozen'})
    conn=Connection([]); conn.set_session=lambda **kw:None
    conn.cur.fetchall=lambda:[('early','GB','ATL',NOW),('night','DAL','NYG',NOW)]
    monkeypatch.setattr(publish.psycopg2,'connect',lambda *a:nullcontext(conn))
    monkeypatch.setattr(publish,'display_rows',lambda *a,**kw:[])
    monkeypatch.setattr(publish,'previously_locked_props',lambda *a:[])
    def bundle(day,schedule,*a,**kw):
        assert [g['game_id'] for g in schedule]==['early','night']
        return dict(cards=[{'game_id':'early'},{'game_id':'night'}],games=2)
    monkeypatch.setattr(discord_matchups,'build_bundle',bundle)
    monkeypatch.setattr(discord_matchups,'preview_markdown',lambda *a:'test preview')
    result=publish.matchup_bundle(NOW.date())
    assert result['cards']==[{'game_id':'early'}] and result['games']==1
    assert list((tmp_path/'reports').glob('*_scope_*.json'))
    assert not (tmp_path/'reports'/f'nfl_discord_matchups_{NOW.date()}.json').exists()
