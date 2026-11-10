from copy import deepcopy
from datetime import timedelta

import pytest

import automatic_price_checks as checks
from nba_props.test_fanduel_live import NOW, page, alert as old_alert, runner, market


def alert(**changes):
    return dict(old_alert(), sport='nba', alert_key='#1', **changes)


def test_main_quote_is_preserved_even_when_alternate_price_is_better():
    body = page()
    alt = deepcopy(market(body))
    alt['marketName'] = 'Test Player - Alt Points'
    alt['runners'][0]['winRunnerOdds']['americanDisplayOdds']['americanOddsInt'] = 130
    body['attachments']['markets']['alt'] = alt
    body['layout']['cards']['c1']['display'][0]['rows'][0]['marketIds'].append('alt')
    result = checks.quote(alert(), '1', body, NOW)
    assert (result['status'], result['price']) == ('available', -130)
    assert result['market_id'] == 'm1'
    runner(body)['runnerStatus'] = 'SUSPENDED'
    assert checks.quote(alert(), '1', body, NOW)['status'] == 'suspended'


def test_conflicting_main_quotes_and_incomplete_catalog_fail_closed():
    body = page()
    duplicate = deepcopy(runner(body))
    duplicate['winRunnerOdds']['americanDisplayOdds']['americanOddsInt'] = 120
    market(body)['runners'].append(duplicate)
    assert checks.quote(alert(), '1', body, NOW)['reason'] == 'conflicting_main_quotes'
    body = page()
    body['layout']['cards']['c1']['attachmentsFullyLoaded'] = False
    assert checks.quote(alert(), '1', body, NOW)['status'] == 'unknown'


def test_changed_missing_and_suspended_are_distinct_from_network_failure():
    body = page()
    runner(body)['handicap'] = 23.5
    result = checks.quote(alert(), '1', body, NOW)
    assert (result['status'], result['line'], result['price']) == ('changed_line', 23.5, -130)
    market(body)['marketName'] = 'Other Player - Points'
    assert checks.quote(alert(), '1', body, NOW)['status'] == 'not_listed'
    class Failed:
        state = 'co'
        def get(self, *args):
            raise checks.legacy.FeedError('access_blocked')
    result = checks.observe(alert(), Failed(), clock=lambda: NOW)
    assert result['status'] == 'unknown' and result['reason'] == 'access_blocked'


def nfl_fixture(stat='receiving_yards', side='under', line=22.5):
    a = dict(old_alert(), sport='nfl', alert_key='M7')
    a.update(market=stat, side=side, line=line, home_team='BAL', away_team='JAX')
    body = page()
    body['attachments']['events']['1']['name'] = 'Jacksonville Jaguars @ Baltimore Ravens'
    tab, title, label = checks.spec(a)
    body['layout']['tabs']['168']['title'] = title
    market(body)['marketName'] = label if stat in checks.GAME else 'Test Player - '+label
    runner(body)['runnerName'] = ('Baltimore Ravens' if stat=='spread' else side.title() if stat=='total' else 'Test Player '+side.title())
    runner(body)['handicap'] = line
    runner(body)['result'] = {}
    return a, body


@pytest.mark.parametrize('stat,side,line', [('receiving_yards','under',22.5), ('receptions','over',3.5),
    ('passing_tds','over',1.5), ('rushing_yards','under',33.5), ('passing_yards','over',215.5),
    ('spread','home',-3.5), ('total','under',44.5)])
def test_nfl_exact_market_identity_and_prices(stat, side, line):
    a, body = nfl_fixture(stat, side, line)
    result = checks.quote(a, '1', body, NOW)
    assert (result['status'], result['line'], result['price']) == ('available', line, -130)
    market(body)['marketName'] = '1st Half '+market(body)['marketName']
    assert checks.quote(a, '1', body, NOW)['status'] == 'not_listed'


@pytest.mark.parametrize('mutation', ['wrong_game','started','no_native','wrong_side','unknown_team'])
def test_identity_and_pregame_guards(mutation):
    a, body = nfl_fixture()
    if mutation=='wrong_game': market(body)['eventId']=2
    if mutation=='started': body['attachments']['events']['1']['inPlay']=True
    if mutation=='no_native': a['native_tip']=None
    if mutation=='wrong_side': runner(body)['result']={'type':'OVER'}
    if mutation=='unknown_team': a['home_team']='Unknown Team'
    assert checks.quote(a, '1', body, NOW)['status']=='unknown'


def test_no_backfill_request_after_check_window_or_game_start():
    class NoRequest:
        state='co'
        def get(self, *args):
            pytest.fail('No request may be made for an expired check')
    a=alert()
    a['sent_at']=NOW-timedelta(minutes=6)
    assert checks.observe(a,NoRequest(),clock=lambda:NOW)['reason']=='check_window_missed'
    a['sent_at']=NOW-timedelta(seconds=5)
    a['native_tip']=NOW
    assert checks.observe(a,NoRequest(),clock=lambda:NOW)['reason']=='game_started'


def test_timeliness_and_source_do_not_turn_into_manual_evidence():
    result=dict(status='available', reason='exact_main_quote', observed_at=NOW.isoformat(), price=-130, line=22.5)
    out=checks.finalize(alert(), result, NOW)
    assert out['timely'] and not out['manual_check'] and not out['wager_placed']
    assert not out['account_availability_verified']
    a=alert(); a['sent_at']=NOW-timedelta(minutes=6)
    assert not checks.finalize(a,result,NOW)['timely']
    assert not checks.finalize(alert(),result,NOW+timedelta(minutes=6))['timely']


def test_once_claims_before_network_and_keeps_failure(monkeypatch):
    events=[]
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,params):events.append(('save',params))
    class Conn:
        def cursor(self):return Cursor()
        def commit(self):events.append(('commit',None))
    class Client:
        calls=0
        state='co'
    monkeypatch.setattr(checks,'pending',lambda *a:[alert()])
    monkeypatch.setattr(checks,'rows',lambda *a: events.append(('claim',None)) or [dict(id=1)])
    def observe(*a,**kw):
        assert [e[0] for e in events]==['claim','commit']
        return dict(status='unknown',reason='server_response_stale')
    monkeypatch.setattr(checks,'observe',observe)
    out=checks.once(Conn(),dict(sha256='hash'), 'nba',client=Client(),clock=lambda:NOW)
    assert out['saved']==1 and out['manual_checks_added']==0
    assert [e[0] for e in events]==['claim','commit','save','commit']


def test_duplicate_claim_does_not_retry_network(monkeypatch):
    class Conn:
        def commit(self):pass
    class Client:
        calls=0
        state='co'
    monkeypatch.setattr(checks,'pending',lambda *a:[alert()])
    monkeypatch.setattr(checks,'rows',lambda *a:[])
    monkeypatch.setattr(checks,'observe',lambda *a,**kw:pytest.fail('Duplicate attempt'))
    assert checks.once(Conn(),dict(sha256='hash'),'nba',client=Client(),clock=lambda:NOW)['saved']==0


def test_paired_ev_uses_worse_first_price_and_preserves_veto(monkeypatch):
    import model_research_report
    a=alert();a.update(event_id='game',commence_time_utc=NOW-timedelta(days=1))
    offer=dict(a,parent_check_id=1)
    monkeypatch.setattr(checks,'rows',lambda *a:[dict(id=9,payload=dict(offer=offer,decision=dict(selected=False)))])
    monkeypatch.setattr(model_research_report,'exact_close',lambda *a:dict(method='exact',probability=.55))
    data=[dict(alert_key='#1',payload=dict(timely=True,status='available',line=22.5,price=-130))]
    result=checks.paired_evidence(None,dict(config={'parents':{'model_comparison':'hash'}}),'nba',data)['groups']['props']
    value=.55*(100/130)-.45
    assert result['paired_offers']==1 and result['model_selected']==0
    assert result['price_only_by_game']['mean']==pytest.approx(value)
    assert result['model_incremental_by_game']['mean']==pytest.approx(-value)
    assert result['price_only_public_quotes_with_exact_closes']==1
    assert result['model_selected_public_quotes_with_exact_closes']==0
    assert not result['betting_authorized']
    data[0]['payload']['timely']=False
    result=checks.paired_evidence(None,dict(config={'parents':{'model_comparison':'hash'}}),'nba',data)['groups']['props']
    assert result['price_only_by_game']['observations']==0
    assert result['model_incremental_by_game']['mean'] is None


def test_estimated_closes_cannot_value_automatic_execution(monkeypatch):
    import model_research_report
    a=alert();a.update(event_id='game',commence_time_utc=NOW-timedelta(days=1))
    monkeypatch.setattr(checks,'rows',lambda *_args:[dict(id=1,payload=dict(offer=a,decision=dict(selected=True)))])
    monkeypatch.setattr(model_research_report,'exact_close',lambda *a:dict(method='shifted',probability=.6))
    data=[dict(alert_key='M1',payload=dict(timely=True,status='available',line=22.5,price=110))]
    group=checks.paired_evidence(None,dict(config={'parents':{'model_comparison':'hash'}}),'nba',data)['groups']['props']
    assert group['price_only_public_quotes_with_exact_closes']==0
    assert group['price_only_by_game']['mean'] is None
