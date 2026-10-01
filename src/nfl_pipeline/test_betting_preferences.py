from copy import deepcopy
from datetime import date
import json

import pytest

from nfl_pipeline import betting_preferences as preference
from nfl_pipeline import lock_ledger as ledger
from nfl_pipeline.discord_matchups import build_bundle
from nfl_pipeline.test_discord_matchups import prop, game_pick, schedule, NOW, DAY


@pytest.mark.parametrize('url,allowed', [
    ('https://sportsbook.fanduel.com/addToBetslip?selectionId=123', True),
    ('https://ia.sportsbook.fanduel.com/addToBetslip?selectionId=123', True),
    ('https://fanduel.com/', True),
    ('https://sportsbook.draftkings.com/offer', False),
    ('https://fanduel.com.example.com/offer', False),
    ('https://example.com/?redirect=https://fanduel.com', False),
    ('https://fanduel.com@draftkings.com/', False),
    ('https://user@fanduel.com/', False),
    ('https://fanduel.com:123/offer', False),
    ('http://sportsbook.fanduel.com/offer', False),
    (None, False),
])
def test_only_real_fanduel_links(url, allowed):
    assert preference.execution_link(url) is allowed


def test_mixed_books_filtered_before_micro_cap_and_research_rank():
    dk = [prop(player=f'DK{i}', book='draftkings', tier='micro_projection', ev=100+i,
               link=f'https://sportsbook.draftkings.com/{i}') for i in range(6)]
    fd = [prop(player=f'FD{i}', tier='micro_projection', ev=.1+i/100) for i in range(6)]
    bundle = build_bundle(DAY, schedule(), [game_pick(book='draftkings'), game_pick()],
        dk+fd+[prop(player='ResearchFD'), prop(player='ResearchDK', book='draftkings', ev=100)], ['frozen'], now=NOW)
    text = json.dumps(bundle)
    assert 'draftkings' not in text.lower() and 'ResearchDK' not in text
    assert text.count('BET $1:') == 5 and 'FD5' in text
    assert 'ResearchFD' in text and 'FanDuel only' in text


def test_wrong_book_url_never_becomes_a_fanduel_bet_or_parlay():
    wrong = prop(player='Wrong', tier='micro_projection', link='https://sportsbook.draftkings.com/bet')
    original = deepcopy(wrong)
    rows = preference.display_rows([wrong])
    assert rows[0]['tier'] == 'paper' and rows[0]['link'] is None
    assert wrong == original
    bundle = build_bundle(DAY, schedule(), [], [wrong], ['frozen'], now=NOW)
    assert 'draftkings' not in json.dumps(bundle).lower()
    assert 'BET $1:' not in json.dumps(bundle)


def test_projection_only_survives_but_never_keeps_untrusted_link():
    row = prop(book=None, line=None, side=None, offer_id=None, price=None,
               link='https://sportsbook.draftkings.com/bet')
    result = preference.display_rows([row])
    assert len(result) == 1 and result[0]['link'] is None


def test_paper_parlays_use_only_fanduel_selections():
    def link(market, selection):
        return f'https://sportsbook.fanduel.com/addToBetslip?marketId={market}&selectionId={selection}'
    rows = [prop(player='FD1', link=link('1.1', 11)), prop(player='FD2', link=link('1.2', 22)),
            prop(player='DK', book='draftkings', link=link('9.9', 999))]
    bundle = build_bundle(DAY, schedule(), [], rows, ['frozen'], now=NOW)
    text = json.dumps(bundle)
    assert 'Research parlay' in text and '999' not in text


class Cursor:
    rowcount = 1
    def __init__(self, rows=()):
        self.rows = rows
        self.calls = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def execute(self, sql, params=None):
        self.calls.append((sql, params))
    def fetchall(self):
        return self.rows
    def fetchone(self):
        return (0,)


class Connection:
    def __init__(self, rows=()):
        self.cur = Cursor(rows)
    def cursor(self, **kwargs):
        return self.cur
    def commit(self):
        pass


@pytest.mark.parametrize('loader', [ledger.lock_game_predictions, ledger.lock_prop_predictions])
def test_new_actionable_ledger_rows_filter_book_before_sql_limit(loader, monkeypatch):
    conn = Connection()
    monkeypatch.setattr(ledger, '_insert_rows', lambda c, r: 0)
    assert loader(conn, ledger.LedgerConfig(tier='micro_projection')) == 0
    sql, params = conn.cur.calls[0]
    assert sql.index('book = %(execution_book)s') < sql.index('LIMIT')
    assert params['execution_book'] == 'fanduel'
    assert "%(tier)s = 'paper'" in sql


@pytest.mark.parametrize('loader,key', [(ledger.lock_game_predictions, 'bet_markets'),
                                        (ledger.lock_prop_predictions, 'bet_stats')])
def test_staked_tiers_filter_bettable_markets_before_sql_limit(loader, key, monkeypatch):
    conn = Connection()
    monkeypatch.setattr(ledger, '_insert_rows', lambda c, r: 0)
    loader(conn, ledger.LedgerConfig(tier='micro_projection'))
    sql, params = conn.cur.calls[0]
    assert sql.index(f'%({key})s') < sql.index('LIMIT')
    assert params[key] == sorted(preference.BET_PROP_STATS if key == 'bet_stats' else preference.BET_GAME_MARKETS)


def test_only_bettable_markets_lock_into_staked_tiers_but_paper_keeps_everything():
    fd = 'https://sportsbook.fanduel.com/bet'
    def row(ident, kind, market, stat, tier='micro_projection'):
        return (kind,ident,date.today(),tier,1,'fanduel',market,stat,'over',40.5,-110,fd,'v','key')
    conn = Connection()
    rows = [row(1,'prop',None,'receiving_yards'), row(2,'prop',None,'rushing_yards'),
            row(3,'prop',None,'passing_yards'), row(4,'game','total',None),
            row(5,'game','spread',None), row(6,'prop',None,'rushing_yards',tier='paper'),
            row(7,'game','total',None,tier='paper')]
    ledger._insert_rows(conn, rows)
    inserted = [p[1] for sql, p in conn.cur.calls if 'INSERT INTO' in sql]
    assert inserted == [1, 6, 7]


def test_defense_in_depth_does_not_lock_draftkings_or_wrong_host():
    def row(ident, book, link, tier='micro_projection'):
        return ('prop',ident,date.today(),tier,1,book,None,'receiving_yards','over',40.5,-110,link,'v','key')
    conn = Connection()
    rows = [row(1,'draftkings','https://sportsbook.draftkings.com/bet'),
            row(2,'fanduel','https://sportsbook.draftkings.com/bet'),
            row(3,'fanduel','https://sportsbook.fanduel.com/bet'),
            row(4,'draftkings','https://sportsbook.draftkings.com/bet',tier='paper')]
    assert ledger._insert_rows(conn, rows) == 2
    inserted = [p for sql, p in conn.cur.calls if 'INSERT INTO' in sql]
    assert [r[1] for r in inserted] == [3,4]
