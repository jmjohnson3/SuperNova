import asyncio
import json
import zlib
from datetime import date, datetime, timezone

import pytest

from nfl_pipeline import run_daily_and_notify as daily
from nfl_pipeline.discord_matchups import build_bundle, embed_pages, DESCRIPTION_LIMIT, CONTRACT

NOW = datetime(2099, 9, 20, 10, tzinfo=timezone.utc)
DAY = date(2099, 9, 20)


def schedule():
    return [
        {'game_id': 'early', 'home_team_abbr': 'DAL', 'away_team_abbr': 'NYG', 'start_ts_utc': '2099-09-20T17:00:00Z'},
        {'game_id': 'late', 'home_team_abbr': 'BUF', 'away_team_abbr': 'KC', 'start_ts_utc': '2099-09-20T20:00:00Z'},
        {'game_id': 'past', 'home_team_abbr': 'PIT', 'away_team_abbr': 'CIN', 'start_ts_utc': '2099-09-20T09:00:00Z'},
    ]


def prop(game='early', player='Receiver', **extra):
    return {'game_id': game, 'player_id': player, 'player_name': player, 'position': 'WR',
        'team_abbr': 'NYG' if game == 'early' else 'KC', 'opponent_abbr': 'DAL' if game == 'early' else 'BUF',
        'stat': 'receiving_yards', 'projection': 55., 'line': 45.5, 'side': 'over',
        'probability': .56, 'price': -110, 'minimum_american_price': -120,
        'ev': .05, 'book': 'fanduel', 'tier': 'paper', 'ledger_locked': True,
        'drift_guard_pass': True, 'offer_id': 1,
        'link': f'https://sportsbook.fanduel.com/addToBetslip?marketId=42.{zlib.crc32(player.encode())}&selectionId=123', **extra}


def game_pick(**extra):
    return {'game_id': 'early', 'home_team_abbr': 'DAL', 'away_team_abbr': 'NYG', 'market': 'spread',
        'tier': 'paper', 'side': 'home', 'line': -3.5, 'price': -110, 'predicted_home_margin': 4,
        'probability': .52, 'ev': .01, 'book': 'fanduel', 'link': 'https://sportsbook.fanduel.com/spread', **extra}


def body(bundle, game):
    return '\n'.join(c['payload']['embeds'][0]['description'] for c in bundle['cards'] if c['game_id'] == game)


def test_each_game_has_its_own_card_in_kickoff_order():
    bundle = build_bundle(DAY, list(reversed(schedule())), [game_pick()],
        [prop(player='EarlyPlayer', tier='micro_projection'),
         prop('late', 'LatePlayer', tier='micro_projection'),
         prop('past', 'PastPlayer', tier='micro_projection')], ['frozen'], now=NOW)
    assert bundle['games'] == 2
    assert [c['game_id'] for c in bundle['cards']] == ['early', 'late']
    assert 'EarlyPlayer' in body(bundle, 'early') and 'LatePlayer' not in body(bundle, 'early')
    assert 'LatePlayer' in body(bundle, 'late') and 'EarlyPlayer' not in body(bundle, 'late')
    assert 'PastPlayer' not in json.dumps(bundle)
    assert 'DAL -3.5' not in json.dumps(bundle)
    assert 'PAPER GAME PICKS' not in json.dumps(bundle)
    assert '<t:' in body(bundle, 'early')
    assert prop(player='EarlyPlayer')['link'] in body(bundle, 'early')
    assert 'https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?' in body(bundle, 'early')


def test_micro_cap_is_slate_wide_not_per_game_and_requires_ledger_lock():
    rows = [prop('early' if i < 4 else 'late', f'Player{i}', tier='micro_projection', ev=i/100) for i in range(8)]
    rows.append(prop(player='Unrecorded', tier='micro_projection', ledger_locked=False, ev=10))
    bundle = build_bundle(DAY, schedule(), [], rows, ['frozen'], now=NOW)
    text = json.dumps(bundle)
    assert text.count('BET $1:') == 3
    assert 'Unrecorded' not in text
    assert 'Player0' not in text and 'Player7' in text
    assert 'Min=' in text and 'Drift=OK' in text
    assert 'Maximum 3 across all games; there is no quota' in text


def test_old_locks_are_separate_and_paper_does_not_become_micro():
    bundle = build_bundle(DAY, schedule(), [], [prop(player='Research'),
        prop(player='Recorded', tier='locked_micro', previously_locked=True)], ['frozen'], now=NOW)
    text = body(bundle, 'early')
    assert 'Research' not in text
    assert 'Already recorded: Recorded' in text
    assert 'NOT ADDITIONAL PLAYS' in text
    assert 'BET $1:' not in text
    assert 'historical quote, not a fresh play' in text


def test_nonactionable_research_rows_are_not_presented_as_bets():
    rows = [prop('early' if i < 8 else 'late', f'Research{i}', ev=i/100) for i in range(16)]
    bundle = build_bundle(DAY, schedule(), [], rows, ['frozen'], now=NOW)
    text = json.dumps(bundle)
    assert 'Research only:' not in text
    assert 'Research15' not in text
    assert '**BETS**' in text and 'SHARP EDGE' in text  # no model bets: the card points at the sharp alerts


def test_starting_skill_section_is_complete_and_independent_of_paper_caps():
    def starter(player, position, rank, stat, projection, *, depth_abb=None, current_line=True):
        row = prop(player=player, position=position, stat=stat, projection=projection,
                   depth_pos_abb=depth_abb or position, depth_pos_rank=rank,
                   depth_pos_slot=rank, roster_status='ACT')
        row['forecast_outputs'] = {'point_forecast': {'value': projection - 1,
                                                       'independent_of_offer': True}}
        if not current_line:
            row.update(book=None, offer_id=None, line=None, side=None, price=None, link=None,
                       starter_current_line=False)
        return row

    rows = [
        starter('QB1', 'QB', 1, 'passing_yards', 251),
        starter('QB2', 'QB', 2, 'passing_yards', 220),
        starter('RB1', 'RB', 1, 'rushing_yards', 71),
        starter('WR1', 'WR', 1, 'receiving_yards', 81),
        starter('WR2', 'WR', 2, 'receiving_yards', 61),
        starter('WR3', 'WR', 3, 'receiving_yards', 41),
        starter('Returner', 'WR', 1, 'receiving_yards', 31, depth_abb='KR'),
        starter('TE1', 'TE', 1, 'receiving_yards', 51, current_line=False),
    ]
    bundle = build_bundle(DAY, schedule(), [], [], ['frozen'], now=NOW,
                          paper_limit=1, starter_prop_rows=rows)
    text = body(bundle, 'early')
    assert 'STARTING SKILL PLAYER PROJECTIONS' in text
    for player in ('QB1', 'RB1', 'WR1', 'WR2', 'TE1'):
        assert player in text
    assert 'QB2' not in text and 'WR3' not in text and 'Returner' not in text
    assert 'Pass Yds 250.0' in text
    assert 'Projection only - no current line' in text
    assert 'FanDuel line available' in text


def test_oversized_game_continues_without_mixing_games_or_cutting_links():
    lines = [f'- Row {i} ' + 'x' * 450 + f' [Bet](<https://example.com/offer/{i}>)' for i in range(24)]
    pages = embed_pages('NYG @ DAL', 'Kickoff at 1 PM', [('PAPER', lines)], 'Release: frozen')
    assert len(pages) > 1
    for page in pages:
        assert page['title'].startswith('NYG @ DAL (')
        assert len(page['description']) <= DESCRIPTION_LIMIT
        assert '**PAPER**' in page['description']
    text = '\n'.join(p['description'] for p in pages)
    for line in lines:
        assert text.count(line) == 1


def test_unreasonably_long_link_is_not_silently_truncated():
    with pytest.raises(ValueError, match='refusing to truncate'):
        embed_pages('Game', 'Kickoff', [('PAPER', ['https://example.com/' + 'x'*4500])], 'frozen')


def test_empty_matchup_and_zero_prop_offers_are_explained():
    bundle = build_bundle(DAY, schedule(), [], [], ['frozen'], now=NOW)
    assert 'No fresh verified FanDuel prop offers' in body(bundle, 'early')
    assert 'SHARP EDGE' in body(bundle, 'early')
    assert 'No qualifying bankroll bets' not in body(bundle, 'early')  # retired model-trial noise
    assert 'No current prop meets' not in body(bundle, 'early')
    empty = build_bundle(DAY, [], [], [], ['frozen'], now=NOW)
    assert empty['cards'] == [] and empty['notice']


def test_poster_sends_separate_embeds_and_records_successes(monkeypatch):
    bundle = build_bundle(DAY, schedule(), [], [prop()], ['frozen'], now=NOW)
    sent = []
    async def send(payload): sent.append(payload)
    monkeypatch.setattr(daily, '_post_payload', send)
    publications = []
    asyncio.run(daily._post_matchups(json.dumps(bundle), publications))
    assert len(sent) == 2
    assert [p['game_id'] for p in publications] == ['early', 'late']
    assert all(p['allowed_mentions'] == {'parse': []} for p in sent)


def test_readiness_issue_still_publishes_projection_and_status_cards(monkeypatch):
    bundle = build_bundle(DAY, schedule(), [], [prop()], ['frozen'], now=NOW,
        cash_readiness={'status': 'operational_issue', 'reason': 'cash_readiness_stale_or_mismatched'})
    sent = []
    async def send(payload): sent.append(payload)
    monkeypatch.setattr(daily, '_post_payload', send)
    publications = []
    asyncio.run(daily._post_matchups(json.dumps(bundle), publications))
    assert len(sent) == len(publications) == 2
    # The model trial's readiness status is noise on cards without its rows; they point at the sharp alerts.
    assert all('**OPERATIONAL ISSUE**' not in p['embeds'][0]['description'] for p in sent)
    assert all('SHARP EDGE' in p['embeds'][0]['description'] for p in sent)
    assert all(not p['forecast_manifest'] for p in publications)


def test_readiness_issue_still_shows_on_a_card_carrying_a_model_bet_row():
    bundle = build_bundle(DAY, schedule(), [], [prop('late', tier='micro_projection')], ['frozen'], now=NOW,
        cash_readiness={'status': 'operational_issue', 'reason': 'cash_readiness_stale_or_mismatched'})
    late = body(bundle, 'late')
    assert '$1 MICRO TEST' in late
    assert '**OPERATIONAL ISSUE**' in late and 'cash_readiness_stale_or_mismatched' in late
    assert '**BETS**' not in late


@pytest.mark.parametrize('unsafe_card', ['cash_ids', 'cash_manifest', 'cash_id_on_paper', 'unmanifested_micro'])
def test_readiness_issue_blocks_staked_cards_before_any_delivery(monkeypatch, unsafe_card):
    rows = [prop('late', tier='micro_projection')] if unsafe_card == 'unmanifested_micro' else []
    bundle = build_bundle(DAY, schedule(), [], rows, ['frozen'], now=NOW,
        cash_readiness={'status': 'operational_issue', 'reason': 'cash_readiness_stale_or_mismatched'})
    card = bundle['cards'][-1]
    if unsafe_card == 'cash_ids':
        card['cash_ledger_ids'] = [123]
    elif unsafe_card == 'cash_manifest':
        card['forecast_manifest'] = [{'tier': 'cash_trial', 'forecast_id': 123}]
    elif unsafe_card == 'cash_id_on_paper':
        card['forecast_manifest'] = [{'tier': 'paper', 'forecast_id': 123, 'cash_ledger_id': 123}]
    else:
        assert not card['forecast_manifest']
        assert '**$1 MICRO TEST**' in card['payload']['embeds'][0]['description']
    sent = []
    async def send(payload): sent.append(payload)
    monkeypatch.setattr(daily, '_post_payload', send)
    publications = []
    with pytest.raises(ValueError, match='Cash publication readiness failed'):
        asyncio.run(daily._post_matchups(json.dumps(bundle), publications))
    assert not sent and not publications


def test_failed_delivery_keeps_prior_successes_visible(monkeypatch):
    bundle = build_bundle(DAY, schedule(), [], [], ['frozen'], now=NOW)
    sent = []
    async def send(payload):
        if sent: raise RuntimeError('test failure')
        sent.append(payload)
    monkeypatch.setattr(daily, '_post_payload', send)
    publications = []
    with pytest.raises(RuntimeError, match='test failure'):
        asyncio.run(daily._post_matchups(json.dumps(bundle), publications))
    assert len(publications) == 1 and publications[0]['game_id'] == 'early'


def test_all_cards_are_validated_before_sending_any(monkeypatch):
    bundle = build_bundle(DAY, schedule(), [], [], ['frozen'], now=NOW)
    bundle['cards'][-1]['payload']['embeds'][0]['description'] = 'x' * 4097
    sent = []
    async def send(payload): sent.append(payload)
    monkeypatch.setattr(daily, '_post_payload', send)
    with pytest.raises(ValueError):
        asyncio.run(daily._post_matchups(json.dumps(bundle), []))
    assert not sent


def test_discord_rate_limit_is_retried_without_dropping_a_game(monkeypatch):
    responses = []
    sleeps = []
    class Response:
        def __init__(self, status): self.status_code = status
        def json(self): return {'retry_after': .5}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            responses.append(kwargs)
            return Response(429 if len(responses) == 1 else 200)
    async def sleep(seconds): sleeps.append(seconds)
    monkeypatch.setattr(daily, '_webhook_url', lambda: 'https://example.com/webhook')
    monkeypatch.setattr(daily.httpx, 'AsyncClient', Client)
    monkeypatch.setattr(daily.asyncio, 'sleep', sleep)
    payload = {'embeds': [{'title': 'Game', 'description': 'Paper'}]}
    asyncio.run(daily._post_payload(payload))
    assert len(responses) == 2 and sleeps == [.5]
    assert all(r['json'] == payload and r['params']['wait'] == 'true' for r in responses)
