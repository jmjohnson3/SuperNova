from datetime import datetime, timedelta, timezone

import pytest

from nba_props import paper_recap as recap

NOW = datetime(2026, 10, 29, 1, tzinfo=timezone.utc)


def fixtures():
    offer = dict(event_id='event', player='Test Player', player_norm='testplayer', market='points',
                 side='over', line=20.5, price=110, book='fanduel',
                 home_team='Boston Celtics', away_team='New York Knicks',
                 commence_time_utc=NOW, decided_at=NOW-timedelta(minutes=30),
                 sent_at=NOW-timedelta(minutes=29))
    games = [dict(game_id='game', status='final', home_team='BOS', away_team='NYK',
                  tip_utc=NOW, home_score=120, away_score=110)]
    players = [dict(game_id='game', player_name='Test Player', team='BOS', opponent='NYK',
                    minutes=32, pts=25),
               dict(game_id='game', player_name='Other Player', team='BOS', opponent='NYK', minutes=32, pts=95),
               dict(game_id='game', player_name='Away Player', team='NYK', opponent='BOS', minutes=32, pts=110)]
    return offer, games, players


def test_verified_win_loss_and_push_at_exact_offer():
    offer, games, players = fixtures()
    assert recap.verify_result(offer, games, players, set())['result'] == 'win'
    assert recap.verify_result(dict(offer, side='under'), games, players, set())['result'] == 'loss'
    assert recap.verify_result(dict(offer, line=25), games, players, set())['result'] == 'push'


@pytest.mark.parametrize('change', ['no_final', 'partial', 'missing_player', 'no_minutes',
                                     'conflict', 'ambiguous', 'late', 'wrong_game', 'missing_stat'])
def test_unverified_results_remain_pending(change):
    offer, games, players = fixtures()
    conflicts = set()
    if change == 'no_final':
        games[0]['status'] = 'live'
    elif change == 'partial':
        players.pop()
    elif change == 'missing_player':
        offer['player'] = 'Missing Player'
    elif change == 'no_minutes':
        players[0]['minutes'] = 0
    elif change == 'conflict':
        conflicts.add('game')
    elif change == 'ambiguous':
        games.append(dict(games[0], game_id='other'))
    elif change == 'late':
        offer['sent_at'] = NOW
    elif change == 'wrong_game':
        players[0]['game_id'] = 'other'
    elif change == 'missing_stat':
        offer['market'] = 'rebounds'
    result = recap.verify_result(offer, games, players, conflicts)
    assert result['result'] is None and result['actual'] is None


def test_duplicate_alerts_are_separated_by_book_and_no_live_price_is_inferred():
    offer, _, _ = fixtures()
    rows = [dict(offer, result='win', close_ev=.03),
            dict(offer, book='espnbet', price=-110, result='win', close_ev=-.01),
            dict(offer, player_norm='second', result='loss', close_ev=None)]
    summary = recap.summarize(rows)
    assert summary['alerts'] == 3 and summary['unique_picks'] == 2
    assert summary['repeated_alerts'] == 1 and summary['games'] == 1
    assert summary['unique_record']['win'] == 1 and summary['unique_record']['loss'] == 1
    assert summary['by_book']['fanduel']['paper_units'] == pytest.approx(.10)
    assert summary['by_book']['espnbet']['paper_units'] == pytest.approx(100/110)
    assert summary['by_book']['fanduel']['timely_checks'] == 0
    assert summary['by_book']['fanduel']['mean_checked_price_ev'] is None


def test_disagreeing_duplicate_results_are_not_counted_as_wins():
    offer, _, _ = fixtures()
    summary = recap.summarize([dict(offer, result='win'), dict(offer, book='espnbet', result=None)])
    assert summary['unique_record'][None] == 1
    assert summary['unique_record']['win'] == 0


def box_payload():
    offer, games, players = fixtures()
    box = dict(gameId='game')
    for side, team in [('home', 'BOS'), ('away', 'NYK')]:
        members = [dict(personId=i+1, firstName=p['player_name'].split()[0], familyName='Player',
                        statistics=dict(points=p['pts'], minutes='32:00', reboundsTotal=4,
                                        assists=3, threePointersMade=2))
                   for i, p in enumerate(players) if p['team'] == team]
        box[side+'Team'] = dict(teamTricode=team, statistics=dict(points=games[0][side+'_score']), players=members)
    return offer, games, dict(boxScoreTraditional=box)


def test_complete_boxscore_can_resolve_missing_game_log_points():
    offer, games, payload = box_payload()
    players = recap.boxscore_players(payload, games[0])
    assert len(players) == 3
    assert recap.verify_result(offer, games, players, set())['result'] == 'win'


@pytest.mark.parametrize('problem', ['wrong_game', 'wrong_team', 'wrong_score', 'partial', 'duplicate'])
def test_boxscore_cannot_override_an_inconsistent_or_ambiguous_result(problem):
    _, games, payload = box_payload()
    box = payload['boxScoreTraditional']
    if problem == 'wrong_game':
        box['gameId'] = 'other'
    elif problem == 'wrong_team':
        box['homeTeam']['teamTricode'] = 'PHX'
    elif problem == 'wrong_score':
        box['homeTeam']['statistics']['points'] = 118
    elif problem == 'partial':
        box['homeTeam']['players'].pop()
    else:
        box['awayTeam']['players'][0]['personId'] = 1
    assert not recap.boxscore_players(payload, games[0])
