"""Collect prospective NFL spread quotes for the existing paper comparisons.

Separate from the frozen cash scanner. Never selects/reserves stakes or sends alerts.
One current spread request for all due games; no historical requests.
"""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os

import psycopg2
import requests

from nfl_pipeline import experiment_registry as registry, sharp_watch as watch
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json
from supernovabets_config import _saved_windows_env

ROOT = Path(__file__).resolve().parents[1]
NAME = 'nfl-spread-quotes-v1'
STATE = ROOT / 'reports/nfl_spread_collection_state.json'
CONFIG = dict(markets=['spreads'], books=['fanduel', 'pinnacle', 'draftkings', 'betmgm'],
              poll_seconds=300, horizon_minutes=60, credit_floor=1000, max_daily_calls=48,
              role='paper quotes only; no cash decision observations or stakes',
              kickoff_match='exact native/provider time and both teams', historical_backfill=False)


def active():
    parent = registry.check_experiment(watch.EXPERIMENT_NAME)
    doc = registry.check_experiment(NAME)
    if doc['config'] != dict(CONFIG, parser_parent=parent['sha256']):
        raise ValueError('Spread collector configuration or parser drift')
    return doc


def register():
    parent = registry.check_experiment(watch.EXPERIMENT_NAME)
    return registry.register_experiment(NAME, dict(CONFIG, parser_parent=parent['sha256']),
                                        {'nfl_spread_collection': Path(__file__)})


def due_games(games, state, now):
    return [g for g in games if 0 < (g['start'] - now).total_seconds() <= CONFIG['horizon_minutes'] * 60
            and (not state.get('last_attempt') or
                 (now - safe_time(state['last_attempt'])).total_seconds() >= CONFIG['poll_seconds'])]


def match_event(event, games):
    matches = [g for g in games if watch.normalize_team(event.get('home_team')) == g['home']
               and watch.normalize_team(event.get('away_team')) == g['away']
               and safe_time(event.get('commence_time')) == g['start']]
    return matches[0] if len(matches) == 1 else None


def observations(event, game, at, doc):
    """Keep original clocks, validate both opposing spread sides, reject conflicts."""
    result = []
    book_counts = {}
    for b in event.get('bookmakers', []):
        book_counts[b.get('key')] = book_counts.get(b.get('key'), 0) + 1
    for b in event.get('bookmakers', []):
        if b.get('key') not in CONFIG['books'] or book_counts[b['key']] != 1:
            continue
        pairs = {}
        conflict = False
        for market in b.get('markets', []):
            if market.get('key') != 'spreads':
                continue
            for side in market.get('outcomes', []):
                team = watch.normalize_team(side.get('name'))
                line = watch._number(side.get('point'))
                if team not in (game['home'], game['away']) or line is None:
                    conflict = True
                    continue
                q = dict(price=side.get('price'), link=side.get('link'), market_key='spreads',
                         book_updated_at=b.get('last_update'), market_updated_at=market.get('last_update'))
                identity = (team, line)
                if identity in pairs and pairs[identity] != q:
                    conflict = True
                pairs[identity] = q
        if conflict:
            continue  # Original response remains in the receipt; no arbitrary duplicate wins.
        for (team, line), quote in pairs.items():
            opponent = game['away'] if team == game['home'] else game['home']
            other = pairs.get((opponent, -line))
            error = 'missing_opposing_spread'
            if other:
                error = watch.quote_error(dict(quote, over=quote['price'], under=other['price']), at)
                error = error or watch.quote_error(dict(other, over=other['price'], under=quote['price']), at)
            body = dict(contract=watch.QUOTE_CONTRACT, observed_at_utc=at.isoformat(),
                        game_id=game['game_id'], event_id=event['id'], commence_time_utc=game['start'].isoformat(),
                        book=b['key'], stat='spread', player_name_norm=team, line=line,
                        quote=quote, validation_error=error, collection_experiment_id=NAME,
                        collection_experiment_sha256=doc['sha256'])
            result.append(dict(body, observation_id=watch._digest(body)))
    return result


def once(conn, doc, *, now=None, session=requests, state_path=STATE):
    now = now or datetime.now(timezone.utc)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    games = due_games(watch._games(now), state, now)
    if not games:
        return dict(status='not_due', next_window='final 60 minutes before kickoff')
    day = now.astimezone(watch._ET).date().isoformat()
    calls = state.get('daily_calls', {}).get(day, 0)
    if calls >= CONFIG['max_daily_calls']:
        return dict(status='daily_credit_cap')
    key = _saved_windows_env('ODDS_API_KEY') or os.getenv('ODDS_API_KEY')
    if not key:
        return dict(status='api_key_missing')
    free = session.get(watch.BASE + '/events', params={'apiKey': key}, timeout=20)
    if free.status_code != 200:
        return dict(status='events_http_error', http=free.status_code)
    try:
        remaining = int(free.headers['x-requests-remaining'])
    except (ValueError, KeyError):
        return dict(status='credit_balance_unknown')
    if remaining - 1 < CONFIG['credit_floor']:
        return dict(status='credit_floor', credits_remaining=remaining)
    events = free.json()
    matched = [g for g in games if sum(match_event(e, [g]) is not None for e in events) == 1]
    if not matched:
        return dict(status='no_matching_events')
    # Record the attempt before HTTP: a timeout cannot trigger unlimited paid retries.
    state['last_attempt'] = now.isoformat()
    state.setdefault('daily_calls', {})[day] = calls + 1
    atomic_json(state_path, state)
    response = session.get(watch.BASE + '/odds', params=dict(apiKey=key, markets='spreads',
        bookmakers=','.join(CONFIG['books']), oddsFormat='american', dateFormat='iso', includeLinks='true'), timeout=20)
    if response.status_code != 200:
        return dict(status='odds_http_error', http=response.status_code)
    at = datetime.now(timezone.utc)
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError('Unexpected spread response')
    receipt = dict(collection_experiment_id=NAME, experiment_sha256=doc['sha256'],
                   requested_at=now.isoformat(), received_at=at.isoformat(), payload=payload,
                   credits_remaining=response.headers.get('x-requests-remaining'))
    digest = watch._digest(receipt)
    atomic_json(ROOT / 'data/nfl_spread_quotes' / (digest + '.json'), receipt)
    saved = []
    for event in payload:
        game = match_event(event, matched)
        if not game or not at < game['start'] or at < safe_time(doc['registered_at']):
            continue
        if sum(e.get('id') == event.get('id') for e in payload) != 1:
            continue
        saved.extend(observations(event, game, at, doc))
    with conn.cursor() as cur:
        watch.persist_observations(cur, saved)
    conn.commit()
    result = dict(status='captured', observed_at=at.isoformat(), quotes=len(saved),
                  valid_quotes=sum(r['validation_error'] is None for r in saved),
                  games=sorted({r['game_id'] for r in saved}), receipt_sha256=digest,
                  credits_remaining=response.headers.get('x-requests-remaining'), wager_placed=False)
    state['latest'] = result
    atomic_json(state_path, state)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'verify', 'once'])
    args = ap.parse_args()
    doc = register() if args.command == 'register' else active()
    result = dict(experiment_id=NAME, sha256=doc['sha256'])
    if args.command == 'once':
        with closing(psycopg2.connect(watch.PG_DSN, connect_timeout=10)) as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT pg_try_advisory_lock(86731446)')
                if not cur.fetchone()[0]:
                    print('{"status":"already_running"}')
                    return
            try:
                result.update(once(conn, doc))
            except requests.RequestException as exc:
                raise RuntimeError('Spread feed network error: ' + type(exc).__name__) from None
    print(json.dumps(result))


if __name__ == '__main__':
    main()
