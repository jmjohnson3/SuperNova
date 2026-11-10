"""FanDuel navigation for NCAA Discord cards; frozen forecasts are unchanged.

Daily: python -m ncaaf_pipeline.discord_links
Existing saved slate: python -m ncaaf_pipeline.discord_links --links-only [--no-post]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
import requests

from ncaaf_pipeline import projections
from nfl_pipeline.fanduel_state import APP_KEY
from nfl_pipeline.fanduel_links import _betslip_url
from supernovabets_config import _saved_windows_env

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'reports/ncaaf_fanduel_links_state.json'
LANDING = 'https://sportsbook.fanduel.com/navigation/ncaaf'
LIMIT = 1900


def school(name):
    plain = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    key = ' '.join(re.sub(r'[^a-z0-9]', ' ', plain.replace("'", '')).split())
    # Explicit school identities only; never fuzzy-match similarly named schools.
    return {'umass': 'massachusetts', 'miami florida': 'miami', 'miami fl': 'miami',
            'miami ohio': 'miami oh', 'connecticut': 'uconn', 'southern california': 'usc',
            'brigham young': 'byu', 'central florida': 'ucf', 'southern methodist': 'smu',
            'texas christian': 'tcu', 'louisiana lafayette': 'louisiana',
            'appalachian state': 'app state', 'fiu': 'florida international',
            'sam houston state': 'sam houston'}.get(key, key)


def instant(value):
    try:
        value = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return value.astimezone(timezone.utc) if value.tzinfo else None
    except (ValueError, TypeError):
        return None


def game_link(game, events, now):
    start = instant(game['start_date'])
    matches = {}
    for event in events:
        teams = str(event.get('name') or '').split(' @ ')
        kickoff = instant(event.get('openDate'))
        eid = str(event.get('eventId') or '')
        if (len(teams) != 2 or not eid.isdigit() or not start or not kickoff
                or min(start, kickoff) <= now or abs((kickoff - start).total_seconds()) > 900
                or school(teams[0]) != school(game['away_team'])
                or school(teams[1]) != school(game['home_team'])):
            continue
        slug = re.sub(r'[^a-z0-9@]+', '-', unicodedata.normalize('NFKD', event['name'])
                      .encode('ascii', 'ignore').decode().lower()).strip('-')
        matches[eid] = f'https://sportsbook.fanduel.com/football/ncaa-football/{slug}-{eid}'
    return next(iter(matches.values())) if len(matches) == 1 else None


def feed_data():
    state = (os.getenv('NFL_FANDUEL_STATE') or _saved_windows_env('NFL_FANDUEL_STATE') or '').lower()
    if not re.fullmatch('[a-z]{2}', state):
        return [], [], 'state_unavailable'
    try:
        response = requests.get(f'https://sbapi.{state}.sportsbook.fanduel.com/api/content-managed-page',
                                params={'page': 'CUSTOM', 'customPageId': 'ncaaf', '_ak': APP_KEY},
                                headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if response.status_code != 200:
            return [], [], f'feed_http_{response.status_code}'
        payload = response.json().get('attachments', {})
        events = [e for e in payload.get('events', {}).values() if isinstance(e, dict)]
        # Navigation can tolerate a cached page; a displayed spread needs a current response.
        clock = parsedate_to_datetime(response.headers.get('Date', ''))
        age = max((datetime.now(timezone.utc) - clock).total_seconds(), float(response.headers.get('Age', 0)))
        if not math.isfinite(age) or not -5 <= age <= 15:
            return events, [], 'quote_response_stale'
        markets = [m for m in payload.get('markets', {}).values() if isinstance(m, dict)]
        return events, markets, 'ok'
    except (requests.RequestException, ValueError, TypeError, AttributeError, OverflowError):
        return [], [], 'feed_unavailable'


def feed():
    events, _, status = feed_data()
    return events, status


def spread_lean(game, events, markets, now):
    """Display an existing forecast against the current main spread, without selecting a bet."""
    url = game_link(game, events, now)
    forecast_at = instant(game.get('generated_at_utc'))
    try:
        margin = float(game['pred_margin'])
    except (KeyError, TypeError, ValueError):
        return None
    if not url or not math.isfinite(margin) or not forecast_at or forecast_at > now:
        return None
    event_id = url.rsplit('-', 1)[-1]
    offers = []
    for market in markets:
        if (str(market.get('eventId')) != event_id or market.get('marketName') != 'Spread'
                or market.get('marketStatus') != 'OPEN' or market.get('inPlay') is not False
                or not re.fullmatch(r'\d+\.\d+', str(market.get('marketId') or ''))):
            continue
        sides = {}
        for runner in market.get('runners') or []:
            side = (runner.get('result') or {}).get('type', '').lower()
            if side not in ('home', 'away') or side in sides:
                sides = {}
                break
            try:
                line = float(runner['handicap'])
                price = float(runner['winRunnerOdds']['americanDisplayOdds']['americanOddsInt'])
            except (KeyError, TypeError, ValueError):
                break
            sid = str(runner.get('selectionId') or '')
            if (runner.get('runnerStatus') != 'ACTIVE' or not sid.isdigit()
                    or school(runner.get('runnerName')) != school(game[f'{side}_team'])
                    or not math.isfinite(line) or not math.isfinite(price)
                    or not price.is_integer() or abs(price) < 100):
                break
            sides[side] = dict(line=line, price=int(price), selection_id=sid)
        if set(sides) != {'home', 'away'} or abs(sides['home']['line'] + sides['away']['line']) > 1e-6:
            continue
        gap = margin + sides['home']['line']
        if abs(gap) < .01:
            continue
        side = 'home' if gap > 0 else 'away'
        pick = sides[side]
        offers.append(dict(team=game[f'{side}_team'], side=side, **pick, margin=margin,
                           url=_betslip_url([(market['marketId'], pick['selection_id'])]),
                           observed_at=now, forecast_at=forecast_at))
    return offers[0] if len(offers) == 1 else None


def lean_text(lean):
    if not lean:
        return 'Model spread lean unavailable at a verified current FanDuel line.'
    team, line, price = lean['team'], lean['line'], lean['price']
    forecast_at = int(lean['forecast_at'].timestamp())
    observed_at = int(lean['observed_at'].timestamp())
    return (f'**Model lean (research): {team} {line:+g} ({price:+d})**\n'
            f'FanDuel observed <t:{observed_at}:t>; forecast saved <t:{forecast_at}:t>. '
            'Model comparison only; not a qualifying price alert or cash recommendation.\n'
            f'[Add {team} {line:+g} to FanDuel slip](<{lean["url"]}>) · Verify line/odds after opening.')


def games(day, *, saved_only=False):
    with psycopg2.connect(projections.PG_DSN) as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            extra = ('AND g.start_date > now() AND NOT g.completed AND EXISTS '
                     '(SELECT 1 FROM ncaaf.projections p WHERE p.game_id=g.game_id AND p.run_date=%s)') if saved_only else ''
            cur.execute(f"""SELECT g.game_id, g.home_team, g.away_team, g.start_date,
                    p.pred_margin, p.generated_at_utc
                FROM ncaaf.games g LEFT JOIN ncaaf.projections p ON p.game_id=g.game_id AND p.run_date=%s
                WHERE (g.start_date AT TIME ZONE 'America/New_York')::date=%s
                {extra} ORDER BY g.start_date,g.game_id""", (day, day, day) if saved_only else (day, day))
            return list(cur.fetchall())


def link_text(url):
    return (f'[Open game on FanDuel](<{url}>)' if url else
            f'[Open FanDuel college football — find game manually](<{LANDING}>)')


def chunks(blocks):
    result, current = [], ''
    for block in blocks:
        if len(block) > LIMIT:
            raise ValueError('NCAA Discord block exceeds message limit')
        if current and len(current) + len(block) + 1 > LIMIT:
            result.append(current)
            current = ''
        current = current + '\n' + block if current else block
    if current:
        result.append(current)
    return result


def decorate(text, rows, links, leans=None):
    blocks = re.split(r'\n(?=`)', text)
    for i, block in enumerate(blocks):
        if not block.startswith('`'):
            continue
        matches = [g for g in rows if re.match(
            rf'^`[^`\n]+` \*\*{re.escape(g["away_team"])} -?\d+ (?:@|vs) '
            rf'{re.escape(g["home_team"])} -?\d+\*\*\n', block)]
        url = links.get(matches[0]['game_id']) if len(matches) == 1 else None
        if leans is not None:
            lean = leans.get(matches[0]['game_id']) if len(matches) == 1 else None
            blocks[i] += '\n      ' + lean_text(lean)
        blocks[i] += '\n      ' + link_text(url)
    return chunks(blocks)


def send(text, *, message_id=None):
    # No cross-sport fallback: these messages belong only in the configured NCAA channel.
    url = os.getenv('NCAAF_DISCORD_WEBHOOK_URL') or _saved_windows_env('NCAAF_DISCORD_WEBHOOK_URL')
    if not url:
        raise RuntimeError('NCAA Discord webhook unavailable')
    try:
        if message_id is not None and not str(message_id).isdigit():
            raise ValueError('Invalid Discord message ID')
        method = requests.patch if message_id else requests.post
        destination = url.rstrip('/') + '/messages/' + str(message_id) if message_id else url
        response = method(destination, params={'wait': 'true'},
                                 json={'content': text, 'allowed_mentions': {'parse': []}, 'flags': 4}, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(f'NCAA Discord delivery failed: HTTP {response.status_code}')
        message = response.json()
        return {'id': message['id'], 'channel_id': message['channel_id']}
    except (requests.RequestException, ValueError, KeyError):
        raise RuntimeError('NCAA Discord delivery could not be confirmed') from None


def run(day, *, links_only=False, no_post=False, refresh=True, force=False):
    delivery, context = [], {}

    def prepare(saved=False):
        if not context:
            rows = games(day, saved_only=saved)
            events, markets, status = feed_data() if rows else ([], [], 'no_games')
            now = datetime.now(timezone.utc)
            context.update(rows=rows, feed_status=status,
                           links={g['game_id']: game_link(g, events, now) for g in rows},
                           leans={g['game_id']: spread_lean(g, events, markets, now) for g in rows})

    def publish(text):
        prepare()
        for part in decorate(text, context['rows'], context['links'], context['leans']):
            delivery.append(send(part))

    if not links_only:
        if no_post:
            raise ValueError('--no-post requires --links-only (preview of saved games)')
        result = projections.run(day, refresh=refresh, force=force, poster=publish)
    else:
        try:
            state = json.loads(STATE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            state = {}
        if not no_post and not force and state.get('posted') == str(day):
            return {'status': 'already_posted', 'day': str(day)}
        prepare(saved=True)
        if not context['rows']:
            return {'status': 'no_upcoming_saved_games', 'day': str(day)}
        blocks = [f'**NCAA football — FanDuel links, {day:%b %d}**\n'
                  'Saved model comparisons at current FanDuel spreads. Research only.']
        for g in context['rows']:
            kickoff = int(g['start_date'].timestamp())
            blocks.append(f'**{g["away_team"]} @ {g["home_team"]}** · <t:{kickoff}:t>\n'
                          + lean_text(context['leans'][g['game_id']]) + '\n'
                          + link_text(context['links'][g['game_id']]))
        parts = chunks(blocks)
        previous = state.get('messages', []) if state.get('posted') == str(day) else []
        for i, part in enumerate(parts):
            if no_post:
                print(part)
            else:
                delivery.append(send(part, message_id=previous[i]['id'] if len(previous) == len(parts) else None))
        result = {'status': 'preview' if no_post else 'posted', 'day': str(day)}
        if not no_post:
            STATE.write_text(json.dumps(dict(posted=str(day), messages=delivery), indent=2), encoding='utf-8')
    return dict(result, discord_messages=delivery, feed_status=context.get('feed_status'),
                direct_links=sum(bool(v) for v in context.get('links', {}).values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--date', type=date.fromisoformat)
    ap.add_argument('--links-only', action='store_true')
    ap.add_argument('--no-post', action='store_true')
    ap.add_argument('--no-refresh', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    day = args.date or datetime.now(timezone.utc).astimezone(projections.ET).date()
    print(json.dumps(run(day, links_only=args.links_only, no_post=args.no_post,
                         refresh=not args.no_refresh, force=args.force), default=str))


if __name__ == '__main__':
    main()
