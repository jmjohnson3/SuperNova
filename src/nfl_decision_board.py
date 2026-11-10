"""Visible NFL paper-decision status, updated in place in the NFL Discord channel.

Reads existing forecasts, immutable decisions and deliveries. Never invents a
selection, refreshes a quote, records a check, or reserves/authorizes a wager.
"""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os

import psycopg2
import requests

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from supernovabets_config import _saved_windows_env
import model_research

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'reports/nfl_decision_board_discord_state.json'
ET = ZoneInfo('America/New_York')


def stamp(value, style='t'):
    at = safe_time(value)
    return f'<t:{int(at.timestamp())}:{style}>' if at else 'unknown'


def phase(kickoff, now, last_quote):
    lead = (kickoff - now).total_seconds() / 60
    if lead <= 0:
        return 'Game started — all pregame offers closed.'
    if lead > 60:
        return 'Waiting for the final-hour scan.'
    age = (now - last_quote).total_seconds() if last_quote else None
    if age is None or not 0 <= age <= 600:
        return 'Quote collection missing or delayed — no current offer established.'
    if lead > 45:
        return 'Collecting quotes; the selection window has not opened.'
    if lead < 15:
        return 'Selection window closed; collecting closing prices.'
    return 'Selection window open; only qualifying exact-price offers receive checks.'


def empty_pick_status(game, now, blocked=None):
    """An unevaluated game is not a failed selection or a no-pick verdict."""
    kick = safe_time(game['start_ts_utc'])
    if now >= kick:
        return 'Pregame offers expired; results go to #record-ledger.'
    if blocked:
        return 'Selections unavailable while paper collection is blocked.'
    if now < kick-timedelta(minutes=60):
        return f'Not evaluated yet — scanning starts {stamp(kick-timedelta(minutes=60))}. Tonight\'s paper selections are still pending.'
    if now < kick-timedelta(minutes=45):
        return f'Not evaluated yet — the paper-selection window opens {stamp(kick-timedelta(minutes=45))}.'
    if now > kick-timedelta(minutes=15):
        count = len(game.get('decisions') or [])
        return f'Selection window ended; {count} paper price qualifiers saved. No new pregame selections will be added.'
    last = safe_time(game.get('last_quote'))
    if not last or not 0 <= (now-last).total_seconds() <= 600:
        return 'Unable to assess current offers — recent scan data is missing. This is not a no-pick verdict.'
    if game.get('decisions'):
        return 'No fresh offer to check. Earlier selections remain in their original alerts; their prices are not renewed here.'
    return 'No current qualifying paper offer in the latest saved scan. More scans can follow before the window closes.'


def render(game, now, blocked=None):
    kick = safe_time(game['start_ts_utc'])
    lines = [f"**NFL PAPER DECISIONS — {game['away_team_abbr']} @ {game['home_team_abbr']}**",
        f"Kickoff {stamp(kick, 'f')} | status as of {stamp(now)}",
        f"Scanning from {stamp(kick-timedelta(minutes=60))}; paper selection {stamp(kick-timedelta(minutes=45))}–{stamp(kick-timedelta(minutes=15))}.",
        '**No cash recommendation.** Paper picks/checks do not request a wager.', '']
    if blocked:
        lines.append('**Paper collection blocked:** '+blocked)
    else:
        lines.append(phase(kick, now, safe_time(game.get('last_quote'))))
    quotes = game.get('quote_times') or {}
    for label, keys in [('Props', ('receiving_yards', 'rushing_yards', 'receptions')),
                        ('Spreads', ('spread',)), ('Totals', ('total',))]:
        times = [safe_time(quotes[k]) for k in keys if quotes.get(k)]
        lines.append(f"{label}: latest saved quote observation {stamp(max(times)) if times else 'none yet'}.")
    forecast = game.get('forecast')
    if forecast:
        margin, total = forecast.get('predicted_home_margin'), forecast.get('predicted_total_points')
        m = f'{float(margin):+.1f}' if margin is not None else 'unknown'
        t = f'{float(total):.1f}' if total is not None else 'unknown'
        lines += ['', f"Game forecast: {game['home_team_abbr']} margin {m}; total {t} (saved {stamp(forecast['created_at_utc'])}).",
                  'Forecasts select or veto the model-assisted paper strategy on the same qualifying FanDuel offers.']
    decisions = game.get('decisions') or []
    lines += ['', '**FanDuel paper picks and checks**']
    fresh = []
    for row in decisions:
        o = row['payload']['offer']; at = safe_time(o.get('decided_at'))
        if not blocked and at and 0 <= (now-at).total_seconds() <= 300 and now < kick:
            fresh.append(row)
    if not fresh:
        lines.append(empty_pick_status(game, now, blocked))
    for row in fresh:
        o, d = row['payload']['offer'], row['payload']['decision']
        name = o.get('player') or (game['home_team_abbr'] if o['side'] == 'home' else
                                  game['away_team_abbr'] if o['side'] == 'away' else 'Game')
        state = 'model selected' if d['selected'] else 'model veto: '+d['reason'].replace('_', ' ')
        text = (f"M{row['id']} — {name} {o['side'].upper()} {float(o['line']):g} {o['market'].replace('_', ' ')} "
                f"at {int(o['price']):+d}; reference EV {float(o['ev']):+.1%}; {state}.")
        if row.get('delivery_status') == 'sent':
            text += f" Check deadline {stamp(safe_time(row['sent_at'])+timedelta(minutes=5))}; original alert time applies."
        else:
            text += ' No confirmed check-alert delivery.'
        if len('\n'.join(lines+[text])) > 1750:
            lines.append('Additional offers are in their original PAPER PRICE CHECK messages.')
            break
        lines.append(text)
    if decisions:
        selected = sum(bool(r['payload']['decision']['selected']) for r in decisions)
        lines.append(f"Saved this game: {len(decisions)} price qualifiers; {selected} model-selected; {len(decisions)-selected} model-vetoed. Older quotes are not renewed here.")
    lines += ['', 'Only an original PAPER PRICE CHECK starts a check window. Send its M ID, actual time and displayed line/odds (or disappeared/suspended) in Codex.']
    content = '\n'.join(lines)
    if len(content) > 2000:
        raise ValueError('NFL decision board exceeds Discord content limit')
    return content


def build(conn, now):
    try:
        doc = model_research.active()
        blocked = None
    except (OSError, ValueError):
        doc = None
        blocked = 'Frozen comparison verification failed; inspect the local report.'
    games = model_research.rows(conn, """SELECT game_id,home_team_abbr,away_team_abbr,start_ts_utc
        FROM raw.nfl_games WHERE game_date_et=%s ORDER BY start_ts_utc,game_id""", (now.astimezone(ET).date(),))
    for g in games:
        quotes = model_research.rows(conn, """SELECT payload->>'stat' AS stat,max(observed_at_utc) AS latest
            FROM bets.nfl_sharp_quote_observations WHERE game_id=%s AND observed_at_utc<=%s GROUP BY 1""", (g['game_id'], now))
        g['quote_times'] = {r['stat']: r['latest'] for r in quotes}
        g['last_quote'] = max(g['quote_times'].values(), default=None)
        forecasts = model_research.rows(conn, """SELECT predicted_home_margin,predicted_total_points,created_at_utc
            FROM bets.nfl_game_predictions WHERE game_id=%s AND created_at_utc<=%s
            AND created_at_utc<%s ORDER BY created_at_utc DESC,id DESC LIMIT 1""", (g['game_id'], now, g['start_ts_utc']))
        g['forecast'] = forecasts[0] if forecasts else None
        g['decisions'] = model_research.rows(conn, """SELECT d.id,d.payload,m.delivery_status,m.sent_at
            FROM bets.model_research_decisions d LEFT JOIN bets.model_research_delivery m USING(id)
            WHERE d.sport='nfl' AND d.experiment_id=%s AND d.experiment_sha256=%s
              AND d.payload->'offer'->>'game_id'=%s AND d.captured_at<=%s ORDER BY d.id""",
            (model_research.NAME, (doc or {}).get('sha256', ''), g['game_id'], now))
        g['content'] = render(g, now, blocked)
    return model_research.clean(dict(built_at=now, date=now.astimezone(ET).date(), games=games,
        blocked=blocked, betting_authorized=False, experiment_id=model_research.NAME))


def response_json(response):
    if response.status_code != 200:
        raise RuntimeError(f'Discord board request failed: HTTP {response.status_code}')
    return response.json()


def publish(board, url, *, state_path=STATE, session=requests):
    if not board['games']:
        return dict(status='no_games', messages=[])
    meta = response_json(session.get(url, timeout=20))
    route = {k: str(meta.get(k) or '') for k in ('id', 'channel_id', 'guild_id')}
    if not all(route.values()):
        raise RuntimeError('NFL webhook route missing')
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(route=route, games={})
    if state['route'] != route:
        raise RuntimeError('NFL decision board webhook route changed')
    links = []
    for game in board['games']:
        gid = game['game_id']; content = game['content']
        fingerprint = hashlib.sha256(content.encode()).hexdigest()
        old = state['games'].get(gid, {})
        mid = old.get('message_id')
        if old.get('status') == 'pending_create' and not mid:
            raise RuntimeError('Prior NFL board creation is uncertain; reconcile before sending again')
        if old.get('fingerprint') == fingerprint:
            links.append(old['message_url'])
            continue
        state['games'][gid] = dict(old, status='updating' if mid else 'pending_create')
        atomic_json(state_path, state)
        body = dict(content=content, allowed_mentions={'parse': []})
        res = (session.patch(url.rstrip('/')+'/messages/'+mid, json=body, timeout=20) if mid else
               session.post(url, params={'wait': 'true'}, json=body, timeout=20))
        sent = response_json(res)
        if str(sent.get('channel_id')) != route['channel_id'] or not sent.get('id'):
            raise RuntimeError('NFL board delivery route mismatch')
        mid = str(sent['id'])
        link = f"https://discord.com/channels/{route['guild_id']}/{route['channel_id']}/{mid}"
        state['games'][gid] = dict(message_id=mid, message_url=link, status='verifying')
        atomic_json(state_path, state)
        verified = response_json(session.get(url.rstrip('/')+'/messages/'+mid, timeout=20))
        if verified.get('content') != content:
            raise RuntimeError('NFL board content did not verify')
        state['games'][gid].update(status='published', fingerprint=fingerprint)
        atomic_json(state_path, state)
        links.append(link)
    return dict(status='updated', messages=links)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--post', action='store_true')
    args = ap.parse_args()
    with closing(psycopg2.connect(PG_DSN, connect_timeout=10)) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='30s'")
            cur.execute('SELECT pg_try_advisory_lock(86731447)')
            if not cur.fetchone()[0]:
                print('{"status":"already_running"}')
                return
        board = build(conn, datetime.now(timezone.utc))
        atomic_json(ROOT/'reports/nfl_decision_board_latest.json', board)
        (ROOT/'reports/nfl_decision_board_latest.md').write_text('\n\n---\n\n'.join(g['content'] for g in board['games']), encoding='utf-8')
        if args.post:
            url = _saved_windows_env('NFL_DISCORD_WEBHOOK_URL') or os.getenv('NFL_DISCORD_WEBHOOK_URL')
            if not url:
                raise RuntimeError('NFL-specific webhook missing; refusing a fallback channel')
            try:
                result = publish(board, url)
            except requests.RequestException as exc:
                raise RuntimeError('NFL board network error: '+type(exc).__name__) from None
        else:
            result = dict(status='preview', games=len(board['games']), blocked=board['blocked'])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
