"""Prospective NBA/NFL FanDuel public-feed observations, never account checks or bets.

Separate from the frozen v1 NBA observer and manual ledgers. One first attempt per
original delivered alert, including failures. No historical API requests or credits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nba_props import fanduel_live as legacy
from nfl_pipeline import experiment_registry as registry
from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name, normalize_team, TEAM_NAME_TO_ABBR

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'src/model_research_experiments'
NAME = 'fanduel-automatic-paper-v2'
SOURCE = 'automated_fanduel_public_feed'
RULES = dict(sports=['nba', 'nfl'], book='fanduel', first_attempt_only=True,
             max_delay_seconds=300, max_record_delay_seconds=300, max_requests=16,
             runtime_seconds=90, max_response_age_seconds=15, tip_tolerance_seconds=900,
             market_policy='main market only; alternate prices never substituted',
             no_historical_backfill=True, manual_checks_added=0, betting_authorized=False)
NBA = {k: (v[0], v[1], v[2]) for k, v in legacy.MARKETS.items()}
NFL = dict(receiving_yards=('receiving-props', 'Receiving Props', 'Receiving Yds'),
           receptions=('receiving-props', 'Receiving Props', 'Total Receptions'),
           rushing_yards=('rushing-props', 'Rushing Props', 'Rushing Yds'),
           passing_yards=('passing-props', 'Passing Props', 'Passing Yds'),
           passing_tds=('passing-props', 'Passing Props', 'Passing TDs'))
GAME = dict(spread=('popular', 'Popular', 'Spread'), total=('popular', 'Popular', 'Total Points'))
SCHEMA = """
CREATE TABLE IF NOT EXISTS bets.automatic_paper_attempts (
 id BIGSERIAL PRIMARY KEY, observer_id TEXT NOT NULL, observer_sha256 TEXT NOT NULL,
 alert_key TEXT NOT NULL, sport TEXT NOT NULL, attempted_at TIMESTAMPTZ NOT NULL,
 alert JSONB NOT NULL, UNIQUE(observer_id,alert_key));
CREATE TABLE IF NOT EXISTS bets.automatic_paper_observations (
 attempt_id BIGINT PRIMARY KEY REFERENCES bets.automatic_paper_attempts(id),
 recorded_at TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL);
CREATE OR REPLACE FUNCTION bets.automatic_paper_immutable() RETURNS trigger AS $$
 BEGIN RAISE EXCEPTION 'Automatic paper evidence is append-only'; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='bets.automatic_paper_attempts'::regclass
               AND tgname='automatic_paper_attempt_immutable') THEN
 CREATE TRIGGER automatic_paper_attempt_immutable BEFORE UPDATE OR DELETE ON bets.automatic_paper_attempts
 FOR EACH ROW EXECUTE FUNCTION bets.automatic_paper_immutable(); END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='bets.automatic_paper_observations'::regclass
               AND tgname='automatic_paper_observation_immutable') THEN
 CREATE TRIGGER automatic_paper_observation_immutable BEFORE UPDATE OR DELETE ON bets.automatic_paper_observations
 FOR EACH ROW EXECUTE FUNCTION bets.automatic_paper_immutable(); END IF;
END $$;
"""


def utcnow():
    return datetime.now(timezone.utc)


def configuration(state):
    from model_research import active as model_active
    from sharp_pipeline.nba_experiment import active as nba_active
    legacy.config(state)
    return dict(RULES, state=state, parents={k: v()['sha256'] for k, v in
                [('nba_price', nba_active), ('model_comparison', model_active)]})


def register(state):
    modules = ('automatic_price_checks', 'nba_props.fanduel_live', 'nfl_pipeline.fanduel_state',
               'nfl_pipeline.markets', 'nfl_pipeline.context_contract', 'nfl_pipeline.integrity',
               'nfl_pipeline.experiment_registry')
    return registry.register_experiment(NAME, configuration(state),
        {m: ROOT / 'src' / Path(*m.split('.')).with_suffix('.py') for m in modules}, root=ROOT, store=STORE)


def active():
    doc = registry.check_experiment(NAME, root=ROOT, store=STORE)
    if doc['config'] != configuration(doc['config']['state']):
        raise ValueError('Automatic observer source or parent strategy drift')
    return doc


def team(sport, name):
    if sport == 'nba':
        return legacy.team_code(name)
    code = normalize_team(name)
    return code if code in set(TEAM_NAME_TO_ABBR.values()) else None


def spec(alert):
    return GAME.get(alert['market']) or (NBA if alert['sport'] == 'nba' else NFL).get(alert['market'])


def event_match(alert, body):
    expected = tuple(team(alert['sport'], alert[k]) for k in ('away_team', 'home_team'))
    if not all(expected):
        raise legacy.FeedError('alert_team_identity_unverified')
    matched = []
    for key, event in (body.get('attachments', {}).get('events') or {}).items():
        names = str(event.get('name', '')).split(' @ ')
        tip = safe_time(event.get('openDate'))
        if (len(names) == 2 and tuple(team(alert['sport'], n) for n in names) == expected and tip
                and abs((tip - safe_time(alert['commence_time_utc'])).total_seconds()) <= RULES['tip_tolerance_seconds']):
            matched.append((str(event.get('eventId') or key), event))
    if len(matched) != 1:
        raise legacy.FeedError('event_missing_or_ambiguous')
    return matched[0]


def quote(alert, event_id, body, now):
    """The main market is fixed in advance; never pick between conflicting prices."""
    try:
        return _quote(alert, event_id, body, now)
    except legacy.FeedError as exc:
        return legacy.unknown(str(exc))
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        return legacy.unknown('response_structure_unverified')


def _quote(alert, event_id, body, now):
    found, event = event_match(alert, body)
    if found != str(event_id):
        return legacy.unknown('event_page_identity_mismatch')
    native = safe_time(alert.get('native_tip'))
    if not native:
        return legacy.unknown('native_schedule_unverified')
    if now >= min(native, safe_time(alert['commence_time_utc']), safe_time(event['openDate'])) or event.get('inPlay') is True:
        return legacy.unknown('game_started')
    if event.get('inPlay') is not False:
        return legacy.unknown('pregame_status_unverified')
    _, title, label = spec(alert)
    layout = body.get('layout') or {}
    tabs = [t for t in (layout.get('tabs') or {}).values() if t.get('title') == title]
    if len(tabs) != 1:
        return legacy.unknown('market_tab_missing')
    cards = [(layout.get('cards') or {}).get(str(c['id']), {}) for c in tabs[0].get('cards', [])]
    if not cards or any(c.get('attachmentsFullyLoaded') is not True for c in cards):
        return legacy.unknown('market_tab_incomplete')
    ids = {str(mid) for c in cards for d in c.get('display', []) for r in d.get('rows', [])
           if str(r.get('eventId')) == str(event_id) for mid in r.get('marketIds', [])}
    markets = body.get('attachments', {}).get('markets') or {}
    if not ids or any(mid not in markets for mid in ids):
        return legacy.unknown('market_attachments_missing')
    candidates = []
    for mid in sorted(ids):
        m = markets[mid]
        name = str(m.get('marketName', ''))
        if alert['market'] in GAME:
            if name != label:
                continue
        else:
            player, _, kind = name.rpartition(' - ')
            if kind != label or normalize_name(player) != normalize_name(alert['player']):
                continue
        if str(m.get('eventId')) != str(event_id) or m.get('inPlay') is not False:
            return legacy.unknown('market_identity_or_pregame_unverified')
        for r in m.get('runners', []):
            name = str(r.get('runnerName', ''))
            if alert['market'] == 'spread':
                expected = alert['home_team'] if alert['side'] == 'home' else alert['away_team']
                if team(alert['sport'], name) != team(alert['sport'], expected):
                    continue
            elif alert['market'] == 'total':
                if name.lower() != alert['side']:
                    continue
            else:
                match = re.fullmatch(r'(.+) (Over|Under)(?: ([0-9]+(?:\.[0-9]+)?))?', name)
                if not match:
                    return legacy.unknown('runner_format_unverified')
                player, side, text_line = match.groups()
                if normalize_name(player) != normalize_name(alert['player']) or side.lower() != alert['side']:
                    continue
                if text_line is not None and number(text_line) != number(r.get('handicap')):
                    return legacy.unknown('runner_line_mismatch')
            line = number(r.get('handicap'))
            if line is None or (alert['market'] != 'spread' and line < 0):
                return legacy.unknown('runner_line_unverified')
            if alert['side'] in ('over', 'under') and r.get('result', {}).get('type') not in (None, alert['side'].upper()):
                return legacy.unknown('runner_side_mismatch')
            odds = (r.get('winRunnerOdds') or {}).get('americanDisplayOdds') or {}
            price = number(odds.get('americanOddsInt', odds.get('americanOdds')))
            status = ('suspended' if m.get('marketStatus') == 'SUSPENDED' or r.get('runnerStatus') == 'SUSPENDED'
                      else 'available' if m.get('marketStatus') == 'OPEN' and r.get('runnerStatus') == 'ACTIVE' else 'unknown')
            if not r.get('selectionId') or (status == 'available' and (price is None or price != int(price) or abs(price) < 100)):
                status = 'unknown'
            candidates.append(dict(status=status, line=line, price=int(price) if price is not None and price == int(price) else None,
                                   market_id=mid, selection_id=str(r.get('selectionId') or '')))
    if not candidates:
        return dict(status='not_listed', reason='main_offer_absent_from_complete_tab', line=None, price=None)
    signatures = {(r['status'], r['line'], r['price']) for r in candidates}
    if len(signatures) != 1:
        return legacy.unknown('conflicting_main_quotes')
    result = dict(candidates[0])
    if result['status'] == 'unknown':
        return legacy.unknown('quote_status_or_price_unverified')
    if result['status'] == 'suspended':
        return dict(result, reason='main_offer_suspended', line=None, price=None)
    same = result['line'] == float(alert['line'])
    return dict(result, status='available' if same else 'changed_line', reason='exact_main_quote' if same else 'main_line_changed')


def observe(alert, client, *, clock=utcnow):
    receipts = []
    at = clock()
    try:
        sent, tip = safe_time(alert['sent_at']), safe_time(alert['commence_time_utc'])
        if not 0 <= (at - sent).total_seconds() <= RULES['max_delay_seconds']:
            raise legacy.FeedError('check_window_missed')
        if not safe_time(alert.get('native_tip')):
            raise legacy.FeedError('native_schedule_unverified')
        if at >= min(tip, safe_time(alert['native_tip'])):
            raise legacy.FeedError('game_started')
        if not spec(alert):
            raise legacy.FeedError('unsupported_market')
        body, meta = client.get('content-managed-page', {'page': 'CUSTOM', 'customPageId': alert['sport']})
        receipts.append(meta)
        eid, _ = event_match(alert, body)
        body, meta = client.get('event-page', {'eventId': eid, 'tab': spec(alert)[0]})
        receipts.append(meta)
        if any(safe_time(m['requested_at']) < sent for m in receipts):
            raise legacy.FeedError('snapshot_before_alert')
        observed = safe_time(meta['received_at'])
        result = dict(quote(alert, eid, body, observed), observed_at=observed, event_id=eid)
    except legacy.FeedError as exc:
        if exc.receipt:
            receipts.append(exc.receipt)
        result = legacy.unknown(str(exc))
    except (TypeError, ValueError, AttributeError, KeyError, IndexError):
        result = legacy.unknown('response_structure_unverified')
    return legacy.clean(dict(result, attempted_at=at, receipts=receipts, source=SOURCE, state=client.state))


def rows(conn, sql, params=()):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def pending(conn, doc, sport, now):
    parents = doc['config']['parents']
    if sport == 'nba':
        alerts = rows(conn, """SELECT '#'||a.alert_id AS alert_key,'nba' AS sport,a.sent_at,d.*
          FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING(decision_id)
          WHERE a.delivery_status='sent' AND a.book='fanduel' AND d.book='fanduel'
          AND a.sent_at>=%s AND a.experiment_sha256=%s AND d.experiment_sha256=%s""",
          (doc['registered_at'], parents['nba_price'], parents['nba_price']))
    else:
        alerts = []
    extra = rows(conn, """SELECT d.id,d.sport,d.payload,m.sent_at FROM bets.model_research_decisions d
        JOIN bets.model_research_delivery m USING(id) WHERE m.delivery_status='sent' AND d.sport=%s
        AND m.sent_at>=%s AND d.experiment_sha256=%s
        AND (d.payload->'offer'->>'parent_check_id') IS NULL""",
        (sport, doc['registered_at'], parents['model_comparison']))
    for r in extra:
        a = r['payload']['offer']
        if a.get('book') == 'fanduel':
            alerts.append(dict(a, alert_key=f"M{r['id']}", sent_at=r['sent_at'], sport=sport))
    done = {r['alert_key'] for r in rows(conn, 'SELECT alert_key FROM bets.automatic_paper_attempts WHERE observer_id=%s', (NAME,))}
    alerts = [a for a in alerts if a['alert_key'] not in done]
    if sport == 'nba':
        games = rows(conn, """SELECT DISTINCT game_id,home_team,away_team,tip_utc FROM nbaprops.game_projections
            WHERE tip_utc>=%s AND tip_utc<=%s""", (now-timedelta(days=1), now+timedelta(days=1)))
    else:
        games = rows(conn, """SELECT game_id,home_team_abbr AS home_team,away_team_abbr AS away_team,start_ts_utc AS tip_utc
            FROM raw.nfl_games WHERE start_ts_utc>=%s AND start_ts_utc<=%s""", (now-timedelta(days=1), now+timedelta(days=1)))
    for a in alerts:
        matches = [g for g in games if all(team(sport, a[k]) and team(sport, a[k]) == team(sport, g[k])
                   for k in ('home_team', 'away_team')) and g['tip_utc'] and
                   abs((safe_time(a['commence_time_utc'])-g['tip_utc']).total_seconds()) <= RULES['tip_tolerance_seconds']]
        a['native_tip'] = min(g['tip_utc'] for g in matches) if len({g['game_id'] for g in matches}) == 1 else None
    # Live deadlines first; expired requests are still recorded without making a request.
    return sorted(alerts, key=lambda a: ((now-safe_time(a['sent_at'])).total_seconds()>300, safe_time(a['sent_at']), a['alert_key']))


def finalize(alert, result, recorded):
    observed, sent = safe_time(result.get('observed_at')), safe_time(alert['sent_at'])
    tips = [safe_time(alert[k]) for k in ('native_tip', 'commence_time_utc') if safe_time(alert.get(k))]
    timely = bool(observed and sent <= observed <= recorded < min(tips) and
                  (observed-sent).total_seconds() <= 300 and (recorded-observed).total_seconds() <= 300)
    return legacy.clean(dict(result, timely=timely, manual_check=False, wager_placed=False,
                             account_availability_verified=False))


def once(conn, doc, sport, *, client=None, clock=utcnow):
    client = client or legacy.Client(doc['config']['state'], folder=ROOT/'data/automatic_price_checks')
    start = time.monotonic()
    saved = 0
    for alert in pending(conn, doc, sport, clock()):
        if time.monotonic()-start > RULES['runtime_seconds'] or client.calls + 2 > RULES['max_requests']:
            break
        claimed = rows(conn, """INSERT INTO bets.automatic_paper_attempts
            (observer_id,observer_sha256,alert_key,sport,attempted_at,alert) VALUES(%s,%s,%s,%s,%s,%s)
            ON CONFLICT(observer_id,alert_key) DO NOTHING RETURNING id""",
            (NAME, doc['sha256'], alert['alert_key'], sport, clock(), psycopg2.extras.Json(legacy.clean(alert))))
        conn.commit()  # A crash remains a failed first attempt; never retry for a better quote.
        if not claimed:
            continue
        result = finalize(alert, observe(alert, client, clock=clock), clock())
        with conn.cursor() as cur:
            cur.execute('INSERT INTO bets.automatic_paper_observations VALUES(%s,%s,%s)',
                        (claimed[0]['id'], clock(), psycopg2.extras.Json(result)))
        conn.commit()
        saved += 1
        if result['reason'] in ('access_blocked','rate_limited'):
            break
    return dict(status='ok', saved=saved, requests=client.calls, manual_checks_added=0, betting_authorized=False)


def paired_evidence(conn, doc, sport, data):
    """Same frozen decisions, valued at the first public quote; separate from manual gates."""
    import pandas as pd
    from model_research_report import exact_close
    from sharp_pipeline.nba_experiment import clustered_interval
    from nfl_pipeline.sharp_math import ev
    models=rows(conn,"""SELECT id,payload FROM bets.model_research_decisions
        WHERE sport=%s AND experiment_sha256=%s ORDER BY id""",(sport,doc['config']['parents']['model_comparison']))
    matched={}
    for m in models:
        parent=m['payload']['offer'].get('parent_check_id')
        key=f'#{parent}' if parent else f"M{m['id']}"
        if key in matched:
            raise ValueError('Ambiguous model comparison for original price alert')
        matched[key]=m['payload']
    records=[]
    for r in data:
        model=matched.get(r['alert_key'])
        if not model:continue
        o,d=model['offer'],model['decision'];p=r['payload']
        if safe_time(o['commence_time_utc'])>utcnow():continue
        close=exact_close(conn,o)
        usable=bool(p.get('timely') and p['status']=='available' and number(p.get('line'))==number(o['line']))
        value=ev(close['probability'],p['price']) if usable and close.get('method')=='exact' else None
        unavailable=bool(p.get('timely') and p['status'] in ('changed_line','suspended','not_listed'))
        opportunity=0.0 if unavailable else value
        records.append(dict(event_id=o['event_id'],day=safe_time(o['commence_time_utc']).astimezone(ZoneInfo('America/New_York')).date(),
            family='games' if o['market'] in GAME else 'props',selected=bool(d['selected']),
            public_price_ev=value,opportunity_ev=opportunity,
            incremental_ev=(0.0 if d['selected'] else -opportunity) if opportunity is not None else None))
    frame=pd.DataFrame(records);groups={}
    if not frame.empty:
        for family,group in frame.groupby('family'):
            selected=group[group.selected]
            groups[family]=dict(paired_offers=len(group),model_selected=len(selected),
                price_only_public_quotes_with_exact_closes=int(group.public_price_ev.notna().sum()),
                model_selected_public_quotes_with_exact_closes=int(selected.public_price_ev.notna().sum()),
                price_only_by_game=clustered_interval(group,'public_price_ev','event_id'),
                model_selected_by_game=clustered_interval(selected,'public_price_ev','event_id'),
                model_incremental_by_game=clustered_interval(group,'incremental_ev','event_id'),
                model_incremental_by_day=clustered_interval(group,'incremental_ev','day'),betting_authorized=False)
    return dict(groups=groups,unpaired_attempts=sum(r['alert_key'] not in matched for r in data),
                note='Independent public-feed diagnostic. Same original offers, including model vetoes. '
                'Reference probabilities are estimates; these EVs are not realized profit or account execution evidence.')


def report(conn, doc, sport):
    data = rows(conn, """SELECT a.alert_key,a.sport,a.attempted_at,a.alert,o.recorded_at,o.payload
        FROM bets.automatic_paper_attempts a LEFT JOIN bets.automatic_paper_observations o ON o.attempt_id=a.id
        WHERE a.observer_sha256=%s AND a.sport=%s ORDER BY a.id""", (doc['sha256'], sport))
    for r in data:
        if r['payload'] is None:
            r['payload'] = dict(status='unknown', reason='attempt_incomplete', timely=False)
    out = legacy.clean(dict(built_at=utcnow(), observer_id=NAME, sha256=doc['sha256'], registered_at=doc['registered_at'],
        sport=sport, state=doc['config']['state'], attempts=len(data),
        statuses=dict(Counter(r['payload']['status'] for r in data)),
        reasons=dict(Counter(r['payload']['reason'] for r in data)),
        timely_available=sum(r['payload'].get('timely', False) and r['payload']['status']=='available' for r in data),
        checks=data, manual_checks_added=0, betting_authorized=False,
        paired_comparison=paired_evidence(conn,doc,sport,data),
        note='Public main-market observations only. Account availability, bet acceptance and profitability remain unverified.'))
    atomic_json(ROOT/f'reports/automatic_price_checks_{sport}_latest.json', out)
    lines = [f'# {sport.upper()} automatic paper checks', '', out['note'], '',
             f"Attempts: {len(data)}; timely exact available: {out['timely_available']}.", '',
             '| Alert | Player / game | Market / side | Original line / price | Observed line / price | Status / reason | Observed UTC | Timely |',
             '|---|---|---|---|---|---|---|---|']
    for r in data:
        a, p = r['alert'], r['payload']
        lines.append(f"| {r['alert_key']} | {a.get('player') or a.get('event_id')} | {a['market']} / {a['side']} | "
                     f"{a['line']} / {a['price']} | {p.get('line')} / {p.get('price')} | "
                     f"{p['status']} / {p['reason']} | {p.get('observed_at')} | {p.get('timely',False)} |")
    (ROOT/f'reports/automatic_price_checks_{sport}_latest.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return {k: v for k, v in out.items() if k != 'checks'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'verify', 'once', 'report'])
    parser.add_argument('--state', default='co')
    parser.add_argument('--sport', choices=['nba', 'nfl'], default='nba')
    args = parser.parse_args()
    doc = register(args.state) if args.command == 'register' else active()
    if args.command in ('register', 'verify'):
        print(json.dumps({k: doc[k] for k in ('experiment_id', 'sha256', 'registered_at')}))
        return
    with closing(psycopg2.connect(PG_DSN, connect_timeout=10)) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_lock(%s)', (86731491 if args.sport=='nba' else 86731492,))
            if not cur.fetchone()[0]:
                print('{"status":"observer_busy"}')
                return
            cur.execute(SCHEMA)
        conn.commit()
        if args.command == 'once':
            print(json.dumps(once(conn, doc, args.sport)))
        print(json.dumps(report(conn, doc, args.sport)))


if __name__ == '__main__':
    main()
