"""Read current Colorado FanDuel NBA quotes for delivered paper alerts.

Automated public-feed evidence only. Never writes manual checks or places bets.
Uses saved alerts, FanDuel GET requests and a separate immutable check ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline import experiment_registry as registry
from nfl_pipeline.context_contract import number, safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.fanduel_state import APP_KEY
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name
from supernovabets_config import _saved_windows_env

ROOT=Path(__file__).resolve().parents[2]
STORE=ROOT/'src/model_research_experiments'
DATA=ROOT/'data/fanduel_live_checks'
NAME='fanduel-direct-check-v1'
LOCK=86731421
MAX_RESPONSE_AGE=15
TIMEOUT=8
MAX_REQUESTS=24
MARKETS={
    'points':('player-points','Player Points','Points'),
    'rebounds':('player-rebounds','Player Rebounds','Rebounds'),
    'assists':('player-assists','Player Assists','Assists'),
    'threes':('player-threes','Player Threes','Made Threes'),
}
RULES=dict(sport='nba',book='fanduel',source='automated_fanduel_public_feed',
    max_check_delay_seconds=300,max_response_age_seconds=MAX_RESPONSE_AGE,
    max_tip_difference_seconds=900,earliest_known_tip=True,first_attempt_only=True,
    manual_checks_changed=False,automatic_betting_authorization=False,
    availability='public state feed only; not logged-in UI availability or bet acceptance')
SCHEMA="""
CREATE TABLE IF NOT EXISTS bets.fanduel_live_checks (
 id BIGSERIAL PRIMARY KEY, alert_id BIGINT NOT NULL UNIQUE REFERENCES bets.nba_paper_price_alerts(alert_id),
 observer_id TEXT NOT NULL, observer_sha256 TEXT NOT NULL, source TEXT NOT NULL
 CHECK(source='automated_fanduel_public_feed'), state TEXT NOT NULL,
 attempted_at TIMESTAMPTZ NOT NULL, observed_at TIMESTAMPTZ, recorded_at TIMESTAMPTZ NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('available','changed_line','suspended','not_listed','unknown')),
 reason TEXT NOT NULL, payload JSONB NOT NULL);
CREATE OR REPLACE FUNCTION bets.fanduel_live_check_immutable() RETURNS trigger AS $$
 BEGIN RAISE EXCEPTION 'Automated FanDuel checks are append-only'; END; $$ LANGUAGE plpgsql;
DO $$ BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='bets.fanduel_live_checks'::regclass
               AND tgname='fanduel_live_check_immutable') THEN
 CREATE TRIGGER fanduel_live_check_immutable BEFORE UPDATE OR DELETE ON bets.fanduel_live_checks
 FOR EACH ROW EXECUTE FUNCTION bets.fanduel_live_check_immutable(); END IF;
END $$;
"""


def utcnow():return datetime.now(timezone.utc)


def clean(value):
    return json.loads(json.dumps(value,default=str,allow_nan=False))


def config(state):
    if not re.fullmatch('[a-z]{2}',state):raise ValueError('Specify a two-letter FanDuel state')
    return dict(RULES,state=state)


def register(state):
    modules=('nba_props.fanduel_live','nfl_pipeline.fanduel_state','nfl_pipeline.markets',
             'nfl_pipeline.context_contract','nfl_pipeline.integrity')
    files={m:ROOT/'src'/Path(*m.split('.')).with_suffix('.py') for m in modules}
    return registry.register_experiment(NAME,config(state),files,root=ROOT,store=STORE)


def active():
    doc=registry.check_experiment(NAME,root=ROOT,store=STORE)
    if doc['config']!=config(doc['config']['state']):raise ValueError('Frozen observer configuration changed')
    return doc


def team_code(name):
    from nba_api.stats.static import teams
    known=teams.get_teams()
    codes={t['abbreviation'] for t in known}
    names={t['full_name']:t['abbreviation'] for t in known}
    names['LA Clippers']='LAC'
    return name if name in codes else names.get(name)


class FeedError(Exception):
    def __init__(self,reason,receipt=None):
        super().__init__(reason)
        self.receipt=receipt


def receipt_reason(meta,now):
    received=safe_time(meta.get('received_at'));started=safe_time(meta.get('requested_at'))
    server=safe_time(meta.get('server_date'))
    age=number(meta.get('cache_age_seconds'))
    if meta.get('cache_age_seconds') is not None and age is None:return 'cache_age_unverified'
    if not received or not started or not server:return 'response_time_unverified'
    if not 0<=(received-started).total_seconds()<=MAX_RESPONSE_AGE:return 'slow_response'
    if not 0<=(now-received).total_seconds()<=MAX_RESPONSE_AGE:return 'response_expired'
    if not -5<=(received-server).total_seconds()<=MAX_RESPONSE_AGE:return 'server_response_stale'
    if age is not None and not 0<=age<=MAX_RESPONSE_AGE:return 'cached_response_stale'
    return None


class Client:
    def __init__(self,state,*,session=None,clock=utcnow,folder=DATA):
        config(state)
        self.state=state;self.session=session or requests.Session();self.clock=clock
        self.folder=folder;self.calls=0

    def get(self,path,params):
        if self.calls>=MAX_REQUESTS:raise FeedError('request_budget_exhausted')
        self.calls+=1;start=self.clock()
        try:
            response=self.session.get(f'https://sbapi.{self.state}.sportsbook.fanduel.com/api/{path}',
                params=dict(params,_ak=APP_KEY),timeout=TIMEOUT,allow_redirects=False)
        except requests.RequestException as exc:
            raise FeedError('network_'+type(exc).__name__) from None
        at=self.clock()
        if response.status_code!=200:
            raise FeedError('access_blocked' if response.status_code in (401,403) else
                            'rate_limited' if response.status_code==429 else f'http_{response.status_code}')
        if 'json' not in response.headers.get('Content-Type',''):raise FeedError('non_json_response')
        try:
            body=response.json();server=parsedate_to_datetime(response.headers['Date'])
        except (ValueError,KeyError,TypeError):raise FeedError('response_format_unverified') from None
        if not isinstance(body,dict):raise FeedError('response_format_unverified')
        data=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        meta=clean(dict(state=self.state,path=path,parameters=params,requested_at=start,received_at=at,
            server_date=server,cache_age_seconds=response.headers.get('Age'),
            body_sha256=hashlib.sha256(data).hexdigest(),http_status=200))
        self.folder.mkdir(parents=True,exist_ok=True)
        target=self.folder/(at.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex+'.json')
        with target.open('x',encoding='utf-8') as f:json.dump(dict(meta=meta,body=body),f,allow_nan=False)
        meta['snapshot_file']=str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)
        reason=receipt_reason(meta,self.clock())
        if reason:raise FeedError(reason,meta)
        return body,meta


def match_event(alert,body):
    expected=(team_code(alert['away_team']),team_code(alert['home_team']))
    if not all(expected):raise FeedError('alert_team_identity_unverified')
    matched=[]
    for key,event in (body.get('attachments',{}).get('events') or {}).items():
        teams=str(event.get('name','')).split(' @ ')
        tip=safe_time(event.get('openDate'))
        if (len(teams)==2 and tuple(team_code(t) for t in teams)==expected and tip
            and abs((tip-safe_time(alert['commence_time_utc'])).total_seconds())<=900):
            matched.append((str(event.get('eventId') or key),event))
    if len(matched)!=1:raise FeedError('event_missing_or_ambiguous')
    return matched[0]


def unknown(reason):return dict(status='unknown',reason=reason,line=None,price=None)


def quote_from_page(alert,event_id,body,now):
    try:return _quote_from_page(alert,event_id,body,now)
    except (AttributeError,TypeError,KeyError,ValueError,IndexError):
        return unknown('response_structure_unverified')


def _quote_from_page(alert,event_id,body,now):
    """Match only fully loaded exact player/stat/side quotes; ignore previous odds."""
    attachments=body.get('attachments') or {}
    events=attachments.get('events') or {}
    event=events.get(str(event_id))
    if not isinstance(event,dict):return unknown('event_page_identity_missing')
    try:resolved,_=match_event(alert,body)
    except FeedError as exc:return unknown(str(exc))
    if resolved!=str(event_id):return unknown('event_page_identity_mismatch')
    native=safe_time(alert.get('native_tip'))
    if not native:return unknown('native_schedule_unverified')
    stop=min(native,safe_time(alert['commence_time_utc']),safe_time(event['openDate']))
    if now>=stop or event.get('inPlay') is True:return unknown('game_started')
    if event.get('inPlay') is not False:return unknown('pregame_status_unverified')
    _,title,label=MARKETS[alert['market']]
    layout=body.get('layout') or {}
    tabs=[t for t in (layout.get('tabs') or {}).values() if t.get('title')==title]
    if len(tabs)!=1:return unknown('market_tab_missing')
    card_ids=[str(c['id']) for c in tabs[0].get('cards',[]) if 'id' in c]
    cards=[(layout.get('cards') or {}).get(c,{}) for c in card_ids]
    if not cards or any(c.get('attachmentsFullyLoaded') is not True for c in cards):return unknown('market_tab_incomplete')
    ids={str(mid) for c in cards for display in c.get('display',[])
         for row in display.get('rows',[]) if str(row.get('eventId'))==str(event_id) for mid in row.get('marketIds',[])}
    markets=attachments.get('markets') or {}
    if any(mid not in markets for mid in ids):return unknown('market_attachments_missing')
    if not ids:return unknown('empty_market_tab')
    candidates=[];main=[];malformed=False
    for mid in sorted(ids):
        market=markets[mid]
        who,_,kind=str(market.get('marketName','')).rpartition(' - ')
        if normalize_name(who)!=normalize_name(alert['player']) or kind not in (label,'Alt '+label):continue
        if str(market.get('eventId'))!=str(event_id):return unknown('market_event_mismatch')
        if market.get('inPlay') is not False:return unknown('market_pregame_status_unverified')
        alternate=kind.startswith('Alt ')
        for runner in market.get('runners',[]):
            name=str(runner.get('runnerName',''))
            found=re.fullmatch(r'(.+) (Over|Under)(?: ([0-9]+(?:\.[0-9]+)?))?',name)
            if not found:malformed=True;continue
            person,side,text_line=found.groups()
            if normalize_name(person)!=normalize_name(alert['player']) or side.lower()!=alert['side']:continue
            if runner.get('result',{}).get('type') not in (None,side.upper()):return unknown('runner_side_mismatch')
            line=number(text_line if alternate else runner.get('handicap'))
            if line is None or line<0:malformed=True;continue
            if not alternate and text_line is not None and number(text_line)!=line:
                return unknown('runner_line_mismatch')
            odds=(runner.get('winRunnerOdds') or {}).get('americanDisplayOdds') or {}
            price=number(odds.get('americanOddsInt'))
            if price is None:price=number(odds.get('americanOdds'))
            status=('suspended' if market.get('marketStatus')=='SUSPENDED' or runner.get('runnerStatus')=='SUSPENDED'
                    else 'available' if market.get('marketStatus')=='OPEN' and runner.get('runnerStatus')=='ACTIVE'
                    else 'unknown')
            if status=='available' and (price is None or price!=int(price) or abs(price)<100):
                status='unknown'
            item=dict(status=status,line=line,price=int(price) if price is not None and price==int(price) else None,
                      market_id=mid,selection_id=str(runner.get('selectionId') or ''),alternate=alternate)
            if not item['selection_id']:item['status']='unknown'
            if not alternate:main.append(item)
            if line==float(alert['line']):candidates.append(item)
    if candidates:
        signatures={(r['status'],r['line'],r['price']) for r in candidates}
        if len(signatures)!=1:return unknown('conflicting_exact_quotes')
        result=dict(candidates[0])
        result['reason']='exact_quote' if result['status']=='available' else 'exact_quote_suspended' if result['status']=='suspended' else 'quote_status_or_price_unverified'
        if result['status']!='available':result.update(line=None,price=None)
        return result
    if malformed:return unknown('runner_format_unverified')
    active_main=[r for r in main if r['status']=='available']
    if len(active_main)==1:return dict(active_main[0],status='changed_line',reason='exact_line_missing_main_line_changed')
    if main:return unknown('unresolved_main_market')
    return dict(status='not_listed',reason='exact_offer_not_listed_in_complete_tab',line=None,price=None)


def observe(alert,client,*,clock=utcnow):
    receipts=[];at=clock()
    try:
        if not safe_time(alert.get('native_tip')):raise FeedError('native_schedule_unverified')
        if at>=min(safe_time(alert['native_tip']),safe_time(alert['commence_time_utc'])):raise FeedError('game_started')
        if alert['market'] not in MARKETS:raise FeedError('unsupported_market')
        body,meta=client.get('content-managed-page',{'page':'CUSTOM','customPageId':'nba'})
        receipts.append(meta);eid,_=match_event(alert,body)
        body,meta=client.get('event-page',{'eventId':eid,'tab':MARKETS[alert['market']][0]})
        receipts.append(meta);observed=safe_time(meta['received_at'])
        if any(safe_time(m['requested_at'])<safe_time(alert['sent_at']) for m in receipts):raise FeedError('snapshot_before_alert')
        result=quote_from_page(alert,eid,body,observed)
        result.update(observed_at=observed,event_id=eid)
    except FeedError as exc:
        if exc.receipt is not None:receipts.append(exc.receipt)
        result=unknown(str(exc))
    except (AttributeError,TypeError,KeyError,ValueError,IndexError):
        result=unknown('response_structure_unverified')
    return clean(dict(result,attempted_at=at,receipts=receipts,source=RULES['source'],state=client.state))


def load_pending(conn,doc,now):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT a.alert_id,a.sent_at,a.experiment_id,a.experiment_sha256,
            d.player,d.player_norm,d.market,d.side,d.line::float,d.price,
            d.book,d.away_team,d.home_team,d.commence_time_utc,d.event_id,d.decided_at
            FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING(decision_id)
            LEFT JOIN bets.fanduel_live_checks c USING(alert_id)
            WHERE a.delivery_status='sent' AND a.book='fanduel' AND d.book='fanduel'
            AND c.alert_id IS NULL AND a.sent_at>=%s AND d.commence_time_utc>%s
            ORDER BY a.sent_at,a.alert_id LIMIT 30""",(doc['registered_at'],now))
        alerts=[dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT DISTINCT game_id,home_team,away_team,tip_utc FROM nbaprops.game_projections
            WHERE tip_utc>%s AND tip_utc<%s""",(now-timedelta(minutes=15),now+timedelta(days=1)))
        games=[dict(g) for g in cur.fetchall()]
    for a in alerts:
        matches=[g for g in games if team_code(a['home_team']) and team_code(a['away_team'])
                 and team_code(a['home_team'])==team_code(g['home_team'])
                 and team_code(a['away_team'])==team_code(g['away_team'])
                 and abs((a['commence_time_utc']-g['tip_utc']).total_seconds())<=900]
        native_ids={g['game_id'] for g in matches}
        a['native_tip']=min(g['tip_utc'] for g in matches) if len(native_ids)==1 else None
    return alerts


def save(conn,doc,alert,result,now):
    observed=safe_time(result.get('observed_at'))
    received=safe_time(alert['sent_at'])
    tip=min(t for t in (safe_time(alert.get('native_tip')),safe_time(alert['commence_time_utc'])) if t)
    timely=bool(observed and received<=observed<=now<tip and (observed-received).total_seconds()<=300
                and (now-observed).total_seconds()<=300)
    payload=clean(dict(result,alert=alert,timely=timely,manual_check=False,wager_placed=False))
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO bets.fanduel_live_checks
            (alert_id,observer_id,observer_sha256,source,state,attempted_at,observed_at,recorded_at,status,reason,payload)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(alert_id) DO NOTHING RETURNING id""",
            (alert['alert_id'],NAME,doc['sha256'],RULES['source'],doc['config']['state'],result['attempted_at'],
             observed,now,result['status'],result['reason'],psycopg2.extras.Json(payload)))
        inserted=cur.fetchone() is not None
    conn.commit()
    return inserted


def report(conn,doc):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT alert_id,status,reason,observed_at,recorded_at,payload
            FROM bets.fanduel_live_checks WHERE observer_sha256=%s ORDER BY id""",(doc['sha256'],))
        checks=[dict(r) for r in cur.fetchall()]
    summary=dict(built_at=utcnow().isoformat(),observer_id=NAME,sha256=doc['sha256'],registered_at=doc['registered_at'],
        state=doc['config']['state'],source=RULES['source'],attempts=len(checks),
        statuses=dict(Counter(r['status'] for r in checks)),
        timely_attempts=sum(bool(r['payload']['timely']) for r in checks),
        timely_usable_observations=sum(bool(r['payload']['timely']) and r['status']!='unknown' for r in checks),
        checks=clean(checks),manual_checks_added=0,betting_authorized=False,
        note='Direct FanDuel public-feed observations. Not logged-in UI checks, confirmed fills or manual-check gate evidence.')
    atomic_json(ROOT/'reports/fanduel_live_checks_latest.json',summary)
    lines=['# Automated FanDuel checks','',summary['note'],'',f"State: {summary['state']}; attempts: {len(checks)}.",'',
           '| NBA alert | Player / market | Alert side / line / price | Status | Observed line / price | Check time UTC | Delay seconds | Timely |',
           '|---|---|---|---|---|---|---|---|']
    for r in checks:
        p=r['payload'];a=p['alert']
        observed=safe_time(r['observed_at'])
        delay=round((observed-safe_time(a['sent_at'])).total_seconds(),1) if observed else None
        lines.append(f"| #{r['alert_id']} | {a['player']} / {a['market']} | {a['side']} {a['line']} / {a['price']} | {r['status']} ({r['reason']}) | {p.get('line')} / {p.get('price')} | {observed} | {delay} | {p['timely']} |")
    (ROOT/'reports/fanduel_live_checks_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run_once():
    doc=active()
    with closing(psycopg2.connect(PG_DSN,connect_timeout=10)) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout='30s'")
            cur.execute('SELECT pg_try_advisory_lock(%s)',(LOCK,))
            if not cur.fetchone()[0]:return dict(status='already_running')
            cur.execute(SCHEMA)
        conn.commit()
        alerts=load_pending(conn,doc,utcnow());client=Client(doc['config']['state']);saved=0
        try:
            for alert in alerts:
                now=utcnow()
                if (now-safe_time(alert['sent_at'])).total_seconds()>300:
                    result=dict(unknown('check_deadline_missed'),attempted_at=now,receipts=[])
                else:result=observe(alert,client)
                saved+=save(conn,doc,alert,result,utcnow())
                if result['reason'] in ('access_blocked','rate_limited'):break
            result=report(conn,doc)
            return dict(status='ok',saved=saved,requests=client.calls,statuses=result['statuses'],manual_checks_added=0)
        finally:client.session.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','once','watch','verify'])
    p.add_argument('--state');p.add_argument('--minutes',type=int,default=180)
    args=p.parse_args()
    if args.command=='register':
        state=(args.state or _saved_windows_env('NFL_FANDUEL_STATE') or '').lower()
        doc=register(state)
        print(json.dumps(dict(status='registered',sha256=doc['sha256'],state=state)));return
    if args.command=='verify':
        doc=active();print(json.dumps(dict(status='verified',sha256=doc['sha256'])));return
    if args.command=='once':print(json.dumps(run_once()));return
    if not 1<=args.minutes<=720:p.error('--minutes must be between 1 and 720')
    deadline=time.monotonic()+args.minutes*60
    while time.monotonic()<deadline:
        print(json.dumps(run_once()),flush=True)
        time.sleep(min(30,max(0,deadline-time.monotonic())))


if __name__=='__main__':main()
