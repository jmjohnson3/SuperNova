"""Guided recording of a real FanDuel paper check using the existing frozen ledger.

python -m nba_props.fanduel_check pending
python -m nba_props.fanduel_check record --alert 123

No sportsbook access, price inference, wager placement, or Discord messages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from sharp_pipeline import price_checks
from sharp_pipeline.nba_experiment import MAX_CHECK_DELAY_SECONDS, MAX_RECORD_LAG_SECONDS

DENVER = ZoneInfo('America/Denver')
STATUSES = {'a':'available','c':'changed_line','d':'disappeared','s':'suspended'}


def pending(conn):
    """Only the requested book; keep late but still-pregame requests visible."""
    return [r for r in price_checks.pending(conn) if r['book']=='fanduel']


def get_alert(conn, identity):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT a.alert_id,a.sent_at,a.delivery_status,d.player,d.market,d.side,
            d.line,d.price,d.book,d.decided_at AS observed_at,d.commence_time_utc,
            c.alert_id AS recorded_check
            FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING(decision_id)
            LEFT JOIN bets.nba_paper_price_checks c USING(alert_id) WHERE a.alert_id=%s""",(identity,))
        row=cur.fetchone()
    if not row or row['book']!='fanduel' or row['delivery_status']!='sent':
        raise ValueError('Choose a confirmed FanDuel NBA paper alert ID')
    if row['recorded_check'] is not None:
        raise ValueError('The first check is already saved and cannot be replaced')
    return dict(row)


def describe(row, now):
    deadline=row['sent_at']+timedelta(seconds=MAX_CHECK_DELAY_SECONDS)
    timing=f"check by {deadline.astimezone(DENVER):%I:%M:%S %p %Z}"
    if now>deadline:timing+=' (late; a real observation can still be saved)'
    return (f"NBA #{row['alert_id']} | {row['player']} | {row['side']} {float(row['line']):g} {row['market']}\n"
            f"FanDuel alert price {int(row['price']):+d} | observed {row['observed_at'].astimezone(DENVER):%I:%M:%S %p %Z}\n"
            f"Alert sent {row['sent_at'].astimezone(DENVER):%I:%M:%S %p %Z} | {timing}\n"
            f"Game starts {row['commence_time_utc'].astimezone(DENVER):%I:%M %p %Z}")


def collect_observation(row, *, ask=input, clock=lambda: datetime.now(timezone.utc), emit=print):
    """No defaults from the API quote; the user supplies each observed value."""
    if clock()>=row['commence_time_utc']:
        raise ValueError('This guided tool only accepts pregame checks')
    emit(describe(row,clock()))
    emit('Read this exact player, market and side on FanDuel. No wager is needed.')
    emit('A blocked page or login problem is not a disappeared offer; cancel if you cannot see the market.')
    text=ask('Observed status: [a] exact line available, [c] line changed, [d] disappeared, [s] suspended, [q] cancel: ').strip().lower()
    if text in {'q',''}:return None
    status=STATUSES.get(text,text)
    line=price=None
    if status in {'available','changed_line'}:
        line=float(ask('Actual displayed line (do not copy the alert without checking): ').strip())
        price=int(ask('Actual displayed American odds, including sign: ').strip())
    elif status not in {'disappeared','suspended'}:
        raise ValueError('Unknown observed status; nothing saved')
    text=ask('Exact checking time with UTC offset, or type now only if you just checked (blank cancels): ').strip()
    if not text:return None
    checked=clock() if text.lower()=='now' else price_checks.checked_time(text)
    at=clock()
    price_checks.validate(row,checked_at=checked,now=at,status=status,line=line,price=price)
    if checked>=row['commence_time_utc'] or at>=row['commence_time_utc']:
        raise ValueError('The game has started; no pregame check saved')
    # Existing report rules determine timeliness. Never change the observed time.
    timely=(checked-row['sent_at']).total_seconds()<=MAX_CHECK_DELAY_SECONDS and (at-checked).total_seconds()<=MAX_RECORD_LAG_SECONDS
    if not timely:emit('This is a late observation; it will not satisfy the timely-check requirement.')
    return dict(checked_at=checked,status=status,line=line,price=price,
                note='User observed FanDuel directly; entered with guided paper-check tool.')


def record_guided(conn, identity, *, ask=input, clock=lambda: datetime.now(timezone.utc), emit=print):
    row=get_alert(conn,identity)
    observation=collect_observation(row,ask=ask,clock=clock,emit=emit)
    if observation is None:return dict(status='cancelled',saved=False)
    saved=price_checks.record(conn,identity,now=clock(),**observation)
    return dict(saved=True,check=saved,wager_placed=False,
                note='Saved immediately to the original NBA check. Linked model comparisons reuse this same observation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=['pending','record'])
    parser.add_argument('--alert',type=int)
    args=parser.parse_args()
    if args.command=='record' and args.alert is None:parser.error('record requires --alert')
    try:
        with psycopg2.connect(PG_DSN,connect_timeout=10) as conn:
            if args.command=='pending':
                found=pending(conn)
                if not found:print('No delivered FanDuel NBA paper alerts await a pregame check.')
                for row in found:print(describe(row,datetime.now(timezone.utc))+'\n')
            else:print(json.dumps(record_guided(conn,args.alert),indent=2,default=str))
    except (ValueError,EOFError,KeyboardInterrupt) as exc:
        parser.exit(1,f'Check not saved: {str(exc) or "cancelled"}\n')


if __name__=='__main__':main()
