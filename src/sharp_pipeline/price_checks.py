"""Record an actual manual sportsbook check; no wagers are placed or confirmed here.

python -m sharp_pipeline.price_checks pending
python -m sharp_pipeline.price_checks record --alert 123 --checked-at 2026-10-28T17:30:00-06:00 --status available --line 22.5 --price -110
python -m sharp_pipeline.price_checks record --alert 123 --checked-at 2026-10-28T17:30:00-06:00 --status disappeared
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.sharp_math import implied


def checked_time(value: str) -> datetime:
    at = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError('Check time must include a UTC offset, such as -06:00 or Z')
    return at.astimezone(timezone.utc)


def validate(alert: dict, *, checked_at: datetime, now: datetime, status: str,
             line: float | None, price: int | None):
    if checked_at.tzinfo is None or now.tzinfo is None:
        raise ValueError('Timezone-aware timestamps are required')
    if not alert.get('sent_at') or alert.get('delivery_status') != 'sent':
        raise ValueError('This request has no confirmed alert delivery')
    if not alert['sent_at'] <= checked_at <= now:
        raise ValueError('Check time must be after alert delivery and no later than now')
    if status not in {'available', 'changed_line', 'disappeared', 'suspended'}:
        raise ValueError('Unknown check status')
    if status in {'available', 'changed_line'}:
        if line is None or not math.isfinite(float(line)) or price is None or not implied(price) or int(price) != price:
            raise ValueError('Record the finite displayed line and a valid integer American price')
        same = abs(float(line) - float(alert['line'])) < 1e-9
        if (status == 'available') != same:
            raise ValueError('Use available for the exact requested line; use changed_line for another line')
    elif line is not None or price is not None:
        raise ValueError('An absent/suspended offer cannot have a displayed line or price')


def record(conn, alert_id: int, *, checked_at: datetime, status: str, line=None, price=None, note=None, now=None):
    now = now or datetime.now(timezone.utc)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT a.*, d.line FROM bets.nba_paper_price_alerts a
                       JOIN bets.sharp_multi_decisions d USING (decision_id) WHERE a.alert_id=%s FOR UPDATE OF a""", (alert_id,))
        alert = cur.fetchone()
        if alert is None:
            raise ValueError('Unknown paper check ID')
        validate(dict(alert), checked_at=checked_at, now=now, status=status, line=line, price=price)
        cur.execute('SELECT * FROM bets.nba_paper_price_checks WHERE alert_id=%s', (alert_id,))
        if cur.fetchone() is not None:
            raise ValueError('A first check is already recorded; it cannot be overwritten or replaced by a later price')
        cur.execute("""INSERT INTO bets.nba_paper_price_checks
            (alert_id, checked_at, recorded_at, status, displayed_line, displayed_price, note)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (alert_id, checked_at, now, status, line, price, note))
        saved = dict(cur.fetchone())
    conn.commit()
    return saved


def pending(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT a.alert_id, a.sent_at, d.player, d.market, d.side, d.line, d.book, d.price,
                              d.decided_at AS observed_at, d.commence_time_utc
            FROM bets.nba_paper_price_alerts a JOIN bets.sharp_multi_decisions d USING (decision_id)
            LEFT JOIN bets.nba_paper_price_checks c USING (alert_id)
            WHERE a.delivery_status='sent' AND c.alert_id IS NULL AND d.commence_time_utc > now()
            ORDER BY a.sent_at""")
        return [dict(r) for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('pending')
    rec = sub.add_parser('record')
    rec.add_argument('--alert', type=int, required=True)
    rec.add_argument('--checked-at', type=checked_time, required=True)
    rec.add_argument('--status', choices=['available', 'changed_line', 'disappeared', 'suspended'], required=True)
    rec.add_argument('--line', type=float)
    rec.add_argument('--price', type=int)
    rec.add_argument('--note')
    args = ap.parse_args()
    try:
        with psycopg2.connect(PG_DSN) as conn:
            result = pending(conn) if args.command == 'pending' else record(conn, args.alert,
                checked_at=args.checked_at, status=args.status, line=args.line, price=args.price, note=args.note)
    except ValueError as exc:
        ap.error(str(exc))
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
