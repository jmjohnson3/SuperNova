"""Live paper checks and model-comparison cards; no wager placement or confirmation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import psycopg2
import psycopg2.extras
import requests
from model_research import PG_DSN, NAME, active, rows, safe_time
from sharp_pipeline.price_checks import checked_time, validate


def card(row):
    offer,decision=row['payload']['offer'],row['payload']['decision']
    parent=offer.get('parent_check_id')
    name=offer.get('player') or f"{offer['away_team']} @ {offer['home_team']}"
    title='MODEL COMPARISON' if parent else 'PAPER PRICE CHECK'
    state='SELECTED IN MODEL PAPER STRATEGY' if decision['selected'] else 'VETOED BY MODEL PAPER STRATEGY'
    lines=[f"**{title} — {row['sport'].upper()} M{row['id']}**",
           f"{name}: {offer['side'].upper()} {float(offer['line']):g} {offer['market']}",
           f"FanDuel {int(offer['price']):+d} | observed {offer['decided_at']}",
           f"**{state}** ({decision['reason']})",
           f"Price-only control: selected | reference estimate {float(offer['fair_probability']):.1%}"]
    f=decision.get('forecast') or {}
    if decision.get('model_probability') is not None:
        lines.append(f"Player model: {decision['model_probability']:.1%} | model EV {decision['model_ev']:+.1%}")
    if decision.get('directional_margin') is not None:
        margin='unknown' if f.get('pred_margin') is None else f"{float(f['pred_margin']):+.1f}"
        total='unknown' if f.get('pred_total') is None else f"{float(f['pred_total']):.1f}"
        lines.append(f"Game model: home margin {margin}, total {total}; side cushion {decision['directional_margin']:+.1f}")
    if parent:
        lines.append(f"Use the original NBA price-check #{parent}; record within 5 minutes of that alert. This comparison does not reset its deadline.")
    else:
        lines.append(f"Check within 5 minutes. Report M{row['id']}, actual check time, line/price or disappeared/suspended in Codex; Discord replies are not imported.")
    lines.append('Paper research only. No wager requested; model agreement is unvalidated. Save the first check even if the price worsens.')
    return '\n'.join(lines)[:1950]


def webhook(sport):
    from supernovabets_config import _saved_windows_env
    key='NBA_DISCORD_WEBHOOK_URL' if sport=='nba' else 'NFL_DISCORD_WEBHOOK_URL'
    return os.getenv(key) or _saved_windows_env(key) or ''


def dispatch(conn,doc,sport,*,poster=None,now=None):
    if poster is None and not webhook(sport): return dict(status='webhook_missing',sent=0)
    count=0
    while True:
        # Lock the decision until claiming its delivery. The claim commits before HTTP.
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT d.* FROM bets.model_research_decisions d
                LEFT JOIN bets.model_research_delivery m USING(id)
                WHERE d.experiment_id=%s AND d.experiment_sha256=%s AND d.sport=%s AND m.id IS NULL
                ORDER BY d.id LIMIT 1 FOR UPDATE OF d SKIP LOCKED""",(NAME,doc['sha256'],sport))
            row=cur.fetchone()
            if not row: conn.commit();break
            at=now or datetime.now(timezone.utc)
            offer=row['payload']['offer']
            age=(at-safe_time(offer['decided_at'])).total_seconds()
            expired=not 0<=age<=300 or at>=safe_time(offer['commence_time_utc'])
            cur.execute("INSERT INTO bets.model_research_delivery(id,delivery_status,attempted_at) VALUES(%s,%s,%s)",
                        (row['id'],'expired' if expired else 'sending',at))
        conn.commit()
        if expired: continue
        try:
            if poster: message_id=poster(card(row))
            else:
                response=requests.post(webhook(sport),params={'wait':'true'},
                    json={'content':card(row),'allowed_mentions':{'parse':[]}},timeout=20)
                if response.status_code!=200: raise RuntimeError('Discord rejected request')
                message_id=response.json().get('id')
            if not message_id: raise RuntimeError('No confirmed message ID')
            state,error='sent',None
        except Exception as exc:
            state,error,message_id='uncertain',type(exc).__name__,None
        sent_at=(now or datetime.now(timezone.utc)) if state=='sent' else None
        with conn.cursor() as cur:
            cur.execute("""UPDATE bets.model_research_delivery SET delivery_status=%s,sent_at=%s,
                message_id=%s,error=%s WHERE id=%s""",(state,sent_at,message_id,error,row['id']))
        conn.commit()
        count+=state=='sent'
    return dict(status='ok',sent=count)


def record(conn,identity,*,checked_at,status,line=None,price=None,note=None,now=None):
    now=now or datetime.now(timezone.utc)
    found=rows(conn,"""SELECT d.*,m.delivery_status,m.sent_at FROM bets.model_research_decisions d
        JOIN bets.model_research_delivery m USING(id) WHERE d.id=%s""",(identity,))
    if not found: raise ValueError('Unknown or undelivered model check')
    row=found[0];offer=row['payload']['offer']
    if offer.get('parent_check_id'):
        raise ValueError(f"Record the original NBA price-check #{offer['parent_check_id']}; it is shared by both strategies")
    validate(dict(row,line=offer['line']),checked_at=checked_at,now=now,status=status,line=line,price=price)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO bets.model_research_checks
            (id,checked_at,recorded_at,status,displayed_line,displayed_price,note)
            VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id""",
            (identity,checked_at,now,status,line,price,note))
        if cur.fetchone() is None: raise ValueError('First check already recorded; it cannot be replaced')
    conn.commit()
    return dict(id=identity,status=status,wager_placed=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['pending','record'])
    ap.add_argument('--id',type=int);ap.add_argument('--checked-at',type=checked_time)
    ap.add_argument('--status',choices=['available','changed_line','disappeared','suspended'])
    ap.add_argument('--line',type=float);ap.add_argument('--price',type=int);ap.add_argument('--note')
    args=ap.parse_args()
    if args.command=='record' and not all((args.id,args.checked_at,args.status)):
        ap.error('record requires --id, --checked-at and --status')
    with psycopg2.connect(PG_DSN,connect_timeout=10) as conn:
        if args.command=='record':
            result=record(conn,args.id,checked_at=args.checked_at,status=args.status,line=args.line,price=args.price,note=args.note)
        else:
            result=rows(conn,"""SELECT d.id,d.sport,d.payload->'offer' AS offer,m.sent_at FROM bets.model_research_decisions d
                JOIN bets.model_research_delivery m USING(id) LEFT JOIN bets.model_research_checks c USING(id)
                WHERE m.delivery_status='sent' AND c.id IS NULL AND (d.payload->'offer'->>'parent_check_id') IS NULL
                AND (d.payload->'offer'->>'commence_time_utc')::timestamptz>now()""")
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
