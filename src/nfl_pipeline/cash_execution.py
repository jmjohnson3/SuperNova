"""Append-only cash recommendation/execution events on the existing NFL ledger."""
import argparse
from datetime import datetime, timezone, timedelta
import json

import psycopg2
import psycopg2.extras

from nfl_pipeline import cash_trial_policy as policy
from nfl_pipeline.betting_preferences import execution_link
from nfl_pipeline.fanduel_links import single_betslip_url
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.offer_selection import forecast_quote_error, valid_price

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.nfl_cash_execution_events (
  event_id BIGSERIAL PRIMARY KEY,
  ledger_id BIGINT NOT NULL REFERENCES bets.nfl_bet_ledger(ledger_id),
  state TEXT NOT NULL CHECK(state IN ('reserved','published','confirmed','not_placed','settled','paused')),
  payload JSONB NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(ledger_id,state)
);
CREATE OR REPLACE FUNCTION bets.nfl_cash_events_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'NFL cash execution events are append-only'; END; $$;
DROP TRIGGER IF EXISTS nfl_cash_events_immutable ON bets.nfl_cash_execution_events;
CREATE TRIGGER nfl_cash_events_immutable BEFORE UPDATE OR DELETE ON bets.nfl_cash_execution_events
FOR EACH ROW EXECUTE FUNCTION bets.nfl_cash_events_immutable();
"""


def schema(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('bets.nfl_cash_execution_events')")
        if cur.fetchone()[0]:
            return
        cur.execute("SET LOCAL lock_timeout='5s'")
        cur.execute(SCHEMA_SQL)


def lock(conn):
    with conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='5s'")
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('nfl_cash_execution'))")


def states(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT DISTINCT ON (ledger_id) ledger_id,state,payload
          FROM bets.nfl_cash_execution_events ORDER BY ledger_id,event_id DESC""")
        return [dict(r['payload'], ledger_id=r['ledger_id'], state=r['state']) for r in cur.fetchall()]


def append(conn, ledger_id, state, payload):
    with conn.cursor() as cur:
        cur.execute('INSERT INTO bets.nfl_cash_execution_events(ledger_id,state,payload) VALUES(%s,%s,%s)',
                    (ledger_id, state, psycopg2.extras.Json(payload)))


def candidate_error(row, now):
    error = forecast_quote_error(row, now)
    if error:
        return error
    if (row.get('book') != 'fanduel' or not execution_link(row.get('link'))
            or not single_betslip_url(row.get('link'))):
        return 'missing_fanduel_link'
    if row.get('drift_guard_pass') is not True or not valid_price(row.get('price')):
        return 'drift_or_price_invalid'
    p = row.get('probability'); push = row.get('push_probability', 0)
    if not isinstance(p, (int, float)) or not 0 < p < 1 or not 0 <= push < 1:
        return 'invalid_probability'
    price = float(row['price']); payout = price/100 if price > 0 else 100/-price
    if (1-push)*(p*(1+payout)-1) <= 1e-12:
        return 'nonpositive_current_ev'
    if row.get('market_probability') is None:
        return 'missing_true_pair'
    return None


def reserve(conn, rows, registration, decision, now):
    """Reserve before network publication. Uncertain sends remain reserved."""
    lock(conn)
    existing = states(conn)
    if decision != 'cash_trial_eligible' or (policy.STORE/'pause.json').exists():
        return [], {'policy_not_eligible': len(rows)}
    accepted = []; excluded = {}
    for row in rows:
        q = (row.get('scoring_replay') or {}).get('offer') or {}
        start = safe_time(row.get('start_ts_utc') or q.get('commence_time_utc'))
        expires = min(start, safe_time(row['quote_fetched_at_utc'])+timedelta(minutes=20)) if start else now
        kind = row.get('source_kind', 'prop')
        player = str(row.get('player_id')) if kind == 'prop' else row.get('market')
        candidate = dict(decision_key=f"{kind}|{row['game_id']}|{player}", day=str(row['game_date_et'])[:10],
            week_key=f"{row['season']}:{row['week']}", stake=1., expires_at=expires.isoformat())
        error = candidate_error(row, now) or policy.budget_error(existing, candidate, now)
        if error:
            excluded[error] = excluded.get(error, 0)+1; continue
        payload = dict(candidate, source_kind=kind, forecast_id=int(row['forecast_id']),
            state='reserved', registration_sha256=registration['sha256'], forecast=row,
            reserved_at=now.isoformat())
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO bets.nfl_bet_ledger(source_kind,prediction_id,game_date_et,tier,stake,
              book,market,stat,side,line,price,link,model_version,prediction_key)
              VALUES(%s,%s,%s,'cash_trial',1,%s,%s,%s,%s,%s,%s,%s,%s,%s)
              ON CONFLICT DO NOTHING RETURNING ledger_id""",
              (kind,row['forecast_id'],candidate['day'],row['book'],row.get('market'),row.get('stat'),
               row['side'],row['line'],row['price'],row['link'],row['model_version'],row.get('prediction_key')))
            result = cur.fetchone()
        if not result:
            continue
        ledger_id = result[0]
        append(conn, ledger_id, 'reserved', payload)
        existing.append(dict(payload, ledger_id=ledger_id))
        accepted.append(dict(row, cash_ledger_id=ledger_id, tier='cash_trial', ledger_locked=True,
                             cash_policy=registration['sha256'], expires_at=expires.isoformat()))
    return accepted, excluded


def publication(ledger_ids):
    if not ledger_ids:
        return
    with psycopg2.connect(PG_DSN) as conn:
        lock(conn)
        for row in states(conn):
            if row['ledger_id'] in ledger_ids and row['state'] == 'reserved':
                append(conn, row['ledger_id'], 'published', dict(row, published_at=datetime.now(timezone.utc).isoformat()))


def transition(row, action, now, *, price=None, result=None, stake=1.0):
    allowed = {'confirmed': ('reserved','published'), 'not_placed': ('reserved','published'),
               'settled': ('confirmed',)}
    if action not in allowed or row['state'] not in allowed[action]:
        raise ValueError('Invalid cash execution state transition')
    payload = dict(row)
    if action == 'confirmed':
        if not valid_price(price):
            raise ValueError('Confirm the actual executed American price')
        import math
        if not math.isfinite(stake) or stake <= 0:
            raise ValueError('Confirm a positive actual stake')
        # Record off-policy execution honestly instead of excluding an adverse fill.
        forecast = row['forecast']; p = forecast['probability']; payout = price/100 if price > 0 else 100/-price
        payload.update(executed_price=price, stake=stake, execution_confirmed_at=now.isoformat(),
            execution_deviation=stake != 1 or now > safe_time(row['expires_at']) or p*(1+payout)-1 <= 1e-12)
    if action == 'settled':
        if result not in ('win','loss','push','void'):
            raise ValueError('Invalid wager result')
        price = payload['executed_price']; payout = price/100 if price > 0 else 100/-price
        payload.update(result=result, profit=payload['stake']*(payout if result=='win' else -1 if result=='loss' else 0))
    payload['state'] = action
    return payload


def reconcile(ledger_id, action, price=None, stake=1.0):
    with psycopg2.connect(PG_DSN) as conn:
        lock(conn)
        row = next((r for r in states(conn) if r['ledger_id']==ledger_id), None)
        if row is None:
            raise ValueError('Unknown cash ledger ID')
        payload = transition(row, action, datetime.now(timezone.utc), price=price, stake=stake)
        append(conn, ledger_id, action, payload)
    return {'ledger_id': ledger_id, 'state': action}


def settle():
    with psycopg2.connect(PG_DSN) as conn:
        lock(conn)
        for row in states(conn):
            if row['state'] != 'confirmed':
                continue
            table = {'prop':'nfl_player_prop_prediction_results', 'game':'nfl_game_prediction_results'}[row['source_kind']]
            with conn.cursor() as cur:
                cur.execute(f'SELECT result FROM bets.{table} WHERE prediction_id=%s', (row['forecast_id'],))
                result = cur.fetchone()
            outcome = 'void' if result and str(result[0]).startswith('void') else result[0] if result else None
            if outcome in ('win','loss','push','void'):
                payload = transition(row, 'settled', datetime.now(timezone.utc), result=outcome)
                append(conn, row['ledger_id'], 'settled', payload)
        current = states(conn)
        profit = sum(float(r.get('profit') or 0) for r in current if r['state'] in ('settled','paused'))
        if profit <= -policy.POLICY.loss_limit and not any(r['state']=='paused' for r in current):
            # Sticky stop even if later wins bring net loss back above the limit.
            row = next(r for r in current if r['state']=='settled')
            append(conn, row['ledger_id'], 'paused', dict(row, pause_reason='cumulative_loss_limit'))
        return dict(confirmed_profit=profit, states=dict(__import__('collections').Counter(r['state'] for r in states(conn))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup', action='store_true')
    p.add_argument('--settle', action='store_true')
    p.add_argument('--ledger-id', type=int)
    p.add_argument('--action', choices=['confirmed','not_placed'])
    p.add_argument('--price', type=int)
    p.add_argument('--stake', type=float, default=1., help='Actual executed stake; off-policy amounts pause further eligibility')
    a = p.parse_args()
    if a.setup:
        with psycopg2.connect(PG_DSN) as conn:
            schema(conn)
        print('NFL cash event schema ready; no wagers placed')
    elif a.settle:
        print(json.dumps(settle()))
    elif a.ledger_id and a.action:
        print(json.dumps(reconcile(a.ledger_id, a.action, a.price, a.stake)))
    else:
        p.error('Use --setup, --settle, or --ledger-id and --action')
