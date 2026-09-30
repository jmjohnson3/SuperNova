"""Opt-in local PostgreSQL check; every inserted test recommendation is rolled back."""
from datetime import datetime, timedelta, timezone
import os

import psycopg2
import pytest

from nfl_pipeline import cash_execution as cash
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.offer_selection import CONTRACT


@pytest.mark.skipif(os.getenv('NFL_TEST_CASH_DB') != '1', reason='Opt-in transaction-rollback PostgreSQL test')
def test_atomic_reservation_dedup_and_append_only():
    now = datetime.now(timezone.utc)
    row = dict(forecast_id=-987654321, source_kind='prop', game_id='cash-test-rollback-only', player_id='test',
        game_date_et=str(now.date()), season=2099,week=99,book='fanduel',stat='receiving_yards',
        link='https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=2',
        execution_contract=CONTRACT, quote_fetched_at_utc=(now-timedelta(seconds=1)).isoformat(),
        prediction_context_cutoff_utc=now.isoformat(),start_ts_utc=(now+timedelta(hours=1)).isoformat(),
        probability=.6,push_probability=0,market_probability=.5,price=110,line=45.5,side='over',
        drift_guard_pass=True,model_version='rollback-test',prediction_key='cash-test-rollback-only')
    conn = psycopg2.connect(PG_DSN)
    other = psycopg2.connect(PG_DSN)
    try:
        cash.lock(conn)
        with other.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_xact_lock(hashtext('nfl_cash_execution'))")
            assert cur.fetchone()[0] is False
        before = len(cash.states(conn))
        accepted, errors = cash.reserve(conn,[row],{'sha256':'test-only'},'cash_trial_eligible',now)
        assert len(accepted)==1 and not errors
        again, errors = cash.reserve(conn,[row],{'sha256':'test-only'},'cash_trial_eligible',now)
        assert not again and errors['decision_already_reserved']==1
        assert len(cash.states(conn))==before+1
        ledger_id=accepted[0]['cash_ledger_id']
        with conn.cursor() as cur:
            cur.execute('SAVEPOINT immutable_event')
            with pytest.raises(psycopg2.errors.RaiseException):
                cur.execute("UPDATE bets.nfl_cash_execution_events SET state='confirmed' WHERE ledger_id=%s",(ledger_id,))
            cur.execute('ROLLBACK TO SAVEPOINT immutable_event')
    finally:
        conn.rollback();conn.close();other.rollback();other.close()
    with psycopg2.connect(PG_DSN) as verify, verify.cursor() as cur:
        cur.execute("SELECT count(*) FROM bets.nfl_bet_ledger WHERE prediction_key='cash-test-rollback-only'")
        assert cur.fetchone()[0]==0
