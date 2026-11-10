"""Isolated PostgreSQL checks; no production alerts, posts, or wagers."""
from datetime import timedelta
import os

import psycopg2
import pytest

from nba_props import fanduel_live as live
from sharp_pipeline import scan,paper_alerts
from sharp_pipeline.test_database_repairs import database
from sharp_pipeline.test_price_checks_database import DOC,NOW,START,init,snapshot,rows

pytestmark=pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')
OBSERVER=dict(sha256='b'*64,registered_at=(NOW-timedelta(minutes=1)).isoformat(),config=live.config('co'))


def setup(conn):
    init(conn)
    scan.store(conn,'nba',snapshot(),NOW,experiment=DOC)
    paper_alerts.dispatch(conn,DOC,now=NOW,poster=lambda text:'test-receipt')
    with conn.cursor() as cur:
        cur.execute(live.SCHEMA)
        cur.execute('CREATE TABLE nbaprops.game_projections (game_id text, home_team text, away_team text, tip_utc timestamptz)')
        cur.execute('INSERT INTO nbaprops.game_projections VALUES (%s,%s,%s,%s)',('game1','MIL','CHI',START))


def observation(**changes):
    return dict(dict(status='available',reason='exact_quote',attempted_at=NOW+timedelta(seconds=20),
        observed_at=NOW+timedelta(seconds=30),line=22.5,price=-130,receipts=[]),**changes)


def test_prospective_fanduel_only_with_native_abbreviations_and_earliest_tip(database):
    setup(database)
    found=live.load_pending(database,OBSERVER,NOW)
    assert len(found)==1 and found[0]['book']=='fanduel' and found[0]['native_tip']==START
    future=dict(OBSERVER,registered_at=(NOW+timedelta(seconds=1)).isoformat())
    assert live.load_pending(database,future,NOW)==[]  # do not reconstruct earlier alerts
    with database.cursor() as cur:
        cur.execute('INSERT INTO nbaprops.game_projections VALUES (%s,%s,%s,%s)',('game1','MIL','CHI',START-timedelta(minutes=10)))
    assert live.load_pending(database,OBSERVER,NOW)[0]['native_tip']==START-timedelta(minutes=10)


def test_first_check_is_immutable_and_does_not_create_manual_check(database,monkeypatch,tmp_path):
    setup(database)
    a=live.load_pending(database,OBSERVER,NOW)[0]
    assert live.save(database,OBSERVER,a,observation(),NOW+timedelta(seconds=31))
    assert not live.save(database,OBSERVER,a,observation(price=130),NOW+timedelta(seconds=32))
    assert live.load_pending(database,OBSERVER,NOW)==[]
    assert rows(database,'nba_paper_price_checks')==[]
    with database.cursor() as cur:
        cur.execute('SAVEPOINT unchanged')
        with pytest.raises(psycopg2.errors.RaiseException):
            cur.execute("UPDATE bets.fanduel_live_checks SET status='unknown'")
        cur.execute('ROLLBACK TO SAVEPOINT unchanged')
    monkeypatch.setattr(live,'ROOT',tmp_path)
    result=live.report(database,OBSERVER)
    assert result['attempts']==result['timely_usable_observations']==1
    assert result['checks'][0]['payload']['price']==-130
    assert result['manual_checks_added']==0 and result['betting_authorized'] is False
    assert live.report(database,dict(OBSERVER,sha256='c'*64))['attempts']==0


def test_unknown_first_attempt_cannot_be_replaced_and_late_check_stays_late(database,monkeypatch,tmp_path):
    setup(database);a=live.load_pending(database,OBSERVER,NOW)[0]
    unknown=dict(live.unknown('network_Timeout'),attempted_at=NOW,receipts=[])
    assert live.save(database,OBSERVER,a,unknown,NOW)
    assert not live.save(database,OBSERVER,a,observation(),NOW+timedelta(seconds=31))
    monkeypatch.setattr(live,'ROOT',tmp_path)
    result=live.report(database,OBSERVER)
    assert result['statuses']=={'unknown':1} and result['timely_usable_observations']==0
    scan.store(database,'nba',snapshot(eid='new'),NOW,experiment=DOC)
    paper_alerts.dispatch(database,DOC,now=NOW,poster=lambda text:'test-receipt-2')
    a=live.load_pending(database,OBSERVER,NOW)[0]
    late=NOW+timedelta(minutes=6)
    assert live.save(database,OBSERVER,a,observation(observed_at=late),late)
    result=live.report(database,OBSERVER)
    assert result['attempts']==2 and result['timely_usable_observations']==0
