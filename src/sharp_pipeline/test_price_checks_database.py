"""Full selection/delivery/check/report path in a rolled-back PostgreSQL schema."""
import os
from datetime import timedelta

import pandas as pd
import psycopg2.extras
import pytest

from sharp_pipeline import nba_experiment as experiment, paper_alerts, price_checks, report, scan
from sharp_pipeline.test_database_repairs import database  # isolated transactional fixture
from sharp_pipeline.test_sharp_multi import NOW, START, bookmaker, event, market

pytestmark = pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS') != '1', reason='opt-in local database test')
DOC = dict(experiment_id=experiment.NAME, sha256='a' * 64, registered_at=(NOW - timedelta(hours=1)).isoformat())


def snapshot(at=NOW, eid='e1', player='Test Player'):
    def prop(over, under):
        return market('player_points', [dict(name='Over', description=player, point=22.5, price=over),
                                       dict(name='Under', description=player, point=22.5, price=under)], at)
    return event([bookmaker('fanduel', [prop(110,-140)], at),
                  bookmaker('draftkings', [prop(120,-150)], at),
                  bookmaker('pinnacle', [prop(-110,-110)], at)], eid)


def init(conn):
    with conn.cursor() as cur:
        cur.execute(scan.SCHEMA_SQL)
        cur.execute(paper_alerts.SCHEMA_SQL)


def rows(conn, table):
    assert table in ('sharp_multi_decisions', 'nba_paper_price_alerts', 'nba_paper_price_checks')
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute('SELECT * FROM bets.' + table)
        return [dict(r) for r in cur.fetchall()]


def test_independent_books_repeats_and_pre_registration_are_not_backfilled(database):
    init(database)
    scan.store(database, 'nba', snapshot(NOW - timedelta(minutes=25)), NOW - timedelta(minutes=25), experiment=DOC)
    assert not rows(database, 'nba_paper_price_alerts')  # early is not a check
    scan.store(database, 'nba', snapshot(), NOW, experiment=DOC)
    scan.store(database, 'nba', snapshot(), NOW, experiment=DOC)  # exact repeat
    scan.store(database, 'nba', snapshot(NOW + timedelta(minutes=5)), NOW + timedelta(minutes=5), experiment=DOC)
    alerts = rows(database, 'nba_paper_price_alerts')
    assert {r['book'] for r in alerts} == {'fanduel','draftkings'}
    assert len(alerts) == 2
    assert len(rows(database, 'sharp_multi_decisions')) == 6
    future_doc = dict(DOC, registered_at=(NOW + timedelta(seconds=1)).isoformat())
    scan.store(database, 'nba', snapshot(eid='pre-registration'), NOW, experiment=future_doc)
    assert len(rows(database, 'nba_paper_price_alerts')) == 2


def test_delivery_and_first_live_check_are_immutable(database):
    init(database)
    scan.store(database, 'nba', snapshot(), NOW, experiment=DOC)
    messages = []
    def poster(text):
        messages.append(text)
        return str(len(messages))
    assert paper_alerts.dispatch(database, DOC, now=NOW, poster=poster)['sent'] == 2
    assert paper_alerts.dispatch(database, DOC, now=NOW, poster=poster)['sent'] == 0
    assert len(messages) == 2
    alert = rows(database, 'nba_paper_price_alerts')[0]
    saved = price_checks.record(database, alert['alert_id'], checked_at=NOW + timedelta(seconds=30),
                               now=NOW + timedelta(seconds=45), status='available', line=22.5, price=-130)
    assert saved['displayed_price'] == -130 and saved['evidence_source'] == 'manual_live'
    with pytest.raises(ValueError, match='cannot be overwritten'):
        price_checks.record(database, alert['alert_id'], checked_at=NOW + timedelta(seconds=50),
                            now=NOW + timedelta(seconds=55), status='available', line=22.5, price=120)
    assert rows(database, 'nba_paper_price_checks')[0]['displayed_price'] == -130


def test_failed_delivery_is_not_retried_and_stale_requests_expire(database):
    init(database)
    scan.store(database, 'nba', snapshot(), NOW, experiment=DOC)
    def fails(text):
        raise TimeoutError('ambiguous HTTP receipt')
    result = paper_alerts.dispatch(database, DOC, now=NOW, poster=fails)
    assert result['uncertain'] == 2 and result['sent'] == 0
    assert paper_alerts.dispatch(database, DOC, now=NOW, poster=lambda text: 'bad')['sent'] == 0
    scan.store(database, 'nba', snapshot(eid='late'), NOW, experiment=DOC)
    assert paper_alerts.dispatch(database, DOC, now=NOW + timedelta(minutes=6), poster=lambda text: 'bad')['expired'] == 2


def test_database_rejects_available_without_price(database):
    init(database)
    scan.store(database, 'nba', snapshot(), NOW, experiment=DOC)
    alert = rows(database, 'nba_paper_price_alerts')[0]
    with database.cursor() as cur:
        cur.execute('SAVEPOINT invalid_check')
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("""INSERT INTO bets.nba_paper_price_checks (alert_id,checked_at,status,displayed_line)
                           VALUES (%s,%s,'available',22.5)""", (alert['alert_id'],NOW))
        cur.execute('ROLLBACK TO SAVEPOINT invalid_check')


def test_report_selects_frozen_cohort_by_book_and_blocks_on_drift(database, monkeypatch, tmp_path):
    init(database)
    (tmp_path / 'reports').mkdir()
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'active', lambda: DOC)
    monkeypatch.setattr(experiment, 'load', lambda: DOC)
    # Avoid production outcome/model tables; this test grades close EV and execution only.
    monkeypatch.setattr(report.outcomes, 'attach', lambda conn, df: df)
    scan.store(database, 'nba', snapshot(eid='legacy'), NOW)
    scan.store(database, 'nba', snapshot(), NOW, experiment=DOC)
    paper_alerts.dispatch(database, DOC, now=NOW, poster=lambda text: 'receipt')
    alerts = rows(database, 'nba_paper_price_alerts')
    for a in alerts:
        price_checks.record(database, a['alert_id'], checked_at=NOW + timedelta(seconds=30),
                            now=NOW + timedelta(seconds=40), status='available', line=22.5,
                            price=-130 if a['book'] == 'fanduel' else 120)
    raw = pd.DataFrame(rows(database, 'sharp_multi_decisions'))
    at = START - timedelta(minutes=5)
    closes = {('nba','e1','props'): (at, snapshot(at))}
    result = experiment.write_report(database, raw, closes, START + timedelta(hours=3))
    assert result['source_status'] == 'frozen'
    assert result['selected_requests'] == 2 and result['other_nba_observations'] == 2
    fd, dk = result['by_book']['fanduel'], result['by_book']['draftkings']
    assert fd['decisions'] == dk['decisions'] == 1
    assert fd['mean_checked_price_ev'] < 0 < dk['mean_checked_price_ev']
    assert fd['with_exact_close'] == 1 and fd['priced_checks'] == 1
    def drift():
        raise ValueError('changed source')
    monkeypatch.setattr(experiment, 'active', drift)
    blocked = experiment.write_report(database, raw, closes, START + timedelta(hours=3))
    assert all(m['status'] == 'blocked_source_drift' for m in blocked['by_book'].values())
    assert blocked['betting_authorized'] is False
