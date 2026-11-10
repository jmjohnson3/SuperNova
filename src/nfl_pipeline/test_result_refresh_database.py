"""Opt-in regressions for receiving evidence surviving a later stats import.

All writes go to a temporary table and are rolled back, including commits made
by the importer under test. No production player rows are changed.
"""
import os
from datetime import date

import psycopg2
import psycopg2.extras
import pytest

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.import_nflverse import _upsert_players

pytestmark = pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS') != '1', reason='opt-in database test')


@pytest.fixture
def database(monkeypatch):
    conn = psycopg2.connect(PG_DSN, connect_timeout=5)
    with conn.cursor() as cur:
        cur.execute('CREATE TEMP TABLE receiving_import_test (LIKE raw.nfl_player_gamelogs INCLUDING ALL)')
    original = psycopg2.extras.execute_values

    def isolated(cur, query, rows, **kwargs):
        assert 'INSERT INTO raw.nfl_player_gamelogs AS g' in query
        return original(cur, query.replace('raw.nfl_player_gamelogs', 'pg_temp.receiving_import_test'), rows, **kwargs)

    monkeypatch.setattr(psycopg2.extras, 'execute_values', isolated)

    class Connection:
        cursor = conn.cursor

        def commit(self):
            pass

    try:
        yield Connection()
    finally:
        conn.rollback()
        conn.close()


def incoming(yards=0, targets=0, receptions=0, payload=None):
    return (2026, 6, 'audit_game', date(2026, 10, 18), 'audit_player', 'Player', 'DAL', 'GB', 'WR', False,
            None, None, None, None, yards, 0, 0, targets, receptions, 0,
            None, None, None, None, None, None, None, None, 'nflverse_player_stats',
            psycopg2.extras.Json(payload or {'snapshot': 'new'}))


def seed_proof(conn, **override):
    _upsert_players(conn, [incoming()])
    proof = dict(receiving_yards=0, targets=0, receptions=0, verified_game_participation=True,
                 offense_snaps=0, special_teams_snaps=9, evidence='complete_reconciled_pbp', **override)
    with conn.cursor() as cur:
        cur.execute("UPDATE pg_temp.receiving_import_test SET offense_snaps=0, raw_json=%s",
                    (psycopg2.extras.Json({'receiving_result_repair': proof, 'old_source_field': True}),))
    return proof


def saved(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT raw_json, ctid::text FROM pg_temp.receiving_import_test')
        return cur.fetchone()


def test_refresh_preserves_matching_proof_and_retry_does_not_rewrite(database):
    proof = seed_proof(database)
    _upsert_players(database, [incoming()])
    payload, version = saved(database)
    assert payload == {'snapshot': 'new', 'receiving_result_repair': proof}
    _upsert_players(database, [incoming()])
    assert saved(database) == (payload, version)


@pytest.mark.parametrize('changes', [dict(yards=5), dict(targets=1), dict(receptions=1), dict(yards=None)])
def test_corrected_or_missing_stats_invalidate_old_proof(database, changes):
    seed_proof(database)
    _upsert_players(database, [incoming(**changes)])
    assert 'receiving_result_repair' not in saved(database)[0]


def test_refresh_discards_incomplete_proof(database):
    _upsert_players(database, [incoming(payload={'snapshot': 'new', 'receiving_result_repair':
                                               {'verified_game_participation': True}})])
    # An older malformed record is not sufficient proof for a later refresh.
    _upsert_players(database, [incoming()])
    assert 'receiving_result_repair' not in saved(database)[0]
