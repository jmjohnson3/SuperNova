"""Run each NFL kickoff wave once near T-90; retry known failures, never uncertain posts."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import uuid

import psycopg2
import psycopg2.extras

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.subprocess_utils import run_subprocess

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT/'reports'/'nfl_pregame_state.json'
CONTRACT = 'nfl-game-aware-pregame-v1'
WINDOW_MINUTES = 90
MIN_REMAINING_MINUTES = 20
MAX_ATTEMPTS = 3


def key(game):
    return str(game['game_id'])+'@'+safe_time(game['start_ts_utc']).isoformat()


def load_state():
    if not STATE.exists():
        return dict(contract=CONTRACT, games={})
    result = json.loads(STATE.read_text(encoding='utf-8'))
    if result.get('contract') != CONTRACT or not isinstance(result.get('games'), dict):
        raise ValueError('Invalid pregame state; refusing to reset completion history')
    return result


def load_games(now):
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT game_id,game_date_et,home_team_abbr,away_team_abbr,start_ts_utc,status
            FROM raw.nfl_games WHERE start_ts_utc BETWEEN %s AND %s
              AND lower(coalesce(status,'')) NOT IN ('final','canceled','cancelled','postponed')
            ORDER BY start_ts_utc,game_id""", (now-timedelta(hours=2),now+timedelta(hours=48)))
        return [dict(r) for r in cur.fetchall()]


@contextmanager
def runner_lock():
    # Session lock is released by PostgreSQL even after an interrupted Python run.
    conn = psycopg2.connect(PG_DSN)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(hashtext('nfl_game_aware_pregame'))")
            acquired = bool(cur.fetchone()[0])
        yield acquired
    finally:
        conn.close()


def plan(games, state, now):
    groups = defaultdict(list); observations = []
    for game in games:
        start = safe_time(game['start_ts_utc']); remaining = (start-now).total_seconds()/60
        previous = state['games'].get(key(game), {})
        status = previous.get('status')
        if status == 'completed':
            action = 'already_completed'
        elif status in ('review_required', 'running'):
            action = 'review_required'
        elif remaining > WINDOW_MINUTES:
            action = 'waiting_for_window'
        elif remaining <= MIN_REMAINING_MINUTES:
            action = 'missed_pregame_window'
        elif previous.get('attempts', 0) >= MAX_ATTEMPTS:
            action = 'retry_limit_reached'
        elif previous.get('attempted_at') and (now-safe_time(previous['attempted_at'])).total_seconds() < 600:
            action = 'retry_backoff'
        else:
            action = 'due'
            groups[str(game['game_date_et'])].append(game)
        observations.append(dict(game_id=game['game_id'], kickoff=start.isoformat(),
            target_at=(start-timedelta(minutes=WINDOW_MINUTES)).isoformat(),
            minutes_to_start=round(remaining,1), action=action, attempts=previous.get('attempts',0)))
    return list(groups.values()), observations


def run_wave(games, run_id, now):
    day = str(games[0]['game_date_et'])
    args = [sys.executable,'-m','nfl_pipeline.run_daily_and_notify','--pregame',
            '--date',day,'--skip-train','--run-id',run_id]
    for game in games:
        args += ['--game-id', str(game['game_id'])]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONIOENCODING'] = 'utf-8'
    # Leave five minutes before kickoff; each scoring/locking stage also checks time.
    seconds = int((min(safe_time(g['start_ts_utc']) for g in games)-now).total_seconds())-300
    rc, out, err = run_subprocess(args,cwd=str(ROOT),env=env,timeout_s=max(1,min(2400,seconds)))
    path = ROOT/'reports'/f'nfl_daily_run_{run_id}.json'
    doc = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    # Provider responses can contain credential-bearing URLs. Do not echo raw errors.
    return rc, doc


def classify_attempt(games, rc, doc):
    expected = {str(g['game_id']) for g in games}
    starts = {str(g['game_id']): safe_time(g['start_ts_utc']) for g in games}
    publications = (doc or {}).get('publications') or []
    published = {str(p.get('game_id')) for p in publications if safe_time(p.get('sent_at'))
                 and str(p.get('game_id')) in starts and safe_time(p['sent_at']) < starts[str(p['game_id'])]}
    if (rc == 0 and doc and doc.get('status') == 'ok' and doc.get('pregame') is True
            and set(doc.get('game_ids') or []) == expected and published == expected
            and all(str(p.get('game_id')) in expected for p in publications)):
        return 'completed'
    # A timeout or a failed/partial Discord attempt may already have delivered a card.
    # Never automatically rerun the whole wave and send another recommendation.
    if rc == 124 or not doc or publications or any(
            s.get('step') in ('NFL Matchup Cards','Discord or runner') for s in doc.get('steps', [])):
        return 'review_required'
    return 'failed'


def process(games, state, *, execute=run_wave, save=atomic_json, clock=lambda:datetime.now(timezone.utc)):
    now = clock(); waves, observations = plan(games,state,now)
    results = []
    for wave in waves:
        now = clock()
        # Recheck after a prior wave or mutex wait; never start work based on an old plan.
        due, _ = plan(wave,state,now)
        if not due:
            continue
        wave = [g for group in due for g in group]
        run_id = 'pregame_'+uuid.uuid4().hex
        for game in wave:
            old = state['games'].get(key(game), {})
            state['games'][key(game)] = dict(status='running',run_id=run_id,attempted_at=now.isoformat(),
                attempts=old.get('attempts',0)+1,game_id=str(game['game_id']),kickoff=safe_time(game['start_ts_utc']).isoformat())
        save(STATE,state)
        try:
            rc, doc = execute(wave,run_id,now)
        except Exception:
            # Delivery may have happened before the child/report failed. Preserve
            # that ambiguity rather than automatically sending another card.
            rc, doc = 1, None
        status = classify_attempt(wave,rc,doc)
        for game in wave:
            state['games'][key(game)].update(status=status,returncode=rc,finished_at=clock().isoformat())
        save(STATE,state)
        results.append(dict(run_id=run_id,game_ids=[g['game_id'] for g in wave],status=status,returncode=rc))
    _, final = plan(games,state,clock())
    failed = any(r['status'] != 'completed' for r in results) or any(r['action'] in (
        'review_required','missed_pregame_window','retry_limit_reached') for r in final)
    return dict(contract=CONTRACT,built_at=clock().isoformat(),status='failed' if failed else 'completed' if results else 'idle',
        target_minutes=WINDOW_MINUTES,min_remaining_minutes=MIN_REMAINING_MINUTES,
        observations=final,runs=results,pipeline_invoked=bool(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Read schedule/state only; no odds, forecasts or Discord')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.dry_run:
        groups, observations = plan(load_games(now),load_state(),now)
        print(json.dumps(dict(status='dry_run',due_games=sum(map(len,groups)),observations=observations),indent=2))
        return
    with runner_lock() as acquired:
        if not acquired:
            print(json.dumps(dict(status='busy',reason='another_pregame_runner_owns_lock')))
            raise SystemExit(75)
        result = process(load_games(now),load_state())
        atomic_json(ROOT/'reports'/'nfl_pregame_latest.json',result)
        if result['runs'] or result['status']=='failed':
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            atomic_json(ROOT/'reports'/f'nfl_pregame_{stamp}.json',result)
        print(json.dumps(result,indent=2))
        if result['status']=='failed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
