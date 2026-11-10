"""Audit all registered receiving result dates without changing forecasts or results."""
from collections import Counter
from datetime import datetime, timezone
import json

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.grade_predictions import _grade_side
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling.receiving_result_reconciliation import (
    ROOT, classify, eligible_error, load_reconciliation_rows, load_registration_config,
)


def summarize(records, config):
    exclusions = Counter()
    eligible = []
    for row in records:
        error = eligible_error(row, config)
        if error:
            exclusions[error] += 1
        else:
            eligible.append(row)
    states = Counter(classify(row, {}) for row in eligible)
    unresolved, mismatches = [], []
    for row in eligible:
        state = classify(row, {})
        detail = dict(forecast_id=int(row['id']), game_id=row['game_id'], player_id=row['player_id'],
                      player=row['player_name'], state=state, saved_grade=row.get('graded_result'))
        if state not in {'pending_game', 'void_nonparticipant', 'settled_verified_participation'}:
            unresolved.append(detail)
        if state == 'settled_verified_participation':
            expected = _grade_side(float(row['actual']), float(row['line']), row['side'])
            if expected is None or expected != row.get('graded_result'):
                mismatches.append(dict(detail, expected_grade=expected))
    return dict(loaded_forecast_rows=len(records), eligible_forecast_rows=len(eligible),
                exclusions=dict(exclusions), states=dict(states),
                unresolved_final_rows=len(unresolved),
                unique_unresolved_player_games=len({(r['game_id'], r['player_id']) for r in unresolved}),
                unresolved=unresolved, grading_mismatches=mismatches)


def check_excluded_captures(records, documents, config, excluded):
    from nfl_pipeline.modeling.receiving_trial_checkpoint import capture_index
    scoped = [dict(d, rows=[r for r in d.get('rows', []) if int(r['forecast_id']) in excluded
                           and r.get('variant') == config['variant']]) for d in documents]
    captured, errors = capture_index(records, scoped, config)
    found = excluded & set(captured)
    loaded = {int(r['id']) for r in records}
    return dict(status='verified_exclusion' if documents and excluded and loaded==excluded and not found else 'needs_review',
                forecast_ids=sorted(excluded), count=len(excluded),
                original_capture_files_examined=len(documents), unexpected_valid_capture_ids=sorted(found),
                rejected_capture_reasons=errors, missing_original_lock_ids=sorted(excluded-loaded),
                result_repair_does_not_create_forecast_captures=True)


def historical_exclusion(config):
    path = ROOT/'reports'/'nfl_receiving_trial_checkpoint_2026-10-18.json'
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    excluded = set(checkpoint['capture']['missing_current_forecast_ids'])
    # Inspect the original benchmark captures themselves, not just a later report.
    paths = sorted((MODEL_ROOT/'model_benchmark'/config['run_id']/'prospective'/'2026-10-18').glob('*.json'))
    documents = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    with psycopg2.connect(PG_DSN, connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            cur.execute('''SELECT p.id,p.created_at_utc,p.forecast_payload,g.start_ts_utc,
                       o.fetched_at_utc AS offer_fetched_at
                FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
                LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id WHERE p.id=ANY(%s)''',
                        (sorted(excluded),))
            records = [dict(r) for r in cur.fetchall()]
    return check_excluded_captures(records, documents, config, excluded)


def run():
    started = datetime.now(timezone.utc)
    config = load_registration_config()
    with psycopg2.connect(PG_DSN, connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            cur.execute('''SELECT DISTINCT game_date_et FROM bets.nfl_player_prop_predictions
                WHERE stat=%s AND book=%s AND model_version=%s AND created_at_utc >= %s::timestamptz
                ORDER BY game_date_et''', (config['stat'], config['book'], config['production_release'],
                                           config['registered_at']))
            days = [r[0] for r in cur.fetchall()]
    summaries = [dict(day=str(day), **summarize(load_reconciliation_rows(day, config), config)) for day in days]
    unresolved = sum(r['unresolved_final_rows'] for r in summaries)
    mismatches = sum(len(r['grading_mismatches']) for r in summaries)
    exclusion = historical_exclusion(config)
    doc = dict(started_at=started.isoformat(), built_at=datetime.now(timezone.utc).isoformat(),
               scope='registered legacy receiving locks, all dates; result integrity only',
               production_release=config['production_release'], registration_at=config['registered_at'],
               status='clear' if not unresolved and not mismatches and exclusion['status']=='verified_exclusion' else 'needs_attention',
               unresolved_final_rows=unresolved, grading_mismatches=mismatches,
               days=summaries, historical_missing_captures=exclusion,
               betting_authorized=False, forecasts_changed=False,
               note='Forecast revisions are not independent decisions. A clear result audit does not establish an edge, '
                    'replace exact closes/live price checks, or admit missing original captures.')
    stem = started.strftime('%Y%m%dT%H%M%S%fZ')
    atomic_json(ROOT/'data'/'nfl_result_audit'/f'{stem}.json', doc)
    atomic_json(ROOT/'reports'/'nfl_receiving_results_audit_latest.json', doc)
    text = ['# NFL receiving result audit', '', f"Built: {doc['built_at']}", f"Status: {doc['status']}",
            f'Unresolved final forecast rows: {unresolved}. Grading mismatches: {mismatches}.', '',
            '| Game date | Eligible forecasts | Unresolved final | Grade mismatches |', '|---|---:|---:|---:|']
    text += [f"| {r['day']} | {r['eligible_forecast_rows']} | {r['unresolved_final_rows']} | {len(r['grading_mismatches'])} |" for r in summaries]
    text += ['', f"Original missing captures excluded: {exclusion['count']}; status: {exclusion['status']}.", '', doc['note']]
    (ROOT/'reports'/'nfl_receiving_results_audit_latest.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return doc


if __name__ == '__main__':
    doc = run()
    print(json.dumps({k:v for k,v in doc.items() if k!='days'}))
