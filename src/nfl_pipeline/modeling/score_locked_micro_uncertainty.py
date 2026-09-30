"""Score uncertainty-only against exact micro ledger IDs, without changing bets."""
import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone

import psycopg2
import psycopg2.extras

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling.score_accuracy_components import load_artifact, score_records


def score_locks(locks, artifact, now):
    rows = []; excluded = Counter()
    component = artifact['models'].get('receiving_conditional')
    if not component or 'probability' not in component.get('enabled_outputs', []):
        return [], {'receiving_uncertainty_not_accepted_for_prospective_test': len(locks)}
    isolated = dict(artifact, models={'receiving_conditional': component})
    cohort = artifact['run_id'] + '-receiving_conditional-locked_micro'
    for record in locks:
        p = record.get('forecast_payload') or {}
        if p.get('stat') != 'receiving_yards':
            excluded['not_receiving_yards'] += 1; continue
        kickoff = safe_time(record.get('start_ts_utc'))
        lock = safe_time(record.get('locked_at_utc'))
        created = safe_time(record.get('created_at_utc'))
        if (not kickoff or not lock or not created or not created <= lock <= now < kickoff
            or date.fromisoformat(artifact['training_end']) >= kickoff.date()):
            excluded['not_prospective'] += 1; continue
        if any(str(record.get('ledger_' + key)) != str(p.get(key)) for key in ('book', 'side', 'model_version')):
            excluded['ledger_forecast_identity_mismatch'] += 1; continue
        try:
            same_offer = (float(record['ledger_line']) == float(p['line'])
                          and int(record['ledger_price']) == int(p['price']))
        except (ValueError, TypeError, KeyError):
            same_offer = False
        if not same_offer:
            excluded['ledger_offer_mismatch'] += 1; continue
        scored, issues = score_records([(int(record['prediction_id']), p)], isolated)
        excluded.update(issues)
        for row in scored.get('receiving_conditional', []):
            row.update(challenger_run=cohort, ledger_id=record['ledger_id'],
                       selection_scope='exact_locked_micro', execution_status=record.get('execution_status'))
            rows.append(row)
    return rows, dict(excluded)


def run(day):
    artifact, manifest = load_artifact()
    if not artifact:
        return {'status': 'waiting_for_uncertainty_artifact', 'rows': 0}
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT l.ledger_id,l.prediction_id,l.locked_at_utc,l.execution_status,
            l.book AS ledger_book,l.side AS ledger_side,l.line AS ledger_line,l.price AS ledger_price,
            l.model_version AS ledger_model_version,p.created_at_utc,p.forecast_payload,g.start_ts_utc
            FROM bets.nfl_bet_ledger l JOIN bets.nfl_player_prop_predictions p ON p.id=l.prediction_id
            JOIN raw.nfl_games g ON g.game_id=p.game_id
            WHERE l.source_kind='prop' AND l.tier='micro_projection' AND l.game_date_et=%s
            ORDER BY l.locked_at_utc,l.ledger_id""", (day,))
        locks = [dict(r) for r in cur.fetchall()]
    before = datetime.now(timezone.utc)
    rows, excluded = score_locks(locks, artifact, before)
    at = datetime.now(timezone.utc)
    starts = {int(r['prediction_id']): safe_time(r['start_ts_utc']) for r in locks}
    # Scoring itself can cross kickoff. Such a row is not a prospective forecast.
    finished = [r for r in rows if at < starts[r['forecast_id']]]
    excluded['crossed_kickoff_during_scoring'] = len(rows)-len(finished)
    cohort = artifact['run_id'] + '-receiving_conditional-locked_micro'
    doc = clean(dict(status='prospective_only', scored_at=at.isoformat(), training_end=artifact['training_end'],
                     source_manifest=manifest, rows=finished, exclusions=excluded, production_changed=False))
    atomic_json(MODEL_ROOT / 'challengers' / cohort / 'prospective' / str(day) / (at.strftime('%Y%m%dT%H%M%S%fZ')+'.json'), doc)
    return dict(status='prospective_only', rows=len(finished), exclusions=excluded, production_changed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--date', required=True)
    print(json.dumps(run(date.fromisoformat(parser.parse_args().date)), indent=2))
