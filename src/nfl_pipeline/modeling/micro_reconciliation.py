"""Account for every micro ledger entry without reconstructing missing old forecasts."""
from collections import Counter
from datetime import datetime, timezone
import json
import math

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json, FEATURE_CONTRACT
from nfl_pipeline.modeling.live_scoring_replay import ROOT, validate_lock


def load_entries():
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT l.ledger_id,l.prediction_id,l.locked_at_utc,l.game_date_et,
            l.result AS ledger_result,l.profit,l.execution_status,l.book AS ledger_book,
            l.side AS ledger_side,l.line AS ledger_line,l.price AS ledger_price,
            l.model_version AS ledger_model_version,
            p.id,p.integrity_version,p.game_id,p.player_id,p.player_name,p.stat,
            p.created_at_utc,p.forecast_payload,g.status,g.start_ts_utc,
            o.fetched_at_utc AS offer_fetched_at,r.result AS graded_result,r.actual_stat
            FROM bets.nfl_bet_ledger l
            LEFT JOIN bets.nfl_player_prop_predictions p ON p.id=l.prediction_id
            LEFT JOIN raw.nfl_games g ON g.game_id=p.game_id
            LEFT JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
            LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
            WHERE l.source_kind='prop' AND l.tier='micro_projection'
            ORDER BY l.ledger_id""")
        return [dict(r) for r in cur.fetchall()]


def classify(row, now):
    result=str(row.get('ledger_result') or '')
    if result.startswith('void') or result=='push':
        return 'void_or_push', 'recorded_'+result
    if row.get('id') is None:
        return 'missing_evaluation_inputs', 'original_prediction_missing'
    if row.get('status') != 'final':
        start=safe_time(row.get('start_ts_utc'))
        if not start:
            return 'missing_result', 'game_identity_or_kickoff_missing'
        return 'pending', 'not_started' if now<start else 'awaiting_final_result'
    if row.get('graded_result')=='void_nonparticipant':
        return 'void_or_push', 'graded_void_nonparticipant'
    if row.get('actual_stat') is None or row.get('graded_result') not in ('win','loss','push','void'):
        return 'missing_result', 'final_result_or_participation_unresolved'
    if row.get('graded_result') in ('void','push'):
        return 'void_or_push', 'graded_'+row['graded_result']
    if row.get('integrity_version') != FEATURE_CONTRACT or not row.get('forecast_payload'):
        return 'missing_evaluation_inputs', 'legacy_missing_immutable_lock_payload'
    p=row['forecast_payload']
    error=validate_lock(row)
    if error:
        return 'missing_evaluation_inputs',error
    when=safe_time(row.get('locked_at_utc'))
    if not when or not safe_time(row['created_at_utc'])<=when<safe_time(row['start_ts_utc']):
        return 'missing_evaluation_inputs','ledger_lock_timing_invalid'
    if any(str(row.get('ledger_'+k))!=str(p.get(k)) for k in ('book','side','model_version')):
        return 'missing_evaluation_inputs','ledger_offer_identity_mismatch'
    try:
        if any(float(row['ledger_'+k])!=float(p[k]) for k in ('line','price')):
            return 'missing_evaluation_inputs','ledger_offer_identity_mismatch'
        probability=float(p['probability'])
        if not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('invalid probability')
        actual=float(row['actual_stat']); line=float(p['line'])
        if not math.isfinite(actual) or not math.isfinite(line) or p['side'] not in ('over','under'):
            raise ValueError('invalid outcome')
        outcome='push' if actual==line else 'win' if ((actual>line)==(p['side']=='over')) else 'loss'
        if outcome!=row['graded_result'] or result!=row['graded_result']:
            return 'missing_result','grading_or_ledger_result_mismatch'
    except (KeyError,TypeError,ValueError):
        return 'missing_evaluation_inputs','invalid_locked_probability_or_offer'
    return 'evaluable','stored_final_probability_available'


def reconcile(entries, scored_ids=None, now=None):
    now=now or datetime.now(timezone.utc); rows=[]; scored=set(scored_ids or ())
    for row in entries:
        category,reason=classify(row,now)
        rows.append(dict(ledger_id=row['ledger_id'],prediction_id=row['prediction_id'],
            day=str(row['game_date_et']),player=row.get('player_name'),stat=row.get('stat'),
            category=category,reason=reason,ledger_result=row.get('ledger_result'),
            graded_result=row.get('graded_result'),execution_status=row.get('execution_status'),
            in_probability_report=row['prediction_id'] in scored if scored_ids is not None else None,
            replay_inputs_present=bool((row.get('forecast_payload') or {}).get('scoring_replay'))))
    return dict(built_at=now.isoformat(),ledger_rows=len(entries),accounted_rows=len(rows),
        categories=dict(Counter(r['category'] for r in rows)),reasons=dict(Counter(r['reason'] for r in rows)),
        historical_record=dict(Counter(r.get('ledger_result') or 'pending' for r in rows)),
        evaluable_but_unscored=([r['prediction_id'] for r in rows if r['category']=='evaluable' and not r['in_probability_report']]
                               if scored_ids is not None else None),
        rows=rows,legacy_rows_added_to_training=False,locks_changed=False,
        note='Historical ledger results are retained, not independently reconstructed. Simulated is not a confirmed cash wager.')


def write_report(doc):
    atomic_json(ROOT/'reports/nfl_micro_reconciliation_latest.json',doc)
    lines=['# NFL Micro Ledger Reconciliation','',json.dumps(doc['categories']), '',doc['note'],'',
        '| Ledger | Prediction | Player | Category | Reason | Ledger result |','|---|---|---|---|---|---|']
    for r in doc['rows']:
        lines.append(f"| {r['ledger_id']} | {r['prediction_id']} | {r['player'] or 'missing original'} | {r['category']} | {r['reason']} | {r['ledger_result']} |")
    (ROOT/'reports/nfl_micro_reconciliation_latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    doc=reconcile(load_entries());write_report(doc);print(json.dumps(doc['categories']))
