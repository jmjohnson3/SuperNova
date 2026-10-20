"""Read-only operational audit of the current frozen receiving paper cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DAY = '2026-10-18'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def receipt_state(receipt, day, now, registration_sha):
    ended = safe_time(receipt.get('ended_at'))
    identity = ((receipt.get('freeze_verification') or {}).get('experiments') or {}).get('receiving') or {}
    if (not registration_sha or receipt.get('date') != str(day) or not ended or ended > now
            or identity.get('sha256') != registration_sha):
        return 'unknown'
    if (now-ended).total_seconds() > 1200:
        return 'stale'
    if any(s.get('exit_code') != 0 for s in receipt.get('steps', [])):
        return 'failed'
    return 'current' if receipt.get('capture_result') is not None else 'unknown'


def build(day, games, quotes, forecasts, documents, closes, receipt, legacy, freezes, now,
          *, registration_sha=None, next_game=None):
    """Summarize source evidence without selecting, pricing, or filling a missing result."""
    from nfl_pipeline.markets import normalize_team
    from nfl_pipeline.modeling.receiving_paper_trial import _settled
    rows = [r for d in documents for r in d.get('rows', []) if r.get('day') == str(day)]
    lookup = {int(r['id']): r for r in forecasts}
    old = legacy.get('capture') or {}
    missing = sorted(set(old.get('missing_current_forecast_ids', [])))
    old_missing = set(old.get('missing_historical_forecast_ids', [])) | set(missing)
    old_captured = set(old.get('captured_forecast_ids', []))
    contamination = sorted(old_missing & {r['forecast_id'] for r in rows})
    receipt_status = receipt_state(receipt, day, now, registration_sha)
    game_rows = []
    for game in games:
        game_id = game['game_id']; start = safe_time(game['start_ts_utc'])
        minutes = (start-now).total_seconds()/60
        due = minutes <= 120
        captured = [r for r in rows if r['game_id'] == game_id]
        selected = [r for r in captured if r['selected']]
        current = [r for r in forecasts if r['game_id'] == game_id and r.get('is_current')]
        same_game = [q for q in quotes if safe_time(q['commence_time_utc']) == start
                     and normalize_team(q['home_team']) == normalize_team(game['home_team_abbr'])
                     and normalize_team(q['away_team']) == normalize_team(game['away_team_abbr'])]
        fresh = [q for q in same_game if q.get('provider') and safe_time(q.get('fetched_at_utc'))
                 and 0 <= (now-safe_time(q['fetched_at_utc'])).total_seconds() <= 1200
                 and all(q.get(k) is not None and abs(float(q[k])) >= 100 for k in ('over_price','under_price'))]
        verified = [r for r in current if registration_sha and r.get('generation_sha') == registration_sha
                    and r.get('captured_provider') and safe_time(r.get('quote_at'))
                    and safe_time(r.get('created_at_utc'))
                    and safe_time(r['quote_at']) <= safe_time(r['created_at_utc']) < start
                    and (safe_time(r['created_at_utc'])-safe_time(r['quote_at'])).total_seconds() <= 1200]
        absent_provider = sum(not r.get('captured_provider') for r in current)
        latest_forecast = max((safe_time(r['created_at_utc']) for r in current), default=None)
        receipt_applies = (receipt_status == 'current' and game_id in receipt.get('game_ids', [])
                           and (latest_forecast is None or safe_time(receipt['ended_at']) >= latest_forecast))
        valid_closes = [r for r in selected if closes.get(r['decision_id'], {}).get('valid') is True]
        states = Counter(_settled(r, lookup.get(r['forecast_id']))[0] for r in captured)
        unsettled = sum(n for state,n in states.items() if state not in {'win','loss','push','verified_nonparticipant_void'})
        stages = dict(
            quotes=dict(status='not_applicable' if minutes <= 0 else 'ready' if fresh else 'needs_refresh' if due else 'not_due',
                        fresh_paired_rows=len(fresh), observed_rows=len(same_game)),
            forecasts=dict(status='verified_generation' if verified or captured else 'missing_generation' if due else 'not_due',
                           current_rows=len(current), verified_generation_rows=len(verified), missing_provider_rows=absent_provider),
            capture=dict(status='captured' if captured else 'examined_no_eligible_captures' if receipt_applies
                         else 'unknown' if due else 'not_due', rows=len(captured), selected=len(selected)),
            exact_close=dict(status='not_due' if minutes > 0 else 'not_applicable' if not selected
                             else 'complete' if len(valid_closes)==len(selected) else 'missing_exact_close',
                             required=len(selected), valid=len(valid_closes),
                             unknown_reasons=dict(Counter(closes.get(r['decision_id'],{}).get('reason','missing_exact_pregame_close')
                                 for r in selected if minutes <= 0 and not closes.get(r['decision_id'],{}).get('valid')))),
            grading=dict(status='not_due' if game.get('status') != 'final' else 'not_applicable' if not captured
                         else 'pending_result_or_participation' if unsettled else 'complete', accounting=dict(states)))
        blockers = [name+':'+stage['status'] for name,stage in stages.items()
                    if stage['status'] in {'needs_refresh','missing_generation','unknown','missing_exact_close','pending_result_or_participation'}]
        game_rows.append(dict(game_id=game_id,kickoff=start.isoformat(),minutes_to_start=round(minutes,1),
                              game_status=game['status'],stages=stages,blockers=blockers))
    blockers = [g['game_id']+':'+b for g in game_rows for b in g['blockers']]
    if freezes.get('status') != 'verified': blockers.insert(0,'active_freeze_verification_failed')
    if contamination or set(missing)&old_captured: blockers.append('historical_missing_capture_admitted')
    upcoming = [safe_time(g['start_ts_utc']) for g in games if safe_time(g['start_ts_utc']) > now]
    earliest = min(upcoming) if upcoming else safe_time((next_game or {}).get('start_ts_utc'))
    complete = bool(game_rows) and all(g['game_status']=='final' and g['stages']['capture']['rows'] > 0
                    and g['stages']['grading']['status']=='complete'
                    and g['stages']['exact_close']['status'] in {'complete','not_applicable'} for g in game_rows)
    action = ('restore_or_explicitly_register_correct_source_cohort' if freezes.get('status') != 'verified'
              else 'resolve_reported_evidence_gaps' if blockers else 'none' if complete
              else 'collect_fresh_quotes_and_forecasts' if upcoming and min(upcoming)-now <= timedelta(minutes=120)
              else 'wait_for_pregame_window' if upcoming or not games else 'await_verified_final_results')
    return clean(dict(contract='nfl-paper-operations-v1',day=str(day),built_at=now.isoformat(),
        status='needs_attention' if blockers else 'complete' if complete else 'pending' if games else 'no_games',
        freezes=freezes,games=game_rows,blockers=blockers,next_action=action,
        earliest_kickoff=earliest.isoformat() if earliest else None,
        next_collection_due_at=(earliest-timedelta(minutes=120)).isoformat() if earliest else None,
        collection_receipt=dict(status=receipt_status,ended_at=receipt.get('ended_at'),
            exclusions=(receipt.get('capture_result') or {}).get('exclusions',{})),
        legacy_missing=dict(day=legacy.get('day'),checkpoint_built_at=legacy.get('built_at'),
            previously_current_missing_count=len(missing),previously_current_missing_ids=missing,
            historical_missing_count=len(old_missing),
            status='unknown' if not legacy else 'integrity_failure' if contamination or set(missing)&old_captured else 'verified_exclusion',
            excluded_from_paper_evidence=bool(legacy) and not contamination and not set(missing)&old_captured,
            unexpected_admitted_ids=contamination),
        source_report_success_used=False,production_changed=False,betting_approved=False,
        note='Only original immutable captures count. Future close/grade stages are not due; missing participation remains unknown.'))


def load_database(day, component_id, capture_ids):
    import psycopg2
    import psycopg2.extras
    from nfl_pipeline.db import PG_DSN
    with psycopg2.connect(PG_DSN, connect_timeout=10) as conn:
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            cur.execute('''SELECT game_id,game_date_et,start_ts_utc,status,home_team_abbr,away_team_abbr
                FROM raw.nfl_games WHERE game_date_et >= %s ORDER BY start_ts_utc''',(day,))
            schedule = [dict(r) for r in cur.fetchall()]
            games = [r for r in schedule if r['game_date_et']==day]
            next_game = next((r for r in schedule if r['game_date_et']>day),None)
            cur.execute('''SELECT provider,home_team,away_team,commence_time_utc,fetched_at_utc,over_price,under_price
                FROM odds.nfl_player_prop_lines WHERE as_of_date=%s AND bookmaker_key='fanduel'
                AND stat='receiving_yards' AND snapshot_role IN ('open','lock','live')''',(day,))
            quotes = [dict(r) for r in cur.fetchall()]
            cur.execute('''SELECT p.id,p.game_id,p.player_id,p.stat,p.is_current,p.created_at_utc,g.status,
                p.forecast_payload #>> '{scoring_replay,offer,provider}' AS captured_provider,
                p.forecast_payload #>> '{scoring_replay,offer,fetched_at_utc}' AS quote_at,
                p.forecast_payload->'stat_model_challengers'->%s->'experiment_generation'->>'registration_sha256' AS generation_sha,
                r.result AS graded_result,r.actual_stat::float AS actual
                FROM bets.nfl_player_prop_predictions p JOIN raw.nfl_games g USING(game_id)
                LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
                WHERE (p.game_date_et=%s AND p.is_current AND p.stat='receiving_yards' AND p.book='fanduel')
                   OR p.id=ANY(%s)''',(component_id,day,list(capture_ids)))
            return games,quotes,[dict(r) for r in cur.fetchall()],next_game


def report(day):
    from nfl_pipeline import experiment_registry
    from nfl_pipeline.modeling import receiving_paper_trial as trial
    from nfl_pipeline.paper_collection import guard_freezes
    now = datetime.now(timezone.utc)
    try:
        freezes = dict(status='verified',**guard_freezes())
    except (ValueError,OSError,TypeError,AttributeError) as exc:
        freezes = dict(status='blocked',reason=str(exc))
    try:
        manifest = experiment_registry.load_experiment(trial.NAME)
        documents = trial.captures(manifest) if manifest else []
    except (ValueError,OSError) as exc:
        freezes = dict(status='blocked',reason=str(exc))
        manifest, documents = None, []
    rows = [r for d in documents for r in d['rows'] if r['day']==str(day)]
    component = (manifest or {}).get('config',{}).get('component_id','')
    games,quotes,forecasts,next_game = load_database(day,component,[r['forecast_id'] for r in rows])
    selected = [r for r in rows if r['selected'] and safe_time(r['start_ts_utc'])<=now]
    quote_lookup = trial.load_close_quotes(selected)
    closes = {r['decision_id']:trial.exact_close(r,quote_lookup.get(r['decision_id'],[]),now) for r in selected}
    receipt = read_json(ROOT/'reports'/f'nfl_paper_collection_{day}.json')
    legacy = read_json(ROOT/'reports'/f'nfl_receiving_trial_checkpoint_{LEGACY_DAY}.json')
    doc = build(day,games,quotes,forecasts,documents,closes,receipt,legacy,freezes,now,
                registration_sha=(manifest or {}).get('sha256'),next_game=next_game)
    for suffix in (str(day),'latest'):
        atomic_json(ROOT/'reports'/f'nfl_paper_operations_{suffix}.json',doc)
        text = [f'# NFL paper operations: {day}','',f"Status: {doc['status']}",
                f"Next action: {doc['next_action']}",f"Earliest kickoff: {doc['earliest_kickoff']}",'',doc['note']]
        for game in doc['games']:
            text += ['',game['game_id'],*['- '+name+': '+stage['status'] for name,stage in game['stages'].items()]]
        text += ['',f"Historical missing current captures excluded: {doc['legacy_missing']['previously_current_missing_count']}",
                 'Cash authorization: none.']
        (ROOT/'reports'/f'nfl_paper_operations_{suffix}.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',type=date.fromisoformat,required=True)
    result = report(parser.parse_args().date)
    print(json.dumps(result,default=str))
    if result['status']=='needs_attention': raise SystemExit(1)
