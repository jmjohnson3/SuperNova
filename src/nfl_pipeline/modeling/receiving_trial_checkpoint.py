"""Prospective receiving trial operations and forecast-only deployment review."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import math

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import benchmark_offers as benchmark
from nfl_pipeline.modeling import receiving_research_trial as trial
from nfl_pipeline.modeling.live_scoring_replay import ROOT, load_rows, validate_lock
from nfl_pipeline.modeling.prospective_market import locked_market


def eligible_error(rec, config):
    p = rec.get('forecast_payload') or {}
    q = (p.get('scoring_replay') or {}).get('offer') or {}
    if rec.get('stat') != config['stat'] or p.get('book') != config['book']:
        return 'outside_market'
    if p.get('model_version') != config['production_release']:
        return 'different_production_release'
    error = validate_lock(rec)
    if error:
        return error
    lock = safe_time(rec['created_at_utc'])
    if lock < safe_time(config['registered_at']):
        return 'before_registration'
    if q.get('market_key') not in config['market_keys'] or q.get('is_alt_line') is True:
        return 'not_common_line'
    evidence = locked_market(rec)
    if evidence['market_probability'] is None:
        return evidence['market_evidence']
    if (lock-safe_time(q['fetched_at_utc'])).total_seconds() > config['rules']['max_quote_age_minutes']*60:
        return 'stale_original_quote'
    return None


def capture_index(records, documents, config):
    lookup = {int(r['id']): r for r in records}
    captured = {}; errors = Counter()
    for doc in sorted(documents, key=lambda d: d.get('scored_at', '')):
        if doc.get('run_id') != config['run_id']:
            continue
        if (doc.get('source_manifest') or {}).get('sha256') != config['artifact_sha256']:
            errors['pinned_artifact_mismatch'] += 1
            continue
        for row in doc.get('rows', []):
            if row.get('variant') != config['variant']:
                continue
            rec = lookup.get(int(row['forecast_id']))
            if rec is None or not benchmark.original_offer_matches(rec, row, doc):
                errors['original_offer_or_timing_mismatch'] += 1
                continue
            q = (rec['forecast_payload'].get('scoring_replay') or {}).get('offer') or {}
            fetched = safe_time(q.get('fetched_at_utc'))
            if not fetched or (safe_time(doc['scored_at'])-fetched).total_seconds() > config['rules']['max_quote_age_minutes']*60:
                errors['quote_expired_before_challenger_capture'] += 1
                continue
            captured.setdefault(int(rec['id']), dict(row, scored_at=doc['scored_at']))
    return captured, dict(errors)


def close_summary(records, ids, closes, now):
    lookup = {int(r['id']): r for r in records}
    started = {i for i in ids if i in lookup and safe_time(lookup[i].get('start_ts_utc'))
               and safe_time(lookup[i]['start_ts_utc']) <= now}
    observations = []; reasons = {}
    for i in sorted(ids):
        rec = lookup.get(i); c = closes.get(i) or {}
        if rec is None:
            reasons[i] = 'missing_original_lock'
            continue
        p = rec['forecast_payload']; close_at = safe_time(c.get('close_fetched_at_utc'))
        lock, start = safe_time(rec['created_at_utc']), safe_time(rec['start_ts_utc'])
        valid = (c.get('valid_close_snapshot_captured') is True and c.get('clv_prob_delta') is not None
                 and math.isfinite(float(c['clv_prob_delta'])))
        reason = c.get('close_quality_reason') or 'missing_close_label'
        if valid and (not close_at or not lock < close_at < start or (start-close_at).total_seconds() > 7200
                      or any(c.get(k) != p.get(k) for k in ('book','stat','side'))
                      or c.get('locked_line') is None or c.get('close_line') is None
                      or float(c['locked_line']) != float(p['line']) or float(c['close_line']) != float(p['line'])):
            valid, reason = False, 'close_label_identity_or_timing_mismatch'
        reasons[i] = None if valid else reason
        observations.append(dict(forecast_id=i, book=p['book'], stat=p['stat'], line=p['line'], side=p['side'],
            locked_at=lock.isoformat(), close_at=close_at.isoformat() if close_at else None,
            valid_exact_capture=valid, status=reason, clv=c.get('clv_prob_delta') if valid else None))
    known = {i: closes[i] for i in started if reasons.get(i) is None}
    unknown = started-set(known)
    return dict(started=len(started), valid=len(known), unknown=len(unknown),
        valid_coverage=len(known)/len(started) if started else None, required_coverage=.9,
        target_met=len(known)/len(started) >= .9 if started else None,
        unknown_reasons=dict(Counter(reasons[i] for i in unknown)), observations=observations,
        average_clv=sum(float(c['clv_prob_delta']) for c in known.values())/len(known) if known else None)


def metrics_for(rows):
    if not rows:
        return {}
    latest = max(rows, key=lambda r: r.get('created_at','')).get('scoring_version','legacy_unknown')
    cohort = [r for r in rows if r.get('scoring_version','legacy_unknown') == latest]
    return next(iter(benchmark.offered_metrics(cohort).values()), {})


def build_checkpoint(day, records, documents, ledger, registration, selections, closes, now):
    if not registration:
        return dict(status='not_registered', day=str(day), production_changed=False, betting_approved=False)
    config = registration['config']
    captured, capture_errors = capture_index(records, documents, config)
    eligible = {}; excluded = Counter()
    for rec in records:
        error = eligible_error(rec, config)
        if error:
            excluded[error] += 1
        else:
            eligible[int(rec['id'])] = rec
    selected_ids = {int(r['forecast_id']) for r in selections}
    ids = set(eligible)
    day_ids = {i for i in ids if str(eligible[i]['forecast_payload']['game_date_et'])[:10] == str(day)}
    current_ids = {i for i in day_ids if eligible[i].get('is_current')}
    missing = current_ids-set(captured)
    day_selected = {int(r['forecast_id']) for r in selections if str(r['day']) == str(day)}
    pin_docs = [d for d in documents if d.get('run_id') == config['run_id']
                and (d.get('source_manifest') or {}).get('sha256') == config['artifact_sha256']]
    validated, validation_errors, pending = benchmark.validate_documents(records, ledger, pin_docs)
    rows = [r for r in validated if r['variant'] == config['variant'] and r['forecast_id'] in ids
            and r['forecast_id'] in captured]
    selected_rows = [r for r in rows if r['forecast_id'] in selected_ids]
    all_metrics = metrics_for(rows); selected_metrics = metrics_for(selected_rows)
    review = benchmark.deployment_review(all_metrics)
    # Capture gaps are operational blockers, never silently dropped success rows.
    if missing:
        review['blockers'].append('missing_current_challenger_captures')
        review['status'] = 'collecting_evidence'
    historical_missing = {i for i in ids-set(captured) if safe_time(eligible[i]['start_ts_utc']) <= now}
    if historical_missing:
        review['blockers'].append('missing_prospective_capture_history')
        review['status'] = 'collecting_evidence'
    fresh_current = sum((now-safe_time((eligible[i]['forecast_payload']['scoring_replay']['offer'])['fetched_at_utc'])).total_seconds()
                        <= config['rules']['max_quote_age_minutes']*60 for i in current_ids)
    settled_ids = {r['forecast_id'] for r in rows}
    day_settled = day_ids & settled_ids
    # Pushes and missing participation remain explicit; neither becomes a loss.
    states = Counter()
    for i in day_ids:
        r = eligible[i]; p = r['forecast_payload']
        state = ('settled_binary' if i in day_settled else 'pending_game' if r.get('status') != 'final'
                 else 'void_nonparticipant' if r.get('graded_result')=='void_nonparticipant'
                 else 'result_or_participation_unresolved' if r.get('actual') is None
                 else 'push' if float(r['actual']) == float(p['line']) else 'missing_validated_capture')
        states[state] += 1
    status = ('capture_incomplete' if missing else 'no_eligible_offers' if not day_ids
              else 'awaiting_settlement' if states['pending_game'] else 'evaluated')
    return dict(built_at=now.isoformat(), day=str(day), status=status,
        run_id=config['run_id'], variant=config['variant'], production_release=config['production_release'],
        registration_sha256=registration['sha256'],
        capture=dict(day_eligible_locks=len(day_ids), current_eligible_locks=len(current_ids),
            eligible_forecast_ids=sorted(day_ids),captured_forecast_ids=sorted(day_ids & set(captured)),
            current_quotes_fresh_now=fresh_current, paired_challenger_locks=len(day_ids & set(captured)),
            missing_current_forecast_ids=sorted(missing), selected_research=len(day_selected),
            missing_historical_forecast_ids=sorted(historical_missing),
            capture_errors=capture_errors, eligibility_exclusions=dict(excluded)),
        settlement=dict(states), validation_exclusions=validation_errors, pending_variant_rows=pending,
        day_metrics=dict(all_eligible=metrics_for([r for r in rows if r['forecast_id'] in day_ids]),
                         fixed_research=metrics_for([r for r in selected_rows if r['forecast_id'] in day_selected])),
        cumulative_metrics=dict(all_eligible=all_metrics, fixed_research=selected_metrics),
        scoring_cohorts=benchmark.offered_metrics(rows),
        close=dict(all_eligible=close_summary(records, day_ids, closes, now),
                   fixed_research=close_summary(records, day_selected, closes, now)),
        forecast_deployment_review=review,
        forecast_next_action='prepare_forecast_only_release' if not review['blockers'] else 'keep_frozen_collect_evidence',
        deployment_requires_bankroll_approval=False, betting_approved=False, production_changed=False,
        limitations=['Research selections are fixed; this report never reselects after results.',
            'Freshness at original lock/capture is separate from whether a price is executable now.',
            'Forecast review does not require ROI or CLV gates. Cash approval remains separate.',
            'A positive checkpoint prepares a controlled release review; it never overwrites model artifacts.'])


def render(doc):
    lines = ['# NFL Receiving Trial Checkpoint', '', f"Date: {doc['day']}; status: {doc['status']}"]
    if 'capture' not in doc:
        return '\n'.join(lines)+'\n'
    capture = doc['capture']
    lines += [f"Pinned variant: {doc['variant']} ({doc['run_id']})", '',
        f"Eligible original locks: {capture['day_eligible_locks']}; paired challenger captures: {capture['paired_challenger_locks']}.",
        f"Current eligible locks: {capture['current_eligible_locks']}; quotes still fresh now: {capture['current_quotes_fresh_now']}.",
        f"Fixed research selections: {capture['selected_research']}; missing current captures: {capture['missing_current_forecast_ids']}.",
        f"Settlement: {doc['settlement']}", '',
        '| Period / scope | Rows | Player-games | Weeks | Production Brier / calibration | Challenger Brier / calibration | Market Brier | Production / challenger 80% coverage |',
        '|---|---:|---:|---:|---|---|---|---|']
    def number(value):
        return 'unknown' if value is None else f'{value:.4f}'
    for period in ('day_metrics', 'cumulative_metrics'):
        for label, scopes in doc[period].items():
            m = scopes.get('real_offers') or {}
            if not m.get('rows'):
                lines.append(f'| {period} / {label} | 0 | - | - | pending | pending | pending | pending |')
                continue
            prod, new = m['production'], m['final']
            market = (m.get('matched_market') or {}).get('market') or {}
            lines.append(f"| {period} / {label} | {m['rows']} | {m['unique_player_games']} | {m['weeks']} | "
                f"{number(prod['brier'])} / {number(prod['calibration_error'])} | {number(new['brier'])} / {number(new['calibration_error'])} | "
                f"{number(market.get('brier'))} | {number(m['production_curve'].get('coverage_80'))} / {number(m['live_curve'].get('coverage_80'))} |")
    review = doc['forecast_deployment_review']
    lines += ['', '## Exact Closes', '', '| Scope | Started locks | Valid | Unknown | Coverage | Meets 90% |',
              '|---|---:|---:|---:|---|---|']
    for scope, m in doc['close'].items():
        coverage = f"{m['valid_coverage']:.1%}" if m['valid_coverage'] is not None else 'pending kickoff'
        lines.append(f"| {scope} | {m['started']} | {m['valid']} | {m['unknown']} | {coverage} | {m['target_met']} |")
    lines += ['', 'Exact locked/close lines, timestamps and unknown reasons are retained in the companion JSON.',
              '', '## Forecast Deployment',
              f"Status: {review['status']}; next action: {doc['forecast_next_action']}",
              *['- '+b for b in review['blockers']], '',
              'Forecast deployment and betting permission are separate. Production was not changed.', *doc['limitations']]
    return '\n'.join(lines)+'\n'


def report(day):
    registration = trial.load_registration()
    documents = [json.loads(p.read_text()) for p in benchmark.STORE.glob('*/prospective/*/*.json')]
    records = load_rows(stat='receiving_yards'); ledger = benchmark.load_ledger()
    selections = [r for d in trial.captures(registration) for r in d['selected']] if registration else []
    closes = trial.load_closes([int(r['id']) for r in records])
    doc = build_checkpoint(day, records, documents, ledger, registration, selections, closes, datetime.now(timezone.utc))
    root = ROOT/'reports'
    for suffix in (str(day), 'latest'):
        atomic_json(root/f'nfl_receiving_trial_checkpoint_{suffix}.json', doc)
        (root/f'nfl_receiving_trial_checkpoint_{suffix}.md').write_text(render(doc), encoding='utf-8')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    doc = report(parser.parse_args().date)
    print(json.dumps({k: doc[k] for k in ('day', 'status', 'capture', 'settlement', 'close', 'forecast_next_action') if k in doc}, default=str))
    if doc['status'] in ('capture_incomplete', 'not_registered'):
        raise SystemExit(1)
