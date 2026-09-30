"""Evaluate fixed prospective selections; never deploy a model or place a wager."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
import logging
import time

import psycopg2

from nfl_pipeline import cash_trial_policy as policy
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import benchmark_offers as benchmark
from nfl_pipeline.modeling import receiving_research_trial as trial
from nfl_pipeline.modeling import receiving_trial_checkpoint as checkpoint
from nfl_pipeline.modeling.live_scoring_replay import ROOT, load_rows
from nfl_pipeline.modeling.scoring_versions import source_hash, SOURCE


def register():
    research = trial.load_registration()
    artifact, manifest = benchmark.load_artifact()
    if research is None:
        raise ValueError('Register the fixed research strategy first')
    cfg = research['config']
    if manifest['sha256'] != cfg['artifact_sha256']:
        raise ValueError('Pinned research artifact mismatch')
    return policy.register(dict(strategy_id=cfg['trial_id'], trial_sha256=research['sha256'],
        release=cfg['production_release'], variant=cfg['variant'], artifact_sha256=manifest['sha256'],
        scoring_version=source_hash(SOURCE), source_kind='prop', stat='receiving_yards',
        book='fanduel', research_registration=research))


def evidence(records, documents, registration, selections, closes, now):
    """Only original post-registration locks, immutable captures and final outcomes."""
    cfg = registration['config']; research = cfg['research_registration']
    captured, errors = checkpoint.capture_index(records, documents, research['config'])
    lookup = {int(r['id']): r for r in records}
    selected_ids = {int(s['forecast_id']) for s in selections}
    output = []; excluded = Counter(errors); unresolved = 0; pending = 0
    for selection in selections:
        captured_at = safe_time(selection.get('captured_at'))
        if int(selection['forecast_id']) not in lookup and (
                not captured_at or captured_at >= safe_time(cfg['registered_at'])):
            unresolved += 1
            excluded['missing_selected_original_lock'] += 1
    for rec in records:
        p = rec.get('forecast_payload') or {}; fid = int(rec['id'])
        locked = safe_time(rec.get('created_at_utc'))
        if not locked or locked < safe_time(cfg['registered_at']):
            excluded['pre_policy_history'] += 1; continue
        fingerprint = (p.get('scoring_replay') or {}).get('scoring_fingerprint')
        if fingerprint != cfg['scoring_version'] or p.get('model_version') != cfg['release']:
            excluded['different_scoring_cohort'] += 1
            unresolved += int(fid in selected_ids)
            continue
        error = checkpoint.eligible_error(rec, research['config'])
        shadow = captured.get(fid)
        if error or not shadow:
            excluded[error or 'missing_immutable_challenger'] += 1
            if fid in selected_ids:
                unresolved += 1
            continue
        if rec.get('status') != 'final':
            if fid in selected_ids:
                pending += 1
            continue
        if rec.get('graded_result')=='void_nonparticipant':
            excluded['verified_void_nonparticipant'] += 1
            continue
        if rec.get('actual') is None:
            excluded['missing_final_participation_or_result'] += 1
            unresolved += int(fid in selected_ids)
            continue
        actual = float(rec['actual']); line = float(p['line']); probability = float(shadow['same_side_probability'])
        market = shadow.get('market_probability')
        # Recompute baseline from the immutable original true pair, not a later report.
        from nfl_pipeline.modeling.prospective_market import locked_market
        market = locked_market(rec)['market_probability']
        lo, hi = shadow.get('live_p10'), shadow.get('live_p90')
        if not all(v is not None and math.isfinite(float(v)) for v in (probability, market, lo, hi)):
            unresolved += int(fid in selected_ids); excluded['missing_probability_or_interval'] += 1; continue
        if not 0 < probability < 1 or not 0 < float(market) < 1 or float(lo) > float(hi):
            unresolved += int(fid in selected_ids); excluded['invalid_probability_or_interval'] += 1; continue
        result = 'push' if actual == line else 'win' if (actual > line) == (p['side'] == 'over') else 'loss'
        price = float(p['price']); payout = price/100 if price > 0 else 100/-price
        verified = checkpoint.close_summary(records, {fid}, closes, now)
        close = verified['observations'][0]
        output.append(dict(forecast_id=fid, selected=fid in selected_ids,
            decision_key=str(rec['game_id'])+'|'+str(rec['player_id']),
            player_id=str(rec['player_id']), team=p.get('team_abbr'), week_key=f"{rec['season']}:{rec['week']}",
            locked_at=locked.isoformat(), kickoff=str(rec['start_ts_utc']),
            result=result, outcome=None if result == 'push' else float(result == 'win'),
            probability=probability, market_probability=float(market), production_probability=float(p['probability']),
            profit=payout if result == 'win' else -1.0 if result == 'loss' else 0.0,
            covered_80=float(lo) <= actual <= float(hi), clv=close['clv'],
            stale_close='stale' in str(close['status']), close_reason=close['status']))
    output.sort(key=lambda r: (r['locked_at'], r['forecast_id']))
    selected = [r for r in output if r['selected']]
    return selected, output, dict(excluded), unresolved, pending


def selection_cohort_accounting(records, selections, registration):
    """Account for every fixed selection without rewriting its registration or cap."""
    cfg = registration['config']; lookup = {int(r['id']): r for r in records}
    rows = []
    for selection in selections:
        fid = int(selection['forecast_id']); rec = lookup.get(fid)
        if rec is None:
            status = 'original_lock_missing'
        else:
            p = rec.get('forecast_payload') or {}; at = safe_time(rec.get('created_at_utc'))
            if not at:
                status = 'lock_timestamp_missing'
            elif at < safe_time(cfg['registered_at']):
                status = 'pre_policy_selection_preserved'
            elif ((p.get('scoring_replay') or {}).get('scoring_fingerprint') != cfg['scoring_version']
                  or p.get('model_version') != cfg['release']):
                status = 'post_policy_wrong_cohort'
            elif rec.get('status') != 'final':
                status = 'post_policy_pending_game'
            elif rec.get('graded_result')=='void_nonparticipant':
                status = 'post_policy_void_nonparticipant'
            elif rec.get('actual') is None:
                status = 'post_policy_missing_participation_or_result'
            else:
                status = 'post_policy_settled_for_evidence_review'
        rows.append(dict(forecast_id=fid, status=status))
    return dict(registered_at=cfg['registered_at'], counts=dict(Counter(r['status'] for r in rows)), rows=rows,
        current_scoring_matches_registration=source_hash(SOURCE)==cfg['scoring_version'],
        historical_slots_replaced=False)


def review(registration, selected, all_rows, unresolved, now, *, store=policy.STORE, pending=0, completed_weeks=None):
    """Each checkpoint is consumed once. Post-registration changes start new evidence."""
    saved = []
    for target in policy.POLICY.checkpoints:
        path = store / 'reviews' / f'{target}.json'
        if path.exists():
            doc = json.loads(path.read_text())
            if doc['registration_sha256'] != registration['sha256']:
                raise ValueError('Mismatched checkpoint registration')
            saved.append(doc); continue
        # A delayed row must not let us replace a losing early selection with a later winner.
        if len(selected) < target or unresolved or pending:
            break
        prefix = selected[:target]
        last_lock = prefix[-1]['locked_at']
        result = policy.evaluate(prefix, [r for r in all_rows if r['locked_at'] <= last_lock], target)
        doc = dict(result, registration_sha256=registration['sha256'], reviewed_at=now.isoformat(),
            forecast_ids=[r['forecast_id'] for r in prefix], evidence_sha256=policy.digest(prefix),
            through_week=max((r['week_key'] for r in prefix), key=lambda k: tuple(map(int,k.split(':')))))
        atomic_json(path, doc); saved.append(doc)
    passed = next((r for r in saved if r['status'] == 'cash_trial_eligible'), None)
    # Review completed weeks only; pending games are not negative evidence.
    pause = store/'pause.json'
    week_order = lambda key: tuple(map(int, key.split(':')))
    monitor_week = max(completed_weeks, key=week_order) if completed_weeks else None
    monitor_path = store/'monitors'/str(monitor_week).replace(':','_')
    should_monitor = completed_weeks is None or (passed and monitor_week and
        week_order(monitor_week) >= week_order(passed['through_week']) and not monitor_path.exists())
    if passed and should_monitor:
        settled_rows = selected if completed_weeks is None else [r for r in selected if r['week_key'] in completed_weeks]
        settled_offers = all_rows if completed_weeks is None else [r for r in all_rows if r['week_key'] in completed_weeks]
        health = policy.evaluate(settled_rows, settled_offers, passed['metrics']['checkpoint'], unresolved=unresolved)
        # Approval significance is evaluated only at the scheduled checkpoint.
        bad = [b for b in health['blockers'] if b not in ('positive_roi_unconfirmed','selected_market_advantage_unconfirmed')]
        if health['metrics']['roi'] is None or health['metrics']['roi'] <= 0:
            bad.append('prospective_roi_no_longer_positive')
        if bad and not pause.exists():
            atomic_json(pause, dict(paused_at=now.isoformat(), blockers=bad, registration_sha256=registration['sha256']))
        if completed_weeks is not None:
            atomic_json(monitor_path, dict(health, monitored_at=now.isoformat(), week=monitor_week))
    return dict(status='paused' if pause.exists() else 'cash_trial_eligible' if passed else 'research',
        checkpoints=saved, next_checkpoint=next((n for n in policy.POLICY.checkpoints if n not in [r['metrics']['checkpoint'] for r in saved]), None),
        pause=json.loads(pause.read_text()) if pause.exists() else None)


def acquire_review_lock(cur, timeout_s=45., key='nfl_cash_policy_review'):
    """Bounded contention is retryable; it is never successful evidence review."""
    deadline = time.monotonic() + timeout_s
    waited = False
    while True:
        cur.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s))', (key,))
        if cur.fetchone()[0]:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Cash policy review is busy; no checkpoint or publication was approved')
        if not waited:
            logging.getLogger(__name__).info('Waiting for concurrent cash policy checkpoint writer')
            waited = True
        time.sleep(min(.5, remaining))


def load_review_inputs(registration, research, now):
    """Expensive immutable-history reads must not hold the policy writer lock."""
    records = load_rows(stat=registration['config']['stat'])
    documents = [json.loads(p.read_text()) for p in benchmark.STORE.glob('*/prospective/*/*.json')]
    selections = [r for d in trial.captures(research) for r in d['selected']]
    closes = trial.load_closes([int(r['id']) for r in records])
    prepared = evidence(records, documents, registration, selections, closes, now)
    return records, selections, prepared


def report(day):
    registration = policy.load_registration()
    now = datetime.now(timezone.utc)
    if not registration:
        raise ValueError('Cash readiness policy is not registered')
    research = trial.load_registration()
    if research['sha256'] != registration['config']['trial_sha256']:
        raise ValueError('Research strategy changed')
    records, selections, prepared = load_review_inputs(registration, research, now)
    selected, all_rows, excluded, unresolved, pending = prepared
    # Serialize report/checkpoint writes across daily, close and manual runners.
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='5s'")
        cur.execute("SET LOCAL statement_timeout='90s'")
        acquire_review_lock(cur)
        if policy.load_registration()['sha256'] != registration['sha256'] or trial.load_registration()['sha256'] != research['sha256']:
            raise ValueError('Registration changed during evidence loading')
        existing_path = ROOT/'reports'/f'nfl_cash_readiness_{day}.json'
        if existing_path.exists():
            existing = json.loads(existing_path.read_text())
            built = safe_time(existing.get('built_at'))
            if built and built > now and existing.get('registration_sha256') == registration['sha256']:
                return existing
        # Do not consume a checkpoint while an earlier fixed selection is unsettled.
        cur.execute("""SELECT season,week FROM raw.nfl_games WHERE season IS NOT NULL AND week IS NOT NULL
            GROUP BY season,week HAVING bool_and(COALESCE(status IN ('final','canceled'),false))""")
        completed_weeks = {f'{s}:{w}' for s,w in cur.fetchall()}
        decision = review(registration, selected, all_rows, unresolved, now, pending=pending,
                          completed_weeks=completed_weeks)
        provisional = policy.evaluate(selected, all_rows, decision['next_checkpoint'] or 200, unresolved=unresolved)
        checkpoint_path = ROOT/'reports'/f'nfl_receiving_trial_checkpoint_{day}.json'
        prior = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
        checkpoint_at = safe_time(prior.get('built_at'))
        deployment = (prior['forecast_deployment_review'] if checkpoint_at and
            0 <= (now-checkpoint_at).total_seconds() <= 1200 and prior.get('registration_sha256')==research['sha256']
            else dict(status='checkpoint_refresh_required', blockers=['current_forecast_checkpoint_missing_or_stale']))
        from nfl_pipeline.cash_execution import states
        executions = states(conn)
        settled = [r for r in executions if r['state'] in ('settled','paused') and r.get('result')]
        confirmed_profit = sum(float(r.get('profit') or 0) for r in settled)
        confirmed_stake = sum(float(r['stake']) for r in settled)
        doc = dict(contract=policy.CONTRACT, day=str(day), built_at=now.isoformat(),
            registration_sha256=registration['sha256'], strategy_id=registration['config']['strategy_id'],
            release=registration['config']['release'], scoring_version=registration['config']['scoring_version'],
            decision=decision, provisional=provisional, exclusions=excluded, pending=pending,
            selection_cohort_accounting=selection_cohort_accounting(records,selections,registration),
            forecast_deployment_review=deployment,
            executions=dict(states=dict(Counter(r['state'] for r in executions)),
                confirmed_record=dict(Counter(r['result'] for r in settled)),
                confirmed_profit=confirmed_profit,
                confirmed_roi=confirmed_profit/confirmed_stake if confirmed_stake else None),
            other_strategies={s:'not_registered_no_cash_approval' for s in
                ('passing_yards','rushing_yards','passing_tds','rushing_tds','receiving_tds','spread','total')},
            limits=registration['config']['policy'], automatically_places_bets=False, production_changed=False)
        for suffix in (str(day), 'latest'):
            atomic_json(ROOT/'reports'/f'nfl_cash_readiness_{suffix}.json', doc)
            lines = ['# NFL Cash Readiness', '', f"Strategy: {doc['strategy_id']}",
                f"Status: {decision['status']}; next review: {decision['next_checkpoint']} settled decisions.",
                f"Post-policy selected decisions: {len(selected)}; pending: {pending}; unresolved: {unresolved}.",
                'Fixed selection accounting: '+json.dumps(doc['selection_cohort_accounting']['counts']),
                '$1 flat; global limits: 5/day, $20/NFL week, pause at $30 cumulative confirmed loss.',
                'Recommendations are not confirmed wagers. Forecast deployment remains a separate review.', '',
                f"Confirmed wager record: {doc['executions']['confirmed_record']}; profit: ${confirmed_profit:.2f}.", '',
                '## Evidence Required', *['- '+b for b in provisional['blockers']], '',
                '## Forecast Review', f"Status: {deployment['status']}",
                *['- '+b for b in deployment['blockers']], '',
                'Other props, spreads and totals remain research until their own strategy is registered and passes.']
            (ROOT/'reports'/f'nfl_cash_readiness_{suffix}.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, default=date.today())
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
    result = report(args.date)
    print(json.dumps(dict(status=result['decision']['status'], pending=result['pending'],
        metrics=result['provisional']['metrics'], blockers=result['provisional']['blockers']), default=str))
