"""Preregistered FanDuel receiving strategy. Never writes a bet or deploys a model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from urllib.parse import urlparse

import numpy as np
import psycopg2
import psycopg2.extras

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling.prospective_market import locked_market

STORE = MODEL_ROOT/'receiving_research_trial'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_registration():
    path = STORE/'registration.json'
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if doc['sha256'] != digest(doc['config']):
        raise ValueError('Research trial registration checksum mismatch')
    return doc


def register(artifact, manifest):
    existing = load_registration()
    if existing:
        if existing['config']['run_id'] != artifact['run_id'] or existing['config']['artifact_sha256'] != manifest['sha256']:
            raise ValueError('Research trial already registered against a different benchmark')
        return existing
    if 'stable_ensemble_calibrated' not in artifact['models']['receiving_yards'].get('prospective_variants', []):
        raise ValueError('Pinned artifact does not contain the receiving challenger')
    config = dict(trial_id='fanduel-common-receiving-v1', contract='nfl-research-strategy-v1',
        registered_at=datetime.now(timezone.utc).isoformat(), run_id=artifact['run_id'],
        artifact_sha256=manifest['sha256'], production_release=artifact['production_release'],
        stat='receiving_yards', book='fanduel', variant='stable_ensemble_calibrated',
        market_keys=['player_reception_yds', 'player_receiving_yards'],
        rules=dict(max_per_date=3, max_per_game=2, max_per_player_game=1, max_quote_age_minutes=20,
            sides='original_locked_side_only', rank='challenger_EV_desc_then_forecast_id',
            require_true_pair=True, require_positive_ev=True, require_positive_market_edge=True,
            require_original_drift_pass=True, require_book_link=True,
            updates='first_eligible_capture_fills_remaining_daily_slots_no_replacements'),
        review=dict(min_independent_weeks=3, min_selected_player_games=50,
            brier_gain_lower_95_above_zero=True, calibration_must_improve=True,
            coverage_80_min=.75, coverage_80_max=.85, min_valid_close_coverage=.90,
            min_clv_beat_rate=.55, min_roi=0., min_average_clv=0.,
            max_player_share=.20, max_team_share=.35, max_week_share=.50),
        automatic_deployment=False, automatic_betting_approval=False, research_only=True)
    result = dict(config=config, sha256=digest(config))
    atomic_json(STORE/'registration.json', result)
    return result


def selection_error(record, row, config, now):
    p = record['forecast_payload']; q = (p.get('scoring_replay') or {}).get('offer') or {}
    if row.get('variant') != config['variant'] or record['stat'] != config['stat'] or p.get('book') != config['book']:
        return 'outside_trial'
    if p.get('model_version') != config['production_release']:
        return 'release_mismatch'
    lock, start = safe_time(record.get('created_at_utc')), safe_time(record.get('start_ts_utc'))
    registered = safe_time(config['registered_at'])
    if not lock or not start or not registered <= lock <= now < start:
        return 'not_a_new_post_registration_pregame_lock'
    if q.get('market_key') not in config['market_keys'] or q.get('is_alt_line') is True:
        return 'not_common_full_game_line'
    if locked_market(record)['market_probability'] is None:
        return 'missing_true_paired_lock_evidence'
    fetched = safe_time(q.get('fetched_at_utc'))
    if not fetched or (now-fetched).total_seconds() > 60*config['rules']['max_quote_age_minutes']:
        return 'quote_too_old_for_research_selection'
    link = q.get(p['side']+'_link')
    parsed = urlparse(str(link or ''))
    if parsed.scheme != 'https' or not (parsed.hostname == 'fanduel.com' or (parsed.hostname or '').endswith('.fanduel.com')):
        return 'missing_valid_fanduel_link'
    if p.get('drift_guard_pass') is not True:
        return 'original_drift_guard_not_passed'
    price = float(p['price']); payout = price/100 if price > 0 else 100/-price
    probability = float(row['same_side_probability'])
    if not np.isfinite(probability) or not 0 < probability < 1 or probability*(1+payout)-1 <= 0:
        return 'nonpositive_current_locked_ev'
    if probability <= locked_market(record)['market_probability']:
        return 'no_positive_market_edge'
    return None


def choose(records, rows, config, now, previous):
    lookup = {int(r['id']): r for r in records}; candidates = []; issues = Counter()
    for row in rows:
        record = lookup[int(row['forecast_id'])]
        error = selection_error(record, row, config, now)
        if error:
            issues[error] += 1; continue
        p = record['forecast_payload']; price = float(p['price'])
        payout = price/100 if price > 0 else 100/-price
        candidates.append(dict(forecast_id=int(record['id']), variant=row['variant'],
            day=str(p['game_date_et'])[:10], game_id=record['game_id'], player_id=record['player_id'],
            team=p.get('team_abbr'), season=record['season'], week=record['week'],
            book=p['book'], stat=p['stat'], side=p['side'], line=p['line'], price=p['price'],
            probability=row['same_side_probability'], market_probability=locked_market(record)['market_probability'],
            ev=row['same_side_probability']*(1+payout)-1, captured_at=now.isoformat()))
    selected = []; used = {(r['game_id'], r['player_id']) for r in previous}
    dates = Counter(r['day'] for r in previous); games = Counter(r['game_id'] for r in previous)
    for c in sorted(candidates, key=lambda c: (-c['ev'], c['forecast_id'])):
        key = (c['game_id'], c['player_id'])
        if key in used or dates[c['day']] >= config['rules']['max_per_date'] or games[c['game_id']] >= config['rules']['max_per_game']:
            issues['fixed_cap_or_duplicate'] += 1; continue
        used.add(key); dates[c['day']] += 1; games[c['game_id']] += 1; selected.append(c)
    return selected, dict(issues)


def captures(registration):
    documents = [json.loads(p.read_text()) for p in sorted((STORE/'captures').glob('*.json'))]
    if any(d.get('registration_sha256') != registration['sha256'] for d in documents):
        raise ValueError('Mixed research strategy registrations')
    return documents


def capture(records, document):
    registration = load_registration()
    if not registration:
        return dict(status='not_registered')
    config = registration['config']
    if (document['run_id'] != config['run_id'] or
            document['source_manifest']['sha256'] != config['artifact_sha256']):
        raise ValueError('Research strategy pinned benchmark mismatch')
    # Exclusive ownership keeps daily caps stable if two schedulers overlap.
    lock = STORE/'.capture.lock'
    owned = False
    try:
        with lock.open('x'):
            owned = True
            previous = [r for d in captures(registration) for r in d['selected']]
            now = datetime.now(timezone.utc)
            selected, excluded = choose(records, document['rows'], config, now, previous)
            result = clean(dict(registration_sha256=registration['sha256'], run_id=config['run_id'],
                scored_at=document['scored_at'], captured_at=now.isoformat(), selected=selected, exclusions=excluded))
            atomic_json(STORE/'captures'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'), result)
    finally:
        if owned:
            lock.unlink()
    return dict(status='research_only', selected=len(selected), exclusions=excluded, betting_approved=False)


def load_closes(ids):
    if not ids:
        return {}
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT prediction_id,valid_close_snapshot_captured,clv_prob_delta,
                       close_quality_reason,book,stat,side,locked_line,close_line,
                       close_fetched_at_utc,minutes_to_start FROM bets.nfl_prediction_clv
                       WHERE source_kind='prop' AND prediction_id=ANY(%s)""", (ids,))
        return {int(r['prediction_id']): dict(r) for r in cur.fetchall()}


def report(records, rows, metric_function):
    registration = load_registration()
    if not registration:
        return dict(status='not_registered', betting_approved=False)
    config = registration['config']; docs = captures(registration)
    selected = [r for d in docs for r in d['selected']]
    keys = {r['forecast_id'] for r in selected}
    matched = [r for r in rows if r['run_id'] == config['run_id'] and r['variant'] == config['variant']
               and r['forecast_id'] in keys]
    metrics = metric_function(matched)
    from nfl_pipeline.modeling.receiving_trial_checkpoint import metrics_for
    scoped = metrics_for(matched)
    record_lookup = {int(r['id']): r for r in records}
    all_candidates = [r for r in rows if r['run_id'] == config['run_id'] and r['variant'] == config['variant']
        and r['source_book'] == config['book'] and r['stat'] == config['stat']
        and safe_time(r['created_at']) >= safe_time(config['registered_at'])
        and (record_lookup[r['forecast_id']]['forecast_payload'].get('scoring_replay') or {}).get('offer', {}).get('market_key') in config['market_keys']]
    candidate_metrics = metrics_for(all_candidates)
    m = scoped.get('real_offers', {}); rules = config['review']
    # Pending/void/push rows remain in the accounting, never invented binary losses.
    lookup = {int(r['id']): r for r in records}
    settled_ids = {r['forecast_id'] for r in matched}
    def state_of(row):
        rec = lookup.get(row['forecast_id'])
        if not rec:
            return 'missing_original_lock'
        if row['forecast_id'] in settled_ids:
            return 'settled_binary'
        if rec.get('status') != 'final':
            return 'pending_game'
        if rec.get('graded_result')=='void_nonparticipant':
            return 'void_nonparticipant'
        if rec.get('actual') is None:
            return 'final_without_verified_participation_or_result'
        if float(rec['actual']) == float(row['line']):
            return 'push'
        return 'missing_validated_shadow'
    state = Counter(state_of(r) for r in selected)
    now = datetime.now(timezone.utc)
    mature_ids = {r['forecast_id'] for r in selected if r['forecast_id'] in lookup
                  and safe_time(lookup[r['forecast_id']].get('start_ts_utc'))
                  and safe_time(lookup[r['forecast_id']]['start_ts_utc']) <= now}
    closes = load_closes(sorted(mature_ids))
    # Use the same exact identity/timing checks as the operational checkpoint.
    from nfl_pipeline.modeling.receiving_trial_checkpoint import close_summary
    verified = close_summary(records,mature_ids,closes,now)
    valid_ids={c['forecast_id'] for c in verified['observations'] if c['valid_exact_capture']}
    valid = [c for i,c in closes.items() if i in valid_ids]
    coverage = len(valid)/len(mature_ids) if mature_ids else None
    close_reasons = verified['unknown_reasons']
    average = float(np.mean([float(c['clv_prob_delta']) for c in valid])) if valid else None
    beat = float(np.mean([float(c['clv_prob_delta']) > 0 for c in valid])) if valid else None
    settled = [r for r in selected if r['forecast_id'] in settled_ids]
    outcomes = {r['forecast_id']: r['outcome'] for r in matched}
    profits = [(r['price']/100 if r['price'] > 0 else 100/-r['price']) if outcomes[r['forecast_id']] else -1 for r in settled]
    concentration = {field: max(Counter(str(r.get(field)) for r in settled).values())/len(settled) if settled else None
                     for field in ('player_id', 'team', 'week')}
    blockers = []
    if m.get('weeks', 0) < rules['min_independent_weeks']:
        blockers.append('independent_weeks_below_review_floor')
    if m.get('unique_player_games', 0) < rules['min_selected_player_games']:
        blockers.append('selected_player_game_sample_below_review_floor')
    for name, gain in (('production', m.get('final_brier_gain', {})),
                       ('market', m.get('matched_market', {}).get('challenger_brier_gain_vs_market', {}))):
        if gain.get('lower_95') is None or gain['lower_95'] <= 0:
            blockers.append('selected_brier_gain_vs_'+name+'_unconfirmed')
    market = m.get('matched_market', {})
    if not market.get('rows') or market.get('pair_coverage', 0) < 1:
        blockers.append('selected_true_pair_evidence_incomplete')
    if not m.get('rows') or m['final']['calibration_error'] >= min(
            m['production']['calibration_error'], market.get('market', {}).get('calibration_error', -1)):
        blockers.append('selected_calibration_not_improved')
    weekly = market.get('weekly', [])
    improving_weeks = sum(w['challenger']['brier'] < min(w['production']['brier'], w['market']['brier']) for w in weekly)
    if not weekly or improving_weeks <= len(weekly)/2:
        blockers.append('advantage_not_consistent_across_weeks')
    interval = m.get('live_curve', {}).get('coverage_80')
    if interval is None or not rules['coverage_80_min'] <= interval <= rules['coverage_80_max']:
        blockers.append('interval_coverage_not_confirmed')
    if coverage is None or coverage < rules['min_valid_close_coverage']:
        blockers.append('valid_close_coverage_below_90_percent')
    if average is None or average <= 0 or beat < rules['min_clv_beat_rate']:
        blockers.append('positive_clv_not_confirmed')
    roi = float(np.mean(profits)) if profits else None
    if roi is None or roi <= 0:
        blockers.append('positive_prospective_roi_not_confirmed')
    for field, limit in (('player_id', 'max_player_share'), ('team', 'max_team_share'), ('week', 'max_week_share')):
        if concentration[field] is None or concentration[field] > rules[limit]:
            blockers.append(field+'_concentration_not_confirmed')
    doc = clean(dict(built_at=datetime.now(timezone.utc).isoformat(), registration=registration,
        status='manual_strategy_review_only' if not blockers else 'collecting_evidence',
        selected=len(selected), accounting=dict(state), metrics=scoped, all_candidates=candidate_metrics,
        scoring_cohorts=metrics,
        improving_independent_weeks=improving_weeks, blockers=blockers,
        hypothetical_flat_unit_roi=roi, valid_close_coverage=coverage, average_clv=average, clv_beat_rate=beat,
        close_eligible_started_selections=len(mature_ids), unknown_closes=len(mature_ids)-len(valid),
        unknown_close_reasons=dict(close_reasons), concentration=concentration,
        deployment_approved=False, betting_approved=False, automatic_promotion=False,
        next_action='Collect new pre-kickoff locks; three weeks is a review floor, never automatic approval.'))
    return doc
