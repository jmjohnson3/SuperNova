"""Versioned NFL cash eligibility. Forecast deployment is a separate decision."""
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
from scipy.stats import t

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json

STORE = MODEL_ROOT / 'cash_trial'
CONTRACT = 'nfl-cash-trial-v1'


@dataclass(frozen=True)
class Policy:
    version: str = CONTRACT
    book: str = 'fanduel'
    stake: float = 1.0
    daily_limit: int = 5
    weekly_stake_limit: float = 20.0
    loss_limit: float = 30.0
    checkpoints: tuple = (50, 100, 200)
    min_weeks: int = 3
    family_alpha: float = .05
    # Fixed family covers receiving/rushing/passing yards, three TD markets,
    # spread and total. Adding/replacing strategies cannot recycle this budget.
    strategy_slots: int = 8
    max_ece: float = .05
    min_close: float = .90
    max_stale: float = .02
    min_clv_beat: float = .55


POLICY = Policy()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
        separators=(',', ':')).encode()).hexdigest()


def register(config, *, store=STORE):
    """Pin an authorization cohort without changing an existing research trial."""
    path = store / 'registration.json'
    if path.exists():
        doc = json.loads(path.read_text())
        if digest(doc['config']) != doc['sha256']:
            raise ValueError('Cash policy registration checksum mismatch')
        for field in config:
            if config.get(field) != doc['config'].get(field):
                raise ValueError('Cash cohort changed; explicit new strategy registration required')
        return doc
    config = dict(config, policy=json.loads(json.dumps(asdict(POLICY))), registered_at=datetime.now(timezone.utc).isoformat())
    doc = dict(config=config, sha256=digest(config))
    atomic_json(path, doc)
    return doc


def load_registration(store=STORE):
    path = store / 'registration.json'
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if digest(doc['config']) != doc['sha256'] or doc['config']['policy'] != json.loads(json.dumps(asdict(POLICY))):
        raise ValueError('Cash policy registration mismatch')
    return doc


def week_lower(values, weeks, alpha):
    """Equal-week, one-sided Student interval; offers are not independent trials."""
    groups = {}
    for value, week in zip(values, weeks):
        groups.setdefault(str(week), []).append(float(value))
    means = np.array([np.mean(g) for g in groups.values()])
    if len(means) < POLICY.min_weeks or not np.isfinite(means).all():
        return None
    return float(means.mean() - t.ppf(1-alpha, len(means)-1)*means.std(ddof=1)/math.sqrt(len(means)))


def ece(rows, key):
    if not rows:
        return None
    bins = {}
    for r in rows:
        bins.setdefault(min(9, int(float(r[key])*10)), []).append(r)
    return sum(abs(sum(float(r[key])-r['outcome'] for r in group)) for group in bins.values())/len(rows)


def evaluate(rows, all_rows, checkpoint, *, unresolved=0):
    """Rows must be validated fixed selections, never chosen from settled winners."""
    blockers = []
    settled = [r for r in rows if r.get('result') in ('win', 'loss', 'push')]
    binary = [r for r in settled if r['result'] != 'push']
    weeks = [r['week_key'] for r in settled]
    alpha = POLICY.family_alpha / (POLICY.strategy_slots * len(POLICY.checkpoints) * 2)
    metrics = dict(decisions=len(settled), binary=len(binary), weeks=len(set(weeks)),
                   alpha_per_test=alpha, checkpoint=checkpoint, unresolved=unresolved)
    if len(settled) < checkpoint or checkpoint not in POLICY.checkpoints:
        blockers.append('checkpoint_sample_incomplete')
    if len(set(weeks)) < POLICY.min_weeks:
        blockers.append('independent_weeks_below_three')
    if unresolved:
        blockers.append('unresolved_selected_evidence')
    if len({r['decision_key'] for r in settled}) != len(settled):
        blockers.append('duplicate_selected_decision')
    roi_lower = week_lower([r['profit'] for r in settled], weeks, alpha)
    gain = [float(r['market_probability']-r['outcome'])**2 - float(r['probability']-r['outcome'])**2 for r in binary]
    gain_lower = week_lower(gain, [r['week_key'] for r in binary], alpha)
    metrics.update(roi=np.mean([r['profit'] for r in settled]).item() if settled else None,
                   roi_lower=roi_lower, market_brier_gain_lower=gain_lower, calibration=ece(binary, 'probability'))
    if roi_lower is None or roi_lower <= 0 or metrics['roi'] is None or metrics['roi'] <= 0:
        blockers.append('positive_roi_unconfirmed')
    if gain_lower is None or gain_lower <= 0:
        blockers.append('selected_market_advantage_unconfirmed')
    if metrics['calibration'] is None:
        blockers.append('selected_calibration_unknown')
    elif metrics['calibration'] > POLICY.max_ece:
        blockers.append('selected_calibration_above_five_percent')
    intervals = [r for r in settled if r.get('covered_80') is not None]
    metrics['coverage_80'] = np.mean([r['covered_80'] for r in intervals]).item() if intervals else None
    if len(intervals) != len(settled) or not .75 <= (metrics['coverage_80'] or 0) <= .85:
        blockers.append('distribution_coverage_not_confirmed')
    # Equal player-game weights prevent duplicate lines from dominating all-offer proof.
    groups = {}
    for r in all_rows:
        if r.get('outcome') is not None:
            groups.setdefault(r['decision_key'], []).append((r['probability']-r['outcome'])**2 - (r['production_probability']-r['outcome'])**2)
    all_delta = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
    metrics['all_offer_brier_delta'] = all_delta
    if all_delta is None or all_delta > 0:
        blockers.append('all_offer_regression_or_missing_proof')
    valid = [r for r in settled if r.get('clv') is not None]
    metrics.update(valid_close_coverage=len(valid)/len(settled) if settled else None,
        stale_close_rate=sum(r.get('stale_close') is True for r in settled)/len(settled) if settled else None,
        average_clv=float(np.mean([r['clv'] for r in valid])) if valid else None,
        clv_beat_rate=sum(r['clv'] > 0 for r in valid)/len(valid) if valid else None,
        unchanged_closes=sum(r['clv'] == 0 for r in valid))
    if metrics['valid_close_coverage'] is None or metrics['valid_close_coverage'] < POLICY.min_close:
        blockers.append('valid_closes_below_90_percent')
    if metrics['stale_close_rate'] is None:
        blockers.append('stale_close_quality_unknown')
    elif metrics['stale_close_rate'] > POLICY.max_stale:
        blockers.append('stale_closes_above_two_percent')
    if not valid or metrics['average_clv'] <= 0 or metrics['clv_beat_rate'] < POLICY.min_clv_beat:
        blockers.append('clv_not_confirming')
    for field, cap in (('player_id', .20), ('team', .35), ('week_key', .50)):
        share = max(Counter(r.get(field) for r in settled).values(), default=0)/max(1, len(settled))
        metrics[field+'_concentration'] = share
        if not settled or any(r.get(field) is None for r in settled) or share > cap:
            blockers.append(field+'_concentration')
    return dict(status='cash_trial_eligible' if not blockers else 'research', blockers=blockers, metrics=metrics,
                model_deployment_approved=False, wagers_placed=False)


def budget_error(reservations, candidate, now):
    """Reservations consume caps until explicitly released; unknown execution blocks."""
    live = [r for r in reservations if r['state'] != 'not_placed']
    if any(r['state'] == 'paused' for r in reservations):
        return 'cash_trial_paused'
    if any(r.get('execution_deviation') for r in reservations):
        return 'execution_deviation_review_required'
    if any(r['state'] in ('reserved', 'published') and safe_time(r['expires_at']) <= now for r in live):
        return 'execution_reconciliation_required'
    if any(r['state'] == 'confirmed' and (now-safe_time(r['expires_at'])).total_seconds() > 86400 for r in live):
        return 'confirmed_result_reconciliation_required'
    profits = [float(r.get('profit') or 0) for r in live if r['state'] == 'settled']
    if sum(profits) <= -POLICY.loss_limit:
        return 'cumulative_loss_limit'
    if any(r['decision_key'] == candidate['decision_key'] for r in reservations):
        return 'decision_already_reserved'
    if sum(r['day'] == candidate['day'] for r in live) >= POLICY.daily_limit:
        return 'daily_limit'
    if sum(float(r['stake']) for r in live if r['week_key'] == candidate['week_key']) + POLICY.stake > POLICY.weekly_stake_limit:
        return 'weekly_stake_limit'
    return None
