"""Versioned whole-CDF challengers. This module never authorizes a wager.

One same-book anchor is shared by every line for a player/game/lock. A market
anchor identifies one probability, not a complete market-implied distribution;
the log-odds shift below is an explicit shape assumption tested out of sample.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import betainc, expit, logit
from scipy.stats import norm, poisson

from nfl_pipeline.integrity import MODEL_ROOT, atomic_json

CONTRACT = 'nfl-distribution-policy-v1'
STORE = MODEL_ROOT / 'distribution_policy'
MARKETS = ('passing_yards', 'rushing_yards', 'receiving_yards', 'receptions',
           'passing_tds', 'rushing_tds', 'receiving_tds', 'spread', 'total', 'game_outcome')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def curve(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if (v.ndim != 1 or w.shape != v.shape or not len(v) or not np.isfinite(v).all()
            or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0):
        raise ValueError('invalid_distribution_curve')
    # All supported NFL stat/score outcomes are integer valued, including negative yards.
    v = np.floor(v + .5)
    support, inverse = np.unique(v, return_inverse=True)
    mass = np.bincount(inverse, weights=w)
    return dict(values=support.tolist(), weights=(mass / mass.sum()).tolist())


def from_cdf(values, cdf):
    f = np.asarray(cdf, float)
    if not np.isfinite(f).all() or (np.diff(f) < -1e-9).any():
        raise ValueError('nonmonotone_raw_distribution')
    f = np.maximum.accumulate(np.clip(f, 0, 1))
    f[-1] = 1.
    return curve(values, np.diff(np.r_[0., f]))


def normal_curve(mean, sigma, nonnegative=False):
    sigma = max(1., float(sigma))
    lo, hi = int(np.floor(mean-9*sigma)), int(np.ceil(mean+9*sigma))
    if nonnegative:
        lo = max(0, lo)
    values = np.arange(lo, max(lo+1, hi)+1)
    return from_cdf(values, norm.cdf(values+.5, loc=mean, scale=sigma))


def prop_curve(stat, projection, metrics, distribution, row, side_prob=None):
    distribution = distribution or {}
    if stat.endswith('_tds') or stat == 'receptions':
        values = np.arange(max(30, int(poisson.ppf(1-1e-12, max(.001, projection))))+1)
        f = poisson.cdf(values, max(.001, projection))
        td_any = row.get('_td_any_probability')
        if stat.endswith('_tds') and td_any is not None and (metrics.get('td_probability_accepted') or metrics.get('td_probability_pass')):
            f = 1-float(td_any)*poisson.sf(values, max(.001, projection))/(-np.expm1(-max(.001, projection)))
        return from_cdf(values, f)
    if distribution.get('kind') == 'empirical_oof_residual' and distribution.get('residual_quantiles'):
        residuals = np.asarray(distribution['residual_quantiles'], float)
        return curve(projection+residuals, np.ones(len(residuals)))
    if distribution.get('accepted_distribution') and distribution.get('kind') in {
            'qb_attempt_ypa_mixture', 'receiver_spike_mixture_v1', 'mixture_yardage_residual'}:
        if side_prob is None:
            raise ValueError('missing_raw_distribution_implementation')
        values = np.arange(-200, 1501)
        return from_cdf(values, [1-side_prob(stat, projection, float(v)+.5, metrics, distribution, row) for v in values])
    sigma = distribution.get('residual_sigma')
    if sigma is None:
        sigma = max(1., float(metrics.get('mae') or metrics.get('baseline_mae') or max(1., projection*.35))*1.35)
    return normal_curve(projection+.35*float(distribution.get('residual_bias') or 0), sigma)


def summary(c):
    v, w = np.asarray(c['values']), np.asarray(c['weights'])
    f = np.cumsum(w)
    q = v[np.searchsorted(f, [.1, .5, .9]).clip(0, len(v)-1)]
    return dict(mean=float(v @ w), p10=float(q[0]), median=float(q[1]), p90=float(q[2]))


def outcomes(c, line):
    v, w = np.asarray(c['values']), np.asarray(c['weights'])
    over, under, push = (float(w[v > line].sum()), float(w[v < line].sum()), float(w[v == line].sum()))
    return dict(over=over, under=under, push=push, conditional_over=over/(over+under) if over+under > 1e-12 else .5)


def calibrate(c, parameters):
    a, b = float(parameters.get('a', 1)), float(parameters.get('b', 1))
    if not (.25 <= a <= 4 and .25 <= b <= 4):
        raise ValueError('invalid_cdf_calibration_parameters')
    f = betainc(a, b, np.clip(np.cumsum(c['weights']), 0, 1))
    return from_cdf(c['values'], f)


def fit_cdf(rows):
    """Fit discrete interval likelihood, once per player-game, not per offer."""
    unique = {r['key']: r for r in reversed(rows)}
    lo, hi = [], []
    for key in sorted(unique):
        r = unique[key]
        v, w = np.asarray(r['curve']['values']), np.asarray(r['curve']['weights'])
        lo.append(w[v < r['actual']].sum()); hi.append(w[v <= r['actual']].sum())
    lo, hi = np.asarray(lo), np.asarray(hi)
    if len(lo) < 20:
        return dict(a=1., b=1.)
    def loss(theta):
        a, b = np.exp(theta)
        return -np.log(np.maximum(1e-12, betainc(a, b, hi)-betainc(a, b, lo))).mean()+.02*np.sum(theta**2)
    result = minimize(loss, [0., 0.], bounds=[(np.log(.25), np.log(4))]*2, method='L-BFGS-B')
    return dict(zip(('a', 'b'), np.exp(result.x).tolist())) if result.success else dict(a=1., b=1.)


def implied(price):
    price = float(price)
    if not np.isfinite(price) or abs(price) < 100:
        raise ValueError('invalid_price')
    return 100/(100+price) if price > 0 else -price/(100-price)


def anchor_for(offers, center):
    pairs = []
    for q in offers:
        if q.get('pair_quality') in ('synthetic', 'cross_book', 'one_sided'):
            continue
        try:
            a, b = implied(q['over_price']), implied(q['under_price'])
            pairs.append(dict(line=float(q['line']), probability=a/(a+b), book=q['bookmaker_key'],
                              over_price=float(q['over_price']), under_price=float(q['under_price']),
                              quote_time=str(q.get('fetched_at_utc')), offer_id=str(q.get('offer_id'))))
        except (KeyError, TypeError, ValueError):
            continue
    return min(pairs, key=lambda q: (abs(q['line']-center), q['line'], q['offer_id'])) if pairs else None


def market_curve(c, anchor):
    if not anchor or not 0 < float(anchor['probability']) < 1:
        raise ValueError('true_paired_anchor_required')
    f = np.clip(np.cumsum(c['weights']), 0, 1)
    def shifted(z):
        mapped = np.where(f <= 0, 0., np.where(f >= 1, 1., expit(logit(np.clip(f, 1e-15, 1-1e-15))+z)))
        return from_cdf(c['values'], mapped)
    target = anchor['probability']
    def error(z):
        return outcomes(shifted(z), anchor['line'])['conditional_over']-target
    if error(-32)*error(32) > 0:
        raise ValueError('market_anchor_outside_curve_support')
    return shifted(brentq(error, -32, 32))


def score(c, policy, anchor):
    calibrated = calibrate(c, policy)
    weight = float(policy.get('model_weight', 1))
    if not 0 <= weight <= 1:
        raise ValueError('invalid_model_weight')
    if weight == 1:
        return calibrated
    market = market_curve(calibrated, anchor)
    return curve(calibrated['values'], weight*np.asarray(calibrated['weights'])+(1-weight)*np.asarray(market['weights']))


def policy_key(market, release, book='fanduel', source='raw_distribution'):
    return '|'.join((CONTRACT, str(release), book, market, source))


def load_registry(name='challengers'):
    path = STORE / (name+'.json')
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding='utf-8'))
    if doc.get('contract') != CONTRACT or doc.get('sha256') != digest(doc.get('policies')):
        raise ValueError('distribution_registry_checksum_mismatch')
    return doc['policies']


def context(c, anchor, key, registry, active, now=None):
    """Policies and curves are copied into immutable locks for exact replay."""
    now = now or datetime.now(timezone.utc).isoformat()
    candidate = registry.get(key)
    deployed = active.get(key)
    for p in (candidate, deployed):
        if p and (p.get('key') != key or p.get('contract') != CONTRACT or p['created_at'] >= now):
            raise ValueError('invalid_or_future_distribution_policy')
    return dict(contract=CONTRACT, key=key, curve=c, anchor=anchor,
                challenger=candidate, deployed=deployed, captured_at=now)


def captured_score(ctx, name, line):
    policy = ctx.get(name)
    if not policy:
        return None
    if policy.get('model_weight', 1) < 1 and not ctx.get('anchor'):
        return None
    c = score(ctx['curve'], policy, ctx['anchor'])
    return dict(policy_id=policy['policy_id'], summary=summary(c), **outcomes(c, line))


def save_registry(policies, name):
    atomic_json(STORE/(name+'.json'), dict(contract=CONTRACT, policies=policies, sha256=digest(policies)))


@lru_cache(maxsize=4)
def source_artifact(path, sha256):
    import joblib
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError('challenger_source_checksum_mismatch')
    return joblib.load(path)


def receiving_curve(descriptor, row):
    import pandas as pd
    from nfl_pipeline.modeling.train_receiving_quantiles import curves
    artifact = source_artifact(descriptor['path'], descriptor['sha256'])
    values, weights = curves(artifact['bundle'], pd.DataFrame([row]))['calibrated_blend']
    return curve(values[0], weights[0])


def workload_descriptor(path, variant='combined'):
    from nfl_pipeline.modeling.role_workload import CONTRACT as WORKLOAD_CONTRACT, VARIANTS
    path = Path(path).resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = source_artifact(str(path), sha)
    if artifact.get('contract') != WORKLOAD_CONTRACT or variant not in VARIANTS:
        raise ValueError('invalid_workload_source')
    return dict(kind='role_workload', contract=WORKLOAD_CONTRACT, path=str(path), sha256=sha,
                variant=variant, stat=artifact['stat'], training_end=artifact['training_end'],
                production_release=artifact['production_release'])


def workload_source(descriptor):
    return 'role_workload:'+descriptor['sha256']+':'+descriptor['variant']


def repair_descriptor(path, variant='joint_calibrated'):
    from nfl_pipeline.modeling.forecast_repair import CONTRACT as REPAIR_CONTRACT, VARIANTS
    path = Path(path).resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = source_artifact(str(path), sha)
    valid = ('small_total',) if artifact.get('market') == 'total' else VARIANTS
    if artifact.get('contract') != REPAIR_CONTRACT or variant not in valid or variant == 'reference':
        raise ValueError('invalid_forecast_repair_source')
    return dict(kind='forecast_repair', contract=REPAIR_CONTRACT, path=str(path), sha256=sha,
        variant=variant, stat=artifact['market'], training_end=artifact['training_end'],
        production_release=artifact['production_release'])


def repair_source(descriptor):
    return 'forecast_repair:'+descriptor['sha256']+':'+descriptor['variant']


def register_source(descriptor):
    """Capture an offline source before fitting an offered-line policy; no deployment."""
    source = repair_source(descriptor)
    key = policy_key(descriptor['stat'], descriptor['production_release'], 'fanduel', source)
    sources = load_registry('sources')
    if key not in sources:
        sources[key] = dict(key=key, source=source, source_artifact=descriptor,
            created_at=datetime.now(timezone.utc).isoformat(), capture_only=True, cash_approved=False)
        save_registry(sources, 'sources')
    return sources[key]


def repair_error(descriptor, row, projection, release, cutoff):
    from nfl_pipeline.context_contract import safe_time
    from nfl_pipeline.modeling.role_workload import STATS
    artifact = source_artifact(descriptor['path'], descriptor['sha256'])
    if (artifact.get('contract') != descriptor['contract'] or artifact['market'] != descriptor['stat']
            or artifact['training_end'] != descriptor['training_end']):
        return 'repair_descriptor_mismatch'
    if release != artifact['production_release']:
        return 'repair_release_mismatch'
    at = safe_time(cutoff)
    if not at or artifact['training_end'][:10] >= at.date().isoformat():
        return 'repair_training_not_before_lock'
    if artifact['market'] == 'total':
        from nfl_pipeline.modeling.forecast_repair import total_inputs
        import pandas as pd
        x = total_inputs(pd.DataFrame([row]))
        if not x[artifact['model'].columns].notna().all(axis=None):
            return 'repair_missing_original_game_inputs'
    else:
        if row.get('position') not in artifact['model'].rate.priors:
            return 'repair_position_without_training_support'
        for name in (STATS[artifact['market']]+'_avg_5', STATS[artifact['market']]+'_avg_10',
                     artifact['market']+'_avg_10', 'n_games_prev_10'):
            try:
                if not np.isfinite(float(row.get(name))): return 'repair_missing_original_inputs'
            except (ValueError, TypeError): return 'repair_missing_original_inputs'
        if projection is None or not np.isfinite(float(projection)):
            return 'repair_missing_original_projection'
    return None


def workload_error(descriptor, row, projection, release, cutoff):
    from nfl_pipeline.context_contract import safe_time
    from nfl_pipeline.modeling.role_workload import STATS
    artifact = source_artifact(descriptor['path'], descriptor['sha256'])
    if (artifact.get('contract') != descriptor['contract'] or artifact['stat'] != descriptor['stat']
            or artifact['training_end'] != descriptor['training_end']):
        return 'workload_descriptor_mismatch'
    if release != artifact['production_release']:
        return 'workload_release_mismatch'
    at = safe_time(cutoff)
    if not at or artifact['training_end'][:10] >= at.date().isoformat():
        return 'workload_training_not_before_lock'
    if row.get('position') not in artifact['bundle']['model'].position_prior:
        return 'workload_position_without_training_support'
    for value in (projection, row.get(STATS[artifact['stat']]+'_avg_5'), row.get(artifact['stat']+'_avg_5')):
        try:
            if not np.isfinite(float(value)):
                return 'workload_missing_original_inputs'
        except (ValueError, TypeError):
            return 'workload_missing_original_inputs'
    return None


def source_curve(descriptor, row, projection=None):
    if descriptor.get('kind') == 'forecast_repair':
        import pandas as pd
        from nfl_pipeline.modeling.forecast_repair import dense_yard_curve
        artifact = source_artifact(descriptor['path'], descriptor['sha256'])
        v, w = artifact['model'].curve(pd.DataFrame([row]), descriptor['variant'],
            projection=None if projection is None else np.array([float(projection)]))
        return dense_yard_curve(v[0], w[0])
    if descriptor.get('kind') != 'role_workload':
        return receiving_curve(descriptor, row)
    import pandas as pd
    artifact = source_artifact(descriptor['path'], descriptor['sha256'])
    v, w, _ = artifact['bundle']['model'].yardage_curve(
        pd.DataFrame([row]), variant=descriptor['variant'], projection=np.array([float(projection)]))
    return curve(v[0], w[0])


def prop_contexts(
    raw, stat, release, book, offers, row, registry, active,
    projection=None, cutoff=None, source_curves=None,
):
    contexts = []
    source_curves = source_curves or {}
    sources = {'raw_distribution': None}
    for policy in list(registry.values())+list(active.values())+list(load_registry('sources').values()):
        descriptor = policy.get('source_artifact')
        if descriptor and policy['key'] == policy_key(stat, release, book, policy['source']):
            sources[policy['source']] = descriptor
    exclusions = {}
    for source, descriptor in sources.items():
        key = policy_key(stat, release, book, source)
        try:
            if descriptor and descriptor.get('kind') == 'stat_component':
                expected_source = 'stat_component:' + str(descriptor.get('component_id') or '')
                if source != expected_source:
                    raise ValueError('stat_component_descriptor_mismatch')
                if source not in source_curves:
                    raise ValueError('live_stat_component_curve_missing')
                c = source_curves[source]
            else:
                c = None
            if descriptor and descriptor.get('kind') in ('role_workload', 'forecast_repair'):
                check = repair_error if descriptor['kind'] == 'forecast_repair' else workload_error
                error = check(descriptor, row, projection, release, cutoff)
                if error:
                    if key in active:
                        raise ValueError(error)
                    exclusions[source] = error
                    continue
            if c is None:
                c = source_curve(descriptor, row, projection) if descriptor else raw
        except (ValueError, FileNotFoundError) as exc:
            if key in active:
                raise
            exclusions[source] = str(exc) if isinstance(exc, ValueError) else 'source_artifact_missing'
            continue
        anchor = anchor_for(offers, summary(c)['median'])
        ctx = context(c, anchor, key, registry, active)
        if descriptor:
            ctx['source_artifact'] = descriptor
        contexts.append(ctx)
    if exclusions:
        contexts[0]['source_exclusions'] = exclusions
    return contexts


def deploy(policy, review):
    """Forecast-only deployment; acceptance is recomputed, never a caller's bool."""
    from nfl_pipeline.modeling.validate_distribution_policy import deployment_blockers
    blockers = deployment_blockers(review)
    if blockers or review.get('policy_id') != policy['policy_id'] or review.get('policy_sha256') != digest(policy):
        raise ValueError('deployment_rejected:'+','.join(blockers or ['policy_evidence_mismatch']))
    active = load_registry('active')
    previous = active.get(policy['key'])
    event = dict(contract=CONTRACT, at=datetime.now(timezone.utc).isoformat(), policy=policy,
                 previous=previous, review=review, cash_approved=False)
    atomic_json(STORE/'deployments'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), event)
    # Exactly one forecast policy can own a release/book/stat at a time.
    prefix = policy['key'].rsplit('|', 1)[0]+'|'
    active = {k: v for k, v in active.items() if not k.startswith(prefix)}
    active[policy['key']] = policy
    save_registry(active, 'active')
    return event
