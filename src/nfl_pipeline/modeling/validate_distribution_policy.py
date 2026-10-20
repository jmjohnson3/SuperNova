"""Forward-only calibration, blend, fixed-ranking review and forecast deployment.

Historical fits are development evidence. Only a policy pinned before kickoff
can pass prospective deployment. No command in this module approves cash bets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling import distribution_policy as dp
from nfl_pipeline.modeling.chronological_calibration import forward_folds
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics, clustered_gain
from mlb_pipeline.atomic_io import atomic_write_text

ROOT = dp.MODEL_ROOT.parents[3]


def positive_side(row):
    return row['side'] in ('over', 'home')


def probability(row, c):
    p = dp.outcomes(c, row['line'])['conditional_over']
    return p if positive_side(row) else 1-p


def row_score(row, policy):
    c = dp.score(row['curve'], policy, row['anchor'])
    s = dp.summary(c)
    v, w = np.asarray(c['values']), np.asarray(c['weights'])
    return dict(row, challenger_probability=probability(row, c), **dp.summary(c),
                expected_interval_coverage=float(w[(v >= s['p10']) & (v <= s['p90'])].sum()),
                challenger_push=dp.outcomes(c, row['line'])['push'])


def metrics(rows, column='challenger_probability'):
    if not rows:
        return dict(rows=0)
    f = pd.DataFrame(rows)
    weight = 1/f.groupby('key').key.transform('size')
    binary = f.actual.notna() & f.actual.ne(f.line)
    result = probability_metrics(f.loc[binary, column], f.loc[binary, 'outcome'], weight.loc[binary])
    unique = f.drop_duplicates('key').dropna(subset=['actual'])
    result['player_games'] = len(unique)
    result['weeks'] = len(unique[['season', 'week']].drop_duplicates())
    if column == 'challenger_probability' and 'p10' in unique:
        result['distribution'] = projection_metrics(unique.actual, unique['mean'], unique.p10, unique.p90)
        result['distribution']['expected_interval_coverage'] = float(unique.expected_interval_coverage.mean())
        result['median'] = projection_metrics(unique.actual, unique['median'])
    return result


def fit_policy(frame, fit_asof=None):
    """Fit CDF on earlier weeks; tune identity-vs-calibration and blend on last week."""
    periods = sorted(set(zip(frame.season.astype(int), frame.week.astype(int))))
    workload = 'source' in frame and frame.source.astype(str).str.startswith(
        ('role_workload:', 'forecast_repair:', 'stat_component:')
    ).all()
    if len(periods) < 2:
        return None
    earlier, held, _ = list(forward_folds(frame))[-1]
    if fit_asof is not None:
        # A policy fitted NOW may use labels observed now. This is not a claim
        # that those labels were available at the historical tuning lock.
        periods_by_row = pd.Series(list(zip(frame.season.astype(int), frame.week.astype(int))), index=frame.index)
        available = pd.to_datetime(frame.label_available_at, utc=True, errors='coerce')
        earlier = frame.loc[periods_by_row.map(lambda p: p < periods[-1]) & available.lt(pd.Timestamp(fit_asof))
                            & ~frame.key.isin(held.key) & frame.actual.notna()].copy()
        held = held.loc[pd.to_datetime(held.label_available_at, utc=True, errors='coerce').lt(pd.Timestamp(fit_asof))]
    if earlier.key.nunique() < 20 or held.key.nunique() < 10:
        return None
    rows = held.to_dict('records')
    fitted = dp.fit_cdf(earlier.to_dict('records'))
    identity = dict(a=1., b=1., model_weight=1.)
    raw = metrics([row_score(r, identity) for r in rows])
    calibrated = metrics([row_score(r, dict(fitted, model_weight=1.)) for r in rows])
    improves = (calibrated.get('brier', 1) < raw.get('brier', 0)
        and calibrated.get('calibration_error', 1) <= raw.get('calibration_error', 0)
        and abs(calibrated['distribution'].get('coverage_80', 0)-.8) <= abs(raw['distribution'].get('coverage_80', 0)-.8)+1e-12)
    shape = fitted if improves else dict(a=1., b=1.)
    scores, comparisons = {}, {}
    # Predeclared weak prior toward the book for a sparse workload challenger.
    # Offers on the same outcome never buy additional effective sample size.
    regularization = .01 * 50 / (50 + held.key.nunique()) if workload else 0.
    grid = [0., .25, .5, .75, 1.] if workload else np.linspace(0, 1, 11)
    for weight in ([1.] if held.market.eq('game_outcome').all() else grid):
        candidate = dict(shape, model_weight=float(weight))
        try:
            comparisons[float(weight)] = metrics([row_score(r, candidate) for r in rows])
            scores[float(weight)] = comparisons[float(weight)].get('brier', float('inf')) + regularization*weight**2
        except ValueError:
            continue
    if not scores:
        return None
    # Ties favor the smaller departure from the market, never forced model trust.
    weight = min(scores, key=lambda w: (scores[w], w))
    return dict(shape, model_weight=weight, calibration_enabled=improves,
                inner_raw=raw, inner_calibrated=calibrated,
                blend_comparisons=comparisons, blend_regularization=regularization,
                calibration_player_games=earlier.key.nunique(), tuning_player_games=held.key.nunique(),
                ranking_support=float(held.key.nunique()/(held.key.nunique()+100)),
                training_periods=periods[:-1], tuning_period=periods[-1],
                fit_scope='current_time_development' if fit_asof is not None else 'historical_lock_time')


def forward_evaluate(frame, universe=None):
    rows, folds = [], []
    for train, test, metadata in forward_folds(frame):
        policy = fit_policy(train)
        if policy is None:
            folds.append(dict(metadata, status='insufficient_earlier_fit_and_tune_weeks'))
            continue
        scored = []
        for r in test.to_dict('records'):
            try:
                s = row_score(r, policy)
                s['scored_curve'] = dp.score(r['curve'], policy, r['anchor'])
                s['policy_ranking_support'] = policy['ranking_support']
                s['raw_model_probability'] = probability(r, r['curve'])
                s['calibrated_model_probability'] = probability(r, dp.calibrate(r['curve'], policy))
                s['market_shape_probability'] = probability(r, dp.score(r['curve'], dict(policy, model_weight=0), r['anchor'])) if r['anchor'] else np.nan
                scored.append(s)
            except ValueError:
                continue
        rows.extend(scored)
        folds.append(dict(metadata, status='evaluated', policy=policy, metrics=metrics(scored)))
    result = dict(folds=folds, metrics=metrics(rows), production=metrics(rows, 'production_probability'),
                  market=metrics(rows, 'market_probability'))
    for name in ('raw_model_probability', 'calibrated_model_probability', 'market_shape_probability'):
        result[name] = metrics(rows, name)
    result['scope'] = 'historical_development_only'
    if rows:
        result['production_cohorts'] = {version: dict(production=metrics(g.to_dict('records'), 'production_probability'),
            challenger=metrics(g.to_dict('records')), market=metrics(g.to_dict('records'), 'market_probability'))
            for version, g in pd.DataFrame(rows).groupby('scoring_version')} if 'scoring_version' in rows[0] else {}
        fixed = [r for r in rows if r.get('exact_selected')]
        result['original_fixed_selections'] = dict(production=metrics(fixed, 'production_probability'),
            challenger=metrics(fixed), market=metrics(fixed, 'market_probability'))
        if universe is not None:
            result['selection_comparison'] = selection_comparison(rows, universe)
    return result, rows


def prepare_props(records, ledger, fixed_ids=()):
    from nfl_pipeline.modeling.fit_market_calibration import prepare_captured
    from nfl_pipeline.modeling.scoring_versions import archived_candidate
    frame, excluded = prepare_captured(records, {int(r['prediction_id']) for r in ledger} | set(fixed_ids), require_market_layer=False)
    source = {int(r['id']): r for r in records}
    ledger_ids = {int(r['prediction_id']) for r in ledger}
    result, cache = [], {}
    for r in frame.to_dict('records'):
        original = source[int(r['id'])]; p = original['forecast_payload']; cap = r['captured']; q = cap['offer']
        ctx = next((c for c in q.get('distribution_policy_variants', []) if c['key'].endswith('|raw_distribution')),
                   q.get('distribution_policy_capture'))
        cache_key = (r['key'], r['stat'], str(p['prediction_context_cutoff_utc']), cap['scoring_fingerprint'])
        try:
            if cache_key not in cache:
                cache[cache_key] = (ctx['curve'] if ctx else dp.prop_curve(r['stat'], cap['projection'], cap['metrics'],
                    cap['distribution'], cap['row'], archived_candidate(cap['scoring_fingerprint']).__globals__['_side_prob']))
            c = cache[cache_key]
            anchor = ctx['anchor'] if ctx else dp.anchor_for([q], dp.summary(c)['median'])
            if not anchor:
                raise ValueError('missing_true_paired_anchor')
            # Test feasibility now; do not silently skip hard offers in evaluation.
            dp.market_curve(c, anchor)
        except ValueError as exc:
            excluded[str(exc)] = excluded.get(str(exc), 0)+1
            continue
        a, b = dp.implied(q['over_price']), dp.implied(q['under_price'])
        market = a/(a+b) if r['side'] == 'over' else b/(a+b)
        base = dict(r, market=r['stat'], curve=c, anchor=anchor, source='raw_distribution',
            context=ctx, payload=p, market_probability=market, production_probability=p['probability'],
            production_p10=p.get('projection_p10'), production_p90=p.get('projection_p90'),
            day=str(p['game_date_et']), batch=p['prediction_context_cutoff_utc'], book=p['book'],
            price=float(p['price']), exact_selected=int(r['id']) in ledger_ids or int(r['id']) in fixed_ids,
            outcome=None if not np.isfinite(r['actual']) or r['actual'] == r['line'] else float((r['actual'] > r['line']) == (r['side'] == 'over')))
        result.append(base)
        for component_id, shadow in (p.get('stat_model_challengers') or {}).items():
            if not shadow.get('same_offer') or not isinstance(shadow.get('curve'), dict):
                continue
            try:
                shadow_curve = dp.curve(shadow['curve']['values'], shadow['curve']['weights'])
                dp.market_curve(shadow_curve, anchor)
            except (KeyError, TypeError, ValueError) as exc:
                reason = 'invalid_stat_component_curve:' + str(exc)
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            descriptor = {
                'kind': 'stat_component',
                'component_id': str(component_id),
                'stat': str(r['stat']),
                'outputs': list(shadow.get('outputs') or ()),
                'base_release': str(r['model_version']),
            }
            result.append(dict(
                base,
                curve=shadow_curve,
                source='stat_component:' + str(component_id),
                source_artifact=descriptor,
                context=None,
            ))
    # Legacy single-line anchors cannot certify cross-line coherence. Reuse one
    # actual captured quote per player/book/batch for retrospective development.
    groups = {}
    for r in result:
        groups.setdefault((r['key'], r['stat'], r['book'], r['batch']), []).append(r)
    for group in groups.values():
        legacy = [r for r in group if not r['context']]
        if legacy:
            anchor = dp.anchor_for([r['captured']['offer'] for r in legacy], dp.summary(legacy[0]['curve'])['median'])
            for r in legacy:
                r['anchor'] = anchor
    return result, excluded


def load_games():
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT p.id,p.game_id,p.season,p.week,p.created_at_utc,p.forecast_payload,g.start_ts_utc,g.status,
            r.actual_home_margin,r.actual_total_points,r.result,r.updated_at_utc AS label_available_at
            FROM bets.nfl_game_predictions p JOIN raw.nfl_games g USING(game_id)
            LEFT JOIN bets.nfl_game_prediction_results r ON r.prediction_id=p.id
            WHERE p.integrity_version='nfl-asof-v2' ORDER BY p.created_at_utc,p.id""")
        records = [dict(r) for r in cur.fetchall()]
    from nfl_pipeline.context_contract import safe_time
    rows, excluded, seen = [], Counter(), set()
    for r in records:
        p = r.get('forecast_payload') or {}; ctx = p.get('distribution_policy_capture')
        if not ctx:
            excluded['missing_locked_game_distribution'] += 1; continue
        quote, lock, start = map(safe_time, (p.get('quote_fetched_at_utc'), r['created_at_utc'], r['start_ts_utc']))
        if not all((quote, lock, start)) or not quote <= lock < start or (lock-quote).total_seconds() > 1200:
            excluded['invalid_game_lock_timing'] += 1; continue
        anchor = ctx.get('anchor')
        if not anchor or p.get('book') != 'fanduel':
            excluded['missing_true_game_pair'] += 1; continue
        market = p['market']; line = float(p['distribution_threshold'])
        key = (r['game_id'], market, p['book'], line, p['side'], p['model_version'])
        if key in seen:
            excluded['later_game_revision'] += 1; continue
        seen.add(key)
        actual = r['actual_home_margin' if market == 'spread' else 'actual_total_points']
        actual = float(actual) if actual is not None and r['status'] == 'final' and r['result'] in ('win', 'loss', 'push') else np.nan
        rows.append(dict(id=r['id'], key=str(r['game_id']), game_id=r['game_id'], player_id='game',
            market=market, stat=market, season=r['season'], week=r['week'], source='raw_distribution',
            model_version=p['model_version'], scoring_version=dp.CONTRACT, book=p['book'], side=p['side'],
            line=line, price=float(p['price']), locked_at=lock, label_available_at=r['label_available_at'],
            actual=actual, outcome=None if not np.isfinite(actual) or actual == line else float((actual > line) == (p['side'] in ('over', 'home'))),
            curve=ctx['curve'], anchor=anchor, context=ctx, payload=p, day=str(p['game_date_et']),
            production_p10=dp.summary(ctx['curve'])['p10'], production_p90=dp.summary(ctx['curve'])['p90'],
            batch=p['prediction_context_cutoff_utc'], production_probability=p['probability'],
            market_probability=anchor['probability'] if p['side'] in ('over', 'home') else 1-anchor['probability'],
            exact_selected=False))
    winner_rows = []
    for r in rows:
        if r['market'] != 'spread' or not r['payload'].get('game_outcome_policy_capture'):
            continue
        ctx = r['payload']['game_outcome_policy_capture']
        s = dp.summary(ctx['curve'])
        winner_rows.append(dict(r, market='game_outcome', stat='game_outcome', line=0., side='home',
            anchor=None, context=ctx, market_probability=np.nan,
            production_probability=r['payload']['game_outcome_distribution']['conditional_over'],
            production_p10=s['p10'], production_p90=s['p90'],
            outcome=None if not np.isfinite(r['actual']) or r['actual'] == 0 else float(r['actual'] > 0)))
    return rows+winner_rows, dict(excluded)


def candidate_at_lock(row, quote, challenger=False):
    """Isolate the replay clock without mutating live/archived module globals."""
    from types import FunctionType
    from nfl_pipeline.context_contract import safe_time
    from nfl_pipeline.modeling.predict_player_props import _candidate_from_offer
    cap = row['captured']
    fingerprint = cap.get('scoring_fingerprint')
    fn = _candidate_from_offer
    if fingerprint and not challenger:
        from nfl_pipeline.modeling.scoring_versions import archived_candidate
        fn = archived_candidate(fingerprint)
    helper = fn.__globals__.get('_game_has_started')
    if fingerprint and helper:
        at = safe_time(row.get('locked_at'))
        if at is None:
            raise ValueError('missing_historical_selection_clock')
        namespace = dict(fn.__globals__, _game_has_started=lambda value: helper(value, now_utc=at))
        isolated = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
        isolated.__kwdefaults__ = fn.__kwdefaults__
        fn = isolated
    if cap.get('exact_overlay_present'):
        raise ValueError('exact_overlay_artifact_not_captured')
    return fn(cap['row'], cap['stat'], cap['projection'], cap['baseline'], cap['metrics'], quote,
              cap['distribution'], min_ev=cap['min_ev'], model_artifact={},
              probability_calibration_by_key={tuple(x['key']): x['value'] for x in cap['calibration']},
              clv_guard_by_key={tuple(x['key']): x['value'] for x in cap['clv_guards']})


def unchanged_selection(rows, challenger=False, conservative=False):
    """Replay the actual prop scorer and actual cap, not a new ranking formula."""
    from nfl_pipeline.modeling.predict_player_props import _candidate_from_offer, _apply_daily_micro_cap
    selected, groups, skipped = [], {}, Counter()
    for r in rows:
        if r['market'] in ('spread', 'total', 'game_outcome'):
            # Same two-sided EV selection as the game scorer, no prop daily cap.
            if not challenger or 'scored_curve' not in r:
                selected.append(dict(r, selected_probability=r['production_probability'], selected_outcome=r['outcome']))
            elif r['market'] == 'game_outcome':
                selected.append(dict(r, selected_probability=r['challenger_probability'], selected_outcome=r['outcome']))
            else:
                mass = dp.outcomes(r['scored_curve'], r['line'])
                anchor = r['anchor']
                if not all(k in anchor for k in ('over_price', 'under_price')):
                    skipped['missing_locked_game_side_prices'] += 1; continue
                options = []
                for positive in (True, False):
                    p = mass['conditional_over'] if positive else 1-mass['conditional_over']
                    price = anchor['over_price' if positive else 'under_price']
                    payout = price/100 if price > 0 else -100/price
                    options.append(((1-mass['push'])*(p*(1+payout)-1), positive, p))
                _, positive, p = max(options, key=lambda x: x[0])
                selected.append(dict(r, selected_probability=p,
                    selected_outcome=None if not np.isfinite(r['actual']) or r['actual']==r['line'] else float((r['actual']>r['line'])==positive)))
            continue
        groups.setdefault((r['day'], r['batch'], r['captured'].get('scoring_fingerprint')), []).append(r)
    for group in groups.values():
        candidates, originals, failed = [], [], False
        for r in sorted(group, key=lambda r: r['id']):
            cap = r['captured']; q = deepcopy(cap['offer'])
            if challenger and 'scored_curve' in r:
                q['distribution_policy_score'] = dict(policy_id='offline_fixed_policy',
                    **dp.outcomes(r['scored_curve'], r['line']), summary=dp.summary(r['scored_curve']))
            try:
                candidate = candidate_at_lock(r, q, challenger and 'scored_curve' in r)
            except (ValueError, KeyError):
                skipped['full_path_replay_failed'] += 1; failed = True; continue
            if not challenger and (candidate['side'] != r['side'] or abs(candidate['probability']-r['production_probability']) > 1e-10):
                skipped['replay_differs_from_locked_production'] += 1; failed = True; continue
            if challenger and 'scored_curve' in r:
                candidate['tier'] = candidate.get('forecast_policy_pre_cash_tier', candidate['tier'])
            candidates.append(candidate); originals.append(r)
        if failed:
            skipped['incomplete_ranking_batch'] += 1
            continue
        ranked = deepcopy(candidates)
        if conservative:
            for candidate, r in zip(ranked, originals):
                if challenger and 'scored_curve' in r:
                    candidate['ev'] = conservative_rank_score(candidate, r.get('policy_ranking_support', 0.))
                    if candidate['ev'] <= 0:
                        candidate['tier'] = 'paper'
        fingerprint = group[0]['captured'].get('scoring_fingerprint')
        cap_fn = _apply_daily_micro_cap
        if fingerprint:
            from nfl_pipeline.modeling.scoring_versions import archived_candidate
            cap_fn = archived_candidate(fingerprint).__globals__.get('_apply_daily_micro_cap')
            if cap_fn is None:
                skipped['missing_archived_ranking'] += 1; continue
        cap_fn(ranked, 3)
        for candidate, rank, r in zip(candidates, ranked, originals):
            if rank['tier'] != 'micro_projection':
                continue
            # Selection can change side. Score the actual selected side/price.
            actual = r['actual']; side = candidate['side']
            selected.append(dict(r, selected_side=side, selected_price=candidate['price'],
                selected_probability=candidate['probability'], selected_ev=candidate['ev'], ranking_score=rank['ev'],
                selected_push=candidate.get('push_probability', 0.),
                selected_market_probability=(r['market_probability'] if side==r['side'] else 1-r['market_probability']) if 'market_probability' in r else np.nan,
                selected_outcome=None if not np.isfinite(actual) or actual == r['line'] else float((actual > r['line']) == (side == 'over'))))
    return selected, dict(skipped)


def conservative_rank_score(candidate, support):
    """Offline ranking penalty, not a replacement probability or confidence bound."""
    p, market = candidate['probability'], candidate.get('market_no_vig_probability')
    if market is None:
        return float('-inf')
    price = candidate['price']
    payout = price/100 if price > 0 else -100/price
    penalty = (1-candidate.get('push_probability', 0.))*(1+payout)*max(0., p-market)*(1-np.clip(support, 0, 1))
    return float(candidate['ev']-penalty)


def selection_comparison(scored, universe):
    batches = {(r['day'], r['batch']) for r in scored}
    replacement = {(r['market'], r['id']): r for r in scored}
    pool = [replacement.get((r['market'], r['id']), r) for r in universe
            if (r['day'], r['batch']) in batches and r['source']=='raw_distribution']
    pool = list({(r['market'], r['id']): r for r in pool}.values())
    output = {}
    for name, changed, conservative in [('production', False, False), ('unchanged_ranking', True, False),
                                       ('conservative_ranking', True, True)]:
        selected, excluded = unchanged_selection(pool, changed, conservative)
        selected = [r for r in selected if (r['market'], r['id']) in replacement]
        output[name] = dict(probability=probability_metrics([r['selected_probability'] for r in selected],
            [r['selected_outcome'] for r in selected]),
            market=probability_metrics([r.get('selected_market_probability', np.nan) for r in selected],
                [r['selected_outcome'] for r in selected]),
            selected=[{k: r.get(k) for k in ('id', 'season', 'week', 'day', 'game_id', 'player_id',
                'selected_side', 'line', 'selected_price', 'selected_probability', 'selected_push',
                'selected_ev', 'ranking_score', 'selected_market_probability', 'selected_outcome')} for r in selected],
            exclusions=excluded)
    output['scope'] = 'counterfactual_saved_offer_universe_no_cash_authorization'
    output['ranking_only_penalty'] = 'positive_model_market_disagreement * (1 - n_player_games/(n_player_games+100))'
    return output


def review(rows, policy, universe=None):
    from nfl_pipeline.context_contract import safe_time
    scored = []
    for r in rows:
        ctx = r.get('context') or {}; pinned = ctx.get('challenger')
        if not pinned or dp.digest(pinned) != dp.digest(policy) or safe_time(r['locked_at']) <= safe_time(policy['created_at']):
            continue
        s = row_score(r, policy)
        s['scored_curve'] = dp.score(r['curve'], policy, r['anchor'])
        s['policy_ranking_support'] = policy.get('ranking_support', 0.)
        scored.append(s)
    batches = {(r['day'], r['batch']) for r in scored}
    replacement = {(r['market'], r['id']): r for r in scored}
    pool = [replacement.get((r['market'], r['id']), r) for r in (universe or rows)
            if (r['day'], r['batch']) in batches and (r['source']=='raw_distribution' or (r['market'], r['id']) in replacement)]
    pool = list({(r['market'], r['id']): r for r in pool}.values())
    baseline_selected, old_errors = unchanged_selection(pool)
    changed_selected, new_errors = unchanged_selection(pool, True)
    baseline_selected = [r for r in baseline_selected if (r['market'], r['id']) in replacement]
    changed_selected = [r for r in changed_selected if (r['market'], r['id']) in replacement]
    fixed = [r for r in scored if r['exact_selected']]
    unique = pd.DataFrame(scored)
    gain = dict(lower_95=None, clusters=0)
    coverage = None
    if not unique.empty:
        binary = unique.loc[unique.outcome.notna()].copy()
        binary['reference_error'] = (binary.production_probability-binary.outcome)**2
        binary['challenger_error'] = (binary.challenger_probability-binary.outcome)**2
        if len(binary):
            gain = clustered_gain(binary.groupby(['season', 'week', 'key'])[['reference_error', 'challenger_error']].mean().reset_index())
        if {'production_p10', 'production_p90'}.issubset(unique.columns):
            u = unique.drop_duplicates('key').dropna(subset=['actual', 'production_p10', 'production_p90'])
            coverage = float(((u.actual >= u.production_p10) & (u.actual <= u.production_p90)).mean()) if len(u) else None
    def selected_metrics(selected):
        return probability_metrics([r.get('selected_probability', r.get('challenger_probability', r['production_probability'])) for r in selected],
            [r.get('selected_outcome', r['outcome']) for r in selected])
    result = dict(policy_id=policy['policy_id'], policy_sha256=dp.digest(policy),
        market=policy['key'].split('|')[3],
        scope='prospective_same_policy_same_offers', production=metrics(scored, 'production_probability'),
        challenger=metrics(scored), market_baseline=metrics(scored, 'market_probability'),
        brier_gain=gain, production_coverage=coverage, fixed_selections=metrics(fixed),
        production_ranked=selected_metrics(baseline_selected), challenger_ranked=selected_metrics(changed_selected),
        ranking_exclusions=dict(Counter(old_errors)+Counter(new_errors)),
        fixed_ids=[r['id'] for r in fixed], unchanged_ranking=True, cash_approved=False)
    if policy.get('source', '').startswith(('role_workload:', 'forecast_repair:', 'stat_component:')) and policy['key'].split('|')[3] not in ('total', 'spread'):
        result['selection_comparison'] = selection_comparison(scored, universe or rows)
        result['fixed_selection_comparison'] = dict(production=metrics(fixed, 'production_probability'),
            challenger=metrics(fixed), market=metrics(fixed, 'market_probability'))
        result['selected_comparison_scope'] = 'identical_challenger_selected_offers'
        result['production_on_challenger_selected'] = probability_metrics(
            [r['production_probability'] if r['selected_side']==r['side'] else 1-r['production_probability'] for r in changed_selected],
            [r['selected_outcome'] for r in changed_selected])
        result['market_on_challenger_selected'] = probability_metrics(
            [r['selected_market_probability'] for r in changed_selected], [r['selected_outcome'] for r in changed_selected])
        result['selected_player_games'] = len({r['key'] for r in changed_selected if r['selected_outcome'] is not None})
    result['selection_blockers'] = selection_blockers(result)
    result['blockers'] = deployment_blockers(result)
    return result


def deployment_blockers(review):
    c, p = review.get('challenger', {}), review.get('production', {})
    blockers = []
    if review.get('scope') != 'prospective_same_policy_same_offers': blockers.append('not_prospective')
    if c.get('weeks', 0) < 3: blockers.append('needs_three_independent_weeks')
    if c.get('player_games', 0) < 60: blockers.append('needs_60_settled_player_games')
    if c.get('brier', 1) >= p.get('brier', 0): blockers.append('final_brier_not_improved')
    if c.get('calibration_error', 1) >= p.get('calibration_error', 0): blockers.append('calibration_not_improved')
    if (review.get('brier_gain', {}).get('lower_95') or 0) <= 0: blockers.append('weekly_gain_uncertain')
    coverage = c.get('distribution', {}).get('coverage_80')
    production_coverage = review.get('production_coverage')
    count_market = review.get('market') in ('passing_tds', 'rushing_tds', 'receiving_tds', 'receptions')
    expected = c.get('distribution', {}).get('expected_interval_coverage', .8) if count_market else .8
    if coverage is None or abs(coverage-expected) > .05 or production_coverage is None or (
            not count_market and abs(coverage-.8) > abs(production_coverage-.8)+1e-12):
        blockers.append('coverage_not_proven')
    # A missing cash-selected cohort cannot veto a demonstrated forecast gain.
    # Observed selection harm and incomplete replay remain forecast regressions.
    if 'selected_probabilities_worse' in selection_blockers(review):
        blockers.append('selected_probabilities_worse')
    if review.get('ranking_exclusions') or not review.get('unchanged_ranking'): blockers.append('ranking_replay_incomplete')
    return blockers


def selection_blockers(review):
    blockers = []
    a, b = review.get('production_ranked', {}), review.get('challenger_ranked', {})
    if review.get('selected_comparison_scope') == 'identical_challenger_selected_offers':
        a = review.get('production_on_challenger_selected', {})
        count = review.get('selected_player_games', 0)
    else:
        count = min(a.get('rows', 0), b.get('rows', 0))
    if count < 20: blockers.append('selected_pick_sample_missing')
    elif b['brier'] > a['brier'] or b['calibration_error'] > a['calibration_error']:
        blockers.append('selected_probabilities_worse')
    return blockers


def workload_rows(props, descriptor):
    """Only original lock-time inputs; prospective captures take precedence later."""
    excluded, eligible = Counter(), []
    for r in props:
        if r['market'] != descriptor['stat'] or r['book'] != 'fanduel':
            continue
        error = dp.workload_error(descriptor, r['captured']['row'], r['captured']['projection'],
                                  r['model_version'], r['payload'].get('prediction_context_cutoff_utc'))
        if error:
            excluded[error] += 1
        else:
            eligible.append(r)
    groups = {}
    for r in eligible:
        groups.setdefault((r['key'], r['book'], r['batch'], r['captured']['projection']), []).append(r)
    artifact = dp.source_artifact(descriptor['path'], descriptor['sha256'])
    groups = list(groups.values())
    result = []
    for offset in range(0, len(groups), 64):
        batch = groups[offset:offset+64]
        print('Workload curves', offset, '/', len(groups), flush=True)
        v, w, _ = artifact['bundle']['model'].yardage_curve(
            pd.DataFrame([g[0]['captured']['row'] for g in batch]), variant=descriptor['variant'],
            projection=np.array([g[0]['captured']['projection'] for g in batch]))
        for group, values, weights in zip(batch, v, w):
            c = dp.curve(values, weights)
            anchor = dp.anchor_for([r['captured']['offer'] for r in group], dp.summary(c)['median'])
            try:
                dp.market_curve(c, anchor)
            except ValueError as exc:
                excluded[str(exc)] += len(group); continue
            for r in group:
                result.append(dict(r, curve=c, anchor=anchor, source=dp.workload_source(descriptor),
                                   source_artifact=descriptor, context=None))
    return result, dict(excluded)


def load_prop_window(season=None, weeks=None):
    from nfl_pipeline.modeling.live_scoring_replay import load_rows
    if season is None:
        return load_rows()
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute('SELECT DISTINCT game_date_et FROM raw.nfl_games WHERE season=%s AND week=ANY(%s) ORDER BY game_date_et',
                    (season, list(weeks)))
        days = [r[0] for r in cur.fetchall()]
    records = []
    for day in days:
        print('Loading original locks', day, flush=True)
        records.extend(r for r in load_rows(day) if r['season']==season and r['week'] in weeks)
    return records


def run(markets=dp.MARKETS, register=False, deploy_approved=False, receiving_artifact=None,
        workload_artifact=None, workload_variant='combined', season=None, weeks=None):
    from nfl_pipeline.modeling.live_scoring_replay import load_rows
    from nfl_pipeline.modeling.final_probability_validation import attach_label_availability, load_ledger
    props, exclusions = [], {}
    if any(m not in ('spread', 'total', 'game_outcome') for m in markets):
        # Preserve the other markets competing under the original daily cap.
        records = attach_label_availability(load_prop_window(season, weeks))
        print('Loaded prop locks', len(records), flush=True)
        from nfl_pipeline.modeling import receiving_research_trial as trial
        registration = trial.load_registration()
        fixed_ids = {int(r['forecast_id']) for d in trial.captures(registration) for r in d.get('selected', [])} if registration else set()
        props, exclusions['props'] = prepare_props(records, load_ledger(), fixed_ids)
    games, exclusions['games'] = load_games() if any(m in ('spread', 'total', 'game_outcome') for m in markets) else ([], {})
    rows = props+[r for r in games if r['market'] in markets]
    requested_source = None
    if workload_artifact:
        descriptor = dp.workload_descriptor(workload_artifact, workload_variant)
        if descriptor['stat'] not in markets:
            raise ValueError('workload_stat_outside_requested_markets')
        alternatives, exclusions['workload'] = workload_rows(props, descriptor)
        rows.extend(alternatives)
        requested_source = dp.workload_source(descriptor)
    if receiving_artifact:
        import hashlib
        from pathlib import Path
        path = Path(receiving_artifact).resolve()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = dp.source_artifact(str(path), sha)
        descriptor = dict(path=str(path), sha256=sha)
        from nfl_pipeline.modeling.receiving_quantile_validation import eligible_error
        lookup = {int(r['id']): r for r in records}
        eligible = [r for r in props if r['market']=='receiving_yards' and not eligible_error(lookup[int(r['id'])], artifact)]
        from nfl_pipeline.modeling.train_receiving_quantiles import curves
        for offset in range(0, len(eligible), 128):
            batch = eligible[offset:offset+128]
            values, weights = curves(artifact['bundle'], pd.DataFrame([r['captured']['row'] for r in batch]))['calibrated_blend']
            for r, v, w in zip(batch, values, weights):
                rows.append(dict(r, curve=dp.curve(v, w), source='receiving_quantile:'+sha, source_artifact=descriptor, context=None))
    # Include immutable alternative curves already scored by the live runner.
    for r in list(props)+list(games):
        variants = (r['payload'].get('distribution_policy_variants', []) if r['market'] in ('total', 'spread', 'game_outcome')
                    else r['captured']['offer'].get('distribution_policy_variants', []))
        for ctx in variants:
            policy = ctx.get('challenger')
            source = policy['source'] if policy else ctx['key'].rsplit('|', 1)[-1]
            if source != 'raw_distribution':
                rows.append(dict(r, curve=ctx['curve'], anchor=ctx['anchor'], context=ctx,
                    source=source, source_artifact=(policy or {}).get('source_artifact') or ctx.get('source_artifact')))
    # Prefer immutable captured curves to development reconstruction of the same source.
    rows = list({(r['market'], r['id'], r['source']): r for r in rows}.values())
    registry = dp.load_registry(); cohorts = {}; prospective = {}; installed = []
    if rows:
        frame = pd.DataFrame(rows)
        # A checksum-pinned challenger generates its own curves through one new
        # path; legacy final-score versions do not change that model. Raw legacy
        # curves remain separated unless equivalence was independently proven.
        frame['calibration_cohort'] = [r.scoring_version if r.source=='raw_distribution' else 'source:'+r.source
                                       for r in frame.itertuples()]
        for key, group in frame.groupby(['market', 'model_version', 'calibration_cohort', 'source', 'book']):
            if key[0] not in markets or key[4] != 'fanduel' or (requested_source and key[3] != requested_source):
                continue
            print('Forward CDF policy', key, len(group), flush=True)
            report, _ = forward_evaluate(group, rows)
            policy = fit_policy(group.loc[group.actual.notna()].copy(), datetime.now(timezone.utc))
            report['current_fit_status'] = 'fitted_for_prospective_test' if policy else 'needs_separate_fit_and_tune_weeks'
            report['current_development_fit'] = policy
            if policy and key[3].startswith(('role_workload:', 'forecast_repair:', 'stat_component:')):
                tune = group.loc[group.season.eq(policy['tuning_period'][0]) & group.week.eq(policy['tuning_period'][1])]
                tuning_rows = []
                for r in tune.to_dict('records'):
                    s = row_score(r, policy)
                    s['scored_curve'] = dp.score(r['curve'], policy, r['anchor'])
                    s['policy_ranking_support'] = policy['ranking_support']
                    tuning_rows.append(s)
                fixed = [r for r in tuning_rows if r['exact_selected']]
                report['tuning_diagnostic'] = dict(scope='used_to_select_policy_not_acceptance',
                    production=metrics(tuning_rows, 'production_probability'), challenger=metrics(tuning_rows),
                    exact_market=metrics(tuning_rows, 'market_probability'),
                    fixed_selections=dict(production=metrics(fixed, 'production_probability'), challenger=metrics(fixed),
                                          market=metrics(fixed, 'market_probability')),
                    selection_comparison=selection_comparison(tuning_rows, rows))
            registry_key = dp.policy_key(key[0], key[1], key[4], key[3])
            if policy and register and registry_key not in registry:
                now = datetime.now(timezone.utc).isoformat()
                doc = clean(dict(policy, contract=dp.CONTRACT, key=registry_key, created_at=now,
                                 training_scoring_version=key[2], source=key[3]))
                descriptor = group.iloc[0].get('source_artifact')
                if isinstance(descriptor, dict):
                    doc['source_artifact'] = descriptor
                if key[3].startswith(('role_workload:', 'forecast_repair:', 'stat_component:')):
                    doc['acceptance_scope'] = 'new_post_registration_locks_only'
                    doc['development_weeks'] = sorted(set(zip(group.season.astype(int), group.week.astype(int))))
                    doc['selection_experiment'] = dict(primary='unchanged_ranking', challenger='conservative_ranking',
                        max_per_batch=3, no_quota=True, automatic_cash_approval=False)
                doc['policy_id'] = dp.digest(doc)
                registry[registry_key] = doc
            cohorts['|'.join(key)] = report
            report['production_scoring_versions'] = sorted(group.scoring_version.unique())
        # Never compare a pinned policy with a silently refitted successor.
        for key, policy in registry.items():
            if policy['key'].split('|')[3] not in markets or (requested_source and policy['source'] != requested_source):
                continue
            matching = [r for r in rows if dp.policy_key(r['market'], r['model_version'], r['book'], r['source']) == key]
            decision = review(matching, policy, rows)
            prospective[key] = decision
            if deploy_approved and not decision['blockers']:
                dp.deploy(policy, decision); installed.append(key)
    if register:
        dp.save_registry(registry, 'challengers')
    report = clean(dict(contract=dp.CONTRACT, built_at=datetime.now(timezone.utc).isoformat(),
        markets={m: dict(eligible_locks=len({r['id'] for r in rows if r['market']==m}),
                        candidate_rows=sum(r['market']==m for r in rows)) for m in markets},
        cohorts=cohorts, prospective=prospective, exclusions=exclusions, installed=installed,
        cash_approved=False, game_outcome_status='Margin win/tie forecasts captured; no captured moneyline pair baseline, no moneyline deployment.',
        limitations=['Historical reconstruction is development evidence, never a prospective policy lock.',
            'Whole-CDF market shape is an assumption from one true paired anchor, not observed alternate-line market probabilities.',
            'Missing original game distributions are excluded, never reconstructed from current sigma.',
            'Forecast deployment does not change stake, ledger, CLV or cash authorization requirements.']))
    write_report(report, len(registry))
    print(json.dumps(dict(markets=report['markets'], exclusions=exclusions, pinned=len(registry), installed=installed), indent=2))
    return report


def write_report(report, pinned):
    for decision in report.get('prospective', {}).values():
        decision['selection_blockers'] = selection_blockers(decision)
        decision['blockers'] = deployment_blockers(decision)
    # Older generated summaries accidentally attached the challenger's range to
    # the production/market probability metrics. A price is not a distribution.
    def correct_scopes(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name in ('production', 'market', 'market_baseline') and isinstance(child, dict):
                    child.pop('distribution', None); child.pop('median', None)
                correct_scopes(child)
        elif isinstance(value, list):
            for child in value:
                correct_scopes(child)
    correct_scopes(report)
    atomic_json(ROOT/'reports/nfl_distribution_policy_latest.json', report)
    lines = ['# NFL Distribution Policy', '',
        'Forecast deployment and cash authorization are separate. No cash approval is granted here.', '',
        '| Market | Unique eligible locks | Curve rows |', '|---|---:|---:|']
    for market, counts in report['markets'].items():
        lines.append(f"| {market} | {counts['eligible_locks']} | {counts['candidate_rows']} |")
    lines += ['', f'Pinned challengers: {pinned}. Installed this run: {len(report["installed"])}.', '',
              '## Prospective Decisions', '']
    for key, decision in report['prospective'].items():
        lines.append('- '+key+': '+(', '.join(decision['blockers']) or 'forecast deployment passed'))
        if decision.get('selection_blockers'):
            lines.append('  Betting-selection evidence: '+', '.join(decision['selection_blockers'])+'. No cash approval.')
    lines += ['', '## Evidence Limits', '', *['- '+s for s in report['limitations']], '',
              'Exclusions: '+json.dumps(report['exclusions'], sort_keys=True), '']
    for key, cohort in report['cohorts'].items():
        if 'role_workload:' not in key:
            continue
        lines += ['', '## Workload Probability Path', '', key, '',
            'Historical development only; not untouched acceptance.', '',
            '| Probability path | Settled rows | Brier | Calibration error |', '|---|---:|---:|---:|']
        for label, name in [('Production', 'production'), ('Raw workload', 'raw_model_probability'),
                            ('Calibrated workload', 'calibrated_model_probability'), ('Learned blend', 'metrics'),
                            ('Exact no-vig offer', 'market')]:
            m = cohort.get(name, {})
            lines.append(f"| {label} | {m.get('rows', 0)} | {m.get('brier', '-')} | {m.get('calibration_error', '-')} |")
        lines += ['', 'Current fit: '+cohort['current_fit_status']]
        fit = cohort.get('current_development_fit')
        if fit:
            lines += [f"Model weight: {fit['model_weight']:.0%}; calibration enabled: {fit['calibration_enabled']}.",
                f"Unique player-games: calibration {fit['calibration_player_games']}, tuning {fit['tuning_player_games']}.",
                'This current-time fit is for new locks, not historical acceptance.']
        diagnostic = cohort.get('tuning_diagnostic')
        if diagnostic:
            lines += ['', '### Tuning Diagnostic (Not Acceptance)', '',
                '| Path | Brier | Calibration error | 80% coverage |', '|---|---:|---:|---:|']
            for name in ('production', 'challenger', 'exact_market'):
                m = diagnostic[name]
                lines.append(f"| {name} | {m.get('brier', '-')} | {m.get('calibration_error', '-')} | {m.get('distribution', {}).get('coverage_80', '-')} |")
            for name, value in diagnostic['selection_comparison'].items():
                if isinstance(value, dict):
                    m = value['probability']
                    lines.append(f"- Tuning {name}: {len(value['selected'])} selected; Brier {m.get('brier', '-')}; exclusions {value['exclusions']}")
        for name, value in cohort.get('selection_comparison', {}).items():
            if isinstance(value, dict):
                lines.append(f"- {name}: {len(value.get('selected', []))} selected; replay exclusions {value.get('exclusions', {})}")
    atomic_write_text(ROOT/'reports/nfl_distribution_policy_latest.md', '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markets', nargs='+', choices=dp.MARKETS, default=list(dp.MARKETS))
    parser.add_argument('--register', action='store_true', help='Pin first available challengers; never silently refit an existing trial')
    parser.add_argument(
        '--deploy-approved', '--deploy-passing',
        dest='deploy_approved', action='store_true',
        help='Forecast-only, per-market deployment after that policy independently passes prospective gates',
    )
    parser.add_argument('--receiving-artifact', help='Checksum-pinned receiving quantile challenger; historical rows remain development only')
    parser.add_argument('--workload-artifact', help='Workload curve with its own calibration, regularized blend and prospective capture')
    parser.add_argument('--workload-variant', choices=('control', 'workload_only', 'tails_only', 'combined'), default='combined')
    parser.add_argument('--season', type=int)
    parser.add_argument('--weeks', type=int, nargs='+')
    parser.add_argument('--render-existing', action='store_true', help='Refresh existing report formatting only; no fit, locks or deployment')
    args = parser.parse_args()
    if (args.season is None) != (args.weeks is None):
        parser.error('--season and --weeks must be supplied together')
    if args.render_existing:
        write_report(json.loads((ROOT/'reports/nfl_distribution_policy_latest.json').read_text(encoding='utf-8')), len(dp.load_registry()))
        return
    run(args.markets, args.register, args.deploy_approved, args.receiving_artifact,
        args.workload_artifact, args.workload_variant, args.season, args.weeks)


if __name__ == '__main__':
    main()
