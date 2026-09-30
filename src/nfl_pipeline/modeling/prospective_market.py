"""Same-book market evidence from immutable inputs, never today's replacement quote."""
from collections import Counter

import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.modeling.evaluation import clustered_gain, probability_metrics


def implied(price):
    price = float(price)
    if not np.isfinite(price) or abs(price) < 100:
        raise ValueError('invalid_american_price')
    return 100 / (100 + price) if price > 0 else -price / (100 - price)


def locked_market(record):
    p = record.get('forecast_payload') or {}
    q = (p.get('scoring_replay') or {}).get('offer') or {}
    def unknown(reason):
        return dict(market_probability=None, market_evidence=reason)
    if not q:
        return unknown('missing_captured_quote')
    if q.get('pair_quality') in ('synthetic', 'cross_book', 'one_sided'):
        return unknown('not_true_same_book_pair')
    try:
        if (not p.get('offer_id') or str(q.get('offer_id')) != str(p['offer_id'])
                or q.get('bookmaker_key') != p.get('book') or q.get('stat') != record['stat']
                or not p.get('offer_player_name_norm')
                or q.get('player_name_norm') != p['offer_player_name_norm']
                or float(q['line']) != float(p['line']) or p.get('side') not in ('over', 'under')):
            return unknown('captured_quote_identity_mismatch')
        fetched = safe_time(q.get('fetched_at_utc'))
        cutoff = safe_time(p.get('prediction_context_cutoff_utc'))
        lock = safe_time(record.get('created_at_utc'))
        start = safe_time(record.get('start_ts_utc'))
        if not all((fetched, cutoff, lock, start)) or not fetched <= cutoff <= lock < start:
            return unknown('captured_quote_timing_invalid')
        quote_start = safe_time(q.get('commence_time_utc'))
        if not q.get('event_id') or quote_start != start:
            return unknown('captured_event_mismatch')
        over, under = implied(q.get('over_price')), implied(q.get('under_price'))
        if float(q[p['side']+'_price']) != float(p['price']):
            return unknown('captured_side_price_mismatch')
        probability = (over if p['side'] == 'over' else under) / (over + under)
    except (KeyError, TypeError, ValueError):
        return unknown('missing_or_invalid_paired_price')
    return dict(market_probability=float(probability), market_evidence='true_same_book_at_lock')


def market_comparison(frame):
    """All three probabilities are evaluated on precisely the same original decisions."""
    p = pd.to_numeric(frame.get('market_probability', pd.Series(np.nan, index=frame.index)), errors='coerce')
    g = frame.loc[p.between(0, 1)].copy()
    reasons = Counter(frame.loc[~p.between(0, 1)].get('market_evidence',
                      pd.Series('missing_captured_quote', index=frame.index)).dropna())
    result = dict(rows=len(g), eligible_rows=len(frame), pair_coverage=len(g)/len(frame) if len(frame) else None,
                  exclusions=dict(reasons))
    if g.empty:
        return dict(result, status='waiting_for_true_paired_matches')
    w = 1/g.groupby(['game_id', 'player_id']).outcome.transform('size')
    result.update(status='matched', unique_player_games=len(g[['game_id', 'player_id']].drop_duplicates()),
                  weeks=len(g[['season', 'week']].drop_duplicates()))
    for label, column in (('production', 'production_probability'), ('challenger', 'same_side_probability'),
                          ('market', 'market_probability')):
        result[label] = probability_metrics(g[column], g.outcome, w)
    result['weekly'] = []
    for (season, week), week_rows in g.groupby(['season', 'week']):
        result['weekly'].append(dict(season=int(season), week=int(week), rows=len(week_rows),
            **{label: probability_metrics(week_rows[column], week_rows.outcome, w.loc[week_rows.index])
               for label, column in (('production', 'production_probability'),
                    ('challenger', 'same_side_probability'), ('market', 'market_probability'))}))
    for label, column in (('production', 'production_probability'), ('challenger', 'same_side_probability')):
        errors = g.assign(reference_error=(g.market_probability-g.outcome)**2,
                          challenger_error=(g[column]-g.outcome)**2)
        errors = errors.groupby(['season', 'week', 'game_id', 'player_id'])[
            ['reference_error', 'challenger_error']].mean().reset_index()
        result[label+'_brier_gain_vs_market'] = clustered_gain(errors)
    return result
