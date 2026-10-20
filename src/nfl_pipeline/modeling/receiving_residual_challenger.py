"""Forward-only FanDuel receiving residual policy and paper selector.

The player model supplies an independent line probability. This module learns
how much of its disagreement with FanDuel survived on later weeks, discounts
wide player distributions, and selects only when conservative EV is positive.
It never changes production scoring or authorizes a wager.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.forecast_store import clean
from nfl_pipeline.integrity import MODEL_ROOT, atomic_json
from nfl_pipeline.modeling.evaluation import probability_metrics
from nfl_pipeline.modeling.prospective_market import locked_market
from nfl_pipeline.modeling.scoring_equivalence import cohort_for, verified_aliases

ROOT = Path(__file__).resolve().parents[3]
STORE = MODEL_ROOT / 'receiving_residual_challenger'
REPORT = ROOT / 'reports' / 'nfl_receiving_residual_challenger_latest.json'
CONTRACT = 'nfl-receiving-residual-challenger-v1'
VARIANT = 'stable_ensemble_calibrated'
TRUST_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
MIN_SUPPORT_WEEKS = 3
MIN_SUPPORT_PLAYER_GAMES = 50
MAX_PER_DAY = 5
MAX_PER_GAME = 2


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(value):
    value = np.clip(np.asarray(value, dtype=float), -30, 30)
    return 1 / (1 + np.exp(-value))


def _probability(independent, market, reliability, trust):
    gap = (_logit(independent) - _logit(market)) * np.asarray(reliability, dtype=float)
    return _sigmoid(_logit(market) + float(trust) * gap)


def _payout(price):
    value = float(price)
    return value / 100 if value > 0 else 100 / -value


def _ev(probability, price, push=0.0):
    nonpush = 1 - float(push)
    return nonpush * (float(probability) * (1 + _payout(price)) - 1)


def _weights(frame):
    return 1 / frame.groupby('player_game').player_game.transform('size')


def _loss(probability, outcome, weight):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcome, dtype=float)
    return float(np.average(-y * np.log(p) - (1 - y) * np.log1p(-p), weights=weight))


def _policy_key(run_id, release, variant=VARIANT):
    # The signal is priced directly from the pinned challenger curve. It does
    # not use any production scoring adjustment, so scoring-code revisions do
    # not fragment this evidence.
    return '|'.join(map(str, (run_id, release, variant)))


def _reliability(frame):
    width = pd.to_numeric(frame.raw_p90, errors='coerce') - pd.to_numeric(frame.raw_p10, errors='coerce')
    center = pd.to_numeric(frame.distribution_mean, errors='coerce').abs().clip(lower=10)
    uncertainty = (width / (2 * center)).clip(lower=0, upper=1).fillna(1)
    return (1 - .5 * uncertainty).clip(lower=.5, upper=1), uncertainty


def prepare(rows):
    """Return one clean, true-paired prospective row per immutable shadow."""
    frame = pd.DataFrame(rows)
    required = {
        'forecast_id', 'run_id', 'production_release', 'scoring_version', 'variant',
        'stat', 'source_book', 'source_side', 'source_price', 'market_probability',
        'independent_same_side_probability', 'outcome', 'season', 'week',
        'game_id', 'player_id', 'created_at', 'label_available_at', 'raw_p10',
        'raw_p90', 'distribution_mean',
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(), {'missing_independent_probability_capture': len(frame)}
    before = len(frame)
    frame = frame.loc[
        frame.variant.eq(VARIANT)
        & frame.stat.eq('receiving_yards')
        & frame.source_book.eq('fanduel')
        & frame.get('market_evidence', pd.Series('', index=frame.index)).eq('true_same_book_at_lock')
    ].copy()
    numeric = ('market_probability', 'independent_same_side_probability', 'outcome', 'source_price')
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.loc[
        frame.market_probability.between(0, 1, inclusive='neither')
        & frame.independent_same_side_probability.between(0, 1, inclusive='neither')
        & frame.outcome.isin((0.0, 1.0))
        & frame.source_side.isin(('over', 'under'))
        & frame.source_price.abs().ge(100)
    ].copy()
    if frame.empty:
        return frame, {'outside_clean_residual_scope': before}
    aliases = verified_aliases()
    frame['scoring_cohort'] = [cohort_for(r.scoring_version, r.forecast_id, aliases)
                               for r in frame.itertuples()]
    frame['player_game'] = frame.game_id.astype(str) + '|' + frame.player_id.astype(str)
    frame['locked_at'] = pd.to_datetime(frame.created_at, utc=True, errors='coerce')
    frame['label_at'] = pd.to_datetime(frame.label_available_at, utc=True, errors='coerce')
    frame['period'] = list(zip(frame.season.astype(int), frame.week.astype(int)))
    frame['reliability'], frame['uncertainty_proxy'] = _reliability(frame)
    frame = frame.loc[frame.locked_at.notna() & frame.label_at.notna()].copy()
    frame = frame.sort_values(['locked_at', 'forecast_id']).drop_duplicates(
        ['run_id', 'variant', 'forecast_id'], keep='first')
    return frame, {'outside_clean_residual_scope': before - len(frame)}


def _best_trust(frame):
    weight = _weights(frame)
    scores = {
        float(trust): _loss(
            _probability(frame.independent_same_side_probability, frame.market_probability,
                         frame.reliability, trust),
            frame.outcome,
            weight,
        )
        for trust in TRUST_GRID
    }
    return min(scores, key=scores.get), scores


def fit_parameters(frame):
    weeks = sorted(frame.period.unique())
    player_games = int(frame.player_game.nunique())
    point_trust, scores = _best_trust(frame)
    leave_one_week_out = []
    if len(weeks) >= MIN_SUPPORT_WEEKS:
        for period in weeks:
            training = frame.loc[frame.period.map(lambda value: value != period)]
            if training.player_game.nunique() < 20:
                continue
            trust, _ = _best_trust(training)
            leave_one_week_out.append(float(trust))
    candidate = float(np.quantile(leave_one_week_out, .10)) if len(leave_one_week_out) >= 3 else 0.0
    week_gains = []
    for period, group in frame.groupby('period'):
        weight = _weights(group)
        revised = _probability(group.independent_same_side_probability, group.market_probability,
                               group.reliability, candidate)
        market_brier = float(np.average((group.market_probability-group.outcome) ** 2, weights=weight))
        revised_brier = float(np.average((revised-group.outcome) ** 2, weights=weight))
        week_gains.append(market_brier-revised_brier)
    mean_gain = float(np.mean(week_gains)) if week_gains else None
    lower_80 = None
    if len(week_gains) >= 3:
        standard_error = float(np.std(week_gains, ddof=1) / np.sqrt(len(week_gains)))
        lower_80 = float(np.mean(week_gains) - 1.281551565545 * standard_error)
    supported = bool(
        len(weeks) >= MIN_SUPPORT_WEEKS
        and player_games >= MIN_SUPPORT_PLAYER_GAMES
        and candidate > 0
        and lower_80 is not None
        and lower_80 > 0
    )
    conservative_trust = candidate if supported else 0.0
    return clean({
        'point_trust': float(point_trust),
        'conservative_trust': float(conservative_trust),
        'conservative_supported': supported,
        'rows': len(frame),
        'player_games': player_games,
        'season_weeks': [list(value) for value in weeks],
        'leave_one_week_out_trust': leave_one_week_out,
        'candidate_conservative_trust': candidate,
        'week_brier_gains': week_gains,
        'mean_week_brier_gain': mean_gain,
        'lower_80_week_brier_gain': lower_80,
        'market_log_loss': scores[0.0],
        'best_log_loss': scores[point_trust],
        'blockers': [] if supported else [
            name for condition, name in (
                (len(weeks) < MIN_SUPPORT_WEEKS, 'needs_three_independent_weeks'),
                (player_games < MIN_SUPPORT_PLAYER_GAMES, 'needs_50_player_games'),
                (candidate <= 0, 'leave_week_out_trust_not_positive'),
                (lower_80 is None or lower_80 <= 0, 'week_clustered_brier_gain_unconfirmed'),
            ) if condition
        ],
    })


def _score(frame, parameters, prefix='conservative'):
    trust = float(parameters[prefix + '_trust'])
    probability = _probability(frame.independent_same_side_probability, frame.market_probability,
                               frame.reliability, trust)
    return pd.Series(probability, index=frame.index)


def select_frame(frame, probability_column='conservative_probability', cap=MAX_PER_DAY):
    """Outcome-blind abstaining ranker; a daily cap is never a quota."""
    if frame.empty or probability_column not in frame:
        return frame.iloc[:0].copy()
    pool = frame.copy()
    pool['selection_ev'] = [
        _ev(probability, price, push)
        for probability, price, push in zip(
            pool[probability_column], pool.source_price,
            pool.get('independent_push_probability', pd.Series(0.0, index=pool.index)).fillna(0),
        )
    ]
    if 'conservative_supported' in pool:
        pool = pool.loc[pool.conservative_supported.fillna(False)]
    pool = pool.loc[
        pool.selection_ev.gt(0)
        & pool[probability_column].gt(pool.market_probability)
    ].copy()
    selected = []
    group_columns = ['run_id', 'production_release', 'day']
    for _, day in pool.groupby(group_columns, sort=True):
        used_players = set()
        games = Counter()
        for index, row in day.sort_values(
                ['selection_ev', probability_column, 'forecast_id'],
                ascending=[False, False, True], kind='stable').iterrows():
            player_game = (row.game_id, row.player_id)
            if player_game in used_players or games[row.game_id] >= MAX_PER_GAME:
                continue
            selected.append(index)
            used_players.add(player_game)
            games[row.game_id] += 1
            if len(used_players) >= cap:
                break
    return frame.loc[selected].assign(selection_ev=pool.loc[selected, 'selection_ev'])


def _metrics(frame, probability):
    if frame.empty:
        return {'rows': 0}
    weight = _weights(frame)
    return probability_metrics(frame[probability], frame.outcome, weight)


def fit(rows):
    frame, exclusions = prepare(rows)
    if frame.empty:
        return clean({
            'contract': CONTRACT,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'waiting_for_independent_probability_captures',
            'input_rows': 0,
            'policies': {},
            'forward_folds': [],
            'exclusions': exclusions,
            'production_changed': False,
            'betting_approved': False,
        })
    policies = {}
    folds = []
    group_columns = ['run_id', 'production_release', 'variant']
    for cohort, group in frame.groupby(group_columns, sort=True):
        periods = sorted(group.period.unique())
        for test_period in periods[1:]:
            test = group.loc[group.period.map(lambda value: value == test_period)].copy()
            cutoff = test.locked_at.min()
            test_games = set(test.player_game)
            training = group.loc[
                group.period.map(lambda value: value < test_period)
                & group.label_at.lt(cutoff)
                & ~group.player_game.isin(test_games)
            ].copy()
            if training.empty:
                folds.append({'cohort': list(cohort), 'test_period': list(test_period),
                              'status': 'no_earlier_available_labels'})
                continue
            parameters = fit_parameters(training)
            test['point_probability'] = _score(test, parameters, 'point')
            test['conservative_probability'] = _score(test, parameters, 'conservative')
            test['conservative_supported'] = parameters['conservative_supported']
            test['day'] = test.locked_at.dt.date.astype(str)
            chosen = select_frame(test)
            folds.append(clean({
                'cohort': list(cohort),
                'test_period': list(test_period),
                'cutoff': cutoff.isoformat(),
                'training_rows': len(training),
                'test_rows': len(test),
                'parameters': parameters,
                'market': _metrics(test, 'market_probability'),
                'independent': _metrics(test, 'independent_same_side_probability'),
                'point_blend': _metrics(test, 'point_probability'),
                'conservative_blend': _metrics(test, 'conservative_probability'),
                'selected': _metrics(chosen, 'conservative_probability'),
                'selected_ids': chosen.forecast_id.astype(int).tolist(),
            }))
        parameters = fit_parameters(group)
        key = _policy_key(*cohort)
        policies[key] = dict(parameters, run_id=cohort[0], production_release=cohort[1],
                             variant=cohort[2],
                             source_scoring_cohorts=sorted(group.scoring_cohort.astype(str).unique()),
                             label_available_through=group.label_at.max().isoformat())
    core = clean({
        'contract': CONTRACT,
        'variant': VARIANT,
        'input_forecast_ids': sorted(frame.forecast_id.astype(int).unique().tolist()),
        'policies': policies,
    })
    policy_id = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return clean({
        **core,
        'policy_id': policy_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'conservative_signal_supported' if any(
            p['conservative_supported'] for p in policies.values()) else 'collecting_residual_evidence',
        'input_rows': len(frame),
        'player_games': int(frame.player_game.nunique()),
        'forward_folds': folds,
        'exclusions': exclusions,
        'production_changed': False,
        'deployment_approved': False,
        'betting_approved': False,
        'limitations': [
            'Only immutable true-paired FanDuel common receiving lines enter training.',
            'Each test week uses labels available before its first lock; overlapping player-games are purged.',
            'Duplicate offers receive inverse player-game weight.',
            'The independent player curve is discounted for interval uncertainty before residual trust is applied.',
            'The selector abstains unless conservative EV is positive; five is a cap, never a quota.',
            'This is a paper challenger. It cannot deploy production or authorize cash.',
        ],
    })


def load_artifact():
    path = STORE / 'latest.json'
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding='utf-8'))
    return artifact if artifact.get('contract') == CONTRACT else None


def apply_document_rows(records, rows, artifact=None):
    artifact = artifact or load_artifact()
    lookup = {int(record['id']): record for record in records}
    aliases = verified_aliases()
    output = []
    for source in rows:
        row = dict(source)
        reason = None
        if row.get('variant') != VARIANT or row.get('stat') != 'receiving_yards' or row.get('source_book') != 'fanduel':
            output.append(row)
            continue
        record = lookup.get(int(row['forecast_id']))
        if not artifact:
            reason = 'residual_policy_not_fitted'
        elif not record:
            reason = 'missing_original_forecast'
        elif row.get('market_evidence') != 'true_same_book_at_lock':
            reason = 'missing_true_paired_market'
        elif row.get('independent_same_side_probability') is None:
            reason = 'missing_independent_probability'
        cutoff = safe_time(row.get('source_context_cutoff'))
        if not reason and (not cutoff or safe_time(artifact.get('created_at')) > cutoff):
            reason = 'residual_policy_not_available_at_lock'
        scoring = cohort_for(
            str(row.get('source_scoring_version') or ''), int(row['forecast_id']), aliases)
        key = _policy_key(row.get('run_id'), row.get('production_release'), row.get('variant'))
        policy = (artifact or {}).get('policies', {}).get(key)
        if not reason and not policy:
            reason = 'no_matching_release_scoring_policy'
        if not reason and safe_time(policy.get('label_available_through')) >= cutoff:
            reason = 'policy_labels_not_strictly_before_lock'
        if reason:
            row['residual_signal'] = {'status': reason, 'policy_id': (artifact or {}).get('policy_id')}
            output.append(row)
            continue
        frame = pd.DataFrame([row])
        reliability, uncertainty = _reliability(frame)
        independent = float(row['independent_same_side_probability'])
        market = float(row['market_probability'])
        point = float(_probability(independent, market, reliability.iloc[0], policy['point_trust']))
        conservative = float(_probability(
            independent, market, reliability.iloc[0], policy['conservative_trust']))
        push = float(row.get('independent_push_probability') or 0)
        conservative_ev = _ev(conservative, row['source_price'], push)
        status = 'eligible_paper_signal' if policy['conservative_supported'] and conservative_ev > 0 and conservative > market else (
            'no_positive_conservative_ev' if policy['conservative_supported'] else 'residual_not_supported')
        row['residual_signal'] = clean({
            'status': status,
            'policy_id': artifact['policy_id'],
            'policy_key': key,
            'source_scoring_cohort': scoring,
            'independent_probability': independent,
            'market_probability': market,
            'uncertainty_proxy': float(uncertainty.iloc[0]),
            'uncertainty_reliability': float(reliability.iloc[0]),
            'point_trust': policy['point_trust'],
            'conservative_trust': policy['conservative_trust'],
            'point_probability': point,
            'conservative_probability': conservative,
            'point_ev': _ev(point, row['source_price'], push),
            'conservative_ev': conservative_ev,
            'paper_only': True,
        })
        output.append(row)
    return output


def _capture_documents():
    path = STORE / 'captures'
    return [json.loads(file.read_text(encoding='utf-8')) for file in sorted(path.glob('*.json'))]


def capture(records, document):
    """Freeze conservative paper selections without touching the wager ledger."""
    artifact = load_artifact()
    if not artifact:
        return {'status': 'waiting_for_residual_policy', 'selected': 0, 'paper_only': True}
    lookup = {int(record['id']): record for record in records}
    now = safe_time(document.get('scored_at')) or datetime.now(timezone.utc)
    candidates = []
    exclusions = Counter()
    for row in document.get('rows', []):
        signal = row.get('residual_signal') or {}
        if signal.get('status') != 'eligible_paper_signal':
            if row.get('variant') == VARIANT:
                exclusions[signal.get('status') or 'missing_residual_signal'] += 1
            continue
        record = lookup.get(int(row['forecast_id']))
        if not record:
            exclusions['missing_original_forecast'] += 1; continue
        start = safe_time(record.get('start_ts_utc'))
        quote = ((record.get('forecast_payload') or {}).get('scoring_replay') or {}).get('offer') or {}
        fetched = safe_time(quote.get('fetched_at_utc'))
        link = urlparse(str(quote.get(str(row['source_side']) + '_link') or ''))
        if not start or now >= start:
            exclusions['not_pregame'] += 1; continue
        if not fetched or not 0 <= (now-fetched).total_seconds() <= 1200:
            exclusions['quote_not_fresh'] += 1; continue
        if locked_market(record)['market_probability'] is None:
            exclusions['missing_exact_true_pair'] += 1; continue
        if link.scheme != 'https' or not (link.hostname == 'fanduel.com' or (link.hostname or '').endswith('.fanduel.com')):
            exclusions['missing_valid_fanduel_link'] += 1; continue
        p = record['forecast_payload']
        candidates.append(clean({
            'forecast_id': int(row['forecast_id']),
            'policy_id': signal['policy_id'],
            'game_id': record['game_id'],
            'player_id': record['player_id'],
            'day': str(p['game_date_et'])[:10],
            'side': row['source_side'],
            'line': row['source_line'],
            'price': row['source_price'],
            'probability': signal['conservative_probability'],
            'market_probability': signal['market_probability'],
            'conservative_ev': signal['conservative_ev'],
            'captured_at': now.isoformat(),
            'paper_only': True,
        }))
    STORE.mkdir(parents=True, exist_ok=True)
    lock = STORE / '.capture.lock'
    owned = False
    try:
        with lock.open('x'):
            owned = True
            previous = [row for doc in _capture_documents() for row in doc.get('selected', [])]
            used_ids = {int(row['forecast_id']) for row in previous}
            daily = Counter((row['policy_id'], row['day']) for row in previous)
            games = Counter((row['policy_id'], row['day'], row['game_id']) for row in previous)
            selected = []
            for row in sorted(candidates, key=lambda value: (-value['conservative_ev'], value['forecast_id'])):
                day_key = (row['policy_id'], row['day'])
                game_key = (*day_key, row['game_id'])
                if row['forecast_id'] in used_ids or daily[day_key] >= MAX_PER_DAY or games[game_key] >= MAX_PER_GAME:
                    exclusions['cap_or_duplicate'] += 1; continue
                selected.append(row)
                used_ids.add(row['forecast_id'])
                daily[day_key] += 1
                games[game_key] += 1
            payload = clean({
                'contract': CONTRACT,
                'policy_id': artifact['policy_id'],
                'captured_at': now.isoformat(),
                'selected': selected,
                'exclusions': dict(exclusions),
                'paper_only': True,
            })
            atomic_json(STORE/'captures'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json'), payload)
    finally:
        if owned:
            lock.unlink()
    return {'status': 'paper_residual_trial', 'selected': len(selected),
            'exclusions': dict(exclusions), 'paper_only': True}


def fit_and_write(rows):
    artifact = fit(rows)
    atomic_json(REPORT, artifact)
    if artifact.get('policies'):
        atomic_json(STORE/'latest.json', artifact)
    lines = [
        '# FanDuel Receiving Residual Challenger', '',
        'Status: ' + artifact['status'],
        f"Independent captured rows: {artifact.get('input_rows', 0)}; player-games: {artifact.get('player_games', 0)}", '',
        '| Policy cohort | Weeks | Player-games | Point trust | Conservative trust | Supported |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for key, policy in artifact.get('policies', {}).items():
        lines.append(f"| {key} | {len(policy['season_weeks'])} | {policy['player_games']} | "
                     f"{policy['point_trust']:.2f} | {policy['conservative_trust']:.2f} | "
                     f"{policy['conservative_supported']} |")
    lines += ['', 'Paper-only. Production and bankroll approval are unchanged.', '',
              *artifact.get('limitations', [])]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return artifact
