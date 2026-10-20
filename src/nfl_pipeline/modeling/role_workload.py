"""Count-first workload curves with held-out, role-specific tails.

The efficiency center is supplied by the conservative forecast. This module
changes opportunity and uncertainty, never learns a new yards-per-opportunity
head, and never reads odds or current-game outcomes during prediction.
"""
import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.modeling.challenger_models import FittedHead, numeric
from nfl_pipeline.modeling.receiving_quantiles import QUANTILES
from nfl_pipeline.modeling.role_context_training import CONTRACT as ROLE_CONTRACT, OUT, LIMITED

CONTRACT = 'nfl-role-workload-distribution-v1'
STATS = {'receiving_yards': 'targets', 'rushing_yards': 'carries', 'passing_yards': 'pass_attempts'}
VARIANTS = ('control', 'workload_only', 'tails_only', 'combined')
GRID = (np.arange(101) + .5) / 101
ERROR_GRID = (np.arange(41) + .5) / 41
PARAMS = dict(n_estimators=100, max_depth=3, num_leaves=7,
              min_child_samples=150, reg_lambda=40, learning_rate=.035)


def verified_context(value):
    if not isinstance(value, dict) or value.get('contract') != ROLE_CONTRACT:
        return {}
    cutoff = safe_time(value.get('cutoff'))
    sources = value.get('provenance') or {}
    if not cutoff or not sources or value.get('status') != 'asof_observed':
        return {}
    for source in sources.values():
        for item in source if isinstance(source, list) else [source]:
            at = safe_time(item.get('observed_at'))
            effective = safe_time(item.get('source_snapshot'))
            if not at or at > cutoff or (effective and effective > cutoff):
                return {}
    return value


def features(frame, stat):
    op = STATS[stat]
    bases = (op, 'offense_snap_share')
    if op == 'targets':
        bases += ('target_share',)
    names = [f'{base}_{agg}_{window}' for base in bases
             for agg in ('avg', 'std') for window in (3, 5, 10)]
    names += ['is_home', 'rest_days', 'n_games_prev_5', 'n_games_prev_10']
    out = pd.DataFrame({c: numeric(frame, c) for c in names}, index=frame.index)
    out['volume_trend'] = numeric(frame, f'{op}_avg_3') - numeric(frame, f'{op}_avg_10')
    out['volume_volatility'] = numeric(frame, f'{op}_std_5') / (numeric(frame, f'{op}_avg_5') + 1)
    for position in ('WR', 'TE', 'RB', 'QB'):
        out['position_' + position] = frame.position.eq(position).astype(float)
    context = frame.get('target_role_evidence', pd.Series(None, index=frame.index, dtype=object)).map(verified_context)
    for key in ('depth_rank', 'depth_movement', 'expected_starter', 'teammate_absences', 'teammate_limited'):
        out['role_' + key] = pd.to_numeric(context.map(lambda r: r.get(key)), errors='coerce')
    for key in ('injury_status', 'practice_status'):
        for label in ('out', 'limited'):
            out[f'role_{key}_{label}'] = context.map(lambda r: np.nan if r.get(key) is None else float(
                str(r[key]).lower() in OUT if label == 'out' else any(t in str(r[key]).lower() for t in LIMITED)))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def roles(frame, stat, position=False):
    prior = numeric(frame, STATS[stat] + '_avg_5')
    threshold = {'receiving_yards': 4, 'rushing_yards': 8, 'passing_yards': 25}[stat]
    result = pd.Series(np.where(prior.isna(), 'unknown', np.where(prior >= threshold, 'higher', 'lower')), index=frame.index)
    return frame.position.fillna('unknown').astype(str) + '|' + result if position else result


def validate(frame, stat):
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Workload training requires unique player-games')
    if frame[['game_id', 'player_id', 'game_date_et']].isna().any().any():
        raise ValueError('Missing player-game identity or date')
    count = numeric(frame, STATS[stat]).to_numpy()
    yards = numeric(frame, stat).to_numpy()
    if not len(count) or not np.isfinite(count).all() or not np.isfinite(yards).all():
        raise ValueError('Workload training requires observed counts and yards')
    if (count < 0).any() or not np.allclose(count, np.rint(count)):
        raise ValueError('Opportunity counts must be nonnegative integers')
    if ((count == 0) & (yards != 0)).any():
        raise ValueError('Nonzero yards with zero opportunity need source repair')
    return count, yards


def count_crps(values, weights, actual):
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    # E|X-y| - 0.5 E|X-X'|, with the pairwise term computed in linear space.
    before = np.cumsum(w, axis=1) - w
    pair = (w * v * (2 * before + w - 1)).sum(axis=1)
    return (w * abs(v - np.asarray(actual)[:, None])).sum(axis=1) - pair


class RoleWorkload:
    def fit(self, frame, stat='receiving_yards'):
        count, _ = validate(frame, stat)
        self.stat = stat
        self.fit_end = str(frame.game_date_et.max())
        self.seen_games = set(frame.game_id)
        self.last_end = self.fit_end
        self.phase = 'fit'
        self.position_prior = frame.assign(_count=count).groupby('position')._count.mean().to_dict()
        self.global_prior = float(count.mean())
        X = features(frame, stat)
        self.support = {c: dict(nonmissing=int(X[c].notna().sum()), distinct=int(X[c].nunique())) for c in X}
        self.columns = [c for c in X if X[c].notna().sum() >= 100 and X[c].nunique() > 1]
        X = self.inputs(frame)
        self.heads = []
        for q in QUANTILES:
            head = FittedHead('quantile', params={**PARAMS, 'alpha': float(q)}).fit(X, count)
            head.constant = float(np.quantile(count, q))
            self.heads.append(head)
        self.alpha = 0.
        return self

    def inputs(self, frame):
        X = features(frame, self.stat).reindex(columns=self.columns)
        return pd.concat([X, X.isna().astype(float).add_suffix('__missing')], axis=1)

    def advance(self, frame, expected, next_phase):
        validate(frame, self.stat)
        if self.phase != expected:
            raise ValueError('Workload calibration phases must run once in order')
        if str(frame.game_date_et.min()) <= self.last_end or self.seen_games.intersection(frame.game_id):
            raise ValueError('Workload calibration must be later with no overlapping games')
        self.last_end = str(frame.game_date_et.max())
        self.seen_games.update(frame.game_id)
        self.phase = next_phase

    def baseline(self, frame):
        prior = frame.position.map(self.position_prior).fillna(self.global_prior)
        return numeric(frame, STATS[self.stat] + '_avg_5').fillna(prior).clip(lower=0).to_numpy()

    def raw_knots(self, frame):
        X = self.inputs(frame)
        return np.sort(np.column_stack([head.predict(X) for head in self.heads]), axis=1)

    def fit_count_tails(self, frame, prior_rows=200):
        if prior_rows <= 0:
            raise ValueError('Role shrinkage needs a positive prior')
        self.advance(frame, 'fit', 'count_tails')
        self.prior_rows = prior_rows
        y = numeric(frame, STATS[self.stat]).to_numpy()
        raw = self.raw_knots(frame)
        errors = y[:, None] - raw
        self.corrections = {'all': np.array([np.quantile(errors[:, j], q) for j, q in enumerate(QUANTILES)])}
        self.count_residuals = {'all': np.quantile(y - self.baseline(frame), GRID)}
        self.role_support = {}
        for level in (False, True):
            keys = roles(frame, self.stat, level)
            for key in sorted(keys.unique()):
                mask = keys.eq(key).to_numpy()
                parent = key.rsplit('|', 1)[-1] if level else 'all'
                n = int(mask.sum())
                a = n / (n + prior_rows)
                correction = np.array([np.quantile(errors[mask, j], q) for j, q in enumerate(QUANTILES)])
                self.corrections[key] = a * correction + (1-a) * self.corrections[parent]
                residual = np.quantile((y-self.baseline(frame))[mask], GRID)
                self.count_residuals[key] = a * residual + (1-a) * self.count_residuals[parent]
                self.role_support[key] = n
        return self

    def _lookup(self, table, frame, role_specific):
        if not role_specific:
            return np.tile(table['all'], (len(frame), 1))
        keys = roles(frame, self.stat, True)
        return np.array([table.get(k, table.get(k.rsplit('|', 1)[-1], table['all'])) for k in keys])

    def count_curve(self, frame, learned=False, role_specific=False):
        base = self.baseline(frame)
        residuals = self._lookup(self.count_residuals, frame, role_specific)
        control = np.rint(np.maximum(0, base[:, None] + residuals))
        if not learned:
            return control, np.full_like(control, 1 / len(GRID))
        knots = np.sort(self.raw_knots(frame) + self._lookup(self.corrections, frame, role_specific), axis=1)
        candidate = np.rint(np.maximum(0, np.array([np.interp(GRID, QUANTILES, r) for r in knots])))
        values = np.concatenate([control, candidate], axis=1)
        weights = np.tile(np.r_[np.full(len(GRID), (1-self.alpha)/len(GRID)),
                                np.full(len(GRID), self.alpha/len(GRID))], (len(frame), 1))
        return values, weights

    def fit_efficiency_tails(self, frame, reference, rates=None):
        expected = 'count_weight' if self.phase == 'count_weight' else 'count_tails'
        self.advance(frame, expected, 'efficiency_tails')
        counts, yards = validate(frame, self.stat)
        rates = (reference.predict(frame) / np.maximum(.5, self.baseline(frame))
                 if rates is None else np.asarray(rates, dtype=float))
        if rates.shape != counts.shape or not np.isfinite(rates).all():
            raise ValueError('Efficiency tails require finite per-row rates')
        positive = counts > 0
        if not positive.any():
            raise ValueError('Efficiency tails require positive observed opportunity')
        error = (yards[positive] - counts[positive]*rates[positive]) / np.sqrt(counts[positive])
        table = {'all': np.quantile(error, ERROR_GRID)}
        subset = frame.loc[positive]
        self.efficiency_support = {}
        for level in (False, True):
            keys = roles(subset, self.stat, level)
            for key in sorted(keys.unique()):
                use = keys.eq(key).to_numpy()
                parent = key.rsplit('|', 1)[-1] if level else 'all'
                n = int(use.sum())
                a = n/(n+self.prior_rows)
                table[key] = a*np.quantile(error[use], ERROR_GRID) + (1-a)*table[parent]
                self.efficiency_support[key] = n
        # Tails may change dispersion/asymmetry, not the fixed efficiency mean.
        self.efficiency_tails = {k: v-v.mean() for k, v in table.items()}
        return self

    def tune(self, frame):
        self.advance(frame, 'efficiency_tails', 'tuned')
        return self._choose_alpha(frame)

    def fit_count_weight(self, frame):
        """Count blend chosen before yard residual/CDF calibration is observed."""
        self.advance(frame, 'count_tails', 'count_weight')
        self.count_tuning_end = str(frame.game_date_et.max())
        return self._choose_alpha(frame)

    def finish_count_first(self):
        if self.phase != 'efficiency_tails' or not getattr(self, 'count_tuning_end', None):
            raise ValueError('Count-first fitting requires earlier count tuning and later efficiency tails')
        self.phase = 'tuned'
        return self

    def _choose_alpha(self, frame):
        actual = numeric(frame, STATS[self.stat]).to_numpy()
        self.tuning = {}
        for alpha in (0., .25, .5, 1.):
            self.alpha = alpha
            self.tuning[alpha] = float(np.mean(count_crps(*self.count_curve(frame, True, True), actual)))
        self.alpha = min(self.tuning, key=lambda a: (self.tuning[a], a))
        return self

    def yardage_curve(self, frame, reference=None, variant='combined', projection=None, rates=None):
        if variant not in VARIANTS:
            raise ValueError('Unknown workload variant')
        if self.phase != 'tuned':
            raise ValueError('Workload curve needs completed chronological fitting')
        if not frame.position.isin(self.position_prior).all():
            raise ValueError('Position has no workload training support')
        point = np.asarray(reference.predict(frame) if projection is None else projection, dtype=float)
        if point.shape != (len(frame),) or not np.isfinite(point).all():
            raise ValueError('Missing conservative point forecast')
        role_specific = variant in ('tails_only', 'combined')
        values, mass = self.count_curve(frame, variant in ('workload_only', 'combined'), role_specific)
        rate = point/np.maximum(.5, self.baseline(frame)) if rates is None else np.asarray(rates, dtype=float)
        if rate.shape != point.shape or not np.isfinite(rate).all():
            raise ValueError('Workload curves require finite per-row rates')
        errors = self._lookup(self.efficiency_tails, frame, role_specific)
        yards = values[:, :, None]*rate[:, None, None] + np.sqrt(values[:, :, None])*errors[:, None, :]
        weights = np.broadcast_to(mass[:, :, None]/len(ERROR_GRID), yards.shape).copy()
        base = self.baseline(frame)
        lower = np.maximum(1., .5*base)[:, None]
        upper = np.maximum({'receiving_yards': 8., 'rushing_yards': 18., 'passing_yards': 38.}[self.stat], 1.4*base)[:, None]
        state = np.where(values < lower, 0, np.where(values >= upper, 2, 1))
        states = np.column_stack([(mass*(state == s)).sum(axis=1) for s in range(3)])
        return (yards.reshape(len(frame), -1), weights.reshape(len(frame), -1),
                dict(opportunity=(values*mass).sum(axis=1), fixed_rate=rate, state_probabilities=states))
