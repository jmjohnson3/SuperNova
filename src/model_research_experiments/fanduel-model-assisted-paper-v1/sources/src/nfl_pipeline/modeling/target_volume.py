"""Small target-volume ablation with shared, unchanged receiving efficiency."""
import numpy as np
import pandas as pd

from nfl_pipeline.context_contract import number
from nfl_pipeline.modeling.challenger_models import FittedHead, numeric
from nfl_pipeline.modeling.role_context_training import OUT, LIMITED

CONTRACT = 'nfl-target-volume-only-v1'
LAGS = ('targets_avg_3', 'targets_avg_5', 'targets_avg_10', 'targets_std_5',
    'target_share_avg_3', 'target_share_avg_5', 'target_share_avg_10',
    'offense_snap_share_avg_3', 'offense_snap_share_avg_5', 'offense_snap_share_avg_10',
    'offense_snap_share_std_5', 'route_participation_proxy_avg_5', 'rest_days', 'is_home')
PARAMS = dict(n_estimators=45, max_depth=2, num_leaves=4, min_child_samples=150,
              reg_lambda=80, learning_rate=.03)


def target_features(frame):
    X = pd.DataFrame({c: numeric(frame, c) for c in LAGS}, index=frame.index)
    for pos in ('WR', 'TE', 'RB'):
        X['position_' + pos] = frame.position.eq(pos).astype(float)
    contexts = frame.get('target_role_evidence', pd.Series(None, index=frame.index, dtype=object))
    for key in ('depth_rank', 'depth_movement', 'expected_starter', 'teammate_absences', 'teammate_limited'):
        X['role_' + key] = contexts.map(lambda r: number(r.get(key)) if isinstance(r, dict) else np.nan)
    for key in ('injury_status', 'practice_status'):
        for label, tags in (('out', OUT), ('limited', LIMITED)):
            X[f'role_{key}_{label}'] = contexts.map(lambda r: float(
                str(r[key]).lower() in tags if label == 'out' else any(t in str(r[key]).lower() for t in tags))
                if isinstance(r, dict) and r.get(key) is not None else np.nan)
    X['target_trend'] = X.targets_avg_3-X.targets_avg_10
    return X.replace([np.inf, -np.inf], np.nan)


def target_states(actual, prior):
    actual, prior = np.asarray(actual), np.asarray(prior)
    return np.where(actual < np.maximum(1., .5*prior), 0,
        np.where(actual >= np.maximum(8., 1.4*prior), 2, 1))


class TargetVolume:
    def fit(self, frame):
        if frame.duplicated(['game_id', 'player_id']).any() or numeric(frame, 'targets').isna().any():
            raise ValueError('Target training requires unique player-games with observed targets')
        y = numeric(frame, 'targets').to_numpy()
        self.position_prior = frame.assign(_y=y).groupby('position')._y.mean().to_dict()
        self.global_prior = float(y.mean())
        self.team_prior = float(numeric(frame.drop_duplicates(['game_id', 'team_abbr']),
            'team_actual_pass_attempts').dropna().mean())
        if not np.isfinite(self.team_prior) or self.team_prior <= 0:
            raise ValueError('No historical team passing volume')
        X = target_features(frame)
        # An empty or constant context field cannot teach an availability effect.
        self.support = {c: dict(nonmissing=int(X[c].notna().sum()), distinct=int(X[c].nunique())) for c in X}
        self.columns = [c for c in X if self.support[c]['nonmissing'] >= 100 and self.support[c]['distinct'] > 1]
        self.state_prior = np.bincount(target_states(y, self.baseline(frame)), minlength=3)/len(y)
        self.count_head = FittedHead('poisson', params=PARAMS).fit(self.features(frame), y)
        self.state_head = FittedHead(classifier=True, params=PARAMS).fit(
            self.features(frame), target_states(y, self.baseline(frame)))
        self.alpha = 0.
        self.offsets = None
        return self

    def features(self, frame):
        X = target_features(frame).reindex(columns=self.columns)
        return pd.concat([X, X.isna().astype(float).add_suffix('__missing')], axis=1)

    def baseline(self, frame, variant='control'):
        fallback = frame.position.map(self.position_prior).fillna(self.global_prior)
        if variant == 'long_role':
            return numeric(frame, 'targets_avg_10').fillna(fallback).clip(lower=0).to_numpy()
        if variant == 'recent_share':
            long_share = numeric(frame, 'target_share_avg_10').where(lambda x: x > 0)
            team = (numeric(frame, 'targets_avg_10')/long_share).clip(15, 65).fillna(self.team_prior)
            return (team*numeric(frame, 'target_share_avg_3').clip(0, 1)).fillna(
                numeric(frame, 'targets_avg_3')).fillna(fallback).clip(lower=0).to_numpy()
        return numeric(frame, 'targets_avg_5').fillna(fallback).clip(lower=0).to_numpy()

    def tune(self, frame):
        y = numeric(frame, 'targets').to_numpy(); base = self.baseline(frame)
        states = target_states(y, base); onehot = np.eye(3)[states]
        self.tuning = {}
        for alpha in (0., .25, .5):
            self.alpha = alpha
            values, weights, probs = self.target_curve(frame)
            mean = (values*weights).sum(axis=1)
            self.tuning[alpha] = float(np.mean((mean-y)**2)/max(1., np.var(y))
                                      + np.mean(np.sum((probs-onehot)**2, axis=1)))
        self.alpha = min(self.tuning, key=self.tuning.get)
        return self

    def fit_residuals(self, frame, reference):
        y = numeric(frame, 'targets').to_numpy(); base = self.baseline(frame)
        state = target_states(y, base)
        grid = np.linspace(.05, .95, 9)
        self.offsets = np.stack([np.quantile((y-base)[state == s], grid)
            if (state == s).sum() >= 20 else np.quantile(y-base, grid) for s in range(3)])
        # Identical efficiency and residual evidence are used by every target variant.
        rate = reference.predict(frame)/np.maximum(.5, base)
        valid = (y > 0) & numeric(frame, 'receiving_yards').notna().to_numpy()
        errors = (numeric(frame, 'receiving_yards').to_numpy()[valid]-rate[valid]*y[valid])/np.sqrt(y[valid])
        self.efficiency_errors = np.quantile(errors, np.linspace(.025, .975, 21))
        return self

    def target_curve(self, frame, variant='challenger', baseline_targets=None):
        base = self.baseline(frame, variant) if baseline_targets is None else np.asarray(baseline_targets)
        alpha = self.alpha if variant == 'challenger' else 0.
        probabilities = np.broadcast_to(self.state_prior, (len(frame), 3)).copy()
        mean = base
        if alpha:
            X = self.features(frame)
            probabilities = (1-alpha)*probabilities + alpha*self.state_head.probabilities(X, 3)
            mean = (1-alpha)*base + alpha*self.count_head.predict(X).clip(0)
        values = (base[:, None, None]+self.offsets[None, :, :]).clip(0)
        weights = np.broadcast_to(probabilities[:, :, None]/values.shape[2], values.shape).copy()
        original_mean = (values*weights).sum(axis=(1, 2))
        values = (values + (mean-original_mean)[:, None, None]).clip(0)
        return values.reshape(len(frame), -1), weights.reshape(len(frame), -1), probabilities

    def yardage_curve(self, frame, reference, variant='challenger', projection=None, baseline_targets=None):
        targets, mass, states = self.target_curve(frame, variant, baseline_targets)
        base = self.baseline(frame) if baseline_targets is None else np.asarray(baseline_targets)
        point = reference.predict(frame) if projection is None else np.asarray(projection)
        if not np.isfinite(point).all() or not np.isfinite(base).all() or (base < 0).any():
            raise ValueError('Missing frozen projection or original workload input')
        rate = point/np.maximum(.5, base)
        values = targets[:, :, None]*rate[:, None, None] + np.sqrt(targets[:, :, None])*self.efficiency_errors
        weights = np.broadcast_to(mass[:, :, None]/len(self.efficiency_errors), values.shape)
        return values.reshape(len(frame), -1), weights.reshape(len(frame), -1), dict(
            targets=(targets*mass).sum(axis=1), rate=rate, state_probabilities=states)
