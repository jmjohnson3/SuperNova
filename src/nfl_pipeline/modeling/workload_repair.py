"""Conditional opportunity/efficiency and asymmetric uncertainty challengers.

Separate classes preserve the semantics of already-pinned workload artifacts.
Only earlier player-game outcomes may be used to fit these models.
"""
import numpy as np
import pandas as pd

from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual, FittedHead, numeric, player_features)
from nfl_pipeline.modeling.workload_depth_model import (
    WorkloadDepthModel, STATS, PARAMS, inputs, states)

CONTRACT = 'workload-repair-v3-conditional-efficiency-asymmetric'


def repair_inputs(frame, stat, kind):
    """Actual targets, current yards and offer prices are never model inputs."""
    base = inputs(frame, kind, stat)
    op = STATS[stat]
    names = [f'{op}_{suffix}_{n}' for suffix in ('avg', 'std') for n in (3, 5, 10)]
    if kind == 'rate':
        names += [f'opp_allowed_{stat}_avg_5', f'{stat}_std_5']
    extra = pd.DataFrame({k: numeric(frame, k) for k in names}, index=frame.index)
    extra['role_volatility'] = numeric(frame, f'{op}_std_5') / (numeric(frame, f'{op}_avg_5') + 1)
    extra['role_trend'] = numeric(frame, f'{op}_avg_3') - numeric(frame, f'{op}_avg_10')
    extra = extra.replace([np.inf, -np.inf], np.nan)
    return pd.concat([base, extra, extra.isna().astype(float).add_suffix('__missing')], axis=1)


class WorkloadRepairModel(WorkloadDepthModel):
    """Team-bounded state volumes with exposure-shrunk state-specific skill."""
    def fit(self, frame, stat):
        if frame.duplicated(['game_id', 'player_id']).any():
            raise ValueError('Workload repair requires unique player-games')
        if numeric(frame, STATS[stat]).isna().any():
            raise ValueError('Missing opportunity is not zero workload')
        super().fit(frame, stat)
        state = states(frame, stat)
        exposure = numeric(frame, STATS[stat]).to_numpy()
        actual = numeric(frame, stat).to_numpy()
        X = repair_inputs(frame, stat, 'workload')
        Xr = repair_inputs(frame, stat, 'rate')
        self.repaired_state = FittedHead(classifier=True, params=PARAMS).fit(X, state)
        self.direct_volume = []
        self.conditional_rate = []
        self.rate_support = []
        # A long-run player/position prior anchors every state, not a single-game YPC/YPT.
        weight = self.rate_weight
        self.rate_weight = 0.
        prior = WorkloadDepthModel.components(self, frame)['rate']
        self.rate_weight = weight
        for s in range(3):
            volume_mask = state == s
            if volume_mask.sum() < 80:
                volume_mask = np.ones(len(frame), dtype=bool)
            self.direct_volume.append(FittedHead('poisson', params=PARAMS).fit(
                X.loc[volume_mask], exposure[volume_mask]))
            mask = (state == s) & (exposure > 0) & np.isfinite(actual)
            if mask.sum() < 80:
                mask = (exposure > 0) & np.isfinite(actual)
            weights = np.minimum(exposure[mask], 40.)
            self.conditional_rate.append(FittedHead(params=PARAMS).fit(
                Xr.loc[mask], actual[mask] / exposure[mask] - prior[mask], weights))
            self.rate_support.append(float(weights.sum() / (weights.sum() + 500.)))
        return self

    def components(self, frame):
        weight = self.rate_weight
        self.rate_weight = 0.
        try:
            base = WorkloadDepthModel.components(self, frame)
        finally:
            self.rate_weight = weight
        X = repair_inputs(frame, self.stat, 'workload')
        Xr = repair_inputs(frame, self.stat, 'rate')
        weights = .95 * self.repaired_state.probabilities(X, 3) + .05 * self.state_prior
        direct = np.column_stack([head.predict(X) for head in self.direct_volume]).clip(0)
        # Both estimates are learned on earlier rows; no normalization over today's offers.
        opportunity = np.minimum(.5 * base['opportunities'] + .5 * direct, base['team_volume'][:, None])
        # Avoid inverted low/high workload paths without consulting the actual outcome.
        opportunity = np.sort(opportunity, axis=1)
        rates = np.column_stack([base['rate'] + weight * support * head.predict(Xr)
                                 for head, support in zip(self.conditional_rate, self.rate_support)])
        rates = rates.clip(0)
        unavailable = numeric(frame, 'wd_injury_out').eq(1).to_numpy()
        opportunity[unavailable] = 0.
        weights[unavailable] = [1., 0., 0.]
        centers = opportunity * rates
        expected_op = (opportunity * weights).sum(axis=1)
        effective_rate = np.divide((centers * weights).sum(axis=1), expected_op,
                                   out=np.zeros(len(frame)), where=expected_op > 0)
        return dict(base, centers=centers, weights=weights, opportunities=opportunity,
                    rates=rates, rate=effective_rate)


class AsymmetricWorkloadResidual(ConditionalResidual):
    """Learn lower and upper error scales separately on held-out player-games."""
    def fit_scale(self, X, errors, center, floor, confidence_errors=None):
        super().fit_scale(X, errors, center, floor, confidence_errors)
        errors = np.asarray(errors, dtype=float)
        self.lower = FittedHead('quantile', params={**PARAMS, 'alpha': .1}).fit(X, errors)
        self.upper = FittedHead('quantile', params={**PARAMS, 'alpha': .9}).fit(X, errors)
        self.global_lower = max(floor, float(-np.quantile(errors, .1)))
        self.global_upper = max(floor, float(np.quantile(errors, .9)))
        return self

    def scales(self, X):
        lo = np.maximum(self.floor, .75 * -self.lower.predict(X) + .25 * self.global_lower)
        hi = np.maximum(self.floor, .75 * self.upper.predict(X) + .25 * self.global_upper)
        return lo, hi

    def fit_residuals(self, X, errors, states=None):
        errors = np.asarray(errors, dtype=float)
        lo, hi = self.scales(X)
        z = errors / np.where(errors < 0, lo, hi)
        grid = np.linspace(.005, .995, 101)
        self.quantiles = {-1: np.quantile(z, grid)}
        if states is not None:
            for state in range(3):
                values = z[np.asarray(states) == state]
                if len(values) >= 60:
                    # Small state samples shrink to the broader error curve.
                    n = len(values)
                    self.quantiles[state] = (n * np.quantile(values, grid) + 200 * self.quantiles[-1]) / (n + 200)
        return self

    def mixture(self, X, centers, weights):
        if centers.ndim == 1:
            centers = centers[:, None]
            weights = np.ones_like(centers)
        lo, hi = self.scales(X)
        values = []; masses = []
        for state in range(centers.shape[1]):
            q = self.quantiles.get(state, self.quantiles[-1])
            scale = np.where(q[None, :] < 0, lo[:, None], hi[:, None])
            values.append(centers[:, state, None] + scale * q)
            masses.append(np.broadcast_to(weights[:, state, None] / len(q), (len(X), len(q))))
        return np.concatenate(values, axis=1), np.concatenate(masses, axis=1)
