"""Isolated opportunity, efficiency and small totals challengers.

All feature lists are pregame allowlists. Current opportunity is a training
target/weight only, never a prediction input. Market prices are not features.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_pipeline.modeling.challenger_models import numeric, curve_summary
from nfl_pipeline.modeling.role_workload import RoleWorkload, STATS, features, validate, count_crps
from nfl_pipeline.modeling.sample_aware_splits import sample_aware_partitions, REQUIREMENTS, describe_blocks
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, game_baseline, line_frame
from nfl_pipeline.modeling.evaluation import probability_metrics
from nfl_pipeline.modeling.component_validation import interval_score
from nfl_pipeline.modeling import distribution_policy as dp

CONTRACT = 'nfl-isolated-forecast-repair-v1'
VARIANTS = ('reference', 'workload', 'efficiency', 'joint', 'joint_calibrated')


class StableEfficiency:
    """Exposure-shrunk longer-term skill plus a regularized residual rate head."""
    def fit(self, frame, stat):
        count, yards = validate(frame, stat)
        self.stat = stat
        self.priors = {}
        for position, g in frame.assign(_n=count, _y=yards).groupby('position'):
            self.priors[position] = float(g._y.sum()/max(1., g._n.sum()))
        positive = count > 0
        x = self.inputs(frame)
        self.columns = [c for c in x if x[c].notna().mean() >= .7 and x[c].nunique() > 1]
        if not self.columns or not positive.any():
            raise ValueError('Insufficient observed efficiency training inputs')
        self.model = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(alpha=500.))
        residual = yards[positive]/count[positive]-self.prior(frame)[positive]
        self.model.fit(x.loc[positive, self.columns], residual, ridge__sample_weight=np.minimum(count[positive], 25.))
        self.strength = 0.
        return self

    def prior(self, frame):
        op = STATS[self.stat]
        exposure = (numeric(frame, op+'_avg_10')*numeric(frame, 'n_games_prev_10').clip(upper=10)).fillna(0).clip(lower=0)
        rate = numeric(frame, self.stat+'_avg_10')/numeric(frame, op+'_avg_10').replace(0, np.nan)
        league = frame.position.map(self.priors)
        if league.isna().any():
            raise ValueError('Efficiency position without training support')
        # Rate skill changes more slowly than weekly opportunity.  The prior is
        # intentionally strongest for QB efficiency, then rushing, then
        # receiving, and is selected independently of the workload model.
        prior_exposure = {'receiving_yards': 120., 'rushing_yards': 160.,
                          'passing_yards': 240.}[self.stat]
        shrink = exposure/(exposure+prior_exposure)
        return (shrink*rate.fillna(league)+(1-shrink)*league).to_numpy()

    def inputs(self, frame):
        op = STATS[self.stat]
        x = pd.DataFrame(index=frame.index)
        for w in (5, 10):
            x['rate_'+str(w)] = numeric(frame, self.stat+f'_avg_{w}')/numeric(frame, op+f'_avg_{w}').replace(0, np.nan)
        for name in ('is_home', 'rest_days', 'n_games_prev_10', f'opp_allowed_{self.stat}_avg_5'):
            x[name] = numeric(frame, name)
        x['position_QB'] = frame.position.eq('QB').astype(float)
        return x.replace([np.inf, -np.inf], np.nan)

    def predict(self, frame, strength=None):
        alpha = self.strength if strength is None else strength
        if not 0 <= alpha <= 1:
            raise ValueError('Efficiency residual strength outside [0,1]')
        return self.prior(frame)+alpha*self.model.predict(self.inputs(frame)[self.columns])

    def tune(self, frame):
        n, y = validate(frame, self.stat)
        mask = n > 0
        if not mask.any():
            raise ValueError('Efficiency tuning needs positive observed exposure')
        self.tuning = {a: float(np.mean((n[mask]*self.predict(frame, a)[mask]-y[mask])**2))
                       for a in (0., .25, .5, 1.)}
        self.strength = min(self.tuning, key=lambda a: (self.tuning[a], a))
        return self


def compressed(values, weights, points=151):
    """Bound CDF fitting cost without clipping outcome tails."""
    grid = (np.arange(points)+.5)/points
    result = []
    for v, w in zip(values, weights):
        order = np.argsort(v)
        at = np.searchsorted(np.cumsum(w[order]/w.sum()), grid)
        result.append(v[order[np.minimum(at, len(order)-1)]])
    v = np.asarray(result)
    return v, np.full_like(v, 1/points)


class PlayerRepair:
    def fit(self, frame, stat):
        self.stat = stat
        early, count_tail, rate_tune, residual, calibration, gate = sample_aware_partitions(
            frame, stat, requirements=REQUIREMENTS[stat])
        self.blocks = describe_blocks((early, count_tail, rate_tune, residual, calibration, gate))
        for b, name in zip(self.blocks, ('fit', 'count_tail', 'rate_tuning', 'efficiency_tail', 'cdf_fit', 'cdf_gate')):
            b['name'] = name
        self.training_end = str(frame.game_date_et.max())
        self.reference = ReferenceRecipe().fit(early, stat)
        self.rate = StableEfficiency().fit(early, stat).tune(rate_tune)
        self.workload = RoleWorkload().fit(early, stat).fit_count_tails(count_tail).fit_count_weight(rate_tune)
        self.joint = deepcopy(self.workload)
        self.workload.fit_efficiency_tails(residual, self.reference).finish_count_first()
        self.joint.fit_efficiency_tails(residual, self.reference, self.rate.predict(residual)).finish_count_first()
        self.reference_errors = np.quantile(numeric(residual, stat)-self.reference.predict(residual), (np.arange(151)+.5)/151)
        v, w = self.curve(calibration, 'joint')
        rows = [dict(key=str(g)+'|'+str(p), curve=dense_yard_curve(a, b), actual=float(y))
                for g, p, a, b, y in zip(calibration.game_id, calibration.player_id, v, w, numeric(calibration, stat))]
        self.cdf = dp.fit_cdf(rows)
        self.cdf_candidate = dict(self.cdf)
        self.cdf_validation = self.gate_calibration(gate)
        if not self.cdf_validation['accepted']:
            self.cdf = dict(a=1., b=1.)
        return self

    def gate_calibration(self, frame):
        evidence = {}
        for name in ('joint', 'joint_calibrated'):
            v, w = self.curve(frame, name)
            s = curve_summary(v, w)
            y = numeric(frame, self.stat).to_numpy()
            lines = line_frame(frame, self.stat, v, w)
            evidence[name] = dict(probability_metrics(lines.probability, lines.outcome, lines.weight),
                below=float((y < s['p10']).mean()), above=float((y > s['p90']).mean()),
                coverage=float(((y >= s['p10']) & (y <= s['p90'])).mean()),
                interval_score=float(np.mean(interval_score(y, s['p10'], s['p90']))))
        a, b = evidence['joint_calibrated'], evidence['joint']
        accepted = (a['brier'] < b['brier'] and a['calibration_error'] <= b['calibration_error']
            and abs(a['coverage']-.8) <= abs(b['coverage']-.8) and a['interval_score'] <= b['interval_score']
            and abs(a['below']-.1) <= abs(b['below']-.1) and abs(a['above']-.1) <= abs(b['above']-.1))
        return dict(accepted=accepted, scores=evidence, rows=len(frame),
            first=str(frame.game_date_et.min()), last=str(frame.game_date_et.max()), scope='later_inner_gate_not_outer_test')

    def curve(self, frame, variant='joint', projection=None):
        if variant not in VARIANTS:
            raise ValueError('Unknown isolated repair variant')
        point = self.reference.predict(frame) if projection is None else np.asarray(projection, dtype=float)
        if variant == 'reference':
            v = point[:, None]+self.reference_errors[None, :]
            return v, np.full_like(v, 1/v.shape[1])
        model = self.workload if variant == 'workload' else self.joint
        rate = None if variant == 'workload' else self.rate.predict(frame)
        v, w, _ = model.yardage_curve(frame, projection=point, rates=rate,
            variant='tails_only' if variant == 'efficiency' else 'combined')
        v, w = compressed(v, w)
        if variant == 'joint_calibrated' and (self.cdf['a'] != 1. or self.cdf['b'] != 1.):
            # A single monotone CDF transform controls every offered line and push.
            transformed = [dp.calibrate(dense_yard_curve(a, b), self.cdf) for a, b in zip(v, w)]
            v, w = compressed_dicts(transformed)
        return v, w


def dense_yard_curve(values, weights):
    # Quantile supports can skip integer yards. Interpolate the CDF between
    # knots, preserving atoms, so observed yards are not assigned fake zero mass.
    c = dp.curve(values, weights)
    support = np.arange(min(c['values']), max(c['values'])+1)
    return dp.from_cdf(support, np.interp(support, c['values'], np.cumsum(c['weights'])))


def compressed_dicts(curves):
    values, weights = [], []
    for c in curves:
        v, w = compressed(np.array([c['values']]), np.array([c['weights']]))
        values.append(v[0]); weights.append(w[0])
    return np.array(values), np.array(weights)


def total_inputs(frame):
    """Small symmetric features, with no actual score, closing line or weather hindsight."""
    x = pd.DataFrame(index=frame.index)
    for name in ('pf', 'pa', 'plays', 'pass_rate', 'yards_per_play', 'red_zone_td_rate'):
        for window in (5, 10):
            x[f'{name}_{window}'] = (numeric(frame, f'home_{name}_avg_{window}')+numeric(frame, f'away_{name}_avg_{window}'))/2
    x['rest_mean'] = (numeric(frame, 'home_rest_days')+numeric(frame, 'away_rest_days'))/2
    return x.replace([np.inf, -np.inf], np.nan)


class SmallTotalRepair:
    def fit(self, frame):
        data = frame.assign(player_id='game')
        from nfl_pipeline.modeling.sample_aware_splits import BlockRequirement
        early, residual, tune = sample_aware_partitions(data, 'passing_yards',
            requirements=(BlockRequirement(100, 8, 70), BlockRequirement(80, 6, 50)))
        self.blocks = describe_blocks((early, residual, tune))
        for b, name in zip(self.blocks, ('fit', 'residual', 'blend_tuning')): b['name'] = name
        self.training_end = str(frame.game_date_et.max())
        x = total_inputs(early)
        self.support = {c: float(x[c].notna().mean()) for c in x}
        self.columns = [c for c in x if self.support[c] >= .9 and x[c].nunique() > 1]
        if not self.columns:
            raise ValueError('Totals have no reliable lagged feature support')
        self.model = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(alpha=500.))
        y = numeric(early, 'total_points_actual').to_numpy()
        self.model.fit(x[self.columns], y-game_baseline(early, 'total_points_actual'))
        self.residuals = {}
        for strength in (0., .25, .5, 1.):
            errors = numeric(residual, 'total_points_actual')-self.point(residual, strength)
            self.residuals[strength] = np.quantile(errors, (np.arange(151)+.5)/151)
        self.tuning = {}
        for strength in self.residuals:
            v, w = self.curve(tune, strength=strength)
            self.tuning[strength] = float(np.mean(count_crps(v, w, numeric(tune, 'total_points_actual'))))
        self.strength = min(self.tuning, key=lambda a: (self.tuning[a], a))
        return self

    def point(self, frame, strength):
        return game_baseline(frame, 'total_points_actual')+strength*self.model.predict(total_inputs(frame)[self.columns])

    def curve(self, frame, variant='small_total', strength=None, projection=None):
        if strength is None: strength = 0. if variant == 'reference' else self.strength
        point = self.point(frame, strength)
        v = np.maximum(0., point[:, None]+self.residuals[strength][None, :])
        return v, np.full_like(v, 1/v.shape[1])
