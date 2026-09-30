"""Player-specific, asymmetric QB tail widths learned from pregame features."""
import numpy as np

from nfl_pipeline.modeling.challenger_models import FittedHead, curve_summary, numeric, player_features
from nfl_pipeline.modeling.component_validation import interval_score
from nfl_pipeline.modeling.evaluation import probability_metrics
from nfl_pipeline.modeling.train_accuracy_challengers import line_frame


def qb_features(frame):
    x = player_features(frame, 'passing_yards')
    x['postseason'] = (numeric(frame, 'week') > np.where(numeric(frame, 'season') <= 2020, 17, 18)).astype(float)
    x['late_regular_week'] = numeric(frame, 'week').isin([17, 18]).astype(float)
    return x


class QBTailDistribution:
    def fit(self, frame):
        x = qb_features(frame)
        prior = numeric(frame, 'passing_yards_avg_5').fillna(0).to_numpy()
        y = numeric(frame, 'passing_yards').to_numpy()-prior
        self.heads = {q: FittedHead('quantile', params=dict(alpha=q, n_estimators=140,
            num_leaves=7, max_depth=3, min_child_samples=45, reg_lambda=20)).fit(x, y)
            for q in (.1, .5, .9)}
        self.strength = 0.
        return self

    def widths(self, frame):
        x = qb_features(frame)
        predictions = np.sort(np.column_stack([self.heads[q].predict(x) for q in (.1, .5, .9)]), axis=1)
        return np.maximum(20., predictions[:, 1]-predictions[:, 0]), np.maximum(20., predictions[:, 2]-predictions[:, 1])

    def transform(self, frame, values, weights, strength=None):
        alpha = self.strength if strength is None else strength
        if not 0 <= alpha <= 1:
            raise ValueError('Tail blend strength must be in [0, 1]')
        if alpha == 0:
            return values, weights
        summary = curve_summary(values, weights)
        low, high = self.widths(frame)
        lower_scale = low/np.maximum(20., summary['median']-summary['p10'])
        upper_scale = high/np.maximum(20., summary['p90']-summary['median'])
        factors = np.where(values < summary['median'][:, None], lower_scale[:, None], upper_scale[:, None])
        delta = values-summary['median'][:, None]
        return summary['median'][:, None] + delta*((1-alpha)+alpha*factors), weights

    def choose_strength(self, frame, values, weights):
        y = numeric(frame, 'passing_yards').to_numpy()
        trials = {}
        for alpha in (0., .25, .5, .75, 1.):
            v, w = self.transform(frame, values, weights, alpha)
            s = curve_summary(v, w)
            lines = line_frame(frame, 'passing_yards', v, w)
            trials[alpha] = dict(interval_score=float(np.mean(interval_score(y, s['p10'], s['p90']))),
                coverage=float(np.mean((y >= s['p10']) & (y <= s['p90']))),
                brier=probability_metrics(lines.probability, lines.outcome, lines.weight)['brier'])
        base = trials[0.]
        eligible = [a for a, m in trials.items() if a > 0
                    and m['brier'] <= base['brier']
                    and abs(m['coverage']-.8) < abs(base['coverage']-.8)
                    and m['interval_score'] < base['interval_score']]
        self.strength = min(eligible, key=lambda a: trials[a]['interval_score']) if eligible else 0.
        self.validation = dict(strength=self.strength, trials=trials, fit_rows=len(frame),
                               selection='earlier_calibration_fit_only_not_outer_test')
        return self
