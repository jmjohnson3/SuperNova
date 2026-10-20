"""Receiving efficiency independent of opportunity, offered lines and prices."""
import numpy as np
import pandas as pd

from nfl_pipeline.modeling.challenger_models import FittedHead, numeric

CONTRACT = 'nfl-receiving-efficiency-v1'
PARAMS = dict(n_estimators=65, max_depth=2, num_leaves=4, min_child_samples=120,
              reg_lambda=60, learning_rate=.035)


class ReceivingEfficiency:
    """Exposure-shrunk player skill plus a small depth/YAC/matchup residual."""
    def __init__(self, prior_targets=60.):
        self.prior_targets = float(prior_targets)
        if self.prior_targets <= 0:
            raise ValueError('Prior exposure must be positive')

    def prior(self, frame):
        league = frame.position.map(self.position_rates).fillna(self.global_rate)
        games = numeric(frame, 'n_games_prev_10').clip(0, 10)
        targets = numeric(frame, 'targets_avg_10') * games
        yards = numeric(frame, 'receiving_yards_avg_10') * games
        known = targets.notna() & yards.notna() & targets.ge(0) & games.gt(0)
        # Missing history is a prior fallback, not observed zero production.
        rate = (yards + self.prior_targets*league)/(targets + self.prior_targets)
        return rate.where(known, league).clip(lower=0).to_numpy(), known.to_numpy()

    def features(self, frame):
        prior, known = self.prior(frame)
        X = pd.DataFrame({'shrunk_ypt': prior, 'player_history_known': known.astype(float)}, index=frame.index)
        for window in (3, 5, 10):
            targets = numeric(frame, f'targets_avg_{window}').where(lambda s: s > 0)
            receptions = numeric(frame, f'receptions_avg_{window}').where(lambda s: s > 0)
            for label, numerator, denominator in (
                ('ypt', 'receiving_yards', targets), ('adot', 'receiving_air_yards', targets),
                ('catch_rate', 'receptions', targets), ('yac_per_catch', 'receiving_yards_after_catch', receptions)):
                X[f'{label}_{window}'] = numeric(frame, f'{numerator}_avg_{window}') / denominator
            X[f'target_history_{window}'] = numeric(frame, f'targets_avg_{window}')
        X['ypt_trend'] = X.ypt_3-X.ypt_10
        X['depth_trend'] = X.adot_3-X.adot_10
        for key in ('opp_allowed_receiving_yards_avg_5', 'rest_days', 'is_home', 'n_games_prev_10'):
            X[key] = numeric(frame, key)
        for pos in ('WR', 'TE', 'RB'):
            X['position_'+pos] = frame.position.eq(pos).astype(float)
        X = X.replace([np.inf, -np.inf], np.nan)
        return pd.concat([X, X.isna().astype(float).add_suffix('__missing')], axis=1)

    def fit(self, frame):
        if frame.duplicated(['game_id', 'player_id']).any():
            raise ValueError('Efficiency training requires unique player-games')
        exposure = numeric(frame, 'targets').to_numpy()
        yards = numeric(frame, 'receiving_yards').to_numpy()
        valid = np.isfinite(exposure) & (exposure > 0) & np.isfinite(yards)
        if not valid.any():
            raise ValueError('Efficiency training needs observed positive target exposure')
        self.global_rate = float(yards[valid].sum()/exposure[valid].sum())
        self.position_rates = {}
        for pos in ('WR', 'TE', 'RB'):
            use = valid & frame.position.eq(pos).to_numpy()
            self.position_rates[pos] = float((yards[use].sum()+100*self.global_rate)/(exposure[use].sum()+100))
        prior, _ = self.prior(frame)
        X = self.features(frame)
        self.head = FittedHead(params=PARAMS).fit(X.loc[valid], yards[valid]/exposure[valid]-prior[valid],
                                                weights=exposure[valid])
        self.feature_support = {c:int(X[c].notna().sum()) for c in X if not c.endswith('__missing')}
        self.fit_player_games = int(valid.sum())
        return self

    def predict(self, frame, residual_weight=1.):
        if not 0 <= residual_weight <= 1:
            raise ValueError('Residual weight must be in [0, 1]')
        prior, _ = self.prior(frame)
        return np.maximum(0., prior+residual_weight*self.head.predict(self.features(frame)))


def frozen_opportunity(frame):
    """The experiment changes rate only, never the five-game target control."""
    values = numeric(frame, 'targets_avg_5').to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Missing or invalid frozen target forecast')
    return values


def forecast(bundle, frame, *, projection=None, residuals=None):
    """Translate a rate improvement into a center shift; leave curve shape intact."""
    targets = frozen_opportunity(frame)
    reference = bundle['reference'].predict(frame) if projection is None else np.asarray(projection, dtype=float)
    errors = bundle['reference_errors'] if residuals is None else np.asarray(residuals, dtype=float)
    if reference.shape != (len(frame),) or not np.isfinite(reference).all():
        raise ValueError('Missing frozen yardage forecast')
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError('Missing frozen residual distribution')
    rate = bundle['model'].predict(frame, bundle['residual_weight'])
    # Reference is a central forecast; its empirical distribution mean is the mean comparator.
    reference_curve = reference[:, None]+errors[None, :]
    reference_mean = reference_curve.mean(axis=1)
    mean_shift = bundle['alpha']*(targets*rate-reference_mean)
    candidate = reference_curve+mean_shift[:, None]
    weights = np.full_like(candidate, 1/candidate.shape[1])
    return dict(reference=reference_curve, candidate=candidate, weights=weights,
                targets=targets, rate=rate, center_shift=mean_shift)
