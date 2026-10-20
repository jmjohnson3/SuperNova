"""Player-game receiving distributions, independent of odds and bet selection."""
import numpy as np
import pandas as pd

from nfl_pipeline.modeling.challenger_models import FittedHead, numeric

CONTRACT = 'nfl-receiving-quantiles-v1'
QUANTILES = np.array([.005, .025, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .995])


def features(frame):
    # Stable workload and efficiency history, not current-game results or prices.
    bases = ('targets', 'receiving_yards', 'receiving_air_yards',
             'receiving_yards_after_catch', 'receptions', 'offense_snap_share')
    names = [f'{base}_{agg}_{window}' for base in bases
             for agg in ('avg', 'std') for window in (5, 10)]
    names += ['rest_days', 'is_home', 'n_games_prev_10', 'opp_allowed_receiving_yards_avg_5']
    out = pd.DataFrame({name: numeric(frame, name) for name in names}, index=frame.index)
    for window in (5, 10):
        targets = numeric(frame, f'targets_avg_{window}').where(lambda x: x > 0)
        for name, numerator in (('ypt', 'receiving_yards'), ('adot', 'receiving_air_yards')):
            out[f'{name}_{window}'] = numeric(frame, f'{numerator}_avg_{window}') / targets
    out['target_trend'] = numeric(frame, 'targets_avg_3') - numeric(frame, 'targets_avg_10')
    out['target_volatility'] = numeric(frame, 'targets_std_5') / (numeric(frame, 'targets_avg_5') + 1)
    for position in ('WR', 'TE', 'RB'):
        out['position_' + position] = frame.position.eq(position).astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    return pd.concat([out, out.isna().astype(float).add_suffix('__missing')], axis=1)


def validate_games(frame):
    if frame.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Quantile training requires unique player-games')
    if frame[['game_id', 'player_id', 'game_date_et']].isna().any().any():
        raise ValueError('Missing player-game identity or date')
    y = numeric(frame, 'receiving_yards').to_numpy()
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Quantile training needs observed receiving yards')
    return y


class ReceivingQuantiles:
    """Conditional quantile heads followed by held-out quantile bias correction."""
    def fit(self, frame):
        y = validate_games(frame)
        X = features(frame)
        self.heads = []
        for q in QUANTILES:
            head = FittedHead('quantile', params=dict(alpha=float(q), n_estimators=180,
                max_depth=3, num_leaves=7, min_child_samples=150, reg_lambda=30,
                learning_rate=.04)).fit(X, y)
            # Small/constant samples use the requested quantile, not a mean fallback.
            head.constant = float(np.quantile(y, q))
            self.heads.append(head)
        self.offsets = np.zeros(len(QUANTILES))
        self.fit_end = str(frame.game_date_et.max())
        self.fit_games = set(frame.game_id)
        self.feature_support = {c: int(X[c].notna().sum()) for c in X if not c.endswith('__missing')}
        return self

    def raw(self, frame):
        X = features(frame)
        # Rearrangement prevents crossed independently fitted quantiles.
        return np.sort(np.column_stack([head.predict(X) for head in self.heads]), axis=1)

    def correct(self, later, prior_rows=200):
        y = validate_games(later)
        if str(later.game_date_et.min()) <= self.fit_end or self.fit_games.intersection(later.game_id):
            raise ValueError('Quantile correction must follow model fitting without overlapping games')
        if prior_rows < 0:
            raise ValueError('Correction prior must be nonnegative')
        raw = self.raw(later)
        self.offsets = np.array([np.quantile(y-raw[:, i], q) for i, q in enumerate(QUANTILES)])
        self.offsets *= len(later)/(len(later)+prior_rows)
        self.correction_end = str(later.game_date_et.max())
        return self

    def curve(self, frame):
        knots = np.sort(self.raw(frame)+self.offsets, axis=1)
        grid = (np.arange(201)+.5)/201
        values = np.array([np.interp(grid, QUANTILES, row) for row in knots])
        # Negative receiving yardage is possible; do not silently clamp it to zero.
        return values, np.full_like(values, 1/len(grid))


def blend(left, right, fraction=.5):
    if not 0 <= fraction <= 1:
        raise ValueError('Blend fraction must be in [0, 1]')
    lv, lw = left
    rv, rw = right
    if len(lv) != len(rv) or lv.shape != lw.shape or rv.shape != rw.shape:
        raise ValueError('Blend requires matching player-games and distribution masses')
    return np.concatenate([lv, rv], axis=1), np.concatenate([(1-fraction)*lw, fraction*rw], axis=1)
