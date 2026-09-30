"""Monotone central calibration with identity tails and later-week acceptance."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from nfl_pipeline.modeling.evaluation import probability_metrics


class TailPreservingCalibration:
    def __init__(self, boundary=.1):
        if not 0 < boundary < .5:
            raise ValueError('Tail boundary must be between zero and one half')
        self.boundary = boundary
        self.knots = np.array([boundary, 1-boundary])
        self.values = self.knots.copy()
        self.strength = 0.
        self.enabled = False
        self.validation = {'enabled': False, 'reason': 'not_fitted'}

    def candidate(self, p, strength=None):
        p = np.asarray(p, dtype=float)
        if not np.isfinite(p).all() or ((p < -1e-10) | (p > 1+1e-10)).any():
            raise ValueError('Calibration needs finite probabilities in [0, 1]')
        p = np.clip(p, 0., 1.)
        center = (p > self.boundary) & (p < 1-self.boundary)
        mapped = np.interp(p, self.knots, self.values)
        alpha = self.strength if strength is None else strength
        return np.where(center, (1-alpha)*p + alpha*mapped, p)

    def predict(self, p):
        return self.candidate(p) if self.enabled else np.asarray(p, dtype=float)

    def fit(self, fit_rows, tune_rows, gate_rows):
        """Three disjoint chronological blocks; the outer test is never consulted."""
        blocks = (fit_rows, tune_rows, gate_rows)
        keys = [set(zip(r.season, r.week)) for r in blocks]
        if any(keys[i] & keys[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError('Calibration blocks overlap')
        if any(not k for k in keys) or not max(keys[0]) < min(keys[1]) or not max(keys[1]) < min(keys[2]):
            raise ValueError('Calibration blocks must be chronological')
        self.enabled = False
        p = fit_rows.probability.to_numpy()
        central = (p > self.boundary) & (p < 1-self.boundary)
        subset = fit_rows.loc[central]
        if subset.player_game.nunique() < 30 or subset.outcome.nunique() < 2:
            self.validation = {'enabled': False, 'reason': 'insufficient_central_rows'}
            return self
        iso = IsotonicRegression(y_min=self.boundary, y_max=1-self.boundary, out_of_bounds='clip')
        iso.fit(subset.probability, subset.outcome, sample_weight=subset.weight)
        self.knots = np.r_[self.boundary, iso.X_thresholds_, 1-self.boundary]
        self.values = np.r_[self.boundary, iso.y_thresholds_, 1-self.boundary]
        # Strength < 1 keeps the entire transform strictly increasing, including anchors.
        losses = {a: probability_metrics(self.candidate(tune_rows.probability, a),
                  tune_rows.outcome, tune_rows.weight)['brier'] for a in (0., .25, .5, .75)}
        self.strength = min(losses, key=losses.get)
        raw = probability_metrics(gate_rows.probability, gate_rows.outcome, gate_rows.weight)
        calibrated = probability_metrics(self.candidate(gate_rows.probability), gate_rows.outcome, gate_rows.weight)
        self.enabled = bool(gate_rows.player_game.nunique() >= 30 and self.strength > 0
                            and calibrated['brier'] < raw['brier']
                            and calibrated['calibration_error'] <= raw['calibration_error'])
        self.validation = dict(enabled=self.enabled, strength=self.strength, tune_brier=losses,
                               raw=raw, calibrated=calibrated, tails='identity_at_and_outside_10_90_percent')
        return self

    def transform(self, values, weights):
        """Transform survival mass once for a coherent CDF at every possible line."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.shape != weights.shape or values.ndim != 2:
            raise ValueError('Expected matching two-dimensional supports and masses')
        if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid distribution')
        if (weights.sum(axis=1) <= 0).any():
            raise ValueError('Empty distribution')
        order = np.argsort(values, axis=1, kind='stable')
        v = np.take_along_axis(values, order, axis=1)
        w = np.take_along_axis(weights, order, axis=1)
        w = w / w.sum(axis=1, keepdims=True)
        cdf = np.minimum(1., np.cumsum(w, axis=1))
        cdf[:, -1] = 1.
        calibrated = 1-self.predict(1-cdf)
        masses = np.diff(np.c_[np.zeros(len(v)), calibrated], axis=1)
        if (masses < -1e-10).any():
            raise ValueError('Nonmonotone calibrated distribution')
        masses = np.maximum(0., masses)
        return v, masses / masses.sum(axis=1, keepdims=True)
