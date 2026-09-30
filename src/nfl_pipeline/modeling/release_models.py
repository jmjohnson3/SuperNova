"""Serializable estimators shared by offline training and live scoring."""
import numpy as np
import pandas as pd


class ConservativeStatModel:
    """Fixed residual shrinkage; never tuned on the evaluation labels."""
    def __init__(self, estimator, baseline_column):
        self.estimator = estimator
        self.baseline_column = baseline_column

    def predict(self, X):
        baseline = pd.to_numeric(X[self.baseline_column], errors="coerce").fillna(0).to_numpy()
        return np.maximum(0, baseline + 0.5 * self.estimator.predict(X))


class RareEventStatModel:
    def __init__(self, estimator, baseline_column, positive_mean):
        self.estimator = estimator
        self.baseline_column = baseline_column
        self.positive_mean = positive_mean

    def predict_any(self, X):
        baseline = pd.to_numeric(X[self.baseline_column], errors="coerce").fillna(0).clip(lower=0).to_numpy()
        return 0.75 * self.estimator.predict_proba(X)[:, 1] + 0.25 * (1 - np.exp(-baseline))

    def predict(self, X):
        return self.positive_mean * self.predict_any(X)
