"""Runtime contracts for independently deployable NFL stat components."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_pipeline.modeling.challenger_models import curve_summary, player_features

CONTRACT = "nfl-stat-runtime-v1"


class TeamOpportunityRuntime:
    """Team budget -> player opportunity -> stable rate -> proper distribution."""

    def __init__(self, bundle: dict, stat: str):
        self.bundle = bundle
        self.stat = stat

    def score(self, frame: pd.DataFrame, production_projection) -> dict:
        from nfl_pipeline.modeling.train_accuracy_components import team_curve

        base = np.asarray(production_projection, dtype=float)
        if base.shape != (len(frame),) or not np.isfinite(base).all():
            raise ValueError("Stat runtime requires finite production projections")
        values, weights, components = team_curve(
            self.bundle,
            frame,
            pd.DataFrame(),
            pd.DataFrame(),
            self.stat,
            base=base,
            snapshot=True,
        )
        summary = curve_summary(values, weights)
        return {
            "values": values,
            "weights": weights,
            "mean": summary["mean"],
            "median": summary["median"],
            "p10": summary["p10"],
            "p90": summary["p90"],
            "opportunity": components["opportunity"],
            "efficiency": components["rate"],
            "limited_probability": components["state_probabilities"][:, 0],
            "normal_probability": components["state_probabilities"][:, 1],
            "spike_probability": components["state_probabilities"][:, 2],
            "allocation_covered": components["covered"],
        }


class ConditionalUncertaintyRuntime:
    """Keep the production center and replace only player-specific uncertainty."""

    def __init__(self, uncertainty, stat: str):
        self.uncertainty = uncertainty
        self.stat = stat

    def score(self, frame: pd.DataFrame, production_projection) -> dict:
        point = np.asarray(production_projection, dtype=float)
        values, weights = self.uncertainty.mixture(
            player_features(frame, self.stat), point, np.ones((len(frame), 1))
        )
        summary = curve_summary(values, weights)
        return {
            "values": values,
            "weights": weights,
            "mean": summary["mean"],
            "median": summary["median"],
            "p10": summary["p10"],
            "p90": summary["p90"],
            "allocation_covered": np.ones(len(frame), dtype=bool),
        }


class QBAttemptsEfficiencyRuntime:
    """QB attempts/workload states multiplied by conditional efficiency."""

    def __init__(self, bundle: dict, stat: str = "passing_yards"):
        self.bundle = bundle
        self.stat = stat

    def score(self, frame: pd.DataFrame, production_projection) -> dict:
        from nfl_pipeline.modeling.component_validation import calibrate_curve
        from nfl_pipeline.modeling.train_accuracy_challengers import mixture_curve

        base = np.asarray(production_projection, dtype=float)
        if base.shape != (len(frame),) or not np.isfinite(base).all():
            raise ValueError("QB stat runtime requires finite production projections")
        values, weights, components = mixture_curve(
            self.bundle["model"], self.bundle["uncertainty"], frame, self.stat
        )
        state_probabilities = np.asarray(components["weights"], dtype=float)
        opportunity = (
            state_probabilities * np.asarray(components["opportunities"], dtype=float)
        ).sum(axis=1)
        rate = np.divide(
            (state_probabilities * np.asarray(components["centers"], dtype=float)).sum(axis=1),
            opportunity,
            out=np.zeros(len(frame), dtype=float),
            where=opportunity > 1e-8,
        )
        tail = self.bundle.get("tail")
        if tail is not None:
            values, weights = tail.transform(values, weights, opportunity)
        calibrator = self.bundle.get("calibrator")
        if calibrator is not None:
            values, weights = calibrate_curve(
                calibrator, frame, self.stat, values, weights
            )
        summary = curve_summary(values, weights)
        return {
            "values": values,
            "weights": weights,
            "mean": summary["mean"],
            "median": summary["median"],
            "p10": summary["p10"],
            "p90": summary["p90"],
            "opportunity": opportunity,
            "efficiency": rate,
            "limited_probability": state_probabilities[:, 0],
            "normal_probability": state_probabilities[:, 1],
            "spike_probability": state_probabilities[:, 2],
            "allocation_covered": np.ones(len(frame), dtype=bool),
        }


def validate_score(score: dict, rows: int) -> None:
    values = np.asarray(score.get("values"), dtype=float)
    weights = np.asarray(score.get("weights"), dtype=float)
    if values.ndim != 2 or values.shape != weights.shape or values.shape[0] != rows:
        raise ValueError("Stat runtime returned an invalid curve matrix")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Stat runtime returned non-finite or negative distribution mass")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Stat runtime distributions must sum to one")
    for name in ("mean", "median", "p10", "p90"):
        value = np.asarray(score.get(name), dtype=float)
        if value.shape != (rows,) or not np.isfinite(value).all():
            raise ValueError(f"Stat runtime returned invalid {name}")
