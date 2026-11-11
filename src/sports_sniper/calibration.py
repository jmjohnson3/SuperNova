"""Probability calibration for the prediction models, fitted only on earlier days.

A model's raw chance is passed through a two-number logistic fit (Platt scaling on the log-odds):

    calibrated = sigmoid(a + b * logit(model))

b near 1 means the model's confidence can be taken at face value; b near 0 means its chance carries no
information and the calibrated chance collapses to the base rate. A second fit asks whether the model adds
anything to the market: the market's log-odds are held fixed and the model's disagreement gets a weight,

    blend = sigmoid(a + logit(consensus) + c * (logit(model) - logit(consensus)))

c near 0 means the market already knows what the model knows. Both are fitted per sport on every settled line
before the day being scored (`walk_forward`), so a calibrated chance is never judged on the games that fitted it.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sports_sniper.providers import results

MIN_LINES = 150         # settled lines needed before a fit is trusted
VERSION = "platt-v1"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def fit_logistic(features: np.ndarray, outcome: np.ndarray, offset: np.ndarray | None = None, ridge: float = 1e-3) -> np.ndarray:
    """Logistic regression by Newton's method (intercept first). `offset` is added to the linear part, unfitted."""
    x = np.column_stack([np.ones(len(outcome)), features]) if features.size else np.ones((len(outcome), 1))
    offset = np.zeros(len(outcome)) if offset is None else offset
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = sigmoid(x @ beta + offset)
        gradient = x.T @ (outcome - p) - ridge * beta
        hessian = (x * (p * (1 - p))[:, None]).T @ x + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit(lines: pd.DataFrame) -> dict | None:
    """Both fits on settled lines that have a model chance; None with fewer than MIN_LINES."""
    use = lines[lines.status.isin(["won", "lost"]) & lines.model_probability.notna() & lines.consensus_probability.notna()]
    if len(use) < MIN_LINES:
        return None
    y = (use.status == "won").to_numpy(dtype=float)
    model, market = logit(use.model_probability), logit(use.consensus_probability)
    a, b = fit_logistic(model[:, None], y)
    a2, c = fit_logistic((model - market)[:, None], y, offset=market)
    return dict(a=float(a), b=float(b), blend_a=float(a2), blend_c=float(c), lines=int(len(use)), version=VERSION)


def apply(coefficients: dict | None, model_probability, consensus_probability=None):
    """(calibrated, blend) chances. Either is NaN where it cannot be worked out."""
    model = np.asarray(model_probability, dtype=float)
    if coefficients is None:
        return np.full(model.shape, np.nan), np.full(model.shape, np.nan)
    calibrated = sigmoid(coefficients["a"] + coefficients["b"] * logit(model))
    calibrated = np.where(np.isnan(model), np.nan, calibrated)
    if consensus_probability is None:
        return calibrated, np.full(model.shape, np.nan)
    market = np.asarray(consensus_probability, dtype=float)
    blend = sigmoid(coefficients["blend_a"] + logit(market) + coefficients["blend_c"] * (logit(model) - logit(market)))
    return calibrated, np.where(np.isnan(model) | np.isnan(market), np.nan, blend)


def days_of(frame: pd.DataFrame) -> pd.Series:
    return frame.commence_time.dt.tz_convert(results.EASTERN).dt.date


def walk_forward(offers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add calibrated_probability and blend_probability (the over), each from a fit on earlier days only.
    Returns the offers and the fit used for each (sport, day)."""
    offers = offers.copy()
    offers["calibrated_probability"] = np.nan
    offers["blend_probability"] = np.nan
    fits: dict[tuple, dict] = {}
    if offers.empty or "model_probability" not in offers:
        return offers, fits
    day = days_of(offers)
    for sport in sorted(offers.sport.unique()):
        mine = offers.sport == sport
        for today in sorted(day[mine].unique()):
            coefficients = fit(offers[mine & (day < today)])
            if coefficients is None:
                continue
            fits[(sport, today)] = coefficients
            rows = mine & (day == today)
            calibrated, blend = apply(coefficients, offers.loc[rows, "model_probability"], offers.loc[rows, "consensus_probability"])
            offers.loc[rows, "calibrated_probability"] = calibrated
            offers.loc[rows, "blend_probability"] = blend
    return offers, fits


def fit_latest(offers: pd.DataFrame) -> dict:
    """The fit on everything settled so far, per sport: what a live scan uses for today's prices."""
    out = {}
    for sport in (sorted(offers.sport.unique()) if len(offers) and "model_probability" in offers else []):
        coefficients = fit(offers[offers.sport == sport])
        if coefficients is not None:
            out[sport] = coefficients
    return out


def annotate(found: pd.DataFrame, fits: dict, over_probability, versions) -> pd.DataFrame:
    """Give flagged prices the model's view of their own side: model_raw (the model's chance), model_probability
    (calibrated on earlier days) and model_agrees (the calibrated chance beats the price's break-even)."""
    found = found.copy()
    if found.empty:
        return found.assign(model_raw=pd.Series(dtype=float), model_version=pd.Series(dtype=object),
                            model_probability=pd.Series(dtype=float), model_agrees=pd.Series(dtype=object))
    over = pd.Series(over_probability, index=found.index, dtype=float)
    found["model_raw"] = over.where(found.side == "over", 1 - over) if len(found) else over
    found["model_version"] = pd.Series(versions, index=found.index, dtype=object)
    found["model_probability"] = np.nan
    if len(found):
        day = days_of(found)
        for (sport, today), coefficients in fits.items():
            rows = (found.sport == sport) & (day == today) & over.notna()
            if rows.any():
                calibrated, _ = apply(coefficients, over[rows])
                found.loc[rows, "model_probability"] = np.where(found.loc[rows, "side"] == "over", calibrated, 1 - calibrated)
    found["model_agrees"] = [None if pd.isna(p) else bool(p > b) for p, b in zip(found.model_probability, found.break_even)]
    return found


def side_probability(coefficients: dict | None, over_probability: float | None, side: str) -> float | None:
    """The calibrated chance of one side from the model's chance of the over; None without a fit or a forecast."""
    if coefficients is None or over_probability is None or pd.isna(over_probability):
        return None
    calibrated = float(apply(coefficients, [over_probability])[0][0])
    return calibrated if side == "over" else 1 - calibrated


def default_path(database_url: str) -> Path | None:
    """Where the latest fit is kept: beside a SQLite database, nowhere for any other."""
    if not database_url.startswith("sqlite:///") or ":memory:" in database_url:
        return None
    return Path(database_url.removeprefix("sqlite:///")).with_name("model_calibration.json")


def save(fits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(fitted_at=datetime.now(timezone.utc).isoformat(), sports=fits), indent=2), encoding="utf-8")


def load(path: Path | None) -> dict:
    """{sport: coefficients} from the last saved fit; empty when there is none or it cannot be read."""
    if path is None:
        return {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in stored.get("sports", {}).items() if all(math.isfinite(v[n]) for n in ("a", "b"))}
    except Exception:
        return {}
