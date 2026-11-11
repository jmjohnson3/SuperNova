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


def fit(lines: pd.DataFrame, source: str = "model", min_lines: int = MIN_LINES) -> dict | None:
    """Both fits on settled lines that have the source's chance; None with fewer than `min_lines`.

    The weight the source gets against the market is never negative: a source the results argue against is
    given no say (weight 0, the market's own chance with its intercept) rather than being bet against.
    """
    column = f"{source}_probability"
    use = lines[lines.status.isin(["won", "lost"]) & lines[column].notna() & lines.consensus_probability.notna()]
    if len(use) < min_lines:
        return None
    y = (use.status == "won").to_numpy(dtype=float)
    model, market = logit(use[column]), logit(use.consensus_probability)
    a, b = fit_logistic(model[:, None], y)
    a2, c = fit_logistic((model - market)[:, None], y, offset=market)
    if c < 0:
        a2, c = fit_logistic(np.empty((len(y), 0)), y, offset=market)[0], 0.0
    return dict(a=float(a), b=float(b), blend_a=float(a2), blend_c=float(c), lines=int(len(use)), version=VERSION)


MIN_MARKET_LINES = 300      # settled lines of one statistic needed before it gets a fit of its own
SHRINK_LINES = 600          # a statistic's own fit counts n / (n + this) against its sport's fit


def mix(sport_fit: dict, market_fit: dict | None) -> dict:
    """A statistic's coefficients: its sport's fit when it has too few lines of its own, otherwise its own fit
    pulled toward the sport's in proportion to how few lines it has."""
    if market_fit is None:
        return dict(sport_fit, own_weight=0.0, own_lines=0)
    w = market_fit["lines"] / (market_fit["lines"] + SHRINK_LINES)
    out = {k: w * market_fit[k] + (1 - w) * sport_fit[k] for k in ("a", "b", "blend_a", "blend_c")}
    return dict(out, lines=sport_fit["lines"], version=VERSION, own_weight=float(w), own_lines=market_fit["lines"])


def calibrate(offers: pd.DataFrame, source: str, *, by_market: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """(calibrated chance, blended chance, fits) for `source`, every row from fits on earlier days only.

    With `by_market`, each statistic is calibrated on its own earlier lines, falling back to (and shrunk toward)
    its sport's fit while it has few. Fits are keyed (sport, day), or (sport, day, market) when by statistic.
    """
    calibrated, blend = np.full(len(offers), np.nan), np.full(len(offers), np.nan)
    fits: dict[tuple, dict] = {}
    column = f"{source}_probability"
    if offers.empty or column not in offers:
        return calibrated, blend, fits
    day = days_of(offers).to_numpy()
    sport_of, market_of = offers.sport.to_numpy(), offers.market.to_numpy()
    for sport in sorted(set(sport_of)):
        mine = sport_of == sport
        for today in sorted(set(day[mine])):
            earlier = offers[mine & (day < today)]
            sport_fit = fit(earlier, source)
            if sport_fit is None:
                continue
            rows = mine & (day == today)
            groups = [(None, rows)] if not by_market else [(m, rows & (market_of == m)) for m in sorted(set(market_of[rows]))]
            for market, where in groups:
                coefficients = sport_fit if market is None else mix(sport_fit, fit(earlier[earlier.market == market], source, MIN_MARKET_LINES))
                fits[(sport, today) if market is None else (sport, today, market)] = coefficients
                c, b = apply(coefficients, offers.loc[where, column], offers.loc[where, "consensus_probability"])
                calibrated[where], blend[where] = c, b
    return calibrated, blend, fits


def latest_by_market(offers: pd.DataFrame, source: str) -> pd.DataFrame:
    """The by-statistic fit on everything settled so far: one row per sport and market."""
    rows = []
    column = f"{source}_probability"
    if offers.empty or column not in offers:
        return pd.DataFrame(rows)
    for sport in sorted(offers.sport.unique()):
        mine = offers[offers.sport == sport]
        sport_fit = fit(mine, source)
        if sport_fit is None:
            continue
        for market in sorted(mine.market.unique()):
            c = mix(sport_fit, fit(mine[mine.market == market], source, MIN_MARKET_LINES))
            rows.append(dict(sport=sport, market=market, own_lines=c["own_lines"], own_weight=c["own_weight"], confidence=c["b"],
                             weight_against_market=c["blend_c"]))
    return pd.DataFrame(rows)


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
    offers["calibrated_probability"], offers["blend_probability"], fits = calibrate(offers, "model")
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
