"""Model against market: which probability source is closest to what happened?

The test set is every two-sided line once, at the first look where a bet book and a sharp reference both priced
it, graded on the over. The sources scored on the same lines:

  book         the bet book's own two prices with its margin removed
  consensus    the other books' no-vig prices (what the Opportunity Board uses)
  model        the repository's prediction model, from a forecast made before the price was seen
  calibrated   the model after calibration fitted on earlier days only
  blend        the market's chance moved by the model's disagreement, the weight fitted on earlier days only

Brier score and log loss measure how close the probabilities were (lower is better). The picks table then asks
the practical question: betting whichever side a source says is underpriced by at least a threshold, at the bet
book's price, what would have happened? A source that cannot beat the book's own price adds no information.
"""
from __future__ import annotations

import math

import pandas as pd

from sports_sniper import calibration, nba_candidate, odds_math, performance
from sports_sniper.config import Settings
from sports_sniper.providers import models

SOURCES = {"book": "Bet book's own price (no-vig)", "consensus": "Market consensus (other books)", "model": "Prediction model",
           "calibrated": "Model, calibrated on earlier days", "blend": "Market moved by the model"}
BASE = ("book", "consensus", "model")
# the NBA candidate (uncertainty measured on later games) and calibration by statistic
CANDIDATE_SOURCES = {"consensus": "Market consensus", "model": "Live model", "calibrated": "Live model, calibrated by sport",
                     "stat_calibrated": "Live model, calibrated by statistic", "candidate": "Candidate (uncertainty from later games)",
                     "candidate_calibrated": "Candidate, calibrated by statistic", "candidate_blend": "Market moved by the candidate"}
CANDIDATE = nba_candidate.over_probabilities        # replaced in tests; None switches the candidate off


def attach_models(offers: pd.DataFrame, settings: Settings, adapters=None) -> tuple[pd.DataFrame, dict]:
    """Add model_probability (the over), model_version, and the walk-forward calibrated and blended chances.
    Returns the offers and the calibration fit used for each (sport, day)."""
    offers = offers.copy()
    values, versions = models.over_probabilities(settings.results_dsn, offers, adapters or models.ADAPTERS) if len(offers) else ([], [])
    offers["model_probability"] = pd.Series(values, index=offers.index, dtype=float)
    offers["model_version"] = pd.Series(versions, index=offers.index, dtype=object)
    offers, fits = calibration.walk_forward(offers)
    params: dict = {}
    candidate = [None] * len(offers)
    if CANDIDATE is not None and len(offers):
        try:
            candidate, params = CANDIDATE(settings.results_dsn, offers)
        except Exception:                    # the projections database is optional
            candidate, params = [None] * len(offers), {}
    offers["candidate_probability"] = pd.Series(candidate, index=offers.index, dtype=float)
    offers["stat_calibrated_probability"], _, _ = calibration.calibrate(offers, "model", by_market=True)
    offers["candidate_calibrated_probability"], offers["candidate_blend_probability"], stat_fits = calibration.calibrate(
        offers, "candidate", by_market=True)
    offers.attrs["candidate_params"], offers.attrs["candidate_fits"] = params, stat_fits
    return offers, fits


def log_loss(probabilities, outcomes) -> float:
    total = 0.0
    for p, o in zip(probabilities, outcomes):
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        total -= math.log(p if o else 1 - p)
    return total / len(outcomes)


def scores(offers: pd.DataFrame, by: str | None = None, sources: tuple[str, ...] = BASE) -> pd.DataFrame:
    """Brier score and log loss of each source on the decided lines that every one of `sources` covers."""
    rows: list[dict] = []
    if not len(offers):
        return pd.DataFrame(rows)
    covered = offers.status.isin(performance.DECIDED)
    for source in sources:
        covered &= offers[f"{source}_probability"].notna()
    decided = offers[covered]
    if not len(decided):
        return pd.DataFrame(rows)
    for key, group in ([("All", decided)] if by is None else decided.groupby(by)):
        outcome = (group.status == "won").astype(int).tolist()
        row = {"group": key, "lines": len(group), "over_rate": sum(outcome) / len(outcome)}
        for source in sources:
            p = group[f"{source}_probability"].tolist()
            row[f"{source}_brier"] = odds_math.brier_score(p, outcome)
            row[f"{source}_log_loss"] = log_loss(p, outcome)
        rows.append(row)
    return pd.DataFrame(rows)


def picks(offers: pd.DataFrame, source: str, threshold: float) -> pd.DataFrame:
    """Bets a source would make: the side it prices at least `threshold` above break-even, at the book's price."""
    column = f"{source}_probability"
    usable = offers[offers[column].notna() & offers.status.isin(performance.SETTLED)] if len(offers) else offers
    rows = []
    for o in usable.itertuples():
        p_over = float(getattr(o, column))
        for side, p, decimal in (("over", p_over, o.over_decimal), ("under", 1 - p_over, o.under_decimal)):
            ev = odds_math.expected_roi(p, decimal)
            if ev >= threshold:
                status = o.status if o.status == "push" else ("won" if (o.status == "won") == (side == "over") else "lost")
                rows.append(dict(event_id=o.event_id, sport=o.sport, market=o.market, side=side, decimal=decimal, probability=p,
                                 ev=ev, status=status, stake=1.0, closing_line_value=None,
                                 profit={"won": decimal - 1, "lost": -1.0}.get(status, 0.0)))
    return pd.DataFrame(rows, columns=["event_id", "sport", "market", "side", "decimal", "probability", "ev", "status", "stake",
                                       "closing_line_value", "profit"])


def picks_table(offers: pd.DataFrame, thresholds=(0.02, 0.04, 0.06), sources=("consensus", "model", "calibrated", "blend")) -> pd.DataFrame:
    """Each source's picks at each threshold, on the lines every listed source covers (so they face the same lines)."""
    common = offers
    for source in sources:
        if len(common):
            common = common[common[f"{source}_probability"].notna()]
    rows = []
    for source in sources:
        for threshold in thresholds:
            s = performance.summarize(picks(common, source, threshold))
            rows.append(dict(source=source, threshold=threshold, bets=s["won"] + s["lost"] + s["push"], won=s["won"], lost=s["lost"],
                             win_rate=s["win_rate"], roi=s["roi"], roi_se=s["roi_se"], estimated_ev=s["estimated_ev"]))
    return pd.DataFrame(rows)


def calibration_table(offers: pd.DataFrame, source: str, bins: int = 5) -> list[dict]:
    decided = offers[offers.status.isin(performance.DECIDED) & offers[f"{source}_probability"].notna()]
    return odds_math.calibration_table(decided[f"{source}_probability"].tolist(), (decided.status == "won").astype(int).tolist(), bins)
