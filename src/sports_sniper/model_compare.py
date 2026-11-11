"""Model against market: which probability source is closest to what happened?

The test set is every two-sided line once, at the first look where a bet book and a sharp reference both priced
it, graded on the over. Three sources are scored on the same lines:

  book        the bet book's own two prices with its margin removed
  consensus   the other books' no-vig prices (what the Opportunity Board uses)
  model       the repository's prediction model, from a forecast made before the price was seen

Brier score and log loss measure how close the probabilities were (lower is better). The picks table then asks
the practical question: betting whichever side a source says is underpriced by at least a threshold, at the bet
book's price, what would have happened? A source that cannot beat the book's own price adds no information.
"""
from __future__ import annotations

import math

import pandas as pd

from sports_sniper import odds_math, performance
from sports_sniper.config import Settings
from sports_sniper.providers import models

SOURCES = {"book": "Bet book's own price (no-vig)", "consensus": "Market consensus (other books)", "model": "Prediction model"}


def attach_models(offers: pd.DataFrame, settings: Settings, nba=None) -> pd.DataFrame:
    """Add model_probability (the over) and model_version to the offers."""
    offers = offers.copy()
    values = (nba or models.nba_over_probabilities)(settings.results_dsn, offers) if len(offers) else []
    offers["model_probability"] = pd.Series(values, index=offers.index, dtype=float)
    offers["model_version"] = [models.NBA_MODEL if pd.notna(v) else None for v in offers.model_probability]
    return offers


def log_loss(probabilities, outcomes) -> float:
    total = 0.0
    for p, o in zip(probabilities, outcomes):
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        total -= math.log(p if o else 1 - p)
    return total / len(outcomes)


def scores(offers: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Brier score and log loss of each source on the lines all three cover (decided overs only)."""
    decided = offers[offers.status.isin(performance.DECIDED) & offers.model_probability.notna()] if len(offers) else offers
    rows = []
    if not len(decided):
        return pd.DataFrame(rows)
    for key, group in ([("All", decided)] if by is None else decided.groupby(by)):
        if not len(group):
            continue
        outcome = (group.status == "won").astype(int).tolist()
        row = {"group": key, "lines": len(group), "over_rate": sum(outcome) / len(outcome)}
        for source in SOURCES:
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


def picks_table(offers: pd.DataFrame, thresholds=(0.02, 0.04, 0.06)) -> pd.DataFrame:
    """Each source's picks at each threshold, on the lines the model covers (so the sources face the same lines)."""
    common = offers[offers.model_probability.notna()] if len(offers) else offers
    rows = []
    for source in ("consensus", "model"):
        for threshold in thresholds:
            s = performance.summarize(picks(common, source, threshold))
            rows.append(dict(source=source, threshold=threshold, bets=s["won"] + s["lost"] + s["push"], won=s["won"], lost=s["lost"],
                             win_rate=s["win_rate"], roi=s["roi"], roi_se=s["roi_se"], estimated_ev=s["estimated_ev"]))
    return pd.DataFrame(rows)


def calibration(offers: pd.DataFrame, source: str, bins: int = 5) -> list[dict]:
    decided = offers[offers.status.isin(performance.DECIDED) & offers[f"{source}_probability"].notna()]
    return odds_math.calibration_table(decided[f"{source}_probability"].tolist(), (decided.status == "won").astype(int).tolist(), bins)
