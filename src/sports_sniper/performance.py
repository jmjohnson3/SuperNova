"""How the paper bets did: record, simulated profit, closing line value and calibration, with their uncertainty."""
from __future__ import annotations

import math

import pandas as pd

from sports_sniper import odds_math

DECIDED = ("won", "lost")
SETTLED = ("won", "lost", "push")


def clustered_se(values: pd.Series, clusters: pd.Series) -> float | None:
    """Standard error of a mean when bets on the same game move together: whole games are the draws."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return None
    deviation = values - values.mean()
    return float(math.sqrt((deviation.groupby(clusters.loc[values.index]).sum() ** 2).sum()) / len(values))


def summarize(bets: pd.DataFrame) -> dict:
    """One row of results. ROI is simulated profit over the stakes of settled bets; voids are refunds and count
    in neither. `closing_value` is the mean edge of the prices taken against the closing fair probability."""
    settled = bets[bets.status.isin(SETTLED)]
    decided = bets[bets.status.isin(DECIDED)]
    wins = int((bets.status == "won").sum())
    staked = float(settled.stake.sum())
    per_stake = settled.profit / settled.stake if len(settled) else pd.Series(dtype=float)
    closing = pd.to_numeric(bets.closing_line_value, errors="coerce").dropna()
    out = dict(bets=int(len(bets)), open=int((bets.status == "open").sum()), won=wins, lost=int((bets.status == "lost").sum()),
               push=int((bets.status == "push").sum()), void=int((bets.status == "void").sum()),
               win_rate=wins / len(decided) if len(decided) else None,
               win_interval=odds_math.wilson_interval(wins, len(decided)) if len(decided) else None,
               staked=staked, profit=float(settled.profit.sum()) if len(settled) else 0.0,
               roi=float(settled.profit.sum() / staked) if staked else None,
               roi_se=clustered_se(per_stake, settled.event_id) if len(settled) else None,
               estimated_ev=float(settled.ev.mean()) if len(settled) and settled.ev.notna().any() else None,
               average_odds=odds_math.decimal_to_american(float(settled.decimal.mean())) if len(settled) else None,
               closing_value=float(closing.mean()) if len(closing) else None, closing_count=int(len(closing)),
               closing_se=clustered_se(closing, bets.event_id) if len(closing) else None,
               beat_close=float((closing > 0).mean()) if len(closing) else None, brier=None, break_even_brier=None)
    scored = decided[decided.probability.notna()]
    if len(scored):
        outcome = (scored.status == "won").astype(int).tolist()
        out["brier"] = odds_math.brier_score(scored.probability.tolist(), outcome)
        out["break_even_brier"] = odds_math.brier_score((1 / scored.decimal).tolist(), outcome)
    return out


def breakdown(bets: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [dict({column: key}, **summarize(group)) for key, group in bets.groupby(column, sort=True)]
    return pd.DataFrame(rows)


def calibration(bets: pd.DataFrame, bins: int = 5) -> list[dict]:
    """Estimated chance against how often the bets actually won, by probability band."""
    scored = bets[bets.status.isin(DECIDED) & bets.probability.notna()]
    return odds_math.calibration_table(scored.probability.tolist(), (scored.status == "won").astype(int).tolist(), bins)
