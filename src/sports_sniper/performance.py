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


def checked_returns(bets: pd.DataFrame, checks: pd.DataFrame, profit) -> pd.DataFrame:
    """Paper bets whose price was confirmed at the book (first answer Got it), re-priced at the price actually
    seen. `profit(status, stake, decimal)` grades one bet. This is the demonstrated record: the same bets and
    results, but only those that could really have been placed, at the price that was really there."""
    columns = list(bets.columns) + ["checked_price"]
    if bets.empty or checks.empty:
        return pd.DataFrame(columns=columns)
    got = checks[checks["first"] & (checks.outcome == "got")]
    seen = {(c.event_id, c.book, c.market, c.player_key, c.line_key, c.side): int(c.flagged_price if pd.isna(c.price_seen) else c.price_seen)
            for c in got.itertuples()}
    keys = [(b.event_id, b.book, b.market, b.player_key, b.line_key, b.side) for b in bets.itertuples()]
    out = bets[[k in seen for k in keys]].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["checked_price"] = [seen[k] for k in keys if k in seen]
    out["decimal"] = out.checked_price.map(odds_math.american_to_decimal)
    out["ev"] = [None if pd.isna(p) else odds_math.expected_roi(p, d) for p, d in zip(out.probability, out.decimal)]
    out["profit"] = [profit(s, stake, d) if s != "open" else None for s, stake, d in zip(out.status, out.stake, out.decimal)]
    return out


def accuracy(bets: pd.DataFrame, column: str) -> pd.DataFrame:
    """Probability accuracy by group on decided bets: the estimate's Brier score beside the price's own break-even,
    with the average estimate against how often the bets won."""
    rows = []
    decided = bets[bets.status.isin(DECIDED) & bets.probability.notna()] if len(bets) else bets
    for key, group in (decided.groupby(column) if len(decided) else []):
        outcome = (group.status == "won").astype(int).tolist()
        wins = sum(outcome)
        rows.append({column: key, "bets": len(group), "estimate": float(group.probability.mean()), "won": wins / len(group),
                     "interval": odds_math.wilson_interval(wins, len(group)),
                     "brier": odds_math.brier_score(group.probability.tolist(), outcome),
                     "break_even_brier": odds_math.brier_score((1 / group.decimal).tolist(), outcome)})
    return pd.DataFrame(rows)


def availability(checks: pd.DataFrame) -> dict:
    """How often a flagged price was still there when it was first opened, from the Got it / Gone answers.
    Only each bet's first answer counts; `corrected` is how many were changed afterwards."""
    corrected = int((~checks["first"]).sum()) if len(checks) and "first" in checks else 0
    checks = checks[checks["first"]] if len(checks) and "first" in checks else checks
    n = int(len(checks))
    got = int((checks.outcome == "got").sum()) if n else 0
    changed = checks[(checks.outcome == "got") & checks.price_seen.notna() & (checks.price_seen != checks.flagged_price)] if n else checks
    return dict(checks=n, got=got, gone=n - got, got_rate=got / n if n else None,
                interval=odds_math.wilson_interval(got, n) if n else None,
                changed=int(len(changed)), worse=int((changed.price_seen < changed.flagged_price).sum()) if n else 0,
                corrected=corrected,
                median_seconds=float(checks.seconds_after.median()) if n else None)


def breakdown(bets: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [dict({column: key}, **summarize(group)) for key, group in bets.groupby(column, sort=True)]
    return pd.DataFrame(rows)


def calibration(bets: pd.DataFrame, bins: int = 5) -> list[dict]:
    """Estimated chance against how often the bets actually won, by probability band."""
    scored = bets[bets.status.isin(DECIDED) & bets.probability.notna()]
    return odds_math.calibration_table(scored.probability.tolist(), (scored.status == "won").astype(int).tolist(), bins)
