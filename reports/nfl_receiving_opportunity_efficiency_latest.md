# NFL Receiving Opportunity + Efficiency v2

Candidate: receiving-oe-v2-20261012T234225243219Z
Frozen production: nfl-20260918T143342Z
Training through: 2026-09-28

This is a forecast shadow candidate, not a betting approval.

## Historical chronological screen

| Output | Candidate | Reference | Pass |
|---|---:|---:|---|
| Expected yards RMSE | 23.967 | 24.524 | True |
| Typical outcome MAE | 16.303 | 16.773 | True |
| Proxy-line Brier | 0.2244 | 0.2312 | True |
| Calibration error | 0.86% | 3.10% | True |
| Nominal 80% coverage | 82.55% | 80.00% | True |

## What changed

- Team-allocation weight selected chronologically: 25%.
- The remaining opportunity weight comes from the player's stable pregame role history.
- Receiving efficiency is estimated separately with 50% empirical shrinkage.
- Tail calibration is mean preserving, so uncertainty cannot silently inflate expected yards.
- Optional probability calibration enabled: False.
- Optional tail repair enabled: False.
- The prospective scorer now uses the exact validated distribution path.

## Prospective status

- Registered outputs: expected_mean, median, probability.
- Superseded shadow candidates: components-20261010T222543607583Z-receiving_conditional, components-20261010T222543607583Z-team_receiving_yards.
- Acceptance starts after this registration. Previously reviewed games are not reused as prospective proof.
- Required next evidence: identical FanDuel lines, at least 100 player-games and 3 independent weeks, plus selected-pick validation.
- Production changed: no.
- Betting approved: no.
