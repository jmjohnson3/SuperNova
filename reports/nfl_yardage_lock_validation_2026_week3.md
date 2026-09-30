# Locked Yardage Repair Validation

Latest archived Discord forecast per player-game/stat, not all generated offers.
Ranking tests use the archived card pool chronologically, not an assertion of actual selected wagers.
Original fixed research IDs are scored separately and never replaced.
Challenger artifact was trained retrospectively on pre-lock history; this is NOT a prospective capture.
Raw interval coverage is not final-CDF coverage. No deployment without coherent full-path validation.
Context counts describe observed/missing evidence, not player health.
Week 3 informed the repair hypothesis; it is diagnostic, not an untouched acceptance set.

## Archived Production

| Stat | Settled rows | Raw Brier | Final Brier | Market Brier |
|---|---:|---:|---:|---:|
| passing_yards | 24 | 0.3022 | 0.2572 | 0.2500 |
| receiving_yards | 82 | 0.2952 | 0.2490 | 0.2492 |
| rushing_yards | 53 | 0.3601 | 0.2733 | 0.2496 |

## Identical-Offer Replay

Only offers with both a challenger replay and valid market evidence appear below.

| Population | Paired rows | Production Brier | Challenger Brier | Market Brier |
|---|---:|---:|---:|---:|
| passing_yards | 0 | - | - | - |
| receiving_yards | 82 | 0.2490 | 0.2555 | 0.2492 |
| rushing_yards | 53 | 0.2733 | 0.2701 | 0.2496 |
| Fixed research selections | 7 | 0.2614 | 0.2626 | 0.2500 |

## Context and Error Components

| Stat | Rows | Injury unknown | True routes | Workload-dominated | Efficiency-dominated | Missing components |
|---|---:|---:|---:|---:|---:|---:|
| passing_yards | 24 | 20 | 0 | 0 | 0 | 24 |
| receiving_yards | 82 | 69 | 0 | 39 | 43 | 0 |
| rushing_yards | 54 | 41 | 0 | 22 | 31 | 1 |

Error components are descriptive accounting, not causal attribution.

## Fixed Research Record

Original selections: {"loss": 4, "win": 3}. These are not confirmed cash wagers.

## Counterfactual Ranking

Reselected from the archived card pool, not actual placed bets or independent proof.

| Ranking | Picks | Wins | Losses | Unsettled / other |
|---|---:|---:|---:|---:|
| ev | 15 | 9 | 6 | 0 |
| conservative | 12 | 5 | 7 | 0 |
| paired_current | 15 | 9 | 6 | 0 |
| challenger_conservative | 13 | 5 | 8 | 0 |

Full stage metrics, intervals, exclusions, and prediction IDs are in the companion JSON.

No production deployment or betting approval.
