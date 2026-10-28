# NFL Live Scoring Replay

Status: evaluated
Unique offer decisions: 543; settled: 534

## Replay Coverage

- matched: 543

## Probability Stages

| Cohort | Stage | Rows | Brier | Calibration error |
|---|---|---:|---:|---:|
| nfl-20260918T143342Z|passing_tds | raw | 40 | 0.2298 | 0.1348 |
| nfl-20260918T143342Z|passing_tds | heuristic | 40 | 0.2176 | 0.0688 |
| nfl-20260918T143342Z|passing_tds | post_exact | 40 | 0.2989 | 0.2656 |
| nfl-20260918T143342Z|passing_tds | final | 40 | 0.2164 | 0.1564 |
| nfl-20260918T143342Z|passing_tds | market | 40 | 0.2164 | 0.1564 |
| nfl-20260918T143342Z|passing_yards | raw | 36 | 0.2502 | 0.0460 |
| nfl-20260918T143342Z|passing_yards | heuristic | 36 | 0.2501 | 0.0445 |
| nfl-20260918T143342Z|passing_yards | post_exact | 36 | 0.2499 | 0.0425 |
| nfl-20260918T143342Z|passing_yards | final | 36 | 0.2500 | 0.0435 |
| nfl-20260918T143342Z|passing_yards | market | 36 | 0.2500 | 0.0435 |
| nfl-20260918T143342Z|receiving_yards | raw | 194 | 0.2498 | 0.0472 |
| nfl-20260918T143342Z|receiving_yards | heuristic | 194 | 0.2495 | 0.0480 |
| nfl-20260918T143342Z|receiving_yards | post_exact | 194 | 0.2496 | 0.0502 |
| nfl-20260918T143342Z|receiving_yards | final | 194 | 0.2493 | 0.0489 |
| nfl-20260918T143342Z|receiving_yards | market | 193 | 0.2493 | 0.0454 |
| nfl-20260918T143342Z|receptions | raw | 174 | 0.2367 | 0.0951 |
| nfl-20260918T143342Z|receptions | heuristic | 174 | 0.2367 | 0.0951 |
| nfl-20260918T143342Z|receptions | post_exact | 174 | 0.2455 | 0.0455 |
| nfl-20260918T143342Z|receptions | final | 174 | 0.2425 | 0.0580 |
| nfl-20260918T143342Z|receptions | market | 174 | 0.2425 | 0.0580 |
| nfl-20260918T143342Z|rushing_yards | raw | 90 | 0.2499 | 0.0099 |
| nfl-20260918T143342Z|rushing_yards | heuristic | 90 | 0.2501 | 0.0112 |
| nfl-20260918T143342Z|rushing_yards | post_exact | 90 | 0.2497 | 0.0139 |
| nfl-20260918T143342Z|rushing_yards | final | 90 | 0.2504 | 0.0127 |
| nfl-20260918T143342Z|rushing_yards | market | 90 | 0.2504 | 0.0127 |

- Missing old scoring inputs are never replaced with current context or calibration.
- Stored-probability evaluation is separate from exact reproducibility; inspect replay_status.
- Multiple books/lines share one player-game outcome and receive inverse multiplicity weights.
- Intervals without locked bounds cannot establish interval coverage.
- Locked p10/p90 describe the stat forecast, not a distribution inferred from independently adjusted book probabilities.

## Prospective Challengers

Status: evaluated
Settled cohort rows: 312; unique pending player-games: 41
Different challenger runs are compared separately, not pooled as independent outcomes.
