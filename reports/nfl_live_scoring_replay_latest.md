# NFL Live Scoring Replay

Status: evaluated
Unique offer decisions: 599; settled: 583

## Replay Coverage

- matched: 599

## Probability Stages

| Cohort | Stage | Rows | Brier | Calibration error |
|---|---|---:|---:|---:|
| nfl-20260918T143342Z|passing_tds | raw | 38 | 0.2344 | 0.1427 |
| nfl-20260918T143342Z|passing_tds | heuristic | 38 | 0.2295 | 0.1378 |
| nfl-20260918T143342Z|passing_tds | post_exact | 38 | 0.2727 | 0.1026 |
| nfl-20260918T143342Z|passing_tds | final | 38 | 0.2346 | 0.1133 |
| nfl-20260918T143342Z|passing_tds | market | 38 | 0.2346 | 0.1133 |
| nfl-20260918T143342Z|passing_yards | raw | 46 | 0.2504 | 0.0719 |
| nfl-20260918T143342Z|passing_yards | heuristic | 46 | 0.2501 | 0.0705 |
| nfl-20260918T143342Z|passing_yards | post_exact | 46 | 0.2499 | 0.0684 |
| nfl-20260918T143342Z|passing_yards | final | 46 | 0.2500 | 0.0694 |
| nfl-20260918T143342Z|passing_yards | market | 46 | 0.2500 | 0.0694 |
| nfl-20260918T143342Z|receiving_yards | raw | 220 | 0.2498 | 0.0331 |
| nfl-20260918T143342Z|receiving_yards | heuristic | 220 | 0.2500 | 0.0375 |
| nfl-20260918T143342Z|receiving_yards | post_exact | 220 | 0.2505 | 0.0483 |
| nfl-20260918T143342Z|receiving_yards | final | 220 | 0.2502 | 0.0366 |
| nfl-20260918T143342Z|receiving_yards | market | 220 | 0.2502 | 0.0366 |
| nfl-20260918T143342Z|receptions | raw | 184 | 0.2498 | 0.0485 |
| nfl-20260918T143342Z|receptions | heuristic | 184 | 0.2498 | 0.0485 |
| nfl-20260918T143342Z|receptions | post_exact | 184 | 0.2646 | 0.0862 |
| nfl-20260918T143342Z|receptions | final | 184 | 0.2498 | 0.0502 |
| nfl-20260918T143342Z|receptions | market | 184 | 0.2498 | 0.0502 |
| nfl-20260918T143342Z|rushing_yards | raw | 95 | 0.2494 | 0.1136 |
| nfl-20260918T143342Z|rushing_yards | heuristic | 95 | 0.2497 | 0.1147 |
| nfl-20260918T143342Z|rushing_yards | post_exact | 95 | 0.2503 | 0.1174 |
| nfl-20260918T143342Z|rushing_yards | final | 95 | 0.2500 | 0.1161 |
| nfl-20260918T143342Z|rushing_yards | market | 95 | 0.2500 | 0.1161 |

- Missing old scoring inputs are never replaced with current context or calibration.
- Stored-probability evaluation is separate from exact reproducibility; inspect replay_status.
- Multiple books/lines share one player-game outcome and receive inverse multiplicity weights.
- Intervals without locked bounds cannot establish interval coverage.
- Locked p10/p90 describe the stat forecast, not a distribution inferred from independently adjusted book probabilities.

## Prospective Challengers

Status: evaluated
Settled cohort rows: 314; unique pending player-games: 37
Different challenger runs are compared separately, not pooled as independent outcomes.
