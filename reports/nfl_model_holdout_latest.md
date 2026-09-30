# NFL Active Release Evaluation

Release: nfl-20260918T143342Z

Fixed candidates evaluated on expanding 2025 week folds and a later 2026 veto window.
The 2026 window was previously examined during research; it is not an untouched prospective test.
Yardage/game Brier uses predeclared proxy lines, not executable sportsbook offers. TD Brier is P(any TD).

| Target | Live | OOF MAE / baseline | Later MAE / baseline | Later Brier / baseline | Later rows |
|---|---|---:|---:|---:|---:|
| passing_yards | accepted | 68.696 / 72.478 | 65.827 / 66.395 | 0.2413 / 0.2430 | 38 |
| rushing_yards | accepted | 17.797 / 18.455 | 18.317 / 19.435 | 0.2431 / 0.2631 | 112 |
| passing_tds | baseline | 0.911 / 0.945 | 0.952 / 0.953 | 0.1768 / 0.1624 | 38 |
| rushing_tds | P(any TD) head | 0.384 / 0.376 | 0.414 / 0.368 | 0.1609 / 0.1618 | 74 |
| receiving_yards | accepted | 16.630 / 17.537 | 17.181 / 17.974 | 0.2457 / 0.2513 | 275 |
| receiving_tds | P(any TD) head | 0.306 / 0.295 | 0.325 / 0.323 | 0.1462 / 0.1622 | 201 |
| home_margin | baseline | 10.105 / 9.670 | 10.391 / 10.882 | 0.2419 / 0.2465 | 17 |
| total_points_actual | baseline | 10.956 / 10.423 | 14.377 / 12.029 | 0.3106 / 0.2384 | 17 |

Legacy workload/spike artifacts are retained for research, but do not overwrite this release.
Forecast acceptance is not bankroll approval. Clean future locks, outcomes, and CLV are still required.
