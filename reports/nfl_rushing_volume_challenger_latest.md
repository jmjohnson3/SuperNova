# NFL Rushing Volume Challenger (prospective)

Status: **collecting_evidence**. Registered 2026-10-22T15:25:26.980851+00:00 with volume run `volume-20261022T151821Z`; candidate = production model projection x model carries / 5-game average carries. Ready for a deployment review only after 3+ weeks with a 95% lower bound on the weekly MAE gain above zero. Nothing here changes a live forecast.

Settled player-games: 78 (captured 109; pending or did not play 31).

| Measure | Production | Challenger |
|---|---:|---:|
| Carries MAE (5-game average vs model) | 3.81 | 3.43 |
| Rushing-yard MAE | 23.30 | 23.09 |
| Bias (projection minus actual) | -6.85 | -7.46 |
| Share above projection | 47% | 50% |

Weekly MAE gain 0.20 yards (95% lower bound -) over 1 week(s).

| Position | Player-games | Production MAE | Challenger MAE |
|---|---:|---:|---:|
| QB | 19 | 10.32 | 10.19 |
| RB | 59 | 27.48 | 27.25 |

Exclusions: earlier_capture_of_same_player_game 521.
