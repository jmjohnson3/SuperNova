# NFL Rushing Volume Challenger (prospective)

Status: **collecting_evidence**. Registered 2026-10-22T15:25:26.980851+00:00 with volume run `volume-20261022T151821Z`; candidate = production model projection x model carries / 5-game average carries. Ready for a deployment review only after 3+ weeks with a 95% lower bound on the weekly MAE gain above zero. Nothing here changes a live forecast.

Settled player-games: 233 (captured 310; pending or did not play 77).

| Measure | Production | Challenger |
|---|---:|---:|
| Carries MAE (5-game average vs model) | 3.75 | 3.46 |
| Rushing-yard MAE | 23.24 | 22.59 |
| Bias (projection minus actual) | -2.14 | -3.32 |
| Share above projection | 46% | 48% |

Weekly MAE gain 0.67 yards (95% lower bound -0.61) over 3 week(s).

| Position | Player-games | Production MAE | Challenger MAE |
|---|---:|---:|---:|
| QB | 57 | 12.03 | 12.26 |
| RB | 176 | 26.88 | 25.93 |

Exclusions: earlier_capture_of_same_player_game 1352.
