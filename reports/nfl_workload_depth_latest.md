# NFL Workload and Target Depth

Offline only. Production and betting gates unchanged.
Historical proxy lines are not real offered-line betting proof.
Reference: the conservative architecture refitted before each fold, not hindsight scoring with the current live artifact.

## Point Forecasts

| Stat | Mean RMSE | Reference RMSE | Median MAE | Reference MAE |
|---|---:|---:|---:|---:|
| receiving_yards | 24.461 | 24.962 | 17.022 | 17.217 |
| rushing_yards | 26.366 | 26.636 | 18.240 | 18.224 |

## Probability Curves

| Stat | Brier | Reference Brier | Nominal 80% Coverage | Probability Gate |
|---|---:|---:|---:|---|
| receiving_yards | 0.2311 | 0.2316 | 90.2% | False |
| rushing_yards | 0.2330 | 0.2389 | 93.1% | False |

## Component Checks

receiving_yards output gates: {"expected_mean": true, "median": true, "probability": false}
receiving_yards verified historical context coverage: {"wd_depth": 0.0, "wd_starter": 0.0, "wd_injury_out": 0.0, "wd_teammate_absences": 0.0}
Paired efficiency MAE: 4.229 versus 4.602; 8596 player-games.
Paired target-depth MAE: 4.190 versus 4.388.

rushing_yards output gates: {"expected_mean": false, "median": false, "probability": false}
rushing_yards verified historical context coverage: {"wd_depth": 0.0, "wd_starter": 0.0, "wd_injury_out": 0.0, "wd_teammate_absences": 0.0}
Paired efficiency MAE: 2.132 versus 2.309; 3788 player-games.

Missing availability stays unknown. No normalization over offered players or actual participants.
Historical point-forecast approval is separate from probability approval and does not enable real-money betting.
