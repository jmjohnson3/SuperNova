# NFL Volume Model (challenger)

Run `volume-20261022T151821Z`. Walk-forward 4-week blocks over seasons 2023, 2024, 2025, 2026; each block trained only on earlier games, fixed hyperparameters. Players who played. Never changes a live forecast.

Serving parity: 191 features match their archived live values on 1293 player-games and are used; 172 are excluded (live_differs 99, not_in_live_snapshot 33, too_few_comparable_rows 28, live_coverage 12).

## targets: ACCEPTED on volume

16658 player-games where every baseline exists. Best simple baseline: `targets_avg_10`.

| Estimate | MAE | RMSE | Bias |
|---|---:|---:|---:|
| model | 1.651 | 2.242 | -0.046 |
| targets_avg_3 | 1.766 | 2.420 | +0.024 |
| targets_avg_5 | 1.724 | 2.341 | +0.036 |
| targets_avg_10 | 1.719 | 2.306 | +0.077 |
| receiver_projected_targets_v3 | 1.752 | 2.478 | -0.644 |

Weekly MAE difference, model minus `targets_avg_10`: -0.059 (95% interval -0.075 to -0.043, 72 weeks; negative = model better).

| Season | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| 2023 | 1.686 | 1.753 | -3.9% |
| 2024 | 1.705 | 1.769 | -3.6% |
| 2025 | 1.543 | 1.628 | -5.2% |
| 2026 | 1.753 | 1.777 | -1.4% |

| Position | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| RB | 1.319 | 1.363 | -3.2% |
| TE | 1.519 | 1.573 | -3.4% |
| WR | 1.923 | 2.012 | -4.4% |

Checks: beats_best_mae yes, beats_best_rmse yes, weekly_interval_excludes_zero yes, no_season_worse_than_2pct yes.

## carries: ACCEPTED on volume

7088 player-games where every baseline exists. Best simple baseline: `carries_avg_5`.

| Estimate | MAE | RMSE | Bias |
|---|---:|---:|---:|
| model | 3.003 | 4.153 | -0.157 |
| carries_avg_3 | 3.167 | 4.405 | +0.039 |
| carries_avg_5 | 3.132 | 4.344 | +0.063 |
| carries_avg_10 | 3.158 | 4.362 | +0.115 |
| rb_projected_carries_v3 | 3.431 | 4.876 | -0.368 |

Weekly MAE difference, model minus `carries_avg_5`: -0.133 (95% interval -0.185 to -0.081, 72 weeks; negative = model better).

| Season | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| 2023 | 3.108 | 3.244 | -4.2% |
| 2024 | 3.053 | 3.198 | -4.5% |
| 2025 | 2.883 | 2.970 | -2.9% |
| 2026 | 2.933 | 3.133 | -6.4% |

| Position | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| QB | 1.784 | 1.892 | -5.8% |
| RB | 3.569 | 3.707 | -3.7% |

Checks: beats_best_mae yes, beats_best_rmse yes, weekly_interval_excludes_zero yes, no_season_worse_than_2pct yes.

## Does it help yards? (live 2026 settled forecasts)

Rescaled: the live projection moved from the 5-game-average volume to the model's out-of-sample volume. Volume x efficiency: model volume times the player's last-10 per-opportunity rate, shrunk toward his position's rate. Same settled rows as the production number.

| Stat | Rows | Production MAE | Rescaled MAE | Volume x efficiency MAE | Above: production / rescaled / built |
|---|---:|---:|---:|---:|---|
| receiving_yards | 989 | 20.91 | 21.38 | 21.22 | 45% / 49% / 46% |
| receptions | 583 | 1.63 | 1.69 | 1.68 | 49% / 53% / 52% |
| rushing_yards | 447 | 19.02 | 18.16 | 18.29 | 42% / 47% / 44% |

Promotion is a manual decision recorded in a doc; this report changes nothing live.
