# NFL Volume Model (challenger)

Run `volume-20261022T025415Z`. Walk-forward 4-week blocks over seasons 2023, 2024, 2025, 2026; each block trained only on earlier games, fixed hyperparameters. Players who played. Never changes a live forecast.

## targets: ACCEPTED on volume

16658 player-games where every baseline exists. Best simple baseline: `targets_avg_10`.

| Estimate | MAE | RMSE | Bias |
|---|---:|---:|---:|
| model | 1.643 | 2.230 | -0.013 |
| targets_avg_3 | 1.766 | 2.420 | +0.024 |
| targets_avg_5 | 1.724 | 2.341 | +0.036 |
| targets_avg_10 | 1.719 | 2.306 | +0.077 |
| receiver_projected_targets_v3 | 1.752 | 2.478 | -0.644 |

Weekly MAE difference, model minus `targets_avg_10`: -0.067 (95% interval -0.083 to -0.051, 72 weeks; negative = model better).

| Season | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| 2023 | 1.684 | 1.753 | -3.9% |
| 2024 | 1.695 | 1.769 | -4.2% |
| 2025 | 1.537 | 1.628 | -5.6% |
| 2026 | 1.730 | 1.777 | -2.6% |

| Position | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| RB | 1.307 | 1.363 | -4.1% |
| TE | 1.523 | 1.573 | -3.2% |
| WR | 1.913 | 2.012 | -5.0% |

Checks: beats_best_mae yes, beats_best_rmse yes, weekly_interval_excludes_zero yes, no_season_worse_than_2pct yes.

## carries: ACCEPTED on volume

7088 player-games where every baseline exists. Best simple baseline: `carries_avg_5`.

| Estimate | MAE | RMSE | Bias |
|---|---:|---:|---:|
| model | 2.929 | 4.034 | -0.103 |
| carries_avg_3 | 3.167 | 4.405 | +0.039 |
| carries_avg_5 | 3.132 | 4.344 | +0.063 |
| carries_avg_10 | 3.158 | 4.362 | +0.115 |
| rb_projected_carries_v3 | 3.431 | 4.876 | -0.368 |

Weekly MAE difference, model minus `carries_avg_5`: -0.212 (95% interval -0.264 to -0.160, 72 weeks; negative = model better).

| Season | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| 2023 | 3.062 | 3.244 | -5.6% |
| 2024 | 2.983 | 3.198 | -6.7% |
| 2025 | 2.777 | 2.970 | -6.5% |
| 2026 | 2.868 | 3.133 | -8.5% |

| Position | Model MAE | Baseline MAE | Change |
|---|---:|---:|---:|
| QB | 1.772 | 1.892 | -6.4% |
| RB | 3.466 | 3.707 | -6.5% |

Checks: beats_best_mae yes, beats_best_rmse yes, weekly_interval_excludes_zero yes, no_season_worse_than_2pct yes.

## Does it help yards? (live 2026 settled forecasts)

Rescaled: the live projection moved from the 5-game-average volume to the model's out-of-sample volume. Volume x efficiency: model volume times the player's last-10 per-opportunity rate, shrunk toward his position's rate. Same settled rows as the production number.

| Stat | Rows | Production MAE | Rescaled MAE | Volume x efficiency MAE | Above: production / rescaled / built |
|---|---:|---:|---:|---:|---|
| receiving_yards | 989 | 20.91 | 21.30 | 21.12 | 45% / 48% / 45% |
| receptions | 583 | 1.63 | 1.67 | 1.66 | 49% / 53% / 51% |
| rushing_yards | 447 | 19.02 | 18.03 | 18.28 | 42% / 45% / 42% |

Promotion is a manual decision recorded in a doc; this report changes nothing live.
