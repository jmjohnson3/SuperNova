# NFL Projection Placement

Held-out = each week scored by a correction fitted only on earlier weeks. A group is corrected only when held-out MAE improves at least 1% and the share above the projection moves toward 50%.

| Group | Rows | Held-out weeks | Identity MAE / above | Scale MAE / above | Linear MAE / above | Chosen | Parameters |
|---|---:|---|---|---|---|---|---|
| rushing_yards|QB | 122 | 3,4,5,6 | 13.53 / 34% | 13.13 / 56% | 13.13 / 49% | scale | {'k': 0.7513} |
| rushing_yards|RB | 336 | 3,4,5,6 | 20.83 / 46% | 21.10 / 58% | 20.99 / 58% | identity | - |
| passing_yards|QB | 150 | 3,4,5,6 | 57.29 / 62% | 56.66 / 55% | 56.21 / 52% | linear | {'a': 9.5859, 'b': 1.0} |

In-sample effect of the chosen correction (all weeks):

| Group | MAE before | MAE after | Above before | Above after | Bias before | Bias after |
|---|---:|---:|---:|---:|---:|---:|
| rushing_yards|QB | 13.41 | 12.73 | 34% | 51% | +0.40 | -4.14 |
| rushing_yards|RB | 20.99 | 20.99 | 44% | 44% | -0.45 | -0.45 |
| passing_yards|QB | 59.85 | 58.88 | 61% | 50% | -14.56 | -4.98 |

Installed to `models/player_props/placement_calibration.json`.
