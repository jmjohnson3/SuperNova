# NFL Projection Error

Built 2026-10-21T21:18:00+00:00. Season 2026, settled games only. One row per player-game-stat: the last forecast before kickoff, scored on the model's own number (not the line-anchored projection) against the official box score.

## Raw accuracy

| Stat | Player-games | Weeks | Actual mean | Projection mean | MAE | Bias | Above projection | Slope | Corr |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | 1005 | 2-6 | 32.03 | 28.45 | 20.80 | -3.58 | 45% | 0.91 | 0.53 |
| receptions | 586 | 4-6 | 3.01 | 2.79 | 1.62 | -0.22 | 48% | 0.80 | 0.53 |
| rushing_yards | 458 | 2-6 | 32.73 | 32.51 | 18.97 | -0.22 | 41% | 0.89 | 0.64 |
| passing_yards | 150 | 2-6 | 229.85 | 215.29 | 59.85 | -14.56 | 61% | 0.59 | 0.34 |

Bias is projection minus actual (positive = too high). The model forecasts a median, so on right-skewed stats a negative mean bias is expected; "Above projection" is the honest check (50% for a well-placed median). Slope regresses the outcome on the projection: below 1 means projections are spread too wide (too sure who will be high or low).

## Model vs the market line (rows with a line)

| Stat | Player-games | Model MAE | Line MAE | Line-anchored MAE | Model bias | Line bias |
|---|---:|---:|---:|---:|---:|---:|
| receiving_yards | 814 | 22.22 | 20.97 | 21.49 | -5.39 | -5.82 |
| receptions | 524 | 1.71 | 1.66 | 1.64 | -0.26 | -0.31 |
| rushing_yards | 378 | 19.51 | 17.67 | 18.92 | -0.65 | -0.22 |
| passing_yards | 148 | 60.33 | 54.57 | 57.37 | -15.08 | -7.39 |

## Volume vs efficiency

Error = volume part + efficiency part, exactly. Shares are of squared error; the interaction term is what remains when both miss in the same direction (positive) or offset (negative).

| Stat | Player-games | MAE | Knew volume | Knew efficiency | Volume share | Efficiency share | Interaction | Volume MAE | Volume bias | Implied eff. | Actual eff. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | 989 | 20.91 | 14.84 | 15.26 | 51% | 57% | -8% | 2.07 | -0.19 | 6.796 | 7.466 |
| receptions | 583 | 1.63 | 0.85 | 1.42 | 76% | 28% | -4% | 2.11 | -0.34 | 0.675 | 0.668 |
| rushing_yards | 447 | 19.02 | 11.60 | 14.07 | 53% | 44% | 3% | 3.37 | -0.03 | 4.079 | 4.131 |
| passing_yards | 150 | 59.85 | 48.57 | 57.25 | 109% | 67% | -76% | 8.42 | -3.04 | 7.305 | 6.952 |

"Knew volume" keeps the model's implied efficiency and uses the actual targets/carries/attempts; "Knew efficiency" keeps the pre-game volume estimate and uses the actual per-opportunity rate. The volume estimate is the 5-game average; implied efficiency is the projection divided by it.

## Pre-game volume estimates (identical rows)

| Volume | Estimate | Player-games | MAE | Bias |
|---|---|---:|---:|---:|
| targets | receiver_projected_targets_v3 | 989 | 2.15 | -0.33 |
| targets | targets_avg_5 | 989 | 2.07 | -0.19 |
| targets | targets_avg_10 | 989 | 2.06 | -0.19 |
| carries | rb_projected_carries_v3 | 451 | 4.28 | +0.37 |
| carries | carries_avg_5 | 451 | 3.34 | -0.04 |
| carries | carries_avg_10 | 451 | 3.38 | +0.09 |

## By position

| Stat | Position | Player-games | MAE | Bias | Slope | Volume share | Efficiency share | Volume bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| receiving_yards | RB | 248 | 13.04 | -1.52 | 0.61 | 41% | 52% | +0.06 |
| receiving_yards | TE | 258 | 18.63 | -1.05 | 0.82 | 49% | 48% | -0.12 |
| receiving_yards | WR | 499 | 25.77 | -5.91 | 0.84 | 52% | 61% | -0.35 |
| receptions | RB | 125 | 1.54 | -0.07 | 0.52 | 78% | 20% | -0.18 |
| receptions | TE | 162 | 1.53 | -0.15 | 0.81 | 69% | 29% | -0.25 |
| receptions | WR | 299 | 1.71 | -0.32 | 0.81 | 78% | 31% | -0.46 |
| rushing_yards | QB | 122 | 13.41 | +0.40 | 0.73 | 39% | 62% | +0.40 |
| rushing_yards | RB | 336 | 20.99 | -0.45 | 0.87 | 56% | 40% | -0.20 |
| passing_yards | QB | 150 | 59.85 | -14.56 | 0.59 | 109% | 67% | -3.04 |

## By projection quartile

| Stat | Projection range | Player-games | Projection mean | Actual mean | MAE | Bias | Above projection |
|---|---|---:|---:|---:|---:|---:|---:|
| receiving_yards | 5.0-13.5 | 252 | 9.01 | 13.60 | 12.06 | -4.59 | 45% |
| receiving_yards | 13.5-22.7 | 251 | 18.06 | 21.33 | 18.07 | -3.28 | 41% |
| receiving_yards | 22.7-38.5 | 251 | 30.56 | 33.76 | 21.95 | -3.20 | 46% |
| receiving_yards | 38.5-110.3 | 251 | 56.26 | 59.51 | 31.13 | -3.25 | 47% |
| receptions | 0.5-1.5 | 147 | 1.04 | 1.53 | 1.15 | -0.49 | 52% |
| receptions | 1.5-2.5 | 146 | 1.94 | 2.05 | 1.30 | -0.11 | 45% |
| receptions | 2.5-3.7 | 146 | 3.00 | 3.51 | 1.79 | -0.51 | 56% |
| receptions | 3.7-9.7 | 147 | 5.17 | 4.94 | 2.25 | +0.23 | 40% |
| rushing_yards | 5.1-14.6 | 115 | 9.70 | 11.60 | 9.16 | -1.90 | 39% |
| rushing_yards | 14.6-24.1 | 114 | 19.64 | 21.11 | 17.08 | -1.48 | 43% |
| rushing_yards | 24.1-47.9 | 114 | 32.94 | 33.56 | 21.66 | -0.62 | 40% |
| rushing_yards | 47.9-135.0 | 115 | 67.65 | 64.56 | 27.97 | +3.09 | 43% |
| passing_yards | 60.6-192.9 | 38 | 159.97 | 198.50 | 66.80 | -38.53 | 74% |
| passing_yards | 192.9-220.5 | 37 | 206.61 | 210.14 | 55.09 | -3.53 | 59% |
| passing_yards | 220.5-242.3 | 37 | 230.16 | 249.03 | 52.31 | -18.87 | 62% |
| passing_yards | 242.3-305.6 | 38 | 264.57 | 261.71 | 64.89 | +2.86 | 50% |

Measurement only: nothing here changes a model, a forecast or a bet.
