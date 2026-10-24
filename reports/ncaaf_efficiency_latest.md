# NCAAF Efficiency Model vs the Closing Line

Out of sample, seasons 2020-2025, 5066 games. Each season's model is trained only on earlier seasons; every rating is as of the week before the game.

| | Close MAE | Close + model MAE | Corr(pred gap, actual gap) | Pred gap SD |
|---|---:|---:|---:|---:|
| Spread | 12.216 | 12.213 | +0.037 | 0.52 |
| Total | 12.618 | 12.618 | +0.016 | 0.66 |

## Spread picks against the closing line

| Predicted gap | Picks | Hit rate | By season |
|---:|---:|---:|---|
| >= 0.5 | 1506 | 53.1% | 2020: 54%, 2021: 47%, 2022: 54%, 2023: 53%, 2024: 54%, 2025: 52% |
| >= 1 | 377 | 51.7% | 2022: 57%, 2023: 49%, 2024: 52%, 2025: 52% |
| >= 1.5 | 67 | 52.2% | 2023: 71%, 2024: 44%, 2025: 62% |
| >= 2 | 6 | 83.3% | 2024: 50%, 2025: 100% |
| >= 3 | 0 | - |  |
| >= 4 | 0 | - |  |

## Total picks against the closing line

| Predicted gap | Picks | Hit rate | By season |
|---:|---:|---:|---|
| >= 0.5 | 1573 | 51.4% | 2021: 51%, 2022: 52%, 2023: 40%, 2024: 57%, 2025: 51% |
| >= 1 | 764 | 50.5% | 2021: 48%, 2024: 71%, 2025: 52% |
| >= 1.5 | 360 | 50.8% | 2021: 47%, 2025: 54% |
| >= 2 | 132 | 54.5% | 2021: 56%, 2025: 53% |
| >= 3 | 10 | 70.0% | 2021: 75%, 2025: 50% |
| >= 4 | 3 | 100.0% | 2021: 100% |

## Spread: predicting the opener-to-close move (2023-2025, 2639 games, correlation +0.089)

| Predicted move | Picks | Avg CLV (pts) | Beat close | Win vs opener |
|---:|---:|---:|---:|---:|
| >= 0.25 | 867 | +0.36 | 47.5% | 48.9% |
| >= 0.5 | 184 | +0.31 | 47.8% | 48.6% |
| >= 1.0 | 3 | -0.58 | 33.3% | 33.3% |
| >= 1.5 | 0 | - | - | - |

## Total: predicting the opener-to-close move (2023-2025, 2639 games, correlation +0.145)

| Predicted move | Picks | Avg CLV (pts) | Beat close | Win vs opener |
|---:|---:|---:|---:|---:|
| >= 0.25 | 1809 | +0.48 | 54.1% | 50.9% |
| >= 0.5 | 1082 | +0.56 | 54.9% | 49.5% |
| >= 1.0 | 265 | +0.61 | 56.2% | 51.1% |
| >= 1.5 | 44 | +1.12 | 72.7% | 52.3% |
