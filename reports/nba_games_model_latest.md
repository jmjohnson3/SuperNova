# NBA Spread/Total Model vs DraftKings (pre-game lines)

Walk-forward; 3289 games with a DraftKings line taken before tip (seasons 2023-24, 2024-25, 2025-26).

| | Model MAE | Ratings only | DraftKings line MAE |
|---|---:|---:|---:|
| Margin | 10.96 | 11.13 | 10.67 |
| Total | 14.99 | 15.02 | 14.42 |

Model weight in a regression of the result on line and model: spread +0.30 (se 0.08), total -0.05 (se 0.08).

## Spread: model disagrees with DraftKings by k points

| k | Picks | Hit rate (52.4% breaks even) | By season |
|---:|---:|---:|---|
| 1 | 2527 | 51.4% | 2023-24: 51%, 2024-25: 53%, 2025-26: 50% |
| 2 | 1878 | 51.6% | 2023-24: 51%, 2024-25: 54%, 2025-26: 50% |
| 3 | 1310 | 51.8% | 2023-24: 50%, 2024-25: 56%, 2025-26: 50% |
| 4 | 828 | 51.7% | 2023-24: 52%, 2024-25: 53%, 2025-26: 50% |
| 5 | 488 | 53.5% | 2023-24: 52%, 2024-25: 57%, 2025-26: 52% |
| 7 | 152 | 57.9% | 2023-24: 53%, 2024-25: 56%, 2025-26: 62% |

## Total: model disagrees with DraftKings by k points

| k | Picks | Hit rate (52.4% breaks even) | By season |
|---:|---:|---:|---|
| 1 | 2718 | 49.3% | 2023-24: 48%, 2024-25: 49%, 2025-26: 50% |
| 2 | 2186 | 48.9% | 2023-24: 48%, 2024-25: 49%, 2025-26: 50% |
| 3 | 1713 | 48.6% | 2023-24: 47%, 2024-25: 48%, 2025-26: 51% |
| 4 | 1263 | 47.9% | 2023-24: 48%, 2024-25: 47%, 2025-26: 49% |
| 5 | 901 | 48.6% | 2023-24: 48%, 2024-25: 49%, 2025-26: 49% |
| 7 | 457 | 47.7% | 2023-24: 44%, 2024-25: 52%, 2025-26: 48% |
