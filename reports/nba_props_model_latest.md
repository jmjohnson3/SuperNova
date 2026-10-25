# NBA Props Model vs DraftKings (pre-game lines)

Walk-forward monthly refits; 58009 player-game-stat rows with a DraftKings line taken before tip (seasons 2023-24, 2024-25, 2025-26). Live-snapshot backfill rows are excluded.

All player-games (projection MAE): pts 4.44, reb 1.84, ast 1.31, fg3m 0.87

| Stat | Rows | Model MAE | Line MAE | Model weight in blend (se) | Log loss model | Log loss DK no-vig | Over rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| pts | 22148 | 5.14 | 5.07 | +0.27 (0.03) | 0.7074 | 0.6917 | 49.0% |
| reb | 21067 | 2.06 | 2.06 | +0.37 (0.03) | 0.7037 | 0.6885 | 47.6% |
| ast | 14794 | 1.78 | 1.78 | +0.39 (0.03) | 0.7251 | 0.6856 | 48.6% |

## All stats: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 58009 | 35025 / 22984 | 50.4% | 49.7% | -4.7% |
| 2% | 49421 | 30668 / 18753 | 50.6% | 49.6% | -4.3% |
| 4% | 41249 | 26408 / 14841 | 50.8% | 49.6% | -4.0% |
| 6% | 33498 | 22208 / 11290 | 50.8% | 49.5% | -3.8% |
| 8% | 26711 | 18377 / 8334 | 50.7% | 49.5% | -3.9% |
| 10% | 20792 | 14759 / 6033 | 51.0% | 49.4% | -3.2% |

## pts: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 22148 | 11266 / 10882 | 50.9% | 49.9% | -4.2% |
| 2% | 18752 | 9591 / 9161 | 51.3% | 49.9% | -3.4% |
| 4% | 15514 | 8012 / 7502 | 51.6% | 49.9% | -3.0% |
| 6% | 12385 | 6478 / 5907 | 51.7% | 49.9% | -2.7% |
| 8% | 9636 | 5140 / 4496 | 51.9% | 49.9% | -2.4% |
| 10% | 7274 | 3941 / 3333 | 52.6% | 49.8% | -0.9% |

## reb: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 21067 | 12071 / 8996 | 50.6% | 49.6% | -4.4% |
| 2% | 17385 | 10216 / 7169 | 50.6% | 49.5% | -4.1% |
| 4% | 13965 | 8437 / 5528 | 50.6% | 49.4% | -4.1% |
| 6% | 10823 | 6796 / 4027 | 50.8% | 49.4% | -3.6% |
| 8% | 8201 | 5359 / 2842 | 50.4% | 49.2% | -4.1% |
| 10% | 6040 | 4065 / 1975 | 50.0% | 49.1% | -4.6% |

## ast: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 14794 | 11688 / 3106 | 49.6% | 49.4% | -5.9% |
| 2% | 13284 | 10861 / 2423 | 49.6% | 49.3% | -5.8% |
| 4% | 11770 | 9959 / 1811 | 49.9% | 49.3% | -5.2% |
| 6% | 10290 | 8934 / 1356 | 49.8% | 49.2% | -5.3% |
| 8% | 8874 | 7878 / 996 | 49.8% | 49.2% | -5.4% |
| 10% | 7478 | 6753 / 725 | 50.2% | 49.2% | -4.4% |

By season at edge > 4%:

- pts: 2023-24: 3471 picks, 51.9%, ROI -2.2%; 2024-25: 5009 picks, 51.2%, ROI -4.0%; 2025-26: 7034 picks, 51.7%, ROI -2.6%
- reb: 2023-24: 3390 picks, 51.7%, ROI -1.7%; 2024-25: 4453 picks, 49.8%, ROI -5.7%; 2025-26: 6122 picks, 50.5%, ROI -4.2%
- ast: 2023-24: 2505 picks, 51.3%, ROI -1.5%; 2024-25: 3903 picks, 49.7%, ROI -5.3%; 2025-26: 5362 picks, 49.4%, ROI -6.7%
