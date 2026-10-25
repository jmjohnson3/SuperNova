# NBA Props Model vs DraftKings (pre-game lines)

Walk-forward monthly refits; 58009 player-game-stat rows with a DraftKings line taken before tip (seasons 2023-24, 2024-25, 2025-26). Live-snapshot backfill rows are excluded.

All player-games (projection MAE): pts 4.53, reb 1.87, ast 1.32, fg3m 0.88

| Stat | Rows | Model MAE | Line MAE | Model weight in blend (se) | Log loss model | Log loss DK no-vig | Over rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| pts | 22148 | 5.20 | 5.07 | +0.14 (0.03) | 0.7139 | 0.6917 | 49.0% |
| reb | 21067 | 2.07 | 2.06 | +0.27 (0.03) | 0.7208 | 0.6885 | 47.6% |
| ast | 14794 | 1.78 | 1.78 | +0.32 (0.03) | 0.7238 | 0.6856 | 48.6% |

## All stats: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 58009 | 36604 / 21405 | 50.1% | 49.6% | -5.3% |
| 2% | 50140 | 32583 / 17557 | 50.1% | 49.5% | -5.3% |
| 4% | 42696 | 28606 / 14090 | 49.8% | 49.5% | -5.7% |
| 6% | 35516 | 24463 / 11053 | 50.0% | 49.5% | -5.3% |
| 8% | 29097 | 20546 / 8551 | 50.0% | 49.4% | -5.3% |
| 10% | 23379 | 16864 / 6515 | 50.0% | 49.4% | -5.1% |

## pts: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 22148 | 11461 / 10687 | 50.7% | 49.9% | -4.6% |
| 2% | 18927 | 9849 / 9078 | 51.1% | 49.9% | -3.9% |
| 4% | 15914 | 8361 / 7553 | 50.9% | 49.9% | -4.2% |
| 6% | 12928 | 6864 / 6064 | 51.1% | 49.8% | -3.8% |
| 8% | 10268 | 5472 / 4796 | 51.2% | 49.8% | -3.4% |
| 10% | 7949 | 4243 / 3706 | 51.4% | 49.8% | -3.0% |

## reb: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 21067 | 14076 / 6991 | 49.6% | 49.4% | -5.9% |
| 2% | 18116 | 12576 / 5540 | 49.3% | 49.4% | -6.4% |
| 4% | 15283 | 11033 / 4250 | 48.9% | 49.3% | -7.0% |
| 6% | 12618 | 9410 / 3208 | 49.0% | 49.2% | -6.7% |
| 8% | 10274 | 7910 / 2364 | 48.9% | 49.2% | -7.0% |
| 10% | 8240 | 6527 / 1713 | 49.0% | 49.2% | -6.8% |

## ast: model P(over) vs DraftKings no-vig

| Edge > | Picks | Over / Under | Hit rate | DK fair expected | ROI at DK price |
|---:|---:|---:|---:|---:|---:|
| 0% | 14794 | 11067 / 3727 | 49.9% | 49.4% | -5.4% |
| 2% | 13097 | 10158 / 2939 | 49.6% | 49.3% | -5.7% |
| 4% | 11499 | 9212 / 2287 | 49.5% | 49.3% | -5.9% |
| 6% | 9970 | 8189 / 1781 | 49.8% | 49.3% | -5.4% |
| 8% | 8555 | 7164 / 1391 | 49.7% | 49.2% | -5.4% |
| 10% | 7190 | 6094 / 1096 | 49.7% | 49.2% | -5.4% |

By season at edge > 4%:

- pts: 2023-24: 3470 picks, 51.6%, ROI -2.8%; 2024-25: 5267 picks, 50.5%, ROI -5.3%; 2025-26: 7177 picks, 50.8%, ROI -4.1%
- reb: 2023-24: 3487 picks, 50.7%, ROI -3.3%; 2024-25: 5118 picks, 47.6%, ROI -9.5%; 2025-26: 6678 picks, 48.9%, ROI -7.1%
- ast: 2023-24: 2504 picks, 51.6%, ROI -0.8%; 2024-25: 3823 picks, 48.9%, ROI -7.0%; 2025-26: 5172 picks, 49.0%, ROI -7.7%
