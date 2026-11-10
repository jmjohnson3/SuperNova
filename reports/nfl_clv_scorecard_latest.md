# NFL CLV Scorecard

Built 2026-11-10T02:05:32.600618+00:00. Judge changes top to bottom: close coverage, CLV, probability accuracy vs FanDuel no-vig, then ROI. Player-games weighted equally.

| Stat | Version | Set | Player-games | Valid close | Beat close | Worse | Avg CLV (prob) | Brier model | Brier market | ROI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| passing_tds | market-calibration-20261001T030458Z | all_offers | 70 | 61.9% | 28.8% | 36.5% | -0.1% | 0.2319 | 0.2319 | -9.3% |
| passing_tds | market-calibration-20261001T030458Z | selected | 67 | 59.0% | 25.3% | 32.9% | +0.0% | 0.2396 | 0.2396 | -3.1% |
| passing_tds | market-calibration-20261001T030458Z | sharp-priced | all_offers | 109 | 98.2% | 36.4% | 44.1% | -0.1% | 0.2321 | 0.2321 | -8.4% |
| passing_tds | market-calibration-20261001T030458Z | sharp-priced | selected | 109 | 98.2% | 34.4% | 52.0% | -0.3% | 0.2313 | 0.2313 | -4.5% |
| passing_tds | pre-calibration | all_offers | 64 | 99.4% | 37.0% | 50.1% | -0.2% | 0.2408 | 0.2423 | +6.6% |
| passing_tds | pre-calibration | selected | 64 | 99.2% | 42.3% | 53.0% | -0.2% | 0.2398 | 0.2424 | +3.8% |
| passing_yards | market-calibration-20261001T030458Z | all_offers | 170 | 59.7% | 1.0% | 28.3% | -0.1% | 0.2500 | 0.2500 | -4.1% |
| passing_yards | market-calibration-20261001T030458Z | selected | 163 | 55.8% | 1.1% | 60.9% | -0.1% | 0.2500 | 0.2500 | -5.4% |
| passing_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 41 | 76.2% | 0.0% | 27.8% | -0.1% | 0.2500 | 0.2500 | -8.3% |
| passing_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 16 | 75.0% | 0.0% | 50.0% | -0.1% | 0.2500 | 0.2500 | +5.7% |
| passing_yards | pre-calibration | all_offers | 64 | 54.4% | 13.0% | 32.7% | -0.0% | 0.2543 | 0.2500 | -11.5% |
| passing_yards | pre-calibration | selected | 64 | 53.4% | 12.7% | 57.1% | -0.1% | 0.2545 | 0.2500 | -11.1% |
| receiving_yards | market-calibration-20261001T030458Z | all_offers | 893 | 72.9% | 3.9% | 45.1% | -0.1% | 0.2499 | 0.2499 | -7.3% |
| receiving_yards | market-calibration-20261001T030458Z | selected | 806 | 70.5% | 4.1% | 60.0% | -0.1% | 0.2499 | 0.2499 | -8.3% |
| receiving_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 881 | 83.4% | 3.0% | 10.9% | +0.0% | 0.2498 | 0.2500 | -6.2% |
| receiving_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 579 | 79.5% | 3.5% | 25.9% | +0.0% | 0.2501 | 0.2501 | -6.5% |
| receiving_yards | pre-calibration | all_offers | 338 | 63.1% | 11.8% | 37.7% | -0.1% | 0.2542 | 0.2500 | -2.1% |
| receiving_yards | pre-calibration | selected | 338 | 63.2% | 11.5% | 57.4% | -0.1% | 0.2540 | 0.2501 | -2.6% |
| receptions | market-calibration-20261001T030458Z | all_offers | 451 | 95.4% | 24.1% | 15.5% | +0.5% | 0.2514 | 0.2514 | -2.1% |
| receptions | market-calibration-20261001T030458Z | selected | 267 | 94.8% | 29.8% | 19.8% | +0.5% | 0.2506 | 0.2506 | -6.7% |
| receptions | market-calibration-20261001T030458Z | sharp-edge | all_offers | 3 | 100.0% | 0.0% | 33.3% | -0.1% | 0.3169 | 0.3044 | -16.0% |
| receptions | market-calibration-20261001T030458Z | sharp-edge | selected | 3 | 100.0% | 0.0% | 33.3% | -0.1% | 0.3169 | 0.3044 | -16.0% |
| receptions | market-calibration-20261001T030458Z | sharp-priced | all_offers | 942 | 95.5% | 15.7% | 14.0% | +0.1% | 0.2470 | 0.2467 | -5.3% |
| receptions | market-calibration-20261001T030458Z | sharp-priced | selected | 795 | 95.4% | 19.1% | 16.6% | +0.1% | 0.2481 | 0.2478 | -5.2% |
| rushing_yards | market-calibration-20261001T030458Z | all_offers | 394 | 70.7% | 1.8% | 38.1% | -0.1% | 0.2491 | 0.2501 | -9.8% |
| rushing_yards | market-calibration-20261001T030458Z | selected | 348 | 68.9% | 2.4% | 56.2% | -0.1% | 0.2492 | 0.2501 | -10.9% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-edge | all_offers | 1 | 0.0% | - | - | - | 0.3116 | 0.2500 | -100.0% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 342 | 76.5% | 3.4% | 19.9% | +0.0% | 0.2495 | 0.2498 | -7.9% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 228 | 71.4% | 3.7% | 40.2% | -0.0% | 0.2499 | 0.2499 | -12.0% |
| rushing_yards | pre-calibration | all_offers | 155 | 67.2% | 12.1% | 38.9% | -0.1% | 0.2664 | 0.2504 | -25.9% |
| rushing_yards | pre-calibration | selected | 155 | 68.0% | 12.0% | 58.2% | -0.1% | 0.2665 | 0.2505 | -27.1% |
