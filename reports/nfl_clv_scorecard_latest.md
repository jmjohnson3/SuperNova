# NFL CLV Scorecard

Built 2026-10-21T17:15:26.881405+00:00. Judge changes top to bottom: close coverage, CLV, probability accuracy vs FanDuel no-vig, then ROI. Player-games weighted equally.

| Stat | Version | Set | Player-games | Valid close | Beat close | Worse | Avg CLV (prob) | Brier model | Brier market | ROI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| passing_tds | market-calibration-20261001T030458Z | all_offers | 65 | 59.5% | 28.4% | 35.3% | -0.1% | 0.2319 | 0.2319 | -9.6% |
| passing_tds | market-calibration-20261001T030458Z | selected | 62 | 57.3% | 22.5% | 33.8% | +0.0% | 0.2368 | 0.2368 | -6.2% |
| passing_tds | market-calibration-20261001T030458Z | sharp-priced | all_offers | 30 | 100.0% | 32.4% | 35.2% | +0.0% | 0.2325 | 0.2325 | +1.1% |
| passing_tds | market-calibration-20261001T030458Z | sharp-priced | selected | 30 | 100.0% | 33.3% | 45.2% | -0.2% | 0.2303 | 0.2303 | +11.2% |
| passing_tds | pre-calibration | all_offers | 64 | 99.4% | 37.0% | 50.1% | -0.2% | 0.2408 | 0.2423 | +6.6% |
| passing_tds | pre-calibration | selected | 64 | 99.2% | 42.3% | 53.0% | -0.2% | 0.2398 | 0.2424 | +3.8% |
| passing_yards | market-calibration-20261001T030458Z | all_offers | 90 | 52.7% | 2.2% | 19.8% | -0.0% | 0.2500 | 0.2500 | -6.4% |
| passing_yards | market-calibration-20261001T030458Z | selected | 89 | 48.8% | 2.4% | 51.9% | -0.1% | 0.2500 | 0.2500 | -6.0% |
| passing_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 10 | 80.0% | 0.0% | 12.5% | -0.0% | 0.2500 | 0.2500 | -24.8% |
| passing_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 6 | 66.7% | 0.0% | 25.0% | -0.1% | 0.2500 | 0.2500 | -6.0% |
| passing_yards | pre-calibration | all_offers | 64 | 54.4% | 13.0% | 32.7% | -0.0% | 0.2543 | 0.2500 | -11.5% |
| passing_yards | pre-calibration | selected | 64 | 53.4% | 12.7% | 57.1% | -0.1% | 0.2545 | 0.2500 | -11.1% |
| receiving_yards | market-calibration-20261001T030458Z | all_offers | 488 | 71.4% | 2.2% | 44.2% | -0.1% | 0.2500 | 0.2500 | -6.0% |
| receiving_yards | market-calibration-20261001T030458Z | selected | 463 | 69.4% | 2.5% | 58.7% | -0.1% | 0.2500 | 0.2500 | -6.2% |
| receiving_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 459 | 85.7% | 2.0% | 4.3% | +0.0% | 0.2496 | 0.2500 | -3.4% |
| receiving_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 348 | 85.9% | 2.1% | 8.4% | -0.0% | 0.2502 | 0.2502 | -4.9% |
| receiving_yards | pre-calibration | all_offers | 338 | 63.1% | 11.8% | 37.7% | -0.1% | 0.2542 | 0.2500 | -2.1% |
| receiving_yards | pre-calibration | selected | 338 | 63.2% | 11.5% | 57.4% | -0.1% | 0.2540 | 0.2501 | -2.6% |
| receptions | market-calibration-20261001T030458Z | all_offers | 382 | 96.5% | 24.0% | 15.0% | +0.6% | 0.2486 | 0.2486 | -2.5% |
| receptions | market-calibration-20261001T030458Z | selected | 225 | 96.4% | 30.5% | 18.8% | +0.7% | 0.2502 | 0.2502 | -5.2% |
| receptions | market-calibration-20261001T030458Z | sharp-edge | all_offers | 2 | 100.0% | 0.0% | 0.0% | +0.0% | 0.3169 | 0.3044 | +26.0% |
| receptions | market-calibration-20261001T030458Z | sharp-edge | selected | 2 | 100.0% | 0.0% | 0.0% | +0.0% | 0.3169 | 0.3044 | +26.0% |
| receptions | market-calibration-20261001T030458Z | sharp-priced | all_offers | 478 | 96.1% | 10.6% | 8.4% | +0.1% | 0.2490 | 0.2484 | -5.2% |
| receptions | market-calibration-20261001T030458Z | sharp-priced | selected | 381 | 95.5% | 15.7% | 10.9% | +0.3% | 0.2484 | 0.2478 | -6.5% |
| rushing_yards | market-calibration-20261001T030458Z | all_offers | 226 | 68.1% | 0.6% | 42.0% | -0.1% | 0.2483 | 0.2501 | -1.1% |
| rushing_yards | market-calibration-20261001T030458Z | selected | 205 | 68.6% | 0.7% | 52.5% | -0.1% | 0.2485 | 0.2501 | -2.1% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-edge | all_offers | 1 | 0.0% | - | - | - | 0.3116 | 0.2500 | -100.0% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 189 | 81.5% | 1.0% | 8.7% | +0.0% | 0.2492 | 0.2499 | -7.2% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 108 | 76.4% | 0.0% | 22.0% | -0.1% | 0.2501 | 0.2500 | -16.0% |
| rushing_yards | pre-calibration | all_offers | 155 | 67.2% | 12.1% | 38.9% | -0.1% | 0.2664 | 0.2504 | -25.9% |
| rushing_yards | pre-calibration | selected | 155 | 68.0% | 12.0% | 58.2% | -0.1% | 0.2665 | 0.2505 | -27.1% |
