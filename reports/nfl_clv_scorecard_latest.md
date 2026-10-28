# NFL CLV Scorecard

Built 2026-10-27T20:38:35.451847+00:00. Judge changes top to bottom: close coverage, CLV, probability accuracy vs FanDuel no-vig, then ROI. Player-games weighted equally.

| Stat | Version | Set | Player-games | Valid close | Beat close | Worse | Avg CLV (prob) | Brier model | Brier market | ROI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| passing_tds | market-calibration-20261001T030458Z | all_offers | 65 | 59.5% | 28.4% | 35.3% | -0.1% | 0.2319 | 0.2319 | -9.6% |
| passing_tds | market-calibration-20261001T030458Z | selected | 62 | 57.3% | 22.5% | 33.8% | +0.0% | 0.2368 | 0.2368 | -6.2% |
| passing_tds | market-calibration-20261001T030458Z | sharp-priced | all_offers | 57 | 100.0% | 34.2% | 40.0% | -0.1% | 0.2226 | 0.2226 | -8.9% |
| passing_tds | market-calibration-20261001T030458Z | sharp-priced | selected | 57 | 100.0% | 34.2% | 48.5% | -0.2% | 0.2222 | 0.2222 | -0.7% |
| passing_tds | pre-calibration | all_offers | 64 | 99.4% | 37.0% | 50.1% | -0.2% | 0.2408 | 0.2423 | +6.6% |
| passing_tds | pre-calibration | selected | 64 | 99.2% | 42.3% | 53.0% | -0.2% | 0.2398 | 0.2424 | +3.8% |
| passing_yards | market-calibration-20261001T030458Z | all_offers | 116 | 55.8% | 1.6% | 25.3% | -0.0% | 0.2500 | 0.2500 | -2.4% |
| passing_yards | market-calibration-20261001T030458Z | selected | 111 | 50.1% | 1.8% | 57.2% | -0.1% | 0.2500 | 0.2500 | -4.3% |
| passing_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 22 | 76.1% | 0.0% | 22.1% | -0.0% | 0.2500 | 0.2500 | -14.5% |
| passing_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 11 | 72.7% | 0.0% | 50.0% | -0.1% | 0.2500 | 0.2500 | +2.5% |
| passing_yards | pre-calibration | all_offers | 64 | 54.4% | 13.0% | 32.7% | -0.0% | 0.2543 | 0.2500 | -11.5% |
| passing_yards | pre-calibration | selected | 64 | 53.4% | 12.7% | 57.1% | -0.1% | 0.2545 | 0.2500 | -11.1% |
| receiving_yards | market-calibration-20261001T030458Z | all_offers | 613 | 73.3% | 3.2% | 46.3% | -0.1% | 0.2499 | 0.2499 | -7.9% |
| receiving_yards | market-calibration-20261001T030458Z | selected | 570 | 70.9% | 3.2% | 60.2% | -0.1% | 0.2498 | 0.2498 | -8.2% |
| receiving_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 602 | 85.3% | 2.6% | 7.1% | +0.0% | 0.2496 | 0.2499 | -5.7% |
| receiving_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 427 | 83.2% | 2.5% | 14.4% | -0.0% | 0.2502 | 0.2502 | -7.4% |
| receiving_yards | pre-calibration | all_offers | 338 | 63.1% | 11.8% | 37.7% | -0.1% | 0.2542 | 0.2500 | -2.1% |
| receiving_yards | pre-calibration | selected | 338 | 63.2% | 11.5% | 57.4% | -0.1% | 0.2540 | 0.2501 | -2.6% |
| receptions | market-calibration-20261001T030458Z | all_offers | 407 | 96.0% | 23.7% | 15.3% | +0.5% | 0.2502 | 0.2502 | -0.4% |
| receptions | market-calibration-20261001T030458Z | selected | 237 | 95.8% | 30.2% | 19.4% | +0.6% | 0.2505 | 0.2505 | -2.9% |
| receptions | market-calibration-20261001T030458Z | sharp-edge | all_offers | 2 | 100.0% | 0.0% | 0.0% | +0.0% | 0.3169 | 0.3044 | +26.0% |
| receptions | market-calibration-20261001T030458Z | sharp-edge | selected | 2 | 100.0% | 0.0% | 0.0% | +0.0% | 0.3169 | 0.3044 | +26.0% |
| receptions | market-calibration-20261001T030458Z | sharp-priced | all_offers | 629 | 96.6% | 12.0% | 11.7% | +0.0% | 0.2466 | 0.2461 | -5.3% |
| receptions | market-calibration-20261001T030458Z | sharp-priced | selected | 502 | 96.5% | 16.4% | 14.4% | +0.1% | 0.2473 | 0.2468 | -5.6% |
| rushing_yards | market-calibration-20261001T030458Z | all_offers | 280 | 69.3% | 1.4% | 39.3% | -0.1% | 0.2488 | 0.2502 | -4.8% |
| rushing_yards | market-calibration-20261001T030458Z | selected | 253 | 68.8% | 1.7% | 52.6% | -0.1% | 0.2489 | 0.2502 | -7.0% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-edge | all_offers | 1 | 0.0% | - | - | - | 0.3116 | 0.2500 | -100.0% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-priced | all_offers | 242 | 81.1% | 2.8% | 15.9% | +0.0% | 0.2493 | 0.2498 | -9.3% |
| rushing_yards | market-calibration-20261001T030458Z | sharp-priced | selected | 148 | 76.4% | 3.1% | 33.8% | -0.0% | 0.2500 | 0.2499 | -18.3% |
| rushing_yards | pre-calibration | all_offers | 155 | 67.2% | 12.1% | 38.9% | -0.1% | 0.2664 | 0.2504 | -25.9% |
| rushing_yards | pre-calibration | selected | 155 | 68.0% | 12.0% | 58.2% | -0.1% | 0.2665 | 0.2505 | -27.1% |
