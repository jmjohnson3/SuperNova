# NBA workload and probability research

reconstructed development evidence; outcome revisions and historical retrieval times are not proven
Fixed chronological fitting, dispersion, calibration and later evaluation periods. No production changes or betting approval.

| Forecast | Rows | MAE | Prior rolling baseline MAE | Bias |
|---|---:|---:|---:|---:|
| minutes | 27921 | 5.039 | 5.019439520818951 | +0.096 |
| pts | 27921 | 4.596 | 4.569350736354166 | +0.025 |
| reb | 27921 | 1.891 | 1.8844578464752713 | +0.028 |
| ast | 27921 | 1.344 | 1.3327400139411718 | -0.003 |
| fg3m | 27921 | 0.896 | 0.882079424280211 | +0.011 |
| fg3a | 27921 | 1.524 | 1.5033772321079468 | +0.003 |

| Book / stat | Offers | Market Brier | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|
| draftkings / ast | 1821 | 0.24699 | 0.25690 | 0.25568 |
| draftkings / pts | 2643 | 0.24936 | 0.25830 | 0.25594 |
| draftkings / reb | 2453 | 0.24893 | 0.25198 | 0.25111 |
| fanduel / ast | 1333 | 0.24700 | 0.25726 | 0.25616 |
| fanduel / pts | 2311 | 0.24934 | 0.25840 | 0.25605 |
| fanduel / reb | 2084 | 0.24928 | 0.25127 | 0.25048 |

Calibration uses fixed integer thresholds on unseen outcomes, not invented historical offers.
Threes have distribution calibration and stat-error results, but no historical real-offer validation.
Game forecast errors, paired uncertainty, calibration bins and chronological boundaries are in the JSON.
A future prospective comparison is required, even if reconstructed historical scores improve.
