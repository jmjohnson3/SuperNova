# NFL Shared Model Benchmark

Offline comparison. Production unchanged; no betting approval.
Same outer rows and nested dates for every family. Inner-week winners are evaluated on untouched later weeks.
Reference is a chronological refit of the conservative recipe, not the actual historical deployed artifact.
Probability scores below use proxy lines. Actual offered lines and original micro IDs are evaluated separately.

| Stat / family | Mean RMSE | Mean bias | Median MAE | Brier | 80% coverage |
|---|---:|---:|---:|---:|---:|
| receiving_yards / boosted_mean | 24.336 | -0.086 | 16.706 | 0.2287 | 79.9% |
| receiving_yards / boosted_mean+tail_cal | 24.357 | -0.389 | 16.786 | 0.2273 | 79.9% |
| receiving_yards / ensemble | 24.395 | +0.155 | 16.737 | 0.2262 | 83.7% |
| receiving_yards / ensemble+tail_cal | 24.389 | -0.022 | 16.720 | 0.2260 | 83.7% |
| receiving_yards / opportunity_rate | 24.565 | +0.054 | 17.041 | 0.2321 | 81.3% |
| receiving_yards / opportunity_rate+tail_cal | 24.665 | -0.902 | 17.150 | 0.2310 | 81.3% |
| receiving_yards / reference | 24.905 | -0.189 | 17.142 | 0.2308 | 80.3% |
| receiving_yards / reference+tail_cal | 24.908 | -0.642 | 17.145 | 0.2300 | 80.3% |
| receiving_yards / reference_conditional | 24.981 | +0.499 | 17.091 | 0.2282 | 79.5% |
| receiving_yards / reference_conditional+tail_cal | 24.967 | +0.633 | 17.097 | 0.2279 | 79.5% |
| receiving_yards / rolling | 25.698 | -0.223 | 17.870 | 0.2396 | 80.2% |
| receiving_yards / rolling+tail_cal | 25.835 | -2.564 | 17.983 | 0.2361 | 80.2% |
| receiving_yards / selected_expected | 24.352 | -0.055 | 16.747 | - | 80.6% |
| receiving_yards / selected_probability | 24.468 | -0.131 | 16.825 | 0.2270 | 81.0% |
| receiving_yards / selected_typical | 24.364 | -0.434 | 16.717 | - | 81.1% |
| receiving_yards / shrunk_rate | 24.935 | -0.285 | 17.257 | 0.2325 | 80.1% |
| receiving_yards / shrunk_rate+tail_cal | 24.973 | -1.273 | 17.294 | 0.2301 | 80.1% |
| receiving_yards / workload_depth | 24.460 | -0.246 | 16.944 | 0.2319 | 80.6% |
| receiving_yards / workload_depth+tail_cal | 24.520 | -1.048 | 16.992 | 0.2302 | 80.6% |
| rushing_yards / boosted_mean | 26.206 | -0.247 | 17.954 | 0.2333 | 78.3% |
| rushing_yards / boosted_mean+tail_cal | 26.198 | -0.248 | 17.951 | 0.2322 | 78.3% |
| rushing_yards / ensemble | 26.150 | -0.246 | 17.895 | 0.2300 | 83.3% |
| rushing_yards / ensemble+tail_cal | 26.144 | -0.384 | 17.896 | 0.2301 | 83.3% |
| rushing_yards / opportunity_rate | 26.484 | -1.217 | 18.277 | 0.2337 | 83.4% |
| rushing_yards / opportunity_rate+tail_cal | 26.482 | -1.236 | 18.309 | 0.2331 | 83.4% |
| rushing_yards / reference | 26.578 | -0.061 | 18.216 | 0.2349 | 78.6% |
| rushing_yards / reference+tail_cal | 26.606 | -0.907 | 18.320 | 0.2341 | 78.6% |
| rushing_yards / reference_conditional | 26.647 | +0.726 | 18.165 | 0.2327 | 78.9% |
| rushing_yards / reference_conditional+tail_cal | 26.607 | +0.307 | 18.171 | 0.2322 | 78.9% |
| rushing_yards / rolling | 27.218 | -0.102 | 18.810 | 0.2411 | 78.5% |
| rushing_yards / rolling+tail_cal | 27.284 | -1.892 | 19.070 | 0.2384 | 78.5% |
| rushing_yards / selected_expected | 26.379 | -0.633 | 18.124 | - | 80.5% |
| rushing_yards / selected_probability | 26.421 | -1.062 | 18.111 | 0.2304 | 81.2% |
| rushing_yards / selected_typical | 26.305 | -0.673 | 17.993 | - | 81.1% |
| rushing_yards / shrunk_rate | 26.696 | -0.257 | 18.359 | 0.2368 | 78.4% |
| rushing_yards / shrunk_rate+tail_cal | 26.736 | -1.380 | 18.439 | 0.2349 | 78.4% |
| rushing_yards / workload_depth | 26.455 | -2.093 | 18.207 | 0.2342 | 80.6% |
| rushing_yards / workload_depth+tail_cal | 26.451 | -2.087 | 18.374 | 0.2336 | 80.6% |
| passing_yards / boosted_mean | 82.489 | +3.042 | 65.786 | 0.1995 | 78.6% |
| passing_yards / boosted_mean+tail_cal | 82.466 | +2.071 | 65.837 | 0.1992 | 78.6% |
| passing_yards / ensemble | 83.114 | +3.031 | 65.918 | 0.1995 | 81.4% |
| passing_yards / ensemble+tail_cal | 83.100 | +1.872 | 65.934 | 0.1989 | 81.4% |
| passing_yards / opportunity_rate | 83.083 | +2.603 | 66.035 | 0.2009 | 82.6% |
| passing_yards / opportunity_rate+tail_cal | 83.119 | +1.315 | 65.898 | 0.2009 | 82.6% |
| passing_yards / qb_tail | 83.096 | +3.047 | 65.918 | 0.1994 | 81.4% |
| passing_yards / qb_tail+tail_cal | 83.083 | +1.888 | 65.934 | 0.1988 | 81.4% |
| passing_yards / reference | 86.080 | +3.116 | 67.620 | 0.2043 | 78.6% |
| passing_yards / reference+tail_cal | 86.083 | +2.453 | 67.692 | 0.2041 | 78.6% |
| passing_yards / reference_conditional | 86.061 | +3.446 | 67.630 | 0.2050 | 79.4% |
| passing_yards / reference_conditional+tail_cal | 86.073 | +2.681 | 67.699 | 0.2049 | 79.4% |
| passing_yards / rolling | 90.227 | +3.575 | 71.136 | 0.2168 | 78.9% |
| passing_yards / rolling+tail_cal | 90.212 | +1.852 | 71.084 | 0.2161 | 78.9% |
| passing_yards / selected_expected | 82.954 | +1.979 | 65.841 | - | 79.6% |
| passing_yards / selected_probability | 82.895 | +1.565 | 65.450 | 0.1992 | 80.5% |
| passing_yards / selected_typical | 83.196 | +1.714 | 66.223 | - | 80.8% |
| passing_yards / shrunk_rate | 89.557 | +3.375 | 70.731 | 0.2157 | 78.4% |
| passing_yards / shrunk_rate+tail_cal | 89.585 | +1.823 | 70.775 | 0.2154 | 78.4% |

## Next Prospective Choices

receiving_yards: {"expected": "reference_conditional", "typical": "reference_conditional", "probability": "reference+tail_cal", "probability_fallback": null}

rushing_yards: {"expected": "workload_depth", "typical": "ensemble", "probability": "workload_depth+tail_cal", "probability_fallback": null}

passing_yards: {"expected": "opportunity_rate", "typical": "ensemble", "probability": "opportunity_rate+tail_cal", "probability_fallback": null}

## Matched Fixed-Ensemble Tests

| Stat / comparison | Brier gain | Week-grouped 95% interval |
|---|---:|---|
| receiving_yards / ensemble_vs_selected_probability | +0.00081 | -2.9692393445064627e-05 to 0.0016067917558774678 |
| receiving_yards / ensemble+tail_cal_vs_selected_probability | +0.00098 | 0.0002913008373089221 to 0.0016359727599679933 |
| rushing_yards / ensemble_vs_selected_probability | +0.00043 | -0.0011310553875755625 to 0.0017716099890480723 |
| rushing_yards / ensemble+tail_cal_vs_selected_probability | +0.00025 | -0.001248153441813915 to 0.0016629043968725628 |
| passing_yards / ensemble_vs_selected_probability | -0.00025 | -0.0017945930781091003 to 0.0013385404887816354 |
| passing_yards / ensemble+tail_cal_vs_selected_probability | +0.00033 | -0.0011186431549637331 to 0.0017931830783115763 |
| passing_yards / qb_tail_vs_ensemble | +0.00006 | -2.8329390556126847e-05 to 0.00021937636944253095 |
| passing_yards / qb_tail+tail_cal_vs_ensemble+tail_cal | +0.00006 | -2.8329390556127304e-05 to 0.0002193763694425287 |

## Final Training Blocks

| Stat / block | Player-games | Weeks | Regular-season rows | Through |
|---|---:|---:|---:|---|
| receiving_yards / model_fit | 27023 | 134 | 25800 | 2025-10-20 |
| receiving_yards / scale | 1013 | 4 | 1013 | 2025-11-17 |
| receiving_yards / residual | 816 | 3 | 816 | 2025-12-08 |
| receiving_yards / cal_fit | 887 | 3 | 887 | 2025-12-29 |
| receiving_yards / cal_tune | 473 | 3 | 286 | 2026-01-18 |
| receiving_yards / selection_gate | 329 | 4 | 275 | 2026-09-17 |
| rushing_yards / model_fit | 11471 | 131 | 10955 | 2025-09-29 |
| rushing_yards / scale | 422 | 4 | 422 | 2025-10-27 |
| rushing_yards / residual | 318 | 3 | 318 | 2025-11-17 |
| rushing_yards / cal_fit | 324 | 3 | 324 | 2025-12-08 |
| rushing_yards / cal_tune | 360 | 3 | 360 | 2025-12-29 |
| rushing_yards / selection_gate | 330 | 7 | 240 | 2026-09-17 |
| passing_yards / model_fit | 3463 | 121 | 3328 | 2024-12-23 |
| passing_yards / scale | 218 | 9 | 185 | 2025-09-22 |
| passing_yards / residual | 196 | 6 | 196 | 2025-11-03 |
| passing_yards / cal_fit | 135 | 4 | 135 | 2025-12-01 |
| passing_yards / cal_tune | 110 | 3 | 110 | 2025-12-22 |
| passing_yards / selection_gate | 143 | 8 | 115 | 2026-09-17 |

receiving_yards final core refit through 2026-09-17.

rushing_yards final core refit through 2026-09-17.

passing_yards final core refit through 2026-09-17.

Core models are refitted after inner selection, before each outer test. Reported OOF losses include calibration-transfer risk.
Calibration blocks expand by whole weeks until both player-game and regular-season sample requirements pass.
Top rows in a descriptive leaderboard are not unbiased winners. Use selected_* policy results and week-grouped intervals.
Historical context coverage is incomplete. Missing information remains unknown.
Central calibration leaves tail probabilities and the 10th/90th percentiles unchanged. It cannot fix bad raw tails.
