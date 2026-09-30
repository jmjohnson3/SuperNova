# MLB Hitter Player-Rate Diagnostic

Generated UTC: 2026-09-01T11:06:35Z
Evidence: one row per player-game; this report cannot replace production models.

## Conservative Player-Rate Challengers

| Stat | Accepted | Source | Reason | Rows | Dates | Alpha | Base MAE | Raw MAE | Selected MAE | Base Any Brier | Raw Any Brier | Selected Any Brier | Selected Rate Bias |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hits | True | lightgbm_rate | mae_and_any_brier_improved | 10854 | 57 | 0.50 | 0.716 | 0.674 | 0.691 | 0.24481 | 0.23107 | 0.23473 | +0.01368 |
| Home Runs | True | lightgbm_rate | mae_and_any_brier_improved | 10504 | 56 | 0.50 | 0.265 | 0.220 | 0.243 | 0.10815 | 0.10444 | 0.10505 | +0.00890 |

| Stat | Source | Alpha | MAE | MAE Gain | Bias | Any Brier | Brier Gain | Rate Bias | Mean Shift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hits | lightgbm_rate | 0.00 | 0.716 | +0.000 | -0.011 | 0.24481 | +0.00000 | +0.00997 | +0.0000 |
| Hits | lightgbm_rate | 0.05 | 0.714 | +0.003 | -0.010 | 0.24348 | +0.00133 | +0.01034 | +0.0010 |
| Hits | lightgbm_rate | 0.10 | 0.711 | +0.006 | -0.009 | 0.24223 | +0.00258 | +0.01071 | +0.0021 |
| Hits | lightgbm_rate | 0.20 | 0.706 | +0.011 | -0.007 | 0.23995 | +0.00486 | +0.01145 | +0.0042 |
| Hits | lightgbm_rate | 0.35 | 0.698 | +0.019 | -0.004 | 0.23704 | +0.00776 | +0.01256 | +0.0073 |
| Hits | lightgbm_rate | 0.50 | 0.691 | +0.026 | -0.001 | 0.23473 | +0.01008 | +0.01368 | +0.0105 |
| Hits | empirical_bayes_residual | 0.00 | 0.719 | +0.000 | -0.005 | 0.24635 | +0.00000 | +0.01154 | +0.0000 |
| Hits | empirical_bayes_residual | 0.05 | 0.719 | -0.000 | -0.003 | 0.24636 | -0.00000 | +0.01200 | +0.0018 |
| Hits | empirical_bayes_residual | 0.10 | 0.720 | -0.000 | -0.002 | 0.24637 | -0.00001 | +0.01246 | +0.0036 |
| Hits | empirical_bayes_residual | 0.20 | 0.720 | -0.001 | +0.002 | 0.24640 | -0.00005 | +0.01338 | +0.0072 |
| Hits | empirical_bayes_residual | 0.35 | 0.721 | -0.002 | +0.007 | 0.24649 | -0.00013 | +0.01477 | +0.0127 |
| Hits | empirical_bayes_residual | 0.50 | 0.722 | -0.002 | +0.013 | 0.24661 | -0.00026 | +0.01616 | +0.0181 |
| Home Runs | lightgbm_rate | 0.00 | 0.265 | +0.000 | +0.054 | 0.10815 | +0.00000 | +0.01675 | +0.0000 |
| Home Runs | lightgbm_rate | 0.05 | 0.263 | +0.002 | +0.051 | 0.10775 | +0.00040 | +0.01597 | -0.0030 |
| Home Runs | lightgbm_rate | 0.10 | 0.260 | +0.004 | +0.048 | 0.10736 | +0.00079 | +0.01518 | -0.0061 |
| Home Runs | lightgbm_rate | 0.20 | 0.256 | +0.009 | +0.042 | 0.10665 | +0.00150 | +0.01361 | -0.0122 |
| Home Runs | lightgbm_rate | 0.35 | 0.249 | +0.016 | +0.033 | 0.10575 | +0.00240 | +0.01125 | -0.0213 |
| Home Runs | lightgbm_rate | 0.50 | 0.243 | +0.022 | +0.024 | 0.10505 | +0.00310 | +0.00890 | -0.0305 |
| Home Runs | empirical_bayes_residual | 0.00 | 0.269 | +0.000 | +0.064 | 0.10738 | +0.00000 | +0.01919 | +0.0000 |
| Home Runs | empirical_bayes_residual | 0.05 | 0.268 | +0.001 | +0.062 | 0.10724 | +0.00014 | +0.01878 | -0.0015 |
| Home Runs | empirical_bayes_residual | 0.10 | 0.267 | +0.002 | +0.061 | 0.10711 | +0.00027 | +0.01837 | -0.0031 |
| Home Runs | empirical_bayes_residual | 0.20 | 0.264 | +0.005 | +0.058 | 0.10686 | +0.00052 | +0.01756 | -0.0061 |
| Home Runs | empirical_bayes_residual | 0.35 | 0.261 | +0.008 | +0.053 | 0.10654 | +0.00085 | +0.01633 | -0.0107 |
| Home Runs | empirical_bayes_residual | 0.50 | 0.257 | +0.012 | +0.049 | 0.10627 | +0.00112 | +0.01511 | -0.0153 |

## Hits Bias Calibration V2

This is a player-game residual repair by lineup slot, projected PA, handedness, player prior, park, and run environment.

| Accepted | Reason | Rows | Dates | Alpha | Base MAE | Selected MAE | MAE Gain | Base Brier | Selected Brier | Brier Gain | Selected Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| False | mae_not_improved,any_brier_not_improved | 8997 | 47 | 0.00 | 0.719 | 0.719 | +0.000 | 0.24635 | 0.24635 | +0.00000 | -0.005 |

| Source | Alpha | MAE | MAE Gain | Bias | Any Brier | Brier Gain | Rate Bias | Mean Shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hit_bias_calibration_v2 | 0.00 | 0.719 | +0.000 | -0.005 | 0.24635 | +0.00000 | +0.01154 | +0.0000 |
| hit_bias_calibration_v2 | 0.15 | 0.720 | -0.000 | -0.003 | 0.24649 | -0.00014 | +0.01222 | +0.0027 |
| hit_bias_calibration_v2 | 0.25 | 0.720 | -0.001 | -0.001 | 0.24659 | -0.00024 | +0.01267 | +0.0045 |
| hit_bias_calibration_v2 | 0.35 | 0.721 | -0.001 | +0.001 | 0.24670 | -0.00035 | +0.01313 | +0.0063 |
| hit_bias_calibration_v2 | 0.50 | 0.721 | -0.002 | +0.004 | 0.24688 | -0.00052 | +0.01381 | +0.0089 |
| hit_bias_calibration_v2 | 0.65 | 0.722 | -0.002 | +0.006 | 0.24707 | -0.00071 | +0.01449 | +0.0116 |

## Hits

Overall rows: 10703; MAE: 0.7161890176803378; count bias: -0.021625766999825685; rate bias: 0.009966536009596903

| Feature | Bucket | Rows | Dates | Count MAE | Count Bias | Pred/PA | Actual/PA | Rate Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pa_outcome_bucket | actual_pa_0_2 | 821 | 56 | 0.667 | +0.507 | 0.2182 | 0.1620 | +0.0562 |
| home_away_bucket | away | 5373 | 57 | 0.727 | -0.011 | 0.2270 | 0.2087 | +0.0184 |
| barrel_bucket | barrel_high | 3364 | 57 | 0.721 | +0.013 | 0.2247 | 0.2077 | +0.0170 |
| lineup_bucket | slot_3_5 | 3659 | 57 | 0.721 | -0.010 | 0.2268 | 0.2122 | +0.0146 |
| xslg_bucket | xslg_average | 6923 | 57 | 0.715 | -0.006 | 0.2255 | 0.2112 | +0.0143 |
| xslg_bucket | xslg_high | 948 | 57 | 0.738 | +0.010 | 0.2335 | 0.2197 | +0.0138 |
| starter_quality_bucket | starter_weak | 1025 | 50 | 0.749 | -0.014 | 0.2383 | 0.2254 | +0.0128 |
| starter_quality_bucket | starter_average | 5828 | 57 | 0.725 | -0.013 | 0.2268 | 0.2142 | +0.0127 |
| platoon_bucket | same_hand | 4133 | 57 | 0.711 | +0.004 | 0.2203 | 0.2090 | +0.0113 |
| barrel_bucket | barrel_average | 4221 | 57 | 0.712 | -0.022 | 0.2234 | 0.2122 | +0.0112 |
| lineup_bucket | slot_1_2 | 2443 | 57 | 0.752 | -0.027 | 0.2384 | 0.2276 | +0.0108 |
| platoon_bucket | opposite_hand | 4976 | 57 | 0.728 | -0.039 | 0.2284 | 0.2182 | +0.0102 |
| starter_quality_bucket | starter_strong | 2853 | 56 | 0.684 | -0.019 | 0.2130 | 0.2053 | +0.0077 |
| pa_outcome_bucket | actual_pa_3_plus | 9882 | 57 | 0.720 | -0.066 | 0.2242 | 0.2181 | +0.0061 |
| lineup_bucket | slot_6_9 | 4601 | 57 | 0.693 | -0.028 | 0.2135 | 0.2077 | +0.0058 |
| platoon_bucket | unknown_hand | 1594 | 57 | 0.692 | -0.034 | 0.2180 | 0.2124 | +0.0056 |
| barrel_bucket | barrel_low | 1878 | 57 | 0.711 | -0.053 | 0.2272 | 0.2247 | +0.0025 |
| starter_quality_bucket | starter_quality_missing | 997 | 51 | 0.721 | -0.090 | 0.2213 | 0.2237 | -0.0024 |
| barrel_bucket | barrel_missing | 1240 | 56 | 0.724 | -0.069 | 0.2171 | 0.2192 | -0.0021 |
| xslg_bucket | xslg_missing | 1240 | 56 | 0.724 | -0.069 | 0.2171 | 0.2192 | -0.0021 |
| xslg_bucket | xslg_low | 1592 | 56 | 0.705 | -0.070 | 0.2156 | 0.2171 | -0.0015 |
| home_away_bucket | home | 5330 | 57 | 0.706 | -0.032 | 0.2204 | 0.2189 | +0.0015 |

### Hits Error Decomposition

Miss decomposition: count miss = PA/opportunity component + per-PA hit-rate component.

Rows: 10703; count bias: -0.022; rate bias: +0.00997; PA component bias: -0.036; rate component bias: +0.014

Top-order under-projection rows: 826; high-PA under-projection rows: 485; high-prior under-projection rows: 67; missing-context rows: 1700

| Component | Rows | Count MAE | Count Bias | Pred/PA | Actual/PA | Rate Bias | PA Bias | Rate Bias Component | Under-Proj Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| player_rate_error | 8236 | 0.860 | -0.037 | 0.2176 | 0.2116 | +0.00598 | -0.043 | +0.006 | 0.418 |
| small_error | 1561 | 0.076 | -0.025 | 0.2477 | 0.2505 | -0.00284 | -0.058 | +0.033 | 0.000 |
| pa_error | 906 | 0.509 | +0.125 | 0.2386 | 0.1704 | +0.06823 | +0.064 | +0.060 | 0.300 |

| Slice | Bucket | Rows | Count Bias | Rate Bias | PA Bias | Rate Component Bias | Under-Proj Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| player_prior_hit_bucket | prior_hit_rate_high | 171 | -0.072 | -0.00219 | -0.055 | -0.017 | 0.392 |
| projected_pa_bucket | projected_pa_low_under_3_8 | 3809 | -0.048 | +0.01619 | -0.075 | +0.027 | 0.380 |
| lineup_detail_bucket | bottom_order_6_9 | 4601 | -0.028 | +0.00585 | -0.018 | -0.010 | 0.371 |
| handedness_bucket | handedness_missing | 1594 | -0.034 | +0.00560 | -0.038 | +0.004 | 0.355 |
| handedness_bucket | opposite_hand | 4976 | -0.039 | +0.01023 | -0.053 | +0.015 | 0.350 |
| run_environment_bucket | team_total_high | 4630 | -0.038 | +0.00458 | -0.031 | -0.006 | 0.350 |
| run_environment_bucket | team_total_mid | 3846 | -0.019 | +0.01145 | -0.037 | +0.018 | 0.350 |
| player_prior_hit_bucket | prior_hit_rate_low | 7568 | -0.025 | +0.00942 | -0.035 | +0.011 | 0.348 |
| park_hit_bucket | hit_park_suppress | 10703 | -0.022 | +0.00997 | -0.036 | +0.014 | 0.347 |
| run_environment_bucket | team_total_low | 2097 | -0.015 | +0.01323 | -0.044 | +0.029 | 0.342 |
| player_prior_hit_bucket | prior_hit_rate_mid | 2964 | -0.011 | +0.01206 | -0.036 | +0.025 | 0.341 |
| handedness_bucket | same_hand | 4133 | +0.004 | +0.01134 | -0.014 | +0.017 | 0.340 |
| lineup_detail_bucket | top_order_1_2 | 2443 | -0.027 | +0.01080 | -0.054 | +0.027 | 0.338 |
| projected_pa_bucket | projected_pa_high_4_4_plus | 1472 | -0.001 | -0.00558 | +0.032 | -0.033 | 0.329 |
| projected_pa_bucket | projected_pa_mid_3_8_4_4 | 5422 | -0.008 | +0.00982 | -0.027 | +0.018 | 0.328 |
| lineup_detail_bucket | middle_order_3_5 | 3659 | -0.010 | +0.01459 | -0.046 | +0.036 | 0.323 |
| run_environment_bucket | team_total_missing | 130 | +0.358 | +0.10528 | -0.030 | +0.388 | 0.231 |

| Player | Date | Slot | Proj PA | Actual PA | Pred H | Actual H | Error | Prior H/PA | Prior Pred H | Dominant | Context |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Hunter Goodman | 2026-06-14 | 4 | 4.30 | 6.00 | 0.88 | 5 | -4.12 | 0.2279 | 0.98 | player_rate_error | middle_order_3_5, projected_pa_mid_3_8_4_4, prior_hit_rate_low, opposite_hand |
| Carson Benge | 2026-06-07 | 1 | 4.30 | 5.00 | 0.95 | 5 | -4.05 | 0.2247 | 0.97 | player_rate_error | top_order_1_2, projected_pa_mid_3_8_4_4, prior_hit_rate_low, opposite_hand |
| Drake Baldwin | 2026-07-19 | 1 | 4.10 | 5.00 | 1.37 | 5 | -3.63 | 0.2308 | 0.95 | player_rate_error | top_order_1_2, projected_pa_mid_3_8_4_4, prior_hit_rate_low, opposite_hand |
| Jackson Holliday | 2026-07-08 | 9 | 3.00 | 4.00 | 0.39 | 4 | -3.61 | 0.1860 | 0.56 | player_rate_error | bottom_order_6_9, projected_pa_low_under_3_8, prior_hit_rate_low, opposite_hand |
| Dalton Rushing | 2026-07-02 | 9 | 3.32 | 5.00 | 0.43 | 4 | -3.56 | 0.2246 | 0.75 | player_rate_error | bottom_order_6_9, projected_pa_low_under_3_8, prior_hit_rate_low, opposite_hand |
| Fernando Tatis Jr. | 2026-07-28 | 1 | 4.60 | 5.00 | 1.48 | 5 | -3.52 | 0.2457 | 1.13 | player_rate_error | top_order_1_2, projected_pa_high_4_4_plus, prior_hit_rate_mid, same_hand |
| Victor Robles | 2026-07-20 | 8 | 3.49 | 4.00 | 0.55 | 4 | -3.45 | 0.2058 | 0.72 | player_rate_error | bottom_order_6_9, projected_pa_low_under_3_8, prior_hit_rate_low, opposite_hand |
| Kyle Stowers | 2026-07-03 | 3 | 4.00 | 5.00 | 0.57 | 4 | -3.43 | 0.2074 | 0.83 | player_rate_error | middle_order_3_5, projected_pa_mid_3_8_4_4, prior_hit_rate_low, opposite_hand |
| Jake McCarthy | 2026-07-03 | 1 | 3.65 | 5.00 | 0.57 | 4 | -3.43 | 0.2487 | 0.91 | player_rate_error | top_order_1_2, projected_pa_low_under_3_8, prior_hit_rate_mid, opposite_hand |
| Kyle Tucker | 2026-07-02 | 7 | 3.85 | 5.00 | 0.60 | 4 | -3.40 | 0.2050 | 0.79 | player_rate_error | bottom_order_6_9, projected_pa_mid_3_8_4_4, prior_hit_rate_low, opposite_hand |
| A.J. Ewing | 2026-07-07 | 1 | 3.50 | 5.00 | 0.61 | 4 | -3.39 | 0.2255 | 0.79 | player_rate_error | top_order_1_2, projected_pa_low_under_3_8, prior_hit_rate_low, handedness_missing |
| Nick Fortes | 2026-07-24 | 9 | 2.82 | 4.00 | 0.62 | 4 | -3.38 | 0.2466 | 0.70 | player_rate_error | bottom_order_6_9, projected_pa_low_under_3_8, prior_hit_rate_mid, opposite_hand |
| Ezequiel Duran | 2026-06-07 | 5 | 3.70 | 5.00 | 0.63 | 4 | -3.37 | 0.2447 | 0.91 | player_rate_error | middle_order_3_5, projected_pa_low_under_3_8, prior_hit_rate_mid, opposite_hand |
| Yandy Díaz | 2026-07-08 | 1 | 4.10 | 4.00 | 0.64 | 4 | -3.36 | 0.2797 | 1.15 | player_rate_error | top_order_1_2, projected_pa_mid_3_8_4_4, prior_hit_rate_mid, same_hand |
| Eduardo Valencia | 2026-07-28 | 8 | 2.50 | 5.00 | 0.71 | 4 | -3.29 | 0.2473 | 0.62 | player_rate_error | bottom_order_6_9, projected_pa_low_under_3_8, prior_hit_rate_mid, handedness_missing |

## Home Runs

Overall rows: 10370; MAE: 0.26652473802635807; count bias: 0.05338851173256188; rate bias: 0.01675062589691759

| Feature | Bucket | Rows | Dates | Count MAE | Count Bias | Pred/PA | Actual/PA | Rate Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pa_outcome_bucket | actual_pa_0_2 | 760 | 55 | 0.173 | +0.128 | 0.0441 | 0.0158 | +0.0283 |
| xslg_bucket | xslg_high | 930 | 56 | 0.361 | +0.076 | 0.0635 | 0.0427 | +0.0208 |
| barrel_bucket | barrel_high | 3279 | 56 | 0.344 | +0.058 | 0.0610 | 0.0427 | +0.0182 |
| lineup_bucket | slot_3_5 | 3553 | 56 | 0.302 | +0.061 | 0.0541 | 0.0360 | +0.0181 |
| barrel_bucket | barrel_average | 4088 | 56 | 0.264 | +0.058 | 0.0479 | 0.0299 | +0.0180 |
| platoon_bucket | opposite_hand | 4834 | 56 | 0.282 | +0.053 | 0.0514 | 0.0336 | +0.0178 |
| starter_quality_bucket | starter_quality_missing | 962 | 50 | 0.256 | +0.047 | 0.0459 | 0.0282 | +0.0176 |
| xslg_bucket | xslg_average | 6714 | 56 | 0.282 | +0.057 | 0.0509 | 0.0333 | +0.0176 |
| home_away_bucket | away | 5220 | 56 | 0.267 | +0.054 | 0.0482 | 0.0307 | +0.0176 |
| starter_quality_bucket | starter_average | 5647 | 56 | 0.271 | +0.054 | 0.0490 | 0.0322 | +0.0169 |
| starter_quality_bucket | starter_weak | 984 | 49 | 0.279 | +0.052 | 0.0509 | 0.0341 | +0.0168 |
| lineup_bucket | slot_6_9 | 4428 | 56 | 0.212 | +0.049 | 0.0411 | 0.0244 | +0.0167 |
| platoon_bucket | same_hand | 3993 | 56 | 0.268 | +0.057 | 0.0476 | 0.0312 | +0.0164 |
| starter_quality_bucket | starter_strong | 2777 | 55 | 0.257 | +0.054 | 0.0460 | 0.0299 | +0.0162 |
| barrel_bucket | barrel_missing | 1197 | 55 | 0.231 | +0.051 | 0.0426 | 0.0265 | +0.0161 |
| xslg_bucket | xslg_missing | 1197 | 55 | 0.231 | +0.051 | 0.0426 | 0.0265 | +0.0161 |
| home_away_bucket | home | 5150 | 56 | 0.266 | +0.053 | 0.0480 | 0.0321 | +0.0159 |
| pa_outcome_bucket | actual_pa_3_plus | 9610 | 56 | 0.274 | +0.047 | 0.0484 | 0.0326 | +0.0158 |
| lineup_bucket | slot_1_2 | 2389 | 56 | 0.314 | +0.050 | 0.0522 | 0.0374 | +0.0148 |
| platoon_bucket | unknown_hand | 1543 | 56 | 0.215 | +0.046 | 0.0391 | 0.0247 | +0.0144 |
| barrel_bucket | barrel_low | 1806 | 56 | 0.155 | +0.035 | 0.0289 | 0.0172 | +0.0116 |
| xslg_bucket | xslg_low | 1529 | 55 | 0.167 | +0.027 | 0.0310 | 0.0200 | +0.0110 |

