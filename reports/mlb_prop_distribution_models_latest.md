# MLB Prop Distribution Models

Generated UTC: 2026-06-19T16:09:52Z
Rows: 78285
Date range: 2026-05-31 to 2026-06-18
Status: ready

## Overall Holdout

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 2128 | 0.174 | 0.513 | -2.2% | 595 | -22.7% | 53.0% |
| market_no_vig | 2128 | 0.168 | 0.498 | -0.8% | 178 | 64.4% | 61.7% |
| distribution | 2128 | 0.177 | 0.516 | 1.4% | 364 | -32.1% | 47.4% |
| distribution_calibrated | 2128 | 0.174 | 0.508 | 0.7% | 384 | -23.5% | 50.2% |
| distribution_empirical_blend | 2128 | 0.174 | 0.509 | 0.2% | 367 | -32.6% | 51.4% |
| event_side_line | 2128 | 0.197 | 0.593 | -6.5% | 1039 | -21.1% | 46.2% |

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases | 896 | 0.168 | 0.166 | 0.163 | 0.163 | 0.172 | -15.9% | 38.7% | -15.7% |
| batter_hits | 780 | 0.209 | 0.202 | 0.201 | 0.203 | 0.203 | -12.1% | -13.8% | 17.3% |
| batter_home_runs | 358 | 0.060 | 0.063 | 0.063 | 0.061 | 0.176 | -61.3% | -64.0% | -56.4% |
| pitcher_strikeouts | 94 | 0.248 | 0.302 | 0.288 | 0.288 | 0.293 | - | -51.0% | -43.9% |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|
| batter_total_bases / over / TB 1.5 | 214 | 0.001 | 6.6% | 1 | 100.0% | 384.0% | False | brier_gain<=0.001, abs_calibration_error>0.05, selected_rows<30 |
| batter_total_bases / over / TB 2.5+ | 52 | 0.025 | 18.7% | 32 | 41.7% | 97.9% | False | rows<80, abs_calibration_error>0.05, clv_beat_rate<0.55 |
| batter_total_bases / under / TB 1.5 | 90 | 0.005 | -3.6% | 38 | 16.6% | -93.4% | False | clv_beat_rate<0.55, avg_clv_price<=0 |

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 5994 | 24081.0 | 1.038 | 1.083 | 0.952 | 1.705 | 0.223 | 0.215 | 0.385 | 0.355 | 0.038 | 0.040 |
| power=power_high | 1562 | 6243.0 | 1.067 | 1.017 | 0.788 | 1.465 | 0.231 | 0.215 | 0.413 | 0.406 | 0.043 | 0.056 |
| power=power_low | 1218 | 4790.0 | 0.974 | 1.033 | 0.880 | 1.819 | 0.220 | 0.227 | 0.329 | 0.318 | 0.020 | 0.023 |
| power=power_mid | 3214 | 13048.0 | 1.047 | 1.133 | 1.103 | 1.656 | 0.221 | 0.211 | 0.393 | 0.345 | 0.042 | 0.038 |
| slot=slot_bottom | 1682 | 5891.0 | 1.025 | 1.011 | 0.763 | 1.553 | 0.213 | 0.207 | 0.343 | 0.339 | 0.024 | 0.034 |
| slot=slot_middle | 1266 | 4977.0 | 0.987 | 1.055 | 1.026 | 1.394 | 0.215 | 0.218 | 0.380 | 0.358 | 0.040 | 0.039 |
| slot=slot_top | 3046 | 13213.0 | 1.061 | 1.119 | 1.004 | 1.791 | 0.232 | 0.218 | 0.406 | 0.362 | 0.043 | 0.042 |
| slot_power=slot_bottom|power_high | 390 | 1387.0 | 1.060 | 0.945 | 0.758 | 1.126 | 0.221 | 0.204 | 0.363 | 0.392 | 0.029 | 0.050 |
| slot_power=slot_bottom|power_low | 478 | 1704.0 | 1.036 | 1.066 | 0.881 | 1.536 | 0.228 | 0.218 | 0.329 | 0.303 | 0.016 | 0.021 |
| slot_power=slot_bottom|power_mid | 814 | 2800.0 | 0.989 | 1.019 | 0.869 | 1.491 | 0.200 | 0.202 | 0.342 | 0.335 | 0.027 | 0.033 |
| slot_power=slot_middle|power_high | 337 | 1323.0 | 1.015 | 0.994 | 0.881 | 1.214 | 0.231 | 0.226 | 0.419 | 0.422 | 0.045 | 0.058 |
| slot_power=slot_middle|power_low | 305 | 1190.0 | 0.940 | 0.987 | 0.977 | 1.241 | 0.208 | 0.228 | 0.316 | 0.322 | 0.022 | 0.023 |
| slot_power=slot_middle|power_mid | 624 | 2464.0 | 1.001 | 1.118 | 1.172 | 1.289 | 0.209 | 0.209 | 0.389 | 0.342 | 0.046 | 0.036 |
| slot_power=slot_top|power_high | 835 | 3533.0 | 1.077 | 1.055 | 0.848 | 1.549 | 0.235 | 0.216 | 0.431 | 0.406 | 0.047 | 0.058 |
| slot_power=slot_top|power_low | 435 | 1896.0 | 0.954 | 1.024 | 0.922 | 1.667 | 0.222 | 0.235 | 0.338 | 0.328 | 0.022 | 0.025 |
| slot_power=slot_top|power_mid | 1776 | 7784.0 | 1.079 | 1.167 | 1.138 | 1.667 | 0.233 | 0.214 | 0.412 | 0.350 | 0.046 | 0.039 |

## Direct Hitter Event Model

- Status: loaded
- Method: hierarchical_conditional_lgbm
- Trained UTC: 2026-06-19T16:03:19.672390+00:00
- Classes: out, walk, single, double, triple, hr
- Production gate: False
- Production eligible artifact: False
- Leakage-safe player priors: 778 players
- PA uncertainty groups: 6
- Direct event TB MAE gain vs independent rates: -0.020
- Explicit TB-state rows: 7437
- Explicit TB-state Brier: 0.690
- Explicit TB-state log loss: 1.329
- Direct-state selected candidate: hierarchical_hr_tail
- Direct-state blend alpha: 0.250
- HR-driven 4+ tail Brier gain: 0.000158

## True-Pair Hitter Line Calibration

- Status: trained
- Evidence: temporal_train_true_pair_non_synthetic_only
- Calibrated line/side groups: 6
- Enabled line/side groups: 1
- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.
- `batter_total_bases|over|TB 1.5`: 7063 rows, method=raw, internal_gain=-0.000, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|over|TB 2.5`: 584 rows, method=beta, internal_gain=0.028, holdout_gain=0.104, cal_before=41.7%, cal_after=26.2%, enabled=True
- `batter_total_bases|over|TB 3.5`: 136 rows, method=beta, internal_gain=0.016, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|over|TB 4.5`: 375 rows, method=beta, internal_gain=0.162, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|under|TB 1.5`: 3350 rows, method=beta, internal_gain=0.001, holdout_gain=0.000, cal_before=-3.6%, cal_after=-3.6%, enabled=False
- `batter_total_bases|under|TB 2.5`: 38 rows, method=raw, internal_gain=-, holdout_gain=-, cal_before=-, cal_after=-, enabled=False

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 35270 | 1080 | 0.247 | 0.250 | 49.1% | 46.4% |
| clv_beat_probability | trained | 31533 | 980 | 0.251 | 0.263 | 38.7% | 46.7% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 6488 | 26026.0 | 0.927 | 2.180 | 0.710 | 0.953 | 1.068 | 38.4% | 38.9% | 0.040 | 0.018 | 0.037 | 0.039 |
| pa=pa_high | 1932 | 8832.0 | 0.945 | 2.440 | 0.862 | 1.092 | 1.179 | 28.6% | 33.1% | 0.043 | 0.015 | 0.046 | 0.042 |
| pa=pa_low | 903 | 2829.0 | 0.869 | 1.579 | 0.914 | 0.739 | 0.912 | 52.9% | 49.1% | 0.045 | 0.021 | 0.019 | 0.033 |
| pa=pa_mid | 3653 | 14365.0 | 0.935 | 1.878 | 0.707 | 0.916 | 1.025 | 39.9% | 39.4% | 0.038 | 0.019 | 0.035 | 0.039 |
| pa_power=pa_high|power_high | 504 | 2282.0 | 1.054 | 1.607 | 1.011 | 0.916 | 1.106 | 29.0% | 34.4% | 0.055 | 0.024 | 0.051 | 0.059 |
| pa_power=pa_high|power_low | 312 | 1409.0 | 0.879 | 1.370 | 1.027 | 1.016 | 1.069 | 30.4% | 29.8% | 0.037 | 0.008 | 0.024 | 0.023 |
| pa_power=pa_high|power_mid | 1116 | 5141.0 | 0.942 | 2.103 | 0.788 | 1.213 | 1.216 | 28.0% | 33.4% | 0.039 | 0.012 | 0.050 | 0.039 |
| pa_power=pa_low|power_high | 211 | 660.0 | 1.088 | 1.025 | 0.960 | 0.894 | 0.919 | 46.9% | 48.4% | 0.044 | 0.040 | 0.032 | 0.053 |
| pa_power=pa_low|power_low | 275 | 872.0 | 0.959 | 1.315 | 1.018 | 0.943 | 1.092 | 45.1% | 47.9% | 0.048 | 0.011 | 0.014 | 0.019 |
| pa_power=pa_low|power_mid | 417 | 1297.0 | 0.742 | 1.361 | 0.952 | 0.807 | 0.836 | 61.2% | 50.2% | 0.042 | 0.019 | 0.017 | 0.033 |
| pa_power=pa_mid|power_high | 1076 | 4219.0 | 1.089 | 1.318 | 0.835 | 0.752 | 0.931 | 40.7% | 40.3% | 0.046 | 0.032 | 0.037 | 0.055 |
| pa_power=pa_mid|power_low | 779 | 3062.0 | 0.827 | 1.706 | 0.797 | 0.975 | 0.986 | 41.7% | 37.3% | 0.035 | 0.009 | 0.020 | 0.021 |
| pa_power=pa_mid|power_mid | 1798 | 7084.0 | 0.932 | 1.893 | 0.865 | 1.082 | 1.109 | 38.7% | 39.8% | 0.035 | 0.015 | 0.040 | 0.037 |
| power=power_high | 1791 | 7161.0 | 1.095 | 1.490 | 0.812 | 0.773 | 0.983 | 38.1% | 39.6% | 0.049 | 0.030 | 0.041 | 0.056 |
| power=power_low | 1366 | 5343.0 | 0.838 | 1.979 | 0.873 | 0.962 | 1.040 | 39.8% | 37.7% | 0.038 | 0.009 | 0.020 | 0.021 |
| power=power_mid | 3331 | 13522.0 | 0.906 | 2.297 | 0.746 | 1.112 | 1.128 | 37.9% | 39.0% | 0.037 | 0.014 | 0.042 | 0.037 |
| slot=slot_bottom | 1826 | 6367.0 | 0.950 | 1.878 | 0.846 | 0.799 | 1.016 | 45.3% | 45.4% | 0.047 | 0.022 | 0.025 | 0.033 |
| slot=slot_middle | 1367 | 5373.0 | 0.892 | 1.554 | 0.881 | 1.016 | 1.032 | 40.7% | 39.0% | 0.033 | 0.018 | 0.039 | 0.038 |
| slot=slot_top | 3295 | 14286.0 | 0.938 | 2.308 | 0.718 | 0.998 | 1.099 | 33.5% | 35.2% | 0.040 | 0.016 | 0.042 | 0.042 |
| slot_power=slot_bottom|power_high | 450 | 1577.0 | 1.182 | 1.121 | 0.938 | 0.802 | 0.945 | 44.7% | 45.7% | 0.051 | 0.041 | 0.030 | 0.050 |
| slot_power=slot_bottom|power_low | 543 | 1923.0 | 0.931 | 1.545 | 0.940 | 0.925 | 1.069 | 42.2% | 43.5% | 0.042 | 0.010 | 0.016 | 0.019 |
| slot_power=slot_bottom|power_mid | 833 | 2867.0 | 0.866 | 1.812 | 0.938 | 0.896 | 1.029 | 47.7% | 46.4% | 0.050 | 0.019 | 0.028 | 0.033 |
| slot_power=slot_middle|power_high | 375 | 1472.0 | 1.023 | 1.166 | 0.976 | 0.883 | 0.966 | 38.9% | 38.6% | 0.043 | 0.031 | 0.043 | 0.057 |
| slot_power=slot_middle|power_low | 340 | 1322.0 | 0.867 | 1.391 | 0.950 | 1.031 | 1.008 | 41.2% | 37.4% | 0.030 | 0.009 | 0.023 | 0.021 |
| slot_power=slot_middle|power_mid | 652 | 2579.0 | 0.884 | 1.385 | 0.948 | 1.132 | 1.085 | 41.6% | 40.1% | 0.029 | 0.016 | 0.044 | 0.036 |
| slot_power=slot_top|power_high | 966 | 4112.0 | 1.058 | 1.611 | 0.869 | 0.818 | 1.013 | 34.8% | 37.1% | 0.050 | 0.026 | 0.044 | 0.058 |
| slot_power=slot_top|power_low | 483 | 2098.0 | 0.816 | 1.509 | 0.969 | 0.979 | 1.019 | 36.2% | 31.5% | 0.039 | 0.008 | 0.022 | 0.023 |
| slot_power=slot_top|power_mid | 1846 | 8076.0 | 0.942 | 2.320 | 0.739 | 1.153 | 1.163 | 32.2% | 35.2% | 0.036 | 0.012 | 0.046 | 0.039 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 630 | 0.249 | 0.240 | 0.010 | use_learned_outcome |
| batter_total_bases | 356 | 0.246 | 0.245 | 0.001 | use_learned_outcome |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 223 | 0.277 | 0.264 | 0.013 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / lay_150_180 / fanduel | 98 | 0.263 | 0.252 | 0.010 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 719 | 0.260 | 0.251 | 0.009 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 1422 | 0.242 | 0.235 | 0.007 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 2580 | 0.242 | 0.237 | 0.006 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 427 | 0.259 | 0.254 | 0.005 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 804 | 0.230 | 0.226 | 0.004 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 2376 | 0.240 | 0.236 | 0.004 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / draftkings | 118 | 0.205 | 0.203 | 0.002 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 456 | 0.256 | 0.254 | 0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 636 | 0.249 | 0.249 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 418 | 0.224 | 0.225 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / lay_130_149 / fanduel | 185 | 0.244 | 0.248 | -0.004 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 371 | 0.369 | 0.377 | -0.007 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 231 | 0.234 | 0.242 | -0.008 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 123 | 0.271 | 0.288 | -0.017 | use_baseline_curve |

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|
| market / side / line_bucket / batter_hits / over / H 0.5 | 10996 | market, side, line_bucket | beta | 0.001 | -0.000 | False |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 0.5 | 10996 | market, side, line_surface, line_bucket | beta | 0.001 | -0.000 | False |
| market / side / line_bucket / batter_total_bases / over / TB 1.5 | 7063 | market, side, line_bucket | raw | -0.000 | - | False |
| market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 1.5 | 7063 | market, side, line_surface, line_bucket | raw | -0.000 | - | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / heavy_lay | 6543 | market, side, line_surface, line_bucket, price_bucket | beta | 0.002 | -0.001 | False |
| market / side / line_bucket / batter_hits / under / H 0.5 | 5391 | market, side, line_bucket | isotonic | 0.002 | -0.000 | False |
| market / side / line_surface / line_bucket / batter_hits / under / common / H 0.5 | 5391 | market, side, line_surface, line_bucket | isotonic | 0.002 | -0.000 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_100_149 | 4956 | market, side, line_surface, line_bucket, price_bucket | raw | -0.000 | - | False |
| market / side / line_bucket / batter_hits / over / H 1.5 | 3572 | market, side, line_bucket | isotonic | 0.024 | -0.023 | False |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 1.5 | 3572 | market, side, line_surface, line_bucket | isotonic | 0.024 | -0.023 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / fanduel | 3561 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.001 | 0.001 | True |
| market / side / line_bucket / batter_total_bases / under / TB 1.5 | 3350 | market, side, line_bucket | beta | 0.001 | 0.002 | True |
| market / side / line_surface / line_bucket / batter_total_bases / under / common / TB 1.5 | 3350 | market, side, line_surface, line_bucket | beta | 0.001 | 0.002 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / draftkings | 2982 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.003 | -0.001 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_150_180 | 2663 | market, side, line_surface, line_bucket, price_bucket | beta | 0.003 | 0.004 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_100_149 | 2609 | market, side, line_surface, line_bucket, price_bucket | beta | 0.001 | 0.003 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_100_149 / draftkings | 2609 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.001 | 0.003 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 2580 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.001 | - | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 2376 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.001 | - | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_150_249 | 2277 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.001 | -0.001 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_150_249 / draftkings | 2277 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.001 | -0.001 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_150_249 | 2231 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.011 | -0.018 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / fanduel | 1433 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.003 | 0.008 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / lay_150_180 | 1422 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.001 | -0.007 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 1422 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.001 | -0.007 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_150_249 / fanduel | 1389 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.012 | -0.013 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / draftkings | 1230 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.007 | -0.018 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / fair_lay | 1175 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.005 | 0.000 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_130_149 | 1162 | market, side, line_surface, line_bucket, price_bucket | beta | 0.006 | -0.009 | False |
| market / side / line_bucket / batter_total_bases / over / TB 2.5+ | 1095 | market, side, line_bucket | beta | 0.069 | -0.066 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_250_499 | 1001 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.058 | -0.028 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_250_499 / fanduel | 1001 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.058 | -0.028 | False |
| market / side / line_bucket / batter_hits / under / H 1.5 | 989 | market, side, line_bucket | isotonic | 0.005 | - | False |
| market / side / line_surface / line_bucket / batter_hits / under / common / H 1.5 | 989 | market, side, line_surface, line_bucket | isotonic | 0.005 | - | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 1.5 / heavy_lay | 916 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.005 | - | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 1.5 / heavy_lay / draftkings | 916 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.005 | - | False |
| market / side / line_bucket / pitcher_strikeouts / over / K 4.5-6.0 | 891 | market, side, line_bucket | beta | 0.005 | 0.019 | True |
| market / side / line_surface / line_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 | 891 | market, side, line_surface, line_bucket | beta | 0.005 | 0.019 | True |
| market / side / line_bucket / pitcher_strikeouts / under / K 4.5-6.0 | 888 | market, side, line_bucket | beta | 0.001 | 0.006 | True |
| market / side / line_surface / line_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 | 888 | market, side, line_surface, line_bucket | beta | 0.001 | 0.006 | True |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 116 | use_event_curve_side_line | event_side_line | 0.187 | 31.7% | 0.254 | 0.214 | 0.242 | 0.242 | 0.244 | 0.187 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 81 | use_distribution_market_blend | distribution_blend | 0.240 | - | 0.244 | 0.247 | 0.241 | 0.241 | 0.240 | 0.256 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 78 | use_distribution_market_blend | distribution_blend | 0.240 | 5.2% | 0.242 | 0.242 | 0.244 | 0.241 | 0.240 | 0.247 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 75 | use_distribution_market_blend | distribution_blend | 0.243 | - | 0.245 | 0.246 | 0.246 | 0.246 | 0.243 | 0.256 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 63 | use_distribution | distribution | 0.229 | 18.4% | 0.242 | 0.245 | 0.229 | 0.229 | 0.232 | 0.234 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 48 | use_event_curve_side_line | event_side_line | 0.268 | 102.6% | 0.302 | 0.295 | 0.331 | 0.331 | 0.331 | 0.268 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 45 | use_event_curve_side_line | event_side_line | 0.207 | 26.7% | 0.210 | 0.208 | 0.209 | 0.209 | 0.210 | 0.207 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 37 | use_event_curve_side_line | event_side_line | 0.240 | 1.3% | 0.240 | 0.240 | 0.243 | 0.241 | 0.242 | 0.240 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 36 | use_event_curve_side_line | event_side_line | 0.210 | 23.3% | 0.242 | 0.226 | 0.245 | 0.238 | 0.237 | 0.210 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 31 | use_distribution_market_blend | distribution_blend | 0.248 | - | 0.254 | 0.252 | 0.249 | 0.249 | 0.248 | 0.259 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 96 | no_bet_negative_roi | distribution | 0.236 | -41.5% | 0.240 | 0.241 | 0.236 | 0.236 | 0.236 | 0.243 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 26 | no_bet_sample | event_side_line | 0.252 | 54.6% | 0.255 | 0.255 | 0.268 | 0.261 | 0.260 | 0.252 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 24 | no_bet_sample | event_side_line | 0.213 | 46.5% | 0.248 | 0.243 | 0.238 | 0.221 | 0.227 | 0.213 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 20 | no_bet_sample | event_side_line | 0.147 | 10.6% | 0.173 | 0.173 | 0.153 | 0.153 | 0.158 | 0.147 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 20 | no_bet_sample | distribution_calibrated | 0.310 | 175.2% | 0.377 | 0.386 | 0.430 | 0.310 | 0.336 | 0.335 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 19 | no_bet_sample | event_side_line | 0.224 | 69.1% | 0.238 | 0.238 | 0.227 | 0.227 | 0.230 | 0.224 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 18 | no_bet_sample | distribution | 0.116 | - | 0.154 | 0.154 | 0.116 | 0.116 | 0.124 | 0.124 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 16 | no_bet_sample | distribution | 0.186 | 38.5% | 0.229 | 0.229 | 0.186 | 0.191 | 0.205 | 0.202 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 15 | no_bet_sample | event_side_line | 0.192 | 69.0% | 0.265 | 0.226 | 0.223 | 0.223 | 0.229 | 0.192 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 14 | no_bet_sample | distribution | 0.226 | 10.7% | 0.254 | 0.252 | 0.226 | 0.226 | 0.230 | 0.229 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 14 | no_bet_sample | market_only | 0.090 | -19.4% | 0.092 | 0.090 | 0.092 | 0.092 | 0.093 | 0.164 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 14 | no_bet_sample | market_only | 0.247 | - | 0.247 | 0.247 | 0.263 | 0.254 | 0.251 | 0.251 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 12 | no_bet_sample | distribution | 0.239 | 89.5% | 0.253 | 0.257 | 0.239 | 0.239 | 0.241 | 0.261 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 11 | no_bet_sample | distribution | 0.194 | - | 0.245 | 0.245 | 0.194 | 0.237 | 0.237 | 0.232 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 11 | no_bet_sample | model_only | 0.233 | 160.0% | 0.233 | 0.242 | 0.258 | 0.238 | 0.239 | 0.254 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 11 | no_bet_sample | event_side_line | 0.242 | 13.5% | 0.253 | 0.253 | 0.269 | 0.259 | 0.254 | 0.242 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 10 | no_bet_sample | distribution_calibrated | 0.267 | 57.7% | 0.275 | 0.290 | 0.324 | 0.267 | 0.274 | 0.316 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 9 | no_bet_sample | distribution | 0.187 | 40.7% | 0.245 | 0.245 | 0.187 | 0.195 | 0.207 | 0.196 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 8 | no_bet_sample | model_only | 0.251 | - | 0.251 | 0.251 | 0.284 | 0.286 | 0.293 | 0.316 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 7 | no_bet_sample | market_only | 0.249 | - | 0.249 | 0.249 | 0.291 | 0.272 | 0.266 | 0.265 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 7 | no_bet_sample | market_only | 0.263 | - | 0.263 | 0.263 | 0.283 | 0.284 | 0.288 | 0.287 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 6 | no_bet_sample | model_only | 0.170 | 58.2% | 0.170 | 0.221 | 0.189 | 0.189 | 0.192 | 0.315 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 6 | no_bet_sample | market_only | 0.242 | - | 0.242 | 0.242 | 0.281 | 0.271 | 0.268 | 0.263 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 5 | no_bet_sample | distribution | 0.117 | - | 0.123 | 0.123 | 0.117 | 0.117 | 0.123 | 0.143 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 5 | no_bet_sample | event_side_line | 0.204 | - | 0.216 | 0.228 | 0.242 | 0.242 | 0.242 | 0.204 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 5 | no_bet_sample | market_only | 0.255 | - | 0.255 | 0.255 | 0.300 | 0.271 | 0.271 | 0.288 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 5 | no_bet_sample | event_side_line | 0.236 | - | 0.244 | 0.244 | 0.243 | 0.245 | 0.247 | 0.236 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 5 | no_bet_sample | market_only | 0.245 | - | 0.245 | 0.245 | 0.273 | 0.258 | 0.254 | 0.249 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 4 | no_bet_sample | model_only | 0.245 | 17.6% | 0.245 | 0.258 | 0.294 | 0.294 | 0.293 | 0.309 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 4 | no_bet_sample | market_only | 0.262 | - | 0.262 | 0.262 | 0.327 | 0.296 | 0.299 | 0.359 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 3 | no_bet_sample | distribution_blend | 0.256 | - | 0.257 | 0.271 | 0.263 | 0.263 | 0.256 | 0.326 |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 3 | no_bet_sample | distribution | 0.264 | - | 0.265 | 0.269 | 0.264 | 0.264 | 0.294 | 0.333 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 3 | no_bet_sample | market_only | 0.196 | - | 0.241 | 0.196 | 0.360 | 0.360 | 0.312 | 0.245 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 3 | no_bet_sample | market_only | 0.171 | - | 0.171 | 0.171 | 0.337 | 0.299 | 0.265 | 0.254 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 3 | no_bet_sample | distribution_calibrated | 0.309 | - | 0.311 | 0.311 | 0.365 | 0.309 | 0.364 | 0.421 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 3 | no_bet_sample | market_only | 0.171 | - | 0.171 | 0.171 | 0.337 | 0.348 | 0.346 | 0.339 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 2 | no_bet_sample | model_only | 0.244 | 142.0% | 0.244 | 0.376 | 0.544 | 0.544 | 0.542 | 0.674 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 2 | no_bet_sample | event_side_line | 0.197 | - | 0.241 | 0.207 | 0.371 | 0.371 | 0.284 | 0.197 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 2 | no_bet_sample | distribution_blend | 0.089 | - | 0.117 | 0.137 | 0.102 | 0.102 | 0.089 | 0.092 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 2 | no_bet_sample | market_only | 0.349 | - | 0.363 | 0.349 | 0.538 | 0.390 | 0.379 | 0.387 |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 2 | no_bet_sample | market_only | 0.207 | - | 0.207 | 0.207 | 0.371 | 0.332 | 0.347 | 0.495 |
| pitcher_strikeouts|under|common|K 6.5-8.0|fair_lay|fanduel | 2 | no_bet_sample | model_only | 0.255 | - | 0.255 | 0.255 | 0.328 | 0.328 | 0.317 | 0.328 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|fanduel | 2 | no_bet_sample | model_only | 0.351 | - | 0.351 | 0.351 | 0.476 | 0.476 | 0.509 | 0.633 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 1 | no_bet_sample | event_side_line | 0.137 | - | 0.243 | 0.193 | 0.213 | 0.213 | 0.168 | 0.137 |
| batter_hits|under|common|H 0.5|lay_150_180|draftkings | 1 | no_bet_sample | market_only | 0.193 | - | 0.201 | 0.193 | 0.213 | 0.213 | 0.249 | 0.337 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 1 | no_bet_sample | model_only | 0.168 | - | 0.168 | 0.168 | 0.414 | 0.335 | 0.305 | 0.285 |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|draftkings | 1 | no_bet_sample | model_only | 0.229 | - | 0.229 | 0.229 | 0.380 | 0.322 | 0.303 | 0.254 |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|fanduel | 1 | no_bet_sample | model_only | 0.231 | - | 0.231 | 0.231 | 0.380 | 0.322 | 0.303 | 0.254 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 1 | no_bet_sample | distribution_calibrated | 0.270 | - | 0.289 | 0.289 | 0.277 | 0.270 | 0.338 | 0.388 |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 1 | no_bet_sample | market_only | 0.343 | - | 0.343 | 0.343 | 0.499 | 0.451 | 0.420 | 0.369 |
