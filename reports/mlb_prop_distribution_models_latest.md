# MLB Prop Distribution Models

Generated UTC: 2026-06-24T08:17:11Z
Rows: 102734
Date range: 2026-05-31 to 2026-06-22
Status: ready

## Overall Holdout

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 6627 | 0.157 | 0.495 | -5.6% | 2148 | -57.1% | 30.6% |
| market_no_vig | 6627 | 0.147 | 0.444 | -4.0% | 317 | 59.8% | 34.7% |
| distribution | 6627 | 0.156 | 0.464 | -4.8% | 2087 | -39.8% | 32.9% |
| distribution_calibrated | 6627 | 0.153 | 0.459 | -4.7% | 1826 | -43.9% | 32.4% |
| distribution_empirical_blend | 6627 | 0.152 | 0.456 | -4.5% | 1661 | -49.8% | 33.3% |
| event_side_line | 6627 | 0.167 | 0.512 | -9.1% | 3351 | -50.6% | 29.7% |

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases | 2738 | 0.150 | 0.145 | 0.143 | 0.141 | 0.146 | -53.1% | -57.1% | -55.5% |
| batter_hits | 2526 | 0.195 | 0.193 | 0.189 | 0.188 | 0.188 | -46.2% | -34.6% | -18.1% |
| batter_home_runs | 1089 | 0.032 | 0.038 | 0.038 | 0.036 | 0.110 | -91.0% | -69.0% | -86.1% |
| pitcher_strikeouts | 274 | 0.257 | 0.254 | 0.253 | 0.255 | 0.267 | - | 4.2% | -22.8% |

## TB/HR True-Pair Production Gates

These gates use holdout rows with true, non-synthetic paired prices only.

| Market / Side / Line | Rows | Brier Gain | Cal Err | Selected | CLV Beat | Avg CLV | Pass | Reasons |
|---|---:|---:|---:|---:|---:|---:|---|---|
| batter_total_bases / over / TB 1.5 | 626 | 0.016 | -2.1% | 85 | 25.2% | 14.1% | False | clv_beat_rate<0.55 |
| batter_total_bases / over / TB 2.5+ | 82 | 0.021 | 15.9% | 49 | 34.2% | -30.5% | False | abs_calibration_error>0.05, clv_beat_rate<0.55, avg_clv_price<=0 |
| batter_total_bases / under / TB 1.5 | 277 | -0.011 | 2.1% | 50 | 46.8% | 16.8% | False | brier_gain<=0.001, clv_beat_rate<0.55 |

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 7664 | 30682.0 | 1.066 | 1.096 | 0.938 | 1.720 | 0.227 | 0.213 | 0.390 | 0.355 | 0.037 | 0.040 |
| power=power_high | 2241 | 8918.0 | 1.078 | 1.022 | 0.811 | 1.365 | 0.231 | 0.213 | 0.414 | 0.405 | 0.044 | 0.056 |
| power=power_low | 1399 | 5471.0 | 1.010 | 1.069 | 0.865 | 1.972 | 0.226 | 0.224 | 0.336 | 0.313 | 0.019 | 0.023 |
| power=power_mid | 4024 | 16293.0 | 1.076 | 1.148 | 1.072 | 1.761 | 0.225 | 0.208 | 0.394 | 0.342 | 0.040 | 0.037 |
| slot=slot_bottom | 2169 | 7562.0 | 1.040 | 1.004 | 0.703 | 1.576 | 0.220 | 0.211 | 0.350 | 0.348 | 0.023 | 0.035 |
| slot=slot_middle | 1615 | 6340.0 | 1.022 | 1.066 | 0.984 | 1.459 | 0.217 | 0.212 | 0.377 | 0.352 | 0.038 | 0.039 |
| slot=slot_top | 3880 | 16780.0 | 1.091 | 1.144 | 1.021 | 1.802 | 0.234 | 0.214 | 0.413 | 0.360 | 0.044 | 0.043 |
| slot_power=slot_bottom|power_high | 559 | 1965.0 | 1.034 | 0.914 | 0.712 | 1.111 | 0.217 | 0.208 | 0.358 | 0.399 | 0.029 | 0.051 |
| slot_power=slot_bottom|power_low | 549 | 1947.0 | 1.074 | 1.096 | 0.823 | 1.699 | 0.240 | 0.219 | 0.342 | 0.306 | 0.014 | 0.021 |
| slot_power=slot_bottom|power_mid | 1061 | 3650.0 | 1.013 | 1.016 | 0.803 | 1.533 | 0.210 | 0.207 | 0.350 | 0.344 | 0.025 | 0.034 |
| slot_power=slot_middle|power_high | 474 | 1843.0 | 1.039 | 0.995 | 0.843 | 1.266 | 0.225 | 0.214 | 0.399 | 0.402 | 0.042 | 0.056 |
| slot_power=slot_middle|power_low | 352 | 1357.0 | 0.968 | 1.032 | 1.003 | 1.338 | 0.214 | 0.224 | 0.329 | 0.316 | 0.023 | 0.023 |
| slot_power=slot_middle|power_mid | 789 | 3140.0 | 1.035 | 1.119 | 1.126 | 1.338 | 0.214 | 0.206 | 0.384 | 0.338 | 0.042 | 0.036 |
| slot_power=slot_top|power_high | 1208 | 5110.0 | 1.100 | 1.075 | 0.899 | 1.404 | 0.239 | 0.215 | 0.441 | 0.408 | 0.051 | 0.058 |
| slot_power=slot_top|power_low | 498 | 2167.0 | 0.978 | 1.046 | 0.919 | 1.746 | 0.221 | 0.227 | 0.335 | 0.317 | 0.021 | 0.024 |
| slot_power=slot_top|power_mid | 2174 | 9503.0 | 1.109 | 1.199 | 1.142 | 1.865 | 0.234 | 0.210 | 0.415 | 0.343 | 0.045 | 0.039 |

## Direct Hitter Event Model

- Status: loaded
- Method: hierarchical_conditional_lgbm
- Trained UTC: 2026-06-24T08:05:33.150921+00:00
- Classes: out, walk, single, double, triple, hr
- Production gate: False
- Production eligible artifact: False
- Leakage-safe player priors: 783 players
- PA uncertainty groups: 6
- Direct event TB MAE gain vs independent rates: -0.020
- Explicit TB-state rows: 7442
- Explicit TB-state Brier: 0.694
- Explicit TB-state log loss: 1.338
- Direct-state selected candidate: convolution
- Direct-state blend alpha: 0.000
- HR-driven 4+ tail Brier gain: 0.000084

## True-Pair Hitter Line Calibration

- Status: trained
- Evidence: temporal_train_true_pair_non_synthetic_only
- Calibrated line/side groups: 6
- Enabled line/side groups: 1
- Synthetic and one-sided FanDuel prices are display-only and cannot train these calibrators.
- `batter_total_bases|over|TB 1.5`: 8929 rows, method=beta, internal_gain=0.003, holdout_gain=-0.001, cal_before=-2.1%, cal_after=-4.2%, enabled=False
- `batter_total_bases|over|TB 2.5`: 804 rows, method=beta, internal_gain=0.048, holdout_gain=0.054, cal_before=33.4%, cal_after=20.9%, enabled=True
- `batter_total_bases|over|TB 3.5`: 186 rows, method=isotonic, internal_gain=0.023, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|over|TB 4.5`: 476 rows, method=beta, internal_gain=0.089, holdout_gain=-, cal_before=-, cal_after=-, enabled=False
- `batter_total_bases|under|TB 1.5`: 4157 rows, method=beta, internal_gain=0.004, holdout_gain=-0.002, cal_before=2.1%, cal_after=2.4%, enabled=False
- `batter_total_bases|under|TB 2.5`: 46 rows, method=raw, internal_gain=-, holdout_gain=-, cal_before=-, cal_after=-, enabled=False

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 45199 | 3172 | 0.236 | 0.243 | 49.2% | 48.2% |
| clv_beat_probability | trained | 41893 | 2959 | 0.228 | 0.258 | 39.0% | 47.4% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 8195 | 32756.0 | 0.961 | 2.295 | 0.671 | 0.942 | 1.086 | 38.0% | 39.1% | 0.042 | 0.018 | 0.037 | 0.039 |
| pa=pa_high | 2148 | 9444.0 | 0.973 | 2.202 | 0.780 | 1.037 | 1.135 | 31.1% | 35.2% | 0.039 | 0.015 | 0.045 | 0.043 |
| pa=pa_low | 1544 | 5360.0 | 0.984 | 1.746 | 0.789 | 0.833 | 1.013 | 45.8% | 43.5% | 0.046 | 0.022 | 0.029 | 0.037 |
| pa=pa_mid | 4503 | 17952.0 | 0.950 | 2.261 | 0.733 | 0.928 | 1.077 | 38.7% | 39.5% | 0.043 | 0.017 | 0.035 | 0.038 |
| pa_power=pa_high|power_high | 647 | 2779.0 | 1.118 | 1.295 | 0.898 | 0.940 | 1.061 | 31.8% | 36.6% | 0.043 | 0.029 | 0.054 | 0.059 |
| pa_power=pa_high|power_low | 306 | 1336.0 | 0.891 | 1.329 | 0.947 | 0.972 | 1.011 | 30.7% | 32.3% | 0.034 | 0.007 | 0.021 | 0.023 |
| pa_power=pa_high|power_mid | 1195 | 5329.0 | 0.948 | 2.043 | 0.872 | 1.126 | 1.192 | 30.7% | 35.2% | 0.038 | 0.010 | 0.046 | 0.040 |
| pa_power=pa_low|power_high | 510 | 1802.0 | 1.057 | 1.154 | 0.920 | 0.770 | 0.890 | 44.5% | 39.9% | 0.042 | 0.032 | 0.033 | 0.056 |
| pa_power=pa_low|power_low | 342 | 1183.0 | 1.015 | 1.395 | 1.016 | 0.915 | 1.133 | 43.6% | 45.6% | 0.046 | 0.010 | 0.012 | 0.017 |
| pa_power=pa_low|power_mid | 692 | 2375.0 | 0.913 | 1.627 | 0.819 | 1.032 | 1.070 | 47.8% | 45.1% | 0.049 | 0.021 | 0.034 | 0.032 |
| pa_power=pa_mid|power_high | 1335 | 5326.0 | 1.118 | 1.526 | 0.847 | 0.806 | 1.010 | 39.9% | 40.9% | 0.047 | 0.028 | 0.041 | 0.054 |
| pa_power=pa_mid|power_low | 908 | 3539.0 | 0.831 | 1.757 | 0.845 | 0.994 | 1.059 | 40.1% | 37.4% | 0.044 | 0.009 | 0.021 | 0.021 |
| pa_power=pa_mid|power_mid | 2260 | 9087.0 | 0.939 | 2.238 | 0.829 | 1.042 | 1.120 | 37.4% | 39.5% | 0.040 | 0.015 | 0.038 | 0.036 |
| power=power_high | 2492 | 9907.0 | 1.121 | 1.487 | 0.748 | 0.799 | 0.999 | 38.7% | 39.6% | 0.045 | 0.029 | 0.043 | 0.055 |
| power=power_low | 1556 | 6058.0 | 0.862 | 2.025 | 0.835 | 0.943 | 1.075 | 39.0% | 38.2% | 0.042 | 0.009 | 0.019 | 0.021 |
| power=power_mid | 4147 | 16791.0 | 0.933 | 2.551 | 0.722 | 1.082 | 1.146 | 37.2% | 39.2% | 0.040 | 0.014 | 0.040 | 0.037 |
| slot=slot_bottom | 2334 | 8101.0 | 0.974 | 1.993 | 0.774 | 0.734 | 1.007 | 44.5% | 44.6% | 0.051 | 0.023 | 0.023 | 0.034 |
| slot=slot_middle | 1723 | 6765.0 | 0.936 | 1.720 | 0.848 | 0.981 | 1.048 | 40.2% | 39.7% | 0.035 | 0.017 | 0.037 | 0.038 |
| slot=slot_top | 4138 | 17890.0 | 0.969 | 2.424 | 0.696 | 1.019 | 1.131 | 33.4% | 35.8% | 0.041 | 0.015 | 0.043 | 0.042 |
| slot_power=slot_bottom|power_high | 633 | 2193.0 | 1.158 | 1.157 | 0.888 | 0.751 | 0.911 | 47.1% | 45.1% | 0.049 | 0.039 | 0.030 | 0.051 |
| slot_power=slot_bottom|power_low | 618 | 2179.0 | 0.954 | 1.595 | 0.924 | 0.871 | 1.099 | 40.8% | 43.0% | 0.049 | 0.010 | 0.014 | 0.019 |
| slot_power=slot_bottom|power_mid | 1083 | 3729.0 | 0.900 | 1.953 | 0.883 | 0.830 | 1.025 | 45.2% | 45.3% | 0.054 | 0.021 | 0.025 | 0.033 |
| slot_power=slot_middle|power_high | 517 | 2013.0 | 1.056 | 1.268 | 0.942 | 0.849 | 0.969 | 39.8% | 40.0% | 0.041 | 0.026 | 0.041 | 0.055 |
| slot_power=slot_middle|power_low | 387 | 1489.0 | 0.871 | 1.425 | 0.934 | 1.052 | 1.045 | 40.1% | 38.2% | 0.036 | 0.009 | 0.024 | 0.021 |
| slot_power=slot_middle|power_mid | 819 | 3263.0 | 0.934 | 1.494 | 0.944 | 1.098 | 1.096 | 40.4% | 40.2% | 0.030 | 0.016 | 0.041 | 0.035 |
| slot_power=slot_top|power_high | 1342 | 5701.0 | 1.103 | 1.560 | 0.840 | 0.875 | 1.051 | 34.4% | 36.9% | 0.045 | 0.026 | 0.049 | 0.057 |
| slot_power=slot_top|power_low | 551 | 2390.0 | 0.836 | 1.528 | 0.948 | 0.974 | 1.042 | 36.3% | 32.8% | 0.040 | 0.008 | 0.021 | 0.023 |
| slot_power=slot_top|power_mid | 2245 | 9799.0 | 0.954 | 2.385 | 0.722 | 1.156 | 1.196 | 32.2% | 35.9% | 0.039 | 0.011 | 0.045 | 0.038 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 1913 | 0.238 | 0.238 | -0.000 | use_baseline_curve |
| batter_total_bases | 985 | 0.246 | 0.257 | -0.011 | use_baseline_curve |
| batter_home_runs | 0 | - | - | - | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 914 | 0.264 | 0.257 | 0.007 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 1016 | 0.237 | 0.233 | 0.005 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 521 | 0.260 | 0.256 | 0.004 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 3238 | 0.245 | 0.241 | 0.004 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / draftkings | 141 | 0.218 | 0.214 | 0.004 | use_direct_event_curve |
| batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 1766 | 0.242 | 0.239 | 0.003 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 2988 | 0.243 | 0.240 | 0.003 | use_direct_event_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 297 | 0.286 | 0.285 | 0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 560 | 0.257 | 0.256 | 0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 775 | 0.252 | 0.253 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / lay_130_149 / fanduel | 241 | 0.247 | 0.248 | -0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / lay_130_149 / draftkings | 87 | 0.251 | 0.253 | -0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 579 | 0.228 | 0.231 | -0.003 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / lay_150_180 / fanduel | 125 | 0.258 | 0.263 | -0.005 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 339 | 0.257 | 0.270 | -0.013 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 481 | 0.364 | 0.377 | -0.013 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 161 | 0.298 | 0.321 | -0.023 | use_baseline_curve |

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Method | Internal Gain | Holdout Gain | Enabled |
|---|---:|---|---|---:|---:|---|
| market / side / line_bucket / batter_hits / over / H 0.5 | 14128 | market, side, line_bucket | isotonic | 0.001 | 0.000 | True |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 0.5 | 14128 | market, side, line_surface, line_bucket | isotonic | 0.001 | 0.000 | True |
| market / side / line_bucket / batter_total_bases / over / TB 1.5 | 8929 | market, side, line_bucket | beta | 0.003 | -0.004 | False |
| market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 1.5 | 8929 | market, side, line_surface, line_bucket | beta | 0.003 | -0.004 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / heavy_lay | 8294 | market, side, line_surface, line_bucket, price_bucket | raw | -0.001 | - | False |
| market / side / line_bucket / batter_hits / under / H 0.5 | 6912 | market, side, line_bucket | beta | 0.001 | -0.001 | False |
| market / side / line_surface / line_bucket / batter_hits / under / common / H 0.5 | 6912 | market, side, line_surface, line_bucket | beta | 0.001 | -0.001 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_100_149 | 6226 | market, side, line_surface, line_bucket, price_bucket | beta | 0.004 | 0.002 | True |
| market / side / line_bucket / batter_hits / over / H 1.5 | 4636 | market, side, line_bucket | isotonic | 0.027 | -0.030 | False |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 1.5 | 4636 | market, side, line_surface, line_bucket | isotonic | 0.027 | -0.030 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / fanduel | 4523 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.000 | - | False |
| market / side / line_bucket / batter_total_bases / under / TB 1.5 | 4157 | market, side, line_bucket | beta | 0.004 | 0.003 | True |
| market / side / line_surface / line_bucket / batter_total_bases / under / common / TB 1.5 | 4157 | market, side, line_surface, line_bucket | beta | 0.004 | 0.003 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / draftkings | 3771 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | raw | -0.001 | - | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_150_180 | 3475 | market, side, line_surface, line_bucket, price_bucket | beta | 0.006 | 0.014 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_100_149 | 3381 | market, side, line_surface, line_bucket, price_bucket | beta | 0.003 | 0.005 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_100_149 / draftkings | 3381 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.003 | 0.005 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 3238 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.004 | 0.007 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 2988 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.003 | -0.002 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_150_249 | 2822 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.001 | -0.002 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 0.5 / plus_150_249 / draftkings | 2822 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.001 | -0.002 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_150_249 | 2797 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.007 | 0.002 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / fanduel | 1863 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.009 | 0.017 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / under / common / TB 1.5 / lay_150_180 | 1766 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.007 | -0.007 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 1766 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | isotonic | 0.007 | -0.007 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_150_249 / fanduel | 1747 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.008 | 0.003 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / lay_150_180 / draftkings | 1612 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.006 | 0.012 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_130_149 | 1474 | market, side, line_surface, line_bucket, price_bucket | beta | 0.001 | 0.004 | True |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / fair_lay | 1474 | market, side, line_surface, line_bucket, price_bucket | isotonic | 0.005 | -0.022 | False |
| market / side / line_bucket / batter_total_bases / over / TB 2.5+ | 1466 | market, side, line_bucket | isotonic | 0.053 | -0.085 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_250_499 | 1397 | market, side, line_surface, line_bucket, price_bucket | beta | 0.052 | -0.054 | False |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_250_499 / fanduel | 1397 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.052 | -0.054 | False |
| market / side / line_bucket / batter_hits / under / H 1.5 | 1227 | market, side, line_bucket | beta | 0.005 | 0.025 | True |
| market / side / line_surface / line_bucket / batter_hits / under / common / H 1.5 | 1227 | market, side, line_surface, line_bucket | beta | 0.005 | 0.025 | True |
| market / side / line_bucket / pitcher_strikeouts / over / K 4.5-6.0 | 1141 | market, side, line_bucket | beta | 0.010 | 0.004 | True |
| market / side / line_surface / line_bucket / pitcher_strikeouts / over / common / K 4.5-6.0 | 1141 | market, side, line_surface, line_bucket | beta | 0.010 | 0.004 | True |
| market / side / line_bucket / pitcher_strikeouts / under / K 4.5-6.0 | 1138 | market, side, line_bucket | beta | 0.003 | -0.013 | False |
| market / side / line_surface / line_bucket / pitcher_strikeouts / under / common / K 4.5-6.0 | 1138 | market, side, line_surface, line_bucket | beta | 0.003 | -0.013 | False |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 1.5 / heavy_lay | 1135 | market, side, line_surface, line_bucket, price_bucket | beta | 0.006 | 0.020 | True |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / under / common / H 1.5 / heavy_lay / draftkings | 1135 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | beta | 0.006 | 0.020 | True |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 318 | use_event_curve_side_line | event_side_line | 0.188 | 29.0% | 0.257 | 0.212 | 0.261 | 0.256 | 0.257 | 0.188 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 244 | use_market_only | market_only | 0.258 | - | 0.264 | 0.258 | 0.267 | 0.262 | 0.260 | 0.267 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 201 | keep_model_only | model_only | 0.246 | - | 0.246 | 0.246 | 0.257 | 0.250 | 0.248 | 0.257 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 186 | use_market_only | market_only | 0.237 | 94.0% | 0.243 | 0.237 | 0.253 | 0.247 | 0.245 | 0.240 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 172 | use_distribution_market_blend | distribution_blend | 0.223 | 20.2% | 0.227 | 0.225 | 0.224 | 0.224 | 0.223 | 0.224 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 141 | use_event_curve_side_line | event_side_line | 0.188 | 32.7% | 0.252 | 0.219 | 0.254 | 0.247 | 0.247 | 0.188 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 131 | use_event_curve_side_line | event_side_line | 0.217 | 106.8% | 0.254 | 0.226 | 0.246 | 0.232 | 0.234 | 0.217 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 118 | use_distribution_market_blend | distribution_blend | 0.257 | - | 0.265 | 0.259 | 0.269 | 0.258 | 0.257 | 0.265 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 114 | keep_model_only | model_only | 0.250 | - | 0.250 | 0.250 | 0.264 | 0.257 | 0.254 | 0.263 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 99 | use_market_only | market_only | 0.208 | 56.8% | 0.221 | 0.208 | 0.225 | 0.225 | 0.221 | 0.216 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 70 | use_distribution_market_blend | distribution_blend | 0.208 | - | 0.213 | 0.213 | 0.219 | 0.208 | 0.208 | 0.214 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 69 | use_distribution_market_blend | distribution_blend | 0.236 | - | 0.262 | 0.241 | 0.237 | 0.237 | 0.236 | 0.244 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 57 | keep_model_only | model_only | 0.242 | - | 0.242 | 0.242 | 0.256 | 0.255 | 0.253 | 0.258 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 55 | keep_model_only | model_only | 0.243 | - | 0.243 | 0.243 | 0.260 | 0.253 | 0.249 | 0.248 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 49 | use_event_curve_side_line | event_side_line | 0.241 | - | 0.246 | 0.246 | 0.265 | 0.265 | 0.256 | 0.241 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 45 | use_event_curve_side_line | event_side_line | 0.219 | 23.3% | 0.224 | 0.236 | 0.231 | 0.230 | 0.234 | 0.219 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 39 | use_distribution | distribution | 0.226 | 35.8% | 0.249 | 0.249 | 0.226 | 0.231 | 0.235 | 0.239 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 37 | keep_model_only | model_only | 0.249 | - | 0.249 | 0.249 | 0.261 | 0.260 | 0.256 | 0.254 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 230 | no_bet_negative_roi | distribution_calibrated | 0.236 | -1.9% | 0.238 | 0.237 | 0.236 | 0.236 | 0.237 | 0.239 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 85 | no_bet_no_edge | distribution_calibrated | 0.193 | 20.7% | 0.205 | 0.205 | 0.213 | 0.193 | 0.193 | 0.202 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 74 | no_bet_no_edge | distribution_calibrated | 0.248 | - | 0.250 | 0.250 | 0.260 | 0.248 | 0.248 | 0.253 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 63 | no_bet_negative_roi | distribution | 0.177 | -34.3% | 0.190 | 0.181 | 0.177 | 0.177 | 0.177 | 0.180 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 45 | no_bet_negative_roi | distribution | 0.225 | -3.9% | 0.226 | 0.246 | 0.225 | 0.225 | 0.232 | 0.239 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 24 | no_bet_sample | event_side_line | 0.256 | 273.8% | 0.301 | 0.308 | 0.327 | 0.327 | 0.329 | 0.256 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 23 | no_bet_sample | event_side_line | 0.185 | 46.3% | 0.275 | 0.214 | 0.218 | 0.244 | 0.240 | 0.185 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 23 | no_bet_sample | event_side_line | 0.099 | - | 0.175 | 0.175 | 0.183 | 0.183 | 0.158 | 0.099 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 23 | no_bet_sample | distribution_calibrated | 0.247 | 42.8% | 0.267 | 0.267 | 0.254 | 0.247 | 0.257 | 0.252 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 22 | no_bet_sample | event_side_line | 0.233 | 49.7% | 0.282 | 0.282 | 0.265 | 0.251 | 0.241 | 0.233 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 21 | no_bet_sample | model_only | 0.272 | 161.0% | 0.272 | 0.300 | 0.273 | 0.274 | 0.288 | 0.304 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 19 | no_bet_sample | event_side_line | 0.237 | 151.5% | 0.373 | 0.328 | 0.379 | 0.277 | 0.300 | 0.237 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 17 | no_bet_sample | event_side_line | 0.217 | 62.6% | 0.569 | 0.242 | 0.311 | 0.311 | 0.295 | 0.217 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 15 | no_bet_sample | distribution_blend | 0.038 | - | 0.049 | 0.066 | 0.039 | 0.039 | 0.038 | 0.224 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 15 | no_bet_sample | market_only | 0.256 | - | 0.509 | 0.256 | 0.295 | 0.295 | 0.290 | 0.283 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 15 | no_bet_sample | market_only | 0.211 | - | 0.211 | 0.211 | 0.228 | 0.228 | 0.238 | 0.238 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 14 | no_bet_sample | event_side_line | 0.103 | - | 0.208 | 0.162 | 0.153 | 0.153 | 0.131 | 0.103 |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 14 | no_bet_sample | event_side_line | 0.254 | -8.1% | 0.259 | 0.259 | 0.304 | 0.303 | 0.274 | 0.254 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 14 | no_bet_sample | distribution | 0.267 | -56.0% | 0.297 | 0.297 | 0.267 | 0.267 | 0.275 | 0.281 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 13 | no_bet_sample | event_side_line | 0.223 | - | 0.522 | 0.242 | 0.264 | 0.264 | 0.256 | 0.223 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 12 | no_bet_sample | model_only | 0.251 | 83.6% | 0.251 | 0.264 | 0.298 | 0.298 | 0.297 | 0.288 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 12 | no_bet_sample | model_only | 0.223 | - | 0.223 | 0.223 | 0.250 | 0.250 | 0.245 | 0.248 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 11 | no_bet_sample | event_side_line | 0.145 | 158.0% | 0.168 | 0.149 | 0.168 | 0.168 | 0.168 | 0.145 |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|fanduel | 11 | no_bet_sample | market_only | 0.267 | - | 0.267 | 0.267 | 0.289 | 0.283 | 0.283 | 0.330 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 11 | no_bet_sample | distribution | 0.249 | - | 0.258 | 0.258 | 0.249 | 0.249 | 0.254 | 0.282 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | distribution_calibrated | 0.249 | -1.9% | 0.264 | 0.264 | 0.251 | 0.249 | 0.249 | 0.291 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 10 | no_bet_sample | market_only | 0.197 | - | 0.197 | 0.197 | 0.231 | 0.237 | 0.217 | 0.233 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | distribution | 0.251 | - | 0.264 | 0.264 | 0.251 | 0.251 | 0.256 | 0.298 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 10 | no_bet_sample | market_only | 0.266 | - | 0.266 | 0.266 | 0.300 | 0.300 | 0.305 | 0.384 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 10 | no_bet_sample | market_only | 0.268 | - | 0.268 | 0.268 | 0.269 | 0.269 | 0.281 | 0.281 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 9 | no_bet_sample | distribution_blend | 0.018 | -100.0% | 0.037 | 0.031 | 0.021 | 0.021 | 0.018 | 0.122 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 9 | no_bet_sample | distribution | 0.244 | - | 0.291 | 0.291 | 0.244 | 0.244 | 0.263 | 0.295 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 9 | no_bet_sample | market_only | 0.237 | - | 0.237 | 0.237 | 0.265 | 0.265 | 0.258 | 0.263 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 8 | no_bet_sample | model_only | 0.260 | - | 0.260 | 0.260 | 0.262 | 0.262 | 0.263 | 0.276 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 8 | no_bet_sample | distribution | 0.223 | 72.6% | 0.254 | 0.254 | 0.223 | 0.223 | 0.249 | 0.254 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 7 | no_bet_sample | event_side_line | 0.170 | - | 0.290 | 0.214 | 0.221 | 0.266 | 0.217 | 0.170 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 7 | no_bet_sample | distribution | 0.238 | 92.6% | 0.250 | 0.250 | 0.238 | 0.246 | 0.247 | 0.256 |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 6 | no_bet_sample | event_side_line | 0.157 | 70.7% | 0.293 | 0.218 | 0.221 | 0.221 | 0.207 | 0.157 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 6 | no_bet_sample | market_only | 0.208 | - | 0.208 | 0.208 | 0.238 | 0.239 | 0.224 | 0.236 |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 6 | no_bet_sample | distribution | 0.234 | -26.5% | 0.270 | 0.270 | 0.234 | 0.253 | 0.275 | 0.308 |
| batter_hits|under|common|H 1.5|lay_150_180|draftkings | 5 | no_bet_sample | event_side_line | 0.032 | 56.8% | 0.187 | 0.163 | 0.144 | 0.088 | 0.073 | 0.032 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 5 | no_bet_sample | distribution | 0.220 | -19.7% | 0.241 | 0.241 | 0.220 | 0.237 | 0.239 | 0.234 |
