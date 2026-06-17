# MLB Prop Distribution Models

Generated UTC: 2026-06-17T08:52:24Z
Rows: 65072
Date range: 2026-05-31 to 2026-06-15
Status: ready

## Overall Holdout

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 5491 | 0.166 | 0.496 | -2.3% | 1188 | 27.3% | 45.5% |
| market_no_vig | 5491 | 0.160 | 0.480 | -1.4% | 314 | 370.7% | 51.6% |
| distribution | 5491 | 0.168 | 0.499 | 0.2% | 1497 | 32.4% | 45.5% |
| distribution_calibrated | 5491 | 0.170 | 0.505 | 0.2% | 1140 | 22.7% | 46.1% |
| distribution_empirical_blend | 5491 | 0.168 | 0.501 | -0.2% | 1004 | 32.6% | 47.6% |
| event_side_line | 5491 | 0.170 | 0.515 | -4.6% | 2613 | 5.7% | 42.9% |

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases | 2342 | 0.163 | 0.164 | 0.166 | 0.164 | 0.162 | -3.4% | 1.7% | -11.5% |
| batter_hits | 2031 | 0.208 | 0.208 | 0.214 | 0.211 | 0.198 | -12.1% | -21.8% | 5.2% |
| batter_home_runs | 920 | 0.063 | 0.062 | 0.063 | 0.063 | 0.107 | 201.1% | 116.3% | 36.4% |
| pitcher_strikeouts | 198 | 0.257 | 0.292 | 0.274 | 0.268 | 0.263 | -59.0% | -34.4% | -23.7% |

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 4577 | 18475.0 | 1.033 | 1.085 | 0.963 | 1.450 | 0.224 | 0.216 | 0.386 | 0.355 | 0.037 | 0.039 |
| power=power_high | 1006 | 4092.0 | 1.079 | 1.039 | 0.801 | 1.450 | 0.236 | 0.218 | 0.426 | 0.409 | 0.044 | 0.057 |
| power=power_low | 994 | 3931.0 | 0.976 | 1.034 | 0.870 | 1.450 | 0.218 | 0.224 | 0.328 | 0.316 | 0.020 | 0.024 |
| power=power_mid | 2577 | 10452.0 | 1.035 | 1.121 | 1.093 | 1.450 | 0.221 | 0.213 | 0.392 | 0.348 | 0.041 | 0.038 |
| slot=slot_bottom | 1287 | 4508.0 | 1.024 | 1.037 | 0.836 | 1.450 | 0.210 | 0.205 | 0.344 | 0.331 | 0.026 | 0.032 |
| slot=slot_middle | 957 | 3784.0 | 1.016 | 1.065 | 0.954 | 1.450 | 0.221 | 0.218 | 0.381 | 0.355 | 0.036 | 0.038 |
| slot=slot_top | 2333 | 10183.0 | 1.041 | 1.108 | 1.015 | 1.450 | 0.230 | 0.221 | 0.406 | 0.365 | 0.043 | 0.042 |
| slot_power=slot_bottom|power_high | 225 | 813.0 | 1.077 | 1.010 | 0.911 | 0.976 | 0.219 | 0.197 | 0.384 | 0.378 | 0.041 | 0.047 |
| slot_power=slot_bottom|power_low | 397 | 1428.0 | 1.017 | 1.032 | 0.747 | 1.450 | 0.218 | 0.213 | 0.311 | 0.299 | 0.014 | 0.021 |
| slot_power=slot_bottom|power_mid | 665 | 2267.0 | 1.001 | 1.042 | 0.888 | 1.450 | 0.202 | 0.202 | 0.351 | 0.335 | 0.029 | 0.033 |
| slot_power=slot_middle|power_high | 215 | 845.0 | 1.058 | 1.027 | 0.805 | 1.450 | 0.250 | 0.231 | 0.446 | 0.430 | 0.041 | 0.060 |
| slot_power=slot_middle|power_low | 242 | 955.0 | 1.001 | 1.070 | 1.047 | 1.450 | 0.224 | 0.224 | 0.347 | 0.316 | 0.025 | 0.023 |
| slot_power=slot_middle|power_mid | 500 | 1984.0 | 0.997 | 1.067 | 1.073 | 1.371 | 0.208 | 0.209 | 0.369 | 0.342 | 0.039 | 0.036 |
| slot_power=slot_top|power_high | 566 | 2434.0 | 1.070 | 1.046 | 0.822 | 1.450 | 0.237 | 0.220 | 0.433 | 0.411 | 0.046 | 0.059 |
| slot_power=slot_top|power_low | 355 | 1548.0 | 0.932 | 0.999 | 0.908 | 1.450 | 0.215 | 0.235 | 0.331 | 0.332 | 0.023 | 0.026 |
| slot_power=slot_top|power_mid | 1412 | 6201.0 | 1.057 | 1.157 | 1.157 | 1.450 | 0.232 | 0.218 | 0.414 | 0.355 | 0.046 | 0.040 |

## Direct Hitter Event Model

- Status: loaded
- Method: linear_multinomial
- Trained UTC: 2026-06-17T08:38:37.498932+00:00
- Classes: out, walk, single, double, triple, hr
- Production gate: False
- Direct event TB MAE gain vs independent rates: -0.007

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 27361 | 2682 | 0.234 | 0.248 | 48.5% | 46.2% |
| clv_beat_probability | trained | 24158 | 2496 | 0.243 | 0.262 | 39.1% | 47.6% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 5063 | 20392.0 | 0.916 | 1.450 | 0.655 | 0.964 | 1.067 | 38.1% | 38.5% | 0.041 | 0.018 | 0.037 | 0.038 |
| pa=pa_high | 1583 | 7291.0 | 0.937 | 1.450 | 0.898 | 1.058 | 1.148 | 28.9% | 32.3% | 0.042 | 0.016 | 0.043 | 0.041 |
| pa=pa_low | 700 | 2189.0 | 0.885 | 1.450 | 0.815 | 0.797 | 0.975 | 50.9% | 49.4% | 0.047 | 0.021 | 0.022 | 0.031 |
| pa=pa_mid | 2780 | 10912.0 | 0.915 | 1.450 | 0.600 | 0.935 | 1.025 | 40.2% | 39.2% | 0.039 | 0.019 | 0.035 | 0.038 |
| pa_power=pa_high|power_high | 385 | 1766.0 | 1.069 | 1.450 | 1.255 | 0.870 | 1.053 | 31.7% | 33.7% | 0.049 | 0.028 | 0.048 | 0.059 |
| pa_power=pa_high|power_low | 274 | 1245.0 | 0.870 | 1.450 | 1.251 | 1.015 | 1.082 | 29.2% | 29.8% | 0.039 | 0.009 | 0.023 | 0.023 |
| pa_power=pa_high|power_mid | 924 | 4280.0 | 0.933 | 1.450 | 0.600 | 1.196 | 1.189 | 27.7% | 32.5% | 0.041 | 0.013 | 0.048 | 0.039 |
| pa_power=pa_low|power_high | 127 | 410.0 | 1.216 | 0.855 | 0.996 | 1.009 | 1.025 | 46.5% | 49.4% | 0.027 | 0.044 | 0.049 | 0.047 |
| pa_power=pa_low|power_low | 223 | 703.0 | 0.941 | 1.376 | 0.854 | 0.788 | 1.066 | 42.6% | 47.8% | 0.060 | 0.011 | 0.010 | 0.019 |
| pa_power=pa_low|power_mid | 350 | 1076.0 | 0.764 | 1.450 | 0.860 | 0.791 | 0.896 | 57.7% | 50.5% | 0.046 | 0.019 | 0.020 | 0.033 |
| pa_power=pa_mid|power_high | 723 | 2834.0 | 1.090 | 1.421 | 0.632 | 0.739 | 0.938 | 39.1% | 39.6% | 0.053 | 0.035 | 0.037 | 0.056 |
| pa_power=pa_mid|power_low | 638 | 2513.0 | 0.833 | 1.450 | 0.647 | 1.058 | 0.994 | 42.3% | 37.7% | 0.032 | 0.009 | 0.023 | 0.021 |
| pa_power=pa_mid|power_mid | 1419 | 5565.0 | 0.905 | 1.450 | 0.703 | 1.089 | 1.092 | 39.7% | 39.7% | 0.035 | 0.015 | 0.040 | 0.037 |
| power=power_high | 1235 | 5010.0 | 1.117 | 1.435 | 0.847 | 0.777 | 0.988 | 37.6% | 38.8% | 0.049 | 0.033 | 0.042 | 0.056 |
| power=power_low | 1135 | 4461.0 | 0.836 | 1.450 | 0.717 | 0.986 | 1.043 | 39.2% | 37.8% | 0.038 | 0.009 | 0.021 | 0.021 |
| power=power_mid | 2693 | 10921.0 | 0.890 | 1.450 | 0.616 | 1.106 | 1.116 | 38.0% | 38.6% | 0.038 | 0.015 | 0.041 | 0.037 |
| slot=slot_bottom | 1425 | 4966.0 | 0.937 | 1.450 | 0.717 | 0.878 | 1.041 | 45.1% | 45.7% | 0.046 | 0.021 | 0.027 | 0.031 |
| slot=slot_middle | 1058 | 4180.0 | 0.914 | 1.450 | 0.743 | 0.950 | 1.034 | 39.2% | 38.8% | 0.039 | 0.019 | 0.035 | 0.037 |
| slot=slot_top | 2580 | 11246.0 | 0.915 | 1.450 | 0.661 | 1.005 | 1.085 | 33.8% | 34.4% | 0.040 | 0.016 | 0.042 | 0.042 |
| slot_power=slot_bottom|power_high | 285 | 1003.0 | 1.253 | 0.921 | 0.862 | 0.932 | 0.992 | 46.0% | 46.5% | 0.040 | 0.046 | 0.041 | 0.047 |
| slot_power=slot_bottom|power_low | 456 | 1629.0 | 0.910 | 1.450 | 0.704 | 0.836 | 1.044 | 42.5% | 43.7% | 0.045 | 0.010 | 0.014 | 0.019 |
| slot_power=slot_bottom|power_mid | 684 | 2334.0 | 0.867 | 1.450 | 0.895 | 0.921 | 1.053 | 46.5% | 46.7% | 0.049 | 0.019 | 0.030 | 0.033 |
| slot_power=slot_middle|power_high | 253 | 994.0 | 1.043 | 1.400 | 1.026 | 0.825 | 0.981 | 37.5% | 38.0% | 0.057 | 0.034 | 0.039 | 0.059 |
| slot_power=slot_middle|power_low | 277 | 1087.0 | 0.893 | 1.450 | 0.800 | 1.147 | 1.079 | 37.2% | 37.5% | 0.036 | 0.009 | 0.027 | 0.021 |
| slot_power=slot_middle|power_mid | 528 | 2099.0 | 0.903 | 1.450 | 0.746 | 1.036 | 1.034 | 41.1% | 39.9% | 0.032 | 0.017 | 0.037 | 0.035 |
| slot_power=slot_top|power_high | 697 | 3013.0 | 1.069 | 1.450 | 0.847 | 0.787 | 0.993 | 34.1% | 35.9% | 0.050 | 0.028 | 0.042 | 0.059 |
| slot_power=slot_top|power_low | 402 | 1745.0 | 0.810 | 1.450 | 0.966 | 1.001 | 0.999 | 36.8% | 31.3% | 0.033 | 0.009 | 0.023 | 0.023 |
| slot_power=slot_top|power_mid | 1481 | 6488.0 | 0.910 | 1.450 | 0.600 | 1.177 | 1.152 | 32.9% | 34.5% | 0.037 | 0.013 | 0.047 | 0.039 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 2031 | 0.208 | 0.208 | 0.000 | use_baseline_curve |
| batter_total_bases | 2342 | 0.164 | 0.166 | -0.001 | use_baseline_curve |
| batter_home_runs | 920 | 0.062 | 0.062 | 0.000 | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_total_bases / over / common / TB 1.5 / lay_150_180 / fanduel | 81 | 0.265 | 0.261 | 0.005 | use_direct_event_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / draftkings | 97 | 0.209 | 0.207 | 0.002 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / lay_130_149 / fanduel | 153 | 0.251 | 0.251 | 0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 1416 | 0.205 | 0.204 | 0.000 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 2629 | 0.155 | 0.155 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 369 | 0.263 | 0.263 | 0.000 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_150_249 / fanduel | 292 | 0.216 | 0.216 | -0.000 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 6192 | 0.076 | 0.076 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 1523 | 0.206 | 0.206 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 2703 | 0.162 | 0.162 | -0.000 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 1099 | 0.242 | 0.242 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_500_plus / fanduel | 290 | 0.102 | 0.102 | -0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 345 | 0.267 | 0.268 | -0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 650 | 0.232 | 0.233 | -0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 1985 | 0.244 | 0.245 | -0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_100_149 / fanduel | 142 | 0.259 | 0.260 | -0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 2202 | 0.237 | 0.238 | -0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 487 | 0.253 | 0.254 | -0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 543 | 0.266 | 0.268 | -0.002 | use_baseline_curve |

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Calibrated Bins |
|---|---:|---|---:|
| market / side / line_bucket / batter_total_bases / over / TB 2.5+ | 13754 | market, side, line_bucket | 5 |
| market / side / line_surface / line_bucket / batter_total_bases / over / alt_tail / TB 2.5+ | 9144 | market, side, line_surface, line_bucket | 4 |
| market / side / line_bucket / batter_hits / over / H 0.5 | 8655 | market, side, line_bucket | 6 |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 0.5 | 8655 | market, side, line_surface, line_bucket | 6 |
| market / side / line_bucket / batter_total_bases / over / TB 1.5 | 7111 | market, side, line_bucket | 7 |
| market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 1.5 | 7111 | market, side, line_surface, line_bucket | 7 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus | 6192 | market, side, line_surface, line_bucket, price_bucket | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 6192 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 3 |
| market / side / line_bucket / batter_hits / over / H 2.5+ | 5827 | market, side, line_bucket | 2 |
| market / side / line_surface / line_bucket / batter_hits / over / alt_tail / H 2.5+ | 5827 | market, side, line_surface, line_bucket | 2 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / alt_tail / H 2.5+ / plus_500_plus | 5808 | market, side, line_surface, line_bucket, price_bucket | 2 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / alt_tail / H 2.5+ / plus_500_plus / fanduel | 5808 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 2 |
| market / side / line_bucket / batter_hits / over / H 1.5 | 5363 | market, side, line_bucket | 6 |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 1.5 | 5363 | market, side, line_surface, line_bucket | 6 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / heavy_lay | 5247 | market, side, line_surface, line_bucket, price_bucket | 5 |
| market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 2.5+ | 4610 | market, side, line_surface, line_bucket | 5 |
| market / side / line_bucket / batter_home_runs / over / HR 0.5 | 4603 | market, side, line_bucket | 4 |
| market / side / line_surface / line_bucket / batter_home_runs / over / common / HR 0.5 | 4603 | market, side, line_surface, line_bucket | 4 |
| market / side / line_bucket / batter_home_runs / over / HR 1.5+ | 4583 | market, side, line_bucket | 1 |
| market / side / line_surface / line_bucket / batter_home_runs / over / alt_tail / HR 1.5+ | 4583 | market, side, line_surface, line_bucket | 1 |
| market / side / line_surface / line_bucket / price_bucket / batter_home_runs / over / alt_tail / HR 1.5+ / plus_500_plus | 4583 | market, side, line_surface, line_bucket, price_bucket | 1 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_home_runs / over / alt_tail / HR 1.5+ / plus_500_plus / fanduel | 4583 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 1 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_100_149 | 4187 | market, side, line_surface, line_bucket, price_bucket | 5 |
| market / side / line_bucket / batter_hits / under / H 0.5 | 4152 | market, side, line_bucket | 6 |
| market / side / line_surface / line_bucket / batter_hits / under / common / H 0.5 | 4152 | market, side, line_surface, line_bucket | 6 |
| market / side / line_surface / line_bucket / price_bucket / batter_home_runs / over / common / HR 0.5 / plus_500_plus | 2977 | market, side, line_surface, line_bucket, price_bucket | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_home_runs / over / common / HR 0.5 / plus_500_plus / fanduel | 2977 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / fanduel | 2958 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 2.5+ / plus_250_499 | 2703 | market, side, line_surface, line_bucket, price_bucket | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 2703 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_250_499 | 2673 | market, side, line_surface, line_bucket, price_bucket | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_250_499 / fanduel | 2673 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_bucket / batter_total_bases / under / TB 1.5 | 2637 | market, side, line_bucket | 5 |
| market / side / line_surface / line_bucket / batter_total_bases / under / common / TB 1.5 | 2637 | market, side, line_surface, line_bucket | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 | 2629 | market, side, line_surface, line_bucket, price_bucket | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 2629 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / draftkings | 2289 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_150_249 | 2242 | market, side, line_surface, line_bucket, price_bucket | 5 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 2202 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_150_180 | 2054 | market, side, line_surface, line_bucket, price_bucket | 5 |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 605 | use_market_only | market_only | 0.069 | 509.1% | 0.072 | 0.069 | 0.071 | 0.071 | 0.071 | 0.083 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 460 | use_market_only | market_only | 0.019 | 1963.9% | 0.019 | 0.019 | 0.019 | 0.019 | 0.019 | 0.109 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 289 | use_event_curve_side_line | event_side_line | 0.190 | 25.5% | 0.236 | 0.205 | 0.237 | 0.248 | 0.242 | 0.190 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 276 | use_event_curve_side_line | event_side_line | 0.141 | 90.0% | 0.153 | 0.148 | 0.146 | 0.150 | 0.151 | 0.141 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 271 | use_market_only | market_only | 0.143 | 110.5% | 0.148 | 0.143 | 0.147 | 0.146 | 0.145 | 0.152 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 254 | use_event_curve_side_line | event_side_line | 0.151 | 134.7% | 0.183 | 0.171 | 0.187 | 0.186 | 0.186 | 0.151 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 244 | keep_model_only | model_only | 0.237 | - | 0.237 | 0.240 | 0.242 | 0.255 | 0.245 | 0.265 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 217 | use_event_curve_side_line | event_side_line | 0.223 | - | 0.227 | 0.227 | 0.231 | 0.227 | 0.226 | 0.223 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 198 | use_distribution | distribution | 0.222 | 0.7% | 0.229 | 0.229 | 0.222 | 0.227 | 0.229 | 0.223 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 175 | use_distribution_market_blend | distribution_blend | 0.251 | 61.3% | 0.253 | 0.257 | 0.252 | 0.252 | 0.251 | 0.261 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 171 | use_event_curve_side_line | event_side_line | 0.159 | 48.9% | 0.185 | 0.173 | 0.195 | 0.186 | 0.185 | 0.159 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 155 | use_event_curve_side_line | event_side_line | 0.197 | 88.8% | 0.221 | 0.204 | 0.223 | 0.231 | 0.229 | 0.197 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 150 | use_market_only | market_only | 0.142 | 400.0% | 0.148 | 0.142 | 0.148 | 0.154 | 0.152 | 0.144 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 147 | use_distribution | distribution | 0.013 | 162.5% | 0.014 | 0.014 | 0.013 | 0.014 | 0.014 | 0.071 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 139 | use_event_curve_side_line | event_side_line | 0.157 | 53.2% | 0.174 | 0.165 | 0.168 | 0.177 | 0.176 | 0.157 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 114 | use_event_curve_side_line | event_side_line | 0.171 | 55.5% | 0.245 | 0.211 | 0.229 | 0.246 | 0.246 | 0.171 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 90 | use_distribution | distribution | 0.221 | 17.9% | 0.244 | 0.243 | 0.221 | 0.243 | 0.243 | 0.239 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 67 | use_market_only | market_only | 0.217 | - | 0.217 | 0.217 | 0.245 | 0.220 | 0.218 | 0.223 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 66 | use_market_only | market_only | 0.218 | - | 0.218 | 0.218 | 0.247 | 0.223 | 0.221 | 0.224 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 55 | use_distribution | distribution | 0.233 | - | 0.247 | 0.241 | 0.233 | 0.254 | 0.251 | 0.249 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 53 | use_market_only | market_only | 0.215 | - | 0.215 | 0.215 | 0.222 | 0.238 | 0.229 | 0.242 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 52 | use_distribution | distribution | 0.223 | 16.2% | 0.236 | 0.236 | 0.223 | 0.251 | 0.247 | 0.236 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 44 | use_event_curve_side_line | event_side_line | 0.162 | 92.2% | 0.177 | 0.177 | 0.175 | 0.167 | 0.172 | 0.162 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 43 | use_distribution | distribution | 0.231 | 7.1% | 0.254 | 0.254 | 0.231 | 0.305 | 0.286 | 0.300 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 32 | use_distribution | distribution | 0.231 | 91.2% | 0.250 | 0.249 | 0.231 | 0.259 | 0.254 | 0.240 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 31 | use_event_curve_side_line | event_side_line | 0.214 | 72.0% | 0.249 | 0.224 | 0.214 | 0.271 | 0.263 | 0.214 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 30 | use_event_curve_side_line | event_side_line | 0.220 | 77.8% | 0.269 | 0.247 | 0.241 | 0.225 | 0.230 | 0.220 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 30 | use_distribution_market_blend | distribution_blend | 0.250 | - | 0.251 | 0.251 | 0.261 | 0.251 | 0.250 | 0.252 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 30 | use_market_only | market_only | 0.022 | - | 0.029 | 0.022 | 0.026 | 0.028 | 0.023 | 0.040 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 296 | no_bet_no_edge | distribution_calibrated | 0.081 | 45.4% | 0.084 | 0.081 | 0.081 | 0.081 | 0.081 | 0.082 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 221 | no_bet_negative_roi | market_only | 0.225 | -100.0% | 0.226 | 0.225 | 0.238 | 0.233 | 0.230 | 0.234 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 118 | no_bet_negative_roi | event_side_line | 0.227 | -2.9% | 0.230 | 0.230 | 0.244 | 0.237 | 0.233 | 0.227 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 32 | no_bet_negative_roi | distribution | 0.050 | -20.0% | 0.065 | 0.058 | 0.050 | 0.063 | 0.060 | 0.055 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 27 | no_bet_sample | distribution_calibrated | 0.242 | 77.5% | 0.250 | 0.250 | 0.255 | 0.242 | 0.244 | 0.246 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 18 | no_bet_sample | event_side_line | 0.187 | 80.0% | 0.269 | 0.226 | 0.247 | 0.214 | 0.219 | 0.187 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 15 | no_bet_sample | event_side_line | 0.162 | 131.7% | 0.299 | 0.190 | 0.365 | 0.340 | 0.311 | 0.162 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 14 | no_bet_sample | event_side_line | 0.135 | - | 0.171 | 0.157 | 0.151 | 0.137 | 0.147 | 0.135 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 14 | no_bet_sample | event_side_line | 0.256 | 128.0% | 0.270 | 0.275 | 0.326 | 0.284 | 0.278 | 0.256 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 13 | no_bet_sample | event_side_line | 0.210 | - | 0.251 | 0.237 | 0.238 | 0.233 | 0.236 | 0.210 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 13 | no_bet_sample | event_side_line | 0.238 | - | 0.259 | 0.245 | 0.297 | 0.259 | 0.256 | 0.238 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 12 | no_bet_sample | model_only | 0.250 | 3.3% | 0.250 | 0.251 | 0.310 | 0.309 | 0.304 | 0.269 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 11 | no_bet_sample | distribution_calibrated | 0.207 | - | 0.289 | 0.266 | 0.334 | 0.207 | 0.221 | 0.247 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 11 | no_bet_sample | market_only | 0.207 | - | 0.259 | 0.207 | 0.299 | 0.237 | 0.254 | 0.256 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 9 | no_bet_sample | event_side_line | 0.226 | - | 0.240 | 0.233 | 0.236 | 0.235 | 0.230 | 0.226 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 9 | no_bet_sample | market_only | 0.199 | - | 0.258 | 0.199 | 0.304 | 0.222 | 0.251 | 0.262 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 8 | no_bet_sample | market_only | 0.185 | - | 0.197 | 0.185 | 0.306 | 0.333 | 0.265 | 0.274 |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 8 | no_bet_sample | market_only | 0.170 | - | 0.217 | 0.170 | 0.302 | 0.320 | 0.275 | 0.323 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 8 | no_bet_sample | event_side_line | 0.182 | - | 0.242 | 0.229 | 0.235 | 0.237 | 0.241 | 0.182 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 8 | no_bet_sample | market_only | 0.063 | - | 0.093 | 0.063 | 0.126 | 0.172 | 0.159 | 0.158 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 8 | no_bet_sample | market_only | 0.264 | - | 0.285 | 0.264 | 0.317 | 0.271 | 0.293 | 0.359 |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|fanduel | 8 | no_bet_sample | distribution_calibrated | 0.234 | 7.8% | 0.261 | 0.239 | 0.237 | 0.234 | 0.235 | 0.248 |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|draftkings | 7 | no_bet_sample | market_only | 0.246 | - | 0.259 | 0.246 | 0.266 | 0.312 | 0.299 | 0.250 |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|fanduel | 7 | no_bet_sample | event_side_line | 0.242 | - | 0.257 | 0.243 | 0.266 | 0.312 | 0.294 | 0.242 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 7 | no_bet_sample | event_side_line | 0.185 | 60.5% | 0.270 | 0.222 | 0.324 | 0.221 | 0.223 | 0.185 |
| batter_hits|under|common|H 0.5|lay_150_180|draftkings | 6 | no_bet_sample | market_only | 0.178 | - | 0.179 | 0.178 | 0.289 | 0.317 | 0.331 | 0.266 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 6 | no_bet_sample | market_only | 0.209 | - | 0.231 | 0.209 | 0.248 | 0.369 | 0.348 | 0.274 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 6 | no_bet_sample | market_only | 0.178 | - | 0.204 | 0.178 | 0.357 | 0.445 | 0.406 | 0.337 |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 5 | no_bet_sample | market_only | 0.272 | - | 0.297 | 0.272 | 0.379 | 0.406 | 0.373 | 0.374 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 5 | no_bet_sample | distribution_calibrated | 0.257 | - | 0.264 | 0.281 | 0.274 | 0.257 | 0.282 | 0.361 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 5 | no_bet_sample | market_only | 0.272 | - | 0.342 | 0.272 | 0.379 | 0.327 | 0.316 | 0.294 |
