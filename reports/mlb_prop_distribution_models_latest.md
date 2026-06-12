# MLB Prop Distribution Models

Generated UTC: 2026-06-12T15:17:57Z
Rows: 40380
Date range: 2026-05-31 to 2026-06-11
Status: ready

## Overall Holdout

| Variant | Rows | Brier | Log Loss | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 1670 | 0.173 | 0.509 | -0.7% | 322 | -41.3% | 40.6% |
| market_no_vig | 1670 | 0.169 | 0.500 | -0.4% | 95 | -4.2% | 37.9% |
| distribution | 1670 | 0.177 | 0.517 | 0.3% | 421 | -6.0% | 46.3% |
| distribution_calibrated | 1670 | 0.175 | 0.512 | 1.4% | 310 | -28.5% | 38.7% |
| distribution_empirical_blend | 1670 | 0.173 | 0.509 | 1.1% | 248 | -21.0% | 38.8% |
| event_side_line | 1670 | 0.166 | 0.489 | -0.0% | 453 | 17.7% | 37.9% |

## Market Holdout

| Market | Rows | Model Brier | Distribution Brier | Cal Dist Brier | Blend Brier | Side-Line Brier | Model ROI | Blend ROI | Side-Line ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases | 700 | 0.181 | 0.185 | 0.184 | 0.182 | 0.176 | -27.1% | -11.1% | 24.1% |
| batter_hits | 622 | 0.195 | 0.203 | 0.200 | 0.199 | 0.187 | -49.3% | -52.8% | -7.2% |
| batter_home_runs | 264 | 0.077 | 0.076 | 0.077 | 0.077 | 0.075 | -85.2% | -31.3% | 106.9% |
| pitcher_strikeouts | 84 | 0.244 | 0.241 | 0.216 | 0.206 | 0.210 | -7.8% | 67.1% | 35.1% |

## Hitter Outcome Shrinkage

| Group | Rows | PA | Hit Mult | TB Mult | HR Mult | XBH Mult | Actual H/PA | Pred H/PA | Actual TB/PA | Pred TB/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 2836 | 11475.0 | 1.017 | 1.078 | 0.939 | 1.450 | 0.222 | 0.218 | 0.379 | 0.350 | 0.035 | 0.038 |
| power=power_high | 472 | 1944.0 | 1.026 | 1.011 | 0.835 | 1.364 | 0.230 | 0.223 | 0.421 | 0.415 | 0.045 | 0.056 |
| power=power_low | 717 | 2828.0 | 0.999 | 1.066 | 0.851 | 1.450 | 0.224 | 0.225 | 0.340 | 0.317 | 0.020 | 0.024 |
| power=power_mid | 1647 | 6703.0 | 1.020 | 1.102 | 1.028 | 1.450 | 0.219 | 0.215 | 0.383 | 0.346 | 0.039 | 0.038 |
| slot=slot_bottom | 808 | 2821.0 | 1.078 | 1.090 | 0.826 | 1.450 | 0.221 | 0.203 | 0.354 | 0.321 | 0.024 | 0.031 |
| slot=slot_middle | 589 | 2354.0 | 1.034 | 1.090 | 0.935 | 1.450 | 0.222 | 0.213 | 0.379 | 0.343 | 0.034 | 0.036 |
| slot=slot_top | 1439 | 6300.0 | 0.982 | 1.061 | 0.989 | 1.450 | 0.223 | 0.227 | 0.390 | 0.366 | 0.040 | 0.041 |
| slot_power=slot_bottom|power_high | 106 | 378.0 | 0.967 | 0.936 | 0.912 | 0.902 | 0.185 | 0.198 | 0.333 | 0.381 | 0.037 | 0.047 |
| slot_power=slot_bottom|power_low | 297 | 1064.0 | 1.058 | 1.074 | 0.752 | 1.450 | 0.228 | 0.212 | 0.326 | 0.297 | 0.014 | 0.022 |
| slot_power=slot_bottom|power_mid | 405 | 1379.0 | 1.110 | 1.139 | 0.910 | 1.450 | 0.226 | 0.198 | 0.381 | 0.324 | 0.029 | 0.033 |
| slot_power=slot_middle|power_high | 92 | 374.0 | 1.128 | 1.121 | 0.926 | 1.321 | 0.283 | 0.227 | 0.529 | 0.429 | 0.048 | 0.058 |
| slot_power=slot_middle|power_low | 163 | 658.0 | 1.021 | 1.085 | 0.968 | 1.426 | 0.222 | 0.215 | 0.347 | 0.306 | 0.023 | 0.024 |
| slot_power=slot_middle|power_mid | 334 | 1322.0 | 0.983 | 1.035 | 0.968 | 1.450 | 0.204 | 0.209 | 0.352 | 0.338 | 0.035 | 0.036 |
| slot_power=slot_top|power_high | 274 | 1192.0 | 0.992 | 0.986 | 0.846 | 1.271 | 0.227 | 0.230 | 0.414 | 0.422 | 0.046 | 0.059 |
| slot_power=slot_top|power_low | 257 | 1106.0 | 0.935 | 1.017 | 0.938 | 1.450 | 0.222 | 0.242 | 0.350 | 0.342 | 0.024 | 0.027 |
| slot_power=slot_top|power_mid | 908 | 4002.0 | 0.999 | 1.097 | 1.085 | 1.450 | 0.222 | 0.222 | 0.394 | 0.356 | 0.043 | 0.039 |

## Direct Hitter Event Model

- Status: loaded
- Method: linear_multinomial
- Trained UTC: 2026-06-12T15:12:15.443692+00:00
- Classes: out, walk, single, double, triple, hr
- Production gate: False
- Direct event TB MAE gain vs independent rates: -0.007

## Event-Curve Side/Line Models

| Target | Status | Train | Holdout | Model Brier | Baseline Brier | Model Avg | Baseline Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| win_probability | trained | 38710 | 1670 | 0.166 | 0.173 | 32.4% | 31.3% |
| clv_beat_probability | trained | 33100 | 1599 | 0.223 | 0.271 | 36.5% | 32.3% |

## TB Component Structure

| Group | Rows | PA | 1B Mult | 2B Mult | 3B Mult | HR Mult | TB Mult | Actual 0 TB | Pred 0 TB | Actual 2B/PA | Pred 2B/PA | Actual HR/PA | Pred HR/PA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 3290 | 13291.0 | 0.901 | 1.450 | 0.758 | 0.950 | 1.057 | 38.4% | 38.1% | 0.041 | 0.019 | 0.035 | 0.037 |
| pa=pa_high | 1052 | 4823.0 | 0.926 | 1.450 | 0.975 | 1.004 | 1.101 | 29.6% | 31.1% | 0.043 | 0.019 | 0.041 | 0.041 |
| pa=pa_low | 439 | 1365.0 | 0.907 | 1.450 | 1.058 | 0.912 | 1.091 | 50.1% | 51.5% | 0.053 | 0.018 | 0.024 | 0.028 |
| pa=pa_mid | 1799 | 7103.0 | 0.894 | 1.450 | 0.600 | 0.925 | 1.008 | 40.8% | 39.0% | 0.038 | 0.019 | 0.033 | 0.036 |
| pa_power=pa_high|power_high | 262 | 1198.0 | 1.104 | 1.119 | 1.181 | 0.850 | 0.988 | 29.8% | 32.1% | 0.047 | 0.039 | 0.045 | 0.061 |
| pa_power=pa_high|power_low | 192 | 867.0 | 0.856 | 1.415 | 1.133 | 0.862 | 1.021 | 32.3% | 28.7% | 0.047 | 0.010 | 0.016 | 0.022 |
| pa_power=pa_high|power_mid | 598 | 2758.0 | 0.918 | 1.450 | 0.755 | 1.181 | 1.172 | 28.6% | 31.3% | 0.040 | 0.013 | 0.048 | 0.039 |
| pa_power=pa_low|power_high | 57 | 161.0 | 1.111 | 0.904 | 1.058 | 0.972 | 0.967 | 56.1% | 56.1% | 0.031 | 0.064 | 0.037 | 0.046 |
| pa_power=pa_low|power_low | 175 | 556.0 | 0.933 | 1.332 | 0.934 | 0.906 | 1.091 | 46.3% | 49.4% | 0.056 | 0.009 | 0.014 | 0.019 |
| pa_power=pa_low|power_mid | 207 | 648.0 | 0.888 | 1.361 | 1.092 | 0.966 | 1.097 | 51.7% | 52.0% | 0.056 | 0.014 | 0.029 | 0.032 |
| pa_power=pa_mid|power_high | 362 | 1437.0 | 1.090 | 1.098 | 0.781 | 0.823 | 0.938 | 43.1% | 40.0% | 0.048 | 0.042 | 0.038 | 0.053 |
| pa_power=pa_mid|power_low | 484 | 1909.0 | 0.858 | 1.450 | 0.694 | 1.135 | 1.069 | 38.6% | 37.3% | 0.036 | 0.009 | 0.026 | 0.021 |
| pa_power=pa_mid|power_mid | 953 | 3757.0 | 0.881 | 1.450 | 0.645 | 0.956 | 1.014 | 41.0% | 39.4% | 0.034 | 0.015 | 0.034 | 0.036 |
| power=power_high | 681 | 2796.0 | 1.126 | 1.084 | 0.990 | 0.795 | 0.951 | 39.1% | 38.3% | 0.046 | 0.042 | 0.041 | 0.056 |
| power=power_low | 851 | 3332.0 | 0.837 | 1.450 | 0.832 | 1.001 | 1.075 | 38.8% | 37.9% | 0.042 | 0.009 | 0.021 | 0.021 |
| power=power_mid | 1758 | 7163.0 | 0.883 | 1.450 | 0.689 | 1.056 | 1.098 | 38.1% | 38.2% | 0.038 | 0.014 | 0.039 | 0.037 |
| slot=slot_bottom | 931 | 3235.0 | 0.982 | 1.450 | 0.768 | 0.897 | 1.098 | 42.7% | 46.3% | 0.050 | 0.020 | 0.026 | 0.030 |
| slot=slot_middle | 684 | 2726.0 | 0.914 | 1.450 | 0.787 | 0.934 | 1.040 | 40.1% | 38.9% | 0.043 | 0.019 | 0.032 | 0.035 |
| slot=slot_top | 1675 | 7330.0 | 0.873 | 1.450 | 0.819 | 0.985 | 1.039 | 35.4% | 33.3% | 0.036 | 0.018 | 0.040 | 0.041 |
| slot_power=slot_bottom|power_high | 155 | 536.0 | 1.190 | 0.878 | 0.928 | 0.952 | 0.963 | 48.4% | 47.4% | 0.041 | 0.057 | 0.039 | 0.045 |
| slot_power=slot_bottom|power_low | 354 | 1259.0 | 0.938 | 1.450 | 0.740 | 0.855 | 1.083 | 41.2% | 44.4% | 0.048 | 0.009 | 0.014 | 0.019 |
| slot_power=slot_bottom|power_mid | 422 | 1440.0 | 0.969 | 1.450 | 0.909 | 0.955 | 1.146 | 41.9% | 47.4% | 0.056 | 0.016 | 0.031 | 0.033 |
| slot_power=slot_middle|power_high | 127 | 511.0 | 1.090 | 1.235 | 0.996 | 0.911 | 1.022 | 39.4% | 38.3% | 0.072 | 0.047 | 0.043 | 0.057 |
| slot_power=slot_middle|power_low | 195 | 778.0 | 0.905 | 1.395 | 0.938 | 1.103 | 1.096 | 36.4% | 37.6% | 0.040 | 0.009 | 0.026 | 0.021 |
| slot_power=slot_middle|power_mid | 362 | 1437.0 | 0.899 | 1.450 | 0.745 | 0.941 | 0.996 | 42.3% | 39.7% | 0.034 | 0.015 | 0.032 | 0.035 |
| slot_power=slot_top|power_high | 399 | 1749.0 | 1.067 | 1.086 | 1.046 | 0.794 | 0.935 | 35.3% | 34.8% | 0.041 | 0.036 | 0.041 | 0.059 |
| slot_power=slot_top|power_low | 302 | 1295.0 | 0.798 | 1.450 | 0.990 | 1.046 | 1.015 | 37.4% | 30.3% | 0.039 | 0.010 | 0.025 | 0.024 |
| slot_power=slot_top|power_mid | 974 | 4286.0 | 0.870 | 1.450 | 0.699 | 1.125 | 1.098 | 34.8% | 33.6% | 0.034 | 0.013 | 0.044 | 0.038 |

## Hitter Outcome Policy

| Market | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_hits | 622 | 0.203 | 0.203 | 0.000 | use_baseline_curve |
| batter_total_bases | 700 | 0.185 | 0.185 | 0.000 | use_baseline_curve |
| batter_home_runs | 264 | 0.076 | 0.076 | 0.000 | use_baseline_curve |

## TB Event Model Bucket Policy

| Bucket | Rows | Base Brier | Learned Brier | Gain | Decision |
|---|---:|---:|---:|---:|---|
| batter_total_bases / over / common / TB 1.5 / fair_lay / draftkings | 220 | 0.269 | 0.268 | 0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / heavy_lay / draftkings | 401 | 0.230 | 0.229 | 0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / fair_lay / fanduel | 281 | 0.259 | 0.258 | 0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / fair_lay / draftkings | 222 | 0.275 | 0.275 | 0.001 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_150_180 / draftkings | 713 | 0.230 | 0.230 | 0.001 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / lay_130_149 / fanduel | 88 | 0.267 | 0.267 | 0.000 | use_baseline_curve |
| batter_total_bases / under / common / TB 1.5 / lay_130_149 / draftkings | 281 | 0.247 | 0.247 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_150_249 / fanduel | 1015 | 0.216 | 0.216 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / draftkings | 1216 | 0.237 | 0.237 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_100_149 / fanduel | 80 | 0.271 | 0.271 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 1365 | 0.234 | 0.234 | 0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_150_249 / fanduel | 817 | 0.193 | 0.193 | 0.000 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_150_249 / fanduel | 166 | 0.232 | 0.232 | 0.000 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 3949 | 0.079 | 0.079 | 0.000 | use_baseline_curve |
| batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 1535 | 0.149 | 0.149 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 1728 | 0.167 | 0.167 | -0.000 | use_baseline_curve |
| batter_total_bases / over / common / TB 2.5+ / plus_500_plus / fanduel | 198 | 0.113 | 0.113 | -0.000 | use_baseline_curve |

## Line-Bucket Probability Calibration

| Group | Rows | Columns | Calibrated Bins |
|---|---:|---|---:|
| market / side / line_bucket / batter_total_bases / over / TB 2.5+ | 8511 | market, side, line_bucket | 5 |
| market / side / line_surface / line_bucket / batter_total_bases / over / alt_tail / TB 2.5+ | 5662 | market, side, line_surface, line_bucket | 4 |
| market / side / line_bucket / batter_hits / over / H 2.5+ | 5418 | market, side, line_bucket | 2 |
| market / side / line_surface / line_bucket / batter_hits / over / alt_tail / H 2.5+ | 5418 | market, side, line_surface, line_bucket | 2 |
| market / side / line_bucket / batter_hits / over / H 0.5 | 5417 | market, side, line_bucket | 6 |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 0.5 | 5417 | market, side, line_surface, line_bucket | 6 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / alt_tail / H 2.5+ / plus_500_plus | 5399 | market, side, line_surface, line_bucket, price_bucket | 2 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / alt_tail / H 2.5+ / plus_500_plus / fanduel | 5399 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 2 |
| market / side / line_bucket / batter_total_bases / over / TB 1.5 | 4387 | market, side, line_bucket | 7 |
| market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 1.5 | 4387 | market, side, line_surface, line_bucket | 7 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus | 3949 | market, side, line_surface, line_bucket, price_bucket | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / alt_tail / TB 2.5+ / plus_500_plus / fanduel | 3949 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 3 |
| market / side / line_bucket / batter_hits / over / H 1.5 | 3300 | market, side, line_bucket | 6 |
| market / side / line_surface / line_bucket / batter_hits / over / common / H 1.5 | 3300 | market, side, line_surface, line_bucket | 6 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / heavy_lay | 3181 | market, side, line_surface, line_bucket, price_bucket | 5 |
| market / side / line_bucket / batter_home_runs / over / HR 0.5 | 2862 | market, side, line_bucket | 4 |
| market / side / line_surface / line_bucket / batter_home_runs / over / common / HR 0.5 | 2862 | market, side, line_surface, line_bucket | 4 |
| market / side / line_surface / line_bucket / batter_total_bases / over / common / TB 2.5+ | 2849 | market, side, line_surface, line_bucket | 5 |
| market / side / line_bucket / batter_home_runs / over / HR 1.5+ | 2844 | market, side, line_bucket | 1 |
| market / side / line_surface / line_bucket / batter_home_runs / over / alt_tail / HR 1.5+ | 2844 | market, side, line_surface, line_bucket | 1 |
| market / side / line_surface / line_bucket / price_bucket / batter_home_runs / over / alt_tail / HR 1.5+ / plus_500_plus | 2844 | market, side, line_surface, line_bucket, price_bucket | 1 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_home_runs / over / alt_tail / HR 1.5+ / plus_500_plus / fanduel | 2844 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 1 |
| market / side / line_bucket / batter_hits / under / H 0.5 | 2655 | market, side, line_bucket | 5 |
| market / side / line_surface / line_bucket / batter_hits / under / common / H 0.5 | 2655 | market, side, line_surface, line_bucket | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 1.5 / plus_100_149 | 2581 | market, side, line_surface, line_bucket, price_bucket | 4 |
| market / side / line_surface / line_bucket / price_bucket / batter_home_runs / over / common / HR 0.5 / plus_500_plus | 1917 | market, side, line_surface, line_bucket, price_bucket | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_home_runs / over / common / HR 0.5 / plus_500_plus / fanduel | 1917 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / fanduel | 1757 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 1.5 / plus_250_499 | 1732 | market, side, line_surface, line_bucket, price_bucket | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 1.5 / plus_250_499 / fanduel | 1732 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / common / TB 2.5+ / plus_250_499 | 1728 | market, side, line_surface, line_bucket, price_bucket | 3 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 2.5+ / plus_250_499 / fanduel | 1728 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 3 |
| market / side / line_bucket / batter_total_bases / under / TB 1.5 | 1654 | market, side, line_bucket | 5 |
| market / side / line_surface / line_bucket / batter_total_bases / under / common / TB 1.5 | 1654 | market, side, line_surface, line_bucket | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 | 1535 | market, side, line_surface, line_bucket, price_bucket | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / alt_tail / TB 2.5+ / plus_250_499 / fanduel | 1535 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_hits / over / common / H 0.5 / heavy_lay / draftkings | 1424 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 4 |
| market / side / line_surface / line_bucket / price_bucket / bookmaker_key / batter_total_bases / over / common / TB 1.5 / plus_100_149 / fanduel | 1365 | market, side, line_surface, line_bucket, price_bucket, bookmaker_key | 3 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / over / common / H 0.5 / lay_150_180 | 1346 | market, side, line_surface, line_bucket, price_bucket | 5 |
| market / side / line_surface / line_bucket / price_bucket / batter_hits / under / common / H 0.5 / plus_100_149 | 1301 | market, side, line_surface, line_bucket, price_bucket | 5 |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | Best Brier | ROI | Model | Market | Distribution | Cal Dist | Blend | Side-Line |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 157 | use_event_curve_side_line | event_side_line | 0.055 | 84.8% | 0.063 | 0.059 | 0.060 | 0.060 | 0.060 | 0.055 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 132 | use_event_curve_side_line | event_side_line | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 102 | use_event_curve_side_line | event_side_line | 0.163 | 17.0% | 0.215 | 0.185 | 0.225 | 0.226 | 0.224 | 0.163 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 100 | use_event_curve_side_line | event_side_line | 0.181 | 28.3% | 0.184 | 0.183 | 0.185 | 0.191 | 0.192 | 0.181 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 81 | use_event_curve_side_line | event_side_line | 0.088 | 63.4% | 0.092 | 0.090 | 0.090 | 0.088 | 0.088 | 0.088 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 74 | use_market_only | market_only | 0.230 | - | 0.231 | 0.230 | 0.241 | 0.232 | 0.232 | 0.238 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 72 | keep_model_only | model_only | 0.237 | - | 0.237 | 0.238 | 0.245 | 0.241 | 0.240 | 0.250 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 72 | use_event_curve_side_line | event_side_line | 0.141 | 97.9% | 0.157 | 0.156 | 0.151 | 0.160 | 0.161 | 0.141 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 69 | use_market_only | market_only | 0.250 | - | 0.260 | 0.250 | 0.277 | 0.265 | 0.262 | 0.263 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 58 | use_event_curve_side_line | event_side_line | 0.216 | 47.1% | 0.226 | 0.226 | 0.241 | 0.233 | 0.231 | 0.216 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 57 | use_event_curve_side_line | event_side_line | 0.077 | 280.0% | 0.103 | 0.096 | 0.098 | 0.095 | 0.093 | 0.077 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 54 | keep_model_only | model_only | 0.237 | 68.0% | 0.237 | 0.243 | 0.251 | 0.242 | 0.241 | 0.249 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 53 | use_distribution_market_blend | distribution_blend | 0.177 | - | 0.182 | 0.177 | 0.182 | 0.177 | 0.177 | 0.183 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 51 | use_distribution | distribution | 0.247 | 66.8% | 0.249 | 0.250 | 0.247 | 0.257 | 0.256 | 0.248 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 33 | use_event_curve_side_line | event_side_line | 0.221 | 5.8% | 0.237 | 0.235 | 0.254 | 0.234 | 0.234 | 0.221 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 32 | keep_model_only | model_only | 0.258 | - | 0.258 | 0.258 | 0.265 | 0.265 | 0.264 | 0.303 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 30 | keep_model_only | model_only | 0.265 | - | 0.265 | 0.265 | 0.275 | 0.286 | 0.283 | 0.308 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 71 | no_bet_negative_roi | event_side_line | 0.179 | -0.5% | 0.197 | 0.193 | 0.208 | 0.210 | 0.210 | 0.179 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 60 | no_bet_negative_roi | distribution | 0.016 | -100.0% | 0.018 | 0.018 | 0.016 | 0.017 | 0.017 | 0.017 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 28 | no_bet_sample | event_side_line | 0.240 | 10.9% | 0.250 | 0.244 | 0.276 | 0.259 | 0.257 | 0.240 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 27 | no_bet_sample | model_only | 0.242 | - | 0.242 | 0.243 | 0.258 | 0.245 | 0.243 | 0.255 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 27 | no_bet_sample | model_only | 0.209 | - | 0.209 | 0.227 | 0.221 | 0.233 | 0.229 | 0.220 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 24 | no_bet_sample | distribution_blend | 0.259 | - | 0.276 | 0.267 | 0.294 | 0.262 | 0.259 | 0.295 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 21 | no_bet_sample | event_side_line | 0.166 | 13.6% | 0.188 | 0.176 | 0.201 | 0.199 | 0.194 | 0.166 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 17 | no_bet_sample | event_side_line | 0.232 | - | 0.248 | 0.240 | 0.275 | 0.267 | 0.258 | 0.232 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 12 | no_bet_sample | distribution | 0.213 | -0.5% | 0.252 | 0.252 | 0.213 | 0.263 | 0.256 | 0.273 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 11 | no_bet_sample | distribution_calibrated | 0.210 | - | 0.229 | 0.238 | 0.249 | 0.210 | 0.217 | 0.223 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 11 | no_bet_sample | distribution | 0.219 | 35.3% | 0.250 | 0.250 | 0.219 | 0.265 | 0.259 | 0.258 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 7 | no_bet_sample | event_side_line | 0.141 | 90.0% | 0.175 | 0.185 | 0.203 | 0.173 | 0.180 | 0.141 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 7 | no_bet_sample | distribution | 0.218 | - | 0.237 | 0.249 | 0.218 | 0.300 | 0.282 | 0.301 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 7 | no_bet_sample | event_side_line | 0.116 | 65.0% | 0.237 | 0.196 | 0.261 | 0.259 | 0.254 | 0.116 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 5 | no_bet_sample | distribution_calibrated | 0.186 | - | 0.248 | 0.261 | 0.206 | 0.186 | 0.193 | 0.213 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 5 | no_bet_sample | distribution_calibrated | 0.187 | - | 0.225 | 0.220 | 0.229 | 0.187 | 0.194 | 0.251 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 5 | no_bet_sample | distribution | 0.206 | 46.3% | 0.248 | 0.261 | 0.206 | 0.211 | 0.206 | 0.210 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 5 | no_bet_sample | event_side_line | 0.177 | 111.2% | 0.244 | 0.310 | 0.326 | 0.295 | 0.240 | 0.177 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 4 | no_bet_sample | event_side_line | 0.158 | - | 0.245 | 0.203 | 0.203 | 0.188 | 0.188 | 0.158 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 4 | no_bet_sample | event_side_line | 0.219 | - | 0.284 | 0.253 | 0.271 | 0.260 | 0.236 | 0.219 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 4 | no_bet_sample | distribution | 0.180 | - | 0.244 | 0.253 | 0.180 | 0.194 | 0.208 | 0.267 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 4 | no_bet_sample | distribution_calibrated | 0.197 | - | 0.258 | 0.232 | 0.263 | 0.197 | 0.199 | 0.240 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 4 | no_bet_sample | event_side_line | 0.176 | 45.0% | 0.244 | 0.253 | 0.180 | 0.207 | 0.217 | 0.176 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 4 | no_bet_sample | model_only | 0.239 | 45.0% | 0.239 | 0.273 | 0.282 | 0.279 | 0.254 | 0.253 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 3 | no_bet_sample | distribution_calibrated | 0.221 | - | 0.251 | 0.253 | 0.236 | 0.221 | 0.227 | 0.243 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 3 | no_bet_sample | event_side_line | 0.177 | - | 0.269 | 0.213 | 0.225 | 0.243 | 0.238 | 0.177 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 3 | no_bet_sample | distribution | 0.236 | 93.5% | 0.253 | 0.253 | 0.236 | 0.277 | 0.273 | 0.255 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 3 | no_bet_sample | event_side_line | 0.191 | 67.6% | 0.306 | 0.214 | 0.308 | 0.316 | 0.274 | 0.191 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 3 | no_bet_sample | distribution_blend | 0.136 | - | 0.243 | 0.297 | 0.305 | 0.137 | 0.136 | 0.253 |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|draftkings | 3 | no_bet_sample | distribution | 0.219 | 32.4% | 0.245 | 0.260 | 0.219 | 0.268 | 0.237 | 0.281 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 3 | no_bet_sample | event_side_line | 0.222 | 10.7% | 0.261 | 0.236 | 0.269 | 0.244 | 0.229 | 0.222 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 3 | no_bet_sample | distribution_blend | 0.217 | - | 0.245 | 0.260 | 0.219 | 0.231 | 0.217 | 0.237 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 3 | no_bet_sample | distribution_blend | 0.217 | - | 0.245 | 0.268 | 0.219 | 0.231 | 0.217 | 0.304 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 3 | no_bet_sample | distribution_calibrated | 0.205 | - | 0.245 | 0.223 | 0.229 | 0.205 | 0.208 | 0.209 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 2 | no_bet_sample | event_side_line | 0.148 | - | 0.262 | 0.191 | 0.278 | 0.256 | 0.268 | 0.148 |
| batter_hits|under|common|H 1.5|lay_150_180|draftkings | 2 | no_bet_sample | market_only | 0.258 | - | 0.277 | 0.258 | 0.290 | 0.280 | 0.302 | 0.358 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 2 | no_bet_sample | distribution_blend | 0.061 | - | 0.124 | 0.138 | 0.170 | 0.063 | 0.061 | 0.109 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 2 | no_bet_sample | distribution_calibrated | 0.133 | - | 0.234 | 0.293 | 0.302 | 0.133 | 0.160 | 0.188 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 2 | no_bet_sample | event_side_line | 0.226 | - | 0.245 | 0.254 | 0.263 | 0.260 | 0.258 | 0.226 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 2 | no_bet_sample | distribution_blend | 0.161 | - | 0.246 | 0.331 | 0.357 | 0.161 | 0.161 | 0.243 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 2 | no_bet_sample | event_side_line | 0.032 | - | 0.246 | 0.204 | 0.192 | 0.044 | 0.035 | 0.032 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 2 | no_bet_sample | distribution_blend | 0.043 | - | 0.246 | 0.187 | 0.192 | 0.044 | 0.043 | 0.049 |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|fanduel | 2 | no_bet_sample | distribution_calibrated | 0.250 | - | 0.257 | 0.260 | 0.262 | 0.250 | 0.257 | 0.294 |
