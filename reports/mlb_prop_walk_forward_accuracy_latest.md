# MLB Prop Walk-Forward Accuracy

- Generated UTC: 2026-06-12T15:50:18+00:00
- Source: bets.mlb_prop_prediction_replay
- Date range: 2026-05-31 to 2026-06-12
- Locked rows: 51248
- Graded rows: 41206
- Pending rows with lock context: 10042
- Unique dates: 13
- Valid CLV rows: 35545 (69.4%)
- Avg CLV price: +0.04
- Live blend policy buckets: 18 exact, 5 line-surface, 5 market-side

This audit is walk-forward: blend weights use only earlier game dates, and valid CLV requires same book/player/stat/side/line close snapshots after lock and before first pitch.

## Overall Probability Variants

| Variant | Rows | Brier | Cal err | EV picks | ROI | CLV beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 41206 | 0.155 | -0.9% | 7713 | -8.6% | 42.8% | +0.19 |
| market_no_vig | 41010 | 0.154 | -1.8% | 4 | +112.0% | - | - |
| distribution | 41206 | 0.155 | +1.2% | 6930 | -14.7% | 40.3% | +0.17 |
| walk_forward_blend | 40183 | 0.153 | -1.6% | 3930 | -5.9% | 45.6% | +0.26 |

## Market And Side

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over | 17858 | 14747 | 13 | walk_forward_blend | 0.145 | 0.145 | 0.146 | 0.143 | 12852 | 37.9% | +0.07 |
| batter_total_bases|over | 17268 | 13560 | 13 | walk_forward_blend | 0.164 | 0.164 | 0.167 | 0.163 | 12056 | 36.8% | +0.08 |
| batter_home_runs|over | 7674 | 6002 | 12 | walk_forward_blend | 0.062 | 0.061 | 0.062 | 0.061 | 5344 | 32.5% | +0.03 |
| batter_hits|under | 4368 | 3479 | 13 | walk_forward_blend | 0.236 | 0.234 | 0.234 | 0.234 | 2800 | 33.7% | -0.12 |
| batter_total_bases|under | 2378 | 1970 | 13 | market_no_vig | 0.243 | 0.240 | 0.242 | 0.241 | 1502 | 32.2% | -0.21 |
| pitcher_strikeouts|over | 790 | 669 | 12 | distribution | 0.249 | 0.246 | 0.239 | 0.251 | 500 | 40.6% | +0.17 |
| pitcher_strikeouts|under | 765 | 644 | 12 | distribution | 0.246 | 0.245 | 0.236 | 0.249 | 491 | 36.9% | -0.13 |
| batter_home_runs|under | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Line Surface

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common | 11805 | 9269 | 13 | walk_forward_blend | 0.213 | 0.213 | 0.215 | 0.212 | 8057 | 39.0% | +0.08 |
| batter_total_bases|over|common | 9678 | 7634 | 13 | walk_forward_blend | 0.211 | 0.212 | 0.215 | 0.210 | 6736 | 38.7% | +0.10 |
| batter_total_bases|over|alt_tail | 7590 | 5926 | 10 | walk_forward_blend | 0.103 | 0.103 | 0.105 | 0.103 | 5320 | 34.4% | +0.06 |
| batter_hits|over|alt_tail | 6053 | 5478 | 9 | model_only | 0.029 | 0.029 | 0.029 | 0.029 | 4795 | 35.9% | +0.04 |
| batter_hits|under|common | 4368 | 3479 | 13 | walk_forward_blend | 0.236 | 0.234 | 0.234 | 0.234 | 2800 | 33.7% | -0.12 |
| batter_home_runs|over|common | 3866 | 3026 | 12 | market_no_vig | 0.114 | 0.114 | 0.115 | 0.114 | 2690 | 34.4% | +0.04 |
| batter_home_runs|over|alt_tail | 3808 | 2976 | 10 | market_no_vig | 0.008 | 0.008 | 0.008 | 0.008 | 2654 | 30.5% | +0.01 |
| batter_total_bases|under|common | 2378 | 1970 | 13 | market_no_vig | 0.243 | 0.240 | 0.242 | 0.241 | 1502 | 32.2% | -0.21 |
| pitcher_strikeouts|over|common | 790 | 669 | 12 | distribution | 0.249 | 0.246 | 0.239 | 0.251 | 500 | 40.6% | +0.17 |
| pitcher_strikeouts|under|common | 765 | 644 | 12 | distribution | 0.246 | 0.245 | 0.236 | 0.249 | 491 | 36.9% | -0.13 |
| batter_home_runs|under|common | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Exact Bucket

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 6034 | 5459 | 9 | model_only | 0.029 | 0.029 | 0.029 | 0.029 | 4776 | 35.8% | +0.04 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 5323 | 4106 | 10 | market_no_vig | 0.079 | 0.078 | 0.079 | 0.078 | 3624 | 30.9% | +0.03 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 3808 | 2976 | 10 | market_no_vig | 0.008 | 0.008 | 0.008 | 0.008 | 2654 | 30.5% | +0.01 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 2610 | 2021 | 12 | model_only | 0.090 | 0.090 | 0.091 | 0.091 | 1749 | 33.9% | +0.02 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 2334 | 1860 | 11 | distribution | 0.224 | 0.225 | 0.224 | 0.224 | 1695 | 34.9% | -0.02 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 2314 | 1800 | 10 | walk_forward_blend | 0.165 | 0.165 | 0.169 | 0.165 | 1577 | 38.4% | +0.03 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 2290 | 1789 | 10 | model_only | 0.153 | 0.154 | 0.153 | 0.153 | 1589 | 43.5% | +0.15 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 2041 | 1635 | 10 | walk_forward_blend | 0.151 | 0.151 | 0.152 | 0.151 | 1521 | 41.9% | +0.08 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 1939 | 1575 | 11 | distribution | 0.232 | 0.232 | 0.230 | 0.233 | 1269 | 36.2% | -0.04 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 1794 | 1442 | 12 | distribution | 0.239 | 0.241 | 0.237 | 0.239 | 1312 | 36.8% | +0.05 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 1809 | 1415 | 13 | distribution | 0.242 | 0.242 | 0.241 | 0.242 | 1202 | 31.2% | -0.24 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 1595 | 1323 | 12 | distribution | 0.232 | 0.233 | 0.230 | 0.232 | 987 | 37.1% | +0.11 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 1606 | 1310 | 13 | market_no_vig | 0.243 | 0.240 | 0.241 | 0.241 | 1089 | 42.1% | +0.29 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 1392 | 1046 | 13 | model_only | 0.212 | 0.216 | 0.219 | 0.213 | 892 | 40.9% | +0.18 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 1219 | 968 | 10 | walk_forward_blend | 0.207 | 0.206 | 0.213 | 0.205 | 895 | 36.4% | +0.03 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 1204 | 968 | 12 | market_no_vig | 0.159 | 0.157 | 0.158 | 0.158 | 904 | 34.6% | +0.05 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 1079 | 870 | 10 | walk_forward_blend | 0.194 | 0.195 | 0.196 | 0.194 | 825 | 37.9% | +0.10 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1001 | 827 | 13 | distribution | 0.233 | 0.235 | 0.232 | 0.234 | 668 | 31.0% | -0.26 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 966 | 754 | 10 | model_only | 0.247 | 0.247 | 0.251 | 0.247 | 668 | 41.2% | +0.07 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 861 | 688 | 11 | walk_forward_blend | 0.243 | 0.243 | 0.244 | 0.241 | 572 | 39.7% | +0.09 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 552 | 453 | 10 | market_no_vig | 0.220 | 0.212 | 0.222 | 0.214 | 369 | 35.2% | -0.22 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 534 | 438 | 13 | distribution | 0.235 | 0.236 | 0.235 | 0.238 | 283 | 24.7% | -0.49 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 502 | 415 | 10 | market_no_vig | 0.215 | 0.212 | 0.218 | 0.213 | 334 | 39.5% | +0.23 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 499 | 370 | 11 | walk_forward_blend | 0.245 | 0.245 | 0.253 | 0.244 | 317 | 47.6% | +0.60 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 411 | 337 | 12 | market_no_vig | 0.258 | 0.250 | 0.256 | 0.256 | 285 | 35.1% | -0.09 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 393 | 309 | 10 | walk_forward_blend | 0.253 | 0.251 | 0.265 | 0.250 | 296 | 28.7% | -0.14 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 368 | 269 | 11 | distribution | 0.247 | 0.251 | 0.245 | 0.247 | 229 | 42.4% | +0.17 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 285 | 248 | 12 | market_no_vig | 0.261 | 0.248 | 0.267 | 0.253 | 221 | 39.8% | +0.08 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 283 | 239 | 11 | walk_forward_blend | 0.255 | 0.250 | 0.271 | 0.250 | 221 | 35.7% | -0.23 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 327 | 220 | 13 | distribution | 0.247 | 0.245 | 0.245 | 0.246 | 185 | 24.9% | -0.62 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 283 | 198 | 10 | market_no_vig | 0.112 | 0.111 | 0.114 | 0.112 | 166 | 34.9% | +0.21 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 279 | 186 | 13 | distribution | 0.252 | 0.247 | 0.244 | 0.248 | 155 | 43.2% | +0.63 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 206 | 173 | 10 | market_no_vig | 0.226 | 0.224 | 0.253 | 0.225 | 163 | 42.9% | +0.44 |
| batter_home_runs|under|common|HR 0.5|missing_price|fanduel | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 200 | 133 | 10 | model_only | 0.107 | 0.108 | 0.108 | 0.107 | 106 | 31.1% | +0.12 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 147 | 127 | 12 | distribution | 0.238 | 0.247 | 0.236 | 0.243 | 90 | 56.7% | +0.85 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 120 | 97 | 9 | market_no_vig | 0.254 | 0.239 | 0.245 | 0.247 | 76 | 34.2% | +0.19 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 116 | 95 | 10 | model_only | 0.250 | 0.258 | 0.262 | 0.252 | 85 | 49.4% | +0.38 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 109 | 87 | 10 | market_no_vig | 0.256 | 0.248 | 0.294 | 0.251 | 84 | 46.4% | +0.40 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 130 | 84 | 10 | distribution | 0.256 | 0.260 | 0.250 | 0.259 | 67 | 40.3% | +0.36 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 101 | 82 | 10 | market_no_vig | 0.227 | 0.212 | 0.235 | 0.220 | 75 | 52.0% | -0.39 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 94 | 75 | 12 | model_only | 0.247 | 0.266 | 0.251 | 0.263 | 43 | 51.2% | +0.27 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 84 | 71 | 9 | market_no_vig | 0.230 | 0.219 | 0.232 | 0.227 | 68 | 32.4% | -0.43 |
| batter_total_bases|under|common|TB 1.5|missing_price|fanduel | 70 | 61 | 1 | model_only | 0.238 | - | 0.252 | - | 0 | - | - |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 68 | 60 | 11 | distribution | 0.247 | 0.249 | 0.239 | 0.247 | 47 | 44.7% | +0.56 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 67 | 58 | 12 | distribution | 0.235 | 0.244 | 0.232 | 0.249 | 50 | 40.0% | -0.57 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 77 | 57 | 10 | distribution | 0.189 | 0.196 | 0.180 | 0.193 | 32 | 56.2% | +0.61 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 66 | 57 | 10 | distribution | 0.244 | 0.248 | 0.238 | 0.246 | 47 | 53.2% | -0.37 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 63 | 53 | 9 | distribution | 0.248 | 0.249 | 0.235 | 0.249 | 44 | 29.5% | +0.25 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 61 | 50 | 9 | distribution | 0.227 | 0.223 | 0.223 | 0.229 | 34 | 26.5% | -1.23 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 59 | 49 | 8 | distribution | 0.249 | 0.249 | 0.245 | 0.250 | 41 | 26.8% | -0.67 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 61 | 49 | 8 | model_only | 0.250 | 0.274 | 0.259 | 0.270 | 46 | 17.4% | -0.47 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 66 | 48 | 9 | market_no_vig | 0.251 | 0.245 | 0.260 | 0.248 | 33 | 24.2% | -0.27 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 52 | 47 | 8 | market_no_vig | 0.249 | 0.234 | 0.235 | 0.241 | 42 | 47.6% | +0.31 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 60 | 46 | 8 | market_no_vig | 0.240 | 0.232 | 0.281 | 0.235 | 46 | 30.4% | -0.56 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 57 | 45 | 11 | market_no_vig | 0.263 | 0.250 | 0.294 | 0.256 | 37 | 48.6% | -0.36 |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 54 | 41 | 10 | walk_forward_blend | 0.260 | 0.250 | 0.250 | 0.243 | 30 | 46.7% | +0.26 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 55 | 40 | 9 | model_only | 0.243 | 0.245 | 0.274 | 0.244 | 29 | 37.9% | +0.33 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 48 | 39 | 10 | distribution | 0.253 | 0.254 | 0.231 | 0.261 | 25 | 44.0% | -0.12 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 52 | 37 | 8 | market_no_vig | 0.253 | 0.235 | 0.283 | 0.247 | 37 | 54.1% | +0.85 |

## Opportunity Diagnostics

| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|batter_pa_3_8_to_4_2 | 7917 | 6708 | 4.02 | - | 0.163 | 0.163 | +19.1% | 5690 |
| batter_hits|batter_pa_under_3_8 | 8479 | 6607 | 3.32 | - | 0.164 | 0.163 | -13.9% | 5563 |
| batter_total_bases|batter_pa_3_8_to_4_2 | 7183 | 5888 | 4.02 | - | 0.181 | 0.180 | -9.2% | 5045 |
| batter_total_bases|batter_pa_under_3_8 | 6630 | 4944 | 3.33 | - | 0.148 | 0.147 | -11.2% | 4239 |
| batter_hits|batter_pa_4_3_plus | 5830 | 4911 | 4.47 | - | 0.160 | 0.159 | +9.1% | 4399 |
| batter_total_bases|batter_pa_4_3_plus | 5833 | 4698 | 4.47 | - | 0.193 | 0.192 | -12.1% | 4274 |
| batter_home_runs|batter_pa_under_3_8 | 3024 | 2252 | 3.32 | - | 0.051 | 0.050 | -68.3% | 1916 |
| batter_home_runs|batter_pa_3_8_to_4_2 | 2764 | 2247 | 4.01 | - | 0.066 | 0.064 | -27.6% | 1930 |
| batter_home_runs|batter_pa_4_3_plus | 2033 | 1638 | 4.47 | - | 0.076 | 0.073 | -62.0% | 1498 |
| pitcher_strikeouts|pitcher_bf_20_to_23 | 830 | 684 | - | 22.2 | 0.246 | 0.245 | +1.0% | 538 |
| pitcher_strikeouts|pitcher_bf_24_plus | 660 | 564 | - | 25.1 | 0.250 | 0.248 | +0.8% | 406 |
| pitcher_strikeouts|pitcher_bf_under_20 | 65 | 65 | - | 18.5 | 0.248 | 0.225 | -38.6% | 47 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 9352 |
| stale_close_before_lock | 5024 |
| missing_lock_snapshot | 901 |
| fallback_other_book_only | 310 |
| no_valid_close_snapshot | 116 |

## Live Probability Policy

| Level | Bucket | Variant | Rows | Dates | Brier Gain | Model Weight |
|---|---|---|---:|---:|---:|---:|
| exact_bucket | batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | market_no_vig | 248 | 12 | +0.013 | 0.000 |
| market_side | pitcher_strikeouts|over | distribution | 669 | 12 | +0.010 | 0.000 |
| line_surface | pitcher_strikeouts|over|common | distribution | 669 | 12 | +0.010 | 0.000 |
| market_side | pitcher_strikeouts|under | distribution | 644 | 12 | +0.010 | 0.400 |
| line_surface | pitcher_strikeouts|under|common | distribution | 644 | 12 | +0.010 | 0.400 |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | market_no_vig | 453 | 10 | +0.009 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | market_no_vig | 337 | 12 | +0.008 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|draftkings | distribution | 186 | 13 | +0.008 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | walk_forward_blend | 239 | 11 | +0.006 | 0.000 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | market_no_vig | 415 | 10 | +0.003 | 0.000 |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | distribution | 1323 | 12 | +0.003 | 0.700 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | walk_forward_blend | 309 | 10 | +0.003 | 0.400 |
| market_side | batter_total_bases|under | market_no_vig | 1970 | 13 | +0.002 | 0.300 |
| line_surface | batter_total_bases|under|common | market_no_vig | 1970 | 13 | +0.002 | 0.300 |
| exact_bucket | batter_hits|under|common|H 0.5|fair_lay|draftkings | distribution | 220 | 13 | +0.002 | 0.200 |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | market_no_vig | 968 | 12 | +0.002 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | market_no_vig | 1310 | 13 | +0.002 | 0.000 |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | market_no_vig | 173 | 10 | +0.002 | 0.000 |
| market_side | batter_hits|over | walk_forward_blend | 14747 | 13 | +0.002 | 0.500 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|draftkings | walk_forward_blend | 688 | 11 | +0.002 | 0.300 |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|draftkings | distribution | 1575 | 11 | +0.002 | 0.700 |
| market_side | batter_hits|under | walk_forward_blend | 3479 | 13 | +0.002 | 0.200 |
| line_surface | batter_hits|under|common | walk_forward_blend | 3479 | 13 | +0.002 | 0.200 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|fanduel | distribution | 269 | 11 | +0.002 | 1.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | distribution | 1442 | 12 | +0.001 | 0.700 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|fanduel | walk_forward_blend | 968 | 10 | +0.001 | 0.300 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|draftkings | walk_forward_blend | 370 | 11 | +0.001 | 0.500 |
| line_surface | batter_hits|over|common | walk_forward_blend | 9269 | 13 | +0.001 | 0.500 |

## Reading This

- `model_only` is the locked player-prop probability.
- `market_no_vig` is the book market baseline after removing vig when both sides were available.
- `distribution` prices the exact line from the locked projected count with stat-specific curves; total bases uses a compound PA/single/double/triple/HR shape.
- `walk_forward_blend` picks a model/market weight from prior dates only.
- A bucket is not real-money ready merely because it appears here; it still needs enough graded rows, valid CLV, ROI, calibration, and concentration checks.
