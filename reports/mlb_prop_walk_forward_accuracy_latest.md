# MLB Prop Walk-Forward Accuracy

- Generated UTC: 2026-06-17T17:53:18+00:00
- Source: bets.mlb_prop_prediction_replay
- Date range: 2026-05-31 to 2026-06-17
- Locked rows: 83646
- Graded rows: 65898
- Pending rows with lock context: 17748
- Unique dates: 18
- Valid CLV rows: 67357 (80.5%)
- Avg CLV price: +0.06
- Live blend policy buckets: 14 exact, 4 line-surface, 5 market-side

This audit is walk-forward: blend weights use only earlier game dates, and valid CLV requires same book/player/stat/side/line close snapshots after lock and before first pitch.

## Overall Probability Variants

| Variant | Rows | Brier | Cal err | EV picks | ROI | CLV beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 65898 | 0.160 | -1.2% | 13133 | -8.2% | 42.4% | +0.20 |
| market_no_vig | 65702 | 0.160 | -2.0% | 4 | +112.0% | - | - |
| distribution | 65898 | 0.162 | +1.3% | 12099 | -9.6% | 39.7% | +0.09 |
| walk_forward_blend | 64875 | 0.159 | -1.8% | 7645 | -10.6% | 43.7% | +0.25 |

## Market And Side

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 29702 | 23004 | 18 | walk_forward_blend | 0.163 | 0.164 | 0.166 | 0.163 | 24357 | 38.4% | +0.13 |
| batter_hits|over | 26910 | 21533 | 18 | walk_forward_blend | 0.162 | 0.162 | 0.163 | 0.161 | 21684 | 39.4% | +0.12 |
| batter_home_runs|over | 13180 | 10138 | 17 | walk_forward_blend | 0.064 | 0.063 | 0.064 | 0.063 | 10788 | 34.0% | +0.04 |
| batter_hits|under | 7208 | 5631 | 18 | walk_forward_blend | 0.234 | 0.232 | 0.234 | 0.232 | 5488 | 32.5% | -0.25 |
| batter_total_bases|under | 3792 | 3138 | 18 | market_no_vig | 0.242 | 0.240 | 0.245 | 0.241 | 2899 | 31.3% | -0.32 |
| pitcher_strikeouts|over | 1366 | 1172 | 16 | market_no_vig | 0.251 | 0.248 | 0.252 | 0.251 | 1075 | 35.7% | -0.28 |
| pitcher_strikeouts|under | 1341 | 1147 | 16 | market_no_vig | 0.251 | 0.248 | 0.251 | 0.251 | 1066 | 44.8% | +0.34 |
| batter_home_runs|under | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Line Surface

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common | 20155 | 15559 | 18 | walk_forward_blend | 0.212 | 0.212 | 0.214 | 0.212 | 16197 | 41.0% | +0.15 |
| batter_total_bases|over|common | 16602 | 12940 | 18 | walk_forward_blend | 0.210 | 0.211 | 0.214 | 0.209 | 13585 | 40.8% | +0.16 |
| batter_total_bases|over|alt_tail | 13100 | 10064 | 15 | walk_forward_blend | 0.103 | 0.103 | 0.105 | 0.103 | 10772 | 35.4% | +0.08 |
| batter_hits|over|alt_tail | 6755 | 5974 | 15 | model_only | 0.030 | 0.030 | 0.030 | 0.030 | 5487 | 34.6% | +0.04 |
| batter_hits|under|common | 7208 | 5631 | 18 | walk_forward_blend | 0.234 | 0.232 | 0.234 | 0.232 | 5488 | 32.5% | -0.25 |
| batter_home_runs|over|common | 6622 | 5095 | 17 | market_no_vig | 0.117 | 0.116 | 0.117 | 0.116 | 5417 | 35.7% | +0.06 |
| batter_home_runs|over|alt_tail | 6558 | 5043 | 15 | market_no_vig | 0.011 | 0.011 | 0.011 | 0.011 | 5371 | 32.3% | +0.02 |
| batter_total_bases|under|common | 3792 | 3138 | 18 | market_no_vig | 0.242 | 0.240 | 0.245 | 0.241 | 2899 | 31.3% | -0.32 |
| pitcher_strikeouts|over|common | 1366 | 1172 | 16 | market_no_vig | 0.251 | 0.248 | 0.252 | 0.251 | 1075 | 35.7% | -0.28 |
| pitcher_strikeouts|under|common | 1341 | 1147 | 16 | market_no_vig | 0.251 | 0.248 | 0.251 | 0.251 | 1066 | 44.8% | +0.34 |
| batter_home_runs|under|common | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Exact Bucket

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 9029 | 6797 | 15 | market_no_vig | 0.076 | 0.076 | 0.076 | 0.076 | 7177 | 32.2% | +0.05 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 6736 | 5955 | 15 | model_only | 0.029 | 0.029 | 0.029 | 0.029 | 5468 | 34.6% | +0.04 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 6558 | 5043 | 15 | market_no_vig | 0.011 | 0.011 | 0.011 | 0.011 | 5371 | 32.3% | +0.02 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 4381 | 3296 | 17 | market_no_vig | 0.089 | 0.089 | 0.089 | 0.089 | 3431 | 35.0% | +0.04 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 4079 | 3248 | 16 | model_only | 0.222 | 0.223 | 0.224 | 0.222 | 3501 | 37.1% | +0.05 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 3923 | 2979 | 15 | walk_forward_blend | 0.161 | 0.161 | 0.162 | 0.160 | 3120 | 40.2% | +0.10 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 3868 | 2927 | 15 | distribution | 0.152 | 0.153 | 0.152 | 0.153 | 3070 | 44.3% | +0.20 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 3636 | 2900 | 15 | walk_forward_blend | 0.154 | 0.153 | 0.156 | 0.153 | 3196 | 41.7% | +0.10 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3253 | 2610 | 16 | market_no_vig | 0.227 | 0.227 | 0.228 | 0.227 | 2534 | 39.9% | +0.04 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 3052 | 2428 | 17 | model_only | 0.236 | 0.240 | 0.237 | 0.237 | 2625 | 38.2% | +0.11 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 2994 | 2260 | 18 | walk_forward_blend | 0.241 | 0.241 | 0.243 | 0.240 | 2337 | 30.1% | -0.36 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 2734 | 2227 | 18 | market_no_vig | 0.240 | 0.239 | 0.242 | 0.239 | 2205 | 45.7% | +0.36 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2556 | 2092 | 17 | walk_forward_blend | 0.229 | 0.228 | 0.227 | 0.227 | 1890 | 36.1% | +0.04 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 2191 | 1759 | 15 | walk_forward_blend | 0.204 | 0.205 | 0.212 | 0.204 | 1934 | 37.0% | +0.03 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 2132 | 1712 | 17 | market_no_vig | 0.164 | 0.162 | 0.165 | 0.163 | 1890 | 36.1% | +0.06 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2378 | 1688 | 18 | model_only | 0.207 | 0.211 | 0.210 | 0.208 | 1747 | 42.0% | +0.25 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 1926 | 1555 | 15 | model_only | 0.201 | 0.203 | 0.209 | 0.202 | 1732 | 39.3% | +0.10 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1584 | 1298 | 18 | model_only | 0.236 | 0.238 | 0.239 | 0.237 | 1289 | 30.6% | -0.31 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1641 | 1220 | 15 | model_only | 0.250 | 0.251 | 0.251 | 0.251 | 1269 | 41.8% | +0.14 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1408 | 1080 | 16 | walk_forward_blend | 0.244 | 0.244 | 0.247 | 0.243 | 1103 | 44.1% | +0.22 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 949 | 785 | 15 | market_no_vig | 0.219 | 0.214 | 0.224 | 0.215 | 741 | 34.7% | -0.43 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 885 | 723 | 18 | model_only | 0.232 | 0.232 | 0.233 | 0.233 | 538 | 24.2% | -0.59 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 871 | 720 | 15 | market_no_vig | 0.216 | 0.214 | 0.221 | 0.214 | 673 | 44.7% | +0.42 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 738 | 598 | 15 | walk_forward_blend | 0.252 | 0.251 | 0.269 | 0.250 | 667 | 31.0% | -0.10 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 818 | 586 | 16 | market_no_vig | 0.251 | 0.250 | 0.255 | 0.250 | 594 | 48.0% | +0.66 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 691 | 571 | 17 | market_no_vig | 0.252 | 0.247 | 0.255 | 0.250 | 591 | 34.5% | -0.25 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 589 | 420 | 16 | distribution | 0.249 | 0.255 | 0.249 | 0.251 | 434 | 40.8% | +0.17 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 470 | 406 | 16 | walk_forward_blend | 0.253 | 0.250 | 0.270 | 0.250 | 432 | 40.3% | +0.03 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 447 | 387 | 17 | market_no_vig | 0.258 | 0.249 | 0.267 | 0.252 | 400 | 39.0% | -0.12 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 566 | 376 | 18 | market_no_vig | 0.249 | 0.248 | 0.249 | 0.249 | 402 | 24.1% | -0.67 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 398 | 336 | 15 | model_only | 0.206 | 0.209 | 0.226 | 0.208 | 368 | 42.7% | +0.34 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 491 | 320 | 15 | distribution | 0.096 | 0.096 | 0.094 | 0.096 | 343 | 37.9% | +0.25 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 487 | 318 | 18 | distribution | 0.254 | 0.249 | 0.248 | 0.250 | 342 | 48.8% | +0.69 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 351 | 223 | 15 | distribution | 0.089 | 0.089 | 0.088 | 0.089 | 254 | 40.2% | +0.25 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 238 | 194 | 16 | model_only | 0.249 | 0.252 | 0.256 | 0.251 | 166 | 52.4% | +0.76 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 190 | 166 | 15 | walk_forward_blend | 0.254 | 0.259 | 0.253 | 0.253 | 171 | 47.4% | +0.17 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 187 | 157 | 15 | walk_forward_blend | 0.245 | 0.245 | 0.284 | 0.244 | 170 | 46.5% | +0.46 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 202 | 154 | 13 | market_no_vig | 0.253 | 0.244 | 0.252 | 0.248 | 166 | 28.3% | -0.01 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 153 | 141 | 16 | model_only | 0.250 | 0.262 | 0.267 | 0.258 | 96 | 57.3% | +1.18 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 223 | 136 | 15 | distribution | 0.253 | 0.255 | 0.252 | 0.255 | 161 | 42.2% | +0.18 |
| batter_home_runs|under|common|HR 0.5|missing_price|fanduel | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 145 | 129 | 15 | market_no_vig | 0.221 | 0.210 | 0.224 | 0.216 | 125 | 55.2% | +0.35 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 136 | 119 | 14 | distribution | 0.228 | 0.218 | 0.214 | 0.224 | 124 | 32.3% | -0.39 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 116 | 110 | 12 | model_only | 0.249 | 0.257 | 0.265 | 0.256 | 103 | 35.9% | +0.17 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 131 | 109 | 14 | distribution | 0.247 | 0.248 | 0.243 | 0.248 | 116 | 44.8% | -0.70 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 131 | 107 | 15 | distribution | 0.248 | 0.250 | 0.241 | 0.249 | 115 | 51.3% | +0.74 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 133 | 100 | 15 | distribution | 0.216 | 0.217 | 0.216 | 0.217 | 62 | 56.5% | +0.88 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 113 | 95 | 12 | distribution | 0.252 | 0.250 | 0.249 | 0.250 | 105 | 23.8% | -0.48 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 115 | 93 | 13 | distribution | 0.249 | 0.249 | 0.239 | 0.250 | 105 | 37.1% | +0.45 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 110 | 91 | 13 | market_no_vig | 0.250 | 0.246 | 0.270 | 0.250 | 69 | 30.4% | -1.33 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 103 | 89 | 13 | model_only | 0.234 | 0.255 | 0.269 | 0.240 | 96 | 44.8% | +0.08 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 106 | 84 | 13 | market_no_vig | 0.235 | 0.219 | 0.229 | 0.224 | 93 | 49.5% | +0.89 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 105 | 82 | 16 | distribution | 0.244 | 0.243 | 0.233 | 0.251 | 87 | 47.1% | -0.28 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 104 | 82 | 13 | walk_forward_blend | 0.252 | 0.249 | 0.270 | 0.249 | 76 | 25.0% | -0.11 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 76 | 71 | 14 | distribution | 0.252 | 0.254 | 0.244 | 0.257 | 56 | 33.9% | -1.19 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 88 | 69 | 12 | distribution | 0.251 | 0.239 | 0.237 | 0.244 | 82 | 46.3% | +0.36 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 77 | 63 | 16 | walk_forward_blend | 0.248 | 0.254 | 0.283 | 0.247 | 57 | 54.4% | -0.08 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 89 | 63 | 14 | model_only | 0.219 | 0.219 | 0.249 | 0.220 | 66 | 48.5% | +0.56 |
| batter_total_bases|under|common|TB 1.5|missing_price|fanduel | 70 | 61 | 1 | model_only | 0.238 | - | 0.252 | - | 0 | - | - |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 64 | 61 | 15 | market_no_vig | 0.267 | 0.233 | 0.263 | 0.262 | 43 | 41.9% | +0.36 |

## Opportunity Diagnostics

| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|batter_pa_under_3_8 | 13178 | 9885 | 3.33 | - | 0.178 | 0.178 | -16.1% | 9782 |
| batter_hits|batter_pa_3_8_to_4_2 | 11950 | 9740 | 4.02 | - | 0.177 | 0.177 | +8.3% | 9685 |
| batter_total_bases|batter_pa_3_8_to_4_2 | 11907 | 9472 | 4.02 | - | 0.180 | 0.180 | -11.4% | 9785 |
| batter_total_bases|batter_pa_under_3_8 | 11594 | 8450 | 3.33 | - | 0.148 | 0.148 | -8.7% | 8671 |
| batter_total_bases|batter_pa_4_3_plus | 9993 | 8220 | 4.49 | - | 0.189 | 0.190 | -19.2% | 8800 |
| batter_hits|batter_pa_4_3_plus | 8990 | 7539 | 4.48 | - | 0.173 | 0.173 | +9.0% | 7705 |
| batter_home_runs|batter_pa_under_3_8 | 5242 | 3792 | 3.33 | - | 0.053 | 0.052 | +11.3% | 3907 |
| batter_home_runs|batter_pa_3_8_to_4_2 | 4604 | 3635 | 4.02 | - | 0.069 | 0.067 | -0.4% | 3803 |
| batter_home_runs|batter_pa_4_3_plus | 3481 | 2846 | 4.48 | - | 0.076 | 0.074 | -42.8% | 3078 |
| pitcher_strikeouts|pitcher_bf_20_to_23 | 1506 | 1270 | - | 22.2 | 0.249 | 0.249 | +2.8% | 1216 |
| pitcher_strikeouts|pitcher_bf_24_plus | 1030 | 918 | - | 25.2 | 0.253 | 0.249 | -6.3% | 800 |
| pitcher_strikeouts|pitcher_bf_under_20 | 171 | 131 | - | 18.7 | 0.245 | 0.229 | -29.8% | 125 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 8562 |
| stale_close_before_lock | 5900 |
| missing_lock_snapshot | 901 |
| fallback_other_book_only | 734 |
| no_valid_close_snapshot | 192 |

## Live Probability Policy

| Level | Bucket | Variant | Rows | Dates | Brier Gain | Model Weight |
|---|---|---|---:|---:|---:|---:|
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | market_no_vig | 154 | 13 | +0.009 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | market_no_vig | 387 | 17 | +0.008 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|draftkings | distribution | 318 | 18 | +0.005 | 0.000 |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | market_no_vig | 785 | 15 | +0.005 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | market_no_vig | 571 | 17 | +0.005 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | walk_forward_blend | 406 | 16 | +0.003 | 0.000 |
| market_side | pitcher_strikeouts|under | market_no_vig | 1147 | 16 | +0.003 | 0.200 |
| line_surface | pitcher_strikeouts|under|common | market_no_vig | 1147 | 16 | +0.003 | 0.200 |
| market_side | pitcher_strikeouts|over | market_no_vig | 1172 | 16 | +0.003 | 0.100 |
| line_surface | pitcher_strikeouts|over|common | market_no_vig | 1172 | 16 | +0.003 | 0.100 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | market_no_vig | 720 | 15 | +0.002 | 0.000 |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | walk_forward_blend | 2092 | 17 | +0.002 | 0.200 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | distribution | 320 | 15 | +0.002 | 0.000 |
| market_side | batter_hits|under | walk_forward_blend | 5631 | 18 | +0.002 | 0.100 |
| line_surface | batter_hits|under|common | walk_forward_blend | 5631 | 18 | +0.002 | 0.100 |
| market_side | batter_total_bases|under | market_no_vig | 3138 | 18 | +0.002 | 0.300 |
| line_surface | batter_total_bases|under|common | market_no_vig | 3138 | 18 | +0.002 | 0.300 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | walk_forward_blend | 598 | 15 | +0.002 | 0.400 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | walk_forward_blend | 157 | 15 | +0.001 | 0.500 |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | market_no_vig | 1712 | 17 | +0.001 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|draftkings | market_no_vig | 586 | 16 | +0.001 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|draftkings | walk_forward_blend | 1080 | 16 | +0.001 | 0.200 |
| market_side | batter_hits|over | walk_forward_blend | 21533 | 18 | +0.001 | 0.600 |

## Reading This

- `model_only` is the locked player-prop probability.
- `market_no_vig` is the book market baseline after removing vig when both sides were available.
- `distribution` prices the exact line from the locked projected count with stat-specific curves; total bases uses a compound PA/single/double/triple/HR shape.
- `walk_forward_blend` picks a model/market weight from prior dates only.
- A bucket is not real-money ready merely because it appears here; it still needs enough graded rows, valid CLV, ROI, calibration, and concentration checks.
