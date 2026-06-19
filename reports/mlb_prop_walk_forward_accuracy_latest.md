# MLB Prop Walk-Forward Accuracy

- Generated UTC: 2026-06-19T19:51:32+00:00
- Source: bets.mlb_prop_prediction_replay
- Date range: 2026-05-31 to 2026-06-19
- Locked rows: 92555
- Graded rows: 79111
- Pending rows with lock context: 13444
- Unique dates: 20
- Valid CLV rows: 71853 (77.6%)
- Avg CLV price: +0.06
- Live blend policy buckets: 12 exact, 4 line-surface, 4 market-side

This audit is walk-forward: blend weights use only earlier game dates, and valid CLV requires same book/player/stat/side/line close snapshots after lock and before first pitch.

## Overall Probability Variants

| Variant | Rows | Brier | Cal err | EV picks | ROI | CLV beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 79111 | 0.161 | -1.3% | 16552 | -11.7% | 41.7% | +0.20 |
| market_no_vig | 78915 | 0.161 | -2.1% | 4 | +112.0% | - | - |
| distribution | 79111 | 0.163 | +1.3% | 15464 | -11.1% | 39.7% | +0.09 |
| walk_forward_blend | 78088 | 0.160 | -1.8% | 10176 | -12.1% | 42.3% | +0.24 |

## Market And Side

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 33149 | 28100 | 20 | walk_forward_blend | 0.164 | 0.164 | 0.167 | 0.163 | 26135 | 38.5% | +0.13 |
| batter_hits|over | 29376 | 25267 | 20 | walk_forward_blend | 0.167 | 0.167 | 0.168 | 0.166 | 22974 | 39.4% | +0.12 |
| batter_home_runs|over | 14724 | 12407 | 19 | walk_forward_blend | 0.064 | 0.064 | 0.064 | 0.063 | 11592 | 34.4% | +0.05 |
| batter_hits|under | 8003 | 6765 | 20 | walk_forward_blend | 0.233 | 0.232 | 0.234 | 0.232 | 5850 | 32.5% | -0.25 |
| batter_total_bases|under | 4151 | 3690 | 20 | market_no_vig | 0.241 | 0.240 | 0.243 | 0.240 | 3069 | 30.9% | -0.34 |
| pitcher_strikeouts|over | 1515 | 1386 | 18 | market_no_vig | 0.249 | 0.247 | 0.254 | 0.250 | 1121 | 35.3% | -0.32 |
| pitcher_strikeouts|under | 1490 | 1361 | 18 | market_no_vig | 0.249 | 0.247 | 0.253 | 0.249 | 1112 | 45.3% | +0.37 |
| batter_home_runs|under | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Line Surface

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common | 22494 | 18965 | 20 | walk_forward_blend | 0.212 | 0.212 | 0.214 | 0.212 | 17363 | 41.0% | +0.15 |
| batter_total_bases|over|common | 18505 | 15764 | 20 | walk_forward_blend | 0.211 | 0.212 | 0.215 | 0.210 | 14559 | 40.8% | +0.17 |
| batter_total_bases|over|alt_tail | 14644 | 12336 | 17 | market_no_vig | 0.103 | 0.103 | 0.104 | 0.103 | 11576 | 35.7% | +0.09 |
| batter_hits|under|common | 8003 | 6765 | 20 | walk_forward_blend | 0.233 | 0.232 | 0.234 | 0.232 | 5850 | 32.5% | -0.25 |
| batter_hits|over|alt_tail | 6882 | 6302 | 16 | model_only | 0.030 | 0.030 | 0.030 | 0.030 | 5611 | 34.5% | +0.04 |
| batter_home_runs|over|common | 7394 | 6231 | 19 | market_no_vig | 0.117 | 0.117 | 0.118 | 0.117 | 5819 | 35.7% | +0.08 |
| batter_home_runs|over|alt_tail | 7330 | 6176 | 17 | market_no_vig | 0.010 | 0.010 | 0.010 | 0.010 | 5773 | 33.0% | +0.02 |
| batter_total_bases|under|common | 4151 | 3690 | 20 | market_no_vig | 0.241 | 0.240 | 0.243 | 0.240 | 3069 | 30.9% | -0.34 |
| pitcher_strikeouts|over|common | 1515 | 1386 | 18 | market_no_vig | 0.249 | 0.247 | 0.254 | 0.250 | 1121 | 35.3% | -0.32 |
| pitcher_strikeouts|under|common | 1490 | 1361 | 18 | market_no_vig | 0.249 | 0.247 | 0.253 | 0.249 | 1112 | 45.3% | +0.37 |
| batter_home_runs|under|common | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Exact Bucket

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 10081 | 8304 | 17 | market_no_vig | 0.076 | 0.076 | 0.076 | 0.076 | 7697 | 32.3% | +0.06 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 6863 | 6283 | 16 | model_only | 0.029 | 0.029 | 0.030 | 0.029 | 5592 | 34.5% | +0.04 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 7330 | 6176 | 17 | market_no_vig | 0.010 | 0.010 | 0.010 | 0.010 | 5773 | 33.0% | +0.02 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 4892 | 4014 | 19 | market_no_vig | 0.090 | 0.090 | 0.091 | 0.090 | 3673 | 34.9% | +0.04 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 4549 | 3986 | 18 | model_only | 0.223 | 0.225 | 0.224 | 0.224 | 3757 | 37.3% | +0.05 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 4387 | 3633 | 17 | market_no_vig | 0.164 | 0.164 | 0.167 | 0.164 | 3351 | 39.8% | +0.09 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 4072 | 3586 | 17 | walk_forward_blend | 0.152 | 0.152 | 0.154 | 0.152 | 3446 | 42.0% | +0.11 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 4321 | 3565 | 17 | model_only | 0.155 | 0.156 | 0.155 | 0.155 | 3308 | 44.0% | +0.19 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3608 | 3155 | 18 | market_no_vig | 0.228 | 0.228 | 0.228 | 0.228 | 2700 | 39.9% | +0.04 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 3405 | 2984 | 19 | model_only | 0.237 | 0.241 | 0.239 | 0.238 | 2812 | 38.5% | +0.12 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 3342 | 2747 | 20 | walk_forward_blend | 0.241 | 0.241 | 0.242 | 0.241 | 2517 | 30.5% | -0.35 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 3024 | 2680 | 20 | market_no_vig | 0.241 | 0.239 | 0.242 | 0.240 | 2341 | 46.3% | +0.38 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2806 | 2477 | 19 | distribution | 0.229 | 0.228 | 0.227 | 0.227 | 1994 | 36.3% | +0.04 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 2441 | 2171 | 17 | model_only | 0.200 | 0.201 | 0.206 | 0.200 | 2073 | 37.0% | +0.03 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 2380 | 2111 | 19 | market_no_vig | 0.164 | 0.163 | 0.165 | 0.163 | 2043 | 36.6% | +0.10 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2656 | 2069 | 20 | model_only | 0.208 | 0.211 | 0.212 | 0.209 | 1888 | 41.7% | +0.23 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 2146 | 1935 | 17 | model_only | 0.202 | 0.204 | 0.210 | 0.203 | 1873 | 39.6% | +0.12 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1727 | 1540 | 20 | model_only | 0.237 | 0.238 | 0.240 | 0.238 | 1358 | 30.3% | -0.33 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1829 | 1479 | 17 | model_only | 0.247 | 0.248 | 0.249 | 0.247 | 1371 | 41.1% | +0.11 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1575 | 1303 | 18 | walk_forward_blend | 0.244 | 0.244 | 0.247 | 0.243 | 1174 | 43.5% | +0.20 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1047 | 936 | 17 | market_no_vig | 0.216 | 0.212 | 0.219 | 0.213 | 787 | 33.4% | -0.46 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 963 | 860 | 17 | market_no_vig | 0.213 | 0.211 | 0.216 | 0.211 | 715 | 46.3% | +0.46 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 980 | 850 | 20 | model_only | 0.231 | 0.231 | 0.232 | 0.232 | 573 | 23.7% | -0.62 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 826 | 746 | 17 | walk_forward_blend | 0.252 | 0.252 | 0.266 | 0.251 | 728 | 32.6% | -0.08 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 905 | 702 | 18 | walk_forward_blend | 0.249 | 0.248 | 0.252 | 0.248 | 647 | 47.9% | +0.65 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 759 | 684 | 19 | market_no_vig | 0.251 | 0.247 | 0.252 | 0.249 | 632 | 34.2% | -0.28 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 658 | 506 | 18 | model_only | 0.249 | 0.255 | 0.251 | 0.250 | 466 | 41.0% | +0.17 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 517 | 474 | 18 | walk_forward_blend | 0.252 | 0.249 | 0.261 | 0.249 | 455 | 40.4% | +0.03 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 644 | 470 | 20 | market_no_vig | 0.249 | 0.248 | 0.251 | 0.248 | 430 | 24.2% | -0.69 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 490 | 451 | 19 | market_no_vig | 0.256 | 0.248 | 0.259 | 0.251 | 423 | 38.1% | -0.12 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 454 | 415 | 17 | model_only | 0.209 | 0.213 | 0.231 | 0.211 | 402 | 43.3% | +0.42 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 565 | 403 | 20 | market_no_vig | 0.253 | 0.249 | 0.250 | 0.250 | 367 | 49.0% | +0.71 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 556 | 393 | 17 | market_no_vig | 0.108 | 0.106 | 0.107 | 0.107 | 363 | 37.2% | +0.22 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 408 | 291 | 17 | market_no_vig | 0.093 | 0.092 | 0.096 | 0.093 | 272 | 39.7% | +0.25 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 263 | 241 | 18 | model_only | 0.248 | 0.251 | 0.252 | 0.250 | 175 | 50.9% | +0.67 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 224 | 209 | 15 | market_no_vig | 0.252 | 0.245 | 0.250 | 0.248 | 176 | 27.8% | +0.01 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 216 | 192 | 17 | distribution | 0.252 | 0.261 | 0.245 | 0.252 | 180 | 46.1% | +0.17 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 210 | 186 | 17 | walk_forward_blend | 0.243 | 0.246 | 0.284 | 0.243 | 180 | 46.1% | +0.50 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 263 | 182 | 17 | model_only | 0.254 | 0.255 | 0.254 | 0.255 | 173 | 40.5% | +0.13 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 165 | 154 | 18 | model_only | 0.248 | 0.259 | 0.268 | 0.255 | 99 | 58.6% | +1.27 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 156 | 140 | 17 | market_no_vig | 0.223 | 0.215 | 0.231 | 0.220 | 129 | 54.3% | +0.33 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 147 | 138 | 17 | distribution | 0.249 | 0.251 | 0.248 | 0.250 | 126 | 52.4% | +0.88 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 148 | 137 | 16 | distribution | 0.228 | 0.222 | 0.217 | 0.226 | 131 | 32.1% | -0.38 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 146 | 137 | 16 | model_only | 0.248 | 0.249 | 0.250 | 0.249 | 126 | 44.4% | -0.81 |
| batter_home_runs|under|common|HR 0.5|missing_price|fanduel | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 143 | 120 | 17 | distribution | 0.208 | 0.207 | 0.206 | 0.208 | 68 | 55.9% | +0.95 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 129 | 120 | 15 | distribution | 0.248 | 0.248 | 0.247 | 0.248 | 115 | 37.4% | +0.51 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 125 | 117 | 14 | market_no_vig | 0.250 | 0.249 | 0.256 | 0.249 | 113 | 23.9% | -0.56 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 131 | 115 | 14 | model_only | 0.248 | 0.255 | 0.267 | 0.254 | 104 | 36.5% | +0.18 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 119 | 109 | 15 | market_no_vig | 0.247 | 0.243 | 0.263 | 0.247 | 73 | 32.9% | -1.25 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 120 | 108 | 18 | distribution | 0.247 | 0.245 | 0.235 | 0.251 | 91 | 48.4% | -0.23 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 109 | 104 | 14 | walk_forward_blend | 0.251 | 0.250 | 0.265 | 0.249 | 80 | 25.0% | -0.10 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 119 | 103 | 15 | market_no_vig | 0.233 | 0.220 | 0.230 | 0.224 | 100 | 47.0% | +0.83 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 111 | 101 | 15 | model_only | 0.235 | 0.252 | 0.269 | 0.240 | 99 | 44.4% | +0.09 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 103 | 94 | 14 | distribution | 0.252 | 0.242 | 0.238 | 0.246 | 86 | 45.3% | +0.29 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 107 | 76 | 16 | model_only | 0.207 | 0.207 | 0.236 | 0.208 | 67 | 46.3% | +0.45 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 80 | 76 | 15 | distribution | 0.251 | 0.253 | 0.249 | 0.256 | 57 | 33.3% | -1.20 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 83 | 74 | 18 | walk_forward_blend | 0.247 | 0.254 | 0.278 | 0.247 | 62 | 54.8% | -0.05 |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 80 | 66 | 17 | distribution | 0.258 | 0.252 | 0.246 | 0.248 | 48 | 33.3% | -0.02 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 71 | 64 | 15 | distribution | 0.251 | 0.240 | 0.232 | 0.247 | 43 | 69.8% | +1.97 |

## Opportunity Diagnostics

| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|batter_pa_under_3_8 | 14387 | 11628 | 3.32 | - | 0.183 | 0.182 | -22.0% | 10380 |
| batter_total_bases|batter_pa_3_8_to_4_2 | 13271 | 11580 | 4.02 | - | 0.178 | 0.178 | -15.0% | 10517 |
| batter_hits|batter_pa_3_8_to_4_2 | 13096 | 11531 | 4.02 | - | 0.182 | 0.182 | +4.9% | 10281 |
| batter_total_bases|batter_pa_under_3_8 | 12876 | 10262 | 3.33 | - | 0.147 | 0.146 | -15.1% | 9275 |
| batter_total_bases|batter_pa_4_3_plus | 11153 | 9948 | 4.49 | - | 0.193 | 0.194 | -10.7% | 9412 |
| batter_hits|batter_pa_4_3_plus | 9896 | 8873 | 4.48 | - | 0.177 | 0.177 | +11.9% | 8163 |
| batter_home_runs|batter_pa_under_3_8 | 5836 | 4617 | 3.32 | - | 0.050 | 0.049 | -14.4% | 4195 |
| batter_home_runs|batter_pa_3_8_to_4_2 | 5142 | 4461 | 4.02 | - | 0.068 | 0.067 | -10.0% | 4093 |
| batter_home_runs|batter_pa_4_3_plus | 3893 | 3464 | 4.49 | - | 0.080 | 0.078 | -42.7% | 3304 |
| pitcher_strikeouts|pitcher_bf_20_to_23 | 1702 | 1514 | - | 22.2 | 0.250 | 0.249 | +1.3% | 1270 |
| pitcher_strikeouts|pitcher_bf_24_plus | 1114 | 1052 | - | 25.2 | 0.252 | 0.248 | -5.6% | 830 |
| pitcher_strikeouts|pitcher_bf_under_20 | 189 | 181 | - | 18.6 | 0.231 | 0.218 | -31.4% | 133 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 11867 |
| stale_close_before_lock | 6183 |
| line_disappeared_at_close | 1309 |
| missing_lock_snapshot | 901 |
| fallback_other_book_only | 259 |
| no_valid_close_snapshot | 183 |

## Live Probability Policy

| Level | Bucket | Variant | Rows | Dates | Brier Gain | Model Weight |
|---|---|---|---:|---:|---:|---:|
| exact_bucket | batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | market_no_vig | 451 | 19 | +0.007 | 0.000 |
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | market_no_vig | 209 | 15 | +0.007 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | distribution | 192 | 17 | +0.007 | 0.800 |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | market_no_vig | 936 | 17 | +0.004 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | market_no_vig | 684 | 19 | +0.004 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|draftkings | market_no_vig | 403 | 20 | +0.004 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | walk_forward_blend | 474 | 18 | +0.003 | 0.000 |
| market_side | pitcher_strikeouts|under | market_no_vig | 1361 | 18 | +0.003 | 0.200 |
| line_surface | pitcher_strikeouts|under|common | market_no_vig | 1361 | 18 | +0.003 | 0.200 |
| market_side | pitcher_strikeouts|over | market_no_vig | 1386 | 18 | +0.003 | 0.100 |
| line_surface | pitcher_strikeouts|over|common | market_no_vig | 1386 | 18 | +0.003 | 0.100 |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | distribution | 2477 | 19 | +0.002 | 0.300 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | market_no_vig | 860 | 17 | +0.002 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | market_no_vig | 2680 | 20 | +0.001 | 0.000 |
| market_side | batter_total_bases|under | market_no_vig | 3690 | 20 | +0.001 | 0.300 |
| line_surface | batter_total_bases|under|common | market_no_vig | 3690 | 20 | +0.001 | 0.300 |
| market_side | batter_hits|under | walk_forward_blend | 6765 | 20 | +0.001 | 0.100 |
| line_surface | batter_hits|under|common | walk_forward_blend | 6765 | 20 | +0.001 | 0.100 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | walk_forward_blend | 746 | 17 | +0.001 | 0.500 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | market_no_vig | 393 | 17 | +0.001 | 0.000 |

## Reading This

- `model_only` is the locked player-prop probability.
- `market_no_vig` is the book market baseline after removing vig when both sides were available.
- `distribution` prices the exact line from the locked projected count with stat-specific curves; total bases uses a compound PA/single/double/triple/HR shape.
- `walk_forward_blend` picks a model/market weight from prior dates only.
- A bucket is not real-money ready merely because it appears here; it still needs enough graded rows, valid CLV, ROI, calibration, and concentration checks.
