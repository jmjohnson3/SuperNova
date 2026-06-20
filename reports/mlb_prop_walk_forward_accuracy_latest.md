# MLB Prop Walk-Forward Accuracy

- Generated UTC: 2026-06-20T00:52:58+00:00
- Source: bets.mlb_prop_prediction_replay
- Date range: 2026-05-31 to 2026-06-19
- Locked rows: 95540
- Graded rows: 79111
- Pending rows with lock context: 16429
- Unique dates: 20
- Valid CLV rows: 78474 (82.1%)
- Avg CLV price: +0.05
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
| batter_total_bases|over | 34156 | 28100 | 20 | walk_forward_blend | 0.164 | 0.164 | 0.167 | 0.163 | 28598 | 38.1% | +0.12 |
| batter_hits|over | 30422 | 25267 | 20 | walk_forward_blend | 0.167 | 0.167 | 0.168 | 0.166 | 24982 | 38.7% | +0.10 |
| batter_home_runs|over | 15168 | 12407 | 19 | walk_forward_blend | 0.064 | 0.064 | 0.064 | 0.063 | 12698 | 34.2% | +0.05 |
| batter_hits|under | 8258 | 6765 | 20 | walk_forward_blend | 0.233 | 0.232 | 0.234 | 0.232 | 6409 | 33.1% | -0.25 |
| batter_total_bases|under | 4270 | 3690 | 20 | market_no_vig | 0.241 | 0.240 | 0.243 | 0.240 | 3320 | 30.7% | -0.36 |
| pitcher_strikeouts|over | 1572 | 1386 | 18 | market_no_vig | 0.249 | 0.247 | 0.254 | 0.250 | 1238 | 36.2% | -0.29 |
| pitcher_strikeouts|under | 1547 | 1361 | 18 | market_no_vig | 0.249 | 0.247 | 0.253 | 0.249 | 1229 | 44.9% | +0.35 |
| batter_home_runs|under | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Line Surface

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common | 23193 | 18965 | 20 | walk_forward_blend | 0.212 | 0.212 | 0.214 | 0.212 | 19028 | 40.3% | +0.13 |
| batter_total_bases|over|common | 19068 | 15764 | 20 | walk_forward_blend | 0.211 | 0.212 | 0.215 | 0.210 | 15916 | 40.3% | +0.15 |
| batter_total_bases|over|alt_tail | 15088 | 12336 | 17 | market_no_vig | 0.103 | 0.103 | 0.104 | 0.103 | 12682 | 35.4% | +0.08 |
| batter_hits|under|common | 8258 | 6765 | 20 | walk_forward_blend | 0.233 | 0.232 | 0.234 | 0.232 | 6409 | 33.1% | -0.25 |
| batter_hits|over|alt_tail | 7229 | 6302 | 17 | model_only | 0.030 | 0.030 | 0.030 | 0.030 | 5954 | 33.6% | +0.03 |
| batter_home_runs|over|common | 7616 | 6231 | 19 | market_no_vig | 0.117 | 0.117 | 0.118 | 0.117 | 6372 | 35.7% | +0.08 |
| batter_home_runs|over|alt_tail | 7552 | 6176 | 17 | market_no_vig | 0.010 | 0.010 | 0.010 | 0.010 | 6326 | 32.8% | +0.02 |
| batter_total_bases|under|common | 4270 | 3690 | 20 | market_no_vig | 0.241 | 0.240 | 0.243 | 0.240 | 3320 | 30.7% | -0.36 |
| pitcher_strikeouts|over|common | 1572 | 1386 | 18 | market_no_vig | 0.249 | 0.247 | 0.254 | 0.250 | 1238 | 36.2% | -0.29 |
| pitcher_strikeouts|under|common | 1547 | 1361 | 18 | market_no_vig | 0.249 | 0.247 | 0.253 | 0.249 | 1229 | 44.9% | +0.35 |
| batter_home_runs|under|common | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Exact Bucket

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 10394 | 8304 | 17 | market_no_vig | 0.076 | 0.076 | 0.076 | 0.076 | 8450 | 32.2% | +0.05 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 7209 | 6283 | 17 | model_only | 0.029 | 0.029 | 0.030 | 0.029 | 5934 | 33.5% | +0.03 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 7552 | 6176 | 17 | market_no_vig | 0.010 | 0.010 | 0.010 | 0.010 | 6326 | 32.8% | +0.02 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 5044 | 4014 | 19 | market_no_vig | 0.090 | 0.090 | 0.091 | 0.090 | 4032 | 35.2% | +0.05 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 4670 | 3986 | 18 | model_only | 0.223 | 0.225 | 0.224 | 0.224 | 4079 | 36.8% | +0.03 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 4517 | 3633 | 17 | market_no_vig | 0.164 | 0.164 | 0.167 | 0.164 | 3667 | 39.2% | +0.07 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 4187 | 3586 | 17 | walk_forward_blend | 0.152 | 0.152 | 0.154 | 0.152 | 3760 | 41.9% | +0.11 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 4450 | 3565 | 17 | model_only | 0.155 | 0.156 | 0.155 | 0.155 | 3632 | 43.0% | +0.15 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3707 | 3155 | 18 | market_no_vig | 0.228 | 0.228 | 0.228 | 0.228 | 2911 | 39.5% | +0.03 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 3498 | 2984 | 19 | model_only | 0.237 | 0.241 | 0.239 | 0.238 | 3056 | 38.2% | +0.10 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 3445 | 2747 | 20 | walk_forward_blend | 0.241 | 0.241 | 0.242 | 0.241 | 2777 | 31.5% | -0.33 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 3113 | 2680 | 20 | market_no_vig | 0.241 | 0.239 | 0.242 | 0.240 | 2537 | 46.6% | +0.40 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2880 | 2477 | 19 | distribution | 0.229 | 0.228 | 0.227 | 0.227 | 2143 | 36.9% | +0.05 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 2507 | 2171 | 17 | model_only | 0.200 | 0.201 | 0.206 | 0.200 | 2249 | 36.5% | +0.01 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 2447 | 2111 | 19 | market_no_vig | 0.164 | 0.163 | 0.165 | 0.163 | 2231 | 36.1% | +0.09 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2740 | 2069 | 20 | model_only | 0.208 | 0.211 | 0.212 | 0.209 | 2088 | 40.2% | +0.15 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 2205 | 1935 | 17 | model_only | 0.202 | 0.204 | 0.210 | 0.203 | 2033 | 39.5% | +0.12 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1765 | 1540 | 20 | model_only | 0.237 | 0.238 | 0.240 | 0.238 | 1456 | 30.9% | -0.32 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1887 | 1479 | 17 | model_only | 0.247 | 0.248 | 0.249 | 0.247 | 1516 | 40.4% | +0.07 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1626 | 1303 | 18 | walk_forward_blend | 0.244 | 0.244 | 0.247 | 0.243 | 1312 | 42.8% | +0.20 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1093 | 936 | 17 | market_no_vig | 0.216 | 0.212 | 0.219 | 0.213 | 856 | 32.5% | -0.49 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 1008 | 860 | 17 | market_no_vig | 0.213 | 0.211 | 0.216 | 0.211 | 783 | 46.9% | +0.48 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 1020 | 850 | 20 | model_only | 0.231 | 0.231 | 0.232 | 0.232 | 625 | 22.4% | -0.64 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 848 | 746 | 17 | walk_forward_blend | 0.252 | 0.252 | 0.266 | 0.251 | 786 | 32.8% | -0.06 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 935 | 702 | 18 | walk_forward_blend | 0.249 | 0.248 | 0.252 | 0.248 | 713 | 46.1% | +0.58 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 777 | 684 | 19 | market_no_vig | 0.251 | 0.247 | 0.252 | 0.249 | 683 | 32.9% | -0.35 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 684 | 506 | 18 | model_only | 0.249 | 0.255 | 0.251 | 0.250 | 520 | 38.8% | +0.06 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 537 | 474 | 18 | walk_forward_blend | 0.252 | 0.249 | 0.261 | 0.249 | 499 | 40.1% | +0.05 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 667 | 470 | 20 | market_no_vig | 0.249 | 0.248 | 0.251 | 0.248 | 488 | 25.6% | -0.68 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 509 | 451 | 19 | market_no_vig | 0.256 | 0.248 | 0.259 | 0.251 | 464 | 37.9% | -0.14 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 470 | 415 | 17 | model_only | 0.209 | 0.213 | 0.231 | 0.211 | 441 | 41.5% | +0.36 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 586 | 403 | 20 | market_no_vig | 0.253 | 0.249 | 0.250 | 0.250 | 426 | 48.6% | +0.68 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 582 | 393 | 17 | market_no_vig | 0.108 | 0.106 | 0.107 | 0.107 | 422 | 35.5% | +0.19 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 431 | 291 | 17 | market_no_vig | 0.093 | 0.092 | 0.096 | 0.093 | 316 | 38.9% | +0.24 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 271 | 241 | 18 | model_only | 0.248 | 0.251 | 0.252 | 0.250 | 186 | 50.0% | +0.61 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 227 | 209 | 15 | market_no_vig | 0.252 | 0.245 | 0.250 | 0.248 | 182 | 27.5% | -0.01 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 227 | 192 | 17 | distribution | 0.252 | 0.261 | 0.245 | 0.252 | 207 | 41.5% | +0.05 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 217 | 186 | 17 | walk_forward_blend | 0.243 | 0.246 | 0.284 | 0.243 | 198 | 43.4% | +0.38 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 277 | 182 | 17 | model_only | 0.254 | 0.255 | 0.254 | 0.255 | 201 | 42.3% | +0.19 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 172 | 154 | 18 | model_only | 0.248 | 0.259 | 0.268 | 0.255 | 114 | 57.9% | +1.26 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 160 | 140 | 17 | market_no_vig | 0.223 | 0.215 | 0.231 | 0.220 | 138 | 52.9% | +0.30 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 150 | 138 | 17 | distribution | 0.249 | 0.251 | 0.248 | 0.250 | 131 | 51.1% | +0.77 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 152 | 137 | 16 | distribution | 0.228 | 0.222 | 0.217 | 0.226 | 140 | 30.7% | -0.38 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 149 | 137 | 16 | model_only | 0.248 | 0.249 | 0.250 | 0.249 | 131 | 45.8% | -0.71 |
| batter_home_runs|under|common|HR 0.5|missing_price|fanduel | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 148 | 120 | 17 | distribution | 0.208 | 0.207 | 0.206 | 0.208 | 69 | 55.1% | +0.94 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 132 | 120 | 15 | distribution | 0.248 | 0.248 | 0.247 | 0.248 | 119 | 37.0% | +0.48 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 129 | 117 | 14 | market_no_vig | 0.250 | 0.249 | 0.256 | 0.249 | 118 | 25.4% | -0.52 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 139 | 115 | 14 | model_only | 0.248 | 0.255 | 0.267 | 0.254 | 126 | 37.3% | +0.30 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 123 | 109 | 15 | market_no_vig | 0.247 | 0.243 | 0.263 | 0.247 | 78 | 30.8% | -1.32 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 124 | 108 | 18 | distribution | 0.247 | 0.245 | 0.235 | 0.251 | 97 | 50.5% | -0.05 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 110 | 104 | 15 | walk_forward_blend | 0.251 | 0.250 | 0.265 | 0.249 | 81 | 24.7% | -0.11 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 122 | 103 | 15 | market_no_vig | 0.233 | 0.220 | 0.230 | 0.224 | 106 | 44.3% | +0.78 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 113 | 101 | 15 | model_only | 0.235 | 0.252 | 0.269 | 0.240 | 106 | 43.4% | +0.09 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 105 | 94 | 14 | distribution | 0.252 | 0.242 | 0.238 | 0.246 | 91 | 47.3% | +0.34 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 117 | 76 | 16 | model_only | 0.207 | 0.207 | 0.236 | 0.208 | 84 | 45.2% | +0.38 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 85 | 76 | 15 | distribution | 0.251 | 0.253 | 0.249 | 0.256 | 65 | 36.9% | -1.02 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 85 | 74 | 18 | walk_forward_blend | 0.247 | 0.254 | 0.278 | 0.247 | 66 | 51.5% | -0.16 |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 83 | 66 | 17 | distribution | 0.258 | 0.252 | 0.246 | 0.248 | 54 | 37.0% | +0.16 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 76 | 64 | 15 | distribution | 0.251 | 0.240 | 0.232 | 0.247 | 52 | 69.2% | +1.69 |

## Opportunity Diagnostics

| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|batter_pa_under_3_8 | 14894 | 11628 | 3.32 | - | 0.183 | 0.182 | -22.0% | 11359 |
| batter_total_bases|batter_pa_3_8_to_4_2 | 13661 | 11580 | 4.03 | - | 0.178 | 0.178 | -15.0% | 11423 |
| batter_hits|batter_pa_3_8_to_4_2 | 13527 | 11531 | 4.02 | - | 0.182 | 0.182 | +4.9% | 11142 |
| batter_total_bases|batter_pa_under_3_8 | 13288 | 10262 | 3.33 | - | 0.147 | 0.146 | -15.1% | 10193 |
| batter_total_bases|batter_pa_4_3_plus | 11477 | 9948 | 4.49 | - | 0.193 | 0.194 | -10.7% | 10302 |
| batter_hits|batter_pa_4_3_plus | 10259 | 8873 | 4.48 | - | 0.177 | 0.177 | +11.9% | 8890 |
| batter_home_runs|batter_pa_under_3_8 | 6020 | 4617 | 3.32 | - | 0.050 | 0.049 | -14.4% | 4617 |
| batter_home_runs|batter_pa_3_8_to_4_2 | 5290 | 4461 | 4.02 | - | 0.068 | 0.067 | -10.0% | 4459 |
| batter_home_runs|batter_pa_4_3_plus | 4005 | 3464 | 4.49 | - | 0.080 | 0.078 | -42.7% | 3622 |
| pitcher_strikeouts|pitcher_bf_20_to_23 | 1786 | 1514 | - | 22.3 | 0.250 | 0.249 | +1.3% | 1432 |
| pitcher_strikeouts|pitcher_bf_24_plus | 1140 | 1052 | - | 25.2 | 0.252 | 0.248 | -5.6% | 890 |
| pitcher_strikeouts|pitcher_bf_under_20 | 193 | 181 | - | 18.6 | 0.231 | 0.218 | -31.4% | 145 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 7927 |
| stale_close_before_lock | 6303 |
| line_disappeared_at_close | 1476 |
| missing_lock_snapshot | 901 |
| fallback_other_book_only | 285 |
| no_valid_close_snapshot | 174 |

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
