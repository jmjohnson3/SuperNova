# MLB Prop Walk-Forward Accuracy

- Generated UTC: 2026-09-01T11:26:28+00:00
- Source: features.mlb_prop_market_training_examples
- Date range: 2026-07-19 to 2026-07-30
- Locked rows: 72250
- Graded rows: 64646
- Pending rows with lock context: 7604
- Unique dates: 12
- Valid CLV rows: 60095 (83.2%)
- Avg CLV price: -0.00
- Live blend policy buckets: 29 exact, 9 line-surface, 6 market-side

This audit is walk-forward: blend weights use only earlier game dates, and valid CLV requires same book/player/stat/side/line close snapshots after lock and before first pitch.

## Overall Probability Variants

| Variant | Rows | Brier | Cal err | EV picks | ROI | CLV beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 64646 | 0.164 | -3.4% | 4530 | -34.8% | 35.8% | +0.02 |
| market_no_vig | 64646 | 0.151 | +1.2% | 5953 | +141.4% | 48.0% | +0.22 |
| distribution | 64646 | 0.169 | -3.5% | 26987 | -22.3% | 39.4% | +0.09 |
| walk_forward_blend | 59701 | 0.151 | +0.9% | 5809 | +127.8% | 47.9% | +0.23 |

## Market And Side

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 27797 | 24723 | 12 | walk_forward_blend | 0.153 | 0.135 | 0.160 | 0.135 | 23233 | 37.5% | +0.00 |
| batter_hits|over | 20135 | 17978 | 12 | walk_forward_blend | 0.199 | 0.183 | 0.202 | 0.183 | 16847 | 39.8% | +0.05 |
| batter_home_runs|over | 12418 | 10980 | 12 | walk_forward_blend | 0.054 | 0.047 | 0.053 | 0.046 | 10401 | 34.3% | +0.02 |
| batter_hits|under | 6138 | 5483 | 12 | market_no_vig | 0.235 | 0.235 | 0.241 | 0.235 | 4883 | 39.4% | -0.14 |
| batter_total_bases|under | 2992 | 2792 | 12 | market_no_vig | 0.244 | 0.242 | 0.265 | 0.243 | 2441 | 36.3% | -0.23 |
| pitcher_strikeouts|over | 1385 | 1345 | 12 | market_no_vig | 0.252 | 0.248 | 0.251 | 0.249 | 1145 | 45.8% | +0.32 |
| pitcher_strikeouts|under | 1385 | 1345 | 12 | market_no_vig | 0.251 | 0.248 | 0.251 | 0.250 | 1145 | 36.9% | -0.30 |

## Line Surface

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common | 18560 | 16465 | 12 | market_no_vig | 0.215 | 0.198 | 0.219 | 0.198 | 15297 | 40.6% | +0.06 |
| batter_total_bases|over|common | 15414 | 13774 | 12 | walk_forward_blend | 0.205 | 0.186 | 0.215 | 0.185 | 12855 | 38.8% | -0.00 |
| batter_total_bases|over|alt_tail | 12383 | 10949 | 12 | walk_forward_blend | 0.088 | 0.071 | 0.090 | 0.071 | 10378 | 35.8% | +0.00 |
| batter_home_runs|over|common | 6211 | 5491 | 12 | walk_forward_blend | 0.099 | 0.086 | 0.096 | 0.085 | 5207 | 35.6% | +0.03 |
| batter_home_runs|over|alt_tail | 6207 | 5489 | 12 | walk_forward_blend | 0.010 | 0.008 | 0.010 | 0.007 | 5194 | 33.0% | +0.01 |
| batter_hits|under|common | 6138 | 5483 | 12 | market_no_vig | 0.235 | 0.235 | 0.241 | 0.235 | 4883 | 39.4% | -0.14 |
| batter_total_bases|under|common | 2992 | 2792 | 12 | market_no_vig | 0.244 | 0.242 | 0.265 | 0.243 | 2441 | 36.3% | -0.23 |
| batter_hits|over|alt_tail | 1575 | 1513 | 12 | market_no_vig | 0.024 | 0.020 | 0.024 | 0.021 | 1550 | 31.5% | +0.01 |
| pitcher_strikeouts|over|common | 1385 | 1345 | 12 | market_no_vig | 0.252 | 0.248 | 0.251 | 0.249 | 1145 | 45.8% | +0.32 |
| pitcher_strikeouts|under|common | 1385 | 1345 | 12 | market_no_vig | 0.251 | 0.248 | 0.251 | 0.250 | 1145 | 36.9% | -0.30 |

## Exact Bucket

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 8590 | 7433 | 12 | walk_forward_blend | 0.064 | 0.054 | 0.067 | 0.054 | 7007 | 33.0% | -0.00 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 6207 | 5489 | 12 | walk_forward_blend | 0.010 | 0.008 | 0.010 | 0.007 | 5194 | 33.0% | +0.01 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 3805 | 3514 | 12 | market_no_vig | 0.229 | 0.205 | 0.233 | 0.206 | 3379 | 34.9% | -0.15 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 4059 | 3494 | 12 | walk_forward_blend | 0.072 | 0.065 | 0.073 | 0.065 | 3276 | 32.5% | +0.00 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 3769 | 3274 | 12 | walk_forward_blend | 0.157 | 0.131 | 0.159 | 0.130 | 3080 | 45.2% | +0.16 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 3691 | 3225 | 12 | walk_forward_blend | 0.148 | 0.119 | 0.148 | 0.117 | 3042 | 40.5% | +0.00 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 3368 | 3121 | 12 | market_no_vig | 0.132 | 0.101 | 0.132 | 0.103 | 2990 | 41.7% | +0.01 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 2992 | 2769 | 12 | market_no_vig | 0.237 | 0.227 | 0.257 | 0.227 | 2659 | 34.7% | -0.15 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 2899 | 2692 | 12 | market_no_vig | 0.229 | 0.229 | 0.234 | 0.230 | 2430 | 39.2% | -0.00 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 2577 | 2260 | 12 | model_only | 0.246 | 0.247 | 0.251 | 0.246 | 2106 | 38.2% | -0.19 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 2412 | 2240 | 12 | walk_forward_blend | 0.242 | 0.242 | 0.265 | 0.242 | 1987 | 43.8% | +0.20 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2099 | 1969 | 12 | market_no_vig | 0.227 | 0.226 | 0.231 | 0.228 | 1746 | 43.3% | +0.08 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 2041 | 1892 | 12 | walk_forward_blend | 0.144 | 0.121 | 0.135 | 0.119 | 1828 | 40.4% | +0.07 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 1982 | 1865 | 12 | walk_forward_blend | 0.201 | 0.173 | 0.207 | 0.173 | 1807 | 34.4% | -0.26 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2160 | 1754 | 12 | walk_forward_blend | 0.205 | 0.176 | 0.200 | 0.174 | 1631 | 42.0% | +0.12 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 1882 | 1746 | 12 | market_no_vig | 0.197 | 0.158 | 0.200 | 0.160 | 1700 | 35.9% | -0.11 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 1575 | 1513 | 12 | market_no_vig | 0.024 | 0.020 | 0.024 | 0.021 | 1550 | 31.5% | +0.01 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1548 | 1313 | 12 | market_no_vig | 0.260 | 0.235 | 0.255 | 0.236 | 1220 | 45.3% | +0.28 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1287 | 1212 | 12 | market_no_vig | 0.245 | 0.243 | 0.268 | 0.244 | 1146 | 37.8% | -0.15 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1238 | 1074 | 12 | market_no_vig | 0.251 | 0.251 | 0.256 | 0.251 | 997 | 42.3% | +0.15 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 836 | 757 | 12 | walk_forward_blend | 0.241 | 0.238 | 0.246 | 0.238 | 506 | 31.4% | -0.55 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 759 | 718 | 12 | walk_forward_blend | 0.213 | 0.212 | 0.227 | 0.211 | 536 | 36.4% | -0.24 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 718 | 680 | 12 | walk_forward_blend | 0.213 | 0.213 | 0.227 | 0.211 | 501 | 51.7% | +0.28 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 716 | 671 | 12 | market_no_vig | 0.249 | 0.241 | 0.286 | 0.241 | 651 | 27.5% | -0.35 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 615 | 526 | 12 | walk_forward_blend | 0.250 | 0.250 | 0.251 | 0.250 | 484 | 44.8% | +0.38 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 504 | 477 | 12 | market_no_vig | 0.244 | 0.244 | 0.283 | 0.244 | 466 | 41.0% | +0.03 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 566 | 440 | 12 | walk_forward_blend | 0.258 | 0.232 | 0.258 | 0.231 | 410 | 45.9% | +0.39 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 569 | 435 | 12 | market_no_vig | 0.250 | 0.249 | 0.253 | 0.249 | 414 | 33.6% | -0.61 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 513 | 386 | 12 | model_only | 0.249 | 0.249 | 0.256 | 0.249 | 367 | 45.8% | +0.60 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 416 | 386 | 12 | market_no_vig | 0.194 | 0.151 | 0.187 | 0.154 | 372 | 41.7% | +0.10 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 492 | 382 | 12 | market_no_vig | 0.104 | 0.087 | 0.104 | 0.092 | 331 | 31.4% | +0.04 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 369 | 348 | 12 | walk_forward_blend | 0.247 | 0.247 | 0.292 | 0.246 | 328 | 49.7% | +0.35 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 321 | 304 | 12 | walk_forward_blend | 0.251 | 0.248 | 0.280 | 0.247 | 287 | 32.4% | -0.40 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 394 | 292 | 12 | market_no_vig | 0.102 | 0.090 | 0.101 | 0.101 | 263 | 39.9% | +0.24 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 221 | 214 | 12 | model_only | 0.248 | 0.248 | 0.253 | 0.249 | 191 | 44.0% | +0.60 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 208 | 202 | 12 | distribution | 0.244 | 0.244 | 0.240 | 0.247 | 178 | 55.1% | +0.97 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 258 | 198 | 12 | market_no_vig | 0.255 | 0.233 | 0.254 | 0.236 | 181 | 44.8% | +0.25 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 159 | 155 | 12 | distribution | 0.235 | 0.234 | 0.223 | 0.236 | 83 | 54.2% | +0.74 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 141 | 138 | 12 | walk_forward_blend | 0.259 | 0.253 | 0.252 | 0.252 | 97 | 48.5% | +0.49 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 143 | 135 | 12 | walk_forward_blend | 0.241 | 0.185 | 0.234 | 0.178 | 131 | 46.6% | +0.16 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 142 | 130 | 11 | market_no_vig | 0.247 | 0.242 | 0.257 | 0.245 | 124 | 30.6% | -0.41 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 115 | 112 | 12 | distribution | 0.261 | 0.254 | 0.252 | 0.256 | 87 | 43.7% | +0.56 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 114 | 109 | 11 | distribution | 0.252 | 0.250 | 0.247 | 0.250 | 104 | 28.8% | -0.54 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 112 | 107 | 12 | market_no_vig | 0.257 | 0.251 | 0.269 | 0.254 | 104 | 44.2% | -0.51 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 111 | 105 | 12 | market_no_vig | 0.179 | 0.139 | 0.178 | 0.141 | 103 | 50.5% | +0.40 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 109 | 104 | 12 | market_no_vig | 0.256 | 0.252 | 0.270 | 0.253 | 102 | 49.0% | +0.50 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 106 | 102 | 12 | market_no_vig | 0.279 | 0.251 | 0.273 | 0.257 | 101 | 43.6% | +0.38 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 131 | 100 | 12 | market_no_vig | 0.203 | 0.169 | 0.205 | 0.173 | 80 | 36.2% | +0.12 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 102 | 98 | 11 | distribution | 0.260 | 0.260 | 0.250 | 0.261 | 99 | 42.4% | -0.73 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 99 | 95 | 12 | market_no_vig | 0.273 | 0.251 | 0.275 | 0.256 | 97 | 32.0% | -0.67 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 95 | 93 | 12 | market_no_vig | 0.248 | 0.247 | 0.260 | 0.250 | 76 | 25.0% | -0.75 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 93 | 91 | 12 | model_only | 0.224 | 0.226 | 0.229 | 0.232 | 67 | 22.4% | -1.45 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 70 | 70 | 11 | market_no_vig | 0.241 | 0.241 | 0.246 | 0.242 | 48 | 37.5% | -0.33 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 70 | 70 | 10 | walk_forward_blend | 0.232 | 0.232 | 0.260 | 0.232 | 55 | 38.2% | -0.25 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 70 | 68 | 12 | walk_forward_blend | 0.267 | 0.256 | 0.266 | 0.253 | 37 | 35.1% | -0.51 |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 84 | 65 | 12 | model_only | 0.238 | 0.238 | 0.261 | 0.239 | 49 | 34.7% | -0.76 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 67 | 62 | 10 | model_only | 0.225 | 0.225 | 0.260 | 0.227 | 59 | 44.1% | +0.05 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 61 | 60 | 12 | distribution | 0.242 | 0.247 | 0.238 | 0.244 | 50 | 52.0% | -0.39 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 80 | 59 | 12 | model_only | 0.241 | 0.241 | 0.259 | 0.242 | 45 | 48.9% | +0.93 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 65 | 59 | 11 | distribution | 0.262 | 0.256 | 0.242 | 0.250 | 54 | 64.8% | +1.25 |

## Opportunity Diagnostics

| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|batter_pa_3_8_to_4_2 | 17980 | 16426 | 4.00 | - | 0.171 | 0.154 | -7.0% | 15500 |
| batter_hits|batter_pa_3_8_to_4_2 | 14813 | 13562 | 3.99 | - | 0.205 | 0.193 | +2.4% | 12696 |
| batter_total_bases|batter_pa_under_3_8 | 9022 | 7572 | 3.34 | - | 0.134 | 0.118 | -22.8% | 6885 |
| batter_hits|batter_pa_under_3_8 | 8591 | 7242 | 3.33 | - | 0.210 | 0.200 | +6.9% | 6621 |
| batter_home_runs|batter_pa_3_8_to_4_2 | 6921 | 6279 | 3.99 | - | 0.059 | 0.051 | -39.7% | 6045 |
| batter_total_bases|batter_pa_4_3_plus | 3787 | 3517 | 4.48 | - | 0.182 | 0.165 | +42.4% | 3289 |
| batter_home_runs|batter_pa_under_3_8 | 4103 | 3411 | 3.33 | - | 0.041 | 0.037 | -36.3% | 3134 |
| batter_hits|batter_pa_4_3_plus | 2869 | 2657 | 4.48 | - | 0.209 | 0.196 | -38.6% | 2413 |
| pitcher_strikeouts|pitcher_bf_20_to_23 | 1564 | 1516 | - | 22.1 | 0.251 | 0.246 | -21.7% | 1280 |
| batter_home_runs|batter_pa_4_3_plus | 1394 | 1290 | 4.48 | - | 0.062 | 0.052 | -41.2% | 1222 |
| pitcher_strikeouts|pitcher_bf_24_plus | 930 | 898 | - | 25.1 | 0.256 | 0.253 | +5.1% | 778 |
| pitcher_strikeouts|pitcher_bf_under_20 | 272 | 272 | - | 18.3 | 0.243 | 0.243 | -2.4% | 228 |
| pitcher_strikeouts|pitcher_bf_unknown | 4 | 4 | - | - | 0.234 | 0.202 | - | 4 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| player_prop_unavailable_at_close | 7902 |
| close_outside_two_hour_window | 2288 |
| line_disappeared_at_close | 996 |
| stale_close_before_lock | 432 |
| fallback_other_book_only | 273 |
| player_market_unavailable_at_close | 260 |
| no_valid_close_snapshot | 4 |

## Live Probability Policy

| Level | Bucket | Variant | Rows | Dates | Brier Gain | Model Weight |
|---|---|---|---:|---:|---:|---:|
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | market_no_vig | 386 | 12 | +0.043 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | market_no_vig | 1746 | 12 | +0.038 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | walk_forward_blend | 1754 | 12 | +0.031 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | walk_forward_blend | 3225 | 12 | +0.031 | 0.000 |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | market_no_vig | 3121 | 12 | +0.031 | 0.000 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|fanduel | walk_forward_blend | 1865 | 12 | +0.028 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|fanduel | walk_forward_blend | 440 | 12 | +0.027 | 0.000 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_250_499|fanduel | walk_forward_blend | 3274 | 12 | +0.027 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|fanduel | market_no_vig | 1313 | 12 | +0.025 | 0.000 |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | walk_forward_blend | 1892 | 12 | +0.025 | 0.200 |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|fanduel | market_no_vig | 3514 | 12 | +0.024 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|fanduel | market_no_vig | 198 | 12 | +0.022 | 0.000 |
| line_surface | batter_total_bases|over|common | walk_forward_blend | 13774 | 12 | +0.020 | 0.000 |
| market_side | batter_total_bases|over | walk_forward_blend | 24723 | 12 | +0.018 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | market_no_vig | 382 | 12 | +0.018 | 0.000 |
| line_surface | batter_total_bases|over|alt_tail | walk_forward_blend | 10949 | 12 | +0.017 | 0.000 |
| line_surface | batter_hits|over|common | market_no_vig | 16465 | 12 | +0.017 | 0.000 |
| market_side | batter_hits|over | walk_forward_blend | 17978 | 12 | +0.016 | 0.000 |
| line_surface | batter_home_runs|over|common | walk_forward_blend | 5491 | 12 | +0.014 | 0.100 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_500_plus|fanduel | market_no_vig | 292 | 12 | +0.012 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | distribution | 155 | 12 | +0.012 | 0.000 |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | walk_forward_blend | 7433 | 12 | +0.010 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | market_no_vig | 2769 | 12 | +0.010 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | market_no_vig | 671 | 12 | +0.009 | 0.000 |
| market_side | batter_home_runs|over | walk_forward_blend | 10980 | 12 | +0.008 | 0.100 |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | walk_forward_blend | 3494 | 12 | +0.007 | 0.100 |
| market_side | pitcher_strikeouts|over | market_no_vig | 1345 | 12 | +0.004 | 0.000 |
| line_surface | pitcher_strikeouts|over|common | market_no_vig | 1345 | 12 | +0.004 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | walk_forward_blend | 304 | 12 | +0.004 | 0.000 |
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | distribution | 202 | 12 | +0.004 | 1.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | walk_forward_blend | 757 | 12 | +0.003 | 0.000 |
| line_surface | batter_hits|over|alt_tail | market_no_vig | 1513 | 12 | +0.003 | 0.000 |
| exact_bucket | batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | market_no_vig | 1513 | 12 | +0.003 | 0.000 |
| market_side | pitcher_strikeouts|under | market_no_vig | 1345 | 12 | +0.003 | 0.000 |
| line_surface | pitcher_strikeouts|under|common | market_no_vig | 1345 | 12 | +0.003 | 0.000 |
| line_surface | batter_home_runs|over|alt_tail | walk_forward_blend | 5489 | 12 | +0.003 | 0.000 |
| exact_bucket | batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | walk_forward_blend | 5489 | 12 | +0.003 | 0.000 |
| market_side | batter_total_bases|under | market_no_vig | 2792 | 12 | +0.002 | 0.000 |
| line_surface | batter_total_bases|under|common | market_no_vig | 2792 | 12 | +0.002 | 0.000 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | walk_forward_blend | 680 | 12 | +0.002 | 1.000 |

## Reading This

- `model_only` is the locked player-prop probability.
- `market_no_vig` is the book market baseline after removing vig when both sides were available.
- `distribution` prices the exact line from the locked projected count with stat-specific curves; total bases uses a compound PA/single/double/triple/HR shape.
- `walk_forward_blend` picks a model/market weight from prior dates only.
- A bucket is not real-money ready merely because it appears here; it still needs enough graded rows, valid CLV, ROI, calibration, and concentration checks.
