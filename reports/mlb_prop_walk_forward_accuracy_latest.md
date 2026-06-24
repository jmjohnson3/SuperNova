# MLB Prop Walk-Forward Accuracy

- Generated UTC: 2026-06-24T15:02:37+00:00
- Source: bets.mlb_prop_prediction_replay
- Date range: 2026-05-31 to 2026-06-24
- Locked rows: 121482
- Graded rows: 107159
- Pending rows with lock context: 14323
- Unique dates: 25
- Valid CLV rows: 104823 (86.3%)
- Avg CLV price: +0.03
- Live blend policy buckets: 24 exact, 4 line-surface, 3 market-side

This audit is walk-forward: blend weights use only earlier game dates, and valid CLV requires same book/player/stat/side/line close snapshots after lock and before first pitch.

## Overall Probability Variants

| Variant | Rows | Brier | Cal err | EV picks | ROI | CLV beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|
| model_only | 107159 | 0.162 | -1.6% | 25160 | -14.3% | 39.2% | +0.15 |
| market_no_vig | 106963 | 0.162 | -2.2% | 4 | +112.0% | - | - |
| distribution | 107159 | 0.164 | +0.5% | 25089 | -13.1% | 38.3% | +0.09 |
| walk_forward_blend | 106136 | 0.161 | -2.0% | 16166 | -14.8% | 39.3% | +0.17 |

## Market And Side

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 44005 | 38577 | 25 | walk_forward_blend | 0.162 | 0.162 | 0.164 | 0.161 | 38520 | 36.9% | +0.09 |
| batter_hits|over | 37757 | 33361 | 25 | walk_forward_blend | 0.172 | 0.172 | 0.174 | 0.172 | 32710 | 37.4% | +0.08 |
| batter_home_runs|over | 19573 | 17070 | 24 | walk_forward_blend | 0.061 | 0.061 | 0.062 | 0.061 | 17141 | 33.4% | +0.04 |
| batter_hits|under | 10522 | 9210 | 25 | walk_forward_blend | 0.232 | 0.232 | 0.234 | 0.231 | 8705 | 32.5% | -0.23 |
| batter_total_bases|under | 5299 | 4831 | 25 | market_no_vig | 0.243 | 0.242 | 0.246 | 0.242 | 4392 | 31.2% | -0.31 |
| pitcher_strikeouts|over | 2102 | 2000 | 23 | market_no_vig | 0.247 | 0.245 | 0.253 | 0.247 | 1682 | 37.2% | -0.17 |
| pitcher_strikeouts|under | 2077 | 1975 | 23 | market_no_vig | 0.247 | 0.245 | 0.252 | 0.247 | 1673 | 43.3% | +0.22 |
| batter_home_runs|under | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Line Surface

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common | 29867 | 26078 | 25 | walk_forward_blend | 0.212 | 0.212 | 0.214 | 0.212 | 25764 | 38.9% | +0.09 |
| batter_total_bases|over|common | 24507 | 21573 | 25 | walk_forward_blend | 0.210 | 0.211 | 0.214 | 0.210 | 21428 | 38.9% | +0.11 |
| batter_total_bases|over|alt_tail | 19498 | 17004 | 22 | market_no_vig | 0.101 | 0.100 | 0.101 | 0.100 | 17092 | 34.3% | +0.06 |
| batter_hits|under|common | 10522 | 9210 | 25 | walk_forward_blend | 0.232 | 0.232 | 0.234 | 0.231 | 8705 | 32.5% | -0.23 |
| batter_home_runs|over|common | 9821 | 8565 | 24 | walk_forward_blend | 0.112 | 0.112 | 0.114 | 0.112 | 8592 | 34.7% | +0.06 |
| batter_home_runs|over|alt_tail | 9752 | 8505 | 22 | market_no_vig | 0.010 | 0.010 | 0.010 | 0.010 | 8549 | 32.0% | +0.02 |
| batter_hits|over|alt_tail | 7890 | 7283 | 21 | market_no_vig | 0.029 | 0.029 | 0.030 | 0.029 | 6946 | 31.8% | +0.02 |
| batter_total_bases|under|common | 5299 | 4831 | 25 | market_no_vig | 0.243 | 0.242 | 0.246 | 0.242 | 4392 | 31.2% | -0.31 |
| pitcher_strikeouts|over|common | 2102 | 2000 | 23 | market_no_vig | 0.247 | 0.245 | 0.253 | 0.247 | 1682 | 37.2% | -0.17 |
| pitcher_strikeouts|under|common | 2077 | 1975 | 23 | market_no_vig | 0.247 | 0.245 | 0.252 | 0.247 | 1673 | 43.3% | +0.22 |
| batter_home_runs|under|common | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |

## Exact Bucket

| Bucket | Rows | Graded | Dates | Best | Model Brier | Market Brier | Dist Brier | Blend Brier | CLV rows | CLV beat | Avg CLV |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 13480 | 11505 | 22 | market_no_vig | 0.075 | 0.073 | 0.074 | 0.074 | 11553 | 31.5% | +0.04 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 9752 | 8505 | 22 | market_no_vig | 0.010 | 0.010 | 0.010 | 0.010 | 8549 | 32.0% | +0.02 |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 7869 | 7262 | 21 | market_no_vig | 0.029 | 0.029 | 0.029 | 0.029 | 6925 | 31.7% | +0.03 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 6509 | 5522 | 24 | walk_forward_blend | 0.087 | 0.087 | 0.088 | 0.086 | 5503 | 34.3% | +0.04 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 5974 | 5384 | 23 | model_only | 0.222 | 0.224 | 0.223 | 0.222 | 5407 | 34.8% | -0.03 |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 5830 | 4996 | 22 | market_no_vig | 0.160 | 0.159 | 0.160 | 0.159 | 4994 | 39.1% | +0.07 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 5774 | 4968 | 22 | model_only | 0.156 | 0.156 | 0.157 | 0.156 | 4985 | 41.8% | +0.10 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 5369 | 4891 | 22 | market_no_vig | 0.151 | 0.150 | 0.151 | 0.150 | 4928 | 40.2% | +0.08 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 4686 | 4202 | 23 | distribution | 0.226 | 0.226 | 0.225 | 0.226 | 3909 | 38.5% | +0.03 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 4472 | 4025 | 24 | model_only | 0.237 | 0.241 | 0.240 | 0.238 | 4028 | 36.5% | +0.05 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 4433 | 3823 | 25 | market_no_vig | 0.243 | 0.242 | 0.245 | 0.242 | 3795 | 30.8% | -0.28 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 3920 | 3570 | 25 | market_no_vig | 0.243 | 0.242 | 0.244 | 0.242 | 3377 | 44.6% | +0.35 |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 3560 | 3213 | 24 | distribution | 0.226 | 0.225 | 0.224 | 0.224 | 2836 | 36.5% | +0.05 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 3591 | 2929 | 25 | model_only | 0.206 | 0.209 | 0.211 | 0.207 | 2951 | 38.9% | +0.11 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 3155 | 2903 | 24 | walk_forward_blend | 0.155 | 0.156 | 0.158 | 0.155 | 2936 | 34.9% | +0.07 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 3175 | 2892 | 22 | walk_forward_blend | 0.204 | 0.203 | 0.208 | 0.203 | 2913 | 35.3% | -0.05 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 2836 | 2629 | 22 | model_only | 0.204 | 0.205 | 0.209 | 0.204 | 2661 | 37.1% | +0.03 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 2445 | 2088 | 22 | walk_forward_blend | 0.250 | 0.250 | 0.254 | 0.250 | 2086 | 39.9% | +0.04 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 2203 | 2007 | 25 | model_only | 0.240 | 0.241 | 0.241 | 0.241 | 1893 | 30.9% | -0.31 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 2089 | 1840 | 23 | market_no_vig | 0.246 | 0.245 | 0.251 | 0.245 | 1795 | 41.2% | +0.19 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1375 | 1270 | 22 | market_no_vig | 0.212 | 0.208 | 0.218 | 0.209 | 1141 | 33.0% | -0.42 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 1259 | 1165 | 22 | market_no_vig | 0.209 | 0.208 | 0.215 | 0.208 | 1032 | 46.0% | +0.44 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 1276 | 1129 | 25 | model_only | 0.234 | 0.234 | 0.237 | 0.235 | 911 | 25.6% | -0.50 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 1088 | 1014 | 22 | walk_forward_blend | 0.252 | 0.252 | 0.265 | 0.251 | 1029 | 31.3% | -0.15 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 1197 | 961 | 23 | market_no_vig | 0.251 | 0.249 | 0.255 | 0.250 | 984 | 42.5% | +0.43 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 962 | 895 | 24 | market_no_vig | 0.250 | 0.247 | 0.257 | 0.249 | 868 | 34.3% | -0.28 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 913 | 724 | 23 | model_only | 0.247 | 0.252 | 0.250 | 0.248 | 731 | 35.4% | +0.03 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 921 | 705 | 25 | model_only | 0.246 | 0.248 | 0.250 | 0.248 | 728 | 25.1% | -0.68 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 676 | 642 | 23 | walk_forward_blend | 0.251 | 0.249 | 0.264 | 0.249 | 643 | 40.6% | +0.08 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 811 | 618 | 25 | market_no_vig | 0.251 | 0.249 | 0.251 | 0.249 | 629 | 48.0% | +0.75 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 627 | 595 | 24 | market_no_vig | 0.254 | 0.249 | 0.261 | 0.250 | 586 | 36.3% | -0.14 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 774 | 592 | 22 | market_no_vig | 0.117 | 0.116 | 0.117 | 0.116 | 610 | 33.9% | +0.14 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 608 | 573 | 22 | walk_forward_blend | 0.202 | 0.201 | 0.212 | 0.200 | 573 | 41.0% | +0.30 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 595 | 449 | 22 | market_no_vig | 0.097 | 0.096 | 0.100 | 0.097 | 468 | 36.3% | +0.16 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 350 | 330 | 23 | model_only | 0.252 | 0.254 | 0.257 | 0.253 | 251 | 49.4% | +0.64 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 387 | 284 | 22 | model_only | 0.250 | 0.254 | 0.254 | 0.252 | 307 | 39.4% | +0.15 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 295 | 279 | 20 | market_no_vig | 0.254 | 0.249 | 0.252 | 0.251 | 236 | 26.7% | -0.08 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 287 | 269 | 22 | distribution | 0.263 | 0.260 | 0.252 | 0.258 | 271 | 38.0% | -0.02 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 285 | 263 | 22 | walk_forward_blend | 0.240 | 0.239 | 0.264 | 0.238 | 272 | 41.2% | +0.31 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 228 | 216 | 23 | model_only | 0.241 | 0.249 | 0.264 | 0.246 | 165 | 53.9% | +0.90 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 216 | 199 | 22 | market_no_vig | 0.223 | 0.213 | 0.222 | 0.219 | 204 | 47.1% | +0.13 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 201 | 189 | 21 | distribution | 0.233 | 0.221 | 0.213 | 0.227 | 191 | 29.3% | -0.35 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 188 | 177 | 19 | model_only | 0.240 | 0.245 | 0.264 | 0.244 | 163 | 37.4% | +0.42 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 183 | 173 | 22 | distribution | 0.250 | 0.251 | 0.249 | 0.251 | 160 | 49.4% | +0.61 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 181 | 171 | 21 | model_only | 0.248 | 0.250 | 0.250 | 0.249 | 158 | 45.6% | -0.59 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 194 | 158 | 22 | market_no_vig | 0.221 | 0.220 | 0.224 | 0.221 | 110 | 49.1% | +0.72 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 167 | 157 | 20 | distribution | 0.248 | 0.248 | 0.246 | 0.248 | 148 | 36.5% | +0.38 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 159 | 152 | 23 | distribution | 0.248 | 0.247 | 0.245 | 0.251 | 128 | 49.2% | -0.10 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 160 | 150 | 20 | market_no_vig | 0.273 | 0.259 | 0.272 | 0.266 | 151 | 37.7% | +0.04 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 162 | 150 | 20 | market_no_vig | 0.255 | 0.253 | 0.264 | 0.255 | 107 | 29.9% | -1.26 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 158 | 148 | 19 | market_no_vig | 0.249 | 0.248 | 0.249 | 0.248 | 140 | 29.3% | -0.32 |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 154 | 137 | 20 | market_no_vig | 0.238 | 0.222 | 0.241 | 0.228 | 150 | 45.3% | +0.64 |
| batter_home_runs|under|common|HR 0.5|missing_price|fanduel | 147 | 135 | 1 | distribution | 0.134 | - | 0.132 | - | 0 | - | - |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 141 | 133 | 19 | distribution | 0.254 | 0.247 | 0.243 | 0.250 | 117 | 48.7% | +0.30 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 137 | 130 | 20 | walk_forward_blend | 0.253 | 0.252 | 0.264 | 0.251 | 103 | 26.2% | -0.12 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 126 | 121 | 20 | market_no_vig | 0.238 | 0.232 | 0.236 | 0.236 | 95 | 56.8% | +0.91 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 149 | 114 | 21 | market_no_vig | 0.205 | 0.200 | 0.223 | 0.204 | 113 | 45.1% | +0.34 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 108 | 106 | 18 | market_no_vig | 0.231 | 0.224 | 0.233 | 0.228 | 98 | 35.7% | +0.27 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 115 | 105 | 23 | market_no_vig | 0.281 | 0.252 | 0.271 | 0.267 | 103 | 47.6% | -0.09 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 111 | 105 | 20 | model_only | 0.248 | 0.249 | 0.257 | 0.251 | 86 | 39.5% | -0.71 |

## Opportunity Diagnostics

| Bucket | Rows | Graded | Avg PA | Avg BF | Model Brier | Market Brier | Model ROI | CLV rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|batter_pa_3_8_to_4_2 | 17679 | 15910 | 4.03 | - | 0.177 | 0.175 | -21.8% | 15481 |
| batter_hits|batter_pa_under_3_8 | 18572 | 15472 | 3.32 | - | 0.186 | 0.187 | -17.9% | 15290 |
| batter_hits|batter_pa_3_8_to_4_2 | 16965 | 15366 | 4.02 | - | 0.187 | 0.186 | -6.8% | 14783 |
| batter_total_bases|batter_pa_under_3_8 | 17042 | 14080 | 3.32 | - | 0.144 | 0.143 | -18.0% | 14035 |
| batter_total_bases|batter_pa_4_3_plus | 14583 | 13418 | 4.48 | - | 0.194 | 0.195 | -6.4% | 13396 |
| batter_hits|batter_pa_4_3_plus | 12742 | 11733 | 4.48 | - | 0.182 | 0.181 | +1.5% | 11342 |
| batter_home_runs|batter_pa_under_3_8 | 7736 | 6347 | 3.32 | - | 0.047 | 0.047 | -23.6% | 6356 |
| batter_home_runs|batter_pa_3_8_to_4_2 | 6865 | 6162 | 4.02 | - | 0.063 | 0.062 | -30.3% | 6061 |
| batter_home_runs|batter_pa_4_3_plus | 5119 | 4696 | 4.48 | - | 0.080 | 0.079 | +14.6% | 4724 |
| pitcher_strikeouts|pitcher_bf_20_to_23 | 2416 | 2284 | - | 22.3 | 0.247 | 0.246 | +1.3% | 1974 |
| pitcher_strikeouts|pitcher_bf_24_plus | 1494 | 1434 | - | 25.3 | 0.249 | 0.246 | -5.6% | 1170 |
| pitcher_strikeouts|pitcher_bf_under_20 | 269 | 257 | - | 18.5 | 0.237 | 0.227 | -31.4% | 211 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| stale_close_before_lock | 7233 |
| close_outside_two_hour_window | 6581 |
| line_disappeared_at_close | 1408 |
| missing_lock_snapshot | 901 |
| fallback_other_book_only | 294 |
| no_valid_close_snapshot | 242 |

## Live Probability Policy

| Level | Bucket | Variant | Rows | Dates | Brier Gain | Model Weight |
|---|---|---|---:|---:|---:|---:|
| exact_bucket | batter_hits|over|common|H 1.5|plus_100_149|fanduel | distribution | 189 | 21 | +0.019 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | market_no_vig | 150 | 20 | +0.015 | 0.200 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | distribution | 269 | 22 | +0.011 | 0.400 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_100_149|draftkings | market_no_vig | 199 | 22 | +0.010 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | market_no_vig | 595 | 24 | +0.006 | 0.000 |
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | market_no_vig | 279 | 20 | +0.005 | 0.000 |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | market_no_vig | 1270 | 22 | +0.003 | 0.000 |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | market_no_vig | 895 | 24 | +0.003 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|draftkings | market_no_vig | 618 | 25 | +0.002 | 0.000 |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | distribution | 152 | 23 | +0.002 | 0.300 |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | market_no_vig | 150 | 20 | +0.002 | 0.300 |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | distribution | 157 | 20 | +0.002 | 0.300 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | walk_forward_blend | 642 | 23 | +0.002 | 0.000 |
| market_side | pitcher_strikeouts|under | market_no_vig | 1975 | 23 | +0.002 | 0.200 |
| line_surface | pitcher_strikeouts|under|common | market_no_vig | 1975 | 23 | +0.002 | 0.200 |
| market_side | pitcher_strikeouts|over | market_no_vig | 2000 | 23 | +0.002 | 0.100 |
| line_surface | pitcher_strikeouts|over|common | market_no_vig | 2000 | 23 | +0.002 | 0.100 |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | market_no_vig | 4891 | 22 | +0.002 | 0.100 |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | walk_forward_blend | 573 | 22 | +0.002 | 0.300 |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | distribution | 3213 | 24 | +0.002 | 0.400 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | walk_forward_blend | 263 | 22 | +0.001 | 0.400 |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | distribution | 173 | 22 | +0.001 | 1.000 |
| line_surface | batter_total_bases|over|alt_tail | market_no_vig | 17004 | 22 | +0.001 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|draftkings | market_no_vig | 961 | 23 | +0.001 | 0.000 |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | market_no_vig | 1165 | 22 | +0.001 | 0.000 |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | market_no_vig | 592 | 22 | +0.001 | 0.000 |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | market_no_vig | 11505 | 22 | +0.001 | 0.000 |
| market_side | batter_hits|under | walk_forward_blend | 9210 | 25 | +0.001 | 0.200 |
| line_surface | batter_hits|under|common | walk_forward_blend | 9210 | 25 | +0.001 | 0.200 |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | market_no_vig | 158 | 22 | +0.001 | 0.000 |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|draftkings | market_no_vig | 1840 | 23 | +0.001 | 0.000 |

## Reading This

- `model_only` is the locked player-prop probability.
- `market_no_vig` is the book market baseline after removing vig when both sides were available.
- `distribution` prices the exact line from the locked projected count with stat-specific curves; total bases uses a compound PA/single/double/triple/HR shape.
- `walk_forward_blend` picks a model/market weight from prior dates only.
- A bucket is not real-money ready merely because it appears here; it still needs enough graded rows, valid CLV, ROI, calibration, and concentration checks.
