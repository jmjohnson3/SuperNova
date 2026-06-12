# MLB Prop Market Residual Models

Generated UTC: 2026-06-12T08:56:06Z
Rows: 38710
Date range: 2026-05-31 to 2026-06-10
Status: ready

## Holdout Variants

| Variant | Rows | Brier | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|
| model_only | 7624 | 0.154 | -1.5% | 1532 | 15.3% | 45.4% |
| market_no_vig | 7624 | 0.151 | -0.3% | 448 | 396.5% | 42.6% |
| market_residual | 7624 | 0.149 | 1.0% | 1498 | 29.6% | 42.4% |

## CLV Target

| Rows | Beat Rate | Avg Prob | Brier | Cal Err | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 7162 | 40.4% | 38.9% | 0.245 | 1.5% | 0.539 | ready |

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | ROI | Best Sel ROI | Best Sel CLV | Model Brier | Market Brier | Residual Brier | Residual Gain | Residual Sel | Residual ROI | Residual CLV | Residual Avg CLV | Residual Cal | Proof Blockers |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 307 | use_market_baseline | market_no_vig | 0.1% | 100.0% | 50.0% | 0.179 | 0.176 | 0.179 | 0.000 | 50 | 32.6% | 42.6% | 0.34 | 3.7% | residual_brier_gain; residual_clv_beat |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 181 | keep_model_only | model_only | 3.6% | 9.0% | 50.9% | 0.237 | 0.238 | 0.250 | -0.014 | 63 | -20.1% | 50.0% | 0.13 | 2.4% | residual_brier_gain; residual_selected_roi; residual_clv_beat |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 61 | keep_model_only | model_only | 7.0% | - | - | 0.229 | 0.240 | 0.247 | -0.018 | 10 | -14.1% | 70.0% | 2.21 | 8.0% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 34 | keep_model_only | model_only | 11.2% | 71.4% | 100.0% | 0.225 | 0.234 | 0.228 | -0.003 | 15 | 25.2% | 60.0% | 0.93 | 12.6% | residual_selected_sample; residual_brier_gain; residual_calibration |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 998 | no_bet_bad_clv | market_residual | 35.4% | 223.5% | 40.3% | 0.025 | 0.025 | 0.024 | 0.001 | 81 | 223.5% | 40.3% | 0.13 | -1.0% | residual_brier_gain; residual_clv_beat |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 727 | no_bet_negative_roi | market_residual | -9.1% | 132.1% | 39.0% | 0.089 | 0.088 | 0.087 | 0.002 | 42 | 132.1% | 39.0% | 0.16 | 1.9% | residual_brier_gain; residual_clv_beat |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 562 | no_bet_negative_roi | market_residual | -35.4% | -100.0% | 66.7% | 0.006 | 0.006 | 0.005 | 0.000 | 6 | -100.0% | 66.7% | 0.11 | -0.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 380 | no_bet_negative_roi | market_residual | -2.0% | 19.4% | 45.5% | 0.214 | 0.208 | 0.203 | 0.010 | 160 | 19.4% | 45.5% | 0.39 | 10.4% | residual_clv_beat; residual_calibration |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 346 | no_bet_no_edge | market_no_vig | 0.3% | - | - | 0.109 | 0.108 | 0.109 | -0.001 | 31 | -39.4% | 46.2% | 0.26 | 2.8% | residual_brier_gain; residual_selected_roi; residual_clv_beat |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 344 | no_bet_negative_roi | market_residual | -18.2% | 108.0% | 63.4% | 0.147 | 0.141 | 0.130 | 0.016 | 50 | 108.0% | 63.4% | 1.11 | -0.5% |  |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 300 | no_bet_negative_roi | market_residual | -19.2% | 48.3% | 39.0% | 0.154 | 0.145 | 0.128 | 0.026 | 77 | 48.3% | 39.0% | 0.27 | -1.1% | residual_clv_beat |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 266 | no_bet_negative_roi | market_no_vig | -26.0% | 118.2% | 54.5% | 0.230 | 0.224 | 0.225 | 0.006 | 130 | -17.8% | 45.2% | 0.60 | -10.3% | residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 257 | no_bet_negative_roi | model_only | -1.9% | - | - | 0.226 | 0.227 | 0.236 | -0.009 | 90 | -5.8% | 30.2% | -0.17 | 0.0% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 242 | no_bet_negative_roi | model_only | -33.9% | -12.3% | 35.5% | 0.217 | 0.221 | 0.235 | -0.017 | 88 | -33.2% | 26.1% | -0.47 | -14.0% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 237 | no_bet_negative_roi | model_only | -24.2% | - | - | 0.230 | 0.233 | 0.237 | -0.007 | 6 | -100.0% | 33.3% | 0.33 | -0.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 225 | no_bet_negative_roi | market_residual | -29.9% | 38.8% | 41.7% | 0.189 | 0.175 | 0.153 | 0.036 | 73 | 38.8% | 41.7% | 0.07 | -6.4% | residual_clv_beat; residual_calibration |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 198 | no_bet_negative_roi | market_residual | -4.4% | 360.0% | 90.0% | 0.155 | 0.151 | 0.145 | 0.010 | 13 | 360.0% | 90.0% | 2.35 | 4.0% | residual_selected_sample |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 196 | no_bet_negative_roi | market_residual | -40.1% | 4.1% | 64.3% | 0.171 | 0.162 | 0.158 | 0.013 | 17 | 4.1% | 64.3% | 0.58 | -2.4% | residual_selected_sample |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 161 | no_bet_bad_clv | market_no_vig | 23.7% | 52.5% | 40.0% | 0.258 | 0.255 | 0.279 | -0.021 | 41 | 38.0% | 48.8% | 0.36 | 16.7% | residual_brier_gain; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 137 | no_bet_bad_clv | market_residual | 4.0% | 51.4% | 43.6% | 0.229 | 0.223 | 0.195 | 0.034 | 39 | 51.4% | 43.6% | -0.19 | 15.9% | residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 125 | no_bet_bad_clv | market_residual | 8.0% | 8.6% | 41.1% | 0.210 | 0.193 | 0.182 | 0.028 | 103 | 8.6% | 41.1% | -0.04 | 1.1% | residual_clv_beat; residual_avg_clv |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 121 | no_bet_bad_clv | market_residual | 12.8% | 9.2% | 15.0% | 0.219 | 0.224 | 0.218 | 0.001 | 63 | 9.2% | 15.0% | -0.91 | 7.5% | residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 115 | no_bet_negative_roi | market_residual | -29.8% | - | - | 0.199 | 0.195 | 0.186 | 0.013 | 0 | - | - | - | 1.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 108 | no_bet_residual_unproven | market_residual | 13.2% | 21.9% | 53.8% | 0.224 | 0.224 | 0.224 | 0.000 | 13 | 21.9% | 53.8% | 0.27 | 12.3% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 78 | no_bet_negative_roi | market_residual | -50.1% | -15.3% | 30.8% | 0.238 | 0.229 | 0.219 | 0.019 | 13 | -15.3% | 30.8% | 0.31 | -20.5% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 70 | no_bet_negative_roi | market_no_vig | -21.6% | - | - | 0.261 | 0.260 | 0.265 | -0.004 | 19 | 19.7% | 7.7% | -1.79 | -12.5% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 63 | no_bet_bad_clv | market_residual | 13.8% | 33.8% | 35.7% | 0.219 | 0.233 | 0.215 | 0.003 | 14 | 33.8% | 35.7% | -0.21 | 10.7% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 61 | no_bet_negative_roi | model_only | -6.8% | 87.0% | 0.0% | 0.246 | 0.249 | 0.253 | -0.007 | 3 | -40.2% | 33.3% | -0.45 | 7.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 57 | no_bet_bad_clv | model_only | 7.2% | 1.9% | 41.0% | 0.238 | 0.247 | 0.279 | -0.040 | 1 | 79.4% | 100.0% | 0.77 | 14.5% | residual_selected_sample; residual_brier_gain; residual_calibration |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 47 | no_bet_negative_roi | market_residual | -18.4% | -6.1% | 66.7% | 0.200 | 0.197 | 0.196 | 0.004 | 9 | -6.1% | 66.7% | 3.41 | 1.8% | residual_selected_sample; residual_selected_roi |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 31 | no_bet_negative_roi | market_residual | -15.3% | 32.2% | 21.4% | 0.276 | 0.245 | 0.226 | 0.050 | 14 | 32.2% | 21.4% | -0.65 | -9.3% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 31 | no_bet_negative_roi | market_residual | -10.3% | 17.0% | 16.7% | 0.249 | 0.236 | 0.199 | 0.050 | 9 | 17.0% | 16.7% | 0.00 | 3.6% | residual_selected_sample; residual_clv_beat |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 30 | no_bet_negative_roi | market_no_vig | -11.5% | - | - | 0.248 | 0.232 | 0.248 | -0.000 | 12 | 6.2% | 100.0% | 3.26 | 1.1% | residual_selected_sample; residual_brier_gain |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 28 | no_bet_negative_roi | market_residual | -7.9% | 80.4% | 100.0% | 0.269 | 0.247 | 0.240 | 0.029 | 2 | 80.4% | 100.0% | 4.58 | 3.1% | residual_selected_sample |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 26 | no_bet_bad_clv | market_no_vig | 26.9% | 40.0% | 20.0% | 0.158 | 0.157 | 0.167 | -0.009 | 5 | 40.0% | 20.0% | -0.51 | 6.7% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 25 | no_bet_negative_roi | market_residual | -90.0% | - | - | 0.219 | 0.176 | 0.084 | 0.134 | 0 | - | - | - | -19.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 24 | no_bet_negative_roi | market_residual | -19.7% | - | - | 0.245 | 0.239 | 0.232 | 0.013 | 0 | - | - | - | -0.7% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 24 | no_bet_negative_roi | market_residual | -19.4% | - | - | 0.244 | 0.241 | 0.238 | 0.006 | 0 | - | - | - | 1.9% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 23 | no_bet_negative_roi | market_residual | -73.3% | 2.5% | 0.0% | 0.231 | 0.176 | 0.133 | 0.097 | 6 | 2.5% | 0.0% | -1.90 | -24.6% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 19 | no_bet_sample | model_only | -26.6% | - | - | 0.259 | 0.263 | 0.264 | -0.004 | 0 | - | - | - | -8.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 18 | no_bet_sample | market_residual | 1.4% | - | - | 0.213 | 0.205 | 0.185 | 0.028 | 0 | - | - | - | 13.5% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 18 | no_bet_sample | market_no_vig | 4.6% | - | - | 0.261 | 0.257 | 0.291 | -0.030 | 0 | - | - | - | 14.0% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 17 | no_bet_sample | market_residual | -34.2% | 15.2% | 62.5% | 0.248 | 0.243 | 0.222 | 0.026 | 8 | 15.2% | 62.5% | 1.57 | -17.2% | residual_selected_sample; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 16 | no_bet_sample | model_only | 1.8% | - | - | 0.240 | 0.245 | 0.262 | -0.022 | 0 | - | - | - | 13.4% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 15 | no_bet_sample | model_only | -7.7% | - | - | 0.247 | 0.253 | 0.297 | -0.049 | 15 | -7.7% | 46.2% | -0.06 | -16.6% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 14 | no_bet_sample | market_residual | -7.1% | - | - | 0.126 | 0.120 | 0.111 | 0.015 | 0 | - | - | - | 6.4% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 14 | no_bet_sample | model_only | 20.8% | - | - | 0.247 | 0.247 | 0.276 | -0.028 | 0 | - | - | - | 22.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 14 | no_bet_sample | market_residual | 22.2% | 18.5% | 54.5% | 0.244 | 0.227 | 0.206 | 0.038 | 13 | 18.5% | 54.5% | 0.94 | 5.9% | residual_selected_sample; residual_clv_beat; residual_calibration |
| batter_hits|under|common|H 1.5|lay_150_180|draftkings | 13 | no_bet_sample | market_residual | 61.8% | 60.6% | 75.0% | 0.231 | 0.178 | 0.147 | 0.085 | 8 | 60.6% | 75.0% | 2.98 | 37.9% | residual_selected_sample; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 13 | no_bet_sample | model_only | 14.7% | - | - | 0.248 | 0.248 | 0.262 | -0.015 | 0 | - | - | - | 15.2% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 12 | no_bet_sample | market_residual | -37.3% | 25.4% | 50.0% | 0.247 | 0.246 | 0.236 | 0.011 | 6 | 25.4% | 50.0% | 2.26 | -20.8% | residual_selected_sample; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 12 | no_bet_sample | market_no_vig | -10.5% | - | - | 0.252 | 0.243 | 0.271 | -0.020 | 5 | 25.2% | 100.0% | 3.01 | -2.2% | residual_selected_sample; residual_brier_gain |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 11 | no_bet_sample | market_residual | 17.1% | 87.1% | 0.0% | 0.263 | 0.250 | 0.235 | 0.028 | 2 | 87.1% | 0.0% | -0.53 | 20.0% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 9 | no_bet_sample | model_only | 39.2% | - | - | 0.273 | 0.279 | 0.293 | -0.019 | 0 | - | - | - | 20.2% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|draftkings | 9 | no_bet_sample | market_residual | 36.4% | 104.7% | 50.0% | 0.285 | 0.260 | 0.208 | 0.077 | 6 | 104.7% | 50.0% | -0.15 | 16.2% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 9 | no_bet_sample | market_residual | 28.0% | 9.9% | 40.0% | 0.240 | 0.216 | 0.208 | 0.032 | 6 | 9.9% | 40.0% | -0.96 | 9.0% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 9 | no_bet_sample | model_only | -27.9% | - | - | 0.243 | 0.266 | 0.333 | -0.090 | 9 | -27.9% | 12.5% | 0.07 | -28.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|under|common|K 6.5-8.0|fair_lay|fanduel | 9 | no_bet_sample | market_residual | 23.9% | 3.7% | 71.4% | 0.256 | 0.257 | 0.250 | 0.006 | 7 | 3.7% | 71.4% | 0.48 | 10.1% | residual_selected_sample; residual_calibration |
| batter_total_bases|under|common|TB 2.5+|lay_130_149|draftkings | 7 | no_bet_sample | market_no_vig | -76.0% | - | - | 0.362 | 0.281 | 0.333 | 0.029 | 5 | -66.4% | 75.0% | 0.69 | -48.2% | residual_selected_sample; residual_selected_roi; residual_calibration |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_100_149|fanduel | 6 | no_bet_sample | market_no_vig | 18.3% | -100.0% | 0.0% | 0.284 | 0.271 | 0.422 | -0.138 | 6 | 18.3% | 16.7% | -1.17 | -11.1% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
