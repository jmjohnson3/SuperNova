# MLB Prop Market Residual Models

Generated UTC: 2026-06-24T08:16:17Z
Rows: 48371
Date range: 2026-05-31 to 2026-06-22
Status: ready

## Holdout Variants

| Variant | Rows | Brier | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|
| model_only | 3172 | 0.247 | -1.6% | 417 | 6.5% | 33.1% |
| market_no_vig | 3172 | 0.234 | 0.4% | 194 | 110.2% | 30.9% |
| market_residual | 3172 | 0.227 | -0.7% | 916 | 20.1% | 35.8% |

## CLV Target

| Rows | Beat Rate | Avg Prob | Brier | Cal Err | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 2959 | 33.7% | 39.8% | 0.217 | -6.1% | 0.626 | ready |

| CLV Variant | Rows | Brier | AUC | Cal Err | Selected |
|---|---:|---:|---:|---:|---|
| clv_v1 | 2959 | 0.223 | 0.590 | -5.9% | False |
| clv_v2 | 2959 | 0.217 | 0.626 | -6.1% | True |

CLV v2 features are trained only from true, non-synthetic paired offers.
Open-to-lock coverage: 92.1%; consensus coverage: 100.0%; true multi-book consensus: 59.2%.

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | ROI | Best Sel ROI | Best Sel CLV | Model Brier | Market Brier | Residual Brier | Residual Gain | Residual Sel | Residual ROI | Residual CLV | Residual Avg CLV | Residual Cal | Proof Blockers |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 244 | use_market_baseline | market_no_vig | 5.4% | - | - | 0.258 | 0.253 | 0.260 | -0.002 | 1 | -100.0% | - | - | 7.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 57 | keep_model_only | model_only | 5.9% | - | - | 0.243 | 0.243 | 0.252 | -0.009 | 0 | - | - | - | 11.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 37 | keep_model_only | model_only | 4.9% | - | - | 0.250 | 0.250 | 0.259 | -0.010 | 0 | - | - | - | 11.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 23 | keep_model_only | model_only | 37.7% | - | - | 0.262 | 0.262 | 0.273 | -0.010 | 0 | - | - | - | 22.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 318 | no_bet_negative_roi | market_residual | -18.2% | 28.7% | 26.1% | 0.250 | 0.209 | 0.168 | 0.082 | 121 | 28.7% | 26.1% | -0.15 | -0.0% | residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 230 | no_bet_negative_roi | market_no_vig | -9.3% | - | - | 0.234 | 0.233 | 0.238 | -0.003 | 171 | -8.0% | 41.0% | 0.15 | -9.3% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 201 | no_bet_negative_roi | model_only | -5.1% | - | - | 0.246 | 0.246 | 0.253 | -0.007 | 3 | -22.7% | 100.0% | 2.27 | 6.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 186 | no_bet_negative_roi | market_residual | -9.8% | 4.0% | 41.8% | 0.245 | 0.237 | 0.231 | 0.014 | 58 | 4.0% | 41.8% | 0.42 | -3.5% | residual_clv_beat |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 172 | no_bet_negative_roi | market_no_vig | -5.9% | - | - | 0.227 | 0.225 | 0.226 | 0.001 | 3 | 0.0% | 0.0% | 0.00 | 3.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 141 | no_bet_negative_roi | market_residual | -8.5% | 39.7% | 34.0% | 0.250 | 0.220 | 0.175 | 0.075 | 51 | 39.7% | 34.0% | -0.24 | 5.4% | residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 131 | no_bet_bad_clv | market_residual | 14.6% | 74.7% | 45.1% | 0.255 | 0.228 | 0.196 | 0.059 | 71 | 74.7% | 45.1% | 0.03 | -2.6% | residual_clv_beat |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 118 | no_bet_negative_roi | market_no_vig | -16.5% | - | - | 0.260 | 0.255 | 0.264 | -0.004 | 49 | -14.5% | 34.9% | 0.51 | -11.2% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 114 | no_bet_negative_roi | model_only | -10.7% | - | - | 0.251 | 0.251 | 0.252 | -0.001 | 0 | - | - | - | -2.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 99 | no_bet_bad_clv | market_residual | 31.7% | 41.7% | 29.3% | 0.224 | 0.209 | 0.204 | 0.020 | 92 | 41.7% | 29.3% | -0.47 | -4.4% | residual_clv_beat; residual_avg_clv |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 85 | no_bet_selected_negative_roi | market_residual | 5.4% | -12.1% | 50.0% | 0.208 | 0.208 | 0.205 | 0.003 | 8 | -12.1% | 50.0% | -0.60 | 5.3% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 74 | no_bet_negative_roi | model_only | -2.4% | - | - | 0.247 | 0.247 | 0.248 | -0.001 | 8 | 7.8% | 62.5% | 1.44 | 0.7% | residual_selected_sample; residual_brier_gain |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 70 | no_bet_negative_roi | model_only | -15.8% | - | - | 0.215 | 0.215 | 0.219 | -0.004 | 0 | - | - | - | 7.6% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 69 | no_bet_negative_roi | market_no_vig | -31.9% | - | - | 0.258 | 0.242 | 0.253 | 0.004 | 24 | -21.4% | 25.0% | -0.76 | -15.7% | residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 63 | no_bet_negative_roi | market_no_vig | -36.9% | -100.0% | 0.0% | 0.197 | 0.187 | 0.196 | 0.001 | 27 | -52.2% | 44.4% | 0.58 | -13.0% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 55 | no_bet_negative_roi | model_only | -8.7% | - | - | 0.241 | 0.241 | 0.245 | -0.003 | 12 | -11.9% | - | - | -5.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 49 | no_bet_negative_roi | market_residual | -25.4% | - | - | 0.248 | 0.248 | 0.245 | 0.002 | 0 | - | - | - | -8.4% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 45 | no_bet_bad_clv | market_residual | 14.9% | 39.2% | 36.8% | 0.219 | 0.234 | 0.200 | 0.019 | 21 | 39.2% | 36.8% | 0.37 | 14.2% | residual_clv_beat; residual_calibration |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 45 | no_bet_negative_roi | model_only | -23.4% | - | - | 0.229 | 0.247 | 0.242 | -0.013 | 3 | -100.0% | 50.0% | 0.44 | -7.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 39 | no_bet_selected_negative_roi | market_residual | 3.3% | -9.2% | 75.0% | 0.249 | 0.249 | 0.245 | 0.004 | 4 | -9.2% | 75.0% | 3.07 | 4.3% | residual_selected_sample; residual_selected_roi |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 24 | no_bet_bad_clv | market_residual | 254.2% | 254.2% | 37.5% | 0.288 | 0.293 | 0.240 | 0.048 | 24 | 254.2% | 37.5% | -0.08 | -2.6% | residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 23 | no_bet_negative_roi | market_residual | -35.2% | 24.6% | 0.0% | 0.268 | 0.220 | 0.171 | 0.097 | 3 | 24.6% | 0.0% | 0.00 | -5.0% | residual_selected_sample; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 23 | no_bet_negative_roi | market_residual | -78.5% | - | - | 0.177 | 0.177 | 0.123 | 0.055 | 0 | - | - | - | -19.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 22 | no_bet_no_edge | market_no_vig | 39.3% | - | - | 0.276 | 0.276 | 0.281 | -0.006 | 6 | 9.8% | - | - | 21.2% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 21 | no_bet_bad_clv | market_residual | 74.3% | 100.0% | 28.6% | 0.270 | 0.297 | 0.245 | 0.025 | 14 | 100.0% | 28.6% | -0.85 | 19.6% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 19 | no_bet_sample | market_residual | 178.4% | 178.4% | 26.3% | 0.405 | 0.355 | 0.256 | 0.149 | 19 | 178.4% | 26.3% | -0.36 | 23.9% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 17 | no_bet_sample | market_no_vig | -33.6% | - | - | 0.572 | 0.242 | 0.295 | 0.277 | 7 | -76.2% | 14.3% | -0.48 | -20.9% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 15 | no_bet_sample | model_only | -100.0% | -100.0% | 66.7% | 0.050 | 0.066 | 0.154 | -0.104 | 15 | -100.0% | 60.0% | 0.89 | -38.3% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 15 | no_bet_sample | market_no_vig | -19.5% | - | - | 0.511 | 0.257 | 0.328 | 0.183 | 9 | -42.2% | 22.2% | -0.14 | -19.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 15 | no_bet_sample | market_residual | -57.9% | - | - | 0.211 | 0.211 | 0.194 | 0.017 | 0 | - | - | - | -18.4% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 14 | no_bet_sample | market_no_vig | -100.0% | - | - | 0.208 | 0.162 | 0.169 | 0.039 | 6 | -100.0% | 0.0% | -1.23 | -40.4% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 14 | no_bet_sample | market_no_vig | 17.6% | - | - | 0.260 | 0.260 | 0.289 | -0.030 | 0 | - | - | - | 21.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 14 | no_bet_sample | market_no_vig | -52.9% | - | - | 0.288 | 0.288 | 0.328 | -0.039 | 14 | -52.9% | 0.0% | -4.11 | -39.0% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 13 | no_bet_sample | market_no_vig | -20.8% | - | - | 0.524 | 0.242 | 0.269 | 0.256 | 7 | -26.9% | 28.6% | 0.10 | -11.9% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 12 | no_bet_sample | model_only | 25.8% | 89.0% | 33.3% | 0.251 | 0.265 | 0.278 | -0.026 | 0 | - | - | - | 20.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 12 | no_bet_sample | market_residual | -45.2% | - | - | 0.225 | 0.225 | 0.212 | 0.013 | 0 | - | - | - | -14.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 11 | no_bet_sample | market_residual | 163.6% | 163.6% | 36.4% | 0.172 | 0.153 | 0.136 | 0.036 | 11 | 163.6% | 36.4% | 0.21 | -9.9% | residual_selected_sample; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|over|common|K <4.5|fair_lay|fanduel | 11 | no_bet_sample | market_no_vig | -46.2% | - | - | 0.267 | 0.267 | 0.280 | -0.014 | 2 | -100.0% | 100.0% | 2.35 | -24.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 11 | no_bet_sample | model_only | -32.3% | - | - | 0.258 | 0.258 | 0.265 | -0.006 | 0 | - | - | - | -13.6% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | market_no_vig | 16.8% | - | - | 0.264 | 0.264 | 0.283 | -0.020 | 0 | - | - | - | 12.0% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 10 | no_bet_sample | market_residual | 48.6% | 22.9% | 33.3% | 0.197 | 0.197 | 0.178 | 0.019 | 4 | 22.9% | 33.3% | -1.29 | 28.5% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 10 | no_bet_sample | market_residual | -22.9% | - | - | 0.264 | 0.264 | 0.251 | 0.012 | 0 | - | - | - | -8.7% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 10 | no_bet_sample | market_no_vig | 55.8% | - | - | 0.266 | 0.266 | 0.279 | -0.014 | 0 | - | - | - | 31.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 10 | no_bet_sample | market_no_vig | 25.8% | - | - | 0.253 | 0.253 | 0.258 | -0.005 | 3 | -27.0% | 100.0% | 1.07 | 16.2% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 9 | no_bet_sample | market_no_vig | -100.0% | -100.0% | 22.2% | 0.037 | 0.031 | 0.146 | -0.109 | 9 | -100.0% | 22.2% | 0.12 | -38.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 9 | no_bet_sample | model_only | -81.0% | - | - | 0.294 | 0.294 | 0.339 | -0.045 | 3 | -100.0% | - | - | -47.6% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 9 | no_bet_sample | market_residual | 8.2% | 60.7% | - | 0.229 | 0.229 | 0.217 | 0.012 | 3 | 60.7% | - | - | 2.3% | residual_selected_sample; residual_clv_beat; residual_avg_clv |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 8 | no_bet_sample | model_only | -14.6% | - | - | 0.253 | 0.253 | 0.258 | -0.005 | 2 | -14.0% | 0.0% | -1.64 | -6.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 8 | no_bet_sample | market_residual | 2.5% | - | - | 0.239 | 0.239 | 0.235 | 0.004 | 0 | - | - | - | 7.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 7 | no_bet_sample | market_residual | -71.4% | - | - | 0.290 | 0.214 | 0.202 | 0.089 | 0 | - | - | - | -30.1% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 7 | no_bet_sample | market_residual | 8.0% | - | - | 0.248 | 0.248 | 0.242 | 0.006 | 0 | - | - | - | 6.6% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 6 | no_bet_sample | market_no_vig | 42.3% | - | - | 0.293 | 0.218 | 0.244 | 0.049 | 0 | - | - | - | 32.2% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 6 | no_bet_sample | market_residual | 36.8% | 10.2% | 33.3% | 0.209 | 0.209 | 0.195 | 0.014 | 3 | 10.2% | 33.3% | -0.77 | 21.1% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 6 | no_bet_sample | market_no_vig | -43.5% | - | - | 0.261 | 0.261 | 0.264 | -0.003 | 2 | -16.2% | 0.0% | 0.00 | -25.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|under|common|H 1.5|lay_150_180|draftkings | 5 | no_bet_sample | market_residual | 56.8% | - | - | 0.187 | 0.163 | 0.153 | 0.034 | 0 | - | - | - | 39.0% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 5 | no_bet_sample | market_residual | 2.6% | -15.3% | 25.0% | 0.238 | 0.238 | 0.238 | 0.001 | 4 | -15.3% | 25.0% | -0.12 | -0.2% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
