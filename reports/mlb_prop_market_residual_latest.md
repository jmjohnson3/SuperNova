# MLB Prop Market Residual Models

Generated UTC: 2026-06-19T07:08:26Z
Rows: 35270
Date range: 2026-05-31 to 2026-06-17
Status: ready

## Holdout Variants

| Variant | Rows | Brier | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|
| model_only | 1612 | 0.232 | 3.3% | 200 | 17.8% | 38.2% |
| market_no_vig | 1612 | 0.227 | 4.9% | 133 | 139.0% | 27.8% |
| market_residual | 1612 | 0.217 | 3.8% | 372 | 64.8% | 32.9% |

## CLV Target

| Rows | Beat Rate | Avg Prob | Brier | Cal Err | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1531 | 32.7% | 37.8% | 0.221 | -5.2% | 0.582 | ready |

| CLV Variant | Rows | Brier | AUC | Cal Err | Selected |
|---|---:|---:|---:|---:|---|
| clv_v1 | 1531 | 0.222 | 0.558 | -4.6% | False |
| clv_v2 | 1531 | 0.221 | 0.582 | -5.2% | True |

CLV v2 features are trained only from true, non-synthetic paired offers.
Open-to-lock coverage: 95.9%; consensus coverage: 100.0%; true multi-book consensus: 59.3%.

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | ROI | Best Sel ROI | Best Sel CLV | Model Brier | Market Brier | Residual Brier | Residual Gain | Residual Sel | Residual ROI | Residual CLV | Residual Avg CLV | Residual Cal | Proof Blockers |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 126 | keep_model_only | model_only | 7.5% | 14.9% | 52.0% | 0.255 | 0.256 | 0.265 | -0.010 | 0 | - | - | - | 12.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 45 | keep_model_only | model_only | 4.8% | - | - | 0.233 | 0.233 | 0.246 | -0.013 | 0 | - | - | - | 15.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 189 | no_bet_bad_clv | market_residual | 7.5% | 39.5% | 35.9% | 0.193 | 0.190 | 0.155 | 0.038 | 68 | 39.5% | 35.9% | 0.10 | 15.7% | residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 140 | no_bet_bad_clv | market_residual | 9.6% | 27.2% | 40.5% | 0.194 | 0.193 | 0.184 | 0.010 | 43 | 27.2% | 40.5% | 0.03 | 6.8% | residual_clv_beat; residual_calibration |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 124 | no_bet_negative_roi | market_no_vig | -24.9% | - | - | 0.228 | 0.228 | 0.231 | -0.002 | 7 | -35.7% | 40.0% | -0.12 | -8.2% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 112 | no_bet_bad_clv | market_residual | 10.1% | 20.3% | 38.9% | 0.269 | 0.256 | 0.253 | 0.016 | 18 | 20.3% | 38.9% | 0.63 | 7.9% | residual_selected_sample; residual_clv_beat; residual_calibration |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 103 | no_bet_negative_roi | model_only | -42.2% | 161.0% | 0.0% | 0.182 | 0.183 | 0.184 | -0.002 | 10 | -22.7% | 33.3% | -0.31 | -12.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 84 | no_bet_bad_clv | market_residual | 3.6% | 39.7% | 33.3% | 0.232 | 0.205 | 0.181 | 0.051 | 39 | 39.7% | 33.3% | 0.01 | -5.1% | residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 68 | no_bet_bad_clv | market_residual | 13.2% | 43.4% | 20.0% | 0.226 | 0.225 | 0.207 | 0.018 | 20 | 43.4% | 20.0% | -0.59 | 20.3% | residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 56 | no_bet_negative_roi | market_residual | -13.2% | -22.2% | 50.0% | 0.252 | 0.252 | 0.252 | 0.000 | 2 | -22.2% | 50.0% | 0.12 | -5.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 53 | no_bet_bad_clv | market_residual | 109.8% | 118.0% | 33.3% | 0.315 | 0.297 | 0.249 | 0.066 | 51 | 118.0% | 33.3% | -0.44 | 12.1% | residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 53 | no_bet_negative_roi | market_residual | -6.8% | 70.5% | 33.3% | 0.253 | 0.244 | 0.225 | 0.028 | 9 | 70.5% | 33.3% | -0.28 | 2.8% | residual_selected_sample; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 51 | no_bet_bad_clv | market_residual | 16.6% | 60.8% | 0.0% | 0.219 | 0.220 | 0.212 | 0.007 | 2 | 60.8% | 0.0% | -1.18 | 13.4% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 50 | no_bet_negative_roi | model_only | -9.2% | - | - | 0.230 | 0.230 | 0.243 | -0.013 | 0 | - | - | - | -3.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 40 | no_bet_negative_roi | market_no_vig | -5.5% | - | - | 0.248 | 0.248 | 0.256 | -0.009 | 0 | - | - | - | 3.3% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 39 | no_bet_negative_roi | market_no_vig | -18.4% | - | - | 0.250 | 0.250 | 0.253 | -0.003 | 4 | -24.5% | 0.0% | 0.00 | -9.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 38 | no_bet_negative_roi | market_residual | -0.2% | -100.0% | - | 0.245 | 0.245 | 0.239 | 0.006 | 1 | -100.0% | - | - | 3.6% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 26 | no_bet_bad_clv | market_residual | 273.1% | 273.1% | 34.6% | 0.297 | 0.275 | 0.234 | 0.063 | 26 | 273.1% | 34.6% | -0.06 | -9.8% | residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 23 | no_bet_negative_roi | model_only | -11.5% | - | - | 0.249 | 0.249 | 0.264 | -0.015 | 0 | - | - | - | 1.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 21 | no_bet_negative_roi | market_residual | -34.4% | 15.2% | 0.0% | 0.261 | 0.231 | 0.199 | 0.062 | 6 | 15.2% | 0.0% | -0.73 | -7.7% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 21 | no_bet_negative_roi | market_residual | -0.9% | - | - | 0.247 | 0.247 | 0.238 | 0.009 | 0 | - | - | - | 8.7% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 21 | no_bet_bad_clv | market_residual | 38.8% | 39.6% | 21.4% | 0.278 | 0.268 | 0.246 | 0.032 | 14 | 39.6% | 21.4% | -0.75 | 7.6% | residual_selected_sample; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 16 | no_bet_sample | market_no_vig | 65.6% | 50.0% | 23.1% | 0.264 | 0.256 | 0.273 | -0.009 | 13 | 50.0% | 23.1% | -0.27 | -0.7% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 15 | no_bet_sample | market_residual | 4.0% | 30.0% | 71.4% | 0.207 | 0.212 | 0.199 | 0.008 | 7 | 30.0% | 71.4% | 0.83 | -2.0% | residual_selected_sample |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 14 | no_bet_sample | model_only | -46.2% | - | - | 0.240 | 0.242 | 0.245 | -0.004 | 0 | - | - | - | -21.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 12 | no_bet_sample | model_only | 22.3% | 12.1% | 40.0% | 0.235 | 0.245 | 0.247 | -0.013 | 1 | 80.7% | 0.0% | 0.00 | 15.8% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_calibration |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 11 | no_bet_sample | model_only | -23.6% | -23.6% | 54.5% | 0.155 | 0.156 | 0.166 | -0.011 | 11 | -23.6% | 54.5% | 0.96 | -21.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 10 | no_bet_sample | market_residual | 9.7% | 82.2% | 0.0% | 0.243 | 0.226 | 0.162 | 0.081 | 3 | 82.2% | 0.0% | 0.00 | 12.6% | residual_selected_sample; residual_clv_beat; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 8 | no_bet_sample | market_residual | 30.7% | 74.3% | 33.3% | 0.248 | 0.232 | 0.188 | 0.060 | 3 | 74.3% | 33.3% | 0.03 | 21.2% | residual_selected_sample; residual_clv_beat; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 8 | no_bet_sample | market_residual | -68.5% | - | - | 0.175 | 0.169 | 0.139 | 0.036 | 0 | - | - | - | -16.5% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 6 | no_bet_sample | market_residual | 20.8% | 141.7% | 0.0% | 0.262 | 0.234 | 0.155 | 0.108 | 3 | 141.7% | 0.0% | -0.30 | 1.6% | residual_selected_sample; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 5 | no_bet_sample | market_residual | -51.2% | - | - | 0.227 | 0.197 | 0.155 | 0.072 | 0 | - | - | - | -6.5% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 5 | no_bet_sample | market_no_vig | 20.0% | 20.0% | 0.0% | 0.157 | 0.156 | 0.204 | -0.046 | 5 | 20.0% | 0.0% | -0.48 | -21.0% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 4 | no_bet_sample | model_only | -55.8% | - | - | 0.243 | 0.269 | 0.252 | -0.009 | 0 | - | - | - | -22.9% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 3 | no_bet_sample | market_residual | 566.7% | 566.7% | 0.0% | 0.307 | 0.299 | 0.261 | 0.046 | 3 | 566.7% | 0.0% | -0.38 | 4.5% | residual_selected_sample; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 3 | no_bet_sample | market_residual | -19.7% | - | - | 0.266 | 0.265 | 0.263 | 0.003 | 0 | - | - | - | -8.7% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 2 | no_bet_sample | market_residual | 72.5% | - | - | 0.218 | 0.210 | 0.203 | 0.016 | 0 | - | - | - | 45.0% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 2 | no_bet_sample | market_residual | 200.0% | 200.0% | 50.0% | 0.362 | 0.335 | 0.215 | 0.148 | 2 | 200.0% | 50.0% | 0.69 | 9.1% | residual_selected_sample; residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 1 | no_bet_sample | model_only | 115.0% | 115.0% | 0.0% | 0.241 | 0.358 | 0.353 | -0.112 | 0 | - | - | - | 59.4% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|under|common|H 0.5|heavy_lay|draftkings | 1 | no_bet_sample | model_only | -100.0% | - | - | 0.356 | 0.374 | 0.432 | -0.076 | 0 | - | - | - | -65.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 1 | no_bet_sample | model_only | -100.0% | - | - | 0.259 | 0.323 | 0.334 | -0.075 | 0 | - | - | - | -57.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 1 | no_bet_sample | market_no_vig | -100.0% | - | - | 0.089 | 0.071 | 0.107 | -0.018 | 1 | -100.0% | 0.0% | -1.54 | -32.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 1 | no_bet_sample | market_no_vig | 105.0% | - | - | 0.297 | 0.297 | 0.344 | -0.047 | 0 | - | - | - | 58.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
