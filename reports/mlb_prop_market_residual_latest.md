# MLB Prop Market Residual Models

Generated UTC: 2026-09-01T10:31:40Z
Rows: 61675
Raw rows before locked-offer dedupe: 143776
Collapsed duplicate locked-offer rows: 82101
Date range: 2026-06-01 to 2026-07-30
Status: ready

## Expanding Walk-Forward OOF Variants

| Variant | Rows | Brier | Cal Err | Selected | ROI | CLV Beat |
|---|---:|---:|---:|---:|---:|---:|
| model_only | 47224 | 0.249 | -1.9% | 4632 | 4.3% | 41.2% |
| market_no_vig | 47224 | 0.238 | 0.6% | 3267 | 149.8% | 42.6% |
| market_residual | 47224 | 0.232 | 0.5% | 13840 | 32.5% | 42.5% |

## CLV Target

| Rows | Beat Rate | Avg Prob | Brier | Cal Err | AUC | Status |
|---:|---:|---:|---:|---:|---:|---|
| 42702 | 40.5% | 40.3% | 0.235 | 0.2% | 0.601 | ready |

| CLV Variant | Rows | Brier | AUC | Cal Err | Selected |
|---|---:|---:|---:|---:|---|
| clv_v1 | 42702 | 0.239 | 0.569 | 0.4% | False |
| clv_v2 | 42702 | 0.235 | 0.601 | 0.2% | True |
| clv_v3 | 42702 | 0.238 | 0.596 | 3.2% | False |
| clv_v4 | 42702 | 0.238 | 0.597 | 3.0% | False |
| clv_v5 | 42702 | 0.238 | 0.597 | 2.9% | False |

CLV direction variants are evaluated with expanding walk-forward folds and trained only from true, non-synthetic paired offers.
Open-to-lock coverage: 88.2%; consensus coverage: 100.0%; true multi-book consensus: 58.0%.

## Expected CLV Magnitude v4

- Enabled: False
- Holdout rows: 42702
- Model MAE / baseline MAE: 1.382 / 1.353
- MAE gain: -0.029
- Sign accuracy: 45.7%
- Positive-CLV AUC: 0.585
- Target: implied-probability movement; positive and non-positive movement sizes are modeled separately.

## Exact Bucket Model Selection

| Bucket | Rows | Decision | Best | ROI | Best Sel ROI | Best Sel CLV | Model Brier | Market Brier | Residual Brier | Residual Gain | Residual Sel | Residual ROI | Residual CLV | Residual Avg CLV | Residual Cal | Proof Blockers |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2814 | use_market_baseline | market_no_vig | 4.3% | - | - | 0.239 | 0.237 | 0.238 | 0.002 | 1320 | -3.3% | 44.7% | 0.15 | 0.8% | residual_brier_gain; residual_selected_roi; residual_clv_beat |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 157 | use_market_residual | market_residual | 5.5% | 19.5% | 58.7% | 0.251 | 0.250 | 0.243 | 0.008 | 84 | 19.5% | 58.7% | 1.12 | 1.4% |  |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 87 | use_market_baseline | market_no_vig | 9.2% | - | - | 0.230 | 0.228 | 0.234 | -0.003 | 17 | -11.7% | 40.8% | 0.16 | 8.3% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 45 | keep_model_only | model_only | 12.3% | - | - | 0.233 | 0.234 | 0.240 | -0.006 | 1 | -100.0% | 0.0% | 0.00 | 14.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 36 | keep_model_only | model_only | 11.3% | 98.0% | - | 0.244 | 0.248 | 0.248 | -0.004 | 5 | 11.1% | 40.8% | -0.34 | 7.5% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 24 | keep_model_only | model_only | 11.2% | - | - | 0.237 | 0.238 | 0.261 | -0.024 | 1 | -100.0% | - | - | 14.0% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 21 | keep_model_only | model_only | 6.9% | - | - | 0.238 | 0.242 | 0.256 | -0.018 | 8 | -34.7% | 66.7% | 1.61 | 3.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | 4720 | no_bet_negative_roi | market_residual | -14.8% | 23.5% | 40.7% | 0.257 | 0.216 | 0.194 | 0.062 | 1367 | 23.5% | 40.7% | 0.12 | 7.7% | residual_clv_beat; residual_calibration |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3824 | no_bet_negative_roi | market_no_vig | -11.7% | - | - | 0.241 | 0.240 | 0.241 | -0.000 | 252 | -15.0% | 47.8% | 0.25 | -5.2% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_calibration |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | 3428 | no_bet_bad_clv | market_residual | 7.9% | 3.1% | 36.5% | 0.254 | 0.253 | 0.250 | 0.004 | 2193 | 3.1% | 36.5% | -0.16 | 1.8% | residual_clv_beat; residual_avg_clv |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 3392 | no_bet_negative_roi | market_no_vig | -14.1% | - | - | 0.237 | 0.237 | 0.240 | -0.003 | 11 | -24.3% | 77.2% | 2.11 | 5.8% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 3054 | no_bet_negative_roi | market_residual | -18.7% | 8.6% | 57.1% | 0.239 | 0.232 | 0.229 | 0.010 | 629 | 8.6% | 57.1% | 0.96 | -5.1% | residual_calibration |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1943 | no_bet_negative_roi | market_residual | -19.8% | 15.8% | 50.1% | 0.268 | 0.232 | 0.215 | 0.053 | 509 | 15.8% | 50.1% | 0.42 | 4.7% | residual_clv_beat |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | 1816 | no_bet_bad_clv | market_residual | 4.0% | 58.8% | 42.7% | 0.230 | 0.213 | 0.193 | 0.037 | 808 | 58.8% | 42.7% | 0.28 | -3.4% | residual_clv_beat |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1720 | no_bet_negative_roi | market_no_vig | -3.6% | - | - | 0.242 | 0.240 | 0.241 | 0.000 | 220 | -2.3% | 34.7% | -0.25 | -0.6% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1686 | no_bet_negative_roi | market_residual | -18.1% | -13.1% | 29.9% | 0.256 | 0.255 | 0.253 | 0.003 | 25 | -13.1% | 29.9% | -0.04 | -6.0% | residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 1595 | no_bet_bad_clv | market_residual | 65.2% | 72.6% | 47.0% | 0.260 | 0.247 | 0.222 | 0.037 | 1452 | 72.6% | 47.0% | 0.21 | 4.4% | residual_clv_beat |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1191 | no_bet_negative_roi | market_residual | -1.8% | 0.1% | 39.2% | 0.213 | 0.212 | 0.212 | 0.001 | 573 | 0.1% | 39.2% | -0.39 | -2.7% | residual_brier_gain; residual_clv_beat; residual_avg_clv |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 1134 | no_bet_bad_clv | market_residual | 1.0% | 2.5% | 24.6% | 0.227 | 0.225 | 0.225 | 0.002 | 309 | 2.5% | 24.6% | -0.90 | 0.9% | residual_brier_gain; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 1116 | no_bet_negative_roi | market_no_vig | -15.5% | - | - | 0.211 | 0.211 | 0.219 | -0.008 | 0 | - | - | - | 8.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 964 | no_bet_negative_roi | market_residual | -21.1% | 21.6% | 50.9% | 0.255 | 0.242 | 0.237 | 0.017 | 119 | 21.6% | 50.9% | 0.75 | -4.8% | residual_clv_beat |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 834 | no_bet_negative_roi | market_residual | -19.8% | -71.2% | 69.5% | 0.255 | 0.255 | 0.250 | 0.005 | 7 | -71.2% | 69.5% | 0.94 | -3.6% | residual_selected_sample; residual_selected_roi |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 803 | no_bet_no_edge | market_no_vig | 2.9% | - | - | 0.244 | 0.243 | 0.246 | -0.002 | 74 | -0.6% | 39.6% | -0.34 | 5.4% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 713 | no_bet_selected_negative_roi | market_residual | 4.9% | -1.7% | 27.4% | 0.254 | 0.251 | 0.249 | 0.005 | 383 | -1.7% | 27.4% | -0.82 | 1.2% | residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 701 | no_bet_negative_roi | market_residual | -10.4% | 6.5% | 53.9% | 0.221 | 0.216 | 0.216 | 0.005 | 296 | 6.5% | 53.9% | 1.16 | -4.2% | residual_clv_beat |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 654 | no_bet_negative_roi | market_residual | -30.0% | 14.9% | 47.6% | 0.293 | 0.229 | 0.205 | 0.088 | 142 | 14.9% | 47.6% | 0.65 | -0.7% | residual_clv_beat |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 634 | no_bet_negative_roi | market_residual | -17.5% | -40.7% | 35.5% | 0.251 | 0.250 | 0.248 | 0.003 | 9 | -40.7% | 35.5% | 1.57 | -2.5% | residual_selected_sample; residual_selected_roi; residual_clv_beat |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 546 | no_bet_negative_roi | market_residual | -13.0% | 2.4% | 70.4% | 0.248 | 0.248 | 0.247 | 0.001 | 11 | 2.4% | 70.4% | 0.78 | 3.4% | residual_selected_sample; residual_brier_gain |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 518 | no_bet_bad_clv | market_residual | 360.2% | 360.2% | 37.1% | 0.344 | 0.324 | 0.233 | 0.111 | 518 | 360.2% | 37.1% | 0.09 | 3.9% | residual_clv_beat |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 483 | no_bet_negative_roi | model_only | -1.4% | 34.0% | 62.3% | 0.248 | 0.249 | 0.250 | -0.002 | 57 | 11.4% | 44.8% | -0.09 | 2.2% | residual_brier_gain; residual_clv_beat; residual_avg_clv |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 415 | no_bet_bad_clv | market_residual | 46.9% | 49.0% | 46.5% | 0.232 | 0.225 | 0.213 | 0.019 | 377 | 49.0% | 46.5% | 0.11 | 0.5% | residual_clv_beat |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 336 | no_bet_bad_clv | market_residual | 29.2% | 58.5% | 41.4% | 0.252 | 0.244 | 0.228 | 0.025 | 207 | 58.5% | 41.4% | 0.21 | 3.2% | residual_clv_beat |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 313 | no_bet_negative_roi | market_no_vig | -3.8% | - | - | 0.249 | 0.246 | 0.249 | -0.001 | 13 | 33.2% | 64.8% | 1.63 | 4.3% | residual_selected_sample; residual_brier_gain |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 267 | no_bet_negative_roi | market_no_vig | -5.8% | - | - | 0.249 | 0.247 | 0.250 | -0.000 | 4 | 46.0% | 0.0% | -4.05 | 3.7% | residual_selected_sample; residual_brier_gain; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 250 | no_bet_negative_roi | market_residual | -21.8% | 18.1% | 51.6% | 0.262 | 0.230 | 0.210 | 0.051 | 58 | 18.1% | 51.6% | 0.74 | 1.2% | residual_clv_beat |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 240 | no_bet_negative_roi | market_residual | -15.3% | 28.9% | 34.2% | 0.261 | 0.240 | 0.224 | 0.037 | 28 | 28.9% | 34.2% | 0.06 | 1.8% | residual_clv_beat |
| batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 235 | no_bet_bad_clv | market_residual | 384.0% | 384.0% | 23.6% | 0.236 | 0.227 | 0.200 | 0.035 | 235 | 384.0% | 23.6% | -0.02 | 0.2% | residual_clv_beat; residual_avg_clv |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 218 | no_bet_negative_roi | market_no_vig | -16.2% | - | - | 0.241 | 0.241 | 0.251 | -0.009 | 112 | -16.8% | 59.6% | 1.41 | -8.7% | residual_brier_gain; residual_selected_roi; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 193 | no_bet_negative_roi | market_no_vig | -13.0% | - | - | 0.241 | 0.241 | 0.245 | -0.004 | 83 | -1.1% | 65.7% | 1.31 | -6.5% | residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 178 | no_bet_residual_unproven | market_residual | 73.4% | 76.2% | 51.0% | 0.254 | 0.244 | 0.222 | 0.032 | 175 | 76.2% | 51.0% | 0.59 | -3.4% | residual_clv_beat |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 175 | no_bet_negative_roi | market_no_vig | -18.5% | - | - | 0.221 | 0.220 | 0.227 | -0.006 | 0 | - | - | - | 6.7% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 165 | no_bet_negative_roi | market_no_vig | -1.0% | - | - | 0.241 | 0.238 | 0.252 | -0.011 | 124 | -4.3% | 34.0% | -1.36 | -5.4% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 157 | no_bet_negative_roi | market_residual | -1.3% | 20.5% | 60.5% | 0.249 | 0.249 | 0.243 | 0.006 | 52 | 20.5% | 60.5% | 1.35 | -1.0% |  |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 156 | no_bet_negative_roi | market_residual | -14.5% | 14.2% | 57.5% | 0.252 | 0.250 | 0.248 | 0.004 | 8 | 14.2% | 57.5% | 1.71 | -2.6% | residual_selected_sample |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 145 | no_bet_negative_roi | model_only | -7.0% | -4.0% | 74.9% | 0.247 | 0.250 | 0.251 | -0.004 | 7 | -36.9% | 8.7% | -0.39 | 2.1% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 140 | no_bet_negative_roi | market_residual | -25.7% | - | - | 0.232 | 0.225 | 0.220 | 0.011 | 0 | - | - | - | 2.3% | residual_selected_sample; residual_selected_roi; residual_clv_beat; residual_avg_clv |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 137 | no_bet_negative_roi | market_no_vig | -15.9% | - | - | 0.260 | 0.254 | 0.266 | -0.006 | 70 | -21.7% | 59.8% | 0.32 | -11.9% | residual_brier_gain; residual_selected_roi; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 133 | no_bet_negative_roi | market_residual | -18.7% | 9.6% | 30.1% | 0.266 | 0.238 | 0.224 | 0.042 | 16 | 9.6% | 30.1% | 0.32 | -0.4% | residual_selected_sample; residual_clv_beat |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 129 | no_bet_negative_roi | market_residual | -27.6% | 3.6% | 45.3% | 0.244 | 0.208 | 0.201 | 0.044 | 57 | 3.6% | 45.3% | 0.04 | -14.8% | residual_clv_beat; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 128 | no_bet_negative_roi | market_no_vig | -10.8% | - | - | 0.254 | 0.243 | 0.243 | 0.011 | 41 | -13.8% | 50.0% | 0.79 | -4.3% | residual_selected_roi; residual_clv_beat |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 121 | no_bet_negative_roi | market_no_vig | -5.8% | - | - | 0.247 | 0.244 | 0.255 | -0.007 | 78 | -9.7% | 36.3% | -0.08 | -6.2% | residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 120 | no_bet_negative_roi | market_no_vig | -9.8% | - | - | 0.256 | 0.250 | 0.256 | 0.001 | 42 | -5.0% | 57.4% | 0.88 | -6.0% | residual_brier_gain; residual_selected_roi; residual_calibration |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 111 | no_bet_negative_roi | market_no_vig | -18.8% | - | - | 0.238 | 0.236 | 0.238 | -0.000 | 25 | -28.5% | 58.3% | 1.73 | -7.2% | residual_brier_gain; residual_selected_roi; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 110 | no_bet_bad_clv | market_residual | 4.0% | 20.1% | 16.1% | 0.243 | 0.236 | 0.234 | 0.010 | 40 | 20.1% | 16.1% | -1.28 | 3.0% | residual_clv_beat; residual_avg_clv |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 107 | no_bet_bad_clv | market_residual | 62.8% | 62.8% | 35.6% | 0.202 | 0.193 | 0.190 | 0.012 | 107 | 62.8% | 35.6% | 0.21 | -12.4% | residual_clv_beat; residual_calibration |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 98 | no_bet_bad_clv | model_only | 1.4% | 32.2% | 0.0% | 0.237 | 0.244 | 0.255 | -0.018 | 12 | -60.9% | 34.9% | -0.39 | 5.2% | residual_selected_sample; residual_brier_gain; residual_selected_roi; residual_clv_beat; residual_avg_clv; residual_calibration |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 97 | no_bet_negative_roi | market_no_vig | -22.3% | - | - | 0.266 | 0.256 | 0.267 | -0.001 | 4 | -59.5% | 67.1% | 1.13 | -3.5% | residual_selected_sample; residual_brier_gain; residual_selected_roi |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 89 | no_bet_negative_roi | market_no_vig | -3.6% | - | - | 0.254 | 0.247 | 0.253 | 0.001 | 2 | -44.8% | 67.1% | 0.03 | 4.0% | residual_selected_sample; residual_brier_gain; residual_selected_roi |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 86 | no_bet_bad_clv | market_residual | 2.2% | 55.1% | 45.1% | 0.272 | 0.250 | 0.242 | 0.030 | 11 | 55.1% | 45.1% | 0.88 | 5.0% | residual_selected_sample; residual_clv_beat; residual_calibration |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 85 | no_bet_negative_roi | market_no_vig | -1.8% | - | - | 0.251 | 0.248 | 0.263 | -0.013 | 29 | 3.9% | 55.8% | 0.84 | 1.7% | residual_brier_gain |
