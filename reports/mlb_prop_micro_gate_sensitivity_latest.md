# MLB Prop Micro Gate Sensitivity

Generated UTC: 2026-09-01T11:23:32Z
Status: **ready**
Usage: diagnostic only. This report does not promote live bets.

## Profile Summary

| Profile | Money? | Passes | Evaluated | Rows | CLV Rows | Clean Dates | CLV Beat | Avg CLV | ROI | Cal Err | Valid Close |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| watch_only | False | 1 | 121 | 50 | 15 | 3 | 50.0% | -0.100 | -5.0% | 10.0% | 85.0% |
| micro_test | True | 0 | 121 | 75 | 20 | 4 | 52.0% | 0.000 | 0.0% | 8.0% | 90.0% |
| strict_micro | True | 0 | 121 | 150 | 30 | 5 | 55.0% | 0.000 | 0.0% | 5.0% | 90.0% |

## Hard Blocker Counts

| Blocker | Buckets |
|---|---:|
| exact_bucket_true_pair_proof_missing | 44 |
| underlying_projection_not_proven | 27 |
| fanduel_hitter_true_pair_proof_missing | 22 |

## Closest Buckets By Profile

| Profile | Bucket | Rows | Clean | ROI | CLV Rows | CLV Beat | Avg CLV | Cal Err | Coverage | Score | Passes | Blockers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| watch_only | pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 129 | 5 | -1.1% | 92 | 52.2% | 0.838 | 2.2% | 98.9% | 0.0 | True | - |
| watch_only | pitcher_strikeouts|over|common|K <4.5|lay_130_149|draftkings | 45 | 4 | -1.3% | 42 | 50.0% | -0.062 | 4.6% | 100.0% | 0.3 | False | needs_5_rows |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 286 | 6 | -5.7% | 209 | 50.7% | 0.729 | 0.3% | 96.8% | 0.7 | False | roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 87 | 5 | -8.1% | 67 | 68.7% | 1.197 | -1.6% | 98.5% | 3.1 | False | roi_below_gate |
| watch_only | batter_hits|over|common|H 0.5|fair_lay|draftkings | 602 | 6 | -7.3% | 547 | 49.2% | 0.729 | -0.4% | 97.2% | 3.1 | False | clv_beat_short_0.008, roi_below_gate |
| watch_only | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2638 | 6 | -3.2% | 2200 | 43.2% | 0.125 | 1.5% | 96.5% | 6.8 | False | clv_beat_short_0.068 |
| watch_only | pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 110 | 5 | -11.5% | 92 | 48.9% | 0.603 | -2.5% | 98.9% | 7.6 | False | clv_beat_short_0.011, roi_below_gate |
| watch_only | batter_hits|over|common|H 0.5|lay_130_149|draftkings | 762 | 6 | -8.3% | 684 | 45.6% | 0.303 | -1.1% | 97.0% | 7.7 | False | clv_beat_short_0.044, roi_below_gate |
| watch_only | batter_hits|over|common|H 1.5|plus_150_249|draftkings | 982 | 6 | -13.3% | 755 | 53.8% | 0.456 | -2.5% | 96.3% | 8.3 | False | roi_below_gate |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 142 | 5 | 1.8% | 134 | 41.8% | -0.294 | 2.5% | 95.7% | 8.3 | False | avg_clv_below_gate, clv_beat_short_0.082 |
| watch_only | pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 39 | 2 | 5.6% | 35 | 54.3% | 0.593 | 6.7% | 97.2% | 8.7 | False | needs_11_rows, needs_1_clean_dates |
| watch_only | batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1580 | 6 | -8.7% | 1437 | 44.9% | 0.237 | -1.7% | 96.2% | 8.8 | False | clv_beat_short_0.051, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 138 | 5 | -13.9% | 129 | 49.6% | 0.193 | -3.6% | 94.9% | 9.3 | False | clv_beat_short_0.004, roi_below_gate |
| watch_only | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|fanduel | 34 | 3 | -13.6% | 32 | 56.2% | 0.218 | -4.7% | 97.0% | 9.6 | False | needs_16_rows, roi_below_gate |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 188 | 5 | -12.3% | 151 | 47.7% | 0.708 | -3.0% | 95.6% | 9.6 | False | clv_beat_short_0.023, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 24 | 2 | 7.5% | 15 | 60.0% | 0.129 | 7.6% | 100.0% | 9.7 | False | needs_1_clean_dates, needs_26_rows |
| watch_only | batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3578 | 6 | -7.4% | 3080 | 42.5% | -0.039 | -1.3% | 96.6% | 9.8 | False | clv_beat_short_0.075, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 95 | 5 | 4.8% | 79 | 40.5% | -0.993 | 5.1% | 97.5% | 9.9 | False | avg_clv_below_gate, clv_beat_short_0.095 |
| watch_only | pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 19 | 2 | 9.2% | 16 | 81.2% | 1.341 | 8.4% | 88.9% | 10.1 | False | needs_1_clean_dates, needs_31_rows |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 133 | 5 | -7.3% | 130 | 41.5% | -0.033 | -2.2% | 99.2% | 10.8 | False | clv_beat_short_0.085, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 125 | 5 | -5.2% | 120 | 39.2% | 0.100 | 0.4% | 98.4% | 11.0 | False | clv_beat_short_0.108, roi_below_gate |
| watch_only | pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 19 | 3 | 9.8% | 15 | 40.0% | 0.477 | 9.4% | 93.8% | 12.1 | False | clv_beat_short_0.100, needs_31_rows |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 203 | 6 | -17.2% | 148 | 54.1% | 1.027 | -5.5% | 95.5% | 12.2 | False | roi_below_gate |
| watch_only | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 42 | 5 | 10.3% | 37 | 37.8% | -0.640 | 9.1% | 100.0% | 13.0 | False | avg_clv_below_gate, clv_beat_short_0.122, needs_8_rows |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 120 | 5 | -18.4% | 105 | 51.4% | -0.204 | -7.3% | 99.1% | 13.4 | False | avg_clv_below_gate, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 245 | 6 | -4.2% | 208 | 36.5% | 0.113 | 0.2% | 96.3% | 13.5 | False | clv_beat_short_0.135 |
| watch_only | batter_hits|under|common|H 0.5|plus_100_149|draftkings | 3191 | 6 | -4.5% | 2916 | 36.0% | -0.221 | 1.9% | 96.8% | 14.1 | False | avg_clv_below_gate, clv_beat_short_0.140 |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 111 | 4 | -11.4% | 100 | 41.0% | 0.041 | -1.7% | 96.2% | 15.4 | False | clv_beat_short_0.090, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 63 | 5 | -12.3% | 56 | 41.1% | 0.365 | -3.9% | 96.6% | 16.2 | False | clv_beat_short_0.089, roi_below_gate |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 112 | 6 | -2.9% | 86 | 33.7% | -0.499 | 3.1% | 95.6% | 16.5 | False | avg_clv_below_gate, clv_beat_short_0.163 |
| watch_only | batter_hits|over|common|H 0.5|plus_100_149|draftkings | 67 | 5 | -21.6% | 58 | 53.4% | 1.015 | -8.3% | 98.3% | 16.6 | False | roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 45 | 5 | 17.8% | 42 | 38.1% | -0.321 | 14.3% | 97.7% | 16.6 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.119, needs_5_rows |
| watch_only | batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1038 | 6 | -2.6% | 806 | 33.1% | -0.390 | 2.9% | 96.0% | 17.0 | False | avg_clv_below_gate, clv_beat_short_0.169 |
| watch_only | batter_hits|under|common|H 0.5|lay_130_149|draftkings | 71 | 4 | 6.9% | 60 | 31.7% | -1.006 | 9.0% | 96.8% | 18.8 | False | avg_clv_below_gate, clv_beat_short_0.183 |
| watch_only | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 154 | 6 | 2.2% | 92 | 30.4% | -1.436 | 5.2% | 93.9% | 20.2 | False | avg_clv_below_gate, clv_beat_short_0.196 |
| watch_only | batter_hits|under|common|H 0.5|fair_lay|draftkings | 674 | 6 | -6.4% | 615 | 30.2% | -0.711 | 2.4% | 97.3% | 21.5 | False | avg_clv_below_gate, clv_beat_short_0.198, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 100 | 6 | 4.7% | 62 | 27.4% | -1.160 | 7.0% | 93.9% | 23.1 | False | avg_clv_below_gate, clv_beat_short_0.226 |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 93 | 5 | -7.4% | 85 | 28.2% | -0.660 | -2.2% | 96.6% | 24.5 | False | avg_clv_below_gate, clv_beat_short_0.218, roi_below_gate |
| watch_only | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 79 | 5 | 11.1% | 54 | 25.9% | -0.745 | 10.9% | 96.4% | 25.3 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.241 |
| watch_only | pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 34 | 1 | 22.2% | 31 | 45.2% | 0.188 | 14.9% | 100.0% | 26.8 | False | calibration_below_gate, clv_beat_short_0.048, needs_16_rows, needs_2_clean_dates |
| micro_test | pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 129 | 5 | -1.1% | 92 | 52.2% | 0.838 | 2.2% | 98.9% | 1.1 | False | roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K <4.5|lay_130_149|draftkings | 45 | 4 | -1.3% | 42 | 50.0% | -0.062 | 4.6% | 100.0% | 5.4 | False | avg_clv_below_gate, clv_beat_short_0.020, needs_30_rows, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 286 | 6 | -5.7% | 209 | 50.7% | 0.729 | 0.3% | 96.8% | 7.0 | False | clv_beat_short_0.013, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 87 | 5 | -8.1% | 67 | 68.7% | 1.197 | -1.6% | 98.5% | 8.1 | False | roi_below_gate |
| micro_test | batter_hits|over|common|H 0.5|fair_lay|draftkings | 602 | 6 | -7.3% | 547 | 49.2% | 0.729 | -0.4% | 97.2% | 10.1 | False | clv_beat_short_0.028, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 142 | 5 | 1.8% | 134 | 41.8% | -0.294 | 2.5% | 95.7% | 10.4 | False | avg_clv_below_gate, clv_beat_short_0.102 |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 95 | 5 | 4.8% | 79 | 40.5% | -0.993 | 5.1% | 97.5% | 12.0 | False | avg_clv_below_gate, clv_beat_short_0.115 |
| micro_test | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2638 | 6 | -3.2% | 2200 | 43.2% | 0.125 | 1.5% | 96.5% | 12.0 | False | clv_beat_short_0.088, roi_below_gate |
| micro_test | batter_hits|over|common|H 1.5|plus_150_249|draftkings | 982 | 6 | -13.3% | 755 | 53.8% | 0.456 | -2.5% | 96.3% | 13.3 | False | roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 110 | 5 | -11.5% | 92 | 48.9% | 0.603 | -2.5% | 98.9% | 14.6 | False | clv_beat_short_0.031, roi_below_gate |
| micro_test | batter_hits|over|common|H 0.5|lay_130_149|draftkings | 762 | 6 | -8.3% | 684 | 45.6% | 0.303 | -1.1% | 97.0% | 14.7 | False | clv_beat_short_0.064, roi_below_gate |
| micro_test | batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1580 | 6 | -8.7% | 1437 | 44.9% | 0.237 | -1.7% | 96.2% | 15.8 | False | clv_beat_short_0.071, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 138 | 5 | -13.9% | 129 | 49.6% | 0.193 | -3.6% | 94.9% | 16.3 | False | clv_beat_short_0.024, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 188 | 5 | -12.3% | 151 | 47.7% | 0.708 | -3.0% | 95.6% | 16.6 | False | clv_beat_short_0.043, roi_below_gate |
| micro_test | batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3578 | 6 | -7.4% | 3080 | 42.5% | -0.039 | -1.3% | 96.6% | 16.8 | False | avg_clv_below_gate, clv_beat_short_0.095, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 203 | 6 | -17.2% | 148 | 54.1% | 1.027 | -5.5% | 95.5% | 17.2 | False | roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 133 | 5 | -7.3% | 130 | 41.5% | -0.033 | -2.2% | 99.2% | 17.8 | False | avg_clv_below_gate, clv_beat_short_0.105, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 42 | 5 | 10.3% | 37 | 37.8% | -0.640 | 9.1% | 100.0% | 17.8 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.142, needs_33_rows |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 125 | 5 | -5.2% | 120 | 39.2% | 0.100 | 0.4% | 98.4% | 18.0 | False | clv_beat_short_0.128, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 39 | 2 | 5.6% | 35 | 54.3% | 0.593 | 6.7% | 97.2% | 18.4 | False | needs_2_clean_dates, needs_36_rows |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 120 | 5 | -18.4% | 105 | 51.4% | -0.204 | -7.3% | 99.1% | 19.0 | False | avg_clv_below_gate, clv_beat_short_0.006, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 245 | 6 | -4.2% | 208 | 36.5% | 0.113 | 0.2% | 96.3% | 19.7 | False | clv_beat_short_0.155, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 24 | 2 | 7.5% | 15 | 60.0% | 0.129 | 7.6% | 100.0% | 20.4 | False | needs_2_clean_dates, needs_51_rows, needs_5_clv_rows |
| micro_test | batter_hits|under|common|H 0.5|plus_100_149|draftkings | 3191 | 6 | -4.5% | 2916 | 36.0% | -0.221 | 1.9% | 96.8% | 20.6 | False | avg_clv_below_gate, clv_beat_short_0.160, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 112 | 6 | -2.9% | 86 | 33.7% | -0.499 | 3.1% | 95.6% | 21.5 | False | avg_clv_below_gate, clv_beat_short_0.183, roi_below_gate |
| micro_test | batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1038 | 6 | -2.6% | 806 | 33.1% | -0.390 | 2.9% | 96.0% | 21.7 | False | avg_clv_below_gate, clv_beat_short_0.189, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 19 | 2 | 9.2% | 16 | 81.2% | 1.341 | 8.4% | 88.9% | 22.1 | False | calibration_below_gate, needs_2_clean_dates, needs_4_clv_rows, needs_56_rows, valid_close_coverage_below_gate |
| micro_test | batter_hits|under|common|H 0.5|lay_130_149|draftkings | 71 | 4 | 6.9% | 60 | 31.7% | -1.006 | 9.0% | 96.8% | 22.1 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.203, needs_4_rows |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 154 | 6 | 2.2% | 92 | 30.4% | -1.436 | 5.2% | 93.9% | 22.3 | False | avg_clv_below_gate, clv_beat_short_0.216 |
| micro_test | pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 45 | 5 | 17.8% | 42 | 38.1% | -0.321 | 14.3% | 97.7% | 22.4 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.139, needs_30_rows |
| micro_test | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 111 | 4 | -11.4% | 100 | 41.0% | 0.041 | -1.7% | 96.2% | 22.4 | False | clv_beat_short_0.110, roi_below_gate |
| micro_test | batter_hits|over|common|H 0.5|plus_100_149|draftkings | 67 | 5 | -21.6% | 58 | 53.4% | 1.015 | -8.3% | 98.3% | 22.4 | False | calibration_below_gate, needs_8_rows, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 63 | 5 | -12.3% | 56 | 41.1% | 0.365 | -3.9% | 96.6% | 24.0 | False | clv_beat_short_0.109, needs_12_rows, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|fanduel | 34 | 3 | -13.6% | 32 | 56.2% | 0.218 | -4.7% | 97.0% | 24.3 | False | needs_1_clean_dates, needs_41_rows, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 100 | 6 | 4.7% | 62 | 27.4% | -1.160 | 7.0% | 93.9% | 25.2 | False | avg_clv_below_gate, clv_beat_short_0.246 |
| micro_test | pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 19 | 3 | 9.8% | 15 | 40.0% | 0.477 | 9.4% | 93.8% | 26.1 | False | calibration_below_gate, clv_beat_short_0.120, needs_1_clean_dates, needs_56_rows, needs_5_clv_rows |
| micro_test | batter_hits|under|common|H 0.5|fair_lay|draftkings | 674 | 6 | -6.4% | 615 | 30.2% | -0.711 | 2.4% | 97.3% | 28.6 | False | avg_clv_below_gate, clv_beat_short_0.218, roi_below_gate |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 79 | 5 | 11.1% | 54 | 25.9% | -0.745 | 10.9% | 96.4% | 29.4 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.261 |
| micro_test | pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 93 | 5 | -7.4% | 85 | 28.2% | -0.660 | -2.2% | 96.6% | 31.5 | False | avg_clv_below_gate, clv_beat_short_0.238, roi_below_gate |
| micro_test | pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 34 | 1 | 22.2% | 31 | 45.2% | 0.188 | 14.9% | 100.0% | 40.4 | False | calibration_below_gate, clv_beat_short_0.068, needs_3_clean_dates, needs_41_rows |
| strict_micro | pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 129 | 5 | -1.1% | 92 | 52.2% | 0.838 | 2.2% | 98.9% | 5.4 | False | clv_beat_short_0.028, needs_21_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 286 | 6 | -5.7% | 209 | 50.7% | 0.729 | 0.3% | 96.8% | 10.0 | False | clv_beat_short_0.043, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 87 | 5 | -8.1% | 67 | 68.7% | 1.197 | -1.6% | 98.5% | 12.3 | False | needs_63_rows, roi_below_gate |
| strict_micro | batter_hits|over|common|H 0.5|fair_lay|draftkings | 602 | 6 | -7.3% | 547 | 49.2% | 0.729 | -0.4% | 97.2% | 13.1 | False | clv_beat_short_0.058, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 142 | 5 | 1.8% | 134 | 41.8% | -0.294 | 2.5% | 95.7% | 13.9 | False | avg_clv_below_gate, clv_beat_short_0.132, needs_8_rows |
| strict_micro | batter_hits|over|common|H 1.5|plus_150_249|draftkings | 982 | 6 | -13.3% | 755 | 53.8% | 0.456 | -2.5% | 96.3% | 14.5 | False | clv_beat_short_0.012, roi_below_gate |
| strict_micro | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2638 | 6 | -3.2% | 2200 | 43.2% | 0.125 | 1.5% | 96.5% | 15.0 | False | clv_beat_short_0.118, roi_below_gate |
| strict_micro | batter_hits|over|common|H 0.5|lay_130_149|draftkings | 762 | 6 | -8.3% | 684 | 45.6% | 0.303 | -1.1% | 97.0% | 17.7 | False | clv_beat_short_0.094, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 203 | 6 | -17.2% | 148 | 54.1% | 1.027 | -5.5% | 95.5% | 18.7 | False | calibration_below_gate, clv_beat_short_0.009, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 95 | 5 | 4.8% | 79 | 40.5% | -0.993 | 5.1% | 97.5% | 18.8 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.145, needs_55_rows |
| strict_micro | batter_hits|over|common|H 0.5|lay_150_180|draftkings | 1580 | 6 | -8.7% | 1437 | 44.9% | 0.237 | -1.7% | 96.2% | 18.8 | False | clv_beat_short_0.101, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 188 | 5 | -12.3% | 151 | 47.7% | 0.708 | -3.0% | 95.6% | 19.6 | False | clv_beat_short_0.073, roi_below_gate |
| strict_micro | batter_hits|over|common|H 0.5|heavy_lay|draftkings | 3578 | 6 | -7.4% | 3080 | 42.5% | -0.039 | -1.3% | 96.6% | 19.8 | False | avg_clv_below_gate, clv_beat_short_0.125, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 138 | 5 | -13.9% | 129 | 49.6% | 0.193 | -3.6% | 94.9% | 20.1 | False | clv_beat_short_0.054, needs_12_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 110 | 5 | -11.5% | 92 | 48.9% | 0.603 | -2.5% | 98.9% | 20.3 | False | clv_beat_short_0.061, needs_40_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K <4.5|lay_130_149|draftkings | 45 | 4 | -1.3% | 42 | 50.0% | -0.062 | 4.6% | 100.0% | 21.4 | False | avg_clv_below_gate, clv_beat_short_0.050, needs_105_rows, needs_1_clean_dates, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 133 | 5 | -7.3% | 130 | 41.5% | -0.033 | -2.2% | 99.2% | 21.9 | False | avg_clv_below_gate, clv_beat_short_0.135, needs_17_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 125 | 5 | -5.2% | 120 | 39.2% | 0.100 | 0.4% | 98.4% | 22.7 | False | clv_beat_short_0.158, needs_25_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 245 | 6 | -4.2% | 208 | 36.5% | 0.113 | 0.2% | 96.3% | 22.7 | False | clv_beat_short_0.185, roi_below_gate |
| strict_micro | batter_hits|under|common|H 0.5|plus_100_149|draftkings | 3191 | 6 | -4.5% | 2916 | 36.0% | -0.221 | 1.9% | 96.8% | 23.6 | False | avg_clv_below_gate, clv_beat_short_0.190, roi_below_gate |
| strict_micro | batter_hits|under|common|H 1.5|heavy_lay|draftkings | 1038 | 6 | -2.6% | 806 | 33.1% | -0.390 | 2.9% | 96.0% | 24.7 | False | avg_clv_below_gate, clv_beat_short_0.219, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 154 | 6 | 2.2% | 92 | 30.4% | -1.436 | 5.2% | 93.9% | 25.4 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.246 |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 120 | 5 | -18.4% | 105 | 51.4% | -0.204 | -7.3% | 99.1% | 26.3 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.036, needs_30_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 112 | 6 | -2.9% | 86 | 33.7% | -0.499 | 3.1% | 95.6% | 27.0 | False | avg_clv_below_gate, clv_beat_short_0.213, needs_38_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 42 | 5 | 10.3% | 37 | 37.8% | -0.640 | 9.1% | 100.0% | 28.8 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.172, needs_108_rows |
| strict_micro | batter_hits|under|common|H 0.5|fair_lay|draftkings | 674 | 6 | -6.4% | 615 | 30.2% | -0.711 | 2.4% | 97.3% | 31.6 | False | avg_clv_below_gate, clv_beat_short_0.248, roi_below_gate |
| strict_micro | batter_hits|over|common|H 0.5|plus_100_149|draftkings | 67 | 5 | -21.6% | 58 | 53.4% | 1.015 | -8.3% | 98.3% | 32.0 | False | calibration_below_gate, clv_beat_short_0.016, needs_83_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 63 | 5 | -12.3% | 56 | 41.1% | 0.365 | -3.9% | 96.6% | 32.0 | False | clv_beat_short_0.139, needs_87_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 45 | 5 | 17.8% | 42 | 38.1% | -0.321 | 14.3% | 97.7% | 33.4 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.169, needs_105_rows |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 100 | 6 | 4.7% | 62 | 27.4% | -1.160 | 7.0% | 93.9% | 33.5 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.276, needs_50_rows |
| strict_micro | pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 39 | 2 | 5.6% | 35 | 54.3% | 0.593 | 6.7% | 97.2% | 33.8 | False | calibration_below_gate, clv_beat_short_0.007, needs_111_rows, needs_3_clean_dates |
| strict_micro | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 111 | 4 | -11.4% | 100 | 41.0% | 0.041 | -1.7% | 96.2% | 36.0 | False | clv_beat_short_0.140, needs_1_clean_dates, needs_39_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|fanduel | 34 | 3 | -13.6% | 32 | 56.2% | 0.218 | -4.7% | 97.0% | 37.3 | False | needs_116_rows, needs_2_clean_dates, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 24 | 2 | 7.5% | 15 | 60.0% | 0.129 | 7.6% | 100.0% | 38.0 | False | calibration_below_gate, needs_126_rows, needs_15_clv_rows, needs_3_clean_dates |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 93 | 5 | -7.4% | 85 | 28.2% | -0.660 | -2.2% | 96.6% | 38.3 | False | avg_clv_below_gate, clv_beat_short_0.268, needs_57_rows, roi_below_gate |
| strict_micro | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 79 | 5 | 11.1% | 54 | 25.9% | -0.745 | 10.9% | 96.4% | 40.1 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.291, needs_71_rows |
| strict_micro | pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 19 | 2 | 9.2% | 16 | 81.2% | 1.341 | 8.4% | 88.9% | 40.1 | False | calibration_below_gate, needs_131_rows, needs_14_clv_rows, needs_3_clean_dates, valid_close_coverage_below_gate |
| strict_micro | batter_hits|under|common|H 0.5|lay_130_149|draftkings | 71 | 4 | 6.9% | 60 | 31.7% | -1.006 | 9.0% | 96.8% | 41.1 | False | avg_clv_below_gate, calibration_below_gate, clv_beat_short_0.233, needs_1_clean_dates, needs_79_rows |
| strict_micro | pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 19 | 3 | 9.8% | 15 | 40.0% | 0.477 | 9.4% | 93.8% | 47.1 | False | calibration_below_gate, clv_beat_short_0.150, needs_131_rows, needs_15_clv_rows, needs_2_clean_dates |
| strict_micro | pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 34 | 1 | 22.2% | 31 | 45.2% | 0.188 | 14.9% | 100.0% | 59.4 | False | calibration_below_gate, clv_beat_short_0.098, needs_116_rows, needs_4_clean_dates |
