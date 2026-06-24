# MLB Prop Bookability Model

Generated UTC: 2026-06-24T08:15:09Z
Rows: 118135
Date range: 2026-05-31 to 2026-06-23
Status: ready

## Close Capture Holdout

| Metric | Value |
|---|---:|
| rows | 4119.000 |
| actual_bookable_rate | 78.2% |
| avg_pred_bookable | 0.841 |
| brier_baseline | 0.182 |
| brier_model | 0.183 |
| log_loss_model | 0.588 |
| auc_model | 0.590 |
| model_usable | no |
| selected_scoring_method | empirical_bucket_rate |

## Line Availability Holdout

| Metric | Value |
|---|---:|
| rows | 3301.000 |
| actual_bookable_rate | 97.5% |
| avg_pred_bookable | 0.994 |
| brier_baseline | 0.024 |
| brier_model | 0.023 |
| log_loss_model | 0.162 |
| auc_model | 0.720 |
| model_usable | yes |
| selected_scoring_method | logistic |

## Close Capture Calibration

| Predicted Bucket | Rows | Actual Bookable | Avg Predicted | Error |
|---|---:|---:|---:|---:|
| 10-20% | 8 | 50.0% | 16.5% | +33.5% |
| 20-30% | 49 | 73.5% | 26.4% | +47.1% |
| 30-40% | 90 | 74.4% | 35.3% | +39.2% |
| 40-50% | 111 | 69.4% | 45.2% | +24.1% |
| 50-60% | 135 | 56.3% | 55.3% | +1.0% |
| 60-70% | 199 | 63.3% | 65.5% | -2.2% |
| 70-80% | 402 | 71.9% | 75.8% | -3.9% |
| 80-90% | 1089 | 79.7% | 85.9% | -6.2% |
| 90-100% | 2036 | 82.3% | 94.5% | -12.2% |

## Line Availability Calibration

| Predicted Bucket | Rows | Actual Available | Avg Predicted | Error |
|---|---:|---:|---:|---:|
| 40-50% | 2 | 100.0% | 47.7% | +52.3% |
| 50-60% | 1 | 0.0% | 52.6% | -52.6% |
| 60-70% | 2 | 100.0% | 67.8% | +32.2% |
| 70-80% | 9 | 66.7% | 76.9% | -10.3% |
| 80-90% | 14 | 57.1% | 86.7% | -29.5% |
| 90-100% | 3273 | 97.8% | 99.6% | -1.8% |

## Close Capture Prediction Gap Audit

| Level | Bucket | Rows | Actual | Predicted | Error | Note |
|---|---|---:|---:|---:|---:|---|
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 10 | 40.0% | 71.9% | -31.9% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | 48 | 60.4% | 87.2% | -26.8% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 20 | 70.0% | 95.6% | -25.6% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|fanduel | 16 | 68.8% | 90.6% | -21.9% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | 53 | 62.3% | 83.9% | -21.7% | model_too_optimistic |
| surface | batter_hits|over|alt_tail | 186 | 100.0% | 81.1% | +18.9% | model_too_pessimistic |
| book_surface | batter_hits|over|alt_tail|fanduel | 186 | 100.0% | 81.1% | +18.9% | model_too_pessimistic |
| exact_bucket | batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 186 | 100.0% | 81.1% | +18.9% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 29 | 82.8% | 65.6% | +17.2% | model_too_pessimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 44 | 56.8% | 72.9% | -16.1% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 189 | 76.2% | 92.1% | -15.9% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 122 | 70.5% | 86.1% | -15.6% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|fanduel | 48 | 72.9% | 88.2% | -15.3% | model_too_optimistic |
| book_surface | batter_total_bases|over|common|draftkings | 156 | 73.7% | 86.7% | -13.0% | model_too_optimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 56 | 69.6% | 81.5% | -11.9% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|fanduel | 72 | 77.8% | 89.7% | -11.9% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|draftkings | 32 | 65.6% | 77.3% | -11.7% | model_too_optimistic |
| surface | batter_total_bases|over|alt_tail | 690 | 75.1% | 85.8% | -10.8% | model_too_optimistic |
| book_surface | batter_total_bases|over|alt_tail|fanduel | 690 | 75.1% | 85.8% | -10.8% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 162 | 79.0% | 89.6% | -10.6% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 210 | 75.2% | 85.4% | -10.2% | model_too_optimistic |
| market_side | batter_total_bases|under | 156 | 73.7% | 82.7% | -9.0% | model_too_optimistic |
| surface | batter_total_bases|under|common | 156 | 73.7% | 82.7% | -9.0% | model_too_optimistic |
| book_surface | batter_total_bases|under|common|draftkings | 156 | 73.7% | 82.7% | -9.0% | model_too_optimistic |
| market_side | batter_total_bases|over | 1536 | 76.9% | 85.8% | -9.0% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 481 | 74.8% | 83.0% | -8.1% | model_too_optimistic |
| book_surface | pitcher_strikeouts|over|common|fanduel | 36 | 80.6% | 88.6% | -8.1% | model_too_optimistic |
| surface | batter_total_bases|over|common | 846 | 78.4% | 85.8% | -7.5% | model_too_optimistic |
| book_surface | batter_hits|over|common|fanduel | 690 | 79.4% | 86.4% | -7.0% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|fanduel | 209 | 82.3% | 89.3% | -7.0% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 29 | 89.7% | 96.6% | -6.9% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 90 | 85.6% | 92.3% | -6.7% | model_too_optimistic |
| book_surface | pitcher_strikeouts|under|common|fanduel | 36 | 80.6% | 87.2% | -6.7% | model_too_optimistic |
| surface | batter_hits|over|common | 1045 | 77.6% | 84.3% | -6.7% | model_too_optimistic |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 113 | 85.0% | 91.6% | -6.6% | model_too_optimistic |
| book_surface | batter_total_bases|over|common|fanduel | 690 | 79.4% | 85.6% | -6.2% | model_too_optimistic |
| book_surface | batter_hits|over|common|draftkings | 355 | 74.1% | 80.1% | -6.0% | model_too_optimistic |
| surface | batter_home_runs|over|alt_tail | 344 | 79.4% | 84.9% | -5.5% | model_too_optimistic |
| book_surface | batter_home_runs|over|alt_tail|fanduel | 344 | 79.4% | 84.9% | -5.5% | model_too_optimistic |
| exact_bucket | batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 344 | 79.4% | 84.9% | -5.5% | model_too_optimistic |

## Close Reason Audit

`clv_unknown_reason` is label-only and is not used as a training feature.

| Reason | Rows | Close Captured | Line Available | Avg Predicted Capture |
|---|---:|---:|---:|---:|
| valid_close | 3219 | 100.0% | 100.0% | 85.2% |
| close_outside_two_hour_window | 569 | 0.0% | - | 86.3% |
| stale_close_before_lock | 223 | 0.0% | - | 68.0% |
| line_disappeared_at_close | 82 | 0.0% | 0.0% | 76.0% |
| fallback_other_book_only | 16 | 0.0% | - | 78.5% |
| no_valid_close_snapshot | 10 | 0.0% | - | 49.8% |

## Least Bookable Buckets

| Bucket | Rows | Close Captured | Line Available | Avail Rows | Stale | No Valid Close |
|---|---:|---:|---:|---:|---:|---:|
| batter_hits|over|common|H 0.5|plus_150_249|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_500_plus|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | 0.0% | 1 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | 0.0% | 1 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 7 | 42.9% | 60.0% | 5 | 28.6% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 33 | 45.5% | 45.5% | 33 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 16 | 50.0% | 50.0% | 16 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 4 | 50.0% | 50.0% | 4 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_130_149|draftkings | 4 | 50.0% | 50.0% | 4 | 0.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 189 | 58.2% | 100.0% | 110 | 7.9% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|draftkings | 24 | 58.3% | 70.0% | 20 | 0.0% | 8.3% |
| batter_total_bases|under|common|TB 2.5+|lay_150_180|draftkings | 15 | 66.7% | 76.9% | 13 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 157 | 68.2% | 71.3% | 150 | 2.5% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_150_180|draftkings | 19 | 68.4% | 68.4% | 19 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 94 | 69.1% | 69.1% | 94 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 96 | 70.8% | 73.9% | 92 | 4.2% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 76 | 71.1% | 77.1% | 70 | 7.9% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|fanduel | 25 | 72.0% | 78.3% | 23 | 8.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 33 | 72.7% | 77.4% | 31 | 6.1% | 0.0% |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 1230 | 73.7% | 100.0% | 907 | 12.7% | 1.9% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 28 | 75.0% | 84.0% | 25 | 10.7% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 332 | 75.6% | 79.2% | 317 | 3.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 135 | 76.3% | 79.8% | 129 | 4.4% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 148 | 76.4% | 100.0% | 113 | 14.9% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 216 | 76.4% | 78.9% | 209 | 0.9% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|draftkings | 51 | 76.5% | 86.7% | 45 | 3.9% | 3.9% |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 120 | 79.2% | 84.8% | 112 | 6.7% | 0.0% |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 586 | 79.7% | 100.0% | 467 | 10.9% | 0.2% |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 382 | 80.1% | 100.0% | 306 | 11.8% | 0.0% |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 758 | 80.2% | 100.0% | 608 | 12.3% | 0.0% |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 92 | 80.4% | 100.0% | 74 | 8.7% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 774 | 81.0% | 100.0% | 627 | 9.8% | 0.5% |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 291 | 81.1% | 85.8% | 275 | 3.8% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|fanduel | 16 | 81.2% | 86.7% | 15 | 6.2% | 0.0% |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 888 | 81.6% | 100.0% | 725 | 9.9% | 0.5% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 105 | 81.9% | 87.8% | 98 | 1.9% | 0.0% |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 878 | 82.6% | 100.0% | 725 | 9.1% | 0.7% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 155 | 82.6% | 87.1% | 147 | 3.9% | 0.0% |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 1239 | 83.3% | 87.9% | 1174 | 3.1% | 0.0% |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 90 | 83.3% | 100.0% | 75 | 7.8% | 0.0% |
