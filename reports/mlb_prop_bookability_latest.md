# MLB Prop Bookability Model

Generated UTC: 2026-06-17T08:51:32Z
Rows: 80343
Date range: 2026-05-31 to 2026-06-16
Status: ready

## Close Capture Holdout

| Metric | Value |
|---|---:|
| rows | 8262.000 |
| actual_bookable_rate | 85.9% |
| avg_pred_bookable | 0.954 |
| brier_baseline | 0.122 |
| brier_model | 0.140 |
| log_loss_model | 0.610 |
| auc_model | 0.554 |
| model_usable | no |
| selected_scoring_method | empirical_bucket_rate |

## Line Availability Holdout

| Metric | Value |
|---|---:|
| rows | - |
| actual_bookable_rate | - |
| avg_pred_bookable | - |
| brier_baseline | - |
| brier_model | - |
| log_loss_model | - |
| auc_model | - |
| model_usable | - |
| selected_scoring_method | empirical_bucket_rate |

## Close Capture Calibration

| Predicted Bucket | Rows | Actual Bookable | Avg Predicted | Error |
|---|---:|---:|---:|---:|
| 10-20% | 7 | 0.0% | 15.9% | -15.9% |
| 20-30% | 9 | 33.3% | 24.9% | +8.5% |
| 30-40% | 18 | 83.3% | 35.4% | +47.9% |
| 40-50% | 54 | 96.3% | 46.0% | +50.3% |
| 50-60% | 146 | 96.6% | 55.8% | +40.8% |
| 60-70% | 200 | 92.0% | 65.5% | +26.5% |
| 70-80% | 200 | 81.0% | 74.9% | +6.1% |
| 80-90% | 129 | 78.3% | 84.7% | -6.4% |
| 90-100% | 7499 | 85.9% | 98.4% | -12.4% |

## Close Capture Prediction Gap Audit

| Level | Bucket | Rows | Actual | Predicted | Error | Note |
|---|---|---:|---:|---:|---:|---|
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 10 | 50.0% | 96.8% | -46.8% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 10 | 50.0% | 89.3% | -39.3% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|draftkings | 89 | 66.3% | 93.7% | -27.4% | model_too_optimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 63 | 68.3% | 92.5% | -24.3% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|fanduel | 177 | 78.5% | 95.9% | -17.4% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 435 | 80.2% | 95.8% | -15.5% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 275 | 77.1% | 92.4% | -15.3% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|fanduel | 64 | 82.8% | 97.1% | -14.3% | model_too_optimistic |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 476 | 80.9% | 94.7% | -13.8% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_250_499|fanduel | 402 | 81.6% | 94.8% | -13.2% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 981 | 82.1% | 95.1% | -13.0% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|plus_100_149|draftkings | 10 | 100.0% | 87.1% | +12.9% | model_too_pessimistic |
| book_surface | batter_hits|over|common|fanduel | 1420 | 84.8% | 96.2% | -11.4% | model_too_optimistic |
| surface | batter_total_bases|over|alt_tail | 1420 | 84.8% | 95.8% | -11.0% | model_too_optimistic |
| book_surface | batter_total_bases|over|alt_tail|fanduel | 1420 | 84.8% | 95.8% | -11.0% | model_too_optimistic |
| surface | batter_home_runs|over|common | 710 | 84.8% | 95.7% | -10.9% | model_too_optimistic |
| book_surface | batter_home_runs|over|common|fanduel | 710 | 84.8% | 95.7% | -10.9% | model_too_optimistic |
| market_side | batter_home_runs|over | 1417 | 84.8% | 95.5% | -10.8% | model_too_optimistic |
| book_surface | batter_total_bases|over|common|fanduel | 1420 | 84.8% | 95.5% | -10.7% | model_too_optimistic |
| surface | batter_home_runs|over|alt_tail | 707 | 84.7% | 95.4% | -10.7% | model_too_optimistic |
| book_surface | batter_home_runs|over|alt_tail|fanduel | 707 | 84.7% | 95.4% | -10.7% | model_too_optimistic |
| exact_bucket | batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 707 | 84.7% | 95.4% | -10.7% | model_too_optimistic |
| market_side | batter_total_bases|over | 3152 | 85.2% | 95.8% | -10.6% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|fanduel | 425 | 86.8% | 97.4% | -10.6% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 21 | 85.7% | 96.0% | -10.3% | model_too_optimistic |
| surface | batter_total_bases|over|common | 1732 | 85.5% | 95.8% | -10.3% | model_too_optimistic |
| book_surface | pitcher_strikeouts|over|common|draftkings | 86 | 86.0% | 96.2% | -10.2% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 36 | 86.1% | 96.2% | -10.1% | model_too_optimistic |
| surface | batter_hits|over|common | 2147 | 85.3% | 95.3% | -10.1% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 0.5|fair_lay|draftkings | 86 | 80.2% | 90.3% | -10.0% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 13 | 84.6% | 94.4% | -9.8% | model_too_optimistic |
| book_surface | pitcher_strikeouts|under|common|draftkings | 86 | 86.0% | 95.7% | -9.7% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 334 | 88.3% | 97.3% | -9.0% | model_too_optimistic |
| market_side | batter_hits|over | 2320 | 86.2% | 95.1% | -8.8% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 261 | 88.5% | 96.9% | -8.4% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|fanduel | 241 | 89.2% | 97.6% | -8.4% | model_too_optimistic |
| book_surface | batter_total_bases|over|common|draftkings | 312 | 88.8% | 97.0% | -8.2% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_100_149|draftkings | 322 | 84.8% | 92.9% | -8.1% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | 71 | 91.5% | 99.2% | -7.7% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 228 | 87.7% | 95.3% | -7.6% | model_too_optimistic |

## Close Reason Audit

`clv_unknown_reason` is label-only and is not used as a training feature.

| Reason | Rows | Close Captured | Line Available | Avg Predicted Capture |
|---|---:|---:|---:|---:|
| valid_close | 7101 | 100.0% | 100.0% | 95.4% |
| close_outside_two_hour_window | 848 | 0.0% | - | 96.5% |
| stale_close_before_lock | 150 | 0.0% | - | 93.2% |
| fallback_other_book_only | 109 | 0.0% | - | 98.1% |
| no_valid_close_snapshot | 54 | 0.0% | - | 74.4% |

## Least Bookable Buckets

| Bucket | Rows | Close Captured | Line Available | Avail Rows | Stale | No Valid Close |
|---|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 4 | 0.0% | - | 0 | 75.0% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | - | 0 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_130_149|draftkings | 2 | 0.0% | - | 0 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 10 | 20.0% | 100.0% | 2 | 10.0% | 20.0% |
| batter_total_bases|under|common|TB 2.5+|lay_150_180|draftkings | 7 | 42.9% | 100.0% | 3 | 0.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 128 | 46.9% | 100.0% | 60 | 10.2% | 1.6% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 59 | 49.2% | 100.0% | 29 | 11.9% | 11.9% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 17 | 52.9% | 100.0% | 9 | 17.6% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_150_180|draftkings | 13 | 53.8% | 100.0% | 7 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 23 | 60.9% | 100.0% | 14 | 13.0% | 8.7% |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 42 | 61.9% | 100.0% | 26 | 23.8% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 110 | 62.7% | 100.0% | 69 | 9.1% | 3.6% |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 838 | 63.0% | 100.0% | 528 | 11.7% | 1.8% |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 147 | 65.3% | 100.0% | 96 | 6.8% | 4.8% |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 68 | 66.2% | 100.0% | 45 | 4.4% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|draftkings | 21 | 66.7% | 100.0% | 14 | 0.0% | 9.5% |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 66.7% | 100.0% | 2 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 62 | 67.7% | 100.0% | 42 | 17.7% | 0.0% |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 477 | 70.6% | 100.0% | 337 | 13.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 60 | 71.7% | 100.0% | 43 | 11.7% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 18 | 72.2% | 100.0% | 13 | 22.2% | 0.0% |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 548 | 73.0% | 100.0% | 400 | 11.5% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 104 | 73.1% | 100.0% | 76 | 10.6% | 2.9% |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 465 | 73.1% | 100.0% | 340 | 11.4% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 216 | 73.1% | 100.0% | 158 | 12.5% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 86 | 73.3% | 100.0% | 63 | 14.0% | 0.0% |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 341 | 73.3% | 100.0% | 250 | 11.4% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 226 | 73.5% | 100.0% | 166 | 8.0% | 2.7% |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 784 | 74.6% | 100.0% | 585 | 11.0% | 0.0% |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 574 | 74.9% | 100.0% | 430 | 11.1% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2290 | 75.2% | 100.0% | 1721 | 10.4% | 0.1% |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 74 | 75.7% | 100.0% | 56 | 2.7% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 74 | 75.7% | 100.0% | 56 | 4.1% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|draftkings | 35 | 77.1% | 100.0% | 27 | 2.9% | 5.7% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 37 | 78.4% | 100.0% | 29 | 10.8% | 0.0% |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1588 | 78.7% | 100.0% | 1250 | 8.9% | 0.2% |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 841 | 78.7% | 100.0% | 662 | 3.8% | 0.0% |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 47 | 78.7% | 100.0% | 37 | 6.4% | 0.0% |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 2334 | 79.3% | 100.0% | 1852 | 7.5% | 0.9% |
