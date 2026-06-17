# MLB Prop Bookability Model

Generated UTC: 2026-06-13T20:03:30Z
Rows: 54313
Date range: 2026-05-31 to 2026-06-13
Status: ready

## Close Capture Holdout

| Metric | Value |
|---|---:|
| rows | 742.000 |
| actual_bookable_rate | 85.7% |
| avg_pred_bookable | 0.938 |
| brier_baseline | 0.125 |
| brier_model | 0.128 |
| log_loss_model | 0.546 |
| auc_model | 0.464 |
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
| 20-30% | 1 | 0.0% | 29.8% | -29.8% |
| 30-40% | 2 | 0.0% | 37.5% | -37.5% |
| 40-50% | 15 | 46.7% | 47.1% | -0.4% |
| 50-60% | 22 | 59.1% | 54.8% | +4.2% |
| 60-70% | 14 | 100.0% | 65.7% | +34.3% |
| 70-80% | 8 | 100.0% | 73.8% | +26.2% |
| 80-90% | 7 | 100.0% | 88.5% | +11.5% |
| 90-100% | 673 | 87.2% | 97.2% | -10.0% |

## Close Capture Prediction Gap Audit

| Level | Bucket | Rows | Actual | Predicted | Error | Note |
|---|---|---:|---:|---:|---:|---|
| exact_bucket | batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 13 | 53.8% | 94.3% | -40.4% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 32 | 68.8% | 97.3% | -28.5% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|draftkings | 36 | 72.2% | 96.3% | -24.1% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 27 | 74.1% | 98.0% | -23.9% | model_too_optimistic |
| book_surface | batter_total_bases|over|common|draftkings | 32 | 75.0% | 97.9% | -22.9% | model_too_optimistic |
| market_side | batter_total_bases|under | 32 | 75.0% | 96.2% | -21.2% | model_too_optimistic |
| surface | batter_total_bases|under|common | 32 | 75.0% | 96.2% | -21.2% | model_too_optimistic |
| book_surface | batter_total_bases|under|common|draftkings | 32 | 75.0% | 96.2% | -21.2% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|fanduel | 21 | 81.0% | 98.5% | -17.6% | model_too_optimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 10 | 80.0% | 96.8% | -16.8% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 34 | 82.4% | 96.6% | -14.3% | model_too_optimistic |
| book_surface | batter_hits|over|common|draftkings | 66 | 80.3% | 92.9% | -12.6% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|fanduel | 38 | 84.2% | 96.5% | -12.2% | model_too_optimistic |
| market_side | batter_hits|under | 66 | 80.3% | 92.4% | -12.1% | model_too_optimistic |
| surface | batter_hits|under|common | 66 | 80.3% | 92.4% | -12.1% | model_too_optimistic |
| book_surface | batter_hits|under|common|draftkings | 66 | 80.3% | 92.4% | -12.1% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 32 | 84.4% | 95.3% | -10.9% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 16 | 87.5% | 96.6% | -9.1% | model_too_optimistic |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 24 | 87.5% | 96.5% | -9.0% | model_too_optimistic |
| market_side | batter_hits|over | 194 | 85.1% | 93.8% | -8.7% | model_too_optimistic |
| surface | batter_hits|over|common | 194 | 85.1% | 93.8% | -8.7% | model_too_optimistic |
| surface | batter_total_bases|over|common | 160 | 85.0% | 93.6% | -8.6% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 41 | 85.4% | 94.0% | -8.6% | model_too_optimistic |
| market_side | batter_total_bases|over | 288 | 86.1% | 93.7% | -7.6% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 24 | 91.7% | 84.9% | +6.8% | model_too_pessimistic |
| book_surface | batter_hits|over|common|fanduel | 128 | 87.5% | 94.2% | -6.7% | model_too_optimistic |
| surface | batter_home_runs|over|alt_tail | 64 | 87.5% | 94.1% | -6.6% | model_too_optimistic |
| book_surface | batter_home_runs|over|alt_tail|fanduel | 64 | 87.5% | 94.1% | -6.6% | model_too_optimistic |
| exact_bucket | batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 64 | 87.5% | 94.1% | -6.6% | model_too_optimistic |
| surface | batter_total_bases|over|alt_tail | 128 | 87.5% | 93.7% | -6.2% | model_too_optimistic |
| book_surface | batter_total_bases|over|alt_tail|fanduel | 128 | 87.5% | 93.7% | -6.2% | model_too_optimistic |
| market_side | batter_home_runs|over | 128 | 87.5% | 93.7% | -6.2% | model_too_optimistic |
| surface | batter_home_runs|over|common | 64 | 87.5% | 93.3% | -5.8% | model_too_optimistic |
| book_surface | batter_home_runs|over|common|fanduel | 64 | 87.5% | 93.3% | -5.8% | model_too_optimistic |
| market_side | pitcher_strikeouts|under | 17 | 100.0% | 94.8% | +5.2% | model_too_pessimistic |
| surface | pitcher_strikeouts|under|common | 17 | 100.0% | 94.8% | +5.2% | model_too_pessimistic |
| book_surface | batter_total_bases|over|common|fanduel | 128 | 87.5% | 92.5% | -5.0% | model_too_optimistic |
| market_side | pitcher_strikeouts|over | 17 | 100.0% | 95.2% | +4.8% | calibrated |
| surface | pitcher_strikeouts|over|common | 17 | 100.0% | 95.2% | +4.8% | calibrated |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|draftkings | 13 | 84.6% | 88.6% | -4.0% | calibrated |

## Close Reason Audit

`clv_unknown_reason` is label-only and is not used as a training feature.

| Reason | Rows | Close Captured | Line Available | Avg Predicted Capture |
|---|---:|---:|---:|---:|
| valid_close | 636 | 100.0% | 100.0% | 94.6% |
| close_outside_two_hour_window | 98 | 0.0% | - | 88.0% |
| fallback_other_book_only | 7 | 0.0% | - | 97.0% |
| stale_close_before_lock | 1 | 0.0% | - | 97.8% |

## Least Bookable Buckets

| Bucket | Rows | Close Captured | Line Available | Avail Rows | Stale | No Valid Close |
|---|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 4 | 0.0% | - | 0 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 4 | 0.0% | - | 0 | 75.0% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | - | 0 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_130_149|draftkings | 2 | 0.0% | - | 0 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|under|common|TB 2.5+|lay_150_180|draftkings | 6 | 33.3% | 100.0% | 2 | 0.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 87 | 44.8% | 100.0% | 39 | 13.8% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 42 | 50.0% | 100.0% | 21 | 16.7% | 16.7% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 10 | 50.0% | 100.0% | 5 | 20.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 11 | 54.5% | 100.0% | 6 | 36.4% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|draftkings | 11 | 54.5% | 100.0% | 6 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 7 | 57.1% | 100.0% | 4 | 28.6% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 24 | 58.3% | 100.0% | 14 | 25.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 57 | 59.6% | 100.0% | 34 | 21.1% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 37 | 62.2% | 100.0% | 23 | 18.9% | 0.0% |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 561 | 62.4% | 100.0% | 350 | 14.3% | 1.4% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 70 | 62.9% | 100.0% | 44 | 11.4% | 0.0% |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 25 | 64.0% | 100.0% | 16 | 8.0% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 140 | 64.3% | 100.0% | 90 | 17.9% | 0.0% |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 219 | 65.3% | 100.0% | 143 | 16.9% | 0.0% |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 55 | 65.5% | 100.0% | 36 | 5.5% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 99 | 65.7% | 100.0% | 65 | 9.1% | 7.1% |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 6 | 66.7% | 100.0% | 4 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 6 | 66.7% | 100.0% | 4 | 33.3% | 0.0% |
| batter_hits|under|common|H 0.5|lay_150_180|draftkings | 3 | 66.7% | 100.0% | 2 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 66.7% | 100.0% | 2 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 3 | 66.7% | 100.0% | 2 | 33.3% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 3 | 66.7% | 100.0% | 2 | 33.3% | 0.0% |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 305 | 68.2% | 100.0% | 208 | 17.0% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|draftkings | 19 | 68.4% | 100.0% | 13 | 5.3% | 0.0% |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 29 | 69.0% | 100.0% | 20 | 10.3% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_150_180|draftkings | 10 | 70.0% | 100.0% | 7 | 0.0% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 289 | 70.2% | 100.0% | 203 | 15.9% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 74 | 70.3% | 100.0% | 52 | 12.2% | 4.1% |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 37 | 70.3% | 100.0% | 26 | 16.2% | 0.0% |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 349 | 70.8% | 100.0% | 247 | 15.2% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 147 | 72.1% | 100.0% | 106 | 10.2% | 1.4% |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 61 | 72.1% | 100.0% | 44 | 3.3% | 0.0% |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 401 | 73.1% | 100.0% | 293 | 14.2% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 26 | 73.1% | 100.0% | 19 | 15.4% | 0.0% |
