# MLB Prop Bookability Model

Generated UTC: 2026-06-18T07:13:47Z
Rows: 84050
Date range: 2026-05-31 to 2026-06-17
Status: ready

## Close Capture Holdout

| Metric | Value |
|---|---:|
| rows | 3707.000 |
| actual_bookable_rate | 89.6% |
| avg_pred_bookable | 0.444 |
| brier_baseline | 0.099 |
| brier_model | 0.310 |
| log_loss_model | 0.831 |
| auc_model | 0.647 |
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
| 00-10% | 6 | 83.3% | 8.2% | +75.2% |
| 10-20% | 231 | 80.5% | 16.3% | +64.3% |
| 20-30% | 593 | 84.5% | 25.2% | +59.2% |
| 30-40% | 738 | 86.2% | 35.2% | +51.0% |
| 40-50% | 749 | 89.7% | 44.9% | +44.8% |
| 50-60% | 669 | 94.5% | 54.7% | +39.8% |
| 60-70% | 463 | 94.0% | 64.5% | +29.5% |
| 70-80% | 194 | 97.9% | 73.9% | +24.0% |
| 80-90% | 60 | 98.3% | 83.9% | +14.4% |
| 90-100% | 4 | 100.0% | 90.4% | +9.6% |

## Close Capture Prediction Gap Audit

| Level | Bucket | Rows | Actual | Predicted | Error | Note |
|---|---|---:|---:|---:|---:|---|
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 40 | 100.0% | 36.3% | +63.7% | model_too_pessimistic |
| surface | batter_hits|over|alt_tail | 92 | 96.7% | 36.5% | +60.2% | model_too_pessimistic |
| book_surface | batter_hits|over|alt_tail|fanduel | 92 | 96.7% | 36.5% | +60.2% | model_too_pessimistic |
| exact_bucket | batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 92 | 96.7% | 36.5% | +60.2% | model_too_pessimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 60 | 90.0% | 30.6% | +59.4% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|draftkings | 149 | 87.9% | 31.0% | +57.0% | model_too_pessimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 109 | 86.2% | 31.6% | +54.7% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|draftkings | 56 | 89.3% | 35.1% | +54.2% | model_too_pessimistic |
| market_side | batter_total_bases|under | 164 | 87.8% | 34.0% | +53.8% | model_too_pessimistic |
| surface | batter_total_bases|under|common | 164 | 87.8% | 34.0% | +53.8% | model_too_pessimistic |
| book_surface | batter_total_bases|under|common|draftkings | 164 | 87.8% | 34.0% | +53.8% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|draftkings | 40 | 95.0% | 42.5% | +52.5% | model_too_pessimistic |
| market_side | batter_hits|under | 315 | 87.3% | 36.0% | +51.3% | model_too_pessimistic |
| surface | batter_hits|under|common | 315 | 87.3% | 36.0% | +51.3% | model_too_pessimistic |
| book_surface | batter_hits|under|common|draftkings | 315 | 87.3% | 36.0% | +51.3% | model_too_pessimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_100_149|draftkings | 134 | 91.8% | 40.6% | +51.2% | model_too_pessimistic |
| book_surface | batter_hits|over|common|draftkings | 315 | 87.3% | 36.2% | +51.1% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 22 | 86.4% | 35.8% | +50.6% | model_too_pessimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | 23 | 100.0% | 49.5% | +50.5% | model_too_pessimistic |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | 52 | 78.8% | 28.4% | +50.4% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|fanduel | 124 | 95.2% | 47.3% | +47.8% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 21 | 100.0% | 52.2% | +47.8% | model_too_pessimistic |
| surface | batter_home_runs|over|alt_tail | 332 | 90.1% | 43.1% | +47.0% | model_too_pessimistic |
| book_surface | batter_home_runs|over|alt_tail|fanduel | 332 | 90.1% | 43.1% | +47.0% | model_too_pessimistic |
| exact_bucket | batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 332 | 90.1% | 43.1% | +47.0% | model_too_pessimistic |
| book_surface | batter_total_bases|over|common|draftkings | 164 | 87.8% | 42.2% | +45.6% | model_too_pessimistic |
| market_side | batter_home_runs|over | 665 | 90.1% | 44.8% | +45.3% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 118 | 84.7% | 39.6% | +45.2% | model_too_pessimistic |
| market_side | batter_hits|over | 1071 | 89.8% | 44.7% | +45.2% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_250_499|fanduel | 184 | 88.0% | 43.1% | +45.0% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 444 | 87.8% | 42.9% | +44.9% | model_too_pessimistic |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 213 | 86.4% | 41.7% | +44.7% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 131 | 85.5% | 41.5% | +44.0% | model_too_pessimistic |
| surface | batter_hits|over|common | 979 | 89.2% | 45.4% | +43.7% | model_too_pessimistic |
| surface | batter_home_runs|over|common | 333 | 90.1% | 46.5% | +43.6% | model_too_pessimistic |
| book_surface | batter_home_runs|over|common|fanduel | 333 | 90.1% | 46.5% | +43.6% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 122 | 95.9% | 52.7% | +43.2% | model_too_pessimistic |
| surface | batter_total_bases|over|common | 828 | 89.6% | 46.7% | +42.9% | model_too_pessimistic |
| market_side | batter_total_bases|over | 1492 | 89.8% | 46.9% | +42.9% | model_too_pessimistic |
| surface | batter_total_bases|over|alt_tail | 664 | 90.1% | 47.2% | +42.9% | model_too_pessimistic |

## Close Reason Audit

`clv_unknown_reason` is label-only and is not used as a training feature.

| Reason | Rows | Close Captured | Line Available | Avg Predicted Capture |
|---|---:|---:|---:|---:|
| valid_close | 3320 | 100.0% | 100.0% | 45.3% |
| close_outside_two_hour_window | 202 | 0.0% | - | 42.4% |
| stale_close_before_lock | 127 | 0.0% | - | 30.0% |
| line_disappeared_at_close | 36 | 0.0% | 0.0% | 29.7% |
| fallback_other_book_only | 14 | 0.0% | - | 44.7% |
| no_valid_close_snapshot | 8 | 0.0% | - | 25.0% |

## Least Bookable Buckets

| Bucket | Rows | Close Captured | Line Available | Avail Rows | Stale | No Valid Close |
|---|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 4 | 0.0% | - | 0 | 75.0% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | - | 0 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_130_149|draftkings | 2 | 0.0% | - | 0 | 0.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_150_249|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_500_plus|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 10 | 20.0% | 100.0% | 2 | 10.0% | 20.0% |
| batter_total_bases|under|common|TB 2.5+|lay_150_180|draftkings | 7 | 42.9% | 100.0% | 3 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 59 | 49.2% | 100.0% | 29 | 11.9% | 11.9% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 136 | 49.3% | 100.0% | 67 | 9.6% | 1.5% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 17 | 52.9% | 100.0% | 9 | 17.6% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_150_180|draftkings | 13 | 53.8% | 100.0% | 7 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 23 | 60.9% | 100.0% | 14 | 13.0% | 8.7% |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 42 | 61.9% | 100.0% | 26 | 23.8% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 110 | 62.7% | 100.0% | 69 | 9.1% | 3.6% |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 878 | 63.1% | 100.0% | 554 | 11.6% | 1.7% |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 147 | 65.3% | 100.0% | 96 | 6.8% | 4.8% |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 69 | 66.7% | 100.0% | 46 | 4.3% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|draftkings | 21 | 66.7% | 100.0% | 14 | 0.0% | 9.5% |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 66.7% | 100.0% | 2 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 62 | 67.7% | 100.0% | 42 | 17.7% | 0.0% |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 499 | 71.3% | 100.0% | 356 | 13.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 60 | 71.7% | 100.0% | 43 | 11.7% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 18 | 72.2% | 100.0% | 13 | 22.2% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 104 | 73.1% | 100.0% | 76 | 10.6% | 2.9% |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 359 | 73.3% | 100.0% | 263 | 11.4% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 90 | 73.3% | 100.0% | 66 | 14.4% | 0.0% |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 564 | 73.4% | 100.0% | 414 | 11.2% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 226 | 73.5% | 100.0% | 166 | 8.0% | 2.7% |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 479 | 73.5% | 100.0% | 352 | 11.1% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 228 | 73.7% | 100.0% | 168 | 11.8% | 0.0% |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 824 | 75.6% | 100.0% | 623 | 10.4% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2408 | 75.6% | 100.0% | 1821 | 10.1% | 0.2% |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 595 | 75.6% | 100.0% | 450 | 10.8% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 74 | 75.7% | 100.0% | 56 | 4.1% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 78 | 76.9% | 100.0% | 60 | 2.6% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|draftkings | 35 | 77.1% | 100.0% | 27 | 2.9% | 5.7% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 37 | 78.4% | 100.0% | 29 | 10.8% | 0.0% |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 888 | 78.6% | 98.7% | 707 | 3.7% | 0.0% |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 1667 | 79.1% | 100.0% | 1319 | 8.7% | 0.2% |
