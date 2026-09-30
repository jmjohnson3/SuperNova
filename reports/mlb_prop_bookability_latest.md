# MLB Prop Bookability Model

Generated UTC: 2026-09-01T10:24:28Z
Rows: 327259
Date range: 2026-06-01 to 2026-07-30
Status: ready

## Close Capture Holdout

| Metric | Value |
|---|---:|
| rows | 66827.000 |
| actual_bookable_rate | 81.2% |
| avg_pred_bookable | 0.812 |
| brier_baseline | 0.153 |
| brier_model | 0.145 |
| log_loss_model | 0.455 |
| auc_model | 0.683 |
| model_usable | yes |
| selected_scoring_method | logistic |

## Line Availability Holdout

| Metric | Value |
|---|---:|
| rows | 63871.000 |
| actual_bookable_rate | 85.6% |
| avg_pred_bookable | 0.855 |
| brier_baseline | 0.124 |
| brier_model | 0.116 |
| log_loss_model | 0.373 |
| auc_model | 0.750 |
| model_usable | yes |
| selected_scoring_method | logistic |

## Close Capture Calibration

| Predicted Bucket | Rows | Actual Bookable | Avg Predicted | Error |
|---|---:|---:|---:|---:|
| 00-10% | 31 | 77.4% | 7.5% | +69.9% |
| 10-20% | 148 | 66.9% | 15.7% | +51.2% |
| 20-30% | 387 | 56.8% | 25.4% | +31.4% |
| 30-40% | 587 | 65.1% | 35.4% | +29.6% |
| 40-50% | 1147 | 63.3% | 45.5% | +17.7% |
| 50-60% | 2200 | 65.7% | 55.6% | +10.2% |
| 60-70% | 4729 | 69.0% | 65.6% | +3.4% |
| 70-80% | 9829 | 75.1% | 75.7% | -0.6% |
| 80-90% | 26586 | 82.8% | 85.7% | -2.9% |
| 90-100% | 21183 | 93.9% | 93.4% | +0.5% |

## Line Availability Calibration

| Predicted Bucket | Rows | Actual Available | Avg Predicted | Error |
|---|---:|---:|---:|---:|
| 00-10% | 64 | 76.6% | 6.1% | +70.4% |
| 10-20% | 190 | 76.3% | 15.5% | +60.8% |
| 20-30% | 259 | 72.6% | 25.2% | +47.4% |
| 30-40% | 321 | 68.5% | 35.2% | +33.4% |
| 40-50% | 540 | 68.5% | 45.3% | +23.2% |
| 50-60% | 1155 | 68.7% | 55.5% | +13.2% |
| 60-70% | 2566 | 74.7% | 65.9% | +8.8% |
| 70-80% | 6138 | 75.2% | 75.6% | -0.4% |
| 80-90% | 22647 | 81.2% | 86.0% | -4.9% |
| 90-100% | 29991 | 95.9% | 94.2% | +1.7% |

## Close Capture Prediction Gap Audit

| Level | Bucket | Rows | Actual | Predicted | Error | Note |
|---|---|---:|---:|---:|---:|---|
| exact_bucket | pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 12 | 33.3% | 84.4% | -51.1% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 24 | 50.0% | 84.6% | -34.6% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 17 | 52.9% | 86.6% | -33.7% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 37 | 94.6% | 63.4% | +31.2% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 38 | 44.7% | 73.7% | -29.0% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 39 | 53.8% | 82.6% | -28.7% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K <4.5|lay_150_180|fanduel | 35 | 57.1% | 82.6% | -25.4% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 2240 | 82.3% | 60.1% | +22.2% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|over|common|K <4.5|lay_130_149|draftkings | 23 | 100.0% | 79.5% | +20.5% | model_too_pessimistic |
| book_surface | batter_total_bases|over|common|draftkings | 2765 | 81.2% | 61.1% | +20.1% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 29 | 72.4% | 92.2% | -19.7% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|plus_100_149|fanduel | 30 | 56.7% | 76.0% | -19.4% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 30 | 66.7% | 85.7% | -19.1% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|fanduel | 230 | 68.3% | 86.6% | -18.4% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K <4.5|fair_lay|fanduel | 30 | 73.3% | 90.9% | -17.5% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 38 | 92.1% | 76.5% | +15.6% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 98 | 96.9% | 81.6% | +15.3% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|draftkings | 655 | 69.9% | 84.8% | -14.8% | model_too_optimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 333 | 88.0% | 73.3% | +14.7% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|under|common|K <4.5|lay_150_180|fanduel | 13 | 100.0% | 85.7% | +14.3% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_130_149|fanduel | 514 | 72.0% | 86.2% | -14.2% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 0.5|lay_130_149|draftkings | 73 | 53.4% | 67.6% | -14.2% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 27 | 100.0% | 86.7% | +13.3% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 14 | 100.0% | 86.9% | +13.1% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|plus_100_149|draftkings | 72 | 54.2% | 67.1% | -12.9% | model_too_optimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 771 | 59.8% | 72.3% | -12.5% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 60 | 85.0% | 72.7% | +12.3% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 64 | 53.1% | 64.9% | -11.8% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 63 | 66.7% | 78.2% | -11.6% | model_too_optimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 1203 | 88.8% | 77.9% | +10.9% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|over|common|K <4.5|fair_lay|draftkings | 33 | 81.8% | 92.7% | -10.9% | model_too_optimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 6.5-8.0|fair_lay|fanduel | 17 | 100.0% | 89.2% | +10.8% | model_too_pessimistic |
| surface | batter_hits|over|alt_tail | 1518 | 98.7% | 88.1% | +10.6% | model_too_pessimistic |
| book_surface | batter_hits|over|alt_tail|fanduel | 1518 | 98.7% | 88.1% | +10.6% | model_too_pessimistic |
| exact_bucket | batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 1518 | 98.7% | 88.1% | +10.6% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 33 | 81.8% | 92.2% | -10.3% | model_too_optimistic |
| exact_bucket | batter_hits|under|common|H 1.5|heavy_lay|draftkings | 694 | 71.0% | 81.3% | -10.2% | model_too_optimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 464 | 92.5% | 82.6% | +9.8% | model_too_pessimistic |
| exact_bucket | pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 103 | 72.8% | 82.6% | -9.8% | model_too_optimistic |
| exact_bucket | batter_hits|over|common|H 0.5|fair_lay|draftkings | 466 | 70.6% | 80.3% | -9.7% | model_too_optimistic |

## Close Reason Audit

`clv_unknown_reason` is label-only and is not used as a training feature.

| Reason | Rows | Close Captured | Line Available | Avg Predicted Capture |
|---|---:|---:|---:|---:|
| valid_close | 55447 | 100.0% | 100.0% | 84.0% |
| player_prop_unavailable_at_close | 7264 | 0.0% | 0.0% | 75.7% |
| close_outside_two_hour_window | 2288 | 0.0% | - | 80.2% |
| line_disappeared_at_close | 917 | 0.0% | 0.0% | 79.8% |
| stale_close_before_lock | 408 | 0.0% | - | 56.2% |
| fallback_other_book_only | 256 | 0.0% | - | 59.2% |
| player_market_unavailable_at_close | 243 | 0.0% | 0.0% | 74.7% |
| no_valid_close_snapshot | 4 | 0.0% | - | 89.6% |

## Least Bookable Buckets

| Bucket | Rows | Close Captured | Line Available | Avail Rows | Stale | No Valid Close |
|---|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|common|TB 1.5|plus_500_plus|fanduel | 10 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_150_249|draftkings | 6 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_250_499|fanduel | 6 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_150_249|fanduel | 5 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_250_499|draftkings | 3 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|over|common|H 1.5|plus_250_499|draftkings | 2 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|under|common|TB 2.5+|heavy_lay|draftkings | 2 | 0.0% | - | 0 | 100.0% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|lay_150_180|fanduel | 2 | 0.0% | 0.0% | 2 | 0.0% | 0.0% |
| batter_hits|over|alt_tail|H 2.5+|plus_150_249|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_500_plus|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_hits|under|common|H 2.5+|heavy_lay|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_100_149|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|draftkings | 1 | 0.0% | - | 0 | 100.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_150_180|draftkings | 5 | 20.0% | 20.0% | 5 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 48 | 31.2% | 33.3% | 45 | 0.0% | 6.2% |
| pitcher_strikeouts|over|common|K 4.5-6.0|heavy_lay|fanduel | 3 | 33.3% | 50.0% | 2 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_150_180|fanduel | 5 | 40.0% | 40.0% | 5 | 0.0% | 0.0% |
| batter_hits|under|common|H 0.5|heavy_lay|draftkings | 19 | 42.1% | 100.0% | 8 | 57.9% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 14 | 42.9% | 42.9% | 14 | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 45 | 44.4% | 50.0% | 40 | 6.7% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_130_149|draftkings | 8 | 50.0% | 50.0% | 8 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|heavy_lay|fanduel | 4 | 50.0% | 50.0% | 4 | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|plus_100_149|fanduel | 4 | 50.0% | 50.0% | 4 | 0.0% | 0.0% |
| batter_hits|over|common|H 1.5|lay_130_149|fanduel | 2 | 50.0% | 100.0% | 1 | 50.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_150_180|draftkings | 54 | 51.9% | 53.8% | 52 | 0.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 585 | 52.0% | 81.3% | 374 | 2.7% | 0.0% |
| batter_total_bases|under|common|TB 2.5+|lay_150_180|draftkings | 19 | 52.6% | 55.6% | 18 | 5.3% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 116 | 53.4% | 56.4% | 110 | 2.6% | 2.6% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 288 | 55.9% | 57.5% | 280 | 0.7% | 0.7% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 108 | 58.3% | 59.4% | 106 | 0.9% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 473 | 58.4% | 60.3% | 458 | 1.5% | 0.2% |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 234 | 60.7% | 62.0% | 229 | 2.1% | 0.0% |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 3530 | 63.0% | 65.9% | 3378 | 2.5% | 0.1% |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 302 | 63.9% | 66.8% | 289 | 2.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 475 | 64.6% | 69.0% | 445 | 3.8% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 64 | 65.6% | 68.9% | 61 | 4.7% | 0.0% |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 297 | 66.7% | 69.2% | 286 | 2.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 66.7% | 66.7% | 3 | 0.0% | 0.0% |
