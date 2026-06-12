# MLB Prop Bookability Model

Generated UTC: 2026-06-12T08:55:36Z
Rows: 44889
Date range: 2026-05-31 to 2026-06-11
Status: ready

## Holdout

| Metric | Value |
|---|---:|
| rows | 2202.000 |
| actual_bookable_rate | 85.3% |
| avg_pred_bookable | 0.579 |
| brier_baseline | 0.129 |
| brier_model | 0.217 |
| log_loss_model | 0.611 |
| auc_model | 0.787 |
| model_usable | no |
| selected_scoring_method | empirical_bucket_rate |

## Prediction Calibration

| Predicted Bucket | Rows | Actual Bookable | Avg Predicted | Error |
|---|---:|---:|---:|---:|
| 00-10% | 39 | 43.6% | 7.2% | +36.4% |
| 10-20% | 208 | 53.4% | 15.8% | +37.5% |
| 20-30% | 311 | 75.9% | 25.0% | +50.9% |
| 30-40% | 273 | 85.0% | 34.8% | +50.2% |
| 40-50% | 209 | 85.6% | 45.0% | +40.7% |
| 50-60% | 175 | 85.7% | 54.8% | +30.9% |
| 60-70% | 80 | 91.2% | 64.8% | +26.5% |
| 70-80% | 77 | 100.0% | 76.1% | +23.9% |
| 80-90% | 231 | 97.0% | 86.1% | +10.9% |
| 90-100% | 599 | 96.8% | 94.7% | +2.1% |

## Prediction Gap Audit

| Level | Bucket | Rows | Actual | Predicted | Error | Note |
|---|---|---:|---:|---:|---:|---|
| surface | batter_hits|over|alt_tail | 60 | 98.3% | 55.8% | +42.6% | model_too_pessimistic |
| book_surface | batter_hits|over|alt_tail|fanduel | 60 | 98.3% | 55.8% | +42.6% | model_too_pessimistic |
| exact_bucket | batter_hits|over|alt_tail|H 2.5+|plus_500_plus|fanduel | 60 | 98.3% | 55.8% | +42.6% | model_too_pessimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | 44 | 97.7% | 57.1% | +40.6% | model_too_pessimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_100_149|draftkings | 73 | 79.5% | 39.7% | +39.8% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|draftkings | 100 | 87.0% | 48.2% | +38.8% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|lay_150_180|draftkings | 30 | 83.3% | 46.2% | +37.2% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 88 | 90.9% | 53.8% | +37.1% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | 95 | 90.5% | 56.2% | +34.3% | model_too_pessimistic |
| market_side | batter_total_bases|under | 108 | 88.9% | 55.3% | +33.6% | model_too_pessimistic |
| surface | batter_total_bases|under|common | 108 | 88.9% | 55.3% | +33.6% | model_too_pessimistic |
| book_surface | batter_total_bases|under|common|draftkings | 108 | 88.9% | 55.3% | +33.6% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 67 | 97.0% | 63.7% | +33.3% | model_too_pessimistic |
| book_surface | batter_total_bases|over|common|draftkings | 108 | 88.9% | 56.3% | +32.5% | model_too_pessimistic |
| exact_bucket | batter_hits|under|common|H 0.5|plus_150_249|draftkings | 75 | 89.3% | 57.0% | +32.4% | model_too_pessimistic |
| market_side | batter_hits|under | 197 | 79.7% | 47.9% | +31.8% | model_too_pessimistic |
| surface | batter_hits|under|common | 197 | 79.7% | 47.9% | +31.8% | model_too_pessimistic |
| book_surface | batter_hits|under|common|draftkings | 197 | 79.7% | 47.9% | +31.8% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 1.5|plus_150_249|fanduel | 88 | 96.6% | 64.9% | +31.7% | model_too_pessimistic |
| exact_bucket | batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | 30 | 93.3% | 61.9% | +31.4% | model_too_pessimistic |
| book_surface | batter_hits|over|common|draftkings | 197 | 79.7% | 49.1% | +30.6% | model_too_pessimistic |
| exact_bucket | batter_hits|over|common|H 0.5|heavy_lay|fanduel | 132 | 91.7% | 61.7% | +30.0% | model_too_pessimistic |
| surface | batter_total_bases|over|common | 468 | 86.3% | 57.3% | +29.1% | model_too_pessimistic |
| market_side | batter_hits|over | 617 | 84.9% | 56.5% | +28.4% | model_too_pessimistic |
| market_side | batter_total_bases|over | 828 | 86.0% | 57.8% | +28.1% | model_too_pessimistic |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | 66 | 92.4% | 64.3% | +28.1% | model_too_pessimistic |
| book_surface | batter_total_bases|over|common|fanduel | 360 | 85.6% | 57.5% | +28.0% | model_too_pessimistic |
| surface | batter_home_runs|over|common | 180 | 85.6% | 57.8% | +27.7% | model_too_pessimistic |
| book_surface | batter_home_runs|over|common|fanduel | 180 | 85.6% | 57.8% | +27.7% | model_too_pessimistic |
| exact_bucket | batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 114 | 81.6% | 54.1% | +27.5% | model_too_pessimistic |
| market_side | batter_home_runs|over | 360 | 85.6% | 58.5% | +27.1% | model_too_pessimistic |
| surface | batter_total_bases|over|alt_tail | 360 | 85.6% | 58.6% | +27.0% | model_too_pessimistic |
| book_surface | batter_total_bases|over|alt_tail|fanduel | 360 | 85.6% | 58.6% | +27.0% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 231 | 82.3% | 55.3% | +27.0% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 122 | 91.0% | 64.1% | +26.9% | model_too_pessimistic |
| surface | batter_hits|over|common | 557 | 83.5% | 56.6% | +26.9% | model_too_pessimistic |
| surface | batter_home_runs|over|alt_tail | 180 | 85.6% | 59.1% | +26.5% | model_too_pessimistic |
| book_surface | batter_home_runs|over|alt_tail|fanduel | 180 | 85.6% | 59.1% | +26.5% | model_too_pessimistic |
| exact_bucket | batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 180 | 85.6% | 59.1% | +26.5% | model_too_pessimistic |
| exact_bucket | batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 99 | 80.8% | 54.8% | +26.0% | model_too_pessimistic |

## Close Reason Audit

`clv_unknown_reason` is label-only and is not used as a training feature.

| Reason | Rows | Actual Bookable | Avg Predicted |
|---|---:|---:|---:|
| valid_close | 1879 | 100.0% | 62.3% |
| stale_close_before_lock | 148 | 0.0% | 22.3% |
| close_outside_two_hour_window | 143 | 0.0% | 35.4% |
| fallback_other_book_only | 26 | 0.0% | 63.5% |
| no_valid_close_snapshot | 6 | 0.0% | 69.6% |

## Least Bookable Buckets

| Bucket | Rows | Bookable | Stale | No Valid Close |
|---|---:|---:|---:|---:|
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|draftkings | 4 | 0.0% | 0.0% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 4 | 0.0% | 75.0% | 0.0% |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 8.5+|lay_130_149|draftkings | 2 | 0.0% | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | 100.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 1 | 0.0% | 100.0% | 0.0% |
| batter_total_bases|under|common|TB 2.5+|lay_150_180|draftkings | 3 | 33.3% | 0.0% | 0.0% |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 16 | 43.8% | 12.5% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 27 | 44.4% | 22.2% | 14.8% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 69 | 46.4% | 15.9% | 0.0% |
| batter_hits|under|common|H 0.5|lay_130_149|draftkings | 18 | 50.0% | 16.7% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 10 | 50.0% | 20.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 2 | 50.0% | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 2 | 50.0% | 50.0% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 11 | 54.5% | 36.4% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 52 | 55.8% | 23.1% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 114 | 58.8% | 21.1% | 0.0% |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 5 | 60.0% | 0.0% | 0.0% |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 176 | 60.2% | 19.9% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 71 | 60.6% | 11.3% | 5.6% |
| pitcher_strikeouts|over|common|K <4.5|lay_150_180|draftkings | 18 | 61.1% | 27.8% | 0.0% |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | 458 | 61.8% | 16.2% | 1.7% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|draftkings | 54 | 63.0% | 13.0% | 0.0% |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 262 | 63.4% | 19.5% | 0.0% |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 52 | 63.5% | 17.3% | 0.0% |
| pitcher_strikeouts|over|common|K <4.5|plus_100_149|fanduel | 9 | 66.7% | 11.1% | 0.0% |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 6 | 66.7% | 33.3% | 0.0% |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_150_180|draftkings | 6 | 66.7% | 33.3% | 0.0% |
| batter_hits|under|common|H 0.5|lay_150_180|draftkings | 3 | 66.7% | 0.0% | 0.0% |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | 277 | 66.8% | 17.7% | 0.0% |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 232 | 66.8% | 18.5% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 37 | 67.6% | 5.4% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 28 | 67.9% | 21.4% | 0.0% |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 331 | 69.2% | 16.6% | 0.0% |
| batter_total_bases|under|common|TB 1.5|plus_100_149|draftkings | 43 | 69.8% | 4.7% | 0.0% |
| batter_total_bases|over|common|TB 1.5|lay_150_180|draftkings | 10 | 70.0% | 0.0% | 0.0% |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|draftkings | 7 | 71.4% | 0.0% | 0.0% |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 1229 | 72.6% | 15.1% | 0.0% |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 430 | 73.7% | 15.6% | 0.0% |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 121 | 74.4% | 11.6% | 1.7% |
