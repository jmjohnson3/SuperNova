# MLB Prop Miss Diagnostic

Generated UTC: 2026-06-19T19:59:59Z
Rows: 78285
Date range: 2026-05-31 to 2026-06-18
Unique dates: 19

## Miss Reason Counts

| Reason | Misses |
|---|---:|
| weak_market_bucket | 55462 |
| bad_bucket_roi | 53106 |
| model_worse_than_market_price | 25768 |
| lost_clv | 16361 |
| bad_player_rate_projection | 11906 |
| bad_player_projection | 7963 |
| bad_opportunity_projection | 7699 |
| bad_clv_bookability | 5708 |
| bad_side_probability | 4514 |
| large_projection_error | 2158 |
| bad_calibration_bucket | 1762 |
| bad_line_price_edge | 601 |
| bad_distribution_pricing | 300 |

## Miss Reasons By Market/Side

| Reason | Market/Side | Misses |
|---|---|---:|
| weak_market_bucket | batter_total_bases|over | 21451 |
| bad_bucket_roi | batter_total_bases|over | 21363 |
| weak_market_bucket | batter_hits|over | 16166 |
| bad_bucket_roi | batter_hits|over | 16145 |
| weak_market_bucket | batter_home_runs|over | 11444 |
| bad_bucket_roi | batter_home_runs|over | 9703 |
| bad_player_rate_projection | batter_total_bases|over | 8783 |
| model_worse_than_market_price | batter_hits|over | 8696 |
| model_worse_than_market_price | batter_total_bases|over | 8110 |
| model_worse_than_market_price | batter_home_runs|over | 6632 |
| lost_clv | batter_total_bases|over | 6200 |
| lost_clv | batter_hits|over | 4567 |
| bad_player_projection | batter_hits|over | 4473 |
| weak_market_bucket | batter_hits|under | 3646 |
| bad_bucket_roi | batter_hits|under | 3610 |
| bad_side_probability | batter_hits|over | 3240 |
| lost_clv | batter_home_runs|over | 3175 |
| bad_opportunity_projection | batter_total_bases|over | 2905 |
| bad_opportunity_projection | batter_hits|over | 2337 |
| bad_clv_bookability | batter_total_bases|over | 2014 |
| bad_clv_bookability | batter_hits|over | 1859 |
| bad_player_projection | batter_total_bases|over | 1748 |
| bad_player_rate_projection | batter_hits|over | 1635 |
| bad_opportunity_projection | batter_home_runs|over | 1571 |
| large_projection_error | batter_total_bases|over | 1556 |
| weak_market_bucket | batter_total_bases|under | 1420 |
| bad_bucket_roi | batter_total_bases|under | 1384 |
| lost_clv | batter_hits|under | 1365 |
| model_worse_than_market_price | batter_hits|under | 1274 |
| bad_clv_bookability | batter_home_runs|over | 1007 |
| bad_side_probability | batter_total_bases|under | 910 |
| bad_player_rate_projection | batter_hits|under | 834 |
| bad_calibration_bucket | batter_hits|over | 833 |
| bad_player_projection | batter_total_bases|under | 758 |
| weak_market_bucket | pitcher_strikeouts|over | 705 |
| weak_market_bucket | pitcher_strikeouts|under | 630 |
| bad_bucket_roi | pitcher_strikeouts|over | 614 |
| lost_clv | batter_total_bases|under | 598 |
| bad_opportunity_projection | batter_hits|under | 451 |
| bad_clv_bookability | batter_hits|under | 417 |

## Projection vs Betting Accuracy by Market/Side

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 28052 | 23.5% | 25.8% | 0.163 | -16.7% | 1.403 | 1.868 | 25551 | 38.8% | +0.14 |
| batter_hits|over | 25128 | 35.7% | 37.1% | 0.166 | -12.1% | 0.699 | 0.892 | 22488 | 39.7% | +0.13 |
| batter_home_runs|over | 12375 | 7.5% | 7.1% | 0.064 | -24.2% | 0.259 | 0.386 | 11321 | 34.5% | +0.05 |
| batter_hits|under | 6559 | 44.4% | 44.7% | 0.233 | -4.9% | 0.696 | 0.890 | 5721 | 32.2% | -0.26 |
| batter_total_bases|under | 3478 | 59.2% | 60.0% | 0.241 | -3.0% | 1.476 | 1.958 | 3026 | 30.6% | -0.35 |
| pitcher_strikeouts|over | 1353 | 47.9% | 49.7% | 0.250 | -7.7% | 1.783 | 2.189 | 1097 | 35.0% | -0.33 |
| pitcher_strikeouts|under | 1340 | 53.0% | 50.3% | 0.250 | -1.8% | 1.772 | 2.178 | 1088 | 45.8% | +0.39 |

## Weak Exact Buckets

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|common|TB 2.5+|lay_130_149|fanduel | 4 | 0.0% | 47.3% | - | -100.0% | 0.753 | 0.964 | 4 | 0.0% | -0.91 |
| batter_home_runs|over|common|HR 0.5|plus_100_149|fanduel | 3 | 0.0% | 21.6% | - | -100.0% | 0.289 | 0.289 | 3 | 66.7% | +0.55 |
| pitcher_strikeouts|under|common|K 8.5+|fair_lay|draftkings | 3 | 0.0% | 47.7% | - | -100.0% | 5.560 | 5.560 | 3 | 0.0% | -1.31 |
| pitcher_strikeouts|under|common|K 8.5+|fair_lay|fanduel | 3 | 0.0% | 47.7% | - | -100.0% | 5.560 | 5.560 | 3 | 0.0% | -1.00 |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | 53.0% | - | -100.0% | 0.423 | 0.447 | 0 | - | - |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|fanduel | 2 | 0.0% | 53.0% | - | -100.0% | 0.423 | 0.447 | 2 | 0.0% | -1.27 |
| batter_hits|under|common|H 0.5|heavy_lay|draftkings | 1 | 0.0% | 59.6% | - | -100.0% | 0.370 | 0.370 | 1 | 100.0% | +0.12 |
| batter_total_bases|over|alt_tail|TB 2.5+|fair_lay|fanduel | 1 | 0.0% | 45.0% | - | -100.0% | 0.485 | 0.485 | 1 | 0.0% | -1.22 |
| batter_total_bases|over|common|TB 2.5+|lay_130_149|draftkings | 1 | 0.0% | 45.0% | - | -100.0% | 0.367 | 0.367 | 1 | 0.0% | -1.92 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 49 | 24.5% | 47.7% | 0.236 | -48.4% | 0.709 | 0.792 | 42 | 59.5% | +1.14 |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 33.3% | 50.0% | 0.240 | -48.4% | 1.234 | 1.387 | 2 | 0.0% | +0.00 |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 23 | 26.1% | 47.9% | 0.236 | -46.7% | 0.661 | 0.697 | 22 | 45.5% | +1.41 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 32 | 31.2% | 49.6% | 0.241 | -41.5% | 2.055 | 2.388 | 31 | 64.5% | +1.14 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 137 | 25.5% | 43.3% | 0.228 | -40.9% | 0.732 | 0.884 | 131 | 32.1% | -0.38 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 56 | 28.6% | 48.8% | 0.238 | -39.0% | 1.133 | 1.502 | 50 | 34.0% | +0.42 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 6176 | 1.0% | 1.3% | 0.010 | -39.0% | 0.259 | 0.387 | 5638 | 33.2% | +0.02 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 24 | 33.3% | 49.2% | 0.249 | -38.3% | 2.126 | 2.418 | 23 | 39.1% | +0.72 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 140 | 26.4% | 41.0% | 0.223 | -38.2% | 0.751 | 0.962 | 129 | 54.3% | +0.33 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 291 | 10.0% | 15.7% | 0.093 | -36.1% | 0.632 | 0.730 | 262 | 40.5% | +0.29 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 120 | 27.5% | 36.0% | 0.208 | -30.5% | 1.149 | 1.443 | 68 | 55.9% | +0.95 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 59 | 35.6% | 49.0% | 0.267 | -24.8% | 2.221 | 2.544 | 44 | 43.2% | +0.40 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 8304 | 8.3% | 11.3% | 0.076 | -24.2% | 1.317 | 1.748 | 7505 | 32.5% | +0.06 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 57 | 43.9% | 51.0% | 0.250 | -23.9% | 1.808 | 2.209 | 52 | 38.5% | -0.01 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 393 | 12.0% | 15.6% | 0.108 | -23.2% | 1.090 | 1.434 | 347 | 37.5% | +0.27 |
| batter_total_bases|over|common|TB 2.5+|fair_lay|draftkings | 10 | 40.0% | 43.8% | 0.275 | -22.7% | 1.600 | 1.761 | 10 | 70.0% | +0.17 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 60 | 48.3% | 53.1% | 0.241 | -21.1% | 1.887 | 2.326 | 31 | 16.1% | -1.84 |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 11 | 45.5% | 48.8% | 0.242 | -20.8% | 2.167 | 2.467 | 10 | 60.0% | +0.53 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 48 | 37.5% | 49.5% | 0.262 | -20.0% | 2.231 | 2.536 | 43 | 14.0% | -1.04 |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 16 | 37.5% | 46.9% | 0.241 | -19.9% | 2.436 | 2.753 | 11 | 0.0% | -2.02 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|fanduel | 192 | 46.4% | 48.3% | 0.252 | -19.7% | 1.762 | 2.312 | 180 | 46.1% | +0.17 |

## Recent Losing Examples

| Date | Player | Bet | Price | Pred | Actual | PA | BF | Model | Market | CLV Reason | Labels |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-05-31 | Matt McLain | batter_hits over 0.5 draftkings | -115.0 | 0.584 | 0.000 | 4.2->1.0 | -->- | 55.5% | 50.1% | stale_close_before_lock | bad_bucket_roi, bad_calibration_bucket, bad_clv_bookability, bad_opportunity_projection, bad_player_projection, weak_market_bucket |
| 2026-05-31 | Alec Bohm | batter_hits under 0.5 draftkings | 149.0 | 0.782 | 1.000 | 3.9->4.0 | -->- | 40.2% | 37.6% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Alec Burleson | batter_hits under 0.5 draftkings | 190.0 | 1.112 | 1.000 | 4.0->4.0 | -->- | 34.9% | 32.3% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Anthony Volpe | batter_hits under 0.5 draftkings | 163.0 | 1.105 | 2.000 | 4.1->4.0 | -->- | 39.9% | 36.3% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Ben Rice | batter_hits under 0.5 draftkings | 158.0 | 1.181 | 2.000 | 4.4->5.0 | -->- | 39.9% | 36.4% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Bobby Witt Jr. | batter_hits under 0.5 draftkings | 186.0 | 1.005 | 2.000 | 4.1->5.0 | -->- | 40.2% | 32.0% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Brandon Lowe | batter_hits under 0.5 draftkings | 152.0 | 0.857 | 1.000 | 3.8->5.0 | -->- | 40.2% | 36.6% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Brooks Lee | batter_hits under 0.5 draftkings | 149.0 | 1.041 | 2.000 | 3.8->4.0 | -->- | 40.2% | 37.8% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Carson Benge | batter_hits under 0.5 draftkings | 193.0 | 1.079 | 1.000 | 4.2->5.0 | -->- | 38.1% | 32.1% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Carter Jensen | batter_hits under 0.5 draftkings | 135.0 | 0.785 | 1.000 | 3.4->4.0 | -->- | 44.0% | 38.2% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Chase Meidroth | batter_hits under 0.5 draftkings | 169.0 | 0.923 | 1.000 | 4.5->3.0 | -->- | 40.2% | 35.8% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | CJ Abrams | batter_hits under 0.5 draftkings | 165.0 | 0.955 | 2.000 | 4.0->4.0 | -->- | 40.2% | 35.8% | nan | bad_bucket_roi, bad_player_rate_projection, lost_clv, weak_market_bucket |
| 2026-05-31 | Cody Bellinger | batter_hits under 0.5 draftkings | 200.0 | 1.127 | 2.000 | 4.2->5.0 | -->- | 35.4% | 32.4% | line_disappeared_at_close | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-05-31 | Colson Montgomery | batter_hits under 0.5 draftkings | 147.0 | 0.762 | 1.000 | 4.3->3.0 | -->- | 44.0% | 37.4% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | Darell Hernaiz | batter_hits under 0.5 draftkings | 130.0 | 0.708 | 1.000 | 3.4->5.0 | -->- | 44.0% | 40.6% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | David Hamilton | batter_hits under 0.5 draftkings | 100.0 | 0.547 | 1.000 | 3.0->3.0 | -->- | 53.8% | 46.5% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Drew Gilbert | batter_hits under 0.5 draftkings | 160.0 | 0.885 | 1.000 | 2.6->6.0 | -->- | 40.2% | 35.9% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | Ezequiel Tovar | batter_hits under 0.5 draftkings | 167.0 | 0.971 | 1.000 | 4.1->3.0 | -->- | 40.2% | 33.9% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Freddie Freeman | batter_hits under 0.5 draftkings | 195.0 | 0.947 | 1.000 | 4.3->5.0 | -->- | 40.2% | 32.4% | line_disappeared_at_close | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-05-31 | Ildemaro Vargas | batter_hits under 0.5 draftkings | 131.0 | 0.860 | 1.000 | 3.6->4.0 | -->- | 44.0% | 39.4% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | J.P. Crawford | batter_hits under 0.5 draftkings | 150.0 | 0.874 | 1.000 | 4.0->5.0 | -->- | 40.2% | 37.5% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Jac Caglianone | batter_hits under 0.5 draftkings | 147.0 | 0.835 | 1.000 | 3.9->4.0 | -->- | 44.0% | 37.1% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Jackson Merrill | batter_hits under 0.5 draftkings | 183.0 | 1.093 | 3.000 | 3.7->4.0 | -->- | 38.1% | 33.1% | nan | bad_bucket_roi, bad_player_rate_projection, lost_clv, weak_market_bucket |
| 2026-05-31 | Jake Mangum | batter_hits under 0.5 draftkings | 172.0 | 0.947 | 1.000 | 3.0->4.0 | -->- | 44.0% | 35.2% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Jeremy Peña | batter_hits under 0.5 draftkings | 150.0 | 0.840 | 1.000 | 3.9->4.0 | -->- | 44.0% | 37.9% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | JJ Wetherholt | batter_hits under 0.5 draftkings | 203.0 | 1.345 | 2.000 | 4.5->4.0 | -->- | 34.9% | 32.8% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Jonathan Aranda | batter_hits under 0.5 draftkings | 201.0 | 1.002 | 1.000 | 4.4->5.0 | -->- | 38.1% | 32.6% | line_disappeared_at_close | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-05-31 | Jorge Mateo | batter_hits under 0.5 draftkings | 138.0 | 0.748 | 1.000 | 3.4->3.0 | -->- | 44.0% | 39.3% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | José Caballero | batter_hits under 0.5 draftkings | 151.0 | 0.820 | 1.000 | 3.2->5.0 | -->- | 40.2% | 37.6% | nan | bad_bucket_roi, bad_opportunity_projection, lost_clv, weak_market_bucket |
| 2026-05-31 | José Ramírez | batter_hits under 0.5 draftkings | 183.0 | 0.951 | 1.000 | 3.9->5.0 | -->- | 39.9% | 33.4% | nan | bad_bucket_roi, weak_market_bucket |
