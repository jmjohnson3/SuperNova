# MLB Prop Miss Diagnostic

Generated UTC: 2026-06-24T15:09:57Z
Rows: 106333
Date range: 2026-05-31 to 2026-06-23
Unique dates: 24

## Miss Reason Counts

| Reason | Misses |
|---|---:|
| weak_market_bucket | 75217 |
| bad_bucket_roi | 73986 |
| model_worse_than_market_price | 33425 |
| lost_clv | 23201 |
| bad_player_rate_projection | 16318 |
| bad_player_projection | 11535 |
| bad_opportunity_projection | 10555 |
| bad_side_probability | 6239 |
| bad_clv_bookability | 5379 |
| large_projection_error | 2973 |
| bad_calibration_bucket | 1657 |
| bad_line_price_edge | 785 |
| bad_distribution_pricing | 415 |

## Miss Reasons By Market/Side

| Reason | Market/Side | Misses |
|---|---|---:|
| weak_market_bucket | batter_total_bases|over | 29594 |
| bad_bucket_roi | batter_total_bases|over | 29470 |
| weak_market_bucket | batter_hits|over | 20916 |
| bad_bucket_roi | batter_hits|over | 20910 |
| weak_market_bucket | batter_home_runs|over | 15817 |
| bad_bucket_roi | batter_home_runs|over | 15724 |
| bad_player_rate_projection | batter_total_bases|over | 11998 |
| model_worse_than_market_price | batter_hits|over | 10542 |
| model_worse_than_market_price | batter_total_bases|over | 9789 |
| model_worse_than_market_price | batter_home_runs|over | 9244 |
| lost_clv | batter_total_bases|over | 9028 |
| lost_clv | batter_hits|over | 6241 |
| bad_player_projection | batter_hits|over | 6220 |
| weak_market_bucket | batter_hits|under | 5009 |
| bad_bucket_roi | batter_hits|under | 4582 |
| bad_side_probability | batter_hits|over | 4546 |
| lost_clv | batter_home_runs|over | 4510 |
| bad_opportunity_projection | batter_total_bases|over | 4081 |
| bad_player_projection | batter_total_bases|over | 3162 |
| bad_opportunity_projection | batter_hits|over | 3007 |
| bad_opportunity_projection | batter_home_runs|over | 2223 |
| model_worse_than_market_price | batter_hits|under | 2220 |
| bad_player_rate_projection | batter_hits|over | 2163 |
| large_projection_error | batter_total_bases|over | 2157 |
| weak_market_bucket | batter_total_bases|under | 1932 |
| lost_clv | batter_hits|under | 1913 |
| bad_clv_bookability | batter_total_bases|over | 1888 |
| bad_bucket_roi | batter_total_bases|under | 1875 |
| bad_clv_bookability | batter_hits|over | 1678 |
| bad_side_probability | batter_total_bases|under | 1159 |
| bad_player_rate_projection | batter_hits|under | 1117 |
| weak_market_bucket | pitcher_strikeouts|under | 986 |
| weak_market_bucket | pitcher_strikeouts|over | 963 |
| bad_clv_bookability | batter_home_runs|over | 916 |
| bad_calibration_bucket | batter_hits|over | 916 |
| bad_bucket_roi | pitcher_strikeouts|under | 845 |
| lost_clv | batter_total_bases|under | 835 |
| bad_player_projection | batter_total_bases|under | 813 |
| model_worse_than_market_price | batter_total_bases|under | 668 |
| bad_opportunity_projection | batter_hits|under | 637 |

## Projection vs Betting Accuracy by Market/Side

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 38529 | 23.2% | 26.3% | 0.162 | -18.0% | 1.408 | 1.859 | 36206 | 37.3% | +0.09 |
| batter_hits|over | 33222 | 37.0% | 38.6% | 0.172 | -12.2% | 0.701 | 0.892 | 30788 | 38.0% | +0.08 |
| batter_home_runs|over | 17038 | 7.2% | 6.8% | 0.061 | -26.2% | 0.260 | 0.380 | 16080 | 33.7% | +0.04 |
| batter_hits|under | 9004 | 44.4% | 44.1% | 0.232 | -5.4% | 0.700 | 0.892 | 8164 | 32.7% | -0.25 |
| batter_total_bases|under | 4619 | 58.2% | 59.3% | 0.243 | -4.5% | 1.483 | 1.954 | 4203 | 31.1% | -0.33 |
| pitcher_strikeouts|over | 1967 | 51.0% | 49.8% | 0.247 | -2.6% | 1.842 | 2.301 | 1646 | 37.3% | -0.17 |
| pitcher_strikeouts|under | 1954 | 49.5% | 50.2% | 0.247 | -8.0% | 1.836 | 2.294 | 1637 | 43.4% | +0.22 |

## Weak Exact Buckets

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|common|TB 2.5+|lay_130_149|fanduel | 4 | 0.0% | 47.3% | - | -100.0% | 0.753 | 0.964 | 4 | 0.0% | -0.91 |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 4 | 0.0% | 49.7% | - | -100.0% | 1.202 | 1.435 | 2 | 100.0% | +3.24 |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|fanduel | 4 | 0.0% | 49.9% | - | -100.0% | 1.202 | 1.435 | 4 | 0.0% | -0.64 |
| batter_home_runs|over|common|HR 0.5|plus_100_149|fanduel | 3 | 0.0% | 21.6% | - | -100.0% | 0.289 | 0.289 | 3 | 66.7% | +0.55 |
| pitcher_strikeouts|under|common|K 8.5+|fair_lay|fanduel | 3 | 0.0% | 47.7% | - | -100.0% | 5.560 | 5.560 | 3 | 0.0% | -1.00 |
| batter_hits|under|common|H 0.5|heavy_lay|draftkings | 1 | 0.0% | 59.6% | - | -100.0% | 0.370 | 0.370 | 1 | 100.0% | +0.12 |
| batter_total_bases|over|alt_tail|TB 2.5+|fair_lay|fanduel | 1 | 0.0% | 45.0% | - | -100.0% | 0.485 | 0.485 | 1 | 0.0% | -1.22 |
| batter_total_bases|over|common|TB 2.5+|lay_130_149|draftkings | 1 | 0.0% | 45.0% | - | -100.0% | 0.367 | 0.367 | 1 | 0.0% | -1.92 |
| pitcher_strikeouts|over|common|K 4.5-6.0|heavy_lay|fanduel | 1 | 0.0% | 60.4% | - | -100.0% | 3.740 | 3.740 | 0 | - | - |
| pitcher_strikeouts|under|common|K 8.5+|fair_lay|draftkings | 4 | 25.0% | 48.9% | 0.227 | -55.0% | 4.665 | 4.916 | 4 | 0.0% | -1.45 |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_150_180|fanduel | 7 | 28.6% | 55.3% | 0.289 | -53.0% | 1.542 | 2.032 | 3 | 100.0% | +1.62 |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 33.3% | 50.0% | 0.240 | -48.4% | 1.234 | 1.387 | 3 | 0.0% | -0.08 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 189 | 24.9% | 43.7% | 0.233 | -42.6% | 0.696 | 0.843 | 188 | 29.3% | -0.36 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 199 | 25.1% | 41.6% | 0.223 | -41.0% | 0.714 | 0.905 | 196 | 48.0% | +0.15 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 106 | 28.3% | 46.3% | 0.231 | -39.4% | 1.363 | 1.738 | 98 | 35.7% | +0.27 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 8505 | 1.0% | 1.5% | 0.010 | -39.0% | 0.261 | 0.381 | 8020 | 32.5% | +0.02 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 72 | 29.2% | 49.1% | 0.247 | -38.4% | 0.641 | 0.717 | 66 | 63.6% | +1.55 |
| pitcher_strikeouts|under|common|K 6.5-8.0|fair_lay|fanduel | 36 | 36.1% | 49.9% | 0.252 | -33.0% | 1.886 | 2.204 | 35 | 48.6% | +0.13 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 449 | 10.5% | 16.1% | 0.097 | -31.6% | 0.635 | 0.747 | 413 | 37.0% | +0.19 |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 30 | 33.3% | 48.3% | 0.258 | -31.0% | 0.671 | 0.714 | 29 | 41.4% | +1.07 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 66 | 33.3% | 48.0% | 0.250 | -29.1% | 2.015 | 2.434 | 58 | 20.7% | -0.61 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 94 | 34.0% | 47.1% | 0.254 | -27.5% | 2.017 | 2.518 | 67 | 41.8% | +0.28 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 11505 | 8.0% | 11.8% | 0.075 | -26.7% | 1.326 | 1.745 | 10744 | 31.7% | +0.04 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 119 | 33.6% | 45.9% | 0.238 | -26.6% | 1.443 | 1.828 | 95 | 56.8% | +0.91 |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 26 | 34.6% | 46.0% | 0.241 | -24.9% | 2.417 | 3.137 | 19 | 5.3% | -1.87 |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 31 | 35.5% | 45.9% | 0.238 | -23.5% | 2.498 | 3.231 | 22 | 50.0% | -0.13 |
| batter_total_bases|over|common|TB 2.5+|fair_lay|draftkings | 10 | 40.0% | 43.8% | 0.275 | -22.7% | 1.600 | 1.761 | 10 | 70.0% | +0.17 |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 20 | 45.0% | 52.7% | 0.259 | -22.0% | 2.440 | 2.848 | 18 | 44.4% | -0.09 |
| pitcher_strikeouts|under|common|K 6.5-8.0|fair_lay|draftkings | 31 | 41.9% | 49.5% | 0.250 | -21.2% | 1.998 | 2.431 | 29 | 34.5% | -0.41 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 2919 | 28.7% | 33.0% | 0.206 | -20.3% | 1.223 | 1.600 | 2679 | 39.7% | +0.14 |

## Recent Losing Examples

| Date | Player | Bet | Price | Pred | Actual | PA | BF | Model | Market | CLV Reason | Labels |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-05-31 | Matt McLain | batter_hits over 0.5 draftkings | -115.0 | 0.584 | 0.000 | 4.2->1.0 | -->- | 55.5% | 50.1% | stale_close_before_lock | bad_bucket_roi, bad_clv_bookability, bad_opportunity_projection, bad_player_projection, weak_market_bucket |
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
