# MLB Prop Miss Diagnostic

Generated UTC: 2026-06-12T15:51:52Z
Rows: 40380
Date range: 2026-05-31 to 2026-06-11
Unique dates: 12

## Miss Reason Counts

| Reason | Misses |
|---|---:|
| weak_market_bucket | 28941 |
| bad_bucket_roi | 28164 |
| model_worse_than_market_price | 14796 |
| lost_clv | 8238 |
| bad_player_rate_projection | 6032 |
| bad_opportunity_projection | 4427 |
| bad_clv_bookability | 4140 |
| bad_player_projection | 3885 |
| bad_side_probability | 2315 |
| bad_calibration_bucket | 1195 |
| large_projection_error | 1185 |
| bad_line_price_edge | 309 |
| bad_distribution_pricing | 88 |
| unclassified_miss | 10 |

## Miss Reasons By Market/Side

| Reason | Market/Side | Misses |
|---|---|---:|
| weak_market_bucket | batter_total_bases|over | 10312 |
| weak_market_bucket | batter_hits|over | 10173 |
| bad_bucket_roi | batter_total_bases|over | 10066 |
| bad_bucket_roi | batter_hits|over | 9995 |
| model_worse_than_market_price | batter_hits|over | 6373 |
| weak_market_bucket | batter_home_runs|over | 5537 |
| bad_bucket_roi | batter_home_runs|over | 5514 |
| model_worse_than_market_price | batter_total_bases|over | 4599 |
| bad_player_rate_projection | batter_total_bases|over | 4000 |
| model_worse_than_market_price | batter_home_runs|over | 3072 |
| lost_clv | batter_total_bases|over | 2870 |
| lost_clv | batter_hits|over | 2791 |
| bad_player_projection | batter_hits|over | 2175 |
| weak_market_bucket | batter_hits|under | 1819 |
| bad_bucket_roi | batter_hits|under | 1800 |
| bad_opportunity_projection | batter_total_bases|over | 1557 |
| bad_opportunity_projection | batter_hits|over | 1548 |
| lost_clv | batter_home_runs|over | 1521 |
| bad_side_probability | batter_hits|over | 1518 |
| bad_clv_bookability | batter_hits|over | 1488 |
| bad_clv_bookability | batter_total_bases|over | 1406 |
| bad_player_rate_projection | batter_hits|over | 1305 |
| bad_player_projection | batter_total_bases|over | 914 |
| bad_opportunity_projection | batter_home_runs|over | 844 |
| large_projection_error | batter_total_bases|over | 783 |
| weak_market_bucket | batter_total_bases|under | 728 |
| bad_clv_bookability | batter_home_runs|over | 724 |
| bad_side_probability | batter_total_bases|under | 610 |
| lost_clv | batter_hits|under | 591 |
| bad_bucket_roi | batter_total_bases|under | 443 |
| model_worse_than_market_price | batter_hits|under | 405 |
| bad_player_rate_projection | batter_hits|under | 394 |
| weak_market_bucket | pitcher_strikeouts|over | 372 |
| bad_player_projection | batter_total_bases|under | 353 |
| bad_calibration_bucket | batter_hits|over | 318 |
| bad_bucket_roi | pitcher_strikeouts|over | 286 |
| bad_calibration_bucket | batter_total_bases|under | 281 |
| bad_clv_bookability | batter_hits|under | 277 |
| lost_clv | batter_total_bases|under | 276 |
| bad_opportunity_projection | batter_hits|under | 265 |

## Projection vs Betting Accuracy by Market/Side

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_hits|over | 14608 | 30.4% | 31.1% | 0.144 | -10.1% | 0.697 | 0.893 | 12564 | 38.1% | +0.07 |
| batter_total_bases|over | 13512 | 23.7% | 24.9% | 0.164 | -15.3% | 1.375 | 1.819 | 11755 | 37.0% | +0.08 |
| batter_home_runs|over | 5970 | 7.3% | 7.1% | 0.062 | -30.6% | 0.249 | 0.373 | 5210 | 32.6% | +0.03 |
| batter_hits|under | 3273 | 44.4% | 45.5% | 0.235 | -4.0% | 0.691 | 0.884 | 2733 | 33.4% | -0.13 |
| batter_total_bases|under | 1758 | 58.6% | 62.8% | 0.243 | -4.0% | 1.457 | 1.897 | 1470 | 32.0% | -0.21 |
| pitcher_strikeouts|over | 636 | 41.5% | 50.1% | 0.251 | -20.2% | 1.646 | 2.063 | 488 | 40.8% | +0.18 |
| pitcher_strikeouts|under | 623 | 60.5% | 50.3% | 0.247 | 12.4% | 1.620 | 2.034 | 479 | 37.0% | -0.13 |

## Weak Exact Buckets

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | 53.0% | - | -100.0% | 0.423 | 0.447 | 0 | - | - |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|fanduel | 2 | 0.0% | 53.0% | - | -100.0% | 0.423 | 0.447 | 2 | 0.0% | -1.27 |
| batter_total_bases|over|common|TB 2.5+|lay_130_149|fanduel | 1 | 0.0% | 52.8% | - | -100.0% | 1.792 | 1.792 | 1 | 0.0% | -0.97 |
| batter_total_bases|under|common|TB 2.5+|fair_lay|draftkings | 1 | 0.0% | 74.7% | - | -100.0% | 2.958 | 2.958 | 1 | 100.0% | +4.14 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 32 | 12.5% | 50.5% | 0.262 | -73.9% | 2.067 | 2.272 | 24 | 58.3% | +1.41 |
| batter_hits|under|common|H 1.5|fair_lay|draftkings | 6 | 16.7% | 56.5% | 0.302 | -70.0% | 1.079 | 1.232 | 6 | 66.7% | -0.04 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 12 | 16.7% | 49.6% | 0.248 | -68.8% | 1.693 | 2.035 | 11 | 27.3% | -0.33 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 22 | 18.2% | 50.9% | 0.262 | -62.5% | 2.072 | 2.269 | 19 | 21.1% | -0.01 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 16 | 25.0% | 50.6% | 0.239 | -52.5% | 1.532 | 1.900 | 15 | 40.0% | -0.80 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 13 | 23.1% | 47.3% | 0.243 | -52.2% | 0.786 | 0.869 | 7 | 57.1% | +1.31 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 2976 | 0.8% | 1.3% | 0.008 | -50.8% | 0.249 | 0.375 | 2587 | 30.6% | +0.01 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|fanduel | 19 | 31.6% | 51.5% | 0.249 | -48.9% | 1.723 | 2.037 | 16 | 37.5% | +0.34 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 82 | 24.4% | 40.8% | 0.227 | -43.9% | 0.774 | 0.893 | 75 | 52.0% | -0.39 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 71 | 25.4% | 42.7% | 0.230 | -42.6% | 0.784 | 0.907 | 68 | 32.4% | -0.43 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 33 | 33.3% | 50.8% | 0.257 | -42.3% | 1.576 | 2.091 | 30 | 36.7% | +0.39 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 57 | 22.8% | 35.8% | 0.189 | -42.3% | 1.103 | 1.386 | 32 | 56.2% | +0.61 |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|fanduel | 10 | 30.0% | 50.7% | 0.257 | -41.5% | 1.233 | 1.700 | 9 | 22.2% | -0.71 |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|draftkings | 13 | 30.8% | 50.6% | 0.253 | -40.9% | 1.642 | 2.102 | 11 | 54.5% | -0.01 |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|fanduel | 8 | 37.5% | 53.5% | 0.265 | -35.5% | 1.503 | 1.770 | 7 | 71.4% | +1.71 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 84 | 38.1% | 51.0% | 0.256 | -29.8% | 0.644 | 0.715 | 67 | 40.3% | +0.36 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 25 | 44.0% | 53.4% | 0.246 | -28.3% | 2.065 | 2.525 | 12 | 8.3% | -1.00 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 54 | 40.7% | 50.4% | 0.247 | -24.5% | 1.646 | 2.000 | 47 | 53.2% | -0.37 |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 2 | 50.0% | 51.2% | 0.247 | -22.5% | 0.848 | 0.935 | 1 | 0.0% | +0.00 |
| batter_total_bases|over|common|TB 2.5+|plus_150_249|fanduel | 870 | 25.6% | 30.1% | 0.194 | -22.2% | 1.533 | 2.016 | 800 | 37.9% | +0.10 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|draftkings | 37 | 45.9% | 51.5% | 0.254 | -21.2% | 1.559 | 2.048 | 25 | 44.0% | -0.12 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 133 | 12.0% | 15.4% | 0.107 | -21.1% | 0.655 | 0.759 | 106 | 31.1% | +0.12 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 4106 | 8.7% | 10.5% | 0.079 | -21.0% | 1.293 | 1.715 | 3531 | 31.0% | +0.03 |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 11 | 45.5% | 51.4% | 0.254 | -21.0% | 0.746 | 0.860 | 11 | 45.5% | +0.40 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 1789 | 18.6% | 22.2% | 0.153 | -20.1% | 0.668 | 0.857 | 1542 | 44.2% | +0.16 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | 97 | 37.1% | 48.4% | 0.254 | -19.3% | 1.824 | 2.286 | 72 | 34.7% | +0.23 |

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
| 2026-05-31 | Cody Bellinger | batter_hits under 0.5 draftkings | 200.0 | 1.127 | 2.000 | 4.2->5.0 | -->- | 35.4% | 32.4% | stale_close_before_lock | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-05-31 | Colson Montgomery | batter_hits under 0.5 draftkings | 147.0 | 0.762 | 1.000 | 4.3->3.0 | -->- | 44.0% | 37.4% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | Darell Hernaiz | batter_hits under 0.5 draftkings | 130.0 | 0.708 | 1.000 | 3.4->5.0 | -->- | 44.0% | 40.6% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | David Hamilton | batter_hits under 0.5 draftkings | 100.0 | 0.547 | 1.000 | 3.0->3.0 | -->- | 53.8% | 46.5% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Drew Gilbert | batter_hits under 0.5 draftkings | 160.0 | 0.885 | 1.000 | 2.6->6.0 | -->- | 40.2% | 35.9% | nan | bad_bucket_roi, bad_opportunity_projection, weak_market_bucket |
| 2026-05-31 | Ezequiel Tovar | batter_hits under 0.5 draftkings | 167.0 | 0.971 | 1.000 | 4.1->3.0 | -->- | 40.2% | 33.9% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Freddie Freeman | batter_hits under 0.5 draftkings | 195.0 | 0.947 | 1.000 | 4.3->5.0 | -->- | 40.2% | 32.4% | stale_close_before_lock | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-05-31 | Ildemaro Vargas | batter_hits under 0.5 draftkings | 131.0 | 0.860 | 1.000 | 3.6->4.0 | -->- | 44.0% | 39.4% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | J.P. Crawford | batter_hits under 0.5 draftkings | 150.0 | 0.874 | 1.000 | 4.0->5.0 | -->- | 40.2% | 37.5% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Jac Caglianone | batter_hits under 0.5 draftkings | 147.0 | 0.835 | 1.000 | 3.9->4.0 | -->- | 44.0% | 37.1% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Jackson Merrill | batter_hits under 0.5 draftkings | 183.0 | 1.093 | 3.000 | 3.7->4.0 | -->- | 38.1% | 33.1% | nan | bad_bucket_roi, bad_player_rate_projection, lost_clv, weak_market_bucket |
| 2026-05-31 | Jake Mangum | batter_hits under 0.5 draftkings | 172.0 | 0.947 | 1.000 | 3.0->4.0 | -->- | 44.0% | 35.2% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | Jeremy Peña | batter_hits under 0.5 draftkings | 150.0 | 0.840 | 1.000 | 3.9->4.0 | -->- | 44.0% | 37.9% | nan | bad_bucket_roi, lost_clv, weak_market_bucket |
| 2026-05-31 | JJ Wetherholt | batter_hits under 0.5 draftkings | 203.0 | 1.345 | 2.000 | 4.5->4.0 | -->- | 34.9% | 32.8% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | Jonathan Aranda | batter_hits under 0.5 draftkings | 201.0 | 1.002 | 1.000 | 4.4->5.0 | -->- | 38.1% | 32.6% | stale_close_before_lock | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-05-31 | Jorge Mateo | batter_hits under 0.5 draftkings | 138.0 | 0.748 | 1.000 | 3.4->3.0 | -->- | 44.0% | 39.3% | nan | bad_bucket_roi, weak_market_bucket |
| 2026-05-31 | José Caballero | batter_hits under 0.5 draftkings | 151.0 | 0.820 | 1.000 | 3.2->5.0 | -->- | 40.2% | 37.6% | nan | bad_bucket_roi, bad_opportunity_projection, lost_clv, weak_market_bucket |
| 2026-05-31 | José Ramírez | batter_hits under 0.5 draftkings | 183.0 | 0.951 | 1.000 | 3.9->5.0 | -->- | 39.9% | 33.4% | nan | bad_bucket_roi, weak_market_bucket |
