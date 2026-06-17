# MLB Prop Miss Diagnostic

Generated UTC: 2026-06-17T17:55:14Z
Rows: 65072
Date range: 2026-05-31 to 2026-06-15
Unique dates: 16

## Miss Reason Counts

| Reason | Misses |
|---|---:|
| weak_market_bucket | 45746 |
| bad_bucket_roi | 44171 |
| model_worse_than_market_price | 22325 |
| lost_clv | 13724 |
| bad_player_rate_projection | 9796 |
| bad_player_projection | 6571 |
| bad_opportunity_projection | 6566 |
| bad_clv_bookability | 5112 |
| bad_side_probability | 3754 |
| bad_calibration_bucket | 1889 |
| large_projection_error | 1812 |
| bad_line_price_edge | 487 |
| bad_distribution_pricing | 192 |
| unclassified_miss | 7 |

## Miss Reasons By Market/Side

| Reason | Market/Side | Misses |
|---|---|---:|
| weak_market_bucket | batter_total_bases|over | 17547 |
| bad_bucket_roi | batter_total_bases|over | 17470 |
| weak_market_bucket | batter_hits|over | 14020 |
| bad_bucket_roi | batter_hits|over | 13988 |
| weak_market_bucket | batter_home_runs|over | 9346 |
| bad_bucket_roi | batter_home_runs|over | 7935 |
| model_worse_than_market_price | batter_hits|over | 7900 |
| model_worse_than_market_price | batter_total_bases|over | 7174 |
| bad_player_rate_projection | batter_total_bases|over | 7053 |
| model_worse_than_market_price | batter_home_runs|over | 5479 |
| lost_clv | batter_total_bases|over | 5133 |
| lost_clv | batter_hits|over | 3942 |
| bad_player_projection | batter_hits|over | 3658 |
| weak_market_bucket | batter_hits|under | 3014 |
| bad_bucket_roi | batter_hits|under | 2982 |
| lost_clv | batter_home_runs|over | 2657 |
| bad_side_probability | batter_hits|over | 2626 |
| bad_opportunity_projection | batter_total_bases|over | 2437 |
| bad_opportunity_projection | batter_hits|over | 2047 |
| bad_clv_bookability | batter_total_bases|over | 1758 |
| bad_clv_bookability | batter_hits|over | 1730 |
| bad_player_rate_projection | batter_hits|over | 1498 |
| bad_player_projection | batter_total_bases|over | 1447 |
| bad_opportunity_projection | batter_home_runs|over | 1316 |
| large_projection_error | batter_total_bases|over | 1275 |
| weak_market_bucket | batter_total_bases|under | 1206 |
| bad_bucket_roi | batter_total_bases|under | 1172 |
| lost_clv | batter_hits|under | 1117 |
| model_worse_than_market_price | batter_hits|under | 935 |
| bad_clv_bookability | batter_home_runs|over | 875 |
| bad_side_probability | batter_total_bases|under | 815 |
| bad_player_rate_projection | batter_hits|under | 696 |
| bad_calibration_bucket | batter_hits|over | 658 |
| bad_player_projection | batter_total_bases|under | 649 |
| weak_market_bucket | pitcher_strikeouts|over | 613 |
| lost_clv | batter_total_bases|under | 503 |
| bad_bucket_roi | pitcher_strikeouts|over | 412 |
| bad_opportunity_projection | batter_hits|under | 384 |
| bad_clv_bookability | batter_hits|under | 376 |
| bad_calibration_bucket | pitcher_strikeouts|over | 336 |

## Projection vs Betting Accuracy by Market/Side

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 22956 | 23.6% | 25.6% | 0.163 | -17.3% | 1.400 | 1.877 | 20783 | 38.7% | +0.13 |
| batter_hits|over | 21394 | 34.5% | 35.7% | 0.161 | -11.6% | 0.700 | 0.895 | 18983 | 39.7% | +0.13 |
| batter_home_runs|over | 10106 | 7.5% | 7.1% | 0.064 | -23.6% | 0.256 | 0.386 | 9190 | 34.0% | +0.04 |
| batter_hits|under | 5425 | 44.4% | 45.0% | 0.233 | -4.9% | 0.698 | 0.892 | 4673 | 32.3% | -0.25 |
| batter_total_bases|under | 2926 | 58.8% | 60.5% | 0.242 | -3.7% | 1.480 | 1.971 | 2526 | 31.2% | -0.33 |
| pitcher_strikeouts|over | 1139 | 46.2% | 50.0% | 0.251 | -11.1% | 1.815 | 2.231 | 904 | 36.3% | -0.27 |
| pitcher_strikeouts|under | 1126 | 54.9% | 50.2% | 0.251 | 2.4% | 1.803 | 2.218 | 895 | 44.2% | +0.31 |

## Weak Exact Buckets

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over|common|TB 2.5+|lay_130_149|fanduel | 4 | 0.0% | 47.3% | - | -100.0% | 0.753 | 0.964 | 4 | 0.0% | -0.91 |
| batter_home_runs|over|common|HR 0.5|plus_100_149|fanduel | 3 | 0.0% | 21.6% | - | -100.0% | 0.289 | 0.289 | 3 | 66.7% | +0.55 |
| pitcher_strikeouts|under|common|K 8.5+|fair_lay|draftkings | 3 | 0.0% | 47.7% | - | -100.0% | 5.560 | 5.560 | 3 | 0.0% | -1.31 |
| pitcher_strikeouts|under|common|K 8.5+|fair_lay|fanduel | 3 | 0.0% | 47.7% | - | -100.0% | 5.560 | 5.560 | 3 | 0.0% | -1.00 |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 2 | 0.0% | 53.0% | - | -100.0% | 0.423 | 0.447 | 0 | - | - |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|fanduel | 2 | 0.0% | 53.0% | - | -100.0% | 0.423 | 0.447 | 2 | 0.0% | -1.27 |
| batter_total_bases|over|alt_tail|TB 2.5+|fair_lay|fanduel | 1 | 0.0% | 45.0% | - | -100.0% | 0.485 | 0.485 | 1 | 0.0% | -1.22 |
| batter_total_bases|over|common|TB 2.5+|lay_130_149|draftkings | 1 | 0.0% | 45.0% | - | -100.0% | 0.367 | 0.367 | 1 | 0.0% | -1.92 |
| pitcher_strikeouts|under|common|K <4.5|heavy_lay|fanduel | 3 | 33.3% | 50.0% | 0.240 | -48.4% | 1.234 | 1.387 | 2 | 0.0% | +0.00 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 119 | 22.7% | 43.2% | 0.228 | -48.2% | 0.714 | 0.849 | 113 | 31.9% | -0.43 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 129 | 24.0% | 40.8% | 0.221 | -44.3% | 0.724 | 0.932 | 118 | 58.5% | +0.40 |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 9 | 33.3% | 48.7% | 0.238 | -42.6% | 2.275 | 2.610 | 8 | 50.0% | -0.29 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 35 | 28.6% | 47.4% | 0.235 | -41.1% | 0.680 | 0.732 | 28 | 64.3% | +1.46 |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|draftkings | 13 | 30.8% | 50.6% | 0.253 | -40.9% | 1.642 | 2.102 | 11 | 54.5% | -0.01 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | 223 | 9.4% | 15.6% | 0.089 | -38.6% | 0.628 | 0.715 | 194 | 41.2% | +0.34 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|fanduel | 24 | 33.3% | 49.2% | 0.249 | -38.3% | 2.126 | 2.418 | 23 | 39.1% | +0.72 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 5043 | 1.1% | 1.3% | 0.011 | -37.0% | 0.256 | 0.388 | 4574 | 32.2% | +0.02 |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 16 | 31.2% | 47.4% | 0.235 | -36.9% | 0.681 | 0.725 | 15 | 60.0% | +2.17 |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|fanduel | 8 | 37.5% | 53.5% | 0.265 | -35.5% | 1.503 | 1.770 | 7 | 71.4% | +1.71 |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 29 | 34.5% | 49.8% | 0.242 | -35.4% | 1.933 | 2.282 | 28 | 60.7% | +1.08 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 44 | 31.8% | 50.0% | 0.257 | -32.5% | 2.197 | 2.530 | 41 | 14.6% | -1.03 |
| batter_total_bases|over|common|TB 2.5+|plus_500_plus|fanduel | 320 | 10.3% | 15.4% | 0.096 | -32.3% | 1.026 | 1.352 | 274 | 39.4% | +0.33 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 57 | 33.3% | 49.1% | 0.266 | -29.2% | 2.205 | 2.539 | 42 | 40.5% | +0.34 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 52 | 44.2% | 52.3% | 0.245 | -27.7% | 1.959 | 2.408 | 28 | 17.9% | -1.65 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 56 | 42.9% | 51.0% | 0.251 | -25.6% | 1.806 | 2.214 | 51 | 37.3% | -0.04 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_500_plus|fanduel | 6797 | 8.4% | 11.0% | 0.076 | -25.0% | 1.306 | 1.743 | 6105 | 32.2% | +0.05 |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|fanduel | 16 | 43.8% | 51.2% | 0.253 | -24.2% | 0.816 | 1.203 | 14 | 42.9% | +0.39 |
| batter_total_bases|over|common|TB 2.5+|fair_lay|draftkings | 10 | 40.0% | 43.8% | 0.275 | -22.7% | 1.600 | 1.761 | 10 | 70.0% | +0.17 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 44 | 36.4% | 49.8% | 0.247 | -22.3% | 1.070 | 1.392 | 38 | 36.8% | +0.73 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 100 | 31.0% | 35.8% | 0.216 | -21.6% | 1.166 | 1.482 | 55 | 54.5% | +0.82 |

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
