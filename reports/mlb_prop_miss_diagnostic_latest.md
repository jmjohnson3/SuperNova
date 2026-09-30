# MLB Prop Miss Diagnostic

Generated UTC: 2026-09-01T11:28:49Z
Rows: 25305
Date range: 2026-06-03 to 2026-07-30
Unique dates: 55

## Miss Reason Counts

| Reason | Misses |
|---|---:|
| weak_market_bucket | 19592 |
| bad_bucket_roi | 19237 |
| bad_calibration_bucket | 10724 |
| lost_clv | 5953 |
| bad_player_rate_projection | 5239 |
| bad_opportunity_projection | 2758 |
| bad_clv_bookability | 2045 |
| bad_player_projection | 1816 |
| model_worse_than_market_price | 1646 |
| bad_line_price_edge | 1257 |
| bad_side_probability | 1019 |
| large_projection_error | 861 |
| bad_distribution_pricing | 435 |
| unclassified_miss | 6 |

## Accuracy By Ledger

| Ledger | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| One-sided FanDuel props | 22248 | 18.5% | 25.4% | 0.137 | -20.1% | 1.041 | 1.486 | 20314 | 39.7% | +0.11 |
| Watch props | 2743 | 48.0% | 53.3% | 0.247 | -4.9% | 1.185 | 1.635 | 2387 | 40.3% | +0.12 |
| Paper common props | 223 | 42.6% | 54.9% | 0.242 | -16.7% | 1.036 | 1.442 | 0 | - | - |
| Lottery props | 51 | 31.4% | 45.0% | 0.236 | -22.5% | 0.753 | 0.943 | 47 | 34.0% | +0.01 |
| Micro projection props | 40 | 45.0% | 44.9% | 0.234 | -8.5% | 1.571 | 1.867 | 36 | 44.4% | -0.36 |

## Primary Miss Cause By Ledger

Each losing pick receives one primary cause using opportunity, projection, distribution, pricing, CLV, and bucket evidence in that order.

| Ledger | Primary cause | Misses | Share of ledger losses |
|---|---|---:|---:|
| One-sided FanDuel props | bad_player_rate_projection | 4920 | 27.1% |
| One-sided FanDuel props | bad_calibration_bucket | 2849 | 15.7% |
| One-sided FanDuel props | lost_clv | 2750 | 15.2% |
| One-sided FanDuel props | bad_opportunity_projection | 2512 | 13.8% |
| One-sided FanDuel props | bad_bucket_roi | 2512 | 13.8% |
| One-sided FanDuel props | bad_player_projection | 1003 | 5.5% |
| One-sided FanDuel props | bad_line_price_edge | 765 | 4.2% |
| One-sided FanDuel props | bad_clv_bookability | 628 | 3.5% |
| Watch props | bad_player_projection | 479 | 33.6% |
| Watch props | bad_player_rate_projection | 285 | 20.0% |
| Watch props | bad_opportunity_projection | 218 | 15.3% |
| One-sided FanDuel props | model_worse_than_market_price | 165 | 0.9% |
| Watch props | bad_bucket_roi | 132 | 9.3% |
| Watch props | lost_clv | 122 | 8.5% |
| Watch props | bad_clv_bookability | 45 | 3.2% |
| Paper common props | bad_clv_bookability | 43 | 33.6% |
| Watch props | bad_calibration_bucket | 39 | 2.7% |
| Watch props | weak_market_bucket | 33 | 2.3% |
| One-sided FanDuel props | bad_distribution_pricing | 32 | 0.2% |
| Paper common props | bad_player_projection | 30 | 23.4% |
| Paper common props | bad_player_rate_projection | 25 | 19.5% |
| Paper common props | bad_opportunity_projection | 24 | 18.8% |
| Watch props | model_worse_than_market_price | 24 | 1.7% |
| Watch props | bad_side_probability | 22 | 1.5% |
| Watch props | bad_distribution_pricing | 22 | 1.5% |
| Micro projection props | bad_player_projection | 16 | 72.7% |
| Lottery props | lost_clv | 8 | 22.9% |
| Lottery props | bad_player_rate_projection | 7 | 20.0% |
| Lottery props | bad_calibration_bucket | 7 | 20.0% |
| Watch props | unclassified_miss | 6 | 0.4% |
| One-sided FanDuel props | weak_market_bucket | 6 | 0.0% |
| Paper common props | bad_side_probability | 5 | 3.9% |
| Lottery props | bad_player_projection | 5 | 14.3% |
| Lottery props | weak_market_bucket | 3 | 8.6% |
| Lottery props | bad_opportunity_projection | 3 | 8.6% |
| Lottery props | model_worse_than_market_price | 2 | 5.7% |
| Micro projection props | bad_player_rate_projection | 2 | 9.1% |
| Paper common props | bad_distribution_pricing | 1 | 0.8% |
| One-sided FanDuel props | bad_side_probability | 1 | 0.0% |
| Micro projection props | lost_clv | 1 | 4.5% |
| Micro projection props | model_worse_than_market_price | 1 | 4.5% |
| Micro projection props | bad_calibration_bucket | 1 | 4.5% |
| Micro projection props | bad_opportunity_projection | 1 | 4.5% |

## Miss Reasons By Market/Side

| Reason | Market/Side | Misses |
|---|---|---:|
| weak_market_bucket | batter_total_bases|over | 10708 |
| bad_bucket_roi | batter_total_bases|over | 10675 |
| bad_calibration_bucket | batter_total_bases|over | 6991 |
| bad_player_rate_projection | batter_total_bases|over | 4515 |
| weak_market_bucket | batter_home_runs|over | 4342 |
| bad_bucket_roi | batter_home_runs|over | 4332 |
| weak_market_bucket | batter_hits|over | 3414 |
| lost_clv | batter_total_bases|over | 3411 |
| bad_bucket_roi | batter_hits|over | 3355 |
| bad_calibration_bucket | batter_home_runs|over | 2389 |
| bad_opportunity_projection | batter_total_bases|over | 1429 |
| lost_clv | batter_home_runs|over | 1175 |
| lost_clv | batter_hits|over | 970 |
| bad_clv_bookability | batter_total_bases|over | 950 |
| bad_calibration_bucket | batter_hits|over | 906 |
| bad_player_projection | batter_total_bases|over | 762 |
| large_projection_error | batter_total_bases|over | 742 |
| model_worse_than_market_price | batter_hits|over | 699 |
| model_worse_than_market_price | batter_total_bases|over | 688 |
| weak_market_bucket | batter_hits|under | 659 |
| bad_player_projection | batter_hits|over | 657 |
| bad_opportunity_projection | batter_home_runs|over | 652 |
| bad_line_price_edge | batter_hits|over | 588 |
| bad_side_probability | batter_hits|over | 565 |
| bad_line_price_edge | batter_total_bases|over | 514 |
| bad_bucket_roi | batter_hits|under | 502 |
| bad_opportunity_projection | batter_hits|over | 474 |
| bad_clv_bookability | batter_home_runs|over | 459 |
| bad_player_rate_projection | batter_hits|over | 424 |
| bad_clv_bookability | batter_hits|over | 379 |
| bad_distribution_pricing | batter_hits|over | 327 |
| weak_market_bucket | batter_total_bases|under | 281 |
| bad_side_probability | batter_total_bases|under | 249 |
| model_worse_than_market_price | batter_home_runs|over | 207 |
| lost_clv | batter_hits|under | 191 |
| weak_market_bucket | pitcher_strikeouts|over | 188 |
| bad_bucket_roi | batter_total_bases|under | 176 |
| bad_calibration_bucket | batter_total_bases|under | 176 |
| bad_player_rate_projection | batter_hits|under | 157 |
| bad_line_price_edge | batter_home_runs|over | 155 |

## Projection vs Betting Accuracy by Market/Side

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_total_bases|over | 13505 | 20.7% | 27.7% | 0.159 | -19.0% | 1.418 | 1.846 | 12427 | 41.3% | +0.15 |
| batter_hits|over | 4719 | 27.7% | 35.7% | 0.159 | -7.8% | 0.711 | 0.885 | 4238 | 39.6% | +0.08 |
| batter_home_runs|over | 4655 | 6.7% | 12.6% | 0.066 | -34.7% | 0.277 | 0.368 | 4175 | 36.4% | +0.05 |
| batter_hits|under | 1119 | 41.1% | 44.2% | 0.236 | -3.4% | 0.714 | 0.910 | 864 | 35.4% | -0.09 |
| batter_total_bases|under | 660 | 57.4% | 63.7% | 0.249 | -4.0% | 1.546 | 2.008 | 560 | 35.9% | -0.07 |
| pitcher_strikeouts|under | 331 | 50.8% | 52.4% | 0.247 | 2.0% | 1.542 | 1.985 | 261 | 48.7% | +0.68 |
| pitcher_strikeouts|over | 316 | 40.5% | 52.8% | 0.260 | -15.3% | 1.986 | 2.430 | 259 | 35.1% | -0.24 |

## Weak Exact Buckets

| Bucket | Rows | Win | Avg Prob | Brier | ROI | MAE | RMSE | CLV Rows | CLV Beat | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batter_home_runs|over|common|HR 0.5|plus_150_249|fanduel | 10 | 0.0% | 49.7% | - | -100.0% | 0.341 | 0.361 | 10 | 30.0% | -0.08 |
| batter_hits|over|alt_tail|H 2.5+|plus_250_499|fanduel | 2 | 0.0% | 23.0% | - | -100.0% | 1.040 | 1.112 | 2 | 0.0% | +0.00 |
| batter_hits|over|common|H 1.5|fair_lay|fanduel | 2 | 0.0% | 53.6% | - | -100.0% | 1.115 | 1.266 | 2 | 50.0% | -0.72 |
| batter_hits|under|common|H 1.5|fair_lay|draftkings | 2 | 0.0% | 54.7% | - | -100.0% | 1.210 | 1.382 | 2 | 50.0% | -0.68 |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 1 | 0.0% | 53.0% | - | -100.0% | 0.279 | 0.279 | 0 | - | - |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|fanduel | 1 | 0.0% | 53.0% | - | -100.0% | 0.279 | 0.279 | 1 | 0.0% | -1.27 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_150_180|fanduel | 1 | 0.0% | 71.2% | - | -100.0% | 3.010 | 3.010 | 1 | 0.0% | +0.00 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_130_149|fanduel | 5 | 20.0% | 63.7% | 0.424 | -64.8% | 3.076 | 3.474 | 5 | 40.0% | -0.70 |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_100_149|fanduel | 6 | 16.7% | 48.8% | 0.223 | -60.0% | 2.905 | 4.406 | 6 | 16.7% | -0.04 |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|fanduel | 5 | 20.0% | 47.4% | 0.230 | -58.0% | 2.199 | 2.386 | 3 | 0.0% | -1.01 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 22 | 27.3% | 53.3% | 0.266 | -47.6% | 2.153 | 2.404 | 21 | 33.3% | -1.79 |
| pitcher_strikeouts|over|common|K 4.5-6.0|lay_150_180|draftkings | 3 | 33.3% | 57.9% | 0.290 | -44.6% | 2.858 | 3.766 | 2 | 0.0% | -2.81 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 26 | 26.9% | 52.6% | 0.280 | -44.1% | 1.923 | 2.325 | 20 | 25.0% | -0.40 |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 6 | 33.3% | 89.1% | 0.484 | -42.8% | 3.746 | 4.755 | 6 | 33.3% | -0.54 |
| batter_hits|over|common|H 1.5|plus_100_149|fanduel | 58 | 25.9% | 50.1% | 0.249 | -41.7% | 0.870 | 1.062 | 53 | 43.4% | -0.32 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | 1958 | 0.8% | 2.2% | 0.008 | -39.2% | 0.288 | 0.374 | 1714 | 36.3% | +0.02 |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|fanduel | 28 | 28.6% | 51.3% | 0.278 | -39.1% | 1.891 | 2.350 | 26 | 23.1% | -0.67 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | 1613 | 7.7% | 14.5% | 0.077 | -38.3% | 0.217 | 0.308 | 1449 | 34.5% | +0.07 |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 29 | 27.6% | 47.7% | 0.236 | -34.9% | 0.669 | 0.886 | 27 | 29.6% | -0.61 |
| batter_total_bases|over|common|TB 2.5+|fair_lay|fanduel | 3 | 33.3% | 57.3% | 0.239 | -34.9% | 2.108 | 2.296 | 3 | 33.3% | -0.81 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|fanduel | 23 | 34.8% | 59.3% | 0.312 | -33.6% | 2.055 | 2.383 | 23 | 39.1% | -0.67 |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 25 | 32.0% | 52.4% | 0.257 | -32.3% | 0.671 | 0.808 | 23 | 69.6% | +1.79 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 22 | 31.8% | 49.7% | 0.246 | -31.7% | 0.995 | 1.272 | 18 | 38.9% | +0.53 |
| batter_hits|under|common|H 1.5|lay_130_149|draftkings | 5 | 40.0% | 63.1% | 0.245 | -31.7% | 0.954 | 1.062 | 5 | 0.0% | -1.04 |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 64 | 18.8% | 30.2% | 0.165 | -30.6% | 1.095 | 1.295 | 53 | 56.6% | +0.54 |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 43 | 32.6% | 48.4% | 0.251 | -27.5% | 1.482 | 1.838 | 29 | 65.5% | +1.86 |
| batter_total_bases|over|common|TB 1.5|lay_150_180|fanduel | 11 | 45.5% | 75.1% | 0.425 | -26.5% | 2.336 | 3.544 | 11 | 9.1% | -2.25 |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 14 | 35.7% | 55.9% | 0.255 | -26.1% | 0.612 | 0.677 | 13 | 61.5% | +2.19 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 17 | 29.4% | 43.4% | 0.223 | -26.1% | 1.318 | 1.575 | 11 | 36.4% | +0.66 |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 103 | 32.0% | 50.0% | 0.261 | -25.8% | 2.016 | 2.480 | 102 | 48.0% | +0.25 |

## Recent Losing Examples

| Date | Ledger | Player | Bet | Price | Pred | Actual | PA | BF | Model | Market | Primary | Labels |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2026-06-03 | Paper common props | Alex Bregman | batter_hits under 0.5 draftkings | 168.0 | 0.938 | 1.000 | 4.4->5.0 | -->- | 41.8% | 35.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Jeff McNeil | batter_hits under 0.5 draftkings | 153.0 | 0.742 | 1.000 | 4.0->4.0 | -->- | 41.8% | 37.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Miguel Amaya | batter_hits under 0.5 draftkings | -102.0 | 0.483 | 1.000 | 2.8->4.0 | -->- | 59.5% | 47.2% | bad_player_projection | bad_bucket_roi, bad_calibration_bucket, bad_clv_bookability, bad_player_projection, bad_side_probability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Miguel Amaya | batter_hits over 2.5 fanduel | 3500.0 | 0.483 | 1.000 | 2.8->4.0 | -->- | 3.1% | 2.8% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Carlos Cortes | batter_hits under 0.5 draftkings | 184.0 | 0.773 | 1.000 | 3.7->3.0 | -->- | 41.8% | 33.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Brent Rooker | batter_hits under 0.5 draftkings | 151.0 | 0.849 | 1.000 | 4.1->4.0 | -->- | 41.8% | 37.3% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Brent Rooker | batter_hits over 2.5 fanduel | 1500.0 | 0.849 | 1.000 | 4.1->4.0 | -->- | 7.2% | 6.2% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Seiya Suzuki | batter_hits under 0.5 draftkings | 162.0 | 0.830 | 1.000 | 3.8->4.0 | -->- | 39.1% | 35.8% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Michael Busch | batter_hits over 1.5 fanduel | 360.0 | 0.917 | 1.000 | 4.4->4.0 | -->- | 24.4% | 21.7% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Michael Busch | batter_hits over 2.5 fanduel | 2200.0 | 0.917 | 1.000 | 4.4->4.0 | -->- | 5.0% | 4.3% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Tyler Soderstrom | batter_hits under 0.5 draftkings | 179.0 | 0.879 | 2.000 | 4.0->5.0 | -->- | 41.8% | 33.6% | bad_player_rate_projection | bad_bucket_roi, bad_clv_bookability, bad_player_rate_projection, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Pete Crow-Armstrong | batter_hits over 2.5 fanduel | 1500.0 | 0.965 | 1.000 | 4.4->5.0 | -->- | 8.0% | 6.2% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Pete Crow-Armstrong | batter_hits under 0.5 draftkings | 167.0 | 0.965 | 1.000 | 4.4->5.0 | -->- | 39.1% | 35.1% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Nick Kurtz | batter_hits under 0.5 draftkings | 173.0 | 0.862 | 1.000 | 4.2->5.0 | -->- | 41.8% | 34.3% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Alex Bregman | batter_home_runs over 1.5 fanduel | 7000.0 | 0.182 | 0.000 | 4.4->5.0 | -->- | 1.5% | 1.4% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Dansby Swanson | batter_home_runs over 1.5 fanduel | 10000.0 | 0.204 | 0.000 | 3.7->4.0 | -->- | 1.8% | 1.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Jeff McNeil | batter_home_runs over 1.5 fanduel | 35000.0 | 0.110 | 0.000 | 4.0->4.0 | -->- | 0.6% | 0.3% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Ian Happ | batter_home_runs over 1.5 fanduel | 5000.0 | 0.217 | 0.000 | 4.3->4.0 | -->- | 2.0% | 2.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Brent Rooker | batter_home_runs over 1.5 fanduel | 5000.0 | 0.250 | 0.000 | 4.1->4.0 | -->- | 2.6% | 2.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Seiya Suzuki | batter_home_runs over 1.5 fanduel | 5000.0 | 0.243 | 1.000 | 3.8->4.0 | -->- | 2.5% | 2.0% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Michael Busch | batter_home_runs over 1.5 fanduel | 8000.0 | 0.220 | 0.000 | 4.4->4.0 | -->- | 2.1% | 1.2% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Pete Crow-Armstrong | batter_home_runs over 1.5 fanduel | 6000.0 | 0.205 | 1.000 | 4.4->5.0 | -->- | 1.8% | 1.6% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Nick Kurtz | batter_home_runs over 1.5 fanduel | 3500.0 | 0.312 | 0.000 | 4.2->5.0 | -->- | 4.0% | 2.8% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Dansby Swanson | batter_total_bases over 4.5 fanduel | 1200.0 | 1.354 | 2.000 | 3.7->4.0 | -->- | 9.6% | 7.7% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Dansby Swanson | batter_total_bases over 2.5 fanduel | 360.0 | 1.354 | 2.000 | 3.7->4.0 | -->- | 23.3% | 21.7% | bad_clv_bookability | bad_bucket_roi, bad_calibration_bucket, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Jeff McNeil | batter_total_bases over 2.5 fanduel | 440.0 | 1.256 | 1.000 | 4.0->4.0 | -->- | 23.3% | 18.5% | bad_clv_bookability | bad_bucket_roi, bad_calibration_bucket, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Jeff McNeil | batter_total_bases over 4.5 fanduel | 1800.0 | 1.256 | 1.000 | 4.0->4.0 | -->- | 8.9% | 5.3% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Jeff McNeil | batter_total_bases over 3.5 fanduel | 800.0 | 1.256 | 1.000 | 4.0->4.0 | -->- | 13.1% | 11.1% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
| 2026-06-03 | Paper common props | Nico Hoerner | batter_total_bases under 1.5 draftkings | -124.0 | 1.500 | 2.000 | 4.4->5.0 | -->- | 64.8% | 51.7% | bad_player_projection | bad_bucket_roi, bad_calibration_bucket, bad_clv_bookability, bad_player_projection, bad_side_probability, weak_market_bucket |
| 2026-06-03 | One-sided FanDuel props | Miguel Amaya | batter_total_bases over 4.5 fanduel | 2000.0 | 0.860 | 1.000 | 2.8->4.0 | -->- | 4.9% | 4.8% | bad_clv_bookability | bad_bucket_roi, bad_clv_bookability, weak_market_bucket |
