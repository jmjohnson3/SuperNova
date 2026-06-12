# MLB Prop Bucket Promotion Report

Generated: 2026-06-11T15:26:02Z
Lookback days: 365
Training rows: 38710
Exact buckets: 123
Promotable buckets: 0
Scope: exact bucket only (market | side | line surface | line bucket | price bucket | book).

## CLV Match Coverage

| Method | Rows |
| --- | --- |
| same_book_exact_line_snapshot | 33100 |

## CLV Validity

| Status | Rows |
| --- | --- |
| valid_movement | 22977 |
| true_no_movement | 10123 |
| unknown | 5610 |

## CLV Unknown Reasons

| Reason | Rows |
| --- | --- |
| stale_close_before_lock | 3685 |
| close_outside_two_hour_window | 1607 |
| fallback_other_book_only | 283 |
| no_valid_close_snapshot | 35 |

## Blocking Metrics

| Metric | Blocks |
| --- | --- |
| sample | 454 |
| other | 421 |
| clv | 360 |
| concentration | 221 |
| roi | 78 |
| calibration | 73 |

## Closest Common Buckets

| Bucket | Status | Ladder | Next | Source | Graded | Need Rows | CLV Rows | Dates | Clean Dates | ROI | CLV Beat | Avg CLV | Cal Err | Metric Gaps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | blocked | watch | watch | closed | 1128 | 0 | 901 / need 0 | 10 / need 0 | 7 / need 5 | -0.4% | 36.7% | +0.10pp | -1.2% | ROI needs > 0.0%, currently -0.4%; CLV beat needs 55.0%, currently 36.7%; bootstrap_roi<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|under|common|TB 1.5|lay_150_180|draftkings | blocked | watch | watch | closed | 713 | 0 | 611 / need 0 | 11 / need 0 | 7 / need 5 | +1.4% | 30.1% | -0.29pp | -0.4% | CLV beat needs 55.0%, currently 30.1%; avg CLV needs > +0.00pp, currently -0.29pp; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | blocked | watch | watch | closed | 353 | 0 | 297 / need 0 | 9 / need 0 | 6 / need 5 | +0.4% | 48.1% | +0.61pp | +5.2% | CLV beat needs 55.0%, currently 48.1%; calibration needs within 5.0%, currently +5.2%; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| batter_home_runs|over|common|HR 0.5|plus_500_plus|fanduel | blocked | watch | watch | closed | 1917 | 0 | 1632 / need 0 | 10 / need 0 | 7 / need 5 | -14.9% | 34.4% | +0.03pp | +0.0% | ROI needs > 0.0%, currently -14.9%; CLV beat needs 55.0%, currently 34.4%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|heavy_lay|fanduel | blocked | watch | watch | closed | 1757 | 0 | 1550 / need 0 | 9 / need 0 | 6 / need 5 | -5.8% | 35.1% | +0.00pp | +0.7% | ROI needs > 0.0%, currently -5.8%; CLV beat needs 55.0%, currently 35.1%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | blocked | watch | watch | closed | 1732 | 0 | 1487 / need 0 | 8 / need 0 | 6 / need 5 | -18.7% | 44.3% | +0.18pp | -3.3% | ROI needs > 0.0%, currently -18.7%; CLV beat needs 55.0%, currently 44.3%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 2.5|plus_250_499|fanduel | blocked | watch | watch | closed | 1728 | 0 | 1470 / need 0 | 8 / need 0 | 6 / need 5 | -10.2% | 39.0% | +0.06pp | -1.0% | ROI needs > 0.0%, currently -10.2%; CLV beat needs 55.0%, currently 39.0%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | blocked | watch | watch | closed | 1424 | 0 | 1159 / need 0 | 9 / need 0 | 6 / need 5 | -7.8% | 36.1% | +0.00pp | +1.1% | ROI needs > 0.0%, currently -7.8%; CLV beat needs 55.0%, currently 36.1%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|fanduel | blocked | watch | watch | closed | 1365 | 0 | 1200 / need 0 | 10 / need 0 | 7 / need 5 | -15.3% | 37.3% | +0.09pp | -1.9% | ROI needs > 0.0%, currently -15.3%; CLV beat needs 55.0%, currently 37.3%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | blocked | watch | watch | closed | 1216 | 0 | 992 / need 0 | 11 / need 0 | 7 / need 5 | -10.6% | 42.4% | +0.29pp | +0.7% | ROI needs > 0.0%, currently -10.6%; CLV beat needs 55.0%, currently 42.4%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | blocked | watch | watch | closed | 1015 | 0 | 851 / need 0 | 11 / need 0 | 7 / need 5 | -13.6% | 41.1% | +0.19pp | -1.4% | ROI needs > 0.0%, currently -13.6%; CLV beat needs 55.0%, currently 41.1%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_home_runs|over|common|HR 0.5|plus_250_499|fanduel | blocked | watch | watch | closed | 908 | 0 | 824 / need 0 | 10 / need 0 | 7 / need 5 | -6.7% | 33.6% | +0.03pp | +0.8% | ROI needs > 0.0%, currently -6.7%; CLV beat needs 55.0%, currently 33.6%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 1.5|plus_150_249|fanduel | blocked | watch | watch | closed | 897 | 0 | 804 / need 0 | 8 / need 0 | 6 / need 5 | -12.4% | 36.4% | +0.05pp | -0.8% | ROI needs > 0.0%, currently -12.4%; CLV beat needs 55.0%, currently 36.4%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 2.5|plus_150_249|fanduel | blocked | watch | watch | closed | 817 | 0 | 747 / need 0 | 8 / need 0 | 6 / need 5 | -21.6% | 38.0% | +0.12pp | -4.3% | ROI needs > 0.0%, currently -21.6%; CLV beat needs 55.0%, currently 38.0%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | blocked | watch | watch | closed | 727 | 0 | 621 / need 0 | 8 / need 0 | 6 / need 5 | -8.5% | 41.7% | +0.09pp | -1.6% | ROI needs > 0.0%, currently -8.5%; CLV beat needs 55.0%, currently 41.7%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | blocked | watch | watch | closed | 619 | 0 | 533 / need 0 | 9 / need 0 | 6 / need 5 | -5.4% | 40.9% | +0.16pp | +1.9% | ROI needs > 0.0%, currently -5.4%; CLV beat needs 55.0%, currently 40.9%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|under|common|H 1.5|heavy_lay|draftkings | blocked | watch | watch | closed | 421 | 0 | 342 / need 0 | 8 / need 0 | 6 / need 5 | +0.6% | 36.0% | -0.19pp | +1.6% | CLV beat needs 55.0%, currently 36.0%; avg CLV needs > +0.00pp, currently -0.19pp; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | blocked | watch | watch | closed | 385 | 0 | 309 / need 0 | 8 / need 0 | 6 / need 5 | -18.6% | 38.8% | +0.18pp | -1.5% | ROI needs > 0.0%, currently -18.6%; CLV beat needs 55.0%, currently 38.8%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | blocked | watch | watch | closed | 264 | 0 | 225 / need 0 | 9 / need 0 | 6 / need 5 | -5.9% | 42.2% | +0.17pp | +0.6% | ROI needs > 0.0%, currently -5.9%; CLV beat needs 55.0%, currently 42.2%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|under|common|TB 1.5|fair_lay|draftkings | blocked | watch | watch | closed | 222 | 0 | 210 / need 0 | 10 / need 0 | 7 / need 5 | -7.2% | 40.5% | +0.08pp | -9.1% | ROI needs > 0.0%, currently -7.2%; CLV beat needs 55.0%, currently 40.5%; calibration needs within 5.0%, currently -9.1%; bootstrap_roi<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| batter_total_bases|over|common|TB 2.5|plus_500_plus|fanduel | blocked | watch | watch | closed | 198 | 0 | 161 / need 0 | 8 / need 0 | 6 / need 5 | -16.9% | 34.2% | +0.23pp | -2.0% | ROI needs > 0.0%, currently -16.9%; CLV beat needs 55.0%, currently 34.2%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | blocked | watch | watch | closed | 173 | 0 | 152 / need 0 | 11 / need 0 | 7 / need 5 | -10.8% | 43.4% | +0.63pp | -1.5% | ROI needs > 0.0%, currently -10.8%; CLV beat needs 55.0%, currently 43.4%; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | blocked | watch | watch | closed | 110 | 40 | 80 / need 0 | 10 / need 0 | 6 / need 5 | -5.9% | 60.0% | +0.98pp | -4.5% | needs 40 more rows; ROI needs > 0.0%, currently -5.9%; bootstrap_rows<150; bootstrap_roi<0.000 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | blocked | watch | watch | closed | 65 | 85 | 40 / need 0 | 10 / need 0 | 6 / need 5 | +28.4% | 52.5% | +0.39pp | +9.4% | needs 85 more rows; CLV beat needs 55.0%, currently 52.5%; calibration needs within 5.0%, currently +9.4%; bootstrap_rows<150; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | blocked | watch | watch | closed | 49 | 101 | 42 / need 0 | 9 / need 0 | 5 / need 5 | +11.3% | 40.5% | +0.22pp | +8.7% | needs 101 more rows; CLV beat needs 55.0%, currently 40.5%; calibration needs within 5.0%, currently +8.7%; bootstrap_rows<150; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | blocked | watch | watch | closed | 45 | 105 | 40 / need 0 | 6 / need 0 | 5 / need 5 | +10.2% | 45.0% | +0.25pp | +14.2% | needs 105 more rows; CLV beat needs 55.0%, currently 45.0%; calibration needs within 5.0%, currently +14.2%; bootstrap_rows<150; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| batter_hits|under|common|H 0.5|plus_100_149|draftkings | blocked | watch | watch | closed | 1301 | 0 | 1121 / need 0 | 11 / need 0 | 7 / need 5 | -7.9% | 30.4% | -0.30pp | -2.2% | ROI needs > 0.0%, currently -7.9%; CLV beat needs 55.0%, currently 30.4%; avg CLV needs > +0.00pp, currently -0.30pp; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|under|common|TB 1.5|heavy_lay|draftkings | blocked | watch | watch | closed | 401 | 0 | 268 / need 0 | 11 / need 0 | 7 / need 5 | -6.7% | 25.4% | -0.45pp | -3.8% | ROI needs > 0.0%, currently -6.7%; CLV beat needs 55.0%, currently 25.4%; avg CLV needs > +0.00pp, currently -0.45pp; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 1.5|fair_lay|fanduel | blocked | watch | watch | closed | 281 | 0 | 263 / need 0 | 8 / need 0 | 6 / need 5 | -15.0% | 28.1% | -0.13pp | -0.4% | ROI needs > 0.0%, currently -15.0%; CLV beat needs 55.0%, currently 28.1%; avg CLV needs > +0.00pp, currently -0.13pp; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|under|common|TB 1.5|lay_130_149|draftkings | blocked | watch | watch | closed | 281 | 0 | 257 / need 0 | 10 / need 0 | 7 / need 5 | -8.2% | 36.6% | -0.03pp | -8.5% | ROI needs > 0.0%, currently -8.2%; CLV beat needs 55.0%, currently 36.6%; avg CLV needs > +0.00pp, currently -0.03pp; calibration needs within 5.0%, currently -8.5%; bootstrap_roi<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | blocked | watch | watch | closed | 220 | 0 | 209 / need 0 | 9 / need 0 | 6 / need 5 | -8.1% | 34.4% | -0.25pp | +3.5% | ROI needs > 0.0%, currently -8.1%; CLV beat needs 55.0%, currently 34.4%; avg CLV needs > +0.00pp, currently -0.25pp; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|under|common|H 0.5|fair_lay|draftkings | blocked | watch | watch | closed | 209 | 0 | 181 / need 0 | 11 / need 0 | 7 / need 5 | -14.0% | 24.3% | -0.62pp | -4.2% | ROI needs > 0.0%, currently -14.0%; CLV beat needs 55.0%, currently 24.3%; avg CLV needs > +0.00pp, currently -0.62pp; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|common|H 1.5|plus_500_plus|fanduel | blocked | watch | watch | closed | 132 | 18 | 105 / need 0 | 8 / need 0 | 6 / need 5 | -20.5% | 31.4% | +0.12pp | -3.2% | needs 18 more rows; ROI needs > 0.0%, currently -20.5%; CLV beat needs 55.0%, currently 31.4%; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|fanduel | blocked | watch | watch | closed | 93 | 57 | 68 / need 0 | 7 / need 0 | 5 / need 5 | -18.4% | 35.3% | +0.23pp | -10.7% | needs 57 more rows; ROI needs > 0.0%, currently -18.4%; CLV beat needs 55.0%, currently 35.3%; calibration needs within 5.0%, currently -10.7%; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | blocked | watch | watch | closed | 44 | 106 | 41 / need 0 | 6 / need 0 | 5 / need 5 | +36.6% | 19.5% | -0.44pp | +14.1% | needs 106 more rows; CLV beat needs 55.0%, currently 19.5%; avg CLV needs > +0.00pp, currently -0.44pp; calibration needs within 5.0%, currently +14.1%; bootstrap_rows<150; bootstrap_avg_clv_price<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | blocked | watch | watch | closed | 83 | 67 | 66 / need 0 | 8 / need 0 | 6 / need 5 | -29.0% | 40.9% | +0.37pp | -12.4% | needs 67 more rows; ROI needs > 0.0%, currently -29.0%; CLV beat needs 55.0%, currently 40.9%; calibration needs within 5.0%, currently -12.4%; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | blocked | watch | watch | closed | 55 | 95 | 31 / need 0 | 8 / need 0 | 6 / need 5 | -40.2% | 54.8% | +0.52pp | -12.2% | needs 95 more rows; ROI needs > 0.0%, currently -40.2%; CLV beat needs 55.0%, currently 54.8%; calibration needs within 5.0%, currently -12.2%; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | blocked | watch | watch | closed | 53 | 97 | 47 / need 0 | 10 / need 0 | 6 / need 5 | -6.0% | 36.2% | -0.73pp | +3.8% | needs 97 more rows; ROI needs > 0.0%, currently -6.0%; CLV beat needs 55.0%, currently 36.2%; avg CLV needs > +0.00pp, currently -0.73pp; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000 |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | blocked | watch | watch | closed | 49 | 101 | 42 / need 0 | 8 / need 0 | 5 / need 5 | -20.5% | 57.1% | -0.03pp | -7.7% | needs 101 more rows; ROI needs > 0.0%, currently -20.5%; avg CLV needs > +0.00pp, currently -0.03pp; calibration needs within 5.0%, currently -7.7%; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000 |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | blocked | watch | watch | closed | 49 | 101 | 40 / need 0 | 7 / need 0 | 5 / need 5 | -3.0% | 22.5% | -0.11pp | +1.1% | needs 101 more rows; ROI needs > 0.0%, currently -3.0%; CLV beat needs 55.0%, currently 22.5%; avg CLV needs > +0.00pp, currently -0.11pp; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000 |

## Alt-Line Lottery Watchlist

| Bucket | Status | Ladder | Next | Source | Graded | Need Rows | CLV Rows | Dates | Clean Dates | ROI | CLV Beat | Avg CLV | Cal Err | Metric Gaps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_hits|over|alt_tail|H 3.5+|plus_500_plus|fanduel | blocked | watch | watch | closed | 2641 | 0 | 2232 / need 0 | 8 / need 0 | 6 / need 5 | +12.9% | 36.4% | +0.02pp | +0.2% | CLV beat needs 55.0%, currently 36.4%; bootstrap_common_only; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|alt_tail|TB 4.5+|plus_500_plus|fanduel | blocked | watch | watch | closed | 2702 | 0 | 2338 / need 0 | 8 / need 0 | 6 / need 5 | -23.6% | 31.6% | +0.04pp | -1.9% | ROI needs > 0.0%, currently -23.6%; CLV beat needs 55.0%, currently 31.6%; bootstrap_common_only; bootstrap_roi<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|alt_tail|TB 3.5|plus_150_249|fanduel | blocked | watch | watch | closed | 164 | 0 | 154 / need 0 | 8 / need 0 | 6 / need 5 | +8.1% | 43.5% | +0.45pp | +5.2% | CLV beat needs 55.0%, currently 43.5%; calibration needs within 5.0%, currently +5.2%; bootstrap_common_only; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55; bootstrap_abs_calibration_error>0.050 |
| batter_home_runs|over|alt_tail|HR 1.5+|plus_500_plus|fanduel | blocked | watch | watch | closed | 2844 | 0 | 2457 / need 0 | 8 / need 0 | 6 / need 5 | -48.5% | 30.8% | +0.02pp | -0.4% | ROI needs > 0.0%, currently -48.5%; CLV beat needs 55.0%, currently 30.8%; bootstrap_common_only; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_hits|over|alt_tail|H 2.5|plus_500_plus|fanduel | blocked | watch | watch | closed | 2758 | 0 | 2399 / need 0 | 8 / need 0 | 6 / need 5 | -27.2% | 35.7% | +0.07pp | -1.3% | ROI needs > 0.0%, currently -27.2%; CLV beat needs 55.0%, currently 35.7%; bootstrap_common_only; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|alt_tail|TB 3.5|plus_250_499|fanduel | blocked | watch | watch | closed | 1408 | 0 | 1258 / need 0 | 8 / need 0 | 6 / need 5 | -16.1% | 41.0% | +0.05pp | -2.6% | ROI needs > 0.0%, currently -16.1%; CLV beat needs 55.0%, currently 41.0%; bootstrap_common_only; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|alt_tail|TB 3.5|plus_500_plus|fanduel | blocked | watch | watch | closed | 1247 | 0 | 1039 / need 0 | 8 / need 0 | 6 / need 5 | -12.6% | 29.6% | +0.04pp | -0.9% | ROI needs > 0.0%, currently -12.6%; CLV beat needs 55.0%, currently 29.6%; bootstrap_common_only; bootstrap_roi<0.000; bootstrap_avg_ev<0.000; bootstrap_clv_beat_rate<0.55 |
| batter_total_bases|over|alt_tail|TB 4.5+|plus_250_499|fanduel | blocked | watch | watch | closed | 127 | 23 | 123 / need 0 | 8 / need 0 | 6 / need 5 | -13.1% | 52.0% | +0.43pp | -0.8% | needs 23 more rows; ROI needs > 0.0%, currently -13.1%; CLV beat needs 55.0%, currently 52.0%; day concentration needs max 35.0%, currently 37.0%; bootstrap_common_only; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000 |
| batter_hits|over|alt_tail|H 2.5|plus_250_499|fanduel | blocked | watch | watch | closed | 19 | 131 | 19 / need 11 | 4 / need 1 | 4 / need 5 | +7.4% | 57.9% | -0.02pp | +4.2% | needs 131 more rows; needs 11 valid CLV closes; needs 1 more unique dates; avg CLV needs > +0.00pp, currently -0.02pp; day concentration needs max 35.0%, currently 42.1%; needs 1 more clean dates; bootstrap_common_only; bootstrap_rows<150; bootstrap_avg_ev<0.000; bootstrap_avg_clv_price<0.000 |
| batter_total_bases|over|alt_tail|TB 4.5+|plus_150_249|fanduel | blocked | watch | watch | closed | 2 | 148 | 2 / need 28 | 1 / need 4 | 1 / need 5 | -100.0% | 100.0% | +1.93pp | -29.2% | needs 148 more rows; needs 28 valid CLV closes; needs 4 more unique dates; ROI needs > 0.0%, currently -100.0%; calibration needs within 5.0%, currently -29.2%; day concentration needs max 35.0%, currently 100.0%; needs 4 more clean dates; bootstrap_common_only; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000 |
| batter_total_bases|over|alt_tail|TB 3.5|plus_100_149|fanduel | blocked | watch | watch | closed | 12 | 138 | 12 / need 18 | 2 / need 3 | 2 / need 5 | -0.4% | 33.3% | -0.10pp | +2.3% | needs 138 more rows; needs 18 valid CLV closes; needs 3 more unique dates; ROI needs > 0.0%, currently -0.4%; CLV beat needs 55.0%, currently 33.3%; avg CLV needs > +0.00pp, currently -0.10pp; day concentration needs max 35.0%, currently 50.0%; needs 3 more clean dates; bootstrap_common_only; bootstrap_rows<150; bootstrap_roi<0.000; bootstrap_avg_ev<0.000 |
