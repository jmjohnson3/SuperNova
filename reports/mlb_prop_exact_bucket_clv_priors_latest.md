# MLB Prop Exact-Bucket CLV Priors

Generated UTC: 2026-07-30T14:39:19Z
Status: **ready**
Evidence: true-paired, non-synthetic, valid-close rows only.

- Rows: 60949
- Global CLV beat: 40.7%
- Global avg CLV: 0.032
- Global valid close coverage: 88.9%
- Global stale close rate: 2.0%

| Bucket | True Pair | Valid CLV | Coverage | Stale | Dates | CLV Beat | Avg CLV | Trial Beat | Micro CLV | Close Quality | Blockers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| batter_total_bases|over|common|TB 1.5|plus_150_249|draftkings | 231 | 143 | 61.9% | 1.7% | 50 | 62.2% | 0.998 | 62.2% | True | valid_close_coverage_below_90 | - |
| pitcher_strikeouts|over|common|K 6.5-8.0|plus_100_149|draftkings | 114 | 85 | 74.6% | 3.5% | 42 | 65.9% | 1.138 | 65.9% | True | valid_close_coverage_below_90, stale_close_rate_above_2 | - |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings | 281 | 199 | 70.8% | 0.7% | 49 | 55.8% | 1.026 | 55.8% | True | valid_close_coverage_below_90 | - |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|draftkings | 158 | 111 | 70.3% | 3.2% | 45 | 55.9% | 1.051 | 55.9% | True | valid_close_coverage_below_90, stale_close_rate_above_2 | - |
| batter_hits|over|common|H 0.5|plus_100_149|draftkings | 88 | 76 | 86.4% | 1.1% | 36 | 55.3% | 1.083 | 55.3% | True | valid_close_coverage_below_90 | - |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|fanduel | 71 | 69 | 97.2% | 0.0% | 34 | 63.8% | 1.026 | 51.4% | False | - | micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 1.5|plus_150_249|draftkings | 1396 | 1091 | 78.2% | 1.7% | 52 | 51.8% | 0.453 | 51.8% | False | valid_close_coverage_below_90 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K 4.5-6.0|plus_100_149|draftkings | 398 | 289 | 72.6% | 1.5% | 51 | 52.2% | 0.797 | 52.2% | False | valid_close_coverage_below_90 | micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 1.5|plus_150_249|fanduel | 883 | 850 | 96.3% | 0.7% | 54 | 49.5% | 0.623 | 49.5% | False | - | micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|fair_lay|draftkings | 799 | 723 | 90.5% | 1.6% | 55 | 49.2% | 0.734 | 49.2% | False | - | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|draftkings | 24 | 20 | 83.3% | 8.3% | 19 | 75.0% | 0.943 | 47.6% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | clv_rows<30, micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 1.5|plus_100_149|draftkings | 4417 | 3779 | 85.6% | 2.4% | 55 | 47.6% | 0.350 | 47.6% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 1.5|plus_250_499|fanduel | 1991 | 1956 | 98.2% | 0.0% | 53 | 47.6% | 0.279 | 47.6% | False | - | micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 1.5|lay_130_149|draftkings | 110 | 100 | 90.9% | 0.9% | 42 | 52.0% | -0.055 | 52.0% | False | - | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| batter_total_bases|over|common|TB 2.5+|plus_100_149|draftkings | 20 | 17 | 85.0% | 0.0% | 12 | 76.5% | 0.914 | 47.0% | False | valid_close_coverage_below_90 | clv_rows<30, micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 1.5|fair_lay|draftkings | 692 | 671 | 97.0% | 1.2% | 54 | 47.5% | 0.220 | 47.5% | False | - | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|fanduel | 43 | 41 | 95.3% | 2.3% | 26 | 58.5% | 0.511 | 46.7% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_250_499|fanduel | 221 | 219 | 99.1% | 0.0% | 51 | 48.9% | 0.461 | 48.9% | False | - | micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|lay_130_149|draftkings | 1108 | 979 | 88.4% | 3.4% | 54 | 46.8% | 0.398 | 46.8% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|draftkings | 178 | 151 | 84.8% | 2.2% | 50 | 48.3% | -0.365 | 48.3% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|draftkings | 197 | 185 | 93.9% | 4.1% | 50 | 47.6% | 0.201 | 47.6% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 1.5|heavy_lay|fanduel | 16 | 16 | 100.0% | 0.0% | 12 | 68.8% | 0.416 | 45.4% | False | - | clv_rows<30, micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|plus_100_149|fanduel | 33 | 30 | 90.9% | 0.0% | 23 | 56.7% | 0.925 | 45.1% | False | - | micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|lay_150_180|draftkings | 2211 | 2000 | 90.5% | 2.2% | 54 | 45.0% | 0.216 | 45.0% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K <4.5|fair_lay|draftkings | 55 | 51 | 92.7% | 1.8% | 32 | 51.0% | 0.367 | 44.7% | False | - | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K <4.5|plus_100_149|fanduel | 138 | 116 | 84.1% | 1.4% | 44 | 47.4% | 0.647 | 47.4% | False | valid_close_coverage_below_90 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K 4.5-6.0|fair_lay|draftkings | 195 | 181 | 92.8% | 4.1% | 48 | 46.4% | -0.193 | 46.4% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|draftkings | 43 | 41 | 95.3% | 2.3% | 30 | 51.2% | -0.082 | 44.3% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|fanduel | 241 | 199 | 82.6% | 0.4% | 48 | 45.2% | 0.493 | 45.2% | False | valid_close_coverage_below_90 | micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 2.5+|plus_250_499|fanduel | 526 | 510 | 97.0% | 0.0% | 51 | 44.3% | 0.084 | 44.3% | False | - | micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 1.5|plus_100_149|draftkings | 169 | 157 | 92.9% | 1.2% | 38 | 45.2% | -0.005 | 45.2% | False | - | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| pitcher_strikeouts|over|common|K 6.5-8.0|lay_130_149|draftkings | 31 | 20 | 64.5% | 6.5% | 17 | 55.0% | -0.185 | 43.6% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | clv_rows<30, micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| batter_hits|over|common|H 0.5|lay_150_180|fanduel | 2529 | 2287 | 90.4% | 2.7% | 53 | 43.6% | 0.101 | 43.6% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| batter_hits|under|common|H 1.5|fair_lay|draftkings | 7 | 7 | 100.0% | 0.0% | 7 | 71.4% | 0.544 | 43.2% | False | - | clv_rows<30, micro_clv_beat_below_trial_gate |
| batter_total_bases|over|common|TB 1.5|plus_250_499|fanduel | 27 | 26 | 96.3% | 0.0% | 19 | 50.0% | 0.529 | 43.0% | False | - | clv_rows<30, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K 8.5+|fair_lay|draftkings | 3 | 3 | 100.0% | 0.0% | 3 | 100.0% | 2.840 | 42.8% | False | - | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|lay_130_149|fanduel | 872 | 781 | 89.6% | 3.1% | 53 | 43.0% | 0.231 | 43.0% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K <4.5|lay_130_149|draftkings | 55 | 50 | 90.9% | 1.8% | 30 | 46.0% | -0.428 | 42.7% | False | - | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| pitcher_strikeouts|under|common|K 4.5-6.0|lay_130_149|fanduel | 157 | 144 | 91.7% | 1.3% | 46 | 43.8% | 0.123 | 43.8% | False | - | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 6.5-8.0|lay_130_149|draftkings | 59 | 48 | 81.4% | 6.8% | 31 | 45.8% | -0.449 | 42.6% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| pitcher_strikeouts|over|common|K 6.5-8.0|fair_lay|fanduel | 43 | 42 | 97.7% | 2.3% | 31 | 45.2% | 0.021 | 42.3% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| batter_hits|under|common|H 1.5|lay_130_149|draftkings | 19 | 16 | 84.2% | 5.3% | 12 | 50.0% | -0.066 | 42.2% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | clv_rows<30, micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|heavy_lay|draftkings | 5057 | 4319 | 85.4% | 2.3% | 54 | 42.2% | -0.027 | 42.2% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| batter_total_bases|over|alt_tail|TB 2.5+|plus_150_249|fanduel | 21 | 21 | 100.0% | 0.0% | 18 | 47.6% | -0.171 | 42.1% | False | - | clv_rows<30, micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| batter_hits|over|common|H 1.5|fair_lay|fanduel | 7 | 7 | 100.0% | 0.0% | 6 | 57.1% | 0.894 | 42.0% | False | - | clv_rows<30, micro_clv_beat_below_trial_gate |
| batter_hits|over|common|H 0.5|fair_lay|fanduel | 329 | 295 | 89.7% | 2.7% | 53 | 42.0% | 0.281 | 42.0% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| batter_total_bases|under|common|TB 2.5+|fair_lay|draftkings | 3 | 3 | 100.0% | 0.0% | 3 | 66.7% | 0.780 | 41.6% | False | - | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K 8.5+|fair_lay|fanduel | 3 | 3 | 100.0% | 0.0% | 3 | 66.7% | -0.747 | 41.6% | False | - | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K 8.5+|plus_100_149|draftkings | 6 | 3 | 50.0% | 0.0% | 3 | 66.7% | 1.607 | 41.6% | False | valid_close_coverage_below_90 | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| batter_hits|under|common|H 0.5|plus_150_249|draftkings | 3749 | 3105 | 82.8% | 2.2% | 54 | 41.6% | 0.108 | 41.6% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K <4.5|lay_130_149|fanduel | 24 | 20 | 83.3% | 4.2% | 16 | 45.0% | 0.634 | 41.6% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | clv_rows<30, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 4.5-6.0|fair_lay|fanduel | 189 | 177 | 93.7% | 4.2% | 47 | 41.8% | 0.163 | 41.8% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|over|common|K <4.5|heavy_lay|fanduel | 1 | 1 | 100.0% | 0.0% | 1 | 100.0% | 0.240 | 41.4% | False | - | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 6.5-8.0|heavy_lay|fanduel | 2 | 1 | 50.0% | 0.0% | 1 | 100.0% | 0.240 | 41.4% | False | valid_close_coverage_below_90 | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 8.5+|lay_150_180|fanduel | 2 | 1 | 50.0% | 0.0% | 1 | 100.0% | 1.220 | 41.4% | False | valid_close_coverage_below_90 | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 8.5+|plus_100_149|draftkings | 1 | 1 | 100.0% | 0.0% | 1 | 100.0% | 13.590 | 41.4% | False | - | clv_rows<30, dates<4, micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 6.5-8.0|plus_100_149|draftkings | 50 | 30 | 60.0% | 4.0% | 23 | 43.3% | 0.250 | 41.4% | False | valid_close_coverage_below_90, stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K 6.5-8.0|fair_lay|fanduel | 41 | 40 | 97.6% | 2.4% | 29 | 42.5% | 0.138 | 41.3% | False | stale_close_rate_above_2 | micro_clv_beat_below_trial_gate |
| pitcher_strikeouts|under|common|K <4.5|lay_150_180|draftkings | 22 | 16 | 72.7% | 0.0% | 14 | 43.8% | -0.527 | 41.2% | False | valid_close_coverage_below_90 | clv_rows<30, micro_clv_beat_below_trial_gate, micro_avg_clv_not_positive |
| batter_hits|under|common|H 0.5|heavy_lay|draftkings | 4 | 4 | 100.0% | 0.0% | 4 | 50.0% | 0.158 | 41.1% | False | - | clv_rows<30, micro_clv_beat_below_trial_gate |
