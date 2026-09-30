# MLB Prop TB 1.5 Close Repair Report

Generated UTC: 2026-07-30T03:11:26Z
Target: `draftkings|batter_total_bases|over|TB 1.5|plus-money`
Status: **ready**

- Plus-money rows: 4995
- Valid exact closes: 3986 (79.8%)
- Additional valid closes needed for 90%: 510
- Stale close rate: 2.5%
- CLV beat: 48.1%
- Avg CLV: 0.374
- ROI: -8.2%

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| valid_close_coverage | 510 | DraftKings TB 1.5 plus-money needs more exact same-book closes before promotion proof counts | keep targeted close capture active every 10 minutes inside T-120 and verify offer counts for DK TB 1.5 before first pitch |
| fallback_other_book_only | 441 | close proof blocker for DK TB 1.5 plus-money | inspect close resolver taxonomy for this bucket |
| player_prop_unavailable_at_close | 334 | close proof blocker for DK TB 1.5 plus-money | player prop menu disappeared; bookability problem |
| stale_close_before_lock | 126 | close proof blocker for DK TB 1.5 plus-money | ordering bug; close timestamp was before the lock and must not count |
| close_outside_two_hour_window | 105 | close proof blocker for DK TB 1.5 plus-money | scheduler timing/capture window miss; add or verify T-120/T-60/T-20/T-10 event-aware close rows |
| player_market_unavailable_at_close | 2 | close proof blocker for DK TB 1.5 plus-money | market disappeared for that player; bookability problem |
| no_valid_close_snapshot | 1 | close proof blocker for DK TB 1.5 plus-money | inspect close resolver taxonomy for this bucket |

## By Price Bucket

| Price Bucket | Rows | Valid | Coverage | Need 90 | CLV Beat | Avg CLV | ROI | Reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fair_lay | 716 | 677 | 94.6% | 0 | 47.6% | 0.219 | -10.2% | {"close_outside_two_hour_window": 8, "player_prop_unavailable_at_close": 23, "stale_close_before_lock": 8, "valid_close": 677} |
| lay_130_149 | 116 | 101 | 87.1% | 4 | 51.5% | -0.061 | -15.7% | {"close_outside_two_hour_window": 3, "line_disappeared_at_close": 5, "player_prop_unavailable_at_close": 6, "stale_close_before_lock": 1, "valid_close": 101} |
| lay_150_180 | 21 | 8 | 38.1% | 11 | 37.5% | -0.419 | 29.6% | {"close_outside_two_hour_window": 2, "line_disappeared_at_close": 8, "player_prop_unavailable_at_close": 3, "valid_close": 8} |
| plus_100_149 | 4731 | 3837 | 81.1% | 421 | 47.6% | 0.351 | -7.7% | {"close_outside_two_hour_window": 103, "fallback_other_book_only": 367, "no_valid_close_snapshot": 1, "player_market_unavailable_at_close": 2, "player_prop_unavailable_at_close": 304, "stale_close_before_lock": 117, "valid_close": 3837} |
| plus_150_249 | 263 | 149 | 56.7% | 88 | 61.7% | 0.969 | -17.7% | {"close_outside_two_hour_window": 2, "fallback_other_book_only": 74, "player_prop_unavailable_at_close": 30, "stale_close_before_lock": 8, "valid_close": 149} |
| plus_250_499 | 1 | 0 | 0.0% | 1 | - | - | - | {"stale_close_before_lock": 1} |

## Plus-Money By Date

| Date | Rows | Valid | Coverage | Need 90 | CLV Beat | Avg CLV | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-22 | 125 | 106 | 84.8% | 7 | 48.1% | 0.121 | -11.8% |
| 2026-06-06 | 114 | 92 | 80.7% | 11 | 50.0% | 0.367 | -22.8% |
| 2026-06-07 | 114 | 95 | 83.3% | 8 | 41.1% | 0.277 | -17.9% |
| 2026-07-10 | 114 | 99 | 86.8% | 4 | 52.5% | 0.483 | -8.3% |
| 2026-06-05 | 113 | 94 | 83.2% | 8 | 47.9% | 0.572 | -10.2% |
| 2026-06-26 | 112 | 88 | 78.6% | 13 | 46.6% | 0.300 | -5.5% |
| 2026-06-14 | 111 | 86 | 77.5% | 14 | 52.3% | 0.423 | -6.8% |
| 2026-06-12 | 110 | 93 | 84.5% | 6 | 54.8% | 0.701 | -2.2% |
| 2026-07-04 | 110 | 87 | 79.1% | 12 | 48.3% | 0.478 | -3.6% |
| 2026-07-05 | 108 | 88 | 81.5% | 10 | 52.3% | 0.401 | 7.8% |
| 2026-07-28 | 108 | 85 | 78.7% | 13 | 49.4% | 0.274 | -7.3% |
| 2026-06-30 | 107 | 91 | 85.0% | 6 | 48.4% | 0.403 | -9.6% |
| 2026-06-03 | 106 | 0 | 0.0% | 96 | - | - | 0.5% |
| 2026-07-18 | 106 | 89 | 84.0% | 7 | 50.6% | 0.492 | 1.5% |
| 2026-06-13 | 105 | 88 | 83.8% | 7 | 52.3% | 0.513 | -21.1% |
| 2026-07-25 | 105 | 88 | 83.8% | 7 | 46.6% | 0.062 | -32.9% |
| 2026-06-10 | 104 | 84 | 80.8% | 10 | 51.2% | 0.512 | -13.7% |
| 2026-07-08 | 102 | 84 | 82.4% | 8 | 50.0% | 0.646 | -8.7% |
| 2026-07-17 | 102 | 80 | 78.4% | 12 | 58.8% | 0.721 | -7.0% |
| 2026-06-17 | 101 | 88 | 87.1% | 3 | 38.6% | 0.328 | 0.1% |
| 2026-06-28 | 101 | 80 | 79.2% | 11 | 30.0% | 0.253 | -30.3% |
| 2026-07-24 | 101 | 80 | 79.2% | 11 | 47.5% | 0.088 | 0.0% |
| 2026-06-27 | 100 | 88 | 88.0% | 2 | 48.9% | 0.385 | 2.0% |
| 2026-07-11 | 100 | 82 | 82.0% | 8 | 47.6% | 0.213 | -21.2% |
| 2026-07-21 | 100 | 77 | 77.0% | 13 | 58.4% | 0.604 | -10.8% |
| 2026-06-24 | 99 | 72 | 72.7% | 18 | 31.9% | -0.199 | -6.4% |
| 2026-07-01 | 99 | 74 | 74.7% | 16 | 54.1% | 0.718 | -8.9% |
| 2026-07-26 | 99 | 81 | 81.8% | 9 | 46.9% | 0.140 | -0.9% |
| 2026-07-12 | 97 | 90 | 92.8% | 0 | 38.9% | 0.064 | -9.7% |
| 2026-06-16 | 96 | 83 | 86.5% | 4 | 59.0% | 0.501 | -20.3% |
| 2026-07-07 | 96 | 81 | 84.4% | 6 | 49.4% | 0.522 | -41.0% |
| 2026-06-09 | 95 | 85 | 89.5% | 1 | 55.3% | 0.619 | 19.0% |
| 2026-06-29 | 95 | 79 | 83.2% | 7 | 60.8% | 0.891 | -4.4% |
| 2026-07-09 | 95 | 79 | 83.2% | 7 | 41.8% | 0.363 | -16.2% |
| 2026-07-19 | 95 | 81 | 85.3% | 5 | 54.3% | 0.718 | -7.2% |
| 2026-06-20 | 94 | 75 | 79.8% | 10 | 44.0% | 0.157 | 0.4% |
| 2026-07-20 | 94 | 80 | 85.1% | 5 | 41.2% | 0.027 | 5.4% |
| 2026-06-19 | 93 | 74 | 79.6% | 10 | 54.1% | 0.703 | 9.0% |
| 2026-06-23 | 93 | 34 | 36.6% | 50 | 35.3% | -0.153 | 8.6% |
| 2026-06-21 | 92 | 74 | 80.4% | 9 | 51.4% | 0.307 | -4.2% |
| 2026-07-03 | 91 | 78 | 85.7% | 4 | 50.0% | 0.447 | -8.0% |
| 2026-07-29 | 91 | 77 | 84.6% | 5 | 51.9% | 0.414 | - |
| 2026-06-22 | 86 | 70 | 81.4% | 8 | 45.7% | 0.286 | -7.2% |
| 2026-07-27 | 85 | 74 | 87.1% | 3 | 55.4% | 0.442 | -0.5% |
| 2026-06-15 | 82 | 71 | 86.6% | 3 | 43.7% | -0.024 | -25.0% |
| 2026-06-11 | 68 | 59 | 86.8% | 3 | 39.0% | 0.260 | 2.4% |
| 2026-06-18 | 61 | 42 | 68.9% | 13 | 57.1% | 0.905 | -13.4% |
| 2026-07-02 | 60 | 49 | 81.7% | 5 | 44.9% | 0.463 | 8.1% |
| 2026-06-08 | 58 | 48 | 82.8% | 5 | 41.7% | 0.151 | -15.1% |
| 2026-07-06 | 56 | 49 | 87.5% | 2 | 55.1% | 0.503 | 2.2% |
| 2026-06-25 | 44 | 38 | 86.4% | 2 | 34.2% | 0.394 | -18.4% |
| 2026-06-04 | 43 | 23 | 53.5% | 16 | 13.0% | -0.031 | -13.4% |
| 2026-07-23 | 25 | 21 | 84.0% | 2 | 42.9% | -0.013 | -24.4% |
| 2026-06-01 | 23 | 22 | 95.7% | 0 | 22.7% | -1.391 | -42.3% |
| 2026-07-16 | 9 | 9 | 100.0% | 0 | 33.3% | 0.593 | -6.0% |
| 2026-06-02 | 2 | 2 | 100.0% | 0 | 50.0% | -0.025 | 129.5% |
