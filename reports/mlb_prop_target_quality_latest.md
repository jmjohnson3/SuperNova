# MLB Prop Target Quality

Generated UTC: 2026-06-20T00:59:33Z
Rows: 94639
Date range: 2026-05-31 to 2026-06-19

## Required Field Coverage

| Field | Present | Missing | Coverage |
|---|---:|---:|---:|
| bookmaker_key | 94639 | 0 | 100.0% |
| market_line | 94639 | 0 | 100.0% |
| market_price | 94639 | 0 | 100.0% |
| prop_offer_id | 93982 | 657 | 99.3% |
| lock_snapshot_id | 94639 | 0 | 100.0% |
| source_created_at | 94639 | 0 | 100.0% |
| actual_value | 78285 | 16354 | 82.7% |
| won | 78285 | 16354 | 82.7% |
| paired_price | 94639 | 0 | 100.0% |
| paired_bookmaker_key | 94639 | 0 | 100.0% |
| paired_price_source | 94639 | 0 | 100.0% |
| pair_quality | 94639 | 0 | 100.0% |
| no_vig_market_prob | 94639 | 0 | 100.0% |
| market_prob_source | 94639 | 0 | 100.0% |
| closing_line | 78474 | 16165 | 82.9% |
| closing_price | 78474 | 16165 | 82.9% |
| closing_snapshot_id | 78474 | 16165 | 82.9% |
| closing_fetched_at_utc | 78474 | 16165 | 82.9% |
| clv_valid | 94639 | 0 | 100.0% |

## CLV / Close Status

| Status | Rows |
|---|---:|
| valid_movement | 55777 |
| true_no_movement | 22697 |
| unknown | 16165 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| none | 78474 |
| close_outside_two_hour_window | 7927 |
| stale_close_before_lock | 6303 |
| line_disappeared_at_close | 1476 |
| fallback_other_book_only | 285 |
| no_valid_close_snapshot | 174 |

## Quality By Date

| Date | Rows | Offer ID | Price+Lock | True Pair | Same-Book Pair | Cross-Book Pair | Synthetic Pair | Any Pair | Graded | Valid Close | Stale Close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-19 | 7929 | 100.0% | 100.0% | 43.4% | 29.9% | 13.4% | 56.6% | 100.0% | 0.0% | 87.4% | 2.3% |
| 2026-06-18 | 2660 | 100.0% | 100.0% | 48.3% | 29.0% | 19.3% | 51.7% | 100.0% | 80.0% | 75.0% | 12.2% |
| 2026-06-17 | 3707 | 100.0% | 100.0% | 45.6% | 25.8% | 19.7% | 54.4% | 100.0% | 93.1% | 89.6% | 3.4% |
| 2026-06-16 | 8262 | 100.0% | 100.0% | 46.4% | 29.2% | 17.2% | 53.6% | 100.0% | 92.4% | 85.9% | 1.7% |
| 2026-06-15 | 5873 | 100.0% | 100.0% | 47.7% | 29.9% | 17.8% | 52.3% | 100.0% | 93.5% | 88.9% | 2.6% |
| 2026-06-14 | 6056 | 100.0% | 100.0% | 49.8% | 31.0% | 18.8% | 50.2% | 100.0% | 85.5% | 82.6% | 3.2% |
| 2026-06-13 | 6581 | 100.0% | 100.0% | 49.1% | 30.4% | 18.7% | 50.9% | 100.0% | 91.3% | 85.9% | 1.0% |
| 2026-06-12 | 8682 | 100.0% | 100.0% | 48.7% | 30.6% | 18.1% | 51.3% | 100.0% | 92.4% | 88.8% | 2.5% |
| 2026-06-11 | 2202 | 100.0% | 100.0% | 49.4% | 31.9% | 17.5% | 50.6% | 100.0% | 75.8% | 85.3% | 6.4% |
| 2026-06-10 | 8414 | 100.0% | 100.0% | 44.2% | 27.3% | 16.9% | 55.8% | 100.0% | 90.6% | 85.5% | 4.4% |
| 2026-06-09 | 6566 | 100.0% | 100.0% | 41.6% | 25.9% | 15.7% | 58.4% | 100.0% | 93.1% | 86.8% | 3.2% |
| 2026-06-08 | 3589 | 100.0% | 100.0% | 41.3% | 26.0% | 15.3% | 58.7% | 100.0% | 92.6% | 86.2% | 2.8% |
| 2026-06-07 | 6521 | 100.0% | 100.0% | 40.1% | 23.9% | 16.1% | 59.9% | 100.0% | 89.4% | 86.5% | 0.9% |
| 2026-06-06 | 4562 | 100.0% | 100.0% | 41.6% | 25.6% | 16.0% | 58.4% | 100.0% | 81.9% | 88.1% | 2.0% |
| 2026-06-05 | 6699 | 100.0% | 100.0% | 42.3% | 26.3% | 16.0% | 57.7% | 100.0% | 92.1% | 86.8% | 3.6% |
| 2026-06-04 | 2348 | 100.0% | 100.0% | 43.0% | 26.2% | 16.7% | 57.0% | 100.0% | 95.1% | 69.8% | 13.3% |
| 2026-06-03 | 3331 | 100.0% | 100.0% | 36.2% | 25.3% | 10.9% | 63.8% | 100.0% | 92.5% | 0.0% | 100.0% |
| 2026-06-02 | 235 | 0.0% | 100.0% | 90.2% | 89.8% | 0.4% | 9.8% | 100.0% | 91.1% | 85.1% | 0.0% |
| 2026-06-01 | 208 | 0.0% | 100.0% | 96.2% | 93.3% | 2.9% | 3.8% | 100.0% | 93.3% | 88.0% | 9.1% |
| 2026-05-31 | 214 | 0.0% | 100.0% | 92.1% | 92.1% | 0.0% | 7.9% | 100.0% | 92.1% | 83.6% | 9.8% |

## Pairing By Date / Market / Book

| Date | Market | Book | Rows | Same-Book Pairs | Cross-Book Pairs | Synthetic Pairs | Missing Pairs | Same-Book Quality | Cross-Book Quality | Synthetic Quality | One-Sided Quality | Raw-Implied Prob | Synthetic Prob |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-19 | batter_total_bases | fanduel | 2604 | 0 | 339 | 2265 | 0 | 0 | 339 | 2265 | 0 | 0 | 2265 |
| 2026-06-19 | batter_hits | fanduel | 1649 | 0 | 725 | 924 | 0 | 0 | 725 | 924 | 0 | 0 | 924 |
| 2026-06-19 | batter_hits | draftkings | 1438 | 1438 | 0 | 0 | 0 | 1438 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-19 | batter_home_runs | fanduel | 1302 | 0 | 0 | 1302 | 0 | 0 | 0 | 1302 | 0 | 0 | 1302 |
| 2026-06-19 | batter_total_bases | draftkings | 626 | 626 | 0 | 0 | 0 | 626 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-19 | pitcher_strikeouts | draftkings | 166 | 166 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-19 | pitcher_strikeouts | fanduel | 144 | 144 | 0 | 0 | 0 | 144 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-18 | batter_total_bases | fanduel | 912 | 0 | 196 | 716 | 0 | 0 | 196 | 716 | 0 | 0 | 716 |
| 2026-06-18 | batter_hits | fanduel | 520 | 0 | 318 | 202 | 0 | 0 | 318 | 202 | 0 | 0 | 202 |
| 2026-06-18 | batter_home_runs | fanduel | 456 | 0 | 0 | 456 | 0 | 0 | 0 | 456 | 0 | 0 | 456 |
| 2026-06-18 | batter_hits | draftkings | 450 | 450 | 0 | 0 | 0 | 450 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-18 | batter_total_bases | draftkings | 220 | 220 | 0 | 0 | 0 | 220 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-18 | pitcher_strikeouts | draftkings | 52 | 52 | 0 | 0 | 0 | 52 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-18 | pitcher_strikeouts | fanduel | 50 | 50 | 0 | 0 | 0 | 50 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-17 | batter_total_bases | fanduel | 1328 | 0 | 271 | 1057 | 0 | 0 | 271 | 1057 | 0 | 0 | 1057 |
| 2026-06-17 | batter_hits | fanduel | 756 | 0 | 461 | 295 | 0 | 0 | 461 | 295 | 0 | 0 | 295 |
| 2026-06-17 | batter_home_runs | fanduel | 665 | 0 | 0 | 665 | 0 | 0 | 0 | 665 | 0 | 0 | 665 |
| 2026-06-17 | batter_hits | draftkings | 630 | 630 | 0 | 0 | 0 | 630 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-17 | batter_total_bases | draftkings | 328 | 328 | 0 | 0 | 0 | 328 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-16 | batter_total_bases | fanduel | 2840 | 0 | 496 | 2344 | 0 | 0 | 496 | 2344 | 0 | 0 | 2344 |
| 2026-06-16 | batter_hits | fanduel | 1593 | 0 | 923 | 670 | 0 | 0 | 923 | 670 | 0 | 0 | 670 |
| 2026-06-16 | batter_hits | draftkings | 1454 | 1454 | 0 | 0 | 0 | 1454 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-16 | batter_home_runs | fanduel | 1417 | 0 | 0 | 1417 | 0 | 0 | 0 | 1417 | 0 | 0 | 1417 |
| 2026-06-16 | batter_total_bases | draftkings | 624 | 624 | 0 | 0 | 0 | 624 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-16 | pitcher_strikeouts | draftkings | 172 | 172 | 0 | 0 | 0 | 172 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-16 | pitcher_strikeouts | fanduel | 162 | 162 | 0 | 0 | 0 | 162 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-15 | batter_total_bases | fanduel | 1984 | 0 | 370 | 1614 | 0 | 0 | 370 | 1614 | 0 | 0 | 1614 |
| 2026-06-15 | batter_hits | fanduel | 1141 | 0 | 678 | 463 | 0 | 0 | 678 | 463 | 0 | 0 | 463 |
| 2026-06-15 | batter_hits | draftkings | 1036 | 1036 | 0 | 0 | 0 | 1036 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-15 | batter_home_runs | fanduel | 992 | 0 | 0 | 992 | 0 | 0 | 0 | 992 | 0 | 0 | 992 |
| 2026-06-15 | batter_total_bases | draftkings | 508 | 508 | 0 | 0 | 0 | 508 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-15 | pitcher_strikeouts | draftkings | 114 | 114 | 0 | 0 | 0 | 114 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-15 | pitcher_strikeouts | fanduel | 98 | 98 | 0 | 0 | 0 | 98 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-14 | batter_total_bases | fanduel | 2056 | 0 | 432 | 1624 | 0 | 0 | 432 | 1624 | 0 | 0 | 1624 |
| 2026-06-14 | batter_hits | fanduel | 1092 | 0 | 705 | 387 | 0 | 0 | 705 | 387 | 0 | 0 | 387 |
| 2026-06-14 | batter_hits | draftkings | 1078 | 1078 | 0 | 0 | 0 | 1078 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-14 | batter_home_runs | fanduel | 1028 | 0 | 0 | 1028 | 0 | 0 | 0 | 1028 | 0 | 0 | 1028 |
| 2026-06-14 | batter_total_bases | draftkings | 574 | 574 | 0 | 0 | 0 | 574 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-14 | pitcher_strikeouts | draftkings | 126 | 126 | 0 | 0 | 0 | 126 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-14 | pitcher_strikeouts | fanduel | 102 | 102 | 0 | 0 | 0 | 102 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-13 | batter_total_bases | fanduel | 2260 | 0 | 446 | 1814 | 0 | 0 | 446 | 1814 | 0 | 0 | 1814 |
| 2026-06-13 | batter_hits | fanduel | 1193 | 0 | 783 | 410 | 0 | 0 | 783 | 410 | 0 | 0 | 410 |
| 2026-06-13 | batter_hits | draftkings | 1152 | 1152 | 0 | 0 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-13 | batter_home_runs | fanduel | 1128 | 0 | 0 | 1128 | 0 | 0 | 0 | 1128 | 0 | 0 | 1128 |
| 2026-06-13 | batter_total_bases | draftkings | 604 | 604 | 0 | 0 | 0 | 604 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-13 | pitcher_strikeouts | draftkings | 136 | 136 | 0 | 0 | 0 | 136 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-13 | pitcher_strikeouts | fanduel | 108 | 108 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | batter_total_bases | fanduel | 2900 | 0 | 586 | 2314 | 0 | 0 | 586 | 2314 | 0 | 0 | 2314 |
| 2026-06-12 | batter_hits | fanduel | 1674 | 0 | 988 | 686 | 0 | 0 | 988 | 686 | 0 | 0 | 686 |
| 2026-06-12 | batter_hits | draftkings | 1508 | 1508 | 0 | 0 | 0 | 1508 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | batter_home_runs | fanduel | 1450 | 0 | 0 | 1450 | 0 | 0 | 0 | 1450 | 0 | 0 | 1450 |
| 2026-06-12 | batter_total_bases | draftkings | 802 | 802 | 0 | 0 | 0 | 802 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | pitcher_strikeouts | draftkings | 180 | 180 | 0 | 0 | 0 | 180 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | pitcher_strikeouts | fanduel | 168 | 168 | 0 | 0 | 0 | 168 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-11 | batter_total_bases | fanduel | 720 | 0 | 142 | 578 | 0 | 0 | 142 | 578 | 0 | 0 | 578 |
| 2026-06-11 | batter_hits | fanduel | 420 | 0 | 244 | 176 | 0 | 0 | 244 | 176 | 0 | 0 | 176 |
| 2026-06-11 | batter_hits | draftkings | 394 | 394 | 0 | 0 | 0 | 394 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-11 | batter_home_runs | fanduel | 360 | 0 | 0 | 360 | 0 | 0 | 0 | 360 | 0 | 0 | 360 |
| 2026-06-11 | batter_total_bases | draftkings | 216 | 216 | 0 | 0 | 0 | 216 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-11 | pitcher_strikeouts | draftkings | 46 | 46 | 0 | 0 | 0 | 46 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-11 | pitcher_strikeouts | fanduel | 46 | 46 | 0 | 0 | 0 | 46 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-10 | batter_total_bases | fanduel | 2500 | 0 | 503 | 1997 | 0 | 0 | 503 | 1997 | 0 | 0 | 1997 |
| 2026-06-10 | batter_hits | fanduel | 2366 | 0 | 922 | 1444 | 0 | 0 | 922 | 1444 | 0 | 0 | 1444 |
| 2026-06-10 | batter_hits | draftkings | 1326 | 1326 | 0 | 0 | 0 | 1326 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-10 | batter_home_runs | fanduel | 1250 | 0 | 0 | 1250 | 0 | 0 | 0 | 1250 | 0 | 0 | 1250 |
| 2026-06-10 | batter_total_bases | draftkings | 716 | 716 | 0 | 0 | 0 | 716 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-10 | pitcher_strikeouts | draftkings | 132 | 132 | 0 | 0 | 0 | 132 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-10 | pitcher_strikeouts | fanduel | 124 | 124 | 0 | 0 | 0 | 124 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-09 | batter_total_bases | fanduel | 1960 | 0 | 364 | 1596 | 0 | 0 | 364 | 1596 | 0 | 0 | 1596 |
| 2026-06-09 | batter_hits | fanduel | 1924 | 0 | 668 | 1256 | 0 | 0 | 668 | 1256 | 0 | 0 | 1256 |
| 2026-06-09 | batter_hits | draftkings | 998 | 998 | 0 | 0 | 0 | 998 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-09 | batter_home_runs | fanduel | 980 | 0 | 0 | 980 | 0 | 0 | 0 | 980 | 0 | 0 | 980 |
| 2026-06-09 | batter_total_bases | draftkings | 478 | 478 | 0 | 0 | 0 | 478 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-09 | pitcher_strikeouts | draftkings | 118 | 118 | 0 | 0 | 0 | 118 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-09 | pitcher_strikeouts | fanduel | 108 | 108 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-08 | batter_total_bases | fanduel | 1068 | 0 | 191 | 877 | 0 | 0 | 191 | 877 | 0 | 0 | 877 |
| 2026-06-08 | batter_hits | fanduel | 1055 | 0 | 359 | 696 | 0 | 0 | 359 | 696 | 0 | 0 | 696 |
| 2026-06-08 | batter_hits | draftkings | 546 | 546 | 0 | 0 | 0 | 546 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-08 | batter_home_runs | fanduel | 534 | 0 | 0 | 534 | 0 | 0 | 0 | 534 | 0 | 0 | 534 |
| 2026-06-08 | batter_total_bases | draftkings | 262 | 262 | 0 | 0 | 0 | 262 | 0 | 0 | 0 | 0 | 0 |

## FanDuel Hitter Market Evidence

| Date | Market | Rows | True Pair | Synthetic | Clean Evidence | Same-Book | Cross-Book | Synthetic Rows | Action |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-19 | batter_total_bases | 2604 | 13.0% | 87.0% | 13.0% | 0 | 339 | 2265 | extract_true_opposite_side_or_demote |
| 2026-06-19 | batter_home_runs | 1302 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1302 | extract_true_opposite_side_or_demote |
| 2026-06-19 | batter_hits | 1649 | 44.0% | 56.0% | 44.0% | 0 | 725 | 924 | extract_true_opposite_side_or_demote |
| 2026-06-18 | batter_total_bases | 912 | 21.5% | 78.5% | 21.5% | 0 | 196 | 716 | extract_true_opposite_side_or_demote |
| 2026-06-18 | batter_home_runs | 456 | 0.0% | 100.0% | 0.0% | 0 | 0 | 456 | extract_true_opposite_side_or_demote |
| 2026-06-18 | batter_hits | 520 | 61.2% | 38.8% | 61.2% | 0 | 318 | 202 | usable |
| 2026-06-17 | batter_total_bases | 1328 | 20.4% | 79.6% | 20.4% | 0 | 271 | 1057 | extract_true_opposite_side_or_demote |
| 2026-06-17 | batter_home_runs | 665 | 0.0% | 100.0% | 0.0% | 0 | 0 | 665 | extract_true_opposite_side_or_demote |
| 2026-06-17 | batter_hits | 756 | 61.0% | 39.0% | 61.0% | 0 | 461 | 295 | usable |
| 2026-06-16 | batter_total_bases | 2840 | 17.5% | 82.5% | 17.5% | 0 | 496 | 2344 | extract_true_opposite_side_or_demote |
| 2026-06-16 | batter_home_runs | 1417 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1417 | extract_true_opposite_side_or_demote |
| 2026-06-16 | batter_hits | 1593 | 57.9% | 42.1% | 57.9% | 0 | 923 | 670 | usable |
| 2026-06-15 | batter_total_bases | 1984 | 18.6% | 81.4% | 18.6% | 0 | 370 | 1614 | extract_true_opposite_side_or_demote |
| 2026-06-15 | batter_home_runs | 992 | 0.0% | 100.0% | 0.0% | 0 | 0 | 992 | extract_true_opposite_side_or_demote |
| 2026-06-15 | batter_hits | 1141 | 59.4% | 40.6% | 59.4% | 0 | 678 | 463 | usable |
| 2026-06-14 | batter_total_bases | 2056 | 21.0% | 79.0% | 21.0% | 0 | 432 | 1624 | extract_true_opposite_side_or_demote |
| 2026-06-14 | batter_home_runs | 1028 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1028 | extract_true_opposite_side_or_demote |
| 2026-06-14 | batter_hits | 1092 | 64.6% | 35.4% | 64.6% | 0 | 705 | 387 | usable |
| 2026-06-13 | batter_total_bases | 2260 | 19.7% | 80.3% | 19.7% | 0 | 446 | 1814 | extract_true_opposite_side_or_demote |
| 2026-06-13 | batter_home_runs | 1128 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1128 | extract_true_opposite_side_or_demote |
| 2026-06-13 | batter_hits | 1193 | 65.6% | 34.4% | 65.6% | 0 | 783 | 410 | usable |
| 2026-06-12 | batter_total_bases | 2900 | 20.2% | 79.8% | 20.2% | 0 | 586 | 2314 | extract_true_opposite_side_or_demote |
| 2026-06-12 | batter_home_runs | 1450 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1450 | extract_true_opposite_side_or_demote |
| 2026-06-12 | batter_hits | 1674 | 59.0% | 41.0% | 59.0% | 0 | 988 | 686 | usable |
| 2026-06-11 | batter_total_bases | 720 | 19.7% | 80.3% | 19.7% | 0 | 142 | 578 | extract_true_opposite_side_or_demote |
| 2026-06-11 | batter_home_runs | 360 | 0.0% | 100.0% | 0.0% | 0 | 0 | 360 | extract_true_opposite_side_or_demote |
| 2026-06-11 | batter_hits | 420 | 58.1% | 41.9% | 58.1% | 0 | 244 | 176 | usable |
| 2026-06-10 | batter_total_bases | 2500 | 20.1% | 79.9% | 20.1% | 0 | 503 | 1997 | extract_true_opposite_side_or_demote |
| 2026-06-10 | batter_home_runs | 1250 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1250 | extract_true_opposite_side_or_demote |
| 2026-06-10 | batter_hits | 2366 | 39.0% | 61.0% | 39.0% | 0 | 922 | 1444 | extract_true_opposite_side_or_demote |
| 2026-06-09 | batter_total_bases | 1960 | 18.6% | 81.4% | 18.6% | 0 | 364 | 1596 | extract_true_opposite_side_or_demote |
| 2026-06-09 | batter_home_runs | 980 | 0.0% | 100.0% | 0.0% | 0 | 0 | 980 | extract_true_opposite_side_or_demote |
| 2026-06-09 | batter_hits | 1924 | 34.7% | 65.3% | 34.7% | 0 | 668 | 1256 | extract_true_opposite_side_or_demote |
| 2026-06-08 | batter_total_bases | 1068 | 17.9% | 82.1% | 17.9% | 0 | 191 | 877 | extract_true_opposite_side_or_demote |
| 2026-06-08 | batter_home_runs | 534 | 0.0% | 100.0% | 0.0% | 0 | 0 | 534 | extract_true_opposite_side_or_demote |
| 2026-06-08 | batter_hits | 1055 | 34.0% | 66.0% | 34.0% | 0 | 359 | 696 | extract_true_opposite_side_or_demote |
| 2026-06-07 | batter_total_bases | 1996 | 19.5% | 80.5% | 19.5% | 0 | 390 | 1606 | extract_true_opposite_side_or_demote |
| 2026-06-07 | batter_home_runs | 998 | 0.0% | 100.0% | 0.0% | 0 | 0 | 998 | extract_true_opposite_side_or_demote |
| 2026-06-07 | batter_hits | 1967 | 33.7% | 66.3% | 33.7% | 0 | 663 | 1304 | extract_true_opposite_side_or_demote |
| 2026-06-06 | batter_total_bases | 1364 | 18.8% | 81.2% | 18.8% | 0 | 257 | 1107 | extract_true_opposite_side_or_demote |
| 2026-06-06 | batter_home_runs | 682 | 0.0% | 100.0% | 0.0% | 0 | 0 | 682 | extract_true_opposite_side_or_demote |
| 2026-06-06 | batter_hits | 1348 | 34.9% | 65.1% | 34.9% | 0 | 471 | 877 | extract_true_opposite_side_or_demote |
| 2026-06-05 | batter_total_bases | 1980 | 18.8% | 81.2% | 18.8% | 0 | 372 | 1608 | extract_true_opposite_side_or_demote |
| 2026-06-05 | batter_home_runs | 1004 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1004 | extract_true_opposite_side_or_demote |
| 2026-06-05 | batter_hits | 1953 | 35.7% | 64.3% | 35.7% | 0 | 698 | 1255 | extract_true_opposite_side_or_demote |
| 2026-06-04 | batter_total_bases | 696 | 20.7% | 79.3% | 20.7% | 0 | 144 | 552 | extract_true_opposite_side_or_demote |
| 2026-06-04 | batter_home_runs | 348 | 0.0% | 100.0% | 0.0% | 0 | 0 | 348 | extract_true_opposite_side_or_demote |
| 2026-06-04 | batter_hits | 688 | 36.2% | 63.8% | 36.2% | 0 | 249 | 439 | extract_true_opposite_side_or_demote |
| 2026-06-03 | batter_total_bases | 1008 | 11.3% | 88.7% | 11.3% | 0 | 114 | 894 | extract_true_opposite_side_or_demote |
| 2026-06-03 | batter_home_runs | 503 | 0.0% | 100.0% | 0.0% | 0 | 0 | 503 | extract_true_opposite_side_or_demote |
| 2026-06-03 | batter_hits | 978 | 25.6% | 74.4% | 25.6% | 0 | 250 | 728 | extract_true_opposite_side_or_demote |
| 2026-06-02 | batter_total_bases | 4 | 25.0% | 75.0% | 25.0% | 0 | 1 | 3 | extract_true_opposite_side_or_demote |
| 2026-06-02 | batter_home_runs | 20 | 0.0% | 100.0% | 0.0% | 0 | 0 | 20 | extract_true_opposite_side_or_demote |
| 2026-06-01 | batter_total_bases | 10 | 40.0% | 60.0% | 40.0% | 0 | 4 | 6 | extract_true_opposite_side_or_demote |
| 2026-06-01 | batter_hits | 4 | 50.0% | 50.0% | 50.0% | 0 | 2 | 2 | usable |
| 2026-05-31 | batter_total_bases | 2 | 0.0% | 100.0% | 0.0% | 0 | 0 | 2 | extract_true_opposite_side_or_demote |
| 2026-05-31 | batter_home_runs | 15 | 0.0% | 100.0% | 0.0% | 0 | 0 | 15 | extract_true_opposite_side_or_demote |

## Problem Examples

| Date | Player | Bet | Missing Fields | Pairing Note | CLV Status | CLV Reason |
|---|---|---|---|---|---|---|
| 2026-05-31 | Ildemaro Vargas | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Nolan Arenado | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Ketel Marte | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | J.P. Crawford | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Josh Naylor | batter_hits under 0.5 draftkings | prop_offer_id |  | true_no_movement |  |
| 2026-05-31 | Randy Arozarena | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Geraldo Perdomo | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Julio Rodríguez | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Ketel Marte | batter_total_bases under 1.5 draftkings | prop_offer_id |  | true_no_movement |  |
| 2026-05-31 | Corbin Carroll | batter_total_bases under 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Merrill Kelly | pitcher_strikeouts under 4.5 draftkings | prop_offer_id |  | true_no_movement |  |
| 2026-05-31 | Matt Olson | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Jorge Mateo | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Eli White | batter_hits under 0.5 draftkings | prop_offer_id |  | unknown | stale_close_before_lock |
| 2026-05-31 | Mauricio Dubón | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Ozzie Albies | batter_hits under 0.5 draftkings | prop_offer_id |  | unknown | line_disappeared_at_close |
| 2026-05-31 | Austin Riley | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Michael Harris II | batter_hits under 0.5 draftkings | prop_offer_id |  | unknown | line_disappeared_at_close |
| 2026-05-31 | Matt McLain | batter_hits over 0.5 draftkings | prop_offer_id |  | unknown | stale_close_before_lock |
| 2026-05-31 | Matt Olson | batter_total_bases under 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Eli White | batter_total_bases under 1.5 draftkings | prop_offer_id |  | unknown | stale_close_before_lock |
| 2026-05-31 | Mauricio Dubón | batter_total_bases under 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Ozzie Albies | batter_total_bases under 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Michael Harris II | batter_total_bases under 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Nick Lodolo | pitcher_strikeouts under 5.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Spencer Strider | pitcher_strikeouts under 6.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | José Ramírez | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Ceddanne Rafaela | batter_hits under 0.5 draftkings | prop_offer_id, actual_value, won |  | unknown | stale_close_before_lock |
| 2026-05-31 | Daniel Schneemann | batter_hits over 0.5 draftkings | prop_offer_id, actual_value, won |  | unknown | stale_close_before_lock |
| 2026-05-31 | Masataka Yoshida | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
