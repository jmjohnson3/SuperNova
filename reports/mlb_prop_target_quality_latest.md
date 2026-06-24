# MLB Prop Target Quality

Generated UTC: 2026-06-24T15:11:28Z
Rows: 120581
Date range: 2026-05-31 to 2026-06-24

## Required Field Coverage

| Field | Present | Missing | Coverage |
|---|---:|---:|---:|
| bookmaker_key | 120581 | 0 | 100.0% |
| market_line | 120581 | 0 | 100.0% |
| market_price | 120581 | 0 | 100.0% |
| prop_offer_id | 119924 | 657 | 99.5% |
| lock_snapshot_id | 120581 | 0 | 100.0% |
| source_created_at | 120581 | 0 | 100.0% |
| actual_value | 106333 | 14248 | 88.2% |
| won | 106333 | 14248 | 88.2% |
| paired_price | 120581 | 0 | 100.0% |
| paired_bookmaker_key | 120581 | 0 | 100.0% |
| paired_price_source | 120581 | 0 | 100.0% |
| pair_quality | 120581 | 0 | 100.0% |
| no_vig_market_prob | 120581 | 0 | 100.0% |
| market_prob_source | 120581 | 0 | 100.0% |
| closing_line | 104823 | 15758 | 86.9% |
| closing_price | 104823 | 15758 | 86.9% |
| closing_snapshot_id | 104823 | 15758 | 86.9% |
| closing_fetched_at_utc | 104823 | 15758 | 86.9% |
| clv_valid | 120581 | 0 | 100.0% |

## CLV / Close Status

| Status | Rows |
|---|---:|
| valid_movement | 72984 |
| true_no_movement | 31839 |
| unknown | 15758 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| none | 104823 |
| stale_close_before_lock | 7233 |
| close_outside_two_hour_window | 6581 |
| line_disappeared_at_close | 1408 |
| fallback_other_book_only | 294 |
| no_valid_close_snapshot | 242 |

## Quality By Date

| Date | Rows | Offer ID | Price+Lock | True Pair | Same-Book Pair | Cross-Book Pair | Synthetic Pair | Any Pair | Graded | Valid Close | Stale Close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-24 | 2446 | 100.0% | 100.0% | 42.2% | 29.7% | 12.6% | 57.8% | 100.0% | 0.0% | 12.0% | 3.5% |
| 2026-06-23 | 4119 | 100.0% | 100.0% | 45.0% | 28.5% | 16.5% | 55.0% | 100.0% | 87.4% | 78.2% | 5.4% |
| 2026-06-22 | 7540 | 100.0% | 100.0% | 46.8% | 29.0% | 17.7% | 53.2% | 100.0% | 87.9% | 90.4% | 3.5% |
| 2026-06-21 | 5500 | 100.0% | 100.0% | 48.5% | 29.3% | 19.2% | 51.5% | 100.0% | 87.1% | 93.7% | 3.9% |
| 2026-06-20 | 6337 | 100.0% | 100.0% | 48.2% | 30.7% | 17.6% | 51.8% | 100.0% | 89.8% | 88.6% | 2.8% |
| 2026-06-19 | 7929 | 100.0% | 100.0% | 48.8% | 29.9% | 18.8% | 51.2% | 100.0% | 92.6% | 91.7% | 2.3% |
| 2026-06-18 | 2660 | 100.0% | 100.0% | 48.3% | 29.0% | 19.3% | 51.7% | 100.0% | 80.0% | 77.4% | 12.2% |
| 2026-06-17 | 3707 | 100.0% | 100.0% | 45.6% | 25.8% | 19.7% | 54.4% | 100.0% | 93.1% | 93.9% | 3.4% |
| 2026-06-16 | 8262 | 100.0% | 100.0% | 46.4% | 29.2% | 17.2% | 53.6% | 100.0% | 92.4% | 94.0% | 1.7% |
| 2026-06-15 | 5873 | 100.0% | 100.0% | 47.7% | 29.9% | 17.8% | 52.3% | 100.0% | 93.5% | 92.3% | 2.6% |
| 2026-06-14 | 6056 | 100.0% | 100.0% | 49.8% | 31.0% | 18.8% | 50.2% | 100.0% | 85.5% | 94.0% | 3.1% |
| 2026-06-13 | 6581 | 100.0% | 100.0% | 49.1% | 30.4% | 18.7% | 50.9% | 100.0% | 91.3% | 92.4% | 1.0% |
| 2026-06-12 | 8682 | 100.0% | 100.0% | 48.7% | 30.6% | 18.1% | 51.3% | 100.0% | 92.4% | 92.0% | 2.5% |
| 2026-06-11 | 2202 | 100.0% | 100.0% | 49.4% | 31.9% | 17.5% | 50.6% | 100.0% | 75.8% | 88.6% | 6.1% |
| 2026-06-10 | 8414 | 100.0% | 100.0% | 44.2% | 27.3% | 16.9% | 55.8% | 100.0% | 90.6% | 93.0% | 4.2% |
| 2026-06-09 | 6566 | 100.0% | 100.0% | 41.6% | 25.9% | 15.7% | 58.4% | 100.0% | 93.1% | 90.6% | 3.2% |
| 2026-06-08 | 3589 | 100.0% | 100.0% | 41.3% | 26.0% | 15.3% | 58.7% | 100.0% | 92.6% | 91.3% | 2.8% |
| 2026-06-07 | 6521 | 100.0% | 100.0% | 40.1% | 23.9% | 16.1% | 59.9% | 100.0% | 89.4% | 97.9% | 0.9% |
| 2026-06-06 | 4562 | 100.0% | 100.0% | 41.6% | 25.6% | 16.0% | 58.4% | 100.0% | 81.9% | 93.9% | 1.9% |
| 2026-06-05 | 6699 | 100.0% | 100.0% | 42.3% | 26.3% | 16.0% | 57.7% | 100.0% | 92.1% | 91.0% | 3.6% |
| 2026-06-04 | 2348 | 100.0% | 100.0% | 43.0% | 26.2% | 16.7% | 57.0% | 100.0% | 95.1% | 69.8% | 13.3% |
| 2026-06-03 | 3331 | 100.0% | 100.0% | 36.2% | 25.3% | 10.9% | 63.8% | 100.0% | 92.5% | 0.0% | 100.0% |
| 2026-06-02 | 235 | 0.0% | 100.0% | 90.2% | 89.8% | 0.4% | 9.8% | 100.0% | 91.1% | 85.1% | 0.0% |
| 2026-06-01 | 208 | 0.0% | 100.0% | 96.2% | 93.3% | 2.9% | 3.8% | 100.0% | 93.3% | 88.0% | 9.1% |
| 2026-05-31 | 214 | 0.0% | 100.0% | 92.1% | 92.1% | 0.0% | 7.9% | 100.0% | 92.1% | 83.6% | 9.8% |

## Pairing By Date / Market / Book

| Date | Market | Book | Rows | Same-Book Pairs | Cross-Book Pairs | Synthetic Pairs | Missing Pairs | Same-Book Quality | Cross-Book Quality | Synthetic Quality | One-Sided Quality | Raw-Implied Prob | Synthetic Prob |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-24 | batter_total_bases | fanduel | 860 | 0 | 96 | 764 | 0 | 0 | 96 | 764 | 0 | 0 | 764 |
| 2026-06-24 | batter_hits | draftkings | 432 | 432 | 0 | 0 | 0 | 432 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-24 | batter_hits | fanduel | 430 | 0 | 211 | 219 | 0 | 0 | 211 | 219 | 0 | 0 | 219 |
| 2026-06-24 | batter_home_runs | fanduel | 430 | 0 | 0 | 430 | 0 | 0 | 0 | 430 | 0 | 0 | 430 |
| 2026-06-24 | batter_total_bases | draftkings | 194 | 194 | 0 | 0 | 0 | 194 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-24 | pitcher_strikeouts | draftkings | 54 | 54 | 0 | 0 | 0 | 54 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-24 | pitcher_strikeouts | fanduel | 46 | 46 | 0 | 0 | 0 | 46 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-23 | batter_total_bases | fanduel | 1380 | 0 | 226 | 1154 | 0 | 0 | 226 | 1154 | 0 | 0 | 1154 |
| 2026-06-23 | batter_hits | fanduel | 876 | 0 | 452 | 424 | 0 | 0 | 452 | 424 | 0 | 0 | 424 |
| 2026-06-23 | batter_hits | draftkings | 710 | 710 | 0 | 0 | 0 | 710 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-23 | batter_home_runs | fanduel | 689 | 0 | 0 | 689 | 0 | 0 | 0 | 689 | 0 | 0 | 689 |
| 2026-06-23 | batter_total_bases | draftkings | 312 | 312 | 0 | 0 | 0 | 312 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-23 | pitcher_strikeouts | draftkings | 80 | 80 | 0 | 0 | 0 | 80 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-23 | pitcher_strikeouts | fanduel | 72 | 72 | 0 | 0 | 0 | 72 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-22 | batter_total_bases | fanduel | 2508 | 0 | 464 | 2044 | 0 | 0 | 464 | 2044 | 0 | 0 | 2044 |
| 2026-06-22 | batter_hits | fanduel | 1591 | 0 | 872 | 719 | 0 | 0 | 872 | 719 | 0 | 0 | 719 |
| 2026-06-22 | batter_hits | draftkings | 1288 | 1288 | 0 | 0 | 0 | 1288 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-22 | batter_home_runs | fanduel | 1251 | 0 | 0 | 1251 | 0 | 0 | 0 | 1251 | 0 | 0 | 1251 |
| 2026-06-22 | batter_total_bases | draftkings | 616 | 616 | 0 | 0 | 0 | 616 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-22 | pitcher_strikeouts | draftkings | 152 | 152 | 0 | 0 | 0 | 152 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-22 | pitcher_strikeouts | fanduel | 134 | 134 | 0 | 0 | 0 | 134 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-21 | batter_total_bases | fanduel | 1916 | 0 | 379 | 1537 | 0 | 0 | 379 | 1537 | 0 | 0 | 1537 |
| 2026-06-21 | batter_hits | fanduel | 1015 | 0 | 678 | 337 | 0 | 0 | 678 | 337 | 0 | 0 | 337 |
| 2026-06-21 | batter_hits | draftkings | 960 | 960 | 0 | 0 | 0 | 960 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-21 | batter_home_runs | fanduel | 957 | 0 | 0 | 957 | 0 | 0 | 0 | 957 | 0 | 0 | 957 |
| 2026-06-21 | batter_total_bases | draftkings | 424 | 424 | 0 | 0 | 0 | 424 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-21 | pitcher_strikeouts | draftkings | 118 | 118 | 0 | 0 | 0 | 118 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-21 | pitcher_strikeouts | fanduel | 110 | 110 | 0 | 0 | 0 | 110 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-20 | batter_total_bases | fanduel | 2156 | 0 | 386 | 1770 | 0 | 0 | 386 | 1770 | 0 | 0 | 1770 |
| 2026-06-20 | batter_hits | fanduel | 1159 | 0 | 727 | 432 | 0 | 0 | 727 | 432 | 0 | 0 | 432 |
| 2026-06-20 | batter_hits | draftkings | 1138 | 1138 | 0 | 0 | 0 | 1138 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-20 | batter_home_runs | fanduel | 1078 | 0 | 0 | 1078 | 0 | 0 | 0 | 1078 | 0 | 0 | 1078 |
| 2026-06-20 | batter_total_bases | draftkings | 512 | 512 | 0 | 0 | 0 | 512 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-20 | pitcher_strikeouts | draftkings | 160 | 160 | 0 | 0 | 0 | 160 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-20 | pitcher_strikeouts | fanduel | 134 | 134 | 0 | 0 | 0 | 134 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-19 | batter_total_bases | fanduel | 2604 | 0 | 536 | 2068 | 0 | 0 | 536 | 2068 | 0 | 0 | 2068 |
| 2026-06-19 | batter_hits | fanduel | 1649 | 0 | 957 | 692 | 0 | 0 | 957 | 692 | 0 | 0 | 692 |
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

## FanDuel Hitter Market Evidence

| Date | Market | Rows | True Pair | Synthetic | Clean Evidence | Same-Book | Cross-Book | Synthetic Rows | Action |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-24 | batter_total_bases | 860 | 11.2% | 88.8% | 11.2% | 0 | 96 | 764 | extract_true_opposite_side_or_demote |
| 2026-06-24 | batter_home_runs | 430 | 0.0% | 100.0% | 0.0% | 0 | 0 | 430 | extract_true_opposite_side_or_demote |
| 2026-06-24 | batter_hits | 430 | 49.1% | 50.9% | 49.1% | 0 | 211 | 219 | extract_true_opposite_side_or_demote |
| 2026-06-23 | batter_total_bases | 1380 | 16.4% | 83.6% | 16.4% | 0 | 226 | 1154 | extract_true_opposite_side_or_demote |
| 2026-06-23 | batter_home_runs | 689 | 0.0% | 100.0% | 0.0% | 0 | 0 | 689 | extract_true_opposite_side_or_demote |
| 2026-06-23 | batter_hits | 876 | 51.6% | 48.4% | 51.6% | 0 | 452 | 424 | usable |
| 2026-06-22 | batter_total_bases | 2508 | 18.5% | 81.5% | 18.5% | 0 | 464 | 2044 | extract_true_opposite_side_or_demote |
| 2026-06-22 | batter_home_runs | 1251 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1251 | extract_true_opposite_side_or_demote |
| 2026-06-22 | batter_hits | 1591 | 54.8% | 45.2% | 54.8% | 0 | 872 | 719 | usable |
| 2026-06-21 | batter_total_bases | 1916 | 19.8% | 80.2% | 19.8% | 0 | 379 | 1537 | extract_true_opposite_side_or_demote |
| 2026-06-21 | batter_home_runs | 957 | 0.0% | 100.0% | 0.0% | 0 | 0 | 957 | extract_true_opposite_side_or_demote |
| 2026-06-21 | batter_hits | 1015 | 66.8% | 33.2% | 66.8% | 0 | 678 | 337 | usable |
| 2026-06-20 | batter_total_bases | 2156 | 17.9% | 82.1% | 17.9% | 0 | 386 | 1770 | extract_true_opposite_side_or_demote |
| 2026-06-20 | batter_home_runs | 1078 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1078 | extract_true_opposite_side_or_demote |
| 2026-06-20 | batter_hits | 1159 | 62.7% | 37.3% | 62.7% | 0 | 727 | 432 | usable |
| 2026-06-19 | batter_total_bases | 2604 | 20.6% | 79.4% | 20.6% | 0 | 536 | 2068 | extract_true_opposite_side_or_demote |
| 2026-06-19 | batter_home_runs | 1302 | 0.0% | 100.0% | 0.0% | 0 | 0 | 1302 | extract_true_opposite_side_or_demote |
| 2026-06-19 | batter_hits | 1649 | 58.0% | 42.0% | 58.0% | 0 | 957 | 692 | usable |
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
