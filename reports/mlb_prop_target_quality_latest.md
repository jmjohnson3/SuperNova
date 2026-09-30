# MLB Prop Target Quality

Generated UTC: 2026-08-29T12:55:04Z
Rows: 327259
Date range: 2026-06-01 to 2026-07-30

## Required Field Coverage

| Field | Present | Missing | Coverage |
|---|---:|---:|---:|
| bookmaker_key | 327259 | 0 | 100.0% |
| market_line | 327259 | 0 | 100.0% |
| market_price | 327259 | 0 | 100.0% |
| prop_offer_id | 326816 | 443 | 99.9% |
| lock_snapshot_id | 327259 | 0 | 100.0% |
| source_created_at | 327259 | 0 | 100.0% |
| actual_value | 295467 | 31792 | 90.3% |
| won | 295467 | 31792 | 90.3% |
| paired_price | 155268 | 171991 | 47.4% |
| paired_bookmaker_key | 155268 | 171991 | 47.4% |
| paired_price_source | 155268 | 171991 | 47.4% |
| pair_quality | 327259 | 0 | 100.0% |
| no_vig_market_prob | 155268 | 171991 | 47.4% |
| market_prob_source | 327259 | 0 | 100.0% |
| closing_line | 272109 | 55150 | 83.1% |
| closing_price | 272109 | 55150 | 83.1% |
| closing_snapshot_id | 272109 | 55150 | 83.1% |
| closing_fetched_at_utc | 272109 | 55150 | 83.1% |
| clv_valid | 327259 | 0 | 100.0% |

## Promotion Evidence Coverage

Feed-wide coverage includes FanDuel lottery ladders. Exact-line training uses only clean common-line true pairs.

| Market | All Rows | Common Rows | Clean True Pairs | Common Pair Rate | Training Pair Rate | Unavoidable One-Sided | Action |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 123732 | 110851 | 93564 | 84.4% | 100.0% | 17287 | train_and_promote_on_clean_true_pairs_only |
| batter_home_runs | 54673 | 27357 | 0 | 0.0% | 0.0% | 27357 | projection_only_no_true_opposite_side |
| batter_total_bases | 136735 | 82134 | 46991 | 57.2% | 100.0% | 35143 | train_and_promote_on_clean_true_pairs_only |
| pitcher_strikeouts | 12119 | 12119 | 12119 | 100.0% | 100.0% | 0 | train_and_promote_on_clean_true_pairs_only |

## CLV / Close Status

| Status | Rows |
|---|---:|
| valid_movement | 197349 |
| true_no_movement | 74760 |
| unknown | 55150 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| none | 272109 |
| player_prop_unavailable_at_close | 34455 |
| close_outside_two_hour_window | 6850 |
| stale_close_before_lock | 6416 |
| line_disappeared_at_close | 5126 |
| player_market_unavailable_at_close | 1199 |
| fallback_other_book_only | 1058 |
| no_valid_close_snapshot | 46 |

## Quality By Date

| Date | Rows | Offer ID | Price+Lock | True Pair | Same-Book Pair | Cross-Book Pair | Synthetic Pair | Any Pair | Graded | Valid Close | Stale Close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-30 | 3232 | 100.0% | 100.0% | 45.5% | 30.2% | 15.3% | 0.0% | 45.5% | 92.9% | 38.4% | 0.2% |
| 2026-07-29 | 5756 | 100.0% | 100.0% | 40.4% | 19.9% | 20.5% | 0.0% | 40.4% | 94.2% | 90.3% | 0.5% |
| 2026-07-28 | 8043 | 100.0% | 100.0% | 49.4% | 29.6% | 19.7% | 0.0% | 49.4% | 86.5% | 81.9% | 2.3% |
| 2026-07-27 | 5850 | 100.0% | 100.0% | 50.2% | 31.2% | 18.9% | 0.0% | 50.2% | 85.0% | 88.1% | 0.0% |
| 2026-07-26 | 5175 | 100.0% | 100.0% | 49.4% | 30.4% | 19.0% | 0.0% | 49.4% | 92.7% | 85.5% | 0.0% |
| 2026-07-25 | 6863 | 100.0% | 100.0% | 46.9% | 29.2% | 17.7% | 0.0% | 46.9% | 90.8% | 86.3% | 0.0% |
| 2026-07-24 | 7560 | 100.0% | 100.0% | 47.6% | 28.9% | 18.6% | 0.0% | 47.6% | 92.6% | 86.3% | 0.0% |
| 2026-07-23 | 1569 | 100.0% | 100.0% | 47.6% | 29.8% | 17.8% | 0.0% | 47.6% | 91.1% | 85.1% | 0.0% |
| 2026-07-22 | 7248 | 100.0% | 100.0% | 48.2% | 28.2% | 20.0% | 0.0% | 48.2% | 89.6% | 84.5% | 0.5% |
| 2026-07-21 | 7893 | 100.0% | 100.0% | 49.1% | 30.9% | 18.1% | 0.0% | 49.1% | 78.1% | 78.9% | 1.9% |
| 2026-07-20 | 7638 | 100.0% | 100.0% | 50.6% | 30.8% | 19.8% | 0.0% | 50.6% | 94.6% | 87.9% | 0.0% |
| 2026-07-19 | 5423 | 100.0% | 100.0% | 49.8% | 29.9% | 19.9% | 0.0% | 49.8% | 91.2% | 85.7% | 0.4% |
| 2026-07-18 | 6033 | 100.0% | 100.0% | 51.5% | 30.6% | 20.9% | 0.0% | 51.5% | 84.3% | 81.2% | 0.9% |
| 2026-07-17 | 7705 | 100.0% | 100.0% | 50.1% | 30.2% | 19.9% | 0.0% | 50.1% | 88.0% | 84.0% | 1.0% |
| 2026-07-16 | 563 | 100.0% | 100.0% | 49.0% | 30.9% | 18.1% | 0.0% | 49.0% | 100.0% | 97.9% | 0.0% |
| 2026-07-12 | 7667 | 100.0% | 100.0% | 49.4% | 30.0% | 19.5% | 0.0% | 49.4% | 95.1% | 60.0% | 27.4% |
| 2026-07-11 | 6643 | 100.0% | 100.0% | 48.4% | 28.8% | 19.6% | 0.0% | 48.4% | 94.4% | 89.2% | 0.0% |
| 2026-07-10 | 8100 | 100.0% | 100.0% | 48.6% | 29.5% | 19.0% | 0.0% | 48.6% | 84.2% | 86.7% | 0.0% |
| 2026-07-09 | 5964 | 100.0% | 100.0% | 48.2% | 29.9% | 18.3% | 0.0% | 48.2% | 93.5% | 88.7% | 0.0% |
| 2026-07-08 | 7951 | 100.0% | 100.0% | 49.2% | 29.6% | 19.6% | 0.0% | 49.2% | 90.1% | 85.9% | 0.0% |
| 2026-07-07 | 8289 | 100.0% | 100.0% | 46.6% | 26.9% | 19.6% | 0.0% | 46.6% | 91.4% | 86.3% | 0.6% |
| 2026-07-06 | 4100 | 100.0% | 100.0% | 51.0% | 31.2% | 19.7% | 0.0% | 51.0% | 93.4% | 87.3% | 0.0% |
| 2026-07-05 | 5782 | 100.0% | 100.0% | 50.7% | 31.0% | 19.7% | 0.0% | 50.7% | 92.0% | 85.7% | 0.0% |
| 2026-07-04 | 7530 | 100.0% | 100.0% | 49.2% | 30.1% | 19.1% | 0.0% | 49.2% | 89.2% | 84.1% | 0.0% |
| 2026-07-03 | 6784 | 100.0% | 100.0% | 49.7% | 32.1% | 17.6% | 0.0% | 49.7% | 92.3% | 87.3% | 0.0% |
| 2026-07-02 | 4505 | 100.0% | 100.0% | 51.3% | 30.9% | 20.4% | 0.0% | 51.3% | 94.6% | 90.2% | 0.0% |
| 2026-07-01 | 6538 | 100.0% | 100.0% | 49.5% | 31.9% | 17.6% | 0.0% | 49.5% | 90.2% | 80.4% | 0.0% |
| 2026-06-30 | 8060 | 100.0% | 100.0% | 49.0% | 30.5% | 18.6% | 0.0% | 49.0% | 89.7% | 85.7% | 0.0% |
| 2026-06-29 | 6912 | 100.0% | 100.0% | 51.8% | 32.2% | 19.6% | 0.0% | 51.8% | 92.7% | 86.1% | 0.0% |
| 2026-06-28 | 5935 | 100.0% | 100.0% | 44.9% | 28.0% | 16.8% | 0.0% | 44.9% | 92.0% | 82.3% | 0.0% |
| 2026-06-27 | 7032 | 100.0% | 100.0% | 44.3% | 28.7% | 15.6% | 0.0% | 44.3% | 93.2% | 86.1% | 0.0% |
| 2026-06-26 | 8013 | 100.0% | 100.0% | 44.3% | 29.2% | 15.1% | 0.0% | 44.3% | 94.4% | 85.7% | 0.0% |
| 2026-06-25 | 2285 | 100.0% | 100.0% | 43.2% | 27.7% | 15.4% | 0.0% | 43.2% | 77.2% | 78.7% | 1.2% |
| 2026-06-24 | 4697 | 100.0% | 100.0% | 45.4% | 28.6% | 16.8% | 0.0% | 45.4% | 91.3% | 77.1% | 0.7% |
| 2026-06-23 | 4119 | 100.0% | 100.0% | 45.0% | 28.5% | 16.5% | 0.0% | 45.0% | 87.4% | 57.1% | 0.2% |
| 2026-06-22 | 7540 | 100.0% | 100.0% | 46.8% | 29.0% | 17.7% | 0.0% | 46.8% | 87.9% | 87.5% | 0.0% |
| 2026-06-21 | 5500 | 100.0% | 100.0% | 48.5% | 29.3% | 19.2% | 0.0% | 48.5% | 87.1% | 81.5% | 0.0% |
| 2026-06-20 | 6337 | 100.0% | 100.0% | 48.2% | 30.7% | 17.6% | 0.0% | 48.2% | 89.8% | 83.7% | 0.0% |
| 2026-06-19 | 7929 | 100.0% | 100.0% | 48.8% | 29.9% | 18.8% | 0.0% | 48.8% | 92.6% | 88.0% | 0.0% |
| 2026-06-18 | 2660 | 100.0% | 100.0% | 48.3% | 29.0% | 19.3% | 0.0% | 48.3% | 80.0% | 75.0% | 7.2% |
| 2026-06-17 | 3707 | 100.0% | 100.0% | 45.6% | 25.8% | 19.7% | 0.0% | 45.6% | 93.1% | 89.6% | 0.5% |
| 2026-06-16 | 8262 | 100.0% | 100.0% | 46.4% | 29.2% | 17.2% | 0.0% | 46.4% | 92.4% | 85.9% | 0.0% |
| 2026-06-15 | 5873 | 100.0% | 100.0% | 47.7% | 29.9% | 17.8% | 0.0% | 47.7% | 93.5% | 88.9% | 0.0% |
| 2026-06-14 | 6056 | 100.0% | 100.0% | 49.8% | 31.0% | 18.8% | 0.0% | 49.8% | 85.5% | 82.6% | 0.0% |
| 2026-06-13 | 6581 | 100.0% | 100.0% | 49.1% | 30.4% | 18.7% | 0.0% | 49.1% | 91.3% | 85.9% | 0.0% |
| 2026-06-12 | 8682 | 100.0% | 100.0% | 48.7% | 30.6% | 18.1% | 0.0% | 48.7% | 92.4% | 88.8% | 0.0% |
| 2026-06-11 | 2202 | 100.0% | 100.0% | 49.4% | 31.9% | 17.5% | 0.0% | 49.4% | 75.8% | 85.3% | 0.0% |
| 2026-06-10 | 8414 | 100.0% | 100.0% | 44.2% | 27.3% | 16.9% | 0.0% | 44.2% | 90.6% | 85.5% | 0.0% |
| 2026-06-09 | 6566 | 100.0% | 100.0% | 41.6% | 25.9% | 15.7% | 0.0% | 41.6% | 93.1% | 86.8% | 0.0% |
| 2026-06-08 | 3589 | 100.0% | 100.0% | 41.3% | 26.0% | 15.3% | 0.0% | 41.3% | 92.6% | 86.2% | 0.0% |
| 2026-06-07 | 6521 | 100.0% | 100.0% | 40.1% | 23.9% | 16.1% | 0.0% | 40.1% | 89.4% | 86.5% | 0.0% |
| 2026-06-06 | 4562 | 100.0% | 100.0% | 41.6% | 25.6% | 16.0% | 0.0% | 41.6% | 81.9% | 88.1% | 0.0% |
| 2026-06-05 | 6699 | 100.0% | 100.0% | 42.3% | 26.3% | 16.0% | 0.0% | 42.3% | 92.1% | 86.8% | 0.0% |
| 2026-06-04 | 2348 | 100.0% | 100.0% | 43.0% | 26.2% | 16.7% | 0.0% | 43.0% | 95.1% | 69.8% | 3.7% |
| 2026-06-03 | 3331 | 100.0% | 100.0% | 36.2% | 25.3% | 10.9% | 0.0% | 36.2% | 92.5% | 0.0% | 100.0% |
| 2026-06-02 | 235 | 0.0% | 100.0% | 90.2% | 89.8% | 0.4% | 0.0% | 90.2% | 91.1% | 85.1% | 0.0% |
| 2026-06-01 | 208 | 0.0% | 100.0% | 96.2% | 93.3% | 2.9% | 0.0% | 96.2% | 93.3% | 88.0% | 0.0% |

## Pairing By Date / Market / Book

| Date | Market | Book | Rows | Same-Book Pairs | Cross-Book Pairs | Synthetic Pairs | Missing Pairs | Same-Book Quality | Cross-Book Quality | Synthetic Quality | One-Sided Quality | Raw-Implied Prob | Synthetic Prob |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-30 | batter_total_bases | fanduel | 1116 | 0 | 176 | 0 | 940 | 0 | 176 | 0 | 940 | 0 | 0 |
| 2026-07-30 | batter_hits | fanduel | 585 | 0 | 320 | 0 | 265 | 0 | 320 | 0 | 265 | 0 | 0 |
| 2026-07-30 | batter_hits | draftkings | 564 | 564 | 0 | 0 | 0 | 564 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-30 | batter_home_runs | fanduel | 555 | 0 | 0 | 0 | 555 | 0 | 0 | 0 | 555 | 0 | 0 |
| 2026-07-30 | batter_total_bases | draftkings | 292 | 292 | 0 | 0 | 0 | 292 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-30 | pitcher_strikeouts | draftkings | 66 | 66 | 0 | 0 | 0 | 66 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-30 | pitcher_strikeouts | fanduel | 54 | 54 | 0 | 0 | 0 | 54 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-29 | batter_total_bases | fanduel | 2248 | 0 | 404 | 0 | 1844 | 0 | 404 | 0 | 1844 | 0 | 0 |
| 2026-07-29 | batter_hits | fanduel | 1238 | 0 | 778 | 0 | 460 | 0 | 778 | 0 | 460 | 0 | 0 |
| 2026-07-29 | batter_home_runs | fanduel | 1124 | 0 | 0 | 0 | 1124 | 0 | 0 | 0 | 1124 | 0 | 0 |
| 2026-07-29 | batter_hits | draftkings | 674 | 674 | 0 | 0 | 0 | 674 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-29 | batter_total_bases | draftkings | 302 | 302 | 0 | 0 | 0 | 302 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-29 | pitcher_strikeouts | fanduel | 106 | 106 | 0 | 0 | 0 | 106 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-29 | pitcher_strikeouts | draftkings | 64 | 64 | 0 | 0 | 0 | 64 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-28 | batter_total_bases | fanduel | 2737 | 0 | 578 | 0 | 2159 | 0 | 578 | 0 | 2159 | 0 | 0 |
| 2026-07-28 | batter_hits | fanduel | 1534 | 0 | 1009 | 0 | 525 | 0 | 1009 | 0 | 525 | 0 | 0 |
| 2026-07-28 | batter_hits | draftkings | 1410 | 1410 | 0 | 0 | 0 | 1410 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-28 | batter_home_runs | fanduel | 1388 | 0 | 0 | 0 | 1388 | 0 | 0 | 0 | 1388 | 0 | 0 |
| 2026-07-28 | batter_total_bases | draftkings | 688 | 688 | 0 | 0 | 0 | 688 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-28 | pitcher_strikeouts | draftkings | 148 | 148 | 0 | 0 | 0 | 148 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-28 | pitcher_strikeouts | fanduel | 138 | 138 | 0 | 0 | 0 | 138 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-27 | batter_total_bases | fanduel | 1944 | 0 | 409 | 0 | 1535 | 0 | 409 | 0 | 1535 | 0 | 0 |
| 2026-07-27 | batter_hits | fanduel | 1108 | 0 | 699 | 0 | 409 | 0 | 699 | 0 | 409 | 0 | 0 |
| 2026-07-27 | batter_hits | draftkings | 1044 | 1044 | 0 | 0 | 0 | 1044 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-27 | batter_home_runs | fanduel | 972 | 0 | 0 | 0 | 972 | 0 | 0 | 0 | 972 | 0 | 0 |
| 2026-07-27 | batter_total_bases | draftkings | 542 | 542 | 0 | 0 | 0 | 542 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-27 | pitcher_strikeouts | draftkings | 122 | 122 | 0 | 0 | 0 | 122 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-27 | pitcher_strikeouts | fanduel | 118 | 118 | 0 | 0 | 0 | 118 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-26 | batter_total_bases | fanduel | 1760 | 0 | 382 | 0 | 1378 | 0 | 382 | 0 | 1378 | 0 | 0 |
| 2026-07-26 | batter_hits | fanduel | 961 | 0 | 601 | 0 | 360 | 0 | 601 | 0 | 360 | 0 | 0 |
| 2026-07-26 | batter_hits | draftkings | 902 | 902 | 0 | 0 | 0 | 902 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-26 | batter_home_runs | fanduel | 880 | 0 | 0 | 0 | 880 | 0 | 0 | 0 | 880 | 0 | 0 |
| 2026-07-26 | batter_total_bases | draftkings | 444 | 444 | 0 | 0 | 0 | 444 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-26 | pitcher_strikeouts | draftkings | 116 | 116 | 0 | 0 | 0 | 116 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-26 | pitcher_strikeouts | fanduel | 112 | 112 | 0 | 0 | 0 | 112 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-25 | batter_total_bases | fanduel | 2324 | 0 | 430 | 0 | 1894 | 0 | 430 | 0 | 1894 | 0 | 0 |
| 2026-07-25 | batter_hits | fanduel | 1371 | 0 | 785 | 0 | 586 | 0 | 785 | 0 | 586 | 0 | 0 |
| 2026-07-25 | batter_hits | draftkings | 1212 | 1212 | 0 | 0 | 0 | 1212 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-25 | batter_home_runs | fanduel | 1162 | 0 | 0 | 0 | 1162 | 0 | 0 | 0 | 1162 | 0 | 0 |
| 2026-07-25 | batter_total_bases | draftkings | 520 | 520 | 0 | 0 | 0 | 520 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-25 | pitcher_strikeouts | draftkings | 146 | 146 | 0 | 0 | 0 | 146 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-25 | pitcher_strikeouts | fanduel | 128 | 128 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-24 | batter_total_bases | fanduel | 2564 | 0 | 511 | 0 | 2053 | 0 | 511 | 0 | 2053 | 0 | 0 |
| 2026-07-24 | batter_hits | fanduel | 1527 | 0 | 896 | 0 | 631 | 0 | 896 | 0 | 631 | 0 | 0 |
| 2026-07-24 | batter_hits | draftkings | 1350 | 1350 | 0 | 0 | 0 | 1350 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-24 | batter_home_runs | fanduel | 1281 | 0 | 0 | 0 | 1281 | 0 | 0 | 0 | 1281 | 0 | 0 |
| 2026-07-24 | batter_total_bases | draftkings | 608 | 608 | 0 | 0 | 0 | 608 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-24 | pitcher_strikeouts | draftkings | 120 | 120 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-24 | pitcher_strikeouts | fanduel | 110 | 110 | 0 | 0 | 0 | 110 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-23 | batter_total_bases | fanduel | 524 | 0 | 93 | 0 | 431 | 0 | 93 | 0 | 431 | 0 | 0 |
| 2026-07-23 | batter_hits | fanduel | 315 | 0 | 186 | 0 | 129 | 0 | 186 | 0 | 129 | 0 | 0 |
| 2026-07-23 | batter_hits | draftkings | 284 | 284 | 0 | 0 | 0 | 284 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-23 | batter_home_runs | fanduel | 262 | 0 | 0 | 0 | 262 | 0 | 0 | 0 | 262 | 0 | 0 |
| 2026-07-23 | batter_total_bases | draftkings | 118 | 118 | 0 | 0 | 0 | 118 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-23 | pitcher_strikeouts | draftkings | 34 | 34 | 0 | 0 | 0 | 34 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-23 | pitcher_strikeouts | fanduel | 32 | 32 | 0 | 0 | 0 | 32 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-22 | batter_total_bases | fanduel | 2532 | 0 | 556 | 0 | 1976 | 0 | 556 | 0 | 1976 | 0 | 0 |
| 2026-07-22 | batter_hits | fanduel | 1404 | 0 | 895 | 0 | 509 | 0 | 895 | 0 | 509 | 0 | 0 |
| 2026-07-22 | batter_home_runs | fanduel | 1266 | 0 | 0 | 0 | 1266 | 0 | 0 | 0 | 1266 | 0 | 0 |
| 2026-07-22 | batter_hits | draftkings | 1168 | 1168 | 0 | 0 | 0 | 1168 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-22 | batter_total_bases | draftkings | 606 | 606 | 0 | 0 | 0 | 606 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-22 | pitcher_strikeouts | fanduel | 144 | 144 | 0 | 0 | 0 | 144 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-22 | pitcher_strikeouts | draftkings | 128 | 128 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-21 | batter_total_bases | fanduel | 2656 | 0 | 548 | 0 | 2108 | 0 | 548 | 0 | 2108 | 0 | 0 |
| 2026-07-21 | batter_hits | fanduel | 1467 | 0 | 882 | 0 | 585 | 0 | 882 | 0 | 585 | 0 | 0 |
| 2026-07-21 | batter_hits | draftkings | 1378 | 1378 | 0 | 0 | 0 | 1378 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-21 | batter_home_runs | fanduel | 1328 | 0 | 0 | 0 | 1328 | 0 | 0 | 0 | 1328 | 0 | 0 |
| 2026-07-21 | batter_total_bases | draftkings | 722 | 722 | 0 | 0 | 0 | 722 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-21 | pitcher_strikeouts | draftkings | 176 | 176 | 0 | 0 | 0 | 176 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-21 | pitcher_strikeouts | fanduel | 166 | 166 | 0 | 0 | 0 | 166 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-20 | batter_total_bases | fanduel | 2528 | 0 | 569 | 0 | 1959 | 0 | 569 | 0 | 1959 | 0 | 0 |
| 2026-07-20 | batter_hits | fanduel | 1494 | 0 | 941 | 0 | 553 | 0 | 941 | 0 | 553 | 0 | 0 |
| 2026-07-20 | batter_hits | draftkings | 1340 | 1340 | 0 | 0 | 0 | 1340 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-20 | batter_home_runs | fanduel | 1264 | 0 | 0 | 0 | 1264 | 0 | 0 | 0 | 1264 | 0 | 0 |
| 2026-07-20 | batter_total_bases | draftkings | 688 | 688 | 0 | 0 | 0 | 688 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-20 | pitcher_strikeouts | draftkings | 170 | 170 | 0 | 0 | 0 | 170 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-20 | pitcher_strikeouts | fanduel | 154 | 154 | 0 | 0 | 0 | 154 | 0 | 0 | 0 | 0 | 0 |
| 2026-07-19 | batter_total_bases | fanduel | 1872 | 0 | 417 | 0 | 1455 | 0 | 417 | 0 | 1455 | 0 | 0 |
| 2026-07-19 | batter_hits | fanduel | 993 | 0 | 662 | 0 | 331 | 0 | 662 | 0 | 331 | 0 | 0 |
| 2026-07-19 | batter_hits | draftkings | 950 | 950 | 0 | 0 | 0 | 950 | 0 | 0 | 0 | 0 | 0 |

## FanDuel Hitter Market Evidence

| Date | Market | Rows | True Pair | Synthetic | Clean Evidence | Same-Book | Cross-Book | Synthetic Rows | Action |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-07-30 | batter_total_bases | 1116 | 15.8% | 0.0% | 15.8% | 0 | 176 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-30 | batter_home_runs | 555 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-30 | batter_hits | 585 | 54.7% | 0.0% | 54.7% | 0 | 320 | 0 | usable |
| 2026-07-29 | batter_total_bases | 2248 | 18.0% | 0.0% | 18.0% | 0 | 404 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-29 | batter_home_runs | 1124 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-29 | batter_hits | 1238 | 62.8% | 0.0% | 62.8% | 0 | 778 | 0 | usable |
| 2026-07-28 | batter_total_bases | 2737 | 21.1% | 0.0% | 21.1% | 0 | 578 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-28 | batter_home_runs | 1388 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-28 | batter_hits | 1534 | 65.8% | 0.0% | 65.8% | 0 | 1009 | 0 | usable |
| 2026-07-27 | batter_total_bases | 1944 | 21.0% | 0.0% | 21.0% | 0 | 409 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-27 | batter_home_runs | 972 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-27 | batter_hits | 1108 | 63.1% | 0.0% | 63.1% | 0 | 699 | 0 | usable |
| 2026-07-26 | batter_total_bases | 1760 | 21.7% | 0.0% | 21.7% | 0 | 382 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-26 | batter_home_runs | 880 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-26 | batter_hits | 961 | 62.5% | 0.0% | 62.5% | 0 | 601 | 0 | usable |
| 2026-07-25 | batter_total_bases | 2324 | 18.5% | 0.0% | 18.5% | 0 | 430 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-25 | batter_home_runs | 1162 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-25 | batter_hits | 1371 | 57.3% | 0.0% | 57.3% | 0 | 785 | 0 | usable |
| 2026-07-24 | batter_total_bases | 2564 | 19.9% | 0.0% | 19.9% | 0 | 511 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-24 | batter_home_runs | 1281 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-24 | batter_hits | 1527 | 58.7% | 0.0% | 58.7% | 0 | 896 | 0 | usable |
| 2026-07-23 | batter_total_bases | 524 | 17.7% | 0.0% | 17.7% | 0 | 93 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-23 | batter_home_runs | 262 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-23 | batter_hits | 315 | 59.0% | 0.0% | 59.0% | 0 | 186 | 0 | usable |
| 2026-07-22 | batter_total_bases | 2532 | 22.0% | 0.0% | 22.0% | 0 | 556 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-22 | batter_home_runs | 1266 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-22 | batter_hits | 1404 | 63.7% | 0.0% | 63.7% | 0 | 895 | 0 | usable |
| 2026-07-21 | batter_total_bases | 2656 | 20.6% | 0.0% | 20.6% | 0 | 548 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-21 | batter_home_runs | 1328 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-21 | batter_hits | 1467 | 60.1% | 0.0% | 60.1% | 0 | 882 | 0 | usable |
| 2026-07-20 | batter_total_bases | 2528 | 22.5% | 0.0% | 22.5% | 0 | 569 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-20 | batter_home_runs | 1264 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-20 | batter_hits | 1494 | 63.0% | 0.0% | 63.0% | 0 | 941 | 0 | usable |
| 2026-07-19 | batter_total_bases | 1872 | 22.3% | 0.0% | 22.3% | 0 | 417 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-19 | batter_home_runs | 936 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-19 | batter_hits | 993 | 66.7% | 0.0% | 66.7% | 0 | 662 | 0 | usable |
| 2026-07-18 | batter_total_bases | 2080 | 23.7% | 0.0% | 23.7% | 0 | 492 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-18 | batter_home_runs | 1040 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-18 | batter_hits | 1067 | 71.8% | 0.0% | 71.8% | 0 | 766 | 0 | usable |
| 2026-07-17 | batter_total_bases | 2636 | 21.9% | 0.0% | 21.9% | 0 | 577 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-17 | batter_home_runs | 1316 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-17 | batter_hits | 1427 | 66.9% | 0.0% | 66.9% | 0 | 954 | 0 | usable |
| 2026-07-16 | batter_total_bases | 180 | 21.7% | 0.0% | 21.7% | 0 | 39 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-16 | batter_home_runs | 90 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-16 | batter_hits | 119 | 52.9% | 0.0% | 52.9% | 0 | 63 | 0 | usable |
| 2026-07-12 | batter_total_bases | 2600 | 21.3% | 0.0% | 21.3% | 0 | 553 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-12 | batter_home_runs | 1299 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-12 | batter_hits | 1470 | 63.9% | 0.0% | 63.9% | 0 | 939 | 0 | usable |
| 2026-07-11 | batter_total_bases | 2292 | 20.5% | 0.0% | 20.5% | 0 | 470 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-11 | batter_home_runs | 1145 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-11 | batter_hits | 1294 | 64.1% | 0.0% | 64.1% | 0 | 830 | 0 | usable |
| 2026-07-10 | batter_total_bases | 2736 | 21.4% | 0.0% | 21.4% | 0 | 586 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-10 | batter_home_runs | 1368 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-10 | batter_hits | 1604 | 59.6% | 0.0% | 59.6% | 0 | 956 | 0 | usable |
| 2026-07-09 | batter_total_bases | 2004 | 19.5% | 0.0% | 19.5% | 0 | 391 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-09 | batter_home_runs | 1001 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-09 | batter_hits | 1177 | 59.6% | 0.0% | 59.6% | 0 | 702 | 0 | usable |
| 2026-07-08 | batter_total_bases | 2672 | 21.7% | 0.0% | 21.7% | 0 | 580 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-08 | batter_home_runs | 1336 | 0.0% | 0.0% | 0.0% | 0 | 0 | 0 | extract_true_opposite_side_or_demote |
| 2026-07-08 | batter_hits | 1591 | 61.7% | 0.0% | 61.7% | 0 | 982 | 0 | usable |

## Problem Examples

| Date | Player | Bet | Missing Fields | Pairing Note | CLV Status | CLV Reason |
|---|---|---|---|---|---|---|
| 2026-06-01 | Mike Trout | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Jorge Soler | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Willi Castro | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Jake McCarthy | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Jo Adell | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Edouard Julien | batter_hits over 0.5 fanduel | prop_offer_id | cross_book_pair | true_no_movement |  |
| 2026-06-01 | Tyler Freeman | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Oswald Peraza | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Ezequiel Tovar | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | TJ Rumfield | batter_hits over 0.5 draftkings | prop_offer_id |  | unknown | player_prop_unavailable_at_close |
| 2026-06-01 | Logan O'Hoppe | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Troy Johnston | batter_hits over 0.5 draftkings | prop_offer_id |  | true_no_movement |  |
| 2026-06-01 | Kyle Karros | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Sterlin Thompson | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Hunter Goodman | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Mike Trout | batter_total_bases over 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Jorge Soler | batter_total_bases over 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Jo Adell | batter_total_bases over 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Tyler Freeman | batter_total_bases under 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Oswald Peraza | batter_total_bases over 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Logan O'Hoppe | batter_total_bases over 1.5 draftkings | prop_offer_id |  | unknown | fallback_other_book_only |
| 2026-06-01 | Vaughn Grissom | batter_total_bases over 1.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Zach Neto | batter_total_bases over 1.5 draftkings | prop_offer_id, actual_value, won |  | true_no_movement |  |
| 2026-06-01 | Kyle Freeland | pitcher_strikeouts under 4.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | José Soriano | pitcher_strikeouts over 6.5 draftkings | prop_offer_id |  | unknown | line_disappeared_at_close |
| 2026-06-01 | Josh Bell | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Victor Caratini | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Byron Buxton | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Andrew Benintendi | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-06-01 | Trevor Larnach | batter_hits over 0.5 draftkings | prop_offer_id |  | valid_movement |  |
