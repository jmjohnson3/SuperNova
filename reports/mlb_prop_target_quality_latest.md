# MLB Prop Target Quality

Generated UTC: 2026-06-12T15:52:16Z
Rows: 50347
Date range: 2026-05-31 to 2026-06-12

## Required Field Coverage

| Field | Present | Missing | Coverage |
|---|---:|---:|---:|
| bookmaker_key | 50347 | 0 | 100.0% |
| market_line | 50347 | 0 | 100.0% |
| market_price | 50347 | 0 | 100.0% |
| prop_offer_id | 49690 | 657 | 98.7% |
| lock_snapshot_id | 50347 | 0 | 100.0% |
| source_created_at | 50347 | 0 | 100.0% |
| actual_value | 40380 | 9967 | 80.2% |
| won | 40380 | 9967 | 80.2% |
| paired_price | 50347 | 0 | 100.0% |
| paired_bookmaker_key | 50347 | 0 | 100.0% |
| paired_price_source | 50347 | 0 | 100.0% |
| pair_quality | 50347 | 0 | 100.0% |
| no_vig_market_prob | 50347 | 0 | 100.0% |
| market_prob_source | 50347 | 0 | 100.0% |
| closing_line | 35545 | 14802 | 70.6% |
| closing_price | 35545 | 14802 | 70.6% |
| closing_snapshot_id | 35545 | 14802 | 70.6% |
| closing_fetched_at_utc | 35545 | 14802 | 70.6% |
| clv_valid | 50347 | 0 | 100.0% |

## CLV / Close Status

| Status | Rows |
|---|---:|
| valid_movement | 24737 |
| unknown | 14802 |
| true_no_movement | 10808 |

## CLV Unknown Reasons

| Reason | Rows |
|---|---:|
| none | 35545 |
| close_outside_two_hour_window | 9352 |
| stale_close_before_lock | 5024 |
| fallback_other_book_only | 310 |
| no_valid_close_snapshot | 116 |

## Quality By Date

| Date | Rows | Offer ID | Price+Lock | True Pair | Same-Book Pair | Cross-Book Pair | Synthetic Pair | Any Pair | Graded | Valid Close | Stale Close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-12 | 5458 | 100.0% | 100.0% | 44.2% | 30.8% | 13.4% | 55.8% | 100.0% | 0.0% | 0.0% | 0.1% |
| 2026-06-11 | 2202 | 100.0% | 100.0% | 49.4% | 31.9% | 17.5% | 50.6% | 100.0% | 75.8% | 85.3% | 6.7% |
| 2026-06-10 | 8414 | 100.0% | 100.0% | 44.2% | 27.3% | 16.9% | 55.8% | 100.0% | 90.6% | 85.5% | 4.5% |
| 2026-06-09 | 6566 | 100.0% | 100.0% | 41.6% | 25.9% | 15.7% | 58.4% | 100.0% | 93.1% | 86.8% | 3.5% |
| 2026-06-08 | 3589 | 100.0% | 100.0% | 41.3% | 26.0% | 15.3% | 58.7% | 100.0% | 92.6% | 86.2% | 3.1% |
| 2026-06-07 | 6521 | 100.0% | 100.0% | 40.1% | 23.9% | 16.1% | 59.9% | 100.0% | 89.4% | 86.5% | 1.0% |
| 2026-06-06 | 4562 | 100.0% | 100.0% | 41.6% | 25.6% | 16.0% | 58.4% | 100.0% | 81.9% | 88.1% | 2.2% |
| 2026-06-05 | 6699 | 100.0% | 100.0% | 42.3% | 26.3% | 16.0% | 57.7% | 100.0% | 92.1% | 86.8% | 4.2% |
| 2026-06-04 | 2348 | 100.0% | 100.0% | 43.0% | 26.2% | 16.7% | 57.0% | 100.0% | 95.1% | 69.8% | 13.7% |
| 2026-06-03 | 3331 | 100.0% | 100.0% | 36.2% | 25.3% | 10.9% | 63.8% | 100.0% | 92.5% | 0.0% | 100.0% |
| 2026-06-02 | 235 | 0.0% | 100.0% | 90.2% | 89.8% | 0.4% | 9.8% | 100.0% | 91.1% | 85.1% | 0.0% |
| 2026-06-01 | 208 | 0.0% | 100.0% | 96.2% | 93.3% | 2.9% | 3.8% | 100.0% | 93.3% | 88.0% | 9.6% |
| 2026-05-31 | 214 | 0.0% | 100.0% | 92.1% | 92.1% | 0.0% | 7.9% | 100.0% | 92.1% | 83.6% | 15.0% |

## Pairing By Date / Market / Book

| Date | Market | Book | Rows | Same-Book Pairs | Cross-Book Pairs | Synthetic Pairs | Missing Pairs | Same-Book Quality | Cross-Book Quality | Synthetic Quality | One-Sided Quality | Raw-Implied Prob | Synthetic Prob |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-12 | batter_total_bases | fanduel | 1888 | 0 | 254 | 1634 | 0 | 0 | 254 | 1634 | 0 | 0 | 1634 |
| 2026-06-12 | batter_hits | draftkings | 966 | 966 | 0 | 0 | 0 | 966 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | batter_hits | fanduel | 944 | 0 | 478 | 466 | 0 | 0 | 478 | 466 | 0 | 0 | 466 |
| 2026-06-12 | batter_home_runs | fanduel | 944 | 0 | 0 | 944 | 0 | 0 | 0 | 944 | 0 | 0 | 944 |
| 2026-06-12 | batter_total_bases | draftkings | 502 | 502 | 0 | 0 | 0 | 502 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | pitcher_strikeouts | fanduel | 108 | 108 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-12 | pitcher_strikeouts | draftkings | 106 | 106 | 0 | 0 | 0 | 106 | 0 | 0 | 0 | 0 | 0 |
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
| 2026-06-08 | pitcher_strikeouts | draftkings | 68 | 68 | 0 | 0 | 0 | 68 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-08 | pitcher_strikeouts | fanduel | 56 | 56 | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-07 | batter_total_bases | fanduel | 1996 | 0 | 390 | 1606 | 0 | 0 | 390 | 1606 | 0 | 0 | 1606 |
| 2026-06-07 | batter_hits | fanduel | 1967 | 0 | 663 | 1304 | 0 | 0 | 663 | 1304 | 0 | 0 | 1304 |
| 2026-06-07 | batter_hits | draftkings | 1034 | 1034 | 0 | 0 | 0 | 1034 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-07 | batter_home_runs | fanduel | 998 | 0 | 0 | 998 | 0 | 0 | 0 | 998 | 0 | 0 | 998 |
| 2026-06-07 | batter_total_bases | draftkings | 526 | 526 | 0 | 0 | 0 | 526 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-06 | batter_total_bases | fanduel | 1364 | 0 | 257 | 1107 | 0 | 0 | 257 | 1107 | 0 | 0 | 1107 |
| 2026-06-06 | batter_hits | fanduel | 1348 | 0 | 471 | 877 | 0 | 0 | 471 | 877 | 0 | 0 | 877 |
| 2026-06-06 | batter_hits | draftkings | 682 | 682 | 0 | 0 | 0 | 682 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-06 | batter_home_runs | fanduel | 682 | 0 | 0 | 682 | 0 | 0 | 0 | 682 | 0 | 0 | 682 |
| 2026-06-06 | batter_total_bases | draftkings | 336 | 336 | 0 | 0 | 0 | 336 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-06 | pitcher_strikeouts | draftkings | 80 | 80 | 0 | 0 | 0 | 80 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-06 | pitcher_strikeouts | fanduel | 70 | 70 | 0 | 0 | 0 | 70 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-05 | batter_total_bases | fanduel | 1980 | 0 | 372 | 1608 | 0 | 0 | 372 | 1608 | 0 | 0 | 1608 |
| 2026-06-05 | batter_hits | fanduel | 1953 | 0 | 698 | 1255 | 0 | 0 | 698 | 1255 | 0 | 0 | 1255 |
| 2026-06-05 | batter_hits | draftkings | 1026 | 1026 | 0 | 0 | 0 | 1026 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-05 | batter_home_runs | fanduel | 1004 | 0 | 0 | 1004 | 0 | 0 | 0 | 1004 | 0 | 0 | 1004 |
| 2026-06-05 | batter_total_bases | draftkings | 506 | 506 | 0 | 0 | 0 | 506 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-05 | pitcher_strikeouts | draftkings | 122 | 122 | 0 | 0 | 0 | 122 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-05 | pitcher_strikeouts | fanduel | 108 | 108 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-04 | batter_total_bases | fanduel | 696 | 0 | 144 | 552 | 0 | 0 | 144 | 552 | 0 | 0 | 552 |
| 2026-06-04 | batter_hits | fanduel | 688 | 0 | 249 | 439 | 0 | 0 | 249 | 439 | 0 | 0 | 439 |
| 2026-06-04 | batter_hits | draftkings | 362 | 362 | 0 | 0 | 0 | 362 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-04 | batter_home_runs | fanduel | 348 | 0 | 0 | 348 | 0 | 0 | 0 | 348 | 0 | 0 | 348 |
| 2026-06-04 | batter_total_bases | draftkings | 206 | 206 | 0 | 0 | 0 | 206 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-04 | pitcher_strikeouts | draftkings | 24 | 24 | 0 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-04 | pitcher_strikeouts | fanduel | 24 | 24 | 0 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-03 | batter_total_bases | fanduel | 1008 | 0 | 114 | 894 | 0 | 0 | 114 | 894 | 0 | 0 | 894 |
| 2026-06-03 | batter_hits | fanduel | 978 | 0 | 250 | 728 | 0 | 0 | 250 | 728 | 0 | 0 | 728 |
| 2026-06-03 | batter_hits | draftkings | 506 | 506 | 0 | 0 | 0 | 506 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-03 | batter_home_runs | fanduel | 503 | 0 | 0 | 503 | 0 | 0 | 0 | 503 | 0 | 0 | 503 |
| 2026-06-03 | batter_total_bases | draftkings | 228 | 228 | 0 | 0 | 0 | 228 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-03 | pitcher_strikeouts | draftkings | 54 | 54 | 0 | 0 | 0 | 54 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-03 | pitcher_strikeouts | fanduel | 54 | 54 | 0 | 0 | 0 | 54 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-02 | batter_hits | draftkings | 114 | 114 | 0 | 0 | 0 | 114 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-02 | batter_total_bases | draftkings | 78 | 78 | 0 | 0 | 0 | 78 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-02 | batter_home_runs | fanduel | 20 | 0 | 0 | 20 | 0 | 0 | 0 | 20 | 0 | 0 | 20 |
| 2026-06-02 | pitcher_strikeouts | draftkings | 19 | 19 | 0 | 0 | 0 | 19 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-02 | batter_total_bases | fanduel | 4 | 0 | 1 | 3 | 0 | 0 | 1 | 3 | 0 | 0 | 3 |
| 2026-06-01 | batter_hits | draftkings | 136 | 136 | 0 | 0 | 0 | 136 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-01 | batter_total_bases | draftkings | 44 | 44 | 0 | 0 | 0 | 44 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-01 | pitcher_strikeouts | draftkings | 14 | 14 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 0 | 0 |
| 2026-06-01 | batter_total_bases | fanduel | 10 | 0 | 4 | 6 | 0 | 0 | 4 | 6 | 0 | 0 | 6 |
| 2026-06-01 | batter_hits | fanduel | 4 | 0 | 2 | 2 | 0 | 0 | 2 | 2 | 0 | 0 | 2 |
| 2026-05-31 | batter_hits | draftkings | 108 | 108 | 0 | 0 | 0 | 108 | 0 | 0 | 0 | 0 | 0 |
| 2026-05-31 | batter_total_bases | draftkings | 69 | 69 | 0 | 0 | 0 | 69 | 0 | 0 | 0 | 0 | 0 |

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
| 2026-05-31 | Ozzie Albies | batter_hits under 0.5 draftkings | prop_offer_id |  | unknown | stale_close_before_lock |
| 2026-05-31 | Austin Riley | batter_hits under 0.5 draftkings | prop_offer_id |  | valid_movement |  |
| 2026-05-31 | Michael Harris II | batter_hits under 0.5 draftkings | prop_offer_id |  | unknown | stale_close_before_lock |
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
