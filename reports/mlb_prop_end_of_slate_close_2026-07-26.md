# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-27T04:11:57Z
Slate: 2026-07-26
Evaluation: **FINAL**
Games final: 15 / 15
Strict clean slate: **False**

- Locked executable offers: 2603
- Valid exact closes: 2231 (85.7%)
- Additional valid closes needed for 90%: 112
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 13 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 112 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 2 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 328 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 28 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| fallback_other_book_only | 9 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| player_market_unavailable_at_close | 7 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2231 |
| player_prop_unavailable_at_close | 328 |
| line_disappeared_at_close | 28 |
| fallback_other_book_only | 9 |
| player_market_unavailable_at_close | 7 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| a9c8aad6e31f322b330deb1a7fdd3eb3 | fanduel | batter_total_bases | 56 | 20 | 64.3% | {"player_prop_unavailable_at_close": 20, "valid_close": 36} |
| 9064f61fe767dfa3c4a613a4f02250d3 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| b61e67dbd760d186d865cc2ff013a7c0 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| 3f5e40b78b4be8ec30a552f1081e4dc0 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| 695de475f6a8dd1f205bda6ba50db0ee | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| c66c0c97123e0bc5e1a3c1b170c9b269 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 714c1847e9003da34be408b74dca72e3 | fanduel | batter_total_bases | 76 | 12 | 84.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 64} |
| d651270eb53d061fc93a471f11c2f53d | fanduel | batter_total_bases | 76 | 12 | 84.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 64} |
| a9c8aad6e31f322b330deb1a7fdd3eb3 | draftkings | batter_hits | 28 | 10 | 64.3% | {"player_prop_unavailable_at_close": 10, "valid_close": 18} |
| a9c8aad6e31f322b330deb1a7fdd3eb3 | fanduel | batter_hits | 28 | 10 | 64.3% | {"player_prop_unavailable_at_close": 10, "valid_close": 18} |
| a9c8aad6e31f322b330deb1a7fdd3eb3 | fanduel | batter_home_runs | 28 | 10 | 64.3% | {"player_prop_unavailable_at_close": 10, "valid_close": 18} |
| 17b69aed6b1c1715ffbd01b155af848d | draftkings | batter_hits | 32 | 10 | 68.8% | {"line_disappeared_at_close": 8, "player_prop_unavailable_at_close": 2, "valid_close": 22} |
| a9c8aad6e31f322b330deb1a7fdd3eb3 | draftkings | batter_total_bases | 18 | 8 | 55.6% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 6, "valid_close": 10} |
| d651270eb53d061fc93a471f11c2f53d | fanduel | batter_home_runs | 38 | 7 | 81.6% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 6, "valid_close": 31} |
| d651270eb53d061fc93a471f11c2f53d | fanduel | batter_hits | 66 | 7 | 89.4% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 6, "valid_close": 59} |
| b61e67dbd760d186d865cc2ff013a7c0 | draftkings | batter_total_bases | 12 | 6 | 50.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 6} |
| 48df7c31ab5c07dc2c7073170f193a8f | draftkings | batter_hits | 26 | 6 | 76.9% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 20} |
| 714c1847e9003da34be408b74dca72e3 | draftkings | batter_total_bases | 26 | 6 | 76.9% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 9064f61fe767dfa3c4a613a4f02250d3 | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 9064f61fe767dfa3c4a613a4f02250d3 | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 9064f61fe767dfa3c4a613a4f02250d3 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| b61e67dbd760d186d865cc2ff013a7c0 | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| b61e67dbd760d186d865cc2ff013a7c0 | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| b61e67dbd760d186d865cc2ff013a7c0 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 3f5e40b78b4be8ec30a552f1081e4dc0 | draftkings | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 3f5e40b78b4be8ec30a552f1081e4dc0 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 3f5e40b78b4be8ec30a552f1081e4dc0 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| c66c0c97123e0bc5e1a3c1b170c9b269 | draftkings | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 695de475f6a8dd1f205bda6ba50db0ee | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| c66c0c97123e0bc5e1a3c1b170c9b269 | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| c66c0c97123e0bc5e1a3c1b170c9b269 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 695de475f6a8dd1f205bda6ba50db0ee | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 714c1847e9003da34be408b74dca72e3 | draftkings | batter_hits | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| 714c1847e9003da34be408b74dca72e3 | fanduel | batter_hits | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| 714c1847e9003da34be408b74dca72e3 | fanduel | batter_home_runs | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| d651270eb53d061fc93a471f11c2f53d | draftkings | batter_hits | 40 | 6 | 85.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 34} |
| 695de475f6a8dd1f205bda6ba50db0ee | fanduel | batter_hits | 61 | 6 | 90.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 55} |
| 17b69aed6b1c1715ffbd01b155af848d | draftkings | batter_total_bases | 16 | 4 | 75.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 12} |
| 48df7c31ab5c07dc2c7073170f193a8f | draftkings | batter_total_bases | 16 | 4 | 75.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 12} |
| 17b69aed6b1c1715ffbd01b155af848d | fanduel | batter_total_bases | 48 | 4 | 91.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 44} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| CLE @ TB | 2026-07-26T16:16:00+00:00 | yes | yes | yes | 32 |
| CHC @ PIT | 2026-07-26T17:36:00+00:00 | yes | yes | yes | 43 |
| TOR @ BOS | 2026-07-26T17:36:00+00:00 | yes | yes | yes | 43 |
| ARI @ WAS | 2026-07-26T17:36:00+00:00 | yes | yes | yes | 45 |
| ATL @ BAL | 2026-07-26T17:36:00+00:00 | yes | yes | yes | 45 |
| LAD @ NYM | 2026-07-26T17:41:00+00:00 | yes | yes | yes | 45 |
| SD @ MIA | 2026-07-26T17:41:00+00:00 | yes | yes | yes | 43 |
| KC @ DET | 2026-07-26T17:43:00+00:00 | yes | yes | yes | 45 |
| ATH @ MIN | 2026-07-26T18:11:00+00:00 | yes | yes | yes | 45 |
| HOU @ CWS | 2026-07-26T18:11:00+00:00 | yes | yes | yes | 45 |
| COL @ MIL | 2026-07-26T18:11:00+00:00 | yes | yes | yes | 45 |
| CIN @ STL | 2026-07-26T18:16:00+00:00 | yes | yes | yes | 45 |
| SEA @ TEX | 2026-07-26T18:36:00+00:00 | yes | yes | yes | 49 |
| LAA @ SF | 2026-07-26T20:06:00+00:00 | no | no | yes | 59 |
| NYY @ PHI | 2026-07-26T23:21:00+00:00 | no | yes | no | 65 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 2 |
| t_minus_60 | 1 |
| t_minus_20 | 1 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| LAA @ SF | 2026-07-26T20:06:00+00:00 | t_minus_120, t_minus_60 | 59 |
| NYY @ PHI | 2026-07-26T23:21:00+00:00 | t_minus_120, t_minus_20 | 65 |
