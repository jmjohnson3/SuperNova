# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-25T03:09:50Z
Slate: 2026-07-24
Evaluation: **PROVISIONAL**
Games final: 11 / 15
Strict clean slate: **False**

- Locked executable offers: 2891
- Valid exact closes: 2513 (86.9%)
- Additional valid closes needed for 90%: 89
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 7 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 89 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 8 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 312 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 38 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 14 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 14 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2513 |
| player_prop_unavailable_at_close | 312 |
| line_disappeared_at_close | 38 |
| player_market_unavailable_at_close | 14 |
| fallback_other_book_only | 14 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 2192276baf2c4e8e5358e9905c801e2f | fanduel | batter_total_bases | 52 | 16 | 69.2% | {"player_prop_unavailable_at_close": 16, "valid_close": 36} |
| 29437f5a0e0b0af43535ee03d438df64 | fanduel | batter_total_bases | 72 | 16 | 77.8% | {"player_prop_unavailable_at_close": 16, "valid_close": 56} |
| 7623b2ff95daaba1c14165f508c506b8 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| d7f4b3c3080d3f6612a4be54f7e0f798 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 74d637de5babda95c445e8cf11c32a15 | fanduel | batter_total_bases | 72 | 12 | 83.3% | {"player_prop_unavailable_at_close": 12, "valid_close": 60} |
| 12a8e27a29d44d46e10583c9c875e5f5 | fanduel | batter_total_bases | 76 | 12 | 84.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 64} |
| 2192276baf2c4e8e5358e9905c801e2f | fanduel | batter_hits | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| 2192276baf2c4e8e5358e9905c801e2f | fanduel | batter_home_runs | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| 2192276baf2c4e8e5358e9905c801e2f | draftkings | batter_hits | 28 | 8 | 71.4% | {"player_prop_unavailable_at_close": 8, "valid_close": 20} |
| 29437f5a0e0b0af43535ee03d438df64 | draftkings | batter_hits | 36 | 8 | 77.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 28} |
| 29437f5a0e0b0af43535ee03d438df64 | fanduel | batter_home_runs | 36 | 8 | 77.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 28} |
| 5dc243f04181b459d0cc18968be70660 | draftkings | batter_hits | 36 | 8 | 77.8% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 74d637de5babda95c445e8cf11c32a15 | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 30} |
| 12a8e27a29d44d46e10583c9c875e5f5 | draftkings | batter_hits | 40 | 8 | 80.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 32} |
| 29437f5a0e0b0af43535ee03d438df64 | fanduel | batter_hits | 62 | 8 | 87.1% | {"player_prop_unavailable_at_close": 8, "valid_close": 54} |
| 5dc243f04181b459d0cc18968be70660 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| e52022e436fbab769c0e31e1eec70f53 | draftkings | batter_total_bases | 14 | 6 | 57.1% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "valid_close": 8} |
| 12a8e27a29d44d46e10583c9c875e5f5 | draftkings | batter_total_bases | 26 | 6 | 76.9% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 6f9f1f656034bccc0a22e55a976cdf3b | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 7623b2ff95daaba1c14165f508c506b8 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 7623b2ff95daaba1c14165f508c506b8 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 7cb10d307b371751677d513fb168b3bd | draftkings | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| cee011b0978086fc2ebee48d455d495e | draftkings | batter_hits | 30 | 6 | 80.0% | {"line_disappeared_at_close": 6, "valid_close": 24} |
| d7f4b3c3080d3f6612a4be54f7e0f798 | draftkings | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| d7f4b3c3080d3f6612a4be54f7e0f798 | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| d7f4b3c3080d3f6612a4be54f7e0f798 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 263aac2dfd923f6a65c5136418407715 | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 74d637de5babda95c445e8cf11c32a15 | fanduel | batter_home_runs | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| 12a8e27a29d44d46e10583c9c875e5f5 | fanduel | batter_home_runs | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| 74d637de5babda95c445e8cf11c32a15 | fanduel | batter_hits | 66 | 6 | 90.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 60} |
| 12a8e27a29d44d46e10583c9c875e5f5 | fanduel | batter_hits | 69 | 6 | 91.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 63} |
| 29437f5a0e0b0af43535ee03d438df64 | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| cee011b0978086fc2ebee48d455d495e | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| 7623b2ff95daaba1c14165f508c506b8 | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 6} |
| e25ed72c9f5c072ef60c242c29ccf1a3 | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 6} |
| 29437f5a0e0b0af43535ee03d438df64 | draftkings | batter_total_bases | 12 | 4 | 66.7% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 8} |
| 5859ce305cb0ae65e5e7f771db28dcdd | draftkings | batter_total_bases | 12 | 4 | 66.7% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 8} |
| d7f4b3c3080d3f6612a4be54f7e0f798 | draftkings | batter_total_bases | 16 | 4 | 75.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 12} |
| 5dc243f04181b459d0cc18968be70660 | draftkings | batter_total_bases | 18 | 4 | 77.8% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 14} |
| 7623b2ff95daaba1c14165f508c506b8 | draftkings | batter_hits | 26 | 4 | 84.6% | {"player_prop_unavailable_at_close": 4, "valid_close": 22} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| COL @ MIL | 2026-07-24T20:11:00+00:00 | yes | yes | yes | 35 |
| CHC @ PIT | 2026-07-24T22:41:00+00:00 | yes | yes | yes | 44 |
| KC @ DET | 2026-07-24T22:41:00+00:00 | yes | yes | yes | 42 |
| TOR @ BOS | 2026-07-24T23:16:00+00:00 | yes | yes | yes | 47 |
| HOU @ CWS | 2026-07-24T23:41:00+00:00 | yes | yes | yes | 47 |
| CIN @ STL | 2026-07-25T00:16:00+00:00 | yes | yes | yes | 47 |
| LAA @ SF | 2026-07-25T02:16:00+00:00 | yes | yes | yes | 47 |
| ARI @ WAS | 2026-07-24T22:46:00+00:00 | no | yes | yes | 44 |
| NYY @ PHI | 2026-07-24T22:46:00+00:00 | no | yes | yes | 44 |
| ATL @ BAL | 2026-07-24T23:06:00+00:00 | no | yes | yes | 46 |
| CLE @ TB | 2026-07-24T23:11:00+00:00 | no | yes | yes | 46 |
| SD @ MIA | 2026-07-24T23:11:00+00:00 | no | yes | yes | 46 |
| LAD @ NYM | 2026-07-24T23:11:00+00:00 | no | yes | yes | 46 |
| SEA @ TEX | 2026-07-25T00:06:00+00:00 | yes | no | yes | 47 |
| ATH @ MIN | 2026-07-25T00:11:00+00:00 | yes | no | yes | 47 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 6 |
| t_minus_60 | 2 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| ARI @ WAS | 2026-07-24T22:46:00+00:00 | t_minus_120 | 44 |
| NYY @ PHI | 2026-07-24T22:46:00+00:00 | t_minus_120 | 44 |
| ATL @ BAL | 2026-07-24T23:06:00+00:00 | t_minus_120 | 46 |
| CLE @ TB | 2026-07-24T23:11:00+00:00 | t_minus_120 | 46 |
| SD @ MIA | 2026-07-24T23:11:00+00:00 | t_minus_120 | 46 |
| LAD @ NYM | 2026-07-24T23:11:00+00:00 | t_minus_120 | 46 |
| SEA @ TEX | 2026-07-25T00:06:00+00:00 | t_minus_60 | 47 |
| ATH @ MIN | 2026-07-25T00:11:00+00:00 | t_minus_60 | 47 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
