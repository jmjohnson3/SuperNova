# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-12T04:01:11Z
Slate: 2026-07-11
Evaluation: **PROVISIONAL**
Games final: 14 / 16
Strict clean slate: **False**

- Locked executable offers: 2874
- Valid exact closes: 2563 (89.2%)
- Additional valid closes needed for 90%: 24
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 16 / 16

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 24 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 0 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 236 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 49 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 13 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 13 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2563 |
| player_prop_unavailable_at_close | 236 |
| line_disappeared_at_close | 49 |
| player_market_unavailable_at_close | 13 |
| fallback_other_book_only | 13 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| a2501dff8c71ab28b0051fafeed81903 | draftkings | batter_hits | 38 | 12 | 68.4% | {"line_disappeared_at_close": 8, "player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 8ec769ee362e5e0e25ac93ee886cdfd4 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| 93a5e6119186fda8b8ae5e4ec2abd550 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 27168ae4cc8b352a2cb7e3f6dab8b610 | draftkings | batter_hits | 44 | 10 | 77.3% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 4, "valid_close": 34} |
| 8ec769ee362e5e0e25ac93ee886cdfd4 | draftkings | batter_total_bases | 14 | 8 | 42.9% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "player_prop_unavailable_at_close": 2, "valid_close": 6} |
| 27168ae4cc8b352a2cb7e3f6dab8b610 | draftkings | batter_total_bases | 28 | 8 | 71.4% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "player_prop_unavailable_at_close": 2, "valid_close": 20} |
| 39a35ef6ca846e78caf1d3d4063cba8b | draftkings | batter_hits | 30 | 8 | 73.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 93986de15fc102a4875daaf7e6250f9e | draftkings | batter_hits | 32 | 8 | 75.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 93a5e6119186fda8b8ae5e4ec2abd550 | draftkings | batter_hits | 34 | 8 | 76.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 39a35ef6ca846e78caf1d3d4063cba8b | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| 93986de15fc102a4875daaf7e6250f9e | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| a2501dff8c71ab28b0051fafeed81903 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 27168ae4cc8b352a2cb7e3f6dab8b610 | fanduel | batter_total_bases | 72 | 8 | 88.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 64} |
| ff676cfabd491a806f9fad9a93924adb | draftkings | batter_total_bases | 10 | 6 | 40.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 4} |
| 8ec769ee362e5e0e25ac93ee886cdfd4 | draftkings | batter_hits | 28 | 6 | 78.6% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 8ec769ee362e5e0e25ac93ee886cdfd4 | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 8ec769ee362e5e0e25ac93ee886cdfd4 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 93a5e6119186fda8b8ae5e4ec2abd550 | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 93a5e6119186fda8b8ae5e4ec2abd550 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 93986de15fc102a4875daaf7e6250f9e | draftkings | batter_total_bases | 20 | 4 | 80.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 16} |
| a2501dff8c71ab28b0051fafeed81903 | draftkings | batter_total_bases | 20 | 4 | 80.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 16} |
| ff676cfabd491a806f9fad9a93924adb | draftkings | batter_hits | 26 | 4 | 84.6% | {"player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 39a35ef6ca846e78caf1d3d4063cba8b | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 39a35ef6ca846e78caf1d3d4063cba8b | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 3e2ce9905fcaa6c74169c61fcb777080 | draftkings | batter_hits | 28 | 4 | 85.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| 93986de15fc102a4875daaf7e6250f9e | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 93986de15fc102a4875daaf7e6250f9e | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| def5adac76d2258bb7ff1a427e09f8a8 | draftkings | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| b552ecd25e0e2e615532c65ff294144f | draftkings | batter_hits | 30 | 4 | 86.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 26} |
| a2501dff8c71ab28b0051fafeed81903 | fanduel | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| a2501dff8c71ab28b0051fafeed81903 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 6b6370ec220748ce5deae40b503bb29c | draftkings | batter_hits | 34 | 4 | 88.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 30} |
| 09d58ac97a07ac9cc6bc658cf72f08a3 | draftkings | batter_hits | 36 | 4 | 88.9% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 32} |
| 27168ae4cc8b352a2cb7e3f6dab8b610 | fanduel | batter_home_runs | 36 | 4 | 88.9% | {"player_prop_unavailable_at_close": 4, "valid_close": 32} |
| ba8bc57255560c92ff44bf4ac63a4fd9 | draftkings | batter_hits | 38 | 4 | 89.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 34} |
| ff676cfabd491a806f9fad9a93924adb | fanduel | batter_total_bases | 48 | 4 | 91.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 44} |
| 3e2ce9905fcaa6c74169c61fcb777080 | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| def5adac76d2258bb7ff1a427e09f8a8 | fanduel | batter_total_bases | 56 | 4 | 92.9% | {"player_prop_unavailable_at_close": 4, "valid_close": 52} |
| b552ecd25e0e2e615532c65ff294144f | fanduel | batter_total_bases | 60 | 4 | 93.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 56} |
| 09d58ac97a07ac9cc6bc658cf72f08a3 | fanduel | batter_total_bases | 64 | 4 | 93.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 60} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| MIL @ PIT | 2026-07-11T16:07:00+00:00 | yes | yes | yes | 23 |
| ATH @ CWS | 2026-07-11T18:11:00+00:00 | yes | yes | yes | 31 |
| LAA @ MIN | 2026-07-11T18:37:00+00:00 | yes | yes | yes | 35 |
| COL @ SF | 2026-07-11T20:06:07+00:00 | yes | yes | yes | 41 |
| NYY @ WAS | 2026-07-11T20:10:00+00:00 | yes | yes | yes | 44 |
| CLE @ MIA | 2026-07-11T20:11:00+00:00 | yes | yes | yes | 44 |
| BOS @ NYM | 2026-07-11T20:11:00+00:00 | yes | yes | yes | 44 |
| MIL @ PIT | 2026-07-11T20:16:00+00:00 | yes | yes | yes | 44 |
| SEA @ TB | 2026-07-11T20:35:00+00:00 | yes | yes | yes | 45 |
| PHI @ DET | 2026-07-11T22:11:00+00:00 | yes | yes | yes | 54 |
| KC @ BAL | 2026-07-11T23:06:00+00:00 | yes | yes | yes | 55 |
| HOU @ TEX | 2026-07-11T23:06:00+00:00 | yes | yes | yes | 56 |
| CHC @ CIN | 2026-07-11T23:11:00+00:00 | yes | yes | yes | 56 |
| ATL @ STL | 2026-07-11T23:20:00+00:00 | yes | yes | yes | 56 |
| TOR @ SD | 2026-07-12T00:41:00+00:00 | yes | yes | yes | 58 |
| ARI @ LAD | 2026-07-12T01:10:54+00:00 | yes | yes | yes | 58 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 0 |
| t_minus_60 | 0 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|

> This slate is still provisional. Coverage must be evaluated again after every game is final.
