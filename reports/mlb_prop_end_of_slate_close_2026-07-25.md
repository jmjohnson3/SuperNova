# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-26T04:12:03Z
Slate: 2026-07-25
Evaluation: **FINAL**
Games final: 15 / 15
Strict clean slate: **False**

- Locked executable offers: 2887
- Valid exact closes: 2497 (86.5%)
- Additional valid closes needed for 90%: 102
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 4 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 102 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 11 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 320 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 44 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 13 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 13 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2497 |
| player_prop_unavailable_at_close | 320 |
| line_disappeared_at_close | 44 |
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
| 18452193d8715008eccc60bd8b4204aa | fanduel | batter_total_bases | 68 | 16 | 76.5% | {"player_prop_unavailable_at_close": 16, "valid_close": 52} |
| 6b06a4bf25ee92a7ac908981664ca759 | fanduel | batter_total_bases | 72 | 16 | 77.8% | {"player_prop_unavailable_at_close": 16, "valid_close": 56} |
| 18452193d8715008eccc60bd8b4204aa | draftkings | batter_hits | 38 | 12 | 68.4% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 10, "valid_close": 26} |
| 87b76cce3c853cd42c5b4437b091cdc6 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| 6713999ff416bdeecd85c7e84327602b | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| f1e238320de4aea5750434c876076d20 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 45d79b3c7b013180e77cfe3675a85eb6 | draftkings | batter_hits | 38 | 10 | 73.7% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 6b06a4bf25ee92a7ac908981664ca759 | fanduel | batter_hits | 63 | 9 | 85.7% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 8, "valid_close": 54} |
| f1e238320de4aea5750434c876076d20 | draftkings | batter_total_bases | 20 | 8 | 60.0% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "player_prop_unavailable_at_close": 2, "valid_close": 12} |
| 18452193d8715008eccc60bd8b4204aa | fanduel | batter_home_runs | 34 | 8 | 76.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 26} |
| 6b06a4bf25ee92a7ac908981664ca759 | draftkings | batter_hits | 36 | 8 | 77.8% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 6b06a4bf25ee92a7ac908981664ca759 | fanduel | batter_home_runs | 36 | 8 | 77.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 28} |
| 4d0a3e98e89c319342e1597db85fdb41 | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| e7cd26360ce2edd30f37cdfc01391fd5 | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| 18452193d8715008eccc60bd8b4204aa | fanduel | batter_hits | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| 45d79b3c7b013180e77cfe3675a85eb6 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 4d0a3e98e89c319342e1597db85fdb41 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| 6b50126f3d5d20aaeb5339572c025bf3 | fanduel | batter_total_bases | 72 | 8 | 88.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 64} |
| 45d79b3c7b013180e77cfe3675a85eb6 | draftkings | batter_total_bases | 22 | 6 | 72.7% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "valid_close": 16} |
| 18452193d8715008eccc60bd8b4204aa | draftkings | batter_total_bases | 26 | 6 | 76.9% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 87b76cce3c853cd42c5b4437b091cdc6 | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 87b76cce3c853cd42c5b4437b091cdc6 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 6713999ff416bdeecd85c7e84327602b | draftkings | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 6713999ff416bdeecd85c7e84327602b | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 6713999ff416bdeecd85c7e84327602b | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| f1e238320de4aea5750434c876076d20 | draftkings | batter_hits | 34 | 6 | 82.4% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 28} |
| f1e238320de4aea5750434c876076d20 | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| f1e238320de4aea5750434c876076d20 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 7d87987ec48dfc47ee32eab874f92200 | draftkings | batter_hits | 38 | 6 | 84.2% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 32} |
| 87b76cce3c853cd42c5b4437b091cdc6 | fanduel | batter_hits | 48 | 6 | 87.5% | {"player_prop_unavailable_at_close": 6, "valid_close": 42} |
| 6b50126f3d5d20aaeb5339572c025bf3 | fanduel | batter_home_runs | 36 | 5 | 86.1% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 31} |
| 87b76cce3c853cd42c5b4437b091cdc6 | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| d65c687b0748c456de9fbecd90808fe1 | draftkings | batter_total_bases | 8 | 4 | 50.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 4} |
| 6b06a4bf25ee92a7ac908981664ca759 | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 6} |
| e7cd26360ce2edd30f37cdfc01391fd5 | draftkings | batter_hits | 28 | 4 | 85.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| e7cd26360ce2edd30f37cdfc01391fd5 | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| e7cd26360ce2edd30f37cdfc01391fd5 | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 159efeea2dcd885c7cbe6edfea2f2603 | draftkings | batter_hits | 32 | 4 | 87.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| 45d79b3c7b013180e77cfe3675a85eb6 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| cf3dedae37cf6426bfdf0b99237f1ba0 | draftkings | batter_hits | 32 | 4 | 87.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 28} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| KC @ DET | 2026-07-25T17:11:00+00:00 | yes | yes | yes | 34 |
| SD @ MIA | 2026-07-25T20:11:00+00:00 | yes | yes | yes | 49 |
| TOR @ BOS | 2026-07-25T20:11:00+00:00 | yes | yes | yes | 50 |
| CHC @ PIT | 2026-07-25T22:41:00+00:00 | yes | yes | yes | 52 |
| ARI @ WAS | 2026-07-25T20:06:00+00:00 | yes | no | yes | 50 |
| LAA @ SF | 2026-07-25T20:06:00+00:00 | yes | no | yes | 50 |
| NYY @ PHI | 2026-07-25T22:06:00+00:00 | yes | no | yes | 52 |
| CLE @ TB | 2026-07-25T22:11:00+00:00 | yes | no | yes | 52 |
| ATL @ BAL | 2026-07-25T23:06:00+00:00 | no | yes | yes | 53 |
| HOU @ CWS | 2026-07-25T23:11:00+00:00 | no | yes | yes | 53 |
| ATH @ MIN | 2026-07-25T23:11:00+00:00 | no | yes | yes | 52 |
| COL @ MIL | 2026-07-25T23:11:00+00:00 | no | yes | yes | 53 |
| SEA @ TEX | 2026-07-25T23:16:00+00:00 | no | yes | yes | 53 |
| LAD @ NYM | 2026-07-25T23:16:00+00:00 | no | yes | yes | 53 |
| CIN @ STL | 2026-07-25T23:16:00+00:00 | no | yes | yes | 53 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 7 |
| t_minus_60 | 4 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| ARI @ WAS | 2026-07-25T20:06:00+00:00 | t_minus_60 | 50 |
| LAA @ SF | 2026-07-25T20:06:00+00:00 | t_minus_60 | 50 |
| NYY @ PHI | 2026-07-25T22:06:00+00:00 | t_minus_60 | 52 |
| CLE @ TB | 2026-07-25T22:11:00+00:00 | t_minus_60 | 52 |
| ATL @ BAL | 2026-07-25T23:06:00+00:00 | t_minus_120 | 53 |
| HOU @ CWS | 2026-07-25T23:11:00+00:00 | t_minus_120 | 53 |
| ATH @ MIN | 2026-07-25T23:11:00+00:00 | t_minus_120 | 52 |
| COL @ MIL | 2026-07-25T23:11:00+00:00 | t_minus_120 | 53 |
| SEA @ TEX | 2026-07-25T23:16:00+00:00 | t_minus_120 | 53 |
| LAD @ NYM | 2026-07-25T23:16:00+00:00 | t_minus_120 | 53 |
| CIN @ STL | 2026-07-25T23:16:00+00:00 | t_minus_120 | 53 |
