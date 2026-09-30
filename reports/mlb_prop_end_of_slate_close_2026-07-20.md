# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-21T04:14:03Z
Slate: 2026-07-20
Evaluation: **PROVISIONAL**
Games final: 11 / 15
Strict clean slate: **False**

- Locked executable offers: 2800
- Valid exact closes: 2485 (88.8%)
- Additional valid closes needed for 90%: 35
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 15 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 35 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 0 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 252 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 45 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 9 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 9 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2485 |
| player_prop_unavailable_at_close | 252 |
| line_disappeared_at_close | 45 |
| player_market_unavailable_at_close | 9 |
| fallback_other_book_only | 9 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 66d5a2a0202c6946683c1a8247be9ed4 | draftkings | batter_hits | 36 | 12 | 66.7% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 417e9c61c9e75d104917a558c350fd33 | fanduel | batter_total_bases | 48 | 12 | 75.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 36} |
| 66d5a2a0202c6946683c1a8247be9ed4 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| ed99d6d31957cf5543b0733a49085559 | fanduel | batter_total_bases | 72 | 12 | 83.3% | {"player_prop_unavailable_at_close": 12, "valid_close": 60} |
| 85be05f7a6300ba953ba0dc345908c33 | draftkings | batter_hits | 26 | 8 | 69.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 18} |
| 85be05f7a6300ba953ba0dc345908c33 | fanduel | batter_total_bases | 44 | 8 | 81.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 36} |
| b6936af60f2ffed1b270c119f7127ae7 | fanduel | batter_total_bases | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| 128e4328bfe1ffa187a85b1e5e2b5ec6 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| c9a69923caed0afdda9fd6b94171b189 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| 417e9c61c9e75d104917a558c350fd33 | fanduel | batter_hits | 24 | 6 | 75.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 18} |
| 417e9c61c9e75d104917a558c350fd33 | fanduel | batter_home_runs | 24 | 6 | 75.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 18} |
| 66d5a2a0202c6946683c1a8247be9ed4 | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 66d5a2a0202c6946683c1a8247be9ed4 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 17257e28ab0601512288eeccc637c3d1 | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 30} |
| c9a69923caed0afdda9fd6b94171b189 | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| ed99d6d31957cf5543b0733a49085559 | draftkings | batter_hits | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| ed99d6d31957cf5543b0733a49085559 | fanduel | batter_home_runs | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| 97617f61842d6317b6167cdf64f137a1 | draftkings | batter_hits | 38 | 6 | 84.2% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 32} |
| ed99d6d31957cf5543b0733a49085559 | fanduel | batter_hits | 63 | 6 | 90.5% | {"player_prop_unavailable_at_close": 6, "valid_close": 57} |
| 0dd84d01f2970e524c07c9ff2a747b7f | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| 97617f61842d6317b6167cdf64f137a1 | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 6} |
| 417e9c61c9e75d104917a558c350fd33 | draftkings | batter_total_bases | 12 | 4 | 66.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 8} |
| 29d4af8b2386afc600f412243c22a53b | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 10} |
| ed99d6d31957cf5543b0733a49085559 | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 10} |
| 417e9c61c9e75d104917a558c350fd33 | draftkings | batter_hits | 22 | 4 | 81.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 18} |
| 85be05f7a6300ba953ba0dc345908c33 | fanduel | batter_hits | 22 | 4 | 81.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 18} |
| 85be05f7a6300ba953ba0dc345908c33 | fanduel | batter_home_runs | 22 | 4 | 81.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 18} |
| 29d4af8b2386afc600f412243c22a53b | draftkings | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 17257e28ab0601512288eeccc637c3d1 | draftkings | batter_total_bases | 30 | 4 | 86.7% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 26} |
| 2d7808af8bc666b615b13864d5730354 | draftkings | batter_hits | 30 | 4 | 86.7% | {"line_disappeared_at_close": 4, "valid_close": 26} |
| b6936af60f2ffed1b270c119f7127ae7 | draftkings | batter_hits | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |
| b6936af60f2ffed1b270c119f7127ae7 | fanduel | batter_home_runs | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |
| f237d860c80df8632a4717d53fee1c25 | draftkings | batter_hits | 30 | 4 | 86.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 26} |
| 128e4328bfe1ffa187a85b1e5e2b5ec6 | draftkings | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 128e4328bfe1ffa187a85b1e5e2b5ec6 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| b829b2871533d73401916f919c2b485a | draftkings | batter_hits | 32 | 4 | 87.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| c9a69923caed0afdda9fd6b94171b189 | fanduel | batter_home_runs | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 29d4af8b2386afc600f412243c22a53b | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| 2d7808af8bc666b615b13864d5730354 | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| 7a53b98354a4d009429693371637853d | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| MIN @ CLE | 2026-07-20T22:41:00+00:00 | yes | yes | yes | 43 |
| PIT @ NYY | 2026-07-20T23:06:00+00:00 | yes | yes | yes | 47 |
| TB @ TOR | 2026-07-20T23:08:00+00:00 | yes | yes | yes | 47 |
| BAL @ BOS | 2026-07-20T23:11:00+00:00 | yes | yes | yes | 47 |
| LAD @ PHI | 2026-07-20T23:11:00+00:00 | yes | yes | yes | 47 |
| SD @ ATL | 2026-07-20T23:16:00+00:00 | yes | yes | yes | 47 |
| NYM @ MIL | 2026-07-20T23:41:00+00:00 | yes | yes | yes | 47 |
| SF @ KC | 2026-07-20T23:41:00+00:00 | yes | yes | yes | 47 |
| CWS @ TEX | 2026-07-21T00:08:00+00:00 | yes | yes | yes | 48 |
| MIA @ HOU | 2026-07-21T00:11:00+00:00 | yes | yes | yes | 48 |
| WAS @ COL | 2026-07-21T00:41:00+00:00 | yes | yes | yes | 48 |
| DET @ CHC | 2026-07-21T00:41:00+00:00 | yes | yes | yes | 49 |
| ATH @ ARI | 2026-07-21T01:41:00+00:00 | yes | yes | yes | 49 |
| CIN @ SEA | 2026-07-21T01:42:00+00:00 | yes | yes | yes | 49 |
| STL @ LAA | 2026-07-21T02:11:00+00:00 | yes | yes | yes | 49 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 0 |
| t_minus_60 | 0 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|

> This slate is still provisional. Coverage must be evaluated again after every game is final.
