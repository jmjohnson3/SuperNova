# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-10T04:05:16Z
Slate: 2026-07-09
Evaluation: **PROVISIONAL**
Games final: 11 / 13
Strict clean slate: **False**

- Locked executable offers: 2575
- Valid exact closes: 2307 (89.6%)
- Additional valid closes needed for 90%: 11
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 11 / 13

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 11 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 2 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 215 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 28 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| fallback_other_book_only | 13 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| player_market_unavailable_at_close | 12 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2307 |
| player_prop_unavailable_at_close | 215 |
| line_disappeared_at_close | 28 |
| fallback_other_book_only | 13 |
| player_market_unavailable_at_close | 12 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| be3485a7acdabede1a849330fcc226dc | fanduel | batter_total_bases | 72 | 16 | 77.8% | {"player_prop_unavailable_at_close": 16, "valid_close": 56} |
| c2c3eb02058f54f27b00aeb1e59224f9 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| 83cf12bed561626b784721179f647eb2 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| cea3285cc6f53d314e7111cbd078141b | draftkings | batter_total_bases | 26 | 10 | 61.5% | {"fallback_other_book_only": 4, "player_market_unavailable_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 16} |
| be3485a7acdabede1a849330fcc226dc | fanduel | batter_home_runs | 36 | 8 | 77.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 28} |
| 2c1eb8ab560fe810d23f21a2acf7cbe6 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 673121d56b103f8d38738980fbffafef | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| a3b9b93cdd33f60538713d337711afe2 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| be3485a7acdabede1a849330fcc226dc | fanduel | batter_hits | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 4690bcfaacd780fc44c057a6a7a4cc46 | fanduel | batter_total_bases | 72 | 8 | 88.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 64} |
| c2c3eb02058f54f27b00aeb1e59224f9 | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| c2c3eb02058f54f27b00aeb1e59224f9 | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| c2c3eb02058f54f27b00aeb1e59224f9 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 83cf12bed561626b784721179f647eb2 | draftkings | batter_hits | 30 | 6 | 80.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 83cf12bed561626b784721179f647eb2 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 83cf12bed561626b784721179f647eb2 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 673121d56b103f8d38738980fbffafef | draftkings | batter_hits | 32 | 6 | 81.2% | {"fallback_other_book_only": 1, "player_prop_unavailable_at_close": 5, "valid_close": 26} |
| a3b9b93cdd33f60538713d337711afe2 | draftkings | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| c2c3eb02058f54f27b00aeb1e59224f9 | draftkings | batter_total_bases | 8 | 4 | 50.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 4} |
| ff7fcfaa73e919c6a69c375313853a91 | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| 2c1eb8ab560fe810d23f21a2acf7cbe6 | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 10} |
| 4690bcfaacd780fc44c057a6a7a4cc46 | draftkings | batter_total_bases | 18 | 4 | 77.8% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 14} |
| 6bfe03bb9d1e3ada87b6238cd7203d64 | draftkings | batter_total_bases | 22 | 4 | 81.8% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 18} |
| 2c1eb8ab560fe810d23f21a2acf7cbe6 | draftkings | batter_hits | 32 | 4 | 87.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| 2c1eb8ab560fe810d23f21a2acf7cbe6 | fanduel | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 2c1eb8ab560fe810d23f21a2acf7cbe6 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 4690bcfaacd780fc44c057a6a7a4cc46 | draftkings | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 673121d56b103f8d38738980fbffafef | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| a3b9b93cdd33f60538713d337711afe2 | fanduel | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| a3b9b93cdd33f60538713d337711afe2 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 2b3ca3496277938aa74f87f24a2c064c | draftkings | batter_hits | 34 | 4 | 88.2% | {"line_disappeared_at_close": 4, "valid_close": 30} |
| 4690bcfaacd780fc44c057a6a7a4cc46 | fanduel | batter_home_runs | 36 | 4 | 88.9% | {"player_prop_unavailable_at_close": 4, "valid_close": 32} |
| 6bfe03bb9d1e3ada87b6238cd7203d64 | draftkings | batter_hits | 38 | 4 | 89.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 34} |
| cea3285cc6f53d314e7111cbd078141b | draftkings | batter_hits | 38 | 4 | 89.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 34} |
| 673121d56b103f8d38738980fbffafef | fanduel | batter_hits | 57 | 4 | 93.0% | {"player_prop_unavailable_at_close": 4, "valid_close": 53} |
| 4690bcfaacd780fc44c057a6a7a4cc46 | fanduel | batter_hits | 68 | 4 | 94.1% | {"player_prop_unavailable_at_close": 4, "valid_close": 64} |
| 9f1eb41cd2972a08e88069d6c5b4bb6c | fanduel | batter_total_bases | 68 | 4 | 94.1% | {"player_prop_unavailable_at_close": 4, "valid_close": 64} |
| cea3285cc6f53d314e7111cbd078141b | fanduel | batter_total_bases | 68 | 4 | 94.1% | {"player_prop_unavailable_at_close": 4, "valid_close": 64} |
| 6bfe03bb9d1e3ada87b6238cd7203d64 | fanduel | batter_total_bases | 72 | 4 | 94.4% | {"player_prop_unavailable_at_close": 4, "valid_close": 68} |
| 673121d56b103f8d38738980fbffafef | fanduel | pitcher_strikeouts | 4 | 2 | 50.0% | {"player_prop_unavailable_at_close": 2, "valid_close": 2} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| ATL @ PIT | 2026-07-09T16:36:00+00:00 | yes | yes | yes | 13 |
| NYY @ TB | 2026-07-09T17:10:21+00:00 | yes | yes | yes | 14 |
| KC @ NYM | 2026-07-09T17:11:00+00:00 | yes | yes | yes | 14 |
| CHC @ BAL | 2026-07-09T17:36:00+00:00 | yes | yes | yes | 14 |
| CLE @ MIN | 2026-07-09T17:41:00+00:00 | yes | yes | yes | 14 |
| BOS @ CWS | 2026-07-09T18:11:00+00:00 | yes | yes | yes | 16 |
| ATH @ DET | 2026-07-09T22:41:00+00:00 | yes | yes | yes | 36 |
| SEA @ MIA | 2026-07-09T22:41:00+00:00 | yes | yes | yes | 37 |
| MIL @ STL | 2026-07-09T23:46:00+00:00 | yes | yes | yes | 39 |
| LAA @ TEX | 2026-07-10T00:06:00+00:00 | yes | yes | yes | 39 |
| ARI @ SD | 2026-07-10T01:41:00+00:00 | yes | yes | yes | 40 |
| PHI @ CIN | 2026-07-09T23:11:00+00:00 | no | yes | yes | 37 |
| COL @ SF | 2026-07-10T01:46:00+00:00 | no | yes | yes | 39 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 2 |
| t_minus_60 | 0 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| PHI @ CIN | 2026-07-09T23:11:00+00:00 | t_minus_120 | 37 |
| COL @ SF | 2026-07-10T01:46:00+00:00 | t_minus_120 | 39 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
