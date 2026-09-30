# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-09T20:15:35Z
Slate: 2026-07-08
Evaluation: **FINAL**
Games final: 15 / 15
Strict clean slate: **False**

- Locked executable offers: 2957
- Valid exact closes: 2565 (86.7%)
- Additional valid closes needed for 90%: 97
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 9 / 16

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 97 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 7 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 330 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 42 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 10 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 10 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2565 |
| player_prop_unavailable_at_close | 330 |
| line_disappeared_at_close | 42 |
| player_market_unavailable_at_close | 10 |
| fallback_other_book_only | 10 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| e885d80c0e3339f69c0a7c1b898c173e | fanduel | batter_total_bases | 68 | 16 | 76.5% | {"player_prop_unavailable_at_close": 16, "valid_close": 52} |
| 79b379f1dea98a6b1596852f00534a6c | fanduel | batter_total_bases | 76 | 16 | 78.9% | {"player_prop_unavailable_at_close": 16, "valid_close": 60} |
| e885d80c0e3339f69c0a7c1b898c173e | draftkings | batter_hits | 40 | 14 | 65.0% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 10, "valid_close": 26} |
| 49de1a8b7d85007610fb077c920e206a | draftkings | batter_hits | 36 | 12 | 66.7% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 8, "valid_close": 24} |
| 49de1a8b7d85007610fb077c920e206a | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| 8c0695707371420ef6b1cf89c2d1b262 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| f30eb3060b32a01dced2737d2398b918 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 1b110742f40a58c2e76a9743faf4a07c | fanduel | batter_total_bases | 72 | 12 | 83.3% | {"player_prop_unavailable_at_close": 12, "valid_close": 60} |
| 57172eebee377baa85502b3c14ce8249 | draftkings | batter_hits | 24 | 10 | 58.3% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 4, "valid_close": 14} |
| 8c0695707371420ef6b1cf89c2d1b262 | draftkings | batter_hits | 34 | 10 | 70.6% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 8, "valid_close": 24} |
| 79b379f1dea98a6b1596852f00534a6c | draftkings | batter_hits | 42 | 10 | 76.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 8, "valid_close": 32} |
| e885d80c0e3339f69c0a7c1b898c173e | draftkings | batter_total_bases | 22 | 8 | 63.6% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 6, "valid_close": 14} |
| e885d80c0e3339f69c0a7c1b898c173e | fanduel | batter_hits | 34 | 8 | 76.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 26} |
| e885d80c0e3339f69c0a7c1b898c173e | fanduel | batter_home_runs | 34 | 8 | 76.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 26} |
| f30eb3060b32a01dced2737d2398b918 | draftkings | batter_hits | 36 | 8 | 77.8% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 1b110742f40a58c2e76a9743faf4a07c | draftkings | batter_hits | 38 | 8 | 78.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 30} |
| 79b379f1dea98a6b1596852f00534a6c | fanduel | batter_home_runs | 38 | 8 | 78.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 30} |
| 57172eebee377baa85502b3c14ce8249 | fanduel | batter_total_bases | 44 | 8 | 81.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 36} |
| 673394fe5878e43a0bdc2feb5f6634d3 | fanduel | batter_total_bases | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| 8d7cadd457d4e05f27a0ef6feab14d62 | fanduel | batter_total_bases | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| 79b379f1dea98a6b1596852f00534a6c | fanduel | batter_hits | 63 | 8 | 87.3% | {"player_prop_unavailable_at_close": 8, "valid_close": 55} |
| 49de1a8b7d85007610fb077c920e206a | draftkings | batter_total_bases | 16 | 6 | 62.5% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 10} |
| 49de1a8b7d85007610fb077c920e206a | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 49de1a8b7d85007610fb077c920e206a | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 8c0695707371420ef6b1cf89c2d1b262 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 8c0695707371420ef6b1cf89c2d1b262 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| f30eb3060b32a01dced2737d2398b918 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| fa2f6fc2aee410abafe5d685acfde5f3 | draftkings | batter_hits | 34 | 6 | 82.4% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| 1b110742f40a58c2e76a9743faf4a07c | fanduel | batter_home_runs | 36 | 6 | 83.3% | {"player_prop_unavailable_at_close": 6, "valid_close": 30} |
| f30eb3060b32a01dced2737d2398b918 | fanduel | batter_hits | 61 | 6 | 90.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 55} |
| 1b110742f40a58c2e76a9743faf4a07c | fanduel | batter_hits | 66 | 6 | 90.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 60} |
| 88cb3f823f958bc6b24e7f918541e63f | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 6} |
| 8c0695707371420ef6b1cf89c2d1b262 | draftkings | batter_total_bases | 12 | 4 | 66.7% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 8} |
| 79b379f1dea98a6b1596852f00534a6c | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 10} |
| f69a7dce841f22ee977afebfd8dfc142 | draftkings | batter_total_bases | 18 | 4 | 77.8% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 14} |
| 1b110742f40a58c2e76a9743faf4a07c | draftkings | batter_total_bases | 20 | 4 | 80.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 16} |
| 57172eebee377baa85502b3c14ce8249 | fanduel | batter_hits | 22 | 4 | 81.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 18} |
| 57172eebee377baa85502b3c14ce8249 | fanduel | batter_home_runs | 22 | 4 | 81.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 18} |
| 673394fe5878e43a0bdc2feb5f6634d3 | draftkings | batter_hits | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 673394fe5878e43a0bdc2feb5f6634d3 | fanduel | batter_home_runs | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| CHC @ BAL | 2026-07-08T22:36:00+00:00 | yes | yes | yes | 28 |
| HOU @ WAS | 2026-07-08T22:46:00+00:00 | yes | yes | yes | 29 |
| PHI @ CIN | 2026-07-08T23:11:00+00:00 | yes | yes | yes | 28 |
| KC @ NYM | 2026-07-08T23:11:00+00:00 | yes | yes | yes | 28 |
| BOS @ CWS | 2026-07-08T23:41:00+00:00 | yes | yes | yes | 29 |
| CLE @ MIN | 2026-07-08T23:41:00+00:00 | yes | yes | yes | 30 |
| LAA @ TEX | 2026-07-09T00:06:00+00:00 | yes | yes | yes | 30 |
| COL @ LAD | 2026-07-09T02:11:00+00:00 | yes | yes | yes | 31 |
| ARI @ SD | 2026-07-09T02:11:00+00:00 | yes | yes | yes | 31 |
| TOR @ SF | 2026-07-08T19:46:00+00:00 | no | yes | yes | 20 |
| NYY @ TB | 2026-07-08T22:41:00+00:00 | no | yes | yes | 27 |
| ATH @ DET | 2026-07-08T22:41:00+00:00 | no | yes | yes | 28 |
| SEA @ MIA | 2026-07-08T22:41:00+00:00 | no | yes | yes | 27 |
| ATL @ PIT | 2026-07-08T22:41:00+00:00 | no | yes | yes | 27 |
| MIL @ STL | 2026-07-08T23:46:00+00:00 | no | yes | yes | 29 |
| CHC @ BAL | 2026-07-09T17:30:00+00:00 | no | no | no | 1 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 7 |
| t_minus_60 | 1 |
| t_minus_20 | 1 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| TOR @ SF | 2026-07-08T19:46:00+00:00 | t_minus_120 | 20 |
| NYY @ TB | 2026-07-08T22:41:00+00:00 | t_minus_120 | 27 |
| ATH @ DET | 2026-07-08T22:41:00+00:00 | t_minus_120 | 28 |
| SEA @ MIA | 2026-07-08T22:41:00+00:00 | t_minus_120 | 27 |
| ATL @ PIT | 2026-07-08T22:41:00+00:00 | t_minus_120 | 27 |
| MIL @ STL | 2026-07-08T23:46:00+00:00 | t_minus_120 | 29 |
| CHC @ BAL | 2026-07-09T17:30:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 1 |
