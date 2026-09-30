# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-11T04:08:06Z
Slate: 2026-07-10
Evaluation: **PROVISIONAL**
Games final: 9 / 15
Strict clean slate: **False**

- Locked executable offers: 2946
- Valid exact closes: 2580 (87.6%)
- Additional valid closes needed for 90%: 72
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 14 / 16

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 72 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 2 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 298 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 46 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 13 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 9 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2580 |
| player_prop_unavailable_at_close | 298 |
| line_disappeared_at_close | 46 |
| player_market_unavailable_at_close | 13 |
| fallback_other_book_only | 9 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 2461d70fb5485aced0029aaea219afeb | fanduel | batter_total_bases | 60 | 20 | 66.7% | {"player_prop_unavailable_at_close": 20, "valid_close": 40} |
| 2461d70fb5485aced0029aaea219afeb | draftkings | batter_hits | 34 | 14 | 58.8% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 10, "valid_close": 20} |
| ca5e8cb9be664ff91b121c15379d7ce9 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| c710eb874e3d11ac6f032fbc186f2e5f | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| 976819aa47554213e70bc968d9356fc4 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 895dfbfbbb57ff9ecc3cb430a9c69076 | fanduel | batter_total_bases | 76 | 12 | 84.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 64} |
| 2461d70fb5485aced0029aaea219afeb | fanduel | batter_hits | 30 | 10 | 66.7% | {"player_prop_unavailable_at_close": 10, "valid_close": 20} |
| 2461d70fb5485aced0029aaea219afeb | fanduel | batter_home_runs | 30 | 10 | 66.7% | {"player_prop_unavailable_at_close": 10, "valid_close": 20} |
| ca5e8cb9be664ff91b121c15379d7ce9 | draftkings | batter_hits | 30 | 8 | 73.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 781ce9e5ac0cd1e3339085c1fb7d7b39 | draftkings | batter_hits | 36 | 8 | 77.8% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 2ac65fed3f431f0601c2e788aee18cab | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 2, "valid_close": 30} |
| 895dfbfbbb57ff9ecc3cb430a9c69076 | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 30} |
| f308816130f3007f771929f15d62862c | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 583d3f0f064fd76fdea2bc583f0eb548 | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| 781ce9e5ac0cd1e3339085c1fb7d7b39 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 250b0373676b10f51ed1c59c93714245 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| cc0b13e921789925a1c0502553d4ec9b | draftkings | batter_hits | 20 | 6 | 70.0% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 14} |
| 2461d70fb5485aced0029aaea219afeb | draftkings | batter_total_bases | 22 | 6 | 72.7% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 16} |
| 583d3f0f064fd76fdea2bc583f0eb548 | draftkings | batter_hits | 30 | 6 | 80.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 24} |
| ca5e8cb9be664ff91b121c15379d7ce9 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| ca5e8cb9be664ff91b121c15379d7ce9 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| c710eb874e3d11ac6f032fbc186f2e5f | fanduel | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| c710eb874e3d11ac6f032fbc186f2e5f | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 976819aa47554213e70bc968d9356fc4 | draftkings | batter_hits | 34 | 6 | 82.4% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 976819aa47554213e70bc968d9356fc4 | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 976819aa47554213e70bc968d9356fc4 | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 895dfbfbbb57ff9ecc3cb430a9c69076 | fanduel | batter_home_runs | 38 | 6 | 84.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 32} |
| 895dfbfbbb57ff9ecc3cb430a9c69076 | fanduel | batter_hits | 67 | 6 | 91.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 61} |
| 76ab48541c0690199e815574c469d162 | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 10} |
| c710eb874e3d11ac6f032fbc186f2e5f | draftkings | batter_total_bases | 14 | 4 | 71.4% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 2, "valid_close": 10} |
| 976819aa47554213e70bc968d9356fc4 | draftkings | batter_total_bases | 18 | 4 | 77.8% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 14} |
| c710eb874e3d11ac6f032fbc186f2e5f | draftkings | batter_hits | 26 | 4 | 84.6% | {"player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 583d3f0f064fd76fdea2bc583f0eb548 | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 583d3f0f064fd76fdea2bc583f0eb548 | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 781ce9e5ac0cd1e3339085c1fb7d7b39 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 250b0373676b10f51ed1c59c93714245 | fanduel | batter_home_runs | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 55c7996867831d050e3a904c541089d8 | draftkings | batter_hits | 36 | 4 | 88.9% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 32} |
| cc0b13e921789925a1c0502553d4ec9b | fanduel | batter_total_bases | 40 | 4 | 90.0% | {"player_prop_unavailable_at_close": 4, "valid_close": 36} |
| 76ab48541c0690199e815574c469d162 | fanduel | batter_total_bases | 56 | 4 | 92.9% | {"player_prop_unavailable_at_close": 4, "valid_close": 52} |
| 781ce9e5ac0cd1e3339085c1fb7d7b39 | fanduel | batter_hits | 58 | 4 | 93.1% | {"player_prop_unavailable_at_close": 4, "valid_close": 54} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| PHI @ DET | 2026-07-10T22:41:00+00:00 | yes | yes | yes | 34 |
| KC @ BAL | 2026-07-10T23:08:00+00:00 | yes | yes | yes | 35 |
| CHC @ CIN | 2026-07-10T23:11:00+00:00 | yes | yes | yes | 37 |
| SEA @ TB | 2026-07-10T23:11:00+00:00 | yes | yes | yes | 36 |
| CLE @ MIA | 2026-07-10T23:11:00+00:00 | yes | yes | yes | 35 |
| ATH @ CWS | 2026-07-10T23:41:00+00:00 | yes | yes | yes | 38 |
| HOU @ TEX | 2026-07-11T00:09:00+00:00 | yes | yes | yes | 38 |
| LAA @ MIN | 2026-07-11T00:11:00+00:00 | yes | yes | yes | 38 |
| ATL @ STL | 2026-07-11T00:16:05+00:00 | yes | yes | yes | 35 |
| MIL @ PIT | 2026-07-11T00:45:00+00:00 | yes | yes | yes | 29 |
| NYY @ WAS | 2026-07-11T00:46:00+00:00 | yes | yes | yes | 43 |
| TOR @ SD | 2026-07-11T01:41:00+00:00 | yes | yes | yes | 43 |
| ARI @ LAD | 2026-07-11T02:11:00+00:00 | yes | yes | yes | 43 |
| COL @ SF | 2026-07-11T02:16:00+00:00 | yes | yes | yes | 42 |
| BOS @ NYM | 2026-07-10T23:50:00+00:00 | no | yes | yes | 39 |
| MIL @ PIT | 2026-07-11T04:50:00+00:00 | no | no | no | 1 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 2 |
| t_minus_60 | 1 |
| t_minus_20 | 1 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| BOS @ NYM | 2026-07-10T23:50:00+00:00 | t_minus_120 | 39 |
| MIL @ PIT | 2026-07-11T04:50:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 1 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
