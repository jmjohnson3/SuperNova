# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-24T04:05:37Z
Slate: 2026-07-23
Evaluation: **FINAL**
Games final: 5 / 5
Strict clean slate: **False**

- Locked executable offers: 751
- Valid exact closes: 647 (86.2%)
- Additional valid closes needed for 90%: 29
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 5 / 5

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 29 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 0 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 90 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 8 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| fallback_other_book_only | 3 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| player_market_unavailable_at_close | 3 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 647 |
| player_prop_unavailable_at_close | 90 |
| line_disappeared_at_close | 8 |
| fallback_other_book_only | 3 |
| player_market_unavailable_at_close | 3 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 36ba7a8a8c46e8cc308c1dd037995889 | fanduel | batter_total_bases | 52 | 12 | 76.9% | {"player_prop_unavailable_at_close": 12, "valid_close": 40} |
| 38d3052ec454022a14314b324e0ff5f3 | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 38d3052ec454022a14314b324e0ff5f3 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| 36ba7a8a8c46e8cc308c1dd037995889 | draftkings | batter_hits | 26 | 6 | 76.9% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 36ba7a8a8c46e8cc308c1dd037995889 | fanduel | batter_hits | 26 | 6 | 76.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 20} |
| 36ba7a8a8c46e8cc308c1dd037995889 | fanduel | batter_home_runs | 26 | 6 | 76.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 20} |
| 22fc220be6958e93fba4354054d8fd16 | fanduel | pitcher_strikeouts | 6 | 4 | 33.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 2} |
| 22fc220be6958e93fba4354054d8fd16 | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"player_prop_unavailable_at_close": 4, "valid_close": 4} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | draftkings | batter_total_bases | 10 | 4 | 60.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 6} |
| 38d3052ec454022a14314b324e0ff5f3 | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | fanduel | batter_hits | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | fanduel | batter_home_runs | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 22fc220be6958e93fba4354054d8fd16 | draftkings | batter_hits | 36 | 4 | 88.9% | {"player_prop_unavailable_at_close": 4, "valid_close": 32} |
| 22fc220be6958e93fba4354054d8fd16 | fanduel | batter_total_bases | 56 | 4 | 92.9% | {"player_prop_unavailable_at_close": 4, "valid_close": 52} |
| 38d3052ec454022a14314b324e0ff5f3 | fanduel | batter_hits | 59 | 4 | 93.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 55} |
| 36ba7a8a8c46e8cc308c1dd037995889 | draftkings | pitcher_strikeouts | 2 | 2 | 0.0% | {"player_prop_unavailable_at_close": 2} |
| 36ba7a8a8c46e8cc308c1dd037995889 | fanduel | pitcher_strikeouts | 2 | 2 | 0.0% | {"player_prop_unavailable_at_close": 2} |
| 38d3052ec454022a14314b324e0ff5f3 | draftkings | batter_total_bases | 12 | 2 | 83.3% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "valid_close": 10} |
| 36ba7a8a8c46e8cc308c1dd037995889 | draftkings | batter_total_bases | 18 | 2 | 88.9% | {"player_prop_unavailable_at_close": 2, "valid_close": 16} |
| 22fc220be6958e93fba4354054d8fd16 | fanduel | batter_home_runs | 28 | 2 | 92.9% | {"player_prop_unavailable_at_close": 2, "valid_close": 26} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | draftkings | batter_hits | 32 | 2 | 93.8% | {"player_prop_unavailable_at_close": 2, "valid_close": 30} |
| 22fc220be6958e93fba4354054d8fd16 | fanduel | batter_hits | 54 | 2 | 96.3% | {"player_prop_unavailable_at_close": 2, "valid_close": 52} |
| 22fc220be6958e93fba4354054d8fd16 | draftkings | batter_total_bases | 16 | 0 | 100.0% | {"valid_close": 16} |
| 38d3052ec454022a14314b324e0ff5f3 | draftkings | pitcher_strikeouts | 4 | 0 | 100.0% | {"valid_close": 4} |
| 38d3052ec454022a14314b324e0ff5f3 | fanduel | pitcher_strikeouts | 4 | 0 | 100.0% | {"valid_close": 4} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | draftkings | pitcher_strikeouts | 2 | 0 | 100.0% | {"valid_close": 2} |
| 5c7e77b42c0db49ce882d3c9473d74c1 | fanduel | pitcher_strikeouts | 2 | 0 | 100.0% | {"valid_close": 2} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| SD @ ATL | 2026-07-23T16:16:00+00:00 | yes | yes | yes | 36 |
| MIN @ CLE | 2026-07-23T17:11:00+00:00 | yes | yes | yes | 48 |
| TB @ TOR | 2026-07-23T19:08:00+00:00 | yes | yes | yes | 52 |
| ARI @ STL | 2026-07-23T21:16:00+00:00 | yes | yes | yes | 67 |
| KC @ DET | 2026-07-23T22:41:00+00:00 | yes | yes | yes | 68 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 0 |
| t_minus_60 | 0 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
