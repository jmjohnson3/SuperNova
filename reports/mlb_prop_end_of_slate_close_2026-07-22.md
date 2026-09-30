# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-23T04:10:15Z
Slate: 2026-07-22
Evaluation: **FINAL**
Games final: 17 / 17
Strict clean slate: **False**

- Locked executable offers: 3058
- Valid exact closes: 2626 (85.9%)
- Additional valid closes needed for 90%: 127
- Stale closes: 0.1%
- T-120/T-60/T-20 complete events: 5 / 17

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 127 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 12 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 358 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 48 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 12 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 12 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| stale_close_before_lock | 2 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2626 |
| player_prop_unavailable_at_close | 358 |
| line_disappeared_at_close | 48 |
| player_market_unavailable_at_close | 12 |
| fallback_other_book_only | 12 |
| stale_close_before_lock | 2 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 5d1eeebcccadcfc65a1ae65709da7845 | fanduel | batter_total_bases | 60 | 16 | 73.3% | {"player_prop_unavailable_at_close": 16, "valid_close": 44} |
| 10ee5044193bffe8f24adf381ebde5ad | fanduel | batter_total_bases | 64 | 16 | 75.0% | {"player_prop_unavailable_at_close": 16, "valid_close": 48} |
| e3d220f8bbf8ac99e58dbf7088695b59 | fanduel | batter_total_bases | 64 | 16 | 75.0% | {"player_prop_unavailable_at_close": 16, "valid_close": 48} |
| 9121395092ebe73ce8eaba439dc5be34 | fanduel | batter_total_bases | 52 | 12 | 76.9% | {"player_prop_unavailable_at_close": 12, "valid_close": 40} |
| a319064ad34e8e60220140acb9035af3 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| fcc83e954b9e74964b190cb0e29366c7 | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| 5d1eeebcccadcfc65a1ae65709da7845 | draftkings | batter_hits | 30 | 10 | 66.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 8, "valid_close": 20} |
| b6df72e0280eb235fcca22cc3d3fc1ab | draftkings | batter_hits | 32 | 10 | 68.8% | {"line_disappeared_at_close": 8, "player_prop_unavailable_at_close": 2, "valid_close": 22} |
| 10ee5044193bffe8f24adf381ebde5ad | fanduel | batter_hits | 34 | 10 | 70.6% | {"player_prop_unavailable_at_close": 8, "stale_close_before_lock": 2, "valid_close": 24} |
| a319064ad34e8e60220140acb9035af3 | draftkings | batter_total_bases | 20 | 8 | 60.0% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "player_prop_unavailable_at_close": 2, "valid_close": 12} |
| 46df00997efe9e30fd9b1130a3774851 | draftkings | batter_hits | 28 | 8 | 71.4% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 5d1eeebcccadcfc65a1ae65709da7845 | fanduel | batter_hits | 30 | 8 | 73.3% | {"player_prop_unavailable_at_close": 8, "valid_close": 22} |
| 5d1eeebcccadcfc65a1ae65709da7845 | fanduel | batter_home_runs | 30 | 8 | 73.3% | {"player_prop_unavailable_at_close": 8, "valid_close": 22} |
| 10ee5044193bffe8f24adf381ebde5ad | fanduel | batter_home_runs | 32 | 8 | 75.0% | {"player_prop_unavailable_at_close": 8, "valid_close": 24} |
| e3d220f8bbf8ac99e58dbf7088695b59 | fanduel | batter_hits | 32 | 8 | 75.0% | {"player_prop_unavailable_at_close": 8, "valid_close": 24} |
| e3d220f8bbf8ac99e58dbf7088695b59 | fanduel | batter_home_runs | 32 | 8 | 75.0% | {"player_prop_unavailable_at_close": 8, "valid_close": 24} |
| a319064ad34e8e60220140acb9035af3 | draftkings | batter_hits | 34 | 8 | 76.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 46df00997efe9e30fd9b1130a3774851 | fanduel | batter_total_bases | 48 | 8 | 83.3% | {"player_prop_unavailable_at_close": 8, "valid_close": 40} |
| 573a8d20a4ce1b6795f2e3ac8d30cae0 | fanduel | batter_total_bases | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| 5a272ddcdf5d8b9d87abacf3a492c021 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 111a955795876d50988b15c219ce0796 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| be25eb82b82629d959c1e5ccb8dcc1e7 | fanduel | batter_total_bases | 72 | 8 | 88.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 64} |
| a362299f25a2814e5b0e40d072bb782f | draftkings | batter_total_bases | 10 | 6 | 40.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 4} |
| 9121395092ebe73ce8eaba439dc5be34 | fanduel | batter_hits | 26 | 6 | 76.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 20} |
| 9121395092ebe73ce8eaba439dc5be34 | fanduel | batter_home_runs | 26 | 6 | 76.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 20} |
| 10ee5044193bffe8f24adf381ebde5ad | draftkings | batter_hits | 28 | 6 | 78.6% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 22} |
| e3d220f8bbf8ac99e58dbf7088695b59 | draftkings | batter_hits | 28 | 6 | 78.6% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 22} |
| a319064ad34e8e60220140acb9035af3 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| a319064ad34e8e60220140acb9035af3 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 573a8d20a4ce1b6795f2e3ac8d30cae0 | draftkings | batter_hits | 32 | 6 | 81.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 26} |
| fcc83e954b9e74964b190cb0e29366c7 | fanduel | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| fcc83e954b9e74964b190cb0e29366c7 | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 111a955795876d50988b15c219ce0796 | draftkings | batter_hits | 38 | 6 | 84.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 32} |
| fcc83e954b9e74964b190cb0e29366c7 | draftkings | batter_hits | 40 | 6 | 85.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 34} |
| 573a8d20a4ce1b6795f2e3ac8d30cae0 | fanduel | batter_hits | 56 | 5 | 91.1% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 51} |
| 5a272ddcdf5d8b9d87abacf3a492c021 | fanduel | batter_hits | 59 | 5 | 91.5% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 54} |
| fe2c8918c10c47fffd27be962809d726 | draftkings | batter_total_bases | 20 | 4 | 80.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 16} |
| 46df00997efe9e30fd9b1130a3774851 | fanduel | batter_hits | 24 | 4 | 83.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 46df00997efe9e30fd9b1130a3774851 | fanduel | batter_home_runs | 24 | 4 | 83.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 20} |
| 573a8d20a4ce1b6795f2e3ac8d30cae0 | draftkings | batter_total_bases | 24 | 4 | 83.3% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "valid_close": 20} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| BAL @ BOS | 2026-07-22T17:36:00+00:00 | yes | yes | yes | 30 |
| SF @ KC | 2026-07-22T18:11:00+00:00 | yes | yes | yes | 32 |
| NYM @ MIL | 2026-07-22T18:11:00+00:00 | yes | yes | yes | 33 |
| WAS @ COL | 2026-07-22T19:11:00+00:00 | yes | yes | yes | 40 |
| MIN @ CLE | 2026-07-22T22:41:00+00:00 | yes | yes | yes | 53 |
| PIT @ NYY | 2026-07-22T17:06:00+00:00 | no | yes | yes | 31 |
| CIN @ SEA | 2026-07-22T19:41:00+00:00 | yes | yes | no | 40 |
| ATH @ ARI | 2026-07-22T19:43:00+00:00 | yes | yes | no | 44 |
| STL @ LAA | 2026-07-22T20:08:00+00:00 | yes | no | yes | 44 |
| LAD @ PHI | 2026-07-22T22:45:00+00:00 | no | yes | yes | 53 |
| TB @ TOR | 2026-07-22T23:08:00+00:00 | no | yes | yes | 53 |
| PIT @ NYY | 2026-07-22T23:09:10+00:00 | no | yes | yes | 52 |
| BAL @ BOS | 2026-07-22T23:10:48+00:00 | no | yes | yes | 52 |
| SD @ ATL | 2026-07-22T23:16:00+00:00 | no | yes | yes | 52 |
| CWS @ TEX | 2026-07-23T00:06:00+00:00 | yes | no | yes | 54 |
| DET @ CHC | 2026-07-23T00:10:00+00:00 | yes | no | yes | 54 |
| MIA @ HOU | 2026-07-23T00:11:00+00:00 | yes | no | yes | 54 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 6 |
| t_minus_60 | 4 |
| t_minus_20 | 2 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| PIT @ NYY | 2026-07-22T17:06:00+00:00 | t_minus_120 | 31 |
| CIN @ SEA | 2026-07-22T19:41:00+00:00 | t_minus_20 | 40 |
| ATH @ ARI | 2026-07-22T19:43:00+00:00 | t_minus_20 | 44 |
| STL @ LAA | 2026-07-22T20:08:00+00:00 | t_minus_60 | 44 |
| LAD @ PHI | 2026-07-22T22:45:00+00:00 | t_minus_120 | 53 |
| TB @ TOR | 2026-07-22T23:08:00+00:00 | t_minus_120 | 53 |
| PIT @ NYY | 2026-07-22T23:09:10+00:00 | t_minus_120 | 52 |
| BAL @ BOS | 2026-07-22T23:10:48+00:00 | t_minus_120 | 52 |
| SD @ ATL | 2026-07-22T23:16:00+00:00 | t_minus_120 | 52 |
| CWS @ TEX | 2026-07-23T00:06:00+00:00 | t_minus_60 | 54 |
| DET @ CHC | 2026-07-23T00:10:00+00:00 | t_minus_60 | 54 |
| MIA @ HOU | 2026-07-23T00:11:00+00:00 | t_minus_60 | 54 |
