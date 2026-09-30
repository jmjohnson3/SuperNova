# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-20T04:02:54Z
Slate: 2026-07-19
Evaluation: **FINAL**
Games final: 16 / 16
Strict clean slate: **False**

- Locked executable offers: 2705
- Valid exact closes: 2332 (86.2%)
- Additional valid closes needed for 90%: 103
- Stale closes: 0.1%
- T-120/T-60/T-20 complete events: 14 / 16

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 103 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 2 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 310 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 43 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 9 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 9 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| stale_close_before_lock | 2 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2332 |
| player_prop_unavailable_at_close | 310 |
| line_disappeared_at_close | 43 |
| player_market_unavailable_at_close | 9 |
| fallback_other_book_only | 9 |
| stale_close_before_lock | 2 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 7bdca69ce161b0f7fa1d383357c64989 | fanduel | batter_total_bases | 52 | 16 | 69.2% | {"player_prop_unavailable_at_close": 16, "valid_close": 36} |
| ffa609f7c73ad0d8396150b3cc744332 | fanduel | batter_total_bases | 52 | 16 | 69.2% | {"player_prop_unavailable_at_close": 16, "valid_close": 36} |
| d1c0cc729b8385c5637b14364ae82ab3 | fanduel | batter_total_bases | 60 | 16 | 73.3% | {"player_prop_unavailable_at_close": 16, "valid_close": 44} |
| d1c0cc729b8385c5637b14364ae82ab3 | draftkings | batter_hits | 34 | 14 | 58.8% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 8, "valid_close": 20} |
| 7bdca69ce161b0f7fa1d383357c64989 | draftkings | batter_hits | 30 | 12 | 60.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 10, "valid_close": 18} |
| f3cb58dab3e8832223686a2ee9639a0d | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| 4a351bac7c9d35b346c76736bd88b1de | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 4a351bac7c9d35b346c76736bd88b1de | draftkings | batter_hits | 38 | 10 | 73.7% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 7bdca69ce161b0f7fa1d383357c64989 | draftkings | batter_total_bases | 16 | 8 | 50.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 8} |
| 7bdca69ce161b0f7fa1d383357c64989 | fanduel | batter_hits | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| 7bdca69ce161b0f7fa1d383357c64989 | fanduel | batter_home_runs | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| ffa609f7c73ad0d8396150b3cc744332 | draftkings | batter_hits | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| ffa609f7c73ad0d8396150b3cc744332 | fanduel | batter_hits | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| ffa609f7c73ad0d8396150b3cc744332 | fanduel | batter_home_runs | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| d1c0cc729b8385c5637b14364ae82ab3 | fanduel | batter_hits | 30 | 8 | 73.3% | {"player_prop_unavailable_at_close": 8, "valid_close": 22} |
| d1c0cc729b8385c5637b14364ae82ab3 | fanduel | batter_home_runs | 30 | 8 | 73.3% | {"player_prop_unavailable_at_close": 8, "valid_close": 22} |
| 60af7c95541bfe00c28e4a14897c6ae4 | draftkings | batter_hits | 38 | 8 | 78.9% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 2, "valid_close": 30} |
| 203e82b1ab8211c90f2f74c5759699ee | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| 9bc4c78b91c95337830ebb3654858de7 | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| e2f5ce09c19effb9e8bba90a0c1c3e44 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 9428684a7aeb42f6f3b5c0c9613c66ca | fanduel | batter_total_bases | 72 | 8 | 88.9% | {"player_prop_unavailable_at_close": 8, "valid_close": 64} |
| e2f5ce09c19effb9e8bba90a0c1c3e44 | draftkings | batter_total_bases | 12 | 6 | 50.0% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "valid_close": 6} |
| d1c0cc729b8385c5637b14364ae82ab3 | draftkings | batter_total_bases | 24 | 6 | 75.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 18} |
| f3cb58dab3e8832223686a2ee9639a0d | draftkings | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| f3cb58dab3e8832223686a2ee9639a0d | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| f3cb58dab3e8832223686a2ee9639a0d | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 4a351bac7c9d35b346c76736bd88b1de | fanduel | batter_hits | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 4a351bac7c9d35b346c76736bd88b1de | fanduel | batter_home_runs | 34 | 6 | 82.4% | {"player_prop_unavailable_at_close": 6, "valid_close": 28} |
| 9bc4c78b91c95337830ebb3654858de7 | fanduel | batter_hits | 52 | 6 | 88.5% | {"player_prop_unavailable_at_close": 4, "stale_close_before_lock": 2, "valid_close": 46} |
| 9428684a7aeb42f6f3b5c0c9613c66ca | fanduel | batter_hits | 65 | 5 | 92.3% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 60} |
| f3cb58dab3e8832223686a2ee9639a0d | draftkings | pitcher_strikeouts | 6 | 4 | 33.3% | {"line_disappeared_at_close": 4, "valid_close": 2} |
| 4a351bac7c9d35b346c76736bd88b1de | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| 203e82b1ab8211c90f2f74c5759699ee | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 203e82b1ab8211c90f2f74c5759699ee | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 9bc4c78b91c95337830ebb3654858de7 | draftkings | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 9bc4c78b91c95337830ebb3654858de7 | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| e7ba65ff4e90d9960a0619d266c10ff5 | draftkings | batter_hits | 28 | 4 | 85.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| 203e82b1ab8211c90f2f74c5759699ee | draftkings | batter_hits | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |
| e9245e202e09448d0020e3f46508dc9d | draftkings | batter_hits | 30 | 4 | 86.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 26} |
| e2f5ce09c19effb9e8bba90a0c1c3e44 | fanduel | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| CWS @ TOR | 2026-07-19T16:16:00+00:00 | yes | yes | yes | 22 |
| LAD @ NYY | 2026-07-19T16:39:00+00:00 | yes | yes | yes | 25 |
| TB @ BOS | 2026-07-19T17:36:00+00:00 | yes | yes | yes | 26 |
| NYM @ PHI | 2026-07-19T17:36:00+00:00 | yes | yes | yes | 29 |
| TEX @ ATL | 2026-07-19T17:36:00+00:00 | yes | yes | yes | 28 |
| PIT @ CLE | 2026-07-19T17:41:00+00:00 | yes | yes | yes | 27 |
| SD @ KC | 2026-07-19T18:11:00+00:00 | yes | yes | yes | 29 |
| MIA @ MIL | 2026-07-19T18:11:00+00:00 | yes | yes | yes | 29 |
| BAL @ HOU | 2026-07-19T18:11:00+00:00 | yes | yes | yes | 29 |
| MIN @ CHC | 2026-07-19T18:21:00+00:00 | yes | yes | yes | 29 |
| WAS @ ATH | 2026-07-19T20:06:00+00:00 | yes | yes | yes | 38 |
| DET @ LAA | 2026-07-19T20:08:00+00:00 | yes | yes | yes | 38 |
| STL @ ARI | 2026-07-19T20:11:00+00:00 | yes | yes | yes | 40 |
| SF @ SEA | 2026-07-19T20:11:00+00:00 | yes | yes | yes | 38 |
| CIN @ COL | 2026-07-19T19:11:00+00:00 | no | yes | yes | 33 |
| LAD @ NYY | 2026-07-19T23:21:00+00:00 | yes | yes | no | 41 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 1 |
| t_minus_60 | 0 |
| t_minus_20 | 1 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| CIN @ COL | 2026-07-19T19:11:00+00:00 | t_minus_120 | 33 |
| LAD @ NYY | 2026-07-19T23:21:00+00:00 | t_minus_20 | 41 |
