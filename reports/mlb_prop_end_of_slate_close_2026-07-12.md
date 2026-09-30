# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-13T01:07:06Z
Slate: 2026-07-12
Evaluation: **FINAL**
Games final: 15 / 15
Strict clean slate: **True**

- Locked executable offers: 2601
- Valid exact closes: 2366 (91.0%)
- Additional valid closes needed for 90%: 0
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 15 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 0 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 0 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 200 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 29 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| player_market_unavailable_at_close | 3 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 3 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2366 |
| player_prop_unavailable_at_close | 200 |
| line_disappeared_at_close | 29 |
| player_market_unavailable_at_close | 3 |
| fallback_other_book_only | 3 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 32eb2951f81d870c61eae4eb3afb7ea8 | draftkings | batter_hits | 34 | 12 | 64.7% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 8, "valid_close": 22} |
| fe2a389f5cf40c61156a794804ec0fff | fanduel | batter_total_bases | 52 | 12 | 76.9% | {"player_prop_unavailable_at_close": 12, "valid_close": 40} |
| 32eb2951f81d870c61eae4eb3afb7ea8 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| 40ac03ae2d2b1dc11d41ced3f411f478 | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| fe2a389f5cf40c61156a794804ec0fff | draftkings | batter_hits | 26 | 8 | 69.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 18} |
| a154d97ebad5a9a06417fe7578e7402f | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 7446dc66dc13eb736b1b6a5a61b450e4 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| fe2a389f5cf40c61156a794804ec0fff | fanduel | batter_hits | 26 | 6 | 76.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 20} |
| fe2a389f5cf40c61156a794804ec0fff | fanduel | batter_home_runs | 26 | 6 | 76.9% | {"player_prop_unavailable_at_close": 6, "valid_close": 20} |
| 32eb2951f81d870c61eae4eb3afb7ea8 | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 32eb2951f81d870c61eae4eb3afb7ea8 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 40ac03ae2d2b1dc11d41ced3f411f478 | draftkings | batter_hits | 30 | 6 | 80.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 40ac03ae2d2b1dc11d41ced3f411f478 | fanduel | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 40ac03ae2d2b1dc11d41ced3f411f478 | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 9cabe1742d3e487a116781a9b61b41ce | draftkings | batter_hits | 34 | 6 | 82.4% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| 7446dc66dc13eb736b1b6a5a61b450e4 | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 30} |
| a154d97ebad5a9a06417fe7578e7402f | fanduel | batter_hits | 57 | 5 | 91.2% | {"line_disappeared_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 52} |
| 16afd7bc765cb79fe5e887e2e28e0b20 | fanduel | pitcher_strikeouts | 6 | 4 | 33.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 2} |
| 32eb2951f81d870c61eae4eb3afb7ea8 | draftkings | batter_total_bases | 22 | 4 | 81.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 18} |
| 9d71a179d7d4dd618d4fae5516f00d95 | draftkings | batter_hits | 28 | 4 | 85.7% | {"line_disappeared_at_close": 4, "valid_close": 24} |
| b4977d853e9f601495f2a4a03c741589 | draftkings | batter_hits | 28 | 4 | 85.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| fd0e4e074c09a0148b633ff72e4450d7 | draftkings | batter_hits | 28 | 4 | 85.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| 16afd7bc765cb79fe5e887e2e28e0b20 | draftkings | batter_hits | 30 | 4 | 86.7% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 26} |
| a154d97ebad5a9a06417fe7578e7402f | draftkings | batter_hits | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 07753111ee3b354abbe8fab4987bb817 | draftkings | batter_hits | 32 | 4 | 87.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "valid_close": 28} |
| a154d97ebad5a9a06417fe7578e7402f | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 7446dc66dc13eb736b1b6a5a61b450e4 | fanduel | batter_home_runs | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 180ee18000bc994ba8c23c2ebc9ef746 | fanduel | batter_total_bases | 48 | 4 | 91.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 44} |
| b4977d853e9f601495f2a4a03c741589 | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| 07753111ee3b354abbe8fab4987bb817 | fanduel | batter_total_bases | 64 | 4 | 93.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 60} |
| 7446dc66dc13eb736b1b6a5a61b450e4 | fanduel | batter_hits | 64 | 4 | 93.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 60} |
| 9cabe1742d3e487a116781a9b61b41ce | fanduel | batter_total_bases | 64 | 4 | 93.8% | {"player_prop_unavailable_at_close": 4, "valid_close": 60} |
| 44263373c8152cfeef60f3501f9f0cbc | fanduel | batter_total_bases | 72 | 4 | 94.4% | {"player_prop_unavailable_at_close": 4, "valid_close": 68} |
| 16afd7bc765cb79fe5e887e2e28e0b20 | draftkings | pitcher_strikeouts | 4 | 2 | 50.0% | {"player_prop_unavailable_at_close": 2, "valid_close": 2} |
| 7446dc66dc13eb736b1b6a5a61b450e4 | fanduel | pitcher_strikeouts | 6 | 2 | 66.7% | {"line_disappeared_at_close": 2, "valid_close": 4} |
| 44263373c8152cfeef60f3501f9f0cbc | draftkings | batter_total_bases | 12 | 2 | 83.3% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "valid_close": 10} |
| fe2a389f5cf40c61156a794804ec0fff | draftkings | batter_total_bases | 14 | 2 | 85.7% | {"player_prop_unavailable_at_close": 2, "valid_close": 12} |
| 40ac03ae2d2b1dc11d41ced3f411f478 | draftkings | batter_total_bases | 16 | 2 | 87.5% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "valid_close": 14} |
| 07753111ee3b354abbe8fab4987bb817 | draftkings | batter_total_bases | 18 | 2 | 88.9% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "valid_close": 16} |
| 7446dc66dc13eb736b1b6a5a61b450e4 | draftkings | batter_total_bases | 18 | 2 | 88.9% | {"player_prop_unavailable_at_close": 2, "valid_close": 16} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| MIL @ PIT | 2026-07-12T16:16:00+00:00 | yes | yes | yes | 26 |
| NYY @ WAS | 2026-07-12T17:36:00+00:00 | yes | yes | yes | 30 |
| KC @ BAL | 2026-07-12T17:36:00+00:00 | yes | yes | yes | 30 |
| PHI @ DET | 2026-07-12T17:41:00+00:00 | yes | yes | yes | 30 |
| BOS @ NYM | 2026-07-12T17:41:00+00:00 | yes | yes | yes | 28 |
| CLE @ MIA | 2026-07-12T17:41:00+00:00 | yes | yes | yes | 29 |
| CHC @ CIN | 2026-07-12T17:42:00+00:00 | yes | yes | yes | 29 |
| SEA @ TB | 2026-07-12T17:43:00+00:00 | yes | yes | yes | 30 |
| ATH @ CWS | 2026-07-12T18:11:00+00:00 | yes | yes | yes | 30 |
| LAA @ MIN | 2026-07-12T18:11:00+00:00 | yes | yes | yes | 30 |
| ATL @ STL | 2026-07-12T18:16:00+00:00 | yes | yes | yes | 31 |
| HOU @ TEX | 2026-07-12T18:36:00+00:00 | yes | yes | yes | 32 |
| COL @ SF | 2026-07-12T20:06:00+00:00 | yes | yes | yes | 32 |
| ARI @ LAD | 2026-07-12T20:11:00+00:00 | yes | yes | yes | 33 |
| TOR @ SD | 2026-07-12T20:11:00+00:00 | yes | yes | yes | 33 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 0 |
| t_minus_60 | 0 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
