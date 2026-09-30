# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-18T04:00:30Z
Slate: 2026-07-17
Evaluation: **PROVISIONAL**
Games final: 10 / 15
Strict clean slate: **False**

- Locked executable offers: 2777
- Valid exact closes: 2314 (83.3%)
- Additional valid closes needed for 90%: 186
- Stale closes: 0.3%
- T-120/T-60/T-20 complete events: 13 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 186 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 2 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| close_outside_two_hour_window | 218 | close was captured too early or after first pitch | scheduler timing problem; targeted captures should reduce this |
| player_prop_unavailable_at_close | 166 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| line_disappeared_at_close | 52 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| stale_close_before_lock | 9 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |
| player_market_unavailable_at_close | 8 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 8 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2314 |
| close_outside_two_hour_window | 218 |
| player_prop_unavailable_at_close | 166 |
| line_disappeared_at_close | 52 |
| stale_close_before_lock | 9 |
| player_market_unavailable_at_close | 8 |
| fallback_other_book_only | 8 |
| no_valid_close_snapshot | 2 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|
| only_early_event_close_snapshots | 218 |

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|
| fanduel | batter_total_bases | only_early_event_close_snapshots | 68 |
| fanduel | batter_hits | only_early_event_close_snapshots | 60 |
| draftkings | batter_hits | only_early_event_close_snapshots | 36 |
| fanduel | batter_home_runs | only_early_event_close_snapshots | 34 |
| draftkings | batter_total_bases | only_early_event_close_snapshots | 14 |
| fanduel | pitcher_strikeouts | only_early_event_close_snapshots | 4 |
| draftkings | pitcher_strikeouts | only_early_event_close_snapshots | 2 |

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| c137da348b9193a96f85af78c01a1aca | fanduel | batter_total_bases | 68 | 68 | 0.0% | {"close_outside_two_hour_window": 68} |
| c137da348b9193a96f85af78c01a1aca | fanduel | batter_hits | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| c137da348b9193a96f85af78c01a1aca | draftkings | batter_hits | 36 | 36 | 0.0% | {"close_outside_two_hour_window": 36} |
| c137da348b9193a96f85af78c01a1aca | fanduel | batter_home_runs | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 34} |
| e0989f5a2163779f2b0c456d7f844ce9 | fanduel | batter_total_bases | 60 | 20 | 66.7% | {"player_prop_unavailable_at_close": 20, "valid_close": 40} |
| c137da348b9193a96f85af78c01a1aca | draftkings | batter_total_bases | 14 | 14 | 0.0% | {"close_outside_two_hour_window": 14} |
| 2e309c8e4682895ea0516d055b78f53a | draftkings | batter_hits | 36 | 12 | 66.7% | {"line_disappeared_at_close": 8, "player_prop_unavailable_at_close": 4, "valid_close": 24} |
| e0989f5a2163779f2b0c456d7f844ce9 | fanduel | batter_hits | 30 | 10 | 66.7% | {"player_prop_unavailable_at_close": 10, "valid_close": 20} |
| e0989f5a2163779f2b0c456d7f844ce9 | fanduel | batter_home_runs | 30 | 10 | 66.7% | {"player_prop_unavailable_at_close": 10, "valid_close": 20} |
| e0989f5a2163779f2b0c456d7f844ce9 | draftkings | batter_hits | 26 | 8 | 69.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 18} |
| 9d0976062d4e6d31f94ff2c8b3baff89 | draftkings | batter_hits | 32 | 8 | 75.0% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 24} |
| e9c736dab72b5dc33dd4f665870c8f2b | draftkings | batter_hits | 32 | 8 | 75.0% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 704ba175e571a9f003f1c83cd86e48e6 | draftkings | batter_hits | 34 | 8 | 76.5% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 7d92a7dff1e2a4e0569b73cf9069f2e3 | draftkings | batter_hits | 40 | 8 | 80.0% | {"line_disappeared_at_close": 8, "valid_close": 32} |
| 2e309c8e4682895ea0516d055b78f53a | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| e9c736dab72b5dc33dd4f665870c8f2b | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 9d0976062d4e6d31f94ff2c8b3baff89 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| 43706fff6ae5bb5201b9add3c0945034 | fanduel | batter_hits | 39 | 7 | 82.1% | {"player_prop_unavailable_at_close": 2, "stale_close_before_lock": 5, "valid_close": 32} |
| e047c3dd3d688fead07d77b8b32852af | draftkings | batter_total_bases | 16 | 6 | 62.5% | {"fallback_other_book_only": 3, "player_market_unavailable_at_close": 3, "valid_close": 10} |
| e0989f5a2163779f2b0c456d7f844ce9 | draftkings | batter_total_bases | 20 | 6 | 70.0% | {"fallback_other_book_only": 1, "player_market_unavailable_at_close": 1, "player_prop_unavailable_at_close": 4, "valid_close": 14} |
| cfb2e85cbd19eb03e617782d90d2c60e | draftkings | batter_hits | 32 | 6 | 81.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 43706fff6ae5bb5201b9add3c0945034 | draftkings | batter_hits | 38 | 6 | 84.2% | {"line_disappeared_at_close": 4, "stale_close_before_lock": 2, "valid_close": 32} |
| c137da348b9193a96f85af78c01a1aca | draftkings | pitcher_strikeouts | 4 | 4 | 0.0% | {"close_outside_two_hour_window": 2, "no_valid_close_snapshot": 2} |
| c137da348b9193a96f85af78c01a1aca | fanduel | pitcher_strikeouts | 4 | 4 | 0.0% | {"close_outside_two_hour_window": 4} |
| e0989f5a2163779f2b0c456d7f844ce9 | draftkings | pitcher_strikeouts | 8 | 4 | 50.0% | {"line_disappeared_at_close": 4, "valid_close": 4} |
| dbf668de78fe1352520d4e2011c77603 | draftkings | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| e9c736dab72b5dc33dd4f665870c8f2b | draftkings | batter_total_bases | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 2e309c8e4682895ea0516d055b78f53a | fanduel | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 2e309c8e4682895ea0516d055b78f53a | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| e9c736dab72b5dc33dd4f665870c8f2b | fanduel | batter_hits | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| e9c736dab72b5dc33dd4f665870c8f2b | fanduel | batter_home_runs | 32 | 4 | 87.5% | {"player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 9d0976062d4e6d31f94ff2c8b3baff89 | fanduel | batter_hits | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| 9d0976062d4e6d31f94ff2c8b3baff89 | fanduel | batter_home_runs | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| e047c3dd3d688fead07d77b8b32852af | draftkings | batter_hits | 34 | 4 | 88.2% | {"player_prop_unavailable_at_close": 4, "valid_close": 30} |
| dbf668de78fe1352520d4e2011c77603 | fanduel | batter_total_bases | 52 | 4 | 92.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 48} |
| 704ba175e571a9f003f1c83cd86e48e6 | fanduel | batter_total_bases | 60 | 4 | 93.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 56} |
| cfb2e85cbd19eb03e617782d90d2c60e | fanduel | batter_total_bases | 60 | 4 | 93.3% | {"player_prop_unavailable_at_close": 4, "valid_close": 56} |
| 43706fff6ae5bb5201b9add3c0945034 | fanduel | batter_total_bases | 68 | 4 | 94.1% | {"player_prop_unavailable_at_close": 4, "valid_close": 64} |
| 25f714905d486c907ee2e0c1377097ea | draftkings | pitcher_strikeouts | 4 | 2 | 50.0% | {"line_disappeared_at_close": 2, "valid_close": 2} |
| 704ba175e571a9f003f1c83cd86e48e6 | draftkings | pitcher_strikeouts | 6 | 2 | 66.7% | {"line_disappeared_at_close": 2, "valid_close": 4} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| TB @ BOS | 2026-07-17T17:36:00+00:00 | yes | yes | yes | 22 |
| LAD @ NYY | 2026-07-17T23:06:00+00:00 | yes | yes | yes | 81 |
| TB @ BOS | 2026-07-17T23:11:00+00:00 | yes | yes | yes | 77 |
| TEX @ ATL | 2026-07-17T23:16:00+00:00 | yes | yes | yes | 81 |
| CWS @ TOR | 2026-07-17T23:16:00+00:00 | yes | yes | yes | 81 |
| MIA @ MIL | 2026-07-17T23:41:00+00:00 | yes | yes | yes | 82 |
| MIN @ CHC | 2026-07-18T00:06:00+00:00 | yes | yes | yes | 82 |
| SD @ KC | 2026-07-18T00:11:00+00:00 | yes | yes | yes | 82 |
| BAL @ HOU | 2026-07-18T00:11:00+00:00 | yes | yes | yes | 82 |
| CIN @ COL | 2026-07-18T00:41:00+00:00 | yes | yes | yes | 82 |
| DET @ LAA | 2026-07-18T01:39:00+00:00 | yes | yes | yes | 83 |
| STL @ ARI | 2026-07-18T01:41:00+00:00 | yes | yes | yes | 83 |
| WAS @ ATH | 2026-07-18T01:41:00+00:00 | yes | yes | yes | 83 |
| PIT @ CLE | 2026-07-17T23:11:00+00:00 | no | no | no | 25 |
| SF @ SEA | 2026-07-18T02:11:00+00:00 | no | yes | yes | 83 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 2 |
| t_minus_60 | 1 |
| t_minus_20 | 1 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| PIT @ CLE | 2026-07-17T23:11:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 25 |
| SF @ SEA | 2026-07-18T02:11:00+00:00 | t_minus_120 | 83 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
