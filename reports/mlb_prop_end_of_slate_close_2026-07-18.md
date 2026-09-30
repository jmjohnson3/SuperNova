# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-19T04:03:21Z
Slate: 2026-07-18
Evaluation: **PROVISIONAL**
Games final: 13 / 16
Strict clean slate: **False**

- Locked executable offers: 2657
- Valid exact closes: 2190 (82.4%)
- Additional valid closes needed for 90%: 202
- Stale closes: 0.9%
- T-120/T-60/T-20 complete events: 7 / 16

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 202 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 9 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 246 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| close_outside_two_hour_window | 152 | close was captured too early or after first pitch | scheduler timing problem; targeted captures should reduce this |
| line_disappeared_at_close | 34 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| stale_close_before_lock | 25 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |
| player_market_unavailable_at_close | 5 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| fallback_other_book_only | 5 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2190 |
| player_prop_unavailable_at_close | 246 |
| close_outside_two_hour_window | 152 |
| line_disappeared_at_close | 34 |
| stale_close_before_lock | 25 |
| player_market_unavailable_at_close | 5 |
| fallback_other_book_only | 5 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|
| only_early_event_close_snapshots | 152 |

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|
| fanduel | batter_total_bases | only_early_event_close_snapshots | 52 |
| draftkings | batter_hits | only_early_event_close_snapshots | 28 |
| fanduel | batter_hits | only_early_event_close_snapshots | 26 |
| fanduel | batter_home_runs | only_early_event_close_snapshots | 26 |
| draftkings | batter_total_bases | only_early_event_close_snapshots | 12 |
| draftkings | pitcher_strikeouts | only_early_event_close_snapshots | 4 |
| fanduel | pitcher_strikeouts | only_early_event_close_snapshots | 4 |

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | fanduel | batter_total_bases | 52 | 52 | 0.0% | {"close_outside_two_hour_window": 52} |
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | draftkings | batter_hits | 28 | 28 | 0.0% | {"close_outside_two_hour_window": 28} |
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | fanduel | batter_hits | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | fanduel | batter_home_runs | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| ded1e582d76a9362d25395cef8921336 | draftkings | batter_hits | 36 | 16 | 55.6% | {"line_disappeared_at_close": 8, "player_prop_unavailable_at_close": 8, "valid_close": 20} |
| 1ac6eaa4fa03ab0a81e5ba09ce7bcaa3 | fanduel | batter_total_bases | 64 | 16 | 75.0% | {"player_prop_unavailable_at_close": 16, "valid_close": 48} |
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | draftkings | batter_total_bases | 12 | 12 | 0.0% | {"close_outside_two_hour_window": 12} |
| 698618fc0740b4cfa7a19834f721993e | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| ded1e582d76a9362d25395cef8921336 | fanduel | batter_total_bases | 56 | 12 | 78.6% | {"player_prop_unavailable_at_close": 12, "valid_close": 44} |
| d8bf9cd37db11dd5e36f7b8d846883b4 | draftkings | batter_total_bases | 20 | 10 | 50.0% | {"fallback_other_book_only": 2, "player_market_unavailable_at_close": 2, "stale_close_before_lock": 6, "valid_close": 10} |
| d8bf9cd37db11dd5e36f7b8d846883b4 | draftkings | batter_hits | 40 | 10 | 75.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 2, "stale_close_before_lock": 6, "valid_close": 30} |
| 054bfd160adb6a374830d05222e3f585 | draftkings | batter_hits | 32 | 8 | 75.0% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| 1ac6eaa4fa03ab0a81e5ba09ce7bcaa3 | draftkings | batter_hits | 32 | 8 | 75.0% | {"player_prop_unavailable_at_close": 8, "valid_close": 24} |
| 1ac6eaa4fa03ab0a81e5ba09ce7bcaa3 | fanduel | batter_hits | 32 | 8 | 75.0% | {"player_prop_unavailable_at_close": 8, "valid_close": 24} |
| 1ac6eaa4fa03ab0a81e5ba09ce7bcaa3 | fanduel | batter_home_runs | 32 | 8 | 75.0% | {"player_prop_unavailable_at_close": 8, "valid_close": 24} |
| 2962ef39359f54f01f0254f086b45881 | draftkings | batter_hits | 32 | 8 | 75.0% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 1b7f4086b3b3d2a039917465354f741b | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| 2962ef39359f54f01f0254f086b45881 | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| fc2051a727661b285a4718bc4738966e | fanduel | batter_total_bases | 56 | 8 | 85.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 48} |
| d8bf9cd37db11dd5e36f7b8d846883b4 | fanduel | batter_hits | 35 | 7 | 80.0% | {"stale_close_before_lock": 7, "valid_close": 28} |
| ded1e582d76a9362d25395cef8921336 | draftkings | batter_total_bases | 24 | 6 | 75.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 18} |
| 2962ef39359f54f01f0254f086b45881 | draftkings | batter_total_bases | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 698618fc0740b4cfa7a19834f721993e | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 698618fc0740b4cfa7a19834f721993e | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 698618fc0740b4cfa7a19834f721993e | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| ded1e582d76a9362d25395cef8921336 | fanduel | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| ded1e582d76a9362d25395cef8921336 | fanduel | batter_home_runs | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 31b5f1a35ef3661e5911f39a57d11ac5 | draftkings | batter_hits | 34 | 6 | 82.4% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 28} |
| 3ccc4f228a6ed52e09f6277f5d002fdc | draftkings | batter_hits | 36 | 6 | 83.3% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 30} |
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | draftkings | pitcher_strikeouts | 4 | 4 | 0.0% | {"close_outside_two_hour_window": 4} |
| b1c11c1965517ee6fc64d8fcb9f8e6e3 | fanduel | pitcher_strikeouts | 4 | 4 | 0.0% | {"close_outside_two_hour_window": 4} |
| 1b7f4086b3b3d2a039917465354f741b | draftkings | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 1b7f4086b3b3d2a039917465354f741b | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 1b7f4086b3b3d2a039917465354f741b | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 2962ef39359f54f01f0254f086b45881 | fanduel | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| 2962ef39359f54f01f0254f086b45881 | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| fc2051a727661b285a4718bc4738966e | draftkings | batter_hits | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| fc2051a727661b285a4718bc4738966e | fanduel | batter_home_runs | 28 | 4 | 85.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 24} |
| af96de180c395ddeb2a1600317946494 | draftkings | batter_hits | 30 | 4 | 86.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 48ca8f5f46eb95a95a448679ea6cf7b1 | fanduel | batter_total_bases | 48 | 4 | 91.7% | {"player_prop_unavailable_at_close": 4, "valid_close": 44} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| PIT @ CLE | 2026-07-18T17:11:00+00:00 | yes | yes | yes | 45 |
| CWS @ TOR | 2026-07-18T19:40:00+00:00 | yes | yes | yes | 65 |
| NYM @ PHI | 2026-07-18T20:06:00+00:00 | yes | yes | yes | 76 |
| SF @ SEA | 2026-07-19T00:09:00+00:00 | yes | yes | yes | 100 |
| PIT @ CLE | 2026-07-19T00:25:17+00:00 | yes | yes | yes | 85 |
| WAS @ ATH | 2026-07-19T02:06:00+00:00 | yes | yes | yes | 101 |
| DET @ LAA | 2026-07-19T02:08:00+00:00 | yes | yes | yes | 101 |
| MIN @ CHC | 2026-07-18T18:21:00+00:00 | no | yes | yes | 55 |
| CIN @ COL | 2026-07-18T19:11:00+00:00 | no | yes | yes | 54 |
| TEX @ ATL | 2026-07-18T20:11:00+00:00 | no | yes | yes | 81 |
| MIA @ MIL | 2026-07-18T20:11:00+00:00 | no | yes | yes | 84 |
| STL @ ARI | 2026-07-18T20:11:00+00:00 | no | yes | yes | 78 |
| BAL @ HOU | 2026-07-18T20:11:00+00:00 | no | yes | yes | 88 |
| SD @ KC | 2026-07-18T20:11:00+00:00 | no | yes | yes | 77 |
| TB @ BOS | 2026-07-18T21:11:00+00:00 | no | yes | yes | 88 |
| LAD @ NYY | 2026-07-19T00:09:00+00:00 | no | no | no | 55 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 9 |
| t_minus_60 | 1 |
| t_minus_20 | 1 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| MIN @ CHC | 2026-07-18T18:21:00+00:00 | t_minus_120 | 55 |
| CIN @ COL | 2026-07-18T19:11:00+00:00 | t_minus_120 | 54 |
| TEX @ ATL | 2026-07-18T20:11:00+00:00 | t_minus_120 | 81 |
| MIA @ MIL | 2026-07-18T20:11:00+00:00 | t_minus_120 | 84 |
| STL @ ARI | 2026-07-18T20:11:00+00:00 | t_minus_120 | 78 |
| BAL @ HOU | 2026-07-18T20:11:00+00:00 | t_minus_120 | 88 |
| SD @ KC | 2026-07-18T20:11:00+00:00 | t_minus_120 | 77 |
| TB @ BOS | 2026-07-18T21:11:00+00:00 | t_minus_120 | 88 |
| LAD @ NYY | 2026-07-19T00:09:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 55 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
