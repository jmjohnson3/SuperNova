# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-22T04:09:33Z
Slate: 2026-07-21
Evaluation: **PROVISIONAL**
Games final: 10 / 15
Strict clean slate: **False**

- Locked executable offers: 2789
- Valid exact closes: 2231 (80.0%)
- Additional valid closes needed for 90%: 280
- Stale closes: 0.1%
- T-120/T-60/T-20 complete events: 12 / 16

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 280 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 4 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| player_prop_unavailable_at_close | 341 | the book still had the game, but the player prop menu was gone for that player | treat as line availability/bookability evidence; extra late snapshots may help, but promotion must require close availability |
| close_outside_two_hour_window | 144 | close was captured too early or after first pitch | scheduler timing problem; targeted captures should reduce this |
| line_disappeared_at_close | 46 | same player and market existed at close, but the exact line moved or disappeared | store nearest-line movement separately; exact-bucket promotion should keep this as not bookable |
| fallback_other_book_only | 13 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| player_market_unavailable_at_close | 12 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |
| stale_close_before_lock | 2 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |

## Failure Reasons

| Reason | Rows |
|---|---:|
| valid_close | 2231 |
| player_prop_unavailable_at_close | 341 |
| close_outside_two_hour_window | 144 |
| line_disappeared_at_close | 46 |
| fallback_other_book_only | 13 |
| player_market_unavailable_at_close | 12 |
| stale_close_before_lock | 2 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|
| only_early_event_close_snapshots | 144 |

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|
| fanduel | batter_total_bases | only_early_event_close_snapshots | 48 |
| draftkings | batter_hits | only_early_event_close_snapshots | 24 |
| fanduel | batter_hits | only_early_event_close_snapshots | 24 |
| fanduel | batter_home_runs | only_early_event_close_snapshots | 24 |
| draftkings | batter_total_bases | only_early_event_close_snapshots | 16 |
| draftkings | pitcher_strikeouts | only_early_event_close_snapshots | 4 |
| fanduel | pitcher_strikeouts | only_early_event_close_snapshots | 4 |

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 4b39d95fa5e27f227c44ec3a8f68aaf4 | fanduel | batter_total_bases | 48 | 48 | 0.0% | {"close_outside_two_hour_window": 48} |
| 4b39d95fa5e27f227c44ec3a8f68aaf4 | draftkings | batter_hits | 24 | 24 | 0.0% | {"close_outside_two_hour_window": 24} |
| 4b39d95fa5e27f227c44ec3a8f68aaf4 | fanduel | batter_hits | 24 | 24 | 0.0% | {"close_outside_two_hour_window": 24} |
| 4b39d95fa5e27f227c44ec3a8f68aaf4 | fanduel | batter_home_runs | 24 | 24 | 0.0% | {"close_outside_two_hour_window": 24} |
| 4b39d95fa5e27f227c44ec3a8f68aaf4 | draftkings | batter_total_bases | 18 | 18 | 0.0% | {"close_outside_two_hour_window": 16, "stale_close_before_lock": 2} |
| a33e389530d74f9495e5d0108d6b038b | fanduel | batter_total_bases | 72 | 16 | 77.8% | {"player_prop_unavailable_at_close": 16, "valid_close": 56} |
| 267dae014ac02839048da8da0bf226c3 | draftkings | batter_hits | 34 | 12 | 64.7% | {"line_disappeared_at_close": 6, "player_prop_unavailable_at_close": 6, "valid_close": 22} |
| a33e389530d74f9495e5d0108d6b038b | draftkings | batter_hits | 40 | 12 | 70.0% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 8, "valid_close": 28} |
| 3440c51cbfebc0ac81d574f6052d2eae | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| 8b96a836e0e20dedb471fa42febff567 | fanduel | batter_total_bases | 60 | 12 | 80.0% | {"player_prop_unavailable_at_close": 12, "valid_close": 48} |
| 0423e391e6f8cfd58c7f2f1c72d0aec7 | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| 267dae014ac02839048da8da0bf226c3 | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| aac7942aa7683280dcbfe7dcfc16d44e | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| ded163387935cca0e01c3281ff323291 | fanduel | batter_total_bases | 64 | 12 | 81.2% | {"player_prop_unavailable_at_close": 12, "valid_close": 52} |
| 299179879cf35ddaecccdd15f48fb177 | fanduel | batter_total_bases | 68 | 12 | 82.4% | {"player_prop_unavailable_at_close": 12, "valid_close": 56} |
| 38b96dae7f6cab01cf931c9faefc2606 | draftkings | batter_total_bases | 18 | 8 | 55.6% | {"fallback_other_book_only": 4, "player_market_unavailable_at_close": 4, "valid_close": 10} |
| aac7942aa7683280dcbfe7dcfc16d44e | draftkings | batter_hits | 34 | 8 | 76.5% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 0d8a10122998e00420770fa704603388 | draftkings | batter_hits | 36 | 8 | 77.8% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 6, "valid_close": 28} |
| a33e389530d74f9495e5d0108d6b038b | fanduel | batter_home_runs | 36 | 8 | 77.8% | {"player_prop_unavailable_at_close": 8, "valid_close": 28} |
| ebe684b6aa76271158a53ba859a81432 | fanduel | batter_total_bases | 60 | 8 | 86.7% | {"player_prop_unavailable_at_close": 8, "valid_close": 52} |
| 0d8a10122998e00420770fa704603388 | fanduel | batter_total_bases | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| a33e389530d74f9495e5d0108d6b038b | fanduel | batter_hits | 64 | 8 | 87.5% | {"player_prop_unavailable_at_close": 8, "valid_close": 56} |
| 38b96dae7f6cab01cf931c9faefc2606 | fanduel | batter_total_bases | 68 | 8 | 88.2% | {"player_prop_unavailable_at_close": 8, "valid_close": 60} |
| 3440c51cbfebc0ac81d574f6052d2eae | draftkings | batter_hits | 28 | 6 | 78.6% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 22} |
| 8b96a836e0e20dedb471fa42febff567 | draftkings | batter_hits | 28 | 6 | 78.6% | {"player_prop_unavailable_at_close": 6, "valid_close": 22} |
| 0d8a10122998e00420770fa704603388 | draftkings | batter_total_bases | 30 | 6 | 80.0% | {"line_disappeared_at_close": 4, "player_prop_unavailable_at_close": 2, "valid_close": 24} |
| 3440c51cbfebc0ac81d574f6052d2eae | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 3440c51cbfebc0ac81d574f6052d2eae | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 8b96a836e0e20dedb471fa42febff567 | fanduel | batter_hits | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 8b96a836e0e20dedb471fa42febff567 | fanduel | batter_home_runs | 30 | 6 | 80.0% | {"player_prop_unavailable_at_close": 6, "valid_close": 24} |
| 0423e391e6f8cfd58c7f2f1c72d0aec7 | fanduel | batter_hits | 32 | 6 | 81.2% | {"fallback_other_book_only": 1, "player_prop_unavailable_at_close": 5, "valid_close": 26} |
| 0423e391e6f8cfd58c7f2f1c72d0aec7 | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 267dae014ac02839048da8da0bf226c3 | fanduel | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| 267dae014ac02839048da8da0bf226c3 | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| aac7942aa7683280dcbfe7dcfc16d44e | fanduel | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| aac7942aa7683280dcbfe7dcfc16d44e | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| ded163387935cca0e01c3281ff323291 | draftkings | batter_hits | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| ded163387935cca0e01c3281ff323291 | fanduel | batter_home_runs | 32 | 6 | 81.2% | {"player_prop_unavailable_at_close": 6, "valid_close": 26} |
| ebe684b6aa76271158a53ba859a81432 | draftkings | batter_hits | 32 | 6 | 81.2% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 26} |
| 299179879cf35ddaecccdd15f48fb177 | draftkings | batter_hits | 34 | 6 | 82.4% | {"line_disappeared_at_close": 2, "player_prop_unavailable_at_close": 4, "valid_close": 28} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| MIN @ CLE | 2026-07-21T22:41:00+00:00 | yes | yes | yes | 48 |
| TB @ TOR | 2026-07-21T23:08:00+00:00 | yes | yes | yes | 52 |
| SD @ ATL | 2026-07-21T23:16:00+00:00 | yes | yes | yes | 52 |
| SF @ KC | 2026-07-21T23:41:00+00:00 | yes | yes | yes | 52 |
| NYM @ MIL | 2026-07-21T23:41:00+00:00 | yes | yes | yes | 52 |
| CWS @ TEX | 2026-07-22T00:06:00+00:00 | yes | yes | yes | 52 |
| DET @ CHC | 2026-07-22T00:06:00+00:00 | yes | yes | yes | 53 |
| MIA @ HOU | 2026-07-22T00:11:00+00:00 | yes | yes | yes | 53 |
| WAS @ COL | 2026-07-22T00:41:00+00:00 | yes | yes | yes | 54 |
| STL @ LAA | 2026-07-22T01:39:00+00:00 | yes | yes | yes | 54 |
| CIN @ SEA | 2026-07-22T01:41:00+00:00 | yes | yes | yes | 54 |
| ATH @ ARI | 2026-07-22T01:41:00+00:00 | yes | yes | yes | 53 |
| PIT @ NYY | 2026-07-21T23:06:00+00:00 | no | no | no | 7 |
| BAL @ BOS | 2026-07-21T23:11:00+00:00 | yes | no | no | 21 |
| LAD @ PHI | 2026-07-22T00:01:00+00:00 | no | no | yes | 52 |
| PIT @ NYY | 2026-07-22T17:06:00+00:00 | no | no | no | 1 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 3 |
| t_minus_60 | 4 |
| t_minus_20 | 3 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| PIT @ NYY | 2026-07-21T23:06:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 7 |
| BAL @ BOS | 2026-07-21T23:11:00+00:00 | t_minus_60, t_minus_20 | 21 |
| LAD @ PHI | 2026-07-22T00:01:00+00:00 | t_minus_120, t_minus_60 | 52 |
| PIT @ NYY | 2026-07-22T17:06:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 1 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
