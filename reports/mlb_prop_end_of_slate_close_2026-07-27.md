# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-27T19:12:41Z
Slate: 2026-07-27
Evaluation: **PROVISIONAL**
Games final: 0 / 12
Strict clean slate: **False**

- Locked executable offers: 1924
- Valid exact closes: 152 (7.9%)
- Additional valid closes needed for 90%: 1580
- Stale closes: 2.4%
- T-120/T-60/T-20 complete events: 1 / 12

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 1580 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 11 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| close_outside_two_hour_window | 1714 | close was captured too early or after first pitch | scheduler timing problem; targeted captures should reduce this |
| stale_close_before_lock | 46 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |
| fallback_other_book_only | 1 | another book had a close but the original book did not | do not use cross-book fallback for promotion; use it only for diagnostics/display |
| player_market_unavailable_at_close | 1 | the player was present at close but this market was unavailable | rank by book/market and repair feed normalization or demote markets with repeat disappearance |

## Failure Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 1714 |
| valid_close | 152 |
| stale_close_before_lock | 46 |
| no_valid_close_snapshot | 10 |
| fallback_other_book_only | 1 |
| player_market_unavailable_at_close | 1 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|
| only_early_event_close_snapshots | 1714 |

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|
| fanduel | batter_total_bases | only_early_event_close_snapshots | 592 |
| draftkings | batter_hits | only_early_event_close_snapshots | 304 |
| fanduel | batter_hits | only_early_event_close_snapshots | 296 |
| fanduel | batter_home_runs | only_early_event_close_snapshots | 296 |
| draftkings | batter_total_bases | only_early_event_close_snapshots | 160 |
| draftkings | pitcher_strikeouts | only_early_event_close_snapshots | 34 |
| fanduel | pitcher_strikeouts | only_early_event_close_snapshots | 32 |

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 8f968dc825f43310933db311e1e61a6c | fanduel | batter_total_bases | 64 | 64 | 0.0% | {"close_outside_two_hour_window": 64} |
| 11fd3db49ab86a1ca24414fd4f9c4af6 | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 4f2695cf655e94d9dd1a7ef77b7861be | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 74749169ead486e15c8bbd906e738bd6 | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 9b0ec0a466486c849457165fd2c565e1 | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 01c25573da43dfef2b25036b3d865245 | fanduel | batter_total_bases | 56 | 56 | 0.0% | {"close_outside_two_hour_window": 56} |
| 302aa3f7c3d634878483c7edadbfdb0f | fanduel | batter_total_bases | 56 | 56 | 0.0% | {"close_outside_two_hour_window": 48, "stale_close_before_lock": 8} |
| 9053a0685edebb19cd19f34cb274be0d | fanduel | batter_total_bases | 52 | 52 | 0.0% | {"close_outside_two_hour_window": 52} |
| ee5f7b3a90890ec4fbb3146543125fe7 | fanduel | batter_total_bases | 52 | 52 | 0.0% | {"close_outside_two_hour_window": 52} |
| cd620c9f1e94a2b82e14a13d09fd7a77 | fanduel | batter_total_bases | 44 | 44 | 0.0% | {"close_outside_two_hour_window": 44} |
| 8f968dc825f43310933db311e1e61a6c | draftkings | batter_hits | 40 | 40 | 0.0% | {"close_outside_two_hour_window": 32, "no_valid_close_snapshot": 8} |
| 5f31d0e2c7e8dbd68ea839d172d9fd00 | fanduel | batter_total_bases | 36 | 36 | 0.0% | {"close_outside_two_hour_window": 36} |
| 11fd3db49ab86a1ca24414fd4f9c4af6 | draftkings | batter_hits | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 30, "stale_close_before_lock": 4} |
| 01c25573da43dfef2b25036b3d865245 | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 30, "stale_close_before_lock": 2} |
| 74749169ead486e15c8bbd906e738bd6 | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 30, "stale_close_before_lock": 2} |
| 8f968dc825f43310933db311e1e61a6c | fanduel | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 8f968dc825f43310933db311e1e61a6c | fanduel | batter_home_runs | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 11fd3db49ab86a1ca24414fd4f9c4af6 | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 11fd3db49ab86a1ca24414fd4f9c4af6 | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 302aa3f7c3d634878483c7edadbfdb0f | draftkings | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 28, "stale_close_before_lock": 2} |
| 4f2695cf655e94d9dd1a7ef77b7861be | draftkings | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 4f2695cf655e94d9dd1a7ef77b7861be | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 4f2695cf655e94d9dd1a7ef77b7861be | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 74749169ead486e15c8bbd906e738bd6 | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 74749169ead486e15c8bbd906e738bd6 | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 9b0ec0a466486c849457165fd2c565e1 | draftkings | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 9b0ec0a466486c849457165fd2c565e1 | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 9b0ec0a466486c849457165fd2c565e1 | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 01c25573da43dfef2b25036b3d865245 | fanduel | batter_hits | 28 | 28 | 0.0% | {"close_outside_two_hour_window": 28} |
| 01c25573da43dfef2b25036b3d865245 | fanduel | batter_home_runs | 28 | 28 | 0.0% | {"close_outside_two_hour_window": 28} |
| 302aa3f7c3d634878483c7edadbfdb0f | fanduel | batter_hits | 28 | 28 | 0.0% | {"close_outside_two_hour_window": 24, "stale_close_before_lock": 4} |
| 302aa3f7c3d634878483c7edadbfdb0f | fanduel | batter_home_runs | 28 | 28 | 0.0% | {"close_outside_two_hour_window": 24, "stale_close_before_lock": 4} |
| ee5f7b3a90890ec4fbb3146543125fe7 | draftkings | batter_hits | 28 | 28 | 0.0% | {"close_outside_two_hour_window": 26, "stale_close_before_lock": 2} |
| 8f968dc825f43310933db311e1e61a6c | draftkings | batter_total_bases | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| 9053a0685edebb19cd19f34cb274be0d | draftkings | batter_hits | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| 9053a0685edebb19cd19f34cb274be0d | fanduel | batter_hits | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| 9053a0685edebb19cd19f34cb274be0d | fanduel | batter_home_runs | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| ee5f7b3a90890ec4fbb3146543125fe7 | fanduel | batter_hits | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| ee5f7b3a90890ec4fbb3146543125fe7 | fanduel | batter_home_runs | 26 | 26 | 0.0% | {"close_outside_two_hour_window": 26} |
| cd620c9f1e94a2b82e14a13d09fd7a77 | draftkings | batter_hits | 24 | 24 | 0.0% | {"close_outside_two_hour_window": 22, "stale_close_before_lock": 2} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| SEA @ TEX | 2026-07-27T18:38:31+00:00 | yes | yes | yes | 21 |
| BAL @ DET | 2026-07-27T22:40:00+00:00 | no | no | no | 21 |
| ARI @ PIT | 2026-07-27T22:40:00+00:00 | no | no | no | 21 |
| PHI @ MIA | 2026-07-27T22:41:00+00:00 | no | no | no | 21 |
| TOR @ WAS | 2026-07-27T22:46:00+00:00 | no | no | no | 21 |
| CLE @ CIN | 2026-07-27T23:10:00+00:00 | no | no | no | 21 |
| ATL @ NYM | 2026-07-27T23:10:00+00:00 | no | no | no | 21 |
| NYY @ CWS | 2026-07-27T23:40:00+00:00 | no | no | no | 21 |
| CHC @ STL | 2026-07-27T23:46:00+00:00 | no | no | no | 20 |
| HOU @ LAA | 2026-07-28T01:38:00+00:00 | no | no | no | 21 |
| BOS @ ATH | 2026-07-28T01:40:00+00:00 | no | no | no | 21 |
| MIL @ SF | 2026-07-28T01:45:00+00:00 | no | no | no | 21 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 11 |
| t_minus_60 | 11 |
| t_minus_20 | 11 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| BAL @ DET | 2026-07-27T22:40:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| ARI @ PIT | 2026-07-27T22:40:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| PHI @ MIA | 2026-07-27T22:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| TOR @ WAS | 2026-07-27T22:46:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| CLE @ CIN | 2026-07-27T23:10:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| ATL @ NYM | 2026-07-27T23:10:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| NYY @ CWS | 2026-07-27T23:40:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| CHC @ STL | 2026-07-27T23:46:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 20 |
| HOU @ LAA | 2026-07-28T01:38:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| BOS @ ATH | 2026-07-28T01:40:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |
| MIL @ SF | 2026-07-28T01:45:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 21 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
