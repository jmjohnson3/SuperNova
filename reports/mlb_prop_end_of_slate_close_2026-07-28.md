# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-28T14:11:25Z
Slate: 2026-07-28
Evaluation: **PROVISIONAL**
Games final: 0 / 16
Strict clean slate: **False**

- Locked executable offers: 2449
- Valid exact closes: 0 (0.0%)
- Additional valid closes needed for 90%: 2205
- Stale closes: 1.5%
- T-120/T-60/T-20 complete events: 0 / 15

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 2205 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 15 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| close_outside_two_hour_window | 2403 | close was captured too early or after first pitch | scheduler timing problem; targeted captures should reduce this |
| stale_close_before_lock | 36 | close snapshot timestamp was before the bet lock | lock/close ordering problem; do not count as CLV |

## Failure Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 2403 |
| stale_close_before_lock | 36 |
| no_valid_close_snapshot | 10 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|
| only_early_event_close_snapshots | 2403 |

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|
| fanduel | batter_total_bases | only_early_event_close_snapshots | 839 |
| draftkings | batter_hits | only_early_event_close_snapshots | 426 |
| fanduel | batter_hits | only_early_event_close_snapshots | 426 |
| fanduel | batter_home_runs | only_early_event_close_snapshots | 426 |
| draftkings | batter_total_bases | only_early_event_close_snapshots | 200 |
| fanduel | pitcher_strikeouts | only_early_event_close_snapshots | 44 |
| draftkings | pitcher_strikeouts | only_early_event_close_snapshots | 42 |

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 83b4bc6720bc5b4953002ee5327e881d | fanduel | batter_total_bases | 68 | 68 | 0.0% | {"close_outside_two_hour_window": 68} |
| 1c02a0524bf42a304b0b1cf019ea12c8 | fanduel | batter_total_bases | 64 | 64 | 0.0% | {"close_outside_two_hour_window": 64} |
| 3200119bc7fbf549a208a197223c9966 | fanduel | batter_total_bases | 64 | 64 | 0.0% | {"close_outside_two_hour_window": 64} |
| 452645d823de15633407805cf5bc269a | fanduel | batter_total_bases | 64 | 64 | 0.0% | {"close_outside_two_hour_window": 64} |
| 9ad4c06c6c9f5cf366798cada970f547 | fanduel | batter_total_bases | 64 | 64 | 0.0% | {"close_outside_two_hour_window": 64} |
| f829a782d542dc9d3eb459cddb29492c | fanduel | batter_total_bases | 64 | 64 | 0.0% | {"close_outside_two_hour_window": 64} |
| 8fa2b64a3a88b00af2cc037c70eaed6e | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 979a29c09433f74c9cf81057e852ddf2 | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 6fb81f375a5fc10737ecb85381f1ac0b | fanduel | batter_total_bases | 56 | 56 | 0.0% | {"close_outside_two_hour_window": 56} |
| b77df217e739f24b8ab3440a9f013da0 | fanduel | batter_total_bases | 56 | 56 | 0.0% | {"close_outside_two_hour_window": 56} |
| 652c3471321a108cc9222b8e5b773ad2 | fanduel | batter_total_bases | 52 | 52 | 0.0% | {"close_outside_two_hour_window": 52} |
| 0f350cb709d6394c946ef1df6fca40af | fanduel | batter_total_bases | 48 | 48 | 0.0% | {"close_outside_two_hour_window": 44, "stale_close_before_lock": 4} |
| a981fee543b7844ab5bf4928bdca43d2 | fanduel | batter_total_bases | 44 | 44 | 0.0% | {"close_outside_two_hour_window": 44} |
| 828e21d2b1a6b79dd714546f52d7a38d | fanduel | batter_total_bases | 40 | 40 | 0.0% | {"close_outside_two_hour_window": 40} |
| 0c2627afd8b3d72890ab2725df592412 | fanduel | batter_total_bases | 39 | 39 | 0.0% | {"close_outside_two_hour_window": 39} |
| 452645d823de15633407805cf5bc269a | draftkings | batter_hits | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 32, "stale_close_before_lock": 2} |
| 83b4bc6720bc5b4953002ee5327e881d | draftkings | batter_hits | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 30, "no_valid_close_snapshot": 4} |
| 83b4bc6720bc5b4953002ee5327e881d | fanduel | batter_hits | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 34} |
| 83b4bc6720bc5b4953002ee5327e881d | fanduel | batter_home_runs | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 34} |
| 9ad4c06c6c9f5cf366798cada970f547 | draftkings | batter_hits | 34 | 34 | 0.0% | {"close_outside_two_hour_window": 32, "stale_close_before_lock": 2} |
| 1c02a0524bf42a304b0b1cf019ea12c8 | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 1c02a0524bf42a304b0b1cf019ea12c8 | fanduel | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 1c02a0524bf42a304b0b1cf019ea12c8 | fanduel | batter_home_runs | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 3200119bc7fbf549a208a197223c9966 | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 3200119bc7fbf549a208a197223c9966 | fanduel | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 3200119bc7fbf549a208a197223c9966 | fanduel | batter_home_runs | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 452645d823de15633407805cf5bc269a | fanduel | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 452645d823de15633407805cf5bc269a | fanduel | batter_home_runs | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 6fb81f375a5fc10737ecb85381f1ac0b | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 28, "stale_close_before_lock": 4} |
| 8fa2b64a3a88b00af2cc037c70eaed6e | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 30, "stale_close_before_lock": 2} |
| 9ad4c06c6c9f5cf366798cada970f547 | fanduel | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 9ad4c06c6c9f5cf366798cada970f547 | fanduel | batter_home_runs | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| f829a782d542dc9d3eb459cddb29492c | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| f829a782d542dc9d3eb459cddb29492c | fanduel | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| f829a782d542dc9d3eb459cddb29492c | fanduel | batter_home_runs | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 8fa2b64a3a88b00af2cc037c70eaed6e | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 8fa2b64a3a88b00af2cc037c70eaed6e | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 979a29c09433f74c9cf81057e852ddf2 | draftkings | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 979a29c09433f74c9cf81057e852ddf2 | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 979a29c09433f74c9cf81057e852ddf2 | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| CLE @ CIN | 2026-07-28T17:41:00+00:00 | no | no | no | 2 |
| TEX @ TB | 2026-07-28T22:41:00+00:00 | no | no | no | 2 |
| ARI @ PIT | 2026-07-28T22:41:00+00:00 | no | no | no | 2 |
| BAL @ DET | 2026-07-28T22:41:00+00:00 | no | no | no | 2 |
| PHI @ MIA | 2026-07-28T22:41:00+00:00 | no | no | no | 2 |
| TOR @ WAS | 2026-07-28T22:46:00+00:00 | no | no | no | 2 |
| ATL @ NYM | 2026-07-28T23:11:00+00:00 | no | no | no | 2 |
| NYY @ CWS | 2026-07-28T23:41:00+00:00 | no | no | no | 2 |
| KC @ MIN | 2026-07-28T23:41:00+00:00 | no | no | no | 2 |
| CHC @ STL | 2026-07-28T23:46:00+00:00 | no | no | no | 1 |
| HOU @ LAA | 2026-07-29T01:39:00+00:00 | no | no | no | 2 |
| BOS @ ATH | 2026-07-29T01:41:00+00:00 | no | no | no | 2 |
| COL @ SD | 2026-07-29T01:41:00+00:00 | no | no | no | 2 |
| MIL @ SF | 2026-07-29T01:46:00+00:00 | no | no | no | 2 |
| SEA @ LAD | 2026-07-29T02:11:00+00:00 | no | no | no | 1 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 15 |
| t_minus_60 | 15 |
| t_minus_20 | 15 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| CLE @ CIN | 2026-07-28T17:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| TEX @ TB | 2026-07-28T22:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| ARI @ PIT | 2026-07-28T22:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| BAL @ DET | 2026-07-28T22:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| PHI @ MIA | 2026-07-28T22:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| TOR @ WAS | 2026-07-28T22:46:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| ATL @ NYM | 2026-07-28T23:11:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| NYY @ CWS | 2026-07-28T23:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| KC @ MIN | 2026-07-28T23:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| CHC @ STL | 2026-07-28T23:46:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 1 |
| HOU @ LAA | 2026-07-29T01:39:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| BOS @ ATH | 2026-07-29T01:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| COL @ SD | 2026-07-29T01:41:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| MIL @ SF | 2026-07-29T01:46:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 2 |
| SEA @ LAD | 2026-07-29T02:11:00+00:00 | t_minus_120, t_minus_60, t_minus_20 | 1 |

> This slate is still provisional. Coverage must be evaluated again after every game is final.
