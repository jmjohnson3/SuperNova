# MLB Prop End-of-Slate Close Diagnostic

Generated UTC: 2026-07-17T03:12:28Z
Slate: 2026-07-16
Evaluation: **FINAL**
Games final: 1 / 1
Strict clean slate: **False**

- Locked executable offers: 178
- Valid exact closes: 0 (0.0%)
- Additional valid closes needed for 90%: 161
- Stale closes: 0.0%
- T-120/T-60/T-20 complete events: 0 / 1

## Action Plan

| Area | Rows | Diagnosis | Next Action |
|---|---:|---|---|
| coverage_gap_to_90 | 161 | additional valid exact closes needed for this slate to clear the clean-promotion threshold | supplemental T-90/T-45/T-30/T-10 captures are enabled; keep the targeted-close task running through every game start |
| required_target_windows | 1 | events missing at least one required T-120/T-60/T-20 capture | verify Task Scheduler history and mutex skips for those event windows |
| close_outside_two_hour_window | 176 | close was captured too early or after first pitch | scheduler timing problem; targeted captures should reduce this |

## Failure Reasons

| Reason | Rows |
|---|---:|
| close_outside_two_hour_window | 176 |
| no_valid_close_snapshot | 2 |

## Outside Two-Hour Window Diagnostic

| Diagnostic | Rows |
|---|---:|
| event_close_window_captured_but_exact_offer_unmatched | 176 |

| Book | Market | Diagnostic | Rows |
|---|---|---|---:|
| fanduel | batter_total_bases | event_close_window_captured_but_exact_offer_unmatched | 60 |
| draftkings | batter_hits | event_close_window_captured_but_exact_offer_unmatched | 32 |
| fanduel | batter_hits | event_close_window_captured_but_exact_offer_unmatched | 30 |
| fanduel | batter_home_runs | event_close_window_captured_but_exact_offer_unmatched | 30 |
| draftkings | batter_total_bases | event_close_window_captured_but_exact_offer_unmatched | 20 |
| draftkings | pitcher_strikeouts | event_close_window_captured_but_exact_offer_unmatched | 2 |
| fanduel | pitcher_strikeouts | event_close_window_captured_but_exact_offer_unmatched | 2 |

## Most Affected Event / Book / Market

| Event ID | Book | Market | Rows | Failed | Valid | Reasons |
|---|---|---|---:|---:|---:|---|
| 7ad2e88701ead09625333188a5b653b2 | fanduel | batter_total_bases | 60 | 60 | 0.0% | {"close_outside_two_hour_window": 60} |
| 7ad2e88701ead09625333188a5b653b2 | draftkings | batter_hits | 32 | 32 | 0.0% | {"close_outside_two_hour_window": 32} |
| 7ad2e88701ead09625333188a5b653b2 | fanduel | batter_hits | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 7ad2e88701ead09625333188a5b653b2 | fanduel | batter_home_runs | 30 | 30 | 0.0% | {"close_outside_two_hour_window": 30} |
| 7ad2e88701ead09625333188a5b653b2 | draftkings | batter_total_bases | 20 | 20 | 0.0% | {"close_outside_two_hour_window": 20} |
| 7ad2e88701ead09625333188a5b653b2 | draftkings | pitcher_strikeouts | 4 | 4 | 0.0% | {"close_outside_two_hour_window": 2, "no_valid_close_snapshot": 2} |
| 7ad2e88701ead09625333188a5b653b2 | fanduel | pitcher_strikeouts | 2 | 2 | 0.0% | {"close_outside_two_hour_window": 2} |

## Targeted Capture

| Event | Start UTC | T-120 | T-60 | T-20 | Close Times |
|---|---|---|---|---|---:|
| NYM @ PHI | 2026-07-16T23:11:00+00:00 | no | yes | yes | 29 |

## Required Target Misses

| Window | Missed Events |
|---|---:|
| t_minus_120 | 1 |
| t_minus_60 | 0 |
| t_minus_20 | 0 |

| Event | Start UTC | Missing Windows | Close Times |
|---|---|---|---:|
| NYM @ PHI | 2026-07-16T23:11:00+00:00 | t_minus_120 | 29 |
