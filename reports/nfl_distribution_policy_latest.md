# NFL Distribution Policy

Forecast deployment and cash authorization are separate. No cash approval is granted here.

| Market | Unique eligible locks | Curve rows |
|---|---:|---:|
| receiving_yards | 1328 | 2870 |
| passing_yards | 218 | 328 |
| rushing_yards | 566 | 759 |
| total | 30 | 59 |

Pinned challengers: 3. Installed this run: 0.

## Prospective Decisions

- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|raw_distribution: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.

## Evidence Limits

- Historical reconstruction is development evidence, never a prospective policy lock.
- Whole-CDF market shape is an assumption from one true paired anchor, not observed alternate-line market probabilities.
- Missing original game distributions are excluded, never reconstructed from current sigma.
- Forecast deployment does not change stake, ledger, CLV or cash authorization requirements.

Exclusions: {"games": {"later_game_revision": 137, "missing_locked_game_distribution": 655}, "props": {"later_revision_same_decision": 3665, "missing_replay_or_exact_overlay_artifact": 8124, "missing_true_paired_market": 31, "stale_original_quote": 384}}


## Workload Probability Path

receiving_yards|nfl-20260918T143342Z|source:role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined|fanduel

Historical development only; not untouched acceptance.

| Probability path | Settled rows | Brier | Calibration error |
|---|---:|---:|---:|
| Production | 0 | - | - |
| Raw workload | 0 | - | - |
| Calibrated workload | 0 | - | - |
| Learned blend | 0 | - | - |
| Exact no-vig offer | 0 | - | - |

Current fit: needs_separate_fit_and_tune_weeks