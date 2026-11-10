# NFL Distribution Policy

Forecast deployment and cash authorization are separate. No cash approval is granted here.

| Market | Unique eligible locks | Curve rows |
|---|---:|---:|
| receiving_yards | 2390 | 8167 |
| passing_yards | 410 | 904 |
| rushing_yards | 1051 | 1729 |
| total | 129 | 257 |

Pinned challengers: 12. Installed this run: 0.

## Prospective Decisions

- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|raw_distribution: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated: calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|stat_component:components-20261010T222543607583Z-qb_tail: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|rushing_yards|forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated: weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|stat_component:components-20261015T203006649687Z-team_receiving_yards: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|stat_component:receiving-oe-v2-20261012T234225243219Z: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|total|raw_distribution: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, weekly_gain_uncertain, coverage_not_proven, selected_probabilities_worse
  Betting-selection evidence: selected_probabilities_worse. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|total|forecast_repair:cc8914500edbe8abe243f67561bbdc3ab446e7e7f2c6518da2577b65f313de96:small_total: needs_three_independent_weeks, needs_60_settled_player_games, weekly_gain_uncertain, coverage_not_proven
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|raw_distribution: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|rushing_yards|raw_distribution: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.

## Evidence Limits

- Historical reconstruction is development evidence, never a prospective policy lock.
- Whole-CDF market shape is an assumption from one true paired anchor, not observed alternate-line market probabilities.
- Missing original game distributions are excluded, never reconstructed from current sigma.
- Forecast deployment does not change stake, ledger, CLV or cash authorization requirements.

Exclusions: {"games": {"later_game_revision": 447, "missing_locked_game_distribution": 655}, "props": {"later_revision_same_decision": 7828, "market_anchor_outside_curve_support": 1, "missing_replay_or_exact_overlay_artifact": 15767, "missing_true_paired_market": 36, "stale_original_quote": 384}}


## Workload Probability Path

receiving_yards|nfl-20260918T143342Z|source:role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined|fanduel

Historical development only; not untouched acceptance.

| Probability path | Settled rows | Brier | Calibration error |
|---|---:|---:|---:|
| Production | 727 | 0.2499356025301178 | 0.027264739115042964 |
| Raw workload | 727 | 0.2608979510137504 | 0.08395497173652608 |
| Calibrated workload | 727 | 0.2608979510137504 | 0.08395497173652626 |
| Learned blend | 727 | 0.2513222060896171 | 0.032516065264908085 |
| Exact no-vig offer | 727 | 0.2499356025301178 | 0.027264739115042964 |

Current fit: fitted_for_prospective_test
Model weight: 0%; calibration enabled: False.
Unique player-games: calibration 647, tuning 148.
This current-time fit is for new locks, not historical acceptance.

### Tuning Diagnostic (Not Acceptance)

| Path | Brier | Calibration error | 80% coverage |
|---|---:|---:|---:|
| production | 0.2500578829684719 | 0.00749303434985977 | - |
| challenger | 0.2501290411728038 | 0.04922942042526125 | 0.8445945945945946 |
| exact_market | 0.2500578829684719 | 0.00749303434985977 | - |
- Tuning production: 0 selected; Brier -; exclusions {}
- Tuning unchanged_ranking: 0 selected; Brier -; exclusions {}
- Tuning conservative_ranking: 0 selected; Brier -; exclusions {}
- production: 0 selected; replay exclusions {}
- unchanged_ranking: 0 selected; replay exclusions {}
- conservative_ranking: 0 selected; replay exclusions {}