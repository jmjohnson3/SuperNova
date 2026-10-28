# NFL Distribution Policy

Forecast deployment and cash authorization are separate. No cash approval is granted here.

| Market | Unique eligible locks | Curve rows |
|---|---:|---:|
| receiving_yards | 1902 | 5727 |
| passing_yards | 323 | 643 |
| rushing_yards | 831 | 1289 |
| total | 79 | 157 |

Pinned challengers: 10. Installed this run: 0.

## Prospective Decisions

- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|raw_distribution: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined: final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated: needs_three_independent_weeks, needs_60_settled_player_games, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|stat_component:components-20261010T222543607583Z-qb_tail: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|rushing_yards|forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated: needs_three_independent_weeks, needs_60_settled_player_games, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|stat_component:components-20261015T203006649687Z-team_receiving_yards: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|stat_component:receiving-oe-v2-20261012T234225243219Z: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|total|raw_distribution: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|total|forecast_repair:cc8914500edbe8abe243f67561bbdc3ab446e7e7f2c6518da2577b65f313de96:small_total: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.

## Evidence Limits

- Historical reconstruction is development evidence, never a prospective policy lock.
- Whole-CDF market shape is an assumption from one true paired anchor, not observed alternate-line market probabilities.
- Missing original game distributions are excluded, never reconstructed from current sigma.
- Forecast deployment does not change stake, ledger, CLV or cash authorization requirements.

Exclusions: {"games": {"later_game_revision": 323, "missing_locked_game_distribution": 655}, "props": {"later_revision_same_decision": 6021, "missing_replay_or_exact_overlay_artifact": 12336, "missing_true_paired_market": 35, "stale_original_quote": 384}}


## Workload Probability Path

receiving_yards|nfl-20260918T143342Z|source:role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined|fanduel

Historical development only; not untouched acceptance.

| Probability path | Settled rows | Brier | Calibration error |
|---|---:|---:|---:|
| Production | 254 | 0.24952690778600511 | 0.04866893158084371 |
| Raw workload | 254 | 0.2422634546756116 | 0.05378845225047834 |
| Calibrated workload | 254 | 0.2422634546756116 | 0.05378845225047838 |
| Learned blend | 254 | 0.24954353083027223 | 0.04865243588313774 |
| Exact no-vig offer | 254 | 0.24952690778600511 | 0.04866893158084371 |

Current fit: fitted_for_prospective_test
Model weight: 50%; calibration enabled: False.
Unique player-games: calibration 331, tuning 156.
This current-time fit is for new locks, not historical acceptance.

### Tuning Diagnostic (Not Acceptance)

| Path | Brier | Calibration error | 80% coverage |
|---|---:|---:|---:|
| production | 0.24952690778600511 | 0.04866893158084371 | - |
| challenger | 0.2407586725506212 | 0.06336253282538651 | 0.8333333333333334 |
| exact_market | 0.24952690778600511 | 0.04866893158084371 | - |
- Tuning production: 0 selected; Brier -; exclusions {}
- Tuning unchanged_ranking: 0 selected; Brier -; exclusions {}
- Tuning conservative_ranking: 0 selected; Brier -; exclusions {}
- production: 0 selected; replay exclusions {}
- unchanged_ranking: 0 selected; replay exclusions {}
- conservative_ranking: 0 selected; replay exclusions {}