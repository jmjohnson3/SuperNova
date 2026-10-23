# NFL Distribution Policy

Forecast deployment and cash authorization are separate. No cash approval is granted here.

| Market | Unique eligible locks | Curve rows |
|---|---:|---:|
| receiving_yards | 1685 | 4642 |
| passing_yards | 283 | 523 |
| rushing_yards | 727 | 1081 |
| total | 54 | 107 |

Pinned challengers: 6. Installed this run: 0.

## Prospective Decisions

- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|raw_distribution: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|receiving_quantile:d1f896b4284f98b8151c4e6b71747fcfb5b446b0ed7f2b6abc7ec66d490c112d: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|receiving_yards|role_workload:658bf1b0fd3ef4132788c5dcfd7c5e35cc4cbf9a14d6edaee839c85dbb64dafe:combined: needs_three_independent_weeks, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|forecast_repair:2259ac0e8a035747d3ef41b78dc91bfc13cba8244a09a2a7f4153e90affc2487:joint_calibrated: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|passing_yards|stat_component:components-20261010T222543607583Z-qb_tail: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.
- nfl-distribution-policy-v1|nfl-20260918T143342Z|fanduel|rushing_yards|forecast_repair:a426cb68d3dad523560c81c234b61857d0dfc34970b916872e33f56799105c3c:joint_calibrated: needs_three_independent_weeks, needs_60_settled_player_games, final_brier_not_improved, calibration_not_improved, weekly_gain_uncertain, coverage_not_proven
  Betting-selection evidence: selected_pick_sample_missing. No cash approval.

## Evidence Limits

- Historical reconstruction is development evidence, never a prospective policy lock.
- Whole-CDF market shape is an assumption from one true paired anchor, not observed alternate-line market probabilities.
- Missing original game distributions are excluded, never reconstructed from current sigma.
- Forecast deployment does not change stake, ledger, CLV or cash authorization requirements.

Exclusions: {"games": {"later_game_revision": 245, "missing_locked_game_distribution": 655}, "props": {"later_revision_same_decision": 4842, "missing_replay_or_exact_overlay_artifact": 10411, "missing_true_paired_market": 31, "stale_original_quote": 384}}


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

Current fit: fitted_for_prospective_test
Model weight: 0%; calibration enabled: False.
Unique player-games: calibration 172, tuning 159.
This current-time fit is for new locks, not historical acceptance.

### Tuning Diagnostic (Not Acceptance)

| Path | Brier | Calibration error | 80% coverage |
|---|---:|---:|---:|
| production | 0.2502279242369318 | 0.012728655614298683 | - |
| challenger | 0.25022792423693 | 0.010108110540930063 | 0.8364779874213837 |
| exact_market | 0.2502279242369318 | 0.012728655614298683 | - |
- Tuning production: 0 selected; Brier -; exclusions {}
- Tuning unchanged_ranking: 0 selected; Brier -; exclusions {}
- Tuning conservative_ranking: 0 selected; Brier -; exclusions {}
- production: 0 selected; replay exclusions {}
- unchanged_ranking: 0 selected; replay exclusions {}
- conservative_ranking: 0 selected; replay exclusions {}