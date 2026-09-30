# NFL Final Probability Validation

Production unchanged. No new real-money approval.

## Actual Locked Micro

{"rows": 5, "brier": 0.3629503217977323, "log_loss": 0.9268049365046427, "calibration_error": 0.4610844552761112, "probability_bias": 0.4610844552761112, "wins": 1, "losses": 4, "mean_probability": 0.6610844552761111, "unique_player_games": 5, "weeks": 1}

Ledger coverage: {"total_micro_prediction_ids": 25, "scored_micro_prediction_ids": 5, "unscored_prediction_ids": [362, 366, 664, 669, 702, 830, 838, 843, 862, 868, 893, 894, 904, 906, 1000, 1033, 1045, 1215, 1284, 1356]}

## Every Micro Entry Accounted For

{"missing_evaluation_inputs": 14, "void_or_push": 6, "evaluable": 5}

Evaluable but unscored: []

Legacy/void rows remain in nfl_micro_reconciliation_latest.md, not in verified probability metrics.

## Exact Replay Coverage

{"matched": 882, "missing_lock_time_scoring_inputs": 240}

## Calibration Experiment

{
  "nfl-20260918T143342Z|0d2afea95579aa80dc6e9c48b3ce34ce5f9c7a15b4301327745e94ffffdec407": {
    "status": "insufficient_later_week_evidence",
    "weeks": 2,
    "unique_player_games": 202,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  },
  "nfl-20260918T143342Z|44f8fbfa4bd9b3417ee88fe4e9f821a8c0f937493a574b6dae5b94dc97355219": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 14,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  },
  "nfl-20260918T143342Z|e29cbbca864ea8e37e57d7ba178ee30ff3dd371a412726d569d8f89b063c1afd": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 215,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  }
}

## Uncertainty-Only Real Offers

| Cohort | Rows | Production Brier | Challenger Brier | Coverage 80% | Weeks |
|---|---:|---:|---:|---:|---:|
| components-20260919T035631590394Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 125 | 0.2603 | 0.2588 | 72.0% | 1 |
| components-20260920T005350695557Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 132 | 0.2592 | 0.2588 | 71.8% | 1 |
| components-20260920T010253075132Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 144 | 0.2579 | 0.2574 | 71.6% | 1 |
| components-20260922T091540317936Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 145 | 0.2483 | 0.2493 | 72.9% | 1 |

Exact micro-ID uncertainty coverage: {"micro_prediction_ids": 25, "exact_pregame_uncertainty_ids": 0, "missing_ids": [362, 366, 664, 669, 702, 830, 838, 843, 862, 868, 893, 894, 904, 906, 1000, 1033, 1045, 1215, 1284, 1356, 3774, 3959, 3993, 4033, 4079], "later_revisions_used": false}

- Actual micro uses exact ledger prediction IDs, never later revised forecasts.
- Missing historical replay inputs remain missing; stored final probabilities can still be scored.
- Counterfactual top five uses earliest offers from a fixed approved pool, not an exact historical batch/ledger replay.
- Week-grouped validation keeps repeated books/lines from inflating training evidence.
- A training label must be verifiably available before the earliest test lock; earlier game dates alone are insufficient.
- A simulated ledger entry is not evidence that money was wagered.
- Challenger probabilities are not automatically installed in live scoring.
