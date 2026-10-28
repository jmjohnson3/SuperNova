# NFL Final Probability Validation

Production unchanged. No new real-money approval.

## Actual Locked Micro

{"rows": 0}

Ledger coverage: {"total_micro_prediction_ids": 0, "scored_micro_prediction_ids": 0, "unscored_prediction_ids": []}

## Every Micro Entry Accounted For

{}

Evaluable but unscored: []

Legacy/void rows remain in nfl_micro_reconciliation_latest.md, not in verified probability metrics.

## Exact Replay Coverage

{"matched": 136}

## Calibration Experiment

{
  "nfl-20260918T143342Z|93ed5e68ecd622b1dcb00e58bd35bd2486c39885e651fc6ae59af70480a0af64": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 16,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  },
  "nfl-20260918T143342Z|97473e84f5541ccb580322c19d616653e1b71d0bec9ab4a5a152668b99b30a7a": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 18,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  },
  "nfl-20260918T143342Z|ab77cb47b2deb5138704d8c150a9a477ed5929b5f4c14a0ece7c030e88c2b7ca": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 18,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  }
}

## Uncertainty-Only Real Offers

| Cohort | Rows | Production Brier | Challenger Brier | Coverage 80% | Weeks |
|---|---:|---:|---:|---:|---:|
| components-20261015T203006649687Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 14 | 0.2519 | 0.2519 | 73.3% | 1 |

Exact micro-ID uncertainty coverage: {"micro_prediction_ids": 0, "exact_pregame_uncertainty_ids": 0, "missing_ids": [], "later_revisions_used": false}

- Actual micro uses exact ledger prediction IDs, never later revised forecasts.
- Missing historical replay inputs remain missing; stored final probabilities can still be scored.
- Counterfactual top five uses earliest offers from a fixed approved pool, not an exact historical batch/ledger replay.
- Week-grouped validation keeps repeated books/lines from inflating training evidence.
- A training label must be verifiably available before the earliest test lock; earlier game dates alone are insufficient.
- A simulated ledger entry is not evidence that money was wagered.
- Challenger probabilities are not automatically installed in live scoring.
