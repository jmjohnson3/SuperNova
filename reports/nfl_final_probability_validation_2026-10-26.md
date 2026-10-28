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

{"matched": 40}

## Calibration Experiment

{
  "nfl-20260918T143342Z|2c46f8e1c0a840378865a096ef836b08c432b14655d62b671841d62eebb16feb": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 15,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  }
}

## Uncertainty-Only Real Offers

| Cohort | Rows | Production Brier | Challenger Brier | Coverage 80% | Weeks |
|---|---:|---:|---:|---:|---:|
| components-20261015T203006649687Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 9 | 0.2500 | 0.2500 | 80.0% | 1 |

Exact micro-ID uncertainty coverage: {"micro_prediction_ids": 0, "exact_pregame_uncertainty_ids": 0, "missing_ids": [], "later_revisions_used": false}

- Actual micro uses exact ledger prediction IDs, never later revised forecasts.
- Missing historical replay inputs remain missing; stored final probabilities can still be scored.
- Counterfactual top five uses earliest offers from a fixed approved pool, not an exact historical batch/ledger replay.
- Week-grouped validation keeps repeated books/lines from inflating training evidence.
- A training label must be verifiably available before the earliest test lock; earlier game dates alone are insufficient.
- A simulated ledger entry is not evidence that money was wagered.
- Challenger probabilities are not automatically installed in live scoring.
