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

{"matched": 130}

## Calibration Experiment

{
  "nfl-20260918T143342Z|1aca1b617d26af98dbe85949a8d3aac233caf09f15cb0b7054f8f4f232d1a767": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 17,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  },
  "nfl-20260918T143342Z|58fb8dac7a026bf4aa9978e87680b773fcf85f1fe82f681afaaf3047891a4eb6": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 16,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  },
  "nfl-20260918T143342Z|af37343232f12aec46617d918e34a087a80c03acef39e32677957806d492f359": {
    "status": "insufficient_later_week_evidence",
    "weeks": 1,
    "unique_player_games": 17,
    "enabled": false,
    "reason": "Need at least two training weeks before a later-week test; no in-sample calibrator is enabled."
  }
}

## Uncertainty-Only Real Offers

| Cohort | Rows | Production Brier | Challenger Brier | Coverage 80% | Weeks |
|---|---:|---:|---:|---:|---:|
| components-20261013T100318276849Z-receiving_conditional|nfl-20260918T143342Z|receiving_yards | 14 | 0.2483 | 0.2483 | 80.0% | 1 |

Exact micro-ID uncertainty coverage: {"micro_prediction_ids": 0, "exact_pregame_uncertainty_ids": 0, "missing_ids": [], "later_revisions_used": false}

- Actual micro uses exact ledger prediction IDs, never later revised forecasts.
- Missing historical replay inputs remain missing; stored final probabilities can still be scored.
- Counterfactual top five uses earliest offers from a fixed approved pool, not an exact historical batch/ledger replay.
- Week-grouped validation keeps repeated books/lines from inflating training evidence.
- A training label must be verifiably available before the earliest test lock; earlier game dates alone are insufficient.
- A simulated ledger entry is not evidence that money was wagered.
- Challenger probabilities are not automatically installed in live scoring.
