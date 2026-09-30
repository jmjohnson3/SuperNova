# NFL live trial evidence

Production and the registered FanDuel common-line receiving strategy remain
unchanged. Model deployment and permission to wager are separate decisions.

## Daily cycle

Starting September 24, 2026, the game-aware pregame path stores a matchup-level
forecast manifest alongside each confirmed Discord publication. It contains exact
forecast IDs, book, side, line, price, model version and displayed tier. The
manifest covers all pages of a matchup; it is not a record of an executed wager.

`python -m nfl_pipeline.live_cycle_audit --date YYYY-MM-DD` verifies that the
publication refers to the immutable prop forecasts and that quotes were fresh at
publication. It also checks the receiving checkpoint for missing challenger
captures, valid exact closes, and unresolved final results. Before T-20 an absent
publication is pending; afterward it is actionable. Future close windows do not
contribute zeros to the coverage denominator. Completed windows need 90% valid
exact capture. A completed cycle does not approve betting.

The pregame wrapper invokes this audit after saving its publication receipts.
The settlement path invokes it after the receiving checkpoint. It is excluded
from the short active-close collection path so model evaluation does not occupy
the close-capture window. Earlier dates are marked pre-contract history rather
than pretending the new publication manifest existed then.

The existing receiving-trial follow-up automation checks the cycle and settled
evidence. It does not duplicate successful publications, replace fixed research
picks, promote a model automatically, or authorize wagers.

## Micro accounting

Final probability validation now includes every micro ledger row in
`reports/nfl_micro_reconciliation_latest.md` and its JSON companion. Categories
are evaluable, pending, void/push, missing result, and missing evaluation inputs.
Each entry retains the ledger result and whether execution is simulated.

The September 23 audit accounted for 25 rows: 5 evaluable, 14 settled legacy rows
without immutable forecast inputs, and 6 prior voids. Four voids have missing
original prediction rows. These are historical integrity limitations, not a
recoverable join to a newer prediction. No old forecast or price was invented,
no void was reversed, and no legacy row was admitted into verified training.

## Uncertainty challenger

`nfl_pipeline.modeling.receiving_uncertainty_repair` trains conditional uncertainty
around a conservative receiving reference, then learns asymmetric tail widths by
position and lagged target-volume group. Sparse groups shrink toward the global
fit. The model-fit, scale, residual, tail-fit, tuning and acceptance blocks are
chronological and disjoint; outer later-week folds remain untouched. Whole
player-game rows are used, not duplicated book offers.

Tail changes transform the full support monotonically, preserve mass and median,
and keep over/under probabilities coherent across lines. Validation reports Brier,
calibration, interval coverage/score, mean RMSE and median MAE. Historical proxy
lines are not real offered-line betting proof. No production-consumed pointer is
written. The September 23 candidate failed robust incremental improvement and
remains offline; the pinned receiving trial was not replaced.

## Separate decisions

The checkpoint can recommend a forecast-only component review after multiweek
Brier, calibration and coverage evidence passes. Bankroll ROI/CLV is not required
to identify a more accurate forecast. The separately registered research strategy
still needs credible selected probabilities versus the true paired market,
executable prices, prospective results, reliable exact closes and limited
concentration before cash review. Neither report automatically grants approval.
