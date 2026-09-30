# NFL Prop Pricing And Lock Diagnostics

## Pricing Contract

New prop locks carry `probability_basis=win_given_no_push`. `probability`
is the selected side's probability conditional on settlement as win or loss.
This is also the basis of two-sided no-vig prices and the binary calibration
training labels, which exclude pushes.

`win_probability`, `loss_probability`, and `push_probability` are unconditional
and sum to one. `over_probability` and `under_probability` are unconditional;
their sum is one minus the push probability. `conditional_over_probability`
retains the binary over probability explicitly.

At integer lines, count distributions retain exact point mass; yardage curves
are discretized at half-unit boundaries. At half lines the push mass is zero.
TD any-event heads rescale the positive Poisson tail to preserve their any-TD
probability without producing an inconsistent 1+ tail. The higher-TD tail is
still a model assumption, not separately validated betting proof.

EV per unit stake is `P(win) * net_payout - P(loss)`. Pushes return the stake.
Minimum American price is the first whole-number price with strictly positive
EV (tolerance 1e-12), not a rounded fair price. For example, a 69% non-push win
probability requires -222 or better, not -223. Exact fair prices do not pass.
Discord labels integer-line probabilities as conditional on no push and shows
the push probability separately.

## Diagnostics

`python -m nfl_pipeline.modeling.final_probability_validation` now also writes:

- `reports/nfl_selected_pick_diagnostic_latest.{json,md}`: all eligible offers,
  exact ledger micro picks, and eligible offers on matching release/dates.
  Breakdowns cover stat/side, line range, workload, confidence, position,
  model-market disagreement, and release. Stage comparisons require exact
  replay; the no-vig comparison uses identical rows with market evidence.
- `reports/nfl_workload_uncertainty_diagnostic_latest.{json,md}`: earliest
  settled player-game locks with below/inside/above interval outcomes, saved
  opportunity feature provenance, and an explicit opportunity/efficiency
  accounting decomposition. This is not causal attribution or a newly inferred
  historical efficiency-head prediction. Missing workload and injury inputs
  remain unknown.
- `reports/nfl_probability_consistency_latest.{json,md}`: final unconditional
  over/under tails compared across lines only within identical lock context,
  release, player, game, stat, and book. An empty comparable set does not pass
  the audit. Independent final calibrators can create reversals; those are
  reported, not silently smoothed or rewritten.

The existing scheduled final-probability step produces these diagnostics.
They do not retrain or deploy a model, select additional bets, send Discord,
or change betting approval requirements. Production model artifacts remain
frozen; pricing code has a new captured fingerprint.

## Historical Integrity

Prior scoring sources were archived before the pricing change. Old locks are
replayed with their original code. Missing historical inputs are not filled
using current models, and old push/probability semantics are not reinterpreted
as though the fix existed then. New traces also retain the context blend
before probability caps; detailed comparisons only use rows that captured it.

The test suite covers integer and half lines, EV/minimum-price agreement,
invalid prices, zero-EV side selection, TD any-event coherence, raw curve
monotonicity, final-calibration reversal detection, ledger-ID population
matching, duplicate weighting, missingness, and workload decomposition.
