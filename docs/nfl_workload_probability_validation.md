# NFL Workload and Final Probability Validation

## Production Boundary

These experiments do not replace the frozen production release, change staking,
relax betting gates, or publish Discord picks. Challenger artifacts live under
`src/nfl_pipeline/modeling/models/workload_depth/`. A historical improvement is
not proof that a particular offered bet has an edge.

## Final Probabilities

`python -m nfl_pipeline.modeling.final_probability_validation`

The evaluator compares raw, heuristic, post-exact-line, and final probabilities
on the same reproducible rows. It separately scores the actual micro ledger
using the original prediction IDs, with matching book, side, line, price, model
version, and pregame lock timing. It never substitutes later revised forecasts.
Missing old capture inputs and excluded ledger IDs remain explicit.

The final-stage calibration experiment uses regularized probability logits and
stat, side, position, workload, line, book, and projection-gap groups. Duplicate
offers share a player-game weight. Training precedes each test week, with games
purged across the boundary. Candidate calibrators need later-week improvement
in all-row Brier, calibration, and selected-pick performance, plus a positive
week-clustered Brier interval. Insufficient history leaves calibration disabled.
Training labels must also have been observed before the earliest test lock.
Immutable final-label observations prevent subsequent routine grading refreshes
from erasing that provenance; changed actual values need fresh provenance.

The counterfactual top-five comparison uses a fixed previously approved pool.
It is not an exact replay of historical batches, caps, and accumulated ledger
state. Actual locked micro performance is a separate output. Simulated ledger
entries are not confirmed wagers.

## Workload and Receiving Depth

`python -m nfl_pipeline.modeling.train_workload_depth --through YYYY-MM-DD`

This trains receiving and rushing challengers on unique player-game rows:

- Independent team pass-attempt and rushing-attempt heads.
- Low, normal, and high target/carry workload probabilities and conditional shares.
- Exposure-weighted player efficiency priors with shrinkage toward position evidence.
- Separate receiving target-depth, catch-rate, completed-air-yards-per-target,
  and yards-after-catch-per-reception heads.
- A separate rushing efficiency head, without receiving feature noise.
- Chronologically fitted uncertainty and held-out calibration.

History is read before any results from that date are ingested into features.
Zero-target games affect opportunity, not yards-per-target skill. Missing injury
and depth observations remain unknown. An explicitly verified out designation
removes opportunity; missing information does not.

Player opportunity is bounded by predicted team volume. The model does **not**
renormalize over offered players or the game's actual participants. That would
make a forecast depend on offer coverage or introduce participation hindsight.
Complete-roster allocation needs separately verified pregame roster snapshots.

The trainer evaluates expanding chronological folds against a refitted
conservative reference and a rolling baseline. Expected means use RMSE/bias;
medians use MAE; probabilities use Brier/calibration and the corresponding
calibrated distribution's interval coverage. Historical lines are explicitly
unpriced proxy lines, not reconstructed offered prices.

Raw corrected historical logs are retrospective training evidence. Their
date-lagged features are not claimed to be immutable historical lock snapshots.
Historical verified injury/depth coverage is reported, not invented.

An existing generated `training.joblib` can be passed with `--cache` to reuse the
expensive SQL/history load for a repeatable experiment. `--through` still bounds
the evaluated and trained dates. No challenger is automatically activated.

## Exact Micro Uncertainty Test

`python -m nfl_pipeline.modeling.score_locked_micro_uncertainty --date YYYY-MM-DD`

The daily runner invokes this after locking the micro ledger. It uses the
existing accepted uncertainty-only component, preserves the point forecast,
and replays the full captured scoring adjustments at the exact locked offer.
Training must predate the game, and scoring must finish before kickoff.
Missing captured inputs are exclusions, never an invitation to use today's
context. The resulting cohort is separate from general offer testing.

The close/grading runner refreshes final-probability validation afterward. The
report remains noncritical to live predictions and cannot promote models or
bets. No additional Windows scheduled task is required.

## Reports and Tests

- `reports/nfl_workload_depth_latest.json` and `.md`
- `reports/nfl_final_probability_validation_latest.json` and `.md`
- `reports/nfl_live_scoring_replay_latest.json`

Run `python -m pytest src/nfl_pipeline -q`.

Tests cover same-day leakage, missing availability, exposure weighting,
separate feature inputs, workload states, candidate-set invariance, chronological
calibration, result-independent ranking, exact ledger identity, and pregame-only
uncertainty capture.
