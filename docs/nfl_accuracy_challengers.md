# NFL Accuracy Challenger Workflow

## Production Contract

Production is frozen by `src/nfl_pipeline/modeling/models/production_freeze.json`.
The release ID and both artifact hashes must match. Publishing a validated
release fails while that freeze exists. Neither challenger training nor shadow
scoring publishes production models, creates bets, or changes bankroll rules.

The daily runner skips training while frozen. The offline training runner still
builds candidate artifacts without `--publish`, then trains accuracy challengers.
It then evaluates individual accuracy components. The daily job prospectively
scores only component outputs that passed their independent historical tests.
Unfreezing is a separate, deliberate release decision, not an automatic response
to one good slate or a historical pass flag.

## Implemented Paths

1. Live forecasts capture the exact offer, distribution, context row, adjustment
   state, scoring fingerprint, and intermediate/final probabilities. Replay calls
   the live scoring function. Missing historical arguments stay missing; current
   context is never substituted for old lock-time evidence.
2. Conditional uncertainty learns error scale and error-within-tolerance confidence
   from held-out errors using position, workload, role volatility, and historical
   variability. Confidence is not a bet's win probability. Production's current
   pooled distribution remains unchanged until a release decision.
3. Receiving, rushing, and passing challengers combine low/normal/spike workload
   probabilities, state-conditional targets/carries/attempts, and yards per
   opportunity. Output includes an expected mean, median, and 80% interval.
4. Nullable context evidence carries injury/practice/depth observation times,
   expected starter from depth, teammate report coverage, and measured-versus-proxy
   route labels. Unknown injuries and unavailable route data are not measured zeros.
   Historical same-week evidence is used only if captured before the lock.
5. Calibration fits global logit and shrunk position/workload/line/book corrections.
   Unknown books inherit broader corrections. Later inner folds must show improved
   Brier before the calibrator is enabled. Increasing lines cannot increase over
   probabilities within a priced curve.
6. Small game challengers use lagged team scoring, pace, efficiency, and rest.
   Their market comparison is a retrospective benchmark, never an independently
   validated or executable market edge.
7. Expanding outer folds hold out complete NFL weeks across multiple seasons.
   Five separate inner blocks fit the model, uncertainty scale, residual curve,
   calibrator, and calibrator acceptance. Projection rows are unique player-games;
   line rows share player-game weights; uncertainty resamples whole NFL weeks.

## Commands

Run from the repository with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_accuracy_challengers
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_accuracy_components
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.score_accuracy_components --date 2026-09-20 --capture-team-context
# Run normal player prediction between team context capture and component scoring.
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.score_accuracy_components --date 2026-09-20
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.live_scoring_replay
.\.venv\Scripts\python.exe -m pytest src/nfl_pipeline -q
```

Training writes immutable run directories below `modeling/models/challengers`.
The latest pointer contains a checksum; artifacts must load in a fresh process.
Daily shadow scoring batches by stat and writes prospective files linked to the
exact immutable production forecast ID. It refuses same-date/future training
outcomes. Post-game replay compares those records against production and groups
results by challenger run, production release, and stat. No automatic promotion
is performed, regardless of historical scores.

## Independent Output Contracts

- Expected mean: RMSE/squared-error improvement and no worse absolute bias.
- Median: MAE/absolute-error improvement. A failed median gate does not reject an
  otherwise improved probability distribution.
- Probabilities: Brier improvement, no worse calibration error, and 80% interval
  coverage within three percentage points of nominal. Gains must retain a
  positive lower 95% bound when resampling entire NFL weeks. Duplicate lines
  share one player-game outcome rather than multiplying the evidence.
- None of these gates changes micro/bankroll eligibility.

The receiving uncertainty component retains the exact locked production point
forecast. It replaces only the residual distribution in a copy of the captured
scoring inputs, then replays the complete live adjustment path. Stored final
probabilities, not raw model probabilities, are evaluated on actual offered lines.
Missing replay inputs or a changed scoring fingerprint make that comparison
unavailable; the scorer never substitutes today's settings into an old lock.

The QB tail component fits asymmetric workload-specific tail corrections before
a separate later calibration-acceptance block. Better coverage alone is not
enough: interval score and Brier must also improve. Line calibration is applied
to a normalized monotone CDF in both evaluation and prospective replay. A failed
latest calibration block vetoes the probability output even if pooled historical
folds look good. It does not veto an independently accepted mean or median;
those can be compared prospectively without pricing any betting line.

Team-allocation models first predict pass/rush budgets, then normalize player
shares across the full pregame roster, with reserved opportunity for unknown
newcomers. Offered players are never renormalized as though they were the whole
team. Training pools use prior-game membership, not the upcoming game's eventual
participants; same-date results cannot enter those pools. Efficiency is a separate
shrunk player-rate head. Insufficient history and unobserved injuries remain
unknown. These models remain offline unless they pass the output-specific tests.

When a team component is accepted, the daily job captures its context before
player prediction. Reusing an earlier locked forecast requires a context snapshot
captured before that forecast's original cutoff. Later context is never backdated.
The old `score_accuracy_shadow` command remains available for broad offline
research but is no longer the default daily comparison path.

## Reports And Limits

- `reports/nfl_accuracy_challengers_latest.md` and `.json`: outer-fold MAE, RMSE,
  bias, interval coverage, Brier, calibration, opportunity accuracy, grouped
  uncertainty, position/season splits, and historical pass status.
- `reports/nfl_live_scoring_replay_latest.md` and `.json`: exact replay coverage,
  stored probability stages, paired final-versus-raw Brier changes, locked stat
  interval coverage, and prospective challenger-versus-production cohorts.
- `reports/nfl_accuracy_components_latest.md` and `.json`: independent mean,
  median, and probability gates, latest QB calibration acceptance, and which
  outputs are enabled for prospective comparison. Component artifacts live under
  `modeling/models/accuracy_components`; they never replace the active release.
- Both nightly training and post-game grading refresh replay. Neither report
  requires reconstructing past forecasts from the current model.

Historical player Brier uses explicitly unpriced proxy lines. It is not sportsbook
ROI/CLV proof. Reference-recipe refits are not the deployed artifact's historical
predictions. Older forecasts without full replay arguments cannot establish exact
reproducibility. True routes/first reads still require a real data source. The
already-examined 2026 results are not an untouched final test. Final adoption needs
new prospective outcomes under these versioned contracts.
