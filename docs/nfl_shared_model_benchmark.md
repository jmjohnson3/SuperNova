# Shared NFL Yardage Benchmark

Production remains frozen. This experiment does not alter bets, thresholds,
Discord rankings, ledger rows, or the production release.

## Training and Selection

`python -m nfl_pipeline.modeling.model_benchmark --cache <player-game-cache>`

The cache is the prepared `training.joblib` from the workload/depth experiment.
It must contain one row per player-game. Targets are passing, receiving and
rushing yards; TDs and game models are outside this experiment.

Families: rolling average, exposure-shrunk rate baseline, conservative boosted
reference, direct boosted mean, reference with conditional uncertainty,
opportunity/efficiency mixture, workload/depth mixture, and an equal-weight CDF
ensemble. Every family uses identical player-game rows and chronological splits.

Outer tests expand through 2024, 2025 and available 2026 weeks, four weeks at a
time. In benchmark v2, disjoint earlier blocks fit provisional models,
uncertainty scale, residual distribution, calibrator, calibration strength, and
output selector. Blocks expand backward by whole weeks until stat-specific
minimum player-game counts, week counts and regular-season counts are satisfied.
Offers cannot inflate sample counts. Postseason rows are retained and identified;
a two-QB playoff week cannot act as a complete calibration block.

After inner selection, all core families are refitted on the complete pre-test
history. Calibrators and uncertainty retain their held-out estimates. Every outer
test evaluates this full refit procedure, including any calibration-transfer
damage, not the earlier provisional model. The production release is never
refitted here. The report records both block dates and the core model-fit cutoff.

Output selection uses only the inner gate, never outer test results:

- Expected yards: distribution-mean RMSE, with bias as a tie breaker.
- Typical outcome: distribution-median MAE.
- Probability: Brier, then calibration error, among curves with inner nominal
  80% coverage between 75% and 85%. If none pass, retain reference as an explicitly
  unapproved comparison fallback. An inner pass does not guarantee outer coverage.

The `selected_*` rows evaluate the complete preselected policy on later weeks.
Per-family leaderboards are descriptive, not hindsight promotion decisions.
Confidence intervals resample NFL weeks. Proxy lines receive inverse player-game
weights; multiple lines do not become independent observations.

Fixed ensemble and centrally calibrated fixed ensemble are compared directly
with the switching policy on matched rows. The mixing weights remain equal and
do not depend on the held-out week. For passing yards, a separate quantile-head
challenger predicts asymmetric 10th/50th/90th-percentile spacing from lagged
attempts, efficiency, role variability and pregame context. Its tail transform
preserves the base median and probability mass. Blend strength is learned only
in an earlier calibration-fit block using Brier, interval score and coverage.
The unchanged ensemble stays in the comparison even when the tail repair loses.

The reference is an earlier-data refit of the conservative architecture. It is
not the current production artifact applied retroactively. Historical feature
data are corrected final histories with date-lagged inputs, not complete archived
lock-time snapshots. Same-week injury/depth data remain incomplete and unknown.

## Tail-Preserving Calibration

The central isotonic map has fixed endpoints at probabilities 0.1 and 0.9 and
is exactly identity outside that interval. Strength is chosen on a later block,
then accepted on a separate still-later block only when Brier improves and
calibration error does not worsen. Rejection means identity, not dropped rows.

The map is applied to survival mass for the entire distribution. It stays
monotone, supports arbitrary lines, and preserves 10th/90th percentiles and tail
probabilities. It does not widen tails to repair poor central calibration, and
cannot repair an already-miscalibrated raw tail. Live postprocessing can still
change a curve; the offered-line evaluation measures that separately.

## Real Offers and Original Micro Selections

`python -m nfl_pipeline.modeling.benchmark_offers --date YYYY-MM-DD`

The daily pipeline runs this after micro locking. It writes immutable shadow
files only. It scores the selected output policy, a point-preserving
conditional-uncertainty variant, both fixed ensembles and both QB tail variants
through the captured complete production scoring path, retaining the original
context, line, book, price and selected side. Previous v1 shadow cohorts remain
readable and are never pooled with a v2 model cohort.

The artifact must predate the original forecast cutoff, and scoring must finish
before kickoff. Frozen historical workload inputs are included with the artifact;
no later database history is substituted. Availability and depth retain each
individual offer's original context. Older locks are excluded, never backdated.

`python -m nfl_pipeline.modeling.benchmark_offers --report`

The close/grade pipeline refreshes this report. It separately scores real offers
and exact original micro ledger IDs, with raw and final Brier, calibration,
yardage error, interval coverage and week-clustered differences. A hypothetical
side flip never changes the original evaluation target. Pending games and pushes
are not losses. A simulated micro recommendation is not proof a wager was placed.
Matched challenger comparisons use identical original forecast IDs. The exact
micro comparison requires verified micro identity on both sides of the pair.

Reports:

- `reports/nfl_model_benchmark_latest.md` and `.json`
- `reports/nfl_benchmark_offers_latest.md` and `.json`

These do not auto-promote models. Historical improvement must survive the full
live path on actual future lines and selected micro picks first.
