# Receiving Coherent Calibration And Downside Experiment

This is an offline challenger, not an automatic production or wagering switch.
The existing FanDuel common-line trial, release, and selections stay pinned.

## Models

`receiving_coherent_repair` fits a conservative reference and player-specific
uncertainty on chronological model/scale/residual blocks. A separate later block
trains a three-class downside head: ordinary outcome, below-range outcome with
less than half the lagged target volume, or below-range outcome without that
large a volume drop. The latter is an efficiency residual, not proven causality.

Only receivers with a known lagged target average of at least four receive the
downside mixture. Missing roles remain unknown; low-role distributions are not
globally widened. Separate low-volume and efficiency-tail supports shrink toward
the shared downside sample. The candidate blends 25% of this conditional mixture
with the reference, rather than replacing it entirely.

Tail-preserving isotonic calibration maps the survival function once for every
possible line. Model fit, downside fit, calibration fit, tuning, selection, and
outer test periods are chronological and game-disjoint. The selected policy must
improve later-fold Brier and calibration without worsening 80% coverage error or
interval score. Week-clustered uncertainty determines whether the improvement is
robust enough to advance. The script never changes a production pointer.

## Full Scoring And Selection

`receiving_selection_experiment` replays genuine archived scoring inputs, including
market/context adjustment, caps, exact-line overlay when reproducible, and recent
calibration. After those adjustments it maps all offered lines in the same
player/book/context into one monotone distribution. It preserves the 10/90
survival anchors and never invents prices for unobserved lines. Integer lines
remain excluded from this receiving experiment until separate push-curve
validation; production push handling is unchanged.

The original and challenger are compared on the original side, line, and price.
Stage metrics, intervals, fixed current selections, conservative selections, and
exact ledger micro IDs are separate. Missing legacy inputs are exclusions. The
new artifact was created after historical locks, so this analysis is retrospective
and receives no prospective credit.

The conservative ranking policy subtracts a fixed model-market disagreement
penalty from EV, with no tuning on outcomes. It uses the same saved pre-cap pool,
processes original batches in time order, and allows at most five selections per
release/scoring cohort/day with one per player-game. Pending outcomes and pushes
remain eligible for selection; they cannot silently turn into losses. Each
cohort is an experiment, not permission for another daily cash budget.

## Commands

```powershell
.venv\Scripts\python.exe -m nfl_pipeline.modeling.receiving_coherent_repair --cache <player-game-cache>
.venv\Scripts\python.exe -m nfl_pipeline.modeling.receiving_selection_experiment --model <explicit-run-model.joblib>
.venv\Scripts\python.exe -m nfl_pipeline.modeling.final_probability_validation
```

Reports:

- `reports/nfl_receiving_coherent_repair_latest.md` and JSON
- `reports/nfl_receiving_selection_experiment_latest.md` and JSON
- Final-probability reports now separate scoring fingerprints, including
  `legacy_unknown`, and train calibration experiments separately by fingerprint.

## September 24 Decision

The chronological proxy-line test spans 46 independent weeks. The downside
candidate improves interval coverage/calibration, but the week-grouped Brier
gain interval crosses zero. It is not accepted for deployment. Real-line replay
has only one settled week and no reproducible exact historical micro inputs.
Better-looking retrospective selections do not override these limitations.

Tonight's ATL/GB cycle remains with the already scheduled pregame/close jobs.
The existing follow-up checks fresh quote/capture/publication, then exact closes
and settlement. Future windows remain pending; 90% valid-close coverage is assessed
after the applicable window completes. Forecast deployment and cash approval
remain independent decisions, and neither is authorized by this experiment.
