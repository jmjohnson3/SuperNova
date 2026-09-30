# Workload Repair and Locked-Line Validation

This experiment does not change the frozen release, the registered receiving
trial, historical locks, or betting approval.

## Model Changes

- `WorkloadRepairModel` learns low/normal/high opportunity states separately
  from efficiency. State-conditioned rate heads shrink toward exposure-weighted
  player/position priors. Team-volume and direct player-volume estimates are
  blended and bounded; offered-player lists are never treated as full rosters.
- Efficiency evidence requires positive observed opportunity. A zero-target
  appearance informs workload, not zero player skill. Missing opportunity is
  rejected in training rather than filled with zero.
- `AsymmetricWorkloadResidual` learns different lower and upper error scales
  from player role/history. A subsequent held-out block supplies residual
  quantiles. A learned confidence head replaces the fixed confidence value in
  the uncertainty-only lock replay.
- The v3 experiment does not refit the central model after learning residuals
  and calibration. The prediction model being evaluated is the same one whose
  held-out errors trained those layers.
- A full-curve calibration guard now requires better Brier, non-worsening
  calibration, coverage nearer the nominal 80%, and non-worsening interval
  score on the earlier acceptance block. Better line Brier alone is insufficient.

## Historical Comparison

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_workload_repair `
  --cache src/nfl_pipeline/modeling/models/workload_depth/workload-depth-20260921T100911119214Z/training.joblib `
  --through 2026-09-18 --seasons 2025 2026
```

The three ablations use identical chronological folds and player-games:
existing workload, conditional efficiency, and conditional efficiency plus
asymmetric uncertainty. Each output has a separate test: mean RMSE/bias,
median MAE, and line Brier/calibration/coverage. Comparisons use week-grouped
resampling. Proxy-line success is not real betting evidence.

Artifacts are isolated under `modeling/models/workload_repair/<run_id>`.
No live pointer is changed. The full workload repair and the uncertainty-only
replay are different candidates and must not be conflated.

## Actual Locks

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.yardage_lock_validation `
  --season 2026 --week 3 --artifact <run-directory>/models.joblib
```

This uses saved FanDuel Discord manifests, original research selections,
timestamp-checked paired quotes, archived final-scoring arguments, and actual
grading. The original projection is preserved in the uncertainty-only replay.
No historical inputs are reconstructed when missing. Model training must end
before the lock date. Week 3 informed the repair hypothesis and therefore is
diagnostic, not an untouched acceptance set. Artifact creation after the lock
is explicitly retrospective, never called a prospective capture.

The report separates original fixed research picks from counterfactual EV and
conservative rankings. Rankings never inspect outcomes, and pending picks
still consume their simulated cap. Side changes do not replace original picks.
Integer-line challenger replay is excluded until its push curve is validated.
Raw distribution intervals are not labeled final-CDF intervals: line-specific
live adjustments can still damage coherence and must pass a later full-curve
test before deployment.

## Context Limitations

The default weekly player-stat feed has target share and air-yards data but
does not supply measured routes or first-read targets. Importer health now
records actual source capabilities and nonmissing field counts. Null primary
aliases fall through to supported secondary aliases, and valid zero route
efficiency values are retained. No proxy is relabeled as measured routes.

Missing injury information remains unknown, not healthy. Historical context
is not repaired using reports published after the original lock. A verified
measured-route source and better timestamped availability coverage remain
external data requirements, not something model code can invent.

## Decision

Do not deploy merely because a challenger beats an older challenger. It must
beat the reference on the intended output, retain coverage and calibration,
and survive the full live scoring/selection path on new locked offers. Forecast
deployment and authorization of cash bets remain separate decisions.

## First Completed Experiment

Run `workload-repair-20260928T210154323193Z` evaluated 2025 and early 2026
folds with history ending September 17. Neither full replacement passed.
The first run exposed calibration-induced over-wide intervals; its original
artifacts and OOF results were retained, not rewritten as if the subsequent
full-curve guard had existed then. The guard was separately checked on the
final bundle's pre-test calibration block and rejected the damaging mapping.
Subsequent training runs apply that guard automatically.

On the September 29 settled Week 3 archived-card audit, the smaller
uncertainty-only change gave rushing final Brier 0.2701 versus production
0.2733, still worse than the market's 0.2496. Receiving worsened to 0.2555
from 0.2490. These are diagnostic comparisons, not promotion evidence.
The original fixed receiving selections are now 3-4 after Monday settlement;
they remain separate from reselected counterfactual picks and confirmed bets.

## Guarded Multiweek Rerun

Run `workload-repair-20260929T155054734302Z` completed the guarded rerun
across 24 independent evaluation weeks. The asymmetric full replacement has:

| Stat | Mean RMSE | Reference RMSE | Line Brier | Reference Brier | 80% coverage |
|---|---:|---:|---:|---:|---:|
| Receiving yards | 23.971 | 24.252 | 0.2434 | 0.2308 | 84.9% |
| Rushing yards | 26.140 | 26.669 | 0.2374 | 0.2349 | 87.9% |

Both expected-mean screens pass, but both probability screens fail. The older
workload variant also beats the new full replacement on point RMSE and line
Brier. There is no evidence to deploy the full replacement as a probability
model. These historical reference refits are not a direct prospective test
against the frozen production artifact.

Fold calibration metadata retains the initial line-only validation decision;
the nested `full_curve_guard.enabled` is the effective decision where present.
The guard removed damaging interval transformations but did not, by itself,
make the raw distributions better than the reference.

The historical cache has no verified depth, starter, injury-out or teammate
absence context for these evaluation rows. Those missingness indicators are
honest, but they cannot teach the model the effect of information it never
observed. The refreshed weekly source updated 959 usage rows with available
fields and still supplied zero measured-route or first-read rows.

Production was not replaced. Remaining requirements include stronger workload
and efficiency estimates, coherent final-CDF validation and new prospective
evidence. True routes/first-read data and timestamped historical availability
still require sources that actually supply those fields. Passing a mean-yardage
test does not authorize a change to the probabilities used for betting.
