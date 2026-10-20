# Target-volume repair decision

## Decision

The target-only challenger `target-volume-20260930T021306134180Z` is rejected.
Production release `nfl-20260918T143342Z`, live scoring, the fixed FanDuel
research selections, and cash approval rules were not changed.

This is an implemented and trained model experiment, not a claim that forecast
accuracy improved in production. Slightly better yardage error did not justify
worse probability calibration. No additional correction was stacked on it.

## Timestamped training inputs

`modeling/role_context_training.py` joins actual observation history to unique
player-games. It uses the earliest original pregame lock and its feature cutoff,
stable player ID, team, season, and compatible game phase. Both observation time
and any source snapshot time must precede the cutoff. Injury/practice reports
must match the game week. Historical locks are never recreated or rewritten.

The September 29 refresh contains 31,053 player-games, including 787 from 2026:

| Evidence | Player-games |
|---|---:|
| Verified original pregame lock | 561 |
| Usable pre-lock context | 405 |
| Depth rank / depth-based expected starter | 405 |
| Depth movement | 404 |
| Practice status | 64 |
| Explicit player injury designation | 0 |
| Observed teammate absence status | 26 |
| Observed teammate limitation information | 340 |

Depth-based expected starter is not confirmed participation. Teammate counts are
partial observed reports, not a complete healthy roster. Missing player injury
designation remains unknown. Route proxies retain their proxy names; this join
does not create measured route or first-read data.

The older training cache had 156 verified locks but zero matching role snapshots
observed early enough. Later observations were excluded. The model drops empty
or constant role features rather than claiming to learn unavailable context.
The new cache is isolated under `models/target_role_context`, not installed as a
production model or substituted into old forecasts.

## Controlled experiment

`modeling/target_volume.py` changes only the target count and low/normal/high
target mixture. The conservative receiving-efficiency recipe and its held-out
residual evidence are shared by every target variant. Small boosted heads use
an explicit lagged-feature allowlist and a chronologically selected shrinkage
weight. No offer duplication is used for player-game model fitting.

Identical expanding-week folds compare a recent-target control, a longer-role
baseline, a recent-share baseline, and the challenger. Model fitting, residual
estimation, and shrinkage tuning use successive earlier periods. Week 3 of 2026
is excluded from the historical acceptance screen because it informed the repair.

| Variant | Target RMSE | Yardage RMSE | Proxy-line Brier | Calibration error |
|---|---:|---:|---:|---:|
| Target control | 2.282 | 24.300 | 0.2272 | 1.46% |
| Longer-role baseline | 2.254 | 24.205 | 0.2267 | 2.96% |
| Recent-share baseline | 2.335 | 24.500 | 0.2299 | 4.40% |
| Target challenger | 2.252 | 24.140 | 0.2287 | 4.61% |

The reports also include MAE, bias, lower/upper interval misses, interval scores,
and week-clustered uncertainty. Proxy lines are not real betting-edge evidence.

## Complete offered-line replay

`modeling/target_volume_validation.py` verifies the original scoring replay before
changing anything. It retains the original quote, line, side, confidence,
calibration, market/CLV adjustments, and ranking rules. Its target-only control
and challenger use the same saved projection-per-target efficiency. The original
fixed research IDs are evaluated separately from counterfactual rankings.

For the current scoring cohort in the Week 3 development replay:

| Pool | Settled offers | Production Brier | Challenger Brier | FanDuel no-vig Brier |
|---|---:|---:|---:|---:|
| Archived eligible offers | 565 | 0.2511 | 0.2489 | 0.2498 |
| Exact fixed research selections | 5 | 0.2616 | 0.2599 | 0.2500 |

Repeated offers are weighted by player-game; they are not independent samples.
The older scoring fingerprint is reported separately. This is the archived pool,
not a reconstructed full book universe. Integer lines are explicitly excluded
until push-aware replacement curves are validated. Missing legacy inputs are
excluded and counted, not reconstructed.

Raw mixture intervals are assessed separately. Per-line market adjustments do
not define a certified whole-distribution CDF, so the report cannot claim final
distribution coverage or grant deployment approval. Week 3 remains diagnostic.
A rejected artifact cannot start prospective captures. Any future passing
artifact still needs untouched, same-offer prospective evaluation after the
development cutoff and a separate deployment decision. Cash approval is separate.

## Operational maintenance

Cash readiness now performs expensive historical reads before acquiring its
policy writer lock. The serialized checkpoint/write section uses bounded
try-lock retries, rechecks registration, and avoids overwriting a newer same-day
report. Exhausted contention fails explicitly; it never becomes successful review
or cash approval. Existing checkpoints and sticky pauses are preserved.

Validated with a real two-connection advisory-lock contention test and a complete
September 28 cash-readiness rerun. The latter remained `research`, with two settled
post-policy selections, two pending, and no confirmed wagers. This is operational
reliability work, not prediction improvement.

## Commands

Run from the repository root using its virtual environment:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.role_context_training --through 2026-09-29
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_target_volume --cache src/nfl_pipeline/modeling/models/workload_depth/workload-depth-20260921T100911119214Z/training.joblib --through 2026-09-18 --seasons 2025 2026
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.target_volume_validation --artifact src/nfl_pipeline/modeling/models/target_volume/target-volume-20260930T021306134180Z/model.joblib --season 2026 --week 3
.\.venv\Scripts\python.exe -m nfl_pipeline.cash_readiness --date 2026-09-28
.\.venv\Scripts\python.exe -m pytest src/nfl_pipeline -q --disable-warnings
```

Generated reports:
- `reports/nfl_target_role_context_latest.md`
- `reports/nfl_target_volume_latest.md`
- `reports/nfl_target_volume_offers_2026_week3.md`

Validation: 441 passed, one skipped. Existing pandas fragmentation warnings remain.

## September 30: Independent output decisions

The follow-up experiment is `target-components-20260930T065356571923Z`.
It no longer requires a component to improve yardage and over/under pricing
together. Target volume, expected yards, typical yards (median), and probability
have separate historical screens and separate deployment status.

The factorial comparison holds efficiency fixed:
- `reference`: conservative forecast and existing pooled uncertainty recipe.
- `targets_only`: independently tuned target mean, unchanged uncertainty shape.
- `uncertainty_only`: new workload/efficiency mixture shape, unchanged expected yards.
- `combined`: both changes, still the same efficiency estimate.

Point shrinkage is selected using earlier target MSE only, not workload-state or
betting labels. Later-week tests use the same player-games and lines. The real-line
replay retains the exact archived uncertainty object for `targets_only`, including
its original confidence and the complete subsequent scoring adjustments. Shape
changes retain the original projection anchor where the point is unchanged.

| Variant | Yardage RMSE | Brier | Calibration error |
|---|---:|---:|---:|
| Reference | 24.380 | 0.2299 | 3.02% |
| Targets only | 24.279 | 0.2303 | 4.60% |
| Uncertainty only | 24.380 | 0.2301 | 1.60% |
| Combined | 24.279 | 0.2307 | 2.85% |

The target-only squared-error gain was positive in week-grouped resampling, but
its absolute mean bias increased from 0.096 to 0.147 yards, failing the existing
no-worse-bias point screen. The target improvement over the longer-role baseline
was not conclusive. These are point-specific decisions, not failed bankroll gates.

The uncertainty-only median passed its independent historical screen: median MAE
16.716 -> 16.629, with positive week-grouped absolute-error improvement. Its
probability screen did not pass. That no longer prevents testing the median alone.

On the current Week 3 scoring cohort, target-only final Brier improved across
archived offers (0.2511 -> 0.2494), but worsened on the five fixed selections
(0.2616 -> 0.2743). No betting probability change was deployed.

### Runtime separation

`forecast_outputs.py` records point semantics/version separately from the pricing
anchor, distribution checksum, scoring version, line, price, and probability.
It never calls a legacy central forecast a certified expected mean. An approved
point-only replacement adds `expected_yards` or `typical_yards` without overwriting
the legacy `projection` used for pricing, EV, ranking, or ledger decisions.

The point-output application API requires that component's own historical and
prospective deployment decision, tied to the production release and approved
before the forecast cutoff. It has no bankroll/CLV/ROI gate. Discord labels any
such separate output with its component version and `point-only`; existing cards
remain unchanged until an output is approved. No historical locks were rewritten.

### Independent prospective test

The `uncertainty_only` median is pinned by checksum in
`models/target_components/point_registration.json`. Existing daily component
scoring captures it on new eligible receiving locks with feature cutoffs on or
after October 1. Only the median is recorded, not replacement bet probabilities.
The close/grading pipeline evaluates it through `target_point_capture`.

The test deduplicates player-games and compares against the original locked
production median. Three independent weeks are a review minimum, not automatic
deployment. A passing later-week MAE comparison opens a point deployment review;
it does not authorize cash or change the registered FanDuel betting trial.
The initial prospective checkpoint has zero eligible results, as expected.

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.target_components --cache src/nfl_pipeline/modeling/models/target_volume/target-volume-20260930T021306134180Z/training.joblib --through 2026-09-18 --seasons 2025 2026
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.target_volume_validation --artifact src/nfl_pipeline/modeling/models/target_components/target-components-20260930T065356571923Z/model.joblib --season 2026 --week 3
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.target_point_capture
```

Follow-up reports: `nfl_target_components_latest.md`,
`nfl_target_components_offers_2026_week3.md`, and
`nfl_target_point_prospective_latest.json` in `reports/`.

Follow-up validation: 455 tests passed, one skipped; production release and
scoring fingerprint remained unchanged.

## October 2: Prospective cohort integrity

The point checkpoint now evaluates each production release/scoring fingerprint
separately. Changing scoring baselines cannot be pooled to meet the independent
week minimum or pass a point deployment review. Repeated offers/captures count
once per player-game within each cohort; unique player-games are also reported
across cohorts. Multiple cohorts require an explicit cohort-specific review, not
an aggregate pass. Captures without scoring identity are excluded.

The refreshed checkpoint has nine settled player-games from one week. Reference
median MAE is 19.145 versus 19.063 for the candidate. Candidate RMSE and bias are
worse; these remain visible even though the pinned output is a median, not an
expected mean. This is insufficient evidence for deployment. The other 130
player-games are future games, now labeled `future_game_pending` rather than
combined with missing verified results.

No production forecasts, betting probabilities, fixed selections, registration,
or cash authorization were changed. This repair improves evaluation integrity;
it is not evidence of improved production prediction accuracy.

## October 5: Isolated receiving-efficiency experiment

Built and trained an independent yards-per-target model, leaving projected targets
and the reference distribution shape unchanged. It uses lagged player exposure,
position-shrunk rate priors, target depth, catch rate, YAC and matchup context.
Observed targets weight training, so a one-target appearance does not carry the
same skill evidence as a full-workload game. Missing history remains explicitly
unknown; current-game outcomes and offered prices are not features.

Chronological tuning selected shrinkage and blending before each later test fold.
Evaluation used 5,279 unique player-games over seven folds, with week-grouped
resampling and separate decisions for expected yards, typical yards and line
probabilities. Previously examined 2026 Week 3+ rows were excluded. These historical
development folds and proxy lines do not establish prospective or betting edge.

| Output | Conservative reference | Rate repair | Rolling baseline |
|---|---:|---:|---:|
| Expected-yards RMSE | 24.232 | 24.285 | 25.036 |
| Expected-yards bias | 0.082 | 0.632 | -0.062 |
| Median MAE | 16.548 | 16.683 | 17.339 |
| Brier | 0.2296 | 0.2330 | 0.2387 |
| Calibration error | 3.27% | 4.90% | 4.46% |
| 80% coverage | 80.05% | 79.64% | 79.33% |

All three output screens rejected the challenger. Beating the rolling baseline
does not justify replacing the stronger reference. The artifact is offline only;
production forecasts, scoring probabilities and betting approval remain unchanged.

Separately, the forecast-output contract now preserves the captured pre-offer
projection as the independent stat forecast. A projection adjusted toward an
offered line is labeled as the pricing anchor instead. Legacy rows without a
captured original are explicitly unverified. This changes metadata for new locks,
not probabilities, historical locks or the numerical scoring path.

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_receiving_efficiency --cache src/nfl_pipeline/modeling/models/target_volume/target-volume-20260930T021306134180Z/training.joblib --through 2026-09-17 --seasons 2025 2026
```

Artifact: `models/receiving_efficiency/receiving-efficiency-20261005T202840002708Z/`.
Report: `reports/nfl_receiving_efficiency_latest.md`.
Validation: 526 tests passed, one skipped.

## October 5: Direct receiving quantiles and calibrated blend

The follow-up model predicts 15 receiving-yard quantiles from lagged workload,
yardage, target depth, catch/YAC and role-variability features. Quantile
rearrangement prevents crossing. A subsequent held-out block corrects quantile
bias. A fixed 50/50 distribution blend with the conservative conditional model
is calibrated on separate fitting, tuning and selection weeks. Calibration
changes one survival distribution rather than pricing each line independently.

On the same 5,279 later player-games:

| Output | Conservative reference | Calibrated blend |
|---|---:|---:|
| Expected-yards RMSE | 24.246 | 23.864 |
| Expected-yards bias | +0.126 | +0.753 |
| Median MAE | 16.541 | 16.149 |
| Proxy-line Brier | 0.2296 | 0.2244 |
| Calibration error | 3.49% | 2.13% |
| 80% coverage | 80.4% | 82.9% |

The median passed against both the pooled and conditional references with
week-grouped resampling. Expected yards did not pass the no-worse-bias screen;
probabilities did not pass the no-worse-coverage screen. The unblended quantile
model also had poor downside coverage. These failures remain visible, without
loosening the screens after inspecting their results.

Replayed 1,547 immutable FanDuel offer forecasts from Weeks 3-4 through their
archived scoring versions. Offers are not independent player-games. One earlier
cohort improved final Brier from 0.2507 to 0.2484, but its seven fixed research
picks worsened slightly from 0.2568 to 0.2575. The newer cohort's final probabilities
were unchanged: its market calibration assigns zero model probability weight.
Training better raw forecasts cannot change those final probabilities under the
existing pricing policy. No market-weight overrides were installed.

The calibrated-blend median is independently pinned in
`models/receiving_quantiles/point_registration.json`. The existing daily accuracy
capture calls it, and the existing point checkpoint evaluates it. Only forecasts
locked after registration qualify. The comparison uses the original pre-offer
distribution median, not the line-anchored pricing projection. At least three
independent weeks are needed for review, not automatic deployment. The older
point registration, fixed FanDuel research selections and cash policy are intact.

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_receiving_quantiles --cache src/nfl_pipeline/modeling/models/target_volume/target-volume-20260930T021306134180Z/training.joblib --through 2026-09-17 --seasons 2025 2026
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.receiving_quantile_validation --artifact src/nfl_pipeline/modeling/models/receiving_quantiles/receiving-quantiles-20261005T204207709687Z/model.joblib --season 2026 --weeks 3 4
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.target_point_capture --quantiles
```

Reports: `nfl_receiving_quantiles_latest.md`,
`nfl_receiving_quantiles_offers_latest.md`, and
`nfl_receiving_quantile_prospective_latest.json` in `reports/`.
