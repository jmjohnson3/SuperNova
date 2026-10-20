# NFL Forecast and Cash Readiness Plan

Updated: 2026-10-10. This is the durable continuation plan for this project.

## Active Implementation Batch

- [x] Chronological calibration repair and full-path replay implemented/tested.
- [x] Prediction-timeout repair and publication/close reconciliation implemented/tested.
- [x] Version-aware receiving probability blend and selected-pick comparison implemented/tested.

These checkboxes mean code implementation, not demonstrated forecast improvement
or cash approval. Prospective acceptance and the 90% close target remain open.

Production forecast deployment and cash authorization are separate decisions.
Do not change betting caps, old locks, fixed research selections, or production
model pointers as a side effect of these repairs. Missing evidence stays unknown.

## Current Follow-Up

- [x] Audit source-version differences and require replay-proven evaluation aliases.
- [x] Reconcile the two October 5 receiving outcomes using participation evidence.
- [x] Recognize exact, timely watcher quotes without rewriting their snapshot role.
- [x] Verify scheduled game-aware pregame and close tasks are enabled.
- [x] Finish refreshed challenger comparison, deployment decision and regression suite.
- [ ] Complete the October 8 live cycle when its actual pregame/close window opens.

The October 8 cycle is future work, not a completed live test. The existing
follow-up monitors it; do not send duplicate Discord messages or invent closes.

## Roadmap and Acceptance Criteria

1. Reliable live cycle: profile prediction stages and expensive context queries;
   keep fresh quotes, scoring, immutable locks, challenger capture, and Discord
   on the game-specific critical path. Keep near-kickoff close capture independent
   of retrospective evaluation. Verify publication manifests even on manual
   recovery runs, with no fabricated receipts. Target >=90% valid exact closes
   for eligible completed games, not future games.
2. Valid evaluation: expanding chronological season/week folds, earlier labels
   available before the evaluation lock, equal player-game weights, separate
   release/scoring cohorts, and reproducible full-path replay. Use actual median
   and transformed distribution intervals rather than assuming symmetric tails.
3. Receiving forecast improvement: test the conservative forecast and calibrated
   quantile blend. Expected yards use RMSE/bias; typical outcome uses median MAE;
   probabilities use Brier/calibration and lower/upper interval coverage. Define
   tolerances before acceptance testing. A median improvement is not a certified
   expected mean and need not meet cash gates to be deployed.
4. Version-aware probability blend: compare model-only, true paired FanDuel
   no-vig, and learned blend on identical locks. Fit on earlier evidence for the
   same forecast/scoring version. Replay all subsequent scoring adjustments.
   Never turn model weight to one merely to force bets. Preserve monotonicity,
   pushes, and consistent EV/min-price/display probabilities.
5. Selected-pick validation: FanDuel common-line receiving yards is the first
   fixed strategy. Compare existing selection against a conservative challenger
   on identical eligible offers and caps. Evaluate all offers, original selected
   research/micro IDs, and counterfactual ranking separately. Group uncertainty
   by independent weeks; Week 3 informed development and is not untouched proof.
6. Evidence cohort: reconcile unresolved selections and scoring-version mismatch.
   Preserve old results and registrations. Register a new version prospectively
   when approved scoring changes require it. Separate recommendations, paper
   results, and user-confirmed wagers. Never reconstruct missing old forecasts.
7. Bounded cash review: retain $1 flat, five/day, $20 weekly stake budget, $30
   cumulative confirmed-loss pause. Fifty selections and three weeks trigger
   review, not automatic approval. Require credible selected probabilities,
   executable positive EV, reliable settlement, ROI/CLV support and manageable
   concentration. Estimate effective sample size and uncertainty. CLV and model
   accuracy do not guarantee profit. Increase stakes only by a separate review.

## Deferred Work

- Finish prospective quantile/median testing and repair receiving bias/coverage.
- Reconcile and explicitly version cash-strategy registration if scoring changes.
- Complete prospective selected-probability and market-baseline review.
- Decide forecast deployment independently from cash-trial authorization.
- Expand to other markets only with their own validation; do not pool their proof.

## Continuation Protocol

Read this file and the latest reports before continuing. Update active checkboxes
only after tests and an actual diagnostic run, and record measured results below.
Keep retrospective development evidence distinct from prospective acceptance.

## Implementation Log

- 2026-10-05: saved agreed plan; first implementation batch in progress.
- 2026-10-06: replaced leave-one-week-out calibration with earlier-week-only
  folds. Labels must have been available before the first test lock. Overlapping
  player-games are purged and duplicated offers share one unit of weight.
  Replay uses archived scoring code and captured inputs; missing legacy inputs
  and versions without the calibration layer are excluded, not reconstructed.
- Calibration uses the actual median for asymmetric intervals and excludes
  pushes from binary scoring without excluding them from selection. Tests check
  that changing later outcomes cannot change the earlier-fitted probabilities.
  The calibrator cannot overwrite production via `--write`.
- Added receiving comparisons for all eligible offers, original micro IDs,
  registered research IDs, and production/blend/conservative ranking. Each
  forecast version, scoring fingerprint and challenger checksum is separate.
  Compare against true-paired FanDuel probabilities on identical offered lines.
  Pending selections remain pending. Per-line blending is not certified as a
  coherent whole-distribution CDF; monotonicity is reported, not assumed.
- Latest completed replay found only one settled NFL week per supported scoring
  cohort, so there was no later-week acceptance fold. In-sample parameter fits
  are diagnostics only. No blend/calibrator was installed and no cash was approved.
- Added per-stage prediction timing and early no-game exit (including safe
  Discord formatting). Optimized as-of context reads, narrowed materialized
  fields, grouped prior-depth lookup, and materialized season aggregates. An
  EXPLAIN run found 7,596 repeated injury-season aggregates consuming ~78 seconds.
  Installed `sql/NFL_context_asof_index.sql` as standalone maintenance, not live DDL.
- Verification found legacy depth ties were nondeterministic. New ordering keeps
  the existing date/rank precedence, then resolves ties by offensive role,
  observation time and stable role fields. The prior-depth lookup uses the same
  ordering. This corrects ambiguous inputs; model weights are unchanged and old
  locks are not rewritten. The runtime audit explicitly discloses tie rules and
  reference execution optimizations rather than claiming an unchanged legacy query.
- All 1,012 context rows matched the reference with those explicit tie rules.
  Verification uses single JSON expansion/narrow materialized fields in the
  reference to keep it bounded; the untouched legacy query exceeded its timeout.
  Runtime varies with concurrent jobs, so the 5.9-second isolated observation is
  not a production latency guarantee. The full dry-run measurement below is the
  end-to-end check.
- Full read-only October 8 prediction dry run: 52 forecasts, 34 snapshot players,
  63.44 seconds, zero saved forecasts and no Discord sends. Stored FanDuel quotes
  were stale and correctly rejected. This was not a fresh-odds/published live cycle.
- Final NFL regression suite: 564 passed, one skipped, 78 existing pandas
  fragmentation warnings. Focused tests cover chronology, future-label exclusion,
  immutable replay, push handling, fixed selections, unknown outcomes, exact
  cross-provider closes, no-game output and deterministic context comparison.
- Publication audit now recognizes exact manifests from manual recovery runs;
  October 5 verified all seven recorded publication pages.
- Cross-provider CLV resolution requires the same book/player/stat/exact line,
  matching home/away teams, kickoff within five minutes of the authoritative
  schedule, and close after lock but before kickoff within two hours. Different
  provider event IDs alone no longer reject a verified same-game FanDuel close.
- Refreshed October 5 derived CLV labels without changing locks. Receiving trial
  coverage increased from 4/21 to 15/21 (71.4%): eleven genuine closes recovered.
  Six remain unknown: two player markets absent, three exact lines absent and one
  missing side price. Two receiving outcomes still lack participation/results.
  The live-cycle audit correctly remains `needs_attention`, not green.

## Repeatable Commands

Run from the repository with its virtual environment. These commands do not
install a model, approve cash or send Discord messages.

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.fit_market_calibration --stat receiving_yards
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.receiving_market_blend --artifact src/nfl_pipeline/modeling/models/receiving_quantiles/receiving-quantiles-20261005T204207709687Z/model.joblib
.\.venv\Scripts\python.exe -m nfl_pipeline.prediction_runtime_audit --date 2026-10-05 --cutoff 2026-10-05T23:45:00+00:00
.\.venv\Scripts\python.exe -m nfl_pipeline.clv_report --date 2026-10-05
.\.venv\Scripts\python.exe -m nfl_pipeline.live_cycle_audit --date 2026-10-05
.\.venv\Scripts\python.exe -m pytest src/nfl_pipeline -q
```

`clv_report` refreshes derived close labels; the audit only reads source data.
The nightly training runner already invokes the repaired calibration tool.
The explicit receiving challenger command stays pinned to the reviewed file;
it must not silently follow a newly trained `latest` artifact.

## October 7 Follow-Up Findings

- Seven scoring fingerprints were present in archived receiving captures. The
  audit reproduced their original probabilities. No cross-version merge was
  justified: sharp-reference blending and median shifting are genuine scoring
  changes; the newer context query also changes the input program. Identical
  conditional pricing is not proof of identical forecast generation.
- `scoring_equivalence` now keeps separate scoring-program and input-program
  signatures. Only equal programs plus successful original and cross-version
  replay can share an evaluation cohort. Original IDs/fingerprints are preserved;
  proof is limited to checked forecast IDs and invalidates on dependency changes.
  It runs during settlement, never on the near-kickoff capture critical path.
- Oscar Delp (14739): verified 24 offensive snaps, zero targets/receiving yards;
  the imported zero is supported by participation. Noah Fant (14673): exact-week
  inactive roster plus complete participation/PBP evidence; recorded as a void,
  not zero yards or a winning under. No unresolved final receiving rows remain
  in the October 5 registered trial.
- Two exact watcher quotes, Pitts 31.5 and Lance 15.5, were already captured five
  minutes before kickoff after their locks. Their original role was `live`.
  The resolver now accepts those immutable watcher observations only under the
  same exact-identity and timing requirements; arbitrary live/lock rows are not
  eligible. Source row ID/role remain visible in the close diagnostic.
- October 5 latest FanDuel cohort: 37/37 valid exact closes. Historical receiving
  trial locks for that date: 17/21 (81.0%), not 90%. Hooper 10.5, Zaccheaus 7.5 and
  Robinson 37.5 were absent from valid-window exact-line observations. London
  79.5 had no selected-side under price. All four stay unknown; the earlier
  18/21 estimate missed the unavailable under. Alternate retrieval was enabled
  in all 13 saved SGO close payloads. All prop locks for the date: 59/66 (89.4%).
- A separate validation timeout involved repeated large archived calibration
  dictionaries and expensive JSONB extraction/sorting. The read-only reader now
  streams original payloads in batches of eight, shares exactly equal captured
  inputs in memory, and sorts references after loading. It never reconstructs
  missing inputs. Numeric types, missing fields and original values are retained.
  The final reader matched all 32 original October 5 receiving payloads exactly;
  its 3,589-row receiving history loaded in 48.63 seconds. The refreshed seven-
  version replay audit again had no failures and justified no cross-version merge.
- A full run under concurrent database load took 1,044.6 seconds. After switching
  to the simpler streaming reader, a complete validation finished in 270.62
  seconds; faster in-memory interning reduced the final rerun to 223.69 seconds.
  Both runs retained the same 2,037 settled unique offers: 1,797 exact replays matched;
  240 legacy decisions lack captured replay inputs. No mismatch was reported.
  The final validation step has a 600-second bound outside active close capture
  and still fails the task on timeout. A successful manual test does not erase
  the scheduler's earlier failed runs or prove the next scheduled run succeeds.
- Refreshed pinned challenger review: latest compatible settled cohort has 13
  player-games from one week. Production/challenger Brier both 0.25097; market
  0.25114; both 80% intervals cover 76.9%. The separately versioned four fixed
  research selections have Brier 0.27251 versus market 0.25000 and 25% interval
  coverage. Neither scope establishes a deployable probability improvement.
  Forecast deployment remains separate from cash approval; neither was changed.
- Full regression suite: 580 passed, one skipped, 78 warnings. The first run's
  watcher-test mismatch was from a concurrent watcher change; the current test
  and full suite passed without weakening or changing that watcher behavior.
  After the final loader optimization, 65 focused regression checks also passed,
  including preservation of unknowns, integer/float types, and hash-collision rejection.
- All 25 historical micro ledger entries are accounted for: five evaluable,
  six void/push, fourteen missing evaluation inputs. The fourteen legacy entries
  are not fabricated or treated as verified probabilities. Refresh-only ledger
  reconciliation made no new locks and needed no additional result updates.
- October 8 preflight has twelve original receiving locks and twelve matching
  pinned captures, all pending the future game. Their old prices are not live
  prices now; the game-aware task must refresh quotes before its new locks.
- Scheduler preflight: pregame and close tasks enabled at ten-minute intervals.
  October 8 TB-DAL kickoff is 00:15 UTC October 9; the game-aware pregame target
  is 22:45 UTC October 8 (4:45 PM Denver). It correctly reports a future window.

## Next Checkpoint

### Whole-Distribution Challenger Work (October 7)

- Implemented a shared stat-specific CDF calibration and learned market-blend
  path for passing/rushing/receiving yards, receptions, all three TD markets,
  spreads, totals, and score-margin win/tie probabilities.
- Calibration uses discrete outcome likelihood on unique player-games. Shape
  fitting, blend tuning, and later-week evaluation are chronological and use
  observed label availability. Calibration can revert to identity; model weight
  can legitimately be zero. A new challenger does not inherit legacy weights.
- All lines for a player/book/lock share one paired-price anchor. Market shape
  outside that anchor is an explicit assumption, not measured alternate odds.
  Synthetic pairs cannot anchor the curve. Missing moneyline prices remain
  unknown: game winner forecasts are not moneyline betting proof.
- New live locks capture raw curves, pinned policies, source checksums, and
  prospective scores. The receiving quantile challenger has a separate source
  identity. Old game locks lacking raw distributions are excluded, not rebuilt
  with current model variance.
- Forecast deployment is separately gated on later-week final Brier,
  calibration, interval coverage, week-grouped uncertainty and unchanged
  selection comparisons. Registry updates are atomic, with prior policies in
  deployment audit events. Cash authorization is never granted by this path.
- Nightly validation pins available challengers once and evaluates deployment;
  it does not silently refit a running trial. Legacy game calibration no longer
  uses leave-one-week-out or supports an unreviewed `--write` installation.
- Full NFL suite: 610 passed, one skipped. Real-data validation inspected
  11,166 prop locks and retained 1,819 unique eligible offers across the available
  books (alternative curves are not counted as additional independent offers).
- Two FanDuel receiving policies are pinned for future locks: the raw-distribution
  challenger with model weight 0.9, and the checksum-pinned quantile challenger
  with model weight 0.5. Their extra CDF correction did not pass inner holdout
  checks, so identity calibration was retained. Neither policy was deployed.
- For the quantile source's available later-week test, Brier was 0.25022 versus
  production 0.25037 and the same-offer market 0.25019. Calibration error was
  4.06% versus production 1.25%; interval coverage was 74.1%. This is not a
  deployment pass. The newer 0.5-weight policy still needs prospective evidence.
- Rushing/receiving TDs have no eligible true-paired rows in this audit. The
  655 legacy game locks lack captured raw distributions. New locks now capture
  these inputs; no unavailable history has been manufactured.
- Whole-CDF deployments affect probability/range outputs only. The existing
  point forecast is retained; the calibrated curve mean is explicitly separate.
  Any new-policy micro-looking row remains paper until separate cash review.
- Final verification: 67 focused tests passed after report-scope cleanup. An
  archived-input smoke test loaded both pinned receiving sources, computed their
  probabilities, and confirmed the new shadow capture left production side and
  probability unchanged. It saved no forecasts and sent no Discord messages.

1. Keep the production model and probability rules stable. Collect a later week
   for the same scoring cohort; do not pool changed fingerprints just to pass a
   sample threshold. Any compatibility alias needs explicit replay proof.
2. Repeat the receiving blend report once later-week outcomes are available.
   Judge held-out Brier, calibration, tail coverage and selected picks separately.
3. Verify the next fresh-odds live run completes capture/publication near kickoff.
   The dry run does not prove fresh quote or real publication success.
4. Reconcile the remaining missing outcomes and exact closes only from evidence.
   Never turn missing participation into zero yards or fabricate historical prices.
5. Continue the deferred roadmap above; make forecast deployment and cash review
   separately. No sample count or short winning streak is automatic approval.

### Role-Conditioned Workload Repair (October 7)

- Added `role_workload.py`: conditional integer target/carry curves, coherent
  low/normal/high workload probabilities, and hierarchically shrunk residual
  tails by prior-volume role and position. Role is a lagged-volume proxy, not
  verified starter status. Unverified or future context remains unknown.
- Efficiency is intentionally held fixed. The point/rate comes from the same
  conservative forecast in each comparison, while efficiency residual tails
  are centered at zero. Zero opportunity produces zero yards. No normalization
  over today's offered players or hindsight participant lists is performed.
- `train_role_workload.py` uses separate whole-week fit, count-tail,
  efficiency-tail and count-blend tuning blocks, then expanding later-week
  evaluations. Duplicate player-games and missing/inconsistent outcomes are
  rejected. The blend strength is selected on opportunity CRPS, not test results.
- Receiving development test: 5,278 unique player-games across 24 NFL weeks.
  Combined versus conservative reference: target RMSE 2.210 versus 2.286;
  median yardage MAE 16.175 versus 16.524; proxy Brier 0.2242 versus 0.2291.
  Higher-prior-volume interval coverage improved from 59.6% to 82.0%.
  Mean-yardage bias worsened to -1.995 from -0.065; aggregate nominal-80%
  coverage was 84.3%. Opportunity and median screens passed, but the expected
  mean and full probability screens did not. These are development findings,
  not prospective approval.
- Applied the same count-first method independently to RB rushing (the cache
  has no QB rushing training support). Across 1,469 later player-games, carry
  RMSE improved from 4.936 to 4.779. Opportunity passed its historical screen;
  mean, median and probability screens did not. Do not extrapolate it to QB
  rushing or deploy it because the receiving experiment improved.
- `validate_role_workload.py` replays the original scoring version with the
  original quotes, adjustments and ranking function. It includes unchanged
  non-target stats in the saved ranking universe, grades a counterfactual
  selection's actual side, and reports original micro/research IDs separately.
  Missing legacy inputs are excluded; ranking is not a reconstructed complete
  sportsbook universe. The loader is bounded to requested slate dates.
- Receiving replay covered 676 eligible FanDuel offers from weeks 3-4. In the
  largest scoring cohort, final probabilities were unchanged because the
  archived market adjustment removed model influence. Elsewhere results were
  mixed, and selected research improvement was not established. Neither old
  weeks nor the improved historical proxy Brier authorize deployment.
- Rushing replay covered 192 eligible RB offers; 81 QB offers were explicitly
  excluded because this artifact has no QB training support. In its 83-row
  scoring cohort the combined final Brier improved from 0.2671 to 0.2605 but
  remained worse than market 0.2500; its 93-row cohort slightly worsened.
  Counterfactual micro selection was empty, not an approval. Keep this
  challenger offline rather than infer general improvement from carry RMSE.
- Verification: full NFL suite passed with 626 tests, one skipped and 78
  existing feature-fragmentation warnings. Training/validation used isolated
  artifacts and read-only original-lock queries. No odds credits or Discord
  publication were needed for these experiments.
- Production, pinned trials, cash limits, Discord and historical locks remain
  unchanged. Next decision: prospective component validation of the improved
  opportunity outputs, followed by challenger-specific calibration/blending
  review. Do not treat a better target count as an accepted yardage mean or
  automatically enable its betting probabilities.

Commands (read-only evidence validation; offline artifacts are written):

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_role_workload --cache src/nfl_pipeline/modeling/models/target_role_context/20260930T022417027001Z/training.joblib --through 2026-09-17 --seasons 2025 2026
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.validate_role_workload --artifact src/nfl_pipeline/modeling/models/role_workload/role-workload-receiving_yards-20261007T212629386014Z/model.joblib --season 2026 --weeks 3 4
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_role_workload --cache src/nfl_pipeline/modeling/models/target_role_context/20260930T022417027001Z/training.joblib --through 2026-09-17 --seasons 2025 2026 --stat rushing_yards
```

## Workload Probability Path (October 8)

- Connected the checksum-pinned receiving workload artifact to the existing
  whole-CDF policy framework. It uses the original unanchored yardage forecast
  as its fixed efficiency input, not the offered line. Unsupported positions,
  missing original history and training that overlaps the lock are excluded.
- Fit beta-CDF calibration on earlier player-games; test identity versus that
  correction and tune the market blend on a later week. Duplicated offers do
  not increase effective sample size. For this source, blend weights are
  0/25/50/75/100%, with a predeclared small quadratic penalty toward the market.
  A further chronological outer week is required for independent evaluation.
- The full distribution is calibrated and blended, preserving monotonic line
  probabilities, complementary sides and integer push mass. When deployed,
  its final probability bypasses the old model's zero-trust adjustments. That
  same final probability drives side selection, EV, minimum executable price,
  drift checks and Discord. Point-forecast deployment remains separate.
- Added an offline conservative ranking comparison. It penalizes the positive
  model-market disagreement according to earlier unique player-game support;
  this is a ranking experiment, not a confidence interval or changed bet price.
  Original scoring/caps are replayed with unchanged competing markets. Five
  is a ceiling, not a quota; failed replay batches are explicitly excluded.
- Repaired the historical selection clock: the scorer evaluates whether a game
  started at the saved lock, never at today's wall clock. Live scorer globals
  are not patched. Prior empty counterfactual rankings should not be interpreted
  as reliable selection evidence until rerun with this correction.
- Preserve the existing fixed FanDuel receiving trial. A new policy is pinned
  separately and captured on future offers; old locks are never rewritten.
  Forecast review compares production on the challenger's exact selected
  offers, so it does not require the old model to have approved cash bets.
  Cash approval, ROI/CLV requirements and stake limits remain separate.

Initial development fit used 169 earlier player-games for calibration and 185
later player-games (380 offer rows) for tuning. Raw workload Brier was 0.2593;
calibrated model-only was 0.2451; the selected 50/50 blend was 0.2425. The
blend's calibration error was 2.34% and its nominal 80% interval coverage was
76.2%. These are tuning results, not untouched acceptance or demonstrated
profitability. The outer week's offers were still unsettled in this run.
Production Brier on the same tuning offers was 0.2504 and exact FanDuel no-vig
Brier was 0.2502. Production calibration error was still better (1.25% versus
2.34%), so the lower Brier alone is not a deployment pass. Six existing fixed
research selections improved Brier from 0.2633 to 0.2226, but their calibration
worsened and interval coverage was only 50%; that tiny tuning subset is not
selected-pick proof. The unchanged and conservative micro screens selected no
rows in this retrospective pool, even after correcting the clock.

Pinned source policy: `56a5b51380724620bfa61025d96b1a9cbeaf080d911e24bfb001cf22cd8da941`.
It is an additional prospective capture, not an active forecast replacement.
The two earlier pinned policies and the fixed research registration were
preserved. Full NFL verification: 640 passed, one skipped; 78 existing pandas
feature-fragmentation warnings remain.

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.validate_distribution_policy --markets receiving_yards --workload-artifact src/nfl_pipeline/modeling/models/role_workload/role-workload-receiving_yards-20261007T212629386014Z/model.joblib --season 2026 --weeks 3 4 5 --register
```

Next: leave this policy fixed, collect new pregame captures, and review final
Brier/calibration/coverage plus unchanged and conservative selections against
production and exact FanDuel no-vig probabilities. Existing weeks used to fit
or design this component cannot count as prospective acceptance. Do not switch
production or authorize cash from the development improvement alone.

## Isolated Forecast Repairs (October 8, Continued)

Implemented and trained `forecast_repair.py` and `train_forecast_repair.py`:

- Rushing: separate carries distribution and exposure-weighted yards-per-carry
  residual model. Longer-term rate estimates shrink toward position priors.
  Workload-only, efficiency-only, joint and calibrated-joint variants are scored
  on identical player-games. Actual carries are training labels/weights, never
  pregame features. Missing context is not labeled healthy.
- Passing: an attempts distribution includes low-volume outcomes, coupled with
  a separately fitted efficiency model and role-specific asymmetric residuals.
  Zero attempts produce zero yards. Negative-yard support and extreme tails
  remain available; no blanket symmetric interval widening is applied.
- Calibration: fit the whole CDF in one chronological block, then test it in a
  separate later block. Reject it when Brier, calibration, interval score, or
  either tail deteriorates. Integer-yard CDF interpolation avoids assigning
  artificial zero mass between quantile knots. Push handling stays coherent.
- Totals: a small regularized residual model with at most 13 symmetric lagged
  features: scoring, points allowed, plays, pass rate, yards/play, red-zone
  efficiency and rest. Keep only inputs with >=90% training coverage. No closing
  price, current-game result or retrospectively observed weather is a feature.
- All fits use unique player-games/games, separate chronological inner blocks,
  expanding outer folds, and NFL-week-clustered comparisons. Mean RMSE/bias,
  median MAE and probability/coverage decisions are separate.

Historical development results across 24 evaluated weeks (not untouched
prospective acceptance, and not actual sportsbook-line proof):

| Component | Reference | Challenger | Remaining concern |
|---|---|---|---|
| Rushing joint curve | Median MAE 17.734; RMSE 26.687; Brier .2338 | Median MAE 17.471; RMSE 26.079; Brier .2316 | Calibration 3.63% vs 3.36%; expected-mean bias gate fails |
| Passing joint curve | RMSE 88.520; median MAE 69.684; Brier .2075; coverage 77.5% | RMSE 85.273; median MAE 67.566; Brier .2020; coverage 82.4% | Calibration 2.80% vs 2.11%; probability screen not passed |
| Small totals curve | Median MAE 11.141; RMSE 13.997; Brier .2545 | Median MAE 10.818; RMSE 13.778; Brier .2450 | Coverage 75.8% vs 77.2%; Brier gain uncertainty still crosses zero |

These comparators are chronological refits/statistical baselines. They must not
be described as verified improvements over immutable live production.

The receiving workload full-path report was refreshed using 8,000 loaded locks.
Its original policy and research registration were preserved. Development
Brier remains .2425 versus production .2504, but calibration remains worse,
and there are not yet enough settled post-registration weeks for acceptance.
The fixed research selections and counterfactual micro screens remain separate;
zero eligible micro selections are not manufactured into a five-pick quota.

Three evaluated sources are pinned in `models/distribution_policy/sources.json`
for capture on future locks: rushing/passing `joint_calibrated` and totals
`small_total`. They are not active replacements. Future captures retain the
exact source curve and descriptor alongside the original forecast. The nightly
distribution-policy step can fit an offered-line policy after sufficient earlier
capture data, pin it, and evaluate it on subsequent independent weeks. Raw-source
captures are not retrospectively relabeled as a prospective fitted policy.

Forecast deployment no longer waits for 20 cash-selected micro picks. Missing
selected-pick evidence is reported separately. It still requires prospective
same-policy/same-offer improvement, three independent weeks, 60 settled outcomes,
better Brier/calibration, appropriate coverage and complete ranking replay.
Observed selected-probability deterioration remains a forecast regression.
No cash, ROI/CLV, stake, or ledger approval is granted by forecast deployment.
This path deploys distributions; raw point-forecast replacements remain a
separate output decision.

Final chronology review moved count-mixture tuning before efficiency residuals
and CDF fitting; calibration no longer shares labels with count-blend tuning.
The two intermediate player-source registrations were replaced only after a
read-only database check proved no forecasts had been locked since registration
and neither source had a fitted or deployed policy. The receiving and totals
registrations were not replaced. Older development artifacts remain available.

Production remains `nfl-20260918T143342Z`; no active replacement passed this run.
No Discord posts, wagers or historical lock rewrites were performed. The
source smoke check loaded all three artifacts, verified normalized mass and
monotone over probabilities, and wrote zero locks.

Commands:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_forecast_repair
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_forecast_repair --pin-latest
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.validate_distribution_policy --register --deploy-passing
```

Next decision: compare new locked curves against the actual saved production
probabilities and FanDuel's identical offers. Do not deploy based on these
development tables or restart the existing receiving trial.

Verification after the final chronology repair: 653 NFL tests passed, one
skipped; 78 existing pandas feature-fragmentation warnings remain. All three
final source artifacts passed a read-only normalization/monotonicity smoke
check. Active distribution replacements: zero. `git diff --check` passed.

## Master Accuracy Implementation (October 10)

The accuracy plan is now implemented as a component pipeline rather than a
single-release replacement:

- Forecasts carry a fixed pregame horizon (`t20`, `t120`, `t24h`, or `early`).
  Prospective samples from different horizons cannot be pooled to pass a gate.
- Player-stat deployment is output-specific. Expected mean, median, and betting
  probability can be accepted independently; a better point forecast no longer
  inherits an unproven probability curve.
- The active registry merges approved outputs by stat. Historical improvement
  only registers a shadow candidate. Post-registration activation requires at
  least 100 unique player-games, three independent weeks, one fixed horizon and
  a positive week-clustered lower confidence bound. Probability additionally
  has to beat production and market Brier, improve calibration and retain 75-85%
  nominal-80% coverage. This path never grants betting approval.
- Receiving and rushing are built from team opportunity first: predicted team
  pass/rush volume, targetable volume, explicit unallocated roster reserve,
  player share allocation and separately shrunk efficiency. Learned strengths
  are selected on later chronological data and can remain zero.
- Passing remains attempts times efficiency with a separately tested tail. A
  failed tail repair cannot veto a proven expected-mean or median improvement.
- Missing injury, depth and teammate context now have explicit indicators.
  Unknown context remains missing. The old fallback prior is retained under the
  named `injury_risk_legacy_imputation` field until a challenger proves a better
  treatment; proxies remain labeled as proxies.
- Training-table reads are cached by source signature, bounded by a configurable
  SQL timeout and no longer repeat the redundant raw-gamelog join.
- Settlement automatically reevaluates registered candidates and activates only
  the output(s) that pass. Production cash rules, Discord tiers and bankroll
  authorization remain separate.
- Scheduled retraining cannot overwrite an in-progress prospective candidate.
  The component ID remains pinned until a complete fixed-horizon review either
  activates its passing outputs or retires its failed outputs. New offline runs
  may still be produced and audited while that immutable trial collects data.

Latest chronological retrain: `components-20261010T222543607583Z`.

| Candidate | Historically eligible outputs | Key later-fold result | Prospective state |
|---|---|---|---|
| Team receiving yards | expected mean, median | RMSE 24.215 vs 24.524; bias -0.284 vs -2.006; median MAE 16.431 vs 16.773 | registered, awaiting 100 rows / 3 weeks at one horizon |
| QB attempts x efficiency | expected mean, median | RMSE 83.329 vs 85.684; median MAE 66.725 vs 67.243 | registered, awaiting 100 rows / 3 weeks at one horizon |
| Receiving conditional uncertainty | probability | Brier .22890 vs .23149; calibration 1.25% vs 2.79% | registered independently, awaiting identical-line FanDuel evidence |
| Team rushing yards | none | RMSE 27.299 vs 26.652 | rejected; no candidate installed |
| Team passing allocation | none | RMSE 88.542 vs 87.035 | rejected; no candidate installed |

The team-allocation internals did improve the parts they are meant to estimate:
pass-attempt MAE 6.429 to 6.015, carry MAE 6.156 to 5.748 and weighted player
share MAE .02011 to .01820. Those gains do not override a failed end-stat gate.

### Independent Probability Challenger Path

- Registry contract v3 permits multiple non-overlapping candidates for one stat.
  Receiving expected mean/median and receiving probability now have separate
  immutable component IDs, evidence files, review decisions and activation paths.
- Nightly retraining preserves any candidate already collecting evidence for an
  output, while allowing a disjoint output candidate to register. Activation or
  rejection removes only the reviewed component and cannot reset its neighbor.
- New forecast payloads store every candidate under `stat_model_challengers`.
  The validator still accepts the legacy singular field for old immutable locks,
  but never combines the two component cohorts.
- Probability evidence requires the exact same FanDuel offer, side, line and
  lock. It must beat both production and FanDuel no-vig Brier, improve
  calibration, retain 75-85% interval coverage and have a positive clustered
  lower confidence bound. Point evidence remains player-game based.
- Forecasts created before a component's registration timestamp are excluded.
  A refreshed trial cannot reuse earlier rows merely because an artifact ID is
  the same. Production scoring, Discord picks and cash authorization remain
  unchanged until the corresponding output passes.

Verification: schema migrations completed; the October 11 13-game no-save
smoke run scored 881 offers and 451 players. All 196 receiving forecast rows
carried both independent shadows. The late-night run had no fresh FanDuel
receiving offers, so it correctly created no identical-line evidence. All 687
NFL tests passed with one skip. The prospective validator currently has zero
settled post-registration rows, so active components and cash approval are
correctly unchanged.
