# NFL Prediction Pipeline Audit

Audit date: 2026-09-17. Scope: the current NFL pipeline, its shared configuration/subprocess dependencies, installed NFL scheduled tasks, model artifacts, logs, and relevant PostgreSQL data. This is not a fresh audit of the separate MLB pipeline.

## Verdict

The pipeline executes, but it is not yet a trustworthy measurement of predictive improvement. There are confirmed defects in historical data completeness, spread semantics, training/live feature parity, evaluation independence, and prediction/ledger identity. These should be repaired before adding another spike model or interpreting accepted-model reports as evidence of a betting edge.

This audit did not change production models, odds, predictions, ledgers, scheduled tasks, or Discord messages. Database checks used read-only sessions. The audit report is the only intentional new file.

## Verification Performed

- Reviewed ingestion, feature generation, opportunity/stat/distribution/game training, live prediction, odds parsing/provider fallback, exact-line training, close resolution, grading, ledgers, orchestration, and tests.
- Ran `python -m pytest src/nfl_pipeline -q`: **19 passed**.
- Ran `python -m compileall -q src/nfl_pipeline`: passed.
- Queried installed Windows NFL tasks: Daily, PrimeTime-Refresh, Close, and Training all last reported exit code 0. Close is installed with a 10-minute repetition and 30-minute execution limit.
- Read recent daily/close/training logs without exposing credentials.
- Reproduced odds-dependent role features with synthetic player histories.
- Built the September 17 live feature snapshot through read-only loaders, without model inference, saving predictions, or posting messages.
- Queried historical coverage, prediction mutations, ledger references, result coverage, and CLV statuses.
- Verified NFLverse spread semantics against its official documentation.

Passing tests and task exit codes do not invalidate the defects below. Current tests do not exercise most of these contracts.

## Priority Findings

### F01 [P1] NFLverse spread signs are reversed in model inputs and baselines

Sources: [schedule import](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/import_nflverse.py:152), [player implied points](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/features.py:182), [game feature load](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/game_features.py:75), [margin baseline](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_game_models.py:341), [live game environment](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_today.py:429).

NFLverse uses positive `spread_line` for a home favorite. The importer preserves that value, but downstream code treats it like a sportsbook home handicap: it negates the value for the expected margin and subtracts it from the home implied team total. Live sportsbook home spreads use the opposite convention, so historical and odds-sourced live features also disagree.

Official definition: [nflfastR reference manual](https://nflverse.r-universe.dev/nflfastR/doc/manual.html). This source defines positive spread values as favoring the home team.

Read-only database comparison:

| Season | Scored games with spread | Correct market-margin MAE | Current reversed baseline MAE |
|---|---:|---:|---:|
| 2024 | 285 | 9.704 | 14.447 |
| 2025 | 285 | 9.670 | 14.396 |
| 2026 | 14 | 10.786 | 14.071 |

The current game report claims margin MAE 10.221 versus baseline 14.071, a 3.851 gain. Against the correctly signed market baseline on these 14 games, the difference is only about 0.565. That still is not an untouched evaluation because of F05.

Repair: preserve provider-native values, introduce an explicitly named canonical `home_handicap` and expected-home-margin conversion, rebuild derived features, then retrain. Prefer fresh eligible sportsbook odds for live market context; do not let an old non-null schedule value override them. Add favorite/underdog and implied-score sum tests. Do not patch only the baseline without rebuilding affected models.

### F02 [P1] The entire 2025 player-stat season is absent

Sources: [historical importer](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/import_nflverse.py:360), [daily steps](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/run_daily_and_notify.py:132), [training runner](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/run_training.py), [live history loader](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:1019).

The database contains 285 scored 2025 games, but **zero 2025 rows in both player gamelogs and player training features**. It contains 5,590 player rows for 2024 and 312 for 2026. The historical feature table jumps directly from 2024 to 2026.

This makes rolling player form, role, efficiency, and opponent history skip an entire season. Players absent from older history are also omitted because the live snapshot iterates historical player IDs rather than the current roster universe. The daily/training runners import current-season context and usage, but do not run the base historical importer to repair this gap.

Repair: backfill 2025 from a verified supported source, validate schedule/player-game joins and season coverage, refresh historical usage/context, and rebuild features/models. Add season continuity and expected-game coverage assertions that prevent a training run from silently accepting a missing season. Add a roster-based cold-start path for players with no history. The audit confirmed the gap, not which earlier import failure originally caused it.

### F03 [P1] Raw player projections change with the set of players offered by the book

Sources: [offer filtering](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:1522), [role features](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:1448), [historical role features](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/features.py:380).

The live snapshot filters out players without offered props before calculating team target/carry/pass shares and role ranks. A partial odds feed can therefore turn a secondary receiver into the apparent primary receiver and inflate target share without any football information changing.

Reproduction: two teammates with historical target averages 8 and 4 produce shares 0.667 and 0.333. Supply an offer for only the first player and his computed share becomes **1.000**. This is a feature error, not a modeling preference.

Historical role denominators are also constructed from that game's player-result rows, rather than an explicitly reconstructed pregame roster. An as-of roster is necessary to avoid allowing eventual participation to determine historical role features.

Repair: create a full as-of player/team forecast snapshot first, calculate roles on that stable universe, then join offers. Test that adding or removing an unrelated book offer cannot change a player's raw forecast or features.

### F04 [P1] Live spike/workload inputs do not match training inputs

Sources: [historical rolling builder](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/features.py:302), [depth movement](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/features.py:425), [live snapshot](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:1550), [live game snapshot](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_today.py:438).

- Historical features include rolling standard deviations; the live player snapshot computes averages only. All 17 tested `*_std_5` inputs were absent in the September 17 snapshot, including targets, carries, passing attempts, snaps, and routes. Downstream defaults erase volatility signals used by the trained spike/workload models.
- `depth_rank_delta` uses a player-group shift on the current one-row-per-player frame. All 26 live players had missing deltas, so genuine depth movement cannot reach this path.
- Live `team_game_number` counts player rows, not distinct team games. The synthetic three-game example returned 6; the actual snapshot reached 1,103. Training counts distinct team games.
- The game snapshot omits injury inputs built by the historical game feature pipeline. Live reindexing substitutes defaults/training fills rather than current QB/OL/skill injury context.

Repair: use one shared as-of feature builder for historical replay and live prediction, with explicit prior depth and distinct team-game history. Compare replayed/live features field by field for the same lock. Fail a model's readiness check if a required feature is unexpectedly missing instead of silently converting it to a neutral default.

### F05 [P1] Repeated model selection uses the same small final holdout

Sources: [player selection](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_models.py:3261), [player split](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_models.py:3127), [game selection](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_game_models.py:535), [distribution fitting](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_distributions.py:965).

Feature sets, direct versus residual models, blend weights, bias repairs, and baseline variants are repeatedly compared against the same holdout. Legacy distribution code even estimates residual width from the holdout outcomes and evaluates probabilities on those same outcomes. In-sample upstream training predictions are also used for several calibration/mixture fits.

Current artifacts evaluate game models on 14 games and receiving yards on 194 player-games. The game total model is marked accepted for about 0.075 MAE improvement on 14 games. Those are selection results, not independent proof of improvement.

The newly added receiver mixture uses an earlier validation window and correctly did not replace the incumbent after failing comparison. However, its upstream artifacts are held fixed, not retrained independently inside every fold. That narrower safeguard does not repair the rest of the stack.

Repair: use expanding week-based outer folds; tune features, mixture weights, calibrators, and thresholds only inside earlier inner folds. Generate upstream opportunity/stat predictions out of fold before training downstream models. Reserve an untouched final season/week block. Report per-week performance and game/week-clustered uncertainty, not just a tiny point-estimate gain. A test should prove that changing final-test labels cannot change fitted artifacts or hyperparameters.

### F06 [P1] Historical context is not reliably point-in-time

Sources: [historical context joins](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/features.py:201), [context upserts](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/import_context.py:292), [live context cutoff](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:1099), [exact-line lock join](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_prop_exact_line_models.py:83).

Historical roster/injury joins choose the latest row by week and update time, without enforcing availability before the forecast lock. Depth checks use a date instead of a precise pre-kickoff timestamp. Closing schedule odds and observed game weather are not substitutes for information available at a morning lock.

Live context filtering uses `updated_at_utc`, but repeated upserts advance that timestamp on previously known facts. A historical replay can therefore lose context that actually existed at the time, while historical training can select later updates. Injury fallback to any earlier week also needs explicit status expiration/confirmation.

The exact-line trainer permits an odds observation up to five minutes after `p.created_at_utc`, so its alleged lock features may come from the future relative to the decision.

Repair: store immutable observations with source-effective time and first-observed time. Require both to be eligible at the decision cutoff. Join exact locked offer IDs, not nearest future prices. Treat retrospective closing/weather datasets as separately labeled research inputs until genuine as-of observations are available.

### F07 [P1] Reruns rewrite and delete the records used as forecast evidence

Sources: [prediction upsert/delete](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:3239), [generic model labels](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:3338), [ledger synchronization](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/lock_ledger.py:71), [ledger settlement](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/lock_ledger.py:226).

Prediction upserts replace price/projection/probability/model fields but retain the first `created_at_utc`. Rows absent from the new card are deleted. Ledgers can then void old selections because the current card changed, and later revive voided rows with a new lock. This is unsuitable as an immutable forecast or execution ledger.

Observed at audit time:

- 305 prop ledger rows; **69 reference deleted predictions**.
- Among 236 surviving references, **14 prediction prices differ from the ledger lock**.
- **186 of 241** current offer-bearing predictions had been updated after kickoff despite being created before kickoff. An update alone does not prove that the forecast value changed, but the code permits it and preserves no prior revision.
- Stored model versions are generic `trained_workload_adjusted` / `baseline_fallback`, not unique model releases.

Ledger settlement uses the mutable prediction result's unit profit rather than calculating from the ledger price. No incorrect winning payout was found in the current rows, but the code permits one when a winning prediction's price changes. Simulated selections also are not evidence that the user actually placed a wager.

Repair: separate latest display cards, immutable forecast revisions, immutable executable decisions, and confirmed executions. Lock probability, distribution, price, offer ID, timestamp, model/data release, and stake. Grade and calculate CLV/P&L from the locked row. Supersede displayed cards without voiding historical decisions or real wagers. Require distinct execution confirmation for claims of realized cash ROI.

### F08 [P1] Final-result refresh and finalized-game validation are incomplete

Sources: [close runner](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/run_close_and_grade.py:220), [PBP player updates](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/import_usage_context.py:691), [prop grading](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/grade_predictions.py:117), [training query](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_models.py:48).

The close runner grades without first refreshing base game results. Current PBP usage can update player counts independently of the schedule. Prop grading joins player rows without a verified final-game/participation check; training likewise has no explicit finalized-game predicate. A partial source update could be treated as a final outcome.

Observed: player gamelogs cover 16 current-season games, but only 14 schedule games have scores. `2026_01_DAL_NYG` and `2026_01_DEN_KC` lack schedule scores; the former already has 14 prop result rows labeled win/loss. This confirms inconsistent result synchronization, not that those particular player outcomes are necessarily wrong. Raw `status='REG'` is a season-type label, not final-game verification.

Repair: ingest authoritative final state and participation before grading/training; handle DNP/void/push rules explicitly. Use idempotent versioned corrections for official stat changes. Schedule an end-of-game/end-of-slate result refresh separately from pre-kickoff close capture.

### F09 [P1] A green scheduled task does not mean the required pipeline succeeded

Sources: [daily criticality flags](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/run_daily_and_notify.py:132), [Discord delivery](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/run_daily_and_notify.py:76), [provider aggregate status](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/crawler_oddsapi.py:1057), [mutex wrapper](C:/Users/josh/Git/SuperNovaBets/scripts/run_with_nfl_mutex.ps1), [close task](C:/Users/josh/Git/SuperNovaBets/scripts/tasks/NFL-Close.xml).

Odds ingestion/parsing, predictions, ledger writes, and reports are marked noncritical in the daily runner. A provider chain can return JSON `status='failed'` while its CLI exits successfully. Discord non-success responses are logged but do not fail the run. Thus Windows result 0 is not an end-to-end health guarantee.

All installed NFL tasks last reported 0, while current prop CLV and odds coverage remain poor. The task wrappers also share a global operational mutex; a long training/refresh run can delay close capture. Close's execution limit is 30 minutes, while the mutex can wait longer. Actual missed windows due specifically to contention were not established in this audit.

Repair: distinguish required operations, optional research, expected no-market days, partial-provider coverage, and failures with structured completion records and exit codes. Require successful ledger persistence before publishing an executable card. Keep near-kickoff capture independent of training, with bounded lock waits and meaningful skip status.

### F10 [P1] CLV proof is mostly missing and timing rules allow post-kickoff closes

Sources: [close quality](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/clv_report.py:21), [prop close matching](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/clv_report.py:197), [offer parsing timestamps](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/parse_oddsapi.py:106).

Stored prop CLV rows:

| Date | CLV records | Valid close records |
|---|---:|---:|
| 2026-09-09 | 34 | 0 |
| 2026-09-10 | 31 | 9 |
| 2026-09-13 | 264 | 0 |

These are raw stored-record counts, not a deduplicated executable-bet coverage denominator. They are still far from adequate proof. At query time no September 14-17 prop offer snapshots were present.

Close eligibility allows through five minutes after kickoff and treats equal lock/close timestamps as valid. Matching relies on date/name/stat/book/line rather than an immutable decision-to-event/offer identity. A generic book snapshot is also insufficient to prove that a missing exact line was truly unavailable, rather than omitted by a partial feed. Parser capture time alone does not prove the source price was freshly updated.

No existing stored valid close was after kickoff in the audit query; the timing defect is a permissive code path, not a demonstrated in-play contamination incident.

Repair: use actual event identity and strictly pre-kickoff executable source observations after the immutable lock. Preserve source quote time separately from fetch time. Separate unavailable line, incomplete market response, stale source quote, missing capture, and timing miss. Measure coverage only over eligible locked offers, with per-game retries and explicit provider freshness/completeness checks.

### F11 [P1] Game prediction lacks the prop path's lock-time boundaries

Sources: [game odds load](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_today.py:280), [game prediction loop](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_today.py:684).

The game line loader selects latest rows without the prop path's cutoff/snapshot-role constraints. The game prediction loop does not exclude already-started games. A historical or late-day rerun can therefore use close/later odds and overwrite a supposed pregame card. This prevents an honest replay even though games remain paper-only.

Repair: require an explicit decision cutoff, reject in-progress games for pregame publication, and persist separate replay runs without touching previously locked forecasts. Add a test where a newer post-kickoff price is present but must not influence a pregame prediction.

### F12 [P2] Model releases are not atomic or coherently versioned

Sources: [player model saves](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_models.py:3616), [game insufficient-data saves](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_game_models.py:656), [distribution saves](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_distributions.py:1221).

Models/reports are written directly to active paths. Some insufficient-data branches overwrite previous artifacts with empty model payloads. A failed or interrupted multi-step training run can leave a mix of old and new opportunity/stat/distribution artifacts. A scheduler mutex reduces overlap between scheduled jobs, but does not prevent partial writes or mismatched releases.

Repair: write candidate artifacts to a versioned release directory, validate checksums/schema/feature contract and cross-artifact compatibility, then atomically switch one manifest pointer. Preserve the last known-good release on failure or insufficient data. Record release and training cutoff on each forecast.

### F13 [P2] True route and first-read data remain unavailable

Sources: [usage ingestion](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/import_usage_context.py), [live history fields](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/predict_player_props.py:1019).

Database coverage is better than older 'everything is zero' summaries suggested, but key gaps remain:

- 2026: 312 player rows; 275 have offense snap share; all 312 have non-null target share, air-yards, and red-zone target fields.
- True routes and first-read fields: zero populated rows in the inspected 2024 and 2026 cohorts.
- 2024 has route proxies, but the current-season participation-route source logged a 404 and no route-proxy coverage was present in the checked current cohort.

Non-null count is not the same as complete or accurate usage data; some values can be legitimate zeroes or derived values. Do not label target/snap proxies as observed routes/first reads.

Repair: retain source/quality/missingness for every usage feature, avoid zero-filling unknown measurements, and use target/snap/air-yard models as the explicitly named fallback until a permitted route feed is working. Evaluate whether each feature group helps on the same folds before adding more heuristic interactions.

### F14 [P2] Tests and architecture do not protect the important statistical contracts

Sources: [safeguard tests](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/test_nfl_pipeline_safeguards.py), [receiver mixture tests](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/test_receiver_spike_mixture.py), [feature builder](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/features.py), [player trainer](C:/Users/josh/Git/SuperNovaBets/src/nfl_pipeline/modeling/train_player_stat_models.py).

All 19 tests pass, but some safeguards assert source-text presence rather than behavior. There is no adequate regression coverage for spread sign, missing seasons, offer-independent roles, live/historical feature equality, lock-time eligibility, finalized-game grading, or immutable ledger behavior. Repeated v2/v3/v4 feature formulas and duplicated scoring paths make parity defects likely. Live feature construction also emits many pandas fragmentation warnings.

Repair: prioritize small behavior tests and a fixture-driven end-to-end replay test. Consolidate shared feature/scoring contracts before introducing more versions. Batch feature column construction after correctness is established. Use coverage and execution evidence before deleting apparent dead code; this audit does not claim every unused helper was exhaustively classified.

## What Is Working

- These are genuine fitted ML models, not only hand-ranked picks; the code also contains many heuristic feature/adjustment layers.
- Training generally uses player-game rows and chronological splits rather than random offer-level splits. The remaining issue is reuse of the same holdout and as-of parity, not a wholly random split.
- Player history excludes the current ET date, which prevents same-day result leakage through that loader. It is conservative rather than a complete per-game timestamp solution.
- Prop odds parsing groups over/under prices by book/player/market/exact line within an event response; it does not inherently fabricate missing opposing prices in that parser.
- The recent receiver mixture did not replace the incumbent after its measured MAE/Brier comparison failed.
- NFL game output explicitly stays paper-only; exact-line training has minimum sample requirements. Those gates do not repair the underlying data/evaluation defects by themselves.
- The shared subprocess utility has a Windows process-tree termination path. The remaining operational issues are truthful success status, source/result freshness, and capture scheduling isolation.

## Repair Order

1. **Restore correct input semantics and complete history.** Repair the spread convention, backfill 2025, refresh authoritative results, validate coverage, and reconstruct the historical feature table. Re-score the simple baselines before judging any challenger.
2. **Make today's features identical to replay features.** Build full-roster player-game snapshots before odds, compute volatility/depth movement consistently, fix team-game counts, and feed current injury context into game models. Freeze a feature schema with missingness/provenance checks.
3. **Preserve evidence.** Introduce immutable forecast/decision/release IDs, repair ledger references and settlement rules, enforce final-result/participation rules, and stop reruns from replacing past evidence.
4. **Repair evaluation.** Use nested expanding folds, OOF upstream predictions for downstream training/calibration, a final untouched test, and clustered confidence intervals. Report baselines chosen without test-label access.
5. **Reassess models on corrected data.** Compare receiver target volume, air-yards role, and YPT tail separately; QB attempts and YPA separately; RB carries and efficiency separately. Improve distributions using proper scoring rules and calibrated line probabilities, not MAE alone.
6. **Restore operational truth and market evidence.** Separate close capture from expensive training, surface provider/Discord/ledger failures, preserve exact lock/close identity, and verify source freshness. Train exact-line models only from those immutable graded decisions.
7. **Release and measure.** Publish an atomic accepted release, hold its identity stable across prospective dates, and evaluate it against a fixed incumbent. Do not infer profitability from a handful of wins or from lower count MAE alone.

## Acceptance Tests for the Repair

- A home favorite has a negative canonical home handicap, positive expected margin, and higher implied points; home plus away implied points equals the total.
- The training input contains the expected seasons and finalized player-game coverage. Missing 2025 is a failing test, not a warning.
- Removing another player's odds does not change the player's forecast, target share, role rank, or workload features.
- Historical replay and live construction produce equal model inputs for an identical observation cutoff, including volatility and depth deltas.
- Adding post-lock context or post-kickoff odds cannot change a previously locked forecast or its training features.
- Altering final-test outcomes does not change a fitted model, calibration, selected feature set, or mixture parameter.
- Rerunning predictions cannot delete a ledger-referenced decision or change its price/probability/model identity.
- A revised display card cannot void a confirmed placed bet. P&L uses the executed/locked price, not the latest card price.
- Partial games do not create final labels; official corrections are versioned and reproducible.
- A failed required ingestion, prediction, ledger, or notification step cannot produce a green daily run.
- Closes after kickoff or with unproven source freshness cannot become valid pregame CLV evidence.
- Artifact write failure leaves the previous complete model release loadable.

## Limits

No paid/live odds requests, retraining, model deployment, scheduler edits, or Discord sends were performed for this audit. Provider entitlements and whether a currently absent market is genuinely unpublished were not re-tested against external APIs. Database counts are a point-in-time snapshot and scheduled jobs may continue updating them. Code-path risks are explicitly distinguished above from observed bad rows. The recommendations improve correctness and the ability to measure accuracy; they do not guarantee profitable bets.
