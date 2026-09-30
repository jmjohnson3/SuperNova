# NFL Audit Repairs and Verification

Date: 2026-09-18. Active release: `nfl-20260918T143342Z`.

## Outcome

The September 20 daily pipeline completed with all 15 executed steps returning zero. Game and player Discord sections were successfully posted from persisted forecasts after ledger writes. No wagers were placed.

- 14 upcoming games; 56 game forecast rows.
- 640 player forecast rows: 5 micro-test, 328 paper, 307 projection-only.
- Ledger: 56 paper game rows, 242 paper prop rows, 5 simulated $1 micro recommendations.
- A repeated micro-ledger call inserted zero additional rows.
- Database checks found zero new ledger orphans/price-identity mismatches, zero versioned props locked after kickoff, and zero selected micro rows missing a paired price, link, or positive calculated EV.
- A transactional attempt to alter a locked forecast price was rejected by the database trigger and rolled back.

## Audit Disposition

| Finding | Repair and Remaining Scope |
|---|---|
| F01: Spread sign | NFLverse home-favored signs are converted to the bookmaker handicap convention. Regression-tested. |
| F02: Missing 2025 history | Imported the new nflverse weekly player-stat endpoint, restored 2025 roster mapping and snap data, rebuilt features, and retrained. Final training used 34,806 eligible player-game rows and 7,293 game rows. |
| F03: Offered players changed team features | Role/snapshot construction precedes offer selection. Offer-subset invariance is tested. New production heads exclude historical constructed share inputs whose as-of denominator cannot be established. |
| F04: Feature parity | Added rolling volatility, actual team-game counts, prior depth context, and shared injury-context loading. Materialized context observations and precomputed counts reduce live query/feature overhead. Full consolidation of legacy feature builders remains cleanup work. |
| F05: Reused holdout | Scheduled training now uses a fixed, conservative candidate with expanding 2025 week folds, target-specific audited features, and train-only imputation. Earlier-fold residuals supply later-fold distribution evaluation. 2026 is a later veto, not an untouched test set or another hyperparameter search. |
| F06: Context/odds leakage | New immutable context observations support as-of cutoffs. Prediction inputs and cutoff are saved. Exact-line joins use the locked offer ID and prohibit quotes fetched after prediction time. Unverifiable old context is excluded from the new training feature contract. |
| F07: Mutable forecasts/ledger | Forecast revisions append rather than overwrite/delete. Database triggers protect locked forecast fields. Ledger deduplication and daily caps survive reruns; result refresh uses the ledger's locked price. Historical orphaned/overwritten legacy records were not fabricated. Previously locked micro rows remain visible as historical decisions, not new plays. |
| F08: Incomplete results/participation | Daily/close/training paths refresh final results. Training/history/prop grading require finalized games and verified offensive participation. A zero-stat participant can be graded; an unverified appearance remains pending review. Book-specific DNP settlement still needs explicit rules. |
| F09: False operational success | Failed important steps and Discord errors produce nonzero runs; local run manifests preserve statuses. Prediction publication follows persistence and ledger steps. Close/training mutexes no longer contend with the daily mutex. Correction to the original audit: the odds crawler CLI already returned nonzero for a failed crawl status. |
| F10: CLV identity/timing | Exact provider/event/book/player/stat/line matching uses the immutable lock offer. Close must follow lock, precede kickoff, and be within 120 minutes. Invalid/legacy evidence does not get a numerical CLV value. Missing capture is unknown, not proof of unavailability. Source-side quote timestamps are retained in raw SGO responses but are not yet normalized into a separate quote-freshness contract. |
| F11: After-start forecasts | Both game and prop scoring/persistence/publication reject started games. The default daily date selects the next future slate. |
| F12: Artifact writes | Production artifacts and manifests are versioned, atomically written, and checksum-verified. Each daily run pins one release. The scheduled exact-line trainer now also saves JSON/joblib atomically. Legacy offline research trainers are not all migrated. |
| F13: Missing usage/provider data | Corrected all-null usage SQL types and backfilled historical usage. SGO bookmaker IDs, nested team names, availability handling, and game-vs-team-total parsing are fixed and live-tested. True 2026 routes/first-read data remain unavailable from the attempted participation source. TheRundown canonical normalization remains incomplete; The Odds API quota was exhausted. SGO supplied the successful live run. |
| F14: Tests/architecture | Added behavioral tests for feature invariance, spread semantics, walk-forward purging, timing, atomic failure, persistence, Discord failure, provider shape, participation, and honest proof states. Large legacy modeling modules and pandas fragmentation warnings remain cleanup work. |

## New Model Evaluation

Lower MAE is better. These are expanding historical fold results, not prospective profits.

| Stat | Model MAE | Rolling Baseline MAE | Live Projection Decision |
|---|---:|---:|---|
| Passing yards | 68.70 | 72.48 | Accepted |
| Rushing yards | 17.80 | 18.45 | Accepted |
| Receiving yards | 16.63 | 17.54 | Accepted |
| Passing TD | 0.91 | 0.94 | Baseline; probability acceptance gate failed |
| Rushing TD | 0.38 | 0.38 | Any-TD probability head accepted, not a claim of better count MAE |
| Receiving TD | 0.31 | 0.29 | Any-TD probability head accepted; count MAE is worse |

The receiving/rushing TD heads are evaluated on P(any TD), not as evidence that their count estimates or alternate TD lines are superior. Spread and total challengers failed expanding-fold baseline comparisons and remain on baseline forecasts. Game baselines use market lines and must not be described as independent model edge.

Line-level Brier checks use predefined proxy thresholds, not historical executable sportsbook offers. The 2026 outcomes have been examined in earlier research. New prospective results are still required before claiming improvement in bet selection or profitability.

## Operational Verification

- `python -m pytest src/nfl_pipeline -q --disable-warnings`: 48 passed; 78 existing pandas/related warnings remain.
- `python -m compileall -q src/nfl_pipeline`: exit 0.
- Full daily command: `python -m nfl_pipeline.run_daily_and_notify --date 2026-09-20 --skip-train`, exit 0.
- Close/result workflow for September 17: exit 0. No retrospective close was invented; no crawl was attempted outside the close window, and there were no forecasts for that date to grade.
- All-date result refresh graded 60 game and 237 prop forecast records. This does not turn legacy records into clean training evidence.
- CLV reclassification processed 172 game and 488 prop forecast records, preserving unknown/invalid states.
- Exact-line trainer has zero eligible settled rows under the new contract and reports `waiting_for_prop_history`.
- The exact-line proof report labels Sunday's 244 true-paired versioned decisions `waiting_for_settlement`, not ready. Any-TD acceptance does not approve alternate TD lines for micro selection.
- Task Scheduler: NFL-Close, NFL-PrimeTime-Refresh, and NFL-Training last reported 0. NFL-Daily still shows the earlier failed morning invocation; the successful manual rerun does not rewrite Task Scheduler history.

## Still Needed

1. Prospective lock/close/results for this stable release. Sunday has not happened yet, so zero valid closes for that slate is expected.
2. Actual route/first-read data with an observable publication time; do not label proxies as measured routes.
3. Independent prospective validation of exact-line probabilities, calibration, ROI, and CLV. Micro recommendations are experiments, not bankroll approval.
4. Complete TheRundown normalization and normalized source-quote provenance.
5. Book-specific settlement rules, stronger report isolation by slate, and further legacy-module/test cleanup.

No full-bankroll readiness or guaranteed profitability is asserted.
