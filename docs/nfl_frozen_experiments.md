# NFL prospective betting experiments

These changes repair evidence collection. Profitability still has to be demonstrated
on later offers, and paper results do not authorize cash betting.

## Registered October 21, 2026: `nfl-sharp-gap-cash-v5`

The price-gap scanner is now the cash strategy (`docs/nfl_cash_readiness.md`). `nfl-sharp-gap-cash-v5`
succeeds `nfl-fanduel-price-gap-v4` with new rules:

- bets at FanDuel, ESPN Bet and DraftKings;
- stakes only gaps seen 15-45 minutes before kickoff, polled every 5 minutes through the final hour;
- one decision per player-game-stat across books;
- the documented cash limits;
- a shadow model filter;
- EV at the sharp close graded on Pinnacle's nearest closing line moved to the bet's line when it has
  moved (yardage only).

**Why v4 collected nothing.** It last collected a decision on October 15. `bets.nfl_sharp_quote_observations`
and `bets.nfl_sharp_observations` were empty, so no decision had a close. The scanner refused every run,
and nothing reported it:

| Dates | Refusal | Cause |
|---|---|---|
| Oct 16, Oct 18 | `research_not_registered` | v2-v4 did not exist yet. |
| Oct 18-20 | `experiment_configuration_drift` | v4 was registered with one market, but the scheduled task's `NFL_SHARP_WATCH_MARKETS` lists four. |
| Oct 21 | `experiment_source_drift` | An edit to `receiving_paper_trial.py` drifted v4. |

That file has nothing to do with pricing. v1-v4 pinned the scanner's whole import closure: 79 files,
from the model trainers to the cash ledger. A display helper in `fanduel_links` and `schema`'s import
of `cash_execution` pull in the rest.

**What v5 changes.**

- **Narrow pin.** v5 pins only the modules that price, select, stake and grade: `sharp_watch`,
  `sharp_strategy`, `sharp_math`, `markets`, `context_contract` and `modeling/sharp_alert_report`.
  Transport, raw storage, prop parsing and link display are not pinned.
- **Outage warning.** A refusal while a game is within 24 hours now posts one Discord warning per
  day and reason.
- **Kickoff tolerance.** In the final minutes The Odds API moves `commence_time` to the actual
  kickoff (PHI-JAX on October 11: 13:30Z became 13:35Z). The exact-kickoff check rejected those
  payloads, which are the closing snapshots. The same matchup within 15 minutes is now accepted.
  In the replay this lifted close coverage in the stake window from 89% to 97%.

v1-v4 remain blocked and their observations are not pooled. Registered with the scheduled task's
four markets (`player_reception_yds,player_rush_yds,player_receptions,alternate_totals`).

## Registered October 19, 2026

Two repairs were made in sequence, so each trial has two new registrations. Only
the v4 pair is live:

- `fanduel-receiving-oe-paper-v4` (v2 -> v3 for the quote repair, v3 -> v4 for the
  resilience fix)
- `nfl-fanduel-price-gap-v4` (same sequence)

Both predecessors pinned a `predict_player_props.py` whose quote `SELECT` omitted
`provider`. That column is required: `receiving_paper_trial.price_record` rejects a
quote without it as `missing_quote_provider`, `paper_operations` filters such quotes
out as stale, and `test_quote_provider_capture` asserts the loaded row carries it.
Adding the column was therefore a repair, not drift — so the pinned versions could
never have priced a record, and restoring them was not an option.

Because the registry is create-once, the repair required successor identifiers.
Pricing and selection rules are unchanged in both: the scanner configuration was
compared against `sharp_watch.experiment_config()` and found identical before
registering, and the receiving rules come from the same `registration_config`.
Only the pinned file contents differ.

The predecessors are superseded before any prospective decision observations were
recorded. At the time of the bump, `fanduel-receiving-oe-paper-v2` had priced zero
of 962 candidate records (661 pre-registration locks, 301 outside the trial), so no
evidence was lost and none is relabelled or pooled. The v3 pair existed only for
minutes, across no live slate, and likewise recorded nothing. v1 through v3 remain
intentionally drifted and blocked; only the v4 registrations collect new evidence.

The same drift had made the gate fatal well beyond the trial. `generation_attestation`
is reached through `stat_model_registry.score_component` inside `build_predictions`,
and `sharp_watch.run` checks its own experiment unguarded, so one dead column stopped
every player prediction and the whole price-gap scanner — not just the paper capture.
Candidate components are capture-only and production projections are final before
they are scored, so a research shadow now degrades instead of aborting.
`_score_stat_candidates` withholds a component that fails its frozen check and logs
a warning, and `sharp_watch.run` reports drift as an `experiment_source_drift` status
row like its other refusals rather than raising into the task runner. Active stat
components stay fatal: they move production projections. Both behaviours are covered
by regression tests. Pinning that fix is what required v4 — the fix edits sources
that v3 pinned, and leaving v3 drifted would have left the scanner declining to scan.

Because `scoring_versions.FILES` hashes `predict_player_props.py`, the fix also moved
the scoring fingerprint. `scoring_capture` archives the new source on first capture
and `archived_candidate` still replays earlier captures from their own archives, so
no stored forecast becomes unreplayable; evidence simply does not pool across the
change, which is the point of the fingerprint.

Verification: both v4 registrations report clean checksums and matching runtime
versions; the three gates that were raising now pass, including the exact
`generation_attestation` call from the failure; `sharp_watch.run` reaches its normal
`no_games_within_24h` return; a full player-prediction run for the next scheduled
game completes and writes forecasts; and 92 registry, paper-trial, quote-capture,
stat-registry and watcher tests pass.

The legacy cash registration is still drifted by the same column and was left alone
— bumping a cash registration is a separate decision. Its check is caught in
`cash_publication.prepare` and degrades to an operational issue, and `NFL Cash
Readiness` is non-critical, so cards are unaffected; but `paper_collection.guard_freezes`
still raises on it.

## Registered October 18, 2026

The following immutable registrations were created and verified locally:

- `fanduel-receiving-oe-paper-v2`
- `nfl-fanduel-price-gap-v2`
- `legacy-cash-receiving-v1-20261018`

The legacy cash registration now matches the repaired scoring implementation and
references its dependency freeze. Previous cash registrations remain archived;
their observations are not pooled into the new cohort. All three registrations
start in research, with no betting approval.

The initial price-gap v1 registration is preserved but superseded before any
prospective decision observations were recorded. Final verification caught its
unnecessary dependency on the daily runner when task timeouts were repaired.
Version 2 separates notification delivery from orchestration and pins the final
reporting layer. Its pricing and selection rules are unchanged. The registry's
all-history listing intentionally reports v1 as blocked; only v2 collects new
price-gap evidence. No old evidence is relabeled.

The same repair also addresses the October 18 operational failures. Scheduled
receiving checkpoints read the requested date and pinned benchmark run instead
of repeatedly evaluating the whole historical corpus. They explicitly identify
date-only reports and cannot use those reports as cumulative deployment proof.
Missing original captures still appear as evidence failures. Daily research
failures no longer stop paper cards, and a cash-readiness operational issue no
longer rejects projection/status cards with no staked instructions. Cash cards
remain blocked by the cash checks.

The full cash review took about 9.5 minutes in the local verification run. Its
daily and close-task budget is now 15 minutes; it stays outside the pregame
critical path. Receiving paper capture has a six-minute task budget. Quote
freshness requirements remain unchanged.

Verification included 100 receiving/scoring tests, 53 cash/reconciliation tests,
48 sharp-price tests, 71 runner/evidence tests, 16 checkpoint tests, 17 Discord
format/publication tests, and five registry tests. A local replay of the saved
seven-card bundle used mocked delivery. No duplicate Discord messages were sent.
After the final transport extraction, 38 watcher/delivery tests and four targeted
runner tests passed. All three active registrations passed their checksum checks.

## Receiving opportunity and efficiency

`fanduel-receiving-oe-paper-v2` uses the registered receiving OE v2 challenger on
the same real FanDuel offer as production. Its original full-roster distribution
prices both sides, including pushes, and chooses its own positive-EV side at the
executable price. Production's side, EV and drift verdict are diagnostics, not
eligibility requirements. Quote freshness, identity, participation/context,
paired prices and links remain required.

The first eligible player-game offer is retained, including unselected offers.
Selection fills the fixed three-per-date and two-per-game caps; later revisions
cannot replace earlier observations or winners. Reports compare challenger,
production and market probabilities on identical rows, separately for all
observations and the fixed selected subset. Exact closes and unknown closes stay
distinct. Original research captures and cash selections remain historical.

The daily and pregame runners capture this experiment after player predictions.
The close runner refreshes its evidence report after grading. Source, artifact,
runtime or active-component drift blocks new evidence; a scheduled run never
silently registers a successor or deploys a challenger.

## FanDuel price discrepancies

`nfl-fanduel-price-gap-v2` is the primary price-gap experiment. It preserves both
books' update timestamps and immutable quote observations. The primary cohort
uses FanDuel receiving-yard offers against an exact-line Pinnacle reference.
Missing, future, stale or excessively separated source timestamps are rejected.
Inferred line conversions and historical mixed-book results remain diagnostics.

Repeated observations of one player-game are correlated. The report retains the
first eligible decision and reports repeat counts and uncertainty grouped by
week. Closing evidence must be an observed, timely exact-line pair. An API quote,
historical snapshot or paper win cannot establish that a wager was executable.

The existing sharp watcher collects observations. The close runner refreshes the
report without making provider requests. Actual attempts and receipt prices are
recorded separately in the private reconciliation journal.

## Freeze and review

Registrations live under
`src/nfl_pipeline/modeling/models/frozen_experiments/`. They pin source checksums,
rules, model artifacts, optional calibration presence/absence and numerical
runtime versions. Small source files are archived beside each registration.
Production model deployment and cash authorization remain separate.

Develop new candidates alongside the frozen versions. A changed dependency is a
blocked experiment, not permission to relabel old evidence. A material strategy
change needs a distinct explicit prospective registration and start time. Keep
old registrations and observations. Automatic stat-component and distribution
policy activation have been removed from the scheduled runners.

Read-only checks and derived reports:

```powershell
.\.venv\Scripts\python.exe -B -m nfl_pipeline.experiment_registry
.\.venv\Scripts\python.exe -B -m nfl_pipeline.experiment_registry --check nfl-fanduel-price-gap-v2
.\.venv\Scripts\python.exe -B -m nfl_pipeline.modeling.receiving_paper_trial report
.\.venv\Scripts\python.exe -B -m nfl_pipeline.modeling.sharp_evidence_report
```

These commands do not fetch sportsbook quotes, send Discord messages or place bets.
The reports may write derived local files and preserve newly verified close evidence.

## Actual wagers and costs

`data/nfl_wager_reconciliation/` is private, gitignored storage. Imports append
immutable events, reject duplicate tickets and conflicting settlements, and keep
paper results outside actual P&L. Missing records mean unknown execution.

Generate the field guide, then create a JSON import document containing only
your actual records:

```powershell
.\.venv\Scripts\python.exe -B -m nfl_pipeline.wager_reconciliation --template
.\.venv\Scripts\python.exe -B -m nfl_pipeline.wager_reconciliation --validate data/nfl_wager_reconciliation/receipts.json
.\.venv\Scripts\python.exe -B -m nfl_pipeline.wager_reconciliation --import-json data/nfl_wager_reconciliation/receipts.json
.\.venv\Scripts\python.exe -B -m nfl_pipeline.wager_reconciliation --report --include-db --output data/nfl_wager_reconciliation/reports/latest.json
```

Each placement needs its sportsbook ticket, exact bet description, recommendation
ID, actual time, American price, stake, currency and receipt reference. Settlement
records use the actual cash returned **including returned stake**. Execution
attempts record availability or rejection and can link the immutable sharp
observation ID. Record data/API and other NFL operating costs separately.

Net profit stays unknown until an explicit complete-cost attestation covers the
reporting period, including a genuinely zero-cost period. Reports separate
currencies and use UTC settlement and cost dates. Optional database cross-checks
do not add legacy recommendations to receipt P&L. Bonus bets need separate
accounting and are not supported by this cash-stake journal.
