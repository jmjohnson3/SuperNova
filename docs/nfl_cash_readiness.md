# NFL Forecast and Cash Trial Contract

## What Changes

Production remains pinned. Forecast deployment is still a separate, manual review;
cash eligibility is computed automatically, not inferred from a legacy `micro_projection`
or `bankroll` tag. There is no automated wager placement.

The first authorization cohort references the existing immutable FanDuel common-line
receiving trial, plus an immutable scoring fingerprint, model artifact checksum and
authorization timestamp. Earlier rows remain visible in research reports, but cannot
be treated as prospective evidence of a newly registered cash policy. Changing a
model, selection rule or scoring version cannot silently pool the old cohort.

Other NFL props, spreads and totals remain research. The global risk policy supports
both game and prop ledger identities, but this initial adapter admits only the pinned
receiving trial. Registering other strategies requires their own prospective captures,
distribution validation and selected-decision evidence; it is not an environment toggle.

## Evaluation

Reviews are frozen at 50, 100 and 200 settled decisions, with at least three NFL weeks.
Counts are review floors, never automatic evidence of profitability. Early insufficient
week coverage consumes that checkpoint as a failure rather than creating daily retries.
Missing selected results or immutable inputs prevent checkpoint consumption. Binary
calibration excludes pushes; flat-stake ROI includes refunded pushes. Missing closes
remain unknown and actual unchanged closes are reported separately.

Selected Brier improvement versus FanDuel and ROI use one-sided equal-week Student
intervals. Alpha is 0.05 / (8 strategies * 3 reviews * 2 tests). This is a conservative
preregistered parametric screen, not a profit guarantee; three week clusters provide
particularly weak information. No daily re-selection or daily significance-based approval.
Each checkpoint stores its forecast IDs and evidence checksum. Subsequent monitoring
can pause a passing strategy, never manufacture a new approval. Pauses are sticky.

Additional gates: selected ECE <=5%, nominal 80% coverage 75-85%, no all-offer Brier
regression, >=90% valid closes, <=2% stale closes, positive average CLV, >=55% CLV beat,
and player/team/week concentration <=20/35/50%. All-offer scores weight each player-game
equally. Legacy inputs are never reconstructed. Game market fallback predictions do not
establish independent skill.

Forecast research continues to use the existing shared chronological benchmark:
expected yards RMSE/bias, typical outcome MAE, and probability Brier/calibration/interval
score. Receiving conditional uncertainty, downside models and conservative ranking
remain challengers until accepted. Passing/rushing opportunity-efficiency challengers,
TD heads, and game challengers do not inherit receiving cash authorization.

## Live Flow

Fresh FanDuel quote -> frozen forecast -> immutable ledger -> pinned challenger capture
-> fixed research selection -> cash readiness -> reservation -> Discord -> exact close
-> settled result -> evidence refresh. The live runner refreshes readiness before each
publication; the close runner grades confirmed executions and refreshes readiness after
settlement. No schema DDL is run by the new readiness or execution paths.

The publication boundary converts all legacy cash-like tiers to research. A valid
authorization, unchanged artifact, current exact quote, positive final EV, drift pass,
true pair and FanDuel link are required to reserve a cash row. Cash probability is the
approved shadow's probability on the original side, not a replacement of production's
saved forecast. Its final probability, original forecast ID and approval hash are stored
in append-only execution events on the existing ledger. Previewing cards does not reserve
capacity. Cash cards expose the locked price, minimum price, expiry and ledger ID.

Reservations across all NFL strategies share a PostgreSQL transaction advisory lock:
$1 flat, at most five recommendations/day and $20 staked/NFL season-week. Uncertain
Discord sends retain capacity. Expired unconfirmed reservations block new cash rows;
they are never silently assumed unplaced. A $30 cumulative confirmed net loss creates
a sticky global pause. Off-policy recorded executions also require review.

## Setup and Reconciliation

The implementation setup runs these once; scheduled jobs only evaluate existing policy:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.cash_execution --setup
.\.venv\Scripts\python.exe -m nfl_pipeline.cash_readiness --register --date 2026-09-26
```

After actually placing a cash recommendation, record its executed price and stake:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.cash_execution --ledger-id 123 --action confirmed --price -110 --stake 1
.\.venv\Scripts\python.exe -m nfl_pipeline.cash_execution --ledger-id 124 --action not_placed
```

These commands record user actions; they never place bets. Actual price/stake deviations
are retained, not filtered out of the results. Grading confirmed executions uses the
actual recorded price. Existing simulated micro history is not converted into cash history.

## Known Evidence Limits

The current release still needs independent prospective weeks. No cash approval is
granted by installing this code. September 24 current FanDuel offers captured closes,
but many earlier exact lines vanished from the provider. That historical coverage gap
is not repairable by inventing closes or switching books. Projection-only Discord rows
are audited as projections, not incorrectly required to have betting prices.
