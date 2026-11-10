# Automatic paper price checks

The NBA and NFL scheduled scanners now run `automatic_price_checks` immediately
after delivering the model-comparison alerts and before slower research reports.
Keep the computer awake and online during the collection windows.

The observer reads FanDuel's current Colorado public feed. It does not log in,
place wagers, verify account restrictions or establish that a bet would be
accepted. It consumes no Odds API credits. Historical snapshots are never used.

## What is recorded

For each original FanDuel alert, the separate ledger stores:

- Original alert ID, delivery time, player/game, market, side, line and price.
- The first attempt time, response time, save time and response receipts/hashes.
- The current main-market line and price, even when the odds are worse.
- Available, changed line, suspended, not listed, or unknown—with the reason.
- Whether the observation and recording met the five-minute limits before kickoff.

An offer is `not_listed` only when its main market is absent from a fully loaded
tab. That does not assert that every alternate market or account has lost it.
A blocked page, stale response, incomplete market, ambiguous identity or network
failure stays `unknown`. A missed deadline is recorded without fetching a later
price and calling it timely. The first attempt is claimed before making network
requests; a crash cannot be silently replaced by a more favorable later quote.

The fixed v2 observer uses the main market only, even when an alternate market
has a better price. Conflicting main quotes stay unknown. The original frozen
NBA v1 observer continues separately for comparison.

## Frozen comparisons and review

`fanduel-automatic-paper-v2` was registered prospectively on November 6, 2026.
Its registration is in
`src/model_research_experiments/fanduel-automatic-paper-v2/registration.json`.
No historical manual checks or v2 observations were backfilled.

The NBA price experiment, NFL experiment, model selections, receiving challenger
and their registrations remain unchanged. NBA alerts linked to an M comparison
are checked once using the original # ID. NFL/game M alerts are checked directly.
Model vetoes remain in the paired price-only control. The report values the same
first public price against an exact reference close and reports game/day-clustered
comparisons. These are reference-based EV diagnostics, not realized profits.

Automatic observations never enter `nba_paper_price_checks` or
`model_research_checks`. They cannot satisfy the frozen manual/account execution
requirements. The existing NBA review still needs 21 days, 20 games, 50 checked
prices paired with exact closes, coverage, availability and uncertainty checks.
NFL is evaluated separately. Counts never authorize stakes.

## Reports and commands

Scheduled runs refresh these files:

- `reports/automatic_price_checks_nba_latest.md` and `.json`
- `reports/automatic_price_checks_nfl_latest.md` and `.json`
- `reports/paper_collection_review_latest.md` and `.json`
- `reports/price_survival_review_latest.md` and `.json`
- `reports/collection_economics_latest.md` and `.json`
- `reports/low_credit_assessment_latest.md` and `.json` (offline proposal only)
- `data/sports_finance/reports/weekly_latest.md` and `.json` (private receipts)

```powershell
.\.venv\Scripts\python.exe -B -m automatic_price_checks verify
.\.venv\Scripts\python.exe -B -m automatic_price_checks once --sport nba
.\.venv\Scripts\python.exe -B -m automatic_price_checks once --sport nfl
.\.venv\Scripts\python.exe -B -m paper_collection_review --refresh
.\.venv\Scripts\python.exe -B -m price_survival_review
.\.venv\Scripts\python.exe -B -m low_credit_assessment
```

Evidence tables are `bets.automatic_paper_attempts` and
`bets.automatic_paper_observations`. Both reject updates/deletes. Raw current
responses are under `data/automatic_price_checks/`.

The review refresher calls the original frozen NBA and model-comparison report
functions directly, before the slower legacy all-history report. It verifies
source identities, reports stale/missing inputs, and never grants betting authority.

The additional price-survival review runs after the time-sensitive collection
steps in both scanner wrappers. It reports worse/same/better prices, unavailable
main offers, unknown failures, queue and delivery delays, and price-only versus
model-assisted results using the same first observation. Model decisions must
already exist before the check; late model matches cannot improve the comparison.
Missing checks and unknown results remain missing. Bootstrap intervals resample
whole games/dates, and each observer, sport and market remains separate.

### Earlier schedule cutoff audit

Some provider start times are later than the earlier native schedule. The legacy
report can therefore choose a close that is too late to establish pregame status.
`native_close_audit` separately reads the last market-family snapshot already
saved before the earlier cutoff, within the existing 22-minute close window.
It requires the same event, teams, market, player, side and exact line. A last
snapshot missing that exact quote, an ambiguous tied snapshot, or an estimated
line conversion cannot create a close.

The survival report retains counts of rejected original closes and separately
identifies replacements recovered from existing saved pregame quotes. This is a
read-only reporting repair: original decisions, receipts, frozen scorers and
historical reports are not rewritten. It makes no historical API requests.
The report records its reader source hashes and model identity. Use this audit
when reviewing close coverage; legacy coverage counts alone are insufficient.

NBA execution reviews now also display each sportsbook/market separately using
the existing review thresholds and audited closes. Passing these diagnostics
still cannot activate a new market, book, model filter or stake.

### Separate 500-credit assessment

The assessment models one NBA prop market, at most two games/day and 62 games per
provider credit cycle. Three polls per game at T−40, T−20 and T−5 cost at most
186 credits for one market and up to ten books. One fully charged retry per poll
adds 186; a 100-credit reserve brings the total to 472. The example schedule uses
the earlier matched native/provider tip and excludes unverified events and later
credit cycles.

This design is not activated and does not establish profitability. Before any
switch, verify free-plan reference-book access and the reset boundary, account
for every other paid collector, and register the narrower prospective protocol.
Current NFL/NCAAF and other NBA markets cannot continue unchanged alongside this
budget. Reduced sampling may miss opportunities and closes. Do not interrupt the
existing frozen collection merely to make the arithmetic fit.

## Costs and actual money

The cost audit uses only the current frozen FanDuel NBA cohort and includes
zero-opportunity calendar days. At a hypothetical true 3% return, covering the
$30 monthly cost requires 1,000 one-dollar bets. The cost ceiling at the observed
offer rate is conditional on every offer being available and that edge being real.
Automatic public quotes are not counted as actual wagers or income.

The rolling seven-day finance review includes imported receipts only and keeps
unknown costs/inventories explicit. The $30 subscription budget is not an invoice;
no billing charge or wager confirmation is invented. No stake or subscription
setting was changed.

The morning market/rotation report's numeric-type failure is repaired by the
separately registered `edge_research_numeric_report` adapter. It decodes numeric
database results only on its read-only reporting connection. Frozen forecasts,
selection rules and the original report source are unchanged.

Source: [The Odds API historical snapshot and quota documentation](https://the-odds-api.com/liveapi/guides/v4/).
Historical timestamps can resolve to an earlier snapshot and do not prove live
account availability.

## NFL slate monitoring

`python -m nfl_slate_readiness --refresh-news` writes the current slate report to
`reports/nfl_slate_readiness_latest.md` and saves timestamped observations under
`data/nfl_slate_monitor`. The NFL sharp watcher runs it after time-sensitive collection;
the close task also refreshes the report after each run. It leaves forecasts, selections,
manual checks and cash authorization unchanged.

The report separates pregame-run completion, saved forecasts, recent two-book quote
observations, all four close checkpoints, automatic-check backlog, late injury changes,
and completed-window evidence problems. ESPN status observations are timestamped
independently. Missing statuses do not certify participation or a complete official
inactive list. News observed before a forecast is never described as later information.

`python -m nfl_exact_close_review` explains missing closes for exactly the frozen NFL
price-gap cohort, separately by book. Different reference lines and estimated conversions
remain diagnostics. This full-history review runs manually/offline so it cannot delay
an imminent quote capture.

Future NFL starter cards include a projection manifest linking each displayed value to
its immutable forecast ID. Projection delivery verification is separate from exact-offer
verification and never establishes an executable price. Older empty manifests remain
unverified; they are not backfilled after the fact.
