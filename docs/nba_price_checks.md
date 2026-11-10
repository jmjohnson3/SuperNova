# NBA paper price checks

The `nba-prop-price-check-v1` experiment collects execution evidence without wagering.
The existing `Sharp-Multi-Watch` job sends **PAPER PRICE CHECK** requests to
`NBA_DISCORD_WEBHOOK_URL`. Each request identifies the player, market, side, line,
sportsbook, American price, observation time, game time and a unique check ID.
The morning NBA digest reports the prospective experiment separately by sportsbook.

## Fixed selection

- Points, rebounds, assists and made threes; FanDuel, DraftKings and theScore/ESPN Bet
  are separate cohorts. No cross-book sample pooling for review.
- At least 3% estimated EV against Pinnacle's power-de-vig probability at the exact
  same line, 15–45 minutes before tip.
- Both quotes must be at most five minutes old, with at most two minutes between
  their source updates. The market update timestamp takes precedence over its book's
  update. Raw snapshots retain both timestamps.
- The first qualifying observation selects one request per book, game, player and
  market. If several sides/lines qualify in that observation, the highest EV wins.
  Later polls remain observations and do not add independent decisions or alerts.
- No model-agreement filter, projection threshold, or model-based stake sizing.

The immutable registration contains the rules, executable source snapshots, runtime
versions and SHA-256 hashes. New rows carry its ID and hash. Pre-registration rows
are historical context, never reconstructed prospective checks. Source/configuration
drift stops new enrolled requests and blocks review; it does not create a new version
automatically. Necessary future changes need an explicitly registered prospective
challenger or version, with evidence kept separate.

## Record a real check

Check the requested sportsbook, player, stat, side and exact line without placing a
wager. Record the first check even when the price worsened or the offer disappeared.
For review, check within five minutes of the Discord alert and save it within five
minutes of checking, both before tip. Slower checks remain in the ledger but do not
satisfy execution coverage. Include your actual UTC offset; Denver changes from
`-06:00` to `-07:00` when daylight saving time ends.

From the repository folder in PowerShell:

```powershell
# List delivered requests still awaiting a check before tip.
.\.venv\Scripts\python.exe -m sharp_pipeline.price_checks pending

# Replace ID/time/line/price with what you actually observed.
.\.venv\Scripts\python.exe -m sharp_pipeline.price_checks record --alert 123 --checked-at 2026-10-28T17:30:00-06:00 --status available --line 22.5 --price -110

# If the exact offer disappeared:
.\.venv\Scripts\python.exe -m sharp_pipeline.price_checks record --alert 123 --checked-at 2026-10-28T17:30:00-06:00 --status disappeared

# If only a different line remains:
.\.venv\Scripts\python.exe -m sharp_pipeline.price_checks record --alert 123 --checked-at 2026-10-28T17:30:00-06:00 --status changed_line --line 23.5 --price -115
```

`available` means the exact line is present, even if its price changed. `changed_line`
does not count as the requested offer being available. `suspended` and `disappeared`
take no fabricated line or price. A first check cannot be overwritten with a more
favorable later check. Optional `--note` can describe what happened.

You can also give Codex the check ID, actual time with timezone, line and price/status
to record. **Discord replies are not automatically imported.** Nothing is inferred
from silence. Only confirmed Discord deliveries can receive check records.

### Guided FanDuel entry

When a paper alert arrives, open the same game, player, stat, side and exact line
in FanDuel. Read its current odds, including worse prices. No wager is needed.
Record within five minutes of the original alert, then save immediately. A
blocked page, login problem or incomplete search is **unable to check**, not a
disappeared offer; leave that observation unknown.

For a guided terminal prompt, use:

```powershell
.\.venv\Scripts\python.exe -B -m nba_props.fanduel_check pending
# Replace 123 with the actual original NBA alert ID.
.\.venv\Scripts\python.exe -B -m nba_props.fanduel_check record --alert 123
```

The prompt displays the original alert time and deadline in Denver time. Enter
the displayed status, line and American odds yourself; it never defaults to the
API price. Enter the actual check timestamp with its UTC offset, or type `now`
only if you just observed the market. Blank time or `q` cancels without writing.
The first observation is saved immediately through the existing immutable
recorder. Late observations remain late and cannot pass execution coverage.

Phone users can send Codex: `NBA #<ID> | checked <time> Denver | <side> <line>
<stat> | odds <price>`, or the ID, time and `disappeared`/`suspended`. State a
different line explicitly if that is all that remains. Send the first check,
including an unfavorable price; do not wait for a better one.

The original NBA check is shared by price-only, model-assisted and linked count
probability comparisons. A later MODEL COMPARISON card does not reset the check
deadline. This guided entry does not change any frozen selection or review rule,
access the sportsbook, place a wager, send Discord messages or monitor future
alerts unattended.

Historical API data cannot substitute for a live check: the provider returns a
snapshot at or before the requested timestamp. See the
[Odds API historical-data documentation](https://the-odds-api.com/historical-odds-data/).

### Automatic FanDuel feed checks

`nba_props.fanduel_live` reads the current public FanDuel feed for the registered
state (Colorado). `Sharp-Multi-Watch` invokes it after each scan's prospective
model captures, before the digest. No additional Odds API credits are used.

For each newly delivered FanDuel NBA alert after this observer's registration,
the checker requests the matching event and player-stat tab. It matches both teams,
the date, player, stat, side and exact line, verifies the complete tab and current
runner status, then saves the returned odds and response timestamps immediately.
The earlier native/provider/FanDuel start time stops pregame checking. Checks have
a five-minute deadline measured from the original Discord delivery. Historical
alerts are not backfilled. Supported markets: points, rebounds, assists and threes.

The first attempt is immutable, including worse odds and failed requests. Missing
offers in complete responses are `not_listed`; incomplete, stale, blocked or failed
responses are `unknown`. A changed line or suspended market is recorded separately.
Requests have timeouts and a per-run budget; access blocks and rate limits stop
further requests for that run. There is no login, browser bypass, bet placement,
new Discord message, or automatic betting approval.

These are **automated public-feed observations**, not manual checks in your
logged-in account or confirmed executions. HTTP freshness describes the retrieved
response; it does not prove when the underlying quote was last updated or that
FanDuel would accept a wager. The existing manual review gate and frozen price,
model-assisted and count-model experiments remain unchanged. Manual checks are
still needed to establish account-level availability under their existing rules.

```powershell
# Run once now; skips old/already checked alerts and never sends alerts itself.
.\.venv\Scripts\python.exe -B -m nba_props.fanduel_live once

# Optional foreground watcher, polling for new alerts every 30 seconds for 3 hours.
.\.venv\Scripts\python.exe -B -m nba_props.fanduel_live watch --minutes 180

# Verify the separately frozen observer before collection.
.\.venv\Scripts\python.exe -B -m nba_props.fanduel_live verify
```

Results: `reports/fanduel_live_checks_latest.md` and `.json`. The report shows
original versus observed odds, delay, status and reason. Counts describe this
observer's attempts only; they do not replace execution coverage or satisfy the
manual-check gate. The append-only ledger is `bets.fanduel_live_checks`; raw
timestamped response receipts and content hashes are in `data/fanduel_live_checks/`.
The existing manual ledger is never written by this script. A database lock and
unique alert key prevent scheduled and optional watcher runs from double-counting.

The observer is frozen as `fanduel-direct-check-v1`, independently of betting
selection. Source drift stops collection. Initial setup uses `register --state co`;
that command cannot overwrite an existing registration. A material change requires
a new prospective observer version rather than rewriting earlier evidence.

## Review requirements, separately for every book

These are minimum review requirements, not a betting approval:

| Evidence | Minimum |
|---|---|
| Distinct selected decisions with exact closes | 50 |
| Collection period | 21 calendar days, spanning three relative week bins |
| Diversity | Seven active game dates and 20 games |
| Observed exact-line close coverage | 90% of selected decisions |
| Mean EV at the alert price, valued using the close | +1.5% |
| Timely manual checks | 90% of selected decisions |
| Exact offer still available | 80% of timely checks |
| Actual checked prices paired with exact closes | 50 |
| Exact-close coverage of available checked offers | 90% |
| Uncertainty | Positive 95% lower bounds for both alert-price and checked-price EV, resampling whole games and whole dates separately |

The report uses a fixed-seed 2,000-draw cluster bootstrap. Bets within the same game
are resampled together, not treated as independent observations. Day clustering is
an additional check. Each sportsbook must pass on its own evidence; the same game
at two books does not create two independent games. These intervals are research
estimates, not proof that Pinnacle's probabilities are the true outcome probabilities.

Unknown checks and missing closes remain unknown and reduce coverage. Timely checks
showing unavailable offers have zero opportunity EV; they are not losing bets or
fake zero-price executions. Checked-price EV uses the requested side's exact closing
probability and the price actually displayed. Estimated line conversions never count
as observed closes. Paper win/loss results and model agreement remain secondary
research comparisons. No actual profit is inferred from these paper checks.

Passing all checks produces `execution_review_only`, with `betting_authorized=false`.
A capped $1 trial still requires a separate explicit review and decision.

## Operations

```powershell
# Verify the frozen registration; exits nonzero on drift.
.\.venv\Scripts\python.exe -m sharp_pipeline.nba_experiment

# Rebuild both reports without posting Discord messages.
.\.venv\Scripts\python.exe -m sharp_pipeline.report
```

Reports: `reports/nba_price_checks_latest.md` and `.json`. The older
`reports/sharp_multi_latest.md` remains historical/context reporting. Ledger tables:
`bets.nba_paper_price_alerts` and `bets.nba_paper_price_checks`, linked to exact source
decisions and snapshots. The outbox prevents duplicate sends and expires requests
older than five minutes. Ambiguous HTTP deliveries remain `uncertain` and are not
automatically resent; interrupted `sending` rows likewise need delivery investigation.
Missing NBA webhook configuration leaves requests pending rather than sending them
to a different sport's channel.

Full NBA runs also write a dated recap of yesterday's delivered paper alerts:

```powershell
.\.venv\Scripts\python.exe -B -m nba_props.paper_recap --date 2026-10-28 --refresh-results
```

The recap verifies final NBA scores and positive player minutes. Player logs must
reconcile with both teams' final scores; otherwise a complete official box score
must resolve the discrepancy or the result remains pending. Saved source snapshots
and any differences appear in the JSON. It does not overwrite the season logs or
change frozen experiment rules. These official NBA requests use no Odds API credits.

`reports/nba_paper_recap_YYYY-MM-DD.md` and `.json` separate each sportsbook's
hypothetical profit at alert prices from unique underlying picks and repeated
cross-book alerts. Exact closes and timely price checks are shown alongside wins
and losses. Missing checks remain unknown; these reports never establish actual
wager profit, authorize stakes, or retroactively create model-assisted selections.
