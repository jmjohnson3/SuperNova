# Predictions ledger

Open [the filterable ledger](../reports/predictions_ledger_latest.html) in a browser.
It shows right, wrong, push, verified void, pending, no-line and excluded records,
plus daily counts and the individual predictions behind them. Filter by prediction
group, sport, sportsbook, version, market, game date, player/team or source ID.

The default view is delivered NBA FanDuel paper alerts. Other views cover locked
NFL model picks, NFL price alerts, NBA/NCAA game selections, model-assisted paper
picks, the consensus/rotation alternatives, and NBA stat forecast accuracy. Legacy
MLB predictions are outside this ledger's current scope.

## Counting rules

- Hit rate is **right / (right + wrong)**. Pushes, voids, pending and excluded
  records are separate and do not increase hit rate.
- Grade a recorded over/under or game side at its saved line. For spreads, the
  saved line belongs to the selected team. Actual spread values show the home
  team's margin. A tie at the threshold is a push.
- Repeated saved picks keep the first decision within a group, model/experiment
  version, sportsbook, player-game and market. A later opposite side or better
  line cannot replace it. Excluded legacy records cannot suppress valid picks.
- Strategies, books and versions have separate records. The same offer can appear
  in multiple comparisons; there is no combined grand-total win rate.
- Stat-only forecasts keep the last saved pregame revision. Without an offered
  line, they show the actual stat and absolute prediction error, not an invented
  win/loss. The UI shows mean error only when a single market is selected.
- NBA game-winner rows are derived from the last saved pregame projected score.
  “Model direction on paired lines” is derived from the point forecast saved
  before that offer. Both are labeled diagnostics, not delivered betting picks.
- NBA results require verified final scores and positive player participation.
  NFL props use official-final games and agreeing, verified participation grades.
  NCAA games use completed official results with verified team aliases. Missing or
  conflicting evidence stays pending. A verified NFL nonparticipant is a void.
- Legacy `voided_stale_card` records are excluded, not counted as participant voids.
- Game dates are Eastern time; the UI displays saved timestamps in Denver time.
- Paper units assume $1 at the recorded price. They do not imply actual wagers,
  executable prices, sportsbook settlement or net profit after costs.

The HTML is a local snapshot with its data embedded. It does not contact sportsbooks
or third-party servers. It can export the currently filtered rows as JSON.

## Refresh

```powershell
.venv\Scripts\python.exe -B -m prediction_ledger
.venv\Scripts\python.exe -B -m prediction_ledger --since 2026-11-01 --through 2026-11-04
```

The default range is September 1, 2026 through today's Eastern game date. Custom
ranges replace the latest report; the next scheduled run restores the default range.

NBA full runs refresh it after results and research reports. NFL-Sharp-Watch also
refreshes the cross-sport ledger each cycle, after time-sensitive captures. The
ledger itself makes no API calls, posts no messages, and opens a read-only database
transaction. It uses the normal pipeline's result refreshes; missing source results
are not repaired or fabricated by the scorecard. A separate lock prevents overlapping
ledger builds. The frozen scoring and selection files remain unchanged.

Outputs:

- `reports/predictions_ledger_latest.html`: filterable scorecard.
- `reports/predictions_ledger_latest.md`: summary by group, book, market and version.
- `reports/predictions_ledger_latest.json`: complete rows and summary counts, build
  timestamp, source-health findings and reporting-code checksum.

Validation:

```powershell
.venv\Scripts\python.exe -B -m pytest src/test_prediction_ledger.py -q
node src/test_prediction_ledger_ui.cjs
```

## Discord record-ledger

The authorized ledger message lives in `#record-ledger`, using the existing
`MLB_RECORD_LEDGER_DISCORD_WEBHOOK_URL` route. MLB's own posts stay separate.
The message shows the original FanDuel NBA alert record, NFL FanDuel model props
and game picks, and NCAA football spreads and totals. Football model records use
the latest saved book/version by forecast time, not by performance; the book and
version are labeled. Other versions remain separate in the attachments.
It also includes a readable text summary and a ZIP containing the full filterable
HTML, JSON rows and summary. Extract the
ZIP and open `predictions_ledger.html` locally; Discord does not run its filters.

NBA full runs and NFL-Sharp-Watch update the same saved message after successfully
building the ledger. Unchanged data does not create another message. Delivery
checks confirm the channel and verify the returned content and attachment names.
No mentions are sent. The saved message link and publication state are in
`reports/predictions_ledger_discord_state.json`; it contains no webhook token.

```powershell
.venv\Scripts\python.exe -B -m prediction_ledger_discord --dry-run
.venv\Scripts\python.exe -B -m prediction_ledger_discord
```

An uncertain first-send outcome blocks another create until reconciled, to avoid
duplicate posts. An existing message is updated by its saved ID. Publishing never
changes any prediction, result, model rule or betting authorization.
