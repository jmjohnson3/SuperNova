# FanDuel model-assisted paper decisions

`fanduel-model-assisted-paper-v1` adds a prospective comparison alongside the
unchanged NBA price-check, NFL sharp-price and NFL receiving experiments. It
never authorizes, places, reserves or confirms a wager.

The price-only control selects the first fresh, exact-line FanDuel offer with at
least 3% estimated EV against Pinnacle, 15–45 minutes before the event. The new
comparison supports NBA points/rebounds/assists/threes and game spreads/totals;
NFL passing/rushing/receiving yards, receptions, passing touchdowns and game
spreads/totals. Only half-integer lines are admitted, avoiding an invented push
probability. Each game/player/market contributes one decision. Both strategies
receive the identical offer; later prices cannot replace the first decision.

## How predictions change the paper decision

- Player props: the saved independent distribution must give positive EV at the
  exact FanDuel line and price. A missing model, nonpositive model EV or unsupported
  participation vetoes the model-assisted selection.
- Game spreads: projected home margin plus the home handicap must be positive
  for a home selection; away selections use the negative projected margin plus
  the away handicap.
- Game totals: projected total must favor the selected over/under. This is a
  directional filter on an already qualifying price discrepancy. No game win
  probability or additional EV is invented from a point projection.

Forecasts must exist before the offer, be no older than 12 hours, and match the
game and player. NBA game matching requires both teams, one unambiguous native
game ID, and at most 15 minutes between scheduled and provider tip times. New
captures stop before either known start time. NBA player eligibility requires an
active lineup observation within 20 minutes and no known Out/Doubtful status.
NFL props require the exact forecast line and independent `raw_over_probability`.
The selected forecast, version, probabilities, scores, context and timestamps are
copied into the append-only decision. Later forecasts cannot change a veto or pick.

The registration pins the scorer, selection, report, checks, parent experiment
identities and NBA training/feature recipe. Daily NBA fits continue under that
fixed recipe on earlier games. Material recipe changes block this experiment;
they do not silently replace its evidence. Historical offers are never backfilled.

## Live checks

Existing NBA prop checks are shared with the original price-only experiment.
The new model comparison card identifies that original check ID and does not
reset its deadline. Game offers and NFL offers receive `M` check IDs. A model veto
still receives a paper check request because its price-only control needs evidence.

Check within five minutes of delivery and record within five minutes of checking,
before the game. Include worse prices, changed lines, disappearance and suspension.
You can give the actual details in Codex; Discord replies are not imported.

```powershell
.\.venv\Scripts\python.exe -B -m model_research_checks pending
# Replace the ID and observed details; omit the M prefix in the command.
.\.venv\Scripts\python.exe -B -m model_research_checks record --id 123 --checked-at 2026-10-29T17:30:00-06:00 --status available --line 22.5 --price -120
.\.venv\Scripts\python.exe -B -m model_research_checks record --id 124 --checked-at 2026-10-29T17:31:00-06:00 --status disappeared
```

Only confirmed delivery can receive a check. First checks are immutable. Ambiguous
Discord deliveries are not retried. All messages say paper research; no new cash
authority follows from model agreement.

## Evidence and costs

The report separates sport and props/games, all FanDuel. Other books' existing
research remains in its original book-specific reports. It compares the paired
price control and model-assisted decisions, exact closing quotes, checked prices,
and game/date cluster intervals. NBA retains at least 21 days, 20 games, seven
active dates, three weeks, 50 checked prices paired with exact closes, 90% close
and timely-check coverage, 80% availability, +1.5% alert-price closing EV, and
positive game/date lower bounds. NFL uses at least 100 checked closes over three
weeks. Counts alone never authorize betting.

The model-assisted strategy must also improve on its paired control with positive
game/date lower bounds. Props require improved Brier scores on common settled
offers. Calibration bins show model and reference estimates versus observed wins;
Pinnacle is a benchmark, not known truth. Missing results/participation remain
pending, not voids or automatic under wins.

The legacy `raw.nba_games` schedule stops in April 2026. This comparison stores
official final scores separately from NBA's scoreboard endpoint. NBA full runs
refresh only dates with prospective decisions lacking a final snapshot, without
Odds API charges. Player grading additionally requires positive minutes, no
unresolved ingestion conflict, and both teams' logged points matching the official
final scores. Missing or partial game logs remain pending.

Diagnostics separate source quote age, observation-to-delivery delay,
delivery-to-check delay, recording delay, reference movement and price slippage.
Missing checks and closes remain unknown. New filters or faster polling require
their own future experiment; these observations do not change the frozen rules.

The economic scenario uses $1 comparison stakes, the **actual checked prices**
valued against exact independent closes, zero opportunities for confirmed missing
offers, and the full elapsed collection period, including days without selections.
The $30 monthly subscription is subtracted once across NFL/NBA. Net stays unknown
until at least three weeks and complete execution evidence for selected offers.
The report includes a conservative estimate from game/date intervals. These are
conditional paper scenarios, not sportsbook profit or grounds to increase stakes.
Actual receipt accounting stays in `sports_finance`.

## Operations

The existing NBA and NFL sharp-watch batch jobs invoke the separate collector
after their scanner. It reuses saved quotes, so this addition buys no API calls.
An unavailable model is an explicit veto. Registration/source drift stops new
captures and cannot silently reset the cohort.

```powershell
.\.venv\Scripts\python.exe -B -m model_research verify
.\.venv\Scripts\python.exe -B -m model_research report
.\.venv\Scripts\python.exe -B -m model_research results
```

Results: `reports/model_research_latest.md` and `.json`.
