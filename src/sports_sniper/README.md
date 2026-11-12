# Sports Betting Sniper

Finds sportsbook prices on player props that look too generous against what other sportsbooks charge for the
identical bet, ranks them, and tracks them as paper bets. **Research and paper trading only.** It never places
a wager, and an estimated edge is an estimate, not a profit.

All four phases are built to the extent the data allows (what is missing is listed at the end). The board still
ranks on market consensus alone: every model added so far has been tested against results and none has earned a
say in what is flagged.

## Run it

```bash
python -m sports_sniper.cli scan          # real saved snapshots if there are any, else sample data
python -m sports_sniper.cli settle        # grade finished paper bets, add closing line value
python -m sports_sniper.cli dashboard     # opens http://localhost:8517
python -m sports_sniper.cli backtest      # replay stored snapshots; writes reports/sniper_backtest_latest.md
python -m sports_sniper.cli evaluate      # the daily evaluation by hand: report, model calibration, parlay links
```

Other commands: `scan --source sample | saved | live`, `scan --sport nba`, `scan --lookback-hours 456` (replay
older saved snapshots), `top`, `paper-bet <id>`, `bankroll --staking flat | kelly`, `quota`,
`init-db`. Tests: `python -m pytest src/sports_sniper -q`. After `pip install -e .` the same commands are
available as `sniper scan`, `sniper dashboard` and so on. Settings are in `.env.example`.

## Where prices come from

| Source | What it is | Cost |
|---|---|---|
| `saved` | Snapshots the scheduled NBA and NFL scanners already keep in Postgres (`bets.sharp_multi_quotes`, `raw.nfl_api_responses`). Real prices, taken in the hour before each game. | Free |
| `live` | The Odds API v4 right now, for games starting inside the look-ahead window. | One credit per market per game |
| `sample` | Invented teams, players and prices shipped with the package. Always labelled hypothetical. | Free |
| `auto` (default) | `saved` when it has anything recent, otherwise `sample`. Never `live`. | Free |

A live scan is always explicit (a command-line flag or a ticked box in the dashboard). It stops at
`SNIPER_MAX_CREDITS_PER_SCAN` and never spends below `SNIPER_CREDIT_FLOOR`, because the same key feeds the
scheduled jobs.

## How an opportunity is priced

1. **Same bet only.** Two prices are compared only when sport, game, market, player and line all match.
   Alternate lines are never merged.
2. **Fair probability from other books.** Each reference book must price both sides at that line. Its margin is
   removed (`odds_math.no_vig`, power method by default, the method the rest of this repository uses) and the
   books are averaged, Pinnacle counting three times. The book being judged is never part of its own reference.
3. **Uncertainty.** The standard error is the disagreement between the reference books, never less than 1.5
   probability points, and 3 points when no sharp book prices the line.
4. **Conservative EV.** EV is worked out again with the probability lowered by one standard error. The board
   ranks by this number and by default shows only bets where it is positive.
5. **Evidence.** *Strong*: positive conservative EV, a sharp reference and a second book. *Moderate*: positive
   conservative EV with a sharp book, or with three retail books. *Speculative*: positive only before the
   haircut, or resting on one or two retail books.
6. **Too good to be true.** An estimated edge above 15% is labelled speculative whatever its references: it
   usually means a stale price or news the other books have already acted on.
7. **Insufficient data.** A one-sided market (anytime touchdown, most alternate lines) gets no probability and
   no EV. If one book pays at least 10% more than the middle price elsewhere it is listed as a price outlier,
   marked Insufficient data, and nothing more is claimed.

Every flagged row carries a sentence built from the numbers that produced it. Prices older than five minutes
at scan time are dropped; in the dashboard a price is "available" only when it comes from the newest look at a
game that has not started, seen in the last 20 minutes. Everything else is shown as expired history.

## Bet now, and whether the price was there

- **The page refreshes itself every minute** (a switch in the sidebar). It only re-reads what the scheduled scanners
  have stored, so it uses no API credits.
- **Bet now** sits above the tabs: flagged prices seen in the last 10 minutes (`SNIPER_FRESH_MINUTES`) that clear the
  bar, soonest game first, each with a countdown to the start, the bet link and the model's view. **When to look**
  lists the games still to start and when their prices should first appear (about 40 minutes before an NBA game, an
  hour before an NFL game), read from the existing pipelines' schedules.
- **Got it / Gone.** After opening a bet at the book, say whether the price was still there (in the Bet now panel, or
  under any board with the price you actually saw). Every answer is kept in `sniper_price_check_log`: the first is the
  measurement, a later one is a correction beside it. It records availability only, never a wager.
- **Performance** then shows the share of flagged prices that were really there, by book, how long after the price was
  stored you checked, and the paper record split by "price was there", "gone" and "not checked". Until now nothing
  measured this; the backtest only shows that under half of flagged prices survive to the next look ten minutes later.

## Paper trading and settlement

- **The bar (since 2026-11-11).** A bet is shown and paper-traded only with an estimated edge of 4% or more
  (`SNIPER_MIN_EV`) at a book in `SNIPER_BET_BOOKS` (FanDuel, DraftKings, theScore Bet). Other books are references
  only. On the first 372 settled paper bets, those under 4% lost and had the smallest closing value.
- **Automatic entries.** Every flagged bet on real prices that clears the bar, with a positive conservative EV and
  strong or moderate evidence, is entered as a paper bet (strategy `auto-consensus-v2`; `v1` is the earlier rule) at the best-priced book, the first time it
  qualifies, one per game, player, market, line and side. Bets saved by hand are strategy `manual`. Sample data is
  never entered automatically and never settled.
- **Replays.** A bet entered more than an hour after its price was seen (a scan of older saved snapshots) is marked
  replayed. The price was real, but the rule was applied afterwards, so forward bets are the clean test.
- **Settlement** reads final statistics from the existing pipelines (`providers/results.py`, read-only): NBA from
  `nbaprops.player_games`, NFL from `raw.nfl_player_gamelogs` with roster names. A game must be complete first:
  NBA when both teams' minutes add up to a whole game, NFL when the schedule row is final with a score. Over and
  under are graded won, lost or push at the line. A player with no line in a complete box score is void only when
  his name is known on one of the two teams; an unrecognised name stays open. Touchdown markets and MLB are not
  settled yet. The Note column says why a bet is still open.
- **Closing line value** is the edge of the price taken against the fair probability at the last look before the
  game: the other books' no-vig price for the identical bet, inside the last 22 minutes and later than the entry.
  Positive means the market moved toward the bet. No later look means no closing value; none is assumed.
- **Performance** shows record, simulated profit and ROI (standard error with whole games as the draws), closing
  line value, calibration and Brier score against the prices' own break-even, by sport, market, book, side and
  evidence, with forward and replayed bets separated.

## Validation and research (Phase 3)

- **Backtest** (`backtest.py`, the Backtest tab, `cli backtest`). Replays every stored snapshot in time order. A rule
  (minimum edge, books, sides, sports, evidence, odds range, minutes before the start) sees only what was in a snapshot at
  that moment, enters once per bet at the price that was stored, and is graded from the same final statistics that
  settle paper bets. It reports record, ROI with a 95% interval (whole games as the draws), estimated against realized
  edge, closing line value, largest fall, longest losing run, and results by period and market. Replaying the first
  automatic rule reproduces the paper ledger's record exactly, which is the check that the engine and the live path agree.
- **Entered one look later.** The same book's price at the next snapshot (about ten minutes on); no bet when the line
  is gone or no longer clears the bar. A closing price is never used as an entry price.
- **Walk-forward.** The minimum edge is chosen on earlier days only (by results or by closing value) and applied to
  the next three game days, so a threshold is never judged on the bets that chose it.
- **Paper bankroll** (`bankroll.py`). Flat or fractional Kelly on the probability lowered by its uncertainty, with
  limits per bet, per day, per sport, per game and per player, an optional stop after a fall, and the largest fall
  tracked. Stakes for a day come from the bankroll at the start of that day. A loss never raises the next stake.
- **Model against market** (`model_compare.py`, `providers/models.py`, the Model Comparison tab). Every two-sided line
  once, at the first look where a bet book and Pinnacle both priced it: Brier score and log loss of the bet book's own
  no-vig price, the other books' consensus and the NBA projection model (a forecast made before the price was seen),
  then what betting each source's picks at the bet book's price would have returned. NFL forecasts are not compared:
  the stored probabilities are already blended toward FanDuel's price.
- The slow part is kept per game in `data/sniper/backtest_cache.pkl` and redone only for games with new prices.
- Historical odds before the first stored snapshot (2026-10-22) are not used; none were bought.

## Predictive intelligence (Phase 2)

- **Model adapters** (`providers/models.py`). NBA: `nbaprops.projections` turned into a chance at the line. NFL: the
  production model's own chance before it is blended toward FanDuel, for receptions and passing touchdowns only (its
  yardage projections are anchored to the line, so they say nothing). Only a forecast made before the price was seen
  is used. MLB has no adapter: that pipeline has had no data since its statistics feed lapsed.
- **Calibration** (`calibration.py`). A two-number logistic fit of the model's log-odds, and a second fit of how much
  weight the model's disagreement deserves against the market. Both are fitted on earlier days only and saved to
  `data/sniper/model_calibration.json` by the daily evaluation.
- **Model view.** Each flagged over/under gets the calibrated model's chance of its own side: "Agrees" when it beats
  the price's break-even. It is recorded with the opportunity and the paper bet, shown on the board and in Performance
  ("By model view"), and available as a backtest rule. It never changes what is flagged or how it is ranked.
- **Predictions** (`predictions.py`, `predictions_view.py`). The forecasts the other pipelines already store, beside the
  latest stored line: NCAAF spreads and totals (`ncaaf.projections`), NBA spreads and totals (`nbaprops.game_projections`,
  lines from the saved snapshots) and player props (`nbaprops.projections`, lines from the Sniper's own stored prices), NFL
  spreads, totals and player props (`bets.nfl_*_predictions`). Read-only and information only: a "lean" is where the model
  sits against the line, it flags nothing and enters no paper bet. NBA lines exist only from about 40 minutes before
  tip-off; NFL game forecasts are the market line and NFL player projections are anchored to FanDuel's line.
- **Breakout Radar** (`breakout.py`). Players whose minutes or shot attempts (NBA), or targets, carries or snap share
  (NFL), over their last few games are above their own earlier average this season by an absolute amount and a share,
  with every recent game above it. The explanation is built from those numbers only; injuries and lineups are not read.
  `validate` checks the signal against settled lines.
- **What the evidence says so far** (`reports/sniper_backtest_latest.md`): the NBA model is close to noise once
  calibrated and adds nothing to the market; the NFL receptions model is slightly better than the market on a few
  hundred lines, too few to act on; flagged breakout players did not go over more often than the price implied.

### NBA candidate, calibration by statistic, and what is demonstrated (2026-11-11)

- **Candidate** (`nba_candidate.py`). The live NBA model sets each statistic's spread from the games it trained on.
  The candidate keeps the live projections and takes its scale (results over projections) and spread from earlier
  days' forecasts and results only, falling back to the live numbers while it has under 400 earlier player-games.
  It changes no pinned file. Its diagnostics also report whether the live model's stated spread matched later
  results, and how much of each miss would remain if the minutes played had been known.
- **Calibration by statistic** (`calibration.calibrate(..., by_market=True)`). Each statistic is calibrated on its own
  earlier lines once it has 300, pulled toward the fit for its sport in proportion to how few it has, and uses the
  sport's fit outright below that. A source's weight against the market is never negative and may stay at zero.
- **Three chronological steps.** Scale and spread from days before the game; calibration from candidate chances of
  earlier days (themselves out of sample); scores on the days after. `reports/sniper_backtest_latest.md` has the
  table by statistic. So far: the live model's stated spread was about right on unseen games (0.9 to 1.0 of the error
  it claimed), the candidate's raw scores are better for assists, points and threes and worse for rebounds, and after
  calibration neither version beats the market consensus. The miss is in the projected means, not the spread, and
  only 2 to 9% of it is playing time by this measure.
- **"Model agrees" is a stated hypothesis**, counted on forward bets only, by market, in the Performance tab, and not
  read until it has 100 settled bets in each group.
- **Demonstrated against research.** Performance opens on forward bets and on a Demonstrated section: the share of
  flagged prices that were really there (first answers only) and the return at the prices actually seen, by sport,
  market and book. Everything else is labelled research on paper. Price checks are an append-only log
  (`sniper_price_check_log`): the first answer is the measurement and later corrections are kept beside it.

## Advanced features (Phase 4)

- **Longshot ladder model** (`ladder.py`). A chance for one-sided alternate lines from the sharp book's main line for
  the same player and the spread of past results around their lines, each priced only from results known before its
  game. Always labelled speculative and never paper-traded: on the alternates stored so far its "underpriced"
  longshots lost. Shown under Longshot Hunter when alternates are in the latest prices.
- **Alerts** (`alerts.py`). With `SNIPER_ALERT_WEBHOOK_URL` set (or one per sport: `_NBA`, `_NFL`, `_MLB`), the scheduled
  pass posts each new automatic paper bet to that Discord channel once, while its price is fresh and its game has not
  started. The address is never logged. They are off on this PC: everything is read in the dashboard.
- **Parlay Lab** (`parlay.py`). Links between statistics measured from several seasons of game logs (same player,
  teammates, opponents) and a Gaussian-copula simulation of the chance that every leg wins, against the independent
  product. The book's same-game parlay price is typed in; it is not in the odds feed.
- **Automatic evaluation.** Once a day (`SNIPER_EVALUATE_EVERY_HOURS`) the scheduled pass rebuilds the backtest report,
  refits the model calibration on every settled line and re-measures the parlay links (`cli evaluate`, about 90
  seconds). The underlying NBA and NFL models are retrained by their own pipelines, not here.

## Scheduled runs

`scripts/sharp_multi_watch.bat` (task `\SuperNovaBets\Sharp-Multi-Watch`, every 10 minutes) runs
`python -m sports_sniper.cli watch` in the background after the scanner has saved its snapshots. One pass stores
the saved snapshots this database has not seen (NBA from that task, NFL from the NFL watch task, up to ten minutes
later), enters the bets they flag as paper bets and settles finished ones. It spends no API credits, never reads
sample data, and logs one or two lines to `logs/sharp_side_YYYYMMDD_sniper.log`. The SQLite file is opened in
write-ahead mode so the dashboard can read while a pass writes; it grows by roughly 10 MB a day of NBA slates.

## What works now (implemented and tested)

- Price math: American and decimal odds, break-even, no-vig (power and multiplicative), EV and ROI, fair odds,
  conservative probability, Wilson interval, Brier score, calibration table.
- The Odds API client (`httpx.AsyncClient`): events, event odds, quota tracking, credit floor and scan budget,
  retries with exponential backoff, rate-limit waits, a short cache, rejected markets skipped and remembered.
  The key is never logged or put in an error. Checked live with free calls only; **no paid live scan has been run.**
- Normalization: player-name matching across books, stale, malformed and duplicate prices dropped and counted.
- Cross-book comparison, best available price, opportunity ranking and evidence labels.
- Longshot Hunter: priced longshots with fair odds, plus one-sided price outliers marked Insufficient data.
- Line movement between stored snapshots (first stored price against the latest; not a true opening price).
- Storage (SQLAlchemy 2, SQLite or PostgreSQL): scans, prices, opportunities, paper bets. Every snapshot is kept
  whole (so the last one before a game is its close) and repeat scans add nothing.
- Paper bets at the observed price: automatic for flagged bets, or saved by hand; never duplicated unless a
  separate entry is asked for.
- Settlement of NBA and NFL over/under props, closing line value, and the performance summary.
- Backtest over stored snapshots with delayed entry and walk-forward, the paper-bankroll simulator, and the
  model-against-market comparison for NBA and NFL.
- Model adapters, walk-forward calibration, the model view on flagged bets, the Breakout Radar and its validation.
- The longshot ladder model and its walk-forward score, Discord alerts (tested with a stand-in for Discord; **no real
  alert has been sent**), the Parlay Lab, and the daily automatic evaluation.
- Streamlit dashboard: Opportunity Board, Predictions, Longshot Hunter, Breakout Radar, Parlay Lab, Line Movement, Performance, Paper
  Bets, Backtest, Model Comparison, with filters.
- NFL and NBA. MLB markets are listed and the code paths are shared, but MLB has not been run.

## What is not built yet

- **Settling touchdown markets and MLB.** The results source has no reliable touchdown total, and MLB has no
  results adapter, so those bets stay open.
- **Model-derived EV on the board.** Every board estimate is market consensus. The models are compared with the
  market and shown as a second opinion; none has beaten it, so none ranks or flags a bet.
- **Live in-game scanning.** The scheduled scanners stop at the start of a game, in-play prices would cost Odds API
  credits on every poll, and there is no live statistics feed to settle or model from.
- **Injuries and lineups in the Breakout Radar**, anytime-touchdown and two-touchdown models, and an MLB adapter.
- **Alternate lines going forward.** The NFL scanner stopped collecting them on Oct 26, so the ladder model has one
  weekend of evidence and nothing new to price unless a live scan asks for alternate markets.
- Backtests on seasons before the stored snapshots (would need bought history).
- **Alembic migrations.** Tables are created directly and new columns are appended on start; there is no
  migration history yet.
- Whether a flagged price is still there is only known for the bets you answer Got it or Gone on.

## What already exists in this repository, and what was reused

- **Reused now:** the saved Odds API snapshots (through `providers/saved_quotes.py`, read-only), the saved API
  key, and the power no-vig convention (a test checks agreement with `nfl_pipeline.sharp_math`).
- **Available for Phase 2:** NBA projections with their dispersion (`nbaprops.projections`), NFL forecasts
  (`bets.nfl_player_prop_predictions`), results for settlement (`nbaprops.player_games`,
  `raw.nfl_player_gamelogs`), MLB models under `src/mlb_pipeline/modeling/`.
- **What those models have shown so far:** in this repository's own tests none of them has beaten the books'
  prices (see `docs/nba_props_model.md`, `reports/nba_selector_backtest_latest.md`,
  `reports/nfl_clv_scorecard_latest.md`). The one measured edge is the price gap against Pinnacle. That is why
  Phase 1 ranks on market consensus, and why a model's EV must earn its place by out-of-sample evidence first.

## Layout

```
config.py          settings (pydantic-settings)
odds_math.py       price and probability arithmetic
markets.py         sports, market keys and labels
normalize.py       Odds API event -> clean quotes
consensus.py       fair probability from the other books
opportunities.py   detection, evidence labels, ranking
movement.py        line movement from stored prices
providers/         odds_api.py (live), saved_quotes.py, results.py, models.py and schedule.py (existing Postgres), sample.py (hypothetical)
db.py              SQLAlchemy models and queries
scanner.py         one scan, end to end
settlement.py      grading rules, closing line value, settling paper bets
performance.py     record, ROI, closing value and calibration with their uncertainty
backtest.py        replay of stored snapshots, rules, delayed entry, walk-forward
bankroll.py        paper-bankroll simulator (flat or fractional Kelly, exposure limits)
model_compare.py   probability sources scored against results
research.py        the report (reports/sniper_backtest_latest.md) and the daily evaluation
calibration.py     walk-forward calibration of model probabilities, by sport or by statistic
nba_candidate.py   NBA candidate: scale and spread measured on later games; workload split
breakout.py        breakout signals from game logs, and their validation
ladder.py          longshot ladder model for one-sided alternate lines
parlay.py          links between statistics and the joint chance of a parlay
alerts.py          Discord alerts for new paper bets (off by default)
cli.py             Typer commands
dashboard/app.py   Streamlit dashboard
fixtures/          sample_odds.json
```

## Keeping the dashboard running, and opening it away from home

The scheduled task `\SuperNovaBets\Sniper-Dashboard` (`scripts/tasks/Sniper-Dashboard.xml`) starts the dashboard at logon and
checks every 5 minutes that it is still up, with no console window. It listens on this PC only
(`http://localhost:8517`); the log is `logs/sniper_dashboard_YYYYMMDD.log`. To stop it for good, disable the task and end
its `python.exe`.

The dashboard has no login, so do not forward the port or put it on a public link. From another device use a private
network: with Tailscale installed on the PC and the phone, `tailscale serve --bg 8517` gives an address only your own
devices can open.
