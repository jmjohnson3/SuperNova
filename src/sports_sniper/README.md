# Sports Betting Sniper

Finds sportsbook prices on player props that look too generous against what other sportsbooks charge for the
identical bet, ranks them, and tracks them as paper bets. **Research and paper trading only.** It never places
a wager, and an estimated edge is an estimate, not a profit.

This is Phase 1: the working scanner, ranking, storage and dashboard. No prediction model is involved yet.

## Run it

```bash
python -m sports_sniper.cli scan          # real saved snapshots if there are any, else sample data
python -m sports_sniper.cli dashboard     # opens http://localhost:8517
```

Other commands: `scan --source sample | saved | live`, `scan --sport nba`, `top`, `paper-bet <id>`, `quota`,
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
6. **Insufficient data.** A one-sided market (anytime touchdown, most alternate lines) gets no probability and
   no EV. If one book pays at least 10% more than the middle price elsewhere it is listed as a price outlier,
   marked Insufficient data, and nothing more is claimed.

Every flagged row carries a sentence built from the numbers that produced it. Prices older than five minutes
at scan time are dropped; in the dashboard a price is "available" only when it comes from the newest look at a
game that has not started, seen in the last 20 minutes. Everything else is shown as expired history.

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
- Storage (SQLAlchemy 2, SQLite or PostgreSQL): scans, prices, opportunities, paper bets. Unchanged prices are
  stored once and repeat scans add nothing.
- Paper bets at the observed price, one per strategy and bet unless a separate entry is asked for.
- Streamlit dashboard: Opportunity Board, Longshot Hunter, Line Movement, Paper Bets, with filters.
- NFL and NBA. MLB markets are listed and the code paths are shared, but MLB has not been run.

## What is not built yet

- **Settlement.** Paper bets stay open; no results, profit, ROI or closing line value are computed yet.
- **Model-derived EV.** Every estimate is market consensus. Model adapters, calibration and model-versus-market
  comparison are Phase 2. So is the Breakout Radar, which needs usage and injury data sources.
- **Backtesting and the bankroll simulator** (Phase 3), alerts, live in-game scanning, Parlay Lab (Phase 4).
- **Alembic migrations.** Tables are created directly (`init-db`); there is no migration history yet.
- Whether a flagged price is still there when you open the sportsbook is not measured by this application.

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
providers/         odds_api.py (live), saved_quotes.py (existing Postgres), sample.py (hypothetical)
db.py              SQLAlchemy models and queries
scanner.py         one scan, end to end
cli.py             Typer commands
dashboard/app.py   Streamlit dashboard
fixtures/          sample_odds.json
```
