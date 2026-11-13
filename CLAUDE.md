# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperNovaBets is a sports betting prediction pipeline with three leagues:
- **`src/nba_pipeline/`** — NBA spread/total/player-prop predictions (production)
- **`src/mlb_pipeline/`** — MLB run line/total/player-prop predictions (new 2026-03-27)
- **`src/nfl_pipeline/`** — NFL player-prop forecasts, game models and a gated cash trial (see [NFL Pipeline](#nfl-pipeline))

Both pipelines share the same PostgreSQL database (`nba` db), the same `raw.api_responses` table, and the NBA `fetcher.py` / `raw_store.py` utilities (the MLB pipeline imports them directly).

## Running the Pipelines

### Environment
```bash
.venv/Scripts/activate        # Windows
source .venv/bin/activate     # Unix/Mac
pip install -e .              # installs both nba_pipeline and mlb_pipeline
```

### NBA — Full daily run
```bash
python -m nba_pipeline.run_daily
python -m nba_pipeline.run_daily --date 2026-02-24
python -m nba_pipeline.run_daily --close-only   # ~6:30 PM ET: capture closing lines for CLV
```

Skip flags: `--skip-crawl`, `--skip-parse`, `--skip-train`, `--skip-predict`, `--skip-scan`, `--backfill-odds`

### NBA — Individual steps
```bash
python -m nba_pipeline.crawler                          # MySportsFeeds (games/boxscores/gamelogs)
python -m nba_pipeline.crawler_oddsapi                  # Odds API (spreads/totals/props)
python -m nba_pipeline.crawler_oddsapi_backfill         # Historical odds backfill
python -m nba_pipeline.parse_all                        # All parsers + apply SQL views
python -m nba_pipeline.compute_elo                      # Elo ratings (run after parse_all)
python -m nba_pipeline.modeling.train_game_models       # Spread/total models
python -m nba_pipeline.modeling.train_player_prop_models
python -m nba_pipeline.modeling.predict_today
python -m nba_pipeline.modeling.predict_player_props
python -m nba_pipeline.modeling.scan_alt_lines_grid
python -m nba_pipeline.modeling.update_outcomes         # Grade completed games
python -m nba_pipeline.modeling.paper_trading_report [--days 90]
```

### MLB — Full daily run (new)
```bash
python -m mlb_pipeline.run_daily
python -m mlb_pipeline.run_daily --date 2025-04-01
```

Skip flags: `--skip-crawl`, `--skip-parse`, `--skip-train`, `--skip-predict`

### MLB — First-time setup
```bash
# 1. Bootstrap DB tables
psql postgresql://josh:password@localhost:5432/nba -f sql/MLB000_schema_bootstrap.sql

# 2. Backfill 2024 season
python -m mlb_pipeline.crawler --start-date 2024-03-20 --end-date 2024-09-29
python -m mlb_pipeline.crawler_oddsapi   # catches up from last saved date automatically

# 3. Parse and apply views
python -m mlb_pipeline.parse_all
```

## Architecture

### Shared infrastructure
- **`raw.api_responses`** — single table storing every raw API payload from all providers/leagues. Key: `(provider, endpoint, url)`. Idempotent via `ON CONFLICT DO UPDATE`. The `payload_sha256` enables dedup.
- **`raw_store.py`** (`save_api_response`) and **`fetcher.py`** (`MySportsFeedsClient`) live in `nba_pipeline/` but are imported directly by `mlb_pipeline/` — not copied.
- **DB**: `postgresql://josh:password@localhost:5432/nba` (hardcoded throughout; no env var needed for local dev)

### NBA data flow
```
MySportsFeeds API                 The Odds API
       │                                │
       ▼                                ▼
raw.api_responses (all payloads, sha256 dedup)
       │
       ▼  parse_* scripts (dimensions before facts)
raw.nba_games / nba_player_gamelogs / nba_boxscore_* / nba_lineups / ...
       │
       ▼  sql/V001–V026 + parse_all._apply_view_fixes()
features.game_training_features_mat   ← MATERIALIZED VIEW (~90s to build)
features.player_training_features     ← VIEW (depends on matview)
       │
       ▼  train_game_models.py / train_player_prop_models.py (XGBoost + LightGBM + Optuna)
src/nba_pipeline/modeling/models/*.json
       │
       ▼  predict_today.py / predict_player_props.py
bets.game_predictions / bets.prop_predictions
```

### MLB data flow (same pattern, no matviews in V1)
```
MySportsFeeds (mlb/ seasons)      The Odds API (baseball_mlb)
       │                                │
       ▼                                ▼
raw.api_responses (same table, different endpoint names)
       │
       ▼  mlb_pipeline/parse_* scripts
raw.mlb_games / mlb_player_gamelogs / mlb_boxscore_* / mlb_starting_pitchers / ...
       │
       ▼  sql/MLB001–MLB006 (applied by parse_all._apply_sql_views)
features.mlb_game_training_features / mlb_game_prediction_features
       │
       ▼  mlb_pipeline/modeling/train_game_models.py
src/mlb_pipeline/modeling/models/*.json
       │
       ▼  mlb_pipeline/modeling/predict_today.py
bets.mlb_game_predictions
```

### SQL view system
- **NBA**: `sql/V001_*` through `sql/V026_*` — applied via `parse_all._apply_view_fixes()`. No migration runner; apply manually when adding new views. Views up to ~V015 build on top of earlier ones; order matters.
- **MLB**: `sql/MLB000_schema_bootstrap.sql` (DDL, run once) + `sql/MLB001–006_*.sql` (views, applied each run by `mlb_pipeline.parse_all`).
- **`game_training_features` and `game_prediction_features` are MATERIALIZED VIEWS** (NBA only) backed by thin wrapper views of the same name. `parse_all._materialize_game_features()` creates/refreshes them. This collapses an ~80s query to <1s for predictions.

### NBA parse pipeline order (order matters)
```
parse_meta        → venues, teams, standings, injuries
parse_games       → nba_games (sync scores from boxscores after this)
parse_player_gamelogs → player stat backbone
parse_lineup      → availability / starters
parse_boxscore    → game + player box scores  ← then re-sync scores
parse_pbp         → play-by-play advanced features
parse_referees    → referee assignments
parse_game_odds / parse_game_odds_historical → odds.nba_game_lines
parse_prop_odds / parse_prop_odds_alt → odds.nba_player_prop_lines
```

### MLB parse pipeline order
```
parse_meta → parse_games → parse_player_gamelogs → parse_boxscore
→ parse_starting_pitchers → parse_game_odds → sync_scores_from_boxscores
→ _apply_sql_views (MLB001–006)
```

### Modeling — shared patterns (both pipelines)

**Two model families for game lines:**
- *Direct*: XGBoost predicting `run_diff` / `total_runs` directly (or `margin` / `total` for NBA)
- *Residual*: predicts deviation from market line; reconstruction = `market_line + residual`. Quality gate: residual MAE must be ≤ direct MAE × 1.02.

**Walk-forward CV**: no leakage. NBA: `min_train_days=60`, `test_window=7d`, `step=7d`. MLB: `min_train_days=120`, `test_window=14d`, `step=14d`.

**Median imputation**: fit on train split only, applied to val/test. Stored in `feature_medians.json`.

**Feature engineering lives in two places** that must stay in sync:
- `make_xy_raw()` in `train_game_models.py` (training)
- `_prep_features()` in `predict_today.py` (inference)
- Both call `add_game_derived_features()` from `modeling/features.py`

**Edge formula** (critical — sign convention):
- `edge_spread = pred_margin_home + market_spread_home` (NBA; positive = home covers)
- `edge_run_line = pred_run_diff + market_run_line` (MLB; same convention)
- `edge_total = pred_total - market_total` (positive = bet OVER)

**Bet thresholds**: NBA spread ≥ 7.0 pts, NBA total ≥ 5.0 pts. MLB run line ≥ 1.5 runs, MLB total ≥ 1.0 runs.

### Model artifacts
NBA: `src/nba_pipeline/modeling/models/`
- `spread_direct_xgb.json`, `total_direct_xgb.json`, `spread_resid_xgb.json`, `total_resid_xgb.json`
- `spread_direct_lgb.txt`, `total_direct_lgb.txt` (LightGBM; 50/50 ensemble with XGB)
- `feature_columns.json`, `feature_medians.json`, `calibration.json`, `optuna_best_params.json`, `feature_importance.json`
- `player_props/`: `points_xgb.json`, `rebounds_xgb.json`, `assists_xgb.json`, `lgb_multi.pkl` (joblib)

MLB: `src/mlb_pipeline/modeling/models/`
- `run_line_direct_xgb.json`, `total_direct_xgb.json`, `run_line_resid_xgb.json`, `total_resid_xgb.json`
- Same calibration/feature JSON files

### Grading and paper trading (NBA)
- `update_outcomes.py` fills `bets.game_predictions.actual_*` and computes `clv_spread`/`clv_total` (Closing Line Value). Run nightly after games complete.
- `paper_trading_report.py` prints W/L records segmented by edge bucket, sport, and prop stat.
- `bets.game_predictions` tracks: predictions, kelly fractions, actual scores, ATS coverage, CLV.
- `bets.prop_predictions` tracks: prop predictions, book lines, edges, kelly fractions, `*_over_hit` boolean.

### External APIs
- **MySportsFeeds v2.1**: `https://api.mysportsfeeds.com/v2.1/pull/{league}/{season_slug}/...`
  - NBA seasons: `2024-2025-regular`, `2025-2026-regular`
  - MLB seasons: `2024-regular`, `2025-regular`
  - API key in `CrawlerConfig.api_key` (same key for both leagues)
- **The Odds API**: sport keys `basketball_nba` / `baseball_mlb`. Key in `OddsCrawlerConfig.oddsapi_key`. Credits reset April 1. Floor: 200 credits (daily), 500 (backfill).

### Notifications
`run_daily_and_notify.py` posts to Discord after the NBA pipeline completes. Webhook URL via `DISCORD_WEBHOOK_URL` env var. `DISCORD_FORMAT=1` env var switches predict scripts to compact Discord output.

## NFL Pipeline

`src/nfl_pipeline/` is a separate, evidence-gated pipeline. Design decisions live in
`docs/nfl_*.md` (read the relevant one before changing a model or policy); generated
evidence lives in `reports/nfl_*_latest.{md,json}`.

### Running
```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.run_daily_and_notify [--date D] [--skip-crawl] [--skip-train] [--skip-predict]
.\.venv\Scripts\python.exe -m nfl_pipeline.run_pregame [--dry-run]         # game-aware pregame refresh
.\.venv\Scripts\python.exe -m nfl_pipeline.run_close_and_grade [--date D] [--force-crawl]
.\.venv\Scripts\python.exe -m nfl_pipeline.run_training [--season S] [--skip-context]
.\.venv\Scripts\python.exe -m pytest src/nfl_pipeline -q --disable-warnings
```

Scheduled via Windows Task Scheduler (`scripts/tasks/NFL-*.xml`, installed by
`scripts/switch_to_nfl_tasks.ps1`, which swaps out the MLB tasks):
| Task | Schedule | Runs |
|---|---|---|
| NFL-Daily | daily 7:30 | `nfl_daily.bat` → `run_daily_and_notify` |
| NFL-PrimeTime-Refresh | every 10 min | `nfl_daily.bat --game-aware` → `run_pregame` |
| NFL-Close | every 10 min from 7:00 | `nfl_close.bat` → `run_close_and_grade` |
| NFL-Training | Tuesday 3:00 | `nfl_training.bat` → `run_training` |

Batch files take jobs through `scripts/run_with_nfl_mutex.ps1` so operational runs never
overlap, log to `logs/nfl_*_YYYYMMDD.log`, and read API keys from HKCU environment
(`ODDS_API_KEY`, `SPORTSGAMEODDS_API_KEY`, `THERUNDOWN_API_KEY`, `NFL_DISCORD_WEBHOOK_URL`;
provider order via `NFL_ODDS_PROVIDER_ORDER`; optional `NFL_FANDUEL_STATE` (betslip IDs are
per state, see `docs/nfl_fanduel_links.md`), `NFL_SHARP_WATCH_MARKETS`, `NFL_SHARP_CREDIT_FLOOR`, `NFL_ODDS_API_RESET_DAY` = day of month the Odds API plan resets).

### Production is pinned
- `modeling/models/active_release.json` / `production_freeze.json` pin the production
  release (currently `nfl-20260918T143342Z`, feature contract `nfl-asof-v2`) by sha256.
  `integrity.py` enforces it; `NFL_MODEL_RELEASE_ID` overrides.
- New models are **challengers**: they are trained into timestamped dirs under
  `modeling/models/<experiment>/` and evaluated against production with walk-forward
  folds and archived-offer replays. Promotion is a manual decision recorded in a doc.
- Never rewrite historical locks/forecasts or reconstruct missing legacy inputs; exclude
  and count them instead. Proxy-line results are not betting-edge evidence.

### Cash strategy: sharp price gaps (since 2026-10-21)
The only NFL strategy that carries money is `sharp_watch` (rules in `sharp_strategy.py`,
registration `nfl-sharp-gap-cash-v9`): FanDuel, ESPN Bet or DraftKings at +3% or better against
Pinnacle's no-vig price, seen 15-45 minutes before kickoff, in receiving yards, rushing yards, receptions and game totals.
Alternate player-prop ladders are no longer scanned (v8): on 7,460 settled rungs `modeling/alt_line_model` was
no better than the books' own prices, its "+EV" rungs lost 8% and cross-book rung gaps lost too
(`LADDER_SCAN = False`; v7's ladder evidence stays in the report). `modeling/sharp_replay` (`all_edges`/`lever_table`) measures what each rule change would
add; the daily/weekly caps, not a lack of gaps, limit how many bets are staked. The model forecasts do not beat FanDuel's line and stay research; they are only a
recorded shadow filter beside each decision.
- `NFL-Sharp-Watch` runs every 10 minutes and polls each game on every run through its final hour
  (stake window plus the closing snapshot). Links and alert text live in `sharp_links.py`, which is not
  pinned, so a book changing its link domain (ESPN Bet is now theScore Bet) needs no new registration. Discord pings only staked bets; confirm placement with `sharp_bets --confirm`.
- Limits (v9, user decision 26 Oct): $1 flat, ≤10 bets/day, ≤2 per game, one per player-game-stat across books, ≤$40 per NFL
  week, sticky pause at $30 cumulative confirmed loss. No automated wager placement.
- v8: the daily bets are shared across kickoff windows (kickoffs within 45 minutes are one window; one
  bet each, the rest by games; cumulative, so unused bets roll forward: `sharp_strategy.window_allocation`).
  The shadow filter also reads ESPN's game-day statuses (`sharp_inactives`, pinned): the bet player's own
  status and teammates newly out since the T-120 forecast, with their projected volume moved to the rest.
- Gates (`modeling/sharp_alert_report`, run by the close task): close capture ≥90% of recent
  decisions, EV at the sharp close (pause if ≤0 after 50; scale review at ≥+1.5% over 150+ and 3+
  weeks), the loss pause, player/team concentration ≤20%/35%. Raising stakes is a manual decision.
- A scan that refuses (unregistered/drifted) while games are near posts one Discord warning a day.
  The registration pins only the pricing, selection, staking and grading modules; changing any of
  them, `sharp_strategy` or `NFL_SHARP_WATCH_MARKETS` needs a new one (`sharp_watch --register`).
- `modeling/sharp_replay` replays the rules over saved scanner payloads (retrospective only).

### Model forecast flow (research)
Fresh FanDuel quote → frozen forecast (`forecast_store` / `forecast_outputs`) → immutable
ledger → pinned challenger capture → fixed research selection → `cash_readiness` →
reservation (`cash_execution`) → Discord → exact close → settled result → evidence refresh.
- The registered FanDuel receiving trial (`models/cash_trial/registration.json`) keeps its
  frozen-model gates (`docs/nfl_cash_readiness.md`) but is research, not the cash strategy.

### Bet scope, market calibration, CLV (see `docs/nfl_market_calibration.md`)
- Only stats in `betting_preferences.BET_PROP_STATS` (receiving yards) can lock into staked tiers;
  every other market (incl. receptions, spreads, totals) is paper/research in Discord with links.
- `models/player_props/market_calibration.json` anchors projections to the line, widens ranges and
  blends probabilities toward FanDuel no-vig. Fitted by `modeling/fit_market_calibration.py`
  (report-only in Tuesday training; `--write` installs). Missing file = identity.
- Judge changes by `reports/nfl_clv_scorecard_latest.md` (CLV first, ROI last).
- Official nflverse box-score stats win in `raw.nfl_player_gamelogs`; play-by-play only fills gaps.
- `modeling/projection_error` measures raw projection error by stat and splits it into volume and
  efficiency. `models/player_props/placement_calibration.json` (`fit_placement_calibration`, validated on
  held-out weeks) moves the raw projection before anchoring; missing file = identity.
- `modeling/volume_model` is the targets/carries model (`docs/nfl_volume_and_placement.md`), trained on
  live/table parity-safe features only. Its targets feed the receptions projection through
  `models/player_props/targets_model.json`; its carries run as the prospective rushing challenger
  (`modeling/rushing_volume_challenger`, captured in daily/pregame runs, graded by the close task).
- The live snapshot and the training table disagree on team-relative, injury-downgrade and depth-movement
  features (`modeling/feature_parity`, `docs/nfl_feature_parity.md`). The production yardage models use
  none of them; the opportunity/workload models behind the workload adjustment do. Check parity before
  training on a feature.
- `modeling/volume_absence` (2026-11-12, research, not scheduled): teammates listed Out or Doubtful, volume returning and
  the player's own volume without them improve the volume forecast (targets 0.7%, carries 1.4%; 10% when a back is
  newly out) but not the yardage projections, which stay worse than production after the rescale. Not adopted.
- `modeling/efficiency_model` (2026-11-12, research, not scheduled): per-opportunity rates judged with the volume known.
  A model beats a shrunk last-10 rate by 1% to 2%; opponent splits by position and weather add almost nothing; no
  variant beats production or the line on live 2026 forecasts. Not adopted.
- `modeling/game_line_test` (2026-11-12, research, not scheduled): spreads and totals against the closing line, 2010 to 2026
  (4,499 games, refit each season on earlier ones). Direct and residual models are both less accurate than the close,
  earn no weight beside it, and no disagreement rule reaches the 52.4% break-even. Game models stay context.

### Model artifacts (NFL)
`.joblib`/`.pkl` binaries and bulky run outputs under `src/nfl_pipeline/modeling/models/`
are gitignored (several exceed GitHub's 100 MB limit). Pointer, registration, manifest and
summary JSON files are tracked. A fresh clone must retrain or copy artifacts before predicting.

## Sharp-gap research: NBA and NCAAF (`src/sharp_pipeline/`)

The NFL price-gap test applied to other sports, research only (no stakes, no pings). NBA player props
(points, rebounds, assists, threes) and NBA and NCAAF spreads, moneylines and totals; no college props
(not bettable in Colorado). Exact-line pricing against Pinnacle no-vig at FanDuel, theScore Bet
(`espnbet`) and DraftKings; +3% seen 15-45 minutes before the start is a decision, graded at
Pinnacle's last quote before the start.
- Task `\SuperNovaBets\Sharp-Multi-Watch` (`scripts/sharp_multi_watch.bat`, every 10 minutes at :05) runs
  `python -m sharp_pipeline.scan`, then `python -m sharp_pipeline.report --digest`. The digest goes to Discord
  after 9 AM ET, one per sport, via `NBA_`/`NCAAF_DISCORD_WEBHOOK_URL`, falling back to
  `MLB_DISCORD_WEBHOOK_URL`.
- Credits: game lines use one bulk call per sport (3 credits for every game); NBA props cost 4 credits per
  game per poll (about T-40/-30/-20/-10). The daily allowance holds back NFL Sunday/primetime reserves
  until the plan resets.
- Tables `bets.sharp_multi_quotes` (append-only quotes) and `bets.sharp_multi_decisions`; report
  `reports/sharp_multi_latest.md`. A sport is ready for a stake review at 50+ graded decisions over 7+
  days, +1.5% at the close and 90% close coverage. Staking it is a separate decision.
- Shares no tables or registration with NFL; it only imports the NFL price math.
- `sharp_pipeline.arb_watch` (background step of both watch tasks, no credits) posts a FanDuel/DraftKings
  arbitrage once: opposite sides at the same line whose prices return more than the total staked. Table
  `bets.cross_book_arbs`. Information only.
- `sharp_pipeline.draftkings_checks` re-quotes DraftKings through The Odds API about a minute after each
  delivered NBA paper alert (1 credit each, 40 a day, `DK_CHECK_DAILY_CREDITS`); DraftKings' own site
  refuses scripted requests (403), so there is no direct reader like FanDuel's. Table
  `bets.draftkings_price_checks`, report `reports/draftkings_price_checks_latest.md`.
- `sharp_pipeline.discord_replies` (background step of the watch task, no credits) reads the user's Discord
  replies to NBA paper alerts with a bot token (`DISCORD_BOT_TOKEN`; a webhook cannot read) and writes
  `reports/nba_reply_checks_latest.md`: the every-alert $1 paper ledger, plus price checks by reply
  (`y`, `y -125`, `b 5`, `m 8.5 -120`, `n`, `s`; table `bets.nba_alert_replies`). The rules are the pinned
  `sharp_pipeline.reply_trial` (registration `nba-reply-check-v1`, alongside the frozen
  `nba-prop-price-check-v1`, whose alerts it reads). One Discord summary a day; research only.
- `sharp_pipeline.outcomes` adds results (NBA from `nbaprops.player_games`, NCAAF from `ncaaf.games`) and the
  NBA prop shadow filter (latest `nbaprops.projections` before the decision) to the report and digest.
- `sharp_pipeline.gap_validity` (2026-11-12, no credits; `reports/price_gap_validity_latest.md`): checks the
  scanners' assumption that Pinnacle is right, on every stored two-sided prop price against results (NBA 12,889 and NFL
  2,823 settled, three weeks). Weight on Pinnacle where it differs from the book: +0.55 (-0.12 to +1.22) overall, +1.29
  (0.00 to 2.52) on gaps of 3%+; those 260 gaps returned +7.8% (-7.3% to +21.2%) against +4.8% claimed. Leaning the right
  way but not shown: it needs roughly 1,500 such bets. Task `\SuperNovaBets\Price-Gap-Validity`
  (`scripts/price_gap_validity.bat`, every second Wednesday 4:00 from 2026-11-25; registered as in
  `switch_to_nfl_tasks.ps1`, with the XML declaration stripped) reruns it and adds a line to the report's progress table
  (`reports/price_gap_validity_history.json`). Report only: no Discord, no stakes. Until it shows otherwise everything
  stays on paper and models stay context (user decision 2026-11-12).

## Sports Betting Sniper (`src/sports_sniper/`, research and paper trading)

Phase 1 of a standalone opportunity scanner with a Streamlit dashboard; details and what is not built yet are in
`src/sports_sniper/README.md`, settings in `.env.example`.
```bash
python -m sports_sniper.cli scan [--source auto|sample|saved|live] [--sport nba]
python -m sports_sniper.cli dashboard        # http://localhost:8517
python -m pytest src/sports_sniper -q
```
- Every estimate is market consensus: other books' no-vig price for the identical bet (same line), Pinnacle
  weighted 3x, the judged book excluded, ranked by conservative EV. One-sided markets get no probability
  ("Insufficient data"). No prediction model feeds it yet (Phase 2).
- Flagged bets on real prices are entered as paper bets automatically (`auto-consensus-v1`); `python -m
  sports_sniper.cli settle` grades them from `nbaprops.player_games` / `raw.nfl_player_gamelogs` once the game is
  complete and adds closing line value from the last stored snapshot. Bets entered from older snapshots are marked
  replayed. `scripts/sharp_multi_watch.bat` runs `sports_sniper.cli watch` in the background every 10 minutes
  (new saved snapshots, paper bets, settlement; no credits; log `logs/sharp_side_YYYYMMDD_sniper.log`).
- `auto` reads the snapshots the scheduled scanners already saved (`providers/saved_quotes.py`, read-only, no
  credits) and falls back to hypothetical sample data. `live` spends Odds API credits from the shared plan and is
  always explicit; it stops at `SNIPER_MAX_CREDITS_PER_SCAN` and `SNIPER_CREDIT_FLOOR` (2,000).
- Phase 3 (2026-11-11): `python -m sports_sniper.cli backtest` replays the stored snapshots (rules, delayed entry,
  walk-forward, paper bankroll, models against market) and writes `reports/sniper_backtest_latest.md`; also the
  Backtest and Model Comparison tabs. On 19 days the 4% rule is break-even with a wide interval, under half of flagged
  edges are still there one look later, and the NBA projection model is worse than the books' own prices.
- Phases 2 and 4 (2026-11-11): model adapters with walk-forward calibration (the "model view" is recorded beside each
  flagged bet and never changes what is flagged), Breakout Radar, longshot ladder model for alternates (speculative;
  its picks lost), Parlay Lab, Discord alerts for new paper bets (built but off: the user wants the Sniper kept in the dashboard, no Discord;
  do not switch them on or add Discord output to it), and a
  daily evaluation inside `watch` (`cli evaluate`, about 90 s: report, calibration, parlay links). Nothing here has
  shown an edge beyond the price gap; do not let a model rank or flag bets without out-of-sample evidence.
- The dashboard is the alert (2026-11-11, the user's choice over Discord): it refreshes itself every minute, a "Bet now"
  panel lists flagged prices seen in the last 10 minutes with a countdown, and Got it / Gone answers
  (`sniper_price_checks`) measure whether flagged prices can really be bet.
- `sports_sniper.nba_candidate` is the NBA probability challenger (live projections, scale and spread from earlier days'
  results) with calibration by statistic; `nba_props/live.py` and `model.py` are pinned, so candidates live outside them.
  Measured 2026-11-11: the live model's stated spread is about right out of sample, and no variant beats consensus.
  Performance shows forward bets by default and separates "Demonstrated" (prices confirmed with Got it / Gone, an
  append-only log) from research on paper.
- Predictions tab (2026-11-11, `predictions.py` / `predictions_view.py`): the stored forecasts for NCAAF, NBA and NFL games
  (spread, total) and NBA/NFL player props beside the latest stored line, read-only. Home-spread convention as above
  (`margin + spread > 0` = home covers). It shows leans only; it must not flag or paper-bet from them. New sports are added
  as one loader each in `predictions.py`.
- Parlay Lab (2026-11-11, `parlay_view.py` / `parlay_legs.py`): up to 12 legs at one chosen book across games, per-leg
  projection, model chance and last-5/10 hit counts, and one betslip link for the parlay. The link only opens the slip.
  Hit counts and projections are context; they do not change any estimate.
- Phone layout (2026-11-11, `phone_view.py`): narrow-screen styling for everyone (compact header, tabs that wrap) and, on a
  phone or with the sidebar's **Phone layout** switch, the board, longshot, Predictions and Parlay tables as cards.
  Display only; new wide tables should go through `phone_view.table`.
- Task `\SuperNovaBets\Sniper-Dashboard` (`scripts/sniper_dashboard.vbs` -> `sniper_dashboard.bat`, at logon and every 5
  minutes) keeps the dashboard up on port 8517, bound to 127.0.0.1 with no console window. The dashboard has no login:
  remote access is through a private network only (the user's choice is Tailscale, `tailscale serve --bg 8517`); never
  bind it to all interfaces or forward the port.
- Its own tables (`sniper_*`) live in SQLite at `data/sniper/sniper.db` unless `SNIPER_DATABASE_URL` says otherwise.
  It shares no tables, registration or pinned module with the NFL/NBA trials and never places a wager.

## NBA data (`src/nba_props/`)
MySportsFeeds lapsed (403 for every league/season), so `nba_pipeline` has had no data since April 2026.
`nba_props` uses free stats.nba.com via `nba_api`: `ingest` (game logs), `model` (walk-forward test against
pre-game DraftKings lines: no edge, `docs/nba_props_model.md`), `availability` (nba.com lineups, ESPN injuries),
`games` (spreads/totals), `live` (tasks `\SuperNovaBets\NBA-Props-Live` and `NBA-News-Watch`), `post` (#nba slate).
About a third of `odds.nba_player_prop_lines` DraftKings history was snapshotted after tip (live lines);
filter snapshot < commence before using it as evidence.
- `nba_props.selector` (2026-11-10, not scheduled): a classifier that starts from the book's own no-vig price and
  adds the projection. Month by month over two seasons (44,699 DraftKings offers) it only breaks even, and its 269
  picks on saved live quotes sat 4.5% below Pinnacle's close (`reports/nba_selector_backtest_latest.md`,
  `reports/nba_selector_latest.md`). Blind unders lose 3.9%, blind overs 8.5%. Do not schedule it or post its picks.
- `nba_props.challenger` (2026-11-11, not scheduled, about 40 minutes): the pinned walk-forward test with extra inputs.
  DraftKings' spread and total add nothing; a separate minutes model forecasts minutes 10% better but moves the
  projections 0.1%; with the minutes actually played the model would beat the line by 0.67 points
  (`reports/nba_props_challenger_latest.md`). New inputs are judged there; the lever is information about minutes.
  `nba_props.minutes_info` (2026-11-12; `--variants baseline,minutes_info`, `--minutes-only`): who is newly out or back,
  the player's own lay-off and tonight's role cut the minutes miss 4.54 to 4.46 and the points miss 5.142 to 5.118
  (line 5.069). Real but small; picks still lose 2% to 5%, so the pinned live model is unchanged. A run keeps the
  variants it does not refit from the stored report.
  `nba_props.stints` (2026-11-12, not scheduled) stores stats.nba.com rotation stints for every game since 2021-22
  (`nbaprops.rotation_stints`, `rotation_games`; a rerun fetches only new games); `nba_props.stint_info` (`stint_info`
  variant, now in the default run, about an hour) cuts the minutes miss 4.46 to 4.42 but the points miss only 5.118
  to 5.114, which cannot be told from zero. Not adopted. History is close to used up for minutes; what is left is
  same-day news and the game itself.
- `nba_props.news_lag` (2026-11-12, not scheduled, stored data only): do prop prices lag lineup and injury news? No. By the
  first saved price after the news watcher sees a change, FanDuel and DraftKings have finished moving, and the news side
  is worth the same as a blind bet (-5.8% at Pinnacle's close on 2,546 bets over 17 days); even the price 10 minutes
  earlier is -2.6%. The official lineup file and ESPN's list are late and the live model over-reacts to a confirmed
  starter (`reports/nba_news_lag_latest.md`). Do not build alerts on news moves.

## NCAAF game models (`src/ncaaf_pipeline/`, research)
CFBD history (`ingest`, key `CFBD_API_KEY`, 1,000 calls/month) feeds opponent-adjusted points ratings
(`ratings`) and efficiency ratings (`efficiency`). Neither beats the closing line out of sample
(`docs/ncaaf_game_models.md`), so they are context only: `projections` posts the day's slate to #ncaa
(task `\SuperNovaBets\NCAAF-Projections`, 7:30 AM MT) and stores it in `ncaaf.projections`.

## Key Gotchas

- **MSF team abbrs**: NBA uses `BRO` (not BKN) and `OKL` (not OKC). MLB uses `SFG→SF`, `SDP→SD`, `KCR→KC`, `TBR→TB`, `WSN→WAS`, `CHW→CWS`. See `TEAM_ABBR_NORMALIZE` in each crawler.
- **Per-game endpoint dedup**: boxscore uses `_boxscore_is_completed()` (checks `raw.nba_boxscore_games`/`raw.mlb_boxscore_games`) instead of `already_fetched()` — prevents caching partial mid-game scores. All other per-game endpoints use `already_fetched()` with `as_of_date=None`.
- **JSONB player IDs** from MySportsFeeds may serialize as floats — always cast via `::numeric::int` in SQL.
- **`game_training_features` matview schema**: `CREATE OR REPLACE VIEW` rejects column reordering — new columns must be appended. If column count changes, `_matview_needs_recreate()` auto-detects and drops/recreates the matview.
- **Incremental parsers**: `parse_prop_odds`, `parse_game_odds_historical` default to `MAX(as_of_date) - 1`. After a backfill, pass an explicit `since_date` to force reprocessing.
- **Score staleness**: `raw.nba_games` / `raw.mlb_games` are the training spine but lag behind reality. `sync_scores_from_boxscores()` is the fix — called twice in `parse_all` (before and after `parse_boxscore`).
