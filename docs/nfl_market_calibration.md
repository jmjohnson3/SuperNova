# Market calibration, bet scope and receptions (October 1)

## Why

Weeks 1-3 of 2026 (2,669 settled over/under prop forecasts, 1,276 paper picks):

| Evidence | Result |
|---|---|
| Paper picks | 48.0% won vs 52.7% breakeven, -8.7% ROI; week 3 worst |
| Brier, model vs FanDuel no-vig | 0.258 vs 0.250 |
| Stated vs realised | model said 56% on its side, won 49%; 60-80% buckets won 46-47% |
| corr(projection - line, actual - line) | negative for every stat (receiving -0.10) |
| Projection MAE vs line MAE | passing 63.0 vs 54.7, receiving 22.2 vs 20.8, rushing 19.8 vs 16.8 |
| p10-p90 coverage | receiving 70%, rushing 73% (target 80%) |
| CLV on picks with a valid close | 16% beat close, 34% same, 50% worse |
| Picked vs not picked | identical (both ~49%, ~-7% ROI) |

The model's disagreement with the line carried no information, and its probabilities turned that
noise into apparent edges. Rushing was worst where the model sat above the line (QB rushing +6 to
+13 yards, backup RBs +10, deep backups +24); recent averages early in a season are mostly stale
2025 games.

## What changed

1. **Bet scope** (`betting_preferences.BET_PROP_STATS`, enforced in `lock_ledger` and scoring).
   Only receiving yards may lock into a staked tier. Rushing/passing yards, passing TDs,
   receptions, spreads and totals are still forecast, graded, and shown in Discord with FanDuel
   links as research rows.
2. **Market calibration layer** in `predict_player_props._candidate_from_offer`, parameters in
   `models/player_props/market_calibration.json`, fitted by `modeling/fit_market_calibration.py`:
   - projection anchor: `projection = line + a * (model - line)`; the raw model value is kept in
     `probability_trace.model_projection`.
   - range widening: distribution evaluated at `p50 + (x - p50) / k`; displayed p10/p90 widened.
   - market blend: `logit(p) = logit(market) + w * (logit(model) - logit(market))`.
   Parameters travel inside the captured `metrics`, so replays are exact. No file = identity.
3. **Receptions** market (`player_receptions`; SportsGameOdds `receiving_receptions`), projection
   `_receptions_projection` (recent receptions mix + projected targets x shrunk catch rate; MAE 1.255
   on 6,369 player-games), Poisson distribution, grading and Discord sections. Paper only.
4. **Gamelog source of record**: official nflverse box-score values win; play-by-play fills gaps
   and owns red-zone/end-zone columns (previously ~950 rows flipped every close run).
5. **CLV-first scorecard** (`modeling/clv_scorecard.py`, close run): valid-close rate, CLV beat
   rate and average move, then Brier vs market, then ROI, by stat and calibration version, with
   player-games weighted equally.

## Fitted parameters (market-calibration-20261001T030458Z)

Fit on replayed weeks 2-3 (week 1 forecasts predate replay capture); leave-one-week-out folds.
A stat keeps a fitted probability trust only if it beat the market's log loss in every fold.

| Stat | a | k | w | Out-of-sample |
|---|---:|---:|---:|---|
| receiving_yards | 0.00 | 1.25 | 0.00 | equals market, beats raw model both weeks; coverage 81% |
| passing_yards | 0.00 | 1.00 | 0.00 | equals market, beats raw model both weeks |
| rushing_yards | 0.00 | 1.00 | 1.00 | beats market both weeks (LL 0.678 vs 0.694; 0.688 vs 0.692) |
| passing_tds | 1.00 | 1.00 | 0.00 | mixed (tie, then worse than market) -> market |
| receptions | 0.35 | 1.00 | 0.00 | mixed on 349 recovered FanDuel lines -> market |

Re-scoring all 2,192 captured offers: the share showing >2% EV falls from 35-63% to 0% in every
stat. **No prop currently qualifies as a $1 bet, receiving yards included.** That is the expected,
honest result until a model shows information the market does not have. Rushing keeps a small,
validated lean (mostly unders) but not enough to clear the vig; it remains research.

## Consequences

- The scoring fingerprint changed. The registered receiving cash trial excludes new forecasts as a
  different scoring cohort (no error); continuing it needs an explicit re-registration.
- 148 rushing and ~60 passing player-games is little evidence. Re-check before relying on any of it.

## Re-checking against later weeks

Tuesday training runs `fit_market_calibration` report-only (`reports/nfl_market_calibration_latest.json`).
Judge by `reports/nfl_clv_scorecard_latest.md` first. To install a refit after review:

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.fit_market_calibration --write
```

## October 1: season-aware rushing baseline

Rolling `rushing_yards_avg_5` crosses seasons, so early-season projections lean on stale 2025 games.
`_season_aware_rushing_estimate` shrinks this season's per-game average toward last season's
(k=3 games; k=1 after a team change) and mixes it 50/50 with this-season carry share x team carries
x shrunk YPC. Live rushing projection = 50% frozen model + 50% this estimate; players with no NFL
history keep the model value. The frozen model's features are unchanged.

| Projection (MAE, rushing yards) | Weeks 1-8, 2020-2025 + 2026 (5,584 games) | 2026 settled forecasts (208) |
|---|---:|---:|
| 5-game cross-season average | 19.36 | 20.69 |
| Season-aware estimate | 18.62 | 19.06 |
| Frozen model | - | 19.25 |
| 50/50 model + estimate (live) | - | 18.62 |
| FanDuel line | - | 16.28 |

Lead backs, weeks 1-3: bias +4.8 -> +1.6 yards. It still carries no information beyond the line
(correlation with the line's error <= 0), so priced rushing stays anchored to the line until a refit says otherwise.

## October 1: RB role-change detection

The carry-share half of the season-aware estimate now reacts to role changes (RBs only):
recency-weighted share (half-life 2 games) plus 25% of the season carry share of RB teammates listed
Out/Doubtful on the pregame injury report for that week, split by the remaining backs' shares.
Explicit depth-chart promotion bumps were tested and hurt (recency already captures promotions).

Fit on 2020-2022, judged on 2023-2024 (weekly pregame injury reports; 2025 reports are missing):

| 2023-2024 holdout | All RB games | Games with an RB teammate ruled out (277) | Bias, teammate out |
|---|---:|---:|---:|
| Season share (previous) | 21.72 | 22.80 | -11.1 |
| Recency + 25% vacated (live) | 21.47 | 21.53 | -5.1 |

Full redistribution of vacated carries over-projected (+9.6 yards): carries also go to QBs/receivers
and team volume shifts. On 149 as-of 2026 forecasts the change is neutral (19.71 -> 19.68); only 8
had a ruled-out teammate, so weeks 4+ will show more.

## October 1: spreads and totals

`predict_today` now applies the same logit blend toward FanDuel's no-vig price before EV ranking
(`models/game_bets/market_calibration.json`, fitted by `modeling/fit_game_market_calibration.py`,
report-only in Tuesday training). Each game counts once per market; pre-contract forecasts use the
latest same-line FanDuel quote that existed when they were made. New forecasts store
`model_probability`, `market_no_vig_probability` and `market_trust`.

31 games, weeks 2-3, leave-one-week-out log loss: spreads 0.6871/0.6875 vs market 0.6871/0.6875
(model adds nothing); totals 0.6930/0.6939 vs 0.6931/0.6941. Both fitters now require a real margin
(log-loss gain >= 0.002 in every fold) before trusting the model, so both markets price at the
market (w=0). Prop parameters are unchanged by the margin (rushing won by 0.016 and 0.004).

## October 1: pregame-knowable role features

`features._add_usage_role_features` computed `*_share_avg_5`, `*_role_rank` and
`team_player_*_avg5_sum` among only the players who actually played each game (post-game
information). Teammates expected before kickoff (played in the team's previous 4 games, not listed
Out/Doubtful that week) now count, at their 5-game average entering the game. On 39,148 training
rows the old shares were higher by +0.015 (carries) and +0.016 (targets) on average; ranks change in
48-57% of rows. 2025 has no injury reports in the database, so no one is excluded for that season.

Live impact today: none. The pinned release does not use these features and has workload/opportunity
adjustment off. The fix keeps future challengers and releases from learning inflated shares. Note the
opportunity-model trainer writes directly to `nfl_player_opportunity_models.joblib` (unpinned; only
loaded when no release exists) - retrain it as a challenger file before any future use.

## October 1: sharp-book reference (Pinnacle / EU exchanges)

The model has no information the market lacks, so the edge has to come from price: FanDuel quotes
that are off-market versus a sharp book. The SportsGameOdds tier here blocks sharp books (Pinnacle,
Circa, BookMaker, BetOnline, LowVig); The Odds API EU region returns Pinnacle NFL props.

- `sharp_lines.py` captures EU-region receiving-yards prices once per game at lock (kickoff within
  150 min, from the pregame quote refresh) and once near close (within 25 min), stored as provider
  `oddsapi_eu`. ~2 credits/game; never below a 150-credit monthly floor (plan: 500/month).
- Scoring attaches the freshest sharp two-sided quote (Pinnacle first, then exchanges) for the same
  player, stat and exact line as `offer.sharp_reference` (captured, so replay is exact). It prices the
  offer at the sharp no-vig line (blended by the stat's probability trust, 0 today), picks the FanDuel
  side with the better EV, and records `sharp_book`, `sharp_line`, `sharp_over_probability`, `sharp_ev`.
- Discord listed FanDuel offers with sharp EV >= 3% in a "SHARP-LINE EDGES - research" section.
  That section was removed on October 2: it compared the morning/pregame snapshot, and live gaps
  come from `sharp_watch` instead. Scoring still records `sharp_ev`, and the CLV scorecard still
  reports it as its own strategy (`| sharp-edge`).
- They stay paper (`SHARP_EDGE_BETS_ENABLED = False`) until the scorecard shows positive CLV over
  2-3 weeks; flipping the flag allows $1 micro locks for receiving yards.

## October 1: same-day injury and inactive statuses (ESPN)

nflverse injury files lag days: Thursday 10/1 forecasts used reports from 9/27 and projected Rico
Dowdle for 25.5 rushing yards although ESPN had listed him Out since 9/30. `live_injuries.py` reads
ESPN's public game summary (free, no key) for each game on the slate and writes each team's current
statuses into `raw.nfl_injuries` for that game's week (source `espn_live`, mapped to nflverse IDs via
roster `espn_id`; unmapped players keep their name). The as-of trigger timestamps when we learned
them, the context loader takes the newest status for the week, and a player ESPN drops from its list
gets a `Cleared` row. It runs before scoring in the morning run and in every T-90 pregame run, when
game-day inactives are posted. First run: 10 entries, 7 mapped, 5 Out; Dowdle is no longer projected.
Out/Doubtful teammates also feed the RB role-change carry redistribution.

## October 1: sharp watch, live alerts and the CLV proof

A lock-time snapshot rarely catches a stale FanDuel price, because those get fixed within minutes.
`sharp_watch.py` runs at the start of every 10-minute close-task invocation, in every mode, and decides
for itself whether to poll:
- **Schedule:** hourly from 24h before kickoff, then every 10 minutes in the last 4 hours.
- **Cost:** one call per game returns FanDuel plus Pinnacle and the exchanges together (named
  bookmakers ≤ 10 cost 1 credit per market).
- **Budget:** a daily allowance spreads the remaining credits over the rest of the month, and a
  credit floor (`NFL_SHARP_CREDIT_FLOOR`, default 150) stops it. The free events call reports the
  live balance. Today's games get first claim on their final 90 minutes (T-90 inactives, every
  10 minutes): an earlier check runs only if those checks still fit in the allowance afterwards.
  Final-window checks stop only at the floor, and any overspend lowers later days' allowances.
- **Markets:** `NFL_SHARP_WATCH_MARKETS`, default `player_reception_yds`. Add more, for example
  `player_rush_yds,player_receptions,totals`, once the plan has the credits. Spreads are stored but
  not alerted, because key numbers make line conversion unsafe.
- **Line conversion:** a sharp quote at a nearby line is moved to FanDuel's line with a normal
  approximation (`sharp_math.py`). It is allowed within ±3 yards for receiving and rushing, ±6 for
  passing and ±1 point for totals; receptions must match exactly.
- **Alerts:** a FanDuel side at ≥ +3% EV against the sharp fair price posts a Discord alert with the
  FanDuel link and the worst price still worth +1% ("take at X or better"). Each distinct price is
  logged once in `bets.nfl_sharp_alerts`.
- **Logged gaps (October 2):** sides from +1% up to +3% EV are also stored, with `tier='logged'`
  and no ping. A 3%+ edge at a single game rarely appears (PIT @ CLE: 12 checks, best side about
  0%), so the alert sample alone would take months. If a logged gap later reaches +3% at the same
  FanDuel price, it is upgraded to `alert` and pinged once.

`modeling/sharp_alert_report.py` grades every alert after kickoff. Its outputs are
`reports/nfl_sharp_alerts_latest.{md,json}`, and it runs in the close settlement steps. It reports:
- **EV at the sharp close:** the FanDuel price, valued at the sharp book's last line before kickoff.
- **FanDuel CLV:** whether FanDuel's own close moved toward the alerted side.
- **Results:** last, because results are noise for weeks.

Logged gaps are graded in their own column. They don't count toward the pass bar, but they show
sooner whether FanDuel-vs-sharp gaps predict the close at all.

Pass bar for enabling `SHARP_EDGE_BETS_ENABLED` with fixed small stakes: at least 100 alerts over at
least 3 weeks, at least 55% beating FanDuel's close, and a mean EV at the sharp close of at least +1%.
Size up only after that, at no more than a fraction of Kelly.

First poll (PIT @ CLE, T-3h): 13 of 14 FanDuel receiving props had a Pinnacle line, 9 exact and 4
within 1-2 yards. Every side was negative EV; the best was about -0.2%. A fairly priced slate is
normal, which is why polling frequency matters.

## October 3: historical sharp-gap backtest (2025)

`modeling/sharp_gap_backtest.py` replays the watcher on The Odds API history for sampled 2025 Sundays.
It pulls FanDuel and the sharp books at T-90, T-45, T-15 and kickoff (T-2). Every FanDuel side is
priced against the sharp fair line, then graded at the sharp close, by FanDuel's own close and by
the result. FanDuel moves props by changing the line at a fixed -112/-112 price, so the close is
compared at the bet's line rather than requiring the same line. Each market gets its own report:
`reports/nfl_sharp_gap_backtest_2025_<market>.md`, plus a sides CSV. Paid responses are cached in
`raw.nfl_api_responses` (endpoint `nfl_player_props_history`), so reruns cost nothing.

| Market (games) | T- | Sides >= +1% | >= +3% | EV of >= +3% at the sharp close | Positive at the close |
|---|---:|---:|---:|---:|---:|
| Receiving yards (120) | 90 / 45 / 15 | 4.4% / 4.4% / 4.5% | 1.7% / 2.0% / 1.3% | +2.7% / +2.5% / +3.3% | 75% / 78% / 92% |
| Rushing yards (60) | 90 / 45 / 15 | 6.1% / 7.8% / 7.1% | 1.4% / 2.7% / 2.0% | +3.8% / +3.5% / +4.8% | 86% / 92% / 100% |

- About 1 in 50 FanDuel sides is at +3% or better, roughly 0.35 per game at each check, so a full
  Sunday should produce about 5-8 alerts per market.
- FanDuel rarely corrects late: only about 4% of sides moved between T-15 and kickoff, so gaps can
  still be bet when an alert arrives.
- Pinnacle's close predicted results slightly better than FanDuel's on 967 receiving overs (log loss
  0.6918 vs 0.6934; a coin flip scores 0.6931). Where they disagreed by more than 2 points,
  Pinnacle's side won 53.7% of 257. That points the right way but is not proof.
- Flat ROI on these buckets (7-93 bets each) is noise.

This supports the live watcher and its 3% alert threshold. It does not replace the live CLV proof:
alerts stay paper until live Sundays match the backtest.

## October 5: first live Sunday, ping window and FanDuel CLV fix

Week 4 Sunday: 7 pings and 29 logged gaps across 12 games, 1,196 credits. Four of the 7 pings fired
13-22 hours before kickoff, a window the backtest never tested. Pinnacle's props are thin and move
then. Pings are now limited to the last `PING_WINDOW_MINUTES` (120) before kickoff, which covers the
backtest's T-90/45/15 checks:
- A +3% gap seen earlier is stored as `tier='early'`, without a ping.
- If it is still +3% at the same FanDuel price inside the window, it is upgraded and pinged once.
- The report re-labels pings sent before this rule existed as early when it reads them; stored rows
  are not rewritten.

The live report's FanDuel CLV had the same flaw as the first backtest: it looked up the close at the
alert's line, but FanDuel moves props by line at a fixed price. It now takes FanDuel's last
two-sided quote before kickoff at any line, moves it to the alert's line (`sharp_math.shift_probability`,
shared with the backtest), and compares no-vig probabilities with FanDuel's quote at alert time.

Week 4 after the fix:

| Tier | Count | EV at the sharp close |
|---|---:|---:|
| In-window alerts | 3 | -6.9% |
| Early | 4 | +1.2% |
| Logged | 29 | -0.8% |

Too few to judge. The pass bar and threshold are unchanged.
