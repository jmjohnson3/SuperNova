# NFL Prediction and Betting Pipeline Guide

Last reviewed: October 16, 2026

This guide describes the active NFL system in `src/nfl_pipeline`. It explains what the scheduled scripts do, how a player projection becomes a probability at a FanDuel line, what appears in Discord, how results and closing-line value are collected, and what must happen before a recommendation is allowed to risk money.

## Purpose and Boundaries

The system has four separate jobs:

| Layer | Question it answers |
|---|---|
| Stat forecast | What is the player's likely football result? |
| Line pricing | What is the probability of over, under, or push at this exact line? |
| Offer selection | Is the current FanDuel price positive expected value and still executable? |
| Betting authorization | Has this fixed strategy earned permission to risk money? |

Those are deliberately not the same decision. A better yardage model can be deployed without authorizing a wager. A projection above a line is not automatically a bet. A paper recommendation is not a sportsbook wager.

The current execution policy is intentionally narrow:

- FanDuel is the execution book.
- Receiving yards is the only player-prop stat currently permitted to enter a staked forecast tier.
- Game spreads and totals are forecast and graded, but the forecast path does not currently authorize game wagers.
- The separate sharp-price watcher is in research mode.
- No script places a bet at FanDuel. The user must place and, where required, confirm every real wager.

MLB has a separate pipeline. This document covers the active NFL implementation.

## System Overview

The high-level flow is:

```text
Historical and same-week data
        |
        v
Pregame features -> player/game forecasts -> outcome distributions
                                              |
Fresh FanDuel offers -> no-vig market curve --+
                                              |
                                              v
                           final probability at each exact line
                                              |
                                              v
                         EV, price guard, context and model gates
                                              |
                    +-------------------------+------------------+
                    |                         |                  |
               paper/research             $1 micro          cash/bankroll
                    |                         |                  |
                    +------------> immutable forecast/ledger <---+
                                              |
                                              v
                     exact close -> result -> CLV/ROI/calibration
                                              |
                                              v
                             challenger and cash-policy review
```

The main orchestration files are:

- [run_daily_and_notify.py](../src/nfl_pipeline/run_daily_and_notify.py): refreshes data, scores games and props, locks eligible ledger rows, and publishes Discord matchup cards.
- [run_pregame.py](../src/nfl_pipeline/run_pregame.py): starts game-aware pregame runs near each kickoff wave.
- [run_close_and_grade.py](../src/nfl_pipeline/run_close_and_grade.py): captures closing prices, refreshes results, grades forecasts, attaches CLV, and updates evidence.
- [run_training.py](../src/nfl_pipeline/run_training.py): runs the slower offline training and validation workflow.

## Daily Lifecycle

### Morning run

The daily runner resolves the next NFL slate rather than assuming that football is played every day. Its normal sequence is:

1. Verify the database schema.
2. Refresh schedules, final results, rosters, depth charts, injuries, and usage history.
3. Refresh same-day injury information.
4. Refresh FanDuel lock quotes and parse normalized offers.
5. Score game markets.
6. Capture team workload context and score player stats.
7. Audit the FanDuel lock contract.
8. Write paper recommendations and at most three qualifying $1 micro recommendations to their ledgers.
9. Capture production and challenger evidence on identical offers.
10. Refresh readiness and cash-policy reports.
11. Build one Discord card per matchup and publish it.

The scheduled daily command skips full retraining. Training has its own weekly task so a long model job cannot delay live scoring.

### Game-aware pregame run

The game-aware runner wakes every five minutes, reads actual kickoff times, and groups games into kickoff waves. A wave is run once inside the pregame window, with retries only for known pre-publication failures.

The fast pregame path prioritizes:

- current roster, depth, and injury context;
- fresh FanDuel quotes;
- game and player scoring;
- immutable forecast and ledger writes;
- Discord publication.

Heavy historical reports are excluded from this path. The child process also leaves a safety margin before kickoff. A PostgreSQL advisory lock and a state file prevent duplicate concurrent waves.

### Close and settlement run

The close runner also wakes every five minutes. It has two modes:

- Capture mode near kickoff: collect and verify exact closing offers quickly.
- Settlement mode after games: refresh final stats, grade rows, attach CLV, update ledgers, and run validation reports.

Capture takes priority. Slow postgame reports are nonfatal and cannot be allowed to consume a later game's close window.

## Data Sources

| Data | Primary implementation | How it is used |
|---|---|---|
| Schedule and player results | nflverse imports | Kickoff identity, historical training rows, final game and player statistics |
| Rosters | nflverse roster releases | Current team, position, active roster context |
| Depth charts | nflverse depth-chart releases | QB/RB/WR/TE role and movement |
| Injury/practice reports | nflverse plus same-day ESPN summaries | Availability, limited-role risk, teammate vacancies |
| Snaps and participation | nflverse snap and participation files | Workload, role stability, route proxies |
| Play by play | nflverse PBP | Air yards, red-zone work, goal-line work, end-zone targets |
| FanDuel and reference odds | provider abstraction | Current lines, paired prices, links, locks, and closes |

The odds provider order is configured as:

1. SportsGameOdds
2. TheRundown
3. The Odds API
4. Manual CSV

Each source still has to produce valid normalized rows. A nonempty raw payload is not treated as a usable offer. TheRundown currently provides a useful raw/game-odds fallback, but its mapping is not trusted as a full player-prop normalizer without a verified live sample.

The normalized odds tables retain provider, event, book, player, market, exact line, both side prices, links, source timestamp, and snapshot role.

## Pregame Context and Leakage Protection

Every live forecast is supposed to use information known before its lock:

- the game's scheduled kickoff;
- the forecast context cutoff;
- timestamped roster, depth, injury, and practice observations;
- quotes fetched before kickoff;
- historical games that were already complete.

The forecast store refuses to persist a pregame forecast after kickoff or for an unknown game. Game-scoped runs refuse rows from other games.

Missing information stays missing. In particular:

- no injury row does not mean healthy;
- no participation evidence does not mean zero yards;
- no exact close does not mean zero CLV;
- a route proxy is not relabeled as measured routes.

Same-day ESPN injury and inactive data supplement slower nflverse files. When a previously listed player disappears from the live report, the system records a timestamped cleared state instead of silently deleting history.

## Player Stat Forecasts

The active prop markets are:

- quarterback passing yards;
- quarterback and running back rushing yards;
- quarterback passing touchdowns;
- running back rushing touchdowns;
- RB/WR/TE receiving yards;
- RB/WR/TE receptions;
- WR/TE receiving touchdowns.

The model first creates one football forecast per player-game. Sportsbook offers are joined afterward to price exact lines. Multiple books, sides, and alternate lines therefore do not become fake independent player-game training samples.

### Opportunity and efficiency

Yardage is decomposed into opportunity and efficiency:

- Receiving yards = target/workload distribution x yards-per-target distribution.
- Rushing yards = carry/workload distribution x yards-per-carry distribution.
- Passing yards = attempt distribution x yards-per-attempt distribution.

Recent evidence is allowed to move workload more than stable player skill. Efficiency uses longer history and heavier shrinkage because one long catch or one poor game should not radically redefine player ability.

Team constraints help keep the forecasts coherent:

- project team pass attempts and rushing attempts;
- allocate targets and carries among eligible players;
- account for starter status, depth, teammate availability, and role volatility;
- keep player allocations within realistic team opportunity.

Workload is represented with limited, normal, and spike paths. This is meant to distinguish a stable starter from a volatile backup and to represent upside volume without simply inflating every mean projection.

### Baselines and fallbacks

Each target is compared with simple alternatives such as recent player averages, longer-term role estimates, position/depth baselines, and market-informed references. A trained model that does not beat the accepted baseline on chronological later data is not automatically used.

The system can fall back to a conservative baseline for a target while a challenger continues collecting evidence.

### Touchdowns

Touchdowns are rare events. The useful output is primarily a probability such as `P(any receiving TD)`, not a normal continuous count forecast. Alternate multi-touchdown lines require separate validation.

## Game Forecasts

Game scoring produces:

- expected home margin for spread markets;
- expected combined points for totals;
- a distribution around each value;
- over/under or home/away probabilities at an offered line.

Game features include team efficiency, pace/play volume, pass rate, yards per play, red-zone performance, market context, and available QB/OL/skill-player injury context.

The current game forecast rows remain paper-only. The code explicitly labels them `nfl_v1_game_projection_only_no_bankroll_gates`, and the active betting preference contains no approved game market.

## Forecast Distributions

A mean is not enough to price a bet. The scripts create a distribution so they can answer:

- expected value or mean;
- median, representing a typical outcome;
- lower and upper ranges such as p10 and p90;
- probability above an exact line;
- probability below an exact line;
- push probability on integer lines.

These outputs are evaluated differently:

| Output | Main evaluation |
|---|---|
| Expected yards | RMSE and bias |
| Typical outcome or median | MAE |
| Over/under probability | Brier score and calibration |
| p10-p90 interval | Empirical coverage, targeting about 80% |

The model registry can deploy a different component for expected mean, median, and probability. Improving MAE does not prove that the betting probability improved.

## Offered-Line Probabilities

For every fresh exact FanDuel offer, player scoring performs this sequence:

1. Build the raw stat distribution.
2. Apply accepted distribution and interval calibration.
3. Calculate over, under, and push probabilities at the exact line.
4. Preserve a complete trace of raw and adjusted probabilities.
5. Build FanDuel's paired no-vig probability curve.
6. Enforce monotonicity across alternate lines: a higher over line cannot be more likely.
7. Use FanDuel as the baseline.
8. Permit only a small, regularized model residual when a deployed chronological probability policy, independent same-line consensus, and adequate context all agree.
9. Recompute the best side, EV, minimum acceptable price, and drift status from that final probability.

Legacy calibrations remain in the audit trace, but they are not allowed to manufacture a live edge when no validated whole-distribution policy is deployed.

The FanDuel residual adjustment is capped at four percentage points. Its model weight is further reduced by weak context, limited-role risk, route proxies, unstable roles, or market dispersion.

This design explains why the displayed independent player projection can be far from the line while the final betting probability remains near FanDuel's 50% no-vig probability. The independent forecast is evidence; the final probability is the betting input.

## Market Baseline, Edge, and Expected Value

American odds contain the sportsbook margin. For a paired over and under:

1. Convert each price to an implied probability.
2. Divide each implied probability by their sum.
3. The result is the no-vig market probability.

For example, both sides at -113 each imply about 53.1% before removing vig, but normalize to 50% each.

For a $1 stake and no push:

```text
EV = P(win) x profit_if_win - P(loss)
```

With pushes:

```text
EV = (1 - P(push)) x [P(win | non-push) x profit_if_win - P(loss | non-push)]
```

A push returns the stake. The final probability used in EV is the same probability used for ranking, the minimum-price calculation, and Discord.

Example: at -113, a 55% win probability has approximately:

```text
EV = 0.55 x (100 / 113) - 0.45 = +0.0367
```

That is about +3.7 cents per dollar in model expectation, not a promise of profit on one bet.

`Edge` in Discord is the final probability minus FanDuel's no-vig probability. It is not merely projected yards minus the line.

## Offer Selection and Ranking

FanDuel selection happens before daily ranking and caps. An offer must have:

- exact game and player identity;
- the requested stat and exact line;
- a quote fetched before the scoring cutoff and before kickoff;
- a quote no more than 20 minutes old;
- valid FanDuel prices;
- a usable FanDuel link;
- a valid event and offer identifier.

Serious betting proof requires a true same-book over/under pair. An opposite side is never synthesized from another book.

The live probability residual additionally requires:

- a deployed chronological probability policy;
- stable independent market consensus supporting the same direction;
- usage-context quality of at least 68%;
- limited-role risk no greater than 42%;
- role stability of at least 55%;
- known same-week injury context;
- route context for receiving markets;
- a verified receiving opportunity/efficiency component for receiving yards.

A micro candidate then needs positive EV, a sufficient model-market difference, acceptable uncertainty, an approved projection model, and a passing drift guard. Current yardage candidates need at least a 2.5 percentage-point edge over the no-vig market baseline.

The conservative ranking score discounts:

- unknown or weak context;
- limited workload;
- unstable roles;
- unusually large model-market disagreement;
- market-price dispersion;
- proxy rather than measured route evidence.

The ranking penalty changes priority, not the displayed final probability or EV. Five picks is never a quota. The current micro maximum is three across the slate, and zero is a valid result.

## Research, Micro, and Bankroll Tiers

| Discord/ledger label | Meaning |
|---|---|
| Projection only | A player forecast exists, but no current eligible line is attached. |
| Paper / research only | Stored and later graded for learning; stake is zero and it is not a betting instruction. |
| $1 MICRO TEST | A qualifying recommendation locked at $1 for live evidence collection. The user must still verify and place it manually. |
| Cash trial | A fixed strategy has passed its registered review and reserved a recommendation under cash limits. Manual placement and confirmation are still required. |
| Bankroll | A separately approved strategy and tier. No NFL game market currently has this approval. |

A paper row may show the side with the less-negative EV when neither side is positive. That is useful diagnostic output, but it remains explicitly `research only`. It must not be read as "the system says to bet this side."

Current code allows only receiving yards into a non-paper player-prop tier. Passing, rushing, receptions, touchdowns, spreads, and totals continue to be stored and graded as research until each earns its own approval.

## Immutable Forecast Locks

Forecast history is append-only:

- a new scoring run marks prior rows as no longer current;
- it inserts a new revision rather than rewriting the old forecast;
- every row receives a decision key, model version, feature contract, timestamp, quote identity, and full forecast payload;
- the store refuses forecasts after kickoff.

The ledger copies the exact selected prediction ID, book, side, line, price, link, model version, timestamp, tier, and stake. A later Discord card cannot cancel or rewrite that historical decision.

The automated micro ledger records recommendations, not confirmed sportsbook wagers. Cash execution has a separate confirmation record so simulated recommendations cannot be mistaken for money actually placed.

## Discord Messages and Bet Links

Discord is a presentation layer over locked database rows. It does not independently invent picks.

Each matchup card can contain:

- cash-trial status;
- bankroll selections;
- actionable $1 micro props;
- starting skill-player projections for QB1, RB1, WR1, WR2, and TE1;
- quote-health warnings.

Players without an eligible FanDuel offer are labeled `Projection only - no current line`. "FanDuel line available" means the player has a matched offer; it does not mean the offer passed the betting gates.

Common fields mean:

| Field | Meaning |
|---|---|
| `model` | The independent player-stat forecast retained for evaluation |
| `line` | FanDuel's exact threshold |
| `range` | The forecast interval, normally p10-p90 |
| `P(over)` or `P(under)` | Final probability used for the decision, conditional on non-push |
| `Mkt` | FanDuel paired no-vig probability |
| `Edge` | Final probability minus market probability |
| `EV` | Expected profit per $1 at the displayed current price |
| `Cur` | Locked/current American price used in scoring |
| `Min` | Worst acceptable price that still has positive EV |
| `Drift` | Whether the observed price still passes that minimum |
| `ctx` | Same-week usage/context quality |
| `spike` | Estimated high-workload probability |
| `depth` | Current depth rank |
| `prac` | Known practice status |

Only sections explicitly labeled as bankroll, cash trial, or `$1 MICRO TEST` are betting instructions. The footer repeats this rule.

FanDuel links are validated before a real tier is allowed. If the saved URL is missing, belongs to another book, or is not a valid direct betslip link when one is required, the row is demoted to paper.

## Close Capture and CLV

The close scheduler runs every five minutes and targets each game at:

- T-120 minutes;
- T-60 minutes;
- T-20 minutes;
- T-5 minutes.

Checkpoint state records which window was actually captured. A later quote cannot be relabeled as a missed earlier checkpoint.

CLV is attached only when the system can match the exact:

- game;
- provider/event identity;
- book;
- player;
- stat;
- side;
- line;
- valid pre-kickoff time.

The current close taxonomy distinguishes:

- `valid_close`;
- `missing_close`;
- `stale_close_before_lock`;
- `close_outside_two_hour_window`;
- `close_after_kickoff`;
- `exact_line_unavailable_in_captured_feed`;
- `same_line_missing_side_price`;
- `player_market_not_observed_in_capture`;
- `exact_line_mismatch`;
- `legacy_lock_unverified`.

CLV is stored as the change in implied probability from lock to close for the selected side. Positive CLV means the close assigned that side a higher implied probability than the locked price did.

CLV is an early signal that a price may have been good; it is not a guarantee of profit. Missing CLV stays null. The code does not substitute a stale alternate line or call unavailable CLV zero movement.

The target is at least 90% valid exact-close coverage among eligible completed rows. Future games are pending, not failures. Repeated historical forecast revisions are not independent bets, so reports also show current and selected cohorts separately.

## Results, Participation, and Grading

After games, the settlement runner:

1. Refreshes the nflverse schedule and player game logs.
2. Refreshes participation and usage data.
3. Joins predictions to exact player, team, and game results.
4. Grades over, under, and push at the stored line.
5. Repairs receiving results only when participation is independently verified.
6. Reconciles unresolved rows.
7. Updates ledger result and profit from the stored price.

Profit per unit is:

- positive-price win: `price / 100`;
- negative-price win: `100 / abs(price)`;
- loss: `-1`;
- push or verified nonparticipant void: `0`.

No offensive snaps or actions is not enough by itself to grade a winning under. Without verified participation, the row remains `pending_participation_review`. A nonparticipant is voided only when the evidence supports it.

## Ledgers and Performance Measurement

The database keeps distinct records for:

- every forecast revision;
- game and player results;
- CLV labels;
- paper recommendations;
- micro recommendations;
- confirmed cash executions;
- sharp-watcher alerts and confirmed placements.

Evaluation separates:

- all eligible offered lines;
- the exact rows selected by the ranking policy;
- paper recommendations;
- micro recommendations;
- confirmed real wagers.

This matters because a model can improve across all offers while its top-ranked picks remain poor. Multiple lines on one player and multiple books on one game are grouped so they do not masquerade as independent evidence.

The main betting metrics are:

- win/loss/push record;
- profit and ROI;
- Brier score;
- calibration error;
- interval coverage;
- CLV beat rate;
- average CLV;
- player, team, game, and week concentration.

## Production Models and Challengers

Production and challenger models are intentionally separate.

Production provides the stable live forecast. Challengers are scored on the same pregame inputs, exact line, side, price, and lock time. They do not affect the live decision until a validation policy accepts them.

Validation uses chronological or week-grouped splits:

- train on earlier data;
- calibrate on independent earlier held-out predictions;
- test on later weeks;
- compare with the current production model;
- compare with simple baselines;
- compare offered-line probabilities with FanDuel's no-vig baseline;
- evaluate selected picks separately from all offers.

The registry supports deployment by stat and output:

- expected mean;
- median;
- full distribution/probability.

This prevents a good receiving-yard mean model from silently replacing a better receiving probability model, or a passing challenger from replacing rushing.

### Current model snapshot

As of this guide's review:

- Base release: `nfl-20260918T143342Z`.
- Feature contract: `nfl-asof-v2`.
- The per-output registry has registered receiving and passing candidates.
- No candidate is active in the registry yet.
- Registered candidates are not betting-approved.

This is why substantial offline work can exist while production remains unchanged: a candidate is an experiment until it wins the metric for its specific output on later data.

## Deployment Versus Betting Approval

There are two independent decisions.

### Forecast deployment

A component may replace a weaker forecast when it improves the relevant later-week metric without unacceptable regressions:

- expected mean: RMSE and bias;
- median: MAE;
- probability: Brier and calibration;
- uncertainty: interval coverage.

It does not need to qualify for cash merely to become the more accurate forecast.

### Betting approval

A fixed selection strategy needs more:

- credible selected-pick probabilities;
- a fresh executable FanDuel price;
- positive conservative EV;
- working links;
- clean exact closes;
- prospective evidence across independent weeks;
- acceptable calibration and interval coverage;
- no severe player, team, or week concentration;
- positive ROI and supportive CLV.

The current registered FanDuel common-line receiving strategy remains research. Its review thresholds are evidence floors, not automatic approval. Reaching 50 decisions or three weeks only triggers review.

## Scheduler and Midnight Rollover

The repository's Windows Task Scheduler definitions currently specify:

| Task | Cadence | Purpose |
|---|---|---|
| NFL-Daily | Daily at 7:30 AM local | Main refresh, forecasts, ledgers, reports, Discord |
| NFL-PrimeTime-Refresh | Every 5 minutes | Game-aware pregame wave detection and scoring |
| NFL-Close | Every 5 minutes | T-120/T-60/T-20/T-5 capture and settlement |
| NFL-Training | Tuesday at 3:00 AM local | Offline training and broad validation |
| NFL-Sharp-Watch | Every 10 minutes | Separate book-versus-sharp research scanner |

The batch wrappers use separate mutexes so duplicate task instances do not overlap.

After midnight, the close runner does not simply abandon yesterday's slate. It first looks for nearby live games, then revisits recent final slates with unresolved grades. Unresolved dates are rotated by their last import attempt. Results are retried more frequently while pending and less frequently once routine refresh is sufficient.

This handles player statistics that publish after the ET date changes without inventing results or rewriting old forecasts.

## Failure Handling and Trust Checks

The pipeline fails closed around money-sensitive operations:

- stale or post-kickoff quotes cannot be locked;
- missing paired prices cannot create no-vig evidence;
- a missing FanDuel link demotes a real tier to paper;
- a game-scoped run cannot write another game's rows;
- missed close checkpoints stay missed;
- missing results and participation stay unresolved;
- an uncertain Discord network response is not blindly reposted;
- advisory locks prevent overlapping forecast, pregame, close, and training work.

Core close capture, parsing, grading, CLV, and ledger refresh failures are operational failures. Slow analytical reports are warnings after the critical work is complete. This means a validation timeout should not erase a successful close capture or stop a result from being graded.

Atomic JSON writes protect release, state, and report artifacts from partial files. Run manifests and live-cycle audits make it possible to trace what was published.

## Separate Sharp-Price Watcher

The sharp watcher is not the player projection selector.

It compares FanDuel with a sharper reference book or consensus, removes vig, measures the price gap, and logs possible edges. It has its own:

- markets;
- polling window;
- evidence report;
- stake limits;
- loss pause;
- manual confirmation command.

Current sharp-watcher staking is disabled, so its Discord rows are `SHARP EDGE - research, not a bet`. Nothing is placed automatically. If staking is later authorized, only a manually confirmed alert counts as a real wager and the taken price is stored so slippage is visible.

## Reports and Where to Look

Start with these:

| Question | Report |
|---|---|
| Did today's workflow complete? | `reports/nfl_daily_run_latest.json` |
| Are data, odds, forecasts, and ledgers present? | `reports/nfl_readiness_latest.md` |
| Were the lock quotes fresh and valid? | `reports/nfl_lock_quote_health_latest.json` |
| Did exact close capture work? | `reports/nfl_close_capture_diagnostic_latest.md` |
| Did the complete lock-to-settlement cycle pass? | `reports/nfl_live_cycle_latest.md` |
| How did final displayed probabilities perform? | `reports/nfl_final_probability_validation_latest.md` |
| Did ranking choose worse errors? | `reports/nfl_selected_pick_diagnostic_latest.md` |
| Is the receiving challenger improving? | `reports/nfl_receiving_trial_checkpoint_latest.md` |
| Which per-stat component can deploy? | `reports/nfl_stat_model_validation_latest.md` |
| Is any cash strategy approved? | `reports/nfl_cash_readiness_latest.md` |
| Is exact-line history sufficient? | `reports/nfl_prop_exact_line_proof_latest.md` |
| What is the CLV record? | `reports/nfl_clv_scorecard_latest.md` |
| How is the sharp watcher performing? | `reports/nfl_sharp_alerts_latest.md` |

The main implementation references are:

- [betting_preferences.py](../src/nfl_pipeline/betting_preferences.py)
- [offer_selection.py](../src/nfl_pipeline/offer_selection.py)
- [predict_player_props.py](../src/nfl_pipeline/modeling/predict_player_props.py)
- [predict_today.py](../src/nfl_pipeline/modeling/predict_today.py)
- [forecast_store.py](../src/nfl_pipeline/forecast_store.py)
- [lock_ledger.py](../src/nfl_pipeline/lock_ledger.py)
- [discord_matchups.py](../src/nfl_pipeline/discord_matchups.py)
- [clv_report.py](../src/nfl_pipeline/clv_report.py)
- [grade_predictions.py](../src/nfl_pipeline/grade_predictions.py)
- [cash_readiness.py](../src/nfl_pipeline/cash_readiness.py)

## End-to-End Example

Suppose FanDuel offers a receiver over/under 63.5 yards at -113 on both sides.

1. The player model may independently forecast 68 yards with a wide role-aware range.
2. The distribution converts that forecast into probabilities at 63.5, including any push probability.
3. FanDuel's paired prices normalize to 50% over and 50% under.
4. A deployed chronological model might estimate 54% over.
5. If independent consensus agrees and context is strong, the residual layer may give the model limited influence, perhaps producing 51.2% rather than accepting all 54%.
6. At -113, 51.2% is negative EV because break-even is about 53.1%.
7. The row can still appear in paper research, but it cannot become a $1 micro instruction.
8. If a later refresh improves the price enough for positive EV and every other gate passes, the exact offer can be locked as a micro recommendation.
9. Near kickoff, the exact 63.5 line and selected-side price are captured again if still available.
10. After the game, the stored forecast is graded and compared with the exact close.

The same logic explains an apparently odd paper side. When both over and under are negative EV, the research row can display whichever side is less negative. Only the explicit tier says whether it is actionable.

## Current Limitations

The most important known limitations are:

- True measured route history is not consistently available; many rows use explicitly labeled pass-route proxies.
- Historical same-week injury and depth context is sparse compared with live context.
- Injuries during a game cannot be predicted. They should appear as tail risk, not as fake pregame certainty.
- NFL player outcomes remain noisy even with good opportunity forecasts.
- Exact-line learned models need more independent locked, closed, and graded live history.
- Alternate lines can disappear before close, making exact CLV genuinely unavailable.
- FanDuel's market is a strong baseline; large model disagreements have not earned broad trust.
- The active per-output registry currently has no deployed challenger.
- Cash approval currently has insufficient post-policy selected evidence.

On the October 15 completed slate, the current FanDuel cohort captured 39 of 39 exact closes, while repeated historical revisions lowered the all-lock-revision figure. The current cohort is the operational signal; duplicated revisions are retained for audit but are not independent bets.

## Operational Checklist

Before kickoff:

- Confirm the correct slate and kickoff times.
- Confirm the pregame task ran for the correct game IDs.
- Confirm FanDuel quotes are less than 20 minutes old.
- Confirm both side prices and a valid link exist.
- Confirm injury, practice, and depth context quality.
- Confirm production and challengers were captured on identical offers.
- Treat only explicit cash, bankroll, or `$1 MICRO TEST` sections as instructions.
- Verify the sportsbook price still meets `Min` and drift is still acceptable.

After kickoff:

- Confirm all four close checkpoints were attempted.
- Confirm exact current FanDuel close coverage is at least 90% after the window completes.
- Leave unavailable exact lines unknown.

After settlement:

- Confirm games are final.
- Confirm player results and participation are attached.
- Resolve pending participation without converting it to zero.
- Refresh ledger result, profit, and CLV.
- Compare production, challenger, and FanDuel on identical offers.
- Review selected picks separately from all offers.
- Keep production stable unless the appropriate later-week metric actually improves.

## Glossary

**Baseline:** A simple or accepted reference forecast the challenger must beat.

**Brier score:** Mean squared error of a probability forecast; lower is better.

**Calibration:** Whether events predicted at a given probability occur at approximately that rate.

**CLV:** Closing-line value; how the exact selected price compares with the same line near kickoff.

**Context quality:** Confidence that current role, injury, depth, and usage information is complete enough to support a decision.

**Drift guard:** A check that the current price has not moved beyond the minimum positive-EV price.

**ECE:** Expected calibration error; lower is better.

**EV:** Expected value per unit staked, calculated from the final probability and executable price.

**Exact line:** The same book, player, stat, side, and numerical threshold that was locked.

**Immutable lock:** A historical decision record that later forecast refreshes cannot rewrite.

**Line anchoring:** Pulling an uncertain projection or probability toward the sportsbook market.

**Market no-vig probability:** The paired book probability after removing the bookmaker margin.

**Micro:** A small live test, currently $1 flat, used to collect prospective evidence.

**Monotonicity:** The requirement that an over becomes no more likely as its line rises.

**Paper:** A zero-stake research recommendation.

**Production:** The stable model/component currently used in live scoring.

**Push:** The actual result equals an integer line; the stake is returned.

**Residual model:** A model that starts from the market and predicts a small correction rather than replacing the market wholesale.

**Sharp watcher:** The separate price-comparison scanner; it is not the player projection model.

**True paired:** Real over and under prices from the same book for the exact same line.

**Challenger:** A candidate model scored alongside production but not live until it passes validation.
