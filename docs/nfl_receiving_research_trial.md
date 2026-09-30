# FanDuel Receiving Research Trial

This is an observational strategy, not a new source of cash bets. Production and
the existing betting permissions remain unchanged. Forecast deployment and
betting authorization are separate manual decisions.

## Registration

`python -m nfl_pipeline.modeling.benchmark_offers --register-receiving-trial`
registers the currently pinned benchmark exactly once. It refuses a different
artifact; the registration has a checked SHA-256 digest. The active registration
is under `models/receiving_research_trial/registration.json` within the NFL model
root. The benchmark pin itself is not changed.

The first strategy is FanDuel full-game common-line receiving yards using
`stable_ensemble_calibrated`. QB and rushing variants continue collecting in
separate cohorts. Predictions locked before registration are diagnostic only;
they cannot be retroactively included in this strategy's selection record.

## Fixed Selection

- Keep the original locked side, line, price, player, game and bookmaker.
- Require a captured true same-book over/under pair; never synthesize a price.
- Require positive challenger EV and an advantage over the no-vig baseline.
- Require the original drift guard to pass, an HTTPS FanDuel link, and a quote
  no older than 20 minutes at capture. These are observed prices, not a promise
  that the same price will still be available later.
- Rank by challenger EV, breaking ties by original forecast ID.
- Fill remaining daily slots prospectively: at most three per date, two per
  game, one per player/game. Never replace an earlier selection with a later
  winner or retrospectively select the best picks of a completed slate.

The existing daily benchmark-shadow step captures these research selections
automatically. Its exclusive capture lock prevents concurrent jobs exceeding
the cap. A failure is an explicit substep failure, not silently ignored data.
No wager, Discord cash recommendation, or micro ledger entry is created here.

## Matched Evaluation

`python -m nfl_pipeline.modeling.benchmark_offers --report` now reports production,
challenger and no-vig probabilities on identical original offers. The all-offer
and exact-original-micro scopes are separate. Missing paired market evidence is
excluded from the three-way comparison, with counts and reasons; the remaining
models are recomputed on that same subset. Multiple offers share one
player-game's evaluation weight, and uncertainty is clustered by NFL week.

`reports/nfl_receiving_research_trial_latest.json` contains the separate fixed
strategy, all its post-registration candidates, selected results, hypothetical
flat-unit ROI, CLV, interval coverage, concentration, and review blockers.
Research selections are not represented as confirmed placed bets.

Three independent weeks and 50 selected player-games are review floors, not
automatic approval. Review also requires a positive week-grouped Brier gain
against production and market, improved calibration, reasonable 80% interval
coverage (75-85%), improvement in a majority of independent weeks, positive
prospective ROI/average CLV, at least 55% CLV beat, and no heavy concentration.
Maximum observed shares are 20% player, 35% team and 50% week. These preregistered
thresholds are screening rules, not guarantees of a betting edge or profit.

Exact valid close coverage must remain at least 90%. The denominator is all
selected games that have started, not only conveniently settled rows. Unknown
or unavailable CLV is never interpreted as zero movement. Even after all
criteria pass, the result is **manual strategy review only**. Executable prices
and a separate explicit betting decision would still be required.

## Alternate Close Capture

SportsGameOdds close requests now use `includeAltLines=true`. Open/lock/live
requests remain main-line-only; close-only alternates never enter live selection.
Available alternate player quotes are preserved by exact book/player/stat/side/
line. Inactive or unknown alternate availability is excluded. Game spread and
total alternates are not passed into the single-main-game-line parser.

No alternate's odds may be borrowed for a main line, and no opposite side is
invented. The existing resolver still requires the same provider/event/book/
player/stat/line, after lock and within two hours before kickoff. Capturing an
alternate does not relax that rule or the 90% target.

The read-only probe is:

```powershell
python -m nfl_pipeline.audit_alt_close_provider --date 2026-09-24
```

It makes at most one provider request and never writes odds, results or CLV.
Its current quotes cannot repair missing historical capture timestamps.

Provider reference: https://sportsgameodds.com/docs/info/v1-to-v2

## FanDuel Execution and Fresh Locks

The `nfl-fanduel-fresh-lock-v1` execution contract selects FanDuel before scoring,
ranking, and daily caps. Every eligible exact line is scored. A DraftKings price
cannot displace a FanDuel offer. Player identity, both teams, kickoff, market,
line, observed quote timestamp, and price must match. Invalid opposite prices
stay unknown, never synthetic no-vig evidence.

The daily workflow refreshes lock quotes after slow imports/training and before
scoring. Quotes older than 20 minutes, quotes observed after the context cutoff,
and post-kickoff locks are rejected. Persistence checks freshness again before
changing any current flags. No old quote or forecast timestamp is rewritten.
Missing fresh offers leave raw stat projections available, not priced plays.
No-game dates skip the refresh API call.

Paper and micro ledger writes precede pinned challenger capture. Capture is a
required substep and now runs before slower diagnostics. The publisher rejects
stale current quotes; previously locked records are shown only as history.
One player-game cannot occupy multiple micro slots through alternate lines or
subsequent runs. Each research section shows at most one line per player-game,
while all scored offers remain available for matched evaluation.

Scoring source is archived by checksum before its first capture. Older immutable
forecasts use their verified archived scorer for replay, rather than silently
being evaluated with a changed implementation. Unknown or tampered versions
fail explicitly. This does not replace or retrain the frozen model artifacts.

The close diagnostic separates all historical locks, each book, current FanDuel
forecasts, and selected non-paper ledger rows. Only completed close windows enter
the final coverage denominator. Future games show pending, not failed capture.
The 90% requirement is unchanged. Counts are forecast rows, not independent bets.

## Verification Commands

```powershell
python -m pytest src/nfl_pipeline -q --disable-warnings
python -m nfl_pipeline.refresh_lock_quotes --date 2026-09-24
python -m nfl_pipeline.modeling.benchmark_offers --report
python -m nfl_pipeline.close_capture_diagnostic --date 2026-09-24
```

`test_fanduel_live_workflow.py` exercises actual selection/scoring, immutable
forecast persistence, ledger insertion, Discord rendering, alternate close
parsing, and matched production/challenger/market evaluation with mocked I/O.
It covers stale/future/wrong-game quotes, invalid pairs, missing exact closes,
no-game dates, duplicate-player caps, and verified replay archives. No test posts
to Discord, calls an odds provider, or places a wager.

## Automated Slate Checkpoint

`python -m nfl_pipeline.modeling.receiving_trial_checkpoint --date 2026-09-24`
creates dated and latest `nfl_receiving_trial_checkpoint` JSON/Markdown reports.
The daily runner calls it immediately after pinned challenger capture. The close
runner calls it after settlement refresh, grading and CLV attachment. It stays out
of the short near-kickoff capture path so evaluation cannot delay odds retrieval.
Missing current challenger captures return nonzero to the scheduler.

The checkpoint records eligible original locks, capture completeness, remaining
settlements, and exact closing quote identity/timestamps. Quote freshness now is
reported separately from freshness at the original lock. A historical quote
becoming stale today neither becomes an executable recommendation nor loses its
legitimate prospective evidence. Stale original quotes and mismatched captures
are counted as exclusions, not silently replaced.

Daily and cumulative tables compare final production/challenger probabilities,
no-vig baseline, Brier, calibration and interval coverage for all eligible common
FanDuel receiving offers and the exact fixed research selections separately.
The all-offer scope is not restricted to positive EV: doing so would hide bad
forecasts. Duplicated lines do not create independent player-game samples.
Started games determine the 90% close denominator; unknown is never flat CLV.

The forecast-only review uses the existing multiweek Brier, calibration and
coverage criteria. It does not require betting ROI/CLV approval. Passing creates
`prepare_forecast_only_release`, not bankroll approval. The existing explicit
release-review setting remains in place; the checkpoint does not overwrite the
pinned model or alter research selections. A receiving-only replacement should
preserve unrelated stat/game heads and be checked against the pinned scorer
before activation.

For September 24 the installed morning and pregame tasks perform fresh lock
runs; NFL-Close runs every ten minutes around kickoff. A separate Codex follow-up
checks the scheduled work and subsequent weekly evidence at 09:45 and 17:45
America/Denver, notifying only on actionable changes, completed evaluation, or a
deployment review. Future closes and results cannot be validated in advance.
