# NBA player-prop model (nba_api)

Status (2026-10-24): **research only; no model-driven NBA prop bets.** NBA bets, when staked, come from the
sharp-gap scanner (`sharp_pipeline`), which prices DraftKings/FanDuel/theScore against Pinnacle.

## Why a new pipeline
The MySportsFeeds subscription lapsed: every request (any league, any season) returns 403 "Access
Restricted", so `nba_pipeline` has had no data since April 2026. `nba_props` uses the free stats.nba.com
API (`nba_api`): `python -m nba_props.ingest --seasons 2021-2026` loads every player-game into
`nbaprops.player_games` in a few calls; `python -m nba_props.ingest` refreshes the current season.

## Historical lines are partly live
`odds.nba_player_prop_lines` (DraftKings 2023-10 to 2026-04) was backfilled from Odds API snapshots at
01:00 UTC. For 1,366 of ~3,000 games (38k of 110k rows) that snapshot is after tip: in-game lines. Any
earlier "model beats the book line" result that used those rows is not evidence. `nba_props.model`
keeps only snapshots before tip (and at most 30 hours before).

## Model and test (`python -m nba_props.model`; reports/nba_props_model_latest.md)
LightGBM (Poisson) per stat on as-of features (decayed minutes, per-minute rates, shot volume, rest,
home, team pace, opponent allowed rates), refit monthly, Nov 2023 to Apr 2026. Negative-binomial
dispersion turns a projection into P(over).

| Stat | Model MAE | DK line MAE | Log loss model | Log loss DK no-vig | ROI, all model picks |
|---|---:|---:|---:|---:|---:|
| Points | 5.20 | 5.07 | 0.714 | 0.692 | -4.6% |
| Rebounds | 2.07 | 2.06 | 0.721 | 0.689 | -5.9% |
| Assists | 1.78 | 1.78 | 0.724 | 0.686 | -5.4% |

The model's weight in a regression of the result on line and model is positive (+0.14 to +0.32), so it
holds a little information the line lacks. A logistic blend anchored on DraftKings no-vig, trained on
earlier seasons only, matched the market's log loss (better in one season, worse in the other for each
stat) and its picks flipped sign between 2024-25 and 2025-26. Most of what the blend adds is shrinking
DraftKings' extreme prices, not the model. Verdict: no reliable edge against even DraftKings' pre-game
line, which is softer than Pinnacle's close.

## Live uses (since 2026-10-24)
- `python -m nba_props.live` (task `\SuperNovaBets\NBA-Props-Live`, 6:45 AM, 11:00 AM, 3:30 PM MT) refreshes
  the game logs and appends the day's projections to `nbaprops.projections` with their run time.
- `sharp_pipeline.outcomes` grades the sharp scanner's decisions with results (NBA from the game logs,
  NCAAF from `ncaaf.games`; a prop with no game log for a played game is void; first halves ungraded) and
  records the shadow filter: the model's probability for the decision's side from the latest projection
  made before it, and whether it is above Pinnacle's no-vig. Both appear in `reports/sharp_multi_latest.md`
  and the digest; EV at the close stays the primary test. Projections are not reconstructed for days
  before the first run, so the shadow split starts with Oct 24 decisions.
- Judge the filter on 100+ NBA prop decisions: it earns a role only if "agrees" beats "disagrees" on EV
  at the close.
- Re-run the test each season; adopt nothing unless it beats the market's log loss in two straight seasons.
