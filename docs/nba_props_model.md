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

## Availability (v2, 2026-10-24)
`nba_props.availability` captures nba.com daily lineups (starters, Active/Inactive; history from 2023-24,
backfilled: 3,977 games) and ESPN's injury report (current only). `model.add_availability` gives each
player-game its starter flag and recent starting rate, and each team and opponent its missing rotation:
rotation players (>= 12 recent minutes, on the team in the last 15 days) listed Inactive, with their recent
minutes, usage, points, rebounds, assists and threes. Live runs use lineup Inactive plus ESPN "Out", the same
meaning as training (Inactive). Same walk-forward test, same rows (v1 report: reports/nba_props_model_v1_latest.md):

| | v1 (no availability) | v2 (availability) | DraftKings |
|---|---:|---:|---:|
| Points MAE, rows with a line | 5.20 | 5.14 | 5.07 |
| Rebounds MAE | 2.07 | 2.06 | 2.06 |
| Points log loss (raw model) | 0.714 | 0.707 | 0.692 |
| Rebounds log loss (raw model) | 0.721 | 0.704 | 0.689 |
| Model weight in result ~ line + model (pts / reb / ast) | 0.14 / 0.27 / 0.32 | 0.27 / 0.37 / 0.39 | |
| Blend anchored on DK no-vig, out of sample (2024-26), log loss | 0.68909 | 0.68845 | 0.68898 |
| Blend picks, +3% vs DK: ROI 2024-25 / 2025-26 | -3.7% / +1.4% | -2.3% / 0.0% | |

Availability is real information the line does not fully hold (the anchored blend now beats DraftKings'
own log loss; v1 did not), but the gain is small and the picks still do not clear the vig against the
pre-game line. Next: the morning snapshot (from Nov 3) tests whether it predicts where morning lines go.

## Spreads and totals (`python -m nba_props.games`; reports/nba_games_model_latest.md)
Decayed ridge team ratings (margin and total fitted separately) plus each side's missing rotation, rest
and back-to-backs; walk-forward monthly, 2023-24 to 2025-26, against DraftKings' pre-game line.
Margin MAE 10.96 vs DraftKings 10.67; total 14.99 vs 14.42. The spread model carries information
(weight +0.30, se 0.08), and when it disagreed by 5+ points it covered 53.5% (488 picks; 7+: 57.9% on
152), 52% or better in every season; totals do not (47-49%, weight -0.05). Too few picks to bet on: live
game projections are recorded (nbaprops.game_projections) so the spread signal can be judged forward
against closing lines.

## Live uses (since 2026-10-24)
- `python -m nba_props.live` (task `\SuperNovaBets\NBA-Props-Live`, 6:45 AM, 11:00 AM, 3:30 PM MT) refreshes
  game logs, lineups and injuries, refits and saves the day's models, and appends projections (props to
  `nbaprops.projections`, games to `nbaprops.game_projections`) with their run time. `--news` (task
  `NBA-News-Watch`, every 10 minutes 8 AM-11 PM MT) captures lineups/injuries and re-projects with the day's
  saved fits when a starter, inactive or injury status changes for a team still to play.
- `python -m nba_props.post` posts the slate to #nba once, ~3 hours before the first tip: projected score,
  model spread/total beside DraftKings and Pinnacle (one 3-credit call), who is out, top players' p/r/a/3s.
- `sharp_pipeline.outcomes` grades the sharp scanner's decisions with results (NBA from the game logs,
  NCAAF from `ncaaf.games`; a prop with no game log for a played game is void; first halves ungraded) and
  records the shadow filter: the model's probability for the decision's side from the latest projection
  made before it, and whether it is above Pinnacle's no-vig. Both appear in `reports/sharp_multi_latest.md`
  and the digest; EV at the close stays the primary test. Projections are not reconstructed for days
  before the first run, so the shadow split starts with Oct 24 decisions.
- Judge the filter on 100+ NBA prop decisions: it earns a role only if "agrees" beats "disagrees" on EV
  at the close.
- Re-run the test each season; adopt nothing unless it beats the market's log loss in two straight seasons.
