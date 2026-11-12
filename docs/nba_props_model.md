# NBA player-prop model (nba_api)

Status (2026-10-28): **research only; no NBA wagering authorized.** The frozen
[paper price-check experiment](nba_price_checks.md) compares each sportsbook with Pinnacle
and measures whether its offers are available during manual checks. Projections stay a shadow comparison.

## Operational repairs (2026-10-28)
Player-log ingestion collapses identical keys and merges complementary missing fields only when
all supplied values agree. Conflicting records are saved in `nbaprops.ingest_conflicts`; the whole
game is excluded from model inputs and grading until a later unambiguous source refresh resolves
the affected keys. Existing raw player rows remain available for investigation.

News refreshes report `no_fits_yet` until daily fits exist and retry unchanged news after fits
become available. Missing player results remain pending; this feed alone cannot verify a void.
Shared-scanner observations retain their snapshot identity, so an early observation cannot block
a later qualifying decision. Repeated observations still count as one decision per player-game-market.
Market timestamps govern freshness. Only observed exact-line closes count toward close coverage
and stake review; estimated line conversions are reported separately.

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

## Challenger: game line and minutes model (2026-11-11; `python -m nba_props.challenger`)
`nba_props.challenger` reruns the walk-forward test with extra inputs and scores every version on the same
22,148 / 21,067 / 14,794 DraftKings points / rebounds / assists lines (reports/nba_props_challenger_latest.md).
It imports the pinned model and changes nothing in it.

| Version | Points miss | Rebounds miss | Assists miss | Points log loss |
|---|---:|---:|---:|---:|
| DraftKings line | 5.069 | 2.060 | 1.777 | 0.6917 |
| Current model (v2) | 5.142 | 2.060 | 1.777 | 0.7070 |
| + DraftKings spread and total | 5.141 | 2.060 | 1.777 | 0.7068 |
| + game line and a minutes model | 5.135 | 2.058 | 1.776 | 0.7063 |
| If the minutes played were known | 4.397 | 1.863 | 1.665 | 0.6130 |

- **The game line adds nothing** (points -0.001, interval -0.003 to +0.001). The model's pace and opponent rates,
  and the prop line it is judged against, already carry it.
- **The minutes model forecasts minutes about 10% better** than the player's last-3 average (4.53 against 5.01
  minutes; 4.82 against 5.50 when the team is missing two or more rotation players), but the projections move by
  0.1%: points -0.007 (-0.011 to -0.002), rebounds -0.002, assists 0. It is built from the same inputs the stat
  model already had (recent minutes, starter, who is out), so it tells that model little it could not work out.
- **Fitting the spread on out-of-sample forecasts** instead of the training fit does not change the log loss. The
  overconfidence is not the count distribution; it is the model disagreeing with a line that knows more.
- **Minutes are the whole gap.** With the minutes actually played, the same model beats the line by 0.67 points
  and its picks hit 67%. That is not attainable (it includes foul trouble, blowouts and injuries during the game),
  but it says where accuracy is: knowing tonight's minutes better, not more box-score features. As a rough guide
  from these numbers, matching the line on points needs a minutes miss near 4.1 instead of 4.5.

Verdict: neither input is adopted; picks still lose 2.4% to 4.9% at DraftKings' prices. The next thing worth
testing is new information about minutes (who takes the minutes when a player sits, from rotation stints;
returns from injury and minutes limits; the timing of injury reports), measured in this same harness. The test
also flatters every version a little: it uses the final lineup file, which can be later than the line snapshot.

## Challenger: new minutes information (2026-11-12; `nba_props.minutes_info`, same harness)
Three groups of inputs from data already held (box scores and the pre-game lineup files), all known before the game:
- **absence**: which rotation teammates are newly out or back against the player's recent games, weighted by how
  alike they are, and how his minutes and per-minute numbers moved in earlier games this season with and without them;
- **returning**: his own lay-off (days off, team games missed, games since he came back, his level before);
- **role**: his minutes in earlier starts and earlier bench games, and whether tonight's lineup is a change.

The minutes forecast alone (average miss in minutes, 86,198 test player-games):

| Player-games | Last-3 average | Current inputs | + absence | + returning | + role | + all three |
|---|---:|---:|---:|---:|---:|---:|
| All | 5.01 | 4.54 | 4.49 | 4.53 | 4.52 | 4.46 |
| 30+ rotation minutes back | 5.28 | 4.69 | 4.54 | 4.66 | 4.64 | 4.51 |
| First 5 games back from a lay-off | 5.41 | 4.72 | 4.63 | 4.63 | 4.68 | 4.54 |
| Role changed tonight | 7.45 | 5.12 | 5.07 | 5.11 | 5.03 | 4.96 |

All three: -0.077 minutes (-0.087 to -0.067), about 1.7%. Absence carries most of it; the groups add up.

Given to the statistic models and a minutes model (`minutes_info` variant), on the same DraftKings lines:

| Version | Points miss | Rebounds miss | Assists miss | Points log loss |
|---|---:|---:|---:|---:|
| DraftKings line | 5.069 | 2.060 | 1.777 | 0.6917 |
| Current model (v2) | 5.142 | 2.060 | 1.777 | 0.7070 |
| + minutes information and model | 5.118 | 2.054 | 1.774 | 0.7052 |

- **The first input that measurably helps**: points -0.023 (-0.032 to -0.015), rebounds -0.006 (-0.010 to -0.002),
  assists -0.002 (-0.006 to +0.001). It closes a third of the points gap to the line; rebounds and assists are now
  a hair better than the line on average miss.
- **It is not an edge**: the probabilities are still worse than DraftKings' no-vig price on all three (assists
  log loss got worse, +0.0019), and picks at a 4% disagreement lose 2.2% (points), 3.6% (rebounds), 5.4% (assists).
- The model's weight beside the line rose (points 0.27 to 0.32, rebounds 0.37 to 0.43): it carries a little
  more information the line does not, but the line still deserves most of the weight.

Verdict: a real but small gain in accuracy, kept as a challenger. The live model (pinned) is unchanged; adopting
these inputs means a new live model version, and that is not worth doing for betting on this evidence. The gap
that remains is information the box score and lineup file do not hold: in-game substitution patterns (rotation
stints from stats.nba.com, reachable at about 2 seconds a game), minutes limits announced by teams, and news
between the lineup file and tip.

## Challenger: rotation stints (2026-11-12; `nba_props.stints`, `nba_props.stint_info`, same harness)
`python -m nba_props.stints` downloads stats.nba.com's rotation stints (who was on the floor, from when to when,
with the stint's plus-minus), one request per game, into `nbaprops.rotation_stints` and `nbaprops.rotation_games`.
All 6,742 finished games from 2021-22 to 2026-11-11 are stored; a rerun fetches only games not stored yet. It is
not scheduled. 22 games whose stints do not add up to the box-score minutes are left out automatically.

Two groups of inputs, from earlier games only:
- **stint patterns**: the player's minutes with garbage time taken out (the fourth quarter of a game 15+ points
  apart), his share of time in the fourth quarter and in close finishes, when he usually first comes on, how his
  minutes are cut up, and whether his last game ended early for him. The score during a game is rebuilt from each
  stint's plus-minus, so garbage time and close finishes are approximate;
- **substitutes**: the newly-out and returning teammates weighted by how much the player's floor time went with
  each one's bench time in earlier shared games.

The minutes forecast alone, on top of the minutes information above (average miss in minutes, 86,419 test
player-games, all with stint inputs):

| Player-games | Minutes information | + stint patterns | + substitutes | + both |
|---|---:|---:|---:|---:|
| All | 4.46 | 4.42 | 4.46 | 4.42 |
| 30+ rotation minutes newly out | 4.81 | 4.77 | 4.81 | 4.77 |
| First 5 games back from a lay-off | 4.54 | 4.49 | 4.55 | 4.48 |
| Role changed tonight | 4.96 | 4.96 | 4.95 | 4.93 |

Both: -0.045 minutes (-0.052 to -0.038), about 1%. The stint patterns carry all of it. The substitute weights add
nothing beyond the absence inputs: with/without minutes from box scores already say who fills whose minutes.

On the same DraftKings lines (`stint_info` variant: minutes information, stints and a minutes model):

| Version | Points miss | Rebounds miss | Assists miss | Points log loss |
|---|---:|---:|---:|---:|
| DraftKings line | 5.069 | 2.060 | 1.777 | 0.6917 |
| Current model (v2) | 5.142 | 2.060 | 1.777 | 0.7070 |
| + minutes information and model | 5.118 | 2.054 | 1.774 | 0.7052 |
| + minutes information, stints and model | 5.114 | 2.052 | 1.775 | 0.7047 |

- **The stints' own step on the lines is too small to tell from zero**: points -0.004 (-0.011 to +0.002), rebounds
  -0.002 (-0.004 to +0.000), assists +0.001 (-0.002 to +0.003), and assists log loss a little worse (+0.0010,
  +0.0003 to +0.0018).
- Against the current model the full set is points -0.028 (-0.038 to -0.018), rebounds -0.008 (-0.012 to -0.005),
  assists -0.002 (-0.005 to +0.002). The model's weight beside the line rose again (points 0.32 to 0.35, rebounds
  0.43 to 0.45).
- **Still not an edge**: the probabilities are worse than DraftKings' no-vig price on all three, and picks at a
  4% disagreement lose 1.6% (points), 3.2% (rebounds), 5.9% (assists).

Verdict: stints improve the minutes forecast a little and the projections by an amount that cannot be told from
zero. Not adopted; the pinned live model is unchanged and the download does not need to be kept current. Taken
together, everything tried from history (minutes model, absences, lay-offs, roles, stints) moved the minutes miss
from 4.54 to 4.42 and the points miss from 5.142 to 5.114, against 4.397 with the minutes actually played. History
is close to used up. What is left is what happens on the day: team-announced minutes limits, news between the
lineup file and tip, and the game itself (fouls, blowouts, injuries).

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
  NCAAF from `ncaaf.games`; missing player results remain pending; first halves ungraded) and
  records the shadow filter: the model's probability for the decision's side from the latest projection
  made before it, and whether it is above Pinnacle's no-vig. Both appear in `reports/sharp_multi_latest.md`
  and the digest; EV at the close stays the primary test. Projections are not reconstructed for days
  before the first run, so the shadow split starts with Oct 24 decisions.
- The "agrees" subset does not select or reject paper checks. Any proposed model filter needs its own
  fixed prospective challenger; subgroup performance cannot change the registered experiment.
- Re-run the test each season; adopt nothing unless it beats the market's log loss in two straight seasons.
