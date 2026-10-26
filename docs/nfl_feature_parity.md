# NFL live-vs-training feature parity (October 23, 2026)

Measured with `modeling/feature_parity.py`, which rebuilds past live snapshots using the production
loaders at each slate's cutoff and joins them to `features.nfl_player_game_training_features` on the
same player-games (`reports/nfl_feature_parity_latest.md`). Weeks 5-6, 627 player-games: 214 features
match, 116 differ and 33 are missing live.

## Where it matters

- **Yardage models: not affected.** The production release's yardage models (receiving 59 features,
  rushing 36, passing 21) use no mismatched feature; the largest difference among their inputs is 0.9%.
  An earlier note said the production model was trained on one definition and scored on the other.
  That is wrong for these models.
- **Opportunity / workload models: affected.** These are the 26 models in
  `models/player_props/nfl_player_opportunity_models.joblib`: targets, carries, QB attempts, snap
  share, limited-usage risk, spike and TD-role probabilities. Each has 10-70 mismatched inputs. They
  drive the workload adjustment that `predict_player_props` applies to every projection
  (`apply_workload_adjustment`), plus the workload and spike probabilities.

## Causes, by size

1. **Team role groups.** The affected features are the teammate totals
   (`team_player_*_avg5_sum`), shares (`*_share_avg_5`) and ranks (`*_role_rank`).
   - Training counts the players who actually played, plus teammates listed Out or Doubtful.
   - Live (`_add_snapshot_role_features`) counts every active-roster QB/RB/WR/TE. That averages
     17.7 per team, including 4.6 who never touch the ball and 2.4 cold-start members given a
     position-average prior.

   | Teammate total | Live | Training | Live, from players who played | ...who did not play | ...cold-start priors |
   |---|---:|---:|---:|---:|---:|
   | Carries | 35.5 | 28.4 | 24.8 | 6.2 | 4.6 |
   | Targets | 40.4 | 35.5 | 30.9 | 4.9 | 4.6 |

   Shares therefore run 15-25% lower live, and ranks 20-45% worse.
2. **Injury downgrade.** `injury_downgrade_score` is present on the same 21% of rows on both sides,
   but averages 0.16 live against 0.06 in training. The two paths read different injury report rows
   for the same week. Teammate injury counts differ the same way (2.25 vs 1.46).
3. **Depth movement.** `depth_rank_delta`, `depth_rank_worse` and `depth_rank_better` differ by
   36-52%. Live takes the previous rank from the roster context (`previous_depth_pos_rank`), while
   training shifts the player's previous game row.
4. **Missing live.** 33 features never reach the live snapshot: weather, the opponent position
   splits, and the full-role and robust averages. No production model uses them; the volume model
   now excludes them.

## Plan

1. **One pregame definition of the role group, used by both sides.** The group is the active roster
   as of the cutoff, excluding players ruled out, with no invented priors: a cold-start member
   counts zero. It lives in a new unpinned module called by `features.py` and
   `_add_snapshot_role_features`. Done when the harness shows the role features match.
2. **One injury-row rule and one previous-depth rule** for both sides, verified the same way.
3. **Retrain the opportunity models** on the corrected table as a challenger. Judge them on volume
   error first (`modeling/volume_model` style), then on the projection error they produce
   (`modeling/projection_error`). Promotion is manual.

Changing `features.py` or `predict_player_props.py` drifts `fanduel-receiving-oe-paper-v5`, so that
trial gets a successor registration with the change. The sharp-gap cash strategy pins neither file
and is unaffected.

## Step 1 done (October 26, 2026): one pregame role group

`nfl_pipeline.role_group` is the shared rule, called by `features._add_usage_role_features` and
`predict_player_props._add_snapshot_role_features`: a member's most recent game (any team) is no older
than his team's 4th most recent game, he is not ruled out (Out/Doubtful, or off the active roster live),
and cold-start players count zero. Non-members are scored against the group but not added to it.
Training rebuilt (6 minutes), harness rerun on weeks 5-7 (856 player-games):

| Feature | Relative difference before | After |
|---|---:|---:|
| team_player_carries_avg5_sum | 29% | 9% |
| team_player_targets_avg5_sum | 16% | 7% |
| team_player_pass_attempts_avg5_sum | 61% | 6% |
| carries_share_avg_5 / carries_role_rank | 24% / 43% | 9% / 8% |
| targets_share_avg_5 / targets_role_rank | 15% / 20% | 8% / 8% |
| pass_attempts_share_avg_5 / role_rank | 38% / 112% | 4% / 5% |

Not yet inside the 2% match bar. Live totals now run 5-7% below training; the likely remainder is
teammates moved to IR midweek, who are not on the injury report, so training still counts them as
expected absentees while live drops them (training has weekly rosters only from 2026). The receiving
paper trial moved to `fanduel-receiving-oe-paper-v6` with this change.

## Steps 2 and 3 done (October 26, 2026): one injury rule, one previous-depth rule

- **Injuries.** `raw.nfl_injuries` keeps one row per player-week, overwritten by every import, so a row
  re-imported after kickoff vanished under the as-of filter: training missed ~450 pregame Out/Questionable
  statuses in weeks 5-7 that live saw, and kept older Questionable rows where the pregame report said Out.
  Training now builds `pg_temp.nfl_injury_pregame`: wherever the context observation log covers a game
  (2026 from 17 September) the latest logged report two hours before kickoff, else the archive row. The
  ruled-out set for role groups reads the same table.
- **Depth.** Training's previous depth rank is now live's: the player's latest chart for the team at least
  six days older than the current one (week-keyed archive charts: the previous chart week), instead of
  his previous game row.

| Feature | After step 1 | After steps 2-3 |
|---|---:|---:|
| injury_downgrade_score | 119% | 9% |
| same_week_teammate_skill_injury_count | 50% | 7% |
| teammate_skill_injury_pressure_score | 40% | 6% |
| depth_rank_delta | 47% | 9% |
| depth_rank_better / worse | 55% / 34% | 15% / 11% |

Matching features 226 -> 247 of 363 (83 differ, 33 never reach live and no production model uses them).
The features build takes 8.6 minutes. Receiving paper trial: v7.

## Opportunity-model challenger (October 26, 2026)

Retrained into `models/opportunity_parity/opportunity-parity-20261026T225152Z/` (not production) and
compared by `modeling/opportunity_challenger` on 2026 weeks 4-7, out of sample for both, scored with the
live function (reports/nfl_opportunity_challenger_latest.md):

- better on 15 of 25: targets -5.1%, receptions -4.7%, carries -3.4%, QB attempts -3.9%, target share
  -7.4%, full-workload and limited-usage probabilities -14%;
- worse on 9, mostly spike / high-usage probabilities: RB carry spike +68%, high carries +49%, high pass
  attempts +47%, spike snap share +11%.

Not promoted. The trainer picks each model's form (direct or residual, blend weight) on its last four
weeks, the same weeks scored here, so the comparison flatters the challenger. Re-run on week 8 and later
(unseen by both) before deciding; a mixed artifact (challenger volume models, production probabilities)
is the likely candidate if the pattern holds.
