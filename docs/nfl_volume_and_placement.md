# Volume model and projection placement (October 21, 2026)

Both follow from `reports/nfl_projection_error_latest.md`:

- The market line beats the model's own number on every stat.
- Volume (targets, carries, attempts) is the larger source of error for every stat except receiving
  yards, where volume and efficiency matter about equally.
- The pipeline's own volume features predict volume worse than a plain average.
- Rushing and passing projections sit in the wrong place every week.

## Volume challenger: accepted on volume, not wired live

`modeling/volume_model.py` is a LightGBM count model (Poisson objective) of targets (WR/TE/RB) and
carries (RB/QB).

- **Training data.** It trains on the as-of feature table, 2019 onward, using every pregame
  feature. Same-game outcome columns are excluded, and a guard drops any feature that tracks the
  outcome too closely to be pregame; it dropped none.
- **Evaluation.** Walk-forward in 4-week blocks over 2023-2026, each block trained only on earlier
  games, with fixed hyperparameters. The run is `volume-20261022T025415Z`, with artifacts under
  `models/volume_model/`.

| Volume | Model MAE / RMSE | Best simple baseline MAE / RMSE | Pipeline `*_v3` MAE | Weekly MAE gain (95%) |
|---|---|---|---:|---|
| Targets | 1.643 / 2.230 | 1.719 / 2.306 (10-game avg) | 1.752 | -0.067 (-0.083 to -0.051) |
| Carries | 2.929 / 4.034 | 3.132 / 4.344 (5-game avg) | 3.431 | -0.212 (-0.264 to -0.160) |

Both beat the baseline in every season (targets -2.6% to -5.6%, carries -5.6% to -8.5%) and at
every position, and pass all four volume checks. That is a much larger gain than the rejected
September target challenger (RMSE 2.282 to 2.252, `docs/nfl_target_volume_repair.md`).

**Does better volume make better yards?** Measured on the same settled 2026 forecasts, using only
walk-forward volume:

| Stat | Production MAE | Rescaled by model volume | Model volume x shrunk efficiency |
|---|---:|---:|---:|
| Rushing yards | 19.02 | **18.03** | 18.28 |
| Receiving yards | 20.91 | 21.30 | 21.12 |
| Receptions | 1.63 | 1.67 | 1.66 |

**Decision.**

- **Carries: ready for a rushing-yards challenger.** It cut live rushing-yard error about 5% both
  ways it was used.
- **Targets: a better volume number with nothing yet to plug into.** It does not improve receiving
  yards. Receiving error is dominated by efficiency, and the production yardage model already prices
  volume about as well as the averages do. The useful change there is narrower: replace
  `receiver_projected_targets_v3` (worse than a plain average) wherever it is an input.

Neither is wired into live scoring. Promotion means a challenger capture judged prospectively,
like every other model change here.

## Placement: installed

`modeling/fit_placement_calibration.py` fits where the raw projection should sit, per stat and
position.

- **Families compared.** Identity, a proportional scale, and a linear pull toward the middle, each
  fitted by least absolute deviation (the loss a median minimizes).
- **Validation.** Expanding by week: every week is scored by a correction fitted only on earlier
  weeks.
- **Rule.** A group is corrected only when held-out MAE improves at least 1% and the share of
  outcomes above the projection moves toward 50%.

| Group | Held-out MAE | Share above projection | Installed |
|---|---|---|---|
| Rushing, QB | 13.53 to 13.13 | 34% to 56% | scale x0.751 |
| Rushing, RB | worse with any correction | (46%) | none |
| Passing, QB | 57.29 to 56.21 | 62% to 52% | +9.6 yards |

Two results differ from the plan:

- **Rushing.** The "rushing sits high" finding was mostly QBs. Every RB correction overshot on
  later weeks, so RBs are left alone.
- **Passing.** The data did not support pulling projections toward the middle. The best linear
  fit wanted no compression at all (the slope hit its 1.0 cap), only a uniform shift up. The large
  mean miss at the low end comes from a few blowouts, which a median fit correctly ignores.

The samples are small (122 QB rushing and 150 passing games), but the direction held in every week.
Refit after each slate with `python -m nfl_pipeline.modeling.fit_placement_calibration`
(report only), and install with `--write`.

**Where it applies.**

- `predict_player_props` applies the artifact (`models/player_props/placement_calibration.json`;
  missing means identity) to the raw model projection, after the QB rushing cap and before market
  anchoring. The trace records `unplaced_model_projection` and `placement_calibration_version`.
- It changes the model numbers on cards and unpriced rows. Priced rushing and passing projections
  stay line-anchored (`projection_trust` 0), so no price, probability or bet changes.
- Because `predict_player_props.py` is part of the scoring fingerprint, the fingerprint moves;
  earlier captures still replay from their archived sources.

## October 22: serving parity, the live rushing challenger, and the targets swap

**Serving parity.** Live forecasts are scored from a snapshot built by a different path from the
training table, and the two disagree on team-relative features. The table is rebuilt after games,
so its team groups are the players who actually played; live sees the pregame roster. Measured on
the same 2026 player-games:

| Feature | Live mean | Training mean |
|---|---:|---:|
| Teammate carry total | 34.0 | 28.4 |
| Carry share | 0.085 | 0.103 |
| Carry role rank | 7.1 | 5.4 |

Player-level averages, depth and game lines match exactly. Of the 363 candidate features, 103 shared
ones differ by more than 5% of rows and 33 are not in the live snapshot at all. *(Corrected October 23: the production yardage models use none of the skewed features; the opportunity and
workload models behind the workload adjustment do. See `docs/nfl_feature_parity.md`.)*

The volume model now trains only on the 191 features whose archived live values match the table
(relative mean difference of 2% or less). Run `volume-20261022T151821Z`, still accepted on volume:

| Volume | Model MAE | Best baseline | Weekly 95% interval |
|---|---:|---:|---|
| Targets | 1.651 | 1.719 | -0.075 to -0.043 |
| Carries | 3.003 | 3.132 | -0.185 to -0.081 |

**Rushing challenger, registered and capturing.** `modeling/rushing_volume_challenger` (registration
`models/rushing_volume_challenger/registration.json`, pinned to that run's checksum) scores every
current pregame RB and QB rushing forecast as:

    candidate = production model projection x model carries / 5-game average carries

- **When it runs.** Captures run in the daily and pregame runs after player predictions
  ("NFL Rushing Volume Challenger"). The close task grades them ("NFL Rushing Volume Challenger
  Review", `reports/nfl_rushing_volume_challenger_latest.md`).
- **What counts.** Grading keeps the last capture before kickoff per player-game, and judges carries
  first, then yards.
- **When it is reviewable.** After 3+ weeks with a 95% lower bound on the weekly MAE gain above
  zero. Deployment remains manual.

The live path was dry-run on weeks 5-6 before registering, with archived live features and a model
trained only on earlier games (172 settled player-games):

- carries MAE 3.30 to 3.15;
- rushing-yard MAE 18.25 to 18.09;
- RBs improved (19.40 to 19.03), QBs worsened (15.02 to 15.43);
- the weekly interval still spanned zero, so it is not yet proven.

**Targets swap.** `receiver_projected_targets_v3` was used as an input in two live places.

- **Receptions projection: swapped.** The production receptions forecast blends recent receptions
  with targets x catch rate. With the targets model in place of v3, on 17,070 walk-forward 2023-26
  player-games:
  - MAE improves 1.320 to 1.308, better in every season;
  - bias improves -0.22 to -0.02;
  - the weekly interval is -0.017 to +0.001, so the gain is just short of 95% significance;
  - outcomes above the projection fall from 48% to 43%.

  It is installed through `models/player_props/targets_model.json`, pinned by checksum; a missing
  or mismatched artifact falls back to v3. Each receiver row carries `_pred_targets_model`, so
  replays see the same input. On today's live snapshot, all 24 receivers were scored.
- **Receiving spike heuristic: no swap needed.** *(Corrected October 22.)* This was first held back
  for a calibrator refit, but the heuristic is not on the live path. Its only consumer is the
  `receiver_spike_mixture_v1` distribution, which was rejected at training
  (`receiver_spike_mixture_accepted` false; Brier -0.0018 against the current mixture), and every
  live receiving forecast prices with `empirical_oof_residual`. Refitting its calibrators would
  change no forecast. If a future release retrains that challenger, it should use the model targets.

The remaining v3 dependence is indirect. `receiver_spike_yards_anchor_v3` and its v2 siblings are
built from v3 inside `features.py`, feed the workload adjustment, and differ about 3% between the
live snapshot and the table. They belong to the serving-parity rebuild of the feature contract.

- **Feature table and training feature lists: unchanged.** These feed the pinned production model
  and future retrains. Changing them would alter production inputs without a retrain. The right
  change is a new release that trains on model targets (out of fold) and on parity-safe features.

## Challenger: teammates ruled out (2026-11-12; `python -m nfl_pipeline.modeling.volume_absence`)
The volume model leaves out the table's teammate-injury scores because they do not match live. This adds a small
set built from sources that are the same before and after a game (earlier box scores and the week's injury
report): recent targets/carries of teammates listed Out or Doubtful (all, and the newly vacated part), volume
returning, the player's own volume in earlier games without those teammates, vacated targets at his position,
quarterback out. Same walk-forward, parameters and parity-safe features as the volume model, 2023 to 2026, with
and without the inputs (`reports/nfl_volume_absence_latest.md`).

| Volume | Player-games | Volume model MAE | With absence inputs | Weekly difference (95%) |
|---|---:|---:|---:|---:|
| Targets, all | 17,900 | 1.656 | 1.644 | -0.010 (-0.014 to -0.005) |
| Targets, 8+ returning | 190 | 1.922 | 1.847 | -0.086 (-0.132 to -0.040) |
| Carries, all | 7,588 | 3.072 | 3.029 | -0.051 (-0.065 to -0.037) |
| Carries, 8+ newly vacated | 399 | 3.598 | 3.235 | -0.443 (-0.640 to -0.247) |
| Carries, 8+ returning | 130 | 3.377 | 3.129 | -0.327 (-0.629 to -0.025) |

- A real gain in the volume forecast, in every season: small for targets (0.7%), larger for carries (1.4%, and
  10% when a back with 8+ carries is newly out). Targets move less because a receiver's absence spreads thinly.
- It does not reach yards. Moving the live 2026 projection to the model's volume is still worse than the
  production number, with or without the inputs: receiving 20.67 production, 21.27 volume model, 21.09 with
  absence; rushing 20.52, 21.27, 21.11; receptions 1.54, 1.63, 1.62. The rushing rescale that looked better on
  weeks 2 to 6 (18.16 against 19.02) is worse on the fuller season.
- Verdict: kept as a challenger, not adopted. The inputs are worth including the next time the volume model is
  refit, but better volume alone has not improved a yardage projection here, and the projection is still behind
  the line. The historical injury report is the final one of the week; live would read it from the observation
  log, which has not been checked for parity.

## Challenger: efficiency (2026-11-12; `python -m nfl_pipeline.modeling.efficiency_model`)
The other half of the projection miss: yards per target, catches per target and yards per carry. Each estimate
is a per-opportunity rate scored as rate x the targets or carries the player actually had, so only efficiency
is judged. Baseline = the player's last-10 rate shrunk toward his position's; model = LightGBM on the
parity-safe features plus that baseline, weighted by opportunities; then the opponent's allowed rates by
position and the weather on top. Same walk-forward as the volume model, 2023 to 2026
(`reports/nfl_efficiency_model_latest.md`).

| Stat, volume known | Player-games | Baseline | Model | + opponent | + opponent and weather | Best, weekly difference (95%) |
|---|---:|---:|---:|---:|---:|---:|
| Receiving yards | 15,501 | 13.852 | 13.767 | 13.747 | 13.746 | -0.123 (-0.233 to -0.013) |
| Receptions | 15,501 | 0.733 | 0.729 | 0.728 | 0.727 | -0.005 (-0.010 to +0.000) |
| Rushing yards | 6,768 | 11.969 | 11.750 | 11.730 | 11.753 | -0.351 (-0.614 to -0.088) |

- Efficiency is close to unforecastable. The best model beats a shrunk last-10 rate by 0.8% on receiving yards,
  0.8% on receptions and 2% on rushing yards. The opponent's allowed rates by position and the weather add
  almost nothing beyond the model, and not in every season.
- On the live 2026 forecasts nothing beats the production number, and the line beats all of them: receiving
  yards with a line 21.27 line, 22.26 production, 22.43 to 22.53 for the rates; rushing 18.89, 20.54, 21.17 to
  21.59; receptions 1.58, 1.60, 1.60 to 1.61.
- Verdict: not adopted, and no reason to bring the opponent splits or a weather forecast into the live
  snapshot for this. With volume (docs above) and efficiency both tested, the yardage projections are near what
  box-score history can give; the line's lead (about 1 yard receiving, 1.7 rushing) is not in these inputs.
