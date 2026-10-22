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
