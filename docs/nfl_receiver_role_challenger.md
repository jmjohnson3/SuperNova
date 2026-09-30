# Role-Aware Receiving Challenger

This repair is isolated from the frozen production release, Discord picks, and bet ledgers.
It does not automatically replace production or relax micro/bankroll rules.

## Model

- Estimate team passing volume, then player targets from prior target share, recent role,
  snaps, current team experience, and timestamp-validated depth/injury/teammate evidence.
- Estimate receiving efficiency separately from up to 40 appearances, with target-weighted
  player priors shrunk toward position rates. A zero-target appearance supplies no YPT label.
- Treat historically low snaps relative to earlier normal snaps as a partial-appearance
  proxy. This is not a diagnosis and does not erase that game's observed targets or catches.
- Select latest-game weight (1 or 0.25), recent-role blend, and prior strength on inner weeks.
- Fit conditional error distributions and probability calibration on separate later blocks.
  Validate calibration on a further block; retain it only if it improves Brier there.
- Keep missing context as unknown with missingness indicators. Historical news coverage is
  sparse, so this experiment cannot yet prove the contribution of teammate/injury inputs.

The model caps individual targets at projected team attempts. It is not a joint allocation
of all current-roster target shares; that remains a separate team-allocation challenger.

## Validation

```powershell
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.train_receiver_role --through 2026-09-19
.\.venv\Scripts\python.exe -m nfl_pipeline.modeling.receiver_role_sensitivity --date 2026-09-20
```

The trainer uses expanding four-week folds across 2024, 2025, and available 2026 games.
Whole games and player-game rows stay together. Features only use earlier dates, including
for games played on the same date. Parameter selection never sees outer-fold outcomes.
Historical corrected stats are not certified original lock-time data, and proxy lines are
not executable offers. Reference models are chronological refits, not today's model applied
to its own training history. Previously examined seasons remain research evidence.

Separate gates assess mean RMSE/bias, median MAE, and line Brier/calibration/80% coverage.
Week-clustered intervals quantify uncertainty. Passing all historical gates still does not
grant production approval: actual pregame forecasts on real offered lines must confirm it.

## Latest-Game Sensitivity

The diagnostic rebuilds all challenger workload/efficiency/uncertainty inputs with the latest
performance fully included, weighted at 0.25, or omitted. It also adjusts that game in team
passing history. Current timestamped context and the original offer/price are held fixed.
Each curve goes through the complete captured live probability adjustments at the same line.
Scoring-version mismatches are excluded, not silently approximated.

Reports include targets, YPT, mean/median, interval, final side probability, side changes,
probability swing, and original production comparison. Each diagnostic saves its input history.
Only scores written before kickoff are prospective; started games are labeled retrospective.
Sensitivity alone is neither a betting filter nor a reason to reverse a pick.

Outputs:

- `reports/nfl_receiver_role_latest.md` and `.json`
- `reports/nfl_receiver_role_sensitivity_DATE.md` and `.json`
- Immutable run directories under `src/nfl_pipeline/modeling/models/receiver_role/`

The sensitivity command writes diagnostics only. It never posts to Discord, places wagers,
changes forecasts, or writes ledger rows.
