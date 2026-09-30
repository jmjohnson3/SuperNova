# Manual MLB Diagnostic Tools

These scripts are intentionally manual. They are not part of the required
scheduler path and should not be treated as daily pass/fail signals unless they
are promoted into `run_daily.py` or `run_daily_and_notify.py` with explicit
ownership and tests.

## `mlb_pipeline.modeling.prop_gate_diagnostics`

Owner: prop model diagnostics.

Purpose: inspect positive-EV prop rows, warning tags, threshold behavior, and
recalibration candidates.

Run when: Discord/paper output looks suspicious, a bucket is over-producing
fake EV, or thresholds/recalibrators changed.

Do not use as: bankroll evidence. It explains gate behavior; it does not prove
locked profitability.

## `mlb_pipeline.modeling.prop_failure_diagnostics`

Owner: prop model diagnostics.

Purpose: slice graded replay history by market, side, line bucket, price
bucket, model family, player, team, and probability bucket.

Run when: a prop market is weak and we need to identify whether the issue is
stat/side, line/price, player/team concentration, or calibration.

Do not use as: a promotion report. Promotion remains exact-bucket, CLV-aware,
and ladder-gated.

## `mlb_pipeline.modeling.real_money_audit_report`

Owner: bankroll gate diagnostics.

Purpose: replay current gates against saved mutable prediction rows.

Run when: checking whether current filters are too strict or too loose against
historical saved predictions.

Do not use as: walk-forward proof or bankroll evidence. Saved prediction rows
can be overwritten by reruns. Use the locked bankroll ledger reports for real
money evidence.

## Promotion Rule

If any manual diagnostic becomes required for real-money readiness, add it to
the scheduler, make failures explicit, and add a regression test describing why
that script is now required.
