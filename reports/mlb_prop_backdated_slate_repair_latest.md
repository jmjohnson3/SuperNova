# MLB Backdated Slate Repair Report

Generated UTC: 2026-07-13T03:53:09Z
Source audit UTC: 2026-07-13T03:52:57Z
Date range: 2026-06-01 to 2026-07-11
Canonical phase: `day_pregame`

This report says whether each near-miss backdated slate can be repaired without inventing lock, close, model-version, or result evidence.

## Summary

| Status | Dates |
|---|---:|
| not_countable | 11 |
| prop_promotion_countable | 4 |

- Strict eligible dates now: -
- Prop-promotion countable dates: 2026-07-02, 2026-07-03, 2026-07-04, 2026-07-05
- Prop close countable dates: 2026-06-12, 2026-06-13, 2026-06-14, 2026-06-15, 2026-06-16, 2026-06-18, 2026-06-19, 2026-06-20, 2026-06-21, 2026-06-22, 2026-06-26, 2026-06-27, 2026-06-29, 2026-06-30, 2026-07-02, 2026-07-03, 2026-07-04, 2026-07-05
- Forecast countable dates: 2026-07-06, 2026-07-07, 2026-07-09

## Near-Miss Repairability

| Date | Status | Prop Promotion | Prop Close | Forecast | Prop Rows | Forecast Rows | Valid CLV | Details | Action |
|---|---|---|---|---|---:|---:|---:|---|---|
| 2026-07-04 | prop_promotion_countable | True | True | False | 6716 | 1668 | 94.1% | strict_only_non_prop_blockers: prop_promotion_countable, core_prop_forecasts_and_prop_close_evidence_pass; remaining blockers are not prop-promotion blockers; pending_forecasts_on_settled_games: repairable, pending 2 (core 0, opportunity 2) | Count this slate for prop-promotion evidence; keep it out of strict all-layer evidence until non-prop blockers are cleared. |
| 2026-07-09 | not_countable | False | False | True | 5964 | 1443 | 89.6% | valid_clv_coverage: not_repairable, need 11 rows, recoverable 0, true-unavailable 268 | Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue. |
| 2026-07-05 | prop_promotion_countable | True | True | False | 5320 | 1673 | 92.4% | strict_only_non_prop_blockers: prop_promotion_countable, core_prop_forecasts_and_prop_close_evidence_pass; remaining blockers are not prop-promotion blockers; pending_forecasts_on_settled_games: repairable, pending 1 (core 0, opportunity 1) | Count this slate for prop-promotion evidence; keep it out of strict all-layer evidence until non-prop blockers are cleared. |
| 2026-07-02 | prop_promotion_countable | True | True | False | 4261 | 821 | 95.4% | strict_only_non_prop_blockers: prop_promotion_countable, core_prop_forecasts_and_prop_close_evidence_pass; remaining blockers are not prop-promotion blockers; model_version_coverage: not_repairable_without_external_artifact_proof, forecast_rows_use_current_or_unknown_model_version | Count this slate for prop-promotion evidence; keep it out of strict all-layer evidence until non-prop blockers are cleared. |
| 2026-07-06 | not_countable | False | False | True | 3828 | 866 | 87.7% | valid_clv_coverage: not_repairable, need 35 rows, recoverable 0, true-unavailable 188 | Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue. |
| 2026-07-10 | not_countable | False | False | False | 8100 | 2235 | 87.6% | pending_forecasts_on_settled_games: repairable, pending 4 (core 0, opportunity 4); valid_clv_coverage: not_repairable, need 72 rows, recoverable 0, true-unavailable 366 | Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue. |
| 2026-07-07 | not_countable | False | False | True | 7577 | 1667 | 87.5% | valid_clv_coverage: not_repairable, need 77 rows, recoverable 0, true-unavailable 386; side_locks_not_before_start: partially_repairable, late_or_missing_start_side_locks_must_be_excluded_then_coverage_recomputed | Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue. |
| 2026-07-08 | not_countable | False | False | False | 7165 | 1631 | 86.7% | pending_forecasts_on_settled_games: repairable, pending 1 (core 0, opportunity 1); valid_clv_coverage: not_repairable, need 97 rows, recoverable 0, true-unavailable 392 | Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue. |
| 2026-07-11 | not_countable | False | False | False | 6643 | 2372 | 89.2% | pending_forecasts_on_settled_games: repairable, pending 2 (core 0, opportunity 2); valid_clv_coverage: not_repairable, need 24 rows, recoverable 0, true-unavailable 311 | Do not count strict promotion; valid CLV miss is mostly true line/player unavailability, not a relabeling issue. |
| 2026-07-03 | prop_promotion_countable | True | True | False | 6263 | 1172 | 94.1% | strict_only_non_prop_blockers: prop_promotion_countable, core_prop_forecasts_and_prop_close_evidence_pass; remaining blockers are not prop-promotion blockers; model_version_coverage: not_repairable_without_external_artifact_proof, forecast_rows_use_current_or_unknown_model_version; pending_forecasts_on_settled_games: repairable, pending 2 (core 0, opportunity 2) | Count this slate for prop-promotion evidence; keep it out of strict all-layer evidence until non-prop blockers are cleared. |
| 2026-06-12 | not_countable | False | True | False | 8018 | 0 | 95.7% | forecast_ledger_missing: not_repairable, cannot_create_prospective_forecasts_after_results_without_hindsight_risk; model_version_coverage: not_repairable, no_immutable_forecast_ledger_rows_for_this_date | Use this date for CLV/bookability research only; do not count it as forecast-proof evidence. |
| 2026-06-16 | not_countable | False | True | False | 7632 | 0 | 92.3% | forecast_ledger_missing: not_repairable, cannot_create_prospective_forecasts_after_results_without_hindsight_risk; model_version_coverage: not_repairable, no_immutable_forecast_ledger_rows_for_this_date | Use this date for CLV/bookability research only; do not count it as forecast-proof evidence. |
| 2026-06-26 | not_countable | False | True | False | 7565 | 0 | 90.8% | forecast_ledger_missing: not_repairable, cannot_create_prospective_forecasts_after_results_without_hindsight_risk; model_version_coverage: not_repairable, no_immutable_forecast_ledger_rows_for_this_date | Use this date for CLV/bookability research only; do not count it as forecast-proof evidence. |
| 2026-06-19 | not_countable | False | True | False | 7343 | 0 | 94.6% | forecast_ledger_missing: not_repairable, cannot_create_prospective_forecasts_after_results_without_hindsight_risk; model_version_coverage: not_repairable, no_immutable_forecast_ledger_rows_for_this_date | Use this date for CLV/bookability research only; do not count it as forecast-proof evidence. |
| 2026-06-30 | not_countable | False | True | False | 7228 | 0 | 95.6% | forecast_ledger_missing: not_repairable, cannot_create_prospective_forecasts_after_results_without_hindsight_risk; model_version_coverage: not_repairable, no_immutable_forecast_ledger_rows_for_this_date | Use this date for CLV/bookability research only; do not count it as forecast-proof evidence. |
