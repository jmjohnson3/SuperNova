# MLB Prop Player-Game Bankroll Model Proof

Generated UTC: 2026-07-30T14:38:29Z
Status: **ready**
Usage: exact-line bankroll models may train/promote only after the player-game stat forecast proves it beats baseline.

- Projection-micro allowed stats: pitcher_strikeouts, batter_hits, batter_total_bases, batter_home_runs
- Allowed stats: -

| Stat | Projection Rows | Dates | MAE | Baseline | Gain | Projection Proven | Projection Micro | True Pair | CLV Rows | CLV Beat | Avg CLV | Exact-Line Training Allowed | Blockers |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| pitcher_strikeouts | 514 | 25 | 1.845 | 1.937 | 0.092 | True | True | 11689 | 9520 | 41.1% | 0.019 | False | active_model_cohort_ungraded, clv_truth_not_confirming |
| batter_hits | 5139 | 25 | 0.724 | 0.726 | 0.002 | True | True | 84664 | 99733 | 38.5% | 0.022 | False | active_model_cohort_ungraded, clv_truth_not_confirming, live_vs_legacy_bias_worse, live_vs_legacy_mae_not_improved |
| batter_total_bases | 7253 | 48 | 1.365 | 1.398 | 0.033 | True | True | 46037 | 112054 | 38.0% | 0.050 | False | clv_truth_not_confirming |
| batter_home_runs | 3476 | 17 | 0.238 | 0.306 | 0.067 | True | True | 0 | 44967 | 35.4% | 0.055 | False | clv_truth_not_confirming, true_pair_rows<150 |
