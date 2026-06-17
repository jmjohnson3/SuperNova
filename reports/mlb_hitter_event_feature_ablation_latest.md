# MLB Hitter Event Feature Ablation

Generated: 2026-06-17T08:42:07.871037+00:00
Rows: 51146 | Train: 43365 | Holdout: 7781
Status: ok

The `+market` row is diagnostic context only; bankroll gating should still rely on locked offer-level reports.

| Feature Set | Event Model | PA MAE | Event Brier | Event Log Loss | Hits MAE | TB MAE | HR MAE | PA Gain | Brier Gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | linear_multinomial | 0.518 | 0.48619 | 1.00185 | 0.639 | 1.258 | 0.204 | - | - |
| +lineup | linear_multinomial | 0.519 | 0.48622 | 1.00225 | 0.639 | 1.252 | 0.199 | -0.0005 | -0.00003 |
| +opportunity_v3 | linear_multinomial | 0.519 | 0.48622 | 1.00225 | 0.639 | 1.252 | 0.199 | 0.0000 | 0.00000 |
| +park | linear_multinomial | 0.519 | 0.48619 | 1.00216 | 0.640 | 1.254 | 0.199 | -0.0003 | 0.00003 |
| +batter_statcast | linear_multinomial | 0.519 | 0.48570 | 0.99983 | 0.641 | 1.253 | 0.199 | -0.0001 | 0.00049 |
| +pitcher_statcast | linear_multinomial | 0.519 | 0.48574 | 1.00012 | 0.641 | 1.253 | 0.199 | -0.0003 | -0.00003 |
| +discipline | linear_multinomial | 0.519 | 0.48562 | 0.99981 | 0.641 | 1.251 | 0.198 | -0.0000 | 0.00011 |
| +market | linear_multinomial | 0.519 | 0.48562 | 0.99981 | 0.641 | 1.251 | 0.198 | -0.0000 | -0.00000 |
