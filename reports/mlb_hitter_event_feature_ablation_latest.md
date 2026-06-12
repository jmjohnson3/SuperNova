# MLB Hitter Event Feature Ablation

Generated: 2026-06-12T15:19:28.237422+00:00
Rows: 51249 | Train: 43355 | Holdout: 7894
Status: ok

The `+market` row is diagnostic context only; bankroll gating should still rely on locked offer-level reports.

| Feature Set | PA MAE | Event Brier | Event Log Loss | Hits MAE | TB MAE | HR MAE | PA Gain | Brier Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.521 | 0.48582 | 1.00059 | 0.637 | 1.248 | 0.200 | - | - |
| +lineup | 0.521 | 0.48585 | 1.00079 | 0.637 | 1.242 | 0.195 | -0.0004 | -0.00003 |
| +park | 0.521 | 0.48578 | 1.00070 | 0.639 | 1.243 | 0.194 | -0.0002 | 0.00007 |
| +batter_statcast | 0.521 | 0.48548 | 0.99908 | 0.640 | 1.243 | 0.195 | 0.0002 | 0.00030 |
| +pitcher_statcast | 0.521 | 0.48538 | 0.99899 | 0.641 | 1.242 | 0.194 | -0.0000 | 0.00010 |
| +discipline | 0.521 | 0.48527 | 0.99885 | 0.640 | 1.239 | 0.194 | 0.0000 | 0.00011 |
| +market | 0.521 | 0.48527 | 0.99885 | 0.640 | 1.239 | 0.194 | 0.0000 | -0.00000 |
