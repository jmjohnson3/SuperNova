# MLB Hitter Event Feature Ablation

Generated: 2026-06-24T08:03:29.128799+00:00
Rows: 51119 | Train: 43360 | Holdout: 7759
Status: ok

Each row removes one feature family from the full LightGBM head. Positive gain means that family hurt the true date holdout.

| Variant | Hit Brier | XBH Brier | HR Brier | 2B/3B Brier | PA MAE |
|---|---:|---:|---:|---:|---:|
| full | 0.17409 | 0.22845 | 0.24019 | 0.07890 | 0.706 |
| without_lineup | 0.17406 | 0.22838 | 0.23993 | 0.07911 | 0.700 |
| without_park | 0.17416 | 0.22852 | 0.24052 | 0.08005 | 0.705 |
| without_batter_statcast | 0.17403 | 0.22949 | 0.24205 | 0.07882 | 0.703 |
| without_pitcher_statcast | 0.17407 | 0.22868 | 0.24063 | 0.07859 | 0.712 |
| without_discipline | 0.17396 | 0.22877 | 0.23979 | 0.07858 | 0.703 |

## Pruning Policy

| Head | Removed Feature Groups | Full Metric |
|---|---|---:|
| hit | +discipline | 0.17409 |
| XBH given hit | none | 0.22845 |
| HR given XBH | +lineup, +discipline | 0.24019 |
| double/triple split | +pitcher_statcast, +discipline | 0.07890 |
| PA | +lineup, +batter_statcast, +discipline | 0.70568 |

Known postgame proxy fields are excluded from this policy. Market fields remain diagnostic-only and cannot enter the player projection heads.
