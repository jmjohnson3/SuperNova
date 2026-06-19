# MLB Hitter Event Feature Ablation

Generated: 2026-06-19T11:23:41.758586+00:00
Rows: 51213 | Train: 43473 | Holdout: 7740
Status: ok

Each row removes one feature family from the full LightGBM head. Positive gain means that family hurt the true date holdout.

| Variant | Hit Brier | XBH Brier | HR Brier | 2B/3B Brier | PA MAE |
|---|---:|---:|---:|---:|---:|
| full | 0.17163 | 0.22782 | 0.24042 | 0.07900 | 0.688 |
| without_lineup | 0.17167 | 0.22814 | 0.24032 | 0.07903 | 0.697 |
| without_park | 0.17164 | 0.22791 | 0.24168 | 0.08007 | 0.691 |
| without_batter_statcast | 0.17170 | 0.22865 | 0.24135 | 0.07856 | 0.691 |
| without_pitcher_statcast | 0.17159 | 0.22808 | 0.24084 | 0.07883 | 0.690 |
| without_discipline | 0.17160 | 0.22791 | 0.24044 | 0.07933 | 0.694 |

## Pruning Policy

| Head | Removed Feature Groups | Full Metric |
|---|---|---:|
| hit | none | 0.17163 |
| XBH given hit | none | 0.22782 |
| HR given XBH | +lineup | 0.24042 |
| double/triple split | +batter_statcast, +pitcher_statcast | 0.07900 |
| PA | none | 0.68848 |

Known postgame proxy fields are excluded from this policy. Market fields remain diagnostic-only and cannot enter the player projection heads.
