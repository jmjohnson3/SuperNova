# MLB Hitter Event Feature Ablation

Generated: 2026-09-02T11:17:57.037974+00:00
Rows: 51826 | Train: 43837 | Holdout: 7989
Status: ok

Each row removes one feature family from the full LightGBM head. Positive gain means that family hurt the true date holdout.

| Variant | Hit Brier | XBH Brier | HR Brier | 2B/3B Brier | PA MAE |
|---|---:|---:|---:|---:|---:|
| full | 0.16922 | 0.21999 | 0.23675 | 0.07290 | 0.697 |
| without_lineup | 0.16919 | 0.22018 | 0.23634 | 0.07305 | 0.694 |
| without_park | 0.16925 | 0.22038 | 0.23651 | 0.07326 | 0.695 |
| without_batter_statcast | 0.16928 | 0.22148 | 0.23958 | 0.07323 | 0.695 |
| without_pitcher_statcast | 0.16915 | 0.22016 | 0.23624 | 0.07255 | 0.691 |
| without_discipline | 0.16922 | 0.22016 | 0.23719 | 0.07295 | 0.701 |

## Pruning Policy

| Head | Removed Feature Groups | Full Metric |
|---|---|---:|
| hit | none | 0.16922 |
| XBH given hit | none | 0.21999 |
| HR given XBH | +lineup, +park, +pitcher_statcast | 0.23675 |
| double/triple split | +pitcher_statcast | 0.07290 |
| PA | +lineup, +park, +pitcher_statcast | 0.69672 |

Known postgame proxy fields are excluded from this policy. Market fields remain diagnostic-only and cannot enter the player projection heads.
