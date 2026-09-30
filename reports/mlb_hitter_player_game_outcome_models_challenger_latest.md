# MLB Hitter Player-Game Outcome Challenger

Generated UTC: 2026-07-14T22:32:00.151495+00:00
Release: hitter-pg-2026-07-06-r2
Rows: 12481 | Train: 10444 | Holdout: 2037
Holdout: 2026-07-06 to 2026-07-12

## Recommendation
- production_status: diagnostic_only
- passes_basic_gate: False
- direct_hits_count_repair_enabled: True
- direct_hits_count_repair_mae_gain: 0.029958470857185504
- direct_hr_count_repair_enabled: True
- direct_hr_count_repair_mae_gain: 0.04688047454353711
- direct_tb_count_repair_enabled: False
- direct_tb_count_repair_mae_gain: -0.02855274889141346
- direct_tb_count_repair_alpha: 0.0

## PA
| Model | Rows | MAE | Bias |
|---|---:|---:|---:|
| Selected | 2037 | 0.6892053826559323 | -0.17193534293135704 |
| Slot prior | 2037 | 0.9077541704969911 | 0.02654720667985309 |
| Existing projected PA | 2026 | 0.8431834563374543 | -0.08488890455820365 |
| Two-part | 2037 | 0.7356982956329052 | -0.017112496320903232 |

## Structured Counts
| Target | Rows | Model MAE | Prior MAE | Bias |
|---|---:|---:|---:|---:|
| Hits | 1970 | 0.6809355029409279 | 0.6750578837135399 | -0.0402229218384379 |
| TB | 1970 | 1.3276894564415536 | 1.2728757223909302 | -0.06122478211275642 |
| HR | 1970 | 0.21957649011024113 | 0.21878805848209693 | -0.004558276663643912 |

## Direct Event Curve
- Active event curve: hierarchical_conditional_lgbm
- Weighted event Brier: 0.482846182890745
- TB-state residual enabled: True
- TB-state validation Brier gain: 2.423170575505118e-05

## Interpretation
Hits and HR direct count repairs are accepted as offline challengers. TB remains diagnostic-only because direct TB/XBH repair did not produce repeated walk-forward gain.
