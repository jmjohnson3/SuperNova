# NBA integer-count probability research

Development analysis on saved walk-forward forecasts and actually recorded pregame DraftKings quotes.
No live model or frozen strategy was changed. These are not FanDuel execution results.

| Period / stat | Offers | Frozen Brier | Count Brier | Market Brier | Count paper ROI | Decision |
|---|---:|---:|---:|---:|---:|---|
| 2025-26_later_period / all | 6917 | 0.25726 | 0.25024 | 0.24858 | +0.63% | consider_prospective_comparison_only |
| 2025-26_later_period / ast | 1821 | 0.26552 | 0.25075 | 0.24699 | -3.48% | consider_prospective_comparison_only |
| 2025-26_later_period / pts | 2643 | 0.25523 | 0.25163 | 0.24936 | +1.92% | consider_prospective_comparison_only |
| 2025-26_later_period / reb | 2453 | 0.25332 | 0.24837 | 0.24893 | +2.14% | consider_prospective_comparison_only |

Lower Brier is better. Paired game/day uncertainty intervals are in the JSON.
Paper ROI assumes the saved offer was available, which was not checked. Costs are not included.
The saved forecasts and historical outcomes are development data. No prospective efficacy is claimed.
No threshold search, model refit, or live registration was performed.
