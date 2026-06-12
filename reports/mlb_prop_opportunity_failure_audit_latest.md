# MLB Prop Opportunity Failure Audit

Generated: 2026-06-12T02:44:09.009941+00:00
Rows scanned: 43184
Status: ok

## Reason Counts

| Reason | Rows |
|---|---:|
| ok_opportunity | 30911 |
| bad_pa_projection | 5694 |
| projected_pa_below_gate | 4188 |
| missing_confirmed_lineup_slot | 2933 |
| no_live_lineup_source | 2933 |
| low_actual_pa | 2585 |
| bottom_order_overprojected | 1547 |

## By Section

| Section | Reason | Rows |
|---|---|---:|
| graded_training | ok_opportunity | 28511 |
| graded_training | bad_pa_projection | 5694 |
| graded_training | projected_pa_below_gate | 3492 |
| active_pending | missing_confirmed_lineup_slot | 2933 |
| active_pending | no_live_lineup_source | 2933 |
| graded_training | low_actual_pa | 2585 |
| active_pending | ok_opportunity | 2400 |
| graded_training | bottom_order_overprojected | 1547 |
| active_pending | projected_pa_below_gate | 696 |

## Market Summary

| Section | Market | Side | Rows | Slot | Proj PA | Avg Proj PA | Avg Actual PA | PA MAE | Bad PA | Low Proj PA | Low Actual PA |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| active_pending | batter_hits | over | 1987 | 48.7% | 100.0% | 3.78 | - | - | - | 12.5% | - |
| active_pending | batter_hits | under | 480 | 46.7% | 100.0% | 3.78 | - | - | - | 11.9% | - |
| active_pending | batter_home_runs | over | 924 | 47.4% | 100.0% | 3.77 | - | - | - | 13.2% | - |
| active_pending | batter_total_bases | over | 2053 | 47.8% | 100.0% | 3.80 | - | - | - | 12.5% | - |
| active_pending | batter_total_bases | under | 205 | 51.2% | 100.0% | 4.05 | - | - | - | 5.9% | - |
| graded_training | batter_hits | over | 14135 | 100.0% | 100.0% | 3.89 | 4.06 | 0.72 | 15.4% | 9.8% | 7.0% |
| graded_training | batter_hits | under | 3124 | 100.0% | 100.0% | 3.89 | 4.05 | 0.73 | 15.4% | 10.1% | 7.3% |
| graded_training | batter_home_runs | over | 5706 | 100.0% | 100.0% | 3.88 | 4.05 | 0.72 | 15.3% | 10.0% | 7.2% |
| graded_training | batter_total_bases | over | 12898 | 100.0% | 100.0% | 3.91 | 4.08 | 0.72 | 15.0% | 9.1% | 6.8% |
| graded_training | batter_total_bases | under | 1672 | 100.0% | 100.0% | 4.13 | 4.29 | 0.69 | 13.6% | 2.6% | 4.4% |

## Example Rows

| Section | Date | Player | Market | Side | Slot | Source | Proj PA | Actual PA | Reasons |
|---|---|---|---|---|---:|---|---:|---:|---|
| active_pending | 2026-06-11 | A.J. Ewing | batter_hits | over | - |  | 3.90 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | A.J. Ewing | batter_hits | under | - |  | 3.90 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Adley Rutschman | batter_hits | over | - |  | 4.40 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Adley Rutschman | batter_hits | under | - |  | 4.40 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Alex Freeland | batter_hits | over | - |  | 3.30 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Alex Freeland | batter_hits | under | - |  | 3.30 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Andrew Benintendi | batter_hits | under | - |  | 3.60 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Andrew Benintendi | batter_hits | over | - |  | 3.60 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Andy Pages | batter_hits | over | - |  | 4.70 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Andy Pages | batter_hits | under | - |  | 4.70 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Austin Riley | batter_hits | over | - |  | 3.40 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Austin Riley | batter_hits | under | - |  | 3.40 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Brandon Lowe | batter_hits | over | - |  | 4.00 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Brandon Lowe | batter_hits | under | - |  | 4.00 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Bryan Reynolds | batter_hits | over | - |  | 4.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Bryan Reynolds | batter_hits | under | - |  | 4.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Chase Meidroth | batter_hits | over | - |  | 4.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Chase Meidroth | batter_hits | under | - |  | 4.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Coby Mayo | batter_hits | over | - |  | 3.80 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Coby Mayo | batter_hits | under | - |  | 3.80 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Cole Young | batter_hits | under | - |  | 4.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Cole Young | batter_hits | over | - |  | 4.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Colt Emerson | batter_hits | over | - |  | 3.60 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Colt Emerson | batter_hits | under | - |  | 3.60 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Colt Keith | batter_hits | over | 6 | actual | 3.00 | - | projected_pa_below_gate |
| active_pending | 2026-06-11 | Colt Keith | batter_hits | under | 6 | actual | 3.00 | - | projected_pa_below_gate |
| active_pending | 2026-06-11 | Colton Cowser | batter_hits | under | - |  | 3.50 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Colton Cowser | batter_hits | over | - |  | 3.50 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Connor Norby | batter_hits | over | - |  | 3.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Connor Norby | batter_hits | under | - |  | 3.20 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Dalton Rushing | batter_hits | over | - |  | 3.50 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Dalton Rushing | batter_hits | under | - |  | 3.50 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Dominic Canzone | batter_hits | over | - |  | 2.90 | - | missing_confirmed_lineup_slot, no_live_lineup_source, projected_pa_below_gate |
| active_pending | 2026-06-11 | Dominic Canzone | batter_hits | under | - |  | 2.90 | - | missing_confirmed_lineup_slot, no_live_lineup_source, projected_pa_below_gate |
| active_pending | 2026-06-11 | Dominic Smith | batter_hits | over | - |  | 3.30 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Dominic Smith | batter_hits | under | - |  | 3.30 | - | missing_confirmed_lineup_slot, no_live_lineup_source |
| active_pending | 2026-06-11 | Edgar Quero | batter_hits | over | - |  | 2.90 | - | missing_confirmed_lineup_slot, no_live_lineup_source, projected_pa_below_gate |
| active_pending | 2026-06-11 | Edgar Quero | batter_hits | under | - |  | 2.90 | - | missing_confirmed_lineup_slot, no_live_lineup_source, projected_pa_below_gate |
| active_pending | 2026-06-11 | Elias Díaz | batter_hits | over | 8 | actual | 2.50 | - | projected_pa_below_gate |
| active_pending | 2026-06-11 | Elias Díaz | batter_hits | under | 8 | actual | 2.50 | - | projected_pa_below_gate |
