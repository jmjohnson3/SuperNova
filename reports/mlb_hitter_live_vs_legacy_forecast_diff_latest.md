# MLB Hitter Live vs Legacy Forecast Diff

Generated UTC: 2026-09-01T11:14:37Z
Phase: day_pregame
Lookback days: 45
Rows: 3520
Freeze status: live_forecast_artifact_pinned

## Graded Comparison

| Stat | Rows | Legacy MAE | Live MAE | Gain | Legacy Bias | Live Bias |
|---|---:|---:|---:|---:|---:|---:|
| Hits | 2540 | 0.717 | 0.734 | -0.018 | 0.091 | -0.306 |
| HR | 2540 | 0.306 | 0.236 | 0.070 | 0.149 | 0.054 |
| TB | 2540 | 1.539 | 1.455 | 0.084 | 0.624 | 0.450 |

### Line Brier

| Line | Rows | Legacy Brier | Live Brier | Gain |
|---|---:|---:|---:|---:|
| TB 1.5 | 2540 | 0.285 | 0.265 | 0.020 |
| TB 2.5 | 2540 | 0.189 | 0.173 | 0.015 |
| TB 3.5 | 2540 | 0.120 | 0.115 | 0.005 |
| HR 0.5 | 2540 | 0.110 | 0.092 | 0.017 |

## Hits Live Bias Repair V1

Prospective repair trained on the actual shadow-vs-legacy ledger rows. It must beat both live shadow and legacy before the predictor can use it.

| Accepted | Reason | Rows | Dates | Alpha | Live MAE | Legacy MAE | Selected MAE | Gain vs Live | Gain vs Legacy | Live Bias | Selected Bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| False | any_brier_does_not_beat_legacy | 2078 | 10 | 0.00 | 0.732 | 0.717 | 0.732 | 0.000 | -0.014 | -0.292 | -0.292 |

| Alpha | MAE | Gain vs Live | Gain vs Legacy | Any Brier | Brier Gain vs Live | Brier Gain vs Legacy | Bias | Mean Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.732 | 0.000 | -0.014 | 0.270 | 0.000 | -0.032 | -0.292 | 0.000 |
| 0.20 | 0.724 | 0.008 | -0.006 | 0.260 | 0.010 | -0.022 | -0.234 | 0.058 |
| 0.35 | 0.719 | 0.013 | -0.001 | 0.254 | 0.016 | -0.016 | -0.190 | 0.102 |
| 0.50 | 0.715 | 0.017 | 0.003 | 0.249 | 0.021 | -0.011 | -0.147 | 0.145 |
| 0.65 | 0.712 | 0.020 | 0.006 | 0.246 | 0.024 | -0.008 | -0.103 | 0.189 |
| 0.80 | 0.710 | 0.022 | 0.008 | 0.243 | 0.027 | -0.006 | -0.059 | 0.232 |
| 0.95 | 0.709 | 0.023 | 0.008 | 0.242 | 0.028 | -0.004 | -0.016 | 0.276 |
| 1.00 | 0.709 | 0.022 | 0.008 | 0.242 | 0.029 | -0.004 | -0.001 | 0.291 |

## Biggest Movers

| Date | Player | Team | Opp | H Old -> Live | HR Old -> Live | TB Old -> Live | PA | Slot | Context |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-30 | Jake Mangum | PIT | CIN | 1.661 -> 0.869 | 0.119 -> 0.088 | 2.645 -> 2.592 | 4.500 | 1.0 | top-order PA support, high projected PA, opposite-hand matchup, park factor missing |
| 2026-07-30 | Chandler Simpson | TB | TEX | 1.561 -> 0.780 | 0.013 -> 0.031 | 1.876 -> 1.908 | 3.800 | 5.0 | middle-order neutral, opposite-hand matchup, park factor missing |
| 2026-07-30 | Shohei Ohtani | LAD | SEA | 1.594 -> 0.835 | 0.477 -> 0.297 | 3.528 -> 3.229 | 4.500 | 1.0 | top-order PA support, high projected PA, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Matt Olson | ATL | WAS | 1.571 -> 0.823 | 0.574 -> 0.342 | 3.731 -> 3.345 | 4.500 | 3.0 | middle-order neutral, high projected PA, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Wilyer Abreu | BOS | ATH | 1.567 -> 0.822 | 0.671 -> 0.389 | 3.451 -> 2.985 | 4.000 | 4.0 | middle-order neutral, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Nick Gonzales | PIT | CIN | 1.585 -> 0.844 | 0.252 -> 0.152 | 2.830 -> 2.659 | 4.500 | 6.0 | middle-order neutral, high projected PA, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Drake Baldwin | ATL | WAS | 1.541 -> 0.809 | 0.600 -> 0.352 | 3.669 -> 3.242 | 4.600 | 1.0 | top-order PA support, high projected PA, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Jac Caglianone | KC | MIN | 1.448 -> 0.724 | 0.631 -> 0.365 | 3.146 -> 2.755 | 3.600 | 3.0 | middle-order neutral, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | JJ Wetherholt | STL | CHC | 1.516 -> 0.796 | 0.474 -> 0.276 | 2.966 -> 2.584 | 4.300 | 1.0 | top-order PA support, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Brandon Lowe | PIT | CIN | 1.504 -> 0.790 | 0.571 -> 0.344 | 3.535 -> 3.143 | 4.300 | 2.0 | top-order PA support, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Yandy Díaz | TB | TEX | 1.515 -> 0.808 | 0.525 -> 0.305 | 3.173 -> 2.757 | 4.700 | 1.0 | top-order PA support, high projected PA, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Kody Clemens | MIN | KC | 1.452 -> 0.761 | 0.354 -> 0.226 | 2.861 -> 2.638 | 4.100 | 4.6 | middle-order neutral, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | James Wood | WAS | ATL | 1.452 -> 0.764 | 0.535 -> 0.322 | 3.642 -> 3.314 | 4.700 | 1.0 | top-order PA support, high projected PA, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Luis Arraez | SF | SD | 1.473 -> 0.788 | 0.036 -> 0.045 | 1.991 -> 2.006 | 4.300 | 1.0 | top-order PA support, same-hand matchup, park factor missing |
| 2026-07-30 | Caleb Durbin | BOS | ATH | 1.363 -> 0.681 | 0.497 -> 0.271 | 2.570 -> 2.313 | 3.700 | 5.0 | middle-order neutral, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Jordan Walker | STL | CHC | 1.466 -> 0.785 | 0.528 -> 0.308 | 3.112 -> 2.693 | 4.200 | 2.0 | top-order PA support, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Bryan Reynolds | PIT | CIN | 1.431 -> 0.754 | 0.519 -> 0.306 | 3.329 -> 2.967 | 4.400 | 3.0 | middle-order neutral, high projected PA, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Esmerlyn Valdez | PIT | CIN | 1.426 -> 0.751 | 0.464 -> 0.296 | 3.558 -> 3.314 | 4.100 | 4.0 | middle-order neutral, pitcher handedness missing, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Michael Harris II | ATL | WAS | 1.408 -> 0.742 | 0.397 -> 0.246 | 3.030 -> 2.777 | 4.100 | 4.0 | middle-order neutral, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Junior Caminero | TB | TEX | 1.421 -> 0.761 | 0.713 -> 0.405 | 3.688 -> 3.125 | 4.500 | 3.0 | middle-order neutral, high projected PA, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Ceddanne Rafaela | BOS | ATH | 1.421 -> 0.764 | 0.526 -> 0.295 | 3.286 -> 2.885 | 4.100 | 2.0 | top-order PA support, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Nicky Lopez | TEX | TB | 1.294 -> 0.647 | 0.009 -> 0.025 | 1.454 -> 1.483 | 3.300 | 9.0 | bottom-order PA drag, low projected PA, same-hand matchup, park factor missing |
| 2026-07-30 | Anthony Seigler | BOS | ATH | 1.292 -> 0.646 | 0.470 -> 0.259 | 3.182 -> 2.851 | 4.100 | 8.0 | bottom-order PA drag, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Ben Rice | NYY | CWS | 1.360 -> 0.718 | 0.469 -> 0.293 | 3.176 -> 2.885 | 4.300 | 2.0 | top-order PA support, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Alec Burleson | STL | CHC | 1.359 -> 0.718 | 0.406 -> 0.252 | 2.947 -> 2.685 | 4.200 | 4.0 | middle-order neutral, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Lars Nootbaar | STL | CHC | 1.268 -> 0.634 | 0.350 -> 0.212 | 2.542 -> 2.295 | 3.900 | 5.7 | middle-order neutral, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Seiya Suzuki | CHC | STL | 1.359 -> 0.730 | 0.351 -> 0.224 | 2.870 -> 2.661 | 4.900 | 2.0 | top-order PA support, high projected PA, same-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Jonathan Aranda | TB | TEX | 1.319 -> 0.698 | 0.452 -> 0.276 | 3.110 -> 2.824 | 4.200 | 2.0 | top-order PA support, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |
| 2026-07-30 | Nico Hoerner | CHC | STL | 1.339 -> 0.720 | 0.054 -> 0.052 | 1.683 -> 1.680 | 4.500 | 6.0 | middle-order neutral, high projected PA, same-hand matchup, park factor missing |
| 2026-07-30 | Pete Crow-Armstrong | CHC | STL | 1.311 -> 0.693 | 0.270 -> 0.185 | 2.764 -> 2.633 | 4.600 | 1.0 | top-order PA support, high projected PA, opposite-hand matchup, park factor missing, TB rebuild pulled projection down |

## True-Pair Offer Check

True-paired active hitter offer rows: 4922

| Stat | Side | Book | Rows | Win Rate | CLV Beat | Avg CLV | Avg EV |
|---|---|---|---:|---:|---:|---:|---:|
| batter_hits | over | draftkings | 1650 | 0.557 | 0.288 | 0.025 | -0.064 |
| batter_hits | under | draftkings | 1650 | 0.557 | 0.310 | -0.028 | -0.063 |
| batter_total_bases | over | draftkings | 811 | 0.406 | 0.326 | 0.112 | -0.065 |
| batter_total_bases | under | draftkings | 811 | 0.406 | 0.322 | -0.114 | -0.094 |
