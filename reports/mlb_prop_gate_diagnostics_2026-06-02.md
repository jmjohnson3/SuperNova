# MLB Blocked Prop Diagnostics

Date: 2026-06-02
Historical window: 2025-12-04 to 2026-06-01
Purpose: explain blocked positive-EV props and identify recalibration candidates.

## Threshold Status

| Threshold | Value | Status | Optimizer Reason |
| --- | --- | --- | --- |
| threshold_strikeouts | 999.000 | disabled | holdout_too_small_disabled |
| threshold_strikeouts_over | 1.000 | active | accepted |
| threshold_strikeouts_under | 999.000 | disabled | holdout_too_small_disabled |
| threshold_hits | 0.750 | active | accepted |
| threshold_total_bases | 999.000 | disabled | holdout_too_small_disabled |
| threshold_total_bases_over | 999.000 | disabled | train_roi_non_positive_disabled |
| threshold_total_bases_under | 0.542 | active | accepted |
| threshold_home_runs_over | 999.000 | disabled | no_train_candidate_passed_min_bets_disabled |
| threshold_home_runs_under | 0.450 | active | no_train_candidate_passed_min_bets_disabled |
| threshold_clf | 999.000 | disabled | holdout_too_small_disabled |
| min_ev | 0.020 | active | no_ev_rows |

## Today By Stat

| Stat | Rows | Positive EV | Bankroll | Blocked | Max Edge | Max EV |
| --- | --- | --- | --- | --- | --- | --- |
| batter_hits | 262 | 8 | 0 | 262 | 0.702 | 0.4721 |
| batter_home_runs | 249 | 0 | 0 | 249 | 0.454 | - |
| batter_total_bases | 252 | 34 | 0 | 252 | 0.824 | 0.3676 |
| pitcher_strikeouts | 28 | 20 | 0 | 28 | 2.092 | 0.4361 |

## Block Reasons

| Stat | Reason | Rows |
| --- | --- | --- |
| batter_hits | below_edge_threshold:threshold_hits | 262 |
| batter_hits | ev_below_min | 253 |
| batter_home_runs | hr_longshot_variance | 249 |
| batter_home_runs | missing_ev | 249 |
| batter_home_runs | missing_price | 249 |
| batter_home_runs | threshold_disabled:threshold_clf | 249 |
| batter_home_runs | unbookable_under | 249 |
| batter_total_bases | missing_ev | 157 |
| batter_total_bases | missing_price | 157 |
| batter_hits | heavy_juice | 145 |
| batter_total_bases | below_edge_threshold:threshold_total_bases_under | 134 |
| batter_total_bases | zero_kelly | 83 |
| batter_total_bases | ev_below_min | 61 |
| batter_total_bases | threshold_disabled:threshold_total_bases_over | 57 |
| batter_total_bases | missing_side | 41 |
| batter_total_bases | threshold_disabled:threshold_total_bases | 41 |
| pitcher_strikeouts | threshold_disabled:threshold_strikeouts_under | 23 |
| batter_hits | zero_kelly | 21 |
| batter_total_bases | heavy_juice | 16 |
| batter_hits | unbookable_under | 7 |
| pitcher_strikeouts | ev_below_min | 7 |
| pitcher_strikeouts | zero_kelly | 5 |
| pitcher_strikeouts | below_edge_threshold:threshold_strikeouts_over | 4 |
| batter_hits | missing_ev | 1 |
| batter_hits | missing_price | 1 |
| pitcher_strikeouts | missing_ev | 1 |
| pitcher_strikeouts | missing_price | 1 |
| pitcher_strikeouts | missing_side | 1 |
| pitcher_strikeouts | threshold_disabled:threshold_strikeouts | 1 |

## Positive-EV Blocked Watchlist

| Player | Team | Stat | Side | Line | Price | Pred | P(over) | Edge | EV | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nick Allen | HOU | batter_hits | under | 0.5 | +111 | 0.360 | 30.2% | +0.140 | +47.2% | below_edge_threshold:threshold_hits; unbookable_under |
| Logan Gilbert | SEA | pitcher_strikeouts | under | 6.5 | -142 | 4.408 | 15.7% | +2.092 | +43.6% | threshold_disabled:threshold_strikeouts_under |
| Jhostynxon Garcia | PIT | batter_hits | under | 0.5 | +110 | 0.413 | 33.8% | +0.087 | +39.0% | below_edge_threshold:threshold_hits; unbookable_under |
| Jonathan Aranda | TB | batter_total_bases | over | 1.5 | +115 | 2.162 | 63.6% | +0.662 | +36.8% | threshold_disabled:threshold_total_bases_over |
| Dillon Dingler | DET | batter_total_bases | over | 1.5 | +100 | 2.324 | 67.5% | +0.824 | +34.9% | threshold_disabled:threshold_total_bases_over |
| Riley Greene | DET | batter_total_bases | over | 1.5 | +135 | 1.908 | 56.9% | +0.408 | +33.6% | threshold_disabled:threshold_total_bases_over |
| Connor Prielipp | MIN | pitcher_strikeouts | under | 4.5 | +113 | 4.024 | 37.6% | +0.476 | +33.0% | threshold_disabled:threshold_strikeouts_under |
| Colt Emerson | SEA | batter_hits | under | 0.5 | +115 | 0.480 | 38.1% | +0.020 | +33.0% | below_edge_threshold:threshold_hits; unbookable_under |
| David Hamilton | MIL | batter_hits | under | 0.5 | +115 | 0.492 | 38.9% | +0.008 | +31.5% | below_edge_threshold:threshold_hits; unbookable_under |
| Grayson Rodriguez | LAA | pitcher_strikeouts | under | 5.5 | +105 | 4.882 | 36.3% | +0.618 | +30.5% | threshold_disabled:threshold_strikeouts_under |
| Michael Busch | CHC | batter_total_bases | over | 1.5 | +185 | 1.541 | 45.6% | +0.041 | +29.9% | threshold_disabled:threshold_total_bases_over |
| Randy Vásquez | SD | pitcher_strikeouts | under | 3.5 | -104 | 2.940 | 33.9% | +0.560 | +29.6% | threshold_disabled:threshold_strikeouts_under |
| Junior Caminero | TB | batter_total_bases | over | 1.5 | +100 | 2.184 | 64.1% | +0.684 | +28.3% | threshold_disabled:threshold_total_bases_over |
| Joey Cantillo | CLE | pitcher_strikeouts | under | 4.5 | -112 | 3.767 | 32.6% | +0.733 | +27.6% | threshold_disabled:threshold_strikeouts_under |
| Bubba Chandler | PIT | pitcher_strikeouts | under | 4.5 | +104 | 4.021 | 37.5% | +0.479 | +27.5% | threshold_disabled:threshold_strikeouts_under |
| Michael Soroka | ARI | pitcher_strikeouts | under | 4.5 | -116 | 3.733 | 31.9% | +0.767 | +26.8% | threshold_disabled:threshold_strikeouts_under |
| Kevin McGonigle | DET | batter_total_bases | over | 1.5 | +130 | 1.804 | 53.8% | +0.304 | +23.8% | threshold_disabled:threshold_total_bases_over |
| Manny Machado | SD | batter_total_bases | over | 1.5 | +125 | 1.843 | 55.0% | +0.343 | +23.7% | threshold_disabled:threshold_total_bases_over |
| Yandy Díaz | TB | batter_total_bases | over | 1.5 | -105 | 2.145 | 63.2% | +0.645 | +23.4% | threshold_disabled:threshold_total_bases_over |
| Patrick Wisdom | SEA | batter_hits | under | 0.5 | -116 | 0.413 | 33.9% | +0.087 | +23.2% | below_edge_threshold:threshold_hits; unbookable_under |
| Gage Jump | ATH | pitcher_strikeouts | under | 4.5 | -102 | 4.057 | 38.2% | +0.443 | +22.3% | threshold_disabled:threshold_strikeouts_under |
| Mitch Garver | SEA | batter_hits | under | 0.5 | -110 | 0.458 | 36.8% | +0.042 | +20.7% | below_edge_threshold:threshold_hits; unbookable_under |
| Brent Rooker | ATH | batter_total_bases | over | 1.5 | +140 | 1.683 | 50.2% | +0.183 | +20.4% | threshold_disabled:threshold_total_bases_over |
| James Wood | WAS | batter_total_bases | over | 1.5 | +115 | 1.871 | 55.8% | +0.371 | +20.0% | threshold_disabled:threshold_total_bases_over |
| Nick Kurtz | ATH | batter_total_bases | over | 1.5 | +115 | 1.869 | 55.7% | +0.369 | +19.9% | threshold_disabled:threshold_total_bases_over |
| Victor Mesa Jr. | TB | batter_total_bases | over | 1.5 | +140 | 1.676 | 49.9% | +0.176 | +19.8% | threshold_disabled:threshold_total_bases_over |
| Jackson Merrill | SD | batter_total_bases | over | 1.5 | +125 | 1.766 | 52.7% | +0.266 | +18.6% | threshold_disabled:threshold_total_bases_over |
| Davis Martin | CWS | pitcher_strikeouts | over | 5.5 | +120 | 5.897 | 53.8% | +0.397 | +18.3% | below_edge_threshold:threshold_strikeouts_over |
| Steven Matz | TB | pitcher_strikeouts | under | 4.5 | -142 | 3.675 | 30.8% | +0.825 | +17.9% | threshold_disabled:threshold_strikeouts_under |
| Kevin Gausman | TOR | pitcher_strikeouts | under | 5.5 | -122 | 4.828 | 35.4% | +0.672 | +17.6% | threshold_disabled:threshold_strikeouts_under |
| Andrew Abbott | CIN | pitcher_strikeouts | under | 4.5 | -140 | 3.717 | 31.6% | +0.783 | +17.2% | threshold_disabled:threshold_strikeouts_under |
| Fernando Tatis Jr. | SD | batter_total_bases | over | 1.5 | +100 | 1.901 | 56.6% | +0.401 | +13.3% | threshold_disabled:threshold_total_bases_over |
| Aaron Judge | NYY | batter_total_bases | over | 1.5 | +100 | 1.876 | 56.0% | +0.377 | +11.9% | threshold_disabled:threshold_total_bases_over |
| Jameson Taillon | CHC | pitcher_strikeouts | under | 4.5 | -104 | 4.302 | 43.0% | +0.198 | +11.8% | threshold_disabled:threshold_strikeouts_under |
| Iván Herrera | STL | batter_total_bases | over | 1.5 | +150 | 1.513 | 44.7% | +0.013 | +11.7% | threshold_disabled:threshold_total_bases_over |
| Eric Lauer | LAD | pitcher_strikeouts | under | 3.5 | -166 | 2.783 | 30.4% | +0.717 | +11.5% | threshold_disabled:threshold_strikeouts_under |
| Noah Cameron | KC | pitcher_strikeouts | under | 5.5 | -154 | 4.713 | 33.4% | +0.786 | +9.9% | threshold_disabled:threshold_strikeouts_under |
| Colson Montgomery | CWS | batter_total_bases | over | 1.5 | +115 | 1.712 | 51.1% | +0.212 | +9.8% | threshold_disabled:threshold_total_bases_over |
| Edgar Quero | CWS | batter_total_bases | under | 1.5 | -201 | 1.020 | 27.2% | +0.480 | +9.1% | below_edge_threshold:threshold_total_bases_under; heavy_juice |
| Ben Rice | NYY | batter_total_bases | over | 1.5 | +120 | 1.659 | 49.4% | +0.160 | +8.7% | threshold_disabled:threshold_total_bases_over |

## Fallback Reopen Test

These would clear the old/default edge threshold and hard filters today. They are not bankroll bets unless the historical recalibration supports reopening the bucket.

| Player | Team | Stat | Side | Line | Price | Pred | P(over) | Edge | EV | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logan Gilbert | SEA | pitcher_strikeouts | under | 6.5 | -142 | 4.408 | 15.7% | +2.092 | +43.6% | threshold_reopen_needed |

## Recalibration Scan

Buckets are stat/side/edge_type. REOPEN_CANDIDATE means the current threshold is disabled, train and holdout have enough sample, and holdout ROI is positive.

| Bucket | Current | Suggested | Train N | Train W-L | Train ROI | Holdout N | Holdout W-L | Holdout ROI | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter_total_bases/under/count | 0.542 | 0.679 | 488 | 346-142 | +35.4% | 43 | 39-4 | +73.2% | ACTIVE_OK |
| batter_hits/over/count | 0.750 | 0.750 | 53 | 38-15 | +36.9% | 35 | 25-10 | +36.4% | ACTIVE_OK |
| pitcher_strikeouts/under/count | disabled | 1.250 | 88 | 53-35 | +15.0% | 9 | 6-3 | +27.3% | KEEP_DISABLED_SAMPLE |
| pitcher_strikeouts/over/count | 1.000 | 1.492 | 43 | 29-14 | +28.8% | 10 | 7-3 | +32.6% | ACTIVE_SAMPLE_LOW |
| batter_total_bases/over/count | disabled | 0.600 | 314 | 146-168 | -11.2% | 70 | 40-30 | +9.2% | KEEP_DISABLED_TRAIN_BAD |
| batter_home_runs/over/count | disabled | - | 0 | 0-0 | - | 0 | 0-0 | - | train_sample_too_small |
