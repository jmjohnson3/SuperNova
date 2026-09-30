# MLB Forecast Repair / Error Decomposition

Generated UTC: 2026-09-01T11:14:55Z
Canonical evidence phase: day_pregame
Prospective graded rows: 22883
Historical player-game fallback rows: 32001
Active diagnostic source: prospective_ledger

Count error is split exactly into opportunity error and per-opportunity player-rate error.
Offer duplicates are never used in this projection audit.

## Market Repair Queue

| Stat | Rows | Dates | MAE | Bias | Opportunity | Player Rate | Dominant Repair |
|---|---:|---:|---:|---:|---:|---:|---|
| batter_hits | 8615 | 25 | 0.727 | -0.167 | 0.144 | 0.684 | player_rate |
| batter_home_runs | 8615 | 25 | 0.246 | 0.052 | 0.032 | 0.248 | player_rate |
| batter_total_bases | 5139 | 25 | 1.384 | 0.160 | 0.290 | 1.322 | player_rate |
| pitcher_strikeouts | 514 | 25 | 1.845 | 0.182 | 0.683 | 1.754 | player_rate |

## Total Bases Repair

- Recommended repair: train_gated_direct_player_game_tb_head
- TB MAE / bias: 1.384 / 0.160
- Opportunity / player-rate error: 0.290 / 1.322
- Actual event rates per PA: {"double": 0.04308318972270093, "hr": 0.03269100214110397, "single": 0.14507284975716747, "triple": 0.004386652044493185}
- Direct TB repair enabled: False
- Walk-forward folds / positive: 0 / 0
- OOF MAE gain: -
- Production blend / bias offset: 0.000 / -

| Actual TB State | Rows | MAE | Bias | Dominant Repair |
|---|---:|---:|---:|---|
| 4+ HR | 580 | 3.382 | -3.381 | player_rate |
| 4+ non-HR | 141 | 2.670 | -2.670 | player_rate |
| 0 | 2148 | 1.478 | 1.478 | player_rate |
| 2-3 | 1048 | 0.824 | -0.657 | player_rate |
| 1 | 1222 | 0.601 | 0.552 | player_rate |

## Frozen Release / Opportunity Slices

| Slice | Rows | Dates | MAE | Bias | Opportunity | Player Rate | Dominant Repair |
|---|---:|---:|---:|---:|---:|---:|---|
| pitcher_strikeouts / current | 38 | 2 | 2.021 | 0.856 | 0.731 | 1.830 | player_rate |
| pitcher_strikeouts / pitcher-k-2026-07-03-r1 | 476 | 23 | 1.831 | 0.128 | 0.680 | 1.748 | player_rate |
| batter_total_bases / hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 1.412 | -0.281 | 0.240 | 1.332 | player_rate |
| batter_total_bases / hitter-pg-2026-07-06-r2 | 4299 | 21 | 1.398 | 0.234 | 0.298 | 1.339 | player_rate |
| batter_total_bases / hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 1.326 | -0.193 | 0.245 | 1.237 | player_rate |
| batter_total_bases / hitter-pg-2026-07-03-r1 | 481 | 2 | 1.257 | -0.199 | 0.251 | 1.187 | player_rate |
| batter_hits / hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 0.781 | -0.423 | 0.088 | 0.738 | player_rate |
| batter_hits / hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 0.777 | -0.437 | 0.091 | 0.719 | player_rate |
| batter_hits / hitter-rate-shadow-2026-07-09-r1 | 3476 | 17 | 0.730 | -0.310 | - | - | - |
| batter_hits / hitter-pg-2026-07-03-r1 | 481 | 2 | 0.721 | -0.381 | 0.095 | 0.685 | player_rate |
| batter_hits / hitter-pg-2026-07-06-r2 | 4299 | 21 | 0.720 | -0.005 | 0.155 | 0.680 | player_rate |
| batter_home_runs / hitter_pg:2026-07-03T08:22:56.064748+00:00 | 210 | 1 | 0.273 | 0.006 | 0.030 | 0.277 | player_rate |
| batter_home_runs / hitter_pg:2026-07-02T14:31:24.102972+00:00 | 149 | 1 | 0.258 | 0.047 | 0.031 | 0.257 | player_rate |
| batter_home_runs / hitter-pg-2026-07-03-r1 | 481 | 2 | 0.256 | 0.023 | 0.033 | 0.250 | player_rate |
| batter_home_runs / hitter-pg-2026-07-06-r2 | 4299 | 21 | 0.249 | 0.060 | 0.032 | 0.246 | player_rate |
| batter_home_runs / hitter-rate-shadow-2026-07-09-r1 | 3476 | 17 | 0.238 | 0.050 | - | - | - |

| Lineup Status | Rows | Dates | MAE | Bias | Dominant Repair |
|---|---:|---:|---:|---:|---|
| not_applicable | 514 | 25 | 1.845 | 0.182 | player_rate |
| confirmed_lineup | 17425 | 25 | 0.714 | 0.002 | player_rate |
| lineup_missing_or_unconfirmed | 4944 | 25 | 0.616 | -0.039 | player_rate |

| Opportunity Error Bucket | Rows | Dates | MAE | Bias | Dominant Repair |
|---|---:|---:|---:|---:|---|
| under_projected_by_1_plus | 2278 | 25 | 1.027 | -0.424 | player_rate |
| under_projected_by_0.5_to_1 | 2611 | 25 | 0.845 | -0.205 | player_rate |
| near_actual_within_0.5 | 6657 | 25 | 0.796 | 0.105 | player_rate |
| over_projected_by_1_plus | 2553 | 24 | 0.741 | 0.420 | player_rate |
| over_projected_by_0.5_to_1 | 1810 | 25 | 0.707 | 0.284 | player_rate |
| missing_opportunity | 6974 | 22 | 0.491 | -0.123 | - |

## Offer-Level Line / Market Slices

These slices are diagnostic only. They are duplicated offer rows and are not used for projection-model selection.

| Market | Side | Rows | Dates | Count MAE | Bias | Win | ROI | CLV Beat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pitcher_strikeouts | over | 5907 | 55 | 1.825 | -0.003 | 0.498 | -0.056 | 0.390 |
| pitcher_strikeouts | under | 5902 | 55 | 1.824 | -0.001 | 0.504 | -0.060 | 0.431 |
| batter_total_bases | under | 12870 | 57 | 1.536 | 0.069 | 0.585 | -0.042 | 0.325 |
| batter_total_bases | over | 110520 | 57 | 1.421 | 0.096 | 0.224 | -0.211 | 0.385 |
| batter_hits | under | 25189 | 57 | 0.723 | -0.023 | 0.445 | -0.041 | 0.351 |
| batter_hits | over | 86201 | 57 | 0.718 | -0.021 | 0.400 | -0.143 | 0.394 |
| batter_home_runs | over | 48878 | 56 | 0.269 | 0.057 | 0.066 | -0.324 | 0.354 |

| Market | Side | Line Bucket | Book | Rows | Count MAE | Bias | ROI | CLV Beat |
|---|---|---|---|---:|---:|---:|---:|---:|
| pitcher_strikeouts | over | K 6.5-8.0 | draftkings | 472 | 2.086 | 0.000 | -0.064 | 0.553 |
| pitcher_strikeouts | under | K 6.5-8.0 | draftkings | 469 | 2.084 | 0.001 | -0.062 | 0.364 |
| pitcher_strikeouts | over | K 6.5-8.0 | fanduel | 392 | 2.020 | 0.036 | -0.019 | 0.387 |
| pitcher_strikeouts | under | K 6.5-8.0 | fanduel | 392 | 2.020 | 0.036 | -0.113 | 0.370 |
| pitcher_strikeouts | over | K 4.5-6.0 | fanduel | 1818 | 1.840 | 0.026 | -0.044 | 0.331 |
| pitcher_strikeouts | under | K 4.5-6.0 | fanduel | 1818 | 1.840 | 0.026 | -0.066 | 0.386 |
| pitcher_strikeouts | over | K 4.5-6.0 | draftkings | 2010 | 1.813 | -0.021 | -0.052 | 0.443 |
| pitcher_strikeouts | under | K 4.5-6.0 | draftkings | 2010 | 1.810 | -0.015 | -0.055 | 0.471 |
| batter_total_bases | over | TB 2.5+ | draftkings | 58 | 1.728 | -0.121 | -0.110 | 0.694 |
| batter_total_bases | under | TB 2.5+ | draftkings | 58 | 1.728 | -0.121 | -0.031 | 0.265 |
| pitcher_strikeouts | under | K <4.5 | draftkings | 602 | 1.654 | 0.000 | -0.032 | 0.581 |
| pitcher_strikeouts | over | K <4.5 | draftkings | 604 | 1.653 | -0.003 | -0.083 | 0.344 |
| pitcher_strikeouts | under | K <4.5 | fanduel | 562 | 1.607 | -0.074 | -0.066 | 0.405 |
| pitcher_strikeouts | over | K <4.5 | fanduel | 562 | 1.607 | -0.074 | -0.077 | 0.316 |
| batter_total_bases | under | TB 1.5 | draftkings | 12812 | 1.535 | 0.069 | -0.042 | 0.325 |
| batter_total_bases | over | TB 1.5 | draftkings | 12760 | 1.535 | 0.073 | -0.094 | 0.460 |
| batter_total_bases | over | TB 1.5 | fanduel | 24451 | 1.407 | 0.098 | -0.174 | 0.377 |
| batter_total_bases | over | TB 2.5+ | fanduel | 73251 | 1.406 | 0.099 | -0.244 | 0.375 |
| batter_hits | over | H 1.5 | draftkings | 3579 | 0.801 | 0.009 | -0.164 | 0.491 |
| batter_hits | under | H 1.5 | draftkings | 3579 | 0.801 | 0.009 | -0.008 | 0.338 |
| batter_hits | over | H 1.5 | fanduel | 24437 | 0.719 | -0.017 | -0.191 | 0.411 |
| batter_hits | over | H 0.5 | fanduel | 24441 | 0.719 | -0.017 | -0.091 | 0.387 |
| batter_hits | over | H 0.5 | draftkings | 21631 | 0.710 | -0.028 | -0.076 | 0.417 |
| batter_hits | under | H 0.5 | draftkings | 21609 | 0.710 | -0.028 | -0.047 | 0.353 |
| batter_hits | over | H 2.5+ | fanduel | 12112 | 0.707 | -0.034 | -0.263 | 0.307 |
| batter_home_runs | over | HR 1.5+ | fanduel | 24424 | 0.270 | 0.057 | -0.436 | 0.341 |
| batter_home_runs | over | HR 0.5 | fanduel | 24454 | 0.269 | 0.057 | -0.212 | 0.366 |

## Direct Player-Game Repair Heads

| Stat | Enabled | Positive Folds | Base MAE | Blended MAE | MAE Gain | Any-Event Brier Gain | Alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| batter_hits | False | 0/0 | - | - | - | - | 0.000 |
| batter_total_bases | False | 0/0 | - | - | - | - | 0.000 |
| batter_home_runs | False | 0/0 | - | - | - | - | 0.000 |
