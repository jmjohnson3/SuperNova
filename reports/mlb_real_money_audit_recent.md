# MLB Real-Money Audit (2026-04-04 to 2026-06-02)

Scope: saved prediction rows re-filtered through current real-money gates.
Limitation: this is not a locked-ledger walk-forward; reruns can overwrite historical predictions.

## Overall

- Pre-cap qualifying bets: 131 (78-53, ROI 16.9%)
- Selected after daily cap: 119 (73-46, ROI 20.9%)
- Bet days: 44/60
- No-bet days: 16/60
- Avg selected bets/day: 1.98
- Global cap: 2.00% per day
- Prop thresholds: K 999, H 0.75, TB 999, HR over 999

## Recent Daily Volume

| Date | Bets | W-L | ROI |
| --- | --- | --- | --- |
| 2026-06-02 | 0 | 0-0 | n/a |
| 2026-06-01 | 0 | 0-0 | n/a |
| 2026-05-31 | 4 | 3-1 | 46.4% |
| 2026-05-30 | 0 | 0-0 | n/a |
| 2026-05-29 | 2 | 1-1 | -19.9% |
| 2026-05-28 | 1 | 0-1 | -100.0% |
| 2026-05-27 | 1 | 1-0 | 58.1% |
| 2026-05-26 | 1 | 1-0 | 160.0% |
| 2026-05-25 | 1 | 1-0 | 78.1% |
| 2026-05-24 | 2 | 2-0 | 100.1% |
| 2026-05-23 | 4 | 2-2 | -14.8% |
| 2026-05-22 | 1 | 1-0 | 83.3% |
| 2026-05-21 | 1 | 1-0 | 158.0% |
| 2026-05-20 | 2 | 0-2 | -100.0% |
| 2026-05-19 | 0 | 0-0 | n/a |
| 2026-05-18 | 2 | 1-1 | -10.0% |
| 2026-05-17 | 0 | 0-0 | n/a |
| 2026-05-16 | 2 | 1-1 | 1.0% |
| 2026-05-15 | 4 | 1-3 | -50.5% |
| 2026-05-14 | 2 | 0-2 | -100.0% |
| 2026-05-13 | 3 | 2-1 | 9.4% |

## Selected Buckets

| Source | Market | Side | Bets | W-L | Win% | ROI | Avg CLV | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| game | run_line | home | 55 | 36-19 | 65.5% | 25.3% | 1.20 | OK |
| game | total | over | 46 | 25-21 | 54.3% | 2.8% | 0.79 | OK |
| game | run_line | away | 18 | 12-6 | 66.7% | 53.8% | 2.06 | OK |

## Price Buckets

| Source | Market | Side | Price | Bets | W-L | ROI | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| game | run_line | home | fair_lay | 5 | 2-3 | -24.8% | FAIL |
| game | total | over | fair_lay | 36 | 20-16 | 4.2% | OK |
| game | run_line | home | lay_150_180 | 30 | 22-8 | 17.3% | OK |
| game | run_line | home | plus_money | 18 | 11-7 | 57.1% | OK |
| game | run_line | away | plus_money | 14 | 9-5 | 58.6% | OK |
| game | total | over | plus_money | 8 | 4-4 | 0.3% | OK |

## Top Rejection Reasons

| Source | Market | Reason | Rows |
| --- | --- | --- | --- |
| prop | batter_hits | no_model_edge; missing_price; missing_ev | 8861 |
| prop | batter_home_runs | no_model_edge; unbookable_under; missing_price; missing_ev; zero_kelly | 5973 |
| prop | batter_total_bases | no_model_edge; missing_price; missing_ev | 4027 |
| prop | batter_total_bases | no_model_edge; missing_price; missing_ev; zero_kelly | 3082 |
| prop | batter_total_bases | no_model_edge; weak_over_bucket; missing_price; missing_ev | 2140 |
| prop | batter_home_runs | unbookable_under; missing_price; missing_ev; zero_kelly | 2055 |
| prop | batter_hits | no_model_edge; unbookable_under; missing_price; missing_ev | 1292 |
| prop | batter_home_runs | no_model_edge; unbookable_under; missing_price; missing_ev | 1154 |
| prop | batter_total_bases | no_model_edge; weak_over_bucket; missing_price; missing_ev; zero_kelly | 1105 |
| prop | batter_walks | no_model_edge; missing_price; missing_ev; zero_kelly | 1032 |
| prop | pitcher_strikeouts | no_model_edge; missing_price; missing_ev | 973 |
| prop | batter_home_runs | unbookable_under; missing_price; missing_ev | 735 |
| prop | batter_walks | no_model_edge; missing_price; missing_ev | 525 |
| prop | batter_hits | no_model_edge; unbookable_under; missing_price; missing_ev; zero_kelly | 521 |
| prop | batter_hits | no_model_edge; missing_price; missing_ev; zero_kelly | 442 |
| prop | pitcher_strikeouts | no_model_edge; missing_price; missing_ev; zero_kelly | 332 |
| prop | batter_hits | missing_price; missing_ev | 87 |
| game | run_line | away_dog_lay_price | 52 |
| game | run_line | heavy_juice; away_dog_lay_price | 46 |
| game | run_line | missing_price | 43 |
| game | run_line | heavy_juice | 24 |
| game | total | total_under_disabled | 23 |
| game | run_line | non_standard_run_line | 17 |
| game | total | missing_price | 16 |
| prop | batter_home_runs | no_model_edge; hr_longshot_variance; missing_price; missing_ev | 7 |
