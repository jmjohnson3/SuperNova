# FanDuel One-Sided Prop Diagnostic

Generated UTC: 2026-08-29T12:56:09Z
Lookback: 30 days
Raw FanDuel hitter offer groups: 1082
Dominant unresolved cause: raw_api_one_sided
Extraction decision: **raw_feed_missing_opposite_side**
Training action: keep FanDuel one-sided hitter props display/research only; exclude from residual training, CLV proof, bucket promotion, and real-money ranking

## Root Cause Trace

| Cause | Offer groups | Share |
|---|---:|---:|
| raw_api_one_sided | 1082 | 100.0% |

## By Stat

| Stat | Raw one-sided | Parser loss | Normalizer loss | Not replayed | Not selected | Clean pair |
|---|---:|---:|---:|---:|---:|---:|
| batter_hits | 324 | 0 | 0 | 0 | 0 | 0 |
| batter_home_runs | 273 | 0 | 0 | 0 | 0 | 0 |
| batter_total_bases | 485 | 0 | 0 | 0 | 0 | 0 |

## Raw Market Surfaces

| Odds API market key | Groups |
|---|---:|
| batter_total_bases_alternate | 485 |
| batter_hits_alternate | 324 |
| batter_home_runs_alternate | 273 |

## Recent Failure Examples

| Date | Player | Stat | Line | API market | Raw O/U | Parsed under | Normalized under | Training rows | Cause |
|---|---|---|---:|---|---|---:|---:|---:|---|
| 2026-07-30 | Alejandro Osuna | batter_hits | 1.5 | batter_hits_alternate | 320.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Wyatt Langford | batter_hits | 1.5 | batter_hits_alternate | 650.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Ben Williamson | batter_hits | 1.5 | batter_hits_alternate | 850.0/None | None | None | 1 | raw_api_one_sided |
| 2026-07-30 | Hunter Feduccia | batter_hits | 1.5 | batter_hits_alternate | 2000.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Yandy Diaz | batter_hits | 2.5 | batter_hits_alternate | 1900.0/None | None | None | 1 | raw_api_one_sided |
| 2026-07-30 | Nicky Lopez | batter_hits | 0.5 | batter_hits_alternate | 320.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Austin Wynns | batter_hits | 0.5 | batter_hits_alternate | 350.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Cedric Mullins | batter_home_runs | 1.5 | batter_home_runs_alternate | 35000.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Austin Wynns | batter_home_runs | 1.5 | batter_home_runs_alternate | 40000.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Jonny Deluca | batter_home_runs | 1.5 | batter_home_runs_alternate | 30000.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Austin Wynns | batter_home_runs | 0.5 | batter_home_runs_alternate | 2700.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Wyatt Langford | batter_home_runs | 0.5 | batter_home_runs_alternate | 3300.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Hunter Feduccia | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Jonathan Aranda | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Ezequiel Duran | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Cameron Cauley | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Taylor Walls | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Nicky Lopez | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Yandy Diaz | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Alejandro Osuna | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Junior Caminero | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Jake Burger | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Brandon Nimmo | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Ben Williamson | batter_home_runs | 0.5 | batter_home_runs_alternate | 7500.0/None | None | None | 1 | raw_api_one_sided |
| 2026-07-30 | Josh Rojas | batter_hits | 0.5 | batter_hits_alternate | 320.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Josh Rojas | batter_home_runs | 0.5 | batter_home_runs_alternate | 3000.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Ivan Herrera | batter_hits | 0.5 | batter_hits_alternate | 240.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Dansby Swanson | batter_hits | 0.5 | batter_hits_alternate | 270.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Seiya Suzuki | batter_hits | 0.5 | batter_hits_alternate | 280.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Michael Conforto | batter_hits | 0.5 | batter_hits_alternate | 270.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Lars Nootbaar | batter_hits | 0.5 | batter_hits_alternate | 320.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Nico Hoerner | batter_hits | 3.5 | batter_hits_alternate | 2200.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Bryan Torres | batter_hits | 3.5 | batter_hits_alternate | 10000.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Pete Crow-Armstrong | batter_hits | 3.5 | batter_hits_alternate | 10000.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Michael Busch | batter_hits | 3.5 | batter_hits_alternate | 20000.0/None | None | None | 0 | raw_api_one_sided |
| 2026-07-30 | Jordan Walker | batter_hits | 1.5 | batter_hits_alternate | 185.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | JJ Wetherholt | batter_hits | 1.5 | batter_hits_alternate | 185.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Alec Burleson | batter_hits | 1.5 | batter_hits_alternate | 240.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Masyn Winn | batter_hits | 1.5 | batter_hits_alternate | 280.0/None | None | None | 2 | raw_api_one_sided |
| 2026-07-30 | Carson Kelly | batter_hits | 1.5 | batter_hits_alternate | 280.0/None | None | None | 2 | raw_api_one_sided |
