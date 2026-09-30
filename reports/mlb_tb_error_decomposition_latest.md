# MLB TB Error Decomposition

Generated UTC: 2026-07-30T16:00:50Z
Status: **ready**
Lookback days: 45
Player-game rows: 6472

## Component Summary

- TB MAE: 1.398
- TB bias: 0.079
- PA MAE: 0.721
- Avg abs PA component TB error: 0.293
- Avg abs hit probability component TB error: 0.684
- Avg abs single/double/triple/HR mix component TB error: 0.901
- Avg abs 4+ HR-tail component error: 1.109
- HR-driven 4+ TB rows: 794
- Non-HR 4+ TB rows: 170

Dominant miss component counts: 4plus_tb_tail_error=2805, hit_probability_error=1763, pa_error=385, single_double_triple_hr_mix_error=1519

## TB Line Pricing

| Side | Book | Line | Surface | Pair | Rows | Win | ROI | Brier | CLV Rows | CLV Beat | Avg CLV |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| over | fanduel | TB 2.5+ | alt_tail | one_sided | 37384 | 9.3% | -0.420 | 0.084 | 34899 | 36.7% | 0.068 |
| over | fanduel | TB 2.5+ | common | one_sided | 17396 | 18.9% | -0.280 | 0.159 | 16210 | 39.4% | 0.061 |
| over | fanduel | TB 1.5 | common | cross_book | 12889 | 40.4% | -0.132 | 0.244 | 12394 | 38.1% | 0.086 |
| over | draftkings | TB 1.5 | common | same_book | 10095 | 41.5% | -0.093 | 0.243 | 8832 | 46.1% | 0.338 |
| under | draftkings | TB 1.5 | common | same_book | 10095 | 58.5% | -0.042 | 0.243 | 8832 | 33.0% | -0.328 |
| over | fanduel | TB 1.5 | common | one_sided | 6720 | 27.3% | -0.261 | 0.207 | 5970 | 36.1% | -0.161 |
| over | fanduel | TB 2.5+ | common | cross_book | 2213 | 39.8% | 0.458 | 0.250 | 2154 | 40.9% | 0.084 |
| over | fanduel | TB 2.5+ | alt_tail | cross_book | 1771 | 41.3% | 2.575 | 0.320 | 1736 | 43.1% | 0.213 |
| under | draftkings | TB 2.5+ | common | same_book | 20 | 60.0% | -0.020 | 0.238 | 14 | 35.7% | -0.323 |
| over | draftkings | TB 2.5+ | common | same_book | 18 | 44.4% | -0.064 | 0.225 | 14 | 64.3% | 0.299 |

## Worst Player-Game TB Misses

| Date | Player | Slot | PA Err | Hit Err | Mix Err | Tail Err | TB Pred | TB Actual | Dominant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-20 | Kyle Schwarber | 2 | -0.500 | -3.041 | -8.240 | -10.784 | 1.623 | 13.000 | 4plus_tb_tail_error |
| 2026-06-27 | Hunter Goodman | 4 | -1.611 | -1.694 | -8.385 | -10.941 | 1.500 | 12.000 | 4plus_tb_tail_error |
| 2026-07-01 | Dansby Swanson | 9 | -1.874 | -1.758 | -8.245 | -11.110 | 1.532 | 12.000 | 4plus_tb_tail_error |
| 2026-06-15 | Colt Keith | 6 | -1.900 | -1.321 | -8.346 | -11.377 | 1.695 | 12.000 | 4plus_tb_tail_error |
| 2026-06-17 | Kyle Stowers | 3 | -1.000 | -2.770 | -6.226 | -6.824 | 1.758 | 11.000 | 4plus_tb_tail_error |
| 2026-07-19 | Hunter Goodman | 3 | -0.026 | -1.746 | -7.453 | -11.112 | 2.792 | 12.000 | 4plus_tb_tail_error |
| 2026-07-10 | Tristan Peters | 9 | -0.500 | -3.162 | -5.742 | -3.363 | 0.991 | 10.000 | single_double_triple_hr_mix_error |
| 2026-06-17 | Sterlin Thompson | 8 | -1.100 | -2.022 | -6.672 | -7.776 | 1.037 | 10.000 | 4plus_tb_tail_error |
| 2026-06-19 | Ty France | 7 | -0.700 | -2.203 | -6.502 | -7.329 | 1.156 | 10.000 | 4plus_tb_tail_error |
| 2026-07-20 | Petey Halpin | 9 | -2.260 | -1.970 | -6.334 | -7.709 | 1.231 | 10.000 | 4plus_tb_tail_error |
| 2026-07-03 | Jake McCarthy | 1 | -1.348 | -3.214 | -5.266 | -7.572 | 1.308 | 10.000 | 4plus_tb_tail_error |
| 2026-07-05 | Heriberto Hernández | 4 | -2.300 | -2.065 | -6.265 | -7.114 | 1.312 | 10.000 | 4plus_tb_tail_error |
| 2026-07-04 | Josh Bell | 5 | -1.400 | -2.242 | -6.207 | -6.942 | 1.338 | 10.000 | 4plus_tb_tail_error |
| 2026-07-03 | Daylen Lile | 6 | -0.251 | -2.391 | -6.163 | -7.351 | 1.408 | 10.000 | 4plus_tb_tail_error |
| 2026-06-15 | Pete Crow-Armstrong | 1 | -0.600 | -2.834 | -5.593 | -3.308 | 1.433 | 10.000 | single_double_triple_hr_mix_error |
| 2026-06-16 | Bryan Reynolds | 3 | -0.800 | -3.029 | -5.315 | -7.377 | 1.500 | 10.000 | 4plus_tb_tail_error |
| 2026-06-20 | Bryce Harper | 3 | -1.000 | -2.946 | -5.244 | -2.709 | 1.600 | 10.000 | single_double_triple_hr_mix_error |
| 2026-07-03 | Kyle Stowers | 3 | -0.999 | -3.284 | -4.909 | -6.789 | 1.664 | 10.000 | 4plus_tb_tail_error |
| 2026-06-20 | Travis Bazzana | 1 | 0.300 | -3.103 | -5.285 | -7.393 | 1.679 | 10.000 | 4plus_tb_tail_error |
| 2026-07-19 | Drake Baldwin | 1 | -0.900 | -3.326 | -4.657 | -7.072 | 2.716 | 11.000 | 4plus_tb_tail_error |
| 2026-06-28 | Luis García Jr. | 2 | -1.141 | -1.697 | -6.182 | -6.780 | 1.824 | 10.000 | 4plus_tb_tail_error |
| 2026-07-20 | Mookie Betts | 4 | -0.996 | -2.238 | -5.315 | -7.447 | 2.096 | 10.000 | 4plus_tb_tail_error |
| 2026-07-20 | Trea Turner | 1 | -1.081 | -1.614 | -5.967 | -7.092 | 2.119 | 10.000 | 4plus_tb_tail_error |
| 2026-06-29 | Cole Young | 8 | 0.858 | -2.371 | -5.402 | -7.225 | 1.407 | 9.000 | 4plus_tb_tail_error |
| 2026-06-19 | Jeremy Peña | 1 | -0.500 | -1.904 | -5.535 | -7.323 | 1.451 | 9.000 | 4plus_tb_tail_error |
| 2026-07-11 | Esmerlyn Valdez | 2 | 0.600 | -2.177 | -5.460 | -7.070 | 1.486 | 9.000 | 4plus_tb_tail_error |
| 2026-06-15 | Nick Kurtz | 1 | -0.300 | -2.104 | -5.342 | -7.080 | 1.500 | 9.000 | 4plus_tb_tail_error |
| 2026-06-30 | Dansby Swanson | 9 | -0.707 | -2.045 | -5.286 | -7.104 | 1.500 | 9.000 | 4plus_tb_tail_error |
| 2026-07-19 | Tyler Stephenson | 6 | 0.005 | -1.640 | -5.780 | -7.168 | 2.582 | 10.000 | 4plus_tb_tail_error |
| 2026-07-04 | Yordan Alvarez | 2 | -0.600 | -2.329 | -4.972 | -6.812 | 1.618 | 9.000 | 4plus_tb_tail_error |
| 2026-06-20 | Ozzie Albies | 4 | 0.000 | -2.000 | -5.248 | -7.008 | 1.752 | 9.000 | 4plus_tb_tail_error |
| 2026-07-19 | Jackson Merrill | 2 | -1.919 | -1.624 | -4.791 | -7.038 | 2.825 | 10.000 | 4plus_tb_tail_error |
| 2026-07-19 | Austin Riley | 7 | -0.172 | -2.056 | -5.057 | -7.206 | 1.846 | 9.000 | 4plus_tb_tail_error |
| 2026-06-23 | Jac Caglianone | 3 | -0.600 | -1.893 | -5.099 | -6.796 | 1.875 | 9.000 | 4plus_tb_tail_error |
| 2026-07-06 | José Caballero | 9 | 0.500 | -1.694 | -5.478 | -7.528 | 0.879 | 8.000 | 4plus_tb_tail_error |
| 2026-07-07 | Nick Loftin | 7 | -2.600 | -2.479 | -4.401 | -3.628 | 0.894 | 8.000 | single_double_triple_hr_mix_error |
| 2026-07-02 | Dalton Rushing | 9 | -1.677 | -3.345 | -3.487 | -3.312 | 0.948 | 8.000 | single_double_triple_hr_mix_error |
| 2026-07-11 | Ezequiel Duran | 6 | 0.000 | -1.209 | -5.817 | -7.216 | 0.974 | 8.000 | 4plus_tb_tail_error |
| 2026-06-21 | JJ Wetherholt | 1 | -1.300 | -1.665 | -5.059 | -7.260 | 1.986 | 9.000 | 4plus_tb_tail_error |
| 2026-06-15 | Eugenio Suárez | 5 | 0.000 | -1.411 | -5.412 | -7.269 | 1.177 | 8.000 | 4plus_tb_tail_error |
