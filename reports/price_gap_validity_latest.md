# Price gaps: is Pinnacle right when a book disagrees?

Built 2026-11-15 02:31 UTC. Research only; stored prices, no credits. Player props and game lines (spread, total, moneyline) with Pinnacle at the identical line in the same look; one row per game, player or market, and book, the first look 15 to 45 minutes before the start.

- NBA game lines: 156 games, 1136 prices, 1100 settled, 2026-10-23 to 2026-11-15.
- NBA player props: 139 games, 14641 prices, 14347 settled, 2026-10-24 to 2026-11-15.
- NCAAF game lines: 229 games, 1834 prices, 1379 settled, 2026-10-23 to 2026-11-15.
- NFL player props: 42 games, 2981 prices, 2909 settled, 2026-10-22 to 2026-11-13.

## Whose price is right

Weight 1 means that where the two differ Pinnacle's price is the right one; 0 means the book's own price is. Intervals resample games.

| Prices | Settled | Games | Average difference | Weight on Pinnacle (95%) |
|---|---:|---:|---:|---:|
| All | 19730 | 500 | 1.4% | +0.51 (-0.09 to +1.12) |
| Game lines | 2474 | 323 | 0.7% | -0.18 (-3.00 to +3.21) |
| Player props | 17256 | 177 | 1.5% | +0.54 (-0.10 to +1.16) |
| NBA | 15447 | 286 | 1.5% | +0.58 (-0.13 to +1.19) |
| NCAAF | 1374 | 172 | 0.7% | -2.15 (-3.00 to +2.04) |
| NFL | 2909 | 42 | 1.3% | +0.46 (-1.14 to +2.06) |
| betmgm | 4498 | 497 | 1.7% | +0.55 (-0.10 to +1.21) |
| draftkings | 5055 | 500 | 1.5% | +0.41 (-0.32 to +1.16) |
| espnbet | 4881 | 496 | 1.2% | +0.46 (-0.49 to +1.39) |
| fanduel | 5296 | 500 | 1.2% | +0.66 (-0.25 to +1.56) |
| NBA ast | 2570 | 135 | 1.5% | +0.64 (-0.97 to +2.39) |
| NBA fg3m | 3571 | 135 | 1.5% | -0.90 (-2.29 to +0.38) |
| NBA moneyline | 604 | 151 | 0.6% | +3.32 (-3.00 to +5.00) |
| NBA pts | 4125 | 135 | 1.6% | +1.44 (+0.38 to +2.47) |
| NBA reb | 4081 | 135 | 1.5% | +0.72 (-0.38 to +1.84) |
| NBA spread | 272 | 74 | 0.7% | -2.50 (-3.00 to +5.00) |
| NBA total | 224 | 75 | 0.7% | +5.00 (-3.00 to +5.00) |
| NCAAF moneyline | 682 | 171 | 0.7% | -3.00 (-3.00 to -0.04) |
| NCAAF spread | 373 | 143 | 0.6% | +5.00 (+1.28 to +5.00) |
| NCAAF total | 319 | 97 | 0.7% | -3.00 (-3.00 to +3.55) |
| NFL receiving_yards | 798 | 42 | 1.1% | -0.61 (-3.00 to +2.66) |
| NFL receptions | 1719 | 42 | 1.5% | +1.04 (-0.93 to +2.85) |
| NFL rushing_yards | 392 | 42 | 1.2% | -1.14 (-3.00 to +2.59) |
| Gap of 3%+ only | 316 | 141 | 5.1% | +1.13 (-0.02 to +2.37) |
| Gap of 3%+, player props | 291 | 123 | 5.4% | +1.24 (+0.05 to +2.49) |

## The better side at the book, by size of the gap: All

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 17890 | -3.8% | 51.1% | -5.3% (-7.5% to -3.2%) |
| 0% to 1.5% | 1087 | +0.6% | 50.0% | -2.6% (-9.3% to +4.2%) |
| 1.5% to 3% | 442 | +2.1% | 43.9% | -14.1% (-22.8% to -4.7%) |
| 3% to 5% | 212 | +3.8% | 50.5% | +0.1% (-15.8% to +14.9%) |
| 5% or more | 104 | +7.1% | 46.2% | -4.2% (-26.6% to +17.9%) |

## The better side at the book, by size of the gap: Game lines

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 2388 | -3.1% | 53.4% | -6.0% (-11.5% to -0.7%) |
| 0% to 1.5% | 56 | +0.6% | 39.3% | -8.2% (-46.9% to +39.4%) |
| 1.5% to 3% | 10 | +2.1% | 10.0% | -80.2% |
| 3% to 5% | 10 | +3.9% | 0.0% | -100.0% |
| 5% or more | 15 | +8.2% | 0.0% | -100.0% (-100.0% to -100.0%) |

## The better side at the book, by size of the gap: Player props

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 15502 | -3.9% | 50.7% | -5.2% (-7.3% to -3.3%) |
| 0% to 1.5% | 1031 | +0.6% | 50.6% | -2.3% (-8.7% to +4.6%) |
| 1.5% to 3% | 432 | +2.1% | 44.7% | -12.5% (-21.9% to -3.1%) |
| 3% to 5% | 202 | +3.8% | 53.0% | +5.1% (-9.9% to +21.7%) |
| 5% or more | 89 | +6.9% | 53.9% | +12.0% (-8.4% to +35.7%) |

## The better side at the book, by size of the gap: NBA

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 13898 | -3.8% | 51.2% | -4.9% (-7.3% to -2.7%) |
| 0% to 1.5% | 905 | +0.6% | 49.8% | -2.9% (-10.5% to +4.7%) |
| 1.5% to 3% | 370 | +2.1% | 45.1% | -11.5% (-21.8% to -1.0%) |
| 3% to 5% | 184 | +3.8% | 51.1% | +1.4% (-15.2% to +18.0%) |
| 5% or more | 90 | +6.9% | 51.1% | +4.6% (-15.6% to +29.9%) |

## The better side at the book, by size of the gap: NCAAF

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 1328 | -3.1% | 52.3% | -8.5% (-15.3% to -1.9%) |
| 0% to 1.5% | 29 | +0.7% | 37.9% | -26.3% (-61.5% to +18.7%) |
| 1.5% to 3% | 7 | +1.9% | 14.3% | -71.7% |
| 3% to 5% | 6 | +3.8% | 0.0% | -100.0% |
| 5% or more | 9 | +9.2% | 0.0% | -100.0% |

## The better side at the book, by size of the gap: NFL

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 2664 | -4.0% | 50.0% | -5.9% (-10.5% to -1.3%) |
| 0% to 1.5% | 153 | +0.6% | 53.6% | +3.7% (-8.7% to +18.1%) |
| 1.5% to 3% | 65 | +2.2% | 40.0% | -22.2% (-50.5% to +5.1%) |
| 3% to 5% | 22 | +3.7% | 59.1% | +16.8% (-33.6% to +50.4%) |
| 5% or more | 5 | +6.3% | 40.0% | +9.8% |

## Gaps of 3% or more

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| All | 316 | +4.9% | 49.1% | -1.3% (-14.5% to +12.3%) |
| Game lines | 25 | +6.5% | 0.0% | -100.0% (-100.0% to -100.0%) |
| Player props | 291 | +4.8% | 53.3% | +7.2% (-6.6% to +20.0%) |
| betmgm | 113 | +4.8% | 54.9% | +6.9% (-10.4% to +24.3%) |
| draftkings | 110 | +4.7% | 50.9% | +1.0% (-17.6% to +20.3%) |
| espnbet | 24 | +4.7% | 41.7% | -17.2% (-59.2% to +23.3%) |
| fanduel | 69 | +5.5% | 39.1% | -12.8% (-39.4% to +15.5%) |
| NBA ast | 20 | +5.3% | 45.0% | -11.9% (-57.9% to +32.9%) |
| NBA fg3m | 74 | +4.9% | 44.6% | -1.5% (-33.1% to +30.8%) |
| NBA moneyline | 10 | +5.6% | 0.0% | -100.0% |
| NBA pts | 100 | +4.7% | 55.0% | +5.0% (-16.7% to +27.9%) |
| NBA reb | 70 | +4.8% | 61.4% | +21.8% (+0.6% to +45.4%) |
| NCAAF moneyline | 15 | +7.1% | 0.0% | -100.0% (-100.0% to -100.0%) |
| NFL receiving_yards | 4 | +4.1% | 50.0% | -6.1% |
| NFL receptions | 21 | +4.3% | 52.4% | +12.9% (-32.7% to +50.3%) |
| NFL rushing_yards | 2 | +4.1% | 100.0% | +86.2% |
| Over side | 108 | +4.6% | 45.4% | -5.9% (-29.0% to +16.4%) |
| Under side | 208 | +5.0% | 51.0% | +1.1% (-14.2% to +17.0%) |

## Gaps of 3% or more by minutes before the start (each window's first look)

| Window | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| 45 to 35 minutes | 266 | +4.6% | 49.6% | +0.4% (-13.1% to +14.5%) |
| 35 to 25 | 279 | +4.9% | 52.0% | +3.9% (-8.8% to +17.9%) |
| 25 to 15 | 310 | +4.7% | 49.0% | -0.1% (-13.1% to +14.0%) |
| 15 to 5 | 281 | +5.0% | 45.6% | -7.5% (-22.4% to +6.5%) |

## Progress, run by run (about 1,500 prop bets on gaps of 3%+ are needed)

| Run | Prop bets on 3%+ gaps | Value claimed | Return on results (95%) | Weight on Pinnacle (95%) | Game-line bets | Their return |
|---|---:|---:|---:|---:|---:|---:|
| 2026-11-13 | 260 | +4.8% | +7.8% (-7.3% to +21.2%) | +1.29 (-0.00 to +2.52) | 0 | - |
| 2026-11-15 | 291 | +4.8% | +7.2% (-6.6% to +20.0%) | +1.24 (+0.05 to +2.49) | 25 | -100.0% |

Value claimed is the bet's expected return if Pinnacle's no-vig price is the true chance. Every price is counted, whether or not a scanner flagged it or the price could really be bet. Game lines were added on 2026-11-14 (NBA and NCAAF; the spread and moneyline are counted from the home side, so each game's two sides are one row). A college game whose teams cannot be matched to a final score stays unsettled. Spreads and totals almost never differ from Pinnacle by 3% at the same line; the game-line gaps are moneyline longshots, where sharing the margin evenly between the two sides overstates the longshot's chance.
