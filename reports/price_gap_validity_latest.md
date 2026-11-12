# Price gaps: is Pinnacle right when a book disagrees?

Built 2026-11-12 23:38 UTC. Research only; stored prices, no credits. Player props with Pinnacle at the identical line in the same look; one row per game, player, statistic and book, the first look 15 to 45 minutes before the start.

- NBA: 124 games, 13128 prices, 12889 settled, 2026-10-24 to 2026-11-12.
- NFL: 41 games, 2895 prices, 2823 settled, 2026-10-22 to 2026-11-10.

## Whose price is right

Weight 1 means that where the two differ Pinnacle's price is the right one; 0 means the book's own price is. Intervals resample games.

| Prices | Settled | Games | Average difference | Weight on Pinnacle (95%) |
|---|---:|---:|---:|---:|
| All | 15712 | 163 | 1.5% | +0.55 (-0.12 to +1.22) |
| NBA | 12889 | 122 | 1.5% | +0.57 (-0.14 to +1.35) |
| NFL | 2823 | 41 | 1.3% | +0.48 (-1.32 to +2.09) |
| betmgm | 3501 | 160 | 1.8% | +0.55 (-0.13 to +1.27) |
| draftkings | 4007 | 163 | 1.7% | +0.52 (-0.23 to +1.31) |
| espnbet | 3902 | 162 | 1.3% | +0.47 (-0.60 to +1.51) |
| fanduel | 4302 | 163 | 1.2% | +0.68 (-0.31 to +1.69) |
| NBA ast | 2298 | 122 | 1.5% | +0.59 (-1.03 to +2.27) |
| NBA fg3m | 3209 | 122 | 1.5% | -0.75 (-2.08 to +0.65) |
| NBA pts | 3704 | 122 | 1.6% | +1.72 (+0.55 to +2.94) |
| NBA reb | 3678 | 122 | 1.6% | +0.40 (-0.83 to +1.53) |
| NFL receiving_yards | 774 | 41 | 1.1% | -0.55 (-3.00 to +2.74) |
| NFL receptions | 1668 | 41 | 1.5% | +1.14 (-0.86 to +3.21) |
| NFL rushing_yards | 381 | 41 | 1.2% | -1.68 (-3.00 to +2.21) |
| Gap of 3%+ only | 260 | 111 | 5.4% | +1.29 (-0.00 to +2.52) |

## The better side at the book, by size of the gap: All

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 14103 | -3.9% | 50.6% | -5.5% (-7.8% to -3.1%) |
| 0% to 1.5% | 946 | +0.6% | 51.4% | -1.0% (-8.0% to +5.7%) |
| 1.5% to 3% | 403 | +2.1% | 44.9% | -12.0% (-22.1% to -2.5%) |
| 3% to 5% | 179 | +3.8% | 52.5% | +4.3% (-12.0% to +20.6%) |
| 5% or more | 81 | +6.8% | 55.6% | +15.5% (-9.3% to +39.5%) |

## The better side at the book, by size of the gap: NBA

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 11524 | -3.9% | 50.7% | -5.3% (-7.6% to -2.8%) |
| 0% to 1.5% | 794 | +0.6% | 51.0% | -1.8% (-10.0% to +5.5%) |
| 1.5% to 3% | 338 | +2.1% | 45.9% | -10.1% (-20.5% to +0.8%) |
| 3% to 5% | 157 | +3.8% | 51.6% | +2.6% (-13.0% to +21.5%) |
| 5% or more | 76 | +6.8% | 56.6% | +15.9% (-7.1% to +39.4%) |

## The better side at the book, by size of the gap: NFL

| Gap against Pinnacle | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Under 0% (no side is good value) | 2579 | -4.0% | 49.9% | -6.2% (-11.2% to -1.3%) |
| 0% to 1.5% | 152 | +0.6% | 53.3% | +3.1% (-9.7% to +17.1%) |
| 1.5% to 3% | 65 | +2.2% | 40.0% | -22.2% (-50.5% to +5.1%) |
| 3% to 5% | 22 | +3.7% | 59.1% | +16.8% (-33.6% to +50.4%) |
| 5% or more | 5 | +6.3% | 40.0% | +9.8% |

## Gaps of 3% or more

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| All | 260 | +4.8% | 53.5% | +7.8% (-7.3% to +21.2%) |
| betmgm | 100 | +4.8% | 58.0% | +13.4% (-3.8% to +33.1%) |
| draftkings | 92 | +4.7% | 52.2% | +3.7% (-15.8% to +24.3%) |
| espnbet | 17 | +4.0% | 52.9% | +6.1% (-42.6% to +54.4%) |
| fanduel | 51 | +5.1% | 47.1% | +4.8% (-25.7% to +40.4%) |
| NBA ast | 20 | +5.3% | 45.0% | -11.9% (-57.9% to +32.9%) |
| NBA fg3m | 66 | +4.8% | 43.9% | -1.8% (-33.4% to +30.3%) |
| NBA pts | 87 | +4.8% | 58.6% | +11.8% (-12.9% to +34.5%) |
| NBA reb | 60 | +4.7% | 58.3% | +15.7% (-7.8% to +41.5%) |
| NFL receiving_yards | 4 | +4.1% | 50.0% | -6.1% |
| NFL receptions | 21 | +4.3% | 52.4% | +12.9% (-32.7% to +50.3%) |
| NFL rushing_yards | 2 | +4.1% | 100.0% | +86.2% |
| Over side | 87 | +4.4% | 50.6% | +4.7% (-17.8% to +29.2%) |
| Under side | 173 | +4.9% | 54.9% | +9.4% (-8.5% to +25.6%) |

## Gaps of 3% or more by minutes before the start (each window's first look)

| Window | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| 45 to 35 minutes | 224 | +4.5% | 52.2% | +5.8% (-8.6% to +20.3%) |
| 35 to 25 | 243 | +4.6% | 55.1% | +10.4% (-4.2% to +24.1%) |
| 25 to 15 | 259 | +4.7% | 52.9% | +7.6% (-6.7% to +22.3%) |
| 15 to 5 | 233 | +4.8% | 47.2% | -4.1% (-18.0% to +10.6%) |

Value claimed is the bet's expected return if Pinnacle's no-vig price is the true chance. Every price is counted, whether or not a scanner flagged it or the price could really be bet. NCAAF and game lines are not included (too few stored).
