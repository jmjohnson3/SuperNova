# A fitted fair price from every book's prop price

Built 2026-11-17 01:47 UTC. Research only; stored prices, no credits. Every price is assumed bettable as stored.

- 7767 props (7361 settled) over 22 game days, 2026-10-22 to 2026-11-16; books: betmgm, draftkings, espnbet, fanatics, fanduel, pinnacle.
- Forecast walk-forward: 5432 props on 15 days, each from a fit on earlier days only.

## Is the fitted price more accurate than the prices it is built from?

Log loss on results, on the props where both prices exist; a difference below zero means the fitted price is better. Intervals resample games.

| Compared with | Props | Fitted | Other | Difference (95%) |
|---|---:|---:|---:|---:|
| Average of the books | 5037 | 0.6887 | 0.6883 | +0.0004 (-0.0010 to +0.0018) |
| betmgm | 3783 | 0.6868 | 0.6879 | -0.0011 (-0.0030 to +0.0007) |
| draftkings | 4536 | 0.6889 | 0.6893 | -0.0003 (-0.0020 to +0.0012) |
| espnbet | 4503 | 0.6891 | 0.6883 | +0.0008 (-0.0007 to +0.0023) |
| fanatics | 411 | 0.6943 | 0.6926 | +0.0016 (-0.0038 to +0.0066) |
| fanduel | 4490 | 0.6891 | 0.6885 | +0.0005 (-0.0010 to +0.0019) |
| pinnacle | 3478 | 0.6880 | 0.6884 | -0.0004 (-0.0020 to +0.0012) |

Against the plain average, by statistic:

| Statistic | Props | Fitted | Average | Difference (95%) |
|---|---:|---:|---:|---:|
| nba ast | 855 | 0.6853 | 0.6843 | +0.0010 (-0.0030 to +0.0047) |
| nba fg3m | 961 | 0.6907 | 0.6881 | +0.0026 (+0.0004 to +0.0050) |
| nba pts | 1212 | 0.6915 | 0.6916 | -0.0001 (-0.0020 to +0.0016) |
| nba reb | 1151 | 0.6900 | 0.6896 | +0.0005 (-0.0011 to +0.0021) |
| nfl receiving_yards | 261 | 0.6919 | 0.6939 | -0.0020 (-0.0092 to +0.0056) |
| nfl receptions | 468 | 0.6787 | 0.6807 | -0.0020 (-0.0090 to +0.0054) |
| nfl rushing_yards | 129 | 0.6891 | 0.6914 | -0.0022 (-0.0143 to +0.0100) |

Against the plain average, by how hard the fit is pulled toward it (higher = closer to the average):

| Pull | Props | Fitted | Average | Difference (95%) |
|---|---:|---:|---:|---:|
| 1.0 | 5037 | 0.6891 | 0.6883 | +0.0008 (-0.0008 to +0.0023) |
| 5.0 | 5037 | 0.6887 | 0.6883 | +0.0004 (-0.0010 to +0.0018) |
| 25.0 | 5037 | 0.6883 | 0.6883 | -0.0000 (-0.0010 to +0.0009) |

## What the fit learned (all days)

1 on the average and 0 elsewhere is the plain average. A positive number on a book means its price deserves more than an equal share.

| Term | Value |
|---|---:|
| average of the books | +1.08 |
| betmgm | -0.00 |
| draftkings | -0.13 |
| espnbet | +0.10 |
| fanatics | +0.01 |
| fanduel | +0.09 |
| pinnacle | -0.06 |
| nba ast (toward the over) | -0.07 |
| nba fg3m (toward the over) | +0.04 |
| nba pts (toward the over) | -0.06 |
| nba reb (toward the over) | -0.05 |
| nfl receiving_yards (toward the over) | -0.13 |
| nfl receptions (toward the over) | -0.16 |
| nfl rushing_yards (toward the over) | -0.14 |

## Bets at 3%+ above: Fitted model

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| FanDuel and DraftKings | 425 | +4.2% | 52.2% | -5.1% (-13.8% to +4.8%) |
| All four books | 659 | +4.3% | 51.6% | -7.2% (-15.2% to +1.0%) |
| NBA ast | 245 | +4.4% | 48.2% | -15.5% (-29.8% to +0.9%) |
| NBA fg3m | 37 | +4.6% | 45.9% | -19.2% (-46.5% to +16.1%) |
| NBA pts | 67 | +4.0% | 41.8% | -18.9% (-44.5% to +5.0%) |
| NBA reb | 15 | +4.5% | 73.3% | +39.4% (-4.5% to +79.9%) |
| NFL receiving_yards | 68 | +3.7% | 39.7% | -24.1% (-44.9% to -2.9%) |
| NFL receptions | 137 | +4.3% | 62.8% | +8.3% (-12.3% to +29.7%) |
| NFL rushing_yards | 90 | +4.3% | 58.9% | +10.8% (-7.6% to +29.2%) |
| 5%+ at all four | 141 | +6.0% | 52.5% | -6.0% (-21.7% to +9.5%) |

## Bets at 3%+ above: Pinnacle's price

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| FanDuel and DraftKings | 97 | +4.5% | 53.6% | +10.7% (-10.9% to +34.1%) |
| All four books | 157 | +4.6% | 53.5% | +9.3% (-7.9% to +29.8%) |
| NBA ast | 13 | +5.0% | 46.2% | -8.2% (-62.6% to +63.0%) |
| NBA fg3m | 41 | +4.7% | 56.1% | +24.1% (-11.6% to +67.2%) |
| NBA pts | 43 | +4.2% | 53.5% | +3.3% (-31.8% to +37.4%) |
| NBA reb | 39 | +4.7% | 61.5% | +22.6% (-6.2% to +53.7%) |
| NFL receiving_yards | 3 | +3.8% | 66.7% | +25.7% |
| NFL receptions | 17 | +4.4% | 29.4% | -35.4% (-73.0% to +8.6%) |
| NFL rushing_yards | 1 | +3.9% | 100.0% | +87.7% |
| 5%+ at all four | 43 | +6.7% | 48.8% | +6.2% (-33.9% to +49.9%) |

## Bets at 3%+ above: Average of the books

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| FanDuel and DraftKings | 4 | +3.4% | 75.0% | +63.5% |
| All four books | 6 | +3.5% | 83.3% | +84.8% |
| NBA fg3m | 4 | +3.4% | 75.0% | +77.2% |
| NBA reb | 2 | +3.7% | 100.0% | +100.0% |

## Bets at 3%+ above: Average of the other books

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| FanDuel and DraftKings | 56 | +3.9% | 44.6% | -1.7% (-30.6% to +27.8%) |
| All four books | 96 | +4.0% | 47.9% | +4.0% (-16.2% to +25.5%) |
| NBA ast | 22 | +4.2% | 45.5% | -2.5% (-47.6% to +40.4%) |
| NBA fg3m | 28 | +4.2% | 35.7% | -14.9% (-55.8% to +29.5%) |
| NBA pts | 15 | +3.9% | 46.7% | -4.7% (-51.8% to +45.3%) |
| NBA reb | 25 | +3.7% | 64.0% | +35.1% (-1.1% to +70.7%) |
| NFL receiving_yards | 3 | +3.2% | 33.3% | -33.3% |
| NFL receptions | 3 | +4.1% | 66.7% | +49.7% |
| 5%+ at all four | 18 | +5.7% | 44.4% | +0.1% (-46.2% to +51.0%) |

## Where the two rules meet (all four books)

| Group | Bets | Value claimed | Hit rate | Return on results (95%) |
|---|---:|---:|---:|---:|
| Both flag the same side | 25 | +4.8% | 60.0% | +18.3% (-18.1% to +53.2%) |
| Model only | 453 | +4.2% | 53.4% | -4.7% (-14.2% to +4.0%) |
| Pinnacle only | 108 | +4.4% | 51.9% | +4.5% (-15.7% to +24.5%) |

Each bet is the better side of one book's price, valued at the rule's fair chance, on the forecast days only. In the last table the Pinnacle-only bets are graded on Pinnacle's side.
