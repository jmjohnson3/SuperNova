# NBA prop challenger: game line and minutes model

Built 2026-11-12 04:41 UTC. Walk-forward from 2023-11-01, monthly refits on earlier games only, scored on DraftKings lines taken before tip. Research only; nothing here feeds a live forecast.

## Average miss on lines (lower is better)

| Statistic | Lines | DraftKings line | Current model | + game line | + both | If minutes were known |
|---|---:|---:|---:|---:|---:|---:|
| Points | 22148 | 5.069 | 5.142 | 5.141 | 5.135 | 4.397 |
| Rebounds | 21067 | 2.060 | 2.060 | 2.060 | 2.058 | 1.863 |
| Assists | 14794 | 1.777 | 1.777 | 1.777 | 1.776 | 1.665 |

## Probability score (log loss, lower is better)

| Statistic | DraftKings no-vig | Current model | + game line | + both | If minutes were known | Current model, honest spread | + game line, honest spread | + both, honest spread | If minutes were known, honest spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Points | 0.6917 | 0.7074 | 0.7073 | 0.7065 | 0.6128 | 0.7070 | 0.7068 | 0.7063 | 0.6130 |
| Rebounds | 0.6885 | 0.7037 | 0.7030 | 0.7035 | 0.6767 | 0.7054 | 0.7057 | 0.7034 | 0.6627 |
| Assists | 0.6856 | 0.7251 | 0.7251 | 0.7237 | 0.7186 | 0.7225 | 0.7222 | 0.7223 | 0.7162 |

## Change against the current model on the same lines (95% interval; negative is better)

| Variant | Statistic | Average miss | Log loss |
|---|---|---:|---:|
| + game line | Points | -0.001 (-0.003 to +0.001) | -0.0002 (-0.0005 to +0.0001) |
| + game line | Rebounds | -0.000 (-0.001 to +0.001) | +0.0002 (-0.0001 to +0.0006) |
| + game line | Assists | +0.000 (-0.001 to +0.001) | -0.0004 (-0.0007 to +0.0000) |
| + both | Points | -0.007 (-0.011 to -0.002) | -0.0007 (-0.0013 to -0.0000) |
| + both | Rebounds | -0.002 (-0.005 to -0.000) | -0.0020 (-0.0032 to -0.0009) |
| + both | Assists | -0.000 (-0.002 to +0.002) | -0.0002 (-0.0010 to +0.0006) |
| If minutes were known | Points | -0.745 (-0.788 to -0.704) | -0.0940 (-0.0996 to -0.0881) |
| If minutes were known | Rebounds | -0.197 (-0.211 to -0.183) | -0.0427 (-0.0483 to -0.0374) |
| If minutes were known | Assists | -0.111 (-0.123 to -0.100) | -0.0064 (-0.0111 to -0.0013) |

## Picks where the model disagrees with DraftKings by more than 4% (honest spread)

| Variant | Statistic | Picks | Hit rate | ROI at DraftKings' price | Model's weight beside the line (se) |
|---|---|---:|---:|---:|---:|
| Current model | Points | 15541 | 51.8% | -2.5% | +0.27 (0.03) |
| Current model | Rebounds | 14364 | 50.5% | -4.2% | +0.37 (0.03) |
| Current model | Assists | 11638 | 50.1% | -4.7% | +0.39 (0.03) |
| + game line | Points | 15543 | 51.9% | -2.4% | +0.27 (0.03) |
| + game line | Rebounds | 14376 | 50.5% | -4.4% | +0.37 (0.03) |
| + game line | Assists | 11626 | 50.0% | -4.9% | +0.39 (0.03) |
| + both | Points | 15602 | 51.7% | -2.8% | +0.28 (0.03) |
| + both | Rebounds | 14214 | 51.1% | -3.4% | +0.40 (0.03) |
| + both | Assists | 11632 | 50.0% | -4.9% | +0.39 (0.03) |
| If minutes were known | Points | 19064 | 67.4% | +26.6% | +0.95 (0.01) |
| If minutes were known | Rebounds | 17894 | 61.3% | +15.5% | +0.92 (0.01) |
| If minutes were known | Assists | 13136 | 55.9% | +5.8% | +0.85 (0.02) |

## Average miss on every player-game (no line needed)

| Statistic | Current model | + game line | + both | If minutes were known |
|---|---:|---:|---:|---:|
| Points | 4.444 | 4.443 | 4.427 | 3.617 |
| Rebounds | 1.841 | 1.841 | 1.835 | 1.576 |
| Assists | 1.307 | 1.307 | 1.304 | 1.192 |

## Minutes model with the game line: average miss in minutes

| Player-games | Rows | Last-3 average | Last-10 average | Minutes model |
|---|---:|---:|---:|---:|
| All | 86198 | 5.01 | 5.31 | 4.53 |
| Starters | 40394 | 4.79 | - | 4.29 |
| Team missing 2+ rotation players | 28397 | 5.50 | - | 4.82 |

Game lines: 100% of the scored lines (81% of all test player-games) have a DraftKings spread and total taken before tip; the rest are left empty. "If minutes were known" uses the minutes actually played and cannot be had before a game.
