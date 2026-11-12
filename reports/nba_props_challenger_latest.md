# NBA prop challenger: game line, minutes model and minutes information

Built 2026-11-12 17:30 UTC. Walk-forward from 2023-11-01, monthly refits on earlier games only, scored on DraftKings lines taken before tip. Research only; nothing here feeds a live forecast.

## Average miss on lines (lower is better)

| Statistic | Lines | DraftKings line | Current model | + game line | + both | + minutes information and model | + minutes information, stints and model | If minutes were known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Points | 22148 | 5.069 | 5.142 | 5.141 | 5.135 | 5.118 | 5.114 | 4.397 |
| Rebounds | 21067 | 2.060 | 2.060 | 2.060 | 2.058 | 2.054 | 2.052 | 1.863 |
| Assists | 14794 | 1.777 | 1.777 | 1.777 | 1.776 | 1.774 | 1.775 | 1.665 |

## Probability score (log loss, lower is better)

| Statistic | DraftKings no-vig | Current model | + game line | + both | + minutes information and model | + minutes information, stints and model | If minutes were known | Current model, honest spread | + game line, honest spread | + both, honest spread | + minutes information and model, honest spread | + minutes information, stints and model, honest spread | If minutes were known, honest spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Points | 0.6917 | 0.7074 | 0.7073 | 0.7065 | 0.7052 | 0.7050 | 0.6128 | 0.7070 | 0.7068 | 0.7063 | 0.7052 | 0.7047 | 0.6130 |
| Rebounds | 0.6885 | 0.7037 | 0.7030 | 0.7035 | 0.7047 | 0.7038 | 0.6767 | 0.7054 | 0.7057 | 0.7034 | 0.7037 | 0.7040 | 0.6627 |
| Assists | 0.6856 | 0.7251 | 0.7251 | 0.7237 | 0.7258 | 0.7263 | 0.7186 | 0.7225 | 0.7222 | 0.7223 | 0.7245 | 0.7255 | 0.7162 |

## Change against the current model on the same lines (95% interval; negative is better)

| Variant | Statistic | Average miss | Log loss |
|---|---|---:|---:|
| + game line | Points | -0.001 (-0.003 to +0.001) | -0.0002 (-0.0005 to +0.0001) |
| + game line | Rebounds | -0.000 (-0.001 to +0.001) | +0.0002 (-0.0001 to +0.0006) |
| + game line | Assists | +0.000 (-0.001 to +0.001) | -0.0004 (-0.0007 to +0.0000) |
| + both | Points | -0.007 (-0.011 to -0.002) | -0.0007 (-0.0013 to -0.0000) |
| + both | Rebounds | -0.002 (-0.005 to -0.000) | -0.0020 (-0.0032 to -0.0009) |
| + both | Assists | -0.000 (-0.002 to +0.002) | -0.0002 (-0.0010 to +0.0006) |
| + minutes information and model | Points | -0.023 (-0.032 to -0.015) | -0.0018 (-0.0035 to -0.0002) |
| + minutes information and model | Rebounds | -0.006 (-0.010 to -0.002) | -0.0017 (-0.0031 to -0.0003) |
| + minutes information and model | Assists | -0.002 (-0.006 to +0.001) | +0.0019 (+0.0005 to +0.0034) |
| + minutes information, stints and model | Points | -0.028 (-0.038 to -0.018) | -0.0023 (-0.0041 to -0.0004) |
| + minutes information, stints and model | Rebounds | -0.008 (-0.012 to -0.005) | -0.0015 (-0.0031 to +0.0001) |
| + minutes information, stints and model | Assists | -0.002 (-0.005 to +0.002) | +0.0030 (+0.0015 to +0.0045) |
| If minutes were known | Points | -0.745 (-0.788 to -0.704) | -0.0940 (-0.0996 to -0.0881) |
| If minutes were known | Rebounds | -0.197 (-0.211 to -0.183) | -0.0427 (-0.0483 to -0.0374) |
| If minutes were known | Assists | -0.111 (-0.123 to -0.100) | -0.0064 (-0.0111 to -0.0013) |

## What the stints add to the minutes information on the same lines (95% interval; negative is better)

| Statistic | Average miss | Log loss |
|---|---:|---:|
| Points | -0.004 (-0.011 to +0.002) | -0.0005 (-0.0016 to +0.0006) |
| Rebounds | -0.002 (-0.004 to +0.000) | +0.0002 (-0.0005 to +0.0010) |
| Assists | +0.001 (-0.002 to +0.003) | +0.0010 (+0.0003 to +0.0018) |

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
| + minutes information and model | Points | 15439 | 52.0% | -2.2% | +0.32 (0.03) |
| + minutes information and model | Rebounds | 14286 | 50.9% | -3.6% | +0.43 (0.03) |
| + minutes information and model | Assists | 11798 | 49.8% | -5.4% | +0.41 (0.03) |
| + minutes information, stints and model | Points | 15266 | 52.3% | -1.6% | +0.35 (0.03) |
| + minutes information, stints and model | Rebounds | 14336 | 51.1% | -3.2% | +0.45 (0.03) |
| + minutes information, stints and model | Assists | 11811 | 49.5% | -5.9% | +0.41 (0.03) |
| If minutes were known | Points | 19064 | 67.4% | +26.6% | +0.95 (0.01) |
| If minutes were known | Rebounds | 17894 | 61.3% | +15.5% | +0.92 (0.01) |
| If minutes were known | Assists | 13136 | 55.9% | +5.8% | +0.85 (0.02) |

## Average miss on every player-game (no line needed)

| Statistic | Current model | + game line | + both | + minutes information and model | + minutes information, stints and model | If minutes were known |
|---|---:|---:|---:|---:|---:|---:|
| Points | 4.443 | 4.443 | 4.427 | 4.408 | 4.399 | 3.617 |
| Rebounds | 1.841 | 1.841 | 1.835 | 1.827 | 1.824 | 1.576 |
| Assists | 1.307 | 1.307 | 1.304 | 1.300 | 1.299 | 1.192 |

## Minutes model with the game line: average miss in minutes

| Player-games | Rows | Last-3 average | Last-10 average | Minutes model |
|---|---:|---:|---:|---:|
| All | 86198 | 5.01 | 5.31 | 4.53 |
| Starters | 40394 | 4.79 | - | 4.29 |
| Team missing 2+ rotation players | 28397 | 5.50 | - | 4.82 |

## New minutes information: the minutes forecast alone (average miss in minutes)

Absence = which rotation teammates are newly out or back and how the player's minutes moved without them; returning = his own lay-off; role = tonight's lineup against his earlier starts and bench games.

| Player-games | Rows | Last-3 average | Current inputs | + absence | + returning | + role | + all three |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 86419 | 5.01 | 4.54 | 4.49 | 4.53 | 4.52 | 4.46 |
| Starters | 40494 | 4.79 | 4.30 | 4.29 | 4.30 | 4.28 | 4.27 |
| Team missing 2+ rotation players | 28473 | 5.49 | 4.83 | 4.77 | 4.82 | 4.80 | 4.74 |
| 30+ rotation minutes newly out | 26714 | 5.64 | 4.89 | 4.84 | 4.88 | 4.86 | 4.81 |
| 30+ rotation minutes back | 17736 | 5.28 | 4.69 | 4.54 | 4.66 | 4.64 | 4.51 |
| First 5 games back from a lay-off | 12520 | 5.41 | 4.72 | 4.63 | 4.63 | 4.68 | 4.54 |
| Role changed tonight | 6894 | 7.44 | 5.12 | 5.06 | 5.11 | 5.03 | 4.96 |

+ all three against current inputs: -0.077 minutes (-0.087 to -0.067).

## Rotation stints on top of the minutes information: the minutes forecast alone (average miss in minutes)

Stint patterns = minutes with garbage time taken out, share of time in the fourth quarter and close finishes, when he first comes on, whether his last game ended early; substitutes = the newly-out and returning teammates weighted by how much his floor time went with their bench time.

| Player-games | Rows | Last-3 average | Minutes information | + stint patterns | + substitutes | + both |
|---|---:|---:|---:|---:|---:|---:|
| All | 86419 | 5.01 | 4.46 | 4.42 | 4.46 | 4.42 |
| Starters | 40494 | 4.79 | 4.27 | 4.24 | 4.27 | 4.23 |
| Team missing 2+ rotation players | 28473 | 5.49 | 4.74 | 4.70 | 4.74 | 4.70 |
| 30+ rotation minutes newly out | 26714 | 5.64 | 4.81 | 4.77 | 4.81 | 4.77 |
| 30+ rotation minutes back | 17736 | 5.28 | 4.51 | 4.48 | 4.51 | 4.47 |
| First 5 games back from a lay-off | 12520 | 5.41 | 4.54 | 4.49 | 4.55 | 4.48 |
| Role changed tonight | 6894 | 7.44 | 4.96 | 4.96 | 4.95 | 4.93 |

+ both against minutes information: -0.045 minutes (-0.052 to -0.038).

Stints: 6742 games stored, 22 left out because their stints do not add up to the box score; 100% of the test player-games have stint inputs.

Kept from an earlier run (their changes are against that run's current model): + game line (2026-11-12 04:41 UTC), + both (2026-11-12 04:41 UTC), If minutes were known (2026-11-12 04:41 UTC).

Game lines: 100% of the scored lines (81% of all test player-games) have a DraftKings spread and total taken before tip; the rest are left empty. "If minutes were known" uses the minutes actually played and cannot be had before a game.
