# NCAAF Ratings Model vs the Market

Walk-forward, seasons 2016-2025, 8583 games (each week predicted from earlier games only).

| | Model MAE | Closing line MAE | Games |
|---|---:|---:|---:|
| Margin | 13.29 | 12.30 | 8435 |
| Total | 13.19 | 12.79 | 7498 |

Blend (actual on market and model): spread model weight -0.015 (se 0.029); total model weight +0.115 (se 0.042). A weight clearly above zero is information the market lacks.

Line movement: correlation between the model's disagreement with the opener and the opener-to-close move = +0.077 on 4325 games (positive = the market moves toward the model).

## Spread vs closing line

| Model differs by | Picks | Per season | Hit rate (52.4% breaks even) |
|---:|---:|---:|---:|
| >= 1 pts | 6989 | 699 | 49.2% |
| >= 2 pts | 5710 | 571 | 48.8% |
| >= 3 pts | 4551 | 455 | 48.5% |
| >= 4 pts | 3570 | 357 | 48.2% |
| >= 5 pts | 2769 | 277 | 47.8% |
| >= 7 pts | 1617 | 162 | 48.2% |

## Total vs closing line

| Model differs by | Picks | Per season | Hit rate (52.4% breaks even) |
|---:|---:|---:|---:|
| >= 1 pts | 6097 | 677 | 50.7% |
| >= 2 pts | 4777 | 531 | 51.0% |
| >= 3 pts | 3632 | 404 | 50.8% |
| >= 4 pts | 2642 | 294 | 51.9% |
| >= 5 pts | 1879 | 209 | 51.1% |
| >= 7 pts | 810 | 90 | 53.5% |

## Spread vs opener

| Model differs by | Picks | Per season | Hit rate (52.4% breaks even) |
|---:|---:|---:|---:|
| >= 1 pts | 3499 | 700 | 49.2% |
| >= 2 pts | 2807 | 561 | 48.8% |
| >= 3 pts | 2157 | 431 | 49.4% |
| >= 4 pts | 1674 | 335 | 49.3% |
| >= 5 pts | 1264 | 253 | 49.0% |
| >= 7 pts | 702 | 140 | 47.9% |

## Total vs opener

| Model differs by | Picks | Per season | Hit rate (52.4% breaks even) |
|---:|---:|---:|---:|
| >= 1 pts | 3337 | 667 | 51.7% |
| >= 2 pts | 2515 | 503 | 52.7% |
| >= 3 pts | 1764 | 353 | 52.0% |
| >= 4 pts | 1182 | 236 | 52.5% |
| >= 5 pts | 750 | 150 | 52.8% |
| >= 7 pts | 263 | 53 | 53.2% |
