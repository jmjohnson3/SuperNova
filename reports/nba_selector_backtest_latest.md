# NBA prop selector: backtest against DraftKings' pre-game price

`nba-ml-selector-v1`. 44,699 points, rebounds and assists offers, 2024-10-22 to 2026-04-25. Each month is predicted by a model trained only on earlier months, and every bet is graded at the price DraftKings offered.

Log loss (lower is better): DraftKings' no-vig price 0.68898, the stat projection alone 0.71204, the selector 0.68900.

| Bets taken | Bets | Won | Result per $1 | Standard error | Unders | 2024-25 | 2025-26 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Every over | 44,699 | 48.5% | -8.5% | 0.5% | 0% |  |  |
| Every under | 44,699 | 51.5% | -3.9% | 0.5% | 100% |  |  |
| Selector, EV at least 0% | 11,064 | 53.4% | -0.3% | 0.9% | 82% | -1.5% (5,459) | +0.9% (5,605) |
| Selector, EV at least 2% | 6,567 | 53.6% | +0.2% | 1.1% | 84% | -1.3% (3,505) | +2.0% (3,062) |
| Selector, EV at least 4% | 3,579 | 53.1% | -0.6% | 1.5% | 87% | -1.6% (2,074) | +0.7% (1,505) |
| Selector, EV at least 6% | 1,749 | 52.5% | -1.3% | 2.1% | 89% | -1.9% (1,132) | -0.2% (617) |
| Selector, EV at least 8% | 892 | 52.6% | -0.9% | 3.0% | 91% | +1.3% (644) | -6.6% (248) |

The paper rule is EV at least 6%. By stat and side:

| Stat and side | Bets | Won | Result per $1 |
|---|---:|---:|---:|
| ast over | 77 | 54.5% | -3.2% |
| ast under | 273 | 52.4% | +0.1% |
| pts over | 75 | 57.3% | +7.2% |
| pts under | 571 | 52.7% | -2.6% |
| reb over | 34 | 58.8% | +6.9% |
| reb under | 719 | 51.3% | -1.8% |

Best threshold: EV at least 2%, +0.2% per $1 with a standard error of 1.1%. An edge needs a result well above zero in every season; anything within two standard errors of zero is break-even at best.
