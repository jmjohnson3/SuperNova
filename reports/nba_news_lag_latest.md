# NBA props: do prices lag lineup and injury news?

Built 2026-11-12 20:06 UTC. Research only; stored data, no credits. 2026-10-24 to 2026-11-12: 17 days, 127 games with saved prices (634 looks, all in the last 50 minutes before tip).

A news move is a live projection that changes 3% or more when the news watcher sees a lineup or injury change. 12610 in all; 2377 have a FanDuel or DraftKings price after the news and 1483 of those prices came within 15 minutes of our seeing it.

Moves are in percent of the projection the price implies, signed so that positive is the direction the model moved. Intervals are 95%, from resampling days.

## Fresh news: where the market's move happened

| Book | Moves | Model's move | Market after our first look | With a price before | Market before our first look | Market ended the model's way |
|---|---:|---:|---:|---:|---:|---:|
| fanduel | 1192 | +11.2% | -0.07% (-0.36% to +0.27%) | 251 | +3.50% (+1.85% to +5.06%) | 71% of 200 |
| draftkings | 1406 | +11.3% | -0.24% (-0.51% to +0.03%) | 280 | +3.93% (+1.41% to +6.67%) | 68% of 238 |
| pinnacle | 691 | +10.0% | +0.20% (-0.04% to +0.42%) | 138 | +2.18% (+1.05% to +3.80%) | 83% of 95 |

## By how old the news was at the first price (FanDuel and DraftKings)

| Age of the news | Moves | Model's move | Market after our first look | With a price before | Market before our first look | Market ended the model's way |
|---|---:|---:|---:|---:|---:|---:|
| Under 15 minutes | 2598 | +11.3% | -0.16% (-0.37% to +0.06%) | 531 | +3.73% (+1.78% to +5.75%) | 69% of 438 |
| 15 to 60 minutes | 916 | +9.3% | -0.39% (-0.82% to +0.19%) | 1 | -2.70% | 100% of 1 |
| 1 to 4 hours | 201 | +8.8% | +0.01% (-0.52% to +0.64%) | 0 | - | - |
| Over 4 hours | 536 | +10.7% | +0.09% (-0.64% to +0.76%) | 33 | +2.17% | 45% of 33 |

## Fresh news by kind (FanDuel and DraftKings)

| Kind | Moves | Model's move | Market after our first look | With a price before | Market before our first look | Market ended the model's way |
|---|---:|---:|---:|---:|---:|---:|
| His own role changed | 550 | +23.1% | -0.19% (-0.80% to +0.43%) | 408 | +4.64% (+2.21% to +7.28%) | 74% of 334 |
| A teammate dropped out or came back | 926 | +6.6% | -0.37% (-0.90% to +0.17%) | 55 | +0.62% (-0.27% to +1.72%) | 48% of 44 |
| Other news | 1122 | +9.3% | +0.02% (-0.52% to +0.48%) | 68 | +0.77% (-0.34% to +1.72%) | 57% of 60 |

## Fresh news by size of the model's move (FanDuel and DraftKings)

| Size | Moves | Model's move | Market after our first look | With a price before | Market before our first look | Market ended the model's way |
|---|---:|---:|---:|---:|---:|---:|
| 3% to 10% | 1481 | +6.1% | -0.13% (-0.49% to +0.31%) | 108 | +1.11% (+0.14% to +2.09%) | 56% of 89 |
| 10% or more | 1117 | +18.1% | -0.20% (-0.58% to +0.10%) | 423 | +4.39% (+2.11% to +6.76%) | 72% of 349 |

Control, players with no news that day (7230 player-statistic-books): first price to last, the implied projection moved -0.15% (-0.28% to -0.03%) on average and 2.38% in size.

## Betting the news side at the first price after the news (FanDuel and DraftKings)

Expected value is at Pinnacle's last no-vig price before tip; the result is the box score. One bet per player, statistic, book and side.

| Bets | Count | Value at Pinnacle's close | Hit rate | Return on results |
|---|---:|---:|---:|---:|
| All news | 4199 | -6.1% (-6.5% to -5.6%) | 51.1% | -3.8% (-7.1% to -0.5%) |
| Fresh news (price within 15 minutes) | 2546 | -5.8% (-6.1% to -5.3%) | 51.5% | -3.9% (-7.7% to +0.6%) |
| Older news | 1653 | -6.5% (-7.2% to -5.8%) | 50.6% | -3.8% (-8.9% to +0.4%) |
| Fresh, 10% or more | 1086 | -5.8% (-6.3% to -5.0%) | 50.3% | -5.4% (-14.4% to +5.1%) |
| Fresh, over side | 631 | -6.3% (-6.9% to -5.6%) | 46.6% | -10.8% (-21.4% to -0.6%) |
| Fresh, under side | 1915 | -5.6% (-5.9% to -5.1%) | 53.1% | -1.6% (-6.3% to +4.4%) |
| Fresh, fanduel | 1161 | -5.6% (-5.9% to -5.1%) | 51.2% | -4.8% (-9.2% to +0.2%) |
| Fresh, draftkings | 1385 | -6.0% (-6.4% to -5.4%) | 51.7% | -3.1% (-6.8% to +1.5%) |
| No news: every over at the first price | 7885 | -6.8% (-7.1% to -6.4%) | 47.3% | -9.4% (-14.0% to -4.6%) |
| No news: every under at the first price | 7885 | -5.4% (-5.7% to -5.2%) | 52.7% | -3.0% (-7.7% to +1.3%) |

## What being one look faster would be worth

Fresh news with a price saved before the watcher saw it: the news side at that earlier price (about 10 minutes sooner; we did not have the news then) against the same bets at the first price after.

| Bets | Count | Value at Pinnacle's close | Hit rate | Return on results |
|---|---:|---:|---:|---:|
| At the price before we saw the news (not available to us) | 573 | -2.6% (-4.1% to -0.9%) | 52.9% | -0.7% (-11.6% to +11.6%) |
| The same bets at the first price after | 573 | -6.2% (-7.2% to -4.8%) | 50.4% | -4.2% (-15.9% to +9.9%) |
| Model moved 10% or more: at the price before | 442 | -1.5% (-3.6% to +0.6%) | 54.3% | +1.6% (-8.7% to +14.6%) |
| Model moved 10% or more: at the first price after | 442 | -6.1% (-7.3% to -4.6%) | 51.8% | -1.8% (-14.5% to +14.0%) |

## Reading it

- "After our first look" is what was still to come when we could first have bet. A lag we could use shows up there, and in a positive value at Pinnacle's close.
- "Before our first look" is the move between the last price before the watcher saw the news and the first price after, for the moves that have both; the book may have had the news first, so this is the part we could not have had.
- Prices are saved every 10 minutes from 50 minutes before tip, so news earlier in the day is only seen at its first price near tip.
- The implied projection uses a normal approximation, so moves of the line are approximate.
