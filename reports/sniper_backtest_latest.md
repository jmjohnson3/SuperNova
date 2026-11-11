# Sports Betting Sniper: backtest

Built 2026-11-11 19:08 UTC. Research on paper only: simulated $1 bets at prices that were stored, graded from final statistics. A past result is not a forecast.

Stored snapshots Oct 22 to Nov 11, 2026, 177 games, 8302 flagged prices.

## Rules compared

| Rule | Settled | W-L-P | ROI | 95% interval | Est. EV | Closing value (bets) | Max drawdown (units) | Longest losing run |
|---|---|---|---|---|---|---|---|---|
| Automatic v1 (any edge, every book) | 371 | 189-182-0 | -3.0% | -13% to +7% | +4.4% | +3.0% (295) | 25.9 | 10 |
| Automatic v2 (4%+, bet books) | 103 | 52-51-0 | +0.5% | -17% to +18% | +5.8% | +3.8% (76) | 6.3 | 3 |
| v2, FanDuel and DraftKings only | 97 | 51-46-0 | +3.9% | -14% to +22% | +5.8% | +3.9% (72) | 6.9 | 3 |
| v2, unders only | 67 | 34-33-0 | -1.3% | -20% to +18% | +6.0% | +4.7% (50) | 7.8 | 4 |
| v2, overs only | 36 | 18-18-0 | +4.0% | -32% to +40% | +5.5% | +2.1% (26) | 5.3 | 3 |
| v2, entered one look later | 45 | 21-24-0 | -8.7% | -37% to +19% | +5.8% | +3.3% (36) | 6.2 | 4 |
| v2, any edge (0%+) | 205 | 105-100-0 | -0.9% | -15% to +13% | +4.5% | +2.5% (151) | 10.8 | 5 |
| v2, model agrees | 39 | 22-17-0 | +21.6% | -7% to +51% | +6.5% | +3.9% (28) | 3.0 | 3 |
| v2, model disagrees | 44 | 21-23-0 | -10.5% | -38% to +17% | +5.4% | +4.1% (33) | 8.6 | 3 |

ROI is profit over stakes; the interval treats whole games as the draws. Closing value is the edge of the price taken against the fair price at the last look before the game. "One look later" enters at the same book's price about ten minutes on and drops the bet when the line is gone or no longer clears the bar.

## By minimum estimated edge (bet books)

| Minimum edge | Settled | W-L-P | ROI | 95% interval | Est. EV | Closing value (bets) | Max drawdown (units) | Longest losing run |
|---|---|---|---|---|---|---|---|---|
| 0%+ | 205 | 105-100-0 | -0.9% | -15% to +13% | +4.5% | +2.5% (151) | 10.8 | 5 |
| 1%+ | 205 | 105-100-0 | -0.9% | -15% to +13% | +4.5% | +2.5% (151) | 10.8 | 5 |
| 2%+ | 205 | 105-100-0 | -0.9% | -15% to +13% | +4.5% | +2.5% (151) | 10.8 | 5 |
| 3%+ | 183 | 93-90-0 | -0.4% | -15% to +14% | +4.7% | +2.8% (132) | 9.8 | 4 |
| 4%+ | 103 | 52-51-0 | +0.5% | -17% to +18% | +5.8% | +3.8% (76) | 6.3 | 3 |
| 5%+ | 59 | 29-30-0 | +0.1% | -24% to +24% | +7.1% | +4.3% (41) | 5.3 | 4 |
| 6%+ | 38 | 16-22-0 | -12.5% | -41% to +16% | +8.1% | +4.8% (26) | 7.0 | 6 |
| 8%+ | 15 | 8-7-0 | +12.6% | -38% to +63% | +10.2% | +5.5% (11) | 3.3 | 3 |

These rows overlap and were all measured on the same bets, so the best-looking row is partly luck. The walk-forward below is the fair version.

## Walk-forward

Minimum edge chosen on earlier days by results (ROI), applied to the next three game days:

| Later days | Edge chosen | Settled | W-L-P | ROI | Closing value |
|---|---|---|---|---|---|
| Oct 29 to Oct 31 | 3% | 25 | 12-13-0 | +2.3% | +3.4% |
| Nov 01 to Nov 05 | 4% | 18 | 10-8-0 | +11.3% | +5.0% |
| Nov 06 to Nov 08 | 4% | 19 | 9-10-0 | -8.5% | +4.2% |
| Nov 09 to Nov 11 | 4% | 11 | 6-5-0 | +0.9% | +3.9% |
| **All later days** | | 73 | 37-36-0 | +1.5% (-19% to +22%) | +4.1% |

Minimum edge chosen on earlier days by closing value, applied to the next three game days:

| Later days | Edge chosen | Settled | W-L-P | ROI | Closing value |
|---|---|---|---|---|---|
| Oct 29 to Oct 31 | 3% | 25 | 12-13-0 | +2.3% | +3.4% |
| Nov 01 to Nov 05 | 4% | 18 | 10-8-0 | +11.3% | +5.0% |
| Nov 06 to Nov 08 | 4% | 19 | 9-10-0 | -8.5% | +4.2% |
| Nov 09 to Nov 11 | 4% | 11 | 6-5-0 | +0.9% | +3.9% |
| **All later days** | | 73 | 37-36-0 | +1.5% (-19% to +22%) | +4.1% |

## Automatic v2 (4%+, bet books): by period

| Days | Settled | W-L-P | ROI | 95% interval | Est. EV | Closing value (bets) | Max drawdown (units) | Longest losing run |
|---|---|---|---|---|---|---|---|---|
| Oct 22 to Oct 28 | 38 | 18-20-0 | -5.8% | -38% to +26% | +5.8% | +3.1% (28) | 6.3 | 3 |
| Oct 29 to Nov 04 | 33 | 17-16-0 | +6.7% | -24% to +38% | +6.3% | +4.3% (24) | 3.3 | 3 |
| Nov 05 to Nov 11 | 32 | 17-15-0 | +1.7% | -25% to +29% | +5.3% | +4.2% (24) | 4.2 | 2 |

## Automatic v2 (4%+, bet books): by market

| Market | Settled | W-L-P | ROI | 95% interval | Est. EV | Closing value (bets) | Max drawdown (units) | Longest losing run |
|---|---|---|---|---|---|---|---|---|
| Assists | 21 | 10-11-0 | -8.7% | -50% to +33% | +6.1% | +3.4% (16) | 4.5 | 3 |
| Points | 36 | 16-20-0 | -14.7% | -44% to +15% | +5.9% | +4.3% (26) | 7.9 | 4 |
| Rebounds | 19 | 13-6-0 | +36.9% | +3% to +71% | +6.0% | +4.1% (14) | 2.0 | 2 |
| Receiving yards | 2 | 0-2-0 | -100.0% | - | +5.2% | +6.3% (1) | 2.0 | 2 |
| Receptions | 4 | 1-3-0 | -57.0% | - | +4.2% | +3.8% (4) | 2.3 | 2 |
| Rushing yards | 4 | 3-1-0 | +40.2% | - | +5.2% | +2.0% (4) | 1.0 | 1 |
| Three-pointers | 17 | 9-8-0 | +19.4% | -36% to +74% | +5.7% | +3.3% (11) | 3.2 | 3 |

## Paper bankroll (Automatic v2 (4%+, bet books))

| Staking | Start | End | Staked | Return on stakes | Bets placed | Largest fall | Lowest point | Left out |
|---|---|---|---|---|---|---|---|---|
| Flat $1 | $100 | $103.12 | $98.00 | +3.2% | 98 | $3.36 (3%) | $96.64 | 5 game limit |
| Quarter Kelly | $100 | $96.01 | $65.71 | -6.1% | 98 | $7.38 (7%) | $95.28 | 4 game limit, 1 stake too small (no edge after the haircut) |

Hypothetical $100. Limits: $5 a bet, $20 a day, $3 a game, $2 a player. Quarter Kelly stakes a quarter of the Kelly amount on the probability lowered by its uncertainty.

## Model against market (NBA)

4502 two-sided lines had a model forecast made before the price was seen. Lower is better.

| Lines | Settled | Bet book's own price | Market consensus | Prediction model |
|---|---|---|---|---|
| NBA (Brier) | 3709 | 0.2468 | 0.2465 | 0.2645 |
| NFL (Brier) | 437 | 0.2432 | 0.2435 | 0.2417 |
| Assists (Brier) | 625 | 0.2444 | 0.2444 | 0.2662 |
| Points (Brier) | 1257 | 0.2486 | 0.2480 | 0.2578 |
| Rebounds (Brier) | 1008 | 0.2439 | 0.2437 | 0.2474 |
| Receptions (Brier) | 437 | 0.2432 | 0.2435 | 0.2417 |
| Three-pointers (Brier) | 819 | 0.2495 | 0.2492 | 0.2944 |

With the model calibrated on earlier days only (the lines after the first fit):

| Lines | Settled | Bet book | Market consensus | Model | Model, calibrated | Market moved by the model |
|---|---|---|---|---|---|---|
| NBA (Brier) | 3493 | 0.2466 | 0.2463 | 0.2647 | 0.2488 | 0.2465 |
| NFL (Brier) | 281 | 0.2464 | 0.2461 | 0.2449 | 0.2504 | 0.2459 |

NBA calibration on 3709 settled lines: the model's log-odds are multiplied by 0.19 (1 would mean its confidence is right, 0 that it says nothing); set against the market, its disagreement gets a weight of 0.09 (0 means the market already knows it).

NFL calibration on 437 settled lines: the model's log-odds are multiplied by 1.59 (1 would mean its confidence is right, 0 that it says nothing); set against the market, its disagreement gets a weight of 0.56 (0 means the market already knows it).

Betting the side each source says is underpriced, at the bet book's price, on those same lines:

| Source | Edge needed | Settled | W-L | ROI | Its own estimate |
|---|---|---|---|---|---|
| Market consensus (other books) | 2%+ | 45 | 25-20 | +13.5% ± 17.4% | +4.3% |
| Market consensus (other books) | 4%+ | 12 | 6-6 | +1.8% ± 34.7% | +8.0% |
| Market consensus (other books) | 6%+ | 6 | 2-4 | -34.0% ± 41.4% | +11.4% |
| Prediction model | 2%+ | 2680 | 1310-1370 | -5.2% ± 1.7% | +21.0% |
| Prediction model | 4%+ | 2452 | 1192-1260 | -5.3% ± 1.8% | +22.6% |
| Prediction model | 6%+ | 2222 | 1067-1155 | -6.1% ± 2.0% | +24.5% |
| Model, calibrated on earlier days | 2%+ | 1797 | 823-974 | -3.7% ± 2.4% | +10.7% |
| Model, calibrated on earlier days | 4%+ | 1436 | 653-783 | -3.2% ± 2.7% | +12.7% |
| Model, calibrated on earlier days | 6%+ | 1137 | 513-624 | -2.6% ± 2.8% | +14.7% |
| Market moved by the model | 2%+ | 577 | 290-287 | -2.3% ± 4.4% | +5.6% |
| Market moved by the model | 4%+ | 351 | 178-173 | +0.2% ± 5.3% | +7.4% |
| Market moved by the model | 6%+ | 206 | 106-100 | +5.1% ± 8.2% | +9.2% |

The NFL model is compared on receptions and passing touchdowns only: its yardage projections are anchored to the line, so they carry no chance of their own.

## Breakout signals

Lines on players whose recent opportunity was above their season baseline (games before that day only), against every other line. Over ROI is betting every over at the bet book's price.

| Sport | Lines | Count | Players | Over won | Market's chance of the over | Over ROI |
|---|---|---|---|---|---|---|
| NBA | No signal | 3808 | 252 | 46.6% | 48.6% | -9.9% ± 1.8% |
| NBA | Breakout signal | 272 | 67 | 46.0% | 49.2% | -12.8% ± 6.8% |
| NFL | No signal | 875 | 251 | 46.4% | 49.6% | -11.7% ± 4.2% |
| NFL | Breakout signal | 215 | 91 | 45.1% | 49.5% | -14.6% ± 7.5% |
