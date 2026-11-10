# NBA alerts: every-alert paper ledger and phone price checks

Trial `nba-reply-check-v1` - frozen. Registered: 2026-11-10T04:55:34.044020+00:00.
Research only. Nothing here places, requests or authorizes a wager.

## If every alert were a $1 bet

Assumes the alerted price was on the screen every time, which is not verified. Results are a secondary test; EV at the close is the primary one.

| Book | Alerts | Settled | Record | Result ($1 each) | Per bet | EV when alerted | EV at close |
|---|---:|---:|---|---:|---:|---:|---:|
| FanDuel | 69 | 65 | 34-31 | +$3.89 | +6.0% | +4.8% | +1.7% |
| DraftKings | 106 | 94 | 45-49 | -$6.39 | -6.8% | +5.0% | +4.1% |
| theScore Bet | 27 | 23 | 13-10 | +$2.67 | +11.6% | +4.3% | +2.3% |
| All books | 202 | 182 | 92-90 | +$0.17 | +0.1% | +4.8% | +3.0% |

All books, per bet: 95% range -16.4% to +18.1% across 51 games. 20 alerts have no result yet.

At other flat stakes: $5 a bet +$0.85; $10 a bet +$1.71; $25 a bet +$4.27.

| Game date | Alerts | Settled | Record | Result ($1 each) |
|---|---:|---:|---|---:|
| 2026-10-29 | 11 | 11 | 5-6 | -$0.06 |
| 2026-10-30 | 25 | 23 | 13-10 | +$2.43 |
| 2026-10-31 | 27 | 27 | 9-18 | -$9.53 |
| 2026-11-01 | 13 | 13 | 5-8 | -$3.32 |
| 2026-11-04 | 33 | 33 | 22-11 | +$10.88 |
| 2026-11-05 | 11 | 11 | 6-5 | +$1.39 |
| 2026-11-06 | 26 | 26 | 14-12 | +$1.35 |
| 2026-11-07 | 20 | 20 | 10-10 | -$0.48 |
| 2026-11-08 | 12 | 12 | 3-9 | -$6.35 |
| 2026-11-09 | 18 | 0 | no results yet | +$0.00 |

## DraftKings at the price found a minute later

0 settled alerts had an automatic re-quote; 0 still had the line.

## Price checks by Discord reply

Each book needs 50 exact closes, 21 days spanning 3 weeks, 7 checking days, 20 games, 90% exact-close coverage, at least +1.5% mean closing EV, timely replies on 90% of alerts on checking days, 80% of checked offers available, 50 checked prices with exact closes, and positive 95% lower bounds by game and by day.

| Book | Alerts | Checking days | Timely checks | Coverage | There | Gone | Moved | Checked-price EV | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FanDuel | 0 | 0 | 0 | - | 0 | 0 | 0 | - | collecting |
| DraftKings | 0 | 0 | 0 | - | 0 | 0 | 0 | - | collecting |
| theScore Bet | 0 | 0 | 0 | - | 0 | 0 | 0 | - | collecting |

## Bets you reported placing

None reported.

Replies read: 0; not understood: 0; edited (not evidence): 0.

## How to reply

Reply to the alert in Discord within 5 minutes: `y` (same line, alerted price or better), `y -125` (same line, now -125), `b 5` (placed $5), `m 8.5 -120` (line moved), `n` (gone), `s` (suspended). Add `max 20` for the slip's stake limit. Send a new reply to correct one; an edited reply does not count.
