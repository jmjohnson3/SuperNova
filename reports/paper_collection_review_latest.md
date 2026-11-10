# Paper collection review

Updated 2026-11-10T02:40:14.146295+00:00.
Automatic public-feed checks do not satisfy frozen manual/account-availability requirements. Counts trigger review only. Missing execution and actual net profit remain unknown.

Frozen report counts may include closes after the earlier native schedule. Use price_survival_review_latest for the stricter audit; unaudited counts cannot establish pregame coverage.

| NBA book | Decisions | Games / 20 | Span / 21 days | Exact closes | Timely manual checks | Checked exact closes / 50 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| fanduel | 69 | 39 | 13 | 69 | 0 | 0 | collecting |
| draftkings | 103 | 52 | 12 | 103 | 0 | 0 | collecting |
| espnbet | 26 | 20 | 13 | 26 | 0 | 0 | collecting |

NBA also requires active days/weeks, coverage, availability and positive game/day-clustered uncertainty bounds.
NFL is evaluated separately: its paired trial requires 100 checked exact closes, 21 days and 20 games, plus its coverage and uncertainty checks.

| Paired sport / family | Price-only decisions | Model-selected decisions | Model-selected checked prices |
|---|---:|---:|---:|
| nba_props | 65 | 33 | 0 |
| nfl_props | 6 | 2 | 0 |

Both strategies reuse the same offer and first observation. Model vetoes remain in the price-only control.

| Automatic public-feed observer | Attempts | Timely exact quotes | Status |
|---|---:|---:|---|
| auto_nba | 20 | 11 | current |
| auto_nfl | 1 | 0 | current |

Monthly data budget: $30.00; comparison stake $1; bankroll reported by user: $50.
At an assumed true 3% return: 1000 one-dollar bets/month to cover the data bill.
Current frozen NBA volume: 65 offers over 12 calendar days. If all were available and a true 3% return existed, the monthly gross would be $4.88; net after data $-25.12.
This is conditional cost planning, not an observed edge. NFL is not assumed to cover the shortfall. No stake increase or subscription change is authorized by this report.

Actual receipt review: 2026-11-04 through 2026-11-10; 0 wager receipts through period end. See private weekly finance report for costs and completeness.
A monthly subscription budget is not a payment receipt; unknown invoices and wager inventories are not filled automatically.

## Sources

- nba: current; 2026-11-10T02:37:10.298376+00:00; `reports\nba_price_checks_latest.json`.
- paired: current; 2026-11-10T02:40:11.362727+00:00; `reports\model_research_latest.json`.
- survival: current; 2026-11-10T02:36:54.994701+00:00; `reports\price_survival_review_latest.json`.
- auto_nba: current; 2026-11-10 02:35:51.118841+00:00; `reports\automatic_price_checks_nba_latest.json`.
- auto_nfl: current; 2026-11-10 02:31:25.374927+00:00; `reports\automatic_price_checks_nfl_latest.json`.
