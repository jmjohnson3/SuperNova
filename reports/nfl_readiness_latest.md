# NFL Readiness Report - 2026-10-25

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: player prop rows parsed
- Game lock/close coverage: 100.0%
- Prop lock/close coverage: 83.2%
- True-paired prop coverage: 21.4%
- Game predictions: 121 | Prop predictions: 3554
- Ledger rows: 516 | CLV rows: 1828
- Valid prop CLV capture: 89.7%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 918

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 20387 rows |
| Depth context | pass | 509954 rows |
| Injury context | pass | 5764 rows |
| Snap context | pass | 100.0% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | pass | 566 rows |
| Prop odds parsed | pass | 138511 rows; player prop rows parsed |
| Game lock/close | pass | 100.0% |
| Prop lock/close | watch | 83.2% |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 918 |
| Game predictions | pass | 121 rows |
| Prop projections | pass | 3554 rows |
| Ledger | pass | 516 rows |
| CLV | pass | 1828 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 20387 | 33 | 2026-10-27 06:30:52.236542-07:00 |
| depth 2026 | 509954 | 32 | 2026-10-27 06:08:56-07:00 |
| injury 2026 | 5764 | 32 | 2026-10-27 06:33:47.603375-07:00 |

## Player Usage Coverage

| Field | Rows | Coverage |
|---|---:|---:|
| offense snap share | 2429 | 100.0% |
| true routes run | 0 | 0.0% |
| pass-route opportunities | 0 | 0.0% |
| pass-route opportunity share | 0 | 0.0% |
| route participation | 0 | 0.0% |
| route participation proxy, training 2026 | 2325 | 100.0% |
| estimated routes proxy, training 2026 | 2233 | 96.0% |
| starter confidence, training 2026 | 2325 | 100.0% |
| rest-risk score, training 2026 | 2325 | 100.0% |
| receiving usage quality v4, training 2026 | 2325 | 100.0% |
| RB usage quality v4, training 2026 | 2325 | 100.0% |
| TD usage quality v4, training 2026 | 2325 | 100.0% |
| live usage context quality v4, training 2026 | 2325 | 100.0% |
| receiver spike correction v4, training 2026 | 2325 | 100.0% |
| RB spike correction v4, training 2026 | 2325 | 100.0% |
| target share | 2412 | 99.3% |
| air-yards share | 2412 | 99.3% |
| WOPR | 2412 | 99.3% |
| first-read targets | 0 | 0.0% |
| end-zone targets | 1846 | 76.0% |
| red-zone touches | 2429 | 100.0% |
| goal-line carries | 2429 | 100.0% |
| goal-line targets | 2429 | 100.0% |

## Raw Odds Payloads

| Endpoint | Role | Payloads | Latest Fetch |
|---|---|---:|---|
| nfl_game_odds | close | 30 | 2026-10-25 17:15:17.946751-07:00 |
| nfl_game_odds | live | 71 | 2026-10-25 17:10:12.285644-07:00 |
| nfl_game_odds | lock | 10 | 2026-10-25 15:21:08.259459-07:00 |
| nfl_player_props | close | 42 | 2026-10-25 17:15:17.973439-07:00 |
| nfl_player_props | live | 71 | 2026-10-25 17:10:12.254458-07:00 |
| nfl_player_props | lock | 22 | 2026-10-25 15:21:25.784374-07:00 |
| nfl_sportsgameodds_events_raw | close | 30 | 2026-10-25 17:15:17.834583-07:00 |
| nfl_sportsgameodds_events_raw | lock | 10 | 2026-10-25 15:21:07.897987-07:00 |

## Prop Payload Diagnostic

| Role | Payloads | Bookmaker Entries | Empty-Book Payloads | Latest Fetch |
|---|---:|---:|---:|---|
| none | 0 | 0 | 0 | - |

## Raw Prop Markets

| Role | Book | Market | Payloads | Outcomes |
|---|---|---|---:|---:|
| none | - | no raw prop markets found | 0 | 0 |

## Parsed Rows

| Type | Rows | Distinct Entities |
|---|---:|---:|
| game odds | 566 | 12 |
| player props | 138511 | 204 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 24 | 24 | 100.0% |
| prop | 1486 | 1236 | 83.2% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 138511 | raw parser output |
| true paired over/under offers | 29613 | 21.4% |
| lock markets | 1486 | exact book/player/stat/line locked |
| close markets | 8547 | exact book/player/stat/line closed |
| lock-close exact matches | 1236 | 83.2% |
| CLV labels | 1707 | valid: 1531 |
| graded prop predictions | 1687 | line-level result rows |
| prop ledger rows | 472 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| valid_close | 1531 |
| exact_line_unavailable_in_captured_feed | 157 |
| player_market_not_observed_in_capture | 14 |
| same_line_missing_side_price | 5 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| game | paper | 44 |
| prop | paper | 472 |
