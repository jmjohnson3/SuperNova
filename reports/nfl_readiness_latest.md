# NFL Readiness Report - 2026-10-22

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: player prop rows parsed
- Game lock/close coverage: 100.0%
- Prop lock/close coverage: 71.3%
- True-paired prop coverage: 34.3%
- Game predictions: 8 | Prop predictions: 361
- Ledger rows: 69 | CLV rows: 156
- Valid prop CLV capture: 78.4%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 918

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 18159 rows |
| Depth context | pass | 491388 rows |
| Injury context | pass | 5089 rows |
| Snap context | pass | 100.0% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | pass | 18 rows |
| Prop odds parsed | pass | 5504 rows; player prop rows parsed |
| Game lock/close | pass | 100.0% |
| Prop lock/close | watch | 71.3% |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 918 |
| Game predictions | pass | 8 rows |
| Prop projections | pass | 361 rows |
| Ledger | pass | 69 rows |
| CLV | pass | 156 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 18159 | 33 | 2026-10-22 06:31:01.990456-07:00 |
| depth 2026 | 491388 | 32 | 2026-10-22 05:59:29-07:00 |
| injury 2026 | 5089 | 32 | 2026-10-22 15:16:05.601855-07:00 |

## Player Usage Coverage

| Field | Rows | Coverage |
|---|---:|---:|
| offense snap share | 2120 | 100.0% |
| true routes run | 0 | 0.0% |
| pass-route opportunities | 0 | 0.0% |
| pass-route opportunity share | 0 | 0.0% |
| route participation | 0 | 0.0% |
| route participation proxy, training 2026 | 2030 | 100.0% |
| estimated routes proxy, training 2026 | 1940 | 95.6% |
| starter confidence, training 2026 | 2030 | 100.0% |
| rest-risk score, training 2026 | 2030 | 100.0% |
| receiving usage quality v4, training 2026 | 2030 | 100.0% |
| RB usage quality v4, training 2026 | 2030 | 100.0% |
| TD usage quality v4, training 2026 | 2030 | 100.0% |
| live usage context quality v4, training 2026 | 2030 | 100.0% |
| receiver spike correction v4, training 2026 | 2030 | 100.0% |
| RB spike correction v4, training 2026 | 2030 | 100.0% |
| target share | 2106 | 99.3% |
| air-yards share | 2106 | 99.3% |
| WOPR | 2106 | 99.3% |
| first-read targets | 0 | 0.0% |
| end-zone targets | 1607 | 75.8% |
| red-zone touches | 2120 | 100.0% |
| goal-line carries | 2120 | 100.0% |
| goal-line targets | 2120 | 100.0% |

## Raw Odds Payloads

| Endpoint | Role | Payloads | Latest Fetch |
|---|---|---:|---|
| nfl_game_odds | close | 5 | 2026-10-22 17:10:15.744226-07:00 |
| nfl_game_odds | live | 6 | 2026-10-22 17:10:12.441816-07:00 |
| nfl_game_odds | lock | 4 | 2026-10-22 15:16:13.302450-07:00 |
| nfl_player_props | close | 6 | 2026-10-22 17:10:15.759822-07:00 |
| nfl_player_props | live | 6 | 2026-10-22 17:10:12.426256-07:00 |
| nfl_player_props | lock | 5 | 2026-10-22 15:16:20.842930-07:00 |
| nfl_sportsgameodds_events_raw | close | 5 | 2026-10-22 17:10:15.619196-07:00 |
| nfl_sportsgameodds_events_raw | lock | 4 | 2026-10-22 15:16:13.202444-07:00 |

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
| game odds | 18 | 1 |
| player props | 5504 | 18 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 2 | 2 | 100.0% |
| prop | 174 | 124 | 71.3% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 5504 | raw parser output |
| true paired over/under offers | 1887 | 34.3% |
| lock markets | 174 | exact book/player/stat/line locked |
| close markets | 868 | exact book/player/stat/line closed |
| lock-close exact matches | 124 | 71.3% |
| CLV labels | 148 | valid: 116 |
| graded prop predictions | 0 | line-level result rows |
| prop ledger rows | 67 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| valid_close | 116 |
| exact_line_unavailable_in_captured_feed | 31 |
| same_line_missing_side_price | 1 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| game | paper | 2 |
| prop | paper | 67 |
