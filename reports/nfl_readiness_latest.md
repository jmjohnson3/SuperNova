# NFL Readiness Report - 2026-10-22

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: player prop rows parsed
- Game lock/close coverage: 0.0%
- Prop lock/close coverage: 0.0%
- True-paired prop coverage: 100.0%
- Game predictions: 6 | Prop predictions: 300
- Ledger rows: 56 | CLV rows: 0
- Valid prop CLV capture: 0.0%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 918

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 18159 rows |
| Depth context | pass | 491388 rows |
| Injury context | pass | 5087 rows |
| Snap context | pass | 100.0% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | pass | 6 rows |
| Prop odds parsed | pass | 212 rows; player prop rows parsed |
| Game lock/close | watch | 0.0% |
| Prop lock/close | watch | 0.0% |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 918 |
| Game predictions | pass | 6 rows |
| Prop projections | pass | 300 rows |
| Ledger | pass | 56 rows |
| CLV | watch | 0 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 18159 | 33 | 2026-10-22 06:31:01.990456-07:00 |
| depth 2026 | 491388 | 32 | 2026-10-22 05:59:29-07:00 |
| injury 2026 | 5087 | 32 | 2026-10-22 06:33:10.660758-07:00 |

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
| nfl_game_odds | lock | 3 | 2026-10-22 06:33:18.861952-07:00 |
| nfl_player_props | lock | 3 | 2026-10-22 06:33:18.893267-07:00 |
| nfl_sportsgameodds_events_raw | lock | 3 | 2026-10-22 06:33:18.460380-07:00 |

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
| game odds | 6 | 1 |
| player props | 212 | 18 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 2 | 0 | 0.0% |
| prop | 126 | 0 | 0.0% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 212 | raw parser output |
| true paired over/under offers | 212 | 100.0% |
| lock markets | 126 | exact book/player/stat/line locked |
| close markets | 0 | exact book/player/stat/line closed |
| lock-close exact matches | 0 | 0.0% |
| CLV labels | 0 | valid: 0 |
| graded prop predictions | 0 | line-level result rows |
| prop ledger rows | 54 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| none | 0 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| game | paper | 2 |
| prop | paper | 54 |
