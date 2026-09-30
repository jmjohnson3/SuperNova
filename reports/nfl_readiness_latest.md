# NFL Readiness Report - 2026-09-30

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: no player prop payloads captured
- Game lock/close coverage: 0.0%
- Prop lock/close coverage: 0.0%
- True-paired prop coverage: 0.0%
- Game predictions: 0 | Prop predictions: 0
- Ledger rows: 0 | CLV rows: 0
- Valid prop CLV capture: 0.0%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 861

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 8586 rows |
| Depth context | pass | 424530 rows |
| Injury context | pass | 1293 rows |
| Snap context | pass | 99.9% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | fail | 0 rows |
| Prop odds parsed | watch | 0 rows; no player prop payloads captured |
| Game lock/close | watch | 0.0% |
| Prop lock/close | watch | no locked prop markets yet |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 861 |
| Game predictions | watch | 0 rows |
| Prop projections | watch | 0 rows |
| Ledger | watch | 0 rows |
| CLV | watch | 0 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 8586 | 33 | 2026-09-29 02:01:04.608375-07:00 |
| depth 2026 | 424530 | 32 | 2026-09-28 23:02:44-07:00 |
| injury 2026 | 1293 | 32 | 2026-09-27 06:32:12.532460-07:00 |

## Player Usage Coverage

| Field | Rows | Coverage |
|---|---:|---:|
| offense snap share | 1082 | 99.9% |
| true routes run | 0 | 0.0% |
| pass-route opportunities | 0 | 0.0% |
| pass-route opportunity share | 0 | 0.0% |
| route participation | 0 | 0.0% |
| route participation proxy, training 2026 | 1052 | 100.0% |
| estimated routes proxy, training 2026 | 977 | 92.9% |
| starter confidence, training 2026 | 1052 | 100.0% |
| rest-risk score, training 2026 | 1052 | 100.0% |
| receiving usage quality v4, training 2026 | 1052 | 100.0% |
| RB usage quality v4, training 2026 | 1052 | 100.0% |
| TD usage quality v4, training 2026 | 1052 | 100.0% |
| live usage context quality v4, training 2026 | 1052 | 100.0% |
| receiver spike correction v4, training 2026 | 1052 | 100.0% |
| RB spike correction v4, training 2026 | 1052 | 100.0% |
| target share | 1078 | 99.5% |
| air-yards share | 1078 | 99.5% |
| WOPR | 1078 | 99.5% |
| first-read targets | 0 | 0.0% |
| end-zone targets | 829 | 76.5% |
| red-zone touches | 1083 | 100.0% |
| goal-line carries | 1083 | 100.0% |
| goal-line targets | 1083 | 100.0% |

## Raw Odds Payloads

| Endpoint | Role | Payloads | Latest Fetch |
|---|---|---:|---|
| none | - | 0 | - |

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
| game odds | 0 | 0 |
| player props | 0 | 0 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 0 | 0 | 0.0% |
| prop | 0 | 0 | 0.0% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 0 | raw parser output |
| true paired over/under offers | 0 | 0.0% |
| lock markets | 0 | exact book/player/stat/line locked |
| close markets | 0 | exact book/player/stat/line closed |
| lock-close exact matches | 0 | 0.0% |
| CLV labels | 0 | valid: 0 |
| graded prop predictions | 0 | line-level result rows |
| prop ledger rows | 0 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| none | 0 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| none | - | 0 |
