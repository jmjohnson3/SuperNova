# NFL Readiness Report - 2026-10-22

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: player prop rows parsed
- Game lock/close coverage: 0.0%
- Prop lock/close coverage: 0.0%
- True-paired prop coverage: 100.0%
- Game predictions: 4 | Prop predictions: 240
- Ledger rows: 38 | CLV rows: 0
- Valid prop CLV capture: 0.0%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 918

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 18151 rows |
| Depth context | pass | 489101 rows |
| Injury context | pass | 4812 rows |
| Snap context | pass | 100.0% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | pass | 4 rows |
| Prop odds parsed | pass | 134 rows; player prop rows parsed |
| Game lock/close | watch | 0.0% |
| Prop lock/close | watch | 0.0% |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 918 |
| Game predictions | pass | 4 rows |
| Prop projections | pass | 240 rows |
| Ledger | pass | 38 rows |
| CLV | watch | 0 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 18151 | 33 | 2026-10-21 06:30:38.190473-07:00 |
| depth 2026 | 489101 | 32 | 2026-10-21 05:53:27-07:00 |
| injury 2026 | 4812 | 32 | 2026-10-21 06:32:27.854107-07:00 |

## Player Usage Coverage

| Field | Rows | Coverage |
|---|---:|---:|
| offense snap share | 2120 | 100.0% |
| true routes run | 0 | 0.0% |
| pass-route opportunities | 0 | 0.0% |
| pass-route opportunity share | 0 | 0.0% |
| route participation | 0 | 0.0% |
| route participation proxy, training 2026 | 2027 | 100.0% |
| estimated routes proxy, training 2026 | 1937 | 95.6% |
| starter confidence, training 2026 | 2027 | 100.0% |
| rest-risk score, training 2026 | 2027 | 100.0% |
| receiving usage quality v4, training 2026 | 2027 | 100.0% |
| RB usage quality v4, training 2026 | 2027 | 100.0% |
| TD usage quality v4, training 2026 | 2027 | 100.0% |
| live usage context quality v4, training 2026 | 2027 | 100.0% |
| receiver spike correction v4, training 2026 | 2027 | 100.0% |
| RB spike correction v4, training 2026 | 2027 | 100.0% |
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
| nfl_game_odds | lock | 2 | 2026-10-21 06:32:35.693589-07:00 |
| nfl_player_props | lock | 2 | 2026-10-21 06:32:35.724903-07:00 |
| nfl_sportsgameodds_events_raw | lock | 2 | 2026-10-21 06:32:35.365528-07:00 |

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
| game odds | 4 | 1 |
| player props | 134 | 16 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 2 | 0 | 0.0% |
| prop | 98 | 0 | 0.0% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 134 | raw parser output |
| true paired over/under offers | 134 | 100.0% |
| lock markets | 98 | exact book/player/stat/line locked |
| close markets | 0 | exact book/player/stat/line closed |
| lock-close exact matches | 0 | 0.0% |
| CLV labels | 0 | valid: 0 |
| graded prop predictions | 0 | line-level result rows |
| prop ledger rows | 36 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| none | 0 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| game | paper | 2 |
| prop | paper | 36 |
