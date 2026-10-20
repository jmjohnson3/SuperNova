# NFL Readiness Report - 2026-10-18

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: player prop rows parsed
- Game lock/close coverage: 0.0%
- Prop lock/close coverage: 0.0%
- True-paired prop coverage: 100.0%
- Game predictions: 24 | Prop predictions: 697
- Ledger rows: 274 | CLV rows: 0
- Valid prop CLV capture: 0.0%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 918

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 15882 rows |
| Depth context | pass | 472884 rows |
| Injury context | pass | 4194 rows |
| Snap context | pass | 100.0% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | pass | 24 rows |
| Prop odds parsed | pass | 590 rows; player prop rows parsed |
| Game lock/close | watch | 0.0% |
| Prop lock/close | watch | 0.0% |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 918 |
| Game predictions | pass | 24 rows |
| Prop projections | pass | 697 rows |
| Ledger | pass | 274 rows |
| CLV | watch | 0 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 15882 | 33 | 2026-10-16 06:32:37.767142-07:00 |
| depth 2026 | 472884 | 32 | 2026-10-16 05:16:59-07:00 |
| injury 2026 | 4194 | 32 | 2026-10-16 06:34:32.504320-07:00 |

## Player Usage Coverage

| Field | Rows | Coverage |
|---|---:|---:|
| offense snap share | 1821 | 100.0% |
| true routes run | 0 | 0.0% |
| pass-route opportunities | 0 | 0.0% |
| pass-route opportunity share | 0 | 0.0% |
| route participation | 0 | 0.0% |
| route participation proxy, training 2026 | 1724 | 100.0% |
| estimated routes proxy, training 2026 | 1636 | 94.9% |
| starter confidence, training 2026 | 1724 | 100.0% |
| rest-risk score, training 2026 | 1724 | 100.0% |
| receiving usage quality v4, training 2026 | 1724 | 100.0% |
| RB usage quality v4, training 2026 | 1724 | 100.0% |
| TD usage quality v4, training 2026 | 1724 | 100.0% |
| live usage context quality v4, training 2026 | 1724 | 100.0% |
| receiver spike correction v4, training 2026 | 1724 | 100.0% |
| RB spike correction v4, training 2026 | 1724 | 100.0% |
| target share | 1811 | 99.5% |
| air-yards share | 1811 | 99.5% |
| WOPR | 1811 | 99.5% |
| first-read targets | 0 | 0.0% |
| end-zone targets | 1386 | 76.1% |
| red-zone touches | 1821 | 100.0% |
| goal-line carries | 1821 | 100.0% |
| goal-line targets | 1821 | 100.0% |

## Raw Odds Payloads

| Endpoint | Role | Payloads | Latest Fetch |
|---|---|---:|---|
| nfl_game_odds | lock | 1 | 2026-10-16 06:34:43.076975-07:00 |
| nfl_player_props | lock | 1 | 2026-10-16 06:34:43.136751-07:00 |
| nfl_sportsgameodds_events_raw | lock | 1 | 2026-10-16 06:34:41.466902-07:00 |

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
| game odds | 24 | 12 |
| player props | 590 | 138 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 24 | 0 | 0.0% |
| prop | 590 | 0 | 0.0% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 590 | raw parser output |
| true paired over/under offers | 590 | 100.0% |
| lock markets | 590 | exact book/player/stat/line locked |
| close markets | 0 | exact book/player/stat/line closed |
| lock-close exact matches | 0 | 0.0% |
| CLV labels | 0 | valid: 0 |
| graded prop predictions | 0 | line-level result rows |
| prop ledger rows | 250 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| none | 0 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| game | paper | 24 |
| prop | paper | 250 |
