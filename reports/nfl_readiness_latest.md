# NFL Readiness Report - 2026-11-09

NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.

## Summary

- Prop market status: player prop rows parsed
- Game lock/close coverage: 100.0%
- Prop lock/close coverage: 70.5%
- True-paired prop coverage: 42.0%
- Game predictions: 4 | Prop predictions: 112
- Ledger rows: 46 | CLV rows: 62
- Valid prop CLV capture: 79.3%
- Exact-line prop model status: insufficient_independent_history | true-paired rows: 918

## Daily Trust Checks

| Check | Status | Detail |
|---|---|---|
| Roster context | pass | 22852 rows |
| Depth context | pass | 547234 rows |
| Injury context | pass | 7658 rows |
| Snap context | pass | 100.0% coverage |
| Route proxy context | pass | 100.0% coverage from training season 2026 |
| Red-zone context | pass | 100.0% coverage |
| Game odds parsed | pass | 10 rows |
| Prop odds parsed | pass | 3256 rows; player prop rows parsed |
| Game lock/close | pass | 100.0% |
| Prop lock/close | watch | 70.5% |
| Exact-line prop models | watch | insufficient_independent_history; true-paired rows: 918 |
| Game predictions | pass | 4 rows |
| Prop projections | pass | 112 rows |
| Ledger | pass | 46 rows |
| CLV | pass | 62 rows |

## Context Freshness

| Data | Rows | Teams | Latest Timestamp |
|---|---:|---:|---|
| roster 2026 | 22852 | 33 | 2026-11-09 06:30:59.318822-08:00 |
| depth 2026 | 547234 | 32 | 2026-11-09 09:24:57-08:00 |
| injury 2026 | 7658 | 32 | 2026-11-09 15:18:48.641168-08:00 |

## Player Usage Coverage

| Field | Rows | Coverage |
|---|---:|---:|
| offense snap share | 3079 | 100.0% |
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
| target share | 3040 | 98.7% |
| air-yards share | 3040 | 98.7% |
| WOPR | 3040 | 98.7% |
| first-read targets | 0 | 0.0% |
| end-zone targets | 2310 | 75.0% |
| red-zone touches | 3079 | 100.0% |
| goal-line carries | 3079 | 100.0% |
| goal-line targets | 3079 | 100.0% |

## Raw Odds Payloads

| Endpoint | Role | Payloads | Latest Fetch |
|---|---|---:|---|
| nfl_game_odds | close | 3 | 2026-11-09 17:10:15.026150-08:00 |
| nfl_game_odds | live | 6 | 2026-11-09 17:10:12.314424-08:00 |
| nfl_game_odds | lock | 2 | 2026-11-09 15:18:57.708372-08:00 |
| nfl_player_props | close | 4 | 2026-11-09 17:10:15.041778-08:00 |
| nfl_player_props | live | 6 | 2026-11-09 17:10:12.268976-08:00 |
| nfl_player_props | lock | 3 | 2026-11-09 15:19:06.195598-08:00 |
| nfl_sportsgameodds_events_raw | close | 3 | 2026-11-09 17:10:14.890070-08:00 |
| nfl_sportsgameodds_events_raw | lock | 2 | 2026-11-09 15:18:57.379671-08:00 |

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
| game odds | 10 | 1 |
| player props | 3256 | 21 |

## Lock To Close Coverage

| Type | Locked | Closed Same Market | Coverage |
|---|---:|---:|---:|
| game | 2 | 2 | 100.0% |
| prop | 129 | 91 | 70.5% |

## Prop Odds Proof

| Proof Layer | Rows | Coverage / Detail |
|---|---:|---|
| parsed prop offers | 3256 | raw parser output |
| true paired over/under offers | 1367 | 42.0% |
| lock markets | 129 | exact book/player/stat/line locked |
| close markets | 747 | exact book/player/stat/line closed |
| lock-close exact matches | 91 | 70.5% |
| CLV labels | 58 | valid: 46 |
| graded prop predictions | 0 | line-level result rows |
| prop ledger rows | 42 | locked paper/micro/bankroll rows |

## Prop CLV Quality

| Status | Rows |
|---|---:|
| valid_close | 46 |
| exact_line_unavailable_in_captured_feed | 12 |

## Ledger Tiers

| Source | Tier | Rows |
|---|---|---:|
| game | paper | 4 |
| prop | paper | 42 |
