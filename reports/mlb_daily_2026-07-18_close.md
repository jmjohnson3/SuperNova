# SuperNovaBets MLB Daily Run (2026-07-18 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 5.9s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 7.1s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 5.9s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 7.4s)
- **Refresh prop replay CLV**: OK (rc=0, 54.6s)
- **Build prop market training table**: OK (rc=0, 98.4s)
- **Prop walk-forward accuracy report**: OK (rc=0, 7.3s)
- **Prop shadow selector report**: OK (rc=0, 5.2s)
- **Prop miss diagnostic report**: OK (rc=0, 90.5s)
- **Prop bucket repair report**: OK (rc=0, 30.9s)
- **TB prop repair report**: OK (rc=0, 31.4s)
- **Prop target quality report**: OK (rc=0, 52.1s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 21.6s)
- **Grade outcomes + ledgers**: OK (rc=0, 53.3s)
- **Grade daily forecast ledger**: OK (rc=0, 6.1s)
- **Daily forecast projection audit**: OK (rc=0, 25.1s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 14.1s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.3s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.3s)
- **Prop drift guard diagnostic**: OK (rc=0, 41.5s)
- **Lock micro projection ledger**: OK (rc=0, 6.5s)
- **Prop micro ledger report**: OK (rc=0, 3.2s)
- **Prop layer promotion control**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 8.7s)
- **TB tail repair challenger**: OK (rc=0, 44.6s)
- **Prop snapshot coverage report**: FAIL (rc=124, 122.8s)
- **Real-money operational prop reports**: OK (rc=0, 30.0s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 49.7s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 9.8s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 184.4s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 1.3s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.3s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 28.4s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 5.6s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 3.5s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 0.4s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 9.8s)
- **Daily slate trust monitor**: OK (rc=0, 231.4s)
- **Grade shadow prop replay**: OK (rc=0, 46.5s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-18 21:45:13,989 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 32 teams today (2026-07-18)
2026-07-18 21:45:14,394 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-07-18
2026-07-18 21:45:14,394 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-07-18
2026-07-18 21:45:14,409 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-18 21:45:14,769 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 16 unique games for season=2026-regular
2026-07-18 21:45:14,863 | INFO | mlb_pipeline.crawler_statsapi | Upserted 16 rows into raw.mlb_games
2026-07-18 21:45:14,904 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6318 completed games, 6307 already done, 2 to fetch
2026-07-18 21:45:16,284 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 2 / 2 games fetched
2026-07-18 21:45:16,425 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=2, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-18 21:45:21,700 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-19. Catching up from 2026-07-20 to 2026-07-17
2026-07-18 21:45:21,716 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-18 window=2026-07-18T04:00:00Z..2026-07-19T04:00:00Z
2026-07-18 21:45:22,533 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-18 | events=2 | credits_remaining=59804
2026-07-18 21:45:22,700 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-19 window=2026-07-19T04:00:00Z..2026-07-20T04:00:00Z
2026-07-18 21:45:23,404 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-19 | events=15 | credits_remaining=59802
2026-07-18 21:45:23,459 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=59802
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-18 21:45:26,063 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-19. Catching up from 2026-07-20 to 2026-07-17
2026-07-18 21:45:26,648 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 2 events (game_date=2026-07-18, as_of=2026-07-18)
2026-07-18 21:45:27,325 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-18 event=ded1e582d76a9362d25395cef8921336 (Washington Nationals@Athletics) | credits=59797
2026-07-18 21:45:28,289 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-07-18 event=9933e32d7b9c200975b046306d80856d (Detroit Tigers@Los Angeles Angels) | credits=59791
2026-07-18 21:45:29,201 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-19, as_of=2026-07-18)
2026-07-18 21:45:29,311 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=59791
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-18 21:45:32,841 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1582 rows into odds.mlb_game_lines (live odds).
2026-07-18 21:45:32,872 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-18
2026-07-18 21:45:32,872 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-18 21:45:32,872 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-19T03:15:32.872497+00:00.
2026-07-18 21:45:32,981 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-17
2026-07-18 21:45:36,747 | INFO | mlb_pipeline.parse_oddsapi | Upserted 163 rows into odds.mlb_player_prop_lines.
2026-07-18 21:45:36,747 | INFO | mlb_pipeline.parse_oddsapi | Processed 163 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-07-18 21:45:36,747 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 6033,
  "run_ids": "all",
  "date_from": "2026-07-18",
  "date_to": "2026-07-18",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 0,
  "replay_rows": 14301,
  "examples": 14301
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 55.2,
  "rows": 248887,
  "graded_rows": 223325,
  "valid_clv_rows": 209991,
  "generated_at_utc": "2026-07-19T02:52:48+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 859,
  "real_candidate_rows": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_shadow_selector_latest.md"
}
```

### Prop miss diagnostic report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 23305,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 229046,
  "bucket_count": 145,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=95109
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 255009,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### FanDuel one-sided diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "root_causes": {
    "raw_api_one_sided": 15005
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 227-151 (60.1%) ROI: +14.6% | Total: 71-68 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/86 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.12 runs
MLB Price CLV Run Line: 83 bets  avg=+0.38%
```

**stderr (tail)**
```
2026-07-18 21:52:24,186 | INFO | mlb_pipeline.modeling.update_outcomes | update_game_outcomes: updated 2 rows
2026-07-18 21:52:24,186 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 2 MLB game outcome rows
2026-07-18 21:52:24,190 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-18 21:52:57,555 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 481 prop rows
2026-07-18 21:52:57,571 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 481 MLB prop outcome rows
2026-07-18 21:53:00,633 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-18 21:53:00,634 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-18 21:53:01,184 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 4 game model-pick ledger rows
2026-07-18 21:53:01,966 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 13 prop model-pick ledger rows
2026-07-18 21:53:01,982 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 17 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 756
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 16834,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 3733,
  "phase": "day_pregame",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_hitter_live_vs_legacy_forecast_diff_latest.md"
}
```

### Prop micro promotion evaluation

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "micro_ready_count": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_promotion_evaluation_latest.md"
}
```

### Prop micro gate sensitivity report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "profiles": {
    "watch_only": 0,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-18', 'micro_rows': 45, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-18",
  "active_prediction_rows": 859,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 8,
      "stake_usd": 8.0
    },
    {
      "model_tier": "watch",
      "rows": 97,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 8,
      "stake_usd": 8.0
    },
    {
      "model_tier": "watch",
      "rows": 97,
      "stake_usd": 0.0
    }
  ]
}
```

### Prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 0,
  "pending": 8,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
}
```

### Prop layer promotion control

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "auto_integrations": {
    "hitter_rate_challenger_production": {
      "enabled": false,
      "layer": "hitter_hits_hr_shadow",
      "target_mode": "production_scoring",
      "blockers": [
        "batter_hits_projection_not_better_than_baseline"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": true,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": []
    },
    "exact_bucket_micro_ladder": {
      "enabled": false,
      "layer": "exact_bucket_micro",
      "target_mode": "micro",
      "blockers": [
        "micro_ready_exact_buckets=0",
        "real_money_kill_switch_active"
      ]
    }
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_layer_promotion_latest.md"
}
```

### Forecast repair error decomposition

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "active_source": "prospective_ledger",
  "prospective_rows": 10912,
  "historical_rows": 24561,
  "tb_repair": "train_gated_direct_player_game_tb_head",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_forecast_repair_error_latest.md"
}
```

### TB tail repair challenger

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 8172,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 124

**stderr (tail)**
```
Timed out after 120s; killed process tree rooted at PID 4104
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-19T03:57:38Z",
  "slate_date": "2026-07-18",
  "lookback_days": 45,
  "reports": {
    "close_coverage_dashboard": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_close_coverage_dashboard_latest.md"
    },
    "player_game_projection_ledger_v2": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_player_game_projection_ledger_v2_latest.md"
    },
    "tb_error_decomposition": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_tb_error_decomposition_latest.md"
    },
    "true_pair_coverage_repair": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_true_pair_coverage_repair_latest.md"
    },
    "shadow_challenger_promotion": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_shadow_challenger_promotion_latest.md"
    },
    "micro_bucket_almost_there": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_micro_bucket_almost_there_latest.md"
    },
    "micro_gate_sensitivity": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
    }
  }
}
```

### Pitcher K-rate challenger diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 869,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-18",
  "evaluation_status": "provisional",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8242378622506586,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 9132
```

### Post-checkpoint micro promotion evaluation

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "micro_ready_count": 0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_promotion_evaluation_latest.md"
}
```

### Post-checkpoint micro gate sensitivity report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "profiles": {
    "watch_only": 0,
    "micro_test": 0,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Post-checkpoint prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-19', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-19
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4402: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-19",
  "active_prediction_rows": 0,
  "locked_rows_attempted": 0,
  "ledger_before": [],
  "ledger_after": []
}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-19
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4402: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 0,
  "pending": 8,
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_micro_ledger_report.json",
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_ledger_latest.md"
}
```

### Post-checkpoint prop layer promotion control

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "auto_integrations": {
    "hitter_rate_challenger_production": {
      "enabled": false,
      "layer": "hitter_hits_hr_shadow",
      "target_mode": "production_scoring",
      "blockers": [
        "batter_hits_projection_not_better_than_baseline"
      ]
    },
    "hitter_pa_v3_production": {
      "enabled": true,
      "layer": "hitter_pa_v3",
      "target_mode": "production_scoring",
      "blockers": []
    },
    "exact_bucket_micro_ladder": {
      "enabled": false,
      "layer": "exact_bucket_micro",
      "target_mode": "micro",
      "blockers": [
        "micro_ready_exact_buckets=0",
        "real_money_kill_switch_active"
      ]
    }
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_layer_promotion_latest.md"
}
```

### Post-checkpoint real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-19T04:02:49Z",
  "slate_date": "2026-07-19",
  "lookback_days": 45,
  "reports": {
    "close_coverage_dashboard": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_close_coverage_dashboard_latest.md"
    },
    "player_game_projection_ledger_v2": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_player_game_projection_ledger_v2_latest.md"
    },
    "tb_error_decomposition": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_tb_error_decomposition_latest.md"
    },
    "true_pair_coverage_repair": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_true_pair_coverage_repair_latest.md"
    },
    "shadow_challenger_promotion": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_shadow_challenger_promotion_latest.md"
    },
    "micro_bucket_almost_there": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_micro_bucket_almost_there_latest.md"
    },
    "micro_gate_sensitivity": {
      "status": "ready",
      "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
    }
  }
}
```

### Daily slate trust monitor

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-18",
  "status": "provisional",
  "decision": "wait_for_final_results_or_repair_failed_checks",
  "failures": [
    "all_games_finalized",
    "valid_close_coverage",
    "targeted_close_captures",
    "hitter_anchor_settled_or_voided",
    "pitcher_anchor_settled_or_voided"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_daily_slate_trust_latest.md"
}
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 915,
  "run_ids": "all_pending",
  "regrade": false
}
```
