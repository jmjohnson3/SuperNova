# SuperNovaBets MLB Daily Run (2026-07-19 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 5.7s)
- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 5.7s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 4.3s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 4.2s)
- **Refresh prop replay CLV**: OK (rc=0, 19.0s)
- **Build prop market training table**: OK (rc=0, 130.2s)
- **Prop walk-forward accuracy report**: OK (rc=0, 16.8s)
- **Prop shadow selector report**: OK (rc=0, 4.2s)
- **Prop miss diagnostic report**: OK (rc=0, 97.6s)
- **Prop bucket repair report**: OK (rc=0, 34.8s)
- **TB prop repair report**: OK (rc=0, 31.7s)
- **Prop target quality report**: OK (rc=0, 51.9s)
- **FanDuel one-sided diagnostic**: OK (rc=0, 21.6s)
- **Grade outcomes + ledgers**: OK (rc=0, 17.7s)
- **Grade daily forecast ledger**: OK (rc=0, 5.2s)
- **Daily forecast projection audit**: OK (rc=0, 17.7s)
- **Hitter live-vs-legacy forecast diff**: OK (rc=0, 9.0s)
- **Prop micro promotion evaluation**: OK (rc=0, 0.4s)
- **Prop micro gate sensitivity report**: OK (rc=0, 0.3s)
- **Prop drift guard diagnostic**: OK (rc=0, 19.6s)
- **Lock micro projection ledger**: OK (rc=0, 6.0s)
- **Prop micro ledger report**: OK (rc=0, 3.3s)
- **Prop layer promotion control**: OK (rc=0, 0.3s)
- **Forecast repair error decomposition**: OK (rc=0, 12.1s)
- **TB tail repair challenger**: OK (rc=0, 34.1s)
- **Prop snapshot coverage report**: FAIL (rc=124, 123.8s)
- **Real-money operational prop reports**: OK (rc=0, 38.3s)
- **Pitcher K-rate challenger diagnostic**: OK (rc=0, 57.0s)
- **End-of-slate prop close diagnostic**: OK (rc=0, 9.3s)
- **Frozen-release five-date checkpoint**: FAIL (rc=124, 184.4s)
- **Post-checkpoint micro promotion evaluation**: OK (rc=0, 5.5s)
- **Post-checkpoint micro gate sensitivity report**: OK (rc=0, 0.5s)
- **Post-checkpoint prop drift guard diagnostic**: OK (rc=0, 37.8s)
- **Post-checkpoint lock micro projection ledger**: OK (rc=0, 5.4s)
- **Post-checkpoint prop micro ledger report**: OK (rc=0, 7.2s)
- **Post-checkpoint prop layer promotion control**: OK (rc=0, 1.5s)
- **Post-checkpoint real-money operational prop reports**: OK (rc=0, 27.5s)
- **Daily slate trust monitor**: OK (rc=0, 436.6s)
- **Grade shadow prop replay**: OK (rc=0, 38.2s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-07-19 21:45:09,954 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 32 teams today (2026-07-19)
2026-07-19 21:45:10,339 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 64 rows for 2026-07-19
2026-07-19 21:45:10,339 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 64 assignments for 2026-07-19
2026-07-19 21:45:10,386 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-07-19 21:45:10,749 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 16 unique games for season=2026-regular
2026-07-19 21:45:10,950 | INFO | mlb_pipeline.crawler_statsapi | Upserted 16 rows into raw.mlb_games
2026-07-19 21:45:11,006 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6336 completed games, 6327 already done, 0 to fetch
2026-07-19 21:45:11,153 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-19 21:45:14,825 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-20. Catching up from 2026-07-21 to 2026-07-18
2026-07-19 21:45:14,825 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-19 window=2026-07-19T04:00:00Z..2026-07-20T04:00:00Z
2026-07-19 21:45:15,709 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-19 | events=0 | credits_remaining=56878
2026-07-19 21:45:15,918 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-07-20 window=2026-07-20T04:00:00Z..2026-07-21T04:00:00Z
2026-07-19 21:45:16,769 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-07-20 | events=15 | credits_remaining=56876
2026-07-19 21:45:16,841 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=56876
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-07-19 21:45:19,840 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-20. Catching up from 2026-07-21 to 2026-07-18
2026-07-19 21:45:20,512 | INFO | mlb_pipeline.crawler_oddsapi | No events found for 2026-07-19 â€” skipping prop fetch
2026-07-19 21:45:21,044 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-07-20, as_of=2026-07-19)
2026-07-19 21:45:21,180 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=unknown
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-07-19 21:45:25,289 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1582 rows into odds.mlb_game_lines (live odds).
2026-07-19 21:45:25,301 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-07-19
2026-07-19 21:45:25,301 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-07-19 21:45:25,301 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-07-20T03:15:25.301326+00:00.
2026-07-19 21:45:25,410 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-07-18
2026-07-19 21:45:25,410 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_prop_odds snapshots found (as_of_date=None).
2026-07-19 21:45:25,410 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 5423,
  "run_ids": "all",
  "date_from": "2026-07-19",
  "date_to": "2026-07-19",
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
  "replay_rows": 19161,
  "examples": 19161
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "reused_fresh_artifact",
  "artifact": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json",
  "age_minutes": 54.8,
  "rows": 247611,
  "graded_rows": 223464,
  "valid_clv_rows": 208823,
  "generated_at_utc": "2026-07-20T02:53:07+00:00"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 160,
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
  "rows": 23354,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 235352,
  "bucket_count": 151,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=97809
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 260432,
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
    "raw_api_one_sided": 14512
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_fanduel_one_sided_diagnostic_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 228-154 (59.7%) ROI: +13.9% | Total: 71-68 (51.1%) ROI: -2.5%
MLB CLV Run Line: beat close 3/90 (3%) avg CLV=+0.10 runs | CLV Total avg=+0.12 runs
MLB Price CLV Run Line: 87 bets  avg=+0.26%
```

**stderr (tail)**
```
2026-07-19 21:52:25,794 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 25 pending predictions.
2026-07-19 21:52:25,794 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-07-19 21:52:25,810 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-07-19 21:52:29,950 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-07-19 21:52:30,027 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-07-19 21:52:30,122 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-07-19 21:52:30,122 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-07-19 21:52:30,325 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-07-19 21:52:30,913 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-07-19 21:52:30,919 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Grade daily forecast ledger

- rc: 0

**stdout (tail)**
```
{
  "schema_ready": true,
  "graded": 0
}
```

### Daily forecast projection audit

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 18815,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_daily_forecast_projection_audit_latest.md"
}
```

### Hitter live-vs-legacy forecast diff

- rc: 0

**stdout (tail)**
```
{
  "rows": 4046,
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
    "watch_only": 1,
    "micro_test": 1,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-19', 'micro_rows': 3, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

### Lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-19",
  "active_prediction_rows": 160,
  "locked_rows_attempted": 0,
  "ledger_before": [
    {
      "model_tier": "micro_projection",
      "rows": 11,
      "stake_usd": 11.0
    },
    {
      "model_tier": "watch",
      "rows": 6,
      "stake_usd": 0.0
    }
  ],
  "ledger_after": [
    {
      "model_tier": "micro_projection",
      "rows": 11,
      "stake_usd": 11.0
    },
    {
      "model_tier": "watch",
      "rows": 6,
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
  "graded": 16,
  "pending": 3,
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
  "prospective_rows": 12284,
  "historical_rows": 25248,
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
  "rows": 8399,
  "accepted": true,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_tb_tail_repair_challenger_latest.md"
}
```

### Prop snapshot coverage report

- rc: 124

**stderr (tail)**
```
Timed out after 120s; killed process tree rooted at PID 11668
```

### Real-money operational prop reports

- rc: 0

**stdout (tail)**
```
{
  "generated_at_utc": "2026-07-20T03:56:31Z",
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

### Pitcher K-rate challenger diagnostic

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 898,
  "mae_gain": 0.0,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_pitcher_k_rate_challenger_latest.md"
}
```

### End-of-slate prop close diagnostic

- rc: 0

**stdout (tail)**
```
{
  "slate_date": "2026-07-19",
  "evaluation_status": "final",
  "strict_clean_slate": false,
  "valid_close_coverage": 0.8621072088724584,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_end_of_slate_close_latest.md"
}
```

### Frozen-release five-date checkpoint

- rc: 124

**stderr (tail)**
```
Timed out after 180s; killed process tree rooted at PID 12356
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
    "watch_only": 1,
    "micro_test": 1,
    "strict_micro": 0
  },
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_micro_gate_sensitivity_latest.md"
}
```

### Post-checkpoint prop drift guard diagnostic

- rc: 0

**stdout (tail)**
```
{'status': 'ok', 'game_date': '2026-07-20', 'micro_rows': 0, 'micro_bettable_now': 0, 'json_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_drift_guard_diagnostic.json', 'report_path': 'C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_drift_guard_diagnostic_latest.md'}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-20
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4402: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint lock micro projection ledger

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "game_date": "2026-07-20",
  "active_prediction_rows": 0,
  "locked_rows_attempted": 0,
  "ledger_before": [],
  "ledger_after": []
}
```

**stderr (tail)**
```
No normalized prop offers available for 2026-07-20
C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\modeling\predict_player_props.py:4402: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
  df = pd.read_sql(SQL_PROP_LINES, conn, params={"game_date": game_date})
```

### Post-checkpoint prop micro ledger report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "graded": 16,
  "pending": 3,
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
  "generated_at_utc": "2026-07-20T04:02:09Z",
  "slate_date": "2026-07-20",
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
  "slate_date": "2026-07-19",
  "status": "fail",
  "decision": "do_not_use_slate_for_clean_promotion_evidence",
  "failures": [
    "valid_close_coverage",
    "targeted_close_captures"
  ],
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_daily_slate_trust_latest.md"
}
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 414,
  "run_ids": "all_pending",
  "regrade": false
}
```
