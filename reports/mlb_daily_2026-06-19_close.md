# SuperNovaBets MLB Daily Run (2026-06-19 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 9.7s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 27.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 127.5s)
- **Refresh prop replay CLV**: OK (rc=0, 19.0s)
- **Build prop market training table**: OK (rc=0, 111.2s)
- **Prop walk-forward accuracy report**: OK (rc=0, 222.3s)
- **Prop shadow selector report**: OK (rc=0, 290.8s)
- **Prop miss diagnostic report**: OK (rc=0, 73.8s)
- **Prop bucket repair report**: OK (rc=0, 21.2s)
- **TB prop repair report**: OK (rc=0, 18.2s)
- **Prop target quality report**: OK (rc=0, 18.2s)
- **Grade outcomes + ledgers**: OK (rc=0, 8.6s)
- **Prop snapshot coverage report**: OK (rc=0, 21.4s)
- **Grade shadow prop replay**: OK (rc=0, 7.6s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-19 12:47:08,641 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-20. Catching up from 2026-06-21 to 2026-06-18
2026-06-19 12:47:08,672 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-19 window=2026-06-19T04:00:00Z..2026-06-20T04:00:00Z
2026-06-19 12:47:09,998 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-19 | events=14 | credits_remaining=76095
2026-06-19 12:47:10,515 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-20 window=2026-06-20T04:00:00Z..2026-06-21T04:00:00Z
2026-06-19 12:47:11,142 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-20 | events=11 | credits_remaining=76093
2026-06-19 12:47:11,145 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=76093
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-19 12:47:14,354 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-20. Catching up from 2026-06-21 to 2026-06-18
2026-06-19 12:47:15,011 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 14 events (game_date=2026-06-19, as_of=2026-06-19)
2026-06-19 12:47:15,566 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=3530063d3d9cd6eb5811962deca7c6ed (Toronto Blue Jays@Chicago Cubs) | credits=76087
2026-06-19 12:47:16,501 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=2969ea2e73c33f58b97b82fe562cf617 (Chicago White Sox@Detroit Tigers) | credits=76081
2026-06-19 12:47:17,346 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=7ef2ffad2e845a48d3df860e1a91b43d (Cincinnati Reds@New York Yankees) | credits=76075
2026-06-19 12:47:18,258 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=6bb329abaf86e99a6c742250c3251cec (San Francisco Giants@Miami Marlins) | credits=76069
2026-06-19 12:47:19,254 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=bdebd82757453b3cdc93ee168dadf2bb (Washington Nationals@Tampa Bay Rays) | credits=76063
2026-06-19 12:47:20,216 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=2bbd4e284d32261d81b8ebd0263116c3 (Milwaukee Brewers@Atlanta Braves) | credits=76057
2026-06-19 12:47:21,104 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=a072cb316f0d887f1c0251b417b34317 (San Diego Padres@Texas Rangers) | credits=76051
2026-06-19 12:47:21,941 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=d06fe764bea53051487b94fc022806d1 (Cleveland Guardians@Houston Astros) | credits=76045
2026-06-19 12:47:22,859 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=828d9387e45dac20bd4336888bc32025 (St. Louis Cardinals@Kansas City Royals) | credits=76039
2026-06-19 12:47:23,701 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=07f4bbde58b19584048fd125a63efb85 (Pittsburgh Pirates@Colorado Rockies) | credits=76033
2026-06-19 12:47:24,560 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=a8bc26eb19d767d6433749f35c2224ee (Los Angeles Angels@Athletics) | credits=76027
2026-06-19 12:47:25,507 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=68137f4094d6c3979398139c92d61aca (Minnesota Twins@Arizona Diamondbacks) | credits=76021
2026-06-19 12:47:26,369 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=150da5f2fa6db9e963e0b56fc2863769 (Baltimore Orioles@Los Angeles Dodgers) | credits=76015
2026-06-19 12:47:27,259 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=5bc4e94f9d24204a3f0dd424ced4a8be (Boston Red Sox@Seattle Mariners) | credits=76009
2026-06-19 12:47:28,139 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 11 events (game_date=2026-06-20, as_of=2026-06-19)
2026-06-19 12:47:29,123 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=f34c6b3b5d4542734f725cdbf1200188 (Cincinnati Reds@New York Yankees) | credits=76009
2026-06-19 12:47:30,161 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=afc026eb30e790d723cc4d99d1551154 (Toronto Blue Jays@Chicago Cubs) | credits=76009
2026-06-19 12:47:30,975 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=3efd1785852b80caaeeb2e4b8e8db0f2 (San Diego Padres@Texas Rangers) | credits=76009
2026-06-19 12:47:31,820 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=954f9616bdfe5989a3ed4cc8cc1b3472 (Milwaukee Brewers@Atlanta Braves) | credits=76009
2026-06-19 12:47:32,727 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=a36cc8e7c18477f87bd91d87f8839f49 (San Francisco Giants@Miami Marlins) | credits=76009
2026-06-19 12:47:33,598 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=d0bf798e1c85fe47d8160217d79b5a3e (Cleveland Guardians@Houston Astros) | credits=76009
2026-06-19 12:47:34,479 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=6344d2ce9df189689ed256d05cfcffb4 (New York Mets@Philadelphia Phillies) | credits=76009
2026-06-19 12:47:35,362 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=4bb2c510f37ac3221423f6fe96d81822 (Pittsburgh Pirates@Colorado Rockies) | credits=76009
2026-06-19 12:47:36,191 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=bc0eb3e2abd5d0ddb3e22dc71abf652a (Minnesota Twins@Arizona Diamondbacks) | credits=76009
2026-06-19 12:47:37,094 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=8e059160b850333f99e5414be7fede74 (Baltimore Orioles@Los Angeles Dodgers) | credits=76009
2026-06-19 12:47:37,944 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-19 event=c17cdf3ec5c9c493c782139a03e71bcc (Boston Red Sox@Seattle Mariners) | credits=76009
2026-06-19 12:47:38,678 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=76009
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-19 12:47:47,406 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1455 rows into odds.mlb_game_lines (live odds).
2026-06-19 12:47:47,459 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-19
2026-06-19 12:47:47,468 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-19 12:47:47,468 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-19T18:17:47.468178+00:00.
2026-06-19 12:47:47,583 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-18
2026-06-19 12:48:47,492 | INFO | mlb_pipeline.parse_oddsapi | Upserted 2347 rows into odds.mlb_player_prop_lines.
2026-06-19 12:48:47,494 | INFO | mlb_pipeline.parse_oddsapi | Processed 2347 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-19 12:48:47,496 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 4944,
  "run_ids": "all",
  "date_from": "2026-06-19",
  "date_to": "2026-06-19",
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
  "replay_rows": 19573,
  "examples": 19573
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 92555,
  "graded_rows": 79111,
  "valid_clv_rows": 71853,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_walk_forward_accuracy_latest.md",
  "json_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\src\\mlb_pipeline\\modeling\\models\\player_props\\prop_walk_forward_accuracy_report.json"
}
```

### Prop shadow selector report

- rc: 0

**stdout (tail)**
```
{
  "status": "ok",
  "active_rows": 2618,
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
  "rows": 78285,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 78285,
  "bucket_count": 133,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=31530
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 91654,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 193-129 (59.9%) ROI: +14.4% | Total: 64-58 (52.5%) ROI: +0.1%
MLB CLV Run Line: beat close 2/31 (6%) avg CLV=+0.19 runs | CLV Total avg=+0.00 runs
MLB Price CLV Run Line: 29 bets  avg=+0.58%
```

**stderr (tail)**
```
2026-06-19 13:02:44,808 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 33 pending predictions.
2026-06-19 13:02:44,808 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-19 13:02:44,861 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-19 13:02:48,320 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-19 13:02:48,343 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-19 13:02:48,545 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-19 13:02:48,558 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-19 13:02:48,771 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-19 13:02:49,655 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-19 13:02:49,663 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-19T15:03:10-04:00
Range: 2026-06-06 to 2026-06-19

## Collection Status

**TARGET MET**

- Clean shadow slates: 11 / 10
- Additional clean slates needed: 0
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-19 | no | 2979 | 2261 | 4944 | 4944 | 16299 | 306 | 6.2% | 0.0% | 1.3% | 2 | 7 | valid_clv_coverage<0.25 |
| 2026-06-18 | no | 2569 | 1592 | 2660 | 2660 | 14372 | 1994 | 75.0% | 0.0% | 12.2% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-17 | yes | 4267 | 380 | 3707 | 3707 | 31248 | 3328 | 89.8% | 0.0% | 3.4% | 2 | 18 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.7% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.6% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.2% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.4% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.4% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.2% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 2.8% | 2 | 19 |  |
| 2026-06-07 | yes | 4496 | 3136 | 6521 | 6521 | 30276 | 5638 | 86.5% | 0.0% | 0.9% | 3 | 19 |  |
| 2026-06-06 | yes | 4345 | 3180 | 4562 | 4562 | 38898 | 4017 | 88.1% | 0.0% | 2.0% | 2 | 19 |  |
```

### Grade shadow prop replay

- rc: 0

**stdout (tail)**
```
{
  "graded_rows": 0,
  "run_ids": "all_pending",
  "regrade": false
}
```
