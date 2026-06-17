# SuperNovaBets MLB Daily Run (2026-06-17 ET)

## Summary

- **Re-crawl closing game odds (Odds API)**: OK (rc=0, 6.3s)
- **Re-crawl closing prop odds (Odds API)**: OK (rc=0, 26.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 19.3s)
- **Refresh prop replay CLV**: OK (rc=0, 8.0s)
- **Build prop market training table**: OK (rc=0, 270.6s)
- **Prop walk-forward accuracy report**: OK (rc=0, 165.6s)
- **Prop shadow selector report**: OK (rc=0, 83.0s)
- **Prop miss diagnostic report**: OK (rc=0, 30.0s)
- **Prop bucket repair report**: OK (rc=0, 10.6s)
- **TB prop repair report**: OK (rc=0, 8.3s)
- **Prop target quality report**: OK (rc=0, 14.5s)
- **Grade outcomes + ledgers**: OK (rc=0, 6.2s)
- **Prop snapshot coverage report**: OK (rc=0, 62.0s)
- **Grade shadow prop replay**: OK (rc=0, 15.3s)

## Outputs (tails)

### Re-crawl closing game odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-17 11:45:10,236 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-18. Catching up from 2026-06-19 to 2026-06-16
2026-06-17 11:45:10,250 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-17 window=2026-06-17T04:00:00Z..2026-06-18T04:00:00Z
2026-06-17 11:45:11,019 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-17 | events=15 | credits_remaining=77964
2026-06-17 11:45:11,294 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-18 window=2026-06-18T04:00:00Z..2026-06-19T04:00:00Z
2026-06-17 11:45:11,839 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-18 | events=8 | credits_remaining=77962
2026-06-17 11:45:11,851 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=77962
```

### Re-crawl closing prop odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-17 11:45:15,811 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-18. Catching up from 2026-06-19 to 2026-06-16
2026-06-17 11:45:17,169 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 15 events (game_date=2026-06-17, as_of=2026-06-17)
2026-06-17 11:45:17,779 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=93a412d9ba2735490b741d924fea7ef1 (New York Mets@Cincinnati Reds) | credits=77956
2026-06-17 11:45:18,773 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=bab3782d628eabfc050d79656a55fa50 (Kansas City Royals@Washington Nationals) | credits=77950
2026-06-17 11:45:19,669 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=70301ecea46dd4e6c12d0c003013a4ee (Miami Marlins@Philadelphia Phillies) | credits=77944
2026-06-17 11:45:20,628 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=aca8c666c9ae92e799b0fc96c3672aab (San Francisco Giants@Atlanta Braves) | credits=77941
2026-06-17 11:45:21,520 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=db80e9d6153bc82e7a3612d55fd10ece (Detroit Tigers@Houston Astros) | credits=77935
2026-06-17 11:45:22,402 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=c8afc0ecd9f70b910f3f7ffbf71274e4 (San Diego Padres@St. Louis Cardinals) | credits=77929
2026-06-17 11:45:23,305 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=a75cee44d2cbdcedb356612df99cd692 (Tampa Bay Rays@Los Angeles Dodgers) | credits=77923
2026-06-17 11:45:24,171 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=44a3fa0a32ffae8dd220f1a4d54ae239 (Los Angeles Angels@Arizona Diamondbacks) | credits=77917
2026-06-17 11:45:24,989 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=1b3b5ac473b471165355ca855853dbb2 (Toronto Blue Jays@Boston Red Sox) | credits=77911
2026-06-17 11:45:25,890 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=9cfb18a4e0091ae6c9ccb03c48deb277 (Chicago White Sox@New York Yankees) | credits=77905
2026-06-17 11:45:26,781 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=a834545628bda1d53eb52e89728b336f (San Francisco Giants@Atlanta Braves) | credits=77899
2026-06-17 11:45:27,668 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=4d8279537f3ea7c4d1d53160da26d2c6 (Cleveland Guardians@Milwaukee Brewers) | credits=77893
2026-06-17 11:45:28,555 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=3c667b5e84c7e18c7697c0f83852679a (Colorado Rockies@Chicago Cubs) | credits=77887
2026-06-17 11:45:29,845 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=ab9ae62c321b17a5d765118cc9bb3fbf (Pittsburgh Pirates@Athletics) | credits=77881
2026-06-17 11:45:30,705 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=1d049adfd45ba69776089a9846e60034 (Baltimore Orioles@Seattle Mariners) | credits=77875
2026-06-17 11:45:31,532 | INFO | mlb_pipeline.crawler_oddsapi | Fetching prop lines for 8 events (game_date=2026-06-18, as_of=2026-06-17)
2026-06-17 11:45:32,382 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=10b797abe19952451af8a428ab530deb (Toronto Blue Jays@Boston Red Sox) | credits=77875
2026-06-17 11:45:33,229 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=872e319f5c105d18457014bc855cdc85 (Cleveland Guardians@Milwaukee Brewers) | credits=77875
2026-06-17 11:45:34,115 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=bb47b41c0a375d2df45331898a1e7ec1 (Minnesota Twins@Texas Rangers) | credits=77875
2026-06-17 11:45:34,927 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=cad6df1fc1aadca9fdbb359d7d43e40f (Baltimore Orioles@Seattle Mariners) | credits=77875
2026-06-17 11:45:35,756 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=7fc018ccf3790baea15d29b581f9d0ce (New York Mets@Philadelphia Phillies) | credits=77875
2026-06-17 11:45:36,547 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=adfa73cc70f5a62d6d0bb903eaa0f24b (Chicago White Sox@New York Yankees) | credits=77875
2026-06-17 11:45:37,355 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=e92122c8f905eb41a57faabf21daf468 (San Francisco Giants@Atlanta Braves) | credits=77875
2026-06-17 11:45:38,155 | INFO | mlb_pipeline.crawler_oddsapi | Prop odds as_of=2026-06-17 event=13234b203ca4ea57ec2f1381b3e6a046 (St. Louis Cardinals@Kansas City Royals) | credits=77875
2026-06-17 11:45:38,406 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=0 skipped=0 credits_remaining=77875
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-17 11:45:42,522 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1447 rows into odds.mlb_game_lines (live odds).
2026-06-17 11:45:42,538 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-17
2026-06-17 11:45:42,539 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-17 11:45:42,539 | INFO | mlb_pipeline.parse_oddsapi | Only assigning close prop snapshots to raw payloads fetched since 2026-06-17T17:15:42.539656+00:00.
2026-06-17 11:45:42,670 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-16
2026-06-17 11:45:57,756 | INFO | mlb_pipeline.parse_oddsapi | Upserted 2569 rows into odds.mlb_player_prop_lines.
2026-06-17 11:45:57,756 | INFO | mlb_pipeline.parse_oddsapi | Processed 2569 immutable close prop snapshot observations for odds.mlb_player_prop_line_snapshots.
2026-06-17 11:45:57,756 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Refresh prop replay CLV

- rc: 0

**stdout (tail)**
```
{
  "refreshed_rows": 2402,
  "run_ids": "all",
  "date_from": "2026-06-17",
  "date_to": "2026-06-17",
  "include_graded": true,
  "only_missing": false
}
```

### Build prop market training table

- rc: 0

**stdout (tail)**
```
{
  "deleted": 82088,
  "replay_rows": 82088,
  "examples": 82088
}
```

### Prop walk-forward accuracy report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 83646,
  "graded_rows": 65898,
  "valid_clv_rows": 67357,
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
  "active_rows": 2402,
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
  "rows": 65072,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_miss_diagnostic_latest.md"
}
```

### Prop bucket repair report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 65072,
  "bucket_count": 132,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_bucket_repair_latest.md"
}
```

### TB prop repair report

- rc: 0

**stdout (tail)**
```
TB repair report status=ok rows=25882
```

### Prop target quality report

- rc: 0

**stdout (tail)**
```
{
  "status": "ready",
  "rows": 82745,
  "report_path": "C:\\Users\\josh\\Git\\SuperNovaBets\\reports\\mlb_prop_target_quality_latest.md"
}
```

### Grade outcomes + ledgers

- rc: 0

**stdout (tail)**
```
MLB Run Line: 192-129 (59.8%) ROI: +14.2% | Total: 61-57 (51.7%) ROI: -1.3%
MLB CLV Run Line: beat close 2/30 (7%) avg CLV=+0.20 runs | CLV Total avg=+0.02 runs
MLB Price CLV Run Line: 28 bets  avg=+0.57%
```

**stderr (tail)**
```
2026-06-17 11:55:52,202 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 33 pending predictions.
2026-06-17 11:55:52,202 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-17 11:55:52,260 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: nothing to update.
2026-06-17 11:55:53,540 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-17 11:55:53,559 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-17 11:55:53,683 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-17 11:55:53,684 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-17 11:55:53,812 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-17 11:55:54,641 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-17 11:55:54,649 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```

### Prop snapshot coverage report

- rc: 0

**stdout (tail)**
```
# MLB Prop Snapshot Coverage

Generated: 2026-06-17T13:56:56-04:00
Range: 2026-06-04 to 2026-06-17

## Collection Status

**TARGET MET**

- Clean shadow slates: 12 / 10
- Additional clean slates needed: 0
- A clean slate needs at least 100 side locks, 100 valid exact side closes, and 25.0% valid-close coverage.
- Missing-lock rate must be <= 2.0%; stale-close-before-lock rate must be <= 5.0%.
- A valid close must be the same event, book, player, stat, side, and exact line; after lock; and within two hours of first pitch.

## Slate Coverage

| Date | Clean | Offers | Open | Locks | Side Locks | Close Obs | Valid Side Locks | Coverage | Missing Lock | Stale Close | Lock Phases | Close Times | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-17 | yes | 2924 | 0 | 2402 | 2402 | 17744 | 1132 | 47.1% | 0.0% | 2.2% | 1 | 7 |  |
| 2026-06-16 | yes | 4276 | 2600 | 8262 | 8262 | 38949 | 7125 | 86.2% | 0.0% | 1.8% | 3 | 17 |  |
| 2026-06-15 | yes | 2927 | 1891 | 5873 | 5873 | 28077 | 5221 | 88.9% | 0.0% | 2.9% | 3 | 18 |  |
| 2026-06-14 | yes | 4648 | 2741 | 6056 | 6056 | 24201 | 5004 | 82.6% | 0.0% | 3.4% | 3 | 16 |  |
| 2026-06-13 | yes | 4564 | 2780 | 6581 | 6581 | 37746 | 5656 | 85.9% | 0.0% | 1.2% | 3 | 19 |  |
| 2026-06-12 | yes | 4438 | 2804 | 8682 | 8682 | 40945 | 7706 | 88.8% | 0.0% | 2.5% | 3 | 17 |  |
| 2026-06-11 | no | 2452 | 1703 | 2202 | 2202 | 14587 | 1879 | 85.3% | 0.0% | 6.7% | 2 | 15 | stale_close_rate>0.05 |
| 2026-06-10 | yes | 4550 | 3350 | 8414 | 8414 | 40087 | 7198 | 85.5% | 0.0% | 4.5% | 3 | 18 |  |
| 2026-06-09 | yes | 4402 | 3237 | 6566 | 6566 | 40669 | 5702 | 86.8% | 0.0% | 3.5% | 2 | 16 |  |
| 2026-06-08 | yes | 2447 | 1705 | 3589 | 3589 | 27069 | 3093 | 86.2% | 0.0% | 3.1% | 2 | 19 |  |
| 2026-06-07 | yes | 4496 | 3136 | 6521 | 6521 | 30276 | 5638 | 86.5% | 0.0% | 1.0% | 3 | 19 |  |
| 2026-06-06 | yes | 4345 | 3180 | 4562 | 4562 | 38898 | 4017 | 88.1% | 0.0% | 2.2% | 2 | 19 |  |
| 2026-06-05 | yes | 4520 | 3154 | 6699 | 6699 | 57725 | 5816 | 86.8% | 0.0% | 4.2% | 2 | 22 |  |
| 2026-06-04 | no | 2505 | 1945 | 2348 | 2348 | 7269 | 1640 | 69.8% | 0.0% | 13.7% | 2 | 10 | stale_close_rate>0.05 |
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
