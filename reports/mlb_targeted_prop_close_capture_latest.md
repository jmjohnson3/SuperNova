# MLB Targeted Prop Close Capture

Generated UTC: 2026-07-30 17:35:18.514316+00:00
Status: **FAILED_FOCUS_LOW_OFFER_COUNT**
Slate: 2026-07-30
Attempts: 3 / 3

## Targets

| Event | Minutes To Start | Reason |
|---|---:|---|
| 459ad5aa37b0d7865d087c8acb7d9d4d | 45 | active_close_window |
| 4ab9265338d5ee8f4e0ecf43a83aa9d7 | 10 | active_close_window |
| c4f54051354f3d027c0f1cabfc630e52 | 30 | active_close_window |

## Attempt Quality

| Attempt | Passed | Low Events |
|---:|---|---:|
| 1 | True | 0 |
| 2 | True | 0 |
| 3 | True | 0 |

## Latest Event Counts

| Event | Fresh Rows | Required | Baseline | Passed |
|---|---:|---:|---:|---|
| 459ad5aa37b0d7865d087c8acb7d9d4d | 299 | 254 | 299 | True |
| 4ab9265338d5ee8f4e0ecf43a83aa9d7 | 303 | 258 | 303 | True |
| c4f54051354f3d027c0f1cabfc630e52 | 288 | 245 | 288 | True |

## Focus Bucket Counts

| Event | Focus | Fresh Rows | Required | Baseline | Enforced | Passed |
|---|---|---:|---:|---:|---|---|
| 459ad5aa37b0d7865d087c8acb7d9d4d | draftkings:batter_total_bases:1.5 | 7 | 10 | 10 | True | False |
| 4ab9265338d5ee8f4e0ecf43a83aa9d7 | draftkings:batter_total_bases:1.5 | 12 | 12 | 13 | True | True |
| c4f54051354f3d027c0f1cabfc630e52 | draftkings:batter_total_bases:1.5 | 4 | 10 | 5 | True | False |

## Error

```
{'message': 'Targeted close capture did not meet fresh offer-count quality thresholds.', 'last_quality': {'passed': True, 'events': [{'event_id': '459ad5aa37b0d7865d087c8acb7d9d4d', 'fresh_close_rows': 299, 'fresh_close_times': 2, 'latest_snapshot_at_utc': datetime.datetime(2026, 7, 30, 10, 37, 42, 33137, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200))), 'baseline_rows': 299, 'required_rows': 254, 'passed': True}, {'event_id': '4ab9265338d5ee8f4e0ecf43a83aa9d7', 'fresh_close_rows': 303, 'fresh_close_times': 2, 'latest_snapshot_at_utc': datetime.datetime(2026, 7, 30, 10, 37, 42, 33137, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200))), 'baseline_rows': 303, 'required_rows': 258, 'passed': True}, {'event_id': 'c4f54051354f3d027c0f1cabfc630e52', 'fresh_close_rows': 288, 'fresh_close_times': 2, 'latest_snapshot_at_utc': datetime.datetime(2026, 7, 30, 10, 37, 42, 33137, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200))), 'baseline_rows': 288, 'required_rows': 245, 'passed': True}], 'low_events': [], 'thresholds': {'min_rows_without_baseline': 50, 'min_rows_floor': 25, 'min_ratio': 0.85}}, 'last_focus_quality': {'passed': False, 'events': [{'event_id': '459ad5aa37b0d7865d087c8acb7d9d4d', 'passed': False, 'focus_rows': [{'event_id': '459ad5aa37b0d7865d087c8acb7d9d4d', 'focus': 'draftkings:batter_total_bases:1.5', 'fresh_rows': 7, 'baseline_rows': 10, 'required_rows': 10, 'enforced': True, 'passed': False}]}, {'event_id': '4ab9265338d5ee8f4e0ecf43a83aa9d7', 'passed': True, 'focus_rows': [{'event_id': '4ab9265338d5ee8f4e0ecf43a83aa9d7', 'focus': 'draftkings:batter_total_bases:1.5', 'fresh_rows': 12, 'baseline_rows': 13, 'required_rows': 12, 'enforced': True, 'passed': True}]}, {'event_id': 'c4f54051354f3d027c0f1cabfc630e52', 'passed': False, 'focus_rows': [{'event_id': 'c4f54051354f3d027c0f1cabfc630e52', 'focus': 'draftkings:batter_total_bases:1.5', 'fresh_rows': 4, 'baseline_rows': 5, 'required_rows': 10, 'enforced': True, 'passed': False}]}], 'low_events': [{'event_id': '459ad5aa37b0d7865d087c8acb7d9d4d', 'passed': False, 'focus_rows': [{'event_id': '459ad5aa37b0d7865d087c8acb7d9d4d', 'focus': 'draftkings:batter_total_bases:1.5', 'fresh_rows': 7, 'baseline_rows': 10, 'required_rows': 10, 'enforced': True, 'passed': False}]}, {'event_id': 'c4f54051354f3d027c0f1cabfc630e52', 'passed': False, 'focus_rows': [{'event_id': 'c4f54051354f3d027c0f1cabfc630e52', 'focus': 'draftkings:batter_total_bases:1.5', 'fresh_rows': 4, 'baseline_rows': 5, 'required_rows': 10, 'enforced': True, 'passed': False}]}], 'focus_specs': [{'bookmaker_key': 'draftkings', 'stat': 'batter_total_bases', 'line': 1.5, 'label': 'draftkings:batter_total_bases:1.5'}], 'thresholds': {'min_rows_floor': 10, 'min_ratio': 0.9}}}
```
