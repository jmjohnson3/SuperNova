# SuperNovaBets MLB Daily Run (2026-08-20 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 12.5s)
- **Re-crawl closing game odds (Odds API)**: FAIL (rc=1, 53.2s)
- **Re-crawl closing prop odds (Odds API)**: FAIL (rc=1, 52.2s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-08-20 21:45:30,503 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 18 teams today (2026-08-20)
2026-08-20 21:45:30,850 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 36 rows for 2026-08-20
2026-08-20 21:45:30,850 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 36 assignments for 2026-08-20
2026-08-20 21:45:30,892 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-08-20 21:45:31,282 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 9 unique games for season=2026-regular
2026-08-20 21:45:31,380 | INFO | mlb_pipeline.crawler_statsapi | Upserted 9 rows into raw.mlb_games
2026-08-20 21:45:31,594 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6764 completed games, 6754 already done, 1 to fetch
2026-08-20 21:45:35,382 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 1 / 1 games fetched
2026-08-20 21:45:35,654 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=1, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 1

**stderr (tail)**
```
2026-08-20 21:45:39,319 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-08-19
2026-08-20 21:45:40,108 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-08-20 21:45:42,196 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-08-20 21:45:45,716 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-08-20 21:45:52,268 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-08-20 21:46:04,908 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 91, in _fetch_with_backoff
    raise RuntimeError(f"401 Unauthorized â€” check your API key: {url}")
RuntimeError: 401 Unauthorized â€” check your API key: https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 495, in <module>
    main()
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 434, in main
    result = _fetch_historical_day(cfg, conn, d)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 201, in _fetch_historical_day
    payload, credits_remaining = _fetch_with_backoff(cfg, url, params)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 112, in _fetch_with_backoff
    raise RuntimeError(f"Failed to fetch after retries: {url}") from last_err
RuntimeError: Failed to fetch after retries: https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds
```

### Re-crawl closing prop odds (Odds API)

- rc: 1

**stderr (tail)**
```
2026-08-20 21:46:31,398 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-08-19
2026-08-20 21:46:32,022 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-08-20 21:46:34,132 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-08-20 21:46:37,811 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-08-20 21:46:44,431 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-08-20 21:46:57,068 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 91, in _fetch_with_backoff
    raise RuntimeError(f"401 Unauthorized â€” check your API key: {url}")
RuntimeError: 401 Unauthorized â€” check your API key: https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 495, in <module>
    main()
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 434, in main
    result = _fetch_historical_day(cfg, conn, d)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 201, in _fetch_historical_day
    payload, credits_remaining = _fetch_with_backoff(cfg, url, params)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 112, in _fetch_with_backoff
    raise RuntimeError(f"Failed to fetch after retries: {url}") from last_err
RuntimeError: Failed to fetch after retries: https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds
```
