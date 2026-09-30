# SuperNovaBets MLB Daily Run (2026-08-29 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 8.4s)
- **Re-crawl closing game odds (Odds API)**: FAIL (rc=1, 55.8s)
- **Re-crawl closing prop odds (Odds API)**: FAIL (rc=1, 53.7s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-08-29 21:45:19,965 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 34 teams today (2026-08-29)
2026-08-29 21:45:20,379 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 68 rows for 2026-08-29
2026-08-29 21:45:20,379 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 68 assignments for 2026-08-29
2026-08-29 21:45:20,383 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-08-29 21:45:20,753 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 17 unique games for season=2026-regular
2026-08-29 21:45:20,924 | INFO | mlb_pipeline.crawler_statsapi | Upserted 17 rows into raw.mlb_games
2026-08-29 21:45:21,013 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6885 completed games, 6876 already done, 0 to fetch
2026-08-29 21:45:21,124 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=0, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 1

**stderr (tail)**
```
2026-08-29 21:45:26,513 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-08-28
2026-08-29 21:45:27,588 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-08-29 21:45:29,726 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-08-29 21:45:33,674 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-08-29 21:45:40,319 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-08-29 21:45:52,930 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
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
2026-08-29 21:46:21,096 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-08-28
2026-08-29 21:46:21,745 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-08-29 21:46:23,844 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-08-29 21:46:27,474 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-08-29 21:46:34,096 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-08-29 21:46:46,725 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
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
