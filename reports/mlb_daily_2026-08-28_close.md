# SuperNovaBets MLB Daily Run (2026-08-28 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 6.2s)
- **Re-crawl closing game odds (Odds API)**: FAIL (rc=1, 72.9s)
- **Re-crawl closing prop odds (Odds API)**: FAIL (rc=1, 53.5s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-08-28 21:45:06,274 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-08-28)
2026-08-28 21:45:06,694 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-08-28
2026-08-28 21:45:06,694 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-08-28
2026-08-28 21:45:06,742 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-08-28 21:45:07,132 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-08-28 21:45:07,229 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-08-28 21:45:07,366 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6868 completed games, 6857 already done, 2 to fetch
2026-08-28 21:45:08,880 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 2 / 2 games fetched
2026-08-28 21:45:09,040 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=2, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 1

**stderr (tail)**
```
2026-08-28 21:45:12,498 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-08-27
2026-08-28 21:45:13,569 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-08-28 21:45:34,970 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-08-28 21:45:38,541 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-08-28 21:45:45,185 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-08-28 21:45:57,803 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
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
2026-08-28 21:46:25,871 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-08-27
2026-08-28 21:46:26,540 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-08-28 21:46:28,654 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-08-28 21:46:32,278 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-08-28 21:46:38,805 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-08-28 21:46:51,437 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
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
