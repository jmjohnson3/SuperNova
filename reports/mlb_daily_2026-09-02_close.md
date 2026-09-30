# SuperNovaBets MLB Daily Run (2026-09-02 ET)

## Summary

- **Refresh final game results (MLB Stats API)**: OK (rc=0, 5.2s)
- **Re-crawl closing game odds (Odds API)**: FAIL (rc=1, 53.2s)
- **Re-crawl closing prop odds (Odds API)**: FAIL (rc=1, 52.9s)

## Outputs (tails)

### Refresh final game results (MLB Stats API)

- rc: 0

**stderr (tail)**
```
2026-09-02 21:45:16,727 | INFO | mlb_pipeline.crawler_statsapi | Fetched probable pitchers for 30 teams today (2026-09-02)
2026-09-02 21:45:17,164 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: upserted 60 rows for 2026-09-02
2026-09-02 21:45:17,164 | INFO | mlb_pipeline.crawler_statsapi | Pre-game umpires: 60 assignments for 2026-09-02
2026-09-02 21:45:17,186 | INFO | mlb_pipeline.crawler_statsapi | Fetching schedule for season=2026-regular ...
2026-09-02 21:45:17,597 | INFO | mlb_pipeline.crawler_statsapi | Schedule returned 15 unique games for season=2026-regular
2026-09-02 21:45:17,665 | INFO | mlb_pipeline.crawler_statsapi | Upserted 15 rows into raw.mlb_games
2026-09-02 21:45:17,733 | INFO | mlb_pipeline.crawler_statsapi | Season=2026-regular: 6941 completed games, 6931 already done, 1 to fetch
2026-09-02 21:45:18,525 | INFO | mlb_pipeline.crawler_statsapi |   Progress: 1 / 1 games fetched
2026-09-02 21:45:18,682 | INFO | mlb_pipeline.crawler_statsapi | Backfill complete: season=2026-regular, fetched=1, errors=0
```

### Re-crawl closing game odds (Odds API)

- rc: 1

**stderr (tail)**
```
2026-09-02 21:45:22,071 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-09-01
2026-09-02 21:45:22,943 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-09-02 21:45:25,068 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-09-02 21:45:28,609 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-09-02 21:45:35,170 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-09-02 21:45:47,794 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
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
2026-09-02 21:46:15,142 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-07-31. Catching up from 2026-08-01 to 2026-09-01
2026-09-02 21:46:15,703 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 1.5s (attempt 1/5)
2026-09-02 21:46:17,827 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 3.0s (attempt 2/5)
2026-09-02 21:46:21,435 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 6.0s (attempt 3/5)
2026-09-02 21:46:28,087 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 12.0s (attempt 4/5)
2026-09-02 21:46:40,648 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed (RuntimeError). sleeping 24.0s (attempt 5/5)
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
