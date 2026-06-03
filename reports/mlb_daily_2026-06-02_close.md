# SuperNovaBets MLB Daily Run (2026-06-02 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 51.6s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 16.6s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 46.4s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-06-02 18:30:45,686 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-06-03. Catching up from 2026-06-04 to 2026-06-01
2026-06-02 18:30:45,686 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-02 window=2026-06-01T18:00:00Z..2026-06-03T04:00:00Z
2026-06-02 18:31:18,811 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed. sleeping 1.5s (attempt 1/5)
Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 787, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 534, in _make_request
    response = conn.getresponse()
               ^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connection.py", line 571, in getresponse
    httplib_response = super().getresponse()
                       ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\http\client.py", line 1423, in getresponse
    response.begin()
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\http\client.py", line 331, in begin
    version, status, reason = self._read_status()
                              ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\http\client.py", line 300, in _read_status
    raise RemoteDisconnected("Remote end closed connection without"
http.client.RemoteDisconnected: Remote end closed connection without response

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\adapters.py", line 644, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 841, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\util\retry.py", line 490, in increment
    raise reraise(type(error), error, _stacktrace)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\util\util.py", line 38, in reraise
    raise value.with_traceback(tb)
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 787, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 534, in _make_request
    response = conn.getresponse()
               ^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connection.py", line 571, in getresponse
    httplib_response = super().getresponse()
                       ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\http\client.py", line 1423, in getresponse
    response.begin()
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\http\client.py", line 331, in begin
    version, status, reason = self._read_status()
                              ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\http\client.py", line 300, in _read_status
    raise RemoteDisconnected("Remote end closed connection without"
urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\src\mlb_pipeline\crawler_oddsapi.py", line 81, in _fetch_with_backoff
    r = requests.get(url, params=params, headers=headers, timeout=cfg.timeout_s)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\api.py", line 73, in get
    return request("get", url, params=params, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\api.py", line 59, in request
    return session.request(method=method, url=url, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\adapters.py", line 659, in send
    raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
2026-06-02 18:31:22,108 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-02 | events=15 | credits_remaining=99838
2026-06-02 18:31:23,031 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-06-03 window=2026-06-02T18:00:00Z..2026-06-04T04:00:00Z
2026-06-02 18:31:23,615 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-06-03 | events=30 | credits_remaining=99836
2026-06-02 18:31:23,645 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=99836
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-06-02 18:31:33,065 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1372 rows into odds.mlb_game_lines (live odds).
2026-06-02 18:31:33,077 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-06-02
2026-06-02 18:31:33,077 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-06-02 18:31:33,360 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-06-01
2026-06-02 18:31:40,455 | INFO | mlb_pipeline.parse_oddsapi | Upserted 4943 rows into odds.mlb_player_prop_lines.
2026-06-02 18:31:40,457 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 177-121 (59.4%) ROI: +13.4% | Total: 56-58 (49.1%) ROI: -6.2%
MLB CLV Run Line: beat close 79/298 (27%) avg CLV=+0.93 runs | CLV Total avg=+0.53 runs
MLB Price CLV Run Line: 227 bets  avg=-1.92%
```

**stderr (tail)**
```
2026-06-02 18:32:13,522 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 31 pending predictions.
2026-06-02 18:32:13,522 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-06-02 18:32:16,282 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 141 rows
2026-06-02 18:32:16,282 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 141 historical rows
2026-06-02 18:32:25,824 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-06-02 18:32:25,832 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
2026-06-02 18:32:26,047 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 game bankroll ledger rows
2026-06-02 18:32:26,088 | INFO | mlb_pipeline.modeling.bankroll_ledger | Graded 0 prop bankroll ledger rows
2026-06-02 18:32:26,089 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB bankroll ledger rows
2026-06-02 18:32:26,143 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 game model-pick ledger rows
2026-06-02 18:32:26,153 | INFO | mlb_pipeline.modeling.model_pick_ledger | Graded 0 prop model-pick ledger rows
2026-06-02 18:32:26,153 | INFO | mlb_pipeline.modeling.update_outcomes | Graded 0 MLB model-pick ledger rows
```
