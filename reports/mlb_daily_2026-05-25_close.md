# SuperNovaBets MLB Daily Run (2026-05-25 ET)

## Summary

- **Re-crawl closing odds (Odds API)**: OK (rc=0, 533.7s)
- **Re-parse closing odds into odds.mlb_game_lines**: OK (rc=0, 9.7s)
- **Grade outcomes + CLV (update_outcomes)**: OK (rc=0, 5.5s)

## Outputs (tails)

### Re-crawl closing odds (Odds API)

- rc: 0

**stderr (tail)**
```
2026-05-25 18:40:41,713 | INFO | mlb_pipeline.crawler_oddsapi | Last saved date: 2026-05-26. Catching up from 2026-05-27 to 2026-05-24
2026-05-25 18:40:43,734 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-25 window=2026-05-24T18:00:00Z..2026-05-26T04:00:00Z
2026-05-25 18:42:23,113 | WARNING | mlb_pipeline.crawler_oddsapi | Fetch failed. sleeping 1.5s (attempt 1/5)
Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 464, in _make_request
    self._validate_conn(conn)
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 1093, in _validate_conn
    conn.connect()
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connection.py", line 796, in connect
    sock_and_verified = _ssl_wrap_socket_and_match_hostname(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connection.py", line 975, in _ssl_wrap_socket_and_match_hostname
    ssl_sock = ssl_wrap_socket(
               ^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\util\ssl_.py", line 483, in ssl_wrap_socket
    ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\util\ssl_.py", line 527, in _ssl_wrap_socket_impl
    return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\ssl.py", line 455, in wrap_socket
    return self.sslsocket_class._create(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\ssl.py", line 1042, in _create
    self.do_handshake()
  File "C:\Users\josh\AppData\Local\Programs\Python\Python312\Lib\ssl.py", line 1320, in do_handshake
    self._sslobj.do_handshake()
ssl.SSLEOFError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 787, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 488, in _make_request
    raise new_e
urllib3.exceptions.SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\adapters.py", line 644, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\connectionpool.py", line 841, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\urllib3\util\retry.py", line 535, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.the-odds-api.com', port=443): Max retries exceeded with url: /v4/sports/baseball_mlb/odds?apiKey=5b6f0290e265c3329b3ed27897d79eaf&regions=us&markets=spreads%2Ctotals&bookmakers=fanduel%2Cdraftkings&oddsFormat=american&dateFormat=iso&commenceTimeFrom=2026-05-24T18%3A00%3A00Z&commenceTimeTo=2026-05-26T04%3A00%3A00Z&includeLinks=true (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)')))

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
  File "C:\Users\josh\Git\SuperNovaBets\.venv\Lib\site-packages\requests\adapters.py", line 675, in send
    raise SSLError(e, request=request)
requests.exceptions.SSLError: HTTPSConnectionPool(host='api.the-odds-api.com', port=443): Max retries exceeded with url: /v4/sports/baseball_mlb/odds?apiKey=5b6f0290e265c3329b3ed27897d79eaf&regions=us&markets=spreads%2Ctotals&bookmakers=fanduel%2Cdraftkings&oddsFormat=american&dateFormat=iso&commenceTimeFrom=2026-05-24T18%3A00%3A00Z&commenceTimeTo=2026-05-26T04%3A00%3A00Z&includeLinks=true (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)')))
2026-05-25 18:42:32,136 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-25 | events=6 | credits_remaining=97366
2026-05-25 18:42:32,954 | INFO | mlb_pipeline.crawler_oddsapi | Fetching live odds for ET date=2026-05-26 window=2026-05-25T18:00:00Z..2026-05-27T04:00:00Z
2026-05-25 18:42:33,576 | INFO | mlb_pipeline.crawler_oddsapi | Live 2026-05-26 | events=19 | credits_remaining=97364
2026-05-25 18:42:33,625 | INFO | mlb_pipeline.crawler_oddsapi | Done. saved=2 skipped=0 credits_remaining=97364
```

### Re-parse closing odds into odds.mlb_game_lines

- rc: 0

**stderr (tail)**
```
2026-05-25 18:42:40,728 | INFO | mlb_pipeline.parse_oddsapi | Upserted 1203 rows into odds.mlb_game_lines (live odds).
2026-05-25 18:42:40,735 | INFO | mlb_pipeline.parse_oddsapi | Incremental historical MLB game odds: since 2026-05-25
2026-05-25 18:42:40,743 | WARNING | mlb_pipeline.parse_oddsapi | No mlb_odds_historical snapshots found (as_of_date=None).
2026-05-25 18:42:40,914 | INFO | mlb_pipeline.parse_oddsapi | Incremental MLB prop odds: since 2026-05-24
2026-05-25 18:42:44,026 | INFO | mlb_pipeline.parse_oddsapi | Upserted 5942 rows into odds.mlb_player_prop_lines.
2026-05-25 18:42:44,027 | INFO | mlb_pipeline.parse_oddsapi | Done.
```

### Grade outcomes + CLV (update_outcomes)

- rc: 0

**stdout (tail)**
```
MLB Run Line: 164-104 (61.2%) ROI: +16.8% | Total: 54-58 (48.2%) ROI: -8.0%
MLB CLV Run Line: beat close 69/268 (26%) avg CLV=+0.91 runs | CLV Total avg=+0.54 runs
MLB Price CLV Run Line: 197 bets  avg=-1.73%
```

**stderr (tail)**
```
2026-05-25 18:42:47,044 | INFO | mlb_pipeline.modeling.update_outcomes | No final MLB games found for 28 pending predictions.
2026-05-25 18:42:47,044 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB game outcome rows
2026-05-25 18:42:47,849 | INFO | mlb_pipeline.modeling.update_outcomes | backfill_clv: updated 140 rows
2026-05-25 18:42:47,849 | INFO | mlb_pipeline.modeling.update_outcomes | Backfilled CLV for 140 historical rows
2026-05-25 18:42:49,515 | INFO | mlb_pipeline.modeling.update_outcomes | update_prop_outcomes: updated 0 prop rows
2026-05-25 18:42:49,522 | INFO | mlb_pipeline.modeling.update_outcomes | Updated 0 MLB prop outcome rows
```
