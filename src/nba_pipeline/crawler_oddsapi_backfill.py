import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import requests

log = logging.getLogger("nba_pipeline.crawler_oddsapi_backfill")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

SQL_INSERT_API_RESPONSE = """
INSERT INTO raw.api_responses (
  provider,
  endpoint,
  season,
  game_slug,
  as_of_date,
  url,
  fetched_at_utc,
  payload,
  payload_sha256
)
VALUES (
  %(provider)s,
  %(endpoint)s,
  %(season)s,
  %(game_slug)s,
  %(as_of_date)s,
  %(url)s,
  %(fetched_at_utc)s,
  %(payload)s::jsonb,
  %(payload_sha256)s
)
ON CONFLICT DO NOTHING;
"""

SQL_ALREADY_FETCHED_URL = """
SELECT 1
FROM raw.api_responses
WHERE provider=%s
  AND endpoint=%s
  AND as_of_date IS NOT DISTINCT FROM %s
  AND url=%s
LIMIT 1;
"""


@dataclass(frozen=True)
class BackfillConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    oddsapi_key: str = "5b6f0290e265c3329b3ed27897d79eaf"

    sport: str = "basketball_nba"
    regions: str = "us"
    markets: str = "spreads,totals"
    bookmakers: str = "draftkings,fanduel"
    odds_format: str = "american"
    date_format: str = "iso"

    # backfill range (ET dates)
    start_et: date = date(2020, 6, 6)
    end_et: date = datetime.now(_ET).date()

    # snapshot time (UTC) per ET day bucket.
    # You said you want 01:00:00Z specifically.
    snapshot_hour_utc: int = 1
    snapshot_minute_utc: int = 0
    snapshot_second_utc: int = 0

    # requests
    timeout_s: int = 30
    max_retries: int = 6
    base_backoff_s: float = 1.5
    sleep_between_calls_s: float = 0.25  # be nice to the API


def _sha256_json(obj: object) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _build_full_url(url: str, params: dict) -> str:
    # deterministic query string
    qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    return f"{url}?{qs}"


def _already_fetched(conn, *, provider: str, endpoint: str, as_of_date: date, full_url: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(SQL_ALREADY_FETCHED_URL, (provider, endpoint, as_of_date, full_url))
        return cur.fetchone() is not None


def _fetch_with_backoff(cfg: BackfillConfig, url: str, params: dict) -> object:
    headers = {"Accept": "application/json"}
    last_err: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=cfg.timeout_s)
            if r.status_code == 429:
                wait = cfg.base_backoff_s * (2 ** (attempt - 1))
                log.warning("429 rate limited. sleeping %.1fs (attempt %d/%d)", wait, attempt, cfg.max_retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = cfg.base_backoff_s * (2 ** (attempt - 1))
            log.warning("Fetch failed. sleeping %.1fs (attempt %d/%d)", wait, attempt, cfg.max_retries, exc_info=True)
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch after retries: {url}") from last_err


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = BackfillConfig()
    if not cfg.oddsapi_key:
        raise RuntimeError("Missing oddsapi_key")

    url = f"https://api.the-odds-api.com/v4/historical/sports/{cfg.sport}/odds"
    provider = "oddsapi"
    endpoint = "nba_odds_historical"

    with psycopg2.connect(cfg.pg_dsn) as conn:
        conn.autocommit = False

        saved = 0
        skipped = 0

        for et_day in _daterange(cfg.start_et, cfg.end_et):
            # Build "date=" snapshot timestamp in UTC (01:00:00Z by your request)
            snap_dt_utc = datetime(
                et_day.year, et_day.month, et_day.day,
                cfg.snapshot_hour_utc, cfg.snapshot_minute_utc, cfg.snapshot_second_utc,
                tzinfo=_UTC,
            )
            snap_ts = snap_dt_utc.isoformat().replace("+00:00", "Z")

            params = {
                "apiKey": cfg.oddsapi_key,
                "regions": cfg.regions,
                "markets": cfg.markets,
                "bookmakers": cfg.bookmakers,
                "oddsFormat": cfg.odds_format,
                "dateFormat": cfg.date_format,
                "date": snap_ts,
            }

            full_url = _build_full_url(url, params)

            if _already_fetched(conn, provider=provider, endpoint=endpoint, as_of_date=et_day, full_url=full_url):
                skipped += 1
                if skipped % 200 == 0:
                    log.info("Progress skipped=%d saved=%d (latest=%s)", skipped, saved, et_day)
                continue

            payload = _fetch_with_backoff(cfg, url, params)
            fetched_at_utc = datetime.now(_UTC)
            payload_sha256 = _sha256_json(payload)

            with conn.cursor() as cur:
                cur.execute(
                    SQL_INSERT_API_RESPONSE,
                    {
                        "provider": provider,
                        "endpoint": endpoint,
                        "season": None,
                        "game_slug": None,
                        "as_of_date": et_day,
                        "url": full_url,
                        "fetched_at_utc": fetched_at_utc,
                        "payload": json.dumps(payload),
                        "payload_sha256": payload_sha256,
                    },
                )

            saved += 1
            if saved % 50 == 0:
                conn.commit()
                log.info("Committed batch saved=%d (latest=%s)", saved, et_day)

            # be polite
            time.sleep(cfg.sleep_between_calls_s)

        conn.commit()
        log.info("Backfill done. saved=%d skipped=%d", saved, skipped)


if __name__ == "__main__":
    main()
