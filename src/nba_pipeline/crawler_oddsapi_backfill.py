import argparse
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

# NBA regular season runs Oct–Jun. Skip these months entirely (no games).
_NBA_OFF_SEASON_MONTHS = {7, 8, 9}  # July, August, September

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

    # Backfill range (ET dates).
    # DEFAULT: current season only (Oct 1, 2025).
    # The Odds API charges ~1 credit per event per market per response.
    # With ~9 games/day and 2 markets, each day costs ~18 credits.
    # A full season (~200 game-days) = ~3,600 credits — well within the 20k limit.
    # Going back to 2020 would cost ~18,000 credits. Don't do that.
    start_et: date = date(2025, 10, 1)
    end_et: date = datetime.now(_ET).date()

    # Safety budget: stop fetching when fewer than this many credits remain.
    # Leaves headroom for daily crawls and other usage.
    min_credits_remaining: int = 500

    # snapshot time (UTC) per ET day bucket.
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


def _fetch_with_backoff(cfg: BackfillConfig, url: str, params: dict) -> tuple[object, int | None]:
    """Returns (payload, credits_remaining). credits_remaining is None if header missing."""
    req_headers = {"Accept": "application/json"}
    last_err: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=req_headers, timeout=cfg.timeout_s)
            if r.status_code == 401:
                raise RuntimeError(f"401 Unauthorized — check your API key: {url}")
            if r.status_code == 429:
                wait = cfg.base_backoff_s * (2 ** (attempt - 1))
                log.warning("429 rate limited. sleeping %.1fs (attempt %d/%d)", wait, attempt, cfg.max_retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining")
            credits_remaining = int(remaining) if remaining is not None else None
            return r.json(), credits_remaining
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

    parser = argparse.ArgumentParser(description="Backfill historical NBA odds from The Odds API.")
    parser.add_argument("--start-date", type=date.fromisoformat, default=None,
                        help="ET start date (YYYY-MM-DD). Default: current season Oct 1.")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None,
                        help="ET end date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--budget", type=int, default=None,
                        help="Stop when fewer than this many credits remain. Default: 500.")
    args = parser.parse_args()

    cfg = BackfillConfig()
    if not cfg.oddsapi_key:
        raise RuntimeError("Missing oddsapi_key")

    start_et = args.start_date or cfg.start_et
    end_et = args.end_date or cfg.end_et
    min_credits = args.budget if args.budget is not None else cfg.min_credits_remaining

    # Estimate cost before starting
    game_days = sum(1 for d in _daterange(start_et, end_et) if d.month not in _NBA_OFF_SEASON_MONTHS)
    log.info(
        "Backfill plan: %s to %s | %d NBA-season days (skipping Jul/Aug/Sep) | "
        "est. credits ~%d (at ~18/day) | budget floor: %d",
        start_et, end_et, game_days, game_days * 18, min_credits,
    )

    url = f"https://api.the-odds-api.com/v4/historical/sports/{cfg.sport}/odds"
    provider = "oddsapi"
    endpoint = "nba_odds_historical"

    with psycopg2.connect(cfg.pg_dsn) as conn:
        conn.autocommit = False

        saved = 0
        skipped = 0
        credits_remaining: int | None = None

        for et_day in _daterange(start_et, end_et):
            # Skip off-season months — no NBA games, pure waste
            if et_day.month in _NBA_OFF_SEASON_MONTHS:
                skipped += 1
                continue

            # Budget guard: stop before we burn through all credits
            if credits_remaining is not None and credits_remaining < min_credits:
                log.warning(
                    "Budget guard: only %d credits remaining (floor=%d). Stopping.",
                    credits_remaining, min_credits,
                )
                break

            # Build "date=" snapshot timestamp in UTC (01:00:00Z)
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
                if skipped % 50 == 0:
                    log.info("Progress skipped=%d saved=%d (latest=%s)", skipped, saved, et_day)
                continue

            payload, credits_remaining = _fetch_with_backoff(cfg, url, params)
            fetched_at_utc = datetime.now(_UTC)
            payload_sha256 = _sha256_json(payload)

            n_events = len(payload.get("data", payload) if isinstance(payload, dict) else payload)
            log.info(
                "Fetched %s | events=%s | credits_remaining=%s",
                et_day, n_events,
                credits_remaining if credits_remaining is not None else "unknown",
            )

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

            time.sleep(cfg.sleep_between_calls_s)

        conn.commit()
        log.info(
            "Backfill done. saved=%d skipped=%d credits_remaining=%s",
            saved, skipped,
            credits_remaining if credits_remaining is not None else "unknown",
        )


if __name__ == "__main__":
    main()
