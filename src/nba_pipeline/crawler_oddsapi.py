# src/nba_pipeline/crawler_oddsapi.py
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
import hashlib, json, logging, time
import psycopg2, requests

log = logging.getLogger("nba_pipeline.crawler_oddsapi")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class OddsCrawlerConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    oddsapi_key: str = "5b6f0290e265c3329b3ed27897d79eaf"
    sport: str = "basketball_nba"
    regions: str = "us"
    markets: str = "spreads,totals"
    bookmakers: str = "fanduel,draftkings"
    odds_format: str = "american"
    date_format: str = "iso"
    max_retries: int = 5
    base_backoff_s: float = 1.5
    timeout_s: int = 30

    # ✅ NEW: shift the UTC start earlier by 4 hours (05:00Z -> 01:00Z)
    commence_from_utc_shift_hours: int = -10
    # usually leave end unshifted
    commence_to_utc_shift_hours: int = 0


def _sha256_json(obj: object) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _to_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(_UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _et_day_window_utc(et_day: date) -> tuple[datetime, datetime]:
    # ET calendar day [00:00, 24:00)
    et_start = datetime.combine(et_day, dtime(0, 0), tzinfo=_ET)
    et_end = et_start + timedelta(days=1)
    return et_start.astimezone(_UTC), et_end.astimezone(_UTC)


def _fetch_with_backoff(cfg: OddsCrawlerConfig, url: str, params: dict) -> object:
    headers = {"Accept": "application/json"}
    last_err = None
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = OddsCrawlerConfig()
    if not cfg.oddsapi_key:
        raise RuntimeError("Missing OddsAPI key.")

    et_target = datetime.now(_ET).date() + timedelta(days=1)

    start_utc, end_utc = _et_day_window_utc(et_target)

    # ✅ Apply the shifts to force 01:00Z
    start_utc = start_utc + timedelta(hours=cfg.commence_from_utc_shift_hours)
    end_utc = end_utc + timedelta(hours=cfg.commence_to_utc_shift_hours)

    commence_from = _to_z(start_utc)
    commence_to = _to_z(end_utc)

    url = f"https://api.the-odds-api.com/v4/sports/{cfg.sport}/odds"
    params = {
        "apiKey": cfg.oddsapi_key,
        "regions": cfg.regions,
        "markets": cfg.markets,
        "bookmakers": cfg.bookmakers,
        "oddsFormat": cfg.odds_format,
        "dateFormat": cfg.date_format,
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }

    log.info("Fetching OddsAPI odds for ET date=%s window=%s..%s", et_target, commence_from, commence_to)

    payload = _fetch_with_backoff(cfg, url, params)
    fetched_at_utc = datetime.now(_UTC)
    payload_sha256 = _sha256_json(payload)

    qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    full_url = f"{url}?{qs}"

    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO raw.api_responses (
                  provider, endpoint, season, game_slug, as_of_date, url, fetched_at_utc, payload, payload_sha256
                )
                VALUES (
                  %(provider)s, %(endpoint)s, %(season)s, %(game_slug)s, %(as_of_date)s,
                  %(url)s, %(fetched_at_utc)s, %(payload)s::jsonb, %(payload_sha256)s
                )
                ON CONFLICT DO NOTHING;
                """,
                {
                    "provider": "oddsapi",
                    "endpoint": "nba_odds",
                    "season": None,
                    "game_slug": None,
                    "as_of_date": et_target,
                    "url": full_url,
                    "fetched_at_utc": fetched_at_utc,
                    "payload": json.dumps(payload),
                    "payload_sha256": payload_sha256,
                },
            )
        conn.commit()

    n = len(payload) if isinstance(payload, list) else None
    log.info("Saved OddsAPI payload events=%s sha256=%s", n, payload_sha256[:10])


if __name__ == "__main__":
    main()
