# src/nba_pipeline/crawler_oddsapi.py
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
import hashlib, json, logging, time
import psycopg2, requests

log = logging.getLogger("nba_pipeline.crawler_oddsapi")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# NBA off-season: no games, skip to save credits
_NBA_OFF_SEASON_MONTHS = {7, 8, 9}  # July, August, September

_PROP_MARKETS = "player_points,player_rebounds,player_assists"
_PROP_ENDPOINT_TMPL = "https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"


@dataclass(frozen=True)
class OddsCrawlerConfig:
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"
    oddsapi_key: str = "f623ce7de5f553aeb1f20289ce6083e4"
    sport: str = "basketball_nba"
    regions: str = "us"
    markets: str = "spreads,totals"
    bookmakers: str = "fanduel,draftkings"
    odds_format: str = "american"
    date_format: str = "iso"
    max_retries: int = 5
    base_backoff_s: float = 1.5
    timeout_s: int = 30
    sleep_between_calls_s: float = 0.25  # politeness delay during catch-up

    # Stop fetching when fewer than this many credits remain
    min_credits_remaining: int = 200

    # Live endpoint: shift commence window to capture late-night ET games
    commence_from_utc_shift_hours: int = -10
    commence_to_utc_shift_hours: int = 0

    # Historical snapshot time: 01:00:00 UTC per ET game day
    snapshot_hour_utc: int = 1
    snapshot_minute_utc: int = 0
    snapshot_second_utc: int = 0


def _sha256_json(obj: object) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _to_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(_UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _et_day_window_utc(et_day: date) -> tuple[datetime, datetime]:
    et_start = datetime.combine(et_day, dtime(0, 0), tzinfo=_ET)
    et_end = et_start + timedelta(days=1)
    return et_start.astimezone(_UTC), et_end.astimezone(_UTC)


def _build_full_url(url: str, params: dict) -> str:
    qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    return f"{url}?{qs}"


def _fetch_with_backoff(cfg: OddsCrawlerConfig, url: str, params: dict) -> tuple[object, int | None]:
    """Returns (payload, credits_remaining). credits_remaining is None if header missing."""
    headers = {"Accept": "application/json"}
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=cfg.timeout_s)
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


def _last_saved_date(conn) -> date | None:
    """Return the most recent as_of_date saved for oddsapi (either endpoint)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(as_of_date)
            FROM raw.api_responses
            WHERE provider = 'oddsapi'
              AND endpoint IN ('nba_odds', 'nba_odds_historical')
        """)
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def _already_fetched(conn, *, as_of_date: date, full_url: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM raw.api_responses
            WHERE provider = 'oddsapi'
              AND as_of_date IS NOT DISTINCT FROM %s
              AND url = %s
            LIMIT 1
        """, (as_of_date, full_url))
        return cur.fetchone() is not None


def _save_payload(conn, *, endpoint: str, as_of_date: date, url: str,
                  payload: object, fetched_at_utc: datetime) -> None:
    payload_sha256 = _sha256_json(payload)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO raw.api_responses (
              provider, endpoint, season, game_slug, as_of_date,
              url, fetched_at_utc, payload, payload_sha256
            )
            VALUES (
              %(provider)s, %(endpoint)s, NULL, NULL, %(as_of_date)s,
              %(url)s, %(fetched_at_utc)s, %(payload)s::jsonb, %(payload_sha256)s
            )
            ON CONFLICT DO NOTHING;
        """, {
            "provider": "oddsapi",
            "endpoint": endpoint,
            "as_of_date": as_of_date,
            "url": url,
            "fetched_at_utc": fetched_at_utc,
            "payload": json.dumps(payload),
            "payload_sha256": payload_sha256,
        })
    conn.commit()


def _fetch_historical_day(cfg: OddsCrawlerConfig, conn, et_day: date) -> int | None:
    """Fetch one past ET day via the historical endpoint.

    Returns credits_remaining if a fetch was made, or None if skipped (already fetched).
    """
    snap_dt_utc = datetime(
        et_day.year, et_day.month, et_day.day,
        cfg.snapshot_hour_utc, cfg.snapshot_minute_utc, cfg.snapshot_second_utc,
        tzinfo=_UTC,
    )
    snap_ts = snap_dt_utc.isoformat().replace("+00:00", "Z")

    url = f"https://api.the-odds-api.com/v4/historical/sports/{cfg.sport}/odds"
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

    if _already_fetched(conn, as_of_date=et_day, full_url=full_url):
        log.debug("Already have historical odds for %s — skipping", et_day)
        return None

    payload, credits_remaining = _fetch_with_backoff(cfg, url, params)
    fetched_at_utc = datetime.now(_UTC)

    n_events = len(payload.get("data", payload) if isinstance(payload, dict) else payload)
    log.info("Historical %s | events=%s | credits_remaining=%s",
             et_day, n_events,
             credits_remaining if credits_remaining is not None else "unknown")

    _save_payload(conn, endpoint="nba_odds_historical", as_of_date=et_day,
                  url=full_url, payload=payload, fetched_at_utc=fetched_at_utc)
    return credits_remaining


def _fetch_live_day(cfg: OddsCrawlerConfig, conn, et_day: date) -> int | None:
    """Fetch a current/future ET day via the live endpoint.

    Returns credits_remaining if a fetch was made, or None if skipped.
    """
    start_utc, end_utc = _et_day_window_utc(et_day)
    start_utc = start_utc + timedelta(hours=cfg.commence_from_utc_shift_hours)
    end_utc = end_utc + timedelta(hours=cfg.commence_to_utc_shift_hours)

    url = f"https://api.the-odds-api.com/v4/sports/{cfg.sport}/odds"
    params = {
        "apiKey": cfg.oddsapi_key,
        "regions": cfg.regions,
        "markets": cfg.markets,
        "bookmakers": cfg.bookmakers,
        "oddsFormat": cfg.odds_format,
        "dateFormat": cfg.date_format,
        "commenceTimeFrom": _to_z(start_utc),
        "commenceTimeTo": _to_z(end_utc),
    }
    full_url = _build_full_url(url, params)

    if _already_fetched(conn, as_of_date=et_day, full_url=full_url):
        log.info("Already have live odds for %s — skipping", et_day)
        return None

    log.info("Fetching live odds for ET date=%s window=%s..%s",
             et_day, params["commenceTimeFrom"], params["commenceTimeTo"])

    payload, credits_remaining = _fetch_with_backoff(cfg, url, params)
    fetched_at_utc = datetime.now(_UTC)

    n = len(payload) if isinstance(payload, list) else None
    log.info("Live %s | events=%s | credits_remaining=%s",
             et_day, n, credits_remaining if credits_remaining is not None else "unknown")

    _save_payload(conn, endpoint="nba_odds", as_of_date=et_day,
                  url=full_url, payload=payload, fetched_at_utc=fetched_at_utc)
    return credits_remaining


def _get_todays_event_ids(conn, et_day: date) -> list[dict]:
    """Return event dicts from the already-fetched game-odds payload for et_day.

    Reuses data already stored in raw.api_responses — no extra API call.
    Returns list of {"event_id": ..., "home_team": ..., "away_team": ...}.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT payload
            FROM raw.api_responses
            WHERE provider = 'oddsapi'
              AND endpoint IN ('nba_odds', 'nba_odds_historical')
              AND as_of_date = %s
            ORDER BY fetched_at_utc DESC
            LIMIT 1
        """, (et_day,))
        row = cur.fetchone()

    if row is None:
        return []

    payload = row[0]
    if isinstance(payload, str):
        payload = json.loads(payload)

    # Historical endpoint wraps data under a "data" key
    if isinstance(payload, dict):
        payload = payload.get("data", [])

    events = []
    for ev in (payload or []):
        if isinstance(ev, dict) and ev.get("id"):
            events.append({
                "event_id": ev["id"],
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
            })
    return events


def _fetch_prop_lines_for_day(
    cfg: OddsCrawlerConfig, conn, as_of_date: date, event_source_date: date | None = None
) -> int | None:
    """Fetch player prop lines for upcoming events and save them as as_of_date.

    event_source_date: which game-odds payload to read event IDs from.
                       Defaults to as_of_date.  Pass tomorrow's date to
                       pre-fetch prop lines for the next day's games while
                       still storing them with today's as_of_date so the
                       leakage guard (as_of_date < game_date) is satisfied.

    Uses the per-event endpoint (3 markets per event).
    Returns credits_remaining after the last fetch, or None if nothing fetched.
    """
    source_date = event_source_date if event_source_date is not None else as_of_date
    events = _get_todays_event_ids(conn, source_date)
    if not events:
        log.info("No event IDs found for %s — skipping prop fetch", source_date)
        return None

    log.info("Fetching prop lines for %d events (source=%s, as_of=%s)",
             len(events), source_date, as_of_date)
    fetched_at_utc = datetime.now(_UTC)
    credits_remaining: int | None = None

    for ev in events:
        event_id = ev["event_id"]
        url = _PROP_ENDPOINT_TMPL.format(sport=cfg.sport, event_id=event_id)
        params = {
            "apiKey": cfg.oddsapi_key,
            "regions": "us",
            "markets": _PROP_MARKETS,
            "bookmakers": "draftkings",
            "oddsFormat": cfg.odds_format,
            "dateFormat": cfg.date_format,
        }
        full_url = _build_full_url(url, params)

        if _already_fetched(conn, as_of_date=as_of_date, full_url=full_url):
            log.debug("Already have prop odds for event %s on %s — skipping", event_id, as_of_date)
            continue

        if credits_remaining is not None and credits_remaining < cfg.min_credits_remaining:
            log.warning("Budget guard: %d credits remaining. Stopping prop fetch.", credits_remaining)
            break

        try:
            payload, credits_remaining = _fetch_with_backoff(cfg, url, params)
        except RuntimeError as exc:
            log.warning("Prop fetch failed for event %s: %s", event_id, exc)
            continue

        log.info("Prop odds as_of=%s source=%s | event=%s | credits_remaining=%s",
                 as_of_date, source_date, event_id,
                 credits_remaining if credits_remaining is not None else "unknown")

        _save_payload(conn, endpoint="nba_prop_odds", as_of_date=as_of_date,
                      url=full_url, payload=payload, fetched_at_utc=fetched_at_utc)

        if credits_remaining is not None and credits_remaining < cfg.min_credits_remaining:
            log.warning("LOW CREDIT BALANCE: %d credits remaining. Stopping prop fetch.", credits_remaining)
            break

        time.sleep(cfg.sleep_between_calls_s)

    return credits_remaining


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = OddsCrawlerConfig()
    if not cfg.oddsapi_key:
        raise RuntimeError("Missing OddsAPI key.")

    today_et = datetime.now(_ET).date()
    tomorrow_et = today_et + timedelta(days=1)

    credits_remaining: int | None = None
    saved = 0
    skipped = 0

    with psycopg2.connect(cfg.pg_dsn) as conn:
        # Find the last date we already have odds for
        last_date = _last_saved_date(conn)

        if last_date is None:
            # No history — start from the current season opener
            season_start_year = today_et.year if today_et.month >= 10 else today_et.year - 1
            catchup_start = date(season_start_year, 10, 1)
            log.info("No prior odds found. Catching up from %s", catchup_start)
        else:
            catchup_start = last_date + timedelta(days=1)
            log.info("Last saved date: %s. Catching up from %s to %s",
                     last_date, catchup_start, today_et - timedelta(days=1))

        # --- Catch-up: historical endpoint for all missing past days ---
        d = catchup_start
        while d < today_et:
            if d.month in _NBA_OFF_SEASON_MONTHS:
                skipped += 1
                d += timedelta(days=1)
                continue

            if credits_remaining is not None and credits_remaining < cfg.min_credits_remaining:
                log.warning("Budget guard: %d credits remaining (floor=%d). Stopping catch-up.",
                            credits_remaining, cfg.min_credits_remaining)
                break

            result = _fetch_historical_day(cfg, conn, d)
            if result is not None:
                credits_remaining = result
                saved += 1
                if credits_remaining < cfg.min_credits_remaining:
                    log.warning("LOW CREDIT BALANCE: %d credits remaining. Stopping.", credits_remaining)
                    d += timedelta(days=1)
                    break
                time.sleep(cfg.sleep_between_calls_s)
            else:
                skipped += 1

            d += timedelta(days=1)

        # --- Live endpoint: today and tomorrow ---
        for live_day in (today_et, tomorrow_et):
            if live_day.month in _NBA_OFF_SEASON_MONTHS:
                continue
            if credits_remaining is not None and credits_remaining < cfg.min_credits_remaining:
                log.warning("Budget guard: %d credits remaining. Skipping live fetch for %s.",
                            credits_remaining, live_day)
                continue
            result = _fetch_live_day(cfg, conn, live_day)
            if result is not None:
                credits_remaining = result
                saved += 1
                if credits_remaining < cfg.min_credits_remaining:
                    log.warning("LOW CREDIT BALANCE: %d credits remaining.", credits_remaining)

        # --- Prop lines: today's events AND tomorrow's upcoming events ---
        # Tomorrow's events are saved with as_of_date=today so the leakage guard
        # (as_of_date < game_date) is satisfied when predicting tomorrow's games.
        if today_et.month not in _NBA_OFF_SEASON_MONTHS:
            if credits_remaining is None or credits_remaining >= cfg.min_credits_remaining:
                # Try today's events (may 404 if games already in progress)
                result = _fetch_prop_lines_for_day(cfg, conn, today_et)
                if result is not None:
                    credits_remaining = result
                # Try tomorrow's upcoming events, stored as today's date
                if credits_remaining is None or credits_remaining >= cfg.min_credits_remaining:
                    result = _fetch_prop_lines_for_day(
                        cfg, conn, today_et, event_source_date=tomorrow_et
                    )
                    if result is not None:
                        credits_remaining = result
            else:
                log.warning("Budget guard: %d credits remaining. Skipping prop fetch.",
                            credits_remaining)

    log.info("Done. saved=%d skipped=%d credits_remaining=%s",
             saved, skipped,
             credits_remaining if credits_remaining is not None else "unknown")


if __name__ == "__main__":
    main()
