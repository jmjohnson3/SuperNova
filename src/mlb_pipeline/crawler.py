import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional
import psycopg2
from zoneinfo import ZoneInfo

from nba_pipeline.fetcher import MySportsFeedsClient, NoContentYetError, RateLimitedError, BadPayloadError
from nba_pipeline.raw_store import save_api_response

log = logging.getLogger("mlb_pipeline.crawler")

MLB_TEAMS = [
    "ari", "atl", "bal", "bos", "chc", "cws", "cin", "cle", "col", "det",
    "hou", "kc",  "laa", "lad", "mia", "mil", "min", "nym", "nyy", "oak",
    "phi", "pit", "sd",  "sea", "sf",  "stl", "tb",  "tex", "tor", "was",
]

# NOTE: parsers import _norm_abbr from here; see also team_abbr.py for unified module
TEAM_ABBR_NORMALIZE: dict[str, str] = {
    # Normalize common MSF variants to canonical lowercase-equivalent upper forms.
    # Add entries here if API payloads use unexpected abbreviations.
    "KC":  "KC",   # Kansas City — no change needed, but document it
    "CWS": "CWS",  # Chicago White Sox
    "NYM": "NYM",
    "NYY": "NYY",
    "LAA": "LAA",
    "LAD": "LAD",
    # Historical / alternate abbreviations seen in some MSF responses:
    "SFG": "SF",
    "SDP": "SD",
    "KCR": "KC",
    "TBR": "TB",
    "WSN": "WAS",
    "CHW": "CWS",
}


def _norm_abbr(abbr: str) -> str:
    a = (abbr or "").strip().upper()
    return TEAM_ABBR_NORMALIZE.get(a, a)


_ET = ZoneInfo("America/New_York")
@dataclass(frozen=True)
class Season:
    league: str
    season_slug: str
    season_start: date  # first possible game date (used to skip crawling future dates against old seasons)
    season_end: date    # last possible game date


@dataclass(frozen=True)
class CrawlerConfig:
    api_key: str = "4359aa1b-cc29-4647-a3e5-7314e2"
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    # "incremental window" behavior:
    # - we start at (last_seen_games_by_date - lookback_days)
    # - and crawl through ET today (+ optionally tomorrow)
    lookback_days: int = 2
    include_tomorrow: bool = True

    commit_every: int = 50


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def et_today() -> date:
    return datetime.now(tz=_ET).date()


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ---------- URL builders ----------

def build_url_games_by_date(season: Season, d: date) -> str:
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}"
        f"/{season.season_slug}/date/{yyyymmdd(d)}/games.json"
    )


def build_url_boxscore(season: Season, game_slug: str) -> str:
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}"
        f"/{season.season_slug}/games/{game_slug}/boxscore.json"
    )


def build_url_lineup(season: Season, game_slug: str) -> str:
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}"
        f"/{season.season_slug}/games/{game_slug}/lineup.json"
    )


def build_url_player_gamelogs(season: Season, d: date, team: str) -> str:
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}"
        f"/{season.season_slug}/date/{yyyymmdd(d)}/player_gamelogs.json?team={team}"
    )


def build_url_injuries(season: Season) -> str:
    # MySportsFeeds injuries endpoint is not season-scoped in the URL
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/injuries.json"


def build_url_standings(season: Season) -> str:
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}"
        f"/{season.season_slug}/standings.json"
    )


def build_url_venues(season: Season) -> str:
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}"
        f"/{season.season_slug}/venues.json"
    )


# ---------- DB helpers ----------


def already_fetched(
    conn,
    *,
    provider: str,
    endpoint: str,
    season: Optional[str],
    game_slug: Optional[str],
    as_of_date: Optional[date],
    url: str,
) -> bool:
    """
    Fast pre-check to avoid hitting the API at all when we already have this exact URL stored.
    """
    q = """
    SELECT 1
    FROM raw.api_responses
    WHERE provider=%s
      AND endpoint=%s
      AND (season IS NOT DISTINCT FROM %s)
      AND (game_slug IS NOT DISTINCT FROM %s)
      AND (as_of_date IS NOT DISTINCT FROM %s)
      AND url=%s
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (provider, endpoint, season, game_slug, as_of_date, url))
        return cur.fetchone() is not None


def _mlb_boxscore_is_completed(conn, *, game_slug: str) -> bool:
    """
    Return True if raw.mlb_boxscore_games already has a COMPLETED record for this game.

    Used instead of already_fetched() for boxscore endpoints: we re-fetch until the
    game is settled so partial mid-game scores never get permanently cached.
    Note: raw.mlb_boxscore_games is populated by parse_boxscore (part of parse_all),
    so this reflects the state from the previous pipeline run.
    """
    q = """
    SELECT 1 FROM raw.mlb_boxscore_games
    WHERE game_slug = %s AND played_status = 'COMPLETED'
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (game_slug,))
        return cur.fetchone() is not None


def last_games_by_date_asof(conn, *, provider: str, season: str) -> Optional[date]:
    """
    Returns the max as_of_date we have for games_by_date for this season.
    If you've never run the incremental crawler yet, this might be NULL.
    """
    q = """
    SELECT MAX(as_of_date)
    FROM raw.api_responses
    WHERE provider=%s
      AND endpoint='games_by_date'
      AND season=%s
      AND as_of_date IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(q, (provider, season))
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


# ---------- bulk pre-check helpers ----------

def _load_url_set(
    conn,
    *,
    provider: str,
    endpoint: str,
    season: Optional[str],
    as_of_date: Optional[date],
) -> set:
    """Load all stored URLs for a specific endpoint/season/date combo into a set.

    One DB round-trip replaces N individual ``already_fetched()`` calls.
    """
    q = """
    SELECT url FROM raw.api_responses
    WHERE provider = %s
      AND endpoint = %s
      AND (season IS NOT DISTINCT FROM %s)
      AND (as_of_date IS NOT DISTINCT FROM %s)
    """
    with conn.cursor() as cur:
        cur.execute(q, (provider, endpoint, season, as_of_date))
        return {row[0] for row in cur.fetchall()}


def _load_all_completed_boxscores(conn) -> set:
    """Return set of all game_slugs that are COMPLETED in raw.mlb_boxscore_games."""
    q = "SELECT game_slug FROM raw.mlb_boxscore_games WHERE played_status = 'COMPLETED'"
    with conn.cursor() as cur:
        cur.execute(q)
        return {row[0] for row in cur.fetchall()}


def _fetch_url_tagged(
    client: "MySportsFeedsClient",
    url: str,
    context: str,
    date_ctx: "date",
    rate_hit: "threading.Event",
) -> "Optional[tuple]":
    """Fetch *url* in a worker thread.  Returns ``(url, payload)`` or ``None``.

    ``context`` is a short string included in log messages (e.g. ``"gamelog
    date=2026-05-21 team=nyy"``).  Sets *rate_hit* on HTTP 429 so sibling
    threads short-circuit their own fetches.
    """
    if rate_hit.is_set():
        return None
    try:
        payload = client.fetch_json(url)
        return url, payload
    except NoContentYetError:
        log.info("Not ready yet (204) %s", context)
        return None
    except RateLimitedError:
        log.warning("Rate limited (429). %s date=%s", context, date_ctx)
        rate_hit.set()
        return None
    except BadPayloadError as e:
        log.warning("Bad payload %s err=%s", context, e)
        return None
    except Exception:
        log.warning("Failed fetch %s url=%s", context, url, exc_info=True)
        return None


# ---------- payload parsing helpers ----------

def extract_game_slugs_from_games_payload(payload: dict) -> list[str]:
    slugs: list[str] = []
    games = payload.get("games") or []
    for g in games:
        sched = g.get("schedule") or {}
        start_time = sched.get("startTime")
        away = _norm_abbr(((sched.get("awayTeam") or {}).get("abbreviation") or ""))
        home = _norm_abbr(((sched.get("homeTeam") or {}).get("abbreviation") or ""))

        if not (start_time and away and home):
            continue

        dt_utc = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        dt_et = dt_utc.astimezone(_ET)
        game_date = dt_et.strftime("%Y%m%d")
        slugs.append(f"{game_date}-{away}-{home}")

    seen: set[str] = set()
    out: list[str] = []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------- crawler core ----------

def crawl_season_incremental(
    *,
    conn,
    client: MySportsFeedsClient,
    season: Season,
    start_date: date,
    end_date: date,
    commit_every: int = 50,
    force_meta: bool = False,
    force_lineups: bool = False,
) -> None:
    log.info(
        "Crawling season=%s start=%s end=%s force_meta=%s force_lineups=%s",
        season.season_slug, start_date, end_date, force_meta, force_lineups,
    )

    saved = 0

    def record_save() -> None:
        nonlocal saved
        saved += 1
        if commit_every and saved % commit_every == 0:
            conn.commit()
            log.info("Committed batch: saved=%d", saved)

    provider = "mysportsfeeds"
    today_et = et_today()

    # Pre-load per-season bulk caches — one DB query each instead of N individual
    # already_fetched() calls spread across the date loop.
    lineup_cached: set = _load_url_set(
        conn, provider=provider, endpoint="lineup",
        season=season.season_slug, as_of_date=None,
    )
    completed_slugs: set = _load_all_completed_boxscores(conn)
    log.info(
        "Pre-loaded caches season=%s: %d lineup URLs, %d completed boxscores",
        season.season_slug, len(lineup_cached), len(completed_slugs),
    )

    # 0) Meta endpoints (daily snapshots)
    meta = [
        ("injuries", build_url_injuries(season)),
        ("standings", build_url_standings(season)),
        ("venues", build_url_venues(season)),
    ]
    for endpoint, url in meta:
        if not force_meta and already_fetched(
            conn,
            provider=provider,
            endpoint=endpoint,
            season=(season.season_slug if endpoint != "injuries" else None),
            game_slug=None,
            as_of_date=today_et,
            url=url,
        ):
            log.info("Skip meta (already fetched): %s", endpoint)
            continue

        try:
            payload = client.fetch_json(url)
        except Exception:
            log.warning("Skipping failed fetch meta endpoint=%s url=%s", endpoint, url, exc_info=True)
            continue

        save_api_response(
            conn,
            provider=provider,
            endpoint=endpoint,
            season=(season.season_slug if endpoint != "injuries" else None),
            as_of_date=today_et,
            url=url,
            payload=payload,
        )
        log.info("Saved %s", endpoint)
        record_save()

    # 1) Per-day: games_by_date + derive slugs
    for d in daterange(start_date, end_date):
        games_url = build_url_games_by_date(season, d)

        if already_fetched(
            conn,
            provider=provider,
            endpoint="games_by_date",
            season=season.season_slug,
            game_slug=None,
            as_of_date=d,
            url=games_url,
        ):
            log.info("Skip games_by_date (already fetched): %s", d)
            # Reuse saved payload slugs to still process per-game endpoints if missing.
            slugs = _load_slugs_from_saved_games_by_date(conn, season=season.season_slug, as_of_date=d)
        else:
            try:
                payload = client.fetch_json(games_url)
            except Exception:
                log.warning("Skipping failed fetch games_by_date date=%s url=%s", d, games_url, exc_info=True)
                continue

            save_api_response(
                conn,
                provider=provider,
                endpoint="games_by_date",
                season=season.season_slug,
                as_of_date=d,
                url=games_url,
                payload=payload,
            )
            log.info("Saved games_by_date date=%s", d)
            record_save()
            slugs = extract_game_slugs_from_games_payload(payload)

        if not slugs:
            continue

        # 2) Per-game endpoints for that date's slugs.
        # If this is a future date, boxscore won't exist yet (204). Skip it.
        # Note: MLB V1 has no playbyplay endpoint — only boxscore + lineup.
        is_future_date = d > today_et

        # Build task list using pre-loaded caches (batch check, no per-slug DB queries).
        _pg_tasks: list = []
        # Lineups are volatile before first pitch.  A cached early-morning
        # payload can be empty or projected, so refresh the active slate even
        # when the lineup URL was seen before.
        _active_lineup_window = (
            (today_et - timedelta(days=1)) <= d <= (today_et + timedelta(days=1))
        )
        for _slug in slugs:
            if not is_future_date and _slug not in completed_slugs:
                _pg_tasks.append(("boxscore", _slug, build_url_boxscore(season, _slug)))
            _lu_url = build_url_lineup(season, _slug)
            if force_lineups or _active_lineup_window or _lu_url not in lineup_cached:
                _pg_tasks.append(("lineup", _slug, _lu_url))

        if _pg_tasks:
            _rate_hit_pg = threading.Event()
            _pg_fut_map: dict = {}
            with ThreadPoolExecutor(max_workers=8) as _ex:
                for _ep, _slug, _url in _pg_tasks:
                    _fut = _ex.submit(
                        _fetch_url_tagged,
                        client, _url,
                        f"endpoint={_ep} game_slug={_slug}",
                        d, _rate_hit_pg,
                    )
                    _pg_fut_map[_fut] = (_ep, _slug, _url)
                for _fut in as_completed(_pg_fut_map):
                    _ep, _slug, _url = _pg_fut_map[_fut]
                    _res = _fut.result()
                    if _res is not None:
                        _, _payload = _res
                        save_api_response(
                            conn,
                            provider=provider,
                            endpoint=_ep,
                            season=season.season_slug,
                            game_slug=_slug,
                            url=_url,
                            payload=_payload,
                        )
                        if _ep == "lineup":
                            lineup_cached.add(_url)
                        record_save()

            if _rate_hit_pg.is_set():
                conn.commit()
                return

            log.info(
                "Per-game date=%s slugs=%d tasks=%d saved=%d",
                d, len(slugs), len(_pg_tasks), saved,
            )

        if d > today_et:
            continue

        # 3) Player gamelogs per team for the date — batch check + parallel HTTP.
        # One DB query replaces 30 individual already_fetched() calls; ThreadPoolExecutor
        # removes the time.sleep(0.25) sequential delay (7.5 s/day → ~1 s wall time).
        _gamelog_cached = _load_url_set(
            conn, provider=provider, endpoint="player_gamelogs",
            season=season.season_slug, as_of_date=d,
        )
        _gl_tasks = [
            (_team, _url)
            for _team in MLB_TEAMS
            if (_url := build_url_player_gamelogs(season, d, _team)) not in _gamelog_cached
        ]
        if _gl_tasks:
            _rate_hit_gl = threading.Event()
            _gl_fut_map: dict = {}
            with ThreadPoolExecutor(max_workers=8) as _ex:
                for _team, _url in _gl_tasks:
                    _fut = _ex.submit(
                        _fetch_url_tagged,
                        client, _url,
                        f"gamelog date={d} team={_team}",
                        d, _rate_hit_gl,
                    )
                    _gl_fut_map[_fut] = (_team, _url)
                for _fut in as_completed(_gl_fut_map):
                    _team, _url = _gl_fut_map[_fut]
                    _res = _fut.result()
                    if _res is not None:
                        _, _payload = _res
                        save_api_response(
                            conn,
                            provider=provider,
                            endpoint="player_gamelogs",
                            season=season.season_slug,
                            as_of_date=d,
                            url=_url,
                            payload=_payload,
                        )
                        record_save()

            if _rate_hit_gl.is_set():
                conn.commit()
                return

        conn.commit()
        log.info("Committed end-of-day date=%s (saved=%d)", d, saved)

    conn.commit()
    log.info("Season incremental complete: saved=%d", saved)


def _load_slugs_from_saved_games_by_date(conn, *, season: str, as_of_date: date) -> list[str]:
    """
    If we already have games_by_date in raw.api_responses, reuse it
    instead of refetching just to get slugs.
    """
    q = """
    SELECT payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='games_by_date'
      AND season=%s
      AND as_of_date=%s
    ORDER BY fetched_at_utc DESC
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (season, as_of_date))
        row = cur.fetchone()
    if not row or row[0] is None:
        return []
    payload = row[0]
    return extract_game_slugs_from_games_payload(payload)


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="MySportsFeeds MLB incremental crawler")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Force start date (YYYY-MM-DD). Overrides incremental lookback. Use for backfill.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Force end date (YYYY-MM-DD). Defaults to today+1 (include_tomorrow). Use for backfill.",
    )
    parser.add_argument(
        "--force-meta",
        action="store_true",
        help="Re-fetch meta endpoints (injuries/standings/venues) even if already fetched today.",
    )
    parser.add_argument(
        "--force-lineups",
        action="store_true",
        help="Re-fetch lineup endpoints even if their URLs were already cached.",
    )
    args = parser.parse_args()

    cfg = CrawlerConfig()
    client = MySportsFeedsClient(api_key=cfg.api_key)

    seasons = [
        Season(league="mlb", season_slug="2024-regular",
               season_start=date(2024, 3, 20), season_end=date(2024, 11, 2)),
        Season(league="mlb", season_slug="2025-regular",
               season_start=date(2025, 3, 18), season_end=date(2025, 11, 2)),
        Season(league="mlb", season_slug="2026-regular",
               season_start=date(2026, 3, 26), season_end=date(2026, 11, 2)),
    ]

    today = et_today()
    default_end = today + (timedelta(days=1) if cfg.include_tomorrow else timedelta(days=0))
    force_start: Optional[date] = date.fromisoformat(args.start_date) if args.start_date else None
    force_end: Optional[date] = date.fromisoformat(args.end_date) if args.end_date else None

    with psycopg2.connect(cfg.pg_dsn) as conn:
        conn.autocommit = False
        try:
            for s in seasons:
                end = force_end if force_end is not None else default_end

                if force_start is not None:
                    start = force_start
                else:
                    last = last_games_by_date_asof(conn, provider="mysportsfeeds", season=s.season_slug)

                    # If never crawled games_by_date, start at (today - lookback) to bootstrap
                    if last is None:
                        start = today - timedelta(days=cfg.lookback_days)
                    else:
                        # crawl forward, but include a small lookback to capture late stat corrections
                        start = min(last, today) - timedelta(days=cfg.lookback_days)

                if start > end:
                    start = end

                # Clamp to season bounds — avoids requesting future dates against old seasons
                # (e.g. requesting 2026 dates from 2024-regular → HTTP 500)
                start = max(start, s.season_start)
                end = min(end, s.season_end)
                if start > end:
                    log.info("Skipping season=%s (crawl window outside season bounds)", s.season_slug)
                    continue

                crawl_season_incremental(
                    conn=conn,
                    client=client,
                    season=s,
                    start_date=start,
                    end_date=end,
                    commit_every=cfg.commit_every,
                    force_meta=args.force_meta,
                    force_lineups=args.force_lineups,
                )

            conn.commit()
        except Exception:
            conn.rollback()
            log.exception("Crawl failed; rolled back")
            raise


if __name__ == "__main__":
    main()
