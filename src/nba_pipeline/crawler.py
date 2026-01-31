import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import psycopg2
from zoneinfo import ZoneInfo

from nba_pipeline.fetcher import MySportsFeedsClient
from nba_pipeline.raw_store import save_api_response

log = logging.getLogger("nba_pipeline.crawler")

TEAM_ABBRS = [
    "atl","bos","bkn","cha","chi","cle","dal","den","det","gsw",
    "hou","ind","lac","lal","mem","mia","mil","min","nop","nyk",
    "okc","orl","phi","phx","por","sac","sas","tor","uta","was",
]

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Season:
    league: str
    season_slug: str

@dataclass(frozen=True)
class CrawlerConfig:
    api_key: str = "4359aa1b-cc29-4647-a3e5-7314e2"
    pg_dsn: str = "postgresql://josh:password@localhost:5432/nba"

    # “incremental window” behavior:
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
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/date/{yyyymmdd(d)}/games.json"

def build_url_boxscore(season: Season, game_slug: str) -> str:
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/games/{game_slug}/boxscore.json"

def build_url_playbyplay(season: Season, game_slug: str) -> str:
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/games/{game_slug}/playbyplay.json"

def build_url_lineup(season: Season, game_slug: str) -> str:
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/games/{game_slug}/lineup.json"

def build_url_player_gamelogs(season: Season, d: date, team: str) -> str:
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/date/{yyyymmdd(d)}/player_gamelogs.json?team={team}"

def build_url_injuries(season: Season) -> str:
    # MySportsFeeds injuries endpoint is not season-scoped in the URL
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/injuries.json"

def build_url_standings(season: Season) -> str:
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/standings.json"

def build_url_venues(season: Season) -> str:
    return f"https://api.mysportsfeeds.com/v2.1/pull/{season.league}/{season.season_slug}/venues.json"


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


def last_games_by_date_asof(conn, *, provider: str, season: str) -> Optional[date]:
    """
    Returns the max as_of_date we have for games_by_date for this season.
    If you’ve never run the incremental crawler yet, this might be NULL.
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


# ---------- payload parsing helpers ----------

def extract_game_slugs_from_games_payload(payload: dict) -> list[str]:
    slugs: list[str] = []
    games = payload.get("games") or []
    for g in games:
        sched = g.get("schedule") or {}
        start_time = sched.get("startTime")  # e.g. "2024-10-23T02:00:00.000Z"
        away = ((sched.get("awayTeam") or {}).get("abbreviation") or "").upper()
        home = ((sched.get("homeTeam") or {}).get("abbreviation") or "").upper()
        if not (start_time and away and home):
            continue

        dt_utc = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        dt_et = dt_utc.astimezone(_ET)
        game_date = dt_et.strftime("%Y%m%d")
        slugs.append(f"{game_date}-{away}-{home}")

    # dedupe preserve order
    seen = set()
    out = []
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
) -> None:
    log.info("Crawling season=%s start=%s end=%s", season.season_slug, start_date, end_date)

    saved = 0

    def record_save() -> None:
        nonlocal saved
        saved += 1
        if commit_every and saved % commit_every == 0:
            conn.commit()
            log.info("Committed batch: saved=%d", saved)

    provider = "mysportsfeeds"
    today_et = et_today()

    # 0) Meta endpoints (daily snapshots)
    meta = [
        ("injuries", build_url_injuries(season)),
        ("standings", build_url_standings(season)),
        ("venues", build_url_venues(season)),
    ]
    for endpoint, url in meta:
        if already_fetched(
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
            # We still need slugs to do per-game endpoints if those are missing.
            # If you want to avoid refetching, you can optionally load the slugs from DB payload here.
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

        # 2) Per-game endpoints for that date’s slugs
        for i, slug in enumerate(slugs, start=1):
            for endpoint_name, url_builder in (
                ("boxscore", build_url_boxscore),
                ("playbyplay", build_url_playbyplay),
                ("lineup", build_url_lineup),
            ):
                url = url_builder(season, slug)

                if already_fetched(
                    conn,
                    provider=provider,
                    endpoint=endpoint_name,
                    season=season.season_slug,
                    game_slug=slug,
                    as_of_date=None,
                    url=url,
                ):
                    continue

                try:
                    payload = client.fetch_json(url)
                except Exception:
                    log.warning("Skipping failed fetch endpoint=%s game_slug=%s url=%s", endpoint_name, slug, url, exc_info=True)
                    continue

                save_api_response(
                    conn,
                    provider=provider,
                    endpoint=endpoint_name,
                    season=season.season_slug,
                    game_slug=slug,
                    url=url,
                    payload=payload,
                )
                record_save()

            if i % 25 == 0:
                log.info("Progress date=%s games=%d/%d (saved=%d)", d, i, len(slugs), saved)

        # 3) Player gamelogs per team for the date
        for team in TEAM_ABBRS:
            url = build_url_player_gamelogs(season, d, team)

            if already_fetched(
                conn,
                provider=provider,
                endpoint="player_gamelogs",
                season=season.season_slug,
                game_slug=None,
                as_of_date=d,
                url=url,
            ):
                continue

            try:
                payload = client.fetch_json(url)
            except Exception:
                log.warning("Skipping failed fetch endpoint=player_gamelogs date=%s team=%s", d, team, exc_info=True)
                continue

            save_api_response(
                conn,
                provider=provider,
                endpoint="player_gamelogs",
                season=season.season_slug,
                as_of_date=d,
                url=url,
                payload=payload,
            )
            record_save()

        conn.commit()
        log.info("Committed end-of-day date=%s (saved=%d)", d, saved)

    conn.commit()
    log.info("Season incremental complete: saved=%d", saved)


def _load_slugs_from_saved_games_by_date(conn, *, season: str, as_of_date: date) -> list[str]:
    """
    Optional: if we already have games_by_date in raw.api_responses, reuse it
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = CrawlerConfig()
    client = MySportsFeedsClient(api_key=cfg.api_key)

    seasons = [
        Season(league="nba", season_slug="2024-2025-regular"),
        Season(league="nba", season_slug="2025-2026-regular"),
    ]

    today = et_today()
    end = today + (timedelta(days=1) if cfg.include_tomorrow else timedelta(days=0))

    with psycopg2.connect(cfg.pg_dsn) as conn:
        conn.autocommit = False
        try:
            for s in seasons:
                last = last_games_by_date_asof(conn, provider="mysportsfeeds", season=s.season_slug)

                # If never crawled games_by_date, start at (today - lookback) to bootstrap
                if last is None:
                    start = today - timedelta(days=cfg.lookback_days)
                else:
                    # crawl forward, but include a small lookback to capture late stat corrections
                    start = min(last, today) - timedelta(days=cfg.lookback_days)

                if start > end:
                    start = end

                crawl_season_incremental(
                    conn=conn,
                    client=client,
                    season=s,
                    start_date=start,
                    end_date=end,
                    commit_every=cfg.commit_every,
                )

            conn.commit()
        except Exception:
            conn.rollback()
            log.exception("Crawl failed; rolled back")
            raise


if __name__ == "__main__":
    main()
