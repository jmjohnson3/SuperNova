# src/mlb_pipeline/crawler_oddsapi_props_backfill.py
"""
Backfill historical MLB player prop lines using The Odds API.

The live prop crawler (crawler_oddsapi.py) only runs daily going forward.
This script fetches historical prop snapshots for past game dates so we can
join book_line to training data (fix #1 for the prop models).

Cost: ~20 credits per game event.
  2025 season (~2,430 games) ≈ 48,600 credits.
  Run --dry-run first to verify estimated cost before committing.

Usage:
    # Estimate cost without spending credits
    python -m mlb_pipeline.crawler_oddsapi_props_backfill --dry-run

    # Backfill full 2025 season
    python -m mlb_pipeline.crawler_oddsapi_props_backfill --start 2025-03-27 --end 2025-09-28

    # Single date
    python -m mlb_pipeline.crawler_oddsapi_props_backfill --start 2025-04-01 --end 2025-04-01

    # Cap spend (stop after using N credits)
    python -m mlb_pipeline.crawler_oddsapi_props_backfill --start 2025-03-27 --max-credits 10000

Snapshot time: 18:00 UTC (2 PM ET) — prop lines are stable by then, before most games start.
Data is stored in raw.api_responses (endpoint='mlb_prop_odds_historical') and parsed into
odds.mlb_player_prop_lines by parse_oddsapi.py (same parser as the live feed).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg2
import requests

log = logging.getLogger("mlb_pipeline.crawler_oddsapi_props_backfill")

_UTC = ZoneInfo("UTC")
_ET  = ZoneInfo("America/New_York")

# Snapshot at 18:00 UTC (2 PM ET) — props are stable, most games haven't started
_SNAPSHOT_HOUR_UTC = 18

_PROP_MARKETS = (
    "pitcher_strikeouts,"
    "batter_hits,batter_hits_alternate,"
    "batter_home_runs,batter_home_runs_alternate,"
    "batter_total_bases,batter_total_bases_alternate,"
    "batter_walks,batter_walks_alternate"
)

_PG_DSN  = "postgresql://josh:password@localhost:5432/nba"
_API_KEY  = "5b6f0290e265c3329b3ed27897d79eaf"
_SPORT    = "baseball_mlb"

# Historical endpoints
_HIST_EVENTS_URL = f"https://api.the-odds-api.com/v4/historical/sports/{_SPORT}/events"
_HIST_ODDS_TMPL  = "https://api.the-odds-api.com/v4/historical/sports/{sport}/events/{event_id}/odds"

# Off-season: skip these months entirely
_OFF_SEASON_MONTHS = {11, 12, 1, 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot_utc(game_date: date) -> str:
    """Return ISO-Z snapshot time for a given ET game date."""
    dt = datetime(game_date.year, game_date.month, game_date.day,
                  _SNAPSHOT_HOUR_UTC, 0, 0, tzinfo=_UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _already_backfilled(conn, game_date: date) -> bool:
    """True if we have at least one historical prop row for this date."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM raw.api_responses
            WHERE provider = 'oddsapi'
              AND endpoint = 'mlb_prop_odds_historical'
              AND as_of_date = %s
            LIMIT 1
            """,
            (game_date,),
        )
        return cur.fetchone() is not None


def _save_raw(conn, game_date: date, event_id: str, payload: dict,
              fetched_at: datetime) -> None:
    """Persist raw API response to raw.api_responses."""
    import hashlib, json
    raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(raw_json.encode()).hexdigest()
    url_key = f"historical_props/{game_date}/{event_id}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw.api_responses
                (provider, endpoint, url, as_of_date, fetched_at_utc, payload, payload_sha256)
            VALUES
                ('oddsapi', 'mlb_prop_odds_historical', %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (provider, endpoint, url) DO UPDATE SET
                as_of_date    = EXCLUDED.as_of_date,
                fetched_at_utc = EXCLUDED.fetched_at_utc,
                payload       = EXCLUDED.payload,
                payload_sha256 = EXCLUDED.payload_sha256
            """,
            (url_key, game_date, fetched_at, raw_json, sha),
        )
    conn.commit()


def _get_historical_events(snapshot_z: str) -> tuple[list[dict], int | None]:
    """Fetch event list at a historical snapshot. Returns (events, credits_remaining)."""
    params = {
        "apiKey": _API_KEY,
        "date":   snapshot_z,
        "dateFormat": "iso",
    }
    r = requests.get(_HIST_EVENTS_URL, params=params, timeout=30)
    r.raise_for_status()
    credits = r.headers.get("x-requests-remaining")
    data = r.json()
    return data.get("data", []), (int(credits) if credits else None)


def _get_historical_props(event_id: str, snapshot_z: str) -> tuple[dict, int | None]:
    """Fetch prop odds for one event at a historical snapshot."""
    url = _HIST_ODDS_TMPL.format(sport=_SPORT, event_id=event_id)
    params = {
        "apiKey":      _API_KEY,
        "date":        snapshot_z,
        "markets":     _PROP_MARKETS,
        "regions":     "us",
        "bookmakers":  "draftkings,fanduel",
        "oddsFormat":  "american",
        "dateFormat":  "iso",
        "includeLinks": "true",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 404:
        return {}, None  # event expired — normal for very old data
    r.raise_for_status()
    credits = r.headers.get("x-requests-remaining")
    return r.json(), (int(credits) if credits else None)


def _et_day_window(game_date: date) -> tuple[datetime, datetime]:
    """Return UTC start/end datetimes for an ET calendar day."""
    start_et = datetime(game_date.year, game_date.month, game_date.day,
                        0, 0, tzinfo=_ET)
    end_et   = start_et + timedelta(days=1)
    return start_et.astimezone(_UTC), end_et.astimezone(_UTC)


def _filter_events_for_date(events: list[dict], game_date: date) -> list[dict]:
    """Keep only events whose commence_time falls in the ET game day."""
    start_utc, end_utc = _et_day_window(game_date)
    result = []
    for ev in events:
        ct = ev.get("commence_time", "")
        if not ct:
            continue
        ct_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=_UTC)
        if start_utc <= ct_dt < end_utc:
            result.append(ev)
    return result


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

def run_backfill(
    start: date,
    end: date,
    max_credits: int | None,
    dry_run: bool,
    min_credits_floor: int = 5_000,
    sleep_s: float = 0.3,
) -> None:
    conn = psycopg2.connect(_PG_DSN)
    credits_spent = 0
    days_done = 0
    events_done = 0

    current = start
    while current <= end:
        if current.month in _OFF_SEASON_MONTHS:
            log.debug("Skipping off-season date %s", current)
            current += timedelta(days=1)
            continue

        if not dry_run and _already_backfilled(conn, current):
            log.info("Already have props for %s — skipping", current)
            current += timedelta(days=1)
            continue

        snapshot_z = _snapshot_utc(current)
        log.info("--- %s  (snapshot %s) ---", current, snapshot_z)

        if dry_run:
            # Estimate only: fetch events list (1 credit) then multiply by 20/event
            try:
                events, cr = _get_historical_events(snapshot_z)
                day_events = _filter_events_for_date(events, current)
                est_cost = len(day_events) * 20
                print(f"  {current}  events={len(day_events)}  est_credits={est_cost}"
                      f"  credits_remaining={cr}")
                credits_spent += 1  # events call
            except Exception as exc:
                print(f"  {current}  ERROR: {exc}")
            current += timedelta(days=1)
            continue

        # Real fetch
        try:
            events, cr = _get_historical_events(snapshot_z)
            credits_spent += 1
            time.sleep(sleep_s)
        except Exception as exc:
            log.warning("Events fetch failed for %s: %s", current, exc)
            current += timedelta(days=1)
            continue

        if cr is not None and cr < min_credits_floor:
            log.error("Credit floor reached (%d remaining). Stopping.", cr)
            break

        day_events = _filter_events_for_date(events, current)
        log.info("  %d events on %s", len(day_events), current)

        fetched_at = datetime.now(_UTC)
        day_events_fetched = 0

        for ev in day_events:
            if max_credits and credits_spent >= max_credits:
                log.warning("--max-credits %d reached. Stopping.", max_credits)
                conn.close()
                return

            event_id   = ev["id"]
            home_team  = ev.get("home_team", "?")
            away_team  = ev.get("away_team", "?")

            try:
                payload, cr = _get_historical_props(event_id, snapshot_z)
                credits_spent += 20  # historical event odds cost
                time.sleep(sleep_s)
            except Exception as exc:
                log.warning("Props fetch failed for event %s: %s", event_id, exc)
                continue

            if not payload:
                log.debug("Empty/404 for event %s — skipping", event_id)
                continue

            bookmakers = payload.get("data", payload).get("bookmakers", [])
            if not bookmakers:
                log.debug("No bookmakers for event %s (%s@%s)", event_id, away_team, home_team)
                continue

            _save_raw(conn, current, event_id, payload, fetched_at)
            day_events_fetched += 1
            log.info("  Saved %s @ %s (event=%s) | credits_remaining=%s",
                     away_team, home_team, event_id,
                     cr if cr is not None else "unknown")

            if cr is not None and cr < min_credits_floor:
                log.error("Credit floor reached (%d remaining). Stopping.", cr)
                conn.close()
                return

        events_done  += day_events_fetched
        days_done    += 1
        current      += timedelta(days=1)

    conn.close()
    print(f"\nBackfill complete: {days_done} days, {events_done} events saved, "
          f"~{credits_spent} credits spent.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Backfill historical MLB player prop lines from The Odds API"
    )
    parser.add_argument(
        "--start", required=True, type=date.fromisoformat,
        help="First game date to backfill (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default=None, type=date.fromisoformat,
        help="Last game date (inclusive). Defaults to yesterday."
    )
    parser.add_argument(
        "--max-credits", type=int, default=None,
        help="Stop after spending this many credits (safety cap)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch event lists only (1 credit/day) and print cost estimate"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="Seconds to sleep between API calls (default 0.3)"
    )
    args = parser.parse_args()

    end = args.end or (date.today() - timedelta(days=1))
    if args.start > end:
        parser.error(f"--start {args.start} is after --end {end}")

    run_backfill(
        start=args.start,
        end=end,
        max_credits=args.max_credits,
        dry_run=args.dry_run,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()
