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
import os
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

from mlb_pipeline.db import PG_DSN as _PG_DSN
_API_KEY  = os.getenv("ODDS_API_KEY", "")
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


def _parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(_UTC)


def _close_snapshot_utc(event: dict, minutes_before_start: int) -> str:
    """Return ISO-Z snapshot time close to first pitch for one event."""
    commence = _parse_utc(event["commence_time"])
    snapshot = commence - timedelta(minutes=int(minutes_before_start))
    return snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")


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
    fetched_key = fetched_at.astimezone(_UTC).strftime("%Y%m%dT%H%M%SZ")
    url_key = f"historical_props/{game_date}/{event_id}/{fetched_key}"
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
    snapshot_mode: str = "fixed",
    close_minutes_before_start: int = 30,
    force: bool = False,
) -> None:
    mode = str(snapshot_mode or "fixed").lower()
    if mode not in {"fixed", "close"}:
        raise ValueError(f"snapshot_mode must be fixed or close, got {snapshot_mode!r}")
    if close_minutes_before_start <= 0:
        raise ValueError("close_minutes_before_start must be positive")
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

        if not dry_run and mode == "fixed" and not force and _already_backfilled(conn, current):
            log.info("Already have props for %s — skipping", current)
            current += timedelta(days=1)
            continue

        snapshot_z = _snapshot_utc(current)
        log.info("--- %s  (events snapshot %s, mode=%s) ---", current, snapshot_z, mode)

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

        day_events_fetched = 0

        for ev in day_events:
            if max_credits and credits_spent >= max_credits:
                log.warning("--max-credits %d reached. Stopping.", max_credits)
                conn.close()
                return

            event_id   = ev["id"]
            home_team  = ev.get("home_team", "?")
            away_team  = ev.get("away_team", "?")
            event_snapshot_z = (
                _close_snapshot_utc(ev, close_minutes_before_start)
                if mode == "close"
                else snapshot_z
            )
            fetched_at = _parse_utc(event_snapshot_z)

            try:
                payload, cr = _get_historical_props(event_id, event_snapshot_z)
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
            log.info("  Saved %s @ %s (event=%s snapshot=%s) | credits_remaining=%s",
                     away_team, home_team, event_id, event_snapshot_z,
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
    parser.add_argument(
        "--snapshot-mode",
        choices=("fixed", "close"),
        default="fixed",
        help="Use the fixed daily snapshot or one close snapshot per event.",
    )
    parser.add_argument(
        "--close-minutes-before-start",
        type=int,
        default=30,
        help="For --snapshot-mode close, fetch this many minutes before first pitch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For fixed-mode backfills, fetch even when historical rows already exist.",
    )
    args = parser.parse_args()

    if not _API_KEY:
        parser.error("ODDS_API_KEY is required")

    end = args.end or (date.today() - timedelta(days=1))
    if args.start > end:
        parser.error(f"--start {args.start} is after --end {end}")

    run_backfill(
        start=args.start,
        end=end,
        max_credits=args.max_credits,
        dry_run=args.dry_run,
        sleep_s=args.sleep,
        snapshot_mode=args.snapshot_mode,
        close_minutes_before_start=args.close_minutes_before_start,
        force=args.force,
    )


if __name__ == "__main__":
    main()
