"""One scan: get snapshots from a source, clean them, store the prices, find and store the opportunities.

Sources:
  sample  invented data shipped with the package (no key, no database)
  saved   snapshots the existing NBA/NFL scanners already keep in Postgres (real prices, no API credits)
  live    The Odds API now (spends credits: one per market per game)
  auto    saved when it has anything recent, otherwise sample. Never live: spending credits is always explicit.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from sports_sniper import db
from sports_sniper.config import Settings
from sports_sniper.markets import SPORT_KEYS
from sports_sniper.normalize import normalize_event, parse_time
from sports_sniper.opportunities import Opportunity, current, detect
from sports_sniper.providers import Snapshot, sample, saved_quotes
from sports_sniper.providers.odds_api import OddsApiClient, OddsApiError, QuotaGuard

log = logging.getLogger(__name__)
SOURCES = ("auto", "sample", "saved", "live")


@dataclass
class ScanResult:
    scan_id: int
    source: str
    sports: list[str]
    events: int
    quotes: int
    new_quotes: int
    opportunities: list[Opportunity]            # from every snapshot scanned
    current: list[Opportunity]                  # from the newest look at each game only
    dropped: Counter = field(default_factory=Counter)
    credits_used: int | None = None
    credits_remaining: int | None = None
    note: str | None = None


async def fetch_live(settings: Settings, sports: list[str], now: datetime, client: OddsApiClient | None = None,
                     ) -> tuple[list[Snapshot], dict]:
    """Upcoming games inside the look-ahead window, soonest first, until the scan's credit budget runs out."""
    own = client is None
    client = client or OddsApiClient(settings.api_key() or "", base_url=settings.odds_api_base_url,
                                     timeout=settings.request_timeout_seconds, max_retries=settings.max_retries,
                                     cache_seconds=settings.cache_seconds, credit_floor=settings.credit_floor,
                                     max_credits=settings.max_credits_per_scan)
    snapshots, notes = [], []
    try:
        await client.sports()                       # free; tells us how many credits are left before spending any
        horizon = now + timedelta(hours=settings.lookahead_hours)
        upcoming = []
        for sport in sports:
            for event in await client.events(SPORT_KEYS[sport]):
                start = parse_time(event.get("commence_time"))
                if start is not None and now < start <= horizon:
                    upcoming.append((start, sport, event["id"]))
        for _, sport, event_id in sorted(upcoming):
            try:
                event = await client.event_odds(SPORT_KEYS[sport], event_id, settings.markets_for(sport), settings.bookmakers)
            except QuotaGuard as stop:
                notes.append(f"stopped early: {stop}")
                break
            if event is not None:
                snapshots.append(Snapshot(sport, datetime.now(timezone.utc), event))
        if client.unsupported:
            notes.append("skipped unsupported markets: " + ", ".join(sorted(client.unsupported)))
    finally:
        quota = client.quota
        if own:
            await client.close()
    return snapshots, dict(credits_used=quota.spent, credits_remaining=quota.remaining, note="; ".join(notes) or None)


def collect(source: str, settings: Settings, sports: list[str], now: datetime) -> tuple[str, list[Snapshot], dict]:
    """(the source actually used, snapshots, details)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source}")
    if source == "live":
        snapshots, meta = asyncio.run(fetch_live(settings, sports, now))
        return "live", snapshots, meta
    if source in ("auto", "saved"):
        try:
            snapshots = saved_quotes.load(settings.saved_quotes_dsn, sports, now - timedelta(hours=settings.lookahead_hours))
        except Exception as error:               # the existing database is optional
            log.warning("saved quotes unavailable (%s)", type(error).__name__)
            snapshots = []
        if snapshots or source == "saved":
            return "saved", snapshots, dict(note=None if snapshots else "no saved snapshots in the look-back window")
    return "sample", sample.load(now), dict(note="hypothetical sample data")


def run_scan(settings: Settings, engine: Engine, source: str = "auto", *, sports: list[str] | None = None,
             now: datetime | None = None, snapshots: list[Snapshot] | None = None) -> ScanResult:
    """Run one scan and store it. `snapshots` bypasses the source (tests, replays)."""
    now = now or datetime.now(timezone.utc)
    sports = [s for s in (sports or settings.sports) if s in SPORT_KEYS]
    meta: dict = {}
    if snapshots is None:
        try:
            source, snapshots, meta = collect(source, settings, sports, now)
        except OddsApiError as error:
            snapshots, meta = [], dict(note=f"odds API: {error}")
    quotes, dropped = [], Counter()
    newest: dict[str, tuple[str, datetime]] = {}
    for snap in snapshots:
        event_id = str((snap.event.get("data") or snap.event).get("id") or "")
        if event_id and (event_id not in newest or snap.observed_at > newest[event_id][1]):
            newest[event_id] = (snap.sport, snap.observed_at)
        found, counts = normalize_event(snap.event, snap.sport, snap.observed_at, max_age_seconds=settings.max_quote_age_seconds)
        quotes += found
        dropped.update({k: v for k, v in counts.items() if k != "kept"})
    found = detect(quotes, settings)
    db.init_db(engine)
    with Session(engine) as session, session.begin():
        scan = db.Scan(started_at=now, source=source, sports=",".join(sports), events=len(newest),
                       quotes=len(quotes), opportunities=len(found), credits_used=meta.get("credits_used"),
                       credits_remaining=meta.get("credits_remaining"), note=meta.get("note"))
        session.add(scan)
        session.flush()
        db.touch_events(session, newest)
        scan.new_quotes = db.insert_new(session, db.QuoteRow, db.quote_rows(quotes, scan.id),
                                        ["event_id", "book", "market", "player_key", "line_key", "side", "last_update"])
        db.insert_new(session, db.OpportunityRow, db.opportunity_rows(found, scan.id, source),
                      ["event_id", "book", "market", "player_key", "line_key", "side", "observed_at"])
        result = ScanResult(scan_id=scan.id, source=source, sports=sports, events=scan.events, quotes=len(quotes),
                            new_quotes=scan.new_quotes, opportunities=found,
                            current=current(found, {k: v[1] for k, v in newest.items()}), dropped=dropped,
                            credits_used=meta.get("credits_used"), credits_remaining=meta.get("credits_remaining"),
                            note=meta.get("note"))
    return result
