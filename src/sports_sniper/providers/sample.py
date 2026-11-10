"""Hypothetical sample data, so the application runs with no key and no database. Nothing in it is real:
the teams, players and prices are invented, and every screen that shows it says so."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

from sports_sniper.markets import SPORT_BY_KEY
from sports_sniper.providers import Snapshot

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_odds.json"


def _stamp(now: datetime) -> str:
    return now.isoformat().replace("+00:00", "Z")


def _dated(event: dict, observed: datetime) -> dict:
    """Fill the fixture's relative times: games start `commence_in_minutes` from now, prices are 40 seconds old."""
    event = copy.deepcopy(event)
    event["commence_time"] = _stamp(observed + timedelta(minutes=event.pop("commence_in_minutes")))
    for book in event["bookmakers"]:
        book["last_update"] = _stamp(observed - timedelta(seconds=40))
        for market in book["markets"]:
            market["last_update"] = book["last_update"]
    return event


def load(now: datetime) -> list[Snapshot]:
    """Two snapshots of each sample game: one 30 minutes ago (before the listed price changes) and one now."""
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    out = []
    for raw in doc["events"]:
        sport = SPORT_BY_KEY[raw["sport_key"]]
        current = _dated(raw, now)
        earlier = _dated(raw, now - timedelta(minutes=doc["earlier"]["minutes_ago"]))
        earlier["commence_time"] = current["commence_time"]
        for change in doc["earlier"]["changes"]:
            if change["event"] != raw["id"]:
                continue
            for book in earlier["bookmakers"]:
                for market in book["markets"]:
                    for outcome in market["outcomes"]:
                        if (book["key"], market["key"], outcome["description"], outcome["name"]) == (
                                change["book"], change["market"], change["player"], change["name"]):
                            outcome["price"] = change["price"]
        out += [Snapshot(sport, now - timedelta(minutes=doc["earlier"]["minutes_ago"]), earlier), Snapshot(sport, now, current)]
    return sorted(out, key=lambda s: s.observed_at)
