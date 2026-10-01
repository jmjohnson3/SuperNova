"""Resolve FanDuel betslip IDs for the bettor's state.

FanDuel market IDs are per state (CO 708.x, NJ 734.x, OH 739.x, IL 717.x ...); only selection IDs are
shared. Provider links carry another state's market ID (SportsGameOdds: IL 717.x; The Odds API: 42.x),
which FanDuel rejects with "Selection not added". This looks the bet up in FanDuel's public event feed
for NFL_FANDUEL_STATE by teams, player, stat, side and exact line, and returns that state's
(marketId, selectionId). Anything that does not match exactly returns None, so the link is shown as
manual selection rather than adding a different line. Publication only; forecasts are not touched.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

import requests

from nfl_pipeline.markets import normalize_name, normalize_team

APP_KEY = "FhMFpcPWXMeyZxOx"  # public key used by FanDuel's own web client
TIMEOUT_S = 8
BUDGET_S = 90  # total network time per process; after that, links fall back to manual selection
MARKET_NAME = {"receiving_yards": "Receiving Yds", "rushing_yards": "Rushing Yds", "passing_yards": "Passing Yds",
               "receptions": "Total Receptions", "passing_tds": "Passing TDs"}
TAB = {"receiving_yards": "receiving-props", "receptions": "receiving-props", "rushing_yards": "rushing-props",
       "passing_yards": "passing-props", "passing_tds": "passing-props", "spread": "", "total": ""}
_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")


def state() -> str | None:
    value = (os.getenv("NFL_FANDUEL_STATE") or "").strip().lower()
    return value if re.fullmatch(r"[a-z]{2}", value) else None


def _player(name) -> str:
    return _SUFFIX.sub("", normalize_name(name))


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


class Resolver:
    def __init__(self, st: str, fetch: Callable[[str, dict], dict] | None = None):
        self.state = st
        self._fetch = fetch or self._http
        self._events: dict[tuple[str, str], str] | None = None
        self._markets: dict[tuple[str, str], dict] = {}
        self._spent = 0.0

    def _http(self, path: str, params: dict) -> dict:
        if self._spent >= BUDGET_S:
            raise TimeoutError("FanDuel lookup budget spent")
        started = time.monotonic()
        try:
            r = requests.get(f"https://sbapi.{self.state}.sportsbook.fanduel.com/api/{path}",
                             params=dict(params, _ak=APP_KEY), headers={"User-Agent": "Mozilla/5.0"}, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        finally:
            self._spent += time.monotonic() - started

    def event_id(self, away: str, home: str) -> str | None:
        if self._events is None:
            self._events = {}
            try:
                page = self._fetch("content-managed-page", {"page": "CUSTOM", "customPageId": "nfl"})
            except Exception:
                return None
            for eid, ev in (page.get("attachments", {}).get("events") or {}).items():
                teams = str(ev.get("name") or "").split(" @ ")
                if len(teams) == 2:
                    self._events[(normalize_team(teams[0]), normalize_team(teams[1]))] = str(ev.get("eventId") or eid)
        return self._events.get((normalize_team(away), normalize_team(home)))

    def markets(self, event_id: str, tab: str) -> dict:
        key = (event_id, tab)
        if key not in self._markets:
            params = {"eventId": event_id}
            if tab:
                params["tab"] = tab
            try:
                self._markets[key] = self._fetch("event-page", params).get("attachments", {}).get("markets") or {}
            except Exception:
                return {}  # not cached: a later row may retry within the budget
        return self._markets[key]

    def resolve(self, *, away, home, stat, side, line, player=None) -> tuple[str, str] | None:
        if stat not in TAB or side is None or line is None:
            return None
        event = self.event_id(away, home)
        if event is None:
            return None
        for market_id, market in self.markets(event, TAB[stat]).items():
            name = str(market.get("marketName") or "")
            runners = market.get("runners") or []
            if stat == "total":
                if name != "Total Points":
                    continue
                pick = [r for r in runners if str(r.get("runnerName")).lower() == side]
            elif stat == "spread":
                if name != "Spread":
                    continue
                team = home if side == "home" else away if side == "away" else None
                pick = [r for r in runners if normalize_team(r.get("runnerName")) == normalize_team(team)]
            else:
                who, _, label = name.rpartition(" - ")
                if label != MARKET_NAME[stat] or _player(who) != _player(player):
                    continue
                pick = [r for r in runners if str(r.get("runnerName") or "").lower().endswith(" " + side)]
            if len(pick) == 1 and _same(pick[0].get("handicap"), line) and pick[0].get("runnerStatus", "ACTIVE") == "ACTIVE":
                return str(market_id), str(pick[0]["selectionId"])
        return None


_resolver: Resolver | None = None


def resolver() -> Resolver | None:
    global _resolver
    st = state()
    if st is None:
        return None
    if _resolver is None or _resolver.state != st:
        _resolver = Resolver(st)
    return _resolver


def _teams(row: dict) -> tuple[str | None, str | None]:
    away, home = row.get("away_team_abbr") or row.get("away"), row.get("home_team_abbr") or row.get("home")
    if (not away or not home) and row.get("game_id"):
        parts = str(row["game_id"]).split("_")  # 2026_04_PIT_CLE = season_week_away_home
        if len(parts) == 4:
            away, home = parts[2], parts[3]
    return away, home


def resolve_row(row: dict) -> tuple[str, str] | None:
    """(marketId, selectionId) for this state, or None (not configured, no exact match, or lookup failed)."""
    r = resolver()
    if r is None:
        return None
    away, home = _teams(row)
    stat = row.get("market") if row.get("market") in ("spread", "total") else row.get("stat")
    try:
        return r.resolve(away=away, home=home, stat=stat, side=row.get("side"), line=row.get("line"),
                         # the book's own spelling first: forecast player_name is short ("A.Rodgers")
                         player=row.get("offer_player_name") or row.get("player_name") or row.get("player"))
    except Exception:
        return None
