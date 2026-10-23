"""Scan NBA props, NBA game lines and NCAAF game lines against Pinnacle, as research.

Every run (the Sharp-Multi-Watch task, every 10 minutes):

- Game lines: one bulk Odds API call per sport (3 credits covers every game) whenever a game starts
  within the next hour. Spreads, moneylines and totals; every side of every line.
- NBA player props: one call per game (4 credits: points, rebounds, assists, threes) at roughly
  T-40, T-30, T-20 and T-10, the last being the close every decision is graded against.

Pricing is exact-line only: a book's price is valued at Pinnacle's no-vig probability for the same
market, player and line (power de-vig, as for NFL). Quotes must be fresh (source update within five
minutes) and the two books' quotes within two minutes of each other. A gap of +3% or better seen
15-45 minutes before the start is a research decision; +1% to +3% is logged. Nothing is staked or
pinged: a nightly digest reports what the scanner would have bet and how it held at the close.

Credits: a daily allowance per run keeps the monthly plan solvent and holds back a reserve for NFL
Sundays before the plan resets (NFL_ODDS_API_RESET_DAY). Lines are polled before props.

    python -m sharp_pipeline.scan            # one run (what the task does)
    python -m sharp_pipeline.scan --dry-run  # plan only: what would be polled, no API calls
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline import sharp_math
from nfl_pipeline.context_contract import safe_time
from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import normalize_name

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "sharp_multi_state.json"
BASE = "https://api.the-odds-api.com/v4/sports"
_ET = ZoneInfo("America/New_York")

BET_BOOKS = ("fanduel", "espnbet", "draftkings")
SHARP_BOOK = "pinnacle"
BOOKMAKERS = ("fanduel", "espnbet", "draftkings", "pinnacle", "betmgm", "fanatics")  # <= 10: one region
SPORTS = {
    "nba": dict(key="basketball_nba", lines=True, props=("player_points", "player_rebounds", "player_assists", "player_threes")),
    "ncaaf": dict(key="americanfootball_ncaaf", lines=True, props=()),  # college props cannot be bet in Colorado
}
LINE_MARKETS = ("h2h", "spreads", "totals")
PROP_STAT = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists",
             "player_threes": "threes"}

MIN_EV = 0.03
LOG_MIN_EV = 0.01
DECISION_WINDOW = (15, 45)        # minutes before the start, as for NFL
LINES_HORIZON = 60                 # poll game lines when any game starts within this many minutes
PROPS_HORIZON = 45                 # NBA props: polls inside the final 45 minutes...
PROPS_SPACING = 9.5                # ...at most one per task run (T-40, T-30, T-20, T-10)
MAX_SOURCE_AGE = 300
MAX_SOURCE_SKEW = 120
CREDIT_FLOOR = 150
NFL_SUNDAY_RESERVE = 700           # credits held back per NFL Sunday still to come before the reset
NFL_PRIMETIME_RESERVE = 60         # per Thursday/Monday game still to come

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.sharp_multi_quotes (
  observation_id TEXT PRIMARY KEY, sport TEXT NOT NULL, event_id TEXT NOT NULL,
  observed_at_utc TIMESTAMPTZ NOT NULL, commence_time_utc TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS sharp_multi_quotes_event ON bets.sharp_multi_quotes (sport, event_id, observed_at_utc);
CREATE TABLE IF NOT EXISTS bets.sharp_multi_decisions (
  decision_id BIGSERIAL PRIMARY KEY, decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  sport TEXT NOT NULL, event_id TEXT NOT NULL, commence_time_utc TIMESTAMPTZ NOT NULL,
  home_team TEXT, away_team TEXT, market TEXT NOT NULL, player TEXT, player_norm TEXT NOT NULL DEFAULT '',
  side TEXT NOT NULL, line NUMERIC NOT NULL, book TEXT NOT NULL, price INTEGER NOT NULL, link TEXT,
  sharp_price_over INTEGER, sharp_price_under INTEGER, fair_probability NUMERIC NOT NULL, ev NUMERIC NOT NULL,
  minutes_to_start NUMERIC NOT NULL, tier TEXT NOT NULL, research_only BOOLEAN NOT NULL DEFAULT TRUE,
  UNIQUE (sport, event_id, market, player_norm, side, line, book, price));
"""


# --- pricing ---------------------------------------------------------------------------------------
def _fresh(timestamps: list[str | None], observed: datetime) -> datetime | None:
    stamps = [safe_time(t) for t in timestamps if t]
    if not stamps or any(t is None or t > observed for t in stamps):
        return None
    newest = max(stamps)
    return newest if (observed - newest).total_seconds() <= MAX_SOURCE_AGE else None


def pairs(book: dict, observed: datetime) -> dict[tuple, dict]:
    """(market, player_norm or 'game', line) -> {'a': price, 'b': price, labels, links, updated}.

    Side 'a' is over / home team; 'b' is under / away team. Spreads key on the home team's line.
    """
    out: dict[tuple, dict] = {}
    home, away = book.get("_home"), book.get("_away")
    for market in book.get("markets", []):
        key = market.get("key")
        updated = _fresh([book.get("last_update"), market.get("last_update")], observed)
        if updated is None:
            continue
        for o in market.get("outcomes", []):
            name, price, point = str(o.get("name") or ""), o.get("price"), o.get("point")
            if key == "h2h":
                side = "a" if name == home else "b" if name == away else None
                k = ("h2h", "game", 0.0)
            elif key == "spreads":
                if point is None:
                    continue
                side = "a" if name == home else "b" if name == away else None
                k = ("spread", "game", float(point) if side == "a" else -float(point))
            elif key == "totals":
                if point is None:
                    continue
                side = "a" if name.lower() == "over" else "b" if name.lower() == "under" else None
                k = ("total", "game", float(point))
            elif key in PROP_STAT:
                if point is None or not o.get("description"):
                    continue
                side = "a" if name.lower() == "over" else "b" if name.lower() == "under" else None
                k = (PROP_STAT[key], normalize_name(o["description"]), float(point))
            else:
                continue
            if side is None:
                continue
            rec = out.setdefault(k, dict(updated=updated, player=o.get("description")))
            rec[side], rec[side + "_link"] = price, o.get("link")
    return {k: v for k, v in out.items() if sharp_math.implied(v.get("a")) and sharp_math.implied(v.get("b"))}


SIDE_LABEL = {"h2h": ("home", "away"), "spread": ("home", "away"), "total": ("over", "under")}


def find_gaps(event: dict, observed: datetime, min_ev: float = LOG_MIN_EV) -> list[dict]:
    """Every bettable side at least min_ev above Pinnacle's no-vig price at the exact same line."""
    books = {}
    for book in event.get("bookmakers", []):
        book = dict(book, _home=event.get("home_team"), _away=event.get("away_team"))
        books[book["key"]] = pairs(book, observed)
    sharp = books.get(SHARP_BOOK) or {}
    gaps = []
    for bet_book in BET_BOOKS:
        for k, offer in (books.get(bet_book) or {}).items():
            ref = sharp.get(k)
            if ref is None or abs((offer["updated"] - ref["updated"]).total_seconds()) > MAX_SOURCE_SKEW:
                continue
            fair_a = sharp_math.no_vig_over(ref["a"], ref["b"])
            if fair_a is None:
                continue
            market, who, line = k
            labels = SIDE_LABEL.get(market, ("over", "under"))
            for side, fair in (("a", fair_a), ("b", 1.0 - fair_a)):
                value = sharp_math.ev(fair, offer[side])
                if value is None or value < min_ev:
                    continue
                gaps.append(dict(market=market, player=offer.get("player") if who != "game" else None,
                                 player_norm="" if who == "game" else who,
                                 side=labels[0] if side == "a" else labels[1],
                                 line=(line if side == "a" or market != "spread" else -line),
                                 book=bet_book, price=int(offer[side]), link=offer.get(side + "_link"),
                                 sharp_price_over=int(ref["a"]), sharp_price_under=int(ref["b"]),
                                 fair_probability=float(fair), ev=float(value)))
    return sorted(gaps, key=lambda g: -g["ev"])


def tier(ev: float, minutes: float) -> str:
    if ev < MIN_EV:
        return "logged"
    closes_at, opens_at = DECISION_WINDOW
    return "early" if minutes > opens_at else "late" if minutes < closes_at else "decision"


# --- credits ---------------------------------------------------------------------------------------
def reset_date(today: date, reset_day: int) -> date:
    year, month = today.year, today.month
    for _ in range(2):
        day = min(reset_day, calendar.monthrange(year, month)[1])
        if date(year, month, day) > today:
            return date(year, month, day)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return date(year, month, min(reset_day, calendar.monthrange(year, month)[1]))


def nfl_reserve(today: date, reset: date) -> int:
    """Credits to hold back for NFL slates before the plan resets (Sundays plus Thursday/Monday)."""
    reserve, day = 0, today
    while day < reset:
        reserve += NFL_SUNDAY_RESERVE if day.weekday() == 6 else NFL_PRIMETIME_RESERVE if day.weekday() in (0, 3) else 0
        day += timedelta(days=1)
    return reserve


def daily_allowance(remaining: int, spent_today: int, today: date, reset_day: int) -> float:
    reset = reset_date(today, reset_day)
    days = max(1, (reset - today).days)
    spare = remaining + spent_today - CREDIT_FLOOR - nfl_reserve(today, reset)
    return max(0.0, spare / days)


# --- run -------------------------------------------------------------------------------------------
def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def plan(events: dict[str, list[dict]], state: dict, now: datetime) -> list[tuple]:
    """Polls due now: ('lines', sport, cost) and ('props', sport, event, cost), soonest first."""
    due = []
    for sport, evs in events.items():
        cfg = SPORTS[sport]
        upcoming = [(e, (safe_time(e["commence_time"]) - now).total_seconds() / 60) for e in evs]
        if cfg["lines"] and any(0 < m <= LINES_HORIZON for _, m in upcoming):
            due.append((min(m for _, m in upcoming if m > 0), ("lines", sport, len(LINE_MARKETS))))
        for event, minutes in upcoming:
            if not cfg["props"] or not 0 < minutes <= PROPS_HORIZON:
                continue
            last = (state.get("props_last_poll") or {}).get(event["id"])
            if last and (now - datetime.fromisoformat(last)).total_seconds() < PROPS_SPACING * 60:
                continue
            due.append((minutes, ("props", sport, event, len(cfg["props"]))))
    return [item for _, item in sorted(due, key=lambda d: (d[1][0] != "lines", d[0]))]


def store(conn, sport: str, event: dict, observed: datetime) -> list[dict]:
    """Persist the event's quotes and every gap; returns decisions and logged gaps."""
    kickoff = safe_time(event["commence_time"])
    minutes = (kickoff - observed).total_seconds() / 60
    gaps = find_gaps(event, observed)
    with conn.cursor() as cur:
        body = dict(sport=sport, event_id=event["id"], observed_at_utc=observed.isoformat(), event=event)
        cur.execute("""INSERT INTO bets.sharp_multi_quotes (observation_id, sport, event_id, observed_at_utc, commence_time_utc, payload)
                       VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (_digest(body), sport, event["id"], observed, kickoff, psycopg2.extras.Json(event)))
        for gap in gaps:
            cur.execute("""INSERT INTO bets.sharp_multi_decisions (decided_at, sport, event_id, commence_time_utc,
                               home_team, away_team, market, player, player_norm, side, line, book, price, link,
                               sharp_price_over, sharp_price_under, fair_probability, ev, minutes_to_start, tier)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (observed, sport, event["id"], kickoff, event.get("home_team"), event.get("away_team"),
                         gap["market"], gap["player"], gap["player_norm"], gap["side"], gap["line"], gap["book"],
                         gap["price"], gap["link"], gap["sharp_price_over"], gap["sharp_price_under"],
                         gap["fair_probability"], gap["ev"], minutes, tier(gap["ev"], minutes)))
    return gaps


def run(*, now: datetime | None = None, session=requests, dry_run: bool = False) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    key = OddsCrawlerConfig().oddsapi_key
    if not key:
        return dict(status="skipped", reason="ODDS_API_KEY is not set")
    state = _load_state()
    events, remaining = {}, None
    for sport, cfg in SPORTS.items():  # the events endpoint is free
        response = session.get(f"{BASE}/{cfg['key']}/events", params={"apiKey": key}, timeout=30)
        response.raise_for_status()
        remaining = int(response.headers.get("x-requests-remaining") or remaining or 0)
        events[sport] = [e for e in response.json() if 0 < (safe_time(e["commence_time"]) - now).total_seconds() <= 3 * 3600]
    due = plan(events, state, now)
    today = str(now.astimezone(_ET).date())
    spent = state.get("spent", {}).get(today, 0)
    allowance = daily_allowance(remaining or 0, spent, now.astimezone(_ET).date(),
                                int(os.getenv("NFL_ODDS_API_RESET_DAY") or 1))
    result = dict(status="ok", credits_remaining=remaining, allowance=round(allowance, 1), spent_today=spent,
                  due=[(d[0], d[1], d[2]["id"] if d[0] == "props" else None) for d in due], polled=[], skipped=[])
    if dry_run or not due:
        return dict(result, status="dry_run" if dry_run else "nothing_due")
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        for item in due:
            cost = item[-1]
            if spent + cost > allowance or (remaining or 0) - cost < CREDIT_FLOOR:
                result["skipped"].append(dict(kind=item[0], sport=item[1], reason="credit_allowance"))
                continue
            sport = item[1]
            params = {"apiKey": key, "bookmakers": ",".join(BOOKMAKERS), "oddsFormat": "american",
                      "dateFormat": "iso", "includeLinks": "true"}
            if item[0] == "lines":
                url = f"{BASE}/{SPORTS[sport]['key']}/odds"
                params["markets"] = ",".join(LINE_MARKETS)
            else:
                url = f"{BASE}/{SPORTS[sport]['key']}/events/{item[2]['id']}/odds"
                params["markets"] = ",".join(SPORTS[sport]["props"])
            response = session.get(url, params=params, timeout=30)
            if not response.ok:
                result["skipped"].append(dict(kind=item[0], sport=sport, reason=f"http_{response.status_code}"))
                continue
            observed = datetime.now(timezone.utc) if session is requests else now
            used = int(response.headers.get("x-requests-last") or cost)
            spent += used
            remaining = int(response.headers.get("x-requests-remaining") or (remaining or 0) - used)
            payload = response.json()
            relevant = payload if item[0] == "lines" else [payload]
            gaps = 0
            for event in relevant:
                minutes = (safe_time(event.get("commence_time")) - observed).total_seconds() / 60
                if item[0] == "lines" and not 0 < minutes <= LINES_HORIZON:
                    continue  # the bulk call returns every game; keep only those near their start
                gaps += len(store(conn, sport, event, observed))
            conn.commit()
            if item[0] == "props":
                state.setdefault("props_last_poll", {})[item[2]["id"]] = now.isoformat()
            result["polled"].append(dict(kind=item[0], sport=sport, credits=used, gaps=gaps,
                                         event=item[2]["id"] if item[0] == "props" else None))
    state.setdefault("spent", {})[today] = spent
    state.update(credits_remaining=remaining, updated_at=now.isoformat())
    atomic_json(STATE, state)
    return dict(result, credits_remaining=remaining, spent_today=spent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
