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
SPREAD_MARKETS = {"spreads": "spread", "alternate_spreads": "spread", "spreads_h1": "spread_h1"}
TOTAL_MARKETS = {"totals": "total", "alternate_totals": "total", "totals_h1": "total_h1"}
PROP_STAT = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists",
             "player_threes": "threes"}

MIN_EV = 0.03
LOG_MIN_EV = 0.01
DECISION_WINDOW = (15, 45)        # minutes before the start, as for NFL
LINES_HORIZON = 60                 # poll game lines when any game starts within this many minutes...
LINES_SPACING = 14.5               # ...at most every other task run (~20 min): one bulk call covers every game
# Typical spread of the final margin / total around the market, and the largest gap a fair price is moved
# across (normal approximation, as the NFL scanner does for yardage). Books often hang -6.5 where Pinnacle
# has -7; without this 60-80% of spreads and totals had nothing to compare against. Football spreads are
# never shifted: margins cluster on 3 and 7, a smooth curve undervalues crossing them, and the only +3%
# college "edges" a shift produced (Oct 24 sample: -3.5 from -2.5, +6.5 from +7.5) were exactly that.
LINE_SHIFT = {"nba": {"spread": (12.5, 1.5), "total": (18.0, 1.5)},
              "ncaaf": {"total": (16.5, 1.5)}}
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
ALTER TABLE bets.sharp_multi_decisions ADD COLUMN IF NOT EXISTS method TEXT NOT NULL DEFAULT 'exact';
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
            elif key in SPREAD_MARKETS:
                if point is None:
                    continue
                side = "a" if name == home else "b" if name == away else None
                k = (SPREAD_MARKETS[key], "game", float(point) if side == "a" else -float(point))
            elif key in TOTAL_MARKETS:
                if point is None:
                    continue
                side = "a" if name.lower() == "over" else "b" if name.lower() == "under" else None
                k = (TOTAL_MARKETS[key], "game", float(point))
            elif key == "team_totals":
                if point is None or not o.get("description"):
                    continue
                side = "a" if name.lower() == "over" else "b" if name.lower() == "under" else None
                k = ("team_total", normalize_name(o["description"]), float(point))
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


SIDE_LABEL = {"h2h": ("home", "away"), "spread": ("home", "away"), "spread_h1": ("home", "away"),
              "total": ("over", "under"), "total_h1": ("over", "under"), "team_total": ("over", "under")}
TEAM_MARKETS = ("spread", "spread_h1")  # keyed on the home line; the away side's line is its negative


def shifted_probability(market: str, sharp_line: float, p_a: float, target_line: float, sigma: float) -> float:
    """P(side a) at target_line from P(side a) at sharp_line. Side a is the home team covering (margin above
    minus its line) or the over (total above the line)."""
    from scipy.stats import norm
    threshold = (lambda line: -line) if market == "spread" else (lambda line: line)
    mean = threshold(sharp_line) + sigma * norm.ppf(p_a)
    return float(1.0 - norm.cdf(threshold(target_line), loc=mean, scale=sigma))


def reference(sharp: dict, key: tuple, sport: str | None) -> tuple[dict, float, str] | None:
    """(Pinnacle quote, fair P(side a) at key's line, 'exact' | 'shifted'), or None."""
    ref = sharp.get(key)
    if ref is not None:
        fair = sharp_math.no_vig_over(ref["a"], ref["b"])
        return (ref, fair, "exact") if fair is not None else None
    market, who, line = key
    sigma, max_gap = (LINE_SHIFT.get(sport or "") or {}).get(market, (None, 0.0))
    if sigma is None:
        return None
    near = [(abs(k[2] - line), k, q) for k, q in sharp.items()
            if k[0] == market and k[1] == who and 0 < abs(k[2] - line) <= max_gap + 1e-9]
    if not near:
        return None
    _, k, ref = min(near, key=lambda n: n[0])
    fair = sharp_math.no_vig_over(ref["a"], ref["b"])
    return None if fair is None else (ref, shifted_probability(market, k[2], fair, line, sigma), "shifted")


def find_gaps(event: dict, observed: datetime, min_ev: float = LOG_MIN_EV, sport: str | None = None) -> list[dict]:
    """Every bettable side at least min_ev above Pinnacle's no-vig price: at the exact same line, or for
    spreads and totals a nearby Pinnacle line moved to the book's (LINE_SHIFT)."""
    books = {}
    for book in event.get("bookmakers", []):
        book = dict(book, _home=event.get("home_team"), _away=event.get("away_team"))
        books[book["key"]] = pairs(book, observed)
    sharp = books.get(SHARP_BOOK) or {}
    gaps = []
    for bet_book in BET_BOOKS:
        for k, offer in (books.get(bet_book) or {}).items():
            found = reference(sharp, k, sport)
            if found is None:
                continue
            ref, fair_a, method = found
            if abs((offer["updated"] - ref["updated"]).total_seconds()) > MAX_SOURCE_SKEW:
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
                                 line=(line if side == "a" or market not in TEAM_MARKETS else -line),
                                 book=bet_book, price=int(offer[side]), link=offer.get(side + "_link"),
                                 sharp_price_over=int(ref["a"]), sharp_price_under=int(ref["b"]),
                                 fair_probability=float(fair), ev=float(value), method=method))
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
        last_lines = (state.get("lines_last_poll") or {}).get(sport)
        spaced = not last_lines or (now - datetime.fromisoformat(last_lines)).total_seconds() >= LINES_SPACING * 60
        if cfg["lines"] and spaced and any(0 < m <= LINES_HORIZON for _, m in upcoming):
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
    gaps = find_gaps(event, observed, sport=sport)
    with conn.cursor() as cur:
        body = dict(sport=sport, event_id=event["id"], observed_at_utc=observed.isoformat(), event=event)
        cur.execute("""INSERT INTO bets.sharp_multi_quotes (observation_id, sport, event_id, observed_at_utc, commence_time_utc, payload)
                       VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (_digest(body), sport, event["id"], observed, kickoff, psycopg2.extras.Json(event)))
        for gap in gaps:
            cur.execute("""INSERT INTO bets.sharp_multi_decisions (decided_at, sport, event_id, commence_time_utc,
                               home_team, away_team, market, player, player_norm, side, line, book, price, link,
                               sharp_price_over, sharp_price_under, fair_probability, ev, minutes_to_start, tier, method)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (observed, sport, event["id"], kickoff, event.get("home_team"), event.get("away_team"),
                         gap["market"], gap["player"], gap["player_norm"], gap["side"], gap["line"], gap["book"],
                         gap["price"], gap["link"], gap["sharp_price_over"], gap["sharp_price_under"],
                         gap["fair_probability"], gap["ev"], minutes, tier(gap["ev"], minutes), gap["method"]))
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
            else:
                state.setdefault("lines_last_poll", {})[sport] = now.isoformat()
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
