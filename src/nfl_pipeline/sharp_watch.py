"""Poll FanDuel and sharp books together and alert when FanDuel is off-market.

A stale FanDuel price is usually corrected within minutes, so one lock-time snapshot catches little.
This polls The Odds API for FanDuel + Pinnacle/exchanges in the same call (named bookmakers: <=10
books cost one region, so a poll is 1 credit per market per game), prices FanDuel at the sharp
no-vig line (moved to FanDuel's line when they differ slightly), and posts a Discord alert with the
FanDuel link and the worst price still worth taking. Every alert is stored once per distinct price in
bets.nfl_sharp_alerts (rows are never deleted) and graded by CLV in sharp_alert_report. Alerts are research, not
bets, until that report shows the edge survives to the close.

It runs on the existing 10-minute close task and decides itself whether a poll is worth the credits:
hourly from 24h before kickoff, every 10 minutes in the last 4 hours, within a daily allowance spread
over the rest of the month (from the live remaining-credit count) and never below a credit floor
(NFL_SHARP_CREDIT_FLOOR). Add markets with NFL_SHARP_WATCH_MARKETS once the plan has credits for them.
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
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
from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig, _full_url, _save_payload
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import STAT_BY_MARKET, normalize_name, normalize_team

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "nfl_sharp_watch_state.json"
PROVIDER = "oddsapi_watch"
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
_ET = ZoneInfo("America/New_York")
SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "matchbook", "smarkets")
BOOKMAKERS = ("fanduel",) + SHARP_BOOKS  # <= 10 named books = one region per market
GAME_MARKETS = {"spreads": "spread", "totals": "total"}
DEFAULT_MARKETS = "player_reception_yds"
MIN_EV = 0.03            # alert threshold vs the sharp fair price
MIN_PRICE_EV = 0.01      # "take it at or better than" keeps at least this much EV
MAX_ALERTS_PER_RUN = 10


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def markets() -> list[str]:
    return [m.strip() for m in (os.getenv("NFL_SHARP_WATCH_MARKETS") or DEFAULT_MARKETS).split(",") if m.strip()]


def interval_minutes(minutes_to_kickoff: float) -> int | None:
    if minutes_to_kickoff <= 0:
        return None
    if minutes_to_kickoff <= 240:
        return 10
    if minutes_to_kickoff <= 1440:
        return 60
    return None


def daily_allowance(remaining: int, today: date, floor: int) -> float:
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    return max(0.0, remaining - floor) / max(1, days_left)


def plan_polls(games: list[dict], state: dict, now: datetime, remaining: int, cost: int,
               floor: int) -> tuple[list[dict], dict[str, str]]:
    today = now.astimezone(_ET).date()
    spent = state.get("spent", {}).get(str(today), 0)
    allowance = daily_allowance(remaining + spent, today, floor)  # allowance set from start-of-day credits
    due, skipped = [], {}
    for game in sorted(games, key=lambda g: g["start"]):
        minutes = (game["start"] - now).total_seconds() / 60
        every = interval_minutes(minutes)
        last = state.get("last_poll", {}).get(game["game_id"])
        if every is None:
            continue
        if last and (now - datetime.fromisoformat(last)).total_seconds() < every * 60 - 30:
            continue
        if remaining - cost < floor:
            skipped[game["game_id"]] = "credit_floor"
            continue
        if spent + cost > allowance:
            skipped[game["game_id"]] = "daily_allowance"
            continue
        due.append(game)
        spent += cost
        remaining -= cost
    return due, skipped


def _pairs(book: dict) -> dict[tuple, dict]:
    """(stat, player_norm or side-group, line) -> {over/home: price, under/away: price, link...} for one book."""
    out: dict[tuple, dict] = {}
    for market in book.get("markets", []):
        key = market.get("key")
        if key in GAME_MARKETS:
            stat = GAME_MARKETS[key]
            for o in market.get("outcomes", []):
                name = str(o.get("name") or "")
                side = "over" if name.lower() == "over" else "under" if name.lower() == "under" else normalize_team(name)
                line = o.get("point")
                if line is None:
                    continue
                if stat == "spread":  # key the pair by the home line; store each team's side separately
                    out.setdefault((stat, side, float(line)), {})["price"] = o.get("price")
                    out[(stat, side, float(line))]["link"] = o.get("link")
                else:
                    rec = out.setdefault((stat, "game", float(line)), {})
                    rec[side] = o.get("price"); rec[f"{side}_link"] = o.get("link")
            continue
        stat = STAT_BY_MARKET.get(key)
        if not stat:
            continue
        for o in market.get("outcomes", []):
            side = str(o.get("name") or "").lower()
            player = o.get("description")
            if side not in ("over", "under") or not player or o.get("point") is None:
                continue
            rec = out.setdefault((stat, normalize_name(player), float(o["point"])), {"player": player})
            rec[side] = o.get("price"); rec[f"{side}_link"] = o.get("link")
    return out


def find_edges(payload: dict, *, min_ev: float = MIN_EV) -> list[dict]:
    books = {b["key"]: _pairs(b) for b in payload.get("bookmakers", [])}
    fanduel = books.get("fanduel") or {}
    sharp_books = [b for b in SHARP_BOOKS if b in books]
    edges = []
    for (stat, who, line), fd in fanduel.items():
        if stat == "spread":
            continue  # spreads need both teams' sides; handled as exact-line two-way below
        if fd.get("over") is None or fd.get("under") is None:
            continue
        best = None
        for book in sharp_books:  # preferred book first, then closest line
            for (s_stat, s_who, s_line), q in books[book].items():
                if s_stat != stat or s_who != who or q.get("over") is None or q.get("under") is None:
                    continue
                fair_over = sharp_math.over_probability_at(stat, s_line, sharp_math.no_vig_over(q["over"], q["under"]), line)
                if fair_over is None:
                    continue
                cand = (abs(s_line - line), dict(book=book, line=s_line, fair_over=fair_over))
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is not None and best[0] == 0:
                break
        if best is None:
            continue
        sharp = best[1]
        for side in ("over", "under"):
            fair = sharp["fair_over"] if side == "over" else 1.0 - sharp["fair_over"]
            value = sharp_math.ev(fair, fd[side])
            if value is not None and value >= min_ev:
                edges.append(dict(stat=stat, player=fd.get("player"), player_norm=who if who != "game" else None,
                                  side=side, line=line, price=int(fd[side]), link=fd.get(f"{side}_link"),
                                  sharp_book=sharp["book"], sharp_line=sharp["line"], fair_probability=fair,
                                  ev=value, minimum_price=sharp_math.minimum_price(fair, MIN_PRICE_EV),
                                  line_gap=line - sharp["line"]))
    return sorted(edges, key=lambda e: -e["ev"])


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bets.nfl_sharp_alerts (
  alert_id BIGSERIAL PRIMARY KEY,
  alerted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  event_id TEXT NOT NULL, game_id TEXT, commence_time_utc TIMESTAMPTZ, home_team TEXT, away_team TEXT,
  stat TEXT NOT NULL, player_name TEXT, player_name_norm TEXT, side TEXT NOT NULL,
  fd_line NUMERIC NOT NULL, fd_price INTEGER NOT NULL, fd_link TEXT,
  sharp_book TEXT NOT NULL, sharp_line NUMERIC NOT NULL, fair_probability NUMERIC NOT NULL,
  ev NUMERIC NOT NULL, minimum_price INTEGER, line_gap NUMERIC, posted BOOLEAN NOT NULL DEFAULT FALSE,
  UNIQUE (event_id, stat, player_name_norm, fd_line, side, fd_price)
);
"""


def _fallback_link(cur, edge: dict, game: dict) -> str | None:
    """FanDuel deep link from the regular quote feed for the same player/stat/line/side, if any."""
    if edge["player_norm"] is None:
        return None
    cur.execute(f"""SELECT {edge['side']}_link FROM odds.nfl_player_prop_lines
        WHERE bookmaker_key='fanduel' AND player_name_norm=%s AND stat=%s AND line=%s AND {edge['side']}_link IS NOT NULL
          AND commence_time_utc BETWEEN %s AND %s ORDER BY fetched_at_utc DESC LIMIT 1""",
                (edge["player_norm"], edge["stat"], edge["line"], game["start"] - timedelta(minutes=30), game["start"] + timedelta(minutes=30)))
    row = cur.fetchone()
    return row[0] if row else None


def format_alert(edge: dict, game: dict) -> str:
    from nfl_pipeline.fanduel_links import single_betslip_url, provider_link
    who = edge["player"] or f"{game['away']} @ {game['home']}"
    label = edge["stat"].replace("_", " ").title()
    price = f"{edge['price']:+d}"
    gap = f" ({edge['sharp_book'].title()} {edge['sharp_line']:g})" if edge["line_gap"] else f" ({edge['sharp_book'].title()})"
    link = single_betslip_url(edge["link"]) or provider_link(edge["link"])
    kickoff = int(game["start"].timestamp())
    return (f"**SHARP EDGE - research, not a bet**\n{who} {edge['side'].upper()} {edge['line']:g} {label} "
            f"({game['away']} @ {game['home']}, <t:{kickoff}:R>)\nFanDuel {price} | fair {edge['fair_probability']:.1%}{gap} | "
            f"EV {edge['ev']:+.1%} | take at {edge['minimum_price']:+d} or better"
            + (f"\n[Open on FanDuel](<{link}>)" if link else "\nNo FanDuel link captured; search manually."))


def _post(text: str) -> None:
    from nfl_pipeline.run_daily_and_notify import _post_payload
    asyncio.run(_post_payload({"content": text[:1950], "allowed_mentions": {"parse": []}}))


def _games(now: datetime) -> list[dict]:
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT game_id, game_date_et, home_team_abbr, away_team_abbr, start_ts_utc FROM raw.nfl_games
            WHERE start_ts_utc > %s AND start_ts_utc <= %s ORDER BY start_ts_utc""", (now, now + timedelta(hours=24)))
        return [dict(zip(("game_id", "day", "home", "away", "start"), r)) for r in cur.fetchall()]


def run(*, now: datetime | None = None, session=requests, post=_post) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    games = _games(now)
    if not games:
        return dict(status="no_games_within_24h")
    cfg = OddsCrawlerConfig()
    if not cfg.oddsapi_key:
        return dict(status="skipped", reason="ODDS_API_KEY is not set")
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        state = {}
    wanted = markets()
    cost = len(wanted)  # named bookmakers <= 10 -> one region-equivalent per market
    floor = _int_env("NFL_SHARP_CREDIT_FLOOR", 150)
    # Time-due first (no API call), then the free events call reports the true remaining credits.
    time_due, _ = plan_polls(games, state, now, 10**9, 0, 0)
    if not time_due:
        return dict(status="nothing_due", markets=wanted)
    events = session.get(f"{BASE}/events", params={"apiKey": cfg.oddsapi_key}, timeout=cfg.timeout_s)
    events.raise_for_status()
    remaining = int(events.headers.get("x-requests-remaining") or 0)
    by_matchup = {(normalize_team(e["home_team"]), normalize_team(e["away_team"])): e for e in events.json()}
    due, skipped = plan_polls(time_due, state, now, remaining, cost, floor)
    result = dict(status="ok", polled=[], skipped=skipped, alerts=0, credits_remaining=remaining, markets=wanted)
    if not due:
        return dict(result, status="budget_hold")
    today = str(now.astimezone(_ET).date())
    new_alerts = []
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute(SCHEMA_SQL)
        conn.commit()
        for game in due:
            event = by_matchup.get((normalize_team(game["home"]), normalize_team(game["away"])))
            if event is None:
                skipped[game["game_id"]] = "no_matching_event"
                continue
            params = {"apiKey": cfg.oddsapi_key, "bookmakers": ",".join(BOOKMAKERS), "markets": ",".join(wanted),
                      "oddsFormat": "american", "dateFormat": "iso", "includeLinks": "true"}
            url = f"{BASE}/events/{event['id']}/odds"
            response = session.get(url, params=params, timeout=cfg.timeout_s)
            if not response.ok:
                skipped[game["game_id"]] = f"http_{response.status_code}"
                continue
            remaining = int(response.headers.get("x-requests-remaining") or remaining - cost)
            payload = response.json()
            for endpoint, keep in (("nfl_player_props", lambda k: k not in GAME_MARKETS), ("nfl_game_odds", lambda k: k in GAME_MARKETS)):
                if any(keep(m.get("key")) for b in payload.get("bookmakers", []) for m in b.get("markets", [])):
                    _save_payload(conn, endpoint=endpoint, snapshot_role="live", as_of_date=game["day"],
                                  url=_full_url(url, params), payload=payload, provider=PROVIDER)
            state.setdefault("last_poll", {})[game["game_id"]] = now.isoformat()
            state.setdefault("spent", {})[today] = state.get("spent", {}).get(today, 0) + cost
            edges = find_edges(payload)
            result["polled"].append(dict(game_id=game["game_id"], books=sorted(b["key"] for b in payload.get("bookmakers", [])),
                                         edges=len(edges)))
            with conn.cursor() as cur:
                for edge in edges:
                    edge["link"] = edge["link"] or _fallback_link(cur, edge, game)
                    cur.execute("""INSERT INTO bets.nfl_sharp_alerts (event_id, game_id, commence_time_utc, home_team, away_team,
                            stat, player_name, player_name_norm, side, fd_line, fd_price, fd_link, sharp_book, sharp_line,
                            fair_probability, ev, minimum_price, line_gap)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING RETURNING alert_id""",
                                (event["id"], game["game_id"], game["start"], game["home"], game["away"], edge["stat"],
                                 edge["player"], edge["player_norm"], edge["side"], edge["line"], edge["price"], edge["link"],
                                 edge["sharp_book"], edge["sharp_line"], edge["fair_probability"], edge["ev"],
                                 edge["minimum_price"], edge["line_gap"]))
                    row = cur.fetchone()
                    if row:
                        new_alerts.append((row[0], edge, game))
            conn.commit()
        for alert_id, edge, game in new_alerts[:MAX_ALERTS_PER_RUN]:
            try:
                post(format_alert(edge, game))
                with conn.cursor() as cur:
                    cur.execute("UPDATE bets.nfl_sharp_alerts SET posted=TRUE WHERE alert_id=%s", (alert_id,))
                conn.commit()
            except Exception as exc:  # an unposted alert is still logged and graded
                result.setdefault("post_errors", []).append(type(exc).__name__)
    from nfl_pipeline.parse_oddsapi import ParseConfig, parse_props
    for day in sorted({g["day"] for g in due}):
        parse_props(ParseConfig(as_of_date=day))
    state.update(credits_remaining=remaining, updated_at=now.isoformat())
    atomic_json(STATE, state)
    return dict(result, alerts=len(new_alerts), credits_remaining=remaining)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
