"""Poll the bettable books and Pinnacle together and alert when a book is off the sharp price.

This is the NFL cash strategy (rules in sharp_strategy). A stale book price is usually corrected
within minutes, so one lock-time snapshot catches little. This polls The Odds API for FanDuel,
ESPN Bet and DraftKings plus Pinnacle in the same call (named bookmakers: <=10 books cost one
region, so a poll is 1 credit per market per game), prices each book at Pinnacle's no-vig line
(moved to the book's line when they differ slightly), and posts a Discord alert with the book link
and the worst price still worth taking. Gaps seen 15-45 minutes before kickoff carry a $1 stake,
within the strategy's daily, weekly, per-game and loss limits. Every alert is stored once per
distinct price in bets.nfl_sharp_alerts (rows are never deleted) and graded at the sharp close in
sharp_alert_report, which also decides whether staking may continue.

It runs every 5 minutes and polls each game every 5 minutes through its final hour, which is both
the stake window and the closing snapshot every decision is graded against. The daily allowance is
today's share, by games, of the spare credits left before the plan resets (NFL_ODDS_API_RESET_DAY;
live remaining-credit count), never below a credit floor (NFL_SHARP_CREDIT_FLOOR). Markets come from
NFL_SHARP_WATCH_MARKETS and are part of the registration, so changing them needs a new one.

A scan that refuses to run (unregistered or drifted registration) while games are close posts one
Discord warning per day and reason: an outage here silently loses every decision and every close.
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import hashlib
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline import sharp_math
from nfl_pipeline import sharp_strategy as strategy
from nfl_pipeline.crawler_oddsapi import OddsCrawlerConfig, _full_url, _save_payload
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.markets import STAT_BY_MARKET, normalize_name, normalize_team
from nfl_pipeline.context_contract import safe_time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "reports" / "nfl_sharp_watch_state.json"
PROVIDER = "oddsapi_watch"
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
_ET = ZoneInfo("America/New_York")
SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "matchbook", "smarkets")
BET_BOOKS = tuple(strategy.BET_BOOKS)  # books we can bet at, priced against SHARP_BOOKS
# Books we cannot bet at today. Naming them is free (<= 10 books cost one region per market) and
# their prices tell us whether opening an account somewhere else would be worth it. Caesars
# (williamhill_us) returned no player props when scanned in October, so it is not named.
REFERENCE_BOOKS = ("betmgm", "fanatics")
BOOKMAKERS = tuple(dict.fromkeys(BET_BOOKS + SHARP_BOOKS + REFERENCE_BOOKS))[:10]
BOOK_LABEL = {"fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM",
              "williamhill_us": "Caesars", "espnbet": "ESPN BET", "fanatics": "Fanatics"}
BOOK_DOMAIN = {"fanduel": "fanduel.com", "draftkings": "draftkings.com", "betmgm": "betmgm.com",
               "williamhill_us": "caesars.com", "espnbet": "espnbet.com", "fanatics": "fanatics.com"}
GAME_MARKETS = {"spreads": "spread", "totals": "total", "alternate_totals": "total",
                "alternate_spreads": "spread"}
# Markets the scanner can price but the modeling pipeline has no stat spec for. Kept local so
# STAT_SPECS, which drives training and scoring, is untouched.
EXTRA_STAT_BY_MARKET = {"player_pass_completions": "pass_completions",
                        "player_pass_attempts": "pass_attempts",
                        "player_rush_attempts": "rush_attempts",
                        "player_anytime_td": "anytime_td"}
# Yes/No markets priced as "over/under 0.5". Worth pricing only with a de-vig that handles longshots:
# proportional de-vig overstates a +650 yes by ~3 points, which is larger than any edge we would bet.
YES_NO_SIDES = {"yes": "over", "no": "under"}
DEFAULT_MARKETS = "player_reception_yds"
MIN_EV = strategy.MIN_EV  # Discord alert threshold vs the sharp fair price
LOG_MIN_EV = 0.01        # smaller gaps are logged (tier 'logged', no ping) so CLV evidence accrues faster
# Pings and stakes only where a gap still holds its edge at the sharp close. 2025 backtests, >= +3%
# gaps, EV measured at Pinnacle's close:
#   receiving  T-720 -0.05% | T-360 +1.74% | T-180 +2.70% | T-90 +2.70% | T-15 +3.32%
#   receptions                              | T-90  -0.02% | T-45 +1.58% | T-15 +3.49%
#   rushing                                 | T-90  +3.75% | T-45 +3.48% | T-15 +4.77%
# One window for every staked market: 45 minutes is inside the range where all three hold their
# edge, and 15 minutes leaves time to place the bet by hand.
STAKE_WINDOW_MINUTES = tuple(strategy.STAKE_WINDOW_MINUTES)
# Fetch a little before the window opens so a forming gap is seen, and through kickoff so every
# decision has a closing snapshot inside MAX_CLOSE_AGE (sharp_alert_report).
POLL_HORIZON_MINUTES = STAKE_WINDOW_MINUTES[1] + 15
MIN_PRICE_EV = 0.01      # "take it at or better than" keeps at least this much EV
MAX_ALERTS_PER_RUN = 10
FINAL_WINDOW_MINUTES = 90  # inactives land ~T-90 and books are slowest to react; never starve this window
QUOTE_CONTRACT = "nfl-sharp-quotes-v2"
MAX_SOURCE_AGE_SECONDS = 300
MAX_SOURCE_SKEW_SECONDS = 120
EXPERIMENT_NAME = strategy.EXPERIMENT_NAME
READINESS_REPORT = ROOT / "reports" / "nfl_sharp_alerts_latest.json"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def markets() -> list[str]:
    return [m.strip() for m in (os.getenv("NFL_SHARP_WATCH_MARKETS") or DEFAULT_MARKETS).split(",") if m.strip()]


def experiment_config():
    """Capture both code constants and environment-selected markets in the frozen manifest."""
    return dict(contract=QUOTE_CONTRACT, bet_books=list(BET_BOOKS), reference_books=["pinnacle"],
                markets=markets(), min_ev=MIN_EV, log_min_ev=LOG_MIN_EV,
                max_source_age_seconds=MAX_SOURCE_AGE_SECONDS, max_source_skew_seconds=MAX_SOURCE_SKEW_SECONDS,
                stake_window_minutes=list(STAKE_WINDOW_MINUTES), poll_horizon_minutes=POLL_HORIZON_MINUTES,
                devig_method=sharp_math.DEVIG_METHOD, line_model=sharp_math.LINE_MODEL,
                cash_enabled=bool(strategy.BETS_ENABLED), staked_stats=sorted(strategy.STAKED_STATS),
                stake=strategy.STAKE, max_bets_per_day=strategy.MAX_BETS_PER_DAY,
                max_stake_per_week=strategy.MAX_STAKE_PER_WEEK, max_bets_per_game=strategy.MAX_BETS_PER_GAME,
                loss_pause=strategy.LOSS_PAUSE, gates=strategy.GATES, shadow_filter=strategy.SHADOW_FILTER,
                selection="best_ev_first_eligible_per_player_game_stat_across_books")


def interval_minutes(minutes_to_kickoff: float) -> int | None:
    """Poll cadence by time to kickoff, or None when a poll would buy nothing actionable.

    Only the final hour is polled: it holds the whole stake window (T-45 to T-15) and the closing
    snapshot every decision is graded against. Earlier gaps are never staked, and the 2025 decay
    curve shows they are worth less by the close, so polling them only spent credits.
    """
    if minutes_to_kickoff <= 0 or minutes_to_kickoff > POLL_HORIZON_MINUTES:
        return None
    return 5


def daily_allowance(remaining: int, today: date, floor: int) -> float:
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    return max(0.0, remaining - floor) / max(1, days_left)


def cycle_end(today: date, reset_day: int) -> date:
    """First credit-reset date after today (reset_day clamped to the month's length)."""
    year, month = today.year, today.month
    for _ in range(2):
        day = min(reset_day, calendar.monthrange(year, month)[1])
        if date(year, month, day) > today:
            return date(year, month, day)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return date(year, month, min(reset_day, calendar.monthrange(year, month)[1]))


def game_share(today: date, games_by_day: dict[date, int], reset_day: int) -> float | None:
    """Today's share of the cycle's spare credits, by games (NFL demand is lumpy: Sunday >> Tuesday)."""
    end = cycle_end(today, reset_day)
    total = sum(n for d, n in games_by_day.items() if today <= d < end)
    return games_by_day.get(today, 0) / total if total else None


def _final_window_polls(minutes_to_kickoff: float) -> int:
    """Polls a game still needs inside its final window, at the cadence it will actually use."""
    count, remaining = 0, min(float(minutes_to_kickoff), FINAL_WINDOW_MINUTES)
    while remaining > 0:
        every = interval_minutes(remaining)
        if every is None:
            break
        count += 1
        remaining -= every
    return count


def plan_polls(games: list[dict], state: dict, now: datetime, remaining: int, cost: int,
               floor: int, share: float | None = None) -> tuple[list[dict], dict[str, str]]:
    today = now.astimezone(_ET).date()
    spent = state.get("spent", {}).get(str(today), 0)
    # Allowance from start-of-day credits: today's share of the cycle by games, else an even share by day.
    allowance = (max(0.0, remaining + spent - floor) * share if share is not None
                 else daily_allowance(remaining + spent, today, floor))
    # Credits today's games will still need in their final windows. Earlier (hourly or T-4h..T-90)
    # polls only spend what is left over, so a tight budget is saved for the window that matters.
    reserve = cost * sum(_final_window_polls((g["start"] - now).total_seconds() / 60) for g in games
                         if g["start"].astimezone(_ET).date() == today)
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
        # Final-window polls are bounded only by the floor: overspend lowers later days' allowance
        # (recomputed from the live balance), and early polls already yield to the reserve.
        if minutes > FINAL_WINDOW_MINUTES:
            if spent + cost > allowance:
                skipped[game["game_id"]] = "daily_allowance"
                continue
            if spent + cost + reserve > allowance:
                skipped[game["game_id"]] = "final_window_reserve"
                continue
        due.append(game)
        spent += cost
        remaining -= cost
    return due, skipped


def _pairs(book: dict) -> dict[tuple, dict]:
    """(stat, player_norm or side-group, line) -> {over/home: price, under/away: price, link...} for one book."""
    out: dict[tuple, dict] = {}
    for market in book.get("markets", []):
        provenance = dict(book_updated_at=book.get("last_update"),
                          market_updated_at=market.get("last_update"), market_key=market.get("key"))
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
                    out.setdefault((stat, side, float(line)), dict(provenance))["price"] = o.get("price")
                    out[(stat, side, float(line))]["link"] = o.get("link")
                else:
                    rec = out.setdefault((stat, "game", float(line)), dict(provenance))
                    rec[side] = o.get("price"); rec[f"{side}_link"] = o.get("link")
            continue
        stat = STAT_BY_MARKET.get(key) or EXTRA_STAT_BY_MARKET.get(key)
        if not stat:
            continue
        for o in market.get("outcomes", []):
            side = str(o.get("name") or "").lower()
            player = o.get("description")
            point = o.get("point")
            if side in YES_NO_SIDES and point is None:
                side, point = YES_NO_SIDES[side], 0.5
            if side not in ("over", "under") or not player or point is None:
                continue
            rec = out.setdefault((stat, normalize_name(player), float(point)), dict(provenance, player=player))
            rec[side] = o.get("price"); rec[f"{side}_link"] = o.get("link")
    return out


def source_time(quote: dict):
    """Prefer the market timestamp; a book timestamp is only a documented fallback."""
    return safe_time(quote.get("market_updated_at") or quote.get("book_updated_at"))


def quote_error(quote: dict, observed_at: datetime) -> str | None:
    observed_at = safe_time(observed_at)
    updated = source_time(quote)
    if updated is None or observed_at is None:
        return "source_timestamp_missing"
    # A future book timestamp is invalid even when the market timestamp looks old.
    stamps = [safe_time(quote.get(k)) for k in ("book_updated_at", "market_updated_at") if quote.get(k)]
    if any(t is None or t > observed_at for t in stamps):
        return "source_timestamp_future_or_invalid"
    if (observed_at - updated).total_seconds() > MAX_SOURCE_AGE_SECONDS:
        return "source_quote_stale"
    if any(sharp_math.implied(quote.get(side)) is None for side in ("over", "under")):
        return "invalid_two_sided_prices"
    return None


def find_edges(payload: dict, *, min_ev: float = MIN_EV, bet_books: tuple[str, ...] = BET_BOOKS,
               now: datetime | None = None) -> list[dict]:
    """Every bettable book's side that is at least min_ev above the sharp no-vig fair price."""
    now = safe_time(now or datetime.now(timezone.utc))
    kickoff = safe_time(payload.get('commence_time'))
    if kickoff is not None and kickoff <= now:
        return []
    books = {b["key"]: _pairs(b) for b in payload.get("bookmakers", [])}
    sharp_books = [b for b in SHARP_BOOKS if b in books]
    edges = []
    for bet_book in bet_books:
        for (stat, who, line), offer in (books.get(bet_book) or {}).items():
            if stat == "spread":
                continue  # spreads need both teams' sides; handled as exact-line two-way below
            if quote_error(offer, now):
                continue
            best = None
            for book in sharp_books:  # preferred book first, then closest line
                # Exchange prices lack the commission/liquidity fields needed to establish
                # an executable fair reference. Keep fetching them for diagnostics only.
                if book != "pinnacle":
                    continue
                for (s_stat, s_who, s_line), q in books[book].items():
                    if s_stat != stat or s_who != who or quote_error(q, now):
                        continue
                    if abs((source_time(offer) - source_time(q)).total_seconds()) > MAX_SOURCE_SKEW_SECONDS:
                        continue
                    fair_over = sharp_math.over_probability_at(stat, s_line, sharp_math.no_vig_over(q["over"], q["under"]), line)
                    if fair_over is None:
                        continue
                    cand = (abs(s_line - line), dict(book=book, line=s_line, fair_over=fair_over, quote=q))
                    if best is None or cand[0] < best[0]:
                        best = cand
                if best is not None and best[0] == 0:
                    break
            if best is None:
                continue
            sharp = best[1]
            for side in ("over", "under"):
                fair = sharp["fair_over"] if side == "over" else 1.0 - sharp["fair_over"]
                value = sharp_math.ev(fair, offer[side])
                if value is not None and value >= min_ev:
                    edges.append(dict(event_id=payload.get('id'), book=bet_book, stat=stat, player=offer.get("player"),
                                      player_norm=who if who != "game" else None,
                                      side=side, line=line, price=int(offer[side]), link=offer.get(f"{side}_link"),
                                      sharp_book=sharp["book"], sharp_line=sharp["line"], fair_probability=fair,
                                      ev=value, minimum_price=sharp_math.minimum_price(fair, MIN_PRICE_EV),
                                      line_gap=line - sharp["line"],
                                      quote_provenance=dict(contract=QUOTE_CONTRACT, observed_at_utc=now.isoformat(),
                                          execution_quote=dict(offer), reference_quote=dict(sharp["quote"]),
                                          reference_method="exact" if line == sharp["line"] else "model_inferred",
                                          devig_method=sharp_math.DEVIG_METHOD)))
    return sorted(edges, key=lambda e: -e["ev"])


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def quote_observations(payload, game, observed_at):
    """Capture all pairs, including rejected ones and closes with no positive gap."""
    rows = []
    for book in payload.get("bookmakers", []):
        for (stat, who, line), quote in _pairs(book).items():
            row = dict(contract=QUOTE_CONTRACT, observed_at_utc=observed_at.isoformat(),
                       game_id=game["game_id"], event_id=payload.get("id"),
                       commence_time_utc=game["start"].isoformat(), book=book["key"], stat=stat,
                       player_name_norm=None if who == "game" else who, line=line, quote=quote,
                       validation_error=quote_error(quote, observed_at))
            rows.append(dict(row, observation_id=_digest(row)))
    return rows


def decision_observation(edge, game, tier, experiment_sha256):
    body = dict(edge, game_id=game["game_id"], commence_time_utc=game["start"].isoformat(),
                tier=tier, experiment_sha256=experiment_sha256)
    return dict(body, observation_id=_digest(body))


def persist_observations(cur, quotes):
    if quotes:
        psycopg2.extras.execute_values(cur, """INSERT INTO bets.nfl_sharp_quote_observations
            (observation_id,game_id,observed_at_utc,payload) VALUES %s ON CONFLICT DO NOTHING""",
            [(r["observation_id"], r["game_id"], r["observed_at_utc"], psycopg2.extras.Json(r)) for r in quotes])


def markets_for(minutes_to_kickoff: float, wanted: list[str]) -> list[str]:
    """Every wanted market inside the poll horizon, none outside it: a poll outside buys data we
    would never bet or grade against."""
    return list(wanted) if 0 < minutes_to_kickoff <= POLL_HORIZON_MINUTES else []


def alert_tier(ev: float, minutes_to_kickoff: float, stat: str | None = None) -> str:
    """'alert' (pinged, may be staked) only inside the stake window; 'early' before it opens and
    'late' after it closes are stored and graded but never pinged; 'logged' is below MIN_EV."""
    if ev < MIN_EV:
        return "logged"
    closes_at, opens_at = STAKE_WINDOW_MINUTES
    if minutes_to_kickoff > opens_at:
        return "early"
    if minutes_to_kickoff < closes_at:
        return "late"
    return "alert"


ALERT_REPORT = READINESS_REPORT


def _report_summary() -> dict:
    try:
        return json.loads(ALERT_REPORT.read_text(encoding="utf-8")).get("summary") or {}
    except (FileNotFoundError, ValueError, TypeError, AttributeError):
        return {}


def realized_profit() -> float:
    """Settled profit on bets actually placed, as the alert report graded them (0.0 if unknown)."""
    try:
        return float(((_report_summary().get("placed") or {}).get("realized_profit")) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def readiness() -> dict:
    """The gate verdict the alert report last computed. Missing means no evidence yet, not a pause."""
    value = _report_summary().get("readiness")
    return value if isinstance(value, dict) else {}


def risk_budget(cur, now: datetime) -> dict[str, Any]:
    """Bets already staked today, stake this NFL week, what is already staked per game and per
    player-game-stat, and whether a loss or readiness pause holds."""
    et_now = now.astimezone(_ET)
    day_start = et_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = day_start - timedelta(days=(et_now.weekday() - 1) % 7)  # NFL week starts Tuesday ET
    cur.execute("""SELECT COUNT(*) FILTER (WHERE alerted_at >= %s),
                          COALESCE(SUM(stake) FILTER (WHERE alerted_at >= %s), 0)
                   FROM bets.nfl_sharp_alerts WHERE stake > 0""", (day_start, week_start))
    today, week = cur.fetchone()
    cur.execute("""SELECT game_id, stat, COALESCE(player_name_norm, '') FROM bets.nfl_sharp_alerts
                   WHERE stake > 0 AND commence_time_utc > %s""", (now - timedelta(hours=6),))
    keys = {tuple(r) for r in cur.fetchall()}
    per_game: dict[str, int] = {}
    for game_id, _, _ in keys:
        per_game[game_id] = per_game.get(game_id, 0) + 1
    profit = realized_profit()
    gate = readiness()
    loss_paused = profit <= strategy.LOSS_PAUSE
    readiness_paused = gate.get("status") == "paused"
    return dict(today=int(today), week=float(week), staked=0.0, staked_count=0,
                staked_keys=keys, per_game=per_game, realized_profit=profit,
                loss_paused=loss_paused, readiness_paused=readiness_paused,
                readiness_reasons=list(gate.get("pause_reasons") or []),
                paused=loss_paused or readiness_paused)


def stake_for(cur, tier: str, now: datetime, budget: dict[str, Any], stat: str | None = None,
              key: tuple | None = None) -> float:
    """Flat stake for a pinged alert, or 0.0 when a cap, a pause, or an unproven market binds.

    `key` is (game_id, stat, player_name_norm): one stake per player-game-stat across all books,
    and at most MAX_BETS_PER_GAME per game. A zero-stake alert is still stored and graded; it
    just carries no money, so evidence keeps accruing.
    """
    stake = float(strategy.STAKE)
    if not strategy.BETS_ENABLED or tier != "alert" or stake <= 0 or budget["paused"]:
        return 0.0
    if stat is not None and stat not in strategy.STAKED_STATS:
        return 0.0  # scanned and graded, but not backtested yet
    if budget["today"] + budget.get("staked_count", 0) + 1 > strategy.MAX_BETS_PER_DAY:
        return 0.0
    if budget["week"] + budget["staked"] + stake > strategy.MAX_STAKE_PER_WEEK:
        return 0.0
    if key is not None:
        if key in budget.get("staked_keys", set()):
            return 0.0  # already staked at another book or an earlier price
        if budget.get("per_game", {}).get(key[0], 0) >= strategy.MAX_BETS_PER_GAME:
            return 0.0
    return stake


def commit_stake(budget: dict[str, Any], stake: float, key: tuple | None) -> None:
    """Count a stake just written so later alerts in the same run respect the same caps."""
    if stake <= 0:
        return
    budget["staked"] += stake
    budget["staked_count"] = budget.get("staked_count", 0) + 1
    if key is not None:
        budget.setdefault("staked_keys", set()).add(key)
        per_game = budget.setdefault("per_game", {})
        per_game[key[0]] = per_game.get(key[0], 0) + 1


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
-- 'alert' = EV >= MIN_EV inside the stake window (T-45..T-15), pinged when it carries a stake;
-- 'early' / 'late' = EV >= MIN_EV before / after that window (stored, no ping);
-- 'logged' = LOG_MIN_EV <= EV < MIN_EV. Rows written before the stake window existed keep their tier.
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'alert';
-- fd_line/fd_price/fd_link predate multi-book scanning; they hold the bettable book's own numbers.
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS book TEXT NOT NULL DEFAULT 'fanduel';
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS stake NUMERIC NOT NULL DEFAULT 0;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS placed_at TIMESTAMPTZ;
-- The price actually obtained, which is rarely the alerted one. Backtest EV assumes no slippage;
-- this is how we find out what that assumption costs.
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS taken_price INTEGER;
-- The model's own view of this bet. It is not what selects the bet - the price gap is - but against
-- 2025 early lines the model's side beat the close ~1% more often than not, and whether that adds to
-- a price edge is only answerable with live bets. Recorded on every alert so the question can be
-- settled from real money rather than argued.
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS model_projection NUMERIC;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS model_agrees BOOLEAN;
-- Shadow model filter (sharp_strategy.SHADOW_FILTER): 'pass', 'flag' or 'no_forecast', with reasons.
-- Graded beside EV at the sharp close; it never skips or sizes a bet until it has earned that.
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS model_filter TEXT;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS model_filter_reasons JSONB;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS initial_observation_id TEXT;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS alert_observation_id TEXT;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS initial_provenance JSONB;
ALTER TABLE bets.nfl_sharp_alerts ADD COLUMN IF NOT EXISTS alert_provenance JSONB;
CREATE TABLE IF NOT EXISTS bets.nfl_sharp_quote_observations (
 observation_id TEXT PRIMARY KEY, game_id TEXT NOT NULL,
 observed_at_utc TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS nfl_sharp_quote_observations_game_time
 ON bets.nfl_sharp_quote_observations(game_id, observed_at_utc);
CREATE TABLE IF NOT EXISTS bets.nfl_sharp_observations (
 observation_id TEXT PRIMARY KEY, observed_at_utc TIMESTAMPTZ NOT NULL, payload JSONB NOT NULL);
CREATE OR REPLACE FUNCTION bets.nfl_sharp_observation_immutable() RETURNS trigger AS $$
 BEGIN RAISE EXCEPTION 'NFL sharp observations are append-only'; END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS nfl_sharp_quote_immutable ON bets.nfl_sharp_quote_observations;
CREATE TRIGGER nfl_sharp_quote_immutable BEFORE UPDATE OR DELETE ON bets.nfl_sharp_quote_observations
 FOR EACH ROW EXECUTE FUNCTION bets.nfl_sharp_observation_immutable();
DROP TRIGGER IF EXISTS nfl_sharp_observation_immutable ON bets.nfl_sharp_observations;
CREATE TRIGGER nfl_sharp_observation_immutable BEFORE UPDATE OR DELETE ON bets.nfl_sharp_observations
 FOR EACH ROW EXECUTE FUNCTION bets.nfl_sharp_observation_immutable();
-- The old key omitted the book (so two books' identical lines collided) and used a nullable
-- player_name_norm, which let every poll re-insert game totals. COALESCE closes both. Those
-- Preserve historical duplicates. If a legacy database lacks this index and contains
-- duplicates, migration must stop for review instead of deleting evidence.
ALTER TABLE bets.nfl_sharp_alerts DROP CONSTRAINT IF EXISTS nfl_sharp_alerts_event_id_stat_player_name_norm_fd_line_sid_key;
CREATE UNIQUE INDEX IF NOT EXISTS nfl_sharp_alerts_offer_key ON bets.nfl_sharp_alerts
  (event_id, book, stat, COALESCE(player_name_norm, ''), fd_line, side, fd_price);
"""


def model_view(cur, edge: dict, game: dict) -> tuple[float | None, bool | None]:
    """(projection, agrees) from the current frozen forecast for this player and stat, if one exists."""
    context = model_context(cur, edge, game)
    return context["projection"], context["agrees"]


FORECAST_SQL = """SELECT projection, line, over_probability, under_probability, injury_report_status,
        roster_status, limited_usage_risk FROM bets.nfl_player_prop_predictions
    WHERE game_id=%s AND stat=%s AND offer_player_name_norm=%s AND is_current
    ORDER BY id DESC LIMIT 1"""
FORECAST_FIELDS = ("projection", "line", "over_probability", "under_probability", "injury_report_status",
                   "roster_status", "limited_usage_risk")


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def shadow_filter(edge: dict, forecast: dict | None) -> tuple[str, list[str]]:
    """('pass' | 'flag' | 'no_forecast', reasons): does the model disagree on playing time or role?

    The price gap selects the bet. This asks whether the model knows something the sharp price
    may not have absorbed yet: an injury or a reduced role for an over, or a strongly different
    view of either side. It is recorded and graded, never applied (sharp_strategy.SHADOW_FILTER).
    """
    if not forecast or _number(forecast.get("projection")) is None:
        return "no_forecast", []
    rules, reasons = strategy.SHADOW_FILTER, []
    status = str(forecast.get("injury_report_status") or "").strip().lower()
    roster = str(forecast.get("roster_status") or "").strip().upper()
    if status in rules["unavailable_injury"]:
        reasons.append(f"injury_{status}")
    if roster and roster != "ACT":
        reasons.append(f"roster_{roster.lower()}")
    if edge["side"] == "over":
        if status in rules["over_risky_injury"]:
            reasons.append(f"over_with_{status}_tag")
        risk = _number(forecast.get("limited_usage_risk"))
        if risk is not None and risk >= rules["over_max_limited_usage_risk"]:
            reasons.append("over_with_limited_usage_risk")
    line = _number(forecast.get("line"))
    side_probability = _number(forecast.get(f"{edge['side']}_probability"))
    fair = _number(edge.get("fair_probability"))
    if (line is not None and side_probability is not None and fair is not None
            and abs(line - float(edge["line"])) < 1e-9
            and side_probability <= fair - rules["max_side_probability_gap"]):
        reasons.append("model_strongly_prefers_other_side")
    return ("flag" if reasons else "pass"), reasons


def model_context(cur, edge: dict, game: dict) -> dict:
    """The current frozen forecast for this player and stat, and the shadow filter's verdict.

    Matched on the book's own spelling of the player, which is what the forecast stores alongside its
    own short display name. Game markets have no player forecast.
    """
    empty = dict(projection=None, agrees=None, filter=None, filter_reasons=[])
    if edge["player_norm"] is None:
        return empty
    cur.execute(FORECAST_SQL, (game["game_id"], edge["stat"], edge["player_norm"]))
    row = cur.fetchone()
    forecast = dict(zip(FORECAST_FIELDS, row)) if row else None
    verdict, reasons = shadow_filter(edge, forecast)
    projection = _number((forecast or {}).get("projection"))
    if projection is None:
        return dict(empty, filter=verdict, filter_reasons=reasons)
    model_side = "over" if projection > float(edge["line"]) else "under"
    return dict(projection=projection, agrees=model_side == edge["side"], filter=verdict, filter_reasons=reasons)


def _fallback_link(cur, edge: dict, game: dict) -> str | None:
    """Book deep link from the regular quote feed for the same player/stat/line/side, if any."""
    if edge["player_norm"] is None:
        return None
    cur.execute(f"""SELECT {edge['side']}_link FROM odds.nfl_player_prop_lines
        WHERE bookmaker_key=%s AND player_name_norm=%s AND stat=%s AND line=%s AND {edge['side']}_link IS NOT NULL
          AND commence_time_utc BETWEEN %s AND %s ORDER BY fetched_at_utc DESC LIMIT 1""",
                (edge["book"], edge["player_norm"], edge["stat"], edge["line"],
                 game["start"] - timedelta(minutes=30), game["start"] + timedelta(minutes=30)))
    row = cur.fetchone()
    return row[0] if row else None


# Query parameters a book uses to carry one specific selection into its betslip.
SELECTION_PARAMS = ("outcomes", "selectionids", "selectionid", "marketid", "addtobetslip", "bet")


def _carries_a_selection(url) -> bool:
    query = (url.query or "").lower()
    return any(p in query for p in SELECTION_PARAMS) or "addtobetslip" in (url.path or "").lower()


def book_link(edge: dict, game: dict) -> tuple[str | None, str | None]:
    """(url, kind) where kind is:
       'betslip'   - resolved to the bettor's own state, one tap adds the exact selection (FanDuel)
       'selection' - the book's own deep link carrying this selection, not state-resolved
       'page'      - a valid link to the book, but it does not name the bet
    Host-checked throughout, so a bad link can never be rendered as the book's own."""
    book = edge.get("book") or "fanduel"
    if book == "fanduel":
        from nfl_pipeline.fanduel_links import betslip_for_row, provider_link
        row = dict(edge, away=game["away"], home=game["home"],
                   market=edge["stat"] if edge["stat"] in ("spread", "total") else None)
        slip = betslip_for_row(row)
        if slip:
            return slip, "betslip"
        fallback = provider_link(edge.get("link"))
        if not fallback:
            return None, None
        return fallback, ("selection" if _carries_a_selection(urlsplit(fallback)) else "page")
    link = str(edge.get("link") or "").strip()
    domain = BOOK_DOMAIN.get(book)
    if not link or not domain or any(c.isspace() or ord(c) < 32 or c in "<>\\" for c in link):
        return None, None
    try:
        url = urlsplit(link)
        host = url.hostname or ""
    except ValueError:
        return None, None
    if url.scheme != "https" or url.username or url.password or url.port is not None:
        return None, None
    if not (host == domain or host.endswith("." + domain)):
        return None, None
    return link, ("selection" if _carries_a_selection(url) else "page")


def format_alert(edge: dict, game: dict, stake: float = 0.0) -> str:
    who = edge["player"] or f"{game['away']} @ {game['home']}"
    label = edge["stat"].replace("_", " ").title()
    book = edge.get("book") or "fanduel"
    book_name = BOOK_LABEL.get(book, book.title())
    gap = f" ({edge['sharp_book'].title()} {edge['sharp_line']:g})" if edge["line_gap"] else f" ({edge['sharp_book'].title()})"
    link, kind = book_link(edge, game)
    kickoff = int(game["start"].timestamp())
    heading = f"**SHARP EDGE - bet ${stake:g} at {book_name}**" if stake > 0 else "**SHARP EDGE - research, not a bet**"
    return (f"{heading}\n{who} {edge['side'].upper()} {edge['line']:g} {label} "
            f"({game['away']} @ {game['home']}, <t:{kickoff}:R>)\n{book_name} {edge['price']:+d} | "
            f"fair {edge['fair_probability']:.1%}{gap} | "
            f"EV {edge['ev']:+.1%} | take at {edge['minimum_price']:+d} or better"
            + (f"\n[Add to slip](<{link}>)" if kind == "betslip"
               else f"\n[Open this bet on {book_name}](<{link}>)" if kind == "selection"
               else f"\n[Open {book_name} - find it manually](<{link}>)" if link
               else f"\nNo {book_name} link captured; search manually."))


def _post(text: str) -> None:
    from nfl_pipeline.sharp_delivery import _post_payload
    asyncio.run(_post_payload({"content": text[:1950], "allowed_mentions": {"parse": []}}))


# The Odds API moves an event's commence_time to the actual kickoff in the last minutes (13:30Z became
# 13:35Z for PHI-JAX on 11 October). Requiring an exact match threw away exactly the snapshots each
# decision is graded against. The same matchup within this tolerance is the same game; the schedule's
# kickoff stays the reference for every window and close.
KICKOFF_TOLERANCE_MINUTES = 15


def same_kickoff(api_commence, scheduled) -> bool:
    api, start = safe_time(api_commence), safe_time(scheduled)
    return api is not None and start is not None and abs((api - start).total_seconds()) <= KICKOFF_TOLERANCE_MINUTES * 60


def _has_games_soon(now: datetime) -> bool:
    """Whether any game kicks off in the next 24 hours; False when the database cannot say."""
    try:
        with psycopg2.connect(PG_DSN, connect_timeout=10) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='15s'")
            cur.execute("SELECT 1 FROM raw.nfl_games WHERE start_ts_utc > %s AND start_ts_utc <= %s LIMIT 1",
                        (now, now + timedelta(hours=24)))
            return cur.fetchone() is not None
    except psycopg2.Error:
        return False


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def warn_once(state: dict, now: datetime, reason: str, text: str, post) -> bool:
    """Post one Discord warning per ET day and reason. Returns whether it was sent.

    The scanner refused every run from 16 to 21 October (unregistered, then a market list that did
    not match its registration, then source drift) and nothing said so; every decision and close
    for those games was lost. A refusal while games are close is an outage, so it is announced.
    """
    day = str(now.astimezone(_ET).date())
    sent = state.setdefault("warnings", {})
    if sent.get(reason) == day:
        return False
    try:
        post(text)
    except Exception:  # an unsent warning is retried on the next run
        return False
    sent[reason] = day
    return True


def _refuse(result: dict, now: datetime, post) -> dict:
    """Return a refusal row, warning once a day if games are close enough for it to cost evidence."""
    if _has_games_soon(now):
        state = _load_state()
        text = (f"**NFL sharp scanner is not running: {result['status']}**\n"
                f"Registration `{EXPERIMENT_NAME}`. No gaps, stakes or closing prices are being captured.\n"
                + (f"{str(result.get('detail'))[:300]}\n" if result.get("detail") else "")
                + "Fix the drift, or register a successor with `python -m nfl_pipeline.sharp_watch --register`.")
        if warn_once(state, now, result["status"], text, post):
            atomic_json(STATE, state)
            result = dict(result, warned=True)
    return result


# Pricing, selection, staking and grading. A registration pins these and nothing else: the
# previous registrations pinned their whole import closure (79 files, from the model trainers to
# the cash ledger), so an unrelated edit to a model trial stopped the scanner. Transport (Discord,
# raw payload storage, prop parsing) and link display are deliberately left out.
PINNED_MODULES = ("nfl_pipeline.sharp_watch", "nfl_pipeline.sharp_strategy", "nfl_pipeline.sharp_math",
                  "nfl_pipeline.markets", "nfl_pipeline.context_contract",
                  "nfl_pipeline.modeling.sharp_alert_report")


def pinned_files(root: Path = ROOT) -> dict[str, Path]:
    return {module: root / "src" / Path(*module.split(".")).with_suffix(".py") for module in PINNED_MODULES}


def normalized_config() -> dict:
    """The scanner config as a registration stores it (JSON: tuples become lists, keys sorted)."""
    return json.loads(json.dumps(experiment_config(), sort_keys=True))


def register(now: datetime | None = None) -> dict:
    """Create the strategy's immutable registration from the current code and market list.

    Create-once: rerunning with identical rules and files returns the same registration; anything
    else needs a new EXPERIMENT_NAME in sharp_strategy.
    """
    from nfl_pipeline import experiment_registry as registry
    try:
        predecessor = registry.load_experiment(strategy.PREDECESSOR)
    except ValueError:
        predecessor = None
    config = dict(scanner=normalized_config(), predecessor_experiment_id=strategy.PREDECESSOR,
                  predecessor_sha256=(predecessor or {}).get("sha256"),
                  pinned_scope="pricing, selection, staking and grading modules only",
                  change_reason=("Cash strategy: FanDuel, ESPN Bet and DraftKings at +3% or better against "
                                 "Pinnacle no-vig, staked 15-45 minutes before kickoff, judged by EV at the "
                                 "sharp close; pin only the modules that price, select, stake and grade"))
    return registry.register_experiment(EXPERIMENT_NAME, config, pinned_files(), now=now)


def _games(now: datetime) -> list[dict]:
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT game_id, game_date_et, home_team_abbr, away_team_abbr, start_ts_utc FROM raw.nfl_games
            WHERE start_ts_utc > %s AND start_ts_utc <= %s ORDER BY start_ts_utc""", (now, now + timedelta(hours=24)))
        return [dict(zip(("game_id", "day", "home", "away", "start"), r)) for r in cur.fetchall()]


def _games_by_day(now: datetime) -> dict[date, int]:
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT (start_ts_utc AT TIME ZONE 'America/New_York')::date, count(*) FROM raw.nfl_games
            WHERE start_ts_utc > %s - interval '1 day' AND start_ts_utc <= %s + interval '40 days' GROUP BY 1""", (now, now))
        return {d: int(n) for d, n in cur.fetchall()}


def run(*, now: datetime | None = None, session=requests, post=_post) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    from nfl_pipeline.experiment_registry import check_experiment, load_experiment
    if load_experiment(EXPERIMENT_NAME) is None:
        return _refuse(dict(status="research_not_registered", experiment_id=EXPERIMENT_NAME), now, post)
    try:
        experiment = check_experiment(EXPERIMENT_NAME)
    except (ValueError, OSError) as exc:
        # Drift has to stop this scanner collecting evidence, not raise into the task
        # runner. Every other refusal here is reported as a status row; an exception
        # took the whole watcher down on 19 October over one unrelated column.
        return _refuse(dict(status="experiment_source_drift", experiment_id=EXPERIMENT_NAME,
                            detail=str(exc)), now, post)
    if experiment["config"].get("scanner") != normalized_config():
        registered = experiment["config"].get("scanner") or {}
        current = normalized_config()
        changed = sorted(k for k in set(registered) | set(current) if registered.get(k) != current.get(k))
        return _refuse(dict(status="experiment_configuration_drift", experiment_id=EXPERIMENT_NAME,
                            detail="changed: " + ", ".join(changed)), now, post)
    games = _games(now)
    if not games:
        return dict(status="no_games_within_24h")
    cfg = OddsCrawlerConfig()
    if not cfg.oddsapi_key:
        return dict(status="skipped", reason="ODDS_API_KEY is not set")
    state = _load_state()
    wanted = markets()
    cost = len(wanted)  # named bookmakers <= 10 -> one region-equivalent per market
    floor = _int_env("NFL_SHARP_CREDIT_FLOOR", 150)
    share = game_share(now.astimezone(_ET).date(), _games_by_day(now), _int_env("NFL_ODDS_API_RESET_DAY", 1))
    # Time-due first (no API call), then the free events call reports the true remaining credits.
    time_due, _ = plan_polls(games, state, now, 10**9, 0, 0)
    if not time_due:
        return dict(status="nothing_due", markets=wanted)
    events = session.get(f"{BASE}/events", params={"apiKey": cfg.oddsapi_key}, timeout=cfg.timeout_s)
    events.raise_for_status()
    remaining = int(events.headers.get("x-requests-remaining") or 0)
    by_matchup = {(normalize_team(e["home_team"]), normalize_team(e["away_team"])): e for e in events.json()}
    due, skipped = plan_polls(time_due, state, now, remaining, cost, floor, share)
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
        with conn.cursor() as cur:
            budget = risk_budget(cur, now)
        result["risk"] = {k: v for k, v in budget.items() if k not in ("staked", "staked_keys", "per_game")}
        if budget["paused"]:
            why = (["loss pause: realized ${:+.2f}".format(budget["realized_profit"])] if budget["loss_paused"] else []) \
                + budget["readiness_reasons"]
            warn_once(state, now, "staking_paused",
                      "**NFL sharp strategy: staking is paused**\n" + "; ".join(why or ["readiness pause"])
                      + "\nGaps are still scanned and graded at stake 0. See reports/nfl_sharp_alerts_latest.md.", post)
        for game in due:
            event = by_matchup.get((normalize_team(game["home"]), normalize_team(game["away"])))
            if event is None:
                skipped[game["game_id"]] = "no_matching_event"
                continue
            # Ask only for the markets whose gaps are still actionable for this kickoff. Budgeting
            # above assumed the full list, so the real spend can only come in under it.
            game_markets = markets_for((game["start"] - now).total_seconds() / 60, wanted)
            if not game_markets:
                skipped[game["game_id"]] = "no_market_in_window"
                continue
            params = {"apiKey": cfg.oddsapi_key, "bookmakers": ",".join(BOOKMAKERS), "markets": ",".join(game_markets),
                      "oddsFormat": "american", "dateFormat": "iso", "includeLinks": "true"}
            url = f"{BASE}/events/{event['id']}/odds"
            response = session.get(url, params=params, timeout=cfg.timeout_s)
            if not response.ok:
                skipped[game["game_id"]] = f"http_{response.status_code}"
                continue
            spent_here = len(game_markets)  # one credit per market, however many books are named
            remaining = int(response.headers.get("x-requests-remaining") or remaining - spent_here)
            payload = response.json()
            observed_at = datetime.now(timezone.utc)
            if not same_kickoff(payload.get("commence_time"), game["start"]):
                skipped[game["game_id"]] = "event_kickoff_mismatch"
                continue
            for endpoint, keep in (("nfl_player_props", lambda k: k not in GAME_MARKETS), ("nfl_game_odds", lambda k: k in GAME_MARKETS)):
                if any(keep(m.get("key")) for b in payload.get("bookmakers", []) for m in b.get("markets", [])):
                    _save_payload(conn, endpoint=endpoint, snapshot_role="live", as_of_date=game["day"],
                                  url=_full_url(url, params), payload=payload, provider=PROVIDER)
            state.setdefault("last_poll", {})[game["game_id"]] = now.isoformat()
            state.setdefault("spent", {})[today] = state.get("spent", {}).get(today, 0) + spent_here
            edges = find_edges(payload, min_ev=LOG_MIN_EV, now=observed_at)
            result["polled"].append(dict(game_id=game["game_id"], markets=game_markets, credits=spent_here,
                                         books=sorted(b["key"] for b in payload.get("bookmakers", [])),
                                         edges=len(edges)))
            with conn.cursor() as cur:
                persist_observations(cur, quote_observations(payload, game, observed_at))
                for edge in edges:
                    edge["link"] = edge["link"] or _fallback_link(cur, edge, game)
                    context = model_context(cur, edge, game)
                    projection, agrees = context["projection"], context["agrees"]
                    tier = alert_tier(edge["ev"], (game["start"] - observed_at).total_seconds() / 60, edge["stat"])
                    edge.update(model_projection=projection, model_agrees=agrees,
                                model_filter=context["filter"], model_filter_reasons=context["filter_reasons"])
                    observation = decision_observation(edge, game, tier, experiment["sha256"])
                    observation_id = observation["observation_id"]
                    cur.execute("""INSERT INTO bets.nfl_sharp_observations(observation_id,observed_at_utc,payload)
                        VALUES (%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (observation_id, observed_at, psycopg2.extras.Json(observation)))
                    key = (game["game_id"], edge["stat"], edge["player_norm"] or "")
                    stake = stake_for(cur, tier, now, budget, edge["stat"], key) if tier == "alert" else 0.0
                    # A logged gap that later reaches the alert threshold at the same price is upgraded and pinged.
                    cur.execute("""INSERT INTO bets.nfl_sharp_alerts (event_id, game_id, commence_time_utc, home_team, away_team,
                            stat, player_name, player_name_norm, side, fd_line, fd_price, fd_link, sharp_book, sharp_line,
                            fair_probability, ev, minimum_price, line_gap, tier, book, stake,
                            model_projection, model_agrees, initial_observation_id, alert_observation_id,
                            initial_provenance, alert_provenance, alerted_at, model_filter, model_filter_reasons)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (event_id, book, stat, COALESCE(player_name_norm, ''), fd_line, side, fd_price) DO UPDATE SET
                            tier='alert', sharp_book=EXCLUDED.sharp_book, sharp_line=EXCLUDED.sharp_line,
                            fair_probability=EXCLUDED.fair_probability, ev=EXCLUDED.ev, stake=EXCLUDED.stake,
                            minimum_price=EXCLUDED.minimum_price, line_gap=EXCLUDED.line_gap,
                            model_projection=EXCLUDED.model_projection, model_agrees=EXCLUDED.model_agrees,
                            model_filter=EXCLUDED.model_filter, model_filter_reasons=EXCLUDED.model_filter_reasons,
                            alert_observation_id=EXCLUDED.alert_observation_id, alert_provenance=EXCLUDED.alert_provenance
                          WHERE bets.nfl_sharp_alerts.tier IN ('logged', 'early') AND EXCLUDED.tier='alert'
                        RETURNING alert_id, tier, stake""",
                                (event["id"], game["game_id"], game["start"], game["home"], game["away"], edge["stat"],
                                 edge["player"], edge["player_norm"], edge["side"], edge["line"], edge["price"], edge["link"],
                                 edge["sharp_book"], edge["sharp_line"], edge["fair_probability"], edge["ev"],
                                 edge["minimum_price"], edge["line_gap"], tier, edge["book"], stake,
                                 projection, agrees, observation_id, observation_id if tier == "alert" else None,
                                 psycopg2.extras.Json(observation), psycopg2.extras.Json(observation) if tier == "alert" else None,
                                 observed_at, context["filter"], psycopg2.extras.Json(context["filter_reasons"])))
                    row = cur.fetchone()
                    if row and row[1] == "alert" and float(row[2] or 0.0) > 0:
                        new_alerts.append((row[0], edge, game, float(row[2])))
                        commit_stake(budget, float(row[2]), key)
                    elif row:
                        # Unstaked alerts (cap, pause, duplicate at another book, unbacktested market)
                        # are graded like the rest but not pinged: Discord carries only bets to place.
                        result["logged"] = result.get("logged", 0) + 1
            conn.commit()
        for alert_id, edge, game, stake in new_alerts[:MAX_ALERTS_PER_RUN]:
            try:
                post(format_alert(edge, game, stake) + f"\nConfirm with: sharp_bets --confirm {alert_id}")
                with conn.cursor() as cur:
                    cur.execute("UPDATE bets.nfl_sharp_alerts SET posted=TRUE WHERE alert_id=%s", (alert_id,))
                conn.commit()
            except Exception as exc:  # an unposted alert is still logged and graded
                result.setdefault("post_errors", []).append(type(exc).__name__)
        result["staked"] = round(budget["staked"], 2)
    from nfl_pipeline.parse_oddsapi import ParseConfig, parse_props
    for day in sorted({g["day"] for g in due}):
        parse_props(ParseConfig(as_of_date=day))
    state.update(credits_remaining=remaining, updated_at=now.isoformat())
    atomic_json(STATE, state)
    return dict(result, alerts=len(new_alerts), credits_remaining=remaining)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true",
                    help="create the strategy registration from the current code and NFL_SHARP_WATCH_MARKETS")
    args = ap.parse_args()
    if args.register:
        doc = register()
        print(json.dumps(dict(experiment_id=doc["experiment_id"], sha256=doc["sha256"],
                              registered_at=doc["registered_at"], pinned=sorted(doc["files"]),
                              markets=doc["config"]["scanner"]["markets"]), indent=2))
        return
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
