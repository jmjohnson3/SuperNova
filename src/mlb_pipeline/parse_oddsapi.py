"""
mlb_pipeline.parse_oddsapi
============================
Parse MLB Odds API payloads from raw.api_responses into:
  - odds.mlb_game_lines          — run-line (spread) and totals per bookmaker
  - odds.mlb_player_prop_lines   — pitcher strikeouts / batter H, HR, TB props

Sources:
  provider='oddsapi', endpoint='mlb_odds'              -> parse_game_odds()
  provider='oddsapi', endpoint='mlb_odds_historical'   -> parse_game_odds_historical()
  provider='oddsapi', endpoint='mlb_prop_odds'         -> parse_prop_odds()

Run:
  python -m mlb_pipeline.parse_oddsapi
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger("mlb_pipeline.parse_oddsapi")

DSN = "postgresql://josh:password@localhost:5432/nba"

# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------

_ODDS_API_TEAM_NAMES: dict[str, str] = {
    "arizona diamondbacks": "ARI",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "oakland athletics": "OAK",
    "athletics": "OAK",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "seattle mariners": "SEA",
    "san francisco giants": "SF",
    "st. louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WAS",
}


def _norm_team_name(name: str) -> str:
    """Normalize a full team name to a 3-letter MLB abbreviation.

    Strips accents, lowercases, and looks up in _ODDS_API_TEAM_NAMES.
    Returns the original name uppercased if no mapping found (graceful fallback).
    """
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    key = ascii_name.lower().strip()
    return _ODDS_API_TEAM_NAMES.get(key, name.upper())


# ---------------------------------------------------------------------------
# Prop market mapping
# ---------------------------------------------------------------------------

# Odds API market key -> internal stat name
_PROP_MARKET_MAP: dict[str, str] = {
    "pitcher_strikeouts":           "pitcher_strikeouts",
    "batter_hits":                  "batter_hits",
    "batter_hits_alternate":        "batter_hits",
    "batter_home_runs":             "batter_home_runs",
    "batter_home_runs_alternate":   "batter_home_runs",
    "batter_total_bases":           "batter_total_bases",
    "batter_total_bases_alternate": "batter_total_bases",
    "batter_walks":                 "batter_walks",
    "batter_walks_alternate":       "batter_walks",
}

# Alternate markets are Over-only multi-line props (FanDuel's format for MLB batter props).
# We only store FanDuel alternate data — books that also have standard lines (e.g. DraftKings)
# are skipped for alternate markets to avoid overwriting the cleaner standard Over/Under row.
_ALTERNATE_MARKETS = frozenset({
    "batter_hits_alternate",
    "batter_home_runs_alternate",
    "batter_total_bases_alternate",
    "batter_walks_alternate",
})

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_GAME_LINES_DDL = """
CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.mlb_game_lines (
    provider              TEXT        NOT NULL,
    as_of_date            DATE        NOT NULL,
    fetched_at_utc        TIMESTAMPTZ NOT NULL,
    event_id              TEXT        NOT NULL,
    commence_time_utc     TEXT,
    bookmaker_key         TEXT        NOT NULL,
    bookmaker_title       TEXT,
    home_team             TEXT,
    away_team             TEXT,
    -- run line (spread)
    spread_home_points    NUMERIC(6,2),
    spread_home_price     INTEGER,
    spread_away_points    NUMERIC(6,2),
    spread_away_price     INTEGER,
    -- totals
    total_points          NUMERIC(6,2),
    total_over_price      INTEGER,
    total_under_price     INTEGER,
    -- betslip deeplinks (present only for post-2026 crawls with includeLinks=true)
    spread_home_link      TEXT,
    spread_away_link      TEXT,
    total_over_link       TEXT,
    total_under_link      TEXT,
    updated_at_utc        TIMESTAMPTZ,
    PRIMARY KEY (provider, fetched_at_utc, event_id, bookmaker_key)
);

CREATE INDEX IF NOT EXISTS idx_mlb_game_lines_date_bk
    ON odds.mlb_game_lines (as_of_date, bookmaker_key);
"""

_GAME_LINES_UPSERT_SQL = """
INSERT INTO odds.mlb_game_lines (
  provider, as_of_date, fetched_at_utc,
  event_id, commence_time_utc,
  bookmaker_key, bookmaker_title,
  home_team, away_team,
  spread_home_points, spread_home_price,
  spread_away_points, spread_away_price,
  total_points, total_over_price, total_under_price,
  spread_home_link, spread_away_link, total_over_link, total_under_link,
  updated_at_utc
)
VALUES %s
ON CONFLICT (provider, fetched_at_utc, event_id, bookmaker_key)
DO UPDATE SET
  commence_time_utc  = EXCLUDED.commence_time_utc,
  bookmaker_title    = EXCLUDED.bookmaker_title,
  home_team          = EXCLUDED.home_team,
  away_team          = EXCLUDED.away_team,
  spread_home_points = EXCLUDED.spread_home_points,
  spread_home_price  = EXCLUDED.spread_home_price,
  spread_away_points = EXCLUDED.spread_away_points,
  spread_away_price  = EXCLUDED.spread_away_price,
  total_points       = EXCLUDED.total_points,
  total_over_price   = EXCLUDED.total_over_price,
  total_under_price  = EXCLUDED.total_under_price,
  spread_home_link   = EXCLUDED.spread_home_link,
  spread_away_link   = EXCLUDED.spread_away_link,
  total_over_link    = EXCLUDED.total_over_link,
  total_under_link   = EXCLUDED.total_under_link,
  updated_at_utc     = EXCLUDED.updated_at_utc
;
"""

_PROP_LINES_DDL = """
CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.mlb_player_prop_lines (
    id                SERIAL PRIMARY KEY,
    as_of_date        DATE        NOT NULL,
    fetched_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_id          TEXT        NOT NULL,
    commence_time_utc TEXT,
    bookmaker_key     TEXT        NOT NULL,
    home_team         TEXT,
    away_team         TEXT,
    player_name       TEXT        NOT NULL,
    player_name_norm  TEXT        NOT NULL,
    stat              TEXT        NOT NULL,
    line              NUMERIC,
    over_price        INTEGER,
    under_price       INTEGER,
    over_link         TEXT,
    under_link        TEXT,
    open_line         NUMERIC,
    updated_at_utc    TIMESTAMPTZ,
    UNIQUE (as_of_date, event_id, bookmaker_key, player_name_norm, stat)
);

CREATE INDEX IF NOT EXISTS idx_mlb_prop_lines_date_bk
    ON odds.mlb_player_prop_lines (as_of_date, bookmaker_key);
"""

_PROP_LINES_UPSERT_SQL = """
INSERT INTO odds.mlb_player_prop_lines (
  as_of_date, fetched_at_utc,
  event_id, commence_time_utc,
  bookmaker_key,
  home_team, away_team,
  player_name, player_name_norm,
  stat, line, over_price, under_price,
  over_link, under_link,
  updated_at_utc,
  open_line
)
VALUES %s
ON CONFLICT (as_of_date, event_id, bookmaker_key, player_name_norm, stat)
DO UPDATE SET
  fetched_at_utc    = EXCLUDED.fetched_at_utc,
  commence_time_utc = EXCLUDED.commence_time_utc,
  home_team         = EXCLUDED.home_team,
  away_team         = EXCLUDED.away_team,
  player_name       = EXCLUDED.player_name,
  line              = EXCLUDED.line,
  over_price        = EXCLUDED.over_price,
  under_price       = EXCLUDED.under_price,
  over_link         = EXCLUDED.over_link,
  under_link        = EXCLUDED.under_link,
  updated_at_utc    = EXCLUDED.updated_at_utc
  -- open_line intentionally omitted: stays as first-seen value for the day
;
"""

# ---------------------------------------------------------------------------
# SQL for loading raw payloads
# ---------------------------------------------------------------------------

_SQL_LOAD_LIVE = """
SELECT as_of_date, fetched_at_utc, payload
FROM raw.api_responses
WHERE provider = 'oddsapi'
  AND endpoint = 'mlb_odds'
  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
ORDER BY fetched_at_utc;
"""

_SQL_LOAD_HISTORICAL = """
SELECT as_of_date, fetched_at_utc, payload
FROM raw.api_responses
WHERE provider = 'oddsapi'
  AND endpoint = 'mlb_odds_historical'
  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
  AND (%(since_date)s IS NULL OR as_of_date >= %(since_date)s)
ORDER BY fetched_at_utc;
"""

_SQL_LOAD_PROP_ODDS = """
SELECT as_of_date, fetched_at_utc, payload
FROM raw.api_responses
WHERE provider = 'oddsapi'
  AND endpoint IN ('mlb_prop_odds', 'mlb_prop_odds_historical')
  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
  AND (%(since_date)s IS NULL OR as_of_date >= %(since_date)s)
ORDER BY fetched_at_utc;
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents, remove non-alpha-space characters."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _get_market(book: dict, key: str) -> Optional[dict]:
    for m in (book.get("markets") or []):
        if m.get("key") == key:
            return m
    return None


def _find_outcome(market: dict, name: str) -> Optional[dict]:
    for o in (market.get("outcomes") or []):
        if o.get("name") == name:
            return o
    return None


# ---------------------------------------------------------------------------
# Row iterators
# ---------------------------------------------------------------------------

def _iter_game_rows(
    as_of_date: date, fetched_at_utc, events: list[dict]
) -> Iterable[tuple]:
    """Yield tuples for odds.mlb_game_lines from a list of Odds API events."""
    provider = "oddsapi"
    for ev in events:
        event_id = ev.get("id")
        commence_time_utc = ev.get("commence_time")
        # The Odds API returns full team names; normalize to abbreviations
        home_team_raw = ev.get("home_team") or ""
        away_team_raw = ev.get("away_team") or ""
        home_team = _norm_team_name(home_team_raw) if home_team_raw else home_team_raw
        away_team = _norm_team_name(away_team_raw) if away_team_raw else away_team_raw

        for book in (ev.get("bookmakers") or []):
            bookmaker_key = book.get("key")
            bookmaker_title = book.get("title")

            spreads = _get_market(book, "spreads")
            totals = _get_market(book, "totals")

            spread_home_points = spread_home_price = None
            spread_away_points = spread_away_price = None
            total_points = total_over_price = total_under_price = None
            spread_home_link = spread_away_link = None
            total_over_link = total_under_link = None

            if spreads:
                # Outcomes are keyed by team name (full name from Odds API)
                oh = _find_outcome(spreads, home_team_raw)
                oa = _find_outcome(spreads, away_team_raw)
                if oh:
                    spread_home_points = _to_num(oh.get("point"))
                    spread_home_price = _to_int(oh.get("price"))
                    spread_home_link = oh.get("link")
                if oa:
                    spread_away_points = _to_num(oa.get("point"))
                    spread_away_price = _to_int(oa.get("price"))
                    spread_away_link = oa.get("link")

            if totals:
                o_over = _find_outcome(totals, "Over")
                o_under = _find_outcome(totals, "Under")
                if o_over:
                    total_points = _to_num(o_over.get("point"))
                    total_over_price = _to_int(o_over.get("price"))
                    total_over_link = o_over.get("link")
                if o_under:
                    if total_points is None:
                        total_points = _to_num(o_under.get("point"))
                    total_under_price = _to_int(o_under.get("price"))
                    total_under_link = o_under.get("link")

            yield (
                provider,
                as_of_date,
                fetched_at_utc,
                event_id,
                commence_time_utc,
                bookmaker_key,
                bookmaker_title,
                home_team,
                away_team,
                spread_home_points,
                spread_home_price,
                spread_away_points,
                spread_away_price,
                total_points,
                total_over_price,
                total_under_price,
                spread_home_link,
                spread_away_link,
                total_over_link,
                total_under_link,
            )


def _iter_prop_rows(
    as_of_date: date, fetched_at_utc, event_payload: dict
) -> Iterable[tuple]:
    """Yield tuples for odds.mlb_player_prop_lines from a single event payload.

    Standard market format (DraftKings, FanDuel pitcher_strikeouts):
      outcome.name        = "Over" or "Under"
      outcome.description = player name  (e.g. "Gerrit Cole")
      outcome.point       = line value
      outcome.price       = American odds integer
      outcome.link        = betslip deeplink (optional)

    Alternate market format (FanDuel batter props — Over-only, multi-line):
      All outcomes have name="Over" with different point values per player.
      We pick the single line with abs(price) closest to 0 (most even-money).
      Only FanDuel rows are stored for alternate markets; other books that have
      both standard and alternate are stored via the standard market only.
    """
    event_id = event_payload.get("id")
    commence_time_utc = event_payload.get("commence_time")
    home_team_raw = event_payload.get("home_team") or ""
    away_team_raw = event_payload.get("away_team") or ""
    home_team = _norm_team_name(home_team_raw) if home_team_raw else home_team_raw
    away_team = _norm_team_name(away_team_raw) if away_team_raw else away_team_raw

    for book in (event_payload.get("bookmakers") or []):
        bookmaker_key = book.get("key")

        for market in (book.get("markets") or []):
            market_key = market.get("key")
            stat = _PROP_MARKET_MAP.get(market_key)
            if stat is None:
                continue

            if market_key in _ALTERNATE_MARKETS:
                # FanDuel alternate format: Over-only, multiple line values per player.
                # Skip non-FanDuel bookmakers — they have standard lines already.
                if bookmaker_key != "fanduel":
                    continue

                # Collect all Over outcomes per player, then pick the most even-money line.
                by_player: dict[str, list[dict]] = {}
                for outcome in (market.get("outcomes") or []):
                    if (outcome.get("name") or "").strip().lower() != "over":
                        continue
                    player = (outcome.get("description") or "").strip()
                    if not player:
                        continue
                    by_player.setdefault(player, []).append({
                        "price": _to_int(outcome.get("price")),
                        "point": _to_num(outcome.get("point")),
                        "link":  outcome.get("link"),
                    })

                for player_name, candidates in by_player.items():
                    if not candidates:
                        continue
                    # Pick the line with abs(price) closest to 0 — the most contested line.
                    best = min(
                        candidates,
                        key=lambda x: abs(x["price"]) if x["price"] is not None else 999999,
                    )
                    yield (
                        as_of_date,
                        fetched_at_utc,
                        event_id,
                        commence_time_utc,
                        bookmaker_key,
                        home_team,
                        away_team,
                        player_name,
                        _normalize_name(player_name),
                        stat,
                        best["point"],   # line
                        best["price"],   # over_price
                        None,            # under_price — FD alternate is Over-only
                        best["link"],    # over_link
                        None,            # under_link
                        fetched_at_utc,
                        best["point"],   # open_line
                    )

            else:
                # Standard format: one Over + one Under per player.
                by_player_std: dict[str, dict] = {}
                for outcome in (market.get("outcomes") or []):
                    side = (outcome.get("name") or "").strip().lower()   # "over" or "under"
                    player = (outcome.get("description") or "").strip()
                    if not player:
                        continue
                    if player not in by_player_std:
                        by_player_std[player] = {
                            "over": None, "under": None, "line": None,
                            "over_link": None, "under_link": None,
                        }
                    price = _to_int(outcome.get("price"))
                    point = _to_num(outcome.get("point"))
                    if side == "over":
                        by_player_std[player]["over"] = price
                        by_player_std[player]["line"] = point
                        by_player_std[player]["over_link"] = outcome.get("link")
                    elif side == "under":
                        by_player_std[player]["under"] = price
                        if by_player_std[player]["line"] is None:
                            by_player_std[player]["line"] = point
                        by_player_std[player]["under_link"] = outcome.get("link")

                for player_name, vals in by_player_std.items():
                    yield (
                        as_of_date,
                        fetched_at_utc,
                        event_id,
                        commence_time_utc,
                        bookmaker_key,
                        home_team,
                        away_team,
                        player_name,
                        _normalize_name(player_name),
                        stat,
                        vals["line"],
                        vals["over"],
                        vals["under"],
                        vals["over_link"],
                        vals["under_link"],
                        fetched_at_utc,
                        vals["line"],   # open_line: preserved by ON CONFLICT (not updated)
                    )


# ---------------------------------------------------------------------------
# Public parse functions
# ---------------------------------------------------------------------------

def parse_game_odds(
    conn,
    as_of_date: Optional[date] = None,
) -> None:
    """Parse mlb_odds payloads (live / today's odds) into odds.mlb_game_lines.

    Reads provider='oddsapi', endpoint='mlb_odds' from raw.api_responses.
    Each payload is expected to be a list of event dicts as returned by the
    Odds API /sports/baseball_mlb/odds endpoint.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_SQL_LOAD_LIVE, {"as_of_date": as_of_date})
        snaps = cur.fetchall()

    if not snaps:
        log.warning("No mlb_odds snapshots found (as_of_date=%s).", as_of_date)
        return

    total_rows = 0
    with conn.cursor() as cur:
        for s in snaps:
            as_of = s["as_of_date"]
            fetched_at = s["fetched_at_utc"]
            payload = s["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            if not isinstance(payload, list):
                log.warning(
                    "Unexpected mlb_odds payload type=%s at %s; skipping.",
                    type(payload), fetched_at,
                )
                continue

            rows = list(_iter_game_rows(as_of, fetched_at, payload))
            if not rows:
                continue

            psycopg2.extras.execute_values(
                cur,
                _GAME_LINES_UPSERT_SQL,
                [(*r, fetched_at) for r in rows],   # append updated_at_utc
                page_size=500,
            )
            total_rows += len(rows)

    conn.commit()
    log.info("Upserted %d rows into odds.mlb_game_lines (live odds).", total_rows)


def parse_game_odds_historical(
    conn,
    as_of_date: Optional[date] = None,
    since_date: Optional[date] = None,
) -> None:
    """Parse mlb_odds_historical payloads into odds.mlb_game_lines.

    The historical endpoint wraps each snapshot as:
      {"data": [...events], "timestamp": ..., "next_timestamp": ..., ...}

    Incremental by default: when both as_of_date and since_date are None,
    computes since_date = MAX(existing as_of_date) - 1 day to avoid re-parsing
    all history on every daily run.

    Pass since_date to force re-parsing from a specific date (e.g. after backfill).
    """
    # Compute incremental since_date
    if since_date is None and as_of_date is None:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT MAX(as_of_date) FROM odds.mlb_game_lines")
                row = cur.fetchone()
                if row and row[0]:
                    since_date = row[0] - timedelta(days=1)
                    log.info("Incremental historical MLB game odds: since %s", since_date)
            except Exception:
                pass  # table may not exist yet on first run
    elif since_date is not None:
        log.info("Forced historical MLB game odds re-parse: since %s", since_date)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_SQL_LOAD_HISTORICAL, {"as_of_date": as_of_date, "since_date": since_date})
        snaps = cur.fetchall()

    if not snaps:
        log.warning("No mlb_odds_historical snapshots found (as_of_date=%s).", as_of_date)
        return

    total_rows = 0
    batch: list[tuple] = []

    def _flush(cur_) -> int:
        if not batch:
            return 0
        psycopg2.extras.execute_values(
            cur_,
            _GAME_LINES_UPSERT_SQL,
            [(*r, r[2]) for r in batch],   # updated_at_utc = fetched_at_utc
            page_size=500,
        )
        n = len(batch)
        batch.clear()
        return n

    with conn.cursor() as cur:
        for s in snaps:
            payload = s["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            # Historical endpoint wraps: {"data": [...events], "timestamp": ...}
            if isinstance(payload, dict) and "data" in payload:
                events = payload["data"]
            elif isinstance(payload, list):
                events = payload
            else:
                log.warning(
                    "Unexpected mlb_odds_historical payload type=%s; skipping.",
                    type(payload),
                )
                continue

            if not events:
                continue

            rows = list(_iter_game_rows(s["as_of_date"], s["fetched_at_utc"], events))
            batch.extend(rows)

            if len(batch) >= 2000:
                total_rows += _flush(cur)
                conn.commit()

        total_rows += _flush(cur)

    conn.commit()
    log.info(
        "Upserted %d rows from mlb_odds_historical into odds.mlb_game_lines.", total_rows
    )


def parse_prop_odds(
    conn,
    as_of_date: Optional[date] = None,
    since_date: Optional[date] = None,
) -> None:
    """Parse mlb_prop_odds payloads into odds.mlb_player_prop_lines.

    Stats parsed: pitcher_strikeouts, batter_hits, batter_home_runs, batter_total_bases.
    FanDuel batter props come from *_alternate markets (Over-only, multi-line);
    DraftKings batter props come from standard markets (Over + Under, single line).

    Incremental by default: when both as_of_date and since_date are None,
    computes since_date = MAX(existing as_of_date) - 1 day.

    Pass since_date to force re-parsing from a specific date (e.g. after backfill).
    """
    # Compute incremental since_date
    if since_date is None and as_of_date is None:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT MAX(as_of_date) FROM odds.mlb_player_prop_lines")
                row = cur.fetchone()
                if row and row[0]:
                    since_date = row[0] - timedelta(days=1)
                    log.info("Incremental MLB prop odds: since %s", since_date)
            except Exception:
                pass  # table may not exist yet on first run
    elif since_date is not None:
        log.info("Forced MLB prop odds re-parse: since %s", since_date)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_SQL_LOAD_PROP_ODDS, {"as_of_date": as_of_date, "since_date": since_date})
        snaps = cur.fetchall()

    if not snaps:
        log.warning("No mlb_prop_odds snapshots found (as_of_date=%s).", as_of_date)
        return

    total_rows = 0
    with conn.cursor() as cur:
        for s in snaps:
            fetched_at = s["fetched_at_utc"]
            payload = s["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            # Historical per-event endpoint wraps under "data" key alongside "timestamp"
            if isinstance(payload, dict) and "timestamp" in payload and "data" in payload:
                payload = payload["data"]

            # Per-event endpoint returns a single event dict; list endpoint returns a list
            if isinstance(payload, dict):
                events = [payload]
            elif isinstance(payload, list):
                events = payload
            else:
                log.warning(
                    "Unexpected mlb_prop_odds payload type=%s; skipping.", type(payload)
                )
                continue

            for event_payload in events:
                rows = list(_iter_prop_rows(s["as_of_date"], fetched_at, event_payload))
                if not rows:
                    continue
                psycopg2.extras.execute_values(
                    cur,
                    _PROP_LINES_UPSERT_SQL,
                    rows,
                    page_size=500,
                )
                total_rows += len(rows)

    conn.commit()
    log.info("Upserted %d rows into odds.mlb_player_prop_lines.", total_rows)


# ---------------------------------------------------------------------------
# Ensure tables
# ---------------------------------------------------------------------------

def _ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_GAME_LINES_DDL)
        cur.execute(_PROP_LINES_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        _ensure_tables(conn)
        parse_game_odds(conn)
        parse_game_odds_historical(conn)
        parse_prop_odds(conn)
        log.info("Done.")
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
