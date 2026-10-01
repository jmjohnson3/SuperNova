"""Fetch NFL games and player props from The Odds API."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import ODDS_API_GAME_MARKETS, ODDS_API_MARKETS, normalize_name, normalize_team
from nfl_pipeline.schema import GAME_LINE_CHANGED, PROP_LINE_CHANGED, ensure_schema
from supernovabets_config import _saved_windows_env

log = logging.getLogger("nfl_pipeline.crawler_oddsapi")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANUAL_CSV_DIR = _ROOT / "data" / "manual_odds" / "nfl"


class OddsApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, error_code: str | None = None, retriable: bool = True) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retriable = retriable


@dataclass(frozen=True)
class OddsCrawlerConfig:
    pg_dsn: str = PG_DSN
    oddsapi_key: str = os.getenv("ODDS_API_KEY") or _saved_windows_env("ODDS_API_KEY") or ""
    sport: str = "americanfootball_nfl"
    regions: str = "us"
    bookmakers: str = "fanduel,draftkings"
    odds_format: str = "american"
    date_format: str = "iso"
    markets: str = ODDS_API_MARKETS
    game_markets: str = ODDS_API_GAME_MARKETS
    snapshot_role: str = "live"
    max_retries: int = 5
    base_backoff_s: float = 1.5
    timeout_s: int = 30
    sleep_between_calls_s: float = 0.25
    single_market_fallback_on_empty: bool = True
    provider_order: str = (
        os.getenv("NFL_ODDS_PROVIDER_ORDER")
        or _saved_windows_env("NFL_ODDS_PROVIDER_ORDER")
        or "sportsgameodds,therundown,oddsapi,manual_csv"
    )
    sports_game_odds_key: str = (
        os.getenv("SPORTSGAMEODDS_API_KEY")
        or os.getenv("SPORTS_GAME_ODDS_API_KEY")
        or os.getenv("SPORTS_ODDS_API_KEY_HEADER")
        or _saved_windows_env("SPORTSGAMEODDS_API_KEY", "SPORTS_GAME_ODDS_API_KEY", "SPORTS_ODDS_API_KEY_HEADER")
        or ""
    )
    therundown_key: str = os.getenv("THERUNDOWN_API_KEY") or _saved_windows_env("THERUNDOWN_API_KEY") or ""
    manual_csv_dir: str = os.getenv("NFL_MANUAL_ODDS_CSV_DIR") or _saved_windows_env("NFL_MANUAL_ODDS_CSV_DIR") or str(_DEFAULT_MANUAL_CSV_DIR)


def _sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_z(dt: datetime) -> str:
    return dt.astimezone(_UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _et_day_window_utc(et_day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(et_day, dtime(0, 0), tzinfo=_ET)
    return start.astimezone(_UTC), (start + timedelta(days=1)).astimezone(_UTC)


def _full_url(url: str, params: dict) -> str:
    qs = "&".join(
        f"{key}={params[key]}"
        for key in sorted(params)
        if key.lower() != "apikey"
    )
    return f"{url}?{qs}"


def _provider_names(raw: str) -> list[str]:
    aliases = {
        "sgo": "sportsgameodds",
        "sports_game_odds": "sportsgameodds",
        "sports-game-odds": "sportsgameodds",
        "the_odds_api": "oddsapi",
        "the-odds-api": "oddsapi",
        "therundown": "therundown",
        "the-rundown": "therundown",
        "the_rundown": "therundown",
        "manual": "manual_csv",
        "manual-csv": "manual_csv",
    }
    out: list[str] = []
    for item in str(raw or "").split(","):
        key = item.strip().lower()
        if not key:
            continue
        out.append(aliases.get(key, key))
    return out or ["sportsgameodds", "therundown", "oddsapi", "manual_csv"]


def _skip_result(provider: str, cfg: OddsCrawlerConfig, et_day: date, reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "provider": provider,
        "game_date": et_day.isoformat(),
        "snapshot_role": cfg.snapshot_role,
        "events": 0,
        "game_odds_saved": 0,
        "player_prop_events_saved": 0,
        "reason": reason,
    }


def _provider_attempt_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": result.get("provider"),
        "status": result.get("status"),
        "events": result.get("events"),
        "game_odds_saved": result.get("game_odds_saved"),
        "player_prop_events_saved": result.get("player_prop_events_saved"),
        "manual_prop_rows_upserted": result.get("manual_prop_rows_upserted"),
        "manual_game_rows_upserted": result.get("manual_game_rows_upserted"),
        "reason": result.get("reason") or result.get("game_odds_error"),
        "player_prop_fetch_failures": result.get("player_prop_fetch_failures"),
    }


def _book_key(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    compact = "".join(ch for ch in text if ch.isalnum())
    aliases = {
        "fanduel": "fanduel",
        "fd": "fanduel",
        "draftkings": "draftkings",
        "dk": "draftkings",
    }
    return aliases.get(compact)


def _book_title(book_key: str | None) -> str | None:
    return {"fanduel": "FanDuel", "draftkings": "DraftKings"}.get(str(book_key or "").lower())


def _clean_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _clean_int(value: Any) -> int | None:
    numeric = _clean_float(value)
    if numeric is None:
        return None
    if 1.01 <= numeric <= 20.0:
        if numeric >= 2.0:
            return int(round((numeric - 1.0) * 100.0))
        return int(round(-100.0 / (numeric - 1.0)))
    return int(round(numeric))


def _first_value(obj: Any, keys: tuple[str, ...], *, max_depth: int = 3) -> Any:
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        for key in keys:
            if key.lower() in lower_map and lower_map[key.lower()] not in (None, ""):
                return lower_map[key.lower()]
        for value in obj.values():
            found = _first_value(value, keys, max_depth=max_depth - 1)
            if found not in (None, ""):
                return found
    elif isinstance(obj, list):
        for value in obj[:20]:
            found = _first_value(value, keys, max_depth=max_depth - 1)
            if found not in (None, ""):
                return found
    return None


def _sgo_odd_parts(odd_id: str) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    parts = [part for part in str(odd_id or "").split("-") if part]
    if len(parts) < 5:
        return None, None, None, None, None
    side_id = parts[-1].lower()
    bet_type_id = parts[-2].lower()
    period_id = parts[-3].lower()
    stat_entity_id = parts[-4]
    stat_id = "-".join(parts[:-4]).lower()
    return stat_id, stat_entity_id, period_id, bet_type_id, side_id


def _sgo_market_key(stat_id: str | None) -> str | None:
    key = str(stat_id or "").lower().replace("-", "_")
    return {
        "passing_yards": "player_pass_yds",
        "passing_touchdowns": "player_pass_tds",
        "passing_tds": "player_pass_tds",
        "rushing_yards": "player_rush_yds",
        "rushing_touchdowns": "player_rush_tds",
        "rushing_tds": "player_rush_tds",
        "receiving_yards": "player_reception_yds",
        "receiving_receptions": "player_receptions",
        "receptions": "player_receptions",
        "receiving_touchdowns": "player_reception_tds",
        "receiving_tds": "player_reception_tds",
    }.get(key)


def _sgo_player_lookup(event: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}

    def visit(obj: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(obj, dict):
            ident = (
                obj.get("playerID") or obj.get("playerId") or obj.get("statEntityID")
                or obj.get("statEntityId") or obj.get("id") or obj.get("entityID")
            )
            name = (
                obj.get("playerName") or obj.get("name") or obj.get("displayName")
                or obj.get("fullName") or obj.get("statEntityName")
            )
            if ident and name:
                lookup[str(ident)] = str(name)
            for value in obj.values():
                visit(value, depth + 1)
        elif isinstance(obj, list):
            for value in obj[:500]:
                visit(value, depth + 1)

    for key in ("players", "statEntities", "participants", "entities"):
        visit(event.get(key))
    return lookup


def _sgo_team_name(event: dict[str, Any], side: str) -> str | None:
    team = (event.get('teams') or {}).get(side) if isinstance(event.get('teams'),dict) else None
    if isinstance(team,dict):
        names = team.get('names') or {}
        if isinstance(names,dict) and (names.get('long') or names.get('short')):
            return str(names.get('long') or names.get('short'))
    direct_keys = (
        f"{side}TeamName",
        f"{side}_team_name",
        f"{side}Team",
        f"{side}_team",
        side,
    )
    value = _first_value(event, direct_keys, max_depth=2)
    if isinstance(value, dict):
        value = _first_value(value, ("name", "displayName", "fullName", "abbreviation"), max_depth=2)
    if value:
        return str(value)
    for key in ("teams", "competitors", "participants"):
        rows = event.get(key)
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                marker = str(row.get("side") or row.get("homeAway") or row.get("teamType") or "").lower()
                is_home = row.get("isHome")
                if marker == side or (side == "home" and is_home is True) or (side == "away" and is_home is False):
                    name = _first_value(row, ("name", "displayName", "fullName", "teamName", "abbreviation"), max_depth=2)
                    if name:
                        return str(name)
    return None


def _event_start(event: dict[str, Any]) -> str | None:
    value = _first_value(
        event,
        (
            "commence_time", "commenceTime", "startTime", "start_time",
            "startsAt", "starts_at", "eventTime", "event_time", "scheduled",
        ),
        max_depth=2,
    )
    return str(value) if value else None


def _sgo_quotes(book_payload: dict, *, include_alts: bool):
    # Never recursively borrow an alternate price/line for the main quote.
    main = {k: v for k, v in book_payload.items() if k not in {'altLines', 'alt_lines'}}
    if main.get('available') is not False:
        yield main, False
    if include_alts:
        for quote in book_payload.get('altLines', []) or []:
            # Retained but inactive alternates are not executable close evidence.
            if isinstance(quote, dict) and quote.get('available') is True:
                yield quote, True


def _canonical_event_from_sgo(event: dict[str, Any], *, include_alts: bool = False) -> dict[str, Any] | None:
    odds_obj = event.get("odds")
    if not isinstance(odds_obj, dict):
        return None
    player_lookup = _sgo_player_lookup(event)
    canonical: dict[str, Any] = {
        "id": str(event.get("eventID") or event.get("eventId") or event.get("id") or ""),
        "commence_time": _event_start(event),
        "home_team": _sgo_team_name(event, "home"),
        "away_team": _sgo_team_name(event, "away"),
        "bookmakers": [],
    }
    books: dict[str, dict[str, Any]] = {}

    def book_rec(book_key: str) -> dict[str, Any]:
        rec = books.get(book_key)
        if rec is None:
            rec = {"key": book_key, "title": _book_title(book_key) or book_key, "markets": {}}
            books[book_key] = rec
        return rec

    def add_market(book_key: str, market_key: str, outcome: dict[str, Any]) -> None:
        rec = book_rec(book_key)
        markets = rec["markets"]
        market = markets.setdefault(market_key, {"key": market_key, "outcomes": []})
        market["outcomes"].append(outcome)

    for odd_key, raw_odd in odds_obj.items():
        if not isinstance(raw_odd, dict):
            continue
        odd_id = str(raw_odd.get("oddID") or raw_odd.get("oddId") or odd_key)
        stat_id, stat_entity_id, period_id, bet_type_id, side_id = _sgo_odd_parts(odd_id)
        if period_id != "game" or raw_odd.get('started') is True or raw_odd.get('ended') is True or raw_odd.get('cancelled') is True:
            continue
        by_book = raw_odd.get("byBookmaker") or raw_odd.get("by_bookmaker") or raw_odd.get("books") or raw_odd.get("bookmakers")
        if not isinstance(by_book, dict):
            continue
        for raw_book_key, book_payload in by_book.items():
            if not isinstance(book_payload,dict):
                continue
            book_key = _book_key(raw_book_key) or _book_key(_first_value(book_payload, ("bookmakerID", "bookmakerId", "bookmaker", "sportsbook")))
            if book_key not in {"fanduel", "draftkings"}:
                continue
            main = next((q for q, alt in _sgo_quotes(book_payload, include_alts=False)), None)
            price = _clean_int(_first_value(main or {}, ("americanOdds", "american_odds", "oddsAmerican", "price", "odds", "bookOdds", "currentOdds"), max_depth=2))
            line = _clean_float(_first_value(main or {}, ("overUnder", "over_under", "line", "point", "points", "spread", "value"), max_depth=2))
            link = _first_value(main or {}, ("link", "url", "deeplink", "deepLink", "deep_link", "affiliateLink", "bookmakerDeepLink", "betLink"), max_depth=2)
            if bet_type_id == "sp" and side_id in {"home", "away"} and price is not None:
                point = line
                if point is None:
                    point = _clean_float(_first_value(raw_odd, ("spread", "point", "points", "line"), max_depth=2))
                add_market(book_key, "spreads", {
                    "name": canonical["home_team"] if side_id == "home" else canonical["away_team"],
                    "point": point,
                    "price": price,
                    "link": link,
                })
            elif bet_type_id == "ou" and stat_entity_id in {'all','total'} and stat_id in {"points", "score", "total", "total-points", "game-total"} and side_id in {"over", "under"} and price is not None:
                point = line
                if point is None:
                    point = _clean_float(_first_value(raw_odd, ("overUnder", "line", "point", "points"), max_depth=2))
                add_market(book_key, "totals", {
                    "name": side_id.title(),
                    "point": point,
                    "price": price,
                    "link": link,
                })
            elif bet_type_id == "ou" and side_id in {"over", "under"}:
                market_key = _sgo_market_key(stat_id)
                player = (
                    _first_value(raw_odd, ("playerName", "statEntityName", "participantName", "name"), max_depth=2)
                    or player_lookup.get(str(stat_entity_id or ""))
                )
                if not market_key or not player:
                    continue
                seen_lines = set()
                for quote, is_alt in _sgo_quotes(book_payload, include_alts=include_alts):
                    price = _clean_int(_first_value(quote, ("americanOdds", "american_odds", "oddsAmerican", "price", "odds", "bookOdds", "currentOdds"), max_depth=2))
                    line = _clean_float(_first_value(quote, ("overUnder", "over_under", "line", "point", "points", "spread", "value"), max_depth=2))
                    if price is None or abs(price) < 100 or line is None or line in seen_lines:
                        continue
                    seen_lines.add(line)
                    add_market(book_key, market_key, {
                        "name": side_id.title(), "description": str(player), "point": line, "price": price,
                        "link": _first_value(quote, ("link", "url", "deeplink", "deepLink", "deep_link", "affiliateLink", "bookmakerDeepLink", "betLink"), max_depth=2),
                        "is_alt_line": is_alt,
                    })
    for rec in books.values():
        rec["markets"] = list(rec["markets"].values())
        if rec["markets"]:
            canonical["bookmakers"].append(rec)
    return canonical if canonical["bookmakers"] else None


def _canonical_events_from_sgo_payload(payload: object, *, include_alts: bool = False) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [event for event in (_canonical_event_from_sgo(row, include_alts=include_alts) for row in data if isinstance(row, dict)) if event]


def _has_market(events: list[dict[str, Any]], wanted: set[str]) -> bool:
    for event in events:
        for book in event.get("bookmakers", []) or []:
            for market in book.get("markets", []) or []:
                if market.get("key") in wanted:
                    return True
    return False


def _api_error_message(response: requests.Response) -> tuple[str, str | None]:
    error_code: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = str(payload.get("message") or payload.get("error") or response.text or "").strip()
        error_code_raw = payload.get("error_code")
        error_code = str(error_code_raw).strip() if error_code_raw else None
        if error_code:
            return f"{response.status_code} from The Odds API [{error_code}]: {message}", error_code
        return f"{response.status_code} from The Odds API: {message}", error_code
    body = response.text.strip().replace("\n", " ")[:300]
    return f"{response.status_code} from The Odds API: {body}", error_code


def _fetch(cfg: OddsCrawlerConfig, url: str, params: dict) -> object:
    if not cfg.oddsapi_key:
        raise RuntimeError("ODDS_API_KEY is not set")
    last_exc: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=cfg.timeout_s)
            if response.status_code == 429:
                wait = cfg.base_backoff_s * (2 ** (attempt - 1))
                log.warning("Odds API rate limited; sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            if not response.ok:
                message, error_code = _api_error_message(response)
                retriable = response.status_code >= 500
                raise OddsApiError(
                    message,
                    status_code=response.status_code,
                    error_code=error_code,
                    retriable=retriable,
                )
            return response.json()
        except OddsApiError as exc:
            last_exc = exc
            if not exc.retriable:
                log.warning("Odds API fetch failed (%s); not retrying", exc)
                raise
            wait = cfg.base_backoff_s * (2 ** (attempt - 1))
            detail = str(exc).strip().replace("\n", " ")[:300]
            log.warning(
                "Odds API fetch failed (%s: %s); sleeping %.1fs",
                exc.__class__.__name__,
                detail,
                wait,
            )
            time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            wait = cfg.base_backoff_s * (2 ** (attempt - 1))
            detail = str(exc).strip().replace("\n", " ")[:300]
            log.warning(
                "Odds API fetch failed (%s: %s); sleeping %.1fs",
                exc.__class__.__name__,
                detail,
                wait,
            )
            time.sleep(wait)
    detail = f": {str(last_exc).strip().replace(chr(10), ' ')[:300]}" if last_exc else ""
    raise RuntimeError(f"Odds API fetch failed after {cfg.max_retries} attempts{detail}") from last_exc


def _save_payload(
    conn,
    *,
    endpoint: str,
    snapshot_role: str,
    as_of_date: date,
    url: str,
    payload: object,
    provider: str = "oddsapi",
) -> None:
    fetched_at = datetime.now(_UTC)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw.nfl_api_responses (
                provider, endpoint, snapshot_role, season, game_slug, as_of_date,
                url, fetched_at_utc, payload, payload_sha256
            )
            VALUES (%(provider)s, %(endpoint)s, %(snapshot_role)s, NULL, NULL, %(as_of_date)s,
                    %(url)s, %(fetched_at)s, %(payload)s::jsonb, %(sha)s)
            """,
            {
                "provider": provider,
                "endpoint": endpoint,
                "snapshot_role": snapshot_role,
                "as_of_date": as_of_date,
                "url": url,
                "fetched_at": fetched_at,
                "payload": json.dumps(payload),
                "sha": _sha256_json(payload),
            },
        )
    conn.commit()


def _payload_bookmaker_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    books = payload.get("bookmakers")
    return len(books) if isinstance(books, list) else 0


def _fetch_the_odds_api_for_date(cfg: OddsCrawlerConfig, et_day: date) -> dict[str, Any]:
    start_utc, end_utc = _et_day_window_utc(et_day)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        events_url = f"https://api.the-odds-api.com/v4/sports/{cfg.sport}/events"
        events_params = {
            "apiKey": cfg.oddsapi_key,
            "dateFormat": cfg.date_format,
            "commenceTimeFrom": _to_z(start_utc),
            "commenceTimeTo": _to_z(end_utc),
        }
        events_payload = _fetch(cfg, events_url, events_params)
        _save_payload(
            conn,
            endpoint="nfl_events",
            snapshot_role=cfg.snapshot_role,
            as_of_date=et_day,
            url=_full_url(events_url, events_params),
            payload=events_payload,
        )

        game_odds_url = f"https://api.the-odds-api.com/v4/sports/{cfg.sport}/odds"
        game_odds_params = {
            "apiKey": cfg.oddsapi_key,
            "regions": cfg.regions,
            "markets": cfg.game_markets,
            "bookmakers": cfg.bookmakers,
            "oddsFormat": cfg.odds_format,
            "dateFormat": cfg.date_format,
            "includeLinks": "true",
            "commenceTimeFrom": _to_z(start_utc),
            "commenceTimeTo": _to_z(end_utc),
        }
        game_odds_saved = 0
        game_odds_error: str | None = None
        try:
            game_odds_payload = _fetch(cfg, game_odds_url, game_odds_params)
            _save_payload(
                conn,
                endpoint="nfl_game_odds",
                snapshot_role=cfg.snapshot_role,
                as_of_date=et_day,
                url=_full_url(game_odds_url, game_odds_params),
                payload=game_odds_payload,
            )
            game_odds_saved = 1
        except Exception as exc:
            game_odds_error = str(exc)
            log.warning("NFL game odds fetch failed for %s; continuing to player props: %s", et_day, exc)

        event_rows = events_payload if isinstance(events_payload, list) else []
        props_saved = 0
        prop_fetch_failures: list[dict[str, str | None]] = []
        single_market_fallbacks: list[dict[str, str | int | None]] = []
        for event in event_rows:
            event_id = event.get("id")
            if not event_id:
                continue
            props_url = f"https://api.the-odds-api.com/v4/sports/{cfg.sport}/events/{event_id}/odds"
            props_params = {
                "apiKey": cfg.oddsapi_key,
                "regions": cfg.regions,
                "markets": cfg.markets,
                "bookmakers": cfg.bookmakers,
                "oddsFormat": cfg.odds_format,
                "dateFormat": cfg.date_format,
                "includeLinks": "true",
            }
            try:
                payload = _fetch(cfg, props_url, props_params)
            except Exception as exc:
                log.warning(
                    "Skipping NFL player props for event %s (%s at %s): %s",
                    event_id,
                    event.get("away_team"),
                    event.get("home_team"),
                    exc,
                )
                prop_fetch_failures.append({
                    "event_id": str(event_id),
                    "away_team": event.get("away_team"),
                    "home_team": event.get("home_team"),
                    "error": str(exc),
                })
                continue
            if cfg.single_market_fallback_on_empty and _payload_bookmaker_count(payload) == 0:
                non_empty_market_payloads = 0
                for market_key in [market.strip() for market in cfg.markets.split(",") if market.strip()]:
                    single_params = dict(props_params)
                    single_params["markets"] = market_key
                    try:
                        single_payload = _fetch(cfg, props_url, single_params)
                    except Exception as exc:
                        prop_fetch_failures.append({
                            "event_id": str(event_id),
                            "away_team": event.get("away_team"),
                            "home_team": event.get("home_team"),
                            "error": f"{market_key}: {exc}",
                        })
                        continue
                    book_count = _payload_bookmaker_count(single_payload)
                    if book_count > 0:
                        _save_payload(
                            conn,
                            endpoint="nfl_player_props",
                            snapshot_role=cfg.snapshot_role,
                            as_of_date=et_day,
                            url=_full_url(props_url, single_params),
                            payload=single_payload,
                        )
                        props_saved += 1
                        non_empty_market_payloads += 1
                    single_market_fallbacks.append({
                        "event_id": str(event_id),
                        "market": market_key,
                        "bookmakers": book_count,
                    })
                    time.sleep(cfg.sleep_between_calls_s)
            _save_payload(
                conn,
                endpoint="nfl_player_props",
                snapshot_role=cfg.snapshot_role,
                as_of_date=et_day,
                url=_full_url(props_url, props_params),
                payload=payload,
            )
            props_saved += 1
            time.sleep(cfg.sleep_between_calls_s)
    hard_failed = bool(event_rows) and game_odds_saved == 0 and props_saved == 0
    partial = bool(game_odds_error or prop_fetch_failures)
    return {
        "status": "failed" if hard_failed else "partial" if partial else "ok",
        "provider": "oddsapi",
        "game_date": et_day.isoformat(),
        "snapshot_role": cfg.snapshot_role,
        "events": len(event_rows),
        "game_odds_saved": game_odds_saved,
        "game_odds_error": game_odds_error,
        "player_prop_events_saved": props_saved,
        "player_prop_fetch_failures": len(prop_fetch_failures),
        "player_prop_failure_examples": prop_fetch_failures[:5],
        "player_prop_single_market_fallbacks": len(single_market_fallbacks),
        "player_prop_single_market_fallback_examples": single_market_fallbacks[:12],
    }


def _fetch_sportsgameodds_for_date(cfg: OddsCrawlerConfig, et_day: date) -> dict[str, Any]:
    if not cfg.sports_game_odds_key:
        return _skip_result("sportsgameodds", cfg, et_day, "SPORTSGAMEODDS_API_KEY is not set")
    start_utc, end_utc = _et_day_window_utc(et_day)
    url = "https://api.sportsgameodds.com/v2/events"
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    raw_pages = 0
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        while True:
            params = {
                "leagueID": "NFL",
                "oddsAvailable": "true",
                "bookmakerID": "fanduel,draftkings",
                "includeAltLines": "true" if cfg.snapshot_role == 'close' else "false",
                "includeOpposingOdds": "true",
                "startsAfter": _to_z(start_utc),
                "startsBefore": _to_z(end_utc),
                "limit": "100",
            }
            if cursor:
                params["cursor"] = cursor
            response = requests.get(
                url,
                params=params,
                headers={"x-api-key": cfg.sports_game_odds_key},
                timeout=cfg.timeout_s,
            )
            if not response.ok:
                message, error_code = _api_error_message(response)
                raise OddsApiError(message, status_code=response.status_code, error_code=error_code, retriable=False)
            payload = response.json()
            if isinstance(payload, dict) and payload.get("success") is False:
                raise OddsApiError(str(payload.get("error") or "SportsGameOdds request failed"), retriable=False)
            raw_pages += 1
            _save_payload(conn,endpoint='nfl_sportsgameodds_events_raw',snapshot_role=cfg.snapshot_role,
                          as_of_date=et_day,url=_full_url(url,params),payload=payload,provider='sportsgameodds')
            canonical = _canonical_events_from_sgo_payload(payload, include_alts=cfg.snapshot_role == 'close')
            events.extend(canonical)
            cursor_value = payload.get("nextCursor") if isinstance(payload, dict) else None
            cursor = str(cursor_value) if cursor_value else None
            if not cursor:
                break
            time.sleep(cfg.sleep_between_calls_s)
        game_saved = 0
        props_saved = 0
        if events and _has_market(events, {"spreads", "totals"}):
            _save_payload(
                conn,
                endpoint="nfl_game_odds",
                snapshot_role=cfg.snapshot_role,
                as_of_date=et_day,
                url=_full_url(url, {"leagueID": "NFL", "oddsAvailable": "true", "provider": "sportsgameodds"}),
                payload=events,
                provider="sportsgameodds",
            )
            game_saved = 1
        if events and _has_market(events, set(ODDS_API_MARKETS.split(","))):
            _save_payload(
                conn,
                endpoint="nfl_player_props",
                snapshot_role=cfg.snapshot_role,
                as_of_date=et_day,
                url=_full_url(url, {"leagueID": "NFL", "oddsAvailable": "true", "provider": "sportsgameodds", "props": "true"}),
                payload=events,
                provider="sportsgameodds",
            )
            props_saved = 1
    return {
        "status": "ok" if game_saved or props_saved else "partial",
        "provider": "sportsgameodds",
        "game_date": et_day.isoformat(),
        "snapshot_role": cfg.snapshot_role,
        "events": len(events),
        "raw_pages": raw_pages,
        "alternate_player_lines_requested": cfg.snapshot_role == 'close',
        "game_odds_saved": game_saved,
        "player_prop_events_saved": props_saved,
        "reason": None if game_saved or props_saved else "SportsGameOdds returned no normalized FanDuel/DraftKings NFL odds for this date",
    }


def _fetch_therundown_for_date(cfg: OddsCrawlerConfig, et_day: date) -> dict[str, Any]:
    if not cfg.therundown_key:
        return _skip_result("therundown", cfg, et_day, "THERUNDOWN_API_KEY is not set")
    # TheRundown's public V2 endpoint is useful as a cheap game-odds fallback.
    # Provider-specific market parsing can be expanded after we see a live payload.
    url = f"https://therundown.io/api/v2/sports/2/events/{et_day.isoformat()}"
    params = {
        "market_ids": "2,3",
        "affiliate_ids": "19,23",
        "main_line": "true",
        "hide_closed": "true",
        "include": "all_periods",
    }
    response = requests.get(
        url,
        params=params,
        headers={"X-TheRundown-Key": cfg.therundown_key},
        timeout=cfg.timeout_s,
    )
    if not response.ok:
        message, error_code = _api_error_message(response)
        raise OddsApiError(message, status_code=response.status_code, error_code=error_code, retriable=False)
    payload = response.json()
    events = payload.get("events") if isinstance(payload, dict) else None
    event_count = len(events) if isinstance(events, list) else 0
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        _save_payload(
            conn,
            endpoint="nfl_therundown_events_raw",
            snapshot_role=cfg.snapshot_role,
            as_of_date=et_day,
            url=_full_url(url, params),
            payload=payload,
            provider="therundown",
        )
    return {
        "status": "partial",
        "provider": "therundown",
        "game_date": et_day.isoformat(),
        "snapshot_role": cfg.snapshot_role,
        "events": event_count,
        "game_odds_saved": 0,
        "player_prop_events_saved": 0,
        "reason": "TheRundown raw payload saved; normalizer is waiting for a live sample before trusting game-line mapping",
    }


def _csv_files(base_dir: Path, stem: str, et_day: date) -> list[Path]:
    day = et_day.isoformat()
    return [
        path for path in (
            base_dir / f"{stem}_{day}.csv",
            base_dir / f"{stem}-{day}.csv",
            base_dir / f"{day}_{stem}.csv",
            base_dir / f"{day}-{stem}.csv",
        )
        if path.exists()
    ]


def _read_csv_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cleaned = {str(k or "").strip().lower(): str(v or "").strip() for k, v in row.items()}
                cleaned["_source_file"] = str(path)
                rows.append(cleaned)
    return rows


def _csv_value(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _parse_csv_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _manual_prop_rows(raw_rows: list[dict[str, str]], cfg: OddsCrawlerConfig, et_day: date) -> list[tuple]:
    grouped: dict[tuple, dict[str, Any]] = {}
    fetched_at = datetime.now(_UTC)
    for row in raw_rows:
        book = _book_key(_csv_value(row, "book", "bookmaker", "bookmaker_key", "sportsbook"))
        stat = _csv_value(row, "stat", "market", "market_key")
        player = _csv_value(row, "player_name", "player", "name")
        line = _clean_float(_csv_value(row, "line", "point", "points"))
        if not book or not stat or not player or line is None:
            continue
        market_key = {
            "passing_yards": "player_pass_yds",
            "player_pass_yds": "player_pass_yds",
            "rushing_yards": "player_rush_yds",
            "player_rush_yds": "player_rush_yds",
            "passing_tds": "player_pass_tds",
            "player_pass_tds": "player_pass_tds",
            "rushing_tds": "player_rush_tds",
            "player_rush_tds": "player_rush_tds",
            "receiving_yards": "player_reception_yds",
            "player_reception_yds": "player_reception_yds",
            "receptions": "player_receptions",
            "player_receptions": "player_receptions",
            "receiving_tds": "player_reception_tds",
            "player_reception_tds": "player_reception_tds",
        }.get(stat.lower())
        stat_norm = {
            "player_pass_yds": "passing_yards",
            "player_rush_yds": "rushing_yards",
            "player_pass_tds": "passing_tds",
            "player_rush_tds": "rushing_tds",
            "player_reception_yds": "receiving_yards",
            "player_receptions": "receptions",
            "player_reception_tds": "receiving_tds",
        }.get(market_key or "")
        if not market_key or not stat_norm:
            continue
        event_id = _csv_value(row, "event_id", "game_id") or f"manual_{et_day.isoformat()}_{normalize_team(_csv_value(row, 'away_team', 'away')) or 'AWAY'}_{normalize_team(_csv_value(row, 'home_team', 'home')) or 'HOME'}"
        key = (event_id, book, normalize_name(player), market_key, line)
        rec = grouped.setdefault(key, {
            "provider": "manual_csv",
            "as_of_date": et_day,
            "fetched_at_utc": _parse_csv_dt(_csv_value(row, "fetched_at_utc", "fetched_at")) or fetched_at,
            "snapshot_role": _csv_value(row, "snapshot_role", "role") or cfg.snapshot_role,
            "event_id": event_id,
            "commence_time_utc": _parse_csv_dt(_csv_value(row, "commence_time_utc", "start_ts_utc", "start_time")),
            "bookmaker_key": book,
            "bookmaker_title": _book_title(book),
            "home_team": _csv_value(row, "home_team", "home"),
            "away_team": _csv_value(row, "away_team", "away"),
            "player_name": player,
            "player_name_norm": normalize_name(player),
            "market_key": market_key,
            "stat": stat_norm,
            "line": line,
            "over_price": None,
            "under_price": None,
            "over_link": None,
            "under_link": None,
        })
        side = str(_csv_value(row, "side") or "").lower()
        if side in {"over", "under"}:
            rec[f"{side}_price"] = _clean_int(_csv_value(row, "price", f"{side}_price"))
            rec[f"{side}_link"] = _csv_value(row, "link", f"{side}_link")
        else:
            rec["over_price"] = _clean_int(_csv_value(row, "over_price", "over"))
            rec["under_price"] = _clean_int(_csv_value(row, "under_price", "under"))
            rec["over_link"] = _csv_value(row, "over_link")
            rec["under_link"] = _csv_value(row, "under_link")
    return [
        (
            rec["provider"], rec["as_of_date"], rec["fetched_at_utc"], rec["snapshot_role"],
            rec["event_id"], rec["commence_time_utc"], rec["bookmaker_key"], rec["bookmaker_title"],
            rec["home_team"], rec["away_team"], rec["player_name"], rec["player_name_norm"],
            rec["market_key"], rec["stat"], rec["line"], rec["over_price"], rec["under_price"],
            rec["over_link"], rec["under_link"],
        )
        for rec in grouped.values()
        if rec["over_price"] is not None or rec["under_price"] is not None
    ]


def _manual_game_rows(raw_rows: list[dict[str, str]], cfg: OddsCrawlerConfig, et_day: date) -> list[tuple]:
    rows: list[tuple] = []
    fetched_at = datetime.now(_UTC)
    for row in raw_rows:
        book = _book_key(_csv_value(row, "book", "bookmaker", "bookmaker_key", "sportsbook"))
        if not book:
            continue
        away_team = _csv_value(row, "away_team", "away")
        home_team = _csv_value(row, "home_team", "home")
        event_id = _csv_value(row, "event_id", "game_id") or f"manual_{et_day.isoformat()}_{normalize_team(away_team) or 'AWAY'}_{normalize_team(home_team) or 'HOME'}"
        rows.append((
            "manual_csv", et_day,
            _parse_csv_dt(_csv_value(row, "fetched_at_utc", "fetched_at")) or fetched_at,
            _csv_value(row, "snapshot_role", "role") or cfg.snapshot_role,
            event_id,
            _parse_csv_dt(_csv_value(row, "commence_time_utc", "start_ts_utc", "start_time")),
            book, _book_title(book), home_team, away_team,
            normalize_team(home_team), normalize_team(away_team),
            _clean_float(_csv_value(row, "spread_home_points", "home_spread")),
            _clean_int(_csv_value(row, "spread_home_price", "home_spread_price")),
            _clean_float(_csv_value(row, "spread_away_points", "away_spread")),
            _clean_int(_csv_value(row, "spread_away_price", "away_spread_price")),
            _clean_float(_csv_value(row, "total_points", "total")),
            _clean_int(_csv_value(row, "total_over_price", "over_price", "over")),
            _clean_int(_csv_value(row, "total_under_price", "under_price", "under")),
            _csv_value(row, "spread_home_link", "home_link"),
            _csv_value(row, "spread_away_link", "away_link"),
            _csv_value(row, "total_over_link", "over_link"),
            _csv_value(row, "total_under_link", "under_link"),
        ))
    return rows


def _insert_manual_props(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO odds.nfl_player_prop_lines (
                provider, as_of_date, fetched_at_utc, snapshot_role, event_id, commence_time_utc,
                bookmaker_key, bookmaker_title, home_team, away_team,
                player_name, player_name_norm, market_key, stat, line,
                over_price, under_price, over_link, under_link
            )
            VALUES %s
            ON CONFLICT (provider, fetched_at_utc, event_id, bookmaker_key, player_name_norm, market_key, line)
            DO UPDATE SET
                snapshot_role = EXCLUDED.snapshot_role,
                commence_time_utc = EXCLUDED.commence_time_utc,
                bookmaker_title = EXCLUDED.bookmaker_title,
                home_team = EXCLUDED.home_team,
                away_team = EXCLUDED.away_team,
                player_name = EXCLUDED.player_name,
                stat = EXCLUDED.stat,
                over_price = EXCLUDED.over_price,
                under_price = EXCLUDED.under_price,
                over_link = EXCLUDED.over_link,
                under_link = EXCLUDED.under_link,
                updated_at_utc = NOW()
            WHERE {PROP_LINE_CHANGED}
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    return len(rows)


def _insert_manual_games(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO odds.nfl_game_lines (
                provider, as_of_date, fetched_at_utc, snapshot_role, event_id, commence_time_utc,
                bookmaker_key, bookmaker_title, home_team, away_team,
                home_team_abbr, away_team_abbr,
                spread_home_points, spread_home_price,
                spread_away_points, spread_away_price,
                total_points, total_over_price, total_under_price,
                spread_home_link, spread_away_link, total_over_link, total_under_link
            )
            VALUES %s
            ON CONFLICT (provider, fetched_at_utc, event_id, bookmaker_key)
            DO UPDATE SET
                snapshot_role = EXCLUDED.snapshot_role,
                commence_time_utc = EXCLUDED.commence_time_utc,
                bookmaker_title = EXCLUDED.bookmaker_title,
                home_team = EXCLUDED.home_team,
                away_team = EXCLUDED.away_team,
                home_team_abbr = EXCLUDED.home_team_abbr,
                away_team_abbr = EXCLUDED.away_team_abbr,
                spread_home_points = EXCLUDED.spread_home_points,
                spread_home_price = EXCLUDED.spread_home_price,
                spread_away_points = EXCLUDED.spread_away_points,
                spread_away_price = EXCLUDED.spread_away_price,
                total_points = EXCLUDED.total_points,
                total_over_price = EXCLUDED.total_over_price,
                total_under_price = EXCLUDED.total_under_price,
                spread_home_link = EXCLUDED.spread_home_link,
                spread_away_link = EXCLUDED.spread_away_link,
                total_over_link = EXCLUDED.total_over_link,
                total_under_link = EXCLUDED.total_under_link,
                updated_at_utc = NOW()
            WHERE {GAME_LINE_CHANGED}
            """,
            rows,
            page_size=1000,
        )
    conn.commit()
    return len(rows)


def _import_manual_csv_for_date(cfg: OddsCrawlerConfig, et_day: date) -> dict[str, Any]:
    base_dir = Path(cfg.manual_csv_dir)
    if not base_dir.exists():
        return _skip_result("manual_csv", cfg, et_day, f"manual CSV directory not found: {base_dir}")
    prop_files = _csv_files(base_dir, "nfl_player_props", et_day) + _csv_files(base_dir, "player_props", et_day)
    game_files = _csv_files(base_dir, "nfl_game_odds", et_day) + _csv_files(base_dir, "game_odds", et_day)
    if not prop_files and not game_files:
        return _skip_result("manual_csv", cfg, et_day, f"no manual CSV files found in {base_dir}")
    prop_rows = _manual_prop_rows(_read_csv_rows(prop_files), cfg, et_day)
    game_rows = _manual_game_rows(_read_csv_rows(game_files), cfg, et_day)
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        prop_count = _insert_manual_props(conn, prop_rows)
        game_count = _insert_manual_games(conn, game_rows)
    return {
        "status": "ok" if prop_count or game_count else "partial",
        "provider": "manual_csv",
        "game_date": et_day.isoformat(),
        "snapshot_role": cfg.snapshot_role,
        "events": 0,
        "game_odds_saved": 1 if game_count else 0,
        "player_prop_events_saved": 1 if prop_count else 0,
        "manual_prop_rows_upserted": prop_count,
        "manual_game_rows_upserted": game_count,
        "manual_prop_files": [str(path) for path in prop_files],
        "manual_game_files": [str(path) for path in game_files],
        "reason": None if prop_count or game_count else "manual CSV files found but no valid odds rows were parsed",
    }


def fetch_for_date(cfg: OddsCrawlerConfig, et_day: date) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    game_saved = 0
    props_saved = 0
    provider_errors: list[dict[str, Any]] = []
    for provider in _provider_names(cfg.provider_order):
        try:
            if provider == "sportsgameodds":
                result = _fetch_sportsgameodds_for_date(cfg, et_day)
            elif provider == "therundown":
                result = _fetch_therundown_for_date(cfg, et_day)
            elif provider == "oddsapi":
                result = _fetch_the_odds_api_for_date(cfg, et_day)
            elif provider == "manual_csv":
                result = _import_manual_csv_for_date(cfg, et_day)
            else:
                result = _skip_result(provider, cfg, et_day, "unknown provider")
        except Exception as exc:
            result = {
                "status": "failed",
                "provider": provider,
                "game_date": et_day.isoformat(),
                "snapshot_role": cfg.snapshot_role,
                "events": 0,
                "game_odds_saved": 0,
                "player_prop_events_saved": 0,
                "reason": str(exc),
            }
            provider_errors.append({"provider": provider, "error": str(exc)})
        attempts.append(result)
        game_saved += int(result.get("game_odds_saved") or 0)
        props_saved += int(result.get("player_prop_events_saved") or 0)
        if game_saved > 0 and props_saved > 0:
            break
    status = "ok" if game_saved > 0 and props_saved > 0 else "partial" if game_saved > 0 or props_saved > 0 else "failed"
    return {
        "status": status,
        "provider": "fallback_chain",
        "game_date": et_day.isoformat(),
        "snapshot_role": cfg.snapshot_role,
        "game_odds_saved": game_saved,
        "player_prop_events_saved": props_saved,
        "provider_order": _provider_names(cfg.provider_order),
        "provider_attempts": [_provider_attempt_summary(result) for result in attempts],
        "provider_errors": provider_errors,
        "manual_csv_dir": str(Path(cfg.manual_csv_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NFL player props from The Odds API")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--sport", default="americanfootball_nfl")
    parser.add_argument("--bookmakers", default="fanduel,draftkings")
    parser.add_argument("--snapshot-role", default="live", choices=("live", "open", "lock", "close"))
    parser.add_argument("--no-single-market-fallback", action="store_true")
    parser.add_argument("--providers", default=None, help="Comma-separated provider order, e.g. sportsgameodds,therundown,oddsapi,manual_csv")
    parser.add_argument("--manual-csv-dir", default=None, help="Directory containing manual NFL odds CSV files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    et_day = date.fromisoformat(args.date) if args.date else datetime.now(_ET).date()
    cfg = OddsCrawlerConfig(
        pg_dsn=args.pg_dsn,
        sport=args.sport,
        bookmakers=args.bookmakers,
        snapshot_role=args.snapshot_role,
        single_market_fallback_on_empty=not args.no_single_market_fallback,
        provider_order=args.providers or os.getenv("NFL_ODDS_PROVIDER_ORDER", "sportsgameodds,therundown,oddsapi,manual_csv"),
        manual_csv_dir=args.manual_csv_dir or os.getenv("NFL_MANUAL_ODDS_CSV_DIR", str(_DEFAULT_MANUAL_CSV_DIR)),
    )
    result = fetch_for_date(cfg, et_day)
    print(json.dumps(result, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
