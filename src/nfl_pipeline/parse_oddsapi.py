"""Parse NFL player prop odds from raw Odds API responses."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import STAT_BY_MARKET, normalize_name, normalize_team
from nfl_pipeline.schema import ensure_schema

log = logging.getLogger("nfl_pipeline.parse_oddsapi")
_ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTIC = ROOT / "reports" / "nfl_prop_parse_diagnostic_latest.md"


@dataclass(frozen=True)
class ParseConfig:
    pg_dsn: str = PG_DSN
    as_of_date: date | None = None
    diagnostic_file: Path = DEFAULT_DIAGNOSTIC


def _parse_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    return {}


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _payload_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "events"):
        nested = payload.get(key)
        if isinstance(nested, list):
            return [event for event in nested if isinstance(event, dict)]
        if isinstance(nested, dict):
            return [nested]
    if isinstance(payload.get("bookmakers"), list) or payload.get("id"):
        return [payload]
    return []


def _outcome_side_and_player(outcome: dict) -> tuple[str | None, str | None]:
    name = str(outcome.get("name") or "").strip()
    desc = str(outcome.get("description") or outcome.get("participant") or outcome.get("player") or outcome.get("player_name") or "").strip()
    lower = name.lower()
    if lower in {"over", "under"}:
        return lower, desc or None
    if lower.startswith("over"):
        return "over", desc or None
    if lower.startswith("under"):
        return "under", desc or None
    desc_lower = desc.lower()
    if desc_lower in {"over", "under"}:
        return desc_lower, name or None
    if desc_lower.startswith("over"):
        return "over", name or None
    if desc_lower.startswith("under"):
        return "under", name or None
    return None, desc or name or None


def _rows_from_event(
    snapshot_role: str,
    as_of_date: date,
    fetched_at_utc: datetime,
    event: dict,
    diagnostics: dict[str, Any],
    provider: str = "oddsapi",
) -> list[tuple]:
    event_id = event.get("id")
    commence = _parse_dt(event.get("commence_time"))
    home = event.get("home_team")
    away = event.get("away_team")
    grouped: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    books = event.get("bookmakers", []) or []
    diagnostics["events_seen"] += 1
    diagnostics["bookmaker_entries"] += len(books)
    if not books:
        diagnostics["events_with_zero_books"] += 1
    for book in books:
        book_key = book.get("key")
        book_title = book.get("title")
        for market in book.get("markets", []) or []:
            market_key = str(market.get("key") or "")
            diagnostics["market_entries"] += 1
            stat = STAT_BY_MARKET.get(market_key)
            if not stat:
                diagnostics["unknown_markets"][market_key or "missing_market_key"] = diagnostics["unknown_markets"].get(market_key or "missing_market_key", 0) + 1
                continue
            for outcome in market.get("outcomes", []) or []:
                diagnostics["outcomes_seen"] += 1
                side, player = _outcome_side_and_player(outcome)
                line = _to_float(outcome.get("point"))
                if side not in {"over", "under"}:
                    diagnostics["skipped_outcomes"]["missing_side"] = diagnostics["skipped_outcomes"].get("missing_side", 0) + 1
                    continue
                if not player:
                    diagnostics["skipped_outcomes"]["missing_player"] = diagnostics["skipped_outcomes"].get("missing_player", 0) + 1
                    continue
                if line is None:
                    diagnostics["skipped_outcomes"]["missing_line"] = diagnostics["skipped_outcomes"].get("missing_line", 0) + 1
                    continue
                key = (str(book_key or ""), normalize_name(player), market_key, float(line))
                rec = grouped.setdefault(key, {
                    "provider": provider,
                    "snapshot_role": snapshot_role,
                    "as_of_date": as_of_date,
                    "fetched_at_utc": fetched_at_utc,
                    "event_id": event_id,
                    "commence_time_utc": commence,
                    "bookmaker_key": book_key,
                    "bookmaker_title": book_title,
                    "home_team": home,
                    "away_team": away,
                    "player_name": player,
                    "player_name_norm": normalize_name(player),
                    "market_key": market_key,
                    "stat": stat,
                    "line": line,
                    "over_price": None,
                    "under_price": None,
                    "over_link": None,
                    "under_link": None,
                })
                rec[f"{side}_price"] = _to_int(outcome.get("price"))
                rec[f"{side}_link"] = outcome.get("link")
    return [
        (
            rec["provider"], rec["as_of_date"], rec["fetched_at_utc"],
            rec["snapshot_role"],
            rec["event_id"], rec["commence_time_utc"], rec["bookmaker_key"],
            rec["bookmaker_title"], rec["home_team"], rec["away_team"],
            rec["player_name"], rec["player_name_norm"], rec["market_key"],
            rec["stat"], rec["line"], rec["over_price"], rec["under_price"],
            rec["over_link"], rec["under_link"],
        )
        for rec in grouped.values()
    ]


def _match_team_outcome(outcome_name: str | None, home: str | None, away: str | None) -> str | None:
    team = normalize_team(outcome_name)
    home_norm = normalize_team(home)
    away_norm = normalize_team(away)
    if team and home_norm and team == home_norm:
        return "home"
    if team and away_norm and team == away_norm:
        return "away"
    return None


def _rows_from_game_odds_payload(
    snapshot_role: str,
    as_of_date: date,
    fetched_at_utc: datetime,
    payload: Any,
    provider: str = "oddsapi",
) -> list[tuple]:
    events = _payload_events(payload)
    rows: list[tuple] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = event.get("id")
        commence = _parse_dt(event.get("commence_time"))
        home = event.get("home_team")
        away = event.get("away_team")
        home_abbr = normalize_team(home)
        away_abbr = normalize_team(away)
        for book in event.get("bookmakers", []) or []:
            rec: dict[str, Any] = {
                "provider": provider,
                "snapshot_role": snapshot_role,
                "as_of_date": as_of_date,
                "fetched_at_utc": fetched_at_utc,
                "event_id": event_id,
                "commence_time_utc": commence,
                "bookmaker_key": book.get("key"),
                "bookmaker_title": book.get("title"),
                "home_team": home,
                "away_team": away,
                "home_team_abbr": home_abbr,
                "away_team_abbr": away_abbr,
                "spread_home_points": None,
                "spread_home_price": None,
                "spread_away_points": None,
                "spread_away_price": None,
                "total_points": None,
                "total_over_price": None,
                "total_under_price": None,
                "spread_home_link": None,
                "spread_away_link": None,
                "total_over_link": None,
                "total_under_link": None,
            }
            has_market = False
            for market in book.get("markets", []) or []:
                market_key = str(market.get("key") or "").lower()
                if market_key == "spreads":
                    for outcome in market.get("outcomes", []) or []:
                        side = _match_team_outcome(outcome.get("name"), home, away)
                        if side not in {"home", "away"}:
                            continue
                        rec[f"spread_{side}_points"] = _to_float(outcome.get("point"))
                        rec[f"spread_{side}_price"] = _to_int(outcome.get("price"))
                        rec[f"spread_{side}_link"] = outcome.get("link")
                        has_market = True
                elif market_key == "totals":
                    for outcome in market.get("outcomes", []) or []:
                        side = str(outcome.get("name") or "").strip().lower()
                        if side not in {"over", "under"}:
                            continue
                        rec["total_points"] = _to_float(outcome.get("point"))
                        rec[f"total_{side}_price"] = _to_int(outcome.get("price"))
                        rec[f"total_{side}_link"] = outcome.get("link")
                        has_market = True
            if has_market:
                rows.append((
                    rec["provider"], rec["as_of_date"], rec["fetched_at_utc"],
                    rec["snapshot_role"],
                    rec["event_id"], rec["commence_time_utc"], rec["bookmaker_key"],
                    rec["bookmaker_title"], rec["home_team"], rec["away_team"],
                    rec["home_team_abbr"], rec["away_team_abbr"],
                    rec["spread_home_points"], rec["spread_home_price"],
                    rec["spread_away_points"], rec["spread_away_price"],
                    rec["total_points"], rec["total_over_price"], rec["total_under_price"],
                    rec["spread_home_link"], rec["spread_away_link"],
                    rec["total_over_link"], rec["total_under_link"],
                ))
    return rows


def parse_props(cfg: ParseConfig) -> dict[str, Any]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT provider, snapshot_role, as_of_date, fetched_at_utc, payload
                FROM raw.nfl_api_responses
                WHERE endpoint = 'nfl_player_props'
                  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
                UNION ALL
                SELECT provider, 'legacy'::text AS snapshot_role, as_of_date, fetched_at_utc, payload
                FROM raw.api_responses
                WHERE endpoint = 'nfl_player_props'
                  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
                ORDER BY fetched_at_utc
                """,
                {"as_of_date": cfg.as_of_date},
            )
            raw_rows = cur.fetchall()
            cur.execute(
                """
                SELECT provider, snapshot_role, as_of_date, fetched_at_utc, payload
                FROM raw.nfl_api_responses
                WHERE endpoint = 'nfl_game_odds'
                  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
                UNION ALL
                SELECT provider, 'legacy'::text AS snapshot_role, as_of_date, fetched_at_utc, payload
                FROM raw.api_responses
                WHERE endpoint = 'nfl_game_odds'
                  AND (%(as_of_date)s IS NULL OR as_of_date = %(as_of_date)s)
                ORDER BY fetched_at_utc
                """,
                {"as_of_date": cfg.as_of_date},
            )
            raw_game_rows = cur.fetchall()
        rows: list[tuple] = []
        diagnostics: dict[str, Any] = {
            "raw_payloads": len(raw_rows),
            "events_seen": 0,
            "events_with_zero_books": 0,
            "bookmaker_entries": 0,
            "market_entries": 0,
            "outcomes_seen": 0,
            "unknown_markets": {},
            "skipped_outcomes": {},
        }
        for raw in raw_rows:
            payload = _parse_payload(raw["payload"])
            for event in _payload_events(payload):
                rows.extend(_rows_from_event(
                    raw["snapshot_role"],
                    raw["as_of_date"],
                    raw["fetched_at_utc"],
                    event,
                    diagnostics,
                    provider=raw["provider"],
                ))
        game_rows: list[tuple] = []
        for raw in raw_game_rows:
            payload = _parse_payload(raw["payload"])
            game_rows.extend(_rows_from_game_odds_payload(
                raw["snapshot_role"],
                raw["as_of_date"],
                raw["fetched_at_utc"],
                payload,
                provider=raw["provider"],
            ))
        if rows:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO odds.nfl_player_prop_lines (
                        provider, as_of_date, fetched_at_utc, snapshot_role, event_id, commence_time_utc,
                        bookmaker_key, bookmaker_title, home_team, away_team,
                        player_name, player_name_norm, market_key, stat, line,
                        over_price, under_price, over_link, under_link
                    )
                    VALUES %s
                    ON CONFLICT (provider, fetched_at_utc, event_id, bookmaker_key, player_name_norm, market_key, line)
                    DO UPDATE SET
                        commence_time_utc = EXCLUDED.commence_time_utc,
                        snapshot_role = EXCLUDED.snapshot_role,
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
                    """,
                    rows,
                    page_size=1000,
                )
            conn.commit()
        if game_rows:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
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
                        commence_time_utc = EXCLUDED.commence_time_utc,
                        snapshot_role = EXCLUDED.snapshot_role,
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
                    """,
                    game_rows,
                    page_size=1000,
                )
            conn.commit()
    result = {
        "status": "ok",
        "raw_payloads": len(raw_rows),
        "raw_game_payloads": len(raw_game_rows),
        "prop_lines_upserted": len(rows),
        "game_lines_upserted": len(game_rows),
        "prop_parse_diagnostic": diagnostics,
    }
    _write_diagnostic(result, cfg.diagnostic_file)
    return result


def _write_diagnostic(payload: dict[str, Any], path: Path) -> str:
    diag = payload.get("prop_parse_diagnostic") or {}
    lines = [
        "# NFL Prop Parse Diagnostic",
        "",
        f"- Raw prop payloads: {payload.get('raw_payloads', 0)}",
        f"- Events seen: {diag.get('events_seen', 0)}",
        f"- Events with zero books: {diag.get('events_with_zero_books', 0)}",
        f"- Bookmaker entries: {diag.get('bookmaker_entries', 0)}",
        f"- Market entries: {diag.get('market_entries', 0)}",
        f"- Outcomes seen: {diag.get('outcomes_seen', 0)}",
        f"- Prop rows upserted: {payload.get('prop_lines_upserted', 0)}",
        "",
        "## Unknown Markets",
        "",
        "| Market | Count |",
        "|---|---:|",
    ]
    unknown = diag.get("unknown_markets") or {}
    if unknown:
        for market, count in sorted(unknown.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"| {market} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend([
        "",
        "## Skipped Outcomes",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ])
    skipped = diag.get("skipped_outcomes") or {}
    if skipped:
        for reason, count in sorted(skipped.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NFL player prop Odds API responses")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--diagnostic-file", default=str(DEFAULT_DIAGNOSTIC))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = ParseConfig(
        pg_dsn=args.pg_dsn,
        as_of_date=date.fromisoformat(args.date) if args.date else None,
        diagnostic_file=Path(args.diagnostic_file),
    )
    print(json.dumps(parse_props(cfg), indent=2))


if __name__ == "__main__":
    main()
