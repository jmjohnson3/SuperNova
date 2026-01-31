from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    _ET = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError as e:
    raise RuntimeError("tzdata missing. On Windows: pip install tzdata") from e


log = logging.getLogger("nba_pipeline.parse_player_gamelogs")

def _derive_game_slug_from_gamelog(gl: dict) -> Optional[str]:
    game = gl.get("game") or {}
    start_time = game.get("startTime")
    away = game.get("awayTeamAbbreviation")
    home = game.get("homeTeamAbbreviation")
    if not (start_time and away and home):
        return None

    dt_utc = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    dt_et = dt_utc.astimezone(_ET)
    game_date = dt_et.strftime("%Y%m%d")

    return f"{game_date}-{str(away).upper()}-{str(home).upper()}"


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    try:
        return int(str(x))
    except Exception:
        return None


def _as_num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(str(x))
    except Exception:
        return None


def _dig(d: dict, *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _unwrap_total(v: Any) -> Any:
    # common MSF shape: {"total": 12} or direct number
    if isinstance(v, dict):
        for k in ("total", "value", "amount"):
            if k in v:
                return v.get(k)
    return v


def _extract_core_stats(stats_obj: dict) -> dict:
    misc = stats_obj.get("miscellaneous") or {}
    off = stats_obj.get("offense") or {}
    reb = stats_obj.get("rebounds") or {}
    fg = stats_obj.get("fieldGoals") or {}
    ft = stats_obj.get("freeThrows") or {}

    min_seconds = misc.get("minSeconds")

    return {
        "minutes": (_as_num(min_seconds) / 60.0) if _as_num(min_seconds) is not None else None,
        "points": _as_int(off.get("pts")),
        "rebounds": _as_int(reb.get("reb")),
        "assists": _as_int(off.get("ast")),
        "threes_made": _as_int(fg.get("fg3PtMade")),
        "fga": _as_int(fg.get("fgAtt")),
        "fta": _as_int(ft.get("ftAtt")),
    }


def build_rows_from_payload(*, season: str, as_of_date, fetched_at_utc, payload: dict) -> list[dict]:
    rows: list[dict] = []
    gamelogs = payload.get("gamelogs") or payload.get("playerGamelogs") or []

    for gl in gamelogs:
        # Many MSF gamelog items look like:
        # { "game": {...}, "player": {...}, "team": {...}, "stats": {... or [..]} }
        game = gl.get("game") or {}
        player = gl.get("player") or {}
        team = gl.get("team") or gl.get("currentTeam") or {}

        # stats might be dict or list-of-dicts
        stats = gl.get("stats") or gl.get("playerStats") or {}
        if isinstance(stats, list):
            stats_obj = stats[0] if stats else {}
        else:
            stats_obj = stats or {}

        pid = _as_int(player.get("id"))
        team_abbr = (team.get("abbreviation") or "").lower()

        # derive opponent/home from game block if present
        away = game.get("awayTeamAbbreviation")
        home = game.get("homeTeamAbbreviation")
        away_u = (away or "").upper()
        home_u = (home or "").upper()
        team_u = team_abbr.upper()

        is_home = None
        opp = None
        if team_u and home_u and away_u:
            if team_u == home_u:
                is_home = True
                opp = away_u
            elif team_u == away_u:
                is_home = False
                opp = home_u

        # prefer game.slug if present, else api_responses.game_slug will be used by SQL query join
        # but we’ll require game_slug from api_responses row, so keep placeholder here.
        # (We’ll set it in the outer loop)
        if pid is None or not team_abbr:
            continue

        core = _extract_core_stats(stats_obj)

        derived_slug = _derive_game_slug_from_gamelog(gl)

        rows.append(
            {
                "season": season,
                "as_of_date": as_of_date,
                "game_slug": derived_slug,
                "player_id": pid,
                "team_abbr": team_abbr,
                "opponent_abbr": opp.lower() if isinstance(opp, str) else None,
                "is_home": is_home,
                "start_ts_utc": _parse_iso(game.get("startTime")),
                "status": game.get("playedStatus") or game.get("scheduleStatus"),
                "minutes": core["minutes"],
                "points": core["points"],
                "rebounds": core["rebounds"],
                "assists": core["assists"],
                "threes_made": core["threes_made"],
                "fga": core["fga"],
                "fta": core["fta"],
                "stats": json.dumps(stats_obj, ensure_ascii=False),
                "raw_json": json.dumps(gl, ensure_ascii=False),
                "source_fetched_at_utc": fetched_at_utc,
            }
        )

    return rows


def parse_all_player_gamelogs(conn, *, commit_every: int = 250) -> int:
    """
    Reads raw.api_responses where endpoint='player_gamelogs' and upserts.
    """
    q = """
    SELECT season, as_of_date, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='player_gamelogs'
      AND season IS NOT NULL
      AND payload IS NOT NULL
    ORDER BY fetched_at_utc ASC
    """

    upserts: list[dict] = []
    processed_payloads = 0
    total_rows = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        for r in cur:
            season = r["season"]
            game_slug = r.get("game_slug")
            as_of_date = r.get("as_of_date")
            fetched_at = r["fetched_at_utc"]
            payload = _ensure_obj(r["payload"])

            rows = build_rows_from_payload(
                season=season,
                as_of_date=as_of_date,
                fetched_at_utc=fetched_at,
                payload=payload,
            )

            # game_slug is derived per gamelog row
            rows = [x for x in rows if x.get("game_slug")]
            upserts.extend(rows)
            processed_payloads += 1

            if processed_payloads % commit_every == 0:
                total_rows += _flush_player_gamelogs(conn, upserts)
                conn.commit()
                upserts.clear()
                log.info("Committed player_gamelogs: payloads=%d total_rows=%d", processed_payloads, total_rows)

    if upserts:
        total_rows += _flush_player_gamelogs(conn, upserts)
        conn.commit()

    log.info("Done player_gamelogs: payloads=%d rows=%d", processed_payloads, total_rows)
    return total_rows


def _flush_player_gamelogs(conn, rows: list[dict]) -> int:
    if not rows:
        return 0

    # Deduplicate within batch (season, game_slug, player_id)
    dedup: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        key = (r["season"], r["game_slug"], int(r["player_id"]))
        dedup[key] = r
    rows = list(dedup.values())

    sql = """
    INSERT INTO raw.nba_player_gamelogs (
      season, as_of_date, game_slug,
      player_id, team_abbr, opponent_abbr, is_home,
      start_ts_utc, status,
      minutes, points, rebounds, assists, threes_made, fga, fta,
      stats, raw_json,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (season, game_slug, player_id) DO UPDATE SET
      as_of_date = EXCLUDED.as_of_date,
      team_abbr = EXCLUDED.team_abbr,
      opponent_abbr = EXCLUDED.opponent_abbr,
      is_home = EXCLUDED.is_home,
      start_ts_utc = EXCLUDED.start_ts_utc,
      status = EXCLUDED.status,
      minutes = EXCLUDED.minutes,
      points = EXCLUDED.points,
      rebounds = EXCLUDED.rebounds,
      assists = EXCLUDED.assists,
      threes_made = EXCLUDED.threes_made,
      fga = EXCLUDED.fga,
      fta = EXCLUDED.fta,
      stats = EXCLUDED.stats,
      raw_json = EXCLUDED.raw_json,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template="""
            (
              %(season)s, %(as_of_date)s, %(game_slug)s,
              %(player_id)s, %(team_abbr)s, %(opponent_abbr)s, %(is_home)s,
              %(start_ts_utc)s, %(status)s,
              %(minutes)s, %(points)s, %(rebounds)s, %(assists)s, %(threes_made)s, %(fga)s, %(fta)s,
              %(stats)s::jsonb, %(raw_json)s::jsonb,
              %(source_fetched_at_utc)s, now()
            )
            """,
            page_size=1000,
        )

    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    dsn = "postgresql://josh:password@localhost:5432/nba"
    if not dsn:
        raise RuntimeError("Missing PG_DSN")
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        parse_all_player_gamelogs(conn)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
