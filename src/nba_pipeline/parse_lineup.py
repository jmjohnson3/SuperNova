from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("nba_pipeline.parse_lineup")


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


def _lower_abbr(x: Any) -> Optional[str]:
    if not x:
        return None
    return str(x).strip().lower()


def _pick_team_blocks(payload: dict) -> list[dict]:
    """
    Try common shapes:
      - payload['lineups'] or payload['teamLineups'] or payload['teams']
      - sometimes under payload['lineup'] as object
    Returns list of team blocks.
    """
    for k in ("lineups", "teamLineups", "teams"):
        v = payload.get(k)
        if isinstance(v, list):
            return v
    v = payload.get("lineup")
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        # may contain home/away
        blocks = []
        for kk in ("home", "away"):
            if isinstance(v.get(kk), dict):
                blocks.append(v[kk])
        return blocks
    return []


def _extract_lists(team_block: dict) -> tuple[list, list, list]:
    """
    Heuristic extraction. We keep raw_json anyway.
    """
    starters = team_block.get("starters") or team_block.get("startingLineup") or []
    bench = team_block.get("bench") or team_block.get("reserves") or []
    scratches = team_block.get("scratches") or team_block.get("inactive") or team_block.get("out") or []
    if not isinstance(starters, list):
        starters = []
    if not isinstance(bench, list):
        bench = []
    if not isinstance(scratches, list):
        scratches = []
    return starters, bench, scratches


def parse_all_lineups(conn, *, commit_every: int = 500) -> int:
    q = """
    SELECT season, game_slug, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='lineup'
      AND season IS NOT NULL
      AND game_slug IS NOT NULL
    ORDER BY fetched_at_utc ASC
    """

    batch: list[dict] = []
    processed = 0
    written = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        for r in cur:
            season = r["season"]
            game_slug = r["game_slug"]
            fetched_at = r["fetched_at_utc"]
            payload = _ensure_obj(r["payload"])

            team_blocks = _pick_team_blocks(payload)
            if not team_blocks:
                # still write a placeholder? usually not needed
                processed += 1
                continue

            for tb in team_blocks:
                team = tb.get("team") or tb.get("currentTeam") or tb.get("homeTeam") or {}
                team_abbr = _lower_abbr(team.get("abbreviation") or tb.get("teamAbbr") or tb.get("abbreviation"))
                if not team_abbr:
                    continue

                # home/away flag if present
                is_home = tb.get("isHome")
                if is_home is None:
                    side = (tb.get("side") or tb.get("homeOrAway") or "").lower()
                    if side in ("home", "h"):
                        is_home = True
                    elif side in ("away", "a"):
                        is_home = False

                starters, bench, scratches = _extract_lists(tb)

                batch.append(
                    {
                        "season": season,
                        "game_slug": game_slug,
                        "team_abbr": team_abbr,
                        "is_home": is_home,
                        "status": tb.get("status") or payload.get("status"),
                        "starters": json.dumps(starters, ensure_ascii=False),
                        "bench": json.dumps(bench, ensure_ascii=False),
                        "scratches": json.dumps(scratches, ensure_ascii=False),
                        "raw_json": json.dumps(tb, ensure_ascii=False),
                        "source_fetched_at_utc": fetched_at,
                    }
                )

            processed += 1

            if processed % commit_every == 0:
                written += _flush(conn, batch)
                conn.commit()
                batch.clear()
                log.info("Committed lineups: payloads=%d rows=%d", processed, written)

    if batch:
        written += _flush(conn, batch)
        conn.commit()

    log.info("Done lineups: payloads=%d rows=%d", processed, written)
    return written


def _flush(conn, rows: list[dict]) -> int:
    if not rows:
        return 0

    # Dedup within batch (season, game_slug, team_abbr)
    dedup: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        key = (r["season"], r["game_slug"], r["team_abbr"])
        dedup[key] = r
    rows = list(dedup.values())

    sql = """
    INSERT INTO raw.nba_game_lineups (
      season, game_slug, team_abbr, is_home,
      status, starters, bench, scratches, raw_json,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (season, game_slug, team_abbr) DO UPDATE SET
      is_home = EXCLUDED.is_home,
      status = EXCLUDED.status,
      starters = EXCLUDED.starters,
      bench = EXCLUDED.bench,
      scratches = EXCLUDED.scratches,
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
              %(season)s, %(game_slug)s, %(team_abbr)s, %(is_home)s,
              %(status)s, %(starters)s::jsonb, %(bench)s::jsonb, %(scratches)s::jsonb, %(raw_json)s::jsonb,
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
        parse_all_lineups(conn)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
