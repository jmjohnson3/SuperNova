"""
Parse referee/official assignments from boxscore API responses.

The MySportsFeeds boxscore payload contains official data that we were
previously ignoring. This parser extracts it into raw.nba_game_referees
so the referee feature views (V013) can compute foul tendencies.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("nba_pipeline.parse_referees")


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            return None
    return None


def _extract_officials(payload: dict) -> list[dict]:
    """
    Try multiple JSON paths to find the officials array in the
    MySportsFeeds boxscore response.

    Known structures:
      1. payload -> game -> officials -> []
      2. payload -> references -> officials -> []
      3. payload -> officials -> []
      4. payload -> game -> officials -> official -> []
    """
    candidates = [
        payload.get("game", {}).get("officials"),
        payload.get("references", {}).get("officials"),
        payload.get("officials"),
        payload.get("game", {}).get("officials", {}).get("official")
        if isinstance(payload.get("game", {}).get("officials"), dict)
        else None,
    ]
    for c in candidates:
        if isinstance(c, list) and len(c) > 0:
            return c
    return []


def _parse_one_official(official: dict) -> Optional[dict]:
    """
    Extract a single official record. Handle both nested and flat formats.

    Nested:  {"official": {"id": 123, "firstName": "Tony", ...}, "title": "Referee"}
    Flat:    {"id": 123, "firstName": "Tony", "lastName": "Brothers", "title": "Referee"}
    """
    # Try nested format
    inner = official.get("official") or official.get("person") or official
    ref_id = _as_int(inner.get("id") or inner.get("officialId"))
    if ref_id is None:
        return None

    first_name = inner.get("firstName") or inner.get("first_name") or ""
    last_name = inner.get("lastName") or inner.get("last_name") or ""
    title = (
        official.get("title")
        or official.get("position")
        or inner.get("title")
        or inner.get("position")
        or ""
    )

    return {
        "referee_id": ref_id,
        "first_name": first_name,
        "last_name": last_name,
        "title": title,
    }


def upsert_game_referees(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.nba_game_referees (
        game_slug, season, referee_id,
        first_name, last_name, title,
        source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug, referee_id) DO UPDATE SET
        season              = EXCLUDED.season,
        first_name          = EXCLUDED.first_name,
        last_name           = EXCLUDED.last_name,
        title               = EXCLUDED.title,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
        updated_at_utc      = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""
        (
            %(game_slug)s, %(season)s, %(referee_id)s,
            %(first_name)s, %(last_name)s, %(title)s,
            %(source_fetched_at_utc)s, now()
        )
        """,
        page_size=500,
    )


def build_raw_referees(conn, *, commit_every: int = 500) -> None:
    """
    Parse all boxscore payloads in raw.api_responses and extract
    referee assignments into raw.nba_game_referees.
    """
    q = """
    SELECT DISTINCT ON (season, game_slug)
           season, game_slug, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'boxscore'
      AND game_slug IS NOT NULL
    ORDER BY season, game_slug, fetched_at_utc DESC
    """

    batch: list[dict] = []
    processed = 0
    refs_found = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        for r in cur:
            season = r["season"]
            game_slug = r["game_slug"]
            fetched_at = r["fetched_at_utc"]
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            officials = _extract_officials(payload)
            for off in officials:
                parsed = _parse_one_official(off)
                if parsed is None:
                    continue
                parsed["game_slug"] = game_slug
                parsed["season"] = season
                parsed["source_fetched_at_utc"] = fetched_at
                batch.append(parsed)
                refs_found += 1

            processed += 1

            if processed % commit_every == 0:
                # Dedup within batch
                seen = set()
                deduped = []
                for row in batch:
                    key = (row["game_slug"], row["referee_id"])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(row)
                log.info(
                    "Upserting batch processed=%d refs_in_batch=%d",
                    processed,
                    len(deduped),
                )
                upsert_game_referees(conn, deduped)
                conn.commit()
                batch.clear()

    # Final flush
    if batch:
        seen = set()
        deduped = []
        for row in batch:
            key = (row["game_slug"], row["referee_id"])
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        log.info(
            "Final upsert processed=%d refs_in_batch=%d",
            processed,
            len(deduped),
        )
        upsert_game_referees(conn, deduped)
        conn.commit()

    log.info(
        "Done. Games processed=%d, referee rows=%d",
        processed,
        refs_found,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dsn = "postgresql://josh:password@localhost:5432/nba"
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        build_raw_referees(conn)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
