from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("nba_pipeline.parse_pbp")


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return int(float(str(x)))  # handles "1.0" from JSON floats
    except Exception:
        return None


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def parse_all_pbp(conn, *, commit_every_games: int = 50) -> int:
    q = """
    SELECT season, game_slug, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='playbyplay'
      AND season IS NOT NULL
      AND game_slug IS NOT NULL
    ORDER BY fetched_at_utc ASC
    """

    processed_games = 0
    total_rows = 0
    batch: list[dict] = []

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        for r in cur:
            season = r["season"]
            game_slug = r["game_slug"]
            fetched_at = r["fetched_at_utc"]
            payload = _ensure_obj(r["payload"])

            plays = payload.get("plays") or payload.get("gamePlays") or []
            if not isinstance(plays, list):
                plays = []

            # MSF PBP event keys (one per play, beside playStatus/description)
            _MSF_EVENT_KEYS = {
                "fieldGoalAttempt", "freeThrowAttempt", "rebound",
                "foul", "turnover", "substitution", "violation", "jumpBall",
                "timeout", "flagrantFoul", "technicalFoul",
            }

            for seq, p in enumerate(plays, start=1):
                # Try common keys; keep raw_json always
                play_id = _safe_str(p.get("playId") or p.get("id") or f"{seq}")

                # MSF format: period is at playStatus.quarter
                play_status = p.get("playStatus") or {}
                period = _as_int(
                    play_status.get("quarter")
                    or play_status.get("period")
                    or p.get("period")
                    or p.get("periodNumber")
                    or p.get("quarter")
                )

                # clock: store secondsElapsed as string, fallback to other keys
                secs = play_status.get("secondsElapsed")
                clock = _safe_str(secs) if secs is not None else _safe_str(
                    p.get("timeRemaining") or p.get("clock") or p.get("time")
                )

                # event_type: the MSF top-level event key (e.g. "fieldGoalAttempt")
                event_type = None
                event_block = None
                for ek in _MSF_EVENT_KEYS:
                    if ek in p:
                        event_type = ek
                        event_block = p[ek]
                        break
                if event_type is None:
                    # fallback for other formats
                    event_type = _safe_str(p.get("type") or p.get("playType") or p.get("eventType"))

                desc = _safe_str(p.get("description") or p.get("text") or p.get("summary"))

                # team_abbr: inside the event block
                team_abbr = None
                if isinstance(event_block, dict):
                    team = event_block.get("team") or {}
                    team_abbr = _safe_str(team.get("abbreviation") if isinstance(team, dict) else None)
                if not team_abbr:
                    team = p.get("team") or {}
                    team_abbr = _safe_str(team.get("abbreviation") if isinstance(team, dict) else None)
                if team_abbr:
                    team_abbr = team_abbr.lower()

                # score: MSF PBP doesn't include running score per play
                score = p.get("score") or {}
                points_home = _as_int(score.get("homeScoreTotal") or score.get("home"))
                points_away = _as_int(score.get("awayScoreTotal") or score.get("away"))

                batch.append(
                    {
                        "season": season,
                        "game_slug": game_slug,
                        "play_id": play_id,
                        "play_seq": seq,
                        "period": period,
                        "clock": clock,
                        "event_type": event_type,
                        "team_abbr": team_abbr,
                        "description": desc,
                        "points_home": points_home,
                        "points_away": points_away,
                        "raw_json": json.dumps(p, ensure_ascii=False),
                        "source_fetched_at_utc": fetched_at,
                    }
                )

            processed_games += 1

            if processed_games % commit_every_games == 0:
                total_rows += _flush(conn, batch)
                conn.commit()
                batch.clear()
                log.info("Committed pbp: games=%d rows=%d", processed_games, total_rows)

    if batch:
        total_rows += _flush(conn, batch)
        conn.commit()

    log.info("Done pbp: games=%d rows=%d", processed_games, total_rows)
    return total_rows


def _flush(conn, rows: list[dict]) -> int:
    if not rows:
        return 0

    # Dedup within batch (season, game_slug, play_id)
    dedup: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        key = (r["season"], r["game_slug"], str(r["play_id"]))
        dedup[key] = r
    rows = list(dedup.values())

    sql = """
    INSERT INTO raw.nba_pbp_plays (
      season, game_slug,
      play_id, play_seq,
      period, clock, event_type, team_abbr, description,
      points_home, points_away,
      raw_json,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (season, game_slug, play_id) DO UPDATE SET
      play_seq = EXCLUDED.play_seq,
      period = EXCLUDED.period,
      clock = EXCLUDED.clock,
      event_type = EXCLUDED.event_type,
      team_abbr = EXCLUDED.team_abbr,
      description = EXCLUDED.description,
      points_home = EXCLUDED.points_home,
      points_away = EXCLUDED.points_away,
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
              %(season)s, %(game_slug)s,
              %(play_id)s, %(play_seq)s,
              %(period)s, %(clock)s, %(event_type)s, %(team_abbr)s, %(description)s,
              %(points_home)s, %(points_away)s,
              %(raw_json)s::jsonb,
              %(source_fetched_at_utc)s, now()
            )
            """,
            page_size=5000,
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
        parse_all_pbp(conn)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
