import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("nba_pipeline.parse_boxscore")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    # MSF timestamps often end in Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None


def _dump_jsonb(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def upsert_boxscore_games(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.nba_boxscore_games (
      game_slug, season,
      start_ts_utc, ended_ts_utc,
      schedule_status, played_status,
      home_team_abbr, away_team_abbr,
      home_team_id, away_team_id,
      home_score_total, away_score_total,
      attendance,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug) DO UPDATE SET
      season = EXCLUDED.season,
      start_ts_utc = EXCLUDED.start_ts_utc,
      ended_ts_utc = EXCLUDED.ended_ts_utc,
      schedule_status = EXCLUDED.schedule_status,
      played_status = EXCLUDED.played_status,
      home_team_abbr = EXCLUDED.home_team_abbr,
      away_team_abbr = EXCLUDED.away_team_abbr,
      home_team_id = EXCLUDED.home_team_id,
      away_team_id = EXCLUDED.away_team_id,
      home_score_total = EXCLUDED.home_score_total,
      away_score_total = EXCLUDED.away_score_total,
      attendance = EXCLUDED.attendance,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""
        (
          %(game_slug)s, %(season)s,
          %(start_ts_utc)s, %(ended_ts_utc)s,
          %(schedule_status)s, %(played_status)s,
          %(home_team_abbr)s, %(away_team_abbr)s,
          %(home_team_id)s, %(away_team_id)s,
          %(home_score_total)s, %(away_score_total)s,
          %(attendance)s,
          %(source_fetched_at_utc)s, now()
        )
        """,
        page_size=500,
    )


def upsert_team_stats(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.nba_boxscore_team_stats (
      game_slug, season, team_abbr, team_id, is_home, stats,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug, team_abbr) DO UPDATE SET
      season = EXCLUDED.season,
      team_id = EXCLUDED.team_id,
      is_home = EXCLUDED.is_home,
      stats = EXCLUDED.stats,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""
        (
          %(game_slug)s, %(season)s, %(team_abbr)s, %(team_id)s, %(is_home)s, %(stats)s::jsonb,
          %(source_fetched_at_utc)s, now()
        )
        """,
        page_size=500,
    )


def upsert_player_stats(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.nba_boxscore_player_stats (
      game_slug, season,
      player_id, team_abbr, team_id, is_home,
      first_name, last_name, position, jersey_number,
      stats,
      source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug, player_id) DO UPDATE SET
      season = EXCLUDED.season,
      team_abbr = EXCLUDED.team_abbr,
      team_id = EXCLUDED.team_id,
      is_home = EXCLUDED.is_home,
      first_name = EXCLUDED.first_name,
      last_name = EXCLUDED.last_name,
      position = EXCLUDED.position,
      jersey_number = EXCLUDED.jersey_number,
      stats = EXCLUDED.stats,
      source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
      updated_at_utc = now()
    ;
    """
    execute_values(
        conn.cursor(),
        sql,
        rows,
        template="""
        (
          %(game_slug)s, %(season)s,
          %(player_id)s, %(team_abbr)s, %(team_id)s, %(is_home)s,
          %(first_name)s, %(last_name)s, %(position)s, %(jersey_number)s,
          %(stats)s::jsonb,
          %(source_fetched_at_utc)s, now()
        )
        """,
        page_size=1000,
    )


def parse_one_boxscore_payload(*, game_slug: str, season: str, fetched_at_utc: datetime, payload: dict) -> tuple[dict, list[dict], list[dict]]:
    """
    Returns:
      (game_row, team_rows, player_rows)
    """
    g = payload.get("game") or {}
    scoring = payload.get("scoring") or {}
    stats = payload.get("stats") or {}

    away_team = g.get("awayTeam") or {}
    home_team = g.get("homeTeam") or {}

    game_row = {
        "game_slug": game_slug,
        "season": season,
        "start_ts_utc": _parse_iso(g.get("startTime")),
        "ended_ts_utc": _parse_iso(g.get("endedTime")),
        "schedule_status": g.get("scheduleStatus"),
        "played_status": g.get("playedStatus"),
        "home_team_abbr": (home_team.get("abbreviation") or "").upper(),
        "away_team_abbr": (away_team.get("abbreviation") or "").upper(),
        "home_team_id": _as_int(home_team.get("id")),
        "away_team_id": _as_int(away_team.get("id")),
        "home_score_total": _as_int(scoring.get("homeScoreTotal")),
        "away_score_total": _as_int(scoring.get("awayScoreTotal")),
        "attendance": _as_int(g.get("attendance")),
        "source_fetched_at_utc": fetched_at_utc,
    }

    team_rows: list[dict] = []
    player_rows: list[dict] = []

    for side_key, is_home in (("away", False), ("home", True)):
        side = stats.get(side_key) or {}
        team_id = game_row["home_team_id"] if is_home else game_row["away_team_id"]
        team_abbr = game_row["home_team_abbr"] if is_home else game_row["away_team_abbr"]

        # teamStats is usually a list of 1 dict
        team_stats_list = side.get("teamStats") or []
        team_stats = team_stats_list[0] if team_stats_list else {}
        team_rows.append(
            {
                "game_slug": game_slug,
                "season": season,
                "team_abbr": team_abbr,
                "team_id": team_id,
                "is_home": is_home,
                "stats": _dump_jsonb(team_stats),
                "source_fetched_at_utc": fetched_at_utc,
            }
        )

        for p in (side.get("players") or []):
            player = p.get("player") or {}
            player_stats_list = p.get("playerStats") or []
            player_stats = player_stats_list[0] if player_stats_list else {}

            pid = _as_int(player.get("id"))
            if pid is None:
                continue

            player_rows.append(
                {
                    "game_slug": game_slug,
                    "season": season,
                    "player_id": pid,
                    "team_abbr": team_abbr,
                    "team_id": team_id,
                    "is_home": is_home,
                    "first_name": player.get("firstName"),
                    "last_name": player.get("lastName"),
                    "position": player.get("position"),
                    "jersey_number": str(player.get("jerseyNumber")) if player.get("jerseyNumber") is not None else None,
                    "stats": _dump_jsonb(player_stats),
                    "source_fetched_at_utc": fetched_at_utc,
                }
            )

    return game_row, team_rows, player_rows


def build_raw_boxscores(conn, *, commit_every: int = 250) -> None:
    """
    Parse all boxscore payloads currently in raw.api_responses.
    Commits in batches so you see progress and avoid huge transactions.
    """
    q = """
    SELECT DISTINCT ON (season, game_slug)
           season, game_slug, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider='mysportsfeeds'
      AND endpoint='boxscore'
      AND game_slug IS NOT NULL
    ORDER BY season, game_slug, fetched_at_utc DESC
    """

    games_batch: list[dict] = []
    teams_batch: list[dict] = []
    players_batch: list[dict] = []
    processed = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q)
        for r in cur:
            season = r["season"]
            game_slug = r["game_slug"]
            fetched_at = r["fetched_at_utc"]
            payload = r["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            game_row, team_rows, player_rows = parse_one_boxscore_payload(
                game_slug=game_slug,
                season=season,
                fetched_at_utc=fetched_at,
                payload=payload,
            )

            games_batch.append(game_row)
            teams_batch.extend(team_rows)
            players_batch.extend(player_rows)
            processed += 1

            def _dedup_by_key(rows: list[dict], key_fn):
                out = {}
                for r in rows:
                    out[key_fn(r)] = r
                return list(out.values())

            # right before upsert_boxscore_games / upsert_team_stats / upsert_player_stats:
            games_batch = _dedup_by_key(games_batch, lambda r: r["game_slug"])
            teams_batch = _dedup_by_key(teams_batch, lambda r: (r["game_slug"], r["team_abbr"]))
            players_batch = _dedup_by_key(players_batch, lambda r: (r["game_slug"], r["player_id"]))

            if processed % commit_every == 0:
                log.info("Upserting batch processed=%d (games=%d teams=%d players=%d)", processed, len(games_batch), len(teams_batch), len(players_batch))
                upsert_boxscore_games(conn, games_batch)
                upsert_team_stats(conn, teams_batch)
                upsert_player_stats(conn, players_batch)
                conn.commit()
                games_batch.clear()
                teams_batch.clear()
                players_batch.clear()

    # final flush
    if games_batch or teams_batch or players_batch:
        log.info("Final upsert processed=%d (games=%d teams=%d players=%d)", processed, len(games_batch), len(teams_batch), len(players_batch))
        upsert_boxscore_games(conn, games_batch)
        upsert_team_stats(conn, teams_batch)
        upsert_player_stats(conn, players_batch)
        conn.commit()

    log.info("Done. Boxscores processed=%d", processed)


def main() -> None:
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dsn = "postgresql://josh:password@localhost:5432/nba"
    if not dsn:
        raise RuntimeError("Set PG_DSN, e.g. postgresql://user:pass@localhost:5432/nba")

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        build_raw_boxscores(conn)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
