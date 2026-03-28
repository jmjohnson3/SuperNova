"""
mlb_pipeline.parse_player_gamelogs
====================================
Parse MLB player_gamelogs payloads from raw.api_responses into
raw.mlb_player_gamelogs.

Source rows:
  provider='mysportsfeeds', endpoint='player_gamelogs', as_of_date IS NOT NULL

Incremental: only processes payloads newer than the MAX(game_date_et) already
stored in raw.mlb_player_gamelogs.

Run:
  python -m mlb_pipeline.parse_player_gamelogs
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    _ET = ZoneInfo("America/New_York")
except Exception as e:
    raise RuntimeError("tzdata missing — on Windows: pip install tzdata") from e

log = logging.getLogger("mlb_pipeline.parse_player_gamelogs")

DSN = "postgresql://josh:password@localhost:5432/nba"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    try:
        return int(str(x))
    except Exception:
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(str(x))
    except Exception:
        return None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _ensure_obj(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unexpected payload type: {type(payload)}")


def _game_date_et(start_time_str: Optional[str]) -> Optional[date]:
    """Convert an ISO UTC timestamp to an ET calendar date."""
    if not start_time_str:
        return None
    try:
        dt_utc = datetime.fromisoformat(str(start_time_str).replace("Z", "+00:00"))
        return dt_utc.astimezone(_ET).date()
    except Exception:
        return None


def _derive_game_slug(log_obj: dict) -> Optional[str]:
    """
    Reconstruct game_slug as YYYYMMDD-AWAY-HOME from the gamelog's game block.

    MSF provides awayTeam/homeTeam at the top-level game object:
      log.game.awayTeam.abbreviation
      log.game.homeTeam.abbreviation
    """
    game = log_obj.get("game") or {}
    start_time = game.get("startTime")
    away_team = game.get("awayTeam") or {}
    home_team = game.get("homeTeam") or {}
    away_abbr = (away_team.get("abbreviation") or "").upper()
    home_abbr = (home_team.get("abbreviation") or "").upper()

    gd = _game_date_et(start_time)
    if not (gd and away_abbr and home_abbr):
        return None

    return f"{gd.strftime('%Y%m%d')}-{away_abbr}-{home_abbr}"


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_PLAYER_GAMELOGS = """
CREATE TABLE IF NOT EXISTS raw.mlb_player_gamelogs (
    season              TEXT        NOT NULL,
    game_slug           TEXT        NOT NULL,
    player_id           INTEGER     NOT NULL,
    game_date_et        DATE,
    team_abbr           TEXT,
    opponent_abbr       TEXT,
    is_home             BOOLEAN,
    start_ts_utc        TIMESTAMPTZ,
    -- Pitching columns
    innings_pitched     NUMERIC(6,3),
    hits_allowed        INTEGER,
    runs_allowed        INTEGER,
    earned_runs         INTEGER,
    walks_allowed       INTEGER,
    strikeouts_pitcher  INTEGER,
    home_runs_allowed   INTEGER,
    is_starter          BOOLEAN,
    -- Batting columns
    at_bats             INTEGER,
    hits                INTEGER,
    doubles             INTEGER,
    triples             INTEGER,
    home_runs           INTEGER,
    rbi                 INTEGER,
    walks_batter        INTEGER,
    strikeouts_batter   INTEGER,
    stolen_bases        INTEGER,
    total_bases         INTEGER,
    -- Raw blob
    stats               JSONB,
    source_fetched_at_utc TIMESTAMPTZ,
    updated_at_utc      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (season, game_slug, player_id)
);
"""


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL_PLAYER_GAMELOGS)
    conn.commit()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_gamelogs_payload(
    season: str,
    payload: dict,
    fetched_at_utc: datetime,
) -> list[dict]:
    """
    Parse a single MSF player_gamelogs payload.

    MSF MLB player_gamelogs shape:
        {
          "gamelogs": [
            {
              "game": {
                "id": ...,
                "startTime": "2025-04-01T18:10:00Z",
                "awayTeam": {"id": ..., "abbreviation": "NYY"},
                "homeTeam": {"id": ..., "abbreviation": "BOS"}
              },
              "player": {"id": 12345, ...},
              "team": {"abbreviation": "NYY", ...},
              "stats": {
                "pitching": {
                  "inningsPitched": 6.333,
                  "hits": 4,
                  "runs": 2,
                  "earnedRuns": 2,
                  "walks": 1,
                  "strikeouts": 8,
                  "homeRuns": 1
                },
                "batting": {
                  "atBats": 0,
                  "hits": 0,
                  "doubles": 0,
                  "triples": 0,
                  "homeRuns": 0,
                  "runsBattedIn": 0,
                  "walks": 0,
                  "strikeouts": 0,
                  "stolenBases": 0
                }
              }
            },
            ...
          ]
        }

    innings_pitched is stored as a decimal where the fractional part represents
    outs recorded (0 = 0 outs, .333 = 1 out, .667 = 2 outs).

    is_starter heuristic: inningsPitched > 0  (simple and works for NL and AL).
    """
    gamelogs = payload.get("gamelogs") or []
    rows: list[dict] = []

    for gl in gamelogs:
        game = gl.get("game") or {}
        player = gl.get("player") or {}
        team = gl.get("team") or {}
        stats_top = gl.get("stats") or {}

        pid = _as_int(player.get("id"))
        if pid is None:
            continue

        team_abbr = (team.get("abbreviation") or "").upper()
        if not team_abbr:
            continue

        start_time = game.get("startTime")
        gd = _game_date_et(start_time)
        game_slug = _derive_game_slug(gl)
        if not game_slug:
            # Cannot reconstruct slug — skip; SQL views can join on date+team if needed
            log.debug("Skipping gamelog for player_id=%s: cannot derive game_slug", pid)
            continue

        # ---- Pitching stats ----
        pitching = stats_top.get("pitching") or {}
        ip_raw = pitching.get("inningsPitched")
        innings_pitched = _as_float(ip_raw)
        hits_allowed = _as_int(pitching.get("hits"))
        runs_allowed = _as_int(pitching.get("runs"))
        earned_runs = _as_int(pitching.get("earnedRuns"))
        walks_allowed = _as_int(pitching.get("walks"))
        strikeouts_pitcher = _as_int(pitching.get("strikeouts"))
        home_runs_allowed = _as_int(pitching.get("homeRuns"))

        # is_starter: True if pitcher recorded any outs/innings
        is_starter = bool(innings_pitched is not None and innings_pitched > 0)

        # ---- Batting stats ----
        batting = stats_top.get("batting") or {}
        at_bats = _as_int(batting.get("atBats"))
        hits = _as_int(batting.get("hits"))
        doubles = _as_int(batting.get("doubles"))
        triples = _as_int(batting.get("triples"))
        home_runs = _as_int(batting.get("homeRuns"))
        rbi = _as_int(batting.get("runsBattedIn"))
        walks_batter = _as_int(batting.get("walks"))
        strikeouts_batter = _as_int(batting.get("strikeouts"))
        stolen_bases = _as_int(batting.get("stolenBases"))

        # total_bases = 1B + 2*(2B) + 3*(3B) + 4*(HR)
        # singles = hits - doubles - triples - home_runs
        if hits is not None:
            _doubles = doubles or 0
            _triples = triples or 0
            _hr = home_runs or 0
            singles = max(hits - _doubles - _triples - _hr, 0)
            total_bases = singles + 2 * _doubles + 3 * _triples + 4 * _hr
        else:
            total_bases = None

        # ---- Home/away derivation ----
        away_team = game.get("awayTeam") or {}
        home_team = game.get("homeTeam") or {}
        away_abbr = (away_team.get("abbreviation") or "").upper()
        home_abbr = (home_team.get("abbreviation") or "").upper()
        is_home: Optional[bool] = None
        opponent_abbr: Optional[str] = None
        if team_abbr and home_abbr and away_abbr:
            if team_abbr == home_abbr:
                is_home = True
                opponent_abbr = away_abbr
            elif team_abbr == away_abbr:
                is_home = False
                opponent_abbr = home_abbr

        # Stats blob for forward-compat storage
        stats_blob = {
            "pitching": pitching,
            "batting": batting,
        }

        rows.append(
            {
                "season": season,
                "game_slug": game_slug,
                "player_id": pid,
                "game_date_et": gd,
                "team_abbr": team_abbr,
                "opponent_abbr": opponent_abbr,
                "is_home": is_home,
                "start_ts_utc": _parse_iso(start_time),
                "innings_pitched": innings_pitched,
                "hits_allowed": hits_allowed,
                "runs_allowed": runs_allowed,
                "earned_runs": earned_runs,
                "walks_allowed": walks_allowed,
                "strikeouts_pitcher": strikeouts_pitcher,
                "home_runs_allowed": home_runs_allowed,
                "is_starter": is_starter,
                "at_bats": at_bats,
                "hits": hits,
                "doubles": doubles,
                "triples": triples,
                "home_runs": home_runs,
                "rbi": rbi,
                "walks_batter": walks_batter,
                "strikeouts_batter": strikeouts_batter,
                "stolen_bases": stolen_bases,
                "total_bases": total_bases,
                "stats": json.dumps(stats_blob, ensure_ascii=False),
                "source_fetched_at_utc": fetched_at_utc,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_mlb_player_gamelogs(conn, rows: list[dict]) -> int:
    if not rows:
        return 0

    # Deduplicate within the batch on PK (season, game_slug, player_id)
    dedup: dict[tuple, dict] = {}
    for r in rows:
        key = (r["season"], r["game_slug"], int(r["player_id"]))
        dedup[key] = r
    rows = list(dedup.values())

    sql = """
    INSERT INTO raw.mlb_player_gamelogs (
        season, game_slug, player_id,
        game_date_et, team_abbr, opponent_abbr, is_home,
        start_ts_utc,
        innings_pitched, hits_allowed, runs_allowed, earned_runs,
        walks_allowed, strikeouts_pitcher, home_runs_allowed, is_starter,
        at_bats, hits, doubles, triples, home_runs,
        rbi, walks_batter, strikeouts_batter, stolen_bases, total_bases,
        stats,
        source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (season, game_slug, player_id) DO UPDATE SET
        game_date_et         = EXCLUDED.game_date_et,
        team_abbr            = EXCLUDED.team_abbr,
        opponent_abbr        = EXCLUDED.opponent_abbr,
        is_home              = EXCLUDED.is_home,
        start_ts_utc         = EXCLUDED.start_ts_utc,
        innings_pitched      = EXCLUDED.innings_pitched,
        hits_allowed         = EXCLUDED.hits_allowed,
        runs_allowed         = EXCLUDED.runs_allowed,
        earned_runs          = EXCLUDED.earned_runs,
        walks_allowed        = EXCLUDED.walks_allowed,
        strikeouts_pitcher   = EXCLUDED.strikeouts_pitcher,
        home_runs_allowed    = EXCLUDED.home_runs_allowed,
        is_starter           = EXCLUDED.is_starter,
        at_bats              = EXCLUDED.at_bats,
        hits                 = EXCLUDED.hits,
        doubles              = EXCLUDED.doubles,
        triples              = EXCLUDED.triples,
        home_runs            = EXCLUDED.home_runs,
        rbi                  = EXCLUDED.rbi,
        walks_batter         = EXCLUDED.walks_batter,
        strikeouts_batter    = EXCLUDED.strikeouts_batter,
        stolen_bases         = EXCLUDED.stolen_bases,
        total_bases          = EXCLUDED.total_bases,
        stats                = EXCLUDED.stats,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
        updated_at_utc       = now()
    ;
    """

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            rows,
            template="""(
                %(season)s, %(game_slug)s, %(player_id)s,
                %(game_date_et)s, %(team_abbr)s, %(opponent_abbr)s, %(is_home)s,
                %(start_ts_utc)s,
                %(innings_pitched)s, %(hits_allowed)s, %(runs_allowed)s, %(earned_runs)s,
                %(walks_allowed)s, %(strikeouts_pitcher)s, %(home_runs_allowed)s, %(is_starter)s,
                %(at_bats)s, %(hits)s, %(doubles)s, %(triples)s, %(home_runs)s,
                %(rbi)s, %(walks_batter)s, %(strikeouts_batter)s, %(stolen_bases)s, %(total_bases)s,
                %(stats)s::jsonb,
                %(source_fetched_at_utc)s, now()
            )""",
            page_size=1000,
        )

    return len(rows)


# ---------------------------------------------------------------------------
# Orchestrator — incremental
# ---------------------------------------------------------------------------

def _get_max_game_date(conn) -> Optional[date]:
    """Return the MAX(game_date_et) already in raw.mlb_player_gamelogs, or None."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(game_date_et) FROM raw.mlb_player_gamelogs")
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        # Table might not exist yet
        return None


def build_mlb_player_gamelogs(conn, *, commit_every: int = 250, full_reload: bool = False) -> int:
    """
    Incrementally parse player_gamelogs payloads from raw.api_responses.

    By default, only processes payloads whose as_of_date is >= (max_game_date - 1 day)
    to handle re-crawls of in-progress games and the day-boundary edge case.
    Pass full_reload=True to reprocess everything from scratch.

    Returns total rows upserted.
    """
    _ensure_table(conn)

    max_date = None if full_reload else _get_max_game_date(conn)
    if max_date is not None:
        log.info("Incremental mode: max game_date_et in DB = %s; will process payloads >= that date", max_date)
        date_filter = "AND as_of_date >= %(since)s::date"
        filter_params = {"since": max_date}
    else:
        log.info("Full-reload mode: processing all player_gamelogs payloads")
        date_filter = ""
        filter_params = {}

    q = f"""
    SELECT season, as_of_date, fetched_at_utc, payload
    FROM raw.api_responses
    WHERE provider = 'mysportsfeeds'
      AND endpoint = 'player_gamelogs'
      AND as_of_date IS NOT NULL
      AND payload IS NOT NULL
      {date_filter}
    ORDER BY as_of_date ASC, fetched_at_utc ASC
    """

    upsert_buffer: list[dict] = []
    processed_payloads = 0
    total_rows_inserted = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q, filter_params)
        for r in cur:
            season = r.get("season") or ""
            fetched_at = r["fetched_at_utc"]
            payload = _ensure_obj(r["payload"])

            try:
                rows = parse_gamelogs_payload(
                    season=season,
                    payload=payload,
                    fetched_at_utc=fetched_at,
                )
            except Exception:
                log.exception(
                    "Error parsing player_gamelogs payload as_of_date=%s — skipping",
                    r.get("as_of_date"),
                )
                continue

            upsert_buffer.extend(rows)
            processed_payloads += 1

            if processed_payloads % commit_every == 0:
                n = upsert_mlb_player_gamelogs(conn, upsert_buffer)
                conn.commit()
                total_rows_inserted += n
                upsert_buffer.clear()
                log.info(
                    "Committed: payloads=%d rows_this_batch=%d total_rows=%d",
                    processed_payloads, n, total_rows_inserted,
                )

    # Final flush
    if upsert_buffer:
        n = upsert_mlb_player_gamelogs(conn, upsert_buffer)
        conn.commit()
        total_rows_inserted += n

    log.info(
        "Done. MLB player_gamelogs: payloads=%d total_rows=%d",
        processed_payloads, total_rows_inserted,
    )
    return total_rows_inserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Parse MLB player gamelogs into raw.mlb_player_gamelogs")
    parser.add_argument(
        "--full-reload",
        action="store_true",
        help="Reprocess all payloads instead of incremental mode",
    )
    args = parser.parse_args()

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        build_mlb_player_gamelogs(conn, full_reload=args.full_reload)
    except Exception:
        conn.rollback()
        log.exception("Failed; rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
