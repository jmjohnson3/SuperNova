"""
mlb_pipeline.crawler_statsapi
==============================
Crawls the free MLB Stats API (statsapi.mlb.com) to populate MLB game data tables
directly (bypassing raw.api_responses since the payload format differs from MSF).

Populates:
  - raw.mlb_games                  — schedule + scores
  - raw.mlb_boxscore_games         — game-level run/hit/error totals
  - raw.mlb_boxscore_player_stats  — per-player batting/pitching stats (JSONB)
  - raw.mlb_player_gamelogs        — structured batting + pitching per player
  - raw.mlb_starting_pitchers      — starting pitchers (from gamesStarted=1)

Run:
  python -m mlb_pipeline.crawler_statsapi [--season 2024] [--season 2025]
  python -m mlb_pipeline.crawler_statsapi --start-date 2024-03-20 --end-date 2024-09-30

No API key required. Respects rate limits with a 0.15s sleep between requests.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timezone, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

log = logging.getLogger("mlb_pipeline.crawler_statsapi")

DSN = "postgresql://josh:password@localhost:5432/nba"
BASE_URL = "https://statsapi.mlb.com/api/v1"
ET = ZoneInfo("America/New_York")
REQUEST_SLEEP = 0.15  # seconds between requests

# MLB Stats API abbr → standard abbr used in rest of pipeline
# (matches Odds API abbreviations from odds.mlb_game_lines)
STATSAPI_ABBR_NORM: dict[str, str] = {
    "AZ":  "ARI",   # Arizona Diamondbacks
    "WSH": "WAS",   # Washington Nationals
}

SEASON_DATE_RANGES: dict[str, tuple[str, str]] = {
    "2024-regular": ("2024-03-20", "2024-09-30"),
    "2025-regular": ("2025-03-27", "2025-10-05"),
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    """GET from MLB Stats API, return parsed JSON."""
    import urllib.request, urllib.parse

    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"User-Agent": "SuperNovaBets/1.0 (sports analytics)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Team abbreviation normalization
# ---------------------------------------------------------------------------

def _norm(abbr: str) -> str:
    return STATSAPI_ABBR_NORM.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Schedule fetch + mlb_games upsert
# ---------------------------------------------------------------------------

def fetch_schedule(season_slug: str, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    """
    Fetch the full regular-season schedule from MLB Stats API.
    Returns a deduplicated list of game dicts (one per gamePk).

    The API sometimes returns the same gamePk on multiple dates (postponed/
    rescheduled games). We deduplicate by gamePk, preferring the entry with
    a real doubleHeader designation over the placeholder 'N' entry, so that
    the gameNumber suffix (-G2) is applied correctly.
    """
    year = int(season_slug.split("-")[0])

    params: dict[str, Any] = {
        "sportId": 1,
        "season": year,
        "gameType": "R",
        "hydrate": "team,venue",
    }
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    log.info("Fetching schedule for season=%s ...", season_slug)
    data = _get("/schedule", params)
    time.sleep(REQUEST_SLEEP)

    # Deduplicate by gamePk: prefer doubleHeader != 'N' entries
    seen: dict[int, dict] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            pk = g.get("gamePk")
            if pk is None:
                continue
            existing = seen.get(pk)
            if existing is None:
                seen[pk] = g
            else:
                # Prefer the doubleheader-aware entry
                if g.get("doubleHeader", "N") != "N" and existing.get("doubleHeader", "N") == "N":
                    seen[pk] = g

    games = list(seen.values())
    log.info("Schedule returned %d unique games for season=%s", len(games), season_slug)
    return games


def _game_slug_from_api(game: dict) -> str | None:
    """Build YYYYMMDD-AWAY-HOME slug from a schedule game object.
    Appends -G2 for the second game of a doubleheader to avoid slug collisions.
    """
    game_date = game.get("officialDate") or game.get("gameDate", "")[:10]
    teams = game.get("teams", {})
    home_abbr = _norm(teams.get("home", {}).get("team", {}).get("abbreviation", ""))
    away_abbr = _norm(teams.get("away", {}).get("team", {}).get("abbreviation", ""))
    if not (game_date and home_abbr and away_abbr):
        return None
    date_str = game_date.replace("-", "")
    slug = f"{date_str}-{away_abbr}-{home_abbr}"
    # Doubleheader suffix: append -G2 for the 2nd game to prevent PK conflict
    game_number = game.get("gameNumber", 1)
    if game_number and int(game_number) > 1:
        slug += "-G2"
    return slug


def _season_from_slug(game_slug: str) -> str:
    """Derive season slug from game_slug like '20240320-LAD-SD' → '2024-regular'."""
    year = game_slug[:4]
    return f"{year}-regular"


def upsert_mlb_games_from_schedule(conn, games: list[dict], season_slug: str) -> int:
    """Parse schedule games and upsert into raw.mlb_games.

    Deduplicates by game_slug, preferring Final/completed games over
    Postponed/Cancelled when two different gamePks map to the same slug.
    """
    # Build slug→game dict, preferring Final over Postponed
    _STATUS_PRIORITY = {"Final": 0, "Completed Early": 1, "Game Over": 2}
    slug_to_game: dict[str, dict] = {}
    for g in games:
        s = _game_slug_from_api(g)
        if not s:
            continue
        existing = slug_to_game.get(s)
        if existing is None:
            slug_to_game[s] = g
        else:
            # Prefer the game that is more "completed"
            new_priority = _STATUS_PRIORITY.get(g.get("status", {}).get("detailedState", ""), 99)
            old_priority = _STATUS_PRIORITY.get(existing.get("status", {}).get("detailedState", ""), 99)
            if new_priority < old_priority:
                slug_to_game[s] = g

    rows = []
    now_utc = datetime.now(timezone.utc)

    for slug, g in slug_to_game.items():

        game_date_str = g.get("officialDate") or g.get("gameDate", "")[:10]
        try:
            game_date_et = date.fromisoformat(game_date_str)
        except Exception:
            continue

        game_date_ts = g.get("gameDate", "")
        start_ts_utc: Optional[datetime] = None
        if game_date_ts:
            try:
                start_ts_utc = datetime.fromisoformat(game_date_ts.replace("Z", "+00:00"))
            except Exception:
                pass

        teams = g.get("teams", {})
        home_abbr = _norm(teams.get("home", {}).get("team", {}).get("abbreviation", ""))
        away_abbr = _norm(teams.get("away", {}).get("team", {}).get("abbreviation", ""))

        venue = g.get("venue", {})
        venue_id = venue.get("id")  # integer venue ID from MLB Stats API

        status_detail = g.get("status", {}).get("detailedState", "")
        if status_detail == "Final":
            status = "final"
        elif status_detail in ("In Progress", "Delayed: Rain"):
            status = "in_progress"
        else:
            status = "scheduled"

        home_score = teams.get("home", {}).get("score")
        away_score = teams.get("away", {}).get("score")

        rows.append({
            "game_slug": slug,
            "season": season_slug,
            "game_date_et": game_date_et,
            "start_ts_utc": start_ts_utc,
            "home_team_abbr": home_abbr,
            "away_team_abbr": away_abbr,
            "venue_id": venue_id,
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "source_fetched_at_utc": now_utc,
        })

    if not rows:
        return 0

    sql = """
    INSERT INTO raw.mlb_games (
        game_slug, season, game_date_et, start_ts_utc,
        home_team_abbr, away_team_abbr, venue_id,
        status, home_score, away_score,
        source_fetched_at_utc, updated_at_utc
    )
    VALUES %s
    ON CONFLICT (game_slug) DO UPDATE SET
        status               = EXCLUDED.status,
        home_score           = EXCLUDED.home_score,
        away_score           = EXCLUDED.away_score,
        source_fetched_at_utc = EXCLUDED.source_fetched_at_utc,
        updated_at_utc       = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="""(
                %(game_slug)s, %(season)s, %(game_date_et)s, %(start_ts_utc)s,
                %(home_team_abbr)s, %(away_team_abbr)s, %(venue_id)s,
                %(status)s, %(home_score)s, %(away_score)s,
                %(source_fetched_at_utc)s, now()
            )""",
            page_size=500,
        )
    conn.commit()
    log.info("Upserted %d rows into raw.mlb_games", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Already-fetched guard
# ---------------------------------------------------------------------------

def _already_fetched_slugs(conn) -> set[str]:
    """Return set of game_slugs already in raw.mlb_boxscore_games."""
    with conn.cursor() as cur:
        cur.execute("SELECT game_slug FROM raw.mlb_boxscore_games")
        return {r[0] for r in cur.fetchall()}


def _completed_game_slugs(conn) -> list[tuple[str, str, str, str]]:
    """Return (game_slug, home_team_abbr, away_team_abbr, season) for final games."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_slug, home_team_abbr, away_team_abbr, season
            FROM raw.mlb_games
            WHERE status = 'final'
            ORDER BY game_date_et ASC
        """)
        return cur.fetchall()


def _gamepk_for_slug(slug: str, schedule_index: dict[str, int]) -> int | None:
    """Look up gamePk from the in-memory schedule index."""
    return schedule_index.get(slug)


# ---------------------------------------------------------------------------
# Boxscore fetch + parse
# ---------------------------------------------------------------------------

def fetch_boxscore(game_pk: int) -> dict:
    """Fetch /game/{gamePk}/boxscore from MLB Stats API."""
    return _get(f"/game/{game_pk}/boxscore")


def fetch_linescore(game_pk: int) -> dict:
    """Fetch /game/{gamePk}/linescore from MLB Stats API."""
    return _get(f"/game/{game_pk}/linescore")


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x))
    except (TypeError, ValueError):
        return None


def _parse_ip(ip_str: Any) -> Optional[float]:
    """Convert MLB innings-pitched notation to decimal innings.

    MLB API stores IP as 'W.O' where W=whole innings, O=outs (0,1,2).
    Examples: '5.0'→5.0, '5.1'→5.333, '5.2'→5.667, '0.1'→0.333, '0.2'→0.667
    """
    if ip_str is None:
        return None
    try:
        v = float(str(ip_str))
        whole = int(v)
        outs = round((v - whole) * 10)  # number of outs in partial inning (0, 1, or 2)
        return whole + outs / 3.0
    except Exception:
        return None


def parse_boxscore(
    game_slug: str,
    season: str,
    game_pk: int,
    boxscore: dict,
    linescore: dict,
    home_team_abbr: str,
    away_team_abbr: str,
) -> tuple[dict, dict, list[dict], list[dict], list[dict], list[dict]]:
    """
    Parse a boxscore + linescore response into structured row dicts.

    Returns:
        boxscore_game_row   : dict for raw.mlb_boxscore_games
        (not used)          : (placeholder)
        player_stat_rows    : list of dicts for raw.mlb_boxscore_player_stats
        gamelog_rows        : list of dicts for raw.mlb_player_gamelogs
        sp_rows             : list of dicts for raw.mlb_starting_pitchers
        ump_rows            : list of dicts for raw.mlb_game_umpires
    """
    now_utc = datetime.now(timezone.utc)
    teams_bs = boxscore.get("teams", {})
    ls_teams = linescore.get("teams", {})

    # --- Game-level row ---
    home_runs = _as_int(ls_teams.get("home", {}).get("runs"))
    away_runs = _as_int(ls_teams.get("away", {}).get("runs"))
    home_hits = _as_int(ls_teams.get("home", {}).get("hits"))
    away_hits = _as_int(ls_teams.get("away", {}).get("hits"))
    home_errors = _as_int(ls_teams.get("home", {}).get("errors"))
    away_errors = _as_int(ls_teams.get("away", {}).get("errors"))
    innings = _as_int(linescore.get("currentInning"))

    game_row = {
        "game_slug": game_slug,
        "season": season,
        "home_team_abbr": home_team_abbr,
        "away_team_abbr": away_team_abbr,
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_hits": home_hits,
        "away_hits": away_hits,
        "home_errors": home_errors,
        "away_errors": away_errors,
        "innings_played": innings,
        "played_status": "COMPLETED",
        "source_fetched_at_utc": now_utc,
    }

    player_stat_rows: list[dict] = []
    gamelog_rows: list[dict] = []
    sp_rows: list[dict] = []

    for side, team_abbr, is_home in [
        ("home", home_team_abbr, True),
        ("away", away_team_abbr, False),
    ]:
        team_data = teams_bs.get(side, {})
        team_id = team_data.get("team", {}).get("id")
        players = team_data.get("players", {})
        pitcher_ids = set(team_data.get("pitchers", []))
        batter_ids = set(team_data.get("batters", []))

        for pid_key, p in players.items():
            # pid_key = "ID607192"
            player_id = _as_int(p.get("person", {}).get("id"))
            if not player_id:
                continue

            person = p.get("person", {})
            full_name = person.get("fullName", "")
            name_parts = full_name.rsplit(" ", 1)
            first_name = name_parts[0] if len(name_parts) > 1 else ""
            last_name = name_parts[-1]

            position = p.get("position", {}).get("abbreviation", "")
            batting_order_raw = p.get("battingOrder")
            # battingOrder is "100","200",...,"900" for 1-9; subs like "401","402"
            batting_order: Optional[int] = None
            if batting_order_raw is not None:
                try:
                    batting_order = int(str(batting_order_raw)) // 100
                except Exception:
                    pass

            stats = p.get("stats", {})
            batting_stats = stats.get("batting", {})
            pitching_stats = stats.get("pitching", {})

            # Build JSONB stats for mlb_boxscore_player_stats
            jsonb_stats: dict = {}
            if batting_stats:
                jsonb_stats["batting"] = batting_stats
            if pitching_stats:
                jsonb_stats["pitching"] = pitching_stats

            player_stat_rows.append({
                "game_slug": game_slug,
                "player_id": player_id,
                "season": season,
                "team_abbr": team_abbr,
                "team_id": _as_int(team_id),
                "is_home": is_home,
                "first_name": first_name,
                "last_name": last_name,
                "primary_position": position,
                "batting_order": batting_order,
                "stats": json.dumps(jsonb_stats),
                "source_fetched_at_utc": now_utc,
            })

            # Build gamelog row (only for players who actually played)
            has_pitching = bool(pitching_stats.get("gamesPlayed"))
            has_batting = bool(batting_stats.get("atBats") or batting_stats.get("plateAppearances"))
            is_starter_pitcher = False  # initialize before conditional block

            if has_pitching or has_batting:
                ip = _parse_ip(pitching_stats.get("inningsPitched"))
                is_starter_pitcher = _as_int(pitching_stats.get("gamesStarted")) == 1

                # Derive game_date_et from slug (YYYYMMDD prefix)
                try:
                    gs_date = game_slug[:8]
                    game_date_et = date(int(gs_date[:4]), int(gs_date[4:6]), int(gs_date[6:8]))
                except Exception:
                    game_date_et = None

                gamelog_rows.append({
                    "season": season,
                    "game_slug": game_slug,
                    "player_id": player_id,
                    "team_abbr": team_abbr,
                    "game_date_et": game_date_et,
                    "is_starter": is_starter_pitcher,
                    # Pitching
                    "innings_pitched": ip,
                    "hits_allowed": _as_int(pitching_stats.get("hits")) if has_pitching else None,
                    "runs_allowed": _as_int(pitching_stats.get("runs")) if has_pitching else None,
                    "earned_runs": _as_int(pitching_stats.get("earnedRuns")) if has_pitching else None,
                    "walks_allowed": _as_int(pitching_stats.get("baseOnBalls")) if has_pitching else None,
                    "strikeouts_pitcher": _as_int(pitching_stats.get("strikeOuts")) if has_pitching else None,
                    "home_runs_allowed": _as_int(pitching_stats.get("homeRuns")) if has_pitching else None,
                    # Batting
                    "at_bats": _as_int(batting_stats.get("atBats")) if has_batting else None,
                    "hits": _as_int(batting_stats.get("hits")) if has_batting else None,
                    "doubles": _as_int(batting_stats.get("doubles")) if has_batting else None,
                    "triples": _as_int(batting_stats.get("triples")) if has_batting else None,
                    "home_runs": _as_int(batting_stats.get("homeRuns")) if has_batting else None,
                    "rbi": _as_int(batting_stats.get("rbi")) if has_batting else None,
                    "walks_batter": _as_int(batting_stats.get("baseOnBalls")) if has_batting else None,
                    "strikeouts_batter": _as_int(batting_stats.get("strikeOuts")) if has_batting else None,
                    "stolen_bases": _as_int(batting_stats.get("stolenBases")) if has_batting else None,
                    "total_bases": _as_int(batting_stats.get("totalBases")) if has_batting else None,
                    "fetched_at_utc": now_utc,
                })

            # Starting pitcher row
            if is_starter_pitcher:
                sp_rows.append({
                    "game_slug": game_slug,
                    "team_abbr": team_abbr,
                    "player_id": player_id,
                    "player_name": full_name,
                    "source": "actual",
                })

    # Extract umpires from boxscore officials array
    ump_rows: list[dict] = []
    for off in boxscore.get("officials", []):
        ump_id = off.get("official", {}).get("id")
        ump_name = off.get("official", {}).get("fullName", "")
        ump_pos = off.get("officialType", "")  # "Home Plate", "First Base", etc.
        if ump_id:
            ump_rows.append({
                "game_slug": game_slug,
                "ump_position": ump_pos,
                "umpire_id": int(ump_id),
                "umpire_name": ump_name,
            })

    return game_row, {}, player_stat_rows, gamelog_rows, sp_rows, ump_rows


# ---------------------------------------------------------------------------
# DB upserts
# ---------------------------------------------------------------------------

def upsert_boxscore_game(conn, row: dict) -> None:
    sql = """
    INSERT INTO raw.mlb_boxscore_games (
        game_slug, season, home_team_abbr, away_team_abbr,
        home_runs, away_runs, home_hits, away_hits, home_errors, away_errors,
        innings_played, played_status, source_fetched_at_utc, updated_at_utc
    ) VALUES (
        %(game_slug)s, %(season)s, %(home_team_abbr)s, %(away_team_abbr)s,
        %(home_runs)s, %(away_runs)s, %(home_hits)s, %(away_hits)s,
        %(home_errors)s, %(away_errors)s,
        %(innings_played)s, %(played_status)s, %(source_fetched_at_utc)s, now()
    )
    ON CONFLICT (game_slug) DO UPDATE SET
        home_runs   = EXCLUDED.home_runs,
        away_runs   = EXCLUDED.away_runs,
        home_hits   = EXCLUDED.home_hits,
        away_hits   = EXCLUDED.away_hits,
        innings_played = EXCLUDED.innings_played,
        played_status  = EXCLUDED.played_status,
        updated_at_utc = now()
    ;
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)


def upsert_player_stats(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_boxscore_player_stats (
        game_slug, player_id, season, team_abbr, team_id, is_home,
        first_name, last_name, primary_position, batting_order,
        stats, source_fetched_at_utc, updated_at_utc
    ) VALUES %s
    ON CONFLICT (game_slug, player_id) DO UPDATE SET
        stats          = EXCLUDED.stats,
        updated_at_utc = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="""(
                %(game_slug)s, %(player_id)s, %(season)s, %(team_abbr)s, %(team_id)s, %(is_home)s,
                %(first_name)s, %(last_name)s, %(primary_position)s, %(batting_order)s,
                %(stats)s::jsonb, %(source_fetched_at_utc)s, now()
            )""",
            page_size=200,
        )


def upsert_gamelogs(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_player_gamelogs (
        season, game_slug, player_id, team_abbr, game_date_et,
        is_starter,
        innings_pitched, hits_allowed, runs_allowed, earned_runs,
        walks_allowed, strikeouts_pitcher, home_runs_allowed,
        at_bats, hits, doubles, triples, home_runs,
        rbi, walks_batter, strikeouts_batter, stolen_bases, total_bases,
        fetched_at_utc
    ) VALUES %s
    ON CONFLICT (season, game_slug, player_id) DO NOTHING
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="""(
                %(season)s, %(game_slug)s, %(player_id)s, %(team_abbr)s, %(game_date_et)s,
                %(is_starter)s,
                %(innings_pitched)s, %(hits_allowed)s, %(runs_allowed)s, %(earned_runs)s,
                %(walks_allowed)s, %(strikeouts_pitcher)s, %(home_runs_allowed)s,
                %(at_bats)s, %(hits)s, %(doubles)s, %(triples)s, %(home_runs)s,
                %(rbi)s, %(walks_batter)s, %(strikeouts_batter)s, %(stolen_bases)s, %(total_bases)s,
                %(fetched_at_utc)s
            )""",
            page_size=200,
        )


def upsert_starting_pitchers(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_starting_pitchers (
        game_slug, team_abbr, player_id, player_name, source, fetched_at_utc
    ) VALUES %s
    ON CONFLICT (game_slug, team_abbr) DO UPDATE SET
        player_id      = EXCLUDED.player_id,
        player_name    = EXCLUDED.player_name,
        source         = EXCLUDED.source,
        fetched_at_utc = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="""(
                %(game_slug)s, %(team_abbr)s, %(player_id)s, %(player_name)s,
                %(source)s, now()
            )""",
            page_size=100,
        )


def upsert_umpires(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
    INSERT INTO raw.mlb_game_umpires (
        game_slug, ump_position, umpire_id, umpire_name,
        source_fetched_at_utc, updated_at_utc
    ) VALUES %s
    ON CONFLICT (game_slug, ump_position) DO UPDATE SET
        umpire_id             = EXCLUDED.umpire_id,
        umpire_name           = EXCLUDED.umpire_name,
        updated_at_utc        = now()
    ;
    """
    with conn.cursor() as cur:
        execute_values(
            cur, sql, rows,
            template="""(
                %(game_slug)s, %(ump_position)s, %(umpire_id)s, %(umpire_name)s,
                now(), now()
            )""",
            page_size=200,
        )


# ---------------------------------------------------------------------------
# Build schedule index (game_slug → gamePk)
# ---------------------------------------------------------------------------

def build_schedule_index(games: list[dict]) -> dict[str, int]:
    """Build a slug → gamePk lookup from raw schedule game objects."""
    index: dict[str, int] = {}
    for g in games:
        slug = _game_slug_from_api(g)
        pk = g.get("gamePk")
        if slug and pk:
            index[slug] = int(pk)
    return index


# ---------------------------------------------------------------------------
# Add game_date_et to mlb_boxscore_games if missing
# ---------------------------------------------------------------------------

def _ensure_boxscore_games_date_col(conn) -> None:
    """Add game_date_et column to mlb_boxscore_games if not present."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE raw.mlb_boxscore_games
            ADD COLUMN IF NOT EXISTS game_date_et DATE;
        """)
    conn.commit()


# ---------------------------------------------------------------------------
# Main backfill orchestrator
# ---------------------------------------------------------------------------

def backfill_season(
    conn,
    season_slug: str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_games: int | None = None,
) -> None:
    """
    Full backfill for a single season:
    1. Fetch schedule → upsert raw.mlb_games
    2. For each final game not yet in mlb_boxscore_games, fetch boxscore
    3. Upsert boxscore_games, player_stats, gamelogs, starting_pitchers
    """
    _ensure_boxscore_games_date_col(conn)

    # Step 1: schedule
    games = fetch_schedule(season_slug, start_date=start_date, end_date=end_date)
    upsert_mlb_games_from_schedule(conn, games, season_slug)

    # Build slug → gamePk index
    schedule_index = build_schedule_index(games)

    # Step 2: boxscores for completed games
    already_done = _already_fetched_slugs(conn)
    completed = _completed_game_slugs(conn)

    # Filter to only games we have in the schedule index
    to_fetch = [
        (slug, home, away, season)
        for slug, home, away, season in completed
        if slug not in already_done and slug in schedule_index
    ]

    if max_games:
        to_fetch = to_fetch[:max_games]

    log.info(
        "Season=%s: %d completed games, %d already done, %d to fetch",
        season_slug, len(completed), len(already_done), len(to_fetch),
    )

    errors = 0
    for i, (slug, home_abbr, away_abbr, season) in enumerate(to_fetch):
        game_pk = schedule_index[slug]

        try:
            boxscore = fetch_boxscore(game_pk)
            time.sleep(REQUEST_SLEEP)
            linescore = fetch_linescore(game_pk)
            time.sleep(REQUEST_SLEEP)

            game_row, _, player_rows, gamelog_rows, sp_rows, ump_rows = parse_boxscore(
                slug, season, game_pk, boxscore, linescore, home_abbr, away_abbr
            )

            upsert_boxscore_game(conn, game_row)
            upsert_player_stats(conn, player_rows)
            upsert_gamelogs(conn, gamelog_rows)
            upsert_starting_pitchers(conn, sp_rows)
            upsert_umpires(conn, ump_rows)
            conn.commit()

            if (i + 1) % 100 == 0 or (i + 1) == len(to_fetch):
                log.info("  Progress: %d / %d games fetched", i + 1, len(to_fetch))

        except Exception as exc:
            conn.rollback()
            log.warning("  Failed game_slug=%s gamePk=%d: %s", slug, game_pk, exc)
            errors += 1
            if errors > 50:
                log.error("Too many errors (%d); aborting season %s", errors, season_slug)
                break

    # Sync mlb_games scores from boxscores
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE raw.mlb_games g
            SET
                home_score     = b.home_runs,
                away_score     = b.away_runs,
                status         = 'final',
                updated_at_utc = now()
            FROM raw.mlb_boxscore_games b
            WHERE g.game_slug = b.game_slug
              AND b.home_runs IS NOT NULL
              AND b.away_runs IS NOT NULL
              AND b.played_status = 'COMPLETED'
              AND (g.home_score IS NULL OR g.away_score IS NULL OR g.status != 'final')
        """)
        synced = cur.rowcount
    conn.commit()

    if synced:
        log.info("Synced scores for %d games from mlb_boxscore_games → mlb_games", synced)

    # Update mlb_games SP IDs from mlb_starting_pitchers
    with conn.cursor() as cur:
        cur.execute("""
            WITH best_sp AS (
                SELECT DISTINCT ON (game_slug, team_abbr)
                    game_slug, team_abbr, player_id
                FROM raw.mlb_starting_pitchers
                WHERE player_id IS NOT NULL
                ORDER BY game_slug, team_abbr,
                    CASE WHEN source = 'actual' THEN 0 ELSE 1 END,
                    player_id
            )
            UPDATE raw.mlb_games g
            SET
                home_sp_id = home_sp.player_id,
                away_sp_id = away_sp.player_id,
                updated_at_utc = now()
            FROM best_sp home_sp
            JOIN best_sp away_sp ON away_sp.game_slug = home_sp.game_slug
            WHERE home_sp.game_slug = g.game_slug
              AND home_sp.team_abbr = g.home_team_abbr
              AND away_sp.team_abbr = g.away_team_abbr
              AND (
                  g.home_sp_id IS DISTINCT FROM home_sp.player_id
                  OR g.away_sp_id IS DISTINCT FROM away_sp.player_id
              )
        """)
        sp_updated = cur.rowcount
    conn.commit()

    if sp_updated:
        log.info("Updated SP IDs for %d games in raw.mlb_games", sp_updated)

    log.info(
        "Backfill complete: season=%s, fetched=%d, errors=%d",
        season_slug, len(to_fetch) - errors, errors,
    )


# ---------------------------------------------------------------------------
# Umpire backfill
# ---------------------------------------------------------------------------

def backfill_umpires(conn, season_slugs: list[str]) -> None:
    """
    One-time backfill: re-fetch boxscores only to extract officials for games
    that are in raw.mlb_boxscore_games but not yet in raw.mlb_game_umpires.

    ~2400 games × 0.15s ≈ 6 min.  Only needs to run once.
    """
    # Build slug → gamePk index from schedule
    schedule_index: dict[str, int] = {}
    for season_slug in season_slugs:
        start_date, end_date = SEASON_DATE_RANGES.get(
            season_slug, ("2024-03-20", "2025-10-05")
        )
        games = fetch_schedule(season_slug, start_date=start_date, end_date=end_date)
        schedule_index.update(build_schedule_index(games))

    # Games with boxscore but no umpire data
    with conn.cursor() as cur:
        cur.execute("""
            SELECT b.game_slug
            FROM raw.mlb_boxscore_games b
            WHERE NOT EXISTS (
                SELECT 1 FROM raw.mlb_game_umpires u
                WHERE u.game_slug = b.game_slug
            )
            ORDER BY b.game_slug
        """)
        slugs_needing = [r[0] for r in cur.fetchall()]

    log.info("Umpire backfill: %d games need umpire data", len(slugs_needing))
    errors = 0

    for i, slug in enumerate(slugs_needing):
        game_pk = schedule_index.get(slug)
        if game_pk is None:
            log.debug("No gamePk found for %s — skipping", slug)
            continue

        try:
            boxscore = fetch_boxscore(game_pk)
            time.sleep(REQUEST_SLEEP)

            ump_rows: list[dict] = []
            for off in boxscore.get("officials", []):
                ump_id = off.get("official", {}).get("id")
                ump_name = off.get("official", {}).get("fullName", "")
                ump_pos = off.get("officialType", "")
                if ump_id:
                    ump_rows.append({
                        "game_slug": slug,
                        "ump_position": ump_pos,
                        "umpire_id": int(ump_id),
                        "umpire_name": ump_name,
                    })

            if ump_rows:
                upsert_umpires(conn, ump_rows)
                conn.commit()

            if (i + 1) % 200 == 0 or (i + 1) == len(slugs_needing):
                log.info("  Umpire backfill progress: %d / %d", i + 1, len(slugs_needing))

        except Exception as exc:
            conn.rollback()
            log.warning("  Umpire backfill failed for %s: %s", slug, exc)
            errors += 1
            if errors > 50:
                log.error("Too many errors (%d); aborting umpire backfill", errors)
                break

    log.info("Umpire backfill complete: processed=%d errors=%d", len(slugs_needing), errors)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="MLB Stats API backfill crawler")
    parser.add_argument("--season", action="append", default=[], help="Season slug(s), e.g. 2024-regular")
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD (overrides season default)")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument("--max-games", type=int, help="Max boxscores to fetch (for testing)")
    parser.add_argument("--backfill-umpires", action="store_true",
                        help="One-time backfill: extract umpires from already-fetched boxscores")
    args = parser.parse_args()

    seasons = args.season or ["2024-regular", "2025-regular"]

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    try:
        if args.backfill_umpires:
            backfill_umpires(conn, seasons)
        else:
            for season_slug in seasons:
                # Derive date range from season if not overridden
                start_date = args.start_date
                end_date = args.end_date
                if not start_date and season_slug in SEASON_DATE_RANGES:
                    start_date, end_date = SEASON_DATE_RANGES[season_slug]

                backfill_season(
                    conn, season_slug,
                    start_date=start_date,
                    end_date=end_date,
                    max_games=args.max_games,
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
