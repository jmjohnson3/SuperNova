"""Import NFL player-game data from nflverse-style CSV files.

The importer accepts local files or URLs. It intentionally writes one row per
player/game so projection training does not duplicate the same player outcome
across many betting offers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import normalize_team
from nfl_pipeline.schema import ensure_schema

log = logging.getLogger("nfl_pipeline.import_nflverse")
_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

DEFAULT_PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
DEFAULT_SCHEDULES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "schedules/games.csv"
)


@dataclass(frozen=True)
class ImportConfig:
    pg_dsn: str = PG_DSN
    seasons: tuple[int, ...] = ()
    player_stats_csv: str | None = None
    player_stats_url_template: str = os.getenv("NFL_PLAYER_STATS_URL_TEMPLATE", DEFAULT_PLAYER_STATS_URL)
    schedules_csv: str | None = None
    schedules_url: str = os.getenv("NFL_SCHEDULES_URL", DEFAULT_SCHEDULES_URL)
    chunksize: int = 50_000
    schedules_only: bool = False


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _clean_pct(value: Any) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        out = float(text.rstrip("%"))
    except (TypeError, ValueError):
        return None
    return out / 100.0 if out > 1.0 else out


def _clean_int(value: Any) -> int | None:
    f = _clean_float(value)
    if f is None:
        return None
    return int(f)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _read_csv(source: str, **kwargs) -> pd.DataFrame:
    log.info("Reading %s", source)
    return pd.read_csv(source, low_memory=False, **kwargs)


def _value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None:
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                return value
    return None


def _json_subset(row: pd.Series, columns: Iterable[str]) -> str:
    data = {}
    for col in columns:
        if col in row.index:
            value = row.get(col)
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            data[col] = value
    return json.dumps(data, default=str)


def _parse_game_date(value: Any) -> date | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return pd.to_datetime(text, errors="coerce").date()
    except Exception:
        return None


def _parse_start_ts(gameday: Any, gametime: Any) -> datetime | None:
    game_date = _parse_game_date(gameday)
    time_text = _clean_text(gametime)
    if game_date is None or not time_text:
        return None
    try:
        parsed_time = pd.to_datetime(time_text, errors="coerce").time()
        return datetime.combine(game_date, parsed_time, tzinfo=_ET).astimezone(_UTC)
    except Exception:
        return None


def _schedule_rows(df: pd.DataFrame) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        game_id = _clean_text(_value(row, "game_id", "gsis_id", "old_game_id"))
        if not game_id:
            continue
        game_day_value = _value(row, "gameday", "game_date", "date")
        game_date = _parse_game_date(game_day_value)
        season_type = _clean_text(_value(row, "game_type", "season_type"))
        rows.append((
            game_id,
            _clean_int(_value(row, "season")),
            _clean_int(_value(row, "week")),
            season_type,
            game_date,
            _parse_start_ts(game_day_value, _value(row, "gametime", "game_time", "start_time")),
            (_clean_text(_value(row, "home_team")) or "").upper() or None,
            (_clean_text(_value(row, "away_team")) or "").upper() or None,
            _clean_float(_value(row, "home_score")),
            _clean_float(_value(row, "away_score")),
            _clean_float(_value(row, "spread_line")),
            _clean_float(_value(row, "total_line")),
            _clean_text(_value(row, "roof")),
            _clean_text(_value(row, "surface")),
            _clean_float(_value(row, "temp")),
            _clean_float(_value(row, "wind")),
            "final" if _clean_float(row.get("home_score")) is not None and _clean_float(row.get("away_score")) is not None else "scheduled",
            "nflverse_schedule",
            _json_subset(row, row.index),
        ))
    return rows


def _player_rows(
    df: pd.DataFrame,
    schedule_by_game: dict[str, dict[str, Any]],
    schedule_by_matchup: dict[tuple[int | None, int | None, str | None, str | None], dict[str, Any]],
) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        game_id = _clean_text(_value(row, "game_id"))
        player_id = _clean_text(_value(row, "player_id", "gsis_id", "pfr_id"))
        team = normalize_team(_clean_text(_value(row, "recent_team", "team", "team_abbr")))
        season = _clean_int(_value(row, "season"))
        week = _clean_int(_value(row, "week"))
        opponent = normalize_team(_clean_text(_value(row, "opponent_team", "opponent", "opponent_abbr")))
        if not player_id or not team:
            continue
        sched = schedule_by_game.get(game_id or "", {}) if game_id else {}
        if not sched:
            sched = schedule_by_matchup.get((season, week, team, opponent), {})
        game_id = game_id or _clean_text(sched.get("game_id"))
        if not game_id:
            continue
        if sched.get("status") != "final":
            continue
        home = normalize_team(sched.get("home_team_abbr"))
        away = normalize_team(sched.get("away_team_abbr"))
        is_home = None
        if team and home and away:
            if team == home:
                is_home = True
            elif team == away:
                is_home = False
        if not opponent and home and away:
            opponent = away if team == home else home if team == away else None
        rows.append((
            season,
            week,
            game_id,
            sched.get("game_date_et") or _parse_game_date(_value(row, "game_date", "gameday")),
            player_id,
            _clean_text(_value(row, "player_name", "player_display_name", "name")),
            team,
            normalize_team(opponent),
            (_clean_text(_value(row, "position", "position_group")) or "").upper() or None,
            is_home,
            _clean_float(_value(row, "passing_yards", "pass_yards")),
            _clean_float(_value(row, "passing_tds", "pass_tds")),
            _clean_float(_value(row, "rushing_yards", "rush_yards")),
            _clean_float(_value(row, "rushing_tds", "rush_tds")),
            _clean_float(_value(row, "receiving_yards", "rec_yards")),
            _clean_float(_value(row, "receiving_tds", "rec_tds")),
            _clean_float(_value(row, "carries", "rush_attempts", "rushing_attempts")),
            _clean_float(_value(row, "targets", "receiving_targets")),
            _clean_float(_value(row, "receptions")),
            _clean_float(_value(row, "attempts", "passing_attempts")),
            _clean_float(_value(row, "routes_run", "routes")),
            _clean_pct(_value(row, "route_participation", "route_participation_pct")),
            _clean_float(_value(row, "snap_share", "offense_pct", "offense_snaps_pct")),
            _clean_pct(_value(row, "target_share")),
            _clean_pct(_value(row, "air_yards_share")),
            _clean_float(_value(row, "wopr")),
            _clean_float(_value(row, "receiving_air_yards")),
            _clean_float(_value(row, "receiving_yards_after_catch")),
            "nflverse_player_stats",
            _json_subset(row, row.index),
        ))
    return rows


def _upsert_schedules(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO raw.nfl_games (
            game_id, season, week, season_type, game_date_et, start_ts_utc,
            home_team_abbr, away_team_abbr, home_score, away_score,
            spread_line, total_line, roof, surface, temp, wind,
            status, source, raw_json
        )
        VALUES %s
        ON CONFLICT (game_id) DO UPDATE SET
            season = EXCLUDED.season,
            week = EXCLUDED.week,
            season_type = EXCLUDED.season_type,
            game_date_et = EXCLUDED.game_date_et,
            start_ts_utc = EXCLUDED.start_ts_utc,
            home_team_abbr = EXCLUDED.home_team_abbr,
            away_team_abbr = EXCLUDED.away_team_abbr,
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score,
            spread_line = EXCLUDED.spread_line,
            total_line = EXCLUDED.total_line,
            roof = EXCLUDED.roof,
            surface = EXCLUDED.surface,
            temp = EXCLUDED.temp,
            wind = EXCLUDED.wind,
            status = EXCLUDED.status,
            source = EXCLUDED.source,
            raw_json = EXCLUDED.raw_json,
            updated_at_utc = NOW()
        WHERE (
            raw.nfl_games.season, raw.nfl_games.week, raw.nfl_games.season_type,
            raw.nfl_games.game_date_et, raw.nfl_games.start_ts_utc,
            raw.nfl_games.home_team_abbr, raw.nfl_games.away_team_abbr,
            raw.nfl_games.home_score, raw.nfl_games.away_score,
            raw.nfl_games.spread_line, raw.nfl_games.total_line,
            raw.nfl_games.roof, raw.nfl_games.surface, raw.nfl_games.temp, raw.nfl_games.wind,
            raw.nfl_games.status, raw.nfl_games.source, raw.nfl_games.raw_json
        ) IS DISTINCT FROM (
            EXCLUDED.season, EXCLUDED.week, EXCLUDED.season_type,
            EXCLUDED.game_date_et, EXCLUDED.start_ts_utc,
            EXCLUDED.home_team_abbr, EXCLUDED.away_team_abbr,
            EXCLUDED.home_score, EXCLUDED.away_score,
            EXCLUDED.spread_line, EXCLUDED.total_line,
            EXCLUDED.roof, EXCLUDED.surface, EXCLUDED.temp, EXCLUDED.wind,
            EXCLUDED.status, EXCLUDED.source, EXCLUDED.raw_json
        )
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def _upsert_players(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO raw.nfl_player_gamelogs AS g (
            season, week, game_id, game_date_et, player_id, player_name,
            team_abbr, opponent_abbr, position, is_home,
            passing_yards, passing_tds, rushing_yards, rushing_tds,
            receiving_yards, receiving_tds, carries, targets, receptions,
            pass_attempts, routes_run, route_participation, snap_share,
            target_share, air_yards_share, wopr, receiving_air_yards,
            receiving_yards_after_catch, source, raw_json
        )
        VALUES %s
        ON CONFLICT (season, week, game_id, player_id, team_abbr) DO UPDATE SET
            game_date_et = EXCLUDED.game_date_et,
            player_name = EXCLUDED.player_name,
            opponent_abbr = EXCLUDED.opponent_abbr,
            position = EXCLUDED.position,
            is_home = EXCLUDED.is_home,
            passing_yards = EXCLUDED.passing_yards,
            passing_tds = EXCLUDED.passing_tds,
            rushing_yards = EXCLUDED.rushing_yards,
            rushing_tds = EXCLUDED.rushing_tds,
            receiving_yards = EXCLUDED.receiving_yards,
            receiving_tds = EXCLUDED.receiving_tds,
            carries = EXCLUDED.carries,
            targets = EXCLUDED.targets,
            receptions = EXCLUDED.receptions,
            pass_attempts = EXCLUDED.pass_attempts,
            routes_run = COALESCE(EXCLUDED.routes_run, g.routes_run),
            route_participation = COALESCE(EXCLUDED.route_participation, g.route_participation),
            snap_share = COALESCE(EXCLUDED.snap_share, g.snap_share),
            target_share = EXCLUDED.target_share,
            air_yards_share = EXCLUDED.air_yards_share,
            wopr = EXCLUDED.wopr,
            receiving_air_yards = EXCLUDED.receiving_air_yards,
            receiving_yards_after_catch = EXCLUDED.receiving_yards_after_catch,
            source = EXCLUDED.source,
            raw_json = EXCLUDED.raw_json,
            updated_at_utc = NOW()
        WHERE (
            g.game_date_et, g.player_name, g.opponent_abbr, g.position, g.is_home,
            g.passing_yards, g.passing_tds, g.rushing_yards, g.rushing_tds,
            g.receiving_yards, g.receiving_tds, g.carries, g.targets, g.receptions,
            g.pass_attempts, g.routes_run, g.route_participation, g.snap_share,
            g.target_share, g.air_yards_share, g.wopr, g.receiving_air_yards,
            g.receiving_yards_after_catch, g.source, g.raw_json
        ) IS DISTINCT FROM (
            EXCLUDED.game_date_et, EXCLUDED.player_name, EXCLUDED.opponent_abbr, EXCLUDED.position, EXCLUDED.is_home,
            EXCLUDED.passing_yards, EXCLUDED.passing_tds, EXCLUDED.rushing_yards, EXCLUDED.rushing_tds,
            EXCLUDED.receiving_yards, EXCLUDED.receiving_tds, EXCLUDED.carries, EXCLUDED.targets, EXCLUDED.receptions,
            EXCLUDED.pass_attempts,
            COALESCE(EXCLUDED.routes_run, g.routes_run),
            COALESCE(EXCLUDED.route_participation, g.route_participation),
            COALESCE(EXCLUDED.snap_share, g.snap_share),
            EXCLUDED.target_share, EXCLUDED.air_yards_share, EXCLUDED.wopr, EXCLUDED.receiving_air_yards,
            EXCLUDED.receiving_yards_after_catch, EXCLUDED.source, EXCLUDED.raw_json
        )
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=5000)
    conn.commit()
    return len(rows)


def _load_schedule_indexes(conn) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[int | None, int | None, str | None, str | None], dict[str, Any]],
]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT game_id, season, week, game_date_et, home_team_abbr, away_team_abbr, status
            FROM raw.nfl_games
        """)
        rows = [dict(row) for row in cur.fetchall()]
    by_game = {str(row["game_id"]): dict(row) for row in rows}
    by_matchup: dict[tuple[int | None, int | None, str | None, str | None], dict[str, Any]] = {}
    for row in rows:
        season = _clean_int(row.get("season"))
        week = _clean_int(row.get("week"))
        home = normalize_team(row.get("home_team_abbr"))
        away = normalize_team(row.get("away_team_abbr"))
        if home and away:
            by_matchup[(season, week, home, away)] = row
            by_matchup[(season, week, away, home)] = row
    return by_game, by_matchup


def import_data(cfg: ImportConfig) -> dict[str, Any]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        schedule_source = cfg.schedules_csv or cfg.schedules_url
        schedules = _read_csv(schedule_source)
        schedule_rows = _schedule_rows(schedules)
        schedules_upserted = _upsert_schedules(conn, schedule_rows)
        schedule_by_game, schedule_by_matchup = _load_schedule_indexes(conn)

        player_rows_total = 0
        sources: list[str] = []
        if cfg.schedules_only:
            player_sources = []
        elif cfg.player_stats_csv:
            player_sources = [cfg.player_stats_csv]
        else:
            player_sources = [
                cfg.player_stats_url_template.format(season=season)
                for season in cfg.seasons
            ]
        for source in player_sources:
            sources.append(source)
            df = _read_csv(source)
            rows = _player_rows(df, schedule_by_game, schedule_by_matchup)
            if df.empty or not rows:
                raise RuntimeError(f"No finalized NFL player rows from {source}")
            player_rows_total += _upsert_players(conn, rows)
    return {
        "status": "ok",
        "schedules_upserted": schedules_upserted,
        "player_rows_upserted": player_rows_total,
        "player_sources": sources,
    }


def _default_seasons() -> tuple[int, ...]:
    year = datetime.now().year
    return tuple(range(max(1999, year - 5), year + 1))


def _parse_seasons(value: str | None) -> tuple[int, ...]:
    if not value:
        return _default_seasons()
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return tuple(sorted(dict.fromkeys(out)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NFL player-game data from nflverse-style CSVs")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--seasons", default=None, help="Comma/range list, e.g. 2021-2025")
    parser.add_argument("--player-stats-csv", default=None, help="Local or URL player_stats CSV")
    parser.add_argument("--player-stats-url-template", default=DEFAULT_PLAYER_STATS_URL)
    parser.add_argument("--schedules-csv", default=None, help="Local or URL schedules CSV")
    parser.add_argument("--schedules-url", default=DEFAULT_SCHEDULES_URL)
    parser.add_argument("--schedules-only", action="store_true")
    args = parser.parse_args()
    cfg = ImportConfig(
        pg_dsn=args.pg_dsn,
        seasons=_parse_seasons(args.seasons),
        player_stats_csv=args.player_stats_csv,
        player_stats_url_template=args.player_stats_url_template,
        schedules_csv=args.schedules_csv,
        schedules_url=args.schedules_url,
        schedules_only=args.schedules_only,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    print(json.dumps(import_data(cfg), indent=2))


if __name__ == "__main__":
    main()
