"""Import NFL roster, depth chart, and injury context from nflverse."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.integrity import nfl_season
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import normalize_name, normalize_team
from nfl_pipeline.schema import changed_where, ensure_schema

ROSTER_CHANGED = changed_where("raw.nfl_rosters", (
    "player_name", "player_name_norm", "position", "depth_chart_position", "jersey_number",
    "roster_status", "status_description_abbr", "years_exp", "height", "weight",
    "birth_date", "college", "espn_id", "sportradar_id", "pfr_id",
    "source", "source_url", "raw_json",
))
DEPTH_CHANGED = changed_where("raw.nfl_depth_charts", (
    "week", "game_type", "snapshot_ts_utc", "team_abbr", "player_id",
    "player_name", "player_name_norm", "espn_id", "pos_grp", "pos_name",
    "pos_abb", "pos_slot", "pos_rank", "source", "source_url", "raw_json",
))
INJURY_CHANGED = changed_where("raw.nfl_injuries", (
    "report_primary_injury", "report_secondary_injury", "report_status",
    "practice_primary_injury", "practice_secondary_injury", "practice_status",
    "source", "source_url", "raw_json",
))

log = logging.getLogger("nfl_pipeline.import_context")

DEFAULT_ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
DEFAULT_DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.csv"
DEFAULT_INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"


@dataclass(frozen=True)
class ContextImportConfig:
    pg_dsn: str = PG_DSN
    seasons: tuple[int, ...] = ()
    roster_url_template: str = DEFAULT_ROSTER_URL
    depth_url_template: str = DEFAULT_DEPTH_URL
    injury_url_template: str = DEFAULT_INJURY_URL
    depth_latest_only: bool = True
    skip_rosters: bool = False
    skip_depth: bool = False
    skip_injuries: bool = False


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


def _clean_float(value: Any) -> float | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _clean_int(value: Any) -> int | None:
    value_f = _clean_float(value)
    if value_f is None:
        return None
    return int(value_f)


def _parse_date(value: Any) -> date | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        return None if pd.isna(parsed) else parsed.date()
    except Exception:
        return None


def _parse_ts(value: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        return None if pd.isna(parsed) else parsed.to_pydatetime()
    except Exception:
        return None


def _value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
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


def _row_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(source: str) -> tuple[pd.DataFrame, str | None]:
    try:
        log.info("Reading %s", source)
        return pd.read_csv(source, low_memory=False), None
    except Exception as exc:
        return pd.DataFrame(), f"{exc.__class__.__name__}: {exc}"


def _latest_depth_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dt" not in df.columns:
        return df
    parsed = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    if parsed.dropna().empty:
        return df
    latest = parsed.max()
    return df.loc[parsed == latest].copy()


def _player_id(row: pd.Series) -> str | None:
    return (
        _clean_text(_value(row, "gsis_id", "player_id"))
        or _clean_text(_value(row, "smart_id"))
        or _clean_text(_value(row, "pfr_id"))
        or _clean_text(_value(row, "sleeper_id"))
    )


def _player_name(row: pd.Series) -> str | None:
    return _clean_text(_value(row, "full_name", "player_name", "player_display_name", "football_name"))


def _roster_rows(df: pd.DataFrame, *, source_url: str) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        season = _clean_int(_value(row, "season"))
        team = normalize_team(_clean_text(_value(row, "team", "recent_team")))
        player_id = _player_id(row)
        player_name = _player_name(row)
        if season is None or not team or not player_id:
            continue
        rows.append((
            season,
            _clean_int(_value(row, "week")) or 0,
            _clean_text(_value(row, "game_type")) or "UNK",
            team,
            player_id,
            player_name,
            normalize_name(player_name),
            (_clean_text(_value(row, "position")) or "").upper() or None,
            _clean_text(_value(row, "depth_chart_position")),
            _clean_int(_value(row, "jersey_number")),
            _clean_text(_value(row, "status")),
            _clean_text(_value(row, "status_description_abbr")),
            _clean_float(_value(row, "years_exp")),
            _clean_float(_value(row, "height")),
            _clean_float(_value(row, "weight")),
            _parse_date(_value(row, "birth_date")),
            _clean_text(_value(row, "college")),
            _clean_text(_value(row, "espn_id")),
            _clean_text(_value(row, "sportradar_id")),
            _clean_text(_value(row, "pfr_id")),
            "nflverse_rosters",
            source_url,
            _json_subset(row, row.index),
        ))
    return rows


def _depth_rows(df: pd.DataFrame, *, season: int, source_url: str) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        row_season = _clean_int(_value(row, "season")) or season
        depth_week = _clean_int(_value(row, "week"))
        game_type = _clean_text(_value(row, "game_type"))
        team = normalize_team(_clean_text(_value(row, "team", "club_code")))
        player_name = _player_name(row)
        player_id = _player_id(row)
        snapshot_ts = _parse_ts(_value(row, "dt"))
        pos_abb = _clean_text(_value(row, "pos_abb", "depth_position", "position"))
        pos_slot = _clean_int(_value(row, "pos_slot"))
        pos_rank = _clean_int(_value(row, "pos_rank", "depth_team"))
        if pos_slot is None:
            pos_slot = pos_rank
        if not team or not player_name:
            continue
        row_hash = _row_hash(row_season, depth_week, game_type, snapshot_ts, team, player_id, player_name, pos_abb, pos_slot, pos_rank)
        rows.append((
            row_hash,
            row_season,
            depth_week,
            game_type,
            snapshot_ts,
            team,
            player_id,
            player_name,
            normalize_name(player_name),
            _clean_text(_value(row, "espn_id")),
            _clean_text(_value(row, "pos_grp", "formation")),
            _clean_text(_value(row, "pos_name", "position")),
            pos_abb,
            pos_slot,
            pos_rank,
            "nflverse_depth_charts",
            source_url,
            _json_subset(row, row.index),
        ))
    return rows


def _injury_rows(df: pd.DataFrame, *, source_url: str) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        season = _clean_int(_value(row, "season"))
        week = _clean_int(_value(row, "week"))
        team = normalize_team(_clean_text(_value(row, "team", "recent_team")))
        player_id = _player_id(row)
        player_name = _player_name(row)
        if season is None or not team or not player_name:
            continue
        row_hash = _row_hash(
            season,
            week,
            team,
            player_id,
            player_name,
            _clean_text(_value(row, "report_status")),
            _clean_text(_value(row, "practice_status")),
            _clean_text(_value(row, "report_primary_injury")),
        )
        rows.append((
            row_hash,
            season,
            _clean_text(_value(row, "season_type")),
            _clean_text(_value(row, "game_type")),
            week,
            team,
            player_id,
            player_name,
            normalize_name(player_name),
            (_clean_text(_value(row, "position")) or "").upper() or None,
            _clean_text(_value(row, "report_primary_injury")),
            _clean_text(_value(row, "report_secondary_injury")),
            _clean_text(_value(row, "report_status")),
            _clean_text(_value(row, "practice_primary_injury")),
            _clean_text(_value(row, "practice_secondary_injury")),
            _clean_text(_value(row, "practice_status")),
            "nflverse_injuries",
            source_url,
            _json_subset(row, row.index),
        ))
    return rows


def _upsert_rosters(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO raw.nfl_rosters (
                season, week, game_type, team_abbr, player_id, player_name,
                player_name_norm, position, depth_chart_position, jersey_number,
                roster_status, status_description_abbr, years_exp, height, weight,
                birth_date, college, espn_id, sportradar_id, pfr_id,
                source, source_url, raw_json
            )
            VALUES %s
            ON CONFLICT (season, week, game_type, team_abbr, player_id) DO UPDATE SET
                player_name = EXCLUDED.player_name,
                player_name_norm = EXCLUDED.player_name_norm,
                position = EXCLUDED.position,
                depth_chart_position = EXCLUDED.depth_chart_position,
                jersey_number = EXCLUDED.jersey_number,
                roster_status = EXCLUDED.roster_status,
                status_description_abbr = EXCLUDED.status_description_abbr,
                years_exp = EXCLUDED.years_exp,
                height = EXCLUDED.height,
                weight = EXCLUDED.weight,
                birth_date = EXCLUDED.birth_date,
                college = EXCLUDED.college,
                espn_id = EXCLUDED.espn_id,
                sportradar_id = EXCLUDED.sportradar_id,
                pfr_id = EXCLUDED.pfr_id,
                source = EXCLUDED.source,
                source_url = EXCLUDED.source_url,
                raw_json = EXCLUDED.raw_json,
                updated_at_utc = NOW()
            WHERE {ROSTER_CHANGED}
            """,
            rows,
            page_size=5000,
        )
    conn.commit()
    return len(rows)


def _upsert_depth(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    rows = list({str(row[0]): row for row in rows if row and row[0]}.values())
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO raw.nfl_depth_charts (
                row_hash, season, week, game_type, snapshot_ts_utc, team_abbr, player_id,
                player_name, player_name_norm, espn_id, pos_grp, pos_name,
                pos_abb, pos_slot, pos_rank, source, source_url, raw_json
            )
            VALUES %s
            ON CONFLICT (row_hash) DO UPDATE SET
                week = EXCLUDED.week,
                game_type = EXCLUDED.game_type,
                snapshot_ts_utc = EXCLUDED.snapshot_ts_utc,
                team_abbr = EXCLUDED.team_abbr,
                player_id = EXCLUDED.player_id,
                player_name = EXCLUDED.player_name,
                player_name_norm = EXCLUDED.player_name_norm,
                espn_id = EXCLUDED.espn_id,
                pos_grp = EXCLUDED.pos_grp,
                pos_name = EXCLUDED.pos_name,
                pos_abb = EXCLUDED.pos_abb,
                pos_slot = EXCLUDED.pos_slot,
                pos_rank = EXCLUDED.pos_rank,
                source = EXCLUDED.source,
                source_url = EXCLUDED.source_url,
                raw_json = EXCLUDED.raw_json,
                updated_at_utc = NOW()
            WHERE {DEPTH_CHANGED}
            """,
            rows,
            page_size=5000,
        )
    conn.commit()
    return len(rows)


def _upsert_injuries(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO raw.nfl_injuries (
                row_hash, season, season_type, game_type, week, team_abbr,
                player_id, player_name, player_name_norm, position,
                report_primary_injury, report_secondary_injury, report_status,
                practice_primary_injury, practice_secondary_injury, practice_status,
                source, source_url, raw_json
            )
            VALUES %s
            ON CONFLICT (row_hash) DO UPDATE SET
                report_primary_injury = EXCLUDED.report_primary_injury,
                report_secondary_injury = EXCLUDED.report_secondary_injury,
                report_status = EXCLUDED.report_status,
                practice_primary_injury = EXCLUDED.practice_primary_injury,
                practice_secondary_injury = EXCLUDED.practice_secondary_injury,
                practice_status = EXCLUDED.practice_status,
                source = EXCLUDED.source,
                source_url = EXCLUDED.source_url,
                raw_json = EXCLUDED.raw_json,
                updated_at_utc = NOW()
            WHERE {INJURY_CHANGED}
            """,
            rows,
            page_size=5000,
        )
    conn.commit()
    return len(rows)


def import_context(cfg: ContextImportConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok", "seasons": list(cfg.seasons), "sources": []}
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        roster_rows = depth_rows = injury_rows = 0
        for season in cfg.seasons:
            if not cfg.skip_rosters:
                source = cfg.roster_url_template.format(season=season)
                df, err = _read_csv(source)
                result["sources"].append({"kind": "roster", "season": season, "source": source, "error": err, "rows_read": int(len(df))})
                if err is None:
                    roster_rows += _upsert_rosters(conn, _roster_rows(df, source_url=source))
            if not cfg.skip_depth:
                source = cfg.depth_url_template.format(season=season)
                df, err = _read_csv(source)
                rows_read = int(len(df))
                if err is None:
                    if cfg.depth_latest_only:
                        df = _latest_depth_snapshot(df)
                    result["sources"].append({
                        "kind": "depth",
                        "season": season,
                        "source": source,
                        "error": err,
                        "rows_read": rows_read,
                        "rows_selected": int(len(df)),
                        "latest_only": cfg.depth_latest_only,
                    })
                    depth_rows += _upsert_depth(conn, _depth_rows(df, season=season, source_url=source))
                else:
                    result["sources"].append({"kind": "depth", "season": season, "source": source, "error": err, "rows_read": rows_read})
            if not cfg.skip_injuries:
                source = cfg.injury_url_template.format(season=season)
                df, err = _read_csv(source)
                result["sources"].append({"kind": "injury", "season": season, "source": source, "error": err, "rows_read": int(len(df))})
                if err is None:
                    injury_rows += _upsert_injuries(conn, _injury_rows(df, source_url=source))
    result.update({
        "roster_rows_upserted": roster_rows,
        "depth_rows_upserted": depth_rows,
        "injury_rows_upserted": injury_rows,
    })
    return result


def _default_seasons() -> tuple[int, ...]:
    return (nfl_season(),)


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
    parser = argparse.ArgumentParser(description="Import NFL roster/depth/injury context from nflverse")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--seasons", default=None, help="Comma/range list, default current year")
    parser.add_argument("--skip-rosters", action="store_true")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-injuries", action="store_true")
    parser.add_argument("--all-depth-snapshots", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = import_context(ContextImportConfig(
        pg_dsn=args.pg_dsn,
        seasons=_parse_seasons(args.seasons),
        depth_latest_only=not args.all_depth_snapshots,
        skip_rosters=args.skip_rosters,
        skip_depth=args.skip_depth,
        skip_injuries=args.skip_injuries,
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
