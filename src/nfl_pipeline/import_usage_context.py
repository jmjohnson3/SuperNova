"""Import NFL snap-count and red-zone usage context from nflverse."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.integrity import nfl_season
from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import normalize_name, normalize_team
from nfl_pipeline.schema import ensure_schema
from nfl_pipeline.integrity import atomic_json

log = logging.getLogger("nfl_pipeline.import_usage_context")

DEFAULT_SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
DEFAULT_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
DEFAULT_PARTICIPATION_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.csv"
DEFAULT_PLAYER_STATS_USAGE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
DEFAULT_ADVANCED_USAGE_URL = os.getenv("NFL_ADVANCED_USAGE_URL_TEMPLATE", DEFAULT_PLAYER_STATS_USAGE_URL)


@dataclass(frozen=True)
class UsageImportConfig:
    pg_dsn: str = PG_DSN
    seasons: tuple[int, ...] = ()
    snap_counts_url_template: str = DEFAULT_SNAP_COUNTS_URL
    pbp_url_template: str = DEFAULT_PBP_URL
    participation_url_template: str = DEFAULT_PARTICIPATION_URL
    advanced_usage_url_template: str = DEFAULT_ADVANCED_USAGE_URL
    skip_snap_counts: bool = False
    skip_pbp: bool = False
    skip_participation: bool = False
    skip_advanced_usage: bool = False
    skip_schema: bool = False


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
        out = float(text.rstrip("%"))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean_pct(value: Any) -> float | None:
    out = _clean_float(value)
    if out is None:
        return None
    return out / 100.0 if out > 1.0 else out


def _clean_int(value: Any) -> int | None:
    out = _clean_float(value)
    return None if out is None else int(out)


def _value(row: pd.Series, *names: str) -> Any:
    lowered = {str(col).lower(): col for col in row.index}
    for name in names:
        actual = name if name in row.index else lowered.get(name.lower())
        if actual is not None and _clean_text(row.get(actual)) is not None:
            return row.get(actual)
    return None


def _read_csv(source: str, **kwargs) -> tuple[pd.DataFrame, str | None]:
    try:
        log.info("Reading %s", source)
        return pd.read_csv(source, low_memory=False, **kwargs), None
    except Exception as exc:
        return pd.DataFrame(), f"{exc.__class__.__name__}: {exc}"


def _parse_seasons(value: str | None) -> tuple[int, ...]:
    if not value:
        year = nfl_season()
        return tuple(range(max(1999, year - 5), year + 1))
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


def _roster_maps(conn, seasons: tuple[int, ...]) -> tuple[dict[tuple[int, str], str], dict[tuple[int, str, str], str]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT season, player_id, player_name_norm, team_abbr, pfr_id
            FROM raw.nfl_rosters
            WHERE season = ANY(%s)
              AND player_id IS NOT NULL
            """,
            (list(seasons),),
        )
        rows = [dict(row) for row in cur.fetchall()]
    by_pfr: dict[tuple[int, str], str] = {}
    by_name_team: dict[tuple[int, str, str], str] = {}
    for row in rows:
        season = int(row["season"])
        player_id = str(row["player_id"])
        pfr_id = _clean_text(row.get("pfr_id"))
        if pfr_id:
            by_pfr[(season, pfr_id)] = player_id
        name = _clean_text(row.get("player_name_norm"))
        team = normalize_team(row.get("team_abbr"))
        if name and team:
            by_name_team[(season, name, team)] = player_id
    return by_pfr, by_name_team


def _update_snap_counts(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            UPDATE raw.nfl_player_gamelogs g
            SET
                offense_snaps = v.offense_snaps,
                offense_snap_share = v.offense_snap_share,
                snap_share = COALESCE(g.snap_share, v.offense_snap_share),
                updated_at_utc = NOW()
            FROM (VALUES %s) AS v(
                season, week, game_id, player_id, team_abbr,
                offense_snaps, offense_snap_share
            )
            WHERE g.season = v.season
              AND g.week = v.week
              AND g.game_id = v.game_id
              AND g.player_id = v.player_id
              AND g.team_abbr = v.team_abbr
              AND (UPPER(COALESCE(g.position, '')) IN ('QB', 'RB', 'FB', 'WR', 'TE')
                   OR v.offense_snaps > 0)
              AND (g.offense_snaps, g.offense_snap_share, g.snap_share) IS DISTINCT FROM (
                  v.offense_snaps::numeric, v.offense_snap_share::numeric,
                  COALESCE(g.snap_share, v.offense_snap_share::numeric))
            """,
            rows,
            page_size=5000,
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _update_advanced_usage(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            UPDATE raw.nfl_player_gamelogs g
            SET
                routes_run = COALESCE(v.routes_run, g.routes_run),
                route_participation = COALESCE(v.route_participation, g.route_participation),
                target_share = COALESCE(v.target_share, g.target_share),
                air_yards_share = COALESCE(v.air_yards_share, g.air_yards_share),
                wopr = COALESCE(v.wopr, g.wopr),
                receiving_air_yards = COALESCE(v.receiving_air_yards, g.receiving_air_yards),
                receiving_yards_after_catch = COALESCE(v.receiving_yards_after_catch, g.receiving_yards_after_catch),
                targets_per_route_run = COALESCE(v.targets_per_route_run, g.targets_per_route_run),
                yards_per_route_run = COALESCE(v.yards_per_route_run, g.yards_per_route_run),
                first_read_targets = COALESCE(v.first_read_targets, g.first_read_targets),
                first_read_target_share = COALESCE(v.first_read_target_share, g.first_read_target_share),
                end_zone_targets = COALESCE(v.end_zone_targets, g.end_zone_targets),
                end_zone_target_share = COALESCE(v.end_zone_target_share, g.end_zone_target_share),
                updated_at_utc = NOW()
            FROM (VALUES %s) AS v(
                season, week, game_id, player_id, team_abbr,
                routes_run, route_participation, target_share, air_yards_share,
                wopr, receiving_air_yards, receiving_yards_after_catch,
                targets_per_route_run, yards_per_route_run,
                first_read_targets, first_read_target_share,
                end_zone_targets, end_zone_target_share
            )
            WHERE g.season = v.season
              AND g.week = v.week
              AND g.game_id = v.game_id
              AND g.player_id = v.player_id
              AND g.team_abbr = v.team_abbr
              AND UPPER(COALESCE(g.position, '')) IN ('RB', 'FB', 'WR', 'TE')
              AND (g.routes_run, g.route_participation, g.target_share, g.air_yards_share,
                   g.wopr, g.receiving_air_yards, g.receiving_yards_after_catch,
                   g.targets_per_route_run, g.yards_per_route_run,
                   g.first_read_targets, g.first_read_target_share,
                   g.end_zone_targets, g.end_zone_target_share)
                  IS DISTINCT FROM (
                   COALESCE(v.routes_run, g.routes_run),
                   COALESCE(v.route_participation, g.route_participation),
                   COALESCE(v.target_share, g.target_share),
                   COALESCE(v.air_yards_share, g.air_yards_share),
                   COALESCE(v.wopr, g.wopr),
                   COALESCE(v.receiving_air_yards, g.receiving_air_yards),
                   COALESCE(v.receiving_yards_after_catch, g.receiving_yards_after_catch),
                   COALESCE(v.targets_per_route_run, g.targets_per_route_run),
                   COALESCE(v.yards_per_route_run, g.yards_per_route_run),
                   COALESCE(v.first_read_targets, g.first_read_targets),
                   COALESCE(v.first_read_target_share, g.first_read_target_share),
                   COALESCE(v.end_zone_targets, g.end_zone_targets),
                   COALESCE(v.end_zone_target_share, g.end_zone_target_share))
            """,
            rows,
            page_size=5000,
            template='(%s::integer,%s::integer,%s::text,%s::text,%s::text,' + ','.join(['%s::numeric'] * 13) + ')',
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _update_pass_route_opportunities(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            UPDATE raw.nfl_player_gamelogs g
            SET
                pass_route_opportunities = v.pass_route_opportunities,
                pass_route_opportunity_share = v.pass_route_opportunity_share,
                updated_at_utc = NOW()
            FROM (VALUES %s) AS v(
                season, week, game_id, player_id, team_abbr,
                pass_route_opportunities, pass_route_opportunity_share
            )
            WHERE g.season = v.season
              AND g.week = v.week
              AND g.game_id = v.game_id
              AND g.player_id = v.player_id
              AND g.team_abbr = v.team_abbr
              AND (g.pass_route_opportunities, g.pass_route_opportunity_share) IS DISTINCT FROM (
                  v.pass_route_opportunities::numeric, v.pass_route_opportunity_share::numeric)
            """,
            rows,
            page_size=5000,
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _clear_non_skill_route_usage(conn, season: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE raw.nfl_player_gamelogs
            SET
                routes_run = NULL,
                route_participation = NULL,
                pass_route_opportunities = NULL,
                pass_route_opportunity_share = NULL,
                targets_per_route_run = NULL,
                yards_per_route_run = NULL,
                first_read_targets = NULL,
                first_read_target_share = NULL,
                end_zone_targets = NULL,
                end_zone_target_share = NULL,
                updated_at_utc = NOW()
            WHERE season = %s
              AND UPPER(COALESCE(position, '')) NOT IN ('RB', 'FB', 'WR', 'TE')
              AND (
                  routes_run IS NOT NULL
                  OR route_participation IS NOT NULL
                  OR pass_route_opportunities IS NOT NULL
                  OR pass_route_opportunity_share IS NOT NULL
                  OR targets_per_route_run IS NOT NULL
                  OR yards_per_route_run IS NOT NULL
                  OR first_read_targets IS NOT NULL
                  OR first_read_target_share IS NOT NULL
                  OR end_zone_targets IS NOT NULL
                  OR end_zone_target_share IS NOT NULL
              )
            """,
            (season,),
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _snap_rows(df: pd.DataFrame, by_pfr: dict[tuple[int, str], str], by_name_team: dict[tuple[int, str, str], str]) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        season = _clean_int(row.get("season"))
        week = _clean_int(row.get("week"))
        game_id = _clean_text(row.get("game_id"))
        team = normalize_team(row.get("team"))
        if season is None or week is None or not game_id or not team:
            continue
        pfr_id = _clean_text(row.get("pfr_player_id"))
        player_id = by_pfr.get((season, pfr_id or ""))
        if not player_id:
            player_id = by_name_team.get((season, normalize_name(_clean_text(row.get("player"))), team))
        if not player_id:
            continue
        rows.append((
            season,
            week,
            game_id,
            player_id,
            team,
            _clean_float(row.get("offense_snaps")),
            _clean_pct(row.get("offense_pct")),
        ))
    return rows


def _advanced_usage_rows(
    df: pd.DataFrame,
    by_name_team: dict[tuple[int, str, str], str],
) -> list[tuple]:
    rows: list[tuple] = []
    for _, row in df.iterrows():
        season = _clean_int(_value(row, "season", "year"))
        week = _clean_int(_value(row, "week", "game_week"))
        game_id = _clean_text(_value(row, "game_id", "nflverse_game_id"))
        team = normalize_team(_value(row, "team", "team_abbr", "recent_team", "posteam"))
        player_id = _clean_text(_value(row, "player_id", "gsis_id", "nfl_id"))
        if not player_id:
            player_name = normalize_name(_clean_text(_value(row, "player", "player_name", "name")))
            if season is not None and player_name and team:
                player_id = by_name_team.get((season, player_name, team))
        if season is None or week is None or not game_id or not team or not player_id:
            continue
        routes = _clean_float(_value(row, "routes_run", "routes", "rr"))
        if routes is not None and routes < 0:
            routes = None
        targets = _clean_float(_value(row, "targets", "tgt"))
        rec_yards = _clean_float(_value(row, "receiving_yards", "rec_yards", "yards"))
        tprr = _clean_float(_value(row, "targets_per_route_run", "tprr"))
        yprr = _clean_float(_value(row, "yards_per_route_run", "yprr"))
        if tprr is None and routes and routes > 0 and targets is not None:
            tprr = targets / routes
        if yprr is None and routes and routes > 0 and rec_yards is not None:
            yprr = rec_yards / routes
        rows.append((
            season,
            week,
            game_id,
            player_id,
            team,
            routes,
            _clean_pct(_value(row, "route_participation", "route_participation_pct", "route_share", "route_pct")),
            _clean_pct(_value(row, "target_share", "target_share_pct", "tgt_share")),
            _clean_pct(_value(row, "air_yards_share", "air_yards_share_pct", "air_share")),
            _clean_float(_value(row, "wopr")),
            _clean_float(_value(row, "receiving_air_yards", "air_yards")),
            _clean_float(_value(row, "receiving_yards_after_catch", "yac")),
            tprr,
            yprr,
            _clean_float(_value(row, "first_read_targets", "first_read_tgts")),
            _clean_pct(_value(row, "first_read_target_share", "first_read_share")),
            _clean_float(_value(row, "end_zone_targets", "endzone_targets", "ez_targets")),
            _clean_pct(_value(row, "end_zone_target_share", "endzone_target_share", "ez_target_share")),
        ))
    return rows


def advanced_usage_health(df: pd.DataFrame, rows: list[tuple]) -> dict[str, Any]:
    """A successful weekly-stats download is not evidence of measured routes."""
    aliases = {
        'routes_run': ('routes_run', 'routes', 'rr'),
        'first_read_targets': ('first_read_targets', 'first_read_tgts'),
        'target_share': ('target_share', 'target_share_pct', 'tgt_share'),
        'air_yards_share': ('air_yards_share', 'air_yards_share_pct', 'air_share'),
    }
    columns = {str(c).lower() for c in df.columns}
    indices = {'routes_run': 5, 'target_share': 7, 'air_yards_share': 8, 'first_read_targets': 14}
    return {
        'capabilities': {field: any(alias in columns for alias in names) for field, names in aliases.items()},
        'matched_nonmissing': {field: sum(r[index] is not None for r in rows) for field, index in indices.items()},
        'routes_are_proxies': False,
        'route_evidence': 'measured_values_present' if any(r[5] is not None for r in rows) else 'missing',
        'missing_routes_action': 'keep_proxy_label; configure a measured-route source before claiming true routes',
    }


def _split_semicolon(value: Any) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _season_week_from_game_id(game_id: str) -> tuple[int | None, int | None]:
    parts = str(game_id or "").split("_")
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None, None


def _participation_rows(df: pd.DataFrame) -> list[tuple]:
    if df.empty or "offense_players" not in df.columns:
        return []
    frame = df.copy()
    game_col = "nflverse_game_id" if "nflverse_game_id" in frame.columns else "game_id"
    frame["game_id"] = frame[game_col].astype(str)
    if "season" not in frame.columns or "week" not in frame.columns:
        parsed = frame["game_id"].map(_season_week_from_game_id)
        frame["season"] = [item[0] for item in parsed]
        frame["week"] = [item[1] for item in parsed]
    frame["team_abbr"] = frame.get("possession_team", pd.Series(index=frame.index)).map(normalize_team)
    pass_like = pd.Series(False, index=frame.index)
    if "route" in frame.columns:
        pass_like |= frame["route"].notna()
    if "time_to_throw" in frame.columns:
        pass_like |= pd.to_numeric(frame["time_to_throw"], errors="coerce").notna()
    if "number_of_pass_rushers" in frame.columns:
        pass_like |= pd.to_numeric(frame["number_of_pass_rushers"], errors="coerce").fillna(0).gt(0)
    frame = frame.loc[pass_like & frame["team_abbr"].notna()].copy()
    if frame.empty:
        return []

    team_dropbacks = (
        frame.groupby(["season", "week", "game_id", "team_abbr"], dropna=False)
        .size()
        .rename("team_pass_like_plays")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    eligible_positions = {"RB", "FB", "WR", "TE"}
    for _, row in frame.iterrows():
        players = _split_semicolon(row.get("offense_players"))
        positions = _split_semicolon(row.get("offense_positions"))
        if not players:
            continue
        for idx, player_id in enumerate(players):
            pos_text = str(positions[idx] if idx < len(positions) else "").upper()
            if positions and pos_text not in eligible_positions:
                continue
            rows.append({
                "season": _clean_int(row.get("season")),
                "week": _clean_int(row.get("week")),
                "game_id": _clean_text(row.get("game_id")),
                "team_abbr": normalize_team(row.get("team_abbr")),
                "player_id": _clean_text(player_id),
                "pass_route_opportunities": 1.0,
            })
    if not rows:
        return []
    usage = pd.DataFrame(rows)
    grouped = (
        usage.groupby(["season", "week", "game_id", "team_abbr", "player_id"], dropna=False)["pass_route_opportunities"]
        .sum()
        .reset_index()
    )
    grouped = grouped.merge(team_dropbacks, on=["season", "week", "game_id", "team_abbr"], how="left")
    grouped["pass_route_opportunity_share"] = grouped["pass_route_opportunities"] / grouped["team_pass_like_plays"].replace(0, pd.NA)
    out: list[tuple] = []
    for _, row in grouped.iterrows():
        out.append((
            _clean_int(row.get("season")),
            _clean_int(row.get("week")),
            _clean_text(row.get("game_id")),
            _clean_text(row.get("player_id")),
            normalize_team(row.get("team_abbr")),
            _clean_float(row.get("pass_route_opportunities")) or 0.0,
            _clean_pct(row.get("pass_route_opportunity_share")),
        ))
    return [row for row in out if all(row[:5])]


def _update_red_zone(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            UPDATE raw.nfl_player_gamelogs g
            SET
                red_zone_carries = v.red_zone_carries,
                red_zone_targets = v.red_zone_targets,
                red_zone_receptions = v.red_zone_receptions,
                red_zone_pass_attempts = v.red_zone_pass_attempts,
                red_zone_pass_tds = v.red_zone_pass_tds,
                red_zone_rush_tds = v.red_zone_rush_tds,
                red_zone_rec_tds = v.red_zone_rec_tds,
                red_zone_touches = v.red_zone_touches,
                goal_line_carries = v.goal_line_carries,
                goal_line_targets = v.goal_line_targets,
                updated_at_utc = NOW()
            FROM (VALUES %s) AS v(
                season, week, game_id, player_id, team_abbr,
                red_zone_carries, red_zone_targets, red_zone_receptions,
                red_zone_pass_attempts, red_zone_pass_tds, red_zone_rush_tds,
                red_zone_rec_tds, red_zone_touches, goal_line_carries,
                goal_line_targets
            )
            WHERE g.season = v.season
              AND g.week = v.week
              AND g.game_id = v.game_id
              AND g.player_id = v.player_id
              AND g.team_abbr = v.team_abbr
              AND (g.red_zone_carries, g.red_zone_targets, g.red_zone_receptions,
                   g.red_zone_pass_attempts, g.red_zone_pass_tds, g.red_zone_rush_tds,
                   g.red_zone_rec_tds, g.red_zone_touches, g.goal_line_carries, g.goal_line_targets)
                  IS DISTINCT FROM (
                   v.red_zone_carries::numeric, v.red_zone_targets::numeric, v.red_zone_receptions::numeric,
                   v.red_zone_pass_attempts::numeric, v.red_zone_pass_tds::numeric, v.red_zone_rush_tds::numeric,
                   v.red_zone_rec_tds::numeric, v.red_zone_touches::numeric, v.goal_line_carries::numeric,
                   v.goal_line_targets::numeric)
            """,
            rows,
            page_size=5000,
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _mark_red_zone_zeros(conn, season: int) -> int:
    columns = (
        "red_zone_carries",
        "red_zone_targets",
        "red_zone_receptions",
        "red_zone_pass_attempts",
        "red_zone_pass_tds",
        "red_zone_rush_tds",
        "red_zone_rec_tds",
        "red_zone_touches",
        "goal_line_carries",
        "goal_line_targets",
    )
    set_sql = ", ".join(f"{col} = COALESCE({col}, 0)" for col in columns)
    where_sql = " OR ".join(f"{col} IS NULL" for col in columns)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE raw.nfl_player_gamelogs
            SET {set_sql}, updated_at_utc = NOW()
            WHERE season = %s
              AND ({where_sql})
            """,
            (season,),
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)


def _pbp_player_usage_rows(df: pd.DataFrame) -> list[tuple]:
    """Build current player-game stat and usage rows from nflverse PBP.

    The weekly player_stats release can lag or be unavailable early in a slate.
    PBP usually lands first, so this keeps current-season target share, air
    yards, red-zone and goal-line context moving without waiting for the
    aggregated file.
    """
    if df.empty or "game_id" not in df.columns:
        return []
    pbp = df.copy()
    pbp["season"] = pd.to_numeric(pbp.get("season"), errors="coerce")
    pbp["week"] = pd.to_numeric(pbp.get("week"), errors="coerce")
    pbp["game_id"] = pbp.get("game_id", pd.Series(index=pbp.index)).astype(str)
    pbp["team_abbr"] = pbp.get("posteam", pd.Series(index=pbp.index)).map(normalize_team)
    pbp["opponent_abbr"] = pbp.get("defteam", pd.Series(index=pbp.index)).map(normalize_team)
    pbp["yardline_100"] = pd.to_numeric(pbp.get("yardline_100"), errors="coerce")
    pbp["pass_attempt_f"] = _flag(pbp.get("pass_attempt", pd.Series(index=pbp.index)))
    pbp["rush_attempt_f"] = _flag(pbp.get("rush_attempt", pd.Series(index=pbp.index)))
    pbp["complete_pass_f"] = _flag(pbp.get("complete_pass", pd.Series(index=pbp.index)))
    pbp["pass_td_f"] = _flag(pbp.get("pass_touchdown", pd.Series(index=pbp.index)))
    pbp["rush_td_f"] = _flag(pbp.get("rush_touchdown", pd.Series(index=pbp.index)))
    pbp["air_yards_f"] = pd.to_numeric(pbp.get("air_yards"), errors="coerce")
    pbp["yac_f"] = pd.to_numeric(pbp.get("yards_after_catch"), errors="coerce")
    pbp["passing_yards_f"] = pd.to_numeric(pbp.get("passing_yards"), errors="coerce").fillna(0.0)
    pbp["rushing_yards_f"] = pd.to_numeric(pbp.get("rushing_yards"), errors="coerce").fillna(0.0)
    pbp["receiving_yards_f"] = pd.to_numeric(pbp.get("receiving_yards"), errors="coerce").fillna(0.0)
    pbp = pbp.loc[pbp["season"].notna() & pbp["week"].notna() & pbp["team_abbr"].notna()].copy()
    if pbp.empty:
        return []

    records: list[pd.DataFrame] = []
    if "passer_player_id" in pbp.columns:
        passers = pbp.loc[pbp["pass_attempt_f"] > 0].copy()
        passers["player_id"] = passers["passer_player_id"]
        passers["player_name"] = passers.get("passer_player_name")
        passers = passers.loc[passers["player_id"].notna()]
        if not passers.empty:
            passers["passing_yards"] = passers["passing_yards_f"]
            passers["passing_tds"] = passers["pass_td_f"]
            passers["pass_attempts"] = 1.0
            passers["red_zone_pass_attempts"] = ((passers["yardline_100"].notna()) & (passers["yardline_100"] <= 20)).astype(float)
            passers["red_zone_pass_tds"] = passers["red_zone_pass_attempts"] * passers["pass_td_f"]
            records.append(passers[[
                "season", "week", "game_id", "player_id", "player_name",
                "team_abbr", "opponent_abbr", "passing_yards", "passing_tds", "pass_attempts",
                "red_zone_pass_attempts", "red_zone_pass_tds",
            ]])

    if "rusher_player_id" in pbp.columns:
        rushers = pbp.loc[pbp["rush_attempt_f"] > 0].copy()
        rushers["player_id"] = rushers["rusher_player_id"]
        rushers["player_name"] = rushers.get("rusher_player_name")
        rushers = rushers.loc[rushers["player_id"].notna()]
        if not rushers.empty:
            rushers["rushing_yards"] = rushers["rushing_yards_f"]
            rushers["rushing_tds"] = rushers["rush_td_f"]
            rushers["carries"] = 1.0
            rushers["red_zone_carries"] = ((rushers["yardline_100"].notna()) & (rushers["yardline_100"] <= 20)).astype(float)
            rushers["goal_line_carries"] = ((rushers["yardline_100"].notna()) & (rushers["yardline_100"] <= 5)).astype(float)
            records.append(rushers[[
                "season", "week", "game_id", "player_id", "player_name",
                "team_abbr", "opponent_abbr", "rushing_yards", "rushing_tds", "carries",
                "red_zone_carries", "goal_line_carries",
            ]])

    if "receiver_player_id" in pbp.columns:
        receivers = pbp.loc[(pbp["pass_attempt_f"] > 0) & pbp["receiver_player_id"].notna()].copy()
        receivers["player_id"] = receivers["receiver_player_id"]
        receivers["player_name"] = receivers.get("receiver_player_name")
        if not receivers.empty:
            receivers["receiving_yards"] = receivers["receiving_yards_f"]
            receivers["receiving_tds"] = receivers["pass_td_f"] * (
                receivers.get("td_player_id", pd.Series(index=receivers.index)).astype(str)
                == receivers["player_id"].astype(str)
            ).astype(float)
            receivers["targets"] = 1.0
            receivers["receptions"] = receivers["complete_pass_f"]
            receivers["receiving_air_yards"] = receivers["air_yards_f"].fillna(0.0)
            receivers["receiving_yards_after_catch"] = receivers["yac_f"].fillna(0.0)
            receivers["red_zone_targets"] = ((receivers["yardline_100"].notna()) & (receivers["yardline_100"] <= 20)).astype(float)
            receivers["red_zone_receptions"] = receivers["red_zone_targets"] * receivers["complete_pass_f"]
            receivers["goal_line_targets"] = ((receivers["yardline_100"].notna()) & (receivers["yardline_100"] <= 5)).astype(float)
            receivers["end_zone_targets"] = (
                (receivers["yardline_100"].notna())
                & (receivers["air_yards_f"].notna())
                & (receivers["air_yards_f"] >= receivers["yardline_100"])
            ).astype(float)
            records.append(receivers[[
                "season", "week", "game_id", "player_id", "player_name",
                "team_abbr", "opponent_abbr", "receiving_yards", "receiving_tds",
                "targets", "receptions", "receiving_air_yards", "receiving_yards_after_catch",
                "red_zone_targets", "red_zone_receptions", "goal_line_targets", "end_zone_targets",
            ]])

    if not records:
        return []

    usage = pd.concat(records, ignore_index=True, sort=False)
    stat_cols = [
        "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "carries", "targets", "receptions",
        "pass_attempts", "receiving_air_yards", "receiving_yards_after_catch",
        "red_zone_carries", "red_zone_targets", "red_zone_receptions",
        "red_zone_pass_attempts", "red_zone_pass_tds", "goal_line_carries",
        "goal_line_targets", "end_zone_targets",
    ]
    for col in stat_cols:
        if col not in usage.columns:
            usage[col] = 0.0
        usage[col] = pd.to_numeric(usage[col], errors="coerce").fillna(0.0)
    grouped = (
        usage.groupby(["season", "week", "game_id", "player_id", "team_abbr"], dropna=False)
        .agg({
            "player_name": "last",
            "opponent_abbr": "last",
            **{col: "sum" for col in stat_cols},
        })
        .reset_index()
    )
    team_pass_attempts = grouped.groupby(["season", "week", "game_id", "team_abbr"], dropna=False)["pass_attempts"].transform("sum")
    team_air_yards = grouped.groupby(["season", "week", "game_id", "team_abbr"], dropna=False)["receiving_air_yards"].transform("sum")
    grouped["target_share"] = grouped["targets"] / team_pass_attempts.replace(0.0, pd.NA)
    grouped["air_yards_share"] = grouped["receiving_air_yards"] / team_air_yards.replace(0.0, pd.NA)
    grouped["wopr"] = 1.5 * grouped["target_share"].fillna(0.0) + 0.7 * grouped["air_yards_share"].fillna(0.0)
    grouped["red_zone_touches"] = grouped["red_zone_carries"] + grouped["red_zone_targets"]
    grouped["red_zone_rush_tds"] = grouped["rushing_tds"]
    grouped["red_zone_rec_tds"] = grouped["receiving_tds"]

    rows: list[tuple] = []
    for _, row in grouped.iterrows():
        rows.append((
            _clean_int(row.get("season")),
            _clean_int(row.get("week")),
            _clean_text(row.get("game_id")),
            _clean_text(row.get("player_id")),
            normalize_team(row.get("team_abbr")),
            _clean_text(row.get("player_name")),
            normalize_team(row.get("opponent_abbr")),
            _clean_float(row.get("passing_yards")) or 0.0,
            _clean_float(row.get("passing_tds")) or 0.0,
            _clean_float(row.get("rushing_yards")) or 0.0,
            _clean_float(row.get("rushing_tds")) or 0.0,
            _clean_float(row.get("receiving_yards")) or 0.0,
            _clean_float(row.get("receiving_tds")) or 0.0,
            _clean_float(row.get("carries")) or 0.0,
            _clean_float(row.get("targets")) or 0.0,
            _clean_float(row.get("receptions")) or 0.0,
            _clean_float(row.get("pass_attempts")) or 0.0,
            _clean_pct(row.get("target_share")),
            _clean_pct(row.get("air_yards_share")),
            _clean_float(row.get("wopr")),
            _clean_float(row.get("receiving_air_yards")) or 0.0,
            _clean_float(row.get("receiving_yards_after_catch")) or 0.0,
            _clean_float(row.get("red_zone_carries")) or 0.0,
            _clean_float(row.get("red_zone_targets")) or 0.0,
            _clean_float(row.get("red_zone_receptions")) or 0.0,
            _clean_float(row.get("red_zone_pass_attempts")) or 0.0,
            _clean_float(row.get("red_zone_pass_tds")) or 0.0,
            _clean_float(row.get("red_zone_rush_tds")) or 0.0,
            _clean_float(row.get("red_zone_rec_tds")) or 0.0,
            _clean_float(row.get("red_zone_touches")) or 0.0,
            _clean_float(row.get("goal_line_carries")) or 0.0,
            _clean_float(row.get("goal_line_targets")) or 0.0,
            _clean_float(row.get("end_zone_targets")) or 0.0,
        ))
    return [row for row in rows if all(row[:5])]


def _upsert_pbp_player_usage(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO raw.nfl_player_gamelogs (
                season, week, game_id, game_date_et, player_id, player_name,
                team_abbr, opponent_abbr, position, is_home,
                passing_yards, passing_tds, rushing_yards, rushing_tds,
                receiving_yards, receiving_tds, carries, targets, receptions,
                pass_attempts, target_share, air_yards_share, wopr,
                receiving_air_yards, receiving_yards_after_catch,
                red_zone_carries, red_zone_targets, red_zone_receptions,
                red_zone_pass_attempts, red_zone_pass_tds, red_zone_rush_tds,
                red_zone_rec_tds, red_zone_touches, goal_line_carries,
                goal_line_targets, end_zone_targets, source
            )
            SELECT
                v.season, v.week, v.game_id, g.game_date_et, v.player_id,
                COALESCE(v.player_name, r.player_name), v.team_abbr, v.opponent_abbr,
                UPPER(COALESCE(r.position, '')),
                CASE
                    WHEN UPPER(v.team_abbr) = UPPER(g.home_team_abbr) THEN TRUE
                    WHEN UPPER(v.team_abbr) = UPPER(g.away_team_abbr) THEN FALSE
                    ELSE NULL
                END,
                v.passing_yards, v.passing_tds, v.rushing_yards, v.rushing_tds,
                v.receiving_yards, v.receiving_tds, v.carries, v.targets, v.receptions,
                v.pass_attempts, v.target_share, v.air_yards_share, v.wopr,
                v.receiving_air_yards, v.receiving_yards_after_catch,
                v.red_zone_carries, v.red_zone_targets, v.red_zone_receptions,
                v.red_zone_pass_attempts, v.red_zone_pass_tds, v.red_zone_rush_tds,
                v.red_zone_rec_tds, v.red_zone_touches, v.goal_line_carries,
                v.goal_line_targets, v.end_zone_targets, 'nflverse_pbp_usage'
            FROM (VALUES %s) AS v(
                season, week, game_id, player_id, team_abbr, player_name,
                opponent_abbr, passing_yards, passing_tds, rushing_yards,
                rushing_tds, receiving_yards, receiving_tds, carries, targets,
                receptions, pass_attempts, target_share, air_yards_share, wopr,
                receiving_air_yards, receiving_yards_after_catch,
                red_zone_carries, red_zone_targets, red_zone_receptions,
                red_zone_pass_attempts, red_zone_pass_tds, red_zone_rush_tds,
                red_zone_rec_tds, red_zone_touches, goal_line_carries,
                goal_line_targets, end_zone_targets
            )
            LEFT JOIN raw.nfl_games g ON g.game_id = v.game_id
            LEFT JOIN LATERAL (
                SELECT player_name, position
                FROM raw.nfl_rosters r
                WHERE r.season = v.season
                  AND r.player_id = v.player_id
                  AND r.team_abbr = v.team_abbr
                  AND COALESCE(r.week, 0) <= COALESCE(v.week, 999)
                ORDER BY COALESCE(r.week, 0) DESC, r.updated_at_utc DESC
                LIMIT 1
            ) r ON TRUE
            WHERE UPPER(COALESCE(r.position, '')) IN ('QB', 'RB', 'FB', 'WR', 'TE')
              AND g.status = 'final'
            -- Official nflverse box-score values win; play-by-play only fills gaps in
            -- them, and owns the red-zone/end-zone columns no other source provides.
            -- Position follows the roster lookup, which is what forecasting reads.
            ON CONFLICT (season, week, game_id, player_id, team_abbr) DO UPDATE SET
                game_date_et = COALESCE(raw.nfl_player_gamelogs.game_date_et, EXCLUDED.game_date_et),
                player_name = COALESCE(raw.nfl_player_gamelogs.player_name, EXCLUDED.player_name),
                opponent_abbr = COALESCE(raw.nfl_player_gamelogs.opponent_abbr, EXCLUDED.opponent_abbr),
                position = COALESCE(NULLIF(EXCLUDED.position, ''), raw.nfl_player_gamelogs.position),
                is_home = COALESCE(raw.nfl_player_gamelogs.is_home, EXCLUDED.is_home),
                passing_yards = COALESCE(raw.nfl_player_gamelogs.passing_yards, EXCLUDED.passing_yards),
                passing_tds = COALESCE(raw.nfl_player_gamelogs.passing_tds, EXCLUDED.passing_tds),
                rushing_yards = COALESCE(raw.nfl_player_gamelogs.rushing_yards, EXCLUDED.rushing_yards),
                rushing_tds = COALESCE(raw.nfl_player_gamelogs.rushing_tds, EXCLUDED.rushing_tds),
                receiving_yards = COALESCE(raw.nfl_player_gamelogs.receiving_yards, EXCLUDED.receiving_yards),
                receiving_tds = COALESCE(raw.nfl_player_gamelogs.receiving_tds, EXCLUDED.receiving_tds),
                carries = COALESCE(raw.nfl_player_gamelogs.carries, EXCLUDED.carries),
                targets = COALESCE(raw.nfl_player_gamelogs.targets, EXCLUDED.targets),
                receptions = COALESCE(raw.nfl_player_gamelogs.receptions, EXCLUDED.receptions),
                pass_attempts = COALESCE(raw.nfl_player_gamelogs.pass_attempts, EXCLUDED.pass_attempts),
                target_share = COALESCE(raw.nfl_player_gamelogs.target_share, EXCLUDED.target_share),
                air_yards_share = COALESCE(raw.nfl_player_gamelogs.air_yards_share, EXCLUDED.air_yards_share),
                wopr = COALESCE(raw.nfl_player_gamelogs.wopr, EXCLUDED.wopr),
                receiving_air_yards = COALESCE(raw.nfl_player_gamelogs.receiving_air_yards, EXCLUDED.receiving_air_yards),
                receiving_yards_after_catch = COALESCE(raw.nfl_player_gamelogs.receiving_yards_after_catch, EXCLUDED.receiving_yards_after_catch),
                red_zone_carries = COALESCE(EXCLUDED.red_zone_carries, raw.nfl_player_gamelogs.red_zone_carries),
                red_zone_targets = COALESCE(EXCLUDED.red_zone_targets, raw.nfl_player_gamelogs.red_zone_targets),
                red_zone_receptions = COALESCE(EXCLUDED.red_zone_receptions, raw.nfl_player_gamelogs.red_zone_receptions),
                red_zone_pass_attempts = COALESCE(EXCLUDED.red_zone_pass_attempts, raw.nfl_player_gamelogs.red_zone_pass_attempts),
                red_zone_pass_tds = COALESCE(EXCLUDED.red_zone_pass_tds, raw.nfl_player_gamelogs.red_zone_pass_tds),
                red_zone_rush_tds = COALESCE(EXCLUDED.red_zone_rush_tds, raw.nfl_player_gamelogs.red_zone_rush_tds),
                red_zone_rec_tds = COALESCE(EXCLUDED.red_zone_rec_tds, raw.nfl_player_gamelogs.red_zone_rec_tds),
                red_zone_touches = COALESCE(EXCLUDED.red_zone_touches, raw.nfl_player_gamelogs.red_zone_touches),
                goal_line_carries = COALESCE(EXCLUDED.goal_line_carries, raw.nfl_player_gamelogs.goal_line_carries),
                goal_line_targets = COALESCE(EXCLUDED.goal_line_targets, raw.nfl_player_gamelogs.goal_line_targets),
                end_zone_targets = COALESCE(EXCLUDED.end_zone_targets, raw.nfl_player_gamelogs.end_zone_targets),
                source = CASE
                    WHEN raw.nfl_player_gamelogs.source IS NULL THEN EXCLUDED.source
                    WHEN raw.nfl_player_gamelogs.source LIKE '%%pbp%%' THEN raw.nfl_player_gamelogs.source
                    ELSE raw.nfl_player_gamelogs.source || '+pbp_usage'
                END,
                updated_at_utc = NOW()
            WHERE (
                raw.nfl_player_gamelogs.game_date_et,
                raw.nfl_player_gamelogs.player_name,
                raw.nfl_player_gamelogs.opponent_abbr,
                raw.nfl_player_gamelogs.position,
                raw.nfl_player_gamelogs.is_home,
                raw.nfl_player_gamelogs.passing_yards,
                raw.nfl_player_gamelogs.passing_tds,
                raw.nfl_player_gamelogs.rushing_yards,
                raw.nfl_player_gamelogs.rushing_tds,
                raw.nfl_player_gamelogs.receiving_yards,
                raw.nfl_player_gamelogs.receiving_tds,
                raw.nfl_player_gamelogs.carries,
                raw.nfl_player_gamelogs.targets,
                raw.nfl_player_gamelogs.receptions,
                raw.nfl_player_gamelogs.pass_attempts,
                raw.nfl_player_gamelogs.target_share,
                raw.nfl_player_gamelogs.air_yards_share,
                raw.nfl_player_gamelogs.wopr,
                raw.nfl_player_gamelogs.receiving_air_yards,
                raw.nfl_player_gamelogs.receiving_yards_after_catch,
                raw.nfl_player_gamelogs.red_zone_carries,
                raw.nfl_player_gamelogs.red_zone_targets,
                raw.nfl_player_gamelogs.red_zone_receptions,
                raw.nfl_player_gamelogs.red_zone_pass_attempts,
                raw.nfl_player_gamelogs.red_zone_pass_tds,
                raw.nfl_player_gamelogs.red_zone_rush_tds,
                raw.nfl_player_gamelogs.red_zone_rec_tds,
                raw.nfl_player_gamelogs.red_zone_touches,
                raw.nfl_player_gamelogs.goal_line_carries,
                raw.nfl_player_gamelogs.goal_line_targets,
                raw.nfl_player_gamelogs.end_zone_targets,
                raw.nfl_player_gamelogs.source
            ) IS DISTINCT FROM (
                COALESCE(raw.nfl_player_gamelogs.game_date_et, EXCLUDED.game_date_et),
                COALESCE(raw.nfl_player_gamelogs.player_name, EXCLUDED.player_name),
                COALESCE(raw.nfl_player_gamelogs.opponent_abbr, EXCLUDED.opponent_abbr),
                COALESCE(NULLIF(EXCLUDED.position, ''), raw.nfl_player_gamelogs.position),
                COALESCE(raw.nfl_player_gamelogs.is_home, EXCLUDED.is_home),
                COALESCE(raw.nfl_player_gamelogs.passing_yards, EXCLUDED.passing_yards),
                COALESCE(raw.nfl_player_gamelogs.passing_tds, EXCLUDED.passing_tds),
                COALESCE(raw.nfl_player_gamelogs.rushing_yards, EXCLUDED.rushing_yards),
                COALESCE(raw.nfl_player_gamelogs.rushing_tds, EXCLUDED.rushing_tds),
                COALESCE(raw.nfl_player_gamelogs.receiving_yards, EXCLUDED.receiving_yards),
                COALESCE(raw.nfl_player_gamelogs.receiving_tds, EXCLUDED.receiving_tds),
                COALESCE(raw.nfl_player_gamelogs.carries, EXCLUDED.carries),
                COALESCE(raw.nfl_player_gamelogs.targets, EXCLUDED.targets),
                COALESCE(raw.nfl_player_gamelogs.receptions, EXCLUDED.receptions),
                COALESCE(raw.nfl_player_gamelogs.pass_attempts, EXCLUDED.pass_attempts),
                COALESCE(raw.nfl_player_gamelogs.target_share, EXCLUDED.target_share),
                COALESCE(raw.nfl_player_gamelogs.air_yards_share, EXCLUDED.air_yards_share),
                COALESCE(raw.nfl_player_gamelogs.wopr, EXCLUDED.wopr),
                COALESCE(raw.nfl_player_gamelogs.receiving_air_yards, EXCLUDED.receiving_air_yards),
                COALESCE(raw.nfl_player_gamelogs.receiving_yards_after_catch, EXCLUDED.receiving_yards_after_catch),
                COALESCE(EXCLUDED.red_zone_carries, raw.nfl_player_gamelogs.red_zone_carries),
                COALESCE(EXCLUDED.red_zone_targets, raw.nfl_player_gamelogs.red_zone_targets),
                COALESCE(EXCLUDED.red_zone_receptions, raw.nfl_player_gamelogs.red_zone_receptions),
                COALESCE(EXCLUDED.red_zone_pass_attempts, raw.nfl_player_gamelogs.red_zone_pass_attempts),
                COALESCE(EXCLUDED.red_zone_pass_tds, raw.nfl_player_gamelogs.red_zone_pass_tds),
                COALESCE(EXCLUDED.red_zone_rush_tds, raw.nfl_player_gamelogs.red_zone_rush_tds),
                COALESCE(EXCLUDED.red_zone_rec_tds, raw.nfl_player_gamelogs.red_zone_rec_tds),
                COALESCE(EXCLUDED.red_zone_touches, raw.nfl_player_gamelogs.red_zone_touches),
                COALESCE(EXCLUDED.goal_line_carries, raw.nfl_player_gamelogs.goal_line_carries),
                COALESCE(EXCLUDED.goal_line_targets, raw.nfl_player_gamelogs.goal_line_targets),
                COALESCE(EXCLUDED.end_zone_targets, raw.nfl_player_gamelogs.end_zone_targets),
                CASE
                    WHEN raw.nfl_player_gamelogs.source IS NULL THEN EXCLUDED.source
                    WHEN raw.nfl_player_gamelogs.source LIKE '%%pbp%%' THEN raw.nfl_player_gamelogs.source
                    ELSE raw.nfl_player_gamelogs.source || '+pbp_usage'
                END
            )
            """,
            rows,
            page_size=5000,
        )
        updated = cur.rowcount
    conn.commit()
    return max(0, updated)


def _red_zone_rows(df: pd.DataFrame) -> list[tuple]:
    if df.empty or "yardline_100" not in df.columns:
        return []
    pbp = df.copy()
    pbp["team_abbr"] = pbp.get("posteam", pd.Series(index=pbp.index)).map(normalize_team)
    pbp["yardline_100"] = pd.to_numeric(pbp["yardline_100"], errors="coerce")
    pbp = pbp.loc[pbp["yardline_100"].notna() & (pbp["yardline_100"] <= 20) & pbp["team_abbr"].notna()].copy()
    if pbp.empty:
        return []
    pbp["is_goal_line"] = (pbp["yardline_100"] <= 5).astype(float)
    pbp["rush_attempt_f"] = _flag(pbp.get("rush_attempt", pd.Series(index=pbp.index)))
    pbp["pass_attempt_f"] = _flag(pbp.get("pass_attempt", pd.Series(index=pbp.index)))
    pbp["complete_pass_f"] = _flag(pbp.get("complete_pass", pd.Series(index=pbp.index)))
    pbp["pass_td_f"] = _flag(pbp.get("pass_touchdown", pd.Series(index=pbp.index)))
    pbp["rush_td_f"] = _flag(pbp.get("rush_touchdown", pd.Series(index=pbp.index)))

    frames: list[pd.DataFrame] = []
    rush = pbp.loc[pbp["rush_attempt_f"] > 0].copy()
    if not rush.empty and "rusher_player_id" in rush.columns:
        rush["player_id"] = rush["rusher_player_id"]
        rush["red_zone_carries"] = 1.0
        rush["goal_line_carries"] = rush["is_goal_line"]
        rush["red_zone_rush_tds"] = rush["rush_td_f"]
        frames.append(rush[["season", "week", "game_id", "player_id", "team_abbr", "red_zone_carries", "goal_line_carries", "red_zone_rush_tds"]])

    targets = pbp.loc[pbp["pass_attempt_f"] > 0].copy()
    if not targets.empty and "receiver_player_id" in targets.columns:
        targets["player_id"] = targets["receiver_player_id"]
        targets = targets.loc[targets["player_id"].notna()].copy()
        targets["red_zone_targets"] = 1.0
        targets["goal_line_targets"] = targets["is_goal_line"]
        targets["red_zone_receptions"] = targets["complete_pass_f"]
        targets["red_zone_rec_tds"] = targets["pass_td_f"] * (targets.get("td_player_id", "") == targets["player_id"]).astype(float)
        frames.append(targets[[
            "season", "week", "game_id", "player_id", "team_abbr",
            "red_zone_targets", "goal_line_targets", "red_zone_receptions", "red_zone_rec_tds",
        ]])

    passers = pbp.loc[pbp["pass_attempt_f"] > 0].copy()
    if not passers.empty and "passer_player_id" in passers.columns:
        passers["player_id"] = passers["passer_player_id"]
        passers = passers.loc[passers["player_id"].notna()].copy()
        passers["red_zone_pass_attempts"] = 1.0
        passers["red_zone_pass_tds"] = passers["pass_td_f"]
        frames.append(passers[["season", "week", "game_id", "player_id", "team_abbr", "red_zone_pass_attempts", "red_zone_pass_tds"]])

    if not frames:
        return []
    usage = pd.concat(frames, ignore_index=True, sort=False)
    for col in (
        "red_zone_carries", "red_zone_targets", "red_zone_receptions",
        "red_zone_pass_attempts", "red_zone_pass_tds", "red_zone_rush_tds",
        "red_zone_rec_tds", "goal_line_carries", "goal_line_targets",
    ):
        if col not in usage.columns:
            usage[col] = 0.0
    grouped = (
        usage.groupby(["season", "week", "game_id", "player_id", "team_abbr"], dropna=False)[[
            "red_zone_carries", "red_zone_targets", "red_zone_receptions",
            "red_zone_pass_attempts", "red_zone_pass_tds", "red_zone_rush_tds",
            "red_zone_rec_tds", "goal_line_carries", "goal_line_targets",
        ]]
        .sum()
        .reset_index()
    )
    grouped["red_zone_touches"] = grouped["red_zone_carries"] + grouped["red_zone_targets"]
    rows: list[tuple] = []
    for _, row in grouped.iterrows():
        rows.append((
            _clean_int(row.get("season")),
            _clean_int(row.get("week")),
            _clean_text(row.get("game_id")),
            _clean_text(row.get("player_id")),
            normalize_team(row.get("team_abbr")),
            _clean_float(row.get("red_zone_carries")) or 0.0,
            _clean_float(row.get("red_zone_targets")) or 0.0,
            _clean_float(row.get("red_zone_receptions")) or 0.0,
            _clean_float(row.get("red_zone_pass_attempts")) or 0.0,
            _clean_float(row.get("red_zone_pass_tds")) or 0.0,
            _clean_float(row.get("red_zone_rush_tds")) or 0.0,
            _clean_float(row.get("red_zone_rec_tds")) or 0.0,
            _clean_float(row.get("red_zone_touches")) or 0.0,
            _clean_float(row.get("goal_line_carries")) or 0.0,
            _clean_float(row.get("goal_line_targets")) or 0.0,
        ))
    return [row for row in rows if all(row[:5])]


def import_usage(cfg: UsageImportConfig) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok", "seasons": list(cfg.seasons), "sources": []}
    with psycopg2.connect(cfg.pg_dsn) as conn:
        if not cfg.skip_schema:
            ensure_schema(conn)
        by_pfr, by_name_team = _roster_maps(conn, cfg.seasons)
        snap_rows_total = snap_updated = rz_rows_total = rz_updated = 0
        pbp_usage_rows_total = pbp_usage_updated = 0
        participation_rows_total = participation_updated = advanced_rows_total = advanced_updated = 0
        route_cleanup_updated = 0
        for season in cfg.seasons:
            season_snap_rows: list[tuple] = []
            if not cfg.skip_snap_counts:
                source = cfg.snap_counts_url_template.format(season=season)
                df, err = _read_csv(source)
                rows = _snap_rows(df, by_pfr, by_name_team) if err is None else []
                season_snap_rows = rows
                snap_rows_total += len(rows)
                snap_updated += _update_snap_counts(conn, rows)
                health = {"kind": "snap_counts", "season": season, "source": source, "error": err,
                    "rows_read": int(len(df)), "rows_matched": len(rows),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "games": sorted(df.game_id.dropna().astype(str).unique().tolist()) if 'game_id' in df else []}
                result["sources"].append(health)
                atomic_json(Path(__file__).resolve().parents[2]/'reports'/f'nfl_snap_source_health_{season}.json', health)
            if cfg.advanced_usage_url_template and not cfg.skip_advanced_usage:
                source = cfg.advanced_usage_url_template.format(season=season)
                df, err = _read_csv(source)
                rows = _advanced_usage_rows(df, by_name_team) if err is None else []
                advanced_rows_total += len(rows)
                updated = _update_advanced_usage(conn, rows)
                advanced_updated += updated
                health = {
                    "kind": "advanced_route_usage",
                    "season": season,
                    "source": source,
                    "error": err,
                    "rows_read": int(len(df)),
                    "rows_matched": len(rows),
                    "rows_updated": updated,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    **advanced_usage_health(df, rows),
                }
                result["sources"].append(health)
                atomic_json(Path(__file__).resolve().parents[2]/'reports'/f'nfl_advanced_usage_source_health_{season}.json', health)
            if not cfg.skip_participation:
                source = cfg.participation_url_template.format(season=season)
                wanted = {
                    "nflverse_game_id", "game_id", "season", "week", "possession_team",
                    "offense_players", "offense_positions", "route", "time_to_throw",
                    "number_of_pass_rushers",
                }
                df, err = _read_csv(source, usecols=lambda col: col in wanted)
                rows = _participation_rows(df) if err is None else []
                participation_rows_total += len(rows)
                updated = _update_pass_route_opportunities(conn, rows)
                participation_updated += updated
                result["sources"].append({
                    "kind": "participation_route_proxy",
                    "season": season,
                    "source": source,
                    "error": err,
                    "rows_read": int(len(df)),
                    "player_game_rows": len(rows),
                    "rows_updated": updated,
                })
            if not cfg.skip_pbp:
                source = cfg.pbp_url_template.format(season=season)
                wanted = {
                    "season", "week", "game_id", "posteam", "yardline_100",
                    "defteam", "passing_yards", "rushing_yards", "receiving_yards",
                    "air_yards", "yards_after_catch",
                    "rush_attempt", "pass_attempt", "complete_pass",
                    "pass_touchdown", "rush_touchdown", "td_player_id",
                    "rusher_player_id", "receiver_player_id", "passer_player_id",
                    "rusher_player_name", "receiver_player_name", "passer_player_name",
                }
                df, err = _read_csv(source, usecols=lambda col: col in wanted)
                usage_rows = _pbp_player_usage_rows(df) if err is None else []
                pbp_usage_rows_total += len(usage_rows)
                usage_updated = _upsert_pbp_player_usage(conn, usage_rows)
                pbp_usage_updated += usage_updated
                if usage_updated and season_snap_rows:
                    snap_updated += _update_snap_counts(conn, season_snap_rows)
                rows = _red_zone_rows(df) if err is None else []
                rz_rows_total += len(rows)
                updated = _update_red_zone(conn, rows)
                zero_filled = _mark_red_zone_zeros(conn, season) if err is None else 0
                rz_updated += updated + zero_filled
                result["sources"].append({
                    "kind": "pbp_current_usage_and_red_zone",
                    "season": season,
                    "source": source,
                    "error": err,
                    "rows_read": int(len(df)),
                    "player_usage_rows": len(usage_rows),
                    "player_usage_rows_updated": usage_updated,
                    "player_game_rows": len(rows),
                    "rows_updated": updated,
                    "zero_filled_rows": zero_filled,
                })
            cleaned = _clear_non_skill_route_usage(conn, season)
            route_cleanup_updated += cleaned
            if cleaned:
                result["sources"].append({
                    "kind": "route_usage_position_cleanup",
                    "season": season,
                    "rows_updated": cleaned,
                })
    result.update({
        "snap_rows_matched": snap_rows_total,
        "snap_rows_updated": snap_updated,
        "advanced_usage_rows_matched": advanced_rows_total,
        "advanced_usage_rows_updated": advanced_updated,
        "pbp_usage_player_game_rows": pbp_usage_rows_total,
        "pbp_usage_rows_updated": pbp_usage_updated,
        "participation_player_game_rows": participation_rows_total,
        "participation_rows_updated": participation_updated,
        "route_usage_position_cleanup_rows": route_cleanup_updated,
        "red_zone_player_game_rows": rz_rows_total,
        "red_zone_rows_updated": rz_updated,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NFL snap-count and red-zone usage context")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--seasons", default=None, help="Comma/range list, e.g. 2021-2025")
    parser.add_argument("--snap-counts-url-template", default=DEFAULT_SNAP_COUNTS_URL)
    parser.add_argument("--pbp-url-template", default=DEFAULT_PBP_URL)
    parser.add_argument("--participation-url-template", default=DEFAULT_PARTICIPATION_URL)
    parser.add_argument("--advanced-usage-url-template", default=DEFAULT_ADVANCED_USAGE_URL)
    parser.add_argument("--skip-snap-counts", action="store_true")
    parser.add_argument("--skip-pbp", action="store_true")
    parser.add_argument("--skip-participation", action="store_true")
    parser.add_argument("--skip-advanced-usage", action="store_true")
    parser.add_argument("--skip-schema", action="store_true", help="Use provisioned schema; no runtime DDL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    result = import_usage(UsageImportConfig(
        pg_dsn=args.pg_dsn,
        seasons=_parse_seasons(args.seasons),
        snap_counts_url_template=args.snap_counts_url_template,
        pbp_url_template=args.pbp_url_template,
        participation_url_template=args.participation_url_template,
        advanced_usage_url_template=args.advanced_usage_url_template,
        skip_snap_counts=args.skip_snap_counts,
        skip_pbp=args.skip_pbp,
        skip_participation=args.skip_participation,
        skip_advanced_usage=args.skip_advanced_usage,
        skip_schema=args.skip_schema,
    ))
    print(json.dumps(result, indent=2, default=str))
    if args.skip_schema and any(s.get('error') for s in result['sources']):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
