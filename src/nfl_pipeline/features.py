"""NFL player-game feature generation."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.markets import STAT_SPECS
from nfl_pipeline.schema import ensure_schema

log = logging.getLogger("nfl_pipeline.features")
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

TARGET_STATS = tuple(spec.stat for spec in STAT_SPECS)
USAGE_STATS = (
    "carries",
    "targets",
    "receptions",
    "pass_attempts",
    "routes_run",
    "pass_route_opportunities",
    "pass_route_opportunity_share",
    "route_participation",
    "snap_share",
    "target_share",
    "air_yards_share",
    "wopr",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "targets_per_route_run",
    "yards_per_route_run",
    "first_read_targets",
    "first_read_target_share",
    "end_zone_targets",
    "end_zone_target_share",
    "offense_snaps",
    "offense_snap_share",
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
ROLLING_STATS = TARGET_STATS + USAGE_STATS
SPARSE_USAGE_STATS = {
    "route_participation",
    "routes_run",
    "pass_route_opportunities",
    "pass_route_opportunity_share",
    "snap_share",
    "target_share",
    "air_yards_share",
    "wopr",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "targets_per_route_run",
    "yards_per_route_run",
    "first_read_targets",
    "first_read_target_share",
    "end_zone_targets",
    "end_zone_target_share",
    "offense_snaps",
    "offense_snap_share",
}
ROLE_SIGNAL_STATS = ("pass_attempts", "carries", "targets", "receptions", "passing_yards", "rushing_yards", "receiving_yards")
VOLATILITY_STATS = (
    "pass_attempts",
    "carries",
    "targets",
    "receptions",
    "rushing_yards",
    "receiving_yards",
    "rushing_tds",
    "receiving_tds",
    "routes_run",
    "pass_route_opportunities",
    "pass_route_opportunity_share",
    "snap_share",
    "offense_snap_share",
    "red_zone_carries",
    "red_zone_targets",
    "goal_line_carries",
    "goal_line_targets",
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    pg_dsn: str = PG_DSN
    min_season: int | None = None


def _load_logs(conn, min_season: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT
            p.season, p.week, p.game_id,
            g.season_type,
            COALESCE(p.game_date_et, g.game_date_et) AS game_date_et,
            p.player_id, p.player_name, UPPER(p.team_abbr) AS team_abbr,
            UPPER(p.opponent_abbr) AS opponent_abbr,
            UPPER(p.position) AS position,
            COALESCE(p.is_home,
                CASE
                    WHEN UPPER(p.team_abbr) = UPPER(g.home_team_abbr) THEN TRUE
                    WHEN UPPER(p.team_abbr) = UPPER(g.away_team_abbr) THEN FALSE
                    ELSE NULL
                END
            ) AS is_home,
            p.passing_yards::float AS passing_yards,
            p.passing_tds::float AS passing_tds,
            p.rushing_yards::float AS rushing_yards,
            p.rushing_tds::float AS rushing_tds,
            p.receiving_yards::float AS receiving_yards,
            p.receiving_tds::float AS receiving_tds,
            p.carries::float AS carries,
            p.targets::float AS targets,
            p.receptions::float AS receptions,
            p.pass_attempts::float AS pass_attempts,
            p.routes_run::float AS routes_run,
            p.pass_route_opportunities::float AS pass_route_opportunities,
            p.pass_route_opportunity_share::float AS pass_route_opportunity_share,
            p.route_participation::float AS route_participation,
            p.snap_share::float AS snap_share,
            p.target_share::float AS target_share,
            p.air_yards_share::float AS air_yards_share,
            p.wopr::float AS wopr,
            p.receiving_air_yards::float AS receiving_air_yards,
            p.receiving_yards_after_catch::float AS receiving_yards_after_catch,
            p.targets_per_route_run::float AS targets_per_route_run,
            p.yards_per_route_run::float AS yards_per_route_run,
            p.first_read_targets::float AS first_read_targets,
            p.first_read_target_share::float AS first_read_target_share,
            p.end_zone_targets::float AS end_zone_targets,
            p.end_zone_target_share::float AS end_zone_target_share,
            p.offense_snaps::float AS offense_snaps,
            p.offense_snap_share::float AS offense_snap_share,
            p.red_zone_carries::float AS red_zone_carries,
            p.red_zone_targets::float AS red_zone_targets,
            p.red_zone_receptions::float AS red_zone_receptions,
            p.red_zone_pass_attempts::float AS red_zone_pass_attempts,
            p.red_zone_pass_tds::float AS red_zone_pass_tds,
            p.red_zone_rush_tds::float AS red_zone_rush_tds,
            p.red_zone_rec_tds::float AS red_zone_rec_tds,
            p.red_zone_touches::float AS red_zone_touches,
            p.goal_line_carries::float AS goal_line_carries,
            p.goal_line_targets::float AS goal_line_targets,
            r.roster_status,
            CASE
                WHEN r.roster_status IS NULL THEN NULL
                WHEN UPPER(r.roster_status) = 'ACT' THEN TRUE
                ELSE FALSE
            END AS roster_is_active,
            r.depth_chart_position,
            d.pos_abb AS depth_pos_abb,
            d.pos_rank::float AS depth_pos_rank,
            d.pos_slot AS depth_pos_slot,
            i.report_status AS injury_report_status,
            i.practice_status AS injury_practice_status,
            COALESCE(team_inj.teammate_skill_injury_count, 0)::float AS same_week_teammate_skill_injury_count,
            COALESCE(team_inj.teammate_skill_injury_score, 0)::float AS same_week_teammate_skill_injury_score,
            COALESCE(team_inj.teammate_receiver_injury_score, 0)::float AS same_week_teammate_receiver_injury_score,
            COALESCE(team_inj.teammate_receiver_out_count, 0)::float AS same_week_teammate_receiver_out_count,
            g.total_line::float AS game_total_line,
            CASE
                WHEN g.spread_line IS NULL THEN NULL
                WHEN UPPER(p.team_abbr) = UPPER(g.home_team_abbr) THEN -g.spread_line::float
                WHEN UPPER(p.team_abbr) = UPPER(g.away_team_abbr) THEN g.spread_line::float
                ELSE NULL
            END AS team_spread_line,
            CASE
                WHEN g.total_line IS NULL OR g.spread_line IS NULL THEN NULL
                WHEN UPPER(p.team_abbr) = UPPER(g.home_team_abbr) THEN g.total_line::float / 2.0 + g.spread_line::float / 2.0
                WHEN UPPER(p.team_abbr) = UPPER(g.away_team_abbr) THEN g.total_line::float / 2.0 - g.spread_line::float / 2.0
                ELSE NULL
            END AS team_implied_points,
            CASE
                WHEN g.total_line IS NULL OR g.spread_line IS NULL THEN NULL
                WHEN UPPER(p.team_abbr) = UPPER(g.home_team_abbr) THEN g.total_line::float / 2.0 - g.spread_line::float / 2.0
                WHEN UPPER(p.team_abbr) = UPPER(g.away_team_abbr) THEN g.total_line::float / 2.0 + g.spread_line::float / 2.0
                ELSE NULL
            END AS opponent_implied_points
        FROM raw.nfl_player_gamelogs p
        LEFT JOIN raw.nfl_games g ON g.game_id = p.game_id
        LEFT JOIN LATERAL (
            SELECT roster_status, depth_chart_position
            FROM raw.nfl_rosters r
            WHERE r.season = p.season
              AND r.player_id = p.player_id
              AND r.team_abbr = UPPER(p.team_abbr)
              AND r.updated_at_utc < g.start_ts_utc
              AND COALESCE(r.week, 0) <= COALESCE(p.week, 999)
            ORDER BY
                COALESCE(r.week, 0) DESC,
                CASE WHEN UPPER(COALESCE(r.roster_status, '')) = 'ACT' THEN 0 ELSE 1 END,
                r.updated_at_utc DESC
            LIMIT 1
        ) r ON TRUE
        LEFT JOIN LATERAL (
            SELECT pos_abb, pos_rank, pos_slot
            FROM raw.nfl_depth_charts d
            WHERE d.season = p.season
              AND d.player_id = p.player_id
              AND d.team_abbr = UPPER(p.team_abbr)
              AND (
                  d.week IS NULL
                  OR p.week IS NULL
                  OR d.week <= p.week
              )
              AND d.updated_at_utc < g.start_ts_utc
              AND COALESCE(d.snapshot_ts_utc,d.updated_at_utc) < g.start_ts_utc
            ORDER BY COALESCE(d.week, 0) DESC, d.snapshot_ts_utc DESC NULLS LAST, d.pos_rank NULLS LAST
            LIMIT 1
        ) d ON TRUE
        LEFT JOIN LATERAL (
            SELECT report_status, practice_status
            FROM raw.nfl_injuries i
            WHERE i.season = p.season
              AND i.player_id = p.player_id
              AND i.team_abbr = UPPER(p.team_abbr)
              AND i.updated_at_utc < g.start_ts_utc
              AND i.week = p.week
            ORDER BY COALESCE(i.week, 0) DESC, i.updated_at_utc DESC
            LIMIT 1
        ) i ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                COUNT(*) FILTER (
                    WHERE UPPER(COALESCE(ti.position, '')) IN ('RB', 'WR', 'TE')
                      AND ti.injury_score > 0
                )::float AS teammate_skill_injury_count,
                SUM(ti.injury_score) FILTER (
                    WHERE UPPER(COALESCE(ti.position, '')) IN ('RB', 'WR', 'TE')
                )::float AS teammate_skill_injury_score,
                SUM(ti.injury_score) FILTER (
                    WHERE UPPER(COALESCE(ti.position, '')) IN ('WR', 'TE')
                )::float AS teammate_receiver_injury_score,
                COUNT(*) FILTER (
                    WHERE UPPER(COALESCE(ti.position, '')) IN ('WR', 'TE')
                      AND ti.is_out
                )::float AS teammate_receiver_out_count
            FROM (
                SELECT DISTINCT ON (COALESCE(player_id, player_name_norm))
                    player_id,
                    player_name_norm,
                    position,
                    CASE
                        WHEN LOWER(COALESCE(report_status, '')) ~ 'injured reserve|reserve|out' THEN TRUE
                        WHEN LOWER(COALESCE(report_status, '')) LIKE '%%doubtful%%' THEN TRUE
                        ELSE FALSE
                    END AS is_out,
                    GREATEST(
                        CASE
                            WHEN LOWER(COALESCE(report_status, '')) ~ 'injured reserve|reserve|out' THEN 1.00
                            WHEN LOWER(COALESCE(report_status, '')) LIKE '%%doubtful%%' THEN 0.85
                            WHEN LOWER(COALESCE(report_status, '')) LIKE '%%questionable%%' THEN 0.45
                            ELSE 0.00
                        END,
                        CASE
                            WHEN LOWER(COALESCE(practice_status, '')) ~ 'did not participate|dnp' THEN 0.65
                            WHEN LOWER(COALESCE(practice_status, '')) LIKE '%%limited%%' THEN 0.25
                            ELSE 0.00
                        END
                    )::float AS injury_score
                FROM raw.nfl_injuries ti
                WHERE ti.season = p.season
                  AND ti.team_abbr = UPPER(p.team_abbr)
                  AND ti.updated_at_utc < g.start_ts_utc
                  AND ti.week = p.week
                  AND COALESCE(ti.player_id, '') <> COALESCE(p.player_id, '')
                ORDER BY COALESCE(player_id, player_name_norm), COALESCE(week, 0) DESC, updated_at_utc DESC
            ) ti
        ) team_inj ON TRUE
        WHERE (%(min_season)s IS NULL OR p.season >= %(min_season)s)
          AND g.status = 'final'
          AND (COALESCE(p.offense_snaps,0)>0 OR COALESCE(p.pass_attempts,0)+COALESCE(p.carries,0)+COALESCE(p.targets,0)>0)
          AND p.season IS NOT NULL
          AND p.week IS NOT NULL
          AND p.game_id IS NOT NULL
          AND p.player_id IS NOT NULL
          AND UPPER(COALESCE(p.position, '')) IN ('QB', 'RB', 'WR', 'TE')
    """
    return pd.read_sql(sql, conn, params={"min_season": min_season})


def _add_player_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week", "game_id"]).copy()
    new_cols: dict[str, pd.Series] = {}
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            df[stat] = 0.0
        raw_values = pd.to_numeric(df[stat], errors="coerce")
        values = raw_values if stat in SPARSE_USAGE_STATS else raw_values.fillna(0.0)
        df[stat] = values
        grouped = values.groupby(df["player_id"])
        shifted = grouped.shift(1)
        for window in (3, 5, 10):
            new_cols[f"{stat}_avg_{window}"] = (
                shifted
                .groupby(df["player_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            if stat in VOLATILITY_STATS and window in (5, 10):
                new_cols[f"{stat}_std_{window}"] = (
                    shifted
                    .groupby(df["player_id"])
                    .rolling(window, min_periods=2)
                    .std()
                    .reset_index(level=0, drop=True)
                )
    player_game_number = df.groupby("player_id").cumcount()
    new_cols["n_games_prev_3"] = np.minimum(player_game_number, 3).astype(int)
    new_cols["n_games_prev_5"] = np.minimum(player_game_number, 5).astype(int)
    new_cols["n_games_prev_10"] = np.minimum(player_game_number, 10).astype(int)
    dates = pd.to_datetime(df["game_date_et"], errors="coerce")
    new_cols["rest_days"] = dates.groupby(df["player_id"]).diff().dt.days
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()


def _add_team_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "week", "game_id", "team_abbr", "player_id"]).copy()
    team_games = (
        df[["team_abbr", "game_id", "season", "week"]]
        .drop_duplicates()
        .sort_values(["team_abbr", "season", "week", "game_id"])
    )
    team_games["team_game_number"] = team_games.groupby("team_abbr").cumcount()
    df = df.merge(team_games[["team_abbr", "game_id", "team_game_number"]], on=["team_abbr", "game_id"], how="left")

    opp_games = team_games.rename(columns={"team_abbr": "opponent_abbr", "team_game_number": "opp_game_number"})
    df = df.merge(opp_games[["opponent_abbr", "game_id", "opp_game_number"]], on=["opponent_abbr", "game_id"], how="left")
    return df


def _add_defense_allowed(df: pd.DataFrame) -> pd.DataFrame:
    allowed = (
        df.groupby(["opponent_abbr", "season", "week", "game_id"], dropna=False)[list(TARGET_STATS)]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"opponent_abbr": "def_team"})
        .sort_values(["def_team", "season", "week", "game_id"])
    )
    for stat in TARGET_STATS:
        col = f"opp_allowed_{stat}_avg_5"
        allowed[col] = (
            allowed.groupby("def_team")[stat]
            .shift(1)
            .groupby(allowed["def_team"])
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    keep = ["def_team", "game_id"] + [f"opp_allowed_{stat}_avg_5" for stat in TARGET_STATS]
    return df.merge(
        allowed[keep],
        left_on=["opponent_abbr", "game_id"],
        right_on=["def_team", "game_id"],
        how="left",
    ).drop(columns=["def_team"], errors="ignore")


def _add_usage_role_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["game_id", "team_abbr"]
    for stat in ROLE_SIGNAL_STATS:
        avg_col = f"{stat}_avg_5"
        if avg_col not in out.columns:
            out[avg_col] = np.nan
        values = pd.to_numeric(out[avg_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        sum_col = f"team_player_{stat}_avg5_sum"
        share_col = f"{stat}_share_avg_5"
        rank_col = f"{stat}_role_rank"
        team_sum = values.groupby([out[col] for col in group_cols]).transform("sum")
        out[sum_col] = team_sum.replace(0.0, np.nan)
        out[share_col] = (values / out[sum_col]).replace([np.inf, -np.inf], np.nan)
        out[rank_col] = (
            values.groupby([out[col] for col in group_cols])
            .rank(method="min", ascending=False)
            .where(team_sum > 0)
        )
    return out


def _injury_downgrade_score(report_status: pd.Series, practice_status: pd.Series) -> pd.Series:
    report = report_status.astype(str).str.lower().fillna("")
    practice = practice_status.astype(str).str.lower().fillna("")
    score = pd.Series(0.0, index=report_status.index, dtype=float)
    score = score.mask(report.str.contains("injured reserve|reserve|out", regex=True, na=False), 1.0)
    score = score.mask(report.str.contains("doubtful", regex=False, na=False), np.maximum(score, 0.85))
    score = score.mask(report.str.contains("questionable", regex=False, na=False), np.maximum(score, 0.45))
    score = score.mask(practice.str.contains("did not participate|dnp", regex=True, na=False), np.maximum(score, 0.65))
    score = score.mask(practice.str.contains("limited", regex=False, na=False), np.maximum(score, 0.25))
    return score.clip(0.0, 1.0)


def _add_context_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    week = pd.to_numeric(out.get("week"), errors="coerce")
    season_type = out.get("season_type", pd.Series(index=out.index, dtype=object)).astype(str).str.lower()
    out["is_week_18"] = (week >= 18).astype(float).fillna(0.0)
    out["is_late_season"] = (week >= 15).astype(float).fillna(0.0)
    out["is_postseason"] = season_type.str.contains("post|playoff", regex=True, na=False).astype(float)

    depth = pd.to_numeric(out.get("depth_pos_rank"), errors="coerce")
    prev_depth = pd.to_numeric(out["previous_depth_pos_rank"], errors="coerce") if "previous_depth_pos_rank" in out else depth.groupby(out["player_id"]).shift(1)
    out["depth_rank_delta"] = (depth - prev_depth).where(depth.notna() & prev_depth.notna())
    out["depth_rank_worse"] = (pd.to_numeric(out["depth_rank_delta"], errors="coerce") > 0.25).astype(float)
    out["depth_rank_better"] = (pd.to_numeric(out["depth_rank_delta"], errors="coerce") < -0.25).astype(float)
    out["depth_rank_change_abs"] = pd.to_numeric(out["depth_rank_delta"], errors="coerce").abs()

    snap = pd.to_numeric(out.get("offense_snap_share_avg_5"), errors="coerce")
    snap = snap.fillna(pd.to_numeric(out.get("snap_share_avg_5"), errors="coerce")).clip(0.0, 1.0)
    active = out.get("roster_is_active", pd.Series(True, index=out.index)).astype("boolean").fillna(True).astype(float)
    depth_score = pd.Series(0.50, index=out.index, dtype=float)
    depth_score = depth_score.mask(depth <= 1.25, 0.92)
    depth_score = depth_score.mask((depth > 1.25) & (depth <= 2.25), 0.68)
    depth_score = depth_score.mask((depth > 2.25) & (depth <= 3.25), 0.42)
    depth_score = depth_score.mask(depth > 3.25, 0.20)
    snap_adjustment = (snap.fillna(0.45) - 0.45) * 0.35
    out["injury_downgrade_score"] = _injury_downgrade_score(
        out.get("injury_report_status", pd.Series(index=out.index, dtype=object)),
        out.get("injury_practice_status", pd.Series(index=out.index, dtype=object)),
    )
    out["starter_confidence"] = (depth_score + snap_adjustment).clip(0.0, 1.0) * active
    out["rest_risk_score"] = (
        0.35 * out["is_week_18"].astype(float)
        + 0.12 * out["is_late_season"].astype(float)
        + 0.25 * out["is_postseason"].astype(float)
        + 0.30 * (1.0 - out["starter_confidence"].astype(float))
        + 0.35 * out["injury_downgrade_score"].astype(float)
        + 0.08 * out["depth_rank_worse"].astype(float)
    ).clip(0.0, 1.0)

    def num(name: str, default: float | None = None) -> pd.Series:
        fallback = np.nan if default is None else default
        return pd.to_numeric(out.get(name, pd.Series(fallback, index=out.index)), errors="coerce")

    team_imp = num("team_implied_points", 22.0).fillna(22.0)
    team_env = (team_imp / 22.0).clip(0.65, 1.45)
    spread = num("team_spread_line", 0.0).fillna(0.0)
    favorite_run_env = (1.0 + (-spread / 14.0)).clip(0.65, 1.35)
    trailing_pass_env = (1.0 + (spread / 14.0)).clip(0.65, 1.35)

    out["snap_share_trend_3_10"] = num("offense_snap_share_avg_3").fillna(num("snap_share_avg_3")) - num("offense_snap_share_avg_10").fillna(num("snap_share_avg_10"))
    out["snap_share_volatility_5"] = num("offense_snap_share_std_5").fillna(num("snap_share_std_5")).fillna(0.0).clip(0.0, 0.75)
    out["targets_trend_3_10"] = (num("targets_avg_3", 0.0).fillna(0.0) - num("targets_avg_10", 0.0).fillna(0.0))
    out["carries_trend_3_10"] = (num("carries_avg_3", 0.0).fillna(0.0) - num("carries_avg_10", 0.0).fillna(0.0))
    out["receptions_trend_3_10"] = (num("receptions_avg_3", 0.0).fillna(0.0) - num("receptions_avg_10", 0.0).fillna(0.0))
    out["receiving_yards_trend_3_10"] = (num("receiving_yards_avg_3", 0.0).fillna(0.0) - num("receiving_yards_avg_10", 0.0).fillna(0.0))
    out["rushing_yards_trend_3_10"] = (num("rushing_yards_avg_3", 0.0).fillna(0.0) - num("rushing_yards_avg_10", 0.0).fillna(0.0))
    out["target_share_trend_3_10"] = (num("target_share_avg_3").fillna(0.0) - num("target_share_avg_10").fillna(0.0))
    out["air_yards_share_trend_3_10"] = (num("air_yards_share_avg_3").fillna(0.0) - num("air_yards_share_avg_10").fillna(0.0))
    out["wopr_trend_3_10"] = (num("wopr_avg_3").fillna(0.0) - num("wopr_avg_10").fillna(0.0))
    out["route_share_trend_3_10"] = (num("pass_route_opportunity_share_avg_3").fillna(num("route_participation_avg_3")).fillna(0.0) - num("pass_route_opportunity_share_avg_10").fillna(num("route_participation_avg_10")).fillna(0.0))
    out["recent_snap_drop_score"] = ((-out["snap_share_trend_3_10"].astype(float).fillna(0.0)) / 0.20).clip(0.0, 1.0)
    out["recent_snap_rise_score"] = (out["snap_share_trend_3_10"].astype(float).fillna(0.0) / 0.20).clip(0.0, 1.0)
    out["recent_target_spike_score"] = (out["targets_trend_3_10"].astype(float).fillna(0.0) / 3.0).clip(0.0, 1.0)
    out["recent_carry_spike_score"] = (out["carries_trend_3_10"].astype(float).fillna(0.0) / 5.0).clip(0.0, 1.0)
    out["recent_route_spike_score"] = (out["route_share_trend_3_10"].astype(float).fillna(0.0) / 0.20).clip(0.0, 1.0)

    targets_std = num("targets_std_5", 0.0).fillna(0.0)
    carries_std = num("carries_std_5", 0.0).fillna(0.0)
    pass_attempts_std = num("pass_attempts_std_5", 0.0).fillna(0.0)
    route_share_std = num("pass_route_opportunity_share_std_5").fillna(num("route_participation_std_5")).fillna(0.0)
    out["usage_volatility_score"] = (
        0.22 * (targets_std / 4.0).clip(0.0, 1.0)
        + 0.22 * (carries_std / 8.0).clip(0.0, 1.0)
        + 0.18 * (pass_attempts_std / 14.0).clip(0.0, 1.0)
        + 0.20 * out["snap_share_volatility_5"].astype(float).clip(0.0, 1.0)
        + 0.18 * route_share_std.clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["limited_workload_risk_score"] = (
        0.40 * out["rest_risk_score"].astype(float)
        + 0.25 * out["injury_downgrade_score"].astype(float)
        + 0.15 * out["depth_rank_worse"].astype(float)
        + 0.12 * (1.0 - snap.fillna(0.45).clip(0.0, 1.0))
        + 0.08 * out["usage_volatility_score"].astype(float)
    ).clip(0.0, 1.0)
    out["full_workload_score"] = (
        out["starter_confidence"].astype(float).clip(0.0, 1.0)
        * (1.0 - out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0))
        * (0.75 + 0.25 * snap.fillna(0.45).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["backup_role_score"] = (
        (1.0 - out["starter_confidence"].astype(float).clip(0.0, 1.0))
        + 0.25 * out["depth_rank_worse"].astype(float)
        + 0.20 * out["recent_snap_drop_score"].astype(float)
    ).clip(0.0, 1.0)
    out["starter_role_stability_score"] = (
        out["starter_confidence"].astype(float).clip(0.0, 1.0)
        * (1.0 - out["recent_snap_drop_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - out["injury_downgrade_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["projected_starter_score"] = (
        0.55 * out["starter_confidence"].astype(float).clip(0.0, 1.0)
        + 0.30 * out["starter_role_stability_score"].astype(float).clip(0.0, 1.0)
        + 0.15 * snap.fillna(0.45).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["depth_movement_risk_score"] = (
        0.55 * out["depth_rank_worse"].astype(float).clip(0.0, 1.0)
        + 0.25 * out["recent_snap_drop_score"].astype(float).clip(0.0, 1.0)
        + 0.20 * out["depth_rank_change_abs"].astype(float).fillna(0.0).clip(0.0, 3.0) / 3.0
    ).clip(0.0, 1.0)
    out["weird_usage_risk_score"] = (
        0.34 * out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0)
        + 0.22 * out["rest_risk_score"].astype(float).clip(0.0, 1.0)
        + 0.18 * out["depth_movement_risk_score"].astype(float).clip(0.0, 1.0)
        + 0.16 * out["usage_volatility_score"].astype(float).clip(0.0, 1.0)
        + 0.10 * out["backup_role_score"].astype(float).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    high_target_role = num("targets_share_avg_5", 0.0).fillna(0.0).clip(0.0, 1.0)
    high_carry_role = num("carries_share_avg_5", 0.0).fillna(0.0).clip(0.0, 1.0)
    out["high_usage_fragility_score"] = (
        np.maximum(high_target_role, high_carry_role)
        * (0.55 * out["limited_workload_risk_score"].astype(float) + 0.45 * out["usage_volatility_score"].astype(float))
    ).clip(0.0, 1.0)
    teammate_skill_injury = num("same_week_teammate_skill_injury_score", 0.0).fillna(0.0).clip(0.0, 4.0)
    teammate_receiver_injury = num("same_week_teammate_receiver_injury_score", 0.0).fillna(0.0).clip(0.0, 4.0)
    teammate_receiver_out = num("same_week_teammate_receiver_out_count", 0.0).fillna(0.0).clip(0.0, 4.0)
    out["same_week_context_confidence_score"] = (
        0.40 * out.get("roster_is_active", pd.Series(False, index=out.index)).astype("boolean").fillna(False).astype(float)
        + 0.25 * pd.to_numeric(out.get("depth_pos_rank", pd.Series(np.nan, index=out.index)), errors="coerce").notna().astype(float)
        + 0.20 * pd.to_numeric(out.get("same_week_teammate_skill_injury_count", pd.Series(np.nan, index=out.index)), errors="coerce").notna().astype(float)
        + 0.15 * out.get("injury_report_status", pd.Series(index=out.index, dtype=object)).notna().astype(float)
    ).clip(0.0, 1.0)
    out["teammate_skill_injury_pressure_score"] = (teammate_skill_injury / 2.50).clip(0.0, 1.0)
    out["teammate_receiver_injury_pressure_score"] = (
        0.70 * (teammate_receiver_injury / 1.60).clip(0.0, 1.0)
        + 0.30 * (teammate_receiver_out / 2.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["role_continuity_score"] = (
        out["projected_starter_score"].astype(float).clip(0.0, 1.0)
        * (1.0 - out["weird_usage_risk_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - out["high_usage_fragility_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["normal_usage_path_score"] = (
        0.45 * out["full_workload_score"].astype(float).clip(0.0, 1.0)
        + 0.35 * out["role_continuity_score"].astype(float).clip(0.0, 1.0)
        + 0.20 * (1.0 - out["weird_usage_risk_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)

    def sigmoid_score(value: pd.Series, center: float, scale: float) -> pd.Series:
        z = ((pd.to_numeric(value, errors="coerce").fillna(0.0) - center) / max(scale, 1e-6)).clip(-35.0, 35.0)
        return (1.0 / (1.0 + np.exp(-z))).clip(0.02, 0.92)

    out["high_pass_attempt_score"] = sigmoid_score(num("pass_attempts_avg_5", 0.0), 31.0, 5.5)
    out["high_carry_score"] = sigmoid_score(num("carries_avg_5", 0.0), 11.0, 3.8)
    out["high_target_score"] = sigmoid_score(num("targets_avg_5", 0.0), 6.0, 2.2)
    spike_snap_cut = pd.Series(0.72, index=out.index, dtype=float)
    pos = out.get("position", pd.Series("", index=out.index)).astype(str).str.upper()
    spike_snap_cut.loc[pos == "QB"] = 0.92
    spike_snap_cut.loc[pos == "RB"] = 0.58
    spike_snap_cut.loc[pos == "TE"] = 0.70
    out["spike_snap_share_score"] = sigmoid_score(snap.fillna(0.0), spike_snap_cut, 0.11)
    out["game_script_pass_boost"] = trailing_pass_env
    out["game_script_rush_boost"] = favorite_run_env
    out["spike_target_opportunity_score"] = pd.concat([
        out["high_target_score"].astype(float),
        out["recent_target_spike_score"].astype(float),
        out["recent_route_spike_score"].astype(float),
        0.75 * out["spike_snap_share_score"].astype(float),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["spike_carry_opportunity_score"] = pd.concat([
        out["high_carry_score"].astype(float),
        out["recent_carry_spike_score"].astype(float),
        0.65 * out["spike_snap_share_score"].astype(float),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["spike_pass_attempt_opportunity_score"] = pd.concat([
        out["high_pass_attempt_score"].astype(float),
        0.50 * out["spike_snap_share_score"].astype(float),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    usage_path_gate = (
        0.55
        + 0.35 * out["role_continuity_score"].astype(float).clip(0.0, 1.0)
        - 0.25 * out["weird_usage_risk_score"].astype(float).clip(0.0, 1.0)
    ).clip(0.20, 1.15)
    out["target_spike_path_score"] = (
        out["spike_target_opportunity_score"].astype(float).clip(0.0, 1.0)
        * trailing_pass_env
        * usage_path_gate
    ).clip(0.0, 1.0)
    out["carry_spike_path_score"] = (
        out["spike_carry_opportunity_score"].astype(float).clip(0.0, 1.0)
        * favorite_run_env
        * usage_path_gate
    ).clip(0.0, 1.0)
    out["pass_spike_path_score"] = (
        out["spike_pass_attempt_opportunity_score"].astype(float).clip(0.0, 1.0)
        * trailing_pass_env
        * usage_path_gate
    ).clip(0.0, 1.0)

    routes_true = pd.to_numeric(out.get("routes_run_avg_5"), errors="coerce")
    route_opportunities = pd.to_numeric(out.get("pass_route_opportunities_avg_5"), errors="coerce")
    targets = pd.to_numeric(out.get("targets_avg_5"), errors="coerce").fillna(0.0).clip(lower=0.0)
    receiving_yards = pd.to_numeric(out.get("receiving_yards_avg_5"), errors="coerce").fillna(0.0).clip(lower=0.0)
    target_share = pd.to_numeric(out.get("target_share_avg_5"), errors="coerce").replace(0.0, np.nan)
    inferred_team_attempts = (targets / target_share).replace([np.inf, -np.inf], np.nan)
    route_participation_proxy = pd.to_numeric(out.get("route_participation_avg_5"), errors="coerce")
    route_participation_proxy = route_participation_proxy.fillna(
        pd.to_numeric(out.get("pass_route_opportunity_share_avg_5"), errors="coerce")
    )
    route_participation_proxy = route_participation_proxy.fillna(snap.fillna(0.0)).clip(0.0, 1.0)
    estimated_routes = routes_true.where(routes_true > 0)
    estimated_routes = estimated_routes.fillna(route_opportunities.where(route_opportunities > 0))
    estimated_routes = estimated_routes.fillna((inferred_team_attempts * route_participation_proxy).where(inferred_team_attempts > 0))
    estimated_routes = estimated_routes.fillna(pd.to_numeric(out.get("offense_snaps_avg_5"), errors="coerce") * route_participation_proxy)
    estimated_routes = estimated_routes.clip(lower=0.0)
    denom = estimated_routes.replace(0.0, np.nan)
    out["has_true_route_history"] = (
        (routes_true > 0) | pd.to_numeric(out.get("route_participation_avg_5"), errors="coerce").notna()
    ).astype(float)
    out["estimated_routes_avg_5"] = estimated_routes
    out["route_participation_proxy_avg_5"] = route_participation_proxy
    out["targets_per_route_proxy_avg_5"] = pd.to_numeric(out.get("targets_per_route_run_avg_5"), errors="coerce")
    out["targets_per_route_proxy_avg_5"] = out["targets_per_route_proxy_avg_5"].fillna(
        (targets / denom).replace([np.inf, -np.inf], np.nan)
    )
    out["yards_per_route_proxy_avg_5"] = pd.to_numeric(out.get("yards_per_route_run_avg_5"), errors="coerce")
    out["yards_per_route_proxy_avg_5"] = out["yards_per_route_proxy_avg_5"].fillna(
        (receiving_yards / denom).replace([np.inf, -np.inf], np.nan)
    )
    out["route_target_intensity_avg_5"] = (
        route_participation_proxy.fillna(0.0).clip(0.0, 1.0)
        * out["targets_per_route_proxy_avg_5"].fillna(0.0).clip(lower=0.0)
    )
    out["route_weighted_target_expectation_avg_5"] = (
        estimated_routes.fillna(0.0).clip(lower=0.0)
        * out["targets_per_route_proxy_avg_5"].fillna(0.0).clip(lower=0.0)
    )
    out["route_weighted_yard_expectation_avg_5"] = (
        estimated_routes.fillna(0.0).clip(lower=0.0)
        * out["yards_per_route_proxy_avg_5"].fillna(0.0).clip(lower=0.0)
    )
    out["red_zone_targets_per_route_proxy_avg_5"] = (
        pd.to_numeric(out.get("red_zone_targets_avg_5"), errors="coerce").fillna(0.0).clip(lower=0.0) / denom
    ).replace([np.inf, -np.inf], np.nan)
    wopr = pd.to_numeric(out.get("wopr_avg_5"), errors="coerce").fillna(0.0).clip(lower=0.0)
    target_role = pd.to_numeric(out.get("targets_share_avg_5"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["first_read_proxy_avg_5"] = pd.to_numeric(out.get("first_read_target_share_avg_5"), errors="coerce")
    first_read_rate = (
        pd.to_numeric(out.get("first_read_targets_avg_5"), errors="coerce").fillna(0.0).clip(lower=0.0) / denom
    ).replace([np.inf, -np.inf], np.nan)
    out["first_read_proxy_avg_5"] = out["first_read_proxy_avg_5"].fillna(first_read_rate)
    out["first_read_proxy_avg_5"] = out["first_read_proxy_avg_5"].fillna(
        (0.55 * wopr + 0.45 * target_role).clip(0.0, 1.0)
    )
    receiver_usage_pos_gate = pos.isin(["RB", "WR", "TE"]).astype(float)
    snap_history_known = ((num("offense_snap_share_avg_5") > 0) | (num("snap_share_avg_5") > 0)).astype(float)
    target_share_known = (num("target_share_avg_5") > 0).astype(float)
    air_yards_known = (
        (num("air_yards_share_avg_5") > 0)
        | (num("receiving_air_yards_avg_5") > 0)
    ).astype(float)
    wopr_known = (num("wopr_avg_5") > 0).astype(float)
    first_read_known = (
        (num("first_read_target_share_avg_5") > 0)
        | (num("first_read_targets_avg_5") > 0)
    ).astype(float)
    receiver_td_usage_known = (
        (num("red_zone_targets_avg_5") > 0)
        | (num("goal_line_targets_avg_5") > 0)
        | (num("end_zone_targets_avg_5") > 0)
    ).astype(float)
    rb_td_usage_known = (
        (num("red_zone_carries_avg_5") > 0)
        | (num("goal_line_carries_avg_5") > 0)
    ).astype(float)
    out["receiving_usage_history_quality_score"] = (
        receiver_usage_pos_gate
        * (
            0.20 * out["has_true_route_history"].astype(float).clip(0.0, 1.0)
            + 0.16 * (estimated_routes.fillna(0.0) > 0).astype(float)
            + 0.16 * (targets > 0).astype(float)
            + 0.14 * target_share_known
            + 0.12 * air_yards_known
            + 0.08 * wopr_known
            + 0.07 * first_read_known
            + 0.07 * snap_history_known
        )
    ).clip(0.0, 1.0)
    out["rb_usage_history_quality_score"] = (
        pos.eq("RB").astype(float)
        * (
            0.24 * (num("carries_avg_5", 0.0).fillna(0.0) > 0).astype(float)
            + 0.18 * (num("carries_avg_10", 0.0).fillna(0.0) > 0).astype(float)
            + 0.16 * snap_history_known
            + 0.14 * (num("carries_share_avg_5", 0.0).fillna(0.0) > 0).astype(float)
            + 0.12 * rb_td_usage_known
            + 0.10 * (num("yards_per_carry_avg_5", 0.0).fillna(0.0) > 0).astype(float)
            + 0.06 * (num("targets_avg_5", 0.0).fillna(0.0) > 0).astype(float)
        )
    ).clip(0.0, 1.0)
    out["td_usage_history_quality_score"] = pd.concat([
        receiver_td_usage_known.astype(float),
        rb_td_usage_known.astype(float),
        first_read_known.astype(float),
        snap_history_known.astype(float),
    ], axis=1).mean(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["receiver_route_quality_score"] = (
        route_participation_proxy.fillna(0.0).clip(0.0, 1.0)
        * (0.50 + 0.50 * target_role)
        * (0.70 + 0.30 * out["first_read_proxy_avg_5"].fillna(0.0).clip(0.0, 1.0))
    ).clip(0.0, 1.5)
    out["receiver_yards_rate_signal"] = (
        0.55 * num("route_weighted_yard_expectation_avg_5", 0.0).fillna(0.0)
        + 0.25 * pd.to_numeric(out.get("receiving_air_yards_avg_5"), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out.get("receiving_yards_after_catch_avg_5"), errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    out["receiver_route_env_score"] = (
        out["receiver_route_quality_score"].astype(float)
        * team_env
        * trailing_pass_env
        * out["full_workload_score"].astype(float).clip(0.0, 1.0)
    ).clip(0.0, 2.5)
    out["receiver_spike_yards_score"] = (
        out["receiver_route_env_score"].astype(float).clip(0.0, 2.5)
        * (0.65 + 0.35 * out["spike_target_opportunity_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.45 * out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 3.0)
    out["rb_rush_role_env_score"] = (
        num("carries_avg_5", 0.0).fillna(0.0).clip(lower=0.0)
        * (0.50 + high_carry_role)
        * favorite_run_env
        * out["full_workload_score"].astype(float).clip(0.0, 1.0)
    ).clip(lower=0.0)
    out["rb_carry_trend_env_score"] = (
        (num("carries_avg_5", 0.0).fillna(0.0) + 0.50 * out["carries_trend_3_10"].astype(float).fillna(0.0))
        .clip(lower=0.0)
        * favorite_run_env
        * (1.0 - out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["rb_spike_rush_score"] = (
        out["rb_rush_role_env_score"].astype(float).clip(lower=0.0)
        * (0.70 + 0.30 * out["spike_carry_opportunity_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.45 * out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["td_goal_line_env_score"] = (
        pd.to_numeric(out.get("goal_line_carries_avg_5"), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("goal_line_targets_avg_5"), errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(out.get("red_zone_touches_avg_5"), errors="coerce").fillna(0.0)
    ) * team_env * out["full_workload_score"].astype(float).clip(0.0, 1.0)
    out["receiving_td_role_score"] = (
        (
            pd.to_numeric(out.get("red_zone_targets_avg_5"), errors="coerce").fillna(0.0)
            + 1.75 * pd.to_numeric(out.get("goal_line_targets_avg_5"), errors="coerce").fillna(0.0)
            + 1.25 * pd.to_numeric(out.get("end_zone_targets_avg_5"), errors="coerce").fillna(0.0)
        )
        * (0.65 + out["first_read_proxy_avg_5"].fillna(0.0).clip(0.0, 1.0))
        * team_env
        * out["full_workload_score"].astype(float).clip(0.0, 1.0)
    ).clip(lower=0.0)
    out["receiving_td_route_redzone_score"] = (
        out["receiving_td_role_score"].astype(float)
        * (0.50 + out["receiver_route_quality_score"].astype(float).clip(0.0, 1.5))
    ).clip(lower=0.0)
    out["receiving_td_any_score"] = (
        0.42 * out["receiving_td_role_score"].astype(float).clip(lower=0.0)
        + 0.26 * out["receiving_td_route_redzone_score"].astype(float).clip(lower=0.0)
        + 0.18 * out["first_read_proxy_avg_5"].astype(float).fillna(0.0).clip(0.0, 1.0)
        + 0.14 * out["spike_target_opportunity_score"].astype(float).clip(0.0, 1.0)
    ).clip(lower=0.0)
    out["receiver_usage_spike_signal"] = (
        0.45 * out["target_spike_path_score"].astype(float).clip(0.0, 1.0)
        + 0.20 * out["spike_snap_share_score"].astype(float).clip(0.0, 1.0)
        + 0.18 * out["recent_target_spike_score"].astype(float).clip(0.0, 1.0)
        + 0.12 * out["recent_route_spike_score"].astype(float).clip(0.0, 1.0)
        + 0.05 * out["game_script_pass_boost"].astype(float).clip(0.65, 1.35)
    ).clip(0.0, 1.0)
    out["rb_usage_spike_signal"] = (
        0.48 * out["carry_spike_path_score"].astype(float).clip(0.0, 1.0)
        + 0.20 * out["spike_snap_share_score"].astype(float).clip(0.0, 1.0)
        + 0.18 * out["recent_carry_spike_score"].astype(float).clip(0.0, 1.0)
        + 0.09 * out["game_script_rush_boost"].astype(float).clip(0.65, 1.35)
        + 0.05 * (pd.to_numeric(out.get("goal_line_carries_avg_5"), errors="coerce").fillna(0.0).clip(0.0, 2.0) / 2.0)
    ).clip(0.0, 1.0)
    out["qb_volume_spike_signal"] = (
        0.55 * out["pass_spike_path_score"].astype(float).clip(0.0, 1.0)
        + 0.25 * out["high_pass_attempt_score"].astype(float).clip(0.0, 1.0)
        + 0.20 * out["game_script_pass_boost"].astype(float).clip(0.65, 1.35)
    ).clip(0.0, 1.0)
    out["role_change_upside_score"] = (
        0.35 * out["depth_rank_better"].astype(float).clip(0.0, 1.0)
        + 0.35 * out["recent_snap_rise_score"].astype(float).clip(0.0, 1.0)
        + 0.15 * out["recent_target_spike_score"].astype(float).clip(0.0, 1.0)
        + 0.15 * out["recent_carry_spike_score"].astype(float).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["workload_floor_score"] = (
        out["full_workload_score"].astype(float).clip(0.0, 1.0)
        * (1.0 - out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.55 * out["weird_usage_risk_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    receiver_pos_gate = pos.isin(["RB", "WR", "TE"]).astype(float)
    route_volume_score = sigmoid_score(out["route_weighted_target_expectation_avg_5"], 5.8, 1.8)
    share_volume_score = sigmoid_score(target_role, 0.18, 0.045)
    first_read_volume_score = sigmoid_score(out["first_read_proxy_avg_5"], 0.17, 0.050)
    target_earning_score = sigmoid_score(out["targets_per_route_proxy_avg_5"], 0.22, 0.055)
    wopr_volume_score = sigmoid_score(wopr, 0.48, 0.11)
    target_trend_score = sigmoid_score(out["targets_trend_3_10"].astype(float), 0.55, 0.65)
    target_share_trend_score = sigmoid_score(out["target_share_trend_3_10"].astype(float), 0.018, 0.026)
    air_share_score = sigmoid_score(num("air_yards_share_avg_5", 0.0).clip(0.0, 1.0), 0.24, 0.065)
    air_share_trend_score = sigmoid_score(out["air_yards_share_trend_3_10"].astype(float), 0.022, 0.032)
    air_volume_score = sigmoid_score(num("receiving_air_yards_avg_5", 0.0).clip(lower=0.0), 58.0, 24.0)
    air_yards_rate_score = sigmoid_score(out["receiver_yards_rate_signal"].astype(float), 48.0, 20.0)
    route_cut = pd.Series(0.72, index=out.index, dtype=float)
    route_cut.loc[pos == "RB"] = 0.42
    route_cut.loc[pos == "TE"] = 0.66
    route_stability_score = sigmoid_score(route_participation_proxy, route_cut, 0.090)
    route_depth_score = sigmoid_score(estimated_routes.fillna(0.0), 30.0, 6.0)
    pass_boost_score = ((out["game_script_pass_boost"].astype(float).clip(0.65, 1.35) - 0.65) / 0.70).clip(0.0, 1.0)
    receiver_health_gate = (
        (0.50 + 0.50 * out["workload_floor_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.42 * out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.30 * out["injury_downgrade_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.24 * out["depth_movement_risk_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.20 * out["weird_usage_risk_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.15, 1.0)
    out["receiver_teammate_vacancy_score"] = (
        receiver_pos_gate
        * out["teammate_receiver_injury_pressure_score"].astype(float).clip(0.0, 1.0)
        * (0.45 + 0.55 * out["receiver_route_quality_score"].astype(float).clip(0.0, 1.5) / 1.5)
        * (0.60 + 0.40 * out["same_week_context_confidence_score"].astype(float).clip(0.0, 1.0))
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_target_eruption_score"] = (
        receiver_pos_gate
        * (
            0.24 * out["high_target_score"].astype(float).clip(0.0, 1.0)
            + 0.18 * route_volume_score.astype(float).clip(0.0, 1.0)
            + 0.15 * share_volume_score.astype(float).clip(0.0, 1.0)
            + 0.14 * target_earning_score.astype(float).clip(0.0, 1.0)
            + 0.11 * target_trend_score.astype(float).clip(0.0, 1.0)
            + 0.09 * target_share_trend_score.astype(float).clip(0.0, 1.0)
            + 0.06 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
            + 0.03 * pass_boost_score.astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.94)
    out["air_yards_spike_path_score"] = (
        receiver_pos_gate
        * (
            0.23 * air_volume_score.astype(float).clip(0.0, 1.0)
            + 0.20 * air_share_score.astype(float).clip(0.0, 1.0)
            + 0.17 * air_share_trend_score.astype(float).clip(0.0, 1.0)
            + 0.15 * wopr_volume_score.astype(float).clip(0.0, 1.0)
            + 0.10 * first_read_volume_score.astype(float).clip(0.0, 1.0)
            + 0.09 * air_yards_rate_score.astype(float).clip(0.0, 1.0)
            + 0.06 * pass_boost_score.astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.94)
    out["receiver_air_yards_eruption_score"] = (
        receiver_pos_gate
        * pd.concat([
            out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
            0.90 * air_volume_score.astype(float).clip(0.0, 1.0),
            0.88 * air_share_trend_score.astype(float).clip(0.0, 1.0),
            0.84 * wopr_volume_score.astype(float).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * receiver_health_gate
    ).clip(0.02, 0.95)
    out["receiver_explosive_spike_score"] = (
        receiver_pos_gate
        * (
            0.40 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.34 * out["receiver_air_yards_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.16 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * pass_boost_score.astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.95)
    out["target_spike_path_score"] = pd.concat([
        out["target_spike_path_score"].astype(float).clip(0.0, 1.0),
        0.90 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0),
        0.72 * out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["receiver_target_eruption_anchor_targets"] = (
        receiver_pos_gate
        * pd.concat([
            out["route_weighted_target_expectation_avg_5"].astype(float).clip(lower=0.0),
            targets.astype(float).clip(lower=0.0) + out["targets_trend_3_10"].astype(float).clip(lower=0.0),
            estimated_routes.fillna(0.0).astype(float).clip(lower=0.0) * out["targets_per_route_proxy_avg_5"].fillna(0.0).astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.96 + 0.24 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["receiver_air_yards_spike_anchor_yards"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_yards_rate_signal"].astype(float).clip(lower=0.0),
            out["route_weighted_yard_expectation_avg_5"].astype(float).clip(lower=0.0),
            num("receiving_air_yards_avg_5", 0.0).clip(lower=0.0) * 0.78,
            receiving_yards.astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.92 + 0.22 * out["receiver_air_yards_eruption_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["receiver_usage_spike_signal"] = pd.concat([
        out["receiver_usage_spike_signal"].astype(float).clip(0.0, 1.0) if "receiver_usage_spike_signal" in out.columns else pd.Series(0.0, index=out.index),
        0.88 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0),
        0.82 * out["receiver_explosive_spike_score"].astype(float).clip(0.0, 1.0),
        0.70 * out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["receiver_spike_volume_score"] = (
        receiver_pos_gate
        * (
            0.18 * out["high_target_score"].astype(float).clip(0.0, 1.0)
            + 0.18 * route_volume_score.astype(float).clip(0.0, 1.0)
            + 0.13 * share_volume_score.astype(float).clip(0.0, 1.0)
            + 0.12 * first_read_volume_score.astype(float).clip(0.0, 1.0)
            + 0.13 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * out["receiver_explosive_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.09 * out["receiver_usage_spike_signal"].astype(float).clip(0.0, 1.0)
            + 0.05 * pass_boost_score.astype(float).clip(0.0, 1.0)
            + 0.07 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0)
            + 0.05 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.92)
    out["receiver_target_command_score"] = (
        receiver_pos_gate
        * (
            0.28 * out["high_target_score"].astype(float).clip(0.0, 1.0)
            + 0.21 * share_volume_score.astype(float).clip(0.0, 1.0)
            + 0.18 * first_read_volume_score.astype(float).clip(0.0, 1.0)
            + 0.18 * target_earning_score.astype(float).clip(0.0, 1.0)
            + 0.13 * wopr_volume_score.astype(float).clip(0.0, 1.0)
            + 0.02 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
        )
        * (0.70 + 0.30 * out["role_continuity_score"].astype(float).clip(0.0, 1.0))
        * receiver_health_gate
    ).clip(0.02, 0.94)
    out["receiver_route_spike_readiness_score"] = (
        receiver_pos_gate
        * (
            0.30 * route_stability_score.astype(float).clip(0.0, 1.0)
            + 0.24 * route_depth_score.astype(float).clip(0.0, 1.0)
            + 0.16 * out["recent_route_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.14 * out["recent_snap_rise_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0)
            + 0.06 * out["starter_role_stability_score"].astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.94)
    out["receiver_target_route_spike_score"] = (
        receiver_pos_gate
        * (
            0.27 * out["receiver_target_command_score"].astype(float).clip(0.0, 1.0)
            + 0.20 * out["receiver_route_spike_readiness_score"].astype(float).clip(0.0, 1.0)
            + 0.13 * out["receiver_spike_volume_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * route_volume_score.astype(float).clip(0.0, 1.0)
            + 0.10 * out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.09 * pass_boost_score.astype(float).clip(0.0, 1.0)
            + 0.06 * out["target_spike_path_score"].astype(float).clip(0.0, 1.0)
            + 0.04 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0)
            + 0.01 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.02, 0.95)
    out["receiver_contextual_spike_score"] = (
        receiver_pos_gate
        * (
            0.36 * out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.19 * out["receiver_target_command_score"].astype(float).clip(0.0, 1.0)
            + 0.16 * out["receiver_route_spike_readiness_score"].astype(float).clip(0.0, 1.0)
            + 0.13 * out["receiver_explosive_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.14 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
            + 0.02 * pass_boost_score.astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_target_route_spike_anchor_targets"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_target_eruption_anchor_targets"].astype(float).clip(lower=0.0),
            out["route_weighted_target_expectation_avg_5"].astype(float).clip(lower=0.0),
            targets.astype(float).clip(lower=0.0) + 0.75 * out["targets_trend_3_10"].astype(float).clip(lower=0.0),
            estimated_routes.fillna(0.0).astype(float).clip(lower=0.0) * out["targets_per_route_proxy_avg_5"].fillna(0.0).astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.92 + 0.22 * out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    yards_per_target_proxy = (
        receiving_yards.replace(0.0, np.nan) / targets.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    yards_per_target_proxy = yards_per_target_proxy.fillna(
        out["yards_per_route_proxy_avg_5"].fillna(0.0).astype(float).clip(lower=0.0)
        / out["targets_per_route_proxy_avg_5"].replace(0.0, np.nan).astype(float)
    ).replace([np.inf, -np.inf], np.nan).fillna(8.4).clip(3.5, 18.0)
    out["receiver_target_route_spike_anchor_yards"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_target_route_spike_anchor_targets"].astype(float).clip(lower=0.0) * yards_per_target_proxy,
            out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_yards_rate_signal"].astype(float).clip(lower=0.0),
            out["route_weighted_yard_expectation_avg_5"].astype(float).clip(lower=0.0),
            receiving_yards.astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.90 + 0.18 * out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    receiver_anchor = pd.concat([
        out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
        out["route_weighted_yard_expectation_avg_5"].astype(float).clip(lower=0.0),
        out["receiver_yards_rate_signal"].astype(float).clip(lower=0.0),
        receiving_yards.astype(float).clip(lower=0.0),
    ], axis=1).max(axis=1).fillna(0.0)
    out["receiver_spike_volume_anchor_yards"] = (
        receiver_pos_gate
        * receiver_anchor
        * (0.82 + 0.30 * out["receiver_spike_volume_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["workload_downside_v2_score"] = pd.concat([
        out["limited_workload_risk_score"].astype(float).clip(0.0, 1.0),
        out["rest_risk_score"].astype(float).clip(0.0, 1.0),
        out["weird_usage_risk_score"].astype(float).clip(0.0, 1.0),
        out["injury_downgrade_score"].astype(float).clip(0.0, 1.0),
        out["depth_movement_risk_score"].astype(float).clip(0.0, 1.0),
        out["high_usage_fragility_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["receiver_projected_targets_v2"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_target_route_spike_anchor_targets"].astype(float).clip(lower=0.0),
            out["receiver_target_eruption_anchor_targets"].astype(float).clip(lower=0.0),
            out["route_weighted_target_expectation_avg_5"].astype(float).clip(lower=0.0),
            targets.astype(float).clip(lower=0.0) + 0.75 * out["targets_trend_3_10"].astype(float).clip(lower=0.0),
            targets.astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.88 + 0.20 * out["receiver_target_command_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.30 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["receiver_high_value_target_score"] = (
        receiver_pos_gate
        * (
            0.24 * out["receiver_target_command_score"].astype(float).clip(0.0, 1.0)
            + 0.20 * out["receiver_route_spike_readiness_score"].astype(float).clip(0.0, 1.0)
            + 0.18 * out["receiver_air_yards_eruption_score"].astype(float).clip(0.0, 1.0)
            + 0.14 * out["receiving_td_route_redzone_score"].astype(float).clip(0.0, 4.0) / 4.0
            + 0.12 * out["first_read_proxy_avg_5"].astype(float).fillna(0.0).clip(0.0, 1.0)
            + 0.12 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_target_spike_v2_score"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0),
            out["receiver_target_eruption_score"].astype(float).clip(0.0, 1.0),
            out["receiver_usage_spike_signal"].astype(float).clip(0.0, 1.0),
            out["receiver_high_value_target_score"].astype(float).clip(0.0, 1.0),
            out["target_spike_path_score"].astype(float).clip(0.0, 1.0),
            0.85 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0),
            0.80 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_air_yards_spike_v2_score"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_air_yards_eruption_score"].astype(float).clip(0.0, 1.0),
            out["air_yards_spike_path_score"].astype(float).clip(0.0, 1.0),
            air_share_score.astype(float).clip(0.0, 1.0),
            air_share_trend_score.astype(float).clip(0.0, 1.0),
            wopr_volume_score.astype(float).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_ypt_efficiency_spike_score"] = (
        receiver_pos_gate
        * (
            0.30 * sigmoid_score(yards_per_target_proxy, 9.4, 2.1).astype(float).clip(0.0, 1.0)
            + 0.24 * air_yards_rate_score.astype(float).clip(0.0, 1.0)
            + 0.18 * out["receiver_air_yards_spike_v2_score"].astype(float).clip(0.0, 1.0)
            + 0.16 * out["receiver_high_value_target_score"].astype(float).clip(0.0, 1.0)
            + 0.12 * out["receiver_explosive_spike_score"].astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_spike_yards_anchor_v2"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_target_route_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_spike_volume_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_projected_targets_v2"].astype(float).clip(lower=0.0) * yards_per_target_proxy,
            out["route_weighted_yard_expectation_avg_5"].astype(float).clip(lower=0.0),
            receiving_yards.astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (
            0.88
            + 0.12 * out["receiver_target_spike_v2_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * out["receiver_air_yards_spike_v2_score"].astype(float).clip(0.0, 1.0)
        )
        * (1.0 - 0.22 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["rb_projected_carries_v2"] = (
        (num("carries_avg_5", 0.0).fillna(0.0).clip(lower=0.0) + 0.65 * out["carries_trend_3_10"].astype(float).fillna(0.0).clip(lower=0.0))
        * out["game_script_rush_boost"].astype(float).clip(0.65, 1.35)
        * (0.72 + 0.28 * out["full_workload_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.36 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["rb_carry_spike_v2_score"] = (
        pos.eq("RB").astype(float)
        * pd.concat([
            out["high_carry_score"].astype(float).clip(0.0, 1.0),
            out["carry_spike_path_score"].astype(float).clip(0.0, 1.0),
            out["rb_usage_spike_signal"].astype(float).clip(0.0, 1.0),
            out["recent_carry_spike_score"].astype(float).clip(0.0, 1.0),
            0.80 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0),
            0.55 * (num("goal_line_carries_avg_5", 0.0).fillna(0.0).clip(0.0, 2.0) / 2.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.55 + 0.45 * out["full_workload_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.40 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["rb_rush_yards_anchor_v2"] = pd.concat([
        out["rb_spike_rush_score"].astype(float).clip(lower=0.0),
        out["rb_rush_role_env_score"].astype(float).clip(lower=0.0),
        out["rb_projected_carries_v2"].astype(float).clip(lower=0.0) * num("yards_per_carry_avg_5", 4.1).fillna(4.1).clip(2.2, 6.8),
        num("rushing_yards_avg_5", 0.0).fillna(0.0).clip(lower=0.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(lower=0.0)
    out["workload_upside_v2_score"] = pd.concat([
        out["receiver_target_spike_v2_score"].astype(float).clip(0.0, 1.0),
        out["rb_carry_spike_v2_score"].astype(float).clip(0.0, 1.0),
        out["pass_spike_path_score"].astype(float).clip(0.0, 1.0),
        out["spike_snap_share_score"].astype(float).clip(0.0, 1.0),
        out["role_change_upside_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["receiving_td_rare_event_score_v2"] = (
        receiver_pos_gate
        * (
            0.30 * sigmoid_score(num("red_zone_targets_avg_5", 0.0).fillna(0.0), 1.1, 0.65).astype(float).clip(0.0, 1.0)
            + 0.22 * sigmoid_score(num("goal_line_targets_avg_5", 0.0).fillna(0.0) + num("end_zone_targets_avg_5", 0.0).fillna(0.0), 0.45, 0.30).astype(float).clip(0.0, 1.0)
            + 0.16 * out["receiver_high_value_target_score"].astype(float).clip(0.0, 1.0)
            + 0.14 * out["first_read_proxy_avg_5"].astype(float).fillna(0.0).clip(0.0, 1.0)
            + 0.10 * (team_env - 0.65).clip(0.0, 0.80) / 0.80
            + 0.08 * sigmoid_score(num("opp_allowed_receiving_tds_avg_5", 0.0).fillna(0.0), 1.25, 0.55).astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.0, 1.0)
    same_week_usage_known = (
        0.34 * out.get("roster_is_active", pd.Series(False, index=out.index)).astype("boolean").fillna(False).astype(float)
        + 0.22 * pd.to_numeric(out.get("depth_pos_rank", pd.Series(np.nan, index=out.index)), errors="coerce").notna().astype(float)
        + 0.18 * out.get("injury_report_status", pd.Series(index=out.index, dtype=object)).notna().astype(float)
        + 0.16 * pd.to_numeric(out.get("same_week_teammate_skill_injury_count", pd.Series(np.nan, index=out.index)), errors="coerce").notna().astype(float)
        + 0.10 * out["role_continuity_score"].astype(float).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["same_week_usage_confidence_v3_score"] = same_week_usage_known
    out["receiver_live_spike_v3_score"] = (
        receiver_pos_gate
        * (
            0.20 * out["receiver_target_spike_v2_score"].astype(float).clip(0.0, 1.0)
            + 0.18 * out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.14 * out["receiver_air_yards_spike_v2_score"].astype(float).clip(0.0, 1.0)
            + 0.12 * out["receiver_high_value_target_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * out["receiver_ypt_efficiency_spike_score"].astype(float).clip(0.0, 1.0)
            + 0.10 * out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0)
            + 0.07 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0)
            + 0.05 * pass_boost_score.astype(float).clip(0.0, 1.0)
            + 0.04 * same_week_usage_known.astype(float).clip(0.0, 1.0)
        )
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_projected_targets_v3"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_projected_targets_v2"].astype(float).clip(lower=0.0),
            out["receiver_target_route_spike_anchor_targets"].astype(float).clip(lower=0.0),
            out["receiver_target_eruption_anchor_targets"].astype(float).clip(lower=0.0),
            out["route_weighted_target_expectation_avg_5"].astype(float).clip(lower=0.0),
            estimated_routes.fillna(0.0).astype(float).clip(lower=0.0) * out["targets_per_route_proxy_avg_5"].fillna(0.0).astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.84 + 0.18 * out["receiver_live_spike_v3_score"].astype(float).clip(0.0, 1.0))
        * (0.90 + 0.10 * same_week_usage_known.astype(float).clip(0.0, 1.0))
        * (1.0 - 0.24 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["receiver_spike_yards_anchor_v3"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_spike_yards_anchor_v2"].astype(float).clip(lower=0.0),
            out["receiver_target_route_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_projected_targets_v3"].astype(float).clip(lower=0.0) * yards_per_target_proxy,
            out["route_weighted_yard_expectation_avg_5"].astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.88 + 0.14 * out["receiver_live_spike_v3_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.18 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["rb_live_carry_v3_score"] = (
        pos.eq("RB").astype(float)
        * pd.concat([
            out["rb_carry_spike_v2_score"].astype(float).clip(0.0, 1.0),
            out["high_carry_score"].astype(float).clip(0.0, 1.0),
            out["carry_spike_path_score"].astype(float).clip(0.0, 1.0),
            out["rb_usage_spike_signal"].astype(float).clip(0.0, 1.0),
            0.75 * out["role_change_upside_score"].astype(float).clip(0.0, 1.0),
            0.65 * out["spike_snap_share_score"].astype(float).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.58 + 0.42 * out["full_workload_score"].astype(float).clip(0.0, 1.0))
        * (0.88 + 0.12 * same_week_usage_known.astype(float).clip(0.0, 1.0))
        * (1.0 - 0.34 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["rb_projected_carries_v3"] = (
        pd.concat([
            out["rb_projected_carries_v2"].astype(float).clip(lower=0.0),
            num("carries_avg_5", 0.0).fillna(0.0).clip(lower=0.0) + 0.85 * out["carries_trend_3_10"].astype(float).fillna(0.0).clip(lower=0.0),
            num("carries_avg_10", 0.0).fillna(0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * out["game_script_rush_boost"].astype(float).clip(0.65, 1.35)
        * (0.82 + 0.16 * out["rb_live_carry_v3_score"].astype(float).clip(0.0, 1.0))
        * (1.0 - 0.26 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["rb_rush_yards_anchor_v3"] = pd.concat([
        out["rb_rush_yards_anchor_v2"].astype(float).clip(lower=0.0),
        out["rb_projected_carries_v3"].astype(float).clip(lower=0.0) * num("yards_per_carry_avg_5", 4.1).fillna(4.1).clip(2.2, 6.8),
        out["rb_spike_rush_score"].astype(float).clip(lower=0.0),
        num("rushing_yards_avg_5", 0.0).fillna(0.0).clip(lower=0.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(lower=0.0)
    out["yardage_projection_volatility_v3_score"] = pd.concat([
        out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0),
        out["usage_volatility_score"].astype(float).clip(0.0, 1.0),
        out["high_usage_fragility_score"].astype(float).clip(0.0, 1.0),
        1.0 - same_week_usage_known.astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    receiver_evidence_quality = (
        0.45 * out["receiving_usage_history_quality_score"].astype(float).clip(0.0, 1.0)
        + 0.35 * same_week_usage_known.astype(float).clip(0.0, 1.0)
        + 0.20 * out["has_true_route_history"].astype(float).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    rb_evidence_quality = (
        0.55 * out["rb_usage_history_quality_score"].astype(float).clip(0.0, 1.0)
        + 0.35 * same_week_usage_known.astype(float).clip(0.0, 1.0)
        + 0.10 * out["role_continuity_score"].astype(float).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["live_usage_context_quality_v4_score"] = pd.concat([
        receiver_evidence_quality,
        rb_evidence_quality,
        same_week_usage_known.astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    out["receiver_spike_under_correction_v4_score"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_live_spike_v3_score"].astype(float).clip(0.0, 1.0),
            out["receiver_target_route_spike_score"].astype(float).clip(0.0, 1.0),
            out["receiver_target_spike_v2_score"].astype(float).clip(0.0, 1.0),
            out["receiver_air_yards_spike_v2_score"].astype(float).clip(0.0, 1.0),
            out["receiver_ypt_efficiency_spike_score"].astype(float).clip(0.0, 1.0),
            out["receiver_teammate_vacancy_score"].astype(float).clip(0.0, 1.0),
            out["role_change_upside_score"].astype(float).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.58 + 0.42 * receiver_evidence_quality)
        * receiver_health_gate
    ).clip(0.0, 1.0)
    out["receiver_spike_yards_anchor_v4"] = (
        receiver_pos_gate
        * pd.concat([
            out["receiver_spike_yards_anchor_v3"].astype(float).clip(lower=0.0),
            out["receiver_target_route_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_air_yards_spike_anchor_yards"].astype(float).clip(lower=0.0),
            out["receiver_projected_targets_v3"].astype(float).clip(lower=0.0) * yards_per_target_proxy,
            out["route_weighted_yard_expectation_avg_5"].astype(float).clip(lower=0.0),
            receiving_yards.astype(float).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.92 + 0.20 * out["receiver_spike_under_correction_v4_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["rb_carry_under_correction_v4_score"] = (
        pos.eq("RB").astype(float)
        * pd.concat([
            out["rb_live_carry_v3_score"].astype(float).clip(0.0, 1.0),
            out["rb_carry_spike_v2_score"].astype(float).clip(0.0, 1.0),
            out["high_carry_score"].astype(float).clip(0.0, 1.0),
            out["carry_spike_path_score"].astype(float).clip(0.0, 1.0),
            out["role_change_upside_score"].astype(float).clip(0.0, 1.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.60 + 0.40 * rb_evidence_quality)
        * (1.0 - 0.30 * out["workload_downside_v2_score"].astype(float).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["rb_rush_yards_anchor_v4"] = (
        pd.concat([
            out["rb_rush_yards_anchor_v3"].astype(float).clip(lower=0.0),
            out["rb_rush_yards_anchor_v2"].astype(float).clip(lower=0.0),
            out["rb_projected_carries_v3"].astype(float).clip(lower=0.0) * num("yards_per_carry_avg_5", 4.1).fillna(4.1).clip(2.2, 6.8),
            num("rushing_yards_avg_5", 0.0).fillna(0.0).clip(lower=0.0),
        ], axis=1).max(axis=1).fillna(0.0)
        * (0.92 + 0.16 * out["rb_carry_under_correction_v4_score"].astype(float).clip(0.0, 1.0))
    ).clip(lower=0.0)
    out["yardage_projection_volatility_v4_score"] = pd.concat([
        out["yardage_projection_volatility_v3_score"].astype(float).clip(0.0, 1.0),
        1.0 - out["live_usage_context_quality_v4_score"].astype(float).clip(0.0, 1.0),
        out["usage_volatility_score"].astype(float).clip(0.0, 1.0),
        out["high_usage_fragility_score"].astype(float).clip(0.0, 1.0),
    ], axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    return out


def build_feature_frame(conn, cfg: FeatureBuildConfig) -> pd.DataFrame:
    df = _load_logs(conn, cfg.min_season)
    if df.empty:
        return df
    df = _add_player_rolling(df)
    df = _add_team_context(df)
    df = _add_defense_allowed(df)
    df = _add_usage_role_features(df)
    df = _add_context_risk_features(df)
    df["is_home"] = df["is_home"].astype("boolean")
    return df


FEATURE_COLUMNS = [
    "season", "week", "season_type", "game_id", "game_date_et", "player_id", "player_name",
    "team_abbr", "opponent_abbr", "position", "is_home", "n_games_prev_3",
    "n_games_prev_5", "n_games_prev_10", "rest_days", "team_game_number",
    "opp_game_number", *TARGET_STATS, *USAGE_STATS,
    "roster_status", "roster_is_active", "depth_chart_position", "depth_pos_abb",
    "depth_pos_rank", "depth_pos_slot", "injury_report_status", "injury_practice_status",
    "same_week_teammate_skill_injury_count", "same_week_teammate_skill_injury_score",
    "same_week_teammate_receiver_injury_score", "same_week_teammate_receiver_out_count",
    "game_total_line", "team_spread_line", "team_implied_points", "opponent_implied_points",
    *[f"{stat}_avg_{window}" for stat in ROLLING_STATS for window in (3, 5, 10)],
    *[f"opp_allowed_{stat}_avg_5" for stat in TARGET_STATS],
    *[f"team_player_{stat}_avg5_sum" for stat in ROLE_SIGNAL_STATS],
    *[f"{stat}_share_avg_5" for stat in ROLE_SIGNAL_STATS],
    *[f"{stat}_role_rank" for stat in ROLE_SIGNAL_STATS],
    "is_week_18", "is_late_season", "is_postseason",
    "starter_confidence", "rest_risk_score", "injury_downgrade_score",
    "depth_rank_delta", "depth_rank_worse", "depth_rank_better", "depth_rank_change_abs",
    "has_true_route_history", "estimated_routes_avg_5",
    "route_participation_proxy_avg_5", "targets_per_route_proxy_avg_5",
    "yards_per_route_proxy_avg_5", "route_target_intensity_avg_5",
    "route_weighted_target_expectation_avg_5", "route_weighted_yard_expectation_avg_5",
    "red_zone_targets_per_route_proxy_avg_5",
    "first_read_proxy_avg_5",
    "snap_share_trend_3_10", "snap_share_volatility_5",
    "targets_trend_3_10", "carries_trend_3_10", "receptions_trend_3_10",
    "receiving_yards_trend_3_10", "rushing_yards_trend_3_10",
    "target_share_trend_3_10", "air_yards_share_trend_3_10", "wopr_trend_3_10",
    "route_share_trend_3_10", "recent_snap_drop_score", "recent_snap_rise_score",
    "recent_target_spike_score", "recent_carry_spike_score", "recent_route_spike_score",
    "usage_volatility_score",
    "limited_workload_risk_score", "full_workload_score", "high_usage_fragility_score",
    "same_week_context_confidence_score", "teammate_skill_injury_pressure_score",
    "teammate_receiver_injury_pressure_score", "receiver_teammate_vacancy_score",
    "receiver_target_eruption_score", "air_yards_spike_path_score",
    "receiver_air_yards_eruption_score", "receiver_explosive_spike_score",
    "receiver_target_eruption_anchor_targets", "receiver_air_yards_spike_anchor_yards",
    "receiver_contextual_spike_score",
    "backup_role_score", "starter_role_stability_score",
    "projected_starter_score", "depth_movement_risk_score", "weird_usage_risk_score",
    "role_continuity_score", "normal_usage_path_score",
    "high_pass_attempt_score", "high_carry_score", "high_target_score", "spike_snap_share_score",
    "game_script_pass_boost", "game_script_rush_boost",
    "spike_target_opportunity_score", "spike_carry_opportunity_score", "spike_pass_attempt_opportunity_score",
    "target_spike_path_score", "carry_spike_path_score", "pass_spike_path_score",
    "receiver_route_quality_score", "receiver_yards_rate_signal", "receiver_route_env_score",
    "receiver_spike_yards_score",
    "rb_rush_role_env_score", "rb_carry_trend_env_score", "rb_spike_rush_score",
    "td_goal_line_env_score", "receiving_td_role_score", "receiving_td_route_redzone_score",
    "receiving_td_any_score",
    "receiver_usage_spike_signal", "rb_usage_spike_signal", "qb_volume_spike_signal",
    "role_change_upside_score", "workload_floor_score",
    "receiver_spike_volume_score", "receiver_spike_volume_anchor_yards",
    "receiver_target_command_score", "receiver_route_spike_readiness_score",
    "receiver_target_route_spike_score", "receiver_target_route_spike_anchor_targets",
    "receiver_target_route_spike_anchor_yards",
    "workload_downside_v2_score", "workload_upside_v2_score",
    "receiver_projected_targets_v2", "receiver_high_value_target_score",
    "receiver_target_spike_v2_score", "receiver_air_yards_spike_v2_score",
    "receiver_ypt_efficiency_spike_score", "receiver_spike_yards_anchor_v2",
    "rb_projected_carries_v2", "rb_carry_spike_v2_score", "rb_rush_yards_anchor_v2",
    "receiving_td_rare_event_score_v2",
    "same_week_usage_confidence_v3_score", "receiver_live_spike_v3_score",
    "receiver_projected_targets_v3", "receiver_spike_yards_anchor_v3",
    "rb_live_carry_v3_score", "rb_projected_carries_v3", "rb_rush_yards_anchor_v3",
    "yardage_projection_volatility_v3_score",
    "receiving_usage_history_quality_score", "rb_usage_history_quality_score",
    "td_usage_history_quality_score", "live_usage_context_quality_v4_score",
    "receiver_spike_under_correction_v4_score", "receiver_spike_yards_anchor_v4",
    "rb_carry_under_correction_v4_score", "rb_rush_yards_anchor_v4",
    "yardage_projection_volatility_v4_score",
    *[f"{stat}_std_{window}" for stat in VOLATILITY_STATS for window in (5, 10)],
]


def _row_tuple(row: pd.Series) -> tuple:
    out: list[Any] = []
    for col in FEATURE_COLUMNS:
        value = row.get(col)
        try:
            if pd.isna(value):
                value = None
        except Exception:
            pass
        if col == "is_home" and value is not None:
            value = bool(value)
        out.append(value)
    return tuple(out)


def write_features(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = None
    rows = [_row_tuple(row) for _, row in df[FEATURE_COLUMNS].iterrows()]
    columns_sql = ", ".join(FEATURE_COLUMNS)
    value_cols = [col for col in FEATURE_COLUMNS if col not in {"game_id", "player_id", "team_abbr"}]
    update_sql = ", ".join(f"{col} = EXCLUDED.{col}" for col in value_cols)
    # Skip no-op rewrites so updated_at_utc keeps meaning "last content change".
    changed_sql = (
        f"({', '.join(f'features.nfl_player_game_training_features.{col}' for col in value_cols)})"
        f" IS DISTINCT FROM ({', '.join(f'EXCLUDED.{col}' for col in value_cols)})"
    )
    sql = f"""
        INSERT INTO features.nfl_player_game_training_features ({columns_sql})
        VALUES %s
        ON CONFLICT (game_id, player_id, team_abbr) DO UPDATE SET
            {update_sql},
            updated_at_utc = NOW()
        WHERE {changed_sql}
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=5000)
    conn.commit()
    return len(rows)


def build_and_write(cfg: FeatureBuildConfig) -> dict[str, Any]:
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_schema(conn)
        df = build_feature_frame(conn, cfg)
        rows = write_features(conn, df)
    return {"status": "ok", "feature_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL player-game training features")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--min-season", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    print(json.dumps(build_and_write(FeatureBuildConfig(pg_dsn=args.pg_dsn, min_season=args.min_season)), indent=2))


if __name__ == "__main__":
    main()
