"""Immutable daily MLB forecasts before offer and bet selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from typing import Any, Iterable

import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd

from mlb_pipeline.db import PG_DSN as _PG_DSN
from .model_release import hitter_rate_shadow_release_id

_SCHEMA_READY = False
TERMINAL_NON_FINAL_GAME_STATUSES = ("postponed", "cancelled", "canceled")


DDL = """
CREATE SCHEMA IF NOT EXISTS bets;
CREATE TABLE IF NOT EXISTS bets.mlb_daily_forecast_ledger (
    id BIGSERIAL PRIMARY KEY,
    forecast_key TEXT NOT NULL UNIQUE,
    forecast_run_id TEXT NOT NULL,
    forecast_phase TEXT NOT NULL,
    locked_at_utc TIMESTAMPTZ NOT NULL,
    game_date_et DATE NOT NULL,
    game_slug TEXT NOT NULL,
    forecast_type TEXT NOT NULL,
    player_id BIGINT,
    player_name TEXT,
    team_abbr TEXT,
    opponent_abbr TEXT,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    stat TEXT NOT NULL,
    projection_value NUMERIC NOT NULL,
    model_only_projection NUMERIC,
    baseline_value NUMERIC,
    baseline_source TEXT,
    probability_value NUMERIC,
    distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    opportunity_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_family TEXT,
    model_version TEXT,
    result_status TEXT NOT NULL DEFAULT 'pending',
    actual_value NUMERIC,
    absolute_error NUMERIC,
    squared_error NUMERIC,
    baseline_absolute_error NUMERIC,
    graded_at_utc TIMESTAMPTZ,
    grade_source TEXT,
    inserted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT mlb_daily_forecast_type_ck CHECK (forecast_type IN ('game', 'player', 'opportunity'))
);
CREATE INDEX IF NOT EXISTS idx_mlb_daily_forecast_date
    ON bets.mlb_daily_forecast_ledger (game_date_et, forecast_type, stat);
CREATE INDEX IF NOT EXISTS idx_mlb_daily_forecast_pending
    ON bets.mlb_daily_forecast_ledger (result_status, game_date_et);
CREATE INDEX IF NOT EXISTS idx_mlb_daily_forecast_player
    ON bets.mlb_daily_forecast_ledger (player_id, game_date_et, stat);
"""


INSERT_SQL = """
INSERT INTO bets.mlb_daily_forecast_ledger (
    forecast_key, forecast_run_id, forecast_phase, locked_at_utc,
    game_date_et, game_slug, forecast_type, player_id, player_name,
    team_abbr, opponent_abbr, home_team_abbr, away_team_abbr, stat,
    projection_value, model_only_projection, baseline_value, baseline_source,
    probability_value, distribution, opportunity_context, model_family, model_version
) VALUES (
    %(forecast_key)s, %(forecast_run_id)s, %(forecast_phase)s, %(locked_at_utc)s,
    %(game_date_et)s, %(game_slug)s, %(forecast_type)s, %(player_id)s, %(player_name)s,
    %(team_abbr)s, %(opponent_abbr)s, %(home_team_abbr)s, %(away_team_abbr)s, %(stat)s,
    %(projection_value)s, %(model_only_projection)s, %(baseline_value)s, %(baseline_source)s,
    %(probability_value)s, %(distribution)s, %(opportunity_context)s, %(model_family)s, %(model_version)s
)
ON CONFLICT (forecast_key) DO NOTHING
"""


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if value is pd.NA:
        return None
    return value


def _json(value: Any) -> psycopg2.extras.Json:
    return psycopg2.extras.Json(_json_safe(value if isinstance(value, (dict, list)) else {}))


def current_forecast_run_id(game_date: date | str | None = None) -> str:
    configured = os.getenv("MLB_FORECAST_RUN_ID", "").strip()
    if configured:
        return configured
    day = str(game_date or os.getenv("MLB_ET_DATE") or date.today().isoformat())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"manual:{day}:{stamp}"


def current_forecast_phase() -> str:
    return os.getenv("MLB_FORECAST_PHASE", "manual").strip() or "manual"


def ensure_daily_forecast_ledger_schema(conn) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    _SCHEMA_READY = True


def _forecast_key(row: dict[str, Any]) -> str:
    identity = "|".join(str(row.get(name) or "") for name in (
        "forecast_run_id", "forecast_phase", "game_date_et", "game_slug",
        "forecast_type", "player_id", "stat",
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _explicit_forecast_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()


def insert_forecasts(conn, rows: Iterable[dict[str, Any]]) -> int:
    payload = []
    for source in rows:
        row = dict(source)
        if _clean_float(row.get("projection_value")) is None:
            continue
        row["forecast_key"] = row.get("forecast_key") or _forecast_key(row)
        row["distribution"] = _json(row.get("distribution"))
        row["opportunity_context"] = _json(row.get("opportunity_context"))
        payload.append(row)
    if not payload:
        return 0
    ensure_daily_forecast_ledger_schema(conn)
    inserted = 0
    with conn.cursor() as cur:
        for row in payload:
            cur.execute(INSERT_SQL + " RETURNING id", row)
            if cur.fetchone() is not None:
                inserted += 1
    conn.commit()
    return inserted


def _base_row(
    row: dict[str, Any],
    *,
    run_id: str,
    phase: str,
    locked_at: datetime,
    forecast_type: str,
    stat: str,
    projection: Any,
) -> dict[str, Any]:
    return {
        "forecast_run_id": run_id,
        "forecast_phase": phase,
        "locked_at_utc": locked_at,
        "game_date_et": row.get("game_date_et"),
        "game_slug": row.get("game_slug"),
        "forecast_type": forecast_type,
        "player_id": row.get("player_id"),
        "player_name": row.get("player_name"),
        "team_abbr": row.get("team_abbr"),
        "opponent_abbr": row.get("opponent_abbr"),
        "home_team_abbr": row.get("home_team_abbr"),
        "away_team_abbr": row.get("away_team_abbr"),
        "stat": stat,
        "projection_value": _clean_float(projection),
        "model_only_projection": None,
        "baseline_value": None,
        "baseline_source": None,
        "probability_value": None,
        "distribution": {},
        "opportunity_context": row.get("opportunity_context") or {},
        "model_family": row.get("model_family") or "unknown",
        "model_version": row.get("model_version") or os.getenv("MLB_MODEL_VERSION") or "current",
    }


def lock_game_forecasts(
    conn,
    rows: Iterable[dict[str, Any]],
    *,
    run_id: str | None = None,
    phase: str | None = None,
) -> int:
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        return 0
    run_id = run_id or current_forecast_run_id(source_rows[0].get("game_date_et"))
    phase = phase or current_forecast_phase()
    locked_at = datetime.now(timezone.utc)
    forecasts: list[dict[str, Any]] = []
    for row in source_rows:
        sigma_run = _clean_float(row.get("sigma_q_rl")) or 3.5
        sigma_total = _clean_float(row.get("sigma_q_total")) or 3.0
        run_diff = _clean_float(row.get("pred_run_diff"))
        total = _clean_float(row.get("pred_total"))
        if run_diff is not None:
            rec = _base_row(
                row, run_id=run_id, phase=phase, locked_at=locked_at,
                forecast_type="game", stat="game_run_diff", projection=run_diff,
            )
            rec.update({
                "model_only_projection": _clean_float(row.get("pred_run_diff_model_only")),
                "baseline_value": (
                    -_clean_float(row.get("market_run_line"))
                    if _clean_float(row.get("market_run_line")) is not None else None
                ),
                "baseline_source": "market_run_line_threshold",
                "distribution": {"family": "quantile_normal", "sigma": sigma_run},
                "model_family": "game_run_distribution",
            })
            forecasts.append(rec)
            p_home = 0.5 * (1.0 + math.erf(run_diff / (sigma_run * math.sqrt(2.0))))
            home = _base_row(
                row, run_id=run_id, phase=phase, locked_at=locked_at,
                forecast_type="game", stat="game_home_win", projection=p_home,
            )
            home.update({
                "probability_value": p_home,
                "baseline_value": 0.5,
                "baseline_source": "coin_flip",
                "distribution": {"family": "bernoulli"},
                "model_family": "game_home_win_from_run_distribution",
            })
            forecasts.append(home)
        if total is not None:
            rec = _base_row(
                row, run_id=run_id, phase=phase, locked_at=locked_at,
                forecast_type="game", stat="game_total", projection=total,
            )
            rec.update({
                "model_only_projection": _clean_float(row.get("pred_total_model_only")),
                "baseline_value": _clean_float(row.get("market_total")),
                "baseline_source": "market_total",
                "distribution": {"family": "quantile_normal", "sigma": sigma_total},
                "model_family": "game_total_distribution",
            })
            forecasts.append(rec)
    return insert_forecasts(conn, forecasts)


def lock_player_forecasts(
    conn,
    pitcher_rows: Iterable[dict[str, Any]],
    batter_rows: Iterable[dict[str, Any]],
    *,
    run_id: str | None = None,
    phase: str | None = None,
) -> int:
    pitchers = [dict(row) for row in pitcher_rows]
    batters = [dict(row) for row in batter_rows]
    all_rows = pitchers + batters
    if not all_rows:
        return 0
    run_id = run_id or current_forecast_run_id(all_rows[0].get("game_date_et"))
    phase = phase or current_forecast_phase()
    locked_at = datetime.now(timezone.utc)
    forecasts: list[dict[str, Any]] = []

    for row in pitchers:
        context = row.get("opportunity_context") or {}
        specs = (
            ("pitcher_strikeouts", row.get("raw_pred_strikeouts", row.get("pred_strikeouts")), row.get("baseline_strikeouts"), "rolling_k9_x_ip", "player"),
            ("pitcher_batters_faced", row.get("projected_bf"), row.get("baseline_bf"), "rolling_bf_5", "opportunity"),
            ("pitcher_pitch_count", row.get("projected_pitch_count"), row.get("baseline_pitch_count"), "rolling_ip_x_16_5", "opportunity"),
            ("pitcher_innings", context.get("projected_ip"), context.get("pitcher_last_ip"), "last_start_ip", "opportunity"),
        )
        for stat, projection, baseline, baseline_source, forecast_type in specs:
            rec = _base_row(
                row, run_id=run_id, phase=phase, locked_at=locked_at,
                forecast_type=forecast_type, stat=stat, projection=projection,
            )
            rec.update({
                "baseline_value": _clean_float(baseline),
                "baseline_source": baseline_source,
                "distribution": {
                    "family": "count_mean_sigma",
                    "mean": _clean_float(projection),
                    "sigma": _clean_float(row.get("sigma_strikeouts")) if stat == "pitcher_strikeouts" else None,
                },
                "model_family": "pitcher_opportunity_v3" if forecast_type == "opportunity" else "pitcher_k_regression",
            })
            forecasts.append(rec)

    for row in batters:
        context = row.get("opportunity_context") or {}
        sigma_map = row.get("sigma_map") or {}
        specs = (
            ("batter_hits", row.get("raw_pred_hits", row.get("pred_hits")), row.get("baseline_hits"), "rolling_10", "player"),
            ("batter_total_bases", row.get("raw_pred_total_bases", row.get("pred_total_bases")), row.get("baseline_total_bases"), "rolling_10", "player"),
            ("batter_home_runs", row.get("raw_pred_home_runs", row.get("pred_home_runs")), row.get("baseline_home_runs"), "rolling_10", "player"),
            ("hitter_plate_appearances", row.get("projected_pa"), row.get("baseline_projected_pa"), "lineup_slot_prior", "opportunity"),
            ("hitter_plate_appearances_challenger", row.get("validated_projected_pa"), row.get("baseline_projected_pa"), "lineup_slot_prior", "opportunity"),
        )
        for stat, projection, baseline, baseline_source, forecast_type in specs:
            rec = _base_row(
                row, run_id=run_id, phase=phase, locked_at=locked_at,
                forecast_type=forecast_type, stat=stat, projection=projection,
            )
            rec.update({
                "baseline_value": _clean_float(baseline),
                "baseline_source": baseline_source,
                "distribution": {
                    "family": "count_mean_sigma",
                    "mean": _clean_float(projection),
                    "sigma": _clean_float(sigma_map.get(stat)),
                    "low_pa_probability": _clean_float(context.get("low_pa_probability")),
                },
                "model_family": (
                    "hitter_pa_v3_challenger" if stat == "hitter_plate_appearances_challenger"
                    else "hitter_pa_baseline" if stat == "hitter_plate_appearances"
                    else "hitter_hits_direct_player_game_blend"
                    if stat == "batter_hits"
                    and str(context.get("hits_count_source") or "").startswith("validated_direct_player_game_blend")
                    else "hitter_tb_direct_player_game_blend"
                    if stat == "batter_total_bases"
                    and str(context.get("tb_count_source") or "").startswith("validated_direct_player_game_blend")
                    else "hitter_tb_component_rebuild_from_rate_challengers"
                    if stat == "batter_total_bases"
                    and str(context.get("tb_count_source") or "").startswith("component_rebuild_")
                    else "hitter_hr_direct_player_game_blend"
                    if stat == "batter_home_runs"
                    and str(context.get("hr_count_source") or "").startswith("validated_direct_player_game_blend")
                    else "hitter_count_regression"
                ),
            })
            forecasts.append(rec)
    return insert_forecasts(conn, forecasts)


def lock_hitter_rate_shadow_forecasts(
    conn,
    batter_rows: Iterable[dict[str, Any]],
    *,
    run_id: str | None = None,
    phase: str | None = None,
) -> int:
    """Lock accepted hits/HR challengers as shadow forecasts only.

    These rows share the same player/stat targets as production forecasts, but
    their explicit forecast keys and shadow model release keep them out of the
    normal promotion gates until prospective results prove the challenger.
    """
    batters = [dict(row) for row in batter_rows]
    if not batters:
        return 0
    run_id = run_id or current_forecast_run_id(batters[0].get("game_date_et"))
    phase = phase or current_forecast_phase()
    locked_at = datetime.now(timezone.utc)
    release_id = hitter_rate_shadow_release_id()
    family = "hitter_rate_shadow_direct_player_game_blend"
    forecasts: list[dict[str, Any]] = []
    specs = (
        (
            "batter_hits",
            "shadow_raw_pred_hits",
            "shadow_baseline_raw_pred_hits",
            "shadow_hits_direct_prediction",
            "shadow_hits_blend_alpha",
        ),
        (
            "batter_home_runs",
            "shadow_raw_pred_home_runs",
            "shadow_baseline_raw_pred_home_runs",
            "shadow_hr_direct_prediction",
            "shadow_hr_blend_alpha",
        ),
    )
    for row in batters:
        sigma_map = row.get("sigma_map") or {}
        base_context = row.get("opportunity_context") or {}
        for stat, projection_key, baseline_key, direct_key, alpha_key in specs:
            projection = _clean_float(row.get(projection_key))
            if projection is None:
                continue
            baseline = _clean_float(row.get(baseline_key))
            direct_projection = _clean_float(row.get(direct_key))
            alpha = _clean_float(row.get(alpha_key))
            context = dict(base_context)
            context.update({
                "shadow_forecast": True,
                "shadow_model_family": family,
                "shadow_model_version": release_id,
                "shadow_stat": stat,
                "shadow_blend_alpha": alpha,
                "shadow_direct_prediction": direct_projection,
                "shadow_baseline_projection": baseline,
                "shadow_production_impact": "none",
            })
            rec_source = dict(row)
            rec_source["opportunity_context"] = context
            rec = _base_row(
                rec_source,
                run_id=run_id,
                phase=phase,
                locked_at=locked_at,
                forecast_type="player",
                stat=stat,
                projection=projection,
            )
            rec.update({
                "forecast_key": _explicit_forecast_key(
                    "shadow_hitter_rate",
                    run_id,
                    phase,
                    rec.get("game_date_et"),
                    rec.get("game_slug"),
                    rec.get("player_id"),
                    stat,
                    family,
                    release_id,
                ),
                "model_only_projection": direct_projection,
                "baseline_value": baseline,
                "baseline_source": "production_without_rate_challenger",
                "distribution": {
                    "family": "count_mean_sigma",
                    "mean": projection,
                    "sigma": _clean_float(sigma_map.get(stat)),
                    "low_pa_probability": _clean_float(context.get("low_pa_probability")),
                    "shadow": True,
                },
                "model_family": family,
                "model_version": release_id,
            })
            forecasts.append(rec)
    return insert_forecasts(conn, forecasts)


def grade_daily_forecasts(conn, *, date_to: date | None = None) -> int:
    ensure_daily_forecast_ledger_schema(conn)
    params = {"date_to": date_to or date.today()}
    updated = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH actual AS (
                SELECT
                    g.game_slug,
                    (g.home_score - g.away_score)::numeric AS actual_run_diff,
                    (g.home_score + g.away_score)::numeric AS actual_total,
                    CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END::numeric AS actual_home_win
                FROM raw.mlb_games g
                WHERE g.status = 'final'
                  AND g.game_date_et <= %(date_to)s
                  AND g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
            )
            UPDATE bets.mlb_daily_forecast_ledger f
            SET actual_value = CASE f.stat
                    WHEN 'game_run_diff' THEN a.actual_run_diff
                    WHEN 'game_total' THEN a.actual_total
                    WHEN 'game_home_win' THEN a.actual_home_win
                END,
                absolute_error = ABS(f.projection_value - CASE f.stat
                    WHEN 'game_run_diff' THEN a.actual_run_diff
                    WHEN 'game_total' THEN a.actual_total
                    WHEN 'game_home_win' THEN a.actual_home_win
                END),
                squared_error = POWER(f.projection_value - CASE f.stat
                    WHEN 'game_run_diff' THEN a.actual_run_diff
                    WHEN 'game_total' THEN a.actual_total
                    WHEN 'game_home_win' THEN a.actual_home_win
                END, 2),
                baseline_absolute_error = CASE WHEN f.baseline_value IS NOT NULL THEN
                    ABS(f.baseline_value - CASE f.stat
                        WHEN 'game_run_diff' THEN a.actual_run_diff
                        WHEN 'game_total' THEN a.actual_total
                        WHEN 'game_home_win' THEN a.actual_home_win
                    END) END,
                result_status = 'graded', graded_at_utc = NOW(), grade_source = 'raw.mlb_games_final'
            FROM actual a
            WHERE f.game_slug = a.game_slug
              AND f.result_status = 'pending'
              AND f.forecast_type = 'game'
            """,
            params,
        )
        updated += cur.rowcount

        cur.execute(
            """
            WITH terminal_non_final AS (
                SELECT game_slug, status
                FROM raw.mlb_games
                WHERE game_date_et <= %(date_to)s
                  AND LOWER(COALESCE(status, '')) = ANY(%(terminal_non_final_statuses)s)
            )
            UPDATE bets.mlb_daily_forecast_ledger f
            SET result_status = 'void',
                actual_value = NULL,
                absolute_error = NULL,
                squared_error = NULL,
                baseline_absolute_error = NULL,
                graded_at_utc = NOW(),
                grade_source = 'raw.mlb_games_terminal_non_final:' || COALESCE(t.status, 'unknown')
            FROM terminal_non_final t
            WHERE f.game_slug = t.game_slug
              AND f.result_status = 'pending'
              AND f.game_date_et <= %(date_to)s
            """,
            {**params, "terminal_non_final_statuses": list(TERMINAL_NON_FINAL_GAME_STATUSES)},
        )
        updated += cur.rowcount

        cur.execute("SELECT to_regclass('features.mlb_prop_market_training_examples')")
        has_opportunity = cur.fetchone()[0] is not None
        opportunity_cte = """
            opportunity AS (
                SELECT game_slug, player_id,
                       MAX(actual_pa)::numeric AS actual_pa,
                       MAX(actual_bf)::numeric AS actual_bf,
                       MAX(actual_ip)::numeric AS actual_ip,
                       MAX(actual_pitch_count_proxy)::numeric AS actual_pitch_count
                FROM features.mlb_prop_market_training_examples
                WHERE result_status = 'graded'
                GROUP BY game_slug, player_id
            ),
        """ if has_opportunity else ""
        opportunity_join = "LEFT JOIN opportunity o ON o.game_slug = gl.game_slug AND o.player_id = gl.player_id" if has_opportunity else ""
        opportunity_values = {
            "hitter_plate_appearances": "COALESCE(o.actual_pa, COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0))" if has_opportunity else "COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0)",
            "hitter_plate_appearances_challenger": "COALESCE(o.actual_pa, COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0))" if has_opportunity else "COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0)",
            "pitcher_batters_faced": "o.actual_bf" if has_opportunity else "NULL",
            "pitcher_innings": "COALESCE(o.actual_ip, gl.innings_pitched)" if has_opportunity else "gl.innings_pitched",
            "pitcher_pitch_count": "o.actual_pitch_count" if has_opportunity else "NULL",
        }
        actual_case = "\n".join([
            "CASE f.stat",
            " WHEN 'pitcher_strikeouts' THEN gl.strikeouts_pitcher::numeric",
            " WHEN 'batter_hits' THEN COALESCE(gl.hits, 0)::numeric",
            " WHEN 'batter_total_bases' THEN COALESCE(gl.total_bases, 0)::numeric",
            " WHEN 'batter_home_runs' THEN COALESCE(gl.home_runs, 0)::numeric",
            *[f" WHEN '{stat}' THEN {expr}" for stat, expr in opportunity_values.items()],
            " END",
        ])
        cur.execute(
            f"""
            WITH {opportunity_cte}
            eligible AS (
                SELECT f.id, f.projection_value, f.baseline_value, {actual_case} AS actual_value
                FROM bets.mlb_daily_forecast_ledger f
                JOIN raw.mlb_games g ON g.game_slug = f.game_slug AND g.status = 'final'
                JOIN raw.mlb_player_gamelogs gl
                  ON gl.game_slug = f.game_slug AND gl.player_id = f.player_id
                {opportunity_join}
                WHERE f.result_status = 'pending'
                  AND f.forecast_type IN ('player', 'opportunity')
                  AND f.game_date_et <= %(date_to)s
            )
            UPDATE bets.mlb_daily_forecast_ledger f
            SET actual_value = e.actual_value,
                absolute_error = ABS(f.projection_value - e.actual_value),
                squared_error = POWER(f.projection_value - e.actual_value, 2),
                baseline_absolute_error = CASE WHEN f.baseline_value IS NOT NULL
                    THEN ABS(f.baseline_value - e.actual_value) END,
                result_status = 'graded', graded_at_utc = NOW(),
                grade_source = 'final_verified_player_participation'
            FROM eligible e
            WHERE f.id = e.id
              AND e.actual_value IS NOT NULL
            """,
            params,
        )
        updated += cur.rowcount

        cur.execute(
            """
            WITH final_games_with_results AS (
                SELECT DISTINCT g.game_slug
                FROM raw.mlb_games g
                JOIN raw.mlb_player_gamelogs gl ON gl.game_slug = g.game_slug
                WHERE g.status = 'final'
                  AND g.game_date_et <= %(date_to)s
            )
            UPDATE bets.mlb_daily_forecast_ledger f
            SET result_status = 'void',
                actual_value = NULL,
                absolute_error = NULL,
                squared_error = NULL,
                baseline_absolute_error = NULL,
                graded_at_utc = NOW(),
                grade_source = 'final_verified_nonparticipant'
            FROM final_games_with_results g
            WHERE f.game_slug = g.game_slug
              AND f.result_status = 'pending'
              AND f.forecast_type IN ('player', 'opportunity')
              AND f.player_id IS NOT NULL
              AND f.game_date_et <= %(date_to)s
              AND NOT EXISTS (
                  SELECT 1
                  FROM raw.mlb_player_gamelogs gl
                  WHERE gl.game_slug = f.game_slug
                    AND gl.player_id = f.player_id
                    AND CASE
                        WHEN f.stat LIKE 'pitcher_%%' THEN
                            COALESCE(
                                gl.innings_pitched,
                                gl.strikeouts_pitcher,
                                gl.hits_allowed,
                                gl.walks_allowed,
                                gl.runs_allowed,
                                gl.earned_runs,
                                gl.home_runs_allowed
                            ) IS NOT NULL
                        WHEN f.stat LIKE 'batter_%%' OR f.stat LIKE 'hitter_%%' THEN
                            COALESCE(
                                gl.at_bats,
                                gl.hits,
                                gl.walks_batter,
                                gl.total_bases,
                                gl.home_runs,
                                gl.doubles,
                                gl.triples
                            ) IS NOT NULL
                        ELSE TRUE
                    END
              )
            """,
            params,
        )
        updated += cur.rowcount
    conn.commit()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage immutable MLB daily forecast ledger")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--ensure-schema", action="store_true")
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--date-to", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    with psycopg2.connect(args.pg_dsn) as conn:
        if args.ensure_schema:
            ensure_daily_forecast_ledger_schema(conn)
        updated = grade_daily_forecasts(conn, date_to=args.date_to) if args.grade else 0
    print(json.dumps({"schema_ready": True, "graded": updated}, indent=2))


if __name__ == "__main__":
    main()
