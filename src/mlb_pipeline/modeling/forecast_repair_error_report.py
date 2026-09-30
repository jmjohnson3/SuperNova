"""Diagnose forecast misses at the player-game projection layer.

The report uses immutable prospective forecasts when they are graded and a
one-row-per-player-game historical fallback while prospective evidence builds.
Offer rows are deliberately excluded from projection-error calculations.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN as _PG_DSN
from .model_release import canonical_forecast_phase

_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_HITTER_STATS = ("batter_hits", "batter_total_bases", "batter_home_runs")


@dataclass(frozen=True)
class ForecastRepairConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 180
    min_group_rows: int = 30
    report_file: str = "mlb_forecast_repair_error_latest.md"
    artifact_file: str = "forecast_repair_error_report.json"


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [column[0] for column in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _table_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
        return bool(cur.fetchone()[0])


def decompose_count_error(frame: pd.DataFrame) -> pd.DataFrame:
    """Split count error into opportunity and per-opportunity rate pieces."""
    out = frame.copy()
    for column in ("projection", "actual", "opportunity_projection", "opportunity_actual"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    valid = (
        out["projection"].notna()
        & out["actual"].notna()
        & out["opportunity_projection"].gt(0)
        & out["opportunity_actual"].gt(0)
    )
    out["projected_rate"] = np.nan
    out["actual_rate"] = np.nan
    out["opportunity_error"] = np.nan
    out["rate_error"] = np.nan
    out.loc[valid, "projected_rate"] = (
        out.loc[valid, "projection"] / out.loc[valid, "opportunity_projection"]
    )
    out.loc[valid, "actual_rate"] = out.loc[valid, "actual"] / out.loc[valid, "opportunity_actual"]
    out.loc[valid, "opportunity_error"] = (
        out.loc[valid, "projected_rate"]
        * (out.loc[valid, "opportunity_projection"] - out.loc[valid, "opportunity_actual"])
    )
    out.loc[valid, "rate_error"] = (
        out.loc[valid, "opportunity_actual"]
        * (out.loc[valid, "projected_rate"] - out.loc[valid, "actual_rate"])
    )
    out["total_error"] = out["projection"] - out["actual"]
    out["decomposition_residual"] = out["total_error"] - out["opportunity_error"] - out["rate_error"]
    return out


def _load_prospective(conn, cutoff: date, phase: str) -> pd.DataFrame:
    if not _table_exists(conn, "bets.mlb_daily_forecast_ledger"):
        return pd.DataFrame()
    hitter_actuals_cte = """
        , hitter_actuals AS (
            SELECT DISTINCT ON (game_slug, player_id)
                   game_slug, player_id,
                   GREATEST(hits - COALESCE(doubles, 0) - COALESCE(triples, 0)
                            - COALESCE(home_runs, 0), 0)::float AS actual_singles,
                   COALESCE(doubles, 0)::float AS actual_doubles,
                   COALESCE(triples, 0)::float AS actual_triples,
                   COALESCE(home_runs, 0)::float AS actual_home_runs
            FROM raw.mlb_player_gamelogs
            WHERE at_bats IS NOT NULL
            ORDER BY game_slug, player_id, fetched_at_utc DESC
        )
    """
    hitter_columns = """
               h.actual_singles::float, h.actual_doubles::float,
               h.actual_triples::float, h.actual_home_runs::float,
               NULLIF(p.opportunity_context->>'confirmed_batting_order', '')::float AS lineup_slot,
               CASE WHEN NULLIF(p.opportunity_context->>'confirmed_batting_order', '') IS NOT NULL
                    THEN 1.0 ELSE 0.0 END::float AS lineup_confirmed_flag,
               CASE WHEN o.actual <= 2 THEN 1.0 ELSE 0.0 END::float AS low_pa_flag
    """
    hitter_join = """
        LEFT JOIN hitter_actuals h
         ON h.game_slug = p.game_slug
         AND h.player_id = p.player_id
    """
    return _query_df(
        conn,
        f"""
        WITH canonical AS (
            SELECT DISTINCT ON (game_date_et, game_slug, COALESCE(player_id, 0), stat, model_version)
                game_date_et, game_slug, player_id, player_name, team_abbr,
                stat, forecast_phase, COALESCE(model_version, 'unknown') AS model_version,
                projection_value::float AS projection,
                baseline_value::float AS baseline, actual_value::float AS actual,
                opportunity_context
            FROM bets.mlb_daily_forecast_ledger
            WHERE result_status = 'graded'
              AND actual_value IS NOT NULL
              AND game_date_et >= %(cutoff)s
              AND forecast_phase = %(phase)s
            ORDER BY game_date_et, game_slug, COALESCE(player_id, 0), stat, model_version,
                     locked_at_utc, id
        )
        {hitter_actuals_cte}
        SELECT p.game_date_et, p.game_slug, p.player_id, p.player_name,
               p.team_abbr, p.stat, p.forecast_phase, p.model_version,
               p.projection, p.baseline,
               p.actual, o.projection AS opportunity_projection,
               o.actual AS opportunity_actual, 'prospective_ledger'::text AS source,
               {hitter_columns}
        FROM canonical p
        LEFT JOIN canonical o
          ON o.game_slug = p.game_slug
         AND o.player_id = p.player_id
         AND o.forecast_phase = p.forecast_phase
         AND o.model_version = p.model_version
         AND o.stat = CASE
             WHEN p.stat = 'pitcher_strikeouts' THEN 'pitcher_batters_faced'
             WHEN p.stat IN ('batter_hits','batter_total_bases','batter_home_runs')
                 THEN 'hitter_plate_appearances'
             ELSE NULL
         END
        {hitter_join}
        WHERE p.stat IN (
            'pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs'
        )
        """,
        {"cutoff": cutoff, "phase": phase},
    )


def _load_historical_hitter_games(conn, cutoff: date) -> pd.DataFrame:
    if not _table_exists(conn, "features.mlb_hitter_player_game_training"):
        return pd.DataFrame()
    wide = _query_df(
        conn,
        """
        SELECT game_date_et, game_slug, player_id, player_name, team_abbr,
               projected_pa::float AS opportunity_projection,
               actual_pa::float AS opportunity_actual,
               'historical_player_game'::text AS model_version,
               model_pred_hits::float, model_pred_total_bases::float,
               model_pred_home_runs::float, actual_hits::float,
               actual_total_bases::float, actual_home_runs::float,
               actual_singles::float, actual_doubles::float,
               actual_triples::float, lineup_slot::float,
               lineup_confirmed_flag::float, low_pa_flag::float
        FROM features.mlb_hitter_player_game_training
        WHERE game_date_et >= %(cutoff)s
          AND actual_pa IS NOT NULL
        ORDER BY game_date_et, game_slug, player_id
        """,
        {"cutoff": cutoff},
    )
    if wide.empty:
        return wide
    rows: list[pd.DataFrame] = []
    for stat, prediction, actual in (
        ("batter_hits", "model_pred_hits", "actual_hits"),
        ("batter_total_bases", "model_pred_total_bases", "actual_total_bases"),
        ("batter_home_runs", "model_pred_home_runs", "actual_home_runs"),
    ):
        part = wide.copy()
        part["stat"] = stat
        part["projection"] = part[prediction]
        part["actual"] = part[actual]
        part["baseline"] = np.nan
        part["forecast_phase"] = "historical_player_game"
        part["source"] = "historical_player_game"
        rows.append(part[[
            "game_date_et", "game_slug", "player_id", "player_name", "team_abbr",
            "stat", "forecast_phase", "model_version", "projection", "baseline", "actual",
            "opportunity_projection", "opportunity_actual", "source",
            "actual_singles", "actual_doubles", "actual_triples", "actual_home_runs",
            "lineup_slot", "lineup_confirmed_flag", "low_pa_flag",
        ]])
    return pd.concat(rows, ignore_index=True).dropna(subset=["projection", "actual"])


def _add_error_slice_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model_version" not in out:
        out["model_version"] = "unknown"
    out["model_version"] = out["model_version"].fillna("unknown").astype(str)
    lineup = pd.to_numeric(out.get("lineup_confirmed_flag"), errors="coerce")
    out["lineup_status"] = np.where(
        out["stat"].astype(str).isin(_HITTER_STATS),
        np.where(lineup.ge(0.5), "confirmed_lineup", "lineup_missing_or_unconfirmed"),
        "not_applicable",
    )
    opp_projection = pd.to_numeric(out.get("opportunity_projection"), errors="coerce")
    opp_actual = pd.to_numeric(out.get("opportunity_actual"), errors="coerce")
    opp_delta = opp_projection - opp_actual
    out["opportunity_delta"] = opp_delta
    out["opportunity_error_bucket"] = np.select(
        [
            opp_delta.isna(),
            opp_delta <= -1.0,
            (opp_delta > -1.0) & (opp_delta <= -0.5),
            opp_delta.abs() < 0.5,
            (opp_delta >= 0.5) & (opp_delta < 1.0),
            opp_delta >= 1.0,
        ],
        [
            "missing_opportunity",
            "under_projected_by_1_plus",
            "under_projected_by_0.5_to_1",
            "near_actual_within_0.5",
            "over_projected_by_0.5_to_1",
            "over_projected_by_1_plus",
        ],
        default="near_actual_within_0.5",
    )
    out["stat_model_version"] = out["stat"].astype(str) + " / " + out["model_version"]
    return out


def _metrics(group: pd.DataFrame) -> dict[str, Any]:
    valid = group.dropna(subset=["projection", "actual"])
    if valid.empty:
        return {"rows": 0}
    error = valid["projection"] - valid["actual"]
    decomposition = valid.dropna(subset=["opportunity_error", "rate_error"])
    opportunity_abs = decomposition["opportunity_error"].abs().mean() if len(decomposition) else np.nan
    rate_abs = decomposition["rate_error"].abs().mean() if len(decomposition) else np.nan
    dominant = None
    if np.isfinite(opportunity_abs) and np.isfinite(rate_abs):
        dominant = "opportunity" if opportunity_abs > rate_abs else "player_rate"
    baseline = valid.dropna(subset=["baseline"])
    return {
        "rows": int(len(valid)),
        "dates": int(valid["game_date_et"].nunique()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
        "baseline_mae": (
            float((baseline["baseline"] - baseline["actual"]).abs().mean())
            if len(baseline) else None
        ),
        "decomposed_rows": int(len(decomposition)),
        "opportunity_mae_component": float(opportunity_abs) if np.isfinite(opportunity_abs) else None,
        "rate_mae_component": float(rate_abs) if np.isfinite(rate_abs) else None,
        "opportunity_bias_component": (
            float(decomposition["opportunity_error"].mean()) if len(decomposition) else None
        ),
        "rate_bias_component": float(decomposition["rate_error"].mean()) if len(decomposition) else None,
        "dominant_error_driver": dominant,
    }


def _group_summaries(frame: pd.DataFrame, column: str, min_rows: int) -> list[dict[str, Any]]:
    if frame.empty or column not in frame:
        return []
    rows = []
    for key, group in frame.groupby(column, dropna=False):
        metrics = _metrics(group)
        if metrics["rows"] < min_rows:
            continue
        rows.append({column: "missing" if pd.isna(key) else str(key), **metrics})
    return sorted(rows, key=lambda row: float(row.get("mae") or 0.0), reverse=True)


def _load_offer_error_slices(conn, cutoff: date) -> pd.DataFrame:
    if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
        return pd.DataFrame()
    return _query_df(
        conn,
        """
        SELECT game_date_et, market, side,
               COALESCE(line_bucket, 'unknown') AS line_bucket,
               COALESCE(bookmaker_key, 'unknown') AS bookmaker_key,
               pred_count::float AS pred_count,
               actual_value::float AS actual_value,
               CASE WHEN won IS TRUE THEN 1 WHEN won IS FALSE THEN 0 ELSE NULL END AS won,
               COALESCE(push, false) AS push,
               profit_units::float AS profit_units,
               CASE WHEN clv_valid IS TRUE THEN 1 WHEN clv_valid IS FALSE THEN 0 ELSE NULL END AS clv_valid,
               CASE WHEN beat_clv_price IS TRUE THEN 1 WHEN beat_clv_price IS FALSE THEN 0 ELSE NULL END AS beat_clv_price
        FROM features.mlb_prop_market_training_examples
        WHERE game_date_et >= %(cutoff)s
          AND market IN ('pitcher_strikeouts','batter_hits','batter_total_bases','batter_home_runs')
          AND pred_count IS NOT NULL
          AND actual_value IS NOT NULL
        """,
        {"cutoff": cutoff},
    )


def _offer_slice_metrics(group: pd.DataFrame) -> dict[str, Any]:
    valid = group.dropna(subset=["pred_count", "actual_value"])
    if valid.empty:
        return {"rows": 0}
    error = pd.to_numeric(valid["pred_count"], errors="coerce") - pd.to_numeric(valid["actual_value"], errors="coerce")
    graded = valid.loc[~valid["push"].fillna(False).astype(bool)]
    won = pd.to_numeric(graded.get("won"), errors="coerce").dropna()
    profit = pd.to_numeric(graded.get("profit_units"), errors="coerce").dropna()
    clv = pd.to_numeric(valid.get("beat_clv_price"), errors="coerce").dropna()
    clv_valid = pd.to_numeric(valid.get("clv_valid"), errors="coerce").dropna()
    return {
        "rows": int(len(valid)),
        "dates": int(pd.to_datetime(valid["game_date_et"]).dt.date.nunique()),
        "count_mae": float(error.abs().mean()),
        "count_bias": float(error.mean()),
        "win_rate": float(won.mean()) if not won.empty else None,
        "roi": float(profit.mean()) if not profit.empty else None,
        "valid_clv_rate": float(clv_valid.mean()) if not clv_valid.empty else None,
        "clv_beat_rate": float(clv.mean()) if not clv.empty else None,
    }


def _offer_group_summaries(frame: pd.DataFrame, cols: list[str], min_rows: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(cols, dropna=False):
        metrics = _offer_slice_metrics(group)
        if metrics["rows"] < min_rows:
            continue
        key_values = key if isinstance(key, tuple) else (key,)
        row = {
            col: "missing" if pd.isna(value) else str(value)
            for col, value in zip(cols, key_values)
        }
        row.update(metrics)
        rows.append(row)
    return sorted(rows, key=lambda row: float(row.get("count_mae") or 0.0), reverse=True)


def _tb_diagnostic(frame: pd.DataFrame, min_rows: int) -> dict[str, Any]:
    tb = frame[frame["stat"].eq("batter_total_bases")].copy()
    if tb.empty:
        return {"rows": 0, "status": "no_rows", "recommended_repair": "collect_graded_tb_forecasts"}
    actual_hr = pd.to_numeric(tb["actual_home_runs"], errors="coerce")
    tb["tb_state"] = np.select(
        [
            tb["actual"].eq(0), tb["actual"].eq(1), tb["actual"].between(2, 3),
            tb["actual"].ge(4) & actual_hr.gt(0),
            tb["actual"].ge(4) & actual_hr.eq(0),
            tb["actual"].ge(4),
        ],
        ["0", "1", "2-3", "4+ HR", "4+ non-HR", "4+ unknown-HR"],
        default="unknown",
    )
    actual_pa = pd.to_numeric(tb["opportunity_actual"], errors="coerce").sum()
    component_rates = {}
    if actual_pa > 0:
        for name, column in (
            ("single", "actual_singles"), ("double", "actual_doubles"),
            ("triple", "actual_triples"), ("hr", "actual_home_runs"),
        ):
            values = pd.to_numeric(tb[column], errors="coerce")
            if values.notna().any():
                component_rates[name] = float(values.sum(min_count=1) / actual_pa)
    summary = _metrics(tb)
    state_rows = _group_summaries(tb, "tb_state", min_rows)
    driver = summary.get("dominant_error_driver")
    recommended = (
        "repair_pa_opportunity_before_tb_rates" if driver == "opportunity"
        else "train_gated_direct_player_game_tb_head" if driver == "player_rate"
        else "collect_more_decomposable_rows"
    )
    return {
        **summary,
        "status": "ready",
        "component_actual_per_pa": component_rates,
        "by_actual_tb_state": state_rows,
        "recommended_repair": recommended,
    }


def _load_count_model_repairs() -> dict[str, dict[str, Any]]:
    path = _MODEL_DIR / "hitter_player_game_outcome_models.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        return {
            stat: {"status": "ready", **(metrics.get(f"direct_{prefix}_count_repair") or {})}
            for stat, prefix in (
                ("batter_hits", "hits"),
                ("batter_total_bases", "tb"),
                ("batter_home_runs", "hr"),
            )
        }
    except (OSError, ValueError):
        return {}


def _write_report(payload: dict[str, Any], cfg: ForecastRepairConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file

    def num(value: Any, digits: int = 3) -> str:
        try:
            return "-" if value is None else f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "-"

    lines = [
        "# MLB Forecast Repair / Error Decomposition",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Canonical evidence phase: {payload.get('canonical_forecast_phase')}",
        f"Prospective graded rows: {payload.get('prospective_rows', 0)}",
        f"Historical player-game fallback rows: {payload.get('historical_rows', 0)}",
        f"Active diagnostic source: {payload.get('active_source')}",
        "",
        "Count error is split exactly into opportunity error and per-opportunity player-rate error.",
        "Offer duplicates are never used in this projection audit.",
        "",
        "## Market Repair Queue",
        "",
        "| Stat | Rows | Dates | MAE | Bias | Opportunity | Player Rate | Dominant Repair |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stat, record in (payload.get("stats") or {}).items():
        lines.append(
            f"| {stat} | {record.get('rows', 0)} | {record.get('dates', 0)} | "
            f"{num(record.get('mae'))} | {num(record.get('bias'))} | "
            f"{num(record.get('opportunity_mae_component'))} | {num(record.get('rate_mae_component'))} | "
            f"{record.get('dominant_error_driver') or '-'} |"
        )
    tb = payload.get("tb") or {}
    repairs = payload.get("count_model_repairs") or {}
    repair = repairs.get("batter_total_bases") or {}
    lines.extend([
        "",
        "## Total Bases Repair",
        "",
        f"- Recommended repair: {tb.get('recommended_repair')}",
        f"- TB MAE / bias: {num(tb.get('mae'))} / {num(tb.get('bias'))}",
        f"- Opportunity / player-rate error: {num(tb.get('opportunity_mae_component'))} / {num(tb.get('rate_mae_component'))}",
        f"- Actual event rates per PA: {json.dumps(tb.get('component_actual_per_pa') or {}, sort_keys=True)}",
        f"- Direct TB repair enabled: {bool(repair.get('enabled'))}",
        f"- Walk-forward folds / positive: {repair.get('walk_forward_folds', 0)} / {repair.get('positive_folds', 0)}",
        f"- OOF MAE gain: {num(repair.get('validation_mae_gain'))}",
        f"- Production blend / bias offset: {num(repair.get('alpha'))} / {num(repair.get('production_bias_offset'))}",
        "",
        "| Actual TB State | Rows | MAE | Bias | Dominant Repair |",
        "|---|---:|---:|---:|---|",
    ])
    for record in tb.get("by_actual_tb_state") or []:
        lines.append(
            f"| {record.get('tb_state')} | {record.get('rows', 0)} | {num(record.get('mae'))} | "
            f"{num(record.get('bias'))} | {record.get('dominant_error_driver') or '-'} |"
        )
    lines.extend([
        "",
        "## Frozen Release / Opportunity Slices",
        "",
        "| Slice | Rows | Dates | MAE | Bias | Opportunity | Player Rate | Dominant Repair |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for record in (payload.get("by_stat_model_version") or [])[:30]:
        lines.append(
            f"| {record.get('stat_model_version')} | {record.get('rows', 0)} | {record.get('dates', 0)} | "
            f"{num(record.get('mae'))} | {num(record.get('bias'))} | "
            f"{num(record.get('opportunity_mae_component'))} | {num(record.get('rate_mae_component'))} | "
            f"{record.get('dominant_error_driver') or '-'} |"
        )
    lines.extend([
        "",
        "| Lineup Status | Rows | Dates | MAE | Bias | Dominant Repair |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for record in (payload.get("by_lineup_status") or [])[:20]:
        lines.append(
            f"| {record.get('lineup_status')} | {record.get('rows', 0)} | {record.get('dates', 0)} | "
            f"{num(record.get('mae'))} | {num(record.get('bias'))} | {record.get('dominant_error_driver') or '-'} |"
        )
    lines.extend([
        "",
        "| Opportunity Error Bucket | Rows | Dates | MAE | Bias | Dominant Repair |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for record in (payload.get("by_opportunity_error_bucket") or [])[:20]:
        lines.append(
            f"| {record.get('opportunity_error_bucket')} | {record.get('rows', 0)} | {record.get('dates', 0)} | "
            f"{num(record.get('mae'))} | {num(record.get('bias'))} | {record.get('dominant_error_driver') or '-'} |"
        )
    offer = payload.get("offer_error_slices") or {}
    lines.extend([
        "",
        "## Offer-Level Line / Market Slices",
        "",
        "These slices are diagnostic only. They are duplicated offer rows and are not used for projection-model selection.",
        "",
        "| Market | Side | Rows | Dates | Count MAE | Bias | Win | ROI | CLV Beat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in (offer.get("market_side") or [])[:20]:
        lines.append(
            f"| {record.get('market')} | {record.get('side')} | {record.get('rows', 0)} | {record.get('dates', 0)} | "
            f"{num(record.get('count_mae'))} | {num(record.get('count_bias'))} | "
            f"{num(record.get('win_rate'))} | {num(record.get('roi'))} | {num(record.get('clv_beat_rate'))} |"
        )
    lines.extend([
        "",
        "| Market | Side | Line Bucket | Book | Rows | Count MAE | Bias | ROI | CLV Beat |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for record in (offer.get("market_side_line_book") or [])[:30]:
        lines.append(
            f"| {record.get('market')} | {record.get('side')} | {record.get('line_bucket')} | {record.get('bookmaker_key')} | "
            f"{record.get('rows', 0)} | {num(record.get('count_mae'))} | {num(record.get('count_bias'))} | "
            f"{num(record.get('roi'))} | {num(record.get('clv_beat_rate'))} |"
        )
    lines.extend([
        "",
        "## Direct Player-Game Repair Heads",
        "",
        "| Stat | Enabled | Positive Folds | Base MAE | Blended MAE | MAE Gain | Any-Event Brier Gain | Alpha |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for stat, record in repairs.items():
        base = record.get("base_validation") or {}
        blended = record.get("blended_validation") or {}
        lines.append(
            f"| {stat} | {bool(record.get('enabled'))} | {record.get('positive_folds', 0)}/{record.get('walk_forward_folds', 0)} | "
            f"{num(base.get('mae'))} | {num(blended.get('mae'))} | {num(record.get('validation_mae_gain'))} | "
            f"{num(record.get('any_brier_gain'), 5)} | {num(record.get('alpha'))} |"
        )
    atomic_write_text(path, "\n".join(lines) + "\n")
    return str(path)


def build_report(cfg: ForecastRepairConfig) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    canonical_phase = canonical_forecast_phase()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        prospective = _load_prospective(conn, cutoff, canonical_phase)
        historical = _load_historical_hitter_games(conn, cutoff)
        offer_slices = _load_offer_error_slices(conn, cutoff)
    prospective = decompose_count_error(prospective) if not prospective.empty else prospective
    historical = decompose_count_error(historical) if not historical.empty else historical
    active = prospective if len(prospective) >= 100 else historical
    active_source = "prospective_ledger" if active is prospective else "historical_player_game_fallback"
    active = _add_error_slice_columns(active) if not active.empty else active
    stats = {
        str(stat): _metrics(group)
        for stat, group in active.groupby("stat")
    } if not active.empty else {}
    offer_error_slices = {
        "market_side": _offer_group_summaries(offer_slices, ["market", "side"], cfg.min_group_rows),
        "market_side_line_book": _offer_group_summaries(
            offer_slices,
            ["market", "side", "line_bucket", "bookmaker_key"],
            cfg.min_group_rows,
        ),
    }
    count_model_repairs = _load_count_model_repairs()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if not active.empty else "no_rows",
        "canonical_forecast_phase": canonical_phase,
        "active_source": active_source,
        "prospective_rows": int(len(prospective)),
        "historical_rows": int(len(historical)),
        "stats": stats,
        "by_stat_model_version": _group_summaries(active, "stat_model_version", cfg.min_group_rows) if not active.empty else [],
        "by_lineup_status": _group_summaries(active, "lineup_status", cfg.min_group_rows) if not active.empty else [],
        "by_opportunity_error_bucket": _group_summaries(active, "opportunity_error_bucket", cfg.min_group_rows) if not active.empty else [],
        "offer_error_slices": offer_error_slices,
        "tb": _tb_diagnostic(active, cfg.min_group_rows),
        "count_model_repairs": count_model_repairs,
        "tb_model_repair": count_model_repairs.get("batter_total_bases") or {},
    }
    payload["report_path"] = _write_report(payload, cfg)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / cfg.artifact_file, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forecast repair and error-decomposition report")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--min-group-rows", type=int, default=30)
    args = parser.parse_args()
    payload = build_report(ForecastRepairConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        min_group_rows=args.min_group_rows,
    ))
    print(json.dumps({
        "status": payload.get("status"),
        "active_source": payload.get("active_source"),
        "prospective_rows": payload.get("prospective_rows"),
        "historical_rows": payload.get("historical_rows"),
        "tb_repair": (payload.get("tb") or {}).get("recommended_repair"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
