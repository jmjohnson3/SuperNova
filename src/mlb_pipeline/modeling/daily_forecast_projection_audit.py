"""Prospective audit for immutable game and player forecasts."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

from .daily_forecast_ledger import ensure_daily_forecast_ledger_schema, grade_daily_forecasts
from .model_release import canonical_forecast_phase
from .prop_training_groups import dedupe_locked_offer_rows, expanding_player_game_folds
from mlb_pipeline.db import PG_DSN as _PG_DSN

_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_PROJECTION_STATS = {
    "game_run_diff", "game_home_win", "game_total",
    "pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs",
    "hitter_plate_appearances_challenger",
}


@dataclass(frozen=True)
class AuditConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 180
    report_file: str = "mlb_daily_forecast_projection_audit_latest.md"
    artifact_file: str = "daily_forecast_projection_audit.json"
    min_dates: int = 5
    min_player_rows: int = 100
    min_game_rows: int = 30
    min_opportunity_rows: int = 50


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{schema}.{table}",))
        return bool(cur.fetchone()[0])


def _is_shadow_model(family: Any, version: Any) -> bool:
    text = f"{family or ''} {version or ''}".lower()
    return "shadow" in text


def _non_shadow_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"model_family", "model_version"}.issubset(df.columns):
        return df
    mask = [
        _is_shadow_model(family, version)
        for family, version in zip(df["model_family"], df["model_version"])
    ]
    return df.loc[~pd.Series(mask, index=df.index)].copy()


def _load_forecasts(conn, cutoff: date, phase: str) -> pd.DataFrame:
    return _query_df(
        conn,
        """
        SELECT DISTINCT ON (
            game_date_et, game_slug, COALESCE(player_id, 0), stat,
            COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown')
        )
            id, forecast_run_id, forecast_phase, locked_at_utc, game_date_et,
            game_slug, forecast_type, player_id, player_name, team_abbr,
            opponent_abbr, stat, projection_value::float AS projection_value,
            model_only_projection::float AS model_only_projection,
            baseline_value::float AS baseline_value, baseline_source,
            probability_value::float AS probability_value,
            actual_value::float AS actual_value, model_family, model_version
        FROM bets.mlb_daily_forecast_ledger
        WHERE result_status = 'graded'
          AND actual_value IS NOT NULL
          AND game_date_et >= %(cutoff)s
          AND forecast_phase = %(phase)s
        ORDER BY game_date_et, game_slug, COALESCE(player_id, 0), stat,
                 COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown'),
                 locked_at_utc, id
        """,
        {"cutoff": cutoff, "phase": phase},
    )


def _ledger_status_counts(conn, cutoff: date, phase: str) -> dict[str, Any]:
    df = _query_df(
        conn,
        """
        WITH canonical AS (
            SELECT DISTINCT ON (
                game_date_et, game_slug, COALESCE(player_id, 0), stat,
                COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown')
            )
                   result_status, forecast_type
            FROM bets.mlb_daily_forecast_ledger
            WHERE game_date_et >= %(cutoff)s
              AND forecast_phase = %(phase)s
            ORDER BY game_date_et, game_slug, COALESCE(player_id, 0), stat,
                     COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown'),
                     locked_at_utc, id
        )
        SELECT result_status, forecast_type, COUNT(*)::int AS rows
        FROM canonical
        GROUP BY result_status, forecast_type
        ORDER BY result_status, forecast_type
        """,
        {"cutoff": cutoff, "phase": phase},
    )
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in df.to_dict(orient="records"):
        status = str(row["result_status"])
        forecast_type = str(row["forecast_type"])
        count = int(row["rows"])
        by_status[status] = by_status.get(status, 0) + count
        by_type[forecast_type] = by_type.get(forecast_type, 0) + count
    return {
        "total": int(sum(by_status.values())),
        "by_status": by_status,
        "by_type": by_type,
    }


def _ledger_model_cohorts(conn, cutoff: date, phase: str) -> list[dict[str, Any]]:
    df = _query_df(
        conn,
        """
        WITH canonical AS (
            SELECT DISTINCT ON (
                game_date_et, game_slug, COALESCE(player_id, 0), stat,
                COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown')
            )
                game_date_et, stat, result_status, locked_at_utc,
                COALESCE(model_family, 'unknown') AS model_family,
                COALESCE(model_version, 'unknown') AS model_version
            FROM bets.mlb_daily_forecast_ledger
            WHERE game_date_et >= %(cutoff)s
              AND forecast_phase = %(phase)s
            ORDER BY game_date_et, game_slug, COALESCE(player_id, 0), stat,
                     COALESCE(model_family, 'unknown'), COALESCE(model_version, 'unknown'),
                     locked_at_utc, id
        )
        SELECT stat, COALESCE(model_family, 'unknown') AS model_family,
               COALESCE(model_version, 'unknown') AS model_version,
               COUNT(*)::int AS rows,
               COUNT(*) FILTER (WHERE result_status = 'pending')::int AS pending_rows,
               COUNT(*) FILTER (WHERE result_status = 'graded')::int AS graded_rows,
               COUNT(DISTINCT game_date_et)::int AS dates,
               COUNT(DISTINCT game_date_et) FILTER (WHERE result_status = 'graded')::int AS graded_dates,
               MAX(locked_at_utc) AS latest_lock
        FROM canonical
        GROUP BY stat, COALESCE(model_family, 'unknown'),
                 COALESCE(model_version, 'unknown')
        ORDER BY MAX(locked_at_utc) DESC
        """,
        {"cutoff": cutoff, "phase": phase},
    )
    rows = []
    for record in df.to_dict(orient="records"):
        record["rows"] = int(record["rows"])
        record["pending_rows"] = int(record["pending_rows"])
        record["graded_rows"] = int(record["graded_rows"])
        record["dates"] = int(record["dates"])
        record["graded_dates"] = int(record["graded_dates"])
        record["latest_lock"] = pd.Timestamp(record["latest_lock"]).isoformat()
        rows.append(record)
    return rows


def _metrics(df: pd.DataFrame, prediction_col: str = "projection_value") -> dict[str, Any]:
    work = df.dropna(subset=[prediction_col, "actual_value"]).copy()
    if work.empty:
        return {"rows": 0}
    pred = pd.to_numeric(work[prediction_col], errors="coerce")
    actual = pd.to_numeric(work["actual_value"], errors="coerce")
    valid = pred.notna() & actual.notna()
    pred = pred.loc[valid]
    actual = actual.loc[valid]
    error = pred - actual
    return {
        "rows": int(len(error)),
        "dates": int(work.loc[valid, "game_date_et"].nunique()),
        "player_games": int(work.loc[valid, ["game_slug", "player_id"]].drop_duplicates().shape[0]),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
        "mean_prediction": float(pred.mean()),
        "mean_actual": float(actual.mean()),
        "brier": float(np.mean(np.square(error))) if work["stat"].eq("game_home_win").all() else None,
    }


def _stat_summary(group: pd.DataFrame, cfg: AuditConfig) -> dict[str, Any]:
    model = _metrics(group)
    baseline = _metrics(group, "baseline_value")
    model_only = _metrics(group, "model_only_projection")
    gain = None
    if model.get("mae") is not None and baseline.get("mae") is not None:
        gain = float(baseline["mae"]) - float(model["mae"])
    forecast_type = str(group["forecast_type"].iloc[0])
    minimum_rows = (
        cfg.min_game_rows if forecast_type == "game"
        else cfg.min_opportunity_rows if forecast_type == "opportunity"
        else cfg.min_player_rows
    )
    blockers = []
    if int(model.get("rows") or 0) < minimum_rows:
        blockers.append(f"rows<{minimum_rows}")
    if int(model.get("dates") or 0) < cfg.min_dates:
        blockers.append(f"dates<{cfg.min_dates}")
    if str(group["stat"].iloc[0]) in _PROJECTION_STATS:
        if int(baseline.get("rows") or 0) < max(30, minimum_rows // 2):
            blockers.append("baseline_coverage")
        elif gain is None or gain <= 0:
            blockers.append("model_not_better_than_simple_baseline")
    return {
        "forecast_type": forecast_type,
        "model": model,
        "baseline": baseline,
        "model_only": model_only,
        "mae_gain_vs_baseline": gain,
        "projection_eligible": not blockers,
        "blockers": blockers,
    }


def _walk_forward_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    folds = expanding_player_game_folds(
        df,
        test_window_days=7,
        min_train_dates=5,
        min_train_rows=1,
        min_holdout_rows=1,
    )
    rows = []
    for fold in folds:
        for stat, group in fold.holdout.groupby("stat"):
            rows.append({
                "fold": fold.fold_index,
                "holdout_start": str(fold.holdout_start),
                "holdout_end": str(fold.holdout_end),
                "stat": str(stat),
                "metrics": _metrics(group),
                "baseline": _metrics(group, "baseline_value"),
            })
    return rows


def _cohort_summaries(df: pd.DataFrame, cfg: AuditConfig) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    for (stat, family, version), group in df.groupby(
        ["stat", "model_family", "model_version"], dropna=False
    ):
        summary = _stat_summary(group, cfg)
        dates = int(summary.get("model", {}).get("dates") or 0)
        rows.append({
            "stat": str(stat),
            "model_family": str(family or "unknown"),
            "model_version": str(version or "unknown"),
            "latest_lock": pd.to_datetime(group["locked_at_utc"], utc=True).max().isoformat(),
            "dates_remaining_to_minimum": max(0, cfg.min_dates - dates),
            "dates_remaining_to_target": max(0, 10 - dates),
            **summary,
        })
    return sorted(rows, key=lambda row: row["latest_lock"], reverse=True)


def _offer_translation_audit(conn, cutoff: date) -> dict[str, Any]:
    if not _table_exists(conn, "features", "mlb_prop_market_training_examples"):
        return {"status": "missing_training_table", "rows": 0, "markets": {}}
    df = _query_df(
        conn,
        """
        SELECT id, replay_id, source_created_at, prop_offer_id, game_date_et,
               game_slug, player_id, market, side, bookmaker_key, market_line,
               market_price, model_prob_side::float, market_prob_side::float,
               CASE WHEN won IS TRUE THEN 1.0 WHEN won IS FALSE THEN 0.0 END AS target,
               pair_quality, true_pair_flag::float, synthetic_pair_flag::float,
               result_status
        FROM features.mlb_prop_market_training_examples
        WHERE game_date_et >= %(cutoff)s
          AND result_status = 'graded'
          AND won IS NOT NULL
        """,
        {"cutoff": cutoff},
    )
    if df.empty:
        return {"status": "no_rows", "rows": 0, "markets": {}}
    df = dedupe_locked_offer_rows(df)
    markets: dict[str, Any] = {}
    for market, group in df.groupby("market"):
        target = pd.to_numeric(group["target"], errors="coerce")
        model = pd.to_numeric(group["model_prob_side"], errors="coerce")
        market_prob = pd.to_numeric(group["market_prob_side"], errors="coerce")
        model_valid = target.notna() & model.notna()
        true_pair = (
            pd.to_numeric(group["true_pair_flag"], errors="coerce").fillna(0.0).ge(0.5)
            & pd.to_numeric(group["synthetic_pair_flag"], errors="coerce").fillna(0.0).lt(0.5)
        )
        market_valid = model_valid & market_prob.notna() & true_pair
        model_brier = float(np.mean(np.square(model.loc[model_valid] - target.loc[model_valid]))) if model_valid.any() else None
        paired_model_brier = float(np.mean(np.square(model.loc[market_valid] - target.loc[market_valid]))) if market_valid.any() else None
        market_brier = float(np.mean(np.square(market_prob.loc[market_valid] - target.loc[market_valid]))) if market_valid.any() else None
        markets[str(market)] = {
            "rows": int(model_valid.sum()),
            "true_pair_rows": int(market_valid.sum()),
            "true_pair_rate": float(market_valid.mean()),
            "model_brier": model_brier,
            "paired_model_brier": paired_model_brier,
            "market_brier": market_brier,
            "model_brier_gain_vs_market": (
                market_brier - paired_model_brier
                if market_brier is not None and paired_model_brier is not None else None
            ),
        }
    return {
        "status": "ready",
        "rows": int(len(df)),
        "raw_rows": int(df.attrs.get("raw_rows", len(df))),
        "deduped_rows": int(df.attrs.get("deduped_rows", 0)),
        "markets": markets,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _write_report(payload: dict[str, Any], cfg: AuditConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Daily Forecast Projection Audit",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Canonical evidence phase: {payload.get('canonical_forecast_phase')}",
        f"Forecast ledger rows: {payload.get('ledger_status', {}).get('total', 0)}",
        f"Pending forecasts: {payload.get('ledger_status', {}).get('by_status', {}).get('pending', 0)}",
        f"Canonical graded forecasts: {payload.get('rows', 0)}",
        f"Date range: {payload.get('date_min')} to {payload.get('date_max')}",
        f"Status: {payload.get('status')}",
        "",
        "## Projection Skill",
        "",
        "| Stat | Type | Rows | Dates | MAE | Baseline MAE | Gain | Bias | Eligible | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for stat, rec in (payload.get("stats") or {}).items():
        model = rec.get("model") or {}
        baseline = rec.get("baseline") or {}
        lines.append(
            f"| {stat} | {rec.get('forecast_type')} | {model.get('rows', 0)} | {model.get('dates', 0)} | "
            f"{_fmt(model.get('mae'))} | {_fmt(baseline.get('mae'))} | "
            f"{_fmt(rec.get('mae_gain_vs_baseline'))} | {_fmt(model.get('bias'))} | "
            f"{bool(rec.get('projection_eligible'))} | {', '.join(rec.get('blockers') or []) or '-'} |"
        )
    lines.extend([
        "",
        "## Model-Version Cohorts",
        "",
        "Each repaired model starts a new prospective evidence clock. Minimum proof is five dates; the target is ten.",
        "",
        "| Stat | Family | Version | Rows | Dates | MAE | Baseline | Eligible | Need 5 / 10 |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ])
    for cohort in payload.get("cohorts") or []:
        model = cohort.get("model") or {}
        baseline = cohort.get("baseline") or {}
        lines.append(
            f"| {cohort.get('stat')} | {cohort.get('model_family')} | {cohort.get('model_version')} | "
            f"{model.get('rows', 0)} | {model.get('dates', 0)} | {_fmt(model.get('mae'))} | "
            f"{_fmt(baseline.get('mae'))} | {bool(cohort.get('projection_eligible'))} | "
            f"{cohort.get('dates_remaining_to_minimum', 0)} / {cohort.get('dates_remaining_to_target', 0)} |"
        )
    lines.extend([
        "",
        "## Active Prospective Collection",
        "",
        "| Stat | Family | Version | Pending | Graded | Graded Dates | Need 5 / 10 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for stat, cohort in (payload.get("active_model_cohorts") or {}).items():
        graded_dates = int(cohort.get("graded_dates") or 0)
        lines.append(
            f"| {stat} | {cohort.get('model_family')} | {cohort.get('model_version')} | "
            f"{cohort.get('pending_rows', 0)} | {cohort.get('graded_rows', 0)} | {graded_dates} | "
            f"{max(0, cfg.min_dates - graded_dates)} / {max(0, 10 - graded_dates)} |"
        )
    lines.extend([
        "",
        "## Offer Translation Skill",
        "",
        "Only true, non-synthetic paired offers are used for market comparison.",
        "",
        "| Market | Rows | True Pair | Pair Rate | Model Brier | Paired Model | Market Brier | Gain vs Market |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for market, rec in ((payload.get("offer_translation") or {}).get("markets") or {}).items():
        lines.append(
            f"| {market} | {rec.get('rows', 0)} | {rec.get('true_pair_rows', 0)} | "
            f"{_fmt_pct(rec.get('true_pair_rate'))} | {_fmt(rec.get('model_brier'))} | "
            f"{_fmt(rec.get('paired_model_brier'))} | {_fmt(rec.get('market_brier'))} | "
            f"{_fmt(rec.get('model_brier_gain_vs_market'))} |"
        )
    lines.extend([
        "",
        "## Walk-Forward Windows",
        "",
        f"Complete seven-day OOF windows: {payload.get('walk_forward_fold_count', 0)}",
    ])
    atomic_write_text(path, "\n".join(lines) + "\n")
    return str(path)


def build_audit(cfg: AuditConfig) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    canonical_phase = canonical_forecast_phase()
    with psycopg2.connect(cfg.pg_dsn) as conn:
        ensure_daily_forecast_ledger_schema(conn)
        grade_daily_forecasts(conn, date_to=datetime.now(timezone.utc).date())
        ledger_status = _ledger_status_counts(conn, cutoff, canonical_phase)
        ledger_cohorts = _ledger_model_cohorts(conn, cutoff, canonical_phase)
        df = _load_forecasts(conn, cutoff, canonical_phase)
        offers = _offer_translation_audit(conn, cutoff)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "bets.mlb_daily_forecast_ledger",
        "canonical_forecast_phase": canonical_phase,
        "ledger_status": ledger_status,
        "ledger_model_cohorts": ledger_cohorts,
        "rows": int(len(df)),
        "date_min": str(df["game_date_et"].min()) if not df.empty else None,
        "date_max": str(df["game_date_et"].max()) if not df.empty else None,
        "stats": {},
        "offer_translation": offers,
    }
    if df.empty:
        payload["status"] = "collecting_prospective_forecasts"
        payload["walk_forward"] = []
    else:
        production_df = _non_shadow_df(df)
        evaluation_df = production_df if not production_df.empty else df
        payload["status"] = "ready"
        payload["stats"] = {
            str(stat): _stat_summary(group, cfg)
            for stat, group in evaluation_df.groupby("stat")
        }
        payload["cohorts"] = _cohort_summaries(df, cfg)
        payload["walk_forward"] = _walk_forward_summary(evaluation_df)
    payload.setdefault("cohorts", [])
    active_cohorts: dict[str, dict[str, Any]] = {}
    shadow_cohorts: dict[str, list[dict[str, Any]]] = {}
    for cohort in ledger_cohorts:
        if _is_shadow_model(cohort.get("model_family"), cohort.get("model_version")):
            shadow_cohorts.setdefault(str(cohort["stat"]), []).append(cohort)
            continue
        active_cohorts.setdefault(str(cohort["stat"]), cohort)
    cohort_metrics = {
        (str(row["stat"]), str(row["model_family"]), str(row["model_version"])): row
        for row in payload.get("cohorts") or []
    }
    projection_gates = {}
    for stat in sorted(_PROJECTION_STATS):
        active = active_cohorts.get(stat)
        matched = cohort_metrics.get((
            stat,
            str((active or {}).get("model_family") or "unknown"),
            str((active or {}).get("model_version") or "unknown"),
        )) if active else None
        projection_gates[stat] = {
            "eligible": bool((matched or {}).get("projection_eligible")),
            "blockers": (
                (matched or {}).get("blockers")
                or (["active_model_cohort_ungraded"] if active else ["no_prospective_rows"])
            ),
            "active_model_family": (active or {}).get("model_family"),
            "active_model_version": (active or {}).get("model_version"),
            "graded_rows": int(((matched or {}).get("model") or {}).get("rows") or 0),
            "graded_dates": int(((matched or {}).get("model") or {}).get("dates") or 0),
            "dates_remaining_to_minimum": (matched or {}).get("dates_remaining_to_minimum", cfg.min_dates),
            "dates_remaining_to_target": (matched or {}).get("dates_remaining_to_target", 10),
        }
    payload["active_model_cohorts"] = active_cohorts
    payload["shadow_model_cohorts"] = shadow_cohorts
    payload["projection_gates"] = projection_gates
    payload["walk_forward_fold_count"] = len({
        int(row["fold"]) for row in payload.get("walk_forward") or []
    })
    payload["report_path"] = _write_report(payload, cfg)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / cfg.artifact_file, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable MLB daily forecasts")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=180)
    args = parser.parse_args()
    payload = build_audit(AuditConfig(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days))
    print(json.dumps({
        "status": payload.get("status"),
        "rows": payload.get("rows", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
