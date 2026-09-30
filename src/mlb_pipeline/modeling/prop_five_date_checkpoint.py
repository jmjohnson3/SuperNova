"""Automatic prospective checkpoint for frozen MLB prop model releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .daily_forecast_ledger import TERMINAL_NON_FINAL_GAME_STATUSES
from .model_release import (
    canonical_forecast_phase,
    hitter_model_release_id,
    hitter_production_artifact_path,
    hitter_rate_shadow_release_id,
    pitcher_model_release_id,
)
from .prop_clean_slate import CleanSlateThresholds, load_clean_slate_rows
from .prop_micro_promotion_evaluation import evaluate as evaluate_micro

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MIN_DATES = 5

FORECAST_SQL = """
WITH canonical AS (
    SELECT DISTINCT ON (game_date_et, game_slug, COALESCE(player_id, 0), stat, model_version)
           game_date_et, game_slug, forecast_type, player_id, player_name, stat,
           projection_value::float, baseline_value::float, actual_value::float,
           result_status, model_family, model_version, locked_at_utc
    FROM bets.mlb_daily_forecast_ledger
    WHERE forecast_phase = %(phase)s
      AND model_version = ANY(%(versions)s)
    ORDER BY game_date_et, game_slug, COALESCE(player_id, 0), stat, model_version,
             locked_at_utc DESC, id DESC
)
SELECT c.*,
       g.status AS game_status,
       EXISTS (
           SELECT 1 FROM raw.mlb_player_gamelogs gl
           WHERE gl.game_slug = c.game_slug
       ) AS game_results_loaded,
       CASE WHEN c.player_id IS NULL THEN NULL ELSE EXISTS (
           SELECT 1 FROM raw.mlb_player_gamelogs gl
           WHERE gl.game_slug = c.game_slug AND gl.player_id = c.player_id
             AND CASE
                 WHEN c.stat LIKE 'pitcher_%%' THEN
                     COALESCE(
                         gl.innings_pitched,
                         gl.strikeouts_pitcher,
                         gl.hits_allowed,
                         gl.walks_allowed,
                         gl.runs_allowed,
                         gl.earned_runs,
                         gl.home_runs_allowed
                     ) IS NOT NULL
                 WHEN c.stat LIKE 'batter_%%' OR c.stat LIKE 'hitter_%%' THEN
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
       ) END AS player_participated
FROM canonical c
LEFT JOIN raw.mlb_games g ON g.game_slug = c.game_slug
ORDER BY game_date_et, game_slug, player_id, stat
"""


def _is_void_checkpoint_status(value: Any) -> bool:
    return str(value or "").startswith("void")


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _game_completion(conn, dates: list[date]) -> dict[date, bool]:
    if not dates:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT game_date_et, COUNT(*)::int AS games,
                   COUNT(*) FILTER (
                       WHERE status = 'final'
                          OR LOWER(COALESCE(status, '')) = ANY(%s)
                   )::int AS terminal_games
            FROM raw.mlb_games
            WHERE game_date_et = ANY(%s)
            GROUP BY game_date_et
            """,
            (list(TERMINAL_NON_FINAL_GAME_STATUSES), dates),
        )
        return {
            game_date: bool(games > 0 and games == terminal_games)
            for game_date, games, terminal_games in cur.fetchall()
        }


def _artifact_integrity(model_dir: Path) -> dict[str, Any]:
    artifact = hitter_production_artifact_path(model_dir)
    metadata_path = model_dir / "hitter_player_game_outcome_models.production.json"
    if not artifact.exists() or not metadata_path.exists():
        return {"valid": False, "reason": "production_artifact_or_metadata_missing"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except Exception as exc:
        return {"valid": False, "reason": f"artifact_integrity_error:{exc}"}
    return {
        "valid": bool(actual_hash == metadata.get("sha256")),
        "expected_sha256": metadata.get("sha256"),
        "actual_sha256": actual_hash,
        "model_release_id": metadata.get("model_release_id"),
        "frozen_at_utc": metadata.get("production_frozen_at_utc"),
    }


def _metric(group: pd.DataFrame) -> dict[str, Any]:
    work = group.dropna(subset=["projection_value", "actual_value"]).copy()
    if work.empty:
        return {"rows": 0, "dates": 0, "projection_pass": False}
    prediction = pd.to_numeric(work["projection_value"], errors="coerce")
    actual = pd.to_numeric(work["actual_value"], errors="coerce")
    baseline = pd.to_numeric(work["baseline_value"], errors="coerce")
    valid = prediction.notna() & actual.notna()
    error = prediction.loc[valid] - actual.loc[valid]
    baseline_valid = valid & baseline.notna()
    baseline_mae = float((baseline.loc[baseline_valid] - actual.loc[baseline_valid]).abs().mean()) if baseline_valid.any() else None
    mae = float(error.abs().mean())
    gain = baseline_mae - mae if baseline_mae is not None else None
    return {
        "rows": int(valid.sum()),
        "dates": int(work.loc[valid, "game_date_et"].nunique()),
        "mae": mae,
        "rmse": float(math.sqrt(float((error ** 2).mean()))),
        "bias": float(error.mean()),
        "baseline_mae": baseline_mae,
        "mae_gain_vs_baseline": gain,
        "projection_pass": bool(gain is not None and gain > 0.0),
    }


def _release_summary(
    df: pd.DataFrame,
    *,
    release_id: str,
    anchor_stat: str,
    required_stats: tuple[str, ...],
    game_completion: dict[date, bool],
) -> dict[str, Any]:
    cohort = df.loc[df["model_version"].astype(str).eq(release_id)].copy()
    anchor = cohort.loc[cohort["stat"].eq(anchor_stat)]
    completed_dates: list[date] = []
    date_rows: list[dict[str, Any]] = []
    for game_date, group in anchor.groupby("game_date_et", dropna=False):
        pending = int(group["checkpoint_status"].eq("pending").sum())
        void = int(group["checkpoint_status"].map(_is_void_checkpoint_status).sum())
        graded = int(group["result_status"].eq("graded").sum())
        games_final = bool(game_completion.get(game_date, False))
        complete = bool(games_final and pending == 0 and graded > 0)
        if complete:
            completed_dates.append(game_date)
        date_rows.append({
            "game_date_et": str(game_date),
            "graded_rows": graded,
            "pending_rows": pending,
            "void_rows": void,
            "games_final": games_final,
            "complete": complete,
        })
    settled = cohort.loc[
        cohort["game_date_et"].isin(completed_dates)
        & cohort["result_status"].eq("graded")
    ]
    metrics = {
        stat: _metric(settled.loc[settled["stat"].eq(stat)])
        for stat in required_stats
    }
    complete_count = len(completed_dates)
    return {
        "release_id": release_id,
        "anchor_stat": anchor_stat,
        "status": "evaluation_ready" if complete_count >= _MIN_DATES else "collecting",
        "completed_dates": [str(value) for value in sorted(completed_dates)],
        "completed_date_count": complete_count,
        "dates_remaining": max(0, _MIN_DATES - complete_count),
        "calendar_dates_seen": int(anchor["game_date_et"].nunique()) if not anchor.empty else 0,
        "pending_anchor_rows": int(anchor["checkpoint_status"].eq("pending").sum()),
        "void_anchor_rows": int(anchor["checkpoint_status"].map(_is_void_checkpoint_status).sum()),
        "by_date": sorted(date_rows, key=lambda row: row["game_date_et"]),
        "metrics": metrics,
    }


def _clean_dates(conn, dates: list[date]) -> dict[str, Any]:
    if not dates:
        return {"clean_dates": [], "count": 0, "rows": []}
    rows = load_clean_slate_rows(
        conn,
        date_from=min(dates),
        date_to=max(dates),
        thresholds=CleanSlateThresholds(),
    )
    date_set = set(dates)
    rows = [row for row in rows if row.get("slate_date") in date_set]
    clean = [str(row["slate_date"]) for row in rows if row.get("clean_slate")]
    return {
        "clean_dates": clean,
        "count": len(clean),
        "rows": [{
            "slate_date": str(row.get("slate_date")),
            "clean_slate": bool(row.get("clean_slate")),
            "valid_clv_coverage": row.get("valid_clv_coverage"),
            "stale_close_rate": row.get("stale_close_rate"),
            "reasons": row.get("clean_slate_reasons") or [],
        } for row in rows],
    }


def build(model_dir: Path = _MODEL_DIR, pg_dsn: str = PG_DSN) -> dict[str, Any]:
    hitter_release = hitter_model_release_id()
    hitter_shadow_release = hitter_rate_shadow_release_id()
    pitcher_release = pitcher_model_release_id()
    versions = sorted({hitter_release, hitter_shadow_release, pitcher_release})
    with psycopg2.connect(pg_dsn) as conn:
        forecasts = _query_df(conn, FORECAST_SQL, {
            "phase": canonical_forecast_phase(),
            "versions": versions,
        })
        if not forecasts.empty:
            forecasts["game_date_et"] = pd.to_datetime(forecasts["game_date_et"]).dt.date
        forecasts["checkpoint_status"] = (
            forecasts["result_status"].astype(str)
            if "result_status" in forecasts
            else pd.Series(dtype=str, index=forecasts.index)
        )
        seen_dates = sorted(forecasts["game_date_et"].dropna().unique()) if not forecasts.empty else []
        completion = _game_completion(conn, seen_dates)
        if not forecasts.empty:
            terminal_non_final_mask = forecasts["game_status"].fillna("").astype(str).str.lower().isin(
                TERMINAL_NON_FINAL_GAME_STATUSES
            )
            forecasts.loc[
                forecasts["result_status"].ne("graded") & terminal_non_final_mask,
                "checkpoint_status",
            ] = "void_game_not_played"
            void_mask = (
                forecasts["result_status"].ne("graded")
                & forecasts["game_date_et"].map(completion).fillna(False)
                & forecasts["game_results_loaded"].fillna(False).astype(bool)
                & forecasts["player_id"].notna()
                & ~forecasts["player_participated"].fillna(False).astype(bool)
            )
            forecasts.loc[void_mask, "checkpoint_status"] = "void_nonparticipant"
        hitter = _release_summary(
            forecasts,
            release_id=hitter_release,
            anchor_stat="batter_total_bases",
            required_stats=(
                "batter_hits", "batter_total_bases", "batter_home_runs",
                "hitter_plate_appearances", "hitter_plate_appearances_challenger",
            ),
            game_completion=completion,
        )
        pitcher = _release_summary(
            forecasts,
            release_id=pitcher_release,
            anchor_stat="pitcher_strikeouts",
            required_stats=(
                "pitcher_strikeouts", "pitcher_batters_faced",
                "pitcher_pitch_count", "pitcher_innings",
            ),
            game_completion=completion,
        )
        hitter_shadow = _release_summary(
            forecasts,
            release_id=hitter_shadow_release,
            anchor_stat="batter_hits",
            required_stats=("batter_hits", "batter_home_runs"),
            game_completion=completion,
        )
        hitter_clean = _clean_dates(conn, [date.fromisoformat(value) for value in hitter["completed_dates"]])
        pitcher_clean = _clean_dates(conn, [date.fromisoformat(value) for value in pitcher["completed_dates"]])
        hitter_shadow_clean = _clean_dates(
            conn,
            [date.fromisoformat(value) for value in hitter_shadow["completed_dates"]],
        )
    hitter["clean_slate_evidence"] = hitter_clean
    pitcher["clean_slate_evidence"] = pitcher_clean
    hitter_shadow["clean_slate_evidence"] = hitter_shadow_clean
    hitter_shadow["micro_ready_buckets"] = 0
    micro = evaluate_micro(model_dir)
    bucket_rows = micro.get("buckets") or []
    release_buckets: dict[str, list[dict[str, Any]]] = {}
    for release_id, release in ((hitter_release, hitter), (pitcher_release, pitcher)):
        rows = [row for row in bucket_rows if str(row.get("stable_release_id") or "") == release_id]
        clean_slate_pass = int((release.get("clean_slate_evidence") or {}).get("count") or 0) >= _MIN_DATES
        for row in rows:
            market_metric = (release.get("metrics") or {}).get(str(row.get("market") or "")) or {}
            row["checkpoint_projection_pass"] = bool(market_metric.get("projection_pass"))
            row["checkpoint_clean_slate_pass"] = bool(clean_slate_pass)
            row["checkpoint_ready"] = bool(
                release["completed_date_count"] >= _MIN_DATES
                and row.get("micro_ready")
                and row["checkpoint_projection_pass"]
                and row["checkpoint_clean_slate_pass"]
            )
        release_buckets[release_id] = rows
    hitter["micro_ready_buckets"] = sum(1 for row in release_buckets[hitter_release] if row.get("checkpoint_ready"))
    pitcher["micro_ready_buckets"] = sum(1 for row in release_buckets[pitcher_release] if row.get("checkpoint_ready"))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "canonical_forecast_phase": canonical_forecast_phase(),
        "minimum_completed_dates": _MIN_DATES,
        "artifact_integrity": _artifact_integrity(model_dir),
        "hitter_release": hitter,
        "hitter_rate_shadow_release": hitter_shadow,
        "pitcher_release": pitcher,
        "release_buckets": release_buckets,
        "status": (
            "micro_ready"
            if hitter["micro_ready_buckets"] + pitcher["micro_ready_buckets"] > 0
            else "evaluation_ready_no_micro_buckets"
            if hitter["status"] == "evaluation_ready" and pitcher["status"] == "evaluation_ready"
            else "collecting"
        ),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_five_date_checkpoint.json", payload)
    report_path = _REPORT_DIR / "mlb_prop_five_date_checkpoint_latest.md"
    report = _render(payload)
    atomic_write_text(report_path, report)
    atomic_write_text(
        _REPORT_DIR / f"mlb_prop_five_date_checkpoint_{datetime.now(timezone.utc).date().isoformat()}.md",
        report,
    )
    payload["report_path"] = str(report_path)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _release_lines(label: str, release: dict[str, Any]) -> list[str]:
    lines = [
        f"## {label}", "",
        f"Release: `{release['release_id']}`",
        f"Status: **{release['status']}**",
        f"Completed dates: {release['completed_date_count']} / {_MIN_DATES}",
        f"Dates remaining: {release['dates_remaining']}",
        f"Pending anchor rows: {release['pending_anchor_rows']}",
        f"Verified nonparticipant voids: {release['void_anchor_rows']}",
        f"Strict clean completed dates: {release['clean_slate_evidence']['count']}",
        f"Checkpoint-ready $1 micro buckets: {release['micro_ready_buckets']}", "",
        "| Stat | Rows | Dates | MAE | Baseline | Gain | Bias | Projection Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stat, metric in release["metrics"].items():
        lines.append(
            f"| {stat} | {metric.get('rows', 0)} | {metric.get('dates', 0)} | {_fmt(metric.get('mae'))} | "
            f"{_fmt(metric.get('baseline_mae'))} | {_fmt(metric.get('mae_gain_vs_baseline'))} | "
            f"{_fmt(metric.get('bias'))} | {metric.get('projection_pass', False)} |"
        )
    lines.extend(["", "| Date | Graded | Pending | Voids | Games Settled | Complete |", "|---|---:|---:|---:|---|---|"])
    for row in release["by_date"]:
        lines.append(
            f"| {row['game_date_et']} | {row['graded_rows']} | {row['pending_rows']} | {row['void_rows']} | "
            f"{row['games_final']} | {row['complete']} |"
        )
    lines.append("")
    return lines


def _render(payload: dict[str, Any]) -> str:
    integrity = payload["artifact_integrity"]
    lines = [
        "# MLB Prop Five-Date Checkpoint", "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Status: **{payload['status']}**",
        f"Canonical phase: `{payload['canonical_forecast_phase']}`",
        f"Frozen hitter artifact valid: **{integrity.get('valid', False)}**", "",
        "A date counts only after every game is final or terminal non-played, and the release's anchor forecasts have zero pending rows.", "",
        *_release_lines("Hitter Release", payload["hitter_release"]),
        *_release_lines("Hitter Hits/HR Shadow", payload["hitter_rate_shadow_release"]),
        *_release_lines("Pitcher Release", payload["pitcher_release"]),
        "## Decision", "",
    ]
    if payload["status"] == "collecting":
        lines.append("Keep production frozen and continue prospective collection. No $1 micro promotion is allowed yet.")
    elif payload["status"] == "micro_ready":
        lines.append("At least one exact bucket passed the completed-date, projection, ROI, CLV, calibration, and close-quality gates for $1 micro review.")
    else:
        lines.append("The five-date evaluation is available, but no exact bucket currently passes every $1 micro gate.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen MLB prop releases after five complete dates")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--pg-dsn", default=PG_DSN)
    args = parser.parse_args()
    payload = build(Path(args.model_dir), args.pg_dsn)
    print(json.dumps({
        "status": payload["status"],
        "hitter_completed_dates": payload["hitter_release"]["completed_date_count"],
        "hitter_rate_shadow_completed_dates": payload["hitter_rate_shadow_release"]["completed_date_count"],
        "pitcher_completed_dates": payload["pitcher_release"]["completed_date_count"],
        "report_path": payload["report_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
