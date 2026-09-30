"""Player-game-first proof gates before exact-line bankroll prop training."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.db import PG_DSN

from .daily_forecast_projection_audit import AuditConfig, build_audit

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_STATS = ("pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs")

SQL_OFFER_PROOF = """
SELECT
    market,
    COUNT(*)::int AS rows,
    COUNT(*) FILTER (
        WHERE COALESCE(true_pair_flag, 0) >= 0.5
          AND COALESCE(synthetic_pair_flag, 0) < 0.5
    )::int AS true_pair_rows,
    COUNT(*) FILTER (WHERE clv_valid IS TRUE)::int AS clv_rows,
    AVG(CASE WHEN clv_valid IS TRUE AND clv_price > 0 THEN 1.0 WHEN clv_valid IS TRUE THEN 0.0 ELSE NULL END)::float AS clv_beat_rate,
    AVG(clv_price::float) FILTER (WHERE clv_valid IS TRUE)::float AS avg_clv_price,
    COUNT(DISTINCT game_date_et)::int AS dates
FROM features.mlb_prop_market_training_examples
WHERE game_date_et >= %(cutoff)s
  AND market = ANY(%(stats)s)
  AND won IS NOT NULL
  AND COALESCE(push, false) IS FALSE
GROUP BY market
ORDER BY market
"""


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _tb_tail_projection_proof(model_dir: Path) -> dict[str, Any]:
    payload = _load_json(model_dir / "prop_tb_tail_repair_challenger.json")
    if not payload or payload.get("status") != "ready":
        return {}
    tb_count = payload.get("tb_count") or {}
    baseline = tb_count.get("baseline") or {}
    direct = tb_count.get("direct_state_expected") or {}
    line = payload.get("true_pair_line_pricing") or {}
    state = payload.get("state_brier") or {}
    mae_gain = _float(tb_count.get("mae_gain"))
    line_gain = _float(line.get("selected_blend_brier_gain_vs_baseline"))
    if line_gain is None:
        line_gain = _float(line.get("calibrated_brier_gain_vs_baseline"))
    state_gain = _float(state.get("gain"))
    accepted = bool(payload.get("accepted"))
    eligible = bool(
        accepted
        and mae_gain is not None and mae_gain > 0.0
        and line_gain is not None and line_gain > 0.0
        and state_gain is not None and state_gain > 0.0
    )
    return {
        "eligible": eligible,
        "source": "prop_tb_tail_repair_challenger",
        "active_model_family": "tb_tail_state_repair",
        "active_model_version": payload.get("generated_at_utc"),
        "graded_rows": int(payload.get("evaluation_rows") or 0),
        "graded_dates": int(payload.get("dates") or 0),
        "mae": direct.get("mae"),
        "baseline_mae": baseline.get("mae"),
        "mae_gain_vs_baseline": mae_gain,
        "line_brier": line.get("selected_blend_brier", line.get("calibrated_brier")),
        "baseline_line_brier": line.get("baseline_brier"),
        "line_brier_gain_vs_baseline": line_gain,
        "state_brier_gain_vs_baseline": state_gain,
        "projection_blockers": [] if eligible else ["tb_tail_repair_not_accepted_or_not_better"],
    }


def _live_hitter_rate_projection_proof(model_dir: Path, stat: str) -> dict[str, Any]:
    """Use prospective live-vs-legacy hitter rate evidence as a per-stat guard."""
    if stat not in {"batter_hits", "batter_home_runs"}:
        return {}
    payload = _load_json(model_dir / "hitter_live_vs_legacy_forecast_diff.json")
    if not payload:
        return {}
    stat_key = "home_runs" if stat == "batter_home_runs" else "hits"
    rec = ((payload.get("graded_comparison") or {}).get(stat_key) or {})
    rows = int(rec.get("rows") or 0)
    dates = int(rec.get("dates") or 0)
    mae_gain = _float(rec.get("mae_gain"))
    live_mae = _float(rec.get("live_mae"))
    legacy_mae = _float(rec.get("legacy_mae"))
    live_bias = _float(rec.get("live_bias"))
    legacy_bias = _float(rec.get("legacy_bias"))
    if stat == "batter_hits":
        repair = payload.get("hits_live_bias_repair_v1") or {}
        selected = repair.get("selected") or {}
        if repair.get("accepted"):
            repair_rows = int(repair.get("rows") or 0)
            repair_dates = int(repair.get("dates") or 0)
            repair_mae_gain = _float(selected.get("mae_gain_vs_legacy"))
            repair_brier_gain = _float(selected.get("any_brier_gain_vs_legacy"))
            repair_bias = _float(selected.get("bias"))
            blockers: list[str] = []
            if repair_rows < 100:
                blockers.append("hits_live_bias_repair_rows<100")
            if repair_dates < 5:
                blockers.append("hits_live_bias_repair_dates<5")
            if repair_mae_gain is None or repair_mae_gain <= 0.0:
                blockers.append("hits_live_bias_repair_mae_not_improved")
            if repair_brier_gain is None or repair_brier_gain <= 0.0:
                blockers.append("hits_live_bias_repair_brier_not_improved")
            if repair_bias is not None and legacy_bias is not None and abs(repair_bias) > abs(legacy_bias) + 0.05:
                blockers.append("hits_live_bias_repair_bias_worse")
            return {
                "eligible": not blockers,
                "source": "hitter_hits_live_bias_repair_v1",
                "active_model_family": "hitter_hits_direct_player_game_blend_hits_bias_calibrated",
                "active_model_version": payload.get("generated_at_utc"),
                "graded_rows": repair_rows,
                "graded_dates": repair_dates,
                "mae": _float(selected.get("mae")),
                "baseline_mae": legacy_mae,
                "mae_gain_vs_baseline": repair_mae_gain,
                "bias": repair_bias,
                "baseline_bias": legacy_bias,
                "line_brier": _float(selected.get("any_brier")),
                "baseline_line_brier": _float((repair.get("legacy") or {}).get("any_brier")),
                "line_brier_gain_vs_baseline": repair_brier_gain,
                "projection_blockers": blockers,
            }
    blockers: list[str] = []
    if rows < 100:
        blockers.append("live_vs_legacy_rows<100")
    if dates < 5:
        blockers.append("live_vs_legacy_dates<5")
    if mae_gain is None or mae_gain <= 0.0:
        blockers.append("live_vs_legacy_mae_not_improved")
    if (
        live_bias is not None
        and legacy_bias is not None
        and abs(live_bias) > abs(legacy_bias) + 0.05
    ):
        blockers.append("live_vs_legacy_bias_worse")

    line_brier = None
    legacy_line_brier = None
    line_brier_gain = None
    if stat == "batter_home_runs":
        brier_rec = rec.get("hr_0_5_brier") or {}
        line_brier = _float(brier_rec.get("live_brier"))
        legacy_line_brier = _float(brier_rec.get("legacy_brier"))
        line_brier_gain = _float(brier_rec.get("brier_gain"))
        if line_brier_gain is None or line_brier_gain <= 0.0:
            blockers.append("hr_0_5_brier_not_improved")

    return {
        "eligible": not blockers,
        "source": "hitter_live_vs_legacy_forecast_diff",
        "active_model_family": (
            "hitter_hr_direct_player_game_blend"
            if stat == "batter_home_runs"
            else "hitter_hits_direct_player_game_blend"
        ),
        "active_model_version": payload.get("generated_at_utc"),
        "graded_rows": rows,
        "graded_dates": dates,
        "mae": live_mae,
        "baseline_mae": legacy_mae,
        "mae_gain_vs_baseline": mae_gain,
        "bias": live_bias,
        "baseline_bias": legacy_bias,
        "line_brier": line_brier,
        "baseline_line_brier": legacy_line_brier,
        "line_brier_gain_vs_baseline": line_brier_gain,
        "projection_blockers": blockers,
    }


def _hitter_outcome_projection_proof(model_dir: Path, stat: str) -> dict[str, Any]:
    """Use player-game hitter outcome artifact for stat-specific repair evidence."""
    if stat not in {"batter_hits", "batter_home_runs"}:
        return {}
    payload = _load_json(model_dir / "hitter_player_game_outcome_models.json")
    if not payload or str(payload.get("status") or "").lower() not in {"ready", "ok"}:
        return {}
    rec = payload.get("recommendation") or {}
    rows = int(payload.get("holdout_rows") or 0)
    dates = 0
    if payload.get("holdout_start") and payload.get("holdout_end"):
        try:
            start = datetime.fromisoformat(str(payload["holdout_start"]))
            end = datetime.fromisoformat(str(payload["holdout_end"]))
            dates = max(1, (end.date() - start.date()).days + 1)
        except Exception:
            dates = 0
    if stat == "batter_hits":
        mae_gain = _float(rec.get("direct_hits_count_repair_mae_gain"))
        brier_gain = None
        enabled = bool(rec.get("direct_hits_count_repair_enabled"))
        active_family = "hitter_hits_direct_player_game_blend"
    else:
        mae_gain = _float(rec.get("direct_hr_count_repair_mae_gain"))
        brier_gain = _float(rec.get("direct_hr_count_repair_brier_gain"))
        enabled = bool(rec.get("direct_hr_count_repair_enabled"))
        active_family = "hitter_hr_direct_player_game_blend"
    blockers: list[str] = []
    if rows < 100:
        blockers.append("hitter_outcome_holdout_rows<100")
    if not enabled:
        blockers.append("direct_count_repair_not_enabled")
    if mae_gain is None or mae_gain <= 0.0:
        blockers.append("direct_count_repair_mae_not_improved")
    if stat == "batter_home_runs" and (brier_gain is None or brier_gain <= 0.0):
        blockers.append("direct_hr_count_repair_brier_not_improved")
    return {
        "eligible": not blockers,
        "source": "hitter_player_game_outcome_models",
        "active_model_family": active_family,
        "active_model_version": payload.get("model_release_id") or payload.get("generated_at_utc"),
        "graded_rows": rows,
        "graded_dates": dates,
        "mae": None,
        "baseline_mae": None,
        "mae_gain_vs_baseline": mae_gain,
        "line_brier": None,
        "baseline_line_brier": None,
        "line_brier_gain_vs_baseline": brier_gain,
        "projection_blockers": blockers,
    }


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,))
        return bool(cur.fetchone()[0])


def _offer_proof(pg_dsn: str, lookback_days: int) -> dict[str, dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, int(lookback_days)))
    with psycopg2.connect(pg_dsn) as conn:
        if not _table_exists(conn, "features.mlb_prop_market_training_examples"):
            return {}
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_OFFER_PROOF, {"cutoff": cutoff, "stats": list(_STATS)})
            return {str(row["market"]): dict(row) for row in cur.fetchall()}


def build(*, pg_dsn: str = PG_DSN, lookback_days: int = 180) -> dict[str, Any]:
    audit = build_audit(AuditConfig(pg_dsn=pg_dsn, lookback_days=lookback_days))
    offer = _offer_proof(pg_dsn, lookback_days)
    projection_gates = audit.get("projection_gates") or {}
    tb_tail_proof = _tb_tail_projection_proof(_MODEL_DIR)
    live_hitter_proofs = {
        stat: _live_hitter_rate_projection_proof(_MODEL_DIR, stat)
        for stat in ("batter_hits", "batter_home_runs")
    }
    outcome_hitter_proofs = {
        stat: _hitter_outcome_projection_proof(_MODEL_DIR, stat)
        for stat in ("batter_hits", "batter_home_runs")
    }
    rows: list[dict[str, Any]] = []
    for stat in _STATS:
        gate = projection_gates.get(stat) or {}
        stat_summary = (audit.get("stats") or {}).get(stat) or {}
        model = stat_summary.get("model") or {}
        baseline = stat_summary.get("baseline") or {}
        audit_gain = _float(stat_summary.get("mae_gain_vs_baseline"))
        projection_blockers = list(gate.get("blockers") or [])
        projection_eligible = bool(gate.get("eligible"))
        if projection_eligible and audit_gain is not None and audit_gain <= 0.0:
            projection_eligible = False
            projection_blockers.append("model_not_better_than_simple_baseline")
        if stat == "batter_total_bases" and tb_tail_proof.get("eligible"):
            gate = {
                **gate,
                "eligible": True,
                "blockers": [],
                "active_model_family": tb_tail_proof.get("active_model_family"),
                "active_model_version": tb_tail_proof.get("active_model_version"),
                "graded_rows": tb_tail_proof.get("graded_rows"),
                "graded_dates": tb_tail_proof.get("graded_dates"),
            }
            model = {
                **model,
                "rows": tb_tail_proof.get("graded_rows"),
                "dates": tb_tail_proof.get("graded_dates"),
                "mae": tb_tail_proof.get("mae"),
            }
            baseline = {
                **baseline,
                "mae": tb_tail_proof.get("baseline_mae"),
            }
            stat_summary = {
                **stat_summary,
                "mae_gain_vs_baseline": tb_tail_proof.get("mae_gain_vs_baseline"),
            }
            projection_blockers = []
            projection_eligible = True
        hitter_live_proof = live_hitter_proofs.get(stat) or {}
        hitter_outcome_proof = outcome_hitter_proofs.get(stat) or {}
        if (
            stat in {"batter_hits", "batter_home_runs"}
            and hitter_live_proof.get("eligible")
            and hitter_outcome_proof.get("eligible")
        ):
            gate = {
                **gate,
                "eligible": True,
                "blockers": [],
                "active_model_family": hitter_live_proof.get("active_model_family"),
                "active_model_version": hitter_live_proof.get("active_model_version"),
                "graded_rows": hitter_live_proof.get("graded_rows"),
                "graded_dates": hitter_live_proof.get("graded_dates"),
            }
            model = {
                **model,
                "rows": hitter_live_proof.get("graded_rows"),
                "dates": hitter_live_proof.get("graded_dates"),
                "mae": hitter_live_proof.get("mae"),
            }
            baseline = {
                **baseline,
                "mae": hitter_live_proof.get("baseline_mae"),
            }
            stat_summary = {
                **stat_summary,
                "mae_gain_vs_baseline": hitter_live_proof.get("mae_gain_vs_baseline"),
            }
            projection_blockers = []
            projection_eligible = True
        elif stat in live_hitter_proofs and hitter_live_proof:
            projection_blockers.extend(hitter_live_proof.get("projection_blockers") or [])
        offer_rec = offer.get(stat) or {}
        true_pair_rows = int(offer_rec.get("true_pair_rows") or 0)
        clv_rows = int(offer_rec.get("clv_rows") or 0)
        clv_beat = _float(offer_rec.get("clv_beat_rate"))
        avg_clv = _float(offer_rec.get("avg_clv_price"))
        blockers = list(projection_blockers)
        if not projection_eligible:
            blockers.append("underlying_player_game_projection_not_proven")
        if true_pair_rows < 150:
            blockers.append("true_pair_rows<150")
        if clv_rows < 30:
            blockers.append("clv_rows<30")
        if clv_beat is None or clv_beat < 0.52:
            blockers.append("clv_truth_not_confirming")
        if avg_clv is None or avg_clv <= 0.0:
            blockers.append("avg_clv_not_positive")
        rows.append({
            "stat": stat,
            "projection_eligible": projection_eligible,
            "projection_blockers": sorted(set(str(value) for value in projection_blockers if value)),
            "active_model_family": gate.get("active_model_family"),
            "active_model_version": gate.get("active_model_version"),
            "projection_rows": int(model.get("rows") or gate.get("graded_rows") or 0),
            "projection_dates": int(model.get("dates") or gate.get("graded_dates") or 0),
            "mae": model.get("mae"),
            "baseline_mae": baseline.get("mae"),
            "mae_gain_vs_baseline": stat_summary.get("mae_gain_vs_baseline"),
            "offer_rows": int(offer_rec.get("rows") or 0),
            "offer_dates": int(offer_rec.get("dates") or 0),
            "true_pair_rows": true_pair_rows,
            "clv_rows": clv_rows,
            "clv_beat_rate": clv_beat,
            "avg_clv_price": avg_clv,
            "projection_micro_allowed": projection_eligible,
            "exact_line_bankroll_training_allowed": not blockers,
            "blockers": sorted(set(str(value) for value in blockers if value)),
            "tb_tail_projection_proof": tb_tail_proof if stat == "batter_total_bases" else None,
            "hitter_live_projection_proof": hitter_live_proof if stat in live_hitter_proofs else None,
            "hitter_outcome_projection_proof": hitter_outcome_proof if stat in outcome_hitter_proofs else None,
        })
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "usage": "player_game_projection_first_bankroll_training_gate",
        "lookback_days": int(lookback_days),
        "forecast_audit_status": audit.get("status"),
        "rows": rows,
        "projection_allowed_stats": [row["stat"] for row in rows if row["projection_micro_allowed"]],
        "allowed_stats": [row["stat"] for row in rows if row["exact_line_bankroll_training_allowed"]],
    }
    _write_outputs(payload)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _pct(value: Any) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.1f}%"


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Player-Game Bankroll Model Proof",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Usage: exact-line bankroll models may train/promote only after the player-game stat forecast proves it beats baseline.",
        "",
        f"- Projection-micro allowed stats: {', '.join(payload.get('projection_allowed_stats') or []) or '-'}",
        f"- Allowed stats: {', '.join(payload.get('allowed_stats') or []) or '-'}",
        "",
        "| Stat | Projection Rows | Dates | MAE | Baseline | Gain | Projection Proven | Projection Micro | True Pair | CLV Rows | CLV Beat | Avg CLV | Exact-Line Training Allowed | Blockers |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| {row.get('stat')} | {row.get('projection_rows')} | {row.get('projection_dates')} | "
            f"{_fmt(row.get('mae'))} | {_fmt(row.get('baseline_mae'))} | "
            f"{_fmt(row.get('mae_gain_vs_baseline'))} | {bool(row.get('projection_eligible'))} | "
            f"{bool(row.get('projection_micro_allowed'))} | "
            f"{row.get('true_pair_rows')} | {row.get('clv_rows')} | {_pct(row.get('clv_beat_rate'))} | "
            f"{_fmt(row.get('avg_clv_price'))} | {bool(row.get('exact_line_bankroll_training_allowed'))} | "
            f"{', '.join(row.get('blockers') or []) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _MODEL_DIR / "prop_player_game_bankroll_model_proof.json"
    report_path = _REPORT_DIR / "mlb_prop_player_game_bankroll_model_proof_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player-game-first bankroll prop model proof report")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=180)
    args = parser.parse_args()
    payload = build(pg_dsn=args.pg_dsn, lookback_days=args.lookback_days)
    print(json.dumps({
        "status": payload.get("status"),
        "allowed_stats": payload.get("allowed_stats"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
