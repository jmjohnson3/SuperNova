"""Combine exact-bucket, model, CLV, and prospective opportunity proof for micro stakes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _checkpoint_projection_gates(checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    min_dates = int(checkpoint.get("minimum_completed_dates") or 5)
    artifact_valid = bool((checkpoint.get("artifact_integrity") or {}).get("valid", True))
    for section in ("hitter_release", "pitcher_release"):
        release = checkpoint.get(section) or {}
        release_id = release.get("release_id")
        completed_dates = int(release.get("completed_date_count") or 0)
        for stat, metric in (release.get("metrics") or {}).items():
            blockers: list[str] = []
            if completed_dates < min_dates:
                blockers.append("dates<5")
            if not metric.get("projection_pass"):
                blockers.append("model_not_better_than_simple_baseline")
            if section == "hitter_release" and not artifact_valid:
                blockers.append("frozen_hitter_artifact_integrity_failed")
            gates[str(stat)] = {
                "eligible": not blockers,
                "blockers": blockers,
                "active_model_family": "five_date_checkpoint",
                "active_model_version": release_id,
                "graded_rows": int(metric.get("rows") or 0),
                "graded_dates": completed_dates,
                "dates_remaining_to_minimum": max(0, min_dates - completed_dates),
                "source": "prop_five_date_checkpoint",
                "mae": metric.get("mae"),
                "baseline_mae": metric.get("baseline_mae"),
                "mae_gain_vs_baseline": metric.get("mae_gain_vs_baseline"),
            }
    return gates


def _backdated_promotion_dates(audit: dict[str, Any]) -> list[str]:
    return sorted(str(value) for value in (audit.get("prop_promotion_countable_dates") or []) if value)


def _tb15_tail_exact_model_proof(
    tb_tail: dict[str, Any],
    *,
    market: str,
    side: str,
    line_bucket: str,
    bookmaker_key: str,
) -> dict[str, Any]:
    if (
        market != "batter_total_bases"
        or side != "over"
        or line_bucket != "TB 1.5"
        or not tb_tail.get("accepted")
        or tb_tail.get("status") != "ready"
    ):
        return {"passes": False}
    pricing = tb_tail.get("true_pair_line_pricing") or {}
    rows = pricing.get("by_line_book_side") or []
    selected: dict[str, Any] | None = None
    for rec in rows:
        if (
            str(rec.get("line_key") or "") == "TB 1.5"
            and str(rec.get("bookmaker_key") or "").lower() == bookmaker_key
            and str(rec.get("side") or "").lower() == side
        ):
            selected = dict(rec)
            break
    if not selected:
        return {"passes": False, "reason": "tb15_tail_no_exact_line_book_side_row"}
    base_brier = _float(selected.get("base_brier"))
    direct_brier = _float(selected.get("direct_brier"))
    calibrated_brier = _float(selected.get("calibrated_brier"))
    brier_candidates = [value for value in (direct_brier, calibrated_brier) if value is not None]
    if not brier_candidates:
        return {"passes": False, "reason": "tb15_tail_missing_brier_metrics", "rows": int(selected.get("rows") or 0)}
    best_brier = min(brier_candidates)
    calibration_error = _float(selected.get("calibrated_calibration_error"))
    line_rows = int(selected.get("rows") or 0)
    passes = bool(
        line_rows >= 150
        and base_brier is not None
        and best_brier <= base_brier - 0.001
        and calibration_error is not None
        and calibration_error <= 0.05
    )
    return {
        "passes": passes,
        "source": "prop_tb_tail_repair_challenger",
        "reason": "tb15_tail_exact_line_brier_and_calibration_pass" if passes else "tb15_tail_exact_line_gate_failed",
        "rows": line_rows,
        "base_brier": base_brier,
        "direct_brier": direct_brier,
        "calibrated_brier": calibrated_brier,
        "best_brier": best_brier,
        "brier_gain": None if base_brier is None else base_brier - best_brier,
        "calibration_error": calibration_error,
    }


def evaluate(model_dir: Path = _MODEL_DIR) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    policy = _load(model_dir / "prop_bucket_reopen_policy.json")
    residual = _load(model_dir / "prop_market_residual_models.json")
    distribution = _load(model_dir / "prop_distribution_models.json")
    opportunity = _load(model_dir / "prop_prospective_opportunity_audit.json")
    projection_audit = _load(model_dir / "daily_forecast_projection_audit.json")
    checkpoint = _load(model_dir / "prop_five_date_checkpoint.json")
    bankroll_proof = _load(model_dir / "prop_player_game_bankroll_model_proof.json")
    exact_clv_priors = _load(model_dir / "prop_exact_bucket_clv_priors.json")
    k_under_repair = _load(model_dir / "prop_k_under_repair.json")
    tb_tail = _load(model_dir / "prop_tb_tail_repair_challenger.json")
    backdated_audit = _load(model_dir / "prop_backdated_slate_eligibility_audit.json")
    backdated_dates = _backdated_promotion_dates(backdated_audit)
    projection_gates = projection_audit.get("projection_gates") or {}
    projection_gates = {
        **projection_gates,
        **_checkpoint_projection_gates(checkpoint),
    }
    residual_buckets = {
        str(row.get("bucket")): row for row in residual.get("bucket_recommendations") or [] if row.get("bucket")
    }
    distribution_buckets = {
        str(row.get("bucket")): row for row in distribution.get("bucket_model_selection") or [] if row.get("bucket")
    }
    bankroll_proof_rows = {
        str(row.get("stat")): row for row in bankroll_proof.get("rows") or [] if row.get("stat")
    }
    exact_clv_buckets = exact_clv_priors.get("buckets") or {}
    k_under_gates = k_under_repair.get("gates") or {}
    clv_target = residual.get("clv_target") or {}
    clv_holdout = clv_target.get("holdout") or {}
    clv_classifier_ready = bool(
        clv_target.get("status") == "ready"
        and (clv_holdout.get("brier") is not None)
        and (clv_holdout.get("auc") is None or float(clv_holdout.get("auc")) >= 0.55)
    )
    magnitude_model = (residual.get("models") or {}).get("clv_magnitude") or {}
    clv_magnitude_ready = bool(magnitude_model.get("enabled"))
    prospective_dates = int(opportunity.get("unique_dates") or 0)
    rows: list[dict[str, Any]] = []
    for key, bucket in (policy.get("ladder_buckets") or {}).items():
        if str(bucket.get("line_surface") or "") != "common":
            continue
        residual_rec = residual_buckets.get(key) or {}
        distribution_rec = distribution_buckets.get(key) or {}
        blockers = list(bucket.get("bootstrap_micro_reasons") or bucket.get("model_reasons") or [])
        residual_decision = str(residual_rec.get("decision") or "missing")
        distribution_decision = str(distribution_rec.get("decision") or "missing")
        market = str(bucket.get("market") or "")
        side = str(bucket.get("side") or "")
        line_bucket = str(bucket.get("line_bucket") or "")
        bookmaker_key = str(bucket.get("bookmaker_key") or "").lower()
        projection_gate = projection_gates.get(market) or {}
        bankroll_proof_rec = bankroll_proof_rows.get(market) or {}
        tb15_tail_exact_proof = _tb15_tail_exact_model_proof(
            tb_tail,
            market=market,
            side=side,
            line_bucket=line_bucket,
            bookmaker_key=bookmaker_key,
        )
        projection_micro_allowed = bool(
            bankroll_proof_rec.get("projection_micro_allowed")
            or bankroll_proof_rec.get("projection_eligible")
            or projection_gate.get("eligible")
            or tb15_tail_exact_proof.get("passes")
        )
        bankroll_training_allowed = bool(bankroll_proof_rec.get("exact_line_bankroll_training_allowed"))
        stable_release_dates = int(projection_gate.get("graded_dates") or 0)
        stable_release_id = projection_gate.get("active_model_version")
        if not projection_gate.get("eligible"):
            blockers.append("underlying_projection_not_proven")
        if bankroll_proof_rows and not bankroll_training_allowed:
            blockers.append("player_game_bankroll_proof_not_passed")
            blockers.extend(str(value) for value in (bankroll_proof_rec.get("blockers") or []))
        if stable_release_dates < 5:
            blockers.append("stable_release_dates<5")
        roi = bucket.get("bootstrap_roi", bucket.get("holdout_roi"))
        clv_beat_rate = bucket.get("bootstrap_clv_beat_rate", bucket.get("holdout_clv_beat_rate"))
        avg_clv_price = bucket.get("bootstrap_avg_clv_price", bucket.get("holdout_avg_clv_price"))
        clean_dates = int(bucket.get("bootstrap_unique_clean_dates") or bucket.get("holdout_unique_clean_dates") or 0)
        backdated_clean_dates = int(
            bucket.get("bootstrap_unique_clean_dates") or bucket.get("holdout_unique_clean_dates") or 0
        )
        valid_close_coverage = bucket.get(
            "bootstrap_valid_close_coverage", bucket.get("holdout_valid_close_coverage")
        )
        if roi is None or float(roi) <= 0.0:
            blockers.append("prospective_roi<=0")
        if clv_beat_rate is None or float(clv_beat_rate) < 0.55:
            blockers.append("prospective_clv_beat_rate<0.55")
        if avg_clv_price is None or float(avg_clv_price) <= 0.0:
            blockers.append("prospective_avg_clv<=0")
        if clean_dates < 5:
            blockers.append("prospective_clean_dates<5")
        if valid_close_coverage is None or float(valid_close_coverage) < 0.90:
            blockers.append("prospective_valid_close_coverage<0.90")
        if int(distribution_rec.get("true_pair_rows") or 0) <= 0 or not distribution_rec.get("true_pair_only"):
            blockers.append("exact_bucket_true_pair_proof_missing")
        model_proof = bool(
            residual_decision == "use_market_residual"
            or distribution_decision in {
                "use_k_v3", "use_event_curve_side_line", "use_distribution",
                "use_distribution_market_blend", "keep_model_only",
            }
            or tb15_tail_exact_proof.get("passes")
        )
        if not model_proof:
            blockers.append("exact_bucket_model_proof_missing")
        if not clv_classifier_ready:
            blockers.append("clv_classifier_not_ready")
        # The magnitude challenger is additive evidence only. A failed holdout gate
        # disables it; it must never block a stronger CLV classifier by itself.
        if (
            str(bucket.get("bookmaker_key") or "").lower() == "fanduel"
            and str(bucket.get("market") or "").startswith("batter_")
            and residual_decision != "use_market_residual"
        ):
            blockers.append("fanduel_hitter_true_pair_proof_missing")
        blockers = sorted(set(str(value) for value in blockers if value))
        structural_micro = bool(bucket.get("bootstrap_micro_eligible") or bucket.get("ladder_tier") == "micro")
        exact_clv = exact_clv_buckets.get(key) or {}
        exact_clv_micro_confirmed = bool(exact_clv.get("micro_clv_confirmed"))
        exact_clv_rows = int(exact_clv.get("clv_rows") or 0)
        trial_clv_beat_rate = (
            _float(exact_clv.get("micro_clv_beat_rate"))
            if exact_clv_rows >= 30
            else _float(clv_beat_rate)
        )
        trial_avg_clv_price = (
            _float(exact_clv.get("micro_avg_clv_price"))
            if exact_clv_rows >= 30
            else _float(avg_clv_price)
        )
        trial_clv_source = "exact_bucket_clv_prior" if exact_clv_rows >= 30 else "bucket_bootstrap"
        k_under_gate_key = f"{line_bucket}|{bookmaker_key}"
        k_under_gate = k_under_gates.get(k_under_gate_key) or {}
        k_under_required = market == "pitcher_strikeouts" and side == "under"
        k_under_repair_allows = bool(k_under_gate.get("micro_allowed")) if k_under_required else True
        micro_trial_blockers: list[str] = []
        if str(bucket.get("line_surface") or "") != "common":
            micro_trial_blockers.append("not_common_line")
        if not projection_micro_allowed:
            micro_trial_blockers.append("player_game_projection_not_proven")
            micro_trial_blockers.extend(str(value) for value in (bankroll_proof_rec.get("projection_blockers") or []))
        if not exact_clv_micro_confirmed:
            micro_trial_blockers.append("exact_bucket_clv_prior_not_confirming")
            micro_trial_blockers.extend(str(value) for value in (exact_clv.get("micro_blockers") or []))
        if bookmaker_key == "fanduel" and market.startswith("batter_") and not exact_clv_micro_confirmed:
            micro_trial_blockers.append("fanduel_synthetic_hitter_evidence_display_only")
        if k_under_required and not k_under_repair_allows:
            micro_trial_blockers.append("k_under_repair_gate_failed")
            micro_trial_blockers.extend(str(value) for value in (k_under_gate.get("blockers") or []))
        if roi is None or float(roi) <= 0.0:
            micro_trial_blockers.append("bootstrap_roi<=0")
        if trial_avg_clv_price is None or trial_avg_clv_price <= 0.0:
            micro_trial_blockers.append(f"{trial_clv_source}_avg_clv<=0")
        if trial_clv_beat_rate is None or trial_clv_beat_rate < 0.54:
            micro_trial_blockers.append(f"{trial_clv_source}_clv_beat_rate<0.54")
        if valid_close_coverage is None or float(valid_close_coverage) < 0.90:
            micro_trial_blockers.append("bootstrap_valid_close_coverage<0.90")
        stale_close_rate = bucket.get("bootstrap_stale_close_rate", bucket.get("holdout_stale_close_rate"))
        if stale_close_rate is not None and float(stale_close_rate) > 0.02:
            micro_trial_blockers.append("bootstrap_stale_close_rate>0.02")
        if int(bucket.get("bootstrap_rows") or bucket.get("holdout_rows") or 0) < 75:
            micro_trial_blockers.append("bootstrap_rows<75")
        if int(bucket.get("bootstrap_clv_price_rows") or bucket.get("holdout_clv_price_rows") or 0) < 30:
            micro_trial_blockers.append("bootstrap_clv_rows<30")
        if clean_dates < 4:
            micro_trial_blockers.append("bootstrap_clean_dates<4")
        micro_trial_blockers = sorted(set(str(value) for value in micro_trial_blockers if value))
        rows.append({
            "bucket": key,
            "market": bucket.get("market"),
            "side": bucket.get("side"),
            "bookmaker_key": bucket.get("bookmaker_key"),
            "ladder_tier": bucket.get("ladder_tier"),
            "structural_micro_eligible": structural_micro,
            "micro_trial_ready": bool(structural_micro or not micro_trial_blockers),
            "micro_trial_blockers": micro_trial_blockers,
            "micro_ready": bool(structural_micro and not blockers),
            "rows": int(bucket.get("bootstrap_rows") or bucket.get("holdout_rows") or 0),
            "roi": roi,
            "clv_rows": int(bucket.get("bootstrap_clv_price_rows") or bucket.get("holdout_clv_price_rows") or 0),
            "clv_beat_rate": clv_beat_rate,
            "avg_clv_price": avg_clv_price,
            "calibration_error": bucket.get("bootstrap_calibration_error", bucket.get("holdout_calibration_error")),
            "clean_dates": clean_dates,
            "promotion_clean_dates": backdated_clean_dates,
            "backdated_promotion_slate_count": len(backdated_dates),
            "valid_close_coverage": valid_close_coverage,
            "true_pair_rows": int(distribution_rec.get("true_pair_rows") or 0),
            "residual_decision": residual_decision,
            "distribution_decision": distribution_decision,
            "projection_gate": projection_gate,
            "player_game_bankroll_proof": bankroll_proof_rec,
            "projection_micro_allowed": projection_micro_allowed,
            "player_game_bankroll_training_allowed": bankroll_training_allowed,
            "exact_bucket_clv_prior": exact_clv,
            "exact_bucket_clv_micro_confirmed": exact_clv_micro_confirmed,
            "trial_clv_source": trial_clv_source,
            "trial_clv_beat_rate": trial_clv_beat_rate,
            "trial_avg_clv_price": trial_avg_clv_price,
            "k_under_repair_gate": k_under_gate,
            "k_under_repair_gate_key": k_under_gate_key if k_under_required else None,
            "k_under_repair_allows": k_under_repair_allows,
            "tb15_tail_exact_model_proof": tb15_tail_exact_proof,
            "stable_release_id": stable_release_id,
            "stable_release_dates": stable_release_dates,
            "blockers": blockers,
        })
    focus_markets = {"batter_total_bases", "pitcher_strikeouts"}
    rows.sort(key=lambda row: (
        str(row.get("market") or "") not in focus_markets,
        not row["micro_ready"],
        not row["micro_trial_ready"],
        len(row["blockers"]),
        -row["rows"],
    ))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "eligibility_start_date": policy.get("eligibility_start_date"),
        "status": "ready" if policy else "missing_policy",
        "clv_selected_variant": clv_target.get("selected_variant"),
        "clv_classifier_ready": clv_classifier_ready,
        "clv_magnitude_ready": clv_magnitude_ready,
        "player_game_bankroll_proof_status": bankroll_proof.get("status"),
        "player_game_projection_allowed_stats": bankroll_proof.get("projection_allowed_stats") or [],
        "player_game_bankroll_allowed_stats": bankroll_proof.get("allowed_stats") or [],
        "prospective_opportunity_dates": prospective_dates,
        "backdated_prop_promotion_slate_count": len(backdated_dates),
        "backdated_prop_promotion_dates": backdated_dates,
        "evaluated_common_buckets": len(rows),
        "micro_ready_count": sum(1 for row in rows if row["micro_ready"]),
        "micro_trial_ready_count": sum(1 for row in rows if row["micro_trial_ready"]),
        "tb_k_micro_ready_count": sum(
            1 for row in rows
            if row["micro_ready"] and str(row.get("market") or "") in focus_markets
        ),
        "tb_k_micro_trial_ready_count": sum(
            1 for row in rows
            if row["micro_trial_ready"] and str(row.get("market") or "") in focus_markets
        ),
        "buckets": rows,
    }
    atomic_write_json(model_dir / "prop_micro_promotion_evaluation.json", payload)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORT_DIR / "mlb_prop_micro_promotion_evaluation_latest.md"
    lines = [
        "# MLB Prop Micro Promotion Evaluation",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Eligibility start: {payload.get('eligibility_start_date')}",
        f"Micro-ready exact buckets: {payload['micro_ready_count']}",
        f"$1 micro-trial exact buckets: {payload['micro_trial_ready_count']}",
        f"TB/K micro-ready exact buckets: {payload['tb_k_micro_ready_count']}",
        f"TB/K micro-trial exact buckets: {payload['tb_k_micro_trial_ready_count']}",
        f"CLV classifier: {payload.get('clv_selected_variant') or '-'} (ready={clv_classifier_ready})",
        f"Expected CLV magnitude model ready: {clv_magnitude_ready}",
        f"Player-game projection proof allowed stats: {', '.join(payload.get('player_game_projection_allowed_stats') or []) or '-'}",
        f"Player-game bankroll proof allowed stats: {', '.join(payload.get('player_game_bankroll_allowed_stats') or []) or '-'}",
        f"Prospective opportunity dates: {prospective_dates}",
        f"Backdated prop-promotion slates: {len(backdated_dates)} ({', '.join(backdated_dates) or '-'})",
        "",
        "| Bucket | Release | Release Dates | Projection Proof | Bankroll Proof | Rows | ROI | CLV Rows | Trial CLV | Trial Avg CLV | CLV Source | Cal Err | Promotion Clean Dates | Micro Trial | Ready | Blockers |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows[:40]:
        lines.append(
            f"| {row['bucket']} | {row.get('stable_release_id') or '-'} | {row['stable_release_dates']} | "
            f"{bool(row.get('projection_micro_allowed'))} | "
            f"{bool(row.get('player_game_bankroll_training_allowed'))} | "
            f"{row['rows']} | {_fmt(row.get('roi'))} | {row['clv_rows']} | "
            f"{_fmt(row.get('trial_clv_beat_rate'))} | {_fmt(row.get('trial_avg_clv_price'))} | "
            f"{row.get('trial_clv_source') or '-'} | "
            f"{_fmt(row.get('calibration_error'))} | {row['promotion_clean_dates']} | "
            f"{row['micro_trial_ready']} | {row['micro_ready']} | "
            f"{', '.join(row['blockers']) or '-'} |"
        )
    lines.extend([
        "",
        "## $1 Micro Trial Blockers",
        "",
        "| Bucket | Trial Ready | CLV Prior | K Repair | Trial Blockers |",
        "|---|---|---|---|---|",
    ])
    for row in rows[:40]:
        clv_prior = row.get("exact_bucket_clv_prior") or {}
        lines.append(
            f"| {row['bucket']} | {row['micro_trial_ready']} | "
            f"{bool(row.get('exact_bucket_clv_micro_confirmed'))} "
            f"({_fmt(clv_prior.get('micro_clv_beat_rate'))}) | "
            f"{row.get('k_under_repair_allows')} | "
            f"{', '.join(row.get('micro_trial_blockers') or []) or '-'} |"
        )
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    payload["report_path"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact MLB prop buckets for $1 micro promotion")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    args = parser.parse_args()
    payload = evaluate(Path(args.model_dir))
    print(json.dumps({
        "status": payload.get("status"),
        "micro_ready_count": payload.get("micro_ready_count", 0),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
