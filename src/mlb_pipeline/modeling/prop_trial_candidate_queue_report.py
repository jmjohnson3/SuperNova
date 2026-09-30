"""Queue exact prop buckets that are closest to earning $1 micro trials.

This report is intentionally about expansion, not suppression.  It mines the
micro-promotion artifact, ranks near-miss exact buckets by their repair gaps,
and keeps one-sided/synthetic FanDuel hitter markets out of trial expansion.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_TRIAL_MIN_ROWS = 75
_TRIAL_MIN_CLV_ROWS = 30
_TRIAL_MIN_CLEAN_DATES = 4
_TRIAL_MIN_CLV_BEAT = 0.54
_TRIAL_MIN_ROI = 0.0
_TRIAL_MIN_AVG_CLV = 0.0
_TRIAL_MIN_CLOSE_COVERAGE = 0.90
_TRIAL_MAX_STALE_CLOSE_RATE = 0.02
_TRIAL_MAX_ABS_CAL_ERROR = 0.08


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _pct(value: Any) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.1f}%"


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _parts(bucket: Any) -> dict[str, str]:
    parts = str(bucket or "").split("|")
    parts += [""] * max(0, 6 - len(parts))
    return {
        "market": parts[0],
        "side": parts[1],
        "surface": parts[2],
        "line_bucket": parts[3],
        "price_bucket": parts[4],
        "book": parts[5].lower(),
    }


def _is_fanduel_hitter(row: dict[str, Any]) -> bool:
    parts = _parts(row.get("bucket"))
    return parts["book"] == "fanduel" and parts["market"].startswith("batter_")


def _focus(row: dict[str, Any]) -> str:
    parts = _parts(row.get("bucket"))
    if (
        parts["book"] == "draftkings"
        and parts["market"] == "pitcher_strikeouts"
        and parts["side"] == "under"
        and parts["line_bucket"] == "K <4.5"
        and parts["price_bucket"].startswith("plus_")
    ):
        return "dk_k_under_low_plus_money"
    if (
        parts["book"] == "draftkings"
        and parts["market"] == "pitcher_strikeouts"
        and parts["side"] == "under"
        and parts["line_bucket"] == "K 4.5-6.0"
    ):
        return "dk_k_under_4_5_6_0_repair"
    if (
        parts["book"] == "draftkings"
        and parts["market"] == "batter_total_bases"
        and parts["side"] == "over"
        and parts["line_bucket"] == "TB 1.5"
        and parts["price_bucket"].startswith("plus_")
    ):
        return "dk_tb15_over_plus_money"
    if _is_fanduel_hitter(row):
        return "fanduel_hitter_display_only"
    if parts["book"] == "draftkings":
        return "draftkings_other"
    return "other"


def _gap(value: Any, threshold: float, *, higher_is_better: bool = True) -> float | None:
    numeric = _float(value)
    if numeric is None:
        return threshold if higher_is_better else 0.0
    if higher_is_better:
        return max(0.0, threshold - numeric)
    return max(0.0, numeric - threshold)


def _bool_missing_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return True
    return str(value).strip().lower() not in {"1", "true", "yes", "y", "on"}


def _metric_gaps(row: dict[str, Any]) -> dict[str, Any]:
    exact = row.get("exact_bucket_clv_prior") or {}
    rows = _int(row.get("rows"))
    clv_rows = _int(row.get("clv_rows"))
    clean_dates = _int(row.get("clean_dates") or row.get("promotion_clean_dates"))
    roi = _float(row.get("roi"))
    clv_beat = _float(row.get("trial_clv_beat_rate") or row.get("clv_beat_rate"))
    avg_clv = _float(row.get("trial_avg_clv_price") or row.get("avg_clv_price"))
    cal = _float(row.get("calibration_error"))
    close_coverage = _float(row.get("valid_close_coverage"))
    exact_close = _float(exact.get("valid_close_coverage"))
    stale_rate = _float(exact.get("stale_close_rate"))
    exact_true_pair_rows = _int(exact.get("true_pair_rows"))
    exact_clv_rows = _int(exact.get("clv_rows"))
    close_denominator = exact_true_pair_rows or rows
    close_valid = round((exact_close or close_coverage or 0.0) * close_denominator)
    return {
        "rows_needed": max(0, _TRIAL_MIN_ROWS - rows),
        "clv_rows_needed": max(0, _TRIAL_MIN_CLV_ROWS - clv_rows),
        "clean_dates_needed": max(0, _TRIAL_MIN_CLEAN_DATES - clean_dates),
        "roi_gap": _gap(roi, _TRIAL_MIN_ROI),
        "clv_beat_gap": _gap(clv_beat, _TRIAL_MIN_CLV_BEAT),
        "avg_clv_gap": _gap(avg_clv, _TRIAL_MIN_AVG_CLV),
        "calibration_gap": (
            max(0.0, abs(cal) - _TRIAL_MAX_ABS_CAL_ERROR)
            if cal is not None
            else None
        ),
        "policy_close_coverage_gap": _gap(close_coverage, _TRIAL_MIN_CLOSE_COVERAGE),
        "exact_close_coverage_gap": _gap(exact_close, _TRIAL_MIN_CLOSE_COVERAGE),
        "valid_closes_needed_for_90": max(0, math.ceil(_TRIAL_MIN_CLOSE_COVERAGE * close_denominator) - close_valid)
        if close_denominator else None,
        "stale_close_gap": _gap(stale_rate, _TRIAL_MAX_STALE_CLOSE_RATE, higher_is_better=False),
        "exact_true_pair_rows": exact_true_pair_rows,
        "exact_clv_rows": exact_clv_rows,
    }


_MODEL_PROOF_ACCEPT_DECISIONS = {
    "use_k_v3",
    "use_event_curve_side_line",
    "use_distribution",
    "use_distribution_market_blend",
    "keep_model_only",
}


def _proof_status(row: dict[str, Any], distribution_rec: dict[str, Any] | None = None) -> dict[str, Any]:
    blockers = set(str(value) for value in (row.get("blockers") or []))
    trial_blockers = set(str(value) for value in (row.get("micro_trial_blockers") or []))
    projection_gate = row.get("projection_gate") or {}
    distribution_rec = distribution_rec or {}
    dist_present = bool(distribution_rec)
    dist_rows = _int(distribution_rec.get("rows"))
    dist_true_pair_rows = _int(distribution_rec.get("true_pair_rows"))
    dist_true_pair_only = bool(distribution_rec.get("true_pair_only"))
    dist_decision = str(distribution_rec.get("decision") or "")
    model_missing = "exact_bucket_model_proof_missing" in blockers
    true_pair_missing = "exact_bucket_true_pair_proof_missing" in blockers
    return {
        "projection_proven": bool(row.get("projection_micro_allowed")) and not trial_blockers.intersection({
            "player_game_projection_not_proven",
        }),
        "projection_source": projection_gate.get("source") or (row.get("player_game_bankroll_proof") or {}).get("active_model_family"),
        "projection_blockers": (row.get("player_game_bankroll_proof") or {}).get("projection_blockers") or [],
        "exact_model_proof_missing": model_missing,
        "exact_model_proof_artifact_missing": model_missing and not dist_present,
        "exact_model_proof_insufficient_sample": model_missing and dist_present and (
            dist_rows < _TRIAL_MIN_ROWS or dist_decision == "no_bet_sample"
        ),
        "exact_model_proof_performance_missing": model_missing and dist_present
        and dist_rows >= _TRIAL_MIN_ROWS
        and dist_decision not in _MODEL_PROOF_ACCEPT_DECISIONS,
        "exact_true_pair_proof_missing": true_pair_missing,
        "exact_true_pair_proof_artifact_missing": true_pair_missing and not dist_present,
        "exact_true_pair_proof_insufficient_sample": true_pair_missing and dist_present and (
            dist_true_pair_rows <= 0 or not dist_true_pair_only
        ),
        "clv_prior_confirmed": bool(row.get("exact_bucket_clv_micro_confirmed")),
        "k_under_repair_allows": row.get("k_under_repair_allows"),
        "k_under_repair_blockers": (row.get("k_under_repair_gate") or {}).get("blockers") or [],
        "distribution_proof_present": dist_present,
        "distribution_rows": dist_rows,
        "distribution_true_pair_rows": dist_true_pair_rows,
        "distribution_decision": dist_decision or "-",
        "distribution_best_variant": distribution_rec.get("best_variant"),
    }


def _failure_metrics(row: dict[str, Any], gaps: dict[str, Any], proof: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if gaps["rows_needed"] > 0:
        failures.append("sample_size")
    if gaps["clv_rows_needed"] > 0:
        failures.append("clv_sample")
    if gaps["clean_dates_needed"] > 0:
        failures.append("clean_dates")
    if (gaps.get("roi_gap") or 0.0) > 0.0:
        failures.append("roi")
    if (gaps.get("clv_beat_gap") or 0.0) > 0.0 or (gaps.get("avg_clv_gap") or 0.0) > 0.0:
        failures.append("clv_confirmation")
    if (gaps.get("policy_close_coverage_gap") or 0.0) > 0.0 or (gaps.get("exact_close_coverage_gap") or 0.0) > 0.0:
        failures.append("close_coverage")
    if (gaps.get("stale_close_gap") or 0.0) > 0.0:
        failures.append("stale_close")
    if (gaps.get("calibration_gap") or 0.0) > 0.0:
        failures.append("calibration")
    if proof["exact_model_proof_missing"]:
        if proof.get("exact_model_proof_artifact_missing"):
            failures.append("model_proof_artifact")
        elif proof.get("exact_model_proof_insufficient_sample"):
            failures.append("model_proof_sample")
        else:
            failures.append("model_proof_performance")
    if proof["exact_true_pair_proof_missing"]:
        if proof.get("exact_true_pair_proof_artifact_missing"):
            failures.append("true_pair_proof_artifact")
        elif proof.get("exact_true_pair_proof_insufficient_sample"):
            failures.append("true_pair_proof_sample")
        else:
            failures.append("true_pair_proof")
    if not proof["projection_proven"]:
        failures.append("projection_proof")
    if _is_fanduel_hitter(row):
        failures.append("fanduel_display_only")
    return sorted(set(failures))


def _repair_class(row: dict[str, Any], failures: list[str]) -> str:
    if row.get("micro_trial_ready") and not _is_fanduel_hitter(row):
        return "trial_ready"
    if "fanduel_display_only" in failures:
        return "display_only_fanduel_one_sided"
    if failures and set(failures).issubset({"close_coverage", "stale_close"}):
        return "close_capture_only"
    if failures and set(failures).issubset({"model_proof_artifact", "true_pair_proof_artifact"}):
        return "proof_refresh_only"
    if failures and set(failures).issubset({"model_proof_sample", "true_pair_proof_sample"}):
        return "proof_sample_only"
    if failures and len(failures) <= 2:
        return "near_miss_1_2_gates"
    if _focus(row) == "dk_k_under_4_5_6_0_repair":
        return "k_rate_leash_repair"
    if _focus(row) == "dk_tb15_over_plus_money":
        return "tb15_close_and_calibration_repair"
    return "multi_gate_repair"


def _priority(row: dict[str, Any], gaps: dict[str, Any], failures: list[str]) -> float:
    focus_bonus = {
        "dk_k_under_low_plus_money": -55.0,
        "dk_tb15_over_plus_money": -40.0,
        "dk_k_under_4_5_6_0_repair": -28.0,
        "draftkings_other": -5.0,
        "other": 8.0,
        "fanduel_hitter_display_only": 999.0,
    }.get(_focus(row), 20.0)
    return (
        focus_bonus
        + len(failures) * 16.0
        + gaps["rows_needed"] / 8.0
        + gaps["clv_rows_needed"] / 3.0
        + gaps["clean_dates_needed"] * 8.0
        + (gaps.get("roi_gap") or 0.0) * 180.0
        + (gaps.get("clv_beat_gap") or 0.0) * 160.0
        + (gaps.get("avg_clv_gap") or 0.0) * 2.0
        + (gaps.get("policy_close_coverage_gap") or 0.0) * 90.0
        + (gaps.get("exact_close_coverage_gap") or 0.0) * 90.0
        + (gaps.get("calibration_gap") or 0.0) * 120.0
        + (gaps.get("stale_close_gap") or 0.0) * 80.0
    )


def _next_action(row: dict[str, Any], failures: list[str]) -> str:
    focus = _focus(row)
    if row.get("micro_trial_ready") and not _is_fanduel_hitter(row):
        return "Queue for $1 micro only when a live offer has positive EV and passes drift guard."
    if focus == "fanduel_hitter_display_only":
        return "Keep in research/display; do not trial unless true same-book opposite-side evidence exists."
    if focus == "dk_k_under_low_plus_money":
        return "Keep as top trial target; repair only the listed gaps and let drift guard decide live plays."
    if focus == "dk_k_under_4_5_6_0_repair":
        return "Repair BF/leash/K-rate before opening; CLV alone has not produced ROI."
    if focus == "dk_tb15_over_plus_money":
        return "Prioritize DK TB 1.5 close coverage and line calibration; require true-paired proof."
    if set(failures).issubset({"model_proof_artifact", "true_pair_proof_artifact"}):
        return "Run targeted exact-bucket proof refresh for this market."
    if "model_proof_sample" in failures or "true_pair_proof_sample" in failures:
        return "Collect more true-paired out-of-fold rows; targeted refresh found too little exact-bucket proof."
    if "model_proof_performance" in failures:
        return "Repair projection/model selection; exact proof exists but did not beat the alternatives."
    if "close_coverage" in failures or "stale_close" in failures:
        return "Target close capture and CLV-label refresh for this exact bucket."
    if "roi" in failures or "calibration" in failures:
        return "Repair projection/line-pricing before trial expansion."
    return "Keep collecting locked rows; revisit after next clean slate."


def _enrich(row: dict[str, Any], distribution_rec: dict[str, Any] | None = None) -> dict[str, Any]:
    gaps = _metric_gaps(row)
    proof = _proof_status(row, distribution_rec)
    failures = _failure_metrics(row, gaps, proof)
    parts = _parts(row.get("bucket"))
    out = dict(row)
    out.update(parts)
    out["focus"] = _focus(row)
    out["metric_gaps"] = gaps
    out["proof_status"] = proof
    out["failed_metrics"] = failures
    out["failed_metric_count"] = len(failures)
    out["repair_class"] = _repair_class(row, failures)
    out["priority_score"] = _priority(row, gaps, failures)
    out["next_action"] = _next_action(row, failures)
    out["trial_allowed_by_market_structure"] = not _is_fanduel_hitter(row)
    return out


def _proof_refresh_markets(rows: list[dict[str, Any]]) -> list[str]:
    markets: set[str] = set()
    for row in rows:
        if row.get("focus") == "fanduel_hitter_display_only":
            continue
        failures = set(row.get("failed_metrics") or [])
        if failures.intersection({"model_proof_artifact", "true_pair_proof_artifact"}):
            market = str(row.get("market") or "")
            if market:
                markets.add(market)
    # K/TB are the near-term expansion markets; keep them first.
    priority = ["pitcher_strikeouts", "batter_total_bases", "batter_hits", "batter_home_runs"]
    return [market for market in priority if market in markets] + sorted(markets - set(priority))


def build(model_dir: Path = _MODEL_DIR) -> dict[str, Any]:
    micro = _load(model_dir / "prop_micro_promotion_evaluation.json")
    distribution = _load(model_dir / "prop_distribution_models.json")
    distribution_buckets = {
        str(row.get("bucket")): row
        for row in (distribution.get("bucket_model_selection") or [])
        if row.get("bucket")
    }
    raw_rows = list(micro.get("buckets") or [])
    rows = [
        _enrich(row, distribution_buckets.get(str(row.get("bucket"))))
        for row in raw_rows
    ]
    rows.sort(key=lambda row: (row["priority_score"], -_int(row.get("rows"))))

    trial_ready = [
        row for row in rows
        if row.get("micro_trial_ready") and row.get("trial_allowed_by_market_structure")
    ]
    near_miss = [
        row for row in rows
        if not row.get("micro_trial_ready")
        and row.get("trial_allowed_by_market_structure")
        and row.get("failed_metric_count", 99) <= 2
    ]
    target_signal = [
        row for row in rows
        if row.get("focus") in {
            "dk_k_under_low_plus_money",
            "dk_k_under_4_5_6_0_repair",
            "dk_tb15_over_plus_money",
        }
    ]
    close_queue = [
        row for row in rows
        if row.get("trial_allowed_by_market_structure")
        and set(row.get("failed_metrics") or []).intersection({"close_coverage", "stale_close"})
    ]
    proof_repair_queue = [
        row for row in rows
        if row.get("trial_allowed_by_market_structure")
        and any(str(metric).endswith("_proof") or "_proof_" in str(metric) for metric in (row.get("failed_metrics") or []))
    ]
    proof_refresh_queue = [
        row for row in proof_repair_queue
        if set(row.get("failed_metrics") or []).intersection({"model_proof_artifact", "true_pair_proof_artifact"})
    ]
    proof_sample_queue = [
        row for row in proof_repair_queue
        if set(row.get("failed_metrics") or []).intersection({"model_proof_sample", "true_pair_proof_sample"})
    ]
    fanduel_display = [row for row in rows if row.get("focus") == "fanduel_hitter_display_only"]
    refresh_markets = _proof_refresh_markets(proof_refresh_queue)
    refresh_command = (
        ".\\.venv\\Scripts\\python.exe -m mlb_pipeline.modeling.refresh_prop_exact_bucket_proof "
        f"--markets {','.join(refresh_markets)} --lookback-days 365 --market-timeout-seconds 900"
        if refresh_markets else None
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if micro else "missing_micro_promotion_evaluation",
        "source": "prop_micro_promotion_evaluation.json",
        "thresholds": {
            "trial_min_rows": _TRIAL_MIN_ROWS,
            "trial_min_clv_rows": _TRIAL_MIN_CLV_ROWS,
            "trial_min_clean_dates": _TRIAL_MIN_CLEAN_DATES,
            "trial_min_clv_beat": _TRIAL_MIN_CLV_BEAT,
            "trial_min_close_coverage": _TRIAL_MIN_CLOSE_COVERAGE,
            "trial_max_stale_close_rate": _TRIAL_MAX_STALE_CLOSE_RATE,
            "trial_max_abs_calibration_error": _TRIAL_MAX_ABS_CAL_ERROR,
        },
        "evaluated_buckets": len(rows),
        "trial_ready_count": len(trial_ready),
        "near_miss_1_2_gate_count": len(near_miss),
        "target_signal_count": len(target_signal),
        "close_repair_count": len(close_queue),
        "proof_repair_count": len(proof_repair_queue),
        "proof_refresh_count": len(proof_refresh_queue),
        "proof_sample_count": len(proof_sample_queue),
        "fanduel_display_only_count": len(fanduel_display),
        "repair_class_counts": dict(Counter(row["repair_class"] for row in rows).most_common()),
        "failed_metric_counts": dict(Counter(metric for row in rows for metric in row["failed_metrics"]).most_common()),
        "proof_refresh_markets": refresh_markets,
        "proof_refresh_command": refresh_command,
        "trial_ready": trial_ready[:25],
        "near_miss_1_2_gates": near_miss[:50],
        "target_signal_buckets": target_signal[:60],
        "close_repair_queue": close_queue[:50],
        "proof_repair_queue": proof_repair_queue[:50],
        "proof_refresh_queue": proof_refresh_queue[:50],
        "proof_sample_queue": proof_sample_queue[:50],
        "fanduel_display_only": fanduel_display[:40],
        "all_ranked": rows[:120],
    }
    _write_outputs(model_dir, payload)
    return payload


def _bucket_line(row: dict[str, Any]) -> str:
    gaps = row.get("metric_gaps") or {}
    proof = row.get("proof_status") or {}
    return (
        f"| `{row.get('bucket')}` | {row.get('focus')} | {row.get('repair_class')} | "
        f"{row.get('failed_metric_count')} | {', '.join(row.get('failed_metrics') or []) or '-'} | "
        f"{_int(row.get('rows'))} | {gaps.get('rows_needed')} | "
        f"{_int(row.get('clv_rows'))} | {gaps.get('clv_rows_needed')} | "
        f"{_pct(row.get('roi'))} | {_pct(row.get('trial_clv_beat_rate') or row.get('clv_beat_rate'))} | "
        f"{_fmt(row.get('trial_avg_clv_price') or row.get('avg_clv_price'))} | "
        f"{_pct(row.get('valid_close_coverage'))} | {_pct((row.get('exact_bucket_clv_prior') or {}).get('valid_close_coverage'))} | "
        f"{gaps.get('valid_closes_needed_for_90') if gaps.get('valid_closes_needed_for_90') is not None else '-'} | "
        f"{proof.get('projection_proven')} | {not proof.get('exact_model_proof_missing')} | "
        f"{not proof.get('exact_true_pair_proof_missing')} | {_fmt(row.get('priority_score'), 1)} | "
        f"{row.get('next_action')} |"
    )


def _section(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| Bucket | Focus | Repair Class | Failed | Failed Metrics | Rows | Rows Need | CLV Rows | CLV Need | ROI | CLV Beat | Avg CLV | Policy Close | Exact Close | Valid Close Need | Projection | Model Proof | True Pair | Score | Next Action |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|",
    ])
    if not rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
        return
    for row in rows:
        lines.append(_bucket_line(row))


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Trial Candidate Queue",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Purpose: find exact buckets that can earn more `$1 micro_projection` trials without broadening bankroll gates.",
        "",
        "Note: the micro trial queue is not bankroll approval. Rows may still show stricter proof/close gaps that must be repaired before starter or bankroll promotion.",
        "",
        f"- Evaluated buckets: {payload.get('evaluated_buckets', 0)}",
        f"- `$1 micro` trial-queued exact buckets: {payload.get('trial_ready_count', 0)}",
        f"- Near misses missing 1-2 metrics: {payload.get('near_miss_1_2_gate_count', 0)}",
        f"- Close repair queue: {payload.get('close_repair_count', 0)}",
        f"- Exact proof repair queue: {payload.get('proof_repair_count', 0)}",
        f"- Proof refresh queue: {payload.get('proof_refresh_count', 0)}",
        f"- Proof sample queue: {payload.get('proof_sample_count', 0)}",
        f"- FanDuel hitter display-only buckets: {payload.get('fanduel_display_only_count', 0)}",
        "",
        "## Proof Refresh",
        "",
        f"- Markets: {', '.join(payload.get('proof_refresh_markets') or []) or '-'}",
        f"- Command: `{payload.get('proof_refresh_command') or '-'}`",
        "",
        "## Failed Metric Counts",
        "",
        "| Metric | Buckets |",
        "|---|---:|",
    ]
    for metric, count in (payload.get("failed_metric_counts") or {}).items():
        lines.append(f"| {metric} | {count} |")
    _section(lines, "$1 Micro Trial Queue", payload.get("trial_ready") or [])
    _section(lines, "Near Misses: 1-2 Gates", payload.get("near_miss_1_2_gates") or [])
    _section(lines, "Target Buckets With Real Signal", payload.get("target_signal_buckets") or [])
    _section(lines, "Close Coverage Repair Queue", payload.get("close_repair_queue") or [])
    _section(lines, "Exact Proof Repair Queue", payload.get("proof_repair_queue") or [])
    _section(lines, "Proof Artifact Refresh Queue", payload.get("proof_refresh_queue") or [])
    _section(lines, "Proof Sample Queue", payload.get("proof_sample_queue") or [])
    _section(lines, "FanDuel Hitter Display Only", payload.get("fanduel_display_only") or [])
    return "\n".join(lines) + "\n"


def _write_outputs(model_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / "prop_trial_candidate_queue.json"
    report_path = _REPORT_DIR / "mlb_prop_trial_candidate_queue_latest.md"
    atomic_write_json(json_path, payload, default=str)
    atomic_write_text(report_path, _render(payload))
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(report_path)
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop exact-bucket $1 trial candidate queue")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    args = parser.parse_args()
    payload = build(Path(args.model_dir))
    print(json.dumps({
        "status": payload.get("status"),
        "trial_ready_count": payload.get("trial_ready_count"),
        "near_miss_1_2_gate_count": payload.get("near_miss_1_2_gate_count"),
        "proof_refresh_markets": payload.get("proof_refresh_markets"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
