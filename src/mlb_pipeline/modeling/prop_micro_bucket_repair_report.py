"""Rank exact prop buckets by how fixable they are for $1 micro trials.

This is not a promotion script.  It mines the promotion and CLV artifacts for
near-miss exact buckets, then points the next repair work at close capture,
model proof refresh, K-rate repair, TB 1.5 calibration, or FanDuel one-sided
market evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _pct(value: Any) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.1f}%"


def _parts(bucket: str) -> dict[str, str]:
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


def _focus(row: dict[str, Any]) -> str:
    parts = _parts(str(row.get("bucket") or ""))
    if (
        parts["book"] == "draftkings"
        and parts["market"] == "pitcher_strikeouts"
        and parts["side"] == "under"
        and parts["line_bucket"] in {"K <4.5", "K 4.5-6.0"}
    ):
        if parts["price_bucket"] == "plus_100_149":
            return "dk_k_under_plus_money"
        return "dk_k_under_fair_lay"
    if (
        parts["book"] == "draftkings"
        and parts["market"] == "batter_total_bases"
        and parts["side"] == "over"
        and parts["line_bucket"] == "TB 1.5"
    ):
        return "dk_tb15_over"
    if parts["book"] == "fanduel" and parts["market"].startswith("batter_"):
        return "fanduel_hitter_display_only"
    return "other"


def _has_any(blockers: set[str], needles: tuple[str, ...]) -> bool:
    return any(any(blocker.startswith(needle) or needle in blocker for needle in needles) for blocker in blockers)


def _repair_class(row: dict[str, Any]) -> str:
    if row.get("micro_trial_ready"):
        return "trial_ready"
    blockers = set(str(value) for value in row.get("micro_trial_blockers") or [])
    all_blockers = blockers | set(str(value) for value in row.get("blockers") or [])
    focus = _focus(row)
    if focus == "fanduel_hitter_display_only":
        return "display_only_fanduel_synthetic_or_one_sided"
    if _has_any(blockers, ("bootstrap_valid_close_coverage", "stale_close_rate")):
        return "close_capture_repair"
    if _has_any(blockers, ("exact_bucket_clv_prior", "clv_beat", "avg_clv")):
        return "clv_confirmation_repair"
    if _has_any(blockers, ("bootstrap_rows", "bootstrap_clv_rows", "bootstrap_clean_dates")):
        return "needs_more_samples"
    if _has_any(blockers, ("bootstrap_roi",)):
        return "projection_or_line_pricing_repair"
    if _has_any(blockers, ("k_under_repair",)):
        return "k_rate_leash_repair"
    if _has_any(all_blockers, ("exact_bucket_model_proof_missing", "exact_bucket_true_pair_proof_missing")):
        return "proof_artifact_refresh"
    if _has_any(blockers, ("player_game_projection",)):
        return "player_game_projection_repair"
    return "mixed_or_low_priority"


def _next_action(row: dict[str, Any]) -> str:
    repair = row.get("repair_class")
    focus = row.get("focus")
    if repair == "trial_ready":
        return "Allow $1 micro only when a live offer has positive EV and passes drift guard."
    if repair == "close_capture_repair":
        return "Rebuild recent CLV labels and inspect close-capture gaps for this exact bucket."
    if repair == "clv_confirmation_repair":
        return "Collect/repair true-paired valid closes; do not open until CLV beat and avg CLV confirm."
    if repair == "needs_more_samples":
        return "Keep locking this exact bucket until rows, CLV rows, and clean dates clear the trial floor."
    if repair == "proof_artifact_refresh":
        return "Refresh distribution/residual proof for this market; split by market if timeout risk appears."
    if repair == "display_only_fanduel_synthetic_or_one_sided":
        return "Keep display/research only unless true FanDuel opposite-side prices are extracted."
    if repair == "k_rate_leash_repair" or focus.startswith("dk_k_under"):
        return "Repair K-rate/leash calibration before opening more K-under buckets."
    if repair == "projection_or_line_pricing_repair" and focus == "dk_tb15_over":
        return "Repair TB 1.5 line pricing and ROI before using the good CLV pocket."
    if repair == "projection_or_line_pricing_repair":
        return "Projection/line-pricing history is negative; fix the model, not the filter."
    return "Review exact blockers; this bucket is not one of the fastest micro paths."


def _score(row: dict[str, Any]) -> float:
    focus = row.get("focus")
    blockers = set(str(value) for value in row.get("micro_trial_blockers") or [])
    rows_needed = max(0, 75 - _int(row.get("rows")))
    clv_rows_needed = max(0, 30 - _int(row.get("clv_rows")))
    dates_needed = max(0, 4 - _int(row.get("clean_dates") or row.get("promotion_clean_dates")))
    clv = _float(row.get("trial_clv_beat_rate") or row.get("clv_beat_rate"))
    avg_clv = _float(row.get("trial_avg_clv_price") or row.get("avg_clv_price"))
    roi = _float(row.get("roi"))
    coverage = _float(row.get("valid_close_coverage"))
    focus_bonus = {
        "dk_k_under_plus_money": -35.0,
        "dk_k_under_fair_lay": -18.0,
        "dk_tb15_over": -25.0,
        "fanduel_hitter_display_only": 80.0,
        "other": 15.0,
    }.get(str(focus), 20.0)
    return (
        focus_bonus
        + len(blockers) * 12.0
        + rows_needed / 8.0
        + clv_rows_needed / 3.0
        + dates_needed * 10.0
        + max(0.0, 0.54 - (clv or 0.0)) * 120.0
        + max(0.0, 0.0 - (avg_clv or 0.0)) * 2.0
        + max(0.0, 0.0 - (roi or 0.0)) * 120.0
        + max(0.0, 0.90 - (coverage or 0.0)) * 120.0
    )


def _enrich(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["focus"] = _focus(out)
    out["repair_class"] = _repair_class(out)
    out["fixability_score"] = _score(out)
    out["next_action"] = _next_action(out)
    exact = out.get("exact_bucket_clv_prior") or {}
    out["exact_valid_close_coverage"] = exact.get("valid_close_coverage")
    out["exact_stale_close_rate"] = exact.get("stale_close_rate")
    out["exact_true_pair_rows"] = exact.get("true_pair_rows")
    out["exact_clv_rows"] = exact.get("clv_rows")
    return out


def build(model_dir: Path = _MODEL_DIR) -> dict[str, Any]:
    micro = _load(model_dir / "prop_micro_promotion_evaluation.json")
    rows = [_enrich(row) for row in (micro.get("buckets") or [])]
    rows.sort(key=lambda row: (row["repair_class"] != "trial_ready", row["fixability_score"], -_int(row.get("rows"))))

    blocker_counts = Counter()
    repair_counts = Counter()
    focus_counts = Counter()
    for row in rows:
        repair_counts[row["repair_class"]] += 1
        focus_counts[row["focus"]] += 1
        for blocker in row.get("micro_trial_blockers") or []:
            blocker_counts[str(blocker)] += 1

    trial_ready = [row for row in rows if row.get("micro_trial_ready")]
    near_miss = [row for row in rows if not row.get("micro_trial_ready") and row.get("focus") != "fanduel_hitter_display_only"]
    k_focus = [row for row in rows if str(row.get("focus")).startswith("dk_k_under")]
    tb15_focus = [row for row in rows if row.get("focus") == "dk_tb15_over"]
    fanduel_display = [row for row in rows if row.get("focus") == "fanduel_hitter_display_only"]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if micro else "missing_micro_promotion_evaluation",
        "source": "prop_micro_promotion_evaluation.json",
        "evaluated_buckets": len(rows),
        "trial_ready_count": len(trial_ready),
        "near_miss_count": len(near_miss),
        "repair_class_counts": dict(repair_counts.most_common()),
        "focus_counts": dict(focus_counts.most_common()),
        "blocker_counts": dict(blocker_counts.most_common()),
        "trial_ready": trial_ready[:20],
        "closest_repairable": near_miss[:40],
        "draftkings_k_focus": k_focus[:40],
        "draftkings_tb15_focus": tb15_focus[:30],
        "fanduel_display_only": fanduel_display[:30],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_micro_bucket_repair_report.json", payload)
    report_path = _REPORT_DIR / "mlb_prop_micro_bucket_repair_latest.md"
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    return payload


def _row_line(row: dict[str, Any]) -> str:
    blockers = ", ".join(row.get("micro_trial_blockers") or []) or "-"
    return (
        f"| {row.get('bucket')} | {row.get('focus')} | {row.get('repair_class')} | "
        f"{_int(row.get('rows'))} | {_pct(row.get('roi'))} | {_int(row.get('clv_rows'))} | "
        f"{_pct(row.get('trial_clv_beat_rate') or row.get('clv_beat_rate'))} | "
        f"{_fmt(row.get('trial_avg_clv_price') or row.get('avg_clv_price'))} | "
        f"{_pct(row.get('valid_close_coverage'))} | {_pct(row.get('exact_valid_close_coverage'))} | "
        f"{_fmt(row.get('fixability_score'), 1)} | {blockers} | {row.get('next_action')} |"
    )


def _section(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| Bucket | Focus | Repair Class | Rows | ROI | CLV Rows | Trial CLV | Avg CLV | Policy Close | Exact Close | Score | Trial Blockers | Next Action |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    if not rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - |")
        return
    for row in rows:
        lines.append(_row_line(row))


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Micro Bucket Almost-There Repair Report",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Usage: diagnostic only. Trial gates still live in `prop_micro_promotion_evaluation`.",
        "",
        f"- Evaluated buckets: {payload.get('evaluated_buckets', 0)}",
        f"- Trial-ready buckets: {payload.get('trial_ready_count', 0)}",
        f"- Repairable near misses: {payload.get('near_miss_count', 0)}",
        "",
        "## Repair Class Counts",
        "",
        "| Repair Class | Buckets |",
        "|---|---:|",
    ]
    for repair, count in (payload.get("repair_class_counts") or {}).items():
        lines.append(f"| {repair} | {count} |")
    lines.extend([
        "",
        "## Top Trial Blockers",
        "",
        "| Blocker | Buckets |",
        "|---|---:|",
    ])
    for blocker, count in list((payload.get("blocker_counts") or {}).items())[:20]:
        lines.append(f"| {blocker} | {count} |")
    _section(lines, "Trial Ready", payload.get("trial_ready") or [])
    _section(lines, "Closest Repairable Buckets", payload.get("closest_repairable") or [])
    _section(lines, "DraftKings K Focus", payload.get("draftkings_k_focus") or [])
    _section(lines, "DraftKings TB 1.5 Focus", payload.get("draftkings_tb15_focus") or [])
    _section(lines, "FanDuel Hitter Display Only", payload.get("fanduel_display_only") or [])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop micro bucket repair report")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    args = parser.parse_args()
    payload = build(Path(args.model_dir))
    print(json.dumps({
        "status": payload.get("status"),
        "trial_ready_count": payload.get("trial_ready_count"),
        "near_miss_count": payload.get("near_miss_count"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
