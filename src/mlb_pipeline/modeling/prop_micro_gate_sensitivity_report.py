"""Micro-gate sensitivity report for MLB prop buckets.

This report does not promote bets. It answers the practical question:
if strict bankroll-style gates are too conservative for $1 testing, which
exact buckets are merely sample-limited and which are blocked by hard
truth-quality issues.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"

_PROFILES = {
    "watch_only": {
        "money": False,
        "rows": 50,
        "clv_rows": 15,
        "clean_dates": 3,
        "clv_beat": 0.50,
        "avg_clv": -0.10,
        "roi": -0.05,
        "calibration": 0.10,
        "valid_close_coverage": 0.85,
    },
    "micro_test": {
        "money": True,
        "rows": 75,
        "clv_rows": 20,
        "clean_dates": 4,
        "clv_beat": 0.52,
        "avg_clv": 0.0,
        "roi": 0.0,
        "calibration": 0.08,
        "valid_close_coverage": 0.90,
    },
    "strict_micro": {
        "money": True,
        "rows": 150,
        "clv_rows": 30,
        "clean_dates": 5,
        "clv_beat": 0.55,
        "avg_clv": 0.0,
        "roi": 0.0,
        "calibration": 0.05,
        "valid_close_coverage": 0.90,
    },
}

_HARD_BLOCKER_PREFIXES = {
    "underlying_projection_not_proven",
    "exact_bucket_true_pair_proof_missing",
    "clv_classifier_not_ready",
    "fanduel_hitter_true_pair_proof_missing",
    "real_money_kill_switch_active",
}


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


def _pct(value: Any, digits: int = 1) -> str:
    numeric = _float(value)
    return "-" if numeric is None else f"{numeric * 100.0:.{digits}f}%"


def _hard_blockers(row: dict[str, Any]) -> list[str]:
    blockers = [str(value) for value in row.get("blockers") or [] if value]
    return sorted({
        blocker for blocker in blockers
        if blocker in _HARD_BLOCKER_PREFIXES
        or blocker.startswith("fanduel_")
        or blocker.startswith("exact_bucket_true_pair")
        or blocker.startswith("underlying_projection")
    })


def _profile_result(row: dict[str, Any], profile_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    hard = _hard_blockers(row)
    blockers: list[str] = list(hard)
    rows = _int(row.get("rows"))
    clv_rows = _int(row.get("clv_rows"))
    clean_dates = _int(row.get("promotion_clean_dates") or row.get("clean_dates"))
    clv_beat = _float(row.get("clv_beat_rate"))
    avg_clv = _float(row.get("avg_clv_price"))
    roi = _float(row.get("roi"))
    calibration = abs(_float(row.get("calibration_error")) or 0.0)
    coverage = _float(row.get("valid_close_coverage"))

    if rows < int(profile["rows"]):
        blockers.append(f"needs_{int(profile['rows']) - rows}_rows")
    if clv_rows < int(profile["clv_rows"]):
        blockers.append(f"needs_{int(profile['clv_rows']) - clv_rows}_clv_rows")
    if clean_dates < int(profile["clean_dates"]):
        blockers.append(f"needs_{int(profile['clean_dates']) - clean_dates}_clean_dates")
    if clv_beat is None or clv_beat < float(profile["clv_beat"]):
        short = float(profile["clv_beat"]) - (clv_beat or 0.0)
        blockers.append(f"clv_beat_short_{short:.3f}")
    if avg_clv is None or avg_clv < float(profile["avg_clv"]):
        blockers.append("avg_clv_below_gate")
    if roi is None or roi < float(profile["roi"]):
        blockers.append("roi_below_gate")
    if calibration > float(profile["calibration"]):
        blockers.append("calibration_below_gate")
    if coverage is None or coverage < float(profile["valid_close_coverage"]):
        blockers.append("valid_close_coverage_below_gate")

    score = (
        max(0, int(profile["rows"]) - rows) / 15.0
        + max(0, int(profile["clv_rows"]) - clv_rows) / 5.0
        + max(0, int(profile["clean_dates"]) - clean_dates) * 8.0
        + max(0.0, float(profile["clv_beat"]) - (clv_beat or 0.0)) * 100.0
        + max(0.0, float(profile["avg_clv"]) - (avg_clv or 0.0)) * 0.5
        + max(0.0, float(profile["roi"]) - (roi or 0.0)) * 100.0
        + max(0.0, calibration - float(profile["calibration"])) * 100.0
        + max(0.0, float(profile["valid_close_coverage"]) - (coverage or 0.0)) * 100.0
        + len(hard) * 50.0
    )
    return {
        "profile": profile_name,
        "money_profile": bool(profile.get("money")),
        "passes": not blockers,
        "score": float(score),
        "hard_blockers": hard,
        "blockers": sorted(set(blockers)),
    }


def build(model_dir: Path = _MODEL_DIR) -> dict[str, Any]:
    micro = _load(model_dir / "prop_micro_promotion_evaluation.json")
    rows = list(micro.get("buckets") or [])
    evaluated: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for profile_name, profile in _PROFILES.items():
        profile_rows = []
        for row in rows:
            result = _profile_result(row, profile_name, profile)
            profile_rows.append({**row, **result})
        profile_rows.sort(key=lambda row: (not row["passes"], row["score"], -_int(row.get("rows"))))
        summary[profile_name] = {
            "passes": sum(1 for row in profile_rows if row["passes"]),
            "evaluated": len(profile_rows),
            "money_profile": bool(profile.get("money")),
            "thresholds": profile,
        }
        evaluated.extend(profile_rows[:40])
    hard_counts: dict[str, int] = {}
    for row in rows:
        for blocker in _hard_blockers(row):
            hard_counts[blocker] = hard_counts.get(blocker, 0) + 1
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if micro else "missing_micro_evaluation",
        "source": "prop_micro_promotion_evaluation.json",
        "profiles": summary,
        "hard_blocker_counts": dict(sorted(hard_counts.items(), key=lambda item: (-item[1], item[0]))),
        "candidates": evaluated,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_micro_gate_sensitivity_report.json", payload)
    report_path = _REPORT_DIR / "mlb_prop_micro_gate_sensitivity_latest.md"
    atomic_write_text(report_path, _render(payload))
    payload["report_path"] = str(report_path)
    return payload


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Micro Gate Sensitivity",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "Usage: diagnostic only. This report does not promote live bets.",
        "",
        "## Profile Summary",
        "",
        "| Profile | Money? | Passes | Evaluated | Rows | CLV Rows | Clean Dates | CLV Beat | Avg CLV | ROI | Cal Err | Valid Close |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in (payload.get("profiles") or {}).items():
        t = rec.get("thresholds") or {}
        lines.append(
            f"| {name} | {bool(rec.get('money_profile'))} | {rec.get('passes', 0)} | {rec.get('evaluated', 0)} | "
            f"{t.get('rows')} | {t.get('clv_rows')} | {t.get('clean_dates')} | {_pct(t.get('clv_beat'))} | "
            f"{_fmt(t.get('avg_clv'))} | {_pct(t.get('roi'))} | {_pct(t.get('calibration'))} | "
            f"{_pct(t.get('valid_close_coverage'))} |"
        )
    lines.extend([
        "",
        "## Hard Blocker Counts",
        "",
        "| Blocker | Buckets |",
        "|---|---:|",
    ])
    for blocker, count in (payload.get("hard_blocker_counts") or {}).items():
        lines.append(f"| {blocker} | {count} |")
    lines.extend([
        "",
        "## Closest Buckets By Profile",
        "",
        "| Profile | Bucket | Rows | Clean | ROI | CLV Rows | CLV Beat | Avg CLV | Cal Err | Coverage | Score | Passes | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in payload.get("candidates") or []:
        lines.append(
            f"| {row.get('profile')} | {row.get('bucket')} | {_int(row.get('rows'))} | "
            f"{_int(row.get('promotion_clean_dates') or row.get('clean_dates'))} | {_pct(row.get('roi'))} | "
            f"{_int(row.get('clv_rows'))} | {_pct(row.get('clv_beat_rate'))} | {_fmt(row.get('avg_clv_price'))} | "
            f"{_pct(row.get('calibration_error'))} | {_pct(row.get('valid_close_coverage'))} | "
            f"{_fmt(row.get('score'), 1)} | {bool(row.get('passes'))} | "
            f"{', '.join(row.get('blockers') or []) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop micro-gate sensitivity report")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    args = parser.parse_args()
    payload = build(Path(args.model_dir))
    print(json.dumps({
        "status": payload.get("status"),
        "profiles": {
            key: value.get("passes", 0)
            for key, value in (payload.get("profiles") or {}).items()
        },
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
