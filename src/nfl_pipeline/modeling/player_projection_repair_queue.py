"""Rank NFL player projection repair targets from the projection audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_AUDIT = MODEL_DIR / "nfl_player_projection_audit.json"
DEFAULT_JSON = MODEL_DIR / "nfl_player_projection_repair_queue.json"
DEFAULT_MD = ROOT / "reports" / "nfl_player_projection_repair_queue_latest.md"


@dataclass(frozen=True)
class RepairQueueConfig:
    audit_file: Path = DEFAULT_AUDIT
    out_file: Path = DEFAULT_JSON
    md_report_file: Path = DEFAULT_MD


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "stats": {}, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "load_failed", "stats": {}, "error": str(exc), "path": str(path)}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _fix_hint(stat: str, group: str) -> str:
    group_l = group.lower()
    if (
        "week" in group_l
        or "rest" in group_l
        or "limited_usage" in group_l
        or "fragility" in group_l
        or "full_workload_failed" in group_l
        or "opportunity_low" in group_l
        or "opportunity_high" in group_l
    ):
        return "starter/rest usage model: starter confidence, Week 18/rest risk, inactive/practice downgrade, and depth movement"
    if stat == "receiving_tds":
        return "rare-event TD head: red-zone targets, goal-line targets, first-read/route role, QB TD tendency, team total, opponent red-zone defense"
    if stat == "receiving_yards":
        return "route/target opportunity: route participation proxy, targets per route, WOPR, air yards, depth role, and matchup"
    if stat == "rushing_tds":
        return "goal-line role: red-zone/goal-line carries, team implied points, game script, and RB depth role"
    if stat == "rushing_yards":
        return "carry-share opportunity: RB depth movement, starter confidence, rest risk, spread/game script, and snap share"
    if stat.startswith("passing"):
        return "QB game script/opportunity: attempts/dropbacks, rest/injury confidence, pace, pass rate, and opponent profile"
    return "player-rate model: shrink noisy player priors and add role/matchup context"


def build_queue(cfg: RepairQueueConfig) -> dict[str, Any]:
    audit = _load_json(cfg.audit_file)
    items: list[dict[str, Any]] = []
    for stat, rec in (audit.get("stats") or {}).items():
        if not isinstance(rec, dict) or rec.get("status") not in {None, "ready"}:
            continue
        model_mae = _float(rec.get("model_mae"))
        baseline_mae = _float(rec.get("baseline_mae"))
        stat_loss = max(0.0, model_mae - baseline_mae)
        abs_bias = abs(_float(rec.get("bias")))
        for row in rec.get("by_error_type") or []:
            if not isinstance(row, dict):
                continue
            rows = int(row.get("rows") or 0)
            gain = _float(row.get("gain_vs_baseline"))
            bucket_loss = max(0.0, -gain)
            bias = abs(_float(row.get("bias")))
            score = rows * (bucket_loss + 0.25 * stat_loss + 0.05 * bias + 0.05 * abs_bias)
            if score <= 0 and rows < 25:
                continue
            items.append({
                "stat": stat,
                "group": row.get("error_type") or "unknown",
                "rows": rows,
                "model_mae": row.get("model_mae"),
                "baseline_mae": row.get("baseline_mae"),
                "gain_vs_baseline": gain,
                "bias": row.get("bias"),
                "stat_model_mae": model_mae,
                "stat_baseline_mae": baseline_mae,
                "stat_projection_pass": bool(rec.get("projection_pass")),
                "priority_score": score,
                "fix_hint": _fix_hint(stat, str(row.get("error_type") or "")),
            })
    items.sort(key=lambda row: row["priority_score"], reverse=True)
    payload = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if audit.get("status") != "missing" else "missing_audit",
        "audit_status": audit.get("status"),
        "items": items[:80],
    }
    _atomic_write_text(cfg.out_file, json.dumps(payload, indent=2, default=str))
    _write_markdown(payload, cfg.md_report_file)
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Player Projection Repair Queue",
        "",
        "This turns the player projection audit into ranked model work. Higher scores mean the model is losing to baseline in a larger, more fixable group.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Audit status: {payload.get('audit_status')}",
        f"- Built at: {payload.get('built_at_utc')}",
        "",
        "| Rank | Stat | Group | Rows | Model MAE | Baseline MAE | Gain | Bias | Stat Pass | Fix Hint |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    items = payload.get("items") or []
    if not items:
        lines.append("| 1 | none | - | 0 | - | - | - | - | - | no repair queue rows |")
    for idx, row in enumerate(items[:40], start=1):
        lines.append(
            f"| {idx} | {row.get('stat')} | {row.get('group')} | {int(row.get('rows') or 0)} | "
            f"{_fmt(row.get('model_mae'))} | {_fmt(row.get('baseline_mae'))} | {_fmt(row.get('gain_vs_baseline'))} | "
            f"{_fmt(row.get('bias'))} | {'yes' if row.get('stat_projection_pass') else 'no'} | {row.get('fix_hint')} |"
        )
    lines.append("")
    text = "\n".join(lines)
    _atomic_write_text(path, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL player projection repair queue")
    parser.add_argument("--audit-file", default=str(DEFAULT_AUDIT))
    parser.add_argument("--out-file", default=str(DEFAULT_JSON))
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD))
    args = parser.parse_args()
    payload = build_queue(RepairQueueConfig(
        audit_file=Path(args.audit_file),
        out_file=Path(args.out_file),
        md_report_file=Path(args.md_report_file),
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
