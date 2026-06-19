"""Real-money kill switch for MLB player props.

This module reads existing shadow/report artifacts and writes one compact
artifact that the prediction step can obey without touching the database.
The safest failure mode is always no real-money prop bets.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


@dataclass(frozen=True)
class PropRealMoneyKillSwitchConfig:
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_real_money_kill_switch.json"
    report_file: str = "mlb_prop_real_money_kill_switch_latest.md"
    max_artifact_age_hours: float = 36.0
    min_valid_close_coverage: float = 0.90
    max_stale_close_rate: float = 0.02
    min_clv_beat_rate: float = 0.55
    require_open_bucket: bool = True


_REQUIRED_ARTIFACTS = {
    "readiness": "prop_bucket_trust_scores.json",
    "reopen_policy": "prop_bucket_reopen_policy.json",
    "shadow_selector": "prop_shadow_selector_report.json",
    "distribution": "prop_distribution_models.json",
    "opportunity": "prop_opportunity_models.json",
    "hitter_outcome": "hitter_player_game_outcome_models.json",
    "target_quality": "prop_target_quality_report.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _age_hours(value: Any) -> float | None:
    dt = _parse_dt(value)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    val = _as_float(value)
    return "-" if val is None else f"{val:.1%}"


def _artifact_generated_at(payload: dict[str, Any]) -> Any:
    return payload.get("generated_at_utc") or payload.get("trained_at_utc")


def evaluate_kill_switch(cfg: PropRealMoneyKillSwitchConfig) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_status: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    for name, filename in _REQUIRED_ARTIFACTS.items():
        path = cfg.model_dir / filename
        payload = _load_json(path) if path.exists() else {}
        artifacts[name] = payload
        generated_at = _artifact_generated_at(payload)
        age = _age_hours(generated_at)
        status = {
            "file": filename,
            "exists": path.exists(),
            "generated_at_utc": generated_at,
            "age_hours": age,
        }
        if not path.exists() or not payload:
            status["status"] = "missing"
            blockers.append(f"artifact_missing:{name}")
        elif age is None:
            status["status"] = "unknown_age"
            blockers.append(f"artifact_age_unknown:{name}")
        elif age > cfg.max_artifact_age_hours:
            status["status"] = "stale"
            blockers.append(f"artifact_stale:{name}")
        else:
            status["status"] = "fresh"
        artifact_status[name] = status

    readiness = artifacts.get("readiness") or {}
    close_quality = readiness.get("close_quality") or {}
    valid_close = _as_float(close_quality.get("valid_close_coverage"))
    stale_close = _as_float(close_quality.get("stale_close_rate"))
    if valid_close is None or valid_close < cfg.min_valid_close_coverage:
        blockers.append(f"valid_close_coverage<{cfg.min_valid_close_coverage:.2f}")
    if stale_close is None or stale_close > cfg.max_stale_close_rate:
        blockers.append(f"stale_close_rate>{cfg.max_stale_close_rate:.2f}")

    policy = artifacts.get("reopen_policy") or {}
    reopen_buckets = policy.get("reopen_buckets") or {}
    open_count = len(reopen_buckets) if isinstance(reopen_buckets, dict) else 0
    if cfg.require_open_bucket and open_count <= 0:
        blockers.append("no_open_prop_buckets")
    for key, rec in list(reopen_buckets.items()) if isinstance(reopen_buckets, dict) else []:
        summary = rec.get("holdout") or rec.get("summary") or {}
        clv_beat = _as_float(summary.get("clv_price_beat_rate") or summary.get("clv_beat_rate"))
        avg_clv = _as_float(summary.get("avg_clv_price"))
        if clv_beat is not None and clv_beat < cfg.min_clv_beat_rate:
            blockers.append(f"open_bucket_clv_beat_low:{key}")
        if avg_clv is not None and avg_clv <= 0.0:
            blockers.append(f"open_bucket_avg_clv_nonpositive:{key}")

    selector = artifacts.get("shadow_selector") or {}
    real_candidates = int(selector.get("real_candidate_rows") or 0)
    if open_count > 0 and real_candidates <= 0:
        blockers.append("selector_real_candidates_zero")

    hitter = artifacts.get("hitter_outcome") or {}
    hitter_rec = hitter.get("recommendation") or {}
    if hitter and not bool(hitter_rec.get("passes_basic_gate")):
        warnings.append("hitter_event_model_not_ready_for_distribution")

    distribution = artifacts.get("distribution") or {}
    event_meta = ((distribution.get("distribution_calibrators") or {}).get("event_model") or {})
    if event_meta.get("status") == "gated_off":
        warnings.append("distribution_event_model_gated_off")

    target_quality = artifacts.get("target_quality") or {}
    for rec in target_quality.get("fanduel_market_evidence") or []:
        stat = str(rec.get("market") or rec.get("stat") or "")
        clean_rate = _as_float(rec.get("clean_market_evidence_rate"))
        synthetic = int(rec.get("synthetic_pairs") or 0)
        if stat in {"batter_hits", "batter_total_bases", "batter_home_runs"} and synthetic > 0 and (clean_rate is None or clean_rate < 0.50):
            warnings.append(f"fanduel_synthetic_evidence_display_only:{stat}")

    blockers = sorted(dict.fromkeys(blockers))
    warnings = sorted(dict.fromkeys(warnings))
    active = bool(blockers)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "disabled" if active else "enabled",
        "active": active,
        "blockers": blockers,
        "warnings": warnings,
        "thresholds": {
            "max_artifact_age_hours": cfg.max_artifact_age_hours,
            "min_valid_close_coverage": cfg.min_valid_close_coverage,
            "max_stale_close_rate": cfg.max_stale_close_rate,
            "min_clv_beat_rate": cfg.min_clv_beat_rate,
            "require_open_bucket": cfg.require_open_bucket,
        },
        "close_quality": close_quality,
        "open_reopen_buckets": open_count,
        "selector_real_candidate_rows": real_candidates,
        "artifact_status": artifact_status,
    }


def load_prop_kill_switch_state(
    model_dir: Path | str = _MODEL_DIR,
    *,
    file_name: str = "prop_real_money_kill_switch.json",
    max_age_hours: float = 36.0,
) -> dict[str, Any]:
    """Load the kill switch; missing/stale switch means real betting is disabled."""
    path = Path(model_dir) / file_name
    if not path.exists():
        return {
            "active": True,
            "status": "disabled",
            "blockers": ["kill_switch_artifact_missing"],
            "warnings": [],
        }
    payload = _load_json(path)
    age = _age_hours(payload.get("generated_at_utc"))
    if age is None or age > max_age_hours:
        blockers = list(payload.get("blockers") or [])
        blockers.append("kill_switch_artifact_stale")
        return {
            **payload,
            "active": True,
            "status": "disabled",
            "blockers": sorted(dict.fromkeys(blockers)),
        }
    return payload


def build_report(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Real-Money Kill Switch",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Status: {payload.get('status')}",
        f"Open buckets: {payload.get('open_reopen_buckets', 0)}",
        f"Selector real candidates: {payload.get('selector_real_candidate_rows', 0)}",
        "",
        "## Close Quality",
        "",
        f"- Valid close coverage: {_fmt_pct((payload.get('close_quality') or {}).get('valid_close_coverage'))}",
        f"- Stale close rate: {_fmt_pct((payload.get('close_quality') or {}).get('stale_close_rate'))}",
        "",
        "## Blockers",
        "",
    ]
    blockers = payload.get("blockers") or []
    lines.extend([f"- {reason}" for reason in blockers] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings") or []
    lines.extend([f"- {reason}" for reason in warnings] or ["- none"])
    lines.extend(["", "## Artifacts", ""])
    lines.append("| Artifact | Status | Age Hours |")
    lines.append("| --- | --- | --- |")
    for name, rec in (payload.get("artifact_status") or {}).items():
        age = rec.get("age_hours")
        age_s = "-" if age is None else f"{float(age):.1f}"
        lines.append(f"| {name} | {rec.get('status')} | {age_s} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], cfg: PropRealMoneyKillSwitchConfig) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    (cfg.model_dir / cfg.out_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (_REPORT_DIR / cfg.report_file).write_text(build_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop real-money kill-switch artifact")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_real_money_kill_switch.json")
    parser.add_argument("--report-file", default="mlb_prop_real_money_kill_switch_latest.md")
    parser.add_argument("--max-artifact-age-hours", type=float, default=36.0)
    parser.add_argument("--min-valid-close-coverage", type=float, default=0.90)
    parser.add_argument("--max-stale-close-rate", type=float, default=0.02)
    parser.add_argument("--min-clv-beat-rate", type=float, default=0.55)
    parser.add_argument("--allow-no-open-buckets", action="store_true")
    args = parser.parse_args()
    cfg = PropRealMoneyKillSwitchConfig(
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        report_file=args.report_file,
        max_artifact_age_hours=args.max_artifact_age_hours,
        min_valid_close_coverage=args.min_valid_close_coverage,
        max_stale_close_rate=args.max_stale_close_rate,
        min_clv_beat_rate=args.min_clv_beat_rate,
        require_open_bucket=not args.allow_no_open_buckets,
    )
    payload = evaluate_kill_switch(cfg)
    write_outputs(payload, cfg)
    print(json.dumps({
        "status": payload.get("status"),
        "blockers": payload.get("blockers"),
        "warnings": payload.get("warnings"),
        "out_file": str(cfg.model_dir / cfg.out_file),
    }, indent=2))


if __name__ == "__main__":
    main()
