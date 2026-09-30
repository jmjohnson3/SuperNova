"""Control panel for promoting MLB prop shadow layers into live use.

This report is intentionally conservative: a layer can become a live scoring
input only after its own prospective projection gate passes. Real-money bucket
promotion still requires the exact-bucket ladder, CLV, close-quality, and kill
switch gates to pass separately.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


@dataclass(frozen=True)
class LayerPromotionConfig:
    model_dir: Path = _MODEL_DIR
    out_file: str = "prop_layer_promotion_control.json"
    report_file: str = "mlb_prop_layer_promotion_latest.md"
    min_projection_dates: int = 5
    min_player_rows: int = 100
    min_opportunity_rows: int = 50
    min_clean_dates: int = 5
    min_clv_auc: float = 0.55
    max_artifact_age_hours: float = 36.0


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


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
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _fresh_blockers(payload: dict[str, Any], label: str, cfg: LayerPromotionConfig) -> list[str]:
    if not payload:
        return [f"{label}_artifact_missing"]
    generated_at = payload.get("generated_at_utc") or payload.get("trained_at_utc")
    age = _age_hours(generated_at)
    if age is None:
        return [f"{label}_artifact_age_unknown"]
    if age > cfg.max_artifact_age_hours:
        return [f"{label}_artifact_stale>{cfg.max_artifact_age_hours:.0f}h"]
    return []


def _metric(release: dict[str, Any], stat: str) -> dict[str, Any]:
    value = (release.get("metrics") or {}).get(stat) or {}
    return value if isinstance(value, dict) else {}


def _pending_terminal_anchor_rows(release: dict[str, Any]) -> int:
    rows = release.get("by_date") or []
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("games_final")):
            total += _as_int(row.get("pending_rows"))
    return total


def _projection_layer(
    *,
    name: str,
    title: str,
    current_mode: str,
    target_mode: str,
    release: dict[str, Any],
    stats: tuple[str, ...],
    cfg: LayerPromotionConfig,
    integration_key: str | None = None,
    projection_gates: dict[str, dict[str, Any]] | None = None,
    gate_stats: tuple[str, ...] = (),
    min_rows: int | None = None,
    extra_blockers: list[str] | None = None,
) -> dict[str, Any]:
    rows_required = min_rows if min_rows is not None else cfg.min_player_rows
    blockers: list[str] = list(extra_blockers or [])
    completed = _as_int(release.get("completed_date_count"))
    pending_terminal = _pending_terminal_anchor_rows(release)
    if completed < cfg.min_projection_dates:
        blockers.append(f"completed_dates<{cfg.min_projection_dates}")
    if pending_terminal > 0:
        blockers.append("pending_terminal_anchor_rows")
    for stat in stats:
        metric = _metric(release, stat)
        if _as_int(metric.get("rows")) < rows_required:
            blockers.append(f"{stat}_rows<{rows_required}")
        if not bool(metric.get("projection_pass")):
            blockers.append(f"{stat}_projection_not_better_than_baseline")
    for stat in gate_stats:
        gate = (projection_gates or {}).get(stat) or {}
        if not gate.get("eligible"):
            blockers.append(f"{stat}_projection_gate_failed")
    blockers = sorted(dict.fromkeys(blockers))
    can_integrate = not blockers
    return {
        "name": name,
        "title": title,
        "current_mode": current_mode,
        "target_mode": target_mode,
        "integration_key": integration_key,
        "can_score_live": can_integrate,
        "can_auto_integrate": bool(integration_key and can_integrate),
        "completed_dates": completed,
        "pending_terminal_anchor_rows": pending_terminal,
        "metrics": {stat: _metric(release, stat) for stat in stats},
        "blockers": blockers,
    }


def _support_layer(
    *,
    name: str,
    title: str,
    ready: bool,
    current_mode: str,
    target_mode: str,
    metrics: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    blockers = sorted(dict.fromkeys(str(item) for item in blockers if item))
    return {
        "name": name,
        "title": title,
        "current_mode": current_mode,
        "target_mode": target_mode,
        "integration_key": None,
        "can_score_live": bool(ready and not blockers),
        "can_auto_integrate": False,
        "metrics": metrics,
        "blockers": blockers,
    }


def _clv_layer(residual: dict[str, Any], cfg: LayerPromotionConfig) -> dict[str, Any]:
    blockers = _fresh_blockers(residual, "market_residual", cfg)
    clv_target = residual.get("clv_target") or {}
    holdout = clv_target.get("holdout") or {}
    auc = _as_float(holdout.get("auc"))
    if clv_target.get("status") != "ready":
        blockers.append("clv_target_not_ready")
    if holdout.get("brier") is None:
        blockers.append("clv_brier_missing")
    if auc is None or auc < cfg.min_clv_auc:
        blockers.append(f"clv_auc<{cfg.min_clv_auc:.2f}")
    return _support_layer(
        name="clv_direction",
        title="CLV Direction Model",
        ready=not blockers,
        current_mode="support",
        target_mode="support_enabled",
        metrics={
            "selected_variant": clv_target.get("selected_variant"),
            "auc": auc,
            "brier": holdout.get("brier"),
            "rows": holdout.get("rows"),
        },
        blockers=blockers,
    )


def _bookability_layer(bookability: dict[str, Any], cfg: LayerPromotionConfig) -> dict[str, Any]:
    blockers = _fresh_blockers(bookability, "bookability", cfg)
    target = (bookability.get("targets") or {}).get("valid_close_snapshot_captured") or {}
    holdout = target.get("holdout") or {}
    if bookability.get("status") != "ready":
        blockers.append("bookability_artifact_not_ready")
    if target.get("status") != "ready":
        blockers.append("valid_close_capture_target_not_ready")
    if _as_int(holdout.get("rows")) <= 0:
        blockers.append("bookability_holdout_missing")
    return _support_layer(
        name="bookability",
        title="Bookability / Close Capture",
        ready=not blockers,
        current_mode="support",
        target_mode="support_enabled",
        metrics={
            "selected_scoring_method": target.get("selected_scoring_method"),
            "holdout_rows": holdout.get("rows"),
            "actual_bookable_rate": holdout.get("actual_bookable_rate"),
            "auc": holdout.get("auc_model"),
            "model_usable": holdout.get("model_usable"),
        },
        blockers=blockers,
    )


def _distribution_layer(distribution: dict[str, Any], cfg: LayerPromotionConfig) -> dict[str, Any]:
    blockers = _fresh_blockers(distribution, "distribution", cfg)
    bucket_rows = distribution.get("bucket_model_selection") or []
    true_pair_buckets = [
        row for row in bucket_rows
        if isinstance(row, dict) and row.get("true_pair_only") and _as_int(row.get("true_pair_rows")) > 0
    ]
    if distribution.get("status") != "ready":
        blockers.append("distribution_artifact_not_ready")
    if not true_pair_buckets:
        blockers.append("no_true_pair_bucket_model_selection")
    return _support_layer(
        name="distribution_selector",
        title="Distribution / Exact-Line Selector",
        ready=not blockers,
        current_mode="shadow_support",
        target_mode="support_enabled",
        metrics={
            "true_pair_bucket_count": len(true_pair_buckets),
            "bucket_count": len(bucket_rows) if isinstance(bucket_rows, list) else 0,
            "usage": distribution.get("usage"),
        },
        blockers=blockers,
    )


def _micro_layer(micro: dict[str, Any], kill_switch: dict[str, Any], cfg: LayerPromotionConfig) -> dict[str, Any]:
    blockers = _fresh_blockers(micro, "micro_promotion", cfg)
    blockers.extend(_fresh_blockers(kill_switch, "kill_switch", cfg))
    ready_count = _as_int(micro.get("micro_ready_count"))
    tb_k_ready = _as_int(micro.get("tb_k_micro_ready_count"))
    if ready_count <= 0:
        blockers.append("micro_ready_exact_buckets=0")
    if kill_switch.get("active") or kill_switch.get("status") == "disabled":
        blockers.append("real_money_kill_switch_active")
    ready_buckets = [
        row for row in (micro.get("buckets") or [])
        if isinstance(row, dict) and row.get("micro_ready")
    ]
    blockers = sorted(dict.fromkeys(blockers))
    return {
        "name": "exact_bucket_micro",
        "title": "Exact Bucket $1 Micro",
        "current_mode": "watch",
        "target_mode": "micro",
        "integration_key": "exact_bucket_micro_ladder",
        "can_score_live": ready_count > 0,
        "can_auto_integrate": bool(ready_count > 0 and not blockers),
        "metrics": {
            "micro_ready_count": ready_count,
            "tb_k_micro_ready_count": tb_k_ready,
            "clv_classifier_ready": micro.get("clv_classifier_ready"),
        },
        "ready_buckets": ready_buckets,
        "blockers": blockers,
    }


def build_control(cfg: LayerPromotionConfig = LayerPromotionConfig()) -> dict[str, Any]:
    model_dir = Path(cfg.model_dir)
    projection = _load(model_dir / "daily_forecast_projection_audit.json")
    checkpoint = _load(model_dir / "prop_five_date_checkpoint.json")
    residual = _load(model_dir / "prop_market_residual_models.json")
    bookability = _load(model_dir / "prop_bookability_model.json")
    distribution = _load(model_dir / "prop_distribution_models.json")
    micro = _load(model_dir / "prop_micro_promotion_evaluation.json")
    kill_switch = _load(model_dir / "prop_real_money_kill_switch.json")
    policy = _load(model_dir / "prop_bucket_reopen_policy.json")

    projection_gates = projection.get("projection_gates") or {}
    hitter_release = checkpoint.get("hitter_release") or {}
    hitter_shadow = checkpoint.get("hitter_rate_shadow_release") or {}
    pitcher_release = checkpoint.get("pitcher_release") or {}
    checkpoint_fresh = _fresh_blockers(checkpoint, "five_date_checkpoint", cfg)
    projection_fresh = _fresh_blockers(projection, "projection_audit", cfg)

    layers = {
        "hitter_hits_hr_shadow": _projection_layer(
            name="hitter_hits_hr_shadow",
            title="Hitter Hits/HR Shadow",
            current_mode="shadow",
            target_mode="production_scoring",
            release=hitter_shadow,
            stats=("batter_hits", "batter_home_runs"),
            cfg=cfg,
            integration_key="hitter_rate_challenger_production",
            min_rows=cfg.min_player_rows,
            extra_blockers=checkpoint_fresh,
        ),
        "hitter_pa_v3": _projection_layer(
            name="hitter_pa_v3",
            title="Hitter PA v3",
            current_mode="challenger",
            target_mode="production_scoring",
            release=hitter_release,
            stats=("hitter_plate_appearances_challenger",),
            cfg=cfg,
            integration_key="hitter_pa_v3_production",
            projection_gates=projection_gates,
            gate_stats=("hitter_plate_appearances_challenger",),
            min_rows=cfg.min_opportunity_rows,
            extra_blockers=checkpoint_fresh + projection_fresh,
        ),
        "tb_projection": _projection_layer(
            name="tb_projection",
            title="Total Bases Projection",
            current_mode="production_tracking",
            target_mode="production_scoring_verified",
            release=hitter_release,
            stats=("batter_total_bases",),
            cfg=cfg,
            projection_gates=projection_gates,
            gate_stats=("batter_total_bases",),
            min_rows=cfg.min_player_rows,
            extra_blockers=checkpoint_fresh + projection_fresh,
        ),
        "pitcher_k_projection": _projection_layer(
            name="pitcher_k_projection",
            title="Pitcher K Projection",
            current_mode="production_tracking",
            target_mode="production_scoring_verified",
            release=pitcher_release,
            stats=("pitcher_strikeouts",),
            cfg=cfg,
            projection_gates=projection_gates,
            gate_stats=("pitcher_strikeouts",),
            min_rows=cfg.min_player_rows,
            extra_blockers=checkpoint_fresh + projection_fresh,
        ),
        "clv_direction": _clv_layer(residual, cfg),
        "bookability": _bookability_layer(bookability, cfg),
        "distribution_selector": _distribution_layer(distribution, cfg),
        "exact_bucket_micro": _micro_layer(micro, kill_switch, cfg),
    }

    ladder_buckets = policy.get("ladder_buckets") or {}
    ladder_counts: dict[str, int] = {}
    if isinstance(ladder_buckets, dict):
        for bucket in ladder_buckets.values():
            if not isinstance(bucket, dict):
                continue
            tier = str(bucket.get("ladder_tier") or "watch")
            ladder_counts[tier] = ladder_counts.get(tier, 0) + 1

    auto_integrations = {
        key: {
            "enabled": bool(layer.get("can_auto_integrate")),
            "layer": name,
            "target_mode": layer.get("target_mode"),
            "blockers": layer.get("blockers") or [],
        }
        for name, layer in layers.items()
        for key in [layer.get("integration_key")]
        if key
    }
    blockers = sorted({
        f"{name}:{reason}"
        for name, layer in layers.items()
        for reason in (layer.get("blockers") or [])
        if reason
    })
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready",
        "automatic_integration_mode": "safe_next_rung",
        "thresholds": {
            "min_projection_dates": cfg.min_projection_dates,
            "min_player_rows": cfg.min_player_rows,
            "min_opportunity_rows": cfg.min_opportunity_rows,
            "min_clean_dates": cfg.min_clean_dates,
            "min_clv_auc": cfg.min_clv_auc,
            "max_artifact_age_hours": cfg.max_artifact_age_hours,
        },
        "kill_switch": {
            "status": kill_switch.get("status"),
            "active": kill_switch.get("active"),
            "blockers": kill_switch.get("blockers") or [],
        },
        "ladder_counts": ladder_counts,
        "layers": layers,
        "auto_integrations": auto_integrations,
        "blockers": blockers,
    }
    return payload


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _as_float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def _write_report(payload: dict[str, Any], cfg: LayerPromotionConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.report_file
    lines = [
        "# MLB Prop Layer Promotion Control",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        f"Automatic integration mode: `{payload.get('automatic_integration_mode')}`",
        f"Kill switch: `{(payload.get('kill_switch') or {}).get('status')}`",
        "",
        "## Layers",
        "",
        "| Layer | Current | Target | Live Scoring | Auto Integrate | Key Blockers |",
        "|---|---|---|---|---|---|",
    ]
    for layer in (payload.get("layers") or {}).values():
        blockers = layer.get("blockers") or []
        lines.append(
            f"| {layer.get('title')} | {layer.get('current_mode')} | {layer.get('target_mode')} | "
            f"{bool(layer.get('can_score_live'))} | {bool(layer.get('can_auto_integrate'))} | "
            f"{', '.join(blockers[:6]) or '-'} |"
        )
    lines.extend([
        "",
        "## Auto Integrations",
        "",
        "| Integration | Enabled | Target | Blockers |",
        "|---|---|---|---|",
    ])
    for key, rec in (payload.get("auto_integrations") or {}).items():
        lines.append(
            f"| {key} | {bool(rec.get('enabled'))} | {rec.get('target_mode')} | "
            f"{', '.join((rec.get('blockers') or [])[:8]) or '-'} |"
        )
    lines.extend([
        "",
        "## Ladder",
        "",
        "| Tier | Buckets |",
        "|---|---:|",
    ])
    for tier, count in sorted((payload.get("ladder_counts") or {}).items()):
        lines.append(f"| {tier} | {count} |")
    lines.append("")
    atomic_write_text(path, "\n".join(lines) + "\n")
    return str(path)


def write_outputs(payload: dict[str, Any], cfg: LayerPromotionConfig) -> dict[str, Any]:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.model_dir / cfg.out_file, payload)
    payload = dict(payload)
    payload["report_path"] = _write_report(payload, cfg)
    return payload


def load_layer_promotion_control(
    model_dir: Path | str = _MODEL_DIR,
    *,
    file_name: str = "prop_layer_promotion_control.json",
    max_age_hours: float = 36.0,
) -> dict[str, Any]:
    path = Path(model_dir) / file_name
    payload = _load(path)
    if not payload:
        return {"status": "missing", "auto_integrations": {}, "blockers": ["layer_control_missing"]}
    age = _age_hours(payload.get("generated_at_utc"))
    if age is None or age > max_age_hours:
        blockers = list(payload.get("blockers") or [])
        blockers.append("layer_control_stale")
        return {**payload, "status": "stale", "blockers": sorted(dict.fromkeys(blockers))}
    return payload


def layer_auto_integration_enabled(
    model_dir: Path | str,
    integration_key: str,
    *,
    file_name: str = "prop_layer_promotion_control.json",
) -> bool:
    control = load_layer_promotion_control(model_dir, file_name=file_name)
    rec = (control.get("auto_integrations") or {}).get(integration_key) or {}
    return bool(control.get("status") == "ready" and rec.get("enabled"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MLB prop layer promotion control artifact")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--out-file", default="prop_layer_promotion_control.json")
    parser.add_argument("--report-file", default="mlb_prop_layer_promotion_latest.md")
    parser.add_argument("--max-artifact-age-hours", type=float, default=36.0)
    args = parser.parse_args()
    cfg = LayerPromotionConfig(
        model_dir=Path(args.model_dir),
        out_file=args.out_file,
        report_file=args.report_file,
        max_artifact_age_hours=args.max_artifact_age_hours,
    )
    payload = write_outputs(build_control(cfg), cfg)
    print(json.dumps({
        "status": payload.get("status"),
        "auto_integrations": payload.get("auto_integrations"),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
