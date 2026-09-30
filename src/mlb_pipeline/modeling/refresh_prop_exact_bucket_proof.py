"""Refresh exact-bucket prop proof artifacts without letting one market wipe all proof."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text
from mlb_pipeline.subprocess_utils import run_subprocess_tree

from .train_prop_distribution_models import _DEFAULT_MARKETS, _merge_market_calibrators

_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _run_distribution_market(
    market: str,
    *,
    model_dir: Path,
    cache_file: str,
    timeout_s: int,
    lookback_days: int,
    max_walk_forward_folds: int,
) -> dict[str, Any]:
    out_file = f"prop_distribution_models.{market}.json"
    report_file = f"mlb_prop_distribution_models_{market}_latest.md"
    cmd = [
        sys.executable,
        "-m",
        "mlb_pipeline.modeling.train_prop_distribution_models",
        "--markets",
        market,
        "--model-dir",
        str(model_dir),
        "--cache-file",
        cache_file,
        "--out-file",
        out_file,
        "--report-file",
        report_file,
        "--lookback-days",
        str(lookback_days),
        "--max-walk-forward-folds",
        str(max_walk_forward_folds),
    ]
    started = datetime.now(timezone.utc)
    rc, stdout, stderr, seconds = run_subprocess_tree(cmd, timeout_s=timeout_s, cwd=str(Path(__file__).resolve().parents[3]))
    payload = _load_json(model_dir / out_file) if rc == 0 else {}
    return {
        "market": market,
        "status": "ok" if rc == 0 and payload else "failed",
        "return_code": rc,
        "seconds": seconds,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "out_file": out_file,
        "report_file": report_file,
        "stdout_tail": "\n".join((stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((stderr or "").splitlines()[-20:]),
        "payload": payload,
    }


def _combine_market_payloads(existing: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run.get("status") == "ok" and run.get("payload")]
    existing_by_market = dict(((existing.get("distribution_calibrators") or {}).get("by_market") or {}))
    market_artifacts = dict(existing_by_market)
    for run in successful:
        payload = run["payload"]
        by_market = ((payload.get("distribution_calibrators") or {}).get("by_market") or {})
        for market, artifacts in by_market.items():
            market_artifacts[str(market)] = dict(artifacts or {})
    calibrators = _merge_market_calibrators(market_artifacts)
    existing_market_training = {
        str(row.get("market")): dict(row)
        for row in (existing.get("market_training") or [])
        if row.get("market")
    }
    bucket_selection = [
        row for row in (existing.get("bucket_model_selection") or [])
        if str(row.get("bucket", "")).split("|", 1)[0] not in {run["market"] for run in successful}
    ]
    line_gates = dict(((existing.get("tb_hr_line_production_gates") or {}).get("groups") or {}))
    for run in successful:
        payload = run["payload"]
        for row in payload.get("market_training") or []:
            if row.get("market"):
                existing_market_training[str(row["market"])] = dict(row)
        bucket_selection.extend(payload.get("bucket_model_selection") or [])
        line_gates.update(((payload.get("tb_hr_line_production_gates") or {}).get("groups") or {}))
    combined = dict(existing)
    combined.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ready" if market_artifacts else "missing_market_artifacts",
        "usage": "shadow_only",
        "partial_market_refresh": True,
        "market_refresh_runs": [
            {key: value for key, value in run.items() if key != "payload"}
            for run in runs
        ],
        "markets": sorted(market_artifacts.keys()),
        "market_training": list(existing_market_training.values()),
        "distribution_calibrators": calibrators,
        "k_v3": calibrators.get("k_v3") or existing.get("k_v3") or {},
        "bucket_model_selection": bucket_selection,
        "tb_hr_line_production_gates": {
            **(existing.get("tb_hr_line_production_gates") or {}),
            "groups": line_gates,
        },
    })
    return combined


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# MLB Prop Exact-Bucket Proof Refresh",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Status: **{payload.get('status')}**",
        "",
        "| Market | Status | Seconds | Return Code |",
        "|---|---|---:|---:|",
    ]
    for run in payload.get("market_refresh_runs") or []:
        lines.append(
            f"| {run.get('market')} | {run.get('status')} | {run.get('seconds')} | {run.get('return_code')} |"
        )
    lines.extend([
        "",
        f"Merged markets: {', '.join(payload.get('markets') or []) or '-'}",
        f"Exact bucket model-selection rows: {len(payload.get('bucket_model_selection') or [])}",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    markets = [market.strip() for market in str(args.markets).split(",") if market.strip()]
    if not markets:
        markets = list(_DEFAULT_MARKETS)
    existing = _load_json(model_dir / "prop_distribution_models.json")
    runs = [
        _run_distribution_market(
            market,
            model_dir=model_dir,
            cache_file=args.cache_file,
            timeout_s=args.market_timeout_seconds,
            lookback_days=args.lookback_days,
            max_walk_forward_folds=args.max_walk_forward_folds,
        )
        for market in markets
    ]
    combined = _combine_market_payloads(existing, runs)
    model_dir.mkdir(parents=True, exist_ok=True)
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "prop_distribution_models.json", combined)
    atomic_write_json(model_dir / "prop_exact_bucket_proof_refresh.json", {
        "generated_at_utc": combined.get("generated_at_utc"),
        "status": combined.get("status"),
        "market_refresh_runs": combined.get("market_refresh_runs"),
        "merged_markets": combined.get("markets"),
        "bucket_model_selection_rows": len(combined.get("bucket_model_selection") or []),
    })
    report = _REPORT_DIR / "mlb_prop_exact_bucket_proof_refresh_latest.md"
    atomic_write_text(report, _render(combined))
    combined["report_path"] = str(report)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh prop exact-bucket proof by market and merge artifacts")
    parser.add_argument("--model-dir", default=str(_MODEL_DIR))
    parser.add_argument("--markets", default=",".join(_DEFAULT_MARKETS))
    parser.add_argument("--cache-file", default="prop_distribution_training_cache.pkl")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--max-walk-forward-folds", type=int, default=1)
    parser.add_argument("--market-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({
        "status": payload.get("status"),
        "markets": payload.get("markets"),
        "bucket_model_selection_rows": len(payload.get("bucket_model_selection") or []),
        "report_path": payload.get("report_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
