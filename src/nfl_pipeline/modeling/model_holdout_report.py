"""Write a concise NFL model holdout report."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from nfl_pipeline.integrity import active_release, MODEL_ROOT


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYER_REPORT = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_player_stat_models_report.json"
DEFAULT_OPPORTUNITY_REPORT = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_player_opportunity_models_report.json"
DEFAULT_DISTRIBUTION_REPORT = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_player_stat_distributions.json"
DEFAULT_OPPORTUNITY_DISTRIBUTION_REPORT = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_player_opportunity_distributions.json"
DEFAULT_PROJECTION_AUDIT = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_player_projection_audit.json"
DEFAULT_RECEIVER_SPIKE_DIAGNOSTIC = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_receiver_spike_miss_diagnostic.json"
DEFAULT_REPAIR_QUEUE = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_player_projection_repair_queue.json"
DEFAULT_EXACT_LINE_PROOF = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_prop_exact_line_proof.json"
DEFAULT_EXACT_LINE_MODELS = Path(__file__).resolve().parent / "models" / "player_props" / "nfl_prop_exact_line_models.json"
DEFAULT_GAME_REPORT = Path(__file__).resolve().parent / "models" / "game_bets" / "nfl_game_models_report.json"
DEFAULT_OUT = ROOT / "reports" / "nfl_model_holdout_latest.md"


@dataclass(frozen=True)
class ReportConfig:
    player_report: Path = DEFAULT_PLAYER_REPORT
    opportunity_report: Path = DEFAULT_OPPORTUNITY_REPORT
    distribution_report: Path = DEFAULT_DISTRIBUTION_REPORT
    opportunity_distribution_report: Path = DEFAULT_OPPORTUNITY_DISTRIBUTION_REPORT
    projection_audit: Path = DEFAULT_PROJECTION_AUDIT
    receiver_spike_diagnostic: Path = DEFAULT_RECEIVER_SPIKE_DIAGNOSTIC
    repair_queue: Path = DEFAULT_REPAIR_QUEUE
    exact_line_proof: Path = DEFAULT_EXACT_LINE_PROOF
    exact_line_models: Path = DEFAULT_EXACT_LINE_MODELS
    game_report: Path = DEFAULT_GAME_REPORT
    out_file: Path = DEFAULT_OUT


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "metrics": {}, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _section(title: str, payload: dict[str, Any]) -> list[str]:
    rows = [f"## {title}", "", "| Model | Rows | MAE | Baseline MAE | Gain | RMSE | Bias | Variant | Accepted |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    metrics = payload.get("metrics") or {}
    if not metrics:
        rows.append(f"| no metrics | - | - | - | - | - | - | - | {payload.get('status', 'missing')} |")
        return rows
    for name, rec in metrics.items():
        if not isinstance(rec, dict):
            continue
        if "projection_accepted" in rec:
            accepted = "yes" if rec.get("projection_accepted") else "no"
        elif "projection_pass" in rec:
            accepted = "yes" if rec.get("projection_pass") else "no"
        else:
            accepted = "yes" if rec.get("accepted") else "no"
        if name == "receiving_tds" and rec.get("td_probability_accepted"):
            accepted = f"{accepted} (TD prob yes)"
        rows.append(
            f"| {name} | {int(rec.get('rows') or rec.get('holdout_rows') or 0)} "
            f"| {_fmt_num(rec.get('mae'))} | {_fmt_num(rec.get('baseline_mae'))} "
            f"| {_fmt_num(rec.get('mae_gain_vs_baseline'))} | {_fmt_num(rec.get('rmse'))} "
            f"| {_fmt_num(rec.get('bias'))} | {rec.get('variant', 'direct')} | {accepted} |"
        )
    return rows


def _distribution_section(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Player Stat Distributions",
        "",
        "| Stat | Kind | Rows | Model MAE | Best Baseline MAE | Sigma | Bias | Brier | Base Brier | Mixture Brier | Receiver Spike Brier | Receiver Spike MAE | Accepted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    metrics = payload.get("distributions") or {}
    if not metrics:
        rows.append(f"| no distributions | - | - | - | - | - | - | - | - | - | - | - | {payload.get('status', 'missing')} |")
        return rows
    for stat, rec in metrics.items():
        if not isinstance(rec, dict):
            continue
        brier = rec.get("line_0_5_brier", rec.get("synthetic_line_brier"))
        base_brier = rec.get("baseline_line_0_5_brier", rec.get("baseline_synthetic_line_brier"))
        accepted = "yes" if rec.get("accepted_distribution") else "no"
        rows.append(
            f"| {stat} | {rec.get('kind', '-')} | {int(rec.get('holdout_rows') or 0)} "
            f"| {_fmt_num(rec.get('model_mae'))} | {_fmt_num(rec.get('best_baseline_mae'))} "
            f"| {_fmt_num(rec.get('residual_sigma'))} | {_fmt_num(rec.get('residual_bias'))} "
            f"| {_fmt_num(brier)} | {_fmt_num(base_brier)} | {_fmt_num(rec.get('mixture_line_brier'))} "
            f"| {_fmt_num(rec.get('receiver_spike_mixture_line_brier'))} "
            f"| {_fmt_num(rec.get('receiver_spike_mixture_mean_mae'))} | {accepted} |"
        )
    receiver = metrics.get("receiving_yards") or {}
    if receiver.get("receiver_spike_mixture_params"):
        rows.extend([
            "",
            f"Receiver spike challenger accepted: **{'yes' if receiver.get('receiver_spike_mixture_accepted') else 'no'}**. "
            "The Accepted column above describes the active distribution.",
            f"On identical pregame proxy lines, current curve Brier is {_fmt_num(receiver.get('receiver_spike_current_line_brier'))}; "
            f"challenger Brier is {_fmt_num(receiver.get('receiver_spike_mixture_line_brier'))}.",
            "The receiver challenger is tuned on earlier validation weeks, then checked on the final holdout. "
            "Its proxy thresholds are not sportsbook lines and do not establish betting edge.",
        ])
    return rows


def _opportunity_distribution_section(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Player Opportunity Distributions",
        "",
        "| Opportunity | Rows | MAE | Baseline MAE | Gain | Sigma | Bias | Brier | Base Brier | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    metrics = payload.get("distributions") or {}
    if not metrics:
        rows.append(f"| no opportunity distributions | - | - | - | - | - | - | - | - | {payload.get('status', 'missing')} |")
        return rows
    for name, rec in metrics.items():
        if not isinstance(rec, dict):
            continue
        accepted = "yes" if rec.get("accepted_distribution") else "no"
        rows.append(
            f"| {name} | {int(rec.get('holdout_rows') or 0)} "
            f"| {_fmt_num(rec.get('mae'))} | {_fmt_num(rec.get('baseline_mae'))} "
            f"| {_fmt_num(rec.get('mae_gain_vs_baseline'))} | {_fmt_num(rec.get('residual_sigma'))} "
            f"| {_fmt_num(rec.get('residual_bias'))} | {_fmt_num(rec.get('synthetic_line_brier'))} "
            f"| {_fmt_num(rec.get('baseline_synthetic_line_brier'))} | {accepted} |"
        )
    return rows


def _projection_audit_section(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Player Projection Audit",
        "",
        "| Stat | Rows | Model MAE | Baseline MAE | Gain | Bias | Model Used | Projection Pass | Dominant Error |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    stats = payload.get("stats") or {}
    if not stats:
        rows.append(f"| no projection audit | - | - | - | - | - | - | - | {payload.get('status', 'missing')} |")
        return rows
    for stat, rec in stats.items():
        if not isinstance(rec, dict):
            continue
        errors = rec.get("by_error_type") or []
        dominant = errors[0].get("error_type") if errors and isinstance(errors[0], dict) else "-"
        gain = None
        if rec.get("model_mae") is not None and rec.get("baseline_mae") is not None:
            gain = float(rec.get("baseline_mae") or 0) - float(rec.get("model_mae") or 0)
        rows.append(
            f"| {stat} | {int(rec.get('holdout_rows') or rec.get('rows') or 0)} "
            f"| {_fmt_num(rec.get('model_mae'))} | {_fmt_num(rec.get('baseline_mae'))} "
            f"| {_fmt_num(gain)} | {_fmt_num(rec.get('bias'))} "
            f"| {'yes' if rec.get('model_used') else 'baseline'} | {'yes' if rec.get('projection_pass') else 'no'} | {dominant} |"
        )
    return rows


def _receiver_spike_section(payload: dict[str, Any]) -> list[str]:
    rec = payload.get("receiving_yards") or {}
    rows = [
        "## Receiver Spike Miss Diagnostic",
        "",
        "| Rows | Model MAE | Baseline MAE | Gain | Bias | Big Misses | Missed Spikes | Target MAE | Target Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not rec:
        rows.append(f"| 0 | - | - | - | - | 0 | 0 | - | {payload.get('status', 'missing')} |")
        return rows
    gain = None
    if rec.get("model_mae") is not None and rec.get("baseline_mae") is not None:
        gain = float(rec.get("baseline_mae") or 0) - float(rec.get("model_mae") or 0)
    rows.append(
        f"| {int(rec.get('holdout_rows') or 0)} | {_fmt_num(rec.get('model_mae'))} | "
        f"{_fmt_num(rec.get('baseline_mae'))} | {_fmt_num(gain)} | {_fmt_num(rec.get('bias'))} | "
        f"{int(rec.get('big_miss_rows') or 0)} | {int(rec.get('missed_spike_rows') or 0)} | "
        f"{_fmt_num(rec.get('target_projection_mae'))} | {_fmt_num(rec.get('target_projection_bias'))} |"
    )
    rows.extend([
        "",
        "| Cause | Rows | Model MAE | Baseline MAE | Gain | Target Err | Air Err | YPT Err | Spike P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in (rec.get("by_miss_cause") or [])[:8]:
        rows.append(
            f"| {row.get('miss_cause')} | {int(row.get('rows') or 0)} | "
            f"{_fmt_num(row.get('model_mae'))} | {_fmt_num(row.get('baseline_mae'))} | "
            f"{_fmt_num(row.get('gain_vs_baseline'))} | {_fmt_num(row.get('avg_target_error'))} | "
            f"{_fmt_num(row.get('avg_air_yards_error'))} | {_fmt_num(row.get('avg_ypt_error_contribution'))} | "
            f"{_fmt_num(row.get('avg_spike_probability'))} |"
        )
    return rows


def _repair_queue_section(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Projection Repair Queue",
        "",
        "| Rank | Stat | Group | Rows | Gain | Bias | Fix Hint |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    items = payload.get("items") or []
    if not items:
        rows.append(f"| 1 | none | - | 0 | - | - | {payload.get('status', 'missing')} |")
        return rows
    for idx, row in enumerate(items[:12], start=1):
        rows.append(
            f"| {idx} | {row.get('stat')} | {row.get('group')} | {int(row.get('rows') or 0)} "
            f"| {_fmt_num(row.get('gain_vs_baseline'))} | {_fmt_num(row.get('bias'))} | {row.get('fix_hint')} |"
        )
    return rows


def _exact_line_section(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Exact-Line Prop Proof",
        "",
        f"- Status: {payload.get('status', 'missing')}",
        f"- Prediction rows: {payload.get('prediction_rows', 0)}",
        "",
        "| Stat | Book | Rows | True Paired | Lock Rows | Close Rows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    offers = payload.get("offer_summary") or []
    if not offers:
        rows.append("| none | - | 0 | 0 | 0 | 0 |")
        return rows
    for row in offers[:30]:
        rows.append(
            f"| {row.get('stat')} | {row.get('book')} | {int(row.get('rows') or 0)} | "
            f"{int(row.get('true_paired_rows') or 0)} | {int(row.get('lock_rows') or 0)} | {int(row.get('close_rows') or 0)} |"
        )
    return rows


def _exact_line_model_section(payload: dict[str, Any]) -> list[str]:
    rows = [
        "## Exact-Line Prop Models",
        "",
        f"- Status: {payload.get('status', 'missing')}",
        f"- True-paired rows: {payload.get('true_paired_rows', 0)}",
        "",
        "| Stat | Side | Status | Train | Holdout | Brier | Model Brier | Market Brier | AUC | Accepted | CLV Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    metrics = payload.get("metrics") or []
    if not metrics:
        rows.append("| none | - | waiting | 0 | 0 | - | - | - | - | no | - |")
        return rows
    for row in metrics[:40]:
        auc = "-" if row.get("auc") is None else _fmt_num(row.get("auc"))
        rows.append(
            f"| {row.get('stat')} | {row.get('side')} | {row.get('status')} | "
            f"{int(row.get('train_rows') or 0)} | {int(row.get('holdout_rows') or 0)} | "
            f"{_fmt_num(row.get('brier'))} | {_fmt_num(row.get('model_probability_brier'))} | "
            f"{_fmt_num(row.get('market_no_vig_brier'))} | {auc} | "
            f"{'yes' if row.get('accepted') else 'no'} | {(row.get('clv') or {}).get('status', '-')} |"
        )
    return rows


def build_report(cfg: ReportConfig) -> str:
    release = active_release()
    if release:
        payload = _load_json(MODEL_ROOT/'releases'/release['release_id']/'report.json')
        lines = ['# NFL Active Release Evaluation', '', f"Release: {release['release_id']}", '',
                 'Fixed candidates evaluated on expanding 2025 week folds and a later 2026 veto window.',
                 'The 2026 window was previously examined during research; it is not an untouched prospective test.',
                 'Yardage/game Brier uses predeclared proxy lines, not executable sportsbook offers. TD Brier is P(any TD).', '',
                 '| Target | Live | OOF MAE / baseline | Later MAE / baseline | Later Brier / baseline | Later rows |',
                 '|---|---|---:|---:|---:|---:|']
        for metrics in payload['metrics'].values():
            for target,m in metrics.items():
                o,t=m['oof'],m['temporal_test']
                mode='accepted' if m['accepted'] else 'baseline'
                if target.endswith('_tds') and m['accepted']:
                    mode='P(any TD) head'
                lines.append(f"| {target} | {mode} | {o['mae']:.3f} / {o['baseline_mae']:.3f} | {t.get('mae',0):.3f} / {t.get('baseline_mae',0):.3f} | {t.get('brier',0):.4f} / {t.get('baseline_brier',0):.4f} | {t['rows']} |")
        lines += ['', 'Legacy workload/spike artifacts are retained for research, but do not overwrite this release.',
                  'Forecast acceptance is not bankroll approval. Clean future locks, outcomes, and CLV are still required.', '']
        text='\n'.join(lines)
        cfg.out_file.parent.mkdir(parents=True,exist_ok=True)
        cfg.out_file.write_text(text,encoding='utf-8')
        return text
    player = _load_json(cfg.player_report)
    opportunity = _load_json(cfg.opportunity_report)
    distribution = _load_json(cfg.distribution_report)
    opportunity_distribution = _load_json(cfg.opportunity_distribution_report)
    projection_audit = _load_json(cfg.projection_audit)
    receiver_spike_diagnostic = _load_json(cfg.receiver_spike_diagnostic)
    repair_queue = _load_json(cfg.repair_queue)
    exact_line_proof = _load_json(cfg.exact_line_proof)
    exact_line_models = _load_json(cfg.exact_line_models)
    game = _load_json(cfg.game_report)
    lines = [
        "# NFL Model Holdout Report",
        "",
        "This report compares current NFL projection models against rolling, role, position/depth, and market/environment baselines where available.",
        "Brier score is not available yet because NFL line-level historical prediction rows have not been locked and graded.",
        "Once live odds snapshots and grades exist, Brier/ROI/CLV should be added from `bets.nfl_*_prediction_results` and `bets.nfl_prediction_clv`.",
        "",
        f"- Player artifact status: {player.get('status', 'unknown')} | rows: {player.get('rows', 0)}",
        f"- Opportunity artifact status: {opportunity.get('status', 'unknown')} | rows: {opportunity.get('rows', 0)}",
        f"- Distribution artifact status: {distribution.get('status', 'unknown')} | rows: {distribution.get('rows', 0)}",
        f"- Opportunity distribution status: {opportunity_distribution.get('status', 'unknown')} | rows: {opportunity_distribution.get('rows', 0)}",
        f"- Projection audit status: {projection_audit.get('status', 'unknown')} | rows: {projection_audit.get('rows', 0)}",
        f"- Receiver spike diagnostic status: {receiver_spike_diagnostic.get('status', 'unknown')} | rows: {receiver_spike_diagnostic.get('rows', 0)}",
        f"- Repair queue status: {repair_queue.get('status', 'unknown')} | items: {len(repair_queue.get('items') or [])}",
        f"- Exact-line proof status: {exact_line_proof.get('status', 'unknown')} | prediction rows: {exact_line_proof.get('prediction_rows', 0)}",
        f"- Exact-line model status: {exact_line_models.get('status', 'unknown')} | true-paired rows: {exact_line_models.get('true_paired_rows', 0)}",
        f"- Game artifact status: {game.get('status', 'unknown')} | rows: {game.get('rows', 0)}",
        "",
        *_section("Player Stat Models", player),
        "",
        *_section("Player Opportunity Models", opportunity),
        "",
        *_opportunity_distribution_section(opportunity_distribution),
        "",
        *_distribution_section(distribution),
        "",
        *_projection_audit_section(projection_audit),
        "",
        *_receiver_spike_section(receiver_spike_diagnostic),
        "",
        *_repair_queue_section(repair_queue),
        "",
        *_exact_line_section(exact_line_proof),
        "",
        *_exact_line_model_section(exact_line_models),
        "",
        *_section("Game Models", game),
        "",
    ]
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    cfg.out_file.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write NFL model holdout report")
    parser.add_argument("--player-report", default=str(DEFAULT_PLAYER_REPORT))
    parser.add_argument("--opportunity-report", default=str(DEFAULT_OPPORTUNITY_REPORT))
    parser.add_argument("--distribution-report", default=str(DEFAULT_DISTRIBUTION_REPORT))
    parser.add_argument("--opportunity-distribution-report", default=str(DEFAULT_OPPORTUNITY_DISTRIBUTION_REPORT))
    parser.add_argument("--projection-audit", default=str(DEFAULT_PROJECTION_AUDIT))
    parser.add_argument("--receiver-spike-diagnostic", default=str(DEFAULT_RECEIVER_SPIKE_DIAGNOSTIC))
    parser.add_argument("--repair-queue", default=str(DEFAULT_REPAIR_QUEUE))
    parser.add_argument("--exact-line-proof", default=str(DEFAULT_EXACT_LINE_PROOF))
    parser.add_argument("--exact-line-models", default=str(DEFAULT_EXACT_LINE_MODELS))
    parser.add_argument("--game-report", default=str(DEFAULT_GAME_REPORT))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    text = build_report(ReportConfig(
        player_report=Path(args.player_report),
        opportunity_report=Path(args.opportunity_report),
        distribution_report=Path(args.distribution_report),
        opportunity_distribution_report=Path(args.opportunity_distribution_report),
        projection_audit=Path(args.projection_audit),
        receiver_spike_diagnostic=Path(args.receiver_spike_diagnostic),
        repair_queue=Path(args.repair_queue),
        exact_line_proof=Path(args.exact_line_proof),
        exact_line_models=Path(args.exact_line_models),
        game_report=Path(args.game_report),
        out_file=Path(args.out_file),
    ))
    print(text)


if __name__ == "__main__":
    main()
