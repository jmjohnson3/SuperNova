"""One post-settlement audit for projections, probabilities, prices and CLV."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json
from nfl_pipeline.modeling.evaluation import probability_metrics, projection_metrics
from nfl_pipeline.modeling import distribution_policy as dp


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = "nfl-post-settlement-evidence-v1"


def load_rows(day: date) -> pd.DataFrame:
    with psycopg2.connect(PG_DSN) as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        conn.set_session(readonly=True)
        cur.execute("SET LOCAL statement_timeout='90s'")
        cur.execute(
            """
            SELECT p.id,p.game_id,p.player_id,p.player_name,p.stat,p.projection::float,
                   p.baseline_projection::float,p.line::float,p.side,p.price,
                   p.probability::float,p.market_no_vig_probability::float,
                   p.projection_p10::float,p.projection_p90::float,p.model_version,
                   p.created_at_utc,p.forecast_payload,r.actual_stat::float AS actual,r.result,
                   r.profit_per_unit::float,c.clv_status,c.clv_prob_delta::float AS clv,
                   c.valid_close_snapshot_captured,c.close_quality_reason,g.season,g.week,
                   EXISTS (
                     SELECT 1 FROM bets.nfl_bet_ledger l
                     WHERE l.source_kind='prop' AND l.prediction_id=p.id
                       AND (l.tier IN ('micro_projection','starter','bankroll') OR l.stake>0)
                   ) AS selected
            FROM bets.nfl_player_prop_predictions p
            JOIN raw.nfl_games g USING (game_id)
            LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id=p.id
            LEFT JOIN bets.nfl_prediction_clv c
              ON c.source_kind='prop' AND c.prediction_id=p.id
            WHERE p.game_date_et=%s AND p.book='fanduel'
              AND p.integrity_version='nfl-asof-v2'
              AND p.line IS NOT NULL AND p.side IN ('over','under')
            ORDER BY p.created_at_utc,p.id
            """,
            (day,),
        )
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def score(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"rows": 0}
    frame = rows.copy()
    frame["weight"] = 1.0 / frame.groupby(["game_id", "player_id"]).id.transform("size")
    binary = frame[frame.result.isin(["win", "loss"])].copy()
    outcome = binary.result.eq("win").astype(float)
    model = probability_metrics(binary.probability, outcome, binary.weight)
    market = probability_metrics(binary.market_no_vig_probability, outcome, binary.weight)
    player_games = (
        frame.sort_values(["created_at_utc", "id"])
        .drop_duplicates(["game_id", "player_id", "stat"], keep="last")
        .dropna(subset=["actual"])
    )
    projection = projection_metrics(
        player_games.actual,
        player_games.projection,
        player_games.projection_p10,
        player_games.projection_p90,
    )
    baseline = projection_metrics(player_games.actual, player_games.baseline_projection)
    closed = frame[frame.clv_status.notna()]
    valid = frame[frame.clv_status.eq("valid_close") & frame.clv.notna()]
    valid_weight = valid.weight if len(valid) else pd.Series(dtype=float)
    close_rate = (
        float(np.average(closed.clv_status.eq("valid_close"), weights=closed.weight))
        if len(closed)
        else None
    )
    clv = {
        "valid_close_rate": close_rate,
        "valid_rows": int(len(valid)),
        "beat_rate": float(np.average(valid.clv.gt(0), weights=valid_weight)) if len(valid) else None,
        "average_probability_delta": float(np.average(valid.clv, weights=valid_weight)) if len(valid) else None,
        "unknown_reasons": dict(Counter(frame.loc[~frame.clv_status.eq("valid_close"), "close_quality_reason"].fillna("missing_close_label"))),
    }
    graded = frame[frame.result.isin(["win", "loss", "push"])].drop_duplicates("id")
    profits = graded.profit_per_unit
    if profits.isna().any():
        wins = graded.result.eq("win")
        losses = graded.result.eq("loss")
        price = graded.price.astype(float)
        fallback = np.where(wins, np.where(price > 0, price / 100, 100 / price.abs()), np.where(losses, -1.0, 0.0))
        profits = profits.fillna(pd.Series(fallback, index=graded.index))
    return _json_safe(
        {
            "rows": int(len(frame)),
            "settled_binary_rows": int(len(binary)),
            "player_games": int(len(player_games)),
            "weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
            "exact_line_results": dict(Counter(frame.result.fillna("pending"))),
            "projection": projection,
            "baseline_projection": baseline,
            "model_probability": model,
            "fanduel_no_vig": market,
            "interval_coverage_80": projection.get("coverage_80"),
            "flat_roi": float(profits.mean()) if len(profits) else None,
            "clv": clv,
        }
    )


def policy_evidence() -> dict[str, Any]:
    path = ROOT / "reports" / "nfl_distribution_policy_latest.json"
    if not path.exists():
        return {"status": "missing_distribution_policy_report", "markets": {}}
    report = json.loads(path.read_text(encoding="utf-8"))
    active = dp.load_registry("active")
    markets: dict[str, list[dict[str, Any]]] = {}
    for key, review in (report.get("prospective") or {}).items():
        market = review.get("market") or key.split("|")[3]
        markets.setdefault(market, []).append(
            {
                "policy_key": key,
                "source": key.split("|", 5)[-1],
                "production": review.get("production"),
                "challenger": review.get("challenger"),
                "fanduel_no_vig": review.get("market_baseline"),
                "brier_gain": review.get("brier_gain"),
                "production_coverage_80": review.get("production_coverage"),
                "blockers": review.get("blockers") or [],
                "selection_blockers": review.get("selection_blockers") or [],
                "forecast_deployed": key in active,
                "cash_approved": False,
            }
        )
    return {"status": "evaluated", "markets": markets}


def build(day: date, rows: pd.DataFrame) -> dict[str, Any]:
    by_stat = {
        stat: {
            "all_offers": score(group),
            "selected_trials": score(group[group.selected]),
        }
        for stat, group in rows.groupby("stat")
    }
    return {
        "contract": CONTRACT,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "date": str(day),
        "fanduel_only": True,
        "stats": by_stat,
        "probability_policies": policy_evidence(),
        "betting_approval": {
            "automatic": False,
            "micro_contract": "$1 flat singles; maximum three; no quota; no parlays",
            "note": "Forecast deployment and cash approval are separate.",
        },
    }


def render(doc: dict[str, Any]) -> str:
    lines = [
        "# NFL Post-Settlement Evidence",
        "",
        f"Date: {doc['date']}. FanDuel offers only.",
        "",
        "| Stat / scope | Rows | MAE | Bias | Brier model / FanDuel | Calibration model / FanDuel | 80% coverage | ROI | Valid close | Avg CLV |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for stat, scopes in sorted(doc["stats"].items()):
        for name, metric in scopes.items():
            projection = metric.get("projection") or {}
            model = metric.get("model_probability") or {}
            market = metric.get("fanduel_no_vig") or {}
            clv = metric.get("clv") or {}
            fmt = lambda value, digits=3: "-" if value is None else f"{value:.{digits}f}"
            pct = lambda value: "-" if value is None else f"{value:.1%}"
            lines.append(
                f"| {stat} / {name} | {metric.get('rows', 0)} | {fmt(projection.get('mae'))} | "
                f"{fmt(projection.get('bias'))} | {fmt(model.get('brier'))} / {fmt(market.get('brier'))} | "
                f"{pct(model.get('calibration_error'))} / {pct(market.get('calibration_error'))} | "
                f"{pct(metric.get('interval_coverage_80'))} | {pct(metric.get('flat_roi'))} | "
                f"{pct(clv.get('valid_close_rate'))} | {pct(clv.get('average_probability_delta'))} |"
            )
    lines += ["", "## Independent Policy Decisions", ""]
    for market, policies in sorted(doc["probability_policies"]["markets"].items()):
        for policy in policies:
            state = "deployed forecast-only" if policy["forecast_deployed"] else "blocked"
            lines.append(f"- {market} | {policy['source']} | {state}: {', '.join(policy['blockers']) or 'all forecast gates passed'}")
    lines += ["", "No policy in this report authorizes bankroll betting. $1 micro remains a separate execution contract.", ""]
    return "\n".join(lines)


def report(day: date) -> dict[str, Any]:
    doc = build(day, load_rows(day))
    for suffix in (str(day), "latest"):
        atomic_json(ROOT / "reports" / f"nfl_settlement_evidence_{suffix}.json", doc)
        (ROOT / "reports" / f"nfl_settlement_evidence_{suffix}.md").write_text(
            render(doc), encoding="utf-8"
        )
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    doc = report(args.date)
    print(json.dumps({"date": doc["date"], "stats": list(doc["stats"]), "policy_status": doc["probability_policies"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
