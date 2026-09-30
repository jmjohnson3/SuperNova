"""Report NFL exact-line prop betting proof once books publish prop offers."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN
from nfl_pipeline.integrity import atomic_json

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
DEFAULT_JSON = MODEL_DIR / "nfl_prop_exact_line_proof.json"
DEFAULT_MD = ROOT / "reports" / "nfl_prop_exact_line_proof_latest.md"


@dataclass(frozen=True)
class ExactLineProofConfig:
    pg_dsn: str = PG_DSN
    game_date: date | None = None
    out_file: Path = DEFAULT_JSON
    md_report_file: Path = DEFAULT_MD


def _rows(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return "-"
    return f"{out:.{digits}f}"


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "0.0%"


def _price_bucket(price: Any) -> str:
    try:
        val = int(price)
    except (TypeError, ValueError):
        return "missing_price"
    if val <= -200:
        return "lay_200_plus"
    if val <= -150:
        return "lay_150_199"
    if val < 100:
        return "fair_lay_100_149"
    if val < 150:
        return "plus_100_149"
    if val < 250:
        return "plus_150_249"
    return "plus_250_plus"


def _line_bucket(stat: str, line: Any) -> str:
    try:
        val = float(line)
    except (TypeError, ValueError):
        return "missing_line"
    if stat.endswith("_tds"):
        return "td_0_5" if val <= 0.5 else "td_alt_1_5_plus"
    if stat == "passing_yards":
        return f"pass_yards_{int(val // 25 * 25)}_{int(val // 25 * 25 + 24)}"
    if stat in {"rushing_yards", "receiving_yards"}:
        return f"yards_{int(val // 10 * 10)}_{int(val // 10 * 10 + 9)}"
    return f"line_{val:g}"


def evidence_status(offers, predictions, buckets):
    if not offers and not predictions:
        return 'waiting_for_props'
    if not any(int(b.get('graded_rows') or 0) for b in buckets):
        return 'waiting_for_settlement'
    if not any(int(b.get('valid_clv_rows') or 0) for b in buckets):
        return 'waiting_for_valid_clv'
    return 'evidence_collected_not_bankroll_approval'


def build_report(cfg: ExactLineProofConfig) -> dict[str, Any]:
    params = {"game_date": cfg.game_date}
    with psycopg2.connect(cfg.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("SET LOCAL statement_timeout = '90s'")
        offer_summary = _rows(
            conn,
            """
            SELECT
                stat,
                bookmaker_key AS book,
                COUNT(*) AS rows,
                COUNT(*) FILTER (WHERE over_price IS NOT NULL AND under_price IS NOT NULL) AS true_paired_rows,
                COUNT(*) FILTER (WHERE snapshot_role = 'lock') AS lock_rows,
                COUNT(*) FILTER (WHERE snapshot_role = 'close') AS close_rows
            FROM odds.nfl_player_prop_lines
            WHERE (%(game_date)s IS NULL OR as_of_date = %(game_date)s)
              AND stat IS NOT NULL
            GROUP BY stat, bookmaker_key
            ORDER BY stat, bookmaker_key
            """,
            params,
        )
        prediction_rows = _rows(
            conn,
            """
            SELECT
                p.id,
                p.game_date_et,
                p.stat,
                p.side,
                p.book,
                p.line::float AS line,
                p.price,
                p.model_version,
                p.probability::float AS probability,
                p.ev::float AS ev,
                r.result,
                r.profit_per_unit::float AS profit_per_unit,
                c.clv_status,
                c.clv_prob_delta::float AS clv_prob_delta
            FROM bets.nfl_player_prop_predictions p
            LEFT JOIN bets.nfl_player_prop_prediction_results r ON r.prediction_id = p.id
            LEFT JOIN bets.nfl_prediction_clv c
              ON c.source_kind = 'prop'
             AND c.prediction_id = p.id
            WHERE (%(game_date)s IS NULL OR p.game_date_et = %(game_date)s)
              AND p.side IN ('over', 'under')
              AND p.line IS NOT NULL
              AND p.integrity_version='nfl-asof-v2'
              AND EXISTS (SELECT 1 FROM odds.nfl_player_prop_lines o WHERE o.id=p.offer_id AND o.over_price IS NOT NULL AND o.under_price IS NOT NULL AND o.fetched_at_utc<=p.created_at_utc)
              AND NOT EXISTS (SELECT 1 FROM bets.nfl_player_prop_predictions earlier
                  WHERE earlier.game_id=p.game_id AND earlier.player_id=p.player_id AND earlier.stat=p.stat
                    AND earlier.side=p.side AND earlier.book=p.book AND earlier.line=p.line
                    AND earlier.integrity_version=p.integrity_version AND earlier.id<p.id)
            """,
            params,
        )

    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in prediction_rows:
        stat = str(row.get("stat") or "")
        key = (
            stat,
            str(row.get("side") or ""),
            str(row.get("book") or ""),
            _line_bucket(stat, row.get("line")),
            _price_bucket(row.get("price")),
            str(row.get("model_version") or ""),
        )
        rec = buckets.setdefault(key, {
            "stat": key[0],
            "side": key[1],
            "book": key[2],
            "line_bucket": key[3],
            "price_bucket": key[4],
            "model_version": key[5],
            "rows": 0,
            "graded_rows": 0,
            "wins": 0,
            "roi_sum": 0.0,
            "brier_sum": 0.0,
            "brier_rows": 0,
            "valid_clv_rows": 0,
            "clv_beats": 0,
            "clv_sum": 0.0,
        })
        rec["rows"] += 1
        result = str(row.get("result") or "").lower()
        profit = row.get("profit_per_unit")
        if result in {"win", "loss", "push"}:
            rec["graded_rows"] += 1
            rec["wins"] += 1 if result == "win" else 0
            rec["roi_sum"] += float(profit or 0.0)
            prob = row.get("probability")
            if prob is not None and result in {"win", "loss"}:
                actual = 1.0 if result == "win" else 0.0
                rec["brier_sum"] += (float(prob) - actual) ** 2
                rec["brier_rows"] += 1
        if row.get("clv_status") == "valid_close" and row.get("clv_prob_delta") is not None:
            delta = float(row["clv_prob_delta"])
            rec["valid_clv_rows"] += 1
            rec["clv_beats"] += 1 if delta > 0 else 0
            rec["clv_sum"] += delta

    exact_buckets: list[dict[str, Any]] = []
    for rec in buckets.values():
        graded = int(rec["graded_rows"])
        clv_rows = int(rec["valid_clv_rows"])
        out = dict(rec)
        out["win_rate"] = (rec["wins"] / graded) if graded else None
        out["roi"] = (rec["roi_sum"] / graded) if graded else None
        out["brier"] = (rec["brier_sum"] / rec["brier_rows"]) if rec["brier_rows"] else None
        out["clv_beat_rate"] = (rec["clv_beats"] / clv_rows) if clv_rows else None
        out["avg_clv_prob_delta"] = (rec["clv_sum"] / clv_rows) if clv_rows else None
        exact_buckets.append(out)
    exact_buckets.sort(
        key=lambda rec: (
            int(rec.get("graded_rows") or 0),
            float(rec.get("roi") or -99),
            float(rec.get("avg_clv_prob_delta") or -99),
        ),
        reverse=True,
    )

    payload = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": evidence_status(offer_summary, prediction_rows, exact_buckets),
        "game_date": cfg.game_date.isoformat() if cfg.game_date else None,
        "offer_summary": offer_summary,
        "prediction_rows": len(prediction_rows),
        "exact_buckets": exact_buckets,
    }
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(cfg.out_file, payload)
    _write_markdown(payload, cfg.md_report_file)
    return payload


def _write_markdown(payload: dict[str, Any], path: Path) -> str:
    lines = [
        "# NFL Exact-Line Prop Proof",
        "",
        "This report stays empty until books publish NFL prop lines. Once they do, it audits exact book/player/stat/line/side buckets before any bankroll label exists.",
        "",
        f"- Status: {payload.get('status')}",
        f"- Game date: {payload.get('game_date') or 'all'}",
        f"- Prediction rows: {payload.get('prediction_rows')}",
        "",
        "## Offer Coverage",
        "",
        "| Stat | Book | Rows | True Paired | Pair Coverage | Lock Rows | Close Rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    offers = payload.get("offer_summary") or []
    if offers:
        for row in offers:
            rows = int(row.get("rows") or 0)
            true_paired = int(row.get("true_paired_rows") or 0)
            lines.append(
                f"| {row.get('stat')} | {row.get('book')} | {rows} | {true_paired} | "
                f"{_pct(true_paired / rows if rows else 0)} | {int(row.get('lock_rows') or 0)} | {int(row.get('close_rows') or 0)} |"
            )
    else:
        lines.append("| none | - | 0 | 0 | 0.0% | 0 | 0 |")
    lines.extend([
        "",
        "## Exact Buckets",
        "",
        "| Stat | Side | Book | Line Bucket | Price Bucket | Model | Rows | Graded | Win% | ROI | Brier | CLV Rows | CLV Beat | Avg CLV |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    buckets = payload.get("exact_buckets") or []
    if buckets:
        for row in buckets[:80]:
            lines.append(
                f"| {row.get('stat')} | {row.get('side')} | {row.get('book')} | {row.get('line_bucket')} | "
                f"{row.get('price_bucket')} | {row.get('model_version')} | {int(row.get('rows') or 0)} | "
                f"{int(row.get('graded_rows') or 0)} | {_pct(row.get('win_rate') or 0)} | {_fmt(row.get('roi'))} | "
                f"{_fmt(row.get('brier'))} | {int(row.get('valid_clv_rows') or 0)} | {_pct(row.get('clv_beat_rate') or 0)} | "
                f"{_fmt(row.get('avg_clv_prob_delta'))} |"
            )
    else:
        lines.append("| none | - | - | - | - | - | 0 | 0 | 0.0% | - | - | 0 | 0.0% | - |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL exact-line prop proof report")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--out-file", default=str(DEFAULT_JSON))
    parser.add_argument("--md-report-file", default=str(DEFAULT_MD))
    args = parser.parse_args()
    payload = build_report(ExactLineProofConfig(
        pg_dsn=args.pg_dsn,
        game_date=date.fromisoformat(args.date) if args.date else None,
        out_file=Path(args.out_file),
        md_report_file=Path(args.md_report_file),
    ))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
