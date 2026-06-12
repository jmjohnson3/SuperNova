"""Audit why hitter prop opportunity is not confirming.

The goal is to separate input failures from real projection failures:

* no player-game join
* no confirmed lineup slot
* missing or low projected PA
* projected PA miss after grading
* bottom-order PA overprojection
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2

_PG_DSN = "postgresql://josh:password@localhost:5432/nba"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_HITTER_MARKETS = ("batter_hits", "batter_total_bases", "batter_home_runs")


@dataclass(frozen=True)
class OpportunityFailureAuditConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 60
    active_days: int = 7
    min_projected_pa: float = 3.2
    bad_pa_error: float = 1.25
    bottom_order_overproject_error: float = 0.75
    report_file: str | None = None
    json_file: str = "prop_opportunity_failure_audit.json"


TRAINING_SQL = """
SELECT
    'graded_training' AS section,
    e.game_date_et,
    e.game_slug,
    e.player_id,
    e.player_name,
    e.player_name_norm,
    e.team_abbr,
    e.market,
    e.side,
    e.bookmaker_key,
    e.market_line,
    COALESCE(pg.lineup_slot, e.confirmed_batting_order)::float AS lineup_slot,
    COALESCE(pg.lineup_source, e.confirmed_lineup_source) AS lineup_source,
    pg.starter_status_source,
    pg.confirmed_starter,
    COALESCE(pg.projected_pa, e.projected_pa)::float AS projected_pa,
    COALESCE(pg.actual_pa, e.actual_pa)::float AS actual_pa,
    e.pred_count::float AS pred_count,
    e.actual_value::float AS actual_value,
    e.won,
    e.profit_units::float AS profit_units
FROM features.mlb_prop_market_training_examples e
LEFT JOIN features.mlb_hitter_player_game_training pg
  ON pg.game_slug = e.game_slug
 AND pg.player_id = e.player_id
WHERE e.game_date_et >= %(training_cutoff)s
  AND e.market IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
  AND e.result_status = 'graded'
ORDER BY e.game_date_et DESC, e.market, e.player_name
"""


ACTIVE_SQL = """
WITH active AS (
    SELECT
        r.*,
        TRIM(LOWER(REGEXP_REPLACE(COALESCE(r.player_name, ''), '[^a-z0-9]+', ' ', 'g'))) AS active_name_norm
    FROM bets.mlb_prop_prediction_replay r
    WHERE r.game_date_et >= %(active_cutoff)s
      AND r.stat IN ('batter_hits', 'batter_total_bases', 'batter_home_runs')
      AND r.actual_value IS NULL
)
SELECT
    'active_pending' AS section,
    r.game_date_et,
    r.game_slug,
    r.player_id,
    r.player_name,
    COALESCE(r.player_name_norm, r.active_name_norm) AS player_name_norm,
    r.team_abbr,
    r.stat AS market,
    r.side,
    r.bookmaker_key,
    r.market_line,
    COALESCE(lu_id.batting_order, lu_name.batting_order)::float AS lineup_slot,
    COALESCE(lu_id.lineup_source, lu_name.lineup_source) AS lineup_source,
    CASE
        WHEN lu_id.batting_order IS NOT NULL THEN 'raw_lineups_player_id'
        WHEN lu_name.batting_order IS NOT NULL THEN 'raw_lineups_name_match'
        ELSE NULL
    END AS starter_status_source,
    CASE WHEN COALESCE(lu_id.batting_order, lu_name.batting_order) BETWEEN 1 AND 9 THEN TRUE ELSE FALSE END AS confirmed_starter,
    hist_pa.projected_pa,
    NULL::float AS actual_pa,
    r.pred_count::float AS pred_count,
    NULL::float AS actual_value,
    NULL::boolean AS won,
    NULL::float AS profit_units
FROM active r
LEFT JOIN raw.mlb_lineups lu_id
  ON lu_id.game_slug = r.game_slug
 AND lu_id.team_abbr = r.team_abbr
 AND lu_id.player_id = r.player_id
LEFT JOIN raw.mlb_lineups lu_name
  ON lu_name.game_slug = r.game_slug
 AND lu_name.team_abbr = r.team_abbr
 AND lu_name.player_name_norm = COALESCE(r.player_name_norm, r.active_name_norm)
LEFT JOIN LATERAL (
    SELECT
        AVG(pa_est)::float AS projected_pa,
        COUNT(*)::int AS pa_games
    FROM (
        SELECT GREATEST(COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0), 0) AS pa_est
        FROM raw.mlb_player_gamelogs gl
        JOIN raw.mlb_games g ON g.game_slug = gl.game_slug
        WHERE gl.player_id = r.player_id
          AND gl.team_abbr = r.team_abbr
          AND g.status = 'final'
          AND g.game_date_et < r.game_date_et
          AND (COALESCE(gl.at_bats, 0) + COALESCE(gl.walks_batter, 0)) > 0
        ORDER BY g.game_date_et DESC, gl.game_slug DESC
        LIMIT 10
    ) recent_pa
) hist_pa ON TRUE
ORDER BY r.game_date_et DESC, r.stat, r.player_name
"""


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_schema = %s AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def _query_df(conn, sql: str, params: dict[str, Any]) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _load(cfg: OpportunityFailureAuditConfig) -> pd.DataFrame:
    training_cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.lookback_days)
    active_cutoff = datetime.now(timezone.utc).date() - timedelta(days=cfg.active_days)
    params = {"training_cutoff": training_cutoff, "active_cutoff": active_cutoff}
    with psycopg2.connect(cfg.pg_dsn) as conn:
        frames: list[pd.DataFrame] = []
        if (
            _table_exists(conn, "features", "mlb_prop_market_training_examples")
            and _table_exists(conn, "features", "mlb_hitter_player_game_training")
        ):
            frames.append(_query_df(conn, TRAINING_SQL, params))
        if _table_exists(conn, "bets", "mlb_prop_prediction_replay") and _table_exists(conn, "raw", "mlb_lineups"):
            frames.append(_query_df(conn, ACTIVE_SQL, params))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    df["game_date_et"] = pd.to_datetime(df["game_date_et"]).dt.date
    for col in ("lineup_slot", "projected_pa", "actual_pa", "pred_count", "actual_value", "market_line", "profit_units"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([float("inf"), float("-inf")], pd.NA)


def _clean_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _classify(row: pd.Series, cfg: OpportunityFailureAuditConfig) -> list[str]:
    reasons: list[str] = []
    section = str(row.get("section") or "")
    slot = _clean_float(row.get("lineup_slot"))
    projected_pa = _clean_float(row.get("projected_pa"))
    actual_pa = _clean_float(row.get("actual_pa"))
    player_id = row.get("player_id")
    source_value = row.get("lineup_source")
    lineup_source = "" if source_value is None or pd.isna(source_value) else str(source_value)

    if player_id is None or pd.isna(player_id):
        reasons.append("missing_player_id")
    if slot is None:
        reasons.append("missing_confirmed_lineup_slot")
    if section == "active_pending" and not lineup_source:
        reasons.append("no_live_lineup_source")
    if projected_pa is None:
        reasons.append("missing_projected_pa")
    elif projected_pa < cfg.min_projected_pa:
        reasons.append("projected_pa_below_gate")

    if actual_pa is not None and projected_pa is not None:
        error = actual_pa - projected_pa
        if abs(error) >= cfg.bad_pa_error:
            reasons.append("bad_pa_projection")
        if slot is not None and slot >= 8 and projected_pa - actual_pa >= cfg.bottom_order_overproject_error:
            reasons.append("bottom_order_overprojected")
    if actual_pa is not None and actual_pa <= 2:
        reasons.append("low_actual_pa")

    return sorted(dict.fromkeys(reasons or ["ok_opportunity"]))


def _reason_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    section_counts: Counter[tuple[str, str]] = Counter()
    market_counts: Counter[tuple[str, str, str]] = Counter()
    for _, row in df.iterrows():
        reasons = row["opportunity_reasons"]
        section = str(row.get("section") or "unknown")
        market = str(row.get("market") or "unknown")
        side = str(row.get("side") or "unknown")
        for reason in reasons:
            counts[reason] += 1
            section_counts[(section, reason)] += 1
            market_counts[(market, side, reason)] += 1
    return [
        {"reason": reason, "rows": rows}
        for reason, rows in counts.most_common()
    ], [
        {"section": section, "reason": reason, "rows": rows}
        for (section, reason), rows in sorted(section_counts.items(), key=lambda x: (-x[1], x[0]))
    ], [
        {"market": market, "side": side, "reason": reason, "rows": rows}
        for (market, side, reason), rows in sorted(market_counts.items(), key=lambda x: (-x[1], x[0]))[:60]
    ]


def _market_summary(df: pd.DataFrame, cfg: OpportunityFailureAuditConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if df.empty:
        return out
    for keys, group in df.groupby(["section", "market", "side"], dropna=False):
        section, market, side = keys
        projected = pd.to_numeric(group["projected_pa"], errors="coerce")
        actual = pd.to_numeric(group["actual_pa"], errors="coerce")
        slot = pd.to_numeric(group["lineup_slot"], errors="coerce")
        pa_mask = projected.notna() & actual.notna()
        out.append({
            "section": section,
            "market": market,
            "side": side,
            "rows": int(len(group)),
            "lineup_slot_rate": float(slot.notna().mean()) if len(group) else None,
            "projected_pa_rate": float(projected.notna().mean()) if len(group) else None,
            "avg_projected_pa": float(projected.mean()) if projected.notna().any() else None,
            "avg_actual_pa": float(actual.mean()) if actual.notna().any() else None,
            "pa_mae": float((actual[pa_mask] - projected[pa_mask]).abs().mean()) if pa_mask.any() else None,
            "bad_pa_projection_rate": float(((actual[pa_mask] - projected[pa_mask]).abs() >= cfg.bad_pa_error).mean()) if pa_mask.any() else None,
            "low_projected_pa_rate": float((projected < cfg.min_projected_pa).mean()) if projected.notna().any() else None,
            "low_actual_pa_rate": float((actual <= 2).mean()) if actual.notna().any() else None,
        })
    return sorted(out, key=lambda r: (r["section"], r["market"], r["side"]))


def _examples(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df[df["opportunity_reasons"].map(lambda reasons: reasons != ["ok_opportunity"])].copy()
    if work.empty:
        return []
    work["reason_text"] = work["opportunity_reasons"].map(lambda reasons: ", ".join(reasons))
    work = work.drop_duplicates(
        subset=["section", "game_date_et", "player_name", "team_abbr", "market", "side", "reason_text"],
        keep="first",
    )
    cols = [
        "section", "game_date_et", "player_name", "team_abbr", "market", "side",
        "lineup_slot", "lineup_source", "projected_pa", "actual_pa", "pred_count",
        "actual_value", "reason_text",
    ]
    return [
        {col: _json_safe(row.get(col)) for col in cols}
        for _, row in work.sort_values(["section", "game_date_et"], ascending=[True, False]).head(80).iterrows()
    ]


def run_opportunity_failure_audit(cfg: OpportunityFailureAuditConfig) -> dict[str, Any]:
    df = _load(cfg)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "status": "ok" if len(df) else "no_rows",
    }
    if df.empty:
        _write_outputs(payload, cfg)
        return payload
    df["opportunity_reasons"] = [_classify(row, cfg) for _, row in df.iterrows()]
    counts, by_section, by_market = _reason_counts(df)
    payload.update({
        "reason_counts": counts,
        "reason_counts_by_section": by_section,
        "reason_counts_by_market_side": by_market,
        "market_summary": _market_summary(df, cfg),
        "examples": _examples(df),
    })
    _write_outputs(payload, cfg)
    return payload


def _write_outputs(payload: dict[str, Any], cfg: OpportunityFailureAuditConfig) -> None:
    cfg_path = _MODEL_DIR / cfg.json_file
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")
    _write_report(payload, cfg.report_file)


def _fmt_pct(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{100.0 * float(value):.1f}%"
    except Exception:
        return "-"


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _write_report(payload: dict[str, Any], report_file: str | None) -> None:
    path = _REPORT_DIR / (report_file or "mlb_prop_opportunity_failure_audit_latest.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MLB Prop Opportunity Failure Audit",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        f"Rows scanned: {payload.get('rows', 0)}",
        f"Status: {payload.get('status')}",
        "",
        "## Reason Counts",
        "",
        "| Reason | Rows |",
        "|---|---:|",
    ]
    for rec in payload.get("reason_counts", [])[:30]:
        lines.append(f"| {rec.get('reason')} | {rec.get('rows')} |")
    lines.extend([
        "",
        "## By Section",
        "",
        "| Section | Reason | Rows |",
        "|---|---|---:|",
    ])
    for rec in payload.get("reason_counts_by_section", [])[:40]:
        lines.append(f"| {rec.get('section')} | {rec.get('reason')} | {rec.get('rows')} |")
    lines.extend([
        "",
        "## Market Summary",
        "",
        "| Section | Market | Side | Rows | Slot | Proj PA | Avg Proj PA | Avg Actual PA | PA MAE | Bad PA | Low Proj PA | Low Actual PA |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rec in payload.get("market_summary", []):
        lines.append(
            f"| {rec.get('section')} | {rec.get('market')} | {rec.get('side')} | {rec.get('rows')} | "
            f"{_fmt_pct(rec.get('lineup_slot_rate'))} | {_fmt_pct(rec.get('projected_pa_rate'))} | "
            f"{_fmt_num(rec.get('avg_projected_pa'))} | {_fmt_num(rec.get('avg_actual_pa'))} | "
            f"{_fmt_num(rec.get('pa_mae'))} | {_fmt_pct(rec.get('bad_pa_projection_rate'))} | "
            f"{_fmt_pct(rec.get('low_projected_pa_rate'))} | {_fmt_pct(rec.get('low_actual_pa_rate'))} |"
        )
    lines.extend([
        "",
        "## Example Rows",
        "",
        "| Section | Date | Player | Market | Side | Slot | Source | Proj PA | Actual PA | Reasons |",
        "|---|---|---|---|---|---:|---|---:|---:|---|",
    ])
    for rec in payload.get("examples", [])[:40]:
        lines.append(
            f"| {rec.get('section')} | {rec.get('game_date_et')} | {rec.get('player_name')} | "
            f"{rec.get('market')} | {rec.get('side')} | {_fmt_num(rec.get('lineup_slot'), 0)} | "
            f"{rec.get('lineup_source') or ''} | {_fmt_num(rec.get('projected_pa'))} | "
            f"{_fmt_num(rec.get('actual_pa'))} | {rec.get('reason_text')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        import decimal

        if isinstance(value, decimal.Decimal):
            return float(value)
    except Exception:
        pass
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit hitter prop opportunity failures.")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--active-days", type=int, default=7)
    parser.add_argument("--report-file")
    parser.add_argument("--json-file", default="prop_opportunity_failure_audit.json")
    args = parser.parse_args()
    cfg = OpportunityFailureAuditConfig(
        pg_dsn=args.pg_dsn,
        lookback_days=args.lookback_days,
        active_days=args.active_days,
        report_file=args.report_file,
        json_file=args.json_file,
    )
    print(json.dumps(run_opportunity_failure_audit(cfg), indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
