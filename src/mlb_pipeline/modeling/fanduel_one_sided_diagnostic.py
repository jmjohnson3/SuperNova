"""Trace FanDuel hitter sides from the raw API payload through model training."""
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

import psycopg2
import psycopg2.extras

from mlb_pipeline.atomic_io import atomic_write_json, atomic_write_text

from mlb_pipeline.db import PG_DSN as _PG_DSN
from mlb_pipeline.parse_oddsapi import _PROP_MARKET_MAP

_REPORT_DIR = Path(__file__).resolve().parents[3] / "reports"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "player_props"
_HITTER_MARKETS = {"batter_hits", "batter_total_bases", "batter_home_runs"}


@dataclass(frozen=True)
class FanDuelDiagnosticConfig:
    pg_dsn: str = _PG_DSN
    lookback_days: int = 30
    out: str = "mlb_fanduel_one_sided_diagnostic_latest.md"
    json_out: str = "fanduel_one_sided_diagnostic.json"
    example_limit: int = 40


def _normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text)).strip()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
        return cur.fetchone()[0] is not None


def _events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict) and "data" in payload:
        payload = payload.get("data")
    if isinstance(payload, dict):
        return [payload]
    return [item for item in (payload or []) if isinstance(item, dict)] if isinstance(payload, list) else []


def _raw_fanduel_groups(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Keep the latest raw outcome state for each event/player/market/line."""
    grouped: dict[tuple, dict[str, Any]] = {}
    for response in rows:
        as_of = response.get("as_of_date")
        fetched = response.get("fetched_at_utc")
        for event in _events(response.get("payload")):
            event_id = str(event.get("id") or "")
            for book in event.get("bookmakers") or []:
                if str(book.get("key") or "").lower() != "fanduel":
                    continue
                for market in book.get("markets") or []:
                    market_key = str(market.get("key") or "")
                    stat = _PROP_MARKET_MAP.get(market_key)
                    if stat not in _HITTER_MARKETS:
                        continue
                    for outcome in market.get("outcomes") or []:
                        side = str(outcome.get("name") or "").strip().lower()
                        player = str(outcome.get("description") or "").strip()
                        line = _float(outcome.get("point"))
                        price = _float(outcome.get("price"))
                        if side not in {"over", "under"} or not player or line is None or price is None:
                            continue
                        key = (as_of, event_id, _normalize_name(player), stat, float(line), market_key)
                        record = grouped.setdefault(
                            key,
                            {
                                "as_of_date": as_of,
                                "event_id": event_id,
                                "player_name": player,
                                "player_name_norm": _normalize_name(player),
                                "stat": stat,
                                "line": float(line),
                                "market_key": market_key,
                                "over_price": None,
                                "under_price": None,
                                "latest_fetched_at_utc": fetched,
                            },
                        )
                        previous = record.get("latest_fetched_at_utc")
                        if previous is not None and fetched is not None and fetched > previous:
                            record["over_price"] = None
                            record["under_price"] = None
                            record["latest_fetched_at_utc"] = fetched
                        if previous is None or fetched is None or fetched >= record.get("latest_fetched_at_utc"):
                            record[f"{side}_price"] = price
    return grouped


def _key(row: dict[str, Any]) -> tuple:
    return (
        row.get("as_of_date"),
        str(row.get("event_id") or ""),
        _normalize_name(row.get("player_name_norm") or row.get("player_name")),
        str(row.get("stat") or row.get("market") or ""),
        float(row.get("line") if row.get("line") is not None else row.get("market_line")),
    )


def _fetch_rows(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _load(cfg: FanDuelDiagnosticConfig) -> tuple[dict[tuple, dict[str, Any]], dict, dict, dict]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(1, cfg.lookback_days))
    with psycopg2.connect(cfg.pg_dsn) as conn:
        raw_responses = _fetch_rows(
            conn,
            """
            SELECT as_of_date, fetched_at_utc, payload
            FROM raw.api_responses
            WHERE provider = 'oddsapi'
              AND endpoint IN ('mlb_prop_odds', 'mlb_prop_odds_historical')
              AND as_of_date >= %(cutoff)s
            ORDER BY fetched_at_utc
            """,
            {"cutoff": cutoff},
        )
        raw = _raw_fanduel_groups(raw_responses)

        parsed_rows = _fetch_rows(
            conn,
            """
            SELECT as_of_date, COALESCE(event_id, '') AS event_id, player_name, player_name_norm,
                   stat, line::float AS line, over_price::float AS over_price,
                   under_price::float AS under_price, over_link, under_link
            FROM odds.mlb_player_prop_lines
            WHERE as_of_date >= %(cutoff)s
              AND LOWER(bookmaker_key) = 'fanduel'
              AND stat IN ('batter_hits','batter_total_bases','batter_home_runs')
            """,
            {"cutoff": cutoff},
        )
        parsed = {_key(row): row for row in parsed_rows}

        normalized: dict[tuple, dict[str, Any]] = {}
        if _table_exists(conn, "features", "mlb_prop_offer_links"):
            normalized_rows = _fetch_rows(
                conn,
                """
                SELECT as_of_date, COALESCE(event_id, '') AS event_id, player_name, player_name_norm,
                       stat, line::float AS line,
                       MAX(price::float) FILTER (WHERE side = 'over') AS over_price,
                       MAX(price::float) FILTER (WHERE side = 'under') AS under_price,
                       BOOL_OR(is_linkable) FILTER (WHERE side = 'over') AS over_linkable,
                       BOOL_OR(is_linkable) FILTER (WHERE side = 'under') AS under_linkable
                FROM features.mlb_prop_offer_links
                WHERE as_of_date >= %(cutoff)s
                  AND LOWER(bookmaker_key) = 'fanduel'
                  AND stat IN ('batter_hits','batter_total_bases','batter_home_runs')
                GROUP BY as_of_date, event_id, player_name, player_name_norm, stat, line
                """,
                {"cutoff": cutoff},
            )
            normalized = {_key(row): row for row in normalized_rows}

        training: dict[tuple, dict[str, Any]] = {}
        if _table_exists(conn, "features", "mlb_prop_market_training_examples"):
            training_rows = _fetch_rows(
                conn,
                """
                SELECT game_date_et AS as_of_date, COALESCE(game_slug, '') AS event_id,
                       player_name, player_name_norm, market AS stat, market_line::float AS line,
                       COUNT(*)::int AS rows,
                       BOOL_OR(pair_quality = 'same_book') AS used_same_book,
                       BOOL_OR(pair_quality = 'cross_book') AS used_cross_book,
                       BOOL_OR(pair_quality = 'synthetic') AS used_synthetic,
                       BOOL_OR(pair_quality = 'one_sided' OR paired_price IS NULL) AS used_one_sided
                FROM features.mlb_prop_market_training_examples
                WHERE game_date_et >= %(cutoff)s
                  AND LOWER(COALESCE(bookmaker_key, '')) = 'fanduel'
                  AND market IN ('batter_hits','batter_total_bases','batter_home_runs')
                GROUP BY game_date_et, game_slug, player_name, player_name_norm, market, market_line
                """,
                {"cutoff": cutoff},
            )
            # Training game_slug is not always the provider event id; retain a fallback key below.
            training = {_key(row): row for row in training_rows}
            for row in training_rows:
                fallback = (row["as_of_date"], "", _normalize_name(row.get("player_name_norm") or row.get("player_name")), row["stat"], float(row["line"]))
                training.setdefault(fallback, row)
    return raw, parsed, normalized, training


def _lookup(mapping: dict[tuple, dict[str, Any]], key: tuple) -> dict[str, Any] | None:
    return mapping.get(key) or mapping.get((key[0], "", key[2], key[3], key[4]))


def _analyze(cfg: FanDuelDiagnosticConfig) -> dict[str, Any]:
    raw, parsed, normalized, training = _load(cfg)
    causes: Counter[str] = Counter()
    market_keys: Counter[str] = Counter()
    by_stat: dict[str, Counter[str]] = defaultdict(Counter)
    by_date: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    for raw_key, row in raw.items():
        comparable = raw_key[:5]
        parsed_row = _lookup(parsed, comparable)
        normalized_row = _lookup(normalized, comparable)
        training_row = _lookup(training, comparable)
        raw_pair = row.get("over_price") is not None and row.get("under_price") is not None
        parsed_pair = bool(parsed_row and parsed_row.get("over_price") is not None and parsed_row.get("under_price") is not None)
        normalized_pair = bool(normalized_row and normalized_row.get("over_price") is not None and normalized_row.get("under_price") is not None)
        used_clean = bool(training_row and (training_row.get("used_same_book") or training_row.get("used_cross_book")))

        if not raw_pair:
            cause = "raw_api_one_sided"
        elif not parsed_pair:
            cause = "parser_dropped_true_under"
        elif not normalized_pair:
            cause = "normalizer_dropped_true_under"
        elif training_row is None:
            cause = "true_pair_not_replayed"
        elif not used_clean:
            cause = "true_pair_not_selected_for_training"
        else:
            cause = "clean_true_pair_used"

        stat = str(row.get("stat"))
        day = str(row.get("as_of_date"))
        causes[cause] += 1
        market_keys[str(row.get("market_key"))] += 1
        by_stat[stat][cause] += 1
        by_date[day][cause] += 1
        if cause != "clean_true_pair_used" and len(examples) < cfg.example_limit:
            examples.append(
                {
                    "date": day,
                    "player": row.get("player_name"),
                    "stat": stat,
                    "line": row.get("line"),
                    "market_key": row.get("market_key"),
                    "raw_over": row.get("over_price"),
                    "raw_under": row.get("under_price"),
                    "parsed_under": parsed_row.get("under_price") if parsed_row else None,
                    "normalized_under": normalized_row.get("under_price") if normalized_row else None,
                    "training_rows": int(training_row.get("rows") or 0) if training_row else 0,
                    "cause": cause,
                }
            )

    actionable = {key: value for key, value in causes.items() if key != "clean_true_pair_used"}
    dominant = max(actionable, key=actionable.get) if actionable else "none"
    parser_or_normalizer_loss = (
        causes.get("parser_dropped_true_under", 0)
        + causes.get("normalizer_dropped_true_under", 0)
    )
    raw_one_sided = causes.get("raw_api_one_sided", 0)
    if parser_or_normalizer_loss > 0:
        extraction_decision = "parser_or_normalizer_repair_required"
        training_action = "repair true under extraction before using FanDuel hitter market evidence"
    elif raw_one_sided > 0 and raw_one_sided >= causes.get("clean_true_pair_used", 0):
        extraction_decision = "raw_feed_missing_opposite_side"
        training_action = (
            "keep FanDuel one-sided hitter props display/research only; exclude from residual training, "
            "CLV proof, bucket promotion, and real-money ranking"
        )
    elif causes.get("clean_true_pair_used", 0) > 0:
        extraction_decision = "true_pairs_available"
        training_action = "use only clean true-paired rows for market evidence"
    else:
        extraction_decision = "insufficient_raw_rows"
        training_action = "collect more raw payloads before deciding"
    return {
        "status": "ready" if raw else "no_rows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lookback_days": cfg.lookback_days,
        "raw_offer_groups": int(len(raw)),
        "parsed_line_groups": int(len(parsed)),
        "normalized_offer_groups": int(len(normalized)),
        "training_groups": int(len(training)),
        "root_causes": dict(causes.most_common()),
        "dominant_root_cause": dominant,
        "extraction_decision": extraction_decision,
        "training_action": training_action,
        "raw_market_keys": dict(market_keys.most_common()),
        "by_stat": {key: dict(value) for key, value in sorted(by_stat.items())},
        "by_date": {key: dict(value) for key, value in sorted(by_date.items(), reverse=True)},
        "examples": examples,
    }


def _pct(value: int, total: int) -> str:
    return f"{(value / total):.1%}" if total else "-"


def _write_report(payload: dict[str, Any], cfg: FanDuelDiagnosticConfig) -> str:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / cfg.out
    total = int(payload.get("raw_offer_groups") or 0)
    lines = [
        "# FanDuel One-Sided Prop Diagnostic",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Lookback: {cfg.lookback_days} days",
        f"Raw FanDuel hitter offer groups: {total}",
        f"Dominant unresolved cause: {payload.get('dominant_root_cause')}",
        f"Extraction decision: **{payload.get('extraction_decision')}**",
        f"Training action: {payload.get('training_action')}",
        "",
        "## Root Cause Trace",
        "",
        "| Cause | Offer groups | Share |",
        "|---|---:|---:|",
    ]
    for cause, count in (payload.get("root_causes") or {}).items():
        lines.append(f"| {cause} | {count} | {_pct(int(count), total)} |")

    lines.extend(["", "## By Stat", "", "| Stat | Raw one-sided | Parser loss | Normalizer loss | Not replayed | Not selected | Clean pair |", "|---|---:|---:|---:|---:|---:|---:|"])
    for stat, counts in (payload.get("by_stat") or {}).items():
        lines.append(
            f"| {stat} | {counts.get('raw_api_one_sided', 0)} | {counts.get('parser_dropped_true_under', 0)} | "
            f"{counts.get('normalizer_dropped_true_under', 0)} | {counts.get('true_pair_not_replayed', 0)} | "
            f"{counts.get('true_pair_not_selected_for_training', 0)} | {counts.get('clean_true_pair_used', 0)} |"
        )

    lines.extend(["", "## Raw Market Surfaces", "", "| Odds API market key | Groups |", "|---|---:|"])
    for key, count in (payload.get("raw_market_keys") or {}).items():
        lines.append(f"| {key} | {count} |")

    lines.extend(["", "## Recent Failure Examples", "", "| Date | Player | Stat | Line | API market | Raw O/U | Parsed under | Normalized under | Training rows | Cause |", "|---|---|---|---:|---|---|---:|---:|---:|---|"])
    for row in payload.get("examples") or []:
        lines.append(
            f"| {row['date']} | {row['player']} | {row['stat']} | {row['line']} | {row['market_key']} | "
            f"{row['raw_over']}/{row['raw_under']} | {row['parsed_under']} | {row['normalized_under']} | "
            f"{row['training_rows']} | {row['cause']} |"
        )
    atomic_write_text(path, "\n".join(lines) + "\n")
    return str(path)


def build_report(cfg: FanDuelDiagnosticConfig) -> dict[str, Any]:
    payload = _analyze(cfg)
    payload["report_path"] = _write_report(payload, cfg)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_MODEL_DIR / cfg.json_out, payload, default=str)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one-sided FanDuel hitter props through the pipeline")
    parser.add_argument("--pg-dsn", default=_PG_DSN)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--out", default="mlb_fanduel_one_sided_diagnostic_latest.md")
    parser.add_argument("--json-out", default="fanduel_one_sided_diagnostic.json")
    args = parser.parse_args()
    payload = build_report(
        FanDuelDiagnosticConfig(
            pg_dsn=args.pg_dsn,
            lookback_days=args.lookback_days,
            out=args.out,
            json_out=args.json_out,
        )
    )
    print(json.dumps({"status": payload["status"], "root_causes": payload["root_causes"], "report_path": payload["report_path"]}, indent=2))


if __name__ == "__main__":
    main()
