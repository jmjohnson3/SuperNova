"""Write an operational NFL readiness report."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras

from nfl_pipeline.db import PG_DSN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "reports" / "nfl_readiness_latest.md"
DEFAULT_EXACT_LINE_MODELS = ROOT / "src" / "nfl_pipeline" / "modeling" / "models" / "player_props" / "nfl_prop_exact_line_models.json"
_ET = ZoneInfo("America/New_York")
REPORT_LOCK_TIMEOUT_MS = 5_000
REPORT_STATEMENT_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class ReadinessConfig:
    pg_dsn: str = PG_DSN
    game_date: date | None = None
    out_file: Path = DEFAULT_OUT


def _rows(conn, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _one(conn, sql: str, params: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(conn, sql, params)
    return rows[0] if rows else {}


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _pct(num: Any, den: Any) -> str:
    try:
        den_f = float(den or 0)
        return "0.0%" if den_f <= 0 else f"{float(num or 0) / den_f:.1%}"
    except (TypeError, ValueError, ZeroDivisionError):
        return "0.0%"


def _check(status: str, label: str, detail: str) -> str:
    return f"| {label} | {status} | {detail} |"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"status": "bad_artifact"}
    except Exception as exc:
        return {"status": f"load_failed: {exc.__class__.__name__}"}


def _write_readiness_issue(cfg: ReadinessConfig, et_day: date, status: str, detail: str) -> str:
    lines = [
        f"# NFL Readiness Report - {et_day}",
        "",
        "NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.",
        "",
        "## Summary",
        "",
        f"- Readiness query status: {status}",
        f"- Detail: {detail}",
        "",
        "## Daily Trust Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
        _check("warn", "Readiness report", detail),
        "",
        "This report is intentionally read-only and uses short lock timeouts so it cannot block live prediction, odds, or ledger jobs.",
        "",
    ]
    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    cfg.out_file.write_text(text, encoding="utf-8")
    return text


def _configure_report_session(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET lock_timeout = %s", (REPORT_LOCK_TIMEOUT_MS,))
        cur.execute("SET statement_timeout = %s", (REPORT_STATEMENT_TIMEOUT_MS,))
        cur.execute("SET idle_in_transaction_session_timeout = %s", (REPORT_STATEMENT_TIMEOUT_MS,))


def build_report(cfg: ReadinessConfig) -> str:
    et_day = cfg.game_date or datetime.now(_ET).date()
    params = {"game_date": et_day, "season": et_day.year}
    exact_line_models = _load_json(DEFAULT_EXACT_LINE_MODELS)
    conn = None
    try:
        conn = psycopg2.connect(cfg.pg_dsn)
        _configure_report_session(conn)
        context = _one(
            conn,
            """
            SELECT
                (SELECT COUNT(*) FROM raw.nfl_rosters WHERE season = %(season)s) AS roster_rows,
                (SELECT COUNT(DISTINCT team_abbr) FROM raw.nfl_rosters WHERE season = %(season)s) AS roster_teams,
                (SELECT MAX(updated_at_utc) FROM raw.nfl_rosters WHERE season = %(season)s) AS roster_latest_update,
                (SELECT COUNT(*) FROM raw.nfl_depth_charts WHERE season = %(season)s) AS depth_rows,
                (SELECT COUNT(DISTINCT team_abbr) FROM raw.nfl_depth_charts WHERE season = %(season)s) AS depth_teams,
                (SELECT MAX(snapshot_ts_utc) FROM raw.nfl_depth_charts WHERE season = %(season)s) AS depth_latest_snapshot,
                (SELECT COUNT(*) FROM raw.nfl_injuries WHERE season = %(season)s) AS injury_rows,
                (SELECT COUNT(DISTINCT team_abbr) FROM raw.nfl_injuries WHERE season = %(season)s) AS injury_teams,
                (SELECT MAX(updated_at_utc) FROM raw.nfl_injuries WHERE season = %(season)s) AS injury_latest_update
            """,
            params,
        )
        usage_context = _one(
            conn,
            """
            SELECT
                COUNT(*) AS player_game_rows,
                COUNT(offense_snap_share) AS offense_snap_share_rows,
                COUNT(routes_run) AS routes_run_rows,
                COUNT(pass_route_opportunities) AS pass_route_opportunity_rows,
                COUNT(pass_route_opportunity_share) AS pass_route_opportunity_share_rows,
                COUNT(route_participation) AS route_participation_rows,
                COUNT(target_share) AS target_share_rows,
                COUNT(air_yards_share) AS air_yards_share_rows,
                COUNT(wopr) AS wopr_rows,
                COUNT(first_read_targets) AS first_read_target_rows,
                COUNT(end_zone_targets) AS end_zone_target_rows,
                COUNT(red_zone_touches) AS red_zone_touches_rows,
                COUNT(goal_line_carries) AS goal_line_carries_rows,
                COUNT(goal_line_targets) AS goal_line_targets_rows
            FROM raw.nfl_player_gamelogs
            WHERE season = %(season)s
              AND UPPER(COALESCE(position, '')) IN ('QB', 'RB', 'WR', 'TE')
            """,
            params,
        )
        feature_usage_context = _one(
            conn,
            """
            WITH latest_feature_season AS (
                SELECT MAX(season) AS season
                FROM features.nfl_player_game_training_features
                WHERE season <= %(season)s
                  AND UPPER(COALESCE(position, '')) IN ('QB', 'RB', 'WR', 'TE')
            )
            SELECT
                MAX(f.season) AS feature_season,
                COUNT(*) AS feature_rows,
                COUNT(f.route_participation_proxy_avg_5) AS route_proxy_rows,
                COUNT(f.estimated_routes_avg_5) AS estimated_route_rows,
                COUNT(f.starter_confidence) AS starter_confidence_rows,
                COUNT(f.rest_risk_score) AS rest_risk_rows,
                COUNT(f.receiving_usage_history_quality_score) AS receiving_usage_quality_rows,
                COUNT(f.rb_usage_history_quality_score) AS rb_usage_quality_rows,
                COUNT(f.td_usage_history_quality_score) AS td_usage_quality_rows,
                COUNT(f.live_usage_context_quality_v4_score) AS live_usage_quality_rows,
                COUNT(f.receiver_spike_under_correction_v4_score) AS receiver_spike_v4_rows,
                COUNT(f.rb_carry_under_correction_v4_score) AS rb_spike_v4_rows
            FROM features.nfl_player_game_training_features f
            JOIN latest_feature_season lfs ON f.season = lfs.season
            WHERE UPPER(COALESCE(f.position, '')) IN ('QB', 'RB', 'WR', 'TE')
            """,
            params,
        )
        raw_payloads = _rows(
            conn,
            """
            SELECT endpoint, snapshot_role, COUNT(*) AS payloads, MAX(fetched_at_utc) AS latest_fetch
            FROM raw.nfl_api_responses
            WHERE as_of_date = %(game_date)s
            GROUP BY endpoint, snapshot_role
            ORDER BY endpoint, snapshot_role
            """,
            params,
        )
        prop_payloads = _rows(
            conn,
            """
            SELECT
                snapshot_role,
                COUNT(*) AS payloads,
                SUM(CASE WHEN jsonb_typeof(
                    CASE
                        WHEN jsonb_typeof(payload->'bookmakers') = 'array' THEN payload->'bookmakers'
                        WHEN jsonb_typeof(payload->'data'->'bookmakers') = 'array' THEN payload->'data'->'bookmakers'
                        ELSE '[]'::jsonb
                    END
                ) = 'array' THEN jsonb_array_length(
                    CASE
                        WHEN jsonb_typeof(payload->'bookmakers') = 'array' THEN payload->'bookmakers'
                        WHEN jsonb_typeof(payload->'data'->'bookmakers') = 'array' THEN payload->'data'->'bookmakers'
                        ELSE '[]'::jsonb
                    END
                ) ELSE 0 END) AS bookmaker_entries,
                SUM(CASE WHEN jsonb_array_length(
                    CASE
                        WHEN jsonb_typeof(payload->'bookmakers') = 'array' THEN payload->'bookmakers'
                        WHEN jsonb_typeof(payload->'data'->'bookmakers') = 'array' THEN payload->'data'->'bookmakers'
                        ELSE '[]'::jsonb
                    END
                ) = 0 THEN 1 ELSE 0 END) AS payloads_with_zero_books,
                MAX(fetched_at_utc) AS latest_fetch
            FROM raw.nfl_api_responses
            WHERE provider = 'oddsapi'
              AND endpoint = 'nfl_player_props'
              AND as_of_date = %(game_date)s
            GROUP BY snapshot_role
            ORDER BY snapshot_role
            """,
            params,
        )
        prop_markets = _rows(
            conn,
            """
            SELECT
                r.snapshot_role,
                book->>'key' AS bookmaker_key,
                market->>'key' AS market_key,
                COUNT(*) AS market_payloads,
                SUM(CASE WHEN jsonb_typeof(market->'outcomes') = 'array' THEN jsonb_array_length(market->'outcomes') ELSE 0 END) AS outcomes
            FROM raw.nfl_api_responses r
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(r.payload->'bookmakers') = 'array' THEN r.payload->'bookmakers'
                    WHEN jsonb_typeof(r.payload->'data'->'bookmakers') = 'array' THEN r.payload->'data'->'bookmakers'
                    ELSE '[]'::jsonb
                END
            ) book
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE WHEN jsonb_typeof(book->'markets') = 'array' THEN book->'markets' ELSE '[]'::jsonb END
            ) market
            WHERE r.provider = 'oddsapi'
              AND r.endpoint = 'nfl_player_props'
              AND r.as_of_date = %(game_date)s
            GROUP BY r.snapshot_role, book->>'key', market->>'key'
            ORDER BY r.snapshot_role, bookmaker_key, market_key
            """,
            params,
        )
        parsed = _one(
            conn,
            """
            SELECT
                (SELECT COUNT(*) FROM odds.nfl_game_lines WHERE as_of_date = %(game_date)s) AS parsed_game_rows,
                (SELECT COUNT(DISTINCT event_id) FROM odds.nfl_game_lines WHERE as_of_date = %(game_date)s) AS parsed_game_events,
                (SELECT COUNT(*) FROM odds.nfl_player_prop_lines WHERE as_of_date = %(game_date)s) AS parsed_prop_rows,
                (SELECT COUNT(DISTINCT player_name_norm) FROM odds.nfl_player_prop_lines WHERE as_of_date = %(game_date)s) AS parsed_prop_players
            """,
            params,
        )
        coverage = _one(
            conn,
            """
            WITH locked_games AS (
                SELECT event_id, bookmaker_key
                FROM odds.nfl_game_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'lock'
                GROUP BY event_id, bookmaker_key
            ),
            closed_games AS (
                SELECT event_id, bookmaker_key
                FROM odds.nfl_game_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'close'
                GROUP BY event_id, bookmaker_key
            ),
            locked_props AS (
                SELECT player_name_norm, stat, line, bookmaker_key
                FROM odds.nfl_player_prop_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'lock'
                GROUP BY player_name_norm, stat, line, bookmaker_key
            ),
            closed_props AS (
                SELECT player_name_norm, stat, line, bookmaker_key
                FROM odds.nfl_player_prop_lines
                WHERE as_of_date = %(game_date)s AND snapshot_role = 'close'
                GROUP BY player_name_norm, stat, line, bookmaker_key
            )
            SELECT
                (SELECT COUNT(*) FROM locked_games) AS locked_game_markets,
                (SELECT COUNT(*) FROM locked_games lg JOIN closed_games cg USING (event_id, bookmaker_key)) AS closed_game_markets,
                (SELECT COUNT(*) FROM locked_props) AS locked_prop_markets,
                (SELECT COUNT(*) FROM locked_props lp JOIN closed_props cp USING (player_name_norm, stat, line, bookmaker_key)) AS closed_prop_markets
            """,
            params,
        )
        forecast = _one(
            conn,
            """
            SELECT
                (SELECT COUNT(*) FROM bets.nfl_game_predictions WHERE game_date_et = %(game_date)s) AS game_predictions,
                (SELECT COUNT(*) FROM bets.nfl_player_prop_predictions WHERE game_date_et = %(game_date)s) AS prop_predictions,
                (SELECT COUNT(*) FROM bets.nfl_bet_ledger WHERE game_date_et = %(game_date)s) AS ledger_rows,
                (SELECT COUNT(*) FROM bets.nfl_prediction_clv WHERE game_date_et = %(game_date)s) AS clv_rows
            """,
            params,
        )
        prop_proof = _one(
            conn,
            """
            WITH lock_props AS (
                SELECT player_name_norm, stat, line, bookmaker_key
                FROM odds.nfl_player_prop_lines
                WHERE as_of_date = %(game_date)s
                  AND snapshot_role = 'lock'
                  AND line IS NOT NULL
                GROUP BY player_name_norm, stat, line, bookmaker_key
            ),
            close_props AS (
                SELECT player_name_norm, stat, line, bookmaker_key
                FROM odds.nfl_player_prop_lines
                WHERE as_of_date = %(game_date)s
                  AND snapshot_role = 'close'
                  AND line IS NOT NULL
                GROUP BY player_name_norm, stat, line, bookmaker_key
            )
            SELECT
                (SELECT COUNT(*) FROM odds.nfl_player_prop_lines WHERE as_of_date = %(game_date)s) AS parsed_prop_rows,
                (SELECT COUNT(*) FROM odds.nfl_player_prop_lines WHERE as_of_date = %(game_date)s AND over_price IS NOT NULL AND under_price IS NOT NULL) AS true_paired_rows,
                (SELECT COUNT(*) FROM lock_props) AS lock_markets,
                (SELECT COUNT(*) FROM close_props) AS close_markets,
                (SELECT COUNT(*) FROM lock_props lp JOIN close_props cp USING (player_name_norm, stat, line, bookmaker_key)) AS exact_lock_close_markets,
                (SELECT COUNT(*) FROM bets.nfl_prediction_clv WHERE game_date_et = %(game_date)s AND source_kind = 'prop') AS clv_labels,
                (SELECT COUNT(*) FROM bets.nfl_prediction_clv WHERE game_date_et = %(game_date)s AND source_kind = 'prop' AND (valid_close_snapshot_captured IS TRUE OR clv_status = 'valid_close')) AS valid_clv_labels,
                (SELECT COUNT(*) FROM bets.nfl_player_prop_prediction_results WHERE game_date_et = %(game_date)s) AS graded_prop_predictions,
                (SELECT COUNT(*) FROM bets.nfl_bet_ledger WHERE game_date_et = %(game_date)s AND source_kind = 'prop') AS prop_ledger_rows
            """,
            params,
        )
        clv_statuses = _rows(
            conn,
            """
            SELECT COALESCE(close_quality_reason, clv_status, 'unknown') AS close_quality_reason, COUNT(*) AS rows
            FROM bets.nfl_prediction_clv
            WHERE game_date_et = %(game_date)s
              AND source_kind = 'prop'
            GROUP BY COALESCE(close_quality_reason, clv_status, 'unknown')
            ORDER BY rows DESC, close_quality_reason
            """,
            params,
        )
        tiers = _rows(
            conn,
            """
            SELECT source_kind, tier, COUNT(*) AS rows
            FROM bets.nfl_bet_ledger
            WHERE game_date_et = %(game_date)s
            GROUP BY source_kind, tier
            ORDER BY source_kind, tier
            """,
            params,
        )
    except (
        psycopg2.errors.DeadlockDetected,
        psycopg2.errors.LockNotAvailable,
        psycopg2.errors.QueryCanceled,
    ) as exc:
        detail = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        return _write_readiness_issue(cfg, et_day, exc.__class__.__name__, detail)
    finally:
        if conn is not None:
            conn.close()

    parsed_prop_rows = int(parsed.get("parsed_prop_rows") or 0)
    raw_prop_payloads = sum(int(row.get("payloads") or 0) for row in prop_payloads)
    prop_book_entries = sum(int(row.get("bookmaker_entries") or 0) for row in prop_payloads)
    if raw_prop_payloads and parsed_prop_rows == 0 and prop_book_entries == 0:
        prop_status = "raw prop endpoint returned event payloads with `bookmakers: []`; parser is not missing markets"
    elif raw_prop_payloads and parsed_prop_rows == 0:
        prop_status = "raw prop payloads exist but parsed rows are zero; inspect market/outcome structure"
    elif parsed_prop_rows:
        prop_status = "player prop rows parsed"
    else:
        prop_status = "no player prop payloads captured"
    game_lock_coverage = (float(coverage.get("closed_game_markets") or 0) / float(coverage.get("locked_game_markets") or 1)) if int(coverage.get("locked_game_markets") or 0) else 0.0
    prop_lock_coverage = (float(coverage.get("closed_prop_markets") or 0) / float(coverage.get("locked_prop_markets") or 1)) if int(coverage.get("locked_prop_markets") or 0) else 0.0
    locked_prop_markets = int(coverage.get("locked_prop_markets") or 0)
    prop_lock_status = "watch" if locked_prop_markets == 0 else "pass" if prop_lock_coverage >= 0.90 else "watch"
    prop_lock_detail = "no locked prop markets yet" if locked_prop_markets == 0 else _pct(coverage.get("closed_prop_markets"), coverage.get("locked_prop_markets"))
    trust_checks = [
        _check("pass" if int(context.get("roster_rows") or 0) > 0 else "fail", "Roster context", f"{_fmt(context.get('roster_rows'))} rows"),
        _check("pass" if int(context.get("depth_rows") or 0) > 0 else "fail", "Depth context", f"{_fmt(context.get('depth_rows'))} rows"),
        _check("warn" if int(context.get("injury_rows") or 0) == 0 else "pass", "Injury context", f"{_fmt(context.get('injury_rows'))} rows"),
        _check("pass" if int(usage_context.get("offense_snap_share_rows") or 0) > 0 else "watch", "Snap context", f"{_pct(usage_context.get('offense_snap_share_rows'), usage_context.get('player_game_rows'))} coverage"),
        _check("pass" if int(feature_usage_context.get("route_proxy_rows") or 0) > 0 else "watch", "Route proxy context", f"{_pct(feature_usage_context.get('route_proxy_rows'), feature_usage_context.get('feature_rows'))} coverage from training season {_fmt(feature_usage_context.get('feature_season'))}"),
        _check("pass" if int(usage_context.get("red_zone_touches_rows") or 0) > 0 else "watch", "Red-zone context", f"{_pct(usage_context.get('red_zone_touches_rows'), usage_context.get('player_game_rows'))} coverage"),
        _check("pass" if int(parsed.get("parsed_game_rows") or 0) > 0 else "fail", "Game odds parsed", f"{_fmt(parsed.get('parsed_game_rows'))} rows"),
        _check("pass" if parsed_prop_rows > 0 else "watch", "Prop odds parsed", f"{parsed_prop_rows} rows; {prop_status}"),
        _check("pass" if game_lock_coverage >= 0.90 else "watch", "Game lock/close", _pct(coverage.get("closed_game_markets"), coverage.get("locked_game_markets"))),
        _check(prop_lock_status, "Prop lock/close", prop_lock_detail),
        _check("pass" if exact_line_models.get("status") == "ready" else "watch", "Exact-line prop models", f"{exact_line_models.get('status', 'missing')}; true-paired rows: {_fmt(exact_line_models.get('true_paired_rows'))}"),
        _check("pass" if int(forecast.get("game_predictions") or 0) > 0 else "watch", "Game predictions", f"{_fmt(forecast.get('game_predictions'))} rows"),
        _check("pass" if int(forecast.get("prop_predictions") or 0) > 0 else "watch", "Prop projections", f"{_fmt(forecast.get('prop_predictions'))} rows"),
        _check("watch" if int(forecast.get("ledger_rows") or 0) == 0 else "pass", "Ledger", f"{_fmt(forecast.get('ledger_rows'))} rows"),
        _check("watch" if int(forecast.get("clv_rows") or 0) == 0 else "pass", "CLV", f"{_fmt(forecast.get('clv_rows'))} rows"),
    ]

    lines = [
        f"# NFL Readiness Report - {et_day}",
        "",
        "NFL bankroll remains closed. Approved props may lock as $1 micro_projection trials once live links, true paired prices, and drift checks pass.",
        "",
        "## Summary",
        "",
        f"- Prop market status: {prop_status}",
        f"- Game lock/close coverage: {_pct(coverage.get('closed_game_markets'), coverage.get('locked_game_markets'))}",
        f"- Prop lock/close coverage: {_pct(coverage.get('closed_prop_markets'), coverage.get('locked_prop_markets'))}",
        f"- True-paired prop coverage: {_pct(prop_proof.get('true_paired_rows'), prop_proof.get('parsed_prop_rows'))}",
        f"- Game predictions: {_fmt(forecast.get('game_predictions'))} | Prop predictions: {_fmt(forecast.get('prop_predictions'))}",
        f"- Ledger rows: {_fmt(forecast.get('ledger_rows'))} | CLV rows: {_fmt(forecast.get('clv_rows'))}",
        f"- Valid prop CLV capture: {_pct(prop_proof.get('valid_clv_labels'), prop_proof.get('clv_labels'))}",
        f"- Exact-line prop model status: {exact_line_models.get('status', 'missing')} | true-paired rows: {_fmt(exact_line_models.get('true_paired_rows'))}",
        "",
        "## Daily Trust Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
        *trust_checks,
        "",
        "## Context Freshness",
        "",
        "| Data | Rows | Teams | Latest Timestamp |",
        "|---|---:|---:|---|",
        f"| roster {et_day.year} | {_fmt(context.get('roster_rows'))} | {_fmt(context.get('roster_teams'))} | {_fmt(context.get('roster_latest_update'))} |",
        f"| depth {et_day.year} | {_fmt(context.get('depth_rows'))} | {_fmt(context.get('depth_teams'))} | {_fmt(context.get('depth_latest_snapshot'))} |",
        f"| injury {et_day.year} | {_fmt(context.get('injury_rows'))} | {_fmt(context.get('injury_teams'))} | {_fmt(context.get('injury_latest_update'))} |",
        "",
        "## Player Usage Coverage",
        "",
        "| Field | Rows | Coverage |",
        "|---|---:|---:|",
        f"| offense snap share | {_fmt(usage_context.get('offense_snap_share_rows'))} | {_pct(usage_context.get('offense_snap_share_rows'), usage_context.get('player_game_rows'))} |",
        f"| true routes run | {_fmt(usage_context.get('routes_run_rows'))} | {_pct(usage_context.get('routes_run_rows'), usage_context.get('player_game_rows'))} |",
        f"| pass-route opportunities | {_fmt(usage_context.get('pass_route_opportunity_rows'))} | {_pct(usage_context.get('pass_route_opportunity_rows'), usage_context.get('player_game_rows'))} |",
        f"| pass-route opportunity share | {_fmt(usage_context.get('pass_route_opportunity_share_rows'))} | {_pct(usage_context.get('pass_route_opportunity_share_rows'), usage_context.get('player_game_rows'))} |",
        f"| route participation | {_fmt(usage_context.get('route_participation_rows'))} | {_pct(usage_context.get('route_participation_rows'), usage_context.get('player_game_rows'))} |",
        f"| route participation proxy, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('route_proxy_rows'))} | {_pct(feature_usage_context.get('route_proxy_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| estimated routes proxy, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('estimated_route_rows'))} | {_pct(feature_usage_context.get('estimated_route_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| starter confidence, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('starter_confidence_rows'))} | {_pct(feature_usage_context.get('starter_confidence_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| rest-risk score, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('rest_risk_rows'))} | {_pct(feature_usage_context.get('rest_risk_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| receiving usage quality v4, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('receiving_usage_quality_rows'))} | {_pct(feature_usage_context.get('receiving_usage_quality_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| RB usage quality v4, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('rb_usage_quality_rows'))} | {_pct(feature_usage_context.get('rb_usage_quality_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| TD usage quality v4, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('td_usage_quality_rows'))} | {_pct(feature_usage_context.get('td_usage_quality_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| live usage context quality v4, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('live_usage_quality_rows'))} | {_pct(feature_usage_context.get('live_usage_quality_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| receiver spike correction v4, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('receiver_spike_v4_rows'))} | {_pct(feature_usage_context.get('receiver_spike_v4_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| RB spike correction v4, training {feature_usage_context.get('feature_season') or '-'} | {_fmt(feature_usage_context.get('rb_spike_v4_rows'))} | {_pct(feature_usage_context.get('rb_spike_v4_rows'), feature_usage_context.get('feature_rows'))} |",
        f"| target share | {_fmt(usage_context.get('target_share_rows'))} | {_pct(usage_context.get('target_share_rows'), usage_context.get('player_game_rows'))} |",
        f"| air-yards share | {_fmt(usage_context.get('air_yards_share_rows'))} | {_pct(usage_context.get('air_yards_share_rows'), usage_context.get('player_game_rows'))} |",
        f"| WOPR | {_fmt(usage_context.get('wopr_rows'))} | {_pct(usage_context.get('wopr_rows'), usage_context.get('player_game_rows'))} |",
        f"| first-read targets | {_fmt(usage_context.get('first_read_target_rows'))} | {_pct(usage_context.get('first_read_target_rows'), usage_context.get('player_game_rows'))} |",
        f"| end-zone targets | {_fmt(usage_context.get('end_zone_target_rows'))} | {_pct(usage_context.get('end_zone_target_rows'), usage_context.get('player_game_rows'))} |",
        f"| red-zone touches | {_fmt(usage_context.get('red_zone_touches_rows'))} | {_pct(usage_context.get('red_zone_touches_rows'), usage_context.get('player_game_rows'))} |",
        f"| goal-line carries | {_fmt(usage_context.get('goal_line_carries_rows'))} | {_pct(usage_context.get('goal_line_carries_rows'), usage_context.get('player_game_rows'))} |",
        f"| goal-line targets | {_fmt(usage_context.get('goal_line_targets_rows'))} | {_pct(usage_context.get('goal_line_targets_rows'), usage_context.get('player_game_rows'))} |",
        "",
        "## Raw Odds Payloads",
        "",
        "| Endpoint | Role | Payloads | Latest Fetch |",
        "|---|---|---:|---|",
    ]
    if raw_payloads:
        for row in raw_payloads:
            lines.append(f"| {_fmt(row.get('endpoint'))} | {_fmt(row.get('snapshot_role'))} | {_fmt(row.get('payloads'))} | {_fmt(row.get('latest_fetch'))} |")
    else:
        lines.append("| none | - | 0 | - |")
    lines.extend([
        "",
        "## Prop Payload Diagnostic",
        "",
        "| Role | Payloads | Bookmaker Entries | Empty-Book Payloads | Latest Fetch |",
        "|---|---:|---:|---:|---|",
    ])
    if prop_payloads:
        for row in prop_payloads:
            lines.append(
                f"| {_fmt(row.get('snapshot_role'))} | {_fmt(row.get('payloads'))} | "
                f"{_fmt(row.get('bookmaker_entries'))} | {_fmt(row.get('payloads_with_zero_books'))} | {_fmt(row.get('latest_fetch'))} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | - |")
    lines.extend([
        "",
        "## Raw Prop Markets",
        "",
        "| Role | Book | Market | Payloads | Outcomes |",
        "|---|---|---|---:|---:|",
    ])
    if prop_markets:
        for row in prop_markets[:80]:
            lines.append(
                f"| {_fmt(row.get('snapshot_role'))} | {_fmt(row.get('bookmaker_key'))} | {_fmt(row.get('market_key'))} | "
                f"{_fmt(row.get('market_payloads'))} | {_fmt(row.get('outcomes'))} |"
            )
    else:
        lines.append("| none | - | no raw prop markets found | 0 | 0 |")
    lines.extend([
        "",
        "## Parsed Rows",
        "",
        "| Type | Rows | Distinct Entities |",
        "|---|---:|---:|",
        f"| game odds | {_fmt(parsed.get('parsed_game_rows'))} | {_fmt(parsed.get('parsed_game_events'))} |",
        f"| player props | {_fmt(parsed.get('parsed_prop_rows'))} | {_fmt(parsed.get('parsed_prop_players'))} |",
        "",
        "## Lock To Close Coverage",
        "",
        "| Type | Locked | Closed Same Market | Coverage |",
        "|---|---:|---:|---:|",
        f"| game | {_fmt(coverage.get('locked_game_markets'))} | {_fmt(coverage.get('closed_game_markets'))} | {_pct(coverage.get('closed_game_markets'), coverage.get('locked_game_markets'))} |",
        f"| prop | {_fmt(coverage.get('locked_prop_markets'))} | {_fmt(coverage.get('closed_prop_markets'))} | {_pct(coverage.get('closed_prop_markets'), coverage.get('locked_prop_markets'))} |",
        "",
        "## Prop Odds Proof",
        "",
        "| Proof Layer | Rows | Coverage / Detail |",
        "|---|---:|---|",
        f"| parsed prop offers | {_fmt(prop_proof.get('parsed_prop_rows'))} | raw parser output |",
        f"| true paired over/under offers | {_fmt(prop_proof.get('true_paired_rows'))} | {_pct(prop_proof.get('true_paired_rows'), prop_proof.get('parsed_prop_rows'))} |",
        f"| lock markets | {_fmt(prop_proof.get('lock_markets'))} | exact book/player/stat/line locked |",
        f"| close markets | {_fmt(prop_proof.get('close_markets'))} | exact book/player/stat/line closed |",
        f"| lock-close exact matches | {_fmt(prop_proof.get('exact_lock_close_markets'))} | {_pct(prop_proof.get('exact_lock_close_markets'), prop_proof.get('lock_markets'))} |",
        f"| CLV labels | {_fmt(prop_proof.get('clv_labels'))} | valid: {_fmt(prop_proof.get('valid_clv_labels'))} |",
        f"| graded prop predictions | {_fmt(prop_proof.get('graded_prop_predictions'))} | line-level result rows |",
        f"| prop ledger rows | {_fmt(prop_proof.get('prop_ledger_rows'))} | locked paper/micro/bankroll rows |",
        "",
        "## Prop CLV Quality",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ])
    if clv_statuses:
        for row in clv_statuses:
            lines.append(f"| {_fmt(row.get('close_quality_reason'))} | {_fmt(row.get('rows'))} |")
    else:
        lines.append("| none | 0 |")
    lines.extend([
        "",
        "## Ledger Tiers",
        "",
        "| Source | Tier | Rows |",
        "|---|---|---:|",
    ])
    if tiers:
        for row in tiers:
            lines.append(f"| {_fmt(row.get('source_kind'))} | {_fmt(row.get('tier'))} | {_fmt(row.get('rows'))} |")
    else:
        lines.append("| none | - | 0 |")
    lines.append("")

    cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    cfg.out_file.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write NFL operational readiness report")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--pg-dsn", default=PG_DSN)
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    print(build_report(ReadinessConfig(
        pg_dsn=args.pg_dsn,
        game_date=date.fromisoformat(args.date) if args.date else None,
        out_file=Path(args.out_file),
    )))


if __name__ == "__main__":
    main()
