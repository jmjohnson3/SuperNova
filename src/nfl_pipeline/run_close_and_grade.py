"""Capture NFL close odds, grade predictions, attach CLV, and refresh ledger."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2

from nfl_pipeline.integrity import nfl_season
from nfl_pipeline.db import PG_DSN
from nfl_pipeline import pregame_checkpoints as checkpoints
from nfl_pipeline.subprocess_utils import run_subprocess

log = logging.getLogger("nfl_pipeline.run_close_and_grade")
_ET = ZoneInfo("America/New_York")
_DEFAULT_CLOSE_WINDOW_BEFORE_MINUTES = 125
_DEFAULT_CLOSE_WINDOW_AFTER_MINUTES = 0
_FUTURE_SLATE_HORIZON_MINUTES = 48 * 60


@dataclass(frozen=True)
class Step:
    label: str
    module: str
    args: tuple[str, ...] = ()
    critical: bool = True
    failure_is_warning: bool = False
    timeout_s: int = 600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(step: Step) -> tuple[int, str, str]:
    env = os.environ.copy()
    src_dir = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    return run_subprocess(
        [sys.executable, "-m", step.module, *step.args],
        cwd=str(_repo_root()),
        env=env,
        timeout_s=step.timeout_s,
    )


def _has_close_work(et_date: date) -> bool:
    try:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM raw.nfl_games
                    WHERE game_date_et = %s
                )
                OR EXISTS (
                    SELECT 1
                    FROM bets.nfl_game_predictions
                    WHERE game_date_et = %s
                )
                OR EXISTS (
                    SELECT 1
                    FROM bets.nfl_player_prop_predictions
                    WHERE game_date_et = %s
                )
                """,
                (et_date, et_date, et_date),
            )
            return bool(cur.fetchone()[0])
    except Exception as exc:
        log.warning("Could not preflight NFL close work for %s: %s", et_date, exc)
        return True


def _active_close_games(
    et_date: date,
    *,
    before_minutes: int = _DEFAULT_CLOSE_WINDOW_BEFORE_MINUTES,
    after_minutes: int = _DEFAULT_CLOSE_WINDOW_AFTER_MINUTES,
) -> list[dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    start_min = now_utc - timedelta(minutes=after_minutes)
    start_max = now_utc + timedelta(minutes=before_minutes)
    try:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    game_id,
                    home_team_abbr,
                    away_team_abbr,
                    start_ts_utc,
                    EXTRACT(EPOCH FROM (start_ts_utc - %(now_utc)s)) / 60.0 AS minutes_to_start
                FROM raw.nfl_games
                WHERE game_date_et = %(game_date)s
                  AND start_ts_utc IS NOT NULL
                  AND start_ts_utc > %(start_min)s
                  AND start_ts_utc <= %(start_max)s
                ORDER BY start_ts_utc, game_id
                """,
                {
                    "game_date": et_date,
                    "now_utc": now_utc,
                    "start_min": start_min,
                    "start_max": start_max,
                },
            )
            return [
                {
                    "game_id": row[0],
                    "home_team_abbr": row[1],
                    "away_team_abbr": row[2],
                    "start_ts_utc": row[3],
                    "minutes_to_start": float(row[4]) if row[4] is not None else None,
                }
                for row in cur.fetchall()
            ]
    except Exception as exc:
        raise RuntimeError(f"Could not verify NFL active close window for {et_date}") from exc


def _upcoming_close_games(
    et_date: date,
    *,
    horizon_minutes: int = _FUTURE_SLATE_HORIZON_MINUTES,
) -> list[dict[str, Any]]:
    """Return future games so heavyweight settlement cannot occupy a live-slate task.

    The scheduled task uses ``IgnoreNew`` to prevent overlapping close crawls. A
    settlement/report sweep begun between kickoff waves can therefore suppress
    the next exact-line capture. Settlement starts only after the dated slate has
    no future kickoffs; the weekly training task owns heavyweight model work.
    """
    return _active_close_games(
        et_date,
        before_minutes=horizon_minutes,
        after_minutes=0,
    )


def _nearby_close_date(
    now_utc: datetime,
    *,
    before_minutes: int = _DEFAULT_CLOSE_WINDOW_BEFORE_MINUTES,
    after_minutes: int = max(_DEFAULT_CLOSE_WINDOW_AFTER_MINUTES, 360),
) -> date | None:
    """Return the nearest active/recent NFL game date, if one exists."""
    start_min = now_utc - timedelta(minutes=after_minutes)
    start_max = now_utc + timedelta(minutes=before_minutes)
    try:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT game_date_et
                FROM raw.nfl_games
                WHERE start_ts_utc IS NOT NULL
                  AND start_ts_utc >= %(start_min)s
                  AND start_ts_utc <= %(start_max)s
                ORDER BY
                  CASE
                    WHEN start_ts_utc >= %(now_utc)s - (%(close_after)s * interval '1 minute')
                     AND start_ts_utc <= %(now_utc)s + (%(close_before)s * interval '1 minute')
                    THEN 0 ELSE 1
                  END,
                  ABS(EXTRACT(EPOCH FROM (start_ts_utc - %(now_utc)s)))
                LIMIT 1
                """,
                {
                    "now_utc": now_utc,
                    "start_min": start_min,
                    "start_max": start_max,
                    "close_after": _DEFAULT_CLOSE_WINDOW_AFTER_MINUTES,
                    "close_before": before_minutes,
                },
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
    except Exception as exc:
        log.warning("Could not resolve nearby NFL close date: %s", exc)
    return None


def _unresolved_result_dates(
    now_utc: datetime,
    *,
    lookback: timedelta = timedelta(days=2),
) -> list[date]:
    """Return recent final slates that still have genuinely unresolved grades.

    A final game is not a settled player-prop slate until the exact locked
    predictions have a verified win/loss/push/void result. Missing joins and
    pending participation remain unresolved; they are never converted to zero.
    """
    try:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='30s'")
            cur.execute(
                """
                WITH unresolved AS (
                    SELECT g.game_date_et, g.start_ts_utc
                    FROM raw.nfl_games g
                    JOIN bets.nfl_player_prop_predictions p ON p.game_id = g.game_id
                    LEFT JOIN bets.nfl_player_prop_prediction_results r
                      ON r.prediction_id = p.id
                    WHERE g.status = 'final'
                      AND g.start_ts_utc >= %(start_min)s
                      AND g.start_ts_utc < %(start_max)s
                      AND p.side IN ('over', 'under')
                      AND p.line IS NOT NULL
                      AND COALESCE(r.result, '') NOT IN
                          ('win', 'loss', 'push', 'void', 'void_nonparticipant')
                    UNION ALL
                    SELECT g.game_date_et, g.start_ts_utc
                    FROM raw.nfl_games g
                    JOIN bets.nfl_game_predictions p ON p.game_id = g.game_id
                    LEFT JOIN bets.nfl_game_prediction_results r
                      ON r.prediction_id = p.id
                    WHERE g.status = 'final'
                      AND g.start_ts_utc >= %(start_min)s
                      AND g.start_ts_utc < %(start_max)s
                      AND p.line IS NOT NULL
                      AND COALESCE(r.result, '') NOT IN
                          ('win', 'loss', 'push', 'void')
                )
                SELECT game_date_et
                FROM unresolved
                GROUP BY game_date_et
                ORDER BY MIN(start_ts_utc), game_date_et
                """,
                {
                    "start_min": now_utc - lookback,
                    "start_max": now_utc - RESULTS_PENDING_AFTER_KICKOFF,
                },
            )
            return [row[0] for row in cur.fetchall() if row[0]]
    except Exception as exc:
        log.warning("Could not resolve unresolved NFL result dates: %s", exc)
        return []


def _select_unresolved_date(candidates: list[date], state: dict[str, Any]) -> date | None:
    """Round-robin unresolved slates by their last import attempt."""
    if not candidates:
        return None
    attempts = state.get("last_results_import_by_date") or {}

    def key(day: date) -> tuple[datetime, date]:
        raw = attempts.get(day.isoformat())
        try:
            observed = datetime.fromisoformat(raw) if raw else datetime.min.replace(tzinfo=timezone.utc)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            observed = datetime.min.replace(tzinfo=timezone.utc)
        return observed, day

    return min(candidates, key=key)


def _default_close_date(
    *,
    before_minutes: int = _DEFAULT_CLOSE_WINDOW_BEFORE_MINUTES,
    after_minutes: int = max(_DEFAULT_CLOSE_WINDOW_AFTER_MINUTES, 360),
) -> date:
    """Prefer capture windows, then revisit unresolved slates across midnight."""
    now_utc = datetime.now(timezone.utc)
    nearby = _nearby_close_date(
        now_utc,
        before_minutes=before_minutes,
        after_minutes=after_minutes,
    )
    if nearby:
        return nearby
    unresolved = _select_unresolved_date(
        _unresolved_result_dates(now_utc),
        _load_settlement_state(),
    )
    return unresolved or datetime.now(_ET).date()


def _locked_prop_count(et_date: date) -> int:
    try:
        with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)::int
                FROM bets.nfl_player_prop_predictions
                WHERE game_date_et = %s
                  AND side IN ('over', 'under')
                  AND line IS NOT NULL
                  AND book IS NOT NULL
                """,
                (et_date,),
            )
            return int(cur.fetchone()[0] or 0)
    except Exception as exc:
        raise RuntimeError(f"Could not verify NFL locked props for {et_date}") from exc


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(stdout[start : end + 1])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _capture_health(et_date, game_ids, since):
    """Only fresh normalized rows count, not a provider's nonempty payload flag."""
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""
            WITH expected AS (
              SELECT DISTINCT p.game_id,o.provider,o.event_id,o.bookmaker_key
              FROM bets.nfl_player_prop_predictions p
              JOIN odds.nfl_player_prop_lines o ON o.id=p.offer_id
              WHERE p.game_date_et=%s AND p.game_id=ANY(%s) AND p.side IN ('over','under')
            )
            SELECT e.game_id,e.bookmaker_key,count(DISTINCT c.id)::int
            FROM expected e JOIN raw.nfl_games g ON g.game_id=e.game_id
            LEFT JOIN odds.nfl_player_prop_lines c ON c.provider=e.provider AND c.event_id=e.event_id
              AND c.bookmaker_key=e.bookmaker_key AND c.snapshot_role='close'
              AND c.fetched_at_utc>=%s AND c.fetched_at_utc>=g.start_ts_utc-interval '120 minutes'
              AND c.fetched_at_utc<g.start_ts_utc
              AND (abs(c.over_price)>=100 OR abs(c.under_price)>=100)
            GROUP BY e.game_id,e.bookmaker_key
        """, (et_date, game_ids, since))
        return [dict(game_id=r[0], book=r[1], fresh_rows=r[2]) for r in cur.fetchall()]


SETTLEMENT_STATE = Path(__file__).resolve().parents[2] / "reports" / "nfl_close_settlement_state.json"
CHECKPOINT_STATE = checkpoints.STATE
RESULTS_PENDING_AFTER_KICKOFF = timedelta(hours=3)
RESULTS_PENDING_LOOKBACK = timedelta(days=4)
RESULTS_RETRY_INTERVAL = timedelta(minutes=30)
RESULTS_REFRESH_INTERVAL = timedelta(hours=6)
RESULT_IMPORT_MODULES = {"nfl_pipeline.import_nflverse", "nfl_pipeline.import_usage_context"}


def _results_pending(now: datetime) -> bool:
    """A game that should be over is not final yet, so new results may be published."""
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='30s'")
        cur.execute("""SELECT EXISTS (SELECT 1 FROM raw.nfl_games
            WHERE start_ts_utc < %s AND start_ts_utc > %s AND COALESCE(status, '') <> 'final')""",
                    (now - RESULTS_PENDING_AFTER_KICKOFF, now - RESULTS_PENDING_LOOKBACK))
        game_pending = bool(cur.fetchone()[0])
    return game_pending or bool(_unresolved_result_dates(now))


def _settlement_signature() -> dict[str, Any]:
    """Cheap fingerprint of everything grading/CLV/evaluation reads.

    updated_at_utc only moves on real content changes (upserts skip no-op rewrites), so an
    unchanged signature means rerunning settlement would recompute identical outputs.
    """
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout='60s'")
        cur.execute("""SELECT
            (SELECT row(count(*), max(updated_at_utc))::text FROM raw.nfl_games WHERE status = 'final'),
            (SELECT max(updated_at_utc)::text FROM raw.nfl_player_gamelogs),
            (SELECT row(count(*), max(id))::text FROM bets.nfl_player_prop_predictions),
            (SELECT row(count(*), max(id))::text FROM bets.nfl_game_predictions),
            (SELECT max(fetched_at_utc)::text FROM odds.nfl_player_prop_lines WHERE snapshot_role = 'close'),
            (SELECT max(fetched_at_utc)::text FROM odds.nfl_game_lines WHERE snapshot_role = 'close'),
            (SELECT max(ledger_id)::text FROM bets.nfl_bet_ledger),
            (SELECT max(event_id)::text FROM bets.nfl_cash_execution_events)""")
        keys = ("final_games", "gamelogs", "prop_forecasts", "game_forecasts", "prop_closes", "game_closes",
                "ledger", "cash_events")
        return dict(zip(keys, cur.fetchone()))


def _load_settlement_state() -> dict[str, Any]:
    try:
        return json.loads(SETTLEMENT_STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def run_for_date(
    et_date: date,
    *,
    force_crawl: bool = False,
    close_window_before_minutes: int = _DEFAULT_CLOSE_WINDOW_BEFORE_MINUTES,
    close_window_after_minutes: int = _DEFAULT_CLOSE_WINDOW_AFTER_MINUTES,
) -> dict[str, Any]:
    date_args = ("--date", et_date.isoformat())
    steps = [
        Step("NFL Close Odds Crawl", "nfl_pipeline.crawler_oddsapi", args=(*date_args, "--snapshot-role", "close"), critical=False, timeout_s=300),
        Step("NFL Close Odds Parse", "nfl_pipeline.parse_oddsapi", args=date_args, critical=True, timeout_s=180),
        # Once per game near kickoff (state-guarded, credit floor); parses its own payloads.
        Step("NFL Sharp Close Lines", "nfl_pipeline.sharp_lines", args=(*date_args, "--role", "close"), critical=False, timeout_s=180),
        Step("NFL Final Results Refresh", "nfl_pipeline.import_nflverse", args=("--seasons",str(nfl_season(et_date))), critical=True, timeout_s=600),
        Step("NFL Final Participation Refresh", "nfl_pipeline.import_usage_context",
             args=("--seasons",str(nfl_season(et_date)),"--skip-schema","--skip-pbp","--skip-participation","--skip-advanced-usage"), timeout_s=180),
        Step("NFL Grade Predictions", "nfl_pipeline.grade_predictions", critical=True, timeout_s=300),
        Step("NFL Verified Receiving Result Repair", "nfl_pipeline.repair_receiving_results", args=(*date_args,"--apply"), timeout_s=180),
        Step("NFL Grade Recovered Receiving Results", "nfl_pipeline.grade_predictions", args=date_args, timeout_s=180),
        Step("NFL Receiving Result Reconciliation", "nfl_pipeline.modeling.receiving_result_reconciliation", args=date_args, timeout_s=180),
        Step("NFL Attach CLV", "nfl_pipeline.clv_report", args=date_args, critical=True, timeout_s=300),
        Step("NFL Close Capture Diagnostic", "nfl_pipeline.close_capture_diagnostic", args=date_args, timeout_s=120),
        Step("NFL Refresh Settled Ledger", "nfl_pipeline.lock_ledger", args=("--refresh-only",), timeout_s=180),
        Step("NFL Live Scoring Replay", "nfl_pipeline.modeling.live_scoring_replay", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=300),
        Step("NFL Per-Stat Model Validation", "nfl_pipeline.modeling.stat_model_validation",
             critical=False, failure_is_warning=True, timeout_s=300),
        Step("NFL Frozen Receiving Paper Evidence", "nfl_pipeline.modeling.receiving_paper_trial",
             args=("report",), critical=False, failure_is_warning=True, timeout_s=180),
        Step("NFL Actual Wager Net Profit", "nfl_pipeline.wager_reconciliation",
             args=("--report", "--output", "data/nfl_wager_reconciliation/reports/latest.json"),
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Scoring Cohort Audit", "nfl_pipeline.modeling.scoring_equivalence",
             critical=False, failure_is_warning=True, timeout_s=600),
        Step(
            "NFL Final Probability Validation",
            "nfl_pipeline.modeling.final_probability_validation",
            args=date_args,
            critical=False,
            failure_is_warning=True,
            timeout_s=300,
        ),
        Step("NFL Receiving Selection Experiment", "nfl_pipeline.modeling.receiving_selection_experiment",
             critical=False, failure_is_warning=True, timeout_s=300),
        Step(
            "NFL Independent Probability Policy Review",
            "nfl_pipeline.modeling.validate_distribution_policy",
            args=(
                "--markets",
                "receiving_yards",
                "passing_yards",
                "rushing_yards",
                "total",
                "--register",
            ),
            critical=False,
            failure_is_warning=True,
            timeout_s=1200,
        ),
        Step("NFL Post-Settlement Evidence", "nfl_pipeline.settlement_evidence", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=180),
        Step("NFL Receiving Trial Checkpoint", "nfl_pipeline.modeling.receiving_trial_checkpoint",
             args=(*date_args, "--incremental"), critical=False, failure_is_warning=True, timeout_s=180),
        Step("NFL Rushing Volume Challenger Review", "nfl_pipeline.modeling.rushing_volume_challenger",
             args=("evaluate",), critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Point Forecast Checkpoint", "nfl_pipeline.modeling.target_point_capture",
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Confirmed Cash Settlement", "nfl_pipeline.cash_execution", args=("--settle",), timeout_s=90),
        Step("NFL Cash Readiness", "nfl_pipeline.cash_readiness", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=900),
        Step("NFL Live Cycle Audit", "nfl_pipeline.live_cycle_audit", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Exact-Line Prop Proof", "nfl_pipeline.modeling.prop_exact_line_proof", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Exact-Line Training Evidence", "nfl_pipeline.modeling.train_prop_exact_line_models",
             critical=False, failure_is_warning=True, timeout_s=300),
        Step("NFL Snapshot Health", "nfl_pipeline.snapshot_health_report", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Readiness Report", "nfl_pipeline.readiness_report", args=date_args,
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL CLV Scorecard", "nfl_pipeline.modeling.clv_scorecard",
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL FanDuel Price Gap Evidence", "nfl_pipeline.modeling.sharp_evidence_report",
             critical=False, failure_is_warning=True, timeout_s=120),
        Step("NFL Research Parlays", "nfl_pipeline.modeling.sharp_parlay_report",
             critical=False, failure_is_warning=True, timeout_s=120),
    ]
    results: list[dict[str, Any]] = []
    status = "ok"
    # The sharp watcher runs on its own NFL-Sharp-Watch task and its own mutex, so a long model step
    # here can never delay an alert. It is deliberately not a step in this runner.
    has_close_work = _has_close_work(et_date)
    locked_prop_count = _locked_prop_count(et_date)
    active_games = _active_close_games(
        et_date,
        before_minutes=close_window_before_minutes,
        after_minutes=close_window_after_minutes,
    )
    checkpoint_state = checkpoints.load_state(CHECKPOINT_STATE)
    checkpoint_plan = checkpoints.plan(active_games, checkpoint_state)
    checkpoint_assignments = checkpoint_plan["due"]
    checkpoint_games = [
        game for game in active_games if str(game["game_id"]) in checkpoint_assignments
    ]
    checkpoints.write_report(
        et_date,
        checkpoint_state,
        checkpoint_plan["observations"],
    )
    upcoming_games = (
        _upcoming_close_games(et_date)
        if not active_games
        else []
    )
    should_crawl_close = force_crawl or bool(checkpoint_games)
    # Keep the 10-minute close path short. Result import/model evaluation cannot
    # occupy the same mutex while a game's final pre-kickoff quotes disappear.
    settlement: dict[str, Any] = {}
    if checkpoint_games or (force_crawl and active_games):
        capture_modules = {'nfl_pipeline.crawler_oddsapi', 'nfl_pipeline.parse_oddsapi', 'nfl_pipeline.sharp_lines',
                           'nfl_pipeline.clv_report', 'nfl_pipeline.close_capture_diagnostic',
                           'nfl_pipeline.snapshot_health_report'}
        steps = [s for s in steps if s.module in capture_modules]
    elif active_games and not force_crawl:
        # The scheduler wakes every five minutes, but provider calls are made
        # only for the four named checkpoints. This preserves distinct evidence
        # and avoids relabeling an arbitrary live quote as T-60 or T-20.
        return {
            "status": "ok",
            "date": et_date.isoformat(),
            "close_window": {
                "mode": "waiting_for_target_checkpoint",
                "active_games": active_games,
                "checkpoint_plan": checkpoint_plan,
                "locked_prop_count": locked_prop_count,
            },
            "settlement": {},
            "steps": results,
        }
    elif not has_close_work and not upcoming_games and not force_crawl:
        # A five-minute scheduler must stay cheap on non-game days. Heavy
        # all-history evaluation belongs only to a dated slate with locks/results.
        return {
            "status": "ok",
            "date": et_date.isoformat(),
            "close_window": {
                "mode": "idle_no_slate_work",
                "active_games": [],
                "checkpoint_plan": checkpoint_plan,
                "locked_prop_count": locked_prop_count,
            },
            "settlement": {},
            "steps": results,
        }
    elif upcoming_games and not force_crawl:
        # Do not let a 20-30 minute settlement/report sweep cross into any later
        # capture window. The next ten-minute invocation will take the hot path.
        return {
            "status": "ok",
            "date": et_date.isoformat(),
            "close_window": {
                "mode": "waiting_for_capture_window",
                "active_games": [],
                "upcoming_games": upcoming_games,
                "locked_prop_count": locked_prop_count,
            },
            "settlement": {},
            "steps": results,
        }
    elif not force_crawl:
        # Settlement mode runs every 10 minutes. Only fetch results when some are due (or a
        # periodic correction refresh), and only re-run grading/evaluation when inputs changed.
        now = datetime.now(timezone.utc)
        state = _load_settlement_state()
        import_history = state.get("last_results_import_by_date") or {}
        last_import = import_history.get(et_date.isoformat())
        if not last_import and (state.get("signature") or {}).get("date") == et_date.isoformat():
            last_import = state.get("last_results_import_at")
        pending = _results_pending(now)
        refresh_interval = RESULTS_RETRY_INTERVAL if pending else RESULTS_REFRESH_INTERVAL
        import_due = (
            not last_import
            or not state.get("last_full_run_ok")
            or now - datetime.fromisoformat(last_import) >= refresh_interval
        )
        before = dict(_settlement_signature(), date=et_date.isoformat())  # dated reports refresh once per day
        settlement = dict(results_pending=pending, results_import_due=import_due,
                          signature_before=before, previous_signature=state.get("signature"))
        if not import_due:
            steps = [s for s in steps if s.module not in RESULT_IMPORT_MODULES]
            if before == state.get("signature") and state.get("last_full_run_ok"):
                return {"status": "ok", "date": et_date.isoformat(),
                        "close_window": {"mode": "settlement_skipped_no_new_inputs", "active_games": [],
                                         "locked_prop_count": locked_prop_count},
                        "settlement": settlement, "steps": results}
    attempt_started = datetime.now(timezone.utc)
    capture_health = []; retry_fired = False
    captured_games = checkpoint_games or (active_games if force_crawl else [])
    for step in steps:
        if step.module == "nfl_pipeline.crawler_oddsapi" and not has_close_work:
            results.append(
                {
                    "label": step.label,
                    "returncode": 0,
                    "stdout_tail": f"Skipped odds API close crawl for {et_date}: no scheduled NFL games or locked predictions.",
                    "stderr_tail": "",
                }
            )
            continue
        if step.module == "nfl_pipeline.crawler_oddsapi" and not should_crawl_close:
            results.append(
                {
                    "label": step.label,
                    "returncode": 0,
                    "stdout_tail": (
                        f"Skipped odds API close crawl for {et_date}: no games inside "
                        f"T-{close_window_before_minutes} to T+{close_window_after_minutes} minutes. "
                        "Scheduler will retry on its next 10-minute run."
                    ),
                    "stderr_tail": "",
                }
            )
            continue
        rc, stdout, stderr = _run(step)
        if step.module == 'nfl_pipeline.parse_oddsapi' and captured_games and rc == 0:
            game_ids = [g['game_id'] for g in captured_games if 0 < g['minutes_to_start'] <= 120]
            try:
                capture_health = _capture_health(et_date, game_ids, attempt_started)
                in_window = bool(game_ids)
                if in_window and any(r['fresh_rows'] == 0 for r in capture_health):
                    retry_fired = True
                    for retry_step in steps[:2]:
                        retry_rc, retry_out, retry_err = _run(retry_step)
                        results.append(dict(label=retry_step.label+' (bounded retry)', returncode=retry_rc,
                                            stdout_tail=retry_out[-1000:], stderr_tail=retry_err[-1000:]))
                        if retry_rc:
                            status = 'failed'
                    capture_health = _capture_health(et_date, game_ids, attempt_started)
                    if any(r['fresh_rows'] == 0 for r in capture_health):
                        rc, stderr = 2, 'Fresh normalized close offers still missing after bounded retry'
            except Exception as exc:
                rc, stderr = 2, 'Close coverage verification failed: '+type(exc).__name__
        rec = {
            "label": step.label,
            "returncode": rc,
            "stdout_tail": stdout.strip()[-1000:],
            "stderr_tail": stderr.strip()[-1000:],
        }
        if step.module == "nfl_pipeline.crawler_oddsapi":
            crawl_payload = _json_from_stdout(stdout)
            if crawl_payload:
                rec["crawl_status"] = crawl_payload.get("status")
                rec["game_odds_saved"] = crawl_payload.get("game_odds_saved")
                rec["player_prop_events_saved"] = crawl_payload.get("player_prop_events_saved")
                rec["player_prop_fetch_failures"] = crawl_payload.get("player_prop_fetch_failures")
            if rc == 0 and locked_prop_count > 0 and int((crawl_payload or {}).get("player_prop_events_saved") or 0) <= 0:
                rc = 2
                rec["returncode"] = rc
                rec["stderr_tail"] = (
                    (rec["stderr_tail"] + "\n") if rec["stderr_tail"] else ""
                ) + (
                    f"Close crawl saved zero player-prop payloads for {locked_prop_count} locked props. "
                    "CLV cannot attach without close-role prop odds."
                )
        results.append(rec)
        if rc != 0:
            if step.failure_is_warning:
                rec["nonfatal"] = True
            else:
                status = "failed"
            if step.critical and not step.failure_is_warning:
                break
    checkpoint_verification: dict[str, dict[str, Any]] = {}
    if checkpoint_games and checkpoint_assignments:
        try:
            checkpoint_verification = checkpoints.verify_capture(
                et_date,
                checkpoint_games,
                attempt_started,
            )
        except Exception as exc:
            checkpoint_verification = {
                str(game["game_id"]): {
                    "game_id": str(game["game_id"]),
                    "error": "checkpoint_verification_failed:" + type(exc).__name__,
                    "verification_passed": False,
                }
                for game in checkpoint_games
            }
        checkpoints.record_attempt(
            checkpoint_state,
            checkpoint_games,
            checkpoint_assignments,
            checkpoint_verification,
        )
        checkpoints.save_state(checkpoint_state, CHECKPOINT_STATE)
        checkpoint_plan = checkpoints.plan(active_games, checkpoint_state)
        checkpoints.write_report(
            et_date,
            checkpoint_state,
            checkpoint_plan["observations"],
        )
        if any(
            not row.get("verification_passed")
            for row in checkpoint_verification.values()
        ):
            status = "failed"
    if settlement:
        state = _load_settlement_state()
        if settlement["results_import_due"] and all(
                r["returncode"] == 0 for r in results if r["label"] in ("NFL Final Results Refresh", "NFL Final Participation Refresh")):
            state["last_results_import_at"] = attempt_started.isoformat()
            import_history = dict(state.get("last_results_import_by_date") or {})
            import_history[et_date.isoformat()] = attempt_started.isoformat()
            state["last_results_import_by_date"] = import_history
        # Signature after this run's own writes, so the next unchanged run is skipped.
        state.update(signature=dict(_settlement_signature(), date=et_date.isoformat()), last_full_run_ok=status == "ok",
                     last_run_at=datetime.now(timezone.utc).isoformat())
        settlement["signature_after"] = state["signature"]
        from nfl_pipeline.integrity import atomic_json
        atomic_json(SETTLEMENT_STATE, state)
    return {
        "status": status,
        "date": et_date.isoformat(),
        "close_window": {
            "force_crawl": force_crawl,
            "before_minutes": close_window_before_minutes,
            "after_minutes": close_window_after_minutes,
            "active_games": active_games,
            "checkpoint_assignments": checkpoint_assignments,
            "checkpoint_plan": checkpoint_plan,
            "checkpoint_verification": checkpoint_verification,
            "locked_prop_count": locked_prop_count,
            "close_crawl_attempted": should_crawl_close and has_close_work,
            "mode": "capture_only" if captured_games else "settlement_and_evaluation",
            "fresh_capture_health": capture_health,
            "retry_fired": retry_fired,
        },
        "settlement": settlement,
        "steps": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NFL close capture, grading, CLV, and ledger refresh")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD")
    parser.add_argument("--force-crawl", action="store_true", help="Capture close odds even outside the kickoff close window")
    parser.add_argument("--close-window-before-minutes", type=int, default=_DEFAULT_CLOSE_WINDOW_BEFORE_MINUTES)
    parser.add_argument("--close-window-after-minutes", type=int, default=_DEFAULT_CLOSE_WINDOW_AFTER_MINUTES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    et_date = (
        date.fromisoformat(args.date)
        if args.date
        else _default_close_date(
            before_minutes=args.close_window_before_minutes,
            after_minutes=max(args.close_window_after_minutes, 360),
        )
    )
    result = run_for_date(
        et_date,
        force_crawl=args.force_crawl,
        close_window_before_minutes=args.close_window_before_minutes,
        close_window_after_minutes=args.close_window_after_minutes,
    )
    print(json.dumps(result, indent=2, default=str))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
