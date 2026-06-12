"""Regression checks for odds date buckets, snapshot safety, and scheduler wiring."""
from __future__ import annotations

import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from mlb_pipeline.crawler_oddsapi import _build_full_url as mlb_build_full_url
from mlb_pipeline.crawler_statsapi import _status_from_schedule_game
from mlb_pipeline.modeling.bankroll_ledger import (
    game_bankroll_pick_key,
    game_bankroll_risk_slot,
    prop_bankroll_pick_key,
    prop_bankroll_risk_slot,
)
from mlb_pipeline.modeling.game_line_clv import game_line_clv, resolve_valid_game_close
from mlb_pipeline.modeling.predict_player_props import _cap_prop_db_rows
from mlb_pipeline.modeling.prop_clean_slate import CleanSlateThresholds, clean_slate_qualifies
from mlb_pipeline.modeling.predict_today import _cap_game_bankroll_rows
from mlb_pipeline.modeling.prop_offer_links import filter_prop_offers_for_game
from mlb_pipeline.modeling.prop_snapshot_coverage_report import slate_qualifies
from mlb_pipeline.modeling.update_outcomes import _resolve_game_close_for_bet
from mlb_pipeline.parse_games import _status_from_game_obj
from mlb_pipeline.parse_oddsapi import _event_matches_as_of_date as mlb_event_matches_as_of_date
from nba_pipeline.crawler_oddsapi import _build_full_url as nba_build_full_url
from nba_pipeline.modeling.update_outcomes import (
    _over_hit as nba_over_hit,
    _resolve_valid_close as resolve_valid_nba_close,
    _spread_bet_result as nba_spread_bet_result,
    _total_bet_result as nba_total_bet_result,
)
from nba_pipeline.parse_oddsapi import _event_matches_as_of_date as nba_event_matches_as_of_date


ROOT = Path(__file__).resolve().parents[3]


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _sql, _params=None):
        return None

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, commence, snapshots):
        self.commence = commence
        self.snapshots = snapshots
        self.calls = 0

    def cursor(self, cursor_factory=None):
        self.calls += 1
        if self.calls == 1:
            return _Cursor([(self.commence,)])
        return _Cursor(self.snapshots)


class _UnexpectedConnection:
    def cursor(self, cursor_factory=None):
        raise AssertionError("No close lookup should run for a market without a bet side")


@pytest.mark.parametrize("builder", [mlb_build_full_url, nba_build_full_url])
def test_persisted_odds_urls_redact_api_keys(builder):
    url = builder("https://example.test/odds", {"markets": "h2h", "apiKey": "secret"})
    assert "secret" not in url
    assert "apikey" not in url.lower()
    assert "markets=h2h" in url


@pytest.mark.parametrize("matcher", [mlb_event_matches_as_of_date, nba_event_matches_as_of_date])
def test_event_date_guard_uses_eastern_time(matcher):
    assert matcher(date(2026, 6, 4), "2026-06-05T01:00:00Z")
    assert not matcher(date(2026, 6, 5), "2026-06-05T01:00:00Z")
    assert not matcher(date(2026, 6, 4), None)


@pytest.mark.parametrize(
    ("abstract", "detailed", "expected"),
    [
        ("Final", "Final", "final"),
        ("Final", "Completed Early", "final"),
        ("Final", "Postponed", "postponed"),
        ("Live", "In Progress", "in_progress"),
        ("Preview", "Scheduled", "scheduled"),
    ],
)
def test_statsapi_schedule_status_normalization(abstract, detailed, expected):
    game = {"status": {"abstractGameState": abstract, "detailedState": detailed}}
    assert _status_from_schedule_game(game) == expected


@pytest.mark.parametrize(
    ("played_status", "schedule_status", "scores", "expected"),
    [
        ("UNPLAYED", "NORMAL", (0, 0), "scheduled"),
        ("COMPLETED", "NORMAL", (5, 3), "final"),
        ("LIVE", "NORMAL", (1, 0), "in_progress"),
        ("UNPLAYED", "POSTPONED", (0, 0), "postponed"),
    ],
)
def test_msf_game_status_prefers_schedule_metadata(played_status, schedule_status, scores, expected):
    home_score, away_score = scores
    game = {
        "schedule": {
            "playedStatus": played_status,
            "scheduleStatus": schedule_status,
        },
        "score": {
            "homeScoreTotal": home_score,
            "awayScoreTotal": away_score,
        },
    }
    assert _status_from_game_obj(game, None) == expected


@pytest.mark.parametrize(
    ("market", "side", "entry", "close", "expected"),
    [
        ("run_line", "home", -1.5, -2.5, 1.0),
        ("run_line", "away", 1.5, 2.5, -1.0),
        ("total", "over", 8.5, 9.0, 0.5),
        ("total", "under", 8.5, 8.0, 0.5),
    ],
)
def test_game_line_clv_is_side_aware(market, side, entry, close, expected):
    assert game_line_clv(market, side, entry, close) == expected


def test_game_close_requires_after_lock_near_first_pitch():
    commence = datetime(2026, 6, 4, 23, 0, tzinfo=timezone.utc)
    row = {
        "game_date_et": date(2026, 6, 4),
        "game_slug": "20260604-LAD-NYM",
        "home_team_abbr": "NYM",
        "away_team_abbr": "LAD",
        "bookmaker_key": "fanduel",
        "market": "total",
        "side": "over",
        "market_line": 8.5,
        "market_price": -110,
        "locked_at_utc": commence - timedelta(hours=3),
    }
    snapshots = [
        {
            "event_id": "event-1",
            "fetched_at_utc": commence - timedelta(minutes=30),
            "total_points": 9.0,
            "total_over_price": -105,
        }
    ]
    close = resolve_valid_game_close(_Connection(commence, snapshots), row)
    assert close["valid"] is True
    assert close["status"] == "valid_movement"
    assert close["closing_line"] == 9.0


def test_game_close_marks_stale_snapshot_unknown():
    commence = datetime(2026, 6, 4, 23, 0, tzinfo=timezone.utc)
    lock_at = commence - timedelta(hours=1)
    row = {
        "game_date_et": date(2026, 6, 4),
        "game_slug": "20260604-LAD-NYM",
        "home_team_abbr": "NYM",
        "away_team_abbr": "LAD",
        "bookmaker_key": "fanduel",
        "market": "total",
        "side": "over",
        "market_line": 8.5,
        "market_price": -110,
        "locked_at_utc": lock_at,
    }
    snapshots = [
        {
            "event_id": "event-1",
            "fetched_at_utc": lock_at - timedelta(minutes=1),
            "total_points": 8.5,
            "total_over_price": -110,
        }
    ]
    close = resolve_valid_game_close(_Connection(commence, snapshots), row)
    assert close["valid"] is False
    assert close["unknown_reason"] == "stale_close_before_lock"


def test_game_close_is_blank_when_market_has_no_bet_side():
    close = _resolve_game_close_for_bet(
        _UnexpectedConnection(),
        {"run_line_bet_side": None, "total_bet_side": "over"},
        "run_line",
    )
    assert close["valid"] is None
    assert close["status"] is None
    assert close["unknown_reason"] is None


def test_prop_offer_filter_keeps_the_correct_doubleheader_event():
    offers = [
        {
            "event_id": "game-1",
            "home_team": "DET",
            "away_team": "BAL",
            "commence_time_utc": "2026-06-04T17:00:00Z",
        },
        {
            "event_id": "game-2",
            "home_team": "DET",
            "away_team": "BAL",
            "commence_time_utc": "2026-06-04T23:00:00Z",
        },
    ]
    filtered = filter_prop_offers_for_game(
        offers,
        team_abbr="DET",
        opponent_abbr="BAL",
        start_ts_utc="2026-06-04T22:59:00Z",
    )
    assert [offer["event_id"] for offer in filtered] == ["game-2"]


def test_daily_runners_force_fresh_prop_fetches():
    for relative in ("src/mlb_pipeline/run_daily.py", "src/mlb_pipeline/run_daily_and_notify.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "--force-props" in text


def test_saved_prediction_audit_only_reports_valid_clv():
    text = (ROOT / "src/mlb_pipeline/modeling/real_money_audit_report.py").read_text(encoding="utf-8")
    assert "clv_rl_valid" in text
    assert "clv_total_valid" in text
    assert "MLB Saved-Prediction Gate Replay" in text
    assert "Do not use this report as bankroll evidence" in text


def test_mlb_runners_rebuild_offers_and_predict_before_slow_training():
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    morning_bat = (ROOT / "scripts/mlb_morning.bat").read_text(encoding="utf-8")

    assert notify_text.index('Step("Build Prop Offer Links"') < notify_text.index('Step("Player Prop Projections"')
    assert notify_text.index('Step("Game Predictions"') < notify_text.index('Step("Train Game Models"')
    assert notify_text.index('Step("Rebuild Prop Offer Links"') < notify_text.index('Step("Player Props (pre-game)"')
    assert 'args=("--force-props",)' in notify_text
    assert "--skip-train" in morning_bat

    assert daily_text.index('name="Build prop offer links"') < daily_text.index('name="Predict player props"')
    assert daily_text.index('name="Predict today"') < daily_text.index('name="Train game models"')
    assert daily_text.index('name="Rebuild prop offer links"') < daily_text.index('name="Re-predict player props + post to Discord"')


def test_game_feature_view_selects_latest_valid_event_snapshot():
    text = (ROOT / "sql/MLB006_mlb_game_features.sql").read_text(encoding="utf-8")
    assert "fetched_at_utc DESC" in text
    assert "AT TIME ZONE 'America/New_York'" in text
    assert "JOIN LATERAL" in text
    assert "fetched_at_utc <= NULLIF(commence_time_utc, '')::timestamptz" in text
    assert "WHERE g.status = 'scheduled'" in text
    assert "WHERE g.status IN ('scheduled', 'in_progress')" not in text


def test_prediction_queries_only_use_pregame_odds_and_games():
    game_text = (ROOT / "src/mlb_pipeline/modeling/predict_today.py").read_text(encoding="utf-8")
    prop_text = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    assert "fetched_at_utc <= NULLIF(commence_time_utc, '')::timestamptz" in game_text
    assert prop_text.count("AND status = 'scheduled'") >= 2
    assert prop_text.count("AND (start_ts_utc IS NULL OR start_ts_utc > NOW())") >= 2


def test_non_discord_prop_formatter_receives_reopen_policy():
    prop_text = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    tree = ast.parse(prop_text)
    formatter = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_print_best_bets"
    )
    assert any(arg.arg == "bucket_reopen_policy" for arg in formatter.args.args)

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_print_best_bets"
    ]
    assert calls
    assert all(any(keyword.arg == "bucket_reopen_policy" for keyword in call.keywords) for call in calls)


def test_game_loaders_preserve_terminal_status_and_source_identity():
    statsapi_text = (ROOT / "src/mlb_pipeline/crawler_statsapi.py").read_text(encoding="utf-8")
    parse_text = (ROOT / "src/mlb_pipeline/parse_games.py").read_text(encoding="utf-8")
    bootstrap_text = (ROOT / "sql/MLB000_schema_bootstrap.sql").read_text(encoding="utf-8")
    assert "source_game_id" in statsapi_text
    assert "source_game_id" in bootstrap_text
    assert "status IN ('final', 'postponed', 'cancelled')" in parse_text
    assert "COALESCE(EXCLUDED.home_score, raw.mlb_games.home_score)" in parse_text
    assert "EXCLUDED.source_fetched_at_utc < raw.mlb_games.source_fetched_at_utc" in parse_text


def test_nba_shared_store_parsers_enforce_nba_url_boundary():
    expected_filters = {
        "src/nba_pipeline/parse_games.py": 2,
        "src/nba_pipeline/parse_boxscore.py": 1,
        "src/nba_pipeline/parse_lineup.py": 1,
        "src/nba_pipeline/parse_player_gamelogs.py": 1,
        "src/nba_pipeline/parse_pbp.py": 1,
        "src/nba_pipeline/parse_referees.py": 1,
        "src/nba_pipeline/parse_meta.py": 3,
    }
    for relative, expected_count in expected_filters.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert text.count("url LIKE '%/nba/%'") >= expected_count


def test_nba_latest_snapshot_loaders_prune_stale_rows():
    text = (ROOT / "src/nba_pipeline/parse_meta.py").read_text(encoding="utf-8")
    assert "DELETE FROM raw.nba_venues" in text
    assert "DELETE FROM raw.nba_injuries" in text
    assert "Latest NBA injuries payload is missing a players list" in text
    assert "DELETE FROM raw.nba_injuries_history h" in text


def test_nba_discord_close_runner_includes_outcome_grading():
    text = (ROOT / "src/nba_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    assert 'Step("Update Outcomes + CLV",     "nba_pipeline.modeling.update_outcomes"' in text


def test_nba_runners_refresh_elo_before_materialized_features():
    parse_all_text = (ROOT / "src/nba_pipeline/parse_all.py").read_text(encoding="utf-8")
    assert "_materialize_game_features(_PG_DSN)" not in parse_all_text

    for relative in (
        "src/nba_pipeline/run_daily.py",
        "src/nba_pipeline/run_daily_and_notify.py",
        "src/nba_pipeline/run_nightly.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert text.index("nba_pipeline.compute_elo") < text.index("nba_pipeline.materialize_features")


def test_elo_rebuilds_remove_stale_derived_rows():
    for relative, table in (
        ("src/mlb_pipeline/compute_elo.py", "raw.mlb_elo"),
        ("src/nba_pipeline/compute_elo.py", "raw.nba_elo"),
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert f'DELETE FROM {table}' in text
        assert "WHERE  status = 'final'" in text


def test_nba_game_odds_and_predictions_are_pregame_only():
    view_text = (ROOT / "sql/V026_game_lines_views.sql").read_text(encoding="utf-8")
    game_text = (ROOT / "src/nba_pipeline/modeling/predict_today.py").read_text(encoding="utf-8")
    prop_text = (ROOT / "src/nba_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    assert "l.fetched_at_utc <= l.commence_time_utc" in view_text
    assert "l.commence_time_utc AT TIME ZONE 'America/New_York'" in view_text
    assert "n.bookmaker_key, n.event_id" in view_text
    assert "gl.fetched_at_utc <= gl.commence_time_utc" in game_text
    assert "gpf.start_ts_utc > NOW()" in game_text
    assert prop_text.count("AND start_ts_utc > NOW()") >= 2


def test_nba_grading_treats_exact_lines_as_pushes():
    assert nba_spread_bet_result(5.0, -5.0, "home") is None
    assert nba_spread_bet_result(5.0, -5.0, "away") is None
    assert nba_total_bet_result(220.0, 220.0, "over") is None
    assert nba_total_bet_result(220.0, 220.0, "under") is None
    assert nba_over_hit(10.0, 10.0) is None


def test_nba_clv_close_requires_after_lock_near_tip():
    tip = datetime(2026, 6, 4, 23, 0, tzinfo=timezone.utc)
    row = {
        "predicted_at_utc": tip - timedelta(hours=3),
        "game_start_ts_utc": tip,
    }
    valid = {
        "close_fetched_at_utc": tip - timedelta(minutes=30),
        "commence_time_utc": tip,
    }
    stale = {
        "close_fetched_at_utc": tip - timedelta(hours=4),
        "commence_time_utc": tip,
    }
    assert resolve_valid_nba_close(row, [stale, valid]) is valid
    assert resolve_valid_nba_close(row, [stale]) is None


def test_prop_offer_table_identity_includes_event_id():
    text = (ROOT / "src/mlb_pipeline/modeling/prop_offer_links.py").read_text(encoding="utf-8")
    assert "UNIQUE (as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side)" in text
    assert "ON CONFLICT (as_of_date, event_id, player_name_norm, stat, bookmaker_key, line, side)" in text


def test_game_bankroll_cap_counts_existing_locked_exposure():
    cfg = SimpleNamespace(bankroll_max_daily_exposure_pct=0.01)
    rows = [
        {
            "game_date_et": date(2026, 6, 4),
            "game_slug": "20260604-LAD-NYM",
            "run_line_bet_side": "home",
            "bankroll_candidate_rl": True,
            "stake_pct_rl": 0.005,
            "edge_run_line": 2.0,
        },
        {
            "game_date_et": date(2026, 6, 4),
            "game_slug": "20260604-BOS-NYY",
            "total_bet_side": "over",
            "bankroll_candidate_total": True,
            "stake_pct_total": 0.005,
            "edge_total": 1.0,
        },
    ]
    capped = _cap_game_bankroll_rows(rows, cfg, existing_exposure_pct=0.005)
    assert capped[0]["bankroll_candidate_rl"] is True
    assert capped[1]["bankroll_candidate_total"] is False
    assert "daily_exposure_cap" in capped[1]["bankroll_reasons_total"]


def test_game_bankroll_cap_preserves_exact_lock_and_blocks_side_flip():
    cfg = SimpleNamespace(bankroll_max_daily_exposure_pct=0.02)
    locked = {
        "game_date_et": date(2026, 6, 4),
        "game_slug": "20260604-LAD-NYM",
        "run_line_bet_side": "home",
    }
    rerun = {
        **locked,
        "bankroll_candidate_rl": True,
        "stake_pct_rl": 0.005,
        "edge_run_line": 2.0,
    }
    side_flip = {
        **locked,
        "run_line_bet_side": "away",
        "bankroll_candidate_rl": True,
        "stake_pct_rl": 0.005,
        "edge_run_line": -2.5,
    }
    capped = _cap_game_bankroll_rows(
        [rerun, side_flip],
        cfg,
        existing_exposure_pct=0.005,
        existing_pick_keys={game_bankroll_pick_key(locked, "run_line")},
        existing_risk_slots={game_bankroll_risk_slot(locked, "run_line")},
    )
    assert capped[0]["bankroll_candidate_rl"] is True
    assert capped[1]["bankroll_candidate_rl"] is False
    assert "already_locked_market" in capped[1]["bankroll_reasons_rl"]


def test_prop_bankroll_cap_preserves_exact_lock_and_blocks_changed_offer():
    cfg = SimpleNamespace(bankroll_max_daily_exposure_pct=0.02)
    base = {
        "game_date_et": date(2026, 6, 4),
        "game_slug": "20260604-LAD-NYM",
        "player_id": 123,
        "player_name": "Example Player",
        "stat": "batter_hits",
        "bet_side": "over",
        "book_line": 0.5,
        "bookmaker_key": "fanduel",
        "prediction_key": "prediction-a",
        "prop_offer_id": 10,
        "bankroll_candidate": True,
        "stake_pct": 0.005,
        "ev": 0.10,
        "edge": 0.08,
    }
    changed_offer = {
        **base,
        "prediction_key": "prediction-b",
        "prop_offer_id": 11,
        "ev": 0.20,
    }
    capped = _cap_prop_db_rows(
        [base.copy(), changed_offer],
        cfg,
        existing_exposure_pct=0.005,
        existing_pick_keys={prop_bankroll_pick_key(base)},
        existing_risk_slots={prop_bankroll_risk_slot(base)},
    )
    assert capped[0]["bankroll_candidate"] is True
    assert capped[1]["bankroll_candidate"] is False
    assert "already_locked_player_stat" in capped[1]["bankroll_reasons"]


def test_locked_bankroll_ledger_is_authoritative_and_immutable():
    ledger_text = (ROOT / "src/mlb_pipeline/modeling/bankroll_ledger.py").read_text(encoding="utf-8")
    model_pick_text = (ROOT / "src/mlb_pipeline/modeling/model_pick_ledger.py").read_text(encoding="utf-8")
    assert "max_daily_exposure_pct" in ledger_text
    assert "pg_advisory_xact_lock" in ledger_text
    assert "sync_pending_game_bankroll_metadata" not in ledger_text
    assert "sync_pending_game_model_pick_metadata" not in model_pick_text


def test_task_xml_files_are_parseable_and_use_expected_local_times():
    expected = {
        "MLB-Morning.xml": "2026-01-01T06:00:00",
        "MLB-PreGame.xml": "2026-01-01T08:30:00",
        "MLB-PreGame-Evening.xml": "2026-01-01T14:30:00",
        "MLB-Close.xml": "2026-01-01T07:45:00",
        "MLB-Training.xml": "2026-01-01T22:30:00",
    }
    namespace = {"task": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    for name, boundary in expected.items():
        root = ElementTree.parse(ROOT / "scripts/tasks" / name).getroot()
        value = root.findtext(".//task:StartBoundary", namespaces=namespace)
        assert value == boundary
        assert root.findtext(".//task:RunLevel", namespaces=namespace) == "LeastPrivilege"
        assert root.findtext(".//task:WakeToRun", namespaces=namespace) == "true"

    close_root = ElementTree.parse(ROOT / "scripts/tasks/MLB-Close.xml").getroot()
    assert close_root.findtext(".//task:Interval", namespaces=namespace) == "PT1H"
    assert close_root.findtext(".//task:Duration", namespaces=namespace) == "PT15H"

    day_root = ElementTree.parse(ROOT / "scripts/tasks/MLB-PreGame.xml").getroot()
    evening_root = ElementTree.parse(ROOT / "scripts/tasks/MLB-PreGame-Evening.xml").getroot()
    assert day_root.findtext(".//task:Arguments", namespaces=namespace) == "day_pregame"
    assert evening_root.findtext(".//task:Arguments", namespaces=namespace) == "evening_pregame"


def test_scheduler_installer_uses_task_path_utf8_and_cleans_legacy_mlb_tasks():
    text = (ROOT / "scripts/install_tasks.ps1").read_text(encoding="utf-8")
    assert '[System.Text.Encoding]::UTF8' in text
    assert 'Register-ScheduledTask -TaskPath $taskPath' in text
    assert 'Start-Process -FilePath "powershell.exe" -Verb RunAs' in text
    assert 'Name = "MLB_7am"' in text
    assert 'Name = "mlb_5am"' in text
    assert 'Name = "MLB-PreGame-Day"' in text
    assert 'Name = "MLB-PreGame-Evening"' in text
    assert 'MLB-Training.xml' in text


def test_scheduler_uses_distinct_shadow_lock_phases():
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    batch_text = (ROOT / "scripts/mlb_pregame.bat").read_text(encoding="utf-8")
    assert 'args=("--phase", "morning")' in notify_text
    assert 'args=("--phase", "morning")' in daily_text
    assert '"day_pregame" if now_et.hour < 14 else "evening_pregame"' in notify_text
    assert "--lock-phase %LOCK_PHASE%" in batch_text


def test_scheduler_batch_files_return_pipeline_exit_codes():
    for relative in (
        "scripts/mlb_morning.bat",
        "scripts/mlb_pregame.bat",
        "scripts/mlb_close.bat",
        "scripts/mlb_training.bat",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "set EXITCODE=%ERRORLEVEL%" in text
        assert "exit /b %EXITCODE%" in text


def test_mlb_runners_return_nonzero_when_critical_steps_fail():
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    assert "if pipeline_failed:" in daily_text
    assert "if halted:" in notify_text
    assert "raise SystemExit(1)" in daily_text
    assert "raise SystemExit(1)" in notify_text


def test_close_runner_updates_prop_snapshot_coverage_report():
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    report_text = (ROOT / "src/mlb_pipeline/modeling/prop_snapshot_coverage_report.py").read_text(encoding="utf-8")
    clean_slate_text = (ROOT / "src/mlb_pipeline/modeling/prop_clean_slate.py").read_text(encoding="utf-8")
    assert "prop_snapshot_coverage_report" in daily_text
    assert "prop_snapshot_coverage_report" in notify_text
    assert 'args=("--skip-live", "--force-props")' in daily_text
    assert 'args=("--skip-live", "--force-props")' in notify_text
    assert "CleanSlateThresholds" in report_text
    assert "c.snapshot_at_utc > l.snapshot_at_utc" in clean_slate_text
    assert "l.commence_time_utc - interval '2 hours'" in clean_slate_text
    assert "FROM odds.mlb_player_prop_line_snapshots c" in clean_slate_text


def test_prop_clv_refresh_and_walk_forward_reports_are_scheduled():
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    replay_text = (ROOT / "src/mlb_pipeline/modeling/prop_replay.py").read_text(encoding="utf-8")
    assert "refresh_prop_replay_clv" in replay_text
    assert "mlb_pipeline.modeling.refresh_prop_replay_clv" in daily_text
    assert "mlb_pipeline.modeling.refresh_prop_replay_clv" in notify_text
    assert "mlb_pipeline.modeling.prop_walk_forward_accuracy_report" in daily_text
    assert "mlb_pipeline.modeling.prop_walk_forward_accuracy_report" in notify_text
    assert "mlb_pipeline.modeling.prop_shadow_selector" in daily_text
    assert "mlb_pipeline.modeling.prop_shadow_selector" in notify_text
    assert "mlb_pipeline.modeling.prop_opportunity_feature_report" in daily_text
    assert "mlb_pipeline.modeling.prop_opportunity_feature_report" in notify_text
    assert 'args=("--include-pending",)' in daily_text
    assert 'args=("--include-pending",)' in notify_text


def test_prop_offer_health_and_selector_sections_are_reported():
    predict_text = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    selector_text = (ROOT / "src/mlb_pipeline/modeling/prop_shadow_selector.py").read_text(encoding="utf-8")
    assert "Prop odds not loaded yet" in predict_text
    assert "best_common_paper_rows" in selector_text
    assert "no_bet_top_rows" in selector_text
    assert "lottery_top_rows" in selector_text
    assert "closest_to_promotion_buckets" in selector_text


def test_ledger_schema_checks_are_not_prediction_time_ddl_by_default():
    ledger_text = (ROOT / "src/mlb_pipeline/modeling/bankroll_ledger.py").read_text(encoding="utf-8")
    snapshot_text = (ROOT / "src/mlb_pipeline/modeling/prop_offer_snapshots.py").read_text(encoding="utf-8")
    assert "_bankroll_ledger_has_required_columns" in ledger_text
    assert "_bankroll_ledger_exists(conn) and _bankroll_ledger_has_required_columns(conn)" in ledger_text
    assert "SET LOCAL lock_timeout = '2s'" in ledger_text
    assert "SET LOCAL lock_timeout = '2s'" in snapshot_text


def test_prop_no_vig_pairing_uses_event_fallbacks():
    links_text = (ROOT / "src/mlb_pipeline/modeling/prop_offer_links.py").read_text(encoding="utf-8")
    training_text = (ROOT / "src/mlb_pipeline/modeling/prop_market_training.py").read_text(encoding="utf-8")
    quality_text = (ROOT / "src/mlb_pipeline/modeling/prop_target_quality_report.py").read_text(encoding="utf-8")
    assert "event_id IS NOT NULL" not in links_text
    assert "fallback_event" in links_text
    assert "(selected_offer.id IS NULL OR pair_same.id IS NULL)" in training_text
    assert "(selected_offer.id IS NULL OR pair_cross.id IS NULL)" in training_text
    assert "synthetic_fanduel_over_only_complement" in training_text
    assert "synthetic_fanduel_over_only" in training_text
    assert "Synthetic Pair" in quality_text


def test_prop_opportunity_features_feed_training_and_betting_layers():
    training_text = (ROOT / "src/mlb_pipeline/modeling/prop_market_training.py").read_text(encoding="utf-8")
    betting_text = (ROOT / "src/mlb_pipeline/modeling/prop_betting_layer.py").read_text(encoding="utf-8")
    direct_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_direct_side_models.py").read_text(encoding="utf-8")
    compare_text = (ROOT / "src/mlb_pipeline/modeling/compare_prop_probability_variants.py").read_text(encoding="utf-8")
    predict_text = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    for column in ("projected_pa", "projected_bf", "projected_pitch_count", "confirmed_batting_order"):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in training_text
        assert f'"{column}"' in betting_text
        assert f'"{column}"' in direct_text
        assert f"{column}::float" in compare_text
    assert "opportunity_features=pitcher_opportunity" in predict_text
    assert "opportunity_features=batter_opportunity" in predict_text
    report_text = (ROOT / "src/mlb_pipeline/modeling/prop_opportunity_feature_report.py").read_text(encoding="utf-8")
    assert "Projection Accuracy" in report_text
    assert "actual_pa::float AS actual_pa" in report_text
    assert "actual_bf::float AS actual_bf" in report_text


def test_snapshot_coverage_requires_a_meaningful_valid_slate():
    assert slate_qualifies(
        side_locks=1000,
        valid_side_locks=300,
        min_valid_locks=100,
        min_valid_coverage=0.25,
    )
    assert not slate_qualifies(
        side_locks=1000,
        valid_side_locks=1,
        min_valid_locks=100,
        min_valid_coverage=0.25,
    )
    assert not slate_qualifies(
        side_locks=1000,
        valid_side_locks=200,
        min_valid_locks=100,
        min_valid_coverage=0.25,
    )


def test_clean_shadow_slate_blocks_stale_and_missing_clv_collection():
    thresholds = CleanSlateThresholds(
        min_side_locks=100,
        min_valid_side_locks=100,
        min_valid_coverage=0.25,
        max_missing_lock_rate=0.02,
        max_stale_close_rate=0.05,
    )
    clean = {
        "side_lock_rows": 500,
        "valid_side_locks": 200,
        "close_times": 2,
        "training_rows": 500,
        "missing_lock_examples": 5,
        "stale_close_examples": 10,
    }
    assert clean_slate_qualifies(clean, thresholds)
    assert not clean_slate_qualifies({**clean, "missing_lock_examples": 30}, thresholds)
    assert not clean_slate_qualifies({**clean, "stale_close_examples": 40}, thresholds)


def test_prop_reopen_policy_requires_clean_slate_ladder_evidence():
    policy_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bucket_reopen_policy.py").read_text(encoding="utf-8")
    report_text = (ROOT / "src/mlb_pipeline/modeling/prop_bucket_promotion_report.py").read_text(encoding="utf-8")
    assert "clean_unique_dates<" in policy_text
    assert "starter_min_clean_unique_dates" in policy_text
    assert "bankroll_min_clean_unique_dates" in policy_text
    assert "Clean Dates" in report_text


def test_prop_bootstrap_micro_and_ladder_artifacts_are_unified():
    policy_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bucket_reopen_policy.py").read_text(encoding="utf-8")
    readiness_text = (ROOT / "src/mlb_pipeline/modeling/prop_real_money_readiness_report.py").read_text(encoding="utf-8")
    selector_text = (ROOT / "src/mlb_pipeline/modeling/prop_shadow_selector.py").read_text(encoding="utf-8")
    promotion_text = (ROOT / "src/mlb_pipeline/modeling/prop_bucket_promotion_report.py").read_text(encoding="utf-8")
    assert "enable_bootstrap_micro" in policy_text
    assert "bootstrap_micro_only" in policy_text
    assert "bootstrap_micro_reasons" in promotion_text
    assert "_OPEN_LADDER_TIERS = {\"micro\", \"starter\", \"bankroll\"}" in readiness_text
    assert "_apply_ladder_policy(scores, cfg)" in readiness_text
    assert "trust_status in {\"bankroll\", \"starter\", \"micro\"}" in selector_text


def test_prop_selector_uses_distribution_opportunity_and_bucket_clv_gates():
    selector_text = (ROOT / "src/mlb_pipeline/modeling/prop_shadow_selector.py").read_text(encoding="utf-8")
    walk_forward_text = (ROOT / "src/mlb_pipeline/modeling/prop_walk_forward_accuracy_report.py").read_text(encoding="utf-8")
    assert "variant == \"distribution\"" in selector_text
    assert "_distribution_side_prob" in selector_text
    assert "_compound_tb_over_prob" in selector_text
    assert "_compound_tb_over_prob" in walk_forward_text
    assert "compound PA/single/double/triple/HR" in walk_forward_text
    assert "opportunity_not_confirming" in selector_text
    assert "bucket_history_not_confirming" in selector_text
    assert "bucket_clv_beat_low" in selector_text
    assert "{\"walk_forward_blend\", \"market_no_vig\", \"distribution\"}" in walk_forward_text


def test_prop_bookability_avoids_clv_reason_leakage_and_reports_repair():
    bookability_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bookability_model.py").read_text(encoding="utf-8")
    categorical_block = bookability_text.split("_CATEGORICAL = [", 1)[1].split("]", 1)[0]
    selector_text = (ROOT / "src/mlb_pipeline/modeling/prop_shadow_selector.py").read_text(encoding="utf-8")
    repair_text = (ROOT / "src/mlb_pipeline/modeling/prop_bucket_repair_report.py").read_text(encoding="utf-8")
    run_daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    opportunity_text = (ROOT / "src/mlb_pipeline/modeling/prop_opportunity_feature_report.py").read_text(encoding="utf-8")
    assert "clv_unknown_reason" not in categorical_block
    assert "`clv_unknown_reason` is label-only" in bookability_text
    assert "selected_scoring_method" in bookability_text
    assert "_bookability_empirical_score" in selector_text
    assert "bookability_model_usable" in selector_text
    assert "Prediction Gap Audit" in bookability_text
    assert "repair_hitter_tb_distribution" in repair_text
    assert "bookability_rate" in repair_text
    assert "Prop bucket repair report" in run_daily_text
    assert "mlb_pipeline.modeling.prop_bucket_repair_report" in notify_text
    assert "confirmed_lineup" in opportunity_text
    assert "low_pa_miss_rate" in opportunity_text
