"""Regression checks for odds date buckets, snapshot safety, and scheduler wiring."""
from __future__ import annotations

import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pandas as pd
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
from mlb_pipeline.modeling.predict_player_props import (
    PredictConfig,
    _cap_prop_db_rows,
    _offer_line_data,
    _print_discord,
)
from mlb_pipeline.modeling.prop_clean_slate import CleanSlateThresholds, clean_slate_qualifies
from mlb_pipeline.modeling.prop_real_money_eligibility import PROP_REAL_MONEY_ELIGIBILITY_START_DATE
from mlb_pipeline.modeling.train_hitter_player_game_outcome_models import (
    _apply_tb_hr_tail_logit_offset,
    _predict_hierarchical_event_probabilities,
    add_leakage_safe_player_priors,
    convolve_hitter_outcomes,
    projected_pa_pmf,
)
from mlb_pipeline.modeling.train_prop_distribution_models import (
    _add_offer_group_weights,
    _apply_true_pair_hitter_line_calibrators,
    _blend_tb_state_curve,
    _empirical_bayes_component_multiplier,
    _fit_walk_forward_calibrator,
    _purge_player_game_overlap,
    _tb_state_over_probability,
)
from mlb_pipeline.modeling.train_prop_opportunity_models import add_pitcher_history_features
from mlb_pipeline.modeling.predict_today import _cap_game_bankroll_rows
from mlb_pipeline.modeling.prop_offer_links import filter_prop_offers_for_game
from mlb_pipeline.modeling.prop_snapshot_coverage_report import slate_qualifies
from mlb_pipeline.modeling.update_outcomes import _resolve_game_close_for_bet
from mlb_pipeline.parse_games import _status_from_game_obj
from mlb_pipeline.parse_oddsapi import _event_matches_as_of_date as mlb_event_matches_as_of_date
from mlb_pipeline.parse_oddsapi import _iter_prop_rows as mlb_iter_prop_rows
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


def test_mlb_runners_kill_subprocess_trees_on_timeout():
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    helper_text = (ROOT / "src/mlb_pipeline/subprocess_utils.py").read_text(encoding="utf-8")
    assert "run_subprocess_tree" in daily_text
    assert "run_subprocess_tree" in notify_text
    assert "taskkill" in helper_text
    assert "/T" in helper_text


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


def test_real_money_close_quality_defaults_are_strict():
    policy_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bucket_reopen_policy.py").read_text(encoding="utf-8")
    readiness_text = (ROOT / "src/mlb_pipeline/modeling/prop_real_money_readiness_report.py").read_text(encoding="utf-8")
    thresholds = CleanSlateThresholds()
    assert thresholds.min_valid_coverage == pytest.approx(0.90)
    assert thresholds.max_stale_close_rate == pytest.approx(0.02)
    assert "clean_slate_min_valid_coverage: float = 0.90" in policy_text
    assert "clean_slate_max_stale_close_rate: float = 0.02" in policy_text
    assert "min_valid_close_coverage: float = 0.90" in readiness_text
    assert "max_stale_close_rate: float = 0.02" in readiness_text
    assert "--min-valid-close-coverage" in readiness_text
    assert "--max-stale-close-rate" in readiness_text


def test_prop_close_resolver_distinguishes_line_disappearance():
    snapshot_text = (ROOT / "src/mlb_pipeline/modeling/prop_offer_snapshots.py").read_text(encoding="utf-8")
    bookability_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bookability_model.py").read_text(encoding="utf-8")
    assert "line_disappeared_at_close" in snapshot_text
    assert "same_book_other_line_at_close" in snapshot_text
    assert "AND line <> %s" in snapshot_text
    assert '"line_disappeared_at_close"' in bookability_text


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
    assert 'args=("--include-pending", "--ensure-schema")' in daily_text
    assert 'args=("--include-pending", "--ensure-schema")' in notify_text
    assert 'args=("--lookback-days", "3", "--include-pending", "--no-replace")' in daily_text
    assert 'args=("--lookback-days", "3", "--include-pending", "--no-replace")' in notify_text


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


def test_fanduel_alternate_props_preserve_true_under_prices():
    fetched_at = datetime(2026, 6, 17, 16, 0, tzinfo=timezone.utc)
    event_payload = {
        "id": "evt_fd_alt",
        "commence_time": "2026-06-17T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "batter_total_bases_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "point": 1.5, "price": 120, "link": "over-link"},
                            {"name": "Under", "description": "Aaron Judge", "point": 1.5, "price": -145, "link": "under-link"},
                            {"name": "Over", "description": "Aaron Judge", "point": 2.5, "price": 240, "link": "over-alt"},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_total_bases_alternate",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "point": 1.5, "price": 115},
                            {"name": "Under", "description": "Aaron Judge", "point": 1.5, "price": -135},
                        ],
                    }
                ],
            },
        ],
    }
    rows = list(mlb_iter_prop_rows(date(2026, 6, 17), fetched_at, event_payload))
    assert len(rows) == 2
    main = [row for row in rows if float(row[10]) == 1.5][0]
    assert main[4] == "fanduel"
    assert main[9] == "batter_total_bases"
    assert main[11] == 120
    assert main[12] == -145
    assert main[13] == "over-link"
    assert main[14] == "under-link"


def test_real_money_policy_excludes_synthetic_fanduel_market_evidence():
    policy_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bucket_reopen_policy.py").read_text(encoding="utf-8")
    readiness_text = (ROOT / "src/mlb_pipeline/modeling/prop_real_money_readiness_report.py").read_text(encoding="utf-8")
    selector_text = (ROOT / "src/mlb_pipeline/modeling/prop_shadow_selector.py").read_text(encoding="utf-8")
    for text in (policy_text, readiness_text):
        assert "clean_market_pair_flag::float" in text
        assert "synthetic_fanduel_over_only" in text
        assert "synthetic_fanduel_over_only_complement" in text
        assert "synthetic_pair_flag::float" in text
        assert "batter_hits','batter_total_bases','batter_home_runs" in text
    assert "_is_fanduel_synthetic_hitter_evidence" in selector_text
    assert "fanduel_synthetic_market_evidence" in selector_text


def test_prop_real_money_kill_switch_is_wired_to_predictions_and_scheduler():
    kill_text = (ROOT / "src/mlb_pipeline/modeling/prop_real_money_kill_switch.py").read_text(encoding="utf-8")
    predict_text = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    notify_text = (ROOT / "src/mlb_pipeline/run_daily_and_notify.py").read_text(encoding="utf-8")
    assert "valid_close_coverage<" in kill_text
    assert "stale_close_rate>" in kill_text
    assert "no_open_prop_buckets" in kill_text
    assert "artifact_stale" in kill_text
    assert "load_prop_kill_switch_state" in predict_text
    assert "_apply_prop_real_money_kill_switch(db_rows, cfg)" in predict_text
    assert "minimum_acceptable_price_missing" in predict_text
    assert "bet_link_missing" in predict_text
    assert "mlb_pipeline.modeling.prop_real_money_kill_switch" in daily_text
    assert "mlb_pipeline.modeling.prop_real_money_kill_switch" in notify_text


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
    distribution_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_distribution_models.py").read_text(encoding="utf-8")
    hitter_outcome_text = (ROOT / "src/mlb_pipeline/modeling/train_hitter_player_game_outcome_models.py").read_text(encoding="utf-8")
    assert "variant == \"distribution\"" in selector_text
    assert "_distribution_side_prob" in selector_text
    assert "_compound_tb_over_prob" in selector_text
    assert "_compound_tb_over_prob" in walk_forward_text
    assert "compound PA/single/double/triple/HR" in walk_forward_text
    assert 'artifact["status"] = "loaded" if artifact["production_eligible"] else "diagnostic_candidate"' in distribution_text
    assert '"production_eligible": bool(artifact.get("production_eligible"))' in distribution_text
    assert "usable_for_distribution" in hitter_outcome_text
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
    assert "_bookability_empirical_components" in selector_text
    assert "_bookability_score" in selector_text
    assert "bookability_model_usable" in selector_text
    assert "Prediction Gap Audit" in bookability_text
    assert "repair_hitter_tb_distribution" in repair_text
    assert "bookability_rate" in repair_text
    assert "Prop bucket repair report" in run_daily_text
    assert "mlb_pipeline.modeling.prop_bucket_repair_report" in notify_text
    assert "confirmed_lineup" in opportunity_text
    assert "low_pa_miss_rate" in opportunity_text


def test_prop_promotion_uses_fixed_prospective_eligibility_cohort():
    policy_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_bucket_reopen_policy.py").read_text(encoding="utf-8")
    readiness_text = (ROOT / "src/mlb_pipeline/modeling/prop_real_money_readiness_report.py").read_text(encoding="utf-8")
    assert PROP_REAL_MONEY_ELIGIBILITY_START_DATE == date(2026, 6, 19)
    assert "legacy_audit_rows" in policy_text
    assert 'legacy_df["game_date_et"] >= cfg.eligibility_start_date' in policy_text
    assert "eligibility_start_date" in readiness_text
    assert "legacy_close_quality" in readiness_text


class _ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, values):
        p = np.full(len(values), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


def test_hierarchical_event_curve_is_coherent_and_normalized():
    models = {
        "hit": _ConstantProbabilityModel(0.25),
        "walk_given_non_hit": _ConstantProbabilityModel(0.10),
        "xbh_given_hit": _ConstantProbabilityModel(0.40),
        "hr_given_xbh": _ConstantProbabilityModel(0.30),
        "triple_given_non_hr_xbh": _ConstantProbabilityModel(0.10),
    }
    frame = pd.DataFrame({"projected_pa": [4.2, 3.8], "team_abbr": ["NYY", "LAD"]})
    probs = _predict_hierarchical_event_probabilities(
        models,
        frame,
        numeric_features=["projected_pa"],
        categorical_features=["team_abbr"],
    )
    assert np.allclose(probs.sum(axis=1).to_numpy(), 1.0)
    assert (probs >= 0.0).all().all()
    assert (probs["p_double"] > probs["p_triple"]).all()


def test_player_event_priors_are_empirical_bayes_and_date_shifted():
    frame = pd.DataFrame(
        {
            "player_id": [42, 42, 42],
            "game_date_et": [date(2026, 6, 1), date(2026, 6, 1), date(2026, 6, 2)],
            "actual_pa": [4, 3, 4],
            "actual_walks": [0, 0, 0],
            "actual_singles": [2, 1, 0],
            "actual_doubles": [0, 0, 0],
            "actual_triples": [0, 0, 0],
            "actual_home_runs": [0, 0, 0],
        }
    )
    enhanced, state = add_leakage_safe_player_priors(frame)
    first_day = enhanced.loc[pd.to_datetime(enhanced["game_date_et"]).dt.date == date(2026, 6, 1)]
    next_day = enhanced.loc[pd.to_datetime(enhanced["game_date_et"]).dt.date == date(2026, 6, 2)]
    assert first_day["player_prior_pa"].eq(0.0).all()
    assert next_day["player_prior_pa"].eq(7.0).all()
    assert next_day["player_prior_hit_rate"].iloc[0] > first_day["player_prior_hit_rate"].iloc[0]
    assert state["42"]["player_prior_pa"] == 11.0


def test_projected_pa_convolution_preserves_explicit_tb_probability_mass():
    row = pd.Series({"confirmed_lineup_flag": 1, "lineup_slot": 3, "home_away": "home"})
    pa_pmf = projected_pa_pmf(4.25, row, {"global": {"bias": 0.0, "sigma": 0.65}})
    curve = convolve_hitter_outcomes(
        {
            "p_out": 0.66,
            "p_walk": 0.08,
            "p_single": 0.15,
            "p_double": 0.06,
            "p_triple": 0.01,
            "p_hr": 0.04,
        },
        pa_pmf,
    )
    assert sum(pa_pmf.values()) == pytest.approx(1.0)
    assert sum(curve["tb_pmf"].values()) == pytest.approx(1.0)
    assert sum(curve["tb_states"].values()) == pytest.approx(1.0)
    assert curve["tb_states"]["tb_4_plus_hr"] > 0.0
    assert curve["tb_states"]["tb_4_plus_non_hr"] > 0.0


def test_two_part_pa_distribution_preserves_low_pa_risk_mass():
    row = pd.Series({
        "lineup_slot": 7,
        "is_home": True,
        "pa_low_probability": 0.24,
        "pa_normal_mean": 4.1,
    })
    uncertainty = {
        "global": {"bias": 0.0, "sigma": 0.65, "low_pa_state_probs": {"0": 0.10, "1": 0.20, "2": 0.70}}
    }
    pmf = projected_pa_pmf(3.8, row, uncertainty)
    assert sum(pmf.values()) == pytest.approx(1.0)
    assert sum(pmf[n] for n in (0, 1, 2)) == pytest.approx(0.24)
    assert sum(pmf[n] for n in range(3, 8)) == pytest.approx(0.76)


def test_grouped_offer_weights_equalize_player_game_outcomes_and_purge_overlap():
    frame = pd.DataFrame({
        "game_slug": ["g1"] * 10 + ["g2"] * 2,
        "player_id": [1] * 10 + [2] * 2,
    })
    weighted = _add_offer_group_weights(frame)
    totals = weighted.groupby("player_game_group")["offer_group_weight"].sum()
    assert totals.iloc[0] == pytest.approx(totals.iloc[1])

    train = pd.DataFrame({"game_slug": ["g1", "g2"], "player_id": [1, 2]})
    holdout = pd.DataFrame({"game_slug": ["g2", "g3"], "player_id": [2, 3]})
    purged_train, _, purged = _purge_player_game_overlap(train, holdout)
    assert purged == 1
    assert set(purged_train["game_slug"]) == {"g1"}


def test_walk_forward_calibrator_auto_disables_without_brier_gain():
    rows = []
    for day in range(10):
        for player in range(20):
            rows.append({
                "game_date_et": date(2026, 5, 1) + timedelta(days=day),
                "game_slug": f"g{day}",
                "player_id": player,
                "p_distribution_side": 0.5,
                "target": (day + player) % 2,
            })
    rec = _fit_walk_forward_calibrator(_add_offer_group_weights(pd.DataFrame(rows)), "p_distribution_side")
    assert rec["enabled"] is False
    assert rec["reason"] == "no_temporal_brier_gain"


def test_tb_state_residual_blend_preserves_probability_mass():
    curve = convolve_hitter_outcomes(
        {"p_out": 0.65, "p_walk": 0.08, "p_single": 0.16, "p_double": 0.06, "p_triple": 0.01, "p_hr": 0.04},
        {4: 1.0},
    )
    row = pd.Series({
        "p_tb_state_tb_0": 0.30,
        "p_tb_state_tb_1": 0.22,
        "p_tb_state_tb_2_3": 0.20,
        "p_tb_state_tb_4_plus_hr": 0.25,
        "p_tb_state_tb_4_plus_non_hr": 0.03,
    })
    blended = _blend_tb_state_curve(curve, row, 0.5)
    assert sum(blended["tb_pmf"].values()) == pytest.approx(1.0)
    assert sum(blended["tb_states"].values()) == pytest.approx(1.0)
    assert blended["tb_states"]["tb_4_plus_hr"] > curve["tb_states"]["tb_4_plus_hr"]


def test_tb_hr_tail_logit_repair_preserves_mass_and_only_lifts_tail():
    probabilities = pd.DataFrame({
        "tb_0": [0.45, 0.35],
        "tb_1": [0.25, 0.25],
        "tb_2_3": [0.20, 0.22],
        "tb_4_plus_hr": [0.08, 0.14],
        "tb_4_plus_non_hr": [0.02, 0.04],
    })
    repaired = _apply_tb_hr_tail_logit_offset(probabilities, 0.30)
    assert np.allclose(repaired.sum(axis=1), 1.0)
    assert (repaired["tb_4_plus_hr"] > probabilities["tb_4_plus_hr"]).all()
    assert (repaired.drop(columns="tb_4_plus_hr").sum(axis=1) < probabilities.drop(columns="tb_4_plus_hr").sum(axis=1)).all()


def test_direct_tb_state_pricing_uses_state_boundaries():
    curve = {
        "tb_states": {
            "tb_0": 0.40,
            "tb_1": 0.25,
            "tb_2_3": 0.20,
            "tb_4_plus_hr": 0.12,
            "tb_4_plus_non_hr": 0.03,
        },
        "tb_pmf": {0: 0.40, 1: 0.25, 2: 0.12, 3: 0.08, 4: 0.15},
    }
    assert _tb_state_over_probability(0.5, curve) == pytest.approx(0.60)
    assert _tb_state_over_probability(1.5, curve) == pytest.approx(0.35)
    assert _tb_state_over_probability(2.5, curve) == pytest.approx(0.23)
    assert _tb_state_over_probability(3.5, curve) == pytest.approx(0.15)


def test_line_calibration_applies_only_to_true_pair_exact_line():
    frame = pd.DataFrame({
        "market": ["batter_total_bases", "batter_total_bases"],
        "side": ["over", "over"],
        "market_line": [1.5, 1.5],
        "true_pair_flag": [1.0, 0.0],
        "synthetic_pair_flag": [0.0, 1.0],
        "market_prob_source": ["paired_no_vig", "synthetic_fanduel_over_only"],
        "p_distribution_side": [0.30, 0.30],
    })
    calibrators = {
        "overall_holdout_enabled": True,
        "groups": {
            "batter_total_bases|over|TB 1.5": {
                "enabled": True,
                "holdout_enabled": True,
                "probability_col": "p_distribution_side",
                "model": {"method": "isotonic", "x_thresholds": [0.0, 1.0], "y_thresholds": [0.1, 0.9]},
            }
        },
    }
    calibrated = _apply_true_pair_hitter_line_calibrators(frame, frame["p_distribution_side"], calibrators)
    assert calibrated.iloc[0] == pytest.approx(0.34)
    assert calibrated.iloc[1] == pytest.approx(0.30)


def test_tb_component_empirical_bayes_can_repair_supported_double_bias():
    supported = _empirical_bayes_component_multiplier(820.0, 360.0, 20000.0, shrink_exposure=700.0)
    sparse = _empirical_bayes_component_multiplier(2.0, 1.0, 10.0, shrink_exposure=700.0)
    assert supported > 1.45
    assert sparse < 1.10


def test_tb_hr_real_candidates_require_true_pair_line_production_gate():
    distribution_text = (ROOT / "src/mlb_pipeline/modeling/train_prop_distribution_models.py").read_text(encoding="utf-8")
    selector_text = (ROOT / "src/mlb_pipeline/modeling/prop_shadow_selector.py").read_text(encoding="utf-8")
    assert "holdout_true_pair_non_synthetic_only" in distribution_text
    assert "tb_hr_line_production_gates" in distribution_text
    assert "tb_hr_line_confirms" in selector_text
    assert "tb_hr_line_production_gate_failed" in selector_text


def test_discord_formatter_receives_exact_bucket_reopen_policy():
    source = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    formatter = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_print_discord"
    )
    assert "bucket_reopen_policy" in [arg.arg for arg in formatter.args.args]
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_print_discord"
    ]
    assert calls
    assert all("bucket_reopen_policy" in {kw.arg for kw in call.keywords} for call in calls)


def test_discord_formatter_handles_empty_bankroll_path(monkeypatch, capsys):
    import mlb_pipeline.modeling.predict_player_props as props_module

    class _Context:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setenv("DISCORD_FORMAT", "1")
    monkeypatch.setattr(props_module, "format_record_summary", lambda **_kwargs: "")
    monkeypatch.setattr(props_module.psycopg2, "connect", lambda *_args, **_kwargs: _Context())
    monkeypatch.setattr(props_module, "locked_bankroll_state", lambda *_args, **_kwargs: (0.0, set(), set()))
    result = _print_discord(
        [],
        [],
        {},
        {},
        PredictConfig(et_date=date(2026, 6, 18)),
        db_rows=[],
        bucket_reopen_policy={},
    )
    output = capsys.readouterr().out
    assert result == []
    assert "No bankroll-qualified player props today" in output
    assert "Top 10 Paper Strikeouts" in output
    assert "Top 10 Paper Total Bases" in output
    assert "Top 10 Paper Hits" in output
    assert "Top 10 Paper Home Runs" in output
    assert "LOTTERY / RESEARCH" not in output
    assert "ALT-LINE LOTTERY / RESEARCH" not in output
    assert "NO-BET SUMMARY" in output


def test_clv_v2_live_offer_features_use_consensus_and_exact_open_price():
    offers = [
        {
            "id": 1, "side": "over", "line": 5.5, "price": -105,
            "bookmaker_key": "draftkings", "open_price": 110,
            "open_line": 5.5, "open_exact_line": True,
        },
        {"id": 2, "side": "over", "line": 5.5, "price": -115, "bookmaker_key": "fanduel"},
        {"id": 3, "side": "under", "line": 5.5, "price": -115, "bookmaker_key": "draftkings"},
    ]
    data = _offer_line_data(offers[0], offers)
    assert data["consensus_book_count"] == 2.0
    assert data["lock_same_book_pair_available"] == 1.0
    assert data["open_to_lock_prob_move"] > 0.0
    assert data["open_to_lock_line_move_side"] == pytest.approx(0.0)


def test_pitcher_history_features_never_use_current_start_outcome():
    frame = pd.DataFrame({
        "player_id": [7, 7, 7],
        "game_slug": ["a", "b", "c"],
        "game_date_et": [date(2026, 5, 1), date(2026, 5, 7), date(2026, 5, 13)],
        "team_abbr": ["SEA", "SEA", "SEA"],
        "actual_bf": [18.0, 24.0, 30.0],
        "actual_pitch_count_proxy": [72.0, 96.0, 120.0],
        "actual_ip": [4.0, 6.0, 8.0],
    })
    enriched = add_pitcher_history_features(frame).sort_values("game_date_et")
    assert pd.isna(enriched.iloc[0]["last_bf"])
    assert enriched.iloc[1]["last_bf"] == pytest.approx(18.0)
    assert enriched.iloc[2]["recent_bf_mean_3"] == pytest.approx(21.0)
    assert enriched.iloc[2]["days_rest"] == pytest.approx(6.0)


def test_head_ablation_runs_before_hitter_model_training():
    source = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    assert source.index("Hitter event feature ablation report") < source.index("Train hitter player-game outcome models")


def test_clv_v2_is_true_pair_only_and_has_movement_features():
    source = (ROOT / "src/mlb_pipeline/modeling/train_prop_market_residual_models.py").read_text(encoding="utf-8")
    assert "open_to_lock_prob_move" in source
    assert "consensus_price_dispersion" in source
    assert "book_lead_lag_prob" in source
    assert "COALESCE(e.true_pair_flag::float" in source
    assert "COALESCE(e.synthetic_pair_flag::float" in source


def test_run_daily_step_names_are_unique():
    import re
    daily_text = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    # Scope to the normal full daily run block only (the `else:` branch that starts
    # after the pre_game and close conditional branches).  Pre-game and close paths
    # intentionally reuse step names that also appear in the main flow, so checking
    # the whole file would produce false positives.
    main_start = daily_text.index("# ── Normal full daily run")
    main_text = daily_text[main_start:]
    names = re.findall(r'name="([^"]+)"', main_text)
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, f"Duplicate step names in normal daily flow of run_daily.py: {duplicates}"


def test_series_game_number_uses_strict_lt_boundary():
    sql = (ROOT / "sql/MLB006_mlb_game_features.sql").read_text(encoding="utf-8")
    # Must use strict < so training and inference compute series position identically.
    # Inclusive <= would count today's scheduled/in-progress games in the training view,
    # creating a training/inference inconsistency.
    assert "AND g2.game_date_et  < g.game_date_et" in sql
    # Only final games should count toward prior-series position
    assert "g2.status IN ('final', 'scheduled', 'in_progress')" not in sql
    # The + 1 offset must be present (prior completed games + today = series position)
    assert ")::INTEGER + 1                                   AS series_game_number" in sql


def test_batter_ab_floor_gate_present_in_both_train_and_predict():
    """Regression: batter AB floor (ab_avg_10 >= 2.5, n_games_prev_10 >= 3) must exist in
    both training and inference SQL so the model never scores batters it never trained on.
    Lower values cause OVER TB predictions at 0.5 lines to hit only ~46% (below breakeven)."""
    train_src = (ROOT / "src/mlb_pipeline/modeling/train_player_prop_models.py").read_text(encoding="utf-8")
    pred_src  = (ROOT / "src/mlb_pipeline/modeling/predict_player_props.py").read_text(encoding="utf-8")

    # Training SQL must hard-code the floor
    assert "AND b.ab_avg_10 >= 2.5" in train_src, "Training SQL missing ab_avg_10 >= 2.5 gate"
    assert "AND b.n_games_prev_10 >= 3" in train_src, "Training SQL missing n_games_prev_10 >= 3 gate"

    # Inference SQL must parametrize the floor (applied via PredictConfig.min_ab_avg_10)
    assert "br.ab_avg_10 >= %(min_ab_avg_10)s" in pred_src, "Prediction SQL missing ab_avg_10 gate"
    assert "br.n_games_prev_10 >= %(min_n_games)s" in pred_src, "Prediction SQL missing n_games_prev_10 gate"

    # Default threshold must match training value so they stay in sync
    assert "min_ab_avg_10: float = 2.5" in pred_src, "PredictConfig.min_ab_avg_10 default ≠ 2.5"
    assert "min_n_games: int = 3" in pred_src, "PredictConfig.min_n_games default ≠ 3"


def test_crawler_weather_is_scheduled_in_crawl_block():
    """Regression: crawler_weather must be wired into the run_daily crawl block so
    game-time temperature and wind features are refreshed on every daily run."""
    source = (ROOT / "src/mlb_pipeline/run_daily.py").read_text(encoding="utf-8")
    # Must appear after the normal crawl block header and before the parse block
    crawl_start = source.index("# ── Normal full daily run")
    parse_start = source.index("if not args.skip_parse:", crawl_start)
    crawl_block = source[crawl_start:parse_start]
    assert "mlb_pipeline.crawler_weather" in crawl_block, (
        "crawler_weather not found in the normal daily crawl block of run_daily.py"
    )
