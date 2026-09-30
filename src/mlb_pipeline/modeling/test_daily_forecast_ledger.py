from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from mlb_pipeline.modeling import daily_forecast_ledger as ledger
from mlb_pipeline.modeling.daily_forecast_projection_audit import AuditConfig, _stat_summary
from mlb_pipeline.modeling.daily_forecast_projection_audit import _is_shadow_model, _non_shadow_df
from mlb_pipeline.modeling.forecast_repair_error_report import _add_error_slice_columns, _tb_diagnostic, decompose_count_error
from mlb_pipeline.modeling.model_release import (
    DEFAULT_CANONICAL_FORECAST_PHASE,
    DEFAULT_HITTER_MODEL_RELEASE_ID,
    DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID,
    canonical_forecast_phase,
    hitter_model_release_id,
)
from mlb_pipeline.modeling.train_hitter_player_game_outcome_models import _select_tb_count_repair
from mlb_pipeline.modeling.prop_clean_slate import (
    CleanSlateThresholds,
    clean_slate_qualifies,
    decorate_clean_slate_row,
)


def test_json_safe_replaces_nested_nonfinite_values() -> None:
    value = ledger._json_safe({"a": np.nan, "b": [np.inf, pd.NA], "c": np.int64(4)})
    assert value == {"a": None, "b": [None, None], "c": 4}


def test_game_forecast_lock_builds_three_forecast_targets(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(ledger, "insert_forecasts", lambda _conn, rows: captured.extend(rows) or len(rows))
    inserted = ledger.lock_game_forecasts(
        object(),
        [{
            "game_date_et": date(2026, 7, 1),
            "game_slug": "20260701-LAD-ATH",
            "home_team_abbr": "ATH",
            "away_team_abbr": "LAD",
            "pred_run_diff": -0.7,
            "pred_total": 10.4,
            "pred_run_diff_model_only": -0.9,
            "pred_total_model_only": 10.1,
            "market_run_line": 1.5,
            "market_total": 11.0,
            "sigma_q_rl": 3.2,
            "sigma_q_total": 2.8,
        }],
        run_id="run-1",
        phase="pregame",
    )
    assert inserted == 3
    assert {row["stat"] for row in captured} == {"game_run_diff", "game_total", "game_home_win"}
    assert all(row["forecast_run_id"] == "run-1" for row in captured)


def test_player_forecast_lock_uses_pre_offer_raw_counts(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(ledger, "insert_forecasts", lambda _conn, rows: captured.extend(rows) or len(rows))
    inserted = ledger.lock_player_forecasts(
        object(),
        [{
            "game_date_et": date(2026, 7, 1),
            "game_slug": "20260701-SF-ARI",
            "player_id": 1,
            "player_name": "Pitcher",
            "pred_strikeouts": 4.5,
            "raw_pred_strikeouts": 4.8,
            "projected_bf": 23.0,
            "projected_pitch_count": 91.0,
            "opportunity_context": {"projected_ip": 5.7, "pitcher_last_ip": 6.0},
        }],
        [{
            "game_date_et": date(2026, 7, 1),
            "game_slug": "20260701-SF-ARI",
            "player_id": 2,
            "player_name": "Hitter",
            "raw_pred_hits": 1.2,
            "raw_pred_total_bases": 1.8,
            "raw_pred_home_runs": 0.18,
            "projected_pa": 4.1,
            "baseline_projected_pa": 4.1,
            "validated_projected_pa": 4.3,
            "sigma_map": {},
            "opportunity_context": {
                "low_pa_probability": 0.08,
                "tb_count_source": "component_rebuild_from_live_hits_hr",
            },
        }],
        run_id="run-1",
        phase="pregame",
    )
    assert inserted == 9
    strikeouts = next(row for row in captured if row["stat"] == "pitcher_strikeouts")
    assert strikeouts["projection_value"] == 4.8
    production_pa = next(row for row in captured if row["stat"] == "hitter_plate_appearances")
    assert production_pa["projection_value"] == 4.1
    challenger = next(row for row in captured if row["stat"] == "hitter_plate_appearances_challenger")
    assert challenger["projection_value"] == 4.3
    assert challenger["baseline_value"] == 4.1
    assert challenger["model_family"] == "hitter_pa_v3_challenger"
    tb = next(row for row in captured if row["stat"] == "batter_total_bases")
    assert tb["model_family"] == "hitter_tb_component_rebuild_from_rate_challengers"


def test_hitter_rate_shadow_forecasts_lock_as_separate_model_version(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(ledger, "insert_forecasts", lambda _conn, rows: captured.extend(rows) or len(rows))
    inserted = ledger.lock_hitter_rate_shadow_forecasts(
        object(),
        [{
            "game_date_et": date(2026, 7, 9),
            "game_slug": "20260709-NYY-BOS",
            "player_id": 99,
            "player_name": "Shadow Hitter",
            "shadow_raw_pred_hits": 1.15,
            "shadow_baseline_raw_pred_hits": 0.95,
            "shadow_hits_direct_prediction": 1.35,
            "shadow_hits_blend_alpha": 0.5,
            "shadow_raw_pred_home_runs": 0.18,
            "shadow_baseline_raw_pred_home_runs": 0.12,
            "shadow_hr_direct_prediction": 0.24,
            "shadow_hr_blend_alpha": 0.5,
            "sigma_map": {"batter_hits": 0.8, "batter_home_runs": 0.25},
            "opportunity_context": {"low_pa_probability": 0.04},
        }],
        run_id="run-shadow",
        phase="day_pregame",
    )
    assert inserted == 2
    assert {row["stat"] for row in captured} == {"batter_hits", "batter_home_runs"}
    assert {row["model_version"] for row in captured} == {DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID}
    assert all(row["model_family"] == "hitter_rate_shadow_direct_player_game_blend" for row in captured)
    assert all((row["opportunity_context"] or {}).get("shadow_forecast") for row in captured)
    assert len({row["forecast_key"] for row in captured}) == 2
    hits = next(row for row in captured if row["stat"] == "batter_hits")
    assert hits["projection_value"] == 1.15
    assert hits["baseline_value"] == 0.95
    assert hits["model_only_projection"] == 1.35


def test_projection_audit_shadow_detection_keeps_production_frame() -> None:
    frame = pd.DataFrame([
        {
            "stat": "batter_hits",
            "model_family": "hitter_count_regression",
            "model_version": DEFAULT_HITTER_MODEL_RELEASE_ID,
            "projection_value": 1.0,
        },
        {
            "stat": "batter_hits",
            "model_family": "hitter_rate_shadow_direct_player_game_blend",
            "model_version": DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID,
            "projection_value": 1.2,
        },
    ])
    production = _non_shadow_df(frame)
    assert len(production) == 1
    assert production.iloc[0]["model_version"] == DEFAULT_HITTER_MODEL_RELEASE_ID
    assert _is_shadow_model("hitter_rate_shadow_direct_player_game_blend", DEFAULT_HITTER_RATE_SHADOW_RELEASE_ID)


def test_release_ids_are_stable_and_environment_overridable(monkeypatch) -> None:
    monkeypatch.delenv("MLB_HITTER_MODEL_RELEASE_ID", raising=False)
    monkeypatch.delenv("MLB_CANONICAL_FORECAST_PHASE", raising=False)
    assert hitter_model_release_id({"trained_at_utc": "tomorrow"}) == DEFAULT_HITTER_MODEL_RELEASE_ID
    assert canonical_forecast_phase() == DEFAULT_CANONICAL_FORECAST_PHASE
    monkeypatch.setenv("MLB_HITTER_MODEL_RELEASE_ID", "hitter-test-r2")
    monkeypatch.setenv("MLB_CANONICAL_FORECAST_PHASE", "evening_pregame")
    assert hitter_model_release_id({"model_release_id": "artifact-r1"}) == "hitter-test-r2"
    assert canonical_forecast_phase() == "evening_pregame"


def test_projection_gate_requires_model_to_beat_simple_baseline() -> None:
    frame = pd.DataFrame({
        "forecast_type": ["player"] * 100,
        "stat": ["batter_hits"] * 100,
        "game_date_et": [date(2026, 6, 1 + i // 20) for i in range(100)],
        "game_slug": [f"g{i}" for i in range(100)],
        "player_id": list(range(100)),
        "projection_value": [1.0] * 100,
        "baseline_value": [1.5] * 100,
        "model_only_projection": [np.nan] * 100,
        "actual_value": [1.0] * 100,
    })
    summary = _stat_summary(frame, AuditConfig())
    assert summary["projection_eligible"]
    assert summary["mae_gain_vs_baseline"] == 0.5


def test_clean_slate_requires_operational_and_exact_valid_close_coverage() -> None:
    row = decorate_clean_slate_row({
        "side_lock_rows": 200,
        "captured_side_locks": 190,
        "line_available_side_locks": 150,
        "valid_side_locks": 150,
        "training_rows": 200,
        "missing_lock_examples": 0,
        "stale_close_examples": 0,
        "close_times": 3,
    }, CleanSlateThresholds())
    assert not clean_slate_qualifies(row, CleanSlateThresholds())
    assert row["operational_close_capture_coverage"] == 0.95
    assert row["valid_clv_coverage"] == 0.75
    assert "valid_close_coverage<0.90" in row["clean_slate_reasons"]


def test_count_error_decomposition_is_exact() -> None:
    frame = decompose_count_error(pd.DataFrame([{
        "projection": 2.0,
        "actual": 1.0,
        "opportunity_projection": 4.0,
        "opportunity_actual": 5.0,
    }]))
    row = frame.iloc[0]
    assert row["opportunity_error"] == -0.5
    assert row["rate_error"] == 1.5
    assert abs(row["decomposition_residual"]) < 1e-12


def test_forecast_error_slices_keep_model_versions_separate() -> None:
    frame = _add_error_slice_columns(pd.DataFrame([{
        "stat": "batter_total_bases",
        "model_version": "hitter-release-a",
        "lineup_confirmed_flag": 1.0,
        "opportunity_projection": 4.0,
        "opportunity_actual": 5.2,
    }]))
    row = frame.iloc[0]
    assert row["stat_model_version"] == "batter_total_bases / hitter-release-a"
    assert row["lineup_status"] == "confirmed_lineup"
    assert row["opportunity_error_bucket"] == "under_projected_by_1_plus"


def test_tb_diagnostic_does_not_turn_unknown_components_into_zero() -> None:
    frame = pd.DataFrame([{
        "stat": "batter_total_bases",
        "game_date_et": date(2026, 7, 1),
        "projection": 2.0,
        "actual": 4.0,
        "baseline": 1.5,
        "opportunity_projection": 4.0,
        "opportunity_actual": 4.0,
        "opportunity_error": 0.0,
        "rate_error": -2.0,
        "actual_singles": np.nan,
        "actual_doubles": np.nan,
        "actual_triples": np.nan,
        "actual_home_runs": np.nan,
    }])
    diagnostic = _tb_diagnostic(frame, min_rows=1)
    assert diagnostic["component_actual_per_pa"] == {}
    assert diagnostic["by_actual_tb_state"][0]["tb_state"] == "4+ unknown-HR"


def test_direct_tb_repair_requires_later_date_gain() -> None:
    rows = []
    for day in range(10):
        for player in range(30):
            actual = float((player + day) % 5)
            rows.append({
                "game_date_et": date(2026, 6, 1) + timedelta(days=day),
                "actual_total_bases": actual,
                "model_pred_total_bases": actual + 0.8,
            })
    holdout = pd.DataFrame(rows)
    policy = _select_tb_count_repair(
        holdout,
        holdout["actual_total_bases"].to_numpy(dtype=float),
    )
    assert policy["enabled"]
    assert policy["alpha"] > 0
    assert policy["validation_mae_gain"] > 0.3
