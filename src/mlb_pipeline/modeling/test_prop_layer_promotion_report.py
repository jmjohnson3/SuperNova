from __future__ import annotations

import json
from datetime import datetime, timezone

from .prop_layer_promotion_report import (
    LayerPromotionConfig,
    build_control,
    layer_auto_integration_enabled,
    write_outputs,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def test_layer_control_blocks_shadow_hitter_rate_without_dates(tmp_path) -> None:
    _write_json(tmp_path / "prop_five_date_checkpoint.json", {
        "generated_at_utc": _now(),
        "hitter_rate_shadow_release": {
            "completed_date_count": 0,
            "pending_anchor_rows": 0,
            "metrics": {
                "batter_hits": {"rows": 0, "projection_pass": False},
                "batter_home_runs": {"rows": 0, "projection_pass": False},
            },
        },
    })
    payload = build_control(LayerPromotionConfig(model_dir=tmp_path))
    layer = payload["layers"]["hitter_hits_hr_shadow"]
    assert not layer["can_auto_integrate"]
    assert "completed_dates<5" in layer["blockers"]
    assert payload["auto_integrations"]["hitter_rate_challenger_production"]["enabled"] is False


def test_layer_control_enables_hitter_rate_after_clean_projection_pass(tmp_path) -> None:
    _write_json(tmp_path / "prop_five_date_checkpoint.json", {
        "generated_at_utc": _now(),
        "hitter_rate_shadow_release": {
            "completed_date_count": 5,
            "pending_anchor_rows": 0,
            "metrics": {
                "batter_hits": {"rows": 640, "dates": 5, "projection_pass": True},
                "batter_home_runs": {"rows": 640, "dates": 5, "projection_pass": True},
            },
        },
    })
    payload = write_outputs(build_control(LayerPromotionConfig(model_dir=tmp_path)), LayerPromotionConfig(model_dir=tmp_path))
    layer = payload["layers"]["hitter_hits_hr_shadow"]
    assert layer["can_auto_integrate"]
    assert payload["auto_integrations"]["hitter_rate_challenger_production"]["enabled"] is True
    assert layer_auto_integration_enabled(tmp_path, "hitter_rate_challenger_production")


def test_layer_control_requires_micro_and_kill_switch_for_auto_micro(tmp_path) -> None:
    _write_json(tmp_path / "prop_micro_promotion_evaluation.json", {
        "generated_at_utc": _now(),
        "micro_ready_count": 1,
        "tb_k_micro_ready_count": 1,
        "clv_classifier_ready": True,
        "buckets": [{"bucket": "pitcher_strikeouts|under|common|K 4.5-6.0|plus_100_149|draftkings", "micro_ready": True}],
    })
    _write_json(tmp_path / "prop_real_money_kill_switch.json", {
        "generated_at_utc": _now(),
        "status": "enabled",
        "active": False,
        "blockers": [],
    })
    payload = build_control(LayerPromotionConfig(model_dir=tmp_path))
    micro = payload["layers"]["exact_bucket_micro"]
    assert micro["can_auto_integrate"]
    assert payload["auto_integrations"]["exact_bucket_micro_ladder"]["enabled"] is True
