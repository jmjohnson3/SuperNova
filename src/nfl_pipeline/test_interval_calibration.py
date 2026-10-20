from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl_pipeline.modeling.fit_interval_calibration import fit_stat


def test_chronological_asymmetric_calibration_accepts_repeatable_tail_repair():
    rows = []
    actuals = [0.0] * 10 + [15.0] * 5 + [50.0] * 70 + [85.0] * 5 + [100.0] * 10
    for week in (1, 2, 3):
        lock = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=7 * week)
        for player, actual in enumerate(actuals):
            rows.append({
                "season": 2026,
                "week": week,
                "key": f"{week}|{player}|receiving_yards",
                "locked_at": lock,
                "label_available_at": lock + timedelta(days=1),
                "p10": 20.0,
                "p50": 50.0,
                "p90": 80.0,
                "actual": actual,
            })
    result = fit_stat(pd.DataFrame(rows))
    assert result["accepted"]
    assert result["lower_tail_scale"] > 1.0
    assert result["upper_tail_scale"] > 1.0
    assert result["forward_calibrated"]["coverage"] == 0.8
    assert result["relative_interval_score_gain"] > 0.02
