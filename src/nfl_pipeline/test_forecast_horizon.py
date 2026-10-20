from datetime import datetime, timedelta, timezone

import pytest

from nfl_pipeline.forecast_horizon import add_horizon, horizon


@pytest.mark.parametrize(
    "minutes,bucket",
    [(20, "t20"), (60, "t120"), (120, "t120"), (1440, "t24h"), (3000, "early")],
)
def test_horizon_buckets_scheduler_jitter(minutes, bucket):
    start = datetime(2026, 10, 11, 17, tzinfo=timezone.utc)
    actual, label = horizon(start, start - timedelta(minutes=minutes))
    assert actual == pytest.approx(minutes)
    assert label == bucket


def test_unknown_and_post_kickoff_horizons_fail_closed():
    start = datetime(2026, 10, 11, 17, tzinfo=timezone.utc)
    assert horizon(None, start) == (None, "unknown")
    assert horizon(start, start + timedelta(seconds=1))[1] == "post_kickoff"


def test_add_horizon_does_not_rewrite_a_locked_label():
    start = datetime(2026, 10, 11, 17, tzinfo=timezone.utc)
    row = {
        "start_ts_utc": start,
        "forecast_horizon_minutes": 21.0,
        "forecast_horizon_bucket": "t20",
    }
    add_horizon(row, cutoff=start - timedelta(hours=2))
    assert row["forecast_horizon_minutes"] == 21.0
    assert row["forecast_horizon_bucket"] == "t20"
