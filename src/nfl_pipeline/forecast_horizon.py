"""Stable pregame forecast-horizon labels.

Forecast quality depends materially on how much same-week information was
available.  These buckets keep early, two-hour, and near-kickoff forecasts
from being evaluated as one interchangeable population.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


TARGET_MINUTES = {
    "t20": 20.0,
    "t120": 120.0,
    "t24h": 24.0 * 60.0,
}


def _utc(value: Any) -> datetime | None:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp.to_pydatetime().astimezone(timezone.utc)


def horizon(start_ts_utc: Any, context_cutoff_utc: Any) -> tuple[float | None, str]:
    """Return minutes before kickoff and a stable evaluation bucket.

    The bands intentionally encompass normal scheduler jitter.  Forecasts
    after kickoff are invalid; forecasts more than 36 hours early remain a
    separate research cohort.
    """
    start = _utc(start_ts_utc)
    cutoff = _utc(context_cutoff_utc)
    if start is None or cutoff is None:
        return None, "unknown"
    minutes = (start - cutoff).total_seconds() / 60.0
    if minutes <= 0:
        return minutes, "post_kickoff"
    if minutes <= 45:
        return minutes, "t20"
    if minutes <= 240:
        return minutes, "t120"
    if minutes <= 36 * 60:
        return minutes, "t24h"
    return minutes, "early"


def add_horizon(row: dict[str, Any], *, cutoff: Any | None = None) -> dict[str, Any]:
    """Attach horizon fields without overwriting a previously locked value."""
    context_cutoff = cutoff or row.get("prediction_context_cutoff_utc")
    minutes, bucket = horizon(row.get("start_ts_utc"), context_cutoff)
    row.setdefault("forecast_horizon_minutes", minutes)
    row.setdefault("forecast_horizon_bucket", bucket)
    return row

