"""Shared prospective eligibility boundary for real-money prop evidence.

Rows before this date remain available for model training and audit reports, but
they cannot promote a bucket.  The boundary is intentionally a fixed source
constant so reruns cannot move it forward and cherry-pick a better sample.
"""
from __future__ import annotations

from datetime import date


PROP_REAL_MONEY_ELIGIBILITY_START_DATE = date(2026, 6, 19)


def parse_eligibility_start_date(value: str | date | None) -> date:
    if value is None:
        return PROP_REAL_MONEY_ELIGIBILITY_START_DATE
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())
