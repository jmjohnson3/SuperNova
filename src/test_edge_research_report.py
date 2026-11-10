from datetime import datetime,timezone
from copy import deepcopy
import os

import pytest

from edge_research_report import typed_offer
from edge_research import reporting
from sharp_pipeline.test_database_repairs import database


def test_saved_offer_timestamps_are_parsed_without_mutating_locked_payload():
    original=dict(decided_at='2026-11-04T23:35:00+00:00',commence_time_utc='2026-11-05T00:00:00+00:00',line=-3.5)
    saved=deepcopy(original);typed=typed_offer(saved)
    assert typed['commence_time_utc']==datetime(2026,11,5,tzinfo=timezone.utc)
    assert typed['decided_at'].tzinfo is not None and saved==original
    with pytest.raises(ValueError):typed_offer(dict(original,commence_time_utc='bad timestamp'))


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')
def test_typed_saved_timestamp_supports_official_result_interval_query(database):
    o=typed_offer(dict(decided_at='2026-11-04T23:35:00+00:00',commence_time_utc='2026-11-05T00:00:00+00:00'))
    with database.cursor() as cur:
        cur.execute("SELECT %s-interval '15 minutes', %s+interval '15 minutes'",(o['commence_time_utc'],o['commence_time_utc']))
        low,high=cur.fetchone()
    assert (high-low).total_seconds()==1800
