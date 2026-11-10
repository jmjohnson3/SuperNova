import json
from datetime import datetime, timedelta, timezone

import paper_collection_review as review


def test_missing_corrupt_stale_future_report_is_never_current(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'ROOT',tmp_path)
    path=tmp_path/'report.json';now=datetime(2026,11,6,tzinfo=timezone.utc)
    assert review.snapshot(path,now)['status']=='unavailable'
    path.write_text('[]')
    assert review.snapshot(path,now)['status']=='unavailable'
    for at in (now-timedelta(days=2),now+timedelta(minutes=1)):
        path.write_text(json.dumps(dict(built_at=at.isoformat())))
        assert review.snapshot(path,now)['status']=='stale'
    path.write_text(json.dumps(dict(built_at=now.isoformat())))
    assert review.snapshot(path,now)['status']=='current'


def test_counts_do_not_clear_failed_coverage_or_authorize_betting():
    m=dict(span_days=21,games=25,priced_checks=50,gates=dict(counts=True,availability=False))
    rule=dict(min_span_days=21,min_games=20,min_priced_checks=50)
    out=review.remaining(m,rule)
    assert (out['days'],out['games'],out['checked_prices_with_exact_closes'])==(0,0,0)
    assert out['other_failed_gates']==['availability']
    assert not out['counts_authorize_betting']
