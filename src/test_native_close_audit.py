from datetime import timedelta

from native_close_audit import select_saved
from sharp_pipeline import report
from test_price_survival_review import fixture,NOW


def test_uses_saved_quote_before_native_start_never_later_provider_close(monkeypatch):
    a=fixture()[0]['alert'];a.update(home_team='Home',away_team='Away')
    body=dict(id='e',home_team='Home',away_team='Away')
    monkeypatch.setattr(report,'payload_kind',lambda p:'props')
    monkeypatch.setattr(report,'row_kind',lambda p:'props')
    seen=[]
    def evidence(o,closes):
        at=next(iter(closes.values()))[0];seen.append(at)
        return dict(method='exact',probability=.51,observed_at=at.isoformat(),source_updated_at=at.isoformat())
    monkeypatch.setattr(report,'close_evidence',evidence)
    snapshots=[dict(observed_at_utc=NOW+timedelta(minutes=m),payload=body) for m in (15,25)]
    out=select_saved(a,a['native_tip'],snapshots)
    assert seen==[NOW+timedelta(minutes=15)] and out['method']=='exact'
    assert select_saved(a,a['native_tip'],snapshots[1:])=={}


def test_conflicting_last_snapshot_or_missing_exact_line_stays_unknown(monkeypatch):
    a=fixture()[0]['alert'];a.update(home_team='Home',away_team='Away')
    body=dict(id='e',home_team='Home',away_team='Away')
    monkeypatch.setattr(report,'payload_kind',lambda p:'props')
    monkeypatch.setattr(report,'row_kind',lambda p:'props')
    monkeypatch.setattr(report,'close_evidence',lambda *args:dict(method='shifted',probability=.6))
    one=dict(observed_at_utc=NOW+timedelta(minutes=15),payload=body)
    assert select_saved(a,a['native_tip'],[one])=={}
    assert select_saved(a,a['native_tip'],[one,dict(one,payload=dict(body,different=True))])=={}
