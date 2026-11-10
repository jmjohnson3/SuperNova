"""One real observation feeds all three paper comparisons; isolated DB writes."""
from datetime import timedelta
import os

import pytest

import model_research as model
import model_research_checks as model_checks
import model_research_report as model_report
from nba_props import count_distribution_trial as count, fanduel_check
from sharp_pipeline import paper_alerts, scan
from sharp_pipeline.test_database_repairs import database
from sharp_pipeline.test_price_checks_database import DOC, NOW, START, init, snapshot
from test_model_research import forecast, offer

pytestmark=pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='isolated PostgreSQL opt-in')


def test_guided_first_check_is_shared_and_comparison_does_not_reset_deadline(database,monkeypatch,tmp_path):
    init(database)
    scan.store(database,'nba',snapshot(),NOW,experiment=DOC)
    paper_alerts.dispatch(database,DOC,now=NOW,poster=lambda text:'test-receipt')
    with database.cursor() as cur:
        cur.execute("SELECT alert_id FROM bets.nba_paper_price_alerts WHERE book='fanduel'")
        check_id=cur.fetchone()[0]
    alert=fanduel_check.get_alert(database,check_id)
    with database.cursor() as cur:
        cur.execute(model.SCHEMA)
        cur.execute(count.SCHEMA)
    o=offer(parent_check_id=check_id,decided_at=NOW,commence_time_utc=START,
            line=float(alert['line']),price=int(alert['price']))
    f=forecast(run_at_utc=NOW-timedelta(minutes=5),tip_utc=START)
    doc=dict(sha256='a'*64,registered_at=(NOW-timedelta(days=1)).isoformat())
    decision=model.assess(o,f)
    ident=model.save(database,doc,o,decision,NOW)
    model_checks.dispatch(database,doc,'nba',now=NOW+timedelta(minutes=4),poster=lambda text:'comparison-receipt')
    with pytest.raises(ValueError,match='original NBA'):
        model_checks.record(database,ident,checked_at=NOW+timedelta(minutes=4),now=NOW+timedelta(minutes=4),
                            status='available',line=o['line'],price=-130)
    replies=iter(['a',str(o['line']),'-130',(NOW+timedelta(minutes=1)).isoformat()])
    saved=fanduel_check.record_guided(database,check_id,ask=lambda _:next(replies),
        clock=lambda:NOW+timedelta(minutes=1,seconds=10),emit=lambda _:None)
    assert saved['saved'] and saved['check']['displayed_price']==-130
    with pytest.raises(ValueError,match='first check'):
        fanduel_check.record_guided(database,check_id,ask=lambda _:'q')

    count_doc=dict(sha256='b'*64,registered_at=(NOW-timedelta(minutes=1)).isoformat(),
                   config=dict(parent_experiment=model.NAME,parent_sha256=doc['sha256']))
    from datetime import datetime
    class Fixed(datetime):
        @classmethod
        def now(cls,tz=None):return NOW+timedelta(minutes=2)
    monkeypatch.setattr(count,'datetime',Fixed)
    assert count.capture(database,count_doc)==1
    class AfterGame(datetime):
        @classmethod
        def now(cls,tz=None):return START+timedelta(hours=3)
    monkeypatch.setattr(count,'datetime',AfterGame)
    monkeypatch.setattr(model_report,'datetime',AfterGame)
    (tmp_path/'reports').mkdir()
    for mod in (model_report,count):
        monkeypatch.setattr(mod,'ROOT',tmp_path)
        monkeypatch.setattr(mod,'exact_close',lambda conn,offer:dict(method='exact',probability=.5,ev=.05))
        monkeypatch.setattr(mod,'verified_result',lambda *args:'loss')
    a=model_report.build_report(database,doc)['groups']['nba_props']
    b=count.build_report(database,count_doc)
    assert a['price_control']['checked_prices']==a['model_assisted']['checked_prices']==1
    assert a['price_control']['mean_checked_price_ev']<0
    assert a['diagnostics']['median_delivery_to_check_seconds']==60  # original receipt, not comparison
    assert b['paired_execution_coverage']==1
    assert not a['model_adds_value_review'] and not b['betting_authorized']


def test_guided_record_rejects_another_sportsbook(database):
    init(database)
    scan.store(database,'nba',snapshot(),NOW,experiment=DOC)
    paper_alerts.dispatch(database,DOC,now=NOW,poster=lambda text:'test-receipt')
    with database.cursor() as cur:
        cur.execute("SELECT alert_id FROM bets.nba_paper_price_alerts WHERE book='draftkings'")
        identity=cur.fetchone()[0]
    with pytest.raises(ValueError,match='FanDuel'):
        fanduel_check.get_alert(database,identity)
