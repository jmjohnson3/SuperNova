from datetime import datetime,timedelta,timezone

import pytest

from nba_props import fanduel_check as tool

NOW=datetime(2026,10,30,23,tzinfo=timezone.utc)


def alert(**changes):
    return dict(dict(alert_id=1,player='Test Player',market='points',side='over',line=22.5,price=110,
                     book='fanduel',delivery_status='sent',sent_at=NOW-timedelta(minutes=1),
                     observed_at=NOW-timedelta(minutes=2),commence_time_utc=NOW+timedelta(minutes=25)),**changes)


def collect(answers, row=None):
    it=iter(answers)
    return tool.collect_observation(row or alert(),ask=lambda _:next(it),clock=lambda:NOW,emit=lambda _:None)


def test_worse_price_is_preserved_and_never_defaulted_to_alert_price():
    r=collect(['a','22.5','-130','now'])
    assert r['price']==-130 and r['line']==22.5 and r['checked_at']==NOW
    with pytest.raises(ValueError):collect(['a','22.5','','now'])


@pytest.mark.parametrize('status',['d','s'])
def test_absent_offer_does_not_manufacture_line_or_odds(status):
    r=collect([status,'now'])
    assert r['line'] is None and r['price'] is None


def test_different_line_must_be_labeled_changed():
    assert collect(['c','23.5','-115','now'])['status']=='changed_line'
    with pytest.raises(ValueError,match='changed_line'):collect(['a','23.5','-115','now'])


@pytest.mark.parametrize('answers',[['q'],['a','22.5','-115','']])
def test_cancel_never_generates_a_check(answers):
    assert collect(answers) is None


def test_invalid_future_or_ambiguous_times_are_rejected():
    for stamp in ['2026-10-30T23:01:00Z','2026-10-30T23:00:00']:
        with pytest.raises(ValueError):collect(['d',stamp])
    with pytest.raises(ValueError,match='pregame'):collect(['d','now'],alert(commence_time_utc=NOW))


def test_late_check_keeps_real_time_for_existing_review_rules():
    r=collect(['d','now'],alert(sent_at=NOW-timedelta(minutes=6)))
    assert r['checked_at']==NOW


def test_pending_does_not_mix_books(monkeypatch):
    monkeypatch.setattr(tool.price_checks,'pending',lambda conn:[alert(),alert(book='draftkings')])
    assert [r['book'] for r in tool.pending(None)]==['fanduel']


def test_record_uses_shared_original_check_and_cancellation_does_not_write(monkeypatch):
    monkeypatch.setattr(tool,'get_alert',lambda conn,identity:alert())
    saved=[]
    def record(conn,identity,**data):
        saved.append((identity,data))
        return dict(alert_id=identity,**data)
    monkeypatch.setattr(tool.price_checks,'record',record)
    replies=iter(['a','22.5','-130','now'])
    result=tool.record_guided(None,1,ask=lambda _:next(replies),clock=lambda:NOW,emit=lambda _:None)
    assert result['saved'] and saved[0][0]==1 and saved[0][1]['price']==-130
    assert not tool.record_guided(None,1,ask=lambda _:'q',clock=lambda:NOW,emit=lambda _:None)['saved']
    assert len(saved)==1
