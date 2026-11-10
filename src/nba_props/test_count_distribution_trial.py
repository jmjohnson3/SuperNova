from datetime import datetime, timedelta, timezone
import os

import pandas as pd
import pytest

import model_research as parent
from nba_props import count_distribution_trial as trial
from sharp_pipeline.test_database_repairs import database
from test_model_research import offer, forecast, NOW

DOC=dict(sha256='b'*64,registered_at=(NOW-timedelta(minutes=1)).isoformat(),
         config=dict(parent_experiment=parent.NAME,parent_sha256='a'*64))


def saved(**changes):
    o=offer();f=forecast(tip_utc=NOW+timedelta(minutes=30),mu=10,disp_a=3)
    return dict(dict(id=1,experiment_id=parent.NAME,experiment_sha256='a'*64,captured_at=NOW,
                     payload=dict(offer=o,decision=parent.assess(o,f))),**changes)


def test_challenger_keeps_original_offer_and_can_disagree_without_changing_parent():
    source=saved()
    scored=trial.score(source,DOC,NOW)
    assert scored and scored['offer']==parent.clean(source['payload']['offer'])
    assert scored['frozen_probability']==source['payload']['decision']['model_probability']
    assert scored['model_probability'] != scored['frozen_probability']
    assert scored['betting_authorized'] is False


@pytest.mark.parametrize('change',['before_registration','stale','started','identity','unavailable','missing','unsupported'])
def test_old_or_ineligible_forecasts_never_become_prospective_captures(change):
    s=saved();doc=dict(DOC);at=NOW
    if change=='before_registration':doc['registered_at']=(NOW+timedelta(seconds=1)).isoformat()
    elif change=='stale':at=NOW+timedelta(seconds=301)
    elif change=='started':s['payload']['offer']['commence_time_utc']=NOW
    elif change=='identity':s['experiment_sha256']='wrong'
    elif change=='unavailable':s['payload']['decision']['forecast']['available']=False
    elif change=='missing':s['payload']['decision']['reason']='missing_forecast'
    else:s['payload']['offer']['market']='threes'
    assert trial.score(s,doc,at) is None


def test_identical_selections_do_not_establish_incremental_value():
    data=[]
    for i in range(100):
        data.append(dict(selected=True,frozen_selected=True,event_id=str(i),day=(NOW+timedelta(days=i%22)).date(),
            close_method='exact',timely_check=True,exact_offer_available=True,checked_price_ev=.04,close_ev=.04,
            opportunity_ev=.04,result='win',paper_profit=1.,model_probability=.8,
            fair_probability=.6,frozen_probability=.5))
    r=trial.summarize(pd.DataFrame(data),22)
    assert r['status']=='collecting' and r['betting_authorized'] is False
    assert r['incremental_opportunity']['game']['lower_95']==0


@pytest.mark.skipif(os.getenv('RUN_DATABASE_REPAIR_TESTS')!='1',reason='opt-in PostgreSQL')
def test_capture_is_append_only_and_idempotent(database,monkeypatch):
    class Fixed(datetime):
        @classmethod
        def now(cls,tz=None):return NOW
    monkeypatch.setattr(trial,'datetime',Fixed)
    with database.cursor() as cur:cur.execute(parent.SCHEMA);cur.execute(trial.SCHEMA)
    s=saved();p=s['payload']
    parent.save(database,dict(sha256='a'*64),p['offer'],p['decision'],NOW)
    assert trial.capture(database,DOC)==1
    assert trial.capture(database,DOC)==0
    with database.cursor() as cur:
        cur.execute('SAVEPOINT immutability')
        with pytest.raises(Exception,match='append-only'):
            cur.execute("UPDATE bets.nba_count_trial SET payload='{}'::jsonb")
        cur.execute('ROLLBACK TO SAVEPOINT immutability')
        cur.execute('SELECT payload FROM bets.nba_count_trial')
        assert cur.fetchone()[0]['betting_authorized'] is False
