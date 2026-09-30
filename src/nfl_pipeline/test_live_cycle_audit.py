from copy import deepcopy
from datetime import date,datetime,timezone

import pytest

from nfl_pipeline import live_cycle_audit as audit
from nfl_pipeline.test_prospective_market_trial import paired_record

DAY=date(2026,9,24)


def inputs():
    r=paired_record();r.update(status='scheduled',actual=None)
    p=r['forecast_payload']
    g=dict(game_id=r['game_id'],start_ts_utc=r['start_ts_utc'],status=r['status'])
    m=dict(kind='prop',forecast_id=r['id'],**{k:p[k] for k in ('book','side','line','price','model_version')})
    pub=dict(game_id=r['game_id'],sent_at='2026-09-22T10:10:00Z',forecast_manifest=[m])
    checkpoint=dict(capture=dict(eligible_forecast_ids=[1],captured_forecast_ids=[1]),
        close={'all_eligible':{'observations':[]}})
    return g,r,pub,checkpoint


def test_future_games_have_no_close_failure_or_synthetic_zero():
    g,r,p,c=inputs()
    doc=audit.build(DAY,[g],[r],c,[],datetime(2026,9,21,tzinfo=timezone.utc))
    assert doc['status']=='pending' and not doc['blockers']
    assert doc['games'][0]['valid_close_coverage'] is None


@pytest.mark.parametrize('field,value',[('line',31.5),('price',150),('book','draftkings')])
def test_discord_manifest_must_match_the_immutable_offer(field,value):
    g,r,p,c=inputs();p['forecast_manifest'][0][field]=value
    assert not audit.publication_matches(p,{1:r},audit.safe_time(g['start_ts_utc']))


def test_missing_manifest_and_stale_price_are_not_verified():
    g,r,p,c=inputs()
    assert audit.publication_matches(p,{1:r},audit.safe_time(g['start_ts_utc']))
    p['sent_at']='2026-09-22T11:00:00Z'
    assert not audit.publication_matches(p,{1:r},audit.safe_time(g['start_ts_utc']))


def test_projection_only_card_rows_do_not_invalidate_priced_manifest():
    g,r,p,c=inputs()
    projection=deepcopy(r);projection['id']=2
    for k in ('book','side','line','price'):
        projection['forecast_payload'][k]=None
    m=dict(kind='prop',forecast_id=2,**{k:projection['forecast_payload'][k] for k in ('book','side','line','price','model_version')})
    p['forecast_manifest'].append(m)
    assert audit.publication_matches(p,{1:r,2:projection},audit.safe_time(g['start_ts_utc']))
    p['forecast_manifest']=[m]
    assert not audit.publication_matches(p,{2:projection},audit.safe_time(g['start_ts_utc']))
    p['forecast_manifest']=[]
    assert not audit.publication_matches(p,{1:r},audit.safe_time(g['start_ts_utc']))


def test_settled_cycle_needs_publication_closes_and_result_but_does_not_approve_cash():
    g,r,p,c=inputs();g['status']='final';r.update(status='final',actual=10.)
    run=dict(status='ok',pregame=True,publications=[p])
    now=datetime(2026,9,23,tzinfo=timezone.utc)
    doc=audit.build(DAY,[g],[r],c,[run],now)
    assert doc['status']=='needs_attention' and doc['games'][0]['valid_close_coverage']==0
    c['close']['all_eligible']['observations']=[{'forecast_id':1,'valid_exact_capture':True}]
    doc=audit.build(DAY,[g],[r],c,[run],now)
    assert doc['status']=='complete' and not doc['betting_approved']
    r['actual']=None
    assert 'unresolved' in ' '.join(audit.build(DAY,[g],[r],c,[run],now)['blockers'])


def test_no_game_and_pre_contract_dates_do_not_fake_a_missing_publication():
    now=datetime(2026,9,23,tzinfo=timezone.utc)
    assert audit.build(DAY,[],[],{},[],now)['status']=='no_games'
    g,r,p,c=inputs()
    assert audit.build(date(2026,9,22),[g],[r],c,[],now)['status']=='pre_contract_history'


def test_later_run_failure_does_not_erase_verified_publication_or_close_blockers():
    g,r,p,c=inputs()
    run=dict(status='failed',pregame=True,publications=[p],
             steps=[dict(step='NFL Live Cycle Audit',returncode=1)])
    now=datetime(2026,9,23,tzinfo=timezone.utc)
    doc=audit.build(DAY,[g],[r],c,[run],now)
    assert doc['games'][0]['discord_exact_prop_manifest_verified']
    assert doc['games'][0]['blockers']==['valid_exact_close_coverage_below_90_percent']
    run['publications']=[]
    doc=audit.build(DAY,[g],[r],c,[run],now)
    assert not doc['games'][0]['discord_exact_prop_manifest_verified']
    assert 'game_aware_discord_manifest_missing_or_invalid' in doc['games'][0]['blockers']
