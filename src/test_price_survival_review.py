from copy import deepcopy
from datetime import datetime,timedelta,timezone

import pytest

from price_survival_review import evaluate,summarize,close_clock_check

NOW=datetime(2026,11,6,23,tzinfo=timezone.utc)


def fixture():
    a=dict(sport='nba',book='fanduel',event_id='e',market='points',player='Test Player',player_norm='test player',
        side='over',line=20.5,price=110,fair_probability=.55,
        sent_at=NOW-timedelta(seconds=30),decided_at=NOW-timedelta(seconds=35),
        native_tip=NOW+timedelta(minutes=20),commence_time_utc=NOW+timedelta(minutes=30))
    receipt=dict(requested_at=NOW-timedelta(seconds=1),received_at=NOW,server_date=NOW,cache_age_seconds=0)
    p=dict(status='available',reason='exact_main_quote',line=20.5,price=-130,observed_at=NOW,
           timely=True,receipts=[deepcopy(receipt),deepcopy(receipt)])
    r=dict(alert_key='#1',alert=a,payload=p,attempted_at=NOW-timedelta(seconds=2),recorded_at=NOW)
    model=dict(captured_at=NOW-timedelta(seconds=5),payload=dict(offer=deepcopy(a),decision=dict(selected=False,model_probability=.51)))
    return r,model,dict(method='exact',probability=.55,observed_at=NOW+timedelta(minutes=15),source_updated_at=NOW+timedelta(minutes=14))


def evaluate_fixture(**changes):
    r,m,c=fixture()
    return evaluate(r,m,c,changes.get('result','loss'),changes.get('now',NOW+timedelta(days=1)))


def test_worse_odds_reprice_both_strategies_at_same_observation():
    out=evaluate_fixture()
    assert out['available'] and out['price_change']=='worse'
    checked=.55*(100/130)-.45
    assert out['checked_price_close_ev']==pytest.approx(checked)
    assert out['original_price_close_ev']==pytest.approx(.55*1.1-.45)
    assert out['incremental_ev']==pytest.approx(-checked)
    assert out['paper_profit_at_observed_price']==-1 and out['incremental_paper_profit']==1
    assert out['queue_seconds']==28 and out['check_delay_seconds']==30
    assert out['paired_model'] and out['model_selected'] is False
    assert not out['account_availability_verified'] and not out['betting_authorized']


@pytest.mark.parametrize('status',['changed_line','not_listed','suspended'])
def test_verified_unavailable_can_be_zero_opportunity_but_not_a_loss(status):
    r,m,c=fixture();r['payload'].update(status=status,line=None,price=None)
    out=evaluate(r,m,c,'win',NOW+timedelta(days=1))
    assert out['unavailable'] and out['opportunity_ev']==0
    assert out['checked_price_close_ev'] is None and out['paper_profit_at_observed_price'] is None


@pytest.mark.parametrize('change',['unknown','stale','late','no_receipts','incomplete','no_sent_at','future_record','wrong_line'])
def test_unknown_or_invalid_checks_never_create_execution_value(change):
    r,m,c=fixture()
    if change=='unknown':r['payload']['status']='unknown'
    if change=='stale':r['payload']['receipts'][1]['cache_age_seconds']=60
    if change=='late':r['alert']['sent_at']=NOW-timedelta(minutes=6)
    if change=='no_receipts':r['payload']['receipts']=[]
    if change=='incomplete':r['payload']=None
    if change=='no_sent_at':r['alert']['sent_at']=None
    if change=='future_record':r['recorded_at']=NOW-timedelta(seconds=1)
    if change=='wrong_line':r['payload']['line']=21.5
    out=evaluate(r,m,c,'loss',NOW+timedelta(days=1))
    assert not out['available']
    assert out['checked_price_close_ev'] is None and out['opportunity_ev'] is None


def test_model_after_observation_or_different_offer_is_not_paired():
    r,m,c=fixture();m['captured_at']=NOW+timedelta(seconds=1)
    out=evaluate(r,m,c,'loss',NOW+timedelta(days=1))
    assert out['available'] and not out['paired_model'] and out['incremental_ev'] is None
    r,m,c=fixture();m['payload']['offer']['side']='under'
    assert not evaluate(r,m,c,'loss',NOW+timedelta(days=1))['paired_model']


@pytest.mark.parametrize('change',['shifted','missing_close','integer_line'])
def test_estimates_and_unknown_push_mass_do_not_inflate_checked_closes(change):
    r,m,c=fixture()
    if change=='shifted':c['method']='shifted'
    if change=='missing_close':c={}
    if change=='integer_line':r['alert']['line']=r['payload']['line']=20
    out=evaluate(r,m,c,'win',NOW+timedelta(days=1))
    assert out['available'] and out['checked_price_close_ev'] is None


def test_pending_results_stay_pending_and_small_positive_sample_cannot_promote():
    r,m,c=fixture();out=evaluate(r,m,c,None,NOW+timedelta(days=1))
    assert out['paper_profit_at_observed_price'] is None and out['brier_improvement'] is None
    summary=summarize([out],'nba')
    assert summary['model_value_status']=='insufficient_public_evidence'
    assert not summary['automatic_filter_promotion'] and not summary['betting_authorized']
    assert summary['model_incremental_ev_by_game']['lower_95'] is None


def test_correlated_rows_use_one_game_cluster_not_independent_bets():
    data=[dict(evaluate_fixture(),alert_key=f'#{n}') for n in range(30)]
    summary=summarize(data,'nba')
    assert summary['model_incremental_ev_by_game']['observations']==30
    assert summary['model_incremental_ev_by_game']['clusters']==1
    assert summary['model_incremental_ev_by_game']['lower_95'] is None
    assert summary['price_changes']['worse']==30


def test_provider_close_after_earlier_native_schedule_cannot_create_evidence():
    r,m,c=fixture();c['observed_at']=NOW+timedelta(minutes=25)
    out=evaluate(r,m,c,'win',NOW+timedelta(days=1))
    assert out['reported_exact_close'] and not out['exact_close']
    assert out['close_clock_reason']=='close_not_before_earliest_schedule'
    assert out['checked_price_close_ev'] is None
    assert summarize([out],'nba')['excluded_reported_exact_closes']==1


def test_missing_close_time_does_not_pass_nba_or_mismatched_nfl_clock():
    r,m,c=fixture();c.pop('observed_at');c.pop('source_updated_at')
    assert not close_clock_check(r['alert'],c)[0]
    r['alert']['sport']='nfl'
    assert not close_clock_check(r['alert'],c)[0]
    r['alert']['native_tip']=r['alert']['commence_time_utc']
    assert close_clock_check(r['alert'],c)==(True,'registered_nfl_reader_enforces_identical_cutoff')
