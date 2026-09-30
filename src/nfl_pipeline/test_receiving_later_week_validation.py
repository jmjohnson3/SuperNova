from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from nfl_pipeline.modeling.receiving_later_week_validation import earlier_blocks, chronological, curve_calibration, coherence_errors
from nfl_pipeline.modeling.receiving_selection_experiment import ranked
from nfl_pipeline.modeling.receiving_result_reconciliation import classify
from nfl_pipeline.cash_readiness import selection_cohort_accounting


def history():
    rows=[]
    for week in range(1,6):
        lock=datetime(2026,9,1,tzinfo=timezone.utc)+timedelta(weeks=week-1)
        for i in range(40):
            rows.append(dict(prediction_id=week*100+i, game_id=f'{week}-{i//4}', player_id=str(i),
                player_game=f'{week}-{i}', model_version='r',scoring_version='s',season=2026,week=week,
                day=lock.date().isoformat(),batch=lock.isoformat(),locked_at=lock,
                label_available_at=lock+timedelta(days=1), side='over',line=40.5,push_probability=0.,
                outcome=float(i%3>0),actual=60. if i%3>0 else 10.,probability=.8,market=.5,payout=1.,
                eligible=True,exact_micro=i<3,is_push=False,projection=50.,p10=10.,p90=100.,
                raw=.85,context=.82,heuristic=.81,uncertainty_proxy=.9))
    return pd.DataFrame(rows)


def test_expanding_calibration_purges_future_labels_and_overlapping_games():
    frame=history(); test=frame.loc[frame.week.eq(5)]
    frame.loc[0,'label_available_at']=test.locked_at.min()+timedelta(days=1)
    frame.loc[1,'game_id']=test.iloc[0].game_id
    blocks=earlier_blocks(frame,test)
    assert blocks is not None
    ids={i for b in blocks for i in b.prediction_id}
    assert 100 not in ids and 101 not in ids
    assert not ids.intersection(test.prediction_id)
    assert [set(b.week) for b in blocks]==[{1,2},{3},{4}]


def test_later_week_labels_cannot_change_same_week_calibration_or_selection():
    frame=history()
    a=chronological(frame)['folds'][-1]
    changed=frame.copy(); changed.loc[changed.week.eq(5),'outcome']=1-changed.loc[changed.week.eq(5),'outcome']
    changed.loc[changed.week.eq(5),'actual']=500.
    b=chronological(changed)['folds'][-1]
    assert a['calibration']==b['calibration']
    assert a['calibrated_reselected']['ids']==b['calibrated_reselected']['ids']
    assert not a['calibration_coverage_verified']


def test_insufficient_weeks_identity_and_pending_consumes_cap():
    frame=history().loc[lambda x:x.week.eq(1)].copy()
    frame.loc[frame.index[:5],['outcome','actual']]=np.nan
    result=chronological(frame)['folds'][0]
    assert result['calibration']['enabled'] is False
    assert result['current_selections']['pending']==5
    assert result['calibrated_reselected']['pending']==5


def test_uncertain_disagreement_discount_is_outcome_blind():
    frame=history().iloc[:2].copy()
    frame['uncertainty_proxy']=[1.,0.]
    assert ranked(frame,cap=1).prediction_id.tolist()==[100]
    assert ranked(frame,cap=1,penalty=.5,uncertainty_aware=True).prediction_id.tolist()==[101]
    frame['actual']=999.;frame['outcome']=1.
    assert ranked(frame,cap=1,penalty=.5,uncertainty_aware=True).prediction_id.tolist()==[101]


def test_missing_participation_is_not_zero_or_void():
    row=dict(status='final', actual=None,result_game_id='g',actual_offense_snaps=None,actual_targets=0)
    assert classify(row,{})=='missing_participation_evidence'
    assert classify(dict(row,result_game_id=None),{})=='missing_exact_player_game_result'
    assert classify(dict(row,actual_offense_snaps=0),{})=='zero_offensive_snaps_requires_book_settlement_review'
    assert classify(dict(row,actual_offense_snaps=10),{})=='missing_stat_with_verified_participation'
    assert classify(dict(row,actual=0,actual_offense_snaps=10),{})=='settled_verified_participation'


def test_post_registration_selection_enters_new_cohort_without_rewriting_old_slots(monkeypatch):
    from nfl_pipeline import cash_readiness as module
    monkeypatch.setattr(module,'source_hash',lambda _: 's')
    at=datetime(2026,9,27,tzinfo=timezone.utc)
    cfg={'config':{'registered_at':at.isoformat(),'scoring_version':'s','release':'r'}}
    old=dict(id=1,created_at_utc=at-timedelta(days=1),status='final',actual=10,
             forecast_payload={'model_version':'r','scoring_replay':{'scoring_fingerprint':'s'}})
    future=dict(old,id=2,created_at_utc=at+timedelta(days=1),status='scheduled',actual=None)
    selections=[{'forecast_id':1},{'forecast_id':2}]; saved=deepcopy(selections)
    result=selection_cohort_accounting([old,future],selections,cfg)
    assert result['counts']=={'pre_policy_selection_preserved':1,'post_policy_pending_game':1}
    future.update(status='final',actual=0)
    result=selection_cohort_accounting([old,future],selections,cfg)
    assert result['counts']['post_policy_settled_for_evidence_review']==1
    assert selections==saved and not result['historical_slots_replaced']


def test_fixed_research_capture_enters_cash_evidence_after_registration():
    from nfl_pipeline.test_receiving_trial_checkpoint import inputs, NOW
    from nfl_pipeline.cash_readiness import evidence
    records, documents, _, research, selections, closes = inputs()
    records[0]['forecast_payload']['scoring_replay']['scoring_fingerprint']='s'
    registration={'config':{'research_registration':research,'release':'production','scoring_version':'s',
                           'registered_at':'2026-09-22T10:00:45Z'}}
    selected, all_rows, _, unresolved, pending=evidence(records,documents,registration,selections,closes,NOW+timedelta(days=1))
    assert len(selected)==len(all_rows)==1 and unresolved==pending==0
    assert selected[0]['forecast_id']==1 and selected[0]['clv'] is None
    records[0]['status']='scheduled';records[0]['actual']=None
    selected, _, _, unresolved, pending=evidence(records,documents,registration,selections,closes,NOW)
    assert selected==[] and pending==1 and unresolved==0
    records[0]['forecast_payload']['scoring_replay']['scoring_fingerprint']='wrong'
    assert evidence(records,documents,registration,selections,closes,NOW)[3]==1


def test_prior_date_caps_do_not_prevent_next_date_fixed_selection():
    from nfl_pipeline.test_prospective_market_trial import paired_record, config, shadow, NOW
    from nfl_pipeline.modeling.receiving_research_trial import choose
    previous=[dict(forecast_id=100+i,day='2026-09-21',game_id='old',player_id=str(i)) for i in range(3)]
    selected, _=choose([paired_record()],[shadow()],config(),NOW,previous)
    assert len(selected)==1
    assert choose([paired_record()],[shadow()],config(),NOW,previous+selected)[0]==[]


def test_final_curve_calibration_reprices_lines_and_preserves_tails():
    from nfl_pipeline.modeling.tail_calibration import TailPreservingCalibration
    c=TailPreservingCalibration();c.enabled=True;c.strength=.5
    c.knots=np.array([.1,.5,.9]);c.values=np.array([.1,.3,.9])
    frame=history().iloc[:3].copy();frame['player_id']='same'
    frame['line']=[20.5,40.5,70.5]
    v=np.arange(101.);w=np.ones(101)/101
    frame['probability']=[w[v>x].sum() for x in frame.line]
    frame['calibrated']=frame.probability
    curves={int(i):(v,w) for i in frame.prediction_id}
    out,n=curve_calibration(frame,c,curves)
    assert n==3 and not np.allclose(out.probability,out.calibrated)
    assert np.allclose(out.p10,out.calibrated_p10)
    assert np.allclose(out.p90,out.calibrated_p90)
    assert coherence_errors(out,'calibrated')['line_reversals']==0
    # Missing/reconstructed supports cannot masquerade as verified coverage.
    _,n=curve_calibration(frame,c,{})
    assert n==0
