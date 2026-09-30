from datetime import date, datetime, timedelta, timezone
import json
import asyncio
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from nfl_pipeline import integrity
from nfl_pipeline.context_contract import context_evidence, validate_evidence
from nfl_pipeline.modeling.challenger_models import (
    ConditionalResidual, FittedHead, HierarchicalCalibration, OpportunityRateModel,
    curve_over, curve_summary, player_features,
)
from nfl_pipeline.modeling.evaluation import expanding_week_folds, inner_partitions, projection_metrics, clustered_gain
from nfl_pipeline.modeling.live_scoring_replay import build_report, prospective_report, validate_lock
from nfl_pipeline.modeling.predict_player_props import _candidate_from_offer
from nfl_pipeline.modeling.scoring_capture import capture, replay
from nfl_pipeline.modeling.train_accuracy_challengers import ReferenceRecipe, SmallGameModel, line_frame


def test_missing_interval_is_unknown_not_failed_coverage():
    result=projection_metrics([10,20],[11,19],[np.nan,15],[np.nan,25])
    assert result['rows']==2
    assert result['interval_rows']==1
    assert result['coverage_80']==1
    unknown=projection_metrics([10],[11],[np.nan],[np.nan])
    assert unknown['coverage_80'] is None
    assert projection_metrics([np.nan],[10])=={'rows':0}


def test_unknown_injury_and_route_proxies_remain_unknown():
    now=datetime(2026,9,18,tzinfo=timezone.utc)
    evidence=context_evidence({'route_participation_proxy_avg_5':.8,'depth_pos_rank':1},now)
    assert evidence['injury_status'] is None and evidence['injury_missing']
    assert evidence['expected_starter_from_depth'] is None
    assert evidence['routes_source']=='proxy_only' and evidence['routes_run_history'] is None
    assert evidence['teammate_injury_count'] is None
    X=player_features(pd.DataFrame([{'position':'WR','context_evidence':evidence}]),'receiving_yards')
    assert np.isnan(X.asof_injury_status_out.iloc[0])
    assert X.asof_injury_status_out__missing.iloc[0]==1


@pytest.mark.parametrize('timestamp',['not-a-time','2026-09-19T00:00:00Z'])
def test_invalid_or_future_observations_not_used(timestamp):
    lock=datetime(2026,9,18,tzinfo=timezone.utc)
    evidence=context_evidence({'injury_report_status':'Out','injury_observed_at':timestamp},lock)
    assert evidence['injury_status'] is None
    evidence['injury_observed_at']=timestamp
    assert validate_evidence(evidence,lock) is None


def test_challenger_features_exclude_outcomes_and_after_lock_market():
    frame=pd.DataFrame([{'position':'WR','receiving_yards':300,'targets':20,
        'targets_avg_5':6,'receiving_yards_avg_5':50,'closing_price':-300,'market_total':70,
        'injury_score':0,'routes_run_avg_5':np.nan}])
    X=player_features(frame,'receiving_yards')
    assert not {'receiving_yards','targets','closing_price','market_total','injury_score'} & set(X.columns)
    assert X.targets_avg_5.iloc[0]==6
    assert X.routes_run_avg_5__missing.iloc[0]==1


def history():
    return pd.DataFrame([{'season':season,'week':week,'game_id':f'{season}-{week}',
        'game_date_et':date(season,1,1)+timedelta(weeks=week),'player_id':str(p)}
        for season in [2023,2024,2025] for week in range(1,19) for p in range(20)])


def test_nested_walk_forward_blocks_have_disjoint_outcomes():
    folds=list(expanding_week_folds(history(),[2024,2025]))
    assert len(folds)==10
    for _,train,test in folds:
        assert train.game_date_et.max()<test.game_date_et.min()
        assert set(train.game_id).isdisjoint(test.game_id)
        partitions=inner_partitions(train)
        for left,right in zip(partitions,partitions[1:]):
            assert left.game_date_et.max()<right.game_date_et.min()
            assert set(left.game_id).isdisjoint(right.game_id)


def test_small_classifiers_return_empirical_mass_not_first_class():
    X=pd.DataFrame({'feature':[1,2,3,4]})
    head=FittedHead(classifier=True).fit(X,[0,1,1,2])
    np.testing.assert_allclose(head.probabilities(X,3),np.tile([.25,.5,.25],(4,1)))


def test_imputation_learns_training_values_only():
    X=pd.DataFrame({'feature':[2.,4.,np.nan],'absent':[np.nan]*3})
    head=FittedHead().fit(X,[1,2,3])
    assert head.columns==['feature']
    assert head.transform(pd.DataFrame({'feature':[np.nan,10000]})).iloc[0,0]==3


def test_no_observed_opportunity_cannot_train_rate():
    with pytest.raises(ValueError,match='positive observed exposure'):
        OpportunityRateModel().fit(pd.DataFrame({'position':['WR'],'targets':[0],'receiving_yards':[0]}),'receiving_yards')


def test_conditional_uncertainty_varies_with_role():
    X=pd.DataFrame({'role_volatility':[0.]*200+[1.]*200})
    errors=np.array([5.]*200+[50.]*200)
    u=ConditionalResidual().fit_scale(X,errors,np.full(400,30.),4.)
    assert u.scale(X)[-1]>u.scale(X)[0]*2
    assert u.confidence(X)[0]>u.confidence(X)[-1]
    u.fit_residuals(X,errors*np.tile([-1,1],200))
    values,weights=u.mixture(X,np.full(400,30.),np.ones((400,1)))
    summary=curve_summary(values,weights)
    assert np.all(summary['p10']<=summary['median'])
    assert np.all(summary['median']<=summary['p90'])
    assert np.all(curve_over(values,weights,np.full(400,20))>=curve_over(values,weights,np.full(400,40)))


def test_weighted_curve_distinguishes_mean_and_median():
    summary=curve_summary(np.array([[0.,10.,100.]]),np.array([[.2,.6,.2]]))
    assert summary['mean'][0]==26
    assert summary['median'][0]==10


def calibration_data(n=80):
    return pd.DataFrame({'probability':[.9]*n,'outcome':[0,1]*(n//2),'weight':[1.]*n,
        'player_game':[str(i) for i in range(n)],'position':'WR','workload_bucket':'normal',
        'line_bucket':'2','book':'unpriced_proxy','line':45.5})


def test_calibrator_needs_later_unique_outcomes_and_disables_harm():
    c=HierarchicalCalibration().fit(calibration_data())
    c.validate(calibration_data().assign(outcome=1))
    assert not c.enabled
    np.testing.assert_allclose(c.predict(calibration_data()),.9)
    c.validate(calibration_data().assign(player_game='one-game'))
    assert not c.enabled
    c.validate(calibration_data())
    assert c.enabled


def test_line_calibration_preserves_survival_curve_order():
    c=HierarchicalCalibration().fit(calibration_data())
    frame=calibration_data(4).assign(player_game='same',line=[10,20,30,40],probability=[.6,.7,.4,.5])
    p=c.candidate(frame)
    assert np.all(np.diff(p)<=0)


def test_unknown_book_falls_back_to_role_and_line_not_only_global():
    c=HierarchicalCalibration().fit(calibration_data())
    # A parent correction still applies to a book with no historical samples.
    c.offsets_by_level={1:{'WR':.5},2:{},3:{},4:{'WR|normal|2|unpriced_proxy':1.}}
    unknown=calibration_data().assign(book='new_book')
    from scipy.special import expit,logit
    expected=expit(c.intercept+c.slope*logit(.9)+.5)
    np.testing.assert_allclose(c.candidate(unknown),expected)


def test_proxy_lines_do_not_multiply_training_evidence():
    frame=pd.DataFrame({'game_id':['g'],'player_id':['p'],'season':[2025],'week':[1],
        'position':['WR'],'receiving_yards_avg_5':[0],'targets_avg_5':[2],'receiving_yards':[10]})
    lines=line_frame(frame,'receiving_yards',np.array([[0.,10.]]),np.array([[.5,.5]]))
    assert len(lines)==1
    assert lines.weight.sum()==1
    assert lines.book.iloc[0]=='unpriced_proxy'


def scoring_fixture():
    row={'game_id':'g','player_id':'p','player_name':'Player','position':'WR','season':2026,'week':2,
        'game_date_et':date(2099,9,20),'start_ts_utc':datetime(2099,9,20,17,tzinfo=timezone.utc)}
    offer={'offer_id':1,'line':45.5,'bookmaker_key':'draftkings','over_price':-110,'under_price':-110,
        'over_link':'https://example.com/over','under_link':'https://example.com/under'}
    metrics={'live_model_version':'frozen','accepted_live':True,'projection_accepted':True}
    distribution={'kind':'empirical_oof_residual','residual_quantiles':[-30,-10,0,10,30]}
    return row,offer,metrics,distribution


def test_full_live_scoring_roundtrips_exactly():
    row,offer,metrics,distribution=scoring_fixture()
    captured=capture(row,'receiving_yards',55,50,metrics,offer,distribution,{}, {},.02,{'version':'frozen'})
    live=_candidate_from_offer(row,'receiving_yards',55,50,metrics,offer,distribution)
    # The JSON round trip is the DB payload boundary.
    restored=replay(json.loads(json.dumps(captured,default=str)))
    assert restored['side']==live['side']
    assert restored['probability']==live['probability']
    assert restored['probability_trace']==live['probability_trace']
    assert set(live['probability_trace'])=={'raw_over','heuristic_over','post_exact_side','final_side','side',
        'probability_basis','push_probability','context_blend_over','market_anchor_over','context_trust'}
    assert live['probability_trace']['probability_basis']=='win_given_no_push'
    captured['scoring_fingerprint']='old'
    with pytest.raises(ValueError,match='code_version'):
        replay(captured)


def test_uncaptured_overlay_never_claims_exact_replay():
    row,offer,metrics,distribution=scoring_fixture()
    captured=capture(row,'receiving_yards',55,50,metrics,offer,distribution,{}, {},.02,
                     {'exact_line_artifact':{'models':['unavailable']}})
    with pytest.raises(ValueError,match='not_captured'):
        replay(captured)


def locked_record():
    row,offer,metrics,distribution=scoring_fixture()
    p=_candidate_from_offer(row,'receiving_yards',55,50,metrics,offer,distribution)
    p['prediction_context_cutoff_utc']='2099-09-20T12:00:00+00:00'
    return {'id':1,'game_id':'g','player_id':'p','stat':'receiving_yards','season':2099,'week':2,
        'side':p['side'],'line':45.5,'book':'draftkings','actual':60.,'forecast_payload':p,
        'created_at_utc':'2099-09-20T12:01:00Z','start_ts_utc':'2099-09-20T17:00:00Z',
        'offer_fetched_at':'2099-09-20T11:59:00Z'}


def test_locked_replay_deduplicates_and_does_not_invent_old_inputs():
    r=locked_record()
    report=build_report([r,{**r,'id':2}])
    assert report['settled_offer_decisions']==1
    assert report['replay_status']=={'missing_lock_time_scoring_inputs':1}
    assert report['exclusions']['later_revision_same_decision']==1
    r['offer_fetched_at']='2099-09-20T12:00:01Z'
    assert validate_lock(r)=='invalid_lock_offer_timing'


def test_pushes_are_not_losses():
    r=locked_record(); r['actual']=r['line']
    report=build_report([r])
    assert report['settled_offer_decisions']==0
    assert report['exclusions']['push_not_binary_target']==1


def test_prospective_shadow_cannot_be_backdated(tmp_path):
    r=locked_record(); p=r['forecast_payload']
    shadow={'forecast_id':1,'challenger_run':'run','production_release':p['model_version'],
        'source_context_cutoff':p['prediction_context_cutoff_utc'],'projection_mean':58.,'p10':10.,'p90':95.}
    path=tmp_path/'run'/'prospective'/'2099-09-20'/'one.json'
    integrity.atomic_json(path,{'scored_at':'2099-09-20T18:00:00Z','training_end':'2099-09-19','rows':[shadow]})
    assert prospective_report([r],tmp_path)['settled_player_games']==0
    integrity.atomic_json(path,{'scored_at':'2099-09-20T13:00:00Z','training_end':'2099-09-19','rows':[shadow]})
    result=prospective_report([r],tmp_path)
    assert result['settled_player_games']==1
    assert not result['automatic_promotion']
    r['actual']=None
    other={**shadow,'challenger_run':'run2'}
    integrity.atomic_json(tmp_path/'run2'/'prospective'/'2099-09-20'/'one.json',
        {'scored_at':'2099-09-20T13:00:00Z','training_end':'2099-09-19','rows':[other]})
    result=prospective_report([r],tmp_path)
    assert result['pending']==2 and result['unique_pending_player_games']==1


def test_frozen_production_cannot_be_published(tmp_path,monkeypatch):
    monkeypatch.setattr(integrity,'MODEL_ROOT',tmp_path)
    integrity.atomic_json(tmp_path/'active_release.json',{'feature_contract':integrity.FEATURE_CONTRACT,
        'release_id':'release','sha256':{'players':'hash'}})
    integrity.freeze_production('test')
    with pytest.raises(RuntimeError,match='frozen'):
        integrity.assert_publication_allowed()
    integrity.atomic_json(tmp_path/'active_release.json',{'feature_contract':integrity.FEATURE_CONTRACT,
        'release_id':'changed','sha256':{'players':'hash'}})
    with pytest.raises(RuntimeError,match='changed'):
        integrity.freeze_production('test')


def test_serialized_reference_has_importable_class(tmp_path):
    frame=pd.DataFrame({'position':['WR']*4,'receiving_yards':[20,40,30,50],
        'receiving_yards_avg_5':[30]*4,'targets_avg_5':[5]*4})
    model=ReferenceRecipe().fit(frame,'receiving_yards')
    assert ReferenceRecipe.__module__!='__main__'
    assert SmallGameModel.__module__!='__main__'
    path=tmp_path/'model.joblib'; integrity.atomic_joblib(path,model)
    np.testing.assert_allclose(joblib.load(path).predict(frame),model.predict(frame))


def test_uncertainty_counts_weeks_not_duplicate_offers():
    frame=pd.DataFrame({'season':[2026]*100,'week':[1]*100,'reference_error':[2]*100,'challenger_error':[1]*100})
    result=clustered_gain(frame)
    assert result['clusters']==1 and result['lower_95'] is None


@pytest.mark.parametrize('skip_predict',[False,True])
def test_daily_frozen_run_scores_without_retraining(tmp_path,monkeypatch,skip_predict):
    from nfl_pipeline import run_daily_and_notify as daily
    calls=[]
    monkeypatch.setattr(sys,'argv',['daily','--date','2099-09-20',*(['--skip-predict'] if skip_predict else [])])
    monkeypatch.setattr(daily,'production_freeze',lambda:{'release_id':'frozen'})
    monkeypatch.setattr(daily,'active_release',lambda:{'release_id':'frozen'})
    monkeypatch.setattr(daily,'_repo_root',lambda:tmp_path)
    monkeypatch.setenv('NFL_MODEL_RELEASE_ID','before')
    def run(step):
        calls.append(step.module)
        if step.module == 'nfl_pipeline.publish_forecasts':
            assert '--json' in step.args and 'matchups' in step.args
            return 0,json.dumps({'contract':'nfl-matchup-cards-v1','cards':[]}),''
        return 0,'ok',''
    async def post(*args): pass
    monkeypatch.setattr(daily,'_run',run)
    monkeypatch.setattr(daily,'_post_section',post)
    asyncio.run(daily.main())
    assert 'nfl_pipeline.run_training' not in calls
    if skip_predict:
        assert 'nfl_pipeline.modeling.score_accuracy_components' not in calls
        assert 'nfl_pipeline.modeling.predict_player_props' not in calls
        return
    assert 'nfl_pipeline.modeling.predict_player_props' in calls
    assert 'nfl_pipeline.modeling.score_accuracy_shadow' not in calls
    scorer='nfl_pipeline.modeling.score_accuracy_components'
    assert calls.count(scorer)==2
    prediction=calls.index('nfl_pipeline.modeling.predict_player_props')
    assert calls.index(scorer)<prediction<len(calls)-1
    lock=calls.index('nfl_pipeline.lock_ledger',prediction)
    capture=calls.index('nfl_pipeline.modeling.benchmark_offers',prediction)
    assert prediction < lock < capture < calls.index(scorer,prediction)
    assert calls.index('nfl_pipeline.refresh_lock_quotes') < prediction


def test_offline_training_does_not_publish_frozen_release(monkeypatch):
    from nfl_pipeline import run_training as training
    calls=[]
    monkeypatch.setattr(training,'production_freeze',lambda:{'release_id':'frozen'})
    def run(step):
        calls.append(step)
        return 0,'ok',''
    monkeypatch.setattr(training,'_run',run)
    result=training.run_training(season='2026',skip_context=True)
    assert result['status']=='ok'
    validated=next(s for s in calls if s.module.endswith('train_validated_release'))
    assert '--publish' not in validated.args
    assert any(s.module.endswith('train_accuracy_challengers') for s in calls)
    assert any(s.module.endswith('train_accuracy_components') for s in calls)
    modules=[s.module for s in calls]
    assert modules.index('nfl_pipeline.modeling.train_accuracy_challengers') < modules.index('nfl_pipeline.modeling.train_accuracy_components') < modules.index('nfl_pipeline.modeling.live_scoring_replay')
